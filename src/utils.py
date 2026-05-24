"""Shared data utilities, loss function, and model factory."""
import random
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


MOD_VALUE = 113
MAX_NUMS = 130  # vocab size for a, b tokens; op token is 0 or 1, so d_vocab = MAX_NUMS + 1


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loss_fn(logits, labels):
    """Cross-entropy loss, handles (batch, seq, vocab) by taking last position."""
    if logits.dim() == 3:
        logits = logits[:, -1]
    logits = logits.to(torch.float64)
    log_probs = logits.log_softmax(dim=-1)
    correct_log_probs = log_probs.gather(dim=-1, index=labels[:, None])[:, 0]
    return -correct_log_probs.mean()


def accuracy_fn(logits, labels):
    if logits.dim() == 3:
        logits = logits[:, -1]
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def create_full_dataset(mod_value: int = MOD_VALUE, max_nums: int = MAX_NUMS):
    """Build all (a, op, b) -> label pairs for both operations."""
    rows_add, rows_mult = [], []
    for a in range(max_nums):
        for b in range(max_nums):
            rows_add.append(([a, 1, b], (a + b) % mod_value))
            rows_mult.append(([a, 0, b], (a * b) % mod_value))

    inputs_add = torch.tensor([r[0] for r in rows_add])
    labels_add = torch.tensor([r[1] for r in rows_add])
    inputs_mult = torch.tensor([r[0] for r in rows_mult])
    labels_mult = torch.tensor([r[1] for r in rows_mult])
    return inputs_add, labels_add, inputs_mult, labels_mult


def make_split_datasets(addition_frac: float, seed: int = 598,
                         train_frac: float = 0.5,
                         mod_value: int = MOD_VALUE, max_nums: int = MAX_NUMS,
                         device: str = "cpu"):
    """
    Returns train_data, train_labels, test_data, test_labels,
            test_add_data, test_add_labels, test_mult_data, test_mult_labels.

    Matches Sullivan's pooled-then-split protocol exactly:
      total_train_size = int((|add| + |mult|) * train_frac)   # = 16 900
      add_train_size   = int(total_train_size * addition_frac)
      mult_train_size  = total_train_size - add_train_size

    So at ratio=0.5: 8 450 add + 8 450 mult = 16 900 train; same again for test.
    Earlier (buggy) version pre-split each operation 50/50, which halved the
    training set and prevented grokking at middle ratios.
    """
    inputs_add, labels_add, inputs_mult, labels_mult = create_full_dataset(mod_value, max_nums)

    rng = torch.Generator()
    rng.manual_seed(seed)

    perm_add  = torch.randperm(len(inputs_add),  generator=rng)
    perm_mult = torch.randperm(len(inputs_mult), generator=rng)

    total_train_size = int((len(inputs_add) + len(inputs_mult)) * train_frac)
    add_train_size   = int(total_train_size * addition_frac)
    mult_train_size  = total_train_size - add_train_size

    # Slice each shuffled pool: first N → train, rest → test
    train_add_x = inputs_add[perm_add[:add_train_size]]
    train_add_y = labels_add[perm_add[:add_train_size]]
    test_add_x  = inputs_add[perm_add[add_train_size:]]
    test_add_y  = labels_add[perm_add[add_train_size:]]

    train_mult_x = inputs_mult[perm_mult[:mult_train_size]]
    train_mult_y = labels_mult[perm_mult[:mult_train_size]]
    test_mult_x  = inputs_mult[perm_mult[mult_train_size:]]
    test_mult_y  = labels_mult[perm_mult[mult_train_size:]]

    # Combined train: shuffle the mix
    train_x = torch.cat([train_add_x, train_mult_x])
    train_y = torch.cat([train_add_y, train_mult_y])
    perm_tr = torch.randperm(len(train_x), generator=rng)
    train_x = train_x[perm_tr]
    train_y = train_y[perm_tr]

    # Combined test: shuffle the mix
    test_x = torch.cat([test_add_x, test_mult_x])
    test_y = torch.cat([test_add_y, test_mult_y])
    perm_te = torch.randperm(len(test_x), generator=rng)
    test_x = test_x[perm_te]
    test_y = test_y[perm_te]

    to = lambda t: t.to(device)
    return (to(train_x), to(train_y), to(test_x), to(test_y),
            to(test_add_x), to(test_add_y), to(test_mult_x), to(test_mult_y))


def make_model(device: str = "cpu", seed: int = 999,
               max_nums: int = MAX_NUMS, mod_value: int = MOD_VALUE, n_ctx: int = 3):
    """Instantiate the HookedTransformer matching Sullivan's exact config."""
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    cfg = HookedTransformerConfig(
        n_layers=1,
        n_heads=4,
        d_model=128,
        d_head=32,
        d_mlp=512,
        act_fn="relu",
        normalization_type="LN",
        d_vocab=max_nums + 1,
        d_vocab_out=mod_value,
        n_ctx=n_ctx,
        init_weights=True,
        device=device,
        seed=seed,
    )
    model = HookedTransformer(cfg)
    # Disable biases (Sullivan's setup)
    for name, param in model.named_parameters():
        if "b_" in name:
            param.requires_grad = False
    return model
