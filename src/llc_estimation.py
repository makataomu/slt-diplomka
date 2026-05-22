"""LLC and drLLC estimation using devinterp 2.0.

Run after training checkpoints are saved.

Usage:
    python llc_estimation.py --ratio 0.5 --seed 0
    python llc_estimation.py --ratio 0.5 --seed 0 --calibrate
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    MOD_VALUE, MAX_NUMS,
    loss_fn, make_split_datasets, make_model,
)

# ─── devinterp version detection ─────────────────────────────────────────────

def _import_llc_estimator():
    """Returns (estimate_fn, api_version) where api_version is 1 or 2."""
    try:
        from devinterp.slt.llc import llc as _llc_v2
        return _llc_v2, 2
    except ImportError:
        pass
    try:
        from devinterp.slt.sampler import estimate_learning_coeff_with_summary, SGLD
        return (estimate_learning_coeff_with_summary, SGLD), 1
    except ImportError:
        raise ImportError(
            "devinterp not found. Install with: pip install devinterp"
        )


# ─── Dataset wrappers ─────────────────────────────────────────────────────────

def _make_hf_dataset(inputs: torch.Tensor, labels: torch.Tensor):
    """Wrap tensors as a minimal HuggingFace-compatible Dataset for devinterp v2."""
    from datasets import Dataset as HFDataset
    return HFDataset.from_dict({
        "inputs": inputs.cpu().numpy().tolist(),
        "labels": labels.cpu().numpy().tolist(),
    })


def _make_dataloader(inputs: torch.Tensor, labels: torch.Tensor, batch_size: int = 256):
    return DataLoader(
        TensorDataset(inputs, labels),
        batch_size=batch_size,
        shuffle=True,
    )


# ─── Loss wrappers for each API version ──────────────────────────────────────

def _loss_fn_v2(model: torch.nn.Module, batch: dict, device: str = "cpu") -> torch.Tensor:
    """loss_fn for devinterp v2: batch is a dict from HFDataset."""
    inputs = torch.tensor(batch["inputs"]).to(device)
    labels = torch.tensor(batch["labels"]).to(device)
    logits = model(inputs)
    return loss_fn(logits, labels)


def _evaluate_v1(model: torch.nn.Module, data):
    """evaluate callback for devinterp v1."""
    inputs, labels = data
    return loss_fn(model(inputs), labels), {}


# ─── Core estimation ──────────────────────────────────────────────────────────

def estimate_llc(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    epsilon: float,
    nbeta: float,
    gamma: float,
    num_chains: int,
    num_draws: int,
    num_burnin_steps: int,
    device: str = "cpu",
    batch_size: int = 256,
) -> dict:
    """
    Estimate LLC (or drLLC when inputs/labels are a filtered subset).
    Returns dict with keys: llc_mean, llc_std, loss_traces (np.ndarray, shape [num_chains, num_draws]).
    """
    estimator, api_version = _import_llc_estimator()
    model = model.to(device)
    model.eval()

    if api_version == 2:
        return _estimate_llc_v2(
            estimator, model, inputs, labels,
            epsilon=epsilon, nbeta=nbeta, gamma=gamma,
            num_chains=num_chains, num_draws=num_draws,
            device=device, batch_size=batch_size,
        )
    else:
        return _estimate_llc_v1(
            estimator, model, inputs, labels,
            epsilon=epsilon, nbeta=nbeta, gamma=gamma,
            num_chains=num_chains, num_draws=num_draws,
            num_burnin_steps=num_burnin_steps,
            device=device, batch_size=batch_size,
        )


def _estimate_llc_v2(llc_fn, model, inputs, labels, *,
                     epsilon, nbeta, gamma, num_chains, num_draws, device, batch_size):
    from functools import partial

    dataset = _make_hf_dataset(inputs, labels)
    _device = device

    custom_loss = partial(_loss_fn_v2, device=_device)

    # observables: empty dict — we only want LLC, which is always computed
    result = llc_fn(
        model,
        dataset=dataset,
        observables={},
        lr=epsilon,
        n_beta=nbeta,
        loss_fn=custom_loss,
        num_chains=num_chains,
        num_draws=num_draws,
    )

    llc_mean = float(result["llc_mean"])
    llc_std = float(result["llc_std"]) if "llc_std" in result else 0.0

    # Extract per-chain loss traces if available
    if "loss_trace" in result:
        traces = np.array(result["loss_trace"])
    else:
        traces = np.zeros((num_chains, num_draws))

    return {"llc_mean": llc_mean, "llc_std": llc_std, "loss_traces": traces}


def _estimate_llc_v1(estimators, model, inputs, labels, *,
                     epsilon, nbeta, gamma, num_chains, num_draws, num_burnin_steps,
                     device, batch_size):
    estimate_learning_coeff_with_summary, SGLD = estimators

    loader = _make_dataloader(inputs.to(device), labels.to(device), batch_size)

    result = estimate_learning_coeff_with_summary(
        model,
        loader=loader,
        evaluate=_evaluate_v1,
        sampling_method=SGLD,
        optimizer_kwargs=dict(lr=epsilon, nbeta=nbeta, localization=gamma),
        num_chains=num_chains,
        num_draws=num_draws,
        num_burnin_steps=num_burnin_steps,
        num_steps_bw_draws=1,
        device=device,
        online=True,
    )

    llc_mean = result.get("llc/mean", result.get("llc_mean", 0.0))
    llc_std = result.get("llc/std", result.get("llc_std", 0.0))

    if "loss_trace" in result:
        traces = np.array(result["loss_trace"])
    else:
        traces = np.zeros((num_chains, num_draws))

    return {"llc_mean": float(llc_mean), "llc_std": float(llc_std), "loss_traces": traces}


# ─── Process all checkpoints for one (ratio, seed) ────────────────────────────

def process_checkpoints(
    checkpoint_dir: Path,
    llc_csv_path: Path,
    test_x, test_y,
    test_add_x, test_add_y,
    test_mult_x, test_mult_y,
    llc_kwargs: dict,
    device: str = "cpu",
    skip_existing: bool = True,
):
    """Iterate all checkpoints, compute LLC + drLLC, save to CSV."""
    checkpoints = sorted(checkpoint_dir.glob("epoch_*.pt"),
                         key=lambda p: int(p.stem.split("_")[1]))

    if not checkpoints:
        print(f"No checkpoints in {checkpoint_dir}")
        return

    # Load already-computed epochs to allow resuming
    done_epochs = set()
    if skip_existing and llc_csv_path.exists():
        import pandas as pd
        done = pd.read_csv(llc_csv_path)
        done_epochs = set(done["epoch"].tolist())

    is_first_write = not llc_csv_path.exists()
    llc_csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["epoch", "LLC", "LLC_std", "drLLC_add", "drLLC_add_std",
                  "drLLC_mult", "drLLC_mult_std"]

    model = make_model(device=device)

    for ckpt_path in checkpoints:
        epoch = int(ckpt_path.stem.split("_")[1])
        if epoch in done_epochs:
            print(f"  Epoch {epoch}: already done, skipping")
            continue

        print(f"  Epoch {epoch}: loading checkpoint…")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        print(f"  Epoch {epoch}: estimating LLC…")
        global_result = estimate_llc(model, test_x, test_y, **llc_kwargs, device=device)

        print(f"  Epoch {epoch}: estimating drLLC_add…")
        add_result = estimate_llc(model, test_add_x, test_add_y, **llc_kwargs, device=device)

        print(f"  Epoch {epoch}: estimating drLLC_mult…")
        mult_result = estimate_llc(model, test_mult_x, test_mult_y, **llc_kwargs, device=device)

        row = {
            "epoch": epoch,
            "LLC": round(global_result["llc_mean"], 6),
            "LLC_std": round(global_result["llc_std"], 6),
            "drLLC_add": round(add_result["llc_mean"], 6),
            "drLLC_add_std": round(add_result["llc_std"], 6),
            "drLLC_mult": round(mult_result["llc_mean"], 6),
            "drLLC_mult_std": round(mult_result["llc_std"], 6),
        }

        mode = "w" if is_first_write else "a"
        with open(llc_csv_path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if is_first_write:
                writer.writeheader()
                is_first_write = False
            writer.writerow(row)

        print(f"  Epoch {epoch}: LLC={row['LLC']:.4f}  "
              f"drLLC_add={row['drLLC_add']:.4f}  drLLC_mult={row['drLLC_mult']:.4f}")


# ─── Calibration helper ───────────────────────────────────────────────────────

def run_calibration(model, test_x, test_y, device: str = "cpu"):
    """
    Interactive calibration: run LLC estimation with current hyperparams and
    print trace statistics so the user can judge chain mixing.
    Returns the loss_traces array (shape: [num_chains, num_draws]).
    """
    # Starting-point hyperparams per PROJECT_CONTEXT.md
    kwargs = dict(
        epsilon=1e-4, nbeta=1.0, gamma=10.0,
        num_chains=8, num_draws=500, num_burnin_steps=100, device=device,
    )
    print("Running calibration with starting hyperparams:")
    for k, v in kwargs.items():
        if k != "device":
            print(f"  {k} = {v}")

    result = estimate_llc(model, test_x, test_y, **kwargs)
    traces = result["loss_traces"]

    print(f"\nLLC mean = {result['llc_mean']:.4f}  std = {result['llc_std']:.4f}")
    if traces.size > 0:
        print(f"Trace stats per chain (should see mixing, not divergence or flatline):")
        for i, chain in enumerate(traces):
            print(f"  Chain {i}: mean={chain.mean():.4f}  std={chain.std():.4f}  "
                  f"min={chain.min():.4f}  max={chain.max():.4f}")

    return result


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ratio", type=float, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--calibrate", action="store_true",
                   help="Run calibration on the final checkpoint only")
    p.add_argument("--epsilon", type=float, default=None,
                   help="Override epsilon from calibration config")
    p.add_argument("--nbeta", type=float, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--num_chains", type=int, default=None)
    p.add_argument("--num_draws", type=int, default=None)
    p.add_argument("--num_burnin_steps", type=int, default=None)
    p.add_argument("--checkpoint_dir", type=str, default="results/checkpoints")
    p.add_argument("--metrics_dir", type=str, default="results/metrics")
    p.add_argument("--config", type=str, default="configs/llc_calibration.yaml")
    p.add_argument("--data_seed", type=int, default=598)
    return p.parse_args()


def load_llc_config(config_path: Path, args) -> dict:
    """Load calibrated hyperparams, with CLI overrides taking priority."""
    import yaml
    defaults = dict(
        epsilon=1e-4, nbeta=1.0, gamma=10.0,
        num_chains=8, num_draws=500, num_burnin_steps=100,
    )
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        if cfg.get("calibrated"):
            defaults.update({k: cfg[k] for k in defaults if k in cfg})
    # CLI overrides
    for k in defaults:
        v = getattr(args, k, None)
        if v is not None:
            defaults[k] = v
    return defaults


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  ratio={args.ratio}  seed={args.seed}")

    (_, _, test_x, test_y,
     test_add_x, test_add_y, test_mult_x, test_mult_y) = make_split_datasets(
        addition_frac=args.ratio, seed=args.data_seed, device=device,
    )
    print(f"Test: {len(test_x)}  Test-add: {len(test_add_x)}  Test-mult: {len(test_mult_x)}")

    ckpt_dir = (Path(args.checkpoint_dir)
                / f"ratio_{args.ratio:.2f}" / f"seed_{args.seed}")

    if args.calibrate:
        # Load the final checkpoint for calibration
        checkpoints = sorted(ckpt_dir.glob("epoch_*.pt"),
                             key=lambda p: int(p.stem.split("_")[1]))
        if not checkpoints:
            print(f"No checkpoints found in {ckpt_dir}")
            return
        latest = checkpoints[-1]
        print(f"Calibrating on {latest}")
        model = make_model(device=device)
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        result = run_calibration(model, test_x, test_y, device=device)
        # Save traces for fig_calibration_traces.pdf
        traces_path = Path(args.metrics_dir) / "calibration_traces.npy"
        traces_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(traces_path, result["loss_traces"])
        print(f"\nTraces saved to {traces_path}")
        print("Update configs/llc_calibration.yaml with the chosen hyperparams.")
        return

    llc_kwargs = load_llc_config(Path(args.config), args)
    print(f"LLC hyperparams: {llc_kwargs}")

    llc_csv = (Path(args.metrics_dir)
               / f"ratio_{args.ratio:.2f}_seed_{args.seed}_llc.csv")

    process_checkpoints(
        checkpoint_dir=ckpt_dir,
        llc_csv_path=llc_csv,
        test_x=test_x, test_y=test_y,
        test_add_x=test_add_x, test_add_y=test_add_y,
        test_mult_x=test_mult_x, test_mult_y=test_mult_y,
        llc_kwargs=llc_kwargs,
        device=device,
    )
    print(f"Done. LLC metrics saved to {llc_csv}")


if __name__ == "__main__":
    main()
