"""LLC and drLLC estimation using devinterp's SGLD optimizer directly.

devinterp 2.0.1 removed estimate_learning_coeff_with_summary and its
sample_single_chain hardcodes data["input_ids"] for language models.
We implement the LLC sampling loop ourselves using the SGLD optimizer,
which is stable and uses the same parameters Sullivan used.

Formula: LLC = nbeta * (mean(draw_losses) - init_loss)

Usage:
    python llc_estimation.py --ratio 0.5 --seed 0
    python llc_estimation.py --ratio 0.5 --seed 0 --calibrate
"""
import argparse
import csv
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import loss_fn, make_split_datasets, make_model

# Suppress devinterp warnings: SGLD is deprecated in name but functional,
# and the nbeta=1 warning is expected during calibration.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="devinterp")
warnings.filterwarnings("ignore", message=".*nbeta set to 1.*")

from devinterp.optim import SGLD
from devinterp.utils import default_nbeta


# ─── core LLC estimation ──────────────────────────────────────────────────────

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
    Run SGLD sampling and estimate LLC.
    Returns dict: llc_mean, llc_std, loss_traces (num_chains × num_draws).

    LLC = nbeta * (mean_sampling_loss - init_loss)
    Runs num_chains independent SGLD chains, each for (num_burnin_steps + num_draws) steps.
    """
    inputs = inputs.to(device)
    labels = labels.to(device)

    # Initial loss at the converged weights
    model.eval()
    with torch.no_grad():
        init_loss = loss_fn(model(inputs), labels).item()

    total_steps = num_burnin_steps + num_draws
    all_traces = []

    for chain_idx in range(num_chains):
        chain_model = deepcopy(model)  # deepcopy preserves device
        optimizer = SGLD(
            chain_model.parameters(),
            lr=epsilon,
            nbeta=nbeta,
            localization=gamma,
        )

        loader = DataLoader(
            TensorDataset(inputs, labels),
            batch_size=batch_size,
            shuffle=True,
        )
        batch_iter = _cycle(loader)

        draw_losses = []
        for step in range(total_steps):
            batch_x, batch_y = next(batch_iter)

            chain_model.train()
            optimizer.zero_grad()
            l = loss_fn(chain_model(batch_x), batch_y)
            l.backward()
            optimizer.step()

            if step >= num_burnin_steps:
                chain_model.eval()
                with torch.no_grad():
                    draw_loss = loss_fn(chain_model(inputs), labels).item()
                draw_losses.append(draw_loss)

        all_traces.append(draw_losses)

    traces = np.array(all_traces)               # (num_chains, num_draws)
    mean_sampling_loss = traces.mean()
    chain_means = traces.mean(axis=1)           # (num_chains,)

    llc_mean = nbeta * (mean_sampling_loss - init_loss)
    llc_std  = nbeta * chain_means.std()

    return {"llc_mean": float(llc_mean), "llc_std": float(llc_std), "loss_traces": traces}


def _cycle(loader):
    while True:
        yield from loader


# ─── process all checkpoints for one (ratio, seed) ───────────────────────────

def process_checkpoints(
    checkpoint_dir: Path,
    llc_csv_path: Path,
    test_x, test_y,
    test_add_x, test_add_y,
    test_mult_x, test_mult_y,
    llc_kwargs: dict,
    device: str = "cpu",
):
    checkpoints = sorted(
        checkpoint_dir.glob("epoch_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if not checkpoints:
        print(f"No checkpoints in {checkpoint_dir}")
        return

    # Resume: skip already-computed epochs
    done_epochs = set()
    if llc_csv_path.exists():
        import pandas as pd
        done_epochs = set(pd.read_csv(llc_csv_path)["epoch"].tolist())

    is_first_write = not llc_csv_path.exists()
    llc_csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["epoch", "LLC", "LLC_std",
                  "drLLC_add", "drLLC_add_std",
                  "drLLC_mult", "drLLC_mult_std"]

    model = make_model(device=device)

    for ckpt_path in checkpoints:
        epoch = int(ckpt_path.stem.split("_")[1])
        if epoch in done_epochs:
            print(f"  Epoch {epoch}: already done, skipping")
            continue

        print(f"  Epoch {epoch}: loading…")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        print(f"  Epoch {epoch}: LLC…")
        r_global = estimate_llc(model, test_x, test_y, **llc_kwargs, device=device)
        print(f"  Epoch {epoch}: drLLC_add…")
        r_add    = estimate_llc(model, test_add_x, test_add_y, **llc_kwargs, device=device)
        print(f"  Epoch {epoch}: drLLC_mult…")
        r_mult   = estimate_llc(model, test_mult_x, test_mult_y, **llc_kwargs, device=device)

        row = {
            "epoch":          epoch,
            "LLC":            round(r_global["llc_mean"], 6),
            "LLC_std":        round(r_global["llc_std"],  6),
            "drLLC_add":      round(r_add["llc_mean"],    6),
            "drLLC_add_std":  round(r_add["llc_std"],     6),
            "drLLC_mult":     round(r_mult["llc_mean"],   6),
            "drLLC_mult_std": round(r_mult["llc_std"],    6),
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


# ─── calibration ──────────────────────────────────────────────────────────────

def run_calibration(model, test_x, test_y, device="cpu", batch_size=256):
    # default_nbeta = batch_size / log(batch_size), per devinterp recommendation
    nbeta_default = default_nbeta(batch_size)
    kwargs = dict(epsilon=1e-4, nbeta=nbeta_default, gamma=10.0,
                  num_chains=8, num_draws=500, num_burnin_steps=500,
                  batch_size=batch_size)
    print("Running calibration with starting hyperparams:")
    for k, v in kwargs.items():
        print(f"  {k} = {v}")

    result = estimate_llc(model, test_x, test_y, **kwargs, device=device)

    print(f"\nLLC mean = {result['llc_mean']:.4f}  std = {result['llc_std']:.4f}")
    traces = result["loss_traces"]
    for i, chain in enumerate(traces):
        print(f"  Chain {i}: mean={chain.mean():.4f}  std={chain.std():.4f}  "
              f"min={chain.min():.4f}  max={chain.max():.4f}")
    return result


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ratio",  type=float, required=True)
    p.add_argument("--seed",   type=int,   default=0)
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--checkpoint_epoch", type=int, default=None,
                   help="Calibrate on this specific epoch (default: latest checkpoint)")
    p.add_argument("--epsilon",          type=float, default=None)
    p.add_argument("--nbeta",            type=float, default=None)
    p.add_argument("--gamma",            type=float, default=None)
    p.add_argument("--num_chains",       type=int,   default=None)
    p.add_argument("--num_draws",        type=int,   default=None)
    p.add_argument("--num_burnin_steps", type=int,   default=None)
    p.add_argument("--checkpoint_dir",   default="results/checkpoints")
    p.add_argument("--metrics_dir",      default="results/metrics")
    p.add_argument("--config",           default="configs/llc_calibration.yaml")
    p.add_argument("--data_seed",        type=int,   default=598)
    return p.parse_args()


def load_llc_config(config_path: Path, args) -> dict:
    import yaml
    defaults = dict(epsilon=1e-4, nbeta=1.0, gamma=10.0,
                    num_chains=8, num_draws=500, num_burnin_steps=100)
    if config_path.exists():
        cfg = yaml.safe_load(config_path.read_text())
        if cfg.get("calibrated"):
            defaults.update({k: cfg[k] for k in defaults if k in cfg})
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
     test_add_x, test_add_y,
     test_mult_x, test_mult_y) = make_split_datasets(
        addition_frac=args.ratio, seed=args.data_seed, device=device,
    )
    print(f"Test: {len(test_x)}  Test-add: {len(test_add_x)}  Test-mult: {len(test_mult_x)}")

    ckpt_dir = (Path(args.checkpoint_dir)
                / f"ratio_{args.ratio:.2f}" / f"seed_{args.seed}")

    if args.calibrate:
        checkpoints = sorted(ckpt_dir.glob("epoch_*.pt"),
                             key=lambda p: int(p.stem.split("_")[1]))
        if not checkpoints:
            print(f"No checkpoints found in {ckpt_dir}")
            return
        if args.checkpoint_epoch is not None:
            available = {int(p.stem.split("_")[1]): p for p in checkpoints}
            # Pick the closest epoch to the requested one
            closest = min(available.keys(), key=lambda e: abs(e - args.checkpoint_epoch))
            latest = available[closest]
            if closest != args.checkpoint_epoch:
                print(f"Epoch {args.checkpoint_epoch} not found; using closest: {closest}")
        else:
            latest = checkpoints[-1]
        print(f"Calibrating on {latest}")
        model = make_model(device=device)
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])

        result = run_calibration(model, test_x, test_y, device=device)

        epoch_tag = int(latest.stem.split("_")[1])
        traces_path = Path(args.metrics_dir) / f"calibration_traces_epoch{epoch_tag}.npy"
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
