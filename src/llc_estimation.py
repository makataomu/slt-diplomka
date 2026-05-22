"""LLC and drLLC estimation using devinterp's v1 sampler API.

Uses estimate_learning_coeff_with_summary from devinterp.slt.sampler,
which works with any torch.nn.Module and custom loss functions.

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
from utils import loss_fn, make_split_datasets, make_model

# ─── devinterp v1 imports ─────────────────────────────────────────────────────

from devinterp.slt.sampler import estimate_learning_coeff_with_summary, SGLD


# ─── evaluate callback (v1 API) ───────────────────────────────────────────────

def _evaluate(model, data):
    inputs, labels = data
    return loss_fn(model(inputs), labels), {}


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
    Estimate LLC (or drLLC when inputs/labels are a filtered subset).
    Returns dict: llc_mean (float), llc_std (float), loss_traces (np.ndarray).
    """
    model = model.to(device)
    model.eval()

    loader = DataLoader(
        TensorDataset(inputs.to(device), labels.to(device)),
        batch_size=batch_size,
        shuffle=True,
    )

    result = estimate_learning_coeff_with_summary(
        model,
        loader=loader,
        evaluate=_evaluate,
        sampling_method=SGLD,
        optimizer_kwargs=dict(lr=epsilon, nbeta=nbeta, localization=gamma),
        num_chains=num_chains,
        num_draws=num_draws,
        num_burnin_steps=num_burnin_steps,
        num_steps_bw_draws=1,
        device=device,
        online=True,
    )

    # Handle both scalar return (old devinterp) and dict return (newer devinterp)
    if hasattr(result, "keys"):
        llc_mean = float(result.get("llc/mean", result.get("llc_mean", 0.0)))
        llc_std  = float(result.get("llc/std",  result.get("llc_std",  0.0)))
        traces   = np.array(result["loss_trace"]) if "loss_trace" in result else np.zeros((num_chains, num_draws))
    else:
        llc_mean = float(result)
        llc_std  = 0.0
        traces   = np.zeros((num_chains, num_draws))

    return {"llc_mean": llc_mean, "llc_std": llc_std, "loss_traces": traces}


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
        r_add = estimate_llc(model, test_add_x, test_add_y, **llc_kwargs, device=device)

        print(f"  Epoch {epoch}: drLLC_mult…")
        r_mult = estimate_llc(model, test_mult_x, test_mult_y, **llc_kwargs, device=device)

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

def run_calibration(model, test_x, test_y, device="cpu"):
    kwargs = dict(epsilon=1e-4, nbeta=1.0, gamma=10.0,
                  num_chains=8, num_draws=500, num_burnin_steps=100)

    print("Running calibration with starting hyperparams:")
    for k, v in kwargs.items():
        print(f"  {k} = {v}")

    result = estimate_llc(model, test_x, test_y, **kwargs, device=device)

    print(f"\nLLC mean = {result['llc_mean']:.4f}  std = {result['llc_std']:.4f}")
    traces = result["loss_traces"]
    if traces.size > 0 and traces.any():
        for i, chain in enumerate(traces):
            print(f"  Chain {i}: mean={chain.mean():.4f}  std={chain.std():.4f}  "
                  f"min={chain.min():.4f}  max={chain.max():.4f}")

    return result


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ratio", type=float, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--epsilon", type=float, default=None)
    p.add_argument("--nbeta",   type=float, default=None)
    p.add_argument("--gamma",   type=float, default=None)
    p.add_argument("--num_chains",       type=int, default=None)
    p.add_argument("--num_draws",        type=int, default=None)
    p.add_argument("--num_burnin_steps", type=int, default=None)
    p.add_argument("--checkpoint_dir", default="results/checkpoints")
    p.add_argument("--metrics_dir",    default="results/metrics")
    p.add_argument("--config",         default="configs/llc_calibration.yaml")
    p.add_argument("--data_seed", type=int, default=598)
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
        latest = checkpoints[-1]
        print(f"Calibrating on {latest}")
        model = make_model(device=device)
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])

        result = run_calibration(model, test_x, test_y, device=device)

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
