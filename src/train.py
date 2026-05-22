"""Training script for modular arithmetic grokking experiment.

Usage:
    python train.py --ratio 0.5 --seed 0
    python train.py --ratio 0.5 --seed 0 --resume
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    MOD_VALUE, MAX_NUMS,
    seed_everything, loss_fn, accuracy_fn,
    make_split_datasets, make_model,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ratio", type=float, required=True,
                   help="Fraction of training examples that are addition (0.0–1.0)")
    p.add_argument("--seed", type=int, default=0, help="Random seed (0, 1, 2 …)")
    p.add_argument("--epochs", type=int, default=6000)
    p.add_argument("--checkpoint_every", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1.0)
    p.add_argument("--resume", action="store_true",
                   help="Resume from latest checkpoint if one exists")
    p.add_argument("--checkpoint_dir", type=str, default="results/checkpoints")
    p.add_argument("--metrics_dir", type=str, default="results/metrics")
    p.add_argument("--data_seed", type=int, default=598)
    return p.parse_args()


def checkpoint_dir_for(args):
    return Path(args.checkpoint_dir) / f"ratio_{args.ratio:.2f}" / f"seed_{args.seed}"


def metrics_path_for(args):
    return Path(args.metrics_dir) / f"ratio_{args.ratio:.2f}_seed_{args.seed}.csv"


def find_latest_checkpoint(ckpt_dir: Path):
    checkpoints = sorted(ckpt_dir.glob("epoch_*.pt"),
                         key=lambda p: int(p.stem.split("_")[1]))
    return checkpoints[-1] if checkpoints else None


def save_checkpoint(ckpt_dir: Path, epoch: int, model, optimizer, row_history: list):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"epoch_{epoch:05d}.pt"
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "row_history": row_history,
    }, path)


def append_metrics_row(csv_path: Path, row: dict, write_header: bool = False):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if write_header else "a"
    with open(csv_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  ratio={args.ratio}  seed={args.seed}")

    seed_everything(args.seed)

    (train_x, train_y, test_x, test_y,
     test_add_x, test_add_y, test_mult_x, test_mult_y) = make_split_datasets(
        addition_frac=args.ratio,
        seed=args.data_seed,
        device=device,
    )
    print(f"Train: {len(train_x)}  Test: {len(test_x)}  "
          f"Test-add: {len(test_add_x)}  Test-mult: {len(test_mult_x)}")

    model = make_model(device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.90, 0.98)
    )

    ckpt_dir = checkpoint_dir_for(args)
    csv_path = metrics_path_for(args)
    start_epoch = 0
    row_history = []

    if args.resume:
        latest = find_latest_checkpoint(ckpt_dir)
        if latest:
            print(f"Resuming from {latest}")
            ckpt = torch.load(latest, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            torch.set_rng_state(ckpt["rng_state"].cpu())
            if ckpt["cuda_rng_state"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state(ckpt["cuda_rng_state"])
            start_epoch = ckpt["epoch"] + 1
            row_history = ckpt["row_history"]
            print(f"Resumed at epoch {start_epoch}")
        else:
            print("No checkpoint found, starting fresh.")

    # Write CSV header only on fresh start
    if start_epoch == 0:
        header_row = {
            "epoch": 0, "train_loss": 0, "test_loss": 0,
            "test_loss_add": 0, "test_loss_mult": 0,
            "train_acc": 0, "test_acc": 0,
            "test_acc_add": 0, "test_acc_mult": 0,
        }
        append_metrics_row(csv_path, header_row, write_header=True)

    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        logits = model(train_x)
        loss = loss_fn(logits, train_y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (epoch + 1) % args.checkpoint_every == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                test_logits = model(test_x)
                test_loss = loss_fn(test_logits, test_y).item()
                test_acc = accuracy_fn(test_logits, test_y)

                add_logits = model(test_add_x)
                test_loss_add = loss_fn(add_logits, test_add_y).item()
                test_acc_add = accuracy_fn(add_logits, test_add_y)

                mult_logits = model(test_mult_x)
                test_loss_mult = loss_fn(mult_logits, test_mult_y).item()
                test_acc_mult = accuracy_fn(mult_logits, test_mult_y)

                train_acc = accuracy_fn(model(train_x), train_y)

            row = {
                "epoch": epoch + 1,
                "train_loss": round(loss.item(), 6),
                "test_loss": round(test_loss, 6),
                "test_loss_add": round(test_loss_add, 6),
                "test_loss_mult": round(test_loss_mult, 6),
                "train_acc": round(train_acc, 4),
                "test_acc": round(test_acc, 4),
                "test_acc_add": round(test_acc_add, 4),
                "test_acc_mult": round(test_acc_mult, 4),
            }
            row_history.append(row)
            append_metrics_row(csv_path, row)
            save_checkpoint(ckpt_dir, epoch + 1, model, optimizer, row_history)

            elapsed = time.time() - t0
            print(f"[{epoch+1:5d}/{args.epochs}] "
                  f"train_loss={loss.item():.4f}  test_loss={test_loss:.4f}  "
                  f"test_acc={test_acc:.3f}  ({elapsed:.0f}s)")

    print(f"Done. Metrics saved to {csv_path}")


if __name__ == "__main__":
    main()
