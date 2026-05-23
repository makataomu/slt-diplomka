"""All figure generation. Run after all metrics CSVs are in results/metrics/.

Usage:
    python plotting.py --metrics_dir results/metrics --figures_dir results/figures
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

matplotlib.rcParams["font.size"] = 11
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["legend.frameon"] = False
matplotlib.rcParams["figure.dpi"] = 150


RATIOS = [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]


def load_all_metrics(metrics_dir: Path):
    """Returns dict: (ratio, seed) -> DataFrame, and merged dict: ratio -> DataFrame (mean over seeds)."""
    pattern = re.compile(r"ratio_(\d+\.\d+)_seed_(\d+)\.csv")
    data = {}
    for f in sorted(metrics_dir.glob("ratio_*.csv")):
        if "_llc" in f.name:
            continue
        m = pattern.match(f.name)
        if not m:
            continue
        ratio, seed = float(m.group(1)), int(m.group(2))
        df = pd.read_csv(f)
        data[(ratio, seed)] = df

    # Merge LLC files
    llc_pattern = re.compile(r"ratio_(\d+\.\d+)_seed_(\d+)_llc\.csv")
    for f in sorted(metrics_dir.glob("ratio_*_llc.csv")):
        m = llc_pattern.match(f.name)
        if not m:
            continue
        ratio, seed = float(m.group(1)), int(m.group(2))
        llc_df = pd.read_csv(f)
        key = (ratio, seed)
        if key in data:
            data[key] = data[key].merge(llc_df, on="epoch", how="left")
        else:
            data[key] = llc_df

    return data


def average_over_seeds(data: dict, ratio: float):
    """Return mean and std DataFrame over all seeds for a given ratio."""
    dfs = [v for (r, s), v in data.items() if r == ratio]
    if not dfs:
        return None, None
    combined = pd.concat(dfs)
    mean = combined.groupby("epoch").mean().reset_index()
    std = combined.groupby("epoch").std().reset_index()
    return mean, std


def ratio_colors(ratios=RATIOS):
    cmap = matplotlib.colormaps.get_cmap("viridis").resampled(len(ratios))
    return {r: cmap(i) for i, r in enumerate(ratios)}


# ─── Figure 1: Loss per ratio ────────────────────────────────────────────────

def fig_loss_per_ratio(data, out_dir):
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharey=False)
    axes = axes.flatten()

    for i, ratio in enumerate(RATIOS):
        ax = axes[i]
        mean, std = average_over_seeds(data, ratio)
        if mean is None:
            ax.set_visible(False)
            continue
        ep = mean["epoch"]
        ax.plot(ep, mean["test_loss"], label="test (all)", color="black", lw=1.2)
        ax.plot(ep, mean["test_loss_add"], label="test (add)", color="tab:blue", lw=1)
        ax.plot(ep, mean["test_loss_mult"], label="test (mult)", color="tab:orange", lw=1)
        ax.set_title(f"add_frac={ratio:.2f}", fontsize=9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_yscale("log")

    # Legend cell
    ax_legend = axes[7]
    ax_legend.axis("off")
    handles = [
        matplotlib.lines.Line2D([0], [0], color="black", lw=1.2, label="test (all)"),
        matplotlib.lines.Line2D([0], [0], color="tab:blue", lw=1, label="test (add)"),
        matplotlib.lines.Line2D([0], [0], color="tab:orange", lw=1, label="test (mult)"),
    ]
    ax_legend.legend(handles=handles, loc="center", fontsize=10)

    fig.suptitle("Test Loss per Addition Ratio", fontsize=12, y=1.01)
    fig.tight_layout()
    path = out_dir / "fig_loss_per_ratio.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}  ({path.stat().st_size} bytes)")


# ─── Figure 2: LLC per ratio (averaged over seeds) ───────────────────────────

def fig_llc_per_ratio(data, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ratio_colors()

    for ratio in RATIOS:
        mean, std = average_over_seeds(data, ratio)
        if mean is None or "LLC" not in mean.columns:
            continue
        ep = mean["epoch"]
        ax.plot(ep, mean["LLC"], label=f"{ratio:.2f}", color=colors[ratio], lw=1.2)
        if std is not None and "LLC" in std.columns:
            ax.fill_between(ep, mean["LLC"] - std["LLC"], mean["LLC"] + std["LLC"],
                            color=colors[ratio], alpha=0.15)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("LLC")
    ax.set_title("Global LLC vs Epoch (mean ± SD over seeds)")
    ax.legend(title="add_frac", loc="upper right", fontsize=9)
    fig.tight_layout()
    path = out_dir / "fig_llc_per_ratio.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}  ({path.stat().st_size} bytes)")


# ─── Figure 3: drLLC per ratio (2×4 grid) ────────────────────────────────────

def fig_drLLC_per_ratio(data, out_dir):
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharey=False)
    axes = axes.flatten()

    for i, ratio in enumerate(RATIOS):
        ax = axes[i]
        mean, std = average_over_seeds(data, ratio)
        if mean is None or "LLC" not in mean.columns:
            ax.set_visible(False)
            continue
        ep = mean["epoch"]
        ax.plot(ep, mean["LLC"], label="LLC", color="black", lw=1.2)
        if "drLLC_add" in mean.columns:
            ax.plot(ep, mean["drLLC_add"], label="drLLC (add)", color="tab:blue", lw=1)
        if "drLLC_mult" in mean.columns:
            ax.plot(ep, mean["drLLC_mult"], label="drLLC (mult)", color="tab:orange", lw=1)
        ax.set_title(f"add_frac={ratio:.2f}", fontsize=9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("LLC")

    ax_legend = axes[7]
    ax_legend.axis("off")
    handles = [
        matplotlib.lines.Line2D([0], [0], color="black", lw=1.2, label="LLC"),
        matplotlib.lines.Line2D([0], [0], color="tab:blue", lw=1, label="drLLC (add)"),
        matplotlib.lines.Line2D([0], [0], color="tab:orange", lw=1, label="drLLC (mult)"),
    ]
    ax_legend.legend(handles=handles, loc="center", fontsize=10)

    fig.suptitle("LLC, drLLC-add, drLLC-mult per Ratio", fontsize=12, y=1.01)
    fig.tight_layout()
    path = out_dir / "fig_drLLC_per_ratio.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}  ({path.stat().st_size} bytes)")


# ─── Figures 4 & 5: Single-ratio close-ups ───────────────────────────────────

def fig_single_ratio_drLLC(data, ratio: float, filename: str, out_dir: Path):
    mean, std = average_over_seeds(data, ratio)
    if mean is None or "LLC" not in mean.columns:
        print(f"No LLC data for ratio {ratio}, skipping {filename}")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ep = mean["epoch"]
    ax.plot(ep, mean["LLC"], label="LLC", color="black", lw=1.5)
    if "drLLC_add" in mean.columns:
        ax.plot(ep, mean["drLLC_add"], label="drLLC (add)", color="tab:blue", lw=1.2)
        if std is not None and "drLLC_add" in std.columns:
            ax.fill_between(ep, mean["drLLC_add"] - std["drLLC_add"],
                            mean["drLLC_add"] + std["drLLC_add"],
                            color="tab:blue", alpha=0.15)
    if "drLLC_mult" in mean.columns:
        ax.plot(ep, mean["drLLC_mult"], label="drLLC (mult)", color="tab:orange", lw=1.2)
        if std is not None and "drLLC_mult" in std.columns:
            ax.fill_between(ep, mean["drLLC_mult"] - std["drLLC_mult"],
                            mean["drLLC_mult"] + std["drLLC_mult"],
                            color="tab:orange", alpha=0.15)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("LLC")
    ax.set_title(f"LLC decomposition — add_frac={ratio:.2f} (mean ± SD over seeds)")
    ax.legend()
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}  ({path.stat().st_size} bytes)")


# ─── Figure 6: Grokking alignment ────────────────────────────────────────────

def find_grokking_epoch(mean_df, col, threshold=0.95):
    """Return first epoch where col >= threshold, or None."""
    hits = mean_df[mean_df[col] >= threshold]
    return int(hits["epoch"].iloc[0]) if len(hits) else None


def find_llc_min_epoch(mean_df, col):
    if col not in mean_df.columns or mean_df[col].isna().all():
        return None
    idx = mean_df[col].idxmin()
    return int(mean_df.loc[idx, "epoch"])


def fig_grokking_alignment(data, out_dir):
    records = []
    for ratio in RATIOS:
        mean, _ = average_over_seeds(data, ratio)
        if mean is None:
            continue
        rec = {"ratio": ratio}
        rec["grok_add"] = find_grokking_epoch(mean, "test_acc_add")
        rec["grok_mult"] = find_grokking_epoch(mean, "test_acc_mult")
        rec["llc_min_add"] = find_llc_min_epoch(mean, "drLLC_add")
        rec["llc_min_mult"] = find_llc_min_epoch(mean, "drLLC_mult")
        records.append(rec)

    df = pd.DataFrame(records)
    if df.empty:
        print("No data for grokking alignment figure.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = ratio_colors()
    color_list = [colors[r] for r in df["ratio"]]

    for ax, task, grok_col, llc_col, title in [
        (axes[0], "Addition", "grok_add", "llc_min_add", "Addition grokking vs drLLC_add minimum"),
        (axes[1], "Multiplication", "grok_mult", "llc_min_mult", "Multiplication grokking vs drLLC_mult minimum"),
    ]:
        valid = df.dropna(subset=[grok_col, llc_col])
        sc = ax.scatter(valid[grok_col], valid[llc_col],
                        c=[colors[r] for r in valid["ratio"]], s=60, zorder=3)
        for _, row in valid.iterrows():
            ax.annotate(f"{row['ratio']:.2f}", (row[grok_col], row[llc_col]),
                        textcoords="offset points", xytext=(4, 4), fontsize=8)
        lim_max = max(valid[[grok_col, llc_col]].max().max() * 1.05, 100)
        ax.plot([0, lim_max], [0, lim_max], "k--", lw=0.8, alpha=0.4, label="perfect alignment")
        ax.set_xlabel(f"Epoch of {task} grokking (acc > 0.95)")
        ax.set_ylabel(f"Epoch of drLLC_{task[:3].lower()} minimum")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=8)

    fig.suptitle("Do drLLC minima align with grokking events?", fontsize=11)
    fig.tight_layout()
    path = out_dir / "fig_grokking_alignment.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}  ({path.stat().st_size} bytes)")


# ─── Figure 7: Calibration traces ────────────────────────────────────────────

def fig_calibration_traces(traces: np.ndarray, out_dir: Path):
    """traces: shape (num_chains, num_draws)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for chain_idx, chain in enumerate(traces):
        ax.plot(chain, lw=0.8, alpha=0.7, label=f"Chain {chain_idx}")
    ax.set_xlabel("Draw")
    ax.set_ylabel("Loss (SGLD)")
    ax.set_title("SGLD chain traces — calibration run")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "fig_calibration_traces.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}  ({path.stat().st_size} bytes)")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics_dir", default="results/metrics")
    p.add_argument("--figures_dir", default="results/figures")
    args = p.parse_args()

    metrics_dir = Path(args.metrics_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    data = load_all_metrics(metrics_dir)
    if not data:
        print(f"No metric CSVs found in {metrics_dir}. Run training first.")
        return

    print(f"Loaded {len(data)} (ratio, seed) runs.")

    fig_loss_per_ratio(data, figures_dir)
    fig_llc_per_ratio(data, figures_dir)
    fig_drLLC_per_ratio(data, figures_dir)
    fig_single_ratio_drLLC(data, 0.50, "fig_drLLC_50_50.pdf", figures_dir)
    fig_single_ratio_drLLC(data, 0.70, "fig_drLLC_70_30.pdf", figures_dir)
    fig_grokking_alignment(data, figures_dir)
    print("All figures done.")


if __name__ == "__main__":
    main()
