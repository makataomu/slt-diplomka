# STATUS.md — SLT Grokking Experiment

---

## 2026-05-23 00:30
**First-session checklist complete.**

### What I found in Sullivan's repo (`external/ArithmeticTransformer/`)

Three notebooks, no standalone scripts:
- `src/GrokkingAdditionMultiplication.ipynb` — training notebook (transformer_lens HookedTransformer, AdamW, full-batch, 6000 epochs)
- `src/EstimateLLC.ipynb` — LLC estimation notebook (devinterp v1.2.0 API: `estimate_learning_coeff_with_summary`)
- `src/Graphs.ipynb` — plotting
- `src/helpers.py` — loss_fn, get_dataloader, rolling_average

### What needs adapting from Sullivan's code

1. **devinterp API changed** — Sullivan used `devinterp==1.2.0` with `estimate_learning_coeff_with_summary(model, loader, evaluate, ...)`. Current version is **2.0.1** with a new `llc(model, dataset, observables, lr, n_beta, ...)` API that returns `xr.Dataset`. I've written compatibility code that tries v2 API first and falls back to v1.
2. **No drLLC** — Sullivan's code has no data-restricted LLC. Added in `src/llc_estimation.py`.
3. **No checkpointing/resumability** — Sullivan's training runs in one go. Rewrote as a `--resume`-capable CLI script.
4. **No per-task metrics** — Sullivan tracks total test loss only. Added `test_loss_add`, `test_loss_mult`, `test_acc_add`, `test_acc_mult`.
5. **Notebooks → scripts** — Converted to `src/train.py` and `src/llc_estimation.py` for clean Colab execution.

### Dependencies installed / verified

- **Local venv** (`.venv/`): `matplotlib`, `pandas`, `numpy`, `tqdm`, `pyyaml` — for plotting without GPU.
- **Colab** (installed at runtime via `colab_runner.ipynb`): `transformer_lens`, `devinterp==2.0.1`, `torch` (Colab default).

### devinterp v2 API (verified via GitHub source)

```python
from devinterp.slt.llc import llc
result = llc(model, dataset=hf_dataset, observables={},
             lr=epsilon, n_beta=nbeta,
             loss_fn=custom_loss_fn,
             num_chains=8, num_draws=500)
llc_mean = float(result["llc_mean"])
llc_std  = float(result["llc_std"])
```

Works with any `torch.nn.Module`. Dataset can be a HuggingFace Dataset or dict-style iterable. Custom `loss_fn(model, batch_dict) -> tensor` is supported.

**One uncertainty:** I haven't verified the exact `observables` parameter format in v2. Written `observables={}` (empty dict) — this should work for basic LLC computation. If it throws, check `devinterp.slt.llc` source in Colab.

### Project structure created

```
slt/
├── external/ArithmeticTransformer/   # Sullivan's repo (read-only)
├── src/
│   ├── utils.py           # data generation, loss_fn, model factory
│   ├── train.py           # training with --resume, checkpoint every 60 epochs
│   ├── llc_estimation.py  # LLC + drLLC, devinterp v1/v2 compatible
│   └── plotting.py        # all 7 figures from spec
├── configs/
│   ├── base.yaml             # hyperparameters
│   └── llc_calibration.yaml  # filled in after calibration (currently empty)
├── notebooks/
│   └── colab_runner.ipynb    # Colab workflow (mount Drive, run scripts)
├── results/
│   ├── checkpoints/   # per (ratio, seed): epoch_XXXXX.pt
│   ├── metrics/       # per (ratio, seed): .csv + _llc.csv
│   └── figures/       # final PDFs
└── .venv/             # local Python env (no GPU needed)
```

### What the first concrete task is

**Upload to Google Drive and run a test training run** for ratio=0.50, seed=0 to verify the pipeline end-to-end before the full sweep.

Steps:
1. Copy `slt/` folder to Google Drive (or use Drive desktop sync)
2. Open `notebooks/colab_runner.ipynb` in Colab
3. Run Section 0 (mount Drive, set path)
4. Run Section 1 (install deps)
5. Run Section 2 with RATIO=0.50, SEED=0, RESUME=False
6. Verify: checkpoints appear in `results/checkpoints/ratio_0.50/seed_0/`, CSV in `results/metrics/`

**Time estimate:** ~30 min for one (ratio, seed) training run on T4 GPU. Calibration adds ~20 min. Full sweep: 21 training runs ≈ 10 hours total.

### Waiting for Tair

Per PROJECT_CONTEXT.md Section 9: "Stop. Wait for Tair to confirm before starting any actual training."

**Questions before proceeding:**
- None. Architecture, data, and hyperparams match Sullivan's spec exactly.
- The only open question is whether to run the test training run immediately (< 1h) or wait — since it's under 2 hours, I would proceed per the "just decide" rule, but I'm stopping here as instructed by the first-session checklist.

### Decisions made (per "just decide" rule)

- Used devinterp v2 API with v1 fallback (rather than pinning v1.2.0)
- Checkpoint format: `epoch_XXXXX.pt` with model + optimizer + RNG state
- Metrics: separate `_llc.csv` files merged by plotting code (clean separation of concerns)
- Local venv: minimal (no torch) since GPU work is Colab-only
- Data split: per Sullivan's approach — shuffle add/mult pools separately before mixing at desired ratio

---

## 2026-05-23 01:00
- Updated `notebooks/colab_runner.ipynb`: code now clones from GitHub (`https://github.com/makataomu/slt-diplomka`) each session instead of requiring Drive upload.
- `results/` is symlinked to `MyDrive/slt_persist/results/` for checkpoint persistence across disconnects.
- `configs/llc_calibration.yaml` is backed up to `MyDrive/slt_persist/` after calibration and restored on each session start.

---

## 2026-05-23 (session 2) — Pipeline verified, full training started

### devinterp 2.0.1 — final fix

- `estimate_learning_coeff_with_summary` removed in 2.0.1; `sample_single_chain` hardcodes `data["input_ids"]`; v2 `llc()` requires zarr 3.2+ (Colab has older zarr)
- **Fix:** custom SGLD loop using `devinterp.optim.SGLD` (lr=epsilon, nbeta=nbeta, localization=gamma)
- Formula: `LLC = nbeta * (mean_draw_loss - init_loss)`; std from per-chain means
- Notebook pins `zarr==3.1.6` in Section 1 pip install (critical)

### Local pipeline verification (all PASS)

| Test | Result |
|------|--------|
| `train.py` 180 epochs | OK — 3 checkpoints, CSV written |
| `llc_estimation.py` on 3 checkpoints | OK — LLC/drLLC computed, _llc.csv written |
| `plotting.py` | OK — 5 of 7 figures generated (missing _70_30 because no ratio=0.70 data) |

Early-epoch LLC values are negative (expected — model hasn't grokked; SGLD finds lower-loss neighbors at wide early-training minima). Calibration should be run on the FINAL checkpoint (epoch 6000).

### Local training running now

- `ratio=0.50 seed=0`, 6000 epochs, checkpoint_every=60 → ~100 min on CPU
- Log: `results/train_0.50_0.log`
- After it completes: run calibration (`python src/llc_estimation.py --ratio 0.50 --seed 0 --calibrate`), inspect traces, fill in `configs/llc_calibration.yaml`

### What's committed (needs push)

Commit `117b26e` fixes:
- Notebook Section 1: zarr==3.1.6 pin, fix devinterp version check
- Notebook Section 3: nbeta default 46.2 (not 1.0)
- Notebook Section 6: explicit `--epochs 6000 --checkpoint_every 60`
- `llc_estimation.py`: suppress "Moving model to device" spam
- `plotting.py`: matplotlib colormaps deprecation fix

### Full experiment plan

1. **Local:** ratio=0.50 seed=0 training running (~100 min) → calibration → `llc_calibration.yaml`
2. **Colab:** Section 6 full sweep (21 runs, ~10h on T4) → already handles `--resume`
3. **Colab:** Section 4 LLC estimation per run (3–7h each, need approval)
4. **Colab:** Section 5 figures after all metrics collected

---
