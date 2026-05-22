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
