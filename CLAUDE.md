# CLAUDE.md

Guidance for Claude Code (or any future contributor) working in this repository.

## What this project is

A B.Tech major project building toward a publishable IEEE research paper:
**"An explainable, communication-efficient federated learning framework for network
intrusion detection that matches centralized detection accuracy under realistic
non-IID conditions, while providing SHAP-based per-alert explanations at near-zero
additional communication cost."**

Full plan, scope discipline, locked technical decisions, and the experiment grid live in
[ROADMAP.md](ROADMAP.md) — read that first for *why* each design choice was made. This
file is about *how the repo is laid out* and *how to run things*.

Primary contribution: federated XAI (client-side SHAP + server-side aggregation of
explanation summaries — raw data and raw SHAP values never leave the client).
Secondary: communication-efficient FL via top-k gradient sparsification.
Flagship privacy config: top-k strategy + DP-SGD (Opacus), reporting ε.

## Repo layout

```
configs/            YAML configs, one per experiment grid cell. Single source of truth
                     for a run — everything is reproducible from `python -m experiments.run_experiment --config <path>`.
data/
  raw/               Downloaded datasets (gitignored). cicids2017_corrected/*.csv from
                     the Engelen et al. WTMC2021 corrected CICIDS2017.
  processed/         Output of src/data/preprocess.py: train/val/test parquet + scaler + metadata.json (gitignored).
  partitions/        Cached Dirichlet/IID client partitions per (alpha, n_clients, seed) (gitignored).
src/
  data/              labels.py (attack taxonomy), preprocess.py (clean/split/scale), partition.py (Dirichlet/IID).
  models/            PyTorch MLP / 1D-CNN classifiers, factory in build.py.
  fl/                Flower client, custom Strategy (top-k sparsification + FedAvg), DP-SGD (Opacus) wrapper, simulation entrypoint.
  xai/               Federated SHAP: client-side local explanation + server-side aggregation of mean(|SHAP|); feature/attack alignment scoring.
  metrics.py         Macro-F1, per-class recall, FPR, confusion matrix helpers used everywhere (training, eval, table generation).
  utils/             Seeding, checkpoint/resume, MLflow logging, model_io (reload a trained model from its checkpoint given its run config — shared by run_experiment's final-metrics pass and run_xai).
experiments/
  run_experiment.py  Single run: reads one YAML config, trains (centralized or federated), logs to MLflow, writes a result JSON. Federated runs get one extra full-metrics evaluation pass on the converged global model so Table 1's per-class recall/FPR are real, not just macro-F1.
  run_grid.py        Iterates the experiment grid (§3 of ROADMAP.md), resumable — skips configs already completed in results/.
  run_xai.py         Federated SHAP + alignment check on one run's converged model (ROADMAP.md Phase 4).
  make_partition_figures.py  Phase-2 gate deliverable: per-client class distribution at alpha in {0.1, 0.5, 1.0, IID}.
  make_tables.py     Regenerates results/tables/*.csv from result JSONs.
  make_figures.py    Regenerates results/figures/*.png (convergence curves, communication savings, SHAP summary, alignment). Uses a colorblind-safe categorical palette validated with the dataviz skill's validator script.
results/             tables/, figures/, shap/ — every artifact here must be regenerable by an experiments/ script (no hand-edited results).
tests/               Sanity tests: partition correctness, model shapes, byte accounting, metric correctness.
mlruns/               MLflow local tracking store (gitignored).
```

## Environment

- Python 3.10, venv at `.venv/` (kept inside this folder per user instruction — nothing
  installed globally). Activate: `./.venv/Scripts/python.exe` (Windows, PowerShell/Git
  Bash) or `.venv\Scripts\activate` in cmd.
- Install: `./.venv/Scripts/python.exe -m pip install -r requirements.txt`.
- CPU only (no CUDA GPU on this machine — integrated Intel Iris Xe). All configs assume
  CPU training; models are intentionally small (MLP / 1D-CNN on tabular flow features)
  so this is fine per ROADMAP.md's compute plan.
- No Kaggle/Colab/browser access from this environment — free-tier GPU scaling described
  in ROADMAP.md is a manual step for the user, not something this environment can do.

## Data

- **CICIDS2017 (corrected)**: downloaded from the official project page for Engelen et
  al., WTMC 2021 (`intrusion-detection.distrinet-research.be/WTMC2021`) — a trusted
  academic source, not a random mirror. Already present under `data/raw/cicids2017_corrected/`.
- **UNSW-NB15 (secondary)**: NOT auto-downloaded. The official distribution (UNSW
  SharePoint) requires interactive access; Kaggle mirrors require an API key. Both need
  a human. See README.md for exact manual steps — the loader code should stay
  dataset-agnostic enough to accept it once supplied, but don't block the primary
  pipeline on it.
- Attack label taxonomy (raw CICIDS2017 label -> paper-facing class) lives in
  `src/data/labels.py` — this is the single place that defines the 9 classes (Benign,
  DoS, DDoS, PortScan, Bot, Brute Force, Web Attack, Infiltration, Heartbleed). If the
  raw data ever contains a label not in that map, `preprocess.py` raises rather than
  silently dropping rows — update the taxonomy, don't catch-and-ignore.

## Conventions

- Every run is driven by one YAML config in `configs/` and produces one entry under
  `results/` — this is what makes every table cell in the paper regenerable with one
  command (ROADMAP.md §1, "Experiment tracking" row). Don't hand-edit result files.
- Report **macro-F1, per-class recall, FPR** as primary metrics; accuracy is secondary
  and explicitly called out as misleading here (~80% of CICIDS2017 is benign).
- Every experiment config runs with >=3 seeds; report mean +/- std (ROADMAP.md §3 rule).
- Communication-efficient strategy reports **real** bytes/round: the top-k client
  sends actual sparse `(indices, values)` ndarrays through Flower's normal
  `NumPyClient.fit()` return, not a simulated/estimated byte count.
- Federated XAI architecture is load-bearing for the paper's novelty claim: SHAP must
  run **client-side** on the converged global model, server aggregates only
  mean(|SHAP|) summaries. A post-hoc centralized SHAP pass would undercut the paper's
  central claim — don't take that shortcut even if convenient.
- Checkpoint/resume is mandatory in the FL simulation loop (`src/fl/simulation.py`) —
  designed originally for free-tier GPU session limits, but also just makes local runs
  safe to interrupt.
- This session's actual experiment grid is deliberately scaled down from ROADMAP.md §3's
  full ~100-run grid (that assumes weeks of Kaggle/Colab time this environment doesn't
  have). Configs and scripts are written to scale back up unchanged — only the seed
  count / client count / round count differ. See README.md for what was actually run
  here vs. what's left as a `--full-grid` flag for the user to run later.

## What's explicitly out of scope here

Per the user's instruction, this repo goes up to the **end of Phase 5** (full
experiments & benchmarking). Phase 6 (writing the actual IEEE paper, advisor review,
venue submission, arXiv preprint, APC funding) is not attempted here — those require the
user's advisor, a LaTeX/Overleaf writing process, and human institutional steps that
can't be done from this sandboxed folder.
