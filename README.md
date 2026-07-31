# Privacy-Preserving, Explainable, Communication-Efficient Federated Learning for Network Intrusion Detection

Implementation for the project described in [ROADMAP.md](ROADMAP.md) — see that file for
the thesis statement, scope discipline, and the full 9-month plan. See
[CLAUDE.md](CLAUDE.md) for repo layout and contributor conventions.

**Status:** Phases 1–5 (data pipeline through full experiment benchmarking) are
implemented and have been run end-to-end on real data. Phase 6 (writing the IEEE paper
itself, advisor review, venue submission) is explicitly out of scope for this repo — see
"What's not done" below.

## What this is

A federated learning framework for network intrusion detection (CICIDS2017) that:

1. Trains a lightweight MLP classifier under FedAvg, with a **communication-efficient
   top-k sparsification strategy** as the improvement over vanilla FedAvg (real sparse
   `(indices, values)` updates sent through Flower — byte counts are measured, not
   simulated).
2. Adds **DP-SGD (Opacus)** as the flagship privacy configuration, reporting a real
   composed epsilon per client across rounds.
3. Runs **federated SHAP**: each client explains its own flagged traffic locally against
   the converged global model; the server aggregates only `mean(|SHAP|)` summaries per
   feature per attack class — raw data and raw per-sample SHAP values never leave the
   client. This federated-XAI architecture is the project's primary novelty claim.
4. Validates explanations against domain knowledge (feature/attack alignment).

## Quickstart

```bash
# from the project root
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt

# 1. Preprocess the raw CICIDS2017 corrected CSVs -> data/processed/*.parquet
./.venv/Scripts/python.exe -m src.data.preprocess

# 2. Sanity tests
./.venv/Scripts/python.exe -m pytest tests/ -q

# 3. Phase-2 gate: per-client non-IID distribution plots
./.venv/Scripts/python.exe -m experiments.make_partition_figures

# 4. One experiment
./.venv/Scripts/python.exe -m experiments.run_experiment --config configs/topk_dp.yaml

# 5. The full experiment grid (resumable — reruns skip already-completed cells)
./.venv/Scripts/python.exe -m experiments.run_grid

# 6. Federated XAI on the flagship run
./.venv/Scripts/python.exe -m experiments.run_xai --run-id topk_dp_alpha0.5_seed0

# 7. Regenerate every table/figure from the run outputs
./.venv/Scripts/python.exe -m experiments.make_tables
./.venv/Scripts/python.exe -m experiments.make_figures
```

Everything (venv, data, checkpoints, MLflow runs, results) lives inside this project
folder — nothing is installed or written globally.

## Results (this run)

All 36 grid cells completed with 0 failures. Full numbers in `results/tables/*.csv`;
figures in `results/figures/`. Headline findings from this scaled-down-but-real run:

- **Communication:** top-k sparsification (top 10% by magnitude) cuts bytes/round from
  789,000 (FedAvg, dense) to 236,640 — a **70% reduction** — while landing within
  0.2 macro-F1 points of FedAvg (0.5455 vs 0.5432) at alpha=0.5. This is the secondary
  contribution's headline number (Figure 2).
- **Privacy cost:** adding DP-SGD (flagship config) drops macro-F1 from 0.545 to 0.365 and
  roughly doubles rounds-to-converge (7.7 -> 15.7). The noise-multiplier ablation shows a
  clean, monotonic epsilon-vs-utility tradeoff (nm=0.5 -> eps=48.7, F1=0.396; nm=2.0 ->
  eps=3.6, F1=0.353) — the expected DP curve, not a broken one.
- **Non-IID robustness:** both FedAvg and top-k degrade from macro-F1 ~0.545 (IID/alpha=1.0)
  to ~0.30 at severe non-IID (alpha=0.1), and top-k tracks FedAvg within noise at every
  alpha level — i.e. the communication savings don't cost extra non-IID robustness.
- **Federated XAI:** client-side SHAP + server-aggregated summaries ran successfully on
  both the flagship (DP) and non-DP top-k models. Feature/attack alignment is modest but
  real (DDoS and PortScan each have 1/5 top-SHAP-features matching known attack
  signatures) — plausibly explained by this run's reduced training (20 rounds, 30% data)
  rather than a pipeline issue; the full-scale grid would be the way to check that.
- Every one of the above numbers should be read as **directional evidence from a scaled
  pilot**, not final paper numbers — see the scale-up table below before citing them
  anywhere.

## Data

- **CICIDS2017 (corrected, Engelen et al., WTMC 2021)** — the primary dataset. Downloaded
  from the paper's official project page
  (`intrusion-detection.distrinet-research.be/WTMC2021`), a trusted academic source, not
  a random mirror. ~1.84M clean flows across 9 classes (Benign + 8 attack categories) after
  dropping inf/NaN rows and duplicates — see `data/processed/metadata.json` for exact
  counts. Taxonomy mapping lives in [src/data/labels.py](src/data/labels.py).
- **UNSW-NB15 (secondary, per ROADMAP.md)** — **not auto-downloaded.** The official UNSW
  distribution now requires an interactive SharePoint request; the old CloudStor mirror
  AARNet used to host it on is offline; Kaggle mirrors require a personal API key. All
  three need a human in the loop, so this was left as a manual step rather than
  guessed/scraped. To add it: download `UNSW_NB15_training-set.csv` +
  `UNSW_NB15_testing-set.csv` yourself (UNSW SharePoint link on
  `research.unsw.edu.au/projects/unsw-nb15-dataset`, or a Kaggle mirror with your own
  `kaggle.json`), drop them under `data/raw/unsw_nb15/`, and extend
  `src/data/preprocess.py` with a `load_raw_unsw()` analogous to `load_raw()` — the rest
  of the pipeline (partitioning, FL, DP, XAI) is dataset-agnostic once you hand it a
  `(X, y, feature_names, class_names)` tuple.

## What was actually run in this environment vs. what's scaled down

This session has no GPU (integrated Intel Iris Xe only — CPU training throughout, 16
cores), no Kaggle/Colab access, and one sitting rather than 9 months. So the *code*
implements ROADMAP.md's full design, but the *experiment grid actually executed here* is
intentionally smaller than ROADMAP.md §3's ~100-run grid:

| Parameter | ROADMAP.md full grid | This run |
|---|---|---|
| Rounds per FL run | ~100s (until convergence) | 20 |
| Training data used | 100% (~1.29M rows) | 30% subsample (~386K rows) |
| Datasets | CICIDS2017 + UNSW-NB15 | CICIDS2017 only |
| Seeds per config | >=3 | 3 (kept — the "mean +/- std" rule was not cut) |
| SecureAgg ablation | implemented | **not implemented** — real secure aggregation needs a
  cryptographic protocol (masking/secret-sharing across clients), which is substantial
  extra engineering beyond a research-code session; documented here as future work rather
  than faked. The DP noise-multiplier sweep (the other half of the privacy ablation) *is*
  implemented and run. |

Total: 36 runs (vs. ~100), each ~1.5–2 minutes on this CPU, run via
`experiments/run_grid.py`. **To scale up:** raise `data_frac` to `1.0` and `n_rounds` to
`100+` in `configs/*.yaml`, add UNSW-NB15 per above, and either run locally with more
patience or port `experiments/run_grid.py` to a Kaggle/Colab notebook (the
checkpoint/resume logic in `src/fl/simulation.py` was built for exactly that free-tier
session-limit scenario, even though it ended up mainly protecting local runs here).

## Known limitations / honest caveats

- **SHAP × DP-trained models:** DP-SGD requires replacing BatchNorm with GroupNorm
  (`src/fl/dp.py`) for Opacus's per-sample gradients to be valid. SHAP's `DeepExplainer`
  has no dedicated propagation rule for GroupNorm and falls back to a generic
  gradient-based rule (logged as a `UserWarning` at runtime) — attributions on DP runs are
  a reasonable approximation, not as rigorously validated as the non-DP path. Worth a
  sentence in the paper's limitations section.
- **Server-side centralized test-set evaluation:** convergence curves (Figure 1) evaluate
  the global model each round against a held-out centralized test split, which is
  standard practice in FL *simulation* research code (the harness holds a labeled
  benchmark set) but is a simplification vs. a real deployment where "the server" would
  not have such a set to itself.
- **Tiny classes:** `Infiltration` (48 flows) and `Heartbleed` (11 flows) in the full
  corrected CICIDS2017 are so rare that most configs get 0 recall on them, especially
  after non-IID partitioning across 10 clients. This is reported honestly (per
  ROADMAP.md's own risk table: "FL loses badly on hard cases → that's a finding, not a
  failure") rather than hidden by only reporting accuracy.
- **Rounds-to-converge** (Table 1) uses a simple "first round reaching 95% of that run's
  own final macro-F1" definition — reasonable for a paper table, but state the definition
  explicitly if a reviewer asks.

## What's explicitly not done here (Phase 6)

Writing the actual IEEE paper (LaTeX/Overleaf), advisor review, venue submission
(IEEE Access / IEEE conference per ROADMAP.md §5), the arXiv preprint, and confirming
APC/registration funding are human/institutional steps outside what a sandboxed coding
session can do. `results/tables/*.csv` and `results/figures/*.png` are ready to drop
straight into Sections III–VI once that writing phase starts.
