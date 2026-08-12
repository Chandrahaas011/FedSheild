# Reproducible Implementation Guide

## 1. Prerequisites

- Python 3.10
- Windows PowerShell or an equivalent shell
- Corrected CICIDS2017 weekday CSVs under `data/raw/`
- RT-IoT2022 extracted at `data/raw/rt-iot2022/RT_IOT2022`
- Official NF-UNSW-NB15-v2 and NF-ToN-IoT-v2 ZIPs at the local paths in
  `configs/netflow_v2_cross.yaml`
- No GPU, paid service, or internet connection after data acquisition

Create the local environment:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 2. Build canonical data

```powershell
.\.venv\Scripts\python.exe -m src.data.research_prepare --force
.\.venv\Scripts\python.exe -m src.data.prepare_rt_iot2022 --config configs/rt_iot2022.yaml
.\.venv\Scripts\python.exe -m src.data.prepare_netflow_v2 --config configs/netflow_v2_cross.yaml
```

The builders:

- retain source/day provenance where available;
- remove non-finite records;
- fingerprint only model input features;
- remove every fingerprint carrying contradictory binary labels;
- collapse exact same-label duplicates before splitting;
- assign deterministic 70/15/15 roles for the within-dataset studies; and
- write dataset hashes, cleaning counts, features, and partition distributions to JSON.

The NetFlow-v2 builder instead creates deterministic bounded canonical samples. The
cross-testbed runner forms source-only 80/20 train/validation roles and leaves the other
dataset untouched as target evaluation data.

Never fit a scaler, selector, calibrator, sampler, or threshold on test data.

## 3. Baseline smoke test

```powershell
.\.venv\Scripts\python.exe -m experiments.run_research_baseline `
  --config configs/research_baseline.yaml `
  --model decision_tree `
  --seed 0
```

Expected output is a result JSON in `results/metrics/` and a serialized model in
`models/artifacts/`. The JSON must contain `status: complete`, dataset metadata,
protocol, seed, model, feature set, metrics, efficiency, durations, and hardware.

## 4. Full experiment matrix

```powershell
# Grouped-random baselines and compact pipeline
.\.venv\Scripts\python.exe -m experiments.run_research_grid

# Compound day/campaign shift
.\.venv\Scripts\python.exe -m experiments.run_shift_grid

# CICIDS2017 attack-family exclusion
.\.venv\Scripts\python.exe -m experiments.run_heldout_grid

# Independent RT-IoT2022 random and attack-family exclusion
.\.venv\Scripts\python.exe -m experiments.run_rt_iot2022_grid

# Five-seed confidence and conformal-abstention matrix
.\.venv\Scripts\python.exe -m experiments.run_uncertainty_grid --stage all

# Frozen robust-distance fusion ablation
.\.venv\Scripts\python.exe -m experiments.run_density_uncertainty_grid --stage all

# Frozen bidirectional standardized NetFlow-v2 transfer
.\.venv\Scripts\python.exe -m experiments.run_cross_netflow_grid `
  --config configs/netflow_v2_cross.yaml
```

Use `--force` only when intentionally replacing an existing result with a rerun under
the current code. Each run is isolated and the master JSONL ledger is rebuilt from
validated individual JSON records.

## 5. Generate paper outputs

```powershell
.\.venv\Scripts\python.exe -m experiments.make_research_outputs
```

This produces 16 CSV tables and 17 figures in PDF, SVG, and PNG. PDF/SVG are intended
for the IEEE paper; PNGs are previews. All values are derived from saved JSON results.

## 6. Quality gates

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q src experiments
```

The tests cover deterministic splitting behavior, disjointness, fingerprint handling,
metric correctness, feature order, serialization-compatible schemas, and master result
validity. A complete study should also satisfy:

- no incomplete result in the master ledger;
- five records for every primary stochastic cell;
- no manually entered paper figure values;
- model artifacts load and predict in the expected feature order; and
- reported numbers match generated CSVs.

## 7. Build the IEEE paper

The simplest Overleaf route needs only one file:

1. Create a blank Overleaf project.
2. Paste the fenced code from `paper/OVERLEAF_LATEX_CODE.md` over its `main.tex`.
3. Replace the four author placeholders and compile with pdfLaTeX.

The inline bibliography and conditional figure fallbacks make the first compilation
self-contained. Follow `paper/OVERLEAF_INSTRUCTIONS.md` to upload the vector figures.
`paper/OVERLEAF_LATEX_CODE.md` is generated from `paper/overleaf_main.tex`; run
`python -m experiments.sync_overleaf_markdown` after changing the canonical source.

With a local TeX distribution containing IEEEtran:

```powershell
Set-Location paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

LaTeX is not required to run experiments. Before submission, replace anonymous author
placeholders, apply the target venue page limit, confirm its double-blind policy, and
obtain advisor approval.

## 8. Interpretation rules

- Report macro-F1 and malicious recall before accuracy.
- Call the CICIDS weekday test compound day/campaign shift, not pure drift.
- Report held-out results per family; do not average them into a claim of universal
  zero-day detection.
- Treat process peak working set as run-level memory, not estimator-only RAM.
- Treat timing as hardware-specific.
- Report the compact feature selector as a negative shift result.
- Treat `unknown` as escalation, not a correct automatic detection.
- Claim conformal coverage only under exchangeability; shifted coverage is empirical.
- Do not promote the fixed distance fusion: it failed on held-out CICIDS PortScan and
  Brute Force and is retained as a negative ablation.
- Do not treat a shared NetFlow schema as evidence of transfer. The target labels in the
  cross-testbed protocol are evaluation-only, source conformal coverage does not carry
  over, and the measured target malicious recall is below 0.017 for every tested cell.
- Interpret paired bootstrap intervals with exact sign-flip tests. With five seed pairs,
  the smallest possible two-sided sign-flip p-value is 0.0625; seeds are not independent
  network deployments.
- Do not claim publication, real-world deployment, or novelty from high accuracy alone.
