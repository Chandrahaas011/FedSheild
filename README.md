# Beyond Random Splits: Cross-Testbed Evaluation of Conformal Abstention for Lightweight Intrusion Detection

Leakage-aware, resource-controlled research on lightweight intrusion detection with
explicit uncertainty handling under campaign and unseen-attack shift.

The repository now contains two related systems:

1. the primary classical-ML research pipeline described in the current paper; and
2. the preserved legacy FedShield federated-learning simulator, retained for provenance
   but not used to support the paper's empirical claims.

## Main finding

Corrected CICIDS2017 models reach approximately 0.995 random-test macro-F1, but the best
Monday--Wednesday to Friday compound-shift result is 0.7759. Held-out-family recall
ranges from zero to 0.9661. RT-IoT2022 independently reproduces the gap: HGB decreases
from 0.9903 random-test macro-F1 to 0.7441--0.9459 across excluded attacks. A
class-conditional conformal reject option converts many confident unseen-attack misses
into `unknown` alerts, but costs about 10% benign escalation and can reject attacks that
the base model already detects. It is a safety mechanism, not a universal robustness fix.
A frozen 50/50 fusion with class-conditional robust-distance novelty was also tested and
rejected: it restored confident misses for held-out CICIDS PortScan and Brute Force.

The strongest external-validity result comes from a frozen bidirectional transfer using
38 common, identifier-excluded NetFlow-v2 features. Source validation macro-F1 remains
0.9359--0.9665, yet target macro-F1 falls to 0.3173--0.4001 and malicious recall to
0.00009--0.01635. Source-calibrated Mondrian abstention does not rescue this shift:
depending on direction and model, 82.17%--99.28% of target attacks remain confident
benign misses. A shared feature schema therefore enables a fair transfer test, but does
not itself provide cross-testbed generalization.

The recommended paper title is:

> **Beyond Random Splits: Cross-Testbed Evaluation of Conformal Abstention for
> Lightweight Intrusion Detection**

## Repository map

```text
configs/                 Frozen experiment configurations
data/raw/                Local datasets (ignored by Git)
data/processed/          Canonical Parquet data and manifests (ignored)
docs/                    Audit, transformation log, and literature review
experiments/             Reproducible experiment and output-generation entry points
models/artifacts/        Serialized fitted models (ignored)
paper/                   IEEEtran draft, Markdown draft, and verified BibTeX
results/metrics/         Machine-readable experiment records and master JSONL
results/figures/         Publication PDF/SVG plus PNG previews (generated, ignored)
results/tables/          Generated CSV tables (ignored)
src/                     Data, model, evaluation, feature, and preserved FL code
tests/                   Reproducibility and correctness checks
```

## Environment

Tested locally with Python 3.10 on CPU-only Windows. No paid API, cloud service, or new
hardware is required.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Data

Large data is intentionally excluded from Git.

- Place the corrected CICIDS2017 weekday CSV files in `data/raw/`. See the
  [official CICIDS2017 page](https://www.unb.ca/cic/datasets/ids-2017.html) and correction
  paper DOI `10.1109/SPW53761.2021.00009`.
- Download the 3.3-MB CC BY 4.0
  [RT-IoT2022 archive](https://archive.ics.uci.edu/static/public/942/rt-iot2022.zip),
  extract `RT_IOT2022`, and set `raw_path` in `configs/rt_iot2022.yaml` if needed.
- The bounded cross-testbed study uses the official University of Queensland
  [NetFlow-v2 collection](https://staff.itee.uq.edu.au/marius/NIDS_datasets/): place the
  NF-UNSW-NB15-v2 and NF-ToN-IoT-v2 ZIP archives at the paths configured in
  `configs/netflow_v2_cross.yaml`. The UQ record permits academic research use with
  citation; raw archives remain local because no broader redistribution permission is
  assumed.

The local archive hashes used for the reported results are recorded in the generated
metadata and result JSON files.

## Reproduce the study

Prepare the canonical datasets:

```powershell
.\.venv\Scripts\python.exe -m src.data.research_prepare --force
.\.venv\Scripts\python.exe -m src.data.prepare_rt_iot2022 --config configs/rt_iot2022.yaml
.\.venv\Scripts\python.exe -m src.data.prepare_netflow_v2 --config configs/netflow_v2_cross.yaml
```

Minimal CICIDS baseline:

```powershell
.\.venv\Scripts\python.exe -m experiments.run_research_baseline --config configs/research_baseline.yaml --model decision_tree --seed 0
```

Complete constrained experiment matrix:

```powershell
.\.venv\Scripts\python.exe -m experiments.run_research_grid --config configs/research_baseline.yaml
.\.venv\Scripts\python.exe -m experiments.run_shift_grid --config configs/research_baseline.yaml
.\.venv\Scripts\python.exe -m experiments.run_heldout_grid --config configs/research_baseline.yaml
.\.venv\Scripts\python.exe -m experiments.run_rt_iot2022_grid --config configs/rt_iot2022.yaml
.\.venv\Scripts\python.exe -m experiments.run_uncertainty_grid --stage all
.\.venv\Scripts\python.exe -m experiments.run_density_uncertainty_grid --stage all
.\.venv\Scripts\python.exe -m experiments.run_cross_netflow_grid --config configs/netflow_v2_cross.yaml
```

Regenerate every reported table and figure from saved JSON results:

```powershell
.\.venv\Scripts\python.exe -m experiments.make_research_outputs
```

Run quality gates:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q src experiments
```

## Results and claims

- `results/metrics/master_results.jsonl` is the canonical experiment ledger.
- Each JSON stores dataset hash, split protocol, seed, code state, parameters, metrics,
  timing, efficiency measurements, and hardware.
- CSVs and figures are generated; no result is manually typed into plotting code.
- Accuracy is secondary; macro-F1, malicious recall, MCC, FPR, Brier score, and ECE are
  retained.
- The campaign-stable compact model is a negative result under shift and is not claimed
  to be drift-robust.
- The conformal method accepts only singleton prediction sets. Empty or ambiguous sets
  become `unknown` alerts; finite-sample coverage guarantees are claimed only under
  exchangeability, never for shifted or held-out tests.
- The ledger contains 281 completed records: 141 baseline/shift records, 65
  probability-uncertainty records, 55 distance-aware ablation records, and 20 frozen
  bidirectional NetFlow-v2 transfer records.
- Generated statistical outputs include paired seed-wise effects, deterministic
  bootstrap intervals, exact sign-flip tests, and risk--coverage area summaries.
- The output builder produces 16 machine-readable CSV tables and 17 publication figures
  in PDF, SVG, and PNG; Figures 16--17 report cross-testbed transfer and family outcomes.
- The literature review was refreshed through 2026. Because recent work already covers
  cross-network degradation, adaptive IDS, and physical MCU deployment, the claimed gap
  is specifically source-frozen, target-untuned transfer of both prediction and
  source-calibrated abstention under a shared feature ontology.

## Research documents

- [Project audit](docs/PROJECT_AUDIT.md)
- [Transformation log](docs/RESEARCH_TRANSFORMATION.md)
- [Literature and novelty review](docs/LITERATURE_REVIEW.md)
- [IEEE-style paper draft](paper/main.md)
- [IEEEtran source](paper/main.tex)
- [Self-contained Overleaf source](paper/overleaf_main.tex)
- [Copy-paste LaTeX code in Markdown](paper/OVERLEAF_LATEX_CODE.md)
- [Overleaf compilation instructions](paper/OVERLEAF_INSTRUCTIONS.md)
- [Verified BibTeX](paper/references.bib)
- [Presentation questions and answers](PRESENTATION_QA.md)

Publication is not guaranteed. The repository provides a defensible draft and
reproducible evidence; author details, venue fit, page limits, and advisor review remain
the student's responsibility.

For the fastest paper handoff, copy the fenced source from
`paper/OVERLEAF_LATEX_CODE.md` into a blank Overleaf project's `main.tex`. It contains
four author placeholders and an inline bibliography, and it compiles without uploaded
figures by rendering labeled fallback boxes. An automated test prevents this Markdown
copy and the canonical `paper/overleaf_main.tex` source from diverging.
