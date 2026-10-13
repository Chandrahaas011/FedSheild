# Know When to Resume: Validation-Gated, Resource-Bounded Recovery for Lightweight IoT Intrusion Detection

Leakage-aware, resource-controlled research on lightweight intrusion detection with
explicit uncertainty handling under campaign and unseen-attack shift.

## Local React + Tailwind demonstration


The frontend is isolated in `ui/`: actual saved-model replay, raw-feature CSV
inference, domain diagnostics, read-only recovery evidence, and an experiment/figure
library. It does not capture or block traffic or activate deployed recovered models.
The supplied bioluminescent fluid background uses the teal/ink palette. A persistent
topbar light/dark control smoothly transitions the interface and fluid colors.
Reduced motion and WebGL initialization failures use a static fallback. Run
`npm test` inside `ui/` for theme-helper checks.
Light mode is the default unless you explicitly save a dark preference. Select
**Motion: on** in the topbar to enable the fluid even when reduced motion is
requested; System respects that accessibility setting, while Off stops animation.
The home page separately links to a presenter-paced start-to-finish demonstration
that trains candidates in memory, validates them, locks a decision, then reveals
untouched audit. It does not overwrite saved results or source artifacts. Local
compute checks took about 3.6 seconds per direction; allow 8–12 minutes to explain.

Build with `npm ci` and `npm run build` inside `ui/`. From the repository root,
run `.\.venv\Scripts\python.exe -m src.demo.server` and open http://127.0.0.1:8765.
Existing local datasets/artifacts are required; a fresh Git clone does not include
them. The saved demo guards remain historical 512-flow artifacts, not the later
256-flow sensitivity protocol. No console output modifies the research ledger.


The repository contains two related systems:

1. the primary classical-ML research pipeline for shift-aware IoT intrusion detection; and
2. the preserved legacy FedShield federated-learning simulator, retained for provenance.

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

The implemented extension adds a source-only 512-flow domain-shift sentinel beside the
classifier and conformal rejector. It uses mean Jensen--Shannon divergence over 38
source-fitted quantile histograms and a threshold calibrated on disjoint source-only
windows. Across both transfer directions and five seeds, it alarms on every mixed and
benign-only target window; mean source-audit false-alarm rates are 1.00% and 4.66%.
Guarded batch latency averages 4.63--5.61 ms and complete artifact size 318--524 KiB.
This is a fail-closed deployment interlock: `DOMAIN_SHIFT_DETECTED` withholds
classification and is not counted as an attack detection.

The completed recovery extension then compares no adaptation, probability recalibration,
and compact retraining using nested balanced target-label budgets of 100, 500, and 2,000.
Review, resumption-validation, and final-audit rows are disjoint. Only compact retraining
ever passes the frozen gate. Final-audit macro-F1 reaches 0.8789--0.9183 across direction
and budget means, versus 0.2016--0.3987 without adaptation. The validation-selected
policy passes final audit in 9 of 10 runs; one unsafe resumption is retained as a negative
result. Target retraining also reduces source-domain macro-F1, so recovered models must
remain domain-specific rather than overwrite the source model.

An explicitly post-hoc sensitivity analysis replaces the point FPR check with its 95%
Wilson upper bound. It keeps nine decisions resumable, gates the failed seed, and yields
zero unsafe resumptions on the same audits; this is exploratory, not confirmatory evidence.
Prospective testing on 20 unused-seed runs did not confirm zero failures: all 20
validation decisions resumed, but two failed untouched audit, one per direction. This
10% observed unsafe-resumption rate is retained as the current confirmatory result.

A second prospective protocol uses unused seeds 15--24 and splits the 10,000-row
validation role into disjoint 5,000-row selection and shadow-validation roles. On the
same 20 seeds, the single-stage counterfactual resumes 16 models with one audit failure;
the two-stage gate resumes 14 with zero observed failures. It safely changes one model
selection but withholds two models that would have passed audit. This is a measured
coverage--risk trade-off, not proof of safe deployment.

The first prospective label-quality study tests symmetric analyst-label errors on unused seeds 25--29.
With clean labels, shadow validation resumes 8/10 runs, seven safely and one unsafely.
At 5% symmetric label noise it resumes only 1/10 safely; at 10% and 20% it resumes none.
For the 2,000-label compact model, mean audit macro-F1 declines from 0.9199 to 0.8915,
0.8534, and 0.7522 as noise rises, while mean FPR reaches 0.2342 at 20%. The system fails
closed under noisy supervision, preserving safety at a severe availability cost.

A prespecified strongly regularized compact tree restores bounded availability without
audit-driven tuning. On unused seeds 30--34, standard-only versus enhanced-policy
coverage is 9/10 versus 9/10 at 0% noise, 3/10 versus 7/10 at 5%, 0/10 versus 5/10 at
10%, and zero for both at 20%. Every enhanced resumption passes audit. At 10% noise,
the robust 2,000-label tree averages 0.9084 audit macro-F1 versus 0.8435 standard. This
supports structural regularization up to moderate noise, not robustness at 20%.

A further frozen study uses unused seeds 35--39 to separate error direction. At 10%
benign-to-malicious corruption, only 6/10 runs resume and all pass audit. At 10%
malicious-to-benign corruption, 9/10 resume but two fail untouched audit, one per
direction. Clean shadow validation therefore reduces risk but cannot guarantee safe
resumption under biased review supervision.

Experiment persistence uses same-directory temporary writes and atomic file replacement.
If a run stops after saving its JSON but before updating the master ledger, recover the
index without rerunning experiments:

```powershell
.\.venv\Scripts\python.exe -m experiments.rebuild_results_ledger
```

Run only one experiment writer at a time. Atomic replacement protects individual files;
it is not a concurrent-writer transaction or a guarantee against disk failure.

The recommended paper title is:

> **Know When to Resume: Validation-Gated, Resource-Bounded Recovery for Lightweight
> IoT Intrusion Detection**

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
.\.venv\Scripts\python.exe -m experiments.run_domain_shift_grid --config configs/netflow_v2_cross.yaml
.\.venv\Scripts\python.exe -m src.data.prepare_netflow_v2 --config configs/netflow_v2_label_blind.yaml --force
.\.venv\Scripts\python.exe -m experiments.run_domain_shift_validation_grid --config configs/netflow_v2_label_blind.yaml
.\.venv\Scripts\python.exe -m experiments.run_safe_resumption_grid --config configs/safe_resumption.yaml
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
- The ledger contains 511 completed records: 141 baseline/shift records, 65
  probability-uncertainty records, 55 distance-aware ablation records, and 20 frozen
  bidirectional NetFlow-v2 transfer records, 10 source-only domain-sentinel runs, and
  10 independent label-blind sentinel-validation runs, 10 primary safe-resumption runs,
  20 prospective Wilson-gate runs, 20 prospective shadow-gate runs, and 40 review-label
  noise robustness runs, 40 regularized-noise recovery runs, and 40 frozen sentinel
  window-sensitivity runs, and 40 class-conditional review-noise runs.
- Generated statistical outputs include paired seed-wise effects, deterministic
  bootstrap intervals, exact sign-flip tests, and risk--coverage area summaries.
- The output builder produces 30 machine-readable CSV tables and 28 publication figures
  in PDF, SVG, and PNG; Figure 28 reports directional review-label risk.
- At 10% malicious-to-benign review corruption, 9/10 runs resume but two fail final
  audit; the reverse corruption direction reduces coverage without audit failures.
- Window sensitivity identifies 256 flows as the smallest tested size passing the
  source-FPR ceiling in all ten cells while detecting every tested 25% target mixture.
- The literature review was refreshed through 2026. Because recent work already covers
  cross-network degradation, adaptive IDS, and physical MCU deployment, the claimed gap
  is a source-frozen, target-untuned separation of flow confidence from batch domain
  validity under a shared feature ontology. A separate global label-blind cap and
  frozen JSD/PSI/KS mixture comparison support the sentinel claim. The JSD statistic
  itself is not claimed as novel. The extended contribution is a validation-gated,
  fixed-budget recovery comparison that also reports unsafe resumption and source-domain
  forgetting rather than assuming that adaptation restores trustworthy operation.

## Reproducibility & Architecture

The codebase separates feature extraction, data preparation, lightweight model training,
calibration, drift detection, and interactive demonstration into distinct, modular components.
All reported baseline metrics and tables can be reproduced deterministically via the experiments runner.
