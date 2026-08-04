# FedShield: Reproducible Federated IDS Evaluation

FedShield is a research prototype for evaluating federated learning (FL) for network
intrusion detection on corrected CICIDS2017 data. It compares dense FedAvg, top-*k*
update sparsification, and client-side DP-SGD under IID and Dirichlet non-IID partitions.
It also produces server-aggregated SHAP feature-importance summaries.

This repository supports a **pilot study**, not a production intrusion-detection system.
The current evidence is one dataset, 30% of its training split, 20 FL rounds, and three
seeds per reported configuration.

## Research claims and boundaries

- **Communication:** top-*k* reduced the measured **client-upload numeric-array payload**
  from 789,000 to 236,640 bytes/round in the legacy pilot (70%). This is not a 70%
  reduction in end-to-end network traffic: every selected client still receives a dense
  global model. New runs record upload, dense server downlink, and their combined
  numeric-array payload separately; framework serialization and transport headers are
  not measured.
- **Privacy:** DP-SGD is implemented with Opacus and an RDP accountant persisted per
  client across rounds. Reported epsilon is a record-level value at delta=1e-5 under the
  accountant's sampling and public-participation assumptions. FL alone, sparse updates,
  and this simulator do not provide deployment-wide confidentiality or client-level DP.
- **XAI:** the standard XAI artifact contains only per-class mean absolute SHAP summaries.
  Per-alert feature values and SHAP vectors are disabled by default and require the
  explicit local-debug export. The current XAI workflow is a simulation harness, not a
  distributed production client deployment.
- **Results:** macro-F1 and rare-class recall are primary. Accuracy is secondary because
  benign flows dominate CICIDS2017. Three seeds support descriptive mean±standard
  deviation, not claims of statistical equivalence.

## Quick start

```powershell
./.venv/Scripts/python.exe -m pip install -r requirements.txt
./.venv/Scripts/python.exe -m pytest tests -q

# Raw corrected CICIDS2017 CSVs must be available under data/raw/cicids2017_corrected/
./.venv/Scripts/python.exe -m src.data.preprocess

# Fresh smoke run: --force removes its matching checkpoint/residual/accountant state.
./.venv/Scripts/python.exe -m experiments.run_experiment --config configs/smoke.yaml --force

# Rebuild pilot artifacts from existing result JSON files.
./.venv/Scripts/python.exe -m experiments.make_tables
./.venv/Scripts/python.exe -m experiments.make_figures

# Server-facing federated-XAI summary artifact only.
./.venv/Scripts/python.exe -m experiments.run_xai --run-id topk_alpha0.5_seed0
```

`--export-local-examples` is a controlled demonstration option. It writes sensitive
per-alert records to `results/shap_local_debug/`; do not use it in a server-facing run.

## Reproducibility

Each run is defined by YAML configuration and writes a JSON result plus checkpoint.
`--force` intentionally starts again from clean run-specific state. Resuming without
`--force` uses the checkpoint. The project records deterministic Python, NumPy, and
PyTorch seeds, but experiments should still be rerun and compared before publication.

## What remains before submission

1. Run the prespecified full experiment matrix with at least five seeds, 100% training
   data, 100+ rounds, and a second dataset such as UNSW-NB15 or ToN-IoT.
2. Add recent FL-IDS baselines (at minimum FedProx) and report confidence intervals or
   paired statistical tests.
3. Evaluate explanation faithfulness, stability across seeds/clients, runtime, and a
   random-feature null baseline; top-feature overlap alone is insufficient validation.
4. Treat secure aggregation, adversarial robustness, deployment networking costs, and
   client-level privacy as future work unless they are actually implemented and tested.

See [research_paper.md](research_paper.md) for the IEEE-style manuscript draft,
[literarysurvey-template.md](literarysurvey-template.md) for the recent literature survey,
and [ROADMAP.md](ROADMAP.md) for the zero-budget submission plan.
