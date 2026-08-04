# Contributor Guide

## Purpose

FedShield is research code for a reproducible federated IDS evaluation. It is not a
production deployment. Preserve the distinction between measured evidence and planned
work in both code and documentation.

## Core invariants

1. Raw traffic and per-alert SHAP values must not enter normal server-facing artifacts.
   `results/shap/` contains aggregate summaries; local-debug records require explicit
   opt-in and belong in `results/shap_local_debug/`.
2. Label communication metrics accurately. `client_upload_bytes` is not total network
   traffic. `total_array_payload_bytes` includes the dense broadcast but still excludes
   framework serialization and transport overhead.
3. DP claims are record-level and conditional on documented assumptions. Do not describe
   plain FL, top-*k*, or a simulated setup as formally private.
4. A forced run must start with no prior run-specific checkpoint, residual, or DP
   accountant state. Do not weaken this reset behavior.
5. Generated tables and figures must come from result JSONs and scripts, never manual
   edits.

## Repository map

| Path | Responsibility |
|---|---|
| `configs/` | Reproducible run definitions |
| `src/data/` | Corrected-CICIDS2017 processing and client partitions |
| `src/fl/` | Flower client, DP-SGD, sparse aggregation, simulation |
| `src/xai/` | Summary-only federated-SHAP aggregation |
| `experiments/` | Runs, result generation, tables, and figures |
| `tests/` | Fast regression checks |
| `results/` | Generated local artifacts; do not hand-edit |

## Commands

```powershell
./.venv/Scripts/python.exe -m pytest tests -q
./.venv/Scripts/python.exe -m experiments.run_experiment --config configs/smoke.yaml --force
./.venv/Scripts/python.exe -m experiments.make_tables
./.venv/Scripts/python.exe -m experiments.make_figures
```

## Research-quality rules

- Use at least five seeds for new primary paper claims; three-seed pilot values are
  descriptive only.
- Report macro-F1, class-wise recall, and benign-to-attack FPR before accuracy.
- Record dataset provenance, preprocessing version, hardware, package versions, seeds,
  and full YAML with each experiment batch.
- Add a test whenever modifying partitioning, payload accounting, privacy-state handling,
  or XAI output boundaries.
- Read `ROADMAP.md` before expanding scope and update `research_paper.md` only after
  regenerated results are available.

## Known technical debt

Flower's legacy `start_simulation` API is deprecated in the installed 1.13.x release.
Migrate to a Flower App before a long-lived release. Secure aggregation, poisoning
defenses, client-level DP, second-dataset loading, and real wire-byte measurement are not
implemented.
