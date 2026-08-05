# Implementation and Reproduction Guide

## Flagship method: RA-TopK

Residual-Adaptive Top-*k* (RA-TopK) combines the current client update with its saved
error-feedback residual and sends the smallest magnitude-ranked prefix that preserves
`adaptive_energy_target` squared-L2 energy. `adaptive_min_ratio` and
`adaptive_max_ratio` bound the upload. Use `configs/ra_topk_dp.yaml` for the DP flagship.

## 1. Environment

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
./.venv/Scripts/python.exe -m pytest tests -q
```

Use the project-local virtual environment. Record the Python version, package versions,
CPU/GPU, RAM, operating system, dataset checksum, and Git commit for every experiment
batch.

## 2. Data

Download the corrected CICIDS2017 CSV files from the Engelen *et al.* project source and
place the five daily files under `data/raw/cicids2017_corrected/`. Then run:

```powershell
./.venv/Scripts/python.exe -m src.data.preprocess
```

The pipeline removes non-finite rows and exact duplicates, maps labels to nine classes,
performs a stratified 70/15/15 split, and fits the scaler on training data only. This is a
random-row research split, not a temporal deployment evaluation.

## 3. Safe development run

```powershell
./.venv/Scripts/python.exe -m experiments.run_experiment --config configs/smoke.yaml --force
```

`--force` clears only the matching run's checkpoint, top-*k* residual, and DP-accountant
state before training. It is appropriate for a fresh smoke test. Omitting `--force`
resumes an interrupted run.

## 4. Read the outputs correctly

The result JSON contains:

- `client_upload_bytes`: client-to-server ndarray payload only.
- `server_downlink_bytes`: dense global-model ndarray payload sent to selected clients.
- `total_array_payload_bytes`: their sum.
- `dp_epsilon_max`: maximum per-client record-level epsilon reported by the RDP
  accountant; it is not a deployment-wide privacy guarantee.

The payload fields exclude Flower serialization, RPC framing, encryption, and transport
headers. Do not call them real network bandwidth.

## 5. XAI workflow

```powershell
./.venv/Scripts/python.exe -m experiments.run_xai --run-id topk_alpha0.5_seed0
```

The standard output under `results/shap/` contains only feature names, classes,
per-class weighted mean absolute SHAP values, and alignment metadata. For a controlled
local demonstration only, add `--export-local-examples`; the sensitive per-alert values
are written separately to `results/shap_local_debug/` and must not be shared with a
coordinator.

## 6. Regenerate artifacts

```powershell
./.venv/Scripts/python.exe -m experiments.make_tables
./.venv/Scripts/python.exe -m experiments.make_figures
```

Figures generated without `--include-local-debug` use no per-alert local records.

## 7. Before submitting any result

1. Run the full predeclared matrix on two datasets with at least five seeds.
2. Verify all configurations have identical data split, model, rounds, and comparison
   conditions except the intended independent variable.
3. Regenerate all tables/figures from JSON after the final batch.
4. Check confidence intervals, class-wise recall, calibration, and explanation quality.
5. Update the manuscript's pilot section with final values and have the advisor review
   claims, citation accuracy, venue requirements, and authorship.
