# FedShield: Residual-Adaptive Private Federated Intrusion Detection

FedShield is a major-project system for training a network intrusion detector across
heterogeneous data owners without centralizing their raw traffic. Its proposed method,
**Residual-Adaptive Top-*k* (RA-TopK)**, changes each client's communication budget from
round to round: it transmits the smallest bounded set of residual-corrected update
coordinates that retains a target fraction of update energy. The system can combine this
method with record-level DP-SGD and privacy-accounting, then reports utility, privacy,
and bidirectional numeric-array payload together.

**Paper title:** *FedShield: Residual-Adaptive Compression for Differentially Private
Federated Intrusion Detection Under Non-IID Traffic.*

## What is novel here

Fixed top-*k* sparsification sends the same fraction of every client update, even when
one client's update is concentrated and another's is diffuse. RA-TopK uses error
feedback and selects the minimum number of largest-magnitude coordinates that reaches a
configured L2-energy target, constrained by a minimum and maximum ratio. This makes the
upload budget responsive to each client's training signal while bounding communication
cost. The research question is whether this adaptive budget preserves IDS utility under
non-IID traffic better than a fixed budget at comparable payload.

This is a real implemented method, not a claim that an untested method has already won.
`configs/ra_topk_dp.yaml` is the flagship configuration. The existing legacy pilot
results are baselines only; RA-TopK must be evaluated through the prescribed matrix
before any result or novelty-performance claim is submitted.

## System scope and honest boundaries

- **Data and federation:** corrected CICIDS2017, simulated Flower clients, IID or
  Dirichlet non-IID partitions. Raw traffic is not sent to the server in this simulator.
- **Privacy:** DP-SGD uses Opacus with per-client RDP accounting. Epsilon is record-level
  at delta=1e-5 under the accounting assumptions. Federated learning, sparsity, and this
  simulator alone do not provide client-level DP, secure aggregation, or deployment-wide
  confidentiality.
- **Communication:** each round records client upload, dense server downlink, and their
  combined numeric-array payload. It excludes framework serialization, headers, retries,
  and real network overhead. Sparse uploads do not make the dense broadcast sparse.
- **Explainability:** the normal artifact contains only per-class, mean-absolute SHAP
  summaries. Per-alert values require `--export-local-examples` and remain local debug
  material, not a server-facing feature.

## Quick start

```powershell
./.venv/Scripts/python.exe -m pip install -r requirements.txt
./.venv/Scripts/python.exe -m pytest tests -q

# Put corrected CICIDS2017 CSVs under data/raw/cicids2017_corrected/ first.
./.venv/Scripts/python.exe -m src.data.preprocess

# Functional check; --force clears only the matching run's checkpoint, residual, and DP state.
./.venv/Scripts/python.exe -m experiments.run_experiment --config configs/smoke.yaml --force

# Flagship RA-TopK + DP run.
./.venv/Scripts/python.exe -m experiments.run_experiment --config configs/ra_topk_dp.yaml --force
```

The flagship parameters are `adaptive_min_ratio`, `adaptive_max_ratio`, and
`adaptive_energy_target`. A client chooses the least number of coordinates needed to
retain the target energy, then clips that number to the two ratios. Residual coordinates
not sent this round are carried forward through error feedback.

## Submission standard

Before presenting RA-TopK as publishable evidence, run at least five seeds, full data,
100+ rounds or convergence, IID plus alpha = 1.0/0.5/0.1, a second public dataset, and
the baselines Centralized, FedAvg, FedProx, fixed top-*k*, fixed top-*k* + DP, RA-TopK,
and RA-TopK + DP. Lead with macro-F1, minority-class recall, false-positive rate,
calibration, epsilon, and both payload measures; use confidence intervals and a
predeclared paired test.

See [research_paper.md](research_paper.md) for the IEEE-style manuscript,
[literarysurvey-template.md](literarysurvey-template.md) for the recent-literature gap,
[implementation_guide.md](implementation_guide.md) for reproducibility, and
[PRESENTATION_QA.md](PRESENTATION_QA.md) for a viva drill.
