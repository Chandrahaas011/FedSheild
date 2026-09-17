# Repository Research Rules

This repository supports the paper **“Know When to Resume: Validation-Gated,
Resource-Bounded Recovery for Lightweight IoT Intrusion Detection.”**

The primary uncertainty method remains probability-based Mondrian conformal abstention.
The fixed robust-distance fusion is a completed negative ablation and must not be
described as an improvement.

The bidirectional NetFlow-v2 result is the strongest external-validity evidence. A
shared schema did not prevent target attack-recall collapse, and source-calibrated
Mondrian abstention remained confidently wrong. Do not describe standardized features
or conformal prediction as cross-testbed robustness solutions.

The primary deployment extension is the source-calibrated 512-flow quantile-JSD
sentinel. Its four operational states are `BENIGN`, `MALICIOUS`, `UNKNOWN`, and
`DOMAIN_SHIFT_DETECTED`. The last is a batch interlock, not an attack label. All target
windows were gated, so describe gross mismatch detection rather than restored target
classification or continuous IDS availability.

The independent label-blind validation uses a global feature-hash cap and compares JSD,
PSI, and KS on identical source roles and controlled target mixtures. Keep JSD as the
prespecified primary statistic. It reaches the frozen alarm criterion at 25% or larger
mixtures in both directions; this supports covariate-mismatch sensitivity, not natural
prevalence, natural temporal drift, or target intrusion-detection accuracy.
A frozen 40-record batching study compares 128--1,024 flows. The existing serialized
artifact remains 512-flow, but 256 flows is the smallest tested setting satisfying the
source-FPR ceiling in all ten cells while detecting every 25% target mixture. Treat 256
as a dataset-bounded operating recommendation, not a universal optimum.

The bounded recovery study uses disjoint target review, resumption-validation, and final-
audit roles. Only compact retraining passes the gate. The selected policy passes final
audit in 9/10 runs; never claim guaranteed safety. A post-hoc Wilson-bound sensitivity
gate rejects the failed seed while retaining nine resumptions; never present this as
preregistered confirmation. A separate prospective 20-run confirmation produced two
unsafe resumptions, so the Wilson gate is not a safety guarantee. A later two-stage
shadow-validation study resumes 14/20 with zero observed
audit failures, versus 16/20 and one failure for the same-seed single-stage counterfactual;
report both the safety gain and two unnecessarily withheld audit-passing models. Target retraining reduces source
performance, so describe recovered models as domain-specific and retain the source model.
Under prospective review-label noise, shadow-gate coverage falls from 8/10 clean runs to
1/10 at 5% noise and zero at 10--20%; never describe fail-closed behavior as robustness
without also reporting the severe availability loss and clean-control audit failure.
A prespecified depth-8, leaf-20, alpha-0.001 tree raises enhanced coverage to 7/10 at 5%
noise and 5/10 at 10%, with zero observed audit failures, but still resumes none at 20%.
Report the frozen design, same-seed standard counterfactual, and 20% failure boundary.
Under class-conditional corruption, 10% benign-to-malicious errors resume 6/10 with no
audit failure, while 10% malicious-to-benign errors resume 9/10 with two audit failures.
Always distinguish these directions; do not summarize them as generic 10% label noise.

## Non-Negotiable Rules

- Never fit preprocessing, feature selection, calibration, or thresholds on test data.
- Never restore contradictory feature fingerprints to improve scores.
- Keep raw/processed datasets and model artifacts out of Git.
- Do not manually type result values into plotting scripts.
- Use macro-F1 and malicious recall before accuracy.
- Label CICIDS weekday evaluation as compound day/campaign shift.
- Do not claim universal zero-day detection from one held-out family.
- Do not claim real-device deployment from desktop timing.
- Preserve the legacy federated code, but do not use old fixed-top-k results as evidence
  for RA-TopK or the research paper.
- Report negative findings.
- Never tune sentinel bins, aggregation, window size, or threshold on target outcomes.

## Canonical Inputs and Outputs

- CICIDS config: `configs/research_baseline.yaml`
- RT-IoT config: `configs/rt_iot2022.yaml`
- Cross-testbed config: `configs/netflow_v2_cross.yaml`
- Label-blind sentinel-validation config: `configs/netflow_v2_label_blind.yaml`
- Sentinel window-sensitivity config: `configs/domain_shift_window_sensitivity.yaml`
- Safe-resumption config: `configs/safe_resumption.yaml`
- Master results: `results/metrics/master_results.jsonl`
- Generated tables: `results/tables/research_*.csv`
- Generated figures: `results/figures/research_figure*`
- Paper: `paper/main.md` and `paper/main.tex`
- Self-contained paper handoff: `paper/overleaf_main.tex`
- Markdown-wrapped LaTeX handoff: `paper/OVERLEAF_LATEX_CODE.md`
- References: `paper/references.bib`
- Audit trail: `docs/PROJECT_AUDIT.md` and `docs/RESEARCH_TRANSFORMATION.md`

## Required Validation After Changes

```powershell
.\.venv\Scripts\python.exe -m experiments.sync_overleaf_markdown
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q src experiments
.\.venv\Scripts\python.exe -m experiments.make_research_outputs
```

Any changed claim must be traceable to a generated CSV and underlying validated JSON
record.
