# Repository Research Rules

This repository supports the paper **“Beyond Random Splits: Cross-Testbed Evaluation
of Conformal Abstention for Lightweight Intrusion Detection.”**

The primary uncertainty method remains probability-based Mondrian conformal abstention.
The fixed robust-distance fusion is a completed negative ablation and must not be
described as an improvement.

The bidirectional NetFlow-v2 result is the strongest external-validity evidence. A
shared schema did not prevent target attack-recall collapse, and source-calibrated
Mondrian abstention remained confidently wrong. Do not describe standardized features
or conformal prediction as cross-testbed robustness solutions.

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

## Canonical Inputs and Outputs

- CICIDS config: `configs/research_baseline.yaml`
- RT-IoT config: `configs/rt_iot2022.yaml`
- Cross-testbed config: `configs/netflow_v2_cross.yaml`
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
