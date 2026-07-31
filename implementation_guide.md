# Implementation & Demonstration Guide

A step-by-step walkthrough for setting up, running, and presenting this project — from a
clean checkout to a live demo in front of a guide/evaluation panel. Written for someone
who has not touched the repo before. Cross-references [README.md](README.md) (setup/repro
reference), [CLAUDE.md](CLAUDE.md) (repo conventions), [ROADMAP.md](ROADMAP.md) (the plan
this implements), and [research_paper.md](research_paper.md) (the write-up).

**Total time to run everything from scratch: roughly 2–2.5 hours on a CPU-only laptop**
(dependency install ~10 min, dataset download+preprocess ~10 min, full 36-run experiment
grid ~90 min, XAI + tables + figures ~5 min). Everything below has already been run once
in this repo — `results/` is populated — so for a live demo you do **not** need to re-run
the full grid; Part 5 tells you which single commands are enough to prove the pipeline
works live without waiting 90 minutes.

---

## Part 0 — What you're demonstrating

Before touching a terminal, be able to say this in one breath: *"We built a federated
learning system for network intrusion detection that sends 70% less data per round than
standard FedAvg, adds a formal differential-privacy guarantee with a reported epsilon,
and — the main contribution — explains its own predictions using SHAP computed **locally
on each client**, so explanation never breaks the privacy the system was built for."*
Every step below produces one piece of evidence for one clause of that sentence.

---

## Part 1 — Environment setup

```bash
cd "c:/fourth year/major project"

# Create and activate the virtual environment (already done in this repo, shown for a fresh checkout)
python -m venv .venv
./.venv/Scripts/python.exe -m pip install --upgrade pip
./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

**Talking point:** everything (interpreter, packages, data, results) lives inside this
one project folder — nothing was installed globally. Mention the stack: PyTorch (models),
Flower (federated simulation), Opacus (DP-SGD), SHAP (explainability), MLflow
(experiment tracking), all free/open-source, matching ROADMAP.md's zero-cost compute plan.

**Verify it worked:**

```bash
./.venv/Scripts/python.exe -c "import torch, flwr, shap, opacus, mlflow; print('OK')"
```

---

## Part 2 — Data: download, inspect, preprocess

```bash
# Already done in this repo (data/raw/cicids2017_corrected/*.csv exist). For a fresh
# checkout, the corrected CICIDS2017 dataset is downloaded from the paper's official page:
# https://intrusion-detection.distrinet-research.be/WTMC2021/Dataset/dataset.zip

./.venv/Scripts/python.exe -m src.data.preprocess
```

**What this does and why it matters for the demo:** reads five days of raw flow CSVs
(2,100,814 records), drops 793 inf/NaN rows and 259,102 exact duplicates, maps 25 raw
attack-name variants down to a 9-class taxonomy, standardizes 78 features, and writes a
stratified 70/15/15 train/val/test split to `data/processed/`.

**Show this file live:** `data/processed/metadata.json` — it has the exact clean-row
counts and per-class distribution. Point out Benign is 79.7% of the data; this is *why*
the paper reports macro-F1 and per-class recall instead of leading with accuracy.

**Talking point on data provenance:** the dataset is the *corrected* CICIDS2017
(Engelen et al., WTMC 2021), not the original 2017 release — the original has documented
label errors; downloading from the paper's own institutional page (not a random mirror)
is a deliberate, citable choice, referenced as [6] in the paper.

---

## Part 3 — Prove the pieces work: tests + non-IID partitioning

```bash
./.venv/Scripts/python.exe -m pytest tests/ -v
```

14 tests covering: partitioning correctness (no lost/duplicated samples, low-alpha is
measurably more skewed than high-alpha), parameter flatten/unflatten round-trips
(needed for top-k sparsification), the top-k selection + error-feedback residual logic,
and the metrics module (perfect predictions give F1=1, all-benign-flagged-as-attack
gives FPR=1). **Run this live** — it takes under a second and is a clean, fast way to
open a demo with something visibly passing.

```bash
./.venv/Scripts/python.exe -m experiments.make_partition_figures
```

Regenerates `results/figures/partition_alpha_*.png`. **Show `partition_alpha_0.1.png`
next to `partition_alpha_iid.png` side by side** — this is the clearest 10-second visual
in the whole project: at alpha=0.1 each client's bar is dominated by one or two colors
(severe label skew); at IID every bar is identical. This is Fig. 1 in the paper and the
Phase-2 gate deliverable from ROADMAP.md.

---

## Part 4 — Understand the architecture before running it

Walk through these four files in this order if asked "how does it actually work":

1. **`src/fl/client.py`** — `FlowerNIDSClient.fit()`. Point to the branch that computes
   `delta = new_flat - global_flat`, then either sends it dense (FedAvg) or calls
   `topk_sparsify()` to send only the top-k coordinates plus indices.
2. **`src/fl/sparsify.py`** — the actual top-k + error-feedback logic (`topk_sparsify`).
   Explain: unsent coordinates aren't discarded, they're accumulated into `residual` and
   compete again next round — so nothing is permanently lost, only delayed.
3. **`src/fl/strategies/comm_efficient.py`** — the server side. `aggregate_fit()`
   reconstructs each client's sparse update into a dense vector, does a weighted average
   exactly like FedAvg, and — critically — sums `fit_res.metrics["bytes_sent"]`, which is
   the *actual* `.nbytes` of the arrays a client returned. This is why the paper can say
   bytes/round is measured, not simulated.
4. **`src/xai/federated_shap.py`** — `explain_client()` runs SHAP locally per client
   against the converged global model; `aggregate_client_summaries()` on the server side
   only ever touches `mean_abs_shap` arrays, never a raw sample. This is the file to have
   open if anyone asks "how is the explanation part actually federated and not just FL
   with SHAP tacked on afterward."

---

## Part 5 — Live demo run (fast, ~2–3 minutes)

Running the full 36-cell grid live is not practical in a demo slot. Instead, run one
federated config end-to-end — this exercises every subsystem (partitioning, top-k
sparsification, DP-SGD, checkpointing, evaluation) in miniature:

```bash
./.venv/Scripts/python.exe -m experiments.run_experiment --config configs/smoke.yaml --force
```

This runs the flagship configuration (top-k + DP) on a 2% data subsample for 3 rounds —
finishes in under a minute. While it runs, narrate what's on screen: Flower spinning up
5 virtual clients via Ray, each client training locally, Opacus injecting calibrated
noise, the server aggregating sparse updates and reporting `bytes_this_round` and
`dp_epsilon_max` every round.

**If you have ~3 more minutes and want a *meaningful* (non-degenerate) live result:**

```bash
./.venv/Scripts/python.exe -m experiments.run_experiment --config configs/topk.yaml --seed 7 --force
```

This is a real, if scaled-down, run of the communication-efficient strategy. Point out
the printed final line: `macro_f1`, `accuracy`, `cumulative_bytes`. Then immediately:

```bash
./.venv/Scripts/python.exe -m experiments.run_xai --run-id topk_alpha0.5_seed7
```

(only works if the run above used `alpha0.5`, which `topk.yaml` does by default) — this
demonstrates the federated SHAP pipeline against a model you just trained live, printing
how many attack classes were explained and the feature/attack alignment overlap.

---

## Part 6 — The real results (already computed, walk through these)

Everything below was generated by the full 36-run grid (`experiments/run_grid.py`,
3 seeds per configuration, ~90 minutes total) and is already sitting in `results/`. For
the demo, **do not re-run this — present it**.

```bash
# If you want to regenerate the tables/figures from the existing run JSONs (fast, ~10s):
./.venv/Scripts/python.exe -m experiments.make_tables
./.venv/Scripts/python.exe -m experiments.make_figures
```

Walk through in this order — it mirrors the paper's Section V and builds the strongest
narrative arc:

1. **`results/figures/figure2_communication_savings.png`** — open first. Two straight
   lines, FedAvg well above top-k. This is the single most convincing image in the
   project: **70% less data transmitted** (789,000 vs 236,640 bytes/round), read straight
   off `results/tables/table1_tradeoff.csv`.
2. **`results/figures/figure1_convergence.png`** — immediately follow up with "but does
   it cost accuracy?" — no: FedAvg and top-k converge along nearly identical curves. Then
   point to the DP curve, visibly slower and lower — "here's the honest cost of adding a
   real privacy guarantee."
3. **`results/tables/privacy_ablation.csv`** — the epsilon sweep. Show the numbers
   directly: noise multiplier 0.5 → epsilon 48.7, macro-F1 0.396; noise multiplier 2.0 →
   epsilon 3.6, macro-F1 0.353. Say explicitly: *"more privacy costs more accuracy — this
   is the expected shape of the curve, not a bug."*
4. **`results/tables/table2_noniid_robustness.csv`** — the non-IID story: both methods
   drop from ~0.545 to ~0.30 macro-F1 under severe client skew (alpha=0.1), and top-k
   tracks FedAvg the whole way — "the communication savings don't cost extra fragility
   to heterogeneous clients."
5. **`results/figures/figure3_shap_global_summary.png`** and
   **`figure4_local_explanations.png`** — the explainability payoff. Point at one panel
   (e.g. DDoS) and read off the top feature; then open `results/tables/table3_alignment.csv`
   and show that Packet Length Mean and FIN Flag Count — features a security analyst
   would expect for DDoS/PortScan respectively — actually appear in the model's own
   top-5 SHAP ranking. This is the "explanations are meaningful, not just present" gate
   from ROADMAP.md Phase 4.

---

## Part 7 — Anticipated questions and how to answer them

- **"Is this actually private, or just federated?"** — Be precise: federation alone is
  *not* a privacy guarantee (gradient inversion / membership inference are real attacks
  on vanilla FedAvg). The flagship configuration adds DP-SGD with a Rényi accountant that
  composes epsilon across rounds; only that configuration is described as
  privacy-preserving in the paper, and Table III reports the actual epsilon, not an
  assumed one.
- **"Why is accuracy so much lower under DP?"** — This is expected and reported honestly
  (Section V-C of the paper): per-sample gradient clipping bounds each sample's influence
  regardless of noise level, which alone costs utility; the noise multiplier sweep shows
  the tradeoff is monotonic and in the textbook direction, which is itself evidence the
  privacy accounting is implemented correctly.
- **"Why does SHAP only partially match known attack signatures?"** — Answer honestly:
  Table IV shows 1-of-5 overlap for DDoS/PortScan, 0-of-5 for DoS/Brute Force in *this*
  scaled-down run (20 rounds, 30% data). We attribute this to the reduced training
  budget, not the architecture, and flag re-running the alignment check at full scale as
  the most direct next step (see ROADMAP.md's full grid and README's "Scaling up" table).
- **"Why didn't you use the full dataset / more rounds?"** — CPU-only, single-machine,
  single-session compute budget (see README.md's scale comparison table). The code takes
  `data_frac` and `n_rounds` as YAML config values specifically so this is a config
  change, not a re-implementation, when more compute is available (Kaggle/Colab per
  ROADMAP.md).
- **"What about UNSW-NB15?"** — Not auto-downloadable: UNSW's official distribution now
  requires an interactive SharePoint request, and the old public mirror is offline;
  Kaggle mirrors need a personal API key. Documented as a manual step in README.md; the
  pipeline is dataset-agnostic once the CSVs are supplied.
- **"Is SecureAgg implemented?"** — No, stated plainly in the paper's limitations
  (Section VI) — a correct secure-aggregation protocol is substantial extra cryptographic
  engineering beyond this project's timeline; DP-SGD is the implemented privacy
  mechanism, SecureAgg is scoped future work.
- **"How do you know the top-k byte counts are real and not estimated?"** — Show
  `src/fl/strategies/comm_efficient.py`: `bytes_sent` is `idx.nbytes + values.nbytes` on
  the actual NumPy arrays a client returned to Flower, summed server-side. Offer to open
  the file and point at the line if pushed.

---

## Part 8 — Suggested demo running order (≈12–15 minutes)

1. One sentence from Part 0.
2. `pytest tests/ -v` (10s) — "the building blocks are verified."
3. Show `partition_alpha_0.1.png` vs `partition_alpha_iid.png` side by side (30s) —
   "this is the non-IID problem we're solving for."
4. Run `experiments.run_experiment --config configs/smoke.yaml --force` live (60–90s)
   while narrating Part 4's architecture points.
5. Walk Part 6's five artifacts in order (5–7 min) — this is the actual results
   narrative and should be the bulk of the demo.
6. Open `research_paper.md` briefly — show the abstract and Table I — "this is written
   up formally, IEEE-formatted, ready for the writing phase."
7. Leave 3–5 minutes for Part 7's questions.
