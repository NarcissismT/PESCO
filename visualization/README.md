# PESCO visualization and report pipeline

This directory turns executable PESCO result records into the plots and tables
specified in the research plan (§17–§19 and §26). It is intentionally separate
from the trainer/environment so it can be used while the Tier 0 simulator is
still being validated.

## Quick smoke test (no real data required)

From the repository root:

```bash
python -m PESCO.visualization --demo \
  --output PESCO/artifacts/demo_report
```

The deterministic demo creates four minimum-pilot worlds (Supported, Refuted,
Insufficient, Invalid), four policy variants, ID/OOD splits, and branch fields.
It is a reproducibility/visualization smoke test only; it must not be quoted as
scientific evidence.

To execute the repository's actual deterministic Tier-0 simulator and trusted
verifier (4 worlds × 4 MVP actions × 4 exploration seeds = 64 branches), use:

```bash
python -m PESCO.visualization --tier0 \
  --output PESCO/artifacts/tier0_report
```

This is still a pilot diagnostic, not a trained-model result. Its evaluator
side oracle action is used only to populate branch-quality diagnostics after
execution; it is not exposed to a policy.

The output directory contains:

* `report.md` — a self-contained Markdown handoff with links to all artifacts;
* `metrics_by_method_split.csv` and `metrics_overall.csv` — plan-aligned
  metrics and denominators;
* `evidence_confusion.csv` — long-form state confusion counts;
* `vrs_cluster_bootstrap.csv` — deterministic question-cluster 95% intervals;
* `summary.json` — machine-readable aggregate;
* `report_metadata.json` — source/mode, bootstrap seed/draws, formats, and VRS weights;
* PNG and SVG figures: overview, evidence confusion matrices, strategy metrics,
  action-choice heatmap, cost frontier, replication/FDR, paired-world preference
  reversal, compute-budget breakdown, inference-budget comparison, belief
  calibration, discovery pass@k/best-of-k, split heatmap, optional ablations,
  and branch trajectories (only when the corresponding fields exist).

`branch_trajectories` is intentionally aggregated for readability: it computes
mean and min/max envelopes by method × world × turn, then pools worlds into
method-level small multiples when more than 12 groups are present. This avoids
large per-branch legends while preserving branch variability in the shaded band.

## Input contract

Input may be a JSON list, a JSON object containing `records`, `runs`,
`results`, `events`, or `branches`, or JSONL (one object per line). The loader
also flattens a parent record with a `branches` list. Canonical per-episode
fields are:

```json
{
  "method": "PESCO-Full",
  "split": "final_ood",
  "question_id": "rq_017",
  "world_id": "world_refuted_017",
  "world_pair_id": "pair_017",
  "snapshot_id": "snap_003",
  "branch_id": "branch_002",
  "seed": 103,
  "true_state": "Refuted",
  "predicted_state": "Refuted",
  "selected_action": "switch_to_alternative_method",
  "valid_claim": true,
  "belief_score": 0.91,
  "task_utility": 0.74,
  "replication_utility": 1.0,
  "discovery_utility": 0.0,
  "cost": 1.8,
  "switch": true,
  "switch_beneficial": true,
  "independent_confirmed": true,
  "new_path_verified": false,
  "turn": 3,
  "utility": 0.74
}
```

The visualizer is an **offline post-evaluation consumer**, not a policy prompt
builder. `true_state`, `world_id`, `hidden_outputs`, and trusted verdict fields
must be written only after an episode is complete and the independent verifier
has run. They are never included in `Observation.to_dict()` or sent back to a
policy. If a trajectory export omits an explicit ground-truth state, the state
accuracy chart reports `NA` rather than treating the model's own verdict as
truth.

Aliases are accepted for early runners (`policy`/`algorithm`,
`world_state`/`evidence_state`, `reported_state`, `total_cost`, etc.). Missing
values stay `NA`; they are never silently counted as zero. VRS is computed as

```text
valid_claim * (alpha*belief + beta*task + gamma*replication + eta*discovery)
  - lambda*cost
```

with frozen-at-run weights supplied by `--vrs-weights alpha,beta,gamma,eta,lambda`.
An explicit per-record `vrs` takes precedence when a runner computes it using a
frozen verifier.

## Reusable command interface

```bash
# Real runner output
python -m PESCO.visualization path/to/results.jsonl \
  --output PESCO/artifacts/final_v0_1 \
  --bootstrap 2000 --formats png,svg

# Use a different title and frozen VRS weights
python -m PESCO.visualization results.json \
  -o PESCO/artifacts/run_42 \
  --title 'PESCO Tier-1 pilot' \
  --vrs-weights 1,1,1,0.5,0.1
```

The module uses only the Python standard library plus NumPy/Matplotlib (already
used by the workspace). It does not require pandas, seaborn, or Plotly.

## Interpretation checklist

Before using a figure in a scientific report, retain the source records and
freeze manifest/verifier digest alongside the artifacts. Report ID and OOD
splits separately; use question/task clusters as the statistical unit; and
distinguish demo/smoke-test artifacts from confirmed Tier 0/1/2 results. The
plots support auditability but cannot by themselves establish independent
replication, global novelty, or unrestricted scientific capability.
