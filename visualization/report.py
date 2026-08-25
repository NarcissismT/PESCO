"""Markdown and machine-readable report generation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .metrics import METRIC_NAMES, aggregate_metrics, bootstrap_ci, confusion_rows, write_csv
from .plots import generate_figures


def _fmt(value: Any, *, percent: bool = False) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{100 * number:.1f}%" if percent else f"{number:.4g}"


def _markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str], percent_columns: Iterable[str] = ()) -> str:
    percent = set(percent_columns)
    if not rows:
        return "_No records._\n"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        values = []
        for column in columns:
            values.append(_fmt(row.get(column), percent=column in percent))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def write_report(
    records: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    title: str = "PESCO experimental report",
    bootstrap: int = 300,
    seed: int = 7,
    formats: Sequence[str] = ("png", "svg"),
    weights: Optional[Mapping[str, float]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary = aggregate_metrics(records, by_split=True, weights=weights)
    summary_all = aggregate_metrics(records, by_split=False, weights=weights)
    confusion = confusion_rows(records)
    ci = bootstrap_ci(records, "vrs", n_boot=bootstrap, seed=seed, weights=weights)
    ci_rows = [{"method": method, "split": split, "vrs_ci_low": interval[0], "vrs_ci_high": interval[1]} for (method, split), interval in sorted(ci.items())]
    joined: List[Dict[str, Any]] = []
    for row in summary:
        extra = next((item for item in ci_rows if item["method"] == row["method"] and item["split"] == row["split"]), {})
        joined.append({**row, **extra})
    write_csv(destination / "metrics_by_method_split.csv", joined)
    write_csv(destination / "metrics_overall.csv", summary_all)
    write_csv(destination / "evidence_confusion.csv", confusion)
    write_csv(destination / "vrs_cluster_bootstrap.csv", ci_rows)
    report_metadata = {
        "title": title,
        "bootstrap_draws": int(bootstrap),
        "bootstrap_seed": int(seed),
        "figure_formats": list(formats),
        "vrs_weights": dict(weights or {"alpha": 1.0, "beta": 1.0, "gamma": 1.0, "eta": 1.0, "lambda": 0.1}),
        "record_count": len(records),
        **dict(metadata or {}),
    }
    (destination / "summary.json").write_text(json.dumps({"metadata": report_metadata, "summary": joined, "overall": summary_all, "confusion": confusion}, indent=2, ensure_ascii=False), encoding="utf-8")
    (destination / "report_metadata.json").write_text(json.dumps(report_metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    figure_paths = generate_figures(records, summary, destination, formats, overall_summary=summary_all)
    figure_names = {path.name for path in figure_paths}

    def figure_link(stem: str) -> Optional[str]:
        for extension in ("png", "svg", "pdf"):
            name = f"{stem}.{extension}"
            if name in figure_names:
                return f"[{name}]({name})"
        return None

    confusion_link = figure_link("evidence_confusion_matrices")
    strategy_stems = ("strategy_metrics", "action_choice_heatmap", "cost_frontier", "replication_fdr", "compute_budget_breakdown", "inference_budget_comparison", "belief_calibration", "discovery_pass_at_k", "preference_reversal", "split_vrs_heatmap")
    strategy_links = [link for stem in strategy_stems if (link := figure_link(stem))]

    all_methods = sorted({str(row.get("method", "Unknown")) for row in records})
    all_splits = sorted({str(row.get("split", "all")) for row in records})
    implementation_by_method: Dict[str, set[str]] = {}
    for row in records:
        method = str(row.get("method", "Unknown"))
        status = row.get("implementation_status")
        if status is not None:
            implementation_by_method.setdefault(method, set()).add(str(status))
    implementation_rows = [
        {"method": method, "implementation_status": "; ".join(sorted(statuses))}
        for method, statuses in sorted(implementation_by_method.items())
    ]
    report_lines = [
        f"# {title}",
        "",
        "> This report is generated from executable result records. `--demo` values are a pipeline smoke test; `--tier0` values are a deterministic pilot diagnostic. Neither is a trained-model scientific claim.",
        "",
        f"- Generated (UTC): `{datetime.now(timezone.utc).isoformat()}`",
        f"- Records: **{len(records)}**",
        f"- Methods: `{', '.join(all_methods) if all_methods else 'none'}`",
        f"- Splits: `{', '.join(all_splits) if all_splits else 'none'}`",
        "- Statistical unit: question/task cluster where `question_id` is available; bootstrap interval uses deterministic pilot resampling.",
        f"- VRS weights: `{json.dumps(report_metadata['vrs_weights'], sort_keys=True)}`",
        "",
        "## Overall result template (§26.1)",
        "",
    ]
    columns = ["method", "vrs", "state_macro_f1", "flip_accuracy", "effective_switch_rate", "invalid_claim_rate", "fdr", "replication_rate", "cost"]
    report_lines.append(_markdown_table(summary_all, columns, {"state_macro_f1", "flip_accuracy", "effective_switch_rate", "invalid_claim_rate", "replication_rate"}))
    report_lines += [
        "## Evidence-state breakdown (§17.2)",
        "",
        "See [evidence_confusion.csv](evidence_confusion.csv)" + (f" and {confusion_link}." if confusion_link else ".") + " Invalid→Supported and Insufficient→Refuted errors deserve explicit audit.",
        "",
        "## Implementation boundary",
        "",
        "Named external methods in the CPU pilot are labelled adapters/reference policies; this is not an external-paper or LLM-checkpoint reimplementation.",
        "",
        "## Strategy correction and discovery (§17.5–§17.12)",
        "",
        "See " + ", ".join(strategy_links) + (" when the corresponding fields are present." if strategy_links else "."),
        "",
        "## Paired-world / branch diagnostics (§9–§10)",
        "",
        "Branch trajectory plots are emitted when records contain `turn`/`step` and `utility`; they aggregate mean and min/max branch spread and cap legends via method-level small multiples. The data contract keeps `question_id`, `world_id`, `snapshot_id`, `branch_id`, and `seed` available for replay audits.",
        "",
        "## Reproducibility and interpretation boundary",
        "",
        "1. Keep the input result file, freeze manifest, verifier digest, and generated `summary.json` together.",
        "2. Do not treat demo records as model evidence; replace them with frozen Tier 0/1/2 runner outputs before making claims.",
        "3. Report ID and OOD splits separately, with cluster bootstrap and preregistered multiple-comparison correction.",
        "4. Missing VRS components remain `NA`; do not interpret them as zero scientific value.",
        "5. A higher VRS with lower cost is desirable, but no single chart proves scientific validity, independent confirmation, or global novelty.",
        "",
        "## Generated artifacts",
        "",
    ]
    if implementation_rows:
        # Insert the compact boundary table immediately after its explanatory line.
        marker = "Named external methods in the CPU pilot are labelled adapters/reference policies; this is not an external-paper or LLM-checkpoint reimplementation."
        marker_index = report_lines.index(marker) + 1
        report_lines.insert(marker_index, "")
        report_lines.insert(marker_index + 1, _markdown_table(implementation_rows, ["method", "implementation_status"]).rstrip("\n"))
    for path in sorted(figure_paths):
        report_lines.append(f"- [{path.name}]({path.name})")
    report_lines += [
        "- [metrics_by_method_split.csv](metrics_by_method_split.csv)",
        "- [metrics_overall.csv](metrics_overall.csv)",
        "- [evidence_confusion.csv](evidence_confusion.csv)",
        "- [vrs_cluster_bootstrap.csv](vrs_cluster_bootstrap.csv)",
        "- [summary.json](summary.json)",
        "- [report_metadata.json](report_metadata.json)",
        "",
    ]
    (destination / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    return {"summary": joined, "overall": summary_all, "confusion": confusion, "figures": [str(path) for path in figure_paths], "report": str(destination / "report.md")}
