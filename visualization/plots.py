"""Matplotlib figures for the PESCO final-results template.

All functions accept plain dictionaries/lists, which keeps the visualizer
usable from lightweight simulators and from future experiment runners.
"""

from __future__ import annotations

import math
import os
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# CI and managed workspaces often expose a read-only ~/.config/matplotlib.
# Select a task-local writable cache before importing matplotlib so the CLI is
# quiet and deterministic without requiring users to set an environment var.
if not os.environ.get("MPLCONFIGDIR"):
    _mpl_cache = Path(tempfile.gettempdir()) / "pesco-matplotlib"
    try:
        _mpl_cache.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(_mpl_cache)
    except OSError:
        pass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from .metrics import STATES, aggregate_metrics, confusion_rows, record_method, record_split, normalize_state


PALETTE = {
    "Base": "#7f8c8d",
    "SFT": "#3498db",
    "GRPO-FourState": "#9b59b6",
    "SMOPD": "#e67e22",
    "DiscoPO": "#16a085",
    "TCPO": "#d35400",
    "CVT-RL": "#2c3e50",
    "Ecpo": "#34495e",
    "PESCO-Offline": "#27ae60",
    "PESCO-Full": "#c0392b",
}


def _methods(records: Sequence[Mapping[str, Any]]) -> List[str]:
    return sorted({record_method(row) for row in records})


def _splits(records: Sequence[Mapping[str, Any]]) -> List[str]:
    preferred = ["train", "dev", "promotion", "final_id", "final_ood", "id", "ood", "all"]
    found = {record_split(row) for row in records}
    return [item for item in preferred if item in found] + sorted(found - set(preferred))


def _save(fig: plt.Figure, output_dir: Path, stem: str, formats: Sequence[str] = ("png", "svg")) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in formats:
        path = output_dir / f"{stem}.{extension.lstrip('.') }"
        # Matplotlib can emit this benign warning for dense 14-method grids even when
        # the tight-bbox output is complete.  Keep CLI/CI logs actionable while still
        # saving both requested formats.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="constrained_layout not applied because axes sizes collapsed to zero.*",
                category=UserWarning,
            )
            fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def _bar_axis(ax: plt.Axes, labels: Sequence[str], values: Sequence[Optional[float]], *, ylabel: str, title: str,
              ylim: Optional[Tuple[float, float]] = None, percentage: bool = False) -> None:
    clean = [0.0 if value is None else float(value) for value in values]
    colors = [PALETTE.get(label, "#4c78a8") for label in labels]
    bars = ax.bar(range(len(labels)), clean, color=colors, alpha=0.9)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    if ylim:
        ax.set_ylim(*ylim)
    for bar, value in zip(bars, values):
        if value is not None:
            label = f"{100 * value:.1f}%" if percentage else f"{value:.3g}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, ha="center", va="bottom", fontsize=8)


def plot_overview(summary: Sequence[Mapping[str, Any]], output_dir: Path, formats: Sequence[str]) -> List[Path]:
    methods = [str(row["method"]) for row in summary if str(row.get("split", "all")) == "all"]
    if not methods:
        methods = [str(row["method"]) for row in summary]
    selected = {str(row["method"]): row for row in summary if str(row.get("split", "all")) == "all"}
    if not selected:
        selected = {str(row["method"]): row for row in summary}
    methods = list(selected)
    metrics = [
        ("vrs", "VRS", False),
        ("state_macro_f1", "State Macro-F1", True),
        ("flip_accuracy", "FlipAcc", True),
        ("replication_rate", "Replication", True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, (name, title, percent) in zip(axes.ravel(), metrics):
        _bar_axis(ax, methods, [selected[m].get(name) for m in methods], ylabel=title, title=title,
                  ylim=(0, 1.05) if percent else None, percentage=percent)
    fig.suptitle("PESCO overall evaluation (plan §26.1)", fontsize=14)
    return _save(fig, output_dir, "overview_metrics", formats)


def _matrix_for_method(records: Sequence[Mapping[str, Any]], method: str) -> List[List[int]]:
    matrix = [[0 for _ in STATES] for _ in STATES]
    index = {state: i for i, state in enumerate(STATES)}
    for row in records:
        if record_method(row) != method:
            continue
        truth = normalize_state(row.get("true_state", row.get("world_state")))
        pred = normalize_state(row.get("predicted_state", row.get("reported_state")))
        if truth in index and pred in index:
            matrix[index[truth]][index[pred]] += 1
    return matrix


def plot_confusion(records: Sequence[Mapping[str, Any]], output_dir: Path, formats: Sequence[str]) -> List[Path]:
    methods = _methods(records)
    if not methods:
        return []
    cols = min(4, max(1, len(methods)))
    rows = math.ceil(len(methods) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.8 * rows), squeeze=False, constrained_layout=True)
    last_image = None
    for i, method in enumerate(methods):
        ax = axes.ravel()[i]
        matrix = _matrix_for_method(records, method)
        last_image = ax.imshow(matrix, cmap="Blues", norm=Normalize(vmin=0, vmax=max(1, max(max(r) for r in matrix))))
        ax.set_xticks(range(len(STATES)), STATES, rotation=35, ha="right")
        ax.set_yticks(range(len(STATES)), STATES)
        ax.set_xlabel("Predicted state")
        ax.set_ylabel("True state")
        ax.set_title(method)
        for r, row in enumerate(matrix):
            for c, value in enumerate(row):
                ax.text(c, r, str(value), ha="center", va="center", color="white" if value > max(1, max(max(x) for x in matrix)) * .55 else "black")
    for ax in axes.ravel()[len(methods):]:
        ax.axis("off")
    if last_image is not None:
        fig.colorbar(last_image, ax=axes.ravel().tolist(), shrink=0.75, label="count")
    fig.suptitle("Evidence-state confusion matrices (plan §17.2)", fontsize=14)
    return _save(fig, output_dir, "evidence_confusion_matrices", formats)


def plot_strategy_metrics(summary: Sequence[Mapping[str, Any]], output_dir: Path, formats: Sequence[str]) -> List[Path]:
    selected = {str(row["method"]): row for row in summary if str(row.get("split", "all")) == "all"}
    if not selected:
        selected = {str(row["method"]): row for row in summary}
    methods = list(selected)
    metrics = [
        ("effective_switch_rate", "Effective switch", True),
        ("unnecessary_switch_rate", "Unnecessary switch", True),
        ("appropriate_persistence", "Appropriate persistence", True),
        ("refutation_acceptance", "Refutation acceptance", True),
        ("underpower_handling", "Underpower handling", True),
        ("invalid_repair_rate", "Invalid repair", True),
        ("invalid_claim_rate", "Invalid claim", True),
        ("vnpr", "VNPR", True),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    for ax, (name, title, percent) in zip(axes.ravel(), metrics):
        _bar_axis(ax, methods, [selected[m].get(name) for m in methods], ylabel=title, title=title,
                  ylim=(0, 1.05), percentage=percent)
    fig.suptitle("Strategy correction and discovery metrics (plan §17.5–§17.10)", fontsize=14)
    return _save(fig, output_dir, "strategy_metrics", formats)


def plot_action_heatmap(records: Sequence[Mapping[str, Any]], output_dir: Path, formats: Sequence[str]) -> List[Path]:
    """Show how methods distribute research actions by hidden evaluator state.

    The state is read only from post-evaluation records; it is never part of a
    policy observation.  Rows lacking a true state are omitted rather than
    assigning an inferred label.
    """
    methods = _methods(records)
    actions = sorted({str(row.get("selected_action", row.get("action", "unknown"))) for row in records
                      if row.get("selected_action", row.get("action")) is not None})
    if not methods or not actions:
        return []
    counts = [[0 for _ in actions] for _ in methods]
    m_index = {method: i for i, method in enumerate(methods)}
    a_index = {action: i for i, action in enumerate(actions)}
    for row in records:
        truth = normalize_state(row.get("true_state", row.get("world_state")))
        if truth is None:
            continue
        method = record_method(row)
        action = str(row.get("selected_action", row.get("action", "unknown")))
        if method in m_index and action in a_index:
            counts[m_index[method]][a_index[action]] += 1
    row_totals = [sum(row) for row in counts]
    if not any(row_totals):
        return []
    proportions = [[value / row_totals[r] if row_totals[r] else 0.0 for value in row] for r, row in enumerate(counts)]
    fig, ax = plt.subplots(figsize=(max(8, len(actions) * 1.25), max(4.5, len(methods) * .55 + 2)), constrained_layout=True)
    image = ax.imshow(proportions, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(actions)), actions, rotation=35, ha="right")
    ax.set_yticks(range(len(methods)), methods)
    ax.set_xlabel("Selected research action")
    ax.set_ylabel("Method")
    ax.set_title("Evidence-conditioned action distribution")
    for r in range(len(methods)):
        for c in range(len(actions)):
            ax.text(c, r, f"{proportions[r][c]:.0%}", ha="center", va="center", color="black" if proportions[r][c] < .65 else "white", fontsize=8)
    fig.colorbar(image, ax=ax, label="within-method proportion")
    return _save(fig, output_dir, "action_choice_heatmap", formats)


def plot_cost_frontier(summary: Sequence[Mapping[str, Any]], output_dir: Path, formats: Sequence[str]) -> List[Path]:
    selected = {str(row["method"]): row for row in summary if str(row.get("split", "all")) == "all"}
    if not selected:
        selected = {str(row["method"]): row for row in summary}
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    plotted = 0
    for method, row in selected.items():
        cost = row.get("cost")
        vrs = row.get("vrs")
        if cost is None or vrs is None:
            continue
        ax.scatter(float(cost), float(vrs), s=75, color=PALETTE.get(method, "#4c78a8"), label=method, edgecolor="white", linewidth=0.7)
        ax.annotate(method, (float(cost), float(vrs)), xytext=(5, 5), textcoords="offset points", fontsize=8)
        plotted += 1
    ax.set_xlabel("Normalized total cost")
    ax.set_ylabel("VRS")
    ax.set_title("Cost-normalized credible scientific value frontier")
    ax.grid(alpha=0.25)
    if plotted:
        ax.legend(loc="best", fontsize=8)
    return _save(fig, output_dir, "cost_frontier", formats)


def plot_reliability(summary: Sequence[Mapping[str, Any]], output_dir: Path, formats: Sequence[str]) -> List[Path]:
    """Compare independent replication with false-discovery rate."""
    selected = {str(row["method"]): row for row in summary if str(row.get("split", "all")) == "all"}
    if not selected:
        selected = {str(row["method"]): row for row in summary}
    methods = [method for method, row in selected.items() if row.get("replication_rate") is not None or row.get("fdr") is not None]
    if not methods:
        return []
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    width = 0.36
    x = list(range(len(methods)))
    replication = [0.0 if selected[m].get("replication_rate") is None else float(selected[m]["replication_rate"]) for m in methods]
    fdr = [0.0 if selected[m].get("fdr") is None else float(selected[m]["fdr"]) for m in methods]
    bars_a = ax.bar([value - width / 2 for value in x], replication, width=width, color="#2ca02c", label="Replication rate")
    bars_b = ax.bar([value + width / 2 for value in x], fdr, width=width, color="#d62728", label="FDR")
    ax.set_xticks(x, methods, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate")
    ax.set_title("Independent confirmation and false discovery control (§17.13)")
    ax.grid(axis="y", alpha=.25)
    for bars in (bars_a, bars_b):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.0%}", ha="center", va="bottom", fontsize=8)
    ax.legend(fontsize=8)
    return _save(fig, output_dir, "replication_fdr", formats)


def plot_compute_budget(records: Sequence[Mapping[str, Any]], output_dir: Path, formats: Sequence[str]) -> List[Path]:
    """Stack the cost ledger fields required by plan §16.4/§17.14."""
    methods = _methods(records)
    categories = [
        ("teacher", ("cost_teacher", "teacher_cost")),
        ("training", ("cost_training", "training_cost", "gpu_cost")),
        ("rollout", ("cost_rollout", "rollout_cost", "cost_environment")),
        ("verification", ("cost_verification", "verifier_cost")),
        ("confirmation", ("cost_confirmation", "confirmation_cost")),
        ("tokens", ("cost_tokens", "token_cost")),
    ]
    totals: Dict[str, Dict[str, float]] = {method: {name: 0.0 for name, _ in categories} for method in methods}
    observed = False
    for row in records:
        method = record_method(row)
        if method not in totals:
            continue
        for name, aliases in categories:
            value = next((row.get(alias) for alias in aliases if row.get(alias) is not None), None)
            if value is None:
                continue
            try:
                totals[method][name] += float(value)
                observed = True
            except (TypeError, ValueError):
                pass
    if not observed:
        return []
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    bottom = [0.0] * len(methods)
    x = list(range(len(methods)))
    for name, _ in categories:
        values = [totals[method][name] for method in methods]
        ax.bar(x, values, bottom=bottom, label=name)
        bottom = [old + value for old, value in zip(bottom, values)]
    ax.set_xticks(x, methods, rotation=30, ha="right")
    ax.set_ylabel("normalized cost units")
    ax.set_title("Matched compute and experiment budget ledger (§16.4)")
    ax.grid(axis="y", alpha=.25)
    ax.legend(fontsize=8, ncol=3)
    return _save(fig, output_dir, "compute_budget_breakdown", formats)


def plot_inference_budget(records: Sequence[Mapping[str, Any]], output_dir: Path, formats: Sequence[str]) -> List[Path]:
    """Compare single-path, fixed-branch and larger-search inference modes."""
    grouped: Dict[Tuple[str, str], List[float]] = {}
    for row in records:
        mode = row.get("inference_mode", row.get("search_mode", row.get("evaluation_mode")))
        if mode is None:
            continue
        value = row.get("vrs", row.get("utility", row.get("task_utility")))
        if value is None:
            continue
        try:
            grouped.setdefault((record_method(row), str(mode)), []).append(float(value))
        except (TypeError, ValueError):
            continue
    if not grouped:
        return []
    methods = sorted({key[0] for key in grouped})
    modes = sorted({key[1] for key in grouped})
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    width = .8 / max(1, len(modes))
    x = list(range(len(methods)))
    for index, mode in enumerate(modes):
        values = [sum(grouped.get((method, mode), [0.0])) / len(grouped.get((method, mode), [1.0])) for method in methods]
        offsets = [value + (index - (len(modes) - 1) / 2) * width for value in x]
        ax.bar(offsets, values, width=width, label=mode)
    ax.set_xticks(x, methods, rotation=30, ha="right")
    ax.set_ylabel("mean VRS / utility")
    ax.set_title("Single-path versus inference-time search budget (§16.6)")
    ax.grid(axis="y", alpha=.25)
    ax.legend(fontsize=8)
    return _save(fig, output_dir, "inference_budget_comparison", formats)


def plot_calibration(records: Sequence[Mapping[str, Any]], output_dir: Path, formats: Sequence[str]) -> List[Path]:
    """Reliability diagram for pre-confirmation belief probabilities."""
    grouped: Dict[str, List[Tuple[float, float]]] = {}
    for row in records:
        probability = row.get("hypothesis_probability", row.get("belief_probability", row.get("belief")))
        truth = normalize_state(row.get("true_state", row.get("world_state")))
        if probability is None or truth is None:
            continue
        try:
            probability = max(0.0, min(1.0, float(probability)))
        except (TypeError, ValueError):
            continue
        # Supported is the registered positive hypothesis in the MVP; callers
        # with multi-hypothesis tasks may provide explicit `belief_target`.
        target = row.get("belief_target")
        if target is None:
            outcome = 1.0 if truth == "Supported" else 0.0
        elif isinstance(target, str):
            normalized_target = normalize_state(target)
            outcome = 1.0 if normalized_target == truth else 0.0
        else:
            outcome = 1.0 if bool(target) else 0.0
        grouped.setdefault(record_method(row), []).append((probability, outcome))
    if not grouped:
        return []
    bins = 10
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ax.plot([0, 1], [0, 1], "k--", alpha=.6, label="perfect calibration")
    for method, values in sorted(grouped.items()):
        xs, ys = [], []
        for index in range(bins):
            lo, hi = index / bins, (index + 1) / bins
            bucket = [outcome for probability, outcome in values if lo <= probability < hi or (index == bins - 1 and probability == hi)]
            if bucket:
                xs.append(sum(probability for probability, _ in values if lo <= probability < hi or (index == bins - 1 and probability == hi)) / len(bucket))
                ys.append(sum(bucket) / len(bucket))
        if xs:
            ax.plot(xs, ys, marker="o", label=method, color=PALETTE.get(method))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("mean committed belief")
    ax.set_ylabel("empirical supported frequency")
    ax.set_title("Pre-confirmation belief calibration (§8)")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    return _save(fig, output_dir, "belief_calibration", formats)


def plot_pass_at_k(records: Sequence[Mapping[str, Any]], output_dir: Path, formats: Sequence[str]) -> List[Path]:
    """Report pass@1, pass@k and best-of-k for candidate discovery (§17.12)."""
    grouped: Dict[Tuple[str, str], List[Tuple[int, bool, float]]] = {}
    for row in records:
        candidate_group = row.get("candidate_group_id", row.get("discovery_group_id"))
        success = row.get("candidate_success", row.get("discovery_pass", row.get("new_path_verified")))
        if candidate_group is None or success is None:
            continue
        try:
            rank = int(row.get("candidate_rank", row.get("rank", 1)))
            utility = float(row.get("candidate_utility", row.get("utility", row.get("task_utility", 0.0))))
            success = bool(success)
        except (TypeError, ValueError):
            continue
        grouped.setdefault((record_method(row), str(candidate_group)), []).append((rank, success, utility))
    if not grouped:
        return []
    per_method: Dict[str, List[Tuple[float, float, float]]] = {}
    for (method, _group), values in grouped.items():
        ordered = sorted(values, key=lambda item: item[0])
        top_k = ordered[: max(1, max(item[0] for item in ordered))]
        pass1 = float(top_k[0][1]) if top_k else 0.0
        passk = float(any(item[1] for item in top_k))
        best = max(item[2] for item in top_k) if top_k else 0.0
        per_method.setdefault(method, []).append((pass1, passk, best))
    methods = sorted(per_method)
    values = []
    for method in methods:
        rows = per_method[method]
        values.append([sum(item[index] for item in rows) / len(rows) for index in range(3)])
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    width = .24
    x = list(range(len(methods)))
    labels = ["pass@1", "pass@k", "best-of-k"]
    for index, label in enumerate(labels):
        ax.bar([value + (index - 1) * width for value in x], [row[index] for row in values], width=width, label=label)
    ax.set_xticks(x, methods, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate / normalized utility")
    ax.set_title("Discovery pass@k and best-of-k (§17.12)")
    ax.grid(axis="y", alpha=.25)
    ax.legend(fontsize=8)
    return _save(fig, output_dir, "discovery_pass_at_k", formats)


def plot_preference_reversal(records: Sequence[Mapping[str, Any]], output_dir: Path, formats: Sequence[str]) -> List[Path]:
    """Visualize action-value changes across paired worlds.

    Runners may provide `world_pair_id`, `world_kind`/`world_state`, selected
    action, and a branch utility.  The function draws a panel per method and
    only connects worlds with at least two observed actions.  It is therefore a
    diagnostic of a confirmed reversal dataset, not an oracle-state input.
    """
    aggregates: Dict[Tuple[str, str, str, str], List[float]] = {}
    for row in records:
        method = record_method(row)
        pair = str(row.get("world_pair_id", row.get("pair_id", "")))
        world = str(row.get("world_kind", row.get("world_state", row.get("world_id", ""))))
        action = str(row.get("selected_action", row.get("action", "")))
        value = row.get("utility", row.get("branch_utility", row.get("task_utility")))
        if not pair or not world or not action or value is None:
            continue
        try:
            aggregates.setdefault((method, pair, world, action), []).append(float(value))
        except (TypeError, ValueError):
            continue
    if not aggregates:
        return []
    methods = sorted({key[0] for key in aggregates})
    cols = min(3, max(1, len(methods)))
    rows_n = math.ceil(len(methods) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(5 * cols, 4 * rows_n), squeeze=False, constrained_layout=True)
    for index, method in enumerate(methods):
        ax = axes.ravel()[index]
        tick_actions: List[str] = []
        pair_world_values: Dict[Tuple[str, str], Dict[str, float]] = {}
        for (m, pair, world, action), values in aggregates.items():
            if m == method:
                pair_world_values.setdefault((pair, world), {})[action] = sum(values) / len(values)
        plotted = 0
        for (pair, world), action_values in sorted(pair_world_values.items()):
            if len(action_values) < 2:
                continue
            actions = sorted(action_values)
            tick_actions = actions
            xs = list(range(len(actions)))
            ys = [action_values[action] for action in actions]
            ax.plot(xs, ys, marker="o", alpha=.55, label=f"{pair}:{world}")
            plotted += 1
        ax.set_xticks(range(len(tick_actions)), tick_actions, rotation=35, ha="right")
        ax.tick_params(axis="x", labelrotation=35)
        ax.set_ylabel("mean branch utility")
        ax.set_title(method)
        ax.grid(alpha=.25)
        if plotted:
            ax.legend(fontsize=6, ncol=2)
    for ax in axes.ravel()[len(methods):]:
        ax.axis("off")
    fig.suptitle("Paired-world action-value diagnostics (§10)", fontsize=14)
    return _save(fig, output_dir, "preference_reversal", formats)


def plot_split_heatmap(summary: Sequence[Mapping[str, Any]], output_dir: Path, formats: Sequence[str]) -> List[Path]:
    methods = sorted({str(row["method"]) for row in summary})
    splits = sorted({str(row.get("split", "all")) for row in summary})
    if len(splits) <= 1 or not methods:
        return []
    index_m = {m: i for i, m in enumerate(methods)}
    index_s = {s: i for i, s in enumerate(splits)}
    matrix = [[float("nan") for _ in splits] for _ in methods]
    for row in summary:
        value = row.get("vrs")
        if value is not None:
            matrix[index_m[str(row["method"])]][index_s[str(row.get("split", "all"))]] = float(value)
    fig, ax = plt.subplots(figsize=(max(7, 1.15 * len(splits)), max(4, .5 * len(methods) + 2)), constrained_layout=True)
    image = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(splits)), splits, rotation=30, ha="right")
    ax.set_yticks(range(len(methods)), methods)
    ax.set_xlabel("Evaluation split")
    ax.set_ylabel("Method")
    ax.set_title("VRS across train/dev/promotion/ID/OOD splits")
    for r in range(len(methods)):
        for c in range(len(splits)):
            value = matrix[r][c]
            if not math.isnan(value):
                ax.text(c, r, f"{value:.3g}", ha="center", va="center", color="white")
    fig.colorbar(image, ax=ax, label="VRS")
    return _save(fig, output_dir, "split_vrs_heatmap", formats)


def plot_ablation(summary: Sequence[Mapping[str, Any]], output_dir: Path, formats: Sequence[str]) -> List[Path]:
    # The final template labels ablations by method name; if absent, don't emit
    # an empty-looking figure.
    candidates = [row for row in summary if any(token in str(row.get("method", "")).lower() for token in ("no-", "ablation", "full"))]
    if len(candidates) < 2:
        return []
    selected = {str(row["method"]): row for row in candidates if str(row.get("split", "all")) == "all"}
    if not selected:
        selected = {str(row["method"]): row for row in candidates}
    methods = list(selected)
    names = [("vrs", "VRS"), ("flip_accuracy", "FlipAcc"), ("refutation_acceptance", "Refutation acceptance"), ("invalid_claim_rate", "Invalid claim"), ("vnpr", "VNPR")]
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    width = 0.15
    for i, (metric, label) in enumerate(names):
        values = [selected[m].get(metric) for m in methods]
        x = [j + (i - 2) * width for j in range(len(methods))]
        ax.bar(x, [0 if value is None else value for value in values], width=width, label=label)
    ax.set_xticks(range(len(methods)), methods, rotation=30, ha="right")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.set_title("PESCO ablations and negative controls (plan §18)")
    ax.grid(axis="y", alpha=.25)
    ax.legend(fontsize=8, ncol=3)
    return _save(fig, output_dir, "ablation_metrics", formats)


def plot_branch_trajectory(
    records: Sequence[Mapping[str, Any]],
    output_dir: Path,
    formats: Sequence[str],
    max_groups: int = 12,
) -> List[Path]:
    """Plot readable aggregate branch-utility trajectories.

    A raw branch log can contain hundreds of branch IDs, and drawing one line
    (and one legend entry) per branch makes the pilot figure unusable.  We first
    aggregate by ``method × world × turn`` and show the mean with a min/max
    envelope.  If that still exceeds ``max_groups``, we aggregate once more by
    ``method × turn``; this gives one line per policy while retaining the
    between-world envelope.  The default keeps the legend bounded at twelve
    entries and preserves the existing public function call signature.
    """

    def _number(value: Any) -> Optional[float]:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    def _world_label(row: Mapping[str, Any]) -> str:
        # Prefer evaluator-provided mechanism/state labels.  If only a world ID
        # is available, extract a recognizable state token; otherwise use a
        # compact ID so grouping remains deterministic.
        for key in ("world_kind", "world_state", "true_state", "evidence_state", "world_id"):
            value = row.get(key)
            if value is None:
                continue
            text = str(value)
            normalized = normalize_state(text)
            if normalized is not None:
                return normalized
            lowered = text.lower()
            for token, label in (("supported", "Supported"), ("refuted", "Refuted"), ("insufficient", "Insufficient"), ("invalid", "Invalid")):
                if token in lowered:
                    return label
            return text
        return "all worlds"

    # First pass: method × world × turn.  Each value is a branch observation,
    # not a pre-aggregated curve, so the envelope exposes actual branch spread.
    grouped: Dict[Tuple[str, str, float], List[float]] = {}
    for row in records:
        turn = _number(row.get("turn", row.get("step", row.get("event_index"))))
        utility = _number(row.get("utility", row.get("branch_utility", row.get("task_utility"))))
        if turn is None or utility is None:
            continue
        key = (record_method(row), _world_label(row), turn)
        grouped.setdefault(key, []).append(utility)
    if not grouped:
        return []

    # Keep world-aware lines when they are readable; otherwise collapse worlds
    # into a policy-level curve.  The fallback is especially useful for the
    # demo's 4 methods × 4 worlds = 16 possible lines.
    world_group_count = len({(method, world) for method, world, _ in grouped})
    aggregate_worlds = world_group_count > max(1, int(max_groups))
    if aggregate_worlds:
        compact: Dict[Tuple[str, float], List[float]] = {}
        for (method, _world, turn), values in grouped.items():
            compact.setdefault((method, turn), []).extend(values)
        curves: Dict[Tuple[str, str], Dict[float, List[float]]] = {}
        for (method, turn), values in compact.items():
            curves.setdefault((method, "all worlds"), {})[turn] = values
        title = "Research branch utility (method-level mean; worlds pooled)"
    else:
        curves = {}
        for (method, world, turn), values in grouped.items():
            curves.setdefault((method, world), {})[turn] = values
        title = "Research branch utility (method × world aggregate)"

    methods = sorted({method for method, _world in curves})
    line_styles = ("-", "--", ":", "-.")

    def _draw_series(ax: plt.Axes, method: str, world: str, turn_values: Mapping[float, Sequence[float]], *, label: str, world_index: int = 0) -> None:
        turns = sorted(turn_values)
        means = [sum(turn_values[turn]) / len(turn_values[turn]) for turn in turns]
        lows = [min(turn_values[turn]) for turn in turns]
        highs = [max(turn_values[turn]) for turn in turns]
        line, = ax.plot(
            turns,
            means,
            marker="o",
            markersize=4,
            linewidth=1.8,
            linestyle=line_styles[world_index % len(line_styles)],
            color=PALETTE.get(method, "#4c78a8"),
            label=label,
            alpha=.9,
        )
        # A single-observation group has zero-height shading, which is harmless
        # and keeps the same visual semantics for real multi-seed trajectories.
        ax.fill_between(turns, lows, highs, color=line.get_color(), alpha=.07, linewidth=0)

    if aggregate_worlds and len(methods) > 1:
        # Method-level small multiples remove the need for a large legend and
        # prevent broad world envelopes from visually merging with other
        # policies.  Each panel has one clearly labelled policy curve.
        ncols = 2
        nrows = math.ceil(len(methods) / ncols)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(11, max(3.4 * nrows, 4.2)),
            squeeze=False,
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        all_panel_turns = sorted({turn for turn_values in curves.values() for turn in turn_values})
        for index, method in enumerate(methods):
            ax = axes.ravel()[index]
            turn_values = curves[(method, "all worlds")]
            _draw_series(ax, method, "all worlds", turn_values, label=method)
            ax.set_title(method)
            ax.grid(alpha=.25)
            ax.set_xlabel("strategy turn")
            ax.set_ylabel("branch / task utility")
        for ax in axes.ravel()[len(methods):]:
            ax.axis("off")
        if len(all_panel_turns) == 1:
            axes.ravel()[0].set_xlim(all_panel_turns[0] - .5, all_panel_turns[0] + .5)
            axes.ravel()[0].set_xticks(all_panel_turns)
        fig.suptitle(title + "\nshading = min/max across worlds and branches", fontsize=13)
        return _save(fig, output_dir, "branch_trajectories", formats)

    # World-aware path: at most max_groups method/world lines, so a compact
    # legend remains useful for comparing confirmed paired worlds.
    fig, ax = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
    plotted = 0
    for group, turn_values in sorted(curves.items()):
        method, world = group
        world_index = sorted({item[1] for item in curves if item[0] == method}).index(world)
        _draw_series(ax, method, world, turn_values, label=f"{method} · {world}", world_index=world_index)
        plotted += 1
    all_turns = sorted({turn for turn_values in curves.values() for turn in turn_values})
    if len(all_turns) == 1:
        ax.set_xlim(all_turns[0] - .5, all_turns[0] + .5)
        ax.set_xticks(all_turns)
    ax.set_xlabel("strategy turn")
    ax.set_ylabel("branch / task utility")
    ax.set_title(title + "\nshading = min/max across branches")
    ax.grid(alpha=.25)
    if plotted:
        ax.legend(fontsize=8, ncol=2 if plotted > 4 else 1, loc="best", title="policy / world")
    return _save(fig, output_dir, "branch_trajectories", formats)


def generate_figures(records: Sequence[Mapping[str, Any]], summary: Sequence[Mapping[str, Any]], output_dir: str | Path,
                    formats: Sequence[str] = ("png", "svg"),
                    overall_summary: Optional[Sequence[Mapping[str, Any]]] = None) -> List[Path]:
    destination = Path(output_dir)
    # Overall figures must use an actual all-split aggregate.  Passing the
    # split-level table here would otherwise select whichever split sorts first
    # and could make the headline chart disagree with metrics_overall.csv.
    headline = list(overall_summary) if overall_summary is not None else list(summary)
    paths: List[Path] = []
    paths += plot_overview(headline, destination, formats)
    paths += plot_confusion(records, destination, formats)
    paths += plot_action_heatmap(records, destination, formats)
    paths += plot_strategy_metrics(headline, destination, formats)
    paths += plot_cost_frontier(headline, destination, formats)
    paths += plot_reliability(headline, destination, formats)
    paths += plot_compute_budget(records, destination, formats)
    paths += plot_inference_budget(records, destination, formats)
    paths += plot_calibration(records, destination, formats)
    paths += plot_pass_at_k(records, destination, formats)
    paths += plot_preference_reversal(records, destination, formats)
    paths += plot_split_heatmap(summary, destination, formats)
    paths += plot_ablation(headline, destination, formats)
    paths += plot_branch_trajectory(records, destination, formats)
    return paths
