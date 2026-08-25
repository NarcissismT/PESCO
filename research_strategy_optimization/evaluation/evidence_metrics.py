from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable, Mapping, Sequence, Tuple

from ..schemas import EvidenceState

STATES = tuple(s.value for s in EvidenceState)


def confusion_matrix(records: Sequence[Mapping[str, object]]) -> dict:
    matrix = {truth: {pred: 0 for pred in STATES} for truth in STATES}
    for record in records:
        truth = str(record.get("true_state", record.get("world_state", ""))).lower()
        pred = str(record.get("predicted_state", record.get("reported_state", ""))).lower()
        if truth in matrix and pred in matrix[truth]:
            matrix[truth][pred] += 1
    return matrix


def evidence_macro_f1(records: Sequence[Mapping[str, object]]) -> float:
    matrix = confusion_matrix(records)
    scores = []
    for state in STATES:
        tp = matrix[state][state]
        fp = sum(matrix[t][state] for t in STATES if t != state)
        fn = sum(matrix[state][p] for p in STATES if p != state)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def state_f1(records: Sequence[Mapping[str, object]]) -> dict:
    matrix = confusion_matrix(records)
    result = {}
    for state in STATES:
        tp = matrix[state][state]
        fp = sum(matrix[t][state] for t in STATES if t != state)
        fn = sum(matrix[state][p] for p in STATES if p != state)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        result[state] = 2 * p * r / (p + r) if p + r else 0.0
    return result

