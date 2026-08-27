"""Shortcut probes for the Tier-1 raw-evidence benchmark.

The review asks for simple tabular baselines in addition to the differentiable
policy.  This module deliberately lives outside the trainer/P2 implementation.  A
probe consumes only the public ``Observation`` payload and uses the evaluator-side
branch utilities *only* as a post-hoc label/metric.  It never uses world IDs,
mechanism families, target actions, or confirmation receipts as input features.

``scikit-learn`` is used when it imports successfully.  The execution environment
for the reference repository does not always contain it, so a small NumPy fallback
is included for bounded diagnostics.  Fallback results are explicitly labelled
``numpy_fallback`` and are not silently presented as sklearn results.  Passing
``strict_sklearn=True`` (or ``--strict-sklearn`` in the runner) instead gives a
fail-closed artifact when sklearn is unavailable.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SHORTCUT_PROBE_SCHEMA = "pesco_shortcut_probe_v0.1"
ACTION_NAMES = (
    "continue_current_method",
    "add_samples_or_seeds",
    "repair_data_split",
    "switch_to_alternative_method",
)

# These names intentionally mirror the public numeric/raw receipt fields used by
# the raw-evidence policy track.  No ID, family, world kind, target, or branch
# outcome appears here.  The confirmation field is kept in ``all_raw`` only so the
# review can quantify its effect; ``without_confirmation`` is the leakage-sensitive
# comparison requested by the review.
_BASE_FEATURE_NAMES = (
    "effect_estimate",
    "ci_low",
    "ci_high",
    "ci_width",
    "log_sample_size",
    "log_seed_count",
    "remaining_budget",
    "turn",
    "hypothesis_probability",
    "current_method_is_method_b",
    "active_hypothesis_is_h_b",
)
_RAW_FEATURE_NAMES = (
    "treatment_confounder_correlation",
    "group_overlap_count",
    "replication_effect_delta",
    "replication_ci_width",
    "replication_sample_size",
    "replication_seed_count",
    "log_confirmation_pass_rate",
    "log_validity_count",
    "log_repeated_runs",
    "log_protocol_change_count",
)
FEATURE_SETS = ("all_raw", "without_confirmation")
MODEL_NAMES = ("logistic_regression", "random_forest", "gradient_boosting")


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _raw_map(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = observation.get("raw_evidence", {})
    if isinstance(raw, Mapping):
        return raw
    try:
        return {str(key): value for key, value in raw}
    except (TypeError, ValueError):
        return {}


def _normalize_feature_set(feature_set: str) -> str:
    normalized = str(feature_set)
    if normalized == "all_current_raw":
        normalized = "all_raw"
    if normalized in {"raw_no_confirmation", "without_confirmations", "no_confirmation"}:
        normalized = "without_confirmation"
    return normalized


def feature_names(
    feature_set: str = "all_raw",
    available_raw_keys: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Return stable feature names for a probe feature set.

    ``available_raw_keys`` lets the probe follow the current dataset schema.  This
    matters after the evaluator removes a receipt such as
    ``log_confirmation_pass_rate``: the current/raw feature set must not fabricate a
    zero-valued column and call it an observed feature.  With no key inventory the
    historical complete vocabulary is returned for backwards-compatible standalone
    encoding and tests.
    """

    normalized = _normalize_feature_set(feature_set)
    if normalized not in FEATURE_SETS:
        raise ValueError(f"unknown shortcut feature set: {feature_set!r}")
    if available_raw_keys is None:
        raw_names = list(_RAW_FEATURE_NAMES)
    else:
        available = {str(key) for key in available_raw_keys}
        raw_names = [name for name in _RAW_FEATURE_NAMES if name in available]
    names = list(_BASE_FEATURE_NAMES) + raw_names
    if normalized == "without_confirmation" and "log_confirmation_pass_rate" in names:
        names.remove("log_confirmation_pass_rate")
    return tuple(names)


def observation_features(
    observation: Mapping[str, Any],
    feature_set: str = "all_raw",
    available_raw_keys: Iterable[str] | None = None,
) -> np.ndarray:
    """Encode one public observation into numeric shortcut-probe features.

    The encoder is intentionally independent of PyTorch and the policy trainer, so a
    clean evaluation environment can run the probes without importing torch.  It
    follows the same numeric scaling conventions as the raw-evidence policy where
    applicable (log scaling for counts).  Missing receipts are represented by zero,
    never by a hidden fallback label.
    """

    names = feature_names(feature_set, available_raw_keys=available_raw_keys)
    interval = observation.get("confidence_interval", (-0.1, 0.1))
    try:
        ci_low, ci_high = float(interval[0]), float(interval[1])
    except (TypeError, ValueError, IndexError):
        ci_low, ci_high = -0.1, 0.1
    if not (math.isfinite(ci_low) and math.isfinite(ci_high)):
        ci_low, ci_high = -0.1, 0.1
    if ci_low > ci_high:
        ci_low, ci_high = ci_high, ci_low
    sample_size = max(0.0, _finite(observation.get("sample_size", 0)))
    seed_count = max(0.0, _finite(observation.get("seed_count", 0)))
    raw = _raw_map(observation)
    values: dict[str, float] = {
        "effect_estimate": _finite(observation.get("effect_estimate", 0.0)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_width": max(0.0, ci_high - ci_low),
        "log_sample_size": math.log1p(sample_size) / 8.0,
        "log_seed_count": math.log1p(seed_count) / 4.0,
        "remaining_budget": _finite(observation.get("remaining_budget", 0.0)) / 6.0,
        "turn": _finite(observation.get("turn", 0.0)) / 6.0,
        "hypothesis_probability": _finite(observation.get("hypothesis_probability", 0.5), 0.5),
        "current_method_is_method_b": 1.0 if observation.get("current_method") == "method_b" else 0.0,
        "active_hypothesis_is_h_b": 1.0 if observation.get("active_hypothesis_id") == "H_B" else 0.0,
    }
    for key in _RAW_FEATURE_NAMES:
        value = _finite(raw.get(key, 0.0))
        if key in {"group_overlap_count", "replication_sample_size", "log_repeated_runs", "log_protocol_change_count"}:
            value = math.log1p(max(0.0, value)) / 8.0
        elif key in {"replication_seed_count", "log_validity_count"}:
            value = math.log1p(max(0.0, value)) / 4.0
        values[key] = value
    result = np.asarray([values[name] for name in names], dtype=np.float64)
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError("shortcut features must be finite")
    return result


def _load_payload(dataset: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(dataset, Mapping):
        return dict(dataset)
    path = Path(dataset)
    return json.loads(path.read_text(encoding="utf-8"))


def _records(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = payload.get("examples", [])
    if not isinstance(raw, list):
        raise ValueError("shortcut probe dataset.examples must be a list")
    return [item for item in raw if isinstance(item, Mapping)]


def _utility_label(record: Mapping[str, Any]) -> int:
    utilities = record.get("branch_utilities")
    if isinstance(utilities, (str, bytes)):
        raise ValueError("shortcut probes require four evaluator branch_utilities per example")
    try:
        utility_count = len(utilities)
    except (TypeError, AttributeError):
        utility_count = -1
    if utility_count != len(ACTION_NAMES):
        raise ValueError("shortcut probes require four evaluator branch_utilities per example")
    values = np.asarray([_finite(item, float("nan")) for item in utilities], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("branch_utilities must be finite")
    # np.argmax has deterministic first-action tie handling.  This label is derived
    # from evaluator utility, not the hidden legacy target-action table.
    return int(np.argmax(values))


def _record_question_id(record: Mapping[str, Any], index: int) -> str:
    # The audit question_id is used only for cluster grouping/metrics, never as an
    # input feature.  Public replay exports may omit it; the neutral policy question
    # ID is deliberately *not* a useful cluster key, so each such row is isolated.
    value = record.get("question_id")
    return str(value) if value not in (None, "") else f"row_{index:06d}"


def _record_family(record: Mapping[str, Any]) -> str:
    metadata = _mapping(record.get("metadata"))
    return str(metadata.get("family", "unknown"))


def _check_records(records: Sequence[Mapping[str, Any]]) -> None:
    if not records:
        raise ValueError("shortcut probe dataset contains no examples")
    for index, record in enumerate(records):
        if not isinstance(record.get("observation"), Mapping):
            raise ValueError(f"example {index} has no public observation")
        _utility_label(record)


def _sklearn_import() -> tuple[dict[str, Any] | None, str | None]:
    try:
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return {
            "GradientBoostingClassifier": GradientBoostingClassifier,
            "RandomForestClassifier": RandomForestClassifier,
            "LogisticRegression": LogisticRegression,
            "make_pipeline": make_pipeline,
            "StandardScaler": StandardScaler,
        }, None
    except Exception as exc:  # ImportError and binary/ABI errors are both explicit.
        return None, f"{type(exc).__name__}: {exc}"


def sklearn_status() -> dict[str, Any]:
    modules, error = _sklearn_import()
    return {
        "available": modules is not None,
        "version": _sklearn_version() if modules is not None else None,
        "import_error": error,
    }


def _sklearn_version() -> str | None:
    try:
        import sklearn

        return str(getattr(sklearn, "__version__", "unknown"))
    except Exception:
        return None


class _NumpyMultinomialLogistic:
    """Small deterministic softmax regression used only when sklearn is absent."""

    implementation = "numpy_fallback"

    def __init__(self, *, seed: int = 0, max_iter: int = 600, learning_rate: float = 0.18, l2: float = 1e-3):
        self.seed = int(seed)
        self.max_iter = int(max_iter)
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(np.clip(shifted, -50.0, 50.0))
        return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_NumpyMultinomialLogistic":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        z = (x - self.mean_) / self.scale_
        n, width = z.shape
        classes = len(ACTION_NAMES)
        counts = np.bincount(y, minlength=classes).astype(np.float64)
        self.weights_ = np.zeros((width, classes), dtype=np.float64)
        self.intercept_ = np.log((counts + 1.0) / (counts.sum() + classes))
        one_hot = np.eye(classes, dtype=np.float64)[y]
        for step in range(max(1, self.max_iter)):
            probabilities = self._softmax(z @ self.weights_ + self.intercept_)
            error = probabilities - one_hot
            grad_w = (z.T @ error) / max(1, n) + self.l2 * self.weights_
            grad_b = error.mean(axis=0)
            rate = self.learning_rate / math.sqrt(1.0 + step / 100.0)
            self.weights_ -= rate * grad_w
            self.intercept_ -= rate * grad_b
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        z = (np.asarray(x, dtype=np.float64) - self.mean_) / self.scale_
        return self._softmax(z @ self.weights_ + self.intercept_)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1).astype(np.int64)


@dataclass
class _ClassNode:
    probabilities: np.ndarray
    feature: int | None = None
    threshold: float | None = None
    left: "_ClassNode | None" = None
    right: "_ClassNode | None" = None


class _ClassificationTree:
    def __init__(self, *, max_depth: int, min_samples_leaf: int, max_features: int, n_classes: int, rng: np.random.Generator):
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = int(max_features)
        self.n_classes = int(n_classes)
        self.rng = rng

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_ClassificationTree":
        self.root = self._grow(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.int64), 0)
        return self

    def _leaf(self, y: np.ndarray) -> _ClassNode:
        counts = np.bincount(y, minlength=self.n_classes).astype(np.float64)
        return _ClassNode(counts / max(1.0, counts.sum()))

    def _grow(self, x: np.ndarray, y: np.ndarray, depth: int) -> _ClassNode:
        node = self._leaf(y)
        if depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf or np.unique(y).size <= 1:
            return node
        feature, threshold, gain = self._best_split(x, y)
        if feature is None or gain <= 1e-12:
            return node
        mask = x[:, feature] <= threshold
        if int(mask.sum()) < self.min_samples_leaf or int((~mask).sum()) < self.min_samples_leaf:
            return node
        node.feature = feature
        node.threshold = float(threshold)
        node.left = self._grow(x[mask], y[mask], depth + 1)
        node.right = self._grow(x[~mask], y[~mask], depth + 1)
        return node

    def _best_split(self, x: np.ndarray, y: np.ndarray) -> tuple[int | None, float | None, float]:
        n, width = x.shape
        parent_counts = np.bincount(y, minlength=self.n_classes).astype(np.float64)
        parent_impurity = 1.0 - float(np.sum((parent_counts / max(1.0, n)) ** 2))
        if parent_impurity <= 1e-12:
            return None, None, 0.0
        feature_count = min(width, max(1, self.max_features))
        if feature_count >= width:
            features = np.arange(width)
        else:
            features = self.rng.choice(width, size=feature_count, replace=False)
        best_gain = 0.0
        best_feature: int | None = None
        best_threshold: float | None = None
        for feature in features:
            order = np.argsort(x[:, feature], kind="mergesort")
            sorted_values = x[order, feature]
            sorted_labels = y[order]
            left_counts = np.zeros(self.n_classes, dtype=np.float64)
            right_counts = parent_counts.copy()
            for position in range(n - 1):
                label = int(sorted_labels[position])
                left_counts[label] += 1.0
                right_counts[label] -= 1.0
                if sorted_values[position] == sorted_values[position + 1]:
                    continue
                left_n = position + 1
                right_n = n - left_n
                if left_n < self.min_samples_leaf or right_n < self.min_samples_leaf:
                    continue
                left_impurity = 1.0 - float(np.sum((left_counts / left_n) ** 2))
                right_impurity = 1.0 - float(np.sum((right_counts / right_n) ** 2))
                gain = parent_impurity - (left_n * left_impurity + right_n * right_impurity) / n
                if gain > best_gain:
                    best_gain = float(gain)
                    best_feature = int(feature)
                    best_threshold = float((sorted_values[position] + sorted_values[position + 1]) / 2.0)
        return best_feature, best_threshold, best_gain

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        result = []
        for row in np.asarray(x, dtype=np.float64):
            node = self.root
            while node.feature is not None and node.left is not None and node.right is not None:
                node = node.left if row[node.feature] <= float(node.threshold) else node.right
            result.append(node.probabilities)
        return np.asarray(result, dtype=np.float64)


class _NumpyRandomForest:
    """Lightweight random CART ensemble, intentionally labelled as a fallback."""

    implementation = "numpy_cart_ensemble"

    def __init__(self, *, seed: int = 0, n_estimators: int = 64, max_depth: int = 5, min_samples_leaf: int = 2):
        self.seed = int(seed)
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_NumpyRandomForest":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        rng = np.random.default_rng(self.seed)
        n, width = x.shape
        max_features = max(1, int(math.sqrt(max(1, width))))
        self.trees_ = []
        for _ in range(max(1, self.n_estimators)):
            sample = rng.integers(0, n, size=n)
            tree = _ClassificationTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features,
                n_classes=len(ACTION_NAMES),
                rng=rng,
            )
            tree.fit(x[sample], y[sample])
            self.trees_.append(tree)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if not self.trees_:
            raise RuntimeError("random forest has not been fitted")
        return np.mean([tree.predict_proba(x) for tree in self.trees_], axis=0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1).astype(np.int64)


@dataclass
class _RegNode:
    value: float
    feature: int | None = None
    threshold: float | None = None
    left: "_RegNode | None" = None
    right: "_RegNode | None" = None


class _RegressionTree:
    def __init__(self, *, max_depth: int, min_samples_leaf: int, max_features: int, rng: np.random.Generator):
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = int(max_features)
        self.rng = rng

    def fit(self, x: np.ndarray, residual: np.ndarray) -> "_RegressionTree":
        self.root = self._grow(np.asarray(x, dtype=np.float64), np.asarray(residual, dtype=np.float64), 0)
        return self

    def _grow(self, x: np.ndarray, y: np.ndarray, depth: int) -> _RegNode:
        node = _RegNode(float(np.mean(y)) if len(y) else 0.0)
        if depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf or np.allclose(y, y[0] if len(y) else 0.0):
            return node
        feature, threshold, gain = self._best_split(x, y)
        if feature is None or gain <= 1e-12:
            return node
        mask = x[:, feature] <= threshold
        if int(mask.sum()) < self.min_samples_leaf or int((~mask).sum()) < self.min_samples_leaf:
            return node
        node.feature, node.threshold = int(feature), float(threshold)
        node.left = self._grow(x[mask], y[mask], depth + 1)
        node.right = self._grow(x[~mask], y[~mask], depth + 1)
        return node

    def _best_split(self, x: np.ndarray, y: np.ndarray) -> tuple[int | None, float | None, float]:
        n, width = x.shape
        total_sum = float(np.sum(y))
        total_sq = float(np.dot(y, y))
        parent_sse = total_sq - total_sum * total_sum / max(1, n)
        if parent_sse <= 1e-12:
            return None, None, 0.0
        feature_count = min(width, max(1, self.max_features))
        features = np.arange(width) if feature_count >= width else self.rng.choice(width, size=feature_count, replace=False)
        best_gain, best_feature, best_threshold = 0.0, None, None
        for feature in features:
            order = np.argsort(x[:, feature], kind="mergesort")
            values, labels = x[order, feature], y[order]
            left_sum = left_sq = 0.0
            for position in range(n - 1):
                label = float(labels[position])
                left_sum += label
                left_sq += label * label
                if values[position] == values[position + 1]:
                    continue
                left_n, right_n = position + 1, n - position - 1
                if left_n < self.min_samples_leaf or right_n < self.min_samples_leaf:
                    continue
                right_sum, right_sq = total_sum - left_sum, total_sq - left_sq
                left_sse = left_sq - left_sum * left_sum / left_n
                right_sse = right_sq - right_sum * right_sum / right_n
                gain = parent_sse - left_sse - right_sse
                if gain > best_gain:
                    best_gain = float(gain)
                    best_feature = int(feature)
                    best_threshold = float((values[position] + values[position + 1]) / 2.0)
        return best_feature, best_threshold, best_gain

    def predict(self, x: np.ndarray) -> np.ndarray:
        result = []
        for row in np.asarray(x, dtype=np.float64):
            node = self.root
            while node.feature is not None and node.left is not None and node.right is not None:
                node = node.left if row[node.feature] <= float(node.threshold) else node.right
            result.append(node.value)
        return np.asarray(result, dtype=np.float64)


class _NumpyGradientBoosting:
    """Multiclass gradient boosting with shallow regression trees."""

    implementation = "numpy_gradient_boosted_trees"

    def __init__(self, *, seed: int = 0, n_estimators: int = 48, learning_rate: float = 0.08, max_depth: int = 2, min_samples_leaf: int = 3):
        self.seed = int(seed)
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(np.clip(shifted, -50.0, 50.0))
        return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_NumpyGradientBoosting":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        n, width = x.shape
        classes = len(ACTION_NAMES)
        counts = np.bincount(y, minlength=classes).astype(np.float64)
        self.base_logits_ = np.log((counts + 1.0) / (counts.sum() + classes))
        logits = np.tile(self.base_logits_, (n, 1))
        one_hot = np.eye(classes, dtype=np.float64)[y]
        rng = np.random.default_rng(self.seed)
        max_features = max(1, int(math.sqrt(max(1, width))))
        self.trees_: list[list[_RegressionTree]] = []
        for _ in range(max(1, self.n_estimators)):
            probabilities = self._softmax(logits)
            round_trees: list[_RegressionTree] = []
            for cls in range(classes):
                tree = _RegressionTree(
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    max_features=max_features,
                    rng=rng,
                )
                tree.fit(x, one_hot[:, cls] - probabilities[:, cls])
                logits[:, cls] += self.learning_rate * tree.predict(x)
                round_trees.append(tree)
            self.trees_.append(round_trees)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        logits = np.tile(self.base_logits_, (len(x), 1))
        for round_trees in self.trees_:
            for cls, tree in enumerate(round_trees):
                logits[:, cls] += self.learning_rate * tree.predict(x)
        return self._softmax(logits)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1).astype(np.int64)


def _make_model(name: str, *, seed: int, sklearn_modules: dict[str, Any] | None) -> tuple[Any, str, dict[str, Any]]:
    if sklearn_modules is not None:
        if name == "logistic_regression":
            model = sklearn_modules["make_pipeline"](
                sklearn_modules["StandardScaler"](),
                sklearn_modules["LogisticRegression"](
                    max_iter=1000,
                    multi_class="multinomial",
                    solver="lbfgs",
                    random_state=int(seed),
                ),
            )
            return model, "sklearn", {"class": "LogisticRegression", "max_iter": 1000, "solver": "lbfgs"}
        if name == "random_forest":
            model = sklearn_modules["RandomForestClassifier"](
                n_estimators=128,
                max_depth=6,
                min_samples_leaf=2,
                random_state=int(seed),
                n_jobs=1,
            )
            return model, "sklearn", {"class": "RandomForestClassifier", "n_estimators": 128, "max_depth": 6, "min_samples_leaf": 2}
        if name == "gradient_boosting":
            model = sklearn_modules["GradientBoostingClassifier"](
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                min_samples_leaf=2,
                random_state=int(seed),
            )
            return model, "sklearn", {"class": "GradientBoostingClassifier", "n_estimators": 100, "learning_rate": 0.05, "max_depth": 3}
    if name == "logistic_regression":
        model = _NumpyMultinomialLogistic(seed=seed)
        return model, model.implementation, {"class": "multinomial_softmax", "max_iter": 600, "l2": 1e-3}
    if name == "random_forest":
        model = _NumpyRandomForest(seed=seed)
        return model, model.implementation, {"class": "CART_ensemble", "n_estimators": 64, "max_depth": 5, "min_samples_leaf": 2}
    if name == "gradient_boosting":
        model = _NumpyGradientBoosting(seed=seed)
        return model, model.implementation, {"class": "multiclass_regression_tree_boosting", "n_estimators": 48, "learning_rate": 0.08, "max_depth": 2}
    raise ValueError(f"unknown shortcut model: {name!r}")


def _bootstrap_ci(values: Sequence[float], clusters: Sequence[str], *, seed: int, replicates: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    unique = list(dict.fromkeys(str(item) for item in clusters))
    by_cluster: dict[str, list[float]] = {cluster: [] for cluster in unique}
    for value, cluster in zip(values, clusters):
        by_cluster[str(cluster)].append(float(value))
    cluster_means = np.asarray([np.mean(by_cluster[cluster]) for cluster in unique], dtype=np.float64)
    point = float(np.mean(cluster_means)) if len(cluster_means) else None
    if point is None:
        return {"point": None, "lower": None, "upper": None, "method": "unavailable", "cluster_count": 0}
    if len(cluster_means) <= 1 or int(replicates) <= 1:
        return {"point": point, "lower": point, "upper": point, "method": "degenerate_question_cluster_bootstrap", "cluster_count": len(cluster_means)}
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(replicates), dtype=np.float64)
    for index in range(int(replicates)):
        draws[index] = float(np.mean(cluster_means[rng.integers(0, len(cluster_means), size=len(cluster_means))]))
    return {
        "point": point,
        "lower": float(np.percentile(draws, 2.5)),
        "upper": float(np.percentile(draws, 97.5)),
        "method": "question_cluster_bootstrap_percentile_95",
        "cluster_count": len(cluster_means),
        "replicates": int(replicates),
    }


def _metrics(records: Sequence[Mapping[str, Any]], predictions: np.ndarray, *, seed: int, bootstrap_replicates: int) -> dict[str, Any]:
    labels = np.asarray([_utility_label(record) for record in records], dtype=np.int64)
    regrets = []
    normalized_regrets = []
    selected_utilities = []
    best_utilities = []
    clusters = []
    families = []
    for index, (record, prediction) in enumerate(zip(records, predictions)):
        utilities = np.asarray([_finite(item) for item in record["branch_utilities"]], dtype=np.float64)
        best = float(np.max(utilities))
        chosen = float(utilities[int(prediction)])
        best_utilities.append(best)
        selected_utilities.append(chosen)
        regrets.append(best - chosen)
        utility_range = max(best - float(np.min(utilities)), 1e-8)
        normalized_regrets.append(max(0.0, (best - chosen) / utility_range))
        clusters.append(_record_question_id(record, index))
        families.append(_record_family(record))
    regrets_array = np.asarray(regrets, dtype=np.float64)
    normalized_regrets_array = np.asarray(normalized_regrets, dtype=np.float64)
    correct = (predictions == labels).astype(np.float64)
    grouped: dict[str, list[int]] = {}
    for index, cluster in enumerate(clusters):
        grouped.setdefault(cluster, []).append(index)
    cluster_accuracy = [float(np.mean(correct[indexes])) for indexes in grouped.values()]
    cluster_regret = [float(np.mean(regrets_array[indexes])) for indexes in grouped.values()]
    cluster_normalized_regret = [
        float(np.mean(normalized_regrets_array[indexes])) for indexes in grouped.values()
    ]
    family_metrics: dict[str, Any] = {}
    for family in sorted(set(families)):
        indexes = [index for index, value in enumerate(families) if value == family]
        family_clusters = [clusters[index] for index in indexes]
        family_metrics[family] = {
            "row_count": len(indexes),
            "question_cluster_count": len(set(family_clusters)),
            "action_accuracy": float(np.mean(correct[indexes])) if indexes else None,
            "mean_regret": float(np.mean(regrets_array[indexes])) if indexes else None,
            "normalized_regret": float(np.mean(normalized_regrets_array[indexes])) if indexes else None,
        }
    return {
        "row_count": len(records),
        "question_cluster_count": len(grouped),
        "action_accuracy": float(np.mean(correct)) if len(correct) else None,
        "question_macro_action_accuracy": float(np.mean(cluster_accuracy)) if cluster_accuracy else None,
        "mean_regret": float(np.mean(regrets_array)) if len(regrets_array) else None,
        "question_macro_regret": float(np.mean(cluster_regret)) if cluster_regret else None,
        "normalized_regret": float(np.mean(normalized_regrets_array)) if len(normalized_regrets_array) else None,
        "question_macro_normalized_regret": (
            float(np.mean(cluster_normalized_regret)) if cluster_normalized_regret else None
        ),
        "selected_utility": float(np.mean(selected_utilities)) if selected_utilities else None,
        "best_utility": float(np.mean(best_utilities)) if best_utilities else None,
        "regret_ci_question_cluster": _bootstrap_ci(regrets, clusters, seed=seed, replicates=bootstrap_replicates),
        "normalized_regret_ci_question_cluster": _bootstrap_ci(
            normalized_regrets,
            clusters,
            seed=int(seed) + 101,
            replicates=bootstrap_replicates,
        ),
        "best_action_distribution": {
            ACTION_NAMES[index]: int(np.sum(labels == index)) for index in range(len(ACTION_NAMES))
        },
        "predicted_action_distribution": {
            ACTION_NAMES[index]: int(np.sum(predictions == index)) for index in range(len(ACTION_NAMES))
        },
        "family_metrics": family_metrics,
    }


def _input_digest(dataset: str | Path | Mapping[str, Any]) -> str | None:
    if isinstance(dataset, Mapping):
        encoded = json.dumps(dataset, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()
    path = Path(dataset)
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def run_shortcut_probe(
    dataset: str | Path | Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    feature_sets: Sequence[str] = FEATURE_SETS,
    models: Sequence[str] = MODEL_NAMES,
    train_split: str = "train",
    eval_splits: Sequence[str] | None = None,
    seed: int = 0,
    bootstrap_replicates: int = 1000,
    strict_sklearn: bool = False,
    max_examples: int | None = None,
) -> dict[str, Any]:
    """Fit/evaluate shortcut probes and optionally write a machine-readable result."""

    payload = _load_payload(dataset)
    all_records = _records(payload)
    if max_examples is not None:
        limit = max(0, int(max_examples))
        all_records = all_records[:limit]
    result: dict[str, Any] = {
        "schema_version": SHORTCUT_PROBE_SCHEMA,
        "status": "initializing",
        "input_dataset_schema": payload.get("schema_version"),
        "input_dataset_digest": _input_digest(dataset),
        "config": {
            "feature_sets": [str(item) for item in feature_sets],
            "models": [str(item) for item in models],
            "train_split": str(train_split),
            "eval_splits": [str(item) for item in eval_splits] if eval_splits is not None else None,
            "seed": int(seed),
            "bootstrap_replicates": int(bootstrap_replicates),
            "strict_sklearn": bool(strict_sklearn),
            "max_examples": max_examples,
        },
        "dependency": {"numpy": np.__version__, "sklearn": sklearn_status()},
        "label_source": "argmax_evaluator_branch_utility_not_hidden_target_action",
        "feature_policy": "public_observation_numeric_only; optional_confirmation_summary_is_ablated_and_never_a_label",
        "sklearn_required_for_formal_claim": True,
        "formal_comparison_authorized": False,
        "fallback_results_diagnostic_only": True,
        "feature_sets": {},
        "models": {},
        "comparisons": {},
    }
    try:
        _check_records(all_records)
    except Exception as exc:
        result.update({"status": "fail_closed_invalid_dataset", "error": f"{type(exc).__name__}: {exc}"})
        return _write_result(result, output_dir)
    sklearn_modules, sklearn_error = _sklearn_import()
    result["dependency"]["sklearn"]["import_error"] = sklearn_error
    if strict_sklearn and sklearn_modules is None:
        result.update({
            "status": "fail_closed_sklearn_unavailable",
            "error": sklearn_error or "sklearn import failed",
            "fallback_used": False,
        })
        return _write_result(result, output_dir)
    available_splits = tuple(dict.fromkeys(str(record.get("split", train_split)) for record in all_records))
    requested_splits = tuple(str(item) for item in eval_splits) if eval_splits is not None else tuple(split for split in available_splits if split != train_split)
    train_records = [record for record in all_records if str(record.get("split", train_split)) == str(train_split)]
    if not train_records:
        result.update({"status": "fail_closed_missing_train_split", "available_splits": list(available_splits), "fallback_used": False})
        return _write_result(result, output_dir)
    train_clusters = {_record_question_id(record, index) for index, record in enumerate(train_records)}
    overlap_by_split = {
        split: sorted({
            _record_question_id(record, index)
            for index, record in enumerate(all_records)
            if str(record.get("split", train_split)) == split
        }.intersection(train_clusters))
        for split in requested_splits
    }
    result["split_cluster_overlap_with_train"] = {
        split: {"count": len(clusters), "question_ids_hidden_audit": clusters[:20]}
        for split, clusters in overlap_by_split.items()
    }
    result["split_boundary_pass"] = all(not clusters for clusters in overlap_by_split.values())
    if not result["split_boundary_pass"]:
        result.update({
            "status": "fail_closed_split_cluster_overlap",
            "error": "one or more evaluation splits share question clusters with train",
            "fallback_used": False,
        })
        return _write_result(result, output_dir)
    # Freeze the feature schema from the training split only.  Looking at held-out
    # receipt names (even without their values) would make the evaluation boundary
    # depend on future data.  Unknown held-out keys are ignored and recorded below.
    available_raw_keys = sorted({
        str(key)
        for record in train_records
        for key in _raw_map(_mapping(record.get("observation"))).keys()
    })
    all_raw_keys = sorted({
        str(key)
        for record in all_records
        for key in _raw_map(_mapping(record.get("observation"))).keys()
    })
    result["available_raw_feature_names"] = [
        name for name in _RAW_FEATURE_NAMES if name in set(available_raw_keys)
    ]
    result["raw_feature_schema"] = {
        "registered_names": list(_RAW_FEATURE_NAMES),
        "available_names": list(result["available_raw_feature_names"]),
        "omitted_names": [name for name in _RAW_FEATURE_NAMES if name not in set(available_raw_keys)],
        "confirmation_summary_present": "log_confirmation_pass_rate" in set(available_raw_keys),
        "heldout_only_names": [name for name in all_raw_keys if name not in set(available_raw_keys)],
        "schema_frozen_from_split": str(train_split),
    }
    train_labels = np.asarray([_utility_label(record) for record in train_records], dtype=np.int64)
    fallback_used = sklearn_modules is None
    for requested_feature_set in feature_sets:
        current_feature_set = _normalize_feature_set(str(requested_feature_set))
        names = feature_names(current_feature_set, available_raw_keys=available_raw_keys)
        result["feature_sets"][current_feature_set] = {"feature_names": list(names)}
        x_train = np.asarray([
            observation_features(_mapping(record["observation"]), current_feature_set, available_raw_keys=available_raw_keys)
            for record in train_records
        ])
        result["feature_sets"][current_feature_set]["feature_count"] = int(len(names))
        result["feature_sets"][current_feature_set]["train_row_count"] = int(len(train_records))
        result["feature_sets"][current_feature_set]["confirmation_feature_included"] = "log_confirmation_pass_rate" in names
        for model_name in models:
            normalized_model = str(model_name)
            model_key = f"{current_feature_set}:{normalized_model}"
            model_record: dict[str, Any] = {
                "status": "initializing",
                "feature_set": current_feature_set,
                "model": normalized_model,
                "label_source": result["label_source"],
            }
            result["models"][model_key] = model_record
            try:
                model, implementation, params = _make_model(normalized_model, seed=int(seed), sklearn_modules=sklearn_modules)
                model.fit(x_train, train_labels)
                model_record.update({"implementation": implementation, "parameters": params, "train_row_count": len(train_records)})
                metrics_by_split: dict[str, Any] = {}
                for split in requested_splits:
                    eval_records = [record for record in all_records if str(record.get("split", train_split)) == split]
                    if not eval_records:
                        metrics_by_split[split] = {"status": "not_available", "row_count": 0}
                        continue
                    x_eval = np.asarray([
                        observation_features(_mapping(record["observation"]), current_feature_set, available_raw_keys=available_raw_keys)
                        for record in eval_records
                    ])
                    predictions = np.asarray(model.predict(x_eval), dtype=np.int64)
                    # sklearn can omit an unseen class in predict_proba, but predict
                    # itself always returns valid training classes; normalize bounds.
                    if np.any(predictions < 0) or np.any(predictions >= len(ACTION_NAMES)):
                        raise ValueError("model emitted an action outside the registered four-action set")
                    metrics_by_split[split] = {"status": "completed", **_metrics(eval_records, predictions, seed=int(seed) + len(split), bootstrap_replicates=int(bootstrap_replicates))}
                model_record.update({"status": "completed", "metrics_by_split": metrics_by_split})
            except Exception as exc:
                model_record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    result["available_splits"] = list(available_splits)
    result["eval_splits"] = list(requested_splits)
    result["fallback_used"] = bool(fallback_used)
    result["status"] = "completed_numpy_fallback" if fallback_used else "completed_sklearn"
    # Feature-ablation deltas make the potential confirmation shortcut explicit.
    for model_name in models:
        model_comparison: dict[str, Any] = {}
        all_key = f"all_raw:{model_name}"
        no_key = f"without_confirmation:{model_name}"
        all_metrics = result["models"].get(all_key, {}).get("metrics_by_split", {})
        no_metrics = result["models"].get(no_key, {}).get("metrics_by_split", {})
        for split in requested_splits:
            left, right = all_metrics.get(split, {}), no_metrics.get(split, {})
            if left.get("status") != "completed" or right.get("status") != "completed":
                continue
            model_comparison[split] = {
                "without_confirmation_minus_all_regret": float(right["mean_regret"] - left["mean_regret"]),
                "without_confirmation_minus_all_action_accuracy": float(right["action_accuracy"] - left["action_accuracy"]),
            }
        result["comparisons"][str(model_name)] = model_comparison
    return _write_result(result, output_dir)


def _write_result(result: dict[str, Any], output_dir: str | Path | None) -> dict[str, Any]:
    if output_dir is None:
        return result
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "shortcut_probe_result.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    result["result_path"] = str(path)
    return result


# Descriptive aliases keep the public evaluation interface easy to discover while
# retaining the singular implementation name used by the runner.
run_shortcut_probes = run_shortcut_probe
extract_shortcut_features = observation_features


__all__ = [
    "ACTION_NAMES",
    "FEATURE_SETS",
    "MODEL_NAMES",
    "SHORTCUT_PROBE_SCHEMA",
    "feature_names",
    "observation_features",
    "extract_shortcut_features",
    "run_shortcut_probe",
    "run_shortcut_probes",
    "sklearn_status",
]
