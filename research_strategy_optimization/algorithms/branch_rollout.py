"""Same-state counterfactual branch execution.

Every candidate option is restored from one immutable environment snapshot.  The
manager deliberately keeps the trusted verdict on the evaluator side and returns
branch records only after all branches have been executed, which prevents the
first option from changing the state seen by later options.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from ..schemas import BranchRecord, ExperimentOutput, ResearchAction, Trajectory, Verdict
from ..environments.snapshot_manager import SnapshotManager
from .leave_one_out_advantage import assign_leave_one_out_advantages


UtilityFn = Callable[[Any, Any, Any], float]
VerifierFn = Callable[[Any, Any], Any]


@dataclass(frozen=True)
class BranchExecution:
    """Result of one option restored from a common snapshot.

    ``record`` is a :class:`BranchRecord` when observations are available; the
    raw output/verdict fields are retained so lightweight test environments do
    not need to implement the complete trajectory schema.
    """

    option: ResearchAction
    output: Any
    verdict: Any
    utility: float
    snapshot_digest: str
    final_snapshot_digest: str
    record: Optional[BranchRecord] = None
    seed_values: tuple[float, ...] = ()

    @property
    def advantage(self) -> float:
        return float(self.record.advantage) if self.record is not None else 0.0

    @property
    def trajectory(self) -> Optional[Trajectory]:
        """Compatibility view used by the tabular trainer."""

        return self.record.trajectory if self.record is not None else None

    @property
    def components(self) -> dict[str, float]:
        return dict(self.record.components) if self.record is not None else {"utility": self.utility}

    def to_dict(self) -> dict[str, Any]:
        return {
            "option": self.option.value,
            "utility": self.utility,
            "advantage": self.advantage,
            "snapshot_digest": self.snapshot_digest,
            "final_snapshot_digest": self.final_snapshot_digest,
            "output": self.output.public_dict() if hasattr(self.output, "public_dict") else self.output,
            "verdict": self.verdict.to_dict() if hasattr(self.verdict, "to_dict") else self.verdict,
        }


def _snapshot_digest(snapshot: Any) -> str:
    digest = getattr(snapshot, "digest", None)
    if digest:
        return str(digest)
    if isinstance(snapshot, Mapping):
        return SnapshotManager.create(snapshot).digest
    raise TypeError("snapshot must expose a digest or be a mapping")


def _restore(clone: Any, snapshot: Any) -> Any:
    """Restore a clone and return it, supporting both in-place and pure APIs."""

    restored = clone.restore(snapshot)
    return clone if restored is None else restored


def _clone_from_snapshot(environment: Any, snapshot: Any) -> Any:
    if hasattr(environment, "clone_from_snapshot"):
        clone = environment.clone_from_snapshot(snapshot)
    else:
        clone = copy.deepcopy(environment)
        clone = _restore(clone, snapshot)
    return clone


def _finite(value: Any, name: str = "utility") -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


class BranchRolloutManager:
    """Execute options from one frozen state with common seeds."""

    def __init__(
        self,
        environment: Any = None,
        verifier: Any = None,
        utility_fn: Optional[UtilityFn] = None,
    ) -> None:
        # ``environment`` is the preferred first argument.  For compatibility
        # with early trainer drafts, a verifier plus a seed tuple may be passed
        # as ``BranchRolloutManager(verifier, seeds)``; the environment is then
        # supplied to ``execute_paired_options``.
        self.default_seeds: tuple[int, ...] = ()
        if isinstance(verifier, Sequence) and not isinstance(verifier, (str, bytes, Mapping)):
            try:
                self.default_seeds = tuple(int(seed) for seed in verifier)
                verifier = None
            except (TypeError, ValueError):
                pass
        if (
            environment is not None
            and not hasattr(environment, "snapshot")
            and (hasattr(environment, "evaluate") or callable(environment))
            and verifier is None
        ):
            verifier = environment
            environment = None
        self.environment = environment
        self.verifier = verifier
        self.utility_fn = utility_fn

    def create_snapshot(self, environment: Any = None) -> Any:
        env = environment or self.environment
        snapshot = env.snapshot()
        # Validate immediately.  A malformed snapshot must never become a
        # counterfactual training example.
        _snapshot_digest(snapshot)
        if hasattr(snapshot, "payload"):
            SnapshotManager.restore(snapshot)
        return snapshot

    def _evaluate(self, output: Any, branch_env: Any, verifier: Any) -> Any:
        evaluator = verifier if verifier is not None else self.verifier
        if evaluator is None:
            return None
        if hasattr(evaluator, "evaluate"):
            return evaluator.evaluate(output, branch_env)
        if callable(evaluator):
            return evaluator(output, branch_env)
        raise TypeError("verifier must expose evaluate() or be callable")

    def _utility(
        self,
        output: Any,
        verdict: Any,
        branch_env: Any,
        utility_fn: Optional[UtilityFn],
    ) -> float:
        fn = utility_fn or self.utility_fn
        if fn is not None:
            return _finite(fn(output, verdict, branch_env))
        # Default to a validity-gated scientific utility rather than the raw surface
        # metric.  This is the critical safety invariant: an invalid high score cannot
        # win a branch comparison.  Adapters may provide a richer registered task
        # utility through ``utility_fn``.
        if isinstance(verdict, Verdict):
            # A generic rollout manager must not turn the four evidence labels into a
            # hidden ``state -> preferred action`` oracle.  Tier-1 callers provide an
            # explicit evaluator-owned utility callback; this fallback is deliberately
            # action-agnostic and rewards only observed protocol transitions,
            # confirmation, precision, and cost.
            if not verdict.validity_pass:
                return _finite(-0.30 - 0.05 * float(getattr(output, "execution_cost", 0.0)))
            signals = set(getattr(output, "validity_signals", ()))
            task = 0.20
            if "split_protocol_updated" in signals:
                task += 0.25
            if "group_held_out_split" in signals and "split_overlap_diagnostic" not in signals:
                task += 0.35
            if any(
                token in signal
                for signal in signals
                for token in ("adjusted", "controlled", "subgroup_metric_estimator")
            ):
                task += 0.30
            if "sample_count_below_precision_target" not in signals and int(getattr(output, "sample_size", 0)) >= 60:
                task += 0.10
            confirmation = 0.10 if verdict.independent_confirmation_passed else 0.0
            cost = float(getattr(output, "execution_cost", 0.0))
            return _finite(task + confirmation - 0.05 * cost)
        if verdict is not None:
            for key in ("utility", "quality_score", "vds_score"):
                if hasattr(verdict, key):
                    return _finite(getattr(verdict, key))
                if isinstance(verdict, Mapping) and key in verdict:
                    return _finite(verdict[key])
        if isinstance(output, Mapping):
            for key in ("utility", "quality", "score", "effect_estimate"):
                if key in output:
                    return _finite(output[key])
        # Tier-0's trusted output intentionally keeps the scientific effect in
        # the raw record while the compact Verdict has no utility field.  Use
        # the effect as a transparent fallback, but never reward an invalid
        # branch as if it were a valid scientific claim.
        if verdict is not None:
            state = getattr(getattr(verdict, "evidence_state", None), "value", None)
            if state == "invalid" or getattr(verdict, "validity_pass", True) is False:
                return 0.0
        if hasattr(output, "effect_estimate"):
            return _finite(getattr(output, "effect_estimate"))
        return 0.0

    def _make_record(
        self,
        option: ResearchAction,
        branch_env: Any,
        initial_observation: Any,
        final_observation: Any,
        output: Any,
        verdict: Any,
        utility: float,
        branch_id: str,
    ) -> Optional[BranchRecord]:
        if initial_observation is None or final_observation is None:
            return None
        question_id = str(getattr(branch_env, "question_id", ""))
        world_id = str(getattr(getattr(branch_env, "world", None), "world_id", "hidden"))
        outputs = [output] if isinstance(output, ExperimentOutput) else []
        verdicts = [verdict] if isinstance(verdict, Verdict) else []
        trajectory = Trajectory(
            question_id=question_id,
            world_id=world_id,
            initial_observation=initial_observation,
            final_observation=final_observation,
            outputs=outputs,
            verdicts=verdicts,
            total_cost=float(getattr(output, "execution_cost", 0.0)),
            branch_id=branch_id,
        )
        components = {"utility": utility}
        return BranchRecord(option=option, trajectory=trajectory, utility=utility, components=components)

    def execute_paired_options(
        self,
        snapshot_or_environment: Any = None,
        options: Iterable[ResearchAction] = (),
        seeds: Optional[Sequence[int]] = None,
        *,
        snapshot: Any = None,
        environment: Any = None,
        verifier: Any = None,
        utility_fn: Optional[UtilityFn] = None,
        normalize_advantages: bool = False,
        require_equivalent_snapshot: bool = True,
    ) -> list[BranchExecution]:
        """Run each option from ``snapshot`` using the same preregistered seeds."""

        # Accept both modern ``(snapshot, options, seeds)`` and the temporary
        # trainer spelling ``(environment, options, snapshot=..., seeds=...)``.
        if snapshot is not None:
            environment = snapshot_or_environment if snapshot_or_environment is not None else environment
            source_snapshot = snapshot
        else:
            source_snapshot = snapshot_or_environment
        if environment is None:
            environment = self.environment
        if environment is None:
            raise ValueError("an environment is required for paired rollout")
        options = [ResearchAction(option) for option in options]
        if not options:
            return []
        seeds = tuple(int(seed) for seed in (seeds or self.default_seeds))
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("paired rollout seeds must be nonempty and unique")
        source_digest = _snapshot_digest(source_snapshot)
        results: list[BranchExecution] = []
        for index, option in enumerate(options):
            branch_env = _clone_from_snapshot(environment, source_snapshot)
            branch_snapshot = self.create_snapshot(branch_env)
            branch_digest = _snapshot_digest(branch_snapshot)
            if require_equivalent_snapshot and branch_digest != source_digest:
                raise ValueError("restored branch does not match source snapshot")
            initial = branch_env.visible_observation() if hasattr(branch_env, "visible_observation") else None
            # Expose the public decision state to an evaluator-side utility callback.
            # This is deliberately not part of ``Observation.to_dict`` or policy
            # inputs; it lets a transition-aware utility score an action against the
            # state from which it was selected (e.g. switch after a refutation).
            if initial is not None:
                try:
                    setattr(branch_env, "_branch_initial_observation", initial)
                except Exception:
                    pass
            output = branch_env.execute_option(option, seeds=seeds)
            verdict = self._evaluate(output, branch_env, verifier)
            utility = self._utility(output, verdict, branch_env, utility_fn)
            final = branch_env.visible_observation() if hasattr(branch_env, "visible_observation") else None
            final_snapshot = self.create_snapshot(branch_env)
            record = self._make_record(
                option, branch_env, initial, final, output, verdict, utility, f"branch_{index:04d}"
            )
            results.append(
                BranchExecution(
                    option=option,
                    output=output,
                    verdict=verdict,
                    utility=utility,
                    snapshot_digest=source_digest,
                    final_snapshot_digest=_snapshot_digest(final_snapshot),
                    record=record,
                    seed_values=tuple(_seed_values(output)),
                )
            )
        # Keep the estimand explicit; assigning advantages never changes the
        # environment or the trusted verdicts.
        records = [result.record for result in results]
        if all(record is not None for record in records):
            assign_leave_one_out_advantages([record for record in records if record is not None], normalize=normalize_advantages)
        return results

    def estimate_paired_effects(self, branch_results: Sequence[Any]) -> dict[tuple[str, str], float]:
        """Return ordered pairwise utility differences ``G_i - G_j``."""

        values = []
        for result in branch_results:
            option = getattr(result, "option", None)
            if option is None and isinstance(result, Mapping):
                option = result.get("option")
            label = option.value if isinstance(option, ResearchAction) else str(option)
            utility = getattr(result, "utility", None)
            if utility is None and isinstance(result, Mapping):
                utility = result.get("utility")
            values.append((label, _finite(utility)))
        effects: dict[tuple[str, str], float] = {}
        for left, left_value in values:
            for right, right_value in values:
                if left != right:
                    effects[(left, right)] = left_value - right_value
        return effects


def _seed_values(output: Any) -> tuple[float, ...]:
    """Extract optional per-seed values for paired-noise diagnostics."""

    if isinstance(output, Mapping):
        values = output.get("seed_values", output.get("paired_seed_values", ()))
    else:
        values = getattr(output, "seed_values", getattr(output, "paired_seed_values", ()))
    if values is None:
        return ()
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        return ()
    return result if all(math.isfinite(value) for value in result) else ()


__all__ = ["BranchExecution", "BranchRolloutManager"]
