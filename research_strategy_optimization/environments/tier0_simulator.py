"""A deterministic, executable paired-world research environment.

The environment is deliberately small but captures the mechanisms that the plan says
must be validated before model training: hidden world mechanisms, dynamic evidence
states, leakage/invalidity precedence, finite budgets, reproducible snapshots, common
random-number seeds, and independent confirmation on held-out seeds.
"""

from __future__ import annotations

import copy
import hashlib
import math
import random
from dataclasses import asdict
from types import MappingProxyType
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..schemas import (
    EvidenceState,
    ExperimentOutput,
    Observation,
    HypothesisBelief,
    Protocol,
    ResearchAction,
    Verdict,
    WorldSpec,
)
from ..evidence.evidence_classifier import classify_evidence
from ..evidence.interval_rules import normal_ci
from .abstract_research_env import ResearchEnvironment
from .budget_tracker import BudgetTracker
from .snapshot_manager import SnapshotManager, canonical_hash


def default_mvp_worlds() -> Tuple[WorldSpec, ...]:
    """Return the four frozen worlds required by the plan's MVP.

    The public task description is identical across worlds; only the hidden mechanism
    differs.  Effects are chosen so the scientifically preferred action is respectively
    continue, switch, sample, and repair.
    """

    return (
        WorldSpec("world_01", "supported", 0.100, 0.025, 0.10, 120),
        WorldSpec("world_02", "refuted", 0.000, 0.090, 0.10, 120),
        # The effect is real but the initial sample is too noisy; ``sample`` resolves
        # the interval into Supported, exercising the dynamic Insufficient transition.
        WorldSpec("world_03", "insufficient", 0.080, 0.020, 0.32, 18),
        WorldSpec("world_04", "invalid", 0.000, 0.060, 0.10, 120, leakage=True),
    )


class Tier0ResearchEnvironment(ResearchEnvironment):
    """A small statistical experiment with method-A and method-B alternatives."""

    world_id_hidden = True
    # Exposed only as implementation metadata on evaluator outputs.  The policy
    # still receives the same Observation schema irrespective of backend.
    BACKEND = "tier0_python"
    SAMPLE_MINIMUM = 288

    QUESTION_TEXT = "Determine whether method A improves group-held-out performance."

    def __init__(
        self,
        worlds: Optional[Iterable[WorldSpec]] = None,
        protocol: Optional[Protocol] = None,
        budget: Optional[int] = None,
    ) -> None:
        world_map = {w.world_id: copy.deepcopy(w) for w in (worlds or default_mvp_worlds())}
        # The frozen manifest is evaluator-owned.  A mapping proxy prevents an online
        # policy/executor from mutating mechanisms after the protocol has been bound.
        self.worlds = MappingProxyType(world_map)
        self.protocol = protocol or Protocol()
        self._budget_limit = int(budget if budget is not None else self.protocol.max_budget)
        self._world: Optional[WorldSpec] = None
        self._question_id = ""
        self._seed = 0
        self._turn = 0
        self._current_method = "method_a"
        self._active_hypothesis_id = "H_A"
        self._hypothesis_beliefs: Dict[str, float] = {"H_A": 0.5, "H_B": 0.5}
        self._belief_turns: Dict[str, int] = {"H_A": 0, "H_B": 0}
        self._sample_size = 0
        self._seed_count = 0
        self._repaired = False
        self._history: List[str] = []
        self._last_output: Optional[ExperimentOutput] = None
        self._budget = BudgetTracker.create(self._budget_limit)

    @property
    def world(self) -> WorldSpec:
        if self._world is None:
            raise RuntimeError("environment has not been reset")
        return self._world

    @property
    def question_id(self) -> str:
        return self._question_id

    def reset(self, question_id: str = "rq_mvp_001", world_id: str = "world_01", seed: int = 0) -> Observation:
        if world_id not in self.worlds:
            raise KeyError(world_id)
        self._world = copy.deepcopy(self.worlds[world_id])
        self._question_id = question_id
        self._seed = int(seed)
        self._turn = 0
        self._current_method = "method_a"
        self._active_hypothesis_id = "H_A"
        self._hypothesis_beliefs = {"H_A": 0.5, "H_B": 0.5}
        self._belief_turns = {"H_A": 0, "H_B": 0}
        self._sample_size = self.world.initial_samples
        self._seed_count = 1
        self._repaired = False
        self._history = ["registered_hypothesis: method A has positive practical effect"]
        self._last_output = None
        self._budget = BudgetTracker.create(self._budget_limit)
        return self.visible_observation()

    def visible_observation(self) -> Observation:
        # The initial observation intentionally reports only public experimental facts.
        # No world ID, latent effect, trusted label, or confirmation status is present.
        if self._last_output is None:
            effect = 0.0
            low, high = -0.10, 0.10
            n = self._sample_size
            seeds = self._seed_count
            signals: Tuple[str, ...] = ("initial_experiment_pending",)
        else:
            effect = self._last_output.effect_estimate
            low, high = self._last_output.ci_low, self._last_output.ci_high
            n = self._last_output.sample_size
            seeds = self._last_output.seed_count
            signals = self._last_output.validity_signals
        return Observation(
            question_id=self._question_id,
            turn=self._turn,
            current_method=self._current_method,
            effect_estimate=effect,
            ci_low=low,
            ci_high=high,
            sample_size=n,
            seed_count=seeds,
            remaining_budget=self._budget.remaining,
            validity_signals=signals,
            history_summary=tuple(self._history[-6:]),
            hypothesis_probability=self._public_belief(),
            active_hypothesis_id=self._active_hypothesis_id,
            hypothesis_beliefs=tuple(
                HypothesisBelief(
                    hypothesis_id=hypothesis_id,
                    probability=probability,
                    turn=self._belief_turns.get(hypothesis_id, self._turn),
                    committed_before_action=True,
                )
                for hypothesis_id, probability in sorted(self._hypothesis_beliefs.items())
            ),
            task_family=(self.world.public_task_family or self.world.question_family),
        )

    def _public_belief(self) -> float:
        """Return the belief for the active hypothesis only.

        Beliefs for method A and method B are kept separately.  In particular, an
        observation after switching to method B cannot overwrite the probability for
        H_A and then be scored against H_A's truth value.
        """

        return float(self._hypothesis_beliefs.get(self._active_hypothesis_id, 0.5))

    @staticmethod
    def _hypothesis_for_method(method: str) -> str:
        return "H_B" if str(method) == "method_b" else "H_A"

    def _update_public_belief(self, output: ExperimentOutput) -> None:
        """Update only the hypothesis supported by a valid, observable result.

        Invalid/leaky/confounded outputs remain visible as diagnostics, but cannot
        create a high-confidence belief that a later repair can harvest as a reward.
        The trusted verifier still makes the final validity decision independently.
        """

        hypothesis_id = self._hypothesis_for_method(output.method)
        self._active_hypothesis_id = hypothesis_id
        self._belief_turns[hypothesis_id] = self._turn
        invalid_surface = bool(
            output.leakage
            or output.confounding
            or any(
                signal in set(output.validity_signals)
                for signal in ("split_overlap_diagnostic", "treatment_confounder_dependence")
            )
        )
        if invalid_surface:
            return
        width = max(1e-6, output.ci_high - output.ci_low)
        score = output.effect_estimate / (0.5 * width + 0.02)
        probability = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, score))))
        self._hypothesis_beliefs[hypothesis_id] = float(max(0.001, min(0.999, probability)))

    def snapshot(self):
        payload = {
            "question_id": self._question_id,
            "world_id_internal": self.world.world_id,
            "seed": self._seed,
            "turn": self._turn,
            "current_method": self._current_method,
            "active_hypothesis_id": self._active_hypothesis_id,
            "hypothesis_beliefs": dict(self._hypothesis_beliefs),
            "belief_turns": dict(self._belief_turns),
            "sample_size": self._sample_size,
            "seed_count": self._seed_count,
            "repaired": self._repaired,
            "history": list(self._history),
            "last_output": asdict(self._last_output) if self._last_output is not None else None,
            "budget": self._budget.state(),
        }
        return SnapshotManager.create(payload)

    def restore(self, snapshot) -> None:
        payload = SnapshotManager.restore(snapshot)
        if payload.get("world_id_internal") not in self.worlds:
            raise ValueError("snapshot world is not registered")
        # Restoring a snapshot may only happen within the same hidden world.  The check
        # prevents accidental cross-world counterfactuals.
        if self._world is not None and payload["world_id_internal"] != self.world.world_id:
            raise ValueError("cannot restore a snapshot from another world")
        self._question_id = str(payload["question_id"])
        self._seed = int(payload["seed"])
        self._turn = int(payload["turn"])
        self._current_method = str(payload["current_method"])
        self._active_hypothesis_id = str(payload.get("active_hypothesis_id", self._hypothesis_for_method(self._current_method)))
        self._hypothesis_beliefs = {
            str(key): float(value)
            for key, value in dict(payload.get("hypothesis_beliefs", {"H_A": 0.5, "H_B": 0.5})).items()
        }
        self._belief_turns = {
            str(key): int(value)
            for key, value in dict(payload.get("belief_turns", {})).items()
        }
        self._hypothesis_beliefs.setdefault("H_A", 0.5)
        self._hypothesis_beliefs.setdefault("H_B", 0.5)
        self._belief_turns.setdefault("H_A", self._turn)
        self._belief_turns.setdefault("H_B", self._turn)
        self._sample_size = int(payload["sample_size"])
        self._seed_count = int(payload["seed_count"])
        self._repaired = bool(payload["repaired"])
        self._history = list(payload["history"])
        raw = payload.get("last_output")
        self._last_output = ExperimentOutput(**raw) if raw is not None else None
        budget = payload["budget"]
        self._budget = BudgetTracker(int(budget["initial"]), int(budget["remaining"]), int(budget["spent"]))

    def clone_from_snapshot(self, snapshot):
        # Preserve the concrete environment class.  A Tier-1 environment inherits
        # the snapshot contract, but its clone must continue to execute the NumPy
        # backend rather than silently falling back to this Tier-0 simulator.
        clone = type(self)(self.worlds.values(), self.protocol, self._budget_limit)
        clone._world = copy.deepcopy(self.world)
        clone.restore(snapshot)
        return clone

    def remaining_budget(self) -> int:
        return self._budget.remaining

    def execute_option(
        self,
        option: ResearchAction,
        seeds: Optional[Sequence[int]] = None,
        confirmation: bool = False,
    ) -> ExperimentOutput:
        if self._world is None:
            raise RuntimeError("reset must be called before execute_option")
        if self._budget.remaining <= 0 and not confirmation:
            raise RuntimeError("research budget exhausted")
        option = ResearchAction(option)
        seeds = tuple(int(s) for s in (seeds or self.protocol.exploration_seeds[:2]))
        if not seeds:
            raise ValueError("at least one seed is required")

        # A branch consumes one high-level budget unit.  Confirmation runs are performed
        # in a cloned environment and do not alter the exploration budget.
        if not confirmation:
            self._budget.consume(1)
        before_method = self._current_method
        if option is ResearchAction.SWITCH:
            self._current_method = "method_b"
        elif option is ResearchAction.REPAIR:
            self._repaired = True
        elif option is ResearchAction.SAMPLE:
            self._sample_size = max(self._sample_size * 16, int(self.SAMPLE_MINIMUM))
            self._seed_count = max(self._seed_count, len(seeds))
        elif option is ResearchAction.REPLICATE:
            self._seed_count += len(seeds)
        elif option is ResearchAction.STOP:
            self._history.append("stop_and_report")
            return self._empty_output(option, seeds, confirmation)

        method = self._current_method
        if option is ResearchAction.CONTINUE:
            method = before_method
        output = self._simulate(method=method, option=option, seeds=seeds, confirmation=confirmation)
        self._last_output = output
        self._turn += 1
        if not confirmation:
            self._update_public_belief(output)
        self._history.append(f"{option.value}:{method}")
        return output

    def _empty_output(self, option: ResearchAction, seeds: Sequence[int], confirmation: bool) -> ExperimentOutput:
        return ExperimentOutput(
            action=option.value,
            method=self._current_method,
            effect_estimate=0.0,
            ci_low=-1.0,
            ci_high=1.0,
            sample_size=self._sample_size,
            seed_count=len(seeds),
            execution_cost=0.0,
            dataset_hash=self._hash(f"dataset:{'confirmation' if confirmation else 'exploration'}:empty:{self._sample_size}:{tuple(seeds)}"),
            code_hash=self._hash("code:" + self._current_method),
            split_hash=self._hash(
                "split:confirmation_hidden"
                if confirmation
                else ("split:repaired" if self._repaired else "split:initial")
            ),
            evaluator_hash=self._hash("trusted-evaluator-v1"),
            seeds=tuple(seeds),
            validity_signals=("no_experiment_executed",),
            hidden_world_id=self.world.world_id,
            latent_effect=self.world.true_effect_a if self._current_method == "method_a" else self.world.true_effect_b,
            leakage=False,
            confounding=False,
            confirmation=confirmation,
            backend=self.BACKEND,
            estimator="tier0_closed_form",
            data_partition="confirmation_hidden" if confirmation else ("repaired" if self._repaired else "random"),
        )

    def _simulate(self, method: str, option: ResearchAction, seeds: Sequence[int], confirmation: bool) -> ExperimentOutput:
        world = self.world
        latent = world.true_effect_a if method == "method_a" else world.true_effect_b
        # Sample size and explicit common seeds control the confidence width.  The random
        # noise is independent of world IDs and reproducible from seed+world offset.
        values: List[float] = []
        for seed in seeds:
            rng = random.Random(int(seed) + world.seed_offset + (10000 if confirmation else 0))
            sigma = world.noise_scale / math.sqrt(max(1, self._sample_size))
            observed = latent + rng.gauss(0.0, sigma)
            values.append(observed)
        effect, low, high = normal_ci(values, self.protocol.confidence_level)
        if self._sample_size < 60:
            # Small-sample pilot intervals are intentionally conservative.  This makes
            # the Insufficient world reachable without relying on a lucky random draw;
            # the same action after ``sample`` becomes precise enough to resolve it.
            half_width = max(high - effect, 0.08)
            low, high = effect - half_width, effect + half_width
        leakage_active = bool(world.leakage and not self._repaired and method == "method_a")
        if leakage_active:
            # Deliberately attractive but invalid surface result.
            effect += 0.20
            low += 0.20
            high += 0.20
        signals: List[str] = []
        if self._sample_size < 60:
            signals.append("sample_count_below_precision_target")
        if leakage_active:
            signals.append("split_overlap_diagnostic")
        if self._repaired:
            signals.append("split_protocol_updated")
        if method == "method_b":
            signals.append("alternative_method_evaluated")
        return ExperimentOutput(
            action=option.value,
            method=method,
            effect_estimate=effect,
            ci_low=low,
            ci_high=high,
            sample_size=self._sample_size,
            seed_count=len(seeds),
            execution_cost=1.0 + 0.001 * self._sample_size,
            # Confirmation uses a distinct hidden partition namespace.  The random
            # seed offset alone is not sufficient evidence of independent data; the
            # resulting dataset and split hashes must differ from exploration.
            dataset_hash=self._hash(
                f"dataset:{'confirmation' if confirmation else 'exploration'}:{self._sample_size}:{tuple(seeds)}"
            ),
            code_hash=self._hash(f"code:{method}"),
            split_hash=self._hash(
                "split:confirmation_hidden"
                if confirmation
                else ("split:repaired" if self._repaired else "split:random")
            ),
            evaluator_hash=self._hash("trusted-evaluator-v1"),
            seeds=tuple(seeds),
            validity_signals=tuple(signals),
            hidden_world_id=world.world_id,
            latent_effect=latent,
            leakage=leakage_active,
            confounding=world.confounding and not self._repaired,
            confirmation=confirmation,
            backend=self.BACKEND,
            estimator="tier0_closed_form",
            data_partition="confirmation_hidden" if confirmation else ("repaired" if self._repaired else "random"),
        )

    def _hash(self, value: str) -> str:
        return "sha256:" + hashlib.sha256(value.encode()).hexdigest()

    def hidden_truth(self, method: str = "method_a") -> int:
        latent = self.world.true_effect_a if method == "method_a" else self.world.true_effect_b
        return int(latent > self.protocol.delta_min)

    def final_submission(self, claim: Mapping[str, object]) -> Verdict:
        if self._last_output is None:
            raise RuntimeError("no experiment to submit")
        return TrustedVerifier(self.protocol).evaluate(self._last_output, self)

    def build_verifier(self) -> "TrustedVerifier":
        return TrustedVerifier(self.protocol)


class TrustedVerifier:
    """Independent verifier implementation.

    It consumes raw output plus a private environment reference, checks immutable hashes,
    applies invalid-precedence state rules, and performs confirmation on distinct seeds.
    """

    def __init__(self, protocol: Optional[Protocol] = None):
        self.protocol = protocol or Protocol()
        self.version = "trusted_verifier_v1"
        self.immutable = True

    # The following plan-shaped methods are intentionally thin aliases around the
    # immutable ``evaluate`` path.  They make the verifier usable through the abstract
    # Tier-1/Tier-2 interface without creating a second, potentially divergent ruleset.
    def assess_validity(self, output: ExperimentOutput, env: Tier0ResearchEnvironment) -> bool:
        return bool(self.evaluate(output, env, confirm=False).validity_pass)

    def classify_evidence(
        self,
        output: ExperimentOutput,
        env: Tier0ResearchEnvironment,
        protocol: Optional[Protocol] = None,
    ) -> Verdict:
        verifier = self if protocol is None or protocol == self.protocol else TrustedVerifier(protocol)
        return verifier.evaluate(output, env, confirm=False)

    def confirm_independently(self, candidate: tuple[ExperimentOutput, Tier0ResearchEnvironment] | Mapping[str, object]) -> Verdict:
        if isinstance(candidate, tuple) and len(candidate) == 2:
            output, env = candidate
            if not isinstance(output, ExperimentOutput):
                raise TypeError("candidate tuple must contain ExperimentOutput and environment")
            return self.evaluate(output, env, confirm=True)
        if isinstance(candidate, Mapping):
            output = candidate.get("output")
            env = candidate.get("environment")
            if isinstance(output, ExperimentOutput) and isinstance(env, Tier0ResearchEnvironment):
                return self.evaluate(output, env, confirm=True)
        raise TypeError("candidate must be (ExperimentOutput, environment) or a mapping with those fields")

    def compute_scientific_utility(self, trajectory: object) -> Mapping[str, float]:
        output = getattr(trajectory, "output", trajectory)
        verdict = getattr(trajectory, "verdict", None)
        if verdict is None and isinstance(trajectory, Mapping):
            verdict = trajectory.get("verdict")
        valid = bool(getattr(verdict, "validity_pass", False)) if verdict is not None else False
        cost = float(getattr(output, "execution_cost", 0.0))
        effect = float(getattr(output, "effect_estimate", 0.0))
        return {
            "validity_gated_effect": effect if valid else 0.0,
            "cost": cost,
            "utility": (effect if valid else 0.0) - 0.05 * cost,
        }

    def produce_audit_record(self, trajectory: object) -> Mapping[str, object]:
        output = getattr(trajectory, "output", trajectory)
        verdict = getattr(trajectory, "verdict", None)
        return {
            "verifier_version": self.version,
            "output": output.public_dict() if hasattr(output, "public_dict") else output,
            "verdict": verdict.to_dict() if hasattr(verdict, "to_dict") else verdict,
        }

    def evaluate(self, output: ExperimentOutput, env: Tier0ResearchEnvironment, confirm: bool = True) -> Verdict:
        reasons: List[str] = []
        if len(set(self.protocol.exploration_seeds)) != len(self.protocol.exploration_seeds):
            reasons.append("protocol_duplicate_exploration_seeds")
        if len(set(self.protocol.confirmation_seeds)) != len(self.protocol.confirmation_seeds):
            reasons.append("protocol_duplicate_confirmation_seeds")
        if set(self.protocol.exploration_seeds) & set(self.protocol.confirmation_seeds):
            reasons.append("exploration_confirmation_seed_overlap")
        if output.evaluator_hash != env._hash("trusted-evaluator-v1"):
            reasons.append("evaluator_hash_mismatch")
        expected_backend = getattr(env, "BACKEND", None)
        if expected_backend is not None and str(output.backend) != str(expected_backend):
            reasons.append("backend_mismatch")
        if output.method not in {"method_a", "method_b"}:
            reasons.append("unknown_method")
        try:
            ResearchAction(output.action)
        except (TypeError, ValueError):
            reasons.append("unknown_action")
        # Tier-0 has a compact closed-form provenance hash.  Concrete Tier-1
        # backends may bind hashes to generated arrays and split masks, so they expose
        # an explicit evaluator-side provenance hook instead of being forced through
        # Tier-0's hash vocabulary.
        if hasattr(env, "validate_output_provenance"):
            try:
                provenance_ok = bool(env.validate_output_provenance(output))
            except Exception:  # pragma: no cover - defensive verifier boundary
                provenance_ok = False
            if not provenance_ok:
                reasons.append("provenance_hash_mismatch")
        else:
            expected_codes = {
                env._hash(f"code:{output.method}"),
                env._hash(f"tier1-code:{output.method}"),
            }
            if output.code_hash not in expected_codes:
                reasons.append("code_hash_mismatch")
            expected_splits = {
                env._hash("split:random"),
                env._hash("split:repaired"),
                env._hash("split:group"),
            }
            if output.confirmation:
                expected_splits.add(env._hash("split:confirmation_hidden"))
            if output.split_hash not in expected_splits:
                reasons.append("split_hash_mismatch")
        if output.sample_size <= 0:
            reasons.append("empty_sample")
        if output.seed_count != len(output.seeds):
            reasons.append("seed_count_mismatch")
        if output.leakage or "split_overlap_diagnostic" in set(output.validity_signals):
            reasons.append("group_overlap_or_data_leakage")
        invalid_signals = set(output.validity_signals)
        if output.confounding or "treatment_confounder_dependence" in invalid_signals:
            reasons.append("uncontrolled_confounding")
        if "variance_estimator_unstable" in invalid_signals:
            reasons.append("unstable_variance_protocol")
        if "metric_scope_mismatch" in invalid_signals:
            reasons.append("metric_scope_mismatch")
        if any(signal in invalid_signals for signal in ("metric_mismatch_diagnostic", "protocol_invalid_diagnostic")):
            reasons.append("invalid_protocol_or_metric")
        if len(set(output.seeds)) != len(output.seeds):
            reasons.append("non_independent_seeds")
        if set(output.seeds) & set(self.protocol.confirmation_seeds) and not output.confirmation:
            reasons.append("exploration_uses_confirmation_seed")
        decision = classify_evidence(
            validity_pass=not reasons,
            effect_estimate=output.effect_estimate,
            confidence_interval=(output.ci_low, output.ci_high),
            protocol=self.protocol,
            invalid_reasons=reasons,
        )
        confirmation_performed = False
        confirmation_passed = False
        confirmation_seeds: Tuple[int, ...] = ()
        confirmation_dataset_hash = ""
        confirmation_split_hash = ""
        confirmation_data_independent = False
        if confirm and decision.state in (EvidenceState.SUPPORTED, EvidenceState.REFUTED):
            confirmation_performed = True
            confirmation_seeds = tuple(self.protocol.confirmation_seeds)
            candidate = env.clone_from_snapshot(env.snapshot())
            # Re-run the *candidate method at the candidate state* on held-out seeds.
            # Calling ``execute_option`` here would apply SAMPLE/REPAIR/SWITCH a second
            # time and silently change the estimand.  Tier-0/Tier-1 environments expose
            # the deterministic private simulation hook; generic injected executors can
            # fall back to their explicit confirmation API.
            if hasattr(candidate, "_simulate"):
                confirmation_output = candidate._simulate(  # noqa: SLF001 - verifier boundary
                    method=output.method,
                    option=ResearchAction(output.action),
                    seeds=confirmation_seeds,
                    confirmation=True,
                )
            else:
                confirmation_output = candidate.execute_option(
                    ResearchAction(output.action), seeds=confirmation_seeds, confirmation=True
                )
            conf_reasons: List[str] = []
            if confirmation_output.leakage or confirmation_output.confounding:
                conf_reasons.append("confirmation_invalid")
            if tuple(confirmation_output.seeds) != confirmation_seeds:
                conf_reasons.append("confirmation_seed_mismatch")
            if confirmation_output.method != output.method:
                conf_reasons.append("confirmation_method_mismatch")
            if confirmation_output.evaluator_hash != env._hash("trusted-evaluator-v1"):
                conf_reasons.append("confirmation_evaluator_hash_mismatch")
            if not confirmation_output.confirmation:
                conf_reasons.append("confirmation_flag_missing")
            if confirmation_output.dataset_hash == output.dataset_hash:
                conf_reasons.append("confirmation_dataset_not_independent")
            if confirmation_output.split_hash == output.split_hash:
                conf_reasons.append("confirmation_split_not_independent")
            if confirmation_output.backend != output.backend:
                conf_reasons.append("confirmation_backend_mismatch")
            confirmation_dataset_hash = str(confirmation_output.dataset_hash)
            confirmation_split_hash = str(confirmation_output.split_hash)
            confirmation_data_independent = not any(
                reason in conf_reasons
                for reason in (
                    "confirmation_dataset_not_independent",
                    "confirmation_split_not_independent",
                    "confirmation_flag_missing",
                )
            )
            conf_decision = classify_evidence(
                validity_pass=not conf_reasons,
                effect_estimate=confirmation_output.effect_estimate,
                confidence_interval=(confirmation_output.ci_low, confirmation_output.ci_high),
                protocol=self.protocol,
                invalid_reasons=conf_reasons,
            )
            confirmation_passed = conf_decision.state is decision.state and conf_decision.state is not EvidenceState.INVALID
        signature_payload = {
            "output": output.public_dict(),
            "state": decision.state.value,
            "reasons": reasons,
            "verifier": self.version,
        }
        signature = canonical_hash(signature_payload)
        return Verdict(
            validity_pass=not reasons,
            evidence_state=decision.state,
            effect_estimate=output.effect_estimate,
            confidence_interval=(output.ci_low, output.ci_high),
            independent_confirmation_performed=confirmation_performed,
            independent_confirmation_passed=confirmation_passed,
            scientific_claim_consistency=True,
            audit_signature=signature,
            execution_cost=output.execution_cost,
            confirmation_seeds=confirmation_seeds,
            invalid_reasons=tuple(reasons),
            method_family="method_a" if output.method == "method_a" else "method_b",
            confirmation_dataset_hash=confirmation_dataset_hash,
            confirmation_split_hash=confirmation_split_hash,
            confirmation_data_independent=confirmation_data_independent,
        )


def build_verifier(protocol: Optional[Protocol] = None) -> TrustedVerifier:
    return TrustedVerifier(protocol)
