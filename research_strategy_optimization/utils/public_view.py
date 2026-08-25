"""Whitelist conversions that prevent trusted labels leaking into policy inputs."""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping

from ..schemas import Observation, Trajectory


def policy_observation(observation: Observation) -> Dict[str, Any]:
    return observation.to_dict()


def assert_public_observation(
    observation: Observation,
    forbidden_tokens: tuple[str, ...] = (
        "world_id",
        "latent_effect",
        "true_effect_a",
        "hidden_world_id",
        "verifier_label",
    ),
) -> None:
    """Fail closed if a policy-visible observation contains a reserved field token."""

    payload = json.dumps(policy_observation(observation), sort_keys=True).lower()
    hits = [token for token in forbidden_tokens if token.lower() in payload]
    if hits:
        raise AssertionError("policy observation contains hidden-field tokens: " + ", ".join(hits))


def public_trajectory(trajectory: Trajectory) -> Dict[str, Any]:
    """Export only policy-visible observations and raw public experiment outputs."""

    return {
        "question_id": trajectory.question_id,
        "world_id": "hidden_from_agent",
        "branch_id": trajectory.branch_id,
        "initial_observation": trajectory.initial_observation.to_dict(),
        "final_observation": trajectory.final_observation.to_dict(),
        "outputs": [output.public_dict() for output in trajectory.outputs],
        "belief_before": trajectory.belief_before,
        "belief_after": trajectory.belief_after,
        "total_cost": trajectory.total_cost,
        "proposal_source": trajectory.proposal_source,
    }
