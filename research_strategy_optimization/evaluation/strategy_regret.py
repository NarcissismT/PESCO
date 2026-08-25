from __future__ import annotations

from typing import Mapping, Sequence


def research_regret(records: Sequence[Mapping[str, object]]) -> float:
    regrets = []
    for r in records:
        best = r.get("best_utility", r.get("oracle_utility"))
        chosen = r.get("chosen_utility", r.get("utility"))
        if best is None or chosen is None:
            continue
        regrets.append(float(best) - float(chosen))
    return sum(regrets) / len(regrets) if regrets else 0.0


def effective_switch_rate(records: Sequence[Mapping[str, object]]) -> float:
    values = [bool(r.get("effective_switch", False)) for r in records if r.get("switch") is not None or r.get("selected_action")]
    return sum(values) / len(values) if values else 0.0


def unnecessary_switch_rate(records: Sequence[Mapping[str, object]]) -> float:
    switches = [r for r in records if bool(r.get("switch", r.get("selected_action") == "switch_to_alternative_method"))]
    return sum(bool(r.get("unnecessary_switch", False)) for r in switches) / len(switches) if switches else 0.0


def appropriate_persistence(records: Sequence[Mapping[str, object]]) -> float:
    candidates = [r for r in records if r.get("current_strategy_optimal") is not None]
    return sum(bool(r.get("persistence_correct", False)) for r in candidates) / len(candidates) if candidates else 0.0

