"""Append-only, tamper-evident registry for hypotheses and beliefs.

The registry is intentionally small, but it enforces the two properties needed by the
PESCO protocol: a belief cannot be overwritten at the same research turn, and every
belief/evidence append is linked to the previous append by a per-hypothesis hash chain.
Public accessors return deep copies, so callers cannot mutate committed records through
an alias.  :meth:`verify` is available for audit/replay checks and detects both chain
tampering and divergence between the typed views and the underlying event stream.
"""

from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import json
import math
from dataclasses import asdict
from typing import Any, Dict, List, Mapping, Optional, Set

from ..schemas import Hypothesis


GENESIS_HASH = "genesis"


def _digest(payload: Any) -> str:
    """Hash a JSON-compatible payload with stable key/number ordering."""

    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


class HypothesisRegistry:
    """Immutable-by-default hypothesis, belief, and evidence ledger.

    ``commit_belief`` uses strictly increasing integer turns.  A duplicate turn is
    rejected even when the proposed probability is identical; silently accepting it
    would allow a caller to create multiple scores for one decision point.  Evidence
    events are allowed to share a research turn (one action can emit several records),
    but their event indices and hash-chain links must remain unique and ordered.
    """

    def __init__(self) -> None:
        self._hypotheses: Dict[str, Hypothesis] = {}
        self._beliefs: Dict[str, List[Dict[str, Any]]] = {}
        self._evidence: Dict[str, List[Dict[str, Any]]] = {}
        self._chains: Dict[str, List[Dict[str, Any]]] = {}
        self._belief_turns: Dict[str, Set[int]] = {}
        self._confirmation_frozen: Set[str] = set()

    def register_before_experiment(self, hypothesis: Hypothesis, protocol: Any = None) -> Hypothesis:
        if not hypothesis.registered_before_confirmation:
            raise ValueError("hypothesis must be registered before confirmation")
        hypothesis_id = str(hypothesis.hypothesis_id)
        if not hypothesis_id:
            raise ValueError("hypothesis_id must be non-empty")
        if hypothesis_id in self._hypotheses:
            raise ValueError(f"hypothesis already registered: {hypothesis_id}")
        self._hypotheses[hypothesis_id] = copy.deepcopy(hypothesis)
        self._beliefs[hypothesis_id] = []
        self._evidence[hypothesis_id] = []
        self._chains[hypothesis_id] = []
        self._belief_turns[hypothesis_id] = set()
        return copy.deepcopy(hypothesis)

    def commit_belief(
        self,
        hypothesis_id: str,
        probability: float,
        turn: Optional[int] = None,
        source: str = "policy",
        timestamp: Optional[str] = None,
    ) -> str:
        """Commit one belief and return its chain event hash.

        Turns are decision-point identifiers, not merely display metadata; they must be
        non-negative and strictly increase for a hypothesis.
        """

        hypothesis_id = str(hypothesis_id)
        self._require(hypothesis_id)
        # The explicit turn is preferred and is required for preregistered replay.  A
        # missing turn gets the next monotone index for compatibility with lightweight
        # callers; it is still committed immutably and cannot be overwritten.
        raw_turn = (max(self._belief_turns[hypothesis_id]) + 1) if turn is None and self._belief_turns[hypothesis_id] else (0 if turn is None else turn)
        try:
            turn = int(raw_turn)
        except (TypeError, ValueError) as error:
            raise ValueError("turn must be an integer") from error
        if isinstance(raw_turn, bool) or (isinstance(raw_turn, float) and raw_turn != turn):
            raise ValueError("turn must be an integer")
        if turn < 0:
            raise ValueError("turn must be non-negative")
        if turn in self._belief_turns[hypothesis_id]:
            raise ValueError(f"belief already committed for hypothesis {hypothesis_id} at turn {turn}")
        prior_turns = self._belief_turns[hypothesis_id]
        if prior_turns and turn <= max(prior_turns):
            raise ValueError("belief turns must be strictly increasing")
        probability = float(probability)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("belief probability must be finite and lie in [0, 1]")
        record = {
            "event_type": "belief_committed",
            "turn": turn,
            "probability": probability,
            "source": str(source),
            "timestamp": str(timestamp) if timestamp is not None else _dt.datetime.now(_dt.timezone.utc).isoformat(),
        }
        event = self._append_event(hypothesis_id, record)
        self._beliefs[hypothesis_id].append(event)
        self._belief_turns[hypothesis_id].add(turn)
        return str(event["record_hash"])

    def append_evidence(self, hypothesis_id: str, evidence_record: Mapping[str, Any]) -> str:
        """Append trusted evidence metadata and return its chain event hash."""

        hypothesis_id = str(hypothesis_id)
        self._require(hypothesis_id)
        if not isinstance(evidence_record, Mapping):
            raise TypeError("evidence_record must be a mapping")
        record = dict(evidence_record)
        # These fields are owned by the registry.  Reject rather than overwrite so a
        # caller cannot smuggle a forged index/hash into an audit payload.
        reserved = {"event_type", "event_index", "previous_event_hash", "record_hash"}
        supplied = reserved.intersection(record)
        if supplied:
            raise ValueError(f"evidence_record contains registry-owned fields: {sorted(supplied)}")
        record["event_type"] = "evidence_appended"
        event = self._append_event(hypothesis_id, record)
        self._evidence[hypothesis_id].append(event)
        return str(event["record_hash"])

    def freeze_confirmation_protocol(self, hypothesis_id: str) -> None:
        self._require(hypothesis_id)
        hypothesis = self._hypotheses[hypothesis_id]
        if not hypothesis.registered_before_confirmation:
            raise ValueError("confirmation protocol was not frozen before confirmation")
        self._confirmation_frozen.add(hypothesis_id)

    def confirmation_protocol_frozen(self, hypothesis_id: str) -> bool:
        self._require(hypothesis_id)
        return hypothesis_id in self._confirmation_frozen

    def beliefs(self, hypothesis_id: str) -> List[Dict[str, Any]]:
        self._require(hypothesis_id)
        return copy.deepcopy(self._beliefs[hypothesis_id])

    def evidence(self, hypothesis_id: str) -> List[Dict[str, Any]]:
        self._require(hypothesis_id)
        return copy.deepcopy(self._evidence[hypothesis_id])

    def hash_chain(self, hypothesis_id: str) -> List[Dict[str, Any]]:
        self._require(hypothesis_id)
        return copy.deepcopy(self._chains[hypothesis_id])

    def hash_chain_tip(self, hypothesis_id: str) -> str:
        self._require(hypothesis_id)
        chain = self._chains[hypothesis_id]
        return str(chain[-1]["record_hash"]) if chain else GENESIS_HASH

    def verify(self) -> bool:
        """Verify all event links and typed-view/chain consistency."""

        try:
            for hypothesis_id in self._hypotheses:
                chain = self._chains[hypothesis_id]
                # Beliefs and evidence have separate typed views but share one append
                # stream, so reconstruct the stream by its registry-owned index.
                typed = sorted(
                    self._beliefs[hypothesis_id] + self._evidence[hypothesis_id],
                    key=lambda event: int(event["event_index"]),
                )
                if len(chain) != len(typed):
                    return False
                previous = GENESIS_HASH
                for index, event in enumerate(chain):
                    if event.get("event_index") != index:
                        return False
                    if event.get("previous_event_hash") != previous:
                        return False
                    expected = _digest({key: value for key, value in event.items() if key != "record_hash"})
                    if expected != event.get("record_hash"):
                        return False
                    if event != typed[index]:
                        return False
                    previous = str(event["record_hash"])
                turns = [int(event["turn"]) for event in self._beliefs[hypothesis_id]]
                if len(turns) != len(set(turns)) or turns != sorted(turns):
                    return False
                if set(turns) != set(self._belief_turns[hypothesis_id]):
                    return False
            return True
        except (KeyError, TypeError, ValueError, OverflowError):
            return False

    # Alias used by audit callers that name the invariant explicitly.
    verify_hash_chain = verify

    def records(self) -> Dict[str, Any]:
        return {
            "hypotheses": {key: asdict(value) for key, value in self._hypotheses.items()},
            "beliefs": copy.deepcopy(self._beliefs),
            "evidence": copy.deepcopy(self._evidence),
            "hash_chains": copy.deepcopy(self._chains),
            "confirmation_protocol_frozen": sorted(self._confirmation_frozen),
            "hash_chain_valid": self.verify(),
        }

    def _append_event(self, hypothesis_id: str, record: Mapping[str, Any]) -> Dict[str, Any]:
        chain = self._chains[hypothesis_id]
        event = dict(record)
        event["event_index"] = len(chain)
        event["previous_event_hash"] = str(chain[-1]["record_hash"]) if chain else GENESIS_HASH
        event["record_hash"] = _digest(event)
        committed = copy.deepcopy(event)
        chain.append(committed)
        return committed

    def _require(self, hypothesis_id: str) -> None:
        if hypothesis_id not in self._hypotheses:
            raise KeyError(hypothesis_id)


__all__ = ["GENESIS_HASH", "HypothesisRegistry"]
