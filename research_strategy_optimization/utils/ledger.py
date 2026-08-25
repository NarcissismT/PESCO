"""Append-only hash-chain audit ledger."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


def canonical_digest(payload: Any) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


class AuditLedger:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    @property
    def tip(self) -> str:
        return self.events[-1]["event_hash"] if self.events else "genesis"

    def append(self, event_type: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "event_type": event_type,
            "event_index": len(self.events),
            "previous_event_hash": self.tip,
            "payload": copy.deepcopy(dict(payload)),
        }
        record["event_hash"] = canonical_digest(record)
        self.events.append(record)
        return copy.deepcopy(record)

    def verify(self) -> bool:
        previous = "genesis"
        for index, event in enumerate(self.events):
            if event.get("event_index") != index or event.get("previous_event_hash") != previous:
                return False
            expected = canonical_digest({k: v for k, v in event.items() if k != "event_hash"})
            if expected != event.get("event_hash"):
                return False
            previous = event["event_hash"]
        return True

    def write_jsonl(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("\n".join(json.dumps(e, ensure_ascii=False, sort_keys=True) for e in self.events) + ("\n" if self.events else ""), encoding="utf-8")

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "AuditLedger":
        """Load an audit chain without trusting its validity."""

        source = Path(path)
        ledger = cls()
        if not source.exists():
            raise FileNotFoundError(source)
        for line_no, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid audit JSONL at line {line_no}") from error
            if not isinstance(event, Mapping):
                raise ValueError(f"audit line {line_no} is not an object")
            ledger.events.append(copy.deepcopy(dict(event)))
        return ledger


def verify_jsonl(path: str | Path) -> bool:
    """Verify an on-disk hash chain in one call."""

    return AuditLedger.from_jsonl(path).verify()
