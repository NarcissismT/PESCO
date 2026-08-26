"""Fail-closed gate for the 1.5B–3B offline LoRA/QLoRA stage (P3).

The review explicitly places model-scale post-training after the CPU promotion gate.
This module records the prerequisite audit and refuses to claim a model experiment
when P2 is NO-GO, the formal split is closed, or the required adapter stack is absent.
"""

from __future__ import annotations

import importlib.metadata as metadata
import json
import platform
from pathlib import Path
from typing import Any, Mapping, Optional

from ..utils.run_manifest import build_run_manifest, write_run_manifest


def _version(name: str) -> Optional[str]:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def _torch_capability() -> dict:
    try:
        import torch
        return {
            "installed": True,
            "version": str(torch.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        }
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"installed": False, "version": None, "cuda_available": False, "device_count": 0, "error": str(exc)}


def run_p3_gate(output_dir: str | Path, p2_result_path: str | Path) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    p2_path = Path(p2_result_path)
    p2 = json.loads(p2_path.read_text(encoding="utf-8")) if p2_path.exists() else {}
    p2_gates = dict(p2.get("gates", {}))
    capability = {
        "transformers": _version("transformers"),
        "peft": _version("peft"),
        "bitsandbytes": _version("bitsandbytes"),
        "accelerate": _version("accelerate"),
        "torch": _torch_capability(),
        "python": platform.python_version(),
    }
    prerequisite_reasons = []
    if not p2:
        prerequisite_reasons.append("p2_result_missing")
    if p2.get("promotion_status") != "GO":
        prerequisite_reasons.append("p2_promotion_gate_not_passed")
    if not bool(p2.get("formal_comparison_authorized", False)):
        prerequisite_reasons.append("formal_cpu_comparison_not_authorized")
    if capability["transformers"] is None:
        prerequisite_reasons.append("transformers_not_installed")
    if capability["peft"] is None:
        prerequisite_reasons.append("peft_not_installed")
    if not capability["torch"].get("cuda_available", False):
        prerequisite_reasons.append("no_cuda_device_for_qlora")
    methods = [
        "SFT offline LoRA",
        "Branch advantage offline LoRA",
        "NoFlip offline LoRA",
        "PESCO-Full offline LoRA",
        "compute-matched inference-time branch search",
    ]
    result = {
        "schema_version": "pesco_tier1_p3_small_model_gate_v0.1",
        "status": "ready_to_run" if not prerequisite_reasons else "no_go_prerequisites_not_met",
        "experiment_executed": False,
        "promotion_prerequisites": {
            "p2_result": str(p2_path),
            "p2_promotion_status": p2.get("promotion_status"),
            "p2_gates": p2_gates,
            "formal_comparison_authorized": bool(p2.get("formal_comparison_authorized", False)),
        },
        "capability_audit": capability,
        "planned_methods": methods,
        "parameter_internalization_test": {
            "search_budget_conditions": [1.0, 0.5, 0.25],
            "primary_metric": "normalized research regret",
            "must_be_run_only_after_promotion": True,
        },
        "blocking_reasons": prerequisite_reasons,
        "online_rl_or_7b_authorized": False,
        "diagnostic_only": True,
    }
    (output / "small_model_gate.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = build_run_manifest(
        experiment="tier1_p3_small_model_gate",
        repo_root=Path(__file__).resolve().parents[2],
        runner_paths=[Path(__file__)],
        data_paths=[p2_path] if p2_path.exists() else [],
        seeds={},
        checkpoint=None,
        status=result["status"],
        diagnostics={"blocking_reasons": prerequisite_reasons},
    )
    write_run_manifest(output / "run_manifest.json", manifest)
    return result


__all__ = ["run_p3_gate"]
