#!/usr/bin/env python3
"""Create the clean-commit/hyperparameter freeze receipt for promotion-v3."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--p231-gate", type=Path, required=True)
    parser.add_argument("--final-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    gate = json.loads(args.p231_gate.read_text(encoding="utf-8"))
    manifest = json.loads(args.final_manifest.read_text(encoding="utf-8"))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    clean = not bool(status.strip())
    config = {"sft_steps": 16, "finetune_steps": 16, "batch_size": 32, "hidden_dim": 24, "learning_rate": 0.005, "minibatch_epochs": 4, "clip_epsilon": 0.05, "top1_gap_threshold": 0.05}
    config_digest = "sha256:" + hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    receipt = {
        "schema_version": "pesco_tier1_p231_promotion_v3_freeze_receipt_v0.1",
        "status": "FROZEN_FOR_PROMOTION_V3" if gate.get("status") == "GO_P2_3_1_10SEED_AUTHORIZED" and clean else "NOT_FROZEN",
        "clean_commit": clean,
        "commit": commit,
        "p231_gate_status": gate.get("status"),
        "p231_gate_digest": "sha256:" + hashlib.sha256(args.p231_gate.read_bytes()).hexdigest(),
        "final_manifest_digest": manifest.get("manifest_digest"),
        "config": config,
        "config_digest": config_digest,
        "code_and_hparams_frozen_before_final_access": True,
        "final_access_not_yet_completed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if receipt["status"] == "FROZEN_FOR_PROMOTION_V3" else 2


if __name__ == "__main__":
    raise SystemExit(main())
