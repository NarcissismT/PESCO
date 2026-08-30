#!/usr/bin/env python3
"""Fail-closed promotion-v5 entrypoint guarded by the P2.3.3 receipt gate.

This command only creates an authorization/commitment receipt.  It never opens
private evaluation data.  Before a real runner may consume private data it must
re-check the gate, dataset commitment, checkpoint bundle, dependency lock and
runtime lock recorded here.
"""
from __future__ import annotations
import argparse, hashlib, json, os, platform, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_digest(payload: dict) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _atomic_sentinel(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        os.write(fd, b"private_data_accessed=false\n")
        os.fsync(fd)
        os.close(fd)
        os.replace(tmp, path)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _checkpoint_bundle(matrix: dict, root: Path) -> tuple[list[dict], bool]:
    rows = []
    ok = True
    for seed in matrix.get("seeds", []):
        seed = int(seed)
        expected = matrix.get("sft_checkpoints", {}).get(str(seed), {})
        manifest_path = root / f"sft_seed_{seed}.json"
        checkpoint_path = root / f"sft_seed_{seed}.pt"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        expected_digest = expected.get("state_dict_sha256")
        manifest_digest = manifest.get("state_dict_sha256")
        row_ok = bool(
            checkpoint_path.exists()
            and manifest_path.exists()
            and expected_digest
            and expected_digest == manifest_digest
            and int(manifest.get("seed", -1)) == seed
            and manifest.get("status") == "completed"
        )
        rows.append({
            "seed": seed,
            "checkpoint": str(checkpoint_path),
            "manifest": str(manifest_path),
            "state_dict_sha256": expected_digest,
            "manifest_state_dict_sha256": manifest_digest,
            "verified": row_ok,
        })
        ok = ok and row_ok
    return rows, bool(ok and len(rows) == 10 and len({r["seed"] for r in rows}) == 10)


def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument('--gate',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--matrix',type=Path,default=ROOT/'artifacts/tier1_p233_matrix_10seed/p233_matrix_result.json')
 p.add_argument('--dataset',type=Path,default=ROOT/'artifacts/tier1_p233_diagnostic/dataset_raw_evidence.json')
 p.add_argument('--collection-manifest',type=Path,default=ROOT/'artifacts/tier1_p233_diagnostic/collection_manifest.json')
 p.add_argument('--checkpoint-root',type=Path,default=ROOT/'artifacts/tier1_p233_matrix_10seed/sft_checkpoints')
 p.add_argument('--dependency-lock',type=Path,default=ROOT/'requirements.txt')
 a=p.parse_args(argv)
 gate=json.loads(a.gate.read_text()); matrix=json.loads(a.matrix.read_text()); collection=json.loads(a.collection_manifest.read_text())
 gate_unsigned=dict(gate); declared_gate_digest=gate_unsigned.pop('audit_sha256',None)
 gate_digest_match=bool(declared_gate_digest and declared_gate_digest == _canonical_digest(gate_unsigned))
 go=gate.get('status')=='GO' and bool(gate.get('p233_go')) and all(bool(v) for v in gate.get('p233_go',{}).values())
 dataset_digest=_sha256(a.dataset) if a.dataset.exists() else None
 dataset_commitment_match=bool(dataset_digest and dataset_digest == collection.get('dataset_sha256'))
 checkpoints, checkpoint_bundle_verified = _checkpoint_bundle(matrix, a.checkpoint_root)
 dependency_digest=_sha256(a.dependency_lock) if a.dependency_lock.exists() else None
 runtime_payload={'python':sys.version.split()[0], 'platform':platform.platform(), 'dependency_lock_sha256':dependency_digest, 'root':str(ROOT)}
 runtime_digest=_canonical_digest(runtime_payload)
 sentinel=Path(str(a.output)+'.sentinel'); _atomic_sentinel(sentinel)
 out={'schema_version':'pesco_promotion_v5_guard_v0.2','status':'AUTHORIZED' if go else 'REFUSED_P233_NO_GO',
      'p233_gate_digest':declared_gate_digest or _sha256(a.gate),'p233_gate_file_sha256':_sha256(a.gate),
      'gate_revalidated':bool(gate_digest_match and go),'dataset':{'path':str(a.dataset),'sha256':dataset_digest,'committed_sha256':collection.get('dataset_sha256'),'commitment_match':dataset_commitment_match},
      'checkpoint_bundle':{'root':str(a.checkpoint_root),'count':len(checkpoints),'verified':checkpoint_bundle_verified,'receipts':checkpoints},
      'dependency_lock':{'path':str(a.dependency_lock),'sha256':dependency_digest,'present':bool(dependency_digest)},
      'runtime_lock':{'binding_kind':'content_addressed_runtime_lock','sha256':runtime_digest,'payload':runtime_payload,'reverify_before_private_access':True},
      'private_data_accessed':False,'sentinel':str(sentinel),'sentinel_created_before_private_access':True,
      'runtime_reverification_contract':{'evaluator_digest':True,'config_digest':True,'dataset_commitment':True,'checkpoint_bundle':True,'dependency_lock':True,'container_or_runtime_lock':True},
      'reason':'P2.3.3 gate must be GO before any private promotion-v5 access; this receipt performs no private access'}
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,sort_keys=True)); print(json.dumps(out,indent=2)); return 0 if go else 2
if __name__=='__main__': raise SystemExit(main())
