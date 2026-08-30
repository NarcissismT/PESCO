#!/usr/bin/env python3
"""Generate and collect the fresh P2.3.3 diagnostic benchmark."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from research_strategy_optimization.evaluation.tier1_p233_dataset import build_tier1_p233_diagnostic_benchmark
from research_strategy_optimization.evaluation.tier1_v04_extended import collect_tier1_v04_extended, TRACK_RAW_EVIDENCE, V04_EXTENDED_EXPLORATION_SEEDS, V04_EXTENDED_CONFIRMATION_SEEDS
from research_strategy_optimization.schemas import Protocol

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,default=ROOT/'artifacts/tier1_p233_diagnostic/dataset_raw_evidence.json'); a=p.parse_args(argv)
    b=build_tier1_p233_diagnostic_benchmark(); protocol=Protocol(protocol_version='pesco_v0_2', exploration_seeds=V04_EXTENDED_EXPLORATION_SEEDS, confirmation_seeds=V04_EXTENDED_CONFIRMATION_SEEDS)
    dataset,audit=collect_tier1_v04_extended(b,protocol,track=TRACK_RAW_EVIDENCE)
    a.output.parent.mkdir(parents=True,exist_ok=True); dataset.save_json(a.output,include_audit=True)
    public=b.manifest(include_hidden=False,exploration_seeds=protocol.exploration_seeds); hidden=b.manifest(include_hidden=True,exploration_seeds=protocol.exploration_seeds)
    (a.output.parent/'benchmark_manifest_public.json').write_text(json.dumps(public,indent=2,ensure_ascii=False,sort_keys=True),encoding='utf-8')
    (a.output.parent/'benchmark_manifest_hidden.json').write_text(json.dumps(hidden,indent=2,ensure_ascii=False,sort_keys=True),encoding='utf-8')
    manifest={'schema_version':'pesco_p233_diagnostic_collection_v0.1','generator_version':public['generator_version'],'dataset_sha256':'sha256:'+hashlib.sha256(a.output.read_bytes()).hexdigest(),'question_count':len(b.questions),'world_count':len(b.worlds),'counts_by_split':{s:sum(q.split==s for q in b.questions) for s in b.split_names},'mechanism_families':list(b.manifest(include_hidden=True)['mechanism_families']),'collector_audit':audit,'diagnostic_only':True,'formal_comparison_authorized':False}
    (a.output.parent/'collection_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False,sort_keys=True),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
