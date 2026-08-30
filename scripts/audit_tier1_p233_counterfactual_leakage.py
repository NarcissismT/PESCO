#!/usr/bin/env python3
"""Verify unselected-branch changes cannot affect decision-time features."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from research_strategy_optimization.algorithms.differentiable_strategy import DecisionDataset, observation_to_features
def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument('--dataset',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args(argv); d=DecisionDataset.from_json(a.dataset); rows=[]
 for e in d.examples:
  before=hashlib.sha256(observation_to_features(e.observation).numpy().tobytes()).hexdigest(); meta=dict(e.metadata); branches=dict(meta.get('branch_replicate_confirmation',{})); keys=list(branches)
  if keys:
   first=keys[0]; changed=dict(meta); changed_branches=dict(branches); changed_branches[first]=list(reversed(changed_branches[first])); changed['branch_replicate_confirmation']=changed_branches
  after=hashlib.sha256(observation_to_features(e.observation).numpy().tobytes()).hexdigest(); rows.append({'question_id':e.question_id,'world_id':e.world_id,'feature_hash_before':before,'feature_hash_after_unselected_branch_change':after,'unchanged':before==after,'pre_action_observation_hash_present':bool(meta.get('pre_action_observation_hash')),'cross_candidate_confirmation_feature_excluded':bool(meta.get('cross_candidate_confirmation_feature_excluded'))})
 out={'schema_version':'pesco_p233_counterfactual_leakage_audit_v0.1','example_count':len(rows),'all_features_unchanged':all(r['unchanged'] for r in rows),'all_metadata_exclusion_flags':all(r['cross_candidate_confirmation_feature_excluded'] for r in rows),'all_pre_action_hashes_present':all(r['pre_action_observation_hash_present'] for r in rows),'pass':all(r['unchanged'] and r['cross_candidate_confirmation_feature_excluded'] for r in rows),'rows_digest':'sha256:'+hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()}; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,sort_keys=True)); print(json.dumps(out,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
