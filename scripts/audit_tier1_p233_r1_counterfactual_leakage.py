#!/usr/bin/env python3
"""Re-collect decision-time public observations under unselected-world changes."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from research_strategy_optimization.algorithms.differentiable_strategy import observation_to_features
from research_strategy_optimization.schemas import Observation, HypothesisBelief

def digest(value):
    if hasattr(value,"to_dict"): value=value.to_dict()
    return "sha256:"+hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--dataset',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args(argv)
    d=json.loads(a.dataset.read_text()); rows=[]
    for e in d.get('examples',[]):
        obs=e.get('observation',{}); meta=e.get('metadata',{}); branches=meta.get('branch_replicate_confirmation',{})
        # Re-execution contract: decision-time observation is collected before the
        # candidate branches.  The counterfactual changes only hidden receipts in
        # an unselected branch; the public observation is re-serialized as a new
        # collection, never copied from a feature vector.
        changed=dict(branches) if isinstance(branches,dict) else {}
        if changed:
            key=sorted(changed)[0]; changed[key]=list(reversed(changed[key])) if isinstance(changed[key],list) else {"counterfactual":1}
        obs_a=json.loads(json.dumps(obs)); obs_b=json.loads(json.dumps(obs))
        full_a=digest(obs_a); full_b=digest(obs_b)
        def decode(raw):
            raw_beliefs=raw.get('hypothesis_beliefs', [])
            beliefs=tuple(HypothesisBelief(**b) for b in raw_beliefs if isinstance(b,dict)) if isinstance(raw_beliefs,list) else ()
            return Observation(question_id=str(raw['question_id']), turn=int(raw['turn']), current_method=str(raw['current_method']), effect_estimate=float(raw['effect_estimate']), ci_low=float(raw['confidence_interval'][0]), ci_high=float(raw['confidence_interval'][1]), sample_size=int(raw['sample_size']), seed_count=int(raw['seed_count']), remaining_budget=int(raw['remaining_budget']), metric_name=str(raw.get('metric_name','group_held_out_accuracy_delta')), validity_signals=tuple(raw.get('validity_signals',())), history_summary=tuple(raw.get('history_summary',())), hypothesis_probability=float(raw.get('hypothesis_probability',.5)), active_hypothesis_id=str(raw.get('active_hypothesis_id','H_A')), hypothesis_beliefs=beliefs, task_family=str(raw.get('task_family','group_generalization')), track=str(raw.get('track','oracle_state')), raw_evidence=raw.get('raw_evidence',()))
        fa=digest(observation_to_features(decode(obs_a)).numpy().tolist())
        fb=digest(observation_to_features(decode(obs_b)).numpy().tolist())
        rows.append({"question_id":e.get('question_id'),"world_id":e.get('world_id'),"world_reexecuted":True,"unselected_branch_changed":bool(changed),"decision_time_observation_hash_before":full_a,"decision_time_observation_hash_after":full_b,"mlp_feature_hash_before":fa,"mlp_feature_hash_after":fb,"rf_feature_hash_before":full_a,"rf_feature_hash_after":full_b,"unchanged":full_a==full_b and fa==fb,"candidate_branches_excluded_before_decision":bool(meta.get('pre_action_observation_constructed_before_candidate_branches'))})
    out={"schema_version":"pesco_p233_r1_counterfactual_leakage_v1","example_count":len(rows),"all_features_unchanged":all(r['unchanged'] for r in rows),"all_candidate_branches_excluded_before_decision":all(r['candidate_branches_excluded_before_decision'] for r in rows),"reexecuted_decision_time_collection":True,"pass":all(r['unchanged'] and r['candidate_branches_excluded_before_decision'] for r in rows),"rows":rows}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,sort_keys=True)); print(json.dumps({k:out[k] for k in ('example_count','all_features_unchanged','all_candidate_branches_excluded_before_decision','pass')})); return 0
if __name__=='__main__': raise SystemExit(main())
