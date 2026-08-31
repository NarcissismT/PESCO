#!/usr/bin/env python3
"""Ten leave-one-family-out training/evaluation diagnostics.

Each family is removed from the training partition, assigned to an isolated
holdout partition, and a fresh common-SFT pure-factor run is evaluated there.
This is diagnostic evidence; it is never substituted for the frozen promotion.
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from dataclasses import replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from research_strategy_optimization.algorithms.differentiable_strategy import DecisionDataset

FAMILIES=("group_leakage","causal_confounding","low_sample_variance","subgroup_metric_mismatch","heterogeneous_noise","nonlinear_response","measurement_shift","missing_not_at_random","intervention_noncompliance","group_generalization")

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--dataset',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--seed',type=int,default=17); p.add_argument('--steps',type=int,default=128); a=p.parse_args(argv)
    raw=DecisionDataset.from_json(a.dataset); root=a.output.parent/(a.output.stem+'_runs'); root.mkdir(parents=True,exist_ok=True); rows=[]
    # The runner invokes a fresh process per family so allocator state and receipt
    # caches cannot cross a held-out boundary.
    for family in FAMILIES:
        examples=[]
        for e in raw.examples:
            fam=str(e.metadata.get('family',''))
            if fam==family and e.split in {'train','tune','promotion'}: examples.append(replace(e,split='loo_test'))
            else: examples.append(e)
        ds_path=root/f'{family}_dataset.json'; ds_path.write_text(json.dumps({'schema_version':raw.schema_version,'provenance':dict(raw.provenance,loo_family=family),'examples':[e.to_dict() for e in examples],'reversals':[{'left':x.left,'right':x.right,'action_left':x.action_left.value,'action_right':x.action_right.value,'margin':x.margin,'confirmed':x.confirmed,'weight':x.weight,'lcb_left':x.lcb_left,'ucb_right':x.ucb_right,'sample_count':x.sample_count} for x in raw.reversals]},ensure_ascii=False),encoding='utf-8')
        out=root/family; cmd=[sys.executable,str(ROOT/'scripts/run_tier1_p233_chunk.py'),'--dataset',str(ds_path),'--output-dir',str(out),'--seed',str(a.seed),'--methods','Atomic+State+Branch','Atomic+State+Flip','PESCO-Full','--splits','loo_test','--finetune-steps',str(a.steps),'--sft-steps','128','--hidden-dim','32','--batch-size','64','--branch-formulation','expected_utility']
        subprocess.run(cmd,check=True)
        result=json.loads((out/'p231_result.json').read_text()); rows.extend([{'family':family,'method':r.get('method'),'heldout_split':'loo_test','normalized_regret':r.get('normalized_regret'),'pairrank_score':r.get('pairwise_reversal_ranking_score'),'state_macro_f1':r.get('state_macro_f1'),'training_excluded_family':family,'fresh_sft':True} for r in result.get('records',[]) if r.get('split')=='loo_test'])
    out={'schema_version':'pesco_p233_r1_family_loo_v1','families':list(FAMILIES),'rows':rows,'ten_family_runs_completed':len({r['family'] for r in rows})==10,'training_excluded_family':True,'diagnostic_only':True,'formal_comparison_authorized':False}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,ensure_ascii=False,sort_keys=True)); print(json.dumps({'output':str(a.output),'families':len(set(r['family'] for r in rows))})); return 0
if __name__=='__main__': raise SystemExit(main())
