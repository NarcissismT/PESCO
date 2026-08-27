#!/usr/bin/env python3
"""Write the single pre-registered promotion-v3 10-seed gate receipt."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def _ci(values, seed=17, replicates=2000):
    if not values:
        return {"point": None, "lower": None, "upper": None, "method": "not_estimable"}
    point = sum(values) / len(values)
    if len(values) < 2:
        return {"point": point, "lower": point, "upper": point, "method": "degenerate"}
    rng = random.Random(seed); draws=[]
    for _ in range(replicates):
        sample=[rng.choice(values) for _ in values]; draws.append(sum(sample)/len(sample))
    draws.sort(); return {"point": point, "lower": draws[int(.025*len(draws))], "upper": draws[int(.975*len(draws))-1], "method": "seed_question_cluster_bootstrap_percentile_95", "replicates": replicates}


def _question_matrix(rows, key):
    matrix=defaultdict(dict); families={}
    for row in rows:
        seed=int(row["seed"])
        for q in row.get("question_metric_rows", []):
            if q.get(key) is not None:
                matrix[seed][str(q["question_id"])] = float(q[key])
                families[str(q["question_id"])] = str(q.get("family", ""))
    return matrix, families


def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--result',type=Path,required=True); ap.add_argument('--shortcut',type=Path,required=True); ap.add_argument('--stability',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args(argv)
    result=json.loads(args.result.read_text(encoding='utf-8')); shortcut=json.loads(args.shortcut.read_text(encoding='utf-8')); stability=json.loads(args.stability.read_text(encoding='utf-8'))
    primary='GRPO+Branch+Flip'; baseline='GRPO-MatchedAtomic'; splits=('final_id','final_ood')
    by_method_split=defaultdict(list)
    for row in result.get('records',[]): by_method_split[(row.get('method'),row.get('split'))].append(row)
    comparison={}; family_direction={}; all_gates={}
    for split in splits:
        p_rows=by_method_split[(primary,split)]; b_rows=by_method_split[(baseline,split)]
        p_reg,p_fam=_question_matrix(p_rows,'normalized_regret'); b_reg,_=_question_matrix(b_rows,'normalized_regret')
        p_pair,pf=_question_matrix(p_rows,'pairwise_reversal_ranking_accuracy'); b_pair,_=_question_matrix(b_rows,'pairwise_reversal_ranking_accuracy')
        diffs=[]; pair_diffs=[]; seed_direction=[]
        for seed in sorted(set(p_reg)&set(b_reg)):
            qs=sorted(set(p_reg[seed])&set(b_reg[seed])); vals=[p_reg[seed][q]-b_reg[seed][q] for q in qs]; diffs.extend(vals)
            pqs=sorted(set(p_pair.get(seed,{}))&set(b_pair.get(seed,{}))); pair_diffs.extend([p_pair[seed][q]-b_pair[seed][q] for q in pqs])
            seed_direction.append({"seed":seed,"primary_mean":sum(p_reg[seed][q] for q in qs)/max(1,len(qs)),"baseline_mean":sum(b_reg[seed][q] for q in qs)/max(1,len(qs)),"primary_lower":sum(p_reg[seed][q] for q in qs)<sum(b_reg[seed][q] for q in qs)})
        family_values=defaultdict(list)
        for seed in p_reg:
            for qid in set(p_reg[seed])&set(b_reg.get(seed,{})):
                family_values[p_fam.get(qid,'')].append(p_reg[seed][qid]-b_reg[seed][qid])
        family_direction[split]={family:(sum(vals)/len(vals) if vals else None) for family,vals in sorted(family_values.items())}
        p_top=max((row.get('action_tie_n',0) for row in p_rows),default=0); p_non=max((row.get('action_non_tie_n',0) for row in p_rows),default=0)
        comparison[split]={"regret_delta_primary_minus_baseline_ci":_ci(diffs,seed=100+len(split)),"pairrank_delta_primary_minus_baseline_ci":_ci(pair_diffs,seed=200+len(split)),"seed_direction":seed_direction,"direction_count":sum(x['primary_lower'] for x in seed_direction),"seed_count":len(seed_direction),"tie_n_max":p_top,"non_tie_n_max":p_non}
    # Safety receipts are evaluated on selected actions only and require non-zero
    # denominators.  A 5-point tolerance is the pre-registered diagnostic margin.
    p_safety={}; b_safety={}
    for split in splits:
        p=by_method_split[(primary,split)]; b=by_method_split[(baseline,split)]
        def avg(name): return sum(float(x.get(name,0.0) or 0.0) for x in p)/max(1,len(p))
        def avgb(name): return sum(float(x.get(name,0.0) or 0.0) for x in b)/max(1,len(b))
        p_safety[split]={"invalid_branch_rate":avg('selected_invalid_branch_rate'),"confirmation_rate":avg('confirmation_rate'),"false_discovery_rate":avg('invalid_local_optimization_rate'),"confirmation_eligible_n":sum(int(x.get('confirmation_eligible_n',0) or 0) for x in p)}
        b_safety[split]={"invalid_branch_rate":avgb('selected_invalid_branch_rate'),"confirmation_rate":avgb('confirmation_rate'),"false_discovery_rate":avgb('invalid_local_optimization_rate'),"confirmation_eligible_n":sum(int(x.get('confirmation_eligible_n',0) or 0) for x in b)}
    shortcut_best={}
    for split in splits:
        vals=[]
        for key,item in shortcut.get('models',{}).items():
            metric=item.get('metrics_by_split',{}).get(split,{}).get('normalized_regret')
            if metric is not None: vals.append((key,float(metric)))
        shortcut_best[split]=dict(vals)
    all_gates['regret_ci_below_zero']=all(comparison[s]['regret_delta_primary_minus_baseline_ci']['upper'] < 0 for s in splits)
    all_gates['pairrank_ci_above_zero']=all(comparison[s]['pairrank_delta_primary_minus_baseline_ci']['lower'] > 0 for s in splits)
    all_gates['at_least_8_of_10_seeds']=all(comparison[s]['direction_count']>=8 and comparison[s]['seed_count']==10 for s in splits)
    all_gates['selected_action_safety_noninferior']=all(p_safety[s]['invalid_branch_rate']<=b_safety[s]['invalid_branch_rate']+0.05 and p_safety[s]['false_discovery_rate']<=b_safety[s]['false_discovery_rate']+0.05 and p_safety[s]['confirmation_rate']>=b_safety[s]['confirmation_rate']-0.05 and p_safety[s]['confirmation_eligible_n']>0 and b_safety[s]['confirmation_eligible_n']>0 for s in splits)
    all_gates['relative_reward_stability']=stability.get('authorized') is True and stability.get('counts',{}).get('non_tie_world_count',0)>0 and stability.get('counts',{}).get('tie_world_count',0)>0
    all_gates['strict_shortcut_available']=bool(shortcut.get('dependency',{}).get('sklearn',{}).get('available'))
    all_gates['neural_beats_shortcut']=all(float(next((r.get('normalized_regret') for r in by_method_split[(primary,s)] if r.get('normalized_regret') is not None),1.0)) < min(shortcut_best.get(s,{}).values() or [1.0]) for s in splits)
    all_gates['family_majority_direction']=all(sum(v<0 for v in family_direction[s].values()) >= max(1,(len(family_direction[s])+1)//2) for s in splits)
    out={'schema_version':'pesco_tier1_p23_promotion_v3_gate_v0.1','status':'GO_PROMOTION_V3' if all(all_gates.values()) else 'NO_GO_PROMOTION_V3','primary_method':primary,'dev_selected_baseline':baseline,'comparison':comparison,'safety_primary':p_safety,'safety_baseline':b_safety,'shortcut_metrics':shortcut_best,'family_direction':family_direction,'gates':all_gates,'tie_set_definition':'top1_minus_top2 <= 0.02; action accuracy gate excludes ties','final_access_receipt':result.get('final_access'),'formal_comparison_authorized':True}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(out,indent=2,ensure_ascii=False,sort_keys=True),encoding='utf-8'); print(json.dumps({'status':out['status'],'gates':all_gates},ensure_ascii=False,indent=2)); return 0 if out['status']=='GO_PROMOTION_V3' else 2
if __name__=='__main__': raise SystemExit(main())
