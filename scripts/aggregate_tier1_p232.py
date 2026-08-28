#!/usr/bin/env python3
"""Paired two-way (seed × question-cluster) bootstrap for P2.3.2."""
from __future__ import annotations
import argparse, json, random
from collections import defaultdict
from pathlib import Path

def paired_bootstrap(a, b, *, seed=17, reps=5000):
    common_seeds=sorted(set(a)&set(b)); common_q=sorted(set().union(*(set(a[s])&set(b[s]) for s in common_seeds)))
    point_vals=[a[s][q]-b[s][q] for s in common_seeds for q in common_q if q in a[s] and q in b[s]]
    if not point_vals: return {"point":None,"lower":None,"upper":None,"replicates":0,"method":"paired_two_way_bootstrap"}
    point=sum(point_vals)/len(point_vals); rng=random.Random(seed); draws=[]
    for _ in range(max(5000,int(reps))):
        ss=[rng.choice(common_seeds) for _ in common_seeds]; qq=[rng.choice(common_q) for _ in common_q]
        vals=[a[s][q]-b[s][q] for s in ss for q in qq if q in a[s] and q in b[s]]
        if vals: draws.append(sum(vals)/len(vals))
    draws.sort(); return {"point":point,"lower":draws[max(0,int(.025*len(draws))-1)],"upper":draws[min(len(draws)-1,int(.975*len(draws)))],"replicates":len(draws),"seed_count":len(common_seeds),"question_count":len(common_q),"method":"paired_two_way_bootstrap_percentile_95"}

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--result",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--split",default="tune"); p.add_argument("--bootstrap-replicates",type=int,default=5000); a=p.parse_args(argv)
    result=json.loads(a.result.read_text(encoding="utf-8")); records=result.get("records",[]); methods=[m for m in result.get("methods",[]) if m!="SFT"]
    matrices={m:{ } for m in methods}; pair_matrices={m:{} for m in methods}; state={m:[] for m in methods}
    for m in methods:
        for row in records:
            if row.get("method")!=m or row.get("split")!=a.split: continue
            seed=int(row["seed"]); matrices[m][seed]={str(q["question_id"]):float(q.get("normalized_regret",q.get("mean_regret",0.0))) for q in row.get("question_metric_rows",[]) if q.get("normalized_regret") is not None}
            pair_matrices[m][seed]={str(q["question_id"]):float(q["pairwise_reversal_ranking_accuracy"]) for q in row.get("question_metric_rows",[]) if q.get("pairwise_reversal_ranking_accuracy") is not None}
            state[m].append({"state_macro_f1":row.get("state_macro_f1"),"invalid_recall":row.get("invalid_recall"),"insufficient_recall":row.get("insufficient_recall"),"state_calibration_ece":row.get("state_calibration_ece"),"seed":seed})
    comparisons={}
    pairs=(
        ("PESCO-Full","GRPO-Atomic","full_vs_atomic_base_regret"),
        ("PESCO-Full","Atomic+Branch","full_vs_no_state_flip_regret"),
        ("PESCO-Full","Atomic+Flip","full_vs_no_branch_regret"),
        ("PESCO-Full","Atomic+State+Flip","full_vs_no_branch_regret_alt"),
        ("Atomic+Branch","GRPO-Atomic","branch_marginal"),
        ("Atomic+Flip","GRPO-Atomic","flip_marginal"),
    )
    for left,right,name in pairs:
        comparisons[name]={"regret_delta_left_minus_right":paired_bootstrap(matrices.get(left,{}),matrices.get(right,{}),seed=101+len(comparisons),reps=a.bootstrap_replicates),"pairrank_delta_left_minus_right":paired_bootstrap(pair_matrices.get(left,{}),pair_matrices.get(right,{}),seed=301+len(comparisons),reps=a.bootstrap_replicates)}
    means={}
    for m in methods:
        vals=[v for s in matrices[m].values() for v in s.values()]; means[m]={"mean_normalized_regret":sum(vals)/len(vals) if vals else None,"state_macro_f1":sum(float(x["state_macro_f1"]) for x in state[m] if x["state_macro_f1"] is not None)/max(1,sum(x["state_macro_f1"] is not None for x in state[m])) if state[m] else None,"invalid_recall":sum(float(x["invalid_recall"]) for x in state[m] if x["invalid_recall"] is not None)/max(1,sum(x["invalid_recall"] is not None for x in state[m])) if state[m] else None,"insufficient_recall":sum(float(x["insufficient_recall"]) for x in state[m] if x["insufficient_recall"] is not None)/max(1,sum(x["insufficient_recall"] is not None for x in state[m])) if state[m] else None}
    out={"schema_version":"pesco_tier1_p232_paired_aggregate_v0.1","split":a.split,"diagnostic_only":True,"formal_comparison_authorized":False,"bootstrap": {"type":"paired_two_way_seed_question_cluster","replicates":a.bootstrap_replicates,"same_seed_draw_for_each_method":True,"same_question_draw_for_each_method":True},"methods":means,"comparisons":comparisons}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,ensure_ascii=False,sort_keys=True),encoding="utf-8"); print(json.dumps({"output":str(a.output),"comparisons":list(comparisons)},ensure_ascii=False)); return 0
if __name__=="__main__": raise SystemExit(main())
