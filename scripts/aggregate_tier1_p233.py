#!/usr/bin/env python3
"""Receipt-only P2.3.3 paired seed×question aggregate and family audit."""
from __future__ import annotations
import argparse, json, random
from collections import defaultdict
from pathlib import Path

def boot(a,b,reps=5000,seed=233):
    seeds=sorted(set(a)&set(b)); qs=sorted(set().union(*(set(a[s])&set(b[s]) for s in seeds))) if seeds else []
    vals=[a[s][q]-b[s][q] for s in seeds for q in qs if q in a[s] and q in b[s]]
    if not vals:return {'point':None,'lower':None,'upper':None,'seed_count':0,'question_count':0,'replicates':0,'aggregation':'seed_x_question_two_way_bootstrap'}
    rng=random.Random(seed); draws=[]
    for _ in range(max(100,int(reps))):
        ss=[rng.choice(seeds) for _ in seeds]; qq=[rng.choice(qs) for _ in qs]; x=[a[s][q]-b[s][q] for s in ss for q in qq if q in a[s] and q in b[s]]; draws.append(sum(x)/len(x))
    draws.sort(); lo=draws[max(0,int(.025*len(draws))-1)]; hi=draws[min(len(draws)-1,int(.975*len(draws)))]
    return {'point':sum(vals)/len(vals),'lower':lo,'upper':hi,'seed_count':len(seeds),'question_count':len(qs),'replicates':len(draws),'aggregation':'seed_x_question_two_way_bootstrap'}

def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument('--result',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--split',default='tune'); p.add_argument('--bootstrap-replicates',type=int,default=5000); p.add_argument('--selection-aggregate',type=Path,default=None,help='tune aggregate used to freeze baseline selection before promotion'); a=p.parse_args(argv)
 d=json.loads(a.result.read_text()); records=d['records']; methods=[m for m in d.get('methods',[]) if m!='SFT']; mat={m:{} for m in methods}; pair={m:{} for m in methods}; fam={m:defaultdict(lambda:defaultdict(list)) for m in methods}; state={m:[] for m in methods}; budgets={}
 for row in records:
  if row.get('split')!=a.split or row.get('method') not in mat: continue
  m=row['method']; s=int(row['seed']); mat[m][s]={q['question_id']:float(q.get('normalized_regret',q.get('mean_regret',0))) for q in row.get('question_metric_rows',[]) if q.get('normalized_regret',q.get('mean_regret')) is not None}; pair[m][s]={q['question_id']:float(q.get('pairwise_reversal_ranking_accuracy')) for q in row.get('question_metric_rows',[]) if q.get('pairwise_reversal_ranking_accuracy') is not None}; budgets.setdefault(m,[]).append(d.get('training_logs',{}).get(str(s),{}).get(m,{}).get('budget_contract',{})); state[m].append(row)
  for q in row.get('question_metric_rows',[]): fam[m][q.get('family','unknown')][s].append(float(q.get('normalized_regret',q.get('mean_regret',0))))
 comparisons={}
 def cmp(l,r,name): comparisons[name]={'left':l,'right':r,'regret_delta_left_minus_right':boot(mat.get(l,{}),mat.get(r,{}),a.bootstrap_replicates,233+len(comparisons)),'pairrank_delta_left_minus_right':boot(pair.get(l,{}),pair.get(r,{}),a.bootstrap_replicates,733+len(comparisons))}
 for l,r,n in [('Atomic+Branch','GRPO-Atomic','AB-A'),('Atomic+State+Branch','Atomic+State','ASB-AS'),('Atomic+Branch+Flip','Atomic+Flip','ABF-AF'),('PESCO-Full','Atomic+State+Flip','Full-ASF'),('Atomic+Flip','GRPO-Atomic','flip_vs_atomic'),('PESCO-Full','GRPO-Atomic','full_vs_atomic')]: cmp(l,r,n)
 means={m:{'mean_normalized_regret':sum(v for x in mat[m].values() for v in x.values())/max(1,sum(len(x) for x in mat[m].values())),'mean_pairrank':sum(v for x in pair[m].values() for v in x.values())/max(1,sum(len(x) for x in pair[m].values()))} for m in methods}
 if a.selection_aggregate is not None:
  selected=json.loads(a.selection_aggregate.read_text())
  selected_method=selected.get('best_nonfull_method')
  best_nonfull=selected_method if selected_method in methods and selected_method!='PESCO-Full' else min((m for m in methods if m!='PESCO-Full'), key=lambda m: means[m]['mean_normalized_regret'], default=None)
 else:
  best_nonfull=min((m for m in methods if m!='PESCO-Full'), key=lambda m: means[m]['mean_normalized_regret'], default=None)
 if best_nonfull is not None:
  cmp('PESCO-Full',best_nonfull,'full_vs_best_nonfull')
 family_loo={}
 for family in sorted(set(k for m in fam.values() for k in m)):
  family_loo[family]={}
  for m in methods:
   vals=[v for q in fam[m].get(family,{}).values() for v in q]; family_loo[family][m]=sum(vals)/len(vals) if vals else None
 direction={}
 for name,c in comparisons.items():
  x=c['regret_delta_left_minus_right']; direction[name]={'point':x['point'],'positive_seed_fraction':sum(1 for s in sorted(set(mat.get(c['left'],{}))&set(mat.get(c['right'],{}))) if sum(mat[c['left']][s].values())/max(1,len(mat[c['left']][s])) < sum(mat[c['right']][s].values())/max(1,len(mat[c['right']][s])))/max(1,len(set(mat.get(c['left'],{}))&set(mat.get(c['right'],{}))))}
 out={'schema_version':'pesco_tier1_p233_receipt_aggregate_v0.1','diagnostic_only':True,'formal_comparison_authorized':False,'split':a.split,'receipt_derived':True,'methods':means,'best_nonfull_method':best_nonfull,'comparisons':comparisons,'family_leave_one_out':family_loo,'direction_summary':direction,'state_receipts':{m:[{'seed':r.get('seed'),'state_macro_f1':r.get('state_macro_f1'),'state_recall':r.get('state_recall'),'invalid_recall':r.get('invalid_recall'),'insufficient_recall':r.get('insufficient_recall'),'confirmation_rate':r.get('confirmation_rate'),'selected_action_validity':r.get('selected_action_validity_rate'),'selected_action_replication':r.get('confirmation_rate')} for r in rows] for m,rows in state.items()},'budget_receipts':budgets,'bootstrap':{'replicates':a.bootstrap_replicates,'same_seed_and_question_resampling':True}}
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,ensure_ascii=False,sort_keys=True)); print(json.dumps({'output':str(a.output),'comparisons':list(comparisons),'methods':methods},indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
