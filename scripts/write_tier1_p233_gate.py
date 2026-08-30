#!/usr/bin/env python3
"""Compute the P2.3.3 gate exclusively from immutable experiment receipts."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path

def _finite(x):
 try:return math.isfinite(float(x))
 except (TypeError,ValueError):return False
def _upper(agg,name): return agg.get('comparisons',{}).get(name,{}).get('regret_delta_left_minus_right',{}).get('upper')
def _lower(agg,name): return agg.get('comparisons',{}).get(name,{}).get('pairrank_delta_left_minus_right',{}).get('lower')
def _rows(result,method,split): return [r for r in result.get('records',[]) if r.get('method')==method and r.get('split')==split]

def build_gate(matrix, aggregate, convergence, shortcut, estimator, stability=None, *, split='promotion'):
 methods=[m for m in matrix.get('methods',[]) if m!='SFT']; logs=matrix.get('training_logs',{})
 rl_atomic=bool(methods and matrix.get('seeds')) and all(bool(logs.get(str(s),{}).get(m,{}).get('atomic_reward_shared',False)) for s in matrix.get('seeds',[]) for m in methods)
 budgets=[logs.get(str(s),{}).get(m,{}).get('budget_contract',{}) for s in matrix.get('seeds',[]) for m in methods]
 budget_keys=('policy_rollout_calls','counterfactual_branch_calls','exploration_seed_executions','confirmation_seed_executions','optimizer_steps','forward_backward_flops')
 budget_equal=bool(budgets) and all(all(_finite(b.get(k)) and float(b.get(k))>0 for k in budget_keys) for b in budgets) and len({tuple(b.get(k) for k in budget_keys) for b in budgets})==1
 full_rows=_rows(matrix,'PESCO-Full',split); state_rows=_rows(matrix,'Atomic+State',split); full_state_rows=_rows(matrix,'Atomic+State+Flip',split)
 def avg(rows,key):
  x=[float(r[key]) for r in rows if _finite(r.get(key))]; return sum(x)/len(x) if x else None
 f1=avg(full_rows,'state_macro_f1'); sf1=avg(state_rows,'state_macro_f1');
 recs=[r.get('state_recall',{}) for r in full_rows]; min_rec=min((float(v) for r in recs for v in r.values() if _finite(v)),default=-1.0)
 invalid=avg(full_rows,'invalid_recall');
 shortcut_models=shortcut.get('models',{}); shortcut_vals=[]
 for name,v in shortcut_models.items():
  if name.startswith('without_confirmation:') and split in v.get('metrics_by_split',{}): shortcut_vals.append(float(v['metrics_by_split'][split].get('normalized_regret',float('inf'))))
 best_shortcut=min(shortcut_vals) if shortcut_vals else float('inf')
 full_mean=aggregate.get('methods',{}).get('PESCO-Full',{}).get('mean_normalized_regret',float('inf'))
 nonfull=[v.get('mean_normalized_regret') for k,v in aggregate.get('methods',{}).items() if k not in {'PESCO-Full','Oracle-Branch-Search'} and _finite(v.get('mean_normalized_regret'))]
 best_nonfull=min(nonfull) if nonfull else float('inf')
 dirs=aggregate.get('direction_summary',{}).get('full_vs_atomic',{}).get('positive_seed_fraction',0.0)
 family=aggregate.get('family_leave_one_out',{}); fam_positive=sum(1 for vals in family.values() if _finite(vals.get('PESCO-Full')) and _finite(vals.get('Atomic+State+Flip')) and vals['PESCO-Full']<vals['Atomic+State+Flip'])
 conv_records = list(convergence.get('records', []))
 max_step = max((int(r.get('steps', 0)) for r in conv_records), default=0)
 max_conv_rows = [r for r in conv_records if int(r.get('steps', 0)) == max_step]
 required_conv_methods = {'GRPO-Atomic', 'Atomic+Branch', 'PESCO-Full'}
 conv_method_set = {str(r.get('method')) for r in max_conv_rows}
 full_validity = avg(full_rows, 'erroneous_repair_rate')
 baseline_validity = avg(_rows(matrix, 'Atomic+State+Flip', split), 'erroneous_repair_rate')
 full_replication = avg(full_rows, 'confirmation_rate')
 baseline_replication = avg(_rows(matrix, 'Atomic+State+Flip', split), 'confirmation_rate')
 gates={
  'no_hidden_truth_used_by_estimator': bool(estimator.get('no_hidden_truth_used_by_estimator') and estimator.get('forced_hidden_truth_invariance')),
  'all_gate_values_derived_from_receipts': bool(aggregate.get('receipt_derived') is True and matrix.get('method_matrix_contract',{}).get('sft_shared_checkpoint') is True),
  'all_methods_share_atomic_reward': rl_atomic,
  'convergence_rule_frozen_and_passed': bool(convergence.get('gates',{}).get('all_steps_executed') and convergence.get('gates',{}).get('all_seeds_executed', True) and max_conv_rows and required_conv_methods.issubset(conv_method_set) and all(bool(r.get('plateau_gate')) and float(r.get('tail_entropy',0.0))>=0.20 and float(r.get('tail_kl',float('inf')))<2.0 for r in max_conv_rows)),
  'branch_factorial_effect_regret_ci_upper_lt_zero': all(_finite(_upper(aggregate,n)) and float(_upper(aggregate,n))<0 for n in ('AB-A','ASB-AS','ABF-AF')),
  'full_vs_atomic_state_flip_regret_ci_upper_lt_zero': _finite(_upper(aggregate,'Full-ASF')) and float(_upper(aggregate,'Full-ASF'))<0,
  'flip_pairrank_delta_ci_lower_gt_zero': _finite(_lower(aggregate,'flip_vs_atomic')) and float(_lower(aggregate,'flip_vs_atomic'))>0,
  'flip_does_not_harm_regret': _finite(_upper(aggregate,'flip_vs_atomic')) and float(_upper(aggregate,'flip_vs_atomic'))<=0,
  'full_state_macro_f1_noninferior_to_atomic_state': f1 is not None and sf1 is not None and f1+0.02>=sf1,
  'min_per_state_recall_pass': min_rec>=0.10,
  'invalid_recall_absolute_floor_pass': invalid is not None and invalid>=0.05,
  'full_vs_best_nonfull_regret_ci_upper_lt_zero': _finite(_upper(aggregate,'full_vs_best_nonfull')) and float(_upper(aggregate,'full_vs_best_nonfull'))<0,
  'full_beats_strict_shortcut_baseline': full_mean<best_shortcut,
  'at_least_8_of_10_seed_directions_positive': float(dirs)>=0.8,
  'family_majority_direction_positive': fam_positive>=(len(family)+1)//2 if family else False,
  'leave_one_family_out_robust': fam_positive>=max(1,len(family)-1) if family else False,
  'environment_execution_budget_matched': budget_equal,
  'selected_action_validity_noninferior': full_validity is not None and baseline_validity is not None and full_validity <= min(0.05, baseline_validity + 1e-12),
  'selected_action_replication_noninferior': full_replication is not None and baseline_replication is not None and full_replication >= max(0.70, baseline_replication - 1e-12),
  'reward_weight_stability_pass': bool((stability or matrix.get('reward_weight_stability',{})).get('pass',False)),
 }
 payload={'schema_version':'pesco_p233_go_v0.1','p233_go':gates,'status':'GO' if all(gates.values()) else 'NO_GO','split':split,'receipt_sources':{'matrix':'p233_matrix_result.json','aggregate':'p233_aggregate.json','convergence':'convergence_summary.json','shortcut':'shortcut_probe_result.json','estimator':'estimator_audit.json'},'thresholds':{'state_f1_noninferiority_margin':0.02,'min_per_state_recall':0.10,'invalid_recall_floor':0.05,'replication_floor':0.70}}
 payload['audit_sha256']='sha256:'+hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
 return payload

def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument('--matrix',type=Path,required=True); p.add_argument('--aggregate',type=Path,required=True); p.add_argument('--convergence',type=Path,required=True); p.add_argument('--shortcut',type=Path,required=True); p.add_argument('--estimator',type=Path,required=True); p.add_argument('--stability',type=Path,required=False); p.add_argument('--output',type=Path,required=True); p.add_argument('--split',default='promotion'); a=p.parse_args(argv)
 out=build_gate(json.loads(a.matrix.read_text()),json.loads(a.aggregate.read_text()),json.loads(a.convergence.read_text()),json.loads(a.shortcut.read_text()),json.loads(a.estimator.read_text()),json.loads(a.stability.read_text()) if a.stability else None,split=a.split); a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,ensure_ascii=False,sort_keys=True)); print(json.dumps({'output':str(a.output),'status':out['status'],'p233_go':out['p233_go']},indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
