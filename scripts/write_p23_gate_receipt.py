#!/usr/bin/env python3
"""Write a fail-closed P2.3 frozen gate receipt from completed diagnostics."""
from __future__ import annotations
import argparse,json
from pathlib import Path

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--matrix',type=Path,required=True); ap.add_argument('--strict-shortcut',type=Path,required=True); ap.add_argument('--fallback-shortcut',type=Path,required=True); ap.add_argument('--reward-stability',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--target',default='GRPO+Branch+Flip'); args=ap.parse_args(argv)
    matrix=json.loads(args.matrix.read_text(encoding='utf-8')); strict=json.loads(args.strict_shortcut.read_text(encoding='utf-8')); fallback=json.loads(args.fallback_shortcut.read_text(encoding='utf-8')); reward=json.loads(args.reward_stability.read_text(encoding='utf-8'))
    gates=matrix.get('aggregation',{}).get('gate_checks',{}).get(args.target,{})
    summary=matrix.get('aggregation',{}).get('promotion_summary',{}); target=summary.get(args.target,{})
    strict_pass=strict.get('status')=='completed_sklearn' and not strict.get('fallback_used') and {v.get('model_name') for v in strict.get('models',{}).values() if v.get('status')=='completed'} >= {'logistic_regression','gradient_boosting'}
    fallback_regs={k:float(v.get('metrics_by_split',{}).get('promotion',{}).get('normalized_regret')) for k,v in fallback.get('models',{}).items() if v.get('metrics_by_split',{}).get('promotion',{}).get('normalized_regret') is not None}
    fallback_best=min(fallback_regs.values()) if fallback_regs else None
    safety=bool(target) and target.get('confirmation_rate') is not None and target.get('validity_rate') is not None and target.get('erroneous_repair_rate') is not None
    checks={
      'matrix_complete_10_seed':bool(matrix.get('matrix_complete')) and len(matrix.get('seeds',[]))==10,
      'regret_vs_frozen_baseline':bool(gates.get('regret_gate')),
      'regret_vs_grpo_matched_atomic':bool(gates.get('regret_vs_grpo_matched_atomic_gate')),
      'pairrank_vs_grpo_matched_atomic':bool(gates.get('pairrank_gate')),
      'seed_direction_8_of_10':int(gates.get('regret_positive_seed_n_vs_sft',0))>=8,
      'family_leave_one_out':bool(gates.get('family_leave_one_out_gate')),
      'required_switch_better_than_sft_and_noflip':bool(gates.get('required_switch_gate_vs_sft_and_noflip')),
      'strict_logistic_gbdt_sklearn':strict_pass,
      'beats_available_shortcut_diagnostic':bool(fallback_best is not None and target.get('mean_regret') is not None and float(target['mean_regret'])<fallback_best),
      'selected_action_safety_metrics_present':safety,
      'global_reward_weight_plus_minus_20_stable':reward.get('overall',{}).get('non_tie_stable_fraction')==1.0,
    }
    passed=all(checks.values())
    out={'schema_version':'pesco_tier1_p23_gate_receipt_v0.1','status':'GO_P3A' if passed else 'NO_GO_REMAIN_P2_3','target_method':args.target,'checks':checks,'target_summary':target,'target_gate_statistics':gates,'frozen_best_non_pesco':matrix.get('aggregation',{}).get('best_non_pesco_method'),'strict_shortcut_status':strict.get('status'),'fallback_shortcut_diagnostic_best_normalized_regret':fallback_best,'fallback_shortcut_formal_authorized':False,'p3a_authorized':passed,'p3b_authorized':False,'small_model_lora_authorized':passed,'online_rl_authorized':False,'formal_final_opened':False,'diagnostic_only':True}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(out,indent=2,ensure_ascii=False,sort_keys=True),encoding='utf-8');print(json.dumps(out,ensure_ascii=False,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
