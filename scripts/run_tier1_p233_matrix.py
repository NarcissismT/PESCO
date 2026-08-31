#!/usr/bin/env python3
"""Run the receipt-bound P2.3.3 CPU method matrix on fresh diagnostic data."""
from __future__ import annotations
import argparse, hashlib, json, sys
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from research_strategy_optimization.algorithms.differentiable_strategy import DecisionDataset
from research_strategy_optimization.evaluation.tier1_p231_diagnostics import P231Config, run_p231_dev_diagnostic
from research_strategy_optimization.utils.run_manifest import build_run_manifest, write_run_manifest

METHODS=("RLOO-Atomic","GRPO-Atomic","GRPO-Stratified-4","FullInfo-ExpectedUtility","Atomic+State","Atomic+Branch","Atomic+Flip","Atomic+State+Branch","Atomic+State+Flip","Atomic+Branch+Flip","PESCO-Full")
SEEDS=(17,23,29,31,37,41,43,47,53,59)

def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument('--dataset',type=Path,default=ROOT/'artifacts/tier1_p233_diagnostic/dataset_raw_evidence.json'); p.add_argument('--output-dir',type=Path,default=ROOT/'artifacts/tier1_p233_matrix_10seed'); p.add_argument('--finetune-steps',type=int,default=64); p.add_argument('--sft-steps',type=int,default=1024); p.add_argument('--batch-size',type=int,default=32); p.add_argument('--hidden-dim',type=int,default=128); p.add_argument('--learning-rate',type=float,default=2e-3); p.add_argument('--sft-learning-rate',type=float,default=3e-3); p.add_argument('--branch-loss-weight',type=float,default=0.30); p.add_argument('--utility-target-weight',type=float,default=0.25); p.add_argument('--pairwise-weight',type=float,default=0.10); p.add_argument('--state-invalid-weight',type=float,default=3.0); p.add_argument('--state-weight',type=float,default=0.2); p.add_argument('--gradient-mode',choices=('sum','pcgrad'),default='sum'); p.add_argument('--branch-formulation',choices=('sibling_advantage','expected_utility'),default='expected_utility'); p.add_argument('--flip-reference-kl-weight',type=float,default=0.5); p.add_argument('--methods',nargs='+',default=METHODS); p.add_argument('--seeds',type=int,nargs='+',default=SEEDS); a=p.parse_args(argv)
 d=DecisionDataset.from_json(a.dataset); cfg=P231Config(sft_steps=a.sft_steps,finetune_steps=a.finetune_steps,batch_size=a.batch_size,hidden_dim=a.hidden_dim,learning_rate=a.learning_rate,sft_learning_rate=a.sft_learning_rate,top1_gap_threshold=0.0,gradient_mode=a.gradient_mode,branch_formulation=a.branch_formulation,state_weight=a.state_weight,branch_loss_weight=a.branch_loss_weight,utility_target_weight=a.utility_target_weight,pairwise_weight=a.pairwise_weight,state_class_weights=(a.state_invalid_weight,1.0,1.0,1.0),flip_reference_kl_weight=a.flip_reference_kl_weight); cfg=P231Config(**{**asdict(cfg),'authentic_factorial':True})
 a.output_dir.mkdir(parents=True,exist_ok=True); result=run_p231_dev_diagnostic(a.output_dir,d,seeds=tuple(a.seeds),config=cfg,methods=tuple(a.methods),eval_splits=('tune','promotion'))
 # Oracle branch search is an evaluator-only upper bound, never a trainable method.
 oracle=[]
 for split in ('tune','promotion'):
  rows=[e for e in d.examples if e.split==split]; oracle.append({'split':split,'method':'Oracle-Branch-Search','question_count':len(set(e.question_id for e in rows)),'mean_regret':0.0,'normalized_regret':0.0,'upper_bound_only':True,'receipt_bound':True})
 result['oracle_branch_search']=oracle; result['method_matrix_contract']={'sft_shared_checkpoint':True,'rl_methods_share_atomic_reward':True,'oracle_excluded_from_training':True,'eval_splits':['tune','promotion'],'top1_gap_threshold':0.0,'authentic_factorial':True,'only_factor_switches_differ':True,'external_action_adapter_forbidden':True,'common_training_budget':int(a.finetune_steps)}
 (a.output_dir/'p233_matrix_result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,sort_keys=True),encoding='utf-8')
 manifest=build_run_manifest(experiment='tier1_p233_method_matrix',repo_root=ROOT,command=sys.argv,runner_paths=[ROOT/'scripts/run_tier1_p233_matrix.py',ROOT/'research_strategy_optimization/evaluation/tier1_p231_diagnostics.py'],data_paths=[a.dataset,a.output_dir/'canonical_reversal_ids.json',a.output_dir/'reward_tensor_audit.json',a.output_dir/'p233_matrix_result.json'],seeds={'training':list(a.seeds)},checkpoint=None,status='completed_diagnostic',diagnostics={'diagnostic_only':True,'formal_comparison_authorized':False,'method_matrix':['SFT',*list(a.methods)],'sft_evaluated_in_records':True,'oracle_upper_bound_only':True,'canonical_pair_digest':result['canonical_pair_digest'],'finetune_steps':a.finetune_steps})
 write_run_manifest(a.output_dir/'run_manifest.json',manifest); print(json.dumps({'output':str(a.output_dir),'canonical_pair_count':result['canonical_pair_count'],'methods':list(a.methods)},indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
