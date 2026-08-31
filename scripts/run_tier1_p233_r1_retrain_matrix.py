#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, subprocess, sys, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SEEDS=(17,23,29,31,37,41,43,47,53,59)
PURE=('GRPO-Atomic','Atomic+State','Atomic+Branch','Atomic+Flip','Atomic+State+Branch','Atomic+State+Flip','Atomic+Branch+Flip','PESCO-Full')
def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument('--dataset',type=Path,required=True); p.add_argument('--base-matrix',type=Path,required=True); p.add_argument('--sft-root',type=Path,required=True); p.add_argument('--root',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--steps',type=int,default=1024); a=p.parse_args(argv); a.root.mkdir(parents=True,exist_ok=True)
 for seed in SEEDS:
  out=a.root/f'seed_{seed}'/'p233_result.json'
  if out.exists(): continue
  out.parent.mkdir(parents=True,exist_ok=True); cmd=[sys.executable,str(ROOT/'scripts/run_tier1_p233_r1_retrain_chunk.py'),'--dataset',str(a.dataset),'--sft-checkpoint',str(a.sft_root/f'seed_{seed}'/'sft_checkpoints'/f'sft_seed_{seed}.pt'),'--output',str(out),'--seed',str(seed),'--steps',str(a.steps)]; print('retrain',seed,flush=True); subprocess.run(cmd,check=True)
 base=json.loads(a.base_matrix.read_text()); pure_records=[]; pure_logs={}; pure_ck={}
 for seed in SEEDS:
  d=json.loads((a.root/f'seed_{seed}'/'p233_result.json').read_text()); pure_records.extend(d['records']); pure_logs.update(d['training_logs']); pure_ck.update(d.get('sft_checkpoints',{}))
 keep=[r for r in base['records'] if r.get('method') not in PURE]; records=keep+pure_records; logs=dict(base.get('training_logs',{})); logs.update(pure_logs); ck=dict(base.get('sft_checkpoints',{})); ck.update(pure_ck); config=dict(base.get('config',{})); config.update({'finetune_steps':a.steps,'branch_loss_weight':.2,'pairwise_weight':.2,'flip_reference_kl_weight':0.,'branch_formulation':'expected_utility','authentic_factorial':True}); cfgdig='sha256:'+hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest(); config['config_digest']=cfgdig
 out={'schema_version':'pesco_tier1_p233_r1_retrained_matrix_v1','seeds':list(SEEDS),'methods':base['methods'],'eval_splits':['tune','promotion'],'config':config,'config_digest':cfgdig,'canonical_pair_digest':base['canonical_pair_digest'],'canonical_pair_count':base['canonical_pair_count'],'canonical_pair_contract':base.get('canonical_pair_contract',{}),'reward_tensor_audit':base.get('reward_tensor_audit',{}),'optimizer_contract':base.get('optimizer_contract',{}),'records':records,'training_logs':logs,'sft_checkpoints':ck,'oracle_branch_search':base.get('oracle_branch_search',[]),'method_matrix_contract':dict(base.get('method_matrix_contract',{}),authentic_factorial=True,only_factor_switches_differ=True,external_action_adapter_forbidden=True,common_training_budget=int(a.steps),retrained_weight_sensitivity=True),'diagnostic_only':True,'formal_comparison_authorized':False}; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,ensure_ascii=False,sort_keys=True)); print({'output':str(a.output),'records':len(records)})
if __name__=='__main__': raise SystemExit(main())
