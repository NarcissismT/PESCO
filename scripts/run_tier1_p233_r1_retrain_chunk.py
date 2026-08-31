#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from research_strategy_optimization.algorithms.differentiable_strategy import DecisionDataset
import research_strategy_optimization.evaluation.tier1_p231_diagnostics as p

def main(argv=None):
    a=argparse.ArgumentParser(); a.add_argument('--dataset',type=Path,required=True); a.add_argument('--sft-checkpoint',type=Path,required=True); a.add_argument('--output',type=Path,required=True); a.add_argument('--seed',type=int,required=True); a.add_argument('--steps',type=int,default=1024); a=a.parse_args(argv)
    d=DecisionDataset.from_json(a.dataset); cfg=p.P231Config(seed=a.seed,sft_steps=256,finetune_steps=a.steps,batch_size=64,hidden_dim=64,learning_rate=.002,branch_loss_weight=.2,pairwise_weight=.2,flip_reference_kl_weight=0.,authentic_factorial=True,branch_formulation='expected_utility'); rows,pay=p.canonical_pair_payload(d,cfg); pairs=p._pair_objects(d,rows); sft,_=p.load_sft_checkpoint(a.sft_checkpoint,hidden_dim=64); records=[]; logs={}
    for method in ('GRPO-Atomic','Atomic+State','Atomic+Branch','Atomic+Flip','Atomic+State+Branch','Atomic+State+Flip','Atomic+Branch+Flip','PESCO-Full'):
        pol,log=p.fit_rollout_method_authentic_factorial(d,sft,method,cfg,pairs); logs[method]=log
        for split in ('tune','promotion'):
            r=p.evaluate_canonical_policy(pol,d,split,pairs,canonical_pair_digest=pay['canonical_pair_digest']); r.update({'method':method,'seed':a.seed,'split':split,'initial_sft_digest':log['initial_sft_digest'],'final_policy_digest':log['final_policy_digest']}); records.append(r)
    out={'schema_version':'pesco_p233_r1_retrain_chunk_v1','seeds':[a.seed],'methods':['SFT','GRPO-Atomic','Atomic+State','Atomic+Branch','Atomic+Flip','Atomic+State+Branch','Atomic+State+Flip','Atomic+Branch+Flip','PESCO-Full'],'config':{**p.asdict(cfg),'config_digest':'r1-set3'},'canonical_pair_digest':pay['canonical_pair_digest'],'canonical_pair_count':len(rows),'canonical_pair_contract':p.verify_canonical_pair_payload(d,pay,cfg),'records':records,'training_logs':{str(a.seed):logs},'sft_checkpoints':{str(a.seed):{'checkpoint':str(a.sft_checkpoint),'state_dict_sha256':log['initial_sft_digest']}}}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,ensure_ascii=False,sort_keys=True)); print({'seed':a.seed,'records':len(records)})
if __name__=='__main__': raise SystemExit(main())
