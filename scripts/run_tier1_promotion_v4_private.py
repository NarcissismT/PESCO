#!/usr/bin/env python3
"""Independent post-freeze promotion-v4 evaluator (private input/output)."""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from research_strategy_optimization.algorithms.differentiable_strategy import DecisionDataset
from research_strategy_optimization.evaluation.tier1_p231_diagnostics import P231Config, run_p231_dev_diagnostic
METHODS=("GRPO-Atomic","Atomic+State","Atomic+Branch","Atomic+Flip","Atomic+State+Branch","Atomic+State+Flip","Atomic+Branch+Flip","PESCO-Full")
SEEDS=(17,23,29,31,37,41,43,47,53,59)
def digest(path): return "sha256:"+hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument("--dataset",type=Path,default=ROOT/"artifacts/tier1_p23_promotion_v4_private/dataset_raw_evidence.json"); p.add_argument("--freeze",type=Path,default=ROOT/"artifacts/tier1_p232_promotion_v4_public/promotion_v4_freeze_receipt.json"); p.add_argument("--output-dir",type=Path,default=ROOT/"artifacts/tier1_p23_promotion_v4_private/final_evaluation"); a=p.parse_args(argv)
 freeze=json.loads(a.freeze.read_text()); head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip();
 if freeze.get("status")!="FROZEN_FOR_PROMOTION_V4" or freeze.get("commit")!=head: raise SystemExit("promotion-v4 freeze/head mismatch")
 sentinel=a.output_dir/"access_sentinel.json"; a.output_dir.mkdir(parents=True,exist_ok=True)
 if sentinel.exists(): raise SystemExit("promotion-v4 access sentinel already exists; one-time evaluation refused")
 dataset=DecisionDataset.from_json(a.dataset); cfg=P231Config(sft_steps=16,finetune_steps=64,batch_size=32,hidden_dim=24,learning_rate=0.002,minibatch_epochs=2,top1_gap_threshold=0.05)
 result=run_p231_dev_diagnostic(a.output_dir,dataset,seeds=SEEDS,config=cfg,methods=METHODS,eval_splits=("final_id","final_ood")); result.update({"diagnostic_only":False,"formal_comparison_authorized":True,"private_final":True,"freeze_receipt_digest":digest(a.freeze),"private_dataset_digest":digest(a.dataset),"final_access":{"locked":True,"first_access_completed":True,"independent_evaluator_process":True,"single_evaluation":True,"training_split":"train_only","evaluation_splits":["final_id","final_ood"],"final_labels_never_used_for_training":True}}); (a.output_dir/"promotion_v4_final_result.json").write_text(json.dumps(result,indent=2,ensure_ascii=False,sort_keys=True)); sentinel.write_text(json.dumps({"schema_version":"pesco_promotion_v4_access_sentinel_v0.1","first_access_completed":True,"evaluator_input_digest":digest(a.dataset),"freeze_digest":digest(a.freeze)},indent=2,sort_keys=True)); (a.output_dir/"final_access_receipt.json").write_text(json.dumps({"schema_version":"pesco_promotion_v4_final_access_receipt_v0.1","single_evaluation":True,"evaluator_input_digest":digest(a.dataset),"freeze_receipt_digest":digest(a.freeze),"training_split":"train_only","evaluation_splits":["final_id","final_ood"],"final_labels_never_used_for_training":True},indent=2,sort_keys=True)); print(json.dumps({"output":str(a.output_dir),"seed_count":10,"methods":list(METHODS),"eval_splits":["final_id","final_ood"],"canonical_pair_digest":result["canonical_pair_digest"]},ensure_ascii=False,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
