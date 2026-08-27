#!/usr/bin/env python3
"""Build the evaluator-only train+final bundle without exposing final labels to training."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research_strategy_optimization.algorithms.differentiable_strategy import DecisionDataset

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--train',type=Path,required=True); ap.add_argument('--final',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args(argv)
    train=DecisionDataset.from_json(args.train); final=DecisionDataset.from_json(args.final)
    examples=[e for e in train.examples if e.split=='train']+list(final.examples)
    ds=DecisionDataset(examples,[], 'pesco_decision_dataset_p2.3_promotion_v3_combined_private', {'final_labels_never_used_for_training':True})
    args.output.parent.mkdir(parents=True,exist_ok=True); ds.save_json(args.output,include_audit=True)
    receipt={'schema_version':'pesco_promotion_v3_private_input_receipt_v0.1','train_examples':sum(e.split=='train' for e in examples),'final_examples':sum(e.split in {'final_id','final_ood'} for e in examples),'final_labels_never_used_for_training':True,'input_digest':'sha256:'+hashlib.sha256(args.output.read_bytes()).hexdigest()}
    args.output.with_name('private_input_receipt.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False,sort_keys=True),encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
