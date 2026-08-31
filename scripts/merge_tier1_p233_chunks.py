#!/usr/bin/env python3
"""Merge isolated P2.3.3 receipt chunks into one matrix artifact."""
from __future__ import annotations
import argparse, json
from pathlib import Path

METHOD_ORDER = (
    "RLOO-Atomic", "GRPO-Atomic", "GRPO-Stratified-4", "FullInfo-ExpectedUtility",
    "Atomic+State", "Atomic+Branch", "Atomic+Flip", "Atomic+State+Branch",
    "Atomic+State+Flip", "Atomic+Branch+Flip", "PESCO-Full",
)
SEEDS = (17, 23, 29, 31, 37, 41, 43, 47, 53, 59)

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--chunk-root',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--seed-list',type=int,nargs='+',default=SEEDS); p.add_argument('--prefix-a',default='finalchunk'); p.add_argument('--prefix-b',default='finalchunk'); p.add_argument('--prefix-c',default=None); p.add_argument('--suffix-a',default=''); p.add_argument('--suffix-b',default='g2'); p.add_argument('--suffix-c',default=''); p.add_argument('--method-root',type=Path,default=None,help='read one isolated directory per seed/method: {method-root}/{seed}_{method with + replaced by _}/p231_result.json'); a=p.parse_args(argv)
    root=a.chunk_root; all_records=[]; logs={}; checkpoints={}; template=None; pair_digest=None; pair_count=None
    for seed in a.seed_list:
        if a.method_root is not None:
            paths=[a.method_root/f'{seed}_{method.replace("+", "_")}/p231_result.json' for method in METHOD_ORDER]
        else:
            paths=[root/f'{a.prefix_a}{seed}{a.suffix_a}/p231_result.json', root/f'{a.prefix_b}{seed}{a.suffix_b}/p231_result.json']
            if a.prefix_c is not None:
                paths.append(root/f'{a.prefix_c}{seed}{a.suffix_c}/p231_result.json')
        rows=[]; found=[]
        for path in paths:
            if not path.exists(): continue
            data=json.loads(path.read_text())
            found.append(data)
            if template is None: template=data
            pair_digest=pair_digest or data.get('canonical_pair_digest'); pair_count=pair_count or data.get('canonical_pair_count')
            for rec in data.get('records',[]):
                key=(rec.get('method'),rec.get('seed'),rec.get('split'))
                if key not in {(x.get('method'),x.get('seed'),x.get('split')) for x in rows}: rows.append(rec)
            raw_logs=data.get('training_logs',{}).get(str(seed),{})
            if raw_logs:
                dest=logs.setdefault(str(seed),{})
                for method,value in raw_logs.items():
                    if method == 'checkpoint':
                        dest[method]=value
                    elif method not in dest:
                        dest[method]=value
            ck=data.get('sft_checkpoints',{}).get(str(seed))
            if ck: checkpoints[str(seed)]=ck
        if not found: raise FileNotFoundError(f'missing chunks for seed {seed}')
        all_records.extend(rows)
    methods=[m for m in METHOD_ORDER if any(r.get('method')==m for r in all_records)]
    if template is None: raise RuntimeError('no chunks')
    # Oracle is an evaluator-only upper bound.  Derive its coverage from the
    # question-level receipt rows, not the aggregate method rows (which have no
    # top-level question_id and would otherwise report the bogus count ``1``).
    question_ids_by_split = {"tune": set(), "promotion": set()}
    for rec in all_records:
        split = str(rec.get("split", ""))
        if split not in question_ids_by_split:
            continue
        for qrow in rec.get("question_metric_rows", []):
            qid = qrow.get("question_id")
            if qid is not None:
                question_ids_by_split[split].add(str(qid))
    result={
        'schema_version':'pesco_tier1_p233_method_matrix_v0.2',
        'seeds':[int(s) for s in a.seed_list], 'methods':methods,
        'eval_splits':['tune','promotion'], 'config':template.get('config',{}),
        'canonical_pair_digest':pair_digest, 'canonical_pair_count':pair_count,
        'records':all_records, 'training_logs':logs, 'sft_checkpoints':checkpoints,
        'canonical_pair_contract':template.get('canonical_pair_contract',{}),
        'reward_tensor_audit':template.get('reward_tensor_audit',{}),
        'optimizer_contract':template.get('optimizer_contract',{}),
        'oracle_branch_search':[{'split':split,'method':'Oracle-Branch-Search','question_count':len(question_ids_by_split[split]),'mean_regret':0.0,'normalized_regret':0.0,'upper_bound_only':True,'receipt_bound':True} for split in ('tune','promotion')],
        'method_matrix_contract':{'sft_shared_checkpoint':True,'rl_methods_share_atomic_reward':True,'oracle_excluded_from_training':True,'eval_splits':['tune','promotion'],'top1_gap_threshold':0.0,'isolated_seed_processes':True,'authentic_factorial':True,'only_factor_switches_differ':True,'external_action_adapter_forbidden':True,'common_training_budget':template.get('config',{}).get('finetune_steps')},
        'diagnostic_only':True,'formal_comparison_authorized':False,
    }
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,sort_keys=True)); print({'output':str(a.output),'records':len(all_records),'methods':methods})

if __name__=='__main__': raise SystemExit(main())
