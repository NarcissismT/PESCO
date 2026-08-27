#!/usr/bin/env python3
"""Aggregate already completed P2.3 isolated cell artifacts without rerunning cells."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from research_strategy_optimization.evaluation.tier1_p23_diagnostics import _aggregate_p23, P23_METHODS
from research_strategy_optimization.utils.run_manifest import build_run_manifest, write_run_manifest

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--output-dir',type=Path,required=True); ap.add_argument('--stage',default='screening'); ap.add_argument('--seeds',type=int,nargs='+',required=True); ap.add_argument('--methods',nargs='+',default=list(P23_METHODS)); ap.add_argument('--bootstrap-replicates',type=int,default=500); args=ap.parse_args(argv)
    cells=args.output_dir/'cells'; records=[]; logs={}; failures=[]; seen_sft_splits=set(); pair_counts={}
    for path in sorted(cells.glob('*/p23_result.json')):
        try: payload=json.loads(path.read_text(encoding='utf-8'))
        except Exception as exc: failures.append({'path':str(path),'error':str(exc)}); continue
        seed_hint=payload.get('seeds',[None])[0]
        for row in payload.get('records',[]):
            if row.get('split') not in {'tune','promotion'}: continue
            seed=int(row.get('seed',seed_hint))
            if row.get('method')=='SFT' and (seed, row.get('split')) in seen_sft_splits: continue
            records.append(row)
            if row.get('method')=='SFT':
                seen_sft_splits.add((seed, row.get('split')))
                if row.get('split')=='promotion':
                    for q in row.get('question_metric_rows',[]):
                        if q.get('pair_count'): pair_counts[str(q['question_id'])]=int(q['pair_count'])
        for seed, seed_logs in payload.get('training_logs',{}).items(): logs.setdefault(str(seed),{}).update(seed_logs)
    aggregate=_aggregate_p23(records, tuple(args.methods), tuple(args.seeds), pairs_by_question=pair_counts, bootstrap_replicates=max(100,int(args.bootstrap_replicates)))
    result={'schema_version':'pesco_tier1_p23_common_protocol_matrix_v0.1','stage':args.stage,'seeds':[int(s) for s in args.seeds],'methods':list(args.methods),'records':records,'training_logs':logs,'aggregation':aggregate,'cell_failures':failures,'matrix_complete':not failures and all(any(int(r.get('seed'))==int(s) and r.get('method')==m for r in records) for s in args.seeds for m in (*args.methods,'SFT')),'eval_splits':['tune','promotion'],'diagnostic_only':True,'formal_comparison_authorized':False}
    (args.output_dir/'p23_matrix_result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,sort_keys=True),encoding='utf-8')
    manifest=build_run_manifest(experiment='tier1_p23_common_protocol_matrix',repo_root=ROOT,command=sys.argv,runner_paths=[ROOT/'scripts/aggregate_tier1_p23_cells.py',ROOT/'research_strategy_optimization/evaluation/tier1_p23_diagnostics.py'],data_paths=sorted(cells.glob('*/p23_result.json')),seeds={'training':[int(s) for s in args.seeds]},checkpoint=None,status='completed_diagnostic' if result['matrix_complete'] else 'failed_closed_incomplete_matrix',diagnostics={'stage':args.stage,'methods':list(args.methods),'matrix_complete':result['matrix_complete'],'cell_failures':len(failures),'common_sft_initialization':True,'formal_comparison_authorized':False})
    write_run_manifest(args.output_dir/'run_manifest.json',manifest)
    print(json.dumps({'matrix_complete':result['matrix_complete'],'cell_count':len(list(cells.glob('*/p23_result.json'))),'record_count':len(records),'promotion_summary':aggregate['promotion_summary'],'gate_checks':aggregate['gate_checks']},ensure_ascii=False))
    return 0 if result['matrix_complete'] else 2
if __name__=='__main__': raise SystemExit(main())
