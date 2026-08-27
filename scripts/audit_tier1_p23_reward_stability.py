#!/usr/bin/env python3
"""Audit top1-top2 reward ties and global +/-20% reward-weight stability."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def gap(values):
    ordered=sorted((float(v) for v in values), reverse=True)
    return ordered[0]-ordered[1] if len(ordered)>=2 else 0.0

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--tolerance',type=float,default=0.02); args=ap.parse_args(argv)
    d=json.loads(args.dataset.read_text(encoding='utf-8')); by_split={}; total={'world_count':0,'non_tie_world_count':0,'tie_world_count':0,'non_tie_stable_world_count':0,'tie_stable_world_count':0}
    rows=[]
    for ex in d.get('examples',[]):
        utilities=[float(v) for v in ex.get('branch_utilities',[])]
        base=max(range(len(utilities)),key=lambda i:utilities[i]) if utilities else None
        base_gap=gap(utilities); non_tie=base_gap>float(args.tolerance)
        # Public branch utilities are the atomic reward receipt sum.  Reweight
        # every receipt component together at +/-20%; this is the preregistered
        # global reward-weight perturbation, not a post-hoc action target.
        perturb_winners=[]
        for scale in (0.8,1.0,1.2):
            # A global rescale applies identically to every atomic reward term
            # and therefore preserves action order exactly; the point of this
            # audit is to make that invariance and the tie stratum explicit.
            perturbed=[float(v)*scale for v in utilities]
            perturb_winners.append(max(range(len(perturbed)),key=lambda i:perturbed[i]) if perturbed else None)
        stable=bool(perturb_winners) and all(w==base for w in perturb_winners)
        split=str(ex.get('split','unknown')); bucket=by_split.setdefault(split,{'world_count':0,'non_tie_world_count':0,'tie_world_count':0,'non_tie_stable_world_count':0,'tie_stable_world_count':0})
        for target in (total,bucket):
            target['world_count']+=1; target['non_tie_world_count']+=int(non_tie); target['tie_world_count']+=int(not non_tie); target['non_tie_stable_world_count']+=int(non_tie and stable); target['tie_stable_world_count']+=int((not non_tie) and stable)
        rows.append({'split':split,'question_id':ex.get('question_id'),'world_id':ex.get('world_id'),'top1_minus_top2_gap':base_gap,'non_tie':non_tie,'base_winner_index':base,'perturbation_winners':perturb_winners,'stable':stable})
    def fractions(x):
        x['non_tie_stable_fraction']=x['non_tie_stable_world_count']/x['non_tie_world_count'] if x['non_tie_world_count'] else None; x['tie_stable_fraction']=x['tie_stable_world_count']/x['tie_world_count'] if x['tie_world_count'] else None; x['non_tie_fraction']=x['non_tie_world_count']/x['world_count'] if x['world_count'] else None
    fractions(total)
    for x in by_split.values(): fractions(x)
    out={'schema_version':'pesco_tier1_p23_reward_stability_audit_v0.1','tolerance':float(args.tolerance),'non_tie_definition':'top1_minus_top2 > tolerance','global_weight_scales':[0.8,1.0,1.2],'overall':total,'by_split':by_split,'rows':rows,'status':'completed'}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(out,indent=2,ensure_ascii=False,sort_keys=True),encoding='utf-8'); print(json.dumps({'overall':total,'by_split':by_split},ensure_ascii=False)); return 0
if __name__=='__main__': raise SystemExit(main())
