#!/usr/bin/env python3
"""Component-wise reward perturbation audit (not a positive scalar rescale)."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

COMPONENTS=("validity_gate","confirmation_bonus","repair_protocol_bonus","heldout_split_bonus","mechanism_transition_bonus","switch_success_bonus","switch_failure_penalty","sample_precision_bonus","state_resolution_bonus","replicate_bonus","execution_cost_penalty")

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--dataset',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args(argv)
    d=json.loads(a.dataset.read_text()); rows=[]; component_rates={}
    for component in COMPONENTS:
        stable=0; total=0; changed=0
        for e in d.get('examples',[]):
            rc=e.get('metadata',{}).get('reward_components',{}); base=np.asarray(e.get('branch_utilities',[]),float)
            if base.size!=4 or not isinstance(rc,dict): continue
            raw=[]
            for action in rc:
                vals=rc[action] if isinstance(rc[action],dict) else {}
                raw.append(sum(float(v) for v in vals.values()))
            base=np.asarray(raw,float); winners=[]
            for sign in (.8,1.0,1.2):
                pert=[]
                for action in rc:
                    vals=rc[action] if isinstance(rc[action],dict) else {}
                    pert.append(sum(float(v)*(sign if name==component else 1.0) for name,v in vals.items()))
                winners.append(int(np.argmax(np.asarray(pert))))
            total+=1; stable+=int(len(set(winners))==1); changed+=int(len(set(winners))>1)
        component_rates[component]={"winner_stability_rate":stable/max(1,total),"changed_winner_rate":changed/max(1,total),"example_count":total}
    overall=float(np.mean([v['winner_stability_rate'] for v in component_rates.values()])) if component_rates else 0.0
    out={"schema_version":"pesco_p233_r1_reward_weight_stability_v1","perturbation":"each reward component independently multiplied by 0.8/1.0/1.2","components":component_rates,"overall_mean_stability":overall,"pass":bool(overall>=0.80 and any(v['changed_winner_rate']>0 for v in component_rates.values())),"receipt_derived":True,"not_global_rescale":True}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,sort_keys=True)); print(json.dumps(out,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
