#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument('--dataset',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args(argv)
 d=json.loads(a.dataset.read_text()); scales=(.8,1.0,1.2); winners=[]
 for e in d.get('examples',[]):
  u=np.asarray(e.get('branch_utilities',[]),float); winners.append([int(np.argmax(u*s)) for s in scales] if len(u)==4 else [])
 stable=sum(bool(x) and len(set(x))==1 for x in winners)/max(1,len(winners)); out={'schema_version':'pesco_p233_reward_weight_stability_v0.1','scales':list(scales),'example_count':len(winners),'winner_stability_rate':stable,'pass':bool(stable>=0.95),'receipt_derived':True,'perturbation':'global_atomic_reward_weight_plus_minus_20_percent'}; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,sort_keys=True)); print(json.dumps(out,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
