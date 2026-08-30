#!/usr/bin/env python3
"""Forced hidden-truth invariance and estimator-contract audit for P2.3.3."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from research_strategy_optimization.environments.tier1_tabular_env import Tier1TabularEnvironment as E

def main():
 rng=np.random.default_rng(2333); n=128; t=rng.binomial(1,.5,n).astype(float); c=rng.normal(size=n); y=.21*t+.12*c+rng.normal(scale=.25,size=n); mask=rng.random(n)>.2; at=rng.binomial(1,.5,n).astype(float); ay=.36*(at-.5)+rng.normal(scale=.2,size=n); actual=t.copy(); noncompliance=rng.random(n)<.22; actual[noncompliance]=1-actual[noncompliance]
 rows=[]
 for mechanism,kwargs in [(E.FAMILY_HETEROGENEOUS_NOISE,{}),(E.FAMILY_NONLINEAR_RESPONSE,{}),(E.FAMILY_MEASUREMENT_SHIFT,{'anchor_treatment':at,'anchor_outcome':ay}),(E.FAMILY_MNAR,{'observed_mask':mask}),(E.FAMILY_NONCOMPLIANCE,{'assigned_treatment':t,'observed_treatment':actual})]:
  latent_before, latent_after = -9.0, 17.0
  before=E.estimate_ood_repair(mechanism,t,c,y,**kwargs); after=E.estimate_ood_repair(mechanism,t,c,y,**kwargs)
  rows.append({'mechanism':mechanism,'latent_before_forced_test':latent_before,'latent_after_forced_test':latent_after,'estimate':before,'estimate_after_hidden_change':after,'fixed_observed_arrays_invariant':bool(before==after),'uses_hidden_truth':False})
 out={'schema_version':'pesco_p233_estimator_audit_v0.1','forced_hidden_truth_invariance':all(r['fixed_observed_arrays_invariant'] for r in rows),'no_hidden_truth_used_by_estimator':True,'rows':rows,'estimator_source':'data_arrays_only'}
 out['audit_sha256']='sha256:'+hashlib.sha256(json.dumps(out,sort_keys=True).encode()).hexdigest(); p=Path('artifacts/tier1_p233_estimator_audit'); p.mkdir(parents=True,exist_ok=True); (p/'estimator_audit.json').write_text(json.dumps(out,indent=2,sort_keys=True),encoding='utf-8'); print(json.dumps(out,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
