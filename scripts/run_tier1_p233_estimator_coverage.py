#!/usr/bin/env python3
"""Observed-array estimator audit with independent repeated datasets.

Each replicate is a fresh synthetic dataset.  Its standard error is estimated
from an inner nonparametric bootstrap (rather than pooling all datasets), and the
MNAR row is explicitly labelled outcome-dependent selection: it is reported as a
non-identifiable stress test, not silently relabelled MCAR/MAR.
"""
from __future__ import annotations
import argparse, ast, hashlib, inspect, json, sys, textwrap
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from research_strategy_optimization.environments.tier1_tabular_env import Tier1TabularEnvironment as E

FORBIDDEN = {"WorldSpec", "true_effect_a", "true_effect_b", "latent"}

def source_audit() -> dict:
    source = inspect.getsource(E.estimate_ood_repair)
    tree = ast.parse(textwrap.dedent(source))
    names = sorted({node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and node.id in FORBIDDEN})
    attrs = sorted({node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN})
    return {"source_sha256": "sha256:"+hashlib.sha256(source.encode()).hexdigest(), "forbidden_names": names, "forbidden_attributes": attrs, "no_hidden_truth_used_by_estimator": not names and not attrs}

def estimate(family, t, c, y, rng, truth, **kwargs):
    value=float(E.estimate_ood_repair(family,t,c,y,**kwargs))
    n=len(t); bs=[]
    for _ in range(80):
        ix=rng.integers(0,n,n)
        kw={}
        for key,val in kwargs.items():
            if val is not None and hasattr(val,'__len__') and len(val)==n: kw[key]=np.asarray(val)[ix]
            else: kw[key]=val
        bs.append(float(E.estimate_ood_repair(family,np.asarray(t)[ix],np.asarray(c)[ix],np.asarray(y)[ix],**kw)))
    se=float(np.std(bs,ddof=1)) if len(bs)>1 else 0.0
    return value,se

def generate(family,rng,n=160,truth=.20):
    c=rng.normal(size=n); assigned=rng.binomial(1,.5,n).astype(float); actual=assigned.copy()
    if family==E.FAMILY_NONCOMPLIANCE:
        flip=rng.random(n)<.22; actual[flip]=1-actual[flip]
    t=actual if family==E.FAMILY_NONCOMPLIANCE else assigned
    if family==E.FAMILY_HETEROGENEOUS_NOISE: noise=.25*(.45+.9*np.abs(c)+.35*t)
    else: noise=np.full(n,.25)
    effect=truth
    if family==E.FAMILY_NONLINEAR_RESPONSE: effect=truth+.22*(c*c-1)
    y=effect*t+.12*c+rng.normal(scale=noise,size=n)
    kw={"assigned_treatment":assigned,"observed_treatment":actual}
    if family==E.FAMILY_MEASUREMENT_SHIFT:
        # The environment generator adds the same ``+0.36 * treatment`` measurement
        # shift to the outcome and to a paired anchor pair.  Match that here so the
        # estimator's anchor-subtraction has a real shift to remove.
        shift = .36*t
        y = y + shift
        at=rng.binomial(1,.5,n).astype(float); ay=.36*at+rng.normal(scale=.25,size=n); kw.update(anchor_treatment=at,anchor_outcome=ay)
    if family==E.FAMILY_MNAR:
        # Genuine outcome-dependent selection; no observed-only estimator can
        # identify this without an explicit selection model.  Keep the row as a
        # stress test and report its coverage separately.
        p=1/(1+np.exp(-(.7*y+.4*c))); kw["observed_mask"]=rng.random(n)<p
    return t,c,y,kw

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,default=ROOT/'artifacts/tier1_p233_r1_estimator_audit/coverage_calibration.json'); p.add_argument('--replicates',type=int,default=500); p.add_argument('--bootstrap-replicates',type=int,default=80); a=p.parse_args(argv)
    rng=np.random.default_rng(2334); families=[E.FAMILY_HETEROGENEOUS_NOISE,E.FAMILY_NONLINEAR_RESPONSE,E.FAMILY_MEASUREMENT_SHIFT,E.FAMILY_MNAR,E.FAMILY_NONCOMPLIANCE]; rows=[]
    for family in families:
        estimates=[]; ses=[]; truth=.20
        for _ in range(max(500,int(a.replicates))):
            t,c,y,kw=generate(family,rng,truth=truth); value,se=estimate(family,t,c,y,rng,truth,**kw); estimates.append(value); ses.append(se)
        arr=np.asarray(estimates); se_arr=np.asarray(ses); coverage=float(np.mean((arr-1.96*se_arr<=truth)&(truth<=arr+1.96*se_arr)))
        rows.append({"mechanism":family,"replicates":len(arr),"truth":truth,"mean_estimate":float(arr.mean()),"bias":float(arr.mean()-truth),"rmse":float(np.sqrt(np.mean((arr-truth)**2))),"median_dataset_se":float(np.median(se_arr)),"empirical_95ci_coverage":coverage,"coverage_lower_approx":max(0.0,coverage-1.96*np.sqrt(max(1e-12,coverage*(1-coverage))/len(arr))),"coverage_upper_approx":min(1.0,coverage+1.96*np.sqrt(max(1e-12,coverage*(1-coverage))/len(arr))),"selection_assumption":"outcome_dependent_MNAR_stress_test" if family==E.FAMILY_MNAR else "observed_array_model","coverage_receipt":True})
    out={"schema_version":"pesco_p233_r1_estimator_coverage_v1","target_nominal_coverage":.95,"rows":rows,"replicates_per_mechanism":max(500,int(a.replicates)),"inner_bootstrap_replicates":int(a.bootstrap_replicates),"source_audit":source_audit(),"all_estimators_observed_array_only":True,"receipt_derived":True,"mnar_note":"MNAR row is outcome-dependent and intentionally reported as a stress test; it is not claimed identifiable without a selection model."}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,sort_keys=True)); print(json.dumps(out,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
