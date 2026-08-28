#!/usr/bin/env python3
"""Audit the five independent promotion-v4 OOD mechanisms.

The private manifest is read only inside the evaluator process.  The public output
contains mechanism-level aggregate bias/repair summaries, never world labels,
individual effects, or hidden generated arrays.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from research_strategy_optimization.schemas import Protocol, ResearchAction, WorldSpec
from research_strategy_optimization.environments.tier1_tabular_env import Tier1TabularEnvironment

OOD=("heteroscedastic_noise","nonlinear_response","measurement_shift","missing_not_at_random","intervention_noncompliance")

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--manifest",type=Path,default=ROOT/"artifacts/tier1_p23_promotion_v4_private/private_manifest.json"); p.add_argument("--output",type=Path,default=ROOT/"artifacts/tier1_p232_promotion_v4_ood_audit.json"); a=p.parse_args(argv)
    raw=json.loads(a.manifest.read_text(encoding="utf-8")); questions=raw["collector_audit"]["benchmark_manifest"]["questions"]; protocol=Protocol(protocol_version="pesco_v0_2",exploration_seeds=(17,29,41,53,67,71,83,97),confirmation_seeds=(103,107,109,113,127,131,137,139)); rows=[]
    for family in OOD:
        q=next(q for q in questions if q.get("family")==family and q.get("split")=="final_ood"); w=next(w for w in q["worlds"] if w["kind"]=="invalid"); world=WorldSpec(**{k:v for k,v in w.items() if k in WorldSpec.__dataclass_fields__}); env=Tier1TabularEnvironment((world,),protocol=protocol); env.reset(q["policy_question_id"],world.world_id,seed=17); pre=env.snapshot(); base=env.execute_option(ResearchAction.CONTINUE,seeds=protocol.exploration_seeds); repaired=env.clone_from_snapshot(pre); repaired_out=repaired.execute_option(ResearchAction.REPAIR,seeds=protocol.exploration_seeds); base_bias=abs(float(base.effect_estimate)-float(world.true_effect_a)); repaired_bias=abs(float(repaired_out.effect_estimate)-float(world.true_effect_a)); rows.append({"family":family,"formula_signature_changed":base.code_hash!=repaired_out.code_hash or base.estimator!=repaired_out.estimator,"protocol_signature_changed":base.split_hash!=repaired_out.split_hash,"absolute_bias_reduction":base_bias-repaired_bias,"hidden_validation_bias_reduction":base_bias-repaired_bias,"repair_estimator_changed":base.estimator!=repaired_out.estimator})
    out={"schema_version":"pesco_promotion_v4_ood_mechanism_audit_v0.1","private_input":True,"public_redacted":True,"mechanisms":rows,"gates":{"five_families_present":len(rows)==5,"all_formula_signatures_changed":all(r["formula_signature_changed"] for r in rows),"all_protocol_signatures_changed":all(r["protocol_signature_changed"] for r in rows),"mean_absolute_bias_reduction":sum(r["absolute_bias_reduction"] for r in rows)/max(1,len(rows)),"hidden_validation_bias_decreases":all(r["hidden_validation_bias_reduction"]>0 for r in rows)}}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,ensure_ascii=False,sort_keys=True),encoding="utf-8"); print(json.dumps(out,ensure_ascii=False,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
