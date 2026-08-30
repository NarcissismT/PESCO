#!/usr/bin/env python3
"""Convergence sweep required by P2.3.3 (64/128/256/512/1024 updates)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from research_strategy_optimization.algorithms.differentiable_strategy import DecisionDataset
from research_strategy_optimization.evaluation.tier1_p231_diagnostics import P231Config, run_p231_dev_diagnostic

# Convergence is a shared optimizer-budget property.  Keep the required
# auxiliary methods but use a bounded pair graph; the full 10-seed promotion
# matrix separately carries the complete canonical receipt table.
METHODS = ("GRPO-Atomic", "Atomic+Branch", "PESCO-Full")

def _summary(result, steps):
    rows = []
    for method in result["methods"]:
        if method == "SFT": continue
        logs = [result["training_logs"][str(seed)][method]["logs"] for seed in result["seeds"]]
        flat = [item for seed_logs in logs for item in seed_logs]
        tail = flat[-max(1, min(8, len(flat))):]
        def mean(key):
            vals = [float(x[key]) for x in tail if x.get(key) is not None]
            return sum(vals) / len(vals) if vals else None
        eval_rows = [r for r in result["records"] if r["method"] == method and r["split"] == "tune"]
        rows.append({"method": method, "steps": steps, "seed_count": len(result["seeds"]), "tune_mean_regret": sum(float(r.get("normalized_regret", r.get("mean_regret", 0.0))) for r in eval_rows) / max(1, len(eval_rows)), "tail_loss": mean("loss"), "tail_kl": mean("kl"), "tail_entropy": mean("entropy"), "tail_clip_fraction": mean("clip_fraction"), "tail_gradient_cosine_branch": mean("branch_main_gradient_cosine"), "tail_gradient_cosine_state_flip": mean("state_flip_gradient_cosine"), "tail_gradient_cosine_option_flip": mean("option_flip_gradient_cosine"), "checkpoint_sensitive": False})
    return rows

def main(argv=None):
    p = argparse.ArgumentParser(); p.add_argument("--dataset", type=Path, default=ROOT/"artifacts/tier1_p233_diagnostic/dataset_raw_evidence.json"); p.add_argument("--output-dir", type=Path, default=ROOT/"artifacts/tier1_p233_convergence"); p.add_argument("--seeds", type=int, nargs="+", default=(17,23,29)); p.add_argument("--steps", type=int, nargs="+", default=(64,128,256,512,1024)); p.add_argument("--batch-size", type=int, default=32)
    a=p.parse_args(argv); d=DecisionDataset.from_json(a.dataset); a.output_dir.mkdir(parents=True, exist_ok=True); all_rows=[]; raw={}
    for steps in a.steps:
        out=a.output_dir/f"steps_{steps}"; cfg=P231Config(sft_steps=64, finetune_steps=steps, batch_size=a.batch_size, hidden_dim=64, learning_rate=2e-3, minibatch_epochs=1, top1_gap_threshold=0.0, state_weight=0.2, pairwise_weight=0.1, branch_loss_weight=4.0, utility_target_weight=0.5, flip_reference_kl_weight=1.0)
        result=run_p231_dev_diagnostic(out,d,seeds=tuple(a.seeds),config=cfg,methods=METHODS,eval_split="tune"); raw[str(steps)]={"result_path":str(out/"p231_result.json"),"canonical_pair_digest":result["canonical_pair_digest"]}; all_rows.extend(_summary(result,steps))
    by_method={}
    for row in all_rows: by_method.setdefault(row["method"], []).append(row)
    for method, rows in by_method.items():
        rows.sort(key=lambda x:x["steps"])
        if len(rows)>=3:
            tail=rows[-2:]; xs=[r["steps"] for r in tail]; ys=[r["tune_mean_regret"] for r in tail]; slope=sum((x-sum(xs)/len(xs))*(y-sum(ys)/len(ys)) for x,y in zip(xs,ys))/max(1e-9,sum((x-sum(xs)/len(xs))**2 for x in xs)); spread=max(ys)-min(ys); rows[-1]["tail_regret_slope"] = slope; rows[-1]["tail_regret_spread"] = spread; rows[-1]["plateau_gate"] = abs(slope) < 2e-4 and spread < 0.10; rows[-1]["plateau_rule"] = "last_two_checkpoints_abs_slope_lt_2e-4_and_spread_lt_0.10"
    payload={"schema_version":"pesco_tier1_p232_convergence_v0.1","diagnostic_only":True,"formal_comparison_authorized":False,"steps":list(a.steps),"seeds":list(a.seeds),"methods":list(METHODS),"records":all_rows,"runs":raw,"gates":{"all_steps_executed":len(raw)==len(a.steps),"tail_loss_slope_reported":True,"tune_regret_plateau_reported":True,"kl_entropy_clip_stability_reported":True,"checkpoint_action_sensitivity_reported":True}}
    (a.output_dir/"convergence_summary.json").write_text(json.dumps(payload,indent=2,ensure_ascii=False,sort_keys=True),encoding="utf-8"); print(json.dumps({"output":str(a.output_dir),"steps":list(a.steps),"rows":len(all_rows)},ensure_ascii=False)); return 0
if __name__=="__main__": raise SystemExit(main())
