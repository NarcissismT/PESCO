"""CLI for generating the PESCO experiment report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .demo import generate_demo_records, write_demo
from .metrics import load_records
from .report import write_report
from .tier0_runner import run_tier0_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate PESCO metrics, figures, and Markdown report")
    parser.add_argument("input", nargs="?", help="JSON/JSONL result records")
    parser.add_argument("--output", "-o", default="PESCO/artifacts/report", help="output directory")
    parser.add_argument("--demo", action="store_true", help="generate deterministic Tier-0 smoke-test records")
    parser.add_argument("--tier0", action="store_true", help="execute the repository Tier-0 simulator and trusted verifier")
    parser.add_argument("--demo-questions", type=int, default=8, help="number of demo research questions")
    parser.add_argument("--seed", type=int, default=17, help="demo/bootstrap seed")
    parser.add_argument("--bootstrap", type=int, default=300, help="question-cluster bootstrap draws")
    parser.add_argument("--formats", default="png,svg", help="comma-separated figure formats")
    parser.add_argument("--title", default="PESCO experimental report", help="report title")
    parser.add_argument("--vrs-weights", default="1,1,1,1,0.1", help="alpha,beta,gamma,eta,lambda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output)
    formats = tuple(item.strip().lower().lstrip(".") for item in args.formats.split(",") if item.strip())
    if any(item not in {"png", "svg", "pdf"} for item in formats):
        raise SystemExit("--formats supports png, svg, and pdf")
    try:
        weights_values = [float(item.strip()) for item in args.vrs_weights.split(",")]
        if len(weights_values) != 5:
            raise ValueError
    except ValueError as exc:
        raise SystemExit("--vrs-weights must contain five comma-separated numbers") from exc
    weights = dict(zip(("alpha", "beta", "gamma", "eta", "lambda"), weights_values))

    if args.demo and args.tier0:
        raise SystemExit("choose at most one of --demo and --tier0")
    if args.demo:
        records = generate_demo_records(args.seed, questions=args.demo_questions)
        output.mkdir(parents=True, exist_ok=True)
        demo_input = output / "demo_results.json"
        demo_input.write_text(json.dumps({"schema_version": "pesco_results_v0.1", "records": records}, indent=2), encoding="utf-8")
        source = str(demo_input)
    elif args.tier0:
        records = run_tier0_records(seeds=None)
        output.mkdir(parents=True, exist_ok=True)
        tier0_input = output / "tier0_results.json"
        tier0_input.write_text(json.dumps({"schema_version": "pesco_results_v0.1", "source": "tier0_simulator", "synthetic_pilot": True, "records": records}, indent=2), encoding="utf-8")
        source = str(tier0_input)
    elif args.input:
        records = load_records(args.input)
        source = str(args.input)
    else:
        raise SystemExit("provide INPUT or use --demo")

    if not records:
        raise SystemExit(f"no result records found in {source}")
    result = write_report(
        records,
        output,
        title=args.title,
        bootstrap=max(0, args.bootstrap),
        seed=args.seed,
        formats=formats,
        weights=weights,
        metadata={"source": source, "input_mode": "demo" if args.demo else ("tier0" if args.tier0 else "file")},
    )
    print(json.dumps({"source": source, "records": len(records), "output": str(output), "report": result["report"], "figures": result["figures"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
