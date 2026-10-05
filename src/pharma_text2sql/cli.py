"""Command line entry point.

    python -m pharma_text2sql "top 10 molecules by sales in France in 2024"
    python -m pharma_text2sql --yes "total units in Germany"      # skip the gate
    python -m pharma_text2sql --schema
"""

from __future__ import annotations

import argparse
import logging
import sys

from . import llm, pipeline, schema


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pharma-text2sql")
    parser.add_argument("question", nargs="*", help="the question, in plain English or French")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="approve the interpretation without asking")
    parser.add_argument("--schema", action="store_true", help="print the table description and exit")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.schema:
        print(schema.get().describe())
        return 0

    question = " ".join(args.question).strip()
    if not question:
        parser.print_help()
        return 2

    mode = "model" if llm.available() else "offline (deterministic planner)"
    print(f"— mode: {mode}\n")

    # Phase 1: understand.
    explanation = pipeline.explain(question)
    print(f"Interpretation: {explanation.interpretation}")
    if explanation.resolved:
        for column, values in explanation.resolved.items():
            print(f"  resolved {column}: {', '.join(sorted(set(values)))}")

    if explanation.needs_clarification:
        print(f"\nNeeds clarification: {explanation.clarification}")
        return 1

    # The validation gate. --yes is for scripts; a person gets asked.
    if not args.yes and sys.stdin.isatty():
        reply = input("\nRun this? [Y/n] ").strip().lower()
        if reply and not reply.startswith("y"):
            print("Cancelled — nothing was executed.")
            return 0

    # Phase 2: execute.
    answer = pipeline.execute(question, skip_clarification=True)
    if not answer.ok:
        print(f"\n{answer.error}")
        if answer.sql:
            print(f"\nRefused SQL:\n{answer.sql}")
        return 1

    print(f"\nSQL:\n{answer.sql}\n")
    print(answer.to_frame().to_string(index=False, max_rows=25))
    print(f"\n{answer.narration}")
    if answer.chart:
        print(f"Suggested chart: {answer.chart['type']} ({answer.chart['x']} / {answer.chart['y']})")
    print(f"\n{answer.row_count} rows in {answer.elapsed_s:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
