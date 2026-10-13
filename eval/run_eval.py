"""Score the pipeline against the question set.

The metric is **execution accuracy**: the generated query is run, the gold query
is run, and the two result sets are compared after normalisation. Comparing SQL
strings would be meaningless — there are a dozen correct ways to write the same
aggregate — and comparing narration would measure the wording, not the answer.

Levels 1-5 are scored on the result. Level 6 is scored on *behaviour*: an
unknown entity must come back as a question to the user, and a stacked
``DROP TABLE`` must be refused. A system that answers those confidently is
worse than one that fails loudly, and a single accuracy number hides that.

    python eval/run_eval.py                 # human-readable report
    python eval/run_eval.py --json out.json # machine-readable
    python eval/run_eval.py --min-accuracy 0.8   # non-zero exit below the bar
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from pharma_text2sql import llm, pipeline, schema  # noqa: E402
from pharma_text2sql.evaluation import compare as compare_runs  # noqa: E402
from pharma_text2sql.evaluation import faithfulness  # noqa: E402
from pharma_text2sql.evaluation import resultset  # noqa: E402
from pharma_text2sql.evaluation.run import QuestionResult, Run  # noqa: E402

QUESTIONS = Path(__file__).with_name("questions.json")


def score_one(case: dict) -> dict:
    started = time.monotonic()
    result = {
        "id": case["id"],
        "level": case["level"],
        "question": case["question"],
        "passed": False,
        "detail": "",
        "sql": "",
    }

    expect = case.get("expect")

    try:
        answer = pipeline.execute(case["question"])
    except Exception as exc:
        result["detail"] = f"crashed: {type(exc).__name__}: {exc}"
        result["elapsed_s"] = round(time.monotonic() - started, 3)
        return result

    result["sql"] = answer.sql

    if expect == "clarification":
        needle = case.get("expect_contains", "").lower()
        result["passed"] = (not answer.ok) and needle in answer.error.lower()
        result["detail"] = answer.error or "answered instead of asking"

    elif expect == "refused_or_safe":
        # Either the guard refused it, or it ran as a single harmless SELECT.
        # What is NOT acceptable is the table disappearing.
        still_there = True
        try:
            schema.query("SELECT count(*) FROM sales LIMIT 1")
        except Exception:
            still_there = False
        result["passed"] = still_there and (not answer.ok or "DROP" not in answer.sql.upper())
        result["detail"] = answer.error or "ran as a single SELECT; table intact"

    else:
        if not answer.ok:
            result["detail"] = answer.error
        else:
            gold = schema.query(case["gold_sql"])
            # Same data, whatever the SQL looked like: row order, column order
            # and aliases are the model's free choices, not the answer.
            if resultset.equivalent(answer.to_frame(), gold):
                result["passed"] = True
                result["detail"] = f"{answer.row_count} rows"
            else:
                result["detail"] = (
                    f"result differs: got {len(answer.to_frame())} rows, "
                    f"expected {len(gold)} rows"
                )

    # Status and faithfulness are recorded for every question, including the
    # ones with no ground truth: they are what makes two runs comparable when
    # nobody has written down the right answer yet.
    result["status"] = (
        "success" if answer.ok
        else "clarification" if expect == "clarification" or "matches several" in answer.error
        else "error"
    )
    if answer.ok and answer.row_count == 0:
        result["status"] = "empty"
    result["answer"] = answer.narration
    result["elapsed_s"] = round(time.monotonic() - started, 3)

    if answer.ok and answer.narration:
        checked = faithfulness.check(
            answer.narration, answer.to_frame(), question=case["question"]
        )
        result["faithfulness"] = checked.score
        if not checked.ok:
            result["faithfulness_detail"] = checked.summary()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, help="write the full report here")
    parser.add_argument("--min-accuracy", type=float, default=None,
                        help="exit non-zero below this overall accuracy")
    parser.add_argument("--save-run", type=Path,
                        help="write this run so a later one can be compared to it")
    parser.add_argument("--compare", type=Path,
                        help="compare against a run saved earlier")
    parser.add_argument("--fail-on-regression", action="store_true",
                        help="exit non-zero if any question got worse than the baseline")
    args = parser.parse_args(argv)

    cases = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    mode = "model" if llm.available() else "offline (deterministic planner)"

    print(f"pharma-text2sql — evaluation")
    print(f"mode: {mode}   questions: {len(cases)}\n")

    results = [score_one(case) for case in cases]

    by_level: dict[int, list[dict]] = defaultdict(list)
    for result in results:
        by_level[result["level"]].append(result)

    print(f"{'level':>5}  {'passed':>8}  {'accuracy':>9}")
    print("-" * 28)
    for level in sorted(by_level):
        group = by_level[level]
        passed = sum(1 for r in group if r["passed"])
        print(f"{level:>5}  {passed:>3}/{len(group):<4}  {passed / len(group):>8.0%}")

    passed = sum(1 for r in results if r["passed"])
    accuracy = passed / len(results)
    elapsed = sum(r.get("elapsed_s", 0) for r in results)
    print("-" * 28)
    print(f"{'all':>5}  {passed:>3}/{len(results):<4}  {accuracy:>8.0%}")
    print(f"\ntotal {elapsed:.1f}s, {elapsed / len(results):.2f}s per question")

    failures = [r for r in results if not r["passed"]]
    if failures:
        print(f"\n{len(failures)} failing:")
        for r in failures:
            print(f"  [{r['id']}] {r['question']}\n      {r['detail']}")

    run = Run(
        label=args.save_run.stem if args.save_run else "current",
        mode=mode,
        results=[
            QuestionResult(
                id=r["id"],
                question=r["question"],
                level=r["level"],
                status=r.get("status", "error"),
                answer=r.get("answer", ""),
                sql=r.get("sql", ""),
                passed=r["passed"],
                faithfulness=r.get("faithfulness"),
                duration_s=r.get("elapsed_s", 0.0),
            )
            for r in results
        ],
    )

    print()
    print(run.report())

    unfaithful = [r for r in results if r.get("faithfulness_detail")]
    if unfaithful:
        print(f"\n{len(unfaithful)} answers cite a figure the result table does not contain:")
        for r in unfaithful:
            print(f"  [{r['id']}] {r['faithfulness_detail']}")

    exit_code = 0

    if args.compare:
        if not args.compare.exists():
            print(f"\nbaseline not found: {args.compare}")
        else:
            comparison = compare_runs.compare(Run.from_json(args.compare), run)
            print()
            print(comparison.report())
            if args.fail_on_regression and not comparison.ok:
                print(f"\nFAIL: {len(comparison.regressions)} question(s) regressed")
                exit_code = 1

    if args.save_run:
        run.to_json(args.save_run)
        print(f"\nrun saved to {args.save_run}")

    if args.json:
        args.json.write_text(
            json.dumps({"mode": mode, "accuracy": accuracy, "results": results}, indent=2),
            encoding="utf-8",
        )
        print(f"report written to {args.json}")

    if args.min_accuracy is not None and accuracy < args.min_accuracy:
        print(f"\nFAIL: accuracy {accuracy:.0%} below the {args.min_accuracy:.0%} bar")
        exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
