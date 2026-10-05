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

QUESTIONS = Path(__file__).with_name("questions.json")
TOLERANCE = 0.005  # 0.5% — floating-point sums reorder depending on the plan


def normalise(frame: pd.DataFrame) -> list[tuple]:
    """Order-insensitive, name-insensitive view of a result set."""
    rows = []
    for row in frame.itertuples(index=False):
        cells = []
        for cell in row:
            if isinstance(cell, (int, float)) and not isinstance(cell, bool):
                cells.append(round(float(cell), 2))
            else:
                cells.append(str(cell))
        rows.append(tuple(cells))
    return sorted(rows, key=lambda r: [str(c) for c in r])


def close_enough(got: list[tuple], want: list[tuple]) -> bool:
    if len(got) != len(want):
        return False
    for row_got, row_want in zip(got, want):
        if len(row_got) != len(row_want):
            return False
        for a, b in zip(row_got, row_want):
            if isinstance(a, float) and isinstance(b, float):
                scale = max(abs(a), abs(b), 1.0)
                if abs(a - b) / scale > TOLERANCE:
                    return False
            elif a != b:
                return False
    return True


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
            if close_enough(normalise(answer.to_frame()), normalise(gold)):
                result["passed"] = True
                result["detail"] = f"{answer.row_count} rows"
            else:
                result["detail"] = (
                    f"result differs: got {len(answer.to_frame())} rows, "
                    f"expected {len(gold)} rows"
                )

    result["elapsed_s"] = round(time.monotonic() - started, 3)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, help="write the full report here")
    parser.add_argument("--min-accuracy", type=float, default=None,
                        help="exit non-zero below this overall accuracy")
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

    if args.json:
        args.json.write_text(
            json.dumps({"mode": mode, "accuracy": accuracy, "results": results}, indent=2),
            encoding="utf-8",
        )
        print(f"\nreport written to {args.json}")

    if args.min_accuracy is not None and accuracy < args.min_accuracy:
        print(f"\nFAIL: accuracy {accuracy:.0%} below the {args.min_accuracy:.0%} bar")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
