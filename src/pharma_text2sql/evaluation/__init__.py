"""Scoring an LLM pipeline: correctness, faithfulness, cost, and regressions.

Four pieces, ordered by how much they cost to run:

* :mod:`faithfulness` — deterministic. Are the figures in the answer actually
  in the result table? Free, runs in CI on every change.
* :mod:`compare` — deterministic. What moved between two runs, question by
  question. Works with no ground truth, which is the usual starting point.
* :mod:`run` — the run model: accuracy when labels exist, status mix, latency
  percentiles, cost.
* :mod:`judge` — a model reading the answers, for what arithmetic cannot
  settle. Costs money, never gates a build, and reports "not scored" rather
  than passing silently when it cannot run.
"""

from . import compare, faithfulness, judge, run  # noqa: F401
from .run import QuestionResult, Run  # noqa: F401

__all__ = ["compare", "faithfulness", "judge", "run", "Run", "QuestionResult"]
