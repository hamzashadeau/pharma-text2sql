"""Compare two runs, question by question.

This is the piece that makes prompt engineering an engineering activity. A
prompt change moves a dozen questions in both directions; the summary line
barely twitches and the net improvement is reported as "slightly better". What
actually happened is that four questions started working and three stopped —
and the three that stopped are the ones a user already relies on.

Comparing per question also works with **no ground truth at all**, which is the
common case: an answer that was fine yesterday and errors today is a regression
whether or not anyone wrote down what the right answer was. That makes a frozen,
human-reviewed baseline a usable substitute for a labelled set, and a far
cheaper one to start with.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .run import QuestionResult, Run

# Worst to best. A question moving down this list is a regression.
STATUS_RANK = {"error": 0, "empty": 1, "clarification": 2, "success": 3}

# Below this, a latency change is noise rather than news.
LATENCY_NOISE_S = 1.0
LATENCY_REGRESSION_FACTOR = 1.5


@dataclass
class Change:
    id: str
    question: str
    kind: str          # regression | improvement | slower | faster
    detail: str


@dataclass
class Comparison:
    baseline: str
    candidate: str
    regressions: list[Change] = field(default_factory=list)
    improvements: list[Change] = field(default_factory=list)
    latency_changes: list[Change] = field(default_factory=list)
    only_in_baseline: list[str] = field(default_factory=list)
    only_in_candidate: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.regressions

    def report(self) -> str:
        lines = [f"{self.candidate} vs {self.baseline}"]
        lines.append(
            f"{len(self.improvements)} improved, {len(self.regressions)} regressed"
        )

        if self.regressions:
            lines.append("\nregressions:")
            for change in self.regressions:
                lines.append(f"  [{change.id}] {change.question}")
                lines.append(f"      {change.detail}")

        if self.improvements:
            lines.append("\nimprovements:")
            for change in self.improvements:
                lines.append(f"  [{change.id}] {change.detail}")

        if self.latency_changes:
            lines.append("\nlatency:")
            for change in self.latency_changes[:10]:
                lines.append(f"  [{change.id}] {change.detail}")

        if self.only_in_candidate or self.only_in_baseline:
            lines.append(
                f"\nquestion set changed: {len(self.only_in_candidate)} added, "
                f"{len(self.only_in_baseline)} removed"
            )

        if self.ok and not self.improvements:
            lines.append("\nno change in outcome")

        return "\n".join(lines)


def compare(baseline: Run, candidate: Run) -> Comparison:
    out = Comparison(baseline=baseline.label, candidate=candidate.label)

    before = {r.id: r for r in baseline.results}
    after = {r.id: r for r in candidate.results}

    out.only_in_baseline = sorted(set(before) - set(after))
    out.only_in_candidate = sorted(set(after) - set(before))

    for key in sorted(set(before) & set(after)):
        _compare_one(before[key], after[key], out)

    return out


def _compare_one(before: QuestionResult, after: QuestionResult, out: Comparison) -> None:
    question = after.question or before.question

    # 1. Ground truth, when the run has any: the clearest signal available.
    if before.passed is not None and after.passed is not None:
        if before.passed and not after.passed:
            out.regressions.append(Change(after.id, question, "regression",
                                          "passed before, fails now"))
            return
        if not before.passed and after.passed:
            out.improvements.append(Change(after.id, question, "improvement",
                                           "fails before, passes now"))
            return

    # 2. Status, which needs no labels at all.
    rank_before = STATUS_RANK.get(before.status, 3)
    rank_after = STATUS_RANK.get(after.status, 3)
    if rank_after < rank_before:
        out.regressions.append(Change(
            after.id, question, "regression",
            f"status {before.status} → {after.status}"
            + (f" ({after.answer[:80]})" if after.status == "error" and after.answer else ""),
        ))
        return
    if rank_after > rank_before:
        out.improvements.append(Change(after.id, question, "improvement",
                                       f"status {before.status} → {after.status}"))
        return

    # 3. Faithfulness: the answer still arrives, but now cites a figure the
    #    table does not contain. Invisible to any status counter.
    if before.faithfulness is not None and after.faithfulness is not None:
        if after.faithfulness < before.faithfulness:
            out.regressions.append(Change(
                after.id, question, "regression",
                f"faithfulness {before.faithfulness:.0%} → {after.faithfulness:.0%}",
            ))
            return
        if after.faithfulness > before.faithfulness:
            out.improvements.append(Change(
                after.id, question, "improvement",
                f"faithfulness {before.faithfulness:.0%} → {after.faithfulness:.0%}",
            ))

    # 4. Latency, reported but never counted as a regression: it is noisy by
    #    nature, and failing a build on one slow provider call teaches people
    #    to ignore the build.
    if before.duration_s > 0 and after.duration_s > before.duration_s * LATENCY_REGRESSION_FACTOR:
        if after.duration_s - before.duration_s >= LATENCY_NOISE_S:
            out.latency_changes.append(Change(
                after.id, question, "slower",
                f"{before.duration_s:.1f}s → {after.duration_s:.1f}s",
            ))
