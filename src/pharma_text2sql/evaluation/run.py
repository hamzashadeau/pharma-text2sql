"""What one evaluation run is, and what it costs.

A run is a list of per-question records plus the operational facts that decide
whether a system can be put in front of users: how long a question takes at the
tail, and what it costs. Accuracy alone never settles that argument — a
pipeline that is 94% right and takes 40 seconds at p95 is not shippable, and
nobody finds that out from an accuracy table.

Latency is reported at p50 and p95, never as a mean. One question that hits a
rate-limit retry pulls the mean somewhere nobody experiences; p95 is the
question a real user is unlucky enough to ask.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# Indicative per-million-token prices, only used to turn token counts into a
# number people can argue about. Override through `prices` — they move, and a
# hard-coded table that silently goes stale is worse than no table.
DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "mistralai/mistral-small": (0.20, 0.60),
    "mistralai/mistral-large": (2.00, 6.00),
}


@dataclass
class QuestionResult:
    id: str
    question: str
    level: str | int = ""
    status: str = "success"          # success | empty | clarification | error
    answer: str = ""
    sql: str = ""
    passed: bool | None = None       # None when the run carries no ground truth
    faithfulness: float | None = None
    judge_verdict: str | None = None
    judge_reason: str = ""
    duration_s: float = 0.0
    model: str = ""
    tokens_in: int = 0
    tokens_out: int = 0

    def cost(self, prices: dict[str, tuple[float, float]] | None = None) -> float:
        table = prices or DEFAULT_PRICES
        if self.model not in table:
            return 0.0
        price_in, price_out = table[self.model]
        return (self.tokens_in * price_in + self.tokens_out * price_out) / 1_000_000


@dataclass
class Run:
    label: str = "run"
    mode: str = ""
    results: list[QuestionResult] = field(default_factory=list)

    # -- aggregates --------------------------------------------------------

    def latency(self) -> dict[str, float]:
        values = np.array([r.duration_s for r in self.results], dtype=float)
        if values.size == 0:
            return {"p50": 0.0, "p95": 0.0, "max": 0.0, "total": 0.0}
        return {
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "max": float(values.max()),
            "total": float(values.sum()),
        }

    def cost(self, prices: dict[str, tuple[float, float]] | None = None) -> dict[str, float]:
        total = sum(r.cost(prices) for r in self.results)
        n = len(self.results) or 1
        return {"total": total, "per_question": total / n, "per_1000": total / n * 1000}

    def status_mix(self) -> dict[str, int]:
        mix: dict[str, int] = {}
        for result in self.results:
            mix[result.status] = mix.get(result.status, 0) + 1
        return dict(sorted(mix.items()))

    def accuracy(self) -> float | None:
        """None when the run has no ground truth — not zero.

        Reporting an unlabelled run as 0% accurate is how an evaluation starts
        lying. The distinction between "wrong" and "not measured" has to
        survive every layer that touches it.
        """
        labelled = [r for r in self.results if r.passed is not None]
        if not labelled:
            return None
        return sum(1 for r in labelled if r.passed) / len(labelled)

    def faithfulness(self) -> float | None:
        scored = [r.faithfulness for r in self.results if r.faithfulness is not None]
        return float(np.mean(scored)) if scored else None

    def by_level(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for level in sorted({str(r.level) for r in self.results}):
            group = [r for r in self.results if str(r.level) == level]
            labelled = [r for r in group if r.passed is not None]
            out[level] = {
                "n": len(group),
                "accuracy": (
                    sum(1 for r in labelled if r.passed) / len(labelled) if labelled else None
                ),
                "p95_s": float(np.percentile([r.duration_s for r in group], 95)),
            }
        return out

    # -- persistence -------------------------------------------------------

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "label": self.label,
                    "mode": self.mode,
                    "results": [asdict(r) for r in self.results],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "Run":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        known = set(QuestionResult.__dataclass_fields__)
        return cls(
            label=raw.get("label", "run"),
            mode=raw.get("mode", ""),
            results=[
                QuestionResult(**{k: v for k, v in item.items() if k in known})
                for item in raw.get("results", [])
            ],
        )

    def report(self, prices: dict[str, tuple[float, float]] | None = None) -> str:
        lines = [f"run: {self.label}" + (f"   mode: {self.mode}" if self.mode else "")]
        lines.append(f"questions: {len(self.results)}")

        accuracy = self.accuracy()
        lines.append(
            f"accuracy: {accuracy:.0%}" if accuracy is not None
            else "accuracy: not measured (no ground truth in this run)"
        )

        faithful = self.faithfulness()
        if faithful is not None:
            unfaithful = sum(1 for r in self.results
                             if r.faithfulness is not None and r.faithfulness < 1.0)
            lines.append(
                f"faithfulness: {faithful:.0%} of cited figures supported"
                f" ({unfaithful} answers citing a figure the table does not contain)"
            )

        mix = ", ".join(f"{k} {v}" for k, v in self.status_mix().items())
        lines.append(f"status: {mix}")

        latency = self.latency()
        lines.append(
            f"latency: p50 {latency['p50']:.1f}s   p95 {latency['p95']:.1f}s   "
            f"max {latency['max']:.1f}s"
        )

        cost = self.cost(prices)
        if cost["total"] > 0:
            lines.append(
                f"cost: ${cost['total']:.4f} total, ${cost['per_1000']:.2f} per 1000 questions"
            )

        return "\n".join(lines)
