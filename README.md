# pharma-text2sql

Ask a pharmaceutical sales database a question in plain English or French, get a
checked SQL query, a result table and a sentence that states what it shows.

The interesting part is not that a language model can write SQL — it can. The
interesting part is everything placed around it so the answer can be trusted:
the terms the user typed are resolved against real database values before any
query is written, the user confirms the interpretation before anything runs,
every generated query is validated on its parse tree, generated Python runs in
a separate locked-down process, and the whole thing is scored against a
question set on every push.

```
                      ┌──────────────────────────────────────────┐
  "top 5 molecules    │ 1. intent extraction      (LLM → JSON)   │
   by sales in the    │ 2. entity resolution      (fuzzy + index)│
   UK in 2024"  ──────▶ 3. interpretation         ───────────────┼──▶  shown to the user
                      └──────────────────────────────────────────┘      ▲
                                                                        │ approved
                      ┌──────────────────────────────────────────┐      │
                      │ 4. SQL generation         (LLM / planner)│◀─────┘
                      │ 5. guard                  (AST, sqlglot) │
                      │ 6. execution              (DuckDB)       │
                      │ 7. narration              (LLM)          │
                      │ 8. chart choice           (result shape) │
                      └──────────────────────────────────────────┘
```

---

## Run it

No API key needed. The pipeline ships with a deterministic planner that answers
the whole evaluation suite offline, so the demo costs nothing.

```bash
git clone https://github.com/hamzashadeau/pharma-text2sql
cd pharma-text2sql
pip install -e ".[dev]"
python data/generate.py                       # synthetic dataset, ~65k rows

python -m pharma_text2sql "top 5 molecules by sales in France in 2024"
pytest -q                                     # 59 tests
python eval/run_eval.py                       # scored evaluation
```

With a key, the model path is used instead — same pipeline, same guards:

```bash
export OPENROUTER_API_KEY=sk-...
python -m pharma_text2sql "which corporation sold the most in the UK last year?"
```

An HTTP surface mirrors the two phases:

```bash
uvicorn pharma_text2sql.api:app --reload       # POST /explain, POST /execute
```

---

## What it does, and why

### The user confirms the interpretation before anything runs

`explain()` understands the question and stops. `execute()` runs it. Between
them, a person sees a sentence like

> I will compute the total sales (EUR) broken down by molecule, restricted to
> country = United Kingdom; year = 2024, keeping the top 5.

A text-to-SQL answer reads exactly as confidently whether or not the query
meant what the user intended. Without a confirmation step a misread question
becomes a number in a slide deck with nothing marking it as wrong. The same
gate doubles as the audit trail — what was asked, how it was understood, what
was approved, what ran — which is what makes this shape of system defensible in
a regulated setting.

### Entities are resolved before SQL is written

Asked about "statins in the UK", a model will happily write
`WHERE country = 'UK' AND atc4 = 'statins'`. Both are wrong, the query is
valid, and the result is empty — which looks like an answer.

So terms are resolved first, against an index of the values the database
actually holds, with three outcomes:

| Question | Outcome |
|---|---|
| `sales of statins in the UK` | resolved → `atc4 = 'C10AA — Statins'`, `country = 'United Kingdom'` |
| `sales of ATORVASTINE` | resolved through a typo → `ATORVASTIN` |
| `total sales in United` | **ambiguous** → "United Kingdom or United States?" |
| `sales of NOTAMOLECULE in Spain` | **unresolved** → asks, instead of silently totalling every molecule in Spain |

That last row is the one that matters. It is the difference between a wrong
number and a question.

### Every query is checked on its parse tree

`sql_guard` parses the generated SQL with sqlglot and refuses anything that is
not a single `SELECT` against the one allowed table: writes, stacked
statements, file-reading table functions, unknown columns. A `LIMIT` is forced
on every query so a question cannot return millions of rows.

The check runs on the **AST, never on the text** — and that is not a detail. A
regex denylist searching for `DROP` refuses a question about eye-drop products,
and one searching for `DELETE` trips over molecule names. Guardrails that
reject legitimate work get switched off, and a guardrail that is switched off
protects nothing.

### Generated Python runs in a process that cannot do anything

Some questions are easier in pandas than in SQL. That means running code a
model wrote, which is only acceptable if it cannot reach anything:

- a **throwaway child process** — an in-process denylist is escapable, and
  `().__class__.__base__.__subclasses__()` is all it takes;
- an **environment allowlist**, so an escape to `os.environ` finds nothing (an
  allowlist, not a denylist: a denylist leaks every secret whose name we forget
  to add);
- an **audit hook** refusing network, process spawn and native-code loading at
  the interpreter level, below any Python-level trick;
- **CPU, memory and file-size limits** where the platform provides them, and a
  wall-clock timeout enforced by the parent, which is what actually kills an
  infinite loop;
- **JSON back, never pickle** — the child is the untrusted side.

`tests/test_sandbox.py` asserts each of these, including the subclasses escape.

### It is measured, not asserted

`eval/run_eval.py` runs a question set across six levels of complexity and
scores **execution accuracy**: the generated query and a gold query are both
run, and the result sets are compared. Comparing SQL strings would be
meaningless — there are a dozen correct ways to write the same aggregate.

```
level    passed   accuracy
----------------------------
    1    4/4         100%     one filter, one aggregate
    2    4/4         100%     group by
    3    4/4         100%     top-N ranking
    4    4/4         100%     monthly / quarterly / yearly series
    5    4/4         100%     several filters at once
    6    4/4         100%     must refuse: unknown entity, injection
----------------------------
  all   24/24        100%
```

Level 6 is scored on *behaviour*, not on a result: an unknown entity must come
back as a question, and `OMEZARIN; DROP TABLE sales` must leave the table
standing. A system that answers those confidently is worse than one that fails
loudly, and a single accuracy number would hide that.

CI runs the suite on every push with `--min-accuracy 0.95`, so a regression in
the guard, the resolver or the planner fails the build.

**Read that 100% for what it is.** The question set and the offline planner were
written together, so this number measures *regression*, not general capability —
it tells you the pipeline still behaves as specified, not that it would handle
any question a market analyst could invent. The honest use of a figure like this
is as a tripwire, and that is how CI uses it.

---

## Demand forecasting and backtesting

A second, independent module: `pharma_text2sql.forecast`. It answers a question
the text-to-SQL side cannot — *how much will this product sell next quarter* —
and it is built around the measurement rather than around the model.

```bash
python -m pharma_text2sql.forecast --level molecule --horizon 3
python -m pharma_text2sql.forecast --csv history.csv \
    --series-col product_code --date-col month --value-col units
```

```
         model  series  mape  wape  mase  bias  beats_seasonal_naive
  holt_winters      10  3.06  3.15  0.68 -2.40                100.00
seasonal_naive      10  5.57  5.80  1.11 -5.37                  0.00
         naive      10  7.74  7.21  1.99 -0.63                 50.00
         drift      10  7.94  7.39  2.04 -0.30                 50.00
moving_average      10 10.98 10.17  2.81 -1.23                 10.00
```

**Baselines first.** Three of the five models are trivial on purpose. "MAPE
3.1%" means nothing until you know that answering *the same month last year*
scores 5.6% on the same data. A forecasting project without baselines cannot
tell whether its model earned anything.

**Rolling origin, never a single split.** Standing at the end of month *t*,
knowing nothing after it, predict *t+1 … t+h*; move one month forward; repeat.
A single train/test split is one draw — a calm test year flatters the model, a
discontinued product damns it. `tests/test_forecast.py` contains a spy model
that records the history it is handed at every origin and asserts no future
value ever leaks into training: that property is what the whole harness exists
to guarantee.

**Three metrics, because MAPE lies on this data.** MAPE divides by the actual,
so a month selling 3 units outweighs a month selling 30 000 — and
pharmaceutical portfolios are full of small months: launches, stockouts,
hospital products ordered in batches. So MAPE is reported (it is what people
ask for) next to **WAPE**, which weights by volume, and **MASE**, which is
scaled by the seasonal-naive error and is the only one comparable across
products of different sizes. Bias is separate because it is actionable:
being 10% low *every* month causes stockouts, while being 10% wrong in both
directions only costs safety stock.

**What the numbers above are worth: nothing.** The demo runs on the synthetic
dataset, whose series are generated as trend × seasonality × noise — which is
precisely the structure Holt-Winters assumes. The model is recovering its own
generating process, so winning is tautological. The table demonstrates that the
harness runs and that the ranking is coherent; **it is not evidence that
Holt-Winters forecasts real demand well.** Only a backtest on real history can
say that, which is exactly why the harness takes an arbitrary
`(series, date, value)` table and knows nothing about where it came from.

---

## Layout

```
src/pharma_text2sql/
├── config.py        settings, P2S_ prefixed
├── schema.py        column list + value index, cached on disk
├── entities.py      step 1 — question → structured intent (LLM or lexical)
├── resolver.py      step 2 — terms → real database values, or a question back
├── planner.py       step 4 — intent → SQL (LLM or template)
├── sql_guard.py     step 5 — AST validation, read-only, forced LIMIT
├── sandbox.py       out-of-process execution of generated Python
├── pipeline.py      the two phases: explain / execute
├── api.py           FastAPI surface
├── cli.py           command line
└── forecast/        demand forecasting, independent of the pipeline
    ├── models.py    naive, seasonal naive, moving average, drift, Holt-Winters
    ├── metrics.py   MAPE, WAPE, MASE, bias — and why three of them
    └── backtest.py  rolling-origin evaluation
data/generate.py     synthetic dataset generator
eval/                question set + scoring
tests/               59 tests
```

Configuration is environment-driven (`P2S_MODELS`, `P2S_MAX_ROWS`,
`P2S_SANDBOX_TIMEOUT`, …); see `config.py`.

---

## The data is invented

Molecules, corporations, products and figures are generated by
`data/generate.py` from a fixed seed. Only the ATC level-1 anatomical classes
are real, because the shape of the hierarchy is part of what makes the
resolution problem realistic. No market data, no company data, nothing
proprietary — the dataset exists to exercise the pipeline, and its numbers mean
nothing outside it.

## Known limits

- The offline planner is a template engine over a structured intent. It covers
  the evaluation set; it is not a general NL2SQL system, and it is not trying
  to be — the model path is.
- Joins are out of scope: the dataset is a single denormalised table, as market
  sales extracts usually are.
- Resource limits are POSIX; on Windows the sandbox keeps the process
  isolation, environment allowlist, audit hook and wall-clock timeout, but not
  the memory and CPU caps.
- Narration quality is not scored. The figures it quotes come from the result
  table, and the table is scored — but how well it reads is not measured, and
  claiming otherwise would need a judge model and a rubric.

## Licence

MIT.
