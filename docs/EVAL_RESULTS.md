# Eval results: `full-2026-10-08`, prompt `p-73b20568bab6`

Every number here comes from the recorded run in `evals/results/full-2026-10-08.*`. The calibration was fitted
and scored offline from that run and made no API calls.

**Prompt version `p-73b20568bab6`** (`queryguard.prompt_version`) hashes the model ids, every instruction text
(generation, second opinion, back-translation, judge), the few-shot examples and the rendered schema, column
comments included. Every row of the run records it. Any edit to any of these produces a new id. This version is
the first with the metric glossary in the generator, the schema comments and the alignment judge (see
[Fixed after first eval](#fixed-after-first-eval)).

```bash
uv run python -m evals.run_eval --live --run-id full-2026-10-08 --reuse-second-sql merged-2026-10-08 \
    --max-calls 480 --max-cost 1.60                      # the run: 450 calls, $1.336
uv run python -m evals.recompute --run-id full-2026-10-08   # labels, agreement, sanity: offline, free
uv run python -m evals.calibrate --run-id full-2026-10-08   # refit, rescore, redraw: free, no DB, no API
```

Outputs: `src/queryguard/validation/calibration.json` (runtime weights),
`evals/results/full-2026-10-08.calibration.json` (every metric below, plus each row's out-of-fold score) and
`docs/calibration.png`.

## The run

194 items, all from `evals/golden.yaml`:

| population | items | what ran under `p-73b20568bab6` | label |
|---|---|---|---|
| generated | 50 | the full pipeline (`run_question`): generation, back-translation, judge, second query | result compared with the golden result |
| mutation | 104 | back-translation and judge on known-wrong SQL from `evals/mutations.py` | wrong |
| golden | 40 | back-translation and judge on the golden SQL itself | correct |

**One exception to "one prompt version":** the second query. For the 144 mutation and golden rows it was reused,
not regenerated. 129 rows come from `live-2026-10-01`, with a generator prompt that had no glossary. 15 rows (the
refund mutations) come from `refund-fix-2026-10-08`, with the current generator prompt. The reused second queries
were re-executed, and agreement was re-derived offline under the current rules. Everything that reads the
glossary was re-run: generation, back-translation (through the schema comments) and the judge.

Models, from the call ledger: `claude-sonnet-5` (84 calls) writes the SQL, both the first and second query.
`claude-haiku-4-5` (366 calls) handles back-translation and the judge.

## 1. Generation accuracy

| category | questions | correct | notes |
|---|---|---|---|
| simple_lookup | 7 | 7 | |
| join | 7 | 7 | join_06 counted correct under a relaxed comparison (see limitations) |
| aggregation | 7 | 7 | agg_05 counted correct under a relaxed comparison (see limitations) |
| date_range | 7 | 6 | **date_05 asked for clarification** instead of answering |
| top_n | 6 | 6 | |
| refund_trap | 6 | 6 | refund_04 fixed (see [Fixed after first eval](#fixed-after-first-eval)) |
| ambiguous | 5 | 5 | 5/5 asked for clarification |
| unanswerable | 5 | 5 | 5/5 refused (`CannotAnswer`) |
| **total** | **50** | **49** | answerable: 39/40 · declines: 10/10 |

- **date_05** ("How many orders were shipped in August 2026?") came back as `ClarificationNeeded`. It was
  answered correctly in `live-2026-10-01`. The eval row doesn't store the readings the model offered, so this run
  can't say why it hesitated. The question doesn't mention revenue, so the glossary isn't an obvious cause.
- **All 5 unanswerable questions were refused.** In `live-2026-10-01`, unans_01 ("average time between shipping
  and delivery") ran a stand-in query. It is now refused: "there is no delivered-at timestamp anywhere".
- **ambig_01** ("revenue last quarter") and **ambig_03** ("which customers spent the most last year") still ask
  for clarification under the glossary. The glossary leaves gross-vs-net and "spent" open on purpose.

## 2. Calibration method

- **Fitting set: 183 rows** (79 correct, 104 wrong): every row with a feature vector. The 10 declines and date_05's
  clarification have no feature vector and stay in the accuracy report. Ambiguous and unanswerable items are never
  fitted. Their label records whether the pipeline declined, not whether an answer was right.
- **All 104 wrong rows are mutations.** The run's only wrong generated answer (date_05) is a clarification, so it
  has no score. The fit contains no organic model error.
- **Model:** logistic regression (scikit-learn, L2, `C=1.0`) on the 13-feature vector from `confidence.encode`.
  `C` was fixed before fitting.
- **Validation:** `GroupKFold`, 5 folds, grouped by golden question id (40 groups). A question's golden SQL, its
  mutations and its generated answer always land in the same fold. Every metric for a fitted model is
  **out-of-fold**. The v0 hand-set weights were never fitted, so they are scored as they stand.
- **Features that never fired.** `sanity_fail`, `sanity_info` and `alignment_missing` are zero on every fitting row
  and keep their v0 weights (see `evals/calibrate.py`).

### With or without self-confidence

| variant | OOF Brier | ECE | AUROC |
|---|---|---|---|
| with self-confidence | 0.04091 | 0.0638 | 0.984 |
| **without self-confidence (chosen)** | 0.04096 | 0.0639 | 0.983 |

Self-confidence is the injected 0.9 on all 144 mutation and golden rows. It varies only on generated rows, and every
one of those is correct, so it can only act as a "this is a generated row" shortcut. It gains 0.00005 Brier, well
below `MIN_BRIER_GAIN = 0.005`, so it stays out. That threshold was set after the first run. It is a judgement call,
not a pre-registered rule.

### Fitted weights (logit space)

| feature | v0 hand-set | calibrated |
|---|---|---|
| bias | −1.00 | +0.51 |
| self_confidence | +1.50 | 0 (dropped) |
| alignment_centered | +3.00 | +1.77 |
| alignment_missing | −0.30 | −0.30 (v0, never fired) |
| discrepancy_count | −0.50 | −1.12 |
| sanity_fail | −2.00 | −2.00 (v0, never fired) |
| sanity_warn | −0.70 | −0.74 |
| sanity_info | −0.10 | −0.10 (v0, never fired) |
| agreement_agree | +1.00 | +2.69 |
| agreement_disagree | −2.00 | −2.24 |
| agreement_incomparable | −0.30 | −1.30 |
| guardrail_rewrote | −0.10 | −0.19 |
| rows_empty | −0.50 | −0.49 |
| rows_capped | −0.30 | −0.17 |

## 3. v0 vs calibrated

183 rows. Calibrated numbers are out-of-fold.

| scorer | Brier ↓ | ECE (10 bins) ↓ | AUROC ↑ | wrong answers < 0.5 ↑ | correct answers < 0.5 (false flags) ↓ |
|---|---|---|---|---|---|
| v0 hand-set | 0.051 | 0.099 | **0.987** | 95.2% (99/104) | **5.1% (4/79)** |
| **calibrated (runtime)** | **0.041** | **0.064** | 0.983 | **97.1% (101/104)** | 7.6% (6/79) |

![Reliability diagram: v0 vs calibrated vs perfect calibration](calibration.png)

The calibrated score is the better probability: Brier is 19% lower and ECE 35% lower. It catches 2 more wrong
answers at 0.5, but it raises 2 more false flags, and v0 ranks slightly better. The 6 calibrated false flags are
gold:join_02 (0.21), gold:agg_05 (0.23), gold:refund_04 (0.29), gen:lookup_03 (0.23), gen:lookup_06 (0.47) and
gen:topn_06 (0.04). The 3 missed mutations are mut:lookup_06__literal_case (0.70), mut:join_03__column_swap (0.56)
and mut:topn_04__fan_out_join (0.67). Scores are bimodal: 152 of 183 fall below 0.2 or above 0.8.

## 4. Detectors

Share of mutations caught by each detector on its own, and by the score below 0.5 (calibrated out-of-fold).
Back-translation fires when alignment is below 0.7. Agreement fires on any outcome except "agree". Sanity fires on
any warn or fail.

| mutation | n | back-translation | agreement | sanity | any detector | calibrated < 0.5 | v0 < 0.5 |
|---|---|---|---|---|---|---|---|
| fan_out_join | 28 | 29% | 100% | 29% | 100% | 96% | 93% |
| drop_where | 25 | 84% | 76% | 16% | 100% | 100% | 96% |
| column_swap | 17 | 82% | 94% | 6% | 100% | 94% | 94% |
| agg_swap | 9 | 78% | 100% | 0% | 100% | 100% | 100% |
| date_shift | 8 | 100% | 88% | 0% | 100% | 100% | 100% |
| literal_case | 7 | 14% | 86% | 43% | 100% | 86% | 86% |
| order_flip | 5 | 100% | 100% | 20% | 100% | 100% | 100% |
| null_flip | 4 | 100% | 100% | 25% | 100% | 100% | 100% |
| inner_to_left | 1 | 0% | 100% | 0% | 100% | 100% | 100% |
| **all mutations** | **104** | **65%** (68) | **91%** (95) | **17%** (18) | **100%** (104) | **97%** (101) | **95%** (99) |
| **false flags on correct answers** | **79** | **7.6%** (6) | **6.3%** (5) | **3.8%** (3) | **16.5%** (13) | **7.6%** (6) | **5.1%** (4) |

- Every mutation trips at least one detector (104/104). Agreement is still the strongest single detector, and the
  one fan-out joins can't get past (28/28). Back-translation still can't see `literal_case` (1/7).
- The judge's 6 false flags include 4 that the glossary caused. See the regression note below.

## 5. Cost and latency

| population | items | calls | cost | median cost / item | median latency | p90 latency |
|---|---|---|---|---|---|---|
| generated (full pipeline) | 50 | 162 | $0.663 | $0.0135 | 10.2 s | 14.1 s |
| mutation (back-translation + judge) | 104 | 208 | $0.491 | $0.0047 | 4.0 s | 4.8 s |
| golden (back-translation + judge) | 40 | 80 | $0.182 | $0.0045 | 3.7 s | 4.4 s |
| **total** | **194** | **450** | **$1.336** | | 4.0 s | |

Costs come from the call ledger `evals/results/full-2026-10-08.llm_calls.jsonl` and match the per-row totals. The
caps were 480 calls and $1.60, and the run finished every item. Mutation and golden latencies leave out the
second query, which was reused, so they understate a full `run_answer`. Project spend on evals so far: $2.106
(`live-2026-10-01`, plus about $0.07 unlogged), $0.283 (`refund-fix-2026-10-08`) and $1.336 (this run).

## Fixed after first eval

### The blind spot

In `live-2026-10-01`, refund_04 ("gross revenue from orders placed in 2025, before refunds") summed every 2025
order, unpaid `pending` ones included: $6,177,714.13 against the golden $5,791,881.33. Every detector passed it,
with alignment 1.0, an agreeing second query and no sanity flag, and it scored 0.96 calibrated. Two `drop_where`
mutations (refund_01, refund_04) passed the same way. All three validators check the SQL against the *question*,
and none of them knew the *business rule* that unpaid orders aren't revenue.

### The fix

1. **A metric glossary** (`llm/glossary.py`). Revenue (gross) = `sum(orders.total_amount)` for
   `status IN ('paid','shipped','delivered','refunded')`. Pending and cancelled orders are not revenue. Net revenue
   also subtracts `refunds.amount`. It appears in the schema comment on `orders.total_amount`
   (`db/init/04_comments.sql`, re-applied in place, since `COMMENT ON` is idempotent) and in the generator's
   system prompt. Unqualified "revenue" (gross vs net) and "spent" stay open on purpose. No few-shot example was
   added.
2. **A `revenue_status` sanity check** (warn). A revenue or spend total that sums `orders.total_amount` without a
   status filter excluding pending and cancelled is flagged, and the flag quotes the rule.
3. **The glossary in the alignment judge**, which a correct status filter must not cost points. The judge had never
   seen the rule. In `refund-fix-2026-10-08` it scored the corrected refund_04 at 0.6 *for* applying it.

### Before / after

| item | `live-2026-10-01` (no glossary) | `refund-fix-2026-10-08` (glossary in generator) | `full-2026-10-08` (glossary in judge too) |
|---|---|---|---|
| gen:refund_04 | **wrong**; alignment 1.0; calibrated 0.96 | correct; alignment **0.6** (flagged) | **correct; alignment 1.0**; calibrated 0.98 |
| mut:refund_01__drop_where | missed; alignment 1.0, agree; 0.94 | caught; 0.02 | **caught**: alignment 0.4, disagree, revenue_status; 0.03 |
| mut:refund_04__drop_where | missed; alignment 1.0, agree; 0.96 | caught; 0.10 | **caught**: alignment 0.4, disagree, revenue_status; 0.02 |
| mut:topn_04__fan_out_join | caught at 0.497 | 0.594 (missed) | **0.67 (missed)**: alignment 1.0, only signal "incomparable" |
| gen:ambig_01 | clarification | not re-run | **clarification** |
| gen:ambig_03 | clarification | not re-run | **clarification** |

Calibrated scores are out-of-fold within their own run's fit. `refund-fix-2026-10-08` was scored inside the merged
set `merged-2026-10-08`, superseded by this run.

| fit | Brier ↓ | ECE ↓ | AUROC ↑ | wrong < 0.5 ↑ | false flags ↓ |
|---|---|---|---|---|---|
| `live-2026-10-01`, calibrated | 0.053 | 0.073 | 0.957 | 97.1% (102/105) | 5.1% (4/79) |
| `full-2026-10-08`, calibrated | **0.041** | **0.064** | **0.983** | 97.1% (101/104) | 7.6% (6/79) |

These two fits are not like-for-like. The first run's only organic wrong answer with a score was refund_04, which
is now correct, so every wrong row in the new fit is a mutation. Part of the better Brier and AUROC is that hard row
leaving.

### What the judge glossary made worse

The judge now applies the revenue rule to questions that never say "revenue". 4 of its 6 false flags on correct
answers are this:

| row | question | alignment before → after | judge's discrepancy |
|---|---|---|---|
| gold:date_02, gen:date_02 | "total order amount for orders placed in March 2026" | 1.0 / 0.95 → 0.4 | "total order amount … (revenue per glossary)" |
| gen:topn_03 | "10 largest orders by total amount" | 0.95 → 0.6 | "asks for revenue orders" |
| gold:agg_07 | line-item revenue after discounts | 0.9 → 0.6 | "includes pending orders" |

None of the four scores below 0.5. Agreement and the absence of other flags keep them up. They still lower the
score, and they make back-translation noisier. The next fix is to scope the judge's rule to questions that use a
glossary term ("revenue", "net revenue"), which needs another live run of the judge (about 194 calls, roughly
$0.20).

## Known limitations

- **The judge's glossary over-reaches** (above): 4 new alignment false flags on non-revenue questions.
- **date_05 now asks for clarification.** It was answered correctly before, and the stored row doesn't record why.
- **Reused second queries.** 129 of the 144 mutation and golden rows carry a second query generated under the
  pre-glossary prompt (see [The run](#the-run)). Agreement on those rows reflects that older generator.
- **Two post-hoc relaxations of the golden comparison** (commit 0ec0b33, made after the first pass of
  `live-2026-10-01` had been looked at):
  - **join_06** (`compare_columns: [product, category]`): the golden query also returns `order_item_id`, which
    the question doesn't ask for.
  - **agg_05** (`null_label_ok: true`): a NULL group may come back under one consistent label, such as
    `COALESCE(approved_by, 'auto-approved')`.

  Both still decide this run. Under strict comparison, both generated answers disagree with the golden result, and
  join and aggregation would each be 6/7. (The general rule that a generated answer may add columns, if it
  contains every golden column, applies to every question and isn't counted as question-specific.)
- **No organic errors in the fit.** All 104 wrong rows are mutations: known, mechanical error types. Real model
  errors look like refund_04 did: plausible, consistent and silent. Expect the calibrated probabilities to be
  overconfident on real traffic until a run with organic errors is labelled.
- **Self-confidence is untested as a signal.** It was dropped because this run can't measure it fairly. Nothing
  has shown it carries no information.
- **Never-fired features.** `sanity_fail`, `sanity_info` and `alignment_missing` keep hand-set weights.
- **One run, one schema.** 40 golden SQLs, 50 questions and one e-commerce database. The confidence intervals on
  all of the above are wide.
