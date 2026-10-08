# Eval results: `live-2026-10-01`

Every number here comes from the recorded run in `evals/results/live-2026-10-01.*`.
The calibration was fitted and scored offline from that run. Calibrating made no API calls.

```bash
uv run python -m evals.calibrate --run-id live-2026-10-01   # refit, rescore, redraw: free, no DB, no API
```

Outputs: `evals/results/live-2026-10-01.calibration.json` (every metric in sections 2–4, plus each row's
out-of-fold score).

**The refund_04 blind spot has since been fixed and re-measured.** See [Fixed after first eval](#fixed-after-first-eval).
The runtime weights (`src/queryguard/validation/calibration.json`) and `docs/calibration.png` now come from the
merged run `merged-2026-10-08`. Sections 1–5 still describe `live-2026-10-01` as recorded.

## The run

194 items, all from `evals/golden.yaml`:

| population | items | what it is | label |
|---|---|---|---|
| generated | 50 | every golden question through the full pipeline (`run_question`) | result compared with the golden result |
| mutation | 104 | known-wrong SQL from `evals/mutations.py`, through `run_answer` | wrong |
| golden | 40 | the golden SQL itself, through `run_answer` | correct |

Models, from the call ledger: `claude-sonnet-5` (216 calls) writes the SQL, both the first and second query. `claude-haiku-4-5` (370 calls) handles back-translation and the alignment judge.

## 1. Generation accuracy

| category | questions | correct | notes |
|---|---|---|---|
| simple_lookup | 7 | 7 | |
| join | 7 | 7 | join_06 counted correct under a relaxed comparison (see limitations) |
| aggregation | 7 | 7 | agg_05 counted correct under a relaxed comparison (see limitations) |
| date_range | 7 | 7 | |
| top_n | 6 | 6 | |
| refund_trap | 6 | 5 | **refund_04 wrong** (see limitations) |
| ambiguous | 5 | 5 | 5/5 asked for clarification |
| unanswerable | 5 | 5 | 4/5 refused (`CannotAnswer`), 1/5 answered with self-confidence 0.25 |
| **total** | **50** | **49** | answerable: 39/40 · declines: 10/10 |

**Clarification and refusal correctness.** All 5 ambiguous questions came back as `ClarificationNeeded`. None
produced SQL. Of the 5 unanswerable questions, 4 were refused with a correct reason, for example "no warehouse
table or column anywhere". The fifth, unans_01 ("average time between shipping and delivery"; there is no
`delivered_at`), ran a stand-in query (`shipped_at - order_date`) with self-confidence 0.25. The harness counts
self-confidence below 0.5 as a decline. The pipeline itself still returned an answer, but the calibrated score
gives that answer **0.016**, so a caller would see it flagged.

## 2. Calibration method

- **Fitting set: 184 rows** (79 correct, 105 wrong): every row with a feature vector except unans_01.
  Unanswerable and ambiguous items are labelled on whether the pipeline declined, not on whether their SQL was
  right, so their label doesn't mean "this answer is correct". The 9 clarifications and refusals have no feature
  vector. All 10 stay in the accuracy report above.
- **Model:** logistic regression (scikit-learn, L2, `C=1.0`) on the 13-feature vector from
  `confidence.encode`. `C` was fixed before fitting. With 184 rows, tuning it on the outer folds would leak.
- **Validation:** `GroupKFold`, 5 folds, grouped by golden question id (40 groups). A question's golden SQL, its
  mutations and its generated answer always land in the same fold. Every metric for a fitted model is
  **out-of-fold**. The v0 weights were never fitted, so they are scored on the same 184 rows as they stand.
- **Features that never fired.** `sanity_fail`, `sanity_info` and `alignment_missing` are zero on every fitting
  row. The fit has no evidence for them and gives them weight 0, which would switch those signals off at runtime.
  They keep their v0 weights instead. That leaves every fitted coefficient and every out-of-fold score unchanged,
  because the feature is zero on all of those rows.

### With or without self-confidence

| variant | OOF Brier | ECE | AUROC | self-confidence weight |
|---|---|---|---|---|
| with self-confidence | 0.05300 | 0.0728 | 0.958 | +0.21 |
| **without self-confidence (chosen)** | 0.05310 | 0.0730 | 0.957 | 0 |

The variant with self-confidence is better by **0.0001** Brier, and that gain comes from the shortcut.
Self-confidence is the injected 0.9 on all 144 mutation and golden rows. It varies only on the 40 generated
rows, and 39 of those are correct. A value other than 0.9 therefore means "generated row, almost certainly
correct". Grouping by question doesn't block this, because the shortcut works across populations, not across
questions. So the out-of-fold comparison is tilted toward the variant that has the shortcut.

The rule in `evals/calibrate.py` keeps self-confidence only if it improves out-of-fold Brier by more than
`MIN_BRIER_GAIN = 0.005`, about a tenth of the score. That threshold was set after seeing the margin, so it is
a judgement call. It is not a pre-registered rule. Adding self-confidence back needs a run where it varies on
wrong answers too, for example generated answers to harder questions.

### Fitted weights (logit space)

| feature | v0 hand-set | calibrated |
|---|---|---|
| bias | −1.00 | −0.40 |
| self_confidence | +1.50 | 0 (dropped) |
| alignment_centered | +3.00 | +2.02 |
| alignment_missing | −0.30 | −0.30 (v0, never fired) |
| discrepancy_count | −0.50 | −0.42 |
| sanity_fail | −2.00 | −2.00 (v0, never fired) |
| sanity_warn | −0.70 | −0.66 |
| sanity_info | −0.10 | −0.10 (v0, never fired) |
| agreement_agree | +1.00 | +2.07 |
| agreement_disagree | −2.00 | −2.33 |
| agreement_incomparable | −0.30 | −1.25 |
| guardrail_rewrote | −0.10 | +0.17 |
| rows_empty | −0.50 | −0.33 (1 row) |
| rows_capped | −0.30 | −0.30 |

The calibration moved weight from self-confidence onto agreement. An agreeing second query is now worth twice
what v0 gave it, and "incomparable" is treated as close to a disagreement.

## 3. v0 vs calibrated

184 rows. Calibrated numbers are out-of-fold.

| scorer | Brier ↓ | ECE (10 bins) ↓ | AUROC ↑ | wrong answers < 0.5 ↑ | correct answers < 0.5 (false flags) ↓ |
|---|---|---|---|---|---|
| v0 hand-set | 0.067 | 0.094 | **0.969** | 92.4% (97/105) | **3.8% (3/79)** |
| calibrated | **0.053** | **0.073** | 0.957 | **97.1% (102/105)** | 5.1% (4/79) |

![Reliability diagram: v0 vs calibrated vs perfect calibration](calibration.png)

*(The diagram has been redrawn from the merged run, see [Fixed after first eval](#fixed-after-first-eval). The
numbers in this section are the original fit's.)*

The calibrated score is the better probability: Brier is 20% lower and ECE 22% lower, and at the 0.5 threshold
it catches 5 more wrong answers. v0 is slightly better at *ranking* (AUROC 0.969 vs 0.957) and raises one fewer
false flag. The extra false flag is gold:refund_06, at 0.38. Scores are bimodal: 156 of 184 fall below 0.2 or
above 0.8. The middle bins of the diagram hold 2 to 5 answers each, so their points are noisy. The histogram
under the diagram shows the counts.

## 4. Detectors

Share of mutations caught by each detector on its own, and by the calibrated score below 0.5 (out-of-fold).
Back-translation fires when alignment is below 0.7. Agreement fires on any outcome except "agree". Sanity fires
on any warn or fail.

| mutation | n | back-translation | agreement | sanity | any detector | calibrated < 0.5 | v0 < 0.5 |
|---|---|---|---|---|---|---|---|
| fan_out_join | 28 | 29% | 100% | 29% | 100% | 100% | 89% |
| drop_where | 25 | 72% | 68% | 8% | 92% | 92% | 88% |
| column_swap | 17 | 88% | 94% | 6% | 100% | 100% | 94% |
| agg_swap | 9 | 78% | 100% | 0% | 100% | 100% | 100% |
| date_shift | 8 | 100% | 88% | 0% | 100% | 100% | 100% |
| literal_case | 7 | 0% | 86% | 43% | 100% | 100% | 100% |
| order_flip | 5 | 100% | 100% | 20% | 100% | 100% | 100% |
| null_flip | 4 | 100% | 100% | 25% | 100% | 100% | 100% |
| inner_to_left | 1 | 100% | 100% | 0% | 100% | 100% | 100% |
| **all mutations** | **104** | **63%** (66) | **89%** (93) | **15%** (16) | **98%** (102) | **98%** (102) | **93%** (97) |
| generated, wrong (refund_04) | 1 | 0% | 0% | 0% | 0% | 0% | 0% |
| **false flags on correct answers** | **79** | **2.5%** (2) | **5.1%** (4) | **3.8%** (3) | **11.4%** (9) | **5.1%** (4) | **3.8%** (3) |

- The detectors complement each other. Back-translation is blind to `literal_case` (0/7), because the question
  reads the same whichever case the literal is in. Agreement catches 6 of those 7. Agreement is the strongest
  single detector, and it is the one fan-out joins can't get past (28/28).
- The calibrated score catches exactly as many mutations as the union of the detectors (102/104), with fewer than
  half the false flags (4/79 vs 9/79).
- The 2 mutations nothing catches are both `drop_where` on refund questions (refund_01, refund_04). Dropping the
  order-status filter gives a query that back-translates to the same question, and that a second query
  reproduces. It is the same blind spot as the generated refund_04 (see limitations).

## 5. Cost and latency

| population | items | calls | cost | median cost / item | median latency | p90 latency |
|---|---|---|---|---|---|---|
| generated (full pipeline) | 50 | 168 | $0.728 | $0.0141 | 9.7 s | 14.9 s |
| mutation | 104 | 303 | $1.010 | $0.0096 | 6.7 s | 9.0 s |
| golden | 40 | 115 | $0.368 | $0.0093 | 6.5 s | 9.1 s |
| **total** | **194** | **586** | **$2.106** | | 7.1 s | |

Costs come from the call ledger `evals/results/live-2026-10-01.llm_calls.jsonl`, and they match the per-row
totals. The first pass also made 4 unlogged calls (about $0.07), in the unanswerable refusals that failed to
parse before commit 0ec0b33. True spend was about 590 calls and **about $2.18**. Latency is the wall-clock time of the
whole pipeline call for an item, API calls and query execution included.

## Fixed after first eval

### What changed (2026-10-08)

The fix states the business rule. It adds no validator signal. Two places:

1. **A metric glossary.** Revenue (gross) = `sum(orders.total_amount)` for
   `status IN ('paid','shipped','delivered','refunded')`. Pending and cancelled orders are **not** revenue. Net revenue
   also subtracts `refunds.amount`. The rule is in the schema comment on `orders.total_amount` (`db/init/04_comments.sql`,
   re-applied to the running database, `COMMENT ON` is idempotent), so it reaches every prompt through the
   introspected schema. It is also in a new **Glossary** section of the system prompt (`llm/prompt.py`). Unqualified
   "revenue" is still a gross-or-net judgement call, and "spent" is left undefined, so the ambiguous questions keep
   their forks. No few-shot example was added.
2. **A `revenue_status` sanity check** (`validation/sanity.py`, warn). It fires when a revenue or spend total sums
   `orders.total_amount` and no status filter excludes `pending` and `cancelled`. The flag quotes the glossary rule.
   It reads the SQL, not the rows, because the rows of a wrong revenue total look like any other number.

### Re-run: `refund-fix-2026-10-08`

Only the refund category went back through the live pipeline: its 6 questions and their 15 mutations, under the new
prompt and schema. The golden refund SQLs were not re-run.

```bash
uv run python -m evals.run_eval --live --run-id refund-fix-2026-10-08 --category refund_trap \
    --population generated --population mutation --max-calls 75 --max-cost 0.50 --concurrency 1
```

21 items, **69 calls, $0.283** from the ledger `evals/results/refund-fix-2026-10-08.llm_calls.jsonl`, under caps of
75 calls and $0.50. The dry-run projection was 69 calls (worst case, every step) and $0.25.

The sanity check is deterministic, so `evals.recompute` now also re-derives sanity flags offline. It was applied to
all 194 rows of `live-2026-10-01` at no cost. It flags exactly 3 rows: gen:refund_04 and the two `drop_where`
mutations. All 3 are wrong. It flags none of the 80 correct answers.

### Before / after

| item | | alignment | agreement | sanity | v0 | calibrated (OOF) | label |
|---|---|---|---|---|---|---|---|
| gen:refund_04 | before | 1.0 | agree | — | 0.94 | 0.96 | **wrong**: $6,177,714.13, pending orders included |
| | after | 0.6 (flagged) | agree | — | 0.76 | 0.88 | **correct**: the golden SQL, status for status |
| mut:refund_01__drop_where | before | 1.0 | agree | — | 0.94 | 0.94 | wrong, **missed** |
| | after | 0.4 | disagree | revenue_status | 0.04 | 0.02 | wrong, **caught** by all 3 detectors |
| mut:refund_04__drop_where | before | 1.0 | agree | — | 0.94 | 0.96 | wrong, **missed** |
| | after | 1.0 | disagree | revenue_status | 0.28 | 0.10 | wrong, **caught** by agreement and sanity |

- **refund_04 is correct.** The model applied the glossary's status filter. Refund generation is now **6/6**
  (was 5/6).
- **Both `drop_where` mutations are caught**, by two independent routes. The new sanity check flags them. The second
  query now applies the glossary on its own, so it disagrees with the mutated SQL. Before the fix it repeated the
  same mistake and agreed.
- **The other 13 refund mutations** stay caught under the calibrated score (all below 0.27). Under v0 they are all below
  0.5 too. mut:refund_06__fan_out_join was a v0 miss before (0.81) and is 0.44 now.

### Recalibrated on the merged run

`evals.merge` builds `merged-2026-10-08`: the 194 `live-2026-10-01` rows, with the 21 re-run items replaced by their
new rows. `evals.calibrate` was refitted on it. Same method, same `C`, same grouped folds.

| fit | scorer | Brier ↓ | ECE ↓ | AUROC ↑ | wrong < 0.5 ↑ | false flags ↓ |
|---|---|---|---|---|---|---|
| live-2026-10-01 | v0 hand-set | 0.067 | 0.094 | 0.969 | 92.4% (97/105) | 3.8% (3/79) |
| live-2026-10-01 | calibrated | 0.053 | 0.073 | 0.957 | 97.1% (102/105) | 5.1% (4/79) |
| merged-2026-10-08 | v0 hand-set | 0.050 | 0.117 | 0.988 | 96.2% (100/104) | 3.8% (3/80) |
| merged-2026-10-08 | **calibrated (now at runtime)** | **0.035** | **0.068** | **0.984** | **99.0% (103/104)** | 5.0% (4/80) |

**These rows are not a like-for-like improvement.** The fitting set changed under the fix. Its only organic wrong
answer (refund_04) is now correct, so all 104 wrong rows are mutations, and the three rows that scored 0.94–0.96
while wrong are now resolved. Most of the gain in Brier and AUROC is those three rows. Self-confidence is still dropped
(gain 0.00004, needs > 0.005). The weights moved modestly. Agreement and discrepancies count for more (agree +2.07 →
+2.65, discrepancy −0.42 → −0.77), and a sanity warning now costs −0.82 (was −0.66).

### What the fix made worse, or left open

- **The alignment judge now false-flags the correct refund_04 (0.6).** Its discrepancy reads: *"original asks for gross
  revenue from all orders … the query filters to only … paid, shipped, delivered, or refunded"*. The judge compares
  the question with the back-translation, and it is never shown the schema or the glossary, so it treats the business
  rule as a deviation. Agreement keeps the score at 0.88, but the detector itself is wrong here. Next fix: give the
  judge the glossary.
- **One mutation slipped back over the line.** mut:topn_04__fan_out_join (not a refund item, not re-run) moved from
  0.497 to 0.594 under the refitted weights. Its only signal is an `incomparable` agreement. It sat on the threshold
  before, and it now sits just above it.
- gold:refund_06 is still a calibrated false flag (0.38 → 0.48).
- **Only the refund category was re-run under the new prompt.** The other 44 generated answers, and every golden
  and non-refund mutation row, were scored under the old prompt. A glossary in every prompt could change those
  answers as well (ambig_01 asks about "revenue"), and this run does not measure that. The merged set mixes two
  prompt versions.

## Known limitations

- **refund_04: a business-rule blind spot** *(fixed; see [Fixed after first eval](#fixed-after-first-eval))*. For "gross revenue from orders placed in 2025, before refunds",
  the model summed every 2025 order, including unpaid `pending` ones: $6,177,714.13 against the golden
  $5,791,881.33. Every detector passed it: alignment 1.0, the second query made the same choice and agreed,
  and no sanity flag fired. The answer scores 0.94 under v0 and 0.96 calibrated. Two `drop_where` mutations
  fail the same way. All three validators check that the SQL matches the *question*. None of them applies the
  *business rule* that unpaid orders aren't revenue. The schema comment on `orders.status` says
  `pending=unpaid`, but nothing says gross revenue excludes unpaid orders, so the model has to infer that and it
  didn't. The fix is a stated revenue definition in the schema comments or the prompt, followed by a fresh run.
  More validator signal won't catch it.
- **Two post-hoc relaxations of the golden comparison** (commit 0ec0b33, made after the first pass of this run
  had been looked at):
  - **join_06** (`compare_columns: [product, category]`): the golden query also returns `order_item_id`, which
    the question doesn't ask for. Only the product and category columns must match.
  - **agg_05** (`null_label_ok: true`): the model returned the auto-approved group as
    `COALESCE(approved_by, 'auto-approved')` instead of NULL. A NULL group may now come back under one
    consistent label.

  Both were judged as answering the question as asked. Both still loosen the comparison after the results were
  seen, so the generated accuracy above depends on them. Without them, join and aggregation would be 6/7 each.
  (The same commit's general rule, that a generated answer may add columns if it contains every golden column,
  relabelled 12 more answers. It applies to every question, so it isn't counted as a question-specific relaxation.)
- **Self-confidence is untested as a signal.** It was dropped because this run can't measure it fairly, not
  because it was shown to carry no information (see section 2).
- **Small, synthetic negatives.** 104 of the 105 wrong answers in the fitting set are mutations:
  known, mechanical error types. Real model errors look like refund_04 (plausible, consistent and
  silent), and this run has one of them. The calibrated probabilities describe this mix. Expect them to be
  overconfident on real traffic until a run with more organic errors is labelled.
- **Never-fired features.** `sanity_fail`, `sanity_info` and `alignment_missing` keep hand-set weights.
  `rows_empty` is fitted from a single row.
- **unans_01 counts as a correct decline** under the harness's self-confidence rule, but the pipeline did return
  a stand-in answer (see section 1).
- **One run, one schema.** 40 golden SQLs, 50 questions and one e-commerce database. The confidence intervals on
  all of the above are wide.
