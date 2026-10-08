# Eval results: `final-2026-10-08`, prompt `p-02ea2fb3b358`

> **Frozen.** These numbers are final for prompt version `p-02ea2fb3b358`. This golden set has already been used
> to find and fix two problems: refund_04 and the judge's glossary scope. More detector or prompt tuning against
> the same 50 questions and 104 mutations would overfit to them, so none will be done. A further change needs a
> fresh, held-out question set to be measured honestly.

Every number here comes from recorded runs in `evals/results/`. The calibration was fitted and scored offline and
made no API calls.

**Prompt version `p-02ea2fb3b358`** (`queryguard.prompt_version`) hashes the model ids, every instruction text
(generation, second opinion, back-translation, judge and the judge's glossary rule), the few-shot examples and
the rendered schema, column comments included. Every eval row records it, and any edit gives a new id.

## The runs behind the numbers

| run | prompt | what ran | calls | cost |
|---|---|---|---|---|
| `full-2026-10-08` | `p-73b20568bab6` | 50 questions through the full pipeline. Back-translation and judge on the 104 mutation and 40 golden SQLs | 450 | $1.336 |
| **`final-2026-10-08`** | **`p-02ea2fb3b358`** | the alignment judge only, re-run on every stored back-translation (183 calls). Everything else copied from `full-2026-10-08` | 183 | $0.187 |

The two prompt versions differ **only in the judge's instructions**: the glossary rule is now scoped to questions
that use a glossary term. Generation, the second-opinion prompt, back-translation, the examples and the schema
are byte-identical. So every judged, generated and back-translated output behind these numbers is what
`p-02ea2fb3b358` produces.

```bash
uv run python -m evals.run_eval --live --run-id full-2026-10-08 --reuse-second-sql merged-2026-10-08 \
    --max-calls 480 --max-cost 1.60
uv run python -m evals.run_eval --live --run-id final-2026-10-08 --judge-only full-2026-10-08 \
    --max-calls 200 --max-cost 0.25 --concurrency 1
uv run python -m evals.recompute --run-id final-2026-10-08   # labels, agreement, sanity: offline, free
uv run python -m evals.calibrate --run-id final-2026-10-08   # refit, rescore, redraw: free, no DB, no API
```

**One exception: the second query.** For the 144 mutation and golden rows it was reused, not regenerated.
129 rows come from `live-2026-10-01`, whose generator prompt had no glossary. 15 rows (the refund mutations) come
from `refund-fix-2026-10-08`, whose generator prompt is the current one. The reused queries were re-executed, and
agreement was re-derived offline under the current rules. This shows up in the results: gold:refund_04 is a false
flag because its reused pre-glossary second query repeats the old pending-orders mistake and disagrees (see
section 3).

The population table:

| population | items | label |
|---|---|---|
| generated | 50 | result compared with the golden result |
| mutation | 104 | known-wrong SQL from `evals/mutations.py`: wrong |
| golden | 40 | the golden SQL itself: correct |

Models: `claude-sonnet-5` writes the first and second query. `claude-haiku-4-5` handles back-translation and the judge.

Outputs: `src/queryguard/validation/calibration.json` (runtime weights),
`evals/results/final-2026-10-08.calibration.json` (every metric below, plus each row's out-of-fold score) and
`docs/calibration.png`.

## 1. Generation accuracy

| category | questions | correct | notes |
|---|---|---|---|
| simple_lookup | 7 | 7 | |
| join | 7 | 7 | join_06 counted correct under a relaxed comparison (see limitations) |
| aggregation | 7 | 7 | agg_05 counted correct under a relaxed comparison (see limitations) |
| date_range | 7 | 6 | **date_05 asked for clarification** instead of answering (see below) |
| top_n | 6 | 6 | |
| refund_trap | 6 | 6 | refund_04 fixed (see [Fixed after first eval](#fixed-after-first-eval)) |
| ambiguous | 5 | 5 | 5/5 asked for clarification |
| unanswerable | 5 | 5 | 5/5 refused (`CannotAnswer`) |
| **total** | **50** | **49** | answerable: 39/40 · declines: 10/10 |

- **All 5 unanswerable questions were refused.** In `live-2026-10-01`, unans_01 ("average time between shipping and
  delivery") ran a stand-in query. It is now refused: "there is no delivered-at timestamp anywhere".
- **ambig_01** ("revenue last quarter") and **ambig_03** ("which customers spent the most last year") still ask for
  clarification. The glossary leaves gross-vs-net and "spent" open on purpose.

### Why date_05 asked for clarification

"How many orders were shipped in August 2026?" was answered correctly on 2026-10-01 and returned
`ClarificationNeeded` in `full-2026-10-08`. The readings the model offered are **not recorded**. Eval rows don't
store a clarification's interpretations, and the call log stores no response text. So the cause can't be read
back, only inferred. Investigated offline, at no cost:

- **The fork is real and material.** "Shipped" can mean the event (`shipped_at` in August: **93** orders, the
  golden answer) or the status (`status = 'shipped'`, still in transit: **19** of those 93; the other 74 have since
  been delivered or refunded). The schema supports both readings.
- **The prompt pulls toward the status reading.** Its ambiguity rule says an enumerated status settles a term
  ("'cancelled orders' means status = 'cancelled'"), and `shipped` is a status value. The golden note says the
  opposite for this question: status = 'shipped' is wrong.
- **It was already borderline.** On 2026-10-01 the model answered with self-confidence **0.75**, the second-lowest
  of the 40 answerable questions. The generation call this time wrote 869 output tokens against 396 then, which
  is consistent with writing out two or three readings.
- **Probably not caused by the glossary.** The glossary doesn't mention shipping, though it does list `shipped`
  among the revenue statuses. The most likely explanation is a question sitting on the ambiguity threshold that
  tipped the other way on one sample. That can't be confirmed without the stored readings.

Not fixed, per the freeze. Storing a clarification's readings in eval rows would make the next such case
diagnosable.

## 2. Calibration method

- **Fitting set: 183 rows** (79 correct, 104 wrong): every row with a feature vector. Declines and date_05's
  clarification have no feature vector and stay in the accuracy report.
- **All 104 wrong rows are mutations.** The run's only wrong generated answer (date_05) is a clarification and has no
  score. The fit contains no organic model error.
- **Model:** logistic regression (scikit-learn, L2, `C=1.0`) on the 13-feature vector from `confidence.encode`.
  `C` was fixed before fitting.
- **Validation:** `GroupKFold`, 5 folds, grouped by golden question id (40 groups). A question's golden SQL, its
  mutations and its generated answer always land in the same fold. Every metric for a fitted model is
  **out-of-fold**. The v0 hand-set weights were never fitted, so they are scored as they stand.
- **Features that never fired** (`sanity_fail`, `sanity_info`, `alignment_missing`) keep their v0 weights.

### With or without self-confidence

| variant | OOF Brier | ECE | AUROC |
|---|---|---|---|
| with self-confidence | 0.0383 | 0.0714 | 0.983 |
| **without self-confidence (chosen)** | 0.0384 | 0.0715 | 0.983 |

Self-confidence is the injected 0.9 on every mutation and golden row, and every generated row is correct. So it
can only act as a "this is a generated row" shortcut. It gains 0.00009 Brier, below `MIN_BRIER_GAIN = 0.005`, so
it stays out. That threshold was set after the first run. It is a judgement call, not a pre-registered rule.

### Fitted weights (logit space)

| feature | v0 hand-set | calibrated |
|---|---|---|
| bias | −1.00 | +0.50 |
| self_confidence | +1.50 | 0 (dropped) |
| alignment_centered | +3.00 | +1.85 |
| alignment_missing | −0.30 | −0.30 (v0, never fired) |
| discrepancy_count | −0.50 | −1.09 |
| sanity_fail | −2.00 | −2.00 (v0, never fired) |
| sanity_warn | −0.70 | −0.66 |
| sanity_info | −0.10 | −0.10 (v0, never fired) |
| agreement_agree | +1.00 | +2.60 |
| agreement_disagree | −2.00 | −2.33 |
| agreement_incomparable | −0.30 | −1.24 |
| guardrail_rewrote | −0.10 | −0.16 |
| rows_empty | −0.50 | −0.30 |
| rows_capped | −0.30 | −0.16 |

## 3. v0 vs calibrated

183 rows. Calibrated numbers are out-of-fold.

| scorer | Brier ↓ | ECE (10 bins) ↓ | AUROC ↑ | wrong answers < 0.5 ↑ | correct answers < 0.5 (false flags) ↓ |
|---|---|---|---|---|---|
| v0 hand-set | 0.051 | 0.109 | **0.987** | 95.2% (99/104) | **5.1% (4/79)** |
| **calibrated (runtime)** | **0.038** | **0.072** | 0.983 | **99.0% (103/104)** | 7.6% (6/79) |

![Reliability diagram: v0 vs calibrated vs perfect calibration](calibration.png)

The calibrated score is the better probability: Brier is 24% lower, ECE 34% lower, and it catches 4 more wrong
answers at 0.5. v0 ranks slightly better and raises 2 fewer false flags. Scores are bimodal: 149 of 183 fall below
0.2 or above 0.8.

- **The one miss:** mut:join_03__column_swap (0.59).
- **The 6 false flags:**
  - gold:join_02 (0.20), gold:agg_05 (0.09) and gen:topn_06 (0.04), as in earlier runs;
  - gold:refund_04 (0.27): its reused pre-glossary second query disagrees;
  - gen:lookup_04 (0.48) and gen:lookup_06 (0.50): the judge listed an extra column as a discrepancy, which its
    instructions say it shouldn't, and these queries have no second query to outweigh it.

## 4. Detectors

Share of mutations caught by each detector on its own, and by the score below 0.5 (calibrated out-of-fold).
Back-translation fires when alignment is below 0.7. Agreement fires on any outcome except "agree". Sanity fires on
any warn or fail.

| mutation | n | back-translation | agreement | sanity | any detector | calibrated < 0.5 | v0 < 0.5 |
|---|---|---|---|---|---|---|---|
| fan_out_join | 28 | 36% | 100% | 29% | 100% | 100% | 93% |
| drop_where | 25 | 80% | 76% | 16% | 100% | 100% | 100% |
| column_swap | 17 | 82% | 94% | 6% | 100% | 94% | 94% |
| agg_swap | 9 | 67% | 100% | 0% | 100% | 100% | 89% |
| date_shift | 8 | 100% | 88% | 0% | 100% | 100% | 88% |
| literal_case | 7 | 14% | 86% | 43% | 100% | 100% | 100% |
| order_flip | 5 | 100% | 100% | 20% | 100% | 100% | 100% |
| null_flip | 4 | 100% | 100% | 25% | 100% | 100% | 100% |
| inner_to_left | 1 | 0% | 100% | 0% | 100% | 100% | 100% |
| **all mutations** | **104** | **65%** (68) | **91%** (95) | **17%** (18) | **100%** (104) | **99%** (103) | **95%** (99) |
| **false flags on correct answers** | **79** | **1.3%** (1) | **6.3%** (5) | **3.8%** (3) | **10.1%** (8) | **7.6%** (6) | **5.1%** (4) |

Every mutation trips at least one detector. Agreement is the strongest single detector, and the one fan-out joins
can't get past (28/28). Back-translation can't see `literal_case` (1/7), because the question reads the same
whichever case the literal is in.

## 5. Cost and latency

| run | population | items | calls | cost | median latency | p90 latency |
|---|---|---|---|---|---|---|
| `full-2026-10-08` | generated (full pipeline) | 50 | 162 | $0.663 | 10.2 s | 14.1 s |
| `full-2026-10-08` | mutation + golden (back-translation + judge) | 144 | 288 | $0.673 | ~3.9 s | ~4.8 s |
| `final-2026-10-08` | judge only, every back-translated row | 183 | 183 | $0.187 | 1.7 s | 2.1 s |

Costs come from the call ledgers `evals/results/<run>.llm_calls.jsonl` and match the per-row totals. Mutation and
golden latencies leave out the reused second query. Total logged eval spend so far is $3.91:
`live-2026-10-01` $2.106 (plus about $0.07 unlogged), `refund-fix-2026-10-08` $0.283, `full-2026-10-08` $1.336,
`final-2026-10-08` $0.187.

## Fixed after first eval

### The blind spot

In `live-2026-10-01`, refund_04 ("gross revenue from orders placed in 2025, before refunds") summed every 2025
order, unpaid `pending` ones included: $6,177,714.13 against the golden $5,791,881.33. Every detector passed it,
with alignment 1.0, an agreeing second query and no sanity flag, and it scored 0.96 calibrated. Two `drop_where`
mutations (refund_01, refund_04) passed the same way. All three validators check the SQL against the *question*,
and none of them knew the *business rule* that unpaid orders aren't revenue.

### The fix, in three steps

1. **A metric glossary** (`llm/glossary.py`). Revenue (gross) = `sum(orders.total_amount)` for
   `status IN ('paid','shipped','delivered','refunded')`. Pending and cancelled orders are not revenue. Net revenue
   also subtracts `refunds.amount`. It appears in the schema comment on `orders.total_amount`
   (`db/init/04_comments.sql`, re-applied in place) and in the generator's system prompt. Unqualified "revenue"
   (gross vs net) and "spent" stay open on purpose. No few-shot example was added.
   Plus a **`revenue_status` sanity check** (warn): a revenue or spend total that sums `orders.total_amount`
   without a status filter excluding pending and cancelled is flagged, and the flag quotes the rule.
2. **The glossary in the alignment judge.** The judge had never seen the rule and scored the corrected refund_04
   at 0.6 *for* applying it. Given the glossary on every question, it then read the rule into questions that never
   said revenue ("total order amount", "largest orders by total amount"), adding 4 false flags.
3. **The judge's rule scoped to questions that use a glossary term** ("revenue"), with the question's own stated
   filter or metric taking precedence ("line-item revenue … excluding cancelled orders"). This is the last change
   before the freeze.

### Before / after

| item | `live-2026-10-01` | `refund-fix-2026-10-08` | `full-2026-10-08` | **`final-2026-10-08`** |
|---|---|---|---|---|
| gen:refund_04 | **wrong**; alignment 1.0; 0.96 | correct; alignment 0.6 | correct; alignment 1.0; 0.98 | **correct; alignment 1.0; 0.98** |
| mut:refund_01__drop_where | missed; 0.94 | caught; 0.02 | caught; 0.03 | **caught**: alignment 0.4, disagree, revenue_status; 0.03 |
| mut:refund_04__drop_where | missed; 0.96 | caught; 0.10 | caught; 0.02 | **caught**: alignment 0.4, disagree, revenue_status; 0.02 |
| mut:topn_04__fan_out_join | caught at 0.497 | 0.594 (missed) | 0.669 (missed) | **caught at 0.35** |
| gen:ambig_01 | clarification | not re-run | clarification | **clarification** |
| gen:ambig_03 | clarification | not re-run | clarification | **clarification** |
| judge false flags (alignment < 0.7 on correct answers) | 2/79 | not comparable | 6/79 | **1/79** |

The four non-revenue rows the unscoped judge had flagged are back: gen:date_02 0.4 → 1.0, gold:date_02 0.4 → 0.85,
gen:topn_03 0.6 → 1.0 and gold:agg_07 0.6 → 0.7. Calibrated scores are out-of-fold within each run's own fit.

| fit | Brier ↓ | ECE ↓ | AUROC ↑ | wrong < 0.5 ↑ | false flags ↓ |
|---|---|---|---|---|---|
| `live-2026-10-01`, calibrated | 0.053 | 0.073 | 0.957 | 97.1% (102/105) | 5.1% (4/79) |
| `full-2026-10-08`, calibrated | 0.041 | 0.064 | 0.983 | 97.1% (101/104) | 7.6% (6/79) |
| **`final-2026-10-08`, calibrated** | **0.038** | 0.072 | **0.983** | **99.0% (103/104)** | 7.6% (6/79) |

These fits are not like-for-like. The first run's one scored organic error (refund_04) is now correct, so every
wrong row since is a mutation. Part of the better Brier and AUROC is that hard row leaving.

## Known limitations

- **Frozen on this set.** The two fixes above were found *and* measured on the same golden set, so these numbers
  flatter the fixes. Only a held-out set can say how they generalise.
- **date_05 now asks for clarification** (see section 1). The cause is inferred, not recorded.
- **Reused second queries.** 129 of the 144 mutation and golden rows carry a second query generated under the
  pre-glossary prompt. gold:refund_04's false flag comes from exactly this.
- **The judge still sometimes lists extra columns as discrepancies**, against its instructions. The two
  borderline false flags (lookup_04, lookup_06) come from this.
- **Two post-hoc relaxations of the golden comparison** (commit 0ec0b33, made after the first pass of
  `live-2026-10-01` had been looked at):
  - **join_06** (`compare_columns: [product, category]`): the golden query also returns `order_item_id`, which
    the question doesn't ask for.
  - **agg_05** (`null_label_ok: true`): a NULL group may come back under one consistent label, such as
    `COALESCE(approved_by, 'auto-approved')`.

  Both still decide this run. Under strict comparison, both generated answers disagree with the golden result, and
  join and aggregation would each be 6/7.
- **Golden vs glossary.** agg_07, topn_01 and topn_05 ask for "line-item revenue … excluding cancelled orders",
  and their golden SQL keeps pending orders. The questions state their own filter, and the judge now defers to it,
  but the golden set and the glossary define "revenue" differently there.
- **No organic errors in the fit.** All 104 wrong rows are mutations. Real model errors look like refund_04 did:
  plausible, consistent and silent. Expect the calibrated probabilities to be overconfident on real traffic.
- **Self-confidence is untested as a signal**, and the never-fired features keep hand-set weights.
- **A known race in the harness:** the budget reads the call ledger while another worker thread may be appending
  to it. A dry run with fast fake calls crashed once on a partial line. Live runs aren't known to be affected,
  and the judge-only run used `--concurrency 1`.
- **One schema.** 40 golden SQLs, 50 questions and one e-commerce database. The confidence intervals are wide.
