# 09 — Model experiments, decision policy & validation

Scripts: `scripts/43_models.py` (E1–E5), `scripts/48_e6_numfeat.py` (E6), `scripts/46_errors.py` (error analysis).
Outputs: `analysis/out/43_models.txt|json`, `48_e6.json`, `46_errors.txt`; models `work/dev_lgb_all*.txt`;
error tables `work/dev_holdout_fp.parquet`, `work/dev_holdout_fn.parquet`.

## Setup (no test data used)

* 43,888 eval S1 (2% hash sample of train S1) split **by S1 entity**: 60% train (1.60M pairs, 90k positives),
  20% valid (536k pairs, 30k pos) for model selection/thresholds, 20% holdout (535k, 30k pos) reported once.
* Metric = leaderboard macro-F0.5 over **all** S1 of the split (singletons: 1 if empty prediction; GT matches missed by
  blocking count as FN). Thresholds chosen on valid only.

## E1 — feature groups (LightGBM) → see 08. All features 0.9638 valid.

## E2 — model families (same features, valid split)

| model | AP | ROC-AUC | macro-F0.5 | micro P | micro R | singleton acc | train time |
|---|---|---|---|---|---|---|---|
| Logistic regression (standardized) | 0.9814 | 0.99859 | 0.9291 | 0.968 | 0.879 | 0.862 | 3 s |
| **LightGBM** (63 leaves, lr 0.05, early stop) | 0.9943 | 0.99959 | **0.9638** | 0.988 | 0.925 | 0.949 | ~30 s |
| XGBoost (hist, depth 8) | 0.9944 | 0.99960 | 0.9636 | 0.989 | 0.923 | 0.953 | 7,708 s* |
| MLP (128,64), 50% of train rows | 0.9918 | 0.99941 | 0.9531 | 0.979 | 0.917 | 0.903 | 39 s |

\*XGBoost ran while macOS `fileproviderd`/Spotlight saturated the CPU (load avg 19 on 11 cores); timing is not
representative, but it brings no accuracy gain either. CatBoost is not installed (not needed: all features are numeric).

**Decision: gradient-boosted trees (LightGBM).** The FNN/MLP is *worse* (−0.011) on identical features: the signal is in
thresholded interactions of a few dozen engineered similarities, which trees capture natively. Logistic regression
underfits the interactions (−0.035).

## E3 — decision policy

Threshold curve (valid, macro-F0.5): 0.30→0.9457, 0.50→0.9590, 0.60→0.9625, **0.65–0.80 → 0.963–0.964 (flat plateau)**,
0.90→0.9563, 0.95→0.9457. Optimum 0.72; the plateau is wide, so choose ~0.75 to buy precision margin for the test
distractor shift (test has 5.75 pool records per S1 vs 4.68 in train).

"Each pool record → only its best S1" (argmax) and argmax+margin gave **no measurable change here** — *because the dev
set is a 2% S1 sample*, so the competing S1s of a pool record are rarely inside it. The competition features
(`r_margin_best`, `rrank`, computed against the full S1 index) already carry most of this signal. Error analysis shows
the argmax constraint *would* fix several of the most confident false positives (e.g. `pune business private limited`
assigned to `Pune Water Pvt Ltd` while an S1 named exactly `Pune Business Private Limited` shares the address;
empty-address `Wood's Family Practice` with two same-named S1). **Must be re-evaluated on a full-partition validation.**

## E4 — per-segment thresholds (country × source)

Tuned on valid: India-S2 0.64, India-S3 0.74, US-S2 0.68, US-S3 0.76. Holdout: 0.9619 vs global 0.9625 → **no gain;
use one global threshold** (also safer for France, which has no segment of its own).

## E5 — France proxy (state unknown)

Holdout: normal 0.9625; state feature blanked at inference 0.9622; trained with 25% state dropout 0.9615–0.9618;
model without state 0.9619. **State features contribute ~0.0005; blanking them is harmless** → France's missing state is
not a risk for the model. (France's *real* risks are unseen address grammar and name patterns; see 10.)

## E6 — house-number closeness

Adding min digit edit distance (truncation = 1) and first-number edit distance: holdout **0.9625 → 0.9642**,
micro recall 0.9245 → 0.9319 at precision 0.9888 → 0.9865.

## Error analysis (holdout, threshold 0.72, before E6)

* Upper bound with a perfect matcher on these candidates: **0.9962** → the model loses 3.4 points, mostly recall.
* Positives accepted by segment: normal 94.9%, indic 98.2%, domain 98.3%, brand 93.5%, weak address 86.1%,
  weak name 85.3%, **empty pool address 52.2%**.
* False positives: 318 vs 28,062 TP; 80% are unmatched distractors, 20% belong to another S1.
* 21 of 463 singleton S1 received a false match (4.5%). They are near-twins created by the distractor generator:
  same name ±1 letter (`PG`/`PUG Added Clinic`) or same address ±house number (`10008`/`10012 Tanner Mill Drive`) —
  indistinguishable from the perturbations applied to *true* copies (`6138`/`6136`). This is a partly irreducible ceiling.

## Validation design (Phase 8)

1. **Split by S1 entity** (hash of `entity_id`), 10 folds. Every pool record has ≤1 true S1, so positives never cross folds.
   Distractors may appear as negatives in several folds; that is harmless because the model uses no IDs or raw tokens.
2. **Full-partition validation for the final pipeline:** run production blocking for the entire train set once; hold out
   one fold (10% of S1, ≈220k) and evaluate the full decision policy (threshold + argmax over *all* S1 + duplicate
   propagation) on it. This is the only way to measure the 1-to-many constraint and true candidate counts.
3. Check split balance: country, match-count distribution, singleton rate (5.58% everywhere), S2/S3 mix, hard segments —
   hash splitting reproduced these within noise on the dev split.
4. Duplicate businesses across folds: S1 names are shared by ~50% of entities (franchises/generic names), so folds contain
   same-named *different* entities — realistic and not leakage. Do not add name-token features (memorization).
5. Any learned lexicon (indic dictionary, distractor tokens, abbreviation map) must be learned **out-of-fold** when used to
   evaluate; current `indic_dict.pkl` was learned on all train pairs (slight optimism for indic segments on train folds).
6. Report per fold: macro-F0.5, micro P/R, singleton accuracy, blocking recall, candidates/S1, reduction ratio, and
   F0.5 by country, source, and hard segment (empty address, brand, domain, indic).
7. Test-shift guard: evaluate the threshold under negative up-weighting (×1.25 distractors) to mimic the higher test
   pool/S1 ratio; pick the threshold on the plateau that is best under both.
8. France: no labels → sanity-only: predicted matches per S1 (train: mean 3.46, 5.6% singletons), score distributions
   vs US/India, and a manual review of ~50 predictions.
