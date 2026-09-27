# 14 — Production freeze (configuration fixed before any test processing)

Frozen now (code in `aws/`, config `aws/config.yaml`):

| component | frozen choice | evidence |
|---|---|---|
| normalization | contract v3 (`aws/scripts/normlib.py`) | 13 |
| representations | n, ncore, core2, ns, a, nums, state, pnum1, pstreet + flags | 13 |
| blocking | rev na (cap 30k, k=20) ∪ rev c3 (cap 5k, k=10) ∪ fwd na (cap 20k, k=10) ∪ sorted-core2 blocks ≤200 | 07 (config F: 98.97% recall on eval sample) |
| features | 61 pairwise (name/address/number/legal/retrieval/competition, no country one-hot) | 08, 09, 12 |
| stage 1 | LightGBM (lr 0.05, 63 leaves, min leaf 100, ff/bf 0.8, early stop 50), cross-fitted on folds 1 / 2 | 09 |
| cascade | stage-2 (final model) scores only pairs with stage-1 p1 ≥ 1e-4 = `candidate_pairs.tsv` | dev: keeps 99.992% of positives, ~8 pairs/S1 |
| stage 2 | stage-1 p + collective sibling-consensus features (global pool competition over all p1), cross-fitted folds {3,4}/{5,6} | 12 F2, dev: 0.9625→0.9707 |
| stage 3 / bigger trees | rejected (no gain: 0.9706 / 0.9708) | 12 |
| decision | selected **automatically on train fold 7 only** among A (threshold), B (best-S1), D (ambiguity abstention), E (margin), F (twins); conservative tie-break (simplest policy, higher threshold within 0.0003); written to `validation/decision.json` and consumed unchanged by `predict_test` | evaluate.py |
| threshold grid | 0.30–0.95 step 0.01 | |
| test | never used for any selection; no leaderboard feedback loop | |

Filled in after the AWS validation run: chosen policy/threshold and holdout scores (see 15/16).
