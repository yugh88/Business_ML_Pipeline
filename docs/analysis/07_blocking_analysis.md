# 07 — Blocking analysis (recall ceiling)

Scripts: `scripts/30_block_keys.py` (exact keys, all 7.6M true pairs), `scripts/retrieval2.py` (memory-safe TF-IDF),
`scripts/31_tfidf_eval.py` (forward S1→pool), `scripts/33_rev_eval.py` (reverse pool→S1), `scripts/32_recall.py`,
`scripts/34_blocking_budget.py`, `34b_blocking_budget.py`. Raw numbers: `analysis/out/30_*.json`, `32_*.json`, `34*.json`.
Candidates: `work/cand_eval/*.parquet`. Misses: `work/blocking_misses_eval.parquet`.

**Evaluation set:** S1 with `hash(entity_id,3)%50==0` → 43,888 S1 (US 26,403 / India 17,485), 152,136 true pairs,
retrieved against the **full** country pool (US 6.19M, India 4.13M), so candidate ranks are realistic.

## 1. Exact keys (full train; recall over all 7.64M pairs; candidates = Σ block products, no materialization)

| key | recall | US | India | cand/S1 | max block | note |
|---|---|---|---|---|---|---|
| normalized core name (`ncore`) | 0.575 | 0.542 | 0.624 | 58.5 | 1,586 | generic names → huge blocks |
| sorted core tokens | 0.598 | | | 59.7 | | word-order swaps |
| sorted core2 (honorific/.com/ID stripped) | 0.622 | 0.579 | 0.688 | 61.6 | 1,599 | |
| 8-char no-space name prefix | 0.823 | | | 2,509 | 65,376 | catches domains/handles, far too many cands |
| full normalized address | 0.152 | | | 0.6 | 23 | S3 addresses rarely identical |
| street (`<num> <word>`) + state | 0.439 | 0.584 | 0.223 | 37.5 | 8,624 | India addresses don't follow this grammar |
| first number + state | 0.678 | | | 1,581 | 36,057 | too broad |
| first name token + state | 0.767 | | | 614 | 21,936 | too broad |

No postal/PIN codes exist (India 6-digit PIN in 0.0% of addresses; the US 5-digit hits (~10%) are house numbers).
→ **Postal blocking is impossible; exact keys alone cap at ~60% recall.**

## 2. TF-IDF top-k retrieval (eval sample)

Views: `name` (core2 word tokens), `addr` (address tokens), `na` (name tokens prefixed + address tokens, one vector),
`c3` (char 3-grams of the space-free core2 name — handles `@henrymanagement`, `hghpackaging.com`).
`cap` = tokens with document frequency > cap dropped from both sides.

Forward (S1 → pool, cap 20k), recall@k:

| view | @10 | @20 | @50 | @100 |
|---|---|---|---|---|
| name | 0.344 | 0.396 | 0.457 | 0.502 |
| addr | 0.780 | 0.820 | 0.852 | 0.869 |
| na | 0.892 | 0.920 | 0.943 | 0.955 |
| c3 | 0.409 | 0.466 | 0.526 | 0.573 |
| union (cand/S1 29 / 61 / 157 / 315) | 0.934 | 0.951 | 0.965 | 0.973 |

**Reverse (pool record → top-k S1)** is the natural direction here because every pool record matches ≤1 S1 and S1 is
deduplicated and 4.7× smaller. Cost per S1 ≈ 4.68·k (upper bound). Recall@k:

| view (cap) | @1 | @5 | @10 | @20 |
|---|---|---|---|---|
| rev na (5k) | 0.897 | 0.936 | 0.946 | 0.955 |
| rev na (30k) | — | 0.970 | 0.977 | 0.982 |
| rev na (100k) | 0.949 | 0.974 | 0.980 | 0.985 |
| rev addr (5k) | 0.730 | 0.834 | 0.856 | 0.871 |
| rev c3 (5k) | 0.364 | 0.524 | 0.572 | 0.615 |

The df cap is the key recall lever: raising 5k→30k gives +3.1pp at k=10 (common-but-informative tokens like
"technologies", city names were being discarded). Work grows ~5× (30k) and ~15× (100k).

## 3. Union configurations (recall vs cost; cost = upper bound cand/S1)

| config | cost/S1 | recall | US | India | addr-empty | weak-name (tsr<50) | weak-addr |
|---|---|---|---|---|---|---|---|
| rev na30k@10 alone | 47 | 0.9765 | 0.985 | 0.964 | 0.713 | 0.929 | 0.892 |
| **A**: rev na30k@10 + rev c3@5 + fwd na@10 + core2-block≤50 | 84 | 0.9840 | 0.990 | 0.976 | 0.805 | 0.941 | 0.924 |
| **C**: same with rev na100k@10 | 84 | 0.9863 | 0.990 | 0.981 | 0.809 | 0.950 | 0.935 |
| **D**: rev na100k@20 + rev c3@10 + fwd na@10 + core2≤200 | 174 | **0.9910** | 0.993 | 0.988 | 0.869 | 0.958 | 0.958 |
| E: D + rev na30k@20 + rev addr@10 + fwd@20 + street≤100 | 328 | 0.9917 | 0.994 | 0.988 | 0.883 | 0.960 | 0.959 |
| F: D with na30k instead of 100k | 174 | 0.9897 | 0.993 | 0.985 | 0.867 | 0.952 | 0.950 |

Leave-one-out on A: dropping rev c3 −0.38pp (it is the only view that catches concatenated domains/handles/typos);
dropping fwd na −0.09pp; dropping core2 block −0.08pp; rev addr adds only +0.02pp once rev na is present.
Exact core2 blocks of unlimited size add +0.6pp but cost +38/S1 (India generic names, e.g. "X Private Limited" ×1,000+).

Recall is nearly flat across S2/S3 (0.989/0.983), match-count (S1 with 1 match: 0.990 vs ≥5 matches: 0.985) and
indic names (0.995) — measured on the "rev na5k/addr/c3@10 + fwd na@10 + core2 + street keys" union. **Weakest segments: empty pool address (13% missed even in D) and India.**

## 4. What is still missed (≈1.4% at config "rev×3@10 + fwd + keys"; 2,157 eval pairs)

* 46% have an **empty pool address** + a generic name with injected tokens (`Pacific Fund` → `Pacific Services Enterprises`,
  `Golden Tattoo` → `Golden Tattoo Inc Enterprises`). `norm2.LEGAL` no longer strips `services/service/enterprises`;
  norm v1 did. These are also intrinsically ambiguous (generic name, no address) — a precision-first matcher
  would mostly reject them even if retrieved.
* Heavy multi-typo names + partial addresses (`Commercial Pctrgchse Ltd`, `Blue Iocetech`).
* Brand-replaced names (`Arcevocalo`) with addresses whose street tokens were dropped.
* Indic state name contradicting the city (`40, GREATER BOMBAY, পশ্চিমবঙ্গ`) — source corruption.

## 5. Recommendation

* **Production blocking = config D** (≈99.1% recall ceiling on train eval; ≤174 cand/S1 upper bound, real count lower
  because views overlap), optionally followed by a cheap first-stage ranker to trim to ~20–30 cand/S1 before the
  expensive feature/model stage; `candidate_pairs.tsv` must then be the trimmed set.
* If compute is tight: config A/C (98.4–98.6% at ≤84/S1).
* Never block on state / PIN / country-specific keys: France has **no** state mapping (100% empty `state`) and no labels.
* Cost (measured, single process, ≤5.5 GB RSS): rev na cap 30k ≈ 2.3e4 multiply-adds per query (≈44M/s) →
  test pool (10M queries) ≈ 1.5 h; cap 100k ≈ 4–6 h. Forward views for 1.7M S1 ≈ 10–20 min each. Safe locally;
  2–3 memory-bounded processes (each ≤4 GB with `max_out`=25M) would cut wall-time proportionally.

## 6. Why the previous session froze (for the record)

`retrieval.py` used `Pool(10)` with 2,000-query chunks and df caps up to 20k on **char-4grams over the whole pool**;
per-chunk products of 10^8 nonzeros × 10 workers. `retrieval2.py` bounds each chunk's product by an estimated
nonzero budget (`max_out`) and aborts if total work exceeds a budget.
