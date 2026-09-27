# 05 — Ground-truth analysis

Sources: `work/02_gt.txt` (previous session, verified), `work/true_pair_sims.parquet` (this session; similarity profile of
all 7,638,365 true pairs, `scripts/21_pair_sims.py`), `analysis/out/23_label_collisions.txt`, `work/label_collision_groups.parquet`.

## Structure

* 2,206,821 S1; every S1 has exactly one GT row; 7,638,365 (S1, pool) pairs; S2 3,693,619 / S3 3,944,746.
* **Every pool record is linked to at most one S1** → the task is a *1-to-many assignment*: for each pool record, pick
  the single best S1 or none. This constraint is free precision.
* No cross-country pairs.
* 73.4% of S2 and 74.6% of S3 records are matched; **2,681,854 pool records (26%) are unmatched distractors**
  (US 1.61M, India 1.07M). Test has 5.75 pool records per S1 vs 4.68 in train → likely a larger distractor share.

## Match-count distribution (per S1)

| matches | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 count | 123,247 | 119,157 | 375,212 | 530,841 | 484,115 | 321,957 | 164,868 | 63,968 | 18,680 | 4,205 | 534 | 37 |
| share | **5.58%** | 5.40% | 17.0% | 24.1% | 21.9% | 14.6% | 7.5% | 2.9% | 0.8% | 0.2% | | |

Mean 3.46 matches per S1 (identical in US 3.459 and India 3.465). Per source: S2 0–5 matches (13% have none),
S3 0–6 (12% none). Categories: both S2+S3 80.5%, S2-only 6.5%, S3-only 7.5%, none 5.6% — identical US vs India.

The distribution is so regular (same singleton rate and mean in both countries) that it is almost certainly generated:
each S1 entity is copied into S2 and S3 0–6 times with noise. **Multiple matches from the same source are noisy copies
of the same entity**, which is why pool near-duplicates exist.

## Class imbalance

* Per S1: 94.4% have ≥1 match; singletons 5.6%. The macro-F0.5 gives singletons 1.0 for an empty prediction.
* Pairwise within-country positive rate ≈ 6.5e-7 (7.6M / 1.18e13). After blocking (07, config D) ≈ 3.5 positives per
  ≤174 candidates per S1 → ~2% positive rate among candidates; after a first-stage trim to ~20–30, ~12–17%.

## Label consistency (suspicious labels)

* Pool groups with identical normalized (core name, address): 522k groups / 1.13M records. Only **186 groups** mix a
  matched record with an unmatched identical twin (US 137, India 49) and **21 groups** link identical records to
  **different** S1 entities (US 4, India 17). → label noise is ~1e-5 of pairs; GT can be trusted.
  The 21 multi-S1 groups are generic India names with minimal addresses (`guntur center | fl , guntur , ap`,
  `mumbai developers | 2 , mumbai , mumbai city , mh`) — genuinely indistinguishable; see manual_checks.md.
* True pairs with **both** weak name (token-set <50) and weak/empty address: US 0.007%, India 0.020% (~940 pairs).
  These are either extremely noisy copies or label errors; either way unlearnable (see manual_checks.md).

## Country behaviour

US and India have the same match-count distribution. Differences are in the *noise*: India has indic-script names
(23% of S2 India pairs, 13% of S3), longer free-form addresses without the `<num> <street>` grammar (street key recall
22% vs 58% US), more generic company names (`X Private Limited`), higher name sharing among S1 (54% vs 47%).

## Source-pair behaviour (true pairs)

| | India S2 | India S3 | US S2 | US S3 |
|---|---|---|---|---|
| identical normalized name | 39.3% | 32.5% | 31.4% | 31.4% |
| identical core (legal-suffix stripped) | 66.6% | 58.5% | 56.1% | 52.5% |
| identical normalized address | 18.4% | 8.2% | 29.4% | 4.4% |
| empty pool address | 3.8% | 4.0% | 4.9% | 4.6% |
| state agrees | 94.5% | 91.6% | 95.0% | 89.7% |
| ≥1 shared address number | 81.3% | 79.3% | 78.1% | 81.0% |
| brand-token flag in pool name | 1.0% | 5.1% | 1.4% | 6.1% |

S3 is harder: it re-orders and trims addresses, substitutes city/county variants, and carries 4–5× more
brand-replaced names. S2 upper-cases addresses and abbreviates (`DR`, `RD`).
