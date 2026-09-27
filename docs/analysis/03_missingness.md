# 03 — Missingness

Source: `analysis/out/20_quality.json` (normalized parquet), `work/01_inventory.txt`, `analysis/out/22_difficulty.txt`.

| split/src | country | empty address | no digits in address | no state resolved |
|---|---|---|---|---|
| train S1 | US / India | 0 / 0 | 0.4% / 12.0% | 0% / 4.9% |
| train S2 | US / India | 3.7% / 2.9% | 11.8% / 11.3% | 3.7% / 7.1% |
| train S3 | US / India | 3.5% / 3.1% | 10.5% / 12.9% | 9.3% / 3.4% |
| test S1 | US / India / France | 0 / 0 / 0 | 0.4% / 12.1% / 0.5% | 0% / 5.0% / **100%** |
| test S2 | US / India / France | 2.9% / 2.3% / 3.1% | | 2.9% / 6.6% / 100% |
| test S3 | US / India / France | 2.8% / 2.5% / 2.9% | | 8.8% / 2.8% / 100% |

Names are never empty. `both_empty` = 0 everywhere.

## Implicit missingness (placeholders)

`NULL`, `<NULL>`, `N/A`, `null` appear as address components (e.g. `##1105 M, NULL, BIRMINGHAM, AL`). `norm2.addr_norm`
drops a component equal to `null` but keeps `n a` (from `N/A`) — see 04 (minor fix: drop `n a` / `na` components).
S1 "addresses" are sometimes only `<number>, <city>, <state>` (e.g. `55 500, SALT LAKE CITY, UT`), i.e. the street is missing
in the reference itself.

## Effect on matching (true pairs)

* Pool address empty in 3.8–4.9% of true pairs. For these, name is the only evidence and names are highly ambiguous
  (01): blocking recall on empty-address pairs is only 80–87% (07), and precision will be poor → these are the prime
  candidates for **abstention** unless the name is rare.
* Missing state in the pool record: 3–9% → state agreement must be a 3-valued feature (agree / disagree / unknown), never
  a filter. France: state unknown for 100% → any state feature is "unknown" for France at test time; a model trained on
  US/India where state is almost always known will see an out-of-distribution value. Mitigation: train with random state
  dropout (set state=unknown for ~20–30% of training pairs), or omit state features.
* Missing numbers: 12% of India S1 addresses have no digits at all → number-agreement features are "unknown", not "disagree".

## Missingness indicators to use as model features

`addr2_empty`, `nums1_empty`, `nums2_empty`, `state_unknown`, `name2_is_domain`, `name2_is_indic`, `name2_brand_flag`.
