# 08 — Feature analysis

Data: `work/dev_feats.parquet` — 2,672,867 candidate pairs for the 43,888 eval S1 (61/S1), built by
`scripts/40_dev_cands.py` (forward na@30 ∪ forward c3@10 ∪ core2 blocks ≤200 ∪ reverse na@20/@10, blocking recall 98.9%)
and `scripts/42_features.py`. Positive rate 5.6%. Negatives are **hard** (all retrieved as similar).
Univariate numbers: `analysis/out/50_univariate.txt`; model gains: `analysis/out/43_models.txt`, `48_e6.json`.

## Univariate separation among hard candidates (AUC / average precision)

| group | feature | AUC | AP |
|---|---|---|---|
| retrieval/competition | `rscore` (reverse TF-IDF cosine of name+address) | 0.984 | **0.869** |
| | `r_margin_best` (this S1's score − pool record's best S1 score) | 0.987 | 0.731 |
| | `rrank` (rank of this S1 in the pool record's reverse list) | 0.981 | 0.720 |
| | `s1_rank_rscore` (rank within the S1's candidate list) | 0.979 | 0.754 |
| address | `a_idf_cov` (rarity-weighted share of S1 address tokens present) | 0.939 | 0.645 |
| | `a_jac` / `a_tsr` / `a_tsort` / `a_pr` | 0.92–0.94 | 0.53–0.63 |
| | `a_max_idf_missing` (rarest S1 address token absent) | 0.860 | 0.562 |
| | `num_shared`, `num_first_eq` | 0.83–0.85 | 0.26–0.28 |
| | `state_agree` (−1/0/1) | 0.849 | 0.167 |
| name | `n_max_idf_shared` (rarest shared name token) | 0.739 | 0.179 |
| | `n_tsr`, `ns_jw`, `n_pr`, `ns_pr`, `n_ratio` | 0.71–0.73 | 0.09–0.10 |
| | `n_eq`, `core_eq`, `core_sorted_eq` | 0.61–0.62 | 0.08 |
| | `name_freq_s1` (how many S1 share this core name; lower = match) | 0.633 | 0.082 |
| flags | `brand2`, `dom2`, `indic2`, `junk2`, `a2_empty` | ≈0.50–0.57 | — |

Name similarity is a weak discriminator **among retrieved candidates** because candidates are already name-similar and
names are shared by many entities. Address specificity and competition (is this the pool record's *best* S1?) decide.

## Multivariate value (LightGBM, macro-F0.5 on valid split; see 09)

| features | AP | macro-F0.5 |
|---|---|---|
| name only | 0.744 | 0.630 |
| address only | 0.832 | 0.823 |
| name + address | 0.9925 | 0.9586 |
| + competition/retrieval (all) | 0.9943 | 0.9638 |
| + house-number closeness (`num_min_ed`, `num_first_ed`) | — | 0.9643 valid / **0.9642 holdout** (+0.0017 vs 0.9625) |

* Name and address are **complementary**: neither alone exceeds 0.83, together 0.959. Cross-field interactions (strong name
  + weak address, empty address + generic name, brand/domain + strong address) are learned by the trees.
* Top LightGBM gain: `r_margin_best` ≫ `rrank` > `num_conflict` > `num_first_eq` > `s1_rank_sum` > `ns_jw` > `n_len_ratio`
  > `n_eq` > `m_gap12` > `a_tsr` > `state_agree` > `name_freq_s1`.
* Binary `num_conflict` is over-trusted: ~1,200 holdout true matches with near-identical house numbers were rejected.
  Number edit distance / truncation features fix part of it (+0.7pp recall at equal precision).

## Features to keep for production

NAME: `n_eq`, `core_eq`, `core_sorted_eq`, token-set/sort/ratio/partial on core names, space-free JW/ratio/partial,
token Jaccard, idf coverage, rarest shared/missing token idf, token counts, length ratio, **S1 name frequency**.
ADDRESS: token-set/sort/partial, Jaccard, idf coverage, rarest missing/shared idf, number shared/first-equal/min edit
distance/first edit distance, `num_conflict`, state agree (3-valued), address-empty/number-missing indicators.
CROSS/CONTEXT: reverse cosine, reverse rank, pool record's best & 2nd-best S1 score, margin to best, per-S1 candidate
rank/count, per-S1 count of core-equal candidates, source (S2/S3), country flag, noise flags (domain/indic/brand/junk).

Candidates to add (not yet tested): distractor-token score (out-of-fold `token_match_rates`), pool-side duplicate
cluster size (identical normalized twins), address idf computed on the pool as well as S1, reverse char-3gram score.
Do **not** use: IDs, row order, raw name tokens (memorization risk).

TF-IDF cosine and char n-gram similarity are already present implicitly via `rscore` (word TF-IDF) and `ns_*`;
a separate char-3gram cosine is a cheap addition once the c3 index exists for blocking.
