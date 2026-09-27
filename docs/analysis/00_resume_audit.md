# 00 — Resume audit (session 2, 2026-09-25)

Machine: macOS, 18 GB RAM, 11 cores. Python env: `.venv` (polars 1.44, duckdb 1.5, pyarrow, rapidfuzz, anyascii,
scipy, sklearn, lightgbm 4.7, xgboost 3.4; **no catboost, no torch**).

## What the previous session completed (all artifacts present and schema-valid)

| Script | Status | Output | Trust |
|---|---|---|---|
| 00_to_parquet.py | ran | `work/{train,test}_s{1,2,3}.parquet`, `train_gt.parquet` (row counts match inventory) | trusted (read with `quote_char=None`, no schema inference — correct for this TSV) |
| 01_inventory.py | ran | `work/01_inventory.txt` | trusted (full-file stats; script sampling only for char-script counts) |
| 02_gt.py | ran | `work/02_gt.txt`, `train_pairs.parquet` (7,638,365 pairs), `train_per_s1.parquet` | trusted; verified row counts |
| 03_examples.py | ad-hoc viewer | stdout only | n/a |
| 04_noise.py | ran | `work/04_noise.txt` (rates on 208k true pairs from 60k S1), `04_fulldiff.json` (4,417 low-name-similarity pairs) | trusted as descriptive rates; tags are heuristic |
| 05_brand.py / 05b / 05c | ran | `brand_candidates.json`; `brand.py` syllable detector | **stdout lost** — coverage/rates not recorded; re-measured in 04_noise_patterns.md |
| 06_indic.py | ran | `indic_dict_trainsplit.pkl` (1,318 tokens) | **held-out evaluation numbers lost** (stdout) |
| 06b_indic_dict.py | ran | `indic_dict.pkl` (name: 1,347 tokens; addr: 15 components) | coverage numbers lost |
| 07_distractors.py | ran | stdout only | **results lost** |
| 07b_vocab.py | ran | `token_match_rates.parquet` (333k country×token rows) | artifact OK, printed conclusions lost |
| 08_abbrev.py | ran | `addr_maps.pkl` (abbr US 36 / India 7; last-component maps) | needs audit (learned typo→full maps) |
| 09_state_map.py | ran | `state_map.pkl` (US 2,045 / India 1,588 component→state keys) | needs audit |
| 10_build_norm.py | ran | `work/*_s{1,2,3}_norm.parquet` for train+test (cols: n, ncore, f_domain, f_indic, f_brand, f_junk, a, nums, state) | row counts match raw; spot-check OK |
| 11_blocking.py | **CRASHED — no output** (`work/cands_*.parquet` absent) | — | — |

## Where the previous session stopped / likely cause of the freeze

`11_blocking.py` + `retrieval.py` was the last step (mtime 01:44, same minute as the last `_norm` file).
It is the almost-certain cause of the freeze:

* loads **all** train S2+S3 norm (10.3M rows) plus derived concat column `na` in RAM;
* builds 5 TF-IDF views including `char4` with `df_cap=20000` via a Python `map_elements` lambda over 6M+ rows;
* `topk()` forks **10 worker processes** (`multiprocessing.Pool(10)`), each computing `Qc @ P` for 2,000 queries at a time.
  With df caps of 5,000–20,000 and ~5–25 tokens per query, one chunk's result can reach 10^7–10^8 nonzeros
  (~0.1–1.2 GB) *per worker*, ×10 workers, on top of the copy-on-write pool matrix → swap death.
* `10_build_norm.py` also used `Pool(10)` with full row lists in RAM, but it finished.

**Policy for this session:** single-process only; per-country; query chunks sized so a chunk result stays < 200 MB;
df caps chosen from measured df distributions; blocking recall measured on true-pair tables first (cheap), and top-k
retrieval only on a fixed S1 evaluation sample.

## Conclusions that can be trusted (from saved artifacts)

* Train: S1 2,206,821 / S2 5,034,616 / S3 5,285,603; test: 1,732,544 / 4,887,273 / 5,082,316.
* GT: one row per S1; 7,638,365 pairs; **no S2/S3 record matches more than one S1** (a hard 1-to-many constraint);
  no cross-country pairs; 5.58% of S1 are singletons (same in US and India).
* No ID / row-order leakage (corr ≈ 0; GT order ≠ S1 order).
* S1 is clean (ASCII-only, Title Case, no empties). S2/S3 carry all the noise (Indic scripts, accents, casing, junk
  prefixes, domains, empty addresses ~3.4%).
* Test adds **France** (S1 259k; S2 703k; S3 732k) — accents in S1 itself, French address grammar (RUE/AV/BD), SARL/SAS.

## Conclusions that need verification (done in later docs)

* Whether `norm2` transformations actually raise true-pair agreement without raising collisions (→ 04_noise_patterns.md).
* Learned typo→full abbreviation map and state map could over-merge (→ 04).
* Indic dictionary accuracy (held-out numbers lost) (→ 04).
* Brand-token detector precision (→ 04).

## Missing analysis (this session)

Blocking recall ceiling (highest priority), feature separability, model comparison, validation design, manual checks,
compute plan, final recommendation. `analysis/` was empty although `norm2.py` refers to `analysis/04_noise_patterns.md`.

---

## Session 2 status (end) — what now exists

New scripts (all single-process, memory-bounded): `20_quality.py`, `21_pair_sims.py`, `30_block_keys.py`, `retrieval2.py`,
`31_tfidf_eval.py`, `32_recall.py`, `33_rev_eval.py`, `34_blocking_budget.py`, `34b_blocking_budget.py`, `40_dev_cands.py`,
`41_norm_audit.py`, `42_features.py`, `43_models.py`, `46_errors.py`, `48_e6_numfeat.py`, `49_manual_cases.py`, `memguard.py`.

New artifacts (do not recompute): `work/true_pair_sims.parquet`, `work/true_pair_keys.parquet`, `work/label_collision_groups.parquet`,
`work/eval_s1_ids.parquet` (eval sample = train S1 with hash(entity_id,3)%50==0), `work/cand_eval/*.parquet` (forward/reverse
retrieval results), `work/blocking_misses_eval.parquet`, `work/dev_cands.parquet` (2.67M dev candidates, labels, reverse
competition stats), `work/dev_feats.parquet` (features), `work/tok_df_{name,addr}.parquet`, `work/name_freq_s1.parquet`,
`work/dev_lgb_all*.txt` (LightGBM), `work/dev_holdout_{fp,fn}.parquet`; summaries in `analysis/out/`.

Reports: 00, 01, 03, 04, 05, 06, 07, 08, 09, 10, manual_checks.md — all complete.

Next session starts at `10_final_recommendation.md` §19 step 1. Wrap every heavy script with `import memguard`.
