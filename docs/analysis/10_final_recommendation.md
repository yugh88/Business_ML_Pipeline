# 10 — Final recommendation (session 2, 2026-09-25)

All numbers are measured on train (no test labels used). Details in 00–09 and `manual_checks.md`.

## 1. Dataset facts
* Train S1 2.21M / S2 5.03M / S3 5.29M (US 60%, India 40%); test S1 1.73M / S2 4.89M / S3 5.08M incl. **France** (15%, no labels).
* 7.64M true pairs; every pool record matches **≤1 S1**; 26% of pool records are unmatched distractors; 5.58% of S1 are
  singletons; mean 3.46 matches/S1 (0–11), identical in US and India (synthetic, regular generator).
* S1 is clean; S2/S3 carry all noise. ~50% of S1 share their core name with another S1 → names are not identifying.
* No postal codes. 3–5% empty pool addresses. Test has 5.75 pool records per S1 vs 4.68 in train (more distractors likely).

## 2. Important noise patterns
Domains/handles (4%), indic scripts (India 11–23% of pairs), brand-replaced names (1–6%), legal-suffix swaps, injected
`Services/Enterprises`, junk decorators, typos/leet, **house-number perturbation**, S3 address re-ordering and
city↔county/state-name substitution, empty addresses, and a generator of **near-twin distractors** (±1 letter / ±house number).

## 3. Safe normalization
NFKC, zero-width & decorator removal, anyascii, learned indic dictionary, lowercase/alnum, abbreviation canonicalization
to the short form, leading-zero strip, placeholder removal (`null`, `n/a`), generic-word removal for the core name,
space-free name variant. Extend for France: `av→ave`, `bd→blvd`, `r.`→`r`, `st-`/`saint`→`st`, `sàrl→sarl`
(norm2.CANON maps avenue→ave and boulevard→blvd but leaves French `av`/`bd` untouched).

## 4. Dangerous normalization
Legal-suffix removal as an equality key (+24.6pp equality on positives but +23.9pp on hard negatives); domain stripping as
equality; any state/country-based filtering (France has no state map; state disagrees in 0.1% of true pairs but is
unknown in 3–9%); expanding abbreviations; treating differing house numbers as a hard conflict.

## 5. Best blocking strategy
Union of: **reverse** TF-IDF (pool record → top-k S1, name+address tokens, df cap 30k–100k) + reverse char-3gram on the
space-free name + forward name+address TF-IDF (S1 → top-10 pool) + exact sorted-core-name blocks ≤200. Implemented
memory-safely in `scripts/retrieval2.py` (single process, bounded chunk products, work budget) + `scripts/memguard.py`.

## 6. Blocking recall (eval sample, 43.9k S1 vs full pool)
| config | cand/S1 (upper bound) | recall |
|---|---|---|
| exact normalized name only | 58 | 0.575 |
| A: rev na30k@10 + rev c3@5 + fwd na@10 + core2≤50 | 84 | 0.9840 |
| C: A with cap 100k | 84 | 0.9863 |
| **D: rev na100k@20 + rev c3@10 + fwd na@10 + core2≤200** | 174 | **0.9910** (US 0.993 / India 0.988) |
Recall ceiling losses concentrate on empty-address pool records (13% missed) — mostly unmatchable anyway.

## 7. Best features
Reverse cosine / reverse rank / margin to the pool record's best S1; address rarity coverage (idf), address token
Jaccard/set ratio, missing-rarest-token idf; number agreement **and number edit distance**; state agree (3-valued);
name set/sort ratios, space-free JW/partial ratio (domains), rarest shared name token, S1 name frequency; flags.

## 8. Recommended model
**LightGBM binary classifier** on ~60 engineered pair features + a decision layer. Holdout macro-F0.5 **0.9642**
(micro P 0.987, R 0.932, singleton accuracy 0.95) vs perfect-matcher ceiling 0.996 on the same candidates.
XGBoost ties (no gain), MLP −0.011, logistic regression −0.035.

## 9. Is an FNN justified? **No.** Same features, worse by 1.1 points, less calibrated on singletons (0.90 vs 0.95).

## 10. Are transformers/embeddings justified? **Not now.** Residual errors are (a) empty-address ambiguity, (b) generator
near-twins that differ only in a house number or one letter, (c) wrong-S1 competition — none is a semantic-similarity
problem a text encoder would solve, and indic transliteration is already solved by the learned dictionary (0.05% zero
overlap). Revisit only if the full-partition validation shows a large residual in brand/domain/typo segments; then a
small char-level cross-encoder re-ranker on the top-5 would be the candidate (≤8B params, MIT/Apache), GPU-trained.

## 11. Validation design
S1-grouped hash folds; one 10% fold held out end-to-end with production blocking over the full train set so the
1-to-many constraint and true candidate counts are measured; out-of-fold learned lexicons; per-segment reporting;
threshold stress test with up-weighted distractors; France sanity checks. (Details: 09 §Validation.)

## 12. Thresholding
Single global probability threshold on the flat plateau 0.65–0.80 (optimum 0.72 on valid); pick ~0.75 for precision
margin against test distractor shift. Segment-specific thresholds gave no gain (0.9619 vs 0.9625) — don't use them.

## 13. Singleton handling
No special classifier needed: empty prediction whenever no candidate passes the threshold after the decision layer
(95% singleton accuracy on holdout). Add: abstain for empty-address pool records whose name is shared by ≥2 S1 unless the
name is rare; abstain when the pool record's top-2 S1 scores are within a small margin.

## 14. Multi-match handling
Predict **all** candidates that pass (mean 3.46 true matches/S1). Then (i) enforce 1-to-many: each pool record goes only to
its highest-probability S1 over the full S1 set; (ii) optionally propagate to identical normalized twins in the pool
**only if the legal suffix also agrees** (186 twins with differing suffixes are deliberate unmatched distractors).

## 15. Compute requirements (measured rates, single process: ~4e7 sparse multiply-adds/s)
| stage | train (full) | test | peak memory | class |
|---|---|---|---|---|
| normalization | done (parquet exists) | done | — | — |
| rev na cap 30k (10M pool queries) | ~2.4 h | ~2.4 h | 3–5 GB | LOCAL CPU |
| rev na cap 100k | ~7 h | ~7 h | 3–5 GB | LOCAL CPU (slow) / AWS BENEFICIAL |
| rev c3 cap 5k | ~35 min | ~35 min | 3–4 GB | LOCAL CPU |
| fwd na cap 20k (S1 queries) | ~15 min | ~12 min | 3 GB | LOCAL CPU |
| core2 blocks (DuckDB) | 1–2 min | 1–2 min | ≤5 GB | LOCAL CPU |
| features (~100–170M pairs) | ~40 min in 1M-pair chunks | ~40 min | ≤4 GB per chunk (polars peaked 9 GB at 2.7M pairs unchunked) | LOCAL MEMORY-INTENSIVE → chunk |
| LightGBM (10–15M training rows) | 10–20 min | inference ~10 min | 3–6 GB | LOCAL CPU |
| GPU / transformers | — | — | — | NOT WORTH RUNNING now |

## 16. Local vs AWS
**Run locally**, sequentially, with `memguard` and chunking: ≈8–10 h wall time for full train + test with cap 30k.
AWS is optional and only buys wall-time: one `r6i.4xlarge` / `m7i.4xlarge` (16 vCPU, 64–128 GB) running 8–12 retrieval
processes in parallel would do the cap-100k blocking for train+test in ~1.5–2 h; upload ≈1.4 GB (normalized parquet only);
≈3–5 instance-hours ≈ **$3–8** on-demand (spot less). No GPU instance is justified. Do not launch until the local
full-partition validation shows cap 100k is worth +0.2–0.5pp.

Environment note: during this session macOS `fileproviderd` (100% CPU) and Spotlight indexing competed for CPU (load 19)
— likely reacting to the 3 GB of new files under `~/Downloads/student_resource/work`. Excluding `work/` from Spotlight
(System Settings → Siri & Spotlight → Spotlight Privacy) and from any cloud sync is recommended before long runs.

## 17. Manual checks for you
See `manual_checks.md`: (1) near-twin distractors vs perturbed true copies (threshold/ceiling), (2) wrong-S1 cases that
argmax should fix, (3) 21 label-conflict groups, (4) unmatched twins with different legal suffixes, (5) brand+corrupt
true pairs, (6) generic-name blocking misses, (7) Devanagari shop-word distractors, (8) France formatting.

## 18. Remaining uncertainties
* The argmax/1-to-many gain and real candidate counts are unmeasured (need full-partition run).
* Test distractor share is unknown (5.75 vs 4.68 pool/S1); precision may drop on test.
* France: zero labels; unseen address grammar; retrieval and features are country-agnostic but untested there.
* `indic_dict.pkl` was learned on all train pairs → train-fold indic results are slightly optimistic; test coverage is
  lower (96.4% vs 97.7% tokens).
* The dev candidate set approximates config D (forward top-30 + reverse re-query), not the exact production union.
* One job (`40_dev_cands.py`) hit an 8 GB RSS / 43 GB footprint transient with 5.5 GB swap in use afterwards; cause not
  isolated (suspects: DuckDB block join or polars joins on strings). All later jobs ran under `memguard`; add it to every
  heavy script.

## 19. Exact next implementation steps
1. Patch normalization: add `services/service/enterprises` to the core-name drop list, French abbreviations (§3),
   placeholder drop (`n a`), and fix `floor→fl` vs Florida; regenerate `*_norm.parquet` in chunks (single process).
2. Build `src/block.py` (from `retrieval2.py`) producing config A (cap 30k) for **full train**: reverse na + reverse c3 for all
   10.3M pool records (per country, per source, checkpoint per chunk to `work/cands_train/`), forward na for all S1, core2 blocks.
3. Build `src/features.py` from `42_features.py` + E6 number features, processing 1M-pair chunks to parquet.
4. Train LightGBM on folds 1–9 (subsample negatives if >15M rows), hold out fold 0 end-to-end; evaluate threshold +
   argmax + abstention rules; decide on config D / cap 100k and twin propagation from the measured gains.
5. Run the same pipeline on test (with France), write `candidate_pairs.tsv` (the exact scored set) and
   `matching_results.tsv`, validate with `utils/validate_submission.py`, sanity-check France.
