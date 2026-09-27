# 13 — Normalization decision (FROZEN contract v3)

Evidence: 04 (step-wise audit on 150k true pairs vs 150k hard negatives), 12 (France), 07 (blocking).
Ablation levels: **N0** raw lowercase; **N1** conservative (NFKC, zero-width/decorators, anyascii, alnum);
**N2 = v3 below** (norm2 + targeted fixes). Measured on train (exact-equality on positives / on hard negatives):

| name level | eq pos | eq neg | token-set sim pos / neg |
|---|---|---|---|
| N0 | 0.159 | 0.039 | 84.8 / 52.8 |
| N1 | 0.259 | 0.059 | 91.3 / 64.7 |
| N2 full name (`n`) | 0.334 | 0.116 | 93.4 / 69.4 |
| N2 core2 (legal+honorific+generic drop, sorted) | 0.647 | 0.383 | 95.5 / 69.7 |

| address level | token Jaccard pos / neg | token-set pos / neg |
|---|---|---|
| N0 | 0.549 / 0.089 | 89.2 / 46.5 |
| N1 | 0.636 / 0.116 | 91.7 / 49.2 |
| N2 | 0.730 / 0.130 | 94.5 / 49.9 |

Because suffix/honorific/generic removal raises negative equality almost as much as positive equality, **core2 is never
used alone as a decision**; both `n` and `core2` are kept, and equality of each is a separate feature.

## Contract
| representation | transformation | preserves | removes | collision risk | blocking | features | exact-key safe? |
|---|---|---|---|---|---|---|---|
| raw name/address | none | everything | — | none | no | flags (junk, domain, indic) | yes |
| `n` | NFKC, zero-width & decorator strip, `#`/`@` strip, domain→label, learned indic dictionary, anyascii, lowercase alnum, leet fix inside words, **`s a r l`→`sarl`** | word order, legal forms, digits | punctuation, case, script | low | — | `n_eq`, ratios | yes |
| `core2` | `n` minus legal suffixes/aliases, honorifics (shri, mr, smt, dr, m s), trailing `com`/`(id n)`/phone, generic `services/service/enterprises/enterprise` | distinctive words | legal form, generic words | **medium** (e.g. `X LLC` vs `X Inc`) | reverse/forward TF-IDF, core2 blocks ≤200 | set/sort ratios, idf overlap | **no** (feature only) |
| `ns` | `core2` without spaces | concatenated handles/domains | spaces | medium | char-3gram retrieval | JW / ratio / partial | no |
| `a` | components kept in order; anyascii, learned indic component dictionary, alnum, abbreviations → **short** canonical form (+ French `av→ave`, `bd→blvd`, `imp→impasse`), leading zeros stripped from numbers, house-number prefixes (`h no`, `door no`, `plot no`…) dropped, placeholder components `null`/`n a` dropped | all numbers, component order | case, punctuation, long/short variants | low | TF-IDF (tokens) | set/sort/partial, Jaccard, idf | yes |
| `nums` | all digit tokens of `a` (leading zeros stripped) | every number | letters | — | — | shared, first-equal, **min/first edit distance** | — |
| `pnum1`, `pstreet` | first number; `<num> <word>` | | | | core-key union (street≤100 optional) | sibling consensus | — |
| `state` | learned component→state map (US 45 codes, India 16 states); France none | | | | **never** used to filter | 3-valued agree feature | — |

Floor/Florida: canonicalize `floor → flr` (was `fl`, which collides with the Florida code `fl` during state resolution).

## Never
- Filter or block by state/country-specific keys; expand abbreviations to long forms; drop or round numbers;
  treat numbers differing by one edit as a hard conflict; use `core2` equality as a decision; learn dictionaries on
  validation folds when measuring them.

## Data repair
Only deterministic formatting repairs listed above; no identity guessing, no external lists beyond generic language
abbreviations (French street types, legal forms) that are visible in the data itself.

Status: **frozen** for production (v3). The dev experiments (09, 14) were run with norm2; the v3 deltas are small
formatting fixes, and the full-partition validation on AWS measures the final system under v3.
