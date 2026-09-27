# 04 — Noise patterns & normalization audit

Sources: `work/04_noise.txt` (previous session: tag rates on 208k true pairs), `analysis/out/41_norm_audit.json`
(this session: each normalization step on 150k true pairs vs 150k **hard negatives** = retrieved-but-wrong candidates),
`analysis/out/44_indic_coverage.txt`, `45_shopword_distractors.txt`, `47_number_noise.txt`, learned-map inspection.

The audit measures, for every cumulative step, equality/similarity on **positives** (help) and on **hard negatives**
(collisions). A step is useful only if it raises positives more than negatives.

## Name normalization (cumulative steps; eq = exact equality, sim = token-set ratio)

| step | eq pos | eq neg | Δpos | Δneg | sim pos | sim neg | verdict |
|---|---|---|---|---|---|---|---|
| N0 lowercase + whitespace | 0.159 | 0.039 | | | 84.8 | 52.8 | baseline |
| N1 NFKC, zero-width, junk decorators, alnum | 0.220 | 0.053 | +6.0 | +1.4 | 86.1 | 53.1 | **safe** |
| N2 anyascii transliteration (indic, accents) | 0.259 | 0.059 | +3.9 | +0.7 | 91.3 | 64.7 | **safe** |
| N3 norm2: indic dictionary, domain strip, leet fix | 0.334 | 0.116 | +7.5 | +5.7 | 93.4 | 69.4 | useful, some collisions (domain strip) |
| N4 drop legal suffixes (core) | 0.580 | 0.355 | **+24.6** | **+23.9** | 94.1 | 69.3 | **dangerous as an equality key** |
| N5 sort tokens | 0.602 | 0.361 | +2.2 | +0.6 | | | safe |
| N6 strip honorifics / `.com` / `(ID: n)` / phone tail | 0.626 | 0.377 | +2.4 | +1.6 | | | mildly useful |
| N7 drop `services/service/enterprises` | 0.647 | 0.383 | +2.1 | +0.6 | 95.5 | 69.7 | **useful** (injected noise tokens) |
| N8 remove spaces | 0.661 | 0.384 | +1.4 | +0.1 | | | useful (domains/handles) |

Conclusions:
* **Legal-suffix removal destroys information**: `X LLC` vs `X Inc` vs `X Private Limited` are distinct entities among the
  hard negatives about as often as they are noisy copies. Keep **both** the full normalized name and the core name as
  separate features (`n_eq`, `core_eq`); never collapse to core only. The model then learns when suffixes matter.
* Transliteration, decorator/zero-width removal, generic-token removal and space-free comparison are safe.
* Space-free comparison (JW / partial ratio on the concatenated name) is the fix for domains and handles
  (98% of domain pairs have zero token overlap, but median space-free JW = 1.0).

## Address normalization (token-set ratio / token Jaccard, pool address non-empty)

| step | tsr pos | tsr neg | Jaccard pos | Jaccard neg | verdict |
|---|---|---|---|---|---|
| A0 lowercase | 89.2 | 46.5 | 0.549 | 0.089 | baseline |
| A1 alnum + anyascii | 91.7 | 49.2 | 0.636 | 0.116 | safe |
| A2 norm2: learned abbreviations, leading-zero strip, house prefix, indic state names | 94.5 | 50.0 | **0.727** | 0.130 | **clearly useful** (+0.09 pos vs +0.014 neg) |
| A3 drop placeholders (`null`, `n/a`) | 94.5 | 49.9 | 0.730 | 0.130 | tiny gain, safe |
| A4 token set (order-free) | same | | | | reordering already handled by set measures |

Agreement signals (true pairs vs hard negatives):
* state equal when both known: **99.9% vs 32.8%**; state disagreement: 0.1% vs 55.0% → near-decisive negative evidence.
* ≥1 shared number when both have numbers: 94.3% vs 19.0%.
* **House numbers are deliberately perturbed** in true pairs (`6138→6136`, `801→901`, `1723→172`, `6707→670`,
  leading zeros `00209→209`). Among address-similar pairs with conflicting numbers, the numbers are within 1 edit or a
  truncation in **73% of positives vs 36% of negatives** → use a number-edit-distance feature rather than a binary
  "numbers conflict". The binary conflict feature made the dev model reject ~1,200 obvious true matches (see 09).

## Learned maps (audit)

* `addr_maps.pkl` abbreviations: US 36 (standard USPS + typo fixes `stret`, `drve`, `avnue`) and India 7
  (`pot→plot`, `fat→flat`, `sop→shop`, `rom→room`, `naar→nagar`, `houe→house`, `rad→road`). Low risk; `cdp→city` merges
  census-place labels with "city" (harmless). Canonicalizing to the **short** form (norm2.CANON) is the right choice.
* `state_map.pkl`: US 2,045 keys → 45 state codes (the US data only contains ~45 states), India 1,588 keys → 16 states.
  Values are learned from true pairs with ≥200 support and ≥90% purity — safe. **Collision:** `floor→fl` (CANON) vs `FL`
  (Florida) — `addr_norm` could resolve state=fl from a floor component; minor.
  **France has no map at all** (state unknown for 100% of France rows).
* Indic dictionary (1,347 name tokens, 15 address components): token coverage train S2 97.7% / test S2 96.4%;
  names fully covered 91.3% train / 85.7% test. After normalization indic pairs have only 0.05% zero token overlap.
  The top **uncovered** test tokens are Devanagari shop words (स्टोर्स stores, स्वीट्स sweets, जनरल general, मोटर्स motors,
  बेकरी bakery, ...). In train, **all 15,355 pool records containing these words are unmatched (0.0%)** — they are a
  distractor-generator signature, and test S2 India has ~2× as many (18.9k vs 9.9k). Transliterating them via anyascii
  is fine; additionally, a "contains known-distractor token" feature (learned out-of-fold from `token_match_rates.parquet`)
  is justified by data but is a generator artifact → flag in documentation.

## Noise catalogue (true-pair rates, S2/S3; from 04_noise.txt and this session)

Names: core-exact after norm 52–67%; word-order swaps ~5–6%; typos (edit ≤2) ~5–6%; added/removed tokens 7–18%;
casing (upper 16–23% S2, lower 5–8%); accent injection 5–7%; domains 3.6–5%; indic script 22%/11% (India S2/S3);
junk prefixes ~1–1.4%; `(ID: n)` / phone suffixes; honorific prefixes (`Shri`, `Mr`, `Smt`, `Dr`) on India domains;
brand replacement (synthetic `Lumbelo`-style names) 1–6%; injected `Services/Enterprises`; acronyms (rare, <0.1%).

Addresses: S2 upper-case 23–89%; S3 reorders components and swaps city↔county/CDP and state code↔name (last component
differs 90% vs 23–43% in S2); components dropped (fewer components 36–52% India); house-number perturbation and
leading zeros (3–4%); `#` prefixes; placeholders `NULL/N/A`; indic-script state names (22% of India pairs); empty (3.8–4.9%).

## Normalization recommendation

Safe (apply everywhere): NFKC, zero-width/junk removal, anyascii, learned indic dictionary, lowercase/alnum,
abbreviation canonicalization to short form, leading-zero stripping, placeholder removal, generic-token removal
(`services/enterprises`) for the core name, space-free variant.
Keep-as-feature-only (not as an equality key): legal-suffix removal, domain stripping, honorific stripping.
Never: state-based filtering, expanding abbreviations to long forms, collapsing numbers.
