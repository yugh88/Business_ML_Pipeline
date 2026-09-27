# 01 — Data quality & forensics

Sources: `work/01_inventory.txt` (previous session, full-file stats), `analysis/out/20_quality.json`
(`scripts/20_quality.py`, this session: normalized duplication/ambiguity), `work/02_gt.txt`.

## Row counts & schema

| split | S1 | S2 | S3 | GT rows |
|---|---|---|---|---|
| train | 2,206,821 (US 1,323,633 / India 883,188) | 5,034,616 | 5,285,603 | 2,206,821 |
| test | 1,732,544 (India 809,986 / US 663,106 / **France 259,452**) | 4,887,273 | 5,082,316 | — |

Schema everywhere: `entity_id, business_name, business_address, country` (all strings). IDs `S{n}-<8..12 digits>`,
no duplicate IDs, no malformed IDs, no train/test ID overlap. IDs and row order carry **no** signal (corr ≈ 1e-4).

## Source character

* **S1 is clean**: Title Case ASCII, no empty fields, no duplicated (name,address). Only oddities: 506 names start
  with `@`, 493 with `#`; 554 addresses with mojibake bytes (`\x80`, `\x93`). Test S1 France has accents (40.8k names).
* **S2/S3 carry all noise** (see 04_noise_patterns.md). S2 addresses are 63% ALL-CAPS (S3 ~0%); S3 reorders components
  and appends different city/state variants (last component differs in 90% of S3 pairs vs 23–43% of S2).
* Junk decorators on names: `>>`, `***`, `--`, `...` prefixes (~15k each per source), trailing `&`, `+`, `[&]`, `(&)`.
* Scripts in names (train S2 names, 200k sample chars): Devanagari, Telugu, Kannada, Tamil, Bengali, Gujarati,
  Malayalam, Oriya, Gurmukhi; zero-width chars. Only in India rows. Latin accents injected in US *and* India names (~7%).
* Domains/handles: ~4% of S2/S3 names (`hghpackaging.com`, `@henrymanagement`, `#coopersroyal`, `...c0m`).
* Placeholder tokens inside addresses: `NULL`, `<NULL>`, `N/A`, `null`.

## Duplication and ambiguity (normalized, within country) — new this session

| | S1 core-name shared by ≥2 records | shared by >10 | pool (S2/S3) core shared | pool identical normalized address shared |
|---|---|---|---|---|
| train US | 46.5% | 18.1% | 47–49% | 30–35% |
| train India | 53.9% | **33.0%** | 51–54% | 19–33% |
| test France | 46.3% | 18.1% | 52% | 45% |

Business names are **not identifying**: half of all S1 entities share their normalized core name with another S1
entity (e.g. `Primary Care Group` ×253, `Bordeaux Club SARL` ×205 in test). Name-only matching cannot be precise; the
address must break ties. S1 (name,address) pairs are unique (S1 is truly deduplicated).

Pool duplicates: 1.13M train pool records sit in 522k groups with identical normalized (core name, address); these
are almost always copies of the same entity (see 05_ground_truth_analysis.md).

## Field lengths

Names: median 24 chars (S1), 25 (S2/S3), max ~120. Addresses: S1 median 41 chars, max 268. Pool addresses: median
37–43 but 3–4% empty. Name tokens ≈3.5 (US) / 3.8 (India) / 3.2 (France); address tokens ≈8 (US) / 14–17 (India) / 9–11 (France).

## Train vs test shift

* **France** exists only in test (15% of test S1, 14% of test pool). No labels, no state map, French address grammar
  (`RUE`, `AV`, `BD`, `Hauts-de-France`), legal forms SARL/SAS. Club/association-heavy names.
* Pool-to-S1 ratio: train 4.68 records per S1; **test 5.75** (US 5.75, India 5.77, France 5.5). Either more matches per
  S1 or (more likely) more unmatched distractors in test → precision estimated on train is optimistic; thresholds
  should be set with margin (see 09).
* Name-only overlap train↔test ~11–12% of distinct names (generic names), exact (name,address) overlap ≈0.
* Accented pool addresses jump from 1.2k (train S2) to 128.9k (test S2) — presumably the France rows (French place
  names); anyascii strips accents, so normalized France addresses are plain ASCII.

## Unicode/character issues

* NFKC + zero-width removal + anyascii transliteration is needed (implemented in `norm.base_clean`/`norm2`).
* Mojibake in ~0.03% of S1 addresses — harmless after alnum filtering.
* Leet substitutions inside words (`MARKETIN6`, `lnternational`, `5s` for `Ss`, `c0m`): ~2% of pool names.
