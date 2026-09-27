# 06 — Match difficulty

Source: `work/true_pair_sims.parquet` (all 7.64M true pairs), `analysis/out/22_difficulty.txt`.
Name similarity = rapidfuzz token_set_ratio on normalized core names; address = token_set_ratio on normalized addresses.

## Name band × address band (all true pairs, % of 7.64M)

| name \ address | ≥90 | 70–90 | 50–70 | <50 | empty |
|---|---|---|---|---|---|
| core exact | 45.07 | 8.90 | 0.85 | 0.04 | 2.65 |
| tsr ≥90 | 20.14 | 3.70 | 0.35 | 0.01 | 1.20 |
| 70–90 | 7.21 | 1.54 | 0.14 | 0.01 | 0.43 |
| 50–70 | 3.94 | 0.79 | 0.08 | 0.00 | 0.11 |
| <50 | 2.38 | 0.42 | 0.04 | 0.00 | 0.01 |

* **~73% are easy** (strong name and strong address).
* **~7% rely on the address alone** (name tsr <70 but address ≥70): domains, brand replacements, indic, acronyms.
* **~4.4% rely on the name alone** (pool address empty 4.4%, or <50): the dangerous zone for precision because names
  are shared by many S1 entities.
* **~0.01% are unsupported by either field** (both weak): unlearnable; ignore.

## Difficulty by name-noise type (true pairs)

| pool-name kind | pairs | zero core-token overlap | median tsr | median JW (no-space) | address tsr ≥70 |
|---|---|---|---|---|---|
| plain | 6.38M | 2.4% | 100 | 1.00 | 93.2% |
| indic script | 551k | 0.05% | 100 | 1.00 | 98.6% |
| domain | 342k | **97.9%** | 65 | 1.00 | 98.2% |
| brand flag | 270k | 42.5% | 100 | 0.59 | 99.4% |
| junk prefix | 90k | 2.0% | 100 | 1.00 | 98.2% |

* After `norm2` the **indic names are no longer hard** (dictionary + anyascii): 0.05% zero overlap. Caveat: the
  dictionary was learned on all train pairs; test coverage is re-checked in 04.
* **Domains** break token features (98% zero token overlap) but are solved by space-free string similarity
  (median JW 1.0, partial ratio 100). → a *space-free* name comparison is a mandatory feature.
* **Brand replacement** (`Butler, Griego and Trenholm P.C.` → `Lumbelo`) destroys the name entirely; the address is
  almost always strong (99.4%). The `f_brand` flag has limited precision (fires on 42.5% overlap cases too).
* Of 269k zero-overlap pairs not explained by indic/domain, 115k carry the brand flag; the rest are hashtags/handles
  (`#coopersroyal`), unrecognized domains (`hasthatrading.c0m`, `reedgulfalgonquincom`, `... - 2011155356`, `(ID: 75752)`),
  honorific-prefixed domains (`Shri spacemarine.com`), acronyms (`EGCS`, `OE`), heavy typos (`Zhooith` ← `Zenith`).

## Difficult segments for the matcher (hypotheses tested in 08/09)

1. Empty pool address + generic name (ambiguous by construction) → abstain unless name is rare/specific.
2. Brand-replaced / domain names + address only → need address specificity (rare tokens, house number agreement).
3. Generic S1 names in India with sparse addresses (`2 , mumbai , mumbai city , mh`) → many near-identical S1.
4. S3 address substitutions (city → county/CDP, state name ↔ code) → state/city canonical maps matter.
