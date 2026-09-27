# 12 — Targeted data forensics (session 3)

Only questions not answered by 01–06 were investigated. Raw outputs: `analysis/out/60–67*`.

## F1. Are distractors random or near-twins? (`60_distractors.txt`)
Top-1 reverse cosine (pool record → best S1), quantiles 10/25/50/75/90%:
true copies 0.60/0.76/0.89/0.96/1.00; **unmatched distractors 0.42/0.55/0.69/0.80/0.88**.
25% of distractors have a top-1 S1 at cosine ≥0.8 → a sizeable **near-twin** population. Their top-1 S1 is a singleton in
9% of cases (base rate 5.6%) → twins are made from arbitrary S1 entities, not preferentially from singletons.
Singleton S1s: best candidate cosine median 0.77 vs 0.97 for matched S1s → mostly separable.

## F2. The generator's "hidden base record" (`64_sibling_numbers.txt`) — the key discovery
In the ambiguous zone (name ≥90, address ≥80, house number conflicting with S1):
a true copy's first house number equals **another true copy's** number in **69.0%** of cases vs **0.27%** for negatives.
For S1 with ≥3 true copies, S1's own house number equals the copies' majority number only **82%** of the time → S1 is
itself a perturbed rendering; copies agree with each other more than with S1.
→ Decision: **collective sibling-consensus features** (number/street/name/address agreement with other probable copies of
the same S1). Measured gain in 14/09 update: holdout 0.9678 → 0.9707.

## F3. Where the remaining loss is (`66_oracle.txt`, stage-2b holdout 0.9707)
Oracle "fix all FN in segment" / "fix all FP": all FN +0.0177, all FP +0.0078; empty pool address FN +0.0062 (largest);
India FN +0.0075 / FP +0.0042; S3 FN +0.0105 / FP +0.0039; brand/domain/weak-name/weak-address each ≤ +0.001.
Perfect matcher on the same candidates: 0.9962 (blocking is not the bottleneck).

## F4. Empty pool addresses (`67_empty_addr.txt`)
| S1 sharing the core name | positives | accepted | negatives | false accepts |
|---|---|---|---|---|
| unique (1) | 660 | 87.7% | 39,863 | 0.05% |
| 2–3 | 210 | 6.2% | 12,012 | 0.12% |
| >3 | 249 | 0.8% | 17,287 | 0.02% |
With no address and several same-named S1, the owner is undecidable from the data → irreducible (≤ +0.004).

## F5. France formatting, unlabeled test text (`62_france_tokens.txt`)
Address token counts (S1 vs S2): `av` 2.2k vs 27.6k, `bd` 2.0k vs 11.6k, `imp` 0.2k vs 6.0k, while S1 mostly uses the
long forms already canonicalized (`r`, `ave`, `blvd`, `impasse`). Names: pool has a stray `r` token (11.7k vs 6 in S1)
from `S.à r.l.`/`Sàrl` splitting. Legal forms SARL/SAS/EURL/SASU/SCI dominate France names.
→ add `av→ave`, `bd→blvd`, `imp→impasse` (address) and `s a r l → sarl` (name). No labels → effect unmeasurable; the
changes are formatting-only and cannot create cross-entity collisions beyond what the long forms already do.

## F6. Model capacity and data size (`63_learning_curve.json`, `68_capacity.json`)
Holdout vs training S1: 3.3k 0.9551 → 6.6k 0.9588 → 13.1k 0.9612 → 26.3k 0.9625 (≈ +0.002 per doubling, shrinking).
Bigger trees (255 leaves) 0.9708 vs 0.9707 → no capacity gain. Full-train fitting (~2M S1) is expected to add ≈ +0.003–0.006.

## F7. Previously established (not recomputed)
Counts, missingness, scripts, casing, duplicates, legal suffixes, abbreviations, indic coverage, label conflicts,
legal-suffix-removal collisions, house-number perturbation: see 01, 03, 04, 05, 06.
