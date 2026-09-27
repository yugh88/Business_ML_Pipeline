# 19 — What the data generator does (and why V5 scored 0.967 on the hidden test)

All findings are label-free on test and verified with labels on train (scripts 110–177, outputs in `analysis/out/`).

## 1. Twin entities (the main hidden-test gap)
* The generator creates **twin businesses**: same name, same street, house number = S1's number **+ k, k ∈ {1,2,3,4,5,7,9}**
  (k = 6 / 8 essentially never; **never minus**). Train: identical-name records at +1..9 are 85% distractors, at −1..9 82% true
  copies (true copies perturb numbers symmetrically). (119, 140)
* **Train twins are single records; test twins come as clusters** (size-2 groups spanning both sources: 12.7 vs 0.89 per 1k S1
  in the US, 14×). Stage-2 collective / consensus features learned "records supported by similar records = copies", so they
  **promote twin clusters** in test. Evidence without labels: V5 accepts +k 43.8 vs −k 18.6 per 1k US S1 (true copies are
  symmetric); the pairwise stage-1 model is symmetric (19.0 vs 15.5); "promoted" records (stage-2 accepts, stage-1 rejects) are
  80/1k (US) and 135/1k (India) in test vs 36 / 69 in train. ≈ 7–10% of test S1 carry a false merge → 0.991 → 0.967. (121, 133, 134)
* Twin copies without a house number / with a truncated number explain the remaining "empty" and "d100+" excess. (122, 131)

## 2. Descriptor siblings
* Sibling businesses add a descriptor word (US/India: group, holdings, enterprises, ventures, exports, overseas, industries,
  infratech, india, solutions, trading, foods, uptown…; train match rate ≈ 0%), usually with a different house number. Test has
  ~2× their density (India especially). (110, 124)
* **France (no train labels)**: address behaviour separates siblings from copy noise — siblings mostly change the number,
  copies keep it. French siblings: holding, participations, international(e), distribution, SNC. French **copy noise**:
  Groupe, France, Développement, & Fils, & Associés, Services, Cie (85–90% keep the exact number). (124–126)

## 3. Copy noise and per-source versions
* Copy noise replaces the category word with a generic one (US: Center/Services/Partners; India adds Sri/Smt/Shri/Dr/Mr;
  France above), typos incl. OCR digits (5ervices, 6roup, 8eatty), DBA/fka/t-a aliases, website forms, legal-suffix changes.
* **Per-source versions**: copies of one entity in one source share a version of the address (zero-padded / truncated /
  ranged number, city alias). Consequences: (a) the house number of a copy's source version is shared by the other copies of that
  source (83%), rarely by the other source (6.6%); (b) **address fingerprint** — a copy's raw address equals another strong
  same-source candidate's raw address 22–25% of the time, siblings/distractors 0.1–2%. (117, 153, 157)
* France: category-word swaps at the same number (Club↔École↔Amicale…) share the fingerprint like copies (19.6% vs 20.4%) →
  mostly copies, not siblings (a rule rejecting them would have been wrong).

## 4. Orphans, singletons, chains
* ~18% of test entities have copies but no S1 record (test pool 5.7 records per S1 vs 4.7 in train) → orphan simulation (V3/V5).
* Orphan-exposed S1 remain the weakest segment (F ≈ 0.968 on holdout after stage-3; errors = look-alike orphan copies appended
  to a correct set). Chain names (core name shared by ≥5 S1, 30% of S1): F ≈ 0.988. (176, 177)
* Name-only (empty-address) copies of chain names are the largest missed-copy bucket and are genuinely ambiguous: exact raw-name
  linking to other copies is only 1–3% precise. (152, 162)

## 5. Things that are NOT the problem (checked)
* Hindi and other Indian scripts: 23% of India pool records, same share in train/test, transliterated (train dictionary +
  anyascii); recall/precision on holdout 0.984–0.999 / 0.997–0.999, better than Latin-script records. (160, 161)
* No ID or row-order leakage (S1↔copy or copy↔copy). (111, 177b)
* PO-box numbers read as house numbers: real but tiny (recall 95.1% vs 98.2% on 0.3% of true pairs).

## 6. What V7 → V8 → V9 do about it
* V7: V5 + rules (twin +k with same-source own-number support, 2-source twin groups, train descriptors, French siblings).
* V8: retrain with **327k synthetic twin/sibling clusters** cloned from real copies (140) + features: signed number difference,
  twin-offset flag, sibling/noise word flags, per-source own-number / group support.
* V9: V8 + **stage-3** — a LightGBM re-reasoning over each S1's candidates after stage 2 (promotion p2−p1, score-weighted house-number
  groups by source, address fingerprints, word classes), trained on out-of-fold scores (folds 1–6), policy tuned on fold 7.
  On V5 scores stage-3 alone adds +0.0015 holdout F0.5 and reduces the test twin excess. (171–175)
* Tested and rejected (inconsistent across folds / worse test diagnostics): S2↔S3 relational similarity features (176),
  fingerprint-based rescue of rejected candidates (154), exact-name linking for empty-address copies (162), second stage-3 pass (174).
