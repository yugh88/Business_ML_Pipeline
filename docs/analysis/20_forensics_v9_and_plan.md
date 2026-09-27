# 20 — Forensics of the leaderboard steps (0.967 → 0.973 → 0.977) and the plan for 0.99+

Evidence: scripts 182–186 (`analysis/out/182_forensics.txt`, `184_density.txt`, `185_country_uncertainty.txt`, `186_self_expected_f.txt`).

## 1. What produced 0.973 → 0.977 (V5s3r → V9)
* 112k test pairs changed (64.9 per 1k S1; removed ≈43, added ≈22). Biggest removals: same-number records with heavily edited
  names 9.6, identical-name / no-number 4.8, identical-name / far number 4.4, no-number / edited name 4.3, +k twins 2.8 per 1k.
* **On the labelled holdout the same kind of removals are 83% true copies** (V9's clean holdout is 0.0006 *lower*). The LB still
  rose +0.004 ⇒ on test these categories are mostly false: a budget over the changes puts the test true-copy rate of the removed
  pairs at ≈42% (≈25 false merges per 1k S1 removed, ≈18 true copies lost). Removed examples confirm it: brand/first-word replaced
  at the same address (Williams's→Hunter Allied Appliance, SY→Yaeger Regional Bsp Clinic, Cruz→Holton Allegro, LZA→Rebert Vanguard
  Music) — train says such first-word swaps are only 28% true; V9 removed 2,086 of them.
* Hypotheses tested and **refuted**: sibling sub-clusters sharing the same name edit (shared-edit rate in test predictions =
  train truth, 12.8 vs 12.8 per 1k); S2↔S3 similarity features (176, worse on test checks); rare-number blocking (≤0.2/1k).
* Twin mechanism is now **closed**: V9 identical-name acceptances are symmetric for +1..9 / −1..9 and +10..99 / −10..99.

## 2. Where the remaining gap is (label-free)
* The model is calibrated on train but **over-confident on test**: self-expected F0.5 (candidates only) holdout 0.9926 vs actual
  0.9915 (0.001 optimism); test 0.9895 vs LB 0.977 (**0.0125 optimism**) ⇒ ≈45–50 confidently-accepted false merges per 1k test S1.
* Test candidate pools contain **1.3–2× the train density of look-alike distractors** (test density − train true-copy density vs
  train negatives, per 1k S1): US identical-name/number-changed +222 (twins; now rejected symmetrically), US swap-extra/number-changed
  +102, **India swap-extra/same-number +54 (true rate 0.82 → 0.72)**, **India identical/same-number +30 (0.967 → 0.945)**, light-edit
  same-number +9, descriptors +365–418 (already rejected). Identical names with empty address are **not** over-represented.
* **France is the least certain country**: weak accepts 12.0% of S1 (US 4.7%, India 6.0%, holdout 4.3–5.0%), S1 with a 0.2–0.8
  candidate 15.8% (US 7.0%), flip rate between versions 106/1k (US 48). 66% of French uncertainty is at the **same house number**
  and is about the name: alias-only names, initials (AC, FB, CG), category swaps (Loisirs→Sport), noise swaps (Club→Et Fils).

## 3. What the hidden test most likely is
The train generator with (a) ≈1.5–2× more look-alike distractor entities per S1 (twins at +1..99, same-address siblings with a
swapped / replaced / added word, co-located alias-named businesses, identical-name same-address entities), (b) distractors with
several copies (clusters), (c) ≈18% of entities without their S1 (orphans), (d) the French locale (no labels).
Each LB gain so far came from making the model see one of these at test density: orphans (+0.030), twins/descriptors + stage-3
(+0.006), synthetic twin clusters (+0.004).

## 4. Plan (V10 and after) — follow the pattern that raised the score
1. **Density-matched hard negatives for every over-represented category** (not only +k twins): same-address siblings with word
   swap / first-word replacement / extra word, identical-name same-address entities (India), twins +10..99, co-located alias
   businesses; 1–3 copies each, both sources, cloned from real copies + one noise op (as V8). Rates set so that train candidate
   densities per category ≈ the test densities measured above (label-free).
2. **France-aware normalization and word classes**: department ↔ region map (state agreement), French street abbreviations
   (R., Av., Bd, All., Rte, Pl., Imp., Ch., Crs, Sq), Nº / N° / No., bis/ter, French legal forms; French copy-noise vs sibling words from
   the address-behaviour + fingerprint tests; alias/initials detectors (initials of the S1 name, synthetic-brand tokens).
3. Retrain full pipeline (AWS ≈8–9 h, ≈$9–10) + stage-3; tune on the augmented fold 7.
4. **Label-free acceptance tests before any submission**: ±k symmetry kept; per-category accepted density ≈ train true-copy
   density in each over-represented category; test self-expected-F optimism shrinks toward the holdout's 0.001; France weak-accept /
   mid-candidate rates fall toward US levels; augmented holdout not worse than V9 by >0.0005.
5. Submit, re-run this forensic on the LB delta, repeat on the next residual category.
