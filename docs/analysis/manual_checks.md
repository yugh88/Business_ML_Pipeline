# Manual checks (high-value only)

Full extracted rows: `analysis/out/49_manual_cases.txt` (script `scripts/49_manual_cases.py`). Views:
`work/label_collision_groups.parquet`, `work/dev_holdout_fp.parquet`, `work/dev_holdout_fn.parquet`,
`work/blocking_misses_eval.parquet`. Look up any ID in `work/train_s{1,2,3}.parquet`.

## 1. Near-twin distractors vs perturbed true copies (decides the precision ceiling and threshold)

| S1 (singleton) | S1 address | merged pool record | pool address | p |
|---|---|---|---|---|
| S1-149161420 Cascade Partners Inc. | 10008 Tanner Mill Drive, Mckinney, TX | S2-228247022 Cascade Partners Inc | 10012 TANNER MILL DRIVE, TX, MCKNINEY | 0.982 |
| S1-856941601 PG Added Clinic | C/O. Biji Joy, Vallikunnel House … Palakkad | S2-393427732 PUG ADDED CLINIC | identical address | 0.989 |
| S1-335230370 Beacon Digital Innovations Inc. | 21 Deer Pond Drive, Leicester, MA | S3-86828362 Beacon Digital Innovations Inc | 24 Deer Pond Dr, Leicester | 0.967 |

Compare with TRUE pairs that the model rejected: S1-473193331 `American Program, 6138 5, Ashland, KY` ↔ S2-227907077
`6136 5`; `Temple Catholic Church 60 Appletree Court` ↔ `76 APPLETREE CT` (see `dev_holdout_fn.parquet`).
**Inspect:** ~20 rows of each kind. **Question:** is there *any* visible cue separating "distractor twin" from "noisy true
copy" (which field is perturbed, by how much, name-letter vs number, S2 vs S3)?
**Decision impact:** if no cue exists → accept the ceiling and raise the threshold slightly (precision-first); if a cue
exists (e.g. distractors perturb the number by +2..+4 but true copies by a single digit edit) → add that feature.

## 2. Wrong-S1 assignments that the 1-to-many constraint should fix

| predicted S1 | pool record | its true S1 |
|---|---|---|
| S1-12295319 Pune Water Pvt. Ltd. | S3-677625883 `pune business private limited` (same address) | S1-793405485 Pune Business Private Limited (same address!) |
| S1-856631358 Wood's Family Practice, Albuquerque | S2-266280893 `Wood's Family Practice`, **empty address** | S1-169565384 Wood's Family Practice P.C., Silver Spring |
| S1-691474330 Smart Federal, Indianapolis | S2-729082550 `Smart Federal`, **empty address** | S1-472767262 Smart Federal Next LLC, Kansas City |

**Inspect:** confirm these are unresolvable without competition. **Decision impact:** if yes → make "argmax over all S1 +
abstain when the top-2 S1 are close" mandatory, and abstain on empty-address pool records whose name is shared by ≥2 S1.

## 3. Label conflicts (identical normalized pool records → different S1)

`S3-606363942 Guntur LLP Center | Floor, Guntur, AP` → S1-386048168; `S3-955043118 Guntur Ltd Center | Floor, Guntur, AP` → S1-581567355;
`S3-511282867` / `S3-13583566` (`Hyderabad … Services | 40, Hyderabad`) → S1-145606005 / S1-986902592;
`S2-58157248` / `S2-550808485` (`Surat … | A-1033-1034 Hari Om Marketring Road`) → S1-838592051 / S1-86030519.
Only 21 such groups in 7.6M pairs. **Inspect:** 3–4 groups. **Decision impact:** if they are generator collisions (generic
city + suffix), treat as irreducible; ensure the model does not propagate matches across identical twins blindly when the
twins' legal suffix differs (keep suffix features).

## 4. Matched record with an unmatched identical twin (possible missing label)

S2-808550830 (UNMATCHED) `Advanced Defense Concepts Co | 19 DICKINSON STREET, SPRINGFIELD, MA` vs S2-505044026 → S1-964751785
`ADVANCED DEFENSE CONCEPTS LLC | 19 DICKINSON ST`; S3-988298393 (UNMATCHED) vs S3-826536339 → S1-718363295 (Lenet Partners);
S2-952942584 (UNMATCHED) vs S2-415148072 → S1-816302378 (Prabhat Brothers LLP vs Private). 186 groups total.
**Pattern:** the unmatched twin differs in the *legal suffix* (Co vs LLC, LLP vs Private). **Decision impact:** if confirmed
as intentional distractors → legal-suffix disagreement is informative (supports keeping `n_eq` separate from `core_eq`,
see 04); do **not** propagate matches to twins with a different suffix.

## 5. Brand-replaced names with corrupted addresses (true pairs that look wrong)

S1-327549953 Baramati Tyres ↔ S3-773045158 `Nexorbi | Plot C-985 Plot No-p-15, Jalochi, MH`;
S1-302708426 Smart Energy Corporation ↔ S3-502949181 `Onyxzeta | E, Waupaca CITY, Wisconsin`;
S1-793715787 Red Foundation ↔ S3-46089653 `Synzeph | A 1902, MH, Chennai`.
**Decision impact:** if these are accepted as real labels, they are unlearnable (≈940 pairs, 0.01%) — ignore; do not add
rules to chase them.

## 6. Blocking misses with strong name and address

S1-750648817 ↔ S2-722148340 (`>> East Management Piate Limited | MUMBAI, DADAR (WEST), B-18, महाराष्ट्र`);
S1-97009106 ↔ S3-649507850 (`Business Trading Pliate-Limited | Plot No. B 93, Ghaziabad, UP`).
Generic India names (`East Management`, `Business Trading`) crowd the reverse top-k. **Decision impact:** decides whether to
spend compute on rev-na cap 100k@20 (config D, +0.5pp recall, ~3× CPU) or accept config A/C.

## 7. Distractor signature

S2-803586074 `सूर्य कंसल्टेंट्स स्टोर्स प्राइवेट लिमिटेड`, S2-266306666, S2-269090609, S2-420412870: Devanagari "stores/
sweets/general/motors/bakery/traders/jewellers/pharmacy" names — 15,355 in train, **0 matched**. **Inspect:** are these
plausible real businesses near real S1s? **Decision impact:** whether to add an explicit distractor-token feature
(data-driven, but a generator artifact — mention in the methodology doc).

## 8. France (test only, no labels)

Eyeball `analysis/out/49_manual_cases.txt` §H: S1 `Dolus Hilaire Construction SARL | 30 bis Rue René Delissen, Dunkerque,
Hauts-de-France`; S2 `Mérignac Collectif Sàrl | Gironde, (4) R. LOUIS BERON, MÉRIGNAC`; `NO 19 R CHARLES JOSSE, ST.-HERBLAIN`.
**Check:** `R.`/`R` = rue, `ST.-` = saint, `Sàrl` = SARL, `bis`, department names (Gironde) vs regions (Hauts-de-France).
**Decision impact:** whether to add a small French canonicalization list (rue/r, avenue/av, boulevard/bd, saint/st, bis/ter,
sarl/sàrl) — generic language normalization, not external entity data — before the test run.
