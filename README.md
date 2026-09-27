# Business Entity Resolution — ML Challenge 2026 (team pipeline)

**Status (2026-09-27): best public LB = 0.980 (V14).**
History: v2 0.936 → V3 0.966 → V5 0.967 → V5s3r 0.973 → V9 0.977 → **V14 0.980**.
Clean labelled holdout (folds 0/8/9, US/India): US 0.9939, India 0.9922.
`docs/analysis/STATE.md` is the single source of truth (every experiment, number and decision, including the ones
that failed) — read it before changing anything.

## Layout
| Path | What |
|---|---|
| `run_pipeline.py`, `scripts/`, `config.yaml`, `configs/`, `launch.sh`, `userdata.sh` | AWS production pipeline (normalize → blocking → features → stage-1/2 GBDTs → decisions). V9/V14 scores come from `configs/config_v8.yaml`. |
| `local_scripts/` | Laptop-side analysis and post-processing, numbered chronologically (`NNN_*.py`); each writes `docs/analysis/out/NNN_*.txt`. |
| `docs/analysis/` | Forensics / design notes (`01`–`20_*.md`), `STATE.md`, all script outputs (`out/`), AWS run reports (`aws_v*`). |
| `utils/validate_submission.py` | Official format validator. |
| `docs/Documentation_template.md` | Methodology template for the final zip (to be filled). |
| `docs/idea.txt` | Relayed "0.994 blueprint" — audited against labelled data in STATE.md (only the cross-encoder part holds). |

Data (`dataset/`), intermediate files (`work/`), outputs (`output*/`, `*.tsv`, `*.parquet`) and model weights are **not**
in git (see `.gitignore`).

## Pipeline (V14)
1. **Normalize** (`scripts/normalize.py`, `normlib.py`): Unicode/transliteration, abbreviations, legal forms, house numbers.
2. **Blocking** (`build_blocking.py`, `retrieval.py`): sparse TF-IDF top-k — reverse pool→S1 (name+address, char-3gram),
   forward S1→pool, exact core-name blocks; per country. Stage-1 cascade keeps p1 ≥ 1e-4 = `candidate_pairs.tsv`.
3. **Features** (`build_features.py`, `featx.py`, `augment_features.py`): ~70 name / address / number / legal / retrieval features.
4. **Train-like-test** (`orphanize.py` + `local_scripts/140_make_aug.py`): hard look-alike orphans and V8 synthetic
   look-alike records (+k house-number twins, descriptor siblings) so that training sees the test's distractors.
5. **Models** (`train_model.py`, `collective.py`, `stage2_xgb2.py`): stage-1 LightGBM (3-way cross-fit) → collective
   sibling/consensus features → stage-2 LightGBM + XGBoost (+XGB2).
6. **Stage 3, per country** (`local_scripts/173c_stage3_country.py`): LightGBM re-reasoning; global + US + India models,
   averaged for US/India, global only for unseen countries (France).
7. **Cross-encoder** (`local_scripts/235`, `238`–`240`): multilingual MiniLM-L12 cross-encoder (Apache-2.0, 118M params,
   fine-tuned on train folds 1–6 only) scores the uncertain band 0.02 ≤ p3 ≤ 0.98; a LightGBM stacker (fit on clean
   fold 7) merges it with p3. Holdout: US +0.0014, India +0.0029. Applied to US/India only (France: untrained, label-free
   checks negative).
8. **Decision** (`scripts/tune_decision.py`): expected-F0.5 set decoder (G), record exclusivity (each pool record goes to its
   best S1), per-source caps (S2 ≤ 5, S3 ≤ 6), α = 0.75 tempering with the stage-1 score for US/India.
9. **Validated rules**: French dotted legal forms (`220`), house-number formats / OCR-digit names (`232`, `233`),
   French identical-name look-alike removal (`236`, `237`).

## Reproduce
Local scripts expect the original working layout (`scripts/` = local scripts, `aws/scripts/` = pipeline code, plus
`work/`, `dataset/`, `analysis/out/`). Recreate it next to the clone:
```bash
mkdir -p ../er/aws ../er/analysis/out ../er/work && cd ../er
ln -s ../Business_ML_Pipeline/local_scripts scripts && ln -s ../../Business_ML_Pipeline/scripts aws/scripts
ln -s /path/to/student_resource/dataset dataset && cp ../Business_ML_Pipeline/utils -r . 2>/dev/null || true
```
```bash
python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
pip install torch transformers                # cross-encoder steps only (MPS or CUDA)
python tools/tsv_to_parquet.py /path/to/dataset ./work          # or local_scripts/00_to_parquet.py
python local_scripts/140_make_aug.py                             # V8 synthetic look-alikes -> work/aug_v8_*.parquet
# AWS stage-1/2 run (~16 vCPU / 128 GB, ~5 h): see launch.sh; CONFIG=configs/config_v8.yaml
export AWS_PROFILE=<profile> REGION=ap-south-1 BUCKET=<private-bucket> ITYPE=r6i.4xlarge CONFIG=configs/config_v8.yaml
./launch.sh check && ./launch.sh upload && ./launch.sh launch && ./launch.sh logs
# local: fetch p2 scores + decisions + V9 base files
bash local_scripts/180_v9_build.sh
# local: per-country stage 3 -> output_v11s3, then V11 -> V14
AUG=v8 python local_scripts/173c_stage3_country.py ...           # see STATE.md "FINAL LOCAL BUILD (V11)"
python local_scripts/229_v11_finalize.py output_v11s3 output_v11pre
python local_scripts/220_fr_legal_rule.py output_v11s3 output_v11pre/matching_results.tsv output_v11
python local_scripts/232_noise_fixes.py && python local_scripts/233_noise_fixes2.py     # V12 additions
python local_scripts/236_fr_threshold.py && python local_scripts/237_fr_threshold_k.py  # V13 French removals
python local_scripts/235_ce_pilot.py 120000     # cross-encoder (~20 min on Apple MPS)
python local_scripts/238_ce_score.py && python local_scripts/239_ce_stack_holdout.py && python local_scripts/240_ce_build_v14.py none
python utils/validate_submission.py --matching output_v14/matching_results.tsv --candidate output_v14/candidate_pairs.tsv --test-dir dataset/test
```
Exact arguments and the order of the V11–V13 assembly are recorded in `docs/analysis/STATE.md`; a single end-to-end
script for the final zip is still to be consolidated.

## What did NOT help (measured, don't repeat without new evidence)
High probability threshold (>0.92) on our scores; zero-tolerance house-number veto (true copies often have ±k numbers);
dedicated singleton gate (≈0); global bipartite matching (identical to our per-record exclusivity); test-density
augmentation retrain (V10 ablation −0.0027); adaptive stop words; pretrained sentence embeddings without fine-tuning;
teammate TSVs T1/T2/T4 (addition types 1–9% precision).

## In progress / next
* **CE v2** (`243a`/`243b`, `244`/`245`): all 314k hard pairs, 96 tokens, 2 epochs; holdout gate vs V14 → V15.
* **French cross-encoder input canonicalization** (`247` t2: accents, N°/BIS numbers, R./AV. abbreviations, legal forms) —
  accept only if US/India holdout holds and French decisions look like US/India ones.
* **Bigger multilingual cross-encoder** (mDeBERTa-v3-base / XLM-R) on a GPU; CE ensembles.
* Final zip: `output/` + `code/business_entity_resolution/` + filled `Documentation_template.md`.

## Rules we follow
No external data lookups; models ≤ 8B params, MIT/Apache-2.0; every S1 once; matches ⊆ candidates; never tune on the
leaderboard; France (unlabelled) handled by country-agnostic fallbacks unless label-free evidence supports more.
