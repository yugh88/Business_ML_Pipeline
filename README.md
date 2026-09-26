# Business Entity Resolution — run instructions (V3 pipeline, public LB 0.966)

Two-stage GBDT entity matcher for the Amazon ML Challenge 2026 Business Entity Resolution task.
Default `config.yaml` = **V3** (the version that scored 0.966 on the public leaderboard).
`configs/config_v3nn.yaml` (+ retrained XGBoost + fusion NN) and `configs/config_v5.yaml` (hard look-alike orphan
simulation) are the next experiments.

## What the pipeline does (one line per stage, all in `scripts/`)
1. `normalize.py` — Unicode/transliteration/abbreviation normalization (`normlib.py`, learned dictionaries `*.pkl`).
2. `build_blocking.py` + `retrieval.py` — sparse TF-IDF top-k retrieval: reverse (pool→S1, name+address and char-3gram),
   forward (S1→pool), exact sorted-name blocks; per-country; memory-bounded.
3. `build_features.py` + `featx.py` + `augment_features.py` — ~70 pairwise features (name/address/number/legal/retrieval).
4. `orphanize.py` — makes TRAIN look like TEST: drops a share of S1 entities but keeps their S2/S3 copies as negatives
   (test pool contains copies of businesses absent from S1).
5. `train_model.py` — stage 1 LightGBM (3-way cross-fit) → learned candidate pruning → collective/sibling features
   (`collective.py`) → stage 2 LightGBM + XGBoost.  (`stage2_xgb2.py`, `stage2_nn.py` = optional extras.)
6. `evaluate.py` — S1-grouped full-partition validation; picks model variant + decision policy on fold 7 only;
   reports holdout folds 0/8/9.
7. `predict_test.py`, `build_submission.py`, `validate_submission.py` — test decisions, TSVs, official validator.
8. `tune_decision.py` + `finalize_submission.py` (run locally afterwards) — joint choice of candidate-pruning threshold
   (smaller candidate set) and decision policy; writes the final TSVs.

## 0. Setup
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # + `pip install torch` only for configs/config_v3nn.yaml
python tools/tsv_to_parquet.py /path/to/student_resource/dataset ./input_parquet
```

## 1. Smoke test on a laptop (~1 min, tiny subset)
```bash
python make_mini.py ./input_parquet ./mini/input
sed -e 's#workdir: .*#workdir: "./mini"#' -e 's/workers: [0-9]*/workers: 2/' -e 's/buckets: {train: [0-9]*, test: [0-9]*}/buckets: {train: 4, test: 3}/' \
    -e 's/num_threads: [0-9]*/num_threads: 4/' -e 's/query_shard_rows: [0-9]*/query_shard_rows: 5000/' config.yaml > mini/config.yaml
ER_CONFIG=mini/config.yaml python run_pipeline.py        # ends with "PASS — no blocking issues found"
```

## 2. Full run (needs ~16 vCPU / 128 GB RAM, ~3–5 h) — recommended on AWS EC2
Local full run works on any Linux box with enough RAM: put the 7 parquet files in `<workdir>/input/` and
`ER_WORKDIR=<workdir> python run_pipeline.py` (S3 is skipped while `s3_prefix` contains CHANGE-ME).

AWS (one EC2 instance, state checkpointed to S3, instance self-terminates; needs an IAM identity that can create a
bucket, an instance role and run EC2):
```bash
export AWS_PROFILE=<profile> REGION=ap-south-1 BUCKET=<new-private-bucket> ITYPE=r6i.4xlarge CONFIG=config.yaml
mkdir -p ../work && cp input_parquet/*.parquet ../work/      # launch.sh uploads ../work/*.parquet
./launch.sh check && ./launch.sh upload && ./launch.sh launch
./launch.sh logs          # progress;  ./launch.sh status
./launch.sh fetch         # when done: output/*.tsv + validation reports into ../output, ../analysis/aws_validation
```
Every stage is resumable: if the instance dies (e.g. spot reclaim, `SPOT=1`), run `./launch.sh launch` again.

## 3. Final decision + smaller candidate set (local, ~2–4 GB RAM)
Download the stage-2 score files (`s3://<bucket>/entity-resolution/p2/`) to `./p2`, then:
```bash
ORPHAN_FRAC=0.16 TUNE_VARIANTS=p2x,p2,p2blend TUNE_SKIP_TEST=1 python scripts/tune_decision.py ./p2 ./input_parquet ./tuned
# copy the printed pick into decision.json, e.g.
# {"tau": 0.05, "variant": "p2x", "policy": "G_expected_f", "a": 1.25, "lam": 0.0, "cap": 1}
python scripts/finalize_submission.py ./p2 ./input_parquet ./output decision.json
python scripts/official_validator.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv \
       --test-dir /path/to/dataset/test --check-ids
```
(`tune_decision.py` needs `train_pairs.parquet` in the input dir for the ordinary-view check:
`python -c "import polars as pl; g=pl.read_parquet('input_parquet/train_gt.parquet'); g.with_columns(pl.col('matched_entity_ids').str.split(',')).explode('matched_entity_ids').filter(pl.col('matched_entity_ids')!='').select(pl.col('source1_entity_id').alias('s1'),pl.col('matched_entity_ids').alias('m')).write_parquet('input_parquet/train_pairs.parquet')"`)

## 4. Where to improve (evidence so far)
* Validation: V3 test-like holdout 0.984 vs public LB 0.966 → remaining shift = test orphans are near-twins of
  businesses present in S1 → `configs/config_v5.yaml` (hard-orphan list `orphan/dropped_s1.parquet`, built from per-S1
  look-alike strength) is the next run.
* Features: add in `scripts/featx.py` (applied to existing feature files by the `augment` stage).
* Models: `configs/config_v3nn.yaml` retrains XGBoost to convergence (+0.0003) and adds a fusion NN (char-trigram text
  encoders + feature MLP + attention); variants are chosen on the validation fold only.
* Rules: no external data; model ≤8B params MIT/Apache; smaller candidate sets rank higher; never tune on the leaderboard.
