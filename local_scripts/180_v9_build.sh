#!/bin/bash
# Lean V9 build: fetch V8 scores, tune, base files, stage-3 variants (multi / single / veto). Diagnostics run afterwards.
set -uo pipefail
cd /Users/yughjuneja/Downloads/student_resource
B=<your-bucket>; P=s3://$B/entity-resolution-v8
for i in 1 2 3 4 5; do aws --region ap-south-1 s3 sync $P/p2/ work/v8_p2/ --only-show-errors && break || sleep 20; done
echo "p2 train $(ls work/v8_p2/train | wc -l) test $(ls work/v8_p2/test | wc -l)"
mkdir -p analysis/aws_v8
aws --region ap-south-1 s3 cp $P/logs/pipeline.log analysis/aws_v8/pipeline.log --only-show-errors || true
.venv/bin/python -u scripts/155_v8_tune.py work/v8_p2 analysis/aws_v8/tuned 2>&1 | grep -v Warn > analysis/out/155_v8_tune.txt
.venv/bin/python - <<'PY'
import json
r = json.load(open("analysis/aws_v8/tuned/tune_report.json")); p = r["pick"]
dec = {k: p[k] for k in ("tau", "variant", "policy")}; dec.update({k: p[k] for k in ("t", "margin", "a", "lam") if k in p}); dec["cap"] = p.get("cap", 0)
dec["why"] = f"V8 fold-7 selection (scripts/155): valid {p['f05']:.5f}"
json.dump(dec, open("analysis/aws_v8/decision_v8.json", "w")); print("decision:", dec)
PY
.venv/bin/python aws/scripts/finalize_submission.py work/v8_p2 work output_v8 analysis/aws_v8/decision_v8.json 2>&1 | grep -v Warn | tail -1
python3 utils/validate_submission.py --matching output_v8/matching_results.tsv --candidate output_v8/candidate_pairs.tsv --test-dir dataset/test | tail -1
.venv/bin/python -u scripts/173_stage3_apply.py work/v8_p2 analysis/aws_v8/decision_v8.json output_v8/candidate_pairs.tsv output_v9 2>&1 | grep -v Warn > analysis/out/173_stage3_apply_v8.log
S3_MULTI=0 .venv/bin/python -u scripts/173_stage3_apply.py work/v8_p2 analysis/aws_v8/decision_v8.json output_v8/candidate_pairs.tsv output_v9b 2>&1 | grep -v Warn > analysis/out/173_stage3_apply_v8_nomulti.log
.venv/bin/python -u scripts/179_veto.py work/v8_p2 analysis/aws_v8/decision_v8.json output_v9 output_v8/matching_results.tsv output_v9veto 2>&1 | grep -v Warn > analysis/out/179_veto_v8.log
echo "V9 BUILD DONE"
