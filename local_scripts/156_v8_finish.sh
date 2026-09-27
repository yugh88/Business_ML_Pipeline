#!/bin/bash
# After the V8 AWS run: fetch scores, tune, build submission, apply France sibling rule, diagnostics. Idempotent.
set -euo pipefail
cd /Users/yughjuneja/Downloads/student_resource
B=<your-bucket>; P=s3://$B/entity-resolution-v8
for i in 1 2 3 4 5; do aws --region ap-south-1 s3 sync $P/p2/ work/v8_p2/ --only-show-errors && break || sleep 20; done
echo "p2 train buckets: $(ls work/v8_p2/train | wc -l)  test: $(ls work/v8_p2/test | wc -l)"
mkdir -p analysis/aws_v8
aws --region ap-south-1 s3 cp $P/logs/pipeline.log analysis/aws_v8/pipeline.log --only-show-errors || true
aws --region ap-south-1 s3 cp $P/models/ analysis/aws_v8/models/ --recursive --exclude "*.txt" --exclude "*.json.gz" --only-show-errors || true
.venv/bin/python -u scripts/155_v8_tune.py work/v8_p2 analysis/aws_v8/tuned 2>&1 | grep -v Warn | tee analysis/out/155_v8_tune.txt
.venv/bin/python - <<'PY'
import json
r = json.load(open("analysis/aws_v8/tuned/tune_report.json")); p = r["pick"]
dec = {k: p[k] for k in ("tau", "variant", "policy")}; dec.update({k: p[k] for k in ("t", "margin", "a", "lam") if k in p}); dec["cap"] = p.get("cap", 0)
dec["why"] = f"V8 fold-7 selection (scripts/155): valid {p['f05']:.5f}"
json.dump(dec, open("analysis/aws_v8/decision_v8.json", "w")); print("decision:", dec)
PY
.venv/bin/python aws/scripts/finalize_submission.py work/v8_p2 work output_v8 analysis/aws_v8/decision_v8.json 2>&1 | grep -v Warn | tail -2
python3 utils/validate_submission.py --matching output_v8/matching_results.tsv --candidate output_v8/candidate_pairs.tsv --test-dir dataset/test | tail -1
.venv/bin/python -u scripts/151_v8_eval.py work/v8_p2 analysis/aws_v8/decision_v8.json output_v8/matching_results.tsv v8 2>&1 | grep -v Warn | tee analysis/out/151_v8.txt
.venv/bin/python -u scripts/170_diag_suite.py v8 work/v8_p2 analysis/aws_v8/decision_v8.json 2>&1 | grep -v Warn > analysis/out/170_run_v8.log
.venv/bin/python -u scripts/173_stage3_apply.py work/v8_p2 analysis/aws_v8/decision_v8.json output_v8/candidate_pairs.tsv output_v9 2>&1 | grep -v Warn > analysis/out/173_stage3_apply_v8.log
S3_MULTI=0 .venv/bin/python -u scripts/173_stage3_apply.py work/v8_p2 analysis/aws_v8/decision_v8.json output_v8/candidate_pairs.tsv output_v9b 2>&1 | grep -v Warn > analysis/out/173_stage3_apply_v8_nomulti.log
.venv/bin/python -u scripts/158_rules_on.py work/v8_p2 analysis/aws_v8/decision_v8.json output_v8/matching_results.tsv output_v8x 2>&1 | grep -v Warn > analysis/out/158_rules_on_v8.log
echo "V8 FINISH DONE"
.venv/bin/python -u scripts/170_diag_suite.py v9 output_v9/p3 output_v9/decision_s3.json 2>&1 | grep -v Warn > analysis/out/170_run_v9.log
echo "V9 DIAG DONE"
.venv/bin/python -u scripts/179_veto.py work/v8_p2 analysis/aws_v8/decision_v8.json output_v9 output_v8/matching_results.tsv output_v9veto 2>&1 | grep -v Warn > analysis/out/179_veto_v8.log
.venv/bin/python -u scripts/176_xsource.py work/v8_p2 analysis/aws_v8/decision_v8.json work/s3cache_xs_v8 v8 2>&1 | grep -v Warn > analysis/out/176_xsource_v8.log
echo "XS ON V8 DONE"
