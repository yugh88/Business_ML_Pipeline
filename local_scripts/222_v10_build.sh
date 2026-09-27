#!/bin/bash
# V10 build (after the full AWS run): fetch scores, tune decision on fold 7, base files, per-country stage-3 ensemble.
# Post-steps (mix-decoder check, French legal-form rule, validator, error budget) run afterwards with their own scripts.
set -uo pipefail
cd /Users/yughjuneja/Downloads/student_resource
B=<your-bucket>; P=s3://$B/entity-resolution-v10
for i in 1 2 3 4 5; do aws --region ap-south-1 s3 sync $P/p2/ work/v10_p2/ --only-show-errors && break || sleep 20; done
echo "p2 train $(ls work/v10_p2/train | wc -l) test $(ls work/v10_p2/test | wc -l)"
mkdir -p analysis/aws_v10
aws --region ap-south-1 s3 cp $P/logs/pipeline.log analysis/aws_v10/pipeline.log --only-show-errors || true
python -u scripts/155_v8_tune.py work/v10_p2 analysis/aws_v10/tuned 2>&1 | grep -v Warn > analysis/out/222_v10_tune.txt
python - <<'PY'
import json
r = json.load(open("analysis/aws_v10/tuned/tune_report.json")); p = r["pick"]
dec = {k: p[k] for k in ("tau", "variant", "policy")}; dec.update({k: p[k] for k in ("t", "margin", "a", "lam") if k in p}); dec["cap"] = p.get("cap", 0)
dec["why"] = f"V10 fold-7 selection (scripts/155): valid {p['f05']:.5f}"
json.dump(dec, open("analysis/aws_v10/decision_v10.json", "w")); print("decision:", dec)
PY
python aws/scripts/finalize_submission.py work/v10_p2 work output_v10base analysis/aws_v10/decision_v10.json 2>&1 | grep -v Warn | tail -1
python3 utils/validate_submission.py --matching output_v10base/matching_results.tsv --candidate output_v10base/candidate_pairs.tsv --test-dir dataset/test | tail -1
AUG=v10 python -u scripts/173c_stage3_country.py work/v10_p2 analysis/aws_v10/decision_v10.json output_v10base/candidate_pairs.tsv output_v10s3 2>&1 | grep -v Warn > analysis/out/222_v10_stage3.log
tail -12 analysis/out/222_v10_stage3.log
echo "V10 BUILD DONE"
