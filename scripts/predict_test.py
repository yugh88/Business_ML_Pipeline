"""Stage predict_test: apply the FROZEN decision (validation/decision.json, chosen on train folds only) to test."""
import os, sys, json
import polars as pl
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import P, log, done, mark_done, s3_push, write_parquet_atomic
from evaluate import load_scored, apply_policy

def main():
    if done("predict_test"):
        return
    dec = json.load(open(P("validation/decision.json")))
    d = load_scored("test", dec["variant"])
    pred = apply_policy(d, dec["policy"], **dec["params"]).join(d.select("s1", "m", "p"), on=["s1", "m"])
    write_parquet_atomic(pred, "submissions/test_matches.parquet")
    log(f"predict_test: {pred.height} matches for {pred['s1'].n_unique()} S1 with {dec['variant']}/{dec['policy']} {dec['params']}")
    s3_push("submissions/test_matches.parquet"); mark_done("predict_test", dict(matches=pred.height))

if __name__ == "__main__":
    main()
