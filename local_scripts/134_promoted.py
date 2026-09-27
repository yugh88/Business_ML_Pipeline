"""'Collectively promoted' records: accepted by the p2x2 policy but NOT by the pairwise p1 policy.
Train holdout TP/FP rate of promoted records vs their density in test, by house-offset kind."""
import sys, json; sys.path.insert(0, "aws/scripts"); import memguard_local
import polars as pl, numpy as np, duckdb
exec(open("scripts/119_offset.py").read().split("rel = pl.read_parquet")[0])
from tune_decision import prep, decide
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(220)
cp2 = json.load(open("analysis/aws_v5/decision_v5.json")); cp1 = dict(policy="G_expected_f", a=1.25, lam=0.0, cap=1)
def promoted(d, sp):
    a2 = decide(prep(d, 0.05, "p2x2"), cp2).with_columns(pl.lit(True).alias("acc2"))
    a1 = decide(prep(d, 0.05, "p1"), cp1).with_columns(pl.lit(True).alias("acc1"))
    x = a2.join(a1, on=["s1", "m"], how="left").with_columns(pl.col("acc1").fill_null(False))
    return attach(x, sp)
d = pl.scan_parquet("work/v5_p2/train/*.parquet").filter(pl.col("fold").is_in([0, 8, 9])).select("s1", "m", "y", "fold", "country", "p1", "pc2", "pa", "p2x2").collect()
x = promoted(d.drop("country"), "train").join(d.select("s1", "m", "y", "country"), on=["s1", "m"]); del d
nS1 = {"US": 334293, "India": 222261}
tr = x.group_by("country", "acc1", "off").agg(pl.len().alias("n"), pl.col("y").mean().round(3).alias("true_rate")).with_columns(
    (1000 * pl.col("n") / pl.col("country").replace_strict(nS1, return_dtype=pl.Float64)).round(2).alias("per1k_tr"))
s1c = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"}); nS1t = dict(s1c.group_by("country").len().rows())
parts = []
for c in ["US", "India", "France"]:
    d = pl.scan_parquet("work/v5_p2/test/*.parquet").filter(pl.col("country") == c).select("s1", "m", "p1", "pc2", "pa", "p2x2").collect()
    parts.append(promoted(d, "test").with_columns(pl.lit(c).alias("country")).group_by("country", "acc1", "off").len().rename({"len": "n_te"})); del d
te = pl.concat(parts).with_columns((1000 * pl.col("n_te") / pl.col("country").replace_strict(nS1t, return_dtype=pl.Float64)).round(2).alias("per1k_te"))
j = te.join(tr, on=["country", "acc1", "off"], how="left").filter(~pl.col("acc1")).sort("country", "off")
print("PROMOTED records (p2x2 accepts, p1 rejects): train true rate & density vs test density per 1k S1")
print(j.select("country", "off", "n", "true_rate", "per1k_tr", "n_te", "per1k_te"))
print("\nTOTAL promoted per 1k S1:", j.group_by("country").agg(pl.col("per1k_tr").sum(), pl.col("per1k_te").sum()).rows())
