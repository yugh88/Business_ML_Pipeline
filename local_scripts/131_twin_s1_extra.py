"""In S1s where a twin (+k, R1/R2) was flagged: how many OTHER predicted records are empty-number (e) / other-offset (o)?
Train holdout (with truth) vs test. Excess e/o in test twin-S1s = twin copies without / with perturbed numbers."""
import sys, json; sys.path.insert(0, "aws/scripts"); import memguard_local
import polars as pl, duckdb
exec(open("scripts/130_rules_v7.py").read().split("gt = duckdb")[0])
def summarize(pr, label, has_y):
    pr = pr.with_columns((RULES["R1_twin_same_src"](pr) | RULES["R2_twin_2src"](pr)).alias("tw"))
    s = pr.group_by("s1").agg(pl.col("tw").any().alias("twin_s1"))
    pr = pr.join(s, on="s1").filter(pl.col("twin_s1") & ~pl.col("tw"))
    k = pl.when(pl.col("off") == "same").then(pl.lit("S")).when(pl.col("off") == "empty").then(pl.lit("e")).when(pl.col("off") == "-1..9").then(pl.lit("M")).when(pl.col("off") == "+1..9").then(pl.lit("P")).otherwise(pl.lit("o"))
    agg = [pl.len().alias("n")] + ([pl.col("y").mean().round(3).alias("true_rate")] if has_y else [])
    t = pr.with_columns(k.alias("k")).group_by("k").agg(*agg).with_columns((pl.col("n") / s.filter(pl.col("twin_s1")).height).round(3).alias("per_twinS1"))
    print(label, "twin S1s:", s.filter(pl.col("twin_s1")).height); print(t.sort("k"))
d = pl.scan_parquet("work/v5_p2/train/*.parquet").filter(pl.col("fold").is_in([0, 8, 9])).select("s1", "m", "y", "fold", "country", "p1", "pc2", "pa", "p2x2").collect()
pr = decide(prep(d.drop("country"), dec["tau"], "p2x2"), dec).select("s1", "m").join(d.select("s1", "m", "y", "country"), on=["s1", "m"]); del d
summarize(enrich(pr, "train"), "TRAIN holdout", True)
p5 = duckdb.connect().execute("select source1_entity_id s1, unnest(string_split(matched_entity_ids, ',')) m from read_csv('output_v5/matching_results.tsv', delim='\t', header=true, all_varchar=true, quote='') where matched_entity_ids is not null and matched_entity_ids<>''").pl()
s1c = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
te = enrich(p5.join(s1c, on="s1"), "test")
for c in ["US", "India", "France"]:
    summarize(te.filter(pl.col("country") == c), f"TEST {c}", False)
