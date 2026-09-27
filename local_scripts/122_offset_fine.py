"""Finer signed-offset bins + 'conflict' (S1's own number carried by another candidate). Train truth vs test preds."""
import sys, json; sys.path.insert(0, "aws/scripts"); import memguard_local
import polars as pl, duckdb
exec(open("scripts/119_offset.py").read().split("rel = pl.read_parquet")[0])
def fine(d):
    i1, i2 = pl.col("h1").cast(pl.Int64, strict=False), pl.col("h2").cast(pl.Int64, strict=False); dd = i2 - i1
    return d.with_columns(pl.when(pl.col("off").is_in(["empty", "same", "nonint"])).then(pl.col("off"))
        .when((dd >= 1) & (dd <= 9)).then(pl.lit("a:+1..9")).when((dd >= 10) & (dd <= 99)).then(pl.lit("b:+10..99")).when(dd >= 100).then(pl.lit("c:+100+"))
        .when((dd <= -1) & (dd >= -9)).then(pl.lit("d:-1..9")).when((dd <= -10) & (dd >= -99)).then(pl.lit("e:-10..99")).otherwise(pl.lit("f:-100+")).alias("fo"))
def conflict(d):   # another candidate (p1>=0.1) of the same S1 carries the S1's own number
    st = (pl.col("p1") >= 0.1) & (pl.col("h2") == pl.col("h1")) & (pl.col("h1") != "")
    return d.with_columns((st.cast(pl.Int32).sum().over("s1") - st.cast(pl.Int32) > 0).alias("conf"))
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(220)
sys.path.insert(0, "scripts"); from tune_decision import prep, decide
dec = json.load(open("analysis/aws_v5/decision_v5.json"))
d = pl.scan_parquet("work/v5_p2/train/*.parquet").filter(pl.col("fold").is_in([0, 8, 9]) & (pl.col("p1") >= 0.02)).select("s1", "m", "y", "fold", "country", "p1", "pc2", "pa", "p2x2").collect()
pr = decide(prep(d.drop("country"), dec["tau"], "p2x2"), dec).select("s1", "m").with_columns(pl.lit(True).alias("pred"))
d = conflict(fine(attach(d, "train"))).join(pr, on=["s1", "m"], how="left").with_columns(pl.col("pred").fill_null(False))
nS1 = {"US": 334293, "India": 222261}
t = d.group_by("country", "fo", "conf").agg(pl.len().alias("n"), pl.col("y").mean().round(3).alias("true_rate"), (pl.col("pred") & pl.col("y")).sum().alias("tp"), (pl.col("pred") & ~pl.col("y")).sum().alias("fp"))
t = t.with_columns((1000 * pl.col("tp") / pl.col("country").replace_strict(nS1, return_dtype=pl.Float64)).round(2).alias("tp_k"), (1000 * pl.col("fp") / pl.col("country").replace_strict(nS1, return_dtype=pl.Float64)).round(2).alias("fp_k"))
del d
p6 = con.execute("select source1_entity_id s1, unnest(string_split(matched_entity_ids, ',')) m from read_csv('output_v6/matching_results.tsv', delim='\t', header=true, all_varchar=true, quote='') where matched_entity_ids is not null and matched_entity_ids<>''").pl().with_columns(pl.lit(True).alias("pred"))
s1c = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"}); nS1t = dict(s1c.group_by("country").len().rows())
parts = []
for c in ["US", "India", "France"]:
    x = pl.scan_parquet("work/v5_p2/test/*.parquet").filter((pl.col("country") == c) & (pl.col("p1") >= 0.02)).select("s1", "m", "country", "p1", "p2x2").collect()
    x = conflict(fine(attach(x, "test"))).join(p6, on=["s1", "m"], how="left").with_columns(pl.col("pred").fill_null(False))
    parts.append(x.group_by("country", "fo", "conf").agg(pl.len().alias("n_te"), pl.col("pred").sum().alias("npred"))); del x
te = pl.concat(parts).with_columns((1000 * pl.col("npred") / pl.col("country").replace_strict(nS1t, return_dtype=pl.Float64)).round(2).alias("pred_k"))
j = te.join(t, on=["country", "fo", "conf"], how="left").with_columns((pl.col("pred_k") - pl.col("tp_k")).round(2).alias("excess_k")).sort("country", "fo", "conf")
print(j.select("country", "fo", "conf", "n_te", "n", "true_rate", "pred_k", "tp_k", "fp_k", "excess_k"))
j.write_parquet("work/offset_fine.parquet")
