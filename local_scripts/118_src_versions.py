"""Per-source house-number version signals (analysis/117: a copy's changed number is a per-SOURCE version; shared by
both sources only 6.6%). For every candidate with a changed number: is that number also carried by candidates of the
OTHER source (twin signature)? does its own source also carry the S1's number (version conflict)?"""
import sys, json; sys.path.insert(0, "aws/scripts"); import memguard_local
import polars as pl, duckdb
sys.path.insert(0, "scripts"); from tune_decision import prep, decide
dec = json.load(open("analysis/aws_v5/decision_v5.json"))
con = duckdb.connect()

def signals(d, split):
    ids1 = d.select(pl.col("s1").alias("entity_id")).unique(); ids2 = d.select(pl.col("m").alias("entity_id")).unique()
    n1 = pl.scan_parquet(f"work/{split}_s1_norm.parquet").select("entity_id", "nums").join(ids1.lazy(), on="entity_id", how="semi").collect()
    n2 = pl.concat([pl.scan_parquet(f"work/{split}_s{s}_norm.parquet").select("entity_id", "nums") for s in (2, 3)]).join(ids2.lazy(), on="entity_id", how="semi").collect()
    hn = lambda c: pl.col(c).str.split(" ").list.first().fill_null("")
    d = d.join(n1.select(pl.col("entity_id").alias("s1"), hn("nums").alias("h1")), on="s1").join(n2.select(pl.col("entity_id").alias("m"), hn("nums").alias("h2")), on="m")
    d = d.with_columns(pl.col("m").str.slice(0, 2).alias("src"))
    st = (pl.col("p1") >= 0.1) & (pl.col("h2") != "")
    d = d.with_columns(
        (st.cast(pl.Int32).sum().over(["s1", "src", "h2"]) - st.cast(pl.Int32)).alias("same_src_same_h"),
        (st.cast(pl.Int32).sum().over(["s1", "h2"]) - st.cast(pl.Int32).sum().over(["s1", "src", "h2"])).alias("oth_src_same_h"),
        (st & (pl.col("h2") == pl.col("h1"))).cast(pl.Int32).sum().over(["s1", "src"]).alias("same_src_s1h"),
        (st & (pl.col("h2") == pl.col("h1"))).cast(pl.Int32).sum().over(["s1"]).alias("all_s1h"))
    d = d.with_columns((pl.col("all_s1h") - pl.col("same_src_s1h")).alias("oth_src_s1h"))
    return d.with_columns(pl.when(pl.col("h2") == "").then(pl.lit("h_empty")).when(pl.col("h1") == "").then(pl.lit("s1_noh")).when(pl.col("h2") == pl.col("h1")).then(pl.lit("h_same"))
        .otherwise(pl.concat_str(pl.lit("CHG|oth="), (pl.col("oth_src_same_h") > 0).cast(pl.Utf8), pl.lit("|ownS1h="), (pl.col("same_src_s1h") > 0).cast(pl.Utf8),
                                 pl.lit("|sib="), (pl.col("same_src_same_h") > 0).cast(pl.Utf8))).alias("sig"))

# TRAIN holdout
d = pl.scan_parquet("work/v5_p2/train/*.parquet").filter(pl.col("fold").is_in([0, 8, 9]) & (pl.col("p1") >= 0.02)).select("s1", "m", "y", "fold", "country", "p1", "pc2", "pa", "p2x2").collect()
pred = decide(prep(d.drop("country"), dec["tau"], "p2x2"), dec).select("s1", "m").with_columns(pl.lit(True).alias("pred"))
d = signals(d, "train").join(pred, on=["s1", "m"], how="left").with_columns(pl.col("pred").fill_null(False))
nS1 = {"US": 334293, "India": 222261}
print("TRAIN holdout candidates (p1>=0.02) by signal: n, true rate, predicted, TP, FP per 1k S1")
t = d.group_by("country", "sig").agg(pl.len().alias("n"), pl.col("y").mean().round(3).alias("true_rate"), pl.col("pred").sum().alias("npred"),
                                      (pl.col("pred") & pl.col("y")).sum().alias("tp"), (pl.col("pred") & ~pl.col("y")).sum().alias("fp"))
t = t.with_columns((1000 * pl.col("tp") / pl.col("country").replace_strict(nS1, return_dtype=pl.Float64)).round(2).alias("tp_k"),
                   (1000 * pl.col("fp") / pl.col("country").replace_strict(nS1, return_dtype=pl.Float64)).round(2).alias("fp_k"),
                   (1000 * pl.col("n") / pl.col("country").replace_strict(nS1, return_dtype=pl.Float64)).round(1).alias("cand_k"))
t.write_parquet("work/src_sig_train.parquet")
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(220)
print(t.sort("country", "sig"))
del d
# TEST
p6 = con.execute("select source1_entity_id s1, unnest(string_split(matched_entity_ids, ',')) m from read_csv('output_v6/matching_results.tsv', delim='\t', header=true, all_varchar=true, quote='') where matched_entity_ids is not null and matched_entity_ids<>''").pl().with_columns(pl.lit(True).alias("pred"))
s1c = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
nS1t = dict(s1c.group_by("country").len().rows())
parts = []
for c in ["US", "India", "France"]:
    d = pl.scan_parquet("work/v5_p2/test/*.parquet").filter((pl.col("country") == c) & (pl.col("p1") >= 0.02)).select("s1", "m", "country", "p1", "p2x2").collect()
    d = signals(d, "test").join(p6, on=["s1", "m"], how="left").with_columns(pl.col("pred").fill_null(False))
    parts.append(d.group_by("country", "sig").agg(pl.len().alias("n"), pl.col("pred").sum().alias("npred"), pl.col("p2x2").mean().round(3).alias("mean_p")))
    del d; print("  test", c, flush=True)
te = pl.concat(parts).with_columns((1000 * pl.col("npred") / pl.col("country").replace_strict(nS1t, return_dtype=pl.Float64)).round(2).alias("pred_k"),
                                   (1000 * pl.col("n") / pl.col("country").replace_strict(nS1t, return_dtype=pl.Float64)).round(1).alias("cand_k"))
te.write_parquet("work/src_sig_test.parquet")
j = te.join(t.select("country", "sig", "true_rate", "tp_k", "fp_k", pl.col("cand_k").alias("cand_k_tr")), on=["country", "sig"], how="left")
print("\nTEST (V6 predictions) vs TRAIN holdout, per 1k S1")
print(j.with_columns((pl.col("pred_k") - pl.col("tp_k")).round(2).alias("excess_k")).sort("country", "sig"))
