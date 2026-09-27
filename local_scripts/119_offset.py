"""Signed house-number offset d = h2 - h1 (pool vs S1) — twin generator signature test (train truth, test predictions)."""
import sys, json; sys.path.insert(0, "aws/scripts"); import memguard_local
import polars as pl, duckdb
con = duckdb.connect()
def attach(d, split):
    ids1 = d.select(pl.col("s1").alias("entity_id")).unique(); ids2 = d.select(pl.col("m").alias("entity_id")).unique()
    n1 = pl.scan_parquet(f"work/{split}_s1_norm.parquet").select("entity_id", "nums").join(ids1.lazy(), on="entity_id", how="semi").collect()
    n2 = pl.concat([pl.scan_parquet(f"work/{split}_s{s}_norm.parquet").select("entity_id", "nums") for s in (2, 3)]).join(ids2.lazy(), on="entity_id", how="semi").collect()
    hn = lambda c: pl.col(c).str.split(" ").list.first().fill_null("")
    d = d.join(n1.select(pl.col("entity_id").alias("s1"), hn("nums").alias("h1")), on="s1").join(n2.select(pl.col("entity_id").alias("m"), hn("nums").alias("h2")), on="m")
    i1, i2 = pl.col("h1").cast(pl.Int64, strict=False), pl.col("h2").cast(pl.Int64, strict=False)
    dd = i2 - i1
    return d.with_columns(pl.when((pl.col("h1") == "") | (pl.col("h2") == "")).then(pl.lit("empty")).when(pl.col("h1") == pl.col("h2")).then(pl.lit("same"))
        .when(dd.is_null()).then(pl.lit("nonint"))
        .when((dd >= 1) & (dd <= 9)).then(pl.lit("+1..9")).when((dd <= -1) & (dd >= -9)).then(pl.lit("-1..9"))
        .when(dd.abs() <= 99).then(pl.lit("|d|10..99")).when(dd >= 100).then(pl.lit("+100")).otherwise(pl.lit("-100")).alias("off"))
rel = pl.read_parquet("work/rel_train.parquet", columns=["s1", "m", "truth", "nrel", "arel", "country_1"])
tr = attach(rel, "train")
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(200)
print("TRAIN: every pool record vs its best S1 — true-copy rate by signed offset (name identical / typo / all)")
for nm, f in [("N0_identical", pl.col("nrel") == "N0_identical"), ("N2/N6_DESC", pl.col("nrel").is_in(["N2_extra_DESCRIPTOR", "N6_word_swap_DESC"])), ("other names", ~pl.col("nrel").is_in(["N0_identical", "N2_extra_DESCRIPTOR", "N6_word_swap_DESC"]))]:
    t = tr.filter(f).group_by("off").agg(pl.len().alias("n"), (pl.col("truth") == "true_copy").mean().round(4).alias("true_rate"), (pl.col("truth") == "distractor").mean().round(4).alias("distr_rate")).sort("off")
    print(nm); print(t.rows())
tr.select("s1", "m", "truth", "nrel", "off").write_parquet("work/offset_train.parquet")

# ---- predicted pairs: train holdout (V5 decision) vs test (V6) by signed offset, per 1k S1
sys.path.insert(0, "scripts"); from tune_decision import prep, decide
dec = json.load(open("analysis/aws_v5/decision_v5.json"))
d = pl.scan_parquet("work/v5_p2/train/*.parquet").filter(pl.col("fold").is_in([0, 8, 9]) & (pl.col("p1") >= 0.02)).select("s1", "m", "y", "fold", "country", "p1", "pc2", "pa", "p2x2").collect()
pr = decide(prep(d.drop("country"), dec["tau"], "p2x2"), dec).select("s1", "m").join(d.select("s1", "m", "y", "country"), on=["s1", "m"])
pr = attach(pr, "train")
nS1 = {"US": 334293, "India": 222261}
a = pr.group_by("country", "off").agg((pl.col("y")).sum().alias("tp"), (~pl.col("y")).sum().alias("fp"))
a = a.with_columns((1000 * pl.col("tp") / pl.col("country").replace_strict(nS1, return_dtype=pl.Float64)).round(2).alias("tp_k"), (1000 * pl.col("fp") / pl.col("country").replace_strict(nS1, return_dtype=pl.Float64)).round(2).alias("fp_k"))
p6 = con.execute("select source1_entity_id s1, unnest(string_split(matched_entity_ids, ',')) m from read_csv('output_v6/matching_results.tsv', delim='\t', header=true, all_varchar=true, quote='') where matched_entity_ids is not null and matched_entity_ids<>''").pl()
s1c = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
nS1t = dict(s1c.group_by("country").len().rows())
te = attach(p6.join(s1c, on="s1"), "test").group_by("country", "off").len().rename({"len": "npred"})
te = te.with_columns((1000 * pl.col("npred") / pl.col("country").replace_strict(nS1t, return_dtype=pl.Float64)).round(2).alias("pred_k"))
j = te.join(a.select("country", "off", "tp_k", "fp_k"), on=["country", "off"], how="left").with_columns((pl.col("pred_k") - pl.col("tp_k")).round(2).alias("excess_k")).sort("country", "off")
print("\nPREDICTED pairs per 1k S1 by signed offset: test V6 (pred_k) vs train holdout V5 (tp_k, fp_k)")
print(j)
