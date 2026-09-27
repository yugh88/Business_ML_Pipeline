"""Final comparison of submission candidates on label-free test checks, relative to the LB anchor V5s3r (0.973).
For each candidate: pred/S1, twin excess (+k minus -k, per 1k S1), promoted (accepted but rejected by V5's pairwise p1 policy),
and pair changes vs V5s3r by category (house offset x name class), per 1k S1.  usage: 181_compare.py <out_dir> [<out_dir> ...]"""
import sys
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl, duckdb
from tune_decision import prep, decide
src = open("scripts/110_relation_taxonomy.py").read()
exec(src.split("def best_pairs")[0]); exec(src[src.index("def name_rel"):src.index("def relate")])
con = duckdb.connect()
load = lambda p: con.execute(f"select source1_entity_id s1, unnest(string_split(matched_entity_ids, ',')) m from read_csv('{p}/matching_results.tsv', delim='\t', header=true, "
                             f"all_varchar=true, quote='') where matched_entity_ids is not null and matched_entity_ids<>''").pl()
s1c = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"}); nS1 = dict(s1c.group_by("country").len().rows())
N = sum(nS1.values()) / 1000
runs = {d: load(d).join(s1c, on="s1") for d in ["output_v5s3r"] + sys.argv[1:]}
allp = pl.concat([r.select("s1", "m") for r in runs.values()]).unique()
ids = pl.concat([allp.select(pl.col("s1").alias("entity_id")), allp.select(pl.col("m").alias("entity_id"))]).unique()
nm = pl.concat([pl.scan_parquet(f"work/test_s{s}_norm.parquet").select("entity_id", "ncore", "nums") for s in (1, 2, 3)]).join(ids.lazy(), on="entity_id", how="semi").collect()
cat = allp.join(nm.rename({"entity_id": "s1", "ncore": "c1", "nums": "n1"}), on="s1").join(nm.rename({"entity_id": "m", "ncore": "c2", "nums": "n2"}), on="m")
h1, h2 = pl.col("n1").str.split(" ").list.first().fill_null(""), pl.col("n2").str.split(" ").list.first().fill_null("")
dd = h2.cast(pl.Int64, strict=False) - h1.cast(pl.Int64, strict=False)
cat = cat.with_columns(pl.when((h1 == "") | (h2 == "")).then(pl.lit("empty")).when(h1 == h2).then(pl.lit("same")).when((dd >= 1) & (dd <= 9)).then(pl.lit("+k"))
                       .when((dd <= -1) & (dd >= -9)).then(pl.lit("-k")).otherwise(pl.lit("other")).alias("off"),
                       pl.Series("nrel", [name_rel(x, y) for x, y in zip(cat["c1"].to_list(), cat["c2"].to_list())])).select("s1", "m", "off", "nrel")
cat = cat.with_columns(pl.when(pl.col("nrel").is_in(["N2_extra_DESCRIPTOR", "N6_word_swap_DESC"])).then(pl.lit("DESC")).when(pl.col("nrel") == "N0_identical").then(pl.lit("N0"))
                       .otherwise(pl.lit("other_name")).alias("ncls"))
# reference pairwise decision (V5 p1) for "promoted"
a1 = []
for c in ["US", "India", "France"]:
    t = pl.scan_parquet("work/v5_p2/test/*.parquet").filter((pl.col("country") == c) & (pl.col("p1") >= 0.05)).select("s1", "m", "p1", "pc2", "pa").collect()
    a1.append(decide(prep(t, 0.05, "p1"), dict(policy="G_expected_f", a=1.25, lam=0.0, cap=1)).select("s1", "m"))
a1 = pl.concat(a1).with_columns(pl.lit(True).alias("a1"))
base = runs["output_v5s3r"].select("s1", "m").with_columns(pl.lit(True).alias("b"))
for name, r in runs.items():
    x = r.join(cat, on=["s1", "m"], how="left").join(a1, on=["s1", "m"], how="left")
    line = f"{name:18s} pairs/S1 {r.height/(N*1000):.4f}"
    for c in ["US", "India", "France"]:
        xc = x.filter(pl.col("country") == c); n = nS1[c] / 1000
        pk, mk = xc.filter(pl.col("off") == "+k").height / n, xc.filter(pl.col("off") == "-k").height / n
        line += f" | {c} twin {pk - mk:5.1f} prom {xc['a1'].is_null().sum() / n:6.1f}"
    print(line)
    if name != "output_v5s3r":
        dfx = r.select("s1", "m").with_columns(pl.lit(True).alias("c")).join(base, on=["s1", "m"], how="full", coalesce=True) \
               .with_columns(pl.col("c").fill_null(False), pl.col("b").fill_null(False)).filter(pl.col("c") != pl.col("b")).join(cat, on=["s1", "m"], how="left")
        t = dfx.group_by("off", "ncls").agg(((pl.col("b") & ~pl.col("c")).sum() / N).round(2).alias("removed_k"), ((pl.col("c") & ~pl.col("b")).sum() / N).round(2).alias("added_k")) \
               .with_columns((pl.col("removed_k") + pl.col("added_k")).alias("_t")).sort("_t", descending=True).drop("_t")
        print("   vs V5s3r (removed/added per 1k):", t.head(9).rows(), " total removed", round(float(t["removed_k"].sum()), 1), "added", round(float(t["added_k"].sum()), 1))
