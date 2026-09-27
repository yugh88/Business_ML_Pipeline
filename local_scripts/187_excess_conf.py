"""Where are V9's remaining test errors, and are they CONFIDENT? Per country x category x confidence bin:
test predicted density vs the same model's holdout TRUE-positive density (clean view) -> excess (label-free FP estimate).
Also FN side: holdout true-copy density that V9 misses vs test 'missing' density cannot be measured -> use predicted totals."""
import sys, json
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl, duckdb
from tune_decision import prep, decide
src = open("scripts/110_relation_taxonomy.py").read(); exec(src.split("def best_pairs")[0]); exec(src[src.index("def name_rel"):src.index("def relate")])
d9 = json.load(open("output_v9/decision_s3.json"))
SYN = pl.concat([pl.read_parquet(f"work/aug_v8_{s}.parquet", columns=["entity_id"]) for s in ("s2", "s3")]).rename({"entity_id": "m"})
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(220)


def cat(d, split):
    ids1 = d.select(pl.col("s1").alias("entity_id")).unique(); ids2 = d.select(pl.col("m").alias("entity_id")).unique()
    n = pl.concat([pl.scan_parquet(f"work/{split}_s{s}_norm.parquet").select("entity_id", "ncore", "a", "nums") for s in (1, 2, 3)]).join(pl.concat([ids1, ids2]).lazy(), on="entity_id", how="semi").collect()
    N = {r[0]: r[1:] for r in n.iter_rows()}
    return d.with_columns(pl.Series("nrel", [name_rel(N[a][0], N[b][0]) for a, b in zip(d["s1"], d["m"])]),
                          pl.Series("arel", [addr_rel(N[a][1], N[b][1], N[a][2], N[b][2]) for a, b in zip(d["s1"], d["m"])]))


def simplify(d):
    return d.with_columns(pl.when(pl.col("nrel").is_in(["N2_extra_DESCRIPTOR", "N6_word_swap_DESC"])).then(pl.lit("desc")).when(pl.col("nrel") == "N0_identical").then(pl.lit("ident"))
                          .when(pl.col("nrel").is_in(["N1_extra_noise", "N4_dropped_words", "N5_typo_subst"])).then(pl.lit("light")).otherwise(pl.lit("heavy")).alias("n"),
                          pl.when(pl.col("arel").is_in(["A0_identical", "A1_same_num"])).then(pl.lit("same_addr")).when(pl.col("arel") == "A3_empty").then(pl.lit("empty"))
                          .when(pl.col("arel") == "A4_different").then(pl.lit("diff_addr")).otherwise(pl.lit("num_chg")).alias("a"),
                          pl.when(pl.col("p3") >= 0.95).then(pl.lit("conf")).otherwise(pl.lit("unsure")).alias("c"))


t1 = pl.read_parquet("work/train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
gt = duckdb.connect().execute("select source1_entity_id s1, hash(source1_entity_id || 'fold') % 10 fold from 'work/train_gt.parquet'").pl() \
       .join(pl.read_parquet("work/v5_dropped_s1.parquet").select("s1"), on="s1", how="anti").filter(pl.col("fold").is_in([0, 8, 9])).join(t1, on="s1")
nh = dict(gt.group_by("country").len().rows())
hp = []
for f in (0, 8, 9):
    p9 = pl.read_parquet(f"output_v9/p3/train/fold{f}.parquet").join(SYN, on="m", how="anti")
    hp.append(decide(prep(p9, d9["tau"], "p3"), d9).select("s1", "m").join(p9.select("s1", "m", "y", "p3"), on=["s1", "m"]))
h = simplify(cat(pl.concat(hp), "train").join(t1, on="s1"))
ref = h.group_by("country", "n", "a", "c").agg(pl.col("y").sum().alias("tp"), (~pl.col("y")).sum().alias("fp"))
ref = ref.with_columns((1000 * pl.col("tp") / pl.col("country").replace_strict(nh, return_dtype=pl.Float64)).alias("tp_k"),
                       (1000 * pl.col("fp") / pl.col("country").replace_strict(nh, return_dtype=pl.Float64)).alias("fp_k"))
s1c = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"}); nt = dict(s1c.group_by("country").len().rows())
res = []
for c in ("US", "India"):
    p3 = pl.read_parquet(f"output_v9/p3/test/{c}.parquet")
    pr = decide(prep(p3, d9["tau"], "p3"), d9).select("s1", "m").join(p3.select("s1", "m", "p3"), on=["s1", "m"])
    x = simplify(cat(pr, "test")).with_columns(pl.lit(c).alias("country"))
    t = x.group_by("country", "n", "a", "c").len().with_columns((1000 * pl.col("len") / nt[c]).alias("te_k"))
    j = t.join(ref.filter(pl.col("country") == c), on=["country", "n", "a", "c"], how="full", coalesce=True).fill_null(0).with_columns((pl.col("te_k") - pl.col("tp_k")).alias("excess_k"))
    res.append(j)
    tot = j.select(pl.col("te_k").sum(), pl.col("tp_k").sum(), pl.col("fp_k").sum()).row(0)
    print(f"==== {c}: test pred/1k {tot[0]:.1f}  holdout TP/1k {tot[1]:.1f}  holdout FP/1k {tot[2]:.1f}  => total excess {tot[0]-tot[1]:.1f}/1k")
    print("   excess by confidence:", j.group_by("c").agg(pl.col("excess_k").sum().round(1), pl.col("fp_k").sum().round(2)).rows())
    print(j.sort("excess_k", descending=True).select("n", "a", "c", pl.col("te_k").round(1), pl.col("tp_k").round(1), pl.col("fp_k").round(2), pl.col("excess_k").round(1)).head(10))
pl.concat(res).write_parquet("work/v9_excess_conf.parquet")
