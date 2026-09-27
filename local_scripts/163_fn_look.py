"""Eyeball non-empty-address true copies that V5 scored but rejected (largest FN bucket)."""
import sys, json; sys.path.insert(0, "aws/scripts"); import memguard_local
import polars as pl
from tune_decision import prep, decide
dec = json.load(open("analysis/aws_v5/decision_v5.json"))
d = pl.scan_parquet("work/v5_p2/train/*.parquet").filter(pl.col("fold") == 0).select("s1", "m", "y", "fold", "p1", "pc2", "pa", "p2x2", "a2_empty").collect()
pr = decide(prep(d, dec["tau"], "p2x2"), dec).with_columns(pl.lit(True).alias("pred"))
x = d.join(pr, on=["s1", "m"], how="left").with_columns(pl.col("pred").fill_null(False))
fn = x.filter(pl.col("y") & ~pl.col("pred") & (pl.col("p1") >= dec["tau"]) & ~pl.col("a2_empty"))
# why rejected: another S1 took this record (competition) vs low p2 vs S1 cap/expected-F cut
mm = d.group_by("m").agg(pl.col("p2x2").max().alias("mmax"))
fn = fn.join(mm, on="m").with_columns(pl.when(pl.col("p2x2") < pl.col("mmax") - 1e-9).then(pl.lit("lost_to_other_S1")).when(pl.col("p2x2") < 0.5).then(pl.lit("p2<0.5")).otherwise(pl.lit("policy_cut")).alias("why"))
print("scored+rejected non-empty FNs:", fn.height, fn.group_by("why").len().sort("len", descending=True).rows())
print("p2x2 quantiles:", [round(float(fn["p2x2"].quantile(q)), 3) for q in (0.1, 0.25, 0.5, 0.75, 0.9)])
s = fn.sample(14, seed=3)
ids = pl.concat([x.filter(pl.col("s1").is_in(s["s1"])).select(pl.col("s1").alias("entity_id")), x.filter(pl.col("s1").is_in(s["s1"])).select(pl.col("m").alias("entity_id"))]).unique()
raw = pl.concat([pl.scan_parquet(f"work/train_s{k}.parquet") for k in (1, 2, 3)]).join(ids.lazy(), on="entity_id", how="semi").collect()
R = {r[0]: (r[1], r[2]) for r in raw.iter_rows()}
for s1, mm_, why in s.select("s1", "m", "why").iter_rows():
    print("=" * 140); print(f"{s1} | {R[s1][0]} | {R[s1][1]}   [missed {mm_}: {why}]")
    for r in x.filter(pl.col("s1") == s1).sort("p2x2", descending=True).head(9).iter_rows(named=True):
        tag = ("PRED " if r["pred"] else "     ") + ("TRUE" if r["y"] else "    ")
        print(f"   {tag} {r['p2x2']:.3f} {r['m']:13s} | {R[r['m']][0]} | {R[r['m']][1]}{'  <== missed' if r['m'] == mm_ else ''}")
