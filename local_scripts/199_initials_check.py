"""Check 191's India-initials signal per country: holdout fingerprint rate for TRUE initials copies by country, vs test accepted.
Also densities per 1k S1 (holdout vs test) of same-address initials candidates and V9 acceptance."""
import sys
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl
exec(open("scripts/191_fr_precision.py").read().split("t1 = pl.read_parquet")[0])
t1 = pl.read_parquet("work/train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
h = pl.concat([pl.read_parquet(f"output_v9/p3/train/fold{f}.parquet") for f in (0, 8, 9)]).join(SYN, on="m", how="anti")
nh = h.join(t1, on="s1").group_by("country").agg(pl.col("s1").n_unique()).rows()
nh = {c: n / 1000 for c, n in nh}
hx = build(h, "train").join(t1, on="s1")
print("HOLDOUT same-address candidates by class (initials / identical) per country:")
print(hx.filter(pl.col("cls").is_in(["initials", "identical", "alias/no_overlap"])).group_by("country", "cls", "y").agg(
    pl.len().alias("n"), pl.col("acc").mean().round(3).alias("acc"), pl.col("fp_shared").mean().round(3).alias("shared")).sort("country", "cls", "y"))
print({c: {k: round(v / nh[c], 2) for k, v in hx.filter((pl.col("cls") == "initials") & (pl.col("country") == c)).group_by("y").len().rows()} for c in nh}, "<- initials cands per 1k holdout S1 by truth")
for c in ("US", "India", "France"):
    tx = build(pl.read_parquet(f"output_v9/p3/test/{c}.parquet"), "test")
    ns = tx["s1"].n_unique() / 1000
    z = tx.filter(pl.col("cls") == "initials")
    print(f"TEST {c}: initials cands {z.height/ns:.2f}/1k S1, acc {z['acc'].mean():.3f}, shared(acc) {z.filter('acc')['fp_shared'].mean():.3f}, identical shared(acc) {tx.filter((pl.col('cls')=='identical') & pl.col('acc'))['fp_shared'].mean():.3f}")
