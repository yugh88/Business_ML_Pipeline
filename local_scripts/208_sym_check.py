"""Label-free check of the mix decoder's removals/additions: twins are +k only, true copies' number noise is symmetric.
Excess (+k) - (-k) among removed pairs = twin false merges removed; among V9 accepted = twins still accepted."""
import sys
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl
nt = dict(pl.read_parquet("work/test_s1.parquet", columns=["country"]).group_by("country").len().rows())
for c in sys.argv[1].split(","):
    sets = {"V9 accepted": pl.read_parquet(f"work/mix_acc_{c}_1.0.parquet"), "mix0.5 removed": pl.read_parquet(f"work/mix05_removed_{c}.parquet"),
            "mix0.5 added": pl.read_parquet(f"work/mix05_added_{c}.parquet")}
    ids = pl.concat([d.select(pl.col(k).alias("entity_id")) for d in sets.values() for k in ("s1", "m")]).unique()
    nm = pl.concat([pl.scan_parquet(f"work/test_s{s}_norm.parquet").select("entity_id", pl.col("nums").str.split(" ").list.first().alias("h")) for s in (1, 2, 3)]) \
           .join(ids.lazy(), on="entity_id", how="semi").collect()
    for name, d in sets.items():
        x = d.select("s1", "m").join(nm.rename({"entity_id": "s1", "h": "h1"}), on="s1").join(nm.rename({"entity_id": "m", "h": "h2"}), on="m")
        x = x.with_columns((pl.col("h2").cast(pl.Int64, strict=False) - pl.col("h1").cast(pl.Int64, strict=False)).alias("d"))
        n = nt[c] / 1000
        pk, mk = x.filter(pl.col("d").is_between(1, 9)).height / n, x.filter(pl.col("d").is_between(-9, -1)).height / n
        pk2, mk2 = x.filter(pl.col("d").is_between(10, 99)).height / n, x.filter(pl.col("d").is_between(-99, -10)).height / n
        print(f"{c:6s} {name:15s} {d.height / n:7.1f}/1k | +1..9 {pk:5.2f} -1..9 {mk:5.2f} excess {pk - mk:+5.2f} | +10..99 {pk2:5.2f} -10..99 {mk2:5.2f} excess {pk2 - mk2:+5.2f}")
