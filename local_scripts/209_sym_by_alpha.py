"""Per country and alpha_d: pairs removed/added vs V9, split into +k / -k house-number offsets (1..9, 10..99) and the rest.
Twins are +k only; true copies' offsets are symmetric -> removed(+k) - removed(-k) = twin false merges removed,
2*removed(-k) ~= true number-noise copies lost. Label-free, independent of the simulator."""
import sys
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl
AD = [0.75, 0.6, 0.5, 0.4, 0.3]
nt = dict(pl.read_parquet("work/test_s1.parquet", columns=["country"]).group_by("country").len().rows())
for c in sys.argv[1].split(","):
    v9 = pl.read_parquet(f"work/mix_acc_{c}_1.0.parquet")
    alls = {ad: pl.read_parquet(f"work/mix_acc_{c}_{ad}.parquet") for ad in AD}
    ids = pl.concat([v9.select(pl.col("s1").alias("entity_id")), v9.select(pl.col("m").alias("entity_id"))] +
                    [d.select(pl.col(k).alias("entity_id")) for d in alls.values() for k in ("s1", "m")]).unique()
    nm = pl.concat([pl.scan_parquet(f"work/test_s{s}_norm.parquet").select("entity_id", pl.col("nums").str.split(" ").list.first().alias("h")) for s in (1, 2, 3)]) \
           .join(ids.lazy(), on="entity_id", how="semi").collect()
    n = nt[c] / 1000
    def split(x):
        x = x.join(nm.rename({"entity_id": "s1", "h": "h1"}), on="s1").join(nm.rename({"entity_id": "m", "h": "h2"}), on="m") \
             .with_columns((pl.col("h2").cast(pl.Int64, strict=False) - pl.col("h1").cast(pl.Int64, strict=False)).alias("d"))
        g = lambda a, b: x.filter(pl.col("d").is_between(a, b)).height / n
        return g(1, 9), g(-9, -1), g(10, 99), g(-99, -10), x.height / n
    for ad, d in alls.items():
        rem, add = split(v9.join(d, on=["s1", "m"], how="anti")), split(d.join(v9, on=["s1", "m"], how="anti"))
        twins = (rem[0] - rem[1]) + (rem[2] - rem[3]); lost_true_num = 2 * (rem[1] + rem[3])
        other = rem[4] - sum(rem[:4])
        print(f"{c:6s} alpha_d={ad:<4} removed {rem[4]:5.1f}/1k [+k {rem[0]:.2f} -k {rem[1]:.2f} | +10..99 {rem[2]:.2f} -10..99 {rem[3]:.2f} | other {other:.2f}]"
              f"  added {add[4]:4.1f}/1k [+k {add[0]:.2f} -k {add[1]:.2f}]  => twin FPs removed ~{twins:.2f}, true number-noise copies lost ~{lost_true_num:.2f}", flush=True)
