"""Twin groups: candidates (p1>=0.02) of an S1 whose house number = S1's + k. Group size / sources, train vs test."""
import sys; sys.path.insert(0, "aws/scripts"); import memguard_local
import polars as pl
exec(open("scripts/119_offset.py").read().split("rel = pl.read_parquet")[0])
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(200)
res = {}
for sp, countries in [("test", ["US", "India", "France"])]:
    for c in countries:
        q = pl.scan_parquet(f"work/v5_p2/{sp}/*.parquet").filter((pl.col("country") == c) & (pl.col("p1") >= 0.02))
        if sp == "train": q = q.filter(pl.col("fold").is_in([0, 8, 9]))
        d = q.select("s1", "m", "y", "p2x2").collect()
        nS1 = d["s1"].n_unique()
        d = attach(d, sp).with_columns(pl.col("m").str.slice(0, 2).alias("src"))
        g = d.filter(pl.col("off").is_in(["+1..9", "-1..9"])).group_by("s1", "h2", "off").agg(pl.len().alias("size"), pl.col("src").n_unique().alias("nsrc"),
                                                                                         pl.col("y").all().alias("all_true") if sp == "train" else pl.lit(None, dtype=pl.Boolean).alias("all_true"))
        t = g.group_by("off", pl.col("size").clip(1, 4).alias("size"), "nsrc").agg(pl.len().alias("groups"), pl.col("all_true").cast(pl.Float64).mean().round(3).alias("true_share"))
        t = t.with_columns((1000 * pl.col("groups") / nS1).round(2).alias("groups_per_1kS1")).sort("off", "size", "nsrc")
        print(f"==== {sp} {c}  (S1 with cands: {nS1})"); print(t)
