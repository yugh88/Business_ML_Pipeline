import sys; sys.path.insert(0, "aws/scripts"); import memguard_local
import polars as pl
exec(open("scripts/125_word_addr2.py").read().split("tr = cands(")[0])
fr = cands("test", pl.col("country") == "France")
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(220)
for w in ["groupe", "france", "developpement", "fils", "associes", "club", "centre", "services", "sci", "cie"]:
    x = fr.filter(pl.col("w") == w)
    print(f"=== {w}: top (kind, missing word) with same/plus_k")
    print(x.group_by("kind", "mw").agg(pl.len(), (pl.col("off") == "same").mean().round(2).alias("same"), (pl.col("off") == "+1..9").mean().round(2).alias("plus_k"), (pl.col("p2x2") > 0.5).mean().round(2).alias("acc")).sort("len", descending=True).head(8).rows())
fr.select("s1", "m", "c1", "c2", "kind", "w", "mw", "off", "p2x2").write_parquet("work/fr_onew.parquet")
