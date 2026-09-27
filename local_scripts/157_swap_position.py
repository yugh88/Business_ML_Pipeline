"""Fingerprint sharing by the POSITION of the swapped word (first word = brand vs later word = category), train vs France."""
import sys; sys.path.insert(0, "aws/scripts"); import memguard_local
import polars as pl
src = open("scripts/153_addr_fingerprint.py").read()
exec(src.split('tr = cands("train"')[0])
def pos(d):
    return d.with_columns(pl.struct("c1", "mw").map_elements(lambda r: (r["c1"].split().index(r["mw"]) if r["mw"] in r["c1"].split() else -1), return_dtype=pl.Int32).alias("ipos"),
                          pl.col("c1").str.split(" ").list.len().alias("nw1"))
tr = cands("train", pl.col("fold").is_in([0, 8, 9])).filter(pl.col("kind") == "SWAP")
tr = pos(fingerprint(tr, "train")) if False else pos(tr)
tr = fingerprint(tr, "train")
tr = tr.with_columns(pl.when(pl.col("ipos") == 0).then(pl.lit("first")).when(pl.col("ipos") == pl.col("nw1") - 1).then(pl.lit("last")).otherwise(pl.lit("mid")).alias("where"))
pl.Config.set_tbl_rows(30)
print("TRAIN SWAP, same number, non-empty address: truth rate and fingerprint by swapped-word position")
print(tr.filter(~pl.col("ra_empty") & (pl.col("off") == "same")).group_by("where", "y").agg(pl.len(), pl.col("fp_share").mean().round(3).alias("addr_shared")).sort("where", "y"))
print(tr.filter(~pl.col("ra_empty") & (pl.col("off") == "same")).group_by("where").agg(pl.len(), pl.col("y").mean().round(3).alias("true_rate")).sort("where"))
fr = pl.read_parquet("work/fr_fingerprint.parquet").filter(pl.col("kind") == "SWAP")
ids = fr.select(pl.col("s1").alias("entity_id")).unique()
c1 = pl.scan_parquet("work/test_s1_norm.parquet").select("entity_id", "ncore").join(ids.lazy(), on="entity_id", how="semi").collect().rename({"entity_id": "s1", "ncore": "c1"})
fr = pos(fr.join(c1, on="s1")).with_columns(pl.when(pl.col("ipos") == 0).then(pl.lit("first")).when(pl.col("ipos") == pl.col("nw1") - 1).then(pl.lit("last")).otherwise(pl.lit("mid")).alias("where"))
print("FRANCE SWAP, same number: fingerprint by class x position")
print(fr.filter(pl.col("off") == "same").group_by("cls", "where").agg(pl.len(), pl.col("fp_share").mean().round(3).alias("addr_shared"), (pl.col("p2x2") > 0.5).mean().round(2).alias("acc")).sort("cls", "where"))
