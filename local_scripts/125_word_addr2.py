"""Same as 124 but split by EXTRA (pure addition) vs SWAP (one word replaced); train classes by word true rate."""
import sys; sys.path.insert(0, "aws/scripts"); import memguard_local
import polars as pl
exec(open("scripts/119_offset.py").read().split("rel = pl.read_parquet")[0])
pl.Config.set_tbl_rows(120); pl.Config.set_tbl_width_chars(220)
def cands(sp, cond):
    d = pl.scan_parquet(f"work/v5_p2/{sp}/*.parquet").filter(cond & (pl.col("p1") >= 0.02)).select("s1", "m", "y", "country", "p1", "p2x2").collect()
    ids1 = d.select(pl.col("s1").alias("entity_id")).unique(); ids2 = d.select(pl.col("m").alias("entity_id")).unique()
    n1 = pl.scan_parquet(f"work/{sp}_s1_norm.parquet").select("entity_id", "ncore").join(ids1.lazy(), on="entity_id", how="semi").collect()
    n2 = pl.concat([pl.scan_parquet(f"work/{sp}_s{s}_norm.parquet").select("entity_id", "ncore") for s in (2, 3)]).join(ids2.lazy(), on="entity_id", how="semi").collect()
    d = d.join(n1.rename({"entity_id": "s1", "ncore": "c1"}), on="s1").join(n2.rename({"entity_id": "m", "ncore": "c2"}), on="m")
    d = d.with_columns(pl.col("c2").str.split(" ").list.set_difference(pl.col("c1").str.split(" ")).alias("E"), pl.col("c1").str.split(" ").list.set_difference(pl.col("c2").str.split(" ")).alias("M"))
    d = d.filter(pl.col("E").list.len() == 1).with_columns(pl.col("E").list.first().alias("w"), pl.when(pl.col("M").list.len() == 0).then(pl.lit("EXTRA")).otherwise(pl.lit("SWAP")).alias("kind"),
                                                           pl.col("M").list.first().alias("mw"))
    return attach(d, sp)
tr = cands("train", pl.col("fold").is_in([0, 1, 2, 3, 8, 9]))
wr = tr.group_by("w").agg(pl.len().alias("nw"), pl.col("y").mean().alias("wr")).filter(pl.col("nw") >= 150)
tr = tr.join(wr, on="w", how="left").with_columns(pl.when(pl.col("wr").is_null()).then(pl.lit("rare")).when(pl.col("wr") < 0.2).then(pl.lit("SIB")).when(pl.col("wr") > 0.8).then(pl.lit("NOISE")).otherwise(pl.lit("mixed")).alias("cls"))
print("TRAIN: address behaviour by kind x word class (and true rate)")
print(tr.group_by("kind", "cls").agg(pl.len(), pl.col("y").mean().round(3).alias("true"), (pl.col("off") == "same").mean().round(3).alias("same"), (pl.col("off") == "+1..9").mean().round(3).alias("plus_k"),
                                      (pl.col("off") == "empty").mean().round(3).alias("empty")).sort("kind", "cls"))
print("\nTRAIN: address behaviour by kind x cls x truth")
print(tr.filter(pl.col("cls").is_in(["SIB", "NOISE"])).group_by("kind", "cls", "y").agg(pl.len(), (pl.col("off") == "same").mean().round(3).alias("same"), (pl.col("off") == "+1..9").mean().round(3).alias("plus_k")).sort("kind", "cls", "y"))
print("\nTRAIN SWAP with SIB word: top (missing -> extra)")
print(tr.filter((pl.col("kind") == "SWAP") & (pl.col("cls") == "SIB")).group_by("mw", "w").agg(pl.len(), pl.col("y").mean().round(3).alias("true"), (pl.col("off") == "same").mean().round(2).alias("same")).sort("len", descending=True).head(25))
fr = cands("test", pl.col("country") == "France")
print("\nFRANCE: address behaviour by kind for top words")
t = fr.group_by("kind", "w").agg(pl.len(), (pl.col("off") == "same").mean().round(3).alias("same"), (pl.col("off") == "+1..9").mean().round(3).alias("plus_k"), (pl.col("off") == "empty").mean().round(3).alias("empty"), (pl.col("p2x2") > 0.5).mean().round(2).alias("acc")).filter(pl.col("len") >= 300).sort("len", descending=True)
print(t.head(80))
t.write_parquet("work/fr_word_kind_addr.parquet")
