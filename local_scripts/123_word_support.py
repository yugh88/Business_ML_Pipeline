"""Extra-word 'support': how many OTHER candidates of the same S1 carry the same extra (non-S1) name word.
Copy noise is per-record (low support); a sibling entity's copies share its word (support>=1). Train truth + test."""
import sys; sys.path.insert(0, "aws/scripts"); import memguard_local
import polars as pl
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(220)
def extra_words(sp, cond):
    d = pl.scan_parquet(f"work/v5_p2/{sp}/*.parquet").filter(cond & (pl.col("p1") >= 0.02)).select("s1", "m", "y", "country", "p1", "p2x2").collect()
    ids1 = d.select(pl.col("s1").alias("entity_id")).unique(); ids2 = d.select(pl.col("m").alias("entity_id")).unique()
    n1 = pl.scan_parquet(f"work/{sp}_s1_norm.parquet").select("entity_id", "ncore").join(ids1.lazy(), on="entity_id", how="semi").collect()
    n2 = pl.concat([pl.scan_parquet(f"work/{sp}_s{s}_norm.parquet").select("entity_id", "ncore") for s in (2, 3)]).join(ids2.lazy(), on="entity_id", how="semi").collect()
    d = d.join(n1.rename({"entity_id": "s1", "ncore": "c1"}), on="s1").join(n2.rename({"entity_id": "m", "ncore": "c2"}), on="m")
    d = d.with_columns(pl.col("c2").str.split(" ").list.set_difference(pl.col("c1").str.split(" ")).alias("E"),
                       pl.col("c1").str.split(" ").list.set_difference(pl.col("c2").str.split(" ")).alias("M"))
    d = d.with_columns(pl.col("E").list.len().alias("nE"), pl.col("M").list.len().alias("nM"))
    one = d.filter(pl.col("nE") == 1).with_columns(pl.col("E").list.first().alias("w"))
    sup = one.group_by("s1", "w").len().rename({"len": "cnt"})
    one = one.join(sup, on=["s1", "w"]).with_columns((pl.col("cnt") - 1).clip(0, 3).alias("support"), pl.when(pl.col("nM") == 0).then(pl.lit("extra")).otherwise(pl.lit("swap")).alias("kind"))
    return one
tr = extra_words("train", pl.col("fold").is_in([0, 8, 9]))
rate = tr.group_by("w").agg(pl.len().alias("n"), pl.col("y").mean().alias("r")).filter(pl.col("n") >= 200)
print("TRAIN one-extra-word candidates: true rate by kind x support")
print(tr.group_by("kind", "support").agg(pl.len(), pl.col("y").mean().round(3).alias("true_rate")).sort("kind", "support"))
print("\nTRAIN by word-class (word true rate<0.2 = 'sibling word', >0.8 = 'noise word') x support")
tr2 = tr.join(rate.select("w", pl.when(pl.col("r") < 0.2).then(pl.lit("sib_word")).when(pl.col("r") > 0.8).then(pl.lit("noise_word")).otherwise(pl.lit("mixed")).alias("wc")), on="w", how="left").with_columns(pl.col("wc").fill_null("rare"))
print(tr2.group_by("kind", "wc", "support").agg(pl.len(), pl.col("y").mean().round(3).alias("true_rate")).sort("kind", "wc", "support"))
te = extra_words("test", pl.col("country") == "France")
print("\nTEST France one-extra-word candidates: accept rate (p2x2>0.5) by kind x support")
print(te.group_by("kind", "support").agg(pl.len(), (pl.col("p2x2") > 0.5).mean().round(3).alias("acc")).sort("kind", "support"))
top = te.group_by("w").agg(pl.len().alias("n"), pl.col("support").mean().round(2).alias("mean_sup"), (pl.col("support") >= 1).mean().round(2).alias("sup_share"), (pl.col("p2x2") > 0.5).mean().round(2).alias("acc")).sort("n", descending=True)
print("\nTEST France top extra words: mean support / share with support>=1 / accept rate")
print(top.head(50))
top.write_parquet("work/fr_extra_word_support.parquet")
tru = tr.group_by("w").agg(pl.len().alias("n"), pl.col("y").mean().round(3).alias("true_rate"), pl.col("support").mean().round(2).alias("mean_sup"), (pl.col("support") >= 1).mean().round(2).alias("sup_share")).sort("n", descending=True)
print("\nTRAIN top extra words: true rate / mean support / share with support>=1")
print(tru.head(40))
