"""Label-free word classifier: siblings (descriptor/entity-type swaps) are generated WITH a house-number change (+k),
copy noise words keep the S1's number. Per extra word: share of candidates whose house number == S1's vs +1..9."""
import sys; sys.path.insert(0, "aws/scripts"); import memguard_local
import polars as pl
exec(open("scripts/119_offset.py").read().split("rel = pl.read_parquet")[0])
pl.Config.set_tbl_rows(120); pl.Config.set_tbl_width_chars(220)
def words(sp, cond):
    d = pl.scan_parquet(f"work/v5_p2/{sp}/*.parquet").filter(cond & (pl.col("p1") >= 0.02)).select("s1", "m", "y", "country", "p1", "p2x2").collect()
    ids1 = d.select(pl.col("s1").alias("entity_id")).unique(); ids2 = d.select(pl.col("m").alias("entity_id")).unique()
    n1 = pl.scan_parquet(f"work/{sp}_s1_norm.parquet").select("entity_id", "ncore").join(ids1.lazy(), on="entity_id", how="semi").collect()
    n2 = pl.concat([pl.scan_parquet(f"work/{sp}_s{s}_norm.parquet").select("entity_id", "ncore") for s in (2, 3)]).join(ids2.lazy(), on="entity_id", how="semi").collect()
    d = d.join(n1.rename({"entity_id": "s1", "ncore": "c1"}), on="s1").join(n2.rename({"entity_id": "m", "ncore": "c2"}), on="m")
    d = d.with_columns(pl.col("c2").str.split(" ").list.set_difference(pl.col("c1").str.split(" ")).alias("E"))
    d = d.filter(pl.col("E").list.len() == 1).with_columns(pl.col("E").list.first().alias("w"))
    d = attach(d, sp)
    return d.group_by("w").agg(pl.len().alias("n"), pl.col("y").cast(pl.Float64).mean().round(3).alias("true_rate"),
                               (pl.col("off") == "same").mean().round(3).alias("same"), (pl.col("off") == "+1..9").mean().round(3).alias("plus_k"),
                               (pl.col("off") == "-1..9").mean().round(3).alias("minus_k"), (pl.col("off") == "empty").mean().round(3).alias("empty"),
                               (pl.col("p2x2") > 0.5).mean().round(2).alias("acc")).filter(pl.col("n") >= 150)
tr = words("train", pl.lit(True))
tr.write_parquet("work/word_addr_train.parquet")
print("TRAIN (all folds) words n>=150: correlation of true_rate with same / plus_k")
print(tr.select(pl.corr("true_rate", "same").alias("corr_same"), pl.corr("true_rate", "plus_k").alias("corr_plusk")))
tr = tr.with_columns(pl.when(pl.col("true_rate") < 0.2).then(pl.lit("SIB")).when(pl.col("true_rate") > 0.8).then(pl.lit("NOISE")).otherwise(pl.lit("mixed")).alias("cls"))
print(tr.group_by("cls").agg(pl.len(), pl.col("same").mean().round(3), pl.col("plus_k").mean().round(3), pl.col("same").min().alias("same_min"), pl.col("same").max().alias("same_max"),
                             pl.col("plus_k").min().alias("pk_min"), pl.col("plus_k").max().alias("pk_max")))
print(tr.sort("n", descending=True).head(60))
te = words("test", pl.col("country") == "France")
te.write_parquet("work/word_addr_france.parquet")
print("\nFRANCE words n>=150 sorted by n")
print(te.sort("n", descending=True).head(110))
