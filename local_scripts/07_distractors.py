import polars as pl, sys, re
sys.path.insert(0,"scripts")
from norm import *
from brand import is_brand_token
pairs=pl.read_parquet("work/train_pairs.parquet")
m=pl.concat([pl.read_parquet("work/train_s2.parquet").with_columns(pl.lit("S2").alias("src")),pl.read_parquet("work/train_s3.parquet").with_columns(pl.lit("S3").alias("src"))])
m=m.with_columns(pl.col("entity_id").is_in(pairs["m"].implicit() if False else pairs["m"].to_list()).alias("matched"))
print("matched rate by src,country:", m.group_by("src","country").agg(pl.col("matched").mean()).sort("src","country").rows())
words=["स्टोर्स","स्वीट्स","जनरल","मोटर्स","ज्वेलर्स","ट्रेडर्स"]
for w in words:
    x=m.filter(pl.col("business_name").str.contains(w))
    print(w, x.height, "matched frac", round(x["matched"].mean(),4) if x.height else None)
feat=m.sample(600000,seed=0).with_columns(
  pl.col("business_name").str.contains(r"[ऀ-෿]").alias("indic_name"),
  pl.col("business_address").str.contains(r"[ऀ-෿]").alias("indic_addr"),
  (pl.col("business_address")=="").alias("addr_empty"),
  pl.col("business_name").str.contains(r"(?i)\.(com|net|org|in)$").alias("domain"),
  pl.col("business_name").str.contains(r"^(>>|\*\*\*|--|\.\.\.)").alias("junk"),
  pl.col("business_name").str.contains(r"[À-ɏ]").alias("accent"),
  (pl.col("business_name").str.to_uppercase()==pl.col("business_name")).alias("upper"),
  pl.col("business_name").str.contains(r"(?i)\b(stores|sweets|general|motors|traders|jewellers|bakery|medicals)\b").alias("shopword"),
  pl.col("business_name").str.contains(r"  ").alias("dblspace"),
  pl.col("business_name").str.len_chars().alias("nlen"), pl.col("business_address").str.len_chars().alias("alen"),
  pl.col("business_address").str.count_matches(",").alias("ncomma"),
)
feat=feat.with_columns(pl.Series("brand",[any(is_brand_token(t) for t in name_key(x).split()) for x in feat["business_name"].to_list()]))
cols=["indic_name","indic_addr","addr_empty","domain","junk","accent","upper","shopword","dblspace","brand","nlen","alen","ncomma"]
print(feat.group_by("country","src","matched").agg([pl.col(c).mean().round(4) for c in cols]).sort("country","src","matched"))
# sample distractors
for c in ["US","India"]:
    print(c, "distractor samples:")
    for r in feat.filter((pl.col("country")==c)&(~pl.col("matched"))).sample(12,seed=3).iter_rows(named=True):
        print("  ",r["src"],"|",r["business_name"][:45],"|",r["business_address"][:70])
