import polars as pl, sys, collections
sys.path.insert(0,"scripts")
from brand import is_brand_token
from norm import name_key
for split in ["train","test"]:
  for s in [1,2,3]:
    n=pl.read_parquet(f"work/{split}_s{s}.parquet",columns=["business_name","country"]).sample(200000,seed=1)
    flags=[any(is_brand_token(t) for t in name_key(x).split()) for x in n["business_name"].to_list()]
    n=n.with_columns(pl.Series("b",flags))
    ex=n.filter(pl.col("b"))["business_name"].head(6).to_list()
    print(split,s,n.group_by("country").agg(pl.col("b").mean()).sort("country").rows(), ex if s==1 else "")
