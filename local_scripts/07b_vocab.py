import polars as pl, sys, pickle, re
sys.path.insert(0,"scripts")
from norm import *
D=pickle.load(open("work/indic_dict.pkl","rb"))["name"]
def key(s):
    toks=[t for t in re.split(r"[\s\-,.()\[\]&/]+", base_clean(s)) if t]
    return " ".join(D.get(t,t) for t in toks)
pairs=pl.read_parquet("work/train_pairs.parquet")
def tok_table(df):
    k=[name_key(key(x)) for x in df["business_name"].to_list()]
    return df.with_columns(pl.Series("k",k)).with_columns(pl.col("k").str.split(" ").list.unique().alias("t")).explode("t").filter(pl.col("t")!="")
s1=tok_table(pl.read_parquet("work/train_s1.parquet").sample(700000,seed=0)).group_by("country","t").len().rename({"len":"n_s1"})
m=pl.concat([pl.read_parquet("work/train_s2.parquet"),pl.read_parquet("work/train_s3.parquet")]).sample(1500000,seed=0)
m=m.with_columns(pl.col("entity_id").is_in(pairs["m"].to_list()).alias("matched"))
mt=tok_table(m).group_by("country","t").agg(pl.len().alias("n_m"),pl.col("matched").sum().alias("n_matched"))
v=mt.join(s1,on=["country","t"],how="left").fill_null(0).with_columns((pl.col("n_matched")/pl.col("n_m")).alias("mrate"))
base=m.group_by("country").agg(pl.col("matched").mean()).rows(); print("base matched rate",base)
neg=v.filter((pl.col("n_m")>=150)&(pl.col("mrate")<0.05)).sort("n_m",descending=True)
print("tokens n>=150 with matched rate<5%:", neg.height, " records affected:", neg["n_m"].sum())
print(neg.head(60).select("country","t","n_m","mrate","n_s1").rows())
pos=v.filter((pl.col("n_m")>=300)&(pl.col("mrate")>0.97)).sort("n_m",descending=True)
print("tokens n>=300 with matched rate>97%:", pos.height); print(pos.head(30).select("country","t","n_m","mrate","n_s1").rows())
v.write_parquet("work/token_match_rates.parquet")
