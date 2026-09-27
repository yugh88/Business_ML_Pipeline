import polars as pl, numpy as np, sys, time, json
sys.path.insert(0,"scripts")
from retrieval import build, topk
NS1=int(sys.argv[1]) if len(sys.argv)>1 else 100000
VIEWS=json.loads(sys.argv[2]) if len(sys.argv)>2 else None
KMAX=100
s1=pl.read_parquet("work/train_s1_norm.parquet").sample(NS1,seed=11)
pool=pl.concat([pl.read_parquet("work/train_s2_norm.parquet"),pl.read_parquet("work/train_s3_norm.parquet")])
for d in ("s1","pool"):
    x=locals()[d]
    locals()[d]  # noqa
s1=s1.with_columns((pl.col("n")+" , "+pl.col("a")).alias("na"))
pool=pool.with_columns((pl.col("n")+" , "+pl.col("a")).alias("na"))
views=VIEWS or [["n","word",5000],["ncore","word",5000],["a","word",5000],["na","word",5000],["n","char4",20000]]
allc=[]
for col,kind,cap in views:
    name=f"{col}_{kind}_{cap}"; t=time.time()
    for c in ["US","India"]:
        q=s1.filter(pl.col("country")==c); p=pool.filter(pl.col("country")==c)
        Q,P=build(q,p,col,kind,cap)
        qi,pi,sc=topk(Q,P,KMAX)
        df=pl.DataFrame({"s1":q["entity_id"].to_numpy()[qi],"m":p["entity_id"].to_numpy()[pi],"score":sc}).with_columns(
            pl.col("score").rank("ordinal",descending=True).over("s1").alias("rank"),pl.lit(name).alias("view"))
        allc.append(df)
    print(name, f"{time.time()-t:.0f}s", flush=True)
C=pl.concat(allc); C.write_parquet(f"work/cands_{NS1}.parquet")
s1.select("entity_id").write_parquet(f"work/cands_{NS1}_s1ids.parquet")
