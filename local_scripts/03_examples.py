import polars as pl, sys
s1=pl.read_parquet("work/train_s1.parquet"); pairs=pl.read_parquet("work/train_pairs.parquet")
m=pl.concat([pl.read_parquet("work/train_s2.parquet"),pl.read_parquet("work/train_s3.parquet")])
country=sys.argv[1]; n=int(sys.argv[2]); seed=int(sys.argv[3])
ids=s1.filter(pl.col("country")==country).sample(n,seed=seed)
for r in ids.iter_rows(named=True):
    print(f"S1 | {r['business_name'][:50]:50} | {r['business_address'][:90]}")
    mm=pairs.filter(pl.col("s1")==r["entity_id"]).join(m,left_on="m",right_on="entity_id").sort("m")
    for x in mm.iter_rows(named=True):
        print(f"{x['src']} | {x['business_name'][:50]:50} | {x['business_address'][:90]}")
    print("-")
