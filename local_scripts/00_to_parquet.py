import polars as pl, time, os
os.makedirs("work", exist_ok=True)
for split in ["train","test"]:
    for s in [1,2,3]:
        t=time.time()
        df=pl.read_csv(f"dataset/{split}/{split}_source{s}.tsv",separator="\t",quote_char=None,
                       infer_schema=False,missing_utf8_is_empty_string=True,encoding="utf8")
        df.write_parquet(f"work/{split}_s{s}.parquet")
        print(split,s,df.shape,df.columns,f"{time.time()-t:.1f}s",flush=True)
gt=pl.read_csv("dataset/train/train_ground_truth.tsv",separator="\t",quote_char=None,infer_schema=False,missing_utf8_is_empty_string=True)
gt.write_parquet("work/train_gt.parquet"); print("gt",gt.shape)
