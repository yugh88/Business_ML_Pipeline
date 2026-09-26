"""Convert the challenge TSVs to the parquet inputs the pipeline expects.
usage: python tools/tsv_to_parquet.py <path/to/dataset> <out_dir>   (writes train_s{1,2,3}, test_s{1,2,3}, train_gt .parquet)"""
import sys, os, polars as pl
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
for split in ("train", "test"):
    for s in (1, 2, 3):
        df = pl.read_csv(f"{src}/{split}/{split}_source{s}.tsv", separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)
        df.write_parquet(f"{out}/{split}_s{s}.parquet"); print(split, s, df.shape)
gt = pl.read_csv(f"{src}/train/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)
gt.write_parquet(f"{out}/train_gt.parquet"); print("gt", gt.shape)
