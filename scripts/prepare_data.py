"""Stage prepare: fetch raw parquet inputs from S3 (or use local copies) and verify row counts."""
import os, pyarrow.parquet as pq
from lib import CFG, P, log, s3_pull, done, mark_done

def main():
    if done("prepare"):
        return
    counts = {}
    for f in CFG["input_files"]:
        if not os.path.exists(P("input", f)):
            s3_pull(f"input/{f}")
        counts[f] = pq.ParquetFile(P("input", f)).metadata.num_rows
        log("input", f, counts[f])
    mark_done("prepare", counts)

if __name__ == "__main__":
    main()
