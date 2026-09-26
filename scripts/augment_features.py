"""Stage augment: add featx.EXTRA columns to existing feature buckets (so v3 reuses v2 retrieval/features)."""
import os, sys, time, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def _bucket(args):
    split, b, env = args
    os.environ.update(env)
    import polars as pl
    from lib import WORK, P, write_parquet_atomic
    from featx import add_extra, EXTRA
    f = P(f"features/{split}/bucket_{b:04d}.parquet")
    if not os.path.exists(f) or "num_kind" in pl.read_parquet_schema(f):
        return
    d = pl.read_parquet(f)
    ng = lambda s: os.path.join(WORK, "normalized", f"{split}_s{s}", "*.parquet")
    s1 = pl.scan_parquet(ng(1)).select(pl.col("entity_id").alias("s1"), pl.col("n").alias("n_1"), pl.col("pnum1").alias("pnum1_1")) \
           .join(d.lazy().select("s1").unique(), on="s1", how="semi").collect()
    po = pl.concat([pl.scan_parquet(ng(s)).select(pl.col("entity_id").alias("m"), pl.col("n").alias("n_2")) for s in (2, 3)]) \
           .join(d.lazy().select("m").unique(), on="m", how="semi").collect()
    x = add_extra(d.select("s1", "m", pl.col("pnum1").alias("pnum1_2")).join(s1, on="s1", how="left").join(po, on="m", how="left"))
    write_parquet_atomic(d.join(x.select(["s1", "m"] + EXTRA), on=["s1", "m"], how="left"), f"features/{split}/bucket_{b:04d}.parquet")

def main():
    from lib import CFG, log, done, mark_done, s3_push
    if done("augment"):
        return
    for split in CFG["splits"]:
        W = CFG["features"]["workers"]; env = {"POLARS_MAX_THREADS": str(max(1, (os.cpu_count() or 4) // W))}
        jobs = [(split, b, env) for b in range(CFG["features"]["buckets"][split])]
        t = time.time()
        if W <= 1:
            for j in jobs: _bucket(j)
        else:
            with mp.get_context("spawn").Pool(W, maxtasksperchild=8) as pool:
                for i, _ in enumerate(pool.imap_unordered(_bucket, jobs)):
                    if i % 100 == 0: log(f"  augment {split} {i}/{len(jobs)} {time.time()-t:.0f}s")
        log(f"augment {split} done {time.time()-t:.0f}s")
    s3_push("features", recursive=True)
    mark_done("augment")

if __name__ == "__main__":
    main()
