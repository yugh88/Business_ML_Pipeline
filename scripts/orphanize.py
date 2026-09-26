"""Stage orphanize (analysis/17 §Hidden-LB gap): simulate the test distribution on train.
Test pool contains clusters of copies of businesses ABSENT from test S1 (orphans): 63.5% of unmatched test records have an
unmatched look-alike sibling vs 18% in train (analysis/out/83). We drop ORPHAN_FRAC of train S1 entities (hash, all folds),
KEEP their S2/S3 copies (now labeled negatives everywhere) and recompute every feature that depends on the S1 set:
reverse-retrieval rank / best / 2nd-best score (per pool record, over remaining S1) and margins.
Evaluation and tuning exclude the dropped S1 (lib.dropped_s1_expr)."""
import os, sys, time, multiprocessing as mp, duckdb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _bucket(args):
    b, env = args
    os.environ.update(env)
    import polars as pl
    from lib import P, write_parquet_atomic, keep_s1_expr
    f = P(f"features/train/bucket_{b:04d}.parquet")
    if not os.path.exists(f) or "orphanized" in pl.read_parquet_schema(f):
        return
    d = pl.read_parquet(f).filter(keep_s1_expr("s1"))
    ms = pl.read_parquet(P("orphan/mstat.parquet")).join(d.select("m").unique(), on="m", how="semi")
    rk = pl.read_parquet(P(f"orphan/rrank_{b:04d}.parquet"))
    d = d.drop("m_best", "m_gap12", "r_margin_best").join(ms, on="m", how="left").join(rk, on=["s1", "m"], how="left")
    d = d.with_columns(pl.col("m_best").fill_null(0), pl.col("m_2nd").fill_null(0),
                       pl.when(pl.col("rrank") < 99).then(pl.col("rrank_new")).otherwise(99).cast(pl.Int16).alias("rrank"))
    d = d.with_columns((pl.col("rscore") - pl.col("m_best")).alias("r_margin_best"), (pl.col("m_best") - pl.col("m_2nd")).alias("m_gap12"),
                       pl.lit(True).alias("orphanized")).drop("m_2nd", "rrank_new")
    write_parquet_atomic(d, f"features/train/bucket_{b:04d}.parquet")


def main():
    from lib import CFG, P, WORK, log, done, mark_done, s3_push, keep_s1_sql, write_kept_s1
    if done("orphanize") or not CFG["model"].get("orphan_frac"):
        return
    t = time.time()
    write_kept_s1()
    con = duckdb.connect(); con.execute(f"SET threads={os.cpu_count()}; SET preserve_insertion_order=false; SET temp_directory='{P('duck_tmp','x')[:-2]}'")
    src = os.path.join(WORK, "features", "train", "*.parquet")
    os.makedirs(P("orphan", "x")[:-2], exist_ok=True)
    # remaining reverse-retrieval rows (rscore > 0 means the pair came from the reverse list)
    con.execute(f"create temp table r as select s1, m, rscore, hash(s1) % {CFG['features']['buckets']['train']} b from '{src}' where rscore > 0 and {keep_s1_sql('s1')}")
    log(f"orphanize: mode={'list' if CFG['model'].get('orphan_list') else 'hash'}")
    con.execute(f"copy (select m, max(rscore) m_best, coalesce(list_sort(list(rscore), 'DESC')[2], 0) m_2nd from r group by m) to '{P('orphan/mstat.parquet')}' (format parquet)")
    con.execute("create temp table rr as select s1, m, b, row_number() over (partition by m order by rscore desc)::smallint rrank_new from r")
    NB = CFG["features"]["buckets"]["train"]
    for b in range(NB):
        con.execute(f"copy (select s1, m, rrank_new from rr where b = {b}) to '{P(f'orphan/rrank_{b:04d}.parquet')}' (format parquet)")
    n_before = con.execute(f"select count(distinct s1) from '{src}'").fetchone()[0]
    log(f"orphanize: reverse stats recomputed {time.time()-t:.0f}s; S1 with candidates before={n_before}")
    W = CFG["features"]["workers"]; env = {"POLARS_MAX_THREADS": str(max(1, (os.cpu_count() or 4) // W))}
    jobs = [(b, env) for b in range(NB)]
    if W <= 1:
        for j in jobs: _bucket(j)
    else:
        with mp.get_context("spawn").Pool(W, maxtasksperchild=8) as pool:
            list(pool.imap_unordered(_bucket, jobs))
    st = con.execute(f"select count(distinct s1), count(*), avg((not y)::int) from '{src}'").fetchone()
    log(f"orphanize done: S1 kept={st[0]} pairs={st[1]} negative share={st[2]:.4f} {time.time()-t:.0f}s")
    s3_push("features/train", recursive=True)
    mark_done("orphanize", dict(s1_kept=st[0], pairs=st[1]))


if __name__ == "__main__":
    main()
