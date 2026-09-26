"""Stage blocking: production candidate generation (analysis/07 config F):
  reverse na (pool->S1, cap 30k, top-20) ∪ reverse c3 (cap 5k, top-10) ∪ forward na (S1->pool, cap 20k, top-10)
  ∪ exact sorted-core2 blocks with pool block size <= 200.
Then DuckDB union -> candidates/{split}/bucket=k/*.parquet with retrieval features, per-pool-record competition
stats (best / 2nd-best reverse score), CV fold and (train) label y."""
import os, glob, duckdb, polars as pl
from lib import CFG, P, WORK, log, done, mark_done, s3_push
import retrieval

B = CFG["blocking"]


def norm_glob(split, s):
    return os.path.join(WORK, "normalized", f"{split}_s{s}", "*.parquet")


def retrieve(split):
    s1 = pl.read_parquet(norm_glob(split, 1), columns=["entity_id", "country", "core2", "a"])
    pool = pl.concat([pl.read_parquet(norm_glob(split, s), columns=["entity_id", "country", "core2", "a"]) for s in (2, 3)])
    countries = sorted(set(s1["country"].unique().to_list()) | set(pool["country"].unique().to_list()))
    for c in countries:
        q = s1.filter(pl.col("country") == c); p = pool.filter(pl.col("country") == c)
        if q.height == 0 or p.height == 0:
            continue
        log(f"retrieval {split} {c}: S1={q.height} pool={p.height}")
        L = lambda df: (df["entity_id"].to_list(), df["core2"].to_list(), df["a"].to_list())
        qi, qc, qa = L(q); pi_, pc, pa_ = L(p)
        common = dict(shard_rows=B["query_shard_rows"], workers=B["workers"], max_out=B["max_out_nnz"], mem_gb=B["worker_mem_gb"], log=log)
        for view, key in (("na", "rev_na"), ("c3", "rev_c3")):
            out = P("cand_raw", split, key, c, "x")[:-2]
            retrieval.run(view, qi, qc, qa, pi_, pc, pa_, B[key]["cap"], B[key]["k"], out, "m", "s1", **common)
        out = P("cand_raw", split, "fwd_na", c, "x")[:-2]
        retrieval.run("na", pi_, pc, pa_, qi, qc, qa, B["fwd_na"]["cap"], B["fwd_na"]["k"], out, "s1", "m", **common)
        del q, p


def union(split):
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{int(0.5 * __import__('psutil').virtual_memory().total / 1e9)}GB'; SET threads={os.cpu_count()};"
                f" SET preserve_insertion_order=false; SET temp_directory='{P('duck_tmp', 'x')[:-2]}'")
    raw = lambda k: os.path.join(WORK, "cand_raw", split, k, "*", "*.parquet")
    ng = lambda s: norm_glob(split, s)
    key = "array_to_string(list_sort(string_split(core2,' ')),' ')"
    con.execute(f"""create temp table blk as
      with s as (select entity_id s1, country, {key} kk from '{ng(1)}' where core2<>''),
      p as (select entity_id m, country, {key} kk from (select * from '{ng(2)}' union all select * from '{ng(3)}') where core2<>''),
      f as (select country, kk, count(*) n from p group by all)
      select s.s1, p.m from s join f using(country, kk) join p using(country, kk) where f.n <= {B['core2_block_max']}""")
    con.execute(f"""create temp table rn as select m, s1, score rscore, rank rrank from '{raw('rev_na')}'""")
    con.execute(f"""create temp table mstat as select m, max(rscore) m_best,
                    coalesce(max(rscore) filter (where rrank=2), 0) m_2nd from rn group by m""")
    con.execute(f"""create temp table cand as
      select s1, m, max(rscore) rscore, min(rrank) rrank, max(c3score) c3score, min(c3rank) c3rank,
             max(fscore) fscore, min(frank) frank, max(in_block) in_block from (
        select s1, m, rscore, rrank, null::float c3score, null::smallint c3rank, null::float fscore, null::smallint frank, 0 in_block from rn
        union all select s1, m, null, null, score, rank, null, null, 0 from '{raw('rev_c3')}'
        union all select s1, m, null, null, null, null, score, rank, 0 from '{raw('fwd_na')}'
        union all select s1, m, null, null, null, null, null, null, 1 from blk) group by s1, m""")
    ylab = "false"
    if split == "train":
        con.execute(f"""create temp table gt as select source1_entity_id s1, trim(unnest(string_split(matched_entity_ids, ','))) m
                        from '{P('input', 'train_gt.parquet')}' where matched_entity_ids is not null and matched_entity_ids<>''""")
        ylab = "(g.m is not null)"
    NB = CFG["features"]["buckets"][split]
    out = P("candidates", split, "x")[:-2]
    con.execute(f"""copy (select c.*, coalesce(ms.m_best, 0) m_best, coalesce(ms.m_2nd, 0) m_2nd, {ylab} y,
                      hash(c.s1) % {NB} bucket, hash(c.s1 || 'fold') % {CFG['model']['folds']} fold
                    from cand c left join mstat ms using(m) {'left join gt g using(s1, m)' if split == 'train' else ''})
                    to '{out}' (format parquet, partition_by (bucket), compression zstd, overwrite_or_ignore)""")
    st = con.execute(f"select count(*), count(distinct s1), sum(y::int) from '{out}/*/*.parquet'").fetchone()
    info = dict(pairs=st[0], s1_with_cands=st[1], positives=st[2])
    if split == "train":
        info["gt_pairs"] = con.execute("select count(*) from gt").fetchone()[0]
        info["blocking_recall"] = st[2] / info["gt_pairs"]
    info["s1_total"] = con.execute(f"select count(*) from '{ng(1)}'").fetchone()[0]
    info["cands_per_s1"] = st[0] / info["s1_total"]
    log(f"union {split}: {info}")
    return info


def main():
    for split in CFG["splits"]:
        if not done(f"retrieve_{split}"):
            retrieve(split); s3_push(f"cand_raw/{split}", recursive=True); mark_done(f"retrieve_{split}")
        if not done(f"union_{split}"):
            info = union(split); s3_push(f"candidates/{split}", recursive=True); mark_done(f"union_{split}", info)


if __name__ == "__main__":
    main()
