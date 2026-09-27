"""Model-development candidate set for eval S1 (approximation of production config D, see 07_blocking_analysis.md):
  fwd na(20k)@30 ∪ fwd c3(20k)@10 ∪ core2_sorted blocks ≤200 ∪ reverse hits already computed for true-match pool records,
then EVERY pool record in that set is re-queried in reverse (na, cap 30k, K=20) against the full S1 index to obtain
competition features (its rank for this S1, its best/2nd-best S1 score).  Output: work/dev_cands.parquet"""
import sys, time, polars as pl, numpy as np, duckdb
sys.path.insert(0, "scripts")
from retrieval2 import *
ev = pl.read_parquet("work/eval_s1_ids.parquet")
con = duckdb.connect(); con.execute("SET memory_limit='4GB'; SET threads=4")
src = open("scripts/30_block_keys.py").read(); exec(src[src.index("CORE2 ="):src.index("sel = ")])
k = KEYS["core2_sorted"]
blk = con.execute(f"""with s as (select e.entity_id s1, e.country, {k} kk from 'work/train_s1_norm.parquet' n join 'work/eval_s1_ids.parquet' e using(entity_id)),
  p as (select entity_id m, country, {k} kk from (select * from 'work/train_s2_norm.parquet' union all select * from 'work/train_s3_norm.parquet')),
  f as (select country, kk, count(*) n from p where kk is not null group by all)
  select s.s1, p.m from s join f using(country,kk) join p using(country,kk) where f.n<=200 and s.kk is not null""").pl()
fwd = pl.concat([pl.read_parquet("work/cand_eval/na_20000_*.parquet").filter(pl.col("rank") <= 30).select("s1", "m"),
                 pl.read_parquet("work/cand_eval/c3_20000_*.parquet").filter(pl.col("rank") <= 10).select("s1", "m")])
rev = pl.read_parquet("work/cand_eval/revna_30000_*.parquet").filter(pl.col("rank") <= 20).join(ev, left_on="s1", right_on="entity_id", how="semi").select("s1", "m")
base = pl.concat([fwd, blk, rev]).unique()
print("base pairs", base.height, "per S1", base.height / ev.height, "unique pool", base["m"].n_unique(), flush=True)
parts = []
for country in ["US", "India"]:
    t = time.time()
    s1 = pl.scan_parquet("work/train_s1_norm.parquet").filter(pl.col("country") == country).select("entity_id", "ncore", "a").collect()
    q = pl.concat([pl.scan_parquet(f"work/train_s{s}_norm.parquet").filter(pl.col("country") == country).select("entity_id", "ncore", "a") for s in (2, 3)]) \
          .join(base.select("m").unique().lazy(), left_on="entity_id", right_on="m", how="semi").collect()
    X = vec(texts(s1, "na"), "na"); PT, idf, keep, df = build_pool(X, 30000); del X
    Q = weight(vec(texts(q, "na"), "na"), idf, keep)
    print(country, "queries", q.height, f"rss={rss():.2f}GB", flush=True)
    qi, pi, sc, work = topk(Q, PT, df, keep, 20, work_budget=1.2e11)
    r = pl.DataFrame({"m": q["entity_id"].to_numpy()[qi], "s1": s1["entity_id"].to_numpy()[pi], "rscore": sc}).with_columns(
        pl.col("rscore").rank("ordinal", descending=True).over("m").cast(pl.Int16).alias("rrank"))
    st = r.group_by("m").agg(pl.col("rscore").max().alias("m_best"), pl.col("rscore").sort(descending=True).get(1, null_on_oob=True).alias("m_2nd"))
    parts.append((r.join(ev, left_on="s1", right_on="entity_id", how="semi"), st)); del r
    print(f"   work={work:.3g} {time.time()-t:.0f}s rss={rss():.2f}GB", flush=True)
    del PT, Q, s1, q
R = pl.concat([p[0] for p in parts]); stats = pl.concat([p[1] for p in parts])
# add reverse hits for eval S1 found from ANY re-queried pool record (symmetrizes candidate generation)
extra = R.filter(pl.col("rrank") <= 10).select("s1", "m")
allc = pl.concat([base, extra]).unique()
out = allc.join(R, on=["s1", "m"], how="left").join(stats, on="m", how="left")
lab = pl.read_parquet("work/train_pairs.parquet").select("s1", "m", pl.lit(True).alias("y"))
out = out.join(lab, on=["s1", "m"], how="left").with_columns(pl.col("y").fill_null(False))
out.write_parquet("work/dev_cands.parquet")
tp = lab.join(ev, left_on="s1", right_on="entity_id", how="semi").height
print("final pairs", out.height, "per S1", out.height / ev.height, "positives", out["y"].sum(), "/", tp, "recall", out["y"].sum() / tp)
