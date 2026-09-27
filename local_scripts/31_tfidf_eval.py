"""Evaluate a TF-IDF retrieval view on the fixed eval S1 sample (hash%50==0) against the FULL country pool.
usage: 31_tfidf_eval.py VIEW CAP K"""
import sys, time, polars as pl, numpy as np
sys.path.insert(0, "scripts")
from retrieval2 import *
view, cap, K = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
BUDGET = 4e10
ev = pl.scan_parquet("work/train_s1_norm.parquet").filter(pl.col("entity_id").hash(3) % 50 == 0)
ev.select("entity_id", "country").collect().write_parquet("work/eval_s1_ids.parquet")
for country in ["US", "India"]:
    t = time.time()
    pool = pl.concat([pl.scan_parquet(f"work/train_s{s}_norm.parquet").filter(pl.col("country") == country).select("entity_id", "ncore", "a") for s in (2, 3)]).collect()
    q = ev.filter(pl.col("country") == country).select("entity_id", "ncore", "a").collect()
    P_raw = vec(texts(pool, view), view)
    PT, idf, keep, df = build_pool(P_raw, cap); del P_raw
    Q = weight(vec(texts(q, view), view), idf, keep)
    print(f"{view} cap={cap} {country}: pool={pool.height} q={q.height} PT.nnz={PT.nnz/1e6:.0f}M rss={rss():.2f}GB build={time.time()-t:.0f}s", flush=True)
    t = time.time()
    qi, pi, sc, work = topk(Q, PT, df, keep, K, work_budget=BUDGET)
    out = pl.DataFrame({"s1": q["entity_id"].to_numpy()[qi], "m": pool["entity_id"].to_numpy()[pi], "score": sc}).with_columns(
        pl.col("score").rank("ordinal", descending=True).over("s1").cast(pl.Int16).alias("rank"))
    out.write_parquet(f"work/cand_eval/{view}_{cap}_{country}.parquet")
    print(f"   work={work:.3g} topk={time.time()-t:.0f}s cands={out.height} rss={rss():.2f}GB", flush=True)
    del pool, PT, Q, out
