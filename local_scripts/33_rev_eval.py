"""REVERSE retrieval: pool record -> top-k S1 (full S1 index of the country). Queries = true-match pool records of eval S1
(for recall) + an equal-size random sample of UNMATCHED pool records (to profile distractor scores).
usage: 33_rev_eval.py VIEW CAP K"""
import sys, time, polars as pl, numpy as np
sys.path.insert(0, "scripts")
from retrieval2 import *
view, cap, K = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ev = pl.read_parquet("work/eval_s1_ids.parquet")
pairs = pl.scan_parquet("work/train_pairs.parquet")
evm = pairs.join(ev.lazy(), left_on="s1", right_on="entity_id", how="semi").select("m")
for country in ["US", "India"]:
    t = time.time()
    s1 = pl.scan_parquet("work/train_s1_norm.parquet").filter(pl.col("country") == country).select("entity_id", "ncore", "a").collect()
    poolc = pl.concat([pl.scan_parquet(f"work/train_s{s}_norm.parquet").filter(pl.col("country") == country).select("entity_id", "ncore", "a") for s in (2, 3)])
    qt = poolc.join(evm, left_on="entity_id", right_on="m", how="semi").collect()
    qu = poolc.join(pairs.select("m"), left_on="entity_id", right_on="m", how="anti").filter(pl.col("entity_id").hash(5) % 400 == 0).collect()
    q = pl.concat([qt.with_columns(pl.lit(True).alias("is_true_m")), qu.with_columns(pl.lit(False).alias("is_true_m"))])
    X = vec(texts(s1, view), view)
    PT, idf, keep, df = build_pool(X, cap); del X
    Q = weight(vec(texts(q, view), view), idf, keep)
    print(f"REV {view} cap={cap} {country}: S1={s1.height} q={q.height} (true {qt.height}, unmatched {qu.height}) rss={rss():.2f}GB build={time.time()-t:.0f}s", flush=True)
    t = time.time()
    qi, pi, sc, work = topk(Q, PT, df, keep, K, work_budget=4e10)
    out = pl.DataFrame({"m": q["entity_id"].to_numpy()[qi], "s1": s1["entity_id"].to_numpy()[pi], "score": sc,
                        "q_true": q["is_true_m"].to_numpy()[qi]}).with_columns(
        pl.col("score").rank("ordinal", descending=True).over("m").cast(pl.Int16).alias("rank"))
    out.write_parquet(f"work/cand_eval/rev{view}_{cap}_{country}.parquet")
    print(f"   work={work:.3g} topk={time.time()-t:.0f}s rows={out.height} rss={rss():.2f}GB", flush=True)
