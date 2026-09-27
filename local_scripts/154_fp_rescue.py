"""FN rescue with the address/name fingerprint: among candidates REJECTED by the V5 decision, those whose raw address
(or raw name) exactly equals that of an ACCEPTED same-source candidate of the same S1. Precision on train holdout
and holdout F0.5 if we add them; test counts per 1k S1."""
import sys, json; sys.path.insert(0, "aws/scripts"); import memguard_local
import polars as pl, numpy as np, duckdb
from tune_decision import prep, decide
dec = json.load(open("analysis/aws_v5/decision_v5.json"))
def f05(tp, fp, n):
    P = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-9), 0); R = np.where(n > 0, tp / np.maximum(n, 1), 0)
    return np.where(n == 0, (tp + fp == 0).astype(float), np.where(tp > 0, 1.25 * P * R / np.maximum(0.25 * P + R, 1e-12), 0.0))
def rescue(d, sp, pmin):
    pr = decide(prep(d, dec["tau"], "p2x2"), dec).with_columns(pl.lit(True).alias("acc"))
    x = d.filter(pl.col("p1") >= dec["tau"]).join(pr, on=["s1", "m"], how="left").with_columns(pl.col("acc").fill_null(False))
    ids = x.select(pl.col("m").alias("entity_id")).unique()
    raw = pl.concat([pl.scan_parquet(f"work/{sp}_s{s}.parquet") for s in (2, 3)]).join(ids.lazy(), on="entity_id", how="semi").collect()
    raw = raw.select(pl.col("entity_id").alias("m"), pl.col("business_address").fill_null("").alias("ra"), pl.col("business_name").fill_null("").alias("rn"))
    x = x.join(raw, on="m").with_columns(pl.col("m").str.slice(0, 2).alias("src"))
    acc = x.filter(pl.col("acc")).select("s1", "src", "ra", "rn", pl.col("m").alias("m2"))
    cand = x.filter(~pl.col("acc") & (pl.col("p2x2") >= pmin))
    ha = cand.filter(pl.col("ra") != "").join(acc.filter(pl.col("ra") != "").select("s1", "src", "ra").unique(), on=["s1", "src", "ra"], how="semi").select("s1", "m").with_columns(pl.lit("addr").alias("how"))
    hn_ = cand.join(acc.select("s1", "src", "rn").unique(), on=["s1", "src", "rn"], how="semi").select("s1", "m").with_columns(pl.lit("name").alias("how"))
    hits = pl.concat([ha, hn_]).unique(subset=["s1", "m"], keep="first")
    # a rescued record must not be the best-S1 match of another S1 (competition): require p2x2 >= m_max over S1s
    mm = d.group_by("m").agg(pl.col("p2x2").max().alias("mmax"))
    hits = hits.join(x.select("s1", "m", "p2x2"), on=["s1", "m"]).join(mm, on="m").filter(pl.col("p2x2") >= pl.col("mmax") - 1e-9)
    return pr.select("s1", "m"), hits
gt = duckdb.connect().execute("select source1_entity_id s1, case when matched_entity_ids is null or matched_entity_ids='' then 0 else len(string_split(matched_entity_ids, ',')) end n, hash(source1_entity_id || 'fold') % 10 fold from 'work/train_gt.parquet'").pl()
gt = gt.join(pl.read_parquet("work/v5_dropped_s1.parquet").select("s1"), on="s1", how="anti")
for pmin in (0.0, 0.05, 0.2):
    for f in [0, 8]:
        d = pl.scan_parquet("work/v5_p2/train/*.parquet").filter(pl.col("fold") == f).select("s1", "m", "y", "fold", "p1", "pc2", "pa", "p2x2").collect()
        pr, hits = rescue(d, "train", pmin)
        hits = hits.join(d.select("s1", "m", "y"), on=["s1", "m"])
        s1f = gt.filter(pl.col("fold") == f)
        def sc(p):
            g = s1f.join(p.join(d.select("s1", "m", "y"), on=["s1", "m"]).group_by("s1").agg(pl.col("y").sum().alias("tp"), (~pl.col("y")).sum().alias("fp")), on="s1", how="left").fill_null(0)
            return f05(g["tp"].to_numpy().astype(float), g["fp"].to_numpy().astype(float), g["n"].to_numpy().astype(float)).mean()
        print(f"pmin {pmin} fold {f}: rescued {hits.height} precision {hits['y'].mean():.4f} by how {hits.group_by('how').agg(pl.len(), pl.col('y').mean().round(4)).rows()} | F {sc(pr):.5f} -> {sc(pl.concat([pr, hits.select('s1','m')]).unique()):.5f}", flush=True)
