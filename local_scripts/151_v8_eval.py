"""V8 evaluation (after the AWS run): holdout F0.5 with and without the synthetic hard negatives, and the label-free
test diagnostics that exposed the V5 gap (analysis/out/119-135): +k vs -k house-offset symmetry, collectively promoted
records (p2 accepts, pairwise p1 rejects), predicted matches per S1 by country.
usage: 151_v8_eval.py <p2_dir> <decision.json> <matching_results.tsv> [label]"""
import sys, json
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl, numpy as np, duckdb
from tune_decision import prep, decide

p2dir, decf, mres = sys.argv[1:4]
label = sys.argv[4] if len(sys.argv) > 4 else p2dir
dec = json.load(open(decf))
V = dec["variant"]
con = duckdb.connect()
SYN = pl.concat([pl.read_parquet(f"work/aug_v8_{s}.parquet", columns=["entity_id"]) for s in ("s2", "s3")]).rename({"entity_id": "m"})


def f05(tp, fp, n):
    P = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-9), 0); R = np.where(n > 0, tp / np.maximum(n, 1), 0)
    return np.where(n == 0, (tp + fp == 0).astype(float), np.where(tp > 0, 1.25 * P * R / np.maximum(0.25 * P + R, 1e-12), 0.0))


def hn(split, d):
    ids1 = d.select(pl.col("s1").alias("entity_id")).unique(); ids2 = d.select(pl.col("m").alias("entity_id")).unique()
    n1 = pl.scan_parquet(f"work/{split}_s1_norm.parquet").select("entity_id", "nums").join(ids1.lazy(), on="entity_id", how="semi").collect()
    n2 = pl.concat([pl.scan_parquet(f"work/{split}_s{s}_norm.parquet").select("entity_id", "nums") for s in (2, 3)]).join(ids2.lazy(), on="entity_id", how="semi").collect()
    f = lambda c: pl.col(c).str.split(" ").list.first().fill_null("")
    d = d.join(n1.select(pl.col("entity_id").alias("s1"), f("nums").alias("h1")), on="s1", how="left") \
         .join(n2.select(pl.col("entity_id").alias("m"), f("nums").alias("h2")), on="m", how="left")
    dd = pl.col("h2").cast(pl.Int64, strict=False) - pl.col("h1").cast(pl.Int64, strict=False)
    return d.with_columns(pl.when((dd >= 1) & (dd <= 9)).then(pl.lit("P")).when((dd <= -1) & (dd >= -9)).then(pl.lit("M")).otherwise(pl.lit("o")).alias("k"))


gt = con.execute("select source1_entity_id s1, case when matched_entity_ids is null or matched_entity_ids='' then 0 else "
                 "len(string_split(matched_entity_ids, ',')) end n, hash(source1_entity_id || 'fold') % 10 fold from 'work/train_gt.parquet'").pl()
gt = gt.join(pl.read_parquet("work/v5_dropped_s1.parquet").select("s1"), on="s1", how="anti")
out = {"label": label, "decision": {k: dec.get(k) for k in ("tau", "variant", "policy", "a", "lam", "t", "cap")}, "holdout": {}}
for f in [0, 8, 9]:
    d = pl.scan_parquet(f"{p2dir}/train/*.parquet").filter(pl.col("fold") == f).select("s1", "m", "y", "fold", "p1", "pc2", "pa", V).collect()
    s1f = gt.filter(pl.col("fold") == f)
    r = {}
    for view, dd in (("augmented", d), ("clean_no_synthetic", d.join(SYN, on="m", how="anti"))):
        pr = decide(prep(dd, dec["tau"], V), dec).join(dd.select("s1", "m", "y"), on=["s1", "m"])
        g = s1f.join(pr.group_by("s1").agg(pl.col("y").sum().alias("tp"), (~pl.col("y")).sum().alias("fp")), on="s1", how="left").fill_null(0)
        r[view] = round(float(f05(g["tp"].to_numpy().astype(float), g["fp"].to_numpy().astype(float), g["n"].to_numpy().astype(float)).mean()), 5)
        if view == "augmented":
            r["synthetic_accepted"] = int(pr.join(SYN, on="m", how="semi").height)
    out["holdout"][f] = r
    print(f"fold {f}: {r}", flush=True)
# ---- test diagnostics
p = con.execute(f"select source1_entity_id s1, unnest(string_split(matched_entity_ids, ',')) m from read_csv('{mres}', delim='\t', header=true, "
                f"all_varchar=true, quote='') where matched_entity_ids is not null and matched_entity_ids<>''").pl()
s1c = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
nS1 = dict(s1c.group_by("country").len().rows())
x = hn("test", p.join(s1c, on="s1"))
out["test"] = {}
for c in ["US", "India", "France"]:
    xc = x.filter(pl.col("country") == c); n = nS1[c] / 1000
    cnt = dict(xc.group_by("k").len().rows())
    t = pl.scan_parquet(f"{p2dir}/test/*.parquet").filter(pl.col("country") == c).select("s1", "m", "p1", "pc2", "pa", V).collect()
    acc1 = decide(prep(t, 0.05, "p1"), dict(policy="G_expected_f", a=1.25, lam=0.0, cap=1)).with_columns(pl.lit(True).alias("a1"))
    prom = xc.join(acc1, on=["s1", "m"], how="left").filter(pl.col("a1").is_null()).height
    out["test"][c] = dict(pred_per_s1=round(xc.height / nS1[c], 4), plus_k_per_1k=round(cnt.get("P", 0) / n, 2), minus_k_per_1k=round(cnt.get("M", 0) / n, 2),
                          promoted_per_1k=round(prom / n, 2))
    print(c, out["test"][c], flush=True)
    del t, acc1
print("reference V5: US +k 43.8 -k 18.6 promoted 79.9 | India +k 57.6 -k 38.9 promoted 135.3 | France +k 21.7 -k 5.9 promoted 111.3 (per 1k S1)")
json.dump(out, open(f"analysis/out/151_eval_{label}.json", "w"), indent=1)
