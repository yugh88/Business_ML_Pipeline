"""Stage-3 as VETO only: keep a pair only if the base decision AND the stage-3 decision accept it (no stage-3 additions),
then optional twin rules. Holdout F0.5 (folds 0/8/9) for base / stage-3 / veto, and a veto submission.
usage: 179_veto.py <p2_dir> <base_decision.json> <s3_dir(with p3/, decision_s3.json, matching_results.tsv)> <base_matching.tsv> <out_dir>"""
import sys, json, os, shutil
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl, numpy as np, duckdb
from tune_decision import prep, decide
p2dir, bdecf, s3dir, base_tsv, outdir = sys.argv[1:6]
bdec = json.load(open(bdecf)); sdec = json.load(open(os.path.join(s3dir, "decision_s3.json"))); V = bdec["variant"]
con = duckdb.connect()
gt = con.execute("select source1_entity_id s1, case when matched_entity_ids is null or matched_entity_ids='' then 0 else len(string_split(matched_entity_ids, ',')) end n, "
                 "hash(source1_entity_id || 'fold') % 10 fold from 'work/train_gt.parquet'").pl().join(pl.read_parquet("work/v5_dropped_s1.parquet").select("s1"), on="s1", how="anti")
def f05(tp, fp, n):
    P = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-9), 0); R = np.where(n > 0, tp / np.maximum(n, 1), 0)
    return np.where(n == 0, (tp + fp == 0).astype(float), np.where(tp > 0, 1.25 * P * R / np.maximum(0.25 * P + R, 1e-12), 0.0))
def sc(pred, d, s1f):
    x = pred.join(d.select("s1", "m", "y").unique(["s1", "m"]), on=["s1", "m"])
    g = s1f.join(x.group_by("s1").agg(pl.col("y").sum().alias("tp"), (~pl.col("y")).sum().alias("fp")), on="s1", how="left").fill_null(0)
    return round(float(f05(g["tp"].to_numpy().astype(float), g["fp"].to_numpy().astype(float), g["n"].to_numpy().astype(float)).mean()), 5)
for f in (0, 8, 9):
    d = pl.scan_parquet(f"{p2dir}/train/*.parquet").filter(pl.col("fold") == f).select("s1", "m", "y", "fold", "p1", "pc2", "pa", V).collect()
    base = decide(prep(d, bdec["tau"], V), bdec).select("s1", "m")
    p3 = pl.read_parquet(os.path.join(s3dir, "p3", "train", f"fold{f}.parquet"))
    s3 = decide(prep(p3, sdec["tau"], "p3"), sdec).select("s1", "m")
    veto = base.join(s3, on=["s1", "m"], how="semi")
    s1f = gt.filter(pl.col("fold") == f)
    print(f"fold {f}: base {sc(base, d, s1f)}  stage3 {sc(s3, d, s1f)}  veto(base AND stage3) {sc(veto, d, s1f)}", flush=True)
load = lambda p: con.execute(f"select source1_entity_id s1, unnest(string_split(matched_entity_ids, ',')) m from read_csv('{p}', delim='\t', header=true, all_varchar=true, quote='') "
                             f"where matched_entity_ids is not null and matched_entity_ids<>''").pl()
b = load(base_tsv); s = load(os.path.join(s3dir, "matching_results.tsv"))
keep = b.join(s, on=["s1", "m"], how="semi")
os.makedirs(outdir, exist_ok=True)
allS1 = pl.read_parquet("work/test_s1.parquet", columns=["entity_id"]).rename({"entity_id": "source1_entity_id"})
agg = keep.group_by("s1").agg(pl.col("m").sort().str.join(",").alias("matched_entity_ids")).rename({"s1": "source1_entity_id"})
allS1.join(agg, on="source1_entity_id", how="left").with_columns(pl.col("matched_entity_ids").fill_null("")).sort("source1_entity_id") \
     .write_csv(os.path.join(outdir, "matching_results.tsv"), separator="\t", quote_style="never")
shutil.copy(os.path.join(s3dir, "candidate_pairs.tsv"), os.path.join(outdir, "candidate_pairs.tsv"))
print(f"test pairs: base {b.height}  stage3 {s.height}  veto {keep.height}  -> {outdir}")
