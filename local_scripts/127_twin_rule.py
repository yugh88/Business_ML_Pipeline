"""Twin-cluster rule on top of V5 predictions: reject an accepted candidate whose house number is S1's + k (k=1..9)
when the S1's own number is supported by other accepted candidates. Variants by where the support must be."""
import sys, json; sys.path.insert(0, "aws/scripts"); import memguard_local
import polars as pl, numpy as np, duckdb
exec(open("scripts/119_offset.py").read().split("rel = pl.read_parquet")[0])
sys.path.insert(0, "scripts"); from tune_decision import prep, decide
dec = json.load(open("analysis/aws_v5/decision_v5.json"))
def f05(tp, fp, n):
    P = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-9), 0); R = np.where(n > 0, tp / np.maximum(n, 1), 0)
    return np.where(n == 0, (tp + fp == 0).astype(float), np.where(tp > 0, 1.25 * P * R / np.maximum(0.25 * P + R, 1e-12), 0.0))
def flags(pr):
    """pr: accepted pairs with h1,h2,src. Adds plusk and support counts among OTHER accepted candidates."""
    i1, i2 = pl.col("h1").cast(pl.Int64, strict=False), pl.col("h2").cast(pl.Int64, strict=False)
    pr = pr.with_columns(((i2 - i1 >= 1) & (i2 - i1 <= 9)).fill_null(False).alias("plusk"), ((pl.col("h2") == pl.col("h1")) & (pl.col("h1") != "")).alias("ish1"))
    return pr.with_columns(pl.col("ish1").cast(pl.Int32).sum().over("s1").alias("A_all"), pl.col("ish1").cast(pl.Int32).sum().over(["s1", "src"]).alias("A_src"),
                           pl.col("plusk").cast(pl.Int32).sum().over(["s1", "h2"]).alias("grp"))
RULES = {"T1_any_support": lambda x: x["plusk"] & (x["A_all"] >= 1),
         "T2_same_src_support": lambda x: x["plusk"] & (x["A_src"] >= 1),
         "T3_support>=2": lambda x: x["plusk"] & (x["A_all"] >= 2),
         "T4_any_support_grp>=2": lambda x: x["plusk"] & (x["A_all"] >= 1) & (x["grp"] >= 2)}
gt = duckdb.connect().execute("select source1_entity_id s1, case when matched_entity_ids is null or matched_entity_ids='' then 0 else len(string_split(matched_entity_ids, ',')) end n, hash(source1_entity_id || 'fold') % 10 fold from 'work/train_gt.parquet'").pl()
gt = gt.join(pl.read_parquet("work/v5_dropped_s1.parquet").select("s1"), on="s1", how="anti")
out = {}
for f in [0, 8, 9]:
    d = pl.scan_parquet("work/v5_p2/train/*.parquet").filter(pl.col("fold") == f).select("s1", "m", "y", "fold", "p1", "pc2", "pa", "p2x2").collect()
    pr = decide(prep(d, dec["tau"], "p2x2"), dec).select("s1", "m").join(d.select("s1", "m", "y"), on=["s1", "m"])
    pr = flags(attach(pr, "train").with_columns(pl.col("m").str.slice(0, 2).alias("src")))
    s1f = gt.filter(pl.col("fold") == f)
    def score(keep):
        x = pr.filter(keep).group_by("s1").agg(pl.col("y").sum().alias("tp"), (~pl.col("y")).sum().alias("fp"))
        g = s1f.join(x, on="s1", how="left").fill_null(0)
        return float(f05(g["tp"].to_numpy().astype(float), g["fp"].to_numpy().astype(float), g["n"].to_numpy().astype(float)).mean())
    base = score(pl.lit(True)); line = f"fold {f}: V5 {base:.5f}"
    for k, r in RULES.items():
        rm = pr.filter(r(pr)); line += f" | {k} {score(~r(pr)):.5f} (rm TP {rm['y'].sum()} FP {(~rm['y']).sum()})"
    print(line, flush=True)
# test: how many removed, and +k vs -k symmetry after rule
p5 = duckdb.connect().execute("select source1_entity_id s1, unnest(string_split(matched_entity_ids, ',')) m from read_csv('output_v5/matching_results.tsv', delim='\t', header=true, all_varchar=true, quote='') where matched_entity_ids is not null and matched_entity_ids<>''").pl()
s1c = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"}); nS1t = dict(s1c.group_by("country").len().rows())
te = flags(attach(p5.join(s1c, on="s1"), "test").with_columns(pl.col("m").str.slice(0, 2).alias("src")))
i1, i2 = pl.col("h1").cast(pl.Int64, strict=False), pl.col("h2").cast(pl.Int64, strict=False)
te = te.with_columns(((i2 - i1 <= -1) & (i2 - i1 >= -9)).fill_null(False).alias("minusk"))
for c in ["US", "India", "France"]:
    x = te.filter(pl.col("country") == c); n = nS1t[c] / 1000
    line = f"TEST {c}: +k pred/1k {x['plusk'].sum()/n:.1f}  -k pred/1k {x['minusk'].sum()/n:.1f}"
    for k, r in RULES.items():
        line += f" | {k}: removes {r(x).sum()/n:.1f}/1k -> +k left {(x['plusk'] & ~r(x)).sum()/n:.1f}"
    print(line)
te.select("s1", "m", "country", "h1", "h2", "src", "plusk", "A_all", "A_src", "grp").write_parquet("work/test_v5_twinflags.parquet")
