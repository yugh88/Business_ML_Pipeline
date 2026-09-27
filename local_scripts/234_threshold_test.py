"""Teammate's decisive ingredient: HIGH probability threshold (>0.92) instead of expected-F decoding. Tested on OUR scores
(V11 per-country stage-3 p3, the V12 base): accept (s1,m) iff q >= t and q is the record's best S1 score (exclusivity) + per-source cap.
q = raw p3 (and the alpha-0.75 pairwise mix for US/India). For t in a grid:
  holdout clean F per country (labelled cost) | test accepted/1k, removed-vs-V12 composition, +k twin share, French address fingerprint |
  holdout-referenced value of the test changes vs V12 (211 method) | simulator delta vs V9 (alpha 0.5 / 0)."""
import sys, json
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl, numpy as np, duckdb
from sklearn.isotonic import IsotonicRegression
from tune_decision import prep, decide
exec(open("scripts/213_excess_profile.py").read().split("t1 = pl.read_parquet")[0])        # annotate()
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(230); pl.Config.set_tbl_hide_dataframe_shape(True)
S3D = "output_v11s3"; dec = json.load(open(f"{S3D}/decision_s3.json")); TAU = dec["tau"]
SYN = pl.concat([pl.read_parquet(f"work/aug_v8_{s}.parquet", columns=["entity_id"]) for s in ("s2", "s3")]).rename({"entity_id": "m"})
tr = pl.scan_parquet("work/v8_p2/train/*.parquet").filter(pl.col("fold").is_in([0, 8, 9])).select("m", "y", "p1").collect().join(SYN, on="m", how="anti")
iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(tr["p1"].to_numpy(), tr["y"].to_numpy()); del tr
CAP = {"S2": 5, "S3": 6}
TS = [0.5, 0.7, 0.8, 0.85, 0.9, 0.92, 0.95, 0.97]


def F(tp, fp, n):
    tp, fp, n = (np.asarray(v, float) for v in (tp, fp, n))
    P = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-9), 0); R = np.where(n > 0, tp / np.maximum(n, 1), 0)
    return np.where(n == 0, (tp + fp == 0).astype(float), np.where(tp > 0, 1.25 * P * R / np.maximum(0.25 * P + R, 1e-12), 0.0))


def score_col(x, alpha):
    x = x.with_columns(pl.Series("ip1", iso.predict(x["p1"].to_numpy())).cast(pl.Float32))
    return x.with_columns((alpha * pl.col("p3") + (1 - alpha) * pl.col("ip1")).alias("q"))


def thr(x, t):
    """threshold decoder with exclusivity (record's best S1) and per-source cap."""
    x = x.filter(pl.col("p1") >= TAU)
    x = x.with_columns(pl.col("q").max().over("m").alias("mmax")).filter((pl.col("q") >= t) & (pl.col("q") >= pl.col("mmax")))
    x = x.with_columns(pl.col("m").str.slice(0, 2).alias("src")).with_columns(pl.col("q").rank("ordinal", descending=True).over(["s1", "src"]).alias("r"))
    return x.filter(pl.col("r") <= pl.col("src").replace_strict(CAP, default=99)).select("s1", "m")


gt = duckdb.connect().execute("select source1_entity_id s1, case when matched_entity_ids is null or matched_entity_ids='' then 0 else len(string_split(matched_entity_ids, ',')) end n, "
                              "hash(source1_entity_id || 'fold') % 10 fold from 'work/train_gt.parquet'").pl()
gt = gt.join(pl.read_parquet("work/v5_dropped_s1.parquet").select("s1"), on="s1", how="anti").filter(pl.col("fold").is_in([0, 8, 9]))
t1 = pl.read_parquet("work/train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"}); gt = gt.join(t1, on="s1")
h = pl.concat([pl.read_parquet(f"{S3D}/p3/train/fold{f}.parquet", columns=["s1", "m", "y", "p1", "p3"]) for f in (0, 8, 9)]).join(SYN, on="m", how="anti").join(t1, on="s1")
nh = dict(gt.group_by("country").len().rows())


def hscore(acc):
    a = acc.join(h.select("s1", "m", "y"), on=["s1", "m"])
    z = gt.join(a.group_by("s1").agg(pl.col("y").sum().alias("tp"), (~pl.col("y")).sum().alias("fp")), on="s1", how="left").fill_null(0)
    z = z.with_columns(pl.Series("f", F(z["tp"], z["fp"], z["n"])))
    return {c: float(z.filter(pl.col("country") == c)["f"].mean()) for c in ("US", "India")}, a


# V12-equivalent holdout decisions (G decoding, alpha 0.75 US/India) as the reference
def gdec(x, alpha):
    x = score_col(x, alpha)
    return decide(prep(x.select("s1", "m", "p1", "q").rename({"q": "qq"}), TAU, "qq"), dec).select("s1", "m")


ref_h = pl.concat([gdec(h.filter(pl.col("country") == c), 0.75) for c in ("US", "India")])
fref, aref = hscore(ref_h)
print(f"HOLDOUT clean F, reference (V12 logic: G expected-F, alpha .75): US {fref['US']:.5f} India {fref['India']:.5f}")
H = {}
for alpha in (1.0, 0.75):
    hx = pl.concat([score_col(h.filter(pl.col("country") == c), alpha) for c in ("US", "India")])
    for t in TS:
        f, a = hscore(thr(hx, t)); H[(alpha, t)] = a
        print(f"  threshold q>={t:<5} (q = {'p3' if alpha == 1 else 'mix .75'}): US {f['US']:.5f} ({f['US'] - fref['US']:+.5f})  India {f['India']:.5f} ({f['India'] - fref['India']:+.5f})", flush=True)
# ---- test side
q = lambda p: duckdb.connect().execute(f"select source1_entity_id s1, trim(unnest(string_split(matched_entity_ids, ','))) m from read_csv('{p}', delim='\t', header=true, all_varchar=true, quote='') "
                                       "where matched_entity_ids is not null and matched_entity_ids<>''").pl()
V12 = q("output_v12/matching_results.tsv")
nt = dict(pl.read_parquet("work/test_s1.parquet", columns=["country"]).group_by("country").len().rows())
fr = pl.read_parquet("work/fr_residual.parquet", columns=["s1", "m", "fp_shared", "ra"])
print("\nTEST: accepted per 1k S1 and what a threshold REMOVES vs V12 (per 1k S1)")
T = {}
for c in ("US", "India", "France"):
    x = score_col(pl.read_parquet(f"{S3D}/p3/test/{c}.parquet", columns=["s1", "m", "p1", "p3"]), 1.0)
    v = V12.join(x.select("s1").unique(), on="s1", how="semi")
    for t in (0.8, 0.9, 0.92, 0.95):
        a = thr(x, t); T[(c, t)] = a
        rem = annotate(v.join(a, on=["s1", "m"], how="anti"), "test")
        pk = rem.filter(pl.col("off") == "+1..9").height; mk = rem.filter(pl.col("off") == "-1..9").height
        fx = rem.join(fr, on=["s1", "m"], how="left").filter(pl.col("ra").fill_null("") != "")
        top = rem.group_by("cls").len().sort("len", descending=True).head(3).rows()
        print(f"  {c:6s} t={t}: accepted {a.height / nt[c] * 1000:6.0f}/1k (V12 {v.height / nt[c] * 1000:.0f}); removes {rem.height / nt[c] * 1000:5.1f}/1k "
              f"[+1..9 {pk / nt[c] * 1000:.1f} vs -1..9 {mk / nt[c] * 1000:.1f}]" + (f" French fingerprint of removed {fx['fp_shared'].mean():.3f}" if c == "France" and fx.height else "")
              + f" top classes {[(k, round(n / nt[c] * 1000, 1)) for k, n in top]}", flush=True)
# holdout-referenced value of threshold(t) vs V12 on test (US/India), 211 method
print("\nHOLDOUT-REFERENCED test value of threshold(q=p3) vs V12 (country F):")
for t in (0.8, 0.9, 0.92, 0.95):
    out = []
    for c in ("US", "India"):
        hr = annotate(aref.filter(pl.col("s1").is_in(gt.filter(pl.col("country") == c)["s1"].to_list())).select("s1", "m", "y").join(H[(1.0, t)].select("s1", "m"), on=["s1", "m"], how="anti"), "train")
        hd = annotate(H[(1.0, t)].filter(pl.col("s1").is_in(gt.filter(pl.col("country") == c)["s1"].to_list())).select("s1", "m", "y").join(aref.select("s1", "m"), on=["s1", "m"], how="anti"), "train")
        k = 1000 / nh[c]
        v = V12.join(T[(c, t)].select("s1").unique(), on="s1", how="semi")
        tr_ = annotate(v.join(T[(c, t)], on=["s1", "m"], how="anti"), "test").group_by("cls", "off").len().with_columns(pl.col("len") / nt[c] * 1000).rename({"len": "rem_t"})
        td_ = annotate(T[(c, t)].join(v, on=["s1", "m"], how="anti"), "test").group_by("cls", "off").len().with_columns(pl.col("len") / nt[c] * 1000).rename({"len": "add_t"})
        hr_ = hr.group_by("cls", "off").agg((pl.col("y").sum() * k).alias("remTP"), ((~pl.col("y")).sum() * k).alias("remFP"))
        hd_ = hd.group_by("cls", "off").agg((pl.col("y").sum() * k).alias("addTP"), ((~pl.col("y")).sum() * k).alias("addFP"))
        z = tr_.join(td_, on=["cls", "off"], how="full", coalesce=True).join(hr_, on=["cls", "off"], how="full", coalesce=True).join(hd_, on=["cls", "off"], how="full", coalesce=True).fill_null(0)
        z = z.with_columns((pl.col("rem_t") - pl.col("remTP") - pl.col("remFP")).alias("rex"), (pl.col("add_t") - pl.col("addTP") - pl.col("addFP")).alias("aex"))
        val = float(z.select(0.21 * pl.col("remFP") - 0.09 * pl.col("remTP") + 0.21 * pl.col("rex").clip(0) - 0.09 * (-pl.col("rex")).clip(0)
                             + 0.09 * pl.col("addTP") - 0.21 * pl.col("addFP") - 0.21 * pl.col("aex").clip(0)).sum().item()) / 1000
        out.append(f"{c} {val:+.5f}")
    print(f"  t={t}: " + "  ".join(out), flush=True)
for t in (0.9, 0.92):
    pl.concat([T[(c, t)] for c in ("US", "India", "France")]).write_parquet(f"work/thr_{t}.parquet")
