"""Country-wise decision audit for the stage-3/pairwise mix decoder (memory-lean, one country at a time).
For alpha_d in a grid: labelled holdout F per country (US/India, folds 0/8/9 clean) and exact expected test F per country
under truth models alpha_t in {1, 0.5, 0.25} (w=1; alpha_t=1 = V9 is right, <=0.5 = what the LB history supports).
Also: what the alpha_d=0.5 decoder changes vs V9 (per 1k S1 by relation class)."""
import sys, json, gc
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl, numpy as np, duckdb
from sklearn.isotonic import IsotonicRegression
from tune_decision import prep, decide
sys.path.insert(0, "scripts"); from simlib import KA, KR, expected_f, odds
d9 = json.load(open("output_v9/decision_s3.json")); TAU = d9["tau"]
AD = [1.0, 0.75, 0.6, 0.5, 0.4, 0.3]; AT = [1.0, 0.5, 0.25]
SYN = pl.concat([pl.read_parquet(f"work/aug_v8_{s}.parquet", columns=["entity_id"]) for s in ("s2", "s3")]).rename({"entity_id": "m"})
tr = pl.scan_parquet("work/v8_p2/train/*.parquet").filter(pl.col("fold").is_in([0, 8, 9])).select("m", "y", "p1").collect().join(SYN, on="m", how="anti")
iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(tr["p1"].to_numpy(), tr["y"].to_numpy()); del tr; gc.collect()


def decode(df, alpha):
    df = df.with_columns(pl.Series("ip1", iso.predict(df["p1"].to_numpy())).cast(pl.Float32))
    df = df.with_columns((alpha * pl.col("p3") + (1 - alpha) * pl.col("ip1")).alias("qa"))
    return decide(prep(df, TAU, "qa"), d9).select("s1", "m")


def F(tp, fp, n):
    tp, fp, n = (np.asarray(v, float) for v in (tp, fp, n))
    P = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-9), 0); R = np.where(n > 0, tp / np.maximum(n, 1), 0)
    return np.where(n == 0, (tp + fp == 0).astype(float), np.where(tp > 0, 1.25 * P * R / np.maximum(0.25 * P + R, 1e-12), 0.0))


# ---- holdout per country
gt = duckdb.connect().execute("select source1_entity_id s1, case when matched_entity_ids is null or matched_entity_ids='' then 0 else len(string_split(matched_entity_ids, ',')) end n, "
                              "hash(source1_entity_id || 'fold') % 10 fold from 'work/train_gt.parquet'").pl().join(pl.read_parquet("work/v5_dropped_s1.parquet").select("s1"), on="s1", how="anti")
gt = gt.filter(pl.col("fold").is_in([0, 8, 9])).join(pl.read_parquet("work/train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"}), on="s1")
x = pl.concat([pl.read_parquet(f"output_v9/p3/train/fold{f}.parquet", columns=["s1", "m", "y", "p1", "p3"]) for f in (0, 8, 9)]).join(SYN, on="m", how="anti")
print("HOLDOUT clean F by country (mean of folds 0/8/9), delta vs alpha_d=1 (V9):")
hbase = {}
for ad in AD:
    a = decode(x, ad).join(x.select("s1", "m", "y"), on=["s1", "m"]).group_by("s1").agg(pl.col("y").sum().alias("tp"), (~pl.col("y")).sum().alias("fp"))
    z = gt.join(a, on="s1", how="left").fill_null(0)
    z = z.with_columns(pl.Series("f", F(z["tp"], z["fp"], z["n"])))
    r = {c: float(z.filter(pl.col("country") == c).group_by("fold").agg(pl.col("f").mean())["f"].mean()) for c in ("US", "India")}
    hbase.setdefault("base", r) if ad == 1.0 else None
    print(f"  alpha_d={ad:<4}  US {r['US']:.5f} ({r['US'] - hbase['base']['US']:+.5f})  India {r['India']:.5f} ({r['India'] - hbase['base']['India']:+.5f})", flush=True)
del x; gc.collect()
# ---- test per country
s1t = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
res = {}
for c in ("US", "India", "France"):
    S1 = s1t.filter(pl.col("country") == c).select("s1").with_row_index("i"); NS = S1.height
    te = pl.read_parquet(f"output_v9/p3/test/{c}.parquet", columns=["s1", "m", "p1", "p3"])
    Q = pl.scan_parquet("work/v8_p2/test/*.parquet").filter(pl.col("country") == c).select("s1", "m", "p1", pl.col("p2x2").alias("q2")).collect()
    Q = Q.with_columns(pl.Series("qp", iso.predict(Q["p1"].to_numpy())).cast(pl.Float32))
    Q = Q.join(te.select("s1", "m", "p3"), on=["s1", "m"], how="left").with_columns(pl.coalesce("p3", "q2").clip(0, 1).cast(pl.Float32).alias("q")).select("s1", "m", "q", "qp")
    cls = pl.read_parquet(f"work/copy_channel_test_{c}.parquet", columns=["s1", "m", "cls"])

    def mat2(df, K):
        d = df.sort(["i", "q"], descending=[False, True]).with_columns(pl.int_range(pl.len()).over("i").alias("j")).filter(pl.col("j") < K)
        M = np.zeros((NS, K), np.float32); Mp = np.zeros((NS, K), np.float32); ii, jj = d["i"].to_numpy(), d["j"].to_numpy()
        M[ii, jj] = d["q"].to_numpy(); Mp[ii, jj] = d["qp"].to_numpy()
        return M.astype(np.float64), Mp.astype(np.float64)
    v9acc = None
    for ad in AD:
        acc = decode(te, ad)
        if ad == 1.0:
            v9acc = acc
        a = acc.join(Q, on=["s1", "m"], how="left").fill_null(0.0).join(S1, on="s1")
        r = Q.join(acc, on=["s1", "m"], how="anti").filter((pl.col("q") >= 1e-3) | (pl.col("qp") >= 1e-3)).join(S1, on="s1")
        k = np.zeros(NS); ki = a.group_by("i").len(); k[ki["i"].to_numpy()] = np.minimum(ki["len"].to_numpy(), KA)
        A, Ap = mat2(a.select("i", "q", "qp"), KA); R, Rp = mat2(r.select("i", "q", "qp"), KR)
        res[(c, ad)] = {at: float(expected_f(odds(at * A + (1 - at) * Ap, 1.0), k, odds(at * R + (1 - at) * Rp, 1.0)).mean()) for at in AT}
        line = " | ".join(f"alpha_t={at}: {res[(c, ad)][at]:.4f} ({res[(c, ad)][at] - res[(c, 1.0)][at]:+.4f})" for at in AT)
        print(f"TEST {c:6s} alpha_d={ad:<4} acc/1k {1000 * acc.height / NS:7.1f} | {line}", flush=True)
        if ad == 0.5:
            rem = v9acc.join(acc, on=["s1", "m"], how="anti").join(cls, on=["s1", "m"], how="left")
            add = acc.join(v9acc, on=["s1", "m"], how="anti").join(cls, on=["s1", "m"], how="left")
            print(f"   alpha_d=0.5 vs V9: removes {1000 * rem.height / NS:.1f}/1k, adds {1000 * add.height / NS:.1f}/1k; removed by class:",
                  rem.group_by("cls").len().sort("len", descending=True).with_columns((pl.col("len") / NS * 1000).round(2)).head(8).rows())
            rem.write_parquet(f"work/mix05_removed_{c}.parquet"); add.write_parquet(f"work/mix05_added_{c}.parquet")
        acc.write_parquet(f"work/mix_acc_{c}_{ad}.parquet")
        del a, r, A, Ap, R, Rp; gc.collect()
    del te, Q, cls; gc.collect()
w = {"US": 663106, "India": 809986, "France": 259452}; W = sum(w.values())
print("\nWEIGHTED over countries (delta vs V9):")
for ad in AD:
    print(f"  alpha_d={ad:<4} " + " | ".join(f"alpha_t={at}: {sum(w[c] * (res[(c, ad)][at] - res[(c, 1.0)][at]) for c in w) / W:+.5f}" for at in AT))
json.dump({f"{c}|{ad}": v for (c, ad), v in res.items()}, open("analysis/out/206_mix_country.json", "w"), indent=1)
