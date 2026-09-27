"""Decoder candidates evaluated two ways: (1) labelled clean holdout F (folds 0/8/9) = cost on train-like data,
(2) exact expected test F under truth models q(alpha_t, w) fitted to the LB history (204). Decoders:
 V9 | mix(a): G on alpha*p3+(1-alpha)*iso(p1) | veto(t): V9 minus pairs with iso(p1)<t | cons: G with lam=0.15 / a=1.0."""
import sys, json
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl, numpy as np, duckdb
from sklearn.isotonic import IsotonicRegression
from tune_decision import prep, decide
sys.path.insert(0, "scripts"); from simlib import KA, KR, expected_f, odds
d9 = json.load(open("output_v9/decision_s3.json")); TAU = d9["tau"]
TRUTH = [(float(a), float(w)) for a, w in (x.split(":") for x in (sys.argv[1] if len(sys.argv) > 1 else "1:1,0.5:1,0.5:0.7").split(","))]
SYN = pl.concat([pl.read_parquet(f"work/aug_v8_{s}.parquet", columns=["entity_id"]) for s in ("s2", "s3")]).rename({"entity_id": "m"})
tr = pl.scan_parquet("work/v8_p2/train/*.parquet").filter(pl.col("fold").is_in([0, 8, 9])).select("m", "y", "p1").collect().join(SYN, on="m", how="anti")
iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(tr["p1"].to_numpy(), tr["y"].to_numpy()); del tr
DEC = {"V9": ("p3", {}), "mix0.75": ("mix", {"alpha": 0.75}), "mix0.5": ("mix", {"alpha": 0.5}), "veto0.02": ("veto", {"t": 0.02}),
       "veto0.1": ("veto", {"t": 0.1}), "cons_lam0.15": ("p3", {"lam": 0.15}), "cons_a1.0": ("p3", {"a": 1.0})}


def decode(df, kind, kw):
    df = df.with_columns(pl.Series("ip1", iso.predict(df["p1"].to_numpy())).cast(pl.Float32))
    c = dict(d9); c.update({k: v for k, v in kw.items() if k in ("a", "lam")})
    if kind == "mix":
        df = df.with_columns((kw["alpha"] * pl.col("p3") + (1 - kw["alpha"]) * pl.col("ip1")).alias("qa"))
        return decide(prep(df, TAU, "qa"), c).select("s1", "m")
    acc = decide(prep(df, TAU, "p3"), c).select("s1", "m")
    if kind == "veto":
        acc = acc.join(df.select("s1", "m", "ip1"), on=["s1", "m"]).filter(pl.col("ip1") >= kw["t"]).select("s1", "m")
    return acc


def F(tp, fp, n):
    tp, fp, n = (np.asarray(v, float) for v in (tp, fp, n))
    P = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-9), 0); R = np.where(n > 0, tp / np.maximum(n, 1), 0)
    return np.where(n == 0, (tp + fp == 0).astype(float), np.where(tp > 0, 1.25 * P * R / np.maximum(0.25 * P + R, 1e-12), 0.0))


# (1) holdout
gt = duckdb.connect().execute("select source1_entity_id s1, case when matched_entity_ids is null or matched_entity_ids='' then 0 else len(string_split(matched_entity_ids, ',')) end n, "
                              "hash(source1_entity_id || 'fold') % 10 fold from 'work/train_gt.parquet'").pl().join(pl.read_parquet("work/v5_dropped_s1.parquet").select("s1"), on="s1", how="anti")
hold = {}
x = pl.concat([pl.read_parquet(f"output_v9/p3/train/fold{f}.parquet", columns=["s1", "m", "y", "p1", "p3"]) for f in (0, 8, 9)]).join(SYN, on="m", how="anti")
for name, (kind, kw) in DEC.items():
    a = decode(x, kind, kw).join(x.select("s1", "m", "y"), on=["s1", "m"]).group_by("s1").agg(pl.col("y").sum().alias("tp"), (~pl.col("y")).sum().alias("fp"))
    for f in (0, 8, 9):
        z = gt.filter(pl.col("fold") == f).join(a, on="s1", how="left").fill_null(0)
        hold.setdefault(name, []).append(float(F(z["tp"], z["fp"], z["n"]).mean()))
print("HOLDOUT clean F (folds 0/8/9):")
for k, v in hold.items():
    print(f"  {k:14s} " + " ".join(f"{x:.5f}" for x in v) + f"   mean {np.mean(v):.5f} ({np.mean(v) - np.mean(hold['V9']):+.5f})", flush=True)
# (2) test under truth models
te = pl.concat([pl.read_parquet(f"output_v9/p3/test/{c}.parquet", columns=["s1", "m", "p1", "p3"]) for c in ("US", "India", "France")])
q2 = pl.scan_parquet("work/v8_p2/test/*.parquet").select("s1", "m", "p1", pl.col("p2x2").alias("q2")).collect()
q2 = q2.with_columns(pl.Series("qp", iso.predict(q2["p1"].to_numpy())))
Q = q2.join(te.select("s1", "m", "p3"), on=["s1", "m"], how="left").with_columns(pl.coalesce("p3", "q2").clip(0, 1).alias("q")).select("s1", "m", "q", "qp"); del q2
S1 = pl.read_parquet("work/test_s1.parquet", columns=["entity_id"]).rename({"entity_id": "s1"}).with_row_index("i"); NS = S1.height


def mat2(df, K):
    d = df.sort(["i", "q"], descending=[False, True]).with_columns(pl.int_range(pl.len()).over("i").alias("j")).filter(pl.col("j") < K)
    M = np.zeros((NS, K)); Mp = np.zeros((NS, K)); ii, jj = d["i"].to_numpy(), d["j"].to_numpy()
    M[ii, jj] = d["q"].to_numpy(); Mp[ii, jj] = d["qp"].to_numpy()
    return M, Mp


print("\nTEST expected F under truth models (alpha_t, w) — minus 0.003 retrieval offset; accepted pairs per 1k S1")
base = {}
for name, (kind, kw) in DEC.items():
    acc = decode(te, kind, kw)
    a = acc.join(Q, on=["s1", "m"], how="left").fill_null(0.0).join(S1, on="s1")
    r = Q.join(acc, on=["s1", "m"], how="anti").filter((pl.col("q") >= 1e-3) | (pl.col("qp") >= 1e-3)).join(S1, on="s1")
    k = np.zeros(NS); ki = a.group_by("i").len(); k[ki["i"].to_numpy()] = np.minimum(ki["len"].to_numpy(), KA)
    A, Ap = mat2(a.select("i", "q", "qp"), KA); R, Rp = mat2(r.select("i", "q", "qp"), KR)
    out = []
    for at, w in TRUTH:
        e = float(expected_f(odds(at * A + (1 - at) * Ap, w), k, odds(at * R + (1 - at) * Rp, w)).mean()) - 0.003
        base.setdefault((at, w), e) if name == "V9" else None
        out.append(f"({at},{w}) {e:.4f} [{e - base[(at, w)]:+.4f}]")
    print(f"  {name:14s} acc/1k {1000 * acc.height / NS:7.1f} | " + " | ".join(out), flush=True)
    acc.write_parquet(f"work/dec_{name}.parquet")
