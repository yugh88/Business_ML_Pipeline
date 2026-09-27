"""Which truth model reproduces the LB history? q_alpha = alpha * p3(V9, collective/stage-3) + (1-alpha) * iso(p1) (pairwise stage-1,
no collective features; isotonic-calibrated on the clean holdout), then odds shift w. If alpha<1 fits the 5 LB points better,
the test's look-alike clusters fool the collective evidence (blind loss sits in 'promoted' pairs)."""
import sys, json
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl, numpy as np, duckdb
from sklearn.isotonic import IsotonicRegression
sys.path.insert(0, "scripts"); from simlib import KA, KR, expected_f, odds
SYN = pl.concat([pl.read_parquet(f"work/aug_v8_{s}.parquet", columns=["entity_id"]) for s in ("s2", "s3")]).rename({"entity_id": "m"})
tr = pl.scan_parquet("work/v8_p2/train/*.parquet").filter(pl.col("fold").is_in([0, 8, 9])).select("m", "y", "p1").collect().join(SYN, on="m", how="anti")
iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(tr["p1"].to_numpy(), tr["y"].to_numpy()); del tr
q2 = pl.scan_parquet("work/v8_p2/test/*.parquet").select("s1", "m", "p1", pl.col("p2x2").alias("q2")).collect()
q2 = q2.with_columns(pl.Series("qp", iso.predict(q2["p1"].to_numpy())))            # attach before any join (join order not guaranteed)
q3 = pl.concat([pl.read_parquet(f"output_v9/p3/test/{c}.parquet", columns=["s1", "m", "p3"]) for c in ("US", "India", "France")])
Q = q2.join(q3, on=["s1", "m"], how="left").with_columns(pl.coalesce("p3", "q2").clip(0, 1).alias("q")).select("s1", "m", "q", "qp"); del q2, q3
S1 = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"}).with_row_index("i"); NS = S1.height
VERS = {"v2": ("output", 0.936), "V3": ("output_v3_auto", 0.966), "V5": ("output_v5", 0.967), "V5s3r": ("output_v5s3r", 0.973), "V9": ("output_v9", 0.977)}


def mat2(df, K):
    d = df.sort(["i", "q"], descending=[False, True]).with_columns(pl.int_range(pl.len()).over("i").alias("j")).filter(pl.col("j") < K)
    M = np.zeros((NS, K)); Mp = np.zeros((NS, K)); ii, jj = d["i"].to_numpy(), d["j"].to_numpy()
    M[ii, jj] = d["q"].to_numpy(); Mp[ii, jj] = d["qp"].to_numpy()
    return M, Mp


cache = {}
for v, (d, lb) in VERS.items():
    acc = duckdb.connect().execute(f"select source1_entity_id s1, unnest(string_split(matched_entity_ids, ',')) m from read_csv('{d}/matching_results.tsv', delim='\t', "
                                   "header=true, all_varchar=true, quote='') where matched_entity_ids is not null and matched_entity_ids<>''").pl()
    a = acc.join(Q, on=["s1", "m"], how="left").with_columns(pl.col("q").fill_null(0.0), pl.col("qp").fill_null(0.0)).join(S1.select("s1", "i"), on="s1")
    r = Q.join(acc, on=["s1", "m"], how="anti").filter((pl.col("q") >= 1e-3) | (pl.col("qp") >= 1e-3)).join(S1.select("s1", "i"), on="s1")
    k = np.zeros(NS); ki = a.group_by("i").len(); k[ki["i"].to_numpy()] = np.minimum(ki["len"].to_numpy(), KA)
    cache[v] = (k, *mat2(a.select("i", "q", "qp"), KA), *mat2(r.select("i", "q", "qp"), KR))
    print("cached", v, flush=True)
del Q
print("predicted E[F] - 0.003 - LB   (alpha = weight of V9 p3 vs calibrated pairwise p1; w = odds shift)")
for alpha in (1.0, 0.75, 0.5, 0.25, 0.0):
    for w in (1.0, 0.7, 0.5):
        diffs = []
        for v, (d, lb) in VERS.items():
            k, A, Ap, R, Rp = cache[v]
            E = expected_f(odds(alpha * A + (1 - alpha) * Ap, w), k, odds(alpha * R + (1 - alpha) * Rp, w))
            diffs.append(float(E.mean()) - 0.003 - lb)
        print(f"  alpha={alpha:<4} w={w:<4} " + "  ".join(f"{v} {x:+.4f}" for v, x in zip(VERS, diffs)) + f"   | spread {np.ptp(diffs):.4f}  mean|err| {np.mean(np.abs(diffs)):.4f}", flush=True)
