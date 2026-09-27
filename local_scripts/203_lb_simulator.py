"""LB-consistent test simulator. Truth model q(s1,m) = V9 p3 on the tau-set, V8 p2x2 for 1e-4<=p1<0.05 candidates, 0 otherwise;
shift model: odds(q) * w (w<1 = more look-alike negatives on test than train). For every submitted version (v2, V3, V5, V5s3r, V9)
compute the EXACT expected macro F0.5 (Poisson-binomial over accepted / rejected candidates) and compare with its leaderboard score.
Candidates-only (missed-retrieval loss is a common offset, ~0.003 on holdout)."""
import sys, json
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl, numpy as np, duckdb
VERS = {"v2": ("output", 0.936), "V3": ("output_v3_auto", 0.966), "V5": ("output_v5", 0.967), "V5s3r": ("output_v5s3r", 0.973), "V9": ("output_v9", 0.977)}
WS = [1.0, 0.7, 0.5, 0.35, 0.25, 0.15, 0.1, 0.05]
KA, KR = 14, 16
q2 = pl.scan_parquet("work/v8_p2/test/*.parquet").select("s1", "m", "country", pl.col("p2x2").alias("q2")).collect()
q3 = pl.concat([pl.read_parquet(f"output_v9/p3/test/{c}.parquet", columns=["s1", "m", "p3"]) for c in ("US", "India", "France")])
Q = q2.join(q3, on=["s1", "m"], how="left").with_columns(pl.coalesce("p3", "q2").clip(0, 1).alias("q")).select("s1", "m", "country", "q"); del q2, q3
S1 = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
S1 = S1.with_row_index("i"); NS = S1.height
cidx = {c: (S1["country"] == c).to_numpy() for c in ("US", "India", "France")}
print(f"test S1 {NS}; candidate pairs in truth model {Q.height}", flush=True)


def mat(df, K):
    """(s1 index, q) rows -> dense (NS, K) matrix of the K largest q per S1 (zeros = padding) + count per S1."""
    d = df.sort(["i", "q"], descending=[False, True]).with_columns(pl.int_range(pl.len()).over("i").alias("j")).filter(pl.col("j") < K)
    M = np.zeros((NS, K), dtype=np.float64); M[d["i"].to_numpy(), d["j"].to_numpy()] = d["q"].to_numpy()
    return M


def pb(P):
    S, K = P.shape; D = np.zeros((S, K + 1)); D[:, 0] = 1.0
    for j in range(K):
        p = P[:, j:j + 1]; N = D * (1 - p); N[:, 1:] += D[:, :-1] * p; D = N
    return D


def expected_f(A, k, R):
    DA, DR = pb(A), pb(R); E = np.zeros(A.shape[0])
    for a in range(DA.shape[1]):
        for b in range(DR.shape[1]):
            if a == 0:   # no true positive: F = 1 only for an empty prediction on an S1 with no true candidate
                f = ((k == 0) & (b == 0)).astype(float)
            else:
                P = a / np.maximum(k, a); R_ = a / (a + b)
                f = 1.25 * P * R_ / (0.25 * P + R_)
            E += DA[:, a] * DR[:, b] * f
    return E


def odds(M, w):
    return np.where(M > 0, w * M / (w * M + 1 - M + 1e-12), 0.0)


res = {}
for v, (d, lb) in VERS.items():
    acc = duckdb.connect().execute(f"select source1_entity_id s1, unnest(string_split(matched_entity_ids, ',')) m from read_csv('{d}/matching_results.tsv', delim='\t', "
                                   "header=true, all_varchar=true, quote='') where matched_entity_ids is not null and matched_entity_ids<>''").pl()
    a = acc.join(Q.select("s1", "m", "q"), on=["s1", "m"], how="left").with_columns(pl.col("q").fill_null(0.0)).join(S1.select("s1", "i"), on="s1")
    r = Q.join(acc, on=["s1", "m"], how="anti").filter(pl.col("q") >= 1e-3).join(S1.select("s1", "i"), on="s1")
    k = np.zeros(NS); ki = a.group_by("i").len(); k[ki["i"].to_numpy()] = ki["len"].to_numpy()
    k = np.minimum(k, KA)
    A0, R0 = mat(a.select("i", "q"), KA), mat(r.select("i", "q"), KR)
    row = {}
    for w in WS:
        E = expected_f(odds(A0, w), k, odds(R0, w))
        row[w] = (float(E.mean()), {c: float(E[m].mean()) for c, m in cidx.items()})
    res[v] = row
    print(f"{v:6s} LB {lb:.3f} | accepted {acc.height} (unscored {int((a['q'] == 0).sum())}) | " + " ".join(f"w={w}: {row[w][0]:.4f}" for w in WS), flush=True)
print("\nPredicted minus LB (after a common -0.003 missed-retrieval offset):")
for w in WS:
    print(f"  w={w:<5}" + "  ".join(f"{v} {res[v][w][0] - 0.003 - VERS[v][1]:+.4f}" for v in VERS) +
          f"   | spread {np.ptp([res[v][w][0] - VERS[v][1] for v in VERS]):.4f}")
json.dump({v: {str(w): res[v][w] for w in WS} for v in res}, open("analysis/out/203_lb_simulator.json", "w"), indent=1)
