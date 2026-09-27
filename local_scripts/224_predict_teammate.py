"""Predict a submission's LB score as V9's known LB (0.977) + the expected value of its DIFFERENCES from V9 (only changed S1 evaluated,
so simulator errors that are common to both cancel). Truth models from the LB-history fit (204): q = alpha*p3(V9) + (1-alpha)*iso(p1),
alpha in {1, .75, .5, .25, 0} (the recent LB deltas support alpha <= 0.5). Also characterizes the change set.
usage: 224_predict_teammate.py <matching_results.tsv> [label]"""
import sys, json
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl, numpy as np, duckdb
from sklearn.isotonic import IsotonicRegression
sys.path.insert(0, "scripts"); from simlib import KA, KR, expected_f, odds
exec(open("scripts/213_excess_profile.py").read().split("t1 = pl.read_parquet")[0])       # annotate()
TSV = sys.argv[1]; LABEL = sys.argv[2] if len(sys.argv) > 2 else "teammate"
con = duckdb.connect()
q = lambda p: con.execute(f"select source1_entity_id s1, trim(unnest(string_split(matched_entity_ids, ','))) m from read_csv('{p}', delim='\t', header=true, all_varchar=true, quote='') "
                          "where matched_entity_ids is not null and matched_entity_ids<>''").pl()
T, V9 = q(TSV), q("output_v9/matching_results.tsv")
s1 = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"}); nt = dict(s1.group_by("country").len().rows())
add, rem = T.join(V9, on=["s1", "m"], how="anti"), V9.join(T, on=["s1", "m"], how="anti")
changed = pl.concat([add.select("s1"), rem.select("s1")]).unique()
SYN = pl.concat([pl.read_parquet(f"work/aug_v8_{s}.parquet", columns=["entity_id"]) for s in ("s2", "s3")]).rename({"entity_id": "m"})
tr = pl.scan_parquet("work/v8_p2/train/*.parquet").filter(pl.col("fold").is_in([0, 8, 9])).select("m", "y", "p1").collect().join(SYN, on="m", how="anti")
iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(tr["p1"].to_numpy(), tr["y"].to_numpy()); del tr
q3 = pl.concat([pl.read_parquet(f"output_v9/p3/test/{c}.parquet", columns=["s1", "m", "p3"]) for c in ("US", "India", "France")]).join(changed, on="s1", how="semi")
Q = pl.scan_parquet("work/v8_p2/test/*.parquet").select("s1", "m", "p1", pl.col("p2x2").alias("q2")).collect().join(changed, on="s1", how="semi")
Q = Q.with_columns(pl.Series("qp", iso.predict(Q["p1"].to_numpy())))
Q = Q.join(q3, on=["s1", "m"], how="left").with_columns(pl.coalesce("p3", "q2").clip(0, 1).alias("q")).select("s1", "m", "q", "qp", "p3")
v9empty = s1.join(V9.select("s1").unique(), on="s1", how="anti").select("s1")
print(f"{LABEL} vs V9: added {add.height} pairs, removed {rem.height}, changed S1 {changed.height} ({changed.height / s1.height:.2%})")
for c in ("US", "India", "France"):
    a = add.join(s1, on="s1").filter(pl.col("country") == c); r = rem.join(s1, on="s1").filter(pl.col("country") == c)
    a_empty = a.join(v9empty, on="s1", how="semi")
    ap3, rp3 = a.join(Q, on=["s1", "m"], how="left")["p3"], r.join(Q, on=["s1", "m"], how="left")["p3"]
    print(f"  {c:6s}: +{a.height / nt[c] * 1000:.1f}/1k (of which on S1 that V9 left EMPTY: {a_empty['s1'].n_unique() / nt[c] * 1000:.1f} S1/1k), mean V9 p3 of added {ap3.mean() or 0:.3f} "
          f"(unscored {ap3.null_count()}); -{r.height / nt[c] * 1000:.1f}/1k, mean p3 of removed {rp3.mean() or 0:.3f}")
    if a.height:
        cl = annotate(a.select("s1", "m"), "test").group_by("cls").len().sort("len", descending=True).head(5)
        print("      added by class:", [(k, round(v / nt[c] * 1000, 2)) for k, v in cl.rows()])
S = changed.join(s1, on="s1").with_row_index("i"); NS = S.height


def mats(acc):
    a = acc.join(S, on="s1").join(Q, on=["s1", "m"], how="left").with_columns(pl.col("q").fill_null(0.0), pl.col("qp").fill_null(0.0))
    r = Q.join(S, on="s1").join(acc, on=["s1", "m"], how="anti").filter((pl.col("q") >= 1e-3) | (pl.col("qp") >= 1e-3))
    def M(df, K):
        d = df.sort(["i", "q"], descending=[False, True]).with_columns(pl.int_range(pl.len()).over("i").alias("j")).filter(pl.col("j") < K)
        A = np.zeros((NS, K)); B = np.zeros((NS, K)); A[d["i"].to_numpy(), d["j"].to_numpy()] = d["q"].to_numpy(); B[d["i"].to_numpy(), d["j"].to_numpy()] = d["qp"].to_numpy()
        return A, B
    k = np.zeros(NS); ki = a.group_by("i").len(); k[ki["i"].to_numpy()] = np.minimum(ki["len"].to_numpy(), KA)
    return (k, *M(a, KA), *M(r, KR))


mT, m9 = mats(T.join(changed, on="s1", how="semi")), mats(V9.join(changed, on="s1", how="semi"))
cm = {c: (S["country"] == c).to_numpy() for c in ("US", "India", "France")}
out = {}
print(f"\nExpected LB change vs V9 (only changed S1; per truth model; the LB history supports alpha <= 0.5):")
for al in (1.0, 0.75, 0.5, 0.25, 0.0):
    eT = expected_f(odds(al * mT[1] + (1 - al) * mT[2], 1.0), mT[0], odds(al * mT[3] + (1 - al) * mT[4], 1.0))
    e9 = expected_f(odds(al * m9[1] + (1 - al) * m9[2], 1.0), m9[0], odds(al * m9[3] + (1 - al) * m9[4], 1.0))
    d = eT - e9
    per = {c: float(d[cm[c]].sum() / nt[c]) for c in cm}
    tot = float(d.sum() / s1.height)
    out[al] = dict(total=tot, **per)
    print(f"  alpha={al:<5} total {tot:+.5f}  | US {per['US']:+.5f}  India {per['India']:+.5f}  France {per['France']:+.5f}  -> predicted LB {0.977 + tot:.4f}")
json.dump({str(k): v for k, v in out.items()}, open(f"analysis/out/224_predict_{LABEL}.json", "w"), indent=1)
