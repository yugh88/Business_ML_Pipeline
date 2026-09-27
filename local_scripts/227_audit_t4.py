"""Independent audit of matching_results (4).tsv (T4 = V9 + 12,917 pairs). Test-only file -> labelled validation = replay of each
addition TYPE on the holdout (V9-rejected candidates of that type; same selection space), with P/R/F per country for V9 vs V9+type.
Test side: fingerprint / generic-name / empty-S1 evidence per type, simulator deltas under several truth models (incl. teammate-
favourable), hard best/worst-case bounds, and hybrids on top of V10."""
import sys, json
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl, numpy as np, duckdb
from sklearn.isotonic import IsotonicRegression
sys.path.insert(0, "scripts"); from simlib import KA, KR, expected_f, odds
exec(open("scripts/213_excess_profile.py").read().split("t1 = pl.read_parquet")[0])        # annotate(): cls, off
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(230); pl.Config.set_tbl_hide_dataframe_shape(True)
con = duckdb.connect()
q = lambda p: con.execute(f"select source1_entity_id s1, trim(unnest(string_split(matched_entity_ids, ','))) m from read_csv('{p}', delim='\t', header=true, all_varchar=true, quote='') "
                          "where matched_entity_ids is not null and matched_entity_ids<>''").pl()


def F(tp, fp, n):
    tp, fp, n = (np.asarray(v, float) for v in (tp, fp, n))
    P = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-9), 0); R = np.where(n > 0, tp / np.maximum(n, 1), 0)
    return np.where(n == 0, (tp + fp == 0).astype(float), np.where(tp > 0, 1.25 * P * R / np.maximum(0.25 * P + R, 1e-12), 0.0))


s1t = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"}); nt = dict(s1t.group_by("country").len().rows()); NT = s1t.height
V9, V10, T4 = q("output_v9/matching_results.tsv"), q("output_v10/matching_results.tsv"), q("matching_results (4).tsv")
add = annotate(T4.join(V9, on=["s1", "m"], how="anti"), "test").join(s1t, on="s1")
v9empty = s1t.join(V9.select("s1").unique(), on="s1", how="anti").select("s1")
ours = pl.scan_parquet("work/v8_p2/test/*.parquet").select("s1", "m").collect().with_columns(pl.lit(True).alias("ours"))
p3 = pl.concat([pl.read_parquet(f"output_v9/p3/test/{c}.parquet", columns=["s1", "m", "p3"]) for c in ("US", "India", "France")])
fr = pl.read_parquet("work/fr_residual.parquet", columns=["s1", "m", "fp_shared", "ra"])
t1c = pl.scan_parquet("work/test_s1_norm.parquet").select(pl.col("entity_id").alias("s1"), "ncore").collect()
cc = t1c.group_by("ncore").len().rename({"len": "name_share"})
add = add.join(ours, on=["s1", "m"], how="left").join(p3, on=["s1", "m"], how="left").join(fr, on=["s1", "m"], how="left").join(t1c.join(cc, on="ncore"), on="s1", how="left") \
         .with_columns(pl.col("s1").is_in(v9empty["s1"].to_list()).alias("on_empty"), pl.col("ours").fill_null(False))
add = add.with_columns(pl.when(pl.col("off") == "same").then(pl.col("cls")).otherwise(pl.lit("num_changed/other")).alias("type"))
print("=" * 25, "TEST: T4 additions by type (per 1k S1 of the country)")
t = add.group_by("country", "type").agg(pl.len().alias("n"), pl.col("on_empty").sum().alias("on_empty"), (~pl.col("ours")).sum().alias("outside_our_cands"),
                                        pl.col("p3").mean().round(3).alias("mean_V9_p3"), (pl.col("name_share") >= 5).mean().round(3).alias("generic_name"),
                                        pl.col("fp_shared").filter(pl.col("ra").fill_null("") != "").mean().round(3).alias("fingerprint"))
print(t.with_columns((pl.col("n") / pl.col("country").replace_strict(nt) * 1000).round(2).alias("per1k")).sort("country", "n", descending=[False, True]))
print("   French copy fingerprint baseline 0.222 (identical name, same address, accepted); look-alike siblings ~0.01")
# ---------------- labelled replay per type
print("=" * 25, "HOLDOUT replay: V9-rejected candidates of the same types (records not accepted elsewhere)")
gt = con.execute("select source1_entity_id s1, case when matched_entity_ids is null or matched_entity_ids='' then 0 else len(string_split(matched_entity_ids, ',')) end n, "
                 "hash(source1_entity_id || 'fold') % 10 fold from 'work/train_gt.parquet'").pl().join(pl.read_parquet("work/v5_dropped_s1.parquet").select("s1"), on="s1", how="anti")
gt = gt.filter(pl.col("fold").is_in([0, 8, 9])).join(pl.read_parquet("work/train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"}), on="s1")
ho = pl.read_parquet("work/copy_channel_holdout.parquet", columns=["s1", "m", "y", "p3", "acc"])
SYNP = pl.concat([pl.read_parquet(f"work/aug_v8_{s}.parquet", columns=["entity_id"]) for s in ("s2", "s3")]).rename({"entity_id": "m"})
low = pl.scan_parquet("work/v8_p2/train/*.parquet").filter(pl.col("fold").is_in([0, 8, 9]) & (pl.col("p1") < 0.05)).select("s1", "m", "y").collect().join(SYNP, on="m", how="anti") \
        .with_columns(pl.lit(None, pl.Float32).alias("p3"), pl.lit(False).alias("acc"))
hp = pl.concat([ho, low.select(ho.columns)])
acc_h = hp.filter("acc").select("s1", "m", "y")
base = gt.join(acc_h.group_by("s1").agg(pl.col("y").sum().alias("tp"), (~pl.col("y")).sum().alias("fp"), pl.len().alias("k")), on="s1", how="left").fill_null(0)
rej = annotate(hp.filter(~pl.col("acc")).join(acc_h.select("m").unique(), on="m", how="anti"), "train").join(gt.select("s1", "country"), on="s1")
rej = rej.with_columns(pl.when(pl.col("off") == "same").then(pl.col("cls")).otherwise(pl.lit("num_changed/other")).alias("type"),
                       pl.col("s1").is_in(base.filter(pl.col("k") == 0)["s1"].to_list()).alias("on_empty"))


def prf(z, tpc, fpc):
    fv = F(z[tpc], z[fpc], z["n"]); out = {}
    for c in ("US", "India"):
        m = (z["country"] == c).to_numpy(); tp, fp, n = z[tpc].to_numpy()[m].sum(), z[fpc].to_numpy()[m].sum(), z["n"].to_numpy()[m].sum()
        out[c] = (fv[m].mean(), tp / max(1, tp + fp), tp / max(1, n))
    return out


b0 = prf(base, "tp", "fp")
print("  V9 holdout:", {c: f"F {v[0]:.5f} P {v[1]:.4f} R {v[2]:.4f}" for c, v in b0.items()})
REPLAY = {}
for ty in ("identical@other", "identical@same", "other", "desc@same", "swap_extra@same", "num_changed/other"):
    for emp in (None, False):
        pool = rej.filter(pl.col("type") == ty)
        if emp is False: pool = pool.filter(~pl.col("on_empty"))
        ad = pool.sort("p3", descending=True, nulls_last=True).unique(subset="m", keep="first")
        z = base.join(ad.group_by("s1").agg(pl.col("y").sum().alias("atp"), (~pl.col("y")).sum().alias("afp")), on="s1", how="left").fill_null(0) \
                .with_columns((pl.col("tp") + pl.col("atp")).alias("tp2"), (pl.col("fp") + pl.col("afp")).alias("fp2"))
        r = prf(z, "tp2", "fp2")
        REPLAY[(ty, emp)] = float(ad["y"].mean() or 0)
        print(f"  +{ty:18s} {'(non-empty S1 only)' if emp is False else '(all S1)          '}: adds {ad.height / gt.height * 1000:6.1f}/1k precision {ad['y'].mean() or 0:.3f} -> "
              + "  ".join(f"{c} dF {r[c][0] - b0[c][0]:+.5f} (P {r[c][1]:.4f} R {r[c][2]:.4f})" for c in r))
    pb = rej.filter((pl.col("type") == ty) & pl.col("p3").is_not_null()).with_columns(pl.col("p3").cut([0.1, 0.3, 0.5, 0.7]).alias("b"))
    print("     precision by V9 p3 bin:", pb.group_by("b").agg(pl.len(), pl.col("y").mean().round(3)).sort("b").rows())
# ---------------- simulator deltas vs V9 for T4 and hybrids
SYN = pl.concat([pl.read_parquet(f"work/aug_v8_{s}.parquet", columns=["entity_id"]) for s in ("s2", "s3")]).rename({"entity_id": "m"})
tr = pl.scan_parquet("work/v8_p2/train/*.parquet").filter(pl.col("fold").is_in([0, 8, 9])).select("m", "y", "p1").collect().join(SYN, on="m", how="anti")
iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(tr["p1"].to_numpy(), tr["y"].to_numpy()); del tr


def delta(T, B, label, favour=None):
    ad, rm = T.join(B, on=["s1", "m"], how="anti"), B.join(T, on=["s1", "m"], how="anti")
    ch = pl.concat([ad.select("s1"), rm.select("s1")]).unique()
    Q = pl.scan_parquet("work/v8_p2/test/*.parquet").select("s1", "m", "p1", pl.col("p2x2").alias("q2")).collect().join(ch, on="s1", how="semi")
    Q = Q.with_columns(pl.Series("qp", iso.predict(Q["p1"].to_numpy()))).join(p3.join(ch, on="s1", how="semi"), on=["s1", "m"], how="left") \
         .with_columns(pl.coalesce("p3", "q2").clip(0, 1).alias("q"))
    if favour is not None:                                      # teammate-favourable: every added pair gets truth prob >= favour
        Q = Q.join(ad.with_columns(pl.lit(True).alias("_a")), on=["s1", "m"], how="left").with_columns(
            pl.when(pl.col("_a").fill_null(False)).then(pl.max_horizontal(pl.col("q"), pl.lit(favour))).otherwise(pl.col("q")).alias("q"),
            pl.when(pl.col("_a").fill_null(False)).then(pl.max_horizontal(pl.col("qp"), pl.lit(favour))).otherwise(pl.col("qp")).alias("qp")).drop("_a")
        miss = ad.join(Q, on=["s1", "m"], how="anti").with_columns(pl.lit(favour).alias("q"), pl.lit(favour).alias("qp"))
        Q = pl.concat([Q.select("s1", "m", "q", "qp"), miss.select("s1", "m", pl.col("q").cast(pl.Float64), pl.col("qp").cast(pl.Float64))], how="vertical_relaxed")
    S = ch.join(s1t, on="s1").with_row_index("i"); NS = S.height
    def mats(acc):
        a = acc.join(S, on="s1").join(Q.select("s1", "m", "q", "qp"), on=["s1", "m"], how="left").with_columns(pl.col("q").fill_null(0.0), pl.col("qp").fill_null(0.0))
        r = Q.select("s1", "m", "q", "qp").join(S, on="s1").join(acc, on=["s1", "m"], how="anti").filter((pl.col("q") >= 1e-3) | (pl.col("qp") >= 1e-3))
        def M(df, K):
            d = df.sort(["i", "q"], descending=[False, True]).with_columns(pl.int_range(pl.len()).over("i").alias("j")).filter(pl.col("j") < K)
            A = np.zeros((NS, K)); Bp = np.zeros((NS, K)); A[d["i"].to_numpy(), d["j"].to_numpy()] = d["q"].to_numpy(); Bp[d["i"].to_numpy(), d["j"].to_numpy()] = d["qp"].to_numpy()
            return A, Bp
        k = np.zeros(NS); ki = a.group_by("i").len(); k[ki["i"].to_numpy()] = np.minimum(ki["len"].to_numpy(), KA)
        return (k, *M(a, KA), *M(r, KR))
    mT, mB = mats(T.join(ch, on="s1", how="semi")), mats(B.join(ch, on="s1", how="semi"))
    cm = {c: (S["country"] == c).to_numpy() for c in ("US", "India", "France")}
    res = []
    for al in (1.0, 0.5, 0.0):
        d = expected_f(odds(al * mT[1] + (1 - al) * mT[2], 1.0), mT[0], odds(al * mT[3] + (1 - al) * mT[4], 1.0)) - \
            expected_f(odds(al * mB[1] + (1 - al) * mB[2], 1.0), mB[0], odds(al * mB[3] + (1 - al) * mB[4], 1.0))
        res.append((al, d.sum() / NT, {c: d[cm[c]].sum() / nt[c] for c in cm}))
    print(f"  {label}: " + " | ".join(f"alpha {al}: {tot:+.5f} (US {pc['US']:+.5f} IN {pc['India']:+.5f} FR {pc['France']:+.5f})" for al, tot, pc in res))
    return res


print("=" * 25, "SIMULATOR: expected LB change vs V9 (only changed S1)")
delta(T4, V9, "T4 vs V9")
delta(T4, V9, "T4 vs V9, TEAMMATE-FAVOURABLE (every addition >= 0.5 true)", favour=0.5)
delta(T4, V9, "T4 vs V9, EXTREME (every addition >= 0.9 true)", favour=0.9)
# hard bounds: all additions true (n = k_V9 + 1 on changed S1: the added pair is the only missing copy) / all false
k9 = V9.group_by("s1").len().rename({"len": "k"})
ab = add.group_by("s1").len().rename({"len": "a"}).join(k9, on="s1", how="left").fill_null(0)
best = (1 - F(ab["k"], 0, ab["k"] + ab["a"])) * 0 + (F(ab["k"] + ab["a"], 0, ab["k"] + ab["a"]) - F(ab["k"], 0, ab["k"] + ab["a"]))
worst = F(ab["k"], ab["a"], ab["k"]) - F(ab["k"], 0, ab["k"])
print(f"HARD BOUNDS vs V9 (assuming V9's own pairs on these S1 are right): all additions true {best.sum() / NT:+.5f}; all additions false {worst.sum() / NT:+.5f}")
# ---------------- hybrids on V10
print("=" * 25, "HYBRIDS on top of V10 (simulator, alpha 1 / 0.5 / 0)")
a10 = T4.join(V9, on=["s1", "m"], how="anti").join(V10.select("m").unique(), on="m", how="anti")
hyb = {"V10 + all T4 additions": a10,
       "V10 + T4 additions on non-empty S1 only": a10.join(v9empty, on="s1", how="anti"),
       "V10 + T4 additions with V9 p3 >= 0.5": a10.join(p3.filter(pl.col("p3") >= 0.5), on=["s1", "m"], how="semi"),
       "V10 + T4 identical-name same-address additions with copy-like fingerprint": a10.join(fr.filter(pl.col("fp_shared")), on=["s1", "m"], how="semi")}
for name, ad in hyb.items():
    print(f"  {name}: {ad.height} pairs")
    if ad.height:
        delta(pl.concat([V10, ad.select("s1", "m")]), V10, name)
