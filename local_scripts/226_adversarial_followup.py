"""Follow-up of the adversarial audit (225), all in the teammates' favour where a choice exists.
 A. T2 (V9 + identical-name SAME-number additions): holdout replay precision/F, French fingerprint, simulator delta.
 B. T1's removal rule (cap 7 matches per S1): holdout replay (drop lowest-p3 first = most favourable, and random).
 C. T1's +k additions with SMARTER selections on the holdout (top-1 per S1, p3 >= 0.1 / 0.3).
 D. simulator with offset-calibrated truth for identical-name +/-k pairs (fixes V9's +k under-scoring), T1/T2 deltas.
 E. France optimistic scenario: T1's French +k additions given the maximum genuine share the pool symmetry allows (0.57)."""
import sys, json
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl, numpy as np, duckdb
from sklearn.isotonic import IsotonicRegression
sys.path.insert(0, "scripts"); from simlib import KA, KR, expected_f, odds
exec(open("scripts/213_excess_profile.py").read().split("t1 = pl.read_parquet")[0])       # annotate()
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(220); pl.Config.set_tbl_hide_dataframe_shape(True)
con = duckdb.connect()
q = lambda p: con.execute(f"select source1_entity_id s1, trim(unnest(string_split(matched_entity_ids, ','))) m from read_csv('{p}', delim='\t', header=true, all_varchar=true, quote='') "
                          "where matched_entity_ids is not null and matched_entity_ids<>''").pl()


def F(tp, fp, n):
    tp, fp, n = (np.asarray(v, float) for v in (tp, fp, n))
    P = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-9), 0); R = np.where(n > 0, tp / np.maximum(n, 1), 0)
    return np.where(n == 0, (tp + fp == 0).astype(float), np.where(tp > 0, 1.25 * P * R / np.maximum(0.25 * P + R, 1e-12), 0.0))


s1t = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"}); nt = dict(s1t.group_by("country").len().rows())
V9 = q("output_v9/matching_results.tsv"); T1, T2 = q("matching_results.tsv"), q("matching_results (2).tsv")
# ---------------- holdout pool (tau-set + cascade-dropped), labels
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
rej = annotate(hp.filter(~pl.col("acc")).join(acc_h.select("m").unique(), on="m", how="anti"), "train").filter(pl.col("cls").str.starts_with("identical")).join(gt.select("s1", "country"), on="s1")


def add_eval(label, ad):
    ad = ad.sort("p3", descending=True, nulls_last=True).unique(subset="m", keep="first")
    z = base.join(ad.group_by("s1").agg(pl.col("y").sum().alias("atp"), (~pl.col("y")).sum().alias("afp")), on="s1", how="left").fill_null(0)
    d = F(z["tp"] + z["atp"], z["fp"] + z["afp"], z["n"]) - F(z["tp"], z["fp"], z["n"])
    print(f"  [{label}] adds {ad.height / gt.height * 1000:.1f}/1k, precision {ad['y'].mean() if ad.height else 0:.3f} -> holdout F US {d[(z['country'] == 'US').to_numpy()].mean():+.5f} "
          f"India {d[(z['country'] == 'India').to_numpy()].mean():+.5f}")


print("=" * 20, "A. T2-type rule on the holdout: V9-rejected identical-name SAME-number / no-number candidates, S1 with >=1 accepted copy")
nonempty = base.filter(pl.col("k") > 0).select("s1")
add_eval("identical-name same number, non-empty S1", rej.filter(pl.col("off").is_in(["same"])).join(nonempty, on="s1", how="semi"))
add_eval("identical-name no number, non-empty S1", rej.filter(pl.col("off").is_in(["no_num"])).join(nonempty, on="s1", how="semi"))
add_eval("... same number, p3 >= 0.3", rej.filter((pl.col("off") == "same") & (pl.col("p3") >= 0.3)).join(nonempty, on="s1", how="semi"))
a2 = T2.join(V9, on=["s1", "m"], how="anti")
fr = pl.read_parquet("work/fr_residual.parquet", columns=["s1", "m", "fp_shared", "ra", "cls", "acc", "p3"])
f2 = a2.join(fr, on=["s1", "m"]).filter(pl.col("ra") != "")
print(f"  T2 French additions with fingerprint info {f2.height}: fingerprint {f2['fp_shared'].mean():.3f} (French copy baseline 0.222, look-alike siblings ~0.01), mean V9 p3 {f2['p3'].mean():.3f}")
print("=" * 20, "B. T1 removal rule = cap 7 matches per S1, replayed on the holdout")
ranked = acc_h.join(hp.select("s1", "m", "p3"), on=["s1", "m"]).with_columns(pl.col("p3").rank("ordinal", descending=True).over("s1").alias("r"),
                                                                             pl.int_range(pl.len()).shuffle(seed=7).over("s1").alias("rr"))
for label, col in (("drop lowest-p3 beyond 7 (favourable)", "r"), ("drop random beyond 7", "rr")):
    kept = ranked.filter(pl.col(col) <= (7 if col == "r" else 6))
    z = gt.join(kept.group_by("s1").agg(pl.col("y").sum().alias("tp2"), (~pl.col("y")).sum().alias("fp2")), on="s1", how="left").join(base.select("s1", "tp", "fp"), on="s1").fill_null(0)
    d = F(z["tp2"], z["fp2"], z["n"]) - F(z["tp"], z["fp"], z["n"])
    rm = ranked.filter(pl.col(col) > (7 if col == "r" else 6))
    print(f"  [{label}] removes {rm.height / gt.height * 1000:.2f}/1k (true {rm['y'].mean():.3f}) -> holdout F US {d[(z['country'] == 'US').to_numpy()].mean():+.5f} India {d[(z['country'] == 'India').to_numpy()].mean():+.5f}")
print(f"  holdout S1 with >= 8 true copies: {gt.filter(pl.col('n') >= 8).height / gt.height:.2%}")
print("=" * 20, "C. T1 +k additions with smarter selections (holdout)")
pk = rej.filter(pl.col("off").is_in(["+1..9", "+10..99"]))
add_eval("top-1 +k per S1 by p3", pk.filter(pl.col("p3").is_not_null()).sort("p3", descending=True).unique(subset="s1", keep="first"))
add_eval("+k with p3 >= 0.1", pk.filter(pl.col("p3") >= 0.1))
add_eval("+k with p3 >= 0.3", pk.filter(pl.col("p3") >= 0.3))
add_eval("+k with p3 >= 0.5", pk.filter(pl.col("p3") >= 0.5))
print("=" * 20, "D. simulator with offset-calibrated truth for identical-name +/-k pairs (teammate-favourable)")
hk = annotate(hp, "train").filter(pl.col("cls").str.starts_with("identical") & pl.col("off").is_in(["+1..9", "-1..9", "+10..99", "-10..99"]))
CAL = {}
for o in ("+1..9", "-1..9", "+10..99", "-10..99"):
    z = hk.filter((pl.col("off") == o) & pl.col("p3").is_not_null())
    CAL[o] = (IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(z["p3"].to_numpy(), z["y"].to_numpy()), float(hk.filter((pl.col("off") == o) & pl.col("p3").is_null())["y"].mean() or 0))
SYN = pl.concat([pl.read_parquet(f"work/aug_v8_{s}.parquet", columns=["entity_id"]) for s in ("s2", "s3")]).rename({"entity_id": "m"})
tr = pl.scan_parquet("work/v8_p2/train/*.parquet").filter(pl.col("fold").is_in([0, 8, 9])).select("m", "y", "p1").collect().join(SYN, on="m", how="anti")
iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(tr["p1"].to_numpy(), tr["y"].to_numpy()); del tr
for name, T, extra in (("T1", T1, None), ("T2", T2, None), ("T1 France-optimistic", T1, "fr57")):
    add, rem = T.join(V9, on=["s1", "m"], how="anti"), V9.join(T, on=["s1", "m"], how="anti")
    changed = pl.concat([add.select("s1"), rem.select("s1")]).unique()
    q3 = pl.concat([pl.read_parquet(f"output_v9/p3/test/{c}.parquet", columns=["s1", "m", "p3"]) for c in ("US", "India", "France")]).join(changed, on="s1", how="semi")
    Q = pl.scan_parquet("work/v8_p2/test/*.parquet").select("s1", "m", "p1", pl.col("p2x2").alias("q2")).collect().join(changed, on="s1", how="semi")
    Q = Q.with_columns(pl.Series("qp", iso.predict(Q["p1"].to_numpy()))).join(q3, on=["s1", "m"], how="left").with_columns(pl.coalesce("p3", "q2").clip(0, 1).alias("q"))
    Qa = annotate(Q.select("s1", "m"), "test")
    Q = Q.join(Qa.select("s1", "m", "cls", "off"), on=["s1", "m"], how="left")
    qc = Q["q"].to_numpy().copy(); p3v = Q["p3"].to_numpy(); cl, of = Q["cls"].to_list(), Q["off"].to_list()
    for i in range(Q.height):
        if cl[i] and cl[i].startswith("identical") and of[i] in CAL:
            qc[i] = CAL[of[i]][0].predict([p3v[i]])[0] if not np.isnan(p3v[i]) else CAL[of[i]][1]
    Q = Q.with_columns(pl.Series("qcal", qc))
    if extra == "fr57":
        fradd = add.join(s1t.filter(pl.col("country") == "France"), on="s1", how="semi").with_columns(pl.lit(True).alias("_fa"))
        Q = Q.join(fradd, on=["s1", "m"], how="left").with_columns(pl.when(pl.col("_fa") & pl.col("off").is_in(["+1..9", "+10..99"])).then(0.57).otherwise(pl.col("qcal")).alias("qcal")).drop("_fa")
    S = changed.join(s1t, on="s1").with_row_index("i"); NS = S.height
    def mats(acc, col):
        a = acc.join(S, on="s1").join(Q, on=["s1", "m"], how="left").with_columns(pl.col(col).fill_null(0.0))
        r = Q.join(S, on="s1").join(acc, on=["s1", "m"], how="anti").filter(pl.col(col) >= 1e-3)
        def M(df, K):
            d = df.sort(["i", col], descending=[False, True]).with_columns(pl.int_range(pl.len()).over("i").alias("j")).filter(pl.col("j") < K)
            A = np.zeros((NS, K)); A[d["i"].to_numpy(), d["j"].to_numpy()] = d[col].to_numpy(); return A
        k = np.zeros(NS); ki = a.group_by("i").len(); k[ki["i"].to_numpy()] = np.minimum(ki["len"].to_numpy(), KA)
        return k, M(a, KA), M(r, KR)
    out = []
    for col in ("q", "qcal"):
        kT, AT, RT = mats(T.join(changed, on="s1", how="semi"), col); k9, A9, R9 = mats(V9.join(changed, on="s1", how="semi"), col)
        d = expected_f(AT, kT, RT) - expected_f(A9, k9, R9)
        cm = {c: (S["country"] == c).to_numpy() for c in ("US", "India", "France")}
        out.append(f"{'V9 p3 truth' if col == 'q' else 'offset-calibrated truth'}: total {d.sum() / s1t.height:+.5f} (US {d[cm['US']].sum() / nt['US']:+.5f}, India {d[cm['India']].sum() / nt['India']:+.5f}, "
                   f"France {d[cm['France']].sum() / nt['France']:+.5f}) -> LB {0.977 + d.sum() / s1t.height:.4f}")
    print(f"  {name}: " + " | ".join(out))
