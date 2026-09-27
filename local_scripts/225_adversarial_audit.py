"""Adversarial audit of the teammate predictions (T1 = matching_results.tsv, T2 = matching_results (2).tsv). Tries to prove the
0.964-0.966 (T1) prediction WRONG:
 (1) LABELLED replay of their apparent rule on the holdout: V9-rejected identical-name candidates by house-number offset
     (+1..9, +10..99, -1..9, -10..99), overall and on S1 that V9 left empty -> true rate, and exact holdout F change if added.
 (2) test rejected-pool asymmetry: how many GENUINE +k copies can the test's V9-rejected +k pool contain (true copies are
     +/- symmetric; generator twins are +k only), vs how many +k pairs T1/T2 added.
 (3) removed pairs (T1): address fingerprint vs copy baseline, set-size pattern.
 (4) simulator bias: (a) truth model with V9's +k under-acceptance corrected by holdout calibration, (b) historic simulator error
     vs how far each submitted version deviates from V9."""
import sys, json
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl, numpy as np, duckdb
from sklearn.isotonic import IsotonicRegression
sys.path.insert(0, "scripts"); from simlib import KA, KR, expected_f, odds
exec(open("scripts/213_excess_profile.py").read().split("t1 = pl.read_parquet")[0])       # annotate(): cls, off
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(230); pl.Config.set_tbl_hide_dataframe_shape(True)
con = duckdb.connect()
q = lambda p: con.execute(f"select source1_entity_id s1, trim(unnest(string_split(matched_entity_ids, ','))) m from read_csv('{p}', delim='\t', header=true, all_varchar=true, quote='') "
                          "where matched_entity_ids is not null and matched_entity_ids<>''").pl()
OFFS = ["+1..9", "-1..9", "+10..99", "-10..99", "trunc", "far", "same", "no_num"]


def F(tp, fp, n):
    tp, fp, n = (np.asarray(v, float) for v in (tp, fp, n))
    P = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-9), 0); R = np.where(n > 0, tp / np.maximum(n, 1), 0)
    return np.where(n == 0, (tp + fp == 0).astype(float), np.where(tp > 0, 1.25 * P * R / np.maximum(0.25 * P + R, 1e-12), 0.0))


s1t = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"}); nt = dict(s1t.group_by("country").len().rows())
V9 = q("output_v9/matching_results.tsv"); SUB = {"T1": q("matching_results.tsv"), "T2": q("matching_results (2).tsv")}
v9empty_t = s1t.join(V9.select("s1").unique(), on="s1", how="anti").select("s1")
print("=" * 30, "(0) what the teammates changed vs V9 (per 1k S1)")
ADD = {}
for k, T in SUB.items():
    a = annotate(T.join(V9, on=["s1", "m"], how="anti"), "test").join(s1t, on="s1").with_columns(pl.col("s1").is_in(v9empty_t["s1"].to_list()).alias("on_empty"))
    ADD[k] = a
    t = a.group_by("country", "off").len().with_columns((pl.col("len") / pl.col("country").replace_strict(nt) * 1000).round(2)).pivot(on="off", index="country", values="len").fill_null(0)
    print(f"{k} added {a.height} (identical-name {a.filter(pl.col('cls').str.starts_with('identical')).height}; on V9-empty S1 {a.filter('on_empty').height}) by offset:")
    print(t.select(["country"] + [o for o in OFFS if o in t.columns]))
# ---------------- (1) labelled replay on the holdout
print("=" * 30, "(1) HOLDOUT replay: V9-rejected IDENTICAL-name candidates by offset (truth known)")
gt = con.execute("select source1_entity_id s1, case when matched_entity_ids is null or matched_entity_ids='' then 0 else len(string_split(matched_entity_ids, ',')) end n, "
                 "hash(source1_entity_id || 'fold') % 10 fold from 'work/train_gt.parquet'").pl().join(pl.read_parquet("work/v5_dropped_s1.parquet").select("s1"), on="s1", how="anti")
gt = gt.filter(pl.col("fold").is_in([0, 8, 9])).join(pl.read_parquet("work/train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"}), on="s1")
ho = pl.read_parquet("work/copy_channel_holdout.parquet", columns=["s1", "m", "y", "p3", "acc"])
SYNP = pl.concat([pl.read_parquet(f"work/aug_v8_{s}.parquet", columns=["entity_id"]) for s in ("s2", "s3")]).rename({"entity_id": "m"})
low = pl.scan_parquet("work/v8_p2/train/*.parquet").filter(pl.col("fold").is_in([0, 8, 9]) & (pl.col("p1") < 0.05)).select("s1", "m", "y").collect().join(SYNP, on="m", how="anti") \
        .with_columns(pl.lit(None, pl.Float32).alias("p3"), pl.lit(False).alias("acc"))
hp = pl.concat([ho, low.select(ho.columns)])
acc_h = hp.filter("acc").select("s1", "m")
rej = annotate(hp.filter(~pl.col("acc")).join(acc_h.select("m").unique(), on="m", how="anti"), "train")      # record not accepted elsewhere
rej = rej.filter(pl.col("cls").str.starts_with("identical")).join(gt.select("s1", "country"), on="s1")
v9empty_h = gt.join(acc_h.select("s1").unique(), on="s1", how="anti").select("s1")
rej = rej.with_columns(pl.col("s1").is_in(v9empty_h["s1"].to_list()).alias("on_empty"))
nh = dict(gt.group_by("country").len().rows())
t = rej.group_by("country", "off").agg((pl.len()).alias("n"), pl.col("y").mean().round(3).alias("true_rate"), pl.col("y").filter(pl.col("on_empty")).mean().round(3).alias("true_rate_on_empty"),
                                        pl.col("on_empty").sum().alias("n_on_empty"))
t = t.with_columns((pl.col("n") / pl.col("country").replace_strict(nh) * 1000).round(2).alias("per1k"))
print(t.filter(pl.col("off").is_in(OFFS[:4])).sort("country", "off"))
base = gt.join(acc_h.join(hp.select("s1", "m", "y"), on=["s1", "m"]).group_by("s1").agg(pl.col("y").sum().alias("tp"), (~pl.col("y")).sum().alias("fp")), on="s1", how="left").fill_null(0)
f0 = F(base["tp"], base["fp"], base["n"])
for rule, flt in (("add ALL rejected identical-name +k (1..99)", pl.col("off").is_in(["+1..9", "+10..99"])),
                  ("... only on V9-empty S1", pl.col("off").is_in(["+1..9", "+10..99"]) & pl.col("on_empty")),
                  ("... only on non-empty S1", pl.col("off").is_in(["+1..9", "+10..99"]) & ~pl.col("on_empty")),
                  ("add rejected identical-name -k (control)", pl.col("off").is_in(["-1..9", "-10..99"]))):
    ad = rej.filter(flt).sort("p3", descending=True, nulls_last=True).unique(subset="m", keep="first")
    z = base.join(ad.group_by("s1").agg(pl.col("y").sum().alias("atp"), (~pl.col("y")).sum().alias("afp")), on="s1", how="left").fill_null(0)
    d = F(z["tp"] + z["atp"], z["fp"] + z["afp"], z["n"]) - F(z["tp"], z["fp"], z["n"])
    print(f"  rule [{rule}]: adds {ad.height} ({ad.height / gt.height * 1000:.1f}/1k), precision {ad['y'].mean() if ad.height else 0:.3f} -> holdout F "
          f"US {d[(z['country'] == 'US').to_numpy()].mean():+.5f}  India {d[(z['country'] == 'India').to_numpy()].mean():+.5f}")
# ---------------- (2) test rejected pool asymmetry
print("=" * 30, "(2) TEST V9-rejected identical-name pool by offset (per 1k S1) and bound on genuine +k copies")
hr = t.select("country", "off", "n")
for c in ("US", "India", "France"):
    tc = pl.read_parquet(f"work/copy_channel_test_{c}.parquet", columns=["s1", "m", "acc"]) if c != "France" else pl.read_parquet("work/fr_residual.parquet", columns=["s1", "m", "acc"])
    lowt = pl.scan_parquet("work/v8_p2/test/*.parquet").filter((pl.col("country") == c) & (pl.col("p1") < 0.05)).select("s1", "m").collect().with_columns(pl.lit(False).alias("acc"))
    pool = pl.concat([tc.select("s1", "m", "acc"), lowt]).filter(~pl.col("acc")).join(V9.select("m").unique(), on="m", how="anti")
    pool = annotate(pool, "test").filter(pl.col("cls").str.starts_with("identical"))
    cnt = {o: pool.filter(pl.col("off") == o).height / nt[c] * 1000 for o in OFFS[:4]}
    # holdout ratio true(+k)/true(-k) among rejected, to translate the -k count into an expected genuine +k count
    hc = rej.filter(pl.col("country") == (c if c != "France" else "US"))
    ratio = {w: (hc.filter((pl.col("off") == f"+{w}") & pl.col("y")).height + 1) / (hc.filter((pl.col("off") == f"-{w}") & pl.col("y")).height + 1) for w in ("1..9", "10..99")}
    tr_neg = {w: rej.filter((pl.col("country") == (c if c != "France" else "US")) & (pl.col("off") == f"-{w}"))["y"].mean() or 0 for w in ("1..9", "10..99")}
    gen = {w: cnt[f"-{w}"] * tr_neg[w] * ratio[w] for w in ("1..9", "10..99")}
    added = {w: ADD["T1"].filter((pl.col("country") == c) & (pl.col("off") == f"+{w}")).height / nt[c] * 1000 for w in ("1..9", "10..99")}
    print(f"  {c:6s}: rejected +1..9 {cnt['+1..9']:.1f} vs -1..9 {cnt['-1..9']:.1f}; +10..99 {cnt['+10..99']:.1f} vs -10..99 {cnt['-10..99']:.1f}  | expected GENUINE +k in pool "
          f"{gen['1..9']:.2f} + {gen['10..99']:.2f} per 1k vs T1 added +k {added['1..9']:.1f} + {added['10..99']:.1f} per 1k -> max precision of T1's +k additions "
          f"{min(1, (gen['1..9'] + gen['10..99']) / max(1e-9, added['1..9'] + added['10..99'])):.2f}")
# ---------------- (3) removed pairs of T1
print("=" * 30, "(3) T1 REMOVED pairs")
rem = V9.join(SUB["T1"], on=["s1", "m"], how="anti")
fr = pl.read_parquet("work/fr_residual.parquet", columns=["s1", "m", "fp_shared", "ra", "cls", "acc"])
rf = rem.join(fr, on=["s1", "m"])
kept_same = fr.filter(pl.col("acc") & (pl.col("ra") != "") & (pl.col("cls") == "identical@same"))
print(f"  France removed with fingerprint info {rf.height}: fingerprint {rf.filter(pl.col('ra') != '')['fp_shared'].mean():.3f} vs kept identical@same copies {kept_same['fp_shared'].mean():.3f}")
k9 = V9.group_by("s1").len().rename({"len": "k_v9"}); kt = SUB["T1"].group_by("s1").len().rename({"len": "k_t1"})
rs = rem.select("s1").unique().join(k9, on="s1").join(kt, on="s1", how="left").fill_null(0)
print("  S1 with removals: V9 set size -> T1 set size (top):", rs.group_by("k_v9", "k_t1").len().sort("len", descending=True).head(8).rows())
p3all = pl.concat([pl.read_parquet(f"output_v9/p3/test/{c}.parquet", columns=["s1", "m", "p3"]) for c in ("US", "India", "France")])
rr = rem.join(p3all, on=["s1", "m"], how="left").with_columns(pl.col("p3").rank("ordinal", descending=True).over("s1").alias("_x"))
v9r = V9.join(p3all, on=["s1", "m"], how="left").with_columns(pl.col("p3").rank("ordinal", descending=True).over("s1").alias("rank_in_set"))
print("  rank of removed pair inside its V9 set (1 = most confident):", rem.join(v9r, on=["s1", "m"]).group_by("rank_in_set").len().sort("rank_in_set").head(6).rows())
# ---------------- (4a) simulator with V9's +k under-acceptance corrected
print("=" * 30, "(4a) truth model with holdout-calibrated q for identical-name +/-k pairs (corrects V9's +k recall 90% vs -k 99%)")
cal = {}
hk = annotate(hp.join(gt.select("s1"), on="s1", how="semi"), "train").filter(pl.col("cls").str.starts_with("identical") & pl.col("off").is_in(OFFS[:4]))
for o in OFFS[:4]:
    z = hk.filter((pl.col("off") == o) & pl.col("p3").is_not_null())
    cal[o] = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(z["p3"].to_numpy(), z["y"].to_numpy()) if z.height > 200 else None
    lowrate = hk.filter((pl.col("off") == o) & pl.col("p3").is_null())["y"].mean()
    cal[o + "_low"] = float(lowrate or 0)
    print(f"  {o}: holdout pairs {z.height}, cascade-dropped true rate {cal[o + '_low']:.3f}")
json.dump({k: v for k, v in cal.items() if k.endswith("_low")}, open("analysis/out/225_cal_low.json", "w"))
# ---------------- (4b) historic simulator error vs deviation from V9
print("=" * 30, "(4b) historic simulator error (alpha 0.5, from 204) vs deviation from V9")
hist = {"v2": ("output", -0.0076), "V3": ("output_v3_auto", -0.0046), "V5": ("output_v5", -0.0008), "V5s3r": ("output_v5s3r", 0.0003)}
for v, (d, err) in hist.items():
    X = q(f"{d}/matching_results.tsv")
    ch = pl.concat([X.join(V9, on=["s1", "m"], how="anti").select("s1"), V9.join(X, on=["s1", "m"], how="anti").select("s1")]).unique().height
    print(f"  {v:6s}: changed S1 {ch / s1t.height:.2%}, sim error {err:+.4f}")
for k, T in SUB.items():
    ch = pl.concat([T.join(V9, on=["s1", "m"], how="anti").select("s1"), V9.join(T, on=["s1", "m"], how="anti").select("s1")]).unique().height
    print(f"  {k:6s}: changed S1 {ch / s1t.height:.2%}")
