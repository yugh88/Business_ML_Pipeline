"""Diagnostic suite for one scored run (internal reasoning; text + json report).
usage: 170_diag_suite.py <label> <p2_dir> <decision.json>
A  holdout (augmented / clean): F0.5, loss decomposition (FP / FN / singleton), FN stage (retrieval / cascade / ranking /
   decision), FP + FN by house-offset kind x name relation, calibration by category, margins
B  test (label-free): per-country densities by category vs train-holdout TP densities (excess = FP estimate),
   +k/-k symmetry, collectively promoted records (p2 accepts / pairwise p1 rejects), per-S1 configurations,
   score distributions per category, margins
Memory: one fold / one country at a time."""
import sys, json, os, gc
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl, numpy as np, duckdb
from tune_decision import prep, decide
src = open("scripts/110_relation_taxonomy.py").read()
exec(src.split("def best_pairs")[0]); exec(src[src.index("def name_rel"):src.index("def relate")])

label, p2dir, decf = sys.argv[1:4]
dec = json.load(open(decf)); V = dec["variant"]; TAU = dec["tau"]
P1DEC = dict(policy="G_expected_f", a=1.25, lam=0.0, cap=1)
OUT = {"label": label, "decision": {k: dec.get(k) for k in ("tau", "variant", "policy", "a", "lam", "t", "cap")}}
LINES = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LINES.append(s)
con = duckdb.connect()
SYN = pl.concat([pl.read_parquet(f"work/aug_v8_{s}.parquet", columns=["entity_id"]) for s in ("s2", "s3")]).rename({"entity_id": "m"}) \
      if os.path.exists("work/aug_v8_s2.parquet") else pl.DataFrame({"m": []}, schema={"m": pl.Utf8})


def f05(tp, fp, n):
    P = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-9), 0); R = np.where(n > 0, tp / np.maximum(n, 1), 0)
    return np.where(n == 0, (tp + fp == 0).astype(float), np.where(tp > 0, 1.25 * P * R / np.maximum(0.25 * P + R, 1e-12), 0.0))


def categorize(d, split):
    """adds off (house offset kind), nrel (name relation), arel, syn flag. d has s1, m."""
    ids1 = d.select(pl.col("s1").alias("entity_id")).unique(); ids2 = d.select(pl.col("m").alias("entity_id")).unique()
    n1 = pl.scan_parquet(f"work/{split}_s1_norm.parquet").select("entity_id", "ncore", "a", "nums").join(ids1.lazy(), on="entity_id", how="semi").collect()
    n2 = pl.concat([pl.scan_parquet(f"work/{split}_s{s}_norm.parquet").select("entity_id", "ncore", "a", "nums") for s in (2, 3)])
    if split == "train" and os.path.exists("work/aug_v8_norm.parquet"):
        n2 = pl.concat([n2, pl.scan_parquet("work/aug_v8_norm.parquet").select("entity_id", "ncore", "a", "nums")])
    n2 = n2.join(ids2.lazy(), on="entity_id", how="semi").collect()
    x = d.join(n1.rename({"entity_id": "s1", "ncore": "c1", "a": "a1", "nums": "n1"}), on="s1", how="left") \
         .join(n2.rename({"entity_id": "m", "ncore": "c2", "a": "a2", "nums": "n2"}), on="m", how="left")
    x = x.with_columns([pl.col(c).fill_null("") for c in ("c1", "c2", "a1", "a2", "n1", "n2")])
    h1, h2 = pl.col("n1").str.split(" ").list.first().fill_null(""), pl.col("n2").str.split(" ").list.first().fill_null("")
    dd = h2.cast(pl.Int64, strict=False) - h1.cast(pl.Int64, strict=False)
    x = x.with_columns(pl.when((h1 == "") | (h2 == "")).then(pl.lit("empty")).when(h1 == h2).then(pl.lit("same"))
                        .when((dd >= 1) & (dd <= 9)).then(pl.lit("+k")).when((dd <= -1) & (dd >= -9)).then(pl.lit("-k"))
                        .when(dd.abs() <= 99).then(pl.lit("d10-99")).when(dd.is_null()).then(pl.lit("nonint")).otherwise(pl.lit("d100+")).alias("off"),
                        pl.Series("nrel", [name_rel(a, b) for a, b in zip(x["c1"].to_list(), x["c2"].to_list())]))
    return x.drop("c1", "c2", "a1", "a2", "n1", "n2").join(SYN.with_columns(pl.lit(True).alias("syn")), on="m", how="left").with_columns(pl.col("syn").fill_null(False))


# ---------------------------------------------------------------- A: holdout
gt = con.execute("select source1_entity_id s1, case when matched_entity_ids is null or matched_entity_ids='' then 0 else "
                 "len(string_split(matched_entity_ids, ',')) end n, hash(source1_entity_id || 'fold') % 10 fold from 'work/train_gt.parquet'").pl()
gt = gt.join(pl.read_parquet("work/v5_dropped_s1.parquet").select("s1"), on="s1", how="anti")
t1c = pl.read_parquet("work/train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
pairs = pl.read_parquet("work/train_pairs.parquet").select("s1", "m")
A = {"folds": {}}; calib_parts, trd_parts, fn_parts, marg_parts = [], [], [], []
for f in [0, 8, 9]:
    d = pl.scan_parquet(f"{p2dir}/train/*.parquet").filter(pl.col("fold") == f).select("s1", "m", "y", "fold", "p1", "pc2", "pa", V).collect()
    s1f = gt.filter(pl.col("fold") == f)
    pr = decide(prep(d, TAU, V), dec).with_columns(pl.lit(True).alias("pred"))
    x = d.join(pr, on=["s1", "m"], how="left").with_columns(pl.col("pred").fill_null(False))
    res = {}
    for view, xx in (("aug", x), ("clean", x.join(SYN, on="m", how="anti"))):
        g = s1f.join(xx.filter(pl.col("pred")).group_by("s1").agg(pl.col("y").sum().alias("tp"), (~pl.col("y")).sum().alias("fp")), on="s1", how="left").fill_null(0)
        tp, fp, n = [g[c].to_numpy().astype(float) for c in ("tp", "fp", "n")]
        F = f05(tp, fp, n); Fp = f05(tp, 0 * fp, n); Fr = f05(np.where(n > 0, n, tp), fp, n)
        res[view] = dict(F=round(float(F.mean()), 5), FP_loss=round(float(Fp.mean() - F.mean()), 5), FN_loss=round(float(Fr.mean() - F.mean()), 5),
                         singleton_FP_loss=round(float(((n == 0) & (tp + fp > 0)).mean()), 5), S1_with_FP=round(float((fp > 0).mean()), 4),
                         S1_with_FN=round(float((tp < n).mean()), 4))
    A["folds"][f] = res
    say(f"[A] fold {f}: {res}")
    # FN stage decomposition (clean truth pairs of kept S1)
    tr = pairs.join(s1f.select("s1"), on="s1", how="semi")
    fnx = tr.join(x.filter(pl.col("pred")).select("s1", "m"), on=["s1", "m"], how="anti").join(d.select("s1", "m", "p1", V), on=["s1", "m"], how="left")
    mm = d.group_by("m").agg(pl.col(V).max().alias("mmax"))
    fnx = fnx.join(mm, on="m", how="left").with_columns(
        pl.when(pl.col("p1").is_null()).then(pl.lit("1_not_retrieved")).when(pl.col("p1") < TAU).then(pl.lit("2_cascade_p1<tau"))
          .when(pl.col(V) < pl.col("mmax") - 1e-9).then(pl.lit("3_lost_to_other_S1")).when(pl.col(V) < 0.5).then(pl.lit("4_scored_low"))
          .otherwise(pl.lit("5_decision_cut")).alias("stage"))
    fn_parts.append(categorize(fnx.select("s1", "m", "stage"), "train").select("stage", "off", "nrel"))
    # FP / TP categories for densities
    xp = categorize(x.filter(pl.col("pred")).select("s1", "m", "y"), "train").join(t1c, on="s1", how="left")
    trd_parts.append(xp.select("country", "off", "nrel", "y", "syn"))
    # calibration on cascade candidates
    cc = categorize(x.filter(pl.col("p1") >= TAU).select("s1", "m", "y", V, "pred").sample(fraction=0.35, seed=f), "train")
    calib_parts.append(cc.select("off", "y", V))
    # margins per S1
    mg = x.filter(pl.col("p1") >= TAU).group_by("s1").agg(pl.col(V).filter(pl.col("pred")).min().alias("min_acc"), pl.col(V).filter(~pl.col("pred")).max().alias("max_rej"),
                                                           pl.col("pred").sum().alias("k"))
    marg_parts.append(mg.select("min_acc", "max_rej", "k").with_columns(pl.lit("train").alias("split")))
    del d, x, pr, xp, cc, fnx, mg; gc.collect()
fn = pl.concat(fn_parts); trd = pl.concat(trd_parts); cal = pl.concat(calib_parts)
nS1_tr = {c: v for c, v in gt.filter(pl.col("fold").is_in([0, 8, 9])).join(t1c, on="s1").group_by("country").len().rows()}
say("\n[A] FN stage x offset (3 holdout folds):")
t = fn.group_by("stage").agg(pl.len().alias("n")).sort("stage"); say(t.rows())
say(fn.group_by("stage", "off").len().sort("len", descending=True).head(16).rows())
say(fn.group_by("stage", "nrel").len().sort("len", descending=True).head(12).rows())
say("\n[A] FP by offset x name relation (real vs synthetic):")
fpx = trd.filter(~pl.col("y"))
say(fpx.group_by("syn", "off").len().sort("len", descending=True).head(14).rows())
say(fpx.group_by("syn", "nrel").len().sort("len", descending=True).head(14).rows())
say("\n[A] calibration by category (p-bin -> mean p, true rate, n):")
cal = cal.with_columns((pl.col(V) * 10).floor().clip(0, 9).alias("bin"))
for cat in ["same", "+k", "-k", "empty", "d100+"]:
    c = cal.filter(pl.col("off") == cat).group_by("bin").agg(pl.col(V).mean().round(3).alias("p"), pl.col("y").mean().round(3).alias("true"), pl.len().alias("n")).sort("bin")
    say(f"  {cat}: {c.rows()}")
cc2 = cal.group_by("bin").agg(pl.col(V).mean().round(3).alias("p"), pl.col("y").mean().round(3).alias("true"), pl.len().alias("n")).sort("bin")
say(f"  ALL: {cc2.rows()}")
# train TP density per category (per 1k kept S1, by country)
trd = trd.with_columns(pl.col("country").fill_null("?"))
dens_tr = trd.filter(~pl.col("syn")).group_by("country", "off", "nrel").agg(pl.col("y").sum().alias("tp"), (~pl.col("y")).sum().alias("fp"))
dens_tr = dens_tr.with_columns((1000 * pl.col("tp") / pl.col("country").replace_strict(nS1_tr, default=1, return_dtype=pl.Float64)).alias("tp_k"),
                               (1000 * pl.col("fp") / pl.col("country").replace_strict(nS1_tr, default=1, return_dtype=pl.Float64)).alias("fp_k"))
OUT["A"] = A
del trd, cal, fn, fpx, fn_parts, trd_parts, calib_parts; gc.collect()
# ---------------------------------------------------------------- B: test
s1c = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"}); nS1 = dict(s1c.group_by("country").len().rows())
B = {}
for c in ["US", "India", "France"]:
    t = pl.scan_parquet(f"{p2dir}/test/*.parquet").filter((pl.col("country") == c) & (pl.col("p1") >= min(TAU, 0.05))).select("s1", "m", "p1", "pc2", "pa", V).collect()
    pr = decide(prep(t, TAU, V), dec).with_columns(pl.lit(True).alias("pred"))
    a1 = decide(prep(t, 0.05, "p1"), P1DEC).with_columns(pl.lit(True).alias("acc1"))
    xp = categorize(pr.select("s1", "m").join(a1, on=["s1", "m"], how="left").with_columns(pl.col("acc1").fill_null(False)), "test")
    n = nS1[c] / 1000
    cnt = xp.group_by("off", "nrel").agg(pl.len().alias("npred"), (~pl.col("acc1")).sum().alias("promoted")).with_columns((pl.col("npred") / n).alias("pred_k"), (pl.col("promoted") / n).alias("prom_k"))
    ref = dens_tr.filter(pl.col("country") == (c if c != "France" else "US")).select("off", "nrel", "tp_k", "fp_k")
    j = cnt.join(ref, on=["off", "nrel"], how="full", coalesce=True).fill_null(0).with_columns((pl.col("pred_k") - pl.col("tp_k")).alias("excess_k"))
    offs = xp.group_by("off").len(); od = {k: v / n for k, v in offs.rows()}
    B[c] = dict(pred_per_s1=round(xp.height / nS1[c], 4), plus_k=round(od.get("+k", 0), 2), minus_k=round(od.get("-k", 0), 2),
                promoted_k=round(float((~xp["acc1"]).sum()) / n, 2), excess_total_k=round(float(j["excess_k"].sum()), 2))
    say(f"\n[B] TEST {c}: {B[c]}")
    say("   top excess categories (off, nrel, pred_k, train tp_k, excess_k, promoted_k):")
    for r in j.sort("excess_k", descending=True).head(10).select("off", "nrel", pl.col("pred_k").round(2), pl.col("tp_k").round(2), pl.col("excess_k").round(2), pl.col("prom_k").round(2)).rows():
        say("    ", r)
    # per-S1 configuration of predicted offsets
    cfg = xp.with_columns(pl.col("off").replace({"same": "S", "+k": "P", "-k": "M", "empty": "e"}, default="o").alias("k")) \
            .group_by("s1").agg(pl.col("k").unique().sort().str.join("").alias("cfg"))
    cf = s1c.filter(pl.col("country") == c).join(cfg, on="s1", how="left").with_columns(pl.col("cfg").fill_null("<none>")).group_by("cfg").len() \
            .with_columns((pl.col("len") / n).round(2).alias("per1k")).sort("len", descending=True)
    say("   S1 configurations per 1k:", cf.head(10).select("cfg", "per1k").rows())
    # margins
    mg = t.filter(pl.col("p1") >= TAU).join(pr, on=["s1", "m"], how="left").with_columns(pl.col("pred").fill_null(False)) \
          .group_by("s1").agg(pl.col(V).filter(pl.col("pred")).min().alias("min_acc"), pl.col(V).filter(~pl.col("pred")).max().alias("max_rej"), pl.col("pred").sum().alias("k"))
    marg_parts.append(mg.select("min_acc", "max_rej", "k").with_columns(pl.lit(f"test_{c}").alias("split")))
    del t, pr, a1, xp, cnt, j, cfg, cf, mg; gc.collect()
OUT["B"] = B
mg = pl.concat(marg_parts)
say("\n[A/B] margins: share of S1 with a rejected candidate within 0.2 of the weakest accepted; weakest accepted < 0.9")
say(mg.group_by("split").agg(((pl.col("min_acc") - pl.col("max_rej")) < 0.2).mean().round(4).alias("tight_margin"), (pl.col("min_acc") < 0.9).mean().round(4).alias("weak_accept"),
                             pl.col("k").mean().round(3).alias("mean_k")).sort("split").rows())
json.dump(OUT, open(f"analysis/out/170_diag_{label}.json", "w"), indent=1, default=str)
open(f"analysis/out/170_diag_{label}.txt", "w").write("\n".join(LINES))
