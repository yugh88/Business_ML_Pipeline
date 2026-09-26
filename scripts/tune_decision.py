"""Post-hoc joint selection of the cascade threshold tau (candidate-set size) and the decision policy, using the
compact stage-2 score files (p2/{train,test}/*.parquet). Light enough to run on a laptop (~2 GB).

Rule (organizers): a smaller candidate set per S1 ranks higher. Selection on the valid fold only:
  1. for every tau: best macro-F0.5 over variants x policies on valid
  2. choose the LARGEST tau (smallest candidate set) whose best F0.5 >= global best - TOL
  3. within it, conservative tie-break (simplest variant/policy, highest threshold)
Holdout folds are reported, never used. Then the final TSVs are written from the chosen tau/policy.

usage: python tune_decision.py <p2_dir> <input_dir> <out_dir> [valid_fold] [holdout_folds]"""
import os, sys, json, time
import numpy as np, polars as pl, duckdb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memguard_local

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import apply_policy, score, POLICY_ORDER, VARIANTS, argk

TAUS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 2e-2, 5e-2]
VARIANTS_USED = [v for v in os.environ.get("TUNE_VARIANTS", "p2").split(",")]
TOL = 0.0003


def configs():
    """Narrowed after the full-data v2 evaluation (B/D/F never differed from A/E by > 0.0001; p2 > p2base everywhere)."""
    for cap in (0, 1):
        for t in np.round(np.arange(0.50, 0.851, 0.02), 3):
            yield dict(policy="A_threshold", t=float(t), cap=cap)
            yield dict(policy="E_margin", t=float(t), margin=0.3, cap=cap)
        for a in (0.7, 0.8, 1.0, 1.25, 1.5):
            for lam in (0.0, 0.05, 0.15, 0.3, 0.5):
                yield dict(policy="G_expected_f", a=a, lam=lam, cap=cap)


CAP = {"S2": 5, "S3": 6}   # max matches per S1 per source observed in train GT (02_gt.txt: n2 <= 5, n3 <= 6)


def decide(d, c):
    pred = apply_policy(d, c["policy"], **argk(c))
    if c.get("cap"):
        pred = pred.join(d.select("s1", "m", "p"), on=["s1", "m"]).with_columns(pl.col("m").str.slice(0, 2).alias("src"))
        pred = pred.with_columns(pl.col("p").rank("ordinal", descending=True).over(["s1", "src"]).alias("_r")) \
                   .filter(pl.col("_r") <= pl.col("src").replace_strict(CAP, default=99)).select("s1", "m")
    return pred


def prep(d, tau, v):
    """Candidate set at tau (p1 >= tau) with pool-record competition recomputed inside that set."""
    x = d.filter(pl.col("p1") >= tau).with_columns(pl.col(v).alias("p"))
    ms = x.group_by("m").agg(pl.col("p").max().alias("m_max"), pl.col("p").sort(descending=True).get(1, null_on_oob=True).fill_null(0).alias("m_2nd"))
    return x.join(ms, on="m").filter(pl.col("p") >= 0.01)


def main():
    p2dir, indir, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    vf = int(sys.argv[4]) if len(sys.argv) > 4 else 7
    hf = [int(x) for x in sys.argv[5].split(",")] if len(sys.argv) > 5 else [0, 8, 9]
    os.makedirs(outdir, exist_ok=True); t0 = time.time()
    con = duckdb.connect()
    s1t = con.execute(f"""select source1_entity_id s1, case when matched_entity_ids is null or matched_entity_ids='' then 0
                          else len(string_split(matched_entity_ids, ',')) end n, hash(source1_entity_id || 'fold') % 10 fold
                          from '{indir}/train_gt.parquet'""").pl()
    fr = float(os.environ.get("ORPHAN_FRAC", "0"))
    olist = os.environ.get("ORPHAN_LIST")   # V5: explicit dropped-S1 list
    dropped = pl.read_parquet(olist).select("s1") if olist else None
    if dropped is not None:
        s1t = s1t.join(dropped, on="s1", how="anti"); fr = max(fr, 1e-9)
    elif fr > 0:   # orphan-simulated train (analysis/17): dropped S1 are not evaluated (same polars hash as lib.keep_s1_expr)
        s1t = s1t.filter((pl.col("s1").hash(777) % 10000) >= int(fr * 10000))
    cols = ["s1", "m", "y", "fold", "p1", "pc2", "pa"] + [v for v in VARIANTS_USED]
    tr = pl.read_parquet(os.path.join(p2dir, "train", "*.parquet"), columns=cols)
    truth = tr.filter(pl.col("y")).select("s1", "m").unique()
    n_s1 = len(s1t)
    res = []
    for tau in TAUS:
        cps = tr.filter(pl.col("p1") >= tau).height / n_s1
        rec = tr.filter((pl.col("p1") >= tau) & pl.col("y")).height / int(s1t["n"].sum())
        for v in VARIANTS_USED:
            dv = prep(tr.filter(pl.col("fold") == vf), tau, v)
            sv = s1t.filter(pl.col("fold") == vf)
            for c in configs():
                r = score(decide(dv, c), truth, sv)
                r.update(c); r.update(variant=v, tau=tau, cands_per_s1=cps, cascade_recall=rec); res.append(r)
        b = max((r for r in res if r["tau"] == tau), key=lambda r: r["f05"])
        print(f"tau={tau:g} cands/S1={cps:.2f} cascade_recall={rec:.5f} best valid={b['f05']:.5f} ({b['variant']}/{b['policy']} {argk(b)}) {time.time()-t0:.0f}s", flush=True)
    gbest = max(r["f05"] for r in res)
    ok_taus = [tau for tau in TAUS if max(r["f05"] for r in res if r["tau"] == tau) >= gbest - TOL]
    tau = max(ok_taus)
    # performance-first (user directive): tolerance applies ONLY to candidate-set size; at the chosen tau the best config wins
    # outright (ties broken toward the data-derived per-source cap, then higher threshold).
    best_at_tau = max(x["f05"] for x in res if x["tau"] == tau)
    near = [r for r in res if r["tau"] == tau and r["f05"] >= best_at_tau - 1e-6]
    pick = sorted(near, key=lambda r: (-r.get("cap", 0), -r.get("t", 0)))[0]
    report = dict(pick=pick, global_best_valid=gbest, per_tau={}, holdout={})
    for tt in TAUS:
        b = max((r for r in res if r["tau"] == tt), key=lambda r: r["f05"])
        report["per_tau"][tt] = dict(best_valid=b["f05"], variant=b["variant"], policy=b["policy"], params=argk(b), cap=b.get("cap"), cands_per_s1=b["cands_per_s1"], cascade_recall=b["cascade_recall"])
    # ORDINARY view (no orphans): same kept S1 / folds, but pool records whose true S1 was dropped are removed.
    orph = None
    if fr > 0:
        pairs = pl.read_parquet(os.path.join(indir, "train_pairs.parquet"), columns=["s1", "m"])
        orph = (pairs.join(dropped, on="s1", how="semi") if dropped is not None else pairs.filter((pl.col("s1").hash(777) % 10000) < int(fr * 10000))).select("m")
    for label, r in (("chosen", pick), ("tau_1e-4_best", max((r for r in res if r["tau"] == 1e-4), key=lambda r: r["f05"])),
                     ("tau_1e-4_nocap", max((r for r in res if r["tau"] == 1e-4 and not r.get("cap")), key=lambda r: r["f05"]))):
        report["holdout"][label] = {}
        for f in hf:
            d = prep(tr.filter(pl.col("fold") == f), r["tau"], r["variant"])
            report["holdout"][label][f] = score(decide(d, r), truth, s1t.filter(pl.col("fold") == f))
        print(label, f"tau={r['tau']:g}", r["variant"], r["policy"], argk(r), "cap", r.get("cap"), "holdout F0.5 (test-like):",
              [round(report["holdout"][label][f]["f05"], 5) for f in hf], flush=True)
        if orph is not None:
            report["holdout"][label + "_ordinary"] = {}
            for f in hf:
                d = prep(tr.filter(pl.col("fold") == f).join(orph, on="m", how="anti"), r["tau"], r["variant"])
                report["holdout"][label + "_ordinary"][f] = score(decide(d, r), truth, s1t.filter(pl.col("fold") == f))
            print("   ordinary view (orphans removed):", [round(report["holdout"][label + "_ordinary"][f]["f05"], 5) for f in hf], flush=True)
    if os.environ.get("TUNE_SKIP_TEST"):
        report["all_valid_results"] = res
        json.dump(report, open(os.path.join(outdir, "tune_report.json"), "w"), indent=1, default=float)
        print("pick:", {k: pick[k] for k in ("tau", "variant", "policy", "cands_per_s1", "f05")}, argk(pick), "cap", pick.get("cap"))
        return
    # ---- final test outputs with the chosen tau / policy ----
    te = pl.read_parquet(os.path.join(p2dir, "test", "*.parquet"), columns=["s1", "m", "p1", "pc2", "pa", pick["variant"]])
    cand = te.filter(pl.col("p1") >= tau).select("s1", "m").unique()
    pred = decide(prep(te.with_columns(pl.lit(False).alias("y"), pl.lit(-1).alias("fold")), tau, pick["variant"]), pick)
    s1 = pl.read_parquet(os.path.join(indir, "test_s1.parquet"), columns=["entity_id"]).rename({"entity_id": "s1"})
    def write(pairs, col, path):
        g = pairs.sort("m").group_by("s1").agg(pl.col("m").str.join(",").alias(col))
        s1.join(g, on="s1", how="left").with_columns(pl.col(col).fill_null("")).sort("s1") \
          .rename({"s1": "source1_entity_id"}).write_csv(path, separator="\t", quote_style="never")
    write(cand, "candidate_entity_ids", os.path.join(outdir, "candidate_pairs.tsv"))
    write(pred, "matched_entity_ids", os.path.join(outdir, "matching_results.tsv"))
    report["test"] = dict(s1=s1.height, cand_pairs=cand.height, cands_per_s1=cand.height / s1.height, matches=pred.height,
                          matches_per_s1=pred.height / s1.height, s1_with_match=pred["s1"].n_unique())
    report["all_valid_results"] = res
    json.dump(report, open(os.path.join(outdir, "tune_report.json"), "w"), indent=1, default=float)
    print("test:", report["test"], f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
