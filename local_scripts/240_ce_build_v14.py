"""V14 = V13 + cross-encoder-stacked band probabilities (239 stackers) through the unchanged V11 decoding.
Test changes per country = new decode - base decode (base must reproduce output_v11pre). Label-free diagnostics:
 change rates and relation-class composition vs the labelled holdout changes (239), holdout-referenced value (211 method),
 +k/-k twin direction of removals, French address fingerprint of adds/removes.
Build: (V13 - CE removes) + (CE adds not previously removed by the French rule), exclusivity + per-source caps kept.
usage: 240_ce_build_v14.py <france_mode: none|pooled|country> [out_dir]"""
import sys, json, pickle, os
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl, numpy as np, duckdb
from sklearn.isotonic import IsotonicRegression
from tune_decision import prep, decide, CAP
exec(open("scripts/239_ce_stack_holdout.py").read().split("# ---------------- data")[0].split("S3D = ")[0])     # imports only
exec("lg = lambda" + open("scripts/239_ce_stack_holdout.py").read().split("lg = lambda")[1].split("# ---------------- data")[0])  # lg/Xlr/Xgb/pred helpers
exec(open("scripts/213_excess_profile.py").read().split("t1 = pl.read_parquet")[0])        # annotate()
FR_MODE = sys.argv[1] if len(sys.argv) > 1 else "none,pooled,country"; OUT = sys.argv[2] if len(sys.argv) > 2 else "output_v14"
S3D = "output_v11s3"; dec = json.load(open(f"{S3D}/decision_s3.json")); TAU = dec["tau"]
SYN = pl.concat([pl.read_parquet(f"work/aug_v8_{s}.parquet", columns=["entity_id"]) for s in ("s2", "s3")]).rename({"entity_id": "m"})
tr = pl.scan_parquet("work/v8_p2/train/*.parquet").filter(pl.col("fold").is_in([0, 8, 9])).select("m", "y", "p1").collect().join(SYN, on="m", how="anti")
iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(tr["p1"].to_numpy(), tr["y"].to_numpy()); del tr
ALPHA = {"US": 0.75, "India": 0.75, "France": 1.0}
ST = pickle.load(open("work/ce_stackers.pkl", "rb"))
q = lambda p: duckdb.connect().execute(f"select source1_entity_id s1, trim(unnest(string_split(matched_entity_ids, ','))) m from read_csv('{p}', delim='\t', header=true, all_varchar=true, quote='') "
                                       "where matched_entity_ids is not null and matched_entity_ids<>''").pl()
V11pre, V11, V12, V13 = (q(f"output_{v}/matching_results.tsv") for v in ("v11pre", "v11", "v12", "v13"))
rule_pairs = pl.concat([V11.join(V11pre, on=["s1", "m"], how="anti"), V12.join(V11, on=["s1", "m"], how="anti")]).unique()
fr_removed = V12.join(V13, on=["s1", "m"], how="anti")
print(f"V13 {V13.height} pairs; rule-added pairs {rule_pairs.height}; French-rule removals {fr_removed.height}", flush=True)
nt = dict(pl.read_parquet("work/test_s1.parquet", columns=["country"]).group_by("country").len().rows())
ce = pl.read_parquet("work/ce_band_test.parquet")


def decode(x, col, alpha):
    x = x.with_columns((alpha * pl.col(col) + (1 - alpha) * pl.col("ip1")).alias("qa"))
    return decide(prep(x.select("s1", "m", "p1", "qa"), TAU, "qa"), dec).select("s1", "m")


CH, PROB = {}, []
for c in ("US", "India", "France"):
    x = pl.read_parquet(f"{S3D}/p3/test/{c}.parquet", columns=["s1", "m", "p1", "p3"]).filter(pl.col("p1") >= TAU)
    x = x.with_columns(pl.Series("ip1", iso.predict(x["p1"].to_numpy())).cast(pl.Float32))
    b = annotate(x.filter(pl.col("p3").is_between(0.02, 0.98)).join(ce, on=["s1", "m"], how="inner"), "test").with_columns(pl.lit(c).alias("country"))
    out = {}
    for mode in ("country", "pooled"):
        (kind, pooled), m = ST[mode]
        out[mode] = pred(m, b, kind, pooled)
    b = b.with_columns(pl.Series("p3c", out["country"]).cast(pl.Float32), pl.Series("p3p", out["pooled"]).cast(pl.Float32))
    x = x.join(b.select("s1", "m", "p3c", "p3p"), on=["s1", "m"], how="left").with_columns(pl.col("p3c").fill_null(pl.col("p3")), pl.col("p3p").fill_null(pl.col("p3")))
    base = decode(x, "p3", ALPHA[c])
    ref = V11pre.join(x.select("s1").unique(), on="s1", how="semi")
    print(f"{c}: band {b.height}, CE-shifted by >0.2: {int((np.abs(b['p3c'].to_numpy() - b['p3'].to_numpy()) > 0.2).sum())}; base decode {base.height} vs V11pre {ref.height} "
          f"(sym diff {base.join(ref, on=['s1', 'm'], how='anti').height + ref.join(base, on=['s1', 'm'], how='anti').height})", flush=True)
    for mode, col in (("country", "p3c"), ("pooled", "p3p")):
        new = decode(x, col, ALPHA[c])
        CH[(c, mode)] = (new.join(base, on=["s1", "m"], how="anti"), base.join(new, on=["s1", "m"], how="anti"))
    PROB.append(x.select("s1", "m", "p3", "p3c", "p3p").with_columns(pl.lit(c).alias("country")))
PROB = pl.concat(PROB)
PROB.write_parquet("work/ce_test_prob.parquet")
pl.concat([z.with_columns(pl.lit(c).alias("country"), pl.lit(md).alias("mode"), pl.lit(kd).alias("kind")) for (c, md), (ad, rm) in CH.items() for kd, z in (("add", ad), ("remove", rm))]).write_parquet("work/ce_test_changes.parquet")

# ---------------- diagnostics
had = pl.read_parquet("work/ce_holdout_added.parquet"); hrm = pl.read_parquet("work/ce_holdout_removed.parquet")
gt = duckdb.connect().execute("select source1_entity_id s1, hash(source1_entity_id || 'fold') % 10 fold from 'work/train_gt.parquet'").pl()
gt = gt.join(pl.read_parquet("work/v5_dropped_s1.parquet").select("s1"), on="s1", how="anti").filter(pl.col("fold").is_in([0, 8, 9]))
nh = dict(gt.join(pl.read_parquet("work/train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"}), on="s1").group_by("country").len().rows())
print("\nCHANGE RATES per 1k S1 (test vs labelled holdout):")
for c in ("US", "India"):
    ad, rm = CH[(c, "country")]
    ha, hr = had.filter(pl.col("country") == c), hrm.filter(pl.col("country") == c)
    print(f"  {c}: test adds {ad.height / nt[c] * 1000:.2f} removes {rm.height / nt[c] * 1000:.2f} | holdout adds {ha.height / nh[c] * 1000:.2f} "
          f"(TP {ha['y'].sum() / nh[c] * 1000:.2f} FP {(~ha['y']).sum() / nh[c] * 1000:.2f}) removes {hr.height / nh[c] * 1000:.2f} (TP {hr['y'].sum() / nh[c] * 1000:.2f} FP {(~hr['y']).sum() / nh[c] * 1000:.2f})")
for mode in ("country", "pooled"):
    ad, rm = CH[("France", mode)]
    print(f"  France [{mode}]: test adds {ad.height / nt['France'] * 1000:.2f} removes {rm.height / nt['France'] * 1000:.2f}")
print("\nHOLDOUT-REFERENCED VALUE of test changes (per country F; excess adds valued as FP, excess removes as FP removed):")
for c in ("US", "India"):
    ad, rm = CH[(c, "country")]
    k = 1000 / nh[c]
    ta = annotate(ad, "test").group_by("cls", "off").len().with_columns(pl.col("len") / nt[c] * 1000).rename({"len": "add_t"})
    tr_ = annotate(rm, "test").group_by("cls", "off").len().with_columns(pl.col("len") / nt[c] * 1000).rename({"len": "rem_t"})
    ha = had.filter(pl.col("country") == c).group_by("cls", "off").agg((pl.col("y").sum() * k).alias("addTP"), ((~pl.col("y")).sum() * k).alias("addFP"))
    hr = hrm.filter(pl.col("country") == c).group_by("cls", "off").agg((pl.col("y").sum() * k).alias("remTP"), ((~pl.col("y")).sum() * k).alias("remFP"))
    z = tr_.join(ta, on=["cls", "off"], how="full", coalesce=True).join(hr, on=["cls", "off"], how="full", coalesce=True).join(ha, on=["cls", "off"], how="full", coalesce=True).fill_null(0)
    z = z.with_columns((pl.col("rem_t") - pl.col("remTP") - pl.col("remFP")).alias("rex"), (pl.col("add_t") - pl.col("addTP") - pl.col("addFP")).alias("aex"))
    val = z.select(0.21 * pl.col("remFP") - 0.09 * pl.col("remTP") + 0.21 * pl.col("rex").clip(0) - 0.09 * (-pl.col("rex")).clip(0)
                   + 0.09 * pl.col("addTP") - 0.21 * pl.col("addFP") - 0.21 * pl.col("aex").clip(0)).sum().item() / 1000
    hol = z.select(0.21 * pl.col("remFP") - 0.09 * pl.col("remTP") + 0.09 * pl.col("addTP") - 0.21 * pl.col("addFP")).sum().item() / 1000
    print(f"  {c}: holdout-rate value {hol:+.5f}; with test excess {val:+.5f}")
    print(z.with_columns(pl.col(pl.Float64).round(2)).sort("rem_t", descending=True).head(8))
print("\nTWIN DIRECTION of test removals (+k twins are generator look-alikes; true-copy noise is symmetric), per 1k S1:")
for c, mode in (("US", "country"), ("India", "country"), ("France", "country"), ("France", "pooled")):
    ad, rm = CH[(c, mode)]
    r = annotate(rm, "test"); a = annotate(ad, "test")
    f = lambda z, o: z.filter(pl.col("off") == o).height / nt[c] * 1000
    print(f"  {c:6s} [{mode}] removes +1..9 {f(r, '+1..9'):.2f} vs -1..9 {f(r, '-1..9'):.2f}; +10..99 {f(r, '+10..99'):.2f} vs -10..99 {f(r, '-10..99'):.2f} | "
          f"adds +1..9 {f(a, '+1..9'):.2f} vs -1..9 {f(a, '-1..9'):.2f} | top removed classes {[(k_, round(n / nt[c] * 1000, 2)) for k_, n in r.group_by('cls').len().sort('len', descending=True).head(4).rows()]}")
# French address fingerprint: does the changed pair share its raw address with a STRONG (p3>=0.9) accepted same-source copy of the S1?
print("\nADDRESS FINGERPRINT of changes (share with a strong accepted copy; labelled refs: true copies ~0.15-0.36 by k, look-alikes 0-0.04):")
for c in ("US", "India", "France"):
    base = V11pre.join(PROB.filter(pl.col("country") == c).select("s1").unique(), on="s1", how="semi").join(PROB.select("s1", "m", "p3"), on=["s1", "m"], how="left").with_columns(pl.col("p3").fill_null(1.0))
    for mode in (("country", "pooled") if c == "France" else ("country",)):
        ad, rm = CH[(c, mode)]
        allp = pl.concat([base.select("s1", "m"), ad]).unique()
        ids = allp.select(pl.col("m").alias("entity_id")).unique()
        raw = pl.concat([pl.scan_parquet(f"work/test_s{s}.parquet").select("entity_id", pl.col("business_address").fill_null("").str.to_lowercase().alias("ra")) for s in (2, 3)]).join(ids.lazy(), on="entity_id", how="semi").collect()
        R = raw.rename({"entity_id": "m"})
        strong = base.filter(pl.col("p3") >= 0.9).join(R, on="m").with_columns(pl.col("m").str.slice(0, 2).alias("src")).select("s1", "src", "ra", pl.col("m").alias("m2"))
        kk = base.group_by("s1").len().rename({"len": "k"})

        def fp(z):
            z = z.join(R, on="m").filter(pl.col("ra") != "").with_columns(pl.col("m").str.slice(0, 2).alias("src"))
            z = z.join(strong, on=["s1", "src", "ra"], how="left").group_by("s1", "m").agg((pl.col("m2").is_not_null() & (pl.col("m2") != pl.col("m"))).any().alias("sh"))
            z = z.join(kk, on="s1", how="left").fill_null(0)
            return z["sh"].mean(), z.filter(pl.col("k") >= 3)["sh"].mean(), z.height
        a_, r_ = fp(ad), fp(rm)
        print(f"  {c:6s} [{mode}] adds: share {a_[0]:.3f} (k>=3: {a_[1]:.3f}, n={a_[2]}) | removes: share {r_[0]:.3f} (k>=3: {r_[1]:.3f}, n={r_[2]})")

# ---------------- build
def build(fr_mode, OUT):
    adds, rems = [], []
    for c in ("US", "India", "France"):
        mode = "country" if c != "France" else fr_mode
        if mode == "none": continue
        ad, rm = CH[(c, mode)]; adds.append(ad); rems.append(rm)
    adds = pl.concat(adds).join(fr_removed, on=["s1", "m"], how="anti"); rems = pl.concat(rems)
    V14 = V13.join(rems, on=["s1", "m"], how="anti")
    adds = adds.join(V14.select("m"), on="m", how="anti")                     # exclusivity: never re-use a record kept in V14
    # per-source caps: only CE adds may be dropped (kept V13 pairs first, then adds by stacked probability)
    col = "p3p" if fr_mode == "pooled" else "p3c"
    adds = adds.join(PROB.select("s1", "m", pl.col(col).alias("pp")), on=["s1", "m"], how="left").with_columns(pl.col("pp").fill_null(0.0), pl.col("m").str.slice(0, 2).alias("src"))
    nk = V14.with_columns(pl.col("m").str.slice(0, 2).alias("src")).group_by("s1", "src").len().rename({"len": "nk"})
    adds = adds.join(nk, on=["s1", "src"], how="left").with_columns(pl.col("nk").fill_null(0)).with_columns(pl.col("pp").rank("ordinal", descending=True).over(["s1", "src"]).alias("r"))
    over = adds.filter(pl.col("nk") + pl.col("r") > pl.col("src").replace_strict(CAP, default=99))
    adds = adds.filter(pl.col("nk") + pl.col("r") <= pl.col("src").replace_strict(CAP, default=99))
    V14 = pl.concat([V14, adds.select("s1", "m")]).unique()
    dup13 = V13.height - V13["m"].n_unique()
    assert V14.height - V14["m"].n_unique() <= dup13, "new record assigned to two S1"          # V11pre/V13 carry 5 decoder ties
    print(f"\nBUILD (France mode {fr_mode}): V13 {V13.height} - removed {V13.join(V14, on=['s1', 'm'], how='anti').height} + added {V14.join(V13, on=['s1', 'm'], how='anti').height} "
          f"(cap-dropped {over.height}) = V14 {V14.height}")
    s1_all = pl.read_csv("output_v13/matching_results.tsv", separator="\t", columns=["source1_entity_id"], schema_overrides={"source1_entity_id": pl.Utf8}, quote_char=None)
    g = V14.sort("m").group_by("s1", maintain_order=True).agg(pl.col("m").str.join(",").alias("matched_entity_ids")).rename({"s1": "source1_entity_id"})
    o = s1_all.join(g, on="source1_entity_id", how="left", maintain_order="left").with_columns(pl.col("matched_entity_ids").fill_null(""))
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/matching_results.tsv", "w") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s, m in o.iter_rows():
            f.write(f"{s}\t{m}\n")
    os.system(f"cp output_v13/candidate_pairs.tsv {OUT}/candidate_pairs.tsv")
    e = o.filter(pl.col("matched_entity_ids") == "").height
    print(f"wrote {OUT}: rows {o.height}, empty {e} (V13 97661), pairs {V14.height}")
    for c in ("US", "India", "France"):
        s = PROB.filter(pl.col("country") == c).select("s1").unique()
        print(f"  {c}: accepted/1k V13 {V13.join(s, on='s1', how='semi').height / nt[c] * 1000:.1f} -> V14 {V14.join(s, on='s1', how='semi').height / nt[c] * 1000:.1f}")


for fm in (FR_MODE.split(",")):
    build(fm, f"{OUT}_fr{fm}")
