"""France has no labels: estimate the true-copy share of the cross-encoder's French changes (adds / removes) label-free with the
address fingerprint (237 method: raw address equals a STRONG p3>=0.9 accepted same-source copy of the same S1; true-copy rate from the
same class x set-size stratum of high-confidence pairs, false rate from labelled train), and VALIDATE the estimator first on the
labelled holdout changes (US/India, 239), where the truth is known. Only '@same' classes are informative (s_true - s_false >= 0.05)."""
import sys, json, os
SUF = os.environ.get("SUF", "")                                     # e.g. "_ce2" -> files written by 244/245 with CE_SET=ce2
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl, numpy as np, duckdb
from sklearn.isotonic import IsotonicRegression
from tune_decision import prep, decide
exec(open("scripts/213_excess_profile.py").read().split("t1 = pl.read_parquet")[0])        # annotate()
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(230); pl.Config.set_tbl_hide_dataframe_shape(True)
S3D = "output_v11s3"; dec = json.load(open(f"{S3D}/decision_s3.json")); TAU = dec["tau"]
SF = {"identical@same": 0.036, "light_edit@same": 0.168, "swap_extra@same": 0.14, "first_word_replaced@same": 0.14, "no_overlap@same": 0.131, "desc@same": 0.01}
q = lambda p: duckdb.connect().execute(f"select source1_entity_id s1, trim(unnest(string_split(matched_entity_ids, ','))) m from read_csv('{p}', delim='\t', header=true, all_varchar=true, quote='') "
                                       "where matched_entity_ids is not null and matched_entity_ids<>''").pl()


def fingerprint(base, changes, split):
    """base: accepted (s1, m, p3); changes: (s1, m, cls, ...) -> changes + shared, k (base set size), plus the high-confidence reference."""
    ids = pl.concat([base.select("m"), changes.select("m")]).unique().rename({"m": "entity_id"})
    R = pl.concat([pl.scan_parquet(f"work/{split}_s{s}.parquet").select("entity_id", pl.col("business_address").fill_null("").str.to_lowercase().alias("ra")) for s in (2, 3)]) \
          .join(ids.lazy(), on="entity_id", how="semi").collect().rename({"entity_id": "m"})
    strong = base.filter(pl.col("p3") >= 0.9).join(R, on="m").with_columns(pl.col("m").str.slice(0, 2).alias("src")).select("s1", "src", "ra", pl.col("m").alias("m2"))
    kk = base.group_by("s1").len().rename({"len": "k"})

    def sh(z):
        z = z.join(R, on="m").filter(pl.col("ra") != "").with_columns(pl.col("m").str.slice(0, 2).alias("src"))
        s = z.join(strong, on=["s1", "src", "ra"], how="left").group_by("s1", "m").agg((pl.col("m2").is_not_null() & (pl.col("m2") != pl.col("m"))).any().alias("shared"))
        return z.join(s, on=["s1", "m"]).join(kk, on="s1", how="left").with_columns(pl.col("k").fill_null(0).clip(1, 6).alias("kb"))
    hc = annotate(base.filter(pl.col("p3") >= 0.99).select("s1", "m"), split)
    ref = sh(hc).group_by("cls", "kb").agg(pl.col("shared").mean().alias("st"), pl.len().alias("n_ref"))
    return sh(changes).join(ref, on=["cls", "kb"], how="left")


def estimate(z, b):
    """per class: estimated true share with low-p3 bias factor b (true copies with low scores share b x the high-confidence rate)."""
    z = z.filter(pl.col("cls").is_in(list(SF)) & pl.col("st").is_not_null()).with_columns(pl.col("cls").replace_strict(SF).alias("sf"))
    z = z.filter(b * pl.col("st") - pl.col("sf") >= 0.05)
    g = z.group_by("cls", "kb").agg(pl.col("shared").mean().alias("sh"), (b * pl.col("st").first()).alias("stb"), pl.col("sf").first(), pl.len().alias("n"),
                                    *([pl.col("y").mean().alias("true_share")] if "y" in z.columns else []))
    g = g.with_columns(((pl.col("sh") - pl.col("sf")) / (pl.col("stb") - pl.col("sf"))).clip(0, 1).alias("pt"))
    agg = [(pl.col("pt") * pl.col("n")).sum() / pl.col("n").sum(), pl.col("n").sum()]
    if "true_share" in g.columns: agg.append((pl.col("true_share") * pl.col("n")).sum() / pl.col("n").sum())
    return g.group_by("cls").agg(*[a.alias(nm) for a, nm in zip(agg, ["est_true", "n", "actual_true"])]).sort("n", descending=True)


# ---------------- A: validate on labelled holdout changes (US/India)
SYN = pl.concat([pl.read_parquet(f"work/aug_v8_{s}.parquet", columns=["entity_id"]) for s in ("s2", "s3")]).rename({"entity_id": "m"})
tr = pl.scan_parquet("work/v8_p2/train/*.parquet").filter(pl.col("fold").is_in([0, 8, 9])).select("m", "y", "p1").collect().join(SYN, on="m", how="anti")
iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(tr["p1"].to_numpy(), tr["y"].to_numpy()); del tr
t1 = pl.read_parquet("work/train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
h = pl.concat([pl.read_parquet(f"{S3D}/p3/train/fold{f}.parquet", columns=["s1", "m", "p1", "p3"]) for f in (0, 8, 9)]).join(SYN, on="m", how="anti").filter(pl.col("p1") >= TAU)
h = h.with_columns(pl.Series("ip1", iso.predict(h["p1"].to_numpy())).cast(pl.Float32)).with_columns((0.75 * pl.col("p3") + 0.25 * pl.col("ip1")).alias("qa"))
hb = decide(prep(h.select("s1", "m", "p1", "qa"), TAU, "qa"), dec).select("s1", "m").join(h.select("s1", "m", "p3"), on=["s1", "m"])
for kind, f in (("ADDS", f"work/ce_holdout_added{SUF}.parquet"), ("REMOVES", f"work/ce_holdout_removed{SUF}.parquet")):
    ch = pl.read_parquet(f)
    z = fingerprint(hb, ch.select("s1", "m", "y", "cls"), "train")
    for b in (0.7, 1.0):
        print(f"HOLDOUT {kind} (labelled), bias factor {b}: estimated vs actual true share by class")
        print(estimate(z, b))
# ---------------- B: France test changes
ch = pl.read_parquet(f"work/ce_test_changes{SUF}.parquet").filter(pl.col("country") == "France")
prob = pl.read_parquet(f"work/ce_test_prob{SUF}.parquet").filter(pl.col("country") == "France")
fb = q("output_v11pre/matching_results.tsv").join(prob.select("s1").unique(), on="s1", how="semi").join(prob.select("s1", "m", "p3"), on=["s1", "m"], how="left").with_columns(pl.col("p3").fill_null(1.0))
NF = prob["s1"].n_unique()
for mode in ("country", "pooled"):
    for kind in ("add", "remove"):
        c = annotate(ch.filter((pl.col("mode") == mode) & (pl.col("kind") == kind)).select("s1", "m"), "test")
        comp = c.group_by("cls").len().with_columns((pl.col("len") / NF * 1000).round(2).alias("per1k")).sort("len", descending=True)
        z = fingerprint(fb, c, "test")
        print(f"\nFRANCE [{mode}] {kind.upper()}S: {c.height} ({c.height / NF * 1000:.1f}/1k S1); composition {comp.select('cls', 'per1k').rows()}")
        for b in (0.7, 1.0):
            e = estimate(z, b)
            print(f"  bias {b}: " + " | ".join(f"{r[0]} est_true {r[1]:.2f} (n {r[2]})" for r in e.rows()))
