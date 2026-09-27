"""Label-free precision estimate per category from the same-source address FINGERPRINT (raw address equals another strong
same-source candidate's raw address). Calibration: holdout US/India TP vs FP fingerprint rates per category; France: its own
near-certain category (identical name, same number) gives the French copy baseline. Accepted V9 records only."""
import sys, json
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl
from tune_decision import prep, decide
src = open("scripts/110_relation_taxonomy.py").read(); exec(src.split("def best_pairs")[0]); exec(src[src.index("def name_rel"):src.index("def relate")])
d9 = json.load(open("output_v9/decision_s3.json"))
SYN = pl.concat([pl.read_parquet(f"work/aug_v8_{s}.parquet", columns=["entity_id"]) for s in ("s2", "s3")]).rename({"entity_id": "m"})
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
INIT = lambda c1, c2: len(c2.replace(" ", "")) <= 4 and c2.replace(" ", "") == "".join(w[0] for w in c1.split())[:len(c2.replace(" ", ""))]


def build(p3, split):
    acc = decide(prep(p3, d9["tau"], "p3"), d9).select("s1", "m").with_columns(pl.lit(True).alias("acc"))
    x = p3.filter(pl.col("p1") >= d9["tau"]).join(acc, on=["s1", "m"], how="left").with_columns(pl.col("acc").fill_null(False))
    ids = pl.concat([x.select(pl.col("s1").alias("entity_id")), x.select(pl.col("m").alias("entity_id"))]).unique()
    nm = pl.concat([pl.scan_parquet(f"work/{split}_s{s}_norm.parquet").select("entity_id", "ncore", "a", "nums") for s in (1, 2, 3)]).join(ids.lazy(), on="entity_id", how="semi").collect()
    N = {r[0]: r[1:] for r in nm.iter_rows()}
    raw = pl.concat([pl.scan_parquet(f"work/{split}_s{s}.parquet").select("entity_id", "business_address") for s in (2, 3)]).join(ids.lazy(), on="entity_id", how="semi").collect()
    RA = dict(zip(raw["entity_id"], raw["business_address"].fill_null("").str.to_lowercase()))
    x = x.with_columns(pl.Series("nrel", [name_rel(N[a][0], N[b][0]) for a, b in zip(x["s1"], x["m"])]),
                       pl.Series("arel", [addr_rel(N[a][1], N[b][1], N[a][2], N[b][2]) for a, b in zip(x["s1"], x["m"])]),
                       pl.Series("init", [INIT(N[a][0], N[b][0]) for a, b in zip(x["s1"], x["m"])]),
                       pl.Series("ra", [RA.get(m, "") for m in x["m"]]), pl.col("m").str.slice(0, 2).alias("src"))
    st = ((pl.col("p3") >= 0.9) & pl.col("acc")).cast(pl.Int32)
    x = x.with_columns(st.alias("_st")).with_columns(((pl.col("_st").sum().over(["s1", "src", "ra"]) - pl.col("_st")) > 0).alias("fp_shared"))
    same = pl.col("arel").is_in(["A0_identical", "A1_same_num"])
    return x.filter(same & (pl.col("ra") != "")).with_columns(
        pl.when(pl.col("init")).then(pl.lit("initials")).when(pl.col("nrel") == "N7_no_overlap").then(pl.lit("alias/no_overlap"))
          .when(pl.col("nrel") == "N0_identical").then(pl.lit("identical")).when(pl.col("nrel").is_in(["N1_extra_noise", "N4_dropped_words", "N5_typo_subst"])).then(pl.lit("light_edit"))
          .when(pl.col("nrel").is_in(["N2_extra_DESCRIPTOR", "N6_word_swap_DESC"])).then(pl.lit("desc_word")).otherwise(pl.lit("swap/extra")).alias("cls"))


t1 = pl.read_parquet("work/train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
h = pl.concat([pl.read_parquet(f"output_v9/p3/train/fold{f}.parquet") for f in (0, 8, 9)]).join(SYN, on="m", how="anti")
hx = build(h, "train").filter(pl.col("acc"))
print("HOLDOUT accepted same-number records: fingerprint-shared rate for TRUE vs FALSE, by class")
print(hx.group_by("cls", "y").agg(pl.len(), pl.col("fp_shared").mean().round(3).alias("shared")).sort("cls", "y"))
ref = hx.group_by("cls").agg(pl.col("fp_shared").filter(pl.col("y")).mean().alias("s_true"), pl.col("fp_shared").filter(~pl.col("y")).mean().alias("s_false"),
                             pl.col("y").mean().round(3).alias("holdout_precision"))
for c in ("US", "India", "France"):
    tx = build(pl.read_parquet(f"output_v9/p3/test/{c}.parquet"), "test").filter(pl.col("acc"))
    base = float(tx.filter(pl.col("cls") == "identical")["fp_shared"].mean())
    t = tx.group_by("cls").agg(pl.len().alias("n_acc"), pl.col("fp_shared").mean().alias("shared")).join(ref, on="cls", how="left")
    # precision estimate: mixture of true (rate = s_true scaled to this country's identical-name baseline) and false (holdout s_false)
    t = t.with_columns((pl.col("s_true") * base / float(hx.filter(pl.col("cls") == "identical")["fp_shared"].mean())).alias("s_true_c"))
    t = t.with_columns(((pl.col("shared") - pl.col("s_false")) / (pl.col("s_true_c") - pl.col("s_false")).clip(1e-3)).clip(0, 1).round(3).alias("est_precision"))
    print(f"==== TEST {c} (identical-name baseline shared rate {base:.3f})")
    print(t.select("cls", "n_acc", pl.col("shared").round(3), pl.col("s_true_c").round(3), pl.col("s_false").round(3), "holdout_precision", "est_precision").sort("n_acc", descending=True))
