"""Full-pipeline error budget on the labelled holdout (folds 0/8/9, kept S1, real records; V9 clean decisions).
Follows every true pair: union retrieval (p1>=1e-4 file ~ union) -> stage-1 cascade (p1>=tau) -> V9 decision; and every FP.
Oracle ladder (per-S1 macro F0.5): V9 | no FP | + decision FN | perfect on tau-set | perfect on p1>=1e-4 set | perfect.
Answers: is retrieval the bottleneck? do missed pairs sit on S1 with no other found copy (full misses)? interactions?"""
import sys, json
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl, numpy as np, duckdb
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(230); pl.Config.set_tbl_hide_dataframe_shape(True)
d9 = json.load(open("output_v9/decision_s3.json")); TAU = d9["tau"]
HF = [0, 8, 9]


def F(tp, fp, n):
    tp, fp, n = (np.asarray(v, float) for v in (tp, fp, n))
    P = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-9), 0); R = np.where(n > 0, tp / np.maximum(n, 1), 0)
    return np.where(n == 0, (tp + fp == 0).astype(float), np.where(tp > 0, 1.25 * P * R / np.maximum(0.25 * P + R, 1e-12), 0.0))


gt = duckdb.connect().execute("select source1_entity_id s1, case when matched_entity_ids is null or matched_entity_ids='' then 0 else len(string_split(matched_entity_ids, ',')) end n, "
                              "hash(source1_entity_id || 'fold') % 10 fold from 'work/train_gt.parquet'").pl()
gt = gt.join(pl.read_parquet("work/v5_dropped_s1.parquet").select("s1"), on="s1", how="anti").filter(pl.col("fold").is_in(HF))
t1 = pl.read_parquet("work/train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
gt = gt.join(t1, on="s1")
tp_pairs = pl.read_parquet("work/train_pairs.parquet", columns=["s1", "m"]).join(gt.select("s1"), on="s1", how="semi")
pos4 = pl.scan_parquet("work/v8_p2/train/*.parquet").filter(pl.col("fold").is_in(HF) & pl.col("y")).select("s1", "m", "p1").collect()
h = pl.read_parquet("work/copy_channel_holdout.parquet", columns=["s1", "m", "y", "p3", "acc", "cls"])
# stage of every true pair
st = tp_pairs.join(pos4, on=["s1", "m"], how="left").join(h.select("s1", "m", "acc"), on=["s1", "m"], how="left").with_columns(
    pl.when(pl.col("p1").is_null()).then(pl.lit("1_not_retrieved(p1<1e-4)")).when(pl.col("p1") < TAU).then(pl.lit("2_cascade_drop"))
      .when(~pl.col("acc").fill_null(False)).then(pl.lit("3_decision_FN")).otherwise(pl.lit("4_found")).alias("stage"))
print(f"HOLDOUT kept S1 {gt.height}, true pairs {tp_pairs.height}; stage of true pairs:")
print(st.group_by("stage").len().with_columns((pl.col("len") / tp_pairs.height).round(5).alias("share")).sort("stage"))
per = st.group_by("s1").agg((pl.col("stage") >= "2").sum().alias("r4"), (pl.col("stage") >= "3").sum().alias("rt"), (pl.col("stage") == "4_found").sum().alias("tp"))
fpc = h.filter(pl.col("acc") & ~pl.col("y")).group_by("s1").len().rename({"len": "fp"})
g = gt.join(per, on="s1", how="left").join(fpc, on="s1", how="left").fill_null(0)
n, r4, rt, tp, fp = (g[c].to_numpy() for c in ("n", "r4", "rt", "tp", "fp"))
lad = {"V9": F(tp, fp, n), "V9 - all FP": F(tp, 0, n), "V9 + all decision FN": F(rt, fp, n), "perfect decode on tau-set": F(rt, 0, n),
       "perfect decode on p1>=1e-4 set (~union)": F(r4, 0, n), "V9 + recover cascade drops": F(tp + (r4 - rt), fp, n),
       "V9 + recover cascade+retrieval misses": F(tp + (n - rt), fp, n)}
print("\nORACLE LADDER (macro F0.5 over holdout S1):")
for k, v in lad.items():
    print(f"  {k:42s} {v.mean():.5f}   (+{v.mean() - lad['V9'].mean():.5f})")
g = g.with_columns(pl.Series("f", lad["V9"]), pl.Series("f_perfect_tau", lad["perfect decode on tau-set"]))
print("\nLOSS BY SEGMENT (share of total V9 loss; loss = sum(1-F)/N):")
N = g.height
g = g.with_columns(pl.when(pl.col("n") == 0).then(pl.when(pl.col("fp") > 0).then(pl.lit("singleton + FP")).otherwise(pl.lit("singleton ok")))
                     .when(pl.col("r4") == 0).then(pl.lit("FULL retrieval miss (no copy retrieved)"))
                     .when(pl.col("rt") == 0).then(pl.lit("FULL miss after cascade"))
                     .when(pl.col("tp") == 0).then(pl.lit("retrieved but none accepted"))
                     .when((pl.col("tp") < pl.col("n")) & (pl.col("fp") > 0)).then(pl.lit("partial recall + FP"))
                     .when(pl.col("tp") < pl.col("n")).then(pl.lit("partial recall only"))
                     .when(pl.col("fp") > 0).then(pl.lit("full recall + FP")).otherwise(pl.lit("perfect")).alias("seg"))
seg = g.group_by("seg").agg(pl.len().alias("S1"), (pl.len() / N * 1000).round(2).alias("per1k"), ((1 - pl.col("f")).sum() / N).round(5).alias("loss"),
                            ((pl.col("f_perfect_tau") - pl.col("f")).sum() / N).round(5).alias("fixable_by_decoding"), pl.col("n").mean().round(2).alias("mean_n"),
                            pl.col("fp").sum().alias("FP"), (pl.col("n") - pl.col("tp")).sum().alias("missed_pairs"))
print(seg.sort("loss", descending=True))
print(f"  total loss {(1 - g['f']).sum() / N:.5f}")
# are retrieval/cascade misses concentrated on S1 with no other found copy?
miss = st.filter(pl.col("stage") < "3").join(g.select("s1", "n", "tp", "fp", "rt"), on="s1")
print(f"\nRETRIEVAL+CASCADE MISSED true pairs: {miss.height} ({miss.height / tp_pairs.height:.4%} of true pairs)")
print("  of which on S1 where V9 found NO copy:", miss.filter(pl.col("tp") == 0).height, f"({miss.filter(pl.col('tp') == 0).height / miss.height:.1%});",
      " S1 with n=1:", miss.filter(pl.col("n") == 1).height)
# marginal value of recovering each missed pair class given V9's current decisions (one pair at a time)
def gain(df):
    return float((F(df["tp"] + 1, df["fp"], df["n"]) - F(df["tp"], df["fp"], df["n"])).sum() / N)
dec_fn = st.filter(pl.col("stage") == "3_decision_FN").join(g.select("s1", "n", "tp", "fp"), on="s1")
print(f"  marginal F value if each were recovered alone: retrieval/cascade misses {gain(miss):.5f}; decision FNs {gain(dec_fn):.5f}")
print(f"  full-miss S1 (no copy retrieved at all): {int(((g['n'] > 0) & (g['r4'] == 0)).sum())} = {1000 * float(((g['n'] > 0) & (g['r4'] == 0)).mean()):.2f}/1k S1; "
      f"with a V9 FP (look-alike accepted instead): {int(((g['n'] > 0) & (g['r4'] == 0) & (g['fp'] > 0)).sum())}")
# decision FN / FP by class and why (exclusivity: another S1 took the record; low p3)
hx = h.join(pl.scan_parquet("work/v8_p2/train/*.parquet").filter(pl.col("fold").is_in(HF)).select("s1", "m", "p1").collect(), on=["s1", "m"], how="left")
ms = pl.read_parquet([f"output_v9/p3/train/fold{f}.parquet" for f in HF], columns=["s1", "m", "p3", "p1"]).filter(pl.col("p1") >= TAU).group_by("m").agg(pl.col("p3").max().alias("m_max"))
hx = hx.join(ms, on="m", how="left").with_columns((pl.col("p3") < pl.col("m_max")).alias("lost_competition"))
e = hx.filter(pl.col("y") != pl.col("acc")).with_columns(pl.when(pl.col("y")).then(pl.lit("FN")).otherwise(pl.lit("FP")).alias("err"))
e = e.join(g.select("s1", "n", "tp", "fp"), on="s1")
e = e.with_columns(pl.when(pl.col("err") == "FN").then(pl.Series(F(e["tp"] + 1, e["fp"], e["n"]) - F(e["tp"], e["fp"], e["n"])))
                     .otherwise(pl.Series(F(e["tp"], e["fp"] - 1, e["n"]) - F(e["tp"], e["fp"], e["n"]))).alias("value"))
print("\nDECISION ERRORS by class (value = F gain if that single error were fixed; per holdout S1):")
print(e.group_by("err", "cls").agg(pl.len().alias("n"), (pl.col("value").sum() / N).round(5).alias("value"), pl.col("lost_competition").mean().round(3).alias("lost_comp"),
                                   pl.col("p3").mean().round(3).alias("mean_p3")).sort("value", descending=True).head(24))
print("\nDECISION FN by reason:", e.filter(pl.col("err") == "FN").group_by(pl.when(pl.col("lost_competition")).then(pl.lit("record went to another S1 (exclusivity)"))
      .when(pl.col("p3") >= 0.5).then(pl.lit("p3>=0.5 but G cut (set-size)")).otherwise(pl.lit("p3<0.5 (scored low)")).alias("why")).agg(pl.len(), (pl.col("value").sum() / N).round(5)).rows())
g.select("s1", "country", "n", "r4", "rt", "tp", "fp", "f", "seg").write_parquet("work/error_budget_s1.parquet")
st.write_parquet("work/error_budget_pairs.parquet"); e.write_parquet("work/error_budget_errors.parquet")
