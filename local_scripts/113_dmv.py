"""Density-matched validation (DMV): holdout F0.5 with false matches re-weighted by the test/train decoy-density ratio
of their (name relation x address relation) category (label-free on test; analysis/out/110)."""
import sys, json; sys.path.insert(0, "aws/scripts"); import memguard_local
import polars as pl, numpy as np, duckdb
sys.path.insert(0, "scripts")
exec(open("scripts/110_relation_taxonomy.py").read().split("def best_pairs")[0])   # NOISE / DESC sets + imports
src = open("scripts/110_relation_taxonomy.py").read()
exec(src[src.index("def name_rel"):src.index("def relate")])                       # name_rel / addr_rel
from tune_decision import prep, decide
rtr = pl.read_parquet("work/rel_train.parquet"); rte = pl.read_parquet("work/rel_test.parquet")
S1_TR = pl.read_parquet("work/train_s1.parquet", columns=["entity_id"]).height - pl.read_parquet("work/v5_dropped_s1.parquet").height
S1_TE = pl.read_parquet("work/test_s1.parquet", columns=["entity_id"]).height
cat = lambda d: d.with_columns((pl.col("nrel") + "|" + pl.col("arel")).alias("c"))
rtr, rte = cat(rtr), cat(rte)
dens = rtr.group_by("c").agg((pl.col("truth") == "true_copy").sum().alias("tc"), (pl.col("truth") == "copy_of_other_S1").sum().alias("oc"),
                             (pl.col("truth") == "distractor").sum().alias("dc")) \
          .join(rte.group_by("c").len().rename({"len": "te"}), on="c", how="full", coalesce=True).fill_null(0)
dens = dens.with_columns(((pl.col("te") / S1_TE - (pl.col("tc") + pl.col("oc")) / S1_TR).clip(0) / ((pl.col("dc") / S1_TR) + 1e-9)).alias("w"))
dens = dens.with_columns(pl.when(pl.col("dc") < 200).then(1.0).otherwise(pl.col("w")).alias("w"))   # too few train decoys -> neutral
print("decoy weight w by category (largest decoy counts):")
print(dens.sort("dc", descending=True).head(20).select("c", "dc", pl.col("w").round(2)).rows())
dens.select("c", "w").write_parquet("work/dmv_weights.parquet")
W = dict(zip(dens["c"].to_list(), dens["w"].to_list()))

drop = pl.read_parquet("work/v5_dropped_s1.parquet").select("s1")
s1t = duckdb.connect().execute("""select source1_entity_id s1, case when matched_entity_ids is null or matched_entity_ids='' then 0
      else len(string_split(matched_entity_ids, ',')) end n, hash(source1_entity_id || 'fold') % 10 fold from 'work/train_gt.parquet'""").pl().join(drop, on="s1", how="anti")
nmx = pl.concat([pl.scan_parquet(f"work/train_s{s}_norm.parquet").select("entity_id", "ncore", "a", "nums") for s in (1, 2, 3)])
desc_words = set(pl.read_parquet("work/descriptor_words_train.parquet")["t"].to_list()) | set(FR)

def pair_cats(pred):
    ids = pl.concat([pred.select(pl.col("s1").alias("entity_id")), pred.select(pl.col("m").alias("entity_id"))]).unique()
    n = nmx.join(ids.lazy(), on="entity_id", how="semi").collect()
    x = pred.join(n.rename({"entity_id": "s1", "ncore": "c1", "a": "a1", "nums": "n1"}), on="s1").join(n.rename({"entity_id": "m", "ncore": "c2", "a": "a2", "nums": "n2"}), on="m")
    return x.with_columns(pl.Series("c", [name_rel(a, b) + "|" + addr_rel(c, d, e, f) for a, b, c, d, e, f in
                                          zip(x["c1"].to_list(), x["c2"].to_list(), x["a1"].to_list(), x["a2"].to_list(), x["n1"].to_list(), x["n2"].to_list())]),
                          pl.col("c2").str.split(" ").list.set_difference(pl.col("c1").str.split(" ")).list.eval(pl.element().is_in(list(desc_words))).list.any().alias("has_desc"))

def f05(tp, fp, n):
    P = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-9), 0); R = np.where(n > 0, tp / np.maximum(n, 1), 0)
    return np.where(n == 0, (tp + fp == 0).astype(float), np.where(tp > 0, 1.25 * P * R / np.maximum(0.25 * P + R, 1e-12), 0.0))

def evaluate(pred_c, truth, s1f, label):
    x = pred_c.join(truth.with_columns(pl.lit(True).alias("t")), on=["s1", "m"], how="left").with_columns(pl.col("t").fill_null(False))
    x = x.with_columns(pl.col("c").replace_strict(W, default=1.0).alias("w"))
    per = x.group_by("s1").agg(pl.col("t").sum().alias("tp"), (~pl.col("t")).sum().alias("fp"), pl.when(~pl.col("t")).then(pl.col("w")).otherwise(0).sum().alias("fpw"))
    g = s1f.join(per, on="s1", how="left").fill_null(0)
    tp, fp, fpw, n = g["tp"].to_numpy().astype(float), g["fp"].to_numpy().astype(float), g["fpw"].to_numpy().astype(float), g["n"].to_numpy().astype(float)
    a, b = f05(tp, fp, n).mean(), f05(tp, fpw, n).mean()
    print(f"  {label:34s} plain holdout F0.5={a:.5f}   DMV (test-density) F0.5={b:.5f}   FP={fp.sum():.0f} weightedFP={fpw.sum():.0f}")
    return a, b

dec = json.load(open("analysis/aws_v5/decision_v5.json"))
res = {}
for f in [0, 8, 9]:
    d = pl.scan_parquet("work/v5_p2/train/*.parquet").filter(pl.col("fold") == f).select("s1", "m", "y", "fold", "p1", "pc2", "pa", "p2x2").collect()
    truth = d.filter(pl.col("y")).select("s1", "m"); s1f = s1t.filter(pl.col("fold") == f)
    pred = pair_cats(decide(prep(d, dec["tau"], "p2x2"), dec))
    print(f"fold {f}:")
    res[f] = {"V5": evaluate(pred, truth, s1f, "V5 decision"),
              "V6": evaluate(pred.filter(~pl.col("has_desc")), truth, s1f, "V6 (= V5 + descriptor rule)")}
json.dump({str(k): v for k, v in res.items()}, open("analysis/out/113_dmv.json", "w"), indent=1)
print("LB reference: V5 = 0.967 (public)")
