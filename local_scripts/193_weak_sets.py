"""Singleton / weak-evidence-set hypothesis. For each S1 with a non-empty accepted set: does the set contain a STRONG-evidence
copy (identical / light-edit / typo name AND same number & street or identical address)? Holdout: share of weak-evidence sets,
their per-S1 F and how many are true singletons. Test: share of weak-evidence sets per country."""
import sys, json
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl, numpy as np, duckdb
from tune_decision import prep, decide
src = open("scripts/110_relation_taxonomy.py").read(); exec(src.split("def best_pairs")[0]); exec(src[src.index("def name_rel"):src.index("def relate")])
d9 = json.load(open("output_v9/decision_s3.json"))
SYN = pl.concat([pl.read_parquet(f"work/aug_v8_{s}.parquet", columns=["entity_id"]) for s in ("s2", "s3")]).rename({"entity_id": "m"})
STRONG_N = ["N0_identical", "N1_extra_noise", "N4_dropped_words", "N5_typo_subst"]


def sets(p3, split):
    acc = decide(prep(p3, d9["tau"], "p3"), d9).select("s1", "m")
    ids = pl.concat([acc.select(pl.col("s1").alias("entity_id")), acc.select(pl.col("m").alias("entity_id"))]).unique()
    n = pl.concat([pl.scan_parquet(f"work/{split}_s{s}_norm.parquet").select("entity_id", "ncore", "a", "nums") for s in (1, 2, 3)]).join(ids.lazy(), on="entity_id", how="semi").collect()
    N = {r[0]: r[1:] for r in n.iter_rows()}
    x = acc.with_columns(pl.Series("nrel", [name_rel(N[a][0], N[b][0]) for a, b in zip(acc["s1"], acc["m"])]),
                         pl.Series("arel", [addr_rel(N[a][1], N[b][1], N[a][2], N[b][2]) for a, b in zip(acc["s1"], acc["m"])]))
    x = x.with_columns((pl.col("nrel").is_in(STRONG_N) & pl.col("arel").is_in(["A0_identical", "A1_same_num"])).alias("strong"))
    return x.group_by("s1").agg(pl.col("strong").any().alias("has_strong"), pl.len().alias("k"),
                                pl.col("arel").filter(~pl.col("strong")).first().alias("arel_weak"), pl.col("nrel").filter(~pl.col("strong")).first().alias("nrel_weak"))


gt = duckdb.connect().execute("select source1_entity_id s1, case when matched_entity_ids is null or matched_entity_ids='' then 0 else len(string_split(matched_entity_ids, ',')) end n, "
                              "hash(source1_entity_id || 'fold') % 10 fold from 'work/train_gt.parquet'").pl().join(pl.read_parquet("work/v5_dropped_s1.parquet").select("s1"), on="s1", how="anti")
hold = gt.filter(pl.col("fold").is_in([0, 8, 9]))
parts = []
for f in (0, 8, 9):
    p3 = pl.read_parquet(f"output_v9/p3/train/fold{f}.parquet").join(SYN, on="m", how="anti")
    s = sets(p3, "train")
    acc = decide(prep(p3, d9["tau"], "p3"), d9).select("s1", "m").join(p3.select("s1", "m", "y"), on=["s1", "m"])
    s = s.join(acc.group_by("s1").agg(pl.col("y").sum().alias("tp"), (~pl.col("y")).sum().alias("fp")), on="s1")
    parts.append(s)
h = pl.concat(parts).join(hold, on="s1")
tp, fp, n = [h[c].to_numpy().astype(float) for c in ("tp", "fp", "n")]
P = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-9), 0); R = np.where(n > 0, tp / np.maximum(n, 1), 0)
h = h.with_columns(pl.Series("f", np.where(n == 0, 0.0, np.where(tp > 0, 1.25 * P * R / np.maximum(0.25 * P + R, 1e-12), 0.0))))
NH = hold.height
print("HOLDOUT: S1 with a non-empty accepted set:", h.height, f"of {NH}")
print(h.group_by("has_strong").agg(pl.len().alias("S1"), (pl.len() / NH * 1000).round(2).alias("per1k"), pl.col("f").mean().round(4).alias("mean_F"),
                                   (pl.col("n") == 0).sum().alias("true_singletons"), ((1 - pl.col("f")).sum() / NH).round(5).alias("F_loss_contrib")).rows())
print("   weak sets by first weak record relation:", h.filter(~pl.col("has_strong")).group_by("arel_weak").agg(pl.len(), pl.col("f").mean().round(3)).sort("len", descending=True).head(6).rows())
s1c = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"}); nt = dict(s1c.group_by("country").len().rows())
for c in ("US", "India", "France"):
    t = sets(pl.read_parquet(f"output_v9/p3/test/{c}.parquet"), "test")
    w = t.filter(~pl.col("has_strong"))
    print(f"TEST {c}: weak-evidence sets {w.height} = {1000*w.height/nt[c]:.2f} per 1k S1;  by relation:",
          w.group_by("arel_weak").len().sort("len", descending=True).head(5).with_columns((pl.col("len") / nt[c] * 1000).round(2)).rows())
