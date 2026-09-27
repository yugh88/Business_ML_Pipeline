"""Empty-address identical-name records next to a twin: holdout (aug, has synthetic twins) true rate by twin presence, and
test share of accepted such records that sit in twin-present S1 (label-free)."""
import sys, json
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl
from tune_decision import prep, decide
exec(open("scripts/187_excess_conf.py").read().split("t1 = pl.read_parquet")[0])      # cat(), simplify(), d9, SYN
def cat(d, split):
    ids1 = d.select(pl.col("s1").alias("entity_id")).unique(); ids2 = d.select(pl.col("m").alias("entity_id")).unique()
    srcs = [pl.scan_parquet(f"work/{split}_s{s}_norm.parquet").select("entity_id", "ncore", "a", "nums") for s in (1, 2, 3)]
    if split == "train":
        srcs.append(pl.scan_parquet("work/aug_v8_norm.parquet").select("entity_id", "ncore", "a", "nums"))
    n = pl.concat(srcs).join(pl.concat([ids1, ids2]).lazy(), on="entity_id", how="semi").collect()
    N = {r[0]: r[1:] for r in n.iter_rows()}
    return d.with_columns(pl.Series("nrel", [name_rel(N[a][0], N[b][0]) for a, b in zip(d["s1"], d["m"])]),
                          pl.Series("arel", [addr_rel(N[a][1], N[b][1], N[a][2], N[b][2]) for a, b in zip(d["s1"], d["m"])]))


def ctx(p3, split):
    acc = decide(prep(p3, d9["tau"], "p3"), d9).select("s1", "m").with_columns(pl.lit(True).alias("acc"))
    x = p3.filter(pl.col("p1") >= d9["tau"]).join(acc, on=["s1", "m"], how="left").with_columns(pl.col("acc").fill_null(False))
    x = cat(x, split)
    ids = pl.concat([x.select(pl.col("s1").alias("entity_id")), x.select(pl.col("m").alias("entity_id"))]).unique()
    nm = pl.concat([pl.scan_parquet(f"work/{split}_s{s}_norm.parquet").select("entity_id", "nums") for s in (1, 2, 3)] +
                   ([pl.scan_parquet("work/aug_v8_norm.parquet").select("entity_id", "nums")] if split == "train" else [])).join(ids.lazy(), on="entity_id", how="semi").collect()
    H = dict(zip(nm["entity_id"], nm["nums"].str.split(" ").list.first().fill_null("")))
    x = x.with_columns(pl.Series("h1", [H.get(s, "") for s in x["s1"]]), pl.Series("h2", [H.get(m, "") for m in x["m"]]))
    dd = pl.col("h2").cast(pl.Int64, strict=False) - pl.col("h1").cast(pl.Int64, strict=False)
    x = x.with_columns(((pl.col("nrel") == "N0_identical") & (dd >= 1) & (dd <= 99)).alias("is_twin"))
    x = x.with_columns(pl.col("is_twin").any().over("s1").alias("twin_present"), (pl.col("is_twin") & pl.col("acc")).any().over("s1").alias("twin_accepted"))
    return x.filter((pl.col("nrel") == "N0_identical") & (pl.col("arel") == "A3_empty"))
h = pl.concat([pl.read_parquet(f"output_v9/p3/train/fold{f}.parquet") for f in (0, 8, 9)])
hx = ctx(h, "train")
print("HOLDOUT (aug) identical-name / empty-address candidates by twin presence:")
print(hx.group_by("twin_present").agg(pl.len().alias("cands"), pl.col("y").mean().round(3).alias("true_rate"), pl.col("acc").mean().round(3).alias("acc_rate"),
                                      pl.col("y").filter(pl.col("acc")).mean().round(3).alias("precision_of_accepted"), pl.col("p3").filter(pl.col("acc")).mean().round(3).alias("mean_p3_acc")).sort("twin_present").rows())
for c in ("US", "India", "France"):
    tx = ctx(pl.read_parquet(f"output_v9/p3/test/{c}.parquet"), "test")
    t = tx.group_by("twin_present").agg(pl.len().alias("cands"), pl.col("acc").mean().round(3).alias("acc_rate"), pl.col("acc").sum().alias("n_acc"), pl.col("p3").filter(pl.col("acc")).mean().round(3).alias("mean_p3_acc"))
    print(f"TEST {c}:", t.sort("twin_present").rows())
