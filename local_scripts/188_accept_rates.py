"""FN signature: acceptance RATE (accepted / cascade candidates) per category, holdout (clean) vs test, V9.
In categories with ~no test-side excess negatives (e.g. US identical name & same number), a lower test acceptance rate = missed copies."""
import sys, json
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl
from tune_decision import prep, decide
exec(open("scripts/187_excess_conf.py").read().split("t1 = pl.read_parquet")[0])        # cat(), simplify(), d9, SYN
t1 = pl.read_parquet("work/train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
def rates(p3):
    acc = decide(prep(p3, d9["tau"], "p3"), d9).select("s1", "m").with_columns(pl.lit(True).alias("acc"))
    x = p3.filter(pl.col("p1") >= d9["tau"]).join(acc, on=["s1", "m"], how="left").with_columns(pl.col("acc").fill_null(False))
    x = simplify(cat(x, "train" if "y" in x.columns and x["y"].any() else "test").with_columns(pl.col("p3")))
    return x
h = pl.concat([pl.read_parquet(f"output_v9/p3/train/fold{f}.parquet") for f in (0, 8, 9)]).join(SYN, on="m", how="anti").join(t1, on="s1")
for c in ("US", "India"):
    hx = rates(h.filter(pl.col("country") == c))
    tx = rates(pl.read_parquet(f"output_v9/p3/test/{c}.parquet"))
    a = hx.group_by("n", "a").agg(pl.len().alias("h_cands"), pl.col("acc").mean().round(4).alias("h_acc"), pl.col("y").mean().round(4).alias("h_true"),
                                  (pl.col("y") & ~pl.col("acc")).sum().alias("h_fn"))
    b = tx.group_by("n", "a").agg(pl.len().alias("t_cands"), pl.col("acc").mean().round(4).alias("t_acc"), (pl.col("p3") > 0.5).mean().round(4).alias("t_p3gt.5"))
    j = a.join(b, on=["n", "a"], how="full", coalesce=True).with_columns((pl.col("t_acc") - pl.col("h_acc")).round(4).alias("d_acc")).sort("h_cands", descending=True)
    print(f"==== {c}"); print(j.head(12))
