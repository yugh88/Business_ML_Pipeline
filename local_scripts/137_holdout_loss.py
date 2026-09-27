"""Decompose V7 holdout loss: FP-only oracle, FN-only oracle; FN by reason (not in candidates / p1<tau / rejected) and kind."""
import sys, json; sys.path.insert(0, "aws/scripts"); import memguard_local
import polars as pl, numpy as np, duckdb
exec(open("scripts/136_v7_variants.py").read().split("gt = duckdb.connect()")[0])
gt = duckdb.connect().execute("select source1_entity_id s1, case when matched_entity_ids is null or matched_entity_ids='' then 0 else len(string_split(matched_entity_ids, ',')) end n, hash(source1_entity_id || 'fold') % 10 fold from 'work/train_gt.parquet'").pl()
gt = gt.join(pl.read_parquet("work/v5_dropped_s1.parquet").select("s1"), on="s1", how="anti")
HOLD = True
f = 0
d = pl.scan_parquet("work/v5_p2/train/*.parquet").filter(pl.col("fold") == f).select("s1", "m", "y", "fold", "country", "p1", "pc2", "pa", "p2x2").collect()
x = build(d.drop("country"), "train").join(d.select("s1", "m", "y", "country"), on=["s1", "m"])
x = enrich(x, "train"); r7 = __import__('functools').reduce(lambda a, b: a | b, [RULES[k](x) for k in RULES])
pred = x.filter(~r7).select("s1", "m", "y", "off")
s1f = gt.filter(pl.col("fold") == f)
per = pred.group_by("s1").agg(pl.col("y").sum().alias("tp"), (~pl.col("y")).sum().alias("fp"))
g = s1f.join(per, on="s1", how="left").fill_null(0)
tp, fp, n = [g[c].to_numpy().astype(float) for c in ("tp", "fp", "n")]
F = f05(tp, fp, n); Fnofp = f05(tp, 0 * fp, n); Fnofn = f05(n * (n > 0) + tp * (n == 0), fp, n)
print(f"fold {f}: V7 F {F.mean():.5f}  | no-FP oracle {Fnofp.mean():.5f} (FP loss {Fnofp.mean()-F.mean():.5f}) | perfect-recall oracle {Fnofn.mean():.5f} (FN loss {Fnofn.mean()-F.mean():.5f})")
print("  singletons: share", (n == 0).mean().round(4), " singletons with predictions (F=0):", int(((n == 0) & (tp + fp > 0)).sum()), " loss", round(float(((n == 0) & (tp + fp > 0)).sum() / len(n)), 5))
print("  S1 with >=1 FP:", int((fp > 0).sum()), f"({(fp>0).mean():.4f})  S1 with >=1 FN:", int((tp < n).sum()), f"({(tp<n).mean():.4f})")
# FN reasons
tr = pl.read_parquet("work/train_pairs.parquet").join(s1f.select("s1"), on="s1")
cand = d.select("s1", "m", "p1", "p2x2")
fn = tr.join(pred.filter(pl.col("y")).select("s1", "m"), on=["s1", "m"], how="anti").join(cand, on=["s1", "m"], how="left")
fn = fn.with_columns(pl.when(pl.col("p1").is_null()).then(pl.lit("not_in_candidates")).when(pl.col("p1") < dec["tau"]).then(pl.lit("p1<tau")).when(pl.col("p2x2") < 0.5).then(pl.lit("p2<0.5")).otherwise(pl.lit("p2>=0.5 but rejected")).alias("why"))
fn = attach(fn, "train")
print("  FN pairs:", fn.height, "per S1:", round(fn.height / s1f.height, 4))
print(fn.group_by("why").len().sort("len", descending=True).rows())
print(fn.group_by("why", "off").len().sort("len", descending=True).head(15).rows())
fpp = pred.filter(~pl.col("y"))
print("  FP pairs:", fpp.height, fpp.group_by("off").len().sort("len", descending=True).rows())
