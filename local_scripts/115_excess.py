"""Label-free FP estimate: predicted pairs per S1 by (name rel x addr rel) category — train holdout (TP/FP split by truth)
vs test (V5/V6 predictions), per country. Same generator for true copies => test excess over train TP density ~ test FPs."""
import sys, json; sys.path.insert(0, "aws/scripts"); import memguard_local
import polars as pl, duckdb
src = open("scripts/110_relation_taxonomy.py").read()
exec(src.split("def best_pairs")[0]); exec(src[src.index("def name_rel"):src.index("def relate")])
sys.path.insert(0, "scripts"); from tune_decision import prep, decide
con = duckdb.connect()

def cats(pred, split):
    ids = pl.concat([pred.select(pl.col("s1").alias("entity_id")), pred.select(pl.col("m").alias("entity_id"))]).unique()
    n = pl.concat([pl.scan_parquet(f"work/{split}_s{s}_norm.parquet").select("entity_id", "ncore", "a", "nums") for s in (1, 2, 3)]).join(ids.lazy(), on="entity_id", how="semi").collect()
    x = pred.join(n.rename({"entity_id": "s1", "ncore": "c1", "a": "a1", "nums": "n1"}), on="s1").join(n.rename({"entity_id": "m", "ncore": "c2", "a": "a2", "nums": "n2"}), on="m")
    return x.with_columns(pl.Series("nrel", [name_rel(a, b) for a, b in zip(x["c1"].to_list(), x["c2"].to_list())]),
                          pl.Series("arel", [addr_rel(a, b, c, d) for a, b, c, d in zip(x["a1"].to_list(), x["a2"].to_list(), x["n1"].to_list(), x["n2"].to_list())]))

dec = json.load(open("analysis/aws_v5/decision_v5.json"))
drop = pl.read_parquet("work/v5_dropped_s1.parquet").select("s1")
t1 = pl.read_parquet("work/train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"}).join(drop, on="s1", how="anti")
t1 = t1.with_columns((pl.col("s1").hash() % 1).alias("dummy"))
rows = []
# train holdout folds 0/8/9
d = pl.scan_parquet("work/v5_p2/train/*.parquet").filter(pl.col("fold").is_in([0, 8, 9])).select("s1", "m", "y", "fold", "country", "p1", "pc2", "pa", "p2x2").collect()
nS1 = {c: v for c, v in d.join(t1.select("s1"), on="s1", how="semi").group_by("country").agg(pl.col("s1").n_unique()).rows()}
# count S1 in those folds (including those without candidates) via gt folds
gtf = con.execute("select source1_entity_id s1, hash(source1_entity_id || 'fold') % 10 fold from 'work/train_gt.parquet'").pl()
hold = gtf.filter(pl.col("fold").is_in([0, 8, 9])).join(t1, on="s1")
nS1 = dict(hold.group_by("country").len().rows())
print("holdout S1 per country:", nS1)
pred = decide(prep(d.drop("country"), dec["tau"], "p2x2"), dec).select("s1", "m").join(d.select("s1", "m", "y", "country"), on=["s1", "m"])
del d
x = cats(pred, "train")
tr = x.group_by("country", "nrel", "arel").agg(pl.col("y").sum().alias("tp"), (~pl.col("y")).sum().alias("fp"))
tr = tr.with_columns(pl.col("country").replace_strict(nS1, return_dtype=pl.Float64).alias("nS1"))
tr.write_parquet("work/excess_train.parquet")
# test predictions
s1c = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
nS1t = dict(s1c.group_by("country").len().rows())
for v in ["output_v5", "output_v6"]:
    p = con.execute(f"select source1_entity_id s1, unnest(string_split(matched_entity_ids, ',')) m from read_csv('{v}/matching_results.tsv', delim='\t', header=true, all_varchar=true, quote='') where matched_entity_ids is not null and matched_entity_ids<>''").pl()
    parts = []
    for c in ["US", "India", "France"]:
        pc = p.join(s1c.filter(pl.col("country") == c), on="s1")
        xc = cats(pc, "test").group_by("country", "nrel", "arel").len().rename({"len": "npred"})
        parts.append(xc); print("  categorized", v, c, flush=True)
    te = pl.concat(parts).with_columns(pl.col("country").replace_strict(nS1t, return_dtype=pl.Float64).alias("nS1"))
    te.write_parquet(f"work/excess_test_{v}.parquet")
