"""Show S1s (test or train) having a PREDICTED pair in a given (nrel, arel) category, with all scored candidates.
Usage: 116_look_cat.py split country nrel arel n seed"""
import sys, json; sys.path.insert(0, "aws/scripts")
import polars as pl, duckdb
src = open("scripts/110_relation_taxonomy.py").read()
exec(src.split("def best_pairs")[0]); exec(src[src.index("def name_rel"):src.index("def relate")])
sp, ctry, NREL, AREL, n, seed = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
buckets = sorted(__import__("glob").glob(f"work/v5_p2/{sp}/*.parquet"))[:16]
d = pl.scan_parquet(buckets).filter((pl.col("country") == ctry) & (pl.col("p1") >= 0.02)).select("s1", "m", "y", "fold", "p1", "pc2", "pa", "p2x2").collect()
if sp == "test":
    pred = duckdb.connect().execute("select source1_entity_id s1, unnest(string_split(matched_entity_ids, ',')) m from read_csv('output_v6/matching_results.tsv', delim='\t', header=true, all_varchar=true, quote='') where matched_entity_ids is not null and matched_entity_ids<>''").pl()
    pred = pred.join(d.select("s1").unique(), on="s1")
else:
    sys.path.insert(0, "scripts"); from tune_decision import prep, decide
    dec = json.load(open("analysis/aws_v5/decision_v5.json"))
    d = d.filter(pl.col("fold").is_in([0, 8, 9]))
    pred = decide(prep(d, dec["tau"], "p2x2"), dec).select("s1", "m")
P = set(zip(pred["s1"], pred["m"]))
ids = pl.concat([d.select(pl.col("s1").alias("entity_id")), d.select(pl.col("m").alias("entity_id"))]).unique()
nrm = pl.concat([pl.scan_parquet(f"work/{sp}_s{s}_norm.parquet").select("entity_id", "ncore", "a", "nums") for s in (1, 2, 3)]).join(ids.lazy(), on="entity_id", how="semi").collect()
raw = pl.concat([pl.scan_parquet(f"work/{sp}_s{s}.parquet") for s in (1, 2, 3)]).join(ids.lazy(), on="entity_id", how="semi").collect()
R = {r[0]: (r[1], r[2]) for r in raw.iter_rows()}; N = {r[0]: r[1:] for r in nrm.iter_rows()}
hits = [(s, m) for s, m in P if name_rel(N[s][0], N[m][0]) == NREL and addr_rel(N[s][1], N[m][1], N[s][2], N[m][2]) == AREL]
import random; random.Random(seed).shuffle(hits)
print(f"{len(hits)} predicted pairs in {NREL} x {AREL} among {len(P)} predicted pairs (sampled buckets)")
for s1, mm in hits[:n]:
    print("=" * 140); print(f"{s1}  | {R[s1][0]} | {R[s1][1]}")
    for r in d.filter(pl.col("s1") == s1).sort("p2x2", descending=True).head(10).iter_rows(named=True):
        flag = ("PRED " if (s1, r["m"]) in P else "     ") + ("TRUE " if (sp == "train" and r["y"]) else "")
        mark = " <==" if r["m"] == mm else ""
        print(f"   {flag:10s}{r['p2x2']:.3f} {r['m']:14s} | {R[r['m']][0]} | {R[r['m']][1]}{mark}")
