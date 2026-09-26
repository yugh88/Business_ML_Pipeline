"""Write the final TSVs from stage-2 score files with a FROZEN decision (chosen on train fold 7, analysis/15).
Memory-light: only test rows with p1 >= tau are read.  usage: finalize_submission.py <p2_dir> <input_dir> <out_dir> <decision.json>"""
import os, sys, json
import polars as pl
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memguard_local  # noqa: F401
from tune_decision import prep, decide

p2dir, indir, outdir, decf = sys.argv[1:5]
dec = json.load(open(decf)); os.makedirs(outdir, exist_ok=True)
te = pl.scan_parquet(os.path.join(p2dir, "test", "*.parquet")).filter(pl.col("p1") >= dec["tau"]) \
       .select("s1", "m", "p1", "pc2", "pa", dec["variant"]).collect()
cand = te.select("s1", "m").unique()
pred = decide(prep(te.with_columns(pl.lit(False).alias("y"), pl.lit(-1).alias("fold")), dec["tau"], dec["variant"]), dec)
s1 = pl.read_parquet(os.path.join(indir, "test_s1.parquet"), columns=["entity_id"]).rename({"entity_id": "s1"})
def write(pairs, col, path):
    g = pairs.sort("m").group_by("s1").agg(pl.col("m").str.join(",").alias(col))
    s1.join(g, on="s1", how="left").with_columns(pl.col(col).fill_null("")).sort("s1").rename({"s1": "source1_entity_id"}) \
      .write_csv(path + ".tmp", separator="\t", quote_style="never"); os.replace(path + ".tmp", path)
write(cand, "candidate_entity_ids", os.path.join(outdir, "candidate_pairs.tsv"))
write(pred, "matched_entity_ids", os.path.join(outdir, "matching_results.tsv"))
k = pred.group_by("s1").len()["len"]; kc = cand.group_by("s1").len()["len"]
info = dict(decision=dec, s1=s1.height, cand_pairs=cand.height, cands_per_s1=cand.height / s1.height,
            cand_quantiles=[float(kc.quantile(q)) for q in (0.5, 0.9, 0.99)] + [int(kc.max())],
            matches=pred.height, matches_per_s1=pred.height / s1.height, s1_empty=s1.height - pred["s1"].n_unique(),
            max_matches=int(k.max()), s2_share=float(pred["m"].str.starts_with("S2").mean()))
json.dump(info, open(os.path.join(outdir, "finalize_info.json"), "w"), indent=1)
print(info)
