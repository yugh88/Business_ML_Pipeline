"""Triangulation signature: inside an S1's accepted set, do ≥2 records share the SAME non-trivial name edit relative to
the S1 (same extra / missing tokens, not just copy-noise words)? Copies carry independent noise; a sibling entity's copies
share the sibling's name. Train truth vs V9 holdout predictions vs V9 test predictions (and V5s3r)."""
import sys, json
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl, duckdb
from tune_decision import prep, decide
wc = json.load(open("aws/scripts/word_classes.json")); NOISE = set(wc["noise"]) | {"llc", "inc", "ltd", "limited", "pvt", "private", "corp", "corporation", "co",
     "company", "plc", "llp", "lp", "pc", "pllc", "sarl", "sas", "sa", "eurl", "sasu", "the", "and", "of", "services", "service", "center", "centre", "cie", "s", "sci"}
con = duckdb.connect()
SYN = pl.concat([pl.read_parquet(f"work/aug_v8_{s}.parquet", columns=["entity_id"]) for s in ("s2", "s3")]).rename({"entity_id": "m"})


def signature_stats(pairs, split, label):
    ids1 = pairs.select(pl.col("s1").alias("entity_id")).unique(); ids2 = pairs.select(pl.col("m").alias("entity_id")).unique()
    n1 = pl.scan_parquet(f"work/{split}_s1_norm.parquet").select("entity_id", "ncore", "nums").join(ids1.lazy(), on="entity_id", how="semi").collect()
    n2 = pl.concat([pl.scan_parquet(f"work/{split}_s{s}_norm.parquet").select("entity_id", "ncore", "nums") for s in (2, 3)]).join(ids2.lazy(), on="entity_id", how="semi").collect()
    x = pairs.join(n1.rename({"entity_id": "s1", "ncore": "c1", "nums": "n1"}), on="s1").join(n2.rename({"entity_id": "m", "ncore": "c2", "nums": "n2"}), on="m")
    E = pl.col("c2").str.split(" ").list.set_difference(pl.col("c1").str.split(" ")).list.sort()
    M = pl.col("c1").str.split(" ").list.set_difference(pl.col("c2").str.split(" ")).list.sort()
    x = x.with_columns(E.alias("E"), M.alias("M"), pl.col("m").str.slice(0, 2).alias("src"), pl.col("n2").str.split(" ").list.first().fill_null("").alias("h2"))
    x = x.with_columns(E.list.eval(pl.element().is_in(list(NOISE)).not_()).list.any().alias("nontrivial"),
                       pl.concat_str(pl.col("E").list.join(" "), pl.lit(" | "), pl.col("M").list.join(" ")).alias("sig"))
    s = x.filter(pl.col("nontrivial"))
    g = s.group_by("s1", "sig").agg(pl.len().alias("n"), pl.col("src").n_unique().alias("nsrc"), pl.col("h2").n_unique().alias("nnum"))
    k = x.group_by("s1").len()
    shared = g.filter(pl.col("n") >= 2)
    out = dict(label=label, S1_with_sets=k.height, records=x.height, nontrivial_share=round(float(x["nontrivial"].mean()), 4),
               S1_shared_sig_per1k=round(1000 * shared["s1"].n_unique() / k.height, 2),
               S1_shared_sig_xsrc_per1k=round(1000 * shared.filter(pl.col("nsrc") == 2)["s1"].n_unique() / k.height, 2),
               records_in_shared_per1k=round(1000 * float(shared["n"].sum()) / k.height, 2))
    print(out, flush=True)
    return x, shared


gt = con.execute("select source1_entity_id s1, hash(source1_entity_id || 'fold') % 10 fold from 'work/train_gt.parquet'").pl()
gt = gt.join(pl.read_parquet("work/v5_dropped_s1.parquet").select("s1"), on="s1", how="anti").filter(pl.col("fold").is_in([0, 8, 9]))
truth = pl.read_parquet("work/train_pairs.parquet").select("s1", "m").join(gt.select("s1"), on="s1", how="semi")
signature_stats(truth, "train", "TRAIN TRUTH (holdout S1, true copy sets)")
d9 = json.load(open("output_v9/decision_s3.json"))
hp = []
for f in (0, 8, 9):
    p9 = pl.read_parquet(f"output_v9/p3/train/fold{f}.parquet").join(SYN, on="m", how="anti")
    hp.append(decide(prep(p9, d9["tau"], "p3"), d9).select("s1", "m").join(p9.select("s1", "m", "y"), on=["s1", "m"]))
hp = pl.concat(hp)
hx, hsh = signature_stats(hp.select("s1", "m"), "train", "V9 HOLDOUT predictions (clean)")
# how often are records in shared-signature groups (holdout predictions) actually true?
hx = hx.join(hp, on=["s1", "m"]).join(hsh.select("s1", "sig", pl.lit(True).alias("in_shared")), on=["s1", "sig"], how="left").with_columns(pl.col("in_shared").fill_null(False))
print("   holdout: true rate of records in shared non-trivial-edit groups:", round(float(hx.filter(pl.col("in_shared"))["y"].mean()), 4),
      " (cross-source groups:", round(float(hx.join(hsh.filter(pl.col("nsrc") == 2).select("s1", "sig"), on=["s1", "sig"], how="semi")["y"].mean()), 4), ")",
      " other non-trivial:", round(float(hx.filter(~pl.col("in_shared") & pl.col("nontrivial"))["y"].mean()), 4))
load = lambda p: con.execute(f"select source1_entity_id s1, unnest(string_split(matched_entity_ids, ',')) m from read_csv('{p}', delim='\t', header=true, all_varchar=true, quote='') "
                             f"where matched_entity_ids is not null and matched_entity_ids<>''").pl()
s1c = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
for name in ("output_v5s3r", "output_v9"):
    p = load(f"{name}/matching_results.tsv")
    for c in ("US", "India", "France"):
        signature_stats(p.join(s1c.filter(pl.col("country") == c), on="s1").select("s1", "m"), "test", f"TEST {name} {c}")
tx, tsh = signature_stats(load("output_v9/matching_results.tsv").select("s1", "m"), "test", "TEST V9 all")
tsh.join(tx.select("s1", "sig", "m", "src", "h2"), on=["s1", "sig"]).write_parquet("work/test_v9_shared_sig.parquet")
print("top shared signatures (test V9):", tsh.group_by("sig").agg(pl.len().alias("S1s"), pl.col("nsrc").mean().round(2)).sort("S1s", descending=True).head(15).rows())
