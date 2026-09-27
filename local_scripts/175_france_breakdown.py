"""France: accepted pairs by category for V5 vs V5+stage-3 (and pairwise p1): where does stage-3 accept more?"""
import sys, json; sys.path.insert(0, "aws/scripts"); import memguard_local
import polars as pl, duckdb
from tune_decision import prep, decide
FRN = {"services", "service", "cie", "compagnie", "fils", "associes", "frs", "groupe", "france", "developpement"}
FRS = {"holding", "participations", "international", "internationale", "distribution", "snc"}
ASSOC = {"club","ecole","amicale","comite","sportive","amis","parents","college","union","fetes","primaire","anciens","maison","sante","pharmacie",
         "loisirs","sport","lycee","culture","jeunes","culturelle","foyer","section","groupement","maternelle","elementaire","collectif","danse",
         "patrimoine","musique","institut","residence","conseil","theatre","atelier","soins","ehpad","medico","sportif","agricole","auto","ateliers",
         "cafe","centre","societe","federation","association","culturel","gestion"}
con = duckdb.connect()
def load(path):
    return con.execute(f"select source1_entity_id s1, unnest(string_split(matched_entity_ids, ',')) m from read_csv('{path}', delim='\t', header=true, all_varchar=true, quote='') where matched_entity_ids is not null and matched_entity_ids<>''").pl()
s1c = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"}).filter(pl.col("country") == "France")
v5 = load("output_v5/matching_results.tsv").join(s1c.select("s1"), on="s1", how="semi").with_columns(pl.lit(True).alias("v5"))
s3 = load("output_v5s3/matching_results.tsv").join(s1c.select("s1"), on="s1", how="semi").with_columns(pl.lit(True).alias("s3"))
t = pl.scan_parquet("work/v5_p2/test/*.parquet").filter((pl.col("country") == "France") & (pl.col("p1") >= 0.05)).select("s1", "m", "p1", "pc2", "pa", "p2x2").collect()
p1 = decide(prep(t, 0.05, "p1"), dict(policy="G_expected_f", a=1.25, lam=0.0, cap=1)).with_columns(pl.lit(True).alias("p1acc"))
x = v5.join(s3, on=["s1", "m"], how="full", coalesce=True).join(p1, on=["s1", "m"], how="left").with_columns([pl.col(c).fill_null(False) for c in ("v5", "s3", "p1acc")])
ids = pl.concat([x.select(pl.col("s1").alias("entity_id")), x.select(pl.col("m").alias("entity_id"))]).unique()
nrm = pl.concat([pl.scan_parquet(f"work/test_s{s}_norm.parquet").select("entity_id", "ncore", "nums") for s in (1, 2, 3)]).join(ids.lazy(), on="entity_id", how="semi").collect()
x = x.join(nrm.rename({"entity_id": "s1", "ncore": "c1", "nums": "n1"}), on="s1").join(nrm.rename({"entity_id": "m", "ncore": "c2", "nums": "n2"}), on="m")
E = pl.col("c2").str.split(" ").list.set_difference(pl.col("c1").str.split(" ")); M = pl.col("c1").str.split(" ").list.set_difference(pl.col("c2").str.split(" "))
h1, h2 = pl.col("n1").str.split(" ").list.first().fill_null(""), pl.col("n2").str.split(" ").list.first().fill_null("")
dd = h2.cast(pl.Int64, strict=False) - h1.cast(pl.Int64, strict=False)
x = x.with_columns(pl.when((h1 == "") | (h2 == "")).then(pl.lit("empty")).when(h1 == h2).then(pl.lit("same")).when((dd >= 1) & (dd <= 9)).then(pl.lit("+k")).otherwise(pl.lit("other")).alias("off"),
                   pl.when(E.list.len() == 0).then(pl.lit("no_extra")).when(E.list.eval(pl.element().is_in(list(FRS))).list.any()).then(pl.lit("FR_SIB"))
                   .when(E.list.eval(pl.element().is_in(list(FRN))).list.all()).then(pl.lit("FR_NOISE"))
                   .when(E.list.eval(pl.element().is_in(list(ASSOC))).list.any() & M.list.eval(pl.element().is_in(list(ASSOC))).list.any()).then(pl.lit("ASSOC_SWAP"))
                   .otherwise(pl.lit("other_extra")).alias("wcls"))
n = s1c.height / 1000
t2 = x.group_by("wcls", "off").agg((pl.col("v5").sum() / n).round(2).alias("v5_k"), (pl.col("s3").sum() / n).round(2).alias("s3_k"), (pl.col("p1acc").sum() / n).round(2).alias("p1_k"),
                                   ((pl.col("s3") & ~pl.col("v5")).sum() / n).round(2).alias("added_k"), ((pl.col("v5") & ~pl.col("s3")).sum() / n).round(2).alias("removed_k")).sort("added_k", descending=True)
pl.Config.set_tbl_rows(30); pl.Config.set_tbl_width_chars(200)
print(t2)
