"""Forensics V5s3r (LB 0.973) -> V9 (LB 0.977). Part 1: pair diff by category on TEST, and the same diff on the labelled
HOLDOUT (clean view) to get the correctness of each category of change on train-like data."""
import sys, json, functools
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl, numpy as np, duckdb
from tune_decision import prep, decide
exec(open("scripts/130_rules_v7.py").read().split("gt = duckdb.connect()")[0])      # enrich(), RULES (R1/R2), f05
src = open("scripts/110_relation_taxonomy.py").read()
exec(src.split("def best_pairs")[0]); exec(src[src.index("def name_rel"):src.index("def relate")])
con = duckdb.connect()
SYN = pl.concat([pl.read_parquet(f"work/aug_v8_{s}.parquet", columns=["entity_id"]) for s in ("s2", "s3")]).rename({"entity_id": "m"})
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(230)


def categorize(d, split):
    ids1 = d.select(pl.col("s1").alias("entity_id")).unique(); ids2 = d.select(pl.col("m").alias("entity_id")).unique()
    n1 = pl.scan_parquet(f"work/{split}_s1_norm.parquet").select("entity_id", "ncore", "nums").join(ids1.lazy(), on="entity_id", how="semi").collect()
    n2 = pl.concat([pl.scan_parquet(f"work/{split}_s{s}_norm.parquet").select("entity_id", "ncore", "nums") for s in (2, 3)]).join(ids2.lazy(), on="entity_id", how="semi").collect()
    x = d.join(n1.rename({"entity_id": "s1", "ncore": "c1", "nums": "n1"}), on="s1", how="left").join(n2.rename({"entity_id": "m", "ncore": "c2", "nums": "n2"}), on="m", how="left")
    x = x.with_columns([pl.col(c).fill_null("") for c in ("c1", "c2", "n1", "n2")])
    h1, h2 = pl.col("n1").str.split(" ").list.first().fill_null(""), pl.col("n2").str.split(" ").list.first().fill_null("")
    dd = h2.cast(pl.Int64, strict=False) - h1.cast(pl.Int64, strict=False)
    x = x.with_columns(pl.when((h1 == "") | (h2 == "")).then(pl.lit("empty")).when(h1 == h2).then(pl.lit("same")).when((dd >= 1) & (dd <= 9)).then(pl.lit("+k"))
                       .when((dd <= -1) & (dd >= -9)).then(pl.lit("-k")).when(dd.abs() <= 99).then(pl.lit("d10-99")).otherwise(pl.lit("d100+")).alias("off"),
                       pl.Series("nrel", [name_rel(a, b) for a, b in zip(x["c1"].to_list(), x["c2"].to_list())]), h2.alias("h2"), pl.col("m").str.slice(0, 2).alias("src"))
    x = x.with_columns(pl.when(pl.col("nrel").is_in(["N2_extra_DESCRIPTOR", "N6_word_swap_DESC"])).then(pl.lit("DESC")).when(pl.col("nrel") == "N0_identical").then(pl.lit("N0"))
                       .when(pl.col("nrel").is_in(["N1_extra_noise", "N4_dropped_words", "N5_typo_subst"])).then(pl.lit("light_edit")).otherwise(pl.lit("heavy_edit")).alias("ncls"))
    return x.drop("c1", "c2", "n1", "n2")


def diff(a, b, split):
    """a = old set, b = new set (s1, m [, y]). Adds per-pair context: status removed/added, S1 set sizes, cross-source number partner."""
    a = a.with_columns(pl.lit(True).alias("in_a")); b = b.with_columns(pl.lit(True).alias("in_b"))
    j = a.join(b, on=["s1", "m"], how="full", coalesce=True).with_columns(pl.col("in_a").fill_null(False), pl.col("in_b").fill_null(False))
    ka = a.group_by("s1").len().rename({"len": "k_old"}); kb = b.group_by("s1").len().rename({"len": "k_new"})
    ch = j.filter(pl.col("in_a") != pl.col("in_b"))
    ch = categorize(ch, split)
    # cross-source partner in the version that contains the pair: another record of the same S1, other source, same house number
    both = categorize(j.select("s1", "m", "in_a", "in_b").join(ch.select("s1").unique(), on="s1", how="semi"), split).select("s1", "m", "in_a", "in_b", "h2", "src")
    def partner(flag):
        s = both.filter(pl.col(flag) & (pl.col("h2") != "")).select("s1", "m", "h2", "src")
        p = s.join(s, on=["s1", "h2"], suffix="_o").filter((pl.col("src") != pl.col("src_o"))).select("s1", "m").unique().with_columns(pl.lit(True).alias(f"xs_{flag}"))
        return p
    ch = ch.join(partner("in_a"), on=["s1", "m"], how="left").join(partner("in_b"), on=["s1", "m"], how="left")
    ch = ch.with_columns(pl.when(pl.col("in_a")).then(pl.col("xs_in_a")).otherwise(pl.col("xs_in_b")).fill_null(False).alias("xs_partner"))
    ch = ch.join(ka, on="s1", how="left").join(kb, on="s1", how="left").with_columns(pl.col("k_old").fill_null(0), pl.col("k_new").fill_null(0))
    return ch.with_columns(pl.when(pl.col("in_a")).then(pl.lit("REMOVED")).otherwise(pl.lit("ADDED")).alias("chg"),
                           (pl.col("k_new") == 0).alias("s1_now_empty"), (pl.col("k_old") == 0).alias("s1_was_empty"))


load = lambda p: con.execute(f"select source1_entity_id s1, unnest(string_split(matched_entity_ids, ',')) m from read_csv('{p}', delim='\t', header=true, all_varchar=true, quote='') "
                             f"where matched_entity_ids is not null and matched_entity_ids<>''").pl()
s1c = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"}); nS1 = dict(s1c.group_by("country").len().rows())
te = diff(load("output_v5s3r/matching_results.tsv"), load("output_v9/matching_results.tsv"), "test").join(s1c, on="s1")
te.write_parquet("work/forensic_test_diff.parquet")
N = sum(nS1.values()) / 1000
print(f"TEST V5s3r->V9: changed pairs {te.height} ({te.height/N:.1f}/1k S1), S1 touched {te['s1'].n_unique()} ({te['s1'].n_unique()/N:.1f}/1k)")
t = te.group_by("chg", "off", "ncls").agg((pl.len() / N).round(2).alias("per1k"), pl.col("xs_partner").mean().round(2).alias("xs_partner"),
                                          pl.col("s1_now_empty").mean().round(2).alias("now_empty"), pl.col("s1_was_empty").mean().round(2).alias("was_empty")).sort("per1k", descending=True)
print(t.head(24))
print("by country:", te.group_by("country", "chg").agg((pl.len() / (pl.col("country").replace_strict(nS1, return_dtype=pl.Float64).first() / 1000)).round(1).alias("per1k")).sort("country", "chg").rows())
print("S1 emptied by V9:", te.filter(pl.col("s1_now_empty")).select("s1").n_unique(), "  S1 newly non-empty:", te.filter(pl.col("s1_was_empty")).select("s1").n_unique())

# ---------------- holdout analog (clean view: no synthetic), with labels
gt = con.execute("select source1_entity_id s1, case when matched_entity_ids is null or matched_entity_ids='' then 0 else len(string_split(matched_entity_ids, ',')) end n, "
                 "hash(source1_entity_id || 'fold') % 10 fold from 'work/train_gt.parquet'").pl().join(pl.read_parquet("work/v5_dropped_s1.parquet").select("s1"), on="s1", how="anti")
t1c = pl.read_parquet("work/train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
d5 = json.load(open("output_v5s3/decision_s3.json")); d9 = json.load(open("output_v9/decision_s3.json"))
parts = []
for f in (0, 8, 9):
    p5 = pl.read_parquet(f"output_v5s3/p3/train/fold{f}.parquet")
    a = decide(prep(p5, d5["tau"], "p3"), d5).select("s1", "m").join(p5.select("s1", "m", "y"), on=["s1", "m"]).join(t1c, on="s1")
    x = enrich(a, "train"); rm = functools.reduce(lambda u, v: u | v, [RULES[k](x) for k in ("R1_twin_same_src", "R2_twin_2src")])
    a = x.filter(~rm).select("s1", "m")                                          # V5s3r analog on holdout
    p9 = pl.read_parquet(f"output_v9/p3/train/fold{f}.parquet").join(SYN, on="m", how="anti")
    b = decide(prep(p9, d9["tau"], "p3"), d9).select("s1", "m")                  # V9 analog (clean view)
    ch = diff(a, b, "train")
    truth = pl.read_parquet("work/train_pairs.parquet").select("s1", "m").with_columns(pl.lit(True).alias("y"))
    parts.append(ch.join(truth, on=["s1", "m"], how="left").with_columns(pl.col("y").fill_null(False)).join(t1c, on="s1"))
ho = pl.concat(parts); ho.write_parquet("work/forensic_holdout_diff.parquet")
NH = gt.filter(pl.col("fold").is_in([0, 8, 9])).height / 1000
print(f"\nHOLDOUT (clean) V5s3r-analog -> V9: changed {ho.height} ({ho.height/NH:.1f}/1k S1)")
h = ho.group_by("chg", "off", "ncls").agg((pl.len() / NH).round(2).alias("per1k"), pl.col("y").mean().round(3).alias("true_rate")).sort("per1k", descending=True)
print(h.head(24))
print("holdout overall: REMOVED true rate", round(float(ho.filter(pl.col("chg") == "REMOVED")["y"].mean()), 3), " ADDED true rate", round(float(ho.filter(pl.col("chg") == "ADDED")["y"].mean()), 3))
