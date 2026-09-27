"""Address fingerprint test: does a candidate's RAW address string exactly equal the raw address of ANOTHER candidate
of the same S1 in the same source (other candidate with p2x2>=0.9)? Copies of one entity share per-source versions;
distinct sibling entities should not. Train: by word class (NOISE vs SIB) and truth. France: by word class."""
import sys; sys.path.insert(0, "aws/scripts"); import memguard_local
import polars as pl
exec(open("scripts/125_word_addr2.py").read().split("tr = cands(")[0])     # cands(sp, cond), attach
def fingerprint(d, sp):
    """d: one-extra-word candidates (s1, m, ...). Compare raw address with other strong candidates of the same S1 & source."""
    allc = pl.scan_parquet(f"work/v5_p2/{sp}/*.parquet").join(d.select("s1").unique().lazy(), on="s1", how="semi") \
             .filter(pl.col("p2x2") >= 0.9).select("s1", "m").collect()
    ids = pl.concat([allc.select(pl.col("m").alias("entity_id")), d.select(pl.col("m").alias("entity_id"))]).unique()
    raw = pl.concat([pl.scan_parquet(f"work/{sp}_s{s}.parquet").select("entity_id", "business_address") for s in (2, 3)]).join(ids.lazy(), on="entity_id", how="semi").collect()
    raw = raw.with_columns(pl.col("business_address").fill_null("").alias("ra"))
    oth = allc.join(raw.rename({"entity_id": "m"}), on="m").select("s1", pl.col("m").alias("m2"), pl.col("m").str.slice(0, 2).alias("src"), "ra")
    x = d.join(raw.rename({"entity_id": "m"}).select("m", "ra"), on="m").with_columns(pl.col("m").str.slice(0, 2).alias("src"))
    j = x.join(oth, on=["s1", "src", "ra"], how="left").filter(pl.col("m2").is_null() | (pl.col("m2") != pl.col("m")))
    hit = j.filter(pl.col("m2").is_not_null()).select("s1", "m").unique().with_columns(pl.lit(True).alias("fp_share"))
    return x.join(hit, on=["s1", "m"], how="left").with_columns(pl.col("fp_share").fill_null(False), (pl.col("ra") == "").alias("ra_empty"))
tr = cands("train", pl.col("fold").is_in([0, 8, 9]))
wr = pl.read_parquet("work/word_addr_train.parquet").select("w", "true_rate", "n")
tr = tr.join(wr, on="w", how="left").with_columns(pl.when(pl.col("true_rate").is_null()).then(pl.lit("rare")).when(pl.col("true_rate") < 0.2).then(pl.lit("SIB"))
                                                  .when(pl.col("true_rate") > 0.8).then(pl.lit("NOISE")).otherwise(pl.lit("mixed")).alias("cls"))
tr = fingerprint(tr, "train")
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
print("TRAIN: share of candidates whose raw address equals another strong same-source candidate's (non-empty addresses)")
print(tr.filter(~pl.col("ra_empty")).group_by("kind", "cls", "y").agg(pl.len(), pl.col("fp_share").mean().round(3).alias("addr_shared"),
                                   (pl.col("off") == "same").mean().round(3).alias("same_num")).sort("kind", "cls", "y"))
print("TRAIN same-number only:")
print(tr.filter(~pl.col("ra_empty") & (pl.col("off") == "same")).group_by("cls", "y").agg(pl.len(), pl.col("fp_share").mean().round(3).alias("addr_shared")).sort("cls", "y"))
fr = cands("test", pl.col("country") == "France")
ASSOC = {"club","ecole","amicale","comite","sportive","amis","parents","college","union","fetes","primaire","anciens","maison","sante","pharmacie",
         "loisirs","sport","lycee","culture","jeunes","culturelle","foyer","section","groupement","maternelle","elementaire","collectif","danse",
         "patrimoine","musique","institut","residence","conseil","theatre","atelier","soins","ehpad","medico","sportif","agricole","auto","ateliers",
         "cafe","centre","societe","federation","association","culturel","gestion"}
FRN = {"services", "service", "cie", "compagnie", "fils", "associes", "frs", "groupe", "france", "developpement"}
FRS = {"holding", "participations", "international", "internationale", "distribution", "snc"}
fr = fr.with_columns(pl.when(pl.col("w").is_in(list(FRN))).then(pl.lit("FR_NOISE?")).when(pl.col("w").is_in(list(FRS))).then(pl.lit("FR_SIB?"))
                     .when(pl.col("w").is_in(list(ASSOC)) & pl.col("mw").is_in(list(ASSOC))).then(pl.lit("ASSOC_SWAP"))
                     .when(pl.col("w").is_in(list(ASSOC))).then(pl.lit("ASSOC_other")).otherwise(pl.lit("other")).alias("cls"))
fr = fingerprint(fr, "test")
print("FRANCE: same statistic by word class")
print(fr.filter(~pl.col("ra_empty")).group_by("kind", "cls").agg(pl.len(), pl.col("fp_share").mean().round(3).alias("addr_shared"),
                                   (pl.col("off") == "same").mean().round(3).alias("same_num"), (pl.col("p2x2") > 0.5).mean().round(2).alias("acc")).sort("kind", "cls"))
print("FRANCE same-number only:")
print(fr.filter(~pl.col("ra_empty") & (pl.col("off") == "same")).group_by("cls").agg(pl.len(), pl.col("fp_share").mean().round(3).alias("addr_shared")).sort("cls"))
fr.select("s1", "m", "kind", "w", "mw", "cls", "off", "p2x2", "fp_share").write_parquet("work/fr_fingerprint.parquet")
