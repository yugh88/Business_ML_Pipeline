"""Label-free relation taxonomy: every pool record vs its best-scoring S1 (V5 p2x2), identical on train and test.
Train rows are split by truth (true copy of that S1 / copy of another S1 / distractor)."""
import sys; sys.path.insert(0, "aws/scripts"); import memguard_local
import polars as pl
from rapidfuzz.distance import Levenshtein

rate = pl.read_parquet("work/extra_token_match_rate.parquet")          # one-extra-word train match rates (n>=300)
NOISE = set(rate.filter(pl.col("match_rate") >= 0.9)["t"].to_list())
desc = pl.read_parquet("work/descriptor_words_train.parquet")["t"].to_list()
FR = ["groupe","holding","participations","france","developpement","distribution","fils","freres","associes","internationale","conseil","gestion","investissements","immobilier"]
DESC = set(desc) | set(FR)

def best_pairs(split, country):
    d = pl.scan_parquet(f"work/v5_p2/{split}/*.parquet").filter((pl.col("p1") >= 1e-4) & (pl.col("country") == country)).select("s1", "m", "y", "p2x2")
    return d.sort("p2x2", descending=True).group_by("m", maintain_order=True).first().collect()

def load_rows(split, ids, srcs):
    q = pl.concat([pl.scan_parquet(f"work/{split}_s{s}_norm.parquet").select("entity_id", "ncore", "a", "nums") for s in srcs])
    return q.join(ids.lazy(), on="entity_id", how="semi").collect()

def name_rel(c1, c2):
    t1, t2 = set(c1.split()), set(c2.split())
    E, M = t2 - t1, t1 - t2
    if not E and not M: return "N0_identical"
    if not t1 & t2: return "N7_no_overlap"
    if E and not M:
        if E & DESC: return "N2_extra_DESCRIPTOR"
        if E <= NOISE: return "N1_extra_noise"
        return "N3_extra_other"
    if M and not E: return "N4_dropped_words"
    # substitution: typo if every extra word is within edit distance 2 of some missing word
    if all(min(Levenshtein.distance(e, m) for m in M) <= 2 for e in E): return "N5_typo_subst"
    return "N6_word_swap_DESC" if E & DESC else "N6_word_swap"

def addr_rel(a1, a2, n1, n2):
    if not a2: return "A3_empty"
    if a1 == a2: return "A0_identical"
    f1, f2 = (n1.split() or [""])[0], (n2.split() or [""])[0]
    t1, t2 = set(a1.replace(",", " ").split()), set(a2.replace(",", " ").split())
    jac = len(t1 & t2) / max(1, len(t1 | t2))
    if f1 and f2 and f1 != f2:
        kind = "trunc" if (f1.startswith(f2) or f2.startswith(f1) or f1.endswith(f2) or f2.endswith(f1)) else ("1digit" if len(f1) == len(f2) and sum(a != b for a, b in zip(f1, f2)) == 1 else "other")
        return f"A2_num_{kind}" + ("_sameStreet" if jac >= 0.4 else "_diffStreet")
    return "A1_same_num" if jac >= 0.4 else "A4_different"

def relate(split):
    parts = []
    countries = ["US", "India"] + (["France"] if split == "test" else [])
    for c in countries:
        b = best_pairs(split, c)
        s1 = load_rows(split, b.select(pl.col("s1").alias("entity_id")).unique(), (1,))
        po = load_rows(split, b.select(pl.col("m").alias("entity_id")).unique(), (2, 3))
        x = b.join(s1.rename({"entity_id": "s1", "ncore": "ncore_1", "a": "a_1", "nums": "nums_1"}), on="s1") \
             .join(po.rename({"entity_id": "m", "ncore": "ncore_2", "a": "a_2", "nums": "nums_2"}), on="m")
        del s1, po, b
        x = x.with_columns(
            pl.Series("nrel", [name_rel(a, b_) for a, b_ in zip(x["ncore_1"].to_list(), x["ncore_2"].to_list())]),
            pl.Series("arel", [addr_rel(a, b_, c_, d) for a, b_, c_, d in zip(x["a_1"].to_list(), x["a_2"].to_list(), x["nums_1"].to_list(), x["nums_2"].to_list())]),
            pl.lit(c).alias("country_1"))
        parts.append(x.select("s1", "m", "y", "p2x2", "country_1", "nrel", "arel")); del x
        print(f"  related {split} {c}", flush=True)
    return pl.concat(parts)

tr = relate("train")
pairs = pl.read_parquet("work/train_pairs.parquet", columns=["m"]).with_columns(pl.lit(True).alias("is_copy_somewhere"))
tr = tr.join(pairs, on="m", how="left").with_columns(
    pl.when(pl.col("y")).then(pl.lit("true_copy")).when(pl.col("is_copy_somewhere").fill_null(False)).then(pl.lit("copy_of_other_S1")).otherwise(pl.lit("distractor")).alias("truth"))
tr.write_parquet("work/rel_train.parquet")
te = relate("test"); te.write_parquet("work/rel_test.parquet")

def table(d, by, label):
    t = d.group_by(by).len().with_columns((pl.col("len") / pl.col("len").sum()).round(4).alias("share")).sort("len", descending=True)
    print(f"\n{label} (n={d.height})"); print(t.head(20).rows())

print("=== NAME relation to best S1 ===")
for k in ["true_copy", "distractor", "copy_of_other_S1"]:
    table(tr.filter(pl.col("truth") == k), "nrel", f"TRAIN {k}")
table(te, "nrel", "TEST all pool records with a candidate")
print("\n=== match rate by NAME relation (train) ===")
print(tr.group_by("nrel").agg(pl.len(), (pl.col("truth") == "true_copy").mean().round(4).alias("true_copy_rate")).sort("len", descending=True).rows())
print("\n=== ADDRESS relation (train truth rate, train share, test share) ===")
a_tr = tr.group_by("arel").agg(pl.len().alias("n_tr"), (pl.col("truth") == "true_copy").mean().round(4).alias("true_rate"))
a_te = te.group_by("arel").len().rename({"len": "n_te"})
j = a_tr.join(a_te, on="arel", how="full", coalesce=True).fill_null(0).with_columns((pl.col("n_tr") / pl.col("n_tr").sum()).round(4).alias("share_tr"), (pl.col("n_te") / pl.col("n_te").sum()).round(4).alias("share_te")).sort("n_tr", descending=True)
print(j.rows())
print("\n=== JOINT name x address: train true rate, per-S1 density train vs test (records per S1) ===")
S1_TR = pl.read_parquet("work/train_s1.parquet", columns=["entity_id"]).height - pl.read_parquet("work/v5_dropped_s1.parquet").height
S1_TE = pl.read_parquet("work/test_s1.parquet", columns=["entity_id"]).height
jt = tr.group_by("nrel", "arel").agg(pl.len().alias("n_tr"), (pl.col("truth") == "true_copy").mean().round(3).alias("true_rate"))
je = te.group_by("nrel", "arel").len().rename({"len": "n_te"})
jj = jt.join(je, on=["nrel", "arel"], how="full", coalesce=True).fill_null(0).with_columns(
    (pl.col("n_tr") / S1_TR).round(4).alias("per_S1_tr"), (pl.col("n_te") / S1_TE).round(4).alias("per_S1_te")).with_columns(
    (pl.col("per_S1_te") / (pl.col("per_S1_tr") + 1e-6)).round(2).alias("test_over_train")).sort("n_te", descending=True)
jj.write_parquet("work/rel_joint.parquet")
print(jj.head(40).rows())
