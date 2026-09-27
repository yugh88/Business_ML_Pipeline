"""Design input for the V10 augmentation: per country x relation class x house-number offset type, candidate density per 1k S1
on TEST vs TRAIN-AS-TRAINED (holdout folds 0/8/9 INCLUDING the V8 synthetic look-alikes, i.e. what the model saw) and V9 acceptance.
Excess test candidates over train = look-alike density the augmentation must add (label-free; test copy density = train)."""
import sys
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl, json
exec(open("scripts/195_copy_channel.py").read().split("def cands")[0].split("d9 = json.load")[0])       # imports + name_rel/addr_rel
exec("def klass" + open("scripts/195_copy_channel.py").read().split("def klass")[1].split("def annotate")[0])
from tune_decision import prep, decide
d9 = json.load(open("output_v9/decision_s3.json"))
pl.Config.set_tbl_rows(120); pl.Config.set_tbl_width_chars(230); pl.Config.set_tbl_hide_dataframe_shape(True)


def offset(h1, h2):
    if not h1 or not h2: return "no_num"
    if not (h1.isdigit() and h2.isdigit()): return "nonnum"
    d = int(h2[:9]) - int(h1[:9])
    if d == 0: return "same"
    if h1.startswith(h2) or h1.endswith(h2) or h2.startswith(h1) or h2.endswith(h1): return "trunc"
    if 1 <= d <= 9: return "+1..9"
    if -9 <= d <= -1: return "-1..9"
    if 10 <= d <= 99: return "+10..99"
    if -99 <= d <= -10: return "-10..99"
    return "far"


def annotate(x, split, extra_norm=None):
    ids = pl.concat([x.select(pl.col("s1").alias("entity_id")), x.select(pl.col("m").alias("entity_id"))]).unique()
    nm = pl.concat([pl.scan_parquet(f"work/{split}_s{s}_norm.parquet").select("entity_id", "ncore", "a", "nums") for s in (1, 2, 3)] +
                   ([pl.scan_parquet(extra_norm).select("entity_id", "ncore", "a", "nums")] if extra_norm else [])).join(ids.lazy(), on="entity_id", how="semi").collect()
    N = {r[0]: r[1:] for r in nm.iter_rows()}
    cl, of = [], []
    for a, b in zip(x["s1"].to_list(), x["m"].to_list()):
        na, nb = N.get(a, ("", "", "")), N.get(b, ("", "", ""))
        cl.append(klass(name_rel(na[0], nb[0]), addr_rel(na[1], nb[1], na[2], nb[2]), na[0], nb[0]))
        of.append(offset((na[2].split() or [""])[0], (nb[2].split() or [""])[0]))
    return x.with_columns(pl.Series("cls", cl), pl.Series("off", of))


t1 = pl.read_parquet("work/train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
SYN = pl.concat([pl.read_parquet(f"work/aug_v8_{s}.parquet", columns=["entity_id"]) for s in ("s2", "s3")]).rename({"entity_id": "m"}).with_columns(pl.lit(True).alias("syn"))
h = pl.concat([pl.read_parquet(f"output_v9/p3/train/fold{f}.parquet", columns=["s1", "m", "y", "p1", "p3"]) for f in (0, 8, 9)]).filter(pl.col("p1") >= d9["tau"])
h = h.join(SYN, on="m", how="left").with_columns(pl.col("syn").fill_null(False)).join(t1, on="s1")
h = annotate(h, "train", "work/aug_v8_norm.parquet")
nsh = dict(h.group_by("country").agg(pl.col("s1").n_unique()).rows())
H = h.group_by("country", "cls", "off").agg((pl.len()).alias("n"), pl.col("y").sum().alias("T"), pl.col("syn").sum().alias("S"))
H = H.with_columns(*[(pl.col(c) / pl.col("country").replace_strict(nsh) * 1000).alias(c + "_h") for c in ("n", "T", "S")]).drop("n", "T", "S")
rows = []
for c in ("US", "India"):
    x = annotate(pl.read_parquet(f"output_v9/p3/test/{c}.parquet", columns=["s1", "m", "p1", "p3"]).filter(pl.col("p1") >= d9["tau"]), "test")
    acc = decide(prep(x.select("s1", "m", "p1", "p3"), d9["tau"], "p3"), d9).select("s1", "m").with_columns(pl.lit(True).alias("acc"))
    x = x.join(acc, on=["s1", "m"], how="left").with_columns(pl.col("acc").fill_null(False))
    ns = x["s1"].n_unique()
    rows.append(x.group_by("cls", "off").agg((pl.len() / ns * 1000).alias("n_t"), (pl.col("acc").sum() / ns * 1000).alias("acc_t")).with_columns(pl.lit(c).alias("country")))
T = pl.concat(rows).join(H, on=["country", "cls", "off"], how="full", coalesce=True).fill_null(0)
T = T.with_columns((pl.col("n_t") - pl.col("n_h")).alias("excess_cands"), (pl.col("n_t") - pl.col("n_h") + pl.col("S_h")).alias("excess_vs_real"))
out = T.select("country", "cls", "off", *[pl.col(k).round(2) for k in ("n_h", "T_h", "S_h", "n_t", "acc_t", "excess_cands", "excess_vs_real")]) \
       .filter((pl.col("n_t") >= 1) | (pl.col("n_h") >= 1)).sort("country", "excess_cands", descending=[False, True])
print("per 1k S1 — n_h: train-as-trained candidates (incl. V8 synthetic S_h), T_h: true copies, n_t/acc_t: test candidates / V9 accepted;")
print("excess_cands = test - train-as-trained (what V10 must still add); excess_vs_real = test - real train (total look-alike excess)")
print(out.head(70))
out.write_parquet("work/excess_profile.parquet")
