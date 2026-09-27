"""Teammate idea — ADAPTIVE normalization: derive stop words / word groups from each dataset's own token statistics.
Decisive tests (uses 195 outputs: V9 candidates + acceptance, holdout with truth):
 (1) adaptive stop list = tokens with document frequency >= 0.3% of that country's records (computed on the dataset itself);
     which of them does our normalizer NOT already handle (legal/alias/generic/word classes)?
 (2) pairs at the same address whose name difference consists ONLY of adaptive stop words ('stop-collapse') and pairs whose
     difference is one frequent word-for-word swap ('group'): holdout truth vs V9, test density + V9 acceptance;
 (3) ablation: accept them all (what an adaptive normalizer effectively does) -> holdout macro F0.5 per fold vs V9."""
import sys, json
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl, numpy as np, duckdb
from normlib import LEGAL, ALIAS, GENERIC
W = json.load(open("aws/scripts/word_classes.json")); KNOWN = LEGAL | ALIAS | GENERIC | set(W["desc"]) | set(W["noise"])
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(220); pl.Config.set_tbl_hide_dataframe_shape(True)
TH = 0.003


def df_tokens(split, country):
    q = pl.concat([pl.scan_parquet(f"work/{split}_s{s}_norm.parquet").filter(pl.col("country") == country).select("ncore") for s in (1, 2, 3)])
    n = q.select(pl.len()).collect().item()
    t = q.select(pl.col("ncore").str.split(" ").list.unique().alias("t")).explode("t").group_by("t").len().collect(engine="streaming")
    return t.with_columns((pl.col("len") / n).alias("df")).filter(pl.col("df") >= TH / 10)


def with_diff(x, split):
    ids = pl.concat([x.select(pl.col("s1").alias("entity_id")), x.select(pl.col("m").alias("entity_id"))]).unique()
    nm = pl.concat([pl.scan_parquet(f"work/{split}_s{s}_norm.parquet").select("entity_id", "ncore") for s in (1, 2, 3)]).join(ids.lazy(), on="entity_id", how="semi").collect()
    x = x.join(nm.rename({"entity_id": "s1", "ncore": "c1"}), on="s1").join(nm.rename({"entity_id": "m", "ncore": "c2"}), on="m")
    a, b = pl.col("c1").str.split(" "), pl.col("c2").str.split(" ")
    return x.with_columns(b.list.set_difference(a).alias("E"), a.list.set_difference(b).alias("M")).drop("c1", "c2")


def flag(x, stop, swaps):
    same = pl.col("cls").str.ends_with("@same") & (pl.col("cls") != "identical@same")
    D = pl.concat_list("E", "M")
    x = x.with_columns((same & (D.list.len() > 0) & D.list.eval(pl.element().is_in(list(stop))).list.all()).alias("stop_collapse"),
                       pl.when((pl.col("E").list.len() == 1) & (pl.col("M").list.len() == 1))
                         .then(pl.concat_str(pl.col("M").list.first(), pl.lit("->"), pl.col("E").list.first())).otherwise(None).alias("swap"))
    return x.with_columns((same & pl.col("swap").is_in(list(swaps))).alias("group_swap"))


def f05(tp, fp, n):
    P = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-9), 0); R = np.where(n > 0, tp / np.maximum(n, 1), 0)
    return np.where(n == 0, (tp + fp == 0).astype(float), np.where(tp > 0, 1.25 * P * R / np.maximum(0.25 * P + R, 1e-12), 0.0))


gt = duckdb.connect().execute("select source1_entity_id s1, case when matched_entity_ids is null or matched_entity_ids='' then 0 else len(string_split(matched_entity_ids, ',')) end n, "
                              "hash(source1_entity_id || 'fold') % 10 fold from 'work/train_gt.parquet'").pl().join(pl.read_parquet("work/v5_dropped_s1.parquet").select("s1"), on="s1", how="anti")
t1 = pl.read_parquet("work/train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1", "country": "country"})
h = pl.read_parquet("work/copy_channel_holdout.parquet", columns=["s1", "m", "y", "p3", "acc", "cls"]).join(t1, on="s1")
h = with_diff(h, "train")
fold = gt.select("s1", "fold")
res = {}
for c in ("US", "India", "France"):
    dte = df_tokens("test", c)
    stop = set(dte.filter(pl.col("df") >= TH)["t"].to_list())
    print(f"\n######## {c}: adaptive stop list (test df >= {TH:.1%}) = {len(stop)} tokens; NOT handled by our normalizer/word classes:")
    un = dte.filter((pl.col("df") >= TH) & ~pl.col("t").is_in(list(KNOWN))).sort("df", descending=True)
    if c != "France":
        dtr = df_tokens("train", c).select("t", pl.col("df").alias("df_train"))
        un = un.join(dtr, on="t", how="left").with_columns((pl.col("df") / pl.col("df_train")).round(2).alias("test/train"))
    print("   ", [(r[0], round(100 * r[2], 2)) + tuple(r[3:]) for r in un.head(45).rows()])
    x = pl.read_parquet(f"work/copy_channel_test_{c}.parquet", columns=["s1", "m", "p3", "acc", "cls"])
    ns = x["s1"].n_unique() / 1000
    x = with_diff(x, "test")
    sw = x.filter(pl.col("cls").str.ends_with("@same") & (pl.col("E").list.len() == 1) & (pl.col("M").list.len() == 1)) \
          .with_columns(pl.concat_str(pl.col("M").list.first(), pl.lit("->"), pl.col("E").list.first()).alias("swap")).group_by("swap").len()
    swaps = set(sw.filter(pl.col("len") >= 50)["swap"].to_list())
    x = flag(x, stop, swaps)
    print(f"  TEST {c}: stop-collapse pairs {x['stop_collapse'].sum()/ns:.1f}/1k S1 (V9 accepts {x.filter('stop_collapse')['acc'].mean() or 0:.3f});"
          f" frequent-swap pairs {x['group_swap'].sum()/ns:.1f}/1k (V9 accepts {x.filter('group_swap')['acc'].mean() or 0:.3f});"
          f" would newly accept {(x['stop_collapse'] & ~x['acc']).sum()/ns:.1f} + {(x['group_swap'] & ~x['acc'] & ~x['stop_collapse']).sum()/ns:.1f} per 1k")
    print("   top rejected stop-collapse differences:", x.filter(pl.col("stop_collapse") & ~pl.col("acc")).with_columns(pl.concat_list("E", "M").list.sort().list.join(" ").alias("d"))
          .group_by("d").len().sort("len", descending=True).head(15).rows())
    print("   top rejected frequent swaps:", x.filter(pl.col("group_swap") & ~pl.col("acc")).group_by("swap").len().sort("len", descending=True).head(15).rows())
    if c == "France":
        continue
    hc = flag(h.filter(pl.col("country") == c), stop, swaps)
    for k in ("stop_collapse", "group_swap"):
        z = hc.filter(k)
        print(f"  HOLDOUT {c} {k}: n={z.height} true_rate {z['y'].mean() or 0:.3f} V9_acc {z['acc'].mean() or 0:.3f} V9_FP {(z['acc'] & ~z['y']).sum()} V9_FN {(~z['acc'] & z['y']).sum()}"
              f" | accept-all adds TP {(~z['acc'] & z['y']).sum()} FP {(~z['acc'] & ~z['y']).sum()}")
    res[c] = hc.select("s1", "m", "y", "acc", "stop_collapse", "group_swap")
hh = pl.concat(res.values())
for f in (0, 8, 9):
    s1f = gt.filter(pl.col("fold") == f)
    z = hh.join(fold.filter(pl.col("fold") == f), on="s1", how="semi")
    def score(keep):
        a = z.filter(keep).group_by("s1").agg(pl.col("y").sum().alias("tp"), (~pl.col("y")).sum().alias("fp"))
        g = s1f.join(a, on="s1", how="left").fill_null(0)
        return float(f05(g["tp"].to_numpy().astype(float), g["fp"].to_numpy().astype(float), g["n"].to_numpy().astype(float)).mean())
    print(f"ABLATION fold {f}: V9 {score(pl.col('acc')):.5f} | +accept stop-collapse {score(pl.col('acc') | pl.col('stop_collapse')):.5f}"
          f" | +accept frequent swaps {score(pl.col('acc') | pl.col('group_swap')):.5f} | both {score(pl.col('acc') | pl.col('stop_collapse') | pl.col('group_swap')):.5f}")
