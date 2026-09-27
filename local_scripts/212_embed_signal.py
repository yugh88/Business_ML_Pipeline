"""External-model test: does a pretrained multilingual sentence embedding (name vs name, address vs address) add information
beyond V9's p3 where decisions are made? Holdout uncertain band (0.02 <= p3 <= 0.98) + a sample of confident pairs.
Logistic calibration fit on fold 0, evaluated on folds 8/9: log-loss, errors at 0.5 and flips (fixed vs broken truth)."""
import sys
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl, numpy as np, duckdb, torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
torch.set_num_threads(8)
MODEL = sys.argv[1] if len(sys.argv) > 1 else "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
h = pl.read_parquet("work/copy_channel_holdout.parquet", columns=["s1", "m", "y", "p3", "acc", "cls"])
fold = duckdb.connect().execute("select source1_entity_id s1, hash(source1_entity_id || 'fold') % 10 fold from 'work/train_gt.parquet'").pl()
h = h.join(fold, on="s1")
band = h.filter(pl.col("p3").is_between(0.02, 0.98))
print(f"holdout band pairs {band.height} (true rate {band['y'].mean():.3f}); by fold", dict(band.group_by("fold").len().rows()), flush=True)
ids = pl.concat([band.select(pl.col("s1").alias("entity_id")), band.select(pl.col("m").alias("entity_id"))]).unique()
raw = pl.concat([pl.scan_parquet(f"work/train_s{s}.parquet").select("entity_id", pl.col("business_name").fill_null(""), pl.col("business_address").fill_null("")) for s in (1, 2, 3)]) \
        .join(ids.lazy(), on="entity_id", how="semi").collect()
model = SentenceTransformer(MODEL, device="cpu")
def emb(texts):
    u = list(dict.fromkeys(texts)); E = model.encode(u, batch_size=256, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    return {t: e for t, e in zip(u, E)}
N = dict(zip(raw["entity_id"], raw["business_name"])); A = dict(zip(raw["entity_id"], raw["business_address"]))
EN = emb(list(N.values())); print("names embedded", len(EN), flush=True)
EA = emb(list(A.values())); print("addresses embedded", len(EA), flush=True)
s1l, ml = band["s1"].to_list(), band["m"].to_list()
band = band.with_columns(pl.Series("sim_n", [float(EN[N[a]] @ EN[N[b]]) for a, b in zip(s1l, ml)]),
                         pl.Series("sim_a", [float(EA[A[a]] @ EA[A[b]]) if A[b] else 0.0 for a, b in zip(s1l, ml)]))
band.select("s1", "m", "y", "p3", "cls", "fold", "sim_n", "sim_a").write_parquet("work/embed_band.parquet")
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
cls = sorted(band["cls"].unique().to_list())
def X(d, with_emb):
    base = [lg(d["p3"].to_numpy())] + [(d["cls"] == c).to_numpy().astype(float) for c in cls]
    if with_emb:
        base += [d["sim_n"].to_numpy(), d["sim_a"].to_numpy()] + [(d["cls"] == c).to_numpy() * d["sim_n"].to_numpy() for c in cls]
    return np.column_stack(base)
tr, te = band.filter(pl.col("fold") == 0), band.filter(pl.col("fold").is_in([8, 9]))
for c in cls:
    z = te.filter(pl.col("cls") == c)
    if z.height > 500 and 0 < z["y"].mean() < 1:
        print(f"  {c:26s} n={z.height:6d} true {z['y'].mean():.3f}  AUC p3 {roc_auc_score(z['y'], z['p3']):.3f}  AUC sim_name {roc_auc_score(z['y'], z['sim_n']):.3f}  AUC sim_addr {roc_auc_score(z['y'], z['sim_a']):.3f}")
res = {}
for we in (False, True):
    m = LogisticRegression(max_iter=2000, C=1.0).fit(X(tr, we), tr["y"].to_numpy())
    p = m.predict_proba(X(te, we))[:, 1]; res[we] = p
    print(f"with_embedding={we}: test log-loss {log_loss(te['y'], p):.5f}  AUC {roc_auc_score(te['y'], p):.4f}  errors@0.5 {int(((p >= 0.5) != te['y'].to_numpy()).sum())}", flush=True)
y = te["y"].to_numpy(); a0, a1 = res[False] >= 0.5, res[True] >= 0.5
print(f"decision flips with embedding: {int((a0 != a1).sum())}; fixed {int(((a1 == y) & (a0 != y)).sum())}, broken {int(((a0 == y) & (a1 != y)).sum())} (folds 8/9 band)")
