"""Cross-encoder PILOT (teammate's key ingredient) on the Apple GPU (MPS): fine-tune a multilingual MiniLM cross-encoder on
"S1 name ; S1 address" vs "pool name ; pool address" pairs from TRAIN folds 1-6 only (hard + easy + V8 synthetic look-alikes),
then test on the labelled holdout uncertain band (work/embed_band.parquet, folds 0/8/9): does its score add information beyond p3?"""
import sys, time, math, random
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
torch.manual_seed(0); random.seed(0)
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
N_HARD, N_EASY, MAXLEN, BS, LR = int(sys.argv[1]) if len(sys.argv) > 1 else 120000, 80000, 64, 32, 4e-5
dev = torch.device("mps")
d = pl.scan_parquet("work/v8_p2/train/*.parquet").filter(pl.col("fold").is_in([1, 2, 3, 4, 5, 6]) & (pl.col("p1") >= 0.05)).select("s1", "m", "y", "p2x2").collect()
hard = d.filter(pl.col("p2x2").is_between(0.02, 0.98)).sample(n=min(N_HARD, d.filter(pl.col("p2x2").is_between(0.02, 0.98)).height), seed=1)
easy = d.filter(~pl.col("p2x2").is_between(0.02, 0.98)).sample(n=N_EASY, seed=2)
tr = pl.concat([hard, easy]).sample(fraction=1.0, shuffle=True, seed=3); nh_, ne_ = hard.height, easy.height; del d, hard, easy; import gc; gc.collect()
print(f"train pairs {tr.height} (hard {nh_}, easy {ne_}), positive rate {tr['y'].mean():.3f}", flush=True)
ev = pl.read_parquet("work/embed_band.parquet", columns=["s1", "m", "y", "p3", "cls", "fold"])
def texts(pairs, split, aug):
    ids = pl.concat([pairs.select(pl.col("s1").alias("entity_id")), pairs.select(pl.col("m").alias("entity_id"))]).unique()
    srcs = [pl.scan_parquet(f"work/{split}_s{s}.parquet") for s in (1, 2, 3)] + ([pl.scan_parquet(f"work/aug_v8_{s}.parquet") for s in ("s2", "s3")] if aug else [])
    raw = pl.concat([s.select("entity_id", pl.col("business_name").fill_null(""), pl.col("business_address").fill_null("")) for s in srcs]).join(ids.lazy(), on="entity_id", how="semi").collect()
    R = {e: f"{n} ; {a}" for e, n, a in raw.iter_rows()}
    return [R.get(a, "") for a in pairs["s1"].to_list()], [R.get(b, "") for b in pairs["m"].to_list()]
ta, tb = texts(tr, "train", True); ea, eb = texts(ev, "train", False)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1).to(dev)
for p_ in model.base_model.embeddings.parameters(): p_.requires_grad = False          # 96M-param multilingual embedding table frozen (memory)
opt = torch.optim.AdamW([p_ for p_ in model.parameters() if p_.requires_grad], lr=LR, weight_decay=0.01)
steps = math.ceil(tr.height / BS); sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
y = torch.tensor(tr["y"].to_numpy().astype(np.float32))
lossf = torch.nn.BCEWithLogitsLoss(); model.train(); t0 = time.time()
for i in range(steps):
    sl = slice(i * BS, (i + 1) * BS)
    enc = tok(ta[sl], tb[sl], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to(dev)
    out = model(**enc).logits.squeeze(-1)
    loss = lossf(out, y[sl].to(dev)); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad()
    if i % 300 == 0:
        print(f"  step {i}/{steps} loss {loss.item():.4f} {(i + 1) * BS / (time.time() - t0):.0f} pairs/s", flush=True)
model.eval(); scores = []
with torch.no_grad():
    for i in range(0, ev.height, 128):
        enc = tok(ea[i:i + 128], eb[i:i + 128], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to(dev)
        scores.append(model(**enc).logits.squeeze(-1).float().cpu().numpy())
ev = ev.with_columns(pl.Series("ce", np.concatenate(scores)))
ev.select("s1", "m", "ce").write_parquet("work/ce_pilot_band.parquet"); model.save_pretrained("work/ce_pilot_model"); tok.save_pretrained("work/ce_pilot_model")
print(f"train {time.time() - t0:.0f}s total; eval done", flush=True)
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
cls = sorted(ev["cls"].unique().to_list())
def X(z, with_ce):
    b = [lg(z["p3"].to_numpy())] + [(z["cls"] == c).to_numpy().astype(float) for c in cls]
    return np.column_stack(b + ([z["ce"].to_numpy()] + [(z["cls"] == c).to_numpy() * z["ce"].to_numpy() for c in cls] if with_ce else []))
a, b = ev.filter(pl.col("fold") == 0), ev.filter(pl.col("fold").is_in([8, 9]))
print(f"band AUC: p3 {roc_auc_score(b['y'], b['p3']):.4f}  cross-encoder alone {roc_auc_score(b['y'], b['ce']):.4f}")
res = {}
for w in (False, True):
    m = LogisticRegression(max_iter=3000).fit(X(a, w), a["y"].to_numpy()); p = m.predict_proba(X(b, w))[:, 1]; res[w] = p
    print(f"  stack with_ce={w}: log-loss {log_loss(b['y'], p):.5f}  AUC {roc_auc_score(b['y'], p):.4f}  errors@0.5 {int(((p >= 0.5) != b['y'].to_numpy()).sum())}")
yy = b["y"].to_numpy(); a0, a1 = res[False] >= 0.5, res[True] >= 0.5
print(f"decision flips: {int((a0 != a1).sum())}; fixed {int(((a1 == yy) & (a0 != yy)).sum())}, broken {int(((a0 == yy) & (a1 != yy)).sum())} (folds 8/9 band, n={b.height})")
