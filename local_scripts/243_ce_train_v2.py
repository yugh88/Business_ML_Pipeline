"""Cross-encoder v2 (scaled pilot, local MPS, no AWS cost): ALL 314k hard pairs (V8 stage-2 p2x2 in 0.02-0.98) + 200k easy pairs from
TRAIN folds 1-6 only (incl. V8 synthetic look-alikes), max 96 tokens (pilot 64 truncated 18% of pairs), 2 epochs, length-bucketed batches,
frozen multilingual embeddings (memory). Checkpoint after each epoch; then scores the uncertain band of train folds 0/7/8/9 and test
(same selection as 238) -> work/ce2_band_{train,test}.parquet. Folds 0/7/8/9 never seen in training."""
import sys, time, math, random, gc, os
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
torch.manual_seed(0); random.seed(0); rng = np.random.default_rng(0)
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
N_EASY, MAXLEN, BS, LR, EPOCHS, OUT = 200000, 96, 32, 4e-5, 2, "work/ce_v2_model"
dev = torch.device("mps")
d = pl.scan_parquet("work/v8_p2/train/*.parquet").filter(pl.col("fold").is_in([1, 2, 3, 4, 5, 6]) & (pl.col("p1") >= 0.05)).select("s1", "m", "y", "p2x2").collect()
hard = d.filter(pl.col("p2x2").is_between(0.02, 0.98)); easy = d.filter(~pl.col("p2x2").is_between(0.02, 0.98)).sample(n=N_EASY, seed=2)
tr = pl.concat([hard, easy]).select("s1", "m", "y"); nh_, ne_ = hard.height, easy.height; del d, hard, easy; gc.collect()
print(f"train pairs {tr.height} (hard {nh_}, easy {ne_}), positive rate {tr['y'].mean():.3f}", flush=True)


def texts(pairs, split, aug):
    ids = pl.concat([pairs.select(pl.col("s1").alias("entity_id")), pairs.select(pl.col("m").alias("entity_id"))]).unique()
    srcs = [pl.scan_parquet(f"work/{split}_s{s}.parquet") for s in (1, 2, 3)] + ([pl.scan_parquet(f"work/aug_v8_{s}.parquet") for s in ("s2", "s3")] if aug else [])
    raw = pl.concat([s.select("entity_id", pl.col("business_name").fill_null(""), pl.col("business_address").fill_null("")) for s in srcs]).join(ids.lazy(), on="entity_id", how="semi").collect()
    R = {e: f"{n} ; {a}" for e, n, a in raw.iter_rows()}
    return [R.get(a, "") for a in pairs["s1"].to_list()], [R.get(b, "") for b in pairs["m"].to_list()]


ta, tb = texts(tr, "train", True); y = tr["y"].to_numpy().astype(np.float32); del tr; gc.collect()
clen = np.array([len(a) + len(b) for a, b in zip(ta, tb)])


def batches():
    """shuffle, then sort by length inside mega-batches of 100 batches (less padding), shuffle batch order."""
    idx = rng.permutation(len(ta)); out = []
    for i in range(0, len(idx), BS * 100):
        mb = idx[i:i + BS * 100]; mb = mb[np.argsort(clen[mb], kind="stable")]
        out += [mb[j:j + BS] for j in range(0, len(mb), BS)]
    random.shuffle(out); return out


tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1).to(dev)
for p_ in model.base_model.embeddings.parameters(): p_.requires_grad = False
opt = torch.optim.AdamW([p_ for p_ in model.parameters() if p_.requires_grad], lr=LR, weight_decay=0.01)
steps = math.ceil(len(ta) / BS) * EPOCHS; sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
lossf = torch.nn.BCEWithLogitsLoss(); model.train(); t0 = time.time(); step = 0
for ep in range(EPOCHS):
    run = 0.0
    for b in batches():
        enc = tok([ta[i] for i in b], [tb[i] for i in b], truncation=True, max_length=MAXLEN, padding=True, pad_to_multiple_of=16, return_tensors="pt").to(dev)
        loss = lossf(model(**enc).logits.squeeze(-1), torch.from_numpy(y[b]).to(dev)); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad(); step += 1
        run = 0.98 * run + 0.02 * loss.item()
        if step % 200 == 0: torch.mps.empty_cache()
        if step % 500 == 0:
            print(f"  epoch {ep} step {step}/{steps} loss(ema) {run:.4f} {step * BS / (time.time() - t0):.0f} pairs/s "
                  f"mps {torch.mps.current_allocated_memory() / 1e9:.2f}/{torch.mps.driver_allocated_memory() / 1e9:.2f} GB", flush=True)
    model.save_pretrained(OUT); tok.save_pretrained(OUT); print(f"epoch {ep} done {time.time() - t0:.0f}s, checkpoint saved", flush=True)
del ta, tb, y, opt; gc.collect(); torch.mps.empty_cache()

# ---------------- score the uncertain band (same selection as 238)
model.eval()
SYN = pl.concat([pl.read_parquet(f"work/aug_v8_{s}.parquet", columns=["entity_id"]) for s in ("s2", "s3")]).rename({"entity_id": "m"})
band = lambda x: x.filter((pl.col("p1") >= 0.05) & pl.col("p3").is_between(0.02, 0.98))


def score(pairs, split, label):
    a, b = texts(pairs, split, False); out = []; t1 = time.time()
    order = np.argsort([len(u) + len(v) for u, v in zip(a, b)], kind="stable")
    with torch.no_grad():
        for i in range(0, len(order), 128):
            o = order[i:i + 128]
            enc = tok([a[j] for j in o], [b[j] for j in o], truncation=True, max_length=MAXLEN, padding=True, pad_to_multiple_of=16, return_tensors="pt").to(dev)
            out.append(model(**enc).logits.squeeze(-1).float().cpu().numpy())
            if (i // 128) % 200 == 0: torch.mps.empty_cache()
    s = np.empty(len(order), np.float32); s[order] = np.concatenate(out)
    print(f"  scored {label}: {len(order)} pairs in {time.time() - t1:.0f}s", flush=True)
    return pairs.select("s1", "m").with_columns(pl.Series("ce", s))


trb = band(pl.concat([pl.read_parquet(f"output_v11s3/p3/train/fold{f}.parquet", columns=["s1", "m", "p1", "p3"]) for f in (0, 7, 8, 9)]).join(SYN, on="m", how="anti"))
score(trb, "train", "train band").write_parquet("work/ce2_band_train.parquet"); gc.collect()
parts = []
for c in ("US", "India", "France"):
    parts.append(score(band(pl.read_parquet(f"output_v11s3/p3/test/{c}.parquet", columns=["s1", "m", "p1", "p3"])), "test", c)); gc.collect()
pl.concat(parts).write_parquet("work/ce2_band_test.parquet")
print(f"DONE {time.time() - t0:.0f}s", flush=True)
