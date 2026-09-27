"""CE v2 trainer + scorer on MPS, memory-lean and RESUMABLE (memguard-safe): texts from 243a; multilingual MiniLM-L12 cross-encoder,
frozen embeddings, max 96 tokens (padded to multiples of 16), batch 16, 2 epochs, deterministic per-epoch length-bucketed batch order,
checkpoint (model/optimizer/scheduler/position) every 3000 steps -> work/ce_v2_ckpt.pt; rerun the same command to resume.
Then scores work/ce2_score_texts.parquet -> work/ce2_band_{train,test}.parquet (s1, m, ce)."""
import sys, time, math, random, gc, os
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl, numpy as np, torch, psutil
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
torch.manual_seed(0)
MODEL, MAXLEN, BS, LR, EPOCHS, CKPT, OUT = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 96, 16, 3e-5, 2, "work/ce_v2_ckpt.pt", "work/ce_v2_model"
dev = torch.device("mps")
t = pl.read_parquet("work/ce2_train_texts.parquet"); ta, tb, y = t["a"].to_list(), t["b"].to_list(), t["y"].to_numpy().astype(np.float32); del t; gc.collect()
clen = np.array([len(a) + len(b) for a, b in zip(ta, tb)], dtype=np.int32)


def batches(ep):
    rng, r2 = np.random.default_rng(100 + ep), random.Random(200 + ep)
    idx = rng.permutation(len(ta)); out = []
    for i in range(0, len(idx), BS * 100):
        mb = idx[i:i + BS * 100]; mb = mb[np.argsort(clen[mb], kind="stable")]
        out += [mb[j:j + BS] for j in range(0, len(mb), BS)]
    r2.shuffle(out); return out


tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1).to(dev)
for p_ in model.base_model.embeddings.parameters(): p_.requires_grad = False
opt = torch.optim.AdamW([p_ for p_ in model.parameters() if p_.requires_grad], lr=LR, weight_decay=0.01)
nb = math.ceil(len(ta) / BS); steps = nb * EPOCHS; sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
ep0, i0, step, run = 0, 0, 0, 0.0
if os.path.exists(CKPT):
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sch.load_state_dict(ck["sch"])
    ep0, i0, step, run = ck["ep"], ck["i"], ck["step"], ck["run"]; del ck; gc.collect()
    print(f"RESUMED at epoch {ep0} batch {i0} step {step}", flush=True)
lossf = torch.nn.BCEWithLogitsLoss(); model.train(); t0 = time.time(); s0 = step
for ep in range(ep0, EPOCHS):
    B = batches(ep)
    for i in range(i0 if ep == ep0 else 0, len(B)):
        b = B[i]
        enc = tok([ta[j] for j in b], [tb[j] for j in b], truncation=True, max_length=MAXLEN, padding=True, pad_to_multiple_of=16, return_tensors="pt").to(dev)
        loss = lossf(model(**enc).logits.squeeze(-1), torch.from_numpy(y[b]).to(dev)); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad(set_to_none=True); step += 1
        run = 0.98 * run + 0.02 * loss.item()
        if step % 50 == 0: torch.mps.empty_cache()
        if step % 1000 == 0:
            print(f"  epoch {ep} step {step}/{steps} loss(ema) {run:.4f} {(step - s0) * BS / (time.time() - t0):.0f} pairs/s "
                  f"mps {torch.mps.driver_allocated_memory() / 1e9:.2f} GB, sys avail {psutil.virtual_memory().available / 1e9:.1f} GB", flush=True)
        if step % 3000 == 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sch": sch.state_dict(), "ep": ep, "i": i + 1, "step": step, "run": run}, CKPT + ".tmp")
            os.replace(CKPT + ".tmp", CKPT)
    i0 = 0
    model.save_pretrained(OUT); tok.save_pretrained(OUT); print(f"epoch {ep} done, step {step}, {time.time() - t0:.0f}s", flush=True)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sch": sch.state_dict(), "ep": ep + 1, "i": 0, "step": step, "run": run}, CKPT + ".tmp"); os.replace(CKPT + ".tmp", CKPT)
del ta, tb, y, opt; gc.collect(); torch.mps.empty_cache()

# ---------------- scoring
model.eval()
S = pl.read_parquet("work/ce2_score_texts.parquet")
for part_set, outp in ((["train"], "work/ce2_band_train.parquet"), (["US", "India", "France"], "work/ce2_band_test.parquet")):
    z = S.filter(pl.col("part").is_in(part_set)); a, b = z["a"].to_list(), z["b"].to_list(); t1 = time.time()
    order = np.argsort(np.array([len(u) + len(v) for u, v in zip(a, b)]), kind="stable"); out = []
    with torch.no_grad():
        for k in range(0, len(order), 128):
            o = order[k:k + 128]
            enc = tok([a[j] for j in o], [b[j] for j in o], truncation=True, max_length=MAXLEN, padding=True, pad_to_multiple_of=16, return_tensors="pt").to(dev)
            out.append(model(**enc).logits.squeeze(-1).float().cpu().numpy())
            if (k // 128) % 100 == 0: torch.mps.empty_cache()
    sc = np.empty(len(order), np.float32); sc[order] = np.concatenate(out)
    z.select("s1", "m").with_columns(pl.Series("ce", sc)).write_parquet(outp)
    print(f"scored {part_set}: {len(order)} pairs in {time.time() - t1:.0f}s -> {outp}", flush=True)
print("DONE", flush=True)
