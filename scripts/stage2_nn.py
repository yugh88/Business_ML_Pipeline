"""Stage stage2_nn: two-stream fusion network (char-CNN text encoders + engineered-feature MLP, cross-attention fusion),
trained with the same K-way cross-fitting as stage 2 on the cascade (stage-2) candidate set. Adds columns p2nn and p2nnb
(= mean of p2 and p2nn) to p2/{split}/*.parquet; evaluate/tune_decision then choose among variants on the valid fold only.
Idea source: last year's winning fusion architecture (pretrained LM + engineered features, cross-attention), adapted to
pairwise entity matching: the text streams read the raw normalized name/address pair at character level."""
import os, sys, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ALPH = "abcdefghijklmnopqrstuvwxyz0123456789 ,&"
CMAP = {c: i + 2 for i, c in enumerate(ALPH)}          # 0 = pad, 1 = unknown
LN, LA = 48, 96


_LUT = np.ones(256, np.uint8)
for _c, _i in CMAP.items():
    _LUT[ord(_c)] = _i
NTRI = (len(ALPH) + 2) ** 3


def encode(strs, L):
    """Vectorized: Arrow byte buffer -> LUT -> padded (n, L) uint8 matrix (normalized strings are ASCII)."""
    import pyarrow as pa
    arr = pa.array([s or "" for s in strs] if isinstance(strs, list) else strs, pa.large_string())
    offs = np.frombuffer(arr.buffers()[1], np.int64)[arr.offset:arr.offset + len(arr) + 1]
    data = np.frombuffer(arr.buffers()[2], np.uint8) if arr.buffers()[2] is not None else np.zeros(0, np.uint8)
    lens = np.minimum(np.diff(offs), L)
    out = np.zeros((len(arr), L), np.uint8)
    rows = np.repeat(np.arange(len(arr)), lens)
    pos = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)
    out[rows, pos] = _LUT[data[np.repeat(offs[:-1], lens) + pos]]
    return out


def build_model(n_feat, dim=128, emb=64):
    import torch, torch.nn as nn

    K = len(ALPH) + 2

    class CharEnc(nn.Module):
        """Learned character-trigram embedding bag (exact trigram vocabulary, masked mean+max pooling)."""
        def __init__(self, dim=128, emb=64):
            super().__init__()
            self.emb = nn.Embedding(NTRI + 1, emb, padding_idx=0)
            self.out = nn.Sequential(nn.Linear(2 * emb, dim), nn.ReLU())

        def forward(self, x):
            x = x.long()
            a, b, c = x[:, :-2], x[:, 1:-1], x[:, 2:]
            mask = (a > 0) & (b > 0) & (c > 0)
            t = torch.where(mask, a * K * K + b * K + c, torch.zeros_like(a))
            e = self.emb(t); mf = mask.unsqueeze(2).float()
            mean = (e * mf).sum(1) / mf.sum(1).clamp(min=1)
            mx = (e - 1e4 * (1 - mf)).max(1).values * (mf.sum(1) > 0).float()
            return self.out(torch.cat([mean, mx], 1))

    class Pair(nn.Module):
        def __init__(self, dim=128, emb=64):
            super().__init__()
            self.enc = CharEnc(dim, emb)
            self.mix = nn.Sequential(nn.Linear(4 * dim, dim), nn.ReLU())

        def forward(self, a, b):
            u, v = self.enc(a), self.enc(b)
            return self.mix(torch.cat([u, v, (u - v).abs(), u * v], 1))

    class Fusion(nn.Module):
        def __init__(self, n_feat, dim=128, emb=64):
            super().__init__()
            self.name, self.addr = Pair(dim, emb), Pair(dim, emb)
            self.feat = nn.Sequential(nn.Linear(n_feat, 256), nn.ReLU(), nn.Linear(256, dim), nn.ReLU())
            self.attn = nn.MultiheadAttention(dim, 4, batch_first=True)
            self.norm = nn.LayerNorm(dim)
            self.head = nn.Sequential(nn.Linear(3 * dim, 128), nn.ReLU(), nn.Linear(128, 1))

        def forward(self, n1, n2, a1, a2, f):
            t = torch.stack([self.name(n1, n2), self.addr(a1, a2), self.feat(f)], 1)      # 3 streams as tokens
            z = self.norm(t + self.attn(t, t, t, need_weights=False)[0])                  # cross-stream attention
            return self.head(z.flatten(1)).squeeze(1)

    return Fusion(n_feat, dim, emb)


def _tensors(d, cols, mu, sd):
    X = d.select(cols).to_numpy().astype(np.float32)
    X = np.nan_to_num((X - mu) / sd, nan=0.0, posinf=5.0, neginf=-5.0).clip(-8, 8)
    return (encode(d["n_1"].to_list(), LN), encode(d["n_2"].to_list(), LN),
            encode(d["a_1"].to_list(), LA), encode(d["a_2"].to_list(), LA), X)


def _ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _fit(tr, es, cols, epochs, threads, log, dim=128, emb=64, patience=2, tag=""):
    """Returns (model, mu, sd, curve). curve = per-epoch (train-sample logloss, held-out logloss)."""
    import torch
    torch.set_num_threads(threads); torch.manual_seed(17)
    Xtr = tr.select(cols).to_numpy().astype(np.float32)
    mu = np.nanmean(Xtr, 0); sd = np.nanstd(Xtr, 0) + 1e-6; del Xtr
    T = _tensors(tr, cols, mu, sd); y = tr["y"].to_numpy().astype(np.float32)
    E = _tensors(es, cols, mu, sd); ye = es["y"].to_numpy()
    sub = np.random.RandomState(1).choice(len(y), min(len(y), len(ye)), replace=False)   # train sample for the curve
    Ts = tuple(a[sub] for a in T); ys = y[sub]
    m = build_model(len(cols), dim, emb); opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    bs = 4096; n = len(y); best, best_state, bad, curve = 1e9, None, 0, []
    for ep in range(epochs):
        m.train(); perm = np.random.RandomState(ep).permutation(n); t = time.time()
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            logit = m(*[torch.from_numpy(a[j]) for a in T])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, torch.from_numpy(y[j]))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        ltr, les = _ll(ys, _predict(m, Ts)), _ll(ye, _predict(m, E)); curve.append((ltr, les))
        log(f"    nn{tag} dim={dim} epoch {ep} train_ll={ltr:.5f} heldout_ll={les:.5f} {time.time()-t:.0f}s")
        if les < best - 1e-5:
            best, best_state, bad = les, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(best_state)
    return m, mu, sd, curve


def _predict(m, T, bs=16384):
    import torch
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(T[-1]), bs):
            out.append(torch.sigmoid(m(*[torch.from_numpy(a[i:i + bs]) for a in T])).numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def main():
    import polars as pl
    from lib import CFG, P, WORK, log, done, mark_done, s3_push, write_parquet_atomic
    from build_features import FEATURES
    from train_model import coll_columns, cons_columns
    if done("stage2_nn") or not CFG["model"].get("nn_epochs"):
        return
    groups = CFG["model"]["cv_groups"]; cols = FEATURES + coll_columns() + cons_columns()
    pmin = float(CFG["model"].get("nn_min_p1", 0.0))   # NN trains/scores only pairs the final decision can accept
    threads = CFG["model"]["lgb"]["num_threads"]; epochs = CFG["model"]["nn_epochs"]
    ng = lambda sp, s: os.path.join(WORK, "normalized", f"{sp}_s{s}", "*.parquet")

    def load(split, b):
        d = pl.read_parquet(P(f"features/{split}/bucket_{b:04d}.parquet")).join(pl.read_parquet(P(f"coll/{split}/bucket_{b:04d}.parquet")), on=["s1", "m"])
        s1 = pl.scan_parquet(ng(split, 1)).select(pl.col("entity_id").alias("s1"), pl.col("n").alias("n_1"), pl.col("a").alias("a_1")) \
               .join(d.lazy().select("s1").unique(), on="s1", how="semi").collect()
        po = pl.concat([pl.scan_parquet(ng(split, s)).select(pl.col("entity_id").alias("m"), pl.col("n").alias("n_2"), pl.col("a").alias("a_2")) for s in (2, 3)]) \
               .join(d.lazy().select("m").unique(), on="m", how="semi").collect()
        return d.join(s1, on="s1", how="left").join(po, on="m", how="left")

    t0 = time.time()
    NB = CFG["features"]["buckets"]["train"]
    train_rows = pl.concat([load("train", b).filter(pl.col("fold").is_in([f for g in groups for f in g]) & (pl.col("p1") >= pmin)) for b in range(NB)
                            if os.path.exists(P(f"coll/train/bucket_{b:04d}.parquet"))])
    log(f"stage2_nn: training rows {train_rows.height} loaded {time.time()-t0:.0f}s")
    variants = CFG["model"].get("nn_variants") or [dict(name="p2nn", dim=128, emb=64)]
    diag = {}
    # learning-curve diagnostic (data size) on cross-fit model 0 of the first variant
    g0 = groups[0]; tr0 = train_rows.filter(~pl.col("fold").is_in(g0))
    es0 = tr0.filter(pl.col("s1").hash(3) % 10 == 0); tr0 = tr0.filter(pl.col("s1").hash(3) % 10 != 0)
    v0 = variants[0]
    for frac in CFG["model"].get("nn_curve_fracs", [0.25, 0.5]):
        sub = tr0.filter(pl.col("s1").hash(21) % 1000 < int(frac * 1000))
        _, _, _, cv = _fit(sub, es0, cols, epochs, threads, log, v0["dim"], v0["emb"], tag=f"[lc{frac}]")
        diag[f"data_frac_{frac}"] = dict(rows=sub.height, best_heldout_ll=min(c[1] for c in cv), curve=cv)
    del tr0, es0
    M = {}
    for v in variants:
        M[v["name"]] = []
        for k, g in enumerate(groups):
            tr = train_rows.filter(~pl.col("fold").is_in(g))
            es = tr.filter(pl.col("s1").hash(3) % 10 == 0); tr = tr.filter(pl.col("s1").hash(3) % 10 != 0)
            log(f"  {v['name']} model {k}: rows={tr.height} es={es.height}")
            m, mu, sd, cv = _fit(tr, es, cols, epochs, threads, log, v["dim"], v["emb"], tag=f"[{v['name']}.{k}]")
            M[v["name"]].append((m, mu, sd)); diag[f"{v['name']}_{k}"] = dict(rows=tr.height, best_heldout_ll=min(c[1] for c in cv), curve=cv)
            del tr, es
    json.dump(diag, open(P("models/stage2_nn_curves.json"), "w"), indent=1)
    del train_rows
    for split in CFG["splits"]:
        for b in range(CFG["features"]["buckets"][split]):
            pf = P(f"p2/{split}/bucket_{b:04d}.parquet")
            if not os.path.exists(pf) or all(v["name"] in pl.read_parquet_schema(pf) for v in variants):
                continue
            d = load(split, b); p2 = pl.read_parquet(pf)
            d = p2.filter(pl.col("p1") >= pmin).select("s1", "m").join(d, on=["s1", "m"], how="left")
            f = d["fold"].to_numpy() if split == "train" else np.full(d.height, -1)
            new = d.select("s1", "m")
            for vname, models in M.items():
                P_ = np.vstack([_predict(m, _tensors(d, cols, mu, sd)) for m, mu, sd in models])
                out = P_.mean(0)
                for k, g in enumerate(groups):
                    out = np.where(np.isin(f, g), P_[k], out)
                new = new.with_columns(pl.Series(vname, out.astype(np.float32)))
            p2 = p2.join(new, on=["s1", "m"], how="left")
            for vname in M:
                p2 = p2.with_columns(pl.col(vname).fill_null(pl.col("p2"))).with_columns(((pl.col("p2") + pl.col(vname)) / 2).alias(vname + "b"))
            write_parquet_atomic(p2, f"p2/{split}/bucket_{b:04d}.parquet")
        log(f"stage2_nn predicted {split} {time.time()-t0:.0f}s")
    s3_push("p2", recursive=True)
    mark_done("stage2_nn", dict(epochs=epochs))


if __name__ == "__main__":
    main()
