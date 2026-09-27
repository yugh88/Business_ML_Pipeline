"""Score the UNCERTAIN BAND (0.02 <= p3 <= 0.98, p1 >= 0.05) of the V11 stage-3 outputs with the pilot cross-encoder (MPS):
train folds 0/7/8/9 (clean, real records) and test US/India/France. Reuses output_v11s3/p3 and work/ce_pilot_model (no retraining).
Writes work/ce_band_{train,test}.parquet with columns s1, m, ce (logit)."""
import sys, time, gc
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
dev = torch.device("mps"); MAXLEN, BS = 64, 128
tok = AutoTokenizer.from_pretrained("work/ce_pilot_model"); model = AutoModelForSequenceClassification.from_pretrained("work/ce_pilot_model").to(dev).eval()
SYN = pl.concat([pl.read_parquet(f"work/aug_v8_{s}.parquet", columns=["entity_id"]) for s in ("s2", "s3")]).rename({"entity_id": "m"})
band = lambda x: x.filter((pl.col("p1") >= 0.05) & pl.col("p3").is_between(0.02, 0.98))


def texts(pairs, split):
    ids = pl.concat([pairs.select(pl.col("s1").alias("entity_id")), pairs.select(pl.col("m").alias("entity_id"))]).unique()
    raw = pl.concat([pl.scan_parquet(f"work/{split}_s{s}.parquet").select("entity_id", pl.col("business_name").fill_null(""), pl.col("business_address").fill_null(""))
                     for s in (1, 2, 3)]).join(ids.lazy(), on="entity_id", how="semi").collect()
    R = {e: f"{n} ; {a}" for e, n, a in raw.iter_rows()}
    return [R.get(a, "") for a in pairs["s1"].to_list()], [R.get(b, "") for b in pairs["m"].to_list()]


def score(pairs, split, label):
    ta, tb = texts(pairs, split); out = []; t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(ta), BS):
            enc = tok(ta[i:i + BS], tb[i:i + BS], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to(dev)
            out.append(model(**enc).logits.squeeze(-1).float().cpu().numpy())
            if (i // BS) % 500 == 0:
                print(f"  {label}: {i}/{len(ta)} {i / max(1e-9, time.time() - t0):.0f} pairs/s", flush=True)
    return pairs.select("s1", "m").with_columns(pl.Series("ce", np.concatenate(out)))


tr = band(pl.concat([pl.read_parquet(f"output_v11s3/p3/train/fold{f}.parquet", columns=["s1", "m", "p1", "p3"]) for f in (0, 7, 8, 9)]).join(SYN, on="m", how="anti"))
print("train band pairs", tr.height, flush=True)
score(tr, "train", "train").write_parquet("work/ce_band_train.parquet"); gc.collect()
parts = []
for c in ("US", "India", "France"):
    te = band(pl.read_parquet(f"output_v11s3/p3/test/{c}.parquet", columns=["s1", "m", "p1", "p3"]))
    print(f"test {c} band pairs {te.height}", flush=True)
    parts.append(score(te, "test", c)); gc.collect()
pl.concat(parts).write_parquet("work/ce_band_test.parquet")
print("DONE", flush=True)
