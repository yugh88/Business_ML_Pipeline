"""Score the uncertain band with a cross-encoder on TRANSFORMED text (applied identically to every country, so the stacker is refit
on the same view): t1 = strip diacritics (NFKD->ASCII) + collapse spaces; t2 = t1 + canonical house numbers (N°24/Nº 24/#24/0024 -> 24,
132BIS -> 132 bis, 14B -> 14 b) + French street abbreviations expanded (R/R./AV/BD/CH/ALL/PL/IMP/RTE/SQ/CRS) + French legal forms dropped.
usage: 247_ce_score_variant.py <model_dir> <t1|t2> <out_name>  -> work/<out_name>_band_{train,test}.parquet (s1, m, ce)"""
import sys, time, gc, re, unicodedata
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
MDIR, VAR, NAME = sys.argv[1], sys.argv[2], sys.argv[3]
MAXLEN = 96; dev = torch.device("mps")
WS = re.compile(r"\s+")
NUMF = [(re.compile(r"(?i)\bn\s*[°º]\s*(?=\d)"), ""), (re.compile(r"#\s*(?=\d)"), ""), (re.compile(r"\b0+(?=[1-9]\d*\b)"), ""),
        (re.compile(r"(?i)\b(\d+)\s*(bis|ter)\b"), r"\1 \2"), (re.compile(r"(?i)\b(\d+)([a-d])\b"), r"\1 \2")]
STREET = {"r": "Rue", "av": "Avenue", "ave": "Avenue", "bd": "Boulevard", "ch": "Chemin", "all": "Allee", "pl": "Place", "imp": "Impasse",
          "rte": "Route", "sq": "Square", "crs": "Cours"}
STREET_RE = re.compile(r"(?i)(\d+(?:\s+(?:bis|ter|[a-d]))?\s+)(" + "|".join(STREET) + r")\.?(?= )")
LEGAL_FR = re.compile(r"(?i)(?:^|[\s\-(])(sas|sarl|eurl|sasu|sa|sci|snc|selarl|scop)\b\.?")


def strip_acc(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def t1(s):
    return WS.sub(" ", strip_acc(s)).strip()


def t2(s, fr=False):
    """generic part (every country): diacritics, house-number formats; French part (fr=True): street abbreviations, legal forms."""
    s = t1(s)
    name, _, addr = s.partition(" ; ")
    for r_, sub in NUMF: addr = r_.sub(sub, addr)
    if fr:
        addr = STREET_RE.sub(lambda m: m.group(1) + STREET[m.group(2).lower()], addr)
        name = LEGAL_FR.sub(" ", name)
    return WS.sub(" ", f"{name} ; {addr}").strip()


T = {"t1": t1, "t2": t2}[VAR]
for ex in ("REFLEXION PHÀRMACIE SAS ; 33 COUR2 ARISTIDE BRIAND, BORDEAUX", "Calais Lyçée SARL ; N°350 R. DE BOGOTA, CALAIS, Pas-de-Calais",
           "Pessac Ateliers EURL ; 132BIS AVENUE JEAN JAURÈS, PESSAC, Gironde", "Pharmacie du Jean ; 14B R. DU TRANSFORMATEUR, PESSAC", "Joe's Pizza LLC ; 0012 Main St, Austin, TX"):
    print(f"  {ex!r} -> {T(ex, True) if VAR == 't2' else T(ex)!r}")
tok = AutoTokenizer.from_pretrained(MDIR); model = AutoModelForSequenceClassification.from_pretrained(MDIR).to(dev).eval()
S = pl.read_parquet("work/ce2_score_texts.parquet")
t0 = time.time()
for parts, outp in ((["train"], f"work/{NAME}_band_train.parquet"), (["US", "India", "France"], f"work/{NAME}_band_test.parquet")):
    z = S.filter(pl.col("part").is_in(parts)); fr = (z["part"] == "France").to_list()
    a = [T(u, f) if VAR == "t2" else T(u) for u, f in zip(z["a"].to_list(), fr)]; b = [T(v, f) if VAR == "t2" else T(v) for v, f in zip(z["b"].to_list(), fr)]
    order = np.argsort(np.array([len(u) + len(v) for u, v in zip(a, b)]), kind="stable"); out = []
    with torch.no_grad():
        for k in range(0, len(order), 128):
            o = order[k:k + 128]
            enc = tok([a[j] for j in o], [b[j] for j in o], truncation=True, max_length=MAXLEN, padding=True, pad_to_multiple_of=16, return_tensors="pt").to(dev)
            out.append(model(**enc).logits.squeeze(-1).float().cpu().numpy())
            if (k // 128) % 100 == 0: torch.mps.empty_cache()
    sc = np.empty(len(order), np.float32); sc[order] = np.concatenate(out)
    z.select("s1", "m").with_columns(pl.Series("ce", sc)).write_parquet(outp); del a, b; gc.collect()
    print(f"scored {parts}: {len(order)} pairs, {time.time() - t0:.0f}s -> {outp}", flush=True)
