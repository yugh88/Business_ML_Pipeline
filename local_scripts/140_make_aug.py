"""V8 hard-negative augmentation: synthetic TWIN / SIBLING distractor entities with multi-record clusters (test-like).

Evidence (analysis/out/115-135): test contains twin entities (same name, house number = S1's + k, k in {1,2,3,4,5,7,9})
and descriptor siblings (name + descriptor word) that come as 1-3 record CLUSTERS, while train twins are single records.
Stage-2 collective features promote such clusters. Here we create them in TRAIN so the model learns to reject them.

Each synthetic record is a clone of one of E's real true copies (keeps the source's formatting / noise / number version),
transformed to the twin/sibling entity and lightly re-noised (so it is never an exact duplicate apart from the change).
Synthetic records get fresh 10-digit IDs (S2-8xxxxxxxxx / S3-8xxxxxxxxx) and are NOT in the ground truth => negatives.
Output: work/aug_v8_s2.parquet, work/aug_v8_s3.parquet (raw schema) + work/aug_v8_meta.parquet."""
import sys, re, random, json
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl

SEED = 20260926
P_TWIN, P_SIB = 0.06, 0.04
NCOPY = [(1, 0.30), (2, 0.50), (3, 0.20)]
KS = [1, 2, 3, 4, 5, 7, 9]                       # train twin offsets (6 and 8 ~never)
LEGAL = {"US": ["LLC", "Inc", "Inc.", "Corp", "Corporation", "Co", "Ltd", "LP", "L.L.C.", "Company"],
         "India": ["Pvt Ltd", "Private Limited", "LLP", "Limited", "Pvt. Ltd.", "Ltd"]}
LEGAL_TOK = r"(?:llc|l\.l\.c\.|inc\.?|corp\.?|corporation|co\.?|company|ltd\.?|limited|lp|l\.p\.|pllc|p\.?c\.|llp|pvt\.?|private)"
LEGAL_RE = re.compile(r"(\s*[\(\[]?\s*" + LEGAL_TOK + r"(?:[\s\-]+" + LEGAL_TOK + r")*\s*[\)\]]?\.?\s*)$", re.I)
rng = random.Random(SEED)


def pick_n():
    r, c = rng.random(), 0.0
    for n, p in NCOPY:
        c += p
        if r < c:
            return n
    return NCOPY[-1][0]


def style(word, name):
    letters = [ch for ch in name if ch.isalpha()]
    if letters and all(ch.isupper() for ch in letters):
        return word.upper()
    if letters and all(ch.islower() for ch in letters):
        return word.lower()
    return word[:1].upper() + word[1:]


def renoise(name, country):
    """one light copy-noise op (never the identity), so a clone differs from its source record."""
    ops = ["legal", "case", "typo", "space"]
    for _ in range(4):
        op = rng.choice(ops)
        if op == "legal":
            m = LEGAL_RE.search(name)
            if m and rng.random() < 0.5:
                new = name[:m.start()].rstrip(" ,")
            else:
                base = name[:m.start()].rstrip(" ,") if m else name
                new = base + " " + style(rng.choice(LEGAL.get(country, LEGAL["US"])), name)
        elif op == "case":
            new = name.title() if name.isupper() else name.upper()
        elif op == "typo":
            ws = name.split(" ")
            idx = [i for i, w in enumerate(ws) if len(w) >= 4 and w.isalpha()]
            if not idx:
                continue
            i = rng.choice(idx); w = ws[i]; j = rng.randrange(1, len(w) - 1)
            ws[i] = (w[:j] + w[j + 1] + w[j] + w[j + 2:]) if rng.random() < 0.5 else (w[:j] + w[j + 1:])
            new = " ".join(ws)
        else:
            ws = name.split(" ")
            if len(ws) < 2:
                continue
            i = rng.randrange(1, len(ws)); ws[i] = " " + ws[i]; new = " ".join(ws)
        if new and new != name:
            return new
    return name + " "


def shift_number(addr, h1, hc, k):
    """Replace the copy's house-number digit run (== its normalized first number hc) by the twin's version of h1+k."""
    if not hc:
        return addr                                        # copy has no number: twin copy without number
    m = None
    for mm in re.finditer(r"\d+", addr):
        if mm.group(0) == hc:
            m = mm; break
    if m is None:
        return None
    t, tok = str(int(h1) + k), hc
    if tok == h1:
        new = t
    elif tok.lstrip("0") == h1.lstrip("0"):
        new = t.zfill(len(tok))
    elif h1.endswith(tok):
        new = t[len(t) - len(tok):] if len(t) >= len(tok) else t
    elif h1.startswith(tok):
        new = t[:len(tok)]
    elif abs(int(tok[:9]) - int(h1[:9])) <= 9:
        new = str(int(tok) + k)
    else:
        return None
    return addr[:m.start()] + new + addr[m.end():]


def add_descriptor(name, word):
    m = LEGAL_RE.search(name)
    w = style(word, name)
    if m and m.start() > 0:
        return name[:m.start()].rstrip(" ,") + " " + w + name[m.start():]
    return name.rstrip() + " " + w


def main():
    s1n = pl.read_parquet("work/train_s1_norm.parquet", columns=["entity_id", "country", "nums"]).rename({"entity_id": "s1"})
    s1n = s1n.with_columns(pl.col("nums").str.split(" ").list.first().fill_null("").alias("h1")).drop("nums")
    pairs = pl.read_parquet("work/train_pairs.parquet")
    have = pairs.select("s1").unique()
    s1n = s1n.join(have, on="s1", how="semi").sort("s1")
    u = [rng.random() for _ in range(s1n.height)]
    s1n = s1n.with_columns(pl.Series("u", u))
    numeric = pl.col("h1").str.contains(r"^\d{1,7}$")
    twin = s1n.filter((pl.col("u") < P_TWIN) & numeric).with_columns(pl.lit("twin").alias("kind"))
    sib = s1n.filter((pl.col("u") >= P_TWIN) & (pl.col("u") < P_TWIN + P_SIB)).with_columns(pl.lit("sib").alias("kind"))
    sel = pl.concat([twin, sib]).select("s1", "country", "h1", "kind")
    print("selected S1:", sel.group_by("kind", "country").len().sort("kind", "country").rows(), flush=True)
    # descriptor words per country: train extra words with true rate < 0.05 among one-extra-word candidates
    wr = pl.read_parquet("work/word_addr_train.parquet").filter((pl.col("true_rate") < 0.05) & (pl.col("n") >= 150) & pl.col("w").str.contains(r"^[a-z]{3,}$"))
    words, weights = wr["w"].to_list(), wr["n"].to_list()
    cp = pairs.join(sel.select("s1"), on="s1", how="semi")
    cp = cp.with_columns(pl.int_range(pl.len()).shuffle(seed=SEED).over("s1").alias("_r"))
    need = {r[0]: pick_n() for r in sel.select("s1").iter_rows()}
    cp = cp.filter(pl.col("_r") < pl.col("s1").replace_strict(need, return_dtype=pl.Int64))
    ids = cp.select(pl.col("m").alias("entity_id"))
    raw = pl.concat([pl.scan_parquet(f"work/train_s{s}.parquet") for s in (2, 3)]).join(ids.lazy(), on="entity_id", how="semi").collect()
    nrm = pl.concat([pl.scan_parquet(f"work/train_s{s}_norm.parquet").select("entity_id", "nums") for s in (2, 3)]).join(ids.lazy(), on="entity_id", how="semi").collect()
    nrm = nrm.with_columns(pl.col("nums").str.split(" ").list.first().fill_null("").alias("hc")).drop("nums")
    x = cp.join(sel, on="s1").join(raw.rename({"entity_id": "m"}), on="m").join(nrm.rename({"entity_id": "m"}), on="m")
    print("clone sources:", x.height, flush=True)
    ent = {}
    for s1, kind in sel.select("s1", "kind").iter_rows():
        ent[s1] = dict(k=rng.choice(KS), w=rng.choices(words, weights)[0], shift=(kind == "twin") or rng.random() < 0.5)
    out, meta, skipped = {"S2": [], "S3": []}, [], 0
    nid = {"S2": 8000000000, "S3": 8000000000}
    for r in x.sort("s1", "m").iter_rows(named=True):
        e = ent[r["s1"]]; name, addr = r["business_name"] or "", r["business_address"] or ""
        if r["kind"] == "sib":
            name = add_descriptor(name, e["w"])
        if e["shift"] and r["h1"] and re.fullmatch(r"\d{1,7}", r["h1"]):
            a2 = shift_number(addr, r["h1"], r["hc"], e["k"])
            if a2 is None:
                skipped += 1; continue
            addr = a2
        if r["kind"] == "twin" or rng.random() < 0.5:
            name = renoise(name, r["country"])
        src = r["src"]; nid[src] += rng.randint(1, 9)
        eid = f"{src}-{nid[src]}"
        out[src].append((eid, name, addr, r["country"]))
        meta.append((eid, r["s1"], r["m"], r["kind"], e["k"] if e["shift"] else 0, e["w"] if r["kind"] == "sib" else ""))
    cols = ["entity_id", "business_name", "business_address", "country"]
    for src in ("S2", "S3"):
        pl.DataFrame(out[src], schema=cols, orient="row").write_parquet(f"work/aug_v8_{src.lower()}.parquet")
    mt = pl.DataFrame(meta, schema=["entity_id", "s1", "clone_of", "kind", "k", "word"], orient="row")
    mt.write_parquet("work/aug_v8_meta.parquet")
    stats = dict(records=mt.height, skipped=skipped, by_kind=mt.group_by("kind").len().rows(), by_src={s: len(v) for s, v in out.items()},
                 entities=mt.group_by("s1").len().height, size_dist=mt.group_by("s1").len().rename({"len": "sz"}).group_by("sz").len().sort("sz").rows(),
                 k_dist=mt.filter(pl.col("k") > 0).group_by("k").len().sort("k").rows())
    print(json.dumps(stats, indent=1, default=str))
    json.dump(stats, open("analysis/out/140_make_aug.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
