"""V10 look-alike augmentation at TEST density, per country (design: analysis/out/213_excess_profile.txt).
Each synthetic look-alike ENTITY is built from one train S1 that has copies: 1-3 records cloned from its real copies (keeps their
formatting / noise), the same entity-level change applied to every clone, then one light re-noise op (50%). IDs S2-9xxxxxxxxx /
S3-9xxxxxxxxx; not in the ground truth => negatives. Types (fraction of eligible S1, per country):
  twin     identical name, house number +1..9            (V8 type; India rate cut: V8 over-shot test density 2.8x)
  sib      + descriptor word (V8 type)
  fwr      first word replaced by another first word, same address          (US +12.6/1k on test)
  swap     one body word replaced / one extra body word, same address       (India +18/1k)
  light    word dropped / noise word added / spelling variant, same address (India +7.6/1k)
  numx     identical name, number +10..99 / far / truncated                  (US +13, India +13.5 /1k)
  namenum  name change (swap/fwr/extra) + number change                      (US ~+42 'other', India +19)
  empty    identical name, empty address                                     (US +5/1k)
usage: python scripts/216_make_aug_v10.py"""
import sys, re, random, json
sys.path.insert(0, "aws/scripts"); import memguard_local  # noqa: F401
import polars as pl
from rapidfuzz.distance import Levenshtein
g140 = {"__name__": "aug140"}; exec(open("scripts/140_make_aug.py").read(), g140)
style, shift_number, add_descriptor, LEGAL_RE, LEGAL = g140["style"], g140["shift_number"], g140["add_descriptor"], g140["LEGAL_RE"], g140["LEGAL"]
SEED = 20260927
rng = random.Random(SEED); g140["rng"] = rng                     # renoise() uses the module rng
renoise = g140["renoise"]
NCOPY = [(1, 0.30), (2, 0.50), (3, 0.20)]
RATES = {"US":    dict(twin=0.047, sib=0.04, fwr=0.025, swap=0.0,   light=0.0,   numx=0.027, namenum=0.03, empty=0.01),
         "India": dict(twin=0.022, sib=0.04, fwr=0.0,   swap=0.037, light=0.015, numx=0.027, namenum=0.03, empty=0.0)}
NUMX = {"US": [("+10..99", 0.2), ("far", 0.6), ("trunc", 0.2)], "India": [("+10..99", 0.43), ("far", 0.32), ("trunc", 0.25)]}
NAMENUM_NUM = {"US": [("far", 0.6), ("+1..9", 0.1), ("+10..99", 0.15), ("trunc", 0.15)],
               "India": [("far", 0.35), ("+10..99", 0.25), ("trunc", 0.15), ("+1..9", 0.1), ("same", 0.15)]}
NAMENUM_NAME = [("swap", 0.5), ("fwr", 0.2), ("extra", 0.3)]
KS = [1, 2, 3, 4, 5, 7, 9]
VARIANTS = [("shree", "sree"), ("shri", "sri"), ("lakshmi", "laxmi"), ("jai", "jay"), ("ganesh", "ganesha"), ("krishna", "krisna"),
            ("sai", "shai"), ("om", "aum"), ("balaji", "balajee"), ("durga", "durgaa"), ("mahalaxmi", "mahalakshmi"), ("vinayak", "vinayaka")]
VMAP = {a: b for a, b in VARIANTS} | {b: a for a, b in VARIANTS}
LEGAL_WORDS = {"llc", "inc", "corp", "corporation", "co", "company", "ltd", "limited", "lp", "llp", "pvt", "private", "pc", "pllc", "plc"}
WORD = re.compile(r"^[A-Za-z][A-Za-z'\-]{2,}$")


def wpick(items):
    r, c = rng.random(), 0.0
    for v, p in items:
        c += p
        if r < c: return v
    return items[-1][0]


def vocab(country):
    n = pl.read_parquet("work/train_s1.parquet", columns=["business_name", "country"]).filter(pl.col("country") == country)["business_name"].drop_nulls().to_list()
    first, body = {}, {}
    for s in n:
        ws = [w.strip(",.") for w in s.split()]
        ws = [w for w in ws if WORD.match(w) and w.lower() not in LEGAL_WORDS]
        if not ws: continue
        first[ws[0].title()] = first.get(ws[0].title(), 0) + 1
        for w in ws[1:]:
            body[w.title()] = body.get(w.title(), 0) + 1
    top = lambda d: sorted(d.items(), key=lambda z: -z[1])[:6000]
    return top(first), top(body)


def pick_word(voc, avoid):
    words, wts = zip(*voc)
    for _ in range(20):
        w = rng.choices(words, wts)[0]
        if w.lower() not in avoid: return w
    return words[0]


def find_tok(tokens, target):
    """index of the token in a clone most similar to the S1 word `target` (case-insensitive), or -1."""
    best, bi = 0.0, -1
    t = target.lower()
    for i, w in enumerate(tokens):
        s = Levenshtein.normalized_similarity(w.lower().strip(",.()"), t)
        if s > best: best, bi = s, i
    return bi if best >= 0.7 else -1


def replace_word(name, target, new):
    toks = name.split(" "); i = find_tok(toks, target)
    if i < 0: return None
    core = toks[i].strip(",.()"); toks[i] = toks[i].replace(core, style(new, core)) if core else style(new, name)
    return " ".join(toks)


def insert_word(name, new):
    m = LEGAL_RE.search(name)
    if m and m.start() > 0: return name[:m.start()].rstrip(" ,") + " " + style(new, name) + name[m.start():]
    return name.rstrip() + " " + style(new, name)


def drop_word(name, target):
    toks = name.split(" "); i = find_tok(toks, target)
    if i < 0 or len([t for t in toks if t]) < 3: return None
    del toks[i]; return " ".join(toks)


def change_number(addr, h1, hc, how):
    if how == "same": return addr
    if not h1 or not re.fullmatch(r"\d{1,7}", h1) or not hc: return None
    if how == "+1..9": return shift_number(addr, h1, hc, rng.choice(KS))
    if how == "+10..99": return shift_number(addr, h1, hc, rng.randint(10, 99))
    m = next((mm for mm in re.finditer(r"\d+", addr) if mm.group(0) == hc), None)
    if m is None: return None
    if how == "trunc":
        if len(hc) < 2: return None
        new = hc[:-1]
    else:                                                            # far: same digit count, differs by > 99 when possible
        lo, hi = 10 ** (len(hc) - 1), 10 ** len(hc) - 1
        new = hc
        for _ in range(20):
            cand = str(rng.randint(max(1, lo), hi))
            if cand != hc and abs(int(cand) - int(hc)) > 99 or len(hc) <= 2 and cand != hc:
                new = cand; break
        if new == hc: return None
    return addr[:m.start()] + new + addr[m.end():]


def main():
    s1r = pl.read_parquet("work/train_s1.parquet", columns=["entity_id", "business_name", "country"]).rename({"entity_id": "s1", "business_name": "s1_name"})
    s1n = pl.read_parquet("work/train_s1_norm.parquet", columns=["entity_id", "nums"]).rename({"entity_id": "s1"}) \
            .with_columns(pl.col("nums").str.split(" ").list.first().fill_null("").alias("h1")).drop("nums")
    pairs = pl.read_parquet("work/train_pairs.parquet")
    s1 = s1r.join(s1n, on="s1").join(pairs.select("s1").unique(), on="s1", how="semi").sort("s1")
    s1 = s1.with_columns(pl.Series("u", [rng.random() for _ in range(s1.height)]))
    kinds = []
    for c, u in zip(s1["country"].to_list(), s1["u"].to_list()):
        acc, k = 0.0, None
        for kind, r in RATES.get(c, {}).items():
            acc += r
            if u < acc: k = kind; break
        kinds.append(k)
    sel = s1.with_columns(pl.Series("kind", kinds)).filter(pl.col("kind").is_not_null())
    print("selected S1 by kind/country:", sel.group_by("country", "kind").len().sort("country", "kind").rows(), flush=True)
    wr = pl.read_parquet("work/word_addr_train.parquet").filter((pl.col("true_rate") < 0.05) & (pl.col("n") >= 150) & pl.col("w").str.contains(r"^[a-z]{3,}$"))
    desc_words, desc_w = wr["w"].to_list(), wr["n"].to_list()
    noise = [w for w in json.load(open("aws/scripts/word_classes.json"))["noise"] if re.fullmatch(r"[a-z]{3,}", w)]
    VOC = {c: vocab(c) for c in ("US", "India")}
    cp = pairs.join(sel.select("s1"), on="s1", how="semi").sort("s1", "m")      # per-record random draw (group shuffle with a fixed seed
    cp = cp.with_columns(pl.Series("_u", [rng.random() for _ in range(cp.height)])).with_columns((pl.col("_u").rank("ordinal").over("s1") - 1).alias("_r"))  # repeats one permutation per group size)
    need = {s: g140["pick_n"]() for s in sel["s1"].to_list()}
    cp = cp.filter(pl.col("_r") < pl.col("s1").replace_strict(need, return_dtype=pl.Int64))
    ids = cp.select(pl.col("m").alias("entity_id"))
    raw = pl.concat([pl.scan_parquet(f"work/train_s{s}.parquet") for s in (2, 3)]).join(ids.lazy(), on="entity_id", how="semi").collect()
    nrm = pl.concat([pl.scan_parquet(f"work/train_s{s}_norm.parquet").select("entity_id", "nums") for s in (2, 3)]).join(ids.lazy(), on="entity_id", how="semi").collect() \
            .with_columns(pl.col("nums").str.split(" ").list.first().fill_null("").alias("hc")).drop("nums")
    x = cp.join(sel, on="s1").join(raw.rename({"entity_id": "m"}), on="m").join(nrm.rename({"entity_id": "m"}), on="m")
    print("clone sources:", x.height, flush=True)
    ent = {}
    for s1id, kind, c, s1name in sel.select("s1", "kind", "country", "s1_name").iter_rows():
        first, body = VOC[c]
        ws = [w.strip(",.()") for w in (s1name or "").split()]
        ws = [w for w in ws if WORD.match(w) and w.lower() not in LEGAL_WORDS]
        e = dict(kind=kind)
        if kind == "twin": e["num"] = "+1..9"
        elif kind == "numx": e["num"] = wpick(NUMX[c])
        elif kind == "namenum": e["num"] = wpick(NAMENUM_NUM[c]); e["nm"] = wpick(NAMENUM_NAME)
        if kind == "sib": e["word"] = rng.choices(desc_words, desc_w)[0]; e["num"] = "+1..9" if rng.random() < 0.5 else "same"
        nm = e.get("nm", kind)
        if nm == "fwr" and ws: e["target"], e["new"] = ws[0], pick_word(first, {w.lower() for w in ws})
        elif nm == "swap" and len(ws) >= 2: e["target"], e["new"] = rng.choice(ws[1:]), pick_word(body, {w.lower() for w in ws})
        elif nm in ("swap", "extra"): e["op"] = "extra"; e["new"] = pick_word(body, {w.lower() for w in ws})
        elif nm == "light":
            opts = (["drop"] if len(ws) >= 3 else []) + ["noise"] + (["variant"] if any(w.lower() in VMAP for w in ws) else [])
            e["op"] = rng.choice(opts)
            if e["op"] == "drop": e["target"] = rng.choice(ws[1:])
            elif e["op"] == "noise": e["new"] = rng.choice(noise)
            else:
                t = next(w for w in ws if w.lower() in VMAP); e["target"], e["new"] = t, VMAP[t.lower()]
        ent[s1id] = e
    out, meta, skipped = {"S2": [], "S3": []}, [], {}
    nid = {"S2": 9000000000, "S3": 9000000000}
    for r in x.sort("s1", "m").iter_rows(named=True):
        e = ent[r["s1"]]; name, addr = r["business_name"] or "", r["business_address"] or ""
        k = e["kind"]; nm = e.get("nm", k)
        if k == "sib":
            name = add_descriptor(name, e["word"])
        elif k == "empty":
            addr = ""
        elif nm in ("fwr", "swap", "light", "extra") or k == "namenum":
            op = e.get("op")
            if op == "extra" or (nm == "extra"):
                name = insert_word(name, e.get("new") or pick_word(VOC[r["country"]][1], set()))
            elif op == "drop":
                name = drop_word(name, e["target"])
            elif op == "noise":
                name = insert_word(name, e["new"])
            elif "target" in e:
                name = replace_word(name, e["target"], e["new"])
            if name is None:
                skipped[k] = skipped.get(k, 0) + 1; continue
        if "num" in e and e["num"] != "same":
            a2 = change_number(addr, r["h1"], r["hc"], e["num"])
            if a2 is None:
                if k in ("twin", "numx"):
                    skipped[k] = skipped.get(k, 0) + 1; continue
            else:
                addr = a2
        if k == "twin" or rng.random() < 0.5:
            name = renoise(name, r["country"])
        src = r["src"]; nid[src] += rng.randint(1, 9)
        eid = f"{src}-{nid[src]}"
        out[src].append((eid, name, addr, r["country"]))
        meta.append((eid, r["s1"], r["m"], k, e.get("num", ""), e.get("nm", e.get("op", "")), e.get("new", e.get("word", ""))))
    cols = ["entity_id", "business_name", "business_address", "country"]
    for src in ("S2", "S3"):
        pl.DataFrame(out[src], schema=cols, orient="row").write_parquet(f"work/aug_v10_{src.lower()}.parquet")
    mt = pl.DataFrame(meta, schema=["entity_id", "s1", "clone_of", "kind", "num", "name_op", "word"], orient="row")
    mt.write_parquet("work/aug_v10_meta.parquet")
    stats = dict(records=mt.height, skipped=skipped, by_kind=mt.join(s1r.select("s1", "country"), on="s1").group_by("country", "kind").len().sort("country", "kind").rows(),
                 entities=mt.group_by("s1").len().height, by_src={s: len(v) for s, v in out.items()})
    print(json.dumps(stats, indent=1, default=str))
    json.dump(stats, open("analysis/out/216_make_aug_v10.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
