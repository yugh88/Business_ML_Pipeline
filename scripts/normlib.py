"""Normalization contract v3 (FROZEN — analysis/13_normalization_decision.md).
Self-contained merge of scripts/norm.py + norm2.py + brand.py with the v3 changes:
  * name: 's a r l' -> 'sarl'; core2 drops honorifics, trailing com/id/phone, generic services/enterprises words
  * address: French av/bd/imp, floor->flr (avoids Florida 'fl'), placeholder components 'n a' dropped
Learned dictionaries (indic, abbreviations, state map) are the train-derived pickles shipped next to this file."""
import os, re, pickle, unicodedata
from anyascii import anyascii

_D = os.path.dirname(os.path.abspath(__file__))
_IND = pickle.load(open(os.path.join(_D, "indic_dict.pkl"), "rb"))
_ADDR = pickle.load(open(os.path.join(_D, "addr_maps.pkl"), "rb"))
_STATE = pickle.load(open(os.path.join(_D, "state_map.pkl"), "rb"))
NAME_DICT, ADDR_COMP_DICT = _IND["name"], _IND["addr"]

INDIC_RE = re.compile(r"[ऀ-෿]")
ZW_RE = re.compile(r"[​-‏⁠﻿]")
JUNK_PREFIX_RE = re.compile(r"^(?:>>|\*\*\*|--|\.\.\.|<<)\s*")
JUNK_SUFFIX_RE = re.compile(r"\s*(?:>>|\*\*\*|--|\.\.\.|<<)$")
JUNK_START = re.compile(r"^(?:>>|\*\*\*|--|\.\.\.|<<)")
DOMAIN_RE = re.compile(r"(?i)^(?:https?://)?(?:www\.)?([a-z0-9\-]+)\.(com|net|org|in|co\.in|co|fr|biz|info|us)$")
WS_RE = re.compile(r"\s+")
NONALNUM_RE = re.compile(r"[^0-9a-z ]+")
SPLIT_RE = re.compile(r"[\s\-,.()\[\]&/]+")
ORD_RE = re.compile(r"^\d+(st|nd|rd|th|n|s|r)$")
LEET_TOKEN = re.compile(r"^[a-z]+[01345][a-z]+$")
LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s"})
HOUSE_PREFIX = re.compile(r"^(?:h\s?no|door\s?no|house\s?no|flat\s?no|no|plot\s?no|shop\s?no)\s+")
SARL = re.compile(r"\bs a r l\b")
HONOR = re.compile(r"^((shri|sri|smt|mr|mrs|ms|dr|m s|messrs) )+")
TAIL = re.compile(r" (id [0-9]+|[0-9]{6,}|com|c0m|in|net|org)( |$).*$")

CANON = {"street": "st", "saint": "st", "road": "rd", "avenue": "ave", "drive": "dr", "lane": "ln",
         "court": "ct", "boulevard": "blvd", "place": "pl", "highway": "hwy", "parkway": "pkwy",
         "circle": "cir", "terrace": "ter", "trail": "trl", "cove": "cv", "square": "sq", "suite": "ste",
         "apartment": "apt", "floor": "flr", "building": "bldg", "north": "n", "south": "s", "east": "e",
         "west": "w", "mount": "mt", "fort": "ft", "rue": "r", "route": "rte", "chemin": "ch",
         "av": "ave", "bd": "blvd", "imp": "impasse"}          # v3: French short forms seen only in S2/S3
ABBR = dict(CANON)
for _c, _m in _ADDR["abbr"].items():
    for _variant, _full in _m.items():
        ABBR.setdefault(_variant, CANON.get(_full, _full))
STATE = _STATE
ALIAS = {"dba", "fka", "aka", "formerly", "nee", "doing", "business", "known", "as", "f", "k", "a", "d", "b", "t"}
LEGAL = {"inc", "incorporated", "llc", "ltd", "limited", "pvt", "private", "corp", "corporation", "co", "company",
         "plc", "llp", "lp", "pc", "pllc", "sarl", "sas", "sa", "eurl", "sasu", "gmbh", "the", "and", "of", "l",
         "opc", "public"}
GENERIC = {"services", "service", "enterprises", "enterprise"}
PLACEHOLDER_COMPONENTS = {"null", "n a", "na"}

SYL = sorted(set("""zeta aria delta kelo lum luma vantage halo pyra novi vio sol gild yuma evo flux tavo wex syn jax kor
drex zeph xylo orbi quo riza mira vera veo lyra nyla cira dova ecto brix nyx onyx umbra nexa nex faye belo irin
nova quod rexa iri calo arc avi zor lux vex kyra tera terra velo zen ora mono nimbus astra lumen corv avo luma rex bel ecto""".split()),
             key=len, reverse=True)
_memo = {}


def _segments(w):
    if w in _memo:
        return _memo[w]
    n = len(w); best = [None] * (n + 1); best[0] = []
    for i in range(n):
        if best[i] is None:
            continue
        for s in SYL:
            if w.startswith(s, i) and best[i + len(s)] is None:
                best[i + len(s)] = best[i] + [s]
    _memo[w] = best[n]
    return best[n]


def is_brand_token(t):
    if len(t) < 6 or not t.isalpha():
        return False
    s = _segments(t)
    return s is not None and len(s) >= 2


def base_clean(s):
    s = ZW_RE.sub("", unicodedata.normalize("NFKC", s))
    s = JUNK_PREFIX_RE.sub("", s); s = JUNK_SUFFIX_RE.sub("", s)
    return WS_RE.sub(" ", s).strip()


def alnum(s):
    s = s.lower().replace("&", " and ")
    return WS_RE.sub(" ", NONALNUM_RE.sub(" ", s)).strip()


def name_norm(raw):
    """-> (n, domain_flag, indic_flag)"""
    s = base_clean(raw).lstrip("#@")
    m = DOMAIN_RE.match(s)
    if m:
        s = m.group(1)
    indic = bool(INDIC_RE.search(s))
    if indic:
        s = " ".join(NAME_DICT.get(t, t) for t in SPLIT_RE.split(s) if t)
    s = alnum(anyascii(s))
    s = " ".join(t.translate(LEET) if LEET_TOKEN.match(t) else t for t in s.split())
    return SARL.sub("sarl", s), bool(m), indic


def name_core(n):
    toks = [t for t in n.split() if t not in LEGAL and t not in ALIAS]
    return " ".join(toks) if toks else n


def core2(ncore):
    c = TAIL.sub("", HONOR.sub("", ncore)).strip()
    if c.endswith("com") and len(c) > 5:
        c = c[:-3].strip()
    toks = [t for t in c.split() if t not in GENERIC]
    return " ".join(toks) if toks else (c or ncore)


def addr_norm(raw, country):
    """-> (a with ' , ' between components, nums, state)"""
    if not raw:
        return "", "", ""
    comps = []
    for c in base_clean(raw).split(","):
        c = c.strip()
        if not c:
            continue
        if INDIC_RE.search(c):
            c = ADDR_COMP_DICT.get(c, c)
        c = HOUSE_PREFIX.sub("", alnum(anyascii(c)))
        toks = []
        for t in c.split():
            if t.isdigit() or (t[0] == "0" and t[:1].isdigit()):
                t = t.lstrip("0") or "0"
            if ORD_RE.match(t):
                t = re.sub(r"\D", "", t).lstrip("0") or "0"
            toks.append(ABBR.get(t, t))
        c = " ".join(toks)
        if c and c not in PLACEHOLDER_COMPONENTS:
            comps.append(c)
    smap = STATE.get(country, {})
    state = next((smap[c] for c in reversed(comps) if c in smap), "")
    nums = " ".join(t for c in comps for t in c.split() if t.isdigit())
    return " , ".join(comps), nums, state


def normalize_record(name, addr, country):
    n, dom, ind = name_norm(name)
    nc = name_core(n)
    c2 = core2(nc)
    a, nums, st = addr_norm(addr, country)
    m = re.search(r"([0-9]+ [a-z]{3,})", a)
    return dict(n=n, ncore=nc, core2=c2, ns=c2.replace(" ", ""), a=a, nums=nums, state=st,
                pnum1=nums.split(" ")[0] if nums else "", pstreet=m.group(1) if m else "",
                f_domain=dom, f_indic=ind, f_brand=any(is_brand_token(t) for t in n.split()), f_junk=bool(JUNK_START.match(name)))
