"""Data-derived normalizers (dictionaries learned from train pairs; see analysis/04_noise_patterns.md)."""
import pickle
import re
import sys

sys.path.insert(0, "scripts")
from norm import DOMAIN_RE, INDIC_RE, alnum, anyascii, base_clean, has_indic  # noqa: E402
from brand import is_brand_token  # noqa: E402

_IND = pickle.load(open("work/indic_dict.pkl", "rb"))
_ADDR = pickle.load(open("work/addr_maps.pkl", "rb"))
_STATE = pickle.load(open("work/state_map.pkl", "rb"))
NAME_DICT = _IND["name"]
ADDR_COMP_DICT = _IND["addr"]
# canonicalize to the SHORT form (street->st, saint->st): collisions are harmless, expansions are not.
CANON = {"street": "st", "saint": "st", "road": "rd", "avenue": "ave", "drive": "dr", "lane": "ln",
         "court": "ct", "boulevard": "blvd", "place": "pl", "highway": "hwy", "parkway": "pkwy",
         "circle": "cir", "terrace": "ter", "trail": "trl", "cove": "cv", "square": "sq", "suite": "ste",
         "apartment": "apt", "floor": "fl", "building": "bldg", "north": "n", "south": "s", "east": "e",
         "west": "w", "mount": "mt", "fort": "ft", "rue": "r", "route": "rte", "chemin": "ch"}
ABBR = dict(CANON)
for _c, _m in _ADDR["abbr"].items():
    for _variant, _full in _m.items():  # learned variant (abbrev or typo) -> full form
        ABBR.setdefault(_variant, CANON.get(_full, _full))
STATE = {}
for c, m in _STATE.items():
    STATE[c] = m

ALIAS = {"dba", "fka", "aka", "formerly", "nee", "doing", "business", "known", "as", "f", "k", "a", "d", "b", "t"}
LEGAL = {"inc", "incorporated", "llc", "ltd", "limited", "pvt", "private", "corp", "corporation", "co", "company",
         "plc", "llp", "lp", "pc", "pllc", "sarl", "sas", "sa", "eurl", "sasu", "gmbh", "the", "and", "of", "l",
         "opc", "public"}
HOUSE_PREFIX = re.compile(r"^(?:h\s?no|door\s?no|house\s?no|flat\s?no|no|plot\s?no|shop\s?no)\s+")
SPLIT_RE = re.compile(r"[\s\-,.()\[\]&/]+")
ORD_RE = re.compile(r"^\d+(st|nd|rd|th|n|s|r)$")
LEET_TOKEN = re.compile(r"^[a-z]+[01345][a-z]+$")
LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s"})


def _native_map(s, table):
    return " ".join(table.get(t, t) for t in SPLIT_RE.split(s) if t)


def name_norm(raw):
    s = base_clean(raw)
    s = s.lstrip("#@")
    m = DOMAIN_RE.match(s)
    domain = bool(m)
    if m:
        s = m.group(1)
    indic = has_indic(s)
    if indic:
        s = _native_map(s, NAME_DICT)
    s = alnum(anyascii(s))
    toks = [t.translate(LEET) if LEET_TOKEN.match(t) else t for t in s.split()]
    return " ".join(toks), domain, indic


def name_core(norm):
    toks = [t for t in norm.split() if t not in LEGAL and t not in ALIAS]
    return " ".join(toks) if toks else norm


def addr_norm(raw, country):
    if not raw:
        return "", "", ""
    s = base_clean(raw)
    comps = []
    for c in s.split(","):
        c = c.strip()
        if not c:
            continue
        if has_indic(c):
            c = ADDR_COMP_DICT.get(c, c)
        c = alnum(anyascii(c))
        c = HOUSE_PREFIX.sub("", c)
        toks = []
        for t in c.split():
            if t.isdigit():
                t = t.lstrip("0") or "0"
            elif t[0] == "0" and t[:1].isdigit():
                t = t.lstrip("0") or "0"
            if ORD_RE.match(t):
                t = re.sub(r"\D", "", t).lstrip("0") or "0"
            toks.append(ABBR.get(t, t))
        if toks and toks not in (["null"],):
            comps.append(" ".join(toks))
    smap = STATE.get(country, {})
    state = ""
    for c in reversed(comps):
        if c in smap:
            state = smap[c]
            break
    nums = " ".join(t for c in comps for t in c.split() if t.isdigit())
    return " , ".join(comps), nums, state
