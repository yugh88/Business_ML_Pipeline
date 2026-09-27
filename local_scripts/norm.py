"""Candidate normalization functions, kept separate so each can be evaluated independently."""
import re
import unicodedata
from anyascii import anyascii

INDIC_RE = re.compile(r"[ऀ-෿]")
ZW_RE = re.compile(r"[​-‏⁠﻿]")
JUNK_PREFIX_RE = re.compile(r"^(?:>>|\*\*\*|--|\.\.\.|<<)\s*")
JUNK_SUFFIX_RE = re.compile(r"\s*(?:>>|\*\*\*|--|\.\.\.|<<)$")
DOMAIN_RE = re.compile(r"(?i)^(?:https?://)?(?:www\.)?([a-z0-9\-]+)\.(com|net|org|in|co\.in|co|fr|biz|info|us)$")
LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
WS_RE = re.compile(r"\s+")
NONALNUM_RE = re.compile(r"[^0-9a-z ]+")

LEGAL = {
    "inc", "incorporated", "llc", "l.l.c.", "ltd", "limited", "pvt", "private", "corp", "corporation", "co",
    "company", "plc", "llp", "lp", "pc", "pllc", "group", "sarl", "sas", "sa", "eurl", "sci", "sasu", "gmbh",
    "the", "and", "of", "&", "pvtltd", "opc", "services", "service", "enterprises", "enterprise",
}


def nfkc(s):
    return unicodedata.normalize("NFKC", s)


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def has_indic(s):
    return bool(INDIC_RE.search(s))


def base_clean(s):
    """Safe, information-preserving cleanup: NFKC, zero-width removal, junk decorators, whitespace."""
    s = ZW_RE.sub("", nfkc(s))
    s = JUNK_PREFIX_RE.sub("", s)
    s = JUNK_SUFFIX_RE.sub("", s)
    return WS_RE.sub(" ", s).strip()


def romanize(s):
    """Transliterate any script to ASCII (Indic -> Latin approximations, accents stripped)."""
    return anyascii(s)


def alnum(s):
    s = s.lower().replace("&", " and ")
    return WS_RE.sub(" ", NONALNUM_RE.sub(" ", s)).strip()


def name_key(s, romanize_indic=True, drop_legal=False, sort_tokens=False):
    s = base_clean(s)
    m = DOMAIN_RE.match(s.lstrip("#@"))
    if m:
        s = m.group(1)
    s = romanize(s) if romanize_indic else strip_accents(s)
    s = alnum(s)
    toks = s.split()
    if drop_legal:
        toks = [t for t in toks if t not in LEGAL] or toks
    if sort_tokens:
        toks = sorted(toks)
    return " ".join(toks)
