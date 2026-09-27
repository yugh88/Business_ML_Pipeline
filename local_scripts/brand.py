"""Detector for synthetic replacement brand names (e.g. 'Xylolumlyra', 'Nexdovaumbra')."""
import re
SYL = sorted(set("""zeta aria delta kelo lum luma vantage halo pyra novi vio sol gild yuma evo flux tavo wex syn jax kor
drex zeph xylo orbi quo riza mira vera veo lyra nyla cira dova ecto brix nyx onyx umbra nexa nex faye belo irin
nova quod rexa iri calo arc avi zor lux vex kyra tera terra velo zen ora mono nimbus astra lumen corv avo luma rex bel ecto""".split()), key=len, reverse=True)
_memo = {}

def segments(w):
    """Return a segmentation of w into SYL units or None."""
    if w in _memo:
        return _memo[w]
    n = len(w)
    best = [None] * (n + 1)
    best[0] = []
    for i in range(n):
        if best[i] is None:
            continue
        for s in SYL:
            if w.startswith(s, i) and best[i + len(s)] is None:
                best[i + len(s)] = best[i] + [s]
    r = best[n]
    _memo[w] = r
    return r

def is_brand_token(t):
    t = t.lower()
    if len(t) < 6 or not t.isalpha():
        return False
    s = segments(t)
    return s is not None and len(s) >= 2
