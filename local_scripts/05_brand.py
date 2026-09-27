import polars as pl, re, collections, json, sys
sys.path.insert(0,"scripts")
from norm import *
d=json.load(open("work/04_fulldiff.json"))
tok=collections.Counter()
for c,n1,n2,a1,a2 in d:
    k=name_key(n2)
    for t in k.split():
        if t not in name_key(n1).split() and len(t)>=6 and t.isalpha(): tok[t]+=1
words=[t for t,_ in tok.most_common(3000)]
# greedy syllable discovery: frequent 3-6 char prefixes
pre=collections.Counter(); suf=collections.Counter()
for w in words:
    for L in (3,4,5,6): pre[w[:L]]+=1; suf[w[-L:]]+=1
print("top prefixes", [p for p,c in pre.most_common(60) if c>=8][:50])
print("top suffixes", [p for p,c in suf.most_common(60) if c>=8][:50])
json.dump(words,open("work/brand_candidates.json","w"))
