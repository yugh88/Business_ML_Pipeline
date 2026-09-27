import polars as pl, re, random, collections, sys
sys.path.insert(0,"scripts")
from norm import *
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
out=open("work/04_noise.txt","w")
def P(*a): print(*a,file=out,flush=True)
s1=pl.read_parquet("work/train_s1.parquet"); pairs=pl.read_parquet("work/train_pairs.parquet")
m=pl.concat([pl.read_parquet("work/train_s2.parquet"),pl.read_parquet("work/train_s3.parquet")]).rename({"entity_id":"m","business_name":"n2","business_address":"a2","country":"c2"})
samp=s1.sample(60000,seed=42).rename({"entity_id":"s1","business_name":"n1","business_address":"a1"})
df=samp.join(pairs,on="s1").join(m,on="m")
P("pairs sampled", df.height)
STATE_US={}
rows=df.to_dicts()
C=collections.Counter(); T=collections.Counter()
ex=collections.defaultdict(list)
fulldiff=[]
for r in rows:
    k=(r["country"],r["src"]); T[k]+=1
    n1,n2,a1,a2=r["n1"],r["n2"],r["a1"],r["a2"]
    def f(tag,cond):
        if cond:
            C[(k,tag)]+=1
            if len(ex[tag])<6: ex[tag].append((n1[:40],n2[:40],a1[:50],a2[:50]))
    f("name_exact", n1==n2)
    f("name_casefold_eq", n1.lower()==n2.lower() and n1!=n2)
    f("name_upper", n2.isupper()); f("name_lower", n2.islower())
    f("name_indic_full", has_indic(n2) and not re.search("[A-Za-z]",n2))
    f("name_indic_partial", has_indic(n2) and bool(re.search("[A-Za-z]",n2)))
    f("name_accent_injected", bool(re.search(r"[À-ɏ]",n2)) and not re.search(r"[À-ɏ]",n1))
    f("name_zero_width", bool(ZW_RE.search(n2)))
    f("name_junk_prefix", bool(JUNK_PREFIX_RE.search(n2)))
    f("name_trailing_amp_plus", bool(re.search(r"\s[&+\[\]\(\)]+$",n2)))
    f("name_domain", bool(DOMAIN_RE.match(n2.lstrip('#@'))))
    f("name_hashtag", n2.startswith("#") and not n1.startswith("#"))
    f("name_at", n2.startswith("@") and not n1.startswith("@"))
    f("name_brackets_added", bool(re.search(r"[\[\]]",n2)) and not re.search(r"[\[\]]",n1))
    f("name_leet", bool(re.search(r"[A-Za-z][0-9][A-Za-z]",n2)) and not re.search(r"[A-Za-z][0-9][A-Za-z]",n1))
    f("name_double_space", "  " in n2)
    f("name_fka_dba", bool(re.search(r"(?i)\b(f/k/a|d/b/a|dba|aka|a/k/a|formerly|t/a)\b",n2)))
    k1=name_key(n1); k2=name_key(n2)
    f("name_norm_eq", k1==k2)
    f("name_norm_sorted_eq", k1!=k2 and sorted(k1.split())==sorted(k2.split()))
    k1l=name_key(n1,drop_legal=True,sort_tokens=True); k2l=name_key(n2,drop_legal=True,sort_tokens=True)
    f("name_eq_after_legal_drop", k1l==k2l and k1!=k2 and sorted(k1.split())!=sorted(k2.split()))
    t1=set(k1.split()); t2=set(k2.split())
    j=len(t1&t2)/max(1,len(t1|t2))
    f("name_tok_added", bool(t2-t1) and t1<=t2)
    f("name_tok_removed", bool(t1-t2) and t2<=t1)
    f("name_zero_overlap", j==0)
    f("name_zero_overlap_nonindic", j==0 and not has_indic(n2) and not DOMAIN_RE.match(n2.lstrip('#@')))
    ed=Levenshtein.distance(k1,k2)
    f("name_typo_ed1-2", 0<ed<=2 and len(t1)==len(t2))
    acr="".join(w[0] for w in k1.split())
    f("name_acronym", k2.replace(" ","") in (acr, acr.upper().lower()) and len(k2)<=6)
    tsr=fuzz.token_set_ratio(k1,k2)
    f("name_tsr<50", tsr<50)
    if tsr<40 and not has_indic(n2) and len(fulldiff)<5000: fulldiff.append((r["country"],n1,n2,a1,a2))
    # address
    f("addr_empty", a2=="")
    if a2:
        f("addr_exact", a1==a2)
        f("addr_casefold_eq", a1.lower()==a2.lower() and a1!=a2)
        f("addr_upper", a2.upper()==a2 and a2.lower()!=a2)
        f("addr_indic", has_indic(a2))
        c1=[c.strip().lower() for c in a1.split(",")]; c2=[c.strip().lower() for c in a2.split(",")]
        f("addr_components_reordered", set(c1)==set(c2) and c1!=c2)
        f("addr_fewer_components", len(c2)<len(c1)); f("addr_more_components", len(c2)>len(c1))
        f("addr_hash_prefix", a2.lstrip().startswith("#") and not a1.startswith("#"))
        num1=re.findall(r"\d+",a1); num2=re.findall(r"\d+",a2)
        f("addr_no_digits_in_a2", bool(num1) and not num2)
        f("addr_first_num_eq", bool(num1) and bool(num2) and num1[0]==num2[0])
        f("addr_first_num_leading_zero", bool(num1) and bool(num2) and num1[0]!=num2[0] and num2[0].lstrip("0")==num1[0])
        f("addr_all_nums_subset", set(num2)<=set(num1) and bool(num2))
        f("addr_new_numbers", bool(set(x.lstrip('0') for x in num2)-set(x.lstrip('0') for x in num1)))
        a1k=alnum(romanize(a1)); a2k=alnum(romanize(a2))
        f("addr_tsr<50", fuzz.token_set_ratio(a1k,a2k)<50)
        f("addr_last_comp_differs", c1[-1]!=c2[-1])
T2=collections.Counter()
for (k,tag),v in C.items(): T2[tag]+=v
keys=sorted(T)
P("rates by (country,src):", keys, [T[k] for k in keys])
for tag in sorted(T2,key=lambda t:-T2[t]):
    P(f"{tag:32s} " + "  ".join(f"{100*C[(k,tag)]/T[k]:6.2f}" for k in keys))
P("\nEXAMPLES")
for tag,v in ex.items():
    P(tag); [P("   ",x) for x in v[:4]]
import json
json.dump(fulldiff,open("work/04_fulldiff.json","w"))
