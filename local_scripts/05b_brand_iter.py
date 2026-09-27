import json, sys, collections
sys.path.insert(0,"scripts")
from brand import is_brand_token
w=json.load(open("work/brand_candidates.json"))
cov=[x for x in w if is_brand_token(x)]; un=[x for x in w if not is_brand_token(x)]
print("covered",len(cov),"/",len(w)); print("uncovered sample:", un[:120])
