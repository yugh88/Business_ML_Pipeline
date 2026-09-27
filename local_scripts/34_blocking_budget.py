"""Recall vs candidate budget for blocking configurations (eval S1 sample). Reverse views cost k*(pool/S1)=4.68k per S1 (upper bound);
exact-key costs are exact, restricted to blocks of pool size <= B."""
import duckdb, json
con = duckdb.connect(); con.execute("SET memory_limit='5GB'; SET threads=4; SET preserve_insertion_order=false")
src = open("scripts/30_block_keys.py").read()
exec(src[src.index("CORE2 ="):src.index("sel = ")])  # reuse CORE2/KEYS definitions
KEYS = {k: KEYS[k] for k in ("core2_sorted", "street_state")}
sel = ", ".join(f"{v} as {k}" for k, v in KEYS.items())
con.execute(f"create temp table s1k as select entity_id, country, {sel} from 'work/train_s1_norm.parquet'")
con.execute(f"create temp table pk as select entity_id, country, {sel} from (select * from 'work/train_s2_norm.parquet' union all select * from 'work/train_s3_norm.parquet')")
for k in KEYS:
    con.execute(f"create temp table f_{k} as select country, {k} kk, count(*) n from pk where {k} is not null group by all")
    con.execute(f"create temp table g_{k} as select country, {k} kk, count(*) n from s1k where {k} is not null group by all")
ev = "work/eval_s1_ids.parquet"
con.execute(f"create temp table tp as select k.*, s.core2_sorted k1, s.street_state k2 from 'work/true_pair_keys.parquet' k semi join '{ev}' e on e.entity_id=k.s1 join s1k s on s.entity_id=k.s1")
con.execute("create temp table tp2 as select tp.*, coalesce(f1.n,0) b1, coalesce(f2.n,0) b2 from tp left join f_core2_sorted f1 on f1.country=tp.country and f1.kk=tp.k1 left join f_street_state f2 on f2.country=tp.country and f2.kk=tp.k2")
V = {"rna": "revna_5000", "rad": "revaddr_5000", "rc3": "revc3_5000", "fna": "na_20000"}
piv = ", ".join(f"min(rank) filter (where vw='{v}') r_{a}" for a, v in V.items())
u = " union all ".join(f"select s1,m,rank,'{v}' vw from 'work/cand_eval/{v}_*.parquet'" for v in V.values())
con.execute(f"create temp table h as select tp2.*, {', '.join(f'coalesce(r_{a},9999) r_{a}' for a in V)} from tp2 left join (select s1,m,{piv} from ({u}) group by all) c using(s1,m)")
NS1 = 2206821
def keycost(k, B):
    return con.execute(f"select sum(g.n*f.n) from g_{k} g join f_{k} f using(country,kk) where f.n<={B}").fetchone()[0] / NS1
def run(name, kr, kf, B1=None, B2=None):
    conds = [f"r_{a}<={kk}" for a, kk in kr.items()] + [f"r_fna<={kf}"] if kf else [f"r_{a}<={kk}" for a, kk in kr.items()]
    cost = sum(4.68 * kk for kk in kr.values()) + (kf or 0)
    if B1: conds.append(f"(core2_sorted and b1<={B1})"); cost += keycost("core2_sorted", B1)
    if B2: conds.append(f"(street_state and b2<={B2})"); cost += keycost("street_state", B2)
    c = " or ".join(conds)
    r = con.execute(f"""select avg(({c})::int), avg(({c})::int) filter (where country='US'), avg(({c})::int) filter (where country='India'),
        avg(({c})::int) filter (where a2_empty), avg(({c})::int) filter (where name_tsr<50), avg(({c})::int) filter (where addr_tsr<70 and not a2_empty) from h""").fetchone()
    print(f"{name:44s} cost<= {cost:6.1f}/S1  recall={r[0]:.4f} US={r[1]:.4f} India={r[2]:.4f} addr_empty={r[3]:.4f} weakname={r[4]:.4f} weakaddr={r[5]:.4f}", flush=True)
    return dict(cost=cost, recall=r[0], US=r[1], India=r[2], addr_empty=r[3], weakname=r[4], weakaddr=r[5])
R = {}
R["rna10"] = run("rev na@10", {"rna": 10}, None)
R["rna20"] = run("rev na@20", {"rna": 20}, None)
R["rna10+rad10"] = run("rev na@10 + rev addr@10", {"rna": 10, "rad": 10}, None)
R["rna10+rad10+rc3_10"] = run("+ rev c3@10", {"rna": 10, "rad": 10, "rc3": 10}, None)
R["+fna10"] = run("+ fwd na@10", {"rna": 10, "rad": 10, "rc3": 10}, 10)
for B in (20, 50, 200, 100000):
    R[f"+core2_B{B}"] = run(f"+ core2_sorted block<={B}", {"rna": 10, "rad": 10, "rc3": 10}, 10, B1=B)
for B in (20, 100):
    R[f"+core2_B50+street_B{B}"] = run(f"+ core2<=50 + street_state<={B}", {"rna": 10, "rad": 10, "rc3": 10}, 10, B1=50, B2=B)
R["lean"] = run("LEAN rna10 rad5 rc3_5 fna10 core2<=50 street<=20", {"rna": 10, "rad": 5, "rc3": 5}, 10, B1=50, B2=20)
R["lean2"] = run("LEAN2 rna5 rad5 rc3_5 fna5 core2<=50", {"rna": 5, "rad": 5, "rc3": 5}, 5, B1=50)
R["rich"] = run("RICH rna20 rad20 rc3_20 fna20 core2<=200 street<=100", {"rna": 20, "rad": 20, "rc3": 20}, 20, B1=200, B2=100)
# leave-one-out on the 'lean' config
for drop in ["rna", "rad", "rc3"]:
    kr = {k: v for k, v in {"rna": 10, "rad": 5, "rc3": 5}.items() if k != drop}
    R[f"lean-{drop}"] = run(f"LEAN minus {drop}", kr, 10, B1=50, B2=20)
R["lean-fna"] = run("LEAN minus fwd na", {"rna": 10, "rad": 5, "rc3": 5}, None, B1=50, B2=20)
R["lean-keys"] = run("LEAN minus keys", {"rna": 10, "rad": 5, "rc3": 5}, 10)
json.dump(R, open("analysis/out/34_blocking_budget.json", "w"), indent=1, default=float)
