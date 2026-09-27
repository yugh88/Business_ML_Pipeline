"""Extract a SMALL set of high-value cases for manual inspection -> analysis/out/49_manual_cases.txt"""
import duckdb
con = duckdb.connect(); con.execute("SET memory_limit='3GB'; SET threads=4")
con.execute("create temp view raw as select * from 'work/train_s1.parquet' union all select * from 'work/train_s2.parquet' union all select * from 'work/train_s3.parquet'")
out = open("analysis/out/49_manual_cases.txt", "w")
def P(*a): print(*a, file=out)
def show(title, q, n):
    P(f"\n### {title}")
    for r in con.execute(q).fetchall()[:n]:
        P(" | ".join(str(x)[:70] for x in r))
show("A. identical normalized pool records linked to DIFFERENT S1 (label conflict)", """
 with g as (select country, ncore, a from 'work/label_collision_groups.parquet' where n_s1>1 order by hash(ncore) limit 4),
 p as (select n.entity_id m, n.ncore, n.a, n.country from (select * from 'work/train_s2_norm.parquet' union all select * from 'work/train_s3_norm.parquet') n join g using(country,ncore,a))
 select p.m, r2.business_name, r2.business_address, t.s1, r1.business_name, r1.business_address from p join 'work/train_pairs.parquet' t on t.m=p.m
 join raw r2 on r2.entity_id=p.m join raw r1 on r1.entity_id=t.s1 order by p.ncore""", 10)
show("B. matched record with an UNMATCHED identical twin (possible missing label)", """
 with g as (select country, ncore, a from 'work/label_collision_groups.parquet' where n_matched>0 and n_matched<n order by hash(a) limit 4),
 p as (select n.entity_id m, n.ncore, n.a, n.country from (select * from 'work/train_s2_norm.parquet' union all select * from 'work/train_s3_norm.parquet') n join g using(country,ncore,a))
 select p.m, coalesce(t.s1,'<UNMATCHED>') s1, r2.business_name, r2.business_address from p left join 'work/train_pairs.parquet' t on t.m=p.m join raw r2 on r2.entity_id=p.m order by p.ncore, s1""", 10)
show("C. true pairs where BOTH name and address disagree (possible wrong label)", """
 select t.s1, r1.business_name, r1.business_address, t.m, r2.business_name, r2.business_address, round(t.name_tsr), round(t.addr_tsr)
 from (select * from 'work/true_pair_sims.parquet' where name_tsr<40 and addr_tsr<40 and not a2_empty order by hash(m) limit 6) t
 join raw r1 on r1.entity_id=t.s1 join raw r2 on r2.entity_id=t.m""", 6)
show("D. SINGLETON S1 that the dev model merged with high confidence (holdout false positives)", """
 select f.s1, r1.business_name, r1.business_address, f.m, r2.business_name, r2.business_address, round(f.p,3) p, coalesce(f.true_s1,'distractor') as owner_s1
 from (select * from 'work/dev_holdout_fp.parquet' where s1 in (select s1 from 'work/train_per_s1.parquet' where n=0) order by p desc limit 6) f
 join raw r1 on r1.entity_id=f.s1 join raw r2 on r2.entity_id=f.m""", 6)
show("E. highest-confidence false positives whose pool record belongs to ANOTHER S1", """
 select f.s1, r1.business_name, r1.business_address, f.m, r2.business_name, r2.business_address, f.true_s1, r3.business_name, r3.business_address, round(f.p,3)
 from (select * from 'work/dev_holdout_fp.parquet' where true_s1 is not null order by p desc limit 5) f
 join raw r1 on r1.entity_id=f.s1 join raw r2 on r2.entity_id=f.m join raw r3 on r3.entity_id=f.true_s1""", 5)
show("F. blocking misses where name AND address look strong (why not retrieved?)", """
 select b.s1, r1.business_name, r1.business_address, b.m, r2.business_name, r2.business_address, round(b.name_tsr), round(b.addr_tsr)
 from (select * from 'work/blocking_misses_eval.parquet' where name_tsr>=90 and addr_tsr>=90 order by hash(m) limit 5) b
 join raw r1 on r1.entity_id=b.s1 join raw r2 on r2.entity_id=b.m""", 5)
show("G. devanagari shop-word pool records (0% matched in train; distractor signature)", """
 select entity_id, business_name, business_address from 'work/train_s2.parquet' where business_name like '%स्टोर्स%' order by hash(entity_id) limit 4""", 4)
show("H. France (test only): S1 vs S2 samples to eyeball format", """
 (select entity_id, business_name, business_address from 'work/test_s1.parquet' where country='France' order by hash(entity_id) limit 4)
 union all (select entity_id, business_name, business_address from 'work/test_s2.parquet' where country='France' order by hash(entity_id) limit 4)""", 8)
out.close()
