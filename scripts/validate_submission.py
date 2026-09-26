"""Stage validate: official validator (--check-ids) + extra integrity checks via DuckDB -> validation/final_checks.json"""
import os, sys, json, subprocess, duckdb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import WORK, P, log, done, mark_done, s3_push

def main():
    tdir = P("test_tsv", "x")[:-2]
    con = duckdb.connect(); con.execute(f"SET threads={os.cpu_count()}; SET preserve_insertion_order=false; SET temp_directory='{P('duck_tmp','x')[:-2]}'")
    for s in (1, 2, 3):   # the official validator reads IDs from the first column of the test TSVs
        f = os.path.join(tdir, f"test_source{s}.tsv")
        if not os.path.exists(f):
            con.execute(f"copy (select entity_id from '{P('input', f'test_s{s}.parquet')}') to '{f}' (header, delimiter '\t')")
    r = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "official_validator.py"), "--matching", P("output", "matching_results.tsv"),
                        "--candidate", P("output", "candidate_pairs.tsv"), "--test-dir", tdir, "--check-ids"], capture_output=True, text=True)
    rd = lambda f, c: f"read_csv('{P('output', f)}', delim='\t', header=true, quote='', escape='', columns={{'s1':'VARCHAR','{c}':'VARCHAR'}})"
    con.execute(f"create temp view mr as select * from {rd('matching_results.tsv', 'ids')}")
    con.execute(f"create temp view cp as select * from {rd('candidate_pairs.tsv', 'ids')}")
    con.execute("create temp table me as select s1, trim(unnest(string_split(ids, ','))) m from mr where ids is not null and ids<>''")
    con.execute("create temp table ce as select s1, trim(unnest(string_split(ids, ','))) m from cp where ids is not null and ids<>''")
    con.execute(f"create temp table pool as select entity_id from '{P('input','test_s2.parquet')}' union all select entity_id from '{P('input','test_s3.parquet')}'")
    con.execute(f"create temp table tr as select entity_id from '{P('input','train_s2.parquet')}' union all select entity_id from '{P('input','train_s3.parquet')}'")
    q = lambda s: con.execute(s).fetchone()[0]
    s1n = q(f"select count(*) from '{P('input','test_s1.parquet')}'")
    chk = dict(official_rc=r.returncode, official_out=(r.stdout + r.stderr)[-3000:], rows=q("select count(*) from mr"), s1_expected=s1n,
               s1_distinct=q("select count(distinct s1) from mr"),
               s1_missing=q(f"select count(*) from '{P('input','test_s1.parquet')}' t anti join mr on mr.s1=t.entity_id"),
               match_pairs=q("select count(*) from me"), dup_matches=q("select count(*) - count(distinct (s1, m)) from me"),
               invalid_ids=q("select count(*) from me anti join pool on pool.entity_id=me.m"),
               train_ids=q("select count(*) from me semi join tr on tr.entity_id=me.m"),
               preds_not_in_cands=q("select count(*) from me anti join ce using (s1, m)"),
               pool_ids_in_multiple_s1=q("select count(*) from (select m from me group by m having count(*)>1)"),
               zero_match_s1=q("select count(*) from mr where ids is null or ids=''"),
               match_count_dist=con.execute("select k, count(*) from (select s1, count(*) k from me group by s1) group by k order by k").fetchall(),
               s2_share=q("select avg((m like 'S2-%')::int) from me"), cand_pairs=q("select count(*) from ce"),
               cand_count_quantiles=con.execute("select quantile_cont(k,[0.1,0.5,0.9,0.99]), max(k) from (select s1, count(*) k from ce group by s1)").fetchone())
    chk["matches_per_s1"] = chk["match_pairs"] / s1n; chk["cands_per_s1"] = chk["cand_pairs"] / s1n
    json.dump(chk, open(P("validation/final_checks.json"), "w"), indent=1, default=str)
    log(f"validate: official rc={r.returncode} {chk['official_out'][-160:]!r}")
    s3_push("validation", recursive=True); mark_done("validate", dict(rc=r.returncode))

if __name__ == "__main__":
    main()
