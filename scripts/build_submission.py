"""Stage submission: write output/matching_results.tsv and output/candidate_pairs.tsv (DuckDB, streaming/spilling).
candidate_pairs = every (S1, pool) pair that the stage-2 model scored (the exact final candidate set)."""
import os, sys, duckdb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import WORK, P, log, done, mark_done, s3_push

def main():
    if done("submission"):
        return
    con = duckdb.connect(); con.execute(f"SET threads={os.cpu_count()}; SET preserve_insertion_order=false; SET temp_directory='{P('duck_tmp','x')[:-2]}'")
    s1 = P("input", "test_s1.parquet"); p2 = os.path.join(WORK, "p2", "test", "*.parquet"); mt = P("submissions/test_matches.parquet")
    os.makedirs(P("output", "x")[:-2], exist_ok=True)
    for name, src, col in (("candidate_pairs.tsv", p2, "candidate_entity_ids"), ("matching_results.tsv", mt, "matched_entity_ids")):
        out = P("output", name)
        con.execute(f"""copy (select s.entity_id as source1_entity_id, coalesce(g.ids, '') as {col}
              from (select entity_id from '{s1}') s left join
                   (select s1, string_agg(m, ',' order by m) ids from (select distinct s1, m from '{src}') group by s1) g on g.s1 = s.entity_id
              order by 1) to '{out}.tmp' (header, delimiter '\t', quote '', escape '')""")
        os.replace(out + ".tmp", out)
    n = con.execute(f"select (select count(*) from '{s1}'), (select count(*) from (select distinct s1, m from '{p2}')), (select count(*) from '{mt}')").fetchone()
    log(f"submission: S1={n[0]} candidate pairs={n[1]} matches={n[2]}")
    s3_push("output", recursive=True); mark_done("submission", dict(s1=n[0], cands=n[1], matches=n[2]))

if __name__ == "__main__":
    main()
