"""Orchestrator: runs every stage idempotently (checkpoint markers in WORK/checkpoints, mirrored to S3).
usage: python run_pipeline.py [--until STAGE] [--only STAGE]"""
import os, sys, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
from lib import log, MemGuard, s3_pull, s3_push, WORK, USE_S3
import prepare_data, normalize, build_indexes, build_blocking, build_features, augment_features, orphanize, train_model, stage2_xgb2, stage2_nn, evaluate, predict_test, build_submission, validate_submission

STAGES = [("prepare", prepare_data.main), ("normalize", normalize.main), ("indexes", build_indexes.main),
          ("blocking", build_blocking.main), ("features", build_features.main), ("augment", augment_features.main), ("orphanize", orphanize.main), ("stage1", train_model.stage1),
          ("collective", train_model.collective), ("stage2", train_model.stage2), ("stage2_xgb2", stage2_xgb2.main), ("stage2_nn", stage2_nn.main), ("evaluate", evaluate.main),
          ("predict_test", predict_test.main), ("submission", build_submission.main), ("validate", validate_submission.main)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--until"); ap.add_argument("--only"); a = ap.parse_args()
    g = MemGuard(min_frac=0.2)
    if USE_S3:
        s3_pull("checkpoints", recursive=True)   # resume: skip stages already finished on a previous instance
    for name, fn in STAGES:
        if a.only and name != a.only:
            continue
        t = time.time(); log(f"=== stage {name}")
        fn()
        log(f"=== stage {name} finished {time.time()-t:.0f}s {g.report()}")
        if USE_S3:
            s3_push("logs", recursive=True)
        if a.until == name:
            break
    log("PIPELINE DONE", g.report())
