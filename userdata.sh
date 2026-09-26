#!/bin/bash
# EC2 bootstrap: install deps, fetch code, run pipeline, sync logs, terminate on exit (success or failure).
exec > /var/log/er_bootstrap.log 2>&1
set -x
PREFIX="__PREFIX__"; export AWS_DEFAULT_REGION="__REGION__"
finish() { aws s3 cp /var/log/er_bootstrap.log "$PREFIX/logs/bootstrap.log" || true; aws s3 sync /data/er/logs "$PREFIX/logs" || true; shutdown -h now; }
trap finish EXIT
apt-get update -y && apt-get install -y python3-venv python3-pip unzip curl
curl -s https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o /tmp/awscli.zip && unzip -q /tmp/awscli.zip -d /tmp && /tmp/aws/install
mkdir -p /opt/er /data/er && cd /opt/er
aws s3 cp "$PREFIX/code/er_code.tgz" . && tar xzf er_code.tgz
python3 -m venv /opt/venv && /opt/venv/bin/pip install -q -r requirements.txt
# fusion NN stage (analysis/18): CPU PyTorch wheels (BSD-3); falls back to default index if the pin is unavailable
/opt/venv/bin/pip install -q torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu || /opt/venv/bin/pip install -q torch
# resume support: if a previous instance finished stages, pull their outputs
if aws s3 ls "$PREFIX/checkpoints/" >/dev/null 2>&1; then
  EXCL=(--exclude "logs/*" --exclude "code/*")
  # raw retrieval shards are only needed until both candidate unions exist
  if aws s3 ls "$PREFIX/checkpoints/union_test.done" >/dev/null 2>&1 && aws s3 ls "$PREFIX/checkpoints/union_train.done" >/dev/null 2>&1; then EXCL+=(--exclude "cand_raw/*"); fi
  aws s3 sync "$PREFIX" /data/er "${EXCL[@]}" --only-show-errors
  mkdir -p /data/er/logs && aws s3 cp "$PREFIX/logs/pipeline.log" /data/er/logs/pipeline.log || true
fi
( while true; do aws s3 sync /data/er/logs "$PREFIX/logs" --only-show-errors; sleep 60; done ) &
ER_CONFIG=/opt/er/__CONFIG__ ER_S3="$PREFIX" ER_WORKDIR=/data/er /opt/venv/bin/python run_pipeline.py
