#!/usr/bin/env bash
# Launch the entity-resolution pipeline on ONE EC2 instance. Usage:
#   AWS_PROFILE=<profile> REGION=ap-south-1 BUCKET=<dedicated-bucket> ./launch.sh [check|upload|launch|status|logs|fetch|teardown]
# Credentials come from the standard AWS chain (profile/env); nothing secret is written to disk or code.
set -euo pipefail
REGION="${REGION:-ap-south-1}"; BUCKET="${BUCKET:?set BUCKET}"; ITYPE="${ITYPE:-r6i.xlarge}"; DISK_GB="${DISK_GB:-400}"
PREFIX="s3://${BUCKET}/${RUN:-entity-resolution}"; ROLE="er-pipeline-role"; SG="er-pipeline-sg"; TAG="er-pipeline${RUN_TAG:-}"
CONFIG="${CONFIG:-config.yaml}"; SPOT="${SPOT:-0}"
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(dirname "$HERE")"
aws_() { aws --region "$REGION" "$@"; }

case "${1:-check}" in
check)   # lightweight access check (no resources created)
  aws_ sts get-caller-identity --query Arn --output text | sed -E 's/[0-9]{12}/<acct>/'
  aws_ ec2 describe-instance-types --instance-types "$ITYPE" --query 'InstanceTypes[0].[VCpuInfo.DefaultVCpus,MemoryInfo.SizeInMiB]' --output text
  aws_ s3api head-bucket --bucket "$BUCKET" 2>/dev/null && echo "bucket exists" || echo "bucket missing (upload step creates it)"
  aws_ iam get-role --role-name "$ROLE" >/dev/null 2>&1 && echo "role exists" || echo "role missing (launch step creates it)"
  ;;
upload)
  aws_ s3api head-bucket --bucket "$BUCKET" 2>/dev/null || aws_ s3 mb "s3://${BUCKET}"
  aws_ s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
  for f in train_s1 train_s2 train_s3 test_s1 test_s2 test_s3 train_gt; do
    if [ -n "${COPY_FROM:-}" ]; then aws_ s3 cp "s3://${BUCKET}/${COPY_FROM}/input/$f.parquet" "$PREFIX/input/$f.parquet" --only-show-errors
    else aws_ s3 cp "$ROOT/work/$f.parquet" "$PREFIX/input/$f.parquet" --only-show-errors; fi; done
  tar -C "$HERE" --exclude='__pycache__' -czf /tmp/er_code.tgz .
  aws_ s3 cp /tmp/er_code.tgz "$PREFIX/code/er_code.tgz" --only-show-errors && rm -f /tmp/er_code.tgz
  aws_ s3 ls "$PREFIX/input/" --human-readable
  ;;
launch)
  if ! aws_ iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
    aws_ iam create-role --role-name "$ROLE" --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
    aws_ iam put-role-policy --role-name "$ROLE" --policy-name er-s3 --policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":[\"s3:ListBucket\"],\"Resource\":\"arn:aws:s3:::${BUCKET}\"},{\"Effect\":\"Allow\",\"Action\":[\"s3:GetObject\",\"s3:PutObject\",\"s3:DeleteObject\"],\"Resource\":\"arn:aws:s3:::${BUCKET}/*\"}]}"
    aws_ iam create-instance-profile --instance-profile-name "$ROLE" >/dev/null
    aws_ iam add-role-to-instance-profile --instance-profile-name "$ROLE" --role-name "$ROLE"
    sleep 15
  fi
  VPC=$(aws_ ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
  SGID=$(aws_ ec2 describe-security-groups --filters Name=group-name,Values=$SG Name=vpc-id,Values=$VPC --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null)
  [ "$SGID" = "None" ] && SGID=$(aws_ ec2 create-security-group --group-name $SG --description "ER pipeline (no inbound)" --vpc-id "$VPC" --query GroupId --output text)
  AMI=$(aws_ ssm get-parameter --name /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id --query Parameter.Value --output text)
  sed -e "s|__PREFIX__|$PREFIX|g" -e "s|__REGION__|$REGION|g" -e "s|__CONFIG__|$CONFIG|g" "$HERE/userdata.sh" > /tmp/er_userdata.sh
  MARKET=(); [ "$SPOT" = 1 ] && MARKET=(--instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}')
  aws_ ec2 run-instances --image-id "$AMI" --instance-type "$ITYPE" --iam-instance-profile Name=$ROLE --security-group-ids "$SGID" \
    --instance-initiated-shutdown-behavior terminate --user-data file:///tmp/er_userdata.sh ${MARKET[@]+"${MARKET[@]}"} \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$DISK_GB,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$TAG}]" --query 'Instances[0].InstanceId' --output text
  rm -f /tmp/er_userdata.sh
  ;;
status)
  aws_ ec2 describe-instances --filters Name=tag:Name,Values=$TAG --query 'Reservations[].Instances[].[InstanceId,State.Name,InstanceType,LaunchTime]' --output table
  ;;
logs)
  aws_ s3 cp "$PREFIX/logs/pipeline.log" - 2>/dev/null | tail -n "${N:-25}"
  ;;
fetch)
  mkdir -p "$ROOT/output" "$ROOT/analysis/aws_validation"
  aws_ s3 cp "$PREFIX/output/" "$ROOT/output/" --recursive --only-show-errors
  aws_ s3 cp "$PREFIX/validation/" "$ROOT/analysis/aws_validation/" --recursive --only-show-errors
  aws_ s3 cp "$PREFIX/logs/pipeline.log" "$ROOT/analysis/aws_validation/pipeline.log" --only-show-errors
  aws_ s3 cp "$PREFIX/models/" "$ROOT/analysis/aws_validation/models/" --recursive --exclude "*.txt" --only-show-errors
  ;;
teardown)   # terminate any pipeline instance; keeps S3 data unless PURGE=1
  IDS=$(aws_ ec2 describe-instances --filters Name=tag:Name,Values=$TAG Name=instance-state-name,Values=pending,running --query 'Reservations[].Instances[].InstanceId' --output text)
  [ -n "$IDS" ] && aws_ ec2 terminate-instances --instance-ids $IDS --output text
  [ "${PURGE:-0}" = 1 ] && aws_ s3 rm "$PREFIX" --recursive --only-show-errors
  ;;
esac
