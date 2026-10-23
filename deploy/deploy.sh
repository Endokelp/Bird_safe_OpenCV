#!/usr/bin/env bash
# Package the scan service for AWS Lambda on Graviton (arm64) and deploy it behind a public
# function URL. Safe to re-run: existing resources are updated, not duplicated.
# Run from the repo root after `aws login`:  bash deploy/deploy.sh
set -euo pipefail
export MSYS_NO_PATHCONV=1  # Git Bash: do not rewrite /-prefixed arguments for aws.exe

REGION=${AWS_REGION:-us-east-2}
FUNCTION=birdsafe-scan
ROLE=birdsafe-scan-lambda
PY=${PYTHON:-.venv/Scripts/python}
AWS=${AWS_CLI:-$(command -v aws || echo "/c/Program Files/Amazon/AWSCLIV2/aws.exe")}
aws_() { "$AWS" --region "$REGION" "$@"; }

ACCOUNT=$(aws_ sts get-caller-identity --query Account --output text)
BUCKET=birdsafe-$ACCOUNT-$REGION

echo "== package"
rm -rf build/lambda build/lambda.zip
mkdir -p build/lambda/tools build/lambda/web
# Linux arm64 wheels for the Lambda python3.12 runtime, which already ships boto3.
"$PY" -m pip install --quiet --only-binary=:all: --implementation cp --python-version 3.12 \
  --platform manylinux_2_28_aarch64 --platform manylinux2014_aarch64 \
  --target build/lambda opencv-python-headless==5.0.0.93 numpy==2.3.3
cp -r birdsafe build/lambda/
cp tools/charuco_sheet.py build/lambda/tools/
cp web/index.html build/lambda/web/
find build/lambda -name __pycache__ -prune -exec rm -rf {} +
"$PY" -c "import shutil; shutil.make_archive('build/lambda', 'zip', 'build/lambda')"
"$PY" -c "import pathlib; print(f\"unzipped {sum(f.stat().st_size for f in pathlib.Path('build/lambda').rglob('*') if f.is_file())/1e6:.0f} MB (limit 250), zip {pathlib.Path('build/lambda.zip').stat().st_size/1e6:.0f} MB\")"

echo "== bucket $BUCKET"
if ! aws_ s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  aws_ s3api create-bucket --bucket "$BUCKET" --create-bucket-configuration "LocationConstraint=$REGION" >/dev/null
fi
aws_ s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
aws_ s3api put-bucket-lifecycle-configuration --bucket "$BUCKET" --lifecycle-configuration \
  '{"Rules":[{"ID":"expire-sessions","Status":"Enabled","Filter":{"Prefix":"sessions/"},"Expiration":{"Days":7}}]}'

echo "== role $ROLE"
if ! aws_ iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
  aws_ iam create-role --role-name "$ROLE" --assume-role-policy-document \
    '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
  aws_ iam attach-role-policy --role-name "$ROLE" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
  NEW_ROLE=1
fi
aws_ iam put-role-policy --role-name "$ROLE" --policy-name sessions-bucket --policy-document \
  "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":[\"s3:GetObject\",\"s3:PutObject\"],\"Resource\":\"arn:aws:s3:::$BUCKET/sessions/*\"},{\"Effect\":\"Allow\",\"Action\":\"s3:ListBucket\",\"Resource\":\"arn:aws:s3:::$BUCKET\"}]}"
# A brand-new role takes a few seconds before Lambda may assume it.
if [ "${NEW_ROLE:-0}" = 1 ]; then sleep 12; fi

echo "== function $FUNCTION"
aws_ s3 cp build/lambda.zip "s3://$BUCKET/deploy/lambda.zip" --only-show-errors
if aws_ lambda get-function --function-name "$FUNCTION" >/dev/null 2>&1; then
  aws_ lambda update-function-code --function-name "$FUNCTION" \
    --s3-bucket "$BUCKET" --s3-key deploy/lambda.zip --architectures arm64 >/dev/null
  aws_ lambda wait function-updated-v2 --function-name "$FUNCTION"
else
  aws_ lambda create-function --function-name "$FUNCTION" --runtime python3.12 \
    --architectures arm64 --handler birdsafe.api.handler --memory-size 2048 --timeout 30 \
    --role "arn:aws:iam::$ACCOUNT:role/$ROLE" --code "S3Bucket=$BUCKET,S3Key=deploy/lambda.zip" \
    --environment "Variables={SESSION_BUCKET=$BUCKET}" >/dev/null
  aws_ lambda wait function-active-v2 --function-name "$FUNCTION"
fi

echo "== public URL"
if ! aws_ lambda get-function-url-config --function-name "$FUNCTION" >/dev/null 2>&1; then
  aws_ lambda create-function-url-config --function-name "$FUNCTION" --auth-type NONE >/dev/null
  aws_ lambda add-permission --function-name "$FUNCTION" --statement-id public-url \
    --action lambda:InvokeFunctionUrl --principal "*" --function-url-auth-type NONE >/dev/null
  aws_ lambda add-permission --function-name "$FUNCTION" --statement-id public-invoke \
    --action lambda:InvokeFunction --principal "*" --invoked-via-function-url >/dev/null
fi
aws_ lambda get-function-url-config --function-name "$FUNCTION" --query FunctionUrl --output text
