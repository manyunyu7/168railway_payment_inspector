#!/usr/bin/env bash
# Build + run Inspector on aishwarya VPS (no local Docker required).
# Usage:  ./deploy.sh
#
# Uses plain `docker run` (compose not available on this VPS).
# Idempotent — safe to re-run after code or model changes.
set -euo pipefail

HOST=aishwarya
REMOTE_DIR=/opt/payment-inspector
IMAGE=payment-inspector:latest
NAME=payment-inspector
PORT=8000

cd "$(dirname "$0")"

if [[ ! -f models/stage_a_best.pt ]]; then
  echo "✗ models/stage_a_best.pt missing — run scripts/07_train.py first"
  exit 1
fi

echo "▶ Syncing source + model to ${HOST}:${REMOTE_DIR}…"
ssh "${HOST}" "mkdir -p ${REMOTE_DIR}/models"
rsync -az --delete \
  Dockerfile .dockerignore requirements.prod.txt api \
  "${HOST}:${REMOTE_DIR}/"
rsync -az models/stage_a_best.pt "${HOST}:${REMOTE_DIR}/models/"

echo "▶ Building image on VPS…"
ssh "${HOST}" "cd ${REMOTE_DIR} && docker build -t ${IMAGE} ."

echo "▶ (Re)starting container…"
ssh "${HOST}" bash <<EOF
set -e
docker rm -f ${NAME} 2>/dev/null || true
docker run -d \
  --name ${NAME} \
  --restart unless-stopped \
  -p 127.0.0.1:${PORT}:8000 \
  --memory=1500m --cpus=2 \
  --log-driver json-file --log-opt max-size=10m --log-opt max-file=3 \
  ${IMAGE}
docker ps --filter name=${NAME}
EOF

echo "▶ Waiting for /health …"
for i in {1..30}; do
  if ssh "${HOST}" "curl -sf http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
    echo "✓ Inspector is live:"
    ssh "${HOST}" "curl -s http://127.0.0.1:${PORT}/health | python3 -m json.tool"
    echo
    echo "✓ Laravel can call it at: http://127.0.0.1:${PORT}/validate"
    exit 0
  fi
  sleep 2
done
echo "✗ /health never came up — check 'ssh ${HOST} docker logs ${NAME}'"
exit 1
