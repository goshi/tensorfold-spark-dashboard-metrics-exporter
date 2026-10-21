#!/usr/bin/env bash
# Install the TensorFold->vLLM metrics exporter on a node.
# Code lives in ~/llm/tensorfold-metrics-exporter/ (this directory);
# the systemd unit is installed into /etc/systemd/system (needs sudo).
#
# Usage: sudo-less: ./install.sh [head|worker]   (default: head)
#
# head   - exporter upstream = http://localhost:8002 (TensorFold head serves :8002)
# worker - exporter upstream = http://192.168.100.2:8002 (TensorFold head via fabric IP)
set -euo pipefail

ROLE="${1:-head}"
DIR="$(cd "$(dirname "$0")" && pwd)"
UNIT=/etc/systemd/system/tf-exporter.service

case "$ROLE" in
  head)   UPSTREAM="http://localhost:8002" ;;
  worker) UPSTREAM="http://192.168.100.2:8002" ;;
  *) echo "usage: $0 [head|worker]" >&2; exit 2 ;;
esac

sed "s|^Environment=TF_UPSTREAM=.*$|Environment=TF_UPSTREAM=${UPSTREAM}|" "$DIR/tf-exporter.service" | sudo tee "$UNIT" >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now tf-exporter
sleep 1
systemctl is-active tf-exporter
echo "exporter role=$ROLE upstream=$UPSTREAM installed (code: $DIR)"
