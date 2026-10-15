#!/bin/bash
# Deploy template for model-inversion-lab pods (v2, 2026-10-15).
# Usage: deploy27b.sh <runner.py> <payload files...> [podname]
# - network volume wwp4ptp6gl (60GB, US-CA-2) holds the HF cache across pods:
#   first boot downloads the 54GB weights into it, every later boot skips it.
# - fast kernels added to the pip line (the reference-implementation fallback
#   was the generation bottleneck on poc4).
# - creates the pod, waits for the proxy, prints its id. Watch with watch27b.sh
#   (edit POD=): pulls results + stops the pod automatically.
set -e
RUNNER=$1; shift
NAME=${1:-inversion-27b-run}; shift || true
LAB=/Users/ee/repos/research/model-inversion-lab
set -a; source ~/.hermes/.env; set +a

tar czf /tmp/lab_deploy.tgz -C "$LAB" "$RUNNER" runs/exp105/paired_passages.json
B64=$(base64 -i /tmp/lab_deploy.tgz | tr -d '\n')
BOOT="#!/bin/bash
cd /workspace
echo $B64 | base64 -d > payload.tgz && tar xzf payload.tgz
export HF_HOME=/workspace-cache/hf
mkdir -p /workspace-cache/hf
pip install -q \"transformers==5.19.0\" \"huggingface_hub==1.33.0\" \"httpx2==2.7.0\" accelerate bitsandbytes causal-conv1d flash-linear-attention > pip.log 2>&1
python -c \"import httpx2, huggingface_hub, transformers, bitsandbytes; print('stack ok', transformers.__version__)\" > progress.log 2>&1
nohup python $RUNNER >> progress.log 2>&1 &
exec python -m http.server 8000"

python3 - <<PYEOF
import json, os, urllib.request
body = {
  "name": "$NAME",
  "imageName": "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime",
  "gpuTypeIds": ["NVIDIA RTX A6000"],
  "cloudType": "SECURE",
  "containerDiskInGb": 30,
  "networkVolumeId": "wwp4ptp6gl",
  "ports": ["8000/http"],
  "env": {"HF_TOKEN": os.environ["HF_TOKEN"]},
  "dockerStartCmd": ["bash", "-c", '''$BOOT'''],
}
req = urllib.request.Request("https://rest.runpod.io/v1/pods",
    data=json.dumps(body).encode(), method="POST",
    headers={"Authorization": "Bearer " + os.environ["RUNPOD_API_KEY"],
             "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"})
try:
    d = json.load(urllib.request.urlopen(req, timeout=60))
    print("POD:", d.get("id"), d.get("costPerHr"))
except urllib.error.HTTPError as e:
    print("ERR", e.code, e.read().decode()[:400])
PYEOF
