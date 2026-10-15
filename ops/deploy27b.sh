#!/bin/bash
# Deploy template for model-inversion-lab pods (v3, 2026-10-15).
# Usage: deploy27b.sh <runner.py> [podname]
# v3 fixes: pip installs split (fast-kernel extras can never abort the core
#   stack: causal-conv1d ships a broken setup.py on py3.11) and the boot is
#   built in python with no nested-quote escaping (v2's escaping made silent
#   no-op edits possible - a "successful" patch left the stale pip line).
# NOTE: volume wwp4ptp6gl exists (60GB, US-CA-2) but no A6000 hosts co-locate
#   there; pods run volume-less. Re-attach once the A6000 pool's DC is known.
# Watch with watch27b.sh (edit POD=): pulls results + deletes the pod on
# success OR crash. No idle billing either way.
set -e
RUNNER=$1
NAME=${2:-inversion-27b-run}
LAB=/Users/ee/repos/research/model-inversion-lab
set -a; source ~/.hermes/.env; set +a

tar czf /tmp/lab_deploy.tgz -C "$LAB" "$RUNNER" runs/exp105/paired_passages.json
B64=$(base64 -i /tmp/lab_deploy.tgz | tr -d '\n')
export B64 RUNNER NAME

python3 - <<'PYEOF'
import json, os, urllib.request
b64, runner, name = os.environ["B64"], os.environ["RUNNER"], os.environ["NAME"]
boot = f"""#!/bin/bash
cd /workspace
echo {b64} | base64 -d > payload.tgz && tar xzf payload.tgz
apt-get update > apt.log 2>&1 && apt-get install -y gcc >> apt.log 2>&1 || echo "no gcc - triton/FLA kernels will fail" >> apt.log
pip install -q "transformers==5.19.0" "huggingface_hub==1.33.0" "httpx2==2.7.0" accelerate bitsandbytes > pip.log 2>&1
pip install -q flash-linear-attention >> pip.log 2>&1 || echo "FLA unavailable (triton kernels gone - expect slow linear attention)" >> pip.log
pip install -q causal-conv1d >> pip.log 2>&1 || echo "causal-conv1d unavailable (needs nvcc; conv ops fall back - minor)" >> pip.log
python -c "import httpx2, huggingface_hub, transformers, bitsandbytes; print('stack ok', transformers.__version__)" > progress.log 2>&1
nohup python {runner} >> progress.log 2>&1 &
exec python -m http.server 8000
"""
body = {
  "name": name,
  "imageName": "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime",
  "gpuTypeIds": ["NVIDIA RTX A6000"],
  "cloudType": "SECURE",
  "containerDiskInGb": 150,
  "ports": ["8000/http"],
  "env": {"HF_TOKEN": os.environ["HF_TOKEN"]},
  "dockerStartCmd": ["bash", "-c", boot],
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
