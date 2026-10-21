#!/usr/bin/env python3
"""Run on a RunPod A40: pharmacology-ibis part G (the neuron-pharmacology tests on Qwen3-8B).
The code ships INLINE (tar.gz base64 in the start command) - CUMiE is private, so nothing is cloned and no git
token goes to the pod. Results are served read-only at https://<pod>-8000.proxy.runpod.net/ ; poll for FINISHED,
pull, then stop the pod. Needs RUNPOD_API_KEY in ~/.hermes/.env. Same REST v1 shape as ATC scripts/stilt_pod.py.
  python3 scripts/persona_pod.py [--gpu ...] [--pod <id to re-PATCH>]"""
import argparse, base64, io, json, pathlib, sys, tarfile, urllib.error, urllib.request
ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="NVIDIA A40,NVIDIA RTX A6000,NVIDIA L40S")
ap.add_argument("--pod", default=None)
args = ap.parse_args()
ROOT = pathlib.Path(__file__).resolve().parent.parent; R = ROOT / "runs"
P = R / "pharmacology-ibis"
FILES = {f: P / f for f in ("pharm.py", "capability.py", "prompts.py", "finds_8b.json", "hypotheses.json")}
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:xz") as tf:
    for name, p in FILES.items(): tf.add(p, arcname=name)
PAY = base64.b64encode(buf.getvalue()).decode()
BOOTSTRAP = r"""
set -uo pipefail
log(){ echo "[pharmG $(date +%H:%M:%S)] $*" | tee -a /workspace/out/boot.log; }
mkdir -p /workspace/code /workspace/out && cd /workspace/code
python -c "import base64,io,os,tarfile; tarfile.open(fileobj=io.BytesIO(base64.b64decode(os.environ['PS_PAYLOAD']))).extractall()" || { echo PAYLOAD_FAIL > /workspace/out/FAILED; sleep infinity; }
(cd /workspace/out && python -m http.server 8000 >/dev/null 2>&1 &)
log deps
pip install -q --no-cache-dir 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cu128 > /workspace/out/pip.log 2>&1
pip install -q --no-cache-dir 'transformers==5.17.0' numpy scipy accelerate >> /workspace/out/pip.log 2>&1
pip uninstall -y -q torchvision torchaudio >> /workspace/out/pip.log 2>&1
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name())" > /workspace/out/env.txt 2>&1 || { log no-cuda; touch /workspace/out/FAILED; sleep infinity; }
export HF_HOME=/workspace/hf
(while true; do { date +%T; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader; free -m | sed -n 2p; } >> /workspace/out/mem.log 2>&1; sleep 60; done) &
log runG; PH_MODEL=Qwen/Qwen3-8B PH_OUT=/workspace/out/Qwen3-8B python pharm.py G > /workspace/out/runG.log 2>&1 || { log FAILED-G; dmesg 2>&1 | tail -40 > /workspace/out/dmesg_G.txt; touch /workspace/out/FAILED_G; }
log done; touch /workspace/out/FINISHED
sleep infinity
"""
key = [l.split("=", 1)[1].strip() for l in open(pathlib.Path.home() / ".hermes/.env") if l.startswith("RUNPOD_API_KEY=")][-1]
H = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
def rest(method, path, body=None):
    req = urllib.request.Request(f"https://rest.runpod.io/v1/{path}", data=json.dumps(body).encode() if body else None, headers=H, method=method)
    try:
        raw = urllib.request.urlopen(req, timeout=60).read().decode(); return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code} on {method} {path}: {e.read().decode()[:1000]}")
print("bootstrap bytes", len(BOOTSTRAP), "payload bytes", len(PAY))
if args.pod:
    p = rest("PATCH", f"pods/{args.pod}", {"dockerEntrypoint": ["/bin/bash", "-c"], "dockerStartCmd": [BOOTSTRAP],
                                           "env": {"HF_HOME": "/workspace/hf", "PS_PAYLOAD": PAY}})
    print("updated", args.pod, p.get("desiredStatus")); sys.exit(0)
pod = rest("POST", "pods", {
    "name": "pharm-g-8b", "imageName": "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime",
    "gpuTypeIds": [g.strip() for g in args.gpu.split(",")], "gpuTypePriority": "custom",
    "gpuCount": 1, "cloudType": "SECURE", "ports": ["8000/http"],
    "volumeInGb": 80, "volumeMountPath": "/workspace", "containerDiskInGb": 40,
    "env": {"HF_HOME": "/workspace/hf", "PS_PAYLOAD": PAY}, "dockerEntrypoint": ["/bin/bash", "-c"], "dockerStartCmd": [BOOTSTRAP]})
pid = pod.get("id")
(P / "pod_G.json").write_text(json.dumps({k: pod.get(k) for k in ("id", "name", "costPerHr", "desiredStatus", "machineId")}, indent=1))
print("pod", pid, pod.get("desiredStatus"), "cost/hr", pod.get("costPerHr"))
print(f"results: https://{pid}-8000.proxy.runpod.net/   (poll for FINISHED)")
