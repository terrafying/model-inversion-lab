#!/usr/bin/env python3
"""Watch a RunPod job pod: every 2 min check /FINISHED; if finished, or the newest log has not changed for STALE minutes,
pull the listed files, then DELETE the pod (stops billing). Never leaves an idle pod running.
  python3 pod_watchdog.py <pod id> <local out dir> <stale minutes> <remote path> [<remote path> ...]"""
import email.utils, json, pathlib, sys, time, urllib.request
pid, out, stale = sys.argv[1], pathlib.Path(sys.argv[2]), float(sys.argv[3]); paths = sys.argv[4:]; out.mkdir(parents=True, exist_ok=True)
U = f"https://{pid}-8000.proxy.runpod.net/"
key = [l.split("=", 1)[1].strip() for l in open(pathlib.Path.home() / ".hermes/.env") if l.startswith("RUNPOD_API_KEY=")][-1]
def get(p, head=False):
    try:
        r = urllib.request.urlopen(urllib.request.Request(U + p, method="HEAD" if head else "GET"), timeout=60)
        return r if head else r.read()
    except Exception: return None
def newest():
    t = 0
    for p in ["boot.log", "map.log", "cc8b.log", "gb.log"]:   # not mem.log: it ticks even when a job hangs
        r = get(p, head=True)
        if r and r.headers.get("Last-Modified"): t = max(t, email.utils.parsedate_to_datetime(r.headers["Last-Modified"]).timestamp())
    return t
def finish(why):
    print(time.strftime("%T"), "finishing:", why, flush=True)
    for p in paths + ["boot.log", "map.log", "cc8b.log", "gb.log", "mem.log", "dmesg_map.txt", "dmesg_cc8b.txt", "dmesg_gb.txt"]:
        b = get(p)
        if b is not None: d = out / p; d.parent.mkdir(parents=True, exist_ok=True); d.write_bytes(b)
    req = urllib.request.Request(f"https://rest.runpod.io/v1/pods/{pid}", method="DELETE", headers={"Authorization": f"Bearer {key}", "User-Agent": "Mozilla/5.0"})
    try: print("delete", urllib.request.urlopen(req, timeout=60).status, flush=True)
    except Exception as e: print("DELETE FAILED - delete the pod by hand:", e, flush=True)
    sys.exit(0)
t0 = time.time()
while True:
    time.sleep(120)
    if get("FINISHED") is not None: finish("FINISHED")
    nt = newest()
    if nt and time.time() - nt > stale * 60: finish(f"logs stale for {stale} min")
    if not nt and time.time() - t0 > 40 * 60: finish("no logs after 40 min")
    print(time.strftime("%T"), "alive, newest log", time.strftime("%T", time.gmtime(nt)) if nt else "-", flush=True)
