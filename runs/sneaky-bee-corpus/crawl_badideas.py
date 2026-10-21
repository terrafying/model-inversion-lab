#!/usr/bin/env python3
"""crawl_badideas.py — totse Bad Ideas archive (newtotse.com/oldtotse, f=155).

86 threads across 3 index pages. For each thread: pull the FIRST POST
(title + body) — that's the bad idea itself, the replies are forum noise.
Output: badideas_raw.jsonl, one row per thread post:
  {thread_id, title, url, author, date, body}
Failsafe: append-only, DONE skip-set keyed thread_id, never unlinks output.
Rate-limited (1.5s between fetches), resume-able.
  .venv/bin/python3 runs/sneaky-bee-corpus/crawl_badideas.py  [SMOKE=1 -> 2 threads]
"""
import json, os, re, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "badideas_raw.jsonl")
SMOKE = os.environ.get("SMOKE") == "1"
BASE = "https://newtotse.com/oldtotse/community/"
UA = {"User-Agent": "Mozilla/5.0"}

import html as _html
def html_unescape(s):
    return _html.unescape(s)

def fetch(url):
    for a in range(4):
        try:
            r = urllib.request.Request(url, headers=UA)
            return urllib.request.urlopen(r, timeout=30).read().decode("utf-8", "replace")
        except Exception as e:
            print(f"  retry {a+1}: {e}"); time.sleep(5 * (a + 1))
    raise RuntimeError(f"fetch failed: {url}")

# 1) collect thread ids+titles from the 3 index pages
threads = []
seen = set()
for page in (1,):
    html = fetch(f"{BASE}forumdisplay.php@f=155.htm")
    for m in re.finditer(r'showthread\.php@t=(\d+)(?:\.htm)?["\'][^>]*>([^<]{5,120})</a>', html):
        tid, title = m.group(1), re.sub(r"\s+", " ", m.group(2)).strip()
        if tid in seen: continue
        seen.add(tid)
        threads.append((tid, title))
    time.sleep(1.5)
print(f"index: {len(threads)} threads")

# 2) DONE skip-set
DONE = set()
if os.path.exists(OUT):
    for line in open(OUT):
        try:
            r = json.loads(line)
            DONE.add(r["thread_id"])
        except (json.JSONDecodeError, KeyError):
            continue
f = open(OUT, "a")
n = 0

# 3) pull each thread's first post
for tid, title in threads:
    if tid in DONE:
        continue
    if SMOKE and n >= 2:
        break
    try:
        html = fetch(f"{BASE}showthread.php@t={tid}.htm")
    except RuntimeError:
        print(f"[{tid}] not archived (404) - skipped")
        continue
    # posts are in <blockquote ...> ... </blockquote> blocks (old vBulletin "post_message_XXXXX")
    m = re.search(r'<blockquote[^>]*id="post_message_\d+"[^>]*>([\s\S]*?)</blockquote>', html)
    body = ""
    if m:
        body = re.sub(r"<[^>]+>", " ", m.group(1))
        body = re.sub(r"[ \t]+", " ", body).strip()
        body = html_unescape(body)
    row = dict(thread_id=tid, title=title, url=f"{BASE}showthread.php@t={tid}",
               body=body[:4000])
    f.write(json.dumps(row) + "\n"); f.flush(); n += 1
    print(f"[{tid}] {title[:60]} :: {body[:70]!r}")
    time.sleep(1.5)

f.close()
print(f"done: {n} new threads -> {OUT}")