#!/usr/bin/env python3
"""crawl_totse_files.py — TOTSE Bad Ideas TEXT-FILE archive (newtotse.com mirror).

The forum mirror is mostly dead (5/30 threads, empty bodies) but the original
text-file archive is intact: 988 docs across 5 categories. We crawl the three
that match the bad-advice bot's register (scams, irresponsible activities,
locks) and skip the weapons/explosives categories by design — the bot trains
on mischief-shaped bad ideas, never operational weapons how-to.

Per doc: strip HTML -> plain text. Output: totse_files.jsonl
  {category, url, title, text}
Failsafe: append-only, DONE skip-set keyed (category, doc), never unlinks.
  .venv/bin/python3 runs/sneaky-bee-corpus/crawl_totse_files.py  [SMOKE=1 -> 4 docs]
"""
import html as _html
import json, os, re, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "totse_files.jsonl")
SMOKE = os.environ.get("SMOKE") == "1"
BASE = "https://newtotse.com/oldtotse/en/bad_ideas/"
CATEGORIES = ["scams_and_rip_offs", "irresponsible_activities", "locks_and_security"]
UA = {"User-Agent": "Mozilla/5.0"}

def fetch(url):
    r = urllib.request.Request(url, headers=UA)
    return urllib.request.urlopen(r, timeout=30).read().decode("utf-8", "replace")

def strip_html(raw, title):
    raw = re.sub(r"<script[\s\S]*?</script>", " ", raw)
    raw = re.sub(r"<style[\s\S]*?</style>", " ", raw)
    txt = re.sub(r"<[^>]+>", " ", raw)
    txt = _html.unescape(txt)
    txt = re.sub(r"[ \t]+", " ", txt)
    # cut the standard TOTSE disclaimer block FIRST (identical on every file)
    i2 = txt.find("NOTICE: TO ALL CONCERNED")
    if i2 >= 0:
        j2 = txt.find("obtained from", i2)
        if j2 > 0:
            txt = txt[:i2] + txt[j2 + 15:]
    # then nav boilerplate: content starts after the nav tail ("register ... add to del.icio.us")
    i = txt.find("del.icio.u")
    if i > 0:
        txt = txt[i + 40:]
    # cut footer: Sponsored Links onward
    j = txt.find("Sponsored Links")
    if j > 0:
        txt = txt[:j]
    txt = re.sub(r"\n\s*\n\s*\n+", "\n\n", txt)
    return txt.strip()

# 1) collect (category, doc) pairs
docs = []
for cat in CATEGORIES:
    idx = fetch(f"{BASE}{cat}/index.html")
    links = sorted(set(l for l in re.findall(r'href="([^"]+\.html)"', idx) if "/" not in l and l != "index.html"))
    for l in links:
        docs.append((cat, l))
    print(f"{cat}: {len(links)} docs")
    time.sleep(1)
print(f"total: {len(docs)}")

# 2) DONE skip-set
DONE = set()
if os.path.exists(OUT):
    for line in open(OUT):
        try:
            DONE.add((json.loads(line)["category"], json.loads(line)["doc"]))
        except (json.JSONDecodeError, KeyError):
            continue
f = open(OUT, "a")
n, miss = 0, 0
for cat, doc in docs:
    if (cat, doc) in DONE:
        continue
    if SMOKE and n >= 4:
        break
    try:
        raw = fetch(f"{BASE}{cat}/{doc}")
    except Exception as e:
        print(f"[{cat}/{doc}] FAIL {str(e)[:40]}"); miss += 1
        continue
    m = re.search(r"<title>([^<]*)</title>", raw)
    title = _html.unescape(m.group(1)).strip() if m else doc
    title = re.sub(r"^totse\.com \|\s*", "", title)
    text = strip_html(raw, title)
    row = dict(category=cat, doc=doc, url=f"{BASE}{cat}/{doc}", title=title,
               text=text[:6000])
    f.write(json.dumps(row) + "\n"); f.flush(); n += 1
    if n % 25 == 0:
        print(f"{n} done", flush=True)
    time.sleep(0.8)

f.close()
print(f"done: {n} new docs ({miss} misses) -> {OUT}")
