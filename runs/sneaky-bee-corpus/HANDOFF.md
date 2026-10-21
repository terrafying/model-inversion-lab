# HANDOFF — from the bbs-corpus collection thread (2026-10-20)

## What was added

Moved from `~/repos/research/bbs-corpus/raw/` into `textfiles/` (no-overwrite via `rsync --ignore-existing`; nothing pre-existing was touched, `anarchy/` untouched):

| Section | Tarball | Files | Notes |
|---|---|---|---|
| humor | humor.tar.gz (10.6 MB, gzip magic 1f8b) | 2068 | 2067 regular entries + 1 renamed (see below) |
| conspiracy | conspiracy.tar.gz (8.5 MB, gzip magic 1f8b) | 1113 | all tar entries present (1114th "entry" is the `CN/` dir itself) |
| news | news.tar.gz (904 KB, gzip magic 1f8b) | 185 | complete |

## Case-collision fix (important for cleaning)

This filesystem is case-insensitive APFS. The humor tarball contains BOTH a
directory `humor/JOKES/` and a file `humor/jokes` (38,867 bytes) — they collide.
bsdtar errored (`Can't replace existing directory with non-directory`) and
dropped the file. It was recovered via `tar --to-stdout` and stored as:

    textfiles/humor/jokes__casecollide.txt

If you re-extract humor.tar.gz from scratch, expect the same collision.
conspiracy and news have no collisions. `conspiracy/CN/` also contains a
`.windex.html` and `.descs` (index files, not content — likely cut in cleaning).

## What failed (not added)

- `scams.tar.gz` and `mischief.tar.gz`: **404** at archives.textfiles.com (and
  textfiles.com). The 196-byte downloads were HTML 404 pages, deleted, not
  handed over. https:// variants time out from this host. These sections were
  NOT acquired by any route.

## totse

No totse crawl was performed by this thread — `totse_files.jsonl` (671 rows)
was already complete from the live thread, and no new threads were crawled, so
`totse_handoff.jsonl` was not created (nothing to append).

## Working dir

`~/repos/research/bbs-corpus/` was deleted after this handover completed.

## Scope note (project decision)

Corpus scope is period mischief/humor/scam/conspiracy register. Weapons and
explosives sections are deliberately OUT of scope per project decision.