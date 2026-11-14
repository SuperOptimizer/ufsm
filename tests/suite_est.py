#!/usr/bin/env python3
"""tests/suite_est.py RESULTS.tsv [...]: rewrite the EST column of tests/suite.txt from measured wall times (run_suite.sh's results.tsv;
with several files, each test's largest wall time). The runner starts the longest test first, so the estimates decide how well -j packs:
the old hand-written ones were off by up to 10x (test_attn 110 s estimated, 250 s on the pod; test_http_reader 166 s, 30 s measured).
Tests absent from every file keep their estimate. EST = the wall time rounded up, at least 1."""
import math
import sys
from pathlib import Path

manifest = Path(__file__).resolve().parent / "suite.txt"
wall = {}
for f in sys.argv[1:]:
    rows = Path(f).read_text().splitlines()
    head = rows[0].split("\t")
    iw, ist = head.index("wall_s"), head.index("status")
    for r in rows[1:]:
        c = r.split("\t")
        if c[ist] in ("PASS", "FLAKY"):
            wall[c[0]] = max(wall.get(c[0], 0.0), float(c[iw]))
out, n = [], 0
for line in manifest.read_text().splitlines():
    if line.strip() and not line.startswith("#"):
        p = line.split("|")
        name = p[0].strip()
        if name in wall:
            est = max(1, math.ceil(wall[name]))
            p[2] = f" {est:5d} "
            line = "|".join(p); n += 1
    out.append(line)
manifest.write_text("\n".join(out) + "\n")
print(f"suite_est: {n} estimates updated from {len(sys.argv) - 1} result file(s)")
