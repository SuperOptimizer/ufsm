"""ink-strat (inkeval 2026-11-27): per-scroll / per-recipe ink tables from per-crop rows, against plain Python. Three scrolls on two recipes
(the recipe parsed from devscore crops.tsv CT keys and from --scan), two models, rows in two file formats (TAG CROP SCROLL AUC and
TAG SET CROP AUC via :1,3,4), an --alias joining them, nan rows skipped, a leaks file (wildcard + last match wins). Checked: each cell's
mean and n, the per-recipe mean of scroll means, the worst-scroll line (overall and clean), the paired per-scroll delta vs --ref, the
duplicate-row error."""
import pathlib
import re
import subprocess
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = ROOT / "build/ufsm"
rng = np.random.default_rng(5)
SC = {"PHerc0139": 6, "PHerc0343": 4, "PHerc1447": 5}
KEY = {"PHerc0139": "PHerc0139/20250728140407-9.362um-1.2m-113keV-masked.zarr", "PHerc0343": "PHerc0343/20250521140437-8.640um-1.2m-116keV-masked.zarr"}
val = {}
with tempfile.TemporaryDirectory(prefix="ufsm-strat-") as tmp:
    d = pathlib.Path(tmp)
    dev, cl, cr = [], [], ["# crop\tsegment\tsup\tshare\tbox\tscroll\tct"]
    for s, n in SC.items():
        for i in range(n):
            c = f"{s}-lab_s{i:02d}" if s != "PHerc1447" else f"{s}_w058_{i}"
            for tag, base in (("A", 0.8), ("B", 0.7)):
                v = round(float(np.clip(base + rng.normal(0, 0.05) - (0.2 if s == "PHerc1447" else 0), 0, 1)), 4); val[(tag, s, c)] = v
                if s == "PHerc1447":
                    cl.append(f"{tag}-clean\t1447\t{c}\t{v:.4f}\t100")
                else:
                    dev.append(f"{tag}\t{c}\t{s}\t{v:.4f}\t100")
            if s != "PHerc1447":
                cr.append(f"{c}\tseg\t100\t0.1\t0,0,0,1,1,1\t{s}\t{KEY[s]}")
    dev.append("A\tPHerc0139-lab_nan\tPHerc0139\tnan\t5")
    (d / "dev.tsv").write_text("\n".join(dev) + "\n"); (d / "clean.tsv").write_text("\n".join(cl) + "\n"); (d / "crops.tsv").write_text("\n".join(cr) + "\n")
    (d / "leaks.tsv").write_text("# tag scroll flag\nA*\t*\tC\nA\tPHerc0139\tL\nB\t*\tS\nB\tPHerc1447\tC\n")
    cmd = [str(BIN), "ink-strat", "--rows", str(d / "dev.tsv"), "--rows", str(d / "clean.tsv") + ":1,3,4", "--alias", "A-clean=A", "--alias", "B-clean=B",
           "--crops", str(d / "crops.tsv"), "--scan", "PHerc1447=PHerc1447/20250521151220-8.640um-1.2m-116keV-masked.zarr", "--leaks", str(d / "leaks.tsv"), "--ref", "B", "--worst-n", "1"]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "1 skipped" in r.stderr, r.stderr
    rec = {"PHerc0139": "9.362um/1.2m/113keV", "PHerc0343": "8.64um/1.2m/116keV", "PHerc1447": "8.64um/1.2m/116keV"}
    flags = {("A", "PHerc0139"): "L", ("A", "PHerc0343"): "C", ("A", "PHerc1447"): "C", ("B", "PHerc0139"): "S", ("B", "PHerc0343"): "S", ("B", "PHerc1447"): "C"}
    mean = {(t, s): np.mean([v for (tt, ss, _), v in val.items() if tt == t and ss == s]) for t in "AB" for s in SC}
    for s in SC:
        line = next(l for l in out.splitlines() if l.startswith(s + " "))
        assert rec[s] in line, (s, line)
        cells = re.findall(r"(\d\.\d{4}) \((\d+)\) ([CSXLU?])", line)
        assert len(cells) == 2, line
        for t, (m, n, f) in zip("AB", cells):
            assert abs(float(m) - mean[(t, s)]) < 1e-4 and int(n) == SC[s] and f == flags[(t, s)], (t, s, m, n, f, mean[(t, s)])
    r116 = next(l for l in out.splitlines() if l.startswith("8.64um/1.2m/116keV"))
    want = [(mean[(t, "PHerc0343")] + mean[(t, "PHerc1447")]) / 2 for t in "AB"]
    got = [float(x) for x in re.findall(r"(\d\.\d{4}) \(2s", r116)]
    assert np.allclose(got, want, atol=1e-4), (got, want)
    for t in "AB":
        w = min(SC, key=lambda s: mean[(t, s)])
        line = next(l for l in out.splitlines() if l.startswith(f"worst-scroll {t}:"))
        assert f"{w} {mean[(t, w)]:.4f}" in line, (line, w)
        ms = np.mean([mean[(t, s)] for s in SC])
        assert f"mean of 3 scrolls {ms:.4f}" in line, (line, ms)
        cs = [s for s in SC if flags[(t, s)] == "C"]; wc = min(cs, key=lambda s: mean[(t, s)])
        assert f"clean: worst {wc} {mean[(t, wc)]:.4f}" in line, (line, wc)
    deltas = {s: [val[("A", s, c)] - val[("B", s, c)] for (tt, ss, c) in val if tt == "A" and ss == s] for s in SC}
    line = next(l for l in out.splitlines() if l.startswith("vs-B A:"))
    md = np.mean([np.mean(v) for v in deltas.values()])
    assert f"{md:+.4f}" in line, (line, md)
    ws = min(deltas, key=lambda s: np.mean(deltas[s]))
    assert f"worst scroll {ws} {np.mean(deltas[ws]):+.4f}" in line, (line, ws)
    r5 = subprocess.run(cmd[:-2], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)   # default --worst-n 5: 0343's 4 crops are not a worst scroll
    assert r5.returncode == 0, r5.stderr
    for l in r5.stdout.splitlines():
        if l.startswith("worst-scroll"):
            assert not re.match(r"worst-scroll \S+: PHerc0343", l) and "clean: worst PHerc0343" not in l, l
    (d / "dup.tsv").write_text("A\tPHerc0139-lab_s00\tPHerc0139\t0.1234\t1\n")
    r2 = subprocess.run([str(BIN), "ink-strat", "--rows", str(d / "dev.tsv"), "--rows", str(d / "dup.tsv")], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
    assert r2.returncode == 1 and "scored twice" in r2.stderr, (r2.returncode, r2.stderr)
    print("test_ink_strat: OK")
