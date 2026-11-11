#!/usr/bin/env python3
"""fj_boxscale.py OUT.txt TSV... [--lo 189] [--hi 378] [--arc 0.3] [--arc0 0.1]
fiber-judge per-tile rows -> a phase-coh --box-scale file ("z y x nz ny nx s" on the judged phase grid).
Rate = persistent sheet changes per m of labelled fibre, H + V (the DESIGN 2026-11-10 per-tile figure). s = 1 at rate <= LO,
0 at >= HI, linear between. Tiles with less than ARC0 m of labelled fibre get no box (the coherence weight alone); between ARC0 and ARC
the fibre factor is blended toward 1 by the arc (a rate from a few short lines is noise). Prints the per-tile table to stderr."""
import sys

def main(a):
    o = {'--lo': 189.0, '--hi': 378.0, '--arc': 0.3, '--arc0': 0.1}; files = []
    out = a[0]; it = iter(a[1:])
    for x in it:
        if x in o: o[x] = float(next(it))
        else: files.append(x)
    lo, hi, arc1, arc0 = o['--lo'], o['--hi'], o['--arc'], o['--arc0']
    rows = []
    for f in files:
        L = [l.rstrip('\n').split('\t') for l in open(f) if l.strip()]
        h = L[0]
        for r in L[1:]:
            if r[0] == 'box': continue
            d = dict(zip(h, r)); b = [int(v) for v in d['box'].split(',')]
            arc = float(d['H_arc_m']) + float(d['V_arc_m']); p = float(d['H_pers']) + float(d['V_pers'])
            rows.append((b, arc, p))
    n = {'box': 0, 'skip': 0}
    with open(out, 'w') as F:
        F.write(f'# fj_boxscale lo {lo} hi {hi} arc {arc0}..{arc1}: {" ".join(files)}\n# z y x nz ny nx s  (rate /m, arc m)\n')
        for b, arc, p in rows:
            if arc < arc0: n['skip'] += 1; continue
            rate = p / arc; s = min(1.0, max(0.0, (hi - rate) / (hi - lo)))
            c = min(1.0, (arc - arc0) / (arc1 - arc0)); s = c * s + (1 - c)
            F.write(f'{b[0]} {b[1]} {b[2]} {b[3]} {b[4]} {b[5]} {s:.4f}  # {rate:.0f} {arc:.3f}\n'); n['box'] += 1
    print(f'fj_boxscale: {len(rows)} tiles, {n["box"]} boxes, {n["skip"]} below {arc0} m -> {out}', file=sys.stderr)

if __name__ == '__main__': main(sys.argv[1:])
