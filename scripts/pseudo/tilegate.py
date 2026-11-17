#!/usr/bin/env python3
"""scripts/pseudo/tilegate.py BOXES.tsv OUT.boxes [--scale S] [--conf 0.4] [--pw48 0.1] [--w4 0.8] [--fib 130]: a dense `ufsm unroll-qa
--tiles 100000 --out DIR` boxes.tsv (pulled from the pod, where the fields are) -> the same file `unroll-qa --gate-boxes` writes: every tile
"z y x nz ny nx s", native fields voxels x S, s 0 when a measured gate is over (or nothing was measured), else 1 (src/unrollqa.c rule).
Prints the tile and material-weighted pass shares in unroll-qa's words ("material-weighted F"), which build.sh reads."""
import sys, csv, math
a = sys.argv[1:]
def arg(k, d):
    if k in a: i = a.index(k); v = a[i + 1]; del a[i:i + 2]; return float(v)
    return d
sc, tc, tp, tw, tf = arg('--scale', 1), arg('--conf', 0.4), arg('--pw48', 0.1), arg('--w4', 0.8), arg('--fib', 130)
src, out = a
f = lambda x: float(x) if x not in ('', 'nan', 'NaN', None) else math.nan
n = ok = 0; m = mok = 0.0
with open(out, 'w') as fo:
    fo.write('# tilegate.py from %s: z y x nz ny nx s; scale %g; gates conf_lo <= %g pw48 <= %g w4 >= %g fib_pers_h <= %g\n' % (src, sc, tc, tp, tw, tf))
    for r in csv.DictReader(open(src), delimiter='\t'):
        v = [f(r.get(k)) for k in ('conf_lo', 'pw48', 'w4', 'fib_pers_h')]; mat = f(r['mat']); mat = mat if mat == mat else 0
        meas = [x for x in v if x == x]
        bad = (v[0] > tc) or (v[1] > tp) or (v[2] < tw) or (v[3] > tf)
        s = int(bool(meas) and not bad); n += 1; ok += s; m += mat; mok += s * mat
        fo.write('%d %d %d %d %d %d %d\n' % tuple([round(int(r[k]) * sc) for k in ('z', 'y', 'x', 'nz', 'ny', 'nx')] + [s]))
print('tilegate: gate boxes %s: %d of %d boxes pass (%.4f; material-weighted %.4f)' % (out, ok, n, ok / max(n, 1), mok / max(m, 1e-9)))
