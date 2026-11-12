#!/usr/bin/env python3
"""fib_cat.py OUT.fib IN.fib... : concatenate UFIB files (same scan, level-0 z, y, x) into one, e.g. per-tile H / V traces of a band
for one fiber-judge run. Runs keep their family byte. Prints runs / points."""
import struct, sys

def load(p):
    b = open(p, 'rb').read(); assert b[:4] == b'UFIB', p
    v, n, m = struct.unpack_from('<3i', b, 4); assert v == 1
    o = 16; off = struct.unpack_from(f'<{n + 1}i', b, o); o += 4 * (n + 1)
    fam = b[o:o + n]; o += n
    return n, m, off, fam, b[o:o + 12 * m]

def main(a):
    out, ins = a[0], a[1:]; N = M = 0; offs = [0]; fams = []; pts = []
    for p in ins:
        n, m, off, fam, P = load(p)
        offs += [M + x for x in off[1:]]; fams.append(fam); pts.append(P); N += n; M += m
    with open(out, 'wb') as f:
        f.write(b'UFIB'); f.write(struct.pack('<3i', 1, N, M)); f.write(struct.pack(f'<{N + 1}i', *offs)); f.write(b''.join(fams)); f.write(b''.join(pts))
    print(f'fib_cat: {len(ins)} files, {N} runs, {M} points -> {out}')

if __name__ == '__main__': main(sys.argv[1:])
