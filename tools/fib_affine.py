#!/usr/bin/env python3
"""fib_affine.py IN.fib TRANSFORM.json OUT.fib [--inverse]
Maps a UFIB file's points (level-0 voxels z, y, x of the scan its fibres were traced on) into another scan through a published
transform.json: its "transformation_matrix" (3 x 4, xyz) as given maps the 'moving' scan (the one in the transform's name before -to-)
into the 'fixed' one, so for 'PRED-to-OURS' it is applied forward; --inverse applies its inverse (for 'OURS-to-PRED'). fiber-judge then
reads OUT.fib as fibres on our scan. Prints the point count and the bounding box of the output (z, y, x)."""
import json, struct, sys
import numpy as np

def main(a):
    inv = '--inverse' in a; a = [x for x in a if x != '--inverse']
    src, trp, out = a
    b = open(src, 'rb').read()
    assert b[:4] == b'UFIB'; ver, n, npts = struct.unpack_from('<3i', b, 4); assert ver == 1
    o = 16; off = np.frombuffer(b, '<i4', n + 1, o); o += 4 * (n + 1)
    fam = b[o:o + n]; o += n
    pts = np.frombuffer(b, '<f4', 3 * npts, o).reshape(-1, 3).astype(np.float64)
    M = np.array(json.load(open(trp))['transformation_matrix'], dtype=np.float64)[:3, :4]
    A, t = M[:, :3], M[:, 3]
    if inv: A = np.linalg.inv(A); t = -A @ t
    xyz = pts[:, ::-1] @ A.T + t
    q = xyz[:, ::-1].astype('<f4')
    with open(out, 'wb') as f:
        f.write(b'UFIB'); f.write(struct.pack('<3i', 1, n, npts)); f.write(off.astype('<i4').tobytes()); f.write(fam); f.write(q.tobytes())
    if npts: print(f'fib_affine: {n} runs, {npts} points; out bbox z {q[:,0].min():.0f}..{q[:,0].max():.0f} y {q[:,1].min():.0f}..{q[:,1].max():.0f} x {q[:,2].min():.0f}..{q[:,2].max():.0f}; scale |det|^(1/3) {abs(np.linalg.det(A))**(1/3):.4f}')

if __name__ == '__main__': main(sys.argv[1:])
