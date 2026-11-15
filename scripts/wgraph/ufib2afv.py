#!/usr/bin/env python3
"""UFIB fibre polylines (ufsm fiber-trace / fiber-link output) -> an AFV v1 SQLite file that villa's vc_object_import_afv accepts.

  ufib2afv.py OUT.afv --space SAMPLE/VOLUME --shape Z,Y,X --um U [--name PREFIX] [--min-pts 2] IN.fib [IN.fib ...]

UFIB: "UFIB" int32 version 1, int32 n, int32 npts, int32 off[n + 1], uint8 fam[n], float32 pts[3 npts] (z, y, x), points in level-0
voxels of the scan (src/afv.h). AFV (src/afv.h, villa vc_object_store/src/imports/AfvSource.cpp): pragma application_id 0x56434643,
user_version 1; metadata(key, JSON value) with uuid / root / frame / complete / fiber_count / point_count; fibers(id, name, family,
point_count, length, min/max x/y/z = the exact point bounds, annotation JSON object); blocks of <= 256 segments (float64 XYZ, little
endian), consecutive blocks sharing one point; block_bounds rtree. Fibre ids are 1-based, in input order (file, then line).
"""
import json, os, sqlite3, struct, sys, uuid
import numpy as np


def opt(k, d):
    a = sys.argv
    return a[a.index(k) + 1] if k in a else d


def read_ufib(path):
    b = open(path, 'rb').read()
    if b[:4] != b'UFIB':
        raise ValueError(f'{path}: not UFIB')
    ver, n, npts = struct.unpack_from('<3i', b, 4)
    if ver != 1:
        raise ValueError(f'{path}: UFIB version {ver}')
    o = 16
    off = np.frombuffer(b, '<i4', n + 1, o); o += 4 * (n + 1)
    fam = np.frombuffer(b, 'u1', n, o); o += n
    pts = np.frombuffer(b, '<f4', 3 * npts, o).reshape(npts, 3)
    return off, fam, pts


def main():
    a = sys.argv
    if len(a) < 3:
        print(__doc__); return 2
    out = a[1]
    space = opt('--space', ''); shape = [int(v) for v in opt('--shape', '').split(',')]; um = float(opt('--um', '0'))
    prefix = opt('--name', 'ufsm'); minp = int(opt('--min-pts', '2'))
    skip = {'--space', '--shape', '--um', '--name', '--min-pts'}
    ins = [x for i, x in enumerate(a[2:], 2) if x not in skip and a[i - 1] not in skip]
    if not space or len(shape) != 3 or um <= 0 or not ins:
        print(__doc__); return 2
    if os.path.exists(out):
        raise SystemExit(f'{out} exists')
    db = sqlite3.connect(out)
    db.executescript('''
PRAGMA application_id=1447249475; PRAGMA user_version=1; PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE fibers(id INTEGER PRIMARY KEY,name TEXT NOT NULL,family TEXT NOT NULL,
 point_count INTEGER NOT NULL CHECK(point_count>=2),length REAL NOT NULL,
 min_x REAL NOT NULL,max_x REAL NOT NULL,min_y REAL NOT NULL,max_y REAL NOT NULL,
 min_z REAL NOT NULL,max_z REAL NOT NULL,annotation TEXT NOT NULL);
CREATE TABLE blocks(id INTEGER PRIMARY KEY,fiber_id INTEGER NOT NULL REFERENCES fibers(id),
 first_segment INTEGER NOT NULL,points BLOB NOT NULL,UNIQUE(fiber_id,first_segment));
CREATE VIRTUAL TABLE block_bounds USING rtree(id,min_x,max_x,min_y,max_y,min_z,max_z);
''')
    fid = 0; bid = 0; tot_pts = 0; fams = {}
    db.execute('BEGIN')
    for path in ins:
        off, fam, pts = read_ufib(path)
        base = os.path.basename(path)
        for r in range(len(off) - 1):
            p = pts[off[r]:off[r + 1]].astype(np.float64)[:, ::-1]          # z y x -> x y z
            if len(p) < minp or len(p) < 2:
                continue
            keep = np.ones(len(p), bool); keep[1:] = np.any(p[1:] != p[:-1], axis=1)   # no zero-length segments
            p = np.ascontiguousarray(p[keep])
            if len(p) < 2:
                continue
            fid += 1
            f = chr(fam[r]) if chr(fam[r]) in 'HV' else '?'
            fams[f] = fams.get(f, 0) + 1
            L = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
            lo, hi = p.min(0), p.max(0)
            ann = {'type': 'vc3d_fiber', 'version': 1, 'source_file': base, 'source_line': r, 'length_voxels': L,
                   'length_mm': L * um / 1000.0, 'hv_classification': {'manual_tag': f}}
            db.execute('INSERT INTO fibers VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                       (fid, f'{prefix}_{fid}', f, len(p), L, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], json.dumps(ann)))
            for s in range(0, len(p) - 1, 256):
                q = p[s:s + 257]; bid += 1
                db.execute('INSERT INTO blocks VALUES(?,?,?,?)', (bid, fid, s, q.astype('<f8').tobytes()))
                ql, qh = q.min(0), q.max(0)
                db.execute('INSERT INTO block_bounds VALUES(?,?,?,?,?,?,?)', (bid, ql[0], qh[0], ql[1], qh[1], ql[2], qh[2]))
            tot_pts += len(p)
    db.execute('CREATE INDEX fibers_length on fibers(length desc,id)')
    frame = {'vc_open_data_coordinate_space': space, 'coordinate_base_shape_zyx': shape, 'vc_open_data_source_coordinate_level': 0,
             'vc_open_data_source_coordinate_scale_factor': 1, 'vc_open_data_source_original_resolution': um}
    root = dict(type='vc3d_fiber_collection', version=1, scale=1, **frame, coordinate_order='XYZ', coordinate_units='voxel',
                voxel_size_mm=um / 1000.0)
    meta = {'uuid': str(uuid.uuid4()), 'root': root, 'frame': frame, 'geometry': 'XYZ binary64 little-endian native L0',
            'block_segments': 256, 'source': 'ufsm fiber-trace (fibhv-d masks) -> scripts/wgraph/ufib2afv.py',
            'provenance': {'operation': 'ufib2afv', 'inputs': [os.path.abspath(x) for x in ins]},
            'complete': True, 'fiber_count': fid, 'point_count': tot_pts, 'block_count': bid}
    for k, v in meta.items():
        db.execute('INSERT INTO metadata VALUES(?,?)', (k, json.dumps(v)))
    db.commit(); db.close()
    print(json.dumps({'out': out, 'fibers': fid, 'points': tot_pts, 'blocks': bid, 'families': fams}))


if __name__ == '__main__':
    sys.exit(main())
