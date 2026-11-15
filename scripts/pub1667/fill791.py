#!/usr/bin/env python3
"""fill791.py: PHerc1667 2.399 um CT with the SAM2 mask's clipped margin filled from the unmasked 2023 7.91 um scan.

Level 2 (9.596 um), z slab by z slab (NS planes):
  ours  = the masked 2.399 store's level 2 (planes with a 2-plane halo, ufsm read)
  ero2  = (ours > 0) eroded by 2 level-2 voxels (the edge voxels are darkened by the masked mean-downsampling + codec ringing)
  dil   = the level-4 mask support dilated by DIL_MM (2-D EDT per level-4 plane, then a +-DIL box in z: >= DIL_MM in 3-D)
  F     = dil & ~ero2: filled from the 2023 scan, sampled trilinearly at the published affine (r2-transform.json, moving 2.399
          l0 xyz -> fixed 7.91 voxel xyz; residual measured 4.4 um median, reg791.py)
  band  = ero2 & inside the level-4 mask by <= BAND_MM: per-slab quantile map 2023 -> ours (the intensity match)
Values in F are >= 1 (0 stays "no data": outside the 2023 field of view or beyond the dilation). Levels 3-5 are 2x2x2 means
(round half up, edge-replicated odd ends) of the filled level 2, as the published pyramid. Written through `ufsm z3-put`
(volcomp q 2 at level 2, q 1 at 3-5 like the source; re-encoding the q-2 decoded CT at q 2 measured rms 0.07, max 3).

The 2023 slices (uint16 TIFF, 22 MB each, 11175) stream from dl.ash2txt into a sparse uint8 file (linear LO..HI -> 1..255) laid
out in 256 x 256 (y, x) bricks, slice-major inside each brick: the transform is a ~81 deg rotation (moving y -> fixed x), so a
moving-frame tile needs a narrow fixed x band over all fixed rows; in slice layout every such read touched whole slices (a plane-
by-plane gather re-read ~15 GB a slab, 24 min). Here a 512^2 x NS moving tile reads ~100 MB of contiguous brick runs into RAM.
Slices behind the window are hole-punched, so the disk holds ~one window (~500 slices, 6 GB). Resumable (OUT/progress.txt).
The dev slice is shared and near full: nothing slab-sized is held in RAM (memmaps on disk).

usage: fill791.py OUTDIR      env: UFSM, TRANSFORM, L4RAW, RAWDIR, NS (128), DIL_MM (2), BAND_MM (0.5), ZSTART, ZEND (testing)
"""
import concurrent.futures as cf
import ctypes, json, os, re, subprocess, sys, time, urllib.request
import numpy as np
import tifffile
from scipy.ndimage import distance_transform_edt, map_coordinates, minimum_filter

UFSM = os.environ.get('UFSM', '/home/forrest/ufsm-pub1667/build/ufsm')
CTROOT = '/vesuvius/ufsm/v2/ct'
CTKEY = 'PHerc1667/20251217075048-2.399um-0.2m-78keV-masked.zarr'
TRANSFORM = os.environ.get('TRANSFORM', '/vesuvius/ufsm/gt/pub1667-unwrap/transforms/r2-transform.json')
REGQ = os.environ.get('REGQ', '/vesuvius/ufsm/gt/pub1667-cmp/reg791/reg_q.npz')
L4RAW = os.environ.get('L4RAW', '/vesuvius/ufsm/gt/pub1667-cmp/ct_l4.raw')
RAWDIR = os.environ.get('RAWDIR', '/vesuvius/ufsm/gt/pub1667-2023raw')
URL = 'https://dl.ash2txt.org/full-scrolls/Scroll4/PHerc1667.volpkg/volumes/20231117161658/'
NS = int(os.environ.get('NS', 128))
DIL_MM = float(os.environ.get('DIL_MM', 2.0))
BAND_MM = float(os.environ.get('BAND_MM', 0.5))
SH2 = (9269, 3808, 3808)
SH4 = (2318, 952, 952)
FY, FX = 3340, 3440
BR = 256
NBY, NBX = (FY + BR - 1) // BR, (FX + BR - 1) // BR
TILE = 512
LO, HI = 8000.0, 56000.0
UM2, UM4 = 9.596, 38.384


def nfixed():
    req = urllib.request.Request(URL, headers={'User-Agent': 'generic-client'})
    names = set(re.findall(r'href="(\d{5})\.tif"', urllib.request.urlopen(req, timeout=60).read().decode()))
    n = len(names)
    assert names == {f'{i:05d}' for i in range(n)}, 'slice names are not 0..n-1'
    return n


class Fixed:
    """sparse uint8 copy of the 2023 stack in (by, bx, slice, 256, 256) bricks, filled on demand by 6 download threads"""
    def __init__(self, n):
        self.n = n
        self.path = f'{RAWDIR}/fixed_b256.bin'
        size = NBY * NBX * n * BR * BR
        with open(self.path, 'ab') as f:
            if f.tell() != size: f.truncate(size)
        self.mm = np.memmap(self.path, np.uint8, 'r+', shape=(NBY, NBX, n, BR, BR))
        self.donef = f'{RAWDIR}/fixed_b256.done'
        self.done = set(int(v) for v in open(self.donef).read().split()) if os.path.exists(self.donef) else set()
        self.pool = cf.ThreadPoolExecutor(6)
        self.fut = {}
        self.libc = ctypes.CDLL('libc.so.6', use_errno=True)

    def _get(self, i):
        tif = f'{RAWDIR}/{i:05d}.tif.part'
        for attempt in range(6):
            r = subprocess.run(['curl', '-sf', '-A', 'generic-client', '--retry', '5', '-o', tif, f'{URL}{i:05d}.tif'])
            if r.returncode == 0: break
            time.sleep(10 * (attempt + 1))
        else:
            raise RuntimeError(f'download {i} failed')
        a = tifffile.imread(tif)
        assert a.shape == (FY, FX) and a.dtype == np.uint16, (i, a.shape, a.dtype)
        u = np.zeros((NBY * BR, NBX * BR), np.uint8)
        u[:FY, :FX] = np.clip(np.rint((a.astype(np.float32) - LO) * (254.0 / (HI - LO))) + 1, 1, 255)
        for by in range(NBY):
            for bx in range(NBX):
                self.mm[by, bx, i] = u[by * BR:(by + 1) * BR, bx * BR:(bx + 1) * BR]
        os.remove(tif)
        return i

    def want(self, lo, hi):
        for i in range(max(lo, 0), min(hi, self.n)):
            if i not in self.done and i not in self.fut: self.fut[i] = self.pool.submit(self._get, i)

    def wait(self, lo, hi):
        self.want(lo, hi)
        for i in range(max(lo, 0), min(hi, self.n)):
            if i in self.fut:
                self.fut.pop(i).result(); self.done.add(i)
        self.mm.flush()
        with open(self.donef, 'w') as f: f.write(' '.join(map(str, sorted(self.done))))

    def block(self, z0, z1, y0, y1, x0, x1):
        """fixed [z0, z1) x [y0, y1) x [x0, x1) (clipped to the stack; outside = 0) as one in-RAM array"""
        out = np.zeros((z1 - z0, y1 - y0, x1 - x0), np.uint8)
        za, zb = max(z0, 0), min(z1, self.n)
        if za >= zb: return out
        for by in range(max(y0, 0) // BR, min(y1 - 1, FY - 1) // BR + 1):
            for bx in range(max(x0, 0) // BR, min(x1 - 1, FX - 1) // BR + 1):
                ya, yb = max(y0, by * BR), min(y1, (by + 1) * BR, FY)
                xa, xb = max(x0, bx * BR), min(x1, (bx + 1) * BR, FX)
                if ya >= yb or xa >= xb: continue
                out[za - z0:zb - z0, ya - y0:yb - y0, xa - x0:xb - x0] = self.mm[by, bx, za:zb, ya - by * BR:yb - by * BR,
                                                                                  xa - bx * BR:xb - bx * BR]
        return out

    def drop_above(self, hi):
        """punch out slices >= hi (the window only moves to lower fixed z): one contiguous run per brick"""
        gone = sorted(i for i in self.done if i >= hi and i not in self.fut)
        if not gone: return
        self.mm.flush()
        fd = os.open(self.path, os.O_RDWR)
        runs, s = [], gone[0]
        for a, b in zip(gone, gone[1:] + [None]):
            if b != a + 1: runs.append((s, a + 1)); s = b
        S = BR * BR
        for by in range(NBY):
            for bx in range(NBX):
                base = (by * NBX + bx) * self.n * S
                for r0, r1 in runs:
                    if self.libc.fallocate(fd, 3, ctypes.c_longlong(base + r0 * S), ctypes.c_longlong((r1 - r0) * S)):
                        raise OSError(ctypes.get_errno(), 'fallocate punch')
        os.close(fd)
        self.done -= set(gone)
        with open(self.donef, 'w') as f: f.write(' '.join(map(str, sorted(self.done))))


def read2(z0, nz, path):
    """level-2 planes into a raw file, returned as a writable memmap"""
    subprocess.run([UFSM, 'read', CTROOT, CTKEY + '/2', str(z0), '0', '0', str(nz), str(SH2[1]), str(SH2[2]), path],
                   check=True, stdout=subprocess.DEVNULL)
    return np.memmap(path, np.uint8, 'r+', shape=(nz, SH2[1], SH2[2]))


def fixed_box(M, z0, z1, y0, y1, x0, x1):
    """fixed (z, y, x) integer bounds of the l0 box of level-2 voxels [z0,z1) x [y0,y1) x [x0,x1)"""
    c = np.array([[x, y, z, 1.0] for x in (4 * x0, 4 * x1) for y in (4 * y0, 4 * y1) for z in (4 * z0, 4 * z1)])
    f = c @ M.T                       # columns: fixed x, y, z
    lo = np.floor(f.min(0)).astype(int) - 2; hi = np.ceil(f.max(0)).astype(int) + 3
    return (lo[2], hi[2]), (lo[1], hi[1]), (lo[0], hi[0])


def down2(a):
    """2x2x2 mean, round half up, odd ends edge-replicated"""
    a = a.astype(np.uint16)
    if a.shape[0] % 2: a = np.concatenate([a, a[-1:]], 0)
    if a.shape[1] % 2: a = np.concatenate([a, a[:, -1:]], 1)
    if a.shape[2] % 2: a = np.concatenate([a, a[:, :, -1:]], 2)
    s = a[0::2] + a[1::2]
    s = s[:, 0::2] + s[:, 1::2]
    s = s[:, :, 0::2] + s[:, :, 1::2]
    return ((s + 4) // 8).astype(np.uint8)


def make_lut(bo, bf, out):
    qs = np.linspace(0.002, 0.998, 250)
    lut = np.zeros(256, np.uint8)
    if bo is not None and bo.size >= 20000:
        qo, qf = np.quantile(bo, qs), np.quantile(bf, qs)
        qf, iu = np.unique(qf, return_index=True); qo = qo[iu]
        xs = np.arange(256)
        m = np.interp(xs, qf, qo)
        lo_s = (qo[1] - qo[0]) / max(qf[1] - qf[0], 1); hi_s = (qo[-1] - qo[-2]) / max(qf[-1] - qf[-2], 1)
        m = np.where(xs < qf[0], qo[0] + (xs - qf[0]) * lo_s, m)
        m = np.where(xs > qf[-1], qo[-1] + (xs - qf[-1]) * hi_s, m)
        lut[1:] = np.clip(np.rint(m[1:]), 1, 255)
        np.save(f'{out}/lut_last.npy', lut)
        return lut, 'slab'
    if os.path.exists(f'{out}/lut_last.npy'):
        return np.load(f'{out}/lut_last.npy'), 'previous'
    q = np.load(REGQ); qo = q['ours'].mean(0); qf = (q['fixed'].mean(0) - LO) * (254.0 / (HI - LO)) + 1   # reg791's 70 boxes
    lut[1:] = np.clip(np.rint(np.interp(np.arange(1, 256), qf, qo)), 1, 255)
    return lut, 'reg791'


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    M = np.array(json.load(open(TRANSFORM))['transformation_matrix'], float)
    l4 = np.memmap(L4RAW, np.uint8, 'r', shape=SH4)
    fx = Fixed(nfixed())
    prog = f'{out}/progress.txt'
    done = set(int(v) for v in open(prog).read().split()) if os.path.exists(prog) else set()
    shp = {3: (4635, 1904, 1904), 4: (2318, 952, 952), 5: (1159, 476, 476)}
    lraw = {}
    for L, s in shp.items():
        p = f'{out}/l{L}.raw'
        with open(p, 'ab') as f:
            if f.tell() != s[0] * s[1] * s[2]: f.truncate(s[0] * s[1] * s[2])
        lraw[L] = np.memmap(p, np.uint8, 'r+', shape=s)
    D4 = int(np.ceil(DIL_MM * 1000 / UM4))
    B4 = BAND_MM * 1000 / UM4
    slabs = [z for z in range(0, SH2[0], NS) if int(os.environ.get('ZSTART', 0)) <= z < int(os.environ.get('ZEND', 10 ** 9))]
    log = open(f'{out}/fill.log', 'a')
    yy, xx = np.mgrid[0:SH2[1], 0:SH2[2]].astype(np.int32)
    for si, Z in enumerate(slabs):
        nz = min(NS, SH2[0] - Z)
        (lo, hi), _, _ = fixed_box(M, Z, Z + nz, 0, SH2[1], 0, SH2[2])
        if si + 1 < len(slabs):   # prefetch the next window while this slab computes
            nZ = slabs[si + 1]; fx.want(*fixed_box(M, nZ, min(nZ + NS, SH2[0]), 0, SH2[1], 0, SH2[2])[0])
        if Z in done: continue
        t0 = time.time()
        fx.wait(lo, hi)
        t1 = time.time()
        h0, h1 = max(Z - 2, 0), min(Z + nz + 2, SH2[0])
        ours = read2(h0, h1 - h0, f'{out}/slab_ours.raw')
        # phase A, per plane: codes 1 = fill, 2 = band sample
        z4a, z4b = max(Z // 4 - D4, 0), min((Z + nz - 1) // 4 + D4 + 1, SH4[0])
        m4 = np.asarray(l4[z4a:z4b]) > 0
        dil2 = np.stack([distance_transform_edt(~p) <= D4 if p.any() else np.zeros_like(p) for p in m4])
        ins2 = np.stack([distance_transform_edt(p) if p.any() else np.zeros(p.shape) for p in m4]).astype(np.float32)
        code = np.memmap(f'{out}/slab_code.raw', np.uint8, 'w+', shape=(nz, SH2[1], SH2[2]))
        for p in range(nz):
            z = Z + p
            q4 = z // 4 - z4a
            dil = np.repeat(np.repeat(dil2[max(q4 - D4, 0):q4 + D4 + 1].any(0), 4, 0), 4, 1)
            ins = np.repeat(np.repeat(ins2[q4], 4, 0), 4, 1)
            k = z - h0
            ero = minimum_filter((np.asarray(ours[max(k - 2, 0):k + 3]) > 0).all(0), size=5, mode='constant', cval=0)
            band = ero & (ins >= 1.5) & (ins <= B4) & ((xx + yy + p) % 5 == 0)   # a 1/5 sample is plenty for 250 quantiles
            c = np.zeros(SH2[1:], np.uint8); c[band] = 2; c[dil & ~ero] = 1
            code[p] = c
        del dil2, ins2
        t2 = time.time()
        # phase B, per 512^2 tile: one in-RAM fixed block, trilinear samples of every coded voxel
        fs = np.memmap(f'{out}/slab_fs.raw', np.uint8, 'w+', shape=(nz, SH2[1], SH2[2]))
        bo, bf, nF, rd = [], [], 0, 0
        for ty in range(0, SH2[1], TILE):
            for tx in range(0, SH2[2], TILE):
                ct = np.asarray(code[:, ty:ty + TILE, tx:tx + TILE])
                if not ct.any(): continue
                (za, zb), (ya, yb), (xa, xb) = fixed_box(M, Z, Z + nz, ty, ty + ct.shape[1], tx, tx + ct.shape[2])
                za, zb = max(za, lo), min(zb, hi)
                blk = fx.block(za, zb, ya, yb, xa, xb); rd += blk.nbytes
                for p in range(nz):
                    idx = np.flatnonzero(ct[p])
                    if idx.size == 0: continue
                    iy, ix = np.divmod(idx, ct.shape[2])
                    py = (iy + ty) * 4 + 1.5; px = (ix + tx) * 4 + 1.5; pz = (Z + p) * 4 + 1.5
                    f_x = M[0, 0] * px + M[0, 1] * py + M[0, 2] * pz + M[0, 3]
                    f_y = M[1, 0] * px + M[1, 1] * py + M[1, 2] * pz + M[1, 3]
                    f_z = M[2, 0] * px + M[2, 1] * py + M[2, 2] * pz + M[2, 3]
                    ok = (f_z >= 0) & (f_z <= fx.n - 1) & (f_y >= 0) & (f_y <= FY - 1) & (f_x >= 0) & (f_x <= FX - 1)
                    v = map_coordinates(blk, [f_z - za, f_y - ya, f_x - xa], order=1, mode='nearest', prefilter=False, output=np.float32)
                    v = np.where(ok, np.clip(np.rint(v), 0, 255), 0).astype(np.uint8)
                    isF = ct[p].ravel()[idx] == 1
                    if isF.any():
                        t_ = np.asarray(fs[p, ty:ty + TILE, tx:tx + TILE]).copy().ravel()
                        t_[idx[isF]] = v[isF]
                        fs[p, ty:ty + TILE, tx:tx + TILE] = t_.reshape(ct.shape[1:])
                        nF += int(isF.sum())
                    bsel = ~isF & (v > 0)
                    if bsel.any():
                        k = Z + p - h0
                        bo.append(np.asarray(ours[k, ty:ty + TILE, tx:tx + TILE]).ravel()[idx[bsel]]); bf.append(v[bsel])
                del blk
        t3 = time.time()
        bo = np.concatenate(bo) if bo else None; bf = np.concatenate(bf) if bf else None
        lut, lutsrc = make_lut(bo, bf, out)
        nb_ = 0 if bo is None else bo.size
        bmed = float(np.median(np.abs(bo.astype(int) - lut[bf].astype(int)))) if nb_ else -1
        # phase C: apply, write level 2, accumulate levels 3-5
        body = ours[Z - h0:Z - h0 + nz]
        for p in range(nz):
            f_ = np.asarray(fs[p]); sel = f_ > 0
            if sel.any(): b_ = np.array(body[p]); b_[sel] = lut[f_[sel]]; body[p] = b_
        body.flush()
        bp = f'{out}/slab_body.raw'
        with open(bp, 'wb') as f:
            for p in range(nz): f.write(np.asarray(body[p]).tobytes())
        subprocess.run([UFSM, 'z3-put', f'{out}/l2.s128', bp, '--shape', ','.join(map(str, SH2)), '--z0', str(Z), '--nz',
                        str(nz), '--shard', '128', '--q', '2', '--threads', '8'], check=True, stdout=subprocess.DEVNULL)
        os.remove(bp)
        for g in range(0, nz, 8):    # 8-plane groups: 4 level-3, 2 level-4, 1 level-5 planes (Z is a multiple of 8)
            a3 = down2(np.asarray(body[g:g + 8])); a4 = down2(a3); a5 = down2(a4)
            z = Z + g
            lraw[3][z // 2:z // 2 + a3.shape[0]] = a3
            lraw[4][z // 4:z // 4 + a4.shape[0]] = a4
            lraw[5][z // 8:z // 8 + a5.shape[0]] = a5
        for L in lraw: lraw[L].flush()
        done.add(Z)
        with open(prog, 'w') as f: f.write(' '.join(map(str, sorted(done))))
        nxt = fixed_box(M, slabs[si + 1], min(slabs[si + 1] + NS, SH2[0]), 0, SH2[1], 0, SH2[2])[0][1] if si + 1 < len(slabs) else 0
        fx.drop_above(max(nxt, 0))
        msg = (f'slab z {Z}..{Z + nz - 1}: fixed {lo}..{hi}  filled {nF / 1e6:.1f} Mvox  band {nb_ / 1e3:.0f} k ({lutsrc} lut, '
               f'median |ours - map| {bmed:.1f})  wait {t1 - t0:.0f} s  masks {t2 - t1:.0f} s  sample {t3 - t2:.0f} s '
               f'(blocks {rd / 1e9:.1f} GB)  total {time.time() - t0:.0f} s')
        print(msg, flush=True); log.write(msg + '\n'); log.flush()
        del ours, fs, body, code
    print('fill791: all slabs done', flush=True)


if __name__ == '__main__':
    main()
