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

The 2023 slices (uint16 TIFF, 22 MB each, 11175) stream from dl.ash2txt into a sparse uint8 file (linear LO..HI -> 1..255);
slices behind the window are hole-punched, so the disk holds ~one window (~500 slices, 6 GB). Resumable (OUT/progress.txt).

usage: fill791.py OUTDIR      env: UFSM, TRANSFORM, L4RAW, RAWDIR, NS (128), DIL_MM (2), BAND_MM (0.5), ZSTART, ZEND (level-2 z range of slabs, testing)
"""
import concurrent.futures as cf
import ctypes, json, os, re, subprocess, sys, tempfile, time, urllib.request
import numpy as np
import tifffile
from scipy.ndimage import distance_transform_edt, minimum_filter

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
LO, HI = 8000.0, 56000.0
UM2, UM4 = 9.596, 38.384


def nfixed():
    req = urllib.request.Request(URL, headers={'User-Agent': 'generic-client'})
    names = set(re.findall(r'href="(\d{5})\.tif"', urllib.request.urlopen(req, timeout=60).read().decode()))
    n = len(names)
    assert names == {f'{i:05d}' for i in range(n)}, 'slice names are not 0..n-1'
    return n


class Fixed:
    """sparse uint8 copy of the 2023 stack, filled on demand by 6 download threads"""
    def __init__(self, n):
        self.n = n
        self.path = f'{RAWDIR}/fixed_u8.bin'
        size = n * FY * FX
        with open(self.path, 'ab') as f:
            if f.tell() != size: f.truncate(size)
        self.mm = np.memmap(self.path, np.uint8, 'r+', shape=(n, FY, FX))
        self.donef = f'{RAWDIR}/fixed_u8.done'
        self.done = set(int(v) for v in open(self.donef).read().split()) if os.path.exists(self.donef) else set()
        self.pool = cf.ThreadPoolExecutor(6)
        self.fut = {}
        self.libc = ctypes.CDLL('libc.so.6', use_errno=True)

    def _get(self, i):
        tif = f'{RAWDIR}/{i:05d}.tif'
        for attempt in range(6):
            r = subprocess.run(['curl', '-sf', '-A', 'generic-client', '--retry', '5', '-o', tif + '.part', f'{URL}{i:05d}.tif'])
            if r.returncode == 0: break
            time.sleep(10 * (attempt + 1))
        else:
            raise RuntimeError(f'download {i} failed')
        a = tifffile.imread(tif + '.part')
        assert a.shape == (FY, FX) and a.dtype == np.uint16, (i, a.shape, a.dtype)
        self.mm[i] = np.clip(np.rint((a.astype(np.float32) - LO) * (254.0 / (HI - LO))) + 1, 1, 255).astype(np.uint8)
        os.remove(tif + '.part')
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

    def drop_above(self, hi):
        """punch out slices >= hi (the window only moves to lower fixed z)"""
        gone = [i for i in self.done if i >= hi and i not in self.fut]
        if not gone: return
        self.mm.flush()
        fd = os.open(self.path, os.O_RDWR)
        for i in gone:
            if self.libc.fallocate(fd, 3, ctypes.c_longlong(i * FY * FX), ctypes.c_longlong(FY * FX)):   # KEEP_SIZE | PUNCH_HOLE
                raise OSError(ctypes.get_errno(), 'fallocate punch')
            self.done.discard(i)
        os.close(fd)


def read2(z0, nz, path):
    """level-2 planes into a raw file, returned as a writable memmap (the dev slice has ~2-4 GB free: nothing slab-sized in RAM)"""
    subprocess.run([UFSM, 'read', CTROOT, CTKEY + '/2', str(z0), '0', '0', str(nz), str(SH2[1]), str(SH2[2]), path],
                   check=True, stdout=subprocess.DEVNULL)
    return np.memmap(path, np.uint8, 'r+', shape=(nz, SH2[1], SH2[2]))


def sample(mm, fz, fy, fx, lo, hi):
    """trilinear samples of the sparse stack at fixed (z, y, x); 0 outside [lo, hi) or the slice. Grouped by the floor slice and read
    as one compact block per slice pair: a memmap gather (map_coordinates) faulted 4 K pages at random and re-read ~3x the window
    from disk once the dev slice squeezed the page cache (10+ min a slab)."""
    n = fz.size
    out = np.zeros(n, np.float32)
    ok = (fz >= lo) & (fz <= hi - 1) & (fz <= mm.shape[0] - 1) & (fy >= 0) & (fy <= FY - 1) & (fx >= 0) & (fx <= FX - 1)
    sel = np.flatnonzero(ok)
    if sel.size == 0: return out
    k = np.minimum(np.floor(fz[sel]).astype(np.int32), mm.shape[0] - 2)
    order = np.argsort(k, kind='stable'); sel = sel[order]; k = k[order]
    ks, starts = np.unique(k, return_index=True)
    ends = np.append(starts[1:], k.size)
    for kk, s0, s1 in zip(ks, starts, ends):
        i = sel[s0:s1]
        z = fz[i] - kk; y = fy[i]; x = fx[i]
        y0 = np.minimum(np.floor(y).astype(np.int32), FY - 2); x0 = np.minimum(np.floor(x).astype(np.int32), FX - 2)
        ya, yb, xa, xb = int(y0.min()), int(y0.max()) + 2, int(x0.min()), int(x0.max()) + 2
        blk = np.asarray(mm[kk:kk + 2, ya:yb, xa:xb]).astype(np.float32)
        ty = (y - y0).astype(np.float32); tx = (x - x0).astype(np.float32); tz = z.astype(np.float32)
        yi = y0 - ya; xi = x0 - xa
        def bil(pl):
            return ((pl[yi, xi] * (1 - tx) + pl[yi, xi + 1] * tx) * (1 - ty) + (pl[yi + 1, xi] * (1 - tx) + pl[yi + 1, xi + 1] * tx) * ty)
        a0 = bil(blk[0]); a1 = bil(blk[1])
        v = a0 * (1 - tz) + a1 * tz
        # a corner from a hole (0 = outside the scan) must not darken the sample: such points get 0 (left unfilled)
        bad = (blk[0][yi, xi] == 0) | (blk[1][yi, xi] == 0)
        out[i] = np.where(bad, 0, v)
    return out


def fz_range(M, z0, z1):
    """fixed z bounds over the l0 box of level-2 planes z0..z1"""
    c = np.array([[x, y, z, 1.0] for x in (0, 4 * SH2[2]) for y in (0, 4 * SH2[1]) for z in (4 * z0, 4 * z1)])
    fz = c @ M[2]
    return int(np.floor(fz.min())) - 2, int(np.ceil(fz.max())) + 3


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
    for si, Z in enumerate(slabs):
        nz = min(NS, SH2[0] - Z)
        lo, hi = fz_range(M, Z, Z + nz)
        if si + 1 < len(slabs):   # prefetch the next window while this slab computes
            nZ = slabs[si + 1]; fx.want(*fz_range(M, nZ, min(nZ + NS, SH2[0])))
        if Z in done: continue
        t0 = time.time()
        fx.wait(lo, hi)
        t1 = time.time()
        h0, h1 = max(Z - 2, 0), min(Z + nz + 2, SH2[0])
        ours = read2(h0, h1 - h0, f'{out}/slab_ours.raw')
        # dilated level-4 support, box in z
        z4a, z4b = max(Z // 4 - D4, 0), min((Z + nz - 1) // 4 + D4 + 1, SH4[0])
        m4 = np.asarray(l4[z4a:z4b]) > 0
        dil2 = np.stack([distance_transform_edt(~p) <= D4 if p.any() else np.zeros_like(p) for p in m4])
        ins2 = np.stack([distance_transform_edt(p) if p.any() else np.zeros(p.shape) for p in m4]).astype(np.float32)
        fs = np.memmap(f'{out}/slab_fs.raw', np.uint8, 'w+', shape=(nz, SH2[1], SH2[2]))
        bo, bf = [], []
        nF = 0
        yy, xx = np.mgrid[0:SH2[1], 0:SH2[2]].astype(np.int32)
        for p in range(nz):
            z = Z + p
            q4 = z // 4 - z4a
            dil = dil2[max(q4 - D4, 0):q4 + D4 + 1].any(0)
            dil = np.repeat(np.repeat(dil, 4, 0), 4, 1)
            ins = np.repeat(np.repeat(ins2[q4], 4, 0), 4, 1)
            k = z - h0
            nb = ours[max(k - 2, 0):k + 3] > 0
            ero = minimum_filter(nb.all(0), size=5, mode='constant', cval=0)
            F = dil & ~ero
            band = ero & (ins >= 1.5) & (ins <= B4)
            band &= (xx + yy + p) % 5 == 0       # a 1/5 sample of the band is plenty for 250 quantiles
            idx = np.flatnonzero(F | band)
            if idx.size == 0: continue
            py = yy.ravel()[idx] * 4 + 1.5; px = xx.ravel()[idx] * 4 + 1.5; pz = z * 4 + 1.5
            c = [M[2, 0] * px + M[2, 1] * py + M[2, 2] * pz + M[2, 3],
                 M[1, 0] * px + M[1, 1] * py + M[1, 2] * pz + M[1, 3],
                 M[0, 0] * px + M[0, 1] * py + M[0, 2] * pz + M[0, 3]]
            v = sample(fx.mm, c[0], c[1], c[2], lo, hi)
            v = np.clip(np.rint(v), 0, 255).astype(np.uint8)
            Ff = F.ravel()[idx]
            fp = np.zeros(SH2[1] * SH2[2], np.uint8); fp[idx[Ff]] = v[Ff]; fs[p] = fp.reshape(SH2[1], SH2[2])
            nF += int(Ff.sum())
            bsel = ~Ff & (v > 0)
            if bsel.any():
                bo.append(ours[k].ravel()[idx[bsel]]); bf.append(v[bsel])
        t2 = time.time()
        # per-slab quantile map 2023 -> ours on the band
        qs = np.linspace(0.002, 0.998, 250)
        lut = np.zeros(256, np.uint8)
        nb_ = 0
        if bo:
            bo = np.concatenate(bo); bf = np.concatenate(bf); nb_ = bo.size
        if nb_ >= 20000:
            qo, qf = np.quantile(bo, qs), np.quantile(bf, qs)
            qf, iu = np.unique(qf, return_index=True); qo = qo[iu]
            xs = np.arange(256)
            m = np.interp(xs, qf, qo)
            lo_s = (qo[1] - qo[0]) / max(qf[1] - qf[0], 1); hi_s = (qo[-1] - qo[-2]) / max(qf[-1] - qf[-2], 1)
            m = np.where(xs < qf[0], qo[0] + (xs - qf[0]) * lo_s, m)
            m = np.where(xs > qf[-1], qo[-1] + (xs - qf[-1]) * hi_s, m)
            lut[1:] = np.clip(np.rint(m[1:]), 1, 255)
            lutsrc = 'slab'
            np.save(f'{out}/lut_last.npy', lut)
        elif os.path.exists(f'{out}/lut_last.npy'):
            lut = np.load(f'{out}/lut_last.npy'); lutsrc = 'previous'
        else:   # no band yet (the bottom slabs): reg791's 70 boxes, mean quantiles
            q = np.load(REGQ); qo = q['ours'].mean(0); qf = (q['fixed'].mean(0) - LO) * (254.0 / (HI - LO)) + 1
            lut[1:] = np.clip(np.rint(np.interp(np.arange(1, 256), qf, qo)), 1, 255); lutsrc = 'reg791'
        body = ours[Z - h0:Z - h0 + nz]
        for p in range(nz):          # plane by plane (memmaps)
            f_ = np.asarray(fs[p]); sel = f_ > 0
            if sel.any(): b_ = np.array(body[p]); b_[sel] = lut[f_[sel]]; body[p] = b_
        body.flush()
        # band check after the map: median |ours - mapped 2023|
        bmed = float(np.median(np.abs(bo.astype(int) - lut[bf].astype(int)))) if nb_ else -1
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
        nxt = fz_range(M, slabs[si + 1], min(slabs[si + 1] + NS, SH2[0]))[1] if si + 1 < len(slabs) else 0
        fx.drop_above(max(nxt, 0))
        msg = (f'slab z {Z}..{Z + nz - 1}: fixed {lo}..{hi}  filled {nF / 1e6:.1f} Mvox  band {nb_ / 1e3:.0f} k ({lutsrc} lut, '
               f'median |ours - map| {bmed:.1f})  wait {t1 - t0:.0f} s  sample {t2 - t1:.0f} s  total {time.time() - t0:.0f} s')
        print(msg, flush=True); log.write(msg + '\n'); log.flush()
        del ours, fs, body
    print('fill791: all slabs done', flush=True)


if __name__ == '__main__':
    main()
