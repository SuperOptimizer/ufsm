/* ufsm recto-sp IN-FIELDS OUT-FIELDS --src ROOT KEY UM [--rows R0,R1] [--stride 4] [--half-um 400] [--half2-um 0] [--prom 25] [--sigma 0.5] [--sep-um 20]
                [--blend 1] [--gate 0.4,2.5] [--min-net-um 0] [--blend-ramp LO,HI,A0,A1] [--min-int 3] [--rsp-in DIR] [--threads N] [--cache DIR] [--gpu 0]
   Recto predictions as a count target for count-field: the sheet spacing sp of a fit-field set replaced by the spacing of recto hits
   along the field normal. Per field cell on a --stride grid (cells with material): the source profile along the principal axis of the
   cell's n n^T over +-half-um (trilinear, step half a source voxel), peaks as recto-ruler (rr_peaks: smoothing --sigma source voxels,
   prominence --prom 0..255, merged closer than --sep-um), the median normal width of the up to 5 hit intervals around the cell centre
   (>= --min-int intervals; --half2-um U2 > half: the profile spans +-U2 and its intervals, >= 2, are used where +-half has too few) = one turn each (--cls MODEL: ufsm recto-cls turns per interval, spacing = their widths / turns) -> rsp (units of h / 8 native voxels, as sp; 0 none), written to the stride block.
   OUT gets sp = net^(1 - a) rsp^a (a = --blend; --blend-ramp LO,HI,A0,A1: a = A0 where the network spacing <= LO um, A1 >= HI, linear between) where rsp / net lies inside --gate, net sp elsewhere; rsp (the raw recto spacing); the
   other planes linked to IN's; fields.json copied. --rsp-in DIR: reuse DIR/rsp (only the blend runs). --rows R0,R1: box rows computed (the
   others keep the net sp). The source is on the fields' scan (UM its voxel um; native = fields' um). Arrays: 512^3 shards, lossless.
   --gpu N (default 0; -1 the CPU): the profiles on device N (src/nn_rectosp.cu, the same codes byte for byte); the CPU path with --cls,
   without a device, or when the device lacks the memory for the source ring. */
#include "fit.h"
#include "optcheck.h"
#include "rectoruler.h"
#define RSG_WEAK __attribute__((weak))
#include "rectogpu.h"
/* predict's device chunk gathers (src/nn_ctdev.cu, nn.h); weak: a binary without the CUDA objects runs the CPU path */
void *nn_ctw_new(int dev) __attribute__((weak));
void nn_ctw_free(void *w) __attribute__((weak));
uint8_t *nn_ctw_host(void *w) __attribute__((weak));
uint8_t *nn_ctw_dev(void *w) __attribute__((weak));
int nn_ctw_put(void *w, uint32_t dm, int host, const int64_t co[3], const int64_t wo[3], const int64_t wn[3], const int64_t sh[3], uint8_t *dwin) __attribute__((weak));
#include "sysinfo.h"
#include "zarr3.h"
#include "z3w.h"
#include "store.h"
#include "json.h"
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <stdatomic.h>

static const char *rs_opt(int argc, char **argv, const char *k, const char *d) { for (int i = 4; i + 1 < argc; i++) if (!strcmp(argv[i], opt_n(k))) return argv[i + 1]; return d; }
static double rs_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static int rs_cmpf(const void *a, const void *b) { const float x = *(const float *)a, y = *(const float *)b; return (x > y) - (x < y); }

static int rs_axis(const uint8_t *const t[6], size_t i, double v[3]) {   /* principal eigenvector of n n^T (z y x) */
    double T[3][3]; const double a[6] = {t[0][i] / 255.0, t[1][i] / 255.0, t[2][i] / 255.0, t[3][i] / 255.0 - 0.5, t[4][i] / 255.0 - 0.5, t[5][i] / 255.0 - 0.5};
    T[0][0] = a[0]; T[1][1] = a[1]; T[2][2] = a[2]; T[0][1] = T[1][0] = a[3]; T[0][2] = T[2][0] = a[4]; T[1][2] = T[2][1] = a[5];
    int k = T[0][0] >= T[1][1] && T[0][0] >= T[2][2] ? 0 : T[1][1] >= T[2][2] ? 1 : 2; v[0] = v[1] = v[2] = 0; v[k] = 1;
    for (int it = 0; it < 12; it++) {
        double w[3] = {0, 0, 0}; for (int p = 0; p < 3; p++) for (int q = 0; q < 3; q++) w[p] += T[p][q] * v[q];
        const double l = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]); if (l < 1e-9) return -1; for (int p = 0; p < 3; p++) v[p] = w[p] / l;
    }
    return 0;
}

typedef struct {
    const fit_fields_buf *b; int r0, nr;   /* field rows [r0, r0 + nr) of the loaded block (box-local rows of b) */
    const uint8_t *src; int64_t so[3], sn[3]; double F;   /* source block (source voxels, global), F = native / source voxel */
    long long org[3]; int h, S; double um, half, half1, prom, sigma, sep; int minint;
    uint8_t *out; int64_t oz0; int H, W;   /* output rows (box rows from oz0), H x W cells */
    const rr_cls *cls;   /* --cls: turns per hit interval from the classifier (features as recto-ruler's rr_feat, along the normal) */
    _Atomic long long ncell, nest;
} rs_job;

static inline double rs_tri(const rs_job *J, double z, double y, double x) {   /* source voxel coords (global) */
    z -= J->so[0]; y -= J->so[1]; x -= J->so[2];
    const int64_t z0 = (int64_t)floor(z), y0 = (int64_t)floor(y), x0 = (int64_t)floor(x);
    if (z0 < 0 || y0 < 0 || x0 < 0 || z0 + 1 >= J->sn[0] || y0 + 1 >= J->sn[1] || x0 + 1 >= J->sn[2]) return NAN;
    const double tz = z - z0, ty = y - y0, tx = x - x0; const size_t sy = J->sn[2], sz = (size_t)J->sn[1] * J->sn[2];
    const uint8_t *q = J->src + z0 * sz + y0 * sy + x0;
    const double a = (1 - ty) * ((1 - tx) * q[0] + tx * q[1]) + ty * ((1 - tx) * q[sy] + tx * q[sy + 1]);
    const double c = (1 - ty) * ((1 - tx) * q[sz] + tx * q[sz + 1]) + ty * ((1 - tx) * q[sz + sy] + tx * q[sz + sy + 1]);
    return (1 - tz) * a + tz * c;
}

/* one row block's source box (native z [org + a0 h - half, org + a1 h + half], the box's y / x +- half), read by a helper thread */
typedef struct { z3 *z; int nt, rc; int64_t so[3], sn[3]; uint8_t *buf; char err[256]; } rs_src;
static void rs_src_box(rs_src *R, const long long org[3], int h, int a0, int a1, double hm, double F, int H, int W, const int64_t *ssh) {
    const double za0 = org[0] + (double)a0 * h; const double lo[3] = {za0 - hm, org[1] - hm, org[2] - hm}, hi[3] = {za0 + (a1 - a0) * h + hm, org[1] + (double)H * h + hm, org[2] + (double)W * h + hm};
    for (int d = 0; d < 3; d++) { int64_t s0 = (int64_t)floor(lo[d] * F) - 1, s1 = (int64_t)ceil(hi[d] * F) + 2; if (s0 < 0) s0 = 0; if (s1 > ssh[d]) s1 = ssh[d]; R->so[d] = s0; R->sn[d] = s1 - s0; }
}
static void *rs_src_main(void *a) {
    rs_src *R = a; R->rc = 0; R->buf = malloc((size_t)R->sn[0] * R->sn[1] * R->sn[2]);
    if (!R->buf) { R->rc = 1; snprintf(R->err, sizeof R->err, "out of memory"); return nullptr; }
    if (z3_read(R->z, R->so, R->sn, R->buf, R->nt)) { R->rc = 1; snprintf(R->err, sizeof R->err, "%s", z3_error()); }
    return nullptr;
}
static void rs_rows(void *ctx, int a0, int a1, int t) {
    rs_job *J = ctx; (void)t;
    const fit_fields *f = &J->b->f; const int H = J->H, W = J->W, S = J->S;
    const double st = 0.5 / J->F;   /* native step = half a source voxel */
    const int n = 2 * (int)ceil(J->half / (st * J->um)) + 1, c = n / 2;
    float *v = malloc(sizeof(float) * n), *tmp = malloc(sizeof(float) * n), *spn = malloc(sizeof(float) * n), *ph = malloc(sizeof(float) * n); uint8_t *mat = malloc(n); memset(mat, 1, n); rr_hit *hp = malloc(sizeof(rr_hit) * (n / 2 + 2));
    long long nc = 0, ne = 0;
    for (int job = a0; job < a1; job++) {   /* job = (row block, y block) */
        const int nyb = (H + S - 1) / S, zb = job / nyb, yb = job % nyb; const int z = J->r0 + zb * S + S / 2, y = yb * S + S / 2;
        if (z >= J->r0 + J->nr) continue;
        const int zz = z < J->b->shape[0] ? z : J->b->shape[0] - 1, yy = y < H ? y : H - 1;
        for (int xb = 0; xb * S < W; xb++) {
            const int xx = xb * S + S / 2 < W ? xb * S + S / 2 : W - 1; const size_t ci = ((size_t)zz * H + yy) * W + xx;
            if (!f->w[ci]) continue;
            double nv[3]; if (rs_axis(f->t, ci, nv)) continue;
            nc++;
            /* the cell centre, native global (z y x) -> source voxels */
            const double p[3] = {J->org[0] + (zz + 0.5) * J->h, J->org[1] + (yy + 0.5) * J->h, J->org[2] + (xx + 0.5) * J->h};
            int ok = 1;
            for (int i = 0; i < n && ok; i++) { const double s = (i - c) * st; v[i] = (float)rs_tri(J, (p[0] + s * nv[0] + 0.5) * J->F - 0.5, (p[1] + s * nv[1] + 0.5) * J->F - 0.5, (p[2] + s * nv[2] + 0.5) * J->F - 0.5); }
            const int nh = rr_peaks(v, mat, n, J->sigma * 2, J->prom, (int)lround(60 / (st * J->um)), J->sep / (st * J->um), hp, tmp);
            int j = -1; for (int k = 0; k + 1 < nh; k++) if (hp[k].pos <= c && hp[k + 1].pos > c) { j = k; break; }
            if (j < 0) continue;
            /* --half2-um: the intervals inside the primary window +-half when there are >= --min-int of them, else all (the wider
               window, >= 2) */
            const double inner = J->half1 > 0 ? J->half1 / (st * J->um) : 1e30; int inonly = 1, mi = J->minint;
            { int ni = 0; for (int k = j - 2; k <= j + 2; k++) if (k >= 0 && k + 1 < nh && hp[k].pos >= c - inner && hp[k + 1].pos <= c + inner) ni++;
              if (ni < J->minint && J->half1 > 0) { inonly = 0; mi = 2; } }
#define RS_IN(k) ((k) >= 0 && (k) + 1 < nh && (!inonly || (hp[k].pos >= c - inner && hp[(k) + 1].pos <= c + inner)))
            float wd[5]; int nw = 0; for (int k = j - 2; k <= j + 2; k++) if (RS_IN(k)) wd[nw++] = hp[k + 1].pos - hp[k].pos;
            if (nw < mi) continue;
            double sp;
            if (J->cls) {   /* the intervals' turns from the classifier: spacing = sum of widths / sum of turns */
                for (int i = 0; i < n; i++) {   /* the fields along the profile (nearest cell): spacing (native), phase */
                    const double s = (i - c) * st; const int cz = (int)floor((p[0] + s * nv[0] - J->org[0]) / J->h), cy = (int)floor((p[1] + s * nv[1] - J->org[1]) / J->h), cx = (int)floor((p[2] + s * nv[2] - J->org[2]) / J->h);
                    spn[i] = 0; ph[i] = NAN;
                    if (cz < 0 || cy < 0 || cx < 0 || cz >= J->b->shape[0] || cy >= H || cx >= W) continue;
                    const size_t q = ((size_t)cz * H + cy) * W + cx; if (!f->w[q]) continue;
                    spn[i] = f->sp[q] * J->h / 8.0f; const int pa = f->pc[q] - 128, pb = f->ps[q] - 128; if (pa * pa + pb * pb >= 256) ph[i] = (float)(atan2(pb, pa) / (2 * M_PI));
                }
                double sw = 0, se = 0; int nint = 0;
                for (int k = j - 2; k <= j + 2; k++) {
                    if (!RS_IN(k)) continue;
                    float fe[RR_NF]; for (int e = 0; e < RR_NF; e++) fe[e] = NAN;
                    const double a = hp[k].pos, b = hp[k + 1].pos; const int ia = (int)ceil(a), ib = (int)floor(b), im = (int)lround(0.5 * (a + b));
                    const double wu = (b - a) * st * J->um; fe[0] = (float)wu; if (spn[im] > 0) { fe[1] = (float)(wu / (spn[im] * J->um)); fe[20] = (float)(spn[im] * J->um); }
                    { float nb[9]; int nn = 0; for (int q = k - 4; q <= k + 4; q++) if (q >= 0 && q + 1 < nh && q != k) nb[nn++] = (hp[q + 1].pos - hp[q].pos) * st * J->um;
                      if (nn >= 2) { qsort(nb, nn, sizeof(float), rs_cmpf); fe[2] = (float)(wu / nb[nn / 2]); } }
                    { double sdn = 0, sph = 0, vmin = 1e9, vs = 0; int ndn = 0, nph = 0, lp = -1, nv2 = 0, cnt = 0;
                      for (int i = ia; i <= ib; i++) { cnt++; if (spn[i] > 0) { sdn += st / spn[i]; ndn++; }
                          if (ph[i] == ph[i]) { if (lp >= 0 && i - lp <= 3) { double d = ph[i] - ph[lp]; d -= floor(d + 0.5); sph += d; nph += i - lp; } lp = i; }
                          if (v[i] == v[i]) { if (v[i] < vmin) vmin = v[i]; vs += v[i]; nv2++; } }
                      if (cnt > 0 && ndn >= 0.8 * cnt) fe[3] = (float)(sdn * (b - a) / ndn);
                      if (cnt > 1 && nph >= 0.8 * (cnt - 1)) fe[4] = (float)(sph * (b - a) / nph);
                      if (nv2) { fe[5] = (float)(vmin / 255); fe[6] = (float)(vs / nv2 / 255); } }
                    fe[7] = hp[k].h / 255; fe[8] = hp[k + 1].h / 255; fe[9] = hp[k].prom / 255; fe[10] = hp[k + 1].prom / 255;
                    fe[11] = (float)(hp[k].wid * st * J->um); fe[12] = (float)(hp[k + 1].wid * st * J->um);
                    if (k > 0) { const int qm = (int)lround(0.5 * (hp[k - 1].pos + hp[k].pos)); if (spn[qm] > 0) fe[17] = (float)((hp[k].pos - hp[k - 1].pos) * st / spn[qm]); }
                    if (k + 2 < nh) { const int qm = (int)lround(0.5 * (hp[k + 1].pos + hp[k + 2].pos)); if (spn[qm] > 0) fe[18] = (float)((hp[k + 2].pos - hp[k + 1].pos) * st / spn[qm]); }
                    fe[21] = 1;
                    float pr[RR_K]; rr_cls_predict(J->cls, fe, pr); double E = 0; for (int e = 0; e < RR_K; e++) E += e * pr[e];
                    sw += b - a; se += E; nint++;
                }
                if (nint < mi || se < 0.5) continue;
                sp = sw / se * st * 8 / J->h;
            } else {
                qsort(wd, nw, sizeof(float), rs_cmpf);
                sp = wd[nw / 2] * st * 8 / J->h;   /* (native voxels -> units of h / 8) */
            }
            const uint8_t q = (uint8_t)(sp < 1 ? 1 : sp > 255 ? 255 : lround(sp));
            ne++;
            for (int a = zb * S; a < zb * S + S && a < J->nr; a++) for (int b = yb * S; b < yb * S + S && b < H; b++) memset(J->out + ((size_t)(a + J->r0 - J->oz0) * H + b) * W + xb * S, q, (size_t)(xb * S + S <= W ? S : W - xb * S));
        }
    }
    atomic_fetch_add(&J->ncell, nc); atomic_fetch_add(&J->nest, ne);
    free(v); free(tmp); free(mat); free(hp); free(spn); free(ph);
}

/* ---- the GPU path (src/nn_rectosp.cu; not with --cls): the CPU's row blocks, each block's profiles on the device ----
   A helper thread runs one block ahead: the field rows (loaded in array-chunk-aligned spans of up to 128 rows, kept for the next block),
   the block's cells (rs_rows' grid and material test) and the source slices the block adds to the device's z ring (decoded once: each
   read ends on the source's chunk rows, the next block starts where it ended), uploaded while the previous block's kernel runs (the
   ring is one block step deeper for that). The main thread runs the kernel and writes the codes into the shard row's rsp exactly as
   rs_rows does. */
int g_rectosp_gpu_blocks;   /* row blocks the last recto-sp ran on the GPU (tests) */
long long g_rectosp_dev_chunks, g_rectosp_ring_chunks;   /* source chunks it decoded on the GPU decoder / gathered into the ring on the device (-1: the host decode path) (tests) */
typedef struct { int sr, q0, q1, a0; int64_t so[3], sn[3], d0, d1; } rs_gblk;   /* shard row, rows, source box, ring slices to add */
typedef struct {
    const char *in; z3 *zs; int nt, H, W, S, shp0; long long org[3], zoff0;
    const rs_gblk *bl; int k;   /* the block this job prepares */
    fit_fields_buf fb; int f0, f1, fok;   /* the cached field rows [f0, f1) */
    uint8_t *stg; size_t stgcap; int32_t *zyx, *blk; uint8_t *t6; int M, Mcap; int rc; char err[256], errf[256]; int rcf; double tup, tdec, tfld;
    void **ctw; int nctw, ngpuw;   /* device decode (chunk workers: the first ngpuw on the GPU decoder, the rest on the CPU; nctw 0: z3_read + rsg_put) */
    _Atomic long long ndev, nhost, nskip;
} rs_gprep;
/* the block's new slices [z0, z1) from whole 128^3 chunks decoded into the device ring (chunk row by chunk row: the ring is a multiple of
   128 slices, so a chunk row's slots are contiguous) */
typedef struct { rs_gprep *R; const rs_gblk *B; int w; _Atomic long *next; long nch; int64_t cy0, cx0, ncy, ncx, cz0; _Atomic int *fail; } rs_cw;
static void *rs_cw_main(void *a) {
    rs_cw *C = a; rs_gprep *R = C->R; const rs_gblk *B = C->B; void *w = R->ctw[C->w]; uint8_t *hc = nn_ctw_host(w), *dc = C->w < R->ngpuw ? nn_ctw_dev(w) : nullptr;
    z3_thread_cpu_decode(!dc);
    const int64_t n128[3] = {128, 128, 128}, *sh = z3_meta_of(R->zs)->shape;
    for (long j; (j = atomic_fetch_add(C->next, 1)) < C->nch && !atomic_load(C->fail);) {
        const int64_t per = C->ncy * C->ncx, cz = C->cz0 + j / per, co[3] = {cz * 128, (C->cy0 + j / C->ncx % C->ncy) * 128, (C->cx0 + j % C->ncx) * 128};
        uint32_t dm = 0;
        if (z3_read_sub_dev(R->zs, co, n128, hc, 1, 0, dc, &dm)) { atomic_store(C->fail, 1); break; }
        const int host = dm != 0xffffffffu;
        if (host && !dm) { const uint64_t *q = (const uint64_t *)hc; uint64_t any = 0; for (size_t i = 0; i < ((size_t)1 << 21) / 8 && !any; i++) any |= q[i];
                           if (!any) { atomic_fetch_add(&R->nskip, 1); continue; } }   /* (all fill: the slices are zeroed) */
        atomic_fetch_add(host ? &R->nhost : &R->ndev, 1);
        const int64_t zl = co[0] > B->d0 ? co[0] : B->d0, zh = co[0] + 128 < B->d1 ? co[0] + 128 : B->d1;
        const int64_t wo[3] = {zl, B->so[1], B->so[2]}, wn[3] = {zh - zl, B->sn[1], B->sn[2]};
        if (nn_ctw_put(w, dm, host, co, wo, wn, sh, rsg_slot(zl))) { atomic_store(C->fail, 1); break; }
    }
    z3_thread_cpu_decode(0);
    return nullptr;
}
static void *rs_gfld_main(void *a) {   /* the block's field rows and cells (beside its source, on a thread of their own) */
    rs_gprep *R = a; const rs_gblk *B = &R->bl[R->k]; R->rcf = 0; R->M = 0; const double tf0 = rs_now();
    const int a0 = B->a0, a1 = B->q1, H = R->H, W = R->W, S = R->S;
    if (!R->fok || a0 < R->f0 || a1 > R->f1) {   /* the field rows: from a0 to the array chunk row's end past a1 (the next block's rows) */
        if (R->fok) { fit_fields_free(&R->fb); R->fok = 0; }
        long long e = ((R->zoff0 + a1 + 127) / 128) * 128 - R->zoff0; if (e < a1) e = a1; if (e > R->shp0) e = R->shp0;
        fit_fields_skip_planes(1 | 2 | 8); fit_fields_ct_planes(0);
        if (fit_fields_load(R->in, a0, (int)e, 1, &R->fb)) { R->rcf = 1; snprintf(R->errf, sizeof R->errf, "cannot load field rows %d..%lld", a0, e); return nullptr; }
        R->f0 = a0; R->f1 = (int)e; R->fok = 1;
    }
    const fit_fields *f = &R->fb.f; const int nyb = (H + S - 1) / S, nzb = (a1 - a0 + S - 1) / S, nxb = (W + S - 1) / S;
    const size_t cap = (size_t)nzb * nyb * nxb;
    if ((int)cap > R->Mcap) { free(R->zyx); free(R->blk); free(R->t6); R->zyx = malloc(cap * 3 * sizeof(int32_t)); R->blk = malloc(cap * 3 * sizeof(int32_t)); R->t6 = malloc(cap * 6); R->Mcap = (int)cap;
                              if (!R->zyx || !R->blk || !R->t6) { R->rcf = 1; snprintf(R->errf, sizeof R->errf, "out of memory"); return nullptr; } }
    int M = 0;
    for (int zb = 0; zb < nzb; zb++) {   /* (rs_rows' cells: J->r0 = 0, J->nr = a1 - a0) */
        const int z = zb * S + S / 2; if (z >= a1 - a0) continue;
        for (int yb = 0; yb < nyb; yb++) { const int y = yb * S + S / 2, yy = y < H ? y : H - 1;
            for (int xb = 0; xb < nxb; xb++) {
                const int xx = xb * S + S / 2 < W ? xb * S + S / 2 : W - 1; const size_t ci = ((size_t)(a0 + z - R->f0) * H + yy) * W + xx;
                if (!f->w[ci]) continue;
                R->zyx[3 * M] = z; R->zyx[3 * M + 1] = yy; R->zyx[3 * M + 2] = xx; R->blk[3 * M] = zb; R->blk[3 * M + 1] = yb; R->blk[3 * M + 2] = xb;
                for (int t = 0; t < 6; t++) R->t6[(size_t)M * 6 + t] = f->t[t][ci];
                M++;
            } }
    }
    R->M = M;
    R->tfld += rs_now() - tf0;
    return nullptr;
}
static void rs_gsrc(rs_gprep *R, const rs_gblk *B) {   /* the slices the block adds to the ring */
    if (B->d1 > B->d0 && R->nctw) {   /* the slices this block adds to the ring, decoded on the device */
        const double td = rs_now();
        if (rsg_zero(B->d0, B->d1 - B->d0)) { R->rc = 1; snprintf(R->err, sizeof R->err, "GPU memset failed"); return; }
        _Atomic long next = 0; _Atomic int fail = 0; const int64_t cz0 = B->d0 / 128, cz1 = (B->d1 + 127) / 128, cy0 = B->so[1] / 128, cy1 = (B->so[1] + B->sn[1] + 127) / 128, cx0 = B->so[2] / 128, cx1 = (B->so[2] + B->sn[2] + 127) / 128;
        rs_cw C[64]; pthread_t th[64]; int up[64] = {0}; const int nw = R->nctw < 64 ? R->nctw : 64;
        for (int t = 0; t < nw; t++) { C[t] = (rs_cw){.R = R, .B = B, .w = t, .next = &next, .nch = (cz1 - cz0) * (cy1 - cy0) * (cx1 - cx0), .cy0 = cy0, .cx0 = cx0, .ncy = cy1 - cy0, .ncx = cx1 - cx0, .cz0 = cz0, .fail = &fail};
                                       up[t] = pthread_create(&th[t], nullptr, rs_cw_main, &C[t]) == 0; if (!up[t]) rs_cw_main(&C[t]); }
        for (int t = 0; t < nw; t++) if (up[t]) pthread_join(th[t], nullptr);
        if (fail) { R->rc = 1; snprintf(R->err, sizeof R->err, "device chunk decode: %s", z3_error()); return; }
        R->tdec += rs_now() - td;
    } else if (B->d1 > B->d0) {   /* the slices this block adds to the ring */
        const double td = rs_now();
        const size_t need = (size_t)(B->d1 - B->d0) * B->sn[1] * B->sn[2];
        if (need > R->stgcap) { free(R->stg); R->stg = malloc(need); R->stgcap = R->stg ? need : 0; if (!R->stg) { R->rc = 1; snprintf(R->err, sizeof R->err, "out of memory"); return; } }
        const int64_t o[3] = {B->d0, B->so[1], B->so[2]}, nn[3] = {B->d1 - B->d0, B->sn[1], B->sn[2]};
        if (z3_read(R->zs, o, nn, R->stg, R->nt)) { R->rc = 1; snprintf(R->err, sizeof R->err, "%s", z3_error()); return; }
        const double tu = rs_now(); R->tdec += tu - td;
        if (rsg_put(B->d0, B->d1 - B->d0, R->stg)) { R->rc = 1; snprintf(R->err, sizeof R->err, "GPU upload failed"); return; }
        R->tup += rs_now() - tu;
    }
}
static void *rs_gprep_main(void *a) {
    rs_gprep *R = a; const rs_gblk *B = &R->bl[R->k]; R->rc = 0;
    pthread_t fth; const int fup = !pthread_create(&fth, nullptr, rs_gfld_main, R); if (!fup) rs_gfld_main(R);
    rs_gsrc(R, B);
    if (fup) pthread_join(fth, nullptr);
    if (!R->rc && R->rcf) { R->rc = 1; memcpy(R->err, R->errf, sizeof R->err); }
    return nullptr;
}

typedef struct { const rs_gprep *R; const uint8_t *q; int a0, a1, q0, q1, b0, H, W, S; uint8_t *rsp; } rs_sc;
static void rs_scatter(void *a, int c0, int c1, int t) {   /* the cells' codes into their stride blocks (cells write disjoint blocks) */
    const rs_sc *C = a; (void)t; const int M = C->R->M, m0 = (int)((long)M * c0 / 64), m1 = (int)((long)M * c1 / 64), S = C->S, H = C->H, W = C->W;
    for (int m = m0; m < m1; m++) {
        if (!C->q[m]) continue;
        const int32_t *bk = C->R->blk + 3 * m; const int zb = bk[0], yb = bk[1], xb = bk[2]; const size_t xl = (size_t)(xb * S + S <= W ? S : W - xb * S);
        for (int a2 = zb * S; a2 < zb * S + S && a2 < C->a1 - C->a0; a2++) { const int r = C->a0 + a2; if (r < C->q0 || r >= C->q1) continue;
            for (int b = yb * S; b < yb * S + S && b < H; b++) memset(C->rsp + ((size_t)(r - C->b0) * H + b) * W + xb * S, C->q[m], xl); }
    }
}
/* one array's shards of the row: pass 0 sp (outside the box: the input sp, read only where the box leaves part of the shard's array
   cells uncovered), pass 1 rsp (shards with box cells only) */
typedef struct rs_fin rs_fin;
typedef struct { rs_fin *F; int pass, rc; uint8_t *shard; } rs_wr;
static void *rs_wr_main(void *arg);
/* one shard row's blend into net and its sp (+ rsp) shard writes (outside the box: the input sp), freeing net and rsp */
struct rs_fin {
    uint8_t *net, *rsp; int nr, b0, b1, H, W, SH, nt, h; int64_t gz0, ash[3]; long long zoff[3]; z3 *zsp; z3w *wsp, *wrs; uint8_t *shard, *shard2;
    double glo, ghi, minnet, blend, rlo, rhi, ra0, ra1, um; int ramp;
    long long nmat, nrep; double tn, tb, tw; int rc;
};
typedef struct { const uint8_t (*lut)[256]; uint8_t *net; const uint8_t *rsp; size_t n; _Atomic long long nmat, nrep; } rs_bl;
static void rs_blend_rows(void *a, int r0, int r1, int t) {
    rs_bl *B = a; (void)t; long long nm = 0, nr = 0; const size_t per = (B->n + 255) / 256, i0 = (size_t)r0 * per, i1 = (size_t)r1 * per < B->n ? (size_t)r1 * per : B->n;
    for (size_t i = i0; i < i1; i++) {
        if (!B->net[i]) continue;
        nm++;
        const uint8_t o = B->lut[B->net[i]][B->rsp[i]]; if (!o) continue;
        B->net[i] = o; nr++;
    }
    atomic_fetch_add(&B->nmat, nm); atomic_fetch_add(&B->nrep, nr);
}
static void *rs_fin_main(void *arg) {
    rs_fin *F = arg; F->rc = 0; F->nmat = F->nrep = 0; const double t0 = rs_now();
    const int H = F->H, W = F->W, SH = F->SH, b0 = F->b0, b1 = F->b1; const int64_t gz0 = F->gz0, *ash = F->ash; const long long *zoff = F->zoff;
    const size_t plane = (size_t)H * W; uint8_t *net = F->net, *rsp = F->rsp; const int nt = F->nt;
    {   const int64_t o[3] = {zoff[0] + b0, zoff[1], zoff[2]}, nn[3] = {F->nr, H, W};   /* the network sp of the rows */
        if (z3_read(F->zsp, o, nn, net, nt)) { fprintf(stderr, "recto-sp: %s\n", z3_error()); F->rc = 1; return nullptr; } }
    const double t1 = rs_now(); F->tn = t1 - t0;
    /* blend into net */
    /* the blended code depends on (net, rsp) alone: a 256 x 256 table of the very values the per-cell rule gives (0: kept). The
       per-cell exp / log over every material cell of a shard row was 43 of 207 s single-threaded on 0332 (2026-11-16) */
    static uint8_t lut[256][256]; static int lut_ok;   /* mgpu: env (a constant table built on first use; every builder writes the same bytes) */
    if (!lut_ok) {
        const double glo = F->glo, ghi = F->ghi, minnet = F->minnet, blend = F->blend, rlo = F->rlo, rhi = F->rhi, ra0 = F->ra0, ra1 = F->ra1, um = F->um; const int h = F->h, ramp = F->ramp;
        for (int n8 = 1; n8 < 256; n8++) for (int r8 = 0; r8 < 256; r8++) {
            uint8_t o = 0;
            if (r8) {
                const double r = (double)r8 / n8;
                if (!(r < glo || r > ghi) && !(n8 * h / 8.0 * um < minnet)) {   /* --min-net-um: only where the network spacing is at least this wide */
                    double a = blend;
                    if (ramp) { const double su = n8 * h / 8.0 * um; a = su <= rlo ? ra0 : su >= rhi ? ra1 : ra0 + (ra1 - ra0) * (su - rlo) / (rhi - rlo); }   /* --blend-ramp: by the network spacing */
                    const double v = exp((1 - a) * log(n8) + a * log(r8)); o = (uint8_t)(v < 1 ? 1 : v > 255 ? 255 : lround(v));
                }
            }
            lut[n8][r8] = o;
        }
        lut_ok = 1;
    }
    {   rs_bl B = {.lut = (const uint8_t (*)[256])lut, .net = net, .rsp = rsp, .n = plane * F->nr};   /* (threaded: 9 of 131 s single on 0332, 2026-11-17) */
        fit_par(256, rs_blend_rows, &B); F->nmat = B.nmat; F->nrep = B.nrep; }
    const double tw = rs_now(); F->tb = tw - t1;
    /* write the shard row: sp here, rsp (when written) on a second thread at the same time (separate arrays) */
    rs_wr P1 = {.F = F, .pass = 1, .shard = F->shard2}; pthread_t th; int up = F->wrs && !pthread_create(&th, nullptr, rs_wr_main, &P1);
    rs_wr P0 = {.F = F, .pass = 0, .shard = F->shard}; rs_wr_main(&P0);
    if (F->wrs && !up) rs_wr_main(&P1);
    if (up) pthread_join(th, nullptr);
    if (P0.rc || P1.rc) { F->rc = 1; return nullptr; }
    free(net); free(rsp); F->tw = rs_now() - tw;
    return nullptr;
}

static void *rs_wr_main(void *arg) {
    rs_wr *A = arg; rs_fin *F = A->F; const int pass = A->pass; A->rc = 0; uint8_t *shard = A->shard;
    const int H = F->H, W = F->W, SH = F->SH, b0 = F->b0, b1 = F->b1, nt = F->nt; const int64_t gz0 = F->gz0, *ash = F->ash; const long long *zoff = F->zoff;
    {
        z3w *w = pass ? F->wrs : F->wsp; const uint8_t *src = pass ? F->rsp : F->net;
        for (int64_t sy = 0; sy * SH < ash[1]; sy++) for (int64_t sx = 0; sx * SH < ash[2]; sx++) {
            memset(shard, 0, (size_t)SH * SH * SH);
            const int64_t o[3] = {gz0, sy * SH, sx * SH}; int64_t nn[3] = {SH, SH, SH}; for (int d = 0; d < 3; d++) if (o[d] + nn[d] > ash[d]) nn[d] = ash[d] - o[d];
            const int covered = o[0] >= zoff[0] + b0 && o[0] + nn[0] <= zoff[0] + b1 && o[1] >= zoff[1] && o[1] + nn[1] <= zoff[1] + H && o[2] >= zoff[2] && o[2] + nn[2] <= zoff[2] + W;
            if (!pass && !covered) {
                uint8_t *tb = malloc((size_t)nn[0] * nn[1] * nn[2]); if (z3_read(F->zsp, o, nn, tb, nt)) { fprintf(stderr, "recto-sp: %s\n", z3_error()); free(tb); A->rc = 1; return nullptr; }
                for (int64_t a = 0; a < nn[0]; a++) for (int64_t b = 0; b < nn[1]; b++) memcpy(shard + ((size_t)a * SH + b) * SH, tb + ((size_t)a * nn[1] + b) * nn[2], nn[2]);
                free(tb); }
            int any = 0;
            for (int a = 0; a < SH; a++) { const int64_t gz = gz0 + a, br = gz - zoff[0]; if (br < b0 || br >= b1) continue;
                for (int b = 0; b < SH; b++) { const int64_t gy = sy * SH + b, by = gy - zoff[1]; if (by < 0 || by >= H) continue;
                    const int64_t gx0 = sx * SH, x0 = gx0 - zoff[2] < 0 ? zoff[2] - gx0 : 0, x1 = gx0 + SH - zoff[2] > W ? W + zoff[2] - gx0 : SH; if (x1 <= x0) continue;
                    memcpy(shard + ((size_t)a * SH + b) * SH + x0, src + ((size_t)(br - b0) * H + by) * W + (gx0 + x0 - zoff[2]), x1 - x0); any = 1; } }
            if (!any && pass) continue;
            if (z3w_write_shard(w, gz0 / SH, sy, sx, shard, nt)) { fprintf(stderr, "recto-sp: write: %s\n", z3w_error()); A->rc = 1; return nullptr; }
        }
    }
    return nullptr;
}

static int rs_link(const char *in, const char *out, const char *name) {
    char a[2048], b[2048], r[4096]; snprintf(a, sizeof a, "%s/%s", in, name); snprintf(b, sizeof b, "%s/%s", out, name);
    if (access(a, F_OK)) return 0;
    if (!realpath(a, r)) return -1;
    unlink(b); return symlink(r, b);
}

int cmd_recto_sp(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm recto-sp IN-FIELDS OUT-FIELDS --src ROOT KEY UM [--rows R0,R1] [--stride 4] [--half-um 400] [--prom 25] [--sigma 0.5] [--sep-um 20]\n"
                                    "       [--blend 1] [--gate 0.4,2.5] [--min-int 3] [--rsp-in DIR] [--threads N] [--cache DIR] [--gpu 0 (-1 CPU)]\n"); return 2; }
    const char *in = argv[2], *out = argv[3]; const int nt = ufsm_threads(opt_atoi("--threads", rs_opt(argc, argv, "--threads", "0")), 16);
    fit_set_threads(nt);
    char path[4096]; snprintf(path, sizeof path, "%s/fields.json", in);
    FILE *fj = fopen(path, "rb"); if (!fj) { fprintf(stderr, "recto-sp: cannot read %s\n", path); return 1; }
    fseek(fj, 0, SEEK_END); long L = ftell(fj); fseek(fj, 0, SEEK_SET); char *jt = malloc(L + 1); if (fread(jt, 1, L, fj) != (size_t)L) { fclose(fj); return 1; } jt[L] = 0; fclose(fj);
    json *j = json_parse(jt, L); if (!j || strcmp(json_str(json_get(j, "format"), ""), "zarr")) { fprintf(stderr, "recto-sp: zarr fields only\n"); return 1; }
    long long org[3], zoff[3]; int shp[3]; for (int d = 0; d < 3; d++) { org[d] = (long long)json_num(json_at(json_get(j, "origin"), d), 0); shp[d] = (int)json_num(json_at(json_get(j, "shape"), d), 0); zoff[d] = (long long)json_num(json_at(json_get(j, "zarr_offset"), d), 0); }
    const int h = (int)json_num(json_get(j, "h"), 2); const double um = json_num(json_get(j, "um"), 9.6); json_free(j);
    mkdir(out, 0775);
    snprintf(path, sizeof path, "%s/fields.json", out); { FILE *o = fopen(path, "wb"); if (!o || fwrite(jt, 1, L, o) != (size_t)L) { fprintf(stderr, "recto-sp: cannot write %s\n", path); return 1; } fclose(o); } free(jt);
    const char *pl[] = {"pc", "ps", "t0", "t1", "t2", "t3", "t4", "t5", "w", "dz", "dy", "dx", "dq", "r", "ct", "cs"};
    for (size_t k = 0; k < sizeof pl / sizeof *pl; k++) if (rs_link(in, out, pl[k])) { fprintf(stderr, "recto-sp: cannot link %s\n", pl[k]); return 1; }
    z3 *zsp = z3_open(store_open(in), "sp", nullptr); if (!zsp) { fprintf(stderr, "recto-sp: %s\n", z3_error()); return 1; }
    const int64_t *ash = z3_meta_of(zsp)->shape;
    int R0 = 0, R1 = shp[0]; { const char *r = rs_opt(argc, argv, "--rows", nullptr); if (r && sscanf(r, "%d,%d", &R0, &R1) != 2) { fprintf(stderr, "recto-sp: --rows R0,R1\n"); return 2; } if (R0 < 0) R0 = 0; if (R1 > shp[0]) R1 = shp[0]; }
    double blend = opt_atof("--blend", rs_opt(argc, argv, "--blend", "1")), glo = 0.4, ghi = 2.5; sscanf(rs_opt(argc, argv, "--gate", "0.4,2.5"), "%lf,%lf", &glo, &ghi);
    const char *rspin = rs_opt(argc, argv, "--rsp-in", nullptr); const double minnet = opt_atof("--min-net-um", rs_opt(argc, argv, "--min-net-um", "0"));
    double rlo = 0, rhi = 0, ra0 = 0, ra1 = 0; const int ramp = sscanf(rs_opt(argc, argv, "--blend-ramp", ""), "%lf,%lf,%lf,%lf", &rlo, &rhi, &ra0, &ra1) == 4 && rhi > rlo;
    z3 *zrin = nullptr; if (rspin && !(zrin = z3_open(store_open(rspin), "rsp", nullptr))) { fprintf(stderr, "recto-sp: --rsp-in: %s\n", z3_error()); return 1; }
    z3 *zs = nullptr; double sum = 0, F = 1; const char *cache = rs_opt(argc, argv, "--cache", nullptr);
    if (!zrin) { for (int i = 4; i + 3 < argc; i++) if (!strcmp(argv[i], opt_n("--src"))) { if (!(zs = z3_open(store_open(argv[i + 1]), argv[i + 2], cache))) { fprintf(stderr, "recto-sp: --src: %s\n", z3_error()); return 1; } sum = atof(argv[i + 3]); }
                 if (!zs || !(sum > 0)) { fprintf(stderr, "recto-sp: --src ROOT KEY UM (or --rsp-in DIR)\n"); return 2; } F = um / sum; }
    rr_cls *cls = nullptr; { const char *cp = rs_opt(argc, argv, "--cls", nullptr); if (cp && !(cls = rr_cls_load(cp))) { fprintf(stderr, "recto-sp: cannot load --cls %s\n", cp); return 1; } }
    const int S = opt_atoi("--stride", rs_opt(argc, argv, "--stride", "4")); const double half = opt_atof("--half-um", rs_opt(argc, argv, "--half-um", "400")), half2 = opt_atof("--half2-um", rs_opt(argc, argv, "--half2-um", "0"));
    const int SH = 512;
    char attrs[256]; snprintf(attrs, sizeof attrs, "{\"ufsm\":{\"content\":\"winding fitter fields (src/fit.h)\",\"h\":%d,\"recto_sp\":1}}", h);
    snprintf(path, sizeof path, "%s/sp", out); char sl[4096]; snprintf(sl, sizeof sl, "%s/sp", out); unlink(sl);
    z3w *wsp = z3w_create(path, ash, SH, 0, 0, attrs); if (!wsp) { fprintf(stderr, "recto-sp: %s\n", z3w_error()); return 1; }
    z3w *wrs = nullptr; if (!zrin) { snprintf(path, sizeof path, "%s/rsp", out); if (!(wrs = z3w_create(path, ash, SH, 0, 0, attrs))) { fprintf(stderr, "recto-sp: %s\n", z3w_error()); return 1; } }
    const int H = shp[1], W = shp[2]; const double t0 = rs_now(); long long tc = 0, te = 0, nrep = 0, nmat = 0;
    uint8_t *shard = malloc((size_t)SH * SH * SH), *shard2 = malloc((size_t)SH * SH * SH);
    const double prom = opt_atof("--prom", rs_opt(argc, argv, "--prom", "25")), sigma = opt_atof("--sigma", rs_opt(argc, argv, "--sigma", "0.5")), sepum = opt_atof("--sep-um", rs_opt(argc, argv, "--sep-um", "20"));
    const int minint = opt_atoi("--min-int", rs_opt(argc, argv, "--min-int", "3"));
    /* --gpu N (default 0; -1 the CPU): the profiles on device N (not with --cls). The plan: every row block of the run (the CPU's blocks),
       each one's source box and the ring slices it adds */
    const int gdev = opt_atoi("--gpu", rs_opt(argc, argv, "--gpu", "0")); g_rectosp_gpu_blocks = 0; g_rectosp_dev_chunks = g_rectosp_ring_chunks = 0;
    rs_gblk *gbl = nullptr; int ngb = 0, gon = 0; rs_gprep GP[2] = {0}; pthread_t gth; int gpend = 0, gnext = 0; double gtk = 0, gtp = 0, gtu = 0, gtr = 0, gtn = 0, gtb = 0, gtw = 0;
    if (zs && !cls && gdev >= 0 && rsg_open && rsg_free_mem(gdev) > 0) {
        const double hm = fmax(half, half2) / um + 2; const int64_t *ssh = z3_meta_of(zs)->shape; const int cz = z3_meta_of(zs)->chunk[0] > 0 ? z3_meta_of(zs)->chunk[0] : 128;
        int cap = 64; gbl = malloc(sizeof *gbl * cap); int64_t bz1 = -1, RZ = 0; int sr = 0;
        for (int64_t gz0 = (zoff[0] / SH) * SH; gz0 < zoff[0] + shp[0]; gz0 += SH, sr++) {
            const int b0 = (int)(gz0 - zoff[0] < 0 ? 0 : gz0 - zoff[0]), b1 = (int)(gz0 + SH - zoff[0] > shp[0] ? shp[0] : gz0 + SH - zoff[0]);
            for (int c0 = b0; c0 < b1; c0 += 64) {
                const int c1 = c0 + 64 < b1 ? c0 + 64 : b1; const int q0 = c0 > R0 ? c0 : R0, q1 = c1 < R1 ? c1 : R1;
                if (q1 <= q0) continue;
                if (ngb == cap) gbl = realloc(gbl, sizeof *gbl * (cap *= 2));
                rs_gblk *B = &gbl[ngb++]; B->sr = sr; B->q0 = q0; B->q1 = q1; B->a0 = (q0 / S) * S;
                rs_src R = {0}; rs_src_box(&R, org, h, B->a0, q1, hm, F, H, W, ssh); for (int d = 0; d < 3; d++) { B->so[d] = R.so[d]; B->sn[d] = R.sn[d]; }
                const int64_t s0 = B->so[0], s1 = s0 + B->sn[0];
                B->d0 = B->d1 = 0;
                if (s1 > bz1) { B->d0 = s0 > bz1 ? s0 : bz1; B->d1 = (s1 + cz - 1) / cz * cz; if (B->d1 > ssh[0]) B->d1 = ssh[0]; if (B->d1 < s1) B->d1 = s1; bz1 = B->d1; }
                if (bz1 - s0 > RZ) RZ = bz1 - s0;   /* (the ring holds [s0, bz1): nothing the block reads is overwritten) */
                if (ngb > 1 && bz1 - gbl[ngb - 2].so[0] > RZ) RZ = bz1 - gbl[ngb - 2].so[0];   /* (nor while the previous block's kernel runs: its upload overlaps that) */
            }
        }
        /* the source decoded on the device (whole 128^3 uint8 chunks into the ring, predict's chunk gathers; UFSM_RECTOSP_DEVDEC=0 off):
           the ring a multiple of 128 slices so a chunk row's slots are contiguous */
        const z3_meta *sm = z3_meta_of(zs);
        const int devdec = sm->chunk[0] == 128 && sm->chunk[1] == 128 && sm->chunk[2] == 128 && sm->esize == 1 && nn_ctw_new && (!getenv("UFSM_RECTOSP_DEVDEC") || atoi(getenv("UFSM_RECTOSP_DEVDEC")));
        if (devdec) RZ = (RZ + 127) / 128 * 128;
        const size_t ny = ngb ? gbl[0].sn[1] : 0, nx = ngb ? gbl[0].sn[2] : 0, ring = (size_t)RZ * ny * nx, fr = rsg_free_mem(gdev);
        const double st = 0.5 / F; const int n = 2 * (int)ceil(fmax(half, half2) / (st * um)) + 1;
        const size_t scr = (size_t)256 << 20;   /* (the cell arrays) */
        if (!ngb) ;
        else if (n > 1000) fprintf(stderr, "recto-sp: profiles of %d samples exceed the GPU kernel's shared memory: the CPU path\n", n);
        else if (ring + scr + ((size_t)512 << 20) > fr) fprintf(stderr, "recto-sp: GPU %d: %.1f GB free, the source ring needs %.1f GB: the CPU path\n", gdev, fr / 1e9, ring / 1e9);
        else if (rsg_open(gdev, RZ, (int64_t)ny, (int64_t)nx)) fprintf(stderr, "recto-sp: GPU %d unavailable: the CPU path\n", gdev);
        else { gon = 1; fprintf(stderr, "recto-sp: GPU %d: %d row blocks, source ring %lld x %zu x %zu (%.1f GB)\n", gdev, ngb, (long long)RZ, ny, nx, ring / 1e9); }
        if (!gon) { rsg_close(); free(gbl); gbl = nullptr; ngb = 0; }
        for (int b = 0; b < 2; b++) GP[b] = (rs_gprep){.in = in, .zs = zs, .nt = nt, .H = H, .W = W, .S = S, .shp0 = shp[0], .org = {org[0], org[1], org[2]}, .zoff0 = zoff[0], .bl = gbl};
        if (gon && devdec) {   /* chunk workers: UFSM_RECTOSP_GPUW (16) on the GPU decoder (UFSM_GPU_DECODE=1 on device gdev unless set), --threads on the CPU */
            char dv[16]; snprintf(dv, sizeof dv, "%d", gdev); setenv("UFSM_GPU_DECODE", "1", 0); setenv("UFSM_GPU_DECODE_DEV", dv, 0);
            const int ng = getenv("UFSM_RECTOSP_GPUW") ? atoi(getenv("UFSM_RECTOSP_GPUW")) : 16, nw = (ng > 0 ? ng : 0) + nt < 64 ? (ng > 0 ? ng : 0) + nt : 64;
            void **ctw = calloc(nw, sizeof *ctw); int ok = ctw != nullptr; for (int t = 0; t < nw && ok; t++) ok = (ctw[t] = nn_ctw_new(gdev)) != nullptr;
            if (ok) for (int b = 0; b < 2; b++) { GP[b].ctw = ctw; GP[b].nctw = nw; GP[b].ngpuw = ng > 0 ? (ng < nw ? ng : nw) : 0; }
            else { for (int t = 0; ctw && t < nw; t++) if (ctw[t]) nn_ctw_free(ctw[t]); free(ctw); fprintf(stderr, "recto-sp: no device chunk workers: the source decoded on the host\n"); }
        }
        if (gon) { GP[0].k = 0; if (pthread_create(&gth, nullptr, rs_gprep_main, &GP[0])) rs_gprep_main(&GP[0]); else gpend = 1; }
    }
    int gsr = 0; rs_fin FW = {0}; pthread_t fth; int fpend = 0;
    /* shard rows of the array covering the box */
    for (int64_t gz0 = (zoff[0] / SH) * SH; gz0 < zoff[0] + shp[0]; gz0 += SH, gsr++) {
        const int b0 = (int)(gz0 - zoff[0] < 0 ? 0 : gz0 - zoff[0]), b1 = (int)(gz0 + SH - zoff[0] > shp[0] ? shp[0] : gz0 + SH - zoff[0]);   /* box rows in this shard row */
        const int nr = b1 - b0; if (nr <= 0) continue;
        const size_t plane = (size_t)H * W; uint8_t *net = malloc(plane * nr), *rsp = calloc(plane * nr, 1);   /* (net: read by the writer thread) */
        if (zrin) { const int64_t o[3] = {zoff[0] + b0, zoff[1], zoff[2]}, nn[3] = {nr, H, W}; if (z3_read(zrin, o, nn, rsp, nt)) { fprintf(stderr, "recto-sp: %s\n", z3_error()); return 1; } }
        else if (gon) {   /* this shard row's blocks on the GPU */
            for (; gnext < ngb && gbl[gnext].sr == gsr; gnext++) {
                rs_gprep *cur = &GP[gnext & 1]; const rs_gblk *B = &gbl[gnext];
                { const double tw = rs_now(); if (gpend) { pthread_join(gth, nullptr); gpend = 0; } gtp += rs_now() - tw; }
                if (cur->rc) { fprintf(stderr, "recto-sp: block rows %d..%d: %s\n", B->q0, B->q1, cur->err); return 1; }
                const double tk = rs_now();
                if (gnext + 1 < ngb) { rs_gprep *nx = &GP[(gnext + 1) & 1];   /* (the next block's fields are this one's cache when they hold its rows) */
                    if (cur->fok) { if (nx->fok) fit_fields_free(&nx->fb); nx->fb = cur->fb; nx->f0 = cur->f0; nx->f1 = cur->f1; nx->fok = 1; cur->fok = 0; }
                    nx->k = gnext + 1; if (pthread_create(&gth, nullptr, rs_gprep_main, nx)) rs_gprep_main(nx); else gpend = 1; }
                const int a0 = B->a0, a1 = B->q1;
                rsg_par P = {.h = h, .F = F, .st = 0.5 / F, .um = um, .prom = prom, .minint = minint, .half1 = half2 > half};
                for (int d = 0; d < 3; d++) { P.so[d] = B->so[d]; P.sn[d] = B->sn[d]; }
                P.org[0] = (double)(org[0] + (long long)a0 * h); P.org[1] = (double)org[1]; P.org[2] = (double)org[2];
                const double hlf = fmax(half, half2); P.n = 2 * (int)ceil(hlf / (P.st * um)) + 1; P.c = P.n / 2;
                P.inner = P.half1 ? half / (P.st * um) : 1e30; P.Wn = (int)lround(60 / (P.st * um)); P.sep = sepum / (P.st * um);
                { const double sig = sigma * 2; const int R = sig > 0 ? (int)ceil(3 * sig) : 0; if (R > RSG_RMAX) { fprintf(stderr, "recto-sp: --sigma too wide for the GPU path (use --gpu -1)\n"); return 1; } P.R = rr_taps(sig, P.gk); }
                uint8_t *q = malloc(cur->M ? cur->M : 1); long long na = 0, ne = 0;
                { const double tr = rs_now(); if (rsg_run(&P, cur->M, cur->zyx, cur->t6, q, &na, &ne)) { fprintf(stderr, "recto-sp: GPU run failed\n"); return 1; } gtr += rs_now() - tr; }
                { rs_sc C = {.R = cur, .q = q, .a0 = a0, .a1 = a1, .q0 = B->q0, .q1 = B->q1, .b0 = b0, .H = H, .W = W, .S = S, .rsp = rsp};   /* rs_rows' writes, clipped to [q0, q1) */
                  fit_par(64, rs_scatter, &C); }
                free(q); tc += na; te += ne; gtk += rs_now() - tk; gtu = GP[0].tup + GP[1].tup; g_rectosp_gpu_blocks++;
                fprintf(stderr, "\rrecto-sp: rows %d / %d (%lld cells, %.3f estimated, %.0f s)", B->q1, shp[0], tc, tc ? (double)te / tc : 0, rs_now() - t0);
            }
        }
        else {
            /* the row blocks of this shard row; each block's source box is read by a helper thread while the previous block's profiles
               run (the source decode was 52 of 173 s serial on 0332, 2026-11-16; the bytes are the same: one z3_read per block as before) */
            int nb = 0, bq0[16], bq1[16], ba0[16];
            for (int c0 = b0; c0 < b1; c0 += 64) {
                const int c1 = c0 + 64 < b1 ? c0 + 64 : b1; const int q0 = c0 > R0 ? c0 : R0, q1 = c1 < R1 ? c1 : R1;
                if (q1 <= q0) continue;
                bq0[nb] = q0; bq1[nb] = q1; ba0[nb] = (q0 / S) * S; nb++;   /* (stride blocks start at multiples of S) */
            }
            const double hm = fmax(half, half2) / um + 2; const int64_t *ssh = z3_meta_of(zs)->shape;
            rs_src RS[2] = {{.z = zs, .nt = nt}, {.z = zs, .nt = nt}}; pthread_t th; int pend = 0;
            for (int k = 0; k < nb; k++) {
                rs_src *cur = &RS[k & 1];
                if (pend) { pthread_join(th, nullptr); pend = 0; }
                else { rs_src_box(cur, org, h, ba0[k], bq1[k], hm, F, H, W, ssh); rs_src_main(cur); }
                if (cur->rc) { fprintf(stderr, "recto-sp: source read: %s\n", cur->err); return 1; }
                if (k + 1 < nb) { rs_src *nx = &RS[(k + 1) & 1]; rs_src_box(nx, org, h, ba0[k + 1], bq1[k + 1], hm, F, H, W, ssh);
                                  if (pthread_create(&th, nullptr, rs_src_main, nx)) rs_src_main(nx); else pend = 1; }
                const int q0 = bq0[k], q1 = bq1[k], a0 = ba0[k], a1 = q1;
                fit_fields_skip_planes(1 | 2); fit_fields_ct_planes(0);
                const int hz = cls ? (int)ceil(fmax(half, half2) / um / h) + 1 : 0, l0 = a0 - hz < 0 ? 0 : a0 - hz, l1 = a1 + hz > shp[0] ? shp[0] : a1 + hz;   /* (--cls: the fields along the profiles: a halo of rows) */
                fit_fields_buf fb; if (fit_fields_load(in, l0, l1, 1, &fb)) { fprintf(stderr, "recto-sp: cannot load field rows %d..%d\n", l0, l1); return 1; }
                /* source block: native z [org + a0 h - half, org + a1 h + half], the box's y / x +- half (rs_src_box) */
                rs_job J = {.b = &fb, .r0 = a0 - l0, .nr = a1 - a0, .F = F, .org = {org[0] + (long long)l0 * h, org[1], org[2]}, .cls = cls, .h = h, .S = S, .um = um, .half = fmax(half, half2), .half1 = half2 > half ? half : 0,
                    .prom = opt_atof("--prom", rs_opt(argc, argv, "--prom", "25")), .sigma = opt_atof("--sigma", rs_opt(argc, argv, "--sigma", "0.5")), .sep = opt_atof("--sep-um", rs_opt(argc, argv, "--sep-um", "20")), .minint = opt_atoi("--min-int", rs_opt(argc, argv, "--min-int", "3")), .H = H, .W = W};
                for (int d = 0; d < 3; d++) { J.so[d] = cur->so[d]; J.sn[d] = cur->sn[d]; }
                J.src = cur->buf;
                /* output: rows a0..a1 of the box into rsp (rows from b0); the block starting before q0 / past b1 is clipped by the row test */
                uint8_t *tmpo = calloc((size_t)(a1 - a0) * plane, 1); J.out = tmpo; J.oz0 = J.r0;
                const int nyb = (H + S - 1) / S, nzb = (a1 - a0 + S - 1) / S;
                fit_par(nzb * nyb, rs_rows, &J);
                for (int r = q0; r < q1; r++) memcpy(rsp + (size_t)(r - b0) * plane, tmpo + (size_t)(r - a0) * plane, plane);
                free(tmpo); free(cur->buf); cur->buf = nullptr; fit_fields_free(&fb);
                tc += J.ncell; te += J.nest;
                fprintf(stderr, "\rrecto-sp: rows %d / %d (%lld cells, %.3f estimated, %.0f s)", q1, shp[0], tc, tc ? (double)te / tc : 0, rs_now() - t0);
            }
        }
        /* the blend and the shard writes of this row in a writer thread, behind the next row's blocks (one row in flight) */
        if (fpend) { pthread_join(fth, nullptr); fpend = 0; if (FW.rc) return 1; nmat += FW.nmat; nrep += FW.nrep; gtn += FW.tn; gtb += FW.tb; gtw += FW.tw; }
        FW = (rs_fin){.net = net, .rsp = rsp, .nr = nr, .b0 = b0, .b1 = b1, .gz0 = gz0, .H = H, .W = W, .SH = SH, .nt = nt, .zsp = zsp, .wsp = wsp, .wrs = wrs, .shard = shard, .shard2 = shard2,
                      .glo = glo, .ghi = ghi, .minnet = minnet, .blend = blend, .ramp = ramp, .rlo = rlo, .rhi = rhi, .ra0 = ra0, .ra1 = ra1, .h = h, .um = um};
        for (int d = 0; d < 3; d++) { FW.ash[d] = ash[d]; FW.zoff[d] = zoff[d]; }
        if (pthread_create(&fth, nullptr, rs_fin_main, &FW)) { rs_fin_main(&FW); if (FW.rc) return 1; nmat += FW.nmat; nrep += FW.nrep; gtn += FW.tn; gtb += FW.tb; gtw += FW.tw; } else fpend = 1;
    }
    if (fpend) { pthread_join(fth, nullptr); fpend = 0; if (FW.rc) return 1; nmat += FW.nmat; nrep += FW.nrep; gtn += FW.tn; gtb += FW.tb; gtw += FW.tw; }
    fprintf(stderr, "\n");
    if (z3w_close(wsp) || (wrs && z3w_close(wrs))) { fprintf(stderr, "recto-sp: close: %s\n", z3w_error()); return 1; }
    printf("recto-sp: %s -> %s: %lld cells probed, %.3f with a recto spacing; sp replaced on %.3f of the material cells (blend %g%s, gate %g..%g), %.0f s\n",
           in, out, tc, tc ? (double)te / tc : 0, nmat ? (double)nrep / nmat : 0, blend, ramp ? " ramp" : "", glo, ghi, rs_now() - t0);
    if (gon) { fprintf(stderr, "recto-sp: GPU path: %.1f s waiting for the helper (fields, cells, source decode), %.1f s on the blocks (kernel %.1f; the helper's uploads %.1f); writer thread: net sp read %.1f, blend %.1f, shard writes %.1f\n", gtp, gtk, gtr, gtu, gtn, gtb, gtw);
               fprintf(stderr, "recto-sp: helper: fields + cells %.1f s, source %.1f s (%s: %lld chunks on the GPU decoder, %lld on the host, %lld fill)\n", GP[0].tfld + GP[1].tfld, GP[0].tdec + GP[1].tdec,
                       GP[0].nctw ? "device decode" : "host decode + upload", (long long)(GP[0].ndev + GP[1].ndev), (long long)(GP[0].nhost + GP[1].nhost), (long long)(GP[0].nskip + GP[1].nskip));
               g_rectosp_dev_chunks = GP[0].ndev + GP[1].ndev; g_rectosp_ring_chunks = GP[0].nctw ? GP[0].ndev + GP[1].ndev + GP[0].nhost + GP[1].nhost + GP[0].nskip + GP[1].nskip : -1;
               if (gpend) pthread_join(gth, nullptr); if (GP[0].ctw) { for (int t = 0; t < GP[0].nctw; t++) nn_ctw_free(GP[0].ctw[t]); free(GP[0].ctw); }
               for (int b = 0; b < 2; b++) { if (GP[b].fok) fit_fields_free(&GP[b].fb); free(GP[b].stg); free(GP[b].zyx); free(GP[b].blk); free(GP[b].t6); } rsg_close(); free(gbl); }
    free(shard); free(shard2); rr_cls_free(cls); z3_close(zsp); if (zs) z3_close(zs); if (zrin) z3_close(zrin);
    return 0;
}
