/* ufsm merge-flag / wrap-mf (src/mergeflag.h) */
#include "mergeflag.h"
#include "mf_core.h"
#include "mfgpu.h"
#include "gpudec.h"
#include "defaulted.h"
#include "dinovol.h"
#include "json.h"
#include "optcheck.h"
#include "store.h"
#include "sysinfo.h"
#include "z3w.h"
#include "zarr3.h"
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <unistd.h>
#include <time.h>

/* a fixed rule with round constants (merge-flag --eval: within 0.002 AUC of the fitted logistic on every split; DESIGN "merge-flag") */
const mf_coef mf_coef_default = {.b = 4.03, .c_amp = -3.25, .c_r = -6.5, .c_sp = 0};   /* p = 1 / (1 + exp(-6.5 (0.62 - R5 - 0.5 amp9))) */
const int mf_thr_default = 179;   /* the knee of 1667's GT coverage curve (DESIGN "merge-flag"): p >= 0.7 */

enum { MF_R = 4 };
static double mf_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static const char *opt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 3; i + 1 < argc; i++) if (!strcmp(argv[i], opt_n(name))) return argv[i + 1];
    return dflt;
}   /* block border (cells): the 9-box */

/* in-place box sum of edge length 2 r + 1 along each axis, valid on [r, n - r) per axis (the rest left as it was); exact integers */
static void box3i(int32_t *a, int D, int H, int W, int r, int64_t *line, int32_t *tmp) {
    for (int z = 0; z < D; z++) for (int y = 0; y < H; y++) {   /* x */
        int32_t *p = a + ((size_t)z * H + y) * W; line[0] = 0; for (int x = 0; x < W; x++) line[x + 1] = line[x] + p[x];
        for (int x = r; x < W - r; x++) tmp[x] = (int32_t)(line[x + r + 1] - line[x - r]);
        for (int x = r; x < W - r; x++) p[x] = tmp[x];
    }
    for (int z = 0; z < D; z++) for (int x = r; x < W - r; x++) {   /* y */
        int32_t *p = a + (size_t)z * H * W + x; line[0] = 0; for (int y = 0; y < H; y++) line[y + 1] = line[y] + p[(size_t)y * W];
        for (int y = r; y < H - r; y++) tmp[y] = (int32_t)(line[y + r + 1] - line[y - r]);
        for (int y = r; y < H - r; y++) p[(size_t)y * W] = tmp[y];
    }
    const size_t HW = (size_t)H * W;
    for (int y = r; y < H - r; y++) for (int x = r; x < W - r; x++) {   /* z */
        int32_t *p = a + (size_t)y * W + x; line[0] = 0; for (int z = 0; z < D; z++) line[z + 1] = line[z] + p[z * HW];
        for (int z = r; z < D - r; z++) tmp[z] = (int32_t)(line[z + r + 1] - line[z - r]);
        for (int z = r; z < D - r; z++) p[z * HW] = tmp[z];
    }
}

/* the CPU reference of the GPU path (src/nn_mf.cu): the 12 integer sums of src/mf_core.h per cell, then mf_final */
void mf_block(const uint8_t *pc, const uint8_t *ps, const uint8_t *sp, const uint8_t *w, int D, int H, int W, const long long g0[3],
              const long long gn[3], const mf_coef *c, uint8_t *out, float *amp9o, float *r5o, float *sp9o) {
    const size_t N = (size_t)D * H * W, HW = (size_t)H * W; const int n3[3] = {D, H, W}; const size_t st[3] = {HW, (size_t)W, 1};
    const int iD = D - 2 * MF_R, iH = H - 2 * MF_R, iW = W - 2 * MF_R;
    const size_t NI = iD > 0 && iH > 0 && iW > 0 ? (size_t)iD * iH * iW : 0;
    int32_t *a = malloc(N * 4), *S = malloc((NI ? NI : 1) * MF_NS * 4);
    const int L = D > H ? (D > W ? D : W) : (H > W ? H : W); int64_t *line = malloc((L + 1) * sizeof *line); int32_t *tmp = malloc(L * 4);
    if (!a || !S || !line || !tmp) abort();
    for (int ch = 0; ch < MF_NS; ch++) {
        if (ch < 6) for (size_t i = 0; i < N; i++) {
            const int m = w[i] > 0;
            a[i] = ch == 0 ? (m ? mf_ampq(pc[i], ps[i]) : 0) : ch == 1 ? m : ch == 2 || ch == 4 ? sp[i] : sp[i] > 0;
        } else {   /* CR / CI of axis ax: z(g' + e_a) conj z(g'), g' = min(g, N_a - 2) */
            const int ax = (ch - 6) / 2, im = (ch - 6) & 1;
            for (int z = 0; z < D; z++) for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
                const int q[3] = {z, y, x}; const size_t i = (size_t)z * HW + (size_t)y * W + x;
                long long g = g0[ax] + q[ax]; if (g > gn[ax] - 2) g = gn[ax] - 2; if (g < 0) g = 0;
                const long long l = g - g0[ax];
                if (gn[ax] < 2 || l < 0 || l + 1 >= n3[ax]) { a[i] = 0; continue; }
                const size_t j0 = i + (size_t)(l - q[ax]) * st[ax], j1 = j0 + st[ax];
                const int ar = pc[j0] - 128, ai = ps[j0] - 128, br = pc[j1] - 128, bi = ps[j1] - 128;
                a[i] = im ? bi * ar - br * ai : br * ar + bi * ai;
            }
        }
        box3i(a, D, H, W, mf_rad(ch), line, tmp);
        size_t k = 0;
        for (int z = MF_R; z < D - MF_R; z++) for (int y = MF_R; y < H - MF_R; y++) for (int x = MF_R; x < W - MF_R; x++, k++)
            S[k * MF_NS + ch] = a[(size_t)z * HW + (size_t)y * W + x];
    }
    memset(out, 0, N);
    size_t k = 0;
    for (int z = MF_R; z < D - MF_R; z++) for (int y = MF_R; y < H - MF_R; y++) for (int x = MF_R; x < W - MF_R; x++, k++) {
        const size_t i = (size_t)z * HW + (size_t)y * W + x; const int32_t *s = S + k * MF_NS;
        if (amp9o || r5o || sp9o) { const mf_feat f = mf_features(s); if (amp9o) amp9o[i] = (float)f.amp9; if (r5o) r5o[i] = (float)f.r5; if (sp9o) sp9o[i] = (float)f.sp9; }
        out[i] = mf_final(s, w[i] > 0, c->b, c->c_amp, c->c_r, c->c_sp);
    }
    free(a); free(S); free(line); free(tmp);
}

/* ---- a fields directory (zarr planes or NAME.raw; crops: zarr_offset / raw_offset + raw_shape) ---- */
typedef struct { char dir[1400]; json *j; int zarr, h; long long shape[3], off[3], rsh[3], origin[3]; double um; } mf_in;
static json *mf_json(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) return nullptr;
    fseek(f, 0, SEEK_END); const long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *t = malloc((size_t)n + 1); if (!t || fread(t, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(t); return nullptr; }
    fclose(f); t[n] = 0; json *j = json_parse(t, (size_t)n); free(t); return j;
}
static int mf_open(mf_in *I, const char *dir) {
    memset(I, 0, sizeof *I); snprintf(I->dir, sizeof I->dir, "%s", dir);
    char p[1500]; snprintf(p, sizeof p, "%s/fields.json", dir);
    if (!(I->j = mf_json(p))) { fprintf(stderr, "merge-flag: cannot read %s\n", p); return -1; }
    I->zarr = !strcmp(json_str(json_get(I->j, "format"), "raw"), "zarr"); I->h = (int)json_num(json_get(I->j, "h"), 8); I->um = json_num(json_get(I->j, "um"), 2.4);
    for (int d = 0; d < 3; d++) {
        I->shape[d] = (long long)json_num(json_at(json_get(I->j, "shape"), d), 0); I->origin[d] = (long long)json_num(json_at(json_get(I->j, "origin"), d), 0);
        I->off[d] = (long long)json_num(json_at(json_get(I->j, I->zarr ? "zarr_offset" : "raw_offset"), d), 0);
        I->rsh[d] = I->zarr ? 0 : (long long)json_num(json_at(json_get(I->j, "raw_shape"), d), (double)I->shape[d]);
        if (I->shape[d] <= 0) { fprintf(stderr, "merge-flag: %s: no shape\n", p); return -1; }
    }
    return 0;
}
typedef struct { z3 *z; FILE *f; const mf_in *I; } mf_rd;
static int mf_rd_open(mf_rd *r, const mf_in *I, const char *name) {
    r->I = I; r->z = nullptr; r->f = nullptr; char p[1500];
    if (I->zarr) { r->z = z3_open(store_open(I->dir), name, nullptr); if (!r->z) { fprintf(stderr, "merge-flag: %s/%s: %s\n", I->dir, name, z3_error()); return -1; } return 0; }
    snprintf(p, sizeof p, "%s/%s.raw", I->dir, name); if (!(r->f = fopen(p, "rb"))) { fprintf(stderr, "merge-flag: cannot open %s\n", p); return -1; }
    return 0;
}
static int mf_rows(mf_rd *r, long long z0, long long nz, uint8_t *out, int nthreads) {   /* whole rows [z0, z0 + nz) of the box */
    const mf_in *I = r->I; const long long H = I->shape[1], W = I->shape[2];
    if (r->z) { const int64_t o[3] = {z0 + I->off[0], I->off[1], I->off[2]}, n[3] = {nz, H, W}; return z3_read(r->z, o, n, out, nthreads) ? -1 : 0; }
    for (long long z = 0; z < nz; z++) for (long long y = 0; y < H; y++) {
        const off_t at = (off_t)(((I->off[0] + z0 + z) * I->rsh[1] + (I->off[1] + y)) * I->rsh[2] + I->off[2]);
        if (fseeko(r->f, at, SEEK_SET) || fread(out + (size_t)(z * H + y) * W, 1, (size_t)W, r->f) != (size_t)W) return -1;
    }
    return 0;
}
static void mf_rd_close(mf_rd *r) { if (r->z) z3_close(r->z); if (r->f) fclose(r->f); }

/* ---- merge-flag: z slabs of SZ rows (+ MF_R each side) in memory, 128 x 128 (y, x) tiles in threads ---- */
enum { SZ = 128, TY = 128 };
typedef struct {
    const mf_in *I; const mf_coef *c; long long z0, nz, rz0, rnz;   /* the slab's rows and the rows read (with the border) */
    const uint8_t *pl[4]; uint8_t *outrows, *mfrows; z3w *zw; int ny, nx; _Atomic int next, fail; _Atomic long long nmat, nflag; int thr;
} mf_job;
static void *mf_worker(void *a) {
    mf_job *J = a; const long long H = J->I->shape[1], W = J->I->shape[2], D = J->I->shape[0];
    const int bd = (int)J->nz + 2 * MF_R, bh = TY + 2 * MF_R, bw = TY + 2 * MF_R; const size_t bn = (size_t)bd * bh * bw;
    uint8_t *b[4], *o = malloc(bn), *sh = J->zw ? malloc((size_t)SZ * TY * TY) : nullptr;
    for (int k = 0; k < 4; k++) b[k] = malloc(bn);
    for (int t; (t = atomic_fetch_add(&J->next, 1)) < J->ny * J->nx;) {
        const long long y0 = (long long)(t / J->nx) * TY, x0 = (long long)(t % J->nx) * TY;
        const long long ty = y0 + TY < H ? TY : H - y0, tx = x0 + TY < W ? TY : W - x0;
        const int hh = (int)ty + 2 * MF_R, ww = (int)tx + 2 * MF_R;   /* the block: edge-replicated outside the box */
        for (int k = 0; k < 4; k++) for (int z = 0; z < bd; z++) for (int y = 0; y < hh; y++) {
            long long gz = J->z0 - MF_R + z, gy = y0 - MF_R + y; gz = gz < 0 ? 0 : gz >= D ? D - 1 : gz; gy = gy < 0 ? 0 : gy >= H ? H - 1 : gy;
            const uint8_t *src = J->pl[k] + ((size_t)(gz - J->rz0) * H + gy) * W; uint8_t *dst = b[k] + ((size_t)z * hh + y) * ww;
            for (int x = 0; x < ww; x++) { long long gx = x0 - MF_R + x; gx = gx < 0 ? 0 : gx >= W ? W - 1 : gx; dst[x] = src[gx]; }
        }
        const long long g0[3] = {J->z0 - MF_R, y0 - MF_R, x0 - MF_R}, gn[3] = {D, H, W};
        mf_block(b[0], b[1], b[2], b[3], bd, hh, ww, g0, gn, J->c, o, nullptr, nullptr, nullptr);
        long long nm = 0, nf = 0;
        if (sh) memset(sh, 0, (size_t)SZ * TY * TY);
        for (long long z = 0; z < J->nz; z++) for (long long y = 0; y < ty; y++) {
            const uint8_t *s = o + ((size_t)(z + MF_R) * hh + (y + MF_R)) * ww + MF_R, *wm = b[3] + ((size_t)(z + MF_R) * hh + (y + MF_R)) * ww + MF_R;
            for (long long x = 0; x < tx; x++) { nm += wm[x] > 0; nf += wm[x] > 0 && s[x] >= J->thr; }
            if (sh) memcpy(sh + ((size_t)z * TY + y) * TY, s, (size_t)tx);
            else memcpy(J->outrows + ((size_t)z * H + y0 + y) * W + x0, s, (size_t)tx);
            if (J->mfrows) memcpy(J->mfrows + ((size_t)z * H + y0 + y) * W + x0, s, (size_t)tx);
        }
        atomic_fetch_add(&J->nmat, nm); atomic_fetch_add(&J->nflag, nf);
        if (sh && z3w_write_shard(J->zw, J->z0 / SZ, y0 / TY, x0 / TY, sh, 1)) atomic_store(&J->fail, 1);
    }
    for (int k = 0; k < 4; k++) free(b[k]);
    free(o); free(sh); return nullptr;
}

static int mf_link_dir(const char *src, const char *out);
static int mf_weight_create(const mf_in *I, const char *src, const char *out, double K, int cut, z3w *zw[3]);
/* rows [z0, z0 + nz) as 128^3 shards, a thread per shard (each shard's encode is one chunk: z3w_write_shard's threads idle) */
typedef struct { z3w *zw; const uint8_t *rows; long long z0, nz, H, W; int fill, ny, nx; _Atomic int next, fail; } mf_wjob;
static void *mf_wpar(void *a) {
    mf_wjob *J = a; uint8_t *sb = malloc((size_t)SZ * SZ * SZ); if (!sb) { atomic_store(&J->fail, 1); return nullptr; }
    for (int t; (t = atomic_fetch_add(&J->next, 1)) < J->ny * J->nx;) {
        const long long y0 = (long long)(t / J->nx) * SZ, x0 = (long long)(t % J->nx) * SZ;
        memset(sb, J->fill, (size_t)SZ * SZ * SZ);
        for (long long z = 0; z < J->nz; z++) for (long long y = y0; y < y0 + SZ && y < J->H; y++)
            memcpy(sb + ((size_t)z * SZ + (y - y0)) * SZ, J->rows + ((size_t)z * J->H + y) * J->W + x0, (size_t)(x0 + SZ < J->W ? SZ : J->W - x0));
        if (z3w_write_shard(J->zw, J->z0 / SZ, y0 / SZ, x0 / SZ, sb, 1)) atomic_store(&J->fail, 1);
    }
    free(sb); return nullptr;
}
static int mf_write_par(z3w *zw, const uint8_t *rows, long long z0, long long nz, long long H, long long W, int fill, int nt) {
    mf_wjob J = {.zw = zw, .rows = rows, .z0 = z0, .nz = nz, .H = H, .W = W, .fill = fill, .ny = (int)((H + SZ - 1) / SZ), .nx = (int)((W + SZ - 1) / SZ)};
    pthread_t th[64]; const int m = nt < 64 ? nt : 64;
    for (int t = 0; t < m; t++) pthread_create(&th[t], nullptr, mf_wpar, &J);
    for (int t = 0; t < m; t++) pthread_join(th[t], nullptr);
    return atomic_load(&J.fail);
}
/* the GPU path, three stages overlapped: slab k + 1's planes read (a reader thread), slab k on the GPU (this thread), slab k - 1's mf
   shards encoded and written (a writer thread). Double buffers: the planes (4 x (128 + 8) rows) and the mf rows */
typedef struct { mf_rd *rd; uint8_t **pl; long long rz0, rnz; int nt, rc; } mf_rjob;
static void *mf_rthread(void *a) {
    mf_rjob *R = a; R->rc = 0;
    for (int k = 0; k < 4 && !R->rc; k++) if (mf_rows(&R->rd[k], R->rz0, R->rnz, R->pl[k], R->nt)) R->rc = 1;
    return nullptr;
}
typedef struct { z3w *zw; FILE *fo; const uint8_t *rows; long long z0, nz, H, W; int nt, rc; } mf_wjob2;
static void *mf_wthread(void *a) {
    mf_wjob2 *J = a; const size_t n = (size_t)J->nz * J->H * J->W; J->rc = 0;
    if (J->zw && mf_write_par(J->zw, J->rows, J->z0, J->nz, J->H, J->W, 0, J->nt)) { fprintf(stderr, "merge-flag: shard write failed: %s\n", z3w_error()); J->rc = 1; }
    if (J->fo && fwrite(J->rows, 1, n, J->fo) != n) { fprintf(stderr, "merge-flag: write failed\n"); J->rc = 1; }
    return nullptr;
}
static int mf_run_gpu(mf_gpu *G, mf_rd *rd, uint8_t *pl0[4], long long D, long long H, long long W, const double cv[4], int thr, z3w *zw, FILE *fo, int nt,
                      uint8_t *mfr, z3w *ww[3], uint8_t *wr[3], double K, int cut, size_t *wmat, double *wsum, long long *nmat, long long *nflag, double tm[4]) {
    const size_t rmax = (size_t)(SZ + 2 * MF_R) * H * W, srow = (size_t)SZ * H * W;
    uint8_t *plb[2][4] = {{0}}, *gr[2]; int rc = 0;   /* (page-locked: the uploads run at the bus rate) */
    (void)pl0;
    for (int b = 0; b < 2; b++) for (int k = 0; k < 4; k++) if (!(plb[b][k] = mf_gpu_host_alloc(rmax))) rc = 1;
    gr[0] = mf_gpu_host_alloc(srow); gr[1] = mf_gpu_host_alloc(srow);
    if (rc || !gr[0] || !gr[1]) { fprintf(stderr, "merge-flag: no page-locked host memory (%.1f GB)\n", (8.0 * rmax + 2.0 * srow) / 1e9); rc = 1; goto done; }
    const int rnt = gpudec_threads(nt);
    const long long ns = (D + SZ - 1) / SZ;
    mf_rjob R = {.rd = rd}; mf_wjob2 Wj = {.zw = zw, .fo = fo, .H = H, .W = W, .nt = nt}; pthread_t rt, wt; int rrun = 0, wrun = 0;
    #define MF_SLAB(i, z0_, nz_, rz0_, rz1_) const long long z0_ = (i) * SZ, nz_ = z0_ + SZ < D ? SZ : D - z0_, rz0_ = z0_ - MF_R < 0 ? 0 : z0_ - MF_R, rz1_ = z0_ + nz_ + MF_R > D ? D : z0_ + nz_ + MF_R
    { MF_SLAB(0, z0, nz, rz0, rz1); (void)z0; (void)nz; R.pl = plb[0]; R.rz0 = rz0; R.rnz = rz1 - rz0; R.nt = rnt; mf_rthread(&R); if (R.rc) { rc = 1; goto done; } }
    for (long long i = 0; i < ns && !rc; i++) {
        MF_SLAB(i, z0, nz, rz0, rz1);
        double t0 = mf_now();
        if (rrun) { pthread_join(rt, nullptr); rrun = 0; if (R.rc) { fprintf(stderr, "merge-flag: cannot read rows near %lld\n", z0); rc = 1; break; } }
        tm[0] += mf_now() - t0;
        if (i + 1 < ns) { MF_SLAB(i + 1, z1_, nz1_, rz0n, rz1n); (void)z1_; (void)nz1_; R.pl = plb[(i + 1) & 1]; R.rz0 = rz0n; R.rnz = rz1n - rz0n; R.nt = rnt;
                          pthread_create(&rt, nullptr, mf_rthread, &R); rrun = 1; }
        uint8_t **pl = plb[i & 1], *g = gr[i & 1]; t0 = mf_now();
        long long cm = 0, cf = 0;
        if (mf_gpu_rows(G, (const uint8_t *const *)pl, D, rz0, rz1 - rz0, z0, nz, cv, thr, g, &cm, &cf)) { rc = 1; break; }
        *nmat += cm; *nflag += cf; tm[1] += mf_now() - t0; t0 = mf_now();
        if (wrun) { pthread_join(wt, nullptr); wrun = 0; if (Wj.rc) { rc = 1; break; } }
        tm[2] += mf_now() - t0;
        Wj.rows = g; Wj.z0 = z0; Wj.nz = nz; pthread_create(&wt, nullptr, mf_wthread, &Wj); wrun = 1;
        if (mfr) {   /* --weighted: the slab's weighted planes (rows z0.. of the read buffer, which starts at rz0), here (no overlap) */
            t0 = mf_now(); const size_t n = (size_t)nz * H * W, o = (size_t)(z0 - rz0) * H * W; const int src[3] = {0, 1, 3};
            for (int k = 0; k < 3; k++) memcpy(wr[k], pl[src[k]] + o, n);
            mf_weight_rows(wr[0], wr[1], wr[2], g, n, K, cut, wmat, wsum);
            for (int k = 0; k < 3 && !rc; k++) if (mf_write_par(ww[k], wr[k], z0, nz, H, W, k < 2 ? 128 : 0, nt)) { fprintf(stderr, "merge-flag --weighted: %s\n", z3w_error()); rc = 1; }
            tm[3] += mf_now() - t0;
        }
    }
    if (rrun) pthread_join(rt, nullptr);
    if (wrun) { pthread_join(wt, nullptr); if (Wj.rc) rc = 1; }
    #undef MF_SLAB
done:
    for (int b = 0; b < 2; b++) for (int k = 0; k < 4; k++) mf_gpu_host_free(plb[b][k]);
    mf_gpu_host_free(gr[0]); mf_gpu_host_free(gr[1]);
    return rc;
}

/* ufsm merge-flag <fields-dir> [--out DIR] [--coef b,amp,r,sp] [--thr T] [--threads N] [--weighted DIR [--mf-w 1 | --mf-cut T]] */
static int cmd_merge_flag_eval(int argc, char **argv, const char *list);
static int cmd_sheet_pairs(int argc, char **argv, const char *list);
static int cmd_merge_flag_apply(int argc, char **argv, const char *src, const char *out);
int cmd_merge_flag(int argc, char **argv) {
    if (argc >= 5 && !strcmp(argv[2], opt_n("--apply"))) return cmd_merge_flag_apply(argc, argv, argv[3], argv[4]);
    if (argc >= 4 && !strcmp(argv[2], opt_n("--eval"))) return cmd_merge_flag_eval(argc, argv, argv[3]);
    if (argc >= 4 && !strcmp(argv[2], opt_n("--sheet-pairs"))) return cmd_sheet_pairs(argc, argv, argv[3]);
    if (argc < 3) {
        fprintf(stderr, "usage: ufsm merge-flag <fields-dir> [--out DIR] [--coef b,c_amp,c_r,c_sp] [--thr 179] [--threads N] [--gpu 0 (-1: CPU)] [--gpu-mem 3]\n"
                        "       ... --weighted DIR: also DIR = the fields with their evidence x (1 - mf / 255) (merge-flag --apply in the same pass; zarr fields)\n"
                        "       ufsm merge-flag --apply FIELDS OUT [--mf-w 1 | --mf-cut T]   FIELDS' evidence down-weighted where merged (pc ps w x (1 - K mf / 255)) for count-field / fit\n"
                        "       ufsm merge-flag --eval LIST [--coef ..] [--threads 4]   the rule against GT intervals (LIST: name scroll fields-dir codes-root codes-key)\n"
                        "         [--dense-pts DIR]: write each box's probe pairs (DIR/NAME.pts) for dinovol-probe pfeats; [--dense-cos DIR]: + the dinovol dense cos (DIR/NAME.dvf)\n"
                        "  the merged-sheet score of every field cell (src/mergeflag.h) as plane mf: zarr fields -> DIR/mf (zarr, 128^3 shards), raw\n"
                        "  fields -> DIR/mf.raw (DIR: default the fields directory; a crop needs --out). Prints the material share flagged (mf >= thr).\n");
        return 2;
    }
    mf_in I; if (mf_open(&I, argv[2])) return 1;
    const int crop = I.off[0] || I.off[1] || I.off[2] || (!I.zarr && (I.rsh[0] != I.shape[0] || I.rsh[1] != I.shape[1] || I.rsh[2] != I.shape[2]));
    const char *out = opt(argc, argv, "--out", argv[2]);
    if (crop && !strcmp(out, argv[2])) { fprintf(stderr, "merge-flag: %s is a crop of another fields set: give --out\n", argv[2]); json_free(I.j); return 2; }
    mf_coef c = mf_coef_default; const char *cs = opt(argc, argv, "--coef", nullptr);
    if (cs && sscanf(cs, "%lf,%lf,%lf,%lf", &c.b, &c.c_amp, &c.c_r, &c.c_sp) != 4) { fprintf(stderr, "merge-flag: --coef b,c_amp,c_r,c_sp\n"); return 2; }
    const int thr = opt_atoi("--thr", opt(argc, argv, "--thr", "179")), nt = ufsm_threads(opt_atoi("--threads", opt(argc, argv, "--threads", "0")), 16);
    const char *names[4] = {"pc", "ps", "sp", "w"}; mf_rd rd[4];
    for (int k = 0; k < 4; k++) if (mf_rd_open(&rd[k], &I, names[k])) return 1;
    mkdir(out, 0775);
    const long long D = I.shape[0], H = I.shape[1], W = I.shape[2];
    z3w *zw = nullptr; FILE *fo = nullptr; char p[1500];
    if (I.zarr) {
        char at[512]; snprintf(at, sizeof at, "{\"ufsm\":{\"content\":\"merge-flag: merged-sheet score x255 per field cell (src/mergeflag.h)\",\"h\":%d,"
                               "\"mf_thr\":%d,\"coef\":[%.6g,%.6g,%.6g,%.6g]}}", I.h, thr, c.b, c.c_amp, c.c_r, c.c_sp);
        snprintf(p, sizeof p, "%s/mf", out); const int64_t sh[3] = {D, H, W};
        if (!(zw = z3w_create(p, sh, SZ, 0, 0, at))) { fprintf(stderr, "merge-flag: %s\n", z3w_error()); return 1; }
    } else { snprintf(p, sizeof p, "%s/mf.raw", out); if (!(fo = fopen(p, "wb"))) { fprintf(stderr, "merge-flag: cannot write %s\n", p); return 1; } }
    uint8_t *pl[4] = {0}; const size_t rmax = (size_t)(SZ + 2 * MF_R) * H * W; for (int k = 0; k < 4; k++) if (!(pl[k] = malloc(rmax))) { fprintf(stderr, "merge-flag: out of memory\n"); return 1; }
    uint8_t *orows = fo ? malloc((size_t)SZ * H * W) : nullptr;
    long long nmat = 0, nflag = 0; int rc = 0; double tm[4] = {0};   /* read, gpu, mf write, weighted planes (GPU path) */
    /* --gpu N (default 0; -1: the CPU threads): the GPU path (src/nn_mf.cu), within 1 LSB of the CPU's; --gpu-mem GB its tiles */
    const int gdev = opt_atoi("--gpu", opt(argc, argv, "--gpu", "0")); mf_gpu *G = nullptr;
    const double cv[4] = {c.b, c.c_amp, c.c_r, c.c_sp};
    if (gdev >= 0) {
        G = mf_gpu_open(gdev, H, W, opt_atof("--gpu-mem", opt(argc, argv, "--gpu-mem", "3")));
        if (!G) UFSM_DEFAULTED("merge-flag: no GPU %d: the CPU path (%d threads)", gdev, nt);
        else if (opt_atoi("--gpu-decode", opt(argc, argv, "--gpu-decode", "1"))) {   /* the field planes' volcomp decode on the same GPU (src/gpudec.h) */
            char dv[16]; snprintf(dv, sizeof dv, "%d", gdev); setenv("UFSM_GPU_DECODE", "1", 0); setenv("UFSM_GPU_DECODE_DEV", dv, 0);
        }
    }
    /* --weighted DIR: the count-field input (the unroll's MFREF): pc ps w x (1 - K mf / 255) beside links to the other planes */
    const char *wd = opt(argc, argv, "--weighted", nullptr); z3w *ww[3] = {0}; uint8_t *mfr = nullptr, *wr[3] = {0}, *sb = nullptr;
    const double K = opt_atof("--mf-w", opt(argc, argv, "--mf-w", "1")); const int cut = opt_atoi("--mf-cut", opt(argc, argv, "--mf-cut", "0")); size_t wmat = 0; double wsum = 0;
    if (wd) {
        if (!I.zarr || crop) { fprintf(stderr, "merge-flag: --weighted needs whole zarr fields\n"); return 2; }
        if (mf_link_dir(argv[2], wd) || mf_weight_create(&I, argv[2], wd, K, cut, ww)) return 1;
        mfr = malloc((size_t)SZ * H * W); sb = malloc((size_t)SZ * SZ * SZ); for (int k = 0; k < 3; k++) wr[k] = malloc((size_t)SZ * H * W);
        if (!mfr || !sb || !wr[0] || !wr[1] || !wr[2]) { fprintf(stderr, "merge-flag: out of memory\n"); return 1; }
    }
    if (G) rc = mf_run_gpu(G, rd, pl, D, H, W, cv, thr, zw, fo, nt, wd ? mfr : nullptr, ww, wr, K, cut, &wmat, &wsum, &nmat, &nflag, tm);
    for (long long z0 = 0; z0 < D && !rc && !G; z0 += SZ) {
        const long long nz = z0 + SZ < D ? SZ : D - z0, rz0 = z0 - MF_R < 0 ? 0 : z0 - MF_R, rz1 = z0 + nz + MF_R > D ? D : z0 + nz + MF_R;
        double t0 = mf_now();
        for (int k = 0; k < 4 && !rc; k++) if (mf_rows(&rd[k], rz0, rz1 - rz0, pl[k], G ? gpudec_threads(nt) : nt)) { fprintf(stderr, "merge-flag: cannot read %s rows %lld..%lld\n", names[k], rz0, rz1); rc = 1; }
        if (rc) break;
        {
        mf_job J = {.I = &I, .c = &c, .z0 = z0, .nz = nz, .rz0 = rz0, .rnz = rz1 - rz0, .pl = {pl[0], pl[1], pl[2], pl[3]}, .outrows = orows, .mfrows = mfr, .zw = zw,
                    .ny = (int)((H + TY - 1) / TY), .nx = (int)((W + TY - 1) / TY), .thr = thr};
        pthread_t th[64]; const int m = nt < 64 ? nt : 64;
        for (int t = 0; t < m; t++) pthread_create(&th[t], nullptr, mf_worker, &J);
        for (int t = 0; t < m; t++) pthread_join(th[t], nullptr);
        if (atomic_load(&J.fail)) { fprintf(stderr, "merge-flag: shard write failed: %s\n", z3w_error()); rc = 1; }
        if (fo && fwrite(orows, 1, (size_t)nz * H * W, fo) != (size_t)(nz * H * W)) { fprintf(stderr, "merge-flag: write failed\n"); rc = 1; }
        nmat += atomic_load(&J.nmat); nflag += atomic_load(&J.nflag);
        }
        if (wd && !rc) {   /* the slab's weighted planes (the rows z0.. of the read buffer: it starts at rz0) */
            const size_t n = (size_t)nz * H * W, o = (size_t)(z0 - rz0) * H * W; const int src[3] = {0, 1, 3};
            for (int k = 0; k < 3; k++) memcpy(wr[k], pl[src[k]] + o, n);
            mf_weight_rows(wr[0], wr[1], wr[2], mfr, n, K, cut, &wmat, &wsum);
            for (int k = 0; k < 3 && !rc; k++) if (mf_write_par(ww[k], wr[k], z0, nz, H, W, k < 2 ? 128 : 0, nt)) { fprintf(stderr, "merge-flag --weighted: %s\n", z3w_error()); rc = 1; }
            tm[3] += mf_now() - t0;
        }
    }
    for (int k = 0; k < 3; k++) { if (ww[k] && z3w_close(ww[k])) rc = 1; free(wr[k]); }
    free(mfr); free(sb); mf_gpu_close(G);
    if (wd && !rc) printf("merge-flag --weighted: %s: mean evidence factor %.4f over %zu material cells\n", wd, wmat ? wsum / wmat : 0, wmat);
    for (int k = 0; k < 4; k++) { free(pl[k]); mf_rd_close(&rd[k]); }
    free(orows);
    if (zw && z3w_close(zw)) { fprintf(stderr, "merge-flag: %s\n", z3w_error()); rc = 1; }
    if (fo && fclose(fo)) rc = 1;
    if (!rc && G) printf("merge-flag gpu: waited for reads %.1f s, GPU %.1f s, waited for writes %.1f s, weighted planes %.1f s\n", tm[0], tm[1], tm[2], tm[3]);
    if (!rc) printf("merge-flag: %lld x %lld x %lld cells, %lld with material, flagged (mf >= %d) %.4f -> %s\n", D, H, W, nmat, thr, nmat ? (double)nflag / nmat : 0, p);
    json_free(I.j);
    return rc;
}

/* ---- merge-flag --eval: the rule against GT, streamed box by box (no per-box arrays kept beyond its intervals) ----
   Truth (GT intervals): in NZ z planes of each box, rays from the fields' axis at NTH angles, one native voxel a step; GT sheet
   crossings = runs of labelled code voxels (code 1..252, winding (code - 1) / 18 mod 14), their middles; two consecutive crossings one
   turn apart (d = 1 or 13 mod 14, >= 2 steps apart) make an interval; n = the net's phase (atan2 of the cell's pc, ps) unwrapped along
   the ray between them, signed by d, the box's sign the median's; merged = n < 0.5 (the phase misses the turn). The interval's features
   are the cell at its middle (mf_block over the whole box, borders replicated). */
enum { EV_NZ = 16, EV_NTH = 720, EV_NF = 4 };
/* features amp9, r5, sp9 (+ f[3] dcos with --dense-cos, else 0); gap (um, along the ray); cs |cos(ray, sheet normal)|; y merged;
   pt: the --dense-pts probe pair (absolute native z y x of m - s/2 u and m + s/2 u: the interval's middle m, the ray direction u, the
   net's own local spacing s = sp h / 8 voxels; no GT position enters) */
typedef struct { float f[EV_NF], gap, cs, pt[6]; uint8_t y; } ev_iv;
static int ev_nf = 3;   /* mgpu: config. The logistic's features: 3, or 4 with --dense-cos */
/* |cos| between the ray (sy, sx) and the GT sheet's in-slice normal at label voxel (ly, lx): the normal = the minor axis of the
   covariance of the same-code label voxels within radius r (2-D, the slice; sheets' z tilt ignored); -1 when too few voxels */
static float ev_cos(const uint8_t *g, long long gny, long long gnx, long long ly, long long lx, int r, double sy, double sx) {
    const uint8_t c = g[(size_t)ly * gnx + lx]; double n = 0, my = 0, mx = 0, yy = 0, xx = 0, xy = 0;
    for (long long y = ly - r; y <= ly + r; y++) for (long long x = lx - r; x <= lx + r; x++) {
        if (y < 0 || x < 0 || y >= gny || x >= gnx || (y - ly) * (y - ly) + (x - lx) * (x - lx) > r * r || g[(size_t)y * gnx + x] != c) continue;
        const double dy = (double)(y - ly), dx = (double)(x - lx); n++; my += dy; mx += dx; yy += dy * dy; xx += dx * dx; xy += dy * dx;
    }
    if (n < 2 * r) return -1;
    my /= n; mx /= n; yy = yy / n - my * my; xx = xx / n - mx * mx; xy = xy / n - my * mx;
    const double th = 0.5 * atan2(2 * xy, xx - yy);   /* the major axis (tangent) angle in (x, y); normal = (-sin, cos) in (x, y) */
    const double nx = -sin(th), ny = cos(th); return (float)fabs(nx * sx + ny * sy);
}
typedef struct { char name[64], scroll[32], fields[1024], croot[1024], ckey[64]; ev_iv *iv; size_t n, cap; int rc; double merged; } ev_box;
static void ev_push(ev_box *B, const ev_iv *v) {
    if (B->n == B->cap) { B->cap = B->cap ? 2 * B->cap : 4096; B->iv = realloc(B->iv, B->cap * sizeof *B->iv); if (!B->iv) abort(); }
    B->iv[B->n++] = *v;
}
static int ev_cmpd(const void *a, const void *b) { const double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }
static int ev_one(ev_box *B) {
    mf_in I; if (mf_open(&I, B->fields)) return -1;
    const long long D = I.shape[0], H = I.shape[1], W = I.shape[2]; const int h = I.h; const size_t N = (size_t)D * H * W;
    const json *ax = json_get(I.j, "axis"); const char *names[4] = {"pc", "ps", "sp", "w"}; uint8_t *pl[4] = {0}; int rc = 0;
    for (int k = 0; k < 4 && !rc; k++) { mf_rd r; if (!(pl[k] = malloc(N)) || mf_rd_open(&r, &I, names[k])) { rc = -1; break; } if (mf_rows(&r, 0, D, pl[k], 4)) rc = -1; mf_rd_close(&r); }
    const int pd = (int)D + 2 * MF_R, ph = (int)H + 2 * MF_R, pw = (int)W + 2 * MF_R; const size_t PN = (size_t)pd * ph * pw;
    uint8_t *b[4] = {0}, *o = rc ? nullptr : malloc(PN); float *fa[3] = {0};
    if (!rc) for (int k = 0; k < 4; k++) {   /* the box with replicated borders */
        if (!(b[k] = malloc(PN))) { rc = -1; break; }
        for (int z = 0; z < pd; z++) for (int y = 0; y < ph; y++) {
            long long gz = z - MF_R, gy = y - MF_R; gz = gz < 0 ? 0 : gz >= D ? D - 1 : gz; gy = gy < 0 ? 0 : gy >= H ? H - 1 : gy;
            const uint8_t *s = pl[k] + ((size_t)gz * H + gy) * W; uint8_t *d = b[k] + ((size_t)z * ph + y) * pw;
            for (int x = 0; x < pw; x++) { long long gx = x - MF_R; gx = gx < 0 ? 0 : gx >= W ? W - 1 : gx; d[x] = s[gx]; }
        }
    }
    for (int k = 0; k < 3 && !rc; k++) if (!(fa[k] = malloc(PN * 4))) rc = -1;
    if (!rc && o) { const long long g0[3] = {-MF_R, -MF_R, -MF_R}, gn[3] = {D, H, W}; mf_block(b[0], b[1], b[2], b[3], pd, ph, pw, g0, gn, &mf_coef_default, o, fa[0], fa[1], fa[2]); }
    for (int k = 0; k < 4; k++) free(b[k]);
    free(o);
    z3 *cz = rc ? nullptr : z3_open(store_open(B->croot), B->ckey, nullptr);
    if (!rc && !cz) { fprintf(stderr, "merge-flag --eval: %s: codes %s/%s: %s\n", B->name, B->croot, B->ckey, z3_error()); rc = -1; }
    const double gtf = rc ? 1 : I.um / atof(B->ckey);   /* label voxels per native voxel */
    const long long ny = H * h, nx = W * h, gy0 = (long long)floor(I.origin[1] * gtf), gx0 = (long long)floor(I.origin[2] * gtf);
    const long long gny = (long long)(ny * gtf), gnx = (long long)(nx * gtf);
    uint8_t *g = rc ? nullptr : malloc((size_t)gny * gnx); const size_t na = ax ? ax->n : 0;
    int *ry = malloc(sizeof(int) * 4 * (ny + nx + 8)), *rx = ry + (ny + nx + 8), *kk = rx + (ny + nx + 8); double *un = malloc(sizeof(double) * (ny + nx + 8) * 2); float *csv = malloc(sizeof(float) * (ny + nx + 8));
    size_t s0 = B->n; double *nn = nullptr; size_t nnn = 0, nnc = 0;
    for (int iz = 0; iz < EV_NZ && !rc; iz++) {
        const int zc = (int)((iz + 0.5) * D / EV_NZ); const long long zn = (long long)zc * h + h / 2;
        const int64_t o3[3] = {(int64_t)((I.origin[0] + zn) * gtf), gy0, gx0}, n3[3] = {1, gny, gnx};
        if (z3_read(cz, o3, n3, g, 4)) { fprintf(stderr, "merge-flag --eval: %s: code read: %s\n", B->name, z3_error()); rc = -1; break; }
        size_t any = 0; for (size_t i = 0; i < (size_t)gny * gnx; i++) { if (g[i] > 252) g[i] = 0; any += g[i] != 0; }
        if (!any || !na) continue;
        const json *a = json_at(ax, (size_t)zn < na ? (size_t)zn : na - 1); const double cy = json_num(json_at(a, 0), 0), cx = json_num(json_at(a, 1), 0);
        const double R0 = hypot((double)ny, (double)nx) + fabs(cy) + fabs(cx);
        for (int t = 0; t < EV_NTH; t++) {
            const double th = t * 2 * M_PI / EV_NTH, sn = sin(th), cs = cos(th); int m = 0;
            for (long long r = 0; r < (long long)R0 && m < ny + nx + 8; r++) {
                const double y = cy + r * sn, x = cx + r * cs;
                if (y >= 0 && x >= 0 && y < ny - 0.5 && x < nx - 0.5) { ry[m] = (int)y; rx[m] = (int)x; un[2 * m] = y; un[2 * m + 1] = x; m++; }
                else if (m) break;
            }
            if (m < 32) continue;
            double last = 0, acc = 0;   /* the phase unwrapped along the ray (cells), turns */
            for (int i = 0; i < m; i++) {
                const size_t c = ((size_t)zc * H + (size_t)(un[2 * i] / h)) * W + (size_t)(un[2 * i + 1] / h);
                const double p = atan2(pl[1][c] - 128.0, pl[0][c] - 128.0) / (2 * M_PI);
                if (i) { double d = p - last; d -= round(d); acc += d; } last = p; un[2 * i] = acc;   /* (y no longer needed: ry) */
            }
            int nc = 0; for (int i = 0; i < m;) {   /* crossings: run middles */
                const long long ly = (long long)(ry[i] * gtf), lx = (long long)(rx[i] * gtf);
                const uint8_t q = g[(size_t)(ly < gny ? ly : gny - 1) * gnx + (lx < gnx ? lx : gnx - 1)];
                if (!q) { i++; continue; }
                int e = i; for (; e < m; e++) { const long long ey = (long long)(ry[e] * gtf), ex = (long long)(rx[e] * gtf); if (!g[(size_t)(ey < gny ? ey : gny - 1) * gnx + (ex < gnx ? ex : gnx - 1)]) break; }
                const int k = (i + e - 1) / 2; const long long ky = (long long)(ry[k] * gtf), kx = (long long)(rx[k] * gtf);
                { const long long cy = ky < gny ? ky : gny - 1, cx = kx < gnx ? kx : gnx - 1; csv[nc] = ev_cos(g, gny, gnx, cy, cx, (int)lrint(8 * gtf), sn, cs); }
                kk[nc++] = k; un[2 * k + 1] = ((g[(size_t)(ky < gny ? ky : gny - 1) * gnx + (kx < gnx ? kx : gnx - 1)] - 1) / 18) % 14; i = e;
            }
            for (int c = 0; c + 1 < nc; c++) {
                const int k0 = kk[c], k1 = kk[c + 1]; const int d = (((int)un[2 * k1 + 1] - (int)un[2 * k0 + 1]) % 14 + 14) % 14;
                if ((d != 1 && d != 13) || k1 - k0 < 2) continue;
                const int km = (k0 + k1) / 2; const size_t pi = ((size_t)(zc + MF_R) * ph + (size_t)(ry[km] / h + MF_R)) * pw + (size_t)(rx[km] / h + MF_R);
                ev_iv v = {.f = {fa[0][pi], fa[1][pi], fa[2][pi], 0.f}, .gap = (float)((k1 - k0) * I.um),
                           .cs = csv[c] < 0 || csv[c + 1] < 0 ? -1.f : 0.5f * (csv[c] + csv[c + 1])};
                {   const size_t ci = ((size_t)zc * H + (size_t)(ry[km] / h)) * W + (size_t)(rx[km] / h); const double sp = pl[2][ci] ? pl[2][ci] * h / 8.0 : 10.0;
                    const double my = ry[km] + 0.5, mx = rx[km] + 0.5, z = (double)(I.origin[0] + zn);
                    const float q[6] = {(float)z, (float)(I.origin[1] + my - 0.5 * sp * sn), (float)(I.origin[2] + mx - 0.5 * sp * cs),
                                        (float)z, (float)(I.origin[1] + my + 0.5 * sp * sn), (float)(I.origin[2] + mx + 0.5 * sp * cs)};
                    memcpy(v.pt, q, sizeof q); }
                ev_push(B, &v);
                if (nnn == nnc) { nnc = nnc ? 2 * nnc : 4096; nn = realloc(nn, nnc * sizeof *nn); }
                nn[nnn++] = (un[2 * k1] - un[2 * k0]) * (d == 1 ? 1 : -1);
            }
        }
    }
    if (!rc && nnn) {   /* the box's phase direction: the median's sign */
        double *t = malloc(nnn * sizeof *t); memcpy(t, nn, nnn * sizeof *t); qsort(t, nnn, sizeof *t, ev_cmpd); const double sg = t[nnn / 2] < 0 ? -1 : 1; free(t);
        size_t nm = 0; for (size_t i = 0; i < nnn; i++) { B->iv[s0 + i].y = nn[i] * sg < 0.5; nm += B->iv[s0 + i].y; }
        B->merged = (double)nm / nnn;
    }
    free(nn); free(ry); free(un); free(csv); free(g); if (cz) z3_close(cz);
    for (int k = 0; k < 4; k++) free(pl[k]);
    for (int k = 0; k < 3; k++) free(fa[k]);
    json_free(I.j); return rc;
}
typedef struct { ev_box *B; int nb; _Atomic int next; } ev_job;
static void *ev_worker(void *a) {
    ev_job *J = a;
    for (int i; (i = atomic_fetch_add(&J->next, 1)) < J->nb;) {
        ev_box *B = &J->B[i]; B->rc = ev_one(B);
        fprintf(stderr, "merge-flag --eval: %s (%s): %zu intervals, merged %.3f%s\n", B->name, B->scroll, B->n, B->merged, B->rc ? " FAILED" : "");
    }
    return nullptr;
}
/* AUC of score s (higher = merged) against y over the selected intervals (ties: average ranks) */

/* ---- merge-flag --sheet-pairs LIST --out PREFIX (dinovol, lead 11-24): matched-distance sheet-identity triplets from the --eval boxes' GT ----
   The --eval rays (EV_NZ z planes, EV_NTH angles from the fields' axis, native voxel steps; GT crossings = middles of labelled code runs,
   wrap = (code - 1) / 18 mod 14). Anchor a = a crossing; adjacent b = the next crossing along the ray when one wrap apart (gap = their
   distance); same s = the crossing of the SAME wrap on the ray rotated by the angle whose arc at a's radius equals the gap (alternately
   + / -), nearest a's radius and within half a gap of it: a point on a's sheet as far away as the adjacent sheet. try9's score per pair:
   the net phase (atan2 pc, ps of the fields' cells) unwrapped along the straight segment, |turns| (same sheet ~ 0, adjacent ~ 1).
   Out: PREFIX.tsv (box scroll gap_um |a-s| |a-b| t9_same t9_adj az ay ax sy sx by bx, absolute native voxels of the fields' level) and
   PREFIX.<scroll>.pts (float32 [n][3][3] z y x of a, s, b per row of that scroll, the order of the tsv's rows of that scroll). At most
   --max-per-box triplets per box (seeded, at most a quarter per GT gap bin 0-50 / 50-100 / 100-200 / >= 200 um). */
typedef struct { double r, y, x; int w; } sp_cross;
static double sp_turns(const uint8_t *pc, const uint8_t *ps, long long H, long long W, int h, int zc, double y0, double x0, double y1, double x1) {
    const double L = hypot(y1 - y0, x1 - x0); const int n = (int)ceil(L) + 1; double acc = 0, last = 0;
    for (int i = 0; i <= n; i++) {
        const double t = (double)i / n, y = y0 + t * (y1 - y0), x = x0 + t * (x1 - x0);
        long long cy = (long long)(y / h), cx = (long long)(x / h); cy = cy < 0 ? 0 : cy >= H ? H - 1 : cy; cx = cx < 0 ? 0 : cx >= W ? W - 1 : cx;
        const size_t c = ((size_t)zc * H + (size_t)cy) * W + (size_t)cx; const double p = atan2(ps[c] - 128.0, pc[c] - 128.0) / (2 * M_PI);
        if (i) { double d = p - last; d -= round(d); acc += d; } last = p;
    }
    return fabs(acc);
}
static int cmd_sheet_pairs(int argc, char **argv, const char *list) {
    const char *out = opt(argc, argv, "--out", nullptr); const int maxb = opt_atoi("--max-per-box", opt(argc, argv, "--max-per-box", "1500"));
    unsigned long long seed = strtoull(opt(argc, argv, "--seed", "1"), nullptr, 10) * 0x9E3779B97F4A7C15ull + 1;
    FILE *f = fopen(list, "r"); if (!f || !out) { fprintf(stderr, "usage: ufsm merge-flag --sheet-pairs LIST (--eval's: name scroll fields codes-root codes-um) --out PREFIX [--max-per-box 1500] [--seed 1]\n"); if (f) fclose(f); return 2; }
    char tp[1400]; snprintf(tp, sizeof tp, "%s.tsv", out); FILE *fo = fopen(tp, "w"); if (!fo) { fprintf(stderr, "sheet-pairs: cannot write %s\n", tp); fclose(f); return 1; }
    fprintf(fo, "# box\tscroll\tgap_um\td_same\td_adj\tt9_same\tt9_adj\taz\tay\tax\tsy\tsx\tby\tbx\n");
    char seen[16][32]; FILE *pf[16] = {0}; int ns = 0, rc = 0; long tot = 0; char ln[4096];
    while (fgets(ln, sizeof ln, f) && !rc) {
        char name[64], scroll[32], fields[1024], croot[1024], ckey[64];
        if (ln[0] == '#' || sscanf(ln, "%63s %31s %1023s %1023s %63s", name, scroll, fields, croot, ckey) != 5) continue;
        int si = 0; while (si < ns && strcmp(seen[si], scroll)) si++;
        if (si == ns) { if (ns == 16) continue; snprintf(seen[ns], 32, "%s", scroll); char pp[1400]; snprintf(pp, sizeof pp, "%s.%s.pts", out, scroll); if (!(pf[ns] = fopen(pp, "wb"))) { rc = 1; break; } ns++; }
        mf_in I; if (mf_open(&I, fields)) { rc = 1; break; }
        const long long D = I.shape[0], H = I.shape[1], W = I.shape[2]; const int h = I.h; const size_t N = (size_t)D * H * W;
        uint8_t *pc = malloc(N), *ps = malloc(N); const char *nm[2] = {"pc", "ps"}; uint8_t *pl[2] = {pc, ps};
        for (int k = 0; k < 2 && !rc; k++) { mf_rd r; if (mf_rd_open(&r, &I, nm[k]) || mf_rows(&r, 0, D, pl[k], 4)) rc = 1; mf_rd_close(&r); }
        z3 *cz = rc ? nullptr : z3_open(store_open(croot), ckey, nullptr);
        if (!rc && !cz) { fprintf(stderr, "sheet-pairs: %s: codes %s/%s: %s\n", name, croot, ckey, z3_error()); rc = 1; }
        const double gtf = I.um / atof(ckey); const long long ny = H * h, nx = W * h, gy0 = (long long)floor(I.origin[1] * gtf), gx0 = (long long)floor(I.origin[2] * gtf);
        const long long gny = (long long)(ny * gtf), gnx = (long long)(nx * gtf); const json *ax = json_get(I.j, "axis"); const size_t na = ax ? ax->n : 0;
        uint8_t *g = malloc((size_t)gny * gnx); const int MC = 4096; sp_cross *cr = malloc(sizeof(sp_cross) * EV_NTH * 64); int *ncr = calloc(EV_NTH, sizeof(int));
        typedef struct { float gap, ds, db, ts, tb, a[3], sp[3], b[3]; } trip; trip *T = nullptr; size_t nt = 0, ct = 0; (void)MC;
        for (int iz = 0; iz < EV_NZ && !rc && na; iz++) {
            const int zc = (int)((iz + 0.5) * D / EV_NZ); const long long zn = (long long)zc * h + h / 2;
            const int64_t o3[3] = {(int64_t)((I.origin[0] + zn) * gtf), gy0, gx0}, n3[3] = {1, gny, gnx};
            if (z3_read(cz, o3, n3, g, 4)) { fprintf(stderr, "sheet-pairs: %s: code read: %s\n", name, z3_error()); rc = 1; break; }
            for (size_t i = 0; i < (size_t)gny * gnx; i++) if (g[i] > 252) g[i] = 0;
            const json *a = json_at(ax, (size_t)zn < na ? (size_t)zn : na - 1); const double cy = json_num(json_at(a, 0), 0), cx = json_num(json_at(a, 1), 0);
            const double R0 = hypot((double)ny, (double)nx) + fabs(cy) + fabs(cx);
            for (int t = 0; t < EV_NTH; t++) {   /* the crossings of each ray */
                const double th = t * 2 * M_PI / EV_NTH, sn = sin(th), cs = cos(th); ncr[t] = 0; long long r = 0;
                while (r < (long long)R0) { const double y = cy + r * sn, x = cx + r * cs; if (y >= 0 && x >= 0 && y < ny - 0.5 && x < nx - 0.5) break; r++; }
                for (; r < (long long)R0 && ncr[t] < 64;) {
                    const double y = cy + r * sn, x = cx + r * cs; if (!(y >= 0 && x >= 0 && y < ny - 0.5 && x < nx - 0.5)) break;
                    long long ly = (long long)(y * gtf), lx = (long long)(x * gtf); ly = ly < gny ? ly : gny - 1; lx = lx < gnx ? lx : gnx - 1;
                    const uint8_t q = g[(size_t)ly * gnx + lx]; if (!q) { r++; continue; }
                    long long e = r; for (;; e++) { const double ey = cy + e * sn, ex = cx + e * cs; if (!(ey >= 0 && ex >= 0 && ey < ny - 0.5 && ex < nx - 0.5)) break;
                        long long my = (long long)(ey * gtf), mx = (long long)(ex * gtf); my = my < gny ? my : gny - 1; mx = mx < gnx ? mx : gnx - 1; if (g[(size_t)my * gnx + mx] != q) break; }
                    const double rm = 0.5 * (r + e - 1); cr[t * 64 + ncr[t]++] = (sp_cross){rm, cy + rm * sn, cx + rm * cs, ((q - 1) / 18) % 14}; r = e;
                }
            }
            for (int t = 0; t < EV_NTH; t++) for (int c = 0; c + 1 < ncr[t]; c++) for (int dir = 0; dir < 2; dir++) {
                const sp_cross *A = &cr[t * 64 + c + dir], *B = &cr[t * 64 + c + 1 - dir]; const int d = ((B->w - A->w) % 14 + 14) % 14;
                const double gap = fabs(B->r - A->r); if ((d != 1 && d != 13) || gap < 2 || A->r < 1) continue;
                int dt = (int)lrint(gap / A->r / (2 * M_PI / EV_NTH)); if (dt < 1) dt = 1; if (dt > EV_NTH / 8) continue;
                const int t2 = ((t + ((c + dir) & 1 ? -dt : dt)) % EV_NTH + EV_NTH) % EV_NTH; const sp_cross *S = nullptr; double best = 0.5 * gap;
                for (int k = 0; k < ncr[t2]; k++) { const sp_cross *C = &cr[t2 * 64 + k]; if (C->w == A->w && fabs(C->r - A->r) < best) { best = fabs(C->r - A->r); S = C; } }
                if (!S) continue;
                if (nt == ct) { ct = ct ? 2 * ct : 4096; T = realloc(T, ct * sizeof *T); }
                const double z = (double)(I.origin[0] + zn);
                T[nt++] = (trip){(float)(gap * I.um), (float)hypot(S->y - A->y, S->x - A->x), (float)gap,
                                 (float)sp_turns(pc, ps, H, W, h, zc, A->y, A->x, S->y, S->x), (float)sp_turns(pc, ps, H, W, h, zc, A->y, A->x, B->y, B->x),
                                 {(float)z, (float)(I.origin[1] + A->y), (float)(I.origin[2] + A->x)}, {(float)z, (float)(I.origin[1] + S->y), (float)(I.origin[2] + S->x)},
                                 {(float)z, (float)(I.origin[1] + B->y), (float)(I.origin[2] + B->x)}};
            }
        }
        size_t keep = nt;
        if (maxb > 0 && nt > (size_t)maxb) {   /* seeded shuffle, then up to maxb / 4 per GT gap bin (0-50, 50-100, 100-200, >= 200 um: the packed bins are rare) */
            for (size_t i = 0; i + 1 < nt; i++) { seed = seed * 6364136223846793005ull + 1442695040888963407ull; const size_t j = i + (size_t)((seed >> 33) % (nt - i)); trip x = T[i]; T[i] = T[j]; T[j] = x; }
            int qb[4] = {0}; keep = 0;
            for (size_t i = 0; i < nt; i++) { const int bn = T[i].gap < 50 ? 0 : T[i].gap < 100 ? 1 : T[i].gap < 200 ? 2 : 3;
                if (qb[bn] < maxb / 4) { qb[bn]++; T[keep++] = T[i]; } }
        }
        for (size_t i = 0; i < keep; i++) { const trip *v = &T[i];
            fprintf(fo, "%s\t%s\t%.1f\t%.2f\t%.2f\t%.4f\t%.4f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n", name, scroll, v->gap, v->ds, v->db, v->ts, v->tb,
                    v->a[0], v->a[1], v->a[2], v->sp[1], v->sp[2], v->b[1], v->b[2]);
            fwrite(v->a, 4, 3, pf[si]); fwrite(v->sp, 4, 3, pf[si]); fwrite(v->b, 4, 3, pf[si]); }
        fprintf(stderr, "sheet-pairs: %s (%s): %zu triplets, kept %zu\n", name, scroll, nt, keep); tot += (long)keep;
        free(T); free(cr); free(ncr); free(g); free(pc); free(ps); if (cz) z3_close(cz); json_free(I.j);
    }
    fclose(f); if (fclose(fo)) rc = 1; for (int k = 0; k < ns; k++) if (pf[k] && fclose(pf[k])) rc = 1;
    fprintf(stderr, "sheet-pairs: %ld triplets -> %s.tsv\n", tot, out);
    return rc;
}
typedef struct { double s; uint8_t y; } ev_sy;
static int ev_cmpsy(const void *a, const void *b) { const double x = ((const ev_sy *)a)->s, y = ((const ev_sy *)b)->s; return x < y ? -1 : x > y; }
static double ev_auc(ev_sy *v, size_t n) {
    qsort(v, n, sizeof *v, ev_cmpsy); double rs = 0; size_t np = 0;
    for (size_t i = 0; i < n;) { size_t j = i; while (j + 1 < n && v[j + 1].s == v[i].s) j++; const double r = (i + j + 2) / 2.0; for (size_t k = i; k <= j; k++) if (v[k].y) { rs += r; np++; } i = j + 1; }
    const size_t nq = n - np; return np && nq ? (rs - np * (np + 1) / 2.0) / ((double)np * nq) : NAN;
}
/* models: score of interval v (higher = merged) */
enum { EM_LOGIT, EM_COEF, EM_R5, EM_AMP, EM_SP, EM_DCOS, EM_N };
static const char *const em_name[EM_N] = {"logistic (fitted on the split's train boxes)", "the rule (--coef / built-in)", "R5 alone (fixed: -R5)", "amp9 alone (-amp9)", "sp9 alone (-sp9)",
                                          "dinovol dcos alone (+dcos)"};
static double em_score(int m, const ev_iv *v, const double *w, const mf_coef *c) {
    switch (m) {
    case EM_LOGIT: return w[0] + w[1] * v->f[0] + w[2] * v->f[1] + w[3] * v->f[2] + w[4] * v->f[3];
    case EM_COEF: return c->b + c->c_amp * v->f[0] + c->c_r * v->f[1] + c->c_sp * v->f[2];
    case EM_R5: return -v->f[1];
    case EM_AMP: return -v->f[0];
    case EM_DCOS: return v->f[3];
    default: return -v->f[2];
    }
}
/* logistic regression on the train boxes' intervals (standardised features, Newton, l2 1e-4); w raw-unit (b, amp, r, sp[, dcos]): ev_nf
   features (3; 4 with --dense-cos) */
static void ev_logit(ev_box *B, int nb, const uint8_t *tr, double w[5]) {
    enum { K = EV_NF + 1 }; const int F = ev_nf, KK = F + 1; double mu[EV_NF] = {0}, sd[EV_NF] = {0}; size_t n = 0;
    for (int b = 0; b < nb; b++) if (tr[b]) for (size_t i = 0; i < B[b].n; i++) { n++; for (int k = 0; k < F; k++) mu[k] += B[b].iv[i].f[k]; }
    for (int k = 0; k < F; k++) mu[k] /= n ? n : 1;
    for (int b = 0; b < nb; b++) if (tr[b]) for (size_t i = 0; i < B[b].n; i++) for (int k = 0; k < F; k++) { const double d = B[b].iv[i].f[k] - mu[k]; sd[k] += d * d; }
    for (int k = 0; k < F; k++) sd[k] = sqrt(sd[k] / (n ? n : 1)) + 1e-9;
    double v[K] = {0};   /* (z1 .. zF, 1) */
    for (int it = 0; it < 40; it++) {
        double g[K] = {0}, Hm[K][K] = {{0}};
        for (int b = 0; b < nb; b++) if (tr[b]) for (size_t i = 0; i < B[b].n; i++) {
            const ev_iv *q = &B[b].iv[i]; double x[K]; for (int k = 0; k < F; k++) x[k] = (q->f[k] - mu[k]) / sd[k]; x[F] = 1;
            double s = 0; for (int k = 0; k < KK; k++) s += v[k] * x[k];
            const double p = 1 / (1 + exp(-s)), e = p - q->y, ww = p * (1 - p);
            for (int k = 0; k < KK; k++) { g[k] += e * x[k]; for (int l = 0; l < KK; l++) Hm[k][l] += ww * x[k] * x[l]; }
        }
        for (int k = 0; k < KK; k++) { g[k] /= n; for (int l = 0; l < KK; l++) Hm[k][l] /= n; if (k < F) { g[k] += 1e-4 * v[k]; Hm[k][k] += 1e-4; } }
        double A[K][K + 1]; for (int k = 0; k < KK; k++) { for (int l = 0; l < KK; l++) A[k][l] = Hm[k][l]; A[k][KK] = g[k]; }
        for (int c = 0; c < KK; c++) {   /* Gauss-Jordan with partial pivoting */
            int p = c; for (int r = c + 1; r < KK; r++) if (fabs(A[r][c]) > fabs(A[p][c])) p = r;
            for (int l = 0; l <= KK; l++) { const double t = A[c][l]; A[c][l] = A[p][l]; A[p][l] = t; }
            if (fabs(A[c][c]) < 1e-15) continue;
            for (int r = 0; r < KK; r++) if (r != c) { const double f = A[r][c] / A[c][c]; for (int l = c; l <= KK; l++) A[r][l] -= f * A[c][l]; }
        }
        for (int k = 0; k < KK; k++) if (fabs(A[k][k]) > 1e-15) v[k] -= A[k][KK] / A[k][k];
    }
    for (int k = 0; k < 5; k++) w[k] = 0;
    w[0] = v[F]; for (int k = 0; k < F; k++) { w[k + 1] = v[k] / sd[k]; w[0] -= v[k] * mu[k] / sd[k]; }
}
/* one report line: model m over the test boxes (te), pooled AUC, the within-box median AUC, P / R at p >= 0.5 for the score models */
static void ev_report(const char *tag, int m, ev_box *B, int nb, const uint8_t *te, const double *w, const mf_coef *c) {
    size_t n = 0; for (int b = 0; b < nb; b++) if (te[b]) n += B[b].n;
    if (!n) return;
    ev_sy *v = malloc(n * sizeof *v); size_t k = 0, tp = 0, fl = 0, pos = 0; double wa[512]; int nw = 0;
    for (int b = 0; b < nb; b++) if (te[b]) {
        const size_t k0 = k;
        for (size_t i = 0; i < B[b].n; i++) { v[k].s = em_score(m, &B[b].iv[i], w, c); v[k].y = B[b].iv[i].y; pos += v[k].y;
                                               if (m <= EM_COEF && v[k].s >= 0) { fl++; tp += v[k].y; } k++; }
        if (B[b].merged > 0.05 && B[b].merged < 0.95 && nw < 512) { ev_sy *u = malloc((k - k0) * sizeof *u); memcpy(u, v + k0, (k - k0) * sizeof *u); wa[nw++] = ev_auc(u, k - k0); free(u); }
    }
    const double a = ev_auc(v, n); qsort(wa, nw, sizeof *wa, ev_cmpd);
    printf("  %-28s %-46s n %8zu merged %.3f AUC %.3f within-box median %.3f (%d boxes)", tag, em_name[m], n, (double)pos / n, a, nw ? wa[nw / 2] : NAN, nw);
    if (m <= EM_COEF) printf(" | p>=0.5 flagged %.3f P %.3f R %.3f", (double)fl / n, fl ? (double)tp / fl : 0, pos ? (double)tp / pos : 0);
    printf("\n"); free(v);
}
/* model m over the test boxes by GT gap bin (0-50-100-200-inf um): AUC, and the recall of merged intervals when the top 6.5 % by score is
   flagged (the unroll's MFLAG operating point: 1667 lost 6.5 % of its cells) */
static void ev_bins(const char *tag, int m, ev_box *B, int nb, const uint8_t *te, const double *w, const mf_coef *c) {
    const double ge[5] = {0, 50, 100, 200, 1e30}; printf("  %-28s %-14s", tag, m == EM_LOGIT ? "logistic" : em_name[m]);
    for (int g = 0; g < 4; g++) {
        size_t n = 0; for (int b = 0; b < nb; b++) if (te[b]) for (size_t i = 0; i < B[b].n; i++) n += B[b].iv[i].gap >= ge[g] && B[b].iv[i].gap < ge[g + 1];
        if (!n) { printf(" | %g-%g: -", ge[g], ge[g + 1] > 1e29 ? INFINITY : ge[g + 1]); continue; }
        ev_sy *v = malloc(n * sizeof *v); size_t k = 0, pos = 0;
        for (int b = 0; b < nb; b++) if (te[b]) for (size_t i = 0; i < B[b].n; i++) { const ev_iv *q = &B[b].iv[i];
            if (q->gap >= ge[g] && q->gap < ge[g + 1]) { v[k].s = em_score(m, q, w, c); v[k].y = q->y; pos += q->y; k++; } }
        const double a = ev_auc(v, n);   /* (sorts v ascending by score) */
        size_t top = (size_t)(0.065 * n), tp = 0; for (size_t i = 0; i < top; i++) tp += v[n - 1 - i].y;
        printf(" | %g-%g n %zu mg %.3f AUC %.3f R@6.5%% %.3f", ge[g], ge[g + 1] > 1e29 ? INFINITY : ge[g + 1], n, (double)pos / n, a, pos ? (double)tp / pos : NAN);
        free(v);
    }
    printf("\n");
}
static int ev_p4(const char *scroll) { return !strncmp(scroll, "paris4", 6); }
/* ufsm merge-flag --eval LIST [--coef b,amp,r,sp] [--threads 4]: LIST lines "name scroll fields-dir codes-root codes-key" (codes key = its
   um, e.g. 9.6 / 4.798). Splits: held-out (train the even boxes of the name-sorted list, test the odd), leave-one-scroll-out (Paris 4 vs
   the rest, both ways); per split the fitted logistic, the given rule and the single features. Prints the logistic fitted on every box. */
static int cmd_merge_flag_eval(int argc, char **argv, const char *list) {
    FILE *f = fopen(list, "r"); if (!f) { fprintf(stderr, "merge-flag --eval: cannot open %s\n", list); return 1; }
    ev_box *B = nullptr; int nb = 0, cap = 0; char ln[4096];
    while (fgets(ln, sizeof ln, f)) {
        if (ln[0] == '#' || ln[0] == '\n') continue;
        if (nb == cap) { cap = cap ? 2 * cap : 64; B = realloc(B, cap * sizeof *B); }
        ev_box *b = &B[nb]; memset(b, 0, sizeof *b);
        if (sscanf(ln, "%63s %31s %1023s %1023s %63s", b->name, b->scroll, b->fields, b->croot, b->ckey) == 5) nb++;
    }
    fclose(f);
    if (!nb) { fprintf(stderr, "merge-flag --eval: no boxes in %s\n", list); return 2; }
    for (int a = 1; a < nb; a++) for (int c = a; c > 0 && strcmp(B[c - 1].name, B[c].name) > 0; c--) { ev_box t = B[c]; B[c] = B[c - 1]; B[c - 1] = t; }   /* (by name: the split) */
    mf_coef c = mf_coef_default; const char *cs = opt(argc, argv, "--coef", nullptr);
    if (cs && sscanf(cs, "%lf,%lf,%lf,%lf", &c.b, &c.c_amp, &c.c_r, &c.c_sp) != 4) { fprintf(stderr, "merge-flag: --coef b,c_amp,c_r,c_sp\n"); return 2; }
    const int nt = opt_atoi("--threads", opt(argc, argv, "--threads", "4"));
    ev_job J = {.B = B, .nb = nb}; pthread_t th[64]; const int m = nt < 1 ? 1 : nt > 64 ? 64 : nt;
    for (int t = 0; t < m; t++) pthread_create(&th[t], nullptr, ev_worker, &J);
    for (int t = 0; t < m; t++) pthread_join(th[t], nullptr);
    size_t tot = 0; int nok = 0; for (int b = 0; b < nb; b++) if (!B[b].rc) { tot += B[b].n; nok++; } else B[b].n = 0;
    printf("merge-flag --eval: %d of %d boxes, %zu GT intervals\n", nok, nb, tot);
    const char *dpts = opt(argc, argv, "--dense-pts", nullptr), *dcos = opt(argc, argv, "--dense-cos", nullptr);
    if (dpts) {   /* the probe pairs per box, DIR/NAME.pts (float32 [2n][3] z y x, absolute native): dinovol-probe pfeats embeds them */
        for (int b = 0; b < nb; b++) if (B[b].n) {
            char fn[1200]; snprintf(fn, sizeof fn, "%s/%s.pts", dpts, B[b].name); FILE *o = fopen(fn, "wb");
            if (!o) { fprintf(stderr, "merge-flag --eval: cannot write %s\n", fn); return 1; }
            for (size_t i = 0; i < B[b].n; i++) fwrite(B[b].iv[i].pt, sizeof(float), 6, o);
            fclose(o);
        }
        printf("merge-flag --eval: probe pairs written to %s/NAME.pts\n", dpts);
        for (int b = 0; b < nb; b++) free(B[b].iv);
        free(B); return 0;
    }
    if (dcos) {   /* f[3] = cos(dense(a), dense(b)) of each interval's probe pair; a box without its DIR/NAME.dvf leaves the run */
        int nd = 0; size_t nx = 0;
        for (int b = 0; b < nb; b++) if (B[b].n) {
            char fn[1200]; snprintf(fn, sizeof fn, "%s/%s.dvf", dcos, B[b].name); long n = 0; int F = 0; float *X = dvp_read_cols(fn, "dense", &n, &F);
            if (!X || n != 2 * (long)B[b].n) { fprintf(stderr, "merge-flag --eval: %s: %s %s (%ld rows, want %zu): box dropped\n", B[b].name, fn, X ? "row count" : "unreadable", n, 2 * B[b].n);
                                              free(X); nx += B[b].n; B[b].n = 0; continue; }
            for (size_t i = 0; i < B[b].n; i++) { const float *a = X + 2 * i * F, *c = a + F; double ab = 0, aa = 0, cc = 0;
                                                  for (int k = 0; k < F; k++) { ab += (double)a[k] * c[k]; aa += (double)a[k] * a[k]; cc += (double)c[k] * c[k]; }
                                                  B[b].iv[i].f[3] = (float)(aa > 0 && cc > 0 ? ab / sqrt(aa * cc) : 0); }
            free(X); nd++;
        }
        ev_nf = 4; printf("merge-flag --eval: --dense-cos %s: %d boxes, %zu intervals dropped (no features)\n", dcos, nd, nx);
    }
    if (opt(argc, argv, "--gap-report", nullptr)) {   /* the rule at --thr against GT sheet gaps, per scroll: P / R of "gap < G um" */
        const double G = opt_atof("--gap-report", opt(argc, argv, "--gap-report", "100")), t = (opt_atoi("--thr", opt(argc, argv, "--thr", "179")) - 0.5) / 255.0;
        const double lt = log(t / (1 - t)); char seen[64][32]; int ns = 0;
        for (int b = 0; b < nb; b++) { int k = 0; while (k < ns && strcmp(seen[k], B[b].scroll)) k++; if (k == ns && ns < 64) snprintf(seen[ns++], 32, "%s", B[b].scroll); }
        for (int k = 0; k < ns; k++) {
            size_t n = 0, fl = 0, sm = 0, tps = 0, mg = 0, tpm = 0; int bx = 0;
            for (int b = 0; b < nb; b++) if (!strcmp(B[b].scroll, seen[k]) && B[b].n) { bx++;
                for (size_t i = 0; i < B[b].n; i++) { const ev_iv *v = &B[b].iv[i]; const int f = em_score(EM_COEF, v, nullptr, &c) >= lt, g = v->gap < G;
                                                       n++; fl += f; sm += g; tps += f && g; mg += v->y; tpm += f && v->y; } }
            printf("gap-report %-10s %3d boxes, %9zu intervals: flagged (mf >= thr) %.3f | gap < %g um %.3f: P %.3f R %.3f | phase-merged %.3f: P %.3f R %.3f\n",
                   seen[k], bx, n, n ? (double)fl / n : 0, G, n ? (double)sm / n : 0, fl ? (double)tps / fl : 0, sm ? (double)tps / sm : 0,
                   n ? (double)mg / n : 0, fl ? (double)tpm / fl : 0, mg ? (double)tpm / mg : 0);
            const double ge[5] = {0, 50, 100, 200, 1e30}; printf("gap-report %-10s phase-merged by GT gap:", seen[k]);
            for (int q = 0; q < 4; q++) { size_t nq = 0, mq = 0;
                for (int b = 0; b < nb; b++) if (!strcmp(B[b].scroll, seen[k])) for (size_t i = 0; i < B[b].n; i++) { const ev_iv *v = &B[b].iv[i];
                    if (v->gap >= ge[q] && v->gap < ge[q + 1]) { nq++; mq += v->y; } }
                printf(" [%g,%g) um %.3f of %.3f%s", ge[q], q < 3 ? ge[q + 1] : INFINITY, nq ? (double)mq / nq : 0, n ? (double)nq / n : 0, q < 3 ? ";" : "\n"); }
            /* along the normal: gap x |cos|; |cos| quantiles; radial >= 200 um misses by normal gap */
            size_t nc = 0; for (int b = 0; b < nb; b++) if (!strcmp(B[b].scroll, seen[k])) for (size_t i = 0; i < B[b].n; i++) nc += B[b].iv[i].cs >= 0;
            double *cv = malloc((nc + 1) * sizeof *cv); size_t ci = 0, nb2[4] = {0}, mb2[4] = {0}, rm[3] = {0};
            const double gn[5] = {0, 50, 100, 150, 1e30};
            for (int b = 0; b < nb; b++) if (!strcmp(B[b].scroll, seen[k])) for (size_t i = 0; i < B[b].n; i++) { const ev_iv *v = &B[b].iv[i]; if (v->cs < 0) continue;
                cv[ci++] = v->cs; const double ng = v->gap * v->cs; int q = 0; while (q < 3 && ng >= gn[q + 1]) q++; nb2[q]++; mb2[q] += v->y;
                if (v->y && v->gap >= 200) rm[ng < 100 ? 0 : ng < 150 ? 1 : 2]++; }
            qsort(cv, ci, sizeof *cv, ev_cmpd);
            printf("gap-report %-10s |cos ray,normal| (%zu of %zu with a normal): p10 %.2f p25 %.2f median %.2f p75 %.2f p90 %.2f\n", seen[k], ci, n,
                   ci ? cv[ci / 10] : 0, ci ? cv[ci / 4] : 0, ci ? cv[ci / 2] : 0, ci ? cv[3 * ci / 4] : 0, ci ? cv[9 * ci / 10] : 0);
            printf("gap-report %-10s phase-merged by NORMAL gap:", seen[k]);
            for (int q = 0; q < 4; q++) printf(" [%g,%g) um %.3f of %.3f%s", gn[q], q < 3 ? gn[q + 1] : INFINITY, nb2[q] ? (double)mb2[q] / nb2[q] : 0, ci ? (double)nb2[q] / ci : 0, q < 3 ? ";" : "\n");
            const size_t rt = rm[0] + rm[1] + rm[2];
            printf("gap-report %-10s misses with radial gap >= 200 um (%zu): normal gap < 100 um %.3f, 100-150 %.3f, >= 150 %.3f\n", seen[k], rt,
                   rt ? (double)rm[0] / rt : 0, rt ? (double)rm[1] / rt : 0, rt ? (double)rm[2] / rt : 0);
            free(cv);
        }
        for (int b = 0; b < nb; b++) free(B[b].iv);
        free(B); return nok == nb ? 0 : 1;
    }
    uint8_t *tr = calloc(nb, 1), *te = calloc(nb, 1); double w[5] = {0};
    /* a box listed for several nets (name NET-BOX) stays on one side of the held-out split: its rank among the distinct BOX names */
    int *rk = calloc(nb, sizeof *rk);
    for (int b = 0; b < nb; b++) {
        const char *bn = strchr(B[b].name, '-') ? strchr(B[b].name, '-') + 1 : B[b].name;
        for (int q = 0; q < nb; q++) {
            const char *qn = strchr(B[q].name, '-') ? strchr(B[q].name, '-') + 1 : B[q].name; int first = 1;
            for (int r = 0; r < q && first; r++) first = strcmp(strchr(B[r].name, '-') ? strchr(B[r].name, '-') + 1 : B[r].name, qn) != 0;
            rk[b] += first && strcmp(qn, bn) < 0;
        }
    }
    for (int split = 0; split < 3; split++) {
        for (int b = 0; b < nb; b++) {
            if (split == 0) tr[b] = rk[b] % 2 == 0; else if (split == 1) tr[b] = ev_p4(B[b].scroll); else tr[b] = !ev_p4(B[b].scroll);
            te[b] = !tr[b];
        }
        const char *sn[3] = {"held-out boxes (even -> odd)", "scroll-out: Paris 4 -> rest", "scroll-out: rest -> Paris 4"};
        ev_logit(B, nb, tr, w);
        printf("%s: logistic b %.4f c_amp %.4f c_r %.4f c_sp %.4f c_dcos %.4f\n", sn[split], w[0], w[1], w[2], w[3], w[4]);
        for (int mm = 0; mm < EM_N - (ev_nf < 4); mm++) ev_report("all test boxes", mm, B, nb, te, w, &c);
        ev_bins("by GT gap", EM_LOGIT, B, nb, te, w, &c);
        if (ev_nf == 4) ev_bins("by GT gap", EM_DCOS, B, nb, te, w, &c);
        if (split == 0) {   /* per scroll of the held-out boxes */
            uint8_t *t2 = calloc(nb, 1);
            for (int p = 1; p >= 0; p--) { for (int b = 0; b < nb; b++) t2[b] = te[b] && ev_p4(B[b].scroll) == p;
                                            for (int mm = 0; mm < 3; mm++) ev_report(p ? "test boxes, Paris 4" : "test boxes, rest", mm, B, nb, t2, w, &c); }
            free(t2);
        }
    }
    for (int b = 0; b < nb; b++) tr[b] = 1;
    ev_logit(B, nb, tr, w);
    printf("merge-flag --eval: logistic on every box: --coef %.4f,%.4f,%.4f,%.4f\n", w[0], w[1], w[2], w[3]);
    for (int b = 0; b < nb; b++) free(B[b].iv);
    free(B); free(tr); free(te); free(rk);
    return nok == nb ? 0 : 1;
}

/* ---- merge-flag --apply FIELDS OUT --mf-w K [--mf-cut T]: a fields set whose evidence in merged cells is down-weighted, for the
   global counting (count-field's ref) and the slab fits: per cell f = 1 - K mf / 255 (--mf-cut T: f = 0 where mf >= T, else 1);
   the phase confidence pc, ps scaled toward 128 by f and the material w by f (kept >= 1 where it was > 0: still material, but its
   density / phase weight in count-field's least squares (w / 255) and the fitter's terms falls). OUT holds links to every other plane
   of FIELDS (and its mf); pc ps w are re-encoded at their source volcomp q. Counts then propagate around merged regions instead of
   through them. */
static float mf_src_q(const char *dir, const char *name) {   /* the volcomp q of a zarr plane (0: lossless / none found) */
    char p[1500]; snprintf(p, sizeof p, "%s/%s/zarr.json", dir, name); json *j = mf_json(p); if (!j) return 0;
    float q = 0; const json *cs = json_get(j, "codecs");
    for (size_t i = 0; cs && i < cs->n; i++) {
        const json *c = json_at(cs, i); const json *in = json_get(json_get(c, "configuration"), "codecs");
        for (size_t k = 0; in && k < in->n; k++) if (!strcmp(json_str(json_get(json_at(in, k), "name"), ""), "volcomp")) q = (float)json_num(json_get(json_get(json_at(in, k), "configuration"), "q"), 0);
        if (!strcmp(json_str(json_get(c, "name"), ""), "volcomp")) q = (float)json_num(json_get(json_get(c, "configuration"), "q"), 0);
    }
    json_free(j); return q;
}
static int mf_link_dir(const char *src, const char *out) {   /* OUT: links to every entry of SRC but pc ps w, fields.json copied */
    if (mkdir(out, 0775) && errno != EEXIST) { fprintf(stderr, "merge-flag: cannot create %s\n", out); return 1; }
    char p[1500], q[1500]; DIR *dd = opendir(src); if (!dd) return 1;
    for (struct dirent *e; (e = readdir(dd));) {
        if (e->d_name[0] == '.' || !strcmp(e->d_name, "pc") || !strcmp(e->d_name, "ps") || !strcmp(e->d_name, "w") || !strcmp(e->d_name, "fields.json")) continue;
        snprintf(p, sizeof p, "%s/%s", src, e->d_name); char rp[PATH_MAX]; if (!realpath(p, rp)) continue;
        snprintf(q, sizeof q, "%s/%s", out, e->d_name); unlink(q); if (symlink(rp, q)) { fprintf(stderr, "merge-flag: cannot link %s\n", q); closedir(dd); return 1; }
    }
    closedir(dd);
    snprintf(p, sizeof p, "%s/fields.json", src); snprintf(q, sizeof q, "%s/fields.json", out); FILE *a = fopen(p, "rb"), *b = fopen(q, "wb"); char buf[1 << 16]; size_t n;
    if (!a || !b) { if (a) fclose(a); if (b) fclose(b); fprintf(stderr, "merge-flag: cannot copy fields.json\n"); return 1; }
    while ((n = fread(buf, 1, sizeof buf, a)) > 0) fwrite(buf, 1, n, b);
    fclose(a); return fclose(b) ? 1 : 0;
}
static int mf_weight_create(const mf_in *I, const char *src, const char *out, double K, int cut, z3w *zw[3]) {   /* OUT's pc ps w writers, at the source's q */
    static const char *const nm[3] = {"pc", "ps", "w"}; const int64_t sh3[3] = {I->shape[0], I->shape[1], I->shape[2]}; char p[1500], at[256];
    for (int k = 0; k < 3; k++) {
        snprintf(at, sizeof at, "{\"ufsm\":{\"content\":\"winding fitter fields (src/fit.h), merge-flag weighted: %s x (1 - %g mf / 255)%s\",\"h\":%d}}", nm[k], K, cut ? " (cut)" : "", I->h);
        snprintf(p, sizeof p, "%s/%s", out, nm[k]);
        if (!(zw[k] = z3w_create(p, sh3, SZ, mf_src_q(src, nm[k]), k < 2 ? 128 : 0, at))) { fprintf(stderr, "merge-flag: %s\n", z3w_error()); return 1; }
    }
    return 0;
}
static int cmd_merge_flag_apply(int argc, char **argv, const char *src, const char *out) {
    mf_in I; if (mf_open(&I, src)) return 1;
    if (!I.zarr || I.off[0] || I.off[1] || I.off[2]) { fprintf(stderr, "merge-flag --apply: zarr fields (not a crop) only\n"); json_free(I.j); return 2; }
    const double K = opt_atof("--mf-w", opt(argc, argv, "--mf-w", "1")); const int cut = opt_atoi("--mf-cut", opt(argc, argv, "--mf-cut", "0"));
    const int nt = ufsm_threads(opt_atoi("--threads", opt(argc, argv, "--threads", "0")), 16);
    z3w *zw[3]; if (mf_link_dir(src, out) || mf_weight_create(&I, src, out, K, cut, zw)) return 1;
    const char *names[4] = {"pc", "ps", "w", "mf"}; mf_rd rd[4]; for (int k = 0; k < 4; k++) if (mf_rd_open(&rd[k], &I, names[k])) return 1;
    const long long D = I.shape[0], H = I.shape[1], W = I.shape[2];
    const size_t rows = (size_t)SZ * H * W; uint8_t *pl[4]; for (int k = 0; k < 4; k++) if (!(pl[k] = malloc(rows))) return 1;
    uint8_t *sb = malloc((size_t)SZ * SZ * SZ); int rc = 0; size_t nmat = 0; double fsum = 0;
    for (long long z0 = 0; z0 < D && !rc; z0 += SZ) {
        const long long nz = z0 + SZ < D ? SZ : D - z0;
        for (int k = 0; k < 4 && !rc; k++) if (mf_rows(&rd[k], z0, nz, pl[k], nt)) { fprintf(stderr, "merge-flag --apply: cannot read %s\n", names[k]); rc = 1; }
        if (rc) break;
        mf_weight_rows(pl[0], pl[1], pl[2], pl[3], (size_t)nz * H * W, K, cut, &nmat, &fsum);
        for (int k = 0; k < 3 && !rc; k++) if (mf_write_par(zw[k], pl[k], z0, nz, H, W, k < 2 ? 128 : 0, nt)) { fprintf(stderr, "merge-flag --apply: %s\n", z3w_error()); rc = 1; }
    }
    for (int k = 0; k < 3; k++) if (z3w_close(zw[k])) rc = 1;
    for (int k = 0; k < 4; k++) { free(pl[k]); mf_rd_close(&rd[k]); }
    free(sb); json_free(I.j);
    if (!rc) printf("merge-flag --apply: %s -> %s: mean evidence factor %.4f over %zu material cells (%s)\n", src, out, nmat ? fsum / nmat : 0, nmat, cut ? "cut" : "linear");
    return rc;
}
