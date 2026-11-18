/* ufsm merge-flag / wrap-mf (src/mergeflag.h) */
#include "mergeflag.h"
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

/* fitted on the wbench boxes' GT intervals (u1 13790 + try9 fields; DESIGN 2026-11-18 "merge-flag"): raw-unit coefficients */
const mf_coef mf_coef_default = {.b = 4.6911, .c_amp = -2.8313, .c_r = -6.6699, .c_sp = -0.1483};
const int mf_thr_default = 128;

enum { MF_R = 4 };
static const char *opt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 3; i + 1 < argc; i++) if (!strcmp(argv[i], opt_n(name))) return argv[i + 1];
    return dflt;
}   /* block border (cells): the 9-box */

/* in-place box mean of edge length 2 r + 1 along each axis, valid on [r, n - r) per axis (the rest left as it was); double sums */
static void box3(float *a, int D, int H, int W, int r, double *line, float *tmp) {
    const double s = 1.0 / (2 * r + 1);
    for (int z = 0; z < D; z++) for (int y = 0; y < H; y++) {   /* x */
        float *p = a + ((size_t)z * H + y) * W; line[0] = 0; for (int x = 0; x < W; x++) line[x + 1] = line[x] + p[x];
        for (int x = r; x < W - r; x++) tmp[x] = (float)((line[x + r + 1] - line[x - r]) * s);
        for (int x = r; x < W - r; x++) p[x] = tmp[x];
    }
    for (int z = 0; z < D; z++) for (int x = r; x < W - r; x++) {   /* y */
        float *p = a + (size_t)z * H * W + x; line[0] = 0; for (int y = 0; y < H; y++) line[y + 1] = line[y] + p[(size_t)y * W];
        for (int y = r; y < H - r; y++) tmp[y] = (float)((line[y + r + 1] - line[y - r]) * s);
        for (int y = r; y < H - r; y++) p[(size_t)y * W] = tmp[y];
    }
    const size_t HW = (size_t)H * W;
    for (int y = r; y < H - r; y++) for (int x = r; x < W - r; x++) {   /* z */
        float *p = a + (size_t)y * W + x; line[0] = 0; for (int z = 0; z < D; z++) line[z + 1] = line[z] + p[z * HW];
        for (int z = r; z < D - r; z++) tmp[z] = (float)((line[z + r + 1] - line[z - r]) * s);
        for (int z = r; z < D - r; z++) p[z * HW] = tmp[z];
    }
}

void mf_block(const uint8_t *pc, const uint8_t *ps, const uint8_t *sp, const uint8_t *w, int D, int H, int W, const long long g0[3],
              const long long gn[3], const mf_coef *c, uint8_t *out, float *amp9o, float *r5o, float *sp9o) {
    const size_t N = (size_t)D * H * W, HW = (size_t)H * W; const int n3[3] = {D, H, W}; const size_t st[3] = {HW, (size_t)W, 1};
    float *am = malloc(N * 4), *ma = malloc(N * 4), *s5 = malloc(N * 4), *k5 = malloc(N * 4), *s9 = malloc(N * 4), *m9 = malloc(N * 4);
    float *cr = malloc(N * 4), *ci = malloc(N * 4), *ka = calloc(N, 4);
    const int L = D > H ? (D > W ? D : W) : (H > W ? H : W); double *line = malloc((L + 1) * sizeof *line); float *tmp = malloc(L * 4);
    if (!am || !ma || !s5 || !k5 || !s9 || !m9 || !cr || !ci || !ka || !line || !tmp) abort();
    for (size_t i = 0; i < N; i++) {
        const float zr = (pc[i] - 128) / 127.0f, zi = (ps[i] - 128) / 127.0f, m = w[i] > 0;
        am[i] = sqrtf(zr * zr + zi * zi) * m; ma[i] = m; s5[i] = s9[i] = sp[i] / 8.0f; k5[i] = m9[i] = sp[i] > 0;
    }
    box3(am, D, H, W, 4, line, tmp); box3(ma, D, H, W, 4, line, tmp);   /* amp9 = box9(amp w>0) / box9(w>0) */
    box3(s9, D, H, W, 4, line, tmp); box3(m9, D, H, W, 4, line, tmp);   /* sp9 */
    box3(s5, D, H, W, 2, line, tmp); box3(k5, D, H, W, 2, line, tmp);   /* sp5 (k5 holds its known-count until the axis loop) */
    for (size_t i = 0; i < N; i++) s5[i] = s5[i] / fmaxf(k5[i], 1e-3f);
    for (int a = 0; a < 3; a++) {   /* c_a = z(g' + e_a) conj z(g'), g' = min(g, N_a - 2) (the last cell repeats the one before) */
        for (int z = 0; z < D; z++) for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
            const int q[3] = {z, y, x}; const size_t i = (size_t)z * HW + (size_t)y * W + x;
            long long g = g0[a] + q[a]; if (g < 0) g = 0; if (g > gn[a] - 2) g = gn[a] - 2; if (g < 0) g = 0;
            const long long l = g - g0[a];   /* the local index of g' (inside the block for every cell the 5-box reads) */
            if (l < 0 || l + 1 >= n3[a]) { cr[i] = ci[i] = 0; continue; }
            const size_t j0 = i + (size_t)(l - q[a]) * st[a], j1 = j0 + st[a];
            if (gn[a] < 2) { cr[i] = ci[i] = 0; continue; }
            const float ar = (pc[j0] - 128) / 127.0f, ai = (ps[j0] - 128) / 127.0f, br = (pc[j1] - 128) / 127.0f, bi = (ps[j1] - 128) / 127.0f;
            cr[i] = br * ar + bi * ai; ci[i] = bi * ar - br * ai;   /* b conj(a) */
        }
        box3(cr, D, H, W, 2, line, tmp); box3(ci, D, H, W, 2, line, tmp);
        for (size_t i = 0; i < N; i++) { const float t = atan2f(ci[i], cr[i]) / (float)(2 * M_PI); ka[i] += t * t; }
    }
    for (int z = 0; z < D; z++) for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        const size_t i = (size_t)z * HW + (size_t)y * W + x;
        if (z < MF_R || y < MF_R || x < MF_R || z >= D - MF_R || y >= H - MF_R || x >= W - MF_R) { out[i] = 0; continue; }
        const float amp9 = am[i] / fmaxf(ma[i], 1e-3f), sp9 = s9[i] / fmaxf(m9[i], 1e-3f), r5 = sqrtf(ka[i]) * s5[i];
        if (amp9o) amp9o[i] = amp9; if (r5o) r5o[i] = r5; if (sp9o) sp9o[i] = sp9;
        if (!w[i]) { out[i] = 0; continue; }
        const double lg = c->b + c->c_amp * amp9 + c->c_r * r5 + c->c_sp * sp9;
        out[i] = (uint8_t)lrint(255.0 / (1.0 + exp(-lg)));
    }
    free(am); free(ma); free(s5); free(k5); free(s9); free(m9); free(cr); free(ci); free(ka); free(line); free(tmp);
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
    const uint8_t *pl[4]; uint8_t *outrows; z3w *zw; int ny, nx; _Atomic int next, fail; _Atomic long long nmat, nflag; int thr;
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
        }
        atomic_fetch_add(&J->nmat, nm); atomic_fetch_add(&J->nflag, nf);
        if (sh && z3w_write_shard(J->zw, J->z0 / SZ, y0 / TY, x0 / TY, sh, 1)) atomic_store(&J->fail, 1);
    }
    for (int k = 0; k < 4; k++) free(b[k]);
    free(o); free(sh); return nullptr;
}

/* ufsm merge-flag <fields-dir> [--out DIR] [--coef b,amp,r,sp] [--thr T] [--threads N] */
int cmd_merge_flag(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: ufsm merge-flag <fields-dir> [--out DIR] [--coef b,c_amp,c_r,c_sp] [--thr 128] [--threads N]\n"
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
    const int thr = opt_atoi("--thr", opt(argc, argv, "--thr", "128")), nt = ufsm_threads(opt_atoi("--threads", opt(argc, argv, "--threads", "0")), 16);
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
    long long nmat = 0, nflag = 0; int rc = 0;
    for (long long z0 = 0; z0 < D && !rc; z0 += SZ) {
        const long long nz = z0 + SZ < D ? SZ : D - z0, rz0 = z0 - MF_R < 0 ? 0 : z0 - MF_R, rz1 = z0 + nz + MF_R > D ? D : z0 + nz + MF_R;
        for (int k = 0; k < 4 && !rc; k++) if (mf_rows(&rd[k], rz0, rz1 - rz0, pl[k], nt)) { fprintf(stderr, "merge-flag: cannot read %s rows %lld..%lld\n", names[k], rz0, rz1); rc = 1; }
        if (rc) break;
        mf_job J = {.I = &I, .c = &c, .z0 = z0, .nz = nz, .rz0 = rz0, .rnz = rz1 - rz0, .pl = {pl[0], pl[1], pl[2], pl[3]}, .outrows = orows, .zw = zw,
                    .ny = (int)((H + TY - 1) / TY), .nx = (int)((W + TY - 1) / TY), .thr = thr};
        pthread_t th[64]; const int m = nt < 64 ? nt : 64;
        for (int t = 0; t < m; t++) pthread_create(&th[t], nullptr, mf_worker, &J);
        for (int t = 0; t < m; t++) pthread_join(th[t], nullptr);
        if (atomic_load(&J.fail)) { fprintf(stderr, "merge-flag: shard write failed: %s\n", z3w_error()); rc = 1; }
        if (fo && fwrite(orows, 1, (size_t)nz * H * W, fo) != (size_t)(nz * H * W)) { fprintf(stderr, "merge-flag: write failed\n"); rc = 1; }
        nmat += atomic_load(&J.nmat); nflag += atomic_load(&J.nflag);
    }
    for (int k = 0; k < 4; k++) { free(pl[k]); mf_rd_close(&rd[k]); }
    free(orows);
    if (zw && z3w_close(zw)) { fprintf(stderr, "merge-flag: %s\n", z3w_error()); rc = 1; }
    if (fo && fclose(fo)) rc = 1;
    if (!rc) printf("merge-flag: %lld x %lld x %lld cells, %lld with material, flagged (mf >= %d) %.4f -> %s\n", D, H, W, nmat, thr, nmat ? (double)nflag / nmat : 0, p);
    json_free(I.j);
    return rc;
}
