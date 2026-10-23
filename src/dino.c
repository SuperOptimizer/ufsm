/* ufsm dino: our own DINO-style embedding training from scratch (ufsm dino-train), its inference (ufsm dino-embed) and the
   view-pair data tool (ufsm halves). See DESIGN.md "Own dense embedding net (dino-train)". */
#include "dino.h"
#include "dinor.h"
#include "band.h"
#include "attn.h"
#include "dinovol.h"
#include "store.h"
#include "zarr3.h"
#include "z3w.h"
#include "zarr2.h"
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

static double dn_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static const char *opt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return dflt;
}

/* ---- ufsm halves: two 4x-pooled views of a fine scan from disjoint sub-voxel sets (Neighbor2Neighbor-style noise-independent pair):
   each 4^3 block of the level-0 scan splits by the parity of (dz + dy + dx) into two 32-voxel halves; a = the even half's mean, b = the
   odd half's, full = the 64-voxel mean. Written as uint8 volcomp arrays (lossless) of the box, at the 4x level's grid ---- */
int cmd_dino_halves(int argc, char **argv) {
    long long bv[6];
    if (argc < 5 || !opt(argc, argv, "--box", nullptr) || sscanf(opt(argc, argv, "--box", ""), "%lld,%lld,%lld,%lld,%lld,%lld", &bv[0], &bv[1], &bv[2], &bv[3], &bv[4], &bv[5]) != 6) {
        fprintf(stderr, "usage: ufsm halves <root> <level-0-array-key> <out-dir> --box z,y,x,nz,ny,nx (4x-level voxels) [--zarr2 1 (raw upstream zarr v2)] [--threads 8] [--slab 16]\n"
                        "  out-dir/{a,b,full}: uint8 arrays of the box (a / b: the even / odd (dz + dy + dx) halves of every 4^3 block, 32 voxels each;\n"
                        "  full: all 64). Attributes record the box origin. Two views of the same tissue with independent scan noise.\n");
        return 2;
    }
    const char *root = argv[2], *key = argv[3], *out = argv[4];
    const int nthreads = atoi(opt(argc, argv, "--threads", "8")), slab = atoi(opt(argc, argv, "--slab", "16"));
    store *st = store_open(root); const int v2 = atoi(opt(argc, argv, "--zarr2", "0"));   /* the raw upstream zarr v2 (uncompressed: the lossy volcomp levels are denoised) */
    z3 *ct = st && !v2 ? z3_open(st, key, opt(argc, argv, "--cache", nullptr)) : nullptr; z2 *ct2 = st && v2 ? z2_open(st, key, nullptr) : nullptr;
    if (!ct && !ct2) { fprintf(stderr, "halves: cannot open %s %s: %s\n", root, key, v2 ? z2_error() : z3_error()); return 1; }
    if (ct2) z2_assume_present(ct2, 1);
    const int64_t nz = bv[3], ny = bv[4], nx = bv[5];
    uint8_t *A = calloc((size_t)nz * ny * nx, 1), *B = calloc((size_t)nz * ny * nx, 1), *F = calloc((size_t)nz * ny * nx, 1);
    uint8_t *buf = malloc((size_t)slab * 4 * ny * 4 * nx * 4);
    if (!A || !B || !F || !buf) { fprintf(stderr, "halves: out of memory\n"); return 1; }
    const double t0 = dn_now();
    for (int64_t z0 = 0; z0 < nz; z0 += slab) {
        const int sz = (int)(nz - z0 < slab ? nz - z0 : slab);
        const int64_t o[3] = {(bv[0] + z0) * 4, bv[1] * 4, bv[2] * 4}, n[3] = {(int64_t)sz * 4, ny * 4, nx * 4};
        if (ct2 ? z2_read(ct2, o, n, buf, nthreads) : z3_read(ct, o, n, buf, nthreads)) { fprintf(stderr, "halves: read failed: %s\n", ct2 ? z2_error() : z3_error()); return 1; }
        for (int z = 0; z < sz; z++)
            for (int64_t y = 0; y < ny; y++)
                for (int64_t x = 0; x < nx; x++) {
                    int sa = 0, sb = 0;
                    for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++) {
                        const uint8_t *r = buf + (((size_t)(z * 4 + a) * ny * 4) + y * 4 + b) * (nx * 4) + x * 4;
                        for (int c = 0; c < 4; c++) { if ((a + b + c) & 1) sb += r[c]; else sa += r[c]; }
                    }
                    const size_t i = ((size_t)(z0 + z) * ny + y) * nx + x;
                    A[i] = (uint8_t)((sa + 16) / 32); B[i] = (uint8_t)((sb + 16) / 32); F[i] = (uint8_t)((sa + sb + 32) / 64);
                }
        fprintf(stderr, "\rhalves: %lld / %lld planes (%.0f s)   ", (long long)(z0 + sz), (long long)nz, dn_now() - t0);
    }
    fprintf(stderr, "\n");
    char attrs[512]; snprintf(attrs, sizeof attrs, "{\"ufsm\":{\"content\":\"4x-pooled sub-voxel halves of %s %s\",\"box_origin_zyx\":[%lld,%lld,%lld]}}", root, key, bv[0], bv[1], bv[2]);
    const char *names[3] = {"a", "b", "full"}; uint8_t *arr[3] = {A, B, F};
    const int64_t shp[3] = {nz, ny, nx}; const int S = 512;
    uint8_t *sh = malloc((size_t)S * S * S);
    for (int k = 0; k < 3; k++) {
        char d[1200]; snprintf(d, sizeof d, "%s/%s", out, names[k]);
        z3w *w = z3w_create(d, shp, S, 0.f, 0, attrs); if (!w) { fprintf(stderr, "halves: %s\n", z3w_error()); return 1; }
        for (int64_t sz = 0; sz * S < nz; sz++) for (int64_t sy = 0; sy * S < ny; sy++) for (int64_t sx = 0; sx * S < nx; sx++) {
            memset(sh, 0, (size_t)S * S * S);
            for (int64_t z = 0; z < S && sz * S + z < nz; z++) for (int64_t y = 0; y < S && sy * S + y < ny; y++) {
                const int64_t w0 = nx - sx * S < S ? nx - sx * S : S;
                memcpy(sh + ((size_t)z * S + y) * S, arr[k] + ((size_t)(sz * S + z) * ny + sy * S + y) * nx + sx * S, (size_t)w0);
            }
            if (z3w_write_shard(w, sz, sy, sx, sh, nthreads)) { fprintf(stderr, "halves: write: %s\n", z3w_error()); return 1; }
        }
        if (z3w_close(w)) { fprintf(stderr, "halves: close: %s\n", z3w_error()); return 1; }
    }
    /* noise check: the a - b difference is pure noise (same tissue); its std / sqrt(2) is the per-half noise */
    double s = 0, s2 = 0, d2 = 0; const size_t nv = (size_t)nz * ny * nx; size_t cnt = 0;
    for (size_t i = 0; i < nv; i++) if (F[i]) { const double d = (double)A[i] - B[i]; s += F[i]; s2 += (double)F[i] * F[i]; d2 += d * d; cnt++; }
    if (cnt) fprintf(stderr, "halves: %zu nonzero voxels, full mean %.2f std %.2f, (a - b) rms %.2f -> per-half noise ~%.2f\n", cnt, s / cnt,
                     sqrt(fmax(s2 / cnt - (s / cnt) * (s / cnt), 0)), sqrt(d2 / cnt), sqrt(d2 / cnt / 2));
    free(A); free(B); free(F); free(buf); free(sh); if (ct) z3_close(ct); if (ct2) z2_close(ct2); store_close(st);
    return 0;
}

/* =============================================================================================================================== */
/* ufsm dino-train                                                                                                                 */
/* =============================================================================================================================== */
#include "json.h"
#include "unet.h"
#include <unistd.h>
#include <signal.h>

enum { DN_MAXSRC = 128, DN_MAXBOX = 256, DN_SLOTS = 6 };
typedef struct {
    char name[64]; z3 *ca, *cb, *codes, *ink, *phase; int halves; int cscale, pscale; float um;   /* phase: the stored winding phase (PHASE_* bytes; sources "phase": {"root", "key", "scale"}) */ float *axz;   /* scroll axis (y, x) per CT plane (sources "axis" + "axis_scale"), or nullptr */   /* CT voxels per code voxel (codes on a coarser grid; 1: the same grid) */
    int64_t org[3];                              /* added to window corners (ct array coordinates) to address the codes (global grid) */
    int nbox; int64_t box[DN_MAXBOX][6]; int nho; int64_t ho[DN_MAXBOX][6];
    double weight, bvol; float center, scale, pap; int64_t shape[3];
    int ct_only;                                 /* unlabelled (U0): SSL terms only (sampled by the ct_only share) */
    int32_t *mcell; int64_t nmcell; int mcs;     /* "mask": {"map": RAW, "min": m} (ufsm scanprep): the cells with code >= m (window centres), cell edge */
} dn_src;
typedef struct { int P, ngt, nssl, nair, margin; int nsrc; dn_src src[DN_MAXSRC]; double wsum; int ident; int nink; double wsum_l, wsum_u, share_u; char excl[8][64]; int nexcl;
    /* recipe views drawn by the loader workers: globals vP^3 (zoom zlo..zhi), vnloc locals vPL^3 (lzlo..lzhi); uw: per global view the stored phase
       resampled into the view + its winding mod 4 (band_wind4_tiled); ikp: the ink-splat probability resampled into the view */
    int vP, vPL, vnloc, uw, ikp; float zlo, zhi, lzlo, lzhi; } dn_data;   /* share_u: the unlabelled sources' share of the draws (< 0: by weight over all; 0: labelled only) */   /* nink: ink points per sample (half ink, half no-ink) */   /* ident: no window geometry (the recipe draws its own views) */
typedef struct {   /* one sample: two CT views, the geometry, the points (output-space voxel indices) */
    uint8_t *ca, *cb; int si; int64_t corner[3]; dn_geom g; int halves; float *ayx;   /* the axis (y, x) per window plane (window coords; --axis-input) */
    int ngt, nssl; int *gidx, *gcode, *gxyz, *sidx;
    int nip, nin; int *ixyz;   /* ink points (window coords, identity geometry): nip ink first, then nin no-ink (sources "ink", the recipe) */
    dr_xf xf[18]; int nxf;     /* the recipe's views (worker-drawn) */
    uint8_t *pv[2], *w4[2], *wc[2], *iv[2], *im[2], *ikw, *phw; int uwv[2], ikv;   /* per global view: phase, winding mod 4, component, ink target / mask (recipe) */
    int ready;
} dn_sample;
typedef struct {
    dn_data *D; dn_sample slot[DN_SLOTS]; pthread_mutex_t mu; pthread_cond_t cv; long produced, consumed; int stop; uint64_t seed; int nthreads;
    pthread_t th[8]; int nworkers; long next_id;
} dn_ring;

static uint64_t rng_next(uint64_t *s) { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; return *s; }
static double rng_u(uint64_t *s) { return (rng_next(s) >> 11) * (1.0 / 9007199254740992.0); }
static int box_hit(const int64_t a[6], const int64_t o[3], int P) {   /* the window [o, o + P)^3 intersects box a */
    for (int d = 0; d < 3; d++) if (o[d] + P <= a[d] || o[d] >= a[d] + a[3 + d]) return 0;
    return 1;
}
/* output index of source voxel s under g: o[d] = flip[d] ? P - 1 - s[perm[d]] : s[perm[d]] */
static void geom_map(const dn_geom *g, int P, const int s[3], int o[3]) { for (int d = 0; d < 3; d++) o[d] = g->flip[d] ? P - 1 - s[g->perm[d]] : s[g->perm[d]]; }

/* recipe views on the loader side: 2 globals (the second overlapping the first by >= 1/4 of the smaller cube per axis) + locals, each with a
   log-uniform zoom and one of the 48 cube symmetries (the intensity augmentation stays with the trainer) */
static void dn_draw_views(const dn_data *D, dn_sample *S, uint64_t *ar) {
    const int Ps = D->P; S->nxf = 2 + D->vnloc;
    for (int v = 0; v < S->nxf; v++) {
        dr_xf *t = &S->xf[v]; const int gl = v < 2; t->n = gl ? D->vP : D->vPL; t->src = 0;
        const float lo = gl ? D->zlo : D->lzlo, hi = gl ? D->zhi : D->lzhi; const float z = expf(logf(lo) + (float)rng_u(ar) * (logf(hi) - logf(lo)));
        t->m = (int)lrintf(t->n * z); if (t->m > Ps) t->m = Ps; if (t->m < 8) t->m = 8;
        for (int d = 0; d < 3; d++) {
            int lo2 = 0, hi2 = Ps - t->m;
            if (v == 1) { const int ov = (t->m < S->xf[0].m ? t->m : S->xf[0].m) / 4; const int a = S->xf[0].off[d] - t->m + ov, b = S->xf[0].off[d] + S->xf[0].m - ov; if (a > lo2) lo2 = a; if (b < hi2) hi2 = b; if (hi2 < lo2) hi2 = lo2; }
            t->off[d] = lo2 + (int)(rng_u(ar) * (hi2 - lo2 + 1)); if (t->off[d] > Ps - t->m) t->off[d] = Ps - t->m;
        }
        int pm[3] = {0, 1, 2}; for (int k = 2; k > 0; k--) { const int j = (int)(rng_next(ar) % (uint64_t)(k + 1)); const int q = pm[k]; pm[k] = pm[j]; pm[j] = q; }
        for (int d = 0; d < 3; d++) { t->perm[d] = pm[d]; t->flip[d] = (int)(rng_next(ar) & 1); }
    }
}
/* a u8 window [Ps^3] resampled into global view v (nearest source voxel; out of range: fill). The map is separable: view axis d reads
   source axis perm[d] alone, so per axis a table of source offsets */
static void dn_view_u8(const dn_sample *S, int v, int Ps, const uint8_t *src, uint8_t *dst, int fill) {
    const dr_xf *t = &S->xf[v]; const int n = t->n; const int64_t st[3] = {(int64_t)Ps * Ps, Ps, 1};
    int64_t *tb = malloc(sizeof(int64_t) * 3 * n);
    for (int d = 0; d < 3; d++) for (int i = 0; i < n; i++) {
        const double o = t->flip[d] ? (double)(n - 1) - i : i; const int q = (int)lrint(t->off[t->perm[d]] + (o + 0.5) * t->m / n - 0.5);
        tb[d * n + i] = q >= 0 && q < Ps ? q * st[t->perm[d]] : -1;
    }
    for (int z = 0; z < n; z++) for (int y = 0; y < n; y++) {
        const int64_t az = tb[z], ay = tb[n + y]; uint8_t *row = dst + ((size_t)z * n + y) * n;
        if (az < 0 || ay < 0) { memset(row, fill, (size_t)n); continue; }
        const uint8_t *sp = src + az + ay; const int64_t *tx = tb + 2 * n;
        for (int x = 0; x < n; x++) row[x] = tx[x] >= 0 ? sp[tx[x]] : (uint8_t)fill;
    }
    free(tb);
}
/* the stored phase of global view v (unknown where the view's nearest CT is 0) and its winding mod 4 (as the train sampler: the phase
   unwrapped over the window before the air test, tiles of about P / 6) */
static void dn_view_phase(const dn_data *D, dn_sample *S, int v) {
    const int n = S->xf[v].n; const size_t n3 = (size_t)n * n * n;
    dn_view_u8(S, v, D->P, S->phw, S->pv[v], PHASE_UNKNOWN);
    uint8_t *ct = malloc(n3); dn_view_u8(S, v, D->P, S->ca, ct, 0);
    size_t known = 0; for (size_t k = 0; k < n3; k++) { if (!ct[k]) S->pv[v][k] = SIDE_UNKNOWN; known += S->pv[v][k] != PHASE_UNKNOWN; }
    if (getenv("UFSM_DENSE_DBG")) { size_t ph = 0, ctn = 0; for (size_t k = 0; k < n3; k++) ctn += ct[k] != 0; for (size_t k = 0; k < (size_t)D->P * D->P * D->P; k++) ph += S->phw[k] != PHASE_UNKNOWN;
        fprintf(stderr, "dense worker: %s view %d (m %d off %d %d %d): window phase known %zu, view CT nonzero %zu, view phase known %zu of %zu\n", D->src[S->si].name, v, S->xf[v].m, S->xf[v].off[0], S->xf[v].off[1], S->xf[v].off[2], ph, ctn, known, n3); }
    if (known < 4096) { free(ct); S->uwv[v] = 0; return; }   /* (no labelled phase in this view: no dense unwrap2 targets) */
    const int wt = (n + 5) / 6;
    if (band_wind4_tiled(S->pv[v], n, n, n, wt, 80, 512, 1, S->w4[v], S->wc[v], nullptr) < 0) { memset(S->w4[v], 255, n3); memset(S->wc[v], 255, n3); }
    free(ct); S->uwv[v] = 1;
}
static int sample_fill(dn_data *D, dn_sample *S, uint64_t *rs, int nthreads, uint8_t *codes_buf) {
    const int P = D->P; const size_t p3 = (size_t)P * P * P;
    for (int tries = 0; tries < 200; tries++) {
        int si = 0;
        if (D->share_u >= 0 && D->wsum_u > 0 && D->wsum_l > 0) {   /* (share 0: labelled sources only) */   /* the unlabelled group with probability share_u, then by weight inside the group */
            const int want_u = rng_u(rs) < D->share_u; double r = rng_u(rs) * (want_u ? D->wsum_u : D->wsum_l); int last = -1;
            for (si = 0; si < D->nsrc; si++) { if (D->src[si].ct_only != want_u) continue; last = si; r -= D->src[si].weight; if (r < 0) break; }
            if (si >= D->nsrc) si = last;
        } else {
            double r = rng_u(rs) * D->wsum;
            for (; si < D->nsrc - 1; si++) { r -= D->src[si].weight; if (r < 0) break; }
        }
        dn_src *s = &D->src[si];
        int64_t o[3];
        if (s->mcell) {   /* a mask cell (window centre), jittered inside the cell; the window clamped into the volume */
            const int64_t c = (int64_t)(rng_u(rs) * (double)s->nmcell) % s->nmcell;
            for (int d = 0; d < 3; d++) {
                const int64_t ctr = (int64_t)s->mcell[3 * c + d] * s->mcs + (int64_t)(rng_u(rs) * s->mcs);
                int64_t v = ctr - P / 2; if (v > s->shape[d] - P) v = s->shape[d] - P; if (v < 0) v = 0; o[d] = v;
            }
        } else {
            double rb = rng_u(rs) * s->bvol; int bi = 0;
            for (; bi < s->nbox - 1; bi++) { const int64_t *b = s->box[bi]; rb -= (double)b[3] * b[4] * b[5]; if (rb < 0) break; }
            const int64_t *b = s->box[bi];
            for (int d = 0; d < 3; d++) o[d] = b[d] + (b[3 + d] > P ? (int64_t)(rng_u(rs) * (double)(b[3 + d] - P + 1)) : 0);
        }
        int bad = 0; for (int h = 0; h < s->nho && !bad; h++) bad = box_hit(s->ho[h], o, P);
        if (bad) continue;
        const int64_t n[3] = {P, P, P};
        if (z3_read(s->ca, o, n, S->ca, nthreads)) return -1;
        if (s->halves) { if (z3_read(s->cb, o, n, S->cb, nthreads)) return -1; }
        else memcpy(S->cb, S->ca, p3);
        /* enough papyrus? */
        size_t pap = 0; for (size_t i = 0; i < p3; i += 7) pap += S->ca[i] > s->pap;
        if (pap * 7 < p3 / 50) continue;   /* < 2% papyrus: mostly air */
        S->si = si; memcpy(S->corner, o, sizeof o); S->halves = s->halves;
        if (s->axz) for (int z = 0; z < P; z++) { const int64_t gz = o[0] + z < 0 ? 0 : o[0] + z >= s->shape[0] ? s->shape[0] - 1 : o[0] + z; S->ayx[2 * z] = s->axz[2 * gz] - (float)o[1]; S->ayx[2 * z + 1] = s->axz[2 * gz + 1] - (float)o[2]; }
        /* geometry: a random axis permutation and flips */
        int perm[3] = {0, 1, 2}; for (int k = 2; k > 0; k--) { const int j = (int)(rng_next(rs) % (uint64_t)(k + 1)); const int t = perm[k]; perm[k] = perm[j]; perm[j] = t; }
        for (int d = 0; d < 3; d++) { S->g.perm[d] = perm[d]; S->g.flip[d] = (int)(rng_next(rs) & 1); }
        if (D->ident) for (int d = 0; d < 3; d++) { S->g.perm[d] = d; S->g.flip[d] = 0; }
        /* GT points: labelled voxels (codes) in the interior, reservoir-sampled */
        S->ngt = 0;
        if (s->codes) {
            const int64_t oc[3] = {o[0] + s->org[0], o[1] + s->org[1], o[2] + s->org[2]};
            if (s->cscale <= 1) { if (z3_read(s->codes, oc, n, codes_buf, nthreads)) return -1; }
            else {   /* codes on a k x coarser grid: read the covering block, nearest-neighbour up */
                const int k = s->cscale; int64_t c0[3], cn[3];
                for (int d = 0; d < 3; d++) { c0[d] = oc[d] / k; cn[d] = (oc[d] + P - 1) / k - c0[d] + 1; }
                uint8_t *cb = malloc((size_t)cn[0] * cn[1] * cn[2]);
                if (z3_read(s->codes, c0, cn, cb, nthreads)) { free(cb); return -1; }
                for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) {
                    const uint8_t *src = cb + (((size_t)((oc[0] + z) / k - c0[0]) * cn[1]) + (oc[1] + y) / k - c0[1]) * cn[2]; uint8_t *dst = codes_buf + ((size_t)z * P + y) * P;
                    for (int x = 0; x < P; x++) dst[x] = src[(oc[2] + x) / k - c0[2]];
                }
                free(cb);
            }
            long seen = 0; const int m = D->margin;
            for (int z = m; z < P - m; z++) for (int y = m; y < P - m; y++) {
                const uint8_t *row = codes_buf + ((size_t)z * P + y) * P;
                for (int x = m; x < P - m; x++) {
                    const int c = row[x]; if (!c || c == 255) continue;
                    long k = seen++;
                    if (k >= D->ngt) { k = (long)(rng_next(rs) % (uint64_t)seen); if (k >= D->ngt) continue; }
                    const int sv[3] = {z, y, x}; int ov[3]; geom_map(&S->g, P, sv, ov);
                    S->gidx[k] = (ov[0] * P + ov[1]) * P + ov[2]; S->gcode[k] = c; S->gxyz[3 * k] = ov[0]; S->gxyz[3 * k + 1] = ov[1]; S->gxyz[3 * k + 2] = ov[2];
                }
            }
            S->ngt = (int)(seen < D->ngt ? seen : D->ngt);
            if (S->ngt < 64) S->ngt = 0;
            if (S->ngt && D->nair) {   /* air voxels (unlabelled, below the papyrus level, inside the scan) as one extra class (code 256) */
                long sa = 0; const float lo = s->center - 0.5f * s->scale;
                for (int z = m; z < P - m; z++) for (int y = m; y < P - m; y++) {
                    const uint8_t *row = S->ca + ((size_t)z * P + y) * P, *crow = codes_buf + ((size_t)z * P + y) * P;
                    for (int x = m; x < P - m; x++) {
                        if (crow[x] || !row[x] || row[x] >= lo) continue;
                        long k = sa++;
                        if (k >= D->nair) { k = (long)(rng_next(rs) % (uint64_t)sa); if (k >= D->nair) continue; }
                        const int sv[3] = {z, y, x}; int ov[3]; geom_map(&S->g, P, sv, ov); const long q = S->ngt + k;
                        S->gidx[q] = (ov[0] * P + ov[1]) * P + ov[2]; S->gcode[q] = 256; S->gxyz[3 * q] = ov[0]; S->gxyz[3 * q + 1] = ov[1]; S->gxyz[3 * q + 2] = ov[2];
                    }
                }
                S->ngt += (int)(sa < D->nair ? sa : D->nair);
            }
        }
        /* SSL points: papyrus voxels (view A above the source's papyrus level) */
        long seen = 0; const int m = D->margin;
        for (int z = m; z < P - m; z++) for (int y = m; y < P - m; y++) {
            const uint8_t *row = S->ca + ((size_t)z * P + y) * P;
            for (int x = m; x < P - m; x++) {
                if (row[x] <= s->pap) continue;
                long k = seen++;
                if (k >= D->nssl) { k = (long)(rng_next(rs) % (uint64_t)seen); if (k >= D->nssl) continue; }
                const int sv[3] = {z, y, x}; int ov[3]; geom_map(&S->g, P, sv, ov);
                S->sidx[k] = (ov[0] * P + ov[1]) * P + ov[2];
            }
        }
        S->nssl = (int)(seen < D->nssl ? seen : D->nssl);
        S->nip = S->nin = 0;
        if (s->ink && D->nink > 0) {   /* ink points: reservoir samples of ink (>= 192) and no-ink (<= 63) labelled voxels in the interior */
            uint8_t *ib = D->ident && D->ikp && S->ikw ? S->ikw : codes_buf;   /* (the recipe's ink-prob views resample this window below) */
            if (z3_read(s->ink, o, n, ib, nthreads)) return -1;
            const int half = D->nink / 2, m = D->margin; long sp = 0, sn = 0; int *pp = S->ixyz, *pn = S->ixyz + 3 * half;
            for (int z = m; z < P - m; z++) for (int y = m; y < P - m; y++) {
                const uint8_t *row = ib + ((size_t)z * P + y) * P;
                for (int x = m; x < P - m; x++) {
                    const int v = row[x]; if (v == 255 || (v > 63 && v < 192)) continue;
                    int *dst; long k;
                    if (v >= 192) { k = sp++; if (k >= half) { k = (long)(rng_next(rs) % (uint64_t)sp); if (k >= half) continue; } dst = pp; }
                    else { k = sn++; if (k >= half) { k = (long)(rng_next(rs) % (uint64_t)sn); if (k >= half) continue; } dst = pn; }
                    dst[3 * k] = z; dst[3 * k + 1] = y; dst[3 * k + 2] = x;
                }
            }
            S->nip = (int)(sp < half ? sp : half); S->nin = (int)(sn < half ? sn : half);
            if (S->nip < 4 || S->nin < 4) S->nip = S->nin = 0;
            else if (S->nip < half) memmove(S->ixyz + 3 * S->nip, pn, (size_t)S->nin * 12);   /* no-ink right after the ink points */
        }
        if (D->ident && D->vP > 0) {   /* the recipe: views drawn here (with their phase / ink resampling: CPU work off the training thread) */
            S->ikv = 0;
            if (D->ikp && s->ink && S->ikw) { if (D->nink <= 0 && z3_read(s->ink, o, n, S->ikw, nthreads)) return -1; S->ikv = 1; }   /* (read above with the ink points) */
            dn_draw_views(D, S, rs);
            for (int v = 0; v < 2; v++) S->uwv[v] = 0;
            if (D->uw && s->phase && S->phw) {
                const int k = s->pscale > 1 ? s->pscale : 1; int64_t c0[3], cn[3];   /* (the codes' grid: + origin, k x coarser) */
                const int64_t oc[3] = {o[0] + s->org[0], o[1] + s->org[1], o[2] + s->org[2]};
                for (int d = 0; d < 3; d++) { c0[d] = oc[d] / k; cn[d] = (oc[d] + P - 1) / k - c0[d] + 1; }
                uint8_t *pb = malloc((size_t)cn[0] * cn[1] * cn[2]);
                if (!pb || z3_read(s->phase, c0, cn, pb, nthreads)) { free(pb); return -1; }
                for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) {   /* nearest-neighbour onto the CT window */
                    const uint8_t *src = pb + (((size_t)((oc[0] + z) / k - c0[0]) * cn[1]) + (oc[1] + y) / k - c0[1]) * cn[2]; uint8_t *dst = S->phw + ((size_t)z * P + y) * P;
                    for (int x = 0; x < P; x++) dst[x] = src[(oc[2] + x) / k - c0[2]];
                }
                free(pb);
                for (int v = 0; v < 2; v++) dn_view_phase(D, S, v);
            }
            if (S->ikv) for (int v = 0; v < 2; v++) {   /* ink-splat prob254 (255 unknown) -> target (0..254) and mask */
                const size_t n3 = (size_t)S->xf[v].n * S->xf[v].n * S->xf[v].n;
                dn_view_u8(S, v, P, S->ikw, S->iv[v], 255);
                for (size_t k = 0; k < n3; k++) { const int u8 = S->iv[v][k] != 255; S->im[v][k] = (uint8_t)u8; if (!u8) S->iv[v][k] = 0; }
            }
        }
        return 0;
    }
    fprintf(stderr, "dino-train: no usable window in 200 tries\n");
    return -1;
}
static void *ring_worker(void *a) {
    dn_ring *R = a; uint64_t rs;
    pthread_mutex_lock(&R->mu); rs = R->seed + 0x9E3779B97F4A7C15ull * (uint64_t)(++R->next_id); pthread_mutex_unlock(&R->mu);
    const size_t p3 = (size_t)R->D->P * R->D->P * R->D->P; uint8_t *cbuf = malloc(p3);
    for (;;) {
        pthread_mutex_lock(&R->mu);
        int k = -1;
        while (!R->stop) { for (int i = 0; i < DN_SLOTS; i++) if (R->slot[i].ready == 0) { k = i; break; } if (k >= 0) break; pthread_cond_wait(&R->cv, &R->mu); }
        if (R->stop) { pthread_mutex_unlock(&R->mu); break; }
        R->slot[k].ready = 1;   /* being filled */
        pthread_mutex_unlock(&R->mu);
        const int rc = sample_fill(R->D, &R->slot[k], &rs, R->nthreads, cbuf);
        pthread_mutex_lock(&R->mu); R->slot[k].ready = rc ? -1 : 2; pthread_cond_broadcast(&R->cv); pthread_mutex_unlock(&R->mu);
        if (rc) break;
    }
    free(cbuf); return nullptr;
}
static dn_sample *ring_get(dn_ring *R) {
    pthread_mutex_lock(&R->mu);
    for (;;) {
        for (int i = 0; i < DN_SLOTS; i++) {
            if (R->slot[i].ready == -1) { pthread_mutex_unlock(&R->mu); return nullptr; }
            if (R->slot[i].ready == 2) { R->slot[i].ready = 3; pthread_mutex_unlock(&R->mu); return &R->slot[i]; }
        }
        pthread_cond_wait(&R->cv, &R->mu);
    }
}
static void ring_put(dn_ring *R, dn_sample *S) { pthread_mutex_lock(&R->mu); S->ready = 0; pthread_cond_broadcast(&R->cv); pthread_mutex_unlock(&R->mu); }

/* an umbilicus / axis file ({"control_points":[{"x","y","z"},...]}, coordinates x scale = CT voxels) -> (y, x) per CT plane z in [0, nz):
   linear between control points sorted by z, constant beyond the ends */
static float *load_axis(const char *path, double scale, int64_t nz) {
    FILE *f = fopen(path, "rb"); if (!f) return nullptr;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET); char *t = malloc(n + 1); if (fread(t, 1, n, f) != (size_t)n) { fclose(f); free(t); return nullptr; } fclose(f); t[n] = 0;
    json *j = json_parse(t, n); free(t); if (!j) return nullptr;
    const json *cp = json_get(j, "control_points"); if (!cp || cp->type != J_ARR || cp->n < 1) { json_free(j); return nullptr; }
    const size_t m = cp->n; double *pz = malloc(m * 8), *py = malloc(m * 8), *px = malloc(m * 8);
    for (size_t i = 0; i < m; i++) { const json *q = json_at(cp, i); pz[i] = json_num(json_get(q, "z"), 0) * scale; py[i] = json_num(json_get(q, "y"), 0) * scale; px[i] = json_num(json_get(q, "x"), 0) * scale; }
    json_free(j);
    for (size_t i = 1; i < m; i++) for (size_t k = i; k > 0 && pz[k - 1] > pz[k]; k--) { double a; a = pz[k]; pz[k] = pz[k - 1]; pz[k - 1] = a; a = py[k]; py[k] = py[k - 1]; py[k - 1] = a; a = px[k]; px[k] = px[k - 1]; px[k - 1] = a; }
    float *axz = malloc((size_t)nz * 2 * sizeof(float)); size_t s = 0;
    for (int64_t z = 0; z < nz; z++) {
        while (s + 1 < m && pz[s + 1] < z) s++;
        double y, x;
        if (z <= pz[0]) { y = py[0]; x = px[0]; } else if (s + 1 >= m) { y = py[m - 1]; x = px[m - 1]; }
        else { const double t2 = (z - pz[s]) / (pz[s + 1] - pz[s] > 1e-9 ? pz[s + 1] - pz[s] : 1); y = py[s] + t2 * (py[s + 1] - py[s]); x = px[s] + t2 * (px[s + 1] - px[s]); }
        axz[2 * z] = (float)y; axz[2 * z + 1] = (float)x;
    }
    free(pz); free(py); free(px);
    return axz;
}
static int parse_box(const json *a, int64_t b[6]) {
    if (!a || a->type != J_ARR || a->n != 6) return -1;
    for (int d = 0; d < 6; d++) b[d] = (int64_t)json_num(json_at(a, d), 0);
    return 0;
}
/* sources JSON: {"sources":[{"name", "root", "ct" (array key at the training grid), "ct_b" (optional: a second, noise-independent view of the same
   grid, e.g. ufsm halves a / b), "cache", "codes": {"root", "key"} (winding codes on the same grid; optional), "origin": [z,y,x] (ct coords + origin =
   codes coords; default 0), "boxes": [[z,y,x,nz,ny,nx]...] (ct coords; windows lie inside), "holdout": [[...]] (windows never touch), "weight",
   "norm": [center, scale] (default: robust statistics of a box sample), "papyrus": raw CT above which a voxel counts as papyrus (default center)}]} */
/* robust input normalisation of a CT: median and IQR / 1.349 of the nonzero voxels of 8 random 128^3 blocks of the boxes */
static void box_norm(z3 *ct, const int64_t (*box)[6], int nbox, uint64_t seed, int nthreads, float *center, float *scale) {
    int64_t hist[256] = {0}; uint8_t *b = malloc(128 * 128 * 128); uint64_t rs = seed;
    for (int r = 0; r < 8; r++) {
        const int64_t *B = box[(int)(rng_next(&rs) % (uint64_t)nbox)]; int64_t o[3];
        for (int d = 0; d < 3; d++) o[d] = B[d] + (int64_t)(rng_u(&rs) * (double)(B[3 + d] - 128 > 0 ? B[3 + d] - 128 : 0));
        const int64_t nn[3] = {128, 128, 128};
        if (!z3_read(ct, o, nn, b, nthreads)) for (int v = 0; v < 128 * 128 * 128; v++) hist[b[v]]++;
    }
    free(b); hist[0] = 0; int64_t tot = 0; for (int v = 1; v < 256; v++) tot += hist[v];
    int64_t acc = 0; int med = 128, q25 = 64, q75 = 192;
    for (int v = 1; v < 256; v++) { const int64_t prev = acc; acc += hist[v];
        if (prev < tot / 4 && acc >= tot / 4) q25 = v;
        if (prev < tot / 2 && acc >= tot / 2) med = v;
        if (prev < tot * 3 / 4 && acc >= tot * 3 / 4) q75 = v;
    }
    *center = (float)med; *scale = (float)(q75 - q25) / 1.349f; if (*scale < 1) *scale = 1;
}
static int load_sources(const char *path, dn_data *D, int nthreads) {
    FILE *f = fopen(path, "rb"); if (!f) { fprintf(stderr, "dino-train: cannot read %s\n", path); return -1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET); char *t = malloc(n + 1); if (fread(t, 1, n, f) != (size_t)n) { fclose(f); return -1; } fclose(f); t[n] = 0;
    json *j = json_parse(t, n); free(t); if (!j) { fprintf(stderr, "dino-train: %s: bad JSON\n", path); return -1; }
    const json *arr = json_get(j, "sources"); if (!arr || arr->type != J_ARR) { fprintf(stderr, "dino-train: no sources\n"); return -1; }
    for (size_t i = 0; i < arr->n && D->nsrc < DN_MAXSRC; i++) {
        const json *s = json_at(arr, i); dn_src *S = &D->src[D->nsrc]; memset(S, 0, sizeof *S);
        snprintf(S->name, sizeof S->name, "%s", json_str(json_get(s, "name"), "?"));
        { int skip = 0; for (int e = 0; e < D->nexcl; e++) skip |= strstr(S->name, D->excl[e]) != nullptr || strstr(json_str(json_get(s, "ct"), ""), D->excl[e]) != nullptr;
          if (skip) { fprintf(stderr, "dino-train: source %s excluded\n", S->name); continue; } }
        const char *root = json_str(json_get(s, "root"), nullptr), *ct = json_str(json_get(s, "ct"), nullptr), *ctb = json_str(json_get(s, "ct_b"), nullptr);
        const char *cache = json_str(json_get(s, "cache"), nullptr);
        if (!root || !ct) { fprintf(stderr, "dino-train: source %zu: root / ct missing\n", i); return -1; }
        store *st = store_open(root);
        char ctk[1024]; snprintf(ctk, sizeof ctk, "%s", ct);
        if (json_get(s, "min_level")) snprintf(ctk, sizeof ctk, "%s/%d", ct, (int)json_num(json_get(s, "min_level"), 0));   /* (train's sources: the group + its level) */
        ct = ctk;
        S->ca = st ? z3_open(st, ct, cache) : nullptr;
        if (!S->ca) { fprintf(stderr, "dino-train: WARNING: source %s skipped: cannot open %s %s: %s\n", S->name, root, ct, z3_error()); if (st) store_close(st); continue; }   /* (one unreadable scan does not stop a long run) */
        if (ctb) { S->cb = z3_open(st, ctb, cache); if (!S->cb) { fprintf(stderr, "dino-train: %s: cannot open %s\n", S->name, ctb); return -1; } S->halves = 1; }
        memcpy(S->shape, z3_meta_of(S->ca)->shape, sizeof S->shape);
        const json *cj = json_get(s, "codes");
        if (cj) {
            store *cs = store_open(json_str(json_get(cj, "root"), "")); S->codes = cs ? z3_open(cs, json_str(json_get(cj, "key"), ""), nullptr) : nullptr;
            if (!S->codes) { fprintf(stderr, "dino-train: %s: cannot open the codes\n", S->name); return -1; }
        }
        S->cscale = (int)json_num(json_get(cj, "scale"), 1);
        const json *pj = json_get(s, "phase");   /* the stored winding phase (unwrap2 heads in the recipe) */
        if (pj) {
            store *ps = store_open(json_str(json_get(pj, "root"), "")); S->phase = ps ? z3_open(ps, json_str(json_get(pj, "key"), ""), nullptr) : nullptr;
            if (!S->phase) { fprintf(stderr, "dino-train: %s: cannot open the phase\n", S->name); return -1; }
            S->pscale = (int)json_num(json_get(pj, "scale"), 1);
            S->um = (float)json_num(json_get(s, "um"), 9.6);   /* (the packed-sheet weight's voxel size) */
        }
        const json *ij = json_get(s, "ink");   /* ink labels on the CT grid (ufsm ink-splat prob254: 254 ink, 0 no ink, 255 unknown) */
        if (ij) {
            store *is = store_open(json_str(json_get(ij, "root"), "")); S->ink = is ? z3_open(is, json_str(json_get(ij, "key"), ""), nullptr) : nullptr;
            if (!S->ink) { fprintf(stderr, "dino-train: %s: cannot open the ink labels\n", S->name); return -1; }
        }
        const json *org = json_get(s, "origin"); for (int d = 0; d < 3; d++) S->org[d] = org ? (int64_t)json_num(json_at(org, d), 0) : 0;
        const json *bx = json_get(s, "boxes");
        if (bx && bx->type == J_ARR) for (size_t k = 0; k < bx->n && S->nbox < DN_MAXBOX; k++) if (!parse_box(json_at(bx, k), S->box[S->nbox])) S->nbox++;
        if (!S->nbox) { for (int d = 0; d < 3; d++) { S->box[0][d] = 0; S->box[0][3 + d] = S->shape[d]; } S->nbox = 1; }
        { int small = 0; for (int k = 0; k < S->nbox; k++) for (int d = 0; d < 3; d++) small |= S->box[k][3 + d] < D->P;
          if (small) { fprintf(stderr, "dino-train: WARNING: source %s skipped: a box (or the volume) is smaller than the %d^3 window\n", S->name, D->P); z3_close(S->ca); continue; } }
        const json *ho = json_get(s, "holdout");
        if (ho && ho->type == J_ARR) for (size_t k = 0; k < ho->n && S->nho < DN_MAXBOX; k++) if (!parse_box(json_at(ho, k), S->ho[S->nho])) S->nho++;
        if (json_get(s, "axis")) { S->axz = load_axis(json_str(json_get(s, "axis"), ""), json_num(json_get(s, "axis_scale"), 1.0), S->shape[0]); if (!S->axz) { fprintf(stderr, "dino-train: %s: cannot read the axis\n", S->name); return -1; } }
        S->weight = json_num(json_get(s, "weight"), 1.0);
        S->ct_only = (int)json_num(json_get(s, "ct_only"), 0) || !S->codes;
        const json *mk = json_get(s, "mask");
        if (mk) {   /* scanprep cell mask: RAW (u8 per cell) + RAW.json {"grid": [z,y,x], "cell": c} */
            const char *mp = json_str(json_get(mk, "map"), ""); const int mmin = (int)json_num(json_get(mk, "min"), 2);
            char jp[1100]; snprintf(jp, sizeof jp, "%s.json", mp); FILE *jf = fopen(jp, "rb");
            if (!jf) { fprintf(stderr, "dino-train: %s: cannot read %s\n", S->name, jp); return -1; }
            char jb[4096]; const size_t jn = fread(jb, 1, sizeof jb - 1, jf); fclose(jf); jb[jn] = 0;
            json *mj = json_parse(jb, jn); if (!mj) { fprintf(stderr, "dino-train: %s: bad %s\n", S->name, jp); return -1; }
            int64_t gd[3]; for (int d = 0; d < 3; d++) gd[d] = (int64_t)json_num(json_at(json_get(mj, "grid"), d), 0);
            S->mcs = (int)json_num(json_get(mj, "cell"), 32); json_free(mj);
            const size_t nc = (size_t)gd[0] * gd[1] * gd[2]; uint8_t *m = malloc(nc); FILE *mf = fopen(mp, "rb");
            if (!mf || fread(m, 1, nc, mf) != nc) { fprintf(stderr, "dino-train: %s: cannot read %s\n", S->name, mp); return -1; }
            fclose(mf);
            S->nmcell = 0; for (size_t c = 0; c < nc; c++) S->nmcell += m[c] >= mmin;
            S->mcell = malloc((size_t)(S->nmcell > 0 ? S->nmcell : 1) * 3 * 4); int64_t k = 0;
            for (int64_t z = 0; z < gd[0]; z++) for (int64_t y = 0; y < gd[1]; y++) for (int64_t x = 0; x < gd[2]; x++)
                if (m[(z * gd[1] + y) * gd[2] + x] >= mmin) { S->mcell[3 * k] = (int32_t)z; S->mcell[3 * k + 1] = (int32_t)y; S->mcell[3 * k + 2] = (int32_t)x; k++; }
            free(m);
            if (!S->nmcell) { fprintf(stderr, "dino-train: %s: no mask cells\n", S->name); return -1; }
        }
        S->bvol = 0; for (int k = 0; k < S->nbox; k++) S->bvol += (double)S->box[k][3] * S->box[k][4] * S->box[k][5];
        const json *nm = json_get(s, "norm");
        if (nm && nm->type == J_ARR) { S->center = (float)json_num(json_at(nm, 0), 100); S->scale = (float)json_num(json_at(nm, 1), 50); }
        else if (nm) { S->center = (float)json_num(json_get(nm, "center"), 100); S->scale = (float)json_num(json_get(nm, "scale"), 50); }   /* (train's form) */
        else box_norm(S->ca, (const int64_t (*)[6])S->box, S->nbox, 12345 + i, nthreads, &S->center, &S->scale);
        S->pap = (float)json_num(json_get(s, "papyrus"), S->center);
        D->wsum += S->weight; if (S->ct_only) D->wsum_u += S->weight; else D->wsum_l += S->weight;
        fprintf(stderr, "dino-train: source %s: %d boxes (%.2f Gvox), %d holdouts, %s, codes %s, weight %g, norm %.1f / %.1f, papyrus > %.0f\n", S->name, S->nbox, S->bvol / 1e9,
                S->nho, S->halves ? "two views (ct / ct_b)" : "one CT (views by noise)", S->codes ? "yes" : "no", S->weight, S->center, S->scale, S->pap);
        D->nsrc++;
    }
    json_free(j);
    return D->nsrc ? 0 : -1;
}

/* precision / storage settings of the production trainer (tests/prof_infer.c with ~/ufsm-data/pi_prod.sh's environment) unless --mx 0 */
static void dn_precision(int fp32) {
    if (fp32) return;   /* --mx 0: the bf16 tensor-core path without MX storage */
    setenv("UFSM_ACT_MX4", "1", 0); setenv("UFSM_XIN_PREC", "8", 0); setenv("UFSM_GRAD_MX8", "1", 0); setenv("UFSM_GRAD_MX4", "1", 0);
    setenv("UFSM_F4_WGRAD", "1", 0); setenv("UFSM_DOWN_NORM", "1", 0);
}
static void dn_precision_after_init(int fp32) {
    if (fp32) { nn_set_tf32(1); return; }
    nn_set_tf32(1); nn_set_f16(1);
    nn_set_prec_policy(getenv("UFSM_PREC_POLICY") ? getenv("UFSM_PREC_POLICY") : "all=fp4:fp4:fp4,enc0.c1=fp16");
    nn_set_gn_stored(1); nn_set_sr(1);
}

/* head slabs (--head-slab Z): the 1^3 head per z-slab of Z planes into one fp16 slab buffer; the points' embeddings gathered from each slab
   after a forward, and in the backward's head step (dn_head_cb) the points' gradients scattered into each slab's fp16 logit gradient
   (scaled by the gradient scale) for the fused slab head backward */
/* the recipe's dense supervised heads (dino-train --recipe --unwrap2 W / --ink-prob W): output channels after the ED embedding channels,
   train.c's unwrap2 layout shifted by ED: ED + 0..1 phase (cos, sin), ED + 2..8 n n^T + density, ED + 9..10 winding mod 4 (cos, sin of a
   quarter-turn gauge), ED + 11 the ink-splat probability; trained per global view from the stored phase / ink labels resampled into the
   view (the loader workers), with train.c's slab losses (pass 0: statistics after the forward; pass 1: gradients in place in the head
   step) */
typedef struct dn_uw {
    int uw, ik;                                      /* this view: unwrap2 targets / ink-prob targets present */
    const uint8_t *pv, *w4, *wc, *ik_t, *ik_m;       /* device views [P^3]: phase, winding mod 4, component, ink target, ink mask */
    uint8_t *sidet, *sidem, *nt, *nm, *sidew, *nw, *onew;   /* slab-sized targets / masks, per-channel weights */
    double *ds, *w4ds; float *w4fin, *sscr, *nscr, *iscr;
    float um, dscale, pum, pw, ttau, tkap; int nstep, dcode, trunc;
    float lp, ln, lw4, lik;                          /* the loss weights of this step (ramped) */
} dn_uw;
typedef struct { unet *u; void *hs; int Z, P, D, CO; const float *e1, *g1, *n1, *e2, *g2, *n2; const int *i1, *i2; int ngt, nssl; float w2; const float *e3, *g3, *n3; const int *i3; int n3p; dn_uw *uw; } dn_hs;
static void dn_gather_slabs(dn_hs *H, const int *i1, int n1, float *e1, float *r1, const int *i2, int n2, float *e2, float *r2) {
    const size_t hw = (size_t)H->P * H->P;
    for (int z0 = 0; z0 < H->P; z0 += H->Z) {
        const int z1 = z0 + H->Z < H->P ? z0 + H->Z : H->P; const size_t ns = (size_t)(z1 - z0) * hw, vb = (size_t)z0 * hw;
        if (unet_head_fwd_z(H->u, z0, z1, H->hs)) { fprintf(stderr, "dino-train: the slab head is not covered\n"); abort(); }
        if (n1) dn_gather_z(H->hs, ns, vb, H->D, i1, n1, e1, r1);
        if (n2) dn_gather_z(H->hs, ns, vb, H->D, i2, n2, e2, r2);
    }
}
/* the dense heads' losses of one slab (logits hs [CO][ns], fp16): pass 0 statistics, pass 1 gradients in place (the rest of the slab zeroed) */
static void dn_uw_slab(dn_hs *H, int pass, int z0, int z1) {
    dn_uw *W = H->uw; const int P = H->P, ED = H->D, nz = z1 - z0; const size_t hw = (size_t)P * P, ns = (size_t)nz * hw, vb = (size_t)z0 * hw;
#define LGC(c) ((char *)H->hs + (size_t)(c) * ns * 2)
    const shape5 one = {1, 1, nz, P, P}, two = {1, 2, nz, P, P}, seven = {1, 7, nz, P, P};
    if (W->uw) {
        nn_phase_targets(W->pv + vb, ns, W->sidet, W->sidem);
        if (W->trunc) nn_phase_trunc(LGC(ED), LGC(ED + 1), W->sidet, W->sidem, ns, W->ttau, W->tkap);
        if (W->pw > 1) nn_packed_mask_z(W->pv, P, P, P, W->nstep, W->um, W->pum, W->pw, W->sidem, z0, z1);
        nn_unwrap_targets2_z(W->pv, P, P, P, W->nstep, W->dscale, W->dcode, W->um, W->pum, W->pw, W->nt, W->nm, z0, z1);
    }
    if (!pass) {
        if (W->uw) {
            nn_wind4_stats(LGC(ED + 9), LGC(ED + 10), W->w4 + vb, W->wc + vb, W->sidem, ns, W->w4ds);
            nn_loss_stats_acc(LGC(ED), W->sidet, W->sidem, W->sidew, two, W->ds);
            nn_loss_stats_acc(LGC(ED + 2), W->nt, W->nm, W->nw, seven, W->ds + 10);
        }
        if (W->ik) nn_loss_stats_acc(LGC(ED + 11), W->ik_t + vb, W->ik_m + vb, W->onew, one, W->ds + 50);
        return;
    }
    const int g16 = nn_get_loss_grad_h16(); nn_set_loss_grad_h16(1);   /* (in place over the fp16 slab: 16-bit gradients, scaled by the gradient scale) */
    if (W->uw) {
        nn_wind4_grad(LGC(ED + 9), LGC(ED + 10), W->w4 + vb, W->wc + vb, W->sidem, ns, W->w4fin, W->lw4, LGC(ED + 9), LGC(ED + 10));
        nn_loss_grad_only(LGC(ED), W->sidet, W->sidem, W->sidew, two, 0.f, LGC(ED), W->sscr); nn_loss_grad_scale(LGC(ED), 2 * ns, W->lp);
        nn_loss_grad_only(LGC(ED + 2), W->nt, W->nm, W->nw, seven, 0.f, LGC(ED + 2), W->nscr); nn_loss_grad_scale(LGC(ED + 2), 7 * ns, W->ln);
    } else nn_zero(LGC(ED), 11 * ns * 2);
    if (W->ik) { nn_loss_grad_only(LGC(ED + 11), W->ik_t + vb, W->ik_m + vb, W->onew, one, 0.f, LGC(ED + 11), W->iscr); nn_loss_grad_scale(LGC(ED + 11), ns, W->lik); }
    else nn_zero(LGC(ED + 11), ns * 2);
    nn_set_loss_grad_h16(g16);
    nn_zero(H->hs, (size_t)ED * ns * 2);
    if (H->CO > ED + 12) nn_zero(LGC(ED + 12), (size_t)(H->CO - ED - 12) * ns * 2);
#undef LGC
}
/* pass 0 over the window (after the student forward of a supervised global view): the statistics, finished into the scratches */
static void dn_uw_stats(dn_hs *H) {
    dn_uw *W = H->uw; const int P = H->P;
    nn_zero(W->ds, 64 * sizeof(double)); nn_zero(W->w4ds, nn_wind4_sums());
    for (int z0 = 0; z0 < P; z0 += H->Z) {
        const int z1 = z0 + H->Z < P ? z0 + H->Z : P;
        if (unet_head_fwd_z(H->u, z0, z1, H->hs)) { fprintf(stderr, "dino-train: the slab head is not covered\n"); abort(); }
        dn_uw_slab(H, 0, z0, z1);
    }
    if (W->uw) { nn_wind4_fin(W->w4ds, W->w4fin); nn_loss_stats_fin(W->ds, W->sidew, (shape5){1, 2, P, P, P}, W->sscr); nn_loss_stats_fin(W->ds + 10, W->nw, (shape5){1, 7, P, P, P}, W->nscr); }
    if (W->ik) nn_loss_stats_fin(W->ds + 50, W->onew, (shape5){1, 1, P, P, P}, W->iscr);
}
static void dn_head_cb(void *ctx) {
    dn_hs *H = ctx; const size_t hw = (size_t)H->P * H->P; const float gs = nn_get_grad_scale();
    const int dense = H->uw && (H->uw->uw || H->uw->ik);
    for (int z0 = 0; z0 < H->P; z0 += H->Z) {
        const int z1 = z0 + H->Z < H->P ? z0 + H->Z : H->P; const size_t ns = (size_t)(z1 - z0) * hw, vb = (size_t)z0 * hw;
        if (dense) {   /* the slab's logits again: the dense losses' gradients in place, the embedding rows zeroed for the point scatters */
            if (unet_head_fwd_z(H->u, z0, z1, H->hs)) { fprintf(stderr, "dino-train: the slab head is not covered\n"); abort(); }
            dn_uw_slab(H, 1, z0, z1);
        } else nn_zero(H->hs, (size_t)H->CO * ns * 2);
        if (H->ngt) dn_scatter_z(H->e1, H->g1, H->n1, H->i1, H->ngt, H->D, ns, vb, gs, H->hs);
        if (H->nssl) dn_scatter_z(H->e2, H->g2, H->n2, H->i2, H->nssl, H->D, ns, vb, H->w2 * gs, H->hs);
        if (H->n3p) dn_scatter_z(H->e3, H->g3, H->n3, H->i3, H->n3p, H->D, ns, vb, gs, H->hs);
        if (unet_head_bwd_z(H->u, z0, z1, H->hs, z0 == 0)) { fprintf(stderr, "dino-train: the fused slab head backward is not covered\n"); abort(); }
    }
}

/* a checkpoint into u; one with fewer outputs (a recipe run before the dense heads) grows: its rows kept, the new rows zero (weights,
   bias, optimiser state), trunk and attention stage as saved. Returns the step or -1 */
static int dn_load_ckpt(unet *u, const char *path) {
    unet_cfg oc; int st = 0;
    if (unet_peek(path, &oc, &st)) return -1;
    if (oc.cout < unet_cfg_of(u)->cout) {
        const int r = unet_load_grow(u, path, oc.cout, 0.f);
        if (r >= 0) fprintf(stderr, "dino-train: %s: head grown %d -> %d outputs (trunk kept, new rows zero)\n", path, oc.cout, unet_cfg_of(u)->cout);
        return r;
    }
    return unet_load(u, path);
}
/* embedding channels of a dino net: the first min(cout, 32) (a recipe net with dense heads: 32 + 12 padded to 48) */
static int dn_embed_dim(const unet_cfg *c) { return c->cout > 32 ? 32 : c->cout; }

/* the EMA network over a box (CT array coords) -> unit int8 embeddings [nz][ny][nx][D] (host), tiles of P with halo h */
static int8_t *dn_embed_box(unet *u, z3 *ct, const int64_t box[6], int P, int h, float center, float scale, int nthreads, int ema, const float *axz, float ars) {
    const int D = dn_embed_dim(unet_cfg_of(u)), CO = unet_cfg_of(u)->cout, in = P - 2 * h; const size_t p3 = (size_t)P * P * P;
    int8_t *out = calloc((size_t)box[3] * box[4] * box[5] * D, 1);
    const int cin = unet_cfg_of(u)->cin; float *ayh = malloc((size_t)P * 8), *ayd = nn_malloc((size_t)P * 8);
    uint8_t *cth = malloc(p3), *ctd = nn_malloc(p3); float *x = nn_malloc(p3 * 4 * cin);
    int8_t *td = nn_malloc((size_t)in * in * in * D), *th = malloc((size_t)in * in * in * D);
    if (!out || !cth || !ctd || !x || !td || !th) { fprintf(stderr, "dino: embed buffers\n"); return nullptr; }
    if (ema) unet_use_ema(u, 1);
    void *hsl = nullptr;   /* (head slabs: one slab's fp16 logits) */
    const dn_geom g0 = {{0, 1, 2}, {0, 0, 0}}; const shape5 xs = {1, cin, P, P, P};
    for (int64_t tz = 0; tz < box[3]; tz += in) for (int64_t ty = 0; ty < box[4]; ty += in) for (int64_t tx = 0; tx < box[5]; tx += in) {
        const int64_t o[3] = {box[0] + tz - h, box[1] + ty - h, box[2] + tx - h}, nn[3] = {P, P, P};
        if (z3_read(ct, o, nn, cth, nthreads)) { fprintf(stderr, "dino: embed read: %s\n", z3_error()); return nullptr; }
        nn_h2d(ctd, cth, p3);
        dn_input(ctd, x, P, g0, center, scale, 1.f, 0.f, 0.f, 0, 0, 0.f, 0, 0.f);
        if (cin == 2 && axz) {
            const int64_t nzv = z3_meta_of(ct)->shape[0];
            for (int z = 0; z < P; z++) { int64_t gz = o[0] + z; gz = gz < 0 ? 0 : gz >= nzv ? nzv - 1 : gz; ayh[2 * z] = axz[2 * gz] - (float)o[1]; ayh[2 * z + 1] = axz[2 * gz + 1] - (float)o[2]; }
            nn_h2d(ayd, ayh, (size_t)P * 8); dn_input_axis(x + p3, P, g0, ayd, ars);
        }
        const void *lg = unet_forward(u, x, xs, 0);
        const int nz = (int)(box[3] - tz < in ? box[3] - tz : in), ny = (int)(box[4] - ty < in ? box[4] - ty : in), nx = (int)(box[5] - tx < in ? box[5] - tx : in);
        if (lg) dn_tile_out(lg, unet_logits_h16(u), P, D, ctd, h, h + nz, h, h + ny, h, h + nx, td, nz, ny, nx, 0, 0, 0);
        else {   /* a head-slab training build (dino-train --head-slab): the head per z-slab */
            const int Z = 16; const size_t hw = (size_t)P * P;
            if (!hsl) hsl = nn_malloc((size_t)CO * Z * hw * 2);
            for (int z0 = 0; z0 < P; z0 += Z) {
                const int z1 = z0 + Z < P ? z0 + Z : P;
                if (z1 <= h || z0 >= h + nz) continue;
                if (unet_head_fwd_z(u, z0, z1, hsl)) { fprintf(stderr, "dino: the slab head is not covered\n"); return nullptr; }
                dn_tile_out_z(hsl, (size_t)(z1 - z0) * hw, (size_t)z0 * hw, P, D, ctd, h, h + nz, h, h + ny, h, h + nx, td, nz, ny, nx, 0, 0, 0);
            }
        }
        nn_d2h(th, td, (size_t)nz * ny * nx * D);
        for (int z = 0; z < nz; z++) for (int y = 0; y < ny; y++)
            memcpy(out + ((((size_t)(tz + z) * box[4]) + ty + y) * box[5] + tx) * D, th + (((size_t)z * ny) + y) * nx * D, (size_t)nx * D);
    }
    if (ema) unet_use_ema(u, 0);
    free(cth); free(th); free(ayh); nn_free(ctd); nn_free(x); nn_free(td); nn_free(ayd); nn_free(hsl);
    return out;
}

/* the in-training eval (--eval SURF-LIST --eval-root R --eval-key K --eval-box B --eval-norm c,s [--eval-codes ROOT KEY]): the EMA net over the
   eval box (tiles of P) -> OUT/eval-emb, dinovol-eval into the train log; 1 when it ran */
static int dn_run_eval(unet *u, int argc, char **argv, int step, int P, int ED, int nthreads, int axis_in, float ars, const char *outd, FILE *lf) {
            /* the EMA net over the eval box -> a dinovol-format group, then dinovol-eval against the surfaces (and codes) */
            long long eb[6]; float ec = 0, es = 0;
            if (sscanf(opt(argc, argv, "--eval-box", ""), "%lld,%lld,%lld,%lld,%lld,%lld", &eb[0], &eb[1], &eb[2], &eb[3], &eb[4], &eb[5]) == 6 &&
                sscanf(opt(argc, argv, "--eval-norm", ""), "%f,%f", &ec, &es) == 2) {
                const double te = dn_now();
                store *est = store_open(opt(argc, argv, "--eval-root", "")); z3 *ez = est ? z3_open(est, opt(argc, argv, "--eval-key", ""), opt(argc, argv, "--eval-cache", nullptr)) : nullptr;
                if (!ez) fprintf(stderr, "dino-train: eval CT: %s\n", z3_error());
                else {
                    const int64_t box[6] = {eb[0], eb[1], eb[2], eb[3], eb[4], eb[5]};
                    float *eax = nullptr;
                    if (axis_in) eax = load_axis(opt(argc, argv, "--eval-axis", ""), atof(opt(argc, argv, "--eval-axis-scale", "1")), z3_meta_of(ez)->shape[0]);
                    int8_t *emb = (axis_in && !eax) ? nullptr : dn_embed_box(u, ez, box, P, 32, ec, es, nthreads, 1, eax, ars);
                    free(eax);
                    char gdir[1200]; snprintf(gdir, sizeof gdir, "%s/eval-emb", outd);
                    if (emb && !dv_write_dense_box(gdir, z3_meta_of(ez)->shape, box, ED, emb)) {
                        const double te2 = dn_now();
                        /* dinovol-eval argv: group, surfaces from the list file, options */
                        char *av[96]; int ac = 0; av[ac++] = "ufsm"; av[ac++] = "dinovol-eval"; av[ac++] = gdir;
                        FILE *sl = fopen(opt(argc, argv, "--eval", ""), "r"); char lines[32][1024]; int nl = 0;
                        while (sl && nl < 32 && fgets(lines[nl], sizeof lines[nl], sl)) { lines[nl][strcspn(lines[nl], "\n")] = 0; if (lines[nl][0]) av[ac++] = lines[nl++]; }
                        if (sl) fclose(sl);
                        av[ac++] = "--n-ref"; av[ac++] = (char *)opt(argc, argv, "--eval-refs", "300");
                        if (opt(argc, argv, "--eval-codes", nullptr)) { int ci = 0; for (int k = 2; k + 2 < argc; k++) if (!strcmp(argv[k], "--eval-codes")) ci = k;
                            av[ac++] = "--codes"; av[ac++] = argv[ci + 1]; av[ac++] = argv[ci + 2]; av[ac++] = "--codes-radius"; av[ac++] = "4"; }
                        av[ac] = nullptr;
                        fflush(stdout); fprintf(stderr, "dino-train: eval at step %d (embed %.1f s)\n", step, te2 - te);
                        if (lf) { fprintf(lf, "eval step %d\n", step); fflush(lf); }
                        int fd = dup(1); if (lf) dup2(fileno(lf), 1);
                        cmd_dinovol_eval(ac, av);
                        fflush(stdout); dup2(fd, 1); close(fd);
                        fprintf(stderr, "dino-train: eval done (%.1f s)\n", dn_now() - te2);
                    }
                    free(emb); z3_close(ez);
                }
                store_close(est);
                return 1;
            }
            return 0;
}

static volatile sig_atomic_t g_dn_stop = 0;
static void dn_sig(int s) { (void)s; g_dn_stop = 1; }

/* a projection head with its teacher (EMA) copy and optimiser state (src/dinor.h dr_head) */
typedef struct { dr_head H; float *p, *pt, *g, *m, *v, *mo, *work; dr_hws w; void *wb; int rows, proto_muon, train_g; } dn_head;
static void dn_head_init(dn_head *h, int Cin, int HH, int HB, int K, int rows, uint64_t seed, int proto_muon) {
    dr_head_layout(&h->H, Cin, HH, HB, K); const dr_head *H = &h->H; h->rows = rows; h->proto_muon = proto_muon;
    h->p = nn_malloc(H->n * 4); h->pt = nn_malloc(H->n * 4); h->g = nn_malloc(H->n * 4); h->m = nn_malloc(H->n * 4); h->v = nn_malloc(H->n * 4); h->mo = nn_malloc(H->n * 4);
    float *b = calloc(H->n, 4); uint64_t r = seed;
#define TN() ((float)(0.02 * sqrt(-2 * log(rng_u(&r) + 1e-12)) * cos(6.283185307 * rng_u(&r))))
    for (size_t i = H->w1; i < H->b1; i++) b[i] = TN();
    for (size_t i = H->w2; i < H->b2; i++) b[i] = TN();
    for (size_t i = H->w3; i < H->b3; i++) b[i] = TN();
#undef TN
    const double bv = 1.0 / sqrt((double)HB);   /* nn.Linear's default init of the weight-normed layer: U(-1/sqrt(in), 1/sqrt(in)) */
    for (size_t i = H->v; i < H->g; i++) b[i] = (float)((2 * rng_u(&r) - 1) * bv);
    for (size_t i = H->g; i < H->n; i++) b[i] = 1.f;
    nn_h2d(h->p, b, H->n * 4); nn_h2d(h->pt, b, H->n * 4); free(b);
    nn_zero(h->m, H->n * 4); nn_zero(h->v, H->n * 4); nn_zero(h->mo, H->n * 4); nn_zero(h->g, H->n * 4);
    size_t wb = dr_muon_work(HH, Cin);
    if (dr_muon_work(HH, HH) > wb) wb = dr_muon_work(HH, HH);
    if (dr_muon_work(HB, HH) > wb) wb = dr_muon_work(HB, HH);
    if (proto_muon && dr_muon_work(K, HB) > wb) wb = dr_muon_work(K, HB);
    h->work = nn_malloc(wb);
    h->wb = nn_malloc(dr_hws_bytes(H, rows)); dr_hws_carve(&h->w, H, rows, h->wb);
}
/* Muon on W1 W2 W3 (and the prototype directions with proto_muon: orthogonalised update, rows renormalised), AdamW on the biases, the
   weight-norm scales g (and the prototypes otherwise); the last layer frozen while step <= freeze; then the teacher EMA */
static void dn_head_step(dn_head *h, float mlr, float hlr, float plr, float wd, int step, int freeze, float mom, int update) {   /* plr: the prototype layer (V, g); update 0: the teacher EMA only (--head-accum) */
    const dr_head *H = &h->H; const int HH = H->H, HB = H->B;
    if (!update) { dr_ema(h->pt, h->p, H->n, mom); return; }
    dr_muon(h->p + H->w1, h->g + H->w1, h->mo + H->w1, HH, H->Cin, mlr, 0.95f, wd, h->work);
    dr_muon(h->p + H->w2, h->g + H->w2, h->mo + H->w2, HH, HH, mlr, 0.95f, wd, h->work);
    dr_muon(h->p + H->w3, h->g + H->w3, h->mo + H->w3, HB, HH, mlr, 0.95f, wd, h->work);
    dr_adamw(h->p + H->b1, h->g + H->b1, h->m + H->b1, h->v + H->b1, HH, hlr, 0.9f, 0.999f, 1e-8f, 0.f, step);
    dr_adamw(h->p + H->b2, h->g + H->b2, h->m + H->b2, h->v + H->b2, HH, hlr, 0.9f, 0.999f, 1e-8f, 0.f, step);
    dr_adamw(h->p + H->b3, h->g + H->b3, h->m + H->b3, h->v + H->b3, HB, hlr, 0.9f, 0.999f, 1e-8f, 0.f, step);
    if (step > freeze) {
        if (h->proto_muon) {
            dr_muon(h->p + H->v, h->g + H->v, h->mo + H->v, H->K, HB, mlr, 0.95f, 0.f, h->work);
            dr_unit_rows(h->p + H->v, H->K, HB);
            if (h->train_g) dr_adamw(h->p + H->g, h->g + H->g, h->m + H->g, h->v + H->g, H->K, plr, 0.9f, 0.999f, 1e-8f, 0.f, step - freeze);
        } else {   /* the prototype directions V (weight decay: scale-free under the weight norm); g only with --proto-g train (no decay) */
            dr_adamw(h->p + H->v, h->g + H->v, h->m + H->v, h->v + H->v, H->g - H->v, plr, 0.9f, 0.999f, 1e-8f, wd, step - freeze);
            if (h->train_g) dr_adamw(h->p + H->g, h->g + H->g, h->m + H->g, h->v + H->g, H->n - H->g, plr, 0.9f, 0.999f, 1e-8f, 0.f, step - freeze);
        }
    }
    dr_ema(h->pt, h->p, H->n, mom);
}
/* 3D block masking (upstream MaskingGenerator3d, mode block): blocks of 4.. remaining cells with log-uniform aspect ratios in [0.3, 1 / 0.3]
   until target cells are masked (or 32 failed attempts in a row) on a G^3 grid; returns the masked count */
static int dn_block_mask(uint8_t *m, int G, int target, uint64_t *r) {
    memset(m, 0, (size_t)G * G * G); int cnt = 0, fails = 0;
    while (cnt < target && fails < 32) {
        const int rem = target - cnt, maxv = rem > 4 ? rem : 4;
        const double vol = 4 + rng_u(r) * (maxv - 4), a1 = exp(log(0.3) + rng_u(r) * (log(1 / 0.3) - log(0.3))), a2 = exp(log(0.3) + rng_u(r) * (log(1 / 0.3) - log(0.3)));
        int d = (int)lrint(cbrt(vol * a1 * a2)), h = (int)lrint(cbrt(vol / a1)), w = (int)lrint(cbrt(vol / a2));
        d = d < 1 ? 1 : d > G ? G : d; h = h < 1 ? 1 : h > G ? G : h; w = w < 1 ? 1 : w > G ? G : w;
        const int z0 = (int)(rng_u(r) * (G - d + 1)), y0 = (int)(rng_u(r) * (G - h + 1)), x0 = (int)(rng_u(r) * (G - w + 1));
        int add = 0;
        for (int z = z0; z < z0 + d; z++) for (int y = y0; y < y0 + h; y++) for (int x = x0; x < x0 + w; x++) {
            uint8_t *c = m + ((size_t)z * G + y) * G + x; if (!*c && cnt + add < target) { *c = 1; add++; } }
        if (add) { cnt += add; fails = 0; } else fails++;
    }
    return cnt;
}

/* ---- the upstream dinovol recipe on our network (dino-train --recipe 1; src/dinor.h) ----
   Per step one source window (the sampler at --source-P, no window geometry) -> 2 global views of P^3 (log-uniform zoom --zoom lo,hi of
   the source cube, offsets such that they overlap by >= 1/4 of the smaller cube on each axis) and --locals local views of --local-P^3
   (zoom --local-zoom), each with one of the 48 cube symmetries and its own intensity augmentation (gamma, contrast, bias, invert, noise).
   Batch 1, every view its own forward (activations of one view at a time): the EMA teacher on both globals, the student on each view
   followed by its backward (the losses of a view depend on it and the teacher only). CLS = the mean of the attention stage's output
   grid tokens (or its first register token, --cls reg). DINO: head MLP -> weight-normed --protos-k prototypes, teacher targets by
   Sinkhorn-Knopp over the current 2 teacher rows plus a queue of earlier teacher logits (--sk-queue; batch 1 has too few rows), student
   global a <- teacher b and b <- a, each local <- both, all terms / (2 + 2 locals); KoLeo on the student global CLS against a queue of
   earlier student CLS; the dense GT sheet SupCon (--gt weight) on the labelled voxels mapped into each global view. Locals run on a second
   network that shares the weights (unet_use_params); its gradient joins before the optimiser step. */
static int dn_recipe(unet *u, const unet_cfg *cfg, dn_data *Dd, int argc, char **argv, int P, int steps0, double minutes, const char *outd, int nthreads, int nwork, uint64_t seed, int HZ) {
    if (!cfg->attn_blocks) { fprintf(stderr, "dino-train --recipe: needs the attention stage (--attn-blocks >= 1) for the CLS tokens\n"); return 2; }
    const int Ps = Dd->P, ED = atoi(opt(argc, argv, "--dim", "32")), CO = cfg->cout, nloc = atoi(opt(argc, argv, "--locals", "4")), PL = atoi(opt(argc, argv, "--local-P", "96"));
    float zlo = 0.7f, zhi = 1.4f, lzlo = 0.8f, lzhi = 1.25f;
    sscanf(opt(argc, argv, "--zoom", "0.7,1.4"), "%f,%f", &zlo, &zhi); sscanf(opt(argc, argv, "--local-zoom", "0.8,1.25"), "%f,%f", &lzlo, &lzhi);
    if ((int)ceilf(P * zhi) > Ps || (int)ceilf(PL * lzhi) > Ps || PL % (1 << (cfg->nlev - 1))) { fprintf(stderr, "dino-train --recipe: views (P %d x %.2f, local %d x %.2f) must fit the source %d; local P a multiple of %d\n", P, zhi, PL, lzhi, Ps, 1 << (cfg->nlev - 1)); return 2; }
    const int K = atoi(opt(argc, argv, "--protos-k", "131072")), HH = atoi(opt(argc, argv, "--head-hidden", "2048")), HB = atoi(opt(argc, argv, "--head-bottleneck", "384"));
    const int SQ = atoi(opt(argc, argv, "--sk-queue", "62")), KQ = atoi(opt(argc, argv, "--koleo-queue", "64")), sk_it = atoi(opt(argc, argv, "--sk-iters", "3"));
    const float ts = (float)atof(opt(argc, argv, "--student-temp", "0.1")), tt0 = (float)atof(opt(argc, argv, "--teacher-temp0", "0.04")), tt1 = (float)atof(opt(argc, argv, "--teacher-temp", "0.07"));
    const int tt_warm = atoi(opt(argc, argv, "--teacher-temp-warmup", "10000")), freeze = atoi(opt(argc, argv, "--freeze-last", "1250")), warm = atoi(opt(argc, argv, "--warmup", "500"));
    const float wk = (float)atof(opt(argc, argv, "--koleo", "0.1")), wdino = (float)atof(opt(argc, argv, "--dino", "1")), wgt = (float)atof(opt(argc, argv, "--gt", "0.1"));
    const float ema0 = (float)atof(opt(argc, argv, "--ema", "0.996")), ema1 = (float)atof(opt(argc, argv, "--ema-end", "0.999"));
    const float lr0 = (float)atof(opt(argc, argv, "--lr", "1e-3")), mlr0 = (float)atof(opt(argc, argv, "--muon-lr", "0.02")), wd = (float)atof(opt(argc, argv, "--wd", "0.04")), clip = (float)atof(opt(argc, argv, "--clip", "3"));
    const float hlr = (float)atof(opt(argc, argv, "--head-lr", "5e-4"));
    /* the prototype layer's AdamW lr: upstream scales lr by batch / 256; at batch 1 (2 teacher rows) the head lr moved every prototype by a
       rank-1 sign step each step and the student logits collapsed (sd 0.05 -> 0.018 within 50 steps of the unfreeze): default head-lr / 128 */
    const float plr = (float)atof(opt(argc, argv, "--proto-lr", "-1")) >= 0 ? (float)atof(opt(argc, argv, "--proto-lr", "-1")) : hlr / 128;
    const float tau = (float)atof(opt(argc, argv, "--tau", "0.1")), hw = (float)atof(opt(argc, argv, "--hard-w", "4")), hr = (float)atof(opt(argc, argv, "--hard-r", "48"));
    const float wvar = (float)atof(opt(argc, argv, "--var", "30")), vgam = 0.8f / sqrtf((float)ED);
    const float noise = (float)atof(opt(argc, argv, "--noise", "0.15"));
    const int cls_reg = !strcmp(opt(argc, argv, "--cls", "pool"), "reg");
    /* --cls-norm 1 (default): the CLS layer-normed (no affine) before the DINO head and KoLeo, as upstream's final LayerNorm. Raw, the pooled
       CLS grew from |4| to |400| and a shared direction dominated it (pairwise cosine 0.5-0.8 across samples): the head saw no differences */
    const int cls_norm = atoi(opt(argc, argv, "--cls-norm", "1"));
    if (cls_reg && cfg->attn_reg < 1) { fprintf(stderr, "dino-train --recipe: --cls reg needs --attn-reg >= 1\n"); return 2; }
    const int log_every = atoi(opt(argc, argv, "--log-every", "20")), save_every = atoi(opt(argc, argv, "--save-every", "2000")), eval_every = atoi(opt(argc, argv, "--eval-every", "2000"));
    const int C = cfg->widths[cfg->nlev - 1];
    int steps = steps0;
    /* the locals' network: the same weights, its own buffers and gradient */
    unet *ul = unet_create(cfg); unet_use_params(ul, unet_live_params(u));
    const size_t np = unet_nparams(u);
    /* iBOT (cross-view masked tokens): token cells of --ibot-patch voxels; a global view is masked with probability --mask-prob, ratio
       U(--mask-ratio); its masked tokens whose centre lies inside the other global view (at most --ibot-max) predict the teacher's iBOT
       targets at the same physical point in that view (dense embeddings sampled at the voxel, unit vectors -> iBOT head) */
    const float wib = (float)atof(opt(argc, argv, "--ibot", "1")), pmask = (float)atof(opt(argc, argv, "--mask-prob", "0.5"));
    float mrlo = 0.1f, mrhi = 0.5f; sscanf(opt(argc, argv, "--mask-ratio", "0.1,0.5"), "%f,%f", &mrlo, &mrhi);
    const int TPS = atoi(opt(argc, argv, "--ibot-patch", "8")), MI = atoi(opt(argc, argv, "--ibot-max", "1024")), KI = atoi(opt(argc, argv, "--ibot-k", opt(argc, argv, "--protos-k", "131072")));
    const int IB = atoi(opt(argc, argv, "--ibot-bottleneck", "256")), GT = P / TPS;
    if (P % TPS) { fprintf(stderr, "dino-train --recipe: --ibot-patch must divide P\n"); return 2; }
    const int pmu = !strcmp(opt(argc, argv, "--proto-opt", "adamw"), "muon");
    /* teacher targets: Sinkhorn-Knopp needs rows >> prototypes; batch 1 has 2 CLS rows (+ the queue) against 8k-131k prototypes, where the
       equipartition flattens every row to ~uniform (H(t) = ln K in every Sinkhorn run). --dino-center M (default 0.9): DINOv1 centring for the
       CLS term (softmax((t - c) / tt), c an EMA of the teacher logits); 0: Sinkhorn over the queue. --ibot-center M: the same for iBOT (0: Sinkhorn) */
    const float dcm = (float)atof(opt(argc, argv, "--dino-center", "0.9")), icm = (float)atof(opt(argc, argv, "--ibot-center", "0"));
    at_set_mx(atoi(opt(argc, argv, "--linear-mx", "0")));   /* the attention stage's and the heads' linears: 0 fp32, 4 MX-fp4, 8 MX-fp8 */
    /* object / region terms (upstream ObjectLoss on our GT: winding codes = sheet objects): up to --objects codes present in both globals,
       --object-points points each; weights ramped over --object-ramp steps */
    const float wobj = (float)atof(opt(argc, argv, "--object", "0.1")), wreg = (float)atof(opt(argc, argv, "--region", "0.1"));
    const int NOB = atoi(opt(argc, argv, "--objects", "6")), NOP = atoi(opt(argc, argv, "--object-points", "48")), oramp = atoi(opt(argc, argv, "--object-ramp", "2000"));
    const int NOT0 = NOB * NOP;
    /* ink term (upstream: SupCon on a 32-d ink readout with two FIFO queues of 512): ink / no-ink labelled points (sources "ink") visible in both
       globals, at most 128 of each per step, through the ink head (Linear ED -> 384 -> GELU -> 32, unit) */
    const float wink = (float)atof(opt(argc, argv, "--ink", "0.1")); const int NIK = wink > 0 ? 256 : 0, IQN = 512;
    const int NOT = NOT0 + NIK;   /* (object points, then ink points: one gathered set) */
    /* dense supervised heads (channels ED..ED + 11; see dn_uw): --unwrap2 W (phase + normals / density x W, winding mod 4 x W x --wind4-lambda),
       --ink-prob W (BCE on the ink-splat probability); both ramp in linearly over --dense-ramp steps from --dense-from (default: the start /
       resume step) */
    const float wuw = (float)atof(opt(argc, argv, "--unwrap2", "0")), wik = (float)atof(opt(argc, argv, "--ink-prob", "0")), w4l = (float)atof(opt(argc, argv, "--wind4-lambda", "0.5"));
    const int dense_ramp = atoi(opt(argc, argv, "--dense-ramp", "1000"));
    if ((wuw > 0 || wik > 0) && CO < ED + 12) { fprintf(stderr, "dino-train --recipe: --unwrap2 / --ink-prob need %d outputs (have %d)\n", ED + 12, CO); return 2; }
    dn_uw UW = {0};
    UW.nstep = atoi(opt(argc, argv, "--normal-step", "1")); UW.dscale = (float)atof(opt(argc, argv, "--density-scale", "4")); UW.dcode = atoi(opt(argc, argv, "--density-code", "1"));
    UW.pw = (float)atof(opt(argc, argv, "--packed-weight", "8")); UW.pum = (float)atof(opt(argc, argv, "--packed-um", "120"));
    UW.ttau = (float)atof(opt(argc, argv, "--phase-trunc", "0")); UW.tkap = (float)atof(opt(argc, argv, "--phase-trunc-conf", "0.5")); const int trunc_from0 = atoi(opt(argc, argv, "--phase-trunc-from", "-1"));   /* (-1: 5000 steps after --dense-from, as train.c) */
    uint8_t *d_pv = nullptr, *d_w4 = nullptr, *d_wc = nullptr, *d_ikt = nullptr, *d_ikm = nullptr;
    if (wuw > 0 || wik > 0) {
        const size_t sp = (size_t)HZ * P * P, gg = (size_t)P * P * P;
        static const uint8_t w1[8] = {1, 1, 1, 1, 1, 1, 1, 1};
        UW.sidet = nn_malloc(2 * sp); UW.sidem = nn_malloc(sp); UW.nt = nn_malloc(7 * sp); UW.nm = nn_malloc(sp);
        UW.sidew = nn_malloc(8); UW.nw = nn_malloc(8); UW.onew = nn_malloc(8); nn_h2d(UW.sidew, w1, 8); nn_h2d(UW.nw, w1, 8); nn_h2d(UW.onew, w1, 8);
        UW.ds = nn_malloc(64 * sizeof(double)); UW.w4ds = nn_malloc(nn_wind4_sums()); UW.w4fin = nn_malloc(nn_wind4_scratch() + 64);
        UW.sscr = nn_malloc(nn_loss_scratch((shape5){1, 2, P, P, P}) + 64); UW.nscr = nn_malloc(nn_loss_scratch((shape5){1, 7, P, P, P}) + 64); UW.iscr = nn_malloc(nn_loss_scratch((shape5){1, 1, P, P, P}) + 64);
        if (wuw > 0) { d_pv = nn_malloc(gg); d_w4 = nn_malloc(gg); d_wc = nn_malloc(gg); }
        if (wik > 0) { d_ikt = nn_malloc(gg); d_ikm = nn_malloc(gg); }
        Dd->uw = wuw > 0; Dd->ikp = wik > 0;
    }
    dr_mlp2 IH; dr_mlp2_layout(&IH, ED, 384, 32);
    float *ip = nn_malloc(IH.n * 4), *ipt = nn_malloc(IH.n * 4), *igr = nn_malloc(IH.n * 4), *im_ = nn_malloc(IH.n * 4), *iv_ = nn_malloc(IH.n * 4), *imo = nn_malloc(IH.n * 4);
    { float *b = calloc(IH.n, 4); uint64_t r3 = seed * 43 + 3; for (size_t i = IH.w1; i < IH.b1; i++) b[i] = (float)(0.02 * sqrt(-2 * log(rng_u(&r3) + 1e-12)) * cos(6.283185307 * rng_u(&r3)));
      for (size_t i = IH.w2; i < IH.b2; i++) b[i] = (float)(0.02 * sqrt(-2 * log(rng_u(&r3) + 1e-12)) * cos(6.283185307 * rng_u(&r3)));
      nn_h2d(ip, b, IH.n * 4); nn_h2d(ipt, b, IH.n * 4); free(b); nn_zero(im_, IH.n * 4); nn_zero(iv_, IH.n * 4); nn_zero(imo, IH.n * 4); }
    void *iws = nn_malloc(dr_mlp2_ws(&IH, NIK > 0 ? NIK : 1)), *iwt = nn_malloc(dr_mlp2_ws(&IH, NIK > 0 ? NIK : 1));
    float *iT = nn_malloc((size_t)2 * (NIK > 0 ? NIK : 1) * 32 * 4), *iS = nn_malloc((size_t)(NIK > 0 ? NIK : 1) * 32 * 4), *idS = nn_malloc((size_t)(NIK > 0 ? NIK : 1) * 32 * 4);
    float *iqp = nn_malloc((size_t)IQN * 32 * 4), *iqn = nn_malloc((size_t)IQN * 32 * 4), *iKP = nn_malloc((size_t)(IQN + 256) * 32 * 4), *iKN = nn_malloc((size_t)(IQN + 256) * 32 * 4);
    int *ikid = nn_malloc((size_t)(IQN + 256) * 4), *iaid = nn_malloc(256 * 4), *hk = malloc((size_t)(IQN + 256) * 4); int iqpf = 0, iqpp = 0, iqnf = 0, iqnp = 0;
    { int a[256]; for (int i = 0; i < 256; i++) a[i] = i; nn_h2d(iaid, a, sizeof a); }
    dr_mlp2 RH; dr_mlp2_layout(&RH, ED, 384, 128);
    float *rp = nn_malloc(RH.n * 4), *rpt = nn_malloc(RH.n * 4), *rg = nn_malloc(RH.n * 4), *rm = nn_malloc(RH.n * 4), *rv = nn_malloc(RH.n * 4), *rmo = nn_malloc(RH.n * 4), *rwk = nn_malloc(dr_muon_work(384, ED) > dr_muon_work(128, 384) ? dr_muon_work(384, ED) : dr_muon_work(128, 384));
    { float *b = calloc(RH.n, 4); uint64_t r2 = seed * 41 + 5; for (size_t i = RH.w1; i < RH.b1; i++) b[i] = (float)(0.02 * sqrt(-2 * log(rng_u(&r2) + 1e-12)) * cos(6.283185307 * rng_u(&r2)));
      for (size_t i = RH.w2; i < RH.b2; i++) b[i] = (float)(0.02 * sqrt(-2 * log(rng_u(&r2) + 1e-12)) * cos(6.283185307 * rng_u(&r2)));
      nn_h2d(rp, b, RH.n * 4); nn_h2d(rpt, b, RH.n * 4); free(b); nn_zero(rm, RH.n * 4); nn_zero(rv, RH.n * 4); nn_zero(rmo, RH.n * 4); }
    void *rws = nn_malloc(dr_mlp2_ws(&RH, NOB > 0 ? NOB : 1)), *rwt = nn_malloc(dr_mlp2_ws(&RH, NOB > 0 ? NOB : 1));
    int *o_idx = nn_malloc((size_t)2 * (NOT > 0 ? NOT : 1) * 4), *o_obj = nn_malloc((size_t)(NOT > 0 ? NOT : 1) * 4), *o_grp = nn_malloc((size_t)(NOT > 0 ? NOT : 1) * 4), *o_rid = nn_malloc(64 * 4);
    uint8_t *o_ones = nn_malloc((size_t)(NOT > 0 ? NOT : 1) + 64);
    float *oT = nn_malloc((size_t)2 * (NOT > 0 ? NOT : 1) * ED * 4), *onT = nn_malloc((size_t)2 * (NOT > 0 ? NOT : 1) * 4), *oS = nn_malloc((size_t)(NOT > 0 ? NOT : 1) * ED * 4), *onS = nn_malloc((size_t)(NOT > 0 ? NOT : 1) * 4), *ogS = nn_malloc((size_t)(NOT > 0 ? NOT : 1) * ED * 4);
    float *omS = nn_malloc(64 * (size_t)ED * 4), *omT = nn_malloc(2 * 64 * (size_t)ED * 4), *ocnt = nn_malloc(2 * 64 * 4), *oyS = nn_malloc(64 * 128 * 4), *oyT = nn_malloc(2 * 64 * 128 * 4), *odyS = nn_malloc(64 * 128 * 4), *odm = nn_malloc(64 * (size_t)ED * 4);
    { uint8_t *o1 = malloc((size_t)NOT + 64); memset(o1, 1, (size_t)NOT + 64); nn_h2d(o_ones, o1, (size_t)NOT + 64); free(o1); int rid[64]; for (int i = 0; i < 64; i++) rid[i] = i + 1; nn_h2d(o_rid, rid, 64 * 4); }
    int *ho_idx = malloc((size_t)2 * (NOT > 0 ? NOT : 1) * 4), *ho_obj = malloc((size_t)(NOT > 0 ? NOT : 1) * 4), *ho_grp = malloc((size_t)(NOT > 0 ? NOT : 1) * 4);   /* the prototype directions on Muon (rows renormalised) */
    /* heads: DINO on the CLS (rows: 2 teacher globals), iBOT on the dense unit embeddings (rows: the masked tokens) */
    dn_head D, I; dn_head_init(&D, C, HH, HB, K, 2, seed * 31 + 7, pmu);
    if (wib > 0) dn_head_init(&I, ED, HH, IB, KI, MI, seed * 37 + 11, pmu);
    /* the weight-norm gains g of the prototype layer: fixed at 1 (upstream norm_last_layer=True; default) or trained (--proto-g train). Trained
       (with the old weight decay), g shrank and flattened every logit row: the student matched near-uniform targets by scale, the EMA teacher
       followed, and H(t) stayed at ln K */
    D.train_g = I.train_g = !strcmp(opt(argc, argv, "--proto-g", "fixed"), "train");
    /* batch-1 remedies for the DINO head: --head-accum N sums its gradient over N steps (one update per N: the large diverse batch upstream
       averages over; the teacher EMA every step); --zcenter M subtracts an EMA (momentum M, from the teacher rows) of the head output z
       before its normalisation and the prototypes (the shared direction that made the prototype updates coherent) */
    const int hacc = atoi(opt(argc, argv, "--head-accum", "1")) > 0 ? atoi(opt(argc, argv, "--head-accum", "1")) : 1;
    const float zcm = (float)atof(opt(argc, argv, "--zcenter", "0"));
    float *zcen = nullptr; if (zcm > 0) { zcen = nn_malloc((size_t)HB * 4); nn_zero(zcen, (size_t)HB * 4); D.w.zc = zcen; }
    /* device buffers */
    const size_t ps3 = (size_t)Ps * Ps * Ps, g3 = (size_t)P * P * P, l3 = (size_t)PL * PL * PL;
    uint8_t *src = nn_malloc(ps3); float *xg = nn_malloc(2 * g3 * 4), *xl = nn_malloc((size_t)(nloc > 0 ? nloc : 1) * l3 * 4);
    dr_xf *xfd = nn_malloc(sizeof(dr_xf) * (2 + nloc)); dr_int *itd = nn_malloc(sizeof(dr_int) * (2 + nloc));
    const int NQ = 2 + SQ; float *tl = nn_malloc((size_t)NQ * K * 4), *tq = nn_malloc((size_t)NQ * K * 4); void *sks = nn_malloc(dr_sk_scratch(NQ, K));
    /* iBOT buffers: per masked global v the teacher targets [MI][KI]; student logits / gradient [MI][KI]; points; the cell mask */
    float *ibq[2] = {nullptr, nullptr}, *ibl = nullptr, *ibd = nullptr, *ieT = nullptr, *inT = nullptr, *ieS = nullptr, *inS = nullptr, *idE = nullptr, *ibw = nullptr; void *ibsk = nullptr, *ibce = nullptr;
    int *isidx = nullptr, *itidx = nullptr, *iid = nullptr, *ihS = malloc((size_t)MI * 4 * 2), *ihT = malloc((size_t)MI * 4 * 2); uint8_t *imask = nullptr;
    if (wib > 0) {
        for (int v = 0; v < 2; v++) ibq[v] = nn_malloc((size_t)MI * KI * 4);
        ibl = nn_malloc((size_t)MI * KI * 4); ibd = nn_malloc((size_t)MI * KI * 4); ibsk = nn_malloc(dr_sk_scratch(MI, KI)); ibce = nn_malloc(dr_ce_scratch(MI));
        ieT = nn_malloc((size_t)2 * MI * ED * 4); inT = nn_malloc((size_t)2 * MI * 4); ieS = nn_malloc((size_t)MI * ED * 4); inS = nn_malloc((size_t)MI * 4); idE = nn_malloc((size_t)MI * ED * 4);
        isidx = nn_malloc((size_t)2 * MI * 4); itidx = nn_malloc((size_t)2 * MI * 4); iid = nn_malloc((size_t)MI * 4); ibw = nn_malloc((size_t)MI * 4); imask = nn_malloc((size_t)GT * GT * GT);
        int *h = malloc((size_t)MI * 4); for (int i = 0; i < MI; i++) h[i] = i; nn_h2d(iid, h, (size_t)MI * 4); free(h);
    }
    uint8_t *useb = calloc((size_t)(K > KI ? K : KI), 1), *useib = calloc((size_t)KI, 1); int *d_am = nn_malloc((size_t)(MI > NQ ? MI : NQ) * 4), *h_am = malloc((size_t)(MI > NQ ? MI : NQ) * 4);
    float *sl = nn_malloc((size_t)K * 4), *dsl = nn_malloc((size_t)K * 4); void *ces = nn_malloc(dr_ce_scratch(1));
    float *clsRS = nn_malloc(8 * 4);   /* (row-norm rstd: teacher rows 0-1, student 4) */
    float *clsT = nn_malloc(2 * (size_t)C * 4), *clsS = nn_malloc((size_t)C * 4), *dcls = nn_malloc((size_t)C * 4), *kq = nn_malloc((size_t)KQ * C * 4);
    float *outs = nn_malloc(64 * 4), *hout = malloc(64 * 4);
    int *pS = nn_malloc(8 * 4), *pT = nn_malloc(8 * 4); float *pW = nn_malloc(8 * 4);
    const int NG = Dd->ngt; int *d_idx = nn_malloc((size_t)NG * 4), *d_code = nn_malloc((size_t)NG * 4), *d_xyz = nn_malloc((size_t)NG * 12);
    float *es = nn_malloc((size_t)NG * ED * 4), *ns = nn_malloc((size_t)NG * 4), *ga = nn_malloc((size_t)NG * ED * 4), *scr = nn_malloc(((size_t)2 * NG + 16) * 4);
    float *tgrad = nullptr; size_t tgn = 0;   /* token gradient [S][C] of the current view */
    float *glz = nn_malloc(l3 * CO * 4); nn_zero(glz, l3 * CO * 4);   /* the locals' (zero) dense logit gradient */
    void *hsl = nn_malloc((size_t)CO * HZ * P * P * 2);
    dn_hs HS = {.u = u, .hs = hsl, .Z = HZ, .P = P, .D = ED, .CO = CO, .uw = (wuw > 0 || wik > 0) ? &UW : nullptr};
    nn_set_logits_h16(1); unet_set_head_ext(u, 1); unet_set_head_cb(u, dn_head_cb, &HS);
    /* resume (--resume CKPT, the watchdog relaunch): the network (weights, EMA teacher, optimiser state) from CKPT, the heads from CKPT.heads
       when their sizes match (a head whose shape changed, or a new one, starts fresh: trunk carried over); queues refill */
    int step0 = 0;
    { const char *rp_ = opt(argc, argv, "--resume", nullptr);
      if (rp_) {
          step0 = dn_load_ckpt(u, rp_);
          if (step0 < 0) { fprintf(stderr, "dino-train --recipe: cannot resume %s (another network configuration?)\n", rp_); return 1; }
          char hp_[4200], rr_[4096]; const char *rq_ = realpath(rp_, rr_) ? rr_ : rp_;   /* (last.ckpt is a symlink: the heads sit beside its target) */
          snprintf(hp_, sizeof hp_, "%s.heads", rq_); FILE *hf = fopen(hp_, "rb");
          struct { float *p; size_t n; const char *nm; } hs[6] = {{D.p, D.H.n, "dino"}, {D.pt, D.H.n, "dino teacher"}, {wib > 0 ? I.p : nullptr, wib > 0 ? I.H.n : 0, "ibot"}, {wib > 0 ? I.pt : nullptr, wib > 0 ? I.H.n : 0, "ibot teacher"}, {rp, RH.n, "region"}, {ip, IH.n, "ink"}};
          long fsz = 0; if (hf) { fseek(hf, 0, SEEK_END); fsz = ftell(hf); fseek(hf, 0, SEEK_SET); }
          size_t want = 0; for (int k = 0; k < 4; k++) want += hs[k].n;
          if (hf && (size_t)fsz >= want * 4) {
              dr_head ibl_; dr_head_layout(&ibl_, ED, HH, IB, KI);   /* (a file written with iBOT on: its two blocks skipped when it is off now) */
              for (int k = 0; k < 4; k++) {
                  if (!hs[k].p) { if (k >= 2 && (size_t)fsz >= (want + 2 * ibl_.n) * 4) fseek(hf, (long)(ibl_.n * 4), SEEK_CUR); continue; }
                  float *h = malloc(hs[k].n * 4); if (fread(h, 4, hs[k].n, hf) == hs[k].n) nn_h2d(hs[k].p, h, hs[k].n * 4); free(h);
              }
              if ((size_t)fsz >= (want + RH.n * 2 + IH.n * 2) * 4) {   /* region and ink heads (student, teacher) */
                  float *h = malloc((RH.n > IH.n ? RH.n : IH.n) * 4);
                  if (fread(h, 4, RH.n, hf) == RH.n) { nn_h2d(rp, h, RH.n * 4); }
                  if (fread(h, 4, RH.n, hf) == RH.n) { nn_h2d(rpt, h, RH.n * 4); }
                  if (fread(h, 4, IH.n, hf) == IH.n) { nn_h2d(ip, h, IH.n * 4); }
                  if (fread(h, 4, IH.n, hf) == IH.n) { nn_h2d(ipt, h, IH.n * 4); }
                  free(h);
              }
              fprintf(stderr, "dino-train --recipe: resumed %s at step %d (heads restored)\n", rp_, step0);
          } else fprintf(stderr, "dino-train --recipe: resumed %s at step %d; heads file missing or of another shape: heads start fresh\n", rp_, step0);
          if (hf) fclose(hf);
      } }
    for (int k = 0; k < (wib > 0 ? 2 : 1); k++) {   /* fixed gains: 1 (a resumed head trained with --proto-g train, or the old decayed g) */
        dn_head *hh = k ? &I : &D; if (hh->train_g) continue;
        const size_t ng = hh->H.n - hh->H.g; float *one = malloc(ng * 4); for (size_t i = 0; i < ng; i++) one[i] = 1.f;
        nn_h2d(hh->p + hh->H.g, one, ng * 4); nn_h2d(hh->pt + hh->H.g, one, ng * 4); free(one);
    }
    const shape5 xsg = {1, cfg->cin, P, P, P}, xsl = {1, cfg->cin, PL, PL, PL};
    unet_build(u, xsg, 1); unet_build(ul, xsl, 1);
    /* data ring (sources at the source window, identity geometry) */
    dn_ring *Rg = calloc(1, sizeof *Rg); Rg->D = Dd; Rg->seed = seed; Rg->nthreads = nthreads; pthread_mutex_init(&Rg->mu, nullptr); pthread_cond_init(&Rg->cv, nullptr);
    for (int i = 0; i < DN_SLOTS; i++) { dn_sample *S = &Rg->slot[i]; S->ca = malloc(ps3); S->cb = malloc(ps3); S->gidx = malloc((size_t)NG * 4); S->gcode = malloc((size_t)NG * 4); S->gxyz = malloc((size_t)NG * 12); S->sidx = malloc(4); S->ayx = malloc((size_t)Ps * 8); S->ixyz = malloc((size_t)(Dd->nink > 0 ? Dd->nink : 1) * 12);
        if (Dd->uw) { S->phw = malloc(ps3); for (int v = 0; v < 2; v++) { S->pv[v] = malloc(g3); S->w4[v] = malloc(g3); S->wc[v] = malloc(g3); } }
        if (Dd->ikp) { S->ikw = malloc(ps3); for (int v = 0; v < 2; v++) { S->iv[v] = malloc(g3); S->im[v] = malloc(g3); } }
        if ((Dd->uw && (!S->phw || !S->pv[1] || !S->w4[1] || !S->wc[1])) || (Dd->ikp && (!S->ikw || !S->iv[1] || !S->im[1]))) { fprintf(stderr, "dino-train --recipe: host memory for the dense targets\n"); return 1; } }
    Dd->vP = P; Dd->vPL = PL; Dd->vnloc = nloc; Dd->zlo = zlo; Dd->zhi = zhi; Dd->lzlo = lzlo; Dd->lzhi = lzhi;   /* the views: drawn by the workers */
    Rg->nworkers = nwork > 8 ? 8 : nwork;
    for (int i = 0; i < Rg->nworkers; i++) pthread_create(&Rg->th[i], nullptr, ring_worker, Rg);
    signal(SIGTERM, dn_sig); signal(SIGINT, dn_sig);
    char logp[1200]; snprintf(logp, sizeof logp, "%s/train.log", outd); FILE *lf = fopen(logp, "a");
    fprintf(stderr, "dino-train --recipe: source %d, globals 2 x %d^3 (zoom %.2f..%.2f), locals %d x %d^3 (zoom %.2f..%.2f), CLS %s, head %d-%d-%d -> %d prototypes, "
                    "Sinkhorn %d iters over 2 + %d queued rows, KoLeo %g (queue %d), GT %g, teacher temp %g -> %g over %d steps, EMA %g -> %g\n",
            Ps, P, zlo, zhi, nloc, PL, lzlo, lzhi, cls_reg ? "register 0" : "grid mean", HH, HH, HB, K, sk_it, SQ, wk, KQ, wgt, tt0, tt1, tt_warm, ema0, ema1);
    uint64_t ar = seed * 1000003 + 99; int qfill = 0, qpos = 0, kfill = 0, kpos = 0, ibv = -1;
    float *dcen = nn_malloc((size_t)K * 4), *icen = nn_malloc((size_t)KI * 4); nn_zero(dcen, (size_t)K * 4); nn_zero(icen, (size_t)KI * 4);   /* (centres: zero at start / resume) */
    double sums[12] = {0}; int nsum = 0; double t_last = dn_now(), t_loop = dn_now(), t_meas0 = 0;
    double dsum[5] = {0}; int dn_nuw = 0, dn_nik = 0; const int dense_from = atoi(opt(argc, argv, "--dense-from", "-1")) >= 0 ? atoi(opt(argc, argv, "--dense-from", "-1")) : step0;
    const int trunc_from = trunc_from0 >= 0 ? trunc_from0 : dense_from + 5000;
    char stopf[1200]; snprintf(stopf, sizeof stopf, "%s/STOP", outd);
    const float gs = nn_get_grad_scale();
    const int terms = 2 + 2 * nloc;
    const int prof = getenv("UFSM_REC_PROF") && ufsm_env_on("UFSM_REC_PROF"); double pt[10] = {0}, pt0 = 0;   /* per-phase seconds (synchronising) */
#define RP(k) do { if (prof) { nn_sync(); const double t_ = dn_now(); if (pt0 > 0) pt[k] += t_ - pt0; pt0 = t_; } } while (0)
    for (int step = step0 + 1; step <= steps && !g_dn_stop; step++) {
        if (minutes > 0 && !step0) {
            if (step == 21) { nn_sync(); t_meas0 = dn_now(); }
            if (step == 201) { nn_sync(); const double ms = (dn_now() - t_meas0) / 180, left = minutes * 60 - (dn_now() - t_loop); steps = 200 + (int)(left / ms); if (steps < 201) steps = 201;
                fprintf(stderr, "dino-train: %.0f ms/step over steps 21..200 -> %d steps for %.1f minutes\n", ms * 1e3, steps, minutes); if (lf) { fprintf(lf, "budget: %.1f ms/step -> %d steps (%.1f min)\n", ms * 1e3, steps, minutes); fflush(lf); } }
        }
        RP(9);
        dn_sample *Sm = ring_get(Rg); if (!Sm) { fprintf(stderr, "dino-train: data loader failed\n"); return 1; }
        const dn_src *sr = &Dd->src[Sm->si];
        RP(0);
        nn_h2d(src, Sm->ca, ps3);
        /* views */
        dr_xf xf[2 + 16]; dr_int it[2 + 16];
        for (int v = 0; v < 2 + nloc; v++) {
            xf[v] = Sm->xf[v];   /* (crop, zoom, symmetry: drawn by the loader worker, which resampled the dense targets with it) */
            dr_int *a = &it[v]; memset(a, 0, sizeof *a);
            a->center = sr->center; a->scale = sr->scale; a->lo = 0.f; a->hi = 255.f;
            a->gamma = rng_u(&ar) < 0.25 ? expf((float)(2 * rng_u(&ar) - 1) * logf(1.25f)) : 1.f;
            a->contrast = rng_u(&ar) < 0.25 ? 0.8f + 0.4f * (float)rng_u(&ar) : 1.f; a->mean = 0.f;
            a->bias = rng_u(&ar) < 0.15 ? 0.15f * (float)(sqrt(-2 * log(rng_u(&ar) + 1e-12)) * cos(6.283185307 * rng_u(&ar))) : 0.f;
            a->invert = rng_u(&ar) < 0.15; a->sigma = noise * (float)rng_u(&ar); a->seed = (uint32_t)(step * 16 + v);
        }
        nn_h2d(xfd, xf, sizeof(dr_xf) * (2 + nloc)); nn_h2d(itd, it, sizeof(dr_int) * (2 + nloc));
        dr_crops(src, Ps, xfd, itd, 2, P, xg);
        if (nloc) dr_crops(src, Ps, xfd + 2, itd + 2, nloc, PL, xl);
        /* GT points per global view: the sample's labelled voxels (source coordinates) mapped into the view */
        int nview[2] = {0, 0}; static int *vidx[2], *vcode[2], *vxyz[2];
        if (!vidx[0]) for (int v = 0; v < 2; v++) { vidx[v] = malloc((size_t)NG * 4); vcode[v] = malloc((size_t)NG * 4); vxyz[v] = malloc((size_t)NG * 12); }
        for (int v = 0; v < 2 && wgt > 0; v++) for (int i = 0; i < Sm->ngt; i++) {
            const double s3[3] = {Sm->gxyz[3 * i], Sm->gxyz[3 * i + 1], Sm->gxyz[3 * i + 2]}; double o[3]; dr_unmap(&xf[v], s3, o);
            int q[3], in = 1; for (int d = 0; d < 3; d++) { q[d] = (int)lrint(o[d]); in &= q[d] >= 4 && q[d] < P - 4; }
            if (!in) continue;
            const int k = nview[v]++; vidx[v][k] = (q[0] * P + q[1]) * P + q[2]; vcode[v][k] = Sm->gcode[i]; vxyz[v][k * 3] = q[0]; vxyz[v][k * 3 + 1] = q[1]; vxyz[v][k * 3 + 2] = q[2];
        }
        /* object points: labelled source points visible in both globals; up to NOB codes with >= 8 such points, NOP points each */
        int nob = 0, nop = 0;
        if (NOT > 0 && (wobj > 0 || wreg > 0)) {
            static int *cand, *ccode; static int ccap;
            if (ccap < Sm->ngt) { free(cand); free(ccode); ccap = Sm->ngt > 1 ? Sm->ngt : 1; cand = malloc((size_t)ccap * 4 * 2); ccode = malloc((size_t)ccap * 4); }
            int nc = 0;
            for (int i = 0; i < Sm->ngt; i++) {
                const double s3[3] = {Sm->gxyz[3 * i], Sm->gxyz[3 * i + 1], Sm->gxyz[3 * i + 2]}; int ok = 1, qi[2];
                for (int v = 0; v < 2 && ok; v++) { double o[3]; dr_unmap(&xf[v], s3, o); int q[3]; for (int d = 0; d < 3; d++) { q[d] = (int)lrint(o[d]); ok &= q[d] >= 2 && q[d] < P - 2; } qi[v] = (q[0] * P + q[1]) * P + q[2]; }
                if (!ok) continue;
                cand[2 * nc] = qi[0]; cand[2 * nc + 1] = qi[1]; ccode[nc] = Sm->gcode[i]; nc++;
            }
            int codes[256], ccount[256], nco = 0;
            for (int i = 0; i < nc; i++) { int k = 0; while (k < nco && codes[k] != ccode[i]) k++; if (k == nco && nco < 256) { codes[nco] = ccode[i]; ccount[nco++] = 0; } if (k < 256) ccount[k]++; }
            for (int k = nco - 1; k > 0; k--) { const int j = (int)(rng_next(&ar) % (uint64_t)(k + 1)); int t2 = codes[k]; codes[k] = codes[j]; codes[j] = t2; t2 = ccount[k]; ccount[k] = ccount[j]; ccount[j] = t2; }
            for (int k = 0; k < nco && nob < NOB; k++) {
                if (ccount[k] < 8) continue;
                int taken = 0, seen = 0, start = nop;
                for (int i = 0; i < nc; i++) {
                    if (ccode[i] != codes[k]) continue;
                    int slot = seen++; if (slot >= NOP) { slot = (int)(rng_next(&ar) % (uint64_t)seen); if (slot >= NOP) continue; } else taken++;
                    ho_idx[start + slot] = cand[2 * i]; ho_idx[NOT + start + slot] = cand[2 * i + 1]; ho_obj[start + slot] = nob + 1; ho_grp[start + slot] = nob;
                }
                nop += taken; nob++;
            }
            if (nob < 2) nob = nop = 0;   /* (no negatives) */
        }
        int npi = 0, nni = 0;   /* ink points visible in both globals (ink first, then no-ink), after the object points */
        if (NIK > 0 && Sm->nip) {
            for (int pass = 0; pass < 2; pass++) {
                const int b0 = pass ? Sm->nip : 0, b1 = pass ? Sm->nip + Sm->nin : Sm->nip; int cnt = 0;
                for (int i = b0; i < b1 && cnt < 128; i++) {
                    const double s3[3] = {Sm->ixyz[3 * i], Sm->ixyz[3 * i + 1], Sm->ixyz[3 * i + 2]}; int ok = 1, qi[2];
                    for (int v = 0; v < 2 && ok; v++) { double o[3]; dr_unmap(&xf[v], s3, o); int q[3]; for (int d = 0; d < 3; d++) { q[d] = (int)lrint(o[d]); ok &= q[d] >= 2 && q[d] < P - 2; } qi[v] = (q[0] * P + q[1]) * P + q[2]; }
                    if (!ok) continue;
                    const int k = nop + npi + nni; ho_idx[k] = qi[0]; ho_idx[NOT + k] = qi[1]; ho_grp[k] = -1; ho_obj[k] = 0; cnt++;
                    if (pass) nni++; else npi++;
                }
            }
            if (npi < 2 || nni < 2) npi = nni = 0;
        }
        const int ntot = nop + npi + nni;
        if (ntot) { nn_h2d(o_idx, ho_idx, (size_t)ntot * 4); nn_h2d(o_idx + NOT, ho_idx + NOT, (size_t)ntot * 4); nn_h2d(o_obj, ho_obj, (size_t)ntot * 4); nn_h2d(o_grp, ho_grp, (size_t)ntot * 4); }
        if (!HS.uw) ring_put(Rg, Sm);   /* (dense heads: the slot's per-view targets are read in the student loop: held until after it) */
        const float oramp_w = oramp > 0 && step < oramp ? (float)step / oramp : 1.f;
        const float tt = step < tt_warm ? tt0 + (tt1 - tt0) * (float)step / (float)tt_warm : tt1;
        const float sched = step <= warm ? (float)step / warm : 1.f;   /* (upstream: constant after the warm-up) */
        const float mom = ema1 - (ema1 - ema0) * 0.5f * (1.f + cosf((float)M_PI * (float)(step < steps ? step : steps) / (float)steps));
        int S0 = 0, Cc = 0, nreg = 0;
        RP(1);
        /* teacher: both globals -> CLS -> head (EMA) -> logits rows 0, 1 of the Sinkhorn buffer */
        /* iBOT masks: per masked global v its masked token cells (labelled points' cells kept visible), the kept tokens (centre inside
           the other view, at most MI): student voxel in v, teacher voxel in the other view */
        int nmi[2] = {0, 0}, masked[2] = {0, 0};
        static uint8_t *cellm[2];
        if (!cellm[0]) for (int v = 0; v < 2; v++) cellm[v] = malloc((size_t)GT * GT * GT);
        for (int v = 0; v < 2 && wib > 0; v++) {
            if (rng_u(&ar) >= pmask) continue;
            const double ratio = mrlo + rng_u(&ar) * (mrhi - mrlo);
            dn_block_mask(cellm[v], GT, (int)lrint(ratio * GT * GT * GT), &ar);
            for (int i = 0; i < nview[v]; i++) { const int *q = &vxyz[v][3 * i]; cellm[v][((size_t)(q[0] / TPS) * GT + q[1] / TPS) * GT + q[2] / TPS] = 0; }
            masked[v] = 1; int seen = 0;
            for (int c = 0; c < GT * GT * GT; c++) {
                if (!cellm[v][c]) continue;
                const int cz = c / (GT * GT), cy = (c / GT) % GT, cx = c % GT, h2 = TPS / 2;
                const double vv[3] = {cz * TPS + h2, cy * TPS + h2, cx * TPS + h2}; double sx[3], o[3]; dr_map(&xf[v], vv, sx); dr_unmap(&xf[1 - v], sx, o);
                int q[3], in = 1; for (int d = 0; d < 3; d++) { in &= o[d] >= -0.5 && o[d] <= P - 0.5; q[d] = (int)lrint(o[d]); q[d] = q[d] < 0 ? 0 : q[d] >= P ? P - 1 : q[d]; }
                if (!in) continue;
                int k = seen++; if (k >= MI) { k = (int)(rng_next(&ar) % (uint64_t)seen); if (k >= MI) continue; }
                ihS[v * MI + k] = (int)((vv[0] * P + vv[1]) * P + vv[2]); ihT[v * MI + k] = (q[0] * P + q[1]) * P + q[2];
            }
            nmi[v] = seen < MI ? seen : MI;
            if (nmi[v] < 16) { masked[v] = 0; nmi[v] = 0; continue; }
            nn_h2d(isidx + v * MI, ihS + v * MI, (size_t)nmi[v] * 4); nn_h2d(itidx + v * MI, ihT + v * MI, (size_t)nmi[v] * 4);
        }
        unet_use_ema(u, 1);
        for (int v = 0; v < 2; v++) {
            unet_forward(u, xg + v * g3, xsg, 0);
            const float *tok = unet_tokens(u, nullptr, &S0, &Cc, &nreg);
            if (cls_reg) nn_d2d(clsT + (size_t)v * C, tok, (size_t)C * 4); else dr_pool_fwd(tok, 1, S0, C, nreg, clsT + (size_t)v * C);
            const int w = 1 - v;   /* the masked view whose targets this teacher view provides */
            if (masked[w]) dn_gather_slabs(&HS, itidx + w * MI, nmi[w], ieT + (size_t)w * MI * ED, inT + w * MI, nullptr, 0, nullptr, nullptr);
            if (ntot) {   /* the object (and ink) points of this view (teacher keys of the other view's anchors), object means through the teacher region head */
                dn_gather_slabs(&HS, o_idx + v * NOT, ntot, oT + (size_t)v * NOT * ED, onT + v * NOT, nullptr, 0, nullptr, nullptr);
                if (npi) { if (dr_mlp2_fwd(&IH, ipt, oT + ((size_t)v * NOT + nop) * ED, npi + nni, iT + (size_t)v * NIK * 32, iwt)) return 1; }
            }
            if (nop) {
                if (wreg > 0) { dr_gmean(oT + (size_t)v * NOT * ED, o_grp, nop, ED, nob, omT + (size_t)v * 64 * ED, ocnt + v * 64); if (dr_mlp2_fwd(&RH, rpt, omT + (size_t)v * 64 * ED, nob, oyT + (size_t)v * 64 * 128, rwt)) return 1; }
            }
        }
        unet_use_ema(u, 0);
        RP(2);
        if (cls_norm) dr_rownorm_fwd(clsT, 2, C, clsT, clsRS);
        D.w.zc_m = zcm;   /* (the teacher rows update the z centre) */
        if (dr_head_fwd(&D.H, D.pt, clsT, 2, tl, &D.w)) return 1;
        D.w.zc_m = 0.f;
        if (dcm > 0) { dr_teacher_probs(tl, 2, K, tt, 0, dcen, tq, sks); dr_center_update(dcen, tl, 2, K, dcm); }   /* (centre: before its update, as DINOv1) */
        else dr_teacher_probs(tl, 2 + qfill, K, tt, sk_it, nullptr, tq, sks);   /* (rows 0, 1 current; the queue after them) */
        dr_argmax_rows(tq, 2, K, d_am); nn_d2h(h_am, d_am, 8); useb[h_am[0]] = useb[h_am[1]] = 1;
        for (int v = 0; v < 2; v++) if (masked[v]) {   /* iBOT teacher targets: Sinkhorn over the view's masked tokens */
            if (dr_head_fwd(&I.H, I.pt, ieT + (size_t)v * MI * ED, nmi[v], ibl, &I.w)) return 1;
            if (icm > 0) { dr_teacher_probs(ibl, nmi[v], KI, tt, 0, icen, ibq[v], ibsk); dr_center_update(icen, ibl, nmi[v], KI, icm); }
            else dr_teacher_probs(ibl, nmi[v], KI, tt, sk_it, nullptr, ibq[v], ibsk);
            ibv = v;
            dr_argmax_rows(ibq[v], nmi[v], KI, d_am); nn_d2h(h_am, d_am, (size_t)nmi[v] * 4); for (int i = 0; i < nmi[v]; i++) useib[h_am[i]] = 1;
        }
        RP(3);
        /* student */
        unet_zero_grad(u); unet_zero_grad(ul); nn_zero(outs, 64 * 4);
        if (tgn < (size_t)S0 * C) { nn_free(tgrad); tgn = (size_t)S0 * C; tgrad = nn_malloc(tgn * 4); }
        float lsum[6] = {0};   /* dino, koleo, gt, var, ibot */
        if (wreg > 0) nn_zero(rg, RH.n * 4);
        if (NIK > 0) nn_zero(igr, IH.n * 4);
        if ((step - 1) % hacc == 0) nn_zero(D.g, D.H.n * 4);   /* (--head-accum: the DINO head's gradient summed over hacc steps) */
        if (wib > 0) nn_zero(I.g, I.H.n * 4);
        const int nmv = masked[0] + masked[1];
        for (int v = 0; v < 2 + nloc; v++) {
            const int gl = v < 2; unet *n = gl ? u : ul;
            const float *x = gl ? xg + v * g3 : xl + (size_t)(v - 2) * l3;
            RP(9);
            if (gl && masked[v]) { nn_h2d(imask, cellm[v], (size_t)GT * GT * GT); dr_mask_cells(xg + v * g3, P, TPS, imask, 0.f); }   /* (the teacher saw it unmasked) */
            unet_forward(n, x, gl ? xsg : xsl, 1);
            RP(4);
            int dense_v = 0;
            if (HS.uw) {   /* the dense heads of a global view: its targets up, pass 0 (statistics) over the slabs */
                UW.uw = gl && wuw > 0 && Sm->uwv[v]; UW.ik = gl && wik > 0 && Sm->ikv;
                const float rmp = dense_ramp > 0 ? fminf(fmaxf((float)(step - dense_from) / dense_ramp, 0.f), 1.f) : 1.f;
                UW.lp = UW.ln = wuw * rmp; UW.lw4 = wuw * w4l * rmp; UW.lik = wik * rmp;
                if (UW.uw) { nn_h2d(d_pv, Sm->pv[v], g3); nn_h2d(d_w4, Sm->w4[v], g3); nn_h2d(d_wc, Sm->wc[v], g3); UW.pv = d_pv; UW.w4 = d_w4; UW.wc = d_wc;
                             UW.um = sr->um * (float)xf[v].m / (float)P; UW.trunc = UW.ttau > 0 && step >= trunc_from; }
                if (UW.ik) { nn_h2d(d_ikt, Sm->iv[v], g3); nn_h2d(d_ikm, Sm->im[v], g3); UW.ik_t = d_ikt; UW.ik_m = d_ikm; }
                dense_v = UW.uw || UW.ik;
                if (dense_v) dn_uw_stats(&HS);
            }
            int Sv = 0; const float *tok = unet_tokens(n, nullptr, &Sv, nullptr, nullptr);
            if (cls_reg) nn_d2d(clsS, tok, (size_t)C * 4); else dr_pool_fwd(tok, 1, Sv, C, nreg, clsS);
            if (cls_norm) dr_rownorm_fwd(clsS, 1, C, clsS, clsRS + 4);
            if (dr_head_fwd(&D.H, D.p, clsS, 1, sl, &D.w)) return 1;
            int ps_[2] = {0, 0}, pt_[2] = {gl ? 1 - v : 0, 1}; float pw_[2] = {wdino / terms, wdino / terms}; const int npair = gl ? 1 : 2;
            nn_h2d(pS, ps_, 8); nn_h2d(pT, pt_, 8); nn_h2d(pW, pw_, 8);
            dr_ce(sl, 1, K, ts, tq, pS, pT, pW, npair, dsl, outs + 16, ces);
            nn_zero(dcls, (size_t)C * 4);
            if (dr_head_bwd(&D.H, D.p, D.g, clsS, dsl, 1, dcls, step > freeze, &D.w)) return 1;
            if (gl && wk > 0 && kfill > 0) dr_koleo_q(clsS, 1, kq, kfill, C, wk / 2, dcls, outs + 17);
            if (cls_norm) dr_rownorm_bwd(dcls, clsS, clsRS + 4, 1, C, dcls);   /* (d / d normed CLS -> d / d pooled CLS) */
            /* d / d CLS -> token gradient (the backward's scaled domain) */
            nn_zero(tgrad, (size_t)Sv * C * 4);
            if (cls_reg) nn_axpy(tgrad, gs, dcls, (size_t)C); else { dn_axpby(dcls, dcls, (size_t)C, gs, 0.f); dr_pool_bwd(dcls, 1, Sv, C, nreg, tgrad); }
            unet_set_token_grad(n, tgrad);
            /* dense GT SupCon on the view's labelled voxels (globals) */
            HS.ngt = 0; HS.nssl = 0;
            if (gl && wgt > 0 && nview[v] >= 64) {
                const int ng = nview[v]; nn_h2d(d_idx, vidx[v], (size_t)ng * 4); nn_h2d(d_code, vcode[v], (size_t)ng * 4); nn_h2d(d_xyz, vxyz[v], (size_t)ng * 12);
                dn_gather_slabs(&HS, d_idx, ng, es, ns, nullptr, 0, nullptr, nullptr);   /* (iBOT points gathered below) */
                dn_supcon(es, d_code, d_xyz, ng, nullptr, d_code, d_xyz, 0, ED, tau, hw, hr, ga, outs + 20, scr);
                if (wvar > 0) dn_varreg(es, ng, ED, vgam, wvar, ga, outs + 24);
                dn_axpby(ga, ga, (size_t)ng * ED, wgt, 0.f);
                HS.e1 = es; HS.g1 = ga; HS.n1 = ns; HS.i1 = d_idx; HS.ngt = ng;
            }
            if (gl && masked[v]) {   /* iBOT: the masked tokens' student embeddings -> iBOT head -> cross-entropy against the cross-view targets */
                const int m = nmi[v];
                dn_gather_slabs(&HS, isidx + v * MI, m, ieS, inS, nullptr, 0, nullptr, nullptr);
                if (dr_head_fwd(&I.H, I.p, ieS, m, ibl, &I.w)) return 1;
                float *hw_ = malloc((size_t)m * 4); for (int i = 0; i < m; i++) hw_[i] = wib / (float)m / (float)nmv; nn_h2d(ibw, hw_, (size_t)m * 4); free(hw_);
                dr_ce(ibl, m, KI, ts, ibq[v], iid, iid, ibw, m, ibd, outs + 28, ibce);
                if (dr_head_bwd(&I.H, I.p, I.g, ieS, ibd, m, idE, step > freeze, &I.w)) return 1;
                HS.e2 = ieS; HS.g2 = idE; HS.n2 = inS; HS.i2 = isidx + v * MI; HS.nssl = m; HS.w2 = 1.f;
            }
            HS.n3p = 0;
            if (gl && ntot) {
                if (!nop) { /* (ink only) */ }
                dn_gather_slabs(&HS, o_idx + v * NOT, ntot, oS, onS, nullptr, 0, nullptr, nullptr);
                nn_zero(ogS, (size_t)ntot * ED * 4);
                if (npi) {   /* ink: anchors of this view vs the teacher's other view + the queues (ink anchors, then no-ink anchors) */
                    const int ni = npi + nni, w2 = 1 - v; const float wv = wink * oramp_w * 0.25f;
                    if (dr_mlp2_fwd(&IH, ip, oS + (size_t)nop * ED, ni, iS, iws)) return 1;
                    nn_zero(idS, (size_t)ni * 32 * 4);
                    nn_d2d(iKP, iT + (size_t)w2 * NIK * 32, (size_t)npi * 32 * 4); nn_d2d(iKP + (size_t)npi * 32, iqp, (size_t)iqpf * 32 * 4);
                    nn_d2d(iKN, iT + ((size_t)w2 * NIK + npi) * 32, (size_t)nni * 32 * 4); nn_d2d(iKN + (size_t)nni * 32, iqn, (size_t)iqnf * 32 * 4);
                    for (int k = 0; k < npi + iqpf; k++) hk[k] = k < npi ? k : -1;
                    nn_h2d(ikid, hk, (size_t)(npi + iqpf) * 4);
                    dr_ink(iS, iaid, npi, iKP, ikid, npi + iqpf, iKN, nni + iqnf, 32, 0.15f, 0.75f, wv, idS, outs + 36);
                    for (int k = 0; k < nni + iqnf; k++) hk[k] = k < nni ? k : -1;
                    nn_h2d(ikid, hk, (size_t)(nni + iqnf) * 4);
                    dr_ink(iS + (size_t)npi * 32, iaid, nni, iKN, ikid, nni + iqnf, iKP, npi + iqpf, 32, 0.15f, 0.75f, wv, idS + (size_t)npi * 32, outs + 36);
                    if (dr_mlp2_bwd(&IH, ip, igr, oS + (size_t)nop * ED, iS, idS, ni, ogS + (size_t)nop * ED, iws)) return 1;
                }
                if (wobj > 0) dr_obj(oS, oT + (size_t)(1 - v) * NOT * ED, o_obj, o_ones, o_ones, nop, ED, 0.15f, 0.75f, wobj * oramp_w / 2, ogS, outs + 32);
                if (wreg > 0 && nob >= 2) {
                    dr_gmean(oS, o_grp, nop, ED, nob, omS, ocnt + 128);
                    if (dr_mlp2_fwd(&RH, rp, omS, nob, oyS, rws)) return 1;
                    nn_zero(odyS, (size_t)nob * 128 * 4);
                    dr_obj(oyS, oyT + (size_t)(1 - v) * 64 * 128, o_rid, o_ones, o_ones, nob, 128, 0.15f, 0.75f, wreg * oramp_w / 2, odyS, outs + 34);
                    nn_zero(odm, (size_t)nob * ED * 4);
                    if (dr_mlp2_bwd(&RH, rp, rg, omS, oyS, odyS, nob, odm, rws)) return 1;
                    dr_gmean_bwd(odm, ocnt + 128, o_grp, nop, ED, ogS);
                }
                HS.e3 = oS; HS.g3 = ogS; HS.n3 = onS; HS.i3 = o_idx + v * NOT; HS.n3p = ntot;
            }
            RP(5);
            if (gl) unet_backward(u, glz); else unet_backward(ul, glz);
            RP(6);
            unet_set_token_grad(n, nullptr);
            if (dense_v) {   /* the dense heads' losses of this view (pass 0's statistics) */
                if (UW.uw) { float o2[5], o7[15], w4h[8]; nn_loss_fetch(UW.sscr, (shape5){1, 2, 1, 1, 1}, o2); nn_loss_fetch(UW.nscr, (shape5){1, 7, 1, 1, 1}, o7); nn_d2h(w4h, UW.w4fin, sizeof w4h);
                             double nb = 0; for (int c = 0; c < 7; c++) nb += o7[c];
                             dsum[0] += 0.5 * (o2[0] + o2[1]); dsum[1] += nb / 7; dsum[2] += w4h[0]; dsum[3] += w4h[1]; dn_nuw++;
                             if (getenv("UFSM_DENSE_DBG")) fprintf(stderr, "dense view %d: phase %g %g (active %g) nrm %g w4 %g acc %g N %g comps %g um %.2f gnorm %g\n", v, o2[0], o2[1], o2[4], nb / 7, w4h[0], w4h[1], w4h[2], w4h[4], UW.um, unet_grad_norm(u)); }
                if (UW.ik) { float o1[3]; nn_loss_fetch(UW.iscr, (shape5){1, 1, 1, 1, 1}, o1); dsum[4] += o1[0]; dn_nik++; }
                UW.uw = UW.ik = 0;
            }
            nn_d2h(hout, outs, 40 * 4);
            lsum[0] += hout[16]; if (gl) { lsum[1] += hout[17]; lsum[2] += HS.ngt ? hout[20] : 0.f; lsum[3] += HS.ngt ? hout[24] : 0.f; lsum[4] += masked[v] ? hout[28] : 0.f; }
            if (gl && wk > 0) { nn_d2d(kq + (size_t)kpos * C, clsS, (size_t)C * 4); kpos = (kpos + 1) % KQ; if (kfill < KQ) kfill++; }
        }
        nn_axpy(unet_grad_ptr(u), 1.f, unet_grad_ptr(ul), np);   /* the locals' share */
        if (HS.uw) ring_put(Rg, Sm);
        const double gnorm = unet_grad_norm(u);
        if (!isfinite(gnorm) || !isfinite(lsum[0])) { fprintf(stderr, "dino-train --recipe: non-finite loss / gradient at step %d\n", step); return 1; }
        if (clip > 0) unet_clip_grad(u, clip);
        RP(9);
        unet_muon(u, mlr0 * sched, 0.95f, lr0 * sched, 0.9f, 0.999f, 1e-8f, wd, step);
        RP(8);
        /* heads: Muon on W1 W2 W3, AdamW on the biases and g, the prototype directions per --proto-opt (frozen for --freeze-last steps) */
        dn_head_step(&D, mlr0 * sched, hlr * sched, plr * sched, wd, step, freeze, mom, step % hacc == 0);
        if (wib > 0) dn_head_step(&I, mlr0 * sched, hlr * sched, plr * sched, wd, step, freeze, mom, 1);
        if (wreg > 0) {   /* region head: Muon on its matrices, AdamW on the biases, EMA teacher */
            dr_muon(rp + RH.w1, rg + RH.w1, rmo + RH.w1, 384, ED, mlr0 * sched, 0.95f, wd, rwk);
            dr_muon(rp + RH.w2, rg + RH.w2, rmo + RH.w2, 128, 384, mlr0 * sched, 0.95f, wd, rwk);
            dr_adamw(rp + RH.b1, rg + RH.b1, rm + RH.b1, rv + RH.b1, 384, hlr * sched, 0.9f, 0.999f, 1e-8f, 0.f, step);
            dr_adamw(rp + RH.b2, rg + RH.b2, rm + RH.b2, rv + RH.b2, 128, hlr * sched, 0.9f, 0.999f, 1e-8f, 0.f, step);
            dr_ema(rpt, rp, RH.n, mom);
        }
        if (NIK > 0) {   /* ink head: as the region head; then the teacher ink embeddings of both views into the FIFO queues */
            dr_muon(ip + IH.w1, igr + IH.w1, imo + IH.w1, 384, ED, mlr0 * sched, 0.95f, wd, rwk);
            dr_muon(ip + IH.w2, igr + IH.w2, imo + IH.w2, 32, 384, mlr0 * sched, 0.95f, wd, rwk);
            dr_adamw(ip + IH.b1, igr + IH.b1, im_ + IH.b1, iv_ + IH.b1, 384, hlr * sched, 0.9f, 0.999f, 1e-8f, 0.f, step);
            dr_adamw(ip + IH.b2, igr + IH.b2, im_ + IH.b2, iv_ + IH.b2, 32, hlr * sched, 0.9f, 0.999f, 1e-8f, 0.f, step);
            dr_ema(ipt, ip, IH.n, mom);
            for (int v = 0; v < 2 && npi; v++) {
                for (int k = 0; k < npi; k++) { nn_d2d(iqp + (size_t)iqpp * 32, iT + ((size_t)v * NIK + k) * 32, 128); iqpp = (iqpp + 1) % IQN; if (iqpf < IQN) iqpf++; }
                for (int k = 0; k < nni; k++) { nn_d2d(iqn + (size_t)iqnp * 32, iT + ((size_t)v * NIK + npi + k) * 32, 128); iqnp = (iqnp + 1) % IQN; if (iqnf < IQN) iqnf++; }
            }
        }
        unet_ema(u, mom);
        RP(7);
        /* the teacher logits into the Sinkhorn queue */
        if (SQ > 0) { for (int v = 0; v < 2; v++) { nn_d2d(tl + (size_t)(2 + qpos) * K, tl + (size_t)v * K, (size_t)K * 4); qpos = (qpos + 1) % SQ; if (qfill < SQ) qfill++; } }
        { float o4[6]; nn_d2h(o4, outs + 32, 24); sums[7] += o4[0]; sums[8] += o4[2]; sums[9] += o4[4]; }
        sums[0] += lsum[0]; sums[1] += lsum[1]; sums[2] += lsum[2] / 2; sums[3] += lsum[3] / 2; sums[4] += gnorm; sums[5] += lsum[4]; sums[6] += nmi[0] + nmi[1]; nsum++;
        if (step % log_every == 0) {
            /* teacher target entropy of row 0 and of the batch mean (prototype usage) */
            float *qh = malloc((size_t)2 * K * 4); nn_d2h(qh, tq, (size_t)2 * K * 4);
            double Hr = 0, Hm = 0; for (int k = 0; k < K; k++) { const double a = qh[k], m = 0.5 * (qh[k] + qh[K + k]); if (a > 0) Hr -= a * log(a); if (m > 0) Hm -= m * log(m); }
            if (getenv("UFSM_DINO_DIAG")) {   /* is the CLS informative, do the logits spread, does the head learn (the DINO term stuck at ln K) */
                float *ct = malloc((size_t)2 * C * 4), *qq = malloc((size_t)(kfill > 0 ? kfill : 1) * C * 4), *tlh = malloc((size_t)2 * K * 4), *slh = malloc((size_t)K * 4);
                nn_d2h(ct, clsT, (size_t)2 * C * 4); if (kfill) nn_d2h(qq, kq, (size_t)kfill * C * 4); nn_d2h(tlh, tl, (size_t)2 * K * 4); nn_d2h(slh, sl, (size_t)K * 4);
                double na = 0, nb = 0, ab = 0; for (int c = 0; c < C; c++) { na += (double)ct[c] * ct[c]; nb += (double)ct[C + c] * ct[C + c]; ab += (double)ct[c] * ct[C + c]; }
                double pc = 0; int npc = 0, nq = kfill;   /* mean pairwise cosine over the queue (student CLS of earlier crops: other samples) */
                for (int i = 0; i < nq; i++) for (int j = i + 1; j < nq; j++) { double x = 0, y = 0, z = 0; for (int c = 0; c < C; c++) { x += (double)qq[(size_t)i * C + c] * qq[(size_t)j * C + c]; y += (double)qq[(size_t)i * C + c] * qq[(size_t)i * C + c]; z += (double)qq[(size_t)j * C + c] * qq[(size_t)j * C + c]; } pc += x / sqrt(y * z + 1e-30); npc++; }
                double sd = 0, mg = 0;   /* per-dimension std across the queue, relative to the mean |value| */
                for (int c = 0; c < C && nq > 1; c++) { double m = 0, v = 0; for (int i = 0; i < nq; i++) m += qq[(size_t)i * C + c]; m /= nq; for (int i = 0; i < nq; i++) { const double d = qq[(size_t)i * C + c] - m; v += d * d; } sd += sqrt(v / nq); mg += fabs(m); }
                double tm = 0, ts2 = 0, tmx = -1e30, tmn = 1e30; int ta = 0; for (int k = 0; k < K; k++) { tm += tlh[k]; ts2 += (double)tlh[k] * tlh[k]; if (tlh[k] > tmx) { tmx = tlh[k]; ta = k; } if (tlh[k] < tmn) tmn = tlh[k]; }
                tm /= K; const double tsd = sqrt(fmax(ts2 / K - tm * tm, 0));
                double sm = 0, ss2 = 0, smx = -1e30; int sa = 0; for (int k = 0; k < K; k++) { sm += slh[k]; ss2 += (double)slh[k] * slh[k]; if (slh[k] > smx) { smx = slh[k]; sa = k; } }
                sm /= K; const double ssd = sqrt(fmax(ss2 / K - sm * sm, 0));
                double z = 0, Hs = 0; for (int k = 0; k < K; k++) z += exp((slh[k] - smx) / ts); for (int k = 0; k < K; k++) { const double pr = exp((slh[k] - smx) / ts) / z; if (pr > 0) Hs -= pr * log(pr); }
                int tb = 0; for (int k = 1; k < K; k++) if (tlh[K + k] > tlh[K + tb]) tb = k;
                const size_t off[9] = {D.H.w1, D.H.b1, D.H.w2, D.H.b2, D.H.w3, D.H.b3, D.H.v, D.H.g, D.H.n}; const char *nm9[8] = {"W1", "b1", "W2", "b2", "W3", "b3", "V", "g"};
                float *gh = malloc(D.H.n * 4); nn_d2h(gh, D.g, D.H.n * 4); char gb[256]; int gl_ = 0;
                for (int t = 0; t < 8; t++) { double q2 = 0; for (size_t i = off[t]; i < off[t + 1]; i++) q2 += (double)gh[i] * gh[i]; gl_ += snprintf(gb + gl_, sizeof gb - gl_, " %s %.2e", nm9[t], sqrt(q2)); }
                fprintf(stderr, "  diag: CLS |a| %.3g |b| %.3g cos(a,b) %.4f | queue %d: pairwise cos %.4f, dim std / |mean| %.4f | teacher logits sd %.4f range %.3f..%.3f argmax %d %d | student (last view) logits sd %.4f, H %.2f, argmax %d | dino head |g|:%s\n",
                        sqrt(na), sqrt(nb), ab / sqrt(na * nb + 1e-30), nq, npc ? pc / npc : 0.0, mg > 0 ? sd / mg : 0.0, tsd, tmn, tmx, ta, tb, ssd, Hs, sa, gb);
                free(ct); free(qq); free(tlh); free(slh); free(gh);
            }
            double Hi = -1;   /* iBOT: the mean teacher-target entropy of the last masked view's first 4 tokens (renormalised rows) */
            if (wib > 0 && ibv >= 0 && nmi[ibv] >= 4) {
                float *ih = malloc((size_t)4 * KI * 4); nn_d2h(ih, ibq[ibv], (size_t)4 * KI * 4); Hi = 0;
                for (int r = 0; r < 4; r++) { double z = 0, h = 0; for (int k = 0; k < KI; k++) z += ih[(size_t)r * KI + k];
                    for (int k = 0; k < KI; k++) { const double a = ih[(size_t)r * KI + k] / z; if (a > 0) h -= a * log(a); } Hi += h / 4; }
                free(ih);
            }
            free(qh);
            const double now = dn_now(), dt = (now - t_last) / nsum;
            char line[1024];
            /* prototype usage: distinct teacher-argmax prototypes since the last reset (every --usage-window steps; DINO sees 2 rows per step) */
            const int uwin = atoi(opt(argc, argv, "--usage-window", "5000")), rst = uwin > 0 && step % uwin < log_every;
            int ud = 0, ui = 0; for (int k = 0; k < K; k++) { ud += useb[k]; if (rst) useb[k] = 0; } for (int k = 0; k < KI && wib > 0; k++) { ui += useib[k]; if (rst) useib[k] = 0; }
            snprintf(line, sizeof line, "step %d dino %.4f ibot %.4f (%.0f tokens) obj %.4f region %.4f ink %.4f koleo %.4f gt %.4f var %.4f | H(t) %.2f H(mean) %.2f (max %.2f) H(ibot t) %.2f (max %.2f) | protos used %d / %d, ibot %d / %d | tt %.3f ema %.5f | gnorm %.3f | %.0f ms/step %.1f Mvox/s",
                     step, sums[0] / nsum, sums[5] / nsum, sums[6] / nsum, sums[7] / nsum, sums[8] / nsum, sums[9] / nsum, sums[1] / nsum, sums[2] / nsum, sums[3] / nsum, Hr, Hm, log((double)K), Hi, log((double)KI), ud, K, ui, KI, tt, mom, sums[4] / nsum, dt * 1e3, (double)(2 * g3 + nloc * l3) / dt / 1e6);
            if (HS.uw) { const size_t ln_ = strlen(line); snprintf(line + ln_, sizeof line - ln_, " | dense: phase %.4f nrm %.4f w4 %.4f acc %.3f (%d views) inkp %.4f (%d views)",
                                     dn_nuw ? dsum[0] / dn_nuw : 0.0, dn_nuw ? dsum[1] / dn_nuw : 0.0, dn_nuw ? dsum[2] / dn_nuw : 0.0, dn_nuw ? dsum[3] / dn_nuw : 0.0, dn_nuw, dn_nik ? dsum[4] / dn_nik : 0.0, dn_nik);
                         memset(dsum, 0, sizeof dsum); dn_nuw = dn_nik = 0; }
            fprintf(stderr, "%s\n", line); if (lf) { fprintf(lf, "%s\n", line); fflush(lf); }
            memset(sums, 0, sizeof sums); nsum = 0; t_last = dn_now();
            if (prof) { fprintf(stderr, "  phases (ms/step): data %.1f views %.1f teacher %.1f head+sinkhorn %.1f student fwd %.1f head/losses %.1f bwd %.1f head optim %.1f unet optim %.1f other %.1f\n",
                                pt[0] * 1e3 / log_every, pt[1] * 1e3 / log_every, pt[2] * 1e3 / log_every, pt[3] * 1e3 / log_every, pt[4] * 1e3 / log_every, pt[5] * 1e3 / log_every, pt[6] * 1e3 / log_every, pt[7] * 1e3 / log_every, pt[8] * 1e3 / log_every, pt[9] * 1e3 / log_every);
                        memset(pt, 0, sizeof pt); }
            if (access(stopf, F_OK) == 0) { fprintf(stderr, "dino-train: STOP file\n"); g_dn_stop = 1; }
        }
        if (step % save_every == 0 || step == steps || g_dn_stop) {
            char cp[1200], ex[256]; snprintf(cp, sizeof cp, "%s/step%06d.ckpt", outd, step);
            if (CO >= ED + 12) snprintf(ex, sizeof ex, "\"task\":\"dino\",\"recipe\":1,\"embed_dim\":%d,\"dense\":{\"phase\":%d,\"normals\":%d,\"wind4\":%d,\"ink_prob\":%d}", ED, ED, ED + 2, ED + 9, ED + 11);
            else snprintf(ex, sizeof ex, "\"task\":\"dino\",\"recipe\":1,\"embed_dim\":%d", ED);
            if (unet_save(u, cp, step, ex)) fprintf(stderr, "dino-train: cannot save %s\n", cp);
            char hpth[1300]; snprintf(hpth, sizeof hpth, "%s.heads", cp);   /* DINO head (student, teacher), then the iBOT head's */
            FILE *hf = fopen(hpth, "wb");
            if (hf) { for (int k = 0; k < (wib > 0 ? 2 : 1); k++) { dn_head *hh = k ? &I : &D; float *h = malloc(hh->H.n * 4);
                          nn_d2h(h, hh->p, hh->H.n * 4); fwrite(h, 4, hh->H.n, hf); nn_d2h(h, hh->pt, hh->H.n * 4); fwrite(h, 4, hh->H.n, hf); free(h); }
                      { float *h = malloc((RH.n > IH.n ? RH.n : IH.n) * 4);   /* then the region and ink heads (student, teacher; also with iBOT off) */
                          nn_d2h(h, rp, RH.n * 4); fwrite(h, 4, RH.n, hf); nn_d2h(h, rpt, RH.n * 4); fwrite(h, 4, RH.n, hf);
                          nn_d2h(h, ip, IH.n * 4); fwrite(h, 4, IH.n, hf); nn_d2h(h, ipt, IH.n * 4); fwrite(h, 4, IH.n, hf); free(h); }
                      fclose(hf); }
            char lp[1200]; snprintf(lp, sizeof lp, "%s/last.ckpt", outd);
            { char prev[1300]; const ssize_t pl = readlink(lp, prev, sizeof prev - 1); const int keep = atoi(opt(argc, argv, "--keep-every", "0"));   /* rotation: the previous checkpoint goes unless a milestone */
              if (pl > 0 && keep > 0) { prev[pl] = 0; int ps_ = 0; if (sscanf(prev, "step%d.ckpt", &ps_) == 1 && ps_ != step && ps_ % keep) { char a1[2600], a2[2600], a3[2600]; snprintf(a1, sizeof a1, "%s/%s", outd, prev); snprintf(a2, sizeof a2, "%s.heads", a1); snprintf(a3, sizeof a3, "%s.protos", a1); unlink(a1); unlink(a2); unlink(a3); } } }
            unlink(lp); if (symlink(cp + strlen(outd) + 1, lp)) {}
        }
        if (opt(argc, argv, "--eval", nullptr) && (step % eval_every == 0 || step == steps)) { dn_run_eval(u, argc, argv, step, P, ED, nthreads, 0, 0.f, outd, lf); t_last = dn_now(); }
    }
    pthread_mutex_lock(&Rg->mu); Rg->stop = 1; pthread_cond_broadcast(&Rg->cv); pthread_mutex_unlock(&Rg->mu);
    for (int i = 0; i < Rg->nworkers; i++) pthread_join(Rg->th[i], nullptr);
    if (lf) fclose(lf);
    return 0;
}

int cmd_dino_train(int argc, char **argv) {
    const char *srcf = opt(argc, argv, "--sources", nullptr), *outd = opt(argc, argv, "--out", nullptr);
    if (!srcf || !outd) {
        fprintf(stderr, "usage: ufsm dino-train --sources S.json --out DIR [--P 256] [--widths 16,32,64,96,128,192] [--steps 20000] [--lr 1e-3] [--muon-lr 0.02]\n"
                        "         [--wd 0.01] [--ema 0.996] [--warmup 500] [--gt-points 2048] [--ssl-points 2048] [--tau 0.1] [--hard-w 4] [--hard-r 48]\n"
                        "         [--ssl 0.3 (weight; 0 = GT contrastive only)] [--protos 1024] [--ts 0.1] [--tt 0.04] [--center-m 0.9]\n"
                        "         [--noise 0.15] [--gain-jit 0.15] [--mask-frac 0] [--mask-cube 16] [--seed 1] [--workers 4] [--threads 4]\n"
                        "         [--eval SURF-LIST --eval-root R --eval-key K --eval-box z,y,x,nz,ny,nx --eval-every 2000 --eval-codes ROOT KEY --eval-norm c,s]\n"
                        "         [--log-every 20] [--save-every 2000] [--resume CKPT] [--dim 32] [--mx 0 (bf16, no MX)] [--var 30] [--var-gamma 0.8 / sqrt(D)] [--gpu 0]\n"
                        "         [--attn-blocks 0 (transformer blocks on the bottleneck)] [--attn-pe peg|rope|none] [--attn-reg 0] [--attn-mlp 0 (8/3 width)]\n"
                        "         [--minutes M (wall-time budget: the step count from the speed of steps 21..200; evals / saves at the end)]\n"
                        "         [--head-slab Z (the head per z-slab of Z planes: no whole-window logits; large windows)] [--mem default|lean]\n"
                        "  Our dense embedding net from scratch (src/dino.h): the U-Net with D = 32 unit embedding channels; GT sheet-identity SupCon on\n"
                        "  labelled voxels (primary) + DINO prototype cross-view term on papyrus voxels (EMA teacher on the other view). STOP file in --out stops.\n");
        return 2;
    }
    const int P = atoi(opt(argc, argv, "--P", "256")), fp32 = !atoi(opt(argc, argv, "--mx", "1")); int steps = atoi(opt(argc, argv, "--steps", "20000"));
    const double minutes = atof(opt(argc, argv, "--minutes", "0"));   /* > 0: a wall-time budget; the step count follows from the speed of steps 21..200 */
    const float lr0 = (float)atof(opt(argc, argv, "--lr", "1e-3")), mlr0 = (float)atof(opt(argc, argv, "--muon-lr", "0.02")), wd = (float)atof(opt(argc, argv, "--wd", "0.01"));
    const float ema = (float)atof(opt(argc, argv, "--ema", "0.996")), clip = (float)atof(opt(argc, argv, "--clip", "5"));
    const int warm = atoi(opt(argc, argv, "--warmup", "500"));
    const float tau = (float)atof(opt(argc, argv, "--tau", "0.1")), hw = (float)atof(opt(argc, argv, "--hard-w", "4")), hr = (float)atof(opt(argc, argv, "--hard-r", "48"));
    const float wssl = (float)atof(opt(argc, argv, "--ssl", "0.3")), ts = (float)atof(opt(argc, argv, "--ts", "0.1")), tt = (float)atof(opt(argc, argv, "--tt", "0.04"));
    const float cm = (float)atof(opt(argc, argv, "--center-m", "0.9")), noise = (float)atof(opt(argc, argv, "--noise", "0.15")), gj = (float)atof(opt(argc, argv, "--gain-jit", "0.15"));
    const float wgt = (float)atof(opt(argc, argv, "--gt", "1"));   /* the GT contrastive term's weight */
    const int use_teacher = atoi(opt(argc, argv, "--teacher", "1"));   /* 0: no EMA teacher forward (debug; GT term over the student points only) */
    const int ED = atoi(opt(argc, argv, "--dim", "32"));   /* embedding channels (16 or 32) */
    if (ED != 16 && ED != 32) { fprintf(stderr, "dino-train: --dim 16 or 32\n"); return 2; }
    const float wvar = (float)atof(opt(argc, argv, "--var", "30")), vgam = (float)atof(opt(argc, argv, "--var-gamma", "0")) > 0 ? (float)atof(opt(argc, argv, "--var-gamma", "0")) : 0.8f / sqrtf((float)ED);
    const float mfrac = (float)atof(opt(argc, argv, "--mask-frac", "0")); const int mcube = atoi(opt(argc, argv, "--mask-cube", "16"));
    const int K = atoi(opt(argc, argv, "--protos", "1024")), log_every = atoi(opt(argc, argv, "--log-every", "20")), save_every = atoi(opt(argc, argv, "--save-every", "2000"));
    const int eval_every = atoi(opt(argc, argv, "--eval-every", "2000")), nthreads = atoi(opt(argc, argv, "--threads", "4")), nwork = atoi(opt(argc, argv, "--workers", "4"));
    const uint64_t seed = strtoull(opt(argc, argv, "--seed", "1"), nullptr, 10);
    dn_data *Dd = calloc(1, sizeof *Dd); Dd->P = P; Dd->ngt = atoi(opt(argc, argv, "--gt-points", "2048")); Dd->nssl = atoi(opt(argc, argv, "--ssl-points", "2048")); Dd->margin = 8; Dd->nair = atoi(opt(argc, argv, "--air-points", "0"));
    { char *t = strdup(opt(argc, argv, "--exclude-src", "")); for (char *q = strtok(t, ","); q && Dd->nexcl < 8; q = strtok(nullptr, ",")) snprintf(Dd->excl[Dd->nexcl++], 64, "%s", q); free(t); }
    Dd->share_u = atof(opt(argc, argv, "--unlabelled-share", "-1"));   /* the ct_only sources' share of the draws (< 0: by weight over all; 0: none) */
    const int recipe = atoi(opt(argc, argv, "--recipe", "0"));   /* the upstream dinovol recipe (dn_recipe): the sampler reads source windows */
    if (recipe) { Dd->P = atoi(opt(argc, argv, "--source-P", "384")); Dd->ident = 1; Dd->nssl = 0; Dd->nink = atof(opt(argc, argv, "--ink", "0.1")) > 0 ? atoi(opt(argc, argv, "--ink-points", "512")) : 0; }
    if (mkdir(outd, 0755) && errno != EEXIST) { fprintf(stderr, "dino-train: cannot create %s\n", outd); return 1; }
    dn_precision(fp32);
    if (nn_init(atoi(opt(argc, argv, "--gpu", "0")))) return 1;
    dn_precision_after_init(fp32);
    if (load_sources(srcf, Dd, nthreads)) return 1;
    const int axis_in = atoi(opt(argc, argv, "--axis-input", "0")); const float ars = (float)atof(opt(argc, argv, "--axis-rs", "0.001"));   /* the axis-distance channel and its scale */
    if (axis_in) for (int i = 0; i < Dd->nsrc; i++) if (!Dd->src[i].axz) { fprintf(stderr, "dino-train: --axis-input needs \"axis\" in every source (%s)\n", Dd->src[i].name); return 2; }
    unet_cfg cfg = {0}; cfg.cin = axis_in ? 2 : 1; cfg.cout = ED; cfg.G = 8;
    if (recipe && (atof(opt(argc, argv, "--unwrap2", "0")) > 0 || atof(opt(argc, argv, "--ink-prob", "0")) > 0)) {   /* dense heads after the embedding (dn_uw) */
        if (ED != 32) { fprintf(stderr, "dino-train --recipe: the dense heads need --dim 32\n"); return 2; }
        cfg.cout = (ED + 12 + 15) / 16 * 16;
    } cfg.down_norm = fp32 ? 0 : 1;
    { char *t = strdup(opt(argc, argv, "--widths", "16,32,64,96,128,192")); for (char *q = strtok(t, ","); q && cfg.nlev < UNET_MAXLEV; q = strtok(nullptr, ",")) cfg.widths[cfg.nlev++] = atoi(q); free(t); }
    if (P % (1 << (cfg.nlev - 1))) { fprintf(stderr, "dino-train: P must be a multiple of %d\n", 1 << (cfg.nlev - 1)); return 2; }
    {   /* the bottleneck transformer stage (src/attn.h) */
        cfg.attn_blocks = atoi(opt(argc, argv, "--attn-blocks", "0")); cfg.attn_reg = atoi(opt(argc, argv, "--attn-reg", "0")); cfg.attn_mlp = atoi(opt(argc, argv, "--attn-mlp", "0"));
        const char *pe = opt(argc, argv, "--attn-pe", "peg");
        cfg.attn_pe = !strcmp(pe, "peg") ? 0 : !strcmp(pe, "rope") ? 1 : !strcmp(pe, "none") ? 2 : -1;
        if (cfg.attn_pe < 0) { fprintf(stderr, "dino-train: --attn-pe peg|rope|none\n"); return 2; }
    }
    unet *u = unet_create(&cfg); unet_init(u, seed); unet_head_bias_zero(u);   /* (no shared -2 bias direction in the embeddings) */
    const int HZ = atoi(opt(argc, argv, "--head-slab", "0"));   /* head slabs of HZ planes: no whole-window logits / logit gradient (large windows) */
    if (HZ && (fp32 || P % HZ)) { fprintf(stderr, "dino-train: --head-slab needs MX storage (--mx 1) and P a multiple of it\n"); return 2; }
    const char *mem = opt(argc, argv, "--mem", "default");   /* lean: train's smallest recompute-1 mode (chunked MX gradients, lean 2, shared encoder a1) */
    if (!strcmp(mem, "lean")) {
        if (fp32) { fprintf(stderr, "dino-train: --mem lean needs MX storage\n"); return 2; }
        unet_set_chunk_up(2); unet_set_recompute(1); unet_set_grad_mx8(1); unet_set_lean(2); unet_set_share_enc_a1(1);
    } else if (strcmp(mem, "default")) { fprintf(stderr, "dino-train: --mem default|lean\n"); return 2; }
    if (getenv("UFSM_MEM_DRY") && ufsm_env_on("UFSM_MEM_DRY")) {   /* the network's training bytes at this window (dry build) + this trainer's buffers */
        unet_set_head_ext(u, HZ > 0);
        const size_t p3d = (size_t)P * P * P, tb = unet_train_bytes(u, (shape5){1, cfg.cin, P, P, P});
        const size_t tr = 2 * p3d + p3d * 4 * cfg.cin + (HZ ? (size_t)cfg.cout * HZ * P * P * 2 : p3d * cfg.cout * 4) + (cfg.cout > ED ? 5 * p3d + (size_t)11 * HZ * P * P : 0);   /* (+ the dense targets) */
        fprintf(stderr, "dino-train: UFSM_MEM_DRY: P %d, mem %s, head slabs %d, attention blocks %d: network %.2f GB (%.1f B per voxel), trainer buffers %.2f GB\n",
                P, mem, HZ, cfg.attn_blocks, tb / 1e9, (double)tb / p3d, tr / 1e9);
        return 0;
    }
    if (!fp32 && ED <= 16 && atoi(opt(argc, argv, "--fp16-logits", "0"))) unet_set_fp16_logits(u, 1);   /* (the MX fp16-logit head takes <= 16 outputs) */
    int step0 = 0;
    const char *resume = opt(argc, argv, "--resume", nullptr);
    if (resume && !recipe) { step0 = unet_load(u, resume); if (step0 < 0) { fprintf(stderr, "dino-train: cannot resume %s\n", resume); return 1; } }   /* (the recipe loads its own) */
    if (recipe) {
        if (!HZ) { fprintf(stderr, "dino-train --recipe: needs --head-slab Z (and MX storage)\n"); return 2; }
        return dn_recipe(u, &cfg, Dd, argc, argv, P, steps, minutes, outd, nthreads, nwork, seed, HZ);
    }
    /* prototypes */
    const size_t kd = (size_t)K * ED;
    float *W = nn_malloc(kd * 4), *Wt = nn_malloc(kd * 4), *gW = nn_malloc(kd * 4), *Wm = nn_malloc(kd * 4), *Wv = nn_malloc(kd * 4), *cen = nn_malloc((size_t)K * 4), *tmean = nn_malloc((size_t)K * 4);
    {
        float *h = malloc(kd * 4); uint64_t rs = seed * 7919 + 3;
        for (size_t i = 0; i < kd; i++) { const double u1 = rng_u(&rs) + 1e-12, u2 = rng_u(&rs); h[i] = (float)(sqrt(-2 * log(u1)) * cos(2 * M_PI * u2)); }
        char pp[1200]; snprintf(pp, sizeof pp, "%s.protos", resume ? resume : "");
        FILE *pf = resume ? fopen(pp, "rb") : nullptr;
        if (pf) { if (fread(h, 4, kd, pf) != kd) fprintf(stderr, "dino-train: short %s\n", pp); fclose(pf); }
        nn_h2d(W, h, kd * 4); nn_h2d(Wt, h, kd * 4); free(h);
        nn_zero(Wm, kd * 4); nn_zero(Wv, kd * 4); nn_zero(cen, (size_t)K * 4);
    }
    /* device buffers */
    const size_t p3 = (size_t)P * P * P; const int NG = Dd->ngt + Dd->nair, NS = Dd->nssl;
    uint8_t *cta = nn_malloc(p3), *ctb = nn_malloc(p3); float *x = nn_malloc(p3 * 4 * cfg.cin), *gl = nn_malloc(HZ ? 4096 : p3 * ED * 4), *d_ayx = nn_malloc((size_t)P * 8);
    void *hsl = HZ ? nn_malloc((size_t)ED * HZ * P * P * 2) : nullptr;
    int *d_gidx = nn_malloc((size_t)NG * 4), *d_gcode = nn_malloc((size_t)NG * 4), *d_gxyz = nn_malloc((size_t)NG * 12), *d_sidx = nn_malloc((size_t)NS * 4);
    float *es_g = nn_malloc((size_t)NG * ED * 4), *et_g = nn_malloc((size_t)NG * ED * 4), *ns_g = nn_malloc((size_t)NG * 4), *nt_g = nn_malloc((size_t)NG * 4), *ga_g = nn_malloc((size_t)NG * ED * 4);
    float *es_s = nn_malloc((size_t)NS * ED * 4), *et_s = nn_malloc((size_t)NS * ED * 4), *ns_s = nn_malloc((size_t)NS * 4), *nt_s = nn_malloc((size_t)NS * 4), *ga_s = nn_malloc((size_t)NS * ED * 4);
    float *scr_sc = nn_malloc(((size_t)2 * NG + 16) * 4), *scr_dn = nn_malloc(dn_dino_scratch(NS, K)), *outs = nn_malloc(64 * 4);
    if (!gl || !x || !scr_dn || (HZ && !hsl)) { fprintf(stderr, "dino-train: device allocation failed\n"); return 1; }
    dn_hs HS = {.u = u, .hs = hsl, .Z = HZ, .P = P, .D = ED, .CO = ED};
    if (HZ) { nn_set_logits_h16(1); unet_set_head_ext(u, 1); unet_set_head_cb(u, dn_head_cb, &HS); unet_build(u, (shape5){1, cfg.cin, P, P, P}, 1); }
    /* data ring */
    dn_ring *R = calloc(1, sizeof *R); R->D = Dd; R->seed = seed; R->nthreads = nthreads; pthread_mutex_init(&R->mu, nullptr); pthread_cond_init(&R->cv, nullptr);
    for (int i = 0; i < DN_SLOTS; i++) {
        dn_sample *S = &R->slot[i]; S->ca = malloc(p3); S->cb = malloc(p3);
        S->gidx = malloc((size_t)NG * 4); S->gcode = malloc((size_t)NG * 4); S->gxyz = malloc((size_t)NG * 12);   /* labelled + air */ S->sidx = malloc((size_t)NS * 4); S->ayx = malloc((size_t)P * 8);
    }
    R->nworkers = nwork > 8 ? 8 : nwork;
    for (int i = 0; i < R->nworkers; i++) pthread_create(&R->th[i], nullptr, ring_worker, R);
    signal(SIGTERM, dn_sig); signal(SIGINT, dn_sig);
    char logp[1200]; snprintf(logp, sizeof logp, "%s/train.log", outd); FILE *lf = fopen(logp, "a");
    fprintf(stderr, "dino-train: P %d, widths", P); for (int l = 0; l < cfg.nlev; l++) fprintf(stderr, " %d", cfg.widths[l]);
    fprintf(stderr, ", %zu params, D %d, GT %d + SSL %d points (weight %g), %d prototypes, %s\n", unet_nparams(u), ED, NG, NS, wssl, K, fp32 ? "bf16 (--mx 0)" : "fp4 / MX storage");
    if (cfg.attn_blocks) fprintf(stderr, "dino-train: attention stage: %d blocks at the bottleneck (%d tokens per window + %d registers), pe %s\n", cfg.attn_blocks,
                                 (P >> (cfg.nlev - 1)) * (P >> (cfg.nlev - 1)) * (P >> (cfg.nlev - 1)), cfg.attn_reg, cfg.attn_pe == 0 ? "peg" : cfg.attn_pe == 1 ? "rope" : "none");
    const shape5 xs = {1, cfg.cin, P, P, P};
    double t_last = dn_now(), sums[13] = {0}; int nsum = 0; uint64_t arng = seed * 1000003 + 17;
    const char *stopf_name = "STOP"; char stopf[1200]; snprintf(stopf, sizeof stopf, "%s/%s", outd, stopf_name);
    float *hout = malloc(64 * 4);
    double t_loop = dn_now(), t_meas0 = 0;
    for (int step = step0 + 1; step <= steps && !g_dn_stop; step++) {
        if (minutes > 0 && !step0) {   /* (a fresh run) */
            if (step == 21) { nn_sync(); t_meas0 = dn_now(); }
            if (step == 201) {
                nn_sync(); const double ms = (dn_now() - t_meas0) / 180, left = minutes * 60 - (dn_now() - t_loop);
                steps = 200 + (int)(left / ms); if (steps < 201) steps = 201;
                fprintf(stderr, "dino-train: %.0f ms/step over steps 21..200 -> %d steps for %.1f minutes\n", ms * 1e3, steps, minutes);
                if (lf) { fprintf(lf, "budget: %.1f ms/step -> %d steps (%.1f min)\n", ms * 1e3, steps, minutes); fflush(lf); }
            }
        }
        dn_sample *S = ring_get(R); if (!S) { fprintf(stderr, "dino-train: data loader failed\n"); return 1; }
        const dn_src *src = &Dd->src[S->si];
        nn_h2d(cta, S->ca, p3); nn_h2d(ctb, S->cb, p3);
        if (S->ngt) { nn_h2d(d_gidx, S->gidx, (size_t)S->ngt * 4); nn_h2d(d_gcode, S->gcode, (size_t)S->ngt * 4); nn_h2d(d_gxyz, S->gxyz, (size_t)S->ngt * 12); }
        if (S->nssl) nn_h2d(d_sidx, S->sidx, (size_t)S->nssl * 4);
        const int ngt = S->ngt, nssl = S->nssl;
        if (axis_in) nn_h2d(d_ayx, S->ayx, (size_t)P * 8);
        ring_put(R, S);   /* (its host buffers are copied) */
        const uint8_t *cs = step & 1 ? ctb : cta, *ctt = step & 1 ? cta : ctb;   /* student / teacher views alternate */
        const float ga1 = 1.f + gj * (2.f * (float)rng_u(&arng) - 1.f), gb1 = gj * (2.f * (float)rng_u(&arng) - 1.f);
        const float ga2 = 1.f + gj * (2.f * (float)rng_u(&arng) - 1.f), gb2 = gj * (2.f * (float)rng_u(&arng) - 1.f);
        const float sg1 = noise * (float)rng_u(&arng), sg2 = noise * (float)rng_u(&arng);
        const void *lg = nullptr; int h16 = 0;
        if (use_teacher) {
        /* teacher (EMA weights), forward only */
        dn_input(ctt, x, P, S->g, src->center, src->scale, ga2, gb2, sg2, (uint32_t)(step * 2 + 1), 0, 0.f, 0, 0.f);
        if (axis_in) dn_input_axis(x + p3, P, S->g, d_ayx, ars);
        unet_use_ema(u, 1);
        lg = unet_forward(u, x, xs, 0); h16 = unet_logits_h16(u);
        if (HZ) dn_gather_slabs(&HS, d_gidx, ngt, et_g, nt_g, d_sidx, nssl, et_s, nt_s);
        else {
        if (ngt) dn_gather(lg, h16, p3, ED, d_gidx, ngt, et_g, nt_g);
        if (nssl) dn_gather(lg, h16, p3, ED, d_sidx, nssl, et_s, nt_s);
        }
        unet_use_ema(u, 0);
        }
        /* student */
        dn_input(cs, x, P, S->g, src->center, src->scale, ga1, gb1, sg1, (uint32_t)(step * 2), mcube, mfrac, (uint32_t)step, 0.f);
        if (axis_in) dn_input_axis(x + p3, P, S->g, d_ayx, ars);
        lg = unet_forward(u, x, xs, 1); h16 = unet_logits_h16(u);
        if (HZ) dn_gather_slabs(&HS, d_gidx, ngt, es_g, ns_g, d_sidx, nssl, es_s, ns_s);
        else {
        if (ngt) dn_gather(lg, h16, p3, ED, d_gidx, ngt, es_g, ns_g);
        if (nssl) dn_gather(lg, h16, p3, ED, d_sidx, nssl, es_s, ns_s);
        nn_zero(gl, p3 * ED * 4);
        }
        HS.ngt = 0; HS.nssl = 0;
        nn_zero(outs, 64 * 4);
        if (ngt) {
            dn_supcon(es_g, d_gcode, d_gxyz, ngt, et_g, d_gcode, d_gxyz, use_teacher ? ngt : 0, ED, tau, hw, hr, ga_g, outs, scr_sc);
            if (wgt != 1.f) dn_axpby(ga_g, ga_g, (size_t)ngt * ED, wgt, 0.f);
            if (wvar > 0) dn_varreg(es_g, ngt, ED, vgam, wvar, ga_g, outs + 12);   /* anti-collapse: per-dimension spread */
            if (HZ) { HS.e1 = es_g; HS.g1 = ga_g; HS.n1 = ns_g; HS.i1 = d_gidx; HS.ngt = ngt; }
            else dn_scatter_grad(es_g, ga_g, ns_g, d_gidx, ngt, ED, p3, 1.f, gl);
            dn_emb_std(es_g, ngt, ED, outs + 8);
        }
        if (nssl && wssl > 0 && use_teacher) {
            dn_dino(es_s, et_s, nssl, W, Wt, cen, K, ED, ts, tt, ga_s, gW, tmean, outs + 4, scr_dn);
            if (HZ) { HS.e2 = es_s; HS.g2 = ga_s; HS.n2 = ns_s; HS.i2 = d_sidx; HS.nssl = nssl; HS.w2 = wssl; }
            else dn_scatter_grad(es_s, ga_s, ns_s, d_sidx, nssl, ED, p3, wssl, gl);
            dn_axpby(gW, gW, kd, 0.f, wssl);
        }
        if (nssl) dn_emb_std(es_s, nssl, ED, outs + 9);
        unet_zero_grad(u);
        unet_backward(u, gl);
        const double gn = unet_grad_norm(u);
        if (clip > 0) unet_clip_grad(u, clip);
        const float sched = step <= warm ? (float)step / warm : 0.5f * (1.f + cosf((float)M_PI * (float)(step - warm) / (float)(steps - warm > 1 ? steps - warm : 1)));
        unet_muon(u, mlr0 * sched, 0.95f, lr0 * sched, 0.9f, 0.999f, 1e-8f, wd, step);
        unet_ema(u, step < 100 ? 0.9f : ema);
        if (nssl && wssl > 0 && use_teacher) {
            dn_adam(W, gW, Wm, Wv, kd, lr0 * sched, 0.9f, 0.999f, 1e-8f, 0.f, step);
            dn_ema(Wt, W, kd, step < 100 ? 0.9f : ema);
            dn_axpby(cen, tmean, (size_t)K, cm, 1.f - cm);
        }
        nn_d2h(hout, outs, 64 * 4);
        /* 0 supcon loss, 1 pos cos, 2 neg cos, 3 anchors, 4 dino loss, 5 batch-mean teacher entropy, 6 mean teacher entropy, 8 std gt, 9 std ssl */
        const double v[13] = {hout[0], hout[1], hout[2], hout[3], hout[4], hout[5], hout[6], hout[8], hout[9], gn, (double)ngt, (double)nssl, hout[12]};
        for (int k = 0; k < 13; k++) sums[k] += v[k];
        nsum++;
        if (!isfinite(hout[0]) || !isfinite(hout[4]) || !isfinite(gn)) { fprintf(stderr, "dino-train: non-finite loss / gradient at step %d\n", step); return 1; }
        if (step % log_every == 0) {
            const double now = dn_now(), dt = (now - t_last) / nsum;
            char line[512];
            snprintf(line, sizeof line, "step %d gt %.4f var %.4f pos %.3f neg %.3f anchors %.0f | ssl %.4f H(mean) %.2f H %.2f | std gt %.4f ssl %.4f | gnorm %.3f | lr %.2e | %.0f ms/step %.1f Mvox/s",
                     step, sums[0] / nsum, sums[12] / nsum, sums[1] / nsum, sums[2] / nsum, sums[3] / nsum, sums[4] / nsum, sums[5] / nsum, sums[6] / nsum, sums[7] / nsum, sums[8] / nsum,
                     sums[9] / nsum, lr0 * sched, dt * 1e3, (double)p3 / dt / 1e6);
            fprintf(stderr, "%s\n", line); if (lf) { fprintf(lf, "%s\n", line); fflush(lf); }
            memset(sums, 0, sizeof sums); nsum = 0; t_last = dn_now();
            if (access(stopf, F_OK) == 0) { fprintf(stderr, "dino-train: STOP file\n"); g_dn_stop = 1; }
        }
        if (step % save_every == 0 || step == steps || g_dn_stop) {
            char cp[1200], ex[256]; snprintf(cp, sizeof cp, "%s/step%06d.ckpt", outd, step);
            snprintf(ex, sizeof ex, "\"task\":\"dino\",\"embed_dim\":%d,\"protos\":%d", ED, K);
            if (unet_save(u, cp, step, ex)) fprintf(stderr, "dino-train: cannot save %s\n", cp);
            char pp[1300]; snprintf(pp, sizeof pp, "%s.protos", cp); float *h = malloc(kd * 4); nn_d2h(h, W, kd * 4);
            FILE *pf = fopen(pp, "wb"); if (pf) { fwrite(h, 4, kd, pf); fclose(pf); } free(h);
            char lp[1200]; snprintf(lp, sizeof lp, "%s/last.ckpt", outd); unlink(lp); if (symlink(cp + strlen(outd) + 1, lp)) {}
        }
        if (opt(argc, argv, "--eval", nullptr) && (step % eval_every == 0 || step == steps)) {
            if (dn_run_eval(u, argc, argv, step, P, ED, nthreads, axis_in, ars, outd, lf)) { t_last = dn_now(); memset(sums, 0, sizeof sums); nsum = 0; }
        }
    }
    pthread_mutex_lock(&R->mu); R->stop = 1; pthread_cond_broadcast(&R->cv); pthread_mutex_unlock(&R->mu);
    for (int i = 0; i < R->nworkers; i++) pthread_join(R->th[i], nullptr);
    if (lf) fclose(lf);
    return 0;
}

/* ufsm dino-embed: a dino-train checkpoint over a box -> the dinovol-format group (dense, unit int8, step 1) */
int cmd_dino_embed(int argc, char **argv) {
    long long b[6]; float c = 0, s = 0;
    const int norm_auto = !strcmp(opt(argc, argv, "--norm", ""), "auto");
    if (argc < 6 || sscanf(opt(argc, argv, "--box", ""), "%lld,%lld,%lld,%lld,%lld,%lld", &b[0], &b[1], &b[2], &b[3], &b[4], &b[5]) != 6 || (!norm_auto && sscanf(opt(argc, argv, "--norm", ""), "%f,%f", &c, &s) != 2)) {
        fprintf(stderr, "usage: ufsm dino-embed <ckpt> <ct-root> <ct-key> <out-group> --box z,y,x,nz,ny,nx --norm center,scale|auto [--P 256] [--halo 32] [--ema 1] [--cache DIR] [--mx 0]\n"
                        "         [--head-slab 1 [--mem lean]] (large windows: the head per z-slab, no whole-window logits)\n"
                        "  --norm auto: dino-train's statistics of the box (median, IQR / 1.349 of nonzero voxels in 8 random 128^3 blocks)\n"); return 2; }
    const int P = atoi(opt(argc, argv, "--P", "256")), fp32 = !atoi(opt(argc, argv, "--mx", "1"));
    dn_precision(fp32);
    if (nn_init(atoi(opt(argc, argv, "--gpu", "0")))) return 1;
    dn_precision_after_init(fp32);
    unet_cfg cfg; int step = 0;
    if (unet_peek(argv[2], &cfg, &step)) { fprintf(stderr, "dino-embed: cannot read %s\n", argv[2]); return 1; }
    unet *u = unet_create(&cfg); if (unet_load(u, argv[2]) < 0) return 1;
    if (!fp32 && cfg.cout <= 16) unet_set_fp16_logits(u, 1);
    if (atoi(opt(argc, argv, "--head-slab", "0"))) {   /* large windows: a training build with head slabs (no whole-window logits), optionally lean */
        if (fp32) { fprintf(stderr, "dino-embed: --head-slab needs MX storage\n"); return 2; }
        if (!strcmp(opt(argc, argv, "--mem", "default"), "lean")) { unet_set_chunk_up(2); unet_set_recompute(1); unet_set_grad_mx8(1); unet_set_lean(2); unet_set_share_enc_a1(1); }
        nn_set_logits_h16(1); unet_set_head_ext(u, 1); unet_build(u, (shape5){1, cfg.cin, P, P, P}, 1);
    }
    store *st = store_open(argv[3]); z3 *ct = st ? z3_open(st, argv[4], opt(argc, argv, "--cache", nullptr)) : nullptr;
    if (!ct) { fprintf(stderr, "dino-embed: %s\n", z3_error()); return 1; }
    const int64_t box[6] = {b[0], b[1], b[2], b[3], b[4], b[5]};
    if (norm_auto) { box_norm(ct, (const int64_t (*)[6])&box, 1, 12345, 8, &c, &s); fprintf(stderr, "dino-embed: box statistics: norm %.1f,%.2f\n", c, s); }
    const double t0 = dn_now();
    float *axz = nullptr;
    if (cfg.cin == 2) { axz = load_axis(opt(argc, argv, "--axis", ""), atof(opt(argc, argv, "--axis-scale", "1")), z3_meta_of(ct)->shape[0]); if (!axz) { fprintf(stderr, "dino-embed: this net takes the axis-distance channel: --axis FILE [--axis-scale S]\n"); return 2; } }
    int8_t *e = dn_embed_box(u, ct, box, P, atoi(opt(argc, argv, "--halo", "32")), c, s, 8, atoi(opt(argc, argv, "--ema", "1")), axz, (float)atof(opt(argc, argv, "--axis-rs", "0.001")));
    if (!e) return 1;
    const double t1 = dn_now();
    if (dv_write_dense_box(argv[5], z3_meta_of(ct)->shape, box, dn_embed_dim(&cfg), e)) return 1;
    fprintf(stderr, "dino-embed: %lld x %lld x %lld voxels in %.1f s (%.1f Mvox/s network + read), written in %.1f s\n", b[3], b[4], b[5], t1 - t0, (double)b[3] * b[4] * b[5] / (t1 - t0) / 1e6, dn_now() - t1);
    free(e);
    return 0;
}

/* ufsm dino-boxes <codes-root> <codes-key> [--block 512] [--min 200000] [--exclude z,y,x,nz,ny,nx]... : block-aligned boxes holding at least
   --min labelled voxels (codes 1..254), as a JSON list for dino-train sources; blocks touching an --exclude box are left out */
int cmd_dino_boxes(int argc, char **argv) {
    if (opt(argc, argv, "--ct-coarse", nullptr)) {   /* papyrus-rich blocks of a CT (no labels): a coarse level read whole */
        store *cs = store_open(argv[2]); z3 *c = cs ? z3_open(cs, opt(argc, argv, "--ct-coarse", ""), opt(argc, argv, "--cache", nullptr)) : nullptr;
        if (!c) { fprintf(stderr, "dino-boxes: %s\n", z3_error()); return 1; }
        const int f = atoi(opt(argc, argv, "--factor", "8")), B = atoi(opt(argc, argv, "--block", "512")), top = atoi(opt(argc, argv, "--top", "8")), b = B / f;
        const z3_meta *m = z3_meta_of(c); const size_t nv = (size_t)m->shape[0] * m->shape[1] * m->shape[2];
        uint8_t *v = malloc(nv); const int64_t o[3] = {0, 0, 0};
        if (!v || z3_read(c, o, m->shape, v, 8)) { fprintf(stderr, "dino-boxes: coarse read: %s\n", z3_error()); return 1; }
        int64_t h[256] = {0}, tot = 0; for (size_t i = 0; i < nv; i++) h[v[i]]++;
        h[0] = 0; for (int i = 1; i < 256; i++) tot += h[i];
        int med = 1; { int64_t a = 0; for (int i = 1; i < 256; i++) { a += h[i]; if (a >= tot / 2) { med = i; break; } } }
        const int thr = atoi(opt(argc, argv, "--thresh", "0")) > 0 ? atoi(opt(argc, argv, "--thresh", "0")) : med;
        typedef struct { double fr; int64_t o[3]; } cand; cand *cd = malloc(sizeof *cd * 100000); int nc = 0;
        uint64_t rs = 99;
        for (int64_t z = 0; z + b <= m->shape[0]; z += b / 2) for (int64_t y = 0; y + b <= m->shape[1]; y += b / 2) for (int64_t x = 0; x + b <= m->shape[2]; x += b / 2) {
            size_t cnt = 0, s = 0;
            for (int64_t a = 0; a < b; a += 2) for (int64_t e = 0; e < b; e += 2) for (int64_t g = 0; g < b; g += 2) { cnt += v[((size_t)(z + a) * m->shape[1] + y + e) * m->shape[2] + x + g] > thr; s++; }
            if (nc < 100000) { cd[nc].fr = (double)cnt / s + 1e-9 * (double)(rng_next(&rs) % 1000); cd[nc].o[0] = z * f; cd[nc].o[1] = y * f; cd[nc].o[2] = x * f; nc++; }
        }
        int first = 1; printf("[");
        for (int k = 0; k < top; k++) {   /* greedy: the richest block not overlapping a chosen one */
            int bi = -1; for (int i = 0; i < nc; i++) if (cd[i].fr >= 0 && (bi < 0 || cd[i].fr > cd[bi].fr)) bi = i;
            if (bi < 0 || cd[bi].fr < 0.2) break;
            printf("%s\n  [%lld,%lld,%lld,%d,%d,%d]", first ? "" : ",", (long long)cd[bi].o[0], (long long)cd[bi].o[1], (long long)cd[bi].o[2], B, B, B); first = 0;
            fprintf(stderr, "block %lld,%lld,%lld: papyrus fraction %.2f\n", (long long)cd[bi].o[0], (long long)cd[bi].o[1], (long long)cd[bi].o[2], cd[bi].fr);
            const int64_t *q = cd[bi].o;
            for (int i = 0; i < nc; i++) { int ov = 1; for (int d = 0; d < 3; d++) ov &= llabs(cd[i].o[d] - q[d]) < B; if (ov) cd[i].fr = -1; }
        }
        printf("\n]\n"); free(v); free(cd); z3_close(c); store_close(cs);
        return 0;
    }
    if (argc < 4) { fprintf(stderr, "usage: ufsm dino-boxes <codes-root> <codes-key> [--block 512] [--min 200000] [--exclude z,y,x,nz,ny,nx]...\n"); return 2; }
    store *st = store_open(argv[2]); z3 *z = st ? z3_open(st, argv[3], nullptr) : nullptr;
    if (!z) { fprintf(stderr, "dino-boxes: %s\n", z3_error()); return 1; }
    const int B = atoi(opt(argc, argv, "--block", "512")); const long mn = atol(opt(argc, argv, "--min", "200000"));
    int64_t ex[32][6]; int nex = 0;
    for (int i = 2; i + 1 < argc && nex < 32; i++) if (!strcmp(argv[i], "--exclude")) { long long v[6]; if (sscanf(argv[i + 1], "%lld,%lld,%lld,%lld,%lld,%lld", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5]) == 6) { for (int d = 0; d < 6; d++) ex[nex][d] = v[d]; nex++; } }
    const z3_meta *m = z3_meta_of(z); int64_t grid[3]; uint8_t *bits = nullptr;
    if (z3_chunk_map(z, grid, &bits) < 0) { fprintf(stderr, "dino-boxes: chunk map: %s\n", z3_error()); return 1; }
    const int cpb = B / m->chunk[0];   /* chunks per block edge */
    uint8_t *buf = malloc((size_t)m->chunk[0] * m->chunk[1] * m->chunk[2]);
    int first = 1; printf("[");
    for (int64_t bz = 0; bz * B < m->shape[0]; bz++) for (int64_t by = 0; by * B < m->shape[1]; by++) for (int64_t bx = 0; bx * B < m->shape[2]; bx++) {
        const int64_t o[3] = {bz * B, by * B, bx * B};
        int bad = 0; for (int e = 0; e < nex && !bad; e++) bad = box_hit(ex[e], o, B);
        if (bad) continue;
        long cnt = 0;
        for (int a = 0; a < cpb; a++) for (int b = 0; b < cpb; b++) for (int c = 0; c < cpb; c++) {
            const int64_t cz = bz * cpb + a, cy = by * cpb + b, cx = bx * cpb + c;
            if (cz >= grid[0] || cy >= grid[1] || cx >= grid[2] || !bits[(cz * grid[1] + cy) * grid[2] + cx]) continue;
            const int64_t oo[3] = {cz * m->chunk[0], cy * m->chunk[1], cx * m->chunk[2]}, nn[3] = {m->chunk[0], m->chunk[1], m->chunk[2]};
            if (z3_read(z, oo, nn, buf, 4)) continue;
            for (size_t i = 0; i < (size_t)nn[0] * nn[1] * nn[2]; i++) cnt += buf[i] && buf[i] != 255;
        }
        if (cnt < mn) continue;
        if (o[0] + B > m->shape[0] || o[1] + B > m->shape[1] || o[2] + B > m->shape[2]) continue;
        printf("%s\n  [%lld,%lld,%lld,%d,%d,%d]", first ? "" : ",", (long long)o[0], (long long)o[1], (long long)o[2], B, B, B);
        fprintf(stderr, "block %lld,%lld,%lld: %ld labelled voxels\n", (long long)o[0], (long long)o[1], (long long)o[2], cnt);
        first = 0;
    }
    printf("\n]\n");
    free(buf); free(bits); z3_close(z); store_close(st);
    return 0;
}
