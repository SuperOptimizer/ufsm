/* ufsm dino: our own DINO-style embedding training from scratch (ufsm dino-train), its inference (ufsm dino-embed) and the
   view-pair data tool (ufsm halves). See DESIGN.md "Own dense embedding net (dino-train)". */
#include "dino.h"
#include "dinor.h"
#include "sslstat.h"
#include "band.h"
#include "attn.h"
#include "dinovol.h"
#include "reg.h"
#include "xnet.h"
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

enum { DN_MAXSRC = 128, DN_MAXBOX = 256, DN_SLOTS = 64, DN_MAXW = 16 };   /* (slots / workers: at most; --slots / --workers) */
typedef struct {
    char name[64]; z3 *ca, *cb, *codes, *ink, *phase; int halves; int cscale, pscale; float um;   /* phase: the stored winding phase (PHASE_* bytes; sources "phase": {"root", "key", "scale"}) */ float *axz;   /* scroll axis (y, x) per CT plane (sources "axis" + "axis_scale"), or nullptr */   /* CT voxels per code voxel (codes on a coarser grid; 1: the same grid) */
    int64_t org[3];                              /* added to window corners (ct array coordinates) to address the codes (global grid) */
    int nbox; int64_t box[DN_MAXBOX][6]; int nho; int64_t ho[DN_MAXBOX][6];
    double weight, bvol; float center, scale, pap; int64_t shape[3];
    int ct_only;                                 /* unlabelled (U0): SSL terms only (sampled by the ct_only share) */
    int nx; const rmap *xm[4]; rvol *xfv[4]; int xlc[4], xlf[4];   /* "xscan": registered partner scans (this CT = the map's coarse side at level xlc) */
    int32_t *mcell; int64_t nmcell; int mcs;     /* "mask": {"map": RAW, "min": m} (ufsm scanprep): the cells with code >= m (window centres), cell edge */
} dn_src;
typedef struct { int P, ngt, nssl, nair, margin; int nsrc; dn_src src[DN_MAXSRC]; double wsum; int ident; float xprob; int nink; double wsum_l, wsum_u, share_u; char excl[8][64]; int nexcl;
    /* recipe views drawn by the loader workers: globals vP^3 (zoom zlo..zhi), vnloc locals vPL^3 (lzlo..lzhi); uw: per global view the stored phase
       resampled into the view + its winding mod 4 (band_wind4_tiled); ikp: the ink-splat probability resampled into the view */
    int vP, vPL, vnloc, uw, ikp; float zlo, zhi, lzlo, lzhi, vovl; } dn_data;   /* vovl: the globals' minimum overlap per axis (fraction of the smaller; 0: 1/4) */   /* share_u: the unlabelled sources' share of the draws (< 0: by weight over all; 0: labelled only) */   /* nink: ink points per sample (half ink, half no-ink) */   /* ident: no window geometry (the recipe draws its own views) */
typedef struct {   /* one sample: two CT views, the geometry, the points (output-space voxel indices) */
    uint8_t *ca, *cb; int si; int64_t corner[3]; dn_geom g; int halves; int hasax; float *ayx; int xs; float xncc;   /* xs: cb = a registered partner scan resampled onto this window */   /* the axis (y, x) per window plane (window coords; --axis-input) */
    int ngt, nssl; int *gidx, *gcode, *gxyz, *sidx;
    int nip, nin; int *ixyz;   /* ink points (window coords, identity geometry): nip ink first, then nin no-ink (sources "ink", the recipe) */
    dr_xf xf[18]; int nxf;     /* the recipe's views (worker-drawn) */
    uint8_t *pv[2], *w4[2], *wc[2], *iv[2], *im[2], *ikw, *phw; int uwv[2], ikv;   /* per global view: phase, winding mod 4, component, ink target / mask (recipe) */
    int ready;
} dn_sample;
typedef struct {
    dn_data *D; dn_sample slot[DN_SLOTS]; int nslots; pthread_mutex_t mu; pthread_cond_t cv; long produced, consumed; int stop; uint64_t seed; int nthreads;
    pthread_t th[DN_MAXW]; int nworkers; long next_id;
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
            if (v == 1) { const int ov = (int)((t->m < S->xf[0].m ? t->m : S->xf[0].m) * (D->vovl > 0 ? D->vovl : 0.25f)); const int a = S->xf[0].off[d] - t->m + ov, b = S->xf[0].off[d] + S->xf[0].m - ov; if (a > lo2) lo2 = a; if (b < hi2) hi2 = b; if (hi2 < lo2) hi2 = lo2; }
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
/* UFSM_LOADER_PROF=1: the loader's per-sample seconds by part (CT read, codes / GT points, ink, the recipe's views + dense targets), all
   workers summed, printed every 50 samples */
static pthread_mutex_t g_lp_mu = PTHREAD_MUTEX_INITIALIZER; static double g_lp[5]; static long g_lpn;
static void lp_add(const double t[5]) {
    pthread_mutex_lock(&g_lp_mu); for (int i = 0; i < 5; i++) g_lp[i] += t[i];
    if (++g_lpn % 50 == 0) { fprintf(stderr, "loader (s/sample, %ld samples): ct %.2f codes %.2f ink %.2f views+dense %.2f total %.2f\n", g_lpn, g_lp[0] / 50, g_lp[1] / 50, g_lp[2] / 50, g_lp[3] / 50, g_lp[4] / 50); memset(g_lp, 0, sizeof g_lp); }
    pthread_mutex_unlock(&g_lp_mu);
}
static int sample_fill(dn_data *D, dn_sample *S, uint64_t *rs, int nthreads, uint8_t *codes_buf) {
    static int lprof = -1; if (lprof < 0) lprof = getenv("UFSM_LOADER_PROF") && ufsm_env_on("UFSM_LOADER_PROF");
    double lt[5] = {0}, lt0 = lprof ? dn_now() : 0, lta = lt0;
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
        if (lprof) { const double t_ = dn_now(); lt[0] += t_ - lta; lta = t_; }
        else memcpy(S->cb, S->ca, p3);
        S->xs = 0;
        if (s->nx && D->xprob > 0 && rng_u(rs) < D->xprob) {   /* cross-scan: a registered partner scan resampled onto this window, intensities matched */
            const int k = (int)(rng_next(rs) % (uint64_t)s->nx), lc = s->xlc[k], lf = s->xlf[k];
            const double org[3] = {(double)o[2], (double)o[1], (double)o[0]}; const int nn3[3] = {P, P, P}, cnt3[3] = {P, P, P}; (void)cnt3;
            rblk fb; int err = 0; uint8_t *fbuf = reg_read_fine(s->xfv[k], lf, s->xm[k], lc, org, nn3, 2.0, &fb, &err);
            if (fbuf) {
                float *t = malloc(p3 * 4); const double sh[3] = {0, 0, 0};
                reg_resample(&fb, s->xm[k], lc, lf, org, nn3, sh, 1, t); free(fbuf);
                double sa = 0, qa = 0, sb = 0, qb = 0; size_t nb = 0;
                for (size_t i = 0; i < p3; i += 3) if (S->ca[i] && t[i] > 0) { sa += S->ca[i]; qa += (double)S->ca[i] * S->ca[i]; sb += t[i]; qb += (double)t[i] * t[i]; nb++; }
                if (nb * 3 > p3 / 4) {   /* >= 25% of the window covered by both scans */
                    const double ma = sa / nb, mb = sb / nb, da = sqrt(fmax(qa / nb - ma * ma, 1e-6)), db = sqrt(fmax(qb / nb - mb * mb, 1e-6));
                    for (size_t i = 0; i < p3; i++) { const double v = t[i] > 0 ? ma + (t[i] - mb) * da / db : 0; S->cb[i] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : lrint(v)); }
                    S->xs = 1;
                    double sab = 0; for (size_t i = 0; i < p3; i += 3) if (S->ca[i] && t[i] > 0) sab += ((double)S->ca[i] - ma) * (t[i] - mb);
                    S->xncc = (float)(sab / nb / (da * db));   /* (diagnostic: the two scans' correlation over the window) */
                } else memcpy(S->cb, S->ca, p3);
                free(t);
            }
        }
        /* enough papyrus? */
        size_t pap = 0; for (size_t i = 0; i < p3; i += 7) pap += S->ca[i] > s->pap;
        if (pap * 7 < p3 / 50) continue;   /* < 2% papyrus: mostly air */
        S->si = si; memcpy(S->corner, o, sizeof o); S->halves = s->halves;
        S->hasax = s->axz != nullptr;
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
        if (lprof) { const double t_ = dn_now(); lt[1] += t_ - lta; lta = t_; }
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
        if (lprof) { const double t_ = dn_now(); lt[2] += t_ - lta; lta = t_; }
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
        if (lprof) { const double t_ = dn_now(); lt[3] += t_ - lta; lt[4] = t_ - lt0; lp_add(lt); }
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
        while (!R->stop) { for (int i = 0; i < R->nslots; i++) if (R->slot[i].ready == 0) { k = i; break; } if (k >= 0) break; pthread_cond_wait(&R->cv, &R->mu); }
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
        for (int i = 0; i < R->nslots; i++) {
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
        {   /* "xscan": [{"map": REG.json, "lc": this CT's level in the map's coarse scan}] -- partner scans for the SSL cross-scan views */
            const json *xj = json_get(s, "xscan");
            if (xj && xj->type == J_ARR) for (size_t k = 0; k < xj->n && S->nx < 4; k++) {
                const json *e = json_at(xj, k); rvol *fv = nullptr; double cum = 0, fum = 0;
                const rmap *m = reg_map_open(json_str(json_get(e, "map"), ""), cache, &fv, &cum, &fum);
                if (!m) { fprintf(stderr, "dino-train: %s: cannot open xscan map %s\n", S->name, json_str(json_get(e, "map"), "")); return -1; }
                const int lc = (int)json_num(json_get(e, "lc"), 0), lf = rvol_level_for(fv, cum * (double)(1 << lc));
                S->xm[S->nx] = m; S->xfv[S->nx] = fv; S->xlc[S->nx] = lc; S->xlf[S->nx] = lf; S->nx++;
                fprintf(stderr, "dino-train: %s: xscan %s (coarse level %d = %.2f um <- fine level %d = %.2f um)\n", S->name, fv->name, lc, cum * (1 << lc), lf, fum * (1 << lf));
            }
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
    if (oc.cin < unet_cfg_of(u)->cin) {   /* more inputs (--axis-input): a warm start, the stem's new input channels zero (the donor's function), head rows kept */
        int rows[256]; const int co = unet_cfg_of(u)->cout; for (int i = 0; i < co && i < 256; i++) rows[i] = i < oc.cout ? i : -1;
        if (unet_init_from_map(u, path, rows) < 0) return -1;
        fprintf(stderr, "dino-train: %s: warm start %d -> %d inputs, %d -> %d outputs (stem grown, optimiser state fresh)\n", path, oc.cin, unet_cfg_of(u)->cin, oc.cout, co);
        return st;
    }
    if (oc.cout < unet_cfg_of(u)->cout) {
        const int r = unet_load_grow(u, path, oc.cout, 0.f);
        if (r >= 0) fprintf(stderr, "dino-train: %s: head grown %d -> %d outputs (trunk kept, new rows zero)\n", path, oc.cout, unet_cfg_of(u)->cout);
        return r;
    }
    return unet_load(u, path);
}
/* embedding channels of a dino net: the first min(cout, 32) (a recipe net with dense heads: 32 + 12 padded to 48) */
static int dn_embed_dim(const unet_cfg *c) { return c->cout == 48 ? 32 : c->cout; }   /* (48: the 32-d embedding + the recipe's dense heads; otherwise all head channels, e.g. --dim 64 distilled) */

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
        if (cin == 5) {   /* (a recipe net with the axis input: train's radial channels from the axis when given, else valid 0) */
            static dr_xf *xid; if (!xid) { xid = nn_malloc(sizeof(dr_xf)); }
            const dr_xf t0 = {{0, 0, 0}, {0, 1, 2}, {0, 0, 0}, P, P, 0}; nn_h2d(xid, &t0, sizeof t0);
            if (axz) { const int64_t nzv = z3_meta_of(ct)->shape[0];
                for (int z = 0; z < P; z++) { int64_t gz = o[0] + z; gz = gz < 0 ? 0 : gz >= nzv ? nzv - 1 : gz; ayh[2 * z] = axz[2 * gz] - (float)o[1]; ayh[2 * z + 1] = axz[2 * gz + 1] - (float)o[2]; }
                nn_h2d(ayd, ayh, (size_t)P * 8); }
            dr_axis_view(xid, 0, P, P, ayd, axz != nullptr, x + p3);
        }
        if (cin == 2 && !axz) nn_zero(x + p3, p3 * 4);   /* (a --mask-input net: nothing masked) */
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
/* ---- the multi-sample SSL batch (--ssl-batch N): upstream DINO on a real batch beside the batch-1 supervised window ----
   Batch 1 has 2 CLS rows: Sinkhorn / centring cannot sharpen them (the DINO term stays at ln K). Each step (every --ssl-every) N samples of
   --ssl-source-P^3 windows from the weighted sources (U0 + labelled; --ssl-unlabelled-share), each with 2 globals of --ssl-P^3 (upstream
   scale 0.18-0.45 of the window: zoom 0.84-1.15) and --ssl-locals locals of --ssl-local-P^3 (0.023-0.057: 0.86-1.16), its own intensity
   augmentation per view. The EMA teacher (shared weights through unet_use_params) on the 2N globals; the CLS (--cls reg: a learned register
   token, else the token mean), layer-normed; the DINO head (AdamW, upstream lr, g fixed) -> --ssl-protos prototypes; Sinkhorn over the
   batch's 2N teacher rows; the student per view (forward, loss over the batch's targets, backward: per-view activations), global (i, v)
   vs teacher (i, 1 - v), each local vs both of its sample; KoLeo per global view chunk over the batch (the other samples' CLS of that chunk:
   the student's when already computed this step, the teacher's otherwise -- never the same sample's other view). The gradient joins u's. */
typedef struct {
    unet *u, *ug, *ul; int N, PG, PL, NL, Ps, C, CC, K, CIN, cls_reg, cls_norm, freeze, skit, nsum, ure; float ts, wdino, wk, hlr, plr, wd, adrop, noise;
    dn_data *Ds; dn_ring *R; uint8_t *src; float *xg, *xl, *glzg, *glzl, *tgrad; size_t tgn; dr_xf *xfd, *xfh; dr_int *itd, *ith;
    dn_head H; float *tl, *tq, *sl, *dsl, *clsT, *clsS, *dcls, *rsS, *rsT, *bank, *bex, *outs, *hout, *d_ayx; void *sks, *ces; int *pS, *pT; float *pW;
    uint64_t ar; uint8_t *use; double sums[8]; shape5 xsg, xsl;
    /* iBOT (--ssl-ibot W): masked 8^3 cells of each global (prob 0.5, ratio 0.1-0.5, <= IM per view), the head features (channels
       --ssl-ibot-ch .. +32 of the output) at the cell centres vs the teacher's at the same physical points of the other global; iBOT head ->
       KI prototypes, Sinkhorn over all the batch's masked rows */
    float cm, icm, *cen, *icen; double dg[8];
    int easy, ccen, cstd; float *cmean, *hmean, *cistd, *histd, *cpre;
    float vv, vc, *vx, *vg, *vgd; double *vcov;
    float simg, sima, *zT, *zbank, *dzh, *zcur; double *zA;
    float zv, zcv, *zvb, *zvh, *zvd; double *zcov; int vramp;   /* vramp: the VICReg weights ramp in linearly over --ssl-vic-ramp steps */   /* --ssl-zvic-var / --ssl-zvic-cov: VICReg on the head's unit bottleneck zn over the batch (hinge std >= 1 / sqrt(B)) */   /* --ssl-simdino G: SimDINO (1/2 |z_s - z_t|^2 on the unit bottleneck - G R_eps(student global z)), --ssl-sim-eps2 */   /* --ssl-vic-var / --ssl-vic-cov: VICReg variance hinge (std >= 1) + off-diagonal covariance on the head-input CLS over the batch */ double st[12];   /* cstd: --ssl-cls-std (per-dimension batch standardisation of the CLS) */   /* --ssl-easy-steps S: shared symmetry / no inversion / mild intensity until S; --ssl-cls-center: the batch mean CLS removed; st: per-step stats */   /* dg: the teacher-side diagnostics of the last 50th step (dn_ssl_diag) */   /* --ssl-center / --ssl-ibot-center M: DINOv1 centring (EMA of the batch's teacher logits) instead of Sinkhorn */
    double zsmin, zsmed;   /* the last step's z per-dimension std: min / median (teacher rows) */
    int exact, *hasaxv, *amd, *amh; double *vmu, *vsd, *zmu, *zsd; float *kg, *kgd;   /* kg: the exact bank's full-batch KoLeo gradient per row (chunk-major) */   /* --ssl-exact-bank: the batch statistics from a no-grad student pass (GradCache); per-step chunk stats */
    float wib; int KI, IM, ich, TPS, GT_, cnt[2048], off[2048]; dn_head I; float *ieT, *inT, *ieS, *inS, *idE, *il, *iq, *isl, *idsl, *iw, *glzib; void *isk, *ice; int *d_hS, *d_hT, *iid, *hS, *hT; uint8_t *cellm, *d_mask; uint8_t *usei;
    /* diag-dv (--ssl-dv-teacher PT): the iBOT targets from the frozen upstream dinovol teacher (its ibot_head over its patch tokens of the
       un-augmented teacher global, at the cell of each mapped point) until --ssl-dv-until (0: always), then from our EMA teacher; the student
       iBOT head predicts dinovol's 131072 prototypes (its last layer initialised from dinovol's and kept frozen) */
    dv_net *dv; dr_head DVH; float *dvP, *dvx, *dvxd, *dvf; void *dvwb; dr_hws dvw; uint8_t *dvu; dr_int *dvid; double *dvm, *dvs, *dvh; int dvuntil, *smsi, lastm; double xnccs; long xnccn; int64_t *dvhist; long srccnt[DN_MAXSRC]; int pcen; float lsc, lsm, lsv; int ilsc; double lsg; int fixdone, Rfix, hwarm, icos, subps; double zvh_l; long zvh_n; float subp0, subp1, izv; uint8_t *subm, *d_subm; float *zvg; double cosacc; long cosn; float sig; int sigS; double sig_l; long sig_n; float *sigA;
    /* --ssl-distill W: dense cosine distillation of dinovol's unit 64-d sheet (object) embedding at the mapped points into head channels
       dch..dch+63 (student: the cell centres of each global; target: dinovol on the partner's un-augmented global at the same physical point);
       --ssl-distill-abs A scales that per-point term, --ssl-distill-rel W adds W x the crop's similarity-matrix match (ss_relsim: rotation-free) */
    float dist; int dch, dnd; float *dcrd, *dacc, *dth, *dse, *dsn, *dge, *dch_h, *dacch; double dist_l; long dist_n; float drel, dabs; double rel_l; long rel_n; float dnce, dthr, dtau; int dnm; double nce_l, nce_pf; long nce_n; double zn_s, zsim_s; long zn_n; float dcc; int dccmin, dkm, dksplit; double cc_l, cc_keep, cc_nc, cc_pos, cc_big, cc_skip; long cc_n, cc_v; int dstr, dnp, *ddp, *ddph; float *dtd; double dtm; long dtmn;   /* (dense mode: --ssl-distill-stride S grid points of every global, its own crop) */
    int itok, nregt, ICIN, *d_tS, *d_tT, *tS, *tT; float ig, *irsS, *irsT;   /* --ssl-ibot-src tok: the iBOT head on the attention stage's tokens (cells = token cells); --ssl-ibot-g */ uint16_t *dvtok;
} dn_ssl;
static int dn_ssl_init(dn_ssl *Q, unet *u, const unet_cfg *cfg, const dn_data *Dd, int argc, char **argv, uint64_t seed, int C, int cls_reg, int cls_norm, float noise) {
    memset(Q, 0, sizeof *Q); Q->u = u; Q->C = C; Q->CC = cls_reg == 2 ? 2 * C : C; Q->cls_reg = cls_reg; Q->cls_norm = cls_norm; Q->noise = noise; Q->CIN = cfg->cin;
    Q->N = atoi(opt(argc, argv, "--ssl-batch", "16")); Q->PG = atoi(opt(argc, argv, "--ssl-P", "128")); Q->PL = atoi(opt(argc, argv, "--ssl-local-P", "64"));
    Q->NL = atoi(opt(argc, argv, "--ssl-locals", "4")); Q->Ps = atoi(opt(argc, argv, "--ssl-source-P", "192")); Q->K = atoi(opt(argc, argv, "--ssl-protos", "16384"));
    Q->ts = 0.1f; Q->wdino = (float)atof(opt(argc, argv, "--ssl-dino", "1")); Q->wk = (float)atof(opt(argc, argv, "--ssl-koleo", "0.1"));
    Q->hlr = (float)atof(opt(argc, argv, "--ssl-head-lr", "1.8e-4")); Q->hwarm = atoi(opt(argc, argv, "--ssl-head-warmup", "0")); Q->plr = (float)atof(opt(argc, argv, "--ssl-proto-lr", opt(argc, argv, "--ssl-head-lr", "1.8e-4")));
    Q->wd = (float)atof(opt(argc, argv, "--ssl-wd", "0.04")); Q->freeze = atoi(opt(argc, argv, "--ssl-freeze-last", "1250")); Q->skit = atoi(opt(argc, argv, "--ssl-sk-iters", "3"));
    Q->adrop = (float)atof(opt(argc, argv, "--axis-drop", "0.15")); Q->ure = atoi(opt(argc, argv, "--usage-window", "5000"));
    const int L = 1 << (cfg->nlev - 1);
    if (Q->N < 2 || Q->N > 1024 || Q->PG % L || Q->PL % L || Q->NL > 8) { fprintf(stderr, "dino-train --ssl-batch: 2..1024 samples, views multiples of %d, <= 8 locals\n", L); return 2; }
    Q->exact = atoi(opt(argc, argv, "--ssl-exact-bank", "0")); Q->kg = Q->exact ? malloc((size_t)2 * Q->N * Q->CC * 4) : nullptr; Q->kgd = nn_malloc((size_t)Q->CC * 4); Q->hasaxv = malloc((size_t)Q->N * 4); Q->amd = nn_malloc((size_t)2 * Q->N * 4); Q->amh = malloc((size_t)2 * Q->N * 4);
    float zlo = 0.84f, zhi = 1.15f, lzlo = 0.86f, lzhi = 1.16f;
    sscanf(opt(argc, argv, "--ssl-zoom", "0.84,1.15"), "%f,%f", &zlo, &zhi); sscanf(opt(argc, argv, "--ssl-local-zoom", "0.86,1.16"), "%f,%f", &lzlo, &lzhi);
    if ((int)ceilf(Q->PG * zhi) > Q->Ps || (int)ceilf(Q->PL * lzhi) > Q->Ps) { fprintf(stderr, "dino-train --ssl-batch: views must fit the source window\n"); return 2; }
    /* the SSL sampler: the sources at the SSL window, no labels drawn (views only) */
    dn_data *Ds = malloc(sizeof *Ds); *Ds = *Dd; Ds->P = Q->Ps; Ds->ngt = 64; Ds->nair = 0; Ds->nink = 0; Ds->uw = 0; Ds->ikp = 0; Ds->ident = 1;
    Ds->vP = Q->PG; Ds->vPL = Q->PL; Ds->vnloc = Q->NL; Ds->zlo = zlo; Ds->zhi = zhi; Ds->lzlo = lzlo; Ds->lzhi = lzhi;
    Ds->share_u = atof(opt(argc, argv, "--ssl-unlabelled-share", "-1")); Ds->vovl = (float)atof(opt(argc, argv, "--ssl-overlap", "0.5")); Q->Ds = Ds; Ds->xprob = (float)atof(opt(argc, argv, "--ssl-xscan", "0"));
    {   /* --ssl-weight-pow p: the SSL sampler draws sources by weight^p (1: by voxel weight; 0.5: flatter, the small scans seen more often) */
        const double wp = atof(opt(argc, argv, "--ssl-weight-pow", "1"));
        if (wp != 1) { Ds->wsum = Ds->wsum_l = Ds->wsum_u = 0;
            for (int k = 0; k < Ds->nsrc; k++) { Ds->src[k].weight = pow(Ds->src[k].weight, wp); Ds->wsum += Ds->src[k].weight; if (Ds->src[k].ct_only) Ds->wsum_u += Ds->src[k].weight; else Ds->wsum_l += Ds->src[k].weight; }
            fprintf(stderr, "dino-train --ssl-weight-pow %g: SSL source weights flattened\n", wp); }
    }
    dn_ring *R = calloc(1, sizeof *R); R->D = Ds; R->seed = seed * 7 + 1234567; R->nthreads = 2; pthread_mutex_init(&R->mu, nullptr); pthread_cond_init(&R->cv, nullptr);
    R->nslots = 2 * Q->N + 4 < DN_SLOTS ? 2 * Q->N + 4 : DN_SLOTS; const size_t ps3 = (size_t)Q->Ps * Q->Ps * Q->Ps;
    for (int i = 0; i < R->nslots; i++) { dn_sample *S = &R->slot[i]; S->ca = malloc(ps3); S->cb = malloc(ps3); S->gidx = malloc(64 * 4); S->gcode = malloc(64 * 4); S->gxyz = malloc(64 * 12); S->sidx = malloc(4); S->ayx = malloc((size_t)Q->Ps * 8); S->ixyz = malloc(12); }
    R->nworkers = atoi(opt(argc, argv, "--ssl-workers", "8")); if (R->nworkers > DN_MAXW) R->nworkers = DN_MAXW;
    for (int i = 0; i < R->nworkers; i++) pthread_create(&R->th[i], nullptr, ring_worker, R);
    Q->R = R;
    /* nets: the globals' and the locals' sizes, the owner's weights (teacher: its EMA) */
    Q->ug = unet_create(cfg); Q->ul = unet_create(cfg);
    unet_use_params(Q->ug, unet_live_params(u)); unet_use_params(Q->ul, unet_live_params(u));
    Q->xsg = (shape5){1, cfg->cin, Q->PG, Q->PG, Q->PG}; Q->xsl = (shape5){1, cfg->cin, Q->PL, Q->PL, Q->PL};
    unet_build(Q->ug, Q->xsg, 1); unet_build(Q->ul, Q->xsl, 1);
    const size_t g3 = (size_t)Q->PG * Q->PG * Q->PG, l3 = (size_t)Q->PL * Q->PL * Q->PL; const int N = Q->N, NV = 2 * N + Q->NL * N, CO = cfg->cout;
    Q->src = nn_malloc((size_t)(Ds->xprob > 0 ? 2 : 1) * N * ps3); Q->xg = nn_malloc((size_t)2 * N * Q->CIN * g3 * 4); Q->xl = nn_malloc((size_t)(Q->NL ? Q->NL : 1) * N * Q->CIN * l3 * 4);
    Q->glzg = nn_malloc(g3 * CO * 4); nn_zero(Q->glzg, g3 * CO * 4); Q->glzl = nn_malloc(l3 * CO * 4); nn_zero(Q->glzl, l3 * CO * 4);
    Q->xfd = nn_malloc(sizeof(dr_xf) * NV); Q->itd = nn_malloc(sizeof(dr_int) * NV); Q->xfh = malloc(sizeof(dr_xf) * NV); Q->ith = malloc(sizeof(dr_int) * NV);
    Q->d_ayx = nn_malloc((size_t)N * Q->Ps * 8);
    const int HH = atoi(opt(argc, argv, "--ssl-head-hidden", "2048")), HB = atoi(opt(argc, argv, "--ssl-head-bottleneck", "256"));
    dn_head_init(&Q->H, Q->CC, HH, HB, Q->K, 2 * N, seed * 59 + 3, 0); Q->H.train_g = 0;
    Q->tl = nn_malloc((size_t)2 * N * Q->K * 4); Q->tq = nn_malloc((size_t)2 * N * Q->K * 4); Q->sks = nn_malloc(dr_sk_scratch(2 * N, Q->K));
    Q->sl = nn_malloc((size_t)Q->K * 4); Q->dsl = nn_malloc((size_t)Q->K * 4); Q->ces = nn_malloc(dr_ce_scratch(1));
    Q->clsT = nn_malloc((size_t)2 * N * Q->CC * 4); Q->clsS = nn_malloc((size_t)Q->CC * 4); Q->dcls = nn_malloc((size_t)Q->CC * 4); Q->rsS = nn_malloc(16); Q->rsT = nn_malloc((size_t)2 * N * 4);
    Q->bank = nn_malloc((size_t)2 * N * Q->CC * 4); Q->bex = nn_malloc((size_t)N * Q->CC * 4); Q->outs = nn_malloc(16 * 4); Q->hout = malloc(16 * 4);
    Q->pS = nn_malloc(8); Q->pT = nn_malloc(8); Q->pW = nn_malloc(8); Q->use = calloc((size_t)Q->K, 1); Q->ar = seed * 977 + 31;
    Q->easy = atoi(opt(argc, argv, "--ssl-easy-steps", "0")); Q->ccen = atoi(opt(argc, argv, "--ssl-cls-center", "0"));
    Q->cmean = nn_malloc((size_t)Q->CC * 4); Q->hmean = malloc((size_t)Q->CC * 4); Q->cstd = atoi(opt(argc, argv, "--ssl-cls-std", "0"));
    Q->cistd = nn_malloc((size_t)Q->CC * 4); Q->histd = malloc((size_t)Q->CC * 4); Q->cpre = nn_malloc((size_t)Q->CC * 4);
    Q->simg = (float)atof(opt(argc, argv, "--ssl-simdino", "0")); Q->sima = 256.f / (float)atof(opt(argc, argv, "--ssl-sim-eps2", "0.5"));
    if (Q->simg > 0) { const int Bz = atoi(opt(argc, argv, "--ssl-head-bottleneck", "256")); Q->sima = (float)Bz / (float)atof(opt(argc, argv, "--ssl-sim-eps2", "0.5"));
        Q->zT = nn_malloc((size_t)2 * Q->N * Bz * 4); Q->zbank = malloc((size_t)2 * Q->N * Bz * 4); Q->dzh = malloc((size_t)Bz * 4); Q->zcur = malloc((size_t)Bz * 4); Q->zA = malloc((size_t)Bz * Bz * 8); }
    Q->zv = (float)atof(opt(argc, argv, "--ssl-zvic-var", "0")); Q->zcv = (float)atof(opt(argc, argv, "--ssl-zvic-cov", "0"));
    if (Q->zv > 0 || Q->zcv > 0) { const int Bz = atoi(opt(argc, argv, "--ssl-head-bottleneck", "256")); Q->zvb = malloc((size_t)2 * Q->N * Bz * 4); Q->zvh = malloc((size_t)Bz * 4); Q->zvd = nn_malloc((size_t)Bz * 4); Q->zcov = malloc((size_t)2 * Bz * Bz * 8);
        Q->zmu = malloc((size_t)2 * Bz * 8); Q->zsd = malloc((size_t)2 * Bz * 8); }
    Q->vramp = atoi(opt(argc, argv, "--ssl-vic-ramp", "0"));
    Q->vv = (float)atof(opt(argc, argv, "--ssl-vic-var", "0")); Q->vc = (float)atof(opt(argc, argv, "--ssl-vic-cov", "0"));
    if (Q->vv > 0 || Q->vc > 0) { Q->vx = malloc((size_t)2 * Q->N * Q->CC * 4); Q->vg = malloc((size_t)Q->CC * 4); Q->vgd = nn_malloc((size_t)Q->CC * 4); Q->vcov = malloc((size_t)2 * Q->CC * Q->CC * 8);
        Q->vmu = malloc((size_t)2 * Q->CC * 8); Q->vsd = malloc((size_t)2 * Q->CC * 8); }
    Q->cm = (float)atof(opt(argc, argv, "--ssl-center", "0")); Q->icm = (float)atof(opt(argc, argv, "--ssl-ibot-center", "0"));
    Q->cen = nn_malloc((size_t)Q->K * 4); nn_zero(Q->cen, (size_t)Q->K * 4);
    Q->wib = (float)atof(opt(argc, argv, "--ssl-ibot", "0")); Q->KI = atoi(opt(argc, argv, "--ssl-ibot-protos", "8192")); Q->IM = atoi(opt(argc, argv, "--ssl-ibot-max", "512"));
    Q->ich = atoi(opt(argc, argv, "--ssl-ibot-ch", "0")); Q->TPS = 8; Q->GT_ = Q->PG / Q->TPS;
    Q->itok = !strcmp(opt(argc, argv, "--ssl-ibot-src", "ch"), "tok"); Q->ICIN = 32; Q->ig = (float)atof(opt(argc, argv, "--ssl-ibot-g", "1"));
    if (Q->itok) {
        if (!cfg->attn_blocks) { fprintf(stderr, "dino-train --ssl-ibot-src tok: needs the attention stage\n"); return 2; }
        Q->TPS = 1 << (cfg->nlev - 1); Q->GT_ = Q->PG / Q->TPS; Q->ICIN = Q->C; Q->nregt = cfg->attn_reg;   /* one masked cell per bottleneck token; the head reads the C-wide token */
    }
    const char *dvck = opt(argc, argv, "--ssl-dv-teacher", nullptr); float *dvPh = nullptr;
    if (dvck) {
        if (Q->wib <= 0 || Q->PG % 16) { fprintf(stderr, "dino-train --ssl-dv-teacher: needs --ssl-ibot > 0 and --ssl-P a multiple of 16\n"); return 2; }
        if (xnet_init(atoi(opt(argc, argv, "--gpu", "0")))) return 1;
        xnet_set_budget((size_t)(atof(opt(argc, argv, "--ssl-dv-mem", "4")) * 1e9));
        Q->dist = (float)atof(opt(argc, argv, "--ssl-distill", "0")); Q->dch = atoi(opt(argc, argv, "--ssl-distill-ch", "0"));
        Q->drel = (float)atof(opt(argc, argv, "--ssl-distill-rel", "0")); Q->dabs = (float)atof(opt(argc, argv, "--ssl-distill-abs", "1"));
        Q->dnce = (float)atof(opt(argc, argv, "--ssl-distill-nce", "0")); Q->dthr = (float)atof(opt(argc, argv, "--ssl-distill-thr", "0.5"));   /* (teacher-graph SupCon: positives = dinovol cos > thr) */
        Q->dcc = (float)atof(opt(argc, argv, "--ssl-distill-cc", "0")); Q->dccmin = atoi(opt(argc, argv, "--ssl-distill-ccmin", "8"));
        Q->dkm = atoi(opt(argc, argv, "--ssl-distill-kmeans", "0")); Q->dksplit = atoi(opt(argc, argv, "--ssl-distill-ksplit", "0"));   /* (labels = spherical k-means of the codes per view (K clusters), optionally split into spatial components) */   /* (pseudo sheet labels: connected components of the dinovol cosine graph -> GT SupCon) */
        Q->dtau = (float)atof(opt(argc, argv, "--ssl-distill-tau", "0.1")); Q->dnm = atoi(opt(argc, argv, "--ssl-distill-m", "512"));
        Q->dstr = atoi(opt(argc, argv, "--ssl-distill-stride", "0"));   /* 0: the iBOT rows (cross-view); S: dense, every S-th voxel per axis of each global (own crop) */   /* (relational: crop-local similarity matrices; abs: the per-point cosine) */
        if (!(Q->dv = dv_load(dvck)) || (Q->dist <= 0 && dv_head_host(dvck, "ibot_head", &Q->DVH, &dvPh))) return 1;
        Q->icos = !strcmp(opt(argc, argv, "--ssl-ibot-loss", "ce"), "cos");   /* cos: regress dinovol's (unit) patch token directly, no prototypes */
        if (Q->dist > 0) { int ni; dv_dense_dims(Q->dv, &Q->dnd, &ni); Q->icos = 0;
            if (Q->dch + Q->dnd > CO) { fprintf(stderr, "dino-train --ssl-distill: head channels %d..%d of %d outputs (--dim %d)\n", Q->dch, Q->dch + Q->dnd - 1, CO, Q->dnd); return 2; } }
        if (Q->dist <= 0 && !Q->icos && Q->DVH.B != atoi(opt(argc, argv, "--ssl-head-bottleneck", "256"))) { fprintf(stderr, "dino-train --ssl-dv-teacher: --ssl-head-bottleneck must be %d (dinovol's iBOT bottleneck)\n", Q->DVH.B); return 2; }
        Q->dvuntil = atoi(opt(argc, argv, "--ssl-dv-until", "0"));
        Q->dvm = calloc(Dd->nsrc, 8); Q->dvs = calloc(Dd->nsrc, 8); Q->dvh = calloc(Dd->nsrc, 8);
        Q->dvhist = calloc((size_t)Dd->nsrc * 256, sizeof *Q->dvhist);   /* dinovol's robust normalisation per source from the histogram of its windows so far */
    }
    if (Q->wib > 0) {
        if (dvck && Q->dist <= 0) Q->KI = Q->icos ? 64 : Q->DVH.K;   /* (cos: a dummy prototype layer) */
        if (Q->ich + 32 > CO || Q->PG % Q->TPS) { fprintf(stderr, "dino-train --ssl-ibot: channels %d..%d of %d outputs\n", Q->ich, Q->ich + 31, CO); return 2; }
        const int R = 2 * N * Q->IM, KI = Q->KI, IM = Q->IM;
        dn_head_init(&Q->I, Q->ICIN, HH, Q->icos ? Q->DVH.Cin : HB, KI, R, seed * 61 + 5, 0); Q->I.train_g = 0;
        if (Q->ig != 1.f) { float *gh = malloc((size_t)KI * 4); for (int k = 0; k < KI; k++) gh[k] = Q->ig; nn_h2d(Q->I.p + Q->I.H.g, gh, (size_t)KI * 4); nn_h2d(Q->I.pt + Q->I.H.g, gh, (size_t)KI * 4); free(gh); }
        if (Q->itok) { Q->irsS = nn_malloc((size_t)IM * 4); Q->irsT = nn_malloc((size_t)R * 4); Q->d_tS = nn_malloc((size_t)2 * N * IM * 4); Q->d_tT = nn_malloc((size_t)2 * N * IM * 4); Q->tS = malloc((size_t)2 * N * IM * 4); Q->tT = malloc((size_t)2 * N * IM * 4); }
        Q->ieT = nn_malloc((size_t)R * Q->ICIN * 4); Q->inT = nn_malloc((size_t)R * 4); Q->ieS = nn_malloc((size_t)IM * Q->ICIN * 4); Q->inS = nn_malloc((size_t)IM * 4); Q->idE = nn_malloc((size_t)IM * Q->ICIN * 4);
        Q->il = nn_malloc((size_t)R * KI * 4); Q->iq = nn_malloc((size_t)R * KI * 4); Q->isk = nn_malloc(dr_sk_scratch(R, KI)); Q->isl = nn_malloc((size_t)IM * KI * 4); Q->idsl = nn_malloc((size_t)IM * KI * 4);
        Q->ice = nn_malloc(dr_ce_scratch(IM)); Q->iw = nn_malloc((size_t)IM * 4); Q->glzib = nn_malloc(g3 * CO * 4); nn_zero(Q->glzib, g3 * CO * 4);
        Q->d_hS = nn_malloc((size_t)2 * N * IM * 4); Q->d_hT = nn_malloc((size_t)2 * N * IM * 4); Q->hS = malloc((size_t)2 * N * IM * 4); Q->hT = malloc((size_t)2 * N * IM * 4);
        Q->iid = nn_malloc((size_t)R * 4); { int *h = malloc((size_t)R * 4); for (int i = 0; i < R; i++) h[i] = i; nn_h2d(Q->iid, h, (size_t)R * 4); free(h); }
        Q->icen = nn_malloc((size_t)KI * 4); nn_zero(Q->icen, (size_t)KI * 4);
        Q->cellm = malloc((size_t)2 * N * Q->GT_ * Q->GT_ * Q->GT_); Q->d_mask = nn_malloc((size_t)Q->GT_ * Q->GT_ * Q->GT_);
        Q->subps = atoi(opt(argc, argv, "--ssl-ibot-sub", "0"));   /* > 0: inside each masked cell only random sub-cells of this edge are masked */
        sscanf(opt(argc, argv, "--ssl-ibot-sub-ratio", "0.3,0.5"), "%f,%f", &Q->subp0, &Q->subp1);
        if (Q->subps > 0) { if (Q->TPS % Q->subps || Q->PG % Q->subps) { fprintf(stderr, "dino-train --ssl-ibot-sub: a divisor of the cell edge %d\n", Q->TPS); return 2; }
            const size_t G2 = (size_t)Q->PG / Q->subps; Q->subm = malloc(G2 * G2 * G2); Q->d_subm = nn_malloc(G2 * G2 * G2); }
        Q->pcen = atoi(opt(argc, argv, "--ssl-ibot-proto-center", "0"));   /* subtract the mean prototype after every prototype update (the shared-direction drift) */
        Q->ilsc = atoi(opt(argc, argv, "--ssl-ibot-logit-scale", "0")); Q->lsc = 0.f;   /* a learnable student logit scale exp(lsc) (Adam, lr 1e-2) */
        Q->sig = (float)atof(opt(argc, argv, "--ssl-sigreg", "0")); Q->sigS = atoi(opt(argc, argv, "--ssl-sigreg-slices", "256"));   /* SIGReg (LeJEPA) on the student's iBOT token bottleneck */
        Q->izv = (float)atof(opt(argc, argv, "--ssl-ibot-zvar", "0"));   /* variance hinge on the student's unit token bottleneck: per-dim std over a view's rows >= 1 / sqrt(B) */
        Q->zvg = nn_malloc((size_t)IM * (HB > 1024 ? HB : 1024) * 4); Q->usei = calloc((size_t)KI, 1);
        if (Q->dv && Q->dist > 0) {   /* distillation buffers (our own EMA teacher keeps the iBOT targets) */
            int nd, ni; const int DC = dv_dense_dims(Q->dv, &nd, &ni);
            if (Q->dstr > 0) { const int gs_ = Q->PG / Q->dstr; Q->dnp = gs_ * gs_ * gs_; Q->ddph = malloc((size_t)2 * N * Q->dnp * 4); Q->ddp = nn_malloc((size_t)2 * N * Q->dnp * 4);
                Q->dtd = malloc((size_t)2 * N * Q->dnp * nd * 4); }
            const size_t MX = (size_t)(Q->dnp > IM ? Q->dnp : IM);
            Q->dcrd = xmalloc_dev(MX * 12); Q->dacc = xmalloc_dev(MX * DC * 4); Q->dacch = malloc(MX * DC * 4); Q->dch_h = malloc(MX * 12);
            Q->dth = malloc((size_t)R * nd * 4); Q->dse = nn_malloc(MX * nd * 4); Q->dsn = nn_malloc(MX * 4); Q->dge = nn_malloc(MX * nd * 4);
            Q->dvf = nn_malloc(g3 * 4); Q->dvu = nn_malloc(g3); Q->smsi = calloc(N, sizeof *Q->smsi);
            dr_int id; memset(&id, 0, sizeof id); id.center = 0.f; id.scale = 1.f; id.gamma = 1.f; id.lo = 0.f; id.hi = 255.f; id.contrast = 1.f;
            Q->dvid = nn_malloc(sizeof id); nn_h2d(Q->dvid, &id, sizeof id);
            fprintf(stderr, "dino-train --ssl-distill %g: dinovol %s's unit %d-d sheet embedding (dense head) at each global's mapped cell centres -> head channels %d..%d (1 - cos)\n",
                    Q->dist, dvck, nd, Q->dch, Q->dch + nd - 1);
        } else if (Q->dv) {   /* the student head's last layer = dinovol's prototypes (frozen); the teacher-side buffers */
            const dr_head *IH = &Q->I.H, *DH = &Q->DVH;
            if (!Q->icos) { nn_h2d(Q->I.p + IH->v, dvPh + DH->v, (size_t)DH->K * DH->B * 4); nn_h2d(Q->I.p + IH->g, dvPh + DH->g, (size_t)DH->K * 4); }
            nn_d2d(Q->I.pt, Q->I.p, IH->n * 4);
            Q->dvP = nn_malloc(DH->n * 4); nn_h2d(Q->dvP, dvPh, DH->n * 4); free(dvPh);
            Q->dvx = malloc((size_t)R * DH->Cin * 4); Q->dvxd = nn_malloc((size_t)R * DH->Cin * 4);
            Q->dvwb = nn_malloc(dr_hws_bytes(DH, R)); dr_hws_carve(&Q->dvw, DH, R, Q->dvwb);
            Q->dvf = nn_malloc(g3 * 4); Q->dvu = nn_malloc(g3); Q->smsi = calloc(N, sizeof *Q->smsi);
            const int G = Q->PG / 8; Q->dvtok = malloc((size_t)G * G * G * DH->Cin * 2);
            dr_int id; memset(&id, 0, sizeof id); id.center = 0.f; id.scale = 1.f; id.gamma = 1.f; id.lo = 0.f; id.hi = 255.f; id.contrast = 1.f;
            Q->dvid = nn_malloc(sizeof id); nn_h2d(Q->dvid, &id, sizeof id);
            fprintf(stderr, "dino-train --ssl-dv-teacher %s (diag-dv): iBOT targets from the frozen dinovol teacher (%d prototypes, bottleneck %d)%s; student iBOT last layer = dinovol's, frozen\n",
                    dvck, DH->K, DH->B, Q->dvuntil > 0 ? ", until the handover step" : "");
            if (Q->dvuntil > 0) fprintf(stderr, "dino-train --ssl-dv-until %d: then the EMA teacher's iBOT head (initialised from the student's)\n", Q->dvuntil);
        }
        fprintf(stderr, "dino-train --ssl-ibot %g: masked %d^3 cells (<= %d per global), head channels %d..%d, iBOT head -> %d prototypes, Sinkhorn over the batch's masked rows\n", Q->wib, Q->TPS, IM, Q->ich, Q->ich + 31, KI);
    }
    fprintf(stderr, "dino-train --ssl-batch: %d samples of %d^3 windows per step (%d workers, %d slots), 2 globals %d^3 (zoom %.2f..%.2f) + %d locals %d^3 (%.2f..%.2f), "
                    "CLS %s, DINO head %d-%d-%d -> %d prototypes (AdamW %.2g, prototypes %.2g, wd %.2g, frozen %d steps), Sinkhorn over the batch's %d teacher rows, KoLeo %g per view chunk\n",
            N, Q->Ps, R->nworkers, R->nslots, Q->PG, zlo, zhi, Q->NL, Q->PL, lzlo, lzhi, cls_reg == 2 ? "register 0 | token mean" : cls_reg ? "register 0" : "token mean", HH, HH, HB, Q->K, Q->hlr, Q->plr, Q->wd, Q->freeze, 2 * N, Q->wk);
    return 0;
}
/* the SSL heads beside a checkpoint (CKPT.ssl): DINO head p, pt, the centre; iBOT head p, pt, its centre (when on) */
static void dn_ssl_io(dn_ssl *Q, const char *ckpt, int save) {
    char fp[1300]; snprintf(fp, sizeof fp, "%s.ssl", ckpt); FILE *f = fopen(fp, save ? "wb" : "rb"); if (!f) return;
    struct { float *d; size_t n; } a[6] = {{Q->H.p, Q->H.H.n}, {Q->H.pt, Q->H.H.n}, {Q->cen, (size_t)Q->K}, {Q->wib > 0 ? Q->I.p : nullptr, Q->wib > 0 ? Q->I.H.n : 0},
                                           {Q->wib > 0 ? Q->I.pt : nullptr, Q->wib > 0 ? Q->I.H.n : 0}, {Q->wib > 0 ? Q->icen : nullptr, Q->wib > 0 ? (size_t)Q->KI : 0}};
    int ok = 1;
    for (int k = 0; k < 6 && ok; k++) { if (!a[k].d) continue; float *h = malloc(a[k].n * 4);
        if (save) { nn_d2h(h, a[k].d, a[k].n * 4); ok = fwrite(h, 4, a[k].n, f) == a[k].n; } else { ok = fread(h, 4, a[k].n, f) == a[k].n; if (ok) nn_h2d(a[k].d, h, a[k].n * 4); }
        free(h); }
    fclose(f);
    if (!save) fprintf(stderr, "dino-train --ssl-batch: heads %s from %s\n", ok ? "restored" : "NOT restored (short file)", fp);
}
/* teacher-side diagnostics (every 50 steps, before the centre update): the head's bottleneck zn cosine between samples, the teacher logit
   std over prototypes per row (raw cosines and after the centre), the centre norm, the prototype directions' mean |cosine| (256 sampled) */
static void dn_ssl_diag(dn_ssl *Q) {
    const int R = 2 * Q->N, K = Q->K, B = Q->H.H.B; float *tl = malloc((size_t)R * K * 4), *cen = malloc((size_t)K * 4), *zn = malloc((size_t)R * B * 4);
    nn_d2h(tl, Q->tl, (size_t)R * K * 4); nn_d2h(cen, Q->cen, (size_t)K * 4); nn_d2h(zn, Q->H.w.zn, (size_t)R * B * 4);
    double sraw = 0, scen = 0, cn = 0;
    for (int r = 0; r < R; r++) { double m = 0, v = 0, mc = 0, vc = 0; for (int k = 0; k < K; k++) { const double a = tl[(size_t)r * K + k], b = a - cen[k]; m += a; v += a * a; mc += b; vc += b * b; }
        m /= K; mc /= K; sraw += sqrt(fmax(v / K - m * m, 0)) / R; scen += sqrt(fmax(vc / K - mc * mc, 0)) / R; }
    for (int k = 0; k < K; k++) cn += (double)cen[k] * cen[k];
    double zc = 0; long nz = 0;
    for (int a = 0; a < R; a++) for (int b = a + 1; b < R; b++) if (a / 2 != b / 2) { double x = 0; for (int c = 0; c < B; c++) x += (double)zn[(size_t)a * B + c] * zn[(size_t)b * B + c]; zc += x; nz++; }
    const int NP = K < 256 ? K : 256; float *V = malloc((size_t)NP * B * 4); double pc = 0; long np_ = 0;
    nn_d2h(V, Q->H.p + Q->H.H.v, (size_t)NP * B * 4);
    for (int a = 0; a < NP; a++) for (int b = a + 1; b < NP; b++) { double x = 0, ya = 0, yb = 0; for (int c = 0; c < B; c++) { x += (double)V[(size_t)a * B + c] * V[(size_t)b * B + c]; ya += (double)V[(size_t)a * B + c] * V[(size_t)a * B + c]; yb += (double)V[(size_t)b * B + c] * V[(size_t)b * B + c]; } pc += fabs(x) / sqrt(ya * yb + 1e-30); np_++; }
    Q->dg[0] = nz ? zc / nz : 0; Q->dg[1] = sraw; Q->dg[2] = scen; Q->dg[3] = sqrt(cn); Q->dg[4] = np_ ? pc / np_ : 0;
    free(tl); free(cen); free(zn); free(V);
}
static void dn_ssl_cls(dn_ssl *Q, unet *n, float *cls) {   /* the CLS of the net's last forward */
    int S = 0, nreg = 0; const float *tok = unet_tokens(n, nullptr, &S, nullptr, &nreg);
    if (Q->cls_reg == 2) { nn_d2d(cls, tok, (size_t)Q->C * 4); dr_pool_fwd(tok, 1, S, Q->C, nreg, cls + Q->C); }   /* (--cls both: register | token mean) */
    else if (Q->cls_reg) nn_d2d(cls, tok, (size_t)Q->C * 4); else dr_pool_fwd(tok, 1, S, Q->C, nreg, cls);
}
static int UF_FIND2(int *par, int x) { while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; } return x; }
static int dn_ssl_step(dn_ssl *Q, int step, float tt, float mom) {
    const int N = Q->N, C = Q->CC, CT = Q->C, K = Q->K, NV = 2 * N + Q->NL * N, CIN = Q->CIN; const size_t g3 = (size_t)Q->PG * Q->PG * Q->PG, l3 = (size_t)Q->PL * Q->PL * Q->PL, ps3 = (size_t)Q->Ps * Q->Ps * Q->Ps;
    const float gs = nn_get_grad_scale(); int *hasax = Q->hasaxv;
    /* N samples: windows up, views (the workers' crops / symmetries) + per-view intensity augmentation */
    static int fixed = -1, nomask = -1, dbg = -1;   /* debug: UFSM_SSL_FIXED=1 one fixed batch (the first step's samples, views and masks) every step;
                                                       UFSM_SSL_NOMASK=1 the student sees unmasked views; UFSM_SSL_DBG=1 gradient norms */
    if (fixed < 0) { fixed = getenv("UFSM_SSL_FIXED") && atoi(getenv("UFSM_SSL_FIXED")); nomask = getenv("UFSM_SSL_NOMASK") && atoi(getenv("UFSM_SSL_NOMASK")); dbg = getenv("UFSM_SSL_DBG") && atoi(getenv("UFSM_SSL_DBG")); }
    const int reuse = fixed && Q->fixdone;
    if (!reuse) {
    for (int i = 0; i < N; i++) {
        dn_sample *Sm = ring_get(Q->R); if (!Sm) { fprintf(stderr, "dino-train --ssl-batch: loader failed\n"); return 1; }
        const dn_src *sr = &Q->Ds->src[Sm->si];
        Q->srccnt[Sm->si]++;
        if (Q->smsi) { Q->smsi[i] = Sm->si; int64_t *hh = Q->dvhist + (size_t)Sm->si * 256; for (size_t v = 0; v < ps3; v += 5) hh[Sm->ca[v]]++;
            if (dv_hist_stats(hh, &Q->dvm[Sm->si], &Q->dvs[Sm->si], &Q->dvh[Sm->si])) { Q->dvm[Sm->si] = sr->center; Q->dvs[Sm->si] = sr->scale; Q->dvh[Sm->si] = 255; } }
        const int xs_ = Q->Ds->xprob > 0 && Sm->xs; if (xs_) nn_h2d(Q->src + (size_t)(N + i) * ps3, Sm->cb, ps3); Q->sums[7] += (double)xs_ / N; if (xs_) { Q->xnccs += Sm->xncc; Q->xnccn++; }
        nn_h2d(Q->src + (size_t)i * ps3, Sm->ca, ps3); hasax[i] = Sm->hasax; if (CIN == 5 && Sm->hasax) nn_h2d(Q->d_ayx + (size_t)i * Q->Ps * 2, Sm->ayx, (size_t)Q->Ps * 8);
        for (int k = 0; k < 2 + Q->NL; k++) {
            const int j = k < 2 ? 2 * i + k : 2 * N + Q->NL * i + (k - 2);
            Q->xfh[j] = Sm->xf[k]; Q->xfh[j].src = k == 1 && xs_ ? N + i : i;   /* (cross-scan: global 1 from the partner scan) */
            dr_int *a = &Q->ith[j]; memset(a, 0, sizeof *a); uint64_t *ar = &Q->ar;
            a->center = sr->center; a->scale = sr->scale; a->lo = 0.f; a->hi = 255.f;
            a->gamma = rng_u(ar) < 0.25 ? expf((float)(2 * rng_u(ar) - 1) * logf(1.25f)) : 1.f;
            a->contrast = rng_u(ar) < 0.25 ? 0.8f + 0.4f * (float)rng_u(ar) : 1.f; a->mean = 0.f;
            a->bias = rng_u(ar) < 0.15 ? 0.15f * (float)(sqrt(-2 * log(rng_u(ar) + 1e-12)) * cos(6.283185307 * rng_u(ar))) : 0.f;
            a->invert = rng_u(ar) < 0.15; a->sigma = Q->noise * (float)rng_u(ar); a->seed = (uint32_t)(step * 4099 + j);
        }
        ring_put(Q->R, Sm);
        if (step <= Q->easy) for (int k = 0; k < 2 + Q->NL; k++) {   /* easy views: every view of the sample in global 0's symmetry; no inversion; mild intensity */
            const int j = k < 2 ? 2 * i + k : 2 * N + Q->NL * i + (k - 2), j0 = 2 * i;
            for (int d = 0; d < 3; d++) { Q->xfh[j].perm[d] = Q->xfh[j0].perm[d]; Q->xfh[j].flip[d] = Q->xfh[j0].flip[d]; }
            dr_int *a = &Q->ith[j]; a->invert = 0; a->gamma = 1.f + 0.5f * (a->gamma - 1.f); a->contrast = 1.f + 0.5f * (a->contrast - 1.f); a->bias *= 0.5f; a->sigma *= 0.5f;
        }
        if (getenv("UFSM_SSL_IDTEST")) { Q->xfh[2 * i + 1] = Q->xfh[2 * i]; Q->ith[2 * i + 1] = Q->ith[2 * i]; }   /* (test: global 1 = global 0 exactly) */
        {   /* the two globals' source-box IoU (diagnostic: overlapping views) */
            const dr_xf *a = &Q->xfh[2 * i], *b = &Q->xfh[2 * i + 1]; double in = 1;
            for (int d = 0; d < 3; d++) { const int lo = a->off[d] > b->off[d] ? a->off[d] : b->off[d], hi = a->off[d] + a->m < b->off[d] + b->m ? a->off[d] + a->m : b->off[d] + b->m; in *= hi > lo ? hi - lo : 0; }
            const double va = (double)a->m * a->m * a->m, vb = (double)b->m * b->m * b->m; Q->sums[6] += in / (va + vb - in) / N;
        }
    }
    nn_h2d(Q->xfd, Q->xfh, sizeof(dr_xf) * NV); nn_h2d(Q->itd, Q->ith, sizeof(dr_int) * NV);
    }
    for (int j = 0; j < NV; j++) {
        const int gl = j < 2 * N, n_ = gl ? Q->PG : Q->PL; const size_t n3 = gl ? g3 : l3; float *x = gl ? Q->xg + (size_t)j * CIN * g3 : Q->xl + (size_t)(j - 2 * N) * CIN * l3;
        dr_crops(Q->src, Q->Ps, Q->xfd + j, Q->itd + j, 1, n_, x);
        if (CIN == 2) nn_zero(x + n3, n3 * 4);
        if (CIN == 5) { const int i = Q->xfh[j].src; dr_axis_view(Q->xfd, j, n_, Q->Ps, Q->d_ayx + (size_t)i * Q->Ps * 2, hasax[i] && rng_u(&Q->ar) >= Q->adrop, x + n3); }
    }
    if (step % 1000 == 0) {   /* the SSL samples drawn per source so far */
        fprintf(stderr, "ssl sources at step %d:", step); long tot = 0;
        for (int k = 0; k < Q->Ds->nsrc; k++) { tot += Q->srccnt[k]; if (Q->srccnt[k]) fprintf(stderr, " %s=%ld", Q->Ds->src[k].name, Q->srccnt[k]); }
        int none = 0; for (int k = 0; k < Q->Ds->nsrc; k++) none += !Q->srccnt[k];
        fprintf(stderr, " | total %ld, %d of %d sources never drawn\n", tot, none, Q->Ds->nsrc);
    }
    int R = reuse ? Q->Rfix : 0;   /* iBOT: the masked cells per global (block masking), their centres mapped into the partner global */
    if (Q->wib > 0 && !reuse) {
        const int G = Q->GT_, TPS = Q->TPS, IM = Q->IM, PG = Q->PG;
        for (int j = 0; j < 2 * N; j++) {
            Q->cnt[j] = 0; Q->off[j] = R; uint8_t *cm = Q->cellm + (size_t)j * G * G * G; memset(cm, 0, (size_t)G * G * G);
            if (rng_u(&Q->ar) >= 0.5) continue;
            dn_block_mask(cm, G, (int)lrint((0.1 + 0.4 * rng_u(&Q->ar)) * G * G * G), &Q->ar);
            const dr_xf *a = &Q->xfh[j], *b = &Q->xfh[j ^ 1]; int seen = 0;
            for (int c = 0; c < G * G * G; c++) {
                if (!cm[c]) continue;
                const int cz = c / (G * G), cy = (c / G) % G, cx = c % G, h2 = TPS / 2;
                const double vv[3] = {cz * TPS + h2, cy * TPS + h2, cx * TPS + h2}; double sx[3], o[3]; dr_map(a, vv, sx); dr_unmap(b, sx, o);
                int q[3], in = 1; for (int d = 0; d < 3; d++) { in &= o[d] >= -0.5 && o[d] <= PG - 0.5; q[d] = (int)lrint(o[d]); q[d] = q[d] < 0 ? 0 : q[d] >= PG ? PG - 1 : q[d]; }
                if (!in) continue;
                int k = seen++; if (k >= IM) { k = (int)(rng_next(&Q->ar) % (uint64_t)seen); if (k >= IM) continue; }
                Q->hS[(size_t)j * IM + k] = (int)((vv[0] * PG + vv[1]) * PG + vv[2]); Q->hT[(size_t)j * IM + k] = (q[0] * PG + q[1]) * PG + q[2];
            }
            const int m = seen < IM ? seen : IM; if (m < (Q->itok ? 2 : 16)) { memset(cm, 0, (size_t)G * G * G); continue; }
            Q->cnt[j] = m; R += m;
            nn_h2d(Q->d_hS + (size_t)j * IM, Q->hS + (size_t)j * IM, (size_t)m * 4); nn_h2d(Q->d_hT + (size_t)j * IM, Q->hT + (size_t)j * IM, (size_t)m * 4);
            if (Q->itok) {   /* token rows (after the registers) of the student's cell centre and of the teacher's mapped point */
                for (int k = 0; k < m; k++) for (int t = 0; t < 2; t++) {
                    const int v = t ? Q->hT[(size_t)j * IM + k] : Q->hS[(size_t)j * IM + k], z = v / (PG * PG) / TPS, y = v / PG % PG / TPS, x = v % PG / TPS;
                    (t ? Q->tT : Q->tS)[(size_t)j * IM + k] = Q->nregt + (z * G + y) * G + x; }
                nn_h2d(Q->d_tS + (size_t)j * IM, Q->tS + (size_t)j * IM, (size_t)m * 4); nn_h2d(Q->d_tT + (size_t)j * IM, Q->tT + (size_t)j * IM, (size_t)m * 4);
            }
        }
    }
    if (dbg && Q->wib > 0 && step % 25 == 1) {   /* correspondence check: the views' (augmented) CT at the student cell centres vs at the mapped teacher points */
        const size_t g3_ = (size_t)Q->PG * Q->PG * Q->PG; float *va = malloc(g3_ * 4), *vb = malloc(g3_ * 4); double nc = 0, ns = 0; int nv = 0;
        for (int j = 0; j < 2 * N && nv < 4; j++) {
            if (!Q->cnt[j]) continue;
            nn_d2h(va, Q->xg + (size_t)j * CIN * g3_, g3_ * 4); nn_d2h(vb, Q->xg + (size_t)(j ^ 1) * CIN * g3_, g3_ * 4);
            const int m = Q->cnt[j]; double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0, sab2 = 0;
            for (int k = 0; k < m; k++) { const double a = va[Q->hS[(size_t)j * Q->IM + k]], b = vb[Q->hT[(size_t)j * Q->IM + k]], b2 = vb[Q->hT[(size_t)j * Q->IM + (k + 7) % m]];
                sa += a; sb += b; saa += a * a; sbb += b * b; sab += a * b; sab2 += a * b2; }
            const double ma = sa / m, mb = sb / m, da = sqrt(fmax(saa / m - ma * ma, 1e-12)), db = sqrt(fmax(sbb / m - mb * mb, 1e-12));
            nc += fabs((sab / m - ma * mb) / (da * db)); ns += fabs((sab2 / m - ma * mb) / (da * db)); nv++;
        }
        if (nv) fprintf(stderr, "dbg step %d: student-cell vs mapped-teacher-point CT |ncc| %.3f (shuffled pairs %.3f) over %d views\n", step, nc / nv, ns / nv, nv);
        free(va); free(vb);
    }
    if (fixed && !Q->fixdone) { Q->fixdone = 1; Q->Rfix = R; fprintf(stderr, "dino-train: UFSM_SSL_FIXED: one fixed batch (%d iBOT rows)\n", R); }
    /* teacher: the 2N globals through the EMA weights; CLS rows, layer norm, head, Sinkhorn over the batch */
    unet_use_params(Q->ug, unet_ema_params(Q->u));
    for (int j = 0; j < 2 * N; j++) {
        const float *lg = unet_forward(Q->ug, Q->xg + (size_t)j * CIN * g3, Q->xsg, 0); dn_ssl_cls(Q, Q->ug, Q->clsT + (size_t)j * C);
        const int w = j ^ 1;   /* (the masked partner whose targets this teacher view holds) */
        if (Q->wib > 0 && Q->cnt[w] && Q->itok) { const float *tk = unet_tokens(Q->ug, nullptr, nullptr, nullptr, nullptr); dr_rows_gather(tk, CT, Q->d_tT + (size_t)w * Q->IM, Q->cnt[w], Q->ieT + (size_t)Q->off[w] * CT);
            dr_rownorm_fwd(Q->ieT + (size_t)Q->off[w] * CT, Q->cnt[w], CT, Q->ieT + (size_t)Q->off[w] * CT, Q->irsT + Q->off[w]); }   /* (a parameter-free LayerNorm: upstream's normed tokens) */
        else if (Q->wib > 0 && Q->cnt[w]) { const int h16 = unet_logits_h16(Q->ug);   /* (the full logits: fp16 or fp32) */
            dn_gather((const char *)lg + (size_t)Q->ich * g3 * (h16 ? 2 : 4), h16, g3, 32, Q->d_hT + (size_t)w * Q->IM, Q->cnt[w], Q->ieT + (size_t)Q->off[w] * 32, Q->inT + Q->off[w]); }
    }
    unet_use_params(Q->ug, unet_live_params(Q->u));
    if (Q->cls_norm) dr_rownorm_fwd(Q->clsT, 2 * N, C, Q->clsT, Q->rsT);
    {   /* per-step stats of the teacher CLS rows: cosine of a sample's two globals vs of different samples (before any batch centring) */
        float *ct = malloc((size_t)2 * N * C * 4); nn_d2h(ct, Q->clsT, (size_t)2 * N * C * 4); double cs = 0, cd = 0; long nd = 0;
        for (int i = 0; i < C; i++) Q->hmean[i] = 0;
        for (int a = 0; a < 2 * N; a++) { for (int i = 0; i < C; i++) Q->hmean[i] += ct[(size_t)a * C + i] / (2 * N);
            for (int b = a + 1; b < 2 * N; b++) { double x = 0, ya = 0, yb = 0; for (int i = 0; i < C; i++) { x += (double)ct[(size_t)a * C + i] * ct[(size_t)b * C + i]; ya += (double)ct[(size_t)a * C + i] * ct[(size_t)a * C + i]; yb += (double)ct[(size_t)b * C + i] * ct[(size_t)b * C + i]; }
                const double cc = x / sqrt(ya * yb + 1e-30); if (a / 2 == b / 2) cs += cc; else { cd += cc; nd++; } } }
        Q->st[0] += cs / N; Q->st[1] += nd ? cd / nd : 0;
        if (Q->cstd) {   /* per-dimension batch standardisation (constants: the teacher batch's mean / std) */
            for (int i = 0; i < C; i++) { double v = 0; for (int a = 0; a < 2 * N; a++) { const double d = ct[(size_t)a * C + i] - Q->hmean[i]; v += d * d; } Q->histd[i] = (float)(1.0 / sqrt(v / (2 * N) + 1e-4)); }
            nn_h2d(Q->cmean, Q->hmean, (size_t)C * 4); nn_h2d(Q->cistd, Q->histd, (size_t)C * 4);
            for (int a = 0; a < 2 * N; a++) dr_cols_affine(Q->clsT + (size_t)a * C, C, Q->cmean, Q->cistd);
        } else if (Q->ccen) { nn_h2d(Q->cmean, Q->hmean, (size_t)C * 4); for (int a = 0; a < 2 * N; a++) nn_axpy(Q->clsT + (size_t)a * C, -1.f, Q->cmean, (size_t)C); }   /* (the batch mean CLS removed: a constant shift) */
        free(ct);
    }
    if (dr_head_fwd(&Q->H.H, Q->H.pt, Q->clsT, 2 * N, Q->tl, &Q->H.w)) return 1;
    if (Q->simg > 0) { nn_d2d(Q->zT, Q->H.w.zn, (size_t)2 * N * Q->H.H.B * 4); nn_d2h(Q->zbank, Q->zT, (size_t)2 * N * Q->H.H.B * 4); }   /* SimDINO targets; coding-rate bank */
    if (Q->zvb) { const int Bz = Q->H.H.B; float *zt = malloc((size_t)2 * N * Bz * 4); nn_d2h(zt, Q->H.w.zn, (size_t)2 * N * Bz * 4);   /* (the z bank, chunk-major: teacher rows) */
        for (int k = 0; k < N; k++) for (int v = 0; v < 2; v++) memcpy(Q->zvb + ((size_t)v * N + k) * Bz, zt + (size_t)(2 * k + v) * Bz, (size_t)Bz * 4);
        free(zt); }
    {   /* the head bottleneck z: same / other sample cosine */
        const int B = Q->H.H.B; float *zn = malloc((size_t)2 * N * B * 4); nn_d2h(zn, Q->H.w.zn, (size_t)2 * N * B * 4); double zs = 0, zd = 0; long nd = 0;
        for (int a = 0; a < 2 * N; a++) for (int b = a + 1; b < 2 * N; b++) { double x = 0; for (int i = 0; i < B; i++) x += (double)zn[(size_t)a * B + i] * zn[(size_t)b * B + i]; if (a / 2 == b / 2) zs += x; else { zd += x; nd++; } }
        Q->st[2] += zs / N; Q->st[3] += nd ? zd / nd : 0;
        double zsd = 0, *zs_ = malloc((size_t)B * 8);
        for (int i = 0; i < B; i++) { double m = 0, v = 0; for (int a = 0; a < 2 * N; a++) m += zn[(size_t)a * B + i]; m /= 2 * N; for (int a = 0; a < 2 * N; a++) { const double d = zn[(size_t)a * B + i] - m; v += d * d; } zs_[i] = sqrt(v / (2 * N - 1)); zsd += zs_[i] / B; }
        for (int i = 1; i < B; i++) { const double t = zs_[i]; int k = i - 1; while (k >= 0 && zs_[k] > t) { zs_[k + 1] = zs_[k]; k--; } zs_[k + 1] = t; }   /* (sorted: min / median) */
        Q->st[8] += zsd; Q->zsmin = zs_[0]; Q->zsmed = zs_[B / 2]; free(zs_); free(zn);
    }
    if (step % 50 == 0) dn_ssl_diag(Q);
    if (Q->cm > 0) { dr_teacher_probs(Q->tl, 2 * N, K, tt, 0, Q->cen, Q->tq, Q->sks); dr_center_update(Q->cen, Q->tl, 2 * N, K, Q->cm); }
    else dr_teacher_probs(Q->tl, 2 * N, K, tt, Q->skit, nullptr, Q->tq, Q->sks);
    { int *am = Q->amd, *hm = Q->amh; dr_argmax_rows(Q->tq, 2 * N, K, am); nn_d2h(hm, am, (size_t)2 * N * 4); for (int j = 0; j < 2 * N; j++) Q->use[hm[j]] = 1;
      dr_argmax_rows(Q->tl, 2 * N, K, am); nn_d2h(hm, am, (size_t)2 * N * 4);   /* (agreement on the raw teacher logits: Sinkhorn's equipartition over 2N << K rows keeps the targets' argmaxes apart) */
      int ag = 0, ab = 0; for (int i = 0; i < N; i++) { ag += hm[2 * i] == hm[2 * i + 1]; ab += hm[2 * i] == hm[2 * ((i + 1) % N) + 1]; } Q->sums[2] += (double)ag / N; Q->st[4] += (double)ab / N; }
    const int dvon = Q->dv && (Q->dvuntil <= 0 || step <= Q->dvuntil);
    if (Q->dist > 0 && Q->dstr > 0 && dvon) {   /* dense distillation targets: dinovol on each un-augmented global, its dense head on an S-grid (random offset) of the same crop */
        int nd, ni; const int DC = dv_dense_dims(Q->dv, &nd, &ni); const int S = Q->dstr, gs_ = Q->PG / S, NP = Q->dnp;
        for (int j = 0; j < 2 * N; j++) {
            static int ddet = -1; if (ddet < 0) ddet = getenv("UFSM_DIST_DET") ? atoi(getenv("UFSM_DIST_DET")) : 0;   /* (debug: offset 0; with UFSM_SSL_FIXED the targets must repeat exactly) */
            int oz = (int)(rng_next(&Q->ar) % (uint64_t)S), oy = (int)(rng_next(&Q->ar) % (uint64_t)S), ox = (int)(rng_next(&Q->ar) % (uint64_t)S); const int si = Q->smsi[j / 2];
            if (ddet) oz = oy = ox = 0;
            dr_crops(Q->src, Q->Ps, Q->xfd + j, Q->dvid, 1, Q->PG, Q->dvf); dr_f2u8(Q->dvf, (long long)g3, Q->dvu);
            nn_sync();
            if (dv_forward(Q->dv, Q->dvu, Q->PG, (float)Q->dvm[si], (float)Q->dvs[si], (float)Q->dvh[si])) return 1;
            int *ix = Q->ddph + (size_t)j * NP;
            for (int a = 0; a < NP; a++) { const int z = oz + S * (a / (gs_ * gs_)), y = oy + S * (a / gs_ % gs_), x = ox + S * (a % gs_);
                ix[a] = (z * Q->PG + y) * Q->PG + x; Q->dch_h[3 * a] = (float)z; Q->dch_h[3 * a + 1] = (float)y; Q->dch_h[3 * a + 2] = (float)x; }
            xh2d(Q->dcrd, Q->dch_h, (size_t)NP * 12);
            if (dv_net_points(Q->dv, Q->dcrd, nullptr, nullptr, NP, Q->dacc)) return 1;
            xd2h(Q->dacch, Q->dacc, (size_t)NP * DC * 4);
            for (int a = 0; a < NP; a++) memcpy(Q->dtd + ((size_t)j * NP + a) * nd, Q->dacch + (size_t)a * DC, (size_t)nd * 4);
        }
        nn_h2d(Q->ddp, Q->ddph, (size_t)2 * N * NP * 4);
        static int ddet2 = -1; if (ddet2 < 0) ddet2 = getenv("UFSM_DIST_DET") ? atoi(getenv("UFSM_DIST_DET")) : 0;
        if (ddet2) { static float *t0; const size_t nt = (size_t)2 * N * NP * nd;
            if (!t0) { t0 = malloc(nt * 4); memcpy(t0, Q->dtd, nt * 4); }
            else { double c = 0, mn = 1; for (size_t k = 0; k < (size_t)2 * N * NP; k++) { double d = 0; for (int e = 0; e < nd; e++) d += (double)t0[k * nd + e] * Q->dtd[k * nd + e]; c += d; if (d < mn) mn = d; }
                fprintf(stderr, "dist det step %d: targets vs step 1 mean cos %.6f, min %.6f\n", step, c / (2.0 * N * NP), mn); } }
    }
    if (Q->dist > 0 && Q->dstr <= 0 && R > 0 && dvon) {   /* distillation targets: dinovol on each un-augmented global, its dense head at the partner's mapped cell centres */
        int nd, ni; const int DC = dv_dense_dims(Q->dv, &nd, &ni);
        for (int j = 0; j < 2 * N; j++) {
            const int w = j ^ 1, m = Q->cnt[w]; if (!m) continue;
            const int si = Q->smsi[j / 2];
            dr_crops(Q->src, Q->Ps, Q->xfd + j, Q->dvid, 1, Q->PG, Q->dvf); dr_f2u8(Q->dvf, (long long)g3, Q->dvu);
            nn_sync();   /* (dinovol runs on xnet's non-blocking stream) */
            if (dv_forward(Q->dv, Q->dvu, Q->PG, (float)Q->dvm[si], (float)Q->dvs[si], (float)Q->dvh[si])) return 1;
            for (int k = 0; k < m; k++) { const int q = Q->hT[(size_t)w * Q->IM + k];
                Q->dch_h[3 * k] = (float)(q / (Q->PG * Q->PG)); Q->dch_h[3 * k + 1] = (float)((q / Q->PG) % Q->PG); Q->dch_h[3 * k + 2] = (float)(q % Q->PG); }
            xh2d(Q->dcrd, Q->dch_h, (size_t)m * 12);
            if (dv_net_points(Q->dv, Q->dcrd, nullptr, nullptr, m, Q->dacc)) return 1;
            xd2h(Q->dacch, Q->dacc, (size_t)m * DC * 4);   /* (synchronous on xnet's stream) */
            for (int k = 0; k < m; k++) memcpy(Q->dth + (size_t)(Q->off[w] + k) * nd, Q->dacch + (size_t)k * DC, (size_t)nd * 4);
        }
        static int dck = -1; if (dck < 0) dck = getenv("UFSM_DIST_CHECK") ? atoi(getenv("UFSM_DIST_CHECK")) : 0;
        if (dck > 0 && step % dck == 1) {   /* alignment check: dinovol on crop w at the student's cell centres hS[w] vs the targets (crop w^1 at hT[w]):
                                               the same physical points seen in two crops; shuffled pairs and a +4 vx z shift for scale */
            double cm = 0, cs = 0, csh = 0; long n_ = 0;
            for (int w = 0; w < 2 * N; w++) {
                const int m = Q->cnt[w]; if (!m) continue; const int si = Q->smsi[w / 2];
                dr_crops(Q->src, Q->Ps, Q->xfd + w, Q->dvid, 1, Q->PG, Q->dvf); dr_f2u8(Q->dvf, (long long)g3, Q->dvu); nn_sync();
                if (dv_forward(Q->dv, Q->dvu, Q->PG, (float)Q->dvm[si], (float)Q->dvs[si], (float)Q->dvh[si])) return 1;
                for (int sh = 0; sh < 2; sh++) {
                    for (int k = 0; k < m; k++) { const int q = Q->hS[(size_t)w * Q->IM + k]; int z = q / (Q->PG * Q->PG) + 4 * sh; z = z >= Q->PG ? Q->PG - 1 : z;
                        Q->dch_h[3 * k] = (float)z; Q->dch_h[3 * k + 1] = (float)((q / Q->PG) % Q->PG); Q->dch_h[3 * k + 2] = (float)(q % Q->PG); }
                    xh2d(Q->dcrd, Q->dch_h, (size_t)m * 12);
                    if (dv_net_points(Q->dv, Q->dcrd, nullptr, nullptr, m, Q->dacc)) return 1;
                    xd2h(Q->dacch, Q->dacc, (size_t)m * DC * 4);
                    for (int k = 0; k < m; k++) { const float *a = Q->dacch + (size_t)k * DC, *t = Q->dth + (size_t)(Q->off[w] + k) * nd, *t2 = Q->dth + (size_t)(Q->off[w] + (k + 7) % m) * nd;
                        double d = 0, d2 = 0; for (int e = 0; e < nd; e++) { d += (double)a[e] * t[e]; d2 += (double)a[e] * t2[e]; }
                        if (sh) csh += d; else { cm += d; cs += d2; n_++; } }
                }
            }
            if (n_) fprintf(stderr, "dist check step %d: dinovol cos at matched points across the two crops %.3f, +4 vx z %.3f, shuffled %.3f (%ld points)\n", step, cm / n_, csh / n_, cs / n_, n_);
        }
    }
    if (Q->wib > 0 && R > 0 && dvon && Q->dist <= 0) {   /* diag-dv: dinovol's tokens of each un-augmented teacher global at the mapped cells -> its iBOT head */
        const int G = Q->PG / 8, E = Q->DVH.Cin;
        for (int j = 0; j < 2 * N; j++) {
            const int w = j ^ 1; if (!Q->cnt[w]) continue;
            const int si = Q->smsi[j / 2];
            dr_crops(Q->src, Q->Ps, Q->xfd + j, Q->dvid, 1, Q->PG, Q->dvf); dr_f2u8(Q->dvf, (long long)g3, Q->dvu);
            nn_sync();   /* (dinovol runs on xnet's non-blocking stream) */
            if (dv_forward(Q->dv, Q->dvu, Q->PG, (float)Q->dvm[si], (float)Q->dvs[si], (float)Q->dvh[si])) return 1;
            xd2h(Q->dvtok, dv_tokens(Q->dv), (size_t)G * G * G * E * 2);   /* (synchronous on xnet's stream) */
            for (int k = 0; k < Q->cnt[w]; k++) {
                const int q = Q->hT[(size_t)w * Q->IM + k], qz = q / (Q->PG * Q->PG), qy = (q / Q->PG) % Q->PG, qx = q % Q->PG;
                const uint16_t *t = Q->dvtok + (((size_t)(qz / 8) * G + qy / 8) * G + qx / 8) * E; float *o = Q->dvx + (size_t)(Q->off[w] + k) * E;
                for (int e = 0; e < E; e++) { const uint16_t hb = t[e]; const uint32_t sg = (uint32_t)(hb >> 15) << 31; int ex = (hb >> 10) & 31; uint32_t mt = hb & 1023, bits;
                    if (ex == 0) { if (!mt) bits = sg; else { ex = 1; while (!(mt & 1024)) { mt <<= 1; ex--; } mt &= 1023; bits = sg | (uint32_t)(ex + 112) << 23 | mt << 13; } }
                    else if (ex == 31) bits = sg | 0x7f800000u | mt << 13; else bits = sg | (uint32_t)(ex + 112) << 23 | mt << 13;
                    memcpy(o + e, &bits, 4); }
            }
        }
        nn_h2d(Q->dvxd, Q->dvx, (size_t)R * E * 4);
        if (Q->icos) dr_unit_rows(Q->dvxd, R, E);   /* (the teacher's unit tokens) */
        if (!Q->icos) { if (dr_head_fwd(&Q->DVH, Q->dvP, Q->dvxd, R, Q->il, &Q->dvw)) return 1;
        if (Q->icm > 0) { dr_teacher_probs(Q->il, R, Q->KI, tt, 0, Q->icen, Q->iq, Q->isk); dr_center_update(Q->icen, Q->il, R, Q->KI, Q->icm); }
        else dr_teacher_probs(Q->il, R, Q->KI, tt, Q->skit, nullptr, Q->iq, Q->isk); }
    } else if (Q->wib > 0 && R > 0) {   /* iBOT targets: the teacher head over all masked rows, Sinkhorn over them */
        if (dr_head_fwd(&Q->I.H, Q->I.pt, Q->ieT, R, Q->il, &Q->I.w)) return 1;
        if (Q->icm > 0) { dr_teacher_probs(Q->il, R, Q->KI, tt, 0, Q->icen, Q->iq, Q->isk); dr_center_update(Q->icen, Q->il, R, Q->KI, Q->icm); }
        else dr_teacher_probs(Q->il, R, Q->KI, tt, Q->skit, nullptr, Q->iq, Q->isk);
    }
    Q->sums[5] = R;
    /* the batch-statistic bank (KoLeo, VICReg, z VICReg), chunk-major (v N + k): the teacher's head-input rows -- or, --ssl-exact-bank,
       the student's own from a no-grad pass with the live weights (GradCache: the full-batch statistics, this row's gradient exact) */
    for (int k = 0; k < N; k++) for (int v = 0; v < 2; v++) nn_d2d(Q->bank + ((size_t)v * N + k) * C, Q->clsT + (size_t)(2 * k + v) * C, (size_t)C * 4);
    if (Q->exact) {
        for (int j = 0; j < 2 * N; j++) {
            const int k = j / 2, v = j % 2; float *row = Q->bank + ((size_t)v * N + k) * C;
            unet_forward(Q->ug, Q->xg + (size_t)j * CIN * g3, Q->xsg, 0); dn_ssl_cls(Q, Q->ug, row);
            if (Q->cls_norm) dr_rownorm_fwd(row, 1, C, row, Q->rsS);
            if (Q->cstd) dr_cols_affine(row, C, Q->cmean, Q->cistd); else if (Q->ccen) nn_axpy(row, -1.f, Q->cmean, (size_t)C);
            if (Q->zvb) { if (dr_head_fwd(&Q->H.H, Q->H.p, row, 1, Q->sl, &Q->H.w)) return 1; nn_d2h(Q->zvb + ((size_t)v * N + k) * Q->H.H.B, Q->H.w.zn, (size_t)Q->H.H.B * 4); }
        }
    }
    {   /* per-chunk statistics of the bank (head input; unit bottleneck): mean, std, covariance -- the full batch's, each row's gradient
           exact against them (src/sslstat.h; tests/test_sslstat.c); with the exact bank also every row's full-batch KoLeo gradient */
        const int needx = Q->vv > 0 || Q->vc > 0 || (Q->exact && Q->wk > 0);
        if (needx) nn_d2h(Q->vx ? Q->vx : (Q->vx = malloc((size_t)2 * N * C * 4)), Q->bank, (size_t)2 * N * C * 4);
        for (int v = 0; v < 2; v++) {
            if (Q->vv > 0 || Q->vc > 0) {
                ss_stats(Q->vx + (size_t)v * N * C, N, C, 1e-4, Q->vmu + (size_t)v * C, Q->vsd + (size_t)v * C, Q->vc > 0 ? Q->vcov + (size_t)v * C * C : nullptr);
                Q->st[5] += ss_vic_loss(Q->vsd + (size_t)v * C, Q->vc > 0 ? Q->vcov + (size_t)v * C * C : nullptr, C, 1.0, Q->vv, Q->vc) / 2;
                double ms = 0; for (int c = 0; c < C; c++) ms += Q->vsd[(size_t)v * C + c] / C; Q->st[7] += ms / 2;
            }
            if (Q->zvb) { const int Bz = Q->H.H.B;
                ss_stats(Q->zvb + (size_t)v * N * Bz, N, Bz, 1e-8, Q->zmu + (size_t)v * Bz, Q->zsd + (size_t)v * Bz, Q->zcv > 0 ? Q->zcov + (size_t)v * Bz * Bz : nullptr);
                Q->st[11] += ss_vic_loss(Q->zsd + (size_t)v * Bz, nullptr, Bz, 1.0 / sqrt((double)Bz), Q->zv, 0) / 2; }
            if (Q->exact && Q->wk > 0) Q->st[9] += ss_koleo_all(Q->vx + (size_t)v * N * C, N, C, Q->wk, 1e-8, Q->kg + (size_t)v * N * C) / 2;
        }
    }
    /* student: every view forward -> loss over the batch's targets -> backward */
    unet_zero_grad(Q->ug); unet_zero_grad(Q->ul); nn_zero(Q->outs, 16 * 4);
    const float wpair = Q->wdino / (float)(N * (2 + 2 * Q->NL));
    const double vr = Q->vramp > 0 ? fmin(1.0, (double)step / Q->vramp) : 1.0;   /* (the VICReg ramp) */
    for (int j = 0; j < NV; j++) {
        const int gl = j < 2 * N, i = gl ? j / 2 : (j - 2 * N) / Q->NL, v = gl ? j % 2 : -1; unet *n = gl ? Q->ug : Q->ul;
        const float *x = gl ? Q->xg + (size_t)j * CIN * g3 : Q->xl + (size_t)(j - 2 * N) * CIN * l3;
        const int ib = gl && Q->wib > 0 && Q->cnt[j] > 0;
        if (ib && !nomask && Q->subps > 0) {   /* fine masking: random sub-cells (ratio subp0..subp1 per cell) of the masked cells */
            const int G = Q->GT_, G2 = Q->PG / Q->subps, f = Q->TPS / Q->subps; const uint8_t *cm = Q->cellm + (size_t)j * G * G * G;
            memset(Q->subm, 0, (size_t)G2 * G2 * G2);
            for (int c = 0; c < G * G * G; c++) { if (!cm[c]) continue; const int cz = c / (G * G), cy = c / G % G, cx = c % G; const double r = Q->subp0 + (Q->subp1 - Q->subp0) * rng_u(&Q->ar);
                for (int a = 0; a < f * f * f; a++) if (rng_u(&Q->ar) < r) Q->subm[((size_t)(cz * f + a / (f * f)) * G2 + cy * f + a / f % f) * G2 + cx * f + a % f] = 1; }
            nn_h2d(Q->d_subm, Q->subm, (size_t)G2 * G2 * G2); dr_mask_cells((float *)x, Q->PG, Q->subps, Q->d_subm, 0.f);
            if (CIN == 2) dr_mask_cells((float *)x + g3, Q->PG, Q->subps, Q->d_subm, 1.f);
        } else if (ib && !nomask) { nn_h2d(Q->d_mask, Q->cellm + (size_t)j * Q->GT_ * Q->GT_ * Q->GT_, (size_t)Q->GT_ * Q->GT_ * Q->GT_); dr_mask_cells((float *)x, Q->PG, Q->TPS, Q->d_mask, 0.f);
                  if (CIN == 2) dr_mask_cells((float *)x + g3, Q->PG, Q->TPS, Q->d_mask, 1.f); }   /* (the teacher saw it unmasked; the mask channel marks the cells) */
        const float *lgs = unet_forward(n, x, gl ? Q->xsg : Q->xsl, 1);
        if (ib) {   /* iBOT: the masked cells' student features -> head -> cross-entropy vs the cross-view targets -> d features into the fp32 logit gradient */
            const int m = Q->cnt[j];
            const int h16 = unet_logits_h16(n);
            if (Q->itok) { dr_rows_gather(unet_tokens(n, nullptr, nullptr, nullptr, nullptr), CT, Q->d_tS + (size_t)j * Q->IM, m, Q->ieS); dr_rownorm_fwd(Q->ieS, m, CT, Q->ieS, Q->irsS); }
            else dn_gather((const char *)lgs + (size_t)Q->ich * g3 * (h16 ? 2 : 4), h16, g3, 32, Q->d_hS + (size_t)j * Q->IM, m, Q->ieS, Q->inS);
            if (dr_head_fwd(&Q->I.H, Q->I.p, Q->ieS, m, Q->isl, &Q->I.w)) return 1;
            Q->I.w.dzn_add = nullptr;
            if ((Q->izv > 0 || Q->sig > 0) && m >= 4) {   /* variance hinge / SIGReg on this view's unit token bottleneck rows (gradient added to d / d zn) */
                const int Bz = Q->I.H.B; float *zh = malloc((size_t)m * Bz * 4), *gh = calloc((size_t)m * Bz, 4); nn_d2h(zh, Q->I.w.zn, (size_t)m * Bz * 4);
                if (Q->sig > 0) {   /* SIGReg: sqrt(B) zn (an isotropic unit-norm cloud has unit variance per direction) vs N(0, 1) on sigS fresh random directions */
                    if (!Q->sigA) Q->sigA = malloc((size_t)Q->sigS * Bz * 4);
                    for (int a = 0; a < Q->sigS; a++) { double nr = 0; for (int c = 0; c < Bz; c++) { const double u1 = rng_u(&Q->ar) + 1e-12, u2 = rng_u(&Q->ar); const double g = sqrt(-2 * log(u1)) * cos(6.283185307179586 * u2); Q->sigA[(size_t)a * Bz + c] = (float)g; nr += g * g; }
                        for (int c = 0; c < Bz; c++) Q->sigA[(size_t)a * Bz + c] /= (float)sqrt(nr); }
                    float *gs = malloc((size_t)m * Bz * 4);
                    Q->sig_l += ss_sigreg_all(zh, m, Bz, Q->sigA, Q->sigS, 17, 3.0, sqrt((double)Bz), Q->sig * Q->wib / (double)(R > 0 ? R : 1), gs) / (Q->wib / (double)(R > 0 ? R : 1)); Q->sig_n++;
                    for (size_t k = 0; k < (size_t)m * Bz; k++) gh[k] += gs[k];
                    free(gs);
                }
                const double gam = 1.0 / sqrt((double)Bz), wv = Q->izv * Q->wib * m / (double)(R > 0 ? R : 1);   /* (this view's share of the batch) */
                double hl = 0;
                for (int d = 0; d < Bz && Q->izv > 0; d++) { double mu = 0, v = 0; for (int k = 0; k < m; k++) mu += zh[(size_t)k * Bz + d] / m; for (int k = 0; k < m; k++) v += (zh[(size_t)k * Bz + d] - mu) * (zh[(size_t)k * Bz + d] - mu) / m;
                    const double sd = sqrt(v + 1e-8); if (sd >= gam) continue; hl += (gam - sd) / Bz;
                    for (int k = 0; k < m; k++) gh[(size_t)k * Bz + d] = (float)(-wv / Bz * (zh[(size_t)k * Bz + d] - mu) / (m * sd)); }
                nn_h2d(Q->zvg, gh, (size_t)m * Bz * 4); Q->I.w.dzn_add = Q->zvg; if (Q->izv > 0) { Q->zvh_l += hl; Q->zvh_n++; } free(zh); free(gh);
            }
            float *hw_ = malloc((size_t)m * 4); for (int k = 0; k < m; k++) hw_[k] = Q->wib / (float)(R > 0 ? R : 1); nn_h2d(Q->iw, hw_, (size_t)m * 4); free(hw_);
            Q->lastm = m;
            if (Q->icos && dvon) {   /* 1 - cos(student unit bottleneck, dinovol's unit token): d / d zn = -w t */
                const int E = Q->DVH.Cin; const float *t = Q->dvxd + (size_t)Q->off[j] * E;
                nn_d2d(Q->I.w.dzn, t, (size_t)m * E * 4); nn_scale(Q->I.w.dzn, -Q->wib / (float)(R > 0 ? R : 1), (size_t)m * E);
                { float *zh = malloc((size_t)m * E * 4), *th = malloc((size_t)m * E * 4); nn_d2h(zh, Q->I.w.zn, (size_t)m * E * 4); nn_d2h(th, t, (size_t)m * E * 4);
                  for (int k = 0; k < m; k++) { double d = 0; for (int e = 0; e < E; e++) d += (double)zh[(size_t)k * E + e] * th[(size_t)k * E + e]; Q->cosacc += d; Q->cosn++; } free(zh); free(th); }
                if (Q->I.w.dzn_add) nn_axpy(Q->I.w.dzn, 1.f, Q->I.w.dzn_add, (size_t)m * E);
                const float *dza = Q->I.w.dzn_add; Q->I.w.dzn_add = nullptr;
                if (dr_head_bwd(&Q->I.H, Q->I.p, Q->I.g, Q->ieS, nullptr, m, Q->idE, 0, &Q->I.w)) return 1;
                (void)dza;
            } else {
            const float esc = Q->ilsc ? expf(Q->lsc) : 1.f;
            if (Q->ilsc) nn_scale(Q->isl, esc, (size_t)m * Q->KI);
            dr_ce(Q->isl, m, Q->KI, Q->ts, Q->iq + (size_t)Q->off[j] * Q->KI, Q->iid, Q->iid, Q->iw, m, Q->idsl, Q->outs + 6, Q->ice); nn_axpy(Q->outs + 8, 1.f, Q->outs + 6, 1);
            if (Q->ilsc) {   /* d L / d lsc = sum dl' l' (the scaled logits); the head sees d / d unscaled = esc dl' */
                const size_t nn_ = (size_t)m * Q->KI; float *a = malloc(nn_ * 4), *b = malloc(nn_ * 4); nn_d2h(a, Q->isl, nn_ * 4); nn_d2h(b, Q->idsl, nn_ * 4);
                double gsum = 0; for (size_t i = 0; i < nn_; i++) gsum += (double)a[i] * b[i]; Q->lsg += gsum; free(a); free(b);
                nn_scale(Q->idsl, esc, nn_);
            }
            if (dr_head_bwd(&Q->I.H, Q->I.p, Q->I.g, Q->ieS, Q->idsl, m, Q->idE, step > Q->freeze && !(Q->dv && Q->dist <= 0), &Q->I.w)) return 1;
            }
            if (Q->itok) dr_rownorm_bwd(Q->idE, Q->ieS, Q->irsS, m, CT, Q->idE);
            if (!Q->itok) dr_scatter_unit_f32(Q->ieS, Q->idE, Q->inS, Q->d_hS + (size_t)j * Q->IM, m, 32, g3, 1.f, Q->glzib + (size_t)Q->ich * g3);
            if (Q->dist > 0 && Q->dstr <= 0 && dvon) {   /* distillation: L = dist / R sum_k (1 - e_k . t_k), e = unit(head channels dch..) at the cell centre; d / d e = -dist / R t */
                const int nd = Q->dnd; float *eh = malloc((size_t)m * nd * 4), *gh = malloc((size_t)m * nd * 4); const float *th = Q->dth + (size_t)Q->off[j] * nd;
                dn_gather((const char *)lgs + (size_t)Q->dch * g3 * (h16 ? 2 : 4), h16, g3, nd, Q->d_hS + (size_t)j * Q->IM, m, Q->dse, Q->dsn);
                nn_d2h(eh, Q->dse, (size_t)m * nd * 4);
                const float wv = Q->dist / (float)(R > 0 ? R : 1);
                for (int k = 0; k < m; k++) { double d = 0; for (int e = 0; e < nd; e++) { d += (double)eh[(size_t)k * nd + e] * th[(size_t)k * nd + e]; gh[(size_t)k * nd + e] = -wv * th[(size_t)k * nd + e]; }
                    Q->dist_l += 1 - d; Q->dist_n++; }
                if (Q->dabs != 1.f) for (size_t i = 0; i < (size_t)m * nd; i++) gh[i] *= Q->dabs;
                if (Q->drel > 0) { const int mr = m < 256 ? m : 256;   /* (the rows are a random subset of the view's cells) */
                    Q->rel_l += ss_relsim(eh, th, mr, nd, 1.0, nullptr); Q->rel_n++; ss_relsim(eh, th, mr, nd, Q->dist * Q->drel * (double)m / (R > 0 ? R : 1), gh); }
                nn_h2d(Q->dge, gh, (size_t)m * nd * 4);
                dr_scatter_unit_f32(Q->dse, Q->dge, Q->dsn, Q->d_hS + (size_t)j * Q->IM, m, nd, g3, 1.f, Q->glzib + (size_t)Q->dch * g3);
                free(eh); free(gh);
            }
            if (dbg && j == 0) { float *h = malloc((size_t)m * Q->ICIN * 4), *nr = malloc((size_t)m * 4); nn_d2h(h, Q->idE, (size_t)m * Q->ICIN * 4); nn_d2h(nr, Q->inS, (size_t)m * 4);
                double a = 0, nm = 0; for (int k = 0; k < m * Q->ICIN; k++) a += (double)h[k] * h[k]; for (int k = 0; k < m; k++) nm += nr[k] / m;
                float *gz = malloc(g3 * 32 * 4); nn_d2h(gz, Q->glzib + (size_t)Q->ich * g3, g3 * 32 * 4); double b = 0; long nz = 0; for (size_t k = 0; k < g3 * 32; k++) { b += (double)gz[k] * gz[k]; nz += gz[k] != 0; }
                fprintf(stderr, "dbg step %d view 0: %d rows, |dL/d unit feat| %.3e, feature norm mean %.3e, |glz| %.3e (%ld nonzero)\n", step, m, sqrt(a), nm, sqrt(b), nz); free(h); free(nr); free(gz); }
        }
        int Sv = 0, nreg = 0; const float *tok = unet_tokens(n, nullptr, &Sv, nullptr, &nreg); (void)tok;
        dn_ssl_cls(Q, n, Q->clsS);
        if (Q->cls_norm) dr_rownorm_fwd(Q->clsS, 1, C, Q->clsS, Q->rsS);
        float *cpre = Q->cpre;   /* (the normed row before the batch transform, for the row-norm backward) */
        if (Q->cstd) { nn_d2d(cpre, Q->clsS, (size_t)C * 4); dr_cols_affine(Q->clsS, C, Q->cmean, Q->cistd); }
        else if (Q->ccen) nn_axpy(Q->clsS, -1.f, Q->cmean, (size_t)C);
        if (dr_head_fwd(&Q->H.H, Q->H.p, Q->clsS, 1, Q->sl, &Q->H.w)) return 1;
        int ps_[2] = {0, 0}, pt_[2] = {gl ? 2 * i + 1 - v : 2 * i, 2 * i + 1}; float pw_[2] = {wpair, wpair}; const int npair = gl ? 1 : 2;
        nn_h2d(Q->pS, ps_, 8); nn_h2d(Q->pT, pt_, 8); nn_h2d(Q->pW, pw_, 8);
        nn_zero(Q->dcls, (size_t)C * 4);
        Q->H.w.dzn_add = nullptr;
        if (gl && Q->zvb) {   /* VICReg on the unit bottleneck: this row's gradient of the full chunk's loss (std target 1 / sqrt(B)) */
            const int Bz = Q->H.H.B; float *gz = malloc((size_t)Bz * 4); nn_d2h(Q->zvh, Q->H.w.zn, (size_t)Bz * 4);
            ss_vic_row_grad(Q->zvh, N, Bz, Q->zmu + (size_t)v * Bz, Q->zsd + (size_t)v * Bz, Q->zcv > 0 ? Q->zcov + (size_t)v * Bz * Bz : nullptr, 1.0 / sqrt((double)Bz), Q->zv * vr, Q->zcv * vr, gz);
            nn_h2d(Q->zvd, gz, (size_t)Bz * 4); Q->H.w.dzn_add = Q->zvd; free(gz);
            if (!Q->exact) memcpy(Q->zvb + ((size_t)v * N + i) * Bz, Q->zvh, (size_t)Bz * 4);
        }
        if (Q->simg > 0) {   /* SimDINO: d / d zn = sum over pairs w (zs - zt); global rows: - G d R / d zs over the chunk (the other samples' current z) */
            const int Bz = Q->H.H.B; float *zs = Q->zcur; nn_d2h(zs, Q->H.w.zn, (size_t)Bz * 4);
            for (int c = 0; c < Bz; c++) Q->dzh[c] = 0;
            double dl = 0;
            float *zth = malloc((size_t)npair * Bz * 4); for (int pp = 0; pp < npair; pp++) nn_d2h(zth + (size_t)pp * Bz, Q->zT + (size_t)pt_[pp] * Bz, (size_t)Bz * 4);
            for (int pp = 0; pp < npair; pp++) for (int c = 0; c < Bz; c++) { const double d = zs[c] - zth[(size_t)pp * Bz + c]; Q->dzh[c] += (float)(wpair * d); dl += wpair * 0.5 * d * d; }
            free(zth);
            if (gl) {   /* coding rate over this chunk: rows 2k + v of the bank (student z where computed this step, teacher otherwise), this row = zs */
                const int Bc = N; double *A = Q->zA; const double al = Q->sima / Bc;
                for (int c = 0; c < Bz * Bz; c++) A[c] = 0;
                for (int c = 0; c < Bz; c++) A[(size_t)c * Bz + c] = 1;
                for (int k = 0; k < N; k++) { const float *zr = k == i ? zs : Q->zbank + (size_t)(2 * k + v) * Bz;
                    for (int a = 0; a < Bz; a++) { const double za = zr[a] * al; if (za == 0) continue; double *Ar = A + (size_t)a * Bz; for (int b2 = 0; b2 <= a; b2++) Ar[b2] += za * zr[b2]; } }
                /* Cholesky of A (lower), then g = A^-1 zs: d R / d zs = al A^-1 zs  (R = 1/2 logdet A) */
                int okc = 1; for (int a = 0; a < Bz && okc; a++) { double *Ar = A + (size_t)a * Bz;
                    for (int b2 = 0; b2 <= a; b2++) { double sm = Ar[b2]; const double *Br = A + (size_t)b2 * Bz; for (int k2 = 0; k2 < b2; k2++) sm -= Ar[k2] * Br[k2];
                        if (b2 == a) { if (sm <= 0) { okc = 0; break; } Ar[a] = sqrt(sm); } else Ar[b2] = sm / Br[b2]; } }
                if (okc) { double *y = malloc((size_t)Bz * 8);
                    for (int a = 0; a < Bz; a++) { double sm = zs[a]; for (int k2 = 0; k2 < a; k2++) sm -= A[(size_t)a * Bz + k2] * y[k2]; y[a] = sm / A[(size_t)a * Bz + a]; }
                    for (int a = Bz - 1; a >= 0; a--) { double sm = y[a]; for (int k2 = a + 1; k2 < Bz; k2++) sm -= A[(size_t)k2 * Bz + a] * y[k2]; y[a] = sm / A[(size_t)a * Bz + a]; }
                    for (int c = 0; c < Bz; c++) Q->dzh[c] -= (float)(Q->simg / (2 * N) * al * y[c]);   /* (each global row's share; 2 chunks) */
                    double ld = 0; for (int a = 0; a < Bz; a++) ld += log(A[(size_t)a * Bz + a]); Q->st[9] += ld / (2 * N);   /* R = 1/2 logdet = sum log diag(L) */
                    free(y); }
                memcpy(Q->zbank + (size_t)(2 * i + v) * Bz, zs, (size_t)Bz * 4);
            }
            Q->st[10] += dl; nn_h2d(Q->H.w.dzn, Q->dzh, (size_t)Bz * 4);
            if (dr_head_bwd(&Q->H.H, Q->H.p, Q->H.g, Q->clsS, nullptr, 1, Q->dcls, 0, &Q->H.w)) return 1;
        } else {
        dr_ce(Q->sl, 1, K, Q->ts, Q->tq, Q->pS, Q->pT, Q->pW, npair, Q->dsl, Q->outs, Q->ces); nn_axpy(Q->outs + 4, 1.f, Q->outs, 1);   /* (dr_ce writes its loss: summed here) */
        if (dr_head_bwd(&Q->H.H, Q->H.p, Q->H.g, Q->clsS, Q->dsl, 1, Q->dcls, step > Q->freeze, &Q->H.w)) return 1;
        }
        if (gl && Q->wk > 0 && Q->exact) { nn_h2d(Q->kgd, Q->kg + ((size_t)v * N + i) * C, (size_t)C * 4); nn_axpy(Q->dcls, 1.f, Q->kgd, (size_t)C); }   /* (the full batch's KoLeo gradient of this row) */
        else if (gl && Q->wk > 0) {   /* KoLeo against the other samples of this view chunk (the bank: teacher rows / the student's as computed) */
            const float *bv = Q->bank + (size_t)v * N * C;   /* (chunk v without this sample: two contiguous copies) */
            if (i > 0) nn_d2d(Q->bex, bv, (size_t)i * C * 4);
            if (i < N - 1) nn_d2d(Q->bex + (size_t)i * C, bv + (size_t)(i + 1) * C, (size_t)(N - 1 - i) * C * 4);
            dr_koleo_q(Q->clsS, 1, Q->bex, N - 1, C, Q->wk / (float)N, Q->dcls, Q->outs + 1);
            if (!Q->exact) nn_d2d(Q->bank + ((size_t)v * N + i) * C, Q->clsS, (size_t)C * 4);
        }
        if (gl && (Q->vv > 0 || Q->vc > 0)) {   /* VICReg on the head input: this row's gradient of the full chunk's loss (src/sslstat.h) */
            static float *xh; static int xhn; if (xhn < C) { free(xh); xh = malloc((size_t)C * 4); xhn = C; }
            nn_d2h(xh, Q->clsS, (size_t)C * 4);
            ss_vic_row_grad(xh, N, C, Q->vmu + (size_t)v * C, Q->vsd + (size_t)v * C, Q->vc > 0 ? Q->vcov + (size_t)v * C * C : nullptr, 1.0, Q->vv * vr, Q->vc * vr, Q->vg);
            nn_h2d(Q->vgd, Q->vg, (size_t)C * 4); nn_axpy(Q->dcls, 1.f, Q->vgd, (size_t)C);
        }
        if (Q->cstd) { dr_cols_scale(Q->dcls, C, Q->cistd); nn_d2d(Q->clsS, cpre, (size_t)C * 4); }   /* (through the batch transform; the normed row back) */
        else if (Q->ccen) nn_axpy(Q->clsS, 1.f, Q->cmean, (size_t)C);   /* (back to the normed row for its backward) */
        if (Q->cls_norm) dr_rownorm_bwd(Q->dcls, Q->clsS, Q->rsS, 1, C, Q->dcls);
        if (Q->tgn < (size_t)Sv * CT) { nn_free(Q->tgrad); Q->tgn = (size_t)Sv * CT; Q->tgrad = nn_malloc(Q->tgn * 4); }
        nn_zero(Q->tgrad, (size_t)Sv * CT * 4);
        if (Q->cls_reg) nn_axpy(Q->tgrad, gs, Q->dcls, (size_t)CT);   /* (the register row) */
        if (Q->cls_reg != 1) { float *dp = Q->dcls + (Q->cls_reg == 2 ? CT : 0); dn_axpby(dp, dp, (size_t)CT, gs, 0.f); dr_pool_bwd(dp, 1, Sv, CT, nreg, Q->tgrad); }
        if (ib && Q->itok) dr_rows_scatter_add(Q->idE, Q->d_tS + (size_t)j * Q->IM, Q->cnt[j], CT, gs, Q->tgrad);   /* (iBOT on the tokens: the scaled token-gradient domain) */
        const int dden = gl && Q->dist > 0 && Q->dstr > 0 && dvon;
        if (dden) {   /* dense distillation: L = dist / (2N NP) sum (1 - e . t) over every global's grid points (+ the relational term on a 256-point subset) */
            const int nd = Q->dnd, NP = Q->dnp, h16 = unet_logits_h16(n); float *eh = malloc((size_t)NP * nd * 4), *gh = malloc((size_t)NP * nd * 4); const float *th = Q->dtd + (size_t)j * NP * nd;
            static int dconst = -1; if (dconst < 0) dconst = getenv("UFSM_DIST_CONST") ? atoi(getenv("UFSM_DIST_CONST")) : 0;
            if (dconst) {   /* (debug: a trivial target -- 1: one constant unit vector; 2: the sign of z - PG/2 on axis 0) */
                float *tc = (float *)th;
                if (dconst == 3) {   /* 3: a trivially local target: unit (CT, its central-difference gradient x 4, 0.3) of the student's own input view */
                    const int PG = Q->PG; static float *xv; if (!xv) xv = malloc(g3 * 4); nn_d2h(xv, x, g3 * 4);
                    for (int k = 0; k < NP; k++) { const int q = Q->ddph[(size_t)j * NP + k], z = q / (PG * PG), y = q / PG % PG, x_ = q % PG;
                        const int zm = z > 0 ? z - 1 : z, zp = z < PG - 1 ? z + 1 : z, ym = y > 0 ? y - 1 : y, yp = y < PG - 1 ? y + 1 : y, xm = x_ > 0 ? x_ - 1 : x_, xp = x_ < PG - 1 ? x_ + 1 : x_;
                        double v[5] = {xv[q], 4.0 * (xv[((size_t)zp * PG + y) * PG + x_] - xv[((size_t)zm * PG + y) * PG + x_]), 4.0 * (xv[((size_t)z * PG + yp) * PG + x_] - xv[((size_t)z * PG + ym) * PG + x_]),
                                       4.0 * (xv[((size_t)z * PG + y) * PG + xp] - xv[((size_t)z * PG + y) * PG + xm]), 0.3}, n2 = 0;
                        for (int e = 0; e < 5; e++) n2 += v[e] * v[e];
                        for (int e = 0; e < nd; e++) tc[(size_t)k * nd + e] = e < 5 ? (float)(v[e] / sqrt(n2)) : 0.f; }
                } else for (int k = 0; k < NP; k++) { const int z = Q->ddph[(size_t)j * NP + k] / (Q->PG * Q->PG); for (int e = 0; e < nd; e++) tc[(size_t)k * nd + e] = e == 0 ? (dconst == 2 && z < Q->PG / 2 ? -1.f : 1.f) : 0.f; } }
            dn_gather((const char *)lgs + (size_t)Q->dch * g3 * (h16 ? 2 : 4), h16, g3, nd, Q->ddp + (size_t)j * NP, NP, Q->dse, Q->dsn);
            nn_d2h(eh, Q->dse, (size_t)NP * nd * 4);
            { float *nh = malloc((size_t)NP * 4); nn_d2h(nh, Q->dsn, (size_t)NP * 4); double a = 0; for (int k = 0; k < NP; k++) a += nh[k]; free(nh);   /* (embedding |z| and the mean pairwise cosine of 64 points) */
              double c = 0; const int st = NP / 64; for (int u = 0; u < 64; u++) for (int v = u + 1; v < 64; v++) { double d = 0; for (int e = 0; e < nd; e++) d += (double)eh[(size_t)u * st * nd + e] * eh[(size_t)v * st * nd + e]; c += d; }
              Q->zn_s += a / NP; Q->zsim_s += c / (64 * 63 / 2); Q->zn_n++; }
            const float wv = Q->dist * Q->dabs / (float)(2 * N * NP);
            for (int k = 0; k < NP; k++) { double d = 0; for (int e = 0; e < nd; e++) { d += (double)eh[(size_t)k * nd + e] * th[(size_t)k * nd + e]; gh[(size_t)k * nd + e] = -wv * th[(size_t)k * nd + e]; }
                Q->dist_l += 1 - d; Q->dist_n++; }
            { double mt[64] = {0}, ms = 0; const int ne = nd < 64 ? nd : 64; for (int k = 0; k < NP; k++) for (int e = 0; e < ne; e++) mt[e] += th[(size_t)k * nd + e] / NP;   /* (|mean target|: the best constant prediction's cosine) */
              for (int e = 0; e < ne; e++) ms += mt[e] * mt[e]; Q->dtm += sqrt(ms); Q->dtmn++; }
            if (Q->drel > 0) {   /* (a random 256-point subset: rows k * stride) */
                const int mr = NP < 256 ? NP : 256, st_ = NP / mr; float *es = malloc((size_t)mr * nd * 4), *ts_ = malloc((size_t)mr * nd * 4), *gs2 = calloc((size_t)mr * nd, 4);
                const int o_ = (int)(rng_next(&Q->ar) % (uint64_t)st_);
                for (int k = 0; k < mr; k++) { memcpy(es + (size_t)k * nd, eh + (size_t)(k * st_ + o_) * nd, (size_t)nd * 4); memcpy(ts_ + (size_t)k * nd, th + (size_t)(k * st_ + o_) * nd, (size_t)nd * 4); }
                Q->rel_l += ss_relsim(es, ts_, mr, nd, 1.0, nullptr); Q->rel_n++; ss_relsim(es, ts_, mr, nd, Q->dist * Q->drel / (2.0 * N), gs2);
                for (int k = 0; k < mr; k++) for (int e = 0; e < nd; e++) gh[(size_t)(k * st_ + o_) * nd + e] += gs2[(size_t)k * nd + e];
                free(es); free(ts_); free(gs2); }
            if (Q->dnce > 0) {   /* teacher-graph SupCon on a random m-point subset: match which points dinovol groups together, not its codes */
                const int mr = NP < Q->dnm ? NP : Q->dnm; float *es = malloc((size_t)mr * nd * 4), *ts_ = malloc((size_t)mr * nd * 4), *gs2 = calloc((size_t)mr * nd, 4); int *pk = malloc((size_t)mr * 4);
                for (int k = 0; k < mr; k++) { pk[k] = (int)(rng_next(&Q->ar) % (uint64_t)NP); memcpy(es + (size_t)k * nd, eh + (size_t)pk[k] * nd, (size_t)nd * 4); memcpy(ts_ + (size_t)k * nd, th + (size_t)pk[k] * nd, (size_t)nd * 4); }
                double pf = 0; Q->nce_l += ss_tnce(es, ts_, mr, nd, Q->dthr, Q->dtau, 1.0, gs2, &pf); Q->nce_pf += pf; Q->nce_n++;
                const float sc = Q->dist * Q->dnce / (float)(2 * N);
                for (int k = 0; k < mr; k++) for (int e = 0; e < nd; e++) gh[(size_t)pk[k] * nd + e] += sc * gs2[(size_t)k * nd + e];
                free(es); free(ts_); free(gs2); free(pk); }
            if (Q->dcc > 0) {   /* pseudo sheet labels: union-find over the S-grid's 6-neighbour edges with dinovol cosine > thr; components of >= ccmin
                                   points become GT codes (256 + id: same component positive, else negative) for the GT SupCon kernel (dn_supcon) on m of their points */
                const int g_ = Q->PG / Q->dstr; int *par = malloc((size_t)NP * 4), *sz = calloc((size_t)NP, 4), *cid = malloc((size_t)NP * 4), *kp = malloc((size_t)NP * 4);
                for (int k = 0; k < NP; k++) par[k] = k;
                #define UF_FIND(x_) ({ int r_ = (x_); while (par[r_] != r_) { par[r_] = par[par[r_]]; r_ = par[r_]; } r_; })
                int *km = nullptr;
                if (Q->dkm > 0) {   /* spherical k-means (k-means++ seeds, 8 Lloyd iterations) on <= 4096 points, then every point to its nearest centre */
                    const int K = Q->dkm, ns = NP < 4096 ? NP : 4096, sst = NP / ns; float *cen = malloc((size_t)K * nd * 4), *dmin = malloc((size_t)ns * 4); double *acc = malloc((size_t)K * nd * 8); int *cnt = malloc((size_t)K * 4);
                    km = malloc((size_t)NP * 4);
                    #define KP(i_) (th + (size_t)(i_) * sst * nd)
                    memcpy(cen, KP(rng_next(&Q->ar) % (uint64_t)ns), (size_t)nd * 4);
                    for (int i = 0; i < ns; i++) dmin[i] = 1e30f;
                    for (int c = 1; c < K; c++) { double tot = 0;
                        for (int i = 0; i < ns; i++) { double d = 0; for (int e = 0; e < nd; e++) d += (double)KP(i)[e] * cen[(size_t)(c - 1) * nd + e]; const float dd = (float)(1 - d); if (dd < dmin[i]) dmin[i] = dd; tot += dmin[i] > 0 ? dmin[i] * dmin[i] : 0; }
                        double r = rng_u(&Q->ar) * tot; int pick = ns - 1; for (int i = 0; i < ns; i++) { r -= dmin[i] > 0 ? dmin[i] * dmin[i] : 0; if (r <= 0) { pick = i; break; } }
                        memcpy(cen + (size_t)c * nd, KP(pick), (size_t)nd * 4); }
                    for (int it = 0; it < 9; it++) {
                        const int last = it == 8, n_ = last ? NP : ns; if (!last) { memset(acc, 0, (size_t)K * nd * 8); memset(cnt, 0, (size_t)K * 4); }
                        for (int i = 0; i < n_; i++) { const float *t_ = last ? th + (size_t)i * nd : KP(i); int bc = 0; double bd = -2;
                            for (int c = 0; c < K; c++) { double d = 0; for (int e = 0; e < nd; e++) d += (double)t_[e] * cen[(size_t)c * nd + e]; if (d > bd) { bd = d; bc = c; } }
                            if (last) km[i] = bc; else { cnt[bc]++; for (int e = 0; e < nd; e++) acc[(size_t)bc * nd + e] += t_[e]; } }
                        if (!last) for (int c = 0; c < K; c++) { if (!cnt[c]) continue; double n2 = 0; for (int e = 0; e < nd; e++) n2 += acc[(size_t)c * nd + e] * acc[(size_t)c * nd + e]; n2 = sqrt(n2) + 1e-12;
                            for (int e = 0; e < nd; e++) cen[(size_t)c * nd + e] = (float)(acc[(size_t)c * nd + e] / n2); }
                    }
                    #undef KP
                    free(cen); free(dmin); free(acc); free(cnt);
                    if (!Q->dksplit) { int rep[256]; for (int c = 0; c < 256; c++) rep[c] = -1; for (int k = 0; k < NP; k++) { const int c = km[k] & 255; if (rep[c] < 0) rep[c] = k; par[k] = rep[c]; } }
                }
                if (Q->dkm <= 0 || Q->dksplit) for (int k = 0; k < NP; k++) { const int z = k / (g_ * g_), y = k / g_ % g_, x_ = k % g_; const int nb[3] = {z + 1 < g_ ? k + g_ * g_ : -1, y + 1 < g_ ? k + g_ : -1, x_ + 1 < g_ ? k + 1 : -1};
                    for (int q = 0; q < 3; q++) { if (nb[q] < 0) continue; int join;
                        if (km) join = km[k] == km[nb[q]];
                        else { double d = 0; for (int e = 0; e < nd; e++) d += (double)th[(size_t)k * nd + e] * th[(size_t)nb[q] * nd + e]; join = d > Q->dthr; }
                        if (join) { const int ra = UF_FIND(k), rb = UF_FIND(nb[q]); if (ra != rb) par[ra] = rb; } } }
                free(km);
                for (int k = 0; k < NP; k++) sz[UF_FIND(k)]++;
                int nk = 0, nc = 0; for (int k = 0; k < NP; k++) cid[k] = -1;
                for (int k = 0; k < NP; k++) { const int r_ = UF_FIND(k); if (sz[r_] < Q->dccmin) continue; if (cid[r_] < 0) cid[r_] = nc++; kp[nk++] = k; }
                #undef UF_FIND
                const int mr = nk < Q->dnm ? nk : Q->dnm;
                { int big = 0; for (int k = 0; k < NP; k++) if (par[k] == k && sz[k] > big) big = sz[k]; Q->cc_big += (double)big / NP; Q->cc_skip += !(mr >= 16 && nc >= 2); Q->cc_v++; Q->cc_keep += (double)nk / NP; Q->cc_nc += nc; }
                if (mr >= 16 && nc >= 2) {
                    for (int k = 0; k < mr; k++) { const int o_ = k + (int)(rng_next(&Q->ar) % (uint64_t)(nk - k)), t_ = kp[k]; kp[k] = kp[o_]; kp[o_] = t_; }   /* (m distinct points) */
                    static float *es_d, *ga_d, *scr_d, *out_d, *gah; static int *code_d, *xyz_d, *codeh, *xyzh, cap; float *esh = malloc((size_t)mr * nd * 4);
                    if (cap < mr) { nn_free(es_d); nn_free(ga_d); nn_free(scr_d); nn_free(code_d); nn_free(xyz_d); free(gah); free(codeh); free(xyzh); cap = mr;
                        es_d = nn_malloc((size_t)cap * nd * 4); ga_d = nn_malloc((size_t)cap * nd * 4); scr_d = nn_malloc(((size_t)2 * cap + 16) * 4); code_d = nn_malloc((size_t)cap * 4); xyz_d = nn_malloc((size_t)cap * 12);
                        gah = malloc((size_t)cap * nd * 4); codeh = malloc((size_t)cap * 4); xyzh = malloc((size_t)cap * 12); if (!out_d) out_d = nn_malloc(16); }
                    for (int k = 0; k < mr; k++) { const int q = kp[k], v_ = Q->ddph[(size_t)j * NP + q]; memcpy(esh + (size_t)k * nd, eh + (size_t)q * nd, (size_t)nd * 4);
                        codeh[k] = 256 + cid[UF_FIND2(par, q)]; xyzh[3 * k] = v_ / (Q->PG * Q->PG); xyzh[3 * k + 1] = v_ / Q->PG % Q->PG; xyzh[3 * k + 2] = v_ % Q->PG; }
                    nn_h2d(es_d, esh, (size_t)mr * nd * 4); nn_h2d(code_d, codeh, (size_t)mr * 4); nn_h2d(xyz_d, xyzh, (size_t)mr * 12);
                    dn_supcon(es_d, code_d, xyz_d, mr, nullptr, code_d, xyz_d, 0, nd, Q->dtau, 1.f, 0.f, ga_d, out_d, scr_d);
                    float oh[4]; nn_d2h(oh, out_d, 16); nn_d2h(gah, ga_d, (size_t)mr * nd * 4);
                    const float sc = Q->dist * Q->dcc / (float)(2 * N);
                    for (int k = 0; k < mr; k++) for (int e = 0; e < nd; e++) gh[(size_t)kp[k] * nd + e] += sc * gah[(size_t)k * nd + e];
                    long np_ = 0; for (int a = 0; a < mr; a++) for (int b = 0; b < mr; b++) np_ += a != b && codeh[a] == codeh[b];
                    Q->cc_l += oh[0]; Q->cc_pos += (double)np_ / ((double)mr * (mr - 1)); Q->cc_n++; free(esh);
                }
                free(par); free(sz); free(cid); free(kp);
            }
            nn_h2d(Q->dge, gh, (size_t)NP * nd * 4);
            dr_scatter_unit_f32(Q->dse, Q->dge, Q->dsn, Q->ddp + (size_t)j * NP, NP, nd, g3, 1.f, Q->glzib + (size_t)Q->dch * g3);
            free(eh); free(gh);
        }
        static int donly = -1; if (donly < 0) donly = getenv("UFSM_DIST_ONLY") && atoi(getenv("UFSM_DIST_ONLY"));
        if (donly) {   /* (debug: the distillation loss alone -- no token / CLS gradient, no iBOT logits; locals contribute nothing) */
            nn_zero(Q->tgrad, Q->tgn * 4);
            if (dden) { const int CO_ = unet_cfg_of(n)->cout, e_ = Q->dch + Q->dnd; if (Q->dch > 0) nn_zero(Q->glzib, (size_t)Q->dch * g3 * 4); if (e_ < CO_) nn_zero(Q->glzib + (size_t)e_ * g3, (size_t)(CO_ - e_) * g3 * 4); }
        }
        unet_set_token_grad(n, Q->tgrad);
        if (!donly || dden) unet_backward(n, ib || dden ? Q->glzib : gl ? Q->glzg : Q->glzl);   /* (debug UFSM_DIST_ONLY: views without distillation are skipped) */
        if (dbg && j == 0) { unet_debug_grads(n, "view 0"); const size_t np_ = unet_nparams(n); float *g = malloc(np_ * 4); nn_d2h(g, unet_grad_ptr(n), np_ * 4); double a = 0; for (size_t k = 0; k < np_; k++) a += (double)g[k] * g[k];
            fprintf(stderr, "dbg step %d view 0: trunk |grad| %.3e over %zu params\n", step, sqrt(a), np_); free(g); }
        unet_set_token_grad(n, nullptr);
        if (ib) nn_zero(Q->glzib + (size_t)Q->ich * g3, (size_t)32 * g3 * 4);
        if ((ib || dden) && Q->dist > 0) nn_zero(Q->glzib + (size_t)Q->dch * g3, (size_t)Q->dnd * g3 * 4);
    }
    const size_t np = unet_nparams(Q->u);
    nn_axpy(unet_grad_ptr(Q->u), 1.f, unet_grad_ptr(Q->ug), np); nn_axpy(unet_grad_ptr(Q->u), 1.f, unet_grad_ptr(Q->ul), np);
    /* the head: AdamW (upstream), the prototype directions after --ssl-freeze-last, g fixed; then its EMA teacher */
    dn_head *h = &Q->H; const dr_head *H = &h->H;
    const float hlr = Q->hwarm > 0 && step < Q->hwarm ? Q->hlr * (float)step / Q->hwarm : Q->hlr;   /* (--ssl-head-warmup: linear, as upstream's lr warmup) */
    dr_adamw(h->p, h->g, h->m, h->v, H->v, hlr, 0.9f, 0.999f, 1e-8f, Q->wd, step);
    if (step > Q->freeze) dr_adamw(h->p + H->v, h->g + H->v, h->m + H->v, h->v + H->v, H->g - H->v, Q->plr, 0.9f, 0.999f, 1e-8f, Q->wd, step);
    dr_ema(h->pt, h->p, H->n, mom); nn_zero(h->g, H->n * 4);
    if (Q->wib > 0) { dn_head *hi = &Q->I; const dr_head *HI = &hi->H;
        dr_adamw(hi->p, hi->g, hi->m, hi->v, HI->v, hlr, 0.9f, 0.999f, 1e-8f, Q->wd, step);
        if (step > Q->freeze && !(Q->dv && Q->dist <= 0)) dr_adamw(hi->p + HI->v, hi->g + HI->v, hi->m + HI->v, hi->v + HI->v, HI->g - HI->v, Q->plr, 0.9f, 0.999f, 1e-8f, Q->wd, step);
        if (step > Q->freeze && !(Q->dv && Q->dist <= 0) && Q->pcen) dr_center_rows(hi->p + HI->v, HI->K, HI->B);
        if (Q->ilsc) {   /* the logit scale: Adam on one scalar */
            const double g_ = Q->lsg; Q->lsg = 0; Q->lsm = 0.9f * Q->lsm + 0.1f * (float)g_; Q->lsv = 0.999f * Q->lsv + 0.001f * (float)(g_ * g_);
            const float mh = Q->lsm / (1.f - powf(0.9f, (float)step)), vh = Q->lsv / (1.f - powf(0.999f, (float)step)); Q->lsc -= 1e-2f * mh / (sqrtf(vh) + 1e-8f);
            if (Q->lsc > 3.f) Q->lsc = 3.f;
            if (Q->lsc < -3.f) Q->lsc = -3.f;
        }
        dr_ema(hi->pt, hi->p, HI->n, mom); nn_zero(hi->g, HI->n * 4); }
    nn_d2h(Q->hout, Q->outs, 16 * 4); Q->sums[0] += Q->wdino > 0 ? Q->hout[4] / Q->wdino : 0; Q->sums[1] += Q->hout[1]; Q->sums[3] += Q->wib > 0 ? Q->hout[8] / Q->wib : 0; Q->sums[4] += R; Q->nsum++;
    return 0;
}
/* the SSL part of the log line: loss means, the teacher target entropy (mean over the batch's rows) and the prototype usage */
static void dn_ssl_log(dn_ssl *Q, int step, char *line, size_t cap) {
    const int R = 2 * Q->N, K = Q->K; float *q = malloc((size_t)R * K * 4); nn_d2h(q, Q->tq, (size_t)R * K * 4);
    double H = 0, Hm = 0, pmax = 0; double *mrow = calloc((size_t)K, 8);
    for (int r = 0; r < R; r++) { double z = 0, h = 0, mx = 0; for (int k = 0; k < K; k++) z += q[(size_t)r * K + k]; for (int k = 0; k < K; k++) { const double a = q[(size_t)r * K + k] / z; mrow[k] += a / R; if (a > 0) h -= a * log(a); if (a > mx) mx = a; } H += h / R; pmax += mx / R; }
    for (int k = 0; k < K; k++) if (mrow[k] > 0) Hm -= mrow[k] * log(mrow[k]);
    int used = 0; for (int k = 0; k < K; k++) used += Q->use[k];
    if (Q->ure > 0 && step % Q->ure < 100) memset(Q->use, 0, (size_t)K);
    const size_t ln = strlen(line);
    snprintf(line + ln, cap - ln, " | ssl: dino %.4f koleo %.4f H(t) %.2f H(mean) %.2f (max %.2f) pmax %.4f agree %.3f protos used %d / %d xscan %.2f (ncc %.3f)", Q->nsum ? Q->sums[0] / Q->nsum : 0.0, Q->nsum ? Q->sums[1] / Q->nsum : 0.0, H, Hm, log((double)K), pmax, Q->nsum ? Q->sums[2] / Q->nsum : 0.0, used, K, Q->nsum ? Q->sums[7] / Q->nsum : 0.0, Q->xnccn ? Q->xnccs / Q->xnccn : 0.0); Q->xnccs = 0; Q->xnccn = 0;
    {   /* does the CLS carry the sample? cosine of a sample's two teacher globals vs of different samples (teacher rows, after the norm) */
        const int C = Q->CC; float *ct = malloc((size_t)R * C * 4); nn_d2h(ct, Q->clsT, (size_t)R * C * 4);
        double cs = 0, cd = 0; long nd = 0;
        for (int a = 0; a < R; a++) for (int b = a + 1; b < R; b++) {
            double x = 0, ya = 0, yb = 0; for (int c = 0; c < C; c++) { x += (double)ct[(size_t)a * C + c] * ct[(size_t)b * C + c]; ya += (double)ct[(size_t)a * C + c] * ct[(size_t)a * C + c]; yb += (double)ct[(size_t)b * C + c] * ct[(size_t)b * C + c]; }
            const double cc = x / sqrt(ya * yb + 1e-30); if (a / 2 == b / 2) cs += cc; else { cd += cc; nd++; } }
        free(ct); const size_t l3_ = strlen(line);
        const double nn_ = Q->nsum ? Q->nsum : 1; { const size_t l4 = strlen(line); snprintf(line + l4, cap - l4, " | vic var %.4f cov %.4f cls std/dim %.4f | z std/dim mean %.4f min %.4f median %.4f (1/sqrt(B) %.4f) zvic %.4f | simdino dist %.4f R %.3f", Q->st[5] / nn_, Q->st[6] / nn_, Q->st[7] / nn_, Q->st[8] / nn_, Q->zsmin, Q->zsmed, 1.0 / sqrt((double)Q->H.H.B), Q->st[11] / nn_, Q->st[10] / nn_, Q->st[9] / nn_); }
        snprintf(line + l3_, cap - l3_, " | avg cls same %.3f other %.3f z same %.3f other %.3f agree-other %.3f | iou %.3f | last cls same %.3f other %.3f | z cos other %.3f | logit sd raw %.4f centred %.4f (/tt) | centre norm %.3f | proto |cos| %.3f",
                 Q->st[0] / nn_, Q->st[1] / nn_, Q->st[2] / nn_, Q->st[3] / nn_, Q->st[4] / nn_, Q->nsum ? Q->sums[6] / Q->nsum : 0.0, cs / (R / 2), nd ? cd / nd : 0.0, Q->dg[0], Q->dg[1], Q->dg[2], Q->dg[3], Q->dg[4]);
    }
    if (Q->wib > 0 && Q->sums[5] >= 4) {   /* iBOT: the mean target entropy of the last step's first 64 masked rows, the loss, rows per step */
        const int r = Q->sums[5] < 64 ? (int)Q->sums[5] : 64, KI = Q->KI; float *qi = malloc((size_t)r * KI * 4); nn_d2h(qi, Q->iq, (size_t)r * KI * 4); double Hi = 0;
        double pmi = 0;
        for (int a = 0; a < r; a++) { double z = 0, h = 0, mx = 0; for (int k = 0; k < KI; k++) z += qi[(size_t)a * KI + k]; for (int k = 0; k < KI; k++) { const double t = qi[(size_t)a * KI + k] / z; if (t > 0) h -= t * log(t); if (t > mx) mx = t; } Hi += h / r; pmi += mx / r; }
        free(qi); const size_t l2 = strlen(line);
        snprintf(line + l2, cap - l2, " | ibot %.4f H(t) %.2f (max %.2f) pmax %.4f rows %.0f", Q->nsum ? Q->sums[3] / Q->nsum : 0.0, Hi, log((double)KI), pmi, Q->nsum ? Q->sums[4] / Q->nsum : 0.0);
        if (Q->lastm >= 4) {   /* the student side (the step's last masked view): softmax(logits / ts) entropy / pmax, unit bottleneck std per dim and cos between its rows */
            const int m = Q->lastm, Bz = Q->I.H.B; float *sl = malloc((size_t)m * KI * 4), *zn = malloc((size_t)m * Bz * 4);
            nn_d2h(sl, Q->isl, (size_t)m * KI * 4); nn_d2h(zn, Q->I.w.zn, (size_t)m * Bz * 4);
            double fsd = 0, fm = 0; { const int IC = Q->ICIN; float *fx = malloc((size_t)m * IC * 4); nn_d2h(fx, Q->ieS, (size_t)m * IC * 4);
              for (int c = 0; c < IC; c++) { double mu = 0, v = 0; for (int a = 0; a < m; a++) mu += fx[(size_t)a * IC + c] / m; for (int a = 0; a < m; a++) v += (fx[(size_t)a * IC + c] - mu) * (fx[(size_t)a * IC + c] - mu) / m; fsd += sqrt(v) / IC; fm += fabs(mu) / IC; }
              free(fx); }
            double Hs = 0, pms = 0;
            for (int a = 0; a < m; a++) { double mx = -1e30, z = 0, h = 0; const float *l = sl + (size_t)a * KI; for (int k = 0; k < KI; k++) mx = fmax(mx, l[k] / Q->ts);
                for (int k = 0; k < KI; k++) z += exp(l[k] / Q->ts - mx);
                for (int k = 0; k < KI; k++) { const double t = exp(l[k] / Q->ts - mx) / z; if (t > 0) h -= t * log(t); } Hs += h / m; pms += 1 / z / m; }
            double sm = 0, co = 0; long no = 0;
            for (int b = 0; b < Bz; b++) { double mu = 0, v = 0; for (int a = 0; a < m; a++) mu += zn[(size_t)a * Bz + b] / m; for (int a = 0; a < m; a++) v += (zn[(size_t)a * Bz + b] - mu) * (zn[(size_t)a * Bz + b] - mu) / m; sm += sqrt(v) / Bz; }
            for (int a = 0; a < m; a++) for (int c = a + 1; c < m; c++) { double d = 0; for (int b = 0; b < Bz; b++) d += zn[(size_t)a * Bz + b] * zn[(size_t)c * Bz + b]; co += d; no++; }
            size_t l5 = strlen(line);
            if (Q->ilsc) { const size_t l8 = strlen(line); snprintf(line + l8, cap - l8, " | ibot logit scale %.3f", expf(Q->lsc)); l5 = strlen(line); }
            if (Q->dist_n) { const size_t l9 = strlen(line); snprintf(line + l9, cap - l9, " | distill 1-cos %.4f", Q->dist_l / Q->dist_n); Q->dist_l = 0; Q->dist_n = 0; l5 = strlen(line); }
            if (Q->dtmn) { const size_t l9 = strlen(line); snprintf(line + l9, cap - l9, " | |mean target| %.3f", Q->dtm / Q->dtmn); Q->dtm = 0; Q->dtmn = 0; l5 = strlen(line); }
            if (Q->zn_n) { const size_t l9 = strlen(line); snprintf(line + l9, cap - l9, " | emb |z| %.4g cos %.3f", Q->zn_s / Q->zn_n, Q->zsim_s / Q->zn_n); Q->zn_s = Q->zsim_s = 0; Q->zn_n = 0; l5 = strlen(line); }
            if (Q->cc_v) { const size_t l9 = strlen(line); snprintf(line + l9, cap - l9, " | distill cc supcon %.4f (%.1f comps, %.2f of points kept, largest %.2f, pos %.3f, skipped %.2f)", Q->cc_n ? Q->cc_l / Q->cc_n : 0.0,
                Q->cc_nc / Q->cc_v, Q->cc_keep / Q->cc_v, Q->cc_big / Q->cc_v, Q->cc_n ? Q->cc_pos / Q->cc_n : 0.0, Q->cc_skip / Q->cc_v);
                Q->cc_l = Q->cc_keep = Q->cc_nc = Q->cc_pos = Q->cc_big = Q->cc_skip = 0; Q->cc_n = Q->cc_v = 0; l5 = strlen(line); }
            if (Q->nce_n) { const size_t l9 = strlen(line); snprintf(line + l9, cap - l9, " | distill nce %.4f pos %.3f", Q->nce_l / Q->nce_n, Q->nce_pf / Q->nce_n); Q->nce_l = 0; Q->nce_pf = 0; Q->nce_n = 0; l5 = strlen(line); }
            if (Q->rel_n) { const size_t l9 = strlen(line); snprintf(line + l9, cap - l9, " | distill rel %.4f", Q->rel_l / Q->rel_n); Q->rel_l = 0; Q->rel_n = 0; l5 = strlen(line); }
            if (Q->sig_n) { const size_t l8 = strlen(line); snprintf(line + l8, cap - l8, " | sigreg %.4f", Q->sig_l / Q->sig_n); Q->sig_l = 0; Q->sig_n = 0; l5 = strlen(line); }
            if (Q->zvh_n) { const size_t l7 = strlen(line); snprintf(line + l7, cap - l7, " | ibot zvar hinge %.5f", Q->zvh_l / Q->zvh_n); Q->zvh_l = 0; Q->zvh_n = 0; l5 = strlen(line); }
            if (Q->icos) { const size_t l6 = strlen(line); snprintf(line + l6, cap - l6, " | ibot cos(student, dinovol token) %.4f", Q->cosn ? Q->cosacc / Q->cosn : 0.0); Q->cosacc = 0; Q->cosn = 0; l5 = strlen(line); }
            snprintf(line + l5, cap - l5, " | ibot student feat std/ch %.4f |mean| %.4f H %.2f pmax %.4f tok z std/dim %.4f (1/sqrt(B) %.4f) cos other %.4f%s", fsd, fm, Hs, pms, sm, 1 / sqrt((double)Bz), no ? co / no : 0.0, Q->dv && Q->dist <= 0 ? (Q->dvuntil <= 0 || step <= Q->dvuntil ? " [dv teacher]" : " [ema teacher]") : "");
            free(sl); free(zn);
        }
    }
    memset(Q->sums, 0, sizeof Q->sums); memset(Q->st, 0, sizeof Q->st); Q->nsum = 0; free(q); free(mrow);
}

static int dn_recipe(unet *u, const unet_cfg *cfg, dn_data *Dd, int argc, char **argv, int P, int steps0, double minutes, const char *outd, int nthreads, int nwork, uint64_t seed, int HZ) {
    if (!cfg->attn_blocks) { setenv("UFSM_VIRTUAL_TOKENS", "1", 1); fprintf(stderr, "dino-train --recipe: no attention stage (conv-only): zero CLS tokens (the CLS losses must be off)\n"); }
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
    const int cls_reg = !strcmp(opt(argc, argv, "--cls", "pool"), "reg") ? 1 : !strcmp(opt(argc, argv, "--cls", "pool"), "both") ? 2 : 0;   /* (both: register | token mean, the SSL batch) */
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
    /* queue InfoNCE self-supervision (works at batch 1; prototypes do not: see dn_head_step): --moco W the CLS (MoCo: student projection of
       view a vs the EMA teacher's of view b, negatives a FIFO of --moco-queue teacher keys of earlier samples; locals vs both globals);
       --dnce W the dense cross-view term (DenseCL / PixPro style, iBOT's masking and correspondences: each masked student token's positive
       is the teacher feature at the same physical point of the other view, negatives the view's other points + a FIFO of --dnce-queue keys) */
    const float wmo = (float)atof(opt(argc, argv, "--moco", "0")), wdn = (float)atof(opt(argc, argv, "--dnce", "0"));
    const float mtau = (float)atof(opt(argc, argv, "--moco-tau", "0.2")), dtau = (float)atof(opt(argc, argv, "--dnce-tau", "0.2"));
    const int MQN = atoi(opt(argc, argv, "--moco-queue", "16384")), DQN = atoi(opt(argc, argv, "--dnce-queue", "16384")), tokp = wib > 0 || wdn > 0;
    dr_mlp2 MP, DP; dr_mlp2_layout(&MP, C, 2048, 256); dr_mlp2_layout(&DP, ED, 512, 128);
    /* --tnce W: dense cross-view InfoNCE on the trunk's attention-stage grid tokens (not the sheet-identity embedding, whose points are
       interchangeable across windows): up to --tnce-max student token cells of a global view, each positive the teacher token at the same
       physical point of the other global (nearest cell), negatives a FIFO of --tnce-queue teacher token keys of earlier samples */
    const float wtn = (float)atof(opt(argc, argv, "--tnce", "0")), ttau = (float)atof(opt(argc, argv, "--tnce-tau", "0.2"));
    const int TQN = atoi(opt(argc, argv, "--tnce-queue", "16384")), TNM = atoi(opt(argc, argv, "--tnce-max", "256"));
    dr_mlp2 TPH; dr_mlp2_layout(&TPH, C, 512, 128);
    float *tp = nullptr, *tpt = nullptr, *tg_ = nullptr, *tm_ = nullptr, *tv_ = nullptr, *tmo = nullptr, *tkb = nullptr, *tqv = nullptr, *tgq = nullptr, *tA = nullptr, *tB = nullptr, *tdA = nullptr, *ttok[2] = {nullptr, nullptr};
    void *tws = nullptr, *twt = nullptr, *tsc = nullptr; int *tiS = nullptr, *tiT = nullptr, tqf = 0, tqp = 0; size_t ttokn = 0;
    if (wtn > 0) {
        uint64_t r5 = seed * 53 + 17; float *b = calloc(TPH.n, 4);
        for (size_t i = TPH.w1; i < TPH.b1; i++) b[i] = (float)(sqrt(-2 * log(rng_u(&r5) + 1e-12)) * cos(6.283185307 * rng_u(&r5)) / sqrt((double)TPH.Cin));
        for (size_t i = TPH.w2; i < TPH.b2; i++) b[i] = (float)(sqrt(-2 * log(rng_u(&r5) + 1e-12)) * cos(6.283185307 * rng_u(&r5)) / sqrt((double)TPH.H));
        tp = nn_malloc(TPH.n * 4); tpt = nn_malloc(TPH.n * 4); nn_h2d(tp, b, TPH.n * 4); nn_h2d(tpt, b, TPH.n * 4); free(b);
        tg_ = nn_malloc(TPH.n * 4); tm_ = nn_malloc(TPH.n * 4); tv_ = nn_malloc(TPH.n * 4); tmo = nn_malloc(TPH.n * 4); nn_zero(tm_, TPH.n * 4); nn_zero(tv_, TPH.n * 4); nn_zero(tmo, TPH.n * 4); nn_zero(tg_, TPH.n * 4);
        tws = nn_malloc(dr_mlp2_ws(&TPH, TNM)); twt = nn_malloc(dr_mlp2_ws(&TPH, TNM)); tkb = nn_malloc((size_t)(TNM + TQN) * 128 * 4); nn_zero(tkb, (size_t)(TNM + TQN) * 128 * 4);
        tqv = nn_malloc((size_t)TNM * 128 * 4); tgq = nn_malloc((size_t)TNM * 128 * 4); tA = nn_malloc((size_t)TNM * C * 4); tB = nn_malloc((size_t)TNM * C * 4); tdA = nn_malloc((size_t)TNM * C * 4);
        tsc = nn_malloc(dr_nce_scratch(TNM, TQN + 1)); tiS = nn_malloc((size_t)2 * TNM * 4); tiT = nn_malloc((size_t)2 * TNM * 4);
    }
    float *mp = nullptr, *mpt = nullptr, *mg = nullptr, *mm = nullptr, *mv = nullptr, *mmo = nullptr, *mkb = nullptr, *mq = nullptr, *mgq = nullptr; void *mws = nullptr, *mwt = nullptr, *msc = nullptr; int *mpos = nullptr, mqf = 0, mqp = 0;
    float *dp = nullptr, *dpt = nullptr, *dg = nullptr, *dm = nullptr, *dv = nullptr, *dmo = nullptr, *dkb = nullptr, *dq = nullptr, *dgq = nullptr; void *dws = nullptr, *dwt = nullptr, *dsc = nullptr; int *dpos = nullptr, dqf = 0, dqp = 0;
    float *pwork = nullptr;
    {   /* projection heads: N(0, 1 / fan-in) weights, zero biases; the teacher a copy */
        uint64_t r4 = seed * 47 + 13;
        for (int k = 0; k < 2; k++) {
            const dr_mlp2 *L = k ? &DP : &MP; if (k ? !(wdn > 0) : !(wmo > 0)) continue;
            float **pp = k ? &dp : &mp, **pt_ = k ? &dpt : &mpt;
            float *b = calloc(L->n, 4); for (size_t i = L->w1; i < L->b1; i++) b[i] = (float)(sqrt(-2 * log(rng_u(&r4) + 1e-12)) * cos(6.283185307 * rng_u(&r4)) / sqrt((double)L->Cin));
            for (size_t i = L->w2; i < L->b2; i++) b[i] = (float)(sqrt(-2 * log(rng_u(&r4) + 1e-12)) * cos(6.283185307 * rng_u(&r4)) / sqrt((double)L->H));
            *pp = nn_malloc(L->n * 4); *pt_ = nn_malloc(L->n * 4); nn_h2d(*pp, b, L->n * 4); nn_h2d(*pt_, b, L->n * 4); free(b);
        }
        size_t wb = 0; const size_t cand[5] = {dr_muon_work(2048, C), dr_muon_work(256, 2048), dr_muon_work(512, ED), dr_muon_work(128, 512), dr_muon_work(512, C)};
        for (int i = 0; i < 5; i++) if (cand[i] > wb) wb = cand[i];
        if (wmo > 0 || wdn > 0 || wtn > 0) pwork = nn_malloc(wb);
        if (wmo > 0) {
            mg = nn_malloc(MP.n * 4); mm = nn_malloc(MP.n * 4); mv = nn_malloc(MP.n * 4); mmo = nn_malloc(MP.n * 4); nn_zero(mm, MP.n * 4); nn_zero(mv, MP.n * 4); nn_zero(mmo, MP.n * 4);
            mws = nn_malloc(dr_mlp2_ws(&MP, 2)); mwt = nn_malloc(dr_mlp2_ws(&MP, 2)); mkb = nn_malloc((size_t)(2 + MQN) * 256 * 4); mq = nn_malloc(256 * 4); mgq = nn_malloc(256 * 4);
            msc = nn_malloc(dr_nce_scratch(1, MQN + 1)); mpos = nn_malloc(2 * 4); const int pz[2] = {0, 1}; nn_h2d(mpos, pz, 8);
        }
        if (wdn > 0) {
            dg = nn_malloc(DP.n * 4); dm = nn_malloc(DP.n * 4); dv = nn_malloc(DP.n * 4); dmo = nn_malloc(DP.n * 4); nn_zero(dm, DP.n * 4); nn_zero(dv, DP.n * 4); nn_zero(dmo, DP.n * 4);
            dws = nn_malloc(dr_mlp2_ws(&DP, MI)); dwt = nn_malloc(dr_mlp2_ws(&DP, MI)); dkb = nn_malloc((size_t)(MI + DQN) * 128 * 4); dq = nn_malloc((size_t)MI * 128 * 4); dgq = nn_malloc((size_t)MI * 128 * 4);
            dsc = nn_malloc(dr_nce_scratch(MI, DQN + 1)); nn_zero(dkb, (size_t)(MI + DQN) * 128 * 4); dpos = nn_malloc((size_t)MI * 4); int *h = malloc((size_t)MI * 4); for (int i = 0; i < MI; i++) h[i] = i; nn_h2d(dpos, h, (size_t)MI * 4); free(h);
        }
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
    /* inputs per view: CT, then with --axis-input (cin 5) train's radial unit vector (3) and a validity channel (dr_axis_view); the axis
       is dropped (valid 0) for a view with probability --axis-drop, and missing for sources without "axis" */
    const int CIN = cfg->cin; const float adrop = (float)atof(opt(argc, argv, "--axis-drop", "0.15"));
    if (CIN != 1 && CIN != 2 && CIN != 5) { fprintf(stderr, "dino-train --recipe: inputs 1 (CT), 2 (--mask-input) or 5 (--axis-input: CT, radial, valid)\n"); return 2; }
    uint8_t *src = nn_malloc(ps3); float *xg = nn_malloc(2 * (size_t)CIN * g3 * 4), *xl = nn_malloc((size_t)(nloc > 0 ? nloc : 1) * CIN * l3 * 4), *d_ayx = nn_malloc((size_t)Ps * 8);
    dr_xf *xfd = nn_malloc(sizeof(dr_xf) * (2 + nloc)); dr_int *itd = nn_malloc(sizeof(dr_int) * (2 + nloc));
    const int NQ = 2 + SQ; float *tl = nn_malloc((size_t)NQ * K * 4), *tq = nn_malloc((size_t)NQ * K * 4); void *sks = nn_malloc(dr_sk_scratch(NQ, K));
    /* iBOT buffers: per masked global v the teacher targets [MI][KI]; student logits / gradient [MI][KI]; points; the cell mask */
    float *ibq[2] = {nullptr, nullptr}, *ibl = nullptr, *ibd = nullptr, *ieT = nullptr, *inT = nullptr, *ieS = nullptr, *inS = nullptr, *idE = nullptr, *ibw = nullptr; void *ibsk = nullptr, *ibce = nullptr;
    int *isidx = nullptr, *itidx = nullptr, *iid = nullptr, *ihS = malloc((size_t)MI * 4 * 2), *ihT = malloc((size_t)MI * 4 * 2); uint8_t *imask = nullptr;
    if (tokp) {
        if (wib > 0) {
            for (int v = 0; v < 2; v++) ibq[v] = nn_malloc((size_t)MI * KI * 4);
            ibl = nn_malloc((size_t)MI * KI * 4); ibd = nn_malloc((size_t)MI * KI * 4); ibsk = nn_malloc(dr_sk_scratch(MI, KI)); ibce = nn_malloc(dr_ce_scratch(MI));
        }
        ieT = nn_malloc((size_t)2 * MI * ED * 4); inT = nn_malloc((size_t)2 * MI * 4); ieS = nn_malloc((size_t)MI * ED * 4); inS = nn_malloc((size_t)MI * 4); idE = nn_malloc((size_t)MI * ED * 4);
        isidx = nn_malloc((size_t)2 * MI * 4); itidx = nn_malloc((size_t)2 * MI * 4); iid = nn_malloc((size_t)MI * 4); ibw = nn_malloc((size_t)MI * 4); imask = nn_malloc((size_t)GT * GT * GT);
        int *h = malloc((size_t)MI * 4); for (int i = 0; i < MI; i++) h[i] = i; nn_h2d(iid, h, (size_t)MI * 4); free(h);
    }
    uint8_t *useb = calloc((size_t)(K > KI ? K : KI), 1), *useib = calloc((size_t)KI, 1); int *d_am = nn_malloc((size_t)(MI > NQ ? MI : NQ) * 4), *h_am = malloc((size_t)(MI > NQ ? MI : NQ) * 4);
    float *sl = nn_malloc((size_t)K * 4), *dsl = nn_malloc((size_t)K * 4); void *ces = nn_malloc(dr_ce_scratch(1));
    float *clsRS = nn_malloc(8 * 4);   /* (row-norm rstd: teacher rows 0-1, student 4) */
    float *kpend = nn_malloc(2 * (size_t)C * 4);
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
                  for (int k = 0; k < 3; k++) {   /* projection heads, when saved by a run with the same terms on */
                      const dr_mlp2 *L = k == 2 ? &TPH : k ? &DP : &MP; float *P_ = k == 2 ? tp : k ? dp : mp, *T_ = k == 2 ? tpt : k ? dpt : mpt; if (!P_) continue;
                      float *g_ = malloc(L->n * 4); if (fread(g_, 4, L->n, hf) == L->n) { nn_h2d(P_, g_, L->n * 4); if (fread(g_, 4, L->n, hf) == L->n) nn_h2d(T_, g_, L->n * 4); } free(g_);
                  }
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
    Rg->nslots = atoi(opt(argc, argv, "--slots", "6")); if (Rg->nslots < 2) Rg->nslots = 2; if (Rg->nslots > DN_SLOTS) Rg->nslots = DN_SLOTS;
    for (int i = 0; i < Rg->nslots; i++) { dn_sample *S = &Rg->slot[i]; S->ca = malloc(ps3); S->cb = malloc(ps3); S->gidx = malloc((size_t)NG * 4); S->gcode = malloc((size_t)NG * 4); S->gxyz = malloc((size_t)NG * 12); S->sidx = malloc(4); S->ayx = malloc((size_t)Ps * 8); S->ixyz = malloc((size_t)(Dd->nink > 0 ? Dd->nink : 1) * 12);
        if (Dd->uw) { S->phw = malloc(ps3); for (int v = 0; v < 2; v++) { S->pv[v] = malloc(g3); S->w4[v] = malloc(g3); S->wc[v] = malloc(g3); } }
        if (Dd->ikp) { S->ikw = malloc(ps3); for (int v = 0; v < 2; v++) { S->iv[v] = malloc(g3); S->im[v] = malloc(g3); } }
        if ((Dd->uw && (!S->phw || !S->pv[1] || !S->w4[1] || !S->wc[1])) || (Dd->ikp && (!S->ikw || !S->iv[1] || !S->im[1]))) { fprintf(stderr, "dino-train --recipe: host memory for the dense targets\n"); return 1; } }
    Dd->vP = P; Dd->vPL = PL; Dd->vnloc = nloc; Dd->zlo = zlo; Dd->zhi = zhi; Dd->lzlo = lzlo; Dd->lzhi = lzhi;   /* the views: drawn by the workers */
    Rg->nworkers = nwork > DN_MAXW ? DN_MAXW : nwork;
    for (int i = 0; i < Rg->nworkers; i++) pthread_create(&Rg->th[i], nullptr, ring_worker, Rg);
    const int SB = atoi(opt(argc, argv, "--ssl-batch", "0")), sse = atoi(opt(argc, argv, "--ssl-every", "1")) > 0 ? atoi(opt(argc, argv, "--ssl-every", "1")) : 1;
    dn_ssl SS; if (SB > 0 && dn_ssl_init(&SS, u, cfg, Dd, argc, argv, seed, C, cls_reg, cls_norm, noise)) return 2;
    if (SB > 0 && opt(argc, argv, "--resume", nullptr)) { char rr[4096]; const char *rq = realpath(opt(argc, argv, "--resume", ""), rr) ? rr : opt(argc, argv, "--resume", ""); dn_ssl_io(&SS, rq, 0); }
    signal(SIGTERM, dn_sig); signal(SIGINT, dn_sig);
    char logp[1200]; snprintf(logp, sizeof logp, "%s/train.log", outd); FILE *lf = fopen(logp, "a");
    fprintf(stderr, "dino-train --recipe: source %d, globals 2 x %d^3 (zoom %.2f..%.2f), locals %d x %d^3 (zoom %.2f..%.2f), CLS %s, head %d-%d-%d -> %d prototypes, "
                    "Sinkhorn %d iters over 2 + %d queued rows, KoLeo %g (queue %d), GT %g, teacher temp %g -> %g over %d steps, EMA %g -> %g\n",
            Ps, P, zlo, zhi, nloc, PL, lzlo, lzhi, cls_reg ? "register 0" : "grid mean", HH, HH, HB, K, sk_it, SQ, wk, KQ, wgt, tt0, tt1, tt_warm, ema0, ema1);
    uint64_t ar = seed * 1000003 + 99; int qfill = 0, qpos = 0, kfill = 0, kpos = 0, ibv = -1;
    float *dcen = nn_malloc((size_t)K * 4), *icen = nn_malloc((size_t)KI * 4); nn_zero(dcen, (size_t)K * 4); nn_zero(icen, (size_t)KI * 4);   /* (centres: zero at start / resume) */
    double sums[20] = {0}; int nsum = 0; double t_last = dn_now(), t_loop = dn_now(), t_meas0 = 0;
    double dsum[5] = {0}; int dn_nuw = 0, dn_nik = 0;
    double *dsrc = calloc((size_t)DN_MAXSRC * 4, sizeof(double));   /* per source: unwrap2 views, normals bce, winding mod 4 bce, its accuracy (the log's "by source") */ const int dense_from = atoi(opt(argc, argv, "--dense-from", "-1")) >= 0 ? atoi(opt(argc, argv, "--dense-from", "-1")) : step0;
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
        static int rfix = -1; static dn_sample *Sfix; if (rfix < 0) rfix = getenv("UFSM_REC_FIXED") && atoi(getenv("UFSM_REC_FIXED"));   /* (debug: one fixed sample, views and augmentations every step) */
        dn_sample *Sm = rfix && Sfix ? Sfix : ring_get(Rg); if (!Sm) { fprintf(stderr, "dino-train: data loader failed\n"); return 1; }
        while (rfix && !Sfix && wgt > 0 && Sm->ngt < 2000) { ring_put(Rg, Sm); Sm = ring_get(Rg); if (!Sm) return 1; }   /* (a labelled sample) */
        if (rfix && !Sfix) { Sfix = Sm; fprintf(stderr, "dino-train: UFSM_REC_FIXED: one fixed sample (source %d, %d GT points)\n", Sm->si, Sm->ngt); }
        if (rfix) ar = seed * 1000003 + 99;
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
            a->invert = rng_u(&ar) < 0.15; a->sigma = noise * (float)rng_u(&ar); a->seed = (uint32_t)((rfix ? 1 : step) * 16 + v);
        }
        nn_h2d(xfd, xf, sizeof(dr_xf) * (2 + nloc)); nn_h2d(itd, it, sizeof(dr_int) * (2 + nloc));
        if (CIN == 1) { dr_crops(src, Ps, xfd, itd, 2, P, xg); if (nloc) dr_crops(src, Ps, xfd + 2, itd + 2, nloc, PL, xl); }
        else {
            if (Sm->hasax) nn_h2d(d_ayx, Sm->ayx, (size_t)Ps * 8);
            for (int v = 0; v < 2 + nloc; v++) {
                const int gl = v < 2, n_ = gl ? P : PL; const size_t n3_ = (size_t)n_ * n_ * n_; float *xv = gl ? xg + (size_t)v * CIN * g3 : xl + (size_t)(v - 2) * CIN * l3;
                dr_crops(src, Ps, xfd + v, itd + v, 1, n_, xv);
                if (CIN == 2) nn_zero(xv + n3_, n3_ * 4);
                else dr_axis_view(xfd, v, n_, Ps, d_ayx, Sm->hasax && rng_u(&ar) >= adrop, xv + n3_);
            }
        }
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
        if (!HS.uw && !rfix) ring_put(Rg, Sm);   /* (dense heads: the slot's per-view targets are read in the student loop: held until after it) */
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
        for (int v = 0; v < 2 && tokp; v++) {
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
            unet_forward(u, xg + (size_t)v * CIN * g3, xsg, 0);
            const float *tok = unet_tokens(u, nullptr, &S0, &Cc, &nreg);
            if (cls_reg) nn_d2d(clsT + (size_t)v * C, tok, (size_t)C * 4); else dr_pool_fwd(tok, 1, S0, C, nreg, clsT + (size_t)v * C);
            if (wtn > 0) {   /* (the teacher's tokens of this view, for the other view's token InfoNCE) */
                if (ttokn < (size_t)S0 * C) { for (int k = 0; k < 2; k++) { nn_free(ttok[k]); ttok[k] = nn_malloc((size_t)S0 * C * 4); } ttokn = (size_t)S0 * C; }
                nn_d2d(ttok[v], tok, (size_t)S0 * C * 4);
            }
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
        if (wmo > 0 && dr_mlp2_fwd(&MP, mpt, clsT, 2, mkb, mwt)) return 1;   /* MoCo: the teacher keys of both globals (rows 0, 1 of the key bank; the queue after) */
        D.w.zc_m = zcm;   /* (the teacher rows update the z centre) */
        if (dr_head_fwd(&D.H, D.pt, clsT, 2, tl, &D.w)) return 1;
        D.w.zc_m = 0.f;
        if (dcm > 0) { dr_teacher_probs(tl, 2, K, tt, 0, dcen, tq, sks); dr_center_update(dcen, tl, 2, K, dcm); }   /* (centre: before its update, as DINOv1) */
        else dr_teacher_probs(tl, 2 + qfill, K, tt, sk_it, nullptr, tq, sks);   /* (rows 0, 1 current; the queue after them) */
        dr_argmax_rows(tq, 2, K, d_am); nn_d2h(h_am, d_am, 8); useb[h_am[0]] = useb[h_am[1]] = 1;
        for (int v = 0; v < 2 && wib > 0; v++) if (masked[v]) {   /* iBOT teacher targets: Sinkhorn over the view's masked tokens */
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
        int moco_calls = 0, dnce_calls = 0, tnce_calls = 0;
        if (wreg > 0) nn_zero(rg, RH.n * 4);
        if (NIK > 0) nn_zero(igr, IH.n * 4);
        if ((step - 1) % hacc == 0) nn_zero(D.g, D.H.n * 4);   /* (--head-accum: the DINO head's gradient summed over hacc steps) */
        if (wib > 0) nn_zero(I.g, I.H.n * 4);
        const int nmv = masked[0] + masked[1];
        int tnm[2] = {0, 0};   /* token InfoNCE pairs per global view: student token rows of view v, teacher rows of view 1 - v at the same point */
        if (wtn > 0 && ttok[0]) {
            const int TS = 1 << (cfg->nlev - 1), g = P / TS; static int *hS, *hT; if (!hS) { hS = malloc((size_t)2 * TNM * 4); hT = malloc((size_t)2 * TNM * 4); }
            for (int v = 0; v < 2; v++) {
                int seen = 0;
                for (int c = 0; c < g * g * g; c++) {
                    const int cz = c / (g * g), cy = (c / g) % g, cx = c % g;
                    const double vv[3] = {cz * TS + TS / 2 - 0.5, cy * TS + TS / 2 - 0.5, cx * TS + TS / 2 - 0.5}; double sx[3], o[3]; dr_map(&xf[v], vv, sx); dr_unmap(&xf[1 - v], sx, o);
                    int q[3], in = 1; for (int d = 0; d < 3; d++) { q[d] = (int)floor(o[d] / TS); in &= o[d] >= 0 && o[d] < P; }
                    if (!in) continue;
                    int k = seen++; if (k >= TNM) { k = (int)(rng_next(&ar) % (uint64_t)seen); if (k >= TNM) continue; }
                    hS[v * TNM + k] = nreg + c; hT[v * TNM + k] = nreg + (q[0] * g + q[1]) * g + q[2];
                }
                tnm[v] = seen < TNM ? seen : TNM;
                if (tnm[v] < 16) { tnm[v] = 0; continue; }
                nn_h2d(tiS + v * TNM, hS + v * TNM, (size_t)tnm[v] * 4); nn_h2d(tiT + v * TNM, hT + v * TNM, (size_t)tnm[v] * 4);
            }
        }
        for (int v = 0; v < 2 + nloc; v++) {
            const int gl = v < 2; unet *n = gl ? u : ul;
            const float *x = gl ? xg + (size_t)v * CIN * g3 : xl + (size_t)(v - 2) * CIN * l3;
            RP(9);
            if (gl && masked[v]) { nn_h2d(imask, cellm[v], (size_t)GT * GT * GT); dr_mask_cells(xg + (size_t)v * CIN * g3, P, TPS, imask, 0.f); }   /* (the teacher saw it unmasked) */
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
            if (wmo > 0) {   /* MoCo: a global against the other global's teacher key; a local against both (halves); negatives the queue */
                if (dr_mlp2_fwd(&MP, mp, clsS, 1, mq, mws)) return 1;
                nn_zero(mgq, 256 * 4);
                const int np_ = gl ? 1 : 2; const float wv = wmo / (float)terms * (gl ? 1.f : 1.f);
                for (int j = 0; j < np_; j++) {   /* positive: that global's teacher key; negatives: the queue (other samples) only */
                    const int key = gl ? 1 - v : j;
                    dr_infonce_p(mq, mkb + (size_t)key * 256, 1, mkb + 2 * 256, mqf, 256, mtau, wv, mgq, outs + 40, msc);
                    moco_calls++;
                }
                if (dr_mlp2_bwd(&MP, mp, mg, clsS, mq, mgq, 1, dcls, mws)) return 1;
            }
            if (gl && wk > 0 && kfill > 0) dr_koleo_q(clsS, 1, kq, kfill, C, wk / 2, dcls, outs + 17);
            if (cls_norm) dr_rownorm_bwd(dcls, clsS, clsRS + 4, 1, C, dcls);   /* (d / d normed CLS -> d / d pooled CLS) */
            /* d / d CLS -> token gradient (the backward's scaled domain) */
            nn_zero(tgrad, (size_t)Sv * C * 4);
            if (cls_reg) nn_axpy(tgrad, gs, dcls, (size_t)C); else { dn_axpby(dcls, dcls, (size_t)C, gs, 0.f); dr_pool_bwd(dcls, 1, Sv, C, nreg, tgrad); }
            if (gl && wtn > 0 && tnm[v]) {   /* token InfoNCE: these student tokens vs the other view's teacher tokens at the same points, queue negatives */
                const int m = tnm[v];
                dr_rows_gather(tok, C, tiS + v * TNM, m, tA); dr_rows_gather(ttok[1 - v], C, tiT + v * TNM, m, tB);
                if (dr_mlp2_fwd(&TPH, tpt, tB, m, tkb, twt) || dr_mlp2_fwd(&TPH, tp, tA, m, tqv, tws)) return 1;
                nn_zero(tgq, (size_t)m * 128 * 4); nn_zero(tdA, (size_t)m * C * 4);
                dr_infonce_p(tqv, tkb, m, tkb + (size_t)TNM * 128, tqf, 128, ttau, wtn / 2, tgq, outs + 48, tsc); tnce_calls++;
                if (dr_mlp2_bwd(&TPH, tp, tg_, tA, tqv, tgq, m, tdA, tws)) return 1;
                dr_rows_scatter_add(tdA, tiS + v * TNM, m, C, gs, tgrad);   /* (the token gradient's scaled domain, as the CLS path) */
                const int n1 = m < TQN - tqp ? m : TQN - tqp;   /* the keys into the FIFO */
                nn_d2d(tkb + (size_t)(TNM + tqp) * 128, tkb, (size_t)n1 * 128 * 4);
                if (m > n1) nn_d2d(tkb + (size_t)TNM * 128, tkb + (size_t)n1 * 128, (size_t)(m - n1) * 128 * 4);
                tqp = (tqp + m) % TQN; tqf = tqf + m < TQN ? tqf + m : TQN;
            }
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
            if (gl && masked[v]) {   /* iBOT / dense InfoNCE: the masked tokens' student embeddings against the cross-view teacher targets */
                const int m = nmi[v];
                dn_gather_slabs(&HS, isidx + v * MI, m, ieS, inS, nullptr, 0, nullptr, nullptr);
                if (wib > 0) {
                    if (dr_head_fwd(&I.H, I.p, ieS, m, ibl, &I.w)) return 1;
                    float *hw_ = malloc((size_t)m * 4); for (int i = 0; i < m; i++) hw_[i] = wib / (float)m / (float)nmv; nn_h2d(ibw, hw_, (size_t)m * 4); free(hw_);
                    dr_ce(ibl, m, KI, ts, ibq[v], iid, iid, ibw, m, ibd, outs + 28, ibce);
                    if (dr_head_bwd(&I.H, I.p, I.g, ieS, ibd, m, idE, step > freeze, &I.w)) return 1;
                } else nn_zero(idE, (size_t)m * ED * 4);
                if (wdn > 0) {   /* keys: the teacher projection of this view's targets (in-batch: positive i, the rest negatives), then the queue */
                    /* key bank: MI in-batch slots (this view's m keys: the positives), then the queue (the negatives: other samples' keys) */
                    if (dr_mlp2_fwd(&DP, dpt, ieT + (size_t)v * MI * ED, m, dkb, dwt)) return 1;
                    if (dr_mlp2_fwd(&DP, dp, ieS, m, dq, dws)) return 1;
                    nn_zero(dgq, (size_t)m * 128 * 4);
                    dr_infonce_p(dq, dkb, m, dkb + (size_t)MI * 128, dqf, 128, dtau, wdn / (float)nmv, dgq, outs + 44, dsc); dnce_calls++;   /* (negatives: the queue only) */
                    if (dr_mlp2_bwd(&DP, dp, dg, ieS, dq, dgq, m, idE, dws)) return 1;
                    {   /* this view's keys into the FIFO (one or two block copies) */
                        const int n1 = m < DQN - dqp ? m : DQN - dqp;
                        nn_d2d(dkb + (size_t)(MI + dqp) * 128, dkb, (size_t)n1 * 128 * 4);
                        if (m > n1) nn_d2d(dkb + (size_t)MI * 128, dkb + (size_t)n1 * 128, (size_t)(m - n1) * 128 * 4);
                        dqp = (dqp + m) % DQN; dqf = dqf + m < DQN ? dqf + m : DQN;
                    }
                }
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
                             { double *q = dsrc + (size_t)Sm->si * 4; q[0] += 1; q[1] += nb / 7; q[2] += w4h[0]; q[3] += w4h[1]; }
                             if (getenv("UFSM_DENSE_DBG")) fprintf(stderr, "dense view %d: phase %g %g (active %g) nrm %g w4 %g acc %g N %g comps %g um %.2f gnorm %g\n", v, o2[0], o2[1], o2[4], nb / 7, w4h[0], w4h[1], w4h[2], w4h[4], UW.um, unet_grad_norm(u)); }
                if (UW.ik) { float o1[3]; nn_loss_fetch(UW.iscr, (shape5){1, 1, 1, 1, 1}, o1); dsum[4] += o1[0]; dn_nik++; }
                UW.uw = UW.ik = 0;
            }
            nn_d2h(hout, outs, 52 * 4);
            lsum[0] += hout[16]; if (gl) { lsum[1] += hout[17]; lsum[2] += HS.ngt ? hout[20] : 0.f; lsum[3] += HS.ngt ? hout[24] : 0.f; lsum[4] += masked[v] ? hout[28] : 0.f; }
            if (gl && wk > 0) nn_d2d(kpend + (size_t)v * C, clsS, (size_t)C * 4);   /* (queued after the view loop: a view never meets its own sample's other view) */
        }
        if (wk > 0) for (int v = 0; v < 2; v++) { nn_d2d(kq + (size_t)kpos * C, kpend + (size_t)v * C, (size_t)C * 4); kpos = (kpos + 1) % KQ; if (kfill < KQ) kfill++; }
        if (SB > 0 && step % sse == 0) { if (dn_ssl_step(&SS, step, tt, mom)) return 1; }   /* the multi-sample SSL batch (its gradient into u) */
        nn_axpy(unet_grad_ptr(u), 1.f, unet_grad_ptr(ul), np);   /* the locals' share */
        if (HS.uw && !rfix) ring_put(Rg, Sm);
        const double gnorm = unet_grad_norm(u);
        if (!isfinite(gnorm) || !isfinite(lsum[0])) { fprintf(stderr, "dino-train --recipe: non-finite loss / gradient at step %d\n", step); return 1; }
        if (clip > 0) unet_clip_grad(u, clip);
        RP(9);
        static int hlog = -1; if (hlog < 0) hlog = getenv("UFSM_HEAD_LOG") ? atoi(getenv("UFSM_HEAD_LOG")) : 0;
        size_t hwo = 0, hbo = 0; int hco = 0, hci = 0; float *hw0 = nullptr, *hg = nullptr;
        if (hlog && step % hlog == 0) {   /* (debug: the output head's gradient and its parameter change this step) */
            unet_head_offsets(u, &hwo, &hbo); hco = unet_cfg_of(u)->cout; hci = (int)((hbo - hwo) / hco);
            hw0 = malloc((size_t)(hci + 1) * hco * 4); hg = malloc((size_t)(hci + 1) * hco * 4);
            nn_d2h(hw0, unet_live_params(u) + hwo, (size_t)(hci + 1) * hco * 4); nn_d2h(hg, unet_grad_ptr(u) + hwo, (size_t)(hci + 1) * hco * 4); }
        unet_muon(u, mlr0 * sched, 0.95f, lr0 * sched, 0.9f, 0.999f, 1e-8f, wd, step);
        if (hw0) { float *hw1 = malloc((size_t)(hci + 1) * hco * 4); nn_d2h(hw1, unet_live_params(u) + hwo, (size_t)(hci + 1) * hco * 4);
            double gw = 0, gb = 0, dw = 0, db = 0, nw = 0, nb = 0; const size_t nW = (size_t)hci * hco;
            for (size_t i = 0; i < nW; i++) { gw += (double)hg[i] * hg[i]; dw += (double)(hw1[i] - hw0[i]) * (hw1[i] - hw0[i]); nw += (double)hw1[i] * hw1[i]; }
            for (int i = 0; i < hco; i++) { gb += (double)hg[nW + i] * hg[nW + i]; db += (double)(hw1[nW + i] - hw0[nW + i]) * (hw1[nW + i] - hw0[nW + i]); nb += (double)hw1[nW + i] * hw1[nW + i]; }
            fprintf(stderr, "head step %d: W %dx%d |gW| %.3e |dW| %.3e |W| %.3e | |gb| %.3e |db| %.3e |b| %.3e b0 %.4f gb0 %.3e | gnorm %.3e\n", step, hco, hci, sqrt(gw), sqrt(dw), sqrt(nw), sqrt(gb), sqrt(db), sqrt(nb), hw1[nW], hg[nW], gnorm);
            free(hw1); free(hw0); free(hg); }
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
        for (int k = 0; k < 3; k++) {   /* the projection heads: Muon on the matrices, AdamW on the biases, EMA teacher */
            const dr_mlp2 *L = k == 2 ? &TPH : k ? &DP : &MP; float *P_ = k == 2 ? tp : k ? dp : mp, *T_ = k == 2 ? tpt : k ? dpt : mpt, *G_ = k == 2 ? tg_ : k ? dg : mg,
                  *M_ = k == 2 ? tm_ : k ? dm : mm, *V_ = k == 2 ? tv_ : k ? dv : mv, *O_ = k == 2 ? tmo : k ? dmo : mmo;
            if (!P_) continue;
            dr_muon(P_ + L->w1, G_ + L->w1, O_ + L->w1, L->H, L->Cin, mlr0 * sched, 0.95f, wd, pwork);
            dr_muon(P_ + L->w2, G_ + L->w2, O_ + L->w2, L->O, L->H, mlr0 * sched, 0.95f, wd, pwork);
            dr_adamw(P_ + L->b1, G_ + L->b1, M_ + L->b1, V_ + L->b1, L->H, hlr * sched, 0.9f, 0.999f, 1e-8f, 0.f, step);
            dr_adamw(P_ + L->b2, G_ + L->b2, M_ + L->b2, V_ + L->b2, L->O, hlr * sched, 0.9f, 0.999f, 1e-8f, 0.f, step);
            dr_ema(T_, P_, L->n, mom); nn_zero(G_, L->n * 4);
        }
        if (wmo > 0) for (int v = 0; v < 2; v++) { nn_d2d(mkb + (size_t)(2 + mqp) * 256, mkb + (size_t)v * 256, 256 * 4); mqp = (mqp + 1) % MQN; if (mqf < MQN) mqf++; }   /* the teacher keys into the queue */
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
        if (moco_calls) { sums[10] += hout[40] / (moco_calls * wmo / terms); sums[11] += hout[41] / moco_calls; }
        if (dnce_calls) { sums[12] += hout[44] / wdn; sums[13] += hout[45] / dnce_calls; sums[14] += 1; }
        if (tnce_calls) { sums[15] += hout[48] / (tnce_calls * wtn / 2); sums[16] += hout[49] / tnce_calls; sums[17] += 1; }
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
            char line[6144];
            /* prototype usage: distinct teacher-argmax prototypes since the last reset (every --usage-window steps; DINO sees 2 rows per step) */
            const int uwin = atoi(opt(argc, argv, "--usage-window", "5000")), rst = uwin > 0 && step % uwin < log_every;
            int ud = 0, ui = 0; for (int k = 0; k < K; k++) { ud += useb[k]; if (rst) useb[k] = 0; } for (int k = 0; k < KI && wib > 0; k++) { ui += useib[k]; if (rst) useib[k] = 0; }
            snprintf(line, sizeof line, "step %d dino %.4f ibot %.4f (%.0f tokens) obj %.4f region %.4f ink %.4f koleo %.4f gt %.4f var %.4f | H(t) %.2f H(mean) %.2f (max %.2f) H(ibot t) %.2f (max %.2f) | protos used %d / %d, ibot %d / %d | tt %.3f ema %.5f | gnorm %.3f | %.0f ms/step %.1f Mvox/s",
                     step, sums[0] / nsum, sums[5] / nsum, sums[6] / nsum, sums[7] / nsum, sums[8] / nsum, sums[9] / nsum, sums[1] / nsum, sums[2] / nsum, sums[3] / nsum, Hr, Hm, log((double)K), Hi, log((double)KI), ud, K, ui, KI, tt, mom, sums[4] / nsum, dt * 1e3, (double)(2 * g3 + nloc * l3) / dt / 1e6);
            if (SB > 0) dn_ssl_log(&SS, step, line, sizeof line);
            if (wtn > 0) { const size_t ln_ = strlen(line); snprintf(line + ln_, sizeof line - ln_, " | tnce %.4f top1 %.3f (negatives %d)",
                                     sums[17] > 0 ? sums[15] / sums[17] : 0.0, sums[17] > 0 ? sums[16] / sums[17] : 0.0, tqf); }
            if (wmo > 0 || wdn > 0) { const size_t ln_ = strlen(line); snprintf(line + ln_, sizeof line - ln_, " | nce: moco %.4f top1 %.3f (negatives %d) dense %.4f top1 %.3f (negatives %d)",
                                     sums[10] / nsum, sums[11] / nsum, mqf, sums[14] > 0 ? sums[12] / sums[14] : 0.0, sums[14] > 0 ? sums[13] / sums[14] : 0.0, dqf); }
            if (HS.uw) { const size_t ln_ = strlen(line); snprintf(line + ln_, sizeof line - ln_, " | dense: phase %.4f nrm %.4f w4 %.4f acc %.3f (%d views) inkp %.4f (%d views)",
                                     dn_nuw ? dsum[0] / dn_nuw : 0.0, dn_nuw ? dsum[1] / dn_nuw : 0.0, dn_nuw ? dsum[2] / dn_nuw : 0.0, dn_nuw ? dsum[3] / dn_nuw : 0.0, dn_nuw, dn_nik ? dsum[4] / dn_nik : 0.0, dn_nik);
                         memset(dsum, 0, sizeof dsum); dn_nuw = dn_nik = 0;
                         for (int k = 0; k < Dd->nsrc; k++) { double *q = dsrc + (size_t)k * 4; if (q[0] <= 0) continue; const size_t ln2 = strlen(line);
                             snprintf(line + ln2, sizeof line - ln2, " | %s nrm %.3f w4 %.3f acc %.3f (%.0f)", Dd->src[k].name, q[1] / q[0], q[2] / q[0], q[3] / q[0], q[0]); q[0] = q[1] = q[2] = q[3] = 0; } }
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
                      for (int k = 0; k < 3; k++) {   /* then the projection heads present (MoCo, dense InfoNCE, token InfoNCE: student, teacher) */
                          const dr_mlp2 *L = k == 2 ? &TPH : k ? &DP : &MP; float *P_ = k == 2 ? tp : k ? dp : mp, *T_ = k == 2 ? tpt : k ? dpt : mpt; if (!P_) continue;
                          float *h = malloc(L->n * 4); nn_d2h(h, P_, L->n * 4); fwrite(h, 4, L->n, hf); nn_d2h(h, T_, L->n * 4); fwrite(h, 4, L->n, hf); free(h);
                      }
                      fclose(hf); }
            char lp[1200]; snprintf(lp, sizeof lp, "%s/last.ckpt", outd);
            { char prev[1300]; const ssize_t pl = readlink(lp, prev, sizeof prev - 1); const int keep = atoi(opt(argc, argv, "--keep-every", "0"));   /* rotation: the previous checkpoint goes unless a milestone */
              if (pl > 0 && keep > 0) { prev[pl] = 0; int ps_ = 0; if (sscanf(prev, "step%d.ckpt", &ps_) == 1 && ps_ != step && ps_ % keep) { char a1[2600], a2[2600], a3[2600]; snprintf(a1, sizeof a1, "%s/%s", outd, prev); snprintf(a2, sizeof a2, "%s.heads", a1); snprintf(a3, sizeof a3, "%s.protos", a1); unlink(a1); unlink(a2); unlink(a3); snprintf(a3, sizeof a3, "%s.ssl", a1); unlink(a3); } } }
            unlink(lp); if (symlink(cp + strlen(outd) + 1, lp)) {}
            if (SB > 0) dn_ssl_io(&SS, cp, 1);
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
                        "         [--attn-blocks 0 (transformer blocks on the bottleneck)] [--attn-pe peg|rope|none] [--attn-reg 0] [--attn-mlp 0 (8/3 width)] [--attn-vres 0 (1: value residual)]\n"
                        "         [--attn-core 1 (attention products: 0 fp32, 1 fp16 / 2 MX-fp8 tensor cores)] [--attn-keep 0|1 (keep each block's forward, no recompute)]\n"
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
    if (ED != 16 && ED != 32 && !(ED == 64 && atof(opt(argc, argv, "--ssl-distill", "0")) > 0)) { fprintf(stderr, "dino-train: --dim 16 or 32 (64 with --ssl-distill)\n"); return 2; }
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
    /* the fp16 gradient scale (train's --gscale, default 1024): the backward keeps activation / logit gradients scaled by it and divides the
       parameter gradients back. Unset it was 1, and the dense heads' per-voxel loss gradients (~1e-9 on a densely labelled Paris 4 window)
       underflowed fp16 to exactly zero: those heads never learned there */
    if (!fp32) nn_set_grad_scale((float)atof(opt(argc, argv, "--gscale", "1024")));
    if (load_sources(srcf, Dd, nthreads)) return 1;
    const int axis_in = atoi(opt(argc, argv, "--axis-input", "0")); const float ars = (float)atof(opt(argc, argv, "--axis-rs", "0.001"));   /* the axis-distance channel and its scale */
    if (axis_in && !recipe) for (int i = 0; i < Dd->nsrc; i++) if (!Dd->src[i].axz) { fprintf(stderr, "dino-train: --axis-input needs \"axis\" in every source (%s)\n", Dd->src[i].name); return 2; }
    unet_cfg cfg = {0}; cfg.cin = axis_in ? (recipe ? 5 : 2) : 1; cfg.cout = ED; cfg.G = 8;
    if (recipe && atoi(opt(argc, argv, "--mask-input", "0"))) {   /* a mask channel (iBOT: masked cells CT 0 and mask 1; 0 elsewhere) */
        if (axis_in) { fprintf(stderr, "dino-train: --mask-input with --axis-input is not supported\n"); return 2; }
        cfg.cin = 2;
    }   /* (the recipe: CT, train's radial (3), valid; else CT, axis distance) */
    if (recipe && (atof(opt(argc, argv, "--unwrap2", "0")) > 0 || atof(opt(argc, argv, "--ink-prob", "0")) > 0)) {   /* dense heads after the embedding (dn_uw) */
        if (ED != 32) { fprintf(stderr, "dino-train --recipe: the dense heads need --dim 32\n"); return 2; }
        cfg.cout = (ED + 12 + 15) / 16 * 16;
    } cfg.down_norm = fp32 ? 0 : 1;
    { char *t = strdup(opt(argc, argv, "--widths", "16,32,64,96,128,192")); for (char *q = strtok(t, ","); q && cfg.nlev < UNET_MAXLEV; q = strtok(nullptr, ",")) cfg.widths[cfg.nlev++] = atoi(q); free(t); }
    if (P % (1 << (cfg.nlev - 1))) { fprintf(stderr, "dino-train: P must be a multiple of %d\n", 1 << (cfg.nlev - 1)); return 2; }
    {   /* the bottleneck transformer stage (src/attn.h) */
        cfg.attn_blocks = atoi(opt(argc, argv, "--attn-blocks", "0")); cfg.attn_reg = atoi(opt(argc, argv, "--attn-reg", "0")); cfg.attn_mlp = atoi(opt(argc, argv, "--attn-mlp", "0")); cfg.attn_vres = atoi(opt(argc, argv, "--attn-vres", "0"));
        const char *pe = opt(argc, argv, "--attn-pe", "peg");
        cfg.attn_pe = !strcmp(pe, "peg") ? 0 : !strcmp(pe, "rope") ? 1 : !strcmp(pe, "none") ? 2 : -1;
        if (cfg.attn_pe < 0) { fprintf(stderr, "dino-train: --attn-pe peg|rope|none\n"); return 2; }
    }
    if (opt(argc, argv, "--attn-out-init", nullptr)) setenv("UFSM_ATTN_OUT_INIT", opt(argc, argv, "--attn-out-init", "0"), 1);   /* (unet_init: the attention out projections at s x std, not 0) */
    /* the attention stage's core: 0 fp32 CUDA cores, 1 fp16 tensor cores, 2 MX-fp8 tensor cores (absent: env UFSM_ATTN_CORE, else 1);
       --attn-keep 1: each block's training forward kept for its backward (no recompute: about 6% faster with 6 blocks at 4096 tokens, + the blocks' forward tensors, ~65 MB per block at
       C 288 x 4097 tokens; absent: env UFSM_ATTN_KEEP, else 0) */
    if (opt(argc, argv, "--attn-core", nullptr)) at_set_core(atoi(opt(argc, argv, "--attn-core", "1")));
    if (opt(argc, argv, "--attn-keep", nullptr)) unet_set_attn_keep(atoi(opt(argc, argv, "--attn-keep", "0")));
    unet *u = unet_create(&cfg); unet_init(u, seed); unet_head_bias_zero(u);
    {   /* --head-init S: the 1^3 head's initial weights x S (live and EMA): a small head lets a regression target (--ssl-distill) move the
           outputs in tens of steps instead of hundreds (Kaiming: |W| ~ 11 for 64 x 16, Muon / AdamW steps ~0.025 per step) */
        const float his = (float)atof(opt(argc, argv, "--head-init", "1"));
        if (his != 1.f) { size_t hw_, hb_; unet_head_offsets(u, &hw_, &hb_); nn_scale((float *)unet_live_params(u) + hw_, his, hb_ - hw_); nn_scale((float *)unet_ema_params(u) + hw_, his, hb_ - hw_); }
    }   /* (no shared -2 bias direction in the embeddings) */
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
    R->nslots = 6;
    for (int i = 0; i < R->nslots; i++) {
        dn_sample *S = &R->slot[i]; S->ca = malloc(p3); S->cb = malloc(p3);
        S->gidx = malloc((size_t)NG * 4); S->gcode = malloc((size_t)NG * 4); S->gxyz = malloc((size_t)NG * 12);   /* labelled + air */ S->sidx = malloc((size_t)NS * 4); S->ayx = malloc((size_t)P * 8);
    }
    R->nworkers = nwork > DN_MAXW ? DN_MAXW : nwork;
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
    if (cfg.cin == 5 && opt(argc, argv, "--axis", nullptr)) { axz = load_axis(opt(argc, argv, "--axis", ""), atof(opt(argc, argv, "--axis-scale", "1")), z3_meta_of(ct)->shape[0]); if (!axz) return 2; }   /* (optional: else valid 0) */
    if (cfg.cin == 2 && opt(argc, argv, "--axis", nullptr)) { axz = load_axis(opt(argc, argv, "--axis", ""), atof(opt(argc, argv, "--axis-scale", "1")), z3_meta_of(ct)->shape[0]); if (!axz) return 2; }
    if (cfg.cin == 2 && !axz) fprintf(stderr, "dino-embed: 2-input net without --axis: the second channel is the --mask-input channel (all zero: nothing masked)\n");
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

/* ufsm dino-seamprobe CKPT SOURCES.json: does the attention stage's token (layer-normed, the iBOT head input) encode the absolute position
   mod 16 -- the phase of the CT mirror's 16^3 codec blocks (a possible SSL shortcut)? Un-augmented 128^3 crops at random origins of the
   unlabelled sources; per token cell the phase (origin + cell edge * g) mod 16 per axis as (cos, sin) of 2 pi phase / 16; a ridge regression
   (closed form, double) fit on the first 3/4 of the crops, R^2 on the rest. A random-init checkpoint is the baseline. */
int cmd_dino_seamprobe(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm dino-seamprobe CKPT SOURCES.json [--crops 40] [--P 128] [--per-crop 512] [--seed 1] [--gpu 0]\n"); return 2; }
    const int ncrop = atoi(opt(argc, argv, "--crops", "40")), P = atoi(opt(argc, argv, "--P", "128")), pc = atoi(opt(argc, argv, "--per-crop", "512"));
    uint64_t rs = (uint64_t)atoll(opt(argc, argv, "--seed", "1")) * 0x9E3779B97F4A7C15ull + 3;
    dn_precision(0);
    if (nn_init(atoi(opt(argc, argv, "--gpu", "0")))) return 1;
    dn_precision_after_init(0);
    unet_cfg cfg; int st = 0; if (unet_peek(argv[2], &cfg, &st)) { fprintf(stderr, "seamprobe: cannot read %s\n", argv[2]); return 1; }
    if (!cfg.attn_blocks) { fprintf(stderr, "seamprobe: the net has no attention stage\n"); return 2; }
    unet *u = unet_create(&cfg); if (unet_load(u, argv[2]) < 0) return 1;
    unet_use_params(u, unet_ema_params(u));
    unet_build(u, (shape5){1, cfg.cin, P, P, P}, 1);
    size_t jn = 0; char *jt = nullptr; { FILE *f = fopen(argv[3], "rb"); if (!f) return 1; fseek(f, 0, SEEK_END); jn = (size_t)ftell(f); fseek(f, 0, SEEK_SET); jt = malloc(jn + 1); if (fread(jt, 1, jn, f) != jn) return 1; fclose(f); jt[jn] = 0; }
    json *J = json_parse(jt, jn); const json *srcs = J ? json_get(J, "sources") : nullptr; if (!srcs) return 1;
    z3 *cts[128]; float cen[128], sca[128]; int ns = 0;
    for (size_t i = 0; i < srcs->n && ns < 128; i++) {
        const json *s = json_at(srcs, i); if (!json_get(s, "ct_only")) continue;
        char key[1024]; snprintf(key, sizeof key, "%s", json_str(json_get(s, "ct"), ""));
        if (json_get(s, "min_level")) snprintf(key, sizeof key, "%s/%d", json_str(json_get(s, "ct"), ""), (int)json_num(json_get(s, "min_level"), 0));
        store *sto = store_open(json_str(json_get(s, "root"), "")); z3 *z = sto ? z3_open(sto, key, json_str(json_get(s, "cache"), nullptr)) : nullptr; if (!z) continue;
        const json *nm = json_get(s, "norm"); cen[ns] = (float)json_num(json_get(nm, "center"), 100); sca[ns] = (float)json_num(json_get(nm, "scale"), 40); cts[ns++] = z;
    }
    if (!ns) { fprintf(stderr, "seamprobe: no unlabelled (ct_only) sources\n"); return 1; }
    const int G = P >> (cfg.nlev - 1), E = P / G, C = cfg.widths[cfg.nlev - 1], D = C + 1, T = 6;
    const size_t p3 = (size_t)P * P * P; uint8_t *ub = malloc(p3); float *xh = calloc(p3 * cfg.cin, 4), *xd = nn_malloc(p3 * cfg.cin * 4), *tk = malloc((size_t)(G * G * G + 8) * C * 4);
    double *XtX = calloc((size_t)D * D, 8), *XtY = calloc((size_t)D * T, 8); double *Fte = malloc((size_t)ncrop * pc * D * 8), *Yte = malloc((size_t)ncrop * pc * T * 8); long nte = 0;
    const int ntr = ncrop * 3 / 4;
    for (int c = 0; c < ncrop; c++) {
        const int si = (int)(rng_next(&rs) % (uint64_t)ns); const z3_meta *m = z3_meta_of(cts[si]); int64_t o[3];
        for (int tries = 0; ; tries++) {
            for (int d = 0; d < 3; d++) o[d] = (int64_t)(rng_u(&rs) * (double)(m->shape[d] - P));
            const int64_t n3[3] = {P, P, P}; if (z3_read(cts[si], o, n3, ub, 8)) return 1;
            size_t nz = 0; for (size_t i = 0; i < p3; i += 11) nz += ub[i] > cen[si]; if (nz * 11 * 10 > p3 || tries > 50) break;   /* >= 10% papyrus */
        }
        for (size_t i = 0; i < p3; i++) xh[i] = ((float)ub[i] - cen[si]) / sca[si];
        nn_h2d(xd, xh, p3 * cfg.cin * 4);
        unet_forward(u, xd, (shape5){1, cfg.cin, P, P, P}, 1);
        int S = 0, nreg = 0, Ct = 0; const float *td = unet_tokens(u, nullptr, &S, &Ct, &nreg); if (!td || Ct != C) { fprintf(stderr, "seamprobe: no tokens\n"); return 1; }
        nn_d2h(tk, td, (size_t)S * C * 4);
        for (int k = 0; k < pc; k++) {
            const int g = (int)(rng_next(&rs) % (uint64_t)(G * G * G)), gz = g / (G * G), gy = g / G % G, gx = g % G; const float *row = tk + (size_t)(nreg + g) * C;
            double f[1025], y[6], mu = 0, v = 0; for (int e = 0; e < C; e++) mu += row[e] / C; for (int e = 0; e < C; e++) v += (row[e] - mu) * (row[e] - mu) / C;
            for (int e = 0; e < C; e++) f[e] = (row[e] - mu) / sqrt(v + 1e-6); f[C] = 1;   /* (layer-normed, + bias) */
            const int64_t ph[3] = {(o[0] + (int64_t)E * gz) % 16, (o[1] + (int64_t)E * gy) % 16, (o[2] + (int64_t)E * gx) % 16};
            for (int d = 0; d < 3; d++) { y[2 * d] = cos(2 * M_PI * ph[d] / 16); y[2 * d + 1] = sin(2 * M_PI * ph[d] / 16); }
            if (c < ntr) { for (int a = 0; a < D; a++) { for (int b = 0; b < D; b++) XtX[(size_t)a * D + b] += f[a] * f[b]; for (int t = 0; t < T; t++) XtY[(size_t)a * T + t] += f[a] * y[t]; } }
            else { memcpy(Fte + (size_t)nte * D, f, D * 8); memcpy(Yte + (size_t)nte * T, y, T * 8); nte++; }
        }
    }
    const double lam = atof(opt(argc, argv, "--ridge", "1"));
    for (int a = 0; a < C; a++) XtX[(size_t)a * D + a] += lam;
    /* Cholesky solve XtX W = XtY */
    double *Lc = malloc((size_t)D * D * 8); memcpy(Lc, XtX, (size_t)D * D * 8);
    for (int j = 0; j < D; j++) { double s = Lc[(size_t)j * D + j]; for (int k = 0; k < j; k++) s -= Lc[(size_t)j * D + k] * Lc[(size_t)j * D + k]; if (s <= 0) s = 1e-12; Lc[(size_t)j * D + j] = sqrt(s);
        for (int i = j + 1; i < D; i++) { double t = Lc[(size_t)i * D + j]; for (int k = 0; k < j; k++) t -= Lc[(size_t)i * D + k] * Lc[(size_t)j * D + k]; Lc[(size_t)i * D + j] = t / Lc[(size_t)j * D + j]; } }
    double *W = malloc((size_t)D * T * 8);
    for (int t = 0; t < T; t++) { double *z = malloc(D * 8);
        for (int i = 0; i < D; i++) { double s = XtY[(size_t)i * T + t]; for (int k = 0; k < i; k++) s -= Lc[(size_t)i * D + k] * z[k]; z[i] = s / Lc[(size_t)i * D + i]; }
        for (int i = D - 1; i >= 0; i--) { double s = z[i]; for (int k = i + 1; k < D; k++) s -= Lc[(size_t)k * D + i] * W[(size_t)k * T + t]; W[(size_t)i * T + t] = s / Lc[(size_t)i * D + i]; }
        free(z); }
    double r2 = 0; static const char *nm[6] = {"z cos", "z sin", "y cos", "y sin", "x cos", "x sin"};
    fprintf(stderr, "seamprobe %s: %d crops (%d fit / %d test), %ld test tokens, token edge %d voxels, %d channels\n", argv[2], ncrop, ntr, ncrop - ntr, nte, E, C);
    for (int t = 0; t < T; t++) {
        double my = 0, ss = 0, se = 0; for (long r = 0; r < nte; r++) my += Yte[r * T + t] / nte;
        for (long r = 0; r < nte; r++) { double p = 0; for (int a = 0; a < D; a++) p += Fte[r * D + a] * W[(size_t)a * T + t]; se += (Yte[r * T + t] - p) * (Yte[r * T + t] - p); ss += (Yte[r * T + t] - my) * (Yte[r * T + t] - my); }
        const double r = ss > 0 ? 1 - se / ss : 0; r2 += r / T; fprintf(stderr, "  %-6s R^2 %.4f\n", nm[t], r);
    }
    fprintf(stderr, "seamprobe: mean test R^2 %.4f (0: no phase information; 1: the token encodes the absolute 16-phase)\n", r2);
    return 0;
}
