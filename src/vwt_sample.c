/* vwt ray sampler (src/vwt.h "sampler"): random ray-aligned slabs + winding-phase targets from stored-phase GT rasters.
   Sources JSON (the trainer's sources format, e.g. /vesuvius/ufsm/wm/sources-a0h.json): "sources": [{name, root, ct, um, min_level |
   max_level, axis, weight, targets.band.phase {root, key, level, phase_sign}, holdout [z y x nz ny nx L0], holdout_file}].
   - The CT level of a source: min_level (else 0); the GT phase grid: its "level" of the same pyramid (CT voxels per GT voxel 2^(level - lv)).
   - Sources with weight <= 1e-3 (holdout dummies) are never drawn; their holdout boxes count for every source of the same CT.
   - holdout_file boxes (wbench holdouts.json, {name, ct, l0}) of the source's CT, grown by 0.5 mm.
   - Top level "exclude_rays": {"file": relative_windings.json, "ct": <ct key of the source whose level-2 voxels the points use>, "r": 52}:
     voxels within r CT voxels of a consecutive-wrap pair segment get pw 0 (the ray-check eval).
   Slabs whose source bbox touches any holdout box are redrawn (no CT and no label from there).
   Per slab: a random source (by weight), a random voxel of a random stored GT chunk (outside the holdouts), the direction = the
   horizontal radial from the axis at its z, tilted by <= tilt_deg in a random azimuth, inward with p 0.5 (flags bit 0); the point on the
   central column at a random k in [32, L - 32). Frame as villawind.c frame_of. CT trilinear (rounded) inside the volume (va), else 0.
   ph: the GT code / 252 (x phase_sign) nearest at each voxel, unwrapped by a 6-neighbour flood fill (steps < 0.25 turn only); components
   < 64 voxels dropped; negated on inward rays so ph rises along +k; comp ids from 1 (uint16, 0 none). pw = 1 on comp, 0 in the tubes.
   A slab with < 1% labelled voxels is redrawn (up to 16 times). */
#include "vwt.h"
#include "vwt_aug.h"
#include "zcache.h"
int jpeg_write_rgb(const char *path, int w, int h, const uint8_t *rgb, int quality);   /* (imgops.h) */
#include "json.h"
#include "sources.h"
#include "store.h"
#include "zarr3.h"
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char name[128]; z3 *ct, *ph, *sh; int lv, plv, psign, shlv; double l0s, um0; axis ax; int hasax; double weight;
    int64_t *chunks; int64_t nch, cgrid[3], csz[3];      /* stored GT chunks (outside holdouts) */
    uint8_t *ok;                                         /* [cgrid]: 1 = drawable chunk (in chunks) */
    double (*box)[6]; int nbox;                          /* holdout boxes, CT voxels zyx */
    double (*seg)[6]; int nseg; double seg_r;            /* exclusion tubes, CT voxels zyx */
} vws_src;

typedef struct { vws_sample s; int state; } vws_slot;   /* 0 free, 1 being filled, 2 ready, 3 handed out */
struct vws {
    vws_src *S; int ns; double wsum; int T, L; float tilt; int train; unsigned seed; int has_shell;
    vws_slot *slot; int nslot; pthread_t *th; int nth; _Atomic int stop; _Atomic long long next_id;
    zcache *zc; int burst, burst_r;   /* the shared decoded-chunk cache (UFSM_VWS_CACHE_GB, 4; 0 off); training draws in bursts of `burst` slabs near one
                                anchor chunk (UFSM_VWS_BURST, 8; 1 off) so neighbouring slabs share decoded chunks */
    pthread_mutex_t mu; pthread_cond_t cv; int last;
};

/* ---- augmentation (src/vwt_aug.h; DA): vwt-train passes its argv here before vws_open; training slabs only ---- */
static vwa_cfg g_vwa; static int g_vwa_ok;
static double g_mv_p = 0, g_mv_cone = 30, g_mv_shift = 16;   /* multiview pairs (vwt.h vws_sample.has_pair) */
void vws_set_args(int argc, char **argv) {
    for (int i = argc - 2; i >= 1; i--) if (!strcmp(argv[i], "--mv-p")) { g_mv_p = atof(argv[i + 1]); break; }
    for (int i = argc - 2; i >= 1; i--) if (!strcmp(argv[i], "--mv-cone")) { g_mv_cone = atof(argv[i + 1]); break; }
    for (int i = argc - 2; i >= 1; i--) if (!strcmp(argv[i], "--mv-shift")) { g_mv_shift = atof(argv[i + 1]); break; }
    if (g_mv_p > 0) fprintf(stderr, "vws: multiview pairs p %.2f, cone %.0f deg, centre shift <= %.0f voxels\n", g_mv_p, g_mv_cone, g_mv_shift);
    g_vwa = vwa_cfg_default();
    if (vwa_cfg_parse(&g_vwa, argc, argv)) exit(2);
    g_vwa_ok = 1; vwa_cfg_print(&g_vwa);
}

/* ---- small helpers ---- */
static uint64_t vws_rng(uint64_t *s) { uint64_t z = (*s += 0x9e3779b97f4a7c15ull); z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull; z = (z ^ (z >> 27)) * 0x94d049bb133111ebull; return z ^ (z >> 31); }
static double vws_unif(uint64_t *s) { return (vws_rng(s) >> 11) * (1.0 / 9007199254740992.0); }
static void cross3(const double a[3], const double b[3], double c[3]) { c[0] = a[1] * b[2] - a[2] * b[1]; c[1] = a[2] * b[0] - a[0] * b[2]; c[2] = a[0] * b[1] - a[1] * b[0]; }
static char *slurp(const char *p, size_t *n) {
    FILE *f = fopen(p, "rb"); if (!f) return nullptr; fseek(f, 0, SEEK_END); long l = ftell(f); fseek(f, 0, SEEK_SET);
    char *t = malloc(l + 1); if (t && fread(t, 1, l, f) != (size_t)l) { free(t); t = nullptr; } fclose(f); if (t) t[l] = 0; *n = (size_t)l; return t;
}
static const char *base_of(const char *p) { const char *s = strrchr(p, '/'); return s ? s + 1 : p; }

/* the frame of villawind.c frame_of (xyz): d unit, ref = axis of d's smallest |component|, a = norm(ref x d), b = d x a,
   o = ray_origin - (T - 1) / 2 (a + b) */
void vws_frame(const double dir[3], const double ray_origin[3], int T, double o[3], double a[3], double b[3], double d[3]) {
    const double n = sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
    for (int c = 0; c < 3; c++) d[c] = dir[c] / n;
    double ref[3] = {0, 0, 0}; int m = 0; for (int c = 1; c < 3; c++) if (fabs(d[c]) < fabs(d[m])) m = c; ref[m] = 1;
    cross3(ref, d, a); const double na = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]); for (int c = 0; c < 3; c++) a[c] /= na;
    cross3(d, a, b);
    for (int c = 0; c < 3; c++) o[c] = ray_origin[c] - 0.5 * (T - 1) * (a[c] + b[c]);
}

/* the flood-fill unwrap of raw phases (turns in [0, 1), NAN none) over the [T][T][L] grid: ph (NAN none), comp (0 none); components
   smaller than minsz are dropped. Returns the number of kept components. */
/* kinc (nullable): an exact phase increment from voxel v - 1 to v along k (NAN: none); such k links join regardless of size (deformed
   slabs: a collapsed gap can step > 1/4 turn between neighbouring voxels, kinc integrates the source GT along the segment) */
int vws_unwrap_k(const float *raw, const float *kinc, int T, int L, int minsz, float *ph, uint16_t *comp);
int vws_unwrap(const float *raw, int T, int L, int minsz, float *ph, uint16_t *comp) { return vws_unwrap_k(raw, nullptr, T, L, minsz, ph, comp); }
int vws_unwrap_k(const float *raw, const float *kinc, int T, int L, int minsz, float *ph, uint16_t *comp) {
    const size_t n = (size_t)T * T * L; int32_t *q = malloc(n * sizeof *q); uint32_t *lab = calloc(n, sizeof *lab);
    if (!q || !lab) { free(q); free(lab); return -1; }
    for (size_t k = 0; k < n; k++) ph[k] = NAN;
    uint32_t id = 0; int kept = 0; static const int di[6] = {1, -1, 0, 0, 0, 0}, dj[6] = {0, 0, 1, -1, 0, 0}, dk[6] = {0, 0, 0, 0, 1, -1};
    for (size_t s = 0; s < n; s++) {
        if (isnan(raw[s]) || lab[s]) continue;
        id++; size_t h = 0, t = 0; q[t++] = (int32_t)s; lab[s] = id; ph[s] = raw[s];
        while (h < t) {
            const size_t c = (size_t)q[h++]; const int i = (int)(c / ((size_t)T * L)), j = (int)(c / L % T), k = (int)(c % L);
            for (int e = 0; e < 6; e++) {
                const int ii = i + di[e], jj = j + dj[e], kk = k + dk[e];
                if (ii < 0 || jj < 0 || kk < 0 || ii >= T || jj >= T || kk >= L) continue;
                const size_t u = ((size_t)ii * T + jj) * L + kk; if (lab[u] || isnan(raw[u])) continue;
                double dp = raw[u] - raw[c]; dp -= floor(dp + 0.5);
                const float kx = !kinc ? NAN : e == 4 ? kinc[u] : e == 5 ? -kinc[c] : NAN;
                if (!isnan(kx)) dp = kx; else if (fabs(dp) >= 0.25) continue;
                lab[u] = id; ph[u] = (float)(ph[c] + dp); q[t++] = (int32_t)u;
            }
        }
        if ((long)t < minsz || kept >= 65535) { for (size_t r = 0; r < t; r++) { ph[q[r]] = NAN; lab[q[r]] = UINT32_MAX; } }
        else { kept++; for (size_t r = 0; r < t; r++) lab[q[r]] = UINT32_MAX - 1 - (uint32_t)kept; }
    }
    for (size_t s = 0; s < n; s++) comp[s] = lab[s] >= UINT32_MAX - 65536 && lab[s] < UINT32_MAX ? (uint16_t)(UINT32_MAX - 1 - lab[s]) : 0;
    free(q); free(lab); return kept;
}

static int vws_seg_hit(const double (*seg)[6], const int *idx, int nidx, double r2, const double p[3]) {   /* p zyx */
    for (int m = 0; m < nidx; m++) {
        const double *s = seg[idx[m]]; double ab[3], ap[3], l2 = 0, t = 0;
        for (int d = 0; d < 3; d++) { ab[d] = s[3 + d] - s[d]; ap[d] = p[d] - s[d]; l2 += ab[d] * ab[d]; t += ab[d] * ap[d]; }
        t = l2 > 0 ? fmin(1, fmax(0, t / l2)) : 0; double q = 0;
        for (int d = 0; d < 3; d++) { const double e = ap[d] - t * ab[d]; q += e * e; }
        if (q <= r2) return 1;
    }
    return 0;
}

/* ---- one slab ---- */
typedef struct { int bsrc, bleft; int64_t bck; uint8_t *ct, *gt, *shb; size_t ccap, gcap, scap; float *raw, *q, *kinc; size_t n; int *sidx; int nsidx; uint8_t *arena; } vws_work;
#define VWS_ARENA ((size_t)64 << 21)   /* the crush map's CT tile cache per worker (64 x 2 MiB) */
typedef struct { crush_cache *cc; crush_layer L; } vws_ctu;
static double vws_ct_xyz(void *ud, const double x[3]) { vws_ctu *u = ud; const double p[3] = {x[2], x[1], x[0]}; return crush_linear_at(u->cc, &u->L, p); }
/* the stored GT phase (turns, wrapped, sign applied) nearest to CT point x (xyz), or -1 (none / outside the read GT box) */
static double vws_gt_at(const uint8_t *gt, const int64_t go[3], const int64_t gn[3], double f, int psign, const double x[3]) {
    const int64_t gz = (int64_t)floor((x[2] + 0.5) / f) - go[0], gy = (int64_t)floor((x[1] + 0.5) / f) - go[1], gx = (int64_t)floor((x[0] + 0.5) / f) - go[2];
    if (gz < 0 || gy < 0 || gx < 0 || gz >= gn[0] || gy >= gn[1] || gx >= gn[2]) return -1;
    const uint8_t c = gt[((size_t)gz * gn[1] + gy) * gn[2] + gx]; if (c >= 252) return -1;
    double ph = psign * c / 252.0; return ph - floor(ph);
}

/* the source bbox (xyz) of a slab whose voxel (i, j, k) samples q (xyz, [T][T][L][3]) */
static void vws_qbox(const float *q, size_t n, double lo[3], double hi[3]) {
    for (int c = 0; c < 3; c++) { lo[c] = 1e300; hi[c] = -1e300; }
    for (size_t v = 0; v < n; v++) for (int c = 0; c < 3; c++) { lo[c] = fmin(lo[c], q[3 * v + c]); hi[c] = fmax(hi[c], q[3 * v + c]); }
}
/* UFSM_VWS_PROF=1: per-stage wall time of vws_fill summed over the workers, printed by vws_close */
enum { VP_FRAME, VP_AUGMAP, VP_LATTICE, VP_QBOX, VP_CT, VP_GT, VP_SAMPLE, VP_KINC, VP_UNWRAP, VP_PW, VP_INTENS, VP_N };
static const char *vp_name[VP_N] = {"frame", "aug map", "aug lattice", "qbox+holdout", "CT read", "GT read", "sample", "exact k", "unwrap", "weights", "intensity"};
enum { VR_HOLD, VR_EMPTY, VR_READ, VR_SHELL, VR_NLAB, VR_NC, VR_KEPT, VR_N };
static const char *vr_name[VR_N] = {"holdout", "empty", "read", "shell", "labels", "components", "weights"};
static _Atomic long long vp_ns[VP_N], vr_n[VR_N], vp_slabs, vp_tries, vp_geo; static int vp_on = -1;
#define REJ(r) { if (vp_on) atomic_fetch_add(&vr_n[r], 1); continue; }
static long long vp_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1000000000ll + t.tv_nsec; }
#define VP(k) do { if (vp_on) { const long long t_ = vp_now(); atomic_fetch_add(&vp_ns[k], t_ - vp_t); vp_t = t_; } } while (0)
static void vp_print(void) {
    if (vp_on <= 0 || !vp_slabs) return; double tot = 0; for (int k = 0; k < VP_N; k++) tot += vp_ns[k];
    fprintf(stderr, "vws prof: %lld slabs (%lld draws, %lld deformed), %.2f thread-s per slab:", vp_slabs, vp_tries, vp_geo, tot / 1e9 / vp_slabs);
    for (int k = 0; k < VP_N; k++) fprintf(stderr, " %s %.3f", vp_name[k], vp_ns[k] / 1e9 / vp_slabs);
    fprintf(stderr, "; rejected:"); for (int k = 0; k < VR_N; k++) fprintf(stderr, " %s %lld", vr_name[k], vr_n[k]); fprintf(stderr, "\n");
}
/* the multiview pair of an undeformed slab `out` (vwt.h): direction within g_mv_cone of d, centre shifted <= g_mv_shift per axis; CT
   only (trilinear, as the slab), outside any holdout box; 0 = filled */
static int vws_pair(vws *V, uint64_t *rs, vws_work *w, vws_src *S, vws_sample *out) {
    const int T = V->T, L = V->L; const size_t n = (size_t)T * T * L;
    const double th = acos(1 - vws_unif(rs) * (1 - cos(g_mv_cone * M_PI / 180))), ph = vws_unif(rs) * 2 * M_PI;   /* uniform on the cap */
    double dir[3]; for (int e = 0; e < 3; e++) dir[e] = cos(th) * out->d[e] + sin(th) * (cos(ph) * out->a[e] + sin(ph) * out->b[e]);
    double c[3]; for (int e = 0; e < 3; e++) c[e] = out->o[e] + 0.5 * (T - 1) * (out->a[e] + out->b[e]) + 0.5 * (L - 1) * out->d[e] + (2 * vws_unif(rs) - 1) * g_mv_shift;
    double ro[3]; for (int e = 0; e < 3; e++) ro[e] = c[e] - 0.5 * (L - 1) * dir[e];
    vws_frame(dir, ro, T, out->po, out->pa, out->pb, out->pd);
    double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
    for (int k = 0; k < 8; k++) for (int e = 0; e < 3; e++) {
        const double v = out->po[e] + (k & 1 ? T - 1 : 0) * out->pa[e] + (k & 2 ? T - 1 : 0) * out->pb[e] + (k & 4 ? L - 1 : 0) * out->pd[e];
        lo[e] = fmin(lo[e], v); hi[e] = fmax(hi[e], v);
    }
    for (int b = 0; b < S->nbox; b++) { const double *B = S->box[b]; if (hi[2] >= B[0] && lo[2] < B[0] + B[3] && hi[1] >= B[1] && lo[1] < B[1] + B[4] && hi[0] >= B[2] && lo[0] < B[2] + B[5]) return -1; }
    const int64_t *Vs = z3_meta_of(S->ct)->shape; int64_t bo[3], bn[3];
    for (int e = 0; e < 3; e++) { const int x = 2 - e; int64_t a = (int64_t)floor(lo[x]) - 1, b = (int64_t)ceil(hi[x]) + 2; a = a < 0 ? 0 : a; b = b > Vs[e] ? Vs[e] : b; if (b - a < 2) return -1; bo[e] = a; bn[e] = b - a; }
    const size_t need = (size_t)bn[0] * bn[1] * bn[2];
    if (need > w->ccap) { free(w->ct); w->ct = malloc(need); w->ccap = need; if (!w->ct) { w->ccap = 0; return -1; } }
    if (zcache_read(V->zc, S->ct, bo, bn, w->ct)) return -1;
    size_t nin = 0;
    for (int i = 0; i < T; i++) for (int j = 0; j < T; j++) for (int k = 0; k < L; k++) {
        const size_t v = ((size_t)i * T + j) * L + k;
        const double x = out->po[0] + i * out->pa[0] + j * out->pb[0] + k * out->pd[0], y = out->po[1] + i * out->pa[1] + j * out->pb[1] + k * out->pd[1],
                     z = out->po[2] + i * out->pa[2] + j * out->pb[2] + k * out->pd[2];
        if (!(x >= 0 && y >= 0 && z >= 0 && x <= Vs[2] - 1 && y <= Vs[1] - 1 && z <= Vs[0] - 1)) { out->pim[v] = 0; out->pva[v] = 0; continue; }
        const double lz = z - bo[0], ly = y - bo[1], lx = x - bo[2];
        int iz = (int)floor(lz), iy = (int)floor(ly), ix = (int)floor(lx);
        iz = iz < 0 ? 0 : iz > bn[0] - 2 ? (int)bn[0] - 2 : iz; iy = iy < 0 ? 0 : iy > bn[1] - 2 ? (int)bn[1] - 2 : iy; ix = ix < 0 ? 0 : ix > bn[2] - 2 ? (int)bn[2] - 2 : ix;
        const double fz = lz - iz, fy = ly - iy, fx = lx - ix;
        const uint8_t *B = w->ct + ((size_t)iz * bn[1] + iy) * bn[2] + ix; const size_t sy = bn[2], sz = (size_t)bn[1] * bn[2];
        const double val = (1 - fz) * ((1 - fy) * ((1 - fx) * B[0] + fx * B[1]) + fy * ((1 - fx) * B[sy] + fx * B[sy + 1]))
                         + fz * ((1 - fy) * ((1 - fx) * B[sz] + fx * B[sz + 1]) + fy * ((1 - fx) * B[sz + sy] + fx * B[sz + sy + 1]));
        const int u = (int)lround(val); out->pim[v] = (uint8_t)(u < 0 ? 0 : u > 255 ? 255 : u); out->pva[v] = 1; nin++;
    }
    if (nin < n / 4) return -1;
    if (g_vwa_ok) vwa_intensity(&g_vwa, vws_rng(rs), out->pim, out->pva, T, L);
    return 0;
}
static int vws_fill(vws *V, uint64_t *rs, vws_work *w, vws_sample *out) {
    if (vp_on < 0) vp_on = getenv("UFSM_VWS_PROF") && atoi(getenv("UFSM_VWS_PROF")) > 0;
    long long vp_t = vp_on ? vp_now() : 0;
    const int T = V->T, L = V->L; const size_t n = (size_t)T * T * L;
    for (int attempt = 0; attempt < 64; attempt++) {
        /* source by weight */
        int si = 0; int64_t ck = -1;
        if (V->train && V->burst > 1 && w->bleft > 0) {   /* a burst: a drawable chunk within +-burst_r chunks of the anchor (UFSM_VWS_BURST_R, 0 = the anchor chunk; 8 tries) */
            vws_src *B = &V->S[w->bsrc]; const int64_t az = w->bck / (B->cgrid[1] * B->cgrid[2]), ay = w->bck / B->cgrid[2] % B->cgrid[1], ax = w->bck % B->cgrid[2];
            for (int t = 0; t < 8 && ck < 0; t++) {
                const int64_t R = V->burst_r, D = 2 * R + 1;
                const int64_t z = az + (int64_t)(vws_rng(rs) % (uint64_t)D) - R, y = ay + (int64_t)(vws_rng(rs) % (uint64_t)D) - R, x = ax + (int64_t)(vws_rng(rs) % (uint64_t)D) - R;
                if (z < 0 || y < 0 || x < 0 || z >= B->cgrid[0] || y >= B->cgrid[1] || x >= B->cgrid[2]) continue;
                const int64_t k = (z * B->cgrid[1] + y) * B->cgrid[2] + x; if (B->ok[k]) { ck = k; si = w->bsrc; }
            }
            w->bleft--;
        }
        if (ck < 0) {
            double r = vws_unif(rs) * V->wsum; for (; si < V->ns - 1; si++) { if (V->S[si].weight <= 1e-3) continue; r -= V->S[si].weight; if (r <= 0) break; }
            while (V->S[si].weight <= 1e-3 || !V->S[si].nch) si = (si + 1) % V->ns;
            ck = V->S[si].chunks[vws_rng(rs) % (uint64_t)V->S[si].nch];
            if (V->train && V->burst > 1) { w->bsrc = si; w->bck = ck; w->bleft = V->burst - 1; }
        }
        vws_src *S = &V->S[si];
        /* a random voxel of the chosen stored GT chunk (GT voxels zyx) -> CT voxels xyz */ const int64_t cz = ck / (S->cgrid[1] * S->cgrid[2]), cy = ck / S->cgrid[2] % S->cgrid[1], cx = ck % S->cgrid[2];
        const double f = ldexp(1, S->plv - S->lv), fd = S->ph ? f : ldexp(1, S->shlv - S->lv);   /* CT voxels per GT / drawn-raster voxel */
        const double pz = (cz * S->csz[0] + vws_unif(rs) * S->csz[0]) * fd, py = (cy * S->csz[1] + vws_unif(rs) * S->csz[1]) * fd, px = (cx * S->csz[2] + vws_unif(rs) * S->csz[2]) * fd;
        double ay = 0, ax = 0; if (S->hasax) { axis_at(&S->ax, pz * S->l0s, &ay, &ax); ay /= S->l0s; ax /= S->l0s; }
        out->axp[0] = ax; out->axp[1] = ay; out->axp[2] = pz; out->has_pair = 0; out->um = S->um0 * S->l0s;
        double rd[3] = {px - ax, py - ay, 0}; double rn = hypot(rd[0], rd[1]); if (rn < 1) { rd[0] = 1; rd[1] = 0; rn = 1; } rd[0] /= rn; rd[1] /= rn;
        /* tilt <= tilt_deg in a random azimuth: d = cos t r + sin t (cos phi z + sin phi (z x r)) */
        const double tt = vws_unif(rs) * V->tilt * M_PI / 180, phi = vws_unif(rs) * 2 * M_PI; const double zr[3] = {-rd[1], rd[0], 0};
        double dir[3] = {cos(tt) * rd[0] + sin(tt) * sin(phi) * zr[0], cos(tt) * rd[1] + sin(tt) * sin(phi) * zr[1], sin(tt) * cos(phi)};
        const int inward = vws_rng(rs) & 1; if (inward) for (int c = 0; c < 3; c++) dir[c] = -dir[c];
        const double nd = sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]); for (int c = 0; c < 3; c++) dir[c] /= nd;
        const int k0 = 32 + (int)(vws_rng(rs) % (uint64_t)(L - 64 > 1 ? L - 64 : 1));
        double ro[3] = {px - k0 * dir[0], py - k0 * dir[1], pz - k0 * dir[2]};
        vws_frame(dir, ro, T, out->o, out->a, out->b, out->d);
        /* the source lattice q (xyz); the geometric aug hook replaces it (src/vwt_aug.h): CT and labels are both read at q */
        if (vp_on) atomic_fetch_add(&vp_tries, 1);
        VP(VP_FRAME);
        {   /* early holdout test on the undeformed slab's bbox (no I/O, no map yet): a slab touching a holdout there is dropped at once */
            double lo0[3] = {1e300, 1e300, 1e300}, hi0[3] = {-1e300, -1e300, -1e300};
            for (int c = 0; c < 8; c++) for (int e = 0; e < 3; e++) {
                const double v = out->o[e] + (c & 1 ? T - 1 : 0) * out->a[e] + (c & 2 ? T - 1 : 0) * out->b[e] + (c & 4 ? L - 1 : 0) * out->d[e];
                lo0[e] = fmin(lo0[e], v); hi0[e] = fmax(hi0[e], v);
            }
            int bad0 = 0;
            for (int b = 0; b < S->nbox && !bad0; b++) { const double *B = S->box[b];
                bad0 = hi0[2] >= B[0] && lo0[2] < B[0] + B[3] && hi0[1] >= B[1] && lo0[1] < B[1] + B[4] && hi0[0] >= B[2] && lo0[0] < B[2] + B[5]; }
            if (bad0) REJ(VR_HOLD)
        }
        vwa *ga = nullptr; int geo = 0; out->T = T; out->L = L;
        if (g_vwa_ok && V->train && g_vwa.on && g_vwa.geo_p > 0) {
            if (!w->arena) w->arena = malloc(VWS_ARENA);
            crush_cache cc;
            if (w->arena && !crush_cache_init(&cc, w->arena, VWS_ARENA)) {
                vws_ctu u = {&cc, {S->ct, CRUSH_LINEAR, 0, 0, 1, {0, 0, 0}, nullptr}};
                ga = vwa_begin(&g_vwa, vws_rng(rs), out, vws_ct_xyz, &u);
                VP(VP_AUGMAP);
                if (ga) { vwa_src_lattice(ga, T, L, w->q); geo = 1; }
                VP(VP_LATTICE);
                vwa_end(ga); ga = nullptr; crush_cache_free(&cc);
            }
        }
        if (!geo) for (int i = 0; i < T; i++) for (int j = 0; j < T; j++) for (int k = 0; k < L; k++) {
            float *qq = w->q + 3 * (((size_t)i * T + j) * L + k);
            for (int c = 0; c < 3; c++) qq[c] = (float)(out->o[c] + i * out->a[c] + j * out->b[c] + k * out->d[c]);
        }
        VP(VP_LATTICE);
        double lo[3], hi[3]; vws_qbox(w->q, n, lo, hi);
        int bad = 0;   /* holdouts: no CT and no label from there */
        for (int b = 0; b < S->nbox && !bad; b++) {
            const double *B = S->box[b];   /* zyx */
            bad = hi[2] >= B[0] && lo[2] < B[0] + B[3] && hi[1] >= B[1] && lo[1] < B[1] + B[4] && hi[0] >= B[2] && lo[0] < B[2] + B[5];
        }
        if (bad) REJ(VR_HOLD)
        VP(VP_QBOX);
        /* CT box (zyx) */
        const int64_t *Vs = z3_meta_of(S->ct)->shape; int64_t bo[3], bn[3]; int empty = 0;
        for (int e = 0; e < 3; e++) { const int x = 2 - e; int64_t a = (int64_t)floor(lo[x]) - 1, b = (int64_t)ceil(hi[x]) + 2; a = a < 0 ? 0 : a; b = b > Vs[e] ? Vs[e] : b; if (b - a < 2) { empty = 1; a = 0; b = 2; } bo[e] = a; bn[e] = b - a; }
        if (empty) REJ(VR_EMPTY)
        const size_t need = (size_t)bn[0] * bn[1] * bn[2];
        if (need > w->ccap) { free(w->ct); w->ct = malloc(need); w->ccap = need; if (!w->ct) return -1; }
        if (zcache_read(V->zc, S->ct, bo, bn, w->ct)) { fprintf(stderr, "vws: CT read failed (%s): %s\n", S->name, z3_error()); REJ(VR_READ) }
        VP(VP_CT);
        /* GT box (zyx, GT voxels): nearest GT voxel of a CT point c: floor((c + 0.5) / f) */
        int64_t go[3] = {0, 0, 0}, gn[3] = {0, 0, 0};
        if (S->ph) {
            const int64_t *G = z3_meta_of(S->ph)->shape;
            for (int e = 0; e < 3; e++) { const int x = 2 - e; int64_t a = (int64_t)floor((lo[x] + 0.5) / f) - 1, b = (int64_t)floor((hi[x] + 0.5) / f) + 2; a = a < 0 ? 0 : a; b = b > G[e] ? G[e] : b; if (b <= a) { a = 0; b = 1; } go[e] = a; gn[e] = b - a; }
            const size_t gneed = (size_t)gn[0] * gn[1] * gn[2];
            if (gneed > w->gcap) { free(w->gt); w->gt = malloc(gneed); w->gcap = gneed; if (!w->gt) return -1; }
            if (zcache_read(V->zc, S->ph, go, gn, w->gt)) { fprintf(stderr, "vws: GT read failed (%s): %s\n", S->name, z3_error()); REJ(VR_READ) }
        }
        /* shell box (zyx, shell voxels), nearest as the GT */
        const double fs = ldexp(1, S->shlv - S->lv); int64_t so[3] = {0, 0, 0}, sn[3] = {0, 0, 0};
        if (S->sh) {
            const int64_t *H = z3_meta_of(S->sh)->shape;
            for (int e = 0; e < 3; e++) { const int x = 2 - e; int64_t a = (int64_t)floor((lo[x] + 0.5) / fs) - 1, b = (int64_t)floor((hi[x] + 0.5) / fs) + 2; a = a < 0 ? 0 : a; b = b > H[e] ? H[e] : b; if (b <= a) { a = 0; b = 1; } so[e] = a; sn[e] = b - a; }
            const size_t sneed = (size_t)sn[0] * sn[1] * sn[2];
            if (sneed > w->scap) { free(w->shb); w->shb = malloc(sneed); w->scap = sneed; if (!w->shb) return -1; }
            if (zcache_read(V->zc, S->sh, so, sn, w->shb)) { fprintf(stderr, "vws: shell read failed (%s): %s\n", S->name, z3_error()); REJ(VR_READ) }
        }
        VP(VP_GT);
        /* tubes near this slab */
        w->nsidx = 0; const double R = S->seg_r;
        for (int m = 0; m < S->nseg; m++) {
            const double *s = S->seg[m];
            if (fmax(s[0], s[3]) + R < lo[2] || fmin(s[0], s[3]) - R > hi[2] || fmax(s[1], s[4]) + R < lo[1] || fmin(s[1], s[4]) - R > hi[1] || fmax(s[2], s[5]) + R < lo[0] || fmin(s[2], s[5]) - R > hi[0]) continue;
            w->sidx = realloc(w->sidx, (w->nsidx + 1) * sizeof *w->sidx); w->sidx[w->nsidx++] = m;
        }
        size_t nlab = 0;
        for (size_t v = 0; v < n; v++) {
            const float *qq = w->q + 3 * v; const double x = qq[0], y = qq[1], z = qq[2];
            if (!(x >= 0 && y >= 0 && z >= 0 && x <= Vs[2] - 1 && y <= Vs[1] - 1 && z <= Vs[0] - 1)) { out->im[v] = 0; out->va[v] = 0; }
            else {
                const double lz = z - bo[0], ly = y - bo[1], lx = x - bo[2];
                int iz = (int)floor(lz), iy = (int)floor(ly), ix = (int)floor(lx);
                iz = iz < 0 ? 0 : iz > bn[0] - 2 ? (int)bn[0] - 2 : iz; iy = iy < 0 ? 0 : iy > bn[1] - 2 ? (int)bn[1] - 2 : iy; ix = ix < 0 ? 0 : ix > bn[2] - 2 ? (int)bn[2] - 2 : ix;
                const double fz = lz - iz, fy = ly - iy, fx = lx - ix;
                const uint8_t *B = w->ct + ((size_t)iz * bn[1] + iy) * bn[2] + ix; const size_t sy = bn[2], sz = (size_t)bn[1] * bn[2];
                const double val = (1 - fz) * ((1 - fy) * ((1 - fx) * B[0] + fx * B[1]) + fy * ((1 - fx) * B[sy] + fx * B[sy + 1]))
                                 + fz * ((1 - fy) * ((1 - fx) * B[sz] + fx * B[sz + 1]) + fy * ((1 - fx) * B[sz + sy] + fx * B[sz + sy + 1]));
                const int u = (int)lround(val); out->im[v] = (uint8_t)(u < 0 ? 0 : u > 255 ? 255 : u); out->va[v] = 1;
            }
            if (out->shell) {
                out->shell[v] = 0;
                if (S->sh) {
                    const int64_t hz = (int64_t)floor((z + 0.5) / fs) - so[0], hy = (int64_t)floor((y + 0.5) / fs) - so[1], hx = (int64_t)floor((x + 0.5) / fs) - so[2];
                    if (hz >= 0 && hy >= 0 && hx >= 0 && hz < sn[0] && hy < sn[1] && hx < sn[2] && w->shb[((size_t)hz * sn[1] + hy) * sn[2] + hx] != 255) out->shell[v] = 1;
                }
            }
            w->raw[v] = NAN;
            if (!S->ph) continue;
            const int64_t gz = (int64_t)floor((z + 0.5) / f) - go[0], gy = (int64_t)floor((y + 0.5) / f) - go[1], gx = (int64_t)floor((x + 0.5) / f) - go[2];
            if (gz < 0 || gy < 0 || gx < 0 || gz >= gn[0] || gy >= gn[1] || gx >= gn[2]) continue;
            const uint8_t c = w->gt[((size_t)gz * gn[1] + gy) * gn[2] + gx]; if (c >= 252) continue;
            double ph = S->psign * c / 252.0; ph -= floor(ph); w->raw[v] = (float)ph; nlab++;
        }
        VP(VP_SAMPLE);
        if (out->shell && w->nsidx) for (size_t v = 0; v < n; v++) if (out->shell[v]) {   /* no shell inside the ray-check tubes */
            const float *qq = w->q + 3 * v; const double p[3] = {qq[2], qq[1], qq[0]};
            if (vws_seg_hit(S->seg, w->sidx, w->nsidx, R * R, p)) out->shell[v] = 0;
        }
        if (!S->ph) {   /* shell-only source: no phase target */
            size_t ns = 0; for (size_t v = 0; v < n; v++) { out->ph[v] = NAN; out->comp[v] = 0; out->pw[v] = 0; ns += out->shell[v]; }
            if (ns < n / 2000) REJ(VR_SHELL)
            if (g_vwa_ok && V->train) vwa_intensity(&g_vwa, vws_rng(rs), out->im, out->va, T, L);
            out->T = T; out->L = L; out->src = si; out->flags = inward;
            return 0;
        }
        if (nlab < n / 100) REJ(VR_NLAB)
        if (geo) {   /* deformed: exact k increments through the source GT along each q segment (substeps <= 0.5 voxel) */
            if (!w->kinc) { w->kinc = malloc(n * sizeof *w->kinc); if (!w->kinc) return -1; }
            for (size_t v = 0; v < n; v++) {
                w->kinc[v] = NAN;
                if (v % L == 0 || isnan(w->raw[v]) || isnan(w->raw[v - 1])) continue;
                const float *a = w->q + 3 * (v - 1), *b = w->q + 3 * v;
                const double len = sqrt((b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1]) + (b[2] - a[2]) * (b[2] - a[2]));
                const int m = (int)ceil(len / 0.5); if (m <= 1) continue;
                double prev = w->raw[v - 1], sum = 0; int ok = 1;
                for (int t = 1; t <= m && ok; t++) {
                    const double x[3] = {a[0] + (b[0] - a[0]) * t / (double)m, a[1] + (b[1] - a[1]) * t / (double)m, a[2] + (b[2] - a[2]) * t / (double)m};
                    const double g = t == m ? w->raw[v] : vws_gt_at(w->gt, go, gn, f, S->psign, x);
                    if (g < 0) { ok = 0; break; }
                    double dd = g - prev; dd -= floor(dd + 0.5); sum += dd; prev = g;
                }
                if (ok) w->kinc[v] = (float)sum;
            }
        }
        VP(VP_KINC);
        const int nc = vws_unwrap_k(w->raw, geo ? w->kinc : nullptr, T, L, 64, out->ph, out->comp); if (nc <= 0) REJ(VR_NC)
        VP(VP_UNWRAP);
        size_t kept = 0;
        for (size_t v = 0; v < n; v++) {
            if (!out->comp[v]) { out->pw[v] = 0; continue; }
            if (inward) out->ph[v] = -out->ph[v];
            const float *qq = w->q + 3 * v; const double p[3] = {qq[2], qq[1], qq[0]};
            out->pw[v] = w->nsidx && vws_seg_hit(S->seg, w->sidx, w->nsidx, R * R, p) ? 0.f : 1.f; kept += out->pw[v] > 0;
        }
        if (kept < n / 100) REJ(VR_KEPT)
        VP(VP_PW);
        if (g_vwa_ok && V->train) vwa_intensity(&g_vwa, vws_rng(rs), out->im, out->va, T, L);
        VP(VP_INTENS); if (vp_on) { atomic_fetch_add(&vp_slabs, 1); atomic_fetch_add(&vp_geo, geo); }
        out->T = T; out->L = L; out->src = si; out->flags = inward;
        if (V->train && g_mv_p > 0 && !geo && out->pim && vws_unif(rs) < g_mv_p) out->has_pair = vws_pair(V, rs, w, S, out) == 0;
        return 0;
    }
    return -1;
}

/* ---- workers ---- */
typedef struct { vws *V; int t; } vws_arg;
static void *vws_worker(void *arg) {
    vws_arg *A = arg; vws *V = A->V; vws_work w = {0}; const size_t n = (size_t)V->T * V->T * V->L;
    w.raw = malloc(n * sizeof *w.raw); w.q = malloc(3 * n * sizeof *w.q); w.n = n;
    for (;;) {
        pthread_mutex_lock(&V->mu); int k = -1;
        while (!V->stop) { for (int s = 0; s < V->nslot; s++) if (V->slot[s].state == 0) { k = s; break; } if (k >= 0) break; pthread_cond_wait(&V->cv, &V->mu); }
        if (V->stop) { pthread_mutex_unlock(&V->mu); break; }
        V->slot[k].state = 1; pthread_mutex_unlock(&V->mu);
        const long long id = atomic_fetch_add(&V->next_id, 1);
        uint64_t r0 = V->train ? ((uint64_t)V->seed << 32) ^ (uint64_t)id * 0x9e3779b97f4a7c15ull ^ (uint64_t)A->t << 56 : ((uint64_t)V->seed << 32) ^ (uint64_t)id;   /* val: sample id i -> seed (seed, i) */
        uint64_t rs = vws_rng(&r0);   /* (mixed once: raw states differing by multiples of the splitmix step would share one stream: duplicate slabs) */
        int rc = vws_fill(V, &rs, &w, &V->slot[k].s);
        pthread_mutex_lock(&V->mu); V->slot[k].state = rc ? 0 : 2; pthread_cond_broadcast(&V->cv); pthread_mutex_unlock(&V->mu);
        if (rc) fprintf(stderr, "vws: no slab after 64 draws\n");
    }
    free(w.raw); free(w.q); free(w.kinc); free(w.arena); free(w.ct); free(w.gt); free(w.shb); free(w.sidx); free(A); return nullptr;
}

int vws_next(vws *V, vws_sample *out) {
    pthread_mutex_lock(&V->mu);
    if (V->last >= 0) { V->slot[V->last].state = 0; V->last = -1; pthread_cond_broadcast(&V->cv); }
    int k = -1;
    for (;;) { for (int s = 0; s < V->nslot; s++) if (V->slot[s].state == 2) { k = s; break; } if (k >= 0) break; pthread_cond_wait(&V->cv, &V->mu); }
    V->slot[k].state = 3; V->last = k; *out = V->slot[k].s; pthread_mutex_unlock(&V->mu);
    return 0;
}

/* ---- open / close ---- */
static int vws_add_box(vws_src *S, double z, double y, double x, double nz, double ny, double nx) {
    S->box = realloc(S->box, (S->nbox + 1) * sizeof *S->box); if (!S->box) return -1;
    double *b = S->box[S->nbox++]; b[0] = z; b[1] = y; b[2] = x; b[3] = nz; b[4] = ny; b[5] = nx; return 0;
}
vws *vws_open(const char *sources_json, int T, int L, float tilt_deg, int nthreads, unsigned seed, int train) {
    size_t tl; char *txt = slurp(sources_json, &tl); if (!txt) { fprintf(stderr, "vws: cannot read %s\n", sources_json); return nullptr; }
    json *J = json_parse(txt, tl); free(txt); const json *arr = J ? json_get(J, "sources") : nullptr;
    if (!arr || arr->type != J_ARR) { fprintf(stderr, "vws: no \"sources\" in %s\n", sources_json); json_free(J); return nullptr; }
    vws *V = calloc(1, sizeof *V); V->T = T; V->L = L; V->tilt = tilt_deg; V->train = train; V->seed = seed; V->last = -1;
    V->S = calloc(arr->n, sizeof *V->S); const char *cache = json_str(json_get(J, "cache"), nullptr);
    /* pass 1: drawable sources (weight > 1e-3) */
    for (size_t i = 0; i < arr->n; i++) {
        const json *e = arr->items[i]; const double wt = json_num(json_get(e, "weight"), 1); if (wt <= 1e-3) continue;
        vws_src *S = &V->S[V->ns]; memset(S, 0, sizeof *S);
        snprintf(S->name, sizeof S->name, "%s", json_str(json_get(e, "name"), "?"));
        const char *root = json_str(json_get(e, "root"), nullptr), *ct = json_str(json_get(e, "ct"), nullptr);
        S->um0 = json_num(json_get(e, "um"), 0); S->lv = (int)json_num(json_get(e, "min_level"), 0); S->l0s = ldexp(1, S->lv); S->weight = wt;
        const json *pt = json_path(e, "targets.band.phase");
        const char *proot = pt ? json_str(json_get(pt, "root"), nullptr) : nullptr, *pkey = pt ? json_str(json_get(pt, "key"), nullptr) : nullptr;
        S->plv = pt ? (int)json_num(json_get(pt, "level"), S->lv) : S->lv; S->psign = pt && json_num(json_get(pt, "phase_sign"), 1) < 0 ? -1 : 1;
        const json *shj = json_get(e, "shell");   /* {"root", "key", "level"}: surface-patch shells (patch-raster codes, 255 none); alone = a shell-only source */
        const char *sroot = shj ? json_str(json_get(shj, "root"), nullptr) : nullptr, *skey = shj ? json_str(json_get(shj, "key"), nullptr) : nullptr;
        S->shlv = shj ? (int)json_num(json_get(shj, "level"), S->lv) : S->lv;
        if (!root || !ct || !(S->um0 > 0) || ((!proot || !pkey) && (!sroot || !skey))) { fprintf(stderr, "vws: %s: needs root, ct, um and targets.band.phase {root, key, level} and / or shell {root, key, level}: skipped\n", S->name); continue; }
        S->ct = pyramid_open_level(store_open(root), ct, S->lv, S->um0, cache);
        S->ph = proot && pkey ? z3_open(store_open(proot), pkey, nullptr) : nullptr;
        S->sh = sroot && skey ? z3_open(store_open(sroot), skey, nullptr) : nullptr;
        if (!S->ct || (proot && !S->ph) || (sroot && !S->sh)) { fprintf(stderr, "vws: %s: cannot open CT level %d, GT %s/%s or shells %s/%s: %s\n", S->name, S->lv, proot ? proot : "-", pkey ? pkey : "-", sroot ? sroot : "-", skey ? skey : "-", z3_error()); REJ(VR_READ) }
        if (S->sh) V->has_shell = 1;
        const char *axp = json_str(json_get(e, "axis"), nullptr); S->hasax = axp && !axis_load(&S->ax, axp);
        if (!S->hasax) fprintf(stderr, "vws: %s: no axis (rays radial from the box centre line x = y = 0)\n", S->name);
        V->wsum += wt; V->ns++;
    }
    /* pass 2: holdout boxes (L0) of every entry onto each drawable source of the same CT; holdout_file boxes (+0.5 mm) */
    for (int s = 0; s < V->ns; s++) {
        vws_src *S = &V->S[s]; const char *myct = nullptr;
        for (size_t i = 0; i < arr->n; i++) if (!strcmp(json_str(json_get(arr->items[i], "name"), ""), S->name)) myct = json_str(json_get(arr->items[i], "ct"), "");
        const char *hof = nullptr;
        for (size_t i = 0; i < arr->n; i++) {
            const json *e = arr->items[i]; if (strcmp(base_of(json_str(json_get(e, "ct"), "")), base_of(myct ? myct : ""))) continue;
            const json *h = json_get(e, "holdout");
            if (h && h->type == J_ARR && h->n == 6) {
                double v[6]; for (int d = 0; d < 6; d++) v[d] = json_num(json_at(h, d), 0) / S->l0s;
                vws_add_box(S, v[0], v[1], v[2], v[3], v[4], v[5]);
            }
            if (json_get(e, "holdout_file")) hof = json_str(json_get(e, "holdout_file"), nullptr);
        }
        if (hof) {
            size_t hl; char *ht = slurp(hof, &hl); json *H = ht ? json_parse(ht, hl) : nullptr; free(ht);
            const json *bx = H ? json_get(H, "boxes") : nullptr; const double m = 500.0 / S->um0;   /* 0.5 mm in L0 voxels */
            for (size_t b = 0; bx && b < bx->n; b++) {
                const json *e = bx->items[b]; if (strcmp(base_of(json_str(json_get(e, "ct"), "")), base_of(myct ? myct : ""))) continue;
                const json *l0 = json_get(e, "l0"); double v[6]; for (int d = 0; d < 6; d++) v[d] = json_num(json_at(l0, d), 0);
                vws_add_box(S, (v[0] - m) / S->l0s, (v[1] - m) / S->l0s, (v[2] - m) / S->l0s, (v[3] + 2 * m) / S->l0s, (v[4] + 2 * m) / S->l0s, (v[5] + 2 * m) / S->l0s);
            }
            json_free(H);
        }
        /* exclusion tubes */
        const json *xr = json_get(J, "exclude_rays");
        if (xr && myct && !strcmp(base_of(json_str(json_get(xr, "ct"), "")), base_of(myct))) {
            size_t rl; char *rt = slurp(json_str(json_get(xr, "file"), ""), &rl); json *R = rt ? json_parse(rt, rl) : nullptr; free(rt);
            const json *C = R ? json_get(R, "collections") : nullptr; S->seg_r = json_num(json_get(xr, "r"), 52);
            const double sc = ldexp(1, (int)json_num(json_get(xr, "level"), 2) - S->lv);   /* CT voxels per point unit */
            for (size_t c = 0; C && c < C->n; c++) {
                const json *P = json_get(C->items[c], "points"); if (!P || P->n < 2) continue;
                double (*v)[4] = malloc(P->n * sizeof *v); size_t nv = 0;
                for (size_t k = 0; k < P->n; k++) {
                    const json *p = json_get(P->items[k], "p"), *wa = json_get(P->items[k], "wind_a"); if (!p || !wa || wa->type != J_NUM) continue;
                    v[nv][0] = json_num(json_at(p, 2), 0) * sc; v[nv][1] = json_num(json_at(p, 1), 0) * sc; v[nv][2] = json_num(json_at(p, 0), 0) * sc; v[nv][3] = json_num(wa, 0); nv++;
                }
                for (size_t a = 0; a < nv; a++) for (size_t b = a + 1; b < nv; b++) if (v[b][3] < v[a][3]) { double t[4]; memcpy(t, v[a], sizeof t); memcpy(v[a], v[b], sizeof t); memcpy(v[b], t, sizeof t); }
                for (size_t k = 0; k + 1 < nv; k++) {
                    S->seg = realloc(S->seg, (S->nseg + 1) * sizeof *S->seg); double *g = S->seg[S->nseg++];
                    for (int d = 0; d < 3; d++) { g[d] = v[k][d]; g[3 + d] = v[k + 1][d]; }
                }
                free(v);
            }
            json_free(R);
        }
        /* stored GT chunks whose centre is outside the holdout boxes */
        z3 *dz = S->ph ? S->ph : S->sh;   /* draws: from the GT's stored chunks, else (shell-only) the shells' */
        uint8_t *bits = nullptr; const int64_t np = z3_chunk_map(dz, S->cgrid, &bits);
        const z3_meta *gm = z3_meta_of(dz); for (int d = 0; d < 3; d++) S->csz[d] = gm->chunk[d];
        const double f = ldexp(1, (S->ph ? S->plv : S->shlv) - S->lv); int64_t kept = 0;
        if (np > 0) {
            S->chunks = malloc(np * sizeof *S->chunks);
            const int64_t ng = S->cgrid[0] * S->cgrid[1] * S->cgrid[2];
            for (int64_t k = 0; k < ng; k++) {
                if (!bits[k]) continue;
                const double c[3] = {((double)(k / (S->cgrid[1] * S->cgrid[2])) + 0.5) * S->csz[0] * f, ((double)(k / S->cgrid[2] % S->cgrid[1]) + 0.5) * S->csz[1] * f, ((double)(k % S->cgrid[2]) + 0.5) * S->csz[2] * f};
                int in = 0; for (int b = 0; b < S->nbox && !in; b++) { const double *B = S->box[b]; in = c[0] >= B[0] && c[0] < B[0] + B[3] && c[1] >= B[1] && c[1] < B[1] + B[4] && c[2] >= B[2] && c[2] < B[2] + B[5]; }
                if (!in) { S->chunks[kept++] = k; if (!S->ok) S->ok = calloc((size_t)ng, 1); if (S->ok) S->ok[k] = 1; }
            }
        }
        free(bits); S->nch = kept;
        fprintf(stderr, "vws: %s: CT level %d (%.3f um), GT level %d, %lld of %lld stored GT chunks outside %d holdout boxes, %d exclusion tubes (r %.0f), weight %g\n",
                S->name, S->lv, S->um0 * S->l0s, S->plv, (long long)kept, (long long)np, S->nbox, S->nseg, S->seg_r, S->weight);
        if (!kept) { V->wsum -= S->weight; S->weight = 0; }
    }
    json_free(J);
    if (!V->ns || V->wsum <= 0) { fprintf(stderr, "vws: no drawable source\n"); vws_close(V); return nullptr; }
    for (int s = 0; s < V->ns; s++) if (V->S[s].weight <= 0) V->S[s].weight = 0;
    for (int s = 0; s < V->ns; s++) if (V->S[s].nch && !V->S[s].ok) { fprintf(stderr, "vws: out of memory\n"); return nullptr; }
    {   const char *e = getenv("UFSM_VWS_CACHE_GB"); const double gb = e ? atof(e) : train ? 4 : 1; V->zc = gb > 0 ? zcache_open((size_t)(gb * 1e9)) : nullptr;
        const char *b = getenv("UFSM_VWS_BURST"); V->burst = b ? atoi(b) : 8; if (V->burst < 1) V->burst = 1;
        const char *br = getenv("UFSM_VWS_BURST_R"); V->burst_r = br ? atoi(br) : 0; if (V->burst_r < 0) V->burst_r = 0;
        fprintf(stderr, "vws: chunk cache %.1f GB, training bursts of %d slabs\n", gb > 0 ? gb : 0., V->burst); }
    pthread_mutex_init(&V->mu, nullptr); pthread_cond_init(&V->cv, nullptr);
    V->nth = nthreads < 1 ? 1 : nthreads; V->nslot = V->nth + 2; V->slot = calloc(V->nslot, sizeof *V->slot);
    const size_t n = (size_t)T * T * L;
    for (int s = 0; s < V->nslot; s++) {
        vws_sample *o = &V->slot[s].s; o->im = malloc(n); o->va = malloc(n); o->ph = malloc(n * 4); o->comp = malloc(n * 2); o->pw = malloc(n * 4);
        o->shell = V->has_shell ? malloc(n) : nullptr;
        o->pim = train && g_mv_p > 0 ? malloc(n) : nullptr; o->pva = o->pim ? malloc(n) : nullptr;
        if (!o->im || !o->va || !o->ph || !o->comp || !o->pw) { fprintf(stderr, "vws: out of memory\n"); return nullptr; }
    }
    V->th = calloc(V->nth, sizeof *V->th);
    for (int t = 0; t < V->nth; t++) { vws_arg *A = malloc(sizeof *A); A->V = V; A->t = t; pthread_create(&V->th[t], nullptr, vws_worker, A); }
    return V;
}

void vws_close(vws *V) {
    vp_print();
    if (!V) return;
    if (V->th) {
        pthread_mutex_lock(&V->mu); V->stop = 1; pthread_cond_broadcast(&V->cv); pthread_mutex_unlock(&V->mu);
        for (int t = 0; t < V->nth; t++) pthread_join(V->th[t], nullptr);
        pthread_mutex_destroy(&V->mu); pthread_cond_destroy(&V->cv);
    }
    for (int s = 0; V->slot && s < V->nslot; s++) { vws_sample *o = &V->slot[s].s; free(o->im); free(o->va); free(o->ph); free(o->comp); free(o->pw); free(o->shell); free(o->pim); free(o->pva); }
    for (int s = 0; s < V->ns; s++) { vws_src *S = &V->S[s]; if (S->ct) z3_close(S->ct); if (S->ph) z3_close(S->ph); if (S->sh) z3_close(S->sh); free(S->chunks); free(S->ok); free(S->box); free(S->seg); }
    if (V->zc) { uint64_t h, m; size_t by; zcache_stats(V->zc, &h, &m, &by); if (vp_on > 0) fprintf(stderr, "vws: chunk cache %llu hits, %llu misses, %.2f GB held\n", (unsigned long long)h, (unsigned long long)m, by / 1e9); zcache_close(V->zc); }
    free(V->S); free(V->slot); free(V->th); free(V);
}

/* ufsm vwt-sample SOURCES OUT N [--T 128] [--L 384] [--tilt 15] [--threads 8] [--seed 1] [--val 0]: N slabs -> OUT/sNNN.{im,va,comp}.u8|u16 raw,
   .ph.f32, .pw.f32 + OUT/meta.tsv; prints the throughput (slabs / s) */
static const char *vo(int argc, char **argv, const char *k, const char *d) { for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return argv[i + 1]; return d; }
#include <time.h>
#include <sys/stat.h>
int cmd_vwt_sample(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: ufsm vwt-sample SOURCES OUT N [--T 128] [--L 384] [--tilt 15] [--threads 8] [--seed 1] [--val 0] [--write 1] [--jpg 0]\n"
                                    "       [--vaug 1 ... (any vwt-train aug flag: the training augmentation, src/vwt_aug.h)]\n"); return 2; }
    for (int i = 5; i < argc; i++) if (!strncmp(argv[i], "--vaug", 6) || !strncmp(argv[i], "--deform", 8) || !strncmp(argv[i], "--aug-crush", 11)) { vws_set_args(argc, argv); break; }
    const int jpg = atoi(vo(argc, argv, "--jpg", "0"));
    const int T = atoi(vo(argc, argv, "--T", "128")), L = atoi(vo(argc, argv, "--L", "384")), N = atoi(argv[4]), wr = atoi(vo(argc, argv, "--write", "1"));
    vws *V = vws_open(argv[2], T, L, (float)atof(vo(argc, argv, "--tilt", "15")), atoi(vo(argc, argv, "--threads", "8")), (unsigned)atoi(vo(argc, argv, "--seed", "1")), !atoi(vo(argc, argv, "--val", "0")));
    if (!V) return 1;
    mkdir(argv[3], 0775); char p[1400]; snprintf(p, sizeof p, "%s/meta.tsv", argv[3]); FILE *m = fopen(p, "w");
    if (m) fprintf(m, "i\tsrc\tinward\to_x\to_y\to_z\td_x\td_y\td_z\tlabelled\tweighted\tcomps\tph_rise\tshell\n");
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0); const size_t n = (size_t)T * T * L;
    for (int i = 0; i < N; i++) {
        vws_sample s; if (vws_next(V, &s)) break;
        size_t lab = 0, wsum = 0, up = 0, dn = 0, nsh = 0; int nc = 0;
        for (size_t v = 0; v < n; v++) { lab += s.comp[v] != 0; wsum += s.pw[v] > 0; nsh += s.shell && s.shell[v]; if (s.comp[v] > nc) nc = s.comp[v];
            if (v % L && s.comp[v] && s.comp[v] == s.comp[v - 1]) { const float d = s.ph[v] - s.ph[v - 1]; up += d > 0; dn += d < 0; } }
        if (m) fprintf(m, "%d\t%d\t%d\t%.1f\t%.1f\t%.1f\t%.3f\t%.3f\t%.3f\t%.4f\t%.4f\t%d\t%.3f\t%.5f\n", i, s.src, s.flags & 1, s.o[0], s.o[1], s.o[2], s.d[0], s.d[1], s.d[2],
                       (double)lab / n, (double)wsum / n, nc, up + dn ? (double)up / (up + dn) : 0.0, (double)nsh / n);
        if (jpg) {   /* the i = T / 2 plane ([j][k], the ray across): CT | wrapped target phase hue over the CT (unlabelled: CT) */
            uint8_t *img = malloc((size_t)2 * T * L * 3);
            for (int j = 0; j < T; j++) for (int k = 0; k < L; k++) {
                const size_t v = ((size_t)(T / 2) * T + j) * L + k; const int g = s.im[v];
                uint8_t *a = img + ((size_t)j * L + k) * 3, *b = img + ((size_t)(T + j) * L + k) * 3; a[0] = a[1] = a[2] = b[0] = b[1] = b[2] = (uint8_t)g;
                if (s.comp[v]) { const double h = s.ph[v] - floor(s.ph[v]);
                    const double cr[3] = {.5 + .5 * cos(2 * M_PI * h), .5 + .5 * cos(2 * M_PI * (h - 1. / 3)), .5 + .5 * cos(2 * M_PI * (h - 2. / 3))};
                    for (int e = 0; e < 3; e++) b[e] = (uint8_t)((g + 255 * cr[e]) / 2); }
            }
            snprintf(p, sizeof p, "%s/s%03d.jpg", argv[3], i); jpeg_write_rgb(p, L, 2 * T, img, 88); free(img);
        }
        if (wr) {
            const char *ext[6] = {"im.u8", "va.u8", "ph.f32", "comp.u16", "pw.f32", "shell.u8"}; const void *buf[6] = {s.im, s.va, s.ph, s.comp, s.pw, s.shell}; const size_t es[6] = {1, 1, 4, 2, 4, 1};
            for (int k = 0; k < (s.shell ? 6 : 5); k++) { snprintf(p, sizeof p, "%s/s%03d.%s", argv[3], i, ext[k]); FILE *f = fopen(p, "wb"); if (f) { fwrite(buf[k], es[k], n, f); fclose(f); } }
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1); const double dt = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
    printf("vwt-sample: %d slabs %dx%dx%d in %.1f s = %.2f slabs/s (%s threads)\n", N, T, T, L, dt, N / dt, vo(argc, argv, "--threads", "8"));
    if (m) fclose(m); vws_close(V); return 0;
}
