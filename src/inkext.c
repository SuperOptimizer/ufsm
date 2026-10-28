/* ufsm ink-ext / ink-ensemble: the external ink networks (inkext.h, inkext_net.c) on our surfaces.

   ink-ext renders each surface's volume itself, in C: per output pixel of the model's grid (the tifxyz grid resampled to the
   model's spacing) the surface point P and unit normal N (bilinear grid, du x dv, as ufsm render), and per layer i the CT at
   P + (i - center) s N (s = the spacing; --reverse: center - i), trilinear on the coarsest CT pyramid level not coarser than
   the spacing, box-averaged over ss^3 sub-samples when the level is finer than the spacing. Rows are rendered on demand into a
   rolling buffer, the decoded 128^3 CT chunks a band needs are read once (LRU past --ct-mem) and looked up lock-free.

   Tiling and blending are each model's own: v8in (inference.py): coverage = closing7x7(max over depth > 0), 64 px tiles at
   stride 21 fully inside it, prob x gaussian(64, 1 sigma)/max summed and divided by the tile count. Hecate (hecate.py): origins
   0, s, .. and the last L - p, empty patches skipped, prob x column support x hann(p) summed over hann sums. Reader V3 (koine
   infer.py, --overlap 0.5 --blend-mode hann): stride 192, empty patches skipped, hann (floor 1e-3) weights.
   Output: the probability on the surface's 2.4 um render grid (ufsm render --scale 1: (w - 1) / scale + 1 pixels wide) as
   <out>/<surf>_<model>.tif (uint8, 255 = 1) and a half-resolution JPEG; --native also the model-grid map. */
#include "inkext.h"
#include "imgops.h"
#include "wload.h"
#include "store.h"
#include "tiffw.h"
#include <unistd.h>
#include "xnet.h"
#include "zarr3.h"
#include <dirent.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

static const char *iopt(int argc, char **argv, const char *k, const char *d) { for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return argv[i + 1]; return d; }
static int iflag(int argc, char **argv, const char *k) { for (int i = 2; i < argc; i++) if (!strcmp(argv[i], k)) return 1; return 0; }
static double inow(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

/* ---- surface ---- */
typedef struct { char name[256]; int w, h; float *X, *Y, *Z; double scale, vox_um; } isurf;
static int isurf_load(isurf *s, const char *dir) {
    memset(s, 0, sizeof *s);
    const char *b = strrchr(dir, '/'); snprintf(s->name, sizeof s->name, "%s", b && b[1] ? b + 1 : dir);
    { size_t l = strlen(s->name); while (l && s->name[l - 1] == '/') s->name[--l] = 0; }
    char p[1400]; snprintf(p, sizeof p, "%s/meta.json", dir); FILE *f = tfx_meta_fopen(p); if (!f) return -1;
    char m[16384] = {0}; const size_t mn = fread(m, 1, sizeof m - 1, f); (void)mn; fclose(f);
    const char *q = strstr(m, "\"scale\""); if (!q || !(q = strchr(q, '['))) return -1;
    s->scale = atof(q + 1); q = strstr(m, "\"voxel_size_um\""); s->vox_um = q ? atof(strchr(q, ':') + 1) : 2.399;
    static const char *nm[3] = {"x", "y", "z"}; float **A[3] = {&s->X, &s->Y, &s->Z};
    for (int k = 0; k < 3; k++) {
        int w, h; snprintf(p, sizeof p, "%s/%s.tif", dir, nm[k]); *A[k] = img_read_f32(p, &w, &h);
        if (!*A[k] || (k && (w != s->w || h != s->h))) return -1;
        s->w = w; s->h = h;
    }
    return s->w >= 2 && s->h >= 2 && s->scale > 0 ? 0 : -1;
}
/* the grid window [c0, c0 + cw) x [r0, r0 + rh) (clipped) */
static int isurf_crop(isurf *s, int c0, int r0, int cw, int rh) {
    if (c0 < 0) c0 = 0; if (r0 < 0) r0 = 0;
    if (c0 + cw > s->w) cw = s->w - c0; if (r0 + rh > s->h) rh = s->h - r0;
    if (cw < 2 || rh < 2) return -1;
    float **A[3] = {&s->X, &s->Y, &s->Z};
    for (int k = 0; k < 3; k++) {
        float *o = malloc((size_t)cw * rh * sizeof *o);
        for (int r = 0; r < rh; r++) memcpy(o + (size_t)r * cw, *A[k] + (size_t)(r0 + r) * s->w + c0, (size_t)cw * sizeof *o);
        free(*A[k]); *A[k] = o;
    }
    s->w = cw; s->h = rh;
    return 0;
}
static void isurf_free(isurf *s) { free(s->X); free(s->Y); free(s->Z); }
/* point and unit normal at grid coordinates (gu, gv); 0 if the four corner vertices are valid */
static inline int surf_pn(const isurf *s, double gu, double gv, double P[3], double N[3], double du[3], double dv[3]) {
    int c0 = (int)gu, r0 = (int)gv; if (c0 >= s->w - 1) c0 = s->w - 2; if (r0 >= s->h - 1) r0 = s->h - 2; if (c0 < 0 || r0 < 0) return -1;
    const double fu = gu - c0, fv = gv - r0; const size_t i00 = (size_t)r0 * s->w + c0, i01 = i00 + 1, i10 = i00 + s->w, i11 = i10 + 1;
    const float *A[3] = {s->Z, s->Y, s->X};
    if (s->Z[i00] < 0 || s->Z[i01] < 0 || s->Z[i10] < 0 || s->Z[i11] < 0) return -1;
    for (int d = 0; d < 3; d++) {
        const double a = A[d][i00], b = A[d][i01], c = A[d][i10], e = A[d][i11];
        P[d] = (1 - fv) * ((1 - fu) * a + fu * b) + fv * ((1 - fu) * c + fu * e);
        du[d] = (1 - fv) * (b - a) + fv * (e - c); dv[d] = (1 - fu) * (c - a) + fu * (e - b);
    }
    N[0] = du[1] * dv[2] - du[2] * dv[1]; N[1] = du[2] * dv[0] - du[0] * dv[2]; N[2] = du[0] * dv[1] - du[1] * dv[0];
    const double nl = sqrt(N[0] * N[0] + N[1] * N[1] + N[2] * N[2]); if (nl < 1e-12) return -1;
    for (int d = 0; d < 3; d++) N[d] /= nl;
    return 0;
}

/* ---- decoded CT chunks of one pyramid level ---- */
enum { CSH = 7, CE = 128 };
typedef struct { int64_t key; uint8_t *p; uint32_t last; } cent;
typedef struct {
    z3 *z; int64_t shape[3]; size_t budget;
    cent *e; size_t ne, cap;            /* resident chunks (p == zero: all zero) */
    int64_t *hk; int32_t *hv; size_t hcap;   /* open-addressing key -> index into e (rebuilt per band) */
    uint8_t *zero; uint32_t epoch; uint64_t ndec;
} ctlev;
static inline int64_t ckey(int64_t a, int64_t b, int64_t c) { return (a << 42) | (b << 21) | c; }
static inline size_t chash(int64_t k, size_t cap) { uint64_t x = (uint64_t)k * 0x9E3779B97F4A7C15ull; return (size_t)(x >> 20) & (cap - 1); }
static const uint8_t *ct_find(const ctlev *L, int64_t k) {
    for (size_t h = chash(k, L->hcap);; h = (h + 1) & (L->hcap - 1)) { if (L->hk[h] == k) return L->e[L->hv[h]].p; if (L->hk[h] == -1) return nullptr; }
}
static void ct_rehash(ctlev *L) {
    size_t cap = 1024; while (cap < L->ne * 2 + 16) cap *= 2;
    if (cap != L->hcap) { free(L->hk); free(L->hv); L->hk = malloc(cap * sizeof *L->hk); L->hv = malloc(cap * sizeof *L->hv); L->hcap = cap; }
    for (size_t i = 0; i < cap; i++) L->hk[i] = -1;
    for (size_t i = 0; i < L->ne; i++) { size_t h = chash(L->e[i].key, cap); while (L->hk[h] != -1) h = (h + 1) & (cap - 1); L->hk[h] = L->e[i].key; L->hv[h] = (int32_t)i; }
}
typedef struct { ctlev *L; int64_t *todo; size_t n; _Atomic size_t next; _Atomic int failed; uint8_t **out; } decjob;
static void *dec_worker(void *a) {
    decjob *J = a; const size_t cv = (size_t)CE * CE * CE;
    for (size_t t; (t = atomic_fetch_add(&J->next, 1)) < J->n;) {
        const int64_t k = J->todo[t], o[3] = {(k >> 42) * CE, ((k >> 21) & 0x1fffff) * CE, (k & 0x1fffff) * CE}, n[3] = {CE, CE, CE};
        if (o[0] >= J->L->shape[0] || o[1] >= J->L->shape[1] || o[2] >= J->L->shape[2]) { J->out[t] = J->L->zero; continue; }
        uint8_t *b = malloc(cv);
        if (!b || z3_read(J->L->z, o, n, b, 1)) { free(b); atomic_store(&J->failed, 1); J->out[t] = J->L->zero; continue; }
        size_t q = 0; const uint64_t *w = (const uint64_t *)b; while (q < cv / 8 && !w[q]) q++;
        if (q == cv / 8) { free(b); J->out[t] = J->L->zero; } else J->out[t] = b;
    }
    return nullptr;
}
static int lcmp(const void *a, const void *b) { const cent *p = a, *q = b; return p->last < q->last ? -1 : p->last > q->last; }
/* make the chunks of keys[0..nk) resident (decoding the missing ones with nth threads), evicting old ones past the budget */
static int ct_need(ctlev *L, const int64_t *keys, size_t nk, int nth) {
    L->epoch++;
    int64_t *todo = malloc((nk + 1) * sizeof *todo); size_t nt = 0;
    for (size_t i = 0; i < nk; i++) {
        const int64_t k = keys[i]; size_t h = chash(k, L->hcap); int found = 0;
        if (L->hcap) for (;; h = (h + 1) & (L->hcap - 1)) { if (L->hk[h] == k) { L->e[L->hv[h]].last = L->epoch; found = 1; break; } if (L->hk[h] == -1) break; }
        if (!found) todo[nt++] = k;
    }
    const size_t cv = (size_t)CE * CE * CE; size_t live = 0;
    for (size_t i = 0; i < L->ne; i++) if (L->e[i].p != L->zero) live++;
    if ((live + nt) * cv > L->budget) {   /* drop the least recently needed chunks this band does not need */
        qsort(L->e, L->ne, sizeof *L->e, lcmp); size_t k = 0;
        for (size_t i = 0; i < L->ne; i++) {
            cent *c = &L->e[i];
            if (c->last != L->epoch && (live + nt) * cv > L->budget) { if (c->p != L->zero) { free(c->p); live--; } continue; }
            L->e[k++] = *c;
        }
        L->ne = k;
    }
    int rc = 0;
    if (nt) {
        uint8_t **out = calloc(nt, sizeof *out); decjob J = {L, todo, nt, 0, 0, out};
        const int T = nth < 1 ? 1 : nth > 64 ? 64 : nth; pthread_t th[64];
        for (int t = 0; t < T; t++) pthread_create(&th[t], nullptr, dec_worker, &J);
        for (int t = 0; t < T; t++) pthread_join(th[t], nullptr);
        if (atomic_load(&J.failed)) { fprintf(stderr, "ink-ext: CT chunk read failed: %s\n", z3_error()); rc = -1; }
        if (L->ne + nt > L->cap) { L->cap = (L->ne + nt) * 2; L->e = realloc(L->e, L->cap * sizeof *L->e); }
        for (size_t i = 0; i < nt; i++) L->e[L->ne++] = (cent){todo[i], out[i], L->epoch};
        L->ndec += nt; free(out);
    }
    free(todo);
    ct_rehash(L);
    return rc;
}
static inline float ct_tri(const ctlev *L, double z, double y, double x, const uint8_t **cp, int64_t *ck) {
    if (z < 0 || y < 0 || x < 0) return 0.f;
    const int64_t z0 = (int64_t)z, y0 = (int64_t)y, x0 = (int64_t)x; const double fz = z - z0, fy = y - y0, fx = x - x0;
    double v[8];
    if ((z0 & (CE - 1)) != CE - 1 && (y0 & (CE - 1)) != CE - 1 && (x0 & (CE - 1)) != CE - 1) {
        const int64_t k = ckey(z0 >> CSH, y0 >> CSH, x0 >> CSH);
        if (k != *ck) { *ck = k; *cp = ct_find(L, k); }
        const uint8_t *p = *cp; if (!p) return 0.f;
        p += (((z0 & (CE - 1)) << (2 * CSH)) + ((y0 & (CE - 1)) << CSH)) + (x0 & (CE - 1));
        v[0] = p[0]; v[1] = p[1]; v[2] = p[CE]; v[3] = p[CE + 1]; v[4] = p[CE * CE]; v[5] = p[CE * CE + 1]; v[6] = p[CE * CE + CE]; v[7] = p[CE * CE + CE + 1];
    } else for (int c = 0; c < 8; c++) {
        const int64_t zz = z0 + (c >> 2 & 1), yy = y0 + (c >> 1 & 1), xx = x0 + (c & 1);
        const uint8_t *p = ct_find(L, ckey(zz >> CSH, yy >> CSH, xx >> CSH));
        v[c] = p ? p[(((zz & (CE - 1)) << (2 * CSH)) + ((yy & (CE - 1)) << CSH)) + (xx & (CE - 1))] : 0;
    }
    double s = 0;
    for (int c = 0; c < 8; c++) s += (c >> 2 & 1 ? fz : 1 - fz) * (c >> 1 & 1 ? fy : 1 - fy) * (c & 1 ? fx : 1 - fx) * v[c];
    return (float)s;
}

/* ---- the surface-volume renderer ---- */
typedef struct {
    const isurf *S; ctlev *L; double lscale;   /* level coordinate = level-0 voxel / lscale */
    int D, reverse, ss; double center, du, shift;  /* center = (D - 1) / 2: layer i at offset i - center (half-integer for even D, as the published renders); shift: + shift layers along +N (--shift) */     /* grid units per model pixel */
    double step_vox;                           /* layer / pixel step in level-0 voxels */
    int Wm, Hm;
} irender;
typedef struct { const irender *R; int r0, r1; uint8_t **rows; _Atomic int next; } rjob;
static void *render_worker(void *a) {
    rjob *J = a; const irender *R = J->R; const int ss = R->ss; const double inv = 1.0 / R->lscale;
    const uint8_t *cp = nullptr; int64_t ck = -1;
    for (int r; (r = atomic_fetch_add(&J->next, 1)) < J->r1 - J->r0;) {
        const int y = J->r0 + r; uint8_t *row = J->rows[r];   /* [D][Wm] */
        memset(row, 0, (size_t)R->D * R->Wm);
        for (int x = 0; x < R->Wm; x++) {
            double P[3], N[3], dU[3], dV[3];
            if (surf_pn(R->S, x * R->du, y * R->du, P, N, dU, dV)) continue;
            double tu[3], tv[3];   /* one model pixel along u / v (level-0 voxels) */
            for (int d = 0; d < 3; d++) { tu[d] = dU[d] * R->du; tv[d] = dV[d] * R->du; }
            for (int i = 0; i < R->D; i++) {
                const double off = ((R->reverse ? R->center - i : i - R->center) + R->shift) * R->step_vox;
                double acc = 0;
                for (int a = 0; a < ss; a++) for (int b = 0; b < ss; b++) for (int c = 0; c < ss; c++) {
                    const double ou = ss > 1 ? (a + 0.5) / ss - 0.5 : 0, ov = ss > 1 ? (b + 0.5) / ss - 0.5 : 0, on = ss > 1 ? ((c + 0.5) / ss - 0.5) * R->step_vox : 0;
                    double q[3]; for (int d = 0; d < 3; d++) q[d] = (P[d] + (off + on) * N[d] + ou * tu[d] + ov * tv[d]) * inv;
                    acc += ct_tri(R->L, q[0], q[1], q[2], &cp, &ck);
                }
                const double v = acc / (ss * ss * ss);
                row[(size_t)i * R->Wm + x] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : lround(v));
            }
        }
    }
    return nullptr;
}
static int i64cmp(const void *a, const void *b) { const int64_t p = *(const int64_t *)a, q = *(const int64_t *)b; return (p > q) - (p < q); }
/* rows [r0, r1) of the model grid -> rows[r - r0] ([D][Wm] each) */
static int render_rows(const irender *R, int r0, int r1, uint8_t **rows, int nth) {
    if (r1 <= r0) return 0;
    /* the chunks: per 16 x 16 pixel block the box of its corner points +- the layer reach */
    const double reach = (R->center + fabs(R->shift) + 1.5) * R->step_vox + 2 * R->du * 4 + 2;
    size_t nk = 0, ck = 4096; int64_t *keys = malloc(ck * sizeof *keys);
    for (int by = r0; by < r1; by += 16) for (int bx = 0; bx < R->Wm; bx += 16) {
        double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300}; int any = 0;
        const int y1 = by + 16 < r1 ? by + 16 : r1, x1 = bx + 16 < R->Wm ? bx + 16 : R->Wm;
        const int g0 = (int)(bx * R->du), g1 = (int)((x1 - 1) * R->du) + 1, h0 = (int)(by * R->du), h1 = (int)((y1 - 1) * R->du) + 1;
        for (int gv = h0; gv <= h1 && gv < R->S->h; gv++) for (int gu = g0; gu <= g1 && gu < R->S->w; gu++) {
            const size_t i = (size_t)gv * R->S->w + gu; if (R->S->Z[i] < 0) continue;
            const double v[3] = {R->S->Z[i], R->S->Y[i], R->S->X[i]}; any = 1;
            for (int d = 0; d < 3; d++) { if (v[d] < lo[d]) lo[d] = v[d]; if (v[d] > hi[d]) hi[d] = v[d]; }
        }
        if (!any) continue;
        int64_t c0[3], c1[3];
        for (int d = 0; d < 3; d++) {
            double a = (lo[d] - reach) / R->lscale - 1, e = (hi[d] + reach) / R->lscale + 2; if (a < 0) a = 0; if (e < 0) e = 0;
            c0[d] = (int64_t)a >> CSH; c1[d] = (int64_t)e >> CSH;
        }
        for (int64_t a = c0[0]; a <= c1[0]; a++) for (int64_t b = c0[1]; b <= c1[1]; b++) for (int64_t c = c0[2]; c <= c1[2]; c++) {
            if (nk == ck) { ck *= 2; keys = realloc(keys, ck * sizeof *keys); }
            keys[nk++] = ckey(a, b, c);
        }
    }
    qsort(keys, nk, sizeof *keys, i64cmp); size_t u = 0; for (size_t i = 0; i < nk; i++) if (!u || keys[i] != keys[u - 1]) keys[u++] = keys[i];
    int rc = ct_need(R->L, keys, u, nth); free(keys);
    rjob J = {R, r0, r1, rows, 0}; const int T = nth < 1 ? 1 : nth > 64 ? 64 : nth; pthread_t th[64];
    for (int t = 0; t < T; t++) pthread_create(&th[t], nullptr, render_worker, &J);
    for (int t = 0; t < T; t++) pthread_join(th[t], nullptr);
    return rc;
}

/* ---- per-model tiling ---- */
static int *axis_starts_hecate(int L, int p, int stride, int *n) {   /* range(0, max(0, L - p), stride) + [max(0, L - p)] */
    const int last = L - p > 0 ? L - p : 0; int c = 0, *v = malloc(((size_t)last / stride + 2) * sizeof *v);
    for (int s = 0; s < last; s += stride) v[c++] = s;
    v[c++] = last; *n = c; return v;
}
static int *axis_starts_koine(int L, int p, int stride, int *n) {   /* _sliding_positions_1d */
    int c = 0, *v = malloc(((size_t)(L > p ? L - p : 0) / stride + 3) * sizeof *v);
    if (L <= p) { v[0] = 0; *n = 1; return v; }
    for (int s = 0; s < (L - p + 1 > 1 ? L - p + 1 : 1); s += stride) v[c++] = s;
    if (v[c - 1] != L - p) v[c++] = L - p;
    *n = c; return v;
}
static int *axis_starts_v8(int L, int p, int stride, int *n) {   /* range(0, L - p + 1, stride) */
    int c = 0, *v = malloc(((size_t)(L >= p ? L - p : 0) / stride + 2) * sizeof *v);
    for (int s = 0; s + p <= L; s += stride) v[c++] = s;
    *n = c; return v;
}
static void gkern64(float *k, int T) {   /* inference.py gkern(T, 1) / max */
    double c[T + 1], k1[T]; for (int i = 0; i <= T; i++) { const double x = -1 + 2.0 * i / T; c[i] = 0.5 * (1 + erf(x / sqrt(2.0))); }
    for (int i = 0; i < T; i++) k1[i] = c[i + 1] - c[i];
    double mx = 0; for (int i = 0; i < T; i++) for (int j = 0; j < T; j++) if (k1[i] * k1[j] > mx) mx = k1[i] * k1[j];
    for (int i = 0; i < T; i++) for (int j = 0; j < T; j++) k[i * T + j] = (float)(k1[i] * k1[j] / mx);
}
static void hann2(float *k, int T, float floor_) {   /* torch.hann_window(T, periodic=False) outer / max, clamped */
    double w[T]; for (int i = 0; i < T; i++) w[i] = T > 1 ? 0.5 - 0.5 * cos(2 * M_PI * i / (T - 1)) : 1;
    double mx = 0; for (int i = 0; i < T; i++) for (int j = 0; j < T; j++) if (w[i] * w[j] > mx) mx = w[i] * w[j];
    for (int i = 0; i < T; i++) for (int j = 0; j < T; j++) { const float v = (float)(w[i] * w[j] / mx); k[i * T + j] = v < floor_ ? floor_ : v; }
}
/* cv2 MORPH_ELLIPSE 7 x 7 */
static const char ell7[7][8] = {"0001000", "0111110", "1111111", "1111111", "1111111", "0111110", "0001000"};

typedef struct {
    int kind; ie_net *net; const ie_spec *sp; int stride, batch, nth, reverse, blend_norm, no_overlap; size_t ct_mem; double px_um, shift;
    ctlev lev[6]; int nlev; double lev_um[6];
    uint8_t **stk; int stk_n, stk_w, stk_h;   /* --layers: a pre-rendered stack (ascending normal offsets) instead of the renderer */
    uint8_t *dump; int dump_req, render_only;  /* --dump-layers: the rendered model-grid stack [D][Hm][Wm] (run_surface allocates) */
    int norm_sheet;   /* --norm sheet (Reader / ink9um): one normalize_robust over a row sample of the whole surface for every tile */
} irun;

/* the network side of run_surface in its own thread: batches handed over by the renderer (double buffered), forward,
   blend into num / den, so the CPU render of the next rows overlaps the GPU */
typedef struct {
    pthread_mutex_t mu; pthread_cond_t cv; int has, quit, fail;
    uint8_t *tiles; int *tx, *ty, nb;            /* the submitted batch */
    irun *X; int D, T, Wm, Hm; const float *win; float *num, *den, *out;
    double tg; long long ntiles;
} gworker;
static void *gpu_worker(void *a) {
    gworker *G = a; irun *X = G->X; const int D = G->D, T = G->T, Wm = G->Wm, Hm = G->Hm;
    xnet_bind();   /* --gpu N: this thread's CUDA calls on the same device as xnet_init (else device 0) */
    for (;;) {
        pthread_mutex_lock(&G->mu);
        while (!G->has && !G->quit) pthread_cond_wait(&G->cv, &G->mu);
        if (!G->has && G->quit) { pthread_mutex_unlock(&G->mu); break; }
        pthread_mutex_unlock(&G->mu);
        const int nb = G->nb; const double t0 = inow();
        if (!G->fail && ie_forward(X->net, G->tiles, nb, G->out, 0)) G->fail = 1;
        G->tg += inow() - t0; G->ntiles += nb;
        if (!G->fail) for (int k = 0; k < nb; k++) {
            const float *o = G->out + (size_t)k * T * T; const uint8_t *tl = G->tiles + (size_t)k * D * T * T;
            const int ph = G->ty[k] + T <= Hm ? T : Hm - G->ty[k], pw = G->tx[k] + T <= Wm ? T : Wm - G->tx[k];
            for (int y = 0; y < ph; y++) for (int x = 0; x < pw; x++) {
                const size_t g = (size_t)(G->ty[k] + y) * Wm + G->tx[k] + x; const float w = G->win[y * T + x];
                if (X->kind == IE_V8IN) { G->num[g] += o[y * T + x] * w; G->den[g] += X->blend_norm ? w : 1.f; }
                else if (ie_is_koine(X->kind)) { G->num[g] += o[y * T + x] * w; G->den[g] += w; }
                else {   /* Hecate: probability x support (column has data) x window; den is the renderer's */
                    int sup = 0; for (int i = 0; i < D && !sup; i++) sup = tl[((size_t)i * T + y) * T + x] != 0;
                    G->num[g] += sup ? o[y * T + x] * w : 0.f;
                }
            }
        }
        pthread_mutex_lock(&G->mu); G->has = 0; pthread_cond_broadcast(&G->cv); pthread_mutex_unlock(&G->mu);
    }
    return nullptr;
}
/* hand the batch over (waits for the previous one); the caller gets the worker's previous buffers back */
static void gpu_submit(gworker *G, uint8_t **tiles, int **tx, int **ty, int nb) {
    pthread_mutex_lock(&G->mu);
    while (G->has) pthread_cond_wait(&G->cv, &G->mu);
    uint8_t *t = G->tiles; int *a = G->tx, *b = G->ty;
    G->tiles = *tiles; G->tx = *tx; G->ty = *ty; G->nb = nb; G->has = 1;
    *tiles = t; *tx = a; *ty = b;
    pthread_cond_broadcast(&G->cv); pthread_mutex_unlock(&G->mu);
}

/* the probability map (model grid Wm x Hm) of one surface */
static float *run_surface(irun *X, const isurf *S, int *pWm, int *pHm, double *pdu, float **pctl) {
    const ie_spec *sp = X->sp; const int D = sp->D, T = sp->T;
    /* CT level: the coarsest not coarser than the spacing (5% slack) */
    int L = 0; for (int l = 0; l < X->nlev; l++) if (X->lev_um[l] <= X->px_um * 1.05) L = l;
    irender R = {0}; R.S = S; R.L = &X->lev[L]; R.lscale = X->lev_um[L] / X->lev_um[0]; R.D = D; R.center = (D - 1) * 0.5; R.reverse = X->reverse; R.shift = X->shift;
    R.ss = (int)lround(X->px_um / X->lev_um[L]); if (R.ss < 1) R.ss = 1;
    R.du = X->px_um / (S->vox_um / S->scale); R.step_vox = X->px_um / X->lev_um[0];
    R.Wm = (int)((S->w - 1) / R.du) + 1; R.Hm = (int)((S->h - 1) / R.du) + 1;
    if (X->stk) { R.Wm = X->stk_w; R.Hm = X->stk_h; }   /* --layers: the stack's own pixel grid */
    const int stk0 = X->stk ? X->stk_n / 2 - D / 2 + (int)lround(X->shift) : 0;   /* the D central layers (koine / hecate rule), moved by --shift */
    if (X->stk && (stk0 < 0 || stk0 + D > X->stk_n)) { fprintf(stderr, "ink-ext: --layers has %d layers, %s needs %d\n", X->stk_n, sp->name, D); return nullptr; }
    const int Wm = R.Wm, Hm = R.Hm; *pWm = Wm; *pHm = Hm; *pdu = R.du;
    fprintf(stderr, "ink-ext: %s %s: %d x %d px at %.3f um, %d layers, CT level %d (%.3f um, %d^3 sub-samples)%s\n", S->name, sp->name, Wm, Hm, X->px_um, D, L, X->lev_um[L], R.ss, X->reverse ? ", reversed" : "");
    if (X->norm_sheet && ie_is_koine(X->kind) && !X->stk) {   /* --norm sheet: normalize_robust of the surface's non-zero voxels over <= 512 rows */
        const int nr = Hm < 512 ? Hm : 512; size_t nv = 0, cv = (size_t)1 << 20; uint8_t *v = malloc(cv), *row = malloc((size_t)D * Wm);
        for (int k = 0; k < nr; k++) {
            const int r = (int)((k + 0.5) * Hm / nr); uint8_t *rp1[1] = {row};
            if (render_rows(&R, r, r + 1, rp1, X->nth)) break;
            for (size_t i = 0; i < (size_t)D * Wm; i++) if (row[i]) { if (nv == cv) { cv *= 2; v = realloc(v, cv); } v[nv++] = row[i]; }
        }
        if (nv > 1000) {
            float lo, hi, a, b; ie_robust_params(v, nv, &lo, &hi, &a, &b); ie_set_norm(X->net, 1, lo, hi, a, b);
            fprintf(stderr, "ink-ext: %s: sheet normalisation over %zu voxels of %d rows: p1 %.0f p99 %.0f median %.1f scale %.2f\n", S->name, nv, nr, lo, hi, -b / a, 1 / a);
        } else { ie_set_norm(X->net, 0, 0, 0, 0, 0); fprintf(stderr, "ink-ext: %s: too few voxels for --norm sheet, per-tile normalisation\n", S->name); }
        free(v); free(row);
    }
    float *num = calloc((size_t)Wm * Hm, sizeof *num), *den = calloc((size_t)Wm * Hm, sizeof *den);
    free(X->dump); X->dump = X->dump_req ? calloc((size_t)D * Wm * Hm, 1) : nullptr;
    float *ctl = pctl ? calloc((size_t)Wm * Hm, sizeof *ctl) : nullptr; if (pctl) *pctl = ctl;   /* the surface layer (CT / 255) */
    uint8_t *cov = X->kind == IE_V8IN ? calloc((size_t)Wm * Hm, 1) : nullptr;   /* v8in: max over depth > 0 */
    float *win = malloc((size_t)T * T * sizeof *win);
    if (X->no_overlap) for (int i = 0; i < T * T; i++) win[i] = 1.f;   /* --no-overlap: stride = tile, blend off */
    else if (X->kind == IE_V8IN) gkern64(win, T); else hann2(win, T, 0.001f);
    int nx, ny; int *xs, *ys;
    const int stride = X->no_overlap ? T : X->stride > 0 ? X->stride : sp->stride;
    if (X->no_overlap) { xs = axis_starts_koine(Wm, T, stride, &nx); ys = axis_starts_koine(Hm, T, stride, &ny); }   /* tiles at 0, T, 2T .. and the last at L - T */
    else if (X->kind == IE_V8IN) { xs = axis_starts_v8(Wm, T, stride, &nx); ys = axis_starts_v8(Hm, T, stride, &ny); }
    else if (ie_is_koine(X->kind)) { xs = axis_starts_koine(Wm, T, stride, &nx); ys = axis_starts_koine(Hm, T, stride, &ny); }
    else { xs = axis_starts_hecate(Wm, T, stride, &nx); ys = axis_starts_hecate(Hm, T, stride, &ny); }
    /* rolling row buffer: rows [rb0, rb1) */
    const int RB = T + 16; uint8_t **ring = malloc(RB * sizeof *ring); for (int i = 0; i < RB; i++) ring[i] = malloc((size_t)D * Wm);
    int rb0 = 0, rb1 = 0;
    const int B = X->batch; uint8_t *tiles = malloc((size_t)B * D * T * T); int *tx = malloc(B * sizeof *tx), *ty = malloc(B * sizeof *ty); int nb = 0;
    float *out = malloc((size_t)B * T * T * sizeof *out);
    gworker G = {0}; pthread_mutex_init(&G.mu, nullptr); pthread_cond_init(&G.cv, nullptr);
    G.X = X; G.D = D; G.T = T; G.Wm = Wm; G.Hm = Hm; G.win = win; G.num = num; G.den = den; G.out = out;
    G.tiles = malloc((size_t)B * D * T * T); G.tx = malloc(B * sizeof *tx); G.ty = malloc(B * sizeof *ty);
    pthread_t gth; pthread_create(&gth, nullptr, gpu_worker, &G);
    uint8_t *closed = cov ? calloc((size_t)Wm * Hm, 1) : nullptr; int closed_to = 0;   /* v8in: closed coverage valid for rows < closed_to */
    long long ntiles = 0, nskip = 0; double tr = 0, tg = 0; int fail = 0;
    #define ROW(r) ring[(r) % RB]
    for (int iy = 0; iy < ny && !fail; iy++) {
        const int y0 = ys[iy], need1 = (y0 + T + (cov ? 8 : 0)) < Hm ? y0 + T + (cov ? 8 : 0) : Hm;
        if (y0 < rb0) { fprintf(stderr, "ink-ext: row order\n"); fail = 1; break; }
        if (y0 > rb1) rb1 = y0;
        if (rb0 < y0 - 8) rb0 = y0 - 8 > rb0 ? y0 - 8 : rb0;   /* keep a few rows behind for the v8in closing */
        if (rb0 < 0) rb0 = 0;
        if (need1 > rb1) {
            if (need1 - rb0 > RB) rb0 = need1 - RB;
            if (rb1 < rb0) rb1 = rb0;
            const double t0 = inow(); uint8_t *rp[RB]; for (int r = rb1; r < need1; r++) rp[r - rb1] = ROW(r);
            if (X->stk) {   /* model layer i = stack layer stk0 + i (reversed: stk0 + D - 1 - i), as the official readers select */
                for (int r = rb1; r < need1; r++) for (int i = 0; i < D; i++)
                    memcpy(ROW(r) + (size_t)i * Wm, X->stk[stk0 + (X->reverse ? D - 1 - i : i)] + (size_t)r * Wm, (size_t)Wm);
            } else if (render_rows(&R, rb1, need1, rp, X->nth)) fail = 1;
            if (X->dump) for (int r = rb1; r < need1; r++) for (int i = 0; i < D; i++) memcpy(X->dump + ((size_t)i * Hm + r) * Wm, ROW(r) + (size_t)i * Wm, (size_t)Wm);
            if (ctl) for (int r = rb1; r < need1; r++) for (int x = 0; x < Wm; x++) ctl[(size_t)r * Wm + x] = ROW(r)[(size_t)sp->center * Wm + x] / 255.f;
            if (cov) for (int r = rb1; r < need1; r++) for (int x = 0; x < Wm; x++) {
                uint8_t m = 0; for (int i = 0; i < D; i++) m |= ROW(r)[(size_t)i * Wm + x];
                cov[(size_t)r * Wm + x] = m != 0;
            }
            rb1 = need1; tr += inow() - t0;
        }
        if (cov) {   /* closing (dilate, erode with the 7 x 7 ellipse; outside pixels neutral) of the rows the tiles need */
            const int c1 = need1 - (need1 < Hm ? 6 : 0);
            for (int y = closed_to; y < c1; y++) for (int x = 0; x < Wm; x++) {
                int all = 1;
                for (int a = -3; a <= 3 && all; a++) for (int b = -3; b <= 3 && all; b++) {
                    if (ell7[a + 3][b + 3] != '1') continue;
                    const int yy = y + a, xx = x + b; if (yy < 0 || yy >= Hm || xx < 0 || xx >= Wm) continue;
                    int dil = 0;   /* dilated value at (yy, xx) */
                    for (int c = -3; c <= 3 && !dil; c++) for (int e = -3; e <= 3; e++) {
                        if (ell7[c + 3][e + 3] != '1') continue;
                        const int y2 = yy + c, x2 = xx + e; if (y2 < 0 || y2 >= Hm || x2 < 0 || x2 >= Wm) continue;
                        if (cov[(size_t)y2 * Wm + x2]) { dil = 1; break; }
                    }
                    if (!dil) all = 0;
                }
                closed[(size_t)y * Wm + x] = (uint8_t)all;
            }
            if (c1 > closed_to) closed_to = c1;
        }
        for (int ix = 0; ix <= nx && !fail; ix++) {
            if (ix < nx) {
                const int x0 = xs[ix];
                if (X->render_only) continue;
                if (X->kind == IE_V8IN && !X->no_overlap) {   /* the tile must lie fully inside the closed coverage */
                    int in = 1; for (int y = y0; y < y0 + T && in; y++) for (int x = x0; x < x0 + T; x++) if (!closed[(size_t)y * Wm + x]) { in = 0; break; }
                    if (!in) continue;
                }
                uint8_t *t = tiles + (size_t)nb * D * T * T; int anyv = 0;
                const int ph = y0 + T <= Hm ? T : Hm - y0, pw = x0 + T <= Wm ? T : Wm - x0;
                if (X->kind == IE_HEC24 || X->kind == IE_HEC96) for (int y = 0; y < ph; y++) for (int x = 0; x < pw; x++) den[(size_t)(y0 + y) * Wm + x0 + x] += win[y * T + x];
                memset(t, 0, (size_t)D * T * T);
                for (int i = 0; i < D; i++) for (int y = 0; y < ph; y++) {
                    const uint8_t *src = ROW(y0 + y) + (size_t)i * Wm + x0; uint8_t *dst = t + ((size_t)i * T + y) * T;
                    memcpy(dst, src, (size_t)pw); if (!anyv) for (int x = 0; x < pw; x++) if (src[x]) { anyv = 1; break; }
                }
                if (!anyv && (X->kind != IE_V8IN || X->no_overlap)) { nskip++; continue; }
                tx[nb] = x0; ty[nb] = y0; nb++;
            }
            if (nb == B || (ix == nx && iy == ny - 1 && nb)) {
                if (G.fail) { fail = 1; break; }
                gpu_submit(&G, &tiles, &tx, &ty, nb); nb = 0;
            }
        }
        if (iy % 8 == 0) fprintf(stderr, "\rink-ext: %s row %d/%d: %lld tiles (%lld empty), render %.0fs, net %.0fs   ", sp->name, iy + 1, ny, G.ntiles, nskip, tr, G.tg);
    }
    #undef ROW
    pthread_mutex_lock(&G.mu); G.quit = 1; pthread_cond_broadcast(&G.cv); pthread_mutex_unlock(&G.mu);
    pthread_join(gth, nullptr);
    if (G.fail) fail = 1;
    tg = G.tg; ntiles = G.ntiles;
    free(G.tiles); free(G.tx); free(G.ty); pthread_mutex_destroy(&G.mu); pthread_cond_destroy(&G.cv);
    fprintf(stderr, "\rink-ext: %s %s: %lld tiles (%lld empty skipped), render %.1fs, net %.1fs (%.1f tiles/s), CT chunks decoded %llu\n", S->name, sp->name, ntiles, nskip, tr, tg, tg > 0 ? ntiles / tg : 0, (unsigned long long)X->lev[L].ndec);
    for (size_t i = 0; i < (size_t)Wm * Hm; i++) {
        const float d = den[i]; float p = ie_is_koine(X->kind) ? (d > 1e-6f ? num[i] / d : 0.f) : (d != 0 ? num[i] / d : 0.f);
        num[i] = !(p > 0) ? 0.f : p > 1 ? 1.f : p;
    }
    for (int i = 0; i < RB; i++) free(ring[i]);
    free(ring); free(tiles); free(tx); free(ty); free(out); free(den); free(cov); free(closed); free(win); free(xs); free(ys);
    if (fail) { free(num); return nullptr; }
    return num;
}

/* the model-grid map resampled (bilinear) onto the surface's ufsm-render grid ((w - 1) / scale + 1 wide), 0 where the surface is invalid */
static uint8_t *to_ref_grid(const isurf *S, const float *p, int Wm, int Hm, double du, int *ow, int *oh) {
    const double gs = 1 / S->scale; *ow = (int)((S->w - 1) * gs) + 1; *oh = (int)((S->h - 1) * gs) + 1;
    uint8_t *o = calloc((size_t)*ow * *oh, 1); if (!o) return nullptr;
    for (int py = 0; py < *oh; py++) for (int px = 0; px < *ow; px++) {
        const double gu = px / gs, gv = py / gs; double P[3], N[3], a[3], b[3];
        if (surf_pn(S, gu, gv, P, N, a, b)) continue;
        double mx = gu / du, my = gv / du; if (mx > Wm - 1) mx = Wm - 1; if (my > Hm - 1) my = Hm - 1;
        const int x0 = (int)mx, y0 = (int)my, x1 = x0 + 1 < Wm ? x0 + 1 : x0, y1 = y0 + 1 < Hm ? y0 + 1 : y0; const double fx = mx - x0, fy = my - y0;
        const double v = (1 - fy) * ((1 - fx) * p[(size_t)y0 * Wm + x0] + fx * p[(size_t)y0 * Wm + x1]) + fy * ((1 - fx) * p[(size_t)y1 * Wm + x0] + fx * p[(size_t)y1 * Wm + x1]);
        o[(size_t)py * *ow + px] = (uint8_t)lround(v * 255);
    }
    return o;
}
static int write_gray_jpeg_half(const char *path, const uint8_t *g, int w, int h);
/* the image stretched from its p1 to its p99 over the pixels > 0 (the surface), then as write_gray_jpeg_half */
static int write_gray_jpeg_half_stretch(const char *path, const uint8_t *g, int w, int h) {
    const size_t n = (size_t)w * h; uint64_t hist[256] = {0}, m = 0;
    for (size_t i = 0; i < n; i++) if (g[i]) { hist[g[i]]++; m++; }
    int lo = 0, hi = 255; uint64_t c = 0; int gl = 0;
    for (int v = 0; v < 256; v++) { c += hist[v]; if (!gl && c >= m / 100) { lo = v; gl = 1; } if (c >= m - m / 100) { hi = v; break; } }
    if (hi <= lo) hi = lo + 1;
    uint8_t *s = malloc(n); if (!s) return -1;
    for (size_t i = 0; i < n; i++) { const int v = g[i] ? (g[i] - lo) * 255 / (hi - lo) : 0; s[i] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v); }
    const int rc = write_gray_jpeg_half(path, s, w, h); free(s);
    return rc;
}
static int write_gray_jpeg_half(const char *path, const uint8_t *g, int w, int h) {
    const int ow = w / 2 > 0 ? w / 2 : 1, oh = h / 2 > 0 ? h / 2 : 1;
    uint8_t *half = malloc((size_t)ow * oh), *rgb = malloc((size_t)ow * oh * 3);
    if (!half || !rgb || pil_resize(g, w, h, 1, half, ow, oh, PIL_BOX, 8)) { free(half); free(rgb); return -1; }
    for (size_t i = 0; i < (size_t)ow * oh; i++) rgb[3 * i] = rgb[3 * i + 1] = rgb[3 * i + 2] = half[i];
    const int rc = jpeg_write_rgb(path, ow, oh, rgb, 90); free(half); free(rgb);
    return rc;
}

/* raw mode (parity tests): tiles of the model's native shape -> its logits */
static int raw_mode(int argc, char **argv, ie_net *net, int kind) {
    const char *in = iopt(argc, argv, "--raw-in", nullptr), *outp = iopt(argc, argv, "--raw-out", nullptr);
    const ie_spec *sp = &ie_specs[kind]; const size_t per = (size_t)sp->D * sp->T * sp->T;
    FILE *f = fopen(in, "rb"); if (!f) { fprintf(stderr, "ink-ext: cannot open %s\n", in); return 1; }
    fseek(f, 0, SEEK_END); const long sz = ftell(f); fseek(f, 0, SEEK_SET);
    if (sz <= 0 || sz % (long)per) { fclose(f); fprintf(stderr, "ink-ext: %s is not a multiple of %zu bytes (%d x %d x %d tiles)\n", in, per, sp->D, sp->T, sp->T); return 1; }
    const int n = (int)(sz / (long)per); uint8_t *t = malloc((size_t)sz); const size_t got = fread(t, 1, (size_t)sz, f); fclose(f);
    if (got != (size_t)sz) { free(t); return 1; }
    const int B = atoi(iopt(argc, argv, "--batch", "2")); float *o = malloc((size_t)n * sp->T * sp->T * sizeof *o);
    const double t0 = inow();
    for (int i = 0; i < n; i += B) { const int b = n - i < B ? n - i : B; if (ie_forward(net, t + (size_t)i * per, b, o + (size_t)i * sp->T * sp->T, 1)) { free(t); free(o); return 1; } }
    xnet_sync(); const double dt = inow() - t0;
    fprintf(stderr, "ink-ext: %d %s tiles in %.2fs (%.1f ms / tile, %.1f TFLOP conv, %.1f TFLOP/s)\n", n, sp->name, dt, 1e3 * dt / n, 2e-12 * xnet_macs(), 2e-12 * xnet_macs() / dt);
    f = fopen(outp, "wb"); const size_t wn = f ? fwrite(o, sizeof *o, (size_t)n * sp->T * sp->T, f) : 0; if (f) fclose(f);
    free(t); free(o);
    return wn == (size_t)n * sp->T * sp->T ? 0 : 1;
}

/* ufsm ink-ext --model M --weights W (--surface DIR ..|--surfaces LIST) --ct ROOT KEY --out DIR [options] */
int cmd_ink_ext(int argc, char **argv) {
    const char *mname = iopt(argc, argv, "--model", nullptr), *weights = iopt(argc, argv, "--weights", nullptr);
    if (!mname || !weights) {
        fprintf(stderr, "usage: ufsm ink-ext --model v8in|hecate24|hecate96|reader3|ink9um --weights W[,W2..] --surface TIFXYZ [--surface ..] --ct ROOT KEY --out DIR\n"
                        "         [--reverse] [--stride S] [--px-um U] [--batch B] [--gpu 0] [--threads 16] [--ct-mem 3] [--cache DIR] [--native] [--suffix S] [--prefix P]\n"
                        "         [--ct-local ROOT --local-levels 1,2] [--ct-um 2.399] [--surf-um U] [--no-overlap] [--norm tile|sheet] [--blend official|norm] [--grid-crop c,r,w,h]\n"
                        "         [--ct-tif] [--jpeg PATH] [--jpeg-stretch] [--layers DIR [--px-crop x,y,w,h]] [--dump-layers DIR] [--render-only]\n"
                        "         [--shift S] [--zwin a,b,..] [--calib]\n"
                        "       ufsm ink-ext --model M --weights W --raw-in TILES.u8 --raw-out LOGITS.f32 [--batch 2]   (parity: native tiles -> logits)\n"
                        "  --weights W1,W2,..: an ensemble (mean probability) of checkpoints of the model (e.g. ink9um seed42 + seed43)\n"
                        "  --shift S: every layer S layers further along +N (fractional ok; --layers: rounded, the window moves in the stack)\n"
                        "  --zwin a,b,..: also average over these extra layer shifts (the ink_9um card's z-window ensemble, e.g. -1,0,1)\n"
                        "  --calib: write (p - 0.25) / 0.5 clipped to [0, 1] (koine BCE label smoothing 0.5: reader3 / ink9um no-ink sits near 0.25)\n"); return 2;
    }
    const int kind = ie_model_kind(mname); if (kind < 0) { fprintf(stderr, "ink-ext: unknown model %s\n", mname); return 2; }
    if (xnet_init(atoi(iopt(argc, argv, "--gpu", "0")))) return 1;
    enum { MAXE = 16 }; ie_net *nets[MAXE]; int nnet = 0;
    {   /* --weights W1,W2,..: each checkpoint loaded once */
        char wl[4096]; snprintf(wl, sizeof wl, "%s", weights);
        for (char *sv = nullptr, *t = strtok_r(wl, ",", &sv); t && nnet < MAXE; t = strtok_r(nullptr, ",", &sv)) {
            const double tl = inow(); nets[nnet] = ie_load(kind, t); if (!nets[nnet]) return 1;
            fprintf(stderr, "ink-ext: %s loaded from %s in %.1fs\n", mname, t, inow() - tl); nnet++;
        }
        if (!nnet) { fprintf(stderr, "ink-ext: no --weights\n"); return 2; }
    }
    ie_net *net = nets[0];
    if (iopt(argc, argv, "--raw-in", nullptr)) return raw_mode(argc, argv, net, kind);
    double zw[MAXE]; int nzw = 0;
    {   /* --zwin: extra layer shifts averaged (default: 0 only) */
        const char *z = iopt(argc, argv, "--zwin", nullptr);
        if (z) { char zb[512]; snprintf(zb, sizeof zb, "%s", z); for (char *sv = nullptr, *t = strtok_r(zb, ",", &sv); t && nzw < MAXE; t = strtok_r(nullptr, ",", &sv)) zw[nzw++] = atof(t); }
        if (!nzw) zw[nzw++] = 0;
    }
    const int calib = iflag(argc, argv, "--calib");
    int ci = -1; for (int i = 2; i + 2 < argc; i++) if (!strcmp(argv[i], "--ct")) ci = i;
    const char *outdir = iopt(argc, argv, "--out", nullptr), *cache = iopt(argc, argv, "--cache", nullptr), *suffix = iopt(argc, argv, "--suffix", "");
    const char *ldir = iopt(argc, argv, "--layers", nullptr);   /* a published surface volume (NN.tif, ascending normal offsets) instead of our render */
    if ((ci < 0 && !ldir) || !outdir) { fprintf(stderr, "ink-ext: need --ct ROOT KEY (or --layers DIR) and --out DIR\n"); return 2; }
    irun X = {0};
    if (ldir) {
        for (;; X.stk_n++) {
            char lp[1200]; snprintf(lp, sizeof lp, "%s/%02d.tif", ldir, X.stk_n); int w, h; uint8_t *im = img_read_u8(lp, &w, &h);
            if (!im) break;
            if (X.stk_n && (w != X.stk_w || h != X.stk_h)) { fprintf(stderr, "ink-ext: %s: size differs\n", lp); return 1; }
            X.stk = realloc(X.stk, (size_t)(X.stk_n + 1) * sizeof *X.stk); X.stk[X.stk_n] = im; X.stk_w = w; X.stk_h = h;
        }
        if (!X.stk_n) { fprintf(stderr, "ink-ext: no %s/00.tif\n", ldir); return 1; }
        const char *pc = iopt(argc, argv, "--px-crop", nullptr); int c[4];   /* x,y,w,h of the layer images */
        if (pc) {
            if (sscanf(pc, "%d,%d,%d,%d", c, c + 1, c + 2, c + 3) != 4 || c[0] < 0 || c[1] < 0 || c[0] + c[2] > X.stk_w || c[1] + c[3] > X.stk_h) { fprintf(stderr, "ink-ext: bad --px-crop\n"); return 2; }
            for (int i = 0; i < X.stk_n; i++) {
                uint8_t *o = malloc((size_t)c[2] * c[3]);
                for (int y = 0; y < c[3]; y++) memcpy(o + (size_t)y * c[2], X.stk[i] + (size_t)(c[1] + y) * X.stk_w + c[0], (size_t)c[2]);
                free(X.stk[i]); X.stk[i] = o;
            }
            X.stk_w = c[2]; X.stk_h = c[3];
        }
        fprintf(stderr, "ink-ext: --layers %s: %d layers of %d x %d\n", ldir, X.stk_n, X.stk_w, X.stk_h);
    }
    X.dump_req = iopt(argc, argv, "--dump-layers", nullptr) != nullptr; X.render_only = iflag(argc, argv, "--render-only"); X.kind = kind; X.net = net; X.sp = &ie_specs[kind]; X.stride = atoi(iopt(argc, argv, "--stride", "0")); X.reverse = iflag(argc, argv, "--reverse");
    X.blend_norm = !strcmp(iopt(argc, argv, "--blend", "official"), "norm");   /* v8in: sum(p g) / sum(g) instead of inference.py sum(p g) / count (its stride grid) */
    X.no_overlap = iflag(argc, argv, "--no-overlap");   /* stride = the tile, uniform weights (no blending), empty tiles skipped, v8in without its coverage-mask rule */
    X.norm_sheet = !strcmp(iopt(argc, argv, "--norm", "tile"), "sheet");   /* Reader / ink9um input normalisation: tile (official, per tile) or sheet */
    X.nth = atoi(iopt(argc, argv, "--threads", "16")); X.px_um = atof(iopt(argc, argv, "--px-um", "0")); if (!(X.px_um > 0)) X.px_um = X.sp->px_um;
    X.ct_mem = (size_t)(atof(iopt(argc, argv, "--ct-mem", "3")) * (1 << 30));
    X.batch = atoi(iopt(argc, argv, "--batch", "0"));
    const double gmem = atof(iopt(argc, argv, "--gpu-mem", "0"));   /* GB of device memory for activations + cache (default: free - 1 GB) */
    if (gmem > 0) xnet_set_budget((size_t)(gmem * (1 << 30)));
    if (X.batch < 1) {   /* fit the batch to the free device memory (half of it) */
        size_t fr = 0, tot = 0; xnet_mem_info(&fr, &tot);
        X.batch = (int)(fr / 2 / ie_tile_bytes_gpu(kind)); if (X.batch < 1) X.batch = 1; if (X.batch > 32) X.batch = 32;
    }
    const double ct_um = atof(iopt(argc, argv, "--ct-um", "2.399"));
    if (ci < 0) { X.nlev = 1; X.lev_um[0] = ct_um; goto surfaces; }   /* --layers without CT */
    store *st = store_open(argv[ci + 1]); if (!st) { fprintf(stderr, "ink-ext: cannot open %s\n", argv[ci + 1]); return 1; }
    /* --ct-local ROOT --local-levels 1,2: those levels read from a complete local copy (missing chunks there would read as 0) */
    const char *lroot = iopt(argc, argv, "--ct-local", nullptr), *llev = iopt(argc, argv, "--local-levels", "");
    store *lst = lroot ? store_open(lroot) : nullptr;
    for (int l = 0; l < 6; l++) {
        char k[1100], lk[8]; snprintf(k, sizeof k, "%s/%d", argv[ci + 2], l);
        int local = 0; if (lst) for (const char *q = llev; *q; q++) if (*q - '0' == l && (q == llev || q[-1] == ',') && (!q[1] || q[1] == ',')) local = 1;
        (void)lk;
        z3 *z = z3_open(local ? lst : st, k, local ? nullptr : cache); if (!z) break;
        if (local) fprintf(stderr, "ink-ext: CT level %d from %s\n", l, lroot);
        const z3_meta *m = z3_meta_of(z);
        if (m->chunk[0] != CE || m->chunk[1] != CE || m->chunk[2] != CE) { fprintf(stderr, "ink-ext: %s: chunks must be 128^3\n", k); return 1; }
        ctlev *L = &X.lev[l]; L->z = z; for (int d = 0; d < 3; d++) L->shape[d] = m->shape[d];
        L->budget = X.ct_mem; L->zero = calloc((size_t)CE * CE * CE, 1); X.nlev = l + 1;
        X.lev_um[l] = ct_um * (1 << l);   /* OME scales are relative (1, 2, 4 ..): the level-0 voxel size is --ct-um */
    }
    if (!X.nlev) { fprintf(stderr, "ink-ext: no CT levels under %s %s: %s\n", argv[ci + 1], argv[ci + 2], z3_error()); return 1; }
    fprintf(stderr, "ink-ext: batch %d, CT %s %s (%d levels, %.3f um), stride %d\n", X.batch, argv[ci + 1], argv[ci + 2], X.nlev, X.lev_um[0], X.stride > 0 ? X.stride : X.sp->stride);
surfaces:;
    char p[1400]; snprintf(p, sizeof p, "mkdir -p '%s'", outdir); if (system(p)) return 1;
    int rc = 0, ns = 0, nsurf = 0;
    for (int a = 2; a + 1 < argc; a++) nsurf += !strcmp(argv[a], "--surface");
    for (int a = 2; a < argc; a++) {
        const int lonly = X.stk && !nsurf && a == 2;   /* --layers without --surface: one pass named --name (default "layers") */
        if (!lonly && (a + 1 >= argc || strcmp(argv[a], "--surface"))) continue;
        isurf S;
        if (lonly) { memset(&S, 0, sizeof S); snprintf(S.name, sizeof S.name, "%s", iopt(argc, argv, "--name", "layers")); S.scale = 1; S.vox_um = X.px_um; S.w = S.h = 2; }
        else if (isurf_load(&S, argv[a + 1])) { fprintf(stderr, "ink-ext: cannot read tifxyz %s\n", argv[a + 1]); rc = 1; continue; }
        { const double su = atof(iopt(argc, argv, "--surf-um", "0")); if (su > 0) S.vox_um = su; }   /* tifxyz voxel size when its meta has none (default 2.399) */
        const char *gc = iopt(argc, argv, "--grid-crop", nullptr); int cc[4];   /* c0,r0,cols,rows of the tifxyz grid */
        if (gc && (sscanf(gc, "%d,%d,%d,%d", cc, cc + 1, cc + 2, cc + 3) != 4 || isurf_crop(&S, cc[0], cc[1], cc[2], cc[3]))) { fprintf(stderr, "ink-ext: bad --grid-crop\n"); return 2; }
        const char *pre = iopt(argc, argv, "--prefix", ""); char nm[600]; snprintf(nm, sizeof nm, "%s%s", pre, S.name);
        if (iflag(argc, argv, "--inward") && !lonly) {   /* the layer order whose read runs radially inward (the published PHerc1447 renders' order, the w058 control) */
            const char *oc = iopt(argc, argv, "--center", nullptr); double cx = X.lev[0].shape[2] * 0.5, cy = X.lev[0].shape[1] * 0.5;
            if (oc) sscanf(oc, "%lf,%lf", &cx, &cy);
            double sum = 0; long cnt = 0;
            for (int r = 0; r + 1 < S.h; r += 4) for (int q = 0; q + 1 < S.w; q += 4) {
                double P[3], N[3], du[3], dv[3]; if (surf_pn(&S, q + 0.5, r + 0.5, P, N, du, dv)) continue;
                const double ry = P[1] - cy, rx = P[2] - cx, rl = sqrt(rx * rx + ry * ry); if (rl < 1) continue;
                sum += (N[1] * ry + N[2] * rx) / rl; cnt++;
            }
            const int outward = cnt && sum / cnt > 0; X.reverse = outward ^ iflag(argc, argv, "--reverse");
            fprintf(stderr, "ink-ext: %s: normal %s (mean radial %.2f about %.0f,%.0f) -> %s layer order\n", S.name, outward ? "outward" : "inward", cnt ? sum / cnt : 0, cx, cy, X.reverse ? "reversed" : "forward");
        }
        const double t0 = inow(); int Wm, Hm; double du;
        float *ctl = nullptr; const int want_ct = iflag(argc, argv, "--ct-tif");
        /* the mean probability over the checkpoints (--weights a,b) x layer windows (--zwin); the first pass gives --ct-tif / --dump-layers */
        const double shift0 = atof(iopt(argc, argv, "--shift", "0")); const int dreq = X.dump_req;
        float *pm = nullptr; uint8_t *dump0 = nullptr; int npass = 0, bad = 0;
        for (int k = 0; k < nnet && !bad; k++) for (int z = 0; z < nzw && !bad; z++) {
            X.net = nets[k]; X.shift = shift0 + zw[z]; X.dump_req = dreq && !npass;
            if (nnet * nzw > 1) fprintf(stderr, "ink-ext: pass %d/%d: weights %d, shift %+.2f layers\n", npass + 1, nnet * nzw, k, X.shift);
            float *q = run_surface(&X, &S, &Wm, &Hm, &du, want_ct && !npass ? &ctl : nullptr);
            if (!q) { bad = 1; break; }
            if (!npass) { pm = q; dump0 = X.dump; X.dump = nullptr; }
            else { for (size_t i = 0; i < (size_t)Wm * Hm; i++) pm[i] += q[i]; free(q); }
            npass++;
        }
        X.dump = dump0; X.dump_req = dreq;
        if (bad || !pm) { rc = 1; free(pm); free(ctl); free(X.dump); X.dump = nullptr; isurf_free(&S); continue; }
        for (size_t i = 0; i < (size_t)Wm * Hm; i++) {
            float v = pm[i] / npass;
            if (calib) { v = (v - 0.25f) / 0.5f; v = v < 0 ? 0.f : v > 1 ? 1.f : v; }
            pm[i] = v;
        }
        if (ctl) {   /* --ct-tif: the model's surface layer on the render grid, <surf>_<model>_ct.tif */
            int cw, chh; uint8_t *cg = to_ref_grid(&S, ctl, Wm, Hm, du, &cw, &chh);
            snprintf(p, sizeof p, "%s/%s_%s%s_ct.tif", outdir, nm, mname, suffix);
            if (!cg || tiffw_write(p, cw, chh, 1, cg, 1)) { fprintf(stderr, "ink-ext: cannot write %s\n", p); rc = 1; }
            free(cg); free(ctl);
        }
        if (iflag(argc, argv, "--native")) {
            uint8_t *g = malloc((size_t)Wm * Hm); for (size_t i = 0; i < (size_t)Wm * Hm; i++) g[i] = (uint8_t)lroundf(pm[i] * 255);
            snprintf(p, sizeof p, "%s/%s_%s%s_native.tif", outdir, nm, mname, suffix); if (tiffw_write(p, Wm, Hm, 1, g, 1)) rc = 1;
            free(g);
        }
        if (X.dump) {   /* --dump-layers DIR: the model's input stack (model layer order) as DIR/NN.tif */
            const char *dd = iopt(argc, argv, "--dump-layers", "."); snprintf(p, sizeof p, "mkdir -p '%s'", dd); if (system(p)) rc = 1;
            for (int i = 0; i < X.sp->D; i++) { snprintf(p, sizeof p, "%s/%02d.tif", dd, i); if (tiffw_write(p, Wm, Hm, 1, X.dump + (size_t)i * Wm * Hm, 1)) rc = 1; }
            free(X.dump); X.dump = nullptr;
        }
        int ow, oh; uint8_t *g;
        if (X.stk) {   /* --layers: the stack's pixel grid is the output grid */
            ow = Wm; oh = Hm; g = malloc((size_t)Wm * Hm); for (size_t i = 0; i < (size_t)Wm * Hm; i++) g[i] = (uint8_t)lroundf(pm[i] * 255);
        } else g = to_ref_grid(&S, pm, Wm, Hm, du, &ow, &oh);
        snprintf(p, sizeof p, "%s/%s_%s%s.tif", outdir, nm, mname, suffix);
        if (!g || tiffw_write(p, ow, oh, 1, g, 1)) { fprintf(stderr, "ink-ext: cannot write %s\n", p); rc = 1; }
        /* --jpeg PATH: the half-res JPEG there (default <out>/<name>_<model><suffix>.jpg); --jpeg-stretch: p1-p99 over the surface */
        const char *jo = iopt(argc, argv, "--jpeg", nullptr);
        if (jo) snprintf(p, sizeof p, "%s", jo); else snprintf(p, sizeof p, "%s/%s_%s%s.jpg", outdir, nm, mname, suffix);
        if (g && (iflag(argc, argv, "--jpeg-stretch") ? write_gray_jpeg_half_stretch(p, g, ow, oh) : write_gray_jpeg_half(p, g, ow, oh))) { fprintf(stderr, "ink-ext: cannot write %s\n", p); rc = 1; }
        double mean = 0; size_t cnt = 0; for (size_t i = 0; i < (size_t)ow * oh; i++) if (g && g[i]) { mean += g[i]; cnt++; }
        fprintf(stderr, "ink-ext: %s -> %s/%s_%s%s.tif (%d x %d, mean %.3f over %zu px), %.1fs\n", S.name, outdir, nm, mname, suffix, ow, oh, cnt ? mean / cnt / 255 : 0, cnt, inow() - t0);
        free(g); free(pm); isurf_free(&S); ns++;
    }
    if (!ns && !rc) fprintf(stderr, "ink-ext: no --surface given\n");
    return rc;
}

/* ufsm ink-ensemble <out.tif> --ink [W:]PATH.tif ... [--norm stretch|rank|none] [--lo-pct 50] [--hi-pct 99.5] [--jpeg OUT.jpg]
   Per model its values over the pixels where any model is > 0: stretch = (v - p_lo) / (p_hi - p_lo) clamped to [0, 1] (each
   model's median .. 99.5th percentile, Reader's 0.25 .. 0.75 compression and the models' different ink fractions removed),
   rank = the value's percentile; then the weighted mean (weights W, default 1) as uint8. */
int cmd_ink_ensemble(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: ufsm ink-ensemble <out.tif> --ink [W:]PATH.tif .. [--norm stretch|rank|none] [--lo-pct 50] [--hi-pct 99.5] [--jpeg OUT.jpg]\n"); return 2; }
    const char *norm = iopt(argc, argv, "--norm", "stretch"); const double plo = atof(iopt(argc, argv, "--lo-pct", "50")), phi = atof(iopt(argc, argv, "--hi-pct", "99.5"));
    enum { MAXI = 16 }; uint8_t *im[MAXI]; double wt[MAXI]; int n = 0, W = 0, H = 0;
    for (int a = 2; a + 1 < argc; a++) {
        if (strcmp(argv[a], "--ink")) continue;
        const char *s = argv[a + 1], *c = strchr(s, ':'); double w = 1;
        if (c && c - s < 16) { char *e; const double v = strtod(s, &e); if (e == c) { w = v; s = c + 1; } }
        int w2, h2; uint8_t *g = img_read_u8(s, &w2, &h2);
        if (!g || n == MAXI || (n && (w2 != W || h2 != H))) { fprintf(stderr, "ink-ensemble: cannot read %s (or size differs)\n", s); return 1; }
        W = w2; H = h2; im[n] = g; wt[n++] = w;
    }
    if (!n) { fprintf(stderr, "ink-ensemble: no --ink\n"); return 2; }
    const size_t np = (size_t)W * H; uint8_t *mask = calloc(np, 1); size_t nm = 0;
    for (size_t i = 0; i < np; i++) { for (int k = 0; k < n; k++) if (im[k][i]) { mask[i] = 1; break; } nm += mask[i]; }
    float *acc = calloc(np, sizeof *acc); double wsum = 0; for (int k = 0; k < n; k++) wsum += wt[k];
    for (int k = 0; k < n; k++) {
        uint64_t h[256] = {0}; for (size_t i = 0; i < np; i++) if (mask[i]) h[im[k][i]]++;
        float lut[256];
        if (!strcmp(norm, "rank")) { uint64_t c = 0; for (int v = 0; v < 256; v++) { lut[v] = nm ? (float)((c + h[v] * 0.5) / nm) : 0; c += h[v]; } }
        else if (!strcmp(norm, "none")) for (int v = 0; v < 256; v++) lut[v] = v / 255.f;
        else {
            double lo = 0, hi = 255; uint64_t c = 0; int gl = 0, gh = 0;
            for (int v = 0; v < 256; v++) { c += h[v]; if (!gl && c >= plo / 100 * nm) { lo = v; gl = 1; } if (!gh && c >= phi / 100 * nm) { hi = v; gh = 1; } }
            if (hi <= lo) hi = lo + 1;
            for (int v = 0; v < 256; v++) { const double t = (v - lo) / (hi - lo); lut[v] = (float)(t < 0 ? 0 : t > 1 ? 1 : t); }
            fprintf(stderr, "ink-ensemble: input %d: stretch %g .. %g (p%g .. p%g)\n", k, lo / 255, hi / 255, plo, phi);
        }
        for (size_t i = 0; i < np; i++) if (mask[i]) acc[i] += (float)(wt[k] / wsum) * lut[im[k][i]];
    }
    uint8_t *o = malloc(np); for (size_t i = 0; i < np; i++) o[i] = mask[i] ? (uint8_t)lroundf(acc[i] * 255) : 0;
    const int npk = atoi(iopt(argc, argv, "--peaks", "0")), pw = atoi(iopt(argc, argv, "--peak-win", "1024"));
    if (npk > 0 && pw > 0 && pw <= W && pw <= H) {   /* the npk non-overlapping pw x pw windows of highest mean ensemble (>= half covered) */
        double *I = calloc((size_t)(W + 1) * (H + 1), sizeof *I), *C = calloc((size_t)(W + 1) * (H + 1), sizeof *C);
        for (int y = 0; y < H; y++) { double r = 0, c = 0; for (int x = 0; x < W; x++) { r += o[(size_t)y * W + x]; c += mask[(size_t)y * W + x];
            I[(size_t)(y + 1) * (W + 1) + x + 1] = I[(size_t)y * (W + 1) + x + 1] + r; C[(size_t)(y + 1) * (W + 1) + x + 1] = C[(size_t)y * (W + 1) + x + 1] + c; } }
        int px[64], py[64], got = 0; const int st = pw / 8 > 0 ? pw / 8 : 1;
        for (int k = 0; k < npk && k < 64; k++) {
            double best = -1; int bx = -1, by = -1;
            for (int y = 0; y + pw <= H; y += st) for (int x = 0; x + pw <= W; x += st) {
                int ov = 0; for (int q = 0; q < got && !ov; q++) ov = abs(x - px[q]) < pw && abs(y - py[q]) < pw;
                if (ov) continue;
                const size_t a = (size_t)y * (W + 1) + x, b = a + pw, c = a + (size_t)pw * (W + 1), d = c + pw;
                const double cnt = C[d] - C[b] - C[c] + C[a]; if (cnt < 0.5 * pw * pw) continue;
                const double m = (I[d] - I[b] - I[c] + I[a]) / cnt; if (m > best) { best = m; bx = x; by = y; }
            }
            if (bx < 0) break;
            px[got] = bx; py[got] = by; got++;
            printf("peak %d,%d,%d,%d %.4f\n", bx, by, pw, pw, best / 255);
        }
        free(I); free(C);
    }
    int rc = tiffw_write(argv[2], W, H, 1, o, 1) ? 1 : 0;
    const char *jp = iopt(argc, argv, "--jpeg", nullptr); if (jp && (iflag(argc, argv, "--jpeg-stretch") ? write_gray_jpeg_half_stretch(jp, o, W, H) : write_gray_jpeg_half(jp, o, W, H))) rc = 1;
    if (rc) fprintf(stderr, "ink-ensemble: cannot write %s\n", argv[2]);
    for (int k = 0; k < n; k++) free(im[k]);
    free(mask); free(acc); free(o);
    return rc;
}

/* ufsm tifxyz-map <src-tifxyz> <src-image.tif> <dst-tifxyz> <out.tif> [--max-dist 1.16] [--mask OUT_MASK.tif]
   An image on one surface's render grid (src: ufsm render --scale 1 / published layers, pixel (x, y) = grid (x scale, y scale))
   carried onto another surface's render grid: per dst pixel its 3D point, the nearest src grid vertex within --max-dist voxels
   (default 1.16 = 10 um at 8.64 um), the src image at that vertex's pixel. Unmatched pixels 0; --mask: 255 where matched. */
typedef struct { int64_t key; int32_t i; } vkey;
static int vk_cmp(const void *a, const void *b) { const int64_t p = ((const vkey *)a)->key, q = ((const vkey *)b)->key; return (p > q) - (p < q); }
static inline int64_t bkey(int64_t z, int64_t y, int64_t x) { return (z << 42) | ((y & 0x1fffff) << 21) | (x & 0x1fffff); }
int cmd_tifxyz_map(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: ufsm tifxyz-map <src-tifxyz> <src-image.tif> <dst-tifxyz> <out.tif> [--max-dist 1.16] [--mask MASK.tif]\n"); return 2; }
    isurf A, Bs; if (isurf_load(&A, argv[2]) || isurf_load(&Bs, argv[4])) { fprintf(stderr, "tifxyz-map: cannot read the tifxyz\n"); return 1; }
    int iw, ih; uint8_t *img = img_read_u8(argv[3], &iw, &ih); if (!img) { fprintf(stderr, "tifxyz-map: cannot read %s\n", argv[3]); return 1; }
    const double md = atof(iopt(argc, argv, "--max-dist", "1.16"));
    double sr = 0; for (size_t i = 0; i + 1 < (size_t)A.w * A.h; i++) if (A.Z[i] >= 0 && A.Z[i + 1] >= 0) { const double a0 = A.Z[i + 1] - A.Z[i], a1 = A.Y[i + 1] - A.Y[i], a2 = A.X[i + 1] - A.X[i]; sr = fmax(sr, 0); sr = sqrt(a0 * a0 + a1 * a1 + a2 * a2); break; }
    sr = fmax(sr * 0.75 + md, md);   /* vertex search radius: ~ the src grid spacing (in-plane) plus the normal tolerance */
    const double bs = sr;
    const size_t nv = (size_t)A.w * A.h; vkey *V = malloc(nv * sizeof *V); size_t n = 0;
    for (size_t i = 0; i < nv; i++) if (A.Z[i] >= 0) V[n++] = (vkey){bkey((int64_t)floor(A.Z[i] / bs), (int64_t)floor(A.Y[i] / bs), (int64_t)floor(A.X[i] / bs)), (int32_t)i};
    qsort(V, n, sizeof *V, vk_cmp);
    const double gs = 1 / Bs.scale; const int ow = (int)((Bs.w - 1) * gs) + 1, oh = (int)((Bs.h - 1) * gs) + 1;
    uint8_t *o = calloc((size_t)ow * oh, 1), *mk = calloc((size_t)ow * oh, 1); size_t matched = 0, valid = 0;
    for (int py = 0; py < oh; py++) for (int px = 0; px < ow; px++) {
        double P[3], N[3], a[3], b[3]; if (surf_pn(&Bs, px / gs, py / gs, P, N, a, b)) continue;
        valid++;
        const int64_t c0 = (int64_t)floor(P[0] / bs), c1 = (int64_t)floor(P[1] / bs), c2 = (int64_t)floor(P[2] / bs);
        double best = sr * sr; int32_t bi = -1;   /* the nearest src vertex within the search radius; the normal distance is tested below */
        for (int dz = -1; dz <= 1; dz++) for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
            const vkey k = {bkey(c0 + dz, c1 + dy, c2 + dx), 0}; vkey *f = bsearch(&k, V, n, sizeof *V, vk_cmp); if (!f) continue;
            while (f > V && f[-1].key == k.key) f--;
            for (; f < V + n && f->key == k.key; f++) {
                const double d0 = A.Z[f->i] - P[0], d1 = A.Y[f->i] - P[1], d2 = A.X[f->i] - P[2], d = d0 * d0 + d1 * d1 + d2 * d2;
                if (d <= best) { best = d; bi = f->i; }
            }
        }
        if (bi < 0) continue;
        /* the point in the src vertex's tangent frame: P - V = a du + b dv + h n (du, dv one grid step), |h| <= max-dist, the src
           image at the projected position (grid c + a, r + b) */
        const int c = bi % A.w, r = bi / A.w; if (c + 1 >= A.w || r + 1 >= A.h) continue;
        const size_t i1 = (size_t)bi + 1, i2 = (size_t)bi + A.w; if (A.Z[i1] < 0 || A.Z[i2] < 0) continue;
        const double du[3] = {A.Z[i1] - A.Z[bi], A.Y[i1] - A.Y[bi], A.X[i1] - A.X[bi]}, dv[3] = {A.Z[i2] - A.Z[bi], A.Y[i2] - A.Y[bi], A.X[i2] - A.X[bi]};
        const double d[3] = {P[0] - A.Z[bi], P[1] - A.Y[bi], P[2] - A.X[bi]};
        double nn[3] = {du[1] * dv[2] - du[2] * dv[1], du[2] * dv[0] - du[0] * dv[2], du[0] * dv[1] - du[1] * dv[0]};
        const double nl = sqrt(nn[0] * nn[0] + nn[1] * nn[1] + nn[2] * nn[2]); if (nl < 1e-9) continue;
        const double h = (d[0] * nn[0] + d[1] * nn[1] + d[2] * nn[2]) / nl; if (fabs(h) > md) continue;
        const double uu = du[0] * du[0] + du[1] * du[1] + du[2] * du[2], vv = dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2], uv = du[0] * dv[0] + du[1] * dv[1] + du[2] * dv[2];
        const double du_d = du[0] * d[0] + du[1] * d[1] + du[2] * d[2], dv_d = dv[0] * d[0] + dv[1] * d[1] + dv[2] * d[2], det = uu * vv - uv * uv;
        if (fabs(det) < 1e-12) continue;
        const double ga = (vv * du_d - uv * dv_d) / det, gb = (uu * dv_d - uv * du_d) / det;
        if (fabs(ga) > 1.5 || fabs(gb) > 1.5) continue;
        const int sx = (int)lround((c + ga) / A.scale), sy = (int)lround((r + gb) / A.scale);
        if (sx < 0 || sy < 0) continue;
        if (sx < iw && sy < ih) { o[(size_t)py * ow + px] = img[(size_t)sy * iw + sx]; mk[(size_t)py * ow + px] = 255; matched++; }
    }
    fprintf(stderr, "tifxyz-map: %zu of %zu pixels within %.2f voxels of %s (%.1f%%)\n", matched, valid, md, argv[2], valid ? 100.0 * matched / valid : 0);
    int rc = tiffw_write(argv[5], ow, oh, 1, o, 1) ? 1 : 0;
    const char *mp = iopt(argc, argv, "--mask", nullptr); if (mp && tiffw_write(mp, ow, oh, 1, mk, 1)) rc = 1;
    free(V); free(o); free(mk); free(img); isurf_free(&A); isurf_free(&Bs);
    return rc;
}

/* ufsm tifxyz-orient <tifxyz>.. --center x,y: per surface the share of its vertices whose unit normal du x dv (ink-ext's forward
   layer direction) points away from the vertical axis through (x, y) (radially outward), and the mean radial component */
int cmd_tifxyz_orient(int argc, char **argv) {
    double cx = 0, cy = 0; const char *c = iopt(argc, argv, "--center", nullptr);
    if (argc < 3 || !c || sscanf(c, "%lf,%lf", &cx, &cy) != 2) { fprintf(stderr, "usage: ufsm tifxyz-orient <tifxyz>.. --center x,y\n"); return 2; }
    for (int a = 2; a < argc; a++) {
        if (!strcmp(argv[a], "--center")) { a++; continue; }
        isurf S; if (isurf_load(&S, argv[a])) { fprintf(stderr, "tifxyz-orient: cannot read %s\n", argv[a]); continue; }
        long n = 0, out = 0; double sum = 0;
        for (int r = 0; r + 1 < S.h; r += 2) for (int q = 0; q + 1 < S.w; q += 2) {
            double P[3], N[3], du[3], dv[3]; if (surf_pn(&S, q + 0.5, r + 0.5, P, N, du, dv)) continue;
            const double ry = P[1] - cy, rx = P[2] - cx, rl = sqrt(rx * rx + ry * ry); if (rl < 1) continue;
            const double rad = (N[1] * ry + N[2] * rx) / rl; sum += rad; n++; out += rad > 0;
        }
        printf("%s: %ld samples, normal outward %.1f%%, mean radial %.3f\n", argv[a], n, n ? 100.0 * out / n : 0, n ? sum / n : 0);
        isurf_free(&S);
    }
    return 0;
}

/* ufsm tifxyz-shards <tifxyz> --shard S [--div F] [--margin M]: the shards (S^3 on the grid of level-0 voxels / F, global indices,
   "z y x" per line on stdout) within M level-0 voxels of any valid vertex of the surface; stderr: the shard box "z0,y0,x0,nz,ny,nx" in
   level voxels (ufsm predict --box, with --full-grid 1 --resume 1 and the complement as predict-done-<level>.txt: only the shards a
   surface passes through) */
int cmd_tifxyz_shards(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: ufsm tifxyz-shards <tifxyz> --shard S [--div F] [--margin M]\n"); return 2; }
    isurf S; if (isurf_load(&S, argv[2])) { fprintf(stderr, "tifxyz-shards: cannot read %s\n", argv[2]); return 1; }
    const double sh = atof(iopt(argc, argv, "--shard", "512")) * atof(iopt(argc, argv, "--div", "1")), mg = atof(iopt(argc, argv, "--margin", "32"));
    if (!(sh > 0)) { fprintf(stderr, "tifxyz-shards: bad --shard / --div\n"); return 2; }
    int64_t *k = nullptr; size_t n = 0, cap = 0; int64_t lo[3] = {INT64_MAX, INT64_MAX, INT64_MAX}, hi[3] = {-1, -1, -1};
    const float *A[3] = {S.Z, S.Y, S.X};
    for (size_t i = 0; i < (size_t)S.w * S.h; i++) {
        if (S.Z[i] < 0) continue;
        int64_t a[3], b[3];
        for (int d = 0; d < 3; d++) { a[d] = (int64_t)floor((A[d][i] - mg) / sh); b[d] = (int64_t)floor((A[d][i] + mg) / sh); if (a[d] < 0) a[d] = 0; }
        for (int64_t z = a[0]; z <= b[0]; z++) for (int64_t y = a[1]; y <= b[1]; y++) for (int64_t x = a[2]; x <= b[2]; x++) {
            if (n == cap) { cap = cap ? cap * 2 : 4096; k = realloc(k, cap * sizeof *k); }
            k[n++] = (z << 42) | (y << 21) | x;
        }
    }
    qsort(k, n, sizeof *k, i64cmp); size_t u = 0;
    for (size_t i = 0; i < n; i++) if (!u || k[i] != k[u - 1]) k[u++] = k[i];
    for (size_t i = 0; i < u; i++) {
        const int64_t c[3] = {k[i] >> 42, (k[i] >> 21) & 0x1fffff, k[i] & 0x1fffff};
        for (int d = 0; d < 3; d++) { if (c[d] < lo[d]) lo[d] = c[d]; if (c[d] > hi[d]) hi[d] = c[d]; }
        printf("%lld %lld %lld\n", (long long)c[0], (long long)c[1], (long long)c[2]);
    }
    const double s1 = atof(iopt(argc, argv, "--shard", "512"));
    if (u) fprintf(stderr, "tifxyz-shards: %zu shards, box %.0f,%.0f,%.0f,%.0f,%.0f,%.0f\n", u, lo[0] * s1, lo[1] * s1, lo[2] * s1, (hi[0] - lo[0] + 1) * s1, (hi[1] - lo[1] + 1) * s1, (hi[2] - lo[2] + 1) * s1);
    free(k); isurf_free(&S);
    return u ? 0 : 1;
}

/* ufsm tifxyz-bbox <tifxyz> <image.tif> [--thresh 0] [--margin 0]: the 3D bounding box (native voxels, "z0 z1 y0 y1 x0 x1") of the
   surface points whose render-grid pixel (x, y) = grid (x scale, y scale) has image > thresh, grown by --margin voxels */
int cmd_tifxyz_bbox(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm tifxyz-bbox <tifxyz> <image.tif> [--thresh 0] [--margin 0]\n"); return 2; }
    isurf S; if (isurf_load(&S, argv[2])) { fprintf(stderr, "tifxyz-bbox: cannot read %s\n", argv[2]); return 1; }
    int iw, ih; uint8_t *img = img_read_u8(argv[3], &iw, &ih); if (!img) { fprintf(stderr, "tifxyz-bbox: cannot read %s\n", argv[3]); return 1; }
    const int th = atoi(iopt(argc, argv, "--thresh", "0")); const double mg = atof(iopt(argc, argv, "--margin", "0"));
    double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300}; long n = 0;
    for (int y = 0; y < ih; y++) for (int x = 0; x < iw; x++) {
        if (img[(size_t)y * iw + x] <= th) continue;
        double P[3], N[3], a[3], b[3]; if (surf_pn(&S, x * S.scale, y * S.scale, P, N, a, b)) continue;
        for (int d = 0; d < 3; d++) { if (P[d] < lo[d]) lo[d] = P[d]; if (P[d] > hi[d]) hi[d] = P[d]; }
        n++;
    }
    if (!n) { fprintf(stderr, "tifxyz-bbox: no pixel above %d\n", th); return 1; }
    printf("%.0f %.0f %.0f %.0f %.0f %.0f\n", floor(lo[0] - mg), ceil(hi[0] + mg), floor(lo[1] - mg), ceil(hi[1] + mg), floor(lo[2] - mg), ceil(hi[2] + mg));
    fprintf(stderr, "tifxyz-bbox: %ld pixels above %d\n", n, th);
    free(img); isurf_free(&S);
    return 0;
}

/* ufsm wl-dump <weights> [sub]: the tensors of a safetensors / torch zip file (name, shape, mean |v|) */
int cmd_wl_dump(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: ufsm wl-dump <weights> [state-dict key]\n"); return 2; }
    wl_dict *d = wl_load(argv[2], argc > 3 ? argv[3] : nullptr);
    if (!d) { fprintf(stderr, "wl-dump: %s\n", wl_error()); return 1; }
    for (int i = 0; i < wl_count(d); i++) {
        const wl_tensor *t = wl_at(d, i); double s = 0; for (int64_t k = 0; k < t->numel; k++) s += fabs(t->data[k]);
        printf("%s [", t->name); for (int k = 0; k < t->ndim; k++) printf(k ? ",%lld" : "%lld", (long long)t->shape[k]);
        printf("] %.6g\n", t->numel ? s / t->numel : 0);
    }
    wl_free(d);
    return 0;
}
