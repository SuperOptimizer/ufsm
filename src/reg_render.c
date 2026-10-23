/* ufsm reg render <config.json>: multi-scan renders of one tifxyz segment through registrations (src/reg.h).
   The segment lives in a base scan's level-0 voxels (tifxyz values x downscale). Every other scan is reached through a chain of reg maps
   (each step c2f: the point is in the pair's fixed / coarse frame, or f2c: in its moving / fine frame); a point is used only where every
   step's pair covers it (a good fit box within cover_mm and no failing evaluation box within that distance) is flagged certified in the depth samples (reg
   inkstat uses only those); images show every point where the scan has data under the map (no masked voxel in the trilinear stencil).
   Per output pixel (px base voxels; 2 x 2 sub-samples): native = each scan at the pyramid level nearest the output pixel (level 0
   only when the pixel is at or below the scan's voxel), the band mean over layers -L..L (one base
   voxel apart along the surface normal, mapped) averaged over the 2 x 2 sub-pixels; matched = each scan at its pyramid level nearest
   match_um, the band mean at the pixel centre. Stored as uint8 raw means (0 = missing), normalised by exact 1st / 99th percentiles.
   Outputs (JPEG, half-res): <variant>_<scan>.jpg, <variant>_rgb.jpg (R G B = config rgb), <variant>_diff.jpg (R G B = the three diff
   pairs, 128 + gain (a - b), a and b normalised to [0, 1]), <variant>_baseline.jpg (the baseline pair, same gain), coverage.jpg, stats.json.
   Config: {"tifxyz": DIR, "downscale": 4, "out": DIR, "layers": 2, "match_um": 9.362, "cover_mm": 1.2, "tol_um": 2.4 (fit / eval residual limit, um), "threads": 12,
            "cache_mb": 2400, "px": 2 (base voxels per output pixel: 2 = half-res of the base-voxel uv grid), "crop": [x0, y0, w, h] (output pixels, optional),
            "scans": [{"name", "root", "key", "um", "chain": [{"map": reg.json, "dir": "c2f" | "f2c", "pair": pair dir}]}],
            "rgb": [n1, n2, n3], "diff": [[a, b], [a, b], [a, b]], "baseline": [a, b]} */
#include "reg.h"
#include "json.h"
#include "imgops.h"
#include "zarr3.h"
#include "inksplat.h"
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

int reg_load_map(const char *path, const char *cache, rvol **cv, rvol **fv, rmap *m, rfield *D);   /* src/reg_view.c */

static char *slurp(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb"); if (!f) return nullptr;
    fseek(f, 0, SEEK_END); const long L = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = malloc(L + 1); if (fread(b, 1, L, f) != (size_t)L) { fclose(f); free(b); return nullptr; }
    b[L] = 0; fclose(f); if (n) *n = L; return b;
}

/* ---------- coverage of one pair: good fit boxes and failing evaluation boxes, in the pair's fixed frame ---------- */
static void ptset_add(ptset *s, const double c[3]) { s->c = realloc(s->c, (s->n + 1) * 3 * sizeof *s->c); memcpy(s->c + 3 * s->n, c, 3 * sizeof *c); s->n++; }
static int hkey(const rcover *cv, const double *p) {
    const long a = (long)floor((p[0] - cv->o[0]) / cv->cell), b = (long)floor((p[1] - cv->o[1]) / cv->cell), c = (long)floor((p[2] - cv->o[2]) / cv->cell);
    return (int)(((unsigned long)(a * 73856093L) ^ (unsigned long)(b * 19349663L) ^ (unsigned long)(c * 83492791L)) % (unsigned long)cv->nh);
}
static int near_any(const rcover *cv, const ptset *s, const int *head, const int *next, const double p[3], double r2) {
    for (int dz = -1; dz <= 1; dz++) for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
        const double q[3] = {p[0] + dx * cv->cell, p[1] + dy * cv->cell, p[2] + dz * cv->cell};
        for (int i = head[hkey(cv, q)]; i >= 0; i = next[i]) {
            const double *c = s->c + 3 * i, e = (c[0] - p[0]) * (c[0] - p[0]) + (c[1] - p[1]) * (c[1] - p[1]) + (c[2] - p[2]) * (c[2] - p[2]);
            if (e <= r2) return 1;
        }
    }
    return 0;
}
int rcover_ok(const rcover *cv, const double p[3]) {
    return near_any(cv, &cv->good, cv->head, cv->next, p, cv->r2) && !near_any(cv, &cv->bad, cv->bhead, cv->bnext, p, cv->rb2);
}
/* residuals.jsonl of final/: used boxes with |v - pred| <= tol are good; eval_after.jsonl: not ok, ncc < 0.3 or |s| > tol is bad */
int rcover_load(rcover *cv, const char *pair, double r_vox, double tol) {
    memset(cv, 0, sizeof *cv);
    char p[2100]; char line[4096]; FILE *f;
    snprintf(p, sizeof p, "%s/final/residuals.jsonl", pair);
    if (!(f = fopen(p, "r"))) { fprintf(stderr, "reg render: cannot read %s\n", p); return -1; }
    while (fgets(line, sizeof line, f)) {
        json *j = json_parse(line, strlen(line)); if (!j) continue;
        if (json_num(json_get(j, "used"), 0) > 0) {
            double c[3], e = 0; for (int d = 0; d < 3; d++) { c[d] = json_num(json_at(json_get(j, "c"), d), 0); const double r = json_num(json_at(json_get(j, "v"), d), 0) - json_num(json_at(json_get(j, "pred"), d), 0); e += r * r; }
            if (sqrt(e) <= tol) ptset_add(&cv->good, c);
        }
        json_free(j);
    }
    fclose(f);
    snprintf(p, sizeof p, "%s/eval_after.jsonl", pair);
    if ((f = fopen(p, "r"))) {
        while (fgets(line, sizeof line, f)) {
            json *j = json_parse(line, strlen(line)); if (!j) continue;
            const char *st = json_str(json_get(j, "status"), "");
            double c[3], s = 0; for (int d = 0; d < 3; d++) { c[d] = json_num(json_at(json_get(j, "c"), d), 0); const double v = json_num(json_at(json_get(j, "s"), d), 0); s += v * v; }
            const int papyrus = strcmp(st, "mask") && strcmp(st, "fine") && strcmp(st, "read");
            if (papyrus && (strcmp(st, "ok") || json_num(json_get(j, "ncc"), 0) < 0.3 || sqrt(s) > tol)) ptset_add(&cv->bad, c);
            json_free(j);
        }
        fclose(f);
    }
    /* radius: max(cover, 1.5 x the median nearest-neighbour spacing of the good boxes), so the covered region has no holes between fit
       boxes (the field interpolates there: hold-out CV); failing evaluation boxes blank half the cover radius */
    double med = 0;
    if (cv->good.n > 1) {
        const int ns = cv->good.n < 1500 ? cv->good.n : 1500; double *d = malloc(ns * sizeof *d);
        for (int a = 0; a < ns; a++) {
            const int i = (int)((long)a * cv->good.n / ns); double best = 1e300;
            for (int k = 0; k < cv->good.n; k++) { if (k == i) continue; const double *x = cv->good.c + 3 * i, *y = cv->good.c + 3 * k;
                const double e = (x[0] - y[0]) * (x[0] - y[0]) + (x[1] - y[1]) * (x[1] - y[1]) + (x[2] - y[2]) * (x[2] - y[2]); if (e < best) best = e; }
            d[a] = sqrt(best);
        }
        for (int a = 1; a < ns; a++) for (int b = a; b > 0 && d[b] < d[b - 1]; b--) { const double t = d[b]; d[b] = d[b - 1]; d[b - 1] = t; }
        med = d[ns / 2]; free(d);
    }
    const double r = fmax(r_vox, 1.5 * med), rb = r_vox / 2;
    cv->r2 = r * r; cv->rb2 = rb * rb; cv->cell = r; cv->nh = 1 << 16;
    cv->head = malloc(cv->nh * sizeof(int)); cv->bhead = malloc(cv->nh * sizeof(int));
    for (int i = 0; i < cv->nh; i++) cv->head[i] = cv->bhead[i] = -1;
    cv->next = malloc((cv->good.n + 1) * sizeof(int)); cv->bnext = malloc((cv->bad.n + 1) * sizeof(int));
    for (int i = 0; i < cv->good.n; i++) { const int k = hkey(cv, cv->good.c + 3 * i); cv->next[i] = cv->head[k]; cv->head[k] = i; }
    for (int i = 0; i < cv->bad.n; i++) { const int k = hkey(cv, cv->bad.c + 3 * i); cv->bnext[i] = cv->bhead[k]; cv->bhead[k] = i; }
    fprintf(stderr, "reg render: %s: %d good fit boxes (median spacing %.0f), %d failing eval boxes; cover radius %.0f, blank radius %.0f voxels\n", pair, cv->good.n, med, cv->bad.n, r, rb);
    return 0;
}

/* ---------- decoded-chunk cache of one volume level (pinned LRU) ---------- */
typedef struct centry { int64_t k; uint8_t *d; int refs, ready; uint64_t tick; struct centry *hnext; } centry;
typedef struct {
    z3 *z; int64_t shape[3]; int ch[3]; int64_t g[3];
    pthread_mutex_t mu; pthread_cond_t cv; centry **h; int nh; size_t n, max; uint64_t tick; atomic_long loads;
} ccache;
static void cc_init(ccache *c, z3 *z, size_t max_chunks) {
    memset(c, 0, sizeof *c); c->z = z; const z3_meta *m = z3_meta_of(z);
    for (int a = 0; a < 3; a++) { c->shape[a] = m->shape[a]; c->ch[a] = m->chunk[a]; c->g[a] = (m->shape[a] + m->chunk[a] - 1) / m->chunk[a]; }
    pthread_mutex_init(&c->mu, nullptr); pthread_cond_init(&c->cv, nullptr);
    c->nh = 1 << 16; c->h = calloc(c->nh, sizeof *c->h); c->max = max_chunks < 16 ? 16 : max_chunks;
}
static int tick_cmp(const void *x, const void *y) { const uint64_t a = (*(centry *const *)x)->tick, b = (*(centry *const *)y)->tick; return a < b ? -1 : a > b; }
static void cc_evict(ccache *c) {   /* under the lock: drop the oldest unpinned entries down to 7/8 of max */
    centry **v = malloc(c->n * sizeof *v); size_t k = 0;
    for (int i = 0; i < c->nh; i++) for (centry *e = c->h[i]; e; e = e->hnext) if (e->ready && !e->refs) v[k++] = e;
    qsort(v, k, sizeof *v, tick_cmp);
    for (size_t i = 0; i < k && c->n > c->max - c->max / 8; i++) {
        centry *e = v[i]; const int hb = (int)((uint64_t)(e->k * 0x9e3779b97f4a7c15ull) >> 48) & (c->nh - 1);
        for (centry **pp = &c->h[hb]; *pp; pp = &(*pp)->hnext) if (*pp == e) { *pp = e->hnext; break; }
        free(e->d); free(e); c->n--;
    }
    free(v);
}
static centry *cc_get(ccache *c, int64_t cz, int64_t cy, int64_t cx) {
    if (cz < 0 || cy < 0 || cx < 0 || cz >= c->g[0] || cy >= c->g[1] || cx >= c->g[2]) return nullptr;
    const int64_t k = (cz * c->g[1] + cy) * c->g[2] + cx; const int hb = (int)((uint64_t)(k * 0x9e3779b97f4a7c15ull) >> 48) & (c->nh - 1);
    pthread_mutex_lock(&c->mu);
    centry *e = c->h[hb]; while (e && e->k != k) e = e->hnext;
    if (e) {
        e->refs++; e->tick = ++c->tick;
        while (!e->ready) pthread_cond_wait(&c->cv, &c->mu);
        if (!e->d) { e->refs--; e = nullptr; }
        pthread_mutex_unlock(&c->mu); return e;
    }
    e = calloc(1, sizeof *e); e->k = k; e->refs = 1; e->tick = ++c->tick; e->hnext = c->h[hb]; c->h[hb] = e; c->n++;
    if (c->n > c->max) cc_evict(c);
    pthread_mutex_unlock(&c->mu);
    const int64_t o[3] = {cz * c->ch[0], cy * c->ch[1], cx * c->ch[2]}, n[3] = {c->ch[0], c->ch[1], c->ch[2]};
    uint8_t *d = malloc((size_t)n[0] * n[1] * n[2]);
    int ok = 0;
    for (int a = 0; a < 4 && d && !ok; a++) { ok = !z3_read(c->z, o, n, d, 1); if (!ok) { fprintf(stderr, "reg render: chunk read failed (%s), retry\n", z3_error()); struct timespec ts = {2 + 8 * a, 0}; nanosleep(&ts, nullptr); } }
    atomic_fetch_add(&c->loads, 1);
    pthread_mutex_lock(&c->mu);
    if (!ok) { free(d); d = nullptr; }
    e->d = d; e->ready = 1; pthread_cond_broadcast(&c->cv);
    if (!d) e->refs--;
    pthread_mutex_unlock(&c->mu);
    return d ? e : nullptr;
}
static void cc_put(ccache *c, centry *e) { if (!e) return; pthread_mutex_lock(&c->mu); e->refs--; pthread_mutex_unlock(&c->mu); }

/* per-thread memo: the last chunk pinned for each cache */
typedef struct { ccache *c; centry *e; int64_t cz, cy, cx; } cmemo;
static inline int vox(cmemo *m, int64_t z, int64_t y, int64_t x, int *v) {
    ccache *c = m->c;
    if (z < 0 || y < 0 || x < 0 || z >= c->shape[0] || y >= c->shape[1] || x >= c->shape[2]) return 0;
    const int64_t cz = z / c->ch[0], cy = y / c->ch[1], cx = x / c->ch[2];
    if (!m->e || cz != m->cz || cy != m->cy || cx != m->cx) {
        cc_put(c, m->e); m->e = cc_get(c, cz, cy, cx); m->cz = cz; m->cy = cy; m->cx = cx;
        if (!m->e) return 0;
    }
    *v = m->e->d[((size_t)(z - cz * c->ch[0]) * c->ch[1] + (y - cy * c->ch[1])) * c->ch[2] + (x - cx * c->ch[2])];
    return 1;
}
/* trilinear at level coords (x y z); -1 if any stencil voxel is missing / masked (0) */
static float tri(cmemo *m, double x, double y, double z) {
    const double fx = floor(x), fy = floor(y), fz = floor(z); const int64_t ix = (int64_t)fx, iy = (int64_t)fy, iz = (int64_t)fz;
    const double ux = x - fx, uy = y - fy, uz = z - fz; double s = 0;
    for (int k = 0; k < 8; k++) {
        int v; if (!vox(m, iz + (k >> 2), iy + (k >> 1 & 1), ix + (k & 1), &v) || !v) return -1;
        s += ((k >> 2) ? uz : 1 - uz) * ((k >> 1 & 1) ? uy : 1 - uy) * ((k & 1) ? ux : 1 - ux) * v;
    }
    return (float)s;
}

/* ---------- config ---------- */
enum { MAXS = 8, MAXCH = 4 };
typedef struct { rmap m; rfield D; int f2c; rcover cov; } rstep;
typedef struct { char name[64]; rvol *v; int nch; rstep st[MAXCH]; int ln, lm, lsn; ccache cn, cm, cs; } rscan;
/* base L0 -> scan L0 through the chain. Returns 1, or 2 when every step's pair also certifies the point (cover: a good fit box near,
   no failing evaluation box near). Images use every mapped point with data; only reg inkstat restricts to certified points. */
static int map_point(const rscan *s, const double p[3], double q[3]) {
    double a[3] = {p[0], p[1], p[2]}; int cert = 1;
    for (int k = 0; k < s->nch; k++) {
        const rstep *t = &s->st[k]; double b[3];
        if (t->f2c) { rmap_f2c(&t->m, a, b); cert &= rcover_ok(&t->cov, b); }
        else { cert &= rcover_ok(&t->cov, a); rmap_c2f(&t->m, a, b); }
        memcpy(a, b, sizeof a);
    }
    memcpy(q, a, sizeof a);
    return 1 + cert;
}

typedef struct {
    int ns; rscan sc[MAXS]; int L, threads; double match_um; int cop, cl0, cl1;   /* composite: 0 mean 1 max 2 top3 3 p90 over layers cl0..cl1 */
    float *X, *Y, *Z; int gw, gh; double f, px;   /* tifxyz grid (base L0 voxels), base voxels per grid step, base voxels per output px */
    int W, H, x0, y0;                          /* half-res output size and crop origin (half-res px) */
    uint8_t *raw[2][MAXS];                     /* [native, matched][scan] raw band means, 0 = missing */
    atomic_int next; int ntx, nty, nreal, T; time_t t0;
} rjob;

/* surface point and unit normal at grid coords (gu, gv); 0 if any of the 4 nodes (or their neighbours for the normal) is invalid */
static int surf_at(const rjob *j, double gu, double gv, double P[3], double N[3]) {
    const int i = (int)floor(gu), k = (int)floor(gv);
    if (i < 1 || k < 1 || i + 2 >= j->gw || k + 2 >= j->gh) return 0;
    const double a = gu - i, b = gv - k; double n[3] = {0, 0, 0};
    P[0] = P[1] = P[2] = 0;
    for (int q = 0; q < 4; q++) {
        const int ii = i + (q & 1), kk = k + (q >> 1); const size_t o = (size_t)kk * j->gw + ii;
        const double w = ((q & 1) ? a : 1 - a) * ((q >> 1) ? b : 1 - b);
        if (j->Z[o] < 0 || j->X[o] < 0) return 0;
        const size_t l = o - 1, r = o + 1, u = o - j->gw, d = o + j->gw;
        if (j->Z[l] < 0 || j->Z[r] < 0 || j->Z[u] < 0 || j->Z[d] < 0 || j->X[l] < 0 || j->X[r] < 0 || j->X[u] < 0 || j->X[d] < 0) return 0;
        const double du[3] = {j->X[r] - j->X[l], j->Y[r] - j->Y[l], j->Z[r] - j->Z[l]}, dv[3] = {j->X[d] - j->X[u], j->Y[d] - j->Y[u], j->Z[d] - j->Z[u]};
        const double c[3] = {du[1] * dv[2] - du[2] * dv[1], du[2] * dv[0] - du[0] * dv[2], du[0] * dv[1] - du[1] * dv[0]};
        const double cn = sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]); if (cn <= 0) return 0;
        P[0] += w * j->X[o]; P[1] += w * j->Y[o]; P[2] += w * j->Z[o];
        for (int d2 = 0; d2 < 3; d2++) n[d2] += w * c[d2] / cn;
    }
    const double nn = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]); if (nn <= 0) return 0;
    for (int d2 = 0; d2 < 3; d2++) N[d2] = n[d2] / nn;
    return 1;
}
/* band mean of one scan at base point P along base normal N (layers -L..L, one base voxel apart), at level lv; -1 if missing */
static float band(const rjob *j, const rscan *s, cmemo *mm, int lv, const double P[3], const double N[3]) {
    double q0[3], q1[3], P1[3] = {P[0] + N[0], P[1] + N[1], P[2] + N[2]};
    if (!map_point(s, P, q0) || !map_point(s, P1, q1)) return -1;
    const double f = (double)(1 << lv); float x[128]; int n = 0;
    for (int t = j->cl0; t <= j->cl1 && n < 128; t++) {
        double q[3]; for (int d = 0; d < 3; d++) q[d] = (q0[d] + t * (q1[d] - q0[d]) + 0.5) / f - 0.5;
        const float v = tri(mm, q[0], q[1], q[2]); if (v < 0) return -1;
        x[n++] = v;
    }
    if (!n) return -1;
    if (j->cop == 1) { float m = x[0]; for (int i = 1; i < n; i++) if (x[i] > m) m = x[i]; return m; }
    if (j->cop >= 2) {   /* top-3 mean / p90: insertion sort */
        for (int a = 1; a < n; a++) for (int b = a; b > 0 && x[b] < x[b - 1]; b--) { const float t2 = x[b]; x[b] = x[b - 1]; x[b - 1] = t2; }
        if (j->cop == 3) return x[(int)floor(0.9 * (n - 1))];
        const int k = n < 3 ? n : 3; double a2 = 0; for (int i = n - k; i < n; i++) a2 += x[i]; return (float)(a2 / k);
    }
    double acc = 0; for (int i = 0; i < n; i++) acc += x[i];
    return (float)(acc / n);
}
static void *render_worker(void *arg) {
    rjob *j = arg; cmemo mn[MAXS], mt[MAXS];
    for (int s = 0; s < j->ns; s++) { mn[s] = (cmemo){&j->sc[s].cn, nullptr, 0, 0, 0}; mt[s] = (cmemo){&j->sc[s].cm, nullptr, 0, 0, 0}; }
    for (;;) {
        const int t = atomic_fetch_add(&j->next, 1); if (t >= j->ntx * j->nty) break;
        /* column bands of 8 tiles, rows within a band: neighbouring tiles share chunks */
        const int band8 = t / (j->nty * 8), rem = t % (j->nty * 8), ty = rem / 8, tx = band8 * 8 + rem % 8;
        if (tx >= j->nreal) continue;
        for (int y = ty * j->T; y < (ty + 1) * j->T && y < j->H; y++) for (int x = tx * j->T; x < (tx + 1) * j->T && x < j->W; x++) {
            const size_t o = (size_t)y * j->W + x;
            const double hx = x + j->x0, hy = y + j->y0;   /* half-res px -> base-voxel uv px (2 hx + 0.5 is the centre) */
            for (int s = 0; s < j->ns; s++) {
                const rscan *sc = &j->sc[s];
                double acc = 0; int n = 0;
                for (int q = 0; q < 4; q++) {   /* native: 2 x 2 sub-pixels (px 2: the base-voxel uv grid's pixels) */
                    double P[3], N[3];
                    if (!surf_at(j, (j->px * hx + (q & 1) * j->px / 2) / j->f, (j->px * hy + (q >> 1) * j->px / 2) / j->f, P, N)) { n = -1; break; }
                    const float v = band(j, sc, &mn[s], sc->ln, P, N); if (v < 0) { n = -1; break; }
                    acc += v; n++;
                }
                j->raw[0][s][o] = n == 4 ? (uint8_t)fmin(255, fmax(1, acc / 4 + 0.5)) : 0;
                double P[3], N[3];
                float v = -1;
                if (surf_at(j, (j->px * hx + j->px / 4) / j->f, (j->px * hy + j->px / 4) / j->f, P, N)) v = band(j, sc, &mt[s], sc->lm, P, N);
                j->raw[1][s][o] = v >= 0 ? (uint8_t)fmin(255, fmax(1, v + 0.5)) : 0;
            }
        }
        const int done = t + 1;
        if (done % 2000 == 0) fprintf(stderr, "  tiles %d / %d (%lds)\n", done, j->ntx * j->nty, (long)(time(nullptr) - j->t0));
    }
    for (int s = 0; s < j->ns; s++) { cc_put(mn[s].c, mn[s].e); cc_put(mt[s].c, mt[s].e); }
    return nullptr;
}

/* ink overlays of every product: <name>_ink.jpg = bright yellow contours (3 px) around the labelled ink regions; <name>_inkside.jpg =
   the render | the render with the ink pixels tinted yellow at 35% (side by side, 8 white columns between) */
static const uint8_t *g_contour, *g_cls; static int g_W, g_H;
static void write_jpg(const char *path, int W, int H, uint8_t *rgb, int q) {
    jpeg_write_rgb(path, W, H, rgb, q);
    if (!g_contour) return;
    char p2[2400]; snprintf(p2, sizeof p2, "%.*s_ink.jpg", (int)(strlen(path) - 4), path);
    uint8_t *c = malloc((size_t)W * H * 3); memcpy(c, rgb, (size_t)W * H * 3);
    for (size_t i = 0; i < (size_t)W * H; i++) if (g_contour[i]) { c[3 * i] = 255; c[3 * i + 1] = 255; c[3 * i + 2] = 0; }
    jpeg_write_rgb(p2, W, H, c, q); free(c);
    const int W2 = 2 * W + 8; uint8_t *sb = malloc((size_t)W2 * H * 3);
    for (int y = 0; y < H; y++) {
        uint8_t *r = sb + (size_t)y * W2 * 3; const uint8_t *src = rgb + (size_t)y * W * 3;
        memcpy(r, src, (size_t)W * 3); memset(r + (size_t)W * 3, 255, 24);
        for (int x = 0; x < W; x++) { uint8_t *o = r + (size_t)(W + 8 + x) * 3; const uint8_t *i3 = src + (size_t)x * 3;
            if (g_cls[(size_t)y * W + x] == 2) { o[0] = (uint8_t)(0.65 * i3[0] + 0.35 * 255); o[1] = (uint8_t)(0.65 * i3[1] + 0.35 * 255); o[2] = (uint8_t)(0.65 * i3[2]); }
            else { o[0] = i3[0]; o[1] = i3[1]; o[2] = i3[2]; } }
    }
    snprintf(p2, sizeof p2, "%.*s_inkside.jpg", (int)(strlen(path) - 4), path);
    jpeg_write_rgb(p2, W2, H, sb, q); free(sb);
}
/* exact percentiles of a uint8 buffer's non-zero values */
static void pct(const uint8_t *v, size_t n, double lo, double hi, double *a, double *b, size_t *cnt) {
    size_t h[256] = {0}; for (size_t i = 0; i < n; i++) h[v[i]]++;
    size_t tot = n - h[0]; *cnt = tot; *a = 0; *b = 255; if (!tot) return;
    size_t c = 0; int i = 1; for (; i < 256 && c + h[i] <= lo * tot; i++) c += h[i];
    *a = i; c = 0; for (i = 1; i < 256 && c + h[i] < hi * tot; i++) c += h[i];
    *b = i; if (*b <= *a) *b = *a + 1;
}
static int find_scan(const rjob *j, const char *n) { for (int s = 0; s < j->ns; s++) if (!strcmp(j->sc[s].name, n)) return s; return -1; }

/* depth samples of labelled pixels: per scan point samples (native: level 0, matched: the matched level) at layers d0 .. d0 + nl - 1 */
typedef struct { rjob *j; const int64_t *idx; size_t n; uint8_t *out; size_t rec; int d0, nl; const uint8_t *cls; atomic_long next; } smp_job;
static void *sample_worker(void *arg) {
    smp_job *q = arg; rjob *j = q->j; cmemo mn[MAXS], mt[MAXS];
    for (int s = 0; s < j->ns; s++) { mn[s] = (cmemo){&j->sc[s].cs, nullptr, 0, 0, 0}; mt[s] = (cmemo){&j->sc[s].cm, nullptr, 0, 0, 0}; }
    for (;;) {
        const long k = atomic_fetch_add(&q->next, 1); if ((size_t)k >= q->n) break;
        const int64_t i = q->idx[k]; const int x = (int)(i % j->W), y = (int)(i / j->W);
        uint8_t *o = q->out + (size_t)k * q->rec; o[0] = q->cls[i];   /* class, then a certified flag per scan, then the values */
        double P[3], N[3];
        if (!surf_at(j, (j->px * (x + j->x0) + j->px / 4) / j->f, (j->px * (y + j->y0) + j->px / 4) / j->f, P, N)) continue;
        for (int s = 0; s < j->ns; s++) {
            const rscan *sc = &j->sc[s]; double q0[3], q1[3], P1[3] = {P[0] + N[0], P[1] + N[1], P[2] + N[2]};
            o[1 + s] = map_point(sc, P, q0) == 2; map_point(sc, P1, q1);
            for (int v = 0; v < 2; v++) {
                const int lv = v ? sc->lm : sc->lsn; const double f = (double)(1 << lv);
                for (int t = 0; t < q->nl; t++) {
                    const double tt = q->d0 + t; double r[3];
                    for (int d = 0; d < 3; d++) r[d] = (q0[d] + tt * (q1[d] - q0[d]) + 0.5) / f - 0.5;
                    const float val = tri(v ? &mt[s] : &mn[s], r[0], r[1], r[2]);
                    o[1 + j->ns + ((size_t)v * j->ns + s) * q->nl + t] = val < 0 ? 0 : (uint8_t)fmin(255, fmax(1, val + 0.5));
                }
            }
        }
    }
    for (int s = 0; s < j->ns; s++) { cc_put(mn[s].c, mn[s].e); cc_put(mt[s].c, mt[s].e); }
    return nullptr;
}

int cmd_reg_render(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm reg render <config.json>   (see src/reg_render.c)\n"); return 2; }
    size_t n; char *txt = slurp(argv[3], &n); if (!txt) { fprintf(stderr, "reg render: cannot read %s\n", argv[3]); return 1; }
    json *cfg = json_parse(txt, n); if (!cfg) { fprintf(stderr, "reg render: bad JSON\n"); free(txt); return 1; }
    {   /* the config beside the outputs (reg inkstat reads the channel names from it) */
        const char *o0 = json_str(json_get(cfg, "out"), "."); char cp[2100]; mkdir(o0, 0755); snprintf(cp, sizeof cp, "%s/config.json", o0);
        char *a = realpath(argv[3], nullptr), *b = realpath(cp, nullptr);
        if (!a || !b || strcmp(a, b)) { FILE *cf = fopen(cp, "w"); if (cf) { fwrite(txt, 1, n, cf); fclose(cf); } }
        free(a); free(b);
    }
    free(txt);
    rjob *j = calloc(1, sizeof *j);
    const char *td = json_str(json_get(cfg, "tifxyz"), ""), *od = json_str(json_get(cfg, "out"), "."), *cache = json_str(json_get(cfg, "cache"), nullptr);
    const double ds = json_num(json_get(cfg, "downscale"), 1), cover_mm = json_num(json_get(cfg, "cover_mm"), 1.2), tol_um = json_num(json_get(cfg, "tol_um"), 2.4);
    j->L = (int)json_num(json_get(cfg, "layers"), 2); j->cl0 = -j->L; j->cl1 = j->L;
    {   /* "composite": {"op": "mean" | "max" | "top3" | "p90", "layers": [a, b]} replaces the band mean over -L .. L */
        const json *cm = json_get(cfg, "composite");
        if (cm) { const char *op = json_str(json_get(cm, "op"), "mean"); j->cop = !strcmp(op, "max") ? 1 : !strcmp(op, "top3") ? 2 : !strcmp(op, "p90") ? 3 : 0;
                  const json *ly = json_get(cm, "layers"); if (ly && ly->n == 2) { j->cl0 = (int)json_num(json_at(ly, 0), -2); j->cl1 = (int)json_num(json_at(ly, 1), 2); } }
    } j->threads = (int)json_num(json_get(cfg, "threads"), 12); j->match_um = json_num(json_get(cfg, "match_um"), 9.362);
    const double cache_mb = json_num(json_get(cfg, "cache_mb"), 2400);
    mkdir(od, 0755);
    /* tifxyz */
    { char p[2100]; int w, h, w2, h2, w3, h3;
      snprintf(p, sizeof p, "%s/x.tif", td); j->X = img_read_f32(p, &w, &h);
      snprintf(p, sizeof p, "%s/y.tif", td); j->Y = img_read_f32(p, &w2, &h2);
      snprintf(p, sizeof p, "%s/z.tif", td); j->Z = img_read_f32(p, &w3, &h3);
      if (!j->X || !j->Y || !j->Z || w != w2 || w != w3 || h != h2 || h != h3) { fprintf(stderr, "reg render: cannot read the tifxyz %s\n", td); return 1; }
      j->gw = w; j->gh = h;
      for (size_t i = 0; i < (size_t)w * h; i++) if (j->Z[i] >= 0 && j->X[i] >= 0) { j->X[i] *= ds; j->Y[i] *= ds; j->Z[i] *= ds; } else { j->X[i] = j->Y[i] = j->Z[i] = -1; }
      snprintf(p, sizeof p, "%s/meta.json", td); size_t mn; char *mt = slurp(p, &mn); json *mj = mt ? json_parse(mt, mn) : nullptr; free(mt);
      const double sc = mj ? json_num(json_at(json_get(mj, "scale"), 0), 0.05) : 0.05; json_free(mj);
      j->f = ds / sc;   /* base voxels per grid step */
    }
    j->px = json_num(json_get(cfg, "px"), 2);
    const int Wf = (int)((j->gw - 1) * j->f / j->px), Hf = (int)((j->gh - 1) * j->f / j->px);
    j->x0 = 0; j->y0 = 0; j->W = Wf; j->H = Hf;
    const json *cr = json_get(cfg, "crop");
    if (cr && cr->n == 4) { j->x0 = (int)json_num(json_at(cr, 0), 0); j->y0 = (int)json_num(json_at(cr, 1), 0); j->W = (int)json_num(json_at(cr, 2), Wf); j->H = (int)json_num(json_at(cr, 3), Hf);
        if (j->x0 + j->W > Wf) j->W = Wf - j->x0;
        if (j->y0 + j->H > Hf) j->H = Hf - j->y0; }
    fprintf(stderr, "reg render: tifxyz %d x %d, %.2f base voxels per grid step, half-res output %d x %d (of %d x %d)\n", j->gw, j->gh, j->f, j->W, j->H, Wf, Hf);
    /* scans */
    const json *ss = json_get(cfg, "scans");
    double base_um = 0;
    for (size_t s = 0; s < (ss ? ss->n : 0) && j->ns < MAXS; s++) {
        const json *e = json_at(ss, s); rscan *sc = &j->sc[j->ns];
        snprintf(sc->name, sizeof sc->name, "%s", json_str(json_get(e, "name"), "scan"));
        sc->v = rvol_open(json_str(json_get(e, "root"), ""), json_str(json_get(e, "key"), ""), cache, json_num(json_get(e, "um"), 0));
        if (!sc->v) return 1;
        const json *ch = json_get(e, "chain");
        for (size_t k = 0; ch && k < ch->n && sc->nch < MAXCH; k++) {
            const json *st = json_at(ch, k); rstep *t = &sc->st[sc->nch];
            if (reg_load_map(json_str(json_get(st, "map"), ""), nullptr, nullptr, nullptr, &t->m, &t->D)) return 1;
            if (t->D.d) t->m.D = &t->D;
            t->f2c = !strcmp(json_str(json_get(st, "dir"), "c2f"), "f2c");
            /* the pair's fixed voxel size from the map JSON */
            size_t mn; char *mt = slurp(json_str(json_get(st, "map"), ""), &mn); json *mj = json_parse(mt, mn); free(mt);
            const double fum = json_num(json_get(json_get(mj, "coarse"), "um"), 1); json_free(mj);
            if (rcover_load(&t->cov, json_str(json_get(st, "pair"), ""), cover_mm * 1000 / fum, tol_um / fum)) return 1;
            sc->nch++;
        }
        if (!sc->nch) base_um = sc->v->um;
        sc->ln = 0; sc->lm = rvol_level_for(sc->v, j->match_um);
        j->ns++;
    }
    if (!j->ns || base_um <= 0) { fprintf(stderr, "reg render: need scans and one base scan (empty chain)\n"); return 1; }
    /* native: the pyramid level nearest the output pixel (px base voxels); level 0 only when the pixel is at or below the scan's voxel */
    for (int s = 0; s < j->ns; s++) j->sc[s].ln = rvol_level_for(j->sc[s].v, j->px * base_um);
    const double sn_um = json_num(json_get(cfg, "sample_native_um"), 2.4);   /* depth samples, native variant: the level <= this */
    for (int s = 0; s < j->ns; s++) j->sc[s].lsn = rvol_level_for(j->sc[s].v, sn_um);
    /* chunk caches: the budget split by voxel count per chunk (2 MB each at 128^3) */
    const size_t per = (size_t)(cache_mb / (2.0 * j->ns * 2)) + 16;
    for (int s = 0; s < j->ns; s++) {
        rscan *sc = &j->sc[s];
        cc_init(&sc->cn, sc->v->z[sc->ln], per * 3 / 2); cc_init(&sc->cm, sc->v->z[sc->lm], per / 2); cc_init(&sc->cs, sc->v->z[sc->lsn], per);
        fprintf(stderr, "reg render: scan %s %.3f um: native level %d (%.3f um), matched level %d (%.3f um), chain %d\n", sc->name, sc->v->um, sc->ln, sc->v->um * (1 << sc->ln), sc->lm, sc->v->um * (1 << sc->lm), sc->nch);
    }
    const size_t NP = (size_t)j->W * j->H;
    for (int v = 0; v < 2; v++) for (int s = 0; s < j->ns; s++) if (!(j->raw[v][s] = calloc(NP, 1))) { fprintf(stderr, "reg render: out of memory\n"); return 1; }
    j->T = 64; j->nreal = (j->W + j->T - 1) / j->T; j->nty = (j->H + j->T - 1) / j->T;
    j->ntx = (j->nreal + 7) / 8 * 8;   /* bands of 8 tile columns; tiles past the real width are skipped */
    j->t0 = time(nullptr); atomic_store(&j->next, 0);
    pthread_t th[64]; const int T = j->threads < 1 ? 1 : j->threads > 64 ? 64 : j->threads;
    for (int t = 0; t < T; t++) pthread_create(&th[t], nullptr, render_worker, j);
    for (int t = 0; t < T; t++) pthread_join(th[t], nullptr);
    fprintf(stderr, "reg render: sampled in %lds\n", (long)(time(nullptr) - j->t0));
    for (int s = 0; s < j->ns; s++) fprintf(stderr, "  %s: %ld native chunk loads, %ld matched\n", j->sc[s].name, atomic_load(&j->sc[s].cn.loads), atomic_load(&j->sc[s].cm.loads));
    /* human ink labels (2-D canvas level whose pixel is this render's pixel): class 2 ink, 1 labelled no-ink, 0 unlabelled */
    uint8_t *cls = nullptr; const json *lb = json_get(cfg, "labels");
    if (lb) {
        /* label pixel (r, c) covers base-voxel uv px [F c, F (c + 1)) (F = "scale": canvas = base-voxel px per label px, default px):
           render px (x, y) takes the label pixel under its centre */
        int64_t LH = 0, LW = 0; const char *ip = json_str(json_get(lb, "ink"), ""), *sp = json_str(json_get(lb, "sup"), "");
        const double F = json_num(json_get(lb, "scale"), j->px);
        uint8_t *t0 = inksplat_label_window(ip, 0, 0, 1, 1, &LH, &LW); free(t0);
        uint8_t *li = LH > 0 ? inksplat_label_window(ip, 0, 0, LH, LW, nullptr, nullptr) : nullptr, *ls = LH > 0 ? inksplat_label_window(sp, 0, 0, LH, LW, nullptr, nullptr) : nullptr;
        if (!li || !ls) { fprintf(stderr, "reg render: cannot read the labels %s / %s\n", ip, sp); return 1; }
        uint8_t *ink = calloc(NP, 1), *sup = calloc(NP, 1);
        for (int y = 0; y < j->H; y++) for (int x = 0; x < j->W; x++) {
            const int64_t r = (int64_t)floor((j->px * (y + j->y0) + j->px / 2) / F), c = (int64_t)floor((j->px * (x + j->x0) + j->px / 2) / F);
            if (r < 0 || c < 0 || r >= LH || c >= LW) continue;
            ink[(size_t)y * j->W + x] = li[(size_t)r * LW + c]; sup[(size_t)y * j->W + x] = ls[(size_t)r * LW + c];
        }
        free(li); free(ls);
        uint8_t mi = 0, ms = 0; for (size_t i = 0; i < NP; i++) { if (ink[i] > mi) mi = ink[i]; if (sup[i] > ms) ms = sup[i]; }
        const int ti = mi <= 1 ? 1 : 128, ts = ms <= 1 ? 1 : 128;
        cls = calloc(NP, 1); size_t nc[3] = {0};
        for (size_t i = 0; i < NP; i++) { cls[i] = sup[i] >= ts ? (ink[i] >= ti ? 2 : 1) : 0; nc[cls[i]]++; }
        fprintf(stderr, "reg render: labels %lld x %lld (render %d x %d at %d, %d): %zu ink, %zu no-ink pixels\n", (long long)LH, (long long)LW, j->W, j->H, j->x0, j->y0, nc[2], nc[1]);
        uint8_t *ct = calloc(NP, 1);
        for (int y = 0; y < j->H; y++) for (int x = 0; x < j->W; x++) {
            const size_t i = (size_t)y * j->W + x; if (cls[i] != 2) continue;
            int edge = 0; for (int d = 0; d < 4 && !edge; d++) { const int yy = y + (d == 0) - (d == 1), xx = x + (d == 2) - (d == 3); if (yy < 0 || xx < 0 || yy >= j->H || xx >= j->W || cls[(size_t)yy * j->W + xx] != 2) edge = 1; }
            if (edge) for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) { const int yy = y + dy, xx = x + dx; if (yy >= 0 && xx >= 0 && yy < j->H && xx < j->W) ct[(size_t)yy * j->W + xx] = 1; }
        }
        g_contour = ct; g_cls = cls; g_W = j->W; g_H = j->H;
        {   /* the label mask on the render grid: ink white, labelled no-ink grey, unlabelled black */
            uint8_t *m3 = malloc(NP * 3); for (size_t i = 0; i < NP; i++) m3[3 * i] = m3[3 * i + 1] = m3[3 * i + 2] = cls[i] == 2 ? 255 : cls[i] == 1 ? 110 : 0;
            char mp[2100]; snprintf(mp, sizeof mp, "%s/labels.jpg", od); jpeg_write_rgb(mp, j->W, j->H, m3, 92); free(m3);
        }
        free(ink); free(sup);
        /* depth samples: up to max_samples per class (hash-thinned), per scan, variant and layer d0 .. d1 */
        const int d0 = (int)json_num(json_at(json_get(cfg, "depth"), 0), -10), d1 = (int)json_num(json_at(json_get(cfg, "depth"), 1), 40), nl = d1 - d0 + 1;
        const double cap = json_num(json_get(cfg, "max_samples"), 150000);
        size_t nsm = 0; int64_t *sx = nullptr;
        { const double pk[3] = {0, fmin(1, cap / fmax(1, nc[1])), fmin(1, cap / fmax(1, nc[2]))};
          sx = malloc((size_t)(2 * cap + 16) * sizeof *sx);
          for (size_t i = 0; i < NP && nsm < 2 * cap + 8; i++) { if (!cls[i]) continue; uint64_t h = i * 0x9e3779b97f4a7c15ull; h ^= h >> 29; if ((double)(h % 1000003) / 1000003.0 < pk[cls[i]]) sx[nsm++] = (int64_t)i; } }
        const size_t rec = 1 + j->ns + (size_t)2 * j->ns * nl;
        uint8_t *sv = calloc(nsm * rec, 1);
        smp_job sj = {.j = j, .idx = sx, .n = nsm, .out = sv, .rec = rec, .d0 = d0, .nl = nl, .cls = cls}; atomic_store(&sj.next, 0);
        pthread_t th2[64]; const int T2 = j->threads < 1 ? 1 : j->threads > 64 ? 64 : j->threads;
        for (int t = 0; t < T2; t++) pthread_create(&th2[t], nullptr, sample_worker, &sj);
        for (int t = 0; t < T2; t++) pthread_join(th2[t], nullptr);
        char sp2[2100]; snprintf(sp2, sizeof sp2, "%s/samples.bin", od); FILE *sf = fopen(sp2, "wb");
        char hdr[4096]; int hl = snprintf(hdr, sizeof hdr, "{\"samples\":%zu,\"scans\":[", nsm);
        for (int s2 = 0; s2 < j->ns; s2++) hl += snprintf(hdr + hl, sizeof hdr - hl, "%s\"%s\"", s2 ? "," : "", j->sc[s2].name);
        hl += snprintf(hdr + hl, sizeof hdr - hl, "],\"variants\":[\"native\",\"matched\"],\"depth\":[%d,%d],\"record\":\"class u8 (1 no-ink, 2 ink), certified u8 per scan, then [variant][scan][layer] u8 raw (0 missing)\"}\n", d0, d1);
        fwrite(hdr, 1, hl, sf); fwrite(sv, rec, nsm, sf); fclose(sf);
        fprintf(stderr, "reg render: %zu depth samples (%d layers) -> %s\n", nsm, nl, sp2);
        free(sv); free(sx);
    }
    /* normalisation, products */
    char p[2300]; FILE *st; snprintf(p, sizeof p, "%s/stats.json", od); st = fopen(p, "w");
    fprintf(st, "{\"tifxyz\":\"%s\",\"half_res\":[%d,%d],\"crop\":[%d,%d],\"layers\":%d,\"composite\":[%d,%d,%d],\"base_voxel_um\":%.4g,\"pixel_um\":%.4g,\"cover_mm\":%.3g,\"tol_um\":%.3g,\"variants\":{", td, j->W, j->H, j->x0, j->y0, j->L, j->cop, j->cl0, j->cl1, base_um, j->px * base_um, cover_mm, tol_um);
    const char *vn[2] = {"native", "matched"};
    const json *rgb = json_get(cfg, "rgb"), *dif = json_get(cfg, "diff"), *bl = json_get(cfg, "baseline");
    uint8_t *img = malloc(NP * 3);
    for (int v = 0; v < 2; v++) {
        float lut[MAXS][256]; fprintf(st, "%s\"%s\":{\"scans\":{", v ? "," : "", vn[v]);
        for (int s = 0; s < j->ns; s++) {
            double a, b; size_t cnt; pct(j->raw[v][s], NP, 0.01, 0.99, &a, &b, &cnt);
            for (int i = 0; i < 256; i++) lut[s][i] = i ? (float)fmin(1, fmax(0, (i - a) / (b - a))) : -1.f;
            fprintf(st, "%s\"%s\":{\"coverage\":%.4f,\"p1\":%g,\"p99\":%g}", s ? "," : "", j->sc[s].name, (double)cnt / NP, a, b);
            for (size_t i = 0; i < NP; i++) { const float x = lut[s][j->raw[v][s][i]]; img[3 * i] = img[3 * i + 1] = img[3 * i + 2] = x < 0 ? 0 : (uint8_t)(x * 254 + 1); }
            snprintf(p, sizeof p, "%s/%s_%s.jpg", od, vn[v], j->sc[s].name); write_jpg(p, j->W, j->H, img, 90);
        }
        fprintf(st, "}");
        /* differences: the common symmetric gain = 127 / max over the diff pairs and the baseline of the 99th percentile |a - b| */
        int dp[4][2]; int nd = 0;
        for (size_t k = 0; dif && k < dif->n && k < 3; k++) { dp[nd][0] = find_scan(j, json_str(json_at(json_at(dif, k), 0), "")); dp[nd][1] = find_scan(j, json_str(json_at(json_at(dif, k), 1), "")); nd++; }
        const int nb3 = nd; if (bl && bl->n == 2) { dp[nd][0] = find_scan(j, json_str(json_at(bl, 0), "")); dp[nd][1] = find_scan(j, json_str(json_at(bl, 1), "")); nd++; }
        double p99[4] = {0}, p50[4] = {0}, mean[4] = {0}; size_t dn[4] = {0};
        for (int k = 0; k < nd; k++) {
            if (dp[k][0] < 0 || dp[k][1] < 0) continue;
            size_t h[512] = {0}; double sm = 0;
            for (size_t i = 0; i < NP; i++) { const float x = lut[dp[k][0]][j->raw[v][dp[k][0]][i]], y = lut[dp[k][1]][j->raw[v][dp[k][1]][i]]; if (x < 0 || y < 0) continue; const int d = (int)lround((x - y) * 255); h[d + 255]++; dn[k]++; sm += x - y; }
            mean[k] = dn[k] ? sm / dn[k] : 0;
            size_t c = 0; int q = 0; for (int i = 0; i <= 255; i++) { const size_t m2 = h[255 + i] + (i ? h[255 - i] : 0); c += m2; if (c >= 0.5 * dn[k] && !p50[k]) p50[k] = (i + 0.5) / 255; if (c >= 0.99 * dn[k]) { q = i; break; } }
            p99[k] = (q + 0.5) / 255;
        }
        double gmax = 1e-3; for (int k = 0; k < nd; k++) gmax = fmax(gmax, p99[k]);
        const double gain = 127 / gmax;
        fprintf(st, ",\"diff_gain\":%.4f,\"diffs\":[", gain);
        for (int k = 0; k < nd; k++) fprintf(st, "%s{\"a\":\"%s\",\"b\":\"%s\",\"baseline\":%d,\"pixels\":%zu,\"mean\":%.4f,\"p50_abs\":%.4f,\"p99_abs\":%.4f}", k ? "," : "", dp[k][0] >= 0 ? j->sc[dp[k][0]].name : "?", dp[k][1] >= 0 ? j->sc[dp[k][1]].name : "?", k >= nb3, dn[k], mean[k], p50[k], p99[k]);
        fprintf(st, "]}");
        /* RGB of the energies */
        int rg[3] = {-1, -1, -1}; for (int c = 0; c < 3 && rgb; c++) rg[c] = find_scan(j, json_str(json_at(rgb, c), ""));
        if (rg[0] >= 0 && rg[1] >= 0 && rg[2] >= 0) {
            for (size_t i = 0; i < NP; i++) { int miss = 0; float x[3]; for (int c = 0; c < 3; c++) { x[c] = lut[rg[c]][j->raw[v][rg[c]][i]]; miss |= x[c] < 0; } for (int c = 0; c < 3; c++) img[3 * i + c] = miss ? 0 : (uint8_t)(x[c] * 254 + 1); }
            snprintf(p, sizeof p, "%s/%s_rgb.jpg", od, vn[v]); write_jpg(p, j->W, j->H, img, 90);
        }
        if (nb3 == 3 && dp[0][0] >= 0 && dp[1][0] >= 0 && dp[2][0] >= 0 && dp[0][1] >= 0 && dp[1][1] >= 0 && dp[2][1] >= 0) {
            for (size_t i = 0; i < NP; i++) { int miss = 0; uint8_t o3[3];
                for (int c = 0; c < 3; c++) { const float x = lut[dp[c][0]][j->raw[v][dp[c][0]][i]], y = lut[dp[c][1]][j->raw[v][dp[c][1]][i]]; miss |= x < 0 || y < 0; o3[c] = (uint8_t)fmin(255, fmax(1, 128 + gain * (x - y) * 1.0)); }
                for (int c = 0; c < 3; c++) img[3 * i + c] = miss ? 0 : o3[c]; }
            snprintf(p, sizeof p, "%s/%s_diff.jpg", od, vn[v]); write_jpg(p, j->W, j->H, img, 90);
        }
        if (nd > nb3 && dp[nb3][0] >= 0 && dp[nb3][1] >= 0) {
            for (size_t i = 0; i < NP; i++) { const float x = lut[dp[nb3][0]][j->raw[v][dp[nb3][0]][i]], y = lut[dp[nb3][1]][j->raw[v][dp[nb3][1]][i]];
                const uint8_t g = x < 0 || y < 0 ? 0 : (uint8_t)fmin(255, fmax(1, 128 + gain * (x - y))); img[3 * i] = img[3 * i + 1] = img[3 * i + 2] = g; }
            snprintf(p, sizeof p, "%s/%s_baseline.jpg", od, vn[v]); write_jpg(p, j->W, j->H, img, 90);
        }
    }
    fprintf(st, "}}\n"); fclose(st);
    /* coverage: R G B = the rgb scans present (native) */
    { int rg[3] = {-1, -1, -1}; for (int c = 0; c < 3 && rgb; c++) rg[c] = find_scan(j, json_str(json_at(rgb, c), ""));
      for (size_t i = 0; i < NP; i++) for (int c = 0; c < 3; c++) img[3 * i + c] = rg[c] >= 0 && j->raw[0][rg[c]][i] ? 200 : 0;
      snprintf(p, sizeof p, "%s/coverage.jpg", od); jpeg_write_rgb(p, j->W, j->H, img, 85); }
    fprintf(stderr, "reg render: wrote %s (%ld s)\n", od, (long)(time(nullptr) - j->t0));
    free(img);
    return 0;
}
