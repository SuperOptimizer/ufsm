/* ufsm vwt-stitch PASSA REF OUT [--threads 16] [--box z,y,x,nz,ny,nx]: pass B of whole-volume ray-model inference (src/vwt_tile.h).

   PASSA: the pass-A directory (tiling.bin, phase.f16, centre.f32, pairs.f32; ufsm vwt-infer). REF: a fit-field set on the same grid
   (wind96's whole-scroll fields: origin 0, h 2); its w plane is linked and its axis table copied. OUT: a fit-field set with REF's shape
   and grid, filled inside the box (default: the tiling's box) with
     pc / ps   128 + 127 cos / sin(2 pi W)   (W = the registered phase; confidence 1)
     t0..t5    n n^T of n = grad W / |grad W| (z y x; diagonal x 255, off-diagonal (. + 0.5) x 255)
     sp        8 / (h |grad W|) (sheet spacing in h / 8 units, clamped 1..255; 0 = none)
     wv        float32 W (turns, one gauge per connected slab component)
   W at a cell centre = the phase of the cell's OWNER slab (the computed slab whose interior holds the cell deepest, normalised by the
   slab's half-extents) + its offset. Offsets: per neighbour pair the median over the shared points of phase_i - phase_j, robust
   (Huber 0.25 turn) least squares over the slab graph by IRLS + conjugate gradients, each component's mean offset 0. */
#include "vwt_tile.h"
#include "sysinfo.h"
#include "json.h"
#include "z3w.h"
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static const char *sopt(int argc, char **argv, const char *k, const char *d) { for (int i = argc - 2; i >= 1; i--) if (!strcmp(argv[i], k)) return argv[i + 1]; return d; }
static float h2f(uint16_t u) {
    const uint32_t s = (u & 0x8000u) << 16; int e = (u >> 10) & 31; uint32_t m = u & 0x3ff, v;
    if (e == 0) { if (!m) v = s; else { e = 1; while (!(m & 0x400)) { m <<= 1; e--; } m &= 0x3ff; v = s | (uint32_t)(e + 112) << 23 | m << 13; } }
    else if (e == 31) v = s | 0x7f800000u | m << 13; else v = s | (uint32_t)(e + 112) << 23 | m << 13;
    float f; memcpy(&f, &v, 4); return f;
}
static int cmpf(const void *a, const void *b) { const float x = *(const float *)a, y = *(const float *)b; return (x > y) - (x < y); }

/* ---- offsets ---- */
typedef struct { int i, j; double d, w; } edge;
static void solve(int n, const uint8_t *on, const edge *E, int ne, double *c) {
    double *r = calloc(n, 8), *p = calloc(n, 8), *q = calloc(n, 8), *b = calloc(n, 8), *ew = malloc((ne ? ne : 1) * 8);
    for (int e = 0; e < ne; e++) ew[e] = E[e].w;
    for (int it = 0; it < 8; it++) {   /* IRLS: Huber weights on the current residuals */
        if (it) for (int e = 0; e < ne; e++) { const double res = fabs((c[E[e].j] - c[E[e].i]) - E[e].d); ew[e] = E[e].w * (res <= 0.25 ? 1 : 0.25 / res); }
        /* normal equations (L + eps I) c = b, L the weighted graph Laplacian, b_j += w d, b_i -= w d */
        memset(b, 0, n * 8); for (int e = 0; e < ne; e++) { b[E[e].j] += ew[e] * E[e].d; b[E[e].i] -= ew[e] * E[e].d; }
#define AX(x, y) do { for (int k = 0; k < n; k++) y[k] = on[k] ? 1e-6 * x[k] : 0; for (int e = 0; e < ne; e++) { const double t = ew[e] * (x[E[e].j] - x[E[e].i]); y[E[e].j] += t; y[E[e].i] -= t; } } while (0)
        AX(c, q); double rr = 0; for (int k = 0; k < n; k++) { r[k] = on[k] ? b[k] - q[k] : 0; p[k] = r[k]; rr += r[k] * r[k]; }
        for (int cg = 0; cg < 2000 && rr > 1e-12; cg++) {
            AX(p, q); double pq = 0; for (int k = 0; k < n; k++) pq += p[k] * q[k]; if (pq <= 0) break;
            const double a = rr / pq; double rn = 0; for (int k = 0; k < n; k++) { c[k] += a * p[k]; r[k] -= a * q[k]; rn += r[k] * r[k]; }
            const double bt = rn / rr; rr = rn; for (int k = 0; k < n; k++) p[k] = r[k] + bt * p[k];
        }
#undef AX
    }
    free(r); free(p); free(q); free(b); free(ew);
}

/* ---- rasterise ---- */
typedef struct {
    const vwt_tiling *t; const uint16_t *ph; const float *cen; const double *off; const uint8_t *on;
    int64_t ro[3], rs[3];   /* REF origin (voxels) and shape (cells) */
    int h; int64_t c0[3], c1[3];   /* the cell box to fill */
    z3w *zw[10]; z3w *zv;
    int nsz, nsy, nsx; int next; pthread_mutex_t mu;
    double (*bb)[6];        /* per slab world bbox zyx lo / hi */
    long filled;
} rctx;
static double gphase(const uint16_t *g, double gu, double gv, double gw, int *ok) {   /* trilinear on the 32 x 32 x 192 grid */
    if (gu < 0 || gv < 0 || gw < 0 || gu > VWT_PU - 1 || gv > VWT_PU - 1 || gw > VWT_PW - 1) { *ok = 0; return 0; }
    int a = (int)gu, b = (int)gv, k = (int)gw; if (a > VWT_PU - 2) a = VWT_PU - 2; if (b > VWT_PU - 2) b = VWT_PU - 2; if (k > VWT_PW - 2) k = VWT_PW - 2;
    const double fa = gu - a, fb = gv - b, fk = gw - k; double s = 0;
    for (int c = 0; c < 8; c++) {
        const int da = c & 1, db = c >> 1 & 1, dk = c >> 2 & 1; const float v = h2f(g[((size_t)(a + da) * VWT_PU + b + db) * VWT_PW + k + dk]);
        if (isnan(v)) { *ok = 0; return 0; }
        s += (da ? fa : 1 - fa) * (db ? fb : 1 - fb) * (dk ? fk : 1 - fk) * v;
    }
    *ok = 1; return s;
}
static void *rworker(void *arg) {
    rctx *R = arg; const vwt_tiling *t = R->t; const int S = 128, T = t->T, L = t->L, h = R->h;
    float *W = malloc((size_t)S * S * S * 4), *G = malloc((size_t)S * S * S * 12), *sc = malloc((size_t)S * S * S * 4);
    uint8_t *pl[9]; for (int p = 0; p < 9; p++) pl[p] = malloc((size_t)S * S * S);
    for (;;) {
        pthread_mutex_lock(&R->mu); const int id = R->next++; pthread_mutex_unlock(&R->mu);
        if (id >= R->nsz * R->nsy * R->nsx) break;
        const int sx = id % R->nsx, sy = id / R->nsx % R->nsy, sz = id / (R->nsx * R->nsy);
        const int64_t z0 = (int64_t)sz * S, y0 = (int64_t)sy * S, x0 = (int64_t)sx * S;   /* cells of REF */
        if (z0 + S <= R->c0[0] || z0 >= R->c1[0] || y0 + S <= R->c0[1] || y0 >= R->c1[1] || x0 + S <= R->c0[2] || x0 >= R->c1[2]) continue;
        for (size_t q = 0; q < (size_t)S * S * S; q++) { sc[q] = -1; W[q] = NAN; }
        /* world (voxel) extent of this shard */
        const double wl[3] = {R->ro[0] + z0 * h, R->ro[1] + y0 * h, R->ro[2] + x0 * h}, wh[3] = {wl[0] + S * h, wl[1] + S * h, wl[2] + S * h};
        int any = 0;
        for (int k = 0; k < t->n; k++) {
            if (!R->on[k]) continue;
            const double *b = R->bb[k]; if (b[3] < wl[0] || b[0] > wh[0] || b[4] < wl[1] || b[1] > wh[1] || b[5] < wl[2] || b[2] > wh[2]) continue;
            const vwt_slab *s = &t->s[k]; const uint16_t *g = R->ph + (size_t)k * VWT_PH_N; const double base = R->cen[k] + R->off[k];
            /* the cells inside the slab bbox */
            int lo[3], hi[3]; for (int e = 0; e < 3; e++) { lo[e] = (int)floor((b[e] - wl[e]) / h); hi[e] = (int)ceil((b[3 + e] - wl[e]) / h); if (lo[e] < 0) lo[e] = 0; if (hi[e] > S) hi[e] = S; }
            for (int cz = lo[0]; cz < hi[0]; cz++) {
                const int64_t gz = z0 + cz; if (gz < R->c0[0] || gz >= R->c1[0]) continue;
                for (int cy = lo[1]; cy < hi[1]; cy++) {
                    const int64_t gy = y0 + cy; if (gy < R->c0[1] || gy >= R->c1[1]) continue;
                    for (int cx = lo[2]; cx < hi[2]; cx++) {
                        const int64_t gx = x0 + cx; if (gx < R->c0[2] || gx >= R->c1[2]) continue;
                        const double p[3] = {wl[2] + (cx + 0.5) * h - 0.5, wl[1] + (cy + 0.5) * h - 0.5, wl[0] + (cz + 0.5) * h - 0.5};   /* xyz voxel coords of the cell centre */
                        double u[3]; vwt_to_slab(s, p, u);
                        const double di = fmin(u[0], T - 1 - u[0]) / (0.5 * (T - 1)), dj = fmin(u[1], T - 1 - u[1]) / (0.5 * (T - 1)), dk = fmin(u[2], L - 1 - u[2]) / (0.5 * (L - 1));
                        const double score = fmin(di, fmin(dj, dk)); const size_t q = ((size_t)cz * S + cy) * S + cx;
                        if (score <= 0.02 || score <= sc[q]) continue;
                        const double gu = (u[0] - 1.5) / 4, gv = (u[1] - 1.5) / 4, gw = (u[2] - 0.5) / 2; int ok, o1, o2, o3, o4, o5, o6;
                        const double v = gphase(g, gu, gv, gw, &ok); if (!ok) continue;
                        const double du = (gphase(g, gu + 0.5, gv, gw, &o1) - gphase(g, gu - 0.5, gv, gw, &o2)) / 4.0, dv = (gphase(g, gu, gv + 0.5, gw, &o3) - gphase(g, gu, gv - 0.5, gw, &o4)) / 4.0,
                                     dw = (gphase(g, gu, gv, gw + 0.5, &o5) - gphase(g, gu, gv, gw - 0.5, &o6)) / 2.0;   /* turns per voxel along a, b, d */
                        if (!(o1 && o2 && o3 && o4 && o5 && o6)) continue;
                        sc[q] = (float)score; W[q] = (float)(base + v);
                        double gr[3]; for (int e = 0; e < 3; e++) gr[e] = du * s->ax[e] + dv * s->bx[e] + dw * s->d[e];   /* xyz */
                        G[q * 3] = (float)gr[2]; G[q * 3 + 1] = (float)gr[1]; G[q * 3 + 2] = (float)gr[0];   /* z y x */
                        any = 1;
                    }
                }
            }
        }
        if (!any) continue;
        for (size_t q = 0; q < (size_t)S * S * S; q++) {
            if (isnan(W[q])) { for (int p = 0; p < 9; p++) pl[p][q] = 0; continue; }
            const double a = 2 * M_PI * W[q]; pl[0][q] = (uint8_t)lround(128 + 127 * cos(a)); pl[1][q] = (uint8_t)lround(128 + 127 * sin(a));
            const double gz = G[q * 3], gy = G[q * 3 + 1], gx = G[q * 3 + 2], gn = sqrt(gz * gz + gy * gy + gx * gx);
            if (gn < 1e-6) { for (int p = 2; p < 9; p++) pl[p][q] = 0; continue; }
            const double nz = gz / gn, ny = gy / gn, nx = gx / gn;
            pl[2][q] = (uint8_t)lround(255 * nz * nz); pl[3][q] = (uint8_t)lround(255 * ny * ny); pl[4][q] = (uint8_t)lround(255 * nx * nx);
            pl[5][q] = (uint8_t)lround(255 * (nz * ny + 0.5)); pl[6][q] = (uint8_t)lround(255 * (nz * nx + 0.5)); pl[7][q] = (uint8_t)lround(255 * (ny * nx + 0.5));
            const double spv = 8.0 / (h * gn); pl[8][q] = (uint8_t)(spv < 1 ? 1 : spv > 255 ? 255 : lround(spv));
        }
        for (int p = 0; p < 9; p++) if (z3w_write_shard(R->zw[p], sz, sy, sx, pl[p], 1)) fprintf(stderr, "vwt-stitch: %s\n", z3w_error());
        if (z3w_write_shard(R->zv, sz, sy, sx, (const uint8_t *)W, 1)) fprintf(stderr, "vwt-stitch: %s\n", z3w_error());
        pthread_mutex_lock(&R->mu); R->filled++; pthread_mutex_unlock(&R->mu);
    }
    free(W); free(G); free(sc); for (int p = 0; p < 9; p++) free(pl[p]);
    return nullptr;
}

static void *map_file(const char *dir, const char *name, size_t need) {
    char p[1400]; snprintf(p, sizeof p, "%s/%s", dir, name); const int fd = open(p, O_RDONLY); if (fd < 0) { fprintf(stderr, "vwt-stitch: cannot open %s\n", p); return nullptr; }
    struct stat st; fstat(fd, &st); if ((size_t)st.st_size < need) { fprintf(stderr, "vwt-stitch: %s is %lld bytes, need %zu\n", p, (long long)st.st_size, need); close(fd); return nullptr; }
    void *m = mmap(nullptr, need, PROT_READ, MAP_SHARED, fd, 0); close(fd); return m == MAP_FAILED ? nullptr : m;
}

int cmd_vwt_stitch(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: ufsm vwt-stitch PASSA REF OUT [--threads 16] [--box z,y,x,nz,ny,nx]\n"); return 2; }
    const char *pa = argv[2], *ref = argv[3], *out = argv[4]; const int nth = ufsm_threads(atoi(sopt(argc, argv, "--threads", "0")), 16);
    char p[1400]; snprintf(p, sizeof p, "%s/tiling.bin", pa); vwt_tiling t; if (vwt_tiling_load(p, &t)) { fprintf(stderr, "vwt-stitch: cannot load %s\n", p); return 1; }
    const uint16_t *ph = map_file(pa, "phase.f16", (size_t)t.n * VWT_PH_N * 2); const float *cen = map_file(pa, "centre.f32", (size_t)t.n * 4);
    const float *pr = map_file(pa, "pairs.f32", (size_t)t.np * 2 * VWT_PAIR_PTS * 4);
    if (!ph || !cen || !pr) return 1;
    uint8_t *on = calloc(t.n, 1); int non = 0; for (int k = 0; k < t.n; k++) { uint32_t bits; memcpy(&bits, cen + k, 4); on[k] = bits && !isnan(cen[k]); non += on[k]; }   /* 0 bits: no part wrote it */
    /* edges */
    edge *E = malloc((size_t)(t.np ? t.np : 1) * sizeof *E); int ne = 0; float *dv = malloc(VWT_PAIR_PTS * 4); double rs = 0;
    for (int q = 0; q < t.np; q++) {
        const int i = t.p[q].i, j = t.p[q].j; if (!on[i] || !on[j]) continue;
        const float *a = pr + ((size_t)q * 2) * VWT_PAIR_PTS, *b = a + VWT_PAIR_PTS; int n = 0;
        for (int k = 0; k < t.p[q].npts; k++) if (!isnan(a[k]) && !isnan(b[k])) dv[n++] = a[k] - b[k];
        if (n < 8) continue;
        qsort(dv, n, 4, cmpf); const double med = dv[n / 2]; double mad = 0; for (int k = 0; k < n; k++) mad += fabs(dv[k] - med); mad /= n; rs += mad;
        E[ne++] = (edge){i, j, med, n / (1.0 + 10 * mad)};
    }
    double *off = calloc(t.n, 8); solve(t.n, on, E, ne, off);
    /* components (union-find over edges) -> zero-mean offsets */
    int *par = malloc(t.n * sizeof *par); for (int k = 0; k < t.n; k++) par[k] = k;
#define FIND(x) ({ int _r = (x); while (par[_r] != _r) _r = par[_r] = par[par[_r]]; _r; })
    for (int e = 0; e < ne; e++) { const int a = FIND(E[e].i), b = FIND(E[e].j); if (a != b) par[a] = b; }
    double *cs = calloc(t.n, 8); int *cn = calloc(t.n, sizeof *cn), ncomp = 0;
    for (int k = 0; k < t.n; k++) if (on[k]) { const int r = FIND(k); cs[r] += off[k]; ncomp += !cn[r]++; }
    for (int k = 0; k < t.n; k++) if (on[k]) { const int r = FIND(k); off[k] -= cs[r] / cn[r]; }
#undef FIND
    double rres[3] = {0, 0, 0};
    for (int e = 0; e < ne; e++) { const double r = fabs((off[E[e].j] - off[E[e].i]) - E[e].d); rres[0] += r; rres[1] += r >= 0.25; rres[2] += r >= 0.5; }
    fprintf(stderr, "vwt-stitch: %d slabs (%d computed), %d pairs used, %d components; pair MAD %.3f turn; residual mean %.3f, >= .25 %.3f, >= .5 %.3f\n",
            t.n, non, ne, ncomp, ne ? rs / ne : 0, ne ? rres[0] / ne : 0, ne ? rres[1] / ne : 0, ne ? rres[2] / ne : 0);
    /* REF grid */
    snprintf(p, sizeof p, "%s/fields.json", ref); FILE *f = fopen(p, "rb"); if (!f) { fprintf(stderr, "vwt-stitch: cannot read %s\n", p); return 1; }
    fseek(f, 0, SEEK_END); const long jl = ftell(f); fseek(f, 0, SEEK_SET); char *jt = malloc(jl + 1); if (fread(jt, 1, jl, f) != (size_t)jl) return 1; fclose(f); jt[jl] = 0;
    json *J = json_parse(jt, jl); free(jt); if (!J) return 1;
    rctx R = {0}; R.t = &t; R.ph = ph; R.cen = cen; R.off = off; R.on = on; R.h = (int)json_num(json_get(J, "h"), 2);
    for (int d = 0; d < 3; d++) { R.ro[d] = (int64_t)json_num(json_at(json_get(J, "origin"), d), 0); R.rs[d] = (int64_t)json_num(json_at(json_get(J, "shape"), d), 0); }
    long long bx[6] = {t.box[0], t.box[1], t.box[2], t.box[3], t.box[4], t.box[5]};
    if (*sopt(argc, argv, "--box", "")) sscanf(sopt(argc, argv, "--box", ""), "%lld,%lld,%lld,%lld,%lld,%lld", bx, bx + 1, bx + 2, bx + 3, bx + 4, bx + 5);
    for (int d = 0; d < 3; d++) { R.c0[d] = (bx[d] - R.ro[d]) / R.h; R.c1[d] = (bx[d] + bx[3 + d] - R.ro[d] + R.h - 1) / R.h; if (R.c0[d] < 0) R.c0[d] = 0; if (R.c1[d] > R.rs[d]) R.c1[d] = R.rs[d]; }
    /* outputs: fields.json = REF's (recto off), planes, w linked */
    mkdir(out, 0755);
    {
        snprintf(p, sizeof p, "%s/fields.json", out); FILE *fo = fopen(p, "w"); if (!fo) return 1;
        fprintf(fo, "{\"format\":\"zarr\",\"origin\":[%lld,%lld,%lld],\"shape\":[%lld,%lld,%lld],\"h\":%d,\"um\":%g,\"recto\":false,\"axis\":[", (long long)R.ro[0], (long long)R.ro[1], (long long)R.ro[2],
                (long long)R.rs[0], (long long)R.rs[1], (long long)R.rs[2], R.h, json_num(json_get(J, "um"), 9.6));
        const json *ax = json_get(J, "axis"); for (size_t i = 0; ax && i < ax->n; i++) fprintf(fo, "%s[%.3f,%.3f]", i ? "," : "", json_num(json_at(json_at(ax, i), 0), 0), json_num(json_at(json_at(ax, i), 1), 0));
        fprintf(fo, "]}\n"); fclose(fo);
        char src[1400], dst[1400]; snprintf(src, sizeof src, "%s/w", ref); char *rp = realpath(src, nullptr); snprintf(dst, sizeof dst, "%s/w", out); unlink(dst);
        if (!rp || symlink(rp, dst)) { fprintf(stderr, "vwt-stitch: cannot link %s\n", src); return 1; } free(rp);
    }
    json_free(J);
    const char *nm[9] = {"pc", "ps", "t0", "t1", "t2", "t3", "t4", "t5", "sp"}; char attrs[512];
    snprintf(attrs, sizeof attrs, "{\"ufsm\":{\"content\":\"vwt ray-model fields (src/vwt_stitch.c)\",\"h\":%d}}", R.h);
    for (int q = 0; q < 9; q++) { snprintf(p, sizeof p, "%s/%s", out, nm[q]); R.zw[q] = z3w_create(p, R.rs, 128, 0.f, 0, attrs); if (!R.zw[q]) { fprintf(stderr, "vwt-stitch: %s\n", z3w_error()); return 1; } }
    snprintf(p, sizeof p, "%s/wv", out); R.zv = z3w_create_typed(p, R.rs, 128, 5, 0.f, NAN, attrs); if (!R.zv) { fprintf(stderr, "vwt-stitch: %s\n", z3w_error()); return 1; }
    R.bb = malloc((size_t)t.n * sizeof *R.bb);
    for (int k = 0; k < t.n; k++) {
        const vwt_slab *s = &t.s[k]; double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
        for (int c = 0; c < 8; c++) { const double i = (c & 1) ? t.T - 1 : 0, j = (c & 2) ? t.T - 1 : 0, kk = (c & 4) ? t.L - 1 : 0;
            for (int e = 0; e < 3; e++) { const double v = s->o[e] + i * s->ax[e] + j * s->bx[e] + kk * s->d[e]; lo[e] = fmin(lo[e], v); hi[e] = fmax(hi[e], v); } }
        R.bb[k][0] = lo[2]; R.bb[k][1] = lo[1]; R.bb[k][2] = lo[0]; R.bb[k][3] = hi[2]; R.bb[k][4] = hi[1]; R.bb[k][5] = hi[0];
    }
    R.nsz = (int)((R.rs[0] + 127) / 128); R.nsy = (int)((R.rs[1] + 127) / 128); R.nsx = (int)((R.rs[2] + 127) / 128); pthread_mutex_init(&R.mu, nullptr);
    pthread_t th[64]; const int nt = nth < 1 ? 1 : nth > 64 ? 64 : nth;
    for (int i = 0; i < nt; i++) pthread_create(&th[i], nullptr, rworker, &R);
    for (int i = 0; i < nt; i++) pthread_join(th[i], nullptr);
    for (int q = 0; q < 9; q++) z3w_close(R.zw[q]);
    z3w_close(R.zv);
    fprintf(stderr, "vwt-stitch: %ld shards of 128^3 cells written to %s\n", R.filled, out);
    return 0;
}
