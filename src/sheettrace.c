/* ufsm sheet-trace: fast sheet patch tracer on a sheet probability (unroll v3 P3b; replaces the CPU-only VC3D grower for corpus scale).
   ufsm sheet-trace --prob P [--prob-key K] --box Z0,Y0,X0,NZ,NY,NX --axis AXIS.json [--axis-scale S] --out DIR [--vs UM]
                    [--step 2] [--thr 64] [--seed-thr 128] [--tol 2] [--amb 0] [--min-cells 100] [--threads N] [--cache DIR]
   Model: around the scroll axis a sheet is locally a height field r(theta, z). The box is resampled on a column grid (z every --step
   voxels, theta every --step voxels of arc at the box's median radius); along each column's radial ray the prob is sampled every 0.5
   voxel (bilinear in the z plane) and its local maxima >= --thr become the column's peaks (sub-voxel by a parabola fit). Patches grow
   breadth-first from unowned peaks >= --seed-thr (strongest first): from an owned peak at r, the 4 neighbour columns are entered at the
   extrapolated r' = r + (r - r_behind) (r_behind: the patch's peak one column further back, else r'); the step is taken only when exactly
   one peak lies within --tol of r' and no other within --tol + --amb (ambiguous steps stop the front), the column has no other peak of
   the patch, and the peak is unowned. Running into a peak of another patch records a same-sheet contact.
   Output: DIR/paths/<name>/ tifxyz (rows z, cols theta, -1 where a patch has no peak; step = --step; winding -1) of the patches with
   >= --min-cells cells; --tile T writes each patch as T x T cell tiles (tiles with >= --min-tile cells), so a patch that switched sheets
   through a merge can be cut by the turn solve; DIR/tiles.tsv: tile, patch, tile row / col, cells; DIR/links.tsv: a b contacts kind
   (same-sheet evidence between output tiles: seam = neighbouring tiles of one patch, contact = a patch's front ran into another's peak,
   rseam = a front ran into a tile of an earlier region).
   --regions R: an R x R grid of xy regions (all z), each with its own ray frame (folded / mixed-orientation boxes); a later region does not
   trace peaks on 2-voxel cells claimed by earlier regions' tiles and links to them (rseam). frame.txt = the frame picked for the whole box.
   Coordinates are box voxels (as VC3D patches traced on the box): patch-solve / patch-score read them directly. */
#include "patchsolve.h"
#include "tiffw.h"
#include "imgops.h"
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static const char *st_opt(int argc, char **argv, const char *k, const char *d) { for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return argv[i + 1]; return d; }

typedef struct { float r; uint8_t p; } st_peak;
typedef struct {
    const uint8_t *prob, *extra; int64_t n[3]; const ps_geo *G; int extra_p; double extra_sep, extra_gap, nms;   /* extra: second sheet evidence (codes / mask) */
    int nzc, ntc, step; double th0, dth, rlo, rmax; int thr;   /* columns: u = th0 + it dth; r = rlo + 0.5 k, k < 2 (rmax - rlo) */
    double ylo, yhi, xlo, xhi;   /* the region: samples outside [ylo, yhi] x [xlo, xhi] are not on the column */
    st_peak **pk; uint16_t *npk;   /* per column (iz * ntc + it) */
    _Atomic int next;
} st_job;

static inline double st_bil(const uint8_t *s, int64_t ny, int64_t nx, double y, double x) {
    const int64_t y0 = (int64_t)floor(y), x0 = (int64_t)floor(x); const double fy = y - y0, fx = x - x0;
    const uint8_t *a = s + y0 * nx + x0;
    return (1 - fy) * ((1 - fx) * a[0] + fx * a[1]) + fy * ((1 - fx) * a[nx] + fx * a[nx + 1]);
}
static void *st_cols(void *arg) {
    st_job *J = arg; const int64_t NY = J->n[1], NX = J->n[2];
    double *prof = malloc(sizeof(double) * (size_t)(2 * (J->rmax - J->rlo) + 8)); st_peak *tmp = malloc(sizeof(st_peak) * (size_t)(J->rmax - J->rlo + 8));
    for (;;) {
        const int iz = atomic_fetch_add(&J->next, 1); if (iz >= J->nzc) break;
        const int z = iz * J->step;
        const uint8_t *sl = J->prob + (size_t)z * NY * NX;
        for (int it = 0; it < J->ntc; it++) {
            const double th = J->th0 + it * J->dth;
            const int L = (int)(2 * (J->rmax - J->rlo)) + 1; int first = -1, last = -1;
            for (int k = 0; k < L; k++) {
                const double r = J->rlo + 0.5 * k; double y, x; ps_frame_pt(J->G, z, r, th, &y, &x);
                if (y < 0 || x < 0 || y > NY - 1.001 || x > NX - 1.001 || y < J->ylo || y > J->yhi || x < J->xlo || x > J->xhi) { prof[k] = -1; continue; }
                prof[k] = st_bil(sl, NY, NX, y, x); if (first < 0) first = k; last = k;
            }
            int np_ = 0;
            if (first >= 0)
                for (int k = first + 1; k < last; k++) {
                    const double v = prof[k]; if (v < J->thr || prof[k - 1] < 0 || prof[k + 1] < 0) continue;
                    if (!(v >= prof[k - 1] && v > prof[k + 1])) continue;
                    const double a = prof[k - 1], c = prof[k + 1], den = a - 2 * v + c; double off = den < 0 ? 0.5 * (a - c) / den : 0;
                    off = off < -0.5 ? -0.5 : off > 0.5 ? 0.5 : off;
                    tmp[np_++] = (st_peak){(float)(J->rlo + 0.5 * (k + off)), (uint8_t)(v > 255 ? 255 : v + 0.5)};
                }
            if (J->nms > 0 && np_ > 1) {   /* --nms: within +-nms voxels along the ray keep only the strongest peak (both faces of one sheet) */
                uint8_t keep[4096]; const int m = np_ < 4096 ? np_ : 4096;
                for (int a = 0; a < m; a++) {
                    keep[a] = 1;
                    for (int c = a - 1; c >= 0 && tmp[a].r - tmp[c].r <= J->nms && keep[a]; c--) if (tmp[c].p > tmp[a].p || (tmp[c].p == tmp[a].p && keep[c])) keep[a] = 0;
                    for (int c = a + 1; c < m && tmp[c].r - tmp[a].r <= J->nms && keep[a]; c++) if (tmp[c].p > tmp[a].p) keep[a] = 0;
                }
                int w = 0; for (int a = 0; a < m; a++) if (keep[a]) tmp[w++] = tmp[a];
                np_ = w;
            }
            if (J->extra) {   /* runs of extra sheet voxels along the ray (nearest voxel): a peak at the run centre where prob has none within extra_sep */
                const uint8_t *se = J->extra + (size_t)z * NY * NX; int run0 = -1; const int np0 = np_;
                double pitch = 0;   /* --extra-gap: the column's median prob-peak spacing */
                if (J->extra_gap > 0 && np0 >= 3) {
                    double g[512]; int m = 0; for (int q = 1; q < np0 && m < 512; q++) g[m++] = tmp[q].r - tmp[q - 1].r;
                    for (int a = 1; a < m; a++) { const double t = g[a]; int c = a - 1; while (c >= 0 && g[c] > t) { g[c + 1] = g[c]; c--; } g[c + 1] = t; }
                    pitch = g[m / 2];
                }
                for (int k = 0; k <= L; k++) {
                    int on = 0;
                    if (k < L && prof[k] >= 0) {
                        const double r = J->rlo + 0.5 * k; double y, x; ps_frame_pt(J->G, z, r, th, &y, &x);
                        const uint8_t v = se[(int64_t)lround(y) * NX + (int64_t)lround(x)]; on = v >= 1 && v <= 252;
                    }
                    if (on && run0 < 0) run0 = k;
                    if (!on && run0 >= 0) {
                        const double rc = J->rlo + 0.25 * (run0 + k - 1); int near = 0;
                        for (int q = 0; q < np0 && !near; q++) near = fabs(tmp[q].r - rc) < J->extra_sep;
                        if (!near && J->extra_gap > 0) {   /* only inside a gap >= extra_gap x pitch between prob peaks, >= 0.3 pitch from both */
                            double lo = -1e9, hi = 1e9; for (int q = 0; q < np0; q++) { if (tmp[q].r < rc && tmp[q].r > lo) lo = tmp[q].r; if (tmp[q].r > rc && tmp[q].r < hi) hi = tmp[q].r; }
                            near = !(pitch > 0 && hi - lo >= J->extra_gap * pitch && rc - lo >= 0.3 * pitch && hi - rc >= 0.3 * pitch);
                        }
                        if (!near && np_ < (int)(J->rmax - J->rlo + 7)) tmp[np_++] = (st_peak){(float)rc, (uint8_t)J->extra_p};
                        run0 = -1;
                    }
                }
                if (np_ > np0)   /* keep the column sorted by r */
                    for (int a = 1; a < np_; a++) { const st_peak t = tmp[a]; int c = a - 1; while (c >= 0 && tmp[c].r > t.r) { tmp[c + 1] = tmp[c]; c--; } tmp[c + 1] = t; }
            }
            const size_t ci = (size_t)iz * J->ntc + it;
            J->npk[ci] = (uint16_t)np_;
            J->pk[ci] = np_ ? malloc(sizeof(st_peak) * (size_t)np_) : nullptr;
            if (np_) memcpy(J->pk[ci], tmp, sizeof(st_peak) * (size_t)np_);
        }
    }
    free(prof); free(tmp); return nullptr;
}

typedef struct { uint32_t col; uint16_t k; } st_cell;
typedef struct { uint32_t a, b; } st_link;
static int st_linkcmp(const void *p, const void *q) { const st_link *u = p, *v = q; if (u->a != v->a) return u->a < v->a ? -1 : 1; return (u->b > v->b) - (u->b < v->b); }
typedef struct { uint32_t col; uint16_t k; uint8_t p; } st_seed;
static int st_seedcmp(const void *p, const void *q) {
    const st_seed *u = p, *v = q; if (u->p != v->p) return u->p > v->p ? -1 : 1;
    if (u->col != v->col) return u->col < v->col ? -1 : 1; return (u->k > v->k) - (u->k < v->k);
}

/* shared state of one sheet-trace run (all regions) */
typedef struct {
    const uint8_t *prob, *extra; int64_t n[3]; const ps_geo *G0; int extra_p; double extra_sep, extra_gap, nms; int step, thr, sthr, minc, tile, mint, nt; double tol, amb, vs; const char *dir, *pdir;
    int32_t *cell; int64_t cn[3];          /* 2-voxel cells -> output tile (-1): peaks of earlier regions' tiles are not traced again */
    st_link *tl; size_t ntl, ctl;          /* tile links: a * 4 + kind (0 seam, 1 contact, 2 region seam), b */
    FILE *tf; int32_t nout, npatch, nkept; size_t npk, kept, owned; long long nstop[4];
    uint8_t *rgb; int jz;                  /* QA image (rgb of the jz plane) or null */
    int nreg_par;
} st_ctx;
static inline int64_t st_cellof(const st_ctx *C, double z, double y, double x) {
    const int64_t cz = (int64_t)lround(z) / 2, cy = (int64_t)lround(y) / 2, cx = (int64_t)lround(x) / 2;
    if (cz < 0 || cy < 0 || cx < 0 || cz >= C->cn[0] || cy >= C->cn[1] || cx >= C->cn[2]) return -1;
    return (cz * C->cn[1] + cy) * C->cn[2] + cx;
}
static int st_tlink(st_ctx *C, int32_t a, int32_t b, int kind) {
    if (a < 0 || b < 0 || a == b) return 0;
    if (C->ntl == C->ctl) { C->ctl = C->ctl ? C->ctl * 2 : 4096; C->tl = realloc(C->tl, C->ctl * sizeof *C->tl); if (!C->tl) return -1; }
    C->tl[C->ntl++] = (st_link){(uint32_t)(a < b ? a : b) * 4u + (uint32_t)kind, (uint32_t)(a < b ? b : a)}; return 0;
}
/* the frame of a region from the structure tensor of its prob (see --dir) */
static void st_pick_frame(const st_ctx *C, ps_geo *G, int64_t y0, int64_t x0, int64_t y1, int64_t x1, int verbose, const char *tag) {
    const int64_t *n = C->n; double jxx = 0, jxy = 0, jyy = 0;
    for (int64_t z = 0; z < n[0]; z += 16) { const uint8_t *sl = C->prob + (size_t)z * n[1] * n[2];
        for (int64_t y = y0 + 1; y + 1 < y1; y += 2) for (int64_t x = x0 + 1; x + 1 < x1; x += 2) {
            const double gx = (double)sl[y * n[2] + x + 1] - sl[y * n[2] + x - 1], gy = (double)sl[(y + 1) * n[2] + x] - sl[(y - 1) * n[2] + x];
            jxx += gx * gx; jxy += gx * gy; jyy += gy * gy; } }
    const double phi = 0.5 * atan2(2 * jxy, jxx - jyy), coh = (jxx + jyy) > 0 ? sqrt((jxx - jyy) * (jxx - jyy) + 4 * jxy * jxy) / (jxx + jyy) : 0;
    double ny_ = sin(phi), nx_ = cos(phi), ay, ax; ps_axis(G, n[0] / 2.0, &ay, &ax);
    const double cy = 0.5 * (y0 + y1), cx = 0.5 * (x0 + x1), ey = cy - ay, ex = cx - ax, el = hypot(ey, ex) + 1e-9;
    if (ny_ * ey + nx_ * ex < 0) { ny_ = -ny_; nx_ = -nx_; }   /* outward */
    const double align = (ny_ * ey + nx_ * ex) / el;
    const int axis_in = ay >= y0 && ax >= x0 && ay < y1 && ax < x1;   /* a region around the axis stays radial (no outward normal) */
    G->par = !strcmp(C->dir, "par") || (!strcmp(C->dir, "auto") && !axis_in && align < 0.7 && coh >= 0.5);   /* parallel only for one clear orientation */
    if (G->par) { G->ny = ny_; G->nx = nx_; G->ty = nx_; G->tx = -ny_; G->cy = cy; G->cx = cx; }
    if (verbose) printf("sheet-trace: %s sheet normal (%.3f, %.3f) coherence %.2f, radial alignment %.2f -> %s rays\n", tag, ny_, nx_, coh, align, G->par ? "parallel" : "radial");
}

/* trace one xy region [y0, y1) x [x0, x1) (all z) in its own frame; tiles are numbered globally and claim their cells */
static int st_region(st_ctx *C, int64_t y0, int64_t x0, int64_t y1, int64_t x1, int verbose) {
    const int64_t *n = C->n; const int step = C->step; const double tol = C->tol, amb = C->amb;
    ps_geo G = *C->G0; char tag[64]; snprintf(tag, sizeof tag, "region y %lld..%lld x %lld..%lld:", (long long)y0, (long long)y1, (long long)x0, (long long)x1);
    st_pick_frame(C, &G, y0, x0, y1, x1, verbose, tag); C->nreg_par += G.par;
    /* column range over the region */
    double thmin = 1e9, thmax = -1e9, rmin = 1e9, rmax = -1e9, rsum = 0; long rcnt = 0; int has_axis = 0;
    for (int64_t z = 0; z < n[0]; z += 8) {
        double ay, ax; ps_axis(&G, (double)z, &ay, &ax);
        if (!G.par && ay >= y0 && ax >= x0 && ay < y1 && ax < x1) has_axis = 1;
        for (int64_t y = y0; y < y1; y += 4) for (int64_t x = x0; x < x1; x += 4) {
            double r, t; ps_frame(&G, (double)z, (double)y, (double)x, &r, &t);
            thmin = t < thmin ? t : thmin; thmax = t > thmax ? t : thmax; rmax = r > rmax ? r : rmax; rmin = r < rmin ? r : rmin;
            if (C->prob[((size_t)z * n[1] + y) * n[2] + x] >= C->thr) { rsum += r; rcnt++; }
        }
    }
    if (has_axis) { thmin = G.t0 - M_PI; thmax = G.t0 + M_PI; rmin = 0; }
    const double rref = rcnt ? rsum / rcnt : rmax / 2, dth = G.par ? (double)step : step / (rref > 4 ? rref : 4);
    const double rlo = G.par ? rmin - 4 : (rmin - 4 > 0 ? rmin - 4 : 0);
    st_job J = {.prob = C->prob, .extra = C->extra, .extra_p = C->extra_p, .extra_sep = C->extra_sep, .extra_gap = C->extra_gap, .nms = C->nms, .n = {n[0], n[1], n[2]}, .G = &G, .step = step, .th0 = thmin - dth, .dth = dth, .rlo = rlo, .rmax = rmax + 4, .thr = C->thr,
                .ylo = (double)y0, .yhi = (double)y1 - 1, .xlo = (double)x0, .xhi = (double)x1 - 1};
    J.nzc = (int)((n[0] - 1) / step) + 1; J.ntc = (int)((thmax - thmin) / dth) + 3;
    const size_t NC = (size_t)J.nzc * J.ntc;
    J.pk = calloc(NC, sizeof *J.pk); J.npk = calloc(NC, sizeof *J.npk); if (!J.pk || !J.npk) return -1;
    {   pthread_t th[64]; for (int t = 0; t < C->nt; t++) pthread_create(&th[t], nullptr, st_cols, &J); for (int t = 0; t < C->nt; t++) pthread_join(th[t], nullptr); }
    size_t npk = 0; for (size_t c = 0; c < NC; c++) npk += J.npk[c];
    if (verbose) printf("sheet-trace: columns %d x %d (z step %d, %s step %d%s), peaks %zu (%.1f per column)\n", J.nzc, J.ntc, step, G.par ? "tangent" : "arc", step,
                        has_axis ? ", holds the axis" : "", npk, NC ? (double)npk / NC : 0);
    size_t *off = malloc((NC + 1) * sizeof *off); off[0] = 0; for (size_t c = 0; c < NC; c++) off[c + 1] = off[c] + J.npk[c];
    int32_t *own = malloc((npk + 1) * 4), *tid = malloc((npk + 1) * 4); int64_t *pcell = malloc((npk + 1) * 8);
    for (size_t c = 0; c < NC; c++) {   /* peaks on cells of earlier regions' tiles: external (own = -2 - tile) */
        const int iz = (int)(c / J.ntc), it = (int)(c % J.ntc);
        for (int k = 0; k < J.npk[c]; k++) {
            double py, px; ps_frame_pt(&G, (double)(iz * step), J.pk[c][k].r, J.th0 + it * J.dth, &py, &px);
            const size_t pid = off[c] + (size_t)k; const int64_t ce = st_cellof(C, iz * step, py, px); pcell[pid] = ce; tid[pid] = -1;
            own[pid] = ce >= 0 && C->cell[ce] >= 0 ? -2 - C->cell[ce] : -1;
        }
    }
    st_seed *sd = malloc((npk + 1) * sizeof *sd); size_t ns = 0;
    for (size_t c = 0; c < NC; c++) for (int k = 0; k < J.npk[c]; k++) if (J.pk[c][k].p >= C->sthr && own[off[c] + (size_t)k] == -1) sd[ns++] = (st_seed){(uint32_t)c, (uint16_t)k, J.pk[c][k].p};
    qsort(sd, ns, sizeof *sd, st_seedcmp);
    st_cell *q = malloc((npk + 1) * sizeof *q); int32_t *colk = malloc(NC * 4); for (size_t c = 0; c < NC; c++) colk[c] = -1;
    typedef struct { uint64_t a, b; uint8_t kind; } st_plink;   /* peak pairs (b < 0 side: external tile in a) */
    st_plink *lk = nullptr; size_t nlk = 0, clk = 0;
    #define ST_LINK(A_, B_, K_) do { if (nlk == clk) { clk = clk ? clk * 2 : 4096; lk = realloc(lk, clk * sizeof *lk); if (!lk) return -1; } lk[nlk++] = (st_plink){(A_), (B_), (K_)}; } while (0)
    float *X = nullptr, *Y = nullptr, *Z = nullptr; size_t cxyz = 0;
    for (size_t si = 0; si < ns; si++) {
        const size_t pid0 = off[sd[si].col] + sd[si].k; if (own[pid0] != -1) continue;
        const int32_t P = C->npatch++;
        size_t qh = 0, qt = 0; q[qt++] = (st_cell){sd[si].col, sd[si].k}; own[pid0] = P; colk[sd[si].col] = sd[si].k;
        while (qh < qt) {
            const st_cell cc = q[qh++]; const int iz = (int)(cc.col / J.ntc), it = (int)(cc.col % J.ntc); const double r = J.pk[cc.col][cc.k].r;
            const int dz[4] = {-1, 1, 0, 0}, dt[4] = {0, 0, -1, 1};
            for (int d = 0; d < 4; d++) {
                const int jz = iz + dz[d], jt = it + dt[d]; if (jz < 0 || jt < 0 || jz >= J.nzc || jt >= J.ntc) continue;
                const uint32_t nc = (uint32_t)(jz * J.ntc + jt); if (colk[nc] >= 0) continue;   /* the patch already has a peak there */
                double rp = r; const int bz = iz - dz[d], bt = it - dt[d];
                if (bz >= 0 && bt >= 0 && bz < J.nzc && bt < J.ntc && colk[bz * J.ntc + bt] >= 0) rp = 2 * r - J.pk[bz * J.ntc + bt][colk[bz * J.ntc + bt]].r;
                int best = -1, nwin = 0; double bd = 1e9;
                for (int k = 0; k < J.npk[nc]; k++) { const double dd = fabs(J.pk[nc][k].r - rp); if (dd <= tol + amb) nwin++; if (dd < bd) { bd = dd; best = k; } }
                if (best < 0 || bd > tol) { C->nstop[0]++; continue; }
                if (nwin != 1) { C->nstop[1]++; continue; }
                const size_t pid = off[nc] + (size_t)best;
                if (own[pid] <= -2) { C->nstop[2]++; ST_LINK((uint64_t)(-2 - own[pid]), off[cc.col] + cc.k, 2); continue; }   /* an earlier region's tile */
                if (own[pid] >= 0) { C->nstop[2]++; if (own[pid] != P) ST_LINK(pid, off[cc.col] + cc.k, 1); continue; }
                C->nstop[3]++;
                own[pid] = P; colk[nc] = best; q[qt++] = (st_cell){nc, (uint16_t)best};
            }
        }
        if ((long)qt >= C->minc) {   /* write the patch as tiles of --tile cells (the whole patch when 0) */
            C->nkept++; C->kept += qt;
            int z0 = J.nzc, z1 = -1, t0 = J.ntc, t1 = -1;
            for (size_t i = 0; i < qt; i++) { const int iz = (int)(q[i].col / J.ntc), it = (int)(q[i].col % J.ntc); z0 = iz < z0 ? iz : z0; z1 = iz > z1 ? iz : z1; t0 = it < t0 ? it : t0; t1 = it > t1 ? it : t1; }
            const int TZ = C->tile > 0 ? C->tile : z1 - z0 + 1, TT = C->tile > 0 ? C->tile : t1 - t0 + 1;
            const int ntz = (z1 - z0) / TZ + 1, ntt = (t1 - t0) / TT + 1;
            int *tcnt = calloc((size_t)ntz * ntt, sizeof *tcnt); int32_t *tmap = malloc((size_t)ntz * ntt * 4);
            for (size_t i = 0; i < qt; i++) { const int iz = (int)(q[i].col / J.ntc), it = (int)(q[i].col % J.ntc); tcnt[((iz - z0) / TZ) * ntt + (it - t0) / TT]++; }
            for (int a = 0; a < ntz * ntt; a++) tmap[a] = tcnt[a] >= C->mint ? C->nout++ : -1;
            if (cxyz < (size_t)TZ * TT) { cxyz = (size_t)TZ * TT; X = realloc(X, cxyz * 4); Y = realloc(Y, cxyz * 4); Z = realloc(Z, cxyz * 4); if (!X || !Y || !Z) return -1; }
            for (int a = 0; a < ntz * ntt; a++) {
                if (tmap[a] < 0) continue;
                const int az = a / ntt, at = a % ntt, bz0 = z0 + az * TZ, bt0 = t0 + at * TT;
                const int h = (bz0 + TZ - 1 > z1 ? z1 : bz0 + TZ - 1) - bz0 + 1, w = (bt0 + TT - 1 > t1 ? t1 : bt0 + TT - 1) - bt0 + 1;
                for (size_t i = 0; i < (size_t)w * h; i++) X[i] = Y[i] = Z[i] = -1;
                for (int iz = bz0; iz < bz0 + h; iz++)
                    for (int it = bt0; it < bt0 + w; it++) {
                        const size_t col = (size_t)iz * J.ntc + it; if (colk[col] < 0) continue;
                        const double rr = J.pk[col][colk[col]].r, th = J.th0 + it * J.dth; const size_t k = (size_t)(iz - bz0) * w + (size_t)(it - bt0);
                        double py, px; ps_frame_pt(&G, (double)(iz * step), rr, th, &py, &px);
                        X[k] = (float)px; Y[k] = (float)py; Z[k] = (float)(iz * step);
                        const size_t pid = off[col] + (size_t)colk[col]; tid[pid] = tmap[a];
                        if (pcell[pid] >= 0 && C->cell[pcell[pid]] < 0) C->cell[pcell[pid]] = tmap[a];
                    }
                char d[1400]; snprintf(d, sizeof d, "%s/st%06d", C->pdir, tmap[a]);
                if (tifxyz_write(d, w, h, X, Y, Z, step, C->vs, -1, "sheet-trace")) { fprintf(stderr, "sheet-trace: cannot write %s\n", d); return -1; }
                fprintf(C->tf, "st%06d\t%d\t%d\t%d\t%d\n", tmap[a], C->nkept - 1, az, at, tcnt[a]);
            }
            if (C->tile > 0)   /* seams between this patch's tiles */
                for (size_t i = 0; i < qt; i++) {
                    const int iz = (int)(q[i].col / J.ntc), it = (int)(q[i].col % J.ntc); const size_t pa = off[q[i].col] + q[i].k;
                    if (iz + 1 < J.nzc && (iz + 1 - z0) / TZ != (iz - z0) / TZ && colk[q[i].col + J.ntc] >= 0) ST_LINK(pa, off[q[i].col + J.ntc] + (size_t)colk[q[i].col + J.ntc], 0);
                    if (it + 1 < J.ntc && (it + 1 - t0) / TT != (it - t0) / TT && colk[q[i].col + 1] >= 0) ST_LINK(pa, off[q[i].col + 1] + (size_t)colk[q[i].col + 1], 0);
                }
            free(tcnt); free(tmap);
        }
        for (size_t i = 0; i < qt; i++) colk[q[i].col] = -1;
    }
    #undef ST_LINK
    for (size_t i = 0; i < nlk; i++) {   /* peak links -> tile links */
        const int32_t a = lk[i].kind == 2 ? (int32_t)lk[i].a : tid[lk[i].a], b = tid[lk[i].b];
        if (st_tlink(C, a, b, lk[i].kind)) return -1;
    }
    if (C->rgb) {   /* QA dots of this region on the jz plane */
        const int W = (int)n[2], H = (int)n[1], jrow = C->jz / step;
        for (int it = 0; it < J.ntc && jrow < J.nzc; it++) {
            const size_t col = (size_t)jrow * J.ntc + it; const double th = J.th0 + it * J.dth;
            for (int k = 0; k < J.npk[col]; k++) {
                const size_t pid = off[col] + (size_t)k; if (own[pid] <= -2) continue;
                const int32_t t = tid[pid]; double py, px; ps_frame_pt(&G, (double)C->jz, J.pk[col][k].r, th, &py, &px);
                const int x = (int)lround(px), y = (int)lround(py);
                uint32_t hsh = (uint32_t)t * 2654435761u; uint8_t c[3] = {(uint8_t)(64 + (hsh >> 24) % 192), (uint8_t)(64 + (hsh >> 16) % 192), (uint8_t)(64 + (hsh >> 8) % 192)};
                if (t < 0) { c[0] = 255; c[1] = 40; c[2] = 40; }   /* untraced peak: red */
                for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
                    const int yy = y + dy, xx = x + dx; if (yy < 0 || xx < 0 || yy >= H || xx >= W) continue;
                    memcpy(C->rgb + 3 * ((size_t)yy * W + xx), c, 3);
                }
            }
        }
    }
    for (size_t i = 0; i < npk; i++) C->owned += own[i] >= 0 || own[i] <= -2;
    C->npk += npk;
    free(X); free(Y); free(Z); free(lk);
    for (size_t c = 0; c < NC; c++) free(J.pk[c]);
    free(J.pk); free(J.npk); free(off); free(own); free(tid); free(pcell); free(sd); free(q); free(colk);
    return 0;
}

int cmd_sheet_trace(int argc, char **argv) {
    long long b[6];
    const char *probp = st_opt(argc, argv, "--prob", nullptr), *axp = st_opt(argc, argv, "--axis", nullptr), *out = st_opt(argc, argv, "--out", nullptr);
    const double axs = atof(st_opt(argc, argv, "--axis-scale", "1"));
    int nt = atoi(st_opt(argc, argv, "--threads", "0")); if (nt <= 0) nt = (int)sysconf(_SC_NPROCESSORS_ONLN); if (nt > 64) nt = 64;
    const int nreg = atoi(st_opt(argc, argv, "--regions", "1"));
    const char *bs = st_opt(argc, argv, "--box", nullptr);
    st_ctx C = {.step = atoi(st_opt(argc, argv, "--step", "2")), .thr = atoi(st_opt(argc, argv, "--thr", "64")), .sthr = atoi(st_opt(argc, argv, "--seed-thr", "128")),
                .minc = atoi(st_opt(argc, argv, "--min-cells", "100")), .tile = atoi(st_opt(argc, argv, "--tile", "0")), .mint = atoi(st_opt(argc, argv, "--min-tile", "32")), .nt = nt,
                .tol = atof(st_opt(argc, argv, "--tol", "2")), .amb = atof(st_opt(argc, argv, "--amb", "0")), .vs = atof(st_opt(argc, argv, "--vs", "9.596")),
                .dir = st_opt(argc, argv, "--dir", "auto"), .nms = atof(st_opt(argc, argv, "--nms", "5"))};
    if (!probp || !axp || !out || !bs || sscanf(bs, "%lld,%lld,%lld,%lld,%lld,%lld", &b[0], &b[1], &b[2], &b[3], &b[4], &b[5]) != 6 || axs <= 0 || C.step < 1 || C.tol <= 0 || nreg < 1 || nreg > 16 ||
        (strcmp(C.dir, "auto") && strcmp(C.dir, "radial") && strcmp(C.dir, "par"))) {
        fprintf(stderr, "usage: ufsm sheet-trace --prob P [--prob-key K] --box Z0,Y0,X0,NZ,NY,NX --axis A [--axis-scale S] --out DIR [--vs UM]\n"
                        "         [--step 2] [--thr 64] [--seed-thr 128] [--tol 2] [--amb 0] [--min-cells 100] [--tile 0] [--min-tile 32] [--dir auto|radial|par]\n"
                        "         [--regions 1] [--nms 5] [--extra CODES [--extra-key K] [--extra-p 128] [--extra-sep 3] [--extra-gap 1.5]] [--jpg QA.jpg] [--threads N] [--cache DIR]\n");
        return 2;
    }
    const int64_t o[3] = {b[0], b[1], b[2]}, n[3] = {b[3], b[4], b[5]};
    ps_geo G; if (ps_geo_init(&G, axp, axs, o, n)) { fprintf(stderr, "sheet-trace: cannot load axis %s\n", axp); return 1; }
    uint8_t *prob = ps_box(probp, st_opt(argc, argv, "--prob-key", nullptr), st_opt(argc, argv, "--cache", nullptr), o, n, nt); if (!prob) return 1;
    C.prob = prob; memcpy(C.n, n, sizeof C.n); C.G0 = &G;
    const char *exp_ = st_opt(argc, argv, "--extra", nullptr);   /* second sheet evidence: a code / mask box (sheet = 1..252), e.g. an old spiral raster */
    if (exp_) {
        C.extra = ps_box(exp_, st_opt(argc, argv, "--extra-key", nullptr), st_opt(argc, argv, "--cache", nullptr), o, n, nt); if (!C.extra) return 1;
        C.extra_p = atoi(st_opt(argc, argv, "--extra-p", "128")); C.extra_sep = atof(st_opt(argc, argv, "--extra-sep", "3"));
        C.extra_gap = atof(st_opt(argc, argv, "--extra-gap", "1.5"));
    }
    for (int d = 0; d < 3; d++) C.cn[d] = (n[d] + 1) / 2;
    const size_t NCELL = (size_t)(C.cn[0] * C.cn[1] * C.cn[2]); C.cell = malloc(NCELL * 4); if (!C.cell) return 1;
    for (size_t i = 0; i < NCELL; i++) C.cell[i] = -1;
    char pdir[1200]; snprintf(pdir, sizeof pdir, "%s/paths", out); C.pdir = pdir;
    if ((mkdir(out, 0755) && errno != EEXIST) || (mkdir(pdir, 0755) && errno != EEXIST)) { fprintf(stderr, "sheet-trace: cannot create %s\n", pdir); return 1; }
    /* the box frame (patch-solve --frame): picked over the whole box */
    {   ps_geo Gb = G; st_pick_frame(&C, &Gb, 0, 0, n[1], n[2], 1, "box:");
        char fp[1300]; snprintf(fp, sizeof fp, "%s/frame.txt", out); if (ps_frame_save(&Gb, fp)) return 1; }
    char tp[1300]; snprintf(tp, sizeof tp, "%s/tiles.tsv", out); C.tf = fopen(tp, "w"); if (!C.tf) return 1;
    fprintf(C.tf, "tile\tpatch\ttile_z\ttile_t\tcells\n");
    const char *jpg = st_opt(argc, argv, "--jpg", nullptr);
    if (jpg) {
        C.jz = (int)(((n[0] - 1) / C.step + 1) / 2) * C.step; const size_t WH = (size_t)n[1] * n[2]; C.rgb = malloc(WH * 3);
        for (size_t i = 0; i < WH; i++) C.rgb[3 * i] = C.rgb[3 * i + 1] = C.rgb[3 * i + 2] = (uint8_t)(prob[(size_t)C.jz * WH + i] / 2);
    }
    /* regions: an nreg x nreg xy grid traced in order of the strongest prob first (so the clearest sheets claim their cells) */
    for (int ry = 0; ry < nreg; ry++) for (int rx = 0; rx < nreg; rx++) {
        const int64_t y0 = n[1] * ry / nreg, y1 = n[1] * (ry + 1) / nreg, x0 = n[2] * rx / nreg, x1 = n[2] * (rx + 1) / nreg;
        if (st_region(&C, y0, x0, y1, x1, nreg == 1 || getenv("UFSM_ST_VERBOSE") != nullptr)) { fprintf(stderr, "sheet-trace: out of memory\n"); return 1; }
    }
    if (fclose(C.tf)) return 1;
    qsort(C.tl, C.ntl, sizeof *C.tl, st_linkcmp);
    char lp[1300]; snprintf(lp, sizeof lp, "%s/links.tsv", out); FILE *lf = fopen(lp, "w"); if (!lf) return 1;
    fprintf(lf, "a\tb\tcontacts\tkind\n"); long npair = 0; const char *kn[3] = {"seam", "contact", "rseam"};
    for (size_t i = 0, j; i < C.ntl; i = j) {
        for (j = i; j < C.ntl && C.tl[j].a == C.tl[i].a && C.tl[j].b == C.tl[i].b; j++);
        fprintf(lf, "st%06u\tst%06u\t%zu\t%s\n", C.tl[i].a / 4, C.tl[i].b, j - i, kn[C.tl[i].a % 4 < 3 ? C.tl[i].a % 4 : 0]); npair++;
    }
    if (fclose(lf)) return 1;
    {   const double tot = (double)(C.nstop[0] + C.nstop[1] + C.nstop[2] + C.nstop[3]) + 1e-9;
        printf("sheet-trace: steps %.0f: taken %.3f, no peak within tol %.3f, ambiguous %.3f, owned %.3f\n", tot, C.nstop[3] / tot, C.nstop[0] / tot, C.nstop[1] / tot, C.nstop[2] / tot); }
    if (nreg > 1) printf("sheet-trace: %d x %d regions, %d with parallel rays\n", nreg, nreg, C.nreg_par);
    printf("sheet-trace: %d patches grown, %d kept (>= %d cells) as %d tiles, holding %zu cells = %.3f of the peaks (%.3f owned incl. small), %ld linked tile pairs -> %s\n",
           C.npatch, C.nkept, C.minc, C.nout, C.kept, C.npk ? (double)C.kept / C.npk : 0, C.npk ? (double)C.owned / C.npk : 0, npair, out);
    if (C.rgb) {
        double ay, ax; ps_axis(&G, (double)C.jz, &ay, &ax); const int W = (int)n[2], H = (int)n[1];
        for (int d = -6; d <= 6; d++) {
            const int x = (int)lround(ax), y = (int)lround(ay);
            if (y >= 0 && y < H && x + d >= 0 && x + d < W) memset(C.rgb + 3 * ((size_t)y * W + x + d), 255, 3);
            if (x >= 0 && x < W && y + d >= 0 && y + d < H) memset(C.rgb + 3 * ((size_t)(y + d) * W + x), 255, 3);
        }
        if (jpeg_write_rgb(jpg, W, H, C.rgb, 90)) fprintf(stderr, "sheet-trace: cannot write %s\n", jpg);
        else printf("sheet-trace: QA image %s (z %d; red = untraced peaks)\n", jpg, C.jz);
        free(C.rgb);
    }
    free(prob); free((void *)C.extra); free(C.cell); free(C.tl);
    return 0;
}
