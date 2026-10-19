/* Box stitching: gauge alignment of overlapping local winding fits, the global gauge, the blended global winding field and
   level-set sheets traced across the boxes (src/stitch.h). */
#include "stitch.h"
#include "json.h"
#include "store.h"
#include "tiffw.h"
#include "z3w.h"
#include "zarr3.h"
#include <complex.h>
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

/* ---- boxes ---- */
int st_box_load(st_box *b, const char *dir) {
    memset(b, 0, sizeof *b);
    char p[1400]; snprintf(p, sizeof p, "%s/fit", dir);
    struct stat sb; const char *fd = stat(p, &sb) == 0 && S_ISDIR(sb.st_mode) ? p : dir;
    b->fs = fit_set_load(fd); if (!b->fs) return -1;
    char *rp = realpath(fd, nullptr); snprintf(b->dir, sizeof b->dir, "%s", rp ? rp : fd); free(rp);   /* absolute: gauge files are read from anywhere */
    char nm[1024]; snprintf(nm, sizeof nm, "%s", dir);
    size_t l = strlen(nm); while (l > 1 && nm[l - 1] == '/') nm[--l] = 0;
    const char *base = strrchr(nm, '/'); snprintf(b->name, sizeof b->name, "%s", base ? base + 1 : nm);
    const fit_model *m = b->fs->global; const int sh[3] = {m->D, m->H, m->W};
    for (int d = 0; d < 3; d++) { b->lo[d] = (double)m->origin[d]; b->hi[d] = b->lo[d] + sh[d] * m->h; }
    b->s = 1; b->g = 0; b->comp = -1;
    st_box_range(b);
    return 0;
}
void st_box_free(st_box *b) { fit_set_free(b->fs); b->fs = nullptr; }
double st_box_W(const st_box *b, const double p[3]) { return fit_set_W(b->fs, p[0] - b->lo[0], p[1] - b->lo[1], p[2] - b->lo[2]); }
void st_box_range(st_box *b) {
    b->wmin = 1e300; b->wmax = -1e300;
    for (int i = 0; i <= 8; i++) for (int j = 0; j <= 8; j++) for (int k = 0; k <= 8; k++) {
        const double p[3] = {b->lo[0] + (b->hi[0] - b->lo[0]) * (0.001 + 0.998 * i / 8), b->lo[1] + (b->hi[1] - b->lo[1]) * (0.001 + 0.998 * j / 8),
                             b->lo[2] + (b->hi[2] - b->lo[2]) * (0.001 + 0.998 * k / 8)};
        const double w = st_box_W(b, p); if (w < b->wmin) b->wmin = w; if (w > b->wmax) b->wmax = w;
    }
}
static double ramp(double t) { return t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t); }
double st_box_phi(const st_box *b, const double p[3]) {
    double w = 1;
    for (int d = 0; d < 3; d++) {
        if (p[d] < b->lo[d] || p[d] >= b->hi[d]) return 0;
        if (b->tlo[d] > 0) w *= ramp((p[d] - b->lo[d]) / b->tlo[d]);
        if (b->thi[d] > 0) w *= ramp((b->hi[d] - p[d]) / b->thi[d]);
    }
    return w;
}
/* a face of box i ramps over its overlap with a face neighbour j of the same component (j reaches past the face, and covers at
   least half of i's extent on both other axes); faces without one are outer faces (hard: the box's weight stays 1 up to them) */
void st_tapers(st_box *b, int nb, int by_comp) {
    for (int i = 0; i < nb; i++) {
        for (int d = 0; d < 3; d++) b[i].tlo[d] = b[i].thi[d] = 0;
        if (b[i].comp < 0) continue;
        for (int j = 0; j < nb; j++) {
            if (j == i || b[j].comp < 0 || (by_comp && b[j].comp != b[i].comp)) continue;
            for (int d = 0; d < 3; d++) {
                int face = 1;
                for (int e = 0; e < 3; e++) {
                    if (e == d) continue;
                    const double ov = fmin(b[i].hi[e], b[j].hi[e]) - fmax(b[i].lo[e], b[j].lo[e]);
                    if (ov < 0.5 * (b[i].hi[e] - b[i].lo[e])) face = 0;
                }
                if (!face) continue;
                const double half = 0.5 * (b[i].hi[d] - b[i].lo[d]);
                if (b[j].lo[d] < b[i].lo[d] && b[j].hi[d] > b[i].lo[d] && b[j].hi[d] < b[i].hi[d]) b[i].tlo[d] = fmin(half, fmax(b[i].tlo[d], b[j].hi[d] - b[i].lo[d]));
                if (b[j].hi[d] > b[i].hi[d] && b[j].lo[d] < b[i].hi[d] && b[j].lo[d] > b[i].lo[d]) b[i].thi[d] = fmin(half, fmax(b[i].thi[d], b[i].hi[d] - b[j].lo[d]));
            }
        }
    }
}
int st_G(const st_box *b, int nb, int comp, const double p[3], double *G) {
    double sw = 0, sg = 0; int n = 0;
    for (int i = 0; i < nb; i++) {
        if (comp >= 0 && b[i].comp != comp) continue;
        const double w = st_box_phi(&b[i], p); if (w <= 0) continue;
        sg += w * (b[i].s * st_box_W(&b[i], p) + b[i].g); sw += w; n++;
    }
    if (sw <= 1e-12) return 0;
    *G = sg / sw; return n;
}

/* ---- pairwise alignment ---- */
enum { ST_HB = 2048, ST_BPT = 16, ST_NB = 2 * ST_HB * ST_BPT };   /* offset histograms: 1/16 turn bins over [-2048, 2048) turns */
static int hbin(double d) { const double q = (d + ST_HB) * ST_BPT; return q < 0 || q >= ST_NB ? -1 : (int)q; }
typedef struct {
    const st_box *A, *B; const st_fields *F; int stride; double min_w;
    const uint8_t *w, *pc, *ps; int64_t o[3], n[3];   /* the block (cells) */
    int64_t c0[3];                                    /* the sampling lattice: cells c0 + j stride */
    double *hist; double wsum; long long ns; int tid, nt;
} al_job;
static void *al_worker(void *arg) {
    al_job *J = arg; const double h = J->F->h;
    for (int64_t z = 0; z < J->n[0]; z++) {
        const int64_t cz = J->o[0] + z; if ((cz - J->c0[0]) % J->stride) continue;
        int64_t row = 0;
        for (int64_t y = 0; y < J->n[1]; y++) {
            const int64_t cy = J->o[1] + y; if ((cy - J->c0[1]) % J->stride) continue;
            if (row++ % J->nt != J->tid) continue;
            for (int64_t x = 0; x < J->n[2]; x++) {
                const int64_t cx = J->o[2] + x; if ((cx - J->c0[2]) % J->stride) continue;
                const size_t i = ((size_t)z * J->n[1] + y) * J->n[2] + x;
                if (!J->w[i]) continue;
                const double conf = fmin(1.0, hypot(J->pc[i] - 128.0, J->ps[i] - 128.0) / 127.0), wt = J->w[i] / 255.0 * conf;
                if (wt < J->min_w) continue;
                const double p[3] = {J->F->org[0] + (cz + 0.5) * h, J->F->org[1] + (cy + 0.5) * h, J->F->org[2] + (cx + 0.5) * h};
                int in = 1; for (int d = 0; d < 3; d++) if (p[d] < J->A->lo[d] || p[d] >= J->A->hi[d] || p[d] < J->B->lo[d] || p[d] >= J->B->hi[d]) in = 0;
                if (!in) continue;
                const double wa = st_box_W(J->A, p), wb = st_box_W(J->B, p);
                const int b0 = hbin(wb - wa), b1 = hbin(wb + wa);
                if (b0 >= 0) J->hist[b0] += wt;
                if (b1 >= 0) J->hist[ST_NB + b1] += wt;
                J->wsum += wt; J->ns++;
            }
        }
    }
    return nullptr;
}
static void al_modes(const double *hi, double tot, int *kb, double *best, int *k2, double *second, double *tight, double *frac, double *local) {
    *best = -1; *second = 0; *kb = 0; *k2 = 0;
    for (int k = -ST_HB + 1; k < ST_HB; k++) {
        const int b0 = (k + ST_HB) * ST_BPT - ST_BPT / 2; double s = 0;
        for (int q = 0; q < ST_BPT; q++) s += hi[b0 + q];
        if (s > *best) { *second = *best; *k2 = *kb; *best = s; *kb = k; }
        else if (s > *second) { *second = s; *k2 = k; }
    }
    if (*second < 0) *second = 0;
    const int bc = (*kb + ST_HB) * ST_BPT; double t = 0;
    for (int q = -ST_BPT / 4; q < ST_BPT / 4; q++) t += hi[bc + q];
    double complex m = 0; for (int b = 0; b < ST_NB; b++) if (hi[b] > 0) m += hi[b] * cexp(I * 2 * M_PI * ((b + 0.5) / ST_BPT - ST_HB));
    *frac = cabs(m) > 0 ? carg(m) / (2 * M_PI) : 0;
    double lc = 0; for (int b = 0; b < ST_NB; b++) { const int q = b % ST_BPT; if (q < ST_BPT / 4 || q >= ST_BPT - ST_BPT / 4) lc += hi[b]; }   /* within 1/4 of an integer */
    *best /= tot; *second /= tot; *tight = t / tot; *local = lc / tot;
}
int st_align_pair(const st_box *b, int ia, int ib, const st_fields *F, int stride, double min_w, int nthreads, st_edge *e) {
    const st_box *A = &b[ia], *B = &b[ib];
    memset(e, 0, sizeof *e); e->a = ia; e->b = ib; e->s = 1;
    double lo[3], hi[3]; e->vol = 1;
    for (int d = 0; d < 3; d++) { lo[d] = fmax(A->lo[d], B->lo[d]); hi[d] = fmin(A->hi[d], B->hi[d]); if (hi[d] <= lo[d]) return 1; e->vol *= hi[d] - lo[d]; }
    int64_t c0[3], c1[3];
    for (int d = 0; d < 3; d++) { c0[d] = (int64_t)floor((lo[d] - F->org[d]) / F->h); c1[d] = (int64_t)ceil((hi[d] - F->org[d]) / F->h); if (c0[d] < 0) c0[d] = 0; }
    if (stride < 1) stride = 1;
    if (nthreads < 1) nthreads = 1;
    if (nthreads > 64) nthreads = 64;
    double *hist = calloc((size_t)nthreads * 2 * ST_NB, sizeof(double));
    const int64_t ny = c1[1] - c0[1], nx = c1[2] - c0[2];
    uint8_t *buf = nullptr; size_t cap = 0; int rc = 0; double wsum = 0; long long ns = 0;
    for (int64_t z0 = c0[0]; z0 < c1[0] && !rc;) {
        const int64_t z1 = fmin((double)c1[0], (double)((z0 / 128 + 1) * 128));   /* blocks on the 128-cell chunk grid */
        const int64_t o[3] = {z0, c0[1], c0[2]}, n[3] = {z1 - z0, ny, nx}; const size_t nv = (size_t)n[0] * ny * nx;
        if (nv * 3 > cap) { free(buf); cap = nv * 3; buf = malloc(cap); if (!buf) { rc = -1; break; } }
        if (F->read(F->ctx, o, n, buf, buf + nv, buf + 2 * nv)) { rc = -1; break; }
        al_job J[64]; pthread_t th[64];
        for (int t = 0; t < nthreads; t++) {
            J[t] = (al_job){A, B, F, stride, min_w, buf, buf + nv, buf + 2 * nv, {o[0], o[1], o[2]}, {n[0], n[1], n[2]}, {c0[0], c0[1], c0[2]},
                            hist + (size_t)t * 2 * ST_NB, 0, 0, t, nthreads};
            pthread_create(&th[t], nullptr, al_worker, &J[t]);
        }
        for (int t = 0; t < nthreads; t++) { pthread_join(th[t], nullptr); wsum += J[t].wsum; ns += J[t].ns; }
        z0 = z1;
    }
    free(buf);
    if (rc) { free(hist); return -1; }
    for (int t = 1; t < nthreads; t++) for (size_t i = 0; i < 2 * (size_t)ST_NB; i++) hist[i] += hist[(size_t)t * 2 * ST_NB + i];
    e->wsum = wsum; e->n = ns;
    if (wsum <= 0) { free(hist); return 1; }
    int kb[2], k2[2]; double best[2], second[2], tight[2], frac[2], local[2];
    for (int q = 0; q < 2; q++) al_modes(hist + (size_t)q * ST_NB, wsum, &kb[q], &best[q], &k2[q], &second[q], &tight[q], &frac[q], &local[q]);
    const int q = best[1] > best[0] ? 1 : 0;   /* q = 1: W_b + W_a ~ k, i.e. W_b ~ -W_a + k */
    e->s = q ? -1 : 1; e->k = kb[q]; e->agree = best[q]; e->tight = tight[q]; e->k2 = k2[q]; e->agree2 = second[q]; e->frac = frac[q]; e->other = best[1 - q]; e->local = local[q];
    free(hist);
    return 0;
}

/* ---- the global gauge ---- */
static int uf_find(int *p, int i) { while (p[i] != i) { p[i] = p[p[i]]; i = p[i]; } return i; }
static const st_edge *g_sort_e;
static int cmp_edge(const void *x, const void *y) {
    const st_edge *a = &g_sort_e[*(const int *)x], *b = &g_sort_e[*(const int *)y];
    if (a->agree != b->agree) return a->agree < b->agree ? 1 : -1;
    return (a->wsum < b->wsum) - (a->wsum > b->wsum);
}
int st_solve(st_box *b, int nb, st_edge *e, int ne, double min_agree, double min_wsum, const uint8_t *excl) {
    int *ord = malloc((ne + 1) * sizeof(int)), nu = 0, *par = malloc(nb * sizeof(int));
    for (int i = 0; i < ne; i++) { e[i].tree = 0; e[i].resid = 0; e[i].used = e[i].agree >= min_agree && e[i].wsum >= min_wsum && !(excl && (excl[e[i].a] || excl[e[i].b])); if (e[i].used) ord[nu++] = i; }
    g_sort_e = e; qsort(ord, nu, sizeof(int), cmp_edge);
    for (int i = 0; i < nb; i++) par[i] = i;
    for (int q = 0; q < nu; q++) { const int a = uf_find(par, e[ord[q]].a), c = uf_find(par, e[ord[q]].b); if (a != c) { par[a] = c; e[ord[q]].tree = 1; } }
    /* breadth first over the tree from the box with the most used edges of each component */
    int *deg = calloc(nb, sizeof(int)), *seen = calloc(nb, sizeof(int)), *queue = malloc(nb * sizeof(int));
    for (int i = 0; i < ne; i++) if (e[i].used) { deg[e[i].a]++; deg[e[i].b]++; }
    for (int i = 0; i < nb; i++) b[i].comp = -1;
    int nc = 0;
    for (;;) {
        int r = -1; for (int i = 0; i < nb; i++) if (!seen[i] && (r < 0 || deg[i] > deg[r])) r = i;
        if (r < 0) break;
        int qh = 0, qt = 0; queue[qt++] = r; seen[r] = 1; b[r].s = 1; b[r].g = 0; b[r].comp = nc;
        while (qh < qt) {
            const int a = queue[qh++];
            for (int i = 0; i < ne; i++) {
                if (!e[i].tree || (e[i].a != a && e[i].b != a)) continue;
                const int c = e[i].a == a ? e[i].b : e[i].a; if (seen[c]) continue;
                if (e[i].a == a) { b[c].s = b[a].s * e[i].s; b[c].g = b[a].g - b[c].s * e[i].k; }   /* W_c ~ s W_a + k */
                else { b[c].s = b[a].s * e[i].s; b[c].g = b[a].g + b[c].s * e[i].s * e[i].k; }       /* W_a ~ s W_c + k: W_c ~ s W_a - s k */
                b[c].comp = nc; seen[c] = 1; queue[qt++] = c;
            }
        }
        nc++;
    }
    int bad = 0;
    for (int i = 0; i < ne; i++) {
        if (!e[i].used || e[i].tree) continue;
        const st_box *A = &b[e[i].a], *B = &b[e[i].b];
        if (A->s * B->s != e[i].s) { e[i].resid = 9999; bad++; continue; }
        e[i].resid = e[i].k - (int)lround(B->s * (A->g - B->g));
        if (e[i].resid) bad++;
    }
    /* components numbered by size; each component's G starts at >= 1 */
    int *size = calloc(nc, sizeof(int)), *rank = malloc(nc * sizeof(int));
    for (int i = 0; i < nb; i++) size[b[i].comp]++;
    for (int c = 0; c < nc; c++) { int r = 0; for (int d = 0; d < nc; d++) if (size[d] > size[c] || (size[d] == size[c] && d < c)) r++; rank[c] = r; }
    for (int i = 0; i < nb; i++) b[i].comp = rank[b[i].comp];
    for (int c = 0; c < nc; c++) {
        double gmin = 1e300;
        for (int i = 0; i < nb; i++) if (b[i].comp == c) { const double m = (b[i].s > 0 ? b[i].wmin : -b[i].wmax) + b[i].g; if (m < gmin) gmin = m; }
        const double sh = 1 - floor(gmin);
        for (int i = 0; i < nb; i++) if (b[i].comp == c) b[i].g += sh;
    }
    free(ord); free(par); free(deg); free(seen); free(queue); free(size); free(rank);
    return bad;
}

/* ---- gauge files ---- */
int st_save(const char *path, const st_box *b, int nb, const st_edge *e, int ne, double um) {
    FILE *f = fopen(path, "w"); if (!f) return -1;
    fprintf(f, "{\"um\": %.9g,\n \"boxes\": [\n", um);
    for (int i = 0; i < nb; i++)
        fprintf(f, "  {\"name\": \"%s\", \"dir\": \"%s\", \"lo\": [%.3f, %.3f, %.3f], \"hi\": [%.3f, %.3f, %.3f], \"s\": %d, \"g\": %.0f, \"comp\": %d, \"wmin\": %.4f, \"wmax\": %.4f}%s\n",
                b[i].name, b[i].dir, b[i].lo[0], b[i].lo[1], b[i].lo[2], b[i].hi[0], b[i].hi[1], b[i].hi[2], b[i].s, b[i].g, b[i].comp, b[i].wmin, b[i].wmax, i + 1 < nb ? "," : "");
    fprintf(f, " ],\n \"edges\": [\n");
    for (int i = 0; i < ne; i++)
        fprintf(f, "  {\"a\": %d, \"b\": %d, \"s\": %d, \"k\": %d, \"agree\": %.5f, \"tight\": %.5f, \"k2\": %d, \"agree2\": %.5f, \"other\": %.5f, \"frac\": %.4f, "
                   "\"local\": %.5f, \"wsum\": %.3f, \"n\": %lld, \"vol\": %.6g, \"tree\": %d, \"resid\": %d, \"used\": %d}%s\n",
                e[i].a, e[i].b, e[i].s, e[i].k, e[i].agree, e[i].tight, e[i].k2, e[i].agree2, e[i].other, e[i].frac, e[i].local, e[i].wsum, e[i].n, e[i].vol, e[i].tree, e[i].resid,
                e[i].used, i + 1 < ne ? "," : "");
    fprintf(f, " ]}\n");
    return fclose(f) ? -1 : 0;
}
int st_load(const char *path, st_box **bo, int *nbo, st_edge **eo, int *neo, double *um) {
    FILE *f = fopen(path, "rb"); if (!f) return -1;
    fseek(f, 0, SEEK_END); const size_t n = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
    char *t = malloc(n + 1); if (fread(t, 1, n, f) != n) { fclose(f); free(t); return -1; } fclose(f); t[n] = 0;
    json *j = json_parse(t, n); free(t); if (!j) return -1;
    const json *jb = json_get(j, "boxes"), *je = json_get(j, "edges");
    const int nb = jb ? (int)jb->n : 0, ne = je ? (int)je->n : 0;
    st_box *b = calloc(nb + 1, sizeof *b); st_edge *e = calloc(ne + 1, sizeof *e);
    if (um) *um = json_num(json_get(j, "um"), 2.4);
    for (int i = 0; i < nb; i++) {
        const json *x = json_at(jb, i);
        if (st_box_load(&b[i], json_str(json_get(x, "dir"), ""))) { fprintf(stderr, "stitch: cannot load the model in %s\n", json_str(json_get(x, "dir"), "")); json_free(j); free(b); free(e); return -1; }
        snprintf(b[i].name, sizeof b[i].name, "%s", json_str(json_get(x, "name"), b[i].name));
        b[i].s = (int)json_num(json_get(x, "s"), 1); b[i].g = json_num(json_get(x, "g"), 0); b[i].comp = (int)json_num(json_get(x, "comp"), -1);
    }
    for (int i = 0; i < ne; i++) {
        const json *x = json_at(je, i);
        e[i] = (st_edge){(int)json_num(json_get(x, "a"), 0), (int)json_num(json_get(x, "b"), 0), (int)json_num(json_get(x, "s"), 1), (int)json_num(json_get(x, "k"), 0),
                         json_num(json_get(x, "agree"), 0), json_num(json_get(x, "tight"), 0), (int)json_num(json_get(x, "k2"), 0), json_num(json_get(x, "agree2"), 0),
                         json_num(json_get(x, "other"), 0), json_num(json_get(x, "frac"), 0), json_num(json_get(x, "local"), 0), json_num(json_get(x, "wsum"), 0), (long long)json_num(json_get(x, "n"), 0),
                         json_num(json_get(x, "vol"), 0), (int)json_num(json_get(x, "tree"), 0), (int)json_num(json_get(x, "resid"), 0), (int)json_num(json_get(x, "used"), 0)};
    }
    json_free(j);
    st_tapers(b, nb, 1);
    *bo = b; *nbo = nb; *eo = e; *neo = ne;
    return 0;
}

/* ---- level-set tracing ---- */
st_sgauge st_sgauge_new(int nb) { return (st_sgauge){nb, calloc(nb, 1), calloc(nb, 1), calloc(nb, sizeof(double)), 0, 0}; }
void st_sgauge_free(st_sgauge *g) { free(g->on); free(g->s); free(g->off); memset(g, 0, sizeof *g); }
void st_sgauge_global(st_sgauge *g, const st_box *b, int comp, double k) {
    for (int i = 0; i < g->nb; i++) { g->on[i] = comp < 0 || b[i].comp == comp; g->s[i] = (int8_t)b[i].s; g->off[i] = b[i].s * (k - b[i].g); }
    g->contacts = g->contact_conflicts = 0;
}
void st_sgauge_seed(st_sgauge *g, int i, double k) { memset(g->on, 0, g->nb); g->on[i] = 1; g->s[i] = 1; g->off[i] = k; g->contacts = g->contact_conflicts = 0; }
int st_F(const st_box *b, const st_sgauge *g, const double p[3], double *F) {
    double sw = 0, sf = 0;
    for (int i = 0; i < g->nb; i++) {
        if (!g->on[i]) continue;
        const double w = st_box_phi(&b[i], p); if (w <= 0) continue;
        sf += w * g->s[i] * (st_box_W(&b[i], p) - g->off[i]); sw += w;
    }
    if (sw <= 1e-12) return 0;
    *F = sf / sw; return 1;
}
typedef struct { const st_box *b; int nb, comp, follow; const st_trace_cfg *c; st_sgauge *g; double gmin2; } tctx;
static int tF(const tctx *T, double z, double y, double x, double *F) { const double p[3] = {z, y, x}; return st_F(T->b, T->g, p, F); }
static int tFgrad(const tctx *T, double z, double y, double x, double *F, double *gy, double *gx) {
    double a, b, c, d; const double e = 0.5;
    if (!tF(T, z, y, x, F) || !tF(T, z, y + e, x, &a) || !tF(T, z, y - e, x, &b) || !tF(T, z, y, x + e, &c) || !tF(T, z, y, x - e, &d)) return 0;
    *gy = (a - b) / (2 * e); *gx = (c - d) / (2 * e); return 1;
}
static int mask_ok(const st_trace_cfg *c, double z, double y, double x) {
    if (!c || !c->mask) return 1;
    const double p[3] = {z, y, x}; int64_t q[3];
    for (int d = 0; d < 3; d++) { q[d] = (int64_t)floor((p[d] - c->mo[d]) / c->mh); if (q[d] < 0 || q[d] >= c->mn[d]) return 0; }
    return c->mask[((size_t)q[0] * c->mn[1] + q[1]) * c->mn[2] + q[2]] != 0;
}
/* Newton onto F = 0 in the plane z from (y, x) (steps of at most 0.4 turn, at most one sheet spacing in total); 0 when it fails */
static int project(const tctx *T, double z, double *y, double *x, double *gyo, double *gxo) {
    double qy = *y, qx = *x, moved = 0;
    for (int it = 0; it < 12; it++) {
        double F, gy, gx; if (!tFgrad(T, z, qy, qx, &F, &gy, &gx)) return 0;
        const double g2 = gy * gy + gx * gx; if (g2 < T->gmin2) return 0;
        if (it == 0 && fabs(F) > 0.5) return 0;
        if (fabs(F) < 1e-3) { *y = qy; *x = qx; *gyo = gy; *gxo = gx; return 1; }
        double sy = -F * gy / g2, sx = -F * gx / g2; const double sl = hypot(sy, sx), cap = 0.4 / sqrt(g2);
        if (sl > cap) { sy *= cap / sl; sx *= cap / sl; }
        qy += sy; qx += sx; moved += fmin(sl, cap);
        if (moved > 1.0 / sqrt(g2)) return 0;
    }
    return 0;
}
/* follow mode: a box not yet in the sheet joins at an accepted vertex where it has weight, at its level nearest the vertex
   (off = round W), once the vertex sits within 1/4 turn of that level or the box's weight reaches 1/2 (a conflicting contact),
   with the sign that makes its gradient agree with the sheet's */
static void contact(const tctx *T, double z, double y, double x, double gy, double gx) {
    if (!T->follow) return;
    const double p[3] = {z, y, x};
    for (int j = 0; j < T->nb; j++) {
        if (T->g->on[j] || (T->comp >= 0 && T->b[j].comp != T->comp)) continue;
        const double w = st_box_phi(&T->b[j], p); if (w <= 0) continue;
        const double Wj = st_box_W(&T->b[j], p), r = Wj - round(Wj);
        if (fabs(r) > 0.25 && w < 0.5) continue;
        const double py[3] = {z, y + 0.5, x}, my[3] = {z, y - 0.5, x}, px[3] = {z, y, x + 0.5}, mx[3] = {z, y, x - 0.5};
        const double dy = st_box_W(&T->b[j], py) - st_box_W(&T->b[j], my), dx = st_box_W(&T->b[j], px) - st_box_W(&T->b[j], mx);
        T->g->on[j] = 1; T->g->s[j] = dy * gy + dx * gx >= 0 ? 1 : -1; T->g->off[j] = round(Wj);
        T->g->contacts++; if (fabs(r) > 0.25) T->g->contact_conflicts++;
    }
}
/* along the contour from the projected point (y, x) (gradient gy, gx): column direction sgn (+1: (gx, -gy), the wraps' increasing
   theta when F grows outward), up to maxn points every tstep; stops at a failed projection, a step off its length, a turn of more
   than 60 degrees per step, closing on the start, or more than max_gap unmasked steps; trailing unmasked points are dropped */
static int trace_dir(const tctx *T, double z, double y, double x, double gy, double gx, int sgn, float *oy, float *ox, float *og, uint8_t *ook, int maxn) {
    double ty = sgn * gx, tx = -sgn * gy; const double tl = hypot(ty, tx); if (tl <= 0) return 0;
    ty /= tl; tx /= tl;
    const double y0 = y, x0 = x, st = T->c->tstep; int n = 0, gap = 0;
    while (n < maxn) {
        double ny = y + st * ty, nx = x + st * tx, ngy, ngx;
        if (!project(T, z, &ny, &nx, &ngy, &ngx)) break;
        const double mv = hypot(ny - y, nx - x); if (mv < 0.3 * st || mv > 2 * st) break;
        double uy = sgn * ngx, ux = -sgn * ngy; const double ul = hypot(uy, ux); uy /= ul; ux /= ul;
        if (uy * ty + ux * tx < 0.5) break;
        if (n > 8 && hypot(ny - y0, nx - x0) < 0.75 * st) break;
        const int ok = mask_ok(T->c, z, ny, nx); gap = ok ? 0 : gap + 1;
        if (gap > T->c->max_gap) break;
        contact(T, z, ny, nx, ngy, ngx);
        oy[n] = (float)ny; ox[n] = (float)nx; og[n] = (float)hypot(ngy, ngx); ook[n] = (uint8_t)ok; n++;
        y = ny; x = nx; ty = uy; tx = ux;
    }
    while (n > 0 && !ook[n - 1]) n--;
    return n;
}
typedef struct { int c0, n, ca; float *y, *x, *g; uint8_t *ok; int nok; } trow;
static void trow_free(trow *r) { free(r->y); free(r->x); free(r->g); free(r->ok); memset(r, 0, sizeof *r); }
/* a whole row from its anchor (column ca at (y, x), already on the contour) */
static int trace_row(const tctx *T, double z, double y, double x, double gy, double gx, int ca, trow *R, float *tmp) {
    const int M = T->c->max_cols; float *ly = tmp, *lx = tmp + M, *lg = tmp + 2 * M, *ry = tmp + 3 * M, *rx = tmp + 4 * M, *rg = tmp + 5 * M;
    uint8_t *lo = (uint8_t *)(tmp + 6 * M), *ro = lo + M;
    contact(T, z, y, x, gy, gx);
    const int nr = trace_dir(T, z, y, x, gy, gx, +1, ry, rx, rg, ro, M), nl = trace_dir(T, z, y, x, gy, gx, -1, ly, lx, lg, lo, M);
    R->n = nl + 1 + nr; R->c0 = ca - nl; R->ca = ca;
    R->y = malloc(R->n * 4); R->x = malloc(R->n * 4); R->g = malloc(R->n * 4); R->ok = malloc(R->n);
    for (int i = 0; i < nl; i++) { R->y[nl - 1 - i] = ly[i]; R->x[nl - 1 - i] = lx[i]; R->g[nl - 1 - i] = lg[i]; R->ok[nl - 1 - i] = lo[i]; }
    R->y[nl] = (float)y; R->x[nl] = (float)x; R->g[nl] = (float)hypot(gy, gx); R->ok[nl] = (uint8_t)mask_ok(T->c, z, y, x);
    for (int i = 0; i < nr; i++) { R->y[nl + 1 + i] = ry[i]; R->x[nl + 1 + i] = rx[i]; R->g[nl + 1 + i] = rg[i]; R->ok[nl + 1 + i] = ro[i]; }
    R->nok = 0; for (int i = 0; i < R->n; i++) R->nok += R->ok[i];
    return R->nok;
}
/* the next row (plane z) from the previous one: its vertices projected into the new plane, nearest the previous anchor column
   first (every column within 16, then every 8th), the first that lands on the contour anchors the new row at its column */
static int next_row(const tctx *T, double z, const trow *P, trow *R, float *tmp) {
    const int a = P->ca - P->c0;
    for (int d = 0, tries = 0; d < P->n && tries < 256; d += d < 16 ? 1 : 8) {
        for (int sgn = 0; sgn < 2; sgn++) {
            if (d == 0 && sgn) continue;
            const int i = sgn ? a - d : a + d; if (i < 0 || i >= P->n || !P->ok[i]) continue;
            tries++;
            double y = P->y[i], x = P->x[i], gy, gx;
            if (!project(T, z, &y, &x, &gy, &gx)) continue;
            if (hypot(y - P->y[i], x - P->x[i]) > 4 * T->c->tstep) continue;   /* the sheet moved too far between two rows */
            if (trace_row(T, z, y, x, gy, gx, P->c0 + i, R, tmp) > 0) return 1;
            trow_free(R);
        }
    }
    return 0;
}
void st_sheet_free(st_sheet *s) { free(s->x); free(s->y); free(s->z); free(s->seam); free(s->spreads); free(s->own); memset(s, 0, sizeof *s); }
static int cmp_float(const void *a, const void *b) { const float x = *(const float *)a, y = *(const float *)b; return (x > y) - (x < y); }
int st_trace(const st_box *b, int nb, int comp, const st_trace_cfg *c, st_sgauge *g, int follow, const double seed[3], st_sheet *s) {
    memset(s, 0, sizeof *s);
    tctx T = {b, nb, comp, follow, c, g, 1.0 / (400.0 * 400.0)};   /* F flatter than one turn per 400 native voxels: degenerate */
    double zlo = 1e300, zhi = -1e300;
    for (int i = 0; i < nb; i++) if (comp < 0 || b[i].comp == comp) { zlo = fmin(zlo, b[i].lo[0]); zhi = fmax(zhi, b[i].hi[0]); }
    if (zlo >= zhi) return -1;
    const double st = c->tstep;
    const int ndn = (int)floor((seed[0] - zlo) / st), nup = (int)floor((zhi - 1e-6 - seed[0]) / st), nr = ndn + nup + 1;
    trow *R = calloc(nr, sizeof *R); float *tmp = malloc((size_t)c->max_cols * 8 * 4);
    double y = seed[1], x = seed[2], gy, gx;
    if (!project(&T, seed[0], &y, &x, &gy, &gx) || trace_row(&T, seed[0], y, x, gy, gx, 0, &R[ndn], tmp) <= 0) { trow_free(&R[ndn]); free(R); free(tmp); return -1; }
    int rlo = ndn, rhi = ndn;
    for (int r = ndn + 1; r < nr; r++) { if (!next_row(&T, seed[0] + (r - ndn) * st, &R[r - 1], &R[r], tmp)) break; rhi = r; }
    for (int r = ndn - 1; r >= 0; r--) { if (!next_row(&T, seed[0] + (r - ndn) * st, &R[r + 1], &R[r], tmp)) break; rlo = r; }
    free(tmp);
    int cmin = 1 << 30, cmax = -(1 << 30);
    for (int r = rlo; r <= rhi; r++) { if (R[r].c0 < cmin) cmin = R[r].c0; if (R[r].c0 + R[r].n > cmax) cmax = R[r].c0 + R[r].n; }
    s->rows = rhi - rlo + 1; s->cols = cmax - cmin; s->z0 = seed[0] + (rlo - ndn) * st;
    const size_t nv = (size_t)s->rows * s->cols;
    s->x = malloc(nv * 4); s->y = malloc(nv * 4); s->z = malloc(nv * 4); s->seam = calloc(nv, 1); s->own = calloc(nb, sizeof(long long));
    for (size_t i = 0; i < nv; i++) s->x[i] = s->y[i] = s->z[i] = -1;
    size_t cap = 1 << 16; s->spreads = malloc(cap * 4);
    for (int r = rlo; r <= rhi; r++) {
        const trow *W = &R[r]; const double z = seed[0] + (r - ndn) * st;
        for (int i = 0; i < W->n; i++) {
            if (!W->ok[i]) continue;
            const size_t v = (size_t)(r - rlo) * s->cols + (W->c0 + i - cmin);
            s->x[v] = W->x[i]; s->y[v] = W->y[i]; s->z[v] = (float)z; s->nvalid++;
            /* seam class: the joined boxes' own level sets near the vertex (first order: s (W - off) / |grad F|) */
            const double p[3] = {z, W->y[i], W->x[i]}; double emin = 1e300, emax = -1e300, wbest = 0; int nbx = 0, ib = -1, unj = 0;
            for (int j = 0; j < nb; j++) {
                if (comp >= 0 && b[j].comp != comp) continue;
                const double w = st_box_phi(&b[j], p); if (w <= 0.02) continue;
                if (!g->on[j]) { if (w >= 0.5) unj = 1; continue; }
                const double ev = g->s[j] * (st_box_W(&b[j], p) - g->off[j]); nbx++;
                if (ev < emin) emin = ev;
                if (ev > emax) emax = ev;
                if (w > wbest) { wbest = w; ib = j; }
            }
            if (ib >= 0) s->own[ib]++;
            s->nunjoined += unj;
            if (nbx < 2) { s->seam[v] = 1; continue; }
            const double dw = emax - emin; s->nseam++;
            s->seam[v] = dw < 0.25 ? 2 : dw < 0.5 ? 3 : 4;
            if (dw >= 0.25 && dw < 0.5) s->nspread++;
            if (dw >= 0.5) s->nconflict++;
            if ((size_t)s->nspreads == cap) { cap *= 2; s->spreads = realloc(s->spreads, cap * 4); }
            const float sp = (float)(dw / fmax(W->g[i], 1e-6)); s->spreads[s->nspreads++] = sp; s->spread_sum += sp;
        }
    }
    for (int j = 0; j < nb; j++) s->nboxes += s->own[j] > 0;
    if (s->nspreads) qsort(s->spreads, s->nspreads, 4, cmp_float);
    for (int r = 0; r < nr; r++) trow_free(&R[r]);
    free(R);
    return s->nvalid ? 0 : -1;
}
int st_seed(const st_box *b, int i, double zf, const st_sgauge *g, const st_trace_cfg *c, double p[3]) {
    const st_box *B = &b[i]; const fit_model *m = B->fs->global; const int na = (int)(m->D * m->h) + 1;
    const double z = B->lo[0] + zf * (B->hi[0] - B->lo[0]); int zi = (int)(z - B->lo[0]); if (zi < 0) zi = 0; if (zi >= na) zi = na - 1;
    const double ay = m->cy[zi] + B->lo[1], ax = m->cx[zi] + B->lo[2], cy = 0.5 * (B->lo[1] + B->hi[1]), cx = 0.5 * (B->lo[2] + B->hi[2]);
    double uy = cy - ay, ux = cx - ax; const double ul = hypot(uy, ux); if (ul < 1e-9) { uy = 1; ux = 0; } else { uy /= ul; ux /= ul; }
    const double L = 0.5 * hypot(B->hi[1] - B->lo[1], B->hi[2] - B->lo[2]);
    double best = 1e300; double tp = -L, fp = 0; int hp = 0;
    for (double t = -L; t <= L; t += 1.0) {
        const double q[3] = {z, cy + t * uy, cx + t * ux}; double F;
        const int ok = st_box_phi(B, q) > 0 && st_F(b, g, q, &F);
        if (ok && hp && (fp < 0) != (F < 0) && fabs(F - fp) < 0.5) {
            double a = tp, fa = fp, bb = t;
            for (int it = 0; it < 30; it++) {
                const double mid = 0.5 * (a + bb), qm[3] = {z, cy + mid * uy, cx + mid * ux}; double Fm;
                if (!st_F(b, g, qm, &Fm)) break;
                if ((Fm < 0) == (fa < 0)) { a = mid; fa = Fm; } else bb = mid;
            }
            const double tm = 0.5 * (a + bb);
            if (fabs(tm) < fabs(best) && mask_ok(c, z, cy + tm * uy, cx + tm * ux)) best = tm;
        }
        hp = ok; tp = t; fp = ok ? F : 0;
    }
    if (best > 1e299) return -1;
    p[0] = z; p[1] = cy + best * uy; p[2] = cx + best * ux;
    return 0;
}

/* ---- ufsm stitch ---- */
static const char *sopt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return dflt;
}
typedef struct { z3 *w, *pc, *ps; int nthreads; } zf_ctx;
static int zf_read(void *ctx, const int64_t o[3], const int64_t n[3], uint8_t *w, uint8_t *pc, uint8_t *ps) {
    zf_ctx *c = ctx;
    if (z3_read(c->w, o, n, w, c->nthreads)) return -1;
    if (pc && z3_read(c->pc, o, n, pc, c->nthreads)) return -1;
    if (ps && z3_read(c->ps, o, n, ps, c->nthreads)) return -1;
    return 0;
}
/* the fit-field zarr (ufsm predict --fit-zarr: arrays w, pc, ps; attributes ufsm.h = native voxels per cell, ufsm.origin_zyx) */
static int zf_open(const char *dir, int nthreads, zf_ctx *c, st_fields *F) {
    memset(c, 0, sizeof *c); c->nthreads = nthreads;
    store *s = store_open(dir); if (!s) return -1;
    c->w = z3_open(s, "w", nullptr); c->pc = z3_open(s, "pc", nullptr); c->ps = z3_open(s, "ps", nullptr);
    if (!c->w || !c->pc || !c->ps) { fprintf(stderr, "stitch: %s: %s\n", dir, z3_error()); return -1; }
    char p[1400]; snprintf(p, sizeof p, "%s/w/zarr.json", dir); FILE *f = fopen(p, "rb"); if (!f) return -1;
    fseek(f, 0, SEEK_END); const size_t n = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
    char *t = malloc(n + 1); const size_t got = fread(t, 1, n, f); fclose(f); t[got] = 0;
    json *j = json_parse(t, got); free(t);
    F->h = j ? json_num(json_path(j, "attributes.ufsm.h"), 1) : 1;
    for (int d = 0; d < 3; d++) F->org[d] = j ? json_num(json_at(json_path(j, "attributes.ufsm.origin_zyx"), d), 0) : 0;
    json_free(j);
    F->read = zf_read; F->ctx = c;
    return 0;
}
static void zf_close(zf_ctx *c) { if (c->w) z3_close(c->w); if (c->pc) z3_close(c->pc); if (c->ps) z3_close(c->ps); }
static void print_edges(const st_box *b, const st_edge *e, int ne) {
    fprintf(stderr, "%-16s %-16s %3s %5s %6s %6s %6s %5s %6s %6s %6s %9s %4s %5s\n", "box a", "box b", "s", "k", "agree", "tight", "local", "k2", "agree2", "other", "frac", "samples", "tree", "resid");
    for (int i = 0; i < ne; i++)
        fprintf(stderr, "%-16s %-16s %+3d %5d %6.3f %6.3f %6.3f %5d %6.3f %6.3f %+6.3f %9lld %4s %5d%s\n", b[e[i].a].name, b[e[i].b].name, e[i].s, e[i].k, e[i].agree, e[i].tight, e[i].local,
                e[i].k2, e[i].agree2, e[i].other, e[i].frac, e[i].n, e[i].tree ? "T" : e[i].used ? "-" : "x", e[i].resid, e[i].used && !e[i].tree && e[i].resid ? "  INCONSISTENT" : "");
}
static void print_boxes(const st_box *b, int nb) {
    for (int i = 0; i < nb; i++)
        fprintf(stderr, "  %-16s comp %2d  G = %+d W %+5.0f   W %7.2f..%7.2f -> G %7.2f..%7.2f\n", b[i].name, b[i].comp, b[i].s, b[i].g, b[i].wmin, b[i].wmax,
                b[i].s > 0 ? b[i].wmin + b[i].g : -b[i].wmax + b[i].g, b[i].s > 0 ? b[i].wmax + b[i].g : -b[i].wmin + b[i].g);
}
static void report_solve(const st_box *b, int nb, const st_edge *e, int ne, int bad) {
    int nc = 0; for (int i = 0; i < nb; i++) if (b[i].comp + 1 > nc) nc = b[i].comp + 1;
    for (int c = 0; c < nc; c++) {
        int n = 0, nt = 0, nu = 0; double ag = 0, mn = 1;
        for (int i = 0; i < nb; i++) n += b[i].comp == c;
        for (int i = 0; i < ne; i++) if (e[i].used && b[e[i].a].comp == c) { nu++; ag += e[i].agree; if (e[i].agree < mn) mn = e[i].agree; nt += e[i].tree; }
        fprintf(stderr, "stitch: component %d: %d boxes, %d used overlaps (%d in the tree), mean agreement %.3f (min %.3f)\n", c, n, nu, nt, nu ? ag / nu : 0, nu ? mn : 0);
    }
    fprintf(stderr, "stitch: %d inconsistent overlap(s) (cycles whose offsets do not close)\n", bad);
    for (int i = 0; i < ne; i++) if (e[i].used && !e[i].tree && e[i].resid)
        fprintf(stderr, "  %s - %s: k %d, the tree path says %d (%s)\n", b[e[i].a].name, b[e[i].b].name, e[i].k, e[i].k - e[i].resid, e[i].resid == 9999 ? "sign" : "offset");
}
static int stitch_align(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: ufsm stitch align <out.json> --fields ZARR [--stride 2] [--min-w 0.05] [--min-agree 0.5] [--min-wsum 50] [--threads 8] BOXDIR...\n"); return 2; }
    const char *out = argv[3], *fz = sopt(argc, argv, "--fields", nullptr);
    const int stride = atoi(sopt(argc, argv, "--stride", "2")), nt = atoi(sopt(argc, argv, "--threads", "8"));
    const double min_w = atof(sopt(argc, argv, "--min-w", "0.05")), min_agree = atof(sopt(argc, argv, "--min-agree", "0.5")), min_wsum = atof(sopt(argc, argv, "--min-wsum", "50"));
    if (!fz) { fprintf(stderr, "stitch align: --fields ZARR (the fit-field zarr the boxes were cropped from)\n"); return 2; }
    st_box *b = calloc(argc, sizeof *b); int nb = 0;
    for (int i = 4; i < argc; i++) {
        if (!strncmp(argv[i], "--", 2)) { i++; continue; }
        if (st_box_load(&b[nb], argv[i])) { fprintf(stderr, "stitch: %s: no fit model, skipped\n", argv[i]); continue; }
        nb++;
    }
    if (nb < 2) { fprintf(stderr, "stitch align: fewer than 2 boxes\n"); return 1; }
    zf_ctx zc; st_fields F; if (zf_open(fz, nt, &zc, &F)) return 1;
    const double um = b[0].fs->global->um;
    st_edge *e = calloc((size_t)nb * nb, sizeof *e); int ne = 0; const double t0 = now_s();
    for (int i = 0; i < nb; i++) for (int j = i + 1; j < nb; j++) {
        const int rc = st_align_pair(b, i, j, &F, stride, min_w, nt, &e[ne]);
        if (rc < 0) { fprintf(stderr, "stitch: cannot read the fields\n"); return 1; }
        if (rc) continue;
        const st_edge *x = &e[ne];
        fprintf(stderr, "stitch: %s - %s: s %+d k %d agree %.3f (tight %.3f, local %.3f, runner-up k %d %.3f, other sign %.3f, frac %+.3f) %lld samples, %.0fs\n",
                b[i].name, b[j].name, x->s, x->k, x->agree, x->tight, x->local, x->k2, x->agree2, x->other, x->frac, x->n, now_s() - t0);
        ne++;
    }
    zf_close(&zc);
    const int bad = st_solve(b, nb, e, ne, min_agree, min_wsum, nullptr);
    print_edges(b, e, ne); print_boxes(b, nb); report_solve(b, nb, e, ne, bad);
    if (st_save(out, b, nb, e, ne, um)) { fprintf(stderr, "stitch: cannot write %s\n", out); return 1; }
    fprintf(stderr, "stitch: %d boxes, %d overlaps -> %s\n", nb, ne, out);
    return 0;
}
static int stitch_solve(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: ufsm stitch solve <in.json> <out.json> [--min-agree 0.5] [--min-wsum 50] [--exclude NAME,NAME]\n"); return 2; }
    st_box *b; st_edge *e; int nb, ne; double um;
    if (st_load(argv[3], &b, &nb, &e, &ne, &um)) { fprintf(stderr, "stitch: cannot read %s\n", argv[3]); return 1; }
    uint8_t *ex = calloc(nb, 1); const char *xl = sopt(argc, argv, "--exclude", "");
    for (int i = 0; i < nb; i++) {
        const size_t l = strlen(b[i].name);
        for (const char *q = strstr(xl, b[i].name); q; q = strstr(q + 1, b[i].name)) if ((q == xl || q[-1] == ',') && (q[l] == ',' || !q[l])) ex[i] = 1;
    }
    const int bad = st_solve(b, nb, e, ne, atof(sopt(argc, argv, "--min-agree", "0.5")), atof(sopt(argc, argv, "--min-wsum", "50")), ex);
    print_edges(b, e, ne); print_boxes(b, nb); report_solve(b, nb, e, ne, bad);
    return st_save(argv[4], b, nb, e, ne, um) ? 1 : 0;
}
/* ufsm stitch field: G of one component on a grid of --step native voxels over its boxes' bounding box (float32, -1 = no box) */
typedef struct { const st_box *b; int nb, comp; double lo[3], step; int64_t o[3], n[3]; float *buf; _Atomic int64_t next; } fld_job;
static void *fld_worker(void *arg) {
    fld_job *J = arg;
    for (int64_t a; (a = atomic_fetch_add(&J->next, 1)) < J->n[0];) {
        for (int64_t y = 0; y < 512; y++) for (int64_t x = 0; x < 512; x++) {
            float *o = J->buf + ((size_t)a * 512 + y) * 512 + x; *o = -1;
            if (y >= J->n[1] || x >= J->n[2]) continue;
            const double p[3] = {J->lo[0] + (J->o[0] + a + 0.5) * J->step, J->lo[1] + (J->o[1] + y + 0.5) * J->step, J->lo[2] + (J->o[2] + x + 0.5) * J->step};
            double G; if (st_G(J->b, J->nb, J->comp, p, &G)) *o = (float)G;
        }
    }
    return nullptr;
}
static int stitch_field(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: ufsm stitch field <gauge.json> <out-group> [--comp 0] [--step 8] [--q 0.004] [--threads 8]\n"); return 2; }
    st_box *b; st_edge *e; int nb, ne; double um;
    if (st_load(argv[3], &b, &nb, &e, &ne, &um)) { fprintf(stderr, "stitch: cannot read %s\n", argv[3]); return 1; }
    const int comp = atoi(sopt(argc, argv, "--comp", "0")), nt = atoi(sopt(argc, argv, "--threads", "8")); const double step = atof(sopt(argc, argv, "--step", "8"));
    double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
    for (int i = 0; i < nb; i++) if (b[i].comp == comp) for (int d = 0; d < 3; d++) { lo[d] = fmin(lo[d], b[i].lo[d]); hi[d] = fmax(hi[d], b[i].hi[d]); }
    if (lo[0] > hi[0]) { fprintf(stderr, "stitch field: component %d has no boxes\n", comp); return 1; }
    int64_t gs[3]; for (int d = 0; d < 3; d++) gs[d] = (int64_t)ceil((hi[d] - lo[d]) / step);
    char lv[32], ld[1400], at[1600]; z3w_level_name(um * step, lv, sizeof lv); snprintf(ld, sizeof ld, "%s/%s", argv[4], lv);
    snprintf(at, sizeof at, "{\"ufsm\":{\"content\":\"stitched global winding G (float32, -1 = no box; sheets at integer G) of ufsm stitch\",\"gauge\":\"%s\",\"component\":%d,"
             "\"origin_zyx\":[%.3f,%.3f,%.3f],\"step\":%.6g,\"native_um\":%.9g}}", argv[3], comp, lo[0], lo[1], lo[2], step, um);
    z3w *zw = z3w_create_typed(ld, gs, 512, 5, (float)atof(sopt(argc, argv, "--q", "0.004")), -1.0, at);
    if (!zw || z3w_write_group(argv[4], (const double[]){um * step}, 1, "ufsm-stitched-winding", at)) { fprintf(stderr, "stitch: %s\n", z3w_error()); return 1; }
    float *buf = malloc((size_t)512 * 512 * 512 * 4); const double t0 = now_s();
    for (int64_t sz = 0; sz * 512 < gs[0]; sz++) for (int64_t sy = 0; sy * 512 < gs[1]; sy++) for (int64_t sx = 0; sx * 512 < gs[2]; sx++) {
        fld_job J = {b, nb, comp, {lo[0], lo[1], lo[2]}, step, {sz * 512, sy * 512, sx * 512},
                     {gs[0] - sz * 512 < 512 ? gs[0] - sz * 512 : 512, gs[1] - sy * 512 < 512 ? gs[1] - sy * 512 : 512, gs[2] - sx * 512 < 512 ? gs[2] - sx * 512 : 512}, buf, 0};
        for (int64_t a = J.n[0]; a < 512; a++) for (size_t q = 0; q < 512 * 512; q++) buf[(size_t)a * 512 * 512 + q] = -1;
        pthread_t th[64]; const int n = nt < 1 ? 1 : nt > 64 ? 64 : nt;
        for (int t = 0; t < n; t++) pthread_create(&th[t], nullptr, fld_worker, &J);
        for (int t = 0; t < n; t++) pthread_join(th[t], nullptr);
        if (z3w_write_shard(zw, sz, sy, sx, (const uint8_t *)buf, n)) { fprintf(stderr, "stitch: %s\n", z3w_error()); return 1; }
        fprintf(stderr, "stitch field: shard %lld,%lld,%lld (%.0fs)\n", (long long)sz, (long long)sy, (long long)sx, now_s() - t0);
    }
    if (z3w_close(zw)) { fprintf(stderr, "stitch: %s\n", z3w_error()); return 1; }
    fprintf(stderr, "stitch field: %lldx%lldx%lld (step %g, origin %.0f,%.0f,%.0f) -> %s\n", (long long)gs[0], (long long)gs[1], (long long)gs[2], step, lo[0], lo[1], lo[2], ld);
    free(buf);
    return 0;
}
/* the material mask for the sheets: the fields' w > 0 max-pooled by pool cells over [lo, hi) (native), cached in the out dir */
static uint8_t *load_mask(const char *fz, const char *outdir, const double lo[3], const double hi[3], int pool, int nt, int64_t mn[3], double mo[3], double *mh) {
    zf_ctx zc; st_fields F; if (zf_open(fz, nt, &zc, &F)) return nullptr;
    int64_t c0[3], c1[3];
    for (int d = 0; d < 3; d++) {
        c0[d] = (int64_t)floor((lo[d] - F.org[d]) / F.h / pool) * pool; if (c0[d] < 0) c0[d] = 0;
        c1[d] = (int64_t)ceil((hi[d] - F.org[d]) / F.h); mn[d] = (c1[d] - c0[d] + pool - 1) / pool; mo[d] = F.org[d] + c0[d] * F.h;
    }
    *mh = F.h * pool; const size_t nm = (size_t)mn[0] * mn[1] * mn[2];
    char p[1400]; snprintf(p, sizeof p, "%s/mask_%lld_%lld_%lld_%lldx%lldx%lld_p%d.raw", outdir, (long long)c0[0], (long long)c0[1], (long long)c0[2], (long long)mn[0], (long long)mn[1], (long long)mn[2], pool);
    uint8_t *m = calloc(nm, 1); FILE *f = fopen(p, "rb");
    if (f) { const size_t got = fread(m, 1, nm, f); fclose(f); if (got == nm) { zf_close(&zc); fprintf(stderr, "stitch: mask %s\n", p); return m; } }
    const double t0 = now_s(); const int64_t ny = mn[1] * pool, nx = mn[2] * pool; uint8_t *blk = malloc((size_t)pool * ny * nx);
    for (int64_t a = 0; a < mn[0]; a++) {
        const int64_t o[3] = {c0[0] + a * pool, c0[1], c0[2]}, n[3] = {pool, ny, nx};
        if (z3_read(zc.w, o, n, blk, nt)) { fprintf(stderr, "stitch: %s\n", z3_error()); free(m); free(blk); zf_close(&zc); return nullptr; }
        for (int64_t z = 0; z < pool; z++) for (int64_t y = 0; y < ny; y++) for (int64_t x = 0; x < nx; x++)
            if (blk[((size_t)z * ny + y) * nx + x]) m[((size_t)a * mn[1] + y / pool) * mn[2] + x / pool] = 1;
    }
    free(blk); zf_close(&zc);
    f = fopen(p, "wb"); if (f) { if (fwrite(m, 1, nm, f) != nm) fprintf(stderr, "stitch: short write %s\n", p); fclose(f); }
    fprintf(stderr, "stitch: mask %lldx%lldx%lld (pool %d) in %.0fs -> %s\n", (long long)mn[0], (long long)mn[1], (long long)mn[2], pool, now_s() - t0, p);
    return m;
}
/* a set of coarse cells (16 native voxels) the sheets of one k cover: open addressing */
typedef struct { uint64_t *k; size_t cap, n; } cellset;
static uint64_t cs_key(double z, double y, double x) {
    return ((uint64_t)(int64_t)floor(z / 16) & 0x1FFFFF) << 42 | ((uint64_t)(int64_t)floor(y / 16) & 0x1FFFFF) << 21 | ((uint64_t)(int64_t)floor(x / 16) & 0x1FFFFF);
}
static size_t cs_h(uint64_t k, size_t cap) { k ^= k >> 33; k *= 0xff51afd7ed558ccdULL; k ^= k >> 33; return (size_t)k & (cap - 1); }
static int cs_has(const cellset *s, uint64_t k) {
    if (!s->cap) return 0;
    for (size_t i = cs_h(k, s->cap);; i = (i + 1) & (s->cap - 1)) { if (s->k[i] == k + 1) return 1; if (!s->k[i]) return 0; }
}
static void cs_add(cellset *s, uint64_t k) {
    if (2 * (s->n + 1) > s->cap) {
        cellset t = {calloc(s->cap ? s->cap * 2 : 1024, 8), s->cap ? s->cap * 2 : 1024, 0};
        for (size_t i = 0; i < s->cap; i++) if (s->k[i]) cs_add(&t, s->k[i] - 1);
        free(s->k); *s = t;
    }
    for (size_t i = cs_h(k, s->cap);; i = (i + 1) & (s->cap - 1)) { if (s->k[i] == k + 1) return; if (!s->k[i]) { s->k[i] = k + 1; s->n++; return; } }
}
typedef struct {
    st_box *b; int nb, comp, follow; st_trace_cfg cfg;
    int k0, k1;                                          /* gauge mode: the levels G = k */
    int nitems, *item_box, *item_k; uint8_t **used; int *ubase, *ucount;   /* follow mode: seeds (box, own level), levels done */
    _Atomic int next;
    const char *out; double um, out_um, step; double min_cm2; double zf[8]; int nzf;
    const char *rdir; int rrows; double rum[4]; int nrum;
    pthread_mutex_t mu; FILE *tsv; _Atomic long long nsheets, nverts; double t0;
} sh_job;
/* the trace grid upsampled by f per axis (bilinear between valid vertices) in coordinates scaled by q (out-um): rows [r0, r1) and
   columns [c0, c1) of the upsampled grid; returns the valid vertex count */
static size_t upsample(const st_sheet *s, int f, double q, int r0, int r1, int c0, int c1, float *X, float *Y, float *Z, uint8_t *seam) {
    size_t nv = 0; const int oc = c1 - c0;
    for (int r = r0; r < r1; r++) for (int c = c0; c < c1; c++) {
        const size_t o = (size_t)(r - r0) * oc + (c - c0);
        X[o] = Y[o] = Z[o] = -1; if (seam) seam[o] = 0;
        const int ra = r / f, ca = c / f; const double fr = (double)(r % f) / f, fc = (double)(c % f) / f;
        double v[3] = {0, 0, 0}; int ok = 1; size_t bi = 0; double bw = -1;
        for (int q2 = 0; q2 < 4 && ok; q2++) {
            const int dr = q2 >> 1, dc = q2 & 1; const double w = (dr ? fr : 1 - fr) * (dc ? fc : 1 - fc);
            if (w <= 0) continue;
            if (ra + dr >= s->rows || ca + dc >= s->cols) { ok = 0; break; }
            const size_t i = (size_t)(ra + dr) * s->cols + ca + dc;
            if (s->z[i] < 0) { ok = 0; break; }
            v[0] += w * s->x[i]; v[1] += w * s->y[i]; v[2] += w * s->z[i];
            if (w > bw) { bw = w; bi = i; }
        }
        if (!ok) continue;
        X[o] = (float)((v[0] + 0.5) * q - 0.5); Y[o] = (float)((v[1] + 0.5) * q - 0.5); Z[o] = (float)((v[2] + 0.5) * q - 0.5);
        if (seam) seam[o] = s->seam[bi];
        nv++;
    }
    return nv;
}
static double grid_cm2(const float *Z, int w, int h, double step_um) {
    size_t quads = 0;
    for (int r = 0; r + 1 < h; r++) for (int c = 0; c + 1 < w; c++) {
        const size_t i = (size_t)r * w + c; if (Z[i] >= 0 && Z[i + 1] >= 0 && Z[i + w] >= 0 && Z[i + w + 1] >= 0) quads++;
    }
    return (double)quads * step_um * step_um * 1e-8;
}
/* the sheet as tifxyz (+ seam.tif, sheet.json: the boxes and their own levels) and its render copies; *cm2 its area */
static int write_sheet(sh_job *J, const st_sheet *s, const st_sgauge *g, const char *name, int seedbox, double level, const double seed[3], double *cm2) {
    int f = (int)lround(J->cfg.tstep / J->step); if (f < 1) f = 1;
    const int R = (s->rows - 1) * f + 1, C = (s->cols - 1) * f + 1; const size_t nv = (size_t)R * C;
    float *X = malloc(nv * 4), *Y = malloc(nv * 4), *Z = malloc(nv * 4); uint8_t *sm = malloc(nv);
    const double q = J->um / J->out_um, stepn = J->cfg.tstep / f;
    upsample(s, f, q, 0, R, 0, C, X, Y, Z, sm);
    *cm2 = grid_cm2(Z, C, R, stepn * J->um);
    int rc = 0;
    if (*cm2 >= J->min_cm2) {
        char d[1400], src[256]; snprintf(d, sizeof d, "%s/%s", J->out, name);
        if (J->follow) snprintf(src, sizeof src, "ufsm stitch (box fits joined along one sheet: seed %s w%03.0f)", J->b[seedbox].name, level);
        else snprintf(src, sizeof src, "ufsm stitch (level set G = %.0f of the gauge-aligned, blended box fits)", level);
        rc = tifxyz_write(d, C, R, X, Y, Z, stepn * q, J->out_um, (int)level, src);
        char p[1500]; snprintf(p, sizeof p, "%s/seam.tif", d);
        if (!rc) rc = tiffw_write(p, C, R, 1, sm, 1);
        snprintf(p, sizeof p, "%s/sheet.json", d); FILE *jf = rc ? nullptr : fopen(p, "w");
        if (jf) {
            fprintf(jf, "{\"name\": \"%s\", \"mode\": \"%s\", \"seed\": {\"box\": \"%s\", \"level\": %.0f, \"zyx_native\": [%.1f, %.1f, %.1f]}, \"area_cm2\": %.4f, \"native_um\": %.9g,\n"
                        " \"contacts\": %lld, \"contact_conflicts\": %lld, \"seam_vertices\": %lld, \"spread_vertices\": %lld, \"conflict_vertices\": %lld, \"unjoined_vertices\": %lld,\n \"boxes\": [",
                    name, J->follow ? "follow" : "gauge", J->b[seedbox].name, level, seed[0], seed[1], seed[2], *cm2, J->um, g->contacts, g->contact_conflicts, s->nseam, s->nspread,
                    s->nconflict, s->nunjoined);
            int first = 1;
            for (int j = 0; j < J->nb; j++) if (g->on[j] && s->own[j])
                { fprintf(jf, "%s\n  {\"box\": \"%s\", \"w\": %.0f, \"s\": %d, \"vertices\": %lld}", first ? "" : ",", J->b[j].name, g->off[j], g->s[j], s->own[j]); first = 0; }
            fprintf(jf, "]}\n"); if (fclose(jf)) rc = -1;
        }
        for (int u = 0; u < J->nrum && !rc && J->rdir; u++) {   /* render copies: row bands in other voxel units, columns cropped to the band */
            const double qr = J->um / J->rum[u];
            for (int r0 = 0; r0 < R && !rc; r0 += J->rrows) {
                const int r1 = r0 + J->rrows < R ? r0 + J->rrows : R; int c0 = C, c1 = 0;
                for (int r = r0; r < r1; r++) for (int c = 0; c < C; c++) if (Z[(size_t)r * C + c] >= 0) { if (c < c0) c0 = c; if (c + 1 > c1) c1 = c + 1; }
                if (c1 <= c0) continue;
                const size_t bn = (size_t)(r1 - r0) * (c1 - c0); float *bx = malloc(bn * 4), *by = malloc(bn * 4), *bz = malloc(bn * 4);
                const size_t bv = upsample(s, f, qr, r0, r1, c0, c1, bx, by, bz, nullptr);
                if (bv >= 1000) {
                    char bd[1600]; snprintf(bd, sizeof bd, "%s/%g/%s_r%05d_c%05d", J->rdir, J->rum[u], name, r0, c0);
                    rc = tifxyz_write(bd, c1 - c0, r1 - r0, bx, by, bz, stepn * qr, J->rum[u], (int)level, src);
                }
                free(bx); free(by); free(bz);
            }
        }
    }
    free(X); free(Y); free(Z); free(sm);
    return rc;
}
static int used_get(const sh_job *J, int i, double k) { const int q = (int)lround(k) - J->ubase[i]; return q >= 0 && q < J->ucount[i] && J->used[i][q]; }
static void used_set(sh_job *J, int i, double k) { const int q = (int)lround(k) - J->ubase[i]; if (q >= 0 && q < J->ucount[i]) J->used[i][q] = 1; }
/* one sheet from a seed: trace, write, report; returns 1 when it was written */
static int do_sheet(sh_job *J, st_sgauge *g, int seedbox, double level, const double p[3], const char *name, cellset *cs) {
    st_sheet s; const double t0 = now_s();
    if (st_trace(J->b, J->nb, J->comp, &J->cfg, g, J->follow, p, &s)) { st_sheet_free(&s); return 0; }
    if (cs) { for (size_t v = 0; v < (size_t)s.rows * s.cols; v++) if (s.z[v] >= 0) cs_add(cs, cs_key(s.z[v], s.y[v], s.x[v])); cs_add(cs, cs_key(p[0], p[1], p[2])); }
    double cm2 = 0; int wrote = 0;
    if (write_sheet(J, &s, g, name, seedbox, level, p, &cm2)) fprintf(stderr, "stitch: cannot write %s\n", name);
    pthread_mutex_lock(&J->mu);
    if (J->follow) for (int j = 0; j < J->nb; j++) if (g->on[j] && s.own[j]) used_set(J, j, g->off[j]);
    if (cm2 >= J->min_cm2) {
        int joined = 0; for (int j = 0; j < J->nb; j++) joined += g->on[j] && s.own[j] > 0;
        const double p50 = s.nspreads ? s.spreads[s.nspreads / 2] * J->um : 0, p90 = s.nspreads ? s.spreads[(long long)(0.9 * (s.nspreads - 1))] * J->um : 0;
        fprintf(J->tsv, "%s\t%s\t%.0f\t%d\t%d\t%lld\t%.3f\t%d\t%lld\t%lld\t%lld\t%.4f\t%.4f\t%.4f\t%.4f\t%.1f\t%.1f\t%.0f,%.0f,%.0f\t%.1f\n", name, J->b[seedbox].name, level,
                s.rows, s.cols, s.nvalid, cm2, joined, g->contacts, g->contact_conflicts, s.nseam, s.nvalid ? (double)s.nseam / s.nvalid : 0, s.nseam ? (double)s.nspread / s.nseam : 0,
                s.nseam ? (double)s.nconflict / s.nseam : 0, s.nvalid ? (double)s.nunjoined / s.nvalid : 0, p50, p90, p[0], p[1], p[2], now_s() - t0);
        fflush(J->tsv);
        fprintf(stderr, "stitch: %s %.2f cm2 (%dx%d trace grid, %d boxes, contacts %lld (%lld conflicting), seam vertices %lld: spread %.3f conflict %.3f, p50 %.1f um p90 %.1f um; unjoined %.3f) %.0fs [%.0fs]\n",
                name, cm2, s.rows, s.cols, joined, g->contacts, g->contact_conflicts, s.nseam, s.nseam ? (double)s.nspread / s.nseam : 0, s.nseam ? (double)s.nconflict / s.nseam : 0,
                p50, p90, s.nvalid ? (double)s.nunjoined / s.nvalid : 0, now_s() - t0, now_s() - J->t0);
        atomic_fetch_add(&J->nsheets, 1); atomic_fetch_add(&J->nverts, s.nvalid); wrote = 1;
    }
    pthread_mutex_unlock(&J->mu);
    st_sheet_free(&s);
    return wrote;
}
static void *sheet_worker(void *arg) {
    sh_job *J = arg; st_sgauge g = st_sgauge_new(J->nb);
    if (J->follow) {
        for (int it; (it = atomic_fetch_add(&J->next, 1)) < J->nitems;) {
            const int i = J->item_box[it], k = J->item_k[it];
            pthread_mutex_lock(&J->mu); const int u = used_get(J, i, k); if (!u) used_set(J, i, k); pthread_mutex_unlock(&J->mu);
            if (u) continue;
            st_sgauge_seed(&g, i, k); double p[3]; int ok = 0;
            for (int q = 0; q < J->nzf && !ok; q++) ok = !st_seed(J->b, i, J->zf[q], &g, &J->cfg, p);
            if (!ok) continue;
            char name[192]; snprintf(name, sizeof name, "%s_w%03d", J->b[i].name, k);
            do_sheet(J, &g, i, k, p, name, nullptr);
        }
    } else {
        for (int k; (k = J->k0 + atomic_fetch_add(&J->next, 1)) <= J->k1;) {
            cellset cs = {0}; int nw = 0;
            for (int i = 0; i < J->nb; i++) {
                if (J->b[i].comp != J->comp) continue;
                for (int q = 0; q < J->nzf; q++) {
                    st_sgauge_global(&g, J->b, J->comp, k);
                    double p[3]; if (st_seed(J->b, i, J->zf[q], &g, &J->cfg, p)) continue;
                    if (cs_has(&cs, cs_key(p[0], p[1], p[2]))) continue;
                    char name[64]; if (nw) snprintf(name, sizeof name, "k%04d_%d", k, nw); else snprintf(name, sizeof name, "k%04d", k);
                    if (do_sheet(J, &g, i, k, p, name, &cs)) nw++;
                }
            }
            free(cs.k);
        }
    }
    st_sgauge_free(&g);
    return nullptr;
}
static int stitch_sheets(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: ufsm stitch sheets <gauge.json> <out-dir> [--mode follow|gauge] [--comp C] [--first K] [--last K] [--boxes NAME,...] [--tstep 2] [--step 1]\n"
                        "       [--out-um 2.399] [--fields ZARR [--mask-pool 8]] [--threads 4] [--min-cm2 0.5] [--max-cols 4000] [--max-gap 50] [--seeds 0.5,0.25,0.75]\n"
                        "       [--render-dir D --render-rows 1042 --render-um 4.798,9.596]\n"
                        "  follow (default): every box level is a seed; a sheet joins the other boxes where it reaches them (all boxes, --comp -1)\n"
                        "  gauge: the levels G = k of the solved gauge of component --comp (0); --first / --last limit k, follow: the seed levels\n");
        return 2;
    }
    sh_job *J = calloc(1, sizeof *J);
    st_edge *e; int ne; if (st_load(argv[3], &J->b, &J->nb, &e, &ne, &J->um)) { fprintf(stderr, "stitch: cannot read %s\n", argv[3]); return 1; }
    free(e);
    J->follow = strcmp(sopt(argc, argv, "--mode", "follow"), "gauge") != 0;
    J->comp = atoi(sopt(argc, argv, "--comp", J->follow ? "-1" : "0")); J->out = argv[4];
    if (J->follow && J->comp < 0) { for (int i = 0; i < J->nb; i++) J->b[i].comp = 0; J->comp = -1; st_tapers(J->b, J->nb, 0); }
    J->cfg.tstep = atof(sopt(argc, argv, "--tstep", "2")); J->step = atof(sopt(argc, argv, "--step", "1")); J->out_um = atof(sopt(argc, argv, "--out-um", "2.399"));
    J->cfg.max_cols = atoi(sopt(argc, argv, "--max-cols", "4000")); J->cfg.max_gap = atoi(sopt(argc, argv, "--max-gap", "50")); J->min_cm2 = atof(sopt(argc, argv, "--min-cm2", "0.5"));
    const int nt = atoi(sopt(argc, argv, "--threads", "4"));
    { char *t = strdup(sopt(argc, argv, "--seeds", "0.5,0.25,0.75")), *sv = nullptr; for (char *q = strtok_r(t, ",", &sv); q && J->nzf < 8; q = strtok_r(nullptr, ",", &sv)) J->zf[J->nzf++] = atof(q); free(t); }
    J->rdir = sopt(argc, argv, "--render-dir", nullptr); J->rrows = atoi(sopt(argc, argv, "--render-rows", "1042"));
    { char *t = strdup(sopt(argc, argv, "--render-um", "4.798,9.596")), *sv = nullptr; for (char *q = strtok_r(t, ",", &sv); q && J->nrum < 4; q = strtok_r(nullptr, ",", &sv)) J->rum[J->nrum++] = atof(q); free(t); }
    const int kf = *sopt(argc, argv, "--first", "") ? atoi(sopt(argc, argv, "--first", "")) : -1000000, kl = *sopt(argc, argv, "--last", "") ? atoi(sopt(argc, argv, "--last", "")) : 1000000;
    const char *bl = sopt(argc, argv, "--boxes", nullptr);   /* follow: seed only in these boxes */
    double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300}, gmin = 1e300, gmax = -1e300;
    for (int i = 0; i < J->nb; i++) if (J->comp < 0 || J->b[i].comp == J->comp) {
        const st_box *B = &J->b[i];
        for (int d = 0; d < 3; d++) { lo[d] = fmin(lo[d], B->lo[d]); hi[d] = fmax(hi[d], B->hi[d]); }
        gmin = fmin(gmin, (B->s > 0 ? B->wmin : -B->wmax) + B->g); gmax = fmax(gmax, (B->s > 0 ? B->wmax : -B->wmin) + B->g);
    }
    if (lo[0] > hi[0]) { fprintf(stderr, "stitch sheets: component %d has no boxes\n", J->comp); return 1; }
    if (J->follow) {   /* seeds: boxes nearest the middle of the stack in z first, every level of each */
        J->used = calloc(J->nb, sizeof *J->used); J->ubase = calloc(J->nb, sizeof(int)); J->ucount = calloc(J->nb, sizeof(int));
        int *ord = malloc(J->nb * sizeof(int)), n = 0; const double zc = 0.5 * (lo[0] + hi[0]);
        for (int i = 0; i < J->nb; i++) {
            J->ubase[i] = (int)floor(J->b[i].wmin) - 4; J->ucount[i] = (int)ceil(J->b[i].wmax) - J->ubase[i] + 5; J->used[i] = calloc(J->ucount[i], 1);
            if (bl) { const char *q = strstr(bl, J->b[i].name); if (!q) continue; }
            ord[n++] = i;
        }
        for (int a = 0; a < n; a++) for (int c = a + 1; c < n; c++) {
            const double da = fabs(0.5 * (J->b[ord[a]].lo[0] + J->b[ord[a]].hi[0]) - zc), dc = fabs(0.5 * (J->b[ord[c]].lo[0] + J->b[ord[c]].hi[0]) - zc);
            if (dc < da) { const int t = ord[a]; ord[a] = ord[c]; ord[c] = t; }
        }
        int cap = 0; for (int a = 0; a < n; a++) cap += J->ucount[ord[a]];
        J->item_box = malloc((cap + 1) * sizeof(int)); J->item_k = malloc((cap + 1) * sizeof(int));
        for (int a = 0; a < n; a++) {
            const st_box *B = &J->b[ord[a]];
            for (int k = (int)ceil(B->wmin); k <= (int)floor(B->wmax); k++) if (k >= kf && k <= kl) { J->item_box[J->nitems] = ord[a]; J->item_k[J->nitems] = k; J->nitems++; }
        }
        free(ord);
    } else { J->k0 = (int)fmax(ceil(gmin), kf); J->k1 = (int)fmin(floor(gmax), kl); }
    char p[1400]; snprintf(p, sizeof p, "mkdir -p '%s'", J->out); if (system(p)) return 1;
    const char *fz = sopt(argc, argv, "--fields", nullptr); uint8_t *mask = nullptr;
    if (fz) {
        if (!(mask = load_mask(fz, J->out, lo, hi, atoi(sopt(argc, argv, "--mask-pool", "8")), 8, J->cfg.mn, J->cfg.mo, &J->cfg.mh))) return 1;
        J->cfg.mask = mask;
    }
    snprintf(p, sizeof p, "%s/sheets.tsv", J->out); J->tsv = fopen(p, "a"); if (!J->tsv) return 1;
    fprintf(J->tsv, "#name\tseed_box\tlevel\trows\tcols\tvertices\tarea_cm2\tboxes\tcontacts\tcontact_conflicts\tseam_vertices\tseam_frac\tspread_frac\tconflict_frac\tunjoined_frac"
                    "\tspread_p50_um\tspread_p90_um\tseed_zyx\tseconds\n");
    pthread_mutex_init(&J->mu, nullptr); J->t0 = now_s();
    if (J->follow) fprintf(stderr, "stitch sheets: follow mode, %d seed levels, trace step %g, out %g um%s\n", J->nitems, J->cfg.tstep, J->out_um, mask ? ", material mask" : "");
    else fprintf(stderr, "stitch sheets: gauge mode, component %d, k %d..%d, trace step %g, out %g um%s\n", J->comp, J->k0, J->k1, J->cfg.tstep, J->out_um, mask ? ", material mask" : "");
    pthread_t th[64]; const int n = nt < 1 ? 1 : nt > 64 ? 64 : nt;
    for (int t = 0; t < n; t++) pthread_create(&th[t], nullptr, sheet_worker, J);
    for (int t = 0; t < n; t++) pthread_join(th[t], nullptr);
    fclose(J->tsv);
    fprintf(stderr, "stitch sheets: %lld sheets, %lld trace vertices in %.0fs -> %s\n", (long long)atomic_load(&J->nsheets), (long long)atomic_load(&J->nverts), now_s() - J->t0, J->out);
    free(mask);
    return 0;
}
static int stitch_info(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm stitch info <gauge.json>\n"); return 2; }
    st_box *b; st_edge *e; int nb, ne; double um;
    if (st_load(argv[3], &b, &nb, &e, &ne, &um)) { fprintf(stderr, "stitch: cannot read %s\n", argv[3]); return 1; }
    int bad = 0; for (int i = 0; i < ne; i++) bad += e[i].used && !e[i].tree && e[i].resid;
    print_edges(b, e, ne); print_boxes(b, nb); report_solve(b, nb, e, ne, bad);
    return 0;
}
int cmd_stitch(int argc, char **argv) {
    const char *sub = argc > 2 ? argv[2] : "";
    if (!strcmp(sub, "align")) return stitch_align(argc, argv);
    if (!strcmp(sub, "solve")) return stitch_solve(argc, argv);
    if (!strcmp(sub, "field")) return stitch_field(argc, argv);
    if (!strcmp(sub, "sheets")) return stitch_sheets(argc, argv);
    if (!strcmp(sub, "info")) return stitch_info(argc, argv);
    fprintf(stderr, "usage: ufsm stitch align|solve|field|sheets|info ...\n"
                    "  align <out.json> --fields ZARR BOXDIR...   pairwise gauge offsets of overlapping box fits + the global gauge\n"
                    "  solve <in.json> <out.json> [--min-agree A] [--exclude NAMES]   the global gauge again (other thresholds)\n"
                    "  field <gauge.json> <out-group> [--step 8]    the blended global winding G (float32 zarr)\n"
                    "  sheets <gauge.json> <out-dir> [--fields ZARR]   level sets G = k as tifxyz across the boxes\n"
                    "  info <gauge.json>                            the overlap table and the gauges\n");
    return 2;
}
