/* wind_assign (wind.h): one winding K and dir per connected component of a surface, voted by its samples that lie on reference
   voxels with a winding. */
#include "wind.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static int cmp_float(const void *a, const void *b) { const float x = *(const float *)a, y = *(const float *)b; return x < y ? -1 : x > y; }

void wind_result_free(wind_result *r) { free(r->q); free(r->flag); r->q = nullptr; r->flag = nullptr; }

int wind_assign(const float *xyz, const uint8_t *valid, int w, int h, void (*axis_fn)(void *, double, double *, double *), void *axis_ud,
                wind_lookup_fn lookup, void *lookup_ud, wind_params p, wind_result *out) {
    memset(out, 0, sizeof *out);
    const size_t N = (size_t)w * h; const int step = p.step < 1 ? 1 : p.step;
    double *phi = malloc(N * sizeof *phi), *theta = malloc(N * sizeof *theta);
    int32_t *comp = malloc(N * sizeof *comp), *queue = malloc(N * sizeof *queue);
    out->q = malloc(N * sizeof *out->q); out->flag = calloc(N, 1);
    if (!phi || !theta || !comp || !queue || !out->q || !out->flag) { free(phi); free(theta); free(comp); free(queue); wind_result_free(out); return -1; }
    for (size_t i = 0; i < N; i++) {
        out->q[i] = NAN; comp[i] = -1;
        if (!valid[i]) continue;
        double cy, cx; axis_fn(axis_ud, xyz[i * 3 + 2], &cy, &cx);
        theta[i] = atan2(xyz[i * 3 + 1] - cy, xyz[i * 3] - cx);
    }
    /* connected components (4-neighbourhood) and phi = theta / 2 pi unwrapped along the grid by breadth-first search */
    int nc = 0; size_t *csize = nullptr;
    for (size_t s = 0; s < N; s++) {
        if (!valid[s] || comp[s] >= 0) continue;
        size_t *t = realloc(csize, (size_t)(nc + 1) * sizeof *csize); if (!t) { free(phi); free(theta); free(comp); free(queue); free(csize); wind_result_free(out); return -1; }
        csize = t; csize[nc] = 0;
        size_t qh = 0, qt = 0; queue[qt++] = (int32_t)s; comp[s] = nc; phi[s] = theta[s] / (2 * M_PI);
        while (qh < qt) {
            const int32_t i = queue[qh++]; const int r = i / w, c = i % w; csize[nc]++;
            const int nb[4][2] = {{r - 1, c}, {r + 1, c}, {r, c - 1}, {r, c + 1}};
            for (int k = 0; k < 4; k++) {
                const int rr = nb[k][0], cc = nb[k][1];
                if (rr < 0 || rr >= h || cc < 0 || cc >= w) continue;
                const size_t j = (size_t)rr * w + cc;
                if (!valid[j] || comp[j] >= 0) continue;
                double d = theta[j] - theta[i]; d -= 2 * M_PI * round(d / (2 * M_PI));
                phi[j] = phi[i] + d / (2 * M_PI); comp[j] = nc; queue[qt++] = (int32_t)j;
            }
        }
        nc++;
    }
    /* samples and their reference windings */
    size_t ns = 0; for (int r = 0; r < h; r += step) for (int c = 0; c < w; c += step) ns += valid[(size_t)r * w + c] != 0;
    float *pts = malloc((ns ? ns : 1) * 3 * sizeof *pts), *dist = malloc((ns ? ns : 1) * sizeof *dist), *qref = malloc((ns ? ns : 1) * sizeof *qref);
    int32_t *sidx = malloc((ns ? ns : 1) * sizeof *sidx);
    int (*hist)[2][14] = calloc((size_t)(nc ? nc : 1), sizeof *hist);
    if (!pts || !dist || !qref || !sidx || !hist) { free(pts); free(dist); free(qref); free(sidx); free(hist); free(phi); free(theta); free(comp); free(queue); free(csize); wind_result_free(out); return -1; }
    size_t k = 0;
    for (int r = 0; r < h; r += step) for (int c = 0; c < w; c += step) {
        const size_t i = (size_t)r * w + c; if (!valid[i]) continue;
        pts[k * 3] = (float)(xyz[i * 3 + 2] * p.scale); pts[k * 3 + 1] = (float)(xyz[i * 3 + 1] * p.scale); pts[k * 3 + 2] = (float)(xyz[i * 3] * p.scale);
        sidx[k++] = (int32_t)i;
    }
    if (ns) lookup(lookup_ud, pts, ns, p.tol, dist, qref);
    for (size_t s = 0; s < ns; s++) {
        if (!(dist[s] <= p.tol) || isnan(qref[s])) continue;
        const int ci = comp[sidx[s]];
        for (int di = 0; di < 2; di++) {
            const double sg = di ? -1 : 1, d = qref[s] - sg * phi[sidx[s]];
            hist[ci][di][((int)(lround(d) % 14) + 14) % 14]++;
        }
    }
    /* per component: (dir, K) of the most votes; verdict; q; dropped cells */
    int *order = malloc((size_t)(nc ? nc : 1) * sizeof *order), *cdir = malloc((size_t)(nc ? nc : 1) * sizeof *cdir), *cK = malloc((size_t)(nc ? nc : 1) * sizeof *cK), *cver = malloc((size_t)(nc ? nc : 1) * sizeof *cver);
    for (int c = 0; c < nc; c++) order[c] = c;
    for (int a = 1; a < nc; a++) for (int b = a; b > 0 && csize[order[b]] > csize[order[b - 1]]; b--) { const int t = order[b]; order[b] = order[b - 1]; order[b - 1] = t; }
    size_t nvalid = 0; for (int c = 0; c < nc; c++) nvalid += csize[c];
    float *cd = malloc((ns ? ns : 1) * sizeof *cd);
    for (int oi = 0; oi < nc; oi++) {
        const int c = order[oi]; int bd = 0, bk = 0;
        {   /* per dir its best K; the dir whose agreeing votes sit closer to integer turns wins unless it has under half the other's
               votes (a component spanning little theta votes much the same K both ways; the wrong dir's residuals are ~uniform) */
            int kb[2] = {0, 0}, cnt[2] = {-1, -1}; double rs[2] = {INFINITY, INFINITY};
            for (int di = 0; di < 2; di++) {
                if (p.dir && (di ? -1 : 1) != p.dir) continue;
                for (int kk = 0; kk < 14; kk++) if (hist[c][di][kk] > cnt[di]) { cnt[di] = hist[c][di][kk]; kb[di] = kk; }
                double sum = 0; size_t na = 0;
                for (size_t s = 0; s < ns; s++) {
                    if (comp[sidx[s]] != c || !(dist[s] <= p.tol) || isnan(qref[s])) continue;
                    const double d = qref[s] - (di ? -1 : 1) * phi[sidx[s]]; const long kr = lround(d);
                    if (((int)(kr % 14) + 14) % 14 == kb[di]) { sum += fabs(d - (double)kr); na++; }
                }
                rs[di] = na ? sum / (double)na : INFINITY;
            }
            bd = cnt[1] < 0 ? 0 : cnt[0] < 0 ? 1 : rs[1] < rs[0] && 2 * cnt[1] >= cnt[0] ? 1 : rs[0] <= rs[1] && 2 * cnt[0] >= cnt[1] ? 0 : cnt[1] > cnt[0];
            bk = kb[bd];
        }
        const double sg = bd ? -1 : 1;
        size_t nsm = 0, nm = 0, na = 0, nd = 0; double res = 0;
        for (size_t s = 0; s < ns; s++) {
            if (comp[sidx[s]] != c) continue;
            nsm++;
            if (!(dist[s] <= p.tol) || isnan(qref[s])) continue;
            nm++; cd[nd++] = dist[s];
            const double d = qref[s] - sg * phi[sidx[s]]; const long kr = lround(d);
            if (((int)(kr % 14) + 14) % 14 == bk) { na++; res += fabs(d - (double)kr); }
        }
        if (nd) qsort(cd, nd, sizeof *cd, cmp_float);
        const double med = nd ? cd[nd / 2] : INFINITY;
        int verdict;
        if (nm < 8 || nm * 20 < nsm) verdict = WA_UNASSIGNED;
        else if (na * 5 < nm * 4) verdict = WA_CONFLICT;
        else if (!p.dense && nm * 2 >= nsm && med <= p.tol / 2) verdict = WA_DUPLICATE;
        else verdict = WA_NEW;
        cdir[c] = bd ? -1 : 1; cK[c] = bk; cver[c] = verdict;
        if (out->ncomp < 16 && csize[c] * 100 >= nvalid) {
            wind_comp *o = &out->comp[out->ncomp++];
            *o = (wind_comp){cdir[c], bk, verdict, csize[c], nsm, nm, na, med, na ? res / (double)na : 0};
        }
    }
    for (size_t i = 0; i < N; i++) {
        if (!valid[i]) continue;
        const int c = comp[i];
        if (cver[c] == WA_UNASSIGNED) { out->flag[i] = 4; continue; }
        out->q[i] = (float)(cK[c] + cdir[c] * phi[i]); out->flag[i] = 1;
    }
    const int half = step / 2;
    for (size_t s = 0; s < ns; s++) {   /* a sample's cell (its step x step block of vertices) is dropped on disagreement / with drop_dup */
        const int32_t i = sidx[s]; const int c = comp[i];
        if (cver[c] == WA_UNASSIGNED || !(dist[s] <= p.tol) || isnan(qref[s])) continue;
        const double d = qref[s] - cdir[c] * phi[i]; const int kv = ((int)(lround(d) % 14) + 14) % 14;
        const uint8_t f = kv != cK[c] ? 2 : p.drop_dup && !p.dense ? 3 : 0;
        if (!f) continue;
        const int r0 = i / w, c0 = i % w;
        for (int r = r0 - half; r <= r0 + half; r++) for (int cc = c0 - half; cc <= c0 + half; cc++) {
            if (r < 0 || r >= h || cc < 0 || cc >= w) continue;
            const size_t j = (size_t)r * w + cc;
            if (comp[j] != c || out->flag[j] == 2) continue;
            out->flag[j] = f; out->q[j] = NAN;
        }
    }
    free(cd); free(order); free(cdir); free(cK); free(cver);
    free(pts); free(dist); free(qref); free(sidx); free(hist); free(phi); free(theta); free(comp); free(queue); free(csize);
    return 0;
}

int wind_seed(const float *xyz, const uint8_t *valid, int w, int h, void (*axis_fn)(void *, double, double *, double *), void *axis_ud,
              double K, wind_seed_info *info, wind_result *out) {
    memset(out, 0, sizeof *out); memset(info, 0, sizeof *info);
    const size_t N = (size_t)w * h;
    double *phi = malloc(N * sizeof *phi), *theta = malloc(N * sizeof *theta), *rad = malloc(N * sizeof *rad);
    int32_t *comp = malloc(N * sizeof *comp), *queue = malloc(N * sizeof *queue);
    out->q = malloc(N * sizeof *out->q); out->flag = calloc(N, 1);
    if (!phi || !theta || !rad || !comp || !queue || !out->q || !out->flag) { free(phi); free(theta); free(rad); free(comp); free(queue); wind_result_free(out); return -1; }
    for (size_t i = 0; i < N; i++) {
        out->q[i] = NAN; comp[i] = -1;
        if (!valid[i]) continue;
        double cy, cx; axis_fn(axis_ud, xyz[i * 3 + 2], &cy, &cx);
        theta[i] = atan2(xyz[i * 3 + 1] - cy, xyz[i * 3] - cx); rad[i] = hypot(xyz[i * 3 + 1] - cy, xyz[i * 3] - cx);
    }
    int nc = 0, best = -1; size_t bsize = 0;
    for (size_t s = 0; s < N; s++) {   /* components and unwrapped phi, as wind_assign */
        if (!valid[s] || comp[s] >= 0) continue;
        size_t qh = 0, qt = 0, cs = 0; queue[qt++] = (int32_t)s; comp[s] = nc; phi[s] = theta[s] / (2 * M_PI);
        while (qh < qt) {
            const int32_t i = queue[qh++]; const int r = i / w, c = i % w; cs++;
            const int nb[4][2] = {{r - 1, c}, {r + 1, c}, {r, c - 1}, {r, c + 1}};
            for (int k = 0; k < 4; k++) {
                const int rr = nb[k][0], cc = nb[k][1];
                if (rr < 0 || rr >= h || cc < 0 || cc >= w) continue;
                const size_t j = (size_t)rr * w + cc;
                if (!valid[j] || comp[j] >= 0) continue;
                double d = theta[j] - theta[i]; d -= 2 * M_PI * round(d / (2 * M_PI));
                phi[j] = phi[i] + d / (2 * M_PI); comp[j] = nc; queue[qt++] = (int32_t)j;
            }
        }
        if (cs > bsize) { bsize = cs; best = nc; }
        nc++;
    }
    if (best < 0) { free(phi); free(theta); free(rad); free(comp); free(queue); return -2; }
    int c0 = w, c1 = -1;
    for (size_t i = 0; i < N; i++) if (comp[i] == best) { const int c = (int)(i % w); if (c < c0) c0 = c; if (c > c1) c1 = c; }
    const int band = (c1 - c0) / 20 + 1;
    double sr[2] = {0, 0}, sp[2] = {0, 0}; size_t sn[2] = {0, 0}, se[2] = {0, 0};   /* radius: 5% column bands; phi: the extreme columns */
    for (size_t i = 0; i < N; i++) {
        if (comp[i] != best) { if (valid[i]) out->flag[i] = 4; continue; }
        const int c = (int)(i % w), e = c < c0 + band ? 0 : c > c1 - band ? 1 : -1;
        if (e >= 0) { sr[e] += rad[i]; sn[e]++; }
        if (c == c0) { sp[0] += phi[i]; se[0]++; } else if (c == c1) { sp[1] += phi[i]; se[1]++; }
    }
    for (int e = 0; e < 2; e++) { if (sn[e]) sr[e] /= (double)sn[e]; if (se[e]) sp[e] /= (double)se[e]; }
    const int st = sr[0] <= sr[1] ? 0 : 1, en = 1 - st;   /* the start: the end nearer the umbilicus */
    const int dir = sp[en] >= sp[st] ? 1 : -1;
    for (size_t i = 0; i < N; i++) if (comp[i] == best) { out->q[i] = (float)(K + dir * (phi[i] - sp[st])); out->flag[i] = 1; info->nassigned++; }
    for (size_t i = 0; i < N; i++) info->nvalid += valid[i] != 0;
    info->dir = dir; info->r_start = sr[st]; info->r_end = sr[en]; info->span = dir * (sp[en] - sp[st]);
    free(phi); free(theta); free(rad); free(comp); free(queue);
    return 0;
}
