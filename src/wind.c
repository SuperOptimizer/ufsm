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

static int cmp_double(const void *a, const void *b) { const double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }
int wind_seed(const float *xyz, const uint8_t *valid, int w, int h, void (*axis_fn)(void *, double, double *, double *), void *axis_ud,
              double K, int sense, double phi0, wind_seed_info *info, wind_result *out) {
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
    double sr[2] = {0, 0}, sp[2] = {0, 0}, rm = 0; size_t sn[2] = {0, 0}, se[2] = {0, 0}, nb = 0;   /* radius: 5% column bands; phi: the extreme columns */
    for (size_t i = 0; i < N; i++) {
        if (comp[i] != best) { if (valid[i]) out->flag[i] = 4; continue; }
        const int c = (int)(i % w), e = c < c0 + band ? 0 : c > c1 - band ? 1 : -1;
        if (e >= 0) { sr[e] += rad[i]; sn[e]++; }
        if (c == c0) { sp[0] += phi[i]; se[0]++; } else if (c == c1) { sp[1] += phi[i]; se[1]++; }
        rm += rad[i]; nb++;
    }
    for (int e = 0; e < 2; e++) { if (sn[e]) sr[e] /= (double)sn[e]; if (se[e]) sp[e] /= (double)se[e]; }
    const int rst = sr[0] <= sr[1] ? 0 : 1, rdir = sp[1 - rst] >= sp[rst] ? 1 : -1;   /* the radius rule: the start nearer the umbilicus */
    const int st = sense ? (sense * (sp[1] - sp[0]) >= 0 ? 0 : 1) : rst, en = 1 - st, dir = sense ? sense : rdir;
    const double span = dir * (sp[en] - sp[st]);
    double off = K - dir * sp[st];   /* q = off + dir phi */
    if (!isnan(phi0) && span < 1.2) {   /* the median of dir (phi - phi0) over a subsample of the component: its turn is K */
        size_t m = 0; const size_t stride = bsize / 4096 + 1; double *v = malloc((bsize / stride + 2) * sizeof *v);
        if (!v) { free(phi); free(theta); free(rad); free(comp); free(queue); wind_result_free(out); return -1; }
        size_t k = 0; for (size_t i = 0; i < N; i++) if (comp[i] == best && k++ % stride == 0) v[m++] = dir * (phi[i] - phi0);
        qsort(v, m, sizeof *v, cmp_double); const double med = v[m / 2]; free(v);
        off = K - floor(med) - dir * phi0;
    } else if (!isnan(phi0)) {
        double wo = dir * (sp[st] - phi0); wo -= floor(wo + 0.5);   /* the start's offset from the seam, in [-0.5, 0.5) */
        off = K + wo - dir * sp[st];
    }
    for (size_t i = 0; i < N; i++) if (comp[i] == best) { out->q[i] = (float)(off + dir * phi[i]); out->flag[i] = 1; info->nassigned++; }
    for (size_t i = 0; i < N; i++) info->nvalid += valid[i] != 0;
    info->dir = dir; info->radial_dir = rdir; info->r_start = sr[st]; info->r_end = sr[en]; info->r_mean = nb ? rm / (double)nb : 0; info->span = span;
    info->q_start = off + dir * sp[st];
    for (int e = 0; e < 2; e++) { info->phi_c[e] = sp[e]; info->r_c[e] = sr[e]; }
    free(phi); free(theta); free(rad); free(comp); free(queue);
    return 0;
}

void wind_seed_seam(const wind_seed_info *in, int n, int sense, double *phi0, double *resultant) {
    double cx = 0, cy = 0;
    for (int k = 0; k < n; k++) {
        const double ps = sense * (in[k].phi_c[1] - in[k].phi_c[0]) >= 0 ? in[k].phi_c[0] : in[k].phi_c[1];
        cx += cos(2 * M_PI * ps); cy += sin(2 * M_PI * ps);
    }
    *phi0 = n ? atan2(cy, cx) / (2 * M_PI) : 0; if (*phi0 < 0) *phi0 += 1;
    *resultant = n ? hypot(cx, cy) / n : 0;
}
void wind_seed_set(const wind_seed_info *in, int n, int *sense, double *phi0, double vote[2], double *resultant) {
    vote[0] = vote[1] = 0;
    for (int k = 0; k < n; k++) vote[in[k].radial_dir > 0 ? 0 : 1] += fabs(in[k].r_c[1] - in[k].r_c[0]);
    *sense = vote[0] >= vote[1] ? 1 : -1;
    wind_seed_seam(in, n, *sense, phi0, resultant);
}

wind_pt *wind_seed_points(const float *xyz, const uint8_t *valid, int w, int h, void (*axis_fn)(void *, double, double *, double *), void *axis_ud,
                          const float *qp, const float *qm, int32_t surf, size_t target, size_t *n) {
    *n = 0; size_t nv = 0;
    for (size_t i = 0; i < (size_t)w * h; i++) nv += valid[i] && !isnan(qp[i]) && !isnan(qm[i]);
    if (!nv) return nullptr;
    int st = target ? (int)ceil(sqrt((double)nv / (double)target)) : 1; if (st < 1) st = 1;
    const size_t cap = (size_t)((h + st - 1) / st) * (size_t)((w + st - 1) / st);
    wind_pt *p = malloc(cap * sizeof *p); if (!p) return nullptr;
    float lo = INFINITY, hi = -INFINITY;   /* the phi range (q under sense +1 is off + phi) */
    for (size_t i = 0; i < (size_t)w * h; i++) if (valid[i] && !isnan(qp[i])) { if (qp[i] < lo) lo = qp[i]; if (qp[i] > hi) hi = qp[i]; }
    const int32_t turn = hi - lo >= 0.9f;
    size_t k = 0;
    for (int r = 0; r < h; r += st) for (int c = 0; c < w; c += st) {
        const size_t i = (size_t)r * w + c;
        if (!valid[i] || isnan(qp[i]) || isnan(qm[i]) || !(xyz[i * 3 + 2] >= 0)) continue;
        double cy, cx; axis_fn(axis_ud, xyz[i * 3 + 2], &cy, &cx);
        const double dy = xyz[i * 3 + 1] - cy, dx = xyz[i * 3] - cx;
        double t = atan2(dy, dx) / (2 * M_PI); t -= floor(t);
        int tb = (int)(t * WIND_TBIN); if (tb >= WIND_TBIN) tb = WIND_TBIN - 1;
        p[k++] = (wind_pt){(int64_t)floor(xyz[i * 3 + 2] / WIND_ZBIN) * WIND_TBIN + tb, (float)hypot(dy, dx), {qp[i], qm[i]}, surf, turn};
    }
    *n = k; return p;
}

double wind_pitch_lo = 0, wind_pitch_hi = 0, wind_min_surface_score = 0.5;
static int cmp_wind_pt(const void *a, const void *b) {
    const wind_pt *x = a, *y = b;
    if (x->key != y->key) return x->key < y->key ? -1 : 1;
    if (x->r != y->r) return x->r < y->r ? -1 : 1;
    return x->surf - y->surf;
}
/* the pair partner of point i in its bin: the previous point (any surface, cross) or the previous point of the same surface (its own
   pairs; up to 64 points back), the dropped surfaces skipped; SIZE_MAX none */
static size_t wind_partner(const wind_pt *p, size_t i, int same, const uint8_t *drop) {
    for (size_t j = i, steps = 0; j-- > 0 && steps < 64; steps++) {
        if (p[j].key != p[i].key) break;
        if (drop && drop[p[j].surf]) continue;
        if (!same || p[j].surf == p[i].surf) return j;
    }
    return SIZE_MAX;
}
/* one candidate's votes over the pairs per bin, the dropped surfaces skipped; per-surface counts when per != nullptr.
   pitch > 0: pairs at least half a pitch apart and whole turns apart (|dq - round(dq)| < 0.25: on one ray the sheets of one gauge
   are; anything else is an unwrapping artefact or a gauge clash, no vote) are radial votes (the outer q >= 0.5 higher: agree, >= 0.5
   lower: disagree), pairs
   of two surfaces under a quarter pitch apart are same-sheet votes (|dq| < 0.5 agree, else disagree: the seam's continuity, where
   wrap K's end meets wrap K + 1's start, is what tells the sense of sets of single wraps); pitch 0: radial votes from 1 voxel. */
static void wind_votes(const wind_pt *p, size_t n, const double *K, int cross, int si, int order, const uint8_t *drop, double pitch, size_t *agree, size_t *dis, wind_surf_vote *per) {
    *agree = *dis = 0;
    const float rmin = pitch > 0 ? (float)(0.5 * pitch) : 1.f, smax = (float)(0.25 * pitch);
    for (size_t i = 0; i < n; i++) {
        if (drop[p[i].surf]) continue;
        const size_t j = wind_partner(p, i, !cross, drop); if (j == SIZE_MAX) continue;
        const wind_pt *a = &p[j], *b = &p[i]; const float dr = b->r - a->r;
        if (a->surf == b->surf && !a->turn) continue;   /* a surface under 0.9 turn has no pairs of its own (only unwrapping artefacts) */
        const double qa = a->q[si] - (order < 0 ? 2 * K[a->surf] : 0), qb = b->q[si] - (order < 0 ? 2 * K[b->surf] : 0), dq = qb - qa;
        int v = 0;
        if (dr >= rmin && fabs(dq - round(dq)) < 0.25 && (pitch <= 0 || dr <= 2.5 * pitch * fabs(round(dq)) + 0.5 * pitch))   /* (one ray: whole turns apart, about a pitch per turn) */
            v = dq >= 0.5 ? 1 : dq <= -0.5 ? -1 : 0;
        else if (a->surf != b->surf && dr < smax) v = fabs(dq) < 0.5 ? 1 : -1;
        if (v > 0) (*agree)++; else if (v < 0) (*dis)++;
        if (v && per) {
            if (v > 0) { per[a->surf].agree++; if (b->surf != a->surf) per[b->surf].agree++; }
            else { per[a->surf].dis++; if (b->surf != a->surf) per[b->surf].dis++; }
        }
    }
}
static int cmp_float_w(const void *a, const void *b) { const float x = *(const float *)a, y = *(const float *)b; return x < y ? -1 : x > y; }
/* the sheet pitch (scan voxels): the median radial gap of pairs one turn apart (|dq| 0.75 .. 1.25 under sense +1, names as given:
   one sheet apart under any sense); 0 without such pairs */
static double wind_pitch(const wind_pt *p, size_t n, const double *K, int cross) {
    size_t m = 0, cap = 0; float *g = nullptr;
    for (size_t i = 0; i < n; i++) {
        const size_t j = wind_partner(p, i, !cross, nullptr); if (j == SIZE_MAX || (p[i].surf == p[j].surf && !p[i].turn)) continue;
        const double dq = fabs((double)p[i].q[0] - p[j].q[0]);
        if (dq < 0.75 || dq > 1.25 || p[i].r - p[j].r < 1.f) continue;
        if (m == cap) { cap = cap ? 2 * cap : 4096; float *t = realloc(g, cap * sizeof *g); if (!t) { free(g); return 0; } g = t; }
        g[m++] = p[i].r - p[j].r;
    }
    double r = 0; if (m >= 20) { qsort(g, m, sizeof *g, cmp_float_w); r = g[m / 2]; }
    free(g); (void)K; return r;
}
int wind_seed_decide(wind_pt *pts, size_t n, const double *K, int nsurf, int cross, int force_sense, int force_order, int fallback_sense,
                     size_t min_pairs, wind_choice *ch, wind_surf_vote *per) {
    memset(ch, 0, sizeof *ch); memset(per, 0, (size_t)(nsurf > 0 ? nsurf : 0) * sizeof *per);
    uint8_t *drop = calloc((size_t)(nsurf > 0 ? nsurf : 1), 1); size_t *da = calloc((size_t)(nsurf > 0 ? nsurf : 1), sizeof *da), *dd = calloc((size_t)(nsurf > 0 ? nsurf : 1), sizeof *dd);
    if (!drop || !da || !dd) { free(drop); free(da); free(dd); return -1; }
    if (n) qsort(pts, n, sizeof *pts, cmp_wind_pt);
    /* each surface's own sense: pairs inside it, q under sense +1 = off + phi, so dq is dphi */
    { for (size_t i = 0; i < n; i++) {
          const size_t j = wind_partner(pts, i, 1, nullptr);
          if (j != SIZE_MAX && pts[i].turn && pts[i].r - pts[j].r >= 1.f && fabs((double)pts[i].q[0] - pts[j].q[0] - round((double)pts[i].q[0] - pts[j].q[0])) < 0.25) {
              const double dq = pts[i].q[0] - pts[j].q[0];
              if (dq >= 0.5) per[pts[i].surf].own_pos++; else if (dq <= -0.5) per[pts[i].surf].own_neg++;
          }
      }
      for (int k = 0; k < nsurf; k++) { const size_t a = per[k].own_pos, b = per[k].own_neg;
          per[k].own_dir = a + b >= min_pairs && a + b > 0 ? (a * 10 >= (a + b) * 6 ? 1 : b * 10 >= (a + b) * 6 ? -1 : 0) : 0; } }
    const int fb = fallback_sense < 0 ? -1 : 1;
    ch->pitch = wind_pitch(pts, n, K, cross);
    ch->pitch_ok = !(ch->pitch > 0 && ((wind_pitch_lo > 0 && ch->pitch < wind_pitch_lo) || (wind_pitch_hi > 0 && ch->pitch > wind_pitch_hi)));
    for (;;) {
        int bs = 0, bo = 0; double best = -1; size_t tot = 0;
        for (int si = 0; si < 2; si++) for (int oi = 0; oi < 2; oi++) {
            const int s = si ? -1 : 1, o = oi ? -1 : 1;
            ch->cand[si][oi] = NAN; ch->cand_n[si][oi] = 0;
            if ((force_sense && s != force_sense) || (force_order && o != force_order)) continue;
            if (!cross && o < 0) continue;   /* without names the order means nothing */
            size_t a, d; wind_votes(pts, n, K, cross, si, o, drop, ch->pitch, &a, &d, nullptr);
            ch->cand_n[si][oi] = a + d; ch->cand[si][oi] = a + d ? (double)a / (double)(a + d) : NAN; tot += a + d;
            const double sc = a + d ? (double)a / (double)(a + d) : 0.5;
            /* the best score; ties to the fallback sense and the names' own order */
            const int pref = (s == fb) * 2 + (o == 1);
            const int bpref = bs ? (bs == fb) * 2 + (bo == 1) : -1;
            if (sc > best + 1e-9 || (fabs(sc - best) <= 1e-9 && pref > bpref)) { best = sc; bs = s; bo = o; }
        }
        if (!bs) { bs = force_sense ? force_sense : fb; bo = force_order ? force_order : 1; }
        ch->sense = bs; ch->order = bo; ch->evidence = tot > 0 && ch->pitch_ok;
        for (int k = 0; k < nsurf; k++) { per[k].agree = per[k].dis = 0; }
        wind_votes(pts, n, K, cross, bs > 0 ? 0 : 1, bo, drop, ch->pitch, &ch->agree, &ch->dis, per);
        ch->score = ch->agree + ch->dis ? (double)ch->agree / (double)(ch->agree + ch->dis) : NAN;
        {   /* how far the choice beats the other sense (same order) and the other order (same sense): NaN when not compared */
            const int si = bs > 0 ? 0 : 1, oi = bo > 0 ? 0 : 1;
            const double c = ch->cand[si][oi], os = ch->cand[1 - si][oi], oo = ch->cand[si][1 - oi];
            ch->sense_margin = isnan(c) || isnan(os) ? NAN : c - os; ch->order_margin = isnan(c) || isnan(oo) ? NAN : c - oo; }
        int worst = -1; double wv = 0;   /* the surface whose votes contradict the choice most */
        for (int k = 0; k < nsurf; k++) {
            const size_t nv = per[k].agree + per[k].dis;
            if (drop[k] || nv < min_pairs || (double)per[k].agree >= wind_min_surface_score * (double)nv) continue;
            const double v = (double)per[k].dis - (1 - wind_min_surface_score) / wind_min_surface_score * (double)per[k].agree;   /* the votes beyond the share */
            if (v > wv || worst < 0) { wv = v; worst = k; }
        }
        if (worst < 0) break;
        drop[worst] = 1; ch->ndropped++; da[worst] = per[worst].agree; dd[worst] = per[worst].dis;   /* (its votes when dropped, for the report) */
    }
    for (int k = 0; k < nsurf; k++) { per[k].dropped = drop[k]; if (drop[k]) { per[k].agree = da[k]; per[k].dis = dd[k]; } }
    free(drop); free(da); free(dd);
    return 0;
}
