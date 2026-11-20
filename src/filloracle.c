/* ufsm fill-oracle: the count-constrained fill and its oracles on one box (levelsolve, 2026-11-18).
   ufsm fill-oracle --fields DIR --model M --codes ROOT KEY --out DIR [--mode base|interp|fill|shape|ref|fillc|refc] [--mask gt|mf]
                    [--mf-thr 179] [--ref R] [--threads N]
   Where the net merges touching sheets its phase runs one cycle across 2-3 GT sheets, so no integer per piece fixes the winding there
   (DESIGN "HerculeX exact integer levels": pieces at the GT integer move PH1-3 by -0.03..+0.015, per-voxel GT integers reach 0.70-0.97).
   The fill ignores the merged phase: along each radial ray of the box (from M's axis, every native z plane, arcs <= 0.7 voxel apart) a
   merged span between its resolved ends a < b gets U(r) = U(a) + N (r - a) / (b - a), exactly N turns at a uniform spacing (the spacing
   prior), instead of the fitted U. N is what the oracle or the ref says the span holds:
     --mask gt   spans = runs of GT intervals (consecutive GT sheet crossings of the codes along the ray, one turn apart) across which the
                 fitted U changes < 1/2 turn (mergeflag's definition of a merged interval); a, b the bounding GT sheets
     --mask mf   spans = runs of cells with the fields' merge-flag plane mf >= --mf-thr; a, b the first unflagged samples outside
   modes (the U written where a span is filled; elsewhere the fitted U):
     base    the fitted U on the raster (the reference row: the scorer reads the raster voxel, not the model)
     interp  U(a) + (U(b) - U(a)) t: the fitted count, spread uniformly (control: no new turns)
     fill    U(a) + s N_gt t, N_gt the GT turns between a and b (a, b inside one GT chain), s the ray's sense of U
     shape   U(a) + s (U_gt(r) - U_gt(a)) (U_gt linear between the GT sheets): the GT's own spacing (the fill's upper bound)
     ref     U(a) + (U_ref(b) - U_ref(a)) t with --ref R (a model / set, e.g. the mf-weighted count-field): N from outside the box
     fillc, refc   fill / ref, and every sample beyond b on the ray shifted by the span's added turns (the count carried outward)
   Output: OUT/U.f32 (float32 [z][y][x], box-local native voxels of the fields), sweep.json + axis.json for `ufsm wind-score OUT CODES KEY`
   (the fitter's --codes score of the raster), and a line of span statistics. CPU only. */
#include "fit.h"
#include "json.h"
#include "optcheck.h"
#include "store.h"
#include "zarr3.h"
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static const char *fo_opt(int argc, char **argv, const char *k, const char *d) { for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], opt_n(k))) return argv[i + 1]; return d; }
static double fo_lerp(const float *v, int n, double z) { if (z <= 0) return v[0]; if (z >= n - 1) return v[n - 1]; const int i = (int)z; const double t = z - i; return v[i] + t * (v[i + 1] - v[i]); }

enum { M_BASE, M_INTERP, M_FILL, M_SHAPE, M_REF, M_FILLC, M_REFC, M_SHAPEC };
typedef struct {
    int n[3]; double off[3];                 /* raster shape (native voxels), the box origin in the model's coordinates */
    const float *U0, *Ur; const uint8_t *mf; int h, mfthr, gtmask, mode; double tol;
    const uint8_t *codes; int64_t clo[3], cn[3]; double cf, org[3];   /* the GT codes of the box (label grid), label voxel in native voxels */
    const fit_model *ax;                     /* the axis (ours' global model) */
    float *sum; uint16_t *cnt;               /* the fill: per raster voxel */
    _Atomic int next; pthread_mutex_t mu;
    long long spans, spans_n, spans_skip, rays, samp, samp_fill, chint; double sumN, sumN0, chfit;
} fo_ctx;

typedef struct { float r; int32_t vox; uint8_t code, mf; } fo_s;

/* a ray of plane z writes only plane z's voxels, and a thread owns whole planes: no atomics */
static void fo_put(fo_ctx *c, int32_t v, float u) { c->sum[v] += u; if (c->cnt[v] < UINT16_MAX) c->cnt[v]++; }
typedef struct { long long spans, filled, skip, samp, chint; double N, N0, chfit; } fo_st;

static void fo_ray(fo_ctx *c, const fo_s *S, int ns, double *ug, int *chain, int *cr, float *U, fo_st *st) {
    /* GT crossings: the middle sample of each run of labelled samples */
    int nc = 0;
    for (int i = 0; i < ns;) {
        if (S[i].code < 1 || S[i].code > 252) { i++; continue; }
        int j = i; while (j < ns && S[j].code >= 1 && S[j].code <= 252) j++;
        cr[nc++] = (i + j - 1) / 2; i = j;
    }
    /* chains: consecutive crossings one turn apart (codes mod 14), one sign; U_gt at each crossing (chain id per crossing, -1 alone) */
    double *ugc = ug + ns; int id = 0;
    for (int k = 0; k < nc; k++) chain[k] = -1;
    for (int k = 0; k + 1 < nc;) {
        const int s0 = k; double sg = 0; ugc[k] = 0;
        while (k + 1 < nc) {
            double dw = (S[cr[k + 1]].code - S[cr[k]].code) / 18.0; dw = fmod(dw + 7 + 140, 14) - 7;
            if (!(fabs(dw) > 0.6 && fabs(dw) < 1.4)) break;
            const double sgn = dw > 0 ? 1 : -1; if (sg != 0 && sgn != sg) break;
            sg = sgn; ugc[k + 1] = ugc[k] + sgn; k++;
        }
        if (k > s0) { for (int q = s0; q <= k; q++) chain[q] = id; id++; }
        k++;   /* (a crossing ends one chain at most: the next starts after it) */
    }
    /* U_gt per sample (NaN outside a chain): linear between consecutive crossings of one chain */
    for (int i = 0; i < ns; i++) ug[i] = NAN;
    for (int k = 0; k + 1 < nc; k++) {
        if (chain[k] < 0 || chain[k] != chain[k + 1]) continue;
        const int a = cr[k], b = cr[k + 1];
        for (int i = a; i <= b; i++) ug[i] = ugc[k] + (ugc[k + 1] - ugc[k]) * (S[i].r - S[a].r) / fmax(1e-6, S[b].r - S[a].r);
    }
    for (int i = 0; i < ns; i++) U[i] = c->U0[S[i].vox];
    const double su = U[ns - 1] - U[0] >= 0 ? 1 : -1;   /* the ray's sense of the fitted U */
    for (int k = 0; k + 1 < nc; k++) if (chain[k] >= 0 && chain[k] == chain[k + 1]) { st->chint++; st->chfit += su * (U[cr[k + 1]] - U[cr[k]]); }
    /* spans [a, b] (sample indices of the resolved ends) and the last sample their carry reaches (the chain's end; the ray's for mf) */
    int sp[4096][3]; int nsp = 0;
    if (c->gtmask == 2) {   /* --mask chain: every GT chain one span (its first and last sheet the ends: the spacing prior over the longest
                               spans the GT gives, i.e. resolved ends far apart) */
        for (int k = 0; k < nc && nsp < 4096;) {
            if (chain[k] < 0) { k++; continue; }
            int e = k; while (e + 1 < nc && chain[e + 1] == chain[k]) e++;
            if (e > k + 1) { sp[nsp][0] = cr[k]; sp[nsp][1] = cr[e]; sp[nsp][2] = cr[e]; nsp++; }
            k = e + 1;
        }
    } else if (c->gtmask) {   /* resolved crossings: every chain interval beside them counts one turn (|s dU - 1| < T); spans between two resolved
                          crossings of a chain that are not neighbours (an interval between them miscounts) */
        int prev = -1;
        for (int k = 0; k < nc && nsp < 4096; k++) {
            if (chain[k] < 0) { prev = -1; continue; }
            const int hl = k > 0 && chain[k - 1] == chain[k], hr = k + 1 < nc && chain[k + 1] == chain[k];
            const int okl = !hl || fabs(su * (U[cr[k]] - U[cr[k - 1]]) - 1) < c->tol, okr = !hr || fabs(su * (U[cr[k + 1]] - U[cr[k]]) - 1) < c->tol;
            if (!hl) prev = -1;
            if (!(okl && okr) || (!hl && !hr)) continue;
            if (prev >= 0 && k > prev + 1) {
                int e = k; while (e + 1 < nc && chain[e + 1] == chain[k]) e++;
                sp[nsp][0] = cr[prev]; sp[nsp][1] = cr[k]; sp[nsp][2] = cr[e]; nsp++;
            }
            prev = k;
        }
    } else {
        for (int i = 1; i < ns && nsp < 4096; i++) {
            if (S[i].mf < c->mfthr || S[i - 1].mf >= c->mfthr) continue;
            int j = i; while (j < ns && S[j].mf >= c->mfthr) j++;
            if (j < ns) { sp[nsp][0] = i - 1; sp[nsp][1] = j; sp[nsp][2] = ns - 1; nsp++; }
            i = j;
        }
    }
    const int carrying = c->mode == M_FILLC || c->mode == M_REFC || c->mode == M_SHAPEC;
    for (int s = 0; s < nsp; s++) {
        const int a = sp[s][0], b = sp[s][1], e = sp[s][2]; double N;
        st->spans++;
        if (c->mode == M_BASE) continue;
        if (c->mode == M_INTERP) N = U[b] - U[a];
        else if (c->mode == M_REF || c->mode == M_REFC) N = c->Ur[S[b].vox] - c->Ur[S[a].vox];
        else { if (!(ug[a] == ug[a]) || !(ug[b] == ug[b])) { st->skip++; continue; } N = su * fabs(ug[b] - ug[a]); }
        st->filled++; st->N += fabs(N); st->N0 += fabs(U[b] - U[a]);
        const double Ua = U[a], ra = S[a].r, rb = S[b].r, add = N - (U[b] - Ua);   /* add: the turns the span gains */
        for (int i = a + 1; i < b; i++) {
            const double t = (S[i].r - ra) / fmax(1e-6, rb - ra);
            U[i] = (float)(c->mode == M_SHAPE || c->mode == M_SHAPEC ? Ua + su * fabs(ug[i] - ug[a]) : Ua + N * t);
            st->samp++;
        }
        if (carrying) for (int i = b; i <= e; i++) U[i] += (float)add;   /* the count carried to the chain's end (later spans start from it) */
    }
    if (nsp && c->mode != M_BASE) { const int lo = sp[0][0] + 1, hi = carrying ? ns - 1 : sp[nsp - 1][1] - 1;
        for (int i = lo; i <= hi; i++) if (U[i] != c->U0[S[i].vox]) fo_put(c, S[i].vox, U[i]); }
}

static void *fo_worker(void *arg) {
    fo_ctx *c = arg; const int *n = c->n; const int na = (int)lround(c->ax->D * c->ax->h) + 1;
    const double dr = 0.5 * fmin(1.0, c->cf);
    const double rmx = 2 * hypot(n[1], n[2]) + 4; const int nmax = (int)(rmx / dr) + 8;
    fo_s *S = malloc((size_t)nmax * sizeof *S); double *ug = malloc((size_t)nmax * 2 * sizeof *ug); int *chain = malloc((size_t)nmax * sizeof *chain), *cr = malloc((size_t)nmax * sizeof *cr);
    float *U = malloc((size_t)nmax * sizeof *U); fo_st st = {0}; long long rays = 0, samp = 0;
    for (int z; (z = atomic_fetch_add(&c->next, 1)) < n[0];) {
        const double pz = z + 0.5, cy = fo_lerp(c->ax->cy, na, pz + c->off[0]) - c->off[1], cx = fo_lerp(c->ax->cx, na, pz + c->off[0]) - c->off[2];
        double r0 = 1e30, r1 = 0;   /* the box's radius range about the axis */
        for (int q = 0; q < 4; q++) { const double yy = q & 1 ? n[1] : 0, xx = q & 2 ? n[2] : 0; r1 = fmax(r1, hypot(yy - cy, xx - cx)); }
        { const double yc = fmin(fmax(cy, 0), n[1]), xc = fmin(fmax(cx, 0), n[2]); r0 = hypot(yc - cy, xc - cx); }
        const int nt = (int)ceil(2 * M_PI * r1 / 0.7);
        const int64_t lz = (int64_t)floor((pz + c->org[0]) / c->cf) - c->clo[0];
        for (int it = 0; it < nt; it++) {
            const double th = 2 * M_PI * (it + 0.5) / nt, sy = sin(th), sx = cos(th); int ns = 0;
            for (double r = r0; r <= r1 && ns < nmax; r += dr) {
                const double y = cy + r * sy, x = cx + r * sx;
                if (y < 0 || x < 0 || y >= n[1] || x >= n[2]) continue;
                const int iy = (int)y, ix = (int)x; fo_s s = {(float)r, (int32_t)(((int64_t)z * n[1] + iy) * n[2] + ix), 0, 0};
                const int64_t ly = (int64_t)floor((y + c->org[1]) / c->cf) - c->clo[1], lx = (int64_t)floor((x + c->org[2]) / c->cf) - c->clo[2];
                if (lz >= 0 && ly >= 0 && lx >= 0 && lz < c->cn[0] && ly < c->cn[1] && lx < c->cn[2]) s.code = c->codes[((size_t)lz * c->cn[1] + ly) * c->cn[2] + lx];
                if (c->mf) s.mf = c->mf[(((size_t)(z / c->h) * (n[1] / c->h)) + iy / c->h) * (n[2] / c->h) + ix / c->h];
                S[ns++] = s;
            }
            if (ns < 4) continue;
            rays++; samp += ns;
            fo_ray(c, S, ns, ug, chain, cr, U, &st);
        }
    }
    pthread_mutex_lock(&c->mu);
    c->spans += st.spans; c->spans_n += st.filled; c->spans_skip += st.skip; c->samp_fill += st.samp; c->sumN += st.N; c->sumN0 += st.N0; c->chint += st.chint; c->chfit += st.chfit;
    c->rays += rays; c->samp += samp;
    pthread_mutex_unlock(&c->mu);
    free(S); free(ug); free(chain); free(cr); free(U);
    return nullptr;
}

int cmd_fill_oracle(int argc, char **argv) {
    const char *fd = fo_opt(argc, argv, "--fields", nullptr), *md = fo_opt(argc, argv, "--model", nullptr), *od = fo_opt(argc, argv, "--out", nullptr);
    const char *rd = fo_opt(argc, argv, "--ref", nullptr), *ms = fo_opt(argc, argv, "--mode", "fill"), *mk = fo_opt(argc, argv, "--mask", "gt");
    int ci = -1; for (int i = 2; i + 2 < argc; i++) if (!strcmp(argv[i], opt_n("--codes"))) ci = i;
    static const char *names[] = {"base", "interp", "fill", "shape", "ref", "fillc", "refc", "shapec"}; int mode = -1;   /* mgpu: debug (fill-oracle mode names, read-only) */
    for (int k = 0; k < 8; k++) if (!strcmp(ms, names[k])) mode = k;
    if (!fd || !md || !od || ci < 0 || mode < 0 || (strcmp(mk, "gt") && strcmp(mk, "mf") && strcmp(mk, "chain")) || ((mode == M_REF || mode == M_REFC) && !rd)) {
        fprintf(stderr, "usage: ufsm fill-oracle --fields DIR --model M --codes ROOT KEY --out DIR [--mode base|interp|fill|shape|ref|fillc|refc]"
                        " [--mask gt|mf] [--mf-thr 179] [--ref R] [--threads N]\n"); return 2; }
    const char *croot = argv[ci + 1], *ckey = argv[ci + 2];
    char p[2048]; snprintf(p, sizeof p, "%s/fields.json", fd);
    FILE *f = fopen(p, "rb"); if (!f) { fprintf(stderr, "fill-oracle: cannot read %s\n", p); return 1; }
    fseek(f, 0, SEEK_END); const long jl = ftell(f); fseek(f, 0, SEEK_SET); char *jb = malloc((size_t)jl + 1);
    if (!jb || fread(jb, 1, (size_t)jl, f) != (size_t)jl) { fclose(f); fprintf(stderr, "fill-oracle: cannot read %s\n", p); return 1; }
    fclose(f); jb[jl] = 0; json *j = json_parse(jb, (size_t)jl); free(jb); if (!j) { fprintf(stderr, "fill-oracle: bad %s\n", p); return 1; }
    long long org[3]; int shp[3]; const int h = (int)json_num(json_get(j, "h"), 1); const double um = json_num(json_get(j, "um"), 2.4);
    for (int d = 0; d < 3; d++) { org[d] = (long long)json_num(json_at(json_get(j, "origin"), d), 0); shp[d] = (int)json_num(json_at(json_get(j, "shape"), d), 0); }
    json_free(j);
    fo_ctx c = {.h = h, .mfthr = opt_atoi("--mf-thr", fo_opt(argc, argv, "--mf-thr", "179")), .gtmask = !strcmp(mk, "gt") ? 1 : !strcmp(mk, "chain") ? 2 : 0, .mode = mode, .tol = opt_atof("--tol", fo_opt(argc, argv, "--tol", "0.3"))};
    pthread_mutex_init(&c.mu, nullptr);
    for (int d = 0; d < 3; d++) { c.n[d] = shp[d] * h; c.org[d] = (double)org[d]; }
    fit_set *fs = fit_set_load(md); if (!fs) { fprintf(stderr, "fill-oracle: cannot load the model %s\n", md); return 1; }
    for (int d = 0; d < 3; d++) c.off[d] = (double)(org[d] - fs->global->origin[d]);
    c.ax = fs->global;
    const size_t nv = (size_t)c.n[0] * c.n[1] * c.n[2];
    float *U0 = malloc(nv * 4); if (!U0) { fprintf(stderr, "fill-oracle: no memory for the raster (%zu voxels)\n", nv); return 1; }
    const int64_t n64[3] = {c.n[0], c.n[1], c.n[2]};
    fit_set_grid_cpu(fs, c.off, n64, 1.0, 1, nullptr, U0); c.U0 = U0;   /* U at box-local native (i + 1/2) */
    float *Ur = nullptr; fit_set *rs = nullptr;
    if (rd && (mode == M_REF || mode == M_REFC)) {
        if (!(rs = fit_set_load(rd))) { fprintf(stderr, "fill-oracle: cannot load the ref %s\n", rd); return 1; }
        double ro[3]; for (int d = 0; d < 3; d++) ro[d] = (double)(org[d] - rs->global->origin[d]);
        Ur = malloc(nv * 4); if (!Ur) return 1; fit_set_grid_cpu(rs, ro, n64, 1.0, 1, nullptr, Ur); c.Ur = Ur;
    }
    uint8_t *mf = nullptr;
    if (!c.gtmask) {
        int ms3[3]; if (!(mf = fit_field_array(fd, "mf", 1, ms3))) { fprintf(stderr, "fill-oracle: --mask mf: no mf plane in %s (ufsm merge-flag)\n", fd); return 1; }
        if (ms3[0] != shp[0] || ms3[1] != shp[1] || ms3[2] != shp[2]) { fprintf(stderr, "fill-oracle: mf shape differs from the fields'\n"); return 1; }
        c.mf = mf;
    }
    /* the GT codes over the box (label grid) */
    z3 *zc = z3_open(store_open(croot), ckey, nullptr); if (!zc) { fprintf(stderr, "fill-oracle: codes %s/%s: %s\n", croot, ckey, z3_error()); return 1; }
    c.cf = atof(ckey) > 0 ? atof(ckey) / um : 1;
    for (int d = 0; d < 3; d++) { c.clo[d] = (int64_t)floor(org[d] / c.cf) - 1; c.cn[d] = (int64_t)ceil(c.n[d] / c.cf) + 3; }
    uint8_t *codes = malloc((size_t)c.cn[0] * c.cn[1] * c.cn[2]); if (!codes) return 1;
    int nth = opt_atoi("--threads", fo_opt(argc, argv, "--threads", "8")); if (nth < 1) nth = 1; if (nth > 64) nth = 64;
    if (z3_read(zc, c.clo, c.cn, codes, nth)) { fprintf(stderr, "fill-oracle: codes read failed: %s\n", z3_error()); return 1; }
    z3_close(zc); c.codes = codes;
    c.sum = calloc(nv, 4); c.cnt = calloc(nv, 2); if (!c.sum || !c.cnt) { fprintf(stderr, "fill-oracle: no memory\n"); return 1; }
    pthread_t th[64]; for (int t = 0; t < nth; t++) pthread_create(&th[t], nullptr, fo_worker, &c); for (int t = 0; t < nth; t++) pthread_join(th[t], nullptr);
    size_t nfill = 0;
    for (size_t i = 0; i < nv; i++) if (c.cnt[i]) { U0[i] = c.sum[i] / c.cnt[i]; nfill++; }
    printf("fill-oracle: GT chain intervals %lld (one turn each), the fitted U across them %.3f turns each\n", c.chint, c.chint ? c.chfit / c.chint : 0);
    printf("fill-oracle %s (mask %s): %lld rays, %lld samples; %lld spans (%lld filled, %lld without GT ends), mean |N| %.3f turns vs fitted %.3f;"
           " %lld span samples, %zu voxels written (%.4f of the box)\n", names[mode], mk, c.rays, c.samp, c.spans, c.spans_n, c.spans_skip,
           c.spans_n ? c.sumN / c.spans_n : 0, c.spans_n ? c.sumN0 / c.spans_n : 0, c.samp_fill, nfill, (double)nfill / nv);
    mkdir(od, 0755);
    snprintf(p, sizeof p, "%s/U.f32", od); f = fopen(p, "wb");
    if (!f || fwrite(U0, 4, nv, f) != nv || fclose(f)) { fprintf(stderr, "fill-oracle: cannot write %s\n", p); return 1; }
    /* the axis in the raster's global voxels (wind-score: scale 1) and sweep.json */
    snprintf(p, sizeof p, "%s/axis.json", od); f = fopen(p, "w"); if (!f) return 1;
    const int na = (int)lround(c.ax->D * c.ax->h) + 1;
    fprintf(f, "{\"control_points\": [");
    for (int z = 0; z <= c.n[0]; z++) fprintf(f, "%s{\"z\": %.3f, \"y\": %.4f, \"x\": %.4f}", z ? ", " : "", z + c.org[0],
                                              fo_lerp(c.ax->cy, na, z + c.off[0]) - c.off[1] + c.org[1], fo_lerp(c.ax->cx, na, z + c.off[0]) - c.off[2] + c.org[2]);
    fprintf(f, "]}\n"); if (fclose(f)) return 1;
    snprintf(p, sizeof p, "%s/sweep.json", od); f = fopen(p, "w"); if (!f) return 1;
    fprintf(f, "{\"shape\": [%d, %d, %d], \"origin_zyx\": [%lld, %lld, %lld], \"um\": %.6f, \"scale\": 1, \"axis\": \"%s/axis.json\"}\n",
            c.n[0], c.n[1], c.n[2], org[0], org[1], org[2], um, od);
    if (fclose(f)) return 1;
    free(U0); free(Ur); free(mf); free(codes); free(c.sum); free(c.cnt);
    return 0;
}
