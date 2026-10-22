#include "sources.h"
#include "json.h"
#include "votes.h"
#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include <stdatomic.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

static const char *g_cache;   /* chunk cache dir for HTTPS stores; from the sources file or sources_set_cache */

int rung_of_um(double um) { return (int)lround(log2(um / 0.6)); }

/* ---- axis ---- */

static int cmp_z(const void *a, const void *b) {
    const double *p = a, *q = b;
    return (p[0] > q[0]) - (p[0] < q[0]);
}

int axis_load(axis *a, const char *path) {
    memset(a, 0, sizeof *a);
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *txt = malloc((size_t)n + 1);
    if (fread(txt, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(txt); return -1; }
    fclose(f);
    json *j = json_parse(txt, (size_t)n);
    free(txt);
    const json *cp = json_get(j, "control_points");
    if (!cp) cp = j && j->type == J_ARR ? j : nullptr;
    if (!cp || cp->n == 0) { json_free(j); return -1; }
    double (*pts)[3] = malloc(cp->n * sizeof *pts);
    for (size_t i = 0; i < cp->n; i++) {
        const json *p = json_at(cp, i);
        pts[i][0] = json_num(json_get(p, "z"), 0);
        pts[i][1] = json_num(json_get(p, "y"), 0);
        pts[i][2] = json_num(json_get(p, "x"), 0);
    }
    qsort(pts, cp->n, sizeof *pts, cmp_z);
    a->n = (int)cp->n;
    a->z = malloc(cp->n * sizeof(double));
    a->y = malloc(cp->n * sizeof(double));
    a->x = malloc(cp->n * sizeof(double));
    for (size_t i = 0; i < cp->n; i++) { a->z[i] = pts[i][0]; a->y[i] = pts[i][1]; a->x[i] = pts[i][2]; }
    free(pts);
    json_free(j);
    return 0;
}

void axis_at(const axis *a, double z, double *y, double *x) {
    if (a->n == 0) { *y = *x = 0; return; }
    if (z <= a->z[0]) { *y = a->y[0]; *x = a->x[0]; return; }
    if (z >= a->z[a->n - 1]) { *y = a->y[a->n - 1]; *x = a->x[a->n - 1]; return; }
    int lo = 0, hi = a->n - 1;
    while (hi - lo > 1) { int m = (lo + hi) / 2; if (a->z[m] <= z) lo = m; else hi = m; }
    double t = a->z[hi] > a->z[lo] ? (z - a->z[lo]) / (a->z[hi] - a->z[lo]) : 0;
    *y = a->y[lo] + t * (a->y[hi] - a->y[lo]);
    *x = a->x[lo] + t * (a->x[hi] - a->x[lo]);
}

/* ---- sources ---- */

static const char *CHNAME[NCH] = {"recto", "sheet"};

static char *sdup(const char *s) { return s ? strdup(s) : nullptr; }

sources *sources_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "sources: cannot open %s\n", path); return nullptr; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *txt = malloc((size_t)n + 1);
    if (fread(txt, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(txt); return nullptr; }
    fclose(f);
    json *j = json_parse(txt, (size_t)n);
    free(txt);
    const json *arr = json_get(j, "sources");
    if (!arr || arr->type != J_ARR) { fprintf(stderr, "sources: no \"sources\" array in %s\n", path); json_free(j); return nullptr; }
    sources *S = calloc(1, sizeof *S);
    S->cache = sdup(json_str(json_get(j, "cache"), nullptr));
    if (S->cache) g_cache = S->cache;
    S->n = (int)arr->n;
    S->src = calloc(arr->n, sizeof *S->src);
    for (size_t i = 0; i < arr->n; i++) {
        const json *e = json_at(arr, i);
        source *s = &S->src[i];
        s->name = sdup(json_str(json_get(e, "name"), "?"));
        s->s = store_open(json_str(json_get(e, "root"), "."));
        s->ct_key = sdup(json_str(json_get(e, "ct"), ""));
        s->um = json_num(json_get(e, "um"), 0);
        s->weight = json_num(json_get(e, "weight"), 1.0);
        s->trust_band = (int)json_num(json_get(e, "trust_band"), 0);
        s->iso = (int)json_num(json_get(e, "iso"), 0);
        s->head = (int)json_num(json_get(e, "head"), 0);
        if (s->head < 0 || s->head > 6) { fprintf(stderr, "sources: %s: head 0..6\n", s->name); json_free(j); sources_free(S); return nullptr; }
        s->min_level = (int)json_num(json_get(e, "min_level"), 0);
        s->max_level = (int)json_num(json_get(e, "max_level"), -1);
        s->geometry = (int)json_num(json_get(e, "geometry"), 1);
        for (int l = 0; l < MAXLEV; l++) s->ct_present[l] = -1; /* unknown until probed */
        const json *tg = json_get(e, "targets");
        for (int c = 0; c < NCH; c++) {
            const json *t = json_get(tg, CHNAME[c]);
            if (!t && c == CH_SHEET) t = json_get(tg, "ink");   /* task unwrap: the ink teacher (prob) in target slot 1 */
            for (int l = 0; l < MAXLEV; l++) s->tgt_present[c][l] = -1;
            if (!t) continue;
            if (t->type == J_STR) { s->tgt_key[c] = strdup(t->str); continue; }
            s->tgt_binary[c] = !strcmp(json_str(json_get(t, "encoding"), ""), "binary");
            s->tgt_prob254[c] = !strcmp(json_str(json_get(t, "encoding"), ""), "prob254");   /* p * 254, 255 = unknown: read as stored */
            s->tgt_prob[c] = !strcmp(json_str(json_get(t, "encoding"), ""), "prob") || s->tgt_prob254[c];
            s->tgt_votes[c] = !strcmp(json_str(json_get(t, "encoding"), ""), "side-votes");
            s->tgt_side[c] = !strcmp(json_str(json_get(t, "encoding"), ""), "side") || s->tgt_votes[c];
            if (s->tgt_votes[c]) {   /* src/votes.h: a side array is a vote array of one set, so either kind of array works here */
                const json *vj = json_get(t, "votes");
                const char *rule = json_str(json_get(vj, "rule"), "unanimous");
                const double vmin = json_num(json_get(vj, "min"), 1), soft = json_num(json_get(vj, "soft"), 0);
                const int majority = !strcmp(rule, "majority");
                if ((!majority && strcmp(rule, "unanimous")) || !(vmin >= 1 && vmin <= VOTE_MAX) || vmin != (int)vmin || (soft != 0 && soft != 1)) {
                    fprintf(stderr, "sources: %s: \"votes\": {\"rule\": \"unanimous\" | \"majority\", \"min\": 1..15, \"soft\": 0 | 1}\n", s->name);
                    json_free(j); sources_free(S); return nullptr;
                }
                s->tgt_votes_soft[c] = soft != 0;
                if (soft) for (int k = 0; k < 256; k++) {
                    int r, v; vote_split((uint8_t)k, &r, &v);
                    s->tgt_vote_lut[c][k] = r + v >= (int)vmin ? (uint8_t)((254 * r + (r + v) / 2) / (r + v)) : 255;
                } else vote_side_lut(majority ? VOTE_MAJORITY : VOTE_UNANIMOUS, (int)vmin, s->tgt_vote_lut[c]);
            }
            if (s->tgt_side[c] && (c != CH_RECTO || s->iso <= 0)) {
                fprintf(stderr, "sources: %s: \"encoding\": \"side\" is a recto target and needs the source's void cutoff \"iso\"\n", s->name); json_free(j); sources_free(S); return nullptr;
            }
            double target_min = json_num(json_get(t, "min_level"), 0);
            if (!isfinite(target_min) || target_min < 0 || target_min >= MAXLEV || target_min != (int)target_min ||
                (target_min && !s->tgt_binary[c] && !s->tgt_prob[c] && !s->tgt_side[c])) {
                fprintf(stderr, "sources: %s/%s: invalid target min_level (binary or prob targets only)\n", s->name, CHNAME[c]);
                json_free(j); sources_free(S); return nullptr;
            }
            s->tgt_min_level[c] = (int)target_min;
            s->tgt_erode[c] = (int)json_num(json_get(t, "erode"), 0); s->tgt_dilate[c] = (int)json_num(json_get(t, "dilate"), 0);
            if (s->tgt_erode[c] < 0 || s->tgt_erode[c] > 84 || s->tgt_dilate[c] < 0 || s->tgt_dilate[c] > 84 || ((s->tgt_erode[c] || s->tgt_dilate[c]) && !s->tgt_binary[c])) {
                fprintf(stderr, "sources: %s/%s: erode / dilate 0..84 voxels, binary targets only\n", s->name, CHNAME[c]); json_free(j); sources_free(S); return nullptr;
            }
            const char *troot = json_str(json_get(t, "root"), nullptr);
            if (troot) s->tgt_store[c] = store_open(troot);
            const char *grp = json_str(json_get(t, "group"), nullptr);
            if (grp) { s->tgt_key[c] = strdup(grp); continue; }
            const json *org = json_get(t, "origins");
            if (!org) { fprintf(stderr, "sources: %s/%s: need a string, {root, group} or {regions|array, size, origins}\n", s->name, CHNAME[c]); continue; }
            regions *r = calloc(1, sizeof *r);
            r->dir = sdup(json_str(json_get(t, "regions"), ""));
            r->array = sdup(json_str(json_get(t, "array"), nullptr));
            r->size = (int)json_num(json_get(t, "size"), 1024);
            r->n = (int)org->n;
            r->origin = calloc(org->n, sizeof *r->origin);
            r->rsize = calloc(org->n, sizeof *r->rsize);
            for (size_t k = 0; k < org->n; k++) {
                for (int d = 0; d < 3; d++) r->origin[k][d] = (int64_t)json_num(json_at(json_at(org, k), (size_t)d), 0);
                r->rsize[k] = (int)json_num(json_at(json_at(org, k), 3), r->size);
            }
            s->reg[c] = r;
            s->reg_z[c] = calloc(org->n, sizeof(z3 *));
        }
        const json *bt = json_get(tg, "band");
        if (bt) {
            const char *broot = json_str(json_get(bt, "root"), nullptr), *bkey = json_str(json_get(bt, "key"), "4.8");
            s->band_radius = (float)json_num(json_get(bt, "radius"), 80); s->band_span = (float)json_num(json_get(bt, "span"), 75);
            store *bs = broot ? store_open(broot) : nullptr;
            s->band = bs ? z3_open(bs, bkey, nullptr) : nullptr;
            if (!s->band || !(s->band_radius > 0) || !(s->band_span >= 0)) {
                fprintf(stderr, "sources: %s: cannot open band target %s/%s (%s)\n", s->name, broot ? broot : "?", bkey, z3_error());
                json_free(j); sources_free(S); return nullptr;
            }
            const json *pt = json_get(bt, "phase");
            if (pt) {
                const char *proot = json_str(json_get(pt, "root"), nullptr), *pkey = json_str(json_get(pt, "key"), "4.8");
                store *ps = proot ? store_open(proot) : nullptr;
                s->phase = ps ? z3_open(ps, pkey, nullptr) : nullptr;
                s->phase_level = (int)json_num(json_get(pt, "level"), 1); s->phase_direct = json_get(pt, "level") != nullptr;
                const double sg = json_num(json_get(pt, "phase_sign"), 1);
                if (sg != 1 && sg != -1) { fprintf(stderr, "sources: %s: \"phase_sign\" must be 1 or -1\n", s->name); json_free(j); sources_free(S); return nullptr; }
                s->phase_sign = (int)sg;
                const z3_meta *bm = z3_meta_of(s->band), *pm = s->phase ? z3_meta_of(s->phase) : nullptr;
                const int sh = s->phase_level - 1;   /* the label grid is level 1: shapes halve per level above it */
                if (!pm || sh < 0 || sh > 4 || pm->shape[0] != (bm->shape[0] + (1 << sh) - 1) >> sh || pm->shape[1] != (bm->shape[1] + (1 << sh) - 1) >> sh || pm->shape[2] != (bm->shape[2] + (1 << sh) - 1) >> sh) {
                    fprintf(stderr, "sources: %s: cannot open the stored phase %s/%s or its shape differs from the band codes'\n", s->name, proot ? proot : "?", pkey);
                    json_free(j); sources_free(S); return nullptr;
                }
                const json *wt = json_get(bt, "winding");   /* the winding mod 14 on the stored phase's grid */
                if (wt) {
                    const char *wroot = json_str(json_get(wt, "root"), nullptr), *wkey = json_str(json_get(wt, "key"), pkey);
                    store *ws = wroot ? store_open(wroot) : nullptr;
                    s->wind = ws ? z3_open(ws, wkey, nullptr) : nullptr;
                    const z3_meta *wm = s->wind ? z3_meta_of(s->wind) : nullptr;
                    if (!wm || memcmp(wm->shape, pm->shape, sizeof wm->shape)) {
                        fprintf(stderr, "sources: %s: cannot open the winding raster %s/%s or its shape differs from the stored phase's\n", s->name, wroot ? wroot : "?", wkey);
                        json_free(j); sources_free(S); return nullptr;
                    }
                }
            }
        }
        const json *aj = json_get(e, "aux");
        if (aj) {
            const char *ar = json_str(json_get(aj, "root"), nullptr), *ak = json_str(json_get(aj, "key"), "9.6");
            store *as = ar ? store_open(ar) : nullptr;
            s->aux = as ? z3_open(as, ak, nullptr) : nullptr; s->aux_level = (int)json_num(json_get(aj, "level"), 2);
            if (!s->aux) { fprintf(stderr, "sources: %s: cannot open the aux volume %s/%s\n", s->name, ar ? ar : "?", ak); json_free(j); sources_free(S); return nullptr; }
        }
        const json *fj = json_get(e, "focus");   /* {"map": RAW, "w": K}: the sampler's packing focus (map grid in RAW.json) */
        if (fj) {
            const char *fp = json_str(json_get(fj, "map"), nullptr); s->focus_w = (float)json_num(json_get(fj, "w"), 0);
            char jp[1200]; snprintf(jp, sizeof jp, "%s.json", fp ? fp : "");
            FILE *jf = fp ? fopen(jp, "rb") : nullptr; char jb[4096] = {0}; size_t jn = jf ? fread(jb, 1, sizeof jb - 1, jf) : 0; if (jf) fclose(jf);
            json *g = jn ? json_parse(jb, jn) : nullptr; size_t tot = 1;
            for (int d = 0; d < 3; d++) { s->focus_grid[d] = g ? (int64_t)json_num(json_at(json_get(g, "grid"), (size_t)d), 0) : 0; tot *= (size_t)(s->focus_grid[d] > 0 ? s->focus_grid[d] : 0); }
            json_free(g);
            FILE *mf = tot && fp ? fopen(fp, "rb") : nullptr;
            s->focus = mf ? malloc(tot) : nullptr;
            if (!s->focus || fread(s->focus, 1, tot, mf) != tot || !(s->focus_w >= 0)) {
                fprintf(stderr, "sources: %s: cannot read the focus map %s (and %s with its grid), or w < 0\n", s->name, fp ? fp : "?", jp);
                if (mf) { fclose(mf); } json_free(j); sources_free(S); return nullptr;
            }
            fclose(mf);
        }
        const json *mj = json_get(e, "mask");   /* {"map": RAW, "min": 2}: ufsm scanprep's cell mask (grid and lev in RAW.json) */
        if (mj) {
            const char *mp = json_str(json_get(mj, "map"), nullptr); s->mask_min = (int)json_num(json_get(mj, "min"), 2);
            char jp[1200]; snprintf(jp, sizeof jp, "%s.json", mp ? mp : "");
            FILE *jf = mp ? fopen(jp, "rb") : nullptr; char jb[4096] = {0}; size_t jn = jf ? fread(jb, 1, sizeof jb - 1, jf) : 0; if (jf) fclose(jf);
            json *g = jn ? json_parse(jb, jn) : nullptr; size_t tot = 1;
            for (int d = 0; d < 3; d++) { s->mask_grid[d] = g ? (int64_t)json_num(json_at(json_get(g, "grid"), (size_t)d), 0) : 0; tot *= (size_t)(s->mask_grid[d] > 0 ? s->mask_grid[d] : 0); }
            s->mask_lev = g ? (int)json_num(json_get(g, "lev"), -1) : -1;
            json_free(g);
            FILE *mf = tot && mp && s->mask_lev >= 0 ? fopen(mp, "rb") : nullptr;
            s->mask = mf ? malloc(tot) : nullptr;
            if (!s->mask || fread(s->mask, 1, tot, mf) != tot || tot >= 4294967295u) {
                fprintf(stderr, "sources: %s: cannot read the mask %s (and %s with its grid and lev)\n", s->name, mp ? mp : "?", jp);
                if (mf) { fclose(mf); } json_free(j); sources_free(S); return nullptr;
            }
            fclose(mf);
        }
        s->ct_only = (int)json_num(json_get(e, "ct_only"), 0);
        const json *nj = json_get(e, "norm");   /* {"center": c, "scale": s}: a fixed per-scan CT normalisation instead of the window z-score */
        if (nj) {
            s->norm_c = (float)json_num(json_get(nj, "center"), 0); s->norm_s = (float)json_num(json_get(nj, "scale"), 0);
            if (!(s->norm_s > 0)) { fprintf(stderr, "sources: %s: \"norm\" needs \"scale\" > 0\n", s->name); json_free(j); sources_free(S); return nullptr; }
        }
        const json *ho = json_get(e, "holdout");
        if (ho && ho->type == J_ARR && ho->n == 6) for (int d = 0; d < 3; d++) { s->hold_o[d] = (int64_t)json_num(json_at(ho, (size_t)d), 0); s->hold_n[d] = (int64_t)json_num(json_at(ho, (size_t)(3 + d)), 0); }
        const json *bx = json_get(e, "boxes");   /* task restore: level-0 boxes [z, y, x, nz, ny, nx] that hold local CT (training windows lie inside one) */
        if (bx && bx->type == J_ARR && bx->n) {
            s->box = calloc(bx->n, sizeof *s->box);
            for (size_t k = 0; k < bx->n; k++) { const json *b = json_at(bx, k); if (!b || b->type != J_ARR || b->n != 6) continue;
                for (int d = 0; d < 6; d++) s->box[s->nbox][d] = (int64_t)json_num(json_at(b, (size_t)d), 0);
                s->nbox++; }
        }
        const char *ap = json_str(json_get(e, "axis"), nullptr);
        if (ap && axis_load(&s->ax, ap)) fprintf(stderr, "sources: %s: cannot load axis %s\n", s->name, ap);
        if (s->um <= 0) {
            z3 *z0 = source_ct(s, 0);
            if (z0) s->um = z3_meta_of(z0)->scale_um;
        }
        if (s->um <= 0) fprintf(stderr, "sources: %s: unknown voxel size (set \"um\")\n", s->name);
    }
    json_free(j);
    return S;
}

void sources_free(sources *S) {
    if (!S) return;
    for (int i = 0; i < S->n; i++) {
        source *s = &S->src[i];
        for (int l = 0; l < MAXLEV; l++) z3_close(s->ct[l]);
        z3_close(s->band); z3_close(s->phase); z3_close(s->wind); z3_close(s->aux); free(s->box); free(s->focus); free(s->mask);
        for (int c = 0; c < NCH; c++) {
            for (int l = 0; l < MAXLEV; l++) z3_close(s->tgt[c][l]);
            if (s->reg[c]) {
                for (int k = 0; k < s->reg[c]->n; k++) z3_close(s->reg_z[c][k]);
                free(s->reg_z[c]);
                z3_close(s->reg_shared[c]);
                free(s->reg[c]->dir);
                free(s->reg[c]->array);
                free(s->reg[c]->origin);
                free(s->reg[c]->rsize);
                free(s->reg[c]);
            }
            free(s->tgt_key[c]);
            store_close(s->tgt_store[c]);
        }
        free(s->ax.z); free(s->ax.y); free(s->ax.x);
        free(s->name); free(s->ct_key);
        store_close(s->s);
    }
    free(S->src);
    free(S->cache);
    free(S);
}

/* open level `level` of a pyramid group: try the integer name, then the OME level whose voxel size is um0 * 2^level */
z3 *pyramid_open_level(store *st, const char *group, int level, double um0, const char *cache) {
    char k[1024];
    z3_level lv[16];
    int nl = z3_group_levels(st, group, lv, 16);
    if (nl > 0 && um0 > 0 && lv[0].um > 0) {        /* OME group: match by voxel size (absolute um, or relative 2^l factors) */
        double want = um0 * (1 << level), rel = (double)(1 << level);
        for (int i = 0; i < nl; i++)
            if (fabs(lv[i].um - want) < 0.02 * want || (lv[0].um == 1.0 && fabs(lv[i].um - rel) < 0.02 * rel)) { snprintf(k, sizeof k, "%s/%s", group, lv[i].path); return z3_open(st, k, cache); }
        return nullptr;
    }
    snprintf(k, sizeof k, "%s/%d", group, level);   /* plain integer level names */
    return z3_open(st, k, cache);
}
/* does the group's multiscales metadata list a dataset for this level (by voxel size), or a plain integer level key exist?
   Used only to decide whether an open failure is worth retrying. */
static int level_listed(store *st, const char *group, int level, double um0) {
    z3_level lv[16]; int nl = z3_group_levels(st, group, lv, 16);
    if (nl <= 0 || um0 <= 0 || lv[0].um <= 0) return level < 8;
    double want = um0 * (1 << level), rel = (double)(1 << level);
    for (int i = 0; i < nl; i++) if (fabs(lv[i].um - want) < 0.02 * want || (lv[0].um == 1.0 && fabs(lv[i].um - rel) < 0.02 * rel)) return 1;
    return 0;
}
static z3 *open_level(store *st, const char *group, int level, double um0) {
    static atomic_uint n_open; unsigned k = atomic_fetch_add(&n_open, 1);
    double t0 = 0; if (getenv("UFSM_OPEN_TRACE")) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); t0 = t.tv_sec + t.tv_nsec * 1e-9; }
    z3 *z = pyramid_open_level(st, group, level, um0, g_cache);
    /* transient failures (remote metadata fetches, many opens in parallel at sampler start) must not silently disable a
       level for the whole run: retry with backoff; a level that is genuinely absent fails all attempts (~3.5 s, once) */
    for (int attempt = 0; !z && attempt < 3 && level_listed(st, group, level, um0); attempt++) {
        usleep((useconds_t)(500000u << attempt));
        z = pyramid_open_level(st, group, level, um0, g_cache);
        if (z) fprintf(stderr, "open %s level %d: succeeded on retry %d\n", group, level, attempt + 1);
    }
    if (!z && level_listed(st, group, level, um0)) fprintf(stderr, "WARNING: %s level %d is listed but cannot be opened: %s\n", group, level, z3_error());
    if (t0) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); fprintf(stderr, "open #%u %s level %d: %s in %.0f ms\n", k, group, level, z ? "ok" : "FAILED", (t.tv_sec + t.tv_nsec * 1e-9 - t0) * 1e3); }
    return z;
}
/* the lazy opens below are called from every sampler worker: one lock (opens are rare, reads never take it).
   Without it two workers racing on the same level both opened it, one handle leaked, and a failed open under the race
   marked the level absent for the whole run (seen as workers spinning forever on draws with no usable level). */
static pthread_mutex_t g_open_mu = PTHREAD_MUTEX_INITIALIZER;
static void open_failed(const source *s, const char *what, int level) { if (getenv("UFSM_DEBUG")) fprintf(stderr, "source %s: %s level %d not available: %s\n", s->name, what, level, z3_error()); }

/* eager parallel open of every level and target of every source (each remote open is a metadata fetch of 70-500 ms; done
   lazily from the sampler workers they serialise on g_open_mu). The lazy path stays as the fallback. */
typedef struct { sources *S; int from, to; } oa_arg;
static void *oa_worker(void *p) {
    oa_arg *a = p;
    for (int k = a->from; k < a->to; k++) {
        int si = k / (MAXLEV * (NCH + 1)), rem = k % (MAXLEV * (NCH + 1)), l = rem / (NCH + 1), c = rem % (NCH + 1);
        source *s = &a->S->src[si];
        if (c == NCH) source_ct(s, l); else if (s->tgt_key[c]) source_tgt(s, c, l);
    }
    return nullptr;
}
void sources_open_all(sources *S, int nthreads) {
    int total = S->n * MAXLEV * (NCH + 1); if (nthreads > total) nthreads = total; if (nthreads < 1) nthreads = 1;
    pthread_t *th = malloc(nthreads * sizeof *th); oa_arg *args = malloc(nthreads * sizeof *args);
    for (int t = 0; t < nthreads; t++) { args[t] = (oa_arg){S, total * t / nthreads, total * (t + 1) / nthreads}; pthread_create(&th[t], nullptr, oa_worker, &args[t]); }
    for (int t = 0; t < nthreads; t++) pthread_join(th[t], nullptr);
    free(th); free(args);
}
/* the open (a remote metadata fetch) runs outside the lock; the lock only publishes the pointer. A racing duplicate open is
   closed. ct_present is cleared on failure so a missing level is not retried per draw. */
z3 *source_ct(source *s, int level) {
    if (level < 0 || level >= MAXLEV || s->ct_present[level] == 0) return nullptr;
    if (!s->ct[level]) {
        z3 *z = open_level(s->s, s->ct_key, level, s->um);
        pthread_mutex_lock(&g_open_mu);
        if (!s->ct[level]) { s->ct[level] = z; z = nullptr; if (!s->ct[level]) { s->ct_present[level] = 0; open_failed(s, "ct", level); } }
        pthread_mutex_unlock(&g_open_mu);
        if (z) z3_close(z);
    }
    return s->ct[level];
}

z3 *source_tgt(source *s, int ch, int level) {
    if (!s->tgt_key[ch] || level < s->tgt_min_level[ch] || level < 0 || level >= MAXLEV || s->tgt_present[ch][level] == 0) return nullptr;
    if (!s->tgt[ch][level]) {
        z3 *z = open_level(s->tgt_store[ch] ? s->tgt_store[ch] : s->s, s->tgt_key[ch], level, s->um);
        pthread_mutex_lock(&g_open_mu);
        if (!s->tgt[ch][level]) { s->tgt[ch][level] = z; z = nullptr; if (!s->tgt[ch][level]) { s->tgt_present[ch][level] = 0; open_failed(s, "target", level); } }
        pthread_mutex_unlock(&g_open_mu);
        if (z) z3_close(z);
    }
    return s->tgt[ch][level];
}

z3 *source_tgt_for_level(source *s, int ch, int level, int *stored_level) {
    if (ch < 0 || ch >= NCH || level < 0 || level >= MAXLEV) return nullptr;
    int actual = (s->tgt_binary[ch] || s->tgt_prob[ch] || s->tgt_side[ch]) && level < s->tgt_min_level[ch] ? s->tgt_min_level[ch] : level;   /* coarser: upsampled */
    if (stored_level) *stored_level = actual;
    return source_tgt(s, ch, actual);
}

static int64_t mask_nearest(int64_t q, int shift, int64_t size) {
    int64_t v = (q + ((int64_t)1 << (shift - 1))) >> shift;
    return v < 0 ? 0 : v >= size ? size - 1 : v;
}
/* the coarse cell containing fine voxel q on a pooled grid (cell a covers fine a 2^shift .. (a + 1) 2^shift - 1): side labels.
   mask_nearest above rounds instead (cell a centred on fine a 2^shift), half a fine cell up at 2x for the voxels it rounds up */
static int64_t pooled_cell(int64_t q, int shift, int64_t size) {
    int64_t v = q >> shift;
    return v < 0 ? 0 : v >= size ? size - 1 : v;
}

/* a probability target stored 2^shift coarser: trilinear between the coarse cell centres (cell j's centre sits at fine
   coordinate f j + (f - 1) / 2, a pooled grid's), clamped at the edges; separable, two interpolated coarse planes at a time */
static int read_prob_trilinear(z3 *z, int shift, const int64_t o[3], const int64_t n[3], uint8_t *out, int nthreads) {
    const z3_meta *m = z3_meta_of(z); const int64_t f = (int64_t)1 << shift;
    int64_t lo[3], cnt[3];
    int32_t *i0[3]; float *wt[3];
    for (int d = 0; d < 3; d++) {
        i0[d] = malloc((size_t)n[d] * sizeof **i0); wt[d] = malloc((size_t)n[d] * sizeof **wt);
        if (!i0[d] || !wt[d]) { for (int e = 0; e <= d; e++) { free(i0[e]); free(wt[e]); } return -1; }
        int64_t mn = INT64_MAX, mx = INT64_MIN;
        for (int64_t k = 0; k < n[d]; k++) {
            const double u = ((double)(o[d] + k) + 0.5) / (double)f - 0.5;   /* coarse coordinate of the fine voxel centre */
            int64_t a = (int64_t)floor(u); double t = u - (double)a;
            if (a < 0) { a = 0; t = 0; } if (a >= m->shape[d] - 1) { a = m->shape[d] - 1; t = 0; }
            i0[d][k] = (int32_t)a; wt[d][k] = (float)t;
            if (a < mn) mn = a; if (a + (t > 0) > mx) mx = a + (t > 0);
        }
        lo[d] = mn; cnt[d] = mx - mn + 1;
        for (int64_t k = 0; k < n[d]; k++) i0[d][k] -= (int32_t)lo[d];
    }
    const size_t cplane = (size_t)cnt[1] * cnt[2], fplane = (size_t)n[1] * n[2];
    uint8_t *low = malloc((size_t)cnt[0] * cplane); float *rx = malloc((size_t)cnt[1] * n[2] * sizeof *rx), *pl[2] = {malloc(fplane * sizeof(float)), malloc(fplane * sizeof(float))};
    int rc = -1; int32_t have[2] = {-1, -1};
    if (!low || !rx || !pl[0] || !pl[1] || z3_read(z, lo, cnt, low, nthreads)) goto done;
    for (int64_t zc = 0; zc < n[0]; zc++) {
        const int32_t za = i0[0][zc], zb = za + (wt[0][zc] > 0); const float tz = wt[0][zc];
        for (int s = 0; s < 2; s++) {   /* the two coarse planes this output plane needs, each interpolated in y and x once */
            const int32_t want = s ? zb : za; int slot = have[0] == want ? 0 : have[1] == want ? 1 : -1;
            if (slot >= 0) continue;
            slot = have[0] == (s ? za : zb) ? 1 : 0;   /* keep the other one */
            const uint8_t *L = low + (size_t)want * cplane;
            for (int64_t yl = 0; yl < cnt[1]; yl++) for (int64_t x = 0; x < n[2]; x++) {
                const int32_t xa = i0[2][x]; const float tx = wt[2][x]; const uint8_t *r = L + (size_t)yl * cnt[2];
                rx[(size_t)yl * n[2] + x] = tx > 0 ? r[xa] + tx * ((float)r[xa + 1] - r[xa]) : r[xa];
            }
            float *P = pl[slot];
            for (int64_t y = 0; y < n[1]; y++) {
                const int32_t ya = i0[1][y]; const float ty = wt[1][y]; const float *ra = rx + (size_t)ya * n[2], *rb = ty > 0 ? ra + n[2] : ra;
                float *q = P + (size_t)y * n[2];
                for (int64_t x = 0; x < n[2]; x++) q[x] = ra[x] + ty * (rb[x] - ra[x]);
            }
            have[slot] = want;
        }
        const float *A = pl[have[0] == za ? 0 : 1], *Bp = pl[have[0] == zb ? 0 : 1];
        uint8_t *dst = out + (size_t)zc * fplane;
        for (size_t k = 0; k < fplane; k++) { const float v = A[k] + tz * (Bp[k] - A[k]); dst[k] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v + 0.5f); }
    }
    rc = 0;
done:
    free(low); free(rx); free(pl[0]); free(pl[1]);
    for (int d = 0; d < 3; d++) { free(i0[d]); free(wt[d]); }
    return rc;
}

int z3_read_label_grid(z3 *z, int binary, int shift, const int64_t o[3], const int64_t n[3], uint8_t *out, int nthreads) {
    if (!z || shift < 0 || shift >= MAXLEV) return -1;
    if (!binary && shift) return read_prob_trilinear(z, shift, o, n, out, nthreads);   /* probabilities: trilinear, not nearest (binary 2: raw codes, nearest) */
    for (int d = 0; d < 3; d++) if (n[d] <= 0) return -1;
    size_t nv = (size_t)n[0] * n[1] * n[2];
    if (!shift) {
        if (z3_read(z, o, n, out, nthreads)) return -1;
        if (binary == 1) for (size_t k = 0; k < nv; k++) out[k] = out[k] ? 254 : 0;
        return 0;
    }
    const z3_meta *m = z3_meta_of(z); int64_t f = (int64_t)1 << shift, co[3], cn[3];
    int64_t (*near)(int64_t, int, int64_t) = binary == 2 ? pooled_cell : mask_nearest;   /* side labels: pooled grids */
    for (int d = 0; d < 3; d++) {
        if (n[d] <= 0 || m->shape[d] <= 0) return -1;
        co[d] = near(o[d], shift, m->shape[d]);
        cn[d] = near(o[d] + n[d] - 1, shift, m->shape[d]) - co[d] + 1;
    }
    uint8_t *low = malloc((size_t)cn[0] * cn[1] * cn[2]);
    int32_t *ixm = malloc((size_t)n[2] * sizeof *ixm);   /* per output x: low-row index, -1 outside the volume */
    if (!low || !ixm) { free(low); free(ixm); return -1; }
    if (z3_read(z, co, cn, low, nthreads)) { free(low); free(ixm); return -1; }
    for (int64_t x = 0; x < n[2]; x++) { int64_t gx = o[2] + x; ixm[x] = gx >= 0 && gx < m->shape[2] * f ? (int32_t)(near(gx, shift, m->shape[2]) - co[2]) : -1; }
    /* consecutive output rows (planes) that read the same low row (plane) are copies: 2^shift of each */
    const size_t rowb = (size_t)n[1] * n[2];
    int64_t pz = -2;
    for (int64_t zc = 0; zc < n[0]; zc++) {
        int64_t gz = o[0] + zc, iz = gz >= 0 && gz < m->shape[0] * f ? near(gz, shift, m->shape[0]) - co[0] : -1;
        uint8_t *plane = out + (size_t)zc * rowb;
        if (zc && iz == pz) { memcpy(plane, plane - rowb, rowb); continue; }
        pz = iz;
        int64_t py = -2;
        for (int64_t y = 0; y < n[1]; y++) {
            int64_t gy = o[1] + y, iy = gy >= 0 && gy < m->shape[1] * f ? near(gy, shift, m->shape[1]) - co[1] : -1;
            uint8_t *dst = plane + (size_t)y * n[2];
            if (y && iy == py) { memcpy(dst, dst - n[2], (size_t)n[2]); continue; }
            py = iy;
            if (iz < 0 || iy < 0) { memset(dst, 0, (size_t)n[2]); continue; }
            const uint8_t *row = low + ((size_t)iz * cn[1] + iy) * cn[2];
            for (int64_t x = 0; x < n[2]; x++) { const uint8_t v = ixm[x] >= 0 ? row[ixm[x]] : 0; dst[x] = binary == 1 ? (v ? 254 : 0) : v; }   /* prob / side codes: the value */
        }
    }
    free(low); free(ixm); return 0;
}

int source_read_target(source *s, int ch, int level, const int64_t o[3], const int64_t n[3], uint8_t *out, int nthreads) {
    int actual = level; z3 *z = source_tgt_for_level(s, ch, level, &actual);
    if (!z) return -1;
    int rc = z3_read_label_grid(z, s->tgt_side[ch] ? 2 : s->tgt_binary[ch] || (z && z3_meta_of(z)->label_binary), actual - level, o, n, out, nthreads);
    if (!rc && s->tgt_votes[ch] && !s->tgt_votes_soft[ch]) {   /* vote codes -> side codes (RV_*): the rest of the sampler sees a side array */
        const size_t nv = (size_t)n[0] * n[1] * n[2]; const uint8_t *lut = s->tgt_vote_lut[ch];
        for (size_t k = 0; k < nv; k++) out[k] = lut[out[k]];
    }
    if (!rc && s->tgt_prob[ch] && !s->tgt_prob254[ch]) {   /* p * 255 -> p * 254: 255 is the ignore code downstream */
        size_t nv = (size_t)n[0] * n[1] * n[2];
        for (size_t k = 0; k < nv; k++) out[k] = (uint8_t)((out[k] * 254 + 127) / 255);
    }
    return rc;
}

int source_region_shared(const source *s, int ch) { return s->reg[ch] && s->reg[ch]->array != nullptr; }

z3 *source_region(source *s, int ch, int i) {
    regions *r = s->reg[ch];
    if (!r || i < 0 || i >= r->n) return nullptr;
    store *st = s->tgt_store[ch] ? s->tgt_store[ch] : s->s;
    if (r->array) {
        if (!s->reg_shared[ch]) { pthread_mutex_lock(&g_open_mu); if (!s->reg_shared[ch]) { s->reg_shared[ch] = open_level(st, r->array, 0, s->um); if (!s->reg_shared[ch]) open_failed(s, "region array", 0); } pthread_mutex_unlock(&g_open_mu); }
        return s->reg_shared[ch];
    }
    if (!s->reg_z[ch][i]) {
        pthread_mutex_lock(&g_open_mu);
        if (s->reg_z[ch][i]) { pthread_mutex_unlock(&g_open_mu); return s->reg_z[ch][i]; }
        char k[1024];
        snprintf(k, sizeof k, "%s/region_%lld_%lld_%lld.zarr", r->dir, (long long)r->origin[i][0],
                 (long long)r->origin[i][1], (long long)r->origin[i][2]);
        s->reg_z[ch][i] = z3_open(st, k, g_cache);
        if (!s->reg_z[ch][i]) open_failed(s, "region", i);
        pthread_mutex_unlock(&g_open_mu);
    }
    return s->reg_z[ch][i];
}

void sources_set_cache(const char *dir) { g_cache = dir; }
