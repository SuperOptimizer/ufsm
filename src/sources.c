#include "sources.h"
#include "fiberraster.h"
#include "fuse.h"
#include "band_onfly.h"
#include "inksplat.h"
#include "dcache.h"
/* weak: the sampler test binaries link sources.o without inksplat.o / dcache.o (an onfly ink target then fails to open) */
[[gnu::weak]] z3 *inkfly_open(z3 *, const char *, const inksplat_cfg *, size_t, const char *, void **);
[[gnu::weak]] z3 *inkfly_open_pooled(z3 *, z3 *, int);
[[gnu::weak]] int inkfly_holdouts(const char *, const char *, inksplat_cfg *);
[[gnu::weak]] void inksplat_defaults(inksplat_cfg *);
[[gnu::weak]] const char *dcache_default_dir(void);
[[gnu::weak]] z3 *z3_open_cached(z3 *, const char *, const char *, int, size_t);
#include "meshcodes.h"
[[gnu::weak]] z3 *mesh_codes_open(const char *, const char *, const char *, int, z3 *, int);   /* (weak: binaries without meshcodes.o: "meshes" then fails to open) */
[[gnu::weak]] const char *mesh_codes_key(const z3 *);
#include "json.h"
#include "votes.h"
#include "srpair.h"
#include <sys/stat.h>
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

/* ---- held-out boxes ---- */
static void src_add_hold(source *s, const json *b, int rej) {   /* one [z, y, x, nz, ny, nx] (level 0); the first one is also hold_o / hold_n */
    if (!b || b->type != J_ARR || b->n != 6) return;
    int64_t v[6]; for (int d = 0; d < 6; d++) v[d] = (int64_t)json_num(json_at(b, (size_t)d), 0);
    if (v[3] <= 0 || v[4] <= 0 || v[5] <= 0) return;
    s->hold = realloc(s->hold, (size_t)(s->nhold + 1) * sizeof *s->hold); s->hold_rej = realloc(s->hold_rej, (size_t)(s->nhold + 1));
    s->hold_rej[s->nhold] = (uint8_t)(rej != 0); memcpy(s->hold[s->nhold++], v, sizeof v);
    if (s->nhold == 1) for (int d = 0; d < 3; d++) { s->hold_o[d] = v[d]; s->hold_n[d] = v[3 + d]; }
}
static const char *src_base(const char *p) {   /* the last path component, ignoring trailing slashes and a trailing "/<level digits>" */
    static _Thread_local char b[512]; snprintf(b, sizeof b, "%s", p ? p : "");
    for (;;) {
        size_t n = strlen(b); while (n && b[n - 1] == '/') b[--n] = 0;
        char *sl = strrchr(b, '/'); const char *t = sl ? sl + 1 : b; int dig = *t != 0;
        for (const char *c = t; *c; c++) dig &= *c >= '0' && *c <= '9';
        if (dig && sl) { *sl = 0; continue; }
        return t;
    }
}
/* "holdout_file": the boxes of a wbench holdouts.json ({"boxes": [{"ct": KEY, "l0": [6]}...]}) whose ct basename is this source's */
static int src_hold_file(source *s, const char *path, const json *rejn) {
    FILE *f = fopen(path, "rb"); if (!f) { fprintf(stderr, "sources: %s: cannot open holdout_file %s\n", s->name, path); return 1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET); char *t = malloc((size_t)n + 1);
    const size_t got = t ? fread(t, 1, (size_t)n, f) : 0; fclose(f);
    json *j = got == (size_t)n ? json_parse(t, (size_t)n) : nullptr; free(t);
    const json *bx = j ? json_get(j, "boxes") : nullptr;
    if (!bx || bx->type != J_ARR) { fprintf(stderr, "sources: %s: holdout_file %s: no \"boxes\" array\n", s->name, path); json_free(j); return 1; }
    char me[512]; snprintf(me, sizeof me, "%s", src_base(s->ct_key)); int added = 0, nrej = 0;
    for (size_t k = 0; k < bx->n; k++) { const json *e = json_at(bx, k);
        if (!strcmp(src_base(json_str(json_get(e, "ct"), "")), me)) {
            const char *nm = json_str(json_get(e, "name"), ""); int rej = 0;
            for (size_t q = 0; rejn && rejn->type == J_ARR && q < rejn->n; q++) rej |= *nm && !strcmp(nm, json_str(json_at(rejn, q), ""));
            src_add_hold(s, json_get(e, "l0"), rej); added++; nrej += rej;
            if (rej) fprintf(stderr, "sources: %s: held-out box %s rejects training windows (holdout_reject)\n", s->name, nm); } }
    json_free(j);
    if (added) fprintf(stderr, "sources: %s: %d held-out boxes from %s (%d reject)\n", s->name, added, path, nrej);
    return 0;
}

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

const char *const SHNAME[SH_N] = {"phase", "normal", "density", "wind4", "rv", "ink", "fibre"};
static const char *CHNAME[NTG] = {"recto", "sheet", "fibre", "fibre_h", "fibre_v"};

static char *sdup(const char *s) { return s ? strdup(s) : nullptr; }

/* an "onfly" phase spec ({root, key, rmax[, pool2]} or {"sets": [...], "tol"}) opened with `extra` more 2x poolings of its codes at rmax / 2^extra
   (extra 1: the phase of the next coarser level). *rmx: the largest rmax used. nullptr (reported) on failure */
/* the codes of an "onfly" phase spec (or one of its "sets"): a stored array {root, key[, pool2]} pooled pool2 + extra times, or
   {"meshes": {"dir", "value" ("q.tif"), "map" (a reg.json: the meshes' scan -> this source's), "level" (the codes' CT level; default plev)}}:
   the codes rasterised on read from the q-assigned meshes (meshcodes.h mesh_codes_open) at level + extra directly (a stored level-L+1
   raster, not the pooled level L: 8x fewer voxels rasterised), then pool2 poolings. key: the dcache key of the phase derived from them
   (stored codes: their path, zarr.json size / mtime and poolings; "" over http: not cached). nullptr (reported) on failure */
static z3 *onfly_codes_open(const source *s, const json *e, const json *of, int extra, int plev, char *key, size_t nkey) {
    key[0] = 0;
    const json *mj = json_get(e, "meshes"); if (!mj && e != of) mj = json_get(of, "meshes");
    const int npool = (int)json_num(json_get(e, "pool2"), json_num(json_get(of, "pool2"), 0));
    z3 *cz = nullptr;
    if (mj) {
        const char *md = json_str(json_get(mj, "dir"), nullptr), *mv = json_str(json_get(mj, "value"), "q.tif"), *mm = json_str(json_get(mj, "map"), nullptr);
        int lv = (int)json_num(json_get(mj, "level"), plev) + extra, pool = 0;
        z3 *ct = md && lv >= 0 && lv < MAXLEV && s->s ? pyramid_open_level(s->s, s->ct_key, lv, s->um, g_cache) : nullptr;
        if (!ct && extra > 0 && md && s->s && (ct = pyramid_open_level(s->s, s->ct_key, lv - extra, s->um, g_cache))) { lv -= extra; pool = extra; }   /* (no CT level there: the level's codes pooled) */
        cz = ct && mesh_codes_open ? mesh_codes_open(md, mv, mm, lv, ct, 0) : nullptr;
        if (ct && !mesh_codes_open) z3_close(ct);
        if (!cz) { fprintf(stderr, "sources: %s: phase onfly \"meshes\": cannot open %s (value %s, level %d of the CT)\n", s->name, md ? md : "?", mv, lv); return nullptr; }
        const char *mk = mesh_codes_key ? mesh_codes_key(cz) : nullptr; if (mk) snprintf(key, nkey, "%s-pool%d%s", mk, npool + pool, npool + pool ? "b" : "");   /* ("b": the bin-search pooling, codes_pool.h, 2026-11-12) */
        for (int q2 = 0; cz && q2 < pool; q2++) { z3 *q = codes_pool2_open(cz); if (!q) { z3_close(cz); cz = nullptr; } else cz = q; }
    } else {
        const char *cr = json_str(json_get(e, "root"), nullptr), *ck = json_str(json_get(e, "key"), "");
        store *cs = cr ? store_open(cr) : nullptr; cz = cs ? z3_open(cs, ck, nullptr) : nullptr;
        if (!cz) { fprintf(stderr, "sources: %s: phase onfly: cannot open the codes %s/%s\n", s->name, cr ? cr : "?", ck); return nullptr; }
        /* the dcache key of stored codes (2026-11-13, scratch1: the stored phase rasters replaced by their codes): the array's path, its
           zarr.json's size and mtime (a rewritten array: a new key), the poolings */
        { char zp[1100]; struct stat st; snprintf(zp, sizeof zp, "%s/%s/zarr.json", cr, ck);
          if (!strstr(cr, "://") && !stat(zp, &st)) snprintf(key, nkey, "codes %s/%s size %lld mtime %lld pool%db", cr, ck, (long long)st.st_size, (long long)st.st_mtime, extra + npool); }
        for (int q2 = 0; cz && q2 < extra; q2++) { z3 *q = codes_pool2_open(cz); if (!q) { z3_close(cz); cz = nullptr; } else cz = q; }
    }
    for (int q2 = 0; cz && q2 < npool; q2++) { z3 *q = codes_pool2_open(cz); if (!q) { z3_close(cz); cz = nullptr; } else cz = q; }
    if (!cz) fprintf(stderr, "sources: %s: phase onfly: cannot pool the codes\n", s->name);
    return cz;
}
/* the phase of codes cz at rmax rm (owned); behind the derived-array cache when key names the codes (meshes): "cache_dir" (default
   $UFSM_DERIVED_CACHE / ~/.cache/ufsm/derived), "cache_mb" (1024) */
static z3 *onfly_phase_of(z3 *cz, float rm, const char *key, const json *of) {
    z3 *z = phase_onfly_open(cz, rm);
    if (!z) { z3_close(cz); return nullptr; }
    if (!key[0] || !z3_open_cached || json_num(json_get(of, "cache"), 1) == 0) return z;   /* ("cache": 0: computed per read, no cache) */

    const char *dcd = json_str(json_get(of, "cache_dir"), dcache_default_dir ? dcache_default_dir() : nullptr);
    /* the cell: the pair rule reads its codes with a halo of rmax + 2, so a 128^3 cell at rmax 160 (the 2.4 um rung) computed 44x its voxels
       (10 workers made no P 1280 batch in 8 minutes on the pod); cells of >= 3 rmax (up to 512: 4.3x at rmax 160) */
    int cell = (int)json_num(json_get(of, "cache_cell"), 0);
    if (cell <= 0) { cell = 128; while (cell < 3 * rm && cell < 512) cell *= 2; }
    char k2[1200]; snprintf(k2, sizeof k2, "phase-onfly-pair-v1 rmax %.9g codes %s cell %d", rm, key, cell);   /* (the cell size: the disk files' layout) */
    return z3_open_cached(z, dcd && *dcd ? dcd : nullptr, k2, cell, (size_t)json_num(json_get(of, "cache_mb"), 1024));
}
/* an "onfly" phase spec ({root, key, rmax[, pool2]}, {meshes, rmax}, or {"sets": [...], "tol"}) opened with `extra` more 2x levels of its
   codes at rmax / 2^extra (extra 1: the phase of the next coarser level). plev: the phase's level (the meshes' default codes level).
   *rmx: the largest rmax used. nullptr (reported) on failure */
static z3 *onfly_phase_open(const source *s, const json *of, int extra, int plev, float *rmx) {
    const char *name = s->name;
    const float rm = (float)json_num(json_get(of, "rmax"), 0), div = (float)(1 << extra);
    const json *sj = json_get(of, "sets");   /* "sets": [{"root", "key", "rmax"} | {"meshes", "rmax"}, ...] + "tol" (0.1): the sets' phases fused per
                                                voxel as phase-fuse (band_onfly.h phase_onfly_fuse_open); "root" / "key" then unused */
    *rmx = 0; char key[1024];
    if (sj && sj->type == J_ARR) {
        z3 *ph[FUSE_MAX]; int np = 0, bad = !sj->n || sj->n > FUSE_MAX;
        for (size_t k = 0; !bad && k < sj->n; k++) {
            const json *e2 = json_at(sj, k);
            const float rm2 = (float)json_num(json_get(e2, "rmax"), rm) / div;
            z3 *c2 = onfly_codes_open(s, e2, of, extra, plev, key, sizeof key);
            z3 *p2 = c2 && rm2 > 0 ? onfly_phase_of(c2, rm2, key, of) : nullptr;
            if (!p2) { if (c2 && !(rm2 > 0)) z3_close(c2);   /* (onfly_phase_of closes its codes on failure) */ fprintf(stderr, "sources: %s: phase onfly set %zu: no codes or no \"rmax\" > 0\n", name, k); bad = 1; break; }
            ph[np++] = p2; if (rm2 * div > *rmx) *rmx = rm2 * div;
        }
        z3 *z = bad ? nullptr : phase_onfly_fuse_open(ph, np, (float)json_num(json_get(of, "tol"), 0.1));
        if (bad) for (int k = 0; k < np; k++) z3_close(ph[k]);
        if (!z) fprintf(stderr, "sources: %s: phase onfly \"sets\": 1..%d sets of {root, key, rmax} / {meshes, rmax}, one shape\n", name, FUSE_MAX);
        return z;
    }
    z3 *cz = onfly_codes_open(s, of, of, extra, plev, key, sizeof key);
    z3 *z = cz && rm > 0 ? onfly_phase_of(cz, rm / div, key, of) : nullptr;
    if (!z) { if (cz && !(rm > 0)) z3_close(cz); fprintf(stderr, "sources: %s: phase onfly: no codes or no \"rmax\" > 0\n", name); }
    *rmx = rm;
    return z;
}
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
        { const json *mj = json_get(e, "ct_mirror"); const char *mr = json_str(json_get(mj, "root"), nullptr);
          if (mr) { s->ct_ms = store_open(mr); s->ct_mkey = sdup(json_str(json_get(mj, "ct"), s->ct_key)); } }
        { const json *lj = json_get(e, "latvol"); const char *lr = json_str(json_get(lj, "root"), nullptr);
          if (lr) { s->lv_s = store_open(lr); s->lv_key = sdup(json_str(json_get(lj, "ct"), s->ct_key)); } }
        s->um = json_num(json_get(e, "um"), 0);
        s->weight = json_num(json_get(e, "weight"), 1.0);
        s->ink_weight = json_num(json_get(e, "ink_weight"), 1.0);
        s->trust_band = (int)json_num(json_get(e, "trust_band"), 0);
        { const char *sc = json_str(json_get(e, "scope"), "global");
          if (!strcmp(sc, "local")) s->scope_local = 1;
          else if (!strcmp(sc, "w4")) s->scope_local = 2;
          else if (strcmp(sc, "global")) { fprintf(stderr, "sources: %s: \"scope\" must be \"local\", \"w4\" or \"global\"\n", s->name); json_free(j); sources_free(S); return nullptr; } }
        s->iso = (int)json_num(json_get(e, "iso"), 0);
        s->pap_level = (int)json_num(json_get(e, "papyrus"), s->iso);
        {   /* "heads": per-head loss multipliers (a number, or {"on": 0|1, "lambda": x}); unknown names are refused (a typo would train a head silently) */
            for (int h = 0; h < SH_N; h++) s->head_w[h] = 1.f;
            const json *hj = json_get(e, "heads");
            if (hj && hj->type != J_OBJ) { fprintf(stderr, "sources: %s: \"heads\" must be an object {\"phase\": 1, \"rv\": 0, ...}\n", s->name); json_free(j); sources_free(S); return nullptr; }
            for (size_t k = 0; hj && k < hj->n; k++) {
                const char *nm = hj->keys[k]; const json *v = hj->items[k]; int h = 0;
                while (h < SH_N && strcmp(SHNAME[h], nm)) h++;
                double w = v && v->type == J_NUM ? v->num : v && v->type == J_OBJ ? json_num(json_get(v, "lambda"), 1) * (json_num(json_get(v, "on"), 1) != 0) : -1;
                if (h == SH_N || !(w >= 0 && w <= 1000)) {
                    fprintf(stderr, "sources: %s: \"heads\": \"%s\": a head of", s->name, nm); for (int q = 0; q < SH_N; q++) fprintf(stderr, " %s", SHNAME[q]);
                    fprintf(stderr, ", value a lambda 0..1000 or {\"on\": 0|1, \"lambda\": x}\n"); json_free(j); sources_free(S); return nullptr;
                }
                s->head_w[h] = (float)w;
            }
        }
        s->head = (int)json_num(json_get(e, "head"), 0);
        if (s->head < 0 || s->head > 6) { fprintf(stderr, "sources: %s: head 0..6\n", s->name); json_free(j); sources_free(S); return nullptr; }
        s->min_level = (int)json_num(json_get(e, "min_level"), 0);
        s->max_level = (int)json_num(json_get(e, "max_level"), -1);
        s->geometry = (int)json_num(json_get(e, "geometry"), 1);
        s->crush_fine = (int)json_num(json_get(e, "crush_fine"), -1);
        if (s->crush_fine > 3) { fprintf(stderr, "sources: %s: crush_fine 0..3\n", s->name); json_free(j); sources_free(S); return nullptr; }
        for (int l = 0; l < MAXLEV; l++) s->ct_present[l] = -1; /* unknown until probed */
        const json *tg = json_get(e, "targets");
        for (int c = 0; c < NTG; c++) {
            const json *t = json_get(tg, CHNAME[c]);
            if (!t && c == CH_SHEET) t = json_get(tg, "ink");   /* task unwrap: the ink teacher (prob) in target slot 1 */
            if (!t && c == CH_FIB) t = json_get(tg, "fiber");
            if (!t && c == CH_FIBH) t = json_get(tg, "fiber_h");
            if (!t && c == CH_FIBV) t = json_get(tg, "fiber_v");
            for (int l = 0; l < MAXLEV; l++) s->tgt_present[c][l] = -1;
            if (!t) continue;
            if (t->type == J_OBJ && json_get(t, "onfly")) {   /* {"onfly": {"segments", "level", "iso", "face", "holdout_file", "cache_mb"}}: the human-ink splat
                                                                 computed per chunk on first read (inksplat.h inkfly_open; == ink-splat --face --merge) */
                if (!inkfly_open) { fprintf(stderr, "sources: %s/%s: an onfly ink target needs inksplat.o in this binary\n", s->name, CHNAME[c]); json_free(j); sources_free(S); return nullptr; }
                const json *of = json_get(t, "onfly"); inksplat_cfg ic; inksplat_defaults(&ic); ic.face = 1;
                ic.level = (int)json_num(json_get(of, "level"), 1); ic.iso = (int)json_num(json_get(of, "iso"), s->iso > 0 ? s->iso : 64); ic.ct_um = s->um; ic.cache = g_cache;
                { const char *vm = json_str(json_get(of, "validation"), "exclude"); ic.val_mode = !strcmp(vm, "only") ? 2 : !strcmp(vm, "off") ? 0 : 1; }   /* the segments' "validation" rasters: held out per pixel */
                const json *fa = json_get(of, "face"); if (fa && fa->n == 2) { ic.face_a = json_num(json_at(fa, 0), -6); ic.face_b = json_num(json_at(fa, 1), 0); }
                const char *sg = json_str(json_get(of, "segments"), nullptr), *hf = json_str(json_get(of, "holdout_file"), json_str(json_get(e, "holdout_file"), nullptr));
                const int L = ic.level;
                if (!sg || !(s->um > 0) || L < 0 || L + 1 >= MAXLEV || (hf && inkfly_holdouts(hf, s->ct_key, &ic))) {
                    fprintf(stderr, "sources: %s/%s: onfly needs \"segments\", the source's \"um\", level 0..%d (and a readable holdout_file)\n", s->name, CHNAME[c], MAXLEV - 2);
                    json_free(j); sources_free(S); return nullptr; }
                z3 *ct = pyramid_open_level(s->s, s->ct_key, L, s->um, g_cache), *ct1 = pyramid_open_level(s->s, s->ct_key, L + 1, s->um, g_cache);
                ic.ct_root = json_str(json_get(e, "root"), "."); ic.ct_key = s->ct_key;   /* (named in the derived cache's key) */
                const char *dcd = json_str(json_get(of, "cache_dir"), dcache_default_dir ? dcache_default_dir() : nullptr);   /* the disk cache (dcache.h): "cache_dir" ("off": none), $UFSM_DERIVED_CACHE, ~/.cache/ufsm/derived */
                z3 *z = ct ? inkfly_open(ct, sg, &ic, (size_t)json_num(json_get(of, "cache_mb"), 2048), dcd && *dcd && strcmp(dcd, "off") && strcmp(dcd, "0") ? dcd : nullptr, &s->ink_fly[c]) : nullptr;
                if (!z) { if (ct1) z3_close(ct1); fprintf(stderr, "sources: %s/%s: cannot open the onfly ink target %s\n", s->name, CHNAME[c], sg); json_free(j); sources_free(S); return nullptr; }
                s->tgt_prob[c] = s->tgt_prob254[c] = 1; s->tgt_min_level[c] = L; s->tgt_key[c] = strdup("onfly");
                for (int l = 0; l < MAXLEV; l++) s->tgt_present[c][l] = 0;
                s->tgt[c][L] = z; s->tgt_present[c][L] = 1;
                if (ct1) { s->tgt[c][L + 1] = inkfly_open_pooled(ct1, z, 0); s->tgt_present[c][L + 1] = s->tgt[c][L + 1] != nullptr; }   /* (closed before level L: sources_free goes up the levels? see below) */
                continue;
            }
            if (t->type == J_STR) { s->tgt_key[c] = strdup(t->str); continue; }
            const json *fj = json_get(t, "fibre");
            if (fj) {   /* the fibre planes on the fly: no stored raster */
                struct fib_spec *f = calloc(1, sizeof *f); const char *pl = json_str(json_get(fj, "plane"), "");
                f->afv = sdup(json_str(json_get(fj, "afv"), nullptr)); f->extra = sdup(json_str(json_get(fj, "extra"), ""));
                f->map = sdup(json_str(json_get(fj, "map"), nullptr)); f->tr = sdup(json_str(json_get(fj, "tr"), nullptr));
                f->rad = json_num(json_get(fj, "radius"), 2); f->guard = json_num(json_get(fj, "guard"), 3); f->ring = json_num(json_get(fj, "ring"), 0);
                f->iso = (int)json_num(json_get(fj, "iso"), 64); f->plane = !strcmp(pl, "V");
                if (!f->afv || (strcmp(pl, "H") && strcmp(pl, "V")) || !(f->rad > 0) || f->guard < 0 || f->ring < 0 || f->rad + f->guard + f->ring > 15) {
                    fprintf(stderr, "sources: %s/%s: \"fibre\": {\"afv\": PATH, \"plane\": \"H\" | \"V\", radius 2, guard 3, ring 0 (radius + guard + ring <= 15), iso 64, extra, map | tr}\n", s->name, CHNAME[c]);
                    free(f); json_free(j); sources_free(S); return nullptr;
                }
                s->tgt_fib[c] = f; s->tgt_key[c] = strdup("#fibre"); s->tgt_prob[c] = s->tgt_prob254[c] = 1;
                continue;
            }
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
            const json *pt0 = json_get(bt, "phase");
            s->phase_only = !broot && pt0 && json_get(pt0, "level");   /* (also an "onfly" phase: the codes are read through it) */   /* phase-only: the band handle aliases the stored phase (below) */
            store *bs = broot ? store_open(broot) : nullptr;
            s->band = bs ? z3_open(bs, bkey, nullptr) : nullptr;
            if ((!s->band && !s->phase_only) || !(s->band_radius > 0) || !(s->band_span >= 0)) {
                fprintf(stderr, "sources: %s: cannot open band target %s/%s (%s)\n", s->name, broot ? broot : "?", bkey, z3_error());
                json_free(j); sources_free(S); return nullptr;
            }
            const json *pt = json_get(bt, "phase");
            if (pt) {
                const char *proot = json_str(json_get(pt, "root"), nullptr), *pkey = json_str(json_get(pt, "key"), "4.8");
                store *ps = proot ? store_open(proot) : nullptr;
                s->phase = ps ? z3_open(ps, pkey, nullptr) : nullptr;
                const json *of = json_get(pt, "onfly");   /* {"root", "key", "rmax"}: the pair-rule phase computed from these codes on read
                                                             (band_onfly.h; byte-identical to phase-raster --pair 1 on the codes' grid) */
                if (of && !s->phase) {
                    float rmx = 0;
                    if (!(s->phase = onfly_phase_open(s, of, 0, (int)json_num(json_get(pt, "level"), 1), &rmx))) { json_free(j); sources_free(S); return nullptr; }
                    s->phase_onfly = rmx;
                }
                s->phase_level = (int)json_num(json_get(pt, "level"), 1); s->phase_direct = json_get(pt, "level") != nullptr;
                if (of && s->phase_onfly > 0 && s->phase_level + 1 < MAXLEV) {   /* the next coarser level (the scale rungs): the same rule on the codes
                    pooled once more at half rmax (byte-identical to phase-raster --gs 4 of a 2x coarser raster), not the phase pooled */
                    float r1 = 0; s->phase_pl[s->phase_level + 1] = onfly_phase_open(s, of, 1, s->phase_level, &r1);
                    if (!s->phase_pl[s->phase_level + 1]) { json_free(j); sources_free(S); return nullptr; }
                }
                const double sg = json_num(json_get(pt, "phase_sign"), 1);
                if (sg != 1 && sg != -1) { fprintf(stderr, "sources: %s: \"phase_sign\" must be 1 or -1\n", s->name); json_free(j); sources_free(S); return nullptr; }
                s->phase_sign = (int)sg;
                if (s->phase_only) {
                    if (!s->phase || s->phase_level < 0 || s->phase_level >= MAXLEV) {
                        fprintf(stderr, "sources: %s: phase-only band: cannot open the stored phase %s/%s, or level not 0..%d\n", s->name, proot ? proot : "?", pkey, MAXLEV - 1);
                        json_free(j); sources_free(S); return nullptr;
                    }
                    s->band = s->phase;   /* (sources_free closes it once) */
                }
                const z3_meta *bm = z3_meta_of(s->band), *pm = s->phase ? z3_meta_of(s->phase) : nullptr;
                const int sh = s->phase_level - 1;   /* the label grid is level 1: shapes halve per level above it */
                if (s->phase_only) {}
                else if (pm && sh == -1) {   /* the phase on the level-0 grid (labels at the training level): twice the label grid */
                    if ((pm->shape[0] + 1) >> 1 != bm->shape[0] || (pm->shape[1] + 1) >> 1 != bm->shape[1] || (pm->shape[2] + 1) >> 1 != bm->shape[2]) {
                        fprintf(stderr, "sources: %s: the level-0 stored phase %s/%s is not twice the band codes' shape\n", s->name, proot ? proot : "?", pkey);
                        json_free(j); sources_free(S); return nullptr;
                    }
                }
                else if (!pm || sh < 0 || sh > 4 ||pm->shape[0] != (bm->shape[0] + (1 << sh) - 1) >> sh || pm->shape[1] != (bm->shape[1] + (1 << sh) - 1) >> sh || pm->shape[2] != (bm->shape[2] + (1 << sh) - 1) >> sh) {
                    fprintf(stderr, "sources: %s: cannot open the stored phase %s/%s or its shape differs from the band codes'\n", s->name, proot ? proot : "?", pkey);
                    json_free(j); sources_free(S); return nullptr;
                }
                const json *ig = json_get(pt, "ignore");   /* a mask on the stored phase's grid: nonzero = not trained on */
                if (ig) {
                    const char *ir = json_str(json_get(ig, "root"), nullptr), *ik = json_str(json_get(ig, "key"), "");
                    store *is = ir ? store_open(ir) : nullptr; s->phase_ign = is ? z3_open(is, ik, nullptr) : nullptr;
                    const z3_meta *im = s->phase_ign ? z3_meta_of(s->phase_ign) : nullptr;
                    if (!im || memcmp(im->shape, pm->shape, sizeof im->shape)) {
                        fprintf(stderr, "sources: %s: cannot open the phase ignore mask %s/%s or its shape differs from the stored phase's\n", s->name, ir ? ir : "?", ik);
                        json_free(j); sources_free(S); return nullptr;
                    }
                }
                const json *pw = json_get(pt, "weight");   /* confidence on the stored phase's grid: 0..255 = keep probability per 8^3 block */
                if (pw) {
                    const char *wr = json_str(json_get(pw, "root"), nullptr), *wk = json_str(json_get(pw, "key"), "");
                    store *ws = wr ? store_open(wr) : nullptr; s->phase_wt = ws ? z3_open(ws, wk, nullptr) : nullptr;
                    const z3_meta *wm = s->phase_wt ? z3_meta_of(s->phase_wt) : nullptr;
                    if (!wm || memcmp(wm->shape, pm->shape, sizeof wm->shape)) {
                        fprintf(stderr, "sources: %s: cannot open the phase weight %s/%s or its shape differs from the stored phase's\n", s->name, wr ? wr : "?", wk);
                        json_free(j); sources_free(S); return nullptr;
                    }
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
        const json *vj = json_get(e, "view");   /* scan fusion: the second-scan view (opened by the sampler, src/view.h) */
        if (vj) { s->view = 1; s->view_level = (int)json_num(json_get(vj, "level"), 0); const char *vm = json_str(json_get(vj, "map"), nullptr); s->view_map = vm ? strdup(vm) : nullptr; }
        const json *xj = json_get(e, "xscan");   /* --aug-xscan partner scans (opened by the sampler) */
        if (xj && xj->type == J_ARR) for (size_t k = 0; k < xj->n && s->nxs < 4; k++) {
            const json *xe = json_at(xj, k); const char *xm = xe && xe->type == J_STR ? json_str(xe, nullptr) : json_str(json_get(xe, "map"), nullptr);
            if (!xm) { fprintf(stderr, "sources: %s: xscan entries are {\"map\": REG.json}\n", s->name); json_free(j); sources_free(S); return nullptr; }
            s->xs_map[s->nxs++] = strdup(xm);
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
        const json *ej = json_get(e, "emb");   /* {"groups": [DIR, ...], "name": "pca64", "level": L}: train --distill feature targets */
        if (ej) {
            const json *gj = json_get(ej, "groups");
            if (gj && gj->type == J_ARR) for (size_t k = 0; k < gj->n && s->nemb < 8; k++) { const char *g = json_str(json_at(gj, k), nullptr); if (g) s->emb_grp[s->nemb++] = strdup(g); }
            s->emb_name = strdup(json_str(json_get(ej, "name"), "pca64")); s->emb_level = (int)json_num(json_get(ej, "level"), -1);
            if (!s->nemb || s->emb_level < 0) { fprintf(stderr, "sources: %s: \"emb\" needs \"groups\": [DIR, ...] and \"level\" >= 0\n", s->name); json_free(j); sources_free(S); return nullptr; }
        }
        const json *ho = json_get(e, "holdout");
        if (ho && ho->type == J_ARR && ho->n == 6 && json_at(ho, 0)->type != J_ARR) src_add_hold(s, ho, 0);
        else if (ho && ho->type == J_ARR) for (size_t k = 0; k < ho->n; k++) src_add_hold(s, json_at(ho, k), 0);
        const char *hf = json_str(json_get(e, "holdout_file"), nullptr);
        if (hf && src_hold_file(s, hf, json_get(e, "holdout_reject"))) { json_free(j); sources_free(S); return nullptr; }
        const char *hm = json_str(json_get(e, "holdout_mode"), nullptr);
        if (hm) { s->hold_mode = !strcmp(hm, "mask") ? 1 : !strcmp(hm, "reject") ? 2 : -1;
                  if (s->hold_mode < 0) { fprintf(stderr, "sources: %s: \"holdout_mode\" is \"mask\" or \"reject\", not \"%s\"\n", s->name, hm); json_free(j); sources_free(S); return nullptr; } }
        const json *bx = json_get(e, "boxes");   /* task restore: level-0 boxes [z, y, x, nz, ny, nx] that hold local CT (training windows lie inside one) */
        if (bx && bx->type == J_ARR && bx->n) {
            s->box = calloc(bx->n, sizeof *s->box);
            for (size_t k = 0; k < bx->n; k++) { const json *b = json_at(bx, k); if (!b || b->type != J_ARR || b->n != 6) continue;
                for (int d = 0; d < 6; d++) s->box[s->nbox][d] = (int64_t)json_num(json_at(b, (size_t)d), 0);
                s->nbox++; }
        }
        const json *sj = json_get(e, "sr_pair");   /* a real SR pair directory (ufsm sr-pairs output) at "root" */
        if (sj) {
            s->srp = srpair_open(json_str(json_get(e, "root"), "."), json_str(json_get(sj, "role"), "train"), (int)json_num(json_get(sj, "lut"), 1), (int)json_num(json_get(sj, "cache"), 0));
            if (!s->srp) { json_free(j); sources_free(S); return nullptr; }
            for (int l = 0; l < MAXLEV; l++) s->ct_present[l] = 0;
            for (int c = 0; c < NTG; c++) for (int l = 0; l < MAXLEV; l++) s->tgt_present[c][l] = 0;
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
        for (int l = 0; l < MAXLEV; l++) z3_close(s->lv[l]);
        for (int l = MAXLEV - 1; l >= 0; l--) { z3_close(s->phase_pl[l]); z3_close(s->phase_ign_pl[l]); }   /* (the pooled views borrow phase / ignore: closed first) */
        if (s->band != s->phase) z3_close(s->band); z3_close(s->phase); z3_close(s->wind); z3_close(s->aux); z3_close(s->phase_ign); z3_close(s->phase_wt); free(s->box); free(s->hold); free(s->hold_rej); free(s->focus); free(s->mask); srpair_free(s->srp);
        for (int c = 0; c < NTG; c++) {
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
            if (s->tgt_fib[c]) { free(s->tgt_fib[c]->afv); free(s->tgt_fib[c]->extra); free(s->tgt_fib[c]->map); free(s->tgt_fib[c]->tr); free(s->tgt_fib[c]); }
            store_close(s->tgt_store[c]);
        }
        free(s->ax.z); free(s->ax.y); free(s->ax.x);
        free(s->name); free(s->ct_key); free(s->ct_mkey); free(s->view_map);
        free(s->lv_key); if (s->lv_s) store_close(s->lv_s);
        for (int k = 0; k < s->nxs; k++) free(s->xs_map[k]);
        for (int k = 0; k < s->nemb; k++) free(s->emb_grp[k]);
        free(s->emb_name);
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
        /* relative form: every listed scale an exact power of two (1, 2, 4, ...) and none the source's um x 2^l. The old test needed level 0
           listed (scale 1.0), so a relative group without a local level 0 (the mirror's json over PHerc1667's levels 1-5, 2026-11-12) was read
           as absolute: 2.0 taken as um, every level off, the source silently excluded */
        int relf = 1, absf = 0;
        for (int i = 0; i < nl; i++) {
            const double e = log2(lv[i].um); relf &= lv[i].um >= 1.0 && fabs(e - round(e)) < 1e-9;
            for (int l = 0; l < 16; l++) absf |= fabs(lv[i].um - um0 * (1 << l)) < 0.02 * um0 * (1 << l);
        }
        const int rel_mode = relf && !absf;
        double want = rel_mode ? (double)(1 << level) : um0 * (1 << level);
        for (int i = 0; i < nl; i++)
            if (fabs(lv[i].um - want) < 0.02 * want) { snprintf(k, sizeof k, "%s/%s", group, lv[i].path); return z3_open(st, k, cache); }
        if (!relf && !absf) {   /* neither form: a group whose scales are not the store's voxel size x 2^l (a wrong "um", or a broken json) */
            static int warned; if (!warned++) { fprintf(stderr, "pyramid: %s: the group's scales (", group); for (int i = 0; i < nl; i++) fprintf(stderr, "%s%g", i ? " " : "", lv[i].um);
                                                 fprintf(stderr, ") are neither %g um x 2^l nor relative 2^l factors\n", um0); }
        }
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
        int si = k / (MAXLEV * (NTG + 1)), rem = k % (MAXLEV * (NTG + 1)), l = rem / (NTG + 1), c = rem % (NTG + 1);
        source *s = &a->S->src[si];
        if (c == NTG) source_ct(s, l); else if (s->tgt_key[c]) source_tgt(s, c, l);
    }
    return nullptr;
}
void sources_open_all(sources *S, int nthreads) {
    int total = S->n * MAXLEV * (NTG + 1); if (nthreads > total) nthreads = total; if (nthreads < 1) nthreads = 1;
    pthread_t *th = malloc(nthreads * sizeof *th); oa_arg *args = malloc(nthreads * sizeof *args);
    for (int t = 0; t < nthreads; t++) { args[t] = (oa_arg){S, total * t / nthreads, total * (t + 1) / nthreads}; pthread_create(&th[t], nullptr, oa_worker, &args[t]); }
    for (int t = 0; t < nthreads; t++) pthread_join(th[t], nullptr);
    free(th); free(args);
}
/* the open (a remote metadata fetch) runs outside the lock; the lock only publishes the pointer. A racing duplicate open is
   closed. ct_present is cleared on failure so a missing level is not retried per draw. */
/* a CT level: the source's store, else its "ct_mirror" */
static z3 *open_ct_level(source *s, int level) {
    z3 *z = open_level(s->s, s->ct_key, level, s->um);
    if (!z && s->ct_ms) { z = open_level(s->ct_ms, s->ct_mkey, level, s->um); if (z) fprintf(stderr, "source %s: CT level %d from the mirror\n", s->name, level); }
    return z;
}
z3 *source_ct(source *s, int level) {
    if (level < 0 || level >= MAXLEV || s->ct_present[level] == 0) return nullptr;
    if (!s->ct[level]) {
        z3 *z = open_ct_level(s, level);
        pthread_mutex_lock(&g_open_mu);
        if (!s->ct[level]) { s->ct[level] = z; z = nullptr; if (!s->ct[level]) { s->ct_present[level] = 0; open_failed(s, "ct", level); } }
        pthread_mutex_unlock(&g_open_mu);
        if (z) z3_close(z);
    }
    return s->ct[level];
}

z3 *source_lat(source *s, int level) {
    if (!s->lv_s || level < 0 || level >= MAXLEV) return nullptr;
    if (!__atomic_load_n(&s->lv_tried[level], __ATOMIC_ACQUIRE)) {
        z3 *ct = source_ct(s, level), *z = ct ? open_level(s->lv_s, s->lv_key, level, s->um) : nullptr;
        if (z && (!z3_latvol(z) || memcmp(z3_meta_of(z)->shape, z3_meta_of(ct)->shape, sizeof z3_meta_of(ct)->shape))) {
            fprintf(stderr, "source %s: latvol level %d is not a latvol array of the CT's shape: not used\n", s->name, level); z3_close(z); z = nullptr; }
        pthread_mutex_lock(&g_open_mu);
        if (!s->lv_tried[level]) { s->lv[level] = z; z = nullptr; __atomic_store_n(&s->lv_tried[level], 1, __ATOMIC_RELEASE); }
        pthread_mutex_unlock(&g_open_mu);
        if (z) z3_close(z);
    }
    return s->lv[level];
}

static int fib_same(const struct fib_spec *a, const struct fib_spec *b) {
    const int ss = !strcmp(a->afv, b->afv) && !strcmp(a->extra, b->extra) && !strcmp(a->map ? a->map : "", b->map ? b->map : "") && !strcmp(a->tr ? a->tr : "", b->tr ? b->tr : "");
    return ss && a->rad == b->rad && a->guard == b->guard && a->ring == b->ring && a->iso == b->iso;
}
/* a "fibre" target: both planes from one computation; the other plane goes to a channel with the same options and the other plane */
static z3 *source_tgt_fibre(source *s, int ch, int level) {
    if (s->tgt[ch][level]) return s->tgt[ch][level];
    const struct fib_spec *f = s->tgt_fib[ch]; z3 *pl[2];
    z3 *c0 = open_ct_level(s, level), *c1 = c0 ? open_ct_level(s, level) : nullptr, *c2 = c1 ? open_ct_level(s, level) : nullptr;
    const int bad = fiber_onfly_open(f->afv, f->extra, f->map, f->tr, f->rad, f->guard, f->ring, f->iso, level, c0, c1, c2, pl);
    pthread_mutex_lock(&g_open_mu);
    if (bad) { if (!s->tgt[ch][level]) s->tgt_present[ch][level] = 0; pthread_mutex_unlock(&g_open_mu); return s->tgt[ch][level]; }
    if (!s->tgt[ch][level]) { s->tgt[ch][level] = pl[f->plane]; pl[f->plane] = nullptr; }
    for (int c = 0; c < NTG; c++) if (c != ch && s->tgt_fib[c] && s->tgt_fib[c]->plane != f->plane && fib_same(s->tgt_fib[c], f) && !s->tgt[c][level] && pl[!f->plane]) {
        s->tgt[c][level] = pl[!f->plane]; pl[!f->plane] = nullptr; }
    pthread_mutex_unlock(&g_open_mu);
    z3_close(pl[0]); z3_close(pl[1]);
    return s->tgt[ch][level];
}
z3 *source_phase_at(source *s, int level, int ign) {
    z3 *b = ign ? s->phase_ign : s->phase;
    if (!b || level < s->phase_level || level >= MAXLEV) return nullptr;
    if (level == s->phase_level) return b;
    z3 **pl = ign ? s->phase_ign_pl : s->phase_pl;
    if (!pl[level] && !ign && s->phase_onfly > 0) return nullptr;   /* (an on-the-fly phase: its next level is made from the pooled codes at
                                                                       load, deeper none; pooling the derived phase would derive 8x the voxels) */
    if (!pl[level]) {
        z3 *under = source_phase_at(s, level - 1, ign); if (!under) return nullptr;
        pthread_mutex_lock(&g_open_mu);
        if (!pl[level]) pl[level] = phase_pool2_view(under, ign);   /* (cheap: no I/O; under the lock so one view per level) */
        pthread_mutex_unlock(&g_open_mu);
    }
    return pl[level];
}
z3 *source_tgt(source *s, int ch, int level) {
    if (!s->tgt_key[ch] || level < s->tgt_min_level[ch] || level < 0 || level >= MAXLEV || s->tgt_present[ch][level] == 0) return nullptr;
    if (s->tgt_fib[ch]) return source_tgt_fibre(s, ch, level);
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
    if (ch < 0 || ch >= NTG || level < 0 || level >= MAXLEV) return nullptr;
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
