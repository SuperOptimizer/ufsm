/* src/occcache.c: the sampler's draw-cell index cached on disk for on-the-fly fibre and ink sources (occcache.h) */
#include "occcache.h"
#include "dcache.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

[[gnu::weak]] int dcache_key_dir(const char *, const char *, char *, size_t);
[[gnu::weak]] int inkfly_draw_key(void *, char *, size_t);
[[gnu::weak]] int inkfly_cells_sup(void *, int, const int64_t[3], uint8_t *, int);

static size_t occ_file(char *k, size_t at, size_t nk, const char *path) {   /* "path:size:mtime_ns" (missing: "path:-") */
    struct stat st; if (at >= nk) return at;
    if (!path || !*path) return at;
    if (stat(path, &st)) return at + (size_t)snprintf(k + at, nk - at, "%s:-;", path);
    return at + (size_t)snprintf(k + at, nk - at, "%s:%lld:%lld;", path, (long long)st.st_size, (long long)st.st_mtim.tv_sec * 1000000000LL + st.st_mtim.tv_nsec);
}
int occ_cache_key(const source *s, char *k, size_t nk) {
    int any = 0; for (int c = 0; c < NTG; c++) any |= s->tgt_fib[c] != nullptr || (s->ink_fly[c] && inkfly_draw_key);
    if (!any || nk < 64) return -1;
    size_t at = (size_t)snprintf(k, nk, "occ-index v1|%s|%s|um %.9g|levels %d %d|iso %d|", s->name ? s->name : "", s->ct_key ? s->ct_key : "", s->um, s->min_level, s->max_level, s->iso);
    for (int c = 0; c < NTG && at < nk; c++) { const struct fib_spec *f = s->tgt_fib[c];
        if (s->ink_fly[c]) {   /* an on-the-fly ink target (2026-11-20): its inputs key (the label cache dir) and held-out boxes, and the draw rule */
            const char *ev = getenv("UFSM_INK_DRAW_SUP"); const int sup = inkfly_cells_sup && !(ev && !strcmp(ev, "0"));
            if (!inkfly_draw_key) return -1;
            at += (size_t)snprintf(k + at, nk - at, "ch%d ink sup %d ", c, sup); if (at >= nk || inkfly_draw_key(s->ink_fly[c], k + at, nk - at)) return -1;
            at += strlen(k + at); if (at < nk) at += (size_t)snprintf(k + at, nk - at, "|"); continue; }
        if (!f) { if (s->tgt_key[c] && at < nk) at += (size_t)snprintf(k + at, nk - at, "ch%d key %s|", c, s->tgt_key[c]); continue; }
        if (at < nk) at += (size_t)snprintf(k + at, nk - at, "ch%d fibre r %.9g g %.9g ring %.9g iso %d piso %d plane %d afv ", c, f->rad, f->guard, f->ring, f->iso, f->piso, f->plane);
        at = occ_file(k, at, nk, f->afv);
        if (f->extra && *f->extra && at < nk) {   /* each extra file ("a.afv,b.afv@map") */
            at += (size_t)snprintf(k + at, nk - at, " extra "); char xs[4096]; snprintf(xs, sizeof xs, "%s", f->extra); char *sv = nullptr;
            for (char *x = strtok_r(xs, ",", &sv); x; x = strtok_r(nullptr, ",", &sv)) { const size_t L = strlen(x); const int vm = L > 4 && !strcmp(x + L - 4, "@map"); if (vm) x[L - 4] = 0;
                at = occ_file(k, at, nk, x); if (vm && at < nk) at += (size_t)snprintf(k + at, nk - at, "@map;"); } }
        if (f->map && at < nk) { at += (size_t)snprintf(k + at, nk - at, " map "); at = occ_file(k, at, nk, f->map); }
        if (f->tr && at < nk) { at += (size_t)snprintf(k + at, nk - at, " tr "); at = occ_file(k, at, nk, f->tr); }
        if (at < nk) at += (size_t)snprintf(k + at, nk - at, "|");
    }
    return at < nk ? 0 : -1;   /* (a truncated key would alias: no cache) */
}
static int occ_path(const char *dir, const char *key, char *p, size_t np) {
    char kd[1200]; if (!dir || !*dir || !dcache_key_dir || dcache_key_dir(dir, key, kd, sizeof kd)) return -1;
    snprintf(p, np, "%s/occ.bin", kd); return 0;
}
int occ_cache_load(const char *dir, const char *key, occ_index *o) {
    char p[1300]; if (occ_path(dir, key, p, sizeof p)) return -1;
    FILE *f = fopen(p, "rb"); if (!f) return -1;
    char m[4]; int32_t lev = 0; int64_t sh[3]; uint64_t n = 0, pn = 0;
    int ok = fread(m, 1, 4, f) == 4 && (!memcmp(m, "UOC1", 4) || !memcmp(m, "UOC2", 4)) && fread(&lev, 4, 1, f) == 1 && fread(sh, 8, 3, f) == 3 && fread(&n, 8, 1, f) == 1 && n > 0;
    const int v2 = ok && m[3] == '2';   /* (UOC2: + the positive cells, the stratified ink draw's list) */
    uint32_t *idx = ok ? malloc(n * sizeof *idx) : nullptr, *pidx = nullptr; ok = ok && idx && fread(idx, sizeof *idx, n, f) == n;
    if (ok && v2) { ok = fread(&pn, 8, 1, f) == 1 && pn <= n; if (ok && pn) { pidx = malloc(pn * sizeof *pidx); ok = pidx && fread(pidx, sizeof *pidx, pn, f) == pn; } }
    { char extra; ok = ok && fread(&extra, 1, 1, f) == 0; }   /* (nothing after the index: a truncated or padded file is a miss) */
    fclose(f);
    if (ok) { const uint64_t tot = (uint64_t)sh[0] * (uint64_t)sh[1] * (uint64_t)sh[2]; for (uint64_t i = 0; i < n && ok; i++) ok = idx[i] < tot; for (uint64_t i = 0; i < pn && ok; i++) ok = pidx[i] < tot; }
    if (!ok) { free(idx); free(pidx); return -1; }
    o->idx = idx; o->n = n; o->lev = lev; for (int d = 0; d < 3; d++) o->shape[d] = sh[d]; o->pidx = pidx; o->pn = pn;
    return 0;
}
int occ_cache_save(const char *dir, const char *key, const occ_index *o) {
    char p[1300], t[1400]; if (!o->n || occ_path(dir, key, p, sizeof p)) return -1;
    snprintf(t, sizeof t, "%s.%d.tmp", p, (int)getpid()); FILE *f = fopen(t, "wb"); if (!f) return -1;
    const int32_t lev = o->lev; const uint64_t n = o->n, pn = o->pidx ? o->pn : 0;
    int ok = fwrite("UOC2", 1, 4, f) == 4 && fwrite(&lev, 4, 1, f) == 1 && fwrite(o->shape, 8, 3, f) == 3 && fwrite(&n, 8, 1, f) == 1 && fwrite(o->idx, sizeof *o->idx, n, f) == n
             && fwrite(&pn, 8, 1, f) == 1 && (!pn || fwrite(o->pidx, sizeof *o->pidx, pn, f) == pn);
    if (fclose(f)) ok = 0;
    if (!ok || rename(t, p)) { unlink(t); return -1; }
    return 0;
}
