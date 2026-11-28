/* ctshell.c (inkeval 2026-11-27): CT shell lists (see ctshell.h): the writer shared by ink-ext --shell-list and ct-shell-list, the
   chunk-mask reader shared by ink-ext --ct-shell and predict --ct-shell, and the ct-shell-list command (a predict box's shell). */
#include "ctshell.h"
#include "optcheck.h"
#include "sources.h"
#include "store.h"
#include "zarr3.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { CE = 128 };
static const double LL_BPV = 0.70;   /* lossless bytes per shell voxel: 1447 w058 5.19 GB / 7.5 Gvox = 0.69 (S3 bytes at volcomp q0) */
static int i64cmp(const void *a, const void *b) { const int64_t p = *(const int64_t *)a, q = *(const int64_t *)b; return (p > q) - (p < q); }
static size_t uniq(int64_t *k, size_t n) { qsort(k, n, sizeof *k, i64cmp); size_t w = 0; for (size_t i = 0; i < n; i++) if (!w || k[i] != k[w - 1]) k[w++] = k[i]; return w; }
static const int64_t SHARD_MASK = ~(int64_t)0 ^ (7ll << 42 | 7ll << 21 | 7);   /* a chunk key's 1024^3 shard: its low 3 bits per axis cleared */

int ctshell_write(const char *slist, int lev, int64_t *keys, size_t n, const int64_t cgr[3], const char *sprev, const char *sq, double maxgb,
                  const char *desc, int *q_out) {
    if (strcmp(sq, "auto") && (!isdigit((unsigned char)sq[0]) || sq[1])) { fprintf(stderr, "ctshell: quality auto | 0..9, not %s\n", sq); free(keys); return 2; }
    size_t u = 0;
    for (size_t i = 0; i < n; i++) { const int64_t k = keys[i]; if (k >> 42 < cgr[0] && ((k >> 21) & 0x1fffff) < cgr[1] && (k & 0x1fffff) < cgr[2]) keys[u++] = k; }   /* (outside the array: read as zero) */
    u = uniq(keys, u);
    const size_t unew = u;
    /* sprev (a shell grown run by run): its chunks are merged in; a shard it had that gains a chunk goes into SLIST.redo ("sz sy sx"): its
       written shard file must go before latvol --resume 1 rewrites it with the larger set. Another quality (or none in its header): all of them */
    int64_t *osh = nullptr, *gsh = nullptr; size_t nos = 0, ng = 0, nredo = 0, no = 0; int pq = -1; char rp[1400] = "";
    if (sprev) { FILE *pf = fopen(sprev, "r"); char ln[512]; snprintf(rp, sizeof rp, "%s.redo", slist);
        if (!pf) { fprintf(stderr, "ctshell: --shell-prev %s: cannot open\n", sprev); free(keys); return 1; }
        int64_t *old = nullptr; size_t oc = 0;
        while (fgets(ln, sizeof ln, pf)) { long long v[6]; int pl, psh, pqq;
            if (ln[0] == '#') { if (sscanf(ln, "# level %d shard %d q %d", &pl, &psh, &pqq) == 3) pq = pqq;
                if (sscanf(ln, "# level %d shard %d", &pl, &psh) == 2 && (pl != lev || psh != 1024)) {
                    fprintf(stderr, "ctshell: --shell-prev %s: level %d shard %d, this list level %d shard 1024\n", sprev, pl, psh, lev); fclose(pf); free(keys); free(old); return 1; }
                continue; }
            if (sscanf(ln, "%lld %lld %lld %lld %lld %lld", v, v + 1, v + 2, v + 3, v + 4, v + 5) != 6) continue;
            int64_t c0[3], c1[3]; for (int d = 0; d < 3; d++) { c0[d] = v[d] / CE; c1[d] = (v[d] + v[3 + d] + CE - 1) / CE; if (c1[d] > cgr[d]) c1[d] = cgr[d]; }
            for (int64_t z = c0[0]; z < c1[0]; z++) for (int64_t y = c0[1]; y < c1[1]; y++) for (int64_t x = c0[2]; x < c1[2]; x++) {
                if (no == oc) { oc = oc ? 2 * oc : 4096; old = realloc(old, oc * sizeof *old); } old[no++] = ctshell_key(z, y, x); } }
        fclose(pf);
        no = uniq(old, no);
        osh = malloc((no ? no : 1) * sizeof *osh); gsh = malloc((u ? u : 1) * sizeof *gsh);
        for (size_t i = 0; i < no; i++) osh[nos++] = old[i] & SHARD_MASK;
        nos = uniq(osh, nos);
        for (size_t i = 0; i < u; i++) { const int64_t k = keys[i], s3 = k & SHARD_MASK;
            if (!bsearch(&k, old, no, sizeof *old, i64cmp) && bsearch(&s3, osh, nos, sizeof *osh, i64cmp)) gsh[ng++] = s3; }
        ng = uniq(gsh, ng);
        keys = realloc(keys, (u + no + 1) * sizeof *keys); if (no) memcpy(keys + u, old, no * sizeof *old); u = uniq(keys, u + no);
        free(old);
        fprintf(stderr, "ctshell: --shell-prev %s: %zu chunks before, %zu after\n", sprev, no, u); }
    /* the quality (lead 11-27: lossless while it fits, a whole-scroll lossless shell, 1447 ~240 GB, would starve /vesuvius) */
    const double est = (double)u * CE * CE * CE * LL_BPV / 1e9; int q;
    if (strcmp(sq, "auto")) q = sq[0] - '0';
    else { q = est <= maxgb ? 0 : 2; if (q) fprintf(stderr, "DEFAULTED: ctshell q2 (lossless est %.3g GB > %.3g)\n", est, maxgb); }
    fprintf(stderr, "ctshell: shell quality q%d (%s; lossless estimate %.3g GB, limit %.3g)\n", q, strcmp(sq, "auto") ? "given" : "auto", est, maxgb);
    if (sprev) { FILE *redo = fopen(rp, "w"); if (!redo) { fprintf(stderr, "ctshell: cannot write %s\n", rp); free(keys); free(osh); free(gsh); return 1; }
        const int full = pq != q; const int64_t *rs = full ? osh : gsh; const size_t rn = full ? nos : ng;
        for (size_t i = 0; i < rn; i++) { fprintf(redo, "%lld %lld %lld\n", (long long)((rs[i] >> 42) >> 3), (long long)(((rs[i] >> 21) & 0x1fffff) >> 3), (long long)((rs[i] & 0x1fffff) >> 3)); nredo++; }
        if (fclose(redo)) { fprintf(stderr, "ctshell: cannot write %s\n", rp); free(keys); free(osh); free(gsh); return 1; }
        fprintf(stderr, "ctshell: --shell-prev %s: %zu shards to rewrite (%s)%s\n", sprev, nredo, rp,
                full ? pq < 0 ? ": OLD has no quality in its header, all rewritten" : ": OLD's quality differs, all rewritten" : ""); }
    free(osh); free(gsh);
    FILE *f = fopen(slist, "w"); size_t nrun = 0;
    if (!f) { fprintf(stderr, "ctshell: cannot write %s\n", slist); free(keys); return 1; }
    fprintf(f, "# level %d shard 1024 q %d (%zu chunks of 128^3%s, x runs; latvol --exact 1 --volcomp q; %s)\n", lev, q, u, sprev ? " incl. --shell-prev" : "", desc);
    for (size_t i = 0; i < u;) {   /* (keys sorted by z, y, x: a run continues while x + 1 follows in the same row and shard) */
        size_t j = i + 1; while (j < u && keys[j] == keys[j - 1] + 1 && (keys[j] & 0x1fffff) % 8) j++;
        const int64_t k = keys[i];
        fprintf(f, "%lld %lld %lld %d %d %lld\n", (long long)((k >> 42) * CE), (long long)(((k >> 21) & 0x1fffff) * CE), (long long)((k & 0x1fffff) * CE), CE, CE, (long long)((j - i) * CE));
        nrun++; i = j; }
    if (fclose(f)) { fprintf(stderr, "ctshell: cannot write %s\n", slist); free(keys); return 1; }
    fprintf(stderr, "ctshell: %s: level %d, %zu chunks (%.1f Gvox; %zu new) in %zu runs\n", slist, lev, u, u * (double)CE * CE * CE / 1e9, unew, nrun);
    if (q_out) *q_out = q;
    free(keys); return 0;
}

int ctshell_mask(const char *path, const int64_t cgr[3], const int64_t shape[3], int *level, int *q, uint8_t **mask, size_t *nck, char *err, size_t errn) {
    *level = -1; *q = -1; *mask = nullptr; *nck = 0;
    FILE *bf = fopen(path, "r"); if (!bf) { snprintf(err, errn, "%s: cannot open", path); return 1; }
    uint8_t *mk = calloc((size_t)(cgr[0] * cgr[1] * cgr[2]), 1); char ln[512]; int ssh = 0, nb = 0, bad = 0;
    while (fgets(ln, sizeof ln, bf)) { long long v[6];
        if (ln[0] == '#') { int a, b, c; if (sscanf(ln, "# level %d shard %d q %d", &a, &b, &c) == 3) *q = c; if (sscanf(ln, "# level %d shard %d", &a, &b) == 2) { *level = a; ssh = b; } continue; }
        if (sscanf(ln, "%lld %lld %lld %lld %lld %lld", v, v + 1, v + 2, v + 3, v + 4, v + 5) != 6) continue;
        nb++; int64_t c0[3], c1[3];   /* (a box must be chunk-aligned: a partial chunk would be half-written in the shell) */
        for (int d = 0; d < 3; d++) { if (v[d] % CE || v[d] < 0 || v[3 + d] < 1 || (v[3 + d] % CE && v[d] + v[3 + d] < shape[d])) bad = 1;
            c0[d] = v[d] / CE; c1[d] = (v[d] + v[3 + d] + CE - 1) / CE; if (c1[d] > cgr[d]) c1[d] = cgr[d]; }
        if (bad) { snprintf(err, errn, "%s: box %lld %lld %lld %lld %lld %lld not on the 128^3 chunk grid", path, v[0], v[1], v[2], v[3], v[4], v[5]); break; }
        for (int64_t z = c0[0]; z < c1[0]; z++) for (int64_t y = c0[1]; y < c1[1]; y++) for (int64_t x = c0[2]; x < c1[2]; x++) {
            uint8_t *m = &mk[(z * cgr[1] + y) * cgr[2] + x]; *nck += !*m; *m = 1; } }
    fclose(bf);
    if (bad) { free(mk); return 2; }
    if (*level < 0 || ssh != 1024 || !nb) { snprintf(err, errn, "%s: level %d, shard %d, %d boxes", path, *level, ssh, nb); free(mk); return 1; }
    *mask = mk; return 0;
}

static const char *sopt_(int argc, char **argv, const char *name, const char *def) {
    for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], opt_n(name))) return argv[i + 1];
    return def;
}
/* ufsm ct-shell-list ROOT KEY OUT --level L --box Z,Y,X,NZ,NY,NX: the shell of a predict box. Every stored 128^3 chunk of level L within
   the box, rounded out to the --shard grid first (predict --full-grid computes whole output shards), grown by --pad voxels (default 1024:
   the windows' reach, window + halo per side; predict --ct-shell refuses a shell that misses one) */
int cmd_ct_shell_list(int argc, char **argv) {
    if (argc < 5 || argv[2][0] == '-') { fprintf(stderr, "usage: ufsm ct-shell-list ROOT KEY OUT --level L [--um U] --box Z,Y,X,NZ,NY,NX [--shard S] [--pad 1024] [--cache DIR]\n"
                                                          "                         [--shell-prev OLD] [--shell-q auto|Q] [--shell-max-gb 100]\n"
                                                          "  the CT shell list of a predict box (src/ctshell.h): latvol --exact 1 --volcomp Q --boxes OUT, then predict --ct-shell\n"); return 2; }
    const char *root = argv[2], *key = argv[3], *out = argv[4], *bs = sopt_(argc, argv, "--box", nullptr);
    const int level = opt_atoi("--level", sopt_(argc, argv, "--level", "0")); const long long pad = opt_atoll("--pad", sopt_(argc, argv, "--pad", "1024")), shd = opt_atoll("--shard", sopt_(argc, argv, "--shard", "0"));
    long long b[6];
    if (!bs || sscanf(bs, "%lld,%lld,%lld,%lld,%lld,%lld", b, b + 1, b + 2, b + 3, b + 4, b + 5) != 6 || pad < 0 || shd < 0) { fprintf(stderr, "ct-shell-list: --box Z,Y,X,NZ,NY,NX (and --pad >= 0)\n"); return 2; }
    store *s = store_open(root); z3 *z = s ? pyramid_open_level(s, key, level, opt_atof("--um", sopt_(argc, argv, "--um", "0")), sopt_(argc, argv, "--cache", nullptr)) : nullptr;
    if (!z) { fprintf(stderr, "ct-shell-list: cannot open level %d of %s %s: %s\n", level, root, key, z3_error()); return 1; }
    const z3_meta *m = z3_meta_of(z);
    if (m->chunk[0] != CE || m->chunk[1] != CE || m->chunk[2] != CE) { fprintf(stderr, "ct-shell-list: %s: chunks are not 128^3\n", key); return 1; }
    int64_t cgr[3], c0[3], c1[3];
    for (int d = 0; d < 3; d++) { cgr[d] = (m->shape[d] + CE - 1) / CE; long long lo = shd ? b[d] / shd * shd : b[d], hi = shd ? (b[d] + b[3 + d] + shd - 1) / shd * shd : b[d] + b[3 + d]; lo -= pad; hi += pad; if (lo < 0) lo = 0; if (hi > m->shape[d]) hi = m->shape[d];
        c0[d] = lo / CE; c1[d] = hi > lo ? (hi + CE - 1) / CE : c0[d]; }
    size_t cap = 4096, n = 0, nunk = 0, nall = 0; int64_t *k = malloc(cap * sizeof *k);
    for (int64_t cz = c0[0]; cz < c1[0]; cz++) for (int64_t cy = c0[1]; cy < c1[1]; cy++) for (int64_t cx = c0[2]; cx < c1[2]; cx++) {
        const int64_t o[3] = {cz * CE, cy * CE, cx * CE}, nn[3] = {CE, CE, CE}; const int r = z3_box_any(z, o, nn); nall++;
        if (r == 0) continue;   /* (air: no stored chunk, read as fill either way) */
        nunk += r < 0;
        if (n == cap) { cap *= 2; k = realloc(k, cap * sizeof *k); } k[n++] = ctshell_key(cz, cy, cx); }
    if (nunk) fprintf(stderr, "ct-shell-list: %zu chunks of unknown presence (index read errors): listed\n", nunk);
    fprintf(stderr, "ct-shell-list: %s %s level %d, box %s + %lld: %zu stored chunks of %zu\n", root, key, level, bs, pad, n, nall);
    char desc[1400]; snprintf(desc, sizeof desc, "ct-shell-list: box %s + %lld, %s %s", bs, pad, root, key);
    const int rc = ctshell_write(out, level, k, n, cgr, sopt_(argc, argv, "--shell-prev", nullptr), sopt_(argc, argv, "--shell-q", "auto"),
                                 opt_atof("--shell-max-gb", sopt_(argc, argv, "--shell-max-gb", "100")), desc, nullptr);
    z3_close(z); store_close(s);
    return rc;
}
