/* ufsm holdout-cover SOURCES.json [--holdouts H.json] [--hist N] [--per-box 1] [--zhist-out F]: how much of each source's labels the held-out boxes cover (2026-11-14,
   datafix: the per-scroll holdouts). A source's labels are its draw cells (sample.c source_occupancy: stored-phase chunks, label cells,
   or the CT cells holding papyrus inside a target's chunks), each a 2^lev level-0 cube; the share of their volume inside the source's
   boxes (its own "holdout" / "holdout_file" ones, or --holdouts H matched by CT basename as sources.c does) per source and per scroll
   (the name before its first '-'). --hist N: each source's cells along z in N bins of its level-0 extent (where to put a slab); --per-box: each box's share alone; --zhist-out F: per source a line "name ct lev cells held ZB" then
   its labels outside the boxes per ZB (256) level-0 z rows, as shares of all its cells (a full-yx z slab adds exactly the sum over its rows). */
#include "sources.h"
#include "optcheck.h"
#include "sample.h"
#include "json.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static const char *hc_opt(int argc, char **argv, const char *k, const char *d) { for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], opt_n(k))) return argv[i + 1]; return d; }
static const char *hc_base(const char *p) {   /* sources.c src_base: the last path component past trailing slashes / level digits */
    static char b[512]; snprintf(b, sizeof b, "%s", p ? p : "");   /* mgpu: main (host-side string helper, not on the GPU threads) */
    for (;;) { size_t n = strlen(b); while (n && b[n - 1] == '/') b[--n] = 0;
        char *sl = strrchr(b, '/'); const char *t = sl ? sl + 1 : b; int dig = *t != 0; for (const char *c = t; *c; c++) dig &= *c >= '0' && *c <= '9';
        if (dig && sl) { *sl = 0; continue; } return t; }
}
int cmd_holdout_cover(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: ufsm holdout-cover SOURCES.json [--holdouts H.json] [--hist N]\n"); return 2; }
    sources *S = sources_load(argv[2]); if (!S) return 1;
    const char *hp = hc_opt(argc, argv, "--holdouts", nullptr); const int nh = opt_atoi("--hist", hc_opt(argc, argv, "--hist", "0")), perbox = opt_atoi("--per-box", hc_opt(argc, argv, "--per-box", "0"));
    const char *zo = hc_opt(argc, argv, "--zhist-out", nullptr); const int64_t ZB = 256; FILE *zf = zo ? fopen(zo, "w") : nullptr;
    if (zo && !zf) { fprintf(stderr, "cannot write %s\n", zo); return 1; }
    json *hj = nullptr;
    if (hp) { FILE *f = fopen(hp, "rb"); if (!f) { fprintf(stderr, "cannot open %s\n", hp); return 1; }
        fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET); char *t = malloc((size_t)n + 1); const size_t rd = fread(t, 1, (size_t)n, f); fclose(f);
        hj = json_parse(t, rd); free(t); if (!hj || !json_get(hj, "boxes")) { fprintf(stderr, "%s: no \"boxes\"\n", hp); return 1; } }
    char scn[64][64]; double sct[64][2]; int nsc = 0;
    for (int i = 0; i < S->n; i++) {
        source *s = &S->src[i];
        int64_t (*bx)[6] = nullptr; int nb = 0;
        if (hj) { const json *b = json_get(hj, "boxes"); char me[512]; snprintf(me, sizeof me, "%s", hc_base(s->ct_key));
            for (size_t k = 0; k < b->n; k++) { const json *e = json_at(b, k); const json *l0 = json_get(e, "l0");
                if (strcmp(hc_base(json_str(json_get(e, "ct"), "")), me) || !l0 || l0->n != 6) continue;
                bx = realloc(bx, (size_t)(nb + 1) * sizeof *bx); for (int d = 0; d < 6; d++) bx[nb][d] = (int64_t)json_num(json_at(l0, (size_t)d), 0); nb++; } }
        else { nb = s->nhold; bx = malloc((size_t)(nb ? nb : 1) * sizeof *bx); for (int k = 0; k < nb; k++) memcpy(bx[k], s->hold[k], sizeof bx[k]); }
        occ_index o = source_occupancy(s);
        if (!o.n) { fprintf(stderr, "holdout-cover: %s: no draw cells\n", s->name); free(bx); continue; }
        const int64_t c = (int64_t)1 << o.lev; double tot = 0, in = 0; double *hz = nh > 0 ? calloc((size_t)nh, sizeof *hz) : nullptr;
        const int64_t zext = o.shape[0] * c; const int64_t nzb = (zext + ZB - 1) / ZB;
        double *pb = perbox && nb ? calloc((size_t)nb, sizeof *pb) : nullptr, *zu = zf ? calloc((size_t)nzb, sizeof *zu) : nullptr;
        for (size_t k = 0; k < o.n; k++) {
            const uint32_t id = o.idx[k]; const int64_t z = id / (o.shape[1] * o.shape[2]), y = (id / o.shape[2]) % o.shape[1], x = id % o.shape[2];
            const int64_t a[3] = {z * c, y * c, x * c}; double best = 0;
            for (int b = 0; b < nb; b++) { double v = 1; for (int d = 0; d < 3 && v > 0; d++) { const int64_t lo = a[d] > bx[b][d] ? a[d] : bx[b][d], hi = a[d] + c < bx[b][d] + bx[b][3 + d] ? a[d] + c : bx[b][d] + bx[b][3 + d];
                    v *= hi > lo ? (double)(hi - lo) / (double)c : 0; } if (v > best) best = v; if (pb) pb[b] += v; }   /* (boxes may overlap: the largest share, a lower bound) */
            tot += 1; in += best;
            if (zu) { const int64_t q0 = a[0] / ZB, q1 = (a[0] + c - 1) / ZB; for (int64_t q = q0; q <= q1 && q < nzb; q++) zu[q] += (1 - best) / (double)(q1 - q0 + 1); }
            if (hz) { int q = (int)((double)a[0] / (double)zext * nh); if (q >= nh) q = nh - 1; hz[q] += 1; }
        }
        printf("%-24s %3d boxes, %9zu cells (level %d, %lld^3 level-0), held out %.4f\n", s->name, nb, o.n, o.lev, (long long)c, in / tot);
        if (pb) { for (int b = 0; b < nb; b++) printf("   box %d [%lld %lld %lld %lld %lld %lld] alone %.4f\n", b, (long long)bx[b][0], (long long)bx[b][1], (long long)bx[b][2],
                                                       (long long)bx[b][3], (long long)bx[b][4], (long long)bx[b][5], pb[b] / tot); free(pb); }
        if (zu) { fprintf(zf, "%s %s %d %.0f %.6f %lld", s->name, hc_base(s->ct_key), o.lev, tot, in / tot, (long long)ZB); for (int64_t q = 0; q < nzb; q++) fprintf(zf, " %.6g", zu[q] / tot); fprintf(zf, "\n"); free(zu); }
        if (hz) { printf("   z (level-0, %lld per bin):", (long long)(zext / nh)); for (int q = 0; q < nh; q++) printf(" %.3f", hz[q] / tot); printf("\n"); free(hz); }
        char sc_[64]; snprintf(sc_, sizeof sc_, "%s", s->name); char *dash = strchr(sc_, '-'); if (dash) *dash = 0;
        int k = 0; while (k < nsc && strcmp(scn[k], sc_)) k++;
        if (k == nsc && nsc < 64) { snprintf(scn[nsc], 64, "%s", sc_); sct[nsc][0] = sct[nsc][1] = 0; nsc++; }
        if (k < 64) { sct[k][0] += tot; sct[k][1] += in; }   /* (cells of different sources summed: a scroll's labels by cell count) */
        free(o.idx); free(bx);
    }
    for (int k = 0; k < nsc; k++) printf("scroll %-14s held out %.4f of %.0f label cells\n", scn[k], sct[k][1] / sct[k][0], sct[k][0]);
    if (zf) fclose(zf);
    json_free(hj); sources_free(S);
    return 0;
}
