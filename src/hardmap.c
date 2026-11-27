/* ufsm hard-map (hardspot 2026-11-27): the sampler's hard-spot map, a port of villa's hard_fraction (tracers/sheets data.hard_windows:
   a fixed share of the training episodes starts at a recorded failure of a model). Ours marks cells of 2^lev level-0 voxels of a scan
   where the winding is hard, from signals we already have:
     --mf ZARR    the merge-flag plane (ufsm merge-flag; uint8 per field cell, 0 off material): cell weight 255 x the share of its
                  material field cells (mf > 0) at mf >= --mf-thr, kept when that share is >= --mf-min (and >= 64 material cells);
                  the field frame from --fields FIELDS.json (um, h, origin; default ZARR/../fields.json)
     --boxes TSV  "z0 y0 x0 z1 y1 x1 w" per line (level-0 voxels of the --um frame, half-open, w in (0, 1]): per-box errors (wbench), the
                  unroll's id-change places, disagreement boxes; a cell takes max(its weight, 255 w) where a box overlaps it
   Out: RAW (uint8, grid z y x, C order) and RAW.json {"grid", "lev", "um", "ct", ...}: a source's "hard": {"map": RAW, "frac": F}
   (src/sources.h). The sampler maps its draw cells into this frame (the same scan at a power-of-two voxel size). */
#include "hardmap.h"
#include "json.h"
#include "optcheck.h"
#include "store.h"
#include "sysinfo.h"
#include "zarr3.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *opt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], opt_n(name))) return argv[i + 1];
    return dflt;
}
static json *read_json(const char *p) {
    FILE *f = fopen(p, "rb"); if (!f) return nullptr;
    fseek(f, 0, SEEK_END); const long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = n > 0 ? malloc((size_t)n) : nullptr; json *j = b && fread(b, 1, (size_t)n, f) == (size_t)n ? json_parse(b, (size_t)n) : nullptr;
    free(b); fclose(f); return j;
}
static int parse3(const char *s, int64_t v[3]) { long long a, b, c; if (!s || sscanf(s, "%lld,%lld,%lld", &a, &b, &c) != 3) return -1; v[0] = a; v[1] = b; v[2] = c; return 0; }

/* the mf plane's counts per out cell: m[k] material field cells, f[k] of them >= thr. Field cell (c) -> level-0 voxel of the field scan
   origin + (c + 0.5) h -> out frame x fum / um -> cell / 2^lev. */
static int mf_counts(const char *zpath, const char *fjp, double um, int lev, const int64_t G[3], int thr, int nt, uint32_t *m, uint32_t *f) {
    char fj[1400]; if (fjp) snprintf(fj, sizeof fj, "%s", fjp); else { snprintf(fj, sizeof fj, "%s", zpath); char *e = strrchr(fj, '/'); if (!e) return -1; snprintf(e, sizeof fj - (size_t)(e - fj), "/fields.json"); }
    json *j = read_json(fj); if (!j) { fprintf(stderr, "hard-map: cannot read %s (the mf plane's field frame; --fields)\n", fj); return -1; }
    const double fum = json_num(json_get(j, "um"), 0); const int h = (int)json_num(json_get(j, "h"), 0); int64_t org[3];
    for (int d = 0; d < 3; d++) org[d] = (int64_t)json_num(json_at(json_get(j, "origin"), (size_t)d), 0);
    json_free(j);
    if (!(fum > 0) || h <= 0) { fprintf(stderr, "hard-map: %s: no um / h\n", fj); return -1; }
    char root[1400]; snprintf(root, sizeof root, "%s", zpath); char *e = strrchr(root, '/'); if (!e) return -1; *e = 0;
    store *st = store_open(root); z3 *z = st ? z3_open(st, e + 1, nullptr) : nullptr;
    if (!z) { fprintf(stderr, "hard-map: cannot open %s: %s\n", zpath, z3_error()); return -1; }
    const z3_meta *mt = z3_meta_of(z); const int64_t D = mt->shape[0], H = mt->shape[1], W = mt->shape[2];
    const double sc = fum / um / ldexp(1.0, lev);   /* field-scan level-0 voxel -> out cell */
    const int64_t SL = 64; uint8_t *buf = malloc((size_t)SL * H * W); int64_t *cy = malloc((size_t)H * sizeof *cy), *cx = malloc((size_t)W * sizeof *cx);
    if (!buf || !cy || !cx) { free(buf); free(cy); free(cx); z3_close(z); return -1; }
    for (int64_t y = 0; y < H; y++) cy[y] = (int64_t)floor((org[1] + (y + 0.5) * h) * sc);
    for (int64_t x = 0; x < W; x++) cx[x] = (int64_t)floor((org[2] + (x + 0.5) * h) * sc);
    fprintf(stderr, "hard-map: %s: %lld x %lld x %lld field cells of %d x %g um, %.2f per out cell side\n", zpath, (long long)D, (long long)H, (long long)W, h, fum, 1.0 / (sc * h));
    int rc = 0;
    for (int64_t z0 = 0; z0 < D && !rc; z0 += SL) {
        const int64_t nz = z0 + SL <= D ? SL : D - z0, o[3] = {z0, 0, 0}, n[3] = {nz, H, W};
        if (z3_read(z, o, n, buf, nt)) { fprintf(stderr, "hard-map: %s: read at z %lld: %s\n", zpath, (long long)z0, z3_error()); rc = -1; break; }
        for (int64_t zz = 0; zz < nz; zz++) {
            const int64_t gz = (int64_t)floor((org[0] + (z0 + zz + 0.5) * h) * sc); if (gz < 0 || gz >= G[0]) continue;
            const uint8_t *p = buf + (size_t)zz * H * W;
            for (int64_t y = 0; y < H; y++) { if (cy[y] < 0 || cy[y] >= G[1]) continue; const size_t rb = ((size_t)gz * G[1] + cy[y]) * G[2];
                for (int64_t x = 0; x < W; x++) { const uint8_t v = p[y * W + x]; if (!v || cx[x] < 0 || cx[x] >= G[2]) continue; m[rb + cx[x]]++; f[rb + cx[x]] += v >= thr; } }
        }
        if ((z0 / SL) % 16 == 15) fprintf(stderr, "hard-map: z %lld / %lld\n", (long long)(z0 + nz), (long long)D);
    }
    free(buf); free(cy); free(cx); z3_close(z);
    return rc;
}

int cmd_hard_map(int argc, char **argv) {
    const char *out = opt(argc, argv, "--out", nullptr), *mfz = opt(argc, argv, "--mf", nullptr), *bx = opt(argc, argv, "--boxes", nullptr);
    const double um = opt_atof("--um", opt(argc, argv, "--um", "0")); const int lev = opt_atoi("--lev", opt(argc, argv, "--lev", "7"));
    int64_t shp[3];
    if (!out || !(um > 0) || lev < 0 || lev > 12 || parse3(opt(argc, argv, "--shape", nullptr), shp) || (!mfz && !bx)) {
        fprintf(stderr, "usage: ufsm hard-map --out RAW --um U --shape Z,Y,X (level-0 voxels of the scan) [--lev 7] [--ct NAME]\n"
                        "         [--mf ZARR [--fields FIELDS.json] [--mf-thr 179] [--mf-min 0.1]] [--boxes TSV (z0 y0 x0 z1 y1 x1 w)] [--threads 6]\n"
                        "  the sampler's hard-spot map (a source's \"hard\": {\"map\": RAW, \"frac\": F}): cells of 2^lev voxels, 0 easy, 1..255 hard weight\n");
        return 2;
    }
    const int thr = opt_atoi("--mf-thr", opt(argc, argv, "--mf-thr", "179")), nt = ufsm_threads(opt_atoi("--threads", opt(argc, argv, "--threads", "6")), 6);
    const double mfmin = opt_atof("--mf-min", opt(argc, argv, "--mf-min", "0.1"));
    const char *ct = opt(argc, argv, "--ct", ""), *fj = opt(argc, argv, "--fields", nullptr);
    const int64_t cs = (int64_t)1 << lev, G[3] = {(shp[0] + cs - 1) / cs, (shp[1] + cs - 1) / cs, (shp[2] + cs - 1) / cs};
    const size_t tot = (size_t)G[0] * G[1] * G[2];
    uint8_t *w = calloc(tot, 1); if (!w) return 1;
    size_t nmf = 0, nbx = 0, nmat = 0;
    if (mfz) {
        uint32_t *m = calloc(tot, sizeof *m), *f = calloc(tot, sizeof *f); if (!m || !f) { free(m); free(f); free(w); return 1; }
        if (mf_counts(mfz, fj, um, lev, G, thr, nt, m, f)) { free(m); free(f); free(w); return 1; }
        for (size_t k = 0; k < tot; k++) { if (m[k] < 64) continue; nmat++; const double sh = (double)f[k] / m[k];
            if (sh >= mfmin) { const long v = lround(255 * sh); w[k] = (uint8_t)(v < 1 ? 1 : v > 255 ? 255 : v); nmf++; } }
        free(m); free(f);
        fprintf(stderr, "hard-map: mf >= %d: %zu of %zu material cells hard (share >= %g)\n", thr, nmf, nmat, mfmin);
    }
    if (bx) {
        FILE *t = fopen(bx, "r"); if (!t) { fprintf(stderr, "hard-map: cannot read %s\n", bx); free(w); return 1; }
        char line[512]; int ln = 0;
        while (fgets(line, sizeof line, t)) {
            ln++; if (line[0] == '#' || line[0] == '\n') continue;
            long long b[6]; double bw;
            if (sscanf(line, "%lld %lld %lld %lld %lld %lld %lf", b, b + 1, b + 2, b + 3, b + 4, b + 5, &bw) != 7 || !(bw > 0 && bw <= 1)) {
                fprintf(stderr, "hard-map: %s:%d: not \"z0 y0 x0 z1 y1 x1 w\" with w in (0, 1]\n", bx, ln); fclose(t); free(w); return 1; }
            int64_t lo[3], hi[3]; int ok = 1;
            for (int d = 0; d < 3; d++) { lo[d] = b[d] / cs; hi[d] = (b[3 + d] + cs - 1) / cs; if (lo[d] < 0) lo[d] = 0; if (hi[d] > G[d]) hi[d] = G[d]; ok &= b[3 + d] > b[d] && lo[d] < hi[d]; }
            if (!ok) { fprintf(stderr, "hard-map: %s:%d: empty or outside the grid\n", bx, ln); fclose(t); free(w); return 1; }
            const long v = lround(255 * bw); const uint8_t q = (uint8_t)(v < 1 ? 1 : v);
            for (int64_t z = lo[0]; z < hi[0]; z++) for (int64_t y = lo[1]; y < hi[1]; y++) for (int64_t x = lo[2]; x < hi[2]; x++) {
                uint8_t *c = &w[((size_t)z * G[1] + y) * G[2] + x]; if (*c < q) *c = q; }
            nbx++;
        }
        fclose(t);
    }
    size_t nh = 0; double ws = 0; for (size_t k = 0; k < tot; k++) if (w[k]) { nh++; ws += w[k]; }
    FILE *o = fopen(out, "wb"); if (!o || fwrite(w, 1, tot, o) != tot || fclose(o)) { fprintf(stderr, "hard-map: cannot write %s\n", out); free(w); return 1; }
    char jp[1400]; snprintf(jp, sizeof jp, "%s.json", out); FILE *jf = fopen(jp, "w");
    if (!jf) { fprintf(stderr, "hard-map: cannot write %s\n", jp); free(w); return 1; }
    fprintf(jf, "{\"format\": \"ufsm-hard-map\", \"grid\": [%lld, %lld, %lld], \"lev\": %d, \"um\": %.6g, \"ct\": \"%s\", \"hard_cells\": %zu, \"mean_w\": %.1f, "
                "\"mf\": \"%s\", \"mf_thr\": %d, \"mf_min\": %g, \"mf_cells\": %zu, \"material_cells\": %zu, \"boxes\": \"%s\", \"nboxes\": %zu}\n",
            (long long)G[0], (long long)G[1], (long long)G[2], lev, um, ct, nh, nh ? ws / nh : 0., mfz ? mfz : "", thr, mfmin, nmf, nmat, bx ? bx : "", nbx);
    if (fclose(jf)) { free(w); return 1; }
    fprintf(stderr, "hard-map: %s: %lld x %lld x %lld cells of %lld voxels at %g um, %zu hard (%.4f of the grid, %.4f of the material cells), mean weight %.1f\n", out,
            (long long)G[0], (long long)G[1], (long long)G[2], (long long)cs, um, nh, (double)nh / tot, nmat ? (double)nh / nmat : NAN, nh ? ws / nh : 0.);
    free(w);
    return 0;
}
