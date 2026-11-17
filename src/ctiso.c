/* ufsm ct-iso: the per-scan air / papyrus cutoff (DESIGN 2026-11-17 "per-scroll air iso"; table scripts/unroll/air_iso.tsv).
   ufsm ct-iso <root> <key> <level> [--grid G] [--n N] [--hist-out F] [--scroll NAME] [--cache DIR] [--header 1]
   ufsm ct-iso --from-hist F [--scroll NAME] [--header 1]       (F: ct-hist lines, one 256-count histogram per box)
   The CT of a G^3 grid of N^3 boxes over the level (default 8^3 boxes of 128^3, chunk-aligned; boxes under 10 % nonzero skipped, as
   ct-hist) pooled into one histogram, 0 (masked air) dropped, smoothed by a gaussian (sigma 3 grey levels):
     air    = the highest local maximum in [32, 90] (every scan we hold, 2.2-9.4 um, 53-116 keV: 39-55; a narrow constant-fill spike
              below 32 on the 2.215 um 0500P2 is outside the range)
     valley = walking down from air to the first local minimum
     pap    = walking up from the valley to the first local maximum (MAN5 has a second, brighter mode at ~172: not this one)
     mid    = (air + pap + 1) / 2, the half-way iso of a voxel half air half papyrus: the packed-region estimate (gaps there are partial
              volume and never reach the air peak)
     depth  = smoothed count at the valley / the smaller peak (1: no valley; > 0.85 is reported as method valley-weak)
   One TSV row on stdout: scroll key level air pap valley mid depth boxes method (no air maximum: "-" fields, method no-air, exit 3). */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "optcheck.h"
#include "store.h"
#include "zarr3.h"
#include "ctiso.h"

static const char *io_opt(int argc, char **argv, const char *k, const char *d) { for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], opt_n(k))) return argv[i + 1]; return d; }

int ct_iso_from_hist(const double h0[256], ct_iso_t *r) {
    double h[256], p[256], g[25], gs = 0;
    memcpy(h, h0, sizeof h); h[0] = 0;
    for (int k = -12; k <= 12; k++) gs += g[k + 12] = exp(-0.5 * (k / 3.0) * (k / 3.0));
    for (int i = 0; i < 256; i++) { double s = 0; for (int k = -12; k <= 12; k++) { const int j = i + k; if (j >= 0 && j < 256) s += h[j] * g[k + 12]; } p[i] = s / gs; }
    memset(r, 0, sizeof *r); r->air = r->pap = r->valley = r->mid = -1; r->depth = 1;
    for (int i = 32; i <= 90; i++) if (p[i] >= p[i - 1] && p[i] > p[i + 1] && (r->air < 0 || p[i] > p[r->air])) r->air = i;
    if (r->air < 0) return 1;
    int v = r->air; while (v < 250 && p[v + 1] < p[v]) v++;
    int k = v; while (k < 250 && p[k + 1] >= p[k]) k++;
    r->valley = v; r->pap = k; r->mid = (r->air + k + 1) / 2;
    const double lo = p[r->air] < p[k] ? p[r->air] : p[k]; r->depth = lo > 0 ? p[v] / lo : 1;
    return 0;
}

int cmd_ct_iso(int argc, char **argv) {
    const char *fh = io_opt(argc, argv, "--from-hist", nullptr), *ho = io_opt(argc, argv, "--hist-out", nullptr), *sc = io_opt(argc, argv, "--scroll", "-");
    const int G = opt_atoi("--grid", io_opt(argc, argv, "--grid", "8")), N = opt_atoi("--n", io_opt(argc, argv, "--n", "128"));
    if ((!fh && (argc < 5 || argv[2][0] == '-')) || G < 2 || G > 64 || N < 16 || N > 512) {
        fprintf(stderr, "usage: ufsm ct-iso <root> <key> <level> [--grid G (8, 2..64)] [--n N (128)] [--hist-out F] [--scroll NAME] [--cache DIR] [--header 1]\n"
                        "       ufsm ct-iso --from-hist F [--scroll NAME] [--header 1]\n"); return 2; }
    double H[256] = {0}; int nb = 0; const char *key = "-", *lv = "-";
    if (fh) {
        FILE *f = fopen(fh, "r"); if (!f) { fprintf(stderr, "ct-iso: cannot read %s\n", fh); return 1; }
        for (;;) { double v[256]; int k = 0; for (; k < 256; k++) if (fscanf(f, "%lf", &v[k]) != 1) break;
                   if (k == 0) break; if (k != 256) { fprintf(stderr, "ct-iso: %s: a line with %d counts\n", fh, k); fclose(f); return 1; }
                   for (k = 0; k < 256; k++) H[k] += v[k]; nb++; }
        fclose(f);
    } else {
        key = argv[3]; lv = argv[4]; char kl[2048]; snprintf(kl, sizeof kl, "%s/%s", key, lv);   /* (the level: its array name, 0..5 or e.g. 9.6) */
        z3 *z = z3_open(store_open(argv[2]), kl, io_opt(argc, argv, "--cache", nullptr));
        if (!z) { fprintf(stderr, "ct-iso: %s\n", z3_error()); return 1; }
        const int64_t *S = z3_meta_of(z)->shape; const size_t T = (size_t)N * N * N;
        uint8_t *b = malloc(T); FILE *fo = ho ? fopen(ho, "w") : nullptr;
        if (!b || (ho && !fo)) { fprintf(stderr, "ct-iso: %s\n", b ? "cannot write --hist-out" : "out of memory"); free(b); z3_close(z); return 1; }
        for (int a = 0; a < G; a++) for (int c = 0; c < G; c++) for (int d = 0; d < G; d++) {
            const int ix[3] = {a, c, d}; int64_t o[3]; const int64_t n[3] = {N, N, N}; int ok = 1;
            for (int q = 0; q < 3; q++) { if (S[q] < N) ok = 0; else o[q] = (S[q] - N) * ix[q] / (G - 1) / 128 * 128; }
            if (!ok) continue;
            if (z3_read(z, o, n, b, 16)) { fprintf(stderr, "ct-iso: %s\n", z3_error()); free(b); if (fo) fclose(fo); z3_close(z); return 1; }
            size_t h[256] = {0}; for (size_t i = 0; i < T; i++) h[b[i]]++;
            if (T - h[0] < T / 10) continue;
            for (int k = 0; k < 256; k++) { H[k] += (double)h[k]; if (fo) fprintf(fo, "%zu%c", h[k], k == 255 ? '\n' : ' '); }
            nb++;
        }
        free(b); if (fo) fclose(fo); z3_close(z);
    }
    if (opt_atoi("--header", io_opt(argc, argv, "--header", "0"))) printf("scroll\tkey\tlevel\tair\tpap\tvalley\tmid\tdepth\tboxes\tmethod\n");
    if (!nb) { fprintf(stderr, "ct-iso: no box with 10 %% material\n"); return 1; }
    ct_iso_t r; const int rc = ct_iso_from_hist(H, &r);
    if (rc) { printf("%s\t%s\t%s\t-\t-\t-\t-\t-\t%d\tno-air\n", sc, key, lv, nb); return 3; }
    printf("%s\t%s\t%s\t%d\t%d\t%d\t%d\t%.2f\t%d\t%s\n", sc, key, lv, r.air, r.pap, r.valley, r.mid, r.depth, nb, r.depth > 0.85 ? "valley-weak" : "valley");
    return 0;
}
