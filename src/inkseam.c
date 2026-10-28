/* ufsm ink-seam (see cmd_ink_seam) */
#include "store.h"
#include "z3w.h"
#include "zarr3.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ufsm ink-seam ZARR --um U [--period 512] [--offset 0] [--n 24] [--min 16] [--seed 1] [--threads 16]: window-seam strength of a uint8
   predict. Samples --n present shards that have present neighbours, reads a period^3 cube centred on a window corner (borders at
   offset + k * period) and reports, per axis, the mean |v(b) - v(b - 1)| across the border plane against the same across every other
   plane of the cube (both over voxel pairs with max >= --min, i.e. not air), plus the mean signed brightness step of the border planes.
   "jump" = border - interior (the ~30 / 9.4 / 6 numbers of the Paris 4 ink notes), "ratio" = border / interior. */
static double ns_opt(int argc, char **argv, const char *k, double d) {
    for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return atof(argv[i + 1]);
    return d;
}
int cmd_ink_seam(int argc, char **argv) {
    if (argc < 3 || argv[2][0] == '-') {
        fprintf(stderr, "usage: ufsm ink-seam ZARR --um U [--key ARRAY] [--period 512] [--offset 0] [--n 24] [--min 16] [--seed 1] [--threads 16]\n"); return 2; }
    const char *in = argv[2]; const double um = ns_opt(argc, argv, "--um", 0);
    const int P = (int)ns_opt(argc, argv, "--period", 512), off = (int)ns_opt(argc, argv, "--offset", 0), N = (int)ns_opt(argc, argv, "--n", 24),
              mn = (int)ns_opt(argc, argv, "--min", 16), nt = (int)ns_opt(argc, argv, "--threads", 16);
    unsigned seed = (unsigned)ns_opt(argc, argv, "--seed", 1);
    if (um <= 0 || P < 8 || P % 2) { fprintf(stderr, "ink-seam: --um > 0 and an even --period >= 8\n"); return 2; }
    char lv[256]; z3w_level_name(um, lv, sizeof lv);
    for (int i = 3; i + 1 < argc; i++) if (!strcmp(argv[i], "--key")) snprintf(lv, sizeof lv, "%s", argv[i + 1]);   /* any uint8 array, e.g. a CT level "0" */
    store *s = store_open(in); z3 *z = s ? z3_open(s, lv, nullptr) : nullptr;
    if (!z) { fprintf(stderr, "ink-seam: cannot open %s/%s: %s\n", in, lv, z3_error()); return 1; }
    const z3_meta *m = z3_meta_of(z);
    int64_t ns[3]; for (int d = 0; d < 3; d++) ns[d] = (m->shape[d] + m->shard[d] - 1) / m->shard[d];
    /* candidate corners: window borders b = off + k P strictly inside the array, cube [b - P/2, b + P/2) in all three axes */
    int64_t nb[3]; for (int d = 0; d < 3; d++) { nb[d] = 0; for (int64_t b = off; b + P / 2 <= m->shape[d]; b += P) if (b - P / 2 >= 0) nb[d]++; }
    if (!nb[0] || !nb[1] || !nb[2]) { fprintf(stderr, "ink-seam: no window corner inside the array\n"); return 1; }
    const size_t nv = (size_t)P * P * P; uint8_t *v = malloc(nv);
    if (!v) { fprintf(stderr, "ink-seam: out of memory\n"); return 1; }
    double bs[3] = {0}, bn[3] = {0}, is[3] = {0}, in_[3] = {0}, sg[3] = {0}; int got = 0, tries = 0;
    srand(seed);
    while (got < N && tries < N * 400) {
        tries++;
        int64_t o[3], b[3];
        for (int d = 0; d < 3; d++) { int64_t k = rand() % nb[d], c = 0; for (b[d] = off; ; b[d] += P) if (b[d] - P / 2 >= 0 && b[d] + P / 2 <= m->shape[d] && c++ == k) break; o[d] = b[d] - P / 2; }
        int ok = 1;   /* every shard the cube touches present (not air) */
        for (int64_t za = o[0] / m->shard[0]; za <= (o[0] + P - 1) / m->shard[0] && ok; za++)
            for (int64_t ya = o[1] / m->shard[1]; ya <= (o[1] + P - 1) / m->shard[1] && ok; ya++)
                for (int64_t xa = o[2] / m->shard[2]; xa <= (o[2] + P - 1) / m->shard[2] && ok; xa++)
                    if (za >= ns[0] || ya >= ns[1] || xa >= ns[2] || z3_shard_present(z, za, ya, xa) != 1) ok = 0;
        if (!ok) continue;
        const int64_t n[3] = {P, P, P};
        if (z3_read(z, o, n, v, nt)) { fprintf(stderr, "ink-seam: read: %s\n", z3_error()); free(v); return 1; }
        double cb[3] = {0}, cbn[3] = {0}, ci[3] = {0}, cin[3] = {0}, cs[3] = {0}; size_t nz = 0;
        for (size_t i = 0; i < nv; i++) nz += v[i] >= mn;
        if (nz < nv / 20) continue;   /* mostly air / no response: skip */
        const size_t st[3] = {(size_t)P * P, (size_t)P, 1};
        for (int d = 0; d < 3; d++)
            for (int k = 1; k < P; k++) {
                double sa = 0, sd = 0; size_t c = 0;
                for (int u = 0; u < P; u++) for (int w = 0; w < P; w++) {
                    size_t i1;
                    if (d == 0) i1 = k * st[0] + (size_t)u * st[1] + w; else if (d == 1) i1 = (size_t)u * st[0] + k * st[1] + w; else i1 = (size_t)u * st[0] + (size_t)w * st[1] + k;
                    const int a = v[i1 - st[d]], bb = v[i1];
                    if ((a > bb ? a : bb) < mn) continue;
                    sa += abs(bb - a); sd += bb - a; c++;
                }
                if (k == P / 2) { cb[d] += sa; cbn[d] += c; cs[d] += c ? fabs(sd / c) : 0; } else { ci[d] += sa; cin[d] += c; }
            }
        for (int d = 0; d < 3; d++) { bs[d] += cb[d]; bn[d] += cbn[d]; is[d] += ci[d]; in_[d] += cin[d]; sg[d] += cs[d]; }
        got++;
        fprintf(stderr, "ink-seam: corner %lld,%lld,%lld  jump z %.1f y %.1f x %.1f\n", (long long)b[0], (long long)b[1], (long long)b[2],
                cbn[0] ? cb[0] / cbn[0] - ci[0] / cin[0] : 0, cbn[1] ? cb[1] / cbn[1] - ci[1] / cin[1] : 0, cbn[2] ? cb[2] / cbn[2] - ci[2] / cin[2] : 0);
    }
    free(v);
    if (!got) { fprintf(stderr, "ink-seam: no non-air corner found\n"); return 1; }
    printf("ink-seam %s: %d corners, period %d offset %d (|dv| over pairs with max >= %d)\n", in, got, P, off, mn);
    double J = 0;
    for (int d = 0; d < 3; d++) {
        const double B = bn[d] ? bs[d] / bn[d] : 0, I = in_[d] ? is[d] / in_[d] : 0;
        printf("  %c: border %.2f  interior %.2f  jump %.2f  ratio %.2f  brightness step %.2f\n", "zyx"[d], B, I, B - I, I > 0 ? B / I : 0, sg[d] / got);
        J += (B - I) / 3;
    }
    printf("  mean jump %.2f\n", J);
    return 0;
}
