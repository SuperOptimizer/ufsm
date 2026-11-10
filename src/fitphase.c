/* ufsm fit-phase-box <fit-set> z y x nz ny nx <out.raw> [--level L] [--step S] [--theta 0|1]
   The winding phase of a joined fit set (fit-join output, or one model dir) over a box of the CT's level-L voxels, as a band-phase
   raster: uint8 floor(252 frac(W)) (0..251), 255 outside the set's global box or where W is not finite. The out grid has one value per
   S^3 level voxels (dims ceil(n / S)); value i sits at level coordinate z + (i + 0.5) S - 0.5 (level voxel index = centre, level
   coordinate c = native (c + 0.5) 2^L - 0.5). W jumps by one turn at the theta cut, so frac(W) is continuous along a sheet: the
   label-free fibre judge (tools/fibjudge.py; DESIGN.md 2026-11-10) unwraps it along fibre polylines exactly like a surf-labels phase.
   Prints the set's box, um and direction. CPU only (fit_set_grid_cpu; threads UFSM_FIT_THREADS). */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fit.h"

static const char *fp_opt(int argc, char **argv, const char *k, const char *d) { for (int i = 10; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return argv[i + 1]; return d; }

int cmd_fit_phase_box(int argc, char **argv) {
    if (argc < 10) { fprintf(stderr, "usage: ufsm fit-phase-box <fit-set> z y x nz ny nx <out.raw> [--level 0] [--step 2] [--theta 0]\n"); return 2; }
    const int level = atoi(fp_opt(argc, argv, "--level", "0")), S = atoi(fp_opt(argc, argv, "--step", "2")), th = atoi(fp_opt(argc, argv, "--theta", "0"));
    if (S < 1 || level < 0 || level > 8) { fprintf(stderr, "fit-phase-box: bad --step / --level\n"); return 2; }
    fit_set *s = fit_set_load(argv[2]); if (!s) { fprintf(stderr, "fit-phase-box: cannot load the fit set %s\n", argv[2]); return 1; }
    const fit_model *g = s->global; const double sc = ldexp(1.0, level);
    int64_t b[6]; for (int i = 0; i < 6; i++) b[i] = atoll(argv[3 + i]);
    const int64_t n[3] = {(b[3] + S - 1) / S, (b[4] + S - 1) / S, (b[5] + S - 1) / S};
    if (n[0] < 1 || n[1] < 1 || n[2] < 1) { fprintf(stderr, "fit-phase-box: empty box\n"); return 2; }
    /* grid point i at box-local native o + (i + 0.5) step: level centre z + (i + 0.5) S - 0.5 -> native (z + (i + 0.5) S) sc - 0.5 */
    const double o[3] = {b[0] * sc - 0.5 - g->origin[0], b[1] * sc - 0.5 - g->origin[1], b[2] * sc - 0.5 - g->origin[2]}, step = S * sc;
    const double ext[3] = {g->D * g->h, g->H * g->h, g->W * g->h};
    printf("fit-phase-box: set %s: %d slabs, origin %lld %lld %lld, extent %.0f %.0f %.0f native voxels of %.3f um, dir %+d; grid %lld x %lld x %lld, step %g native\n", argv[2], s->n,
           g->origin[0], g->origin[1], g->origin[2], ext[0], ext[1], ext[2], g->um, g->dir, (long long)n[0], (long long)n[1], (long long)n[2], step);
    const size_t nv = (size_t)n[0] * n[1] * n[2]; float *w = malloc(nv * sizeof *w); uint8_t *out = malloc(nv);
    if (!w || !out || fit_set_grid_cpu(s, o, n, step, th, nullptr, w)) { fprintf(stderr, "fit-phase-box: evaluation failed\n"); return 1; }
    size_t in = 0;
    for (int64_t i = 0; i < n[0]; i++) for (int64_t j = 0; j < n[1]; j++) for (int64_t k = 0; k < n[2]; k++) {
        const size_t q = ((size_t)i * n[1] + j) * n[2] + k; const double p[3] = {o[0] + (i + 0.5) * step, o[1] + (j + 0.5) * step, o[2] + (k + 0.5) * step};
        const int ok = p[0] >= 0 && p[1] >= 0 && p[2] >= 0 && p[0] <= ext[0] && p[1] <= ext[1] && p[2] <= ext[2] && isfinite(w[q]);
        if (!ok) { out[q] = 255; continue; }
        int v = (int)floor((w[q] - floor(w[q])) * 252); out[q] = (uint8_t)(v < 0 ? 0 : v > 251 ? 251 : v); in++;
    }
    FILE *f = fopen(argv[9], "wb"); if (!f || fwrite(out, 1, nv, f) != nv || fclose(f)) { fprintf(stderr, "fit-phase-box: cannot write %s\n", argv[9]); return 1; }
    printf("fit-phase-box: %zu of %zu points inside the set -> %s\n", in, nv, argv[9]);
    free(w); free(out); fit_set_free(s); return 0;
}
