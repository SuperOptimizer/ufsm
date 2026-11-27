/* ufsm phase-cmp A.json B.json: the on-the-fly phase of the same sources under two specs (w24 2026-11-27: rmax 80 vs 160 at the 2.4 um
   rung). Per source (matched by index; A and B must list the same sources), boxes of --P^3 at random labelled chunks of B's phase (its
   chunk map: the meshes' / codes' extent), both phases read there: voxels known in B, known in A, and known in B but unknown in A -- the
   share a smaller rmax leaves unknown (sheet gaps wider than it). Printed per source as a DEFAULTED line when that share is > 0, so a
   run log carries the count. GPU pair rule on --gpu (default 0; -1 the CPU). Deterministic for a --seed. */
#include "band_onfly.h"
#include "band.h"
#include "meshcodes.h"
#include "sources.h"
#include "store.h"
#include "optcheck.h"
#include "zarr3.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* options from argv[4] (argv[2], argv[3]: the two sources files) */
static const char *opt(int argc, char **argv, const char *k, const char *d) { for (int i = 4; i + 1 < argc; i++) if (!strcmp(argv[i], opt_n(k))) return argv[i + 1]; return d; }
static uint64_t rng_next(uint64_t *s) { uint64_t x = *s += 0x9e3779b97f4a7c15ull; x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull; x = (x ^ (x >> 27)) * 0x94d049bb133111ebull; return x ^ (x >> 31); }

int cmd_phase_cmp(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm phase-cmp A.json B.json [--boxes 24] [--P 256] [--seed 1] [--gpu 0] [--slots 2] [--mb 1024]\n"
                                    "  per source: the share of B's known phase voxels that A leaves unknown, in --boxes random --P^3 boxes at B's labelled chunks\n"); return 2; }
    const int nb = opt_atoi("--boxes", opt(argc, argv, "--boxes", "24")), P = opt_atoi("--P", opt(argc, argv, "--P", "256")), gpu = opt_atoi("--gpu", opt(argc, argv, "--gpu", "0"));
    uint64_t seed = (uint64_t)opt_atoll("--seed", opt(argc, argv, "--seed", "1"));
    if (nb < 1 || P < 16) { fprintf(stderr, "phase-cmp: --boxes >= 1, --P >= 16\n"); return 2; }
    store_global_init();
    sources *A = sources_load(argv[2]), *B = sources_load(argv[3]);
    if (!A || !B) { fprintf(stderr, "phase-cmp: cannot load the sources\n"); return 1; }
    if (A->n != B->n) { fprintf(stderr, "phase-cmp: %d vs %d sources\n", A->n, B->n); return 1; }
    if (phase_onfly_config(gpu, opt_atoi("--slots", opt(argc, argv, "--slots", "2")), opt_atoi("--mb", opt(argc, argv, "--mb", "1024")), 0)) { fprintf(stderr, "phase-cmp: --gpu %d: no GPU pair rule\n", gpu); return 1; }
    mesh_codes_gpu_config(gpu);
    const size_t N = (size_t)P * P * P; uint8_t *a = malloc(N), *b = malloc(N);
    if (!a || !b) return 1;
    int rc = 0;
    for (int i = 0; i < A->n; i++) {
        source *sa = &A->src[i], *sb = &B->src[i];
        if (!sa->phase || !sb->phase) { printf("phase-cmp: %s: no phase in %s\n", sb->name, sa->phase ? "B" : "A"); continue; }
        int64_t g[3]; uint8_t *bits = nullptr; const int64_t nc = z3_chunk_map(sb->phase, g, &bits);
        const z3_meta *m = z3_meta_of(sb->phase);
        if (nc <= 0 || !bits) { printf("phase-cmp: %s: no labelled chunk\n", sb->name); free(bits); continue; }
        int64_t *idx = malloc((size_t)nc * sizeof *idx), k = 0;
        for (int64_t q = 0; q < g[0] * g[1] * g[2]; q++) if (bits[q]) idx[k++] = q;
        uint64_t kb = 0, ka = 0, lost = 0, vox = 0; int boxes = 0;
        for (int t = 0; t < nb && k; t++) {
            const int64_t q = idx[rng_next(&seed) % (uint64_t)k], c[3] = {q / (g[1] * g[2]), (q / g[2]) % g[1], q % g[2]};
            int64_t o[3]; const int64_t n[3] = {P, P, P};
            for (int d = 0; d < 3; d++) { o[d] = c[d] * m->chunk[d] + m->chunk[d] / 2 - P / 2; if (o[d] < 0) o[d] = 0; if (o[d] + P > m->shape[d]) o[d] = m->shape[d] - P > 0 ? m->shape[d] - P : 0; }
            if (z3_read(sa->phase, o, n, a, 8) || z3_read(sb->phase, o, n, b, 8)) { fprintf(stderr, "phase-cmp: %s: read failed\n", sb->name); rc = 1; break; }
            for (size_t v = 0; v < N; v++) { const int kna = a[v] != PHASE_UNKNOWN, knb = b[v] != PHASE_UNKNOWN; kb += knb; ka += kna; lost += knb && !kna; }
            vox += N; boxes++;
        }
        free(idx); free(bits);
        const double sh = kb ? (double)lost / kb : 0;
        printf("phase-cmp: %s: %d boxes of %d^3: known B %.4f, A %.4f of the voxels; A leaves %.4f of B's known voxels unknown\n", sb->name, boxes, P,
               vox ? (double)kb / vox : 0, vox ? (double)ka / vox : 0, sh);
        if (lost) printf("DEFAULTED: phase-cmp: %s: %.2f %% of the phase voxels known in B (%s) stay unknown in A (%s): sheet gaps wider than A's rmax\n",
                         sb->name, 100 * sh, argv[3], argv[2]);
    }
    free(a); free(b); sources_free(A); sources_free(B);
    return rc;
}
