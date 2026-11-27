/* ufsm phase-cmp A.json B.json: the on-the-fly phase of the same sources under two specs (w24 2026-11-27: rmax 80 vs 160 at the 2.4 um
   rung). Per source (matched by index; A and B must list the same sources), boxes of --P^3 at random labelled chunks of B's phase (its
   chunk map: the meshes' / codes' extent), both phases read there: voxels known in B, known in A, and known in B but unknown in A -- the
   share a smaller rmax leaves unknown (sheet gaps wider than it). Printed per source as a DEFAULTED line when that share is > 0, so a
   run log carries the count. GPU pair rule on --gpu (default 0; -1 the CPU). Deterministic for a --seed. */
#include "band_onfly.h"
#include "band.h"
#include "meshcodes.h"
#include "reg.h"
#include "sources.h"
#include "store.h"
#include "optcheck.h"
#include "zarr3.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* options from argv[4] (argv[2], argv[3]: the two sources files) */
static const char *opt(int argc, char **argv, const char *k, const char *d) { for (int i = 4; i + 1 < argc; i++) if (!strcmp(argv[i], opt_n(k))) return argv[i + 1]; return d; }
static uint64_t rng_next(uint64_t *s) { uint64_t x = *s += 0x9e3779b97f4a7c15ull; x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull; x = (x ^ (x >> 27)) * 0x94d049bb133111ebull; return x ^ (x >> 31); }

int cmd_phase_cmp(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm phase-cmp A.json B.json [--boxes 24] [--P 256] [--seed 1] [--gpu 0] [--slots 2] [--mb 1024] [--map REG.json]\n"
                                    "  --map: A on another scan of the scroll (the map's fine grid; B on its coarse grid): B sampled at A's voxel centres (rmap_f2c, nearest)\n"
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
    setvbuf(stdout, nullptr, _IOLBF, 0);   /* (a line per source as it finishes) */
    const size_t N = (size_t)P * P * P; uint8_t *a = malloc(N), *b = malloc(N);
    if (!a || !b) return 1;
    /* --map (w11 2026-11-27: 1.1 um labels from meshes mapped 2.4 -> 1.129 vs the 2.4 um labels): A's boxes on the fine grid, B read over
       the coarse box the mapped corners span (+ 2) and sampled at each A voxel's mapped centre (nearest coarse voxel) */
    const char *mp = opt(argc, argv, "--map", nullptr); const rmap *M = mp ? reg_map_load(mp) : nullptr;
    if (mp && !M) { fprintf(stderr, "phase-cmp: cannot load the map %s\n", mp); return 1; }
    uint8_t *bc = nullptr; size_t bcap = 0;
    int rc = 0;
    for (int i = 0; i < A->n; i++) {
        source *sa = &A->src[i], *sb = &B->src[i];
        if (!sa->phase || !sb->phase) { printf("phase-cmp: %s: no phase in %s\n", sb->name, sa->phase ? "B" : "A"); continue; }
        z3 *ref = M ? sa->phase : sb->phase;   /* (the boxes on B's labelled chunks; --map: on A's, the grid they are read on) */
        int64_t g[3]; uint8_t *bits = nullptr; const int64_t nc = z3_chunk_map(ref, g, &bits);
        const z3_meta *m = z3_meta_of(ref);
        if (nc <= 0 || !bits) { printf("phase-cmp: %s: no labelled chunk\n", sb->name); free(bits); continue; }
        int64_t *idx = malloc((size_t)nc * sizeof *idx), k = 0;
        for (int64_t q = 0; q < g[0] * g[1] * g[2]; q++) if (bits[q]) idx[k++] = q;
        uint64_t kb = 0, ka = 0, lost = 0, vox = 0, both = 0, eq[4] = {0}; int boxes = 0;   /* eq: |A - B| (circular, codes of 252) 0, <= 2, <= 6, <= 13 */
        for (int t = 0; t < nb && k; t++) {
            const int64_t q = idx[rng_next(&seed) % (uint64_t)k], c[3] = {q / (g[1] * g[2]), (q / g[2]) % g[1], q % g[2]};
            int64_t o[3]; const int64_t n[3] = {P, P, P};
            for (int d = 0; d < 3; d++) { o[d] = c[d] * m->chunk[d] + m->chunk[d] / 2 - P / 2; if (o[d] < 0) o[d] = 0; if (o[d] + P > m->shape[d]) o[d] = m->shape[d] - P > 0 ? m->shape[d] - P : 0; }
            if (z3_read(sa->phase, o, n, a, 8)) { fprintf(stderr, "phase-cmp: %s: read failed\n", sa->name); rc = 1; break; }
            if (!M) { if (z3_read(sb->phase, o, n, b, 8)) { fprintf(stderr, "phase-cmp: %s: read failed\n", sb->name); rc = 1; break; } }
            else {   /* the coarse box: the mapped corners' bounds (x y z points), + 2 */
                double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
                for (int k = 0; k < 8; k++) { const double f[3] = {(double)(o[2] + ((k & 1) ? P : 0)), (double)(o[1] + ((k & 2) ? P : 0)), (double)(o[0] + ((k & 4) ? P : 0))}; double c[3];
                    rmap_f2c(M, f, c); if (!k && getenv("UFSM_PCMP_DEBUG")) fprintf(stderr, "f2c(%g %g %g) = %g %g %g\n", f[0], f[1], f[2], c[0], c[1], c[2]);
                    for (int d = 0; d < 3; d++) { if (c[d] < lo[d]) lo[d] = c[d]; if (c[d] > hi[d]) hi[d] = c[d]; } }
                const z3_meta *mb = z3_meta_of(sb->phase); int64_t co[3], cn[3];
                for (int d = 0; d < 3; d++) { const int e = 2 - d; co[d] = (int64_t)floor(lo[e]) - 2; cn[d] = (int64_t)ceil(hi[e]) + 3 - co[d];
                    if (co[d] < 0) { cn[d] += co[d]; co[d] = 0; } if (co[d] + cn[d] > mb->shape[d]) cn[d] = mb->shape[d] - co[d]; }
                memset(b, PHASE_UNKNOWN, N);
                if (cn[0] > 0 && cn[1] > 0 && cn[2] > 0) {
                    const size_t cnn = (size_t)cn[0] * cn[1] * cn[2];
                    if (cnn > bcap) { free(bc); bcap = 0; if (!(bc = malloc(cnn))) { rc = 1; break; } bcap = cnn; }
                    if (z3_read(sb->phase, co, cn, bc, 8)) { fprintf(stderr, "phase-cmp: %s: read failed\n", sb->name); rc = 1; break; }
                    if (getenv("UFSM_PCMP_DEBUG")) { size_t kk = 0; for (size_t q = 0; q < cnn; q++) kk += bc[q] != PHASE_UNKNOWN; fprintf(stderr, "phase-cmp --map: A box %lld,%lld,%lld -> B box %lld,%lld,%lld + %lld,%lld,%lld: %zu known\n", (long long)o[0], (long long)o[1], (long long)o[2], (long long)co[0], (long long)co[1], (long long)co[2], (long long)cn[0], (long long)cn[1], (long long)cn[2], kk); }
                    for (int64_t z = 0; z < P; z++) for (int64_t y = 0; y < P; y++) for (int64_t x = 0; x < P; x++) {
                        const double f[3] = {(double)(o[2] + x), (double)(o[1] + y), (double)(o[0] + z)}; double c[3]; rmap_f2c(M, f, c);
                        const int64_t q[3] = {(int64_t)llround(c[2]) - co[0], (int64_t)llround(c[1]) - co[1], (int64_t)llround(c[0]) - co[2]};
                        if (q[0] < 0 || q[1] < 0 || q[2] < 0 || q[0] >= cn[0] || q[1] >= cn[1] || q[2] >= cn[2]) continue;
                        b[((size_t)z * P + (size_t)y) * P + (size_t)x] = bc[((size_t)q[0] * cn[1] + (size_t)q[1]) * cn[2] + (size_t)q[2]];
                    }
                }
            }
            if (getenv("UFSM_PCMP_DUMP") && !boxes) {   /* (the first box's A and B phases as raw uint8, P^3 z-major, and its origin) */
                char fp[1200]; snprintf(fp, sizeof fp, "%s-%s-A.raw", getenv("UFSM_PCMP_DUMP"), sa->name); FILE *fa = fopen(fp, "wb"); if (fa) { fwrite(a, 1, N, fa); fclose(fa); }
                snprintf(fp, sizeof fp, "%s-%s-B.raw", getenv("UFSM_PCMP_DUMP"), sa->name); FILE *fb = fopen(fp, "wb"); if (fb) { fwrite(b, 1, N, fb); fclose(fb); }
                fprintf(stderr, "phase-cmp: dumped %s box %lld,%lld,%lld size %d\n", sa->name, (long long)o[0], (long long)o[1], (long long)o[2], P); }
            for (size_t v = 0; v < N; v++) {
                const int kna = a[v] != PHASE_UNKNOWN, knb = b[v] != PHASE_UNKNOWN; kb += knb; ka += kna; lost += knb && !kna;
                if (kna && knb) { int d = abs((int)a[v] - (int)b[v]); if (d > PHASE_PERIOD / 2) d = PHASE_PERIOD - d; both++; eq[0] += d == 0; eq[1] += d <= 2; eq[2] += d <= 6; eq[3] += d <= 13; }
            }
            vox += N; boxes++;
        }
        free(idx); free(bits);
        const double sh = kb ? (double)lost / kb : 0;
        printf("phase-cmp: %s: %d boxes of %d^3: known B %.4f, A %.4f of the voxels; A leaves %.4f of B's known voxels unknown\n", sb->name, boxes, P,
               vox ? (double)kb / vox : 0, vox ? (double)ka / vox : 0, sh);
        if (both) printf("phase-cmp: %s: agreement on the %llu voxels known in both: exact %.4f, within 2 codes %.4f, 6 %.4f, 13 (5 %% of a turn) %.4f\n", sb->name,
                         (unsigned long long)both, (double)eq[0] / both, (double)eq[1] / both, (double)eq[2] / both, (double)eq[3] / both);
        if (lost) printf("DEFAULTED: phase-cmp: %s: %.2f %% of the phase voxels known in B (%s) stay unknown in A (%s): sheet gaps wider than A's rmax\n",
                         sb->name, 100 * sh, argv[3], argv[2]);
    }
    free(a); free(b); free(bc); sources_free(A); sources_free(B);
    return rc;
}
