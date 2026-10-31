/* ufsm phase-packing <phase-root> <phase-key> [--um 9.6] [--packed-um 100] [--box z,y,x,nz,ny,nx] [--map OUT.raw] [--threads 16]
   How packed the labelled sheets are, from a stored label phase raster (PHASE_* codes: 0..251 = phase x 252, 255 unknown): per voxel
   whose +z / +y / +x neighbours are known, the local sheet frequency g = |wrapped phase gradient| (turns per voxel) and the local spacing
   um / g. --box (raster voxels): prints the share of those voxels with spacing < --packed-um and the spacing percentiles. --map: per
   stored chunk of the raster (the chunk grid, row-major uint8, 0 = no label, else 1 + round(254 x packed share)), the input of the
   sampler's "focus" weighting (src/sources.h); OUT.json beside it records the grid. */
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "store.h"
#include "sysinfo.h"
#include "zarr3.h"
#include "z3w.h"

static const char *pk_opt(int argc, char **argv, const char *k, const char *d) { for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return argv[i + 1]; return d; }
static inline double pk_wrap(int a, int b) { double d = (b - a) / 252.0; return d - round(d); }
/* the voxels of a block (n[0] x n[1] x n[2], with one extra plane per axis for the +1 neighbours at n+1): counts known / packed, spacing histogram */
static void pk_block(const uint8_t *p, const int64_t n1[3], const int64_t n[3], double gthr, long long *nk, long long *np, long long hist[64], double um) {
    for (int64_t z = 0; z < n[0]; z++) for (int64_t y = 0; y < n[1]; y++) for (int64_t x = 0; x < n[2]; x++) {
        const size_t i = ((size_t)z * n1[1] + y) * n1[2] + x; const uint8_t a = p[i];
        if (a > 251) continue;
        const uint8_t bz = p[i + (size_t)n1[1] * n1[2]], by = p[i + n1[2]], bx = p[i + 1];
        if (bz > 251 || by > 251 || bx > 251) continue;
        const double dz = pk_wrap(a, bz), dy = pk_wrap(a, by), dx = pk_wrap(a, bx), g = sqrt(dz * dz + dy * dy + dx * dx);
        (*nk)++; if (g > gthr) (*np)++;
        if (hist) { const double sp = g > 1e-6 ? um / g : 1e6; int b = (int)(sp / 10); if (b > 63) b = 63; hist[b]++; }
    }
}
typedef struct { z3 *z; const int64_t *g; const uint8_t *bits; int64_t C; double gthr; uint8_t *map; _Atomic int64_t next; int64_t total; const int64_t *shape; } pk_job;
static void *pk_worker(void *a) {
    pk_job *J = a; const int64_t C = J->C; uint8_t *buf = malloc((size_t)(C + 1) * (C + 1) * (C + 1));
    for (int64_t k; (k = atomic_fetch_add(&J->next, 1)) < J->total;) {
        if (!J->bits[k]) continue;
        const int64_t cz = k / (J->g[1] * J->g[2]), cy = (k / J->g[2]) % J->g[1], cx = k % J->g[2];
        int64_t o[3] = {cz * C, cy * C, cx * C}, n[3], n1[3];
        for (int d = 0; d < 3; d++) { n[d] = C; if (o[d] + n[d] > J->shape[d]) n[d] = J->shape[d] - o[d]; n1[d] = n[d] + (o[d] + n[d] < J->shape[d] ? 1 : 0); if (n1[d] == n[d]) n[d]--; }
        if (n[0] < 1 || n[1] < 1 || n[2] < 1 || z3_read(J->z, o, n1, buf, 1)) continue;
        long long nk = 0, np = 0; pk_block(buf, n1, n, J->gthr, &nk, &np, nullptr, 0);
        J->map[k] = nk < 1000 ? 0 : (uint8_t)(1 + lround(254.0 * np / nk));
    }
    free(buf); return nullptr;
}
int cmd_phase_packing(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm phase-packing <phase-root> <phase-key> [--um 9.6] [--packed-um 100] [--box z,y,x,nz,ny,nx] [--map OUT.raw] [--threads 16]\n"); return 2; }
    z3 *z = z3_open(store_open(argv[2]), argv[3], nullptr); if (!z) { fprintf(stderr, "phase-packing: %s\n", z3_error()); return 1; }
    const z3_meta *m = z3_meta_of(z); const double um = atof(pk_opt(argc, argv, "--um", "9.6")), pum = atof(pk_opt(argc, argv, "--packed-um", "100"));
    const double gthr = um / pum; const int nt = ufsm_threads(atoi(pk_opt(argc, argv, "--threads", "0")), 16);
    const char *bs = pk_opt(argc, argv, "--box", nullptr), *mp = pk_opt(argc, argv, "--map", nullptr);
    if (bs) {
        long long b[6]; if (sscanf(bs, "%lld,%lld,%lld,%lld,%lld,%lld", b, b + 1, b + 2, b + 3, b + 4, b + 5) != 6) { fprintf(stderr, "phase-packing: --box z,y,x,nz,ny,nx\n"); return 2; }
        long long nk = 0, np = 0, hist[64] = {0};
        for (long long z0 = 0; z0 < b[3]; z0 += 64) {   /* z slabs of 64 planes (+1) */
            const int64_t o[3] = {b[0] + z0, b[1], b[2]}, n[3] = {b[3] - z0 < 64 ? b[3] - z0 : 64, b[4], b[5]}, n1[3] = {n[0] + 1, n[1] + 1, n[2] + 1};
            uint8_t *buf = malloc((size_t)n1[0] * n1[1] * n1[2]);
            if (!buf || z3_read(z, o, n1, buf, nt)) { fprintf(stderr, "phase-packing: read failed\n"); return 1; }
            pk_block(buf, n1, n, gthr, &nk, &np, hist, um); free(buf);
        }
        long long c = 0; double q[3] = {0.1, 0.5, 0.9}, qv[3] = {0, 0, 0}; int qi = 0;
        for (int k = 0; k < 64 && qi < 3; k++) { c += hist[k]; while (qi < 3 && c >= q[qi] * nk) qv[qi++] = (k + 0.5) * 10; }
        printf("phase-packing %s: %lld labelled voxels with neighbours, spacing < %g um: %.4f; spacing p10 %.0f p50 %.0f p90 %.0f um\n", bs, nk, pum, nk ? (double)np / nk : 0, qv[0], qv[1], qv[2]);
    }
    if (mp) {
        int64_t g[3]; uint8_t *bits = nullptr; const int64_t ns = z3_chunk_map(z, g, &bits);
        if (ns <= 0) { fprintf(stderr, "phase-packing: no chunk map\n"); return 1; }
        pk_job J = {.z = z, .g = g, .bits = bits, .C = m->chunk[0], .gthr = gthr, .map = calloc((size_t)(g[0] * g[1] * g[2]), 1), .total = g[0] * g[1] * g[2], .shape = m->shape};
        pthread_t th[64]; const int n = nt < 64 ? nt : 64;
        for (int t = 0; t < n; t++) pthread_create(&th[t], nullptr, pk_worker, &J);
        for (int t = 0; t < n; t++) pthread_join(th[t], nullptr);
        FILE *f = fopen(mp, "wb"); const size_t tot = (size_t)(g[0] * g[1] * g[2]);
        if (!f || fwrite(J.map, 1, tot, f) != tot || fclose(f)) { fprintf(stderr, "phase-packing: cannot write %s\n", mp); return 1; }
        char jp[1200]; snprintf(jp, sizeof jp, "%s.json", mp); f = fopen(jp, "w");
        if (f) { fprintf(f, "{\"grid\":[%lld,%lld,%lld],\"cell\":%lld,\"packed_um\":%g,\"um\":%g,\"content\":\"0 no label, else 1 + round(254 x share of labelled voxels with spacing < packed_um)\"}\n",
                         (long long)g[0], (long long)g[1], (long long)g[2], (long long)m->chunk[0], pum, um); fclose(f); }
        long long nl = 0, hi = 0; for (size_t k = 0; k < tot; k++) if (J.map[k]) { nl++; hi += J.map[k] > 128; }
        printf("phase-packing map: %lld x %lld x %lld cells of %lld, %lld labelled, %lld more than half packed -> %s\n", (long long)g[0], (long long)g[1], (long long)g[2], (long long)m->chunk[0], nl, hi, mp);
        free(J.map); free(bits);
    }
    z3_close(z);
    return 0;
}

/* ufsm phase-conflict <a-root> <a-key> <b-root> <b-key> <out-array-dir> [--tol 0.25] [--threads 16]
   Where two label phase rasters of one scan (e.g. the Paris 4 June / July published-wrap sets, same grid) disagree: a uint8 zarr on their
   grid (512^3 shards), 1 where both are known and their wrapped difference exceeds --tol turns, else 0. Detection only (no repair): the
   sampler's "phase": {..., "ignore": {"root", "key"}} makes those voxels unknown, so neither set's phase there is trained on. */
int cmd_phase_conflict(int argc, char **argv) {
    if (argc < 7) { fprintf(stderr, "usage: ufsm phase-conflict <a-root> <a-key> <b-root> <b-key> <out-array-dir> [--tol 0.25] [--threads 16]\n"); return 2; }
    z3 *a = z3_open(store_open(argv[2]), argv[3], nullptr), *b = z3_open(store_open(argv[4]), argv[5], nullptr);
    if (!a || !b) { fprintf(stderr, "phase-conflict: %s\n", z3_error()); return 1; }
    const z3_meta *ma = z3_meta_of(a), *mb = z3_meta_of(b);
    if (memcmp(ma->shape, mb->shape, sizeof ma->shape)) { fprintf(stderr, "phase-conflict: the rasters differ in shape\n"); return 1; }
    const double tol = atof(pk_opt(argc, argv, "--tol", "0.25")); const int nt = ufsm_threads(atoi(pk_opt(argc, argv, "--threads", "0")), 16), S = 512;
    if (!(tol > 0 && tol < 0.5)) { fprintf(stderr, "phase-conflict: --tol in (0, 0.5)\n"); return 2; }
    char at[256]; snprintf(at, sizeof at, "{\"ufsm\":{\"content\":\"1 where two label phase rasters are both known and differ by more than %g turn\"}}", tol);
    z3w *w = z3w_create(argv[6], ma->shape, S, 0.f, 0, at); if (!w) { fprintf(stderr, "phase-conflict: %s\n", z3w_error()); return 1; }
    int64_t ga[3], gb[3]; uint8_t *ba = nullptr, *bb = nullptr;
    if (z3_chunk_map(a, ga, &ba) <= 0 || z3_chunk_map(b, gb, &bb) <= 0 || ma->chunk[0] != mb->chunk[0] || S % ma->chunk[0]) { fprintf(stderr, "phase-conflict: no chunk maps\n"); return 1; }
    const int64_t C = ma->chunk[0], k = S / C, ns[3] = {(ma->shape[0] + S - 1) / S, (ma->shape[1] + S - 1) / S, (ma->shape[2] + S - 1) / S};
    uint8_t *pa = malloc((size_t)S * S * S), *pb = malloc((size_t)S * S * S), *out = malloc((size_t)S * S * S);
    long long nboth = 0, nconf = 0; int nsh = 0; const int lim = (int)lround(tol * 252);
    for (int64_t sz = 0; sz < ns[0]; sz++) for (int64_t sy = 0; sy < ns[1]; sy++) for (int64_t sx = 0; sx < ns[2]; sx++) {
        int both = 0;   /* a shard needs a stored chunk in both rasters */
        for (int64_t cz = sz * k; cz < (sz + 1) * k && cz < ga[0] && !both; cz++) for (int64_t cy = sy * k; cy < (sy + 1) * k && cy < ga[1] && !both; cy++)
            for (int64_t cx = sx * k; cx < (sx + 1) * k && cx < ga[2]; cx++) { const int64_t q = (cz * ga[1] + cy) * ga[2] + cx; if (ba[q] && bb[q]) { both = 1; break; } }
        if (!both) continue;
        const int64_t o[3] = {sz * S, sy * S, sx * S}; int64_t n[3];
        for (int d = 0; d < 3; d++) n[d] = o[d] + S < ma->shape[d] ? S : ma->shape[d] - o[d];
        if (z3_read(a, o, n, pa, nt) || z3_read(b, o, n, pb, nt)) { fprintf(stderr, "phase-conflict: read failed: %s\n", z3_error()); return 1; }
        memset(out, 0, (size_t)S * S * S); long long nc = 0, nb = 0;
        for (int64_t z = 0; z < n[0]; z++) for (int64_t y = 0; y < n[1]; y++) for (int64_t x = 0; x < n[2]; x++) {
            const size_t i = ((size_t)z * n[1] + y) * n[2] + x; const int p = pa[i], q = pb[i];
            if (p > 251 || q > 251) continue;
            nb++; int dd = abs(p - q); if (dd > 126) dd = 252 - dd;
            if (dd > lim) { out[((size_t)z * S + y) * S + x] = 1; nc++; }
        }
        nboth += nb; nconf += nc; nsh++;
        if (nc && z3w_write_shard(w, sz, sy, sx, out, nt)) { fprintf(stderr, "phase-conflict: %s\n", z3w_error()); return 1; }
    }
    z3w_close(w); free(pa); free(pb); free(out); free(ba); free(bb); z3_close(a); z3_close(b);
    printf("phase-conflict: %d shards with both sets, %lld voxels known in both, %lld (%.4f) differ by more than %g turn -> %s\n", nsh, nboth, nconf, nboth ? (double)nconf / nboth : 0, tol, argv[6]);
    return 0;
}
