/* ufsm ink-gate OUT --ink ROOT KEY --ink-um U --phase ROOT KEY --phase-um U [--ignore ROOT KEY] [--lo 0] [--hi 125] [--ink-max 255]
                  [--shard 512] [--threads N]
   Pseudo ink targets: an ink prediction (uint8 probability, 0..ink-max) restricted to a band of a stored label phase raster
   (PHASE_* codes: 0..251 = phase x 252, 252..255 unknown), e.g. the recto half k..k+0.5 of every turn = codes lo 0 .. hi 125.
   Output on the phase raster's grid (OUT/<phase-um>, OME group OUT/zarr.json, 512^3 shards, lossless), "prob254" encoding:
   p x 254 where the phase code lies in [lo, hi] (lo > hi wraps through 251 / 0) and the ignore mask (same grid as the phase) is 0,
   else 255 (unknown). The ink is read nearest-neighbour at the phase voxel centres (both on the scan's global grid, origin 0;
   voxel sizes --ink-um / --phase-um). Only shards holding stored phase chunks are visited. Prints the gated / ink voxel counts. */
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

static const char *ig_opt(int argc, char **argv, const char *k, const char *d) { for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return argv[i + 1]; return d; }
static int ig_opt2(int argc, char **argv, const char *k, const char **a, const char **b) {
    for (int i = 2; i + 2 < argc; i++) if (!strcmp(argv[i], k)) { *a = argv[i + 1]; *b = argv[i + 2]; return 1; }
    return 0;
}

typedef struct {
    const char *ir, *ik, *pr, *pk, *gr, *gk;   /* ink, phase, ignore (root, key) */
    double ium, pum; int lo, hi, imax, S;
    int64_t shape[3], ishape[3], ns[3]; uint8_t *present; z3w *zw; uint8_t lut[256];
    _Atomic long next; _Atomic int fail; _Atomic long long ngate, nink, nknown;
} ig_job;

static inline int ig_in(const ig_job *J, int c) { return c <= 251 && (J->lo <= J->hi ? c >= J->lo && c <= J->hi : c >= J->lo || c <= J->hi); }

static void *ig_worker(void *arg) {
    ig_job *J = arg; const int S = J->S; const size_t n = (size_t)S * S * S;
    store *sp = store_open(J->pr), *si = store_open(J->ir), *sg = J->gr ? store_open(J->gr) : nullptr;
    z3 *zp = sp ? z3_open(sp, J->pk, nullptr) : nullptr, *zi = si ? z3_open(si, J->ik, nullptr) : nullptr, *zg = sg ? z3_open(sg, J->gk, nullptr) : nullptr;
    uint8_t *p = malloc(n), *g = zg ? malloc(n) : nullptr, *o = malloc(n), *ib = nullptr; size_t icap = 0;
    int64_t *map[3] = {malloc(S * sizeof(int64_t)), malloc(S * sizeof(int64_t)), malloc(S * sizeof(int64_t))};
    if (!zp || !zi || (J->gr && !zg) || !p || !o || (zg && !g)) { fprintf(stderr, "ink-gate: cannot open the inputs: %s\n", z3_error()); atomic_store(&J->fail, 1); goto done; }
    const double r = J->pum / J->ium;
    for (;;) {
        const long k = atomic_fetch_add(&J->next, 1); if (k >= J->ns[0] * J->ns[1] * J->ns[2] || atomic_load(&J->fail)) break;
        if (J->present && !J->present[k]) continue;
        const int64_t so[3] = {k / (J->ns[1] * J->ns[2]) * S, k / J->ns[2] % J->ns[1] * S, k % J->ns[2] * S}; int64_t sn[3], io[3], in[3];
        for (int d = 0; d < 3; d++) {
            sn[d] = so[d] + S <= J->shape[d] ? S : J->shape[d] - so[d];
            int64_t a = INT64_MAX, b = -1;
            for (int64_t i = 0; i < sn[d]; i++) { int64_t v = (int64_t)floor((so[d] + i + 0.5) * r); if (v < 0) v = 0; if (v >= J->ishape[d]) v = J->ishape[d] - 1; map[d][i] = v; if (v < a) a = v; if (v > b) b = v; }
            io[d] = a; in[d] = b - a + 1;
            for (int64_t i = 0; i < sn[d]; i++) map[d][i] -= a;
        }
        const int64_t pn[3] = {sn[0], sn[1], sn[2]};
        const size_t inn = (size_t)in[0] * in[1] * in[2];
        if (inn > icap) { free(ib); ib = malloc(inn); icap = inn; if (!ib) { atomic_store(&J->fail, 1); break; } }
        if (z3_read(zp, so, pn, p, 1) || z3_read(zi, io, in, ib, 1) || (zg && z3_read(zg, so, pn, g, 1))) { fprintf(stderr, "ink-gate: %s\n", z3_error()); atomic_store(&J->fail, 1); break; }
        memset(o, 255, n); long long ng = 0, ni = 0, nk = 0;
        for (int64_t z = 0; z < sn[0]; z++) for (int64_t y = 0; y < sn[1]; y++) {
            const size_t src = ((size_t)z * sn[1] + y) * sn[2], dst = ((size_t)z * S + y) * S, irow = ((size_t)map[0][z] * in[1] + map[1][y]) * in[2];
            for (int64_t x = 0; x < sn[2]; x++) {
                const int c = p[src + x]; if (c > 251) continue; nk++;
                if (!ig_in(J, c) || (g && g[src + x])) continue;
                const uint8_t v = J->lut[ib[irow + map[2][x]]]; o[dst + x] = v; ng++; if (v >= 127) ni++;
            }
        }
        atomic_fetch_add(&J->ngate, ng); atomic_fetch_add(&J->nink, ni); atomic_fetch_add(&J->nknown, nk);
        if (ng && z3w_write_shard(J->zw, so[0] / S, so[1] / S, so[2] / S, o, 1)) { fprintf(stderr, "ink-gate: %s\n", z3w_error()); atomic_store(&J->fail, 1); break; }
    }
done:
    free(p); free(g); free(o); free(ib); for (int d = 0; d < 3; d++) free(map[d]);
    if (zp) z3_close(zp); if (zi) z3_close(zi); if (zg) z3_close(zg); if (sp) store_close(sp); if (si) store_close(si); if (sg) store_close(sg);
    return nullptr;
}

int cmd_ink_gate(int argc, char **argv) {
    ig_job J = {0};
    if (argc < 3 || !ig_opt2(argc, argv, "--ink", &J.ir, &J.ik) || !ig_opt2(argc, argv, "--phase", &J.pr, &J.pk)) {
        fprintf(stderr, "usage: ufsm ink-gate OUT --ink ROOT KEY --ink-um U --phase ROOT KEY --phase-um U [--ignore ROOT KEY] [--lo 0] [--hi 125] [--ink-max 255] [--shard 512] [--threads N]\n");
        return 2;
    }
    ig_opt2(argc, argv, "--ignore", &J.gr, &J.gk);
    const char *out = argv[2];
    J.ium = atof(ig_opt(argc, argv, "--ink-um", "0")); J.pum = atof(ig_opt(argc, argv, "--phase-um", "0"));
    J.lo = atoi(ig_opt(argc, argv, "--lo", "0")); J.hi = atoi(ig_opt(argc, argv, "--hi", "125")); J.imax = atoi(ig_opt(argc, argv, "--ink-max", "255"));
    J.S = atoi(ig_opt(argc, argv, "--shard", "512")); int nth = atoi(ig_opt(argc, argv, "--threads", "0")); if (nth <= 0) nth = ufsm_ncpu();
    if (!(J.ium > 0 && J.pum > 0) || J.lo < 0 || J.lo > 251 || J.hi < 0 || J.hi > 251 || J.imax < 1 || J.imax > 255 || J.S <= 0 || J.S % 128) { fprintf(stderr, "ink-gate: bad option value\n"); return 2; }
    for (int v = 0; v < 256; v++) { int q = (int)lround(254.0 * (v > J.imax ? J.imax : v) / J.imax); J.lut[v] = (uint8_t)q; }
    store *sp = store_open(J.pr), *si = store_open(J.ir); z3 *zp = sp ? z3_open(sp, J.pk, nullptr) : nullptr, *zi = si ? z3_open(si, J.ik, nullptr) : nullptr;
    if (!zp || !zi) { fprintf(stderr, "ink-gate: cannot open %s %s/%s\n", zp ? "the ink" : "the phase", zp ? J.ir : J.pr, zp ? J.ik : J.pk); return 1; }
    memcpy(J.shape, z3_meta_of(zp)->shape, sizeof J.shape); memcpy(J.ishape, z3_meta_of(zi)->shape, sizeof J.ishape);
    if (J.gr) {
        store *sg = store_open(J.gr); z3 *zg = sg ? z3_open(sg, J.gk, nullptr) : nullptr;
        if (!zg || memcmp(z3_meta_of(zg)->shape, J.shape, sizeof J.shape)) { fprintf(stderr, "ink-gate: cannot open the ignore mask %s/%s or its shape differs from the phase's\n", J.gr, J.gk); return 1; }
        z3_close(zg); store_close(sg);
    }
    for (int d = 0; d < 3; d++) J.ns[d] = (J.shape[d] + J.S - 1) / J.S;
    int64_t g[3]; uint8_t *bits = nullptr;
    if (z3_chunk_map(zp, g, &bits) >= 0 && bits) {
        const int64_t c = z3_meta_of(zp)->chunk[0];
        J.present = calloc((size_t)(J.ns[0] * J.ns[1] * J.ns[2]), 1);
        for (int64_t a = 0; a < g[0]; a++) for (int64_t b = 0; b < g[1]; b++) for (int64_t e = 0; e < g[2]; e++)
            if (bits[(a * g[1] + b) * g[2] + e]) J.present[((a * c / J.S) * J.ns[1] + b * c / J.S) * J.ns[2] + e * c / J.S] = 1;
        free(bits);
    }
    z3_close(zp); z3_close(zi); store_close(sp); store_close(si);
    char lv[64], p[2048]; z3w_level_name(J.pum, lv, sizeof lv); snprintf(p, sizeof p, "%s/%s", out, lv);
    char at[512]; snprintf(at, sizeof at, "{\"ink_gate\":{\"lo\":%d,\"hi\":%d,\"encoding\":\"prob254\"}}", J.lo, J.hi);
    if (z3w_write_group(out, &J.pum, 1, "ink-gate", at)) { fprintf(stderr, "ink-gate: %s\n", z3w_error()); return 1; }
    J.zw = z3w_create(p, J.shape, J.S, 0, 255, at);
    if (!J.zw) { fprintf(stderr, "ink-gate: %s\n", z3w_error()); return 1; }
    const int nw = nth < 64 ? nth : 64; pthread_t th[64];
    for (int t = 0; t < nw; t++) pthread_create(&th[t], nullptr, ig_worker, &J);
    for (int t = 0; t < nw; t++) pthread_join(th[t], nullptr);
    free(J.present);
    if (z3w_close(J.zw) || atomic_load(&J.fail)) { fprintf(stderr, "ink-gate: FAILED\n"); return 1; }
    const long long ng = atomic_load(&J.ngate), ni = atomic_load(&J.nink), nk = atomic_load(&J.nknown);
    printf("ink-gate: %lld of %lld labelled voxels in codes %d..%d (%.3f), %lld with p >= .5 (%.4f of gated), %.3f mm3 gated -> %s\n",
           ng, nk, J.lo, J.hi, nk ? (double)ng / nk : 0, ni, ng ? (double)ni / ng : 0, ng * J.pum * J.pum * J.pum * 1e-9, p);
    return 0;
}
