/* ufsm phase-coh PHASE_ROOT PHASE_KEY CT_ROOT CT_KEY OUT --weight T0,T1 [--weight2 T0,T1 -> OUT/weight2.zarr] [--cell 32] [--zband Z0,Z1] [--threads N]
   The CT-coherence confidence of an already rendered stored phase (surf-labels' --weight-coh after the fact: re-gating a render
   without redoing it). Per cell of --cell^3 voxels (the cell grid from the array origin, as surf-labels' blocks), r = the correlation
   of the CT with cos(2 pi phase) over the voxels with a known phase and CT > 0 (none: fewer than 5% of the cell's voxels, or no
   variance); per known-phase voxel weight 255 x clamp((r - T0) / (T1 - T0)), 0 elsewhere -> OUT/weight.zarr/<PHASE_KEY>, the
   sampler's sources "phase": {"weight": {"root": OUT/weight.zarr, "key": PHASE_KEY}} (8^3 blocks keep their phase with probability
   w / 255). The CT must be on the phase's grid (the --like level of the render). surf-labels computes r on the phase before its
   ignore bake, phase-coh on the stored (baked) phase: cells with ignored voxels can differ; cells without are the same within
   float rounding. Prints the kept-weight histogram (share of known voxels per weight bin, mean weight).
   --box-scale FILE [--box-default D]: an external per-box factor multiplied into both weights (lines "z y x nz ny nx s" on the phase
   grid, e.g. a fibre judge's per-tile weight; '#' comments; a voxel in several boxes takes the first), D (default 1) outside every box.
   --papyrus T [--pap-dilate R] [--phase-out 1]: a voxel papyrus mask. The pair-rule fill between two wraps labels everything between
   them, air gaps and the case included (surf-labels ignores only CT 0 = outside the scan mask); a known voxel whose CT is < T and that
   has no CT >= T within R voxels (a (2R+1)^3 cube, default R 2) gets weight 0 in both weights; with --phase-out 1 the phase itself is
   written with those voxels unknown -> OUT/phase.zarr/<PHASE_KEY> (for the sampler, which keeps or drops whole 8^3 blocks by the weight,
   and for fiber-judge --phase). Air-masked known voxels are counted. */
#include "band.h"
#include "store.h"
#include "sysinfo.h"
#include "z3w.h"
#include "zarr3.h"
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *pc_opt(int argc, char **argv, const char *k, const char *d) { for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return argv[i + 1]; return d; }

enum { PC_S = 512, PC_WB = 10 };
typedef struct {
    z3 *ph, *ct; z3w *w, *w2; int64_t shape[3], ns[3]; int cell; double t0, t1, u0, u1; long long zb0, zb1;
    _Atomic long next; long nitems; _Atomic int fail; float cs[PHASE_PERIOD];
    int nbox; double bdef; int64_t (*box)[6]; double *bs; _Atomic long long inbox, wpre;   /* --box-scale; wpre: the weight sum before it */
    int pap_t, pap_r; z3w *po; _Atomic long long air;   /* --papyrus T, --pap-dilate R, --phase-out; air: known voxels masked */
    _Atomic long long known, hist[PC_WB + 1], hist2[PC_WB + 1], cells, cells_r; _Atomic long long wsum, wsum2, ge[3];   /* known voxels in cells with r >= 0.2 / 0.3 / 0.4 */   /* hist bin PC_WB: weight 255 */
} pc_job;

/* the dilated papyrus mask of a shard: m[i] = 1 where some CT >= T lies within R voxels (cube), read with an R halo */
static int pc_papmask(pc_job *J, const int64_t o[3], const int64_t n[3], uint8_t *m, uint8_t *h, uint8_t *t) {
    const int R = J->pap_r; int64_t ho[3], hn[3], lo[3];
    for (int d = 0; d < 3; d++) { ho[d] = o[d] - R < 0 ? 0 : o[d] - R; const int64_t e = o[d] + n[d] + R > J->shape[d] ? J->shape[d] : o[d] + n[d] + R; hn[d] = e - ho[d]; lo[d] = o[d] - ho[d]; }
    if (z3_read(J->ct, ho, hn, h, 1)) return -1;
    const size_t HN = (size_t)(hn[0] * hn[1] * hn[2]);
    for (size_t i = 0; i < HN; i++) h[i] = h[i] >= J->pap_t;
    if (R > 0) for (int ax = 2; ax >= 0; ax--) {   /* separable running max along x, y, z: h -> t -> h */
        const int64_t st = ax == 2 ? 1 : ax == 1 ? hn[2] : hn[1] * hn[2], L = hn[ax];
        for (size_t i = 0; i < HN; i++) {
            const int64_t c = (int64_t)(i / (size_t)st) % L; uint8_t v = 0;
            for (int64_t k = c - R < 0 ? 0 : c - R, k1 = c + R >= L ? L - 1 : c + R; k <= k1 && !v; k++) v = h[i + (k - c) * st];
            t[i] = v; }
        memcpy(h, t, HN); }
    for (int64_t z = 0; z < n[0]; z++) for (int64_t y = 0; y < n[1]; y++)
        memcpy(m + ((size_t)z * n[1] + y) * n[2], h + (((size_t)(z + lo[0]) * hn[1] + y + lo[1]) * hn[2] + lo[2]), (size_t)n[2]);
    return 0;
}

static int pc_shard(pc_job *J, long it, uint8_t *ph, uint8_t *ct, uint8_t *w, uint8_t *w2, double *acc, float *r, uint8_t *pm, uint8_t *ph1, uint8_t *ph2) {
    const int64_t sz = it / (J->ns[1] * J->ns[2]), sy = it / J->ns[2] % J->ns[1], sx = it % J->ns[2];
    if (J->zb1 > J->zb0 && ((sz + 1) * PC_S <= J->zb0 || sz * PC_S >= J->zb1)) return 0;
    const int64_t o[3] = {sz * PC_S, sy * PC_S, sx * PC_S}; int64_t n[3];
    for (int d = 0; d < 3; d++) n[d] = J->shape[d] - o[d] < PC_S ? J->shape[d] - o[d] : PC_S;
    int bl[256], nbl = 0;   /* the --box-scale boxes meeting this shard */
    for (int b = 0; b < J->nbox && nbl < 256; b++) { int hit = 1; for (int d = 0; d < 3; d++) hit &= J->box[b][d] < o[d] + n[d] && J->box[b][d] + J->box[b][d + 3] > o[d]; if (hit) bl[nbl++] = b; }
    if (z3_read(J->ph, o, n, ph, 1)) return -1;
    const size_t NN = (size_t)(n[0] * n[1] * n[2]); size_t nk = 0;
    for (size_t i = 0; i < NN; i++) nk += ph[i] != PHASE_UNKNOWN;
    if (!nk) return 0;   /* (no shard written: the weight is 0 there) */
    if (z3_read(J->ct, o, n, ct, 1)) return -1;
    if (J->pap_t > 0 && pc_papmask(J, o, n, pm, ph1, ph2)) return -1;
    const int C = J->cell; const int64_t nc[3] = {(n[0] + C - 1) / C, (n[1] + C - 1) / C, (n[2] + C - 1) / C}; const size_t NC = (size_t)(nc[0] * nc[1] * nc[2]);
    memset(acc, 0, NC * 6 * sizeof *acc);
    for (int64_t z = 0; z < n[0]; z++) for (int64_t y = 0; y < n[1]; y++) for (int64_t x = 0; x < n[2]; x++) {
        const size_t i = ((size_t)z * n[1] + y) * n[2] + x; const int p = ph[i], v = ct[i];
        if (p == PHASE_UNKNOWN || !v) continue;
        double *a = acc + 6 * (((size_t)(z / C) * nc[1] + y / C) * nc[2] + x / C); const double c = J->cs[p];
        a[0] += 1; a[1] += v; a[2] += (double)v * v; a[3] += c; a[4] += c * c; a[5] += v * c;
    }
    long long cells = 0, cells_r = 0;
    for (size_t k = 0; k < NC; k++) {
        const double *a = acc + 6 * k, m = a[0]; r[k] = NAN; if (m > 0) cells++;
        if (m < 0.05 * C * C * C) continue;
        const double vv = a[2] / m - (a[1] / m) * (a[1] / m), cc = a[4] / m - (a[3] / m) * (a[3] / m), cv = a[5] / m - (a[1] / m) * (a[3] / m);
        if (vv > 1e-9 && cc > 1e-9) { r[k] = (float)(cv / sqrt(vv * cc)); cells_r++; }
    }
    memset(w, 0, (size_t)PC_S * PC_S * PC_S); if (J->w2) memset(w2, 0, (size_t)PC_S * PC_S * PC_S);
    long long hist[PC_WB + 1] = {0}, hist2[PC_WB + 1] = {0}, ws = 0, ws2 = 0, ge[3] = {0}, inb = 0, wpre = 0, air = 0;
    for (int64_t z = 0; z < n[0]; z++) for (int64_t y = 0; y < n[1]; y++) for (int64_t x = 0; x < n[2]; x++) {
        const size_t i = ((size_t)z * n[1] + y) * n[2] + x; if (ph[i] == PHASE_UNKNOWN) continue;
        const float rr = r[((size_t)(z / C) * nc[1] + y / C) * nc[2] + x / C];
        if (isfinite(rr)) { if (rr >= 0.2f) ge[0]++; if (rr >= 0.3f) ge[1]++; if (rr >= 0.4f) ge[2]++; }
        double wv = 0; if (isfinite(rr)) { wv = (rr - J->t0) / (J->t1 - J->t0); wv = wv < 0 ? 0 : wv > 1 ? 1 : wv; }
        double bsc = 1;
        if (J->nbox) { bsc = J->bdef; const int64_t g[3] = {o[0] + z, o[1] + y, o[2] + x};
                       for (int j = 0; j < nbl; j++) { const int64_t *B = J->box[bl[j]]; if (g[0] >= B[0] && g[0] < B[0] + B[3] && g[1] >= B[1] && g[1] < B[1] + B[4] && g[2] >= B[2] && g[2] < B[2] + B[5]) { bsc = J->bs[bl[j]]; inb++; break; } }
                       wpre += lround(255 * wv); wv *= bsc; }
        if (J->pap_t > 0 && !pm[i]) { bsc = 0; wv = 0; air++; if (J->po) ph[i] = PHASE_UNKNOWN; }
        const int q = (int)lround(255 * wv); w[((size_t)z * PC_S + y) * PC_S + x] = (uint8_t)q; ws += q;
        hist[q == 255 ? PC_WB : q * PC_WB / 255]++;
        if (J->w2) { double u = 0; if (isfinite(rr)) { u = (rr - J->u0) / (J->u1 - J->u0); u = u < 0 ? 0 : u > 1 ? 1 : u; } u *= bsc;
                     const int q2 = (int)lround(255 * u); w2[((size_t)z * PC_S + y) * PC_S + x] = (uint8_t)q2; ws2 += q2; hist2[q2 == 255 ? PC_WB : q2 * PC_WB / 255]++; }
    }
    if (z3w_write_shard(J->w, sz, sy, sx, w, 1) || (J->w2 && z3w_write_shard(J->w2, sz, sy, sx, w2, 1))) return -1;
    if (J->po) {   /* the masked phase: the shard re-packed to PC_S^3 (fill unknown) */
        memset(ph1, PHASE_UNKNOWN, (size_t)PC_S * PC_S * PC_S);
        for (int64_t z = 0; z < n[0]; z++) for (int64_t y = 0; y < n[1]; y++) memcpy(ph1 + ((size_t)z * PC_S + y) * PC_S, ph + ((size_t)z * n[1] + y) * n[2], (size_t)n[2]);
        if (z3w_write_shard(J->po, sz, sy, sx, ph1, 1)) return -1; }
    atomic_fetch_add(&J->air, air);
    atomic_fetch_add(&J->wsum2, ws2); for (int t = 0; t < 3; t++) atomic_fetch_add(&J->ge[t], ge[t]); for (int b = 0; b <= PC_WB; b++) atomic_fetch_add(&J->hist2[b], hist2[b]);
    atomic_fetch_add(&J->inbox, inb); atomic_fetch_add(&J->wpre, wpre);
    atomic_fetch_add(&J->known, (long long)nk); atomic_fetch_add(&J->wsum, ws); atomic_fetch_add(&J->cells, cells); atomic_fetch_add(&J->cells_r, cells_r);
    for (int b = 0; b <= PC_WB; b++) atomic_fetch_add(&J->hist[b], hist[b]);
    return 0;
}

static void *pc_run(void *arg) {
    pc_job *J = arg; const size_t S3 = (size_t)PC_S * PC_S * PC_S, NC = (size_t)(PC_S / J->cell + 1) * (PC_S / J->cell + 1) * (PC_S / J->cell + 1);
    uint8_t *ph = malloc(S3), *ct = malloc(S3), *w = malloc(S3), *w2 = J->w2 ? malloc(S3) : nullptr; double *acc = malloc(NC * 6 * sizeof *acc); float *r = malloc(NC * sizeof *r);
    const size_t SH = (size_t)(PC_S + 2 * J->pap_r) * (PC_S + 2 * J->pap_r) * (PC_S + 2 * J->pap_r); const int pap = J->pap_t > 0;
    uint8_t *pm = pap ? malloc(S3) : nullptr, *ph1 = pap || J->po ? malloc(SH) : nullptr, *ph2 = pap ? malloc(SH) : nullptr;
    if (!ph || !ct || !w || (J->w2 && !w2) || !acc || !r || (pap && (!pm || !ph1 || !ph2))) atomic_store(&J->fail, 1);
    else for (long it; (it = atomic_fetch_add(&J->next, 1)) < J->nitems && !atomic_load(&J->fail);)
        if (pc_shard(J, it, ph, ct, w, w2, acc, r, pm, ph1, ph2)) { fprintf(stderr, "phase-coh: shard %ld failed: %s %s\n", it, z3_error(), z3w_error()); atomic_store(&J->fail, 1); }
    free(ph); free(ct); free(w); free(w2); free(acc); free(r); free(pm); free(ph1); free(ph2);
    return nullptr;
}

int cmd_phase_coh(int argc, char **argv) {
    if (argc < 7) { fprintf(stderr, "usage: ufsm phase-coh PHASE_ROOT PHASE_KEY CT_ROOT CT_KEY OUT --weight T0,T1 [--weight2 T0,T1] [--cell 32] [--zband Z0,Z1] [--box-scale FILE [--box-default D]] [--papyrus T [--pap-dilate R] [--phase-out 1]] [--threads N]\n"); return 2; }
    pc_job J = {0};
    if (sscanf(pc_opt(argc, argv, "--weight", ""), "%lf,%lf", &J.t0, &J.t1) != 2 || !(J.t1 > J.t0)) { fprintf(stderr, "phase-coh: --weight T0,T1 with T1 > T0\n"); return 2; }
    J.cell = atoi(pc_opt(argc, argv, "--cell", "32")); if (J.cell < 4 || PC_S % J.cell) { fprintf(stderr, "phase-coh: --cell must divide %d\n", PC_S); return 2; }
    if (sscanf(pc_opt(argc, argv, "--zband", ""), "%lld,%lld", &J.zb0, &J.zb1) != 2) { J.zb0 = 0; J.zb1 = -1; }
    J.bdef = atof(pc_opt(argc, argv, "--box-default", "1"));
    J.pap_t = atoi(pc_opt(argc, argv, "--papyrus", "0")); J.pap_r = atoi(pc_opt(argc, argv, "--pap-dilate", "2"));
    if (J.pap_t < 0 || J.pap_t > 255 || J.pap_r < 0 || J.pap_r > 16) { fprintf(stderr, "phase-coh: --papyrus 1..255, --pap-dilate 0..16\n"); return 2; }
    const int phase_out = atoi(pc_opt(argc, argv, "--phase-out", "0"));
    if (phase_out && !J.pap_t) { fprintf(stderr, "phase-coh: --phase-out needs --papyrus\n"); return 2; }
    if (pc_opt(argc, argv, "--box-scale", nullptr)) {
        FILE *f = fopen(pc_opt(argc, argv, "--box-scale", ""), "r"); if (!f) { fprintf(stderr, "phase-coh: cannot read --box-scale %s\n", pc_opt(argc, argv, "--box-scale", "")); return 1; }
        char ln[512]; int cap = 0;
        while (fgets(ln, sizeof ln, f)) { long long v[6]; double s; if (ln[0] == '#' || sscanf(ln, "%lld %lld %lld %lld %lld %lld %lf", v, v + 1, v + 2, v + 3, v + 4, v + 5, &s) != 7) continue;
            if (J.nbox == cap) { cap = cap ? 2 * cap : 64; J.box = realloc(J.box, (size_t)cap * sizeof *J.box); J.bs = realloc(J.bs, (size_t)cap * sizeof *J.bs); }
            for (int d = 0; d < 6; d++) { J.box[J.nbox][d] = v[d]; }
            J.bs[J.nbox++] = s < 0 ? 0 : s > 1 ? 1 : s; }
        fclose(f); if (!J.nbox) { fprintf(stderr, "phase-coh: --box-scale: no boxes read\n"); return 2; } }
    const int nt = ufsm_threads(atoi(pc_opt(argc, argv, "--threads", "0")), 8);
    J.ph = z3_open(store_open(argv[2]), argv[3], nullptr); J.ct = z3_open(store_open(argv[4]), argv[5], nullptr);
    if (!J.ph || !J.ct) { fprintf(stderr, "phase-coh: cannot open %s: %s\n", !J.ph ? argv[2] : argv[4], z3_error()); return 1; }
    const int64_t *ps = z3_meta_of(J.ph)->shape, *cs = z3_meta_of(J.ct)->shape;
    if (memcmp(ps, cs, 3 * sizeof *ps)) { fprintf(stderr, "phase-coh: the CT (%lld,%lld,%lld) is not on the phase's grid (%lld,%lld,%lld)\n",
                                                  (long long)cs[0], (long long)cs[1], (long long)cs[2], (long long)ps[0], (long long)ps[1], (long long)ps[2]); return 1; }
    for (int d = 0; d < 3; d++) { J.shape[d] = ps[d]; J.ns[d] = (ps[d] + PC_S - 1) / PC_S; }
    J.nitems = (long)(J.ns[0] * J.ns[1] * J.ns[2]);
    for (int k = 0; k < PHASE_PERIOD; k++) J.cs[k] = (float)cos(2 * M_PI * k / PHASE_PERIOD);   /* (surf-labels' table) */
    char p[1400], attrs[600]; const double um = atof(argv[3]);
    snprintf(attrs, sizeof attrs, "{\"ufsm\":{\"content\":\"phase confidence 0..255: 255 x clamp((r - %g) / (%g - %g)) of the cell's CT coherence r, 0 where the phase is unknown\",\"source\":\"phase-coh %s %s, cell %d\"}}",
             J.t0, J.t1, J.t0, argv[2], argv[3], J.cell);
    snprintf(p, sizeof p, "%s/weight.zarr/%s", argv[6], argv[3]); J.w = z3w_create(p, J.shape, PC_S, 0, 0, attrs);
    snprintf(p, sizeof p, "%s/weight.zarr", argv[6]);
    if (!J.w || z3w_write_group(p, &um, 1, "phase-coh-weight", attrs)) { fprintf(stderr, "phase-coh: %s\n", z3w_error()); return 1; }
    if (phase_out) {
        char a3[700]; snprintf(a3, sizeof a3, "{\"ufsm\":{\"content\":\"band field winding phase (src/band.h PHASE_*: 0..251, 255 unknown), unknown where CT < %d with no CT >= %d within %d voxels\",\"source\":\"phase-coh --papyrus %s %s\"}}",
                               J.pap_t, J.pap_t, J.pap_r, argv[2], argv[3]);
        snprintf(p, sizeof p, "%s/phase.zarr/%s", argv[6], argv[3]); J.po = z3w_create(p, J.shape, PC_S, 0, PHASE_UNKNOWN, a3);
        snprintf(p, sizeof p, "%s/phase.zarr", argv[6]);
        if (!J.po || z3w_write_group(p, &um, 1, "phase-coh-phase", a3)) { fprintf(stderr, "phase-coh: %s\n", z3w_error()); return 1; } }
    if (pc_opt(argc, argv, "--weight2", nullptr)) {   /* a second ramp from the same pass -> OUT/weight2.zarr (e.g. 0.3,0.6 for a w4 arm) */
        if (sscanf(pc_opt(argc, argv, "--weight2", ""), "%lf,%lf", &J.u0, &J.u1) != 2 || !(J.u1 > J.u0)) { fprintf(stderr, "phase-coh: --weight2 T0,T1 with T1 > T0\n"); return 2; }
        char a2[600]; snprintf(a2, sizeof a2, "{\"ufsm\":{\"content\":\"phase confidence 0..255: 255 x clamp((r - %g) / (%g - %g)) of the cell's CT coherence r, 0 where the phase is unknown\",\"source\":\"phase-coh %s %s, cell %d\"}}",
                               J.u0, J.u1, J.u0, argv[2], argv[3], J.cell);
        snprintf(p, sizeof p, "%s/weight2.zarr/%s", argv[6], argv[3]); J.w2 = z3w_create(p, J.shape, PC_S, 0, 0, a2);
        snprintf(p, sizeof p, "%s/weight2.zarr", argv[6]);
        if (!J.w2 || z3w_write_group(p, &um, 1, "phase-coh-weight", a2)) { fprintf(stderr, "phase-coh: %s\n", z3w_error()); return 1; } }
    const int m = nt < 64 ? nt : 64; pthread_t th[64];
    for (int q = 0; q < m; q++) pthread_create(&th[q], nullptr, pc_run, &J);
    for (int q = 0; q < m; q++) pthread_join(th[q], nullptr);
    if (z3w_close(J.w) || (J.w2 && z3w_close(J.w2)) || (J.po && z3w_close(J.po)) || atomic_load(&J.fail)) { fprintf(stderr, "phase-coh: failed\n"); return 1; }
    const long long kn = J.known;
    printf("phase-coh: %lld known-phase voxels, %lld cells with labels (%lld with a coherence), mean weight %.4f; weight %g..%g\n", kn, (long long)J.cells, (long long)J.cells_r,
           kn ? (double)J.wsum / (255.0 * kn) : 0, J.t0, J.t1);
    printf("phase-coh: known voxels in cells with coherence >= 0.2 / 0.3 / 0.4: %.4f %.4f %.4f\n", kn ? (double)J.ge[0] / kn : 0, kn ? (double)J.ge[1] / kn : 0, kn ? (double)J.ge[2] / kn : 0);
    printf("phase-coh: share of known voxels by weight (0-0.1 ... 0.9-1, =1):");
    for (int b = 0; b <= PC_WB; b++) printf(" %.4f", kn ? (double)J.hist[b] / kn : 0);
    printf("\n");
    if (J.nbox) printf("phase-coh: --box-scale %d boxes (default %g outside): %lld known voxels in boxes (%.4f); weight sum kept %.4f of the coherence weight\n", J.nbox, J.bdef,
                       (long long)J.inbox, kn ? (double)J.inbox / kn : 0, J.wpre ? (double)J.wsum / J.wpre : 0);
    if (J.pap_t) printf("phase-coh: --papyrus %d dilate %d: %lld known voxels in air masked (%.4f of known)%s\n", J.pap_t, J.pap_r, (long long)J.air, kn ? (double)J.air / kn : 0, J.po ? ", masked phase written" : "");
    if (J.w2) { printf("phase-coh: weight2 %g..%g mean %.4f, by weight:", J.u0, J.u1, kn ? (double)J.wsum2 / (255.0 * kn) : 0);
                for (int b = 0; b <= PC_WB; b++) { printf(" %.4f", kn ? (double)J.hist2[b] / kn : 0); }
                printf("\n"); }
    z3_close(J.ph); z3_close(J.ct);
    return 0;
}
