/* ufsm side-gate <phase-root> <phase-key> <ct-root> <ct-key> <out-group> <out-um> [--iso 64] [--runs A,B] [--len-um LO,HI] [--cap 64]
                  [--gt ROOT KEY] [--box z,y,x,nz,ny,nx] [--block 128] [--shard 512] [--threads 16] [--cache DIR] [--no-out 1]
   A local, GT-free gate for side labels from OUR surfaces (surf-labels phase rasters, train --rv): our wraps hop sheets in packed papyrus,
   which puts a turn of our phase over 0 or 2+ CT sheets. Per papyrus voxel x (CT >= --iso) with a known phase: walk along the phase
   normal inward until the phase wraps (our wrap k) and outward until it wraps again (k + 1), up to --cap voxels each way; along that
   one-turn interval count the papyrus runs (CT >= iso separated by >= 1 void voxel) R and the interval length L (um). The voxel is kept
   when R in [A, B] and L in [LO, HI]; else unknown. Output: the gated phase (PHASE_*, 255 unknown) on the same grid, 512^3 shards.
   --gt ROOT KEY (a published-GT phase on the same grid): the papyrus side agreement (both phases on the same half turn) of every
   (R, L bin) cell and of the kept set: the calibration table for choosing --runs / --len-um. */
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

static const char *sg_opt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return dflt;
}
enum { SG_R = 6, SG_L = 8 };   /* runs 0..4, 5+ ; length bins */
static const double sg_lb[SG_L - 1] = {40, 60, 80, 100, 130, 170, 230};   /* um */
typedef struct {
    z3 *ph, *ct, *gt; int iso, cap, B, H, S, r0, r1; double l0, l1, um;
    int64_t shape[3], bo[3], bn[3];
} sg_job;
typedef struct { long long n[SG_R][SG_L], a[SG_R][SG_L], none_n, none_a, pap, kept, kept_gt, kept_a; } sg_stats;

static inline float sg_dturn(int a, int b) { int d = b - a; d %= PHASE_PERIOD; if (d > PHASE_PERIOD / 2) d -= PHASE_PERIOD; if (d <= -PHASE_PERIOD / 2) d += PHASE_PERIOD; return (float)d / PHASE_PERIOD; }
static inline int sg_lbin(double l) { int b = 0; while (b < SG_L - 1 && l >= sg_lb[b]) b++; return b; }

static int sg_block(sg_job *J, const int64_t o[3], const int n[3], uint8_t *out, sg_stats *st) {
    const int H = J->H, D = n[0] + 2 * H, Y = n[1] + 2 * H, X = n[2] + 2 * H; const size_t N = (size_t)D * Y * X;
    uint8_t *ph = malloc(N), *ct = malloc(N), *gt = J->gt ? malloc((size_t)n[0] * n[1] * n[2]) : nullptr; int rc = -1;
    if (!ph || !ct || (J->gt && !gt)) goto done;
    { const int64_t ho[3] = {o[0] - H, o[1] - H, o[2] - H}, hn[3] = {D, Y, X}, nn[3] = {n[0], n[1], n[2]};
      if (z3_read(J->ph, ho, hn, ph, 1) || z3_read(J->ct, ho, hn, ct, 1) || (gt && z3_read(J->gt, o, nn, gt, 1))) goto done; }
    const size_t sz = (size_t)Y * X, sy = X;
    for (int z = H; z < H + n[0]; z++) for (int y = H; y < H + n[1]; y++) for (int x = H; x < H + n[2]; x++) {
        const size_t i = ((size_t)z * Y + y) * X + x, oi = ((size_t)(z - H) * n[1] + (y - H)) * n[2] + (x - H);
        const int v = ph[i]; out[oi] = PHASE_UNKNOWN;
        if (v >= PHASE_PERIOD || ct[i] < J->iso) { continue; }
        const int a0 = ph[i - sz], a1 = ph[i + sz], b0 = ph[i - sy], b1 = ph[i + sy], c0 = ph[i - 1], c1 = ph[i + 1];
        st->pap++;
        const int g_known = a0 < PHASE_PERIOD && a1 < PHASE_PERIOD && b0 < PHASE_PERIOD && b1 < PHASE_PERIOD && c0 < PHASE_PERIOD && c1 < PHASE_PERIOD;
        int R = -1; double L = -1;
        if (g_known) {
            const float g0 = sg_dturn(a0, a1), g1 = sg_dturn(b0, b1), g2 = sg_dturn(c0, c1), gl = sqrtf(g0 * g0 + g1 * g1 + g2 * g2);
            if (gl > 1e-4f) {
                const float nz = g0 / gl, ny = g1 / gl, nx = g2 / gl;   /* outward: the phase rises */
                int ends[2] = {-1, -1}, ok = 1;
                for (int dir = -1; dir <= 1 && ok; dir += 2) {   /* -1 inward to wrap k, +1 outward to wrap k + 1 */
                    int prev = v, t;
                    for (t = 1; t <= J->cap; t++) {
                        const int qz = (int)lrintf(z + dir * t * nz), qy = (int)lrintf(y + dir * t * ny), qx = (int)lrintf(x + dir * t * nx);
                        if (qz < 0 || qy < 0 || qx < 0 || qz >= D || qy >= Y || qx >= X) { ok = 0; break; }
                        const int w = ph[((size_t)qz * Y + qy) * X + qx]; if (w >= PHASE_PERIOD) { ok = 0; break; }
                        const float d = sg_dturn(prev, w);   /* the step's phase change; the raw codes wrap where the turn ends */
                        if (dir > 0 ? (w < prev && d > 0) : (w > prev && d < 0)) break;
                        prev = w;
                    }
                    if (t > J->cap) ok = 0;
                    ends[dir > 0] = t;
                }
                if (ok) {   /* runs along [-ends[0] + 1, ends[1] - 1] */
                    int runs = 0, inrun = 0;
                    for (int t = -ends[0] + 1; t <= ends[1] - 1; t++) {
                        const int qz = (int)lrintf(z + t * nz), qy = (int)lrintf(y + t * ny), qx = (int)lrintf(x + t * nx);
                        const int pap = ct[((size_t)qz * Y + qy) * X + qx] >= J->iso;
                        if (pap && !inrun) runs++;
                        inrun = pap;
                    }
                    R = runs; L = (ends[0] + ends[1] - 1) * J->um;
                }
            }
        }
        const int gv = gt ? gt[oi] : PHASE_UNKNOWN, has_gt = gv < PHASE_PERIOD, agree = has_gt && ((v < PHASE_PERIOD / 2) == (gv < PHASE_PERIOD / 2));
        if (R < 0) { if (has_gt) { st->none_n++; st->none_a += agree; } continue; }
        const int rb = R > SG_R - 1 ? SG_R - 1 : R, lb = sg_lbin(L);
        if (has_gt) { st->n[rb][lb]++; st->a[rb][lb] += agree; }
        if (R >= J->r0 && R <= J->r1 && L >= J->l0 && L <= J->l1) { out[oi] = (uint8_t)v; st->kept++; if (has_gt) { st->kept_gt++; st->kept_a += agree; } }
    }
    rc = 0;
done:
    free(ph); free(ct); free(gt); return rc;
}

typedef struct { sg_job *J; int64_t so[3]; uint8_t *buf; _Atomic int next; int nb; _Atomic int fail; sg_stats st; pthread_mutex_t mu; } sg_shard;
static void *sg_worker(void *arg) {
    sg_shard *T = arg; sg_job *J = T->J; const int B = J->B, S = J->S, nb1 = S / B;
    uint8_t *ob = malloc((size_t)B * B * B); sg_stats st; memset(&st, 0, sizeof st);
    if (!ob) { atomic_store(&T->fail, 1); return nullptr; }
    for (int b; (b = atomic_fetch_add(&T->next, 1)) < T->nb && !atomic_load(&T->fail);) {
        const int bz = b / (nb1 * nb1), by = (b / nb1) % nb1, bx = b % nb1;
        const int64_t o[3] = {T->so[0] * S + (int64_t)bz * B, T->so[1] * S + (int64_t)by * B, T->so[2] * S + (int64_t)bx * B}; int n[3], skip = 0;
        for (int d = 0; d < 3; d++) { n[d] = (int)(J->shape[d] - o[d] < B ? J->shape[d] - o[d] : B); if (n[d] <= 0 || o[d] + n[d] <= J->bo[d] || o[d] >= J->bo[d] + J->bn[d]) skip = 1; }
        if (skip) continue;
        if (sg_block(J, o, n, ob, &st)) { atomic_store(&T->fail, 1); break; }
        for (int z = 0; z < n[0]; z++) for (int y = 0; y < n[1]; y++)
            memcpy(T->buf + ((size_t)(bz * B + z) * S + by * B + y) * S + bx * B, ob + ((size_t)z * n[1] + y) * n[2], n[2]);
    }
    pthread_mutex_lock(&T->mu);
    long long *a = (long long *)&T->st, *s = (long long *)&st; for (size_t k = 0; k < sizeof st / sizeof(long long); k++) a[k] += s[k];
    pthread_mutex_unlock(&T->mu); free(ob); return nullptr;
}

int cmd_side_gate(int argc, char **argv) {
    if (argc < 8) { fprintf(stderr, "usage: ufsm side-gate <phase-root> <phase-key> <ct-root> <ct-key> <out-group> <out-um> [--iso 64] [--runs A,B] [--len-um LO,HI] [--cap 64]\n"
                                    "       [--gt ROOT KEY] [--box z,y,x,nz,ny,nx] [--block 128] [--shard 512] [--threads 16] [--cache DIR] [--no-out 1]\n"); return 2; }
    sg_job J; memset(&J, 0, sizeof J); const char *cache = sg_opt(argc, argv, "--cache", nullptr);
    J.ph = z3_open(store_open(argv[2]), argv[3], cache); J.ct = z3_open(store_open(argv[4]), argv[5], cache);
    if (!J.ph || !J.ct) { fprintf(stderr, "side-gate: %s\n", z3_error()); return 1; }
    J.iso = atoi(sg_opt(argc, argv, "--iso", "64")); J.cap = atoi(sg_opt(argc, argv, "--cap", "64")); J.B = atoi(sg_opt(argc, argv, "--block", "128"));
    J.S = atoi(sg_opt(argc, argv, "--shard", "512")); J.H = J.cap + 2; J.um = atof(argv[7]); J.r0 = 1; J.r1 = 1; J.l0 = 0; J.l1 = 1e9;
    if (*sg_opt(argc, argv, "--runs", "") && sscanf(sg_opt(argc, argv, "--runs", ""), "%d,%d", &J.r0, &J.r1) != 2) { fprintf(stderr, "side-gate: --runs A,B\n"); return 2; }
    if (*sg_opt(argc, argv, "--len-um", "") && sscanf(sg_opt(argc, argv, "--len-um", ""), "%lf,%lf", &J.l0, &J.l1) != 2) { fprintf(stderr, "side-gate: --len-um LO,HI\n"); return 2; }
    if (J.S % J.B || J.cap < 4 || !(J.um > 0)) { fprintf(stderr, "side-gate: bad options\n"); return 2; }
    const z3_meta *cm = z3_meta_of(J.ct), *pm = z3_meta_of(J.ph);
    for (int d = 0; d < 3; d++) { J.shape[d] = cm->shape[d]; J.bn[d] = cm->shape[d]; if (pm->shape[d] != cm->shape[d]) { fprintf(stderr, "side-gate: phase and CT shapes differ\n"); return 1; } }
    if (*sg_opt(argc, argv, "--box", "")) { long long v[6]; if (sscanf(sg_opt(argc, argv, "--box", ""), "%lld,%lld,%lld,%lld,%lld,%lld", v, v + 1, v + 2, v + 3, v + 4, v + 5) != 6) return 2; for (int d = 0; d < 3; d++) { J.bo[d] = v[d]; J.bn[d] = v[3 + d]; } }
    for (int i = 1; i + 2 < argc; i++) if (!strcmp(argv[i], "--gt") && !(J.gt = z3_open(store_open(argv[i + 1]), argv[i + 2], cache))) { fprintf(stderr, "side-gate: --gt: %s\n", z3_error()); return 1; }
    const int nt = ufsm_threads(atoi(sg_opt(argc, argv, "--threads", "0")), 16), noout = atoi(sg_opt(argc, argv, "--no-out", "0"));
    z3w *w = nullptr;
    if (!noout) {
        char lv[32], od[1400], at[600]; z3w_level_name(J.um, lv, sizeof lv); snprintf(od, sizeof od, "%s/%s", argv[6], lv);
        snprintf(at, sizeof at, "{\"ufsm\":{\"content\":\"side-gate of %s/%s: papyrus runs per turn %d..%d, turn length %g..%g um, CT %s iso %d\"}}", argv[2], argv[3], J.r0, J.r1, J.l0, J.l1, argv[5], J.iso);
        if (!(w = z3w_create(od, J.shape, J.S, 0, PHASE_UNKNOWN, at)) || z3w_write_group(argv[6], &J.um, 1, "ufsm-side-gate", nullptr)) { fprintf(stderr, "side-gate: %s\n", z3w_error()); return 1; }
    }
    const int S = J.S; const int64_t ns[3] = {(J.shape[0] + S - 1) / S, (J.shape[1] + S - 1) / S, (J.shape[2] + S - 1) / S};
    uint8_t *buf = malloc((size_t)S * S * S); sg_stats tot; memset(&tot, 0, sizeof tot); int nsh = 0; if (!buf) return 1;
    for (int64_t sz = J.bo[0] / S; sz < ns[0] && sz * S < J.bo[0] + J.bn[0]; sz++)
    for (int64_t sy = J.bo[1] / S; sy < ns[1] && sy * S < J.bo[1] + J.bn[1]; sy++)
    for (int64_t sx = J.bo[2] / S; sx < ns[2] && sx * S < J.bo[2] + J.bn[2]; sx++) {
        if (z3_shard_present(J.ph, sz * S / pm->shard[0], sy * S / pm->shard[1], sx * S / pm->shard[2]) == 0) continue;
        memset(buf, PHASE_UNKNOWN, (size_t)S * S * S);
        sg_shard T; memset(&T, 0, sizeof T); T.J = &J; T.so[0] = sz; T.so[1] = sy; T.so[2] = sx; T.buf = buf; T.nb = (S / J.B) * (S / J.B) * (S / J.B); pthread_mutex_init(&T.mu, nullptr);
        pthread_t th[64]; const int n = nt > 64 ? 64 : nt;
        for (int t = 0; t < n; t++) pthread_create(&th[t], nullptr, sg_worker, &T);
        for (int t = 0; t < n; t++) pthread_join(th[t], nullptr);
        if (atomic_load(&T.fail)) { fprintf(stderr, "side-gate: read failed: %s\n", z3_error()); return 1; }
        long long *a = (long long *)&tot, *s = (long long *)&T.st; for (size_t k = 0; k < sizeof tot / sizeof(long long); k++) a[k] += s[k];
        if (w && z3w_write_shard(w, sz, sy, sx, buf, nt)) { fprintf(stderr, "side-gate: %s\n", z3w_error()); return 1; }
        nsh++; fprintf(stderr, "\rside-gate: %d shards, papyrus %lld, kept %lld   ", nsh, tot.pap, tot.kept);
    }
    fprintf(stderr, "\n"); if (w) z3w_close(w);
    printf("side-gate: %d shards; papyrus with our phase %lld, kept %lld (%.4f) by runs %d..%d, length %g..%g um\n", nsh, tot.pap, tot.kept, tot.pap ? (double)tot.kept / tot.pap : 0, J.r0, J.r1, J.l0, J.l1);
    if (J.gt) {
        long long ng = tot.none_n; for (int r = 0; r < SG_R; r++) for (int l = 0; l < SG_L; l++) ng += tot.n[r][l];
        printf("side-gate: vs GT on papyrus (%lld voxels with both): kept %lld (share %.4f) agreement %.4f; no interval (walk failed) share %.4f agreement %.4f\n",
               ng, tot.kept_gt, ng ? (double)tot.kept_gt / ng : 0, tot.kept_gt ? (double)tot.kept_a / tot.kept_gt : 0, ng ? (double)tot.none_n / ng : 0, tot.none_n ? (double)tot.none_a / tot.none_n : 0);
        printf("side-gate: calibration (rows: papyrus runs in the voxel's turn; columns: turn length um; cell = share of GT voxels / side agreement)\n  runs ");
        for (int l = 0; l < SG_L; l++) { if (!l) printf("  <%-11.0f", sg_lb[0]); else if (l < SG_L - 1) printf(" %3.0f-%-8.0f", sg_lb[l - 1], sg_lb[l]); else printf(" >=%-10.0f", sg_lb[SG_L - 2]); }
        printf("\n");
        for (int r = 0; r < SG_R; r++) {
            printf("  %s%d ", r == SG_R - 1 ? ">=" : "  ", r);
            for (int l = 0; l < SG_L; l++) printf(" %.3f/%.3f ", ng ? (double)tot.n[r][l] / ng : 0, tot.n[r][l] ? (double)tot.a[r][l] / tot.n[r][l] : 0);
            printf("\n");
        }
    }
    free(buf); z3_close(J.ph); z3_close(J.ct); if (J.gt) z3_close(J.gt);
    return 0;
}
