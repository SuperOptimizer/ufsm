/* ufsm side-face <phase-root> <phase-key> <ct-root> <ct-key> <out-group> <out-um> [--iso 64] [--phase-scale 1|2] [--reach 3]
                  [--cap 48] [--cell 16] [--per-voxel 0] [--min-cell 8] [--box z,y,x,nz,ny,nx] [--block 128] [--shard 512] [--threads 16] [--cache DIR]
                  [--agree ROOT KEY] [--no-out 1]
   Face-anchored recto / verso labels (user 11-02, rv): the side rule k..k+0.5 = recto, k+0.5..k+1 = verso of a pair-rule phase raster
   (band.h PHASE_*), but with each wrap moved onto its sheet's RECTO (inner) face in the CT, so every sheet gets one clean verso -> recto
   transition at a real face instead of wherever the (noisy, ~40% mid-sheet) wrap happens to cut it.
   Per papyrus voxel x (CT >= --iso) with a known phase and gradient g (central circular differences, turns per voxel): walk inward along
   -g / |g| (toward the umbilicus: the phase falls) in 1-voxel steps up to min(--reach x the local spacing 1/|g|, --cap) voxels until the
   CT drops below --iso: the last papyrus voxel q is the inner face of x's sheet run (touching sheets: the run's innermost face). The wrap
   offset there is delta = phi(q). delta is smoothed into a field: the circular mean of the found voxels of each --cell^3 cell (16; >=
   --min-cell samples, resultant >= 0.5; else unknown) (--per-voxel 1: a voxel's own delta where its walk found a face: noisy in packed
   papyrus); the face-anchored phase is phi' = phi - delta (mod 1 turn): ~0 at the inner faces, rising outward by the GT spacing, so
   touching sheets keep alternating (the mod).
   Void voxels with a known phase take their cell's delta (or keep phi). Output: <out-group>/<out-um> phi' codes (PHASE_*, 255 unknown),
   the CT grid, 512^3 shards: a drop-in "phase" for train sources (train --rv: recto = phi' < 1/2 turn on papyrus).
   --phase-scale 2: the phase raster is on a 2x coarser grid than the CT / output (e.g. PHerc0139 GT 18.72 on the 9.362 scan): read
   circular-trilinearly. Validation printed at the end: papyrus with a known phase, found / cell / unknown shares, the recto share, and
   at every verso -> recto boundary on papyrus (x recto, its inward neighbour verso) the mean CT profile along the normal (-4..+4 voxels)
   and the share whose inward neighbour is void (a real face), for the raw phase and the face-anchored one. --agree ROOT KEY (another
   phase raster on the output grid, e.g. our surfaces' phase put through side-face too): the papyrus side agreement with it.
   --no-snap 1: no walks, the output is the phase as read (with --phase-scale 2: --interp linear (circular trilinear) or nearest, the
   sampler's read of a one-level-coarser stored phase): with --agree, the side agreement of a coarse label copy with the fine one. */
#include "band.h"
#include "store.h"
#include "sysinfo.h"
#include "z3w.h"
#include "zarr3.h"
#include <dirent.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const char *sf_opt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return dflt;
}

enum { SF_PROF = 9 };   /* CT profile samples -4..+4 */
typedef struct {
    z3 *ph, *ct, *ag; int pscale, iso, cap, cell, mincell, B, H, S; double reach;
    int64_t shape[3], bo[3], bn[3]; int pervox, nosnap, nearest;
} sf_job;

static float sf_cs[PHASE_PERIOD], sf_sn[PHASE_PERIOD];
static inline int sf_code(float x, float y) {   /* circular mean -> phase code */
    double a = atan2(y, x) / (2 * M_PI); a -= floor(a); int c = (int)lround(a * PHASE_PERIOD); return c >= PHASE_PERIOD ? 0 : c;
}
static inline float sf_dturn(int a, int b) {   /* (b - a) in turns, wrapped to (-1/2, 1/2] */
    int d = b - a; d %= PHASE_PERIOD; if (d > PHASE_PERIOD / 2) d -= PHASE_PERIOD; if (d <= -PHASE_PERIOD / 2) d += PHASE_PERIOD; return (float)d / PHASE_PERIOD;
}

/* one block [o, o + n) of the output grid (+ halo H): the face-anchored phase into out (n^3, C order) */
static int sf_block(sf_job *J, const int64_t o[3], const int n[3], uint8_t *out, long long st[8], double prof[2][SF_PROF], long long bs[4], long long ags[2]) {
    const int H = J->H, D = n[0] + 2 * H, Y = n[1] + 2 * H, X = n[2] + 2 * H;
    const size_t N = (size_t)D * Y * X;
    uint8_t *ph = malloc(N), *ct = malloc(N), *dl = malloc(N), *pp = malloc(N); float *gz = malloc(N * 3 * sizeof(float));
    int rc = -1;
    if (!ph || !ct || !dl || !pp || !gz) goto done;
    { const int64_t ho[3] = {o[0] - H, o[1] - H, o[2] - H}, hn[3] = {D, Y, X};
      if (z3_read(J->ct, ho, hn, ct, 1)) goto done;
      if (J->pscale == 1) { if (z3_read(J->ph, ho, hn, ph, 1)) goto done; }
      else {   /* 2x coarser phase: circular trilinear at voxel centres ((x + 0.5) / 2 - 0.5) */
          const int64_t co[3] = {(ho[0] >> 1) - 1, (ho[1] >> 1) - 1, (ho[2] >> 1) - 1}, cn[3] = {D / 2 + 3, Y / 2 + 3, X / 2 + 3};
          uint8_t *c = malloc((size_t)cn[0] * cn[1] * cn[2]); if (!c) goto done;
          if (z3_read(J->ph, co, cn, c, 1)) { free(c); goto done; }
          if (J->nearest) {   /* --interp nearest: what the sampler's phase_window does for a stored phase one level coarser */
              for (int z = 0; z < D; z++) for (int y = 0; y < Y; y++) for (int x = 0; x < X; x++)
                  ph[((size_t)z * Y + y) * X + x] = c[((size_t)(((ho[0] + z + 1) >> 1) - co[0]) * cn[1] + (((ho[1] + y + 1) >> 1) - co[1])) * cn[2] + (((ho[2] + x + 1) >> 1) - co[2])];
          } else
          for (int z = 0; z < D; z++) for (int y = 0; y < Y; y++) for (int x = 0; x < X; x++) {
              const double p[3] = {(ho[0] + z + 0.5) / 2 - 0.5 - co[0], (ho[1] + y + 0.5) / 2 - 0.5 - co[1], (ho[2] + x + 0.5) / 2 - 0.5 - co[2]};
              int b[3]; double f[3]; for (int d = 0; d < 3; d++) { b[d] = (int)floor(p[d]); f[d] = p[d] - b[d]; if (b[d] < 0) { b[d] = 0; f[d] = 0; } if (b[d] > cn[d] - 2) { b[d] = (int)cn[d] - 2; f[d] = 1; } }
              float sx = 0, sy = 0, wk = 0;
              for (int k = 0; k < 8; k++) {
                  const double w = (k & 4 ? f[0] : 1 - f[0]) * (k & 2 ? f[1] : 1 - f[1]) * (k & 1 ? f[2] : 1 - f[2]);
                  const int v = c[((size_t)(b[0] + (k >> 2 & 1)) * cn[1] + b[1] + (k >> 1 & 1)) * cn[2] + b[2] + (k & 1)];
                  if (w < 1e-7 || v >= PHASE_PERIOD) continue;
                  sx += (float)w * sf_cs[v]; sy += (float)w * sf_sn[v]; wk += (float)w;
              }
              ph[((size_t)z * Y + y) * X + x] = wk < 0.5f || sx * sx + sy * sy < 1e-8f ? PHASE_UNKNOWN : (uint8_t)sf_code(sx, sy);
          }
          free(c);
      } }
    /* gradient (turns / voxel) where the 6 neighbours are known */
    for (int z = 1; z < D - 1; z++) for (int y = 1; y < Y - 1; y++) for (int x = 1; x < X - 1; x++) {
        const size_t i = ((size_t)z * Y + y) * X + x; float *g = gz + 3 * i; g[0] = g[1] = g[2] = NAN;
        if (ph[i] >= PHASE_PERIOD) continue;
        const size_t sz = (size_t)Y * X, sy = X;
        const int a0 = ph[i - sz], a1 = ph[i + sz], b0 = ph[i - sy], b1 = ph[i + sy], c0 = ph[i - 1], c1 = ph[i + 1];
        if (a0 >= PHASE_PERIOD || a1 >= PHASE_PERIOD || b0 >= PHASE_PERIOD || b1 >= PHASE_PERIOD || c0 >= PHASE_PERIOD || c1 >= PHASE_PERIOD) continue;
        g[0] = sf_dturn(a0, a1) / 2; g[1] = sf_dturn(b0, b1) / 2; g[2] = sf_dturn(c0, c1) / 2;
    }
    /* walks: delta per papyrus voxel (dl 255 = none) */
    memset(dl, 255, N); memset(pp, 0, N);
    const int C0 = J->cell, wz0 = H - C0 > 1 ? H - C0 : 1, wy0 = wz0, wx0 = wz0;   /* walks: the interior and one cell around it (the cell means) */
    for (int z = wz0; z < D - wz0; z++) for (int y = wy0; y < Y - wy0; y++) for (int x = wx0; x < X - wx0; x++) {
        const size_t i = ((size_t)z * Y + y) * X + x; const float *g = gz + 3 * i;
        if (ct[i] < J->iso || ph[i] >= PHASE_PERIOD || !(g[0] == g[0])) continue;
        const float gl = sqrtf(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]); if (!(gl > 1e-3f)) continue;
        pp[i] = 1;
        const float nz = -g[0] / gl, ny = -g[1] / gl, nx = -g[2] / gl; int R = (int)(J->reach / gl); if (R > J->cap) R = J->cap;
        size_t last = i;
        for (int t = 1; t <= R; t++) {
            const int qz = (int)lrintf(z + t * nz), qy = (int)lrintf(y + t * ny), qx = (int)lrintf(x + t * nx);
            if (qz < 0 || qy < 0 || qx < 0 || qz >= D || qy >= Y || qx >= X) break;
            const size_t q = ((size_t)qz * Y + qy) * X + qx;
            if (!ct[q]) break;
            if (ct[q] < J->iso) { if (ph[last] < PHASE_PERIOD) dl[i] = ph[last]; break; }
            last = q;
        }
    }
    /* cell means of delta; output */
    const int C = J->cell, cz = (D + C - 1) / C, cy = (Y + C - 1) / C, cx = (X + C - 1) / C;
    float *cs = calloc((size_t)cz * cy * cx * 3, sizeof(float)); if (!cs) goto done;
    for (int z = 0; z < D; z++) for (int y = 0; y < Y; y++) for (int x = 0; x < X; x++) {
        const size_t i = ((size_t)z * Y + y) * X + x; if (dl[i] == 255) continue;
        float *c = cs + 3 * (((size_t)(z / C) * cy + y / C) * cx + x / C); c[0] += sf_cs[dl[i]]; c[1] += sf_sn[dl[i]]; c[2] += 1;
    }
    for (int z = H; z < H + n[0]; z++) for (int y = H; y < H + n[1]; y++) for (int x = H; x < H + n[2]; x++) {
        const size_t i = ((size_t)z * Y + y) * X + x, oi = ((size_t)(z - H) * n[1] + (y - H)) * n[2] + (x - H);
        int v = ph[i];
        if (v >= PHASE_PERIOD) { out[oi] = PHASE_UNKNOWN; continue; }
        if (J->nosnap) { out[oi] = (uint8_t)v; if (pp[i]) { st[0]++; st[1]++; st[5] += v < PHASE_PERIOD / 2; st[4] += v < PHASE_PERIOD / 2; } continue; }   /* --no-snap 1: the phase as read */
        int d = J->pervox ? dl[i] : 255, how = 0;
        if (d == 255) { const float *c = cs + 3 * (((size_t)(z / C) * cy + y / C) * cx + x / C);
                        if (c[2] >= J->mincell && c[0] * c[0] + c[1] * c[1] >= 0.25f * c[2] * c[2]) { d = sf_code(c[0], c[1]); how = 1; } else how = 2; }
        if (pp[i]) { st[0]++; st[how == 0 ? 1 : how == 1 ? 2 : 3]++; st[5] += v < PHASE_PERIOD / 2; }
        if (how == 2) { out[oi] = pp[i] ? PHASE_UNKNOWN : (uint8_t)v; continue; }
        const int w = ((v - d) % PHASE_PERIOD + PHASE_PERIOD) % PHASE_PERIOD; out[oi] = (uint8_t)w;
        if (pp[i]) st[4] += w < PHASE_PERIOD / 2;
    }
    /* boundary profiles: x papyrus recto, its inward neighbour verso (raw: ph; anchored: recomputed from out where inside) */
    for (int z = H; z < H + n[0]; z++) for (int y = H; y < H + n[1]; y++) for (int x = H; x < H + n[2]; x++) {
        const size_t i = ((size_t)z * Y + y) * X + x; if (!pp[i]) continue;
        const float *g = gz + 3 * i; const float gl = sqrtf(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
        const float nz = -g[0] / gl, ny = -g[1] / gl, nx = -g[2] / gl;
        const int qz = (int)lrintf(z + nz), qy = (int)lrintf(y + ny), qx = (int)lrintf(x + nx);
        const size_t q = ((size_t)qz * Y + qy) * X + qx;
        for (int k = 0; k < 2; k++) {   /* 0 raw, 1 face-anchored */
            int vx, vq;
            if (!k) { vx = ph[i]; vq = ph[q]; }
            else { if (qz < H || qy < H || qx < H || qz >= H + n[0] || qy >= H + n[1] || qx >= H + n[2]) continue;
                   vx = out[((size_t)(z - H) * n[1] + (y - H)) * n[2] + (x - H)]; vq = out[((size_t)(qz - H) * n[1] + (qy - H)) * n[2] + (qx - H)]; }
            if (vx >= PHASE_PERIOD || vq >= PHASE_PERIOD) continue;
            if (!(vx < PHASE_PERIOD / 2 && vq >= PHASE_PERIOD / 2)) continue;   /* recto here, verso one voxel inward */
            bs[2 * k]++; bs[2 * k + 1] += ct[q] < J->iso;
            for (int s = -4; s <= 4; s++) {
                const int pz = (int)lrintf(z - s * nz), py = (int)lrintf(y - s * ny), px = (int)lrintf(x - s * nx);   /* (s < 0: inward) */
                prof[k][s + 4] += (pz >= 0 && py >= 0 && px >= 0 && pz < D && py < Y && px < X) ? ct[((size_t)pz * Y + py) * X + px] : 0;
            }
        }
    }
    if (J->ag) {   /* side agreement on papyrus with another (anchored) phase raster */
        uint8_t *a = malloc((size_t)n[0] * n[1] * n[2]);
        if (a && !z3_read(J->ag, o, (const int64_t[3]){n[0], n[1], n[2]}, a, 1)) {
            for (int z = 0; z < n[0]; z++) for (int y = 0; y < n[1]; y++) for (int x = 0; x < n[2]; x++) {
                const size_t oi = ((size_t)z * n[1] + y) * n[2] + x, i = ((size_t)(z + H) * Y + y + H) * X + x + H;
                if (!pp[i] || out[oi] >= PHASE_PERIOD || a[oi] >= PHASE_PERIOD) continue;
                ags[0]++; ags[1] += (out[oi] < PHASE_PERIOD / 2) == (a[oi] < PHASE_PERIOD / 2);
            }
        }
        free(a);
    }
    free(cs); rc = 0;
done:
    free(ph); free(ct); free(dl); free(pp); free(gz);
    return rc;
}

typedef struct { sf_job *J; int64_t so[3]; uint8_t *buf; _Atomic int next; int nb; _Atomic int fail; long long st[8]; long long bs[4]; long long ags[2]; double prof[2][SF_PROF]; pthread_mutex_t mu; } sf_shard;
static void *sf_worker(void *arg) {
    sf_shard *T = arg; sf_job *J = T->J; const int B = J->B, S = J->S, nb1 = S / B;
    uint8_t *ob = malloc((size_t)B * B * B); if (!ob) { atomic_store(&T->fail, 1); return nullptr; }
    long long st[8] = {0}, bs[4] = {0}, ags[2] = {0}; double prof[2][SF_PROF] = {{0}};
    for (int b; (b = atomic_fetch_add(&T->next, 1)) < T->nb && !atomic_load(&T->fail);) {
        const int bz = b / (nb1 * nb1), by = (b / nb1) % nb1, bx = b % nb1;
        int64_t o[3] = {T->so[0] * S + (int64_t)bz * B, T->so[1] * S + (int64_t)by * B, T->so[2] * S + (int64_t)bx * B}; int n[3];
        int skip = 0;
        for (int d = 0; d < 3; d++) { n[d] = (int)(J->shape[d] - o[d] < B ? J->shape[d] - o[d] : B); if (n[d] <= 0) skip = 1;
                                       if (o[d] + n[d] <= J->bo[d] || o[d] >= J->bo[d] + J->bn[d]) skip = 1; }
        if (skip) continue;
        if (sf_block(J, o, n, ob, st, prof, bs, ags)) { atomic_store(&T->fail, 1); break; }
        for (int z = 0; z < n[0]; z++) for (int y = 0; y < n[1]; y++)
            memcpy(T->buf + ((size_t)(bz * B + z) * S + by * B + y) * S + bx * B, ob + ((size_t)z * n[1] + y) * n[2], n[2]);
    }
    pthread_mutex_lock(&T->mu);
    for (int k = 0; k < 8; k++) T->st[k] += st[k];
    for (int k = 0; k < 4; k++) T->bs[k] += bs[k];
    T->ags[0] += ags[0]; T->ags[1] += ags[1];
    for (int k = 0; k < 2; k++) for (int s = 0; s < SF_PROF; s++) T->prof[k][s] += prof[k][s];
    pthread_mutex_unlock(&T->mu);
    free(ob); return nullptr;
}

int cmd_side_face(int argc, char **argv) {
    if (argc < 8) { fprintf(stderr, "usage: ufsm side-face <phase-root> <phase-key> <ct-root> <ct-key> <out-group> <out-um> [--iso 64] [--phase-scale 1|2] [--reach 3] [--cap 48]\n"
                                    "       [--cell 16] [--per-voxel 0] [--min-cell 8] [--box z,y,x,nz,ny,nx] [--block 128] [--shard 512] [--threads 16] [--cache DIR] [--agree ROOT KEY] [--no-out 1] [--no-snap 1] [--interp linear|nearest]\n"); return 2; }
    for (int k = 0; k < PHASE_PERIOD; k++) { sf_cs[k] = (float)cos(2 * M_PI * k / PHASE_PERIOD); sf_sn[k] = (float)sin(2 * M_PI * k / PHASE_PERIOD); }
    sf_job J; memset(&J, 0, sizeof J);
    const char *cache = sf_opt(argc, argv, "--cache", nullptr);
    J.ph = z3_open(store_open(argv[2]), argv[3], cache); J.ct = z3_open(store_open(argv[4]), argv[5], cache);
    if (!J.ph || !J.ct) { fprintf(stderr, "side-face: %s\n", z3_error()); return 1; }
    J.pscale = atoi(sf_opt(argc, argv, "--phase-scale", "1")); J.iso = atoi(sf_opt(argc, argv, "--iso", "64")); J.reach = atof(sf_opt(argc, argv, "--reach", "3"));
    J.cap = atoi(sf_opt(argc, argv, "--cap", "48")); J.cell = atoi(sf_opt(argc, argv, "--cell", "16")); J.mincell = atoi(sf_opt(argc, argv, "--min-cell", "8")); J.pervox = atoi(sf_opt(argc, argv, "--per-voxel", "0"));
    J.nosnap = atoi(sf_opt(argc, argv, "--no-snap", "0")); J.nearest = !strcmp(sf_opt(argc, argv, "--interp", "linear"), "nearest");
    J.B = atoi(sf_opt(argc, argv, "--block", "128")); J.S = atoi(sf_opt(argc, argv, "--shard", "512")); J.H = J.cap + 2;
    if ((J.pscale != 1 && J.pscale != 2) || J.iso < 1 || J.cap < 2 || J.cell < 1 || J.B < 16 || J.S % J.B) { fprintf(stderr, "side-face: bad options (--phase-scale 1|2, --shard a multiple of --block)\n"); return 2; }
    const z3_meta *cm = z3_meta_of(J.ct), *pm = z3_meta_of(J.ph);
    for (int d = 0; d < 3; d++) { J.shape[d] = cm->shape[d]; J.bn[d] = cm->shape[d];
        if (J.pscale == 1 ? pm->shape[d] != cm->shape[d] : (pm->shape[d] < cm->shape[d] / 2 || pm->shape[d] > cm->shape[d] / 2 + 1)) {
            fprintf(stderr, "side-face: phase shape %lld vs CT %lld on axis %d (--phase-scale %d)\n", (long long)pm->shape[d], (long long)cm->shape[d], d, J.pscale); return 1; } }
    if (*sf_opt(argc, argv, "--box", "")) { long long v[6]; if (sscanf(sf_opt(argc, argv, "--box", ""), "%lld,%lld,%lld,%lld,%lld,%lld", v, v + 1, v + 2, v + 3, v + 4, v + 5) != 6) { fprintf(stderr, "side-face: --box z,y,x,nz,ny,nx\n"); return 2; }
                                           for (int d = 0; d < 3; d++) { J.bo[d] = v[d]; J.bn[d] = v[3 + d]; } }
    { const char *ar = nullptr, *ak = nullptr; for (int i = 1; i + 2 < argc; i++) if (!strcmp(argv[i], "--agree")) { ar = argv[i + 1]; ak = argv[i + 2]; }
      if (ar && !(J.ag = z3_open(store_open(ar), ak, cache))) { fprintf(stderr, "side-face: --agree: %s\n", z3_error()); return 1; } }
    const int nt = ufsm_threads(atoi(sf_opt(argc, argv, "--threads", "0")), 16), noout = atoi(sf_opt(argc, argv, "--no-out", "0"));
    z3w *w = nullptr;
    if (!noout) {
        char lv[32], od[1400], at[600]; const double um = atof(argv[7]); z3w_level_name(um, lv, sizeof lv); snprintf(od, sizeof od, "%s/%s", argv[6], lv);
        snprintf(at, sizeof at, "{\"ufsm\":{\"content\":\"face-anchored phase (side-face of %s/%s, CT %s, iso %d): 0 at each sheet's inner (recto) face\"}}", argv[2], argv[3], argv[5], J.iso);
        if (!(w = z3w_create(od, J.shape, J.S, 0, PHASE_UNKNOWN, at)) || z3w_write_group(argv[6], &um, 1, "ufsm-side-face", nullptr)) { fprintf(stderr, "side-face: %s\n", z3w_error()); return 1; }
    }
    const int S = J.S; const int64_t ns[3] = {(J.shape[0] + S - 1) / S, (J.shape[1] + S - 1) / S, (J.shape[2] + S - 1) / S};
    uint8_t *buf = malloc((size_t)S * S * S); if (!buf) return 1;
    long long st[8] = {0}, bs[4] = {0}, ags[2] = {0}; double prof[2][SF_PROF] = {{0}}; int nsh = 0;
    const int pf = J.pscale;
    for (int64_t sz = J.bo[0] / S; sz < ns[0] && sz * S < J.bo[0] + J.bn[0]; sz++)
    for (int64_t sy = J.bo[1] / S; sy < ns[1] && sy * S < J.bo[1] + J.bn[1]; sy++)
    for (int64_t sx = J.bo[2] / S; sx < ns[2] && sx * S < J.bo[2] + J.bn[2]; sx++) {
        int present = 0;   /* any phase shard under this output shard */
        for (int64_t a = sz * S / pf / pm->shard[0]; a <= ((sz + 1) * S - 1) / pf / pm->shard[0] && !present; a++)
        for (int64_t b = sy * S / pf / pm->shard[1]; b <= ((sy + 1) * S - 1) / pf / pm->shard[1] && !present; b++)
        for (int64_t c = sx * S / pf / pm->shard[2]; c <= ((sx + 1) * S - 1) / pf / pm->shard[2] && !present; c++) present = z3_shard_present(J.ph, a, b, c) != 0;
        if (!present) continue;
        memset(buf, PHASE_UNKNOWN, (size_t)S * S * S);
        sf_shard T; memset(&T, 0, sizeof T); T.J = &J; T.so[0] = sz; T.so[1] = sy; T.so[2] = sx; T.buf = buf; T.nb = (S / J.B) * (S / J.B) * (S / J.B); pthread_mutex_init(&T.mu, nullptr);
        pthread_t th[64]; const int n = nt > 64 ? 64 : nt;
        for (int t = 0; t < n; t++) pthread_create(&th[t], nullptr, sf_worker, &T);
        for (int t = 0; t < n; t++) pthread_join(th[t], nullptr);
        if (atomic_load(&T.fail)) { fprintf(stderr, "side-face: read failed: %s\n", z3_error()); return 1; }
        for (int k = 0; k < 8; k++) st[k] += T.st[k];
        for (int k = 0; k < 4; k++) bs[k] += T.bs[k];
        ags[0] += T.ags[0]; ags[1] += T.ags[1];
        for (int k = 0; k < 2; k++) for (int s = 0; s < SF_PROF; s++) prof[k][s] += T.prof[k][s];
        if (w && z3w_write_shard(w, sz, sy, sx, buf, nt)) { fprintf(stderr, "side-face: %s\n", z3w_error()); return 1; }
        nsh++;
        fprintf(stderr, "\rside-face: %d shards, papyrus %lld (face %.3f)   ", nsh, st[0], st[0] ? (double)st[1] / st[0] : 0);
    }
    fprintf(stderr, "\n");
    if (w) z3w_close(w);
    const double P = st[0] ? (double)st[0] : 1;
    printf("side-face: %d shards; papyrus with a known phase %lld: face found %.4f, cell delta %.4f, unknown %.4f; recto share raw %.4f, face-anchored %.4f\n",
           nsh, st[0], st[1] / P, st[2] / P, st[3] / P, st[5] / P, (st[1] + st[2]) ? (double)st[4] / (st[1] + st[2]) : 0);
    for (int k = 0; k < 2; k++) {
        printf("side-face: %s verso->recto boundaries on papyrus %lld, inward neighbour void %.4f; mean CT along the normal (-4 inward .. +4 outward):",
               k ? "face-anchored" : "raw          ", bs[2 * k], bs[2 * k] ? (double)bs[2 * k + 1] / bs[2 * k] : 0);
        for (int s = 0; s < SF_PROF; s++) printf(" %.1f", bs[2 * k] ? prof[k][s] / bs[2 * k] : 0);
        printf("\n");
    }
    if (J.ag) printf("side-face: side agreement on papyrus with %s: %lld voxels, %.4f\n", sf_opt(argc, argv, "--agree", ""), ags[0], ags[0] ? (double)ags[1] / ags[0] : 0);
    free(buf); z3_close(J.ph); z3_close(J.ct); if (J.ag) z3_close(J.ag);
    return 0;
}
