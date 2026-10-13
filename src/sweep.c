/* predict --cond-sweep: conditioned inference over a box (src/sweep.h) */
#include "sweep.h"
#include "nn.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

static double sw_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static float sig2(float l) { return 2.f / (1.f + expf(-l)) - 1.f; }   /* the soft-BCE heads' value in [-1, 1] */
static uint8_t u8c(double v) { v = 127.5 + 127.5 * v + 0.5; return (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v); }

/* the CT window at level coordinates o (zero outside the volume) */
static int read_window(z3 *ct, const int64_t o[3], int W, uint8_t *buf, int nthreads) {
    const z3_meta *m = z3_meta_of(ct); int64_t ro[3], rn[3];
    memset(buf, 0, (size_t)W * W * W);
    for (int d = 0; d < 3; d++) { ro[d] = o[d] < 0 ? 0 : o[d]; const int64_t e = o[d] + W < m->shape[d] ? o[d] + W : m->shape[d]; rn[d] = e - ro[d]; if (rn[d] <= 0) return 0; }
    uint8_t *tmp = malloc((size_t)rn[0] * rn[1] * rn[2]);
    if (!tmp || z3_read(ct, ro, rn, tmp, nthreads)) { free(tmp); return -1; }
    for (int64_t a = 0; a < rn[0]; a++) for (int64_t b = 0; b < rn[1]; b++)
        memcpy(buf + ((size_t)(a + ro[0] - o[0]) * W + (b + ro[1] - o[1])) * W + (ro[2] - o[2]), tmp + ((size_t)a * rn[1] + b) * rn[2], (size_t)rn[2]);
    free(tmp);
    return 0;
}

size_t sweep_unwrap(const uint8_t *c14, const uint8_t *p1, const uint8_t *done, const int64_t bn[3], const float *cy, const float *cx, int dir, size_t seed, float *U) {
    const size_t N = (size_t)bn[0] * bn[1] * bn[2], P2 = (size_t)bn[1] * bn[2];
    float *q = malloc(N * sizeof *q); uint8_t *cf = malloc(N);
    if (!q || !cf) { free(q); free(cf); return 0; }
    for (size_t g = 0; g < N; g++) {   /* q mod 14: the 14-turn phasor picks the turn, phi + dir theta / 2 pi the fraction */
        U[g] = NAN;
        if (done[g] != 1) { q[g] = NAN; cf[g] = 0; continue; }
        const int64_t z = (int64_t)(g / P2), y = (int64_t)((g / bn[2]) % bn[1]), x = (int64_t)(g % bn[2]);
        const double th = atan2(y - cy[z], x - cx[z]) / (2 * M_PI);
        const double pc = (p1[g] - 127.5) / 127.5, ps = (p1[N + g] - 127.5) / 127.5, cc = (c14[g] - 127.5) / 127.5, cs = (c14[N + g] - 127.5) / 127.5;
        double f = atan2(ps, pc) / (2 * M_PI) + dir * th; f -= floor(f);
        double qc = atan2(cs, cc) / (2 * M_PI) * 14; if (qc < 0) qc += 14;
        double d = qc - f; d -= 14 * round(d / 14);
        double qq = f + round(d); qq -= 14 * floor(qq / 14);
        q[g] = (float)qq;
        const double conf = fmin(hypot(cc, cs), hypot(pc, ps));
        cf[g] = (uint8_t)(1 + (int)(fmin(conf, 1.0) * 14.99));
    }
    /* quality-guided flood (16 buckets by confidence, highest first): a neighbour gets U + its q difference wrapped to (-7, 7] */
    uint32_t *bk[16] = {0}; size_t bl[16] = {0}, bc[16] = {0}; int top = 0; size_t reached = 0;
#define PUSH(i) do { const int b_ = cf[i]; if (bl[b_] == bc[b_]) { bc[b_] = bc[b_] ? 2 * bc[b_] : 4096; bk[b_] = realloc(bk[b_], bc[b_] * sizeof(uint32_t)); } bk[b_][bl[b_]++] = (uint32_t)(i); if (b_ > top) top = b_; } while (0)
    if (seed < N && done[seed] == 1 && N < 0xffffffffu) { U[seed] = q[seed]; PUSH(seed); reached = 1; }
    while (top > 0) {
        if (!bl[top]) { top--; continue; }
        const size_t g = bk[top][--bl[top]];
        const int64_t z = (int64_t)(g / P2), y = (int64_t)((g / bn[2]) % bn[1]), x = (int64_t)(g % bn[2]);
        const int64_t nb[6][3] = {{z - 1, y, x}, {z + 1, y, x}, {z, y - 1, x}, {z, y + 1, x}, {z, y, x - 1}, {z, y, x + 1}};
        for (int k = 0; k < 6; k++) {
            if (nb[k][0] < 0 || nb[k][1] < 0 || nb[k][2] < 0 || nb[k][0] >= bn[0] || nb[k][1] >= bn[1] || nb[k][2] >= bn[2]) continue;
            const size_t h = ((size_t)nb[k][0] * bn[1] + nb[k][1]) * bn[2] + nb[k][2];
            if (done[h] != 1 || U[h] == U[h]) continue;
            double d = (double)q[h] - q[g]; d -= 14 * round(d / 14);
            U[h] = (float)(U[g] + d); PUSH(h); reached++;
        }
    }
#undef PUSH
    for (int b = 0; b < 16; b++) free(bk[b]);
    free(q); free(cf);
    return reached;
}

int cond_sweep(const sweep_cfg *c) {
    const int W = c->W, halo = c->halo, stride = W - 2 * halo;
    if (stride <= 0) { fprintf(stderr, "cond-sweep: window %d too small for halo %d\n", W, halo); return 2; }
    const size_t w3 = (size_t)W * W * W, N = (size_t)c->bn[0] * c->bn[1] * c->bn[2], xb = c->h16 ? 2 : 4;
    uint8_t *ctd = nn_malloc(w3), *dc14 = nn_malloc(2 * N), *dp1 = nn_malloc(2 * N), *drec = nn_malloc(N), *ddone = nn_malloc(N);
    void *xd = nn_malloc(7 * w3 * xb), *stsc = nn_malloc(64); float *dyd = nn_malloc(2 * (size_t)W * sizeof(float));
    if (!ctd || !dc14 || !dp1 || !drec || !ddone || !xd || !stsc || !dyd) { fprintf(stderr, "cond-sweep: out of device memory (box canvas %.2f GB)\n", 6.0 * N / 1e9); return 1; }
    nn_zero(ddone, N); nn_zero(dc14, 2 * N); nn_zero(dp1, 2 * N); nn_zero(drec, N);
    const shape5 xs = {1, 7, W, W, W};
    unet_build(c->u, xs, 0);
    uint8_t *buf = malloc(w3); float *dyo = malloc(2 * (size_t)W * sizeof(float)), *lh = malloc(3 * w3 * sizeof(float));
    int64_t nt[3]; for (int d = 0; d < 3; d++) nt[d] = (c->bn[d] + stride - 1) / stride;
    const long ntiles = (long)(nt[0] * nt[1] * nt[2]);
    int seeded = 0; size_t seed = 0; long nair = 0, nwin = 0; const double t0 = sw_now();
    for (long ti = 0; ti < ntiles; ti++) {
        const long tj = c->reverse ? ntiles - 1 - ti : ti;
        const int64_t j[3] = {tj / (nt[1] * nt[2]), (tj / nt[2]) % nt[1], tj % nt[2]};
        const int64_t t[3] = {-halo + j[0] * stride, -halo + j[1] * stride, -halo + j[2] * stride}, o[3] = {c->bo[0] + t[0], c->bo[1] + t[1], c->bo[2] + t[2]};
        if (read_window(c->ct, o, W, buf, c->nthreads)) { fprintf(stderr, "cond-sweep: CT read failed: %s\n", z3_error()); return 1; }
        size_t nz; double sum, sq;
        nn_h2d(ctd, buf, w3); nn_pred_stats(ctd, w3, stsc, &nz, &sum, &sq);
        if (!nz) { nair++; continue; }
        const double mean = sum / (double)w3, var = sq / (double)w3 - mean * mean, sd = sqrt(var > 0 ? var : 0) + 1e-3;
        for (int z = 0; z < W; z++) { double cy = 0, cx = 0; axis_at(c->ax, (double)(o[0] + z) * c->scale, &cy, &cx); dyo[z] = (float)((double)o[1] - cy / c->scale); dyo[W + z] = (float)((double)o[2] - cx / c->scale); }
        nn_h2d(dyd, dyo, 2 * (size_t)W * sizeof(float));
        const float *lg = nullptr;
        for (int pass = 0; pass < 2; pass++) {
            nn_pred_input(ctd, W, (float)mean, (float)(1.0 / sd), dyd, dyd + W, c->axis_flag, xd, c->h16);
            nn_cond_gather(dc14, ddone, ctd, W, (int)t[0], (int)t[1], (int)t[2], (int)c->bn[0], (int)c->bn[1], (int)c->bn[2], N, xd, c->h16);
            lg = unet_forward_x(c->u, xd, xs, 0, c->h16);
            if (seeded) break;
            /* the first window with material: a 4^3 seed cube at its most confident papyrus voxel, the 14-turn phasor from its 1-turn phase
               plus dir theta / 2 pi (unwrapped about the centre: gauge 0), then the window again, conditioned on the seed */
            nn_sync(); nn_d2h(lh, lg, w3 * sizeof(float)); nn_d2h(lh + w3, lg + 2 * w3, 2 * w3 * sizeof(float));
            double best = -1; int bz = -1, by = -1, bx = -1;
            for (int z = halo; z < W - halo; z++) for (int y = halo; y < W - halo; y++) for (int x = halo; x < W - halo; x++) {
                if (t[0] + z < 0 || t[1] + y < 0 || t[2] + x < 0 || t[0] + z >= c->bn[0] || t[1] + y >= c->bn[1] || t[2] + x >= c->bn[2]) continue;
                const size_t k = ((size_t)z * W + y) * W + x; if (!buf[k]) continue;
                const double r = hypot(sig2(lh[w3 + k]), sig2(lh[2 * w3 + k]));
                if (r > best) { best = r; bz = z; by = y; bx = x; }
            }
            if (bz < 0) break;   /* nothing of the box in this window's interior */
            double fc = 0; int nseed = 0;
            for (int dz = -2; dz < 2; dz++) for (int dy = -2; dy < 2; dy++) for (int dx = -2; dx < 2; dx++) {
                const int z = bz + dz, y = by + dy, x = bx + dx;
                if (z < halo || y < halo || x < halo || z >= W - halo || y >= W - halo || x >= W - halo) continue;
                if (t[0] + z < 0 || t[1] + y < 0 || t[2] + x < 0 || t[0] + z >= c->bn[0] || t[1] + y >= c->bn[1] || t[2] + x >= c->bn[2]) continue;
                const size_t k = ((size_t)z * W + y) * W + x; if (!buf[k]) continue;
                const double pc = sig2(lh[w3 + k]), ps = sig2(lh[2 * w3 + k]);
                double f = atan2(ps, pc) / (2 * M_PI) + c->dir * atan2(dyo[z] + y, dyo[W + z] + x) / (2 * M_PI); f -= floor(f);
                if (!nseed) fc = f;   /* (the loop's first voxel sets the gauge) */
                double d = f - fc; d -= round(d); const double q = fc + d;
                const size_t g = ((size_t)(t[0] + z) * c->bn[1] + (t[1] + y)) * c->bn[2] + (t[2] + x);
                const uint8_t v[6] = {u8c(cos(2 * M_PI * q / 14)), u8c(sin(2 * M_PI * q / 14)), u8c(pc), u8c(ps), u8c(sig2(lh[k])), 1};
                nn_h2d(dc14 + g, v, 1); nn_h2d(dc14 + N + g, v + 1, 1); nn_h2d(dp1 + g, v + 2, 1); nn_h2d(dp1 + N + g, v + 3, 1); nn_h2d(drec + g, v + 4, 1); nn_h2d(ddone + g, v + 5, 1);
                if (z == bz && y == by && x == bx) seed = g;
                nseed++;
            }
            seeded = 1;
            fprintf(stderr, "cond-sweep: seed of %d voxels at box (%lld, %lld, %lld), phase confidence %.2f\n", nseed, (long long)(t[0] + bz), (long long)(t[1] + by), (long long)(t[2] + bx), best);
        }
        nn_cond_place(lg, ctd, W, halo, (int)t[0], (int)t[1], (int)t[2], (int)c->bn[0], (int)c->bn[1], (int)c->bn[2], N, c->ch14, dc14, dp1, drec, ddone);
        nwin++;
        if (nwin % 25 == 0) fprintf(stderr, "\rcond-sweep: window %ld / %ld (%ld air)  %.0fs   ", ti + 1, ntiles, nair, sw_now() - t0);
    }
    const char *e = nn_check(); if (e) { fprintf(stderr, "\ncond-sweep: %s\n", e); return 1; }
    fprintf(stderr, "\rcond-sweep: %ld windows (%ld air) in %.0fs%20s\n", nwin, nair, sw_now() - t0, "");
    uint8_t *hc14 = malloc(2 * N), *hp1 = malloc(2 * N), *hdone = malloc(N); float *U = malloc(N * sizeof(float));
    float *cy = malloc((size_t)c->bn[0] * sizeof(float)), *cx = malloc((size_t)c->bn[0] * sizeof(float));
    if (!hc14 || !hp1 || !hdone || !U || !cy || !cx) { fprintf(stderr, "cond-sweep: out of host memory\n"); return 1; }
    nn_sync(); nn_d2h(hc14, dc14, 2 * N); nn_d2h(hp1, dp1, 2 * N); nn_d2h(hdone, ddone, N);
    for (int64_t z = 0; z < c->bn[0]; z++) { double y = 0, x = 0; axis_at(c->ax, (double)(c->bo[0] + z) * c->scale, &y, &x); cy[z] = (float)(y / c->scale - c->bo[1]); cx[z] = (float)(x / c->scale - c->bo[2]); }
    size_t npap = 0; for (size_t g = 0; g < N; g++) npap += hdone[g] == 1;
    if (c->raw) {   /* diagnostic: the 14-turn head's own value (turns, [0, 14)) per voxel, no snapping or flood, as <out>-raw/U.f32 for wind-score */
        char rd[1400], rp[1500]; snprintf(rd, sizeof rd, "%s-raw", c->out);
        if (mkdir(rd, 0755) && errno != EEXIST) { fprintf(stderr, "cond-sweep: cannot create %s\n", rd); return 1; }
        for (size_t g = 0; g < N; g++) {
            if (hdone[g] != 1) { U[g] = NAN; continue; }
            double qc = atan2((hc14[N + g] - 127.5) / 127.5, (hc14[g] - 127.5) / 127.5) / (2 * M_PI) * 14; if (qc < 0) qc += 14;
            U[g] = (float)qc;
        }
        snprintf(rp, sizeof rp, "%s/U.f32", rd); FILE *rf = fopen(rp, "wb");
        if (!rf || fwrite(U, sizeof(float), N, rf) != N || fclose(rf)) { fprintf(stderr, "cond-sweep: cannot write %s\n", rp); return 1; }
        snprintf(rp, sizeof rp, "%s/sweep.json", rd); rf = fopen(rp, "w"); if (!rf) return 1;
        fprintf(rf, "{\"content\":\"the 14-turn head's value (turns) per voxel, raw\",\"shape\":[%lld,%lld,%lld],\"origin_zyx\":[%lld,%lld,%lld],\"um\":%.9g,\"scale\":%.9g,\"axis\":\"%s\"}\n",
                (long long)c->bn[0], (long long)c->bn[1], (long long)c->bn[2], (long long)c->bo[0], (long long)c->bo[1], (long long)c->bo[2], c->um, c->scale, c->axis_path ? c->axis_path : "");
        fclose(rf);
    }
    const size_t reached = seeded ? sweep_unwrap(hc14, hp1, hdone, c->bn, cy, cx, c->dir, seed, U) : 0;
    if (!seeded) for (size_t g = 0; g < N; g++) U[g] = NAN;
    fprintf(stderr, "cond-sweep: U reached %zu of %zu papyrus voxels (%.4f) from the seed\n", reached, npap, npap ? (double)reached / npap : 0);
    if (mkdir(c->out, 0755) && errno != EEXIST) { fprintf(stderr, "cond-sweep: cannot create %s\n", c->out); return 1; }
    char p[1400]; snprintf(p, sizeof p, "%s/U.f32", c->out);
    FILE *f = fopen(p, "wb"); if (!f || fwrite(U, sizeof(float), N, f) != N || fclose(f)) { fprintf(stderr, "cond-sweep: cannot write %s\n", p); return 1; }
    snprintf(p, sizeof p, "%s/sweep.json", c->out); f = fopen(p, "w");
    if (!f) { fprintf(stderr, "cond-sweep: cannot write %s\n", p); return 1; }
    fprintf(f, "{\"content\":\"U (turns; sheets where U - dir theta / 2 pi is an integer), float32 [z][y][x] in U.f32, NaN where not reached\",\"shape\":[%lld,%lld,%lld],"
               "\"origin_zyx\":[%lld,%lld,%lld],\"um\":%.9g,\"scale\":%.9g,\"axis\":\"%s\",\"dir\":%d,\"seed\":%zu,\"window\":%d,\"halo\":%d,\"reverse\":%d,\"checkpoint\":\"%s\","
               "\"papyrus\":%zu,\"reached\":%zu}\n", (long long)c->bn[0], (long long)c->bn[1], (long long)c->bn[2], (long long)c->bo[0], (long long)c->bo[1], (long long)c->bo[2],
            c->um, c->scale, c->axis_path ? c->axis_path : "", c->dir, seed, W, halo, c->reverse, c->ckpt, npap, reached);
    fclose(f);
    free(hc14); free(hp1); free(hdone); free(U); free(cy); free(cx); free(buf); free(dyo); free(lh);
    nn_free(ctd); nn_free(dc14); nn_free(dp1); nn_free(drec); nn_free(ddone); nn_free(xd); nn_free(stsc); nn_free(dyd);
    fprintf(stderr, "cond-sweep: wrote %s/U.f32 and sweep.json\n", c->out);
    return 0;
}
