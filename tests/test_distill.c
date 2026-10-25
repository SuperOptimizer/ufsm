/* train --distill pieces:
   (1) sample_emb_coords at one voxel per cell equals the sampler's own pull (spatial_aug_pull + the cube symmetry, as write_spatial) for
       random warps (rotation, elastic, zoom, affine) and symmetries; at 8^3 voxels per cell without a warp, the symmetric cell centre;
   (2) sample_emb_window on a written int8 zarr group whose cell vectors encode their own position (slow sines): every head cell's target
       must point at the analytic vector of its source position (level scaling, cell step and centres, symmetry, warp), zeros outside the
       array, a second group fills the cells the first lacks;
   (3) nn_distill_loss: the loss sums and the gradient against central finite differences in double precision; voxels without a target
       give 0 gradient (GPU). */
#include "embz.h"
#include "nn.h"
#include "sample_emb.h"
#include "spatial_augment.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)
static uint64_t rng = 0x243F6A8885A308D3ull;
static uint32_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)(rng >> 11); }
static void rand_sym(int perm[3], int flip[3]) {
    perm[0] = 0; perm[1] = 1; perm[2] = 2;
    for (int i = 2; i > 0; i--) { const int j = (int)(rnd() % (unsigned)(i + 1)), t = perm[i]; perm[i] = perm[j]; perm[j] = t; }
    for (int d = 0; d < 3; d++) flip[d] = (int)(rnd() & 1);
}

enum { C = 8, STEP = 8, LE = 1, LW = 2 };   /* channels; cell step (embedding voxels); embedding level; window level (2 embedding voxels each) */
static const double FR = 2 * M_PI / 300;   /* the field's frequency (per embedding voxel) */
static void field(const double e[3], double v[C]) {   /* the unit vector at embedding-voxel position e (z, y, x) */
    for (int d = 0; d < 3; d++) { v[2 * d] = cos(FR * e[d] + d); v[2 * d + 1] = sin(FR * e[d] + d); }
    v[6] = v[7] = 0;
    for (int c = 0; c < C; c++) v[c] /= sqrt(3.0);
}
static int write_group(const char *g, const int64_t n[3], const int64_t skip_lo[3], const int64_t skip_hi[3]) {   /* cells inside skip: 0 */
    char p[1200]; mkdir(g, 0755);
    snprintf(p, sizeof p, "%s/zarr.json", g); FILE *f = fopen(p, "w"); if (!f) return 1;
    fputs("{\"zarr_format\":3,\"node_type\":\"group\",\"attributes\":{}}", f); fclose(f);
    snprintf(p, sizeof p, "%s/pca", g);
    zw w; char attrs[256]; snprintf(attrs, sizeof attrs, "{\"step_zyx\":[%d,%d,%d],\"value_scale\":0.007874015748031496}", STEP, STEP, STEP);
    if (zw_create(&w, p, n, C, "int8", 1, 32, 16, attrs)) return 1;
    const int64_t m[3] = {(n[0] + 15) / 16 * 16, (n[1] + 15) / 16 * 16, (n[2] + 15) / 16 * 16};   /* (zw_put: whole chunks) */
    int8_t *buf = calloc((size_t)m[0] * m[1] * m[2] * C, 1);
    for (int64_t z = 0; z < n[0]; z++) for (int64_t y = 0; y < n[1]; y++) for (int64_t x = 0; x < n[2]; x++) {
        const int64_t q[3] = {z, y, x}; int in = 1;
        for (int d = 0; d < 3; d++) in &= q[d] >= skip_lo[d] && q[d] < skip_hi[d];
        if (in) continue;
        double e[3], v[C]; for (int d = 0; d < 3; d++) e[d] = q[d] * STEP + .5 * (STEP - 1);
        field(e, v);
        for (int c = 0; c < C; c++) buf[(((size_t)z * m[1] + y) * m[2] + x) * C + c] = (int8_t)lround(127 * v[c]);
    }
    const int rc = zw_put(&w, 0, 0, 0, (int)m[0], (int)m[1], (int)m[2], buf) || zw_flush(&w);
    zw_free(&w); free(buf);
    return rc;
}

int main(void) {
    /* (1) coordinates */
    {
        const int P = 48; double worst = 0;
        float *V = malloc((size_t)P * P * P * 3 * sizeof *V), table[6 * 48], src[3], jac[3];
        for (int t = 0; t < 12; t++) {
            int perm[3], flip[3]; rand_sym(perm, flip);
            spatial_aug a; spatial_aug_make(&a, 1000 + t, P, 25, 1, 1, 1);
            if (t & 1) spatial_aug_zoom(&a, .8 + .05 * (t % 5));
            if (t & 2) spatial_aug_affine(&a, 77 + t, .1, .06);
            spatial_aug_tables(&a, table);
            sample_emb_coords(P, P, 1, perm, flip, &a, V);
            for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) for (int x = 0; x < P; x++) {
                spatial_aug_pull(&a, table, z, y, x, src, jac);
                float v[3]; for (int d = 0; d < 3; d++) v[perm[d]] = flip[d] ? P - 1 - src[d] : src[d];
                const float *o = V + 3 * (((size_t)z * P + y) * P + x);
                for (int d = 0; d < 3; d++) if (fabs(o[d] - v[d]) > worst) worst = fabs(o[d] - v[d]);
            }
        }
        printf("coords vs the sampler's pull: worst %.2g voxels\n", worst);
        CHECK(worst < 2e-3, "sample_emb_coords differs from spatial_aug_pull + symmetry by %g", worst);
        const int Pz = 32, Q = 64, dv = 8; int perm[3] = {2, 0, 1}, flip[3] = {1, 0, 1};
        sample_emb_coords(Pz, Q, dv, perm, flip, nullptr, V);   /* (perm moves z: a test of the formula, the sampler keeps z on slabs) */
        const int n[3] = {Pz, Q, Q}; double w2 = 0;
        for (int gz = 0; gz < Pz / dv; gz++) for (int gy = 0; gy < Q / dv; gy++) for (int gx = 0; gx < Q / dv; gx++) {
            const int g[3] = {gz, gy, gx}; double v[3];
            for (int d = 0; d < 3; d++) { const double u = dv * g[d] + .5 * (dv - 1); v[perm[d]] = flip[d] ? n[d] - 1 - u : u; }
            const float *o = V + 3 * (((size_t)gz * (Q / dv) + gy) * (Q / dv) + gx);
            for (int d = 0; d < 3; d++) if (fabs(o[d] - v[d]) > w2) w2 = fabs(o[d] - v[d]);
        }
        CHECK(w2 < 1e-5, "cell centres without a warp: %g", w2);
        free(V);
    }
    /* (2) windows from written groups */
    {
        char dir[] = "/tmp/ufsm_test_distill_XXXXXX"; if (!mkdtemp(dir)) { perror("mkdtemp"); return 1; }
        char g1[1100], g2[1100]; snprintf(g1, sizeof g1, "%s/a", dir); snprintf(g2, sizeof g2, "%s/b", dir);
        const int64_t n[3] = {40, 44, 48}, none[3] = {0, 0, 0}, hole_lo[3] = {0, 0, 0}, hole_hi[3] = {40, 44, 12};   /* a lacks x cells 0..11, b has them */
        if (write_group(g1, n, hole_lo, hole_hi) || write_group(g2, n, none, none)) { fprintf(stderr, "cannot write the groups\n"); return 1; }
        source s; memset(&s, 0, sizeof s); s.name = "fixture"; s.nemb = 1; s.emb_grp[0] = g1; s.emb_name = "pca"; s.emb_level = LE;
        const int P = 64, dv = 8, G = P / dv; const size_t ng = (size_t)G * G * G;
        int8_t *out = malloc(ng * C); float *V = malloc(ng * 3 * sizeof *V);
        const double f = ldexp(1.0, LW - LE);
        double worst = 1; size_t nset = 0, nzero = 0, nbad0 = 0;
        for (int t = 0; t < 10; t++) {
            int perm[3], flip[3]; rand_sym(perm, flip);
            spatial_aug a; spatial_aug_make(&a, 500 + t, P, 20, 1, 1, 1);
            if (t & 1) spatial_aug_affine(&a, 9 + t, .1, .06);
            const spatial_aug *sa = t % 3 ? &a : nullptr;
            int64_t o[3] = {10 + t, 20 + 3 * t, 30 + 2 * t};
            if (t == 7) { s.nemb = 2; s.emb_grp[1] = g2; o[2] = 2; }   /* x below the array's start, a's hole, b behind a */
            if (sample_emb_window(&s, C, dv, LW, o, P, P, perm, flip, sa, out)) { CHECK(0, "sample_emb_window failed"); break; }
            sample_emb_coords(P, P, dv, perm, flip, sa, V);
            for (size_t q = 0; q < ng; q++) {
                double e[3], v[C], dot = 0, l2 = 0; int in = 1;
                for (int d = 0; d < 3; d++) {
                    e[d] = (o[d] + V[3 * q + d] + .5) * f - .5;
                    const double c = (e[d] - .5 * (STEP - 1)) / STEP;
                    in &= c >= 0 && c <= n[d] - 1;
                    if (s.nemb == 1 && d == 2) in &= c >= hole_hi[2];
                }
                int any = 0; for (int c = 0; c < C; c++) any |= out[(size_t)c * ng + q];
                if (!any) { nzero++; if (in) nbad0++; continue; }
                if (!in) continue;   /* (past the array's last cell centres: the nearer corners alone) */
                field(e, v);
                for (int c = 0; c < C; c++) { dot += v[c] * out[(size_t)c * ng + q]; l2 += (double)out[(size_t)c * ng + q] * out[(size_t)c * ng + q]; }
                const double cs = dot / sqrt(l2);
                if (cs < worst) worst = cs;
                nset++;
            }
            s.nemb = 1;
        }
        printf("windows: %zu cells with targets (worst cos %.5f), %zu without (%zu of them inside the array)\n", nset, worst, nzero, nbad0);
        CHECK(nset > 1000 && worst > 0.9995, "targets off their source position: worst cos %g over %zu", worst, nset);
        CHECK(nzero > 0 && nbad0 == 0, "missing targets inside the array: %zu (of %zu empty)", nbad0, nzero);
        free(out); free(V);
        char cmd[1200]; snprintf(cmd, sizeof cmd, "rm -rf %s", dir); if (system(cmd)) {}
    }
    /* (3) the loss on the GPU */
    if (nn_init(getenv("UFSM_GPU") ? atoi(getenv("UFSM_GPU")) : 0)) { printf("no cuda: the loss check skipped\n"); return fails ? 1 : 0; }
    {
        const int N = 2, Cc = 5; const size_t S = 37, M = (size_t)N * Cc * S; const float w = 0.7f;
        float *y = malloc(M * 4), *g = malloc(M * 4); int8_t *t = malloc(M);
        for (size_t i = 0; i < M; i++) { y[i] = (float)((int)(rnd() % 2001) - 1000) / 500.f; t[i] = (int8_t)((int)(rnd() % 255) - 127); }
        for (int b = 0; b < N; b++) for (size_t i = 0; i < S; i += 5) for (int c = 0; c < Cc; c++) t[((size_t)b * Cc + c) * S + i] = 0;   /* no target */
        float *dy = nn_malloc(M * 4), *dg = nn_malloc(M * 4); int8_t *dt = nn_malloc(M); double *dsum = nn_malloc(16), hs[2];
        nn_h2d(dy, y, M * 4); nn_h2d(dt, t, M);
        nn_distill_loss(dy, dt, N, Cc, S, w, dg, dsum);
        nn_d2h(g, dg, M * 4); nn_d2h(hs, dsum, 16);
        const char *e = nn_check(); CHECK(!e, "cuda: %s", e ? e : "");
        double L = 0; size_t cnt = 0; double *yd = malloc(M * sizeof *yd); for (size_t i = 0; i < M; i++) yd[i] = y[i];
        #define LOSS(Y, OUT, CNT) do { double s_ = 0; size_t k_ = 0; for (int b = 0; b < N; b++) for (size_t i = 0; i < S; i++) { double yy = 0, tt = 0, yt = 0; \
            for (int c = 0; c < Cc; c++) { const size_t j = ((size_t)b * Cc + c) * S + i; yy += Y[j] * Y[j]; tt += (double)t[j] * t[j]; yt += Y[j] * t[j]; } \
            if (tt > 0) { s_ += 1 - yt / (sqrt(yy) * sqrt(tt)); k_++; } } OUT = s_; CNT = k_; } while (0)
        LOSS(yd, L, cnt);
        CHECK(cnt == (size_t)hs[1] && fabs(L - hs[0]) < 1e-3 * (1 + L), "loss sums %g / %g vs %g / %zu", hs[0], hs[1], L, cnt);
        double worst = 0, gmax = 0;
        for (size_t i = 0; i < M; i++) if (fabs(g[i]) > gmax) gmax = fabs(g[i]);
        for (size_t i = 0; i < M; i++) {
            const double h = 1e-5, o = yd[i]; double lp, lm; size_t k; (void)k;
            yd[i] = o + h; LOSS(yd, lp, k); yd[i] = o - h; LOSS(yd, lm, k); yd[i] = o;
            const double fd = w * (lp - lm) / (2 * h) / (double)cnt, err = fabs(fd - g[i]) / gmax;
            if (err > worst) worst = err;
        }
        int zero_ok = 1;
        for (int b = 0; b < N; b++) for (size_t i = 0; i < S; i += 5) for (int c = 0; c < Cc; c++) zero_ok &= g[((size_t)b * Cc + c) * S + i] == 0.f;
        printf("distill loss: %g over %zu voxels; gradient vs finite differences worst rel %.2g\n", L / cnt, cnt, worst);
        CHECK(worst < 1e-3, "distill gradient: worst rel err %g", worst);
        CHECK(zero_ok, "gradient at voxels without a target");
        nn_free(dy); nn_free(dg); nn_free(dt); nn_free(dsum); free(y); free(g); free(t); free(yd);
    }
    if (fails) { fprintf(stderr, "%d failures\n", fails); return 1; }
    printf("test_distill: ok\n");
    return 0;
}
