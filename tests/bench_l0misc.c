/* Level-0 kernels outside the 16-channel convs of a 16-wide production net, at the trainer's sizes: dec[0]'s conv1 with the fused
   2x upsample (lp_f4up.cu, 16 + 16 -> 16 MX-fp4, the coarse and skip GN+SiLU, the output's GN statistics), the GPU-built network
   input (nn_input_rad, plain and warped records) and the loss (fp16 logits, statistics + fp16 gradient). Times per call and an FNV
   hash of every output (the bits a speed change must keep).
   usage: bench_l0misc [P = 512] [reps = 5] [what = all | f4u | rad | loss] */
#include "nn.h"
#include "nn_lp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t rs = 2463534242ull;
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)(rs >> 16); }
static float frand(void) { return (float)(rnd() & 0xffffff) * 0x1p-23f - 1.f; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static uint64_t fnv_dev(const void *p, size_t n) {
    unsigned char *h = malloc(n); nn_sync(); nn_d2h(h, p, n);
    uint64_t x = 1469598103934665603ull; for (size_t i = 0; i < n; i++) { x ^= h[i]; x *= 1099511628211ull; }
    free(h); return x;
}
static float *dev_floats(size_t n, float scale, float off) { float *h = malloc(n * 4); for (size_t i = 0; i < n; i++) h[i] = off + scale * frand(); float *d = nn_malloc(n * 4); nn_h2d(d, h, n * 4); free(h); return d; }
/* an MX-fp4 16-channel tensor of random nibbles with scales 2^-3 .. 2^0 (the data rows, then one scale byte per voxel) */
static void *mx4_rand(shape5 s) {
    const size_t b = nn_mx4_bytes(s), sc = (size_t)s.n * shape_spatial(s) * ((s.c + 15) / 16);
    unsigned char *h = malloc(b);
    for (size_t i = 0; i < b - sc; i += 4) { const uint32_t r = rnd() ^ (rnd() << 16); memcpy(h + i, &r, b - sc - i < 4 ? b - sc - i : 4); }
    for (size_t i = b - sc; i < b; i++) h[i] = (unsigned char)(124 + (rnd() & 3));
    void *d = nn_malloc(b); nn_h2d(d, h, b); free(h); nn_set_storage(d, b, 4); return d;
}
#define TIMEIT(label, reps, call) do { call; nn_sync(); double t0_ = now(); for (int r_ = 0; r_ < (reps); r_++) { call; } nn_sync(); \
    printf("  %-34s %8.3f ms\n", label, 1e3 * (now() - t0_) / (reps)); } while (0)

int main(int argc, char **argv) {
    const int P = argc > 1 ? atoi(argv[1]) : 512, reps = argc > 2 ? atoi(argv[2]) : 5;
    const char *what = argc > 3 ? argv[3] : "all";
    nn_init(getenv("UFSM_GPU") ? atoi(getenv("UFSM_GPU")) : 0);
    nn_set_tf32(1); nn_set_prec(3); nn_set_act_bf16(1); nn_set_grad_bf16(1); nn_set_f16(1); nn_set_grad_scale(1024);
    const size_t S = (size_t)P * P * P;
    printf("bench_l0misc P %d, %d reps\n", P, reps);
    if (!strcmp(what, "all") || !strcmp(what, "f4u")) {   /* dec[0]'s conv1: up part from the coarse dec[1].a2 (16 ch), skip enc[0].a2 (16 ch) */
        const int G = 4;
        const shape5 fs = {1, 32, P, P, P}, co = {1, 16, P / 2, P / 2, P / 2}, sk = {1, 16, P, P, P}, o16 = {1, 16, P, P, P};
        void *xc = mx4_rand(co), *xs = mx4_rand(sk), *y = nn_malloc(nn_mx4_bytes(o16)); nn_zero(y, nn_mx4_bytes(o16)); nn_set_storage(y, nn_mx4_bytes(o16), 4);
        float *gc = dev_floats(16, 0.3f, 1.f), *bc = dev_floats(16, 0.3f, 0.f), *mc = dev_floats(G, 0.05f, 0.f), *rc = dev_floats(G, 0.2f, 1.2f);
        float *gs = dev_floats(16, 0.3f, 1.f), *bs = dev_floats(16, 0.3f, 0.f), *ms = dev_floats(G, 0.05f, 0.f), *rsd = dev_floats(G, 0.2f, 1.2f);
        float *w = dev_floats((size_t)16 * 32 * 27, 0.05f, 0.f), *b = dev_floats(16, 0.05f, 0.f), *m2 = nn_malloc(G * 4), *r2 = nn_malloc(G * 4);
        nn_gn_t g1 = {gc, bc, mc, rc, G}, g2 = {gs, bs, ms, rsd, G};
        int rv = 0;
        TIMEIT("dec0 conv1 (fused up, f4u)", reps, rv |= nn_conv3d_fwd_x(xc, &g1, xs, &g2, 16, 1, fs, w, b, 16, 3, 1, (float *)y, G, 1e-5f, m2, r2));
        float hm[8], hr[8]; nn_d2h(hm, m2, G * 4); nn_d2h(hr, r2, G * 4);
        printf("    rv %d, y %016llx, mean %.9g %.9g, rstd %.9g %.9g\n", rv, (unsigned long long)fnv_dev(y, nn_mx4_bytes(o16)), hm[0], hm[1], hr[0], hr[1]);
        nn_free(xc); nn_free(xs); nn_free(y);
    }
    if (!strcmp(what, "all") || !strcmp(what, "rad")) {   /* the network input from the fp16 CT channel and the radial record (plain, then warped) */
        const shape5 xs = {1, 4, P, P, P};
        const size_t rb = NN_RAD_REC(P), yb = nn_mx_bytes(xs, 4);
        float *ctf = dev_floats(S, 2.f, 0.f); void *ct = nn_malloc(S * 2); nn_f32_to_h16(ctf, S, ct, 1.f); nn_free(ctf);
        uint8_t *rec = calloc(1, rb); int32_t hd[8] = {1000, 2000, P, P, 1, 2 | 0 << 2 | 1 << 4 | 1 << 9, 0, 0}; memcpy(rec, hd, sizeof hd);
        float *cyz = (float *)(rec + sizeof hd), *cxz = cyz + P;
        for (int z = 0; z < P; z++) { cyz[z] = 1000.f + 213.37f + 0.1f * z; cxz[z] = 2000.f + 119.11f - 0.05f * z; }
        void *drec = nn_malloc(rb), *y = nn_malloc(yb); nn_h2d(drec, rec, rb); nn_set_storage(y, yb, 4);
        TIMEIT("input_rad (plain record)", reps, nn_input_rad(ct, xs, drec, y));
        printf("    y %016llx\n", (unsigned long long)fnv_dev(y, yb));
        /* warped: record flag 1 (+2: affine), the per-axis tables, R, A, Ainv, scale (sample.c write_spatial's layout) */
        ((int32_t *)rec)[6] = 3;
        float *tab = cxz + P, *R = tab + 6 * P, *A = R + 9, *Ai = A + 9;
        for (int d = 0; d < 3; d++) for (int k = 0; k < P; k++) { tab[d * P + k] = 0.3f * frand(); tab[(3 + d) * P + k] = 0.02f * frand(); }
        const float rot[9] = {0.99f, -0.1f, 0.02f, 0.1f, 0.99f, -0.03f, -0.02f, 0.03f, 1.f}, af[9] = {1.02f, 0.01f, 0.f, 0.f, 0.98f, 0.02f, 0.01f, 0.f, 1.f};
        memcpy(R, rot, sizeof rot); memcpy(A, af, sizeof af); memcpy(Ai, af, sizeof af); R[27] = 1.05f;
        nn_h2d(drec, rec, rb);
        TIMEIT("input_rad (warped record)", reps, nn_input_rad(ct, xs, drec, y));
        printf("    y %016llx\n", (unsigned long long)fnv_dev(y, yb));
        free(rec); nn_free(ct); nn_free(drec); nn_free(y);
    }
    if (!strcmp(what, "all") || !strcmp(what, "loss")) {   /* the recto / verso loss: fp16 logits, unpacked targets and mask, fp16 gradient */
        const shape5 s = {1, 2, P, P, P};
        float *lgf = dev_floats(2 * S, 4.f, 0.f);
        uint8_t *ht = malloc(2 * S), *hm = malloc(S);
        for (size_t i = 0; i < S; i++) { const uint32_t r = rnd(); hm[i] = (r & 7) != 0; ht[i] = (r >> 3 & 1) ? 255 : 0; ht[S + i] = (r >> 3 & 1) ? 0 : (r >> 4 & 1) ? 255 : 0; }
        void *lg = nn_malloc(2 * S * 2), *gl = nn_malloc(2 * S * 2); uint8_t *t = nn_malloc(2 * S), *m = nn_malloc(S), *wt = nn_malloc(2);
        const uint8_t hw[2] = {1, 1};
        nn_f32_to_h16(lgf, 2 * S, lg, 1.f); nn_free(lgf); nn_h2d(t, ht, 2 * S); nn_h2d(m, hm, S); nn_h2d(wt, hw, 2); free(ht); free(hm);
        float *scr = nn_malloc(nn_loss_scratch(s));
        nn_set_logits_h16(1); nn_set_loss_grad_h16(1);
        TIMEIT("loss (stats + fp16 gradient)", reps, nn_loss_async_tol((const float *)lg, t, m, wt, s, 1.f, (float *)gl, scr, nullptr));
        float hs[16]; nn_d2h(hs, scr, sizeof hs);
        printf("    gl %016llx, stats %016llx (bce %.9g %.9g, dice %.9g %.9g)\n", (unsigned long long)fnv_dev(gl, 2 * S * 2), (unsigned long long)fnv_dev(scr, 10 * 4),
               hs[10], hs[11], hs[12], hs[13]);
        TIMEIT("loss (statistics only)", reps, nn_loss_async_tol((const float *)lg, t, m, wt, s, 1.f, nullptr, scr, nullptr));
        nn_set_logits_h16(0); nn_set_loss_grad_h16(0);
        nn_free(lg); nn_free(gl); nn_free(t); nn_free(m); nn_free(wt); nn_free(scr);
    }
    const char *e = nn_check();
    if (e) { printf("CUDA error: %s\n", e); return 1; }
    return 0;
}
