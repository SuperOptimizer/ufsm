/* Level-0 slab rings (stage S4): device memory and step time of the production net (7 levels 16..256, MX-fp4 activations and
   gradients, fp8 stem, MX-fp4 input from the CT channel and radial records, recompute 1, slab-local GroupNorm on the kernel path) in
   one mode: env UFSM_TEST_P (window, default 512), UFSM_TEST_S (slab, default 32), BENCH_MODE: plain (the unstreamed plain layout),
   stream (stream 2 on whole tensors, the whole head), ring (stream 2 in rings, the stream head, the input ring), BENCH_STEPS (3).
   Prints the dry build's bytes (UFSM_MEM_REPORT=1 adds the breakdown), the device use after the build and at the peak (free memory
   sampled after each op group), the rings' physical bytes and the step time (the steps after the first). */
#include "nn.h"
#include "unet.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static size_t g_minfree = (size_t)-1;
static void sample(void) { nn_sync(); const size_t f = nn_mem_free(); if (f < g_minfree) g_minfree = f; }
static int head_fwd(void *c, int z0, int z1, const void *lg) { (void)c; (void)z0; (void)z1; (void)lg; return 0; }
static int head_bwd(void *c, int z0, int z1, void *lg) {   /* a constant logit gradient (the step's cost, not its values) */
    const size_t *hw = c; nn_zero(lg, (size_t)2 * (z1 - z0) * *hw * 2); return 0;
}

int main(void) {
    const int P = getenv("UFSM_TEST_P") ? atoi(getenv("UFSM_TEST_P")) : 512, S = getenv("UFSM_TEST_S") ? atoi(getenv("UFSM_TEST_S")) : 32;
    const char *mode = getenv("BENCH_MODE") ? getenv("BENCH_MODE") : "ring";
    const int steps = getenv("BENCH_STEPS") ? atoi(getenv("BENCH_STEPS")) : 3;
    const int ring = !strcmp(mode, "ring"), stream = ring || !strcmp(mode, "stream");
    nn_init(0);
    const size_t free0 = nn_mem_free();
    nn_set_tf32(1); nn_set_act_bf16(1); nn_set_grad_bf16(1); nn_set_f16(1); nn_set_grad_scale(1024); nn_set_sr(1); nn_set_gn_stored(1);
    nn_set_prec(3); if (nn_set_prec_policy("all=fp4:fp4:fp4,enc0.c1=fp16")) return 2;
    setenv("UFSM_F4_WGRAD", "1", 1);
    unet_set_act_mx4(1); unet_set_grad_mx8(1); unet_set_grad_mx4(1); unet_set_input_prec(4); unet_set_recompute(1); unet_set_chunk_up(1); unet_set_lean(0);
    unet_set_share_enc_a1(0); unet_set_a0_share(0); unet_set_l0_3buf(0); unet_set_xin_offload(0); unet_set_l0_slab_kern(1);
    unet_set_l0_stream(stream ? 2 : 0); unet_set_l0_ring(ring);
    unet_cfg cfg = {7, {16, 16, 32, 64, 128, 192, 256}, 4, 2, 8, 0}; cfg.l0_norm = 4; cfg.l0_slab = S;
    const shape5 xs = {1, 4, P, P, P};
    const size_t S3 = (size_t)P * P * P, hw = (size_t)P * P;
    unet *u = unet_create(&cfg); unet_init(u, 77);
    /* the CT channel (fp16) and radial records with an axis, as the trainer's --gpu-input batch */
    void *ct = nn_malloc(S3 * 2); nn_zero(ct, S3 * 2);
    const size_t rb = NN_RAD_REC(P); uint8_t *h = calloc(1, rb); int32_t *hi = (int32_t *)h; float *hf = (float *)(h + 32);
    hi[0] = 0; hi[1] = 0; hi[2] = P; hi[3] = P; hi[4] = 1; hi[5] = 0 | 1 << 2 | 2 << 4;
    for (int z = 0; z < P; z++) { hf[z] = P / 2.f; hf[P + z] = P / 2.f; }
    void *rec = nn_malloc(rb); nn_h2d(rec, h, rb); free(h);
    unet_set_input_rad(u, rec);
    if (ring) unet_set_l0_head(u, head_fwd, head_bwd, (void *)&hw);
    void *gl = nullptr;   /* the whole fp16 logit gradient (no stream head) */
    if (!ring) { gl = nn_malloc(S3 * 2 * 2); nn_zero(gl, S3 * 2 * 2); }
    const size_t dry = unet_train_bytes(u, xs);
    const double nv = (double)S3;
    printf("P %d slab %d mode %s: dry build %.2f GB (%.1f B per voxel)\n", P, S, mode, dry / 1e9, dry / nv);
    fflush(stdout);
    if (getenv("BENCH_DRY")) return 0;
    unet_build(u, xs, 1);
    sample();
    const size_t built = free0 - nn_mem_free();
    double t = 0; int nt = 0;
    for (int k = 0; k < steps; k++) {
        nn_sync(); const double t0 = now_ms();
        nn_set_sr_step(100 + k);
        unet_forward_x(u, ct, xs, 1, 1); sample();
        unet_zero_grad(u);
        unet_backward_x(u, gl, 1); sample();
        nn_sync();
        if (k) { t += now_ms() - t0; nt++; }
    }
    const char *e = nn_check();
    printf("  device: %.2f GB after the build (incl. the CT channel %.2f GB, weights / optimizer / context), peak %.2f GB of %.2f GB free at start; rings %.2f GB; "
           "step %.0f ms (%.0f Mvox/s)%s%s\n",
           built / 1e9, S3 * 2 / 1e9, (free0 - g_minfree) / 1e9, free0 / 1e9, nn_ring_bytes() / 1e9, nt ? t / nt : 0, nt ? nv / (t / nt) / 1e3 : 0, e ? " " : "", e ? e : "");
    unet_free(u); nn_free(ct); nn_free(rec); nn_free(gl);
    return e != nullptr;
}
