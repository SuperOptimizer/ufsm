/* Per-layer GPU time of the inference forward (event profiler), for one precision setting.
   usage: prof_infer [P] [B] [iters]   env: UFSM_PREC_POLICY / UFSM_PREC, UFSM_F16, UFSM_DOWN_NORM, UFSM_ACT_MX8 / UFSM_ACT_MX4,
   UFSM_XIN_MX / UFSM_XIN_PREC, UFSM_GN_STORED, UFSM_SR, UFSM_BF16, UFSM_WIDTHS (e.g. 32,64,96,128,160,192), UFSM_COUT.
   Prints, per conv slot, the minimum over iterations of its forward time, the category totals (min over iterations),
   and the end-to-end forward time (wall clock, min and median). */
#include "unet.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }
int main(int argc, char **argv) {
    int P = argc > 1 ? atoi(argv[1]) : 96, B = argc > 2 ? atoi(argv[2]) : 2, it = argc > 3 ? atoi(argv[3]) : 10;
    nn_init(getenv("UFSM_GPU") ? atoi(getenv("UFSM_GPU")) : 0);
    nn_set_tf32(1);
    if (getenv("UFSM_F16")) nn_set_f16(1);
    if (getenv("UFSM_PREC")) nn_set_prec(atoi(getenv("UFSM_PREC")));
    if (getenv("UFSM_PREC_POLICY") && nn_set_prec_policy(getenv("UFSM_PREC_POLICY"))) return 2;
    if (getenv("UFSM_GN_STORED")) nn_set_gn_stored(1);
    if (getenv("UFSM_SR")) nn_set_sr(1);                                                  /* training as the trainer: SR (MX-fp4 gradients need it) */
    if (getenv("UFSM_BF16")) { nn_set_act_bf16(1); nn_set_grad_bf16(1); nn_set_grad_scale(1024); }   /* 16-bit storage modes */
    unet_cfg cfg = {4, {16, 32, 64, 80}, 4, 2, 8};
    if (getenv("UFSM_WIDTHS")) { char *t = strdup(getenv("UFSM_WIDTHS")); cfg.nlev = 0; for (char *q = strtok(t, ","); q && cfg.nlev < UNET_MAXLEV; q = strtok(nullptr, ",")) cfg.widths[cfg.nlev++] = atoi(q); free(t); }
    if (getenv("UFSM_COUT")) cfg.cout = atoi(getenv("UFSM_COUT"));
    if (getenv("UFSM_ENC_BLOCKS")) { char *t = strdup(getenv("UFSM_ENC_BLOCKS")); int l = 0; for (char *q = strtok(t, ","); q && l < UNET_MAXLEV; q = strtok(nullptr, ",")) cfg.enc_blocks[l++] = atoi(q); free(t); }
    if (getenv("UFSM_DEC_WIDTHS")) { char *t = strdup(getenv("UFSM_DEC_WIDTHS")); int l = 0; for (char *q = strtok(t, ","); q && l < UNET_MAXLEV; q = strtok(nullptr, ",")) cfg.dec_widths[l++] = atoi(q); free(t); }   /* narrower decoder levels */
    cfg.down_norm = getenv("UFSM_DOWN_NORM") ? atoi(getenv("UFSM_DOWN_NORM")) : 0;
    unet *u = unet_create(&cfg); unet_init(u, 1);
    shape5 xs = {B, 4, P, P, P};
    if (getenv("UFSM_MEM_REPORT")) unet_infer_bytes(u, xs);
    float *x = nn_malloc(shape_numel(xs) * 4); nn_zero(x, shape_numel(xs) * 4);
    { float *h = malloc(shape_numel(xs) * 4); for (size_t i = 0; i < shape_numel(xs); i++) h[i] = (float)((i * 2654435761u) % 1000) / 500.f - 1.f; nn_h2d(x, h, shape_numel(xs) * 4); free(h); }
    const int train = getenv("UFSM_TRAIN") != nullptr;   /* training step (forward + backward) instead of inference */
    shape5 os = xs; os.c = cfg.cout;
    float *gl = nn_malloc(shape_numel(os) * 4);
    { float *h = malloc(shape_numel(os) * 4); for (size_t i = 0; i < shape_numel(os); i++) h[i] = (float)((i * 2246822519u) % 2000) / 1e6f - 1e-3f; nn_h2d(gl, h, shape_numel(os) * 4); free(h); }
#define STEP() do { unet_forward(u, x, xs, train); if (train) { unet_zero_grad(u); unet_backward(u, gl); } } while (0)
    if (getenv("UFSM_MEM_REPORT")) { nn_sync(); fprintf(stderr, "free before the first forward %.3f GB\n", nn_mem_free() / 1e9); STEP(); nn_sync(); fprintf(stderr, "free after it %.3f GB (unet build %.3f GB)\n", nn_mem_free() / 1e9, unet_infer_bytes(u, xs) / 1e9); }
    for (int i = 0; i < 3; i++) STEP();   /* build + warm-up */
    nn_sync();
    char slot[UNET_NSLOT][16] = {{0}};   /* layer ids: enc i, down L + i, dec 3L - 3 - i, head 3L - 2; slot 2 id + conv */
    { const int L = cfg.nlev;
      for (int l = 0; l <= 3 * L - 2; l++)
          for (int c = 0; c < 2; c++) {
              char *d = slot[2 * l + c];
              if (l < L) snprintf(d, 16, "enc%d.c%d", l, c + 1);
              else if (l < 2 * L - 1) { if (!c) snprintf(d, 16, "down%d", l - L); }
              else if (l < 3 * L - 2) snprintf(d, 16, "dec%d.c%d", 3 * L - 3 - l, c + 1);
              else if (!c) snprintf(d, 16, "head");
          } }
    double best[UNET_NSLOT][3], cbest[8], tot[64];
    for (int i = 0; i < UNET_NSLOT; i++) for (int j = 0; j < 3; j++) best[i][j] = 1e30;
    for (int i = 0; i < 8; i++) cbest[i] = 1e30;
    unet_prof_layers_on(1);
    double lay[UNET_NSLOT][3];
    unet_prof_layers(lay);   /* reset */
    for (int k = 0; k < it; k++) {
        STEP(); nn_sync();
        unet_prof_layers(lay); double cat[8]; unet_prof_cats(cat);
        for (int i = 0; i < UNET_NSLOT; i++) for (int j = 0; j < 3; j++) if (lay[i][j] < best[i][j]) best[i][j] = lay[i][j];
        for (int i = 0; i < 8; i++) if (cat[i] < cbest[i]) cbest[i] = cat[i];
    }
    unet_prof_layers_on(0);
    for (int k = 0; k < it && k < 64; k++) { nn_sync(); double t0 = now(); STEP(); nn_sync(); tot[k] = (now() - t0) * 1e3; }
    int n = it < 64 ? it : 64; qsort(tot, n, sizeof(double), cmpd);
    double sum = 0;
    printf("%s P=%d B=%d policy=%s f16=%d\n", train ? "train" : "infer", P, B, getenv("UFSM_PREC_POLICY") ? getenv("UFSM_PREC_POLICY") : "-", getenv("UFSM_F16") != nullptr);
    for (int i = 0; i < UNET_NSLOT; i++) if (slot[i][0]) {
        if (train) { printf("  %-8s fwd %7.3f  bwd_data %7.3f  bwd_w %7.3f ms\n", slot[i], best[i][0], best[i][1], best[i][2]); sum += best[i][0] + best[i][1] + best[i][2]; }
        else { printf("  %-8s %7.3f ms\n", slot[i], best[i][0]); sum += best[i][0]; }
    }
    printf("  convs    %7.3f ms   (gn %.3f, elementwise %.3f, up %.3f)\n", sum, cbest[3], cbest[4], cbest[5]);
    printf("  %s  min %.3f ms, median %.3f ms\n", train ? "step   " : "forward", tot[0], tot[n / 2]);
    return 0;
}
