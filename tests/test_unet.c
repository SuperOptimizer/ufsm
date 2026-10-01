/* Whole-network finite-difference check on a tiny U-Net, then a timing benchmark at 128^3. */
#include "unet.h"
#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static float frand(void) { return (float)rand() / RAND_MAX * 2.f - 1.f; }

/* L = sum gy * logits */
static double loss_of(unet *u, const float *dx, shape5 xs, const float *dgy, size_t nl, float *h) {
    const float *lg = unet_forward(u, dx, xs, 1);
    nn_d2h(h, lg, nl * 4);
    float *gy = malloc(nl * 4);
    nn_d2h(gy, dgy, nl * 4);
    double s = 0;
    for (size_t i = 0; i < nl; i++) s += (double)gy[i] * h[i];
    free(gy);
    return s;
}

static char ckpt_path[64];
int main(void) {
    snprintf(ckpt_path, sizeof ckpt_path, "/tmp/ufsm_test_unet_%d.ckpt", (int)getpid());
    if (nn_init(getenv("UFSM_GPU") ? atoi(getenv("UFSM_GPU")) : 0)) { printf("no cuda\n"); return 1; }
    nn_set_tf32(0);   /* finite-difference checks need the exact fp32 kernels; the benchmark re-enables tensor cores */
    srand(3);
    unet_cfg cfg = {3, {4, 6, 8}, 4, 2, 2};
    if (getenv("UFSM_TINY")) { cfg.nlev = 0; char *t = strdup(getenv("UFSM_TINY")); for (char *q = strtok(t, ","); q; q = strtok(nullptr, ",")) cfg.widths[cfg.nlev++] = atoi(q); }
    unet *u = unet_create(&cfg);
    unet_init(u, 1);
    size_t np = unet_nparams(u);
    printf("tiny unet: %zu params\n", np);
    shape5 xs = {getenv("UFSM_TN") ? atoi(getenv("UFSM_TN")) : 2, 4, 8, 8, 16};
    size_t nx = shape_numel(xs);
    shape5 os = unet_out_shape(u, xs);
    size_t nl = shape_numel(os);
    float *x = malloc(nx * 4), *gy = malloc(nl * 4), *h = malloc(nl * 4);
    for (size_t i = 0; i < nx; i++) x[i] = frand();
    for (size_t i = 0; i < nl; i++) gy[i] = frand();
    float *dx = nn_malloc(nx * 4), *dgy = nn_malloc(nl * 4);
    nn_h2d(dx, x, nx * 4); nn_h2d(dgy, gy, nl * 4);
    /* analytic grads */
    unet_forward(u, dx, xs, 1);
    unet_zero_grad(u);
    unet_backward(u, dgy);
    const char *e = nn_check();
    if (e) { printf("cuda error: %s\n", e); return 1; }
    float *g = malloc(np * 4), *p = malloc(np * 4);
    /* access the flat arrays through save/load: simpler to re-read via the checkpoint file */
    unet_save(u, ckpt_path, 0, nullptr);
    FILE *f = fopen(ckpt_path, "rb");
    char line[4096]; fseek(f, 4, SEEK_SET); fgets(line, sizeof line, f);
    fread(p, 4, np, f); fclose(f);
    /* grads: not in the checkpoint; recover by saving after copying g into m? Instead expose via a second file: use adamw with lr to read? Simplest: compute analytic grads here via a private hook — we reuse unet_grad_norm and per-param FD on a random subset. */
    double gnorm = unet_grad_norm(u);
    printf("grad norm %.4g\n", gnorm);
    /* finite differences on a random subset of parameters: dL/dp ~ (L(p+e)-L(p-e))/2e; compare with analytic via the
       directional derivative trick: pick K params, build direction d = e_k, analytic = g_k. We read g_k by nudging the
       parameter and using AdamW? No — read the grad array directly through a tiny API: we saved nothing. Use the
       following: unet_adamw with b1=0, b2=1-1e-30 is messy; instead compare the full gradient via random directions:
       L(p + t v) - L(p - t v) / 2t = g . v for random v; g . v is obtained from grad norm identity? Not enough.
       => We check g . v for K random sparse directions using the grad extracted by a temporary save of u->g: done by
       calling unet_save after swapping (hack-free): the checkpoint stores m; we set m = g via adamw step 1 with
       lr = 0 and b1 = 0: m <- 0*m + 1*g. */
    unet_adamw(u, 0.f, 0.f, 0.999f, 1e-8f, 0.f, 1);
    unet_save(u, ckpt_path, 0, nullptr);
    f = fopen(ckpt_path, "rb");
    fseek(f, 4, SEEK_SET); fgets(line, sizeof line, f);
    fread(p, 4, np, f); fread(g, 4, np, f); fread(g, 4, np, f); fclose(f);   /* third array = m = g */
    double worst = 0, gmax = 0;
    for (size_t i = 0; i < np; i++) if (fabs(g[i]) > gmax) gmax = fabs(g[i]);
    int K = 60, bad = 0;
    for (int k = 0; k < K; k++) {
        size_t i = (size_t)rand() % np;
        float o = p[i], eps = 1e-2f;
        float *dp = nullptr; (void)dp;
        /* nudge parameter i on the device: rewrite the whole param array (small net) */
        p[i] = o + eps; { float *tmp = malloc(np * 4); memcpy(tmp, p, np * 4); FILE *w = fopen(ckpt_path, "r+b"); fseek(w, 4, SEEK_SET); fgets(line, sizeof line, w); fwrite(tmp, 4, np, w); fclose(w); free(tmp); }
        unet_load(u, ckpt_path);
        double lp = loss_of(u, dx, xs, dgy, nl, h);
        p[i] = o - eps; { FILE *w = fopen(ckpt_path, "r+b"); fseek(w, 4, SEEK_SET); fgets(line, sizeof line, w); fwrite(p, 4, np, w); fclose(w); }
        unet_load(u, ckpt_path);
        double lm = loss_of(u, dx, xs, dgy, nl, h);
        p[i] = o;
        double fd = (lp - lm) / (2 * eps), err = fabs(fd - g[i]) / (gmax + 1e-9);
        if (err > worst) worst = err;
        if (err > 5e-3) { bad++; printf("  param %zu: fd %.5g analytic %.5g\n", i, fd, g[i]); }
    }
    { FILE *w = fopen(ckpt_path, "r+b"); fseek(w, 4, SEEK_SET); fgets(line, sizeof line, w); fwrite(p, 4, np, w); fclose(w); }
    unet_load(u, ckpt_path);
    printf("param fd check: worst rel err %.3g over %d params (%d bad)\n", worst, K, bad);
    /* EMA/use_ema path sanity: ema == p right after init copy? we changed p via adamw(lr=0) so p unchanged: outputs equal */
    unet_use_ema(u, 1);
    double le = loss_of(u, dx, xs, dgy, nl, h);
    unet_use_ema(u, 0);
    double ll = loss_of(u, dx, xs, dgy, nl, h);
    printf("ema-vs-live loss %.6g vs %.6g\n", le, ll);
    unet_free(u);

    /* ---- benchmark: the real config at 128^3 ---- */
    if (getenv("UFSM_PREC")) nn_set_prec(atoi(getenv("UFSM_PREC")));   /* 1 bf16, 2 fp8, 3 fp4 fwd + fp8 wgrad */
    if (getenv("UFSM_PREC_POLICY") && nn_set_prec_policy(getenv("UFSM_PREC_POLICY"))) return 1;
    nn_set_tf32(getenv("UFSM_FP32") ? 0 : 1);
    unet_cfg big = {4, {16, 32, 64, 80}, 4, 1, 8};
    unet *b = unet_create(&big);
    unet_init(b, 2);
    printf("model (16,32,64,80): %zu params\n", unet_nparams(b));
    int BP = getenv("UFSM_P") ? atoi(getenv("UFSM_P")) : 96, BB = getenv("UFSM_B") ? atoi(getenv("UFSM_B")) : 1;
    shape5 bs = {BB, 4, BP, BP, BP};
    size_t bn = shape_numel(bs), bl = (size_t)bs.n * BP * BP * BP;
    float *bx = nn_malloc(bn * 4), *bg = nn_malloc(bl * 4);
    nn_zero(bx, bn * 4); nn_zero(bg, bl * 4);
    /* tensor-core path vs exact fp32 on the full model (random input) */
    {
        float *hx = malloc(bn * 4); for (size_t i = 0; i < bn; i++) hx[i] = frand(); nn_h2d(bx, hx, bn * 4); free(hx);
        float *l1 = malloc(bl * 4), *l2 = malloc(bl * 4);
        nn_set_tf32(0); nn_d2h(l1, unet_forward(b, bx, bs, 1), bl * 4);
        nn_set_tf32(1); nn_d2h(l2, unet_forward(b, bx, bs, 1), bl * 4);
        double md = 0, mx = 0; for (size_t i = 0; i < bl; i++) { if (fabs(l1[i] - l2[i]) > md) md = fabs(l1[i] - l2[i]); if (fabs(l1[i]) > mx) mx = fabs(l1[i]); }
        printf("logits tensor-core vs fp32: max abs diff %.3g (max |logit| %.3g)\n", md, mx);
        free(l1); free(l2);
        /* whole-network gradient: tensor-core path (fused gn, bf16 activations, accumulate modes) vs exact fp32 */
        {
            float *hg = malloc(bl * 4); for (size_t i = 0; i < bl; i++) hg[i] = frand(); nn_h2d(bg, hg, bl * 4); free(hg);
            size_t np = unet_nparams(b); float *g1 = malloc(np * 4), *g2 = malloc(np * 4);
            nn_set_tf32(0); unet_forward(b, bx, bs, 1); unet_zero_grad(b); unet_backward(b, bg); unet_grad_d2h(b, g1);
            nn_set_tf32(1); unet_forward(b, bx, bs, 1); unet_zero_grad(b); unet_backward(b, bg); unet_grad_d2h(b, g2);
            double d2 = 0, n2 = 0; for (size_t i = 0; i < np; i++) { d2 += (double)(g1[i] - g2[i]) * (g1[i] - g2[i]); n2 += (double)g1[i] * g1[i]; }
            double rel = sqrt(d2 / (n2 > 0 ? n2 : 1));
            printf("grads tensor-core vs fp32: rel L2 diff %.3g%s\n", rel, rel < 0.05 ? "" : "  FAIL");
            if (rel >= 0.05) bad++;
            free(g1); free(g2);
            nn_zero(bg, bl * 4);
        }
        nn_set_tf32(getenv("UFSM_FP32") ? 0 : 1);
    }
    unet_forward(b, bx, bs, 1);
    unet_zero_grad(b); unet_backward(b, bg); nn_sync();
    printf("activations: %.2f GB, free after build: %.2f GB\n", unet_activation_bytes(b) / 1e9, nn_mem_free() / 1e9);
    unet_prof_report();   /* reset after the tiny-net checks */
    double t0 = now();
    int iters = 5;
    for (int i = 0; i < iters; i++) { unet_forward(b, bx, bs, 1); unet_zero_grad(b); unet_backward(b, bg); unet_adamw(b, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f, i + 1); }
    nn_sync();
    double dt = (now() - t0) / iters;
    printf("train step (B=%d, %d^3): %.3f s  (%.1f samples/s)\n", BB, BP, dt, BB / dt);
    unet_prof_report();
    t0 = now();
    for (int i = 0; i < iters; i++) unet_forward(b, bx, bs, 0);
    nn_sync();
    printf("inference forward (B=%d, %d^3): %.3f s\n", BB, BP, (now() - t0) / iters);
    e = nn_check();
    printf(e ? "cuda error: %s\n" : "unet ok\n", e ? e : "");
    remove(ckpt_path);
    return e != nullptr || bad;
}
