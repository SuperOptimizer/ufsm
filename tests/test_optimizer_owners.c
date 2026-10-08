/* Model-level ownership checks: decay once, ignore another optimizer's stale moments,
   and update packed Adam moments/ordinary EMA once. No forward/backward is needed. */
#ifndef TEST_UNET_SOURCE
#define TEST_UNET_SOURCE "../src/unet.c"
#endif
#include TEST_UNET_SOURCE

static int failures;
static void mark(unsigned char *owner, const convp *c) {
    size_t n = (size_t)c->cout * c->cin * c->k * c->k * c->k;
    for (size_t i = c->w; i < c->w + n; i++) if (owner[i]++) abort();
}
static unsigned char *owners(const unet *u) {
    unsigned char *owner = calloc(u->np, 1);
    /* Build the reference from the architecture, independently of for_each_conv3/plain_ranges. */
    for (int l = 0; l < u->cfg.nlev; l++) { mark(owner, &u->enc[l].c1); mark(owner, &u->enc[l].c2); }
    for (int l = 0; l + 1 < u->cfg.nlev; l++) mark(owner, &u->down[l]);
    for (int l = u->cfg.nlev - 2; l >= 0; l--) { mark(owner, &u->dec[l].c1); mark(owner, &u->dec[l].c2); }
    return owner;
}
static void check(const char *name, const float *got, const float *want, size_t n, double tol) {
    double worst = 0; int finite = 1;
    for (size_t i = 0; i < n; i++) { finite &= isfinite(got[i]) && isfinite(want[i]); if (isfinite(got[i])) worst = fmax(worst, fabs((double)got[i] - want[i])); }
    int ok = finite && worst <= tol;
    printf("  %-36s max error %.3g %s\n", name, worst, ok ? "ok" : "FAIL"); failures += !ok;
}
static void mixed_case(const unet_cfg *cfg, int algorithm, int stale) {
    unet *u = unet_create(cfg); unet_init(u, 19);
    size_t n = u->np; float *before = malloc(n * 4), *want = calloc(n, 4), *got = malloc(n * 4), *mom = calloc(n, 4);
    unsigned char *owner = owners(u);
    nn_d2h(before, u->p, n * 4);
    const float lm = stale ? 0.f : 0.03f, la = 0.007f, wd = stale ? 0.f : 0.2f;
    for (size_t i = 0; i < n; i++) { want[i] = before[i] * (1.f - (owner[i] ? lm : la) * wd); mom[i] = stale && owner[i] ? 0.1f : 0.f; }
    nn_h2d(u->m, mom, n * 4);
    for (size_t i = 0; i < n; i++) got[i] = stale && owner[i] ? 0.02f : 0.f;
    nn_h2d(u->v, got, n * 4); unet_zero_grad(u);
    if (algorithm == 2) unet_anvil(u, lm, wd, 1, 100, la, 0.9f, 0.999f, 1e-8f, wd);
    else {
        if (algorithm) setenv("UFSM_MUON_UNBATCHED", "1", 1); else unsetenv("UFSM_MUON_UNBATCHED");
        unet_muon(u, lm, 0.95f, la, 0.9f, 0.999f, 1e-8f, wd, 1);
    }
    nn_d2h(got, u->p, n * 4); check(stale ? "stale Adam conv moments ignored" : "decay applied by one owner", got, want, n, 2e-7);
    nn_d2h(got, u->m, n * 4); check("conv Adam moments unchanged", got, mom, n, 1e-8);
    free(owner); free(before); free(want); free(got); free(mom); unet_free(u);
}
static void packed_case(const unet_cfg *cfg, int bits) {
    unet *u = unet_create(cfg); unet_init(u, 19); unet_set_wq(u, bits);
    size_t n = u->np; float *before = malloc(n * 4), *want = calloc(n, 4), *got = malloc(n * 4), *grad = malloc(n * 4);
    unsigned char *owner = owners(u);
    nn_d2h(before, u->ema, n * 4);
    for (size_t i = 0; i < n; i++) { grad[i] = 0.03125f; want[i] = (1.f - 0.9f) * grad[i]; }
    unet_grad_h2d(u, grad); unet_adamw(u, 0.001f, 0.9f, 0.999f, 1e-8f, 0.01f, 1);
    nn_d2h(got, u->m, n * 4); check("packed Adam first moment once", got, want, n, 1e-8);
    for (size_t i = 0; i < n; i++) want[i] = (1.f - 0.999f) * grad[i] * grad[i];
    nn_d2h(got, u->v, n * 4); check("packed Adam second moment once", got, want, n, 1e-10);
    nn_d2h(got, u->p, n * 4); unet_ema(u, 0.9f);
    for (size_t i = 0; i < n; i++) want[i] = 0.9f * before[i] + (1.f - 0.9f) * got[i];
    nn_d2h(got, u->ema, n * 4);
    /* Packed conv EMA is requantized; ordinary params must follow the scalar EMA exactly once. */
    for (size_t i = 0; i < n; i++) if (owner[i]) want[i] = got[i];
    check("ordinary EMA once with packed convs", got, want, n, 2e-7);
    free(owner); free(before); free(want); free(got); free(grad); unet_free(u);
}
static void input_case(int bits) {
    unet_cfg cfg = {2, {16,32}, 4, 2, 8, 1}; shape5 xs = {1,4,16,16,16};
    size_t n = shape_numel(xs), no = n / 2;
    float *h = malloc(n * 4), *got = malloc(no * 4), *want = malloc(no * 4), *xf = nn_malloc(n * 4);
    void *xh = nn_malloc(n * 2);
    nn_set_prec(bits == 4 ? 3 : 2); nn_set_f16(1); nn_set_sr(0); nn_set_prec_policy("");
    unet_set_act_mx4(bits == 4); unet_set_act_mx8(bits == 8); unet_set_grad_mx8(0); unet_set_recompute(1);
    for (size_t i = 0; i < n; i++) h[i] = (float)((i * 2654435761u) % 1009) / 504.f - 1.f;
    nn_h2d(xf, h, n * 4); nn_f32_to_h16(xf, n, xh, 1.f);
    unet *reused = unet_create(&cfg); unet_init(reused, 19);
    const int modes[] = {0,4,8,4,0,8};
    for (int k = 0; k < 6; k++) {
        int precision = modes[k], mx = precision != 0; unet_set_input_prec(precision); unet_set_input_mx(mx);
        const float *p = unet_forward_x(reused, xh, xs, 0, 1); nn_d2h(got, p, no * 4);
        int storage = mx ? reused->xin && nn_storage(reused->xin) == precision : reused->xin == nullptr;
        printf("input transition bodyMX%d input%d: storage %s\n", bits, precision, storage ? "ok" : "FAIL"); failures += !storage;
        unet *fresh = unet_create(&cfg); unet_init(fresh, 19);
        p = unet_forward_x(fresh, xh, xs, 0, 1); nn_d2h(want, p, no * 4);
        check("input transition matches fresh buffers", got, want, no, 2e-3); unet_free(fresh);
    }
    unet_set_input_prec(8); unet_set_input_mx(1); unet_set_grad_mx8(1);
    size_t planned = unet_train_bytes(reused, xs);
    unet_forward_x(reused, xh, xs, 1, 1);
    int accounted = reused->act_bytes == planned && (reused->xin_shared ? reused->xin == reused->dec[0].a2 : nn_storage(reused->xin) == 8);   /* input offload: in dec[0].a2's buffer (holding a2 after the forward) */
    printf("bodyMX%d stemMX8 dry memory includes input: %s\n", bits, accounted ? "ok" : "FAIL"); failures += !accounted;
    unet_set_grad_mx8(0);
    unet_free(reused); nn_free(xf); nn_free(xh); free(h); free(got); free(want); unet_set_input_prec(0); unet_set_input_mx(0);
}
int main(void) {
    if (nn_init(0)) return 1;
    const int levels[] = {1, 2, 4};
    for (int c = 0; c < 3; c++) for (int dn = 0; dn <= 1; dn++) {
        unet_cfg cfg = {levels[c], {4, 8, 12, 16}, 4, 2, 2, dn};
        printf("optimizer owners: levels %d down_norm %d\n", cfg.nlev, dn);
        for (int algorithm = 0; algorithm < 3; algorithm++) for (int stale = 0; stale <= 1; stale++) mixed_case(&cfg, algorithm, stale);
        packed_case(&cfg, 8); packed_case(&cfg, 4);
    }
    unsetenv("UFSM_MUON_UNBATCHED"); input_case(8); input_case(4);
    nn_sync(); const char *e = nn_check(); if (e) { fprintf(stderr, "%s\n", e); failures++; }
    printf("optimizer ownership: %s\n", failures ? "FAIL" : "ok"); return failures != 0;
}
