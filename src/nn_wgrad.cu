/* CUDA ops: wgrad section of the former nn.cu */
#include "nn_common.cuh"

int f4_wgrad(void) { static int v = -1; if (v < 0) v = getenv("UFSM_F4_WGRAD") ? atoi(getenv("UFSM_F4_WGRAD")) : 0; return v; }
int f4_had_w(void) {   /* bit 0: Hadamard (UFSM_F4_HAD_W), bit 1: stochastic rounding of x as well (UFSM_F4_SRX), bit 2: H16 variant */
    static int v = -1;
    if (v < 0) { const int hw = getenv("UFSM_F4_HAD_W") ? atoi(getenv("UFSM_F4_HAD_W")) : 0;   /* 1: H32 per 32-position block, 2: H16 along x (faster) */
                 v = (hw ? 1 : 0) | (hw == 2 ? 4 : 0) | (getenv("UFSM_F4_SRX") && atoi(getenv("UFSM_F4_SRX")) ? 2 : 0); }
    return v;
}
void launch_bwd_w_tc(const void *x, int xbf, shape5 xs, const void *gy, int gybf, shape5 ys, float *gw, float *gb, gnp_t gp, split_t sp) {
    exec_prec(2, eff_prec_w() == 3 && f4_wgrad() && (!sp.up || ISMX(x)) ? 3 : ISMX(x) || ((eff_prec_w() == 2 || eff_prec_w() == 3) && !sp.up) ? 2 : 1);
    if (eff_prec_w() == 3 && f4_wgrad() && (!sp.up || ISMX(x))) {   /* sp.up: MX only (fused upsample in the tile decode) */
        if (sr_on()) sp.sr = sr_seed();
        lp_bwd_w_f4(x, ISMX(x) ? MXDT(x) : LPDT(xbf), xs, gy, ISMX(gy) ? MXDT(gy) : LPDT(gybf), ys, gw, gb, gp, sp, f4_had_w());
        return;
    }
    if (ISMX(x)) {
        if (sr_on()) sp.sr = sr_seed();   /* MX activations still need SR when gy is requantised to fp8 */
        lp_bwd_w_f8(x, MXDT(x), xs, gy, ISMX(gy) ? MXDT(gy) : LPDT(gybf), ys, gw, gb, gp, sp); return;
    }
    if ((eff_prec_w() == 2 || eff_prec_w() == 3) && !sp.up) { if (sr_on()) sp.sr = sr_seed(); }
    if ((eff_prec_w() == 2 || eff_prec_w() == 3) && !sp.up) { lp_bwd_w_f8(x, LPDT(xbf), xs, gy, LPDT(gybf), ys, gw, gb, gp, sp); return; }   /* prec 4 (fp16) keeps the 16-bit kernel for the weight gradient */
    if (g_h16) launch_bwd_w_tc_h<f16>(x, xbf, xs, gy, gybf, ys, gw, gb, gp, sp);
    else launch_bwd_w_tc_h<bf16>(x, xbf, xs, gy, gybf, ys, gw, gb, gp, sp);
}
extern "C" void nn_conv3d_bwd_weight(const float *x, shape5 xs, const float *gy, shape5 ys, int k, int stride, float *gw, float *gb) {
    if (k == 1 || !g_tf32) exec_prec(2, 0);
    else if (stride == 2) exec_prec(2, ISMX(x) || eff_prec_w() == 2 || eff_prec_w() == 3 ? 2 : 1);
    size_t So = shape_spatial(ys);
    if (k == 3 && stride == 1 && g_tf32) {
        gnp_t none = {nullptr, nullptr, nullptr, nullptr, 0};
        split_t ns = {nullptr, 0, nullptr, 0};
        launch_bwd_w_tc(x, ABF, xs, gy, GBF, ys, gw, gb, none, ns);
        KCHECK();
        return;
    }
    if (k == 3 && stride == 2 && g_tf32 && ISMX(x)) { lp_bwd_w_s2_f8(x, MXDT(x), xs, gy, ISMX(gy) ? MXDT(gy) : LPDT(GBF), ys, gw, gb, gnp_t{}); KCHECK(); return; }
    if (k == 1 && stride == 1 && g_tf32 && ISMX(x)) {
        lp_bwd_w1_mx(x, MXDT(x), xs, gy, LPDT(GBF), ys, gw, gnp_t{});
        size_t So1 = shape_spatial(ys);
        if (gb) { if (GBF && g_h16) bias_grad_k<f16><<<dim3(ys.c, KSLAB), 256>>>((const f16 *)gy, gb, ys.n, ys.c, So1); else if (GBF) bias_grad_k<bf16><<<dim3(ys.c, KSLAB), 256>>>((const bf16 *)gy, gb, ys.n, ys.c, So1); else bias_grad_k<float><<<dim3(ys.c, KSLAB), 256>>>(gy, gb, ys.n, ys.c, So1); }
        KCHECK(); return;
    }
    if (k == 3 && stride == 2 && g_tf32 && (eff_prec_w() == 2 || eff_prec_w() == 3)) { lp_bwd_w_s2_f8(x, LPDT(ABF), xs, gy, LPDT(GBF), ys, gw, gb, gnp_t{}); KCHECK(); return; }
    if (k == 3 && stride == 2 && g_tf32) {
        if (g_h16) bwd_w_s2_h<f16>(x, xs, gy, ys, gw, gb); else bwd_w_s2_h<bf16>(x, xs, gy, ys, gw, gb);
        KCHECK();
        return;
    }
    if (k == 3) {
        int nxt = (ys.w + WTX - 1) / WTX, nyt = (ys.h + WTY - 1) / WTY, nzt = (ys.d + WTZ - 1) / WTZ;
        dim3 grid(xs.c, (ys.c + WCO - 1) / WCO, nxt * nyt * nzt * ys.n);
        if (stride == 1) conv_bwd_w3_k<1><<<grid, 256>>>(x, gy, gw, xs.n, xs.c, xs.d, xs.h, xs.w, ys.c, ys.d, ys.h, ys.w);
        else if (stride == 2) conv_bwd_w3_k<2><<<grid, 256>>>(x, gy, gw, xs.n, xs.c, xs.d, xs.h, xs.w, ys.c, ys.d, ys.h, ys.w);
        else { fprintf(stderr, "nn_conv3d_bwd_weight: unsupported stride %d\n", stride); abort(); }
    } else if (k == 1 && stride == 1 && xs.c * ys.c <= 128 && ys.c <= 8) {
        if (g_h16) bwd_w1_h<f16>(x, xs, gy, ys, gw, So); else bwd_w1_h<bf16>(x, xs, gy, ys, gw, So);
    } else { fprintf(stderr, "nn_conv3d_bwd_weight: unsupported k=%d stride=%d\n", k, stride); abort(); }
    if (gb) { if (GBF && g_h16) bias_grad_k<f16><<<dim3(ys.c, KSLAB), 256>>>((const f16 *)gy, gb, ys.n, ys.c, So); else if (GBF) bias_grad_k<bf16><<<dim3(ys.c, KSLAB), 256>>>((const bf16 *)gy, gb, ys.n, ys.c, So); else bias_grad_k<float><<<dim3(ys.c, KSLAB), 256>>>(gy, gb, ys.n, ys.c, So); }
    KCHECK();
}
extern "C" int nn_conv3d_fwd_gn(const float *x, shape5 xs, int G, const float *gamma, const float *beta, const float *mean, const float *rstd,
                                const float *w, const float *b, int cout, float *y) {
    if (G > xs.c) G = xs.c;   /* GroupNorm uses min(G, C) groups */
    if (!g_tf32) return -1;
    gnp_t gp = {gamma, beta, mean, rstd, G};
    split_t ns = {nullptr, 0, nullptr, 0};
    conv_fwd_tc(x, ABF, xs, w, b, cout, y, ABF, gp, nullptr, 0, ns);
    KCHECK();
    return 0;
}
extern "C" int nn_conv3d_fwd_gn_stats(const float *x, shape5 xs, int G_in, const float *gamma, const float *beta, const float *mean, const float *rstd,
                                      const float *w, const float *b, int cout, float *y, int G_out, float eps, float *omean, float *orstd) {
    if (G_in > xs.c) G_in = xs.c; if (G_out > cout) G_out = cout;
    if (!g_tf32) return -1;
    gnp_t gp = {gamma, beta, mean, rstd, G_in};
    int NG = xs.n * G_out;
    double *sums = nullptr;
    const int fused = gn_fused_stored(y, cout);
    if ((!g_gn_stored || fused) && G_out && omean && orstd) { sums = gn_dsums((size_t)2 * NG); cudaMemsetAsync(sums, 0, (size_t)2 * NG * sizeof(double)); }
    split_t ns = zs_split(split_t{}, xs.d);
    ns.stored_stats = fused;
    conv_fwd_tc(x, ABF, xs, w, b, cout, y, ABF, gp, sums, G_out, ns);
    KCHECK();
    if (G_out && omean && orstd) {
        if (g_gn_stored && !fused) { shape5 ys = xs; ys.c = cout; return nn_gn_stats(y, ys, G_out, eps, omean, orstd); }
        zs_reduce(sums, 2 * NG);
        gn_finalize_k<<<nblk(NG, 128), 128>>>(sums, NG, zs_len((size_t)(cout / G_out) * shape_spatial(xs), xs.d), eps, omean, orstd);
    }
    KCHECK();
    return 0;
}
extern "C" int nn_conv3d_fwd_split(const float *x, const float *x2, int c_split, shape5 xs, int G_in, const float *gamma, const float *beta, const float *mean, const float *rstd,
                                   const float *w, const float *b, int cout, float *y, int G_out, float eps, float *omean, float *orstd) {
    if (G_in > xs.c) G_in = xs.c; if (G_out > cout) G_out = cout;
    if (!g_tf32) return -1;
    gnp_t gp = {gamma, beta, mean, rstd, G_in};
    split_t sp = {x2, c_split, nullptr, 0};
    sp = zs_split(sp, xs.d);
    double *sums = nullptr;
    const int fused = gn_fused_stored(y, cout);
    sp.stored_stats = fused;
    if ((!g_gn_stored || fused) && G_out && omean && orstd) { int NG = xs.n * G_out; sums = gn_dsums((size_t)2 * NG); cudaMemsetAsync(sums, 0, (size_t)2 * NG * sizeof(double)); }
    conv_fwd_tc(x, ABF, xs, w, b, cout, y, ABF, gp, sums, G_out, sp);
    KCHECK();
    if (G_out && omean && orstd) {
        if (g_gn_stored && !fused) { shape5 ys = xs; ys.c = cout; return nn_gn_stats(y, ys, G_out, eps, omean, orstd); }
        zs_reduce(sums, 2 * xs.n * G_out); gn_finalize_k<<<nblk(xs.n * G_out, 128), 128>>>(sums, xs.n * G_out, zs_len((size_t)(cout / G_out) * shape_spatial(xs), xs.d), eps, omean, orstd);
    }
    KCHECK();
    return 0;
}
extern "C" int nn_conv3d_bwd_data_split(const float *gy, shape5 ys, const float *w, shape5 xs, float *gx, float *gx2, int o_split, float *scratch) {
    if (!g_tf32) return -1;
    int T = 27;
    size_t nw = (size_t)ys.c * xs.c * T;
    flip_w_k<<<nblk(nw, 256), 256>>>(w, scratch, ys.c, xs.c, T);
    gnp_t none = {nullptr, nullptr, nullptr, nullptr, 0};
    split_t sp = {nullptr, 0, gx2, o_split};
    int save = g_pass; g_pass = 1;
    conv_fwd_tc(gy, GBF, ys, scratch, nullptr, xs.c, gx, GBF, none, nullptr, 0, sp);
    g_pass = save;
    KCHECK();
    return 0;
}
extern "C" int nn_conv3d_bwd_data_range(const float *gy, shape5 ys, const float *w, shape5 xs, int c0, int nc, float *gx, float *scratch) {
    if (!g_tf32 || c0 < 0 || nc <= 0 || c0 + nc > xs.c) return -1;
    const int T = 27;
    size_t nw = (size_t)ys.c * xs.c * T;
    flip_w_k<<<nblk(nw, 256), 256>>>(w, scratch, ys.c, xs.c, T);   /* scratch[ci][co][t]: rows ci = output channels of this conv */
    gnp_t none = {}; split_t ns = {};
    shape5 gs = xs; gs.c = nc;
    int save = g_pass; g_pass = 1;
    conv_fwd_tc(gy, GBF, ys, scratch + (size_t)c0 * ys.c * T, nullptr, nc, gx, GBF, none, nullptr, 0, ns);
    g_pass = save;
    (void)gs;
    KCHECK();
    return 0;
}
extern "C" void nn_gn_silu_apply(const float *x, shape5 s, int G, const float *gamma, const float *beta, const float *mean, const float *rstd, float *y) {
    if (G > s.c) G = s.c;
    if (ISMX(x) || ISMX(y)) { if (!ISMX(y)) { fprintf(stderr, "gn_silu_apply: MX input needs an MX output\n"); abort(); } lp_gn_silu_apply_mx(x, ISMX(x) ? MXDT(x) : LPDT(ABF), s, G, gamma, beta, mean, rstd, y, MXDT(y)); KCHECK(); return; }   /* MX or 16-bit in, MX out */
    size_t n = shape_numel(s);
    if (ABF && g_h16) gn_apply_k<1, f16, f16><<<dim3(s.n * s.c, KSLAB), 256>>>((const f16 *)x, gamma, beta, mean, rstd, (f16 *)y, s.c, G, shape_spatial(s));
    else if (ABF) gn_apply_k<1, bf16, bf16><<<dim3(s.n * s.c, KSLAB), 256>>>((const bf16 *)x, gamma, beta, mean, rstd, (bf16 *)y, s.c, G, shape_spatial(s));
    else gn_apply_k<1, float, float><<<dim3(s.n * s.c, KSLAB), 256>>>(x, gamma, beta, mean, rstd, y, s.c, G, shape_spatial(s));
    KCHECK();
}
extern "C" int nn_conv3d_bwd_weight_gn(const float *x, shape5 xs, int G, const float *gamma, const float *beta, const float *mean, const float *rstd,
                                       const float *gy, shape5 ys, float *gw, float *gb) {
    if (G > xs.c) G = xs.c;
    if (!g_tf32) return -1;
    gnp_t gp = {gamma, beta, mean, rstd, G};
    split_t ns = {nullptr, 0, nullptr, 0};
    launch_bwd_w_tc(x, ABF, xs, gy, GBF, ys, gw, gb, gp, ns);
    KCHECK();
    return 0;
}
extern "C" int nn_conv3d_bwd_weight_split(const float *x, const float *x2, int c_split, shape5 xs, int G, const float *gamma, const float *beta, const float *mean, const float *rstd,
                                          const float *gy, shape5 ys, float *gw, float *gb) {
    if (G > xs.c) G = xs.c;
    if (!g_tf32) return -1;
    gnp_t gp = {gamma, beta, mean, rstd, G};
    split_t sp = {x2, c_split, nullptr, 0};
    launch_bwd_w_tc(x, ABF, xs, gy, GBF, ys, gw, gb, gp, sp);
    KCHECK();
    return 0;
}
