/* CUDA ops: fusedgn section of the former nn.cu */
#include "nn_common.cuh"

int mx_up_w_ok(const float *x, const float *x2, int c_split, shape5 xs) {
    static int on = -1; if (on < 0) on = getenv("UFSM_MX_UP_W") ? atoi(getenv("UFSM_MX_UP_W")) : 1;
    if (!on || !ISMX(x) || !ISMX(x2) || MXDT(x) != MXDT(x2) || c_split % 16 || (xs.c - c_split) % 16) return 0;
    if (eff_prec_w() == 3 && f4_wgrad()) {
        const int h = f4_had_w();
        static int lay = -1; if (lay < 0) lay = getenv("UFSM_F4W_LAYOUT") ? atoi(getenv("UFSM_F4W_LAYOUT")) : 1;
        if ((h & 3) && !(h & 4)) return 0;   /* LY 0 */
        if (!(h & 7) && lay == 0) return 0;
    }
    return 1;
}
int up_kernel_ok(const float *x, const float *x2, int wgrad) {
    static int mxup = -1; if (mxup < 0) mxup = getenv("UFSM_MX_UP") ? atoi(getenv("UFSM_MX_UP")) : 1;
    if (ISMX(x) || ISMX(x2)) return mxup && !wgrad && ISMX(x) && MXDT(x) == MXDT(x2);
    return 1;
}
int xsplit(const float *x2, const nn_gn_t *gx, const nn_gn_t *gx2, int c_split, int up, shape5 xs, split_t *sp) {
    *sp = split_t{};
    if (up && (!x2 || ((xs.d | xs.h | xs.w) & 1))) return -1;   /* up: decoder split, even dims (a GN on x: MX forward only, see the callers) */
    if (!x2) return 0;
    sp->up = up;
    if ((gx && gx->G) && !(gx2 && gx2->G)) { fprintf(stderr, "conv: a split input with a GroupNorm on x needs one on x2\n"); return -1; }
    sp->x2 = x2; sp->c_split = c_split; sp->gp2 = to_gnp(gx2);
    return 0;
}
extern "C" int nn_conv3d_fwd_x(const float *x, const nn_gn_t *gx, const float *x2, const nn_gn_t *gx2, int c_split, int up, shape5 xs,
                               const float *w, const float *b, int cout, int k, int stride, float *y, int G_out, float eps, float *omean, float *orstd) {
    if (!g_tf32) return -1;
    gnp_t gp = to_gnp(gx);
    shape5 ys = nn_conv3d_out_shape(xs, cout, k, stride);
    if (k == 3 && stride == 1) {
        split_t sp; if (xsplit(x2, gx, gx2, c_split, up, xs, &sp)) return -1;
        if (up && !up_kernel_ok(x, x2, 0)) return -1;
        if (up && gx && gx->G && !ISMX(x)) return -1;   /* GN+SiLU of the coarse x inside the up staging: MX kernels only */
        double *sums = nullptr;
        const int fused = gn_fused_stored(y, cout);
        sp.stored_stats = fused;
        if (G_out) {
            sp = zs_split(sp, ys.d);
            if ((!g_gn_stored || fused) && omean && orstd) { int NG = xs.n * G_out; sums = gn_dsums((size_t)2 * NG); cudaMemsetAsync(sums, 0, (size_t)2 * NG * sizeof(double)); }
        }
        conv_fwd_tc(x, ABF, xs, w, b, cout, y, ABF, gp, sums, G_out, sp);
        KCHECK();
        if (G_out && omean && orstd) {
            if (g_gn_stored && !fused) return nn_gn_stats(y, ys, G_out, eps, omean, orstd);
            zs_reduce(sums, 2 * xs.n * G_out); gn_finalize_k<<<nblk(xs.n * G_out, 128), 128>>>(sums, xs.n * G_out, zs_len((size_t)(cout / G_out) * shape_spatial(ys), ys.d), eps, omean, orstd);
        }
        KCHECK();
        return 0;
    }
    if (x2 || up || G_out) return -1;
    if (k == 3 && stride == 2) { conv_fwd_tc_s2(x, ABF, xs, w, b, cout, y, ABF, ys, gp); KCHECK(); return 0; }
    if (k == 1 && stride == 1) {
        exec_prec(0, 0);
        if (ISMX(x)) lp_conv1_fwd_mx(x, MXDT(x), xs, w, b, cout, y, gp);
        else { size_t S = shape_spatial(xs); dim3 gr(nblk((size_t)xs.n * S, 256));
            if (ABF && g_h16) conv1_f_k<f16, float><<<gr, 256>>>((const f16 *)x, w, b, y, xs.n, xs.c, cout, S, gp);
            else if (ABF) conv1_f_k<bf16, float><<<gr, 256>>>((const bf16 *)x, w, b, y, xs.n, xs.c, cout, S, gp);
            else conv1_f_k<float, float><<<gr, 256>>>(x, w, b, y, xs.n, xs.c, cout, S, gp); }
        KCHECK();
        return 0;
    }
    return -1;
}
extern "C" int nn_conv3d_bwd_weight_x(const float *x, const nn_gn_t *gx, const float *x2, const nn_gn_t *gx2, int c_split, int up, shape5 xs,
                                      const float *gy, shape5 ys, int k, int stride, float *gw, float *gb) {
    if (!g_tf32) return -1;
    gnp_t gp = to_gnp(gx);
    if (k == 3 && stride == 1) {
        split_t sp; if (xsplit(x2, gx, gx2, c_split, up, xs, &sp)) return -1;
        if (up && !up_kernel_ok(x, x2, 1) && !mx_up_w_ok(x, x2, c_split, xs)) return -1;
        if (up && gx && gx->G) return -1;   /* transformed up input is forward-only */
        launch_bwd_w_tc(x, ABF, xs, gy, GBF, ys, gw, gb, gp, sp);
        KCHECK();
        return 0;
    }
    if (x2 || up) return -1;
    if (k == 3 && stride == 2) {
        exec_prec(2, ISMX(x) || eff_prec_w() == 2 || eff_prec_w() == 3 ? 2 : 1);
        if (ISMX(x)) lp_bwd_w_s2_f8(x, MXDT(x), xs, gy, ISMX(gy) ? MXDT(gy) : LPDT(GBF), ys, gw, gb, gp);
        else if (eff_prec_w() == 2 || eff_prec_w() == 3) lp_bwd_w_s2_f8(x, LPDT(ABF), xs, gy, LPDT(GBF), ys, gw, gb, gp);
        else if (g_h16) bwd_w_s2_h<f16>(x, xs, gy, ys, gw, gb, gp); else bwd_w_s2_h<bf16>(x, xs, gy, ys, gw, gb, gp);
        KCHECK();
        return 0;
    }
    if (k == 1 && stride == 1) {
        exec_prec(2, 0);
        size_t So = shape_spatial(ys);
        if (ISMX(x)) lp_bwd_w1_mx(x, MXDT(x), xs, gy, LPDT(GBF), ys, gw, gp);
        else if (xs.c * ys.c <= 128 && ys.c <= 8) { if (g_h16) bwd_w1_h<f16>(x, xs, gy, ys, gw, So, gp); else bwd_w1_h<bf16>(x, xs, gy, ys, gw, So, gp); }
        else return -1;
        if (gb) { if (GBF && g_h16) bias_grad_k<f16><<<dim3(ys.c, KSLAB), 256>>>((const f16 *)gy, gb, ys.n, ys.c, So); else if (GBF) bias_grad_k<bf16><<<dim3(ys.c, KSLAB), 256>>>((const bf16 *)gy, gb, ys.n, ys.c, So); else bias_grad_k<float><<<dim3(ys.c, KSLAB), 256>>>(gy, gb, ys.n, ys.c, So); }
        KCHECK();
        return 0;
    }
    return -1;
}
extern "C" void nn_silu_bwd_gn(const float *x, shape5 s, int G, const float *gamma, const float *beta, const float *mean, const float *rstd, const float *gy, float *gx) {
    if (G > s.c) G = s.c;
    size_t n = shape_numel(s);
    if (ABF && g_h16) silu_bwd_gn_k<f16><<<nblk(n, 256), 256>>>((const f16 *)x, gamma, beta, mean, rstd, gy, gx, s.n, s.c, G, shape_spatial(s));
    else if (ABF) silu_bwd_gn_k<bf16><<<nblk(n, 256), 256>>>((const bf16 *)x, gamma, beta, mean, rstd, gy, gx, s.n, s.c, G, shape_spatial(s));
    else silu_bwd_gn_k<float><<<nblk(n, 256), 256>>>(x, gamma, beta, mean, rstd, gy, gx, s.n, s.c, G, shape_spatial(s));
    KCHECK();
}
__global__ void gn_group_sums_k(const float *st, const float *gamma, int N, int C, int G, float *AB, float f) {   /* AB[2*ng] = f sum st*gamma, AB[2*ng+1] = f sum st2*gamma */
    int ng = blockIdx.x * blockDim.x + threadIdx.x;
    if (ng >= N * G) return;
    int n = ng / G, g = ng % G, cpg = C / G;
    float A = 0.f, B = 0.f;
    for (int cc = g * cpg; cc < (g + 1) * cpg; cc++) { A += st[2 * (n * C + cc)] * gamma[cc]; B += st[2 * (n * C + cc) + 1] * gamma[cc]; }
    AB[2 * ng] = A * f; AB[2 * ng + 1] = B * f;
}
float gn_bwd_reduce(double *ds, float *st, int NC, int D, float *ggamma, float *gbeta, int N, int C) {
    if (!zs_on()) return 1.f;
    gn_param_grad_k2<<<nblk(C, 128), 128>>>(st, ggamma, gbeta, N, C);
    zs_reduce(ds, 2 * NC);
    d2f_k<<<nblk(2 * NC, 128), 128>>>(ds, st, 2 * NC);
    return (float)D / (float)zs_len((size_t)D, D);
}
extern "C" void nn_gn_silu_bwd(const float *x, shape5 s, int G, const float *gamma, const float *beta, const float *mean, const float *rstd, const float *gy,
                               float *gx, float *ggamma, float *gbeta, float *scratch) {
    if (G > s.c) G = s.c;
    if (ISMX(x)) {   /* MX x: voxel-major passes; gradients keep their storage */
        int NC = s.n * s.c;
        double *ds = gn_dsums((size_t)2 * NC);
        float *AB = scratch + 2 * NC;
        const int gdt = ISMX(gy) ? MXDT(gy) : LPDT(GBF), xdt = MXDT(x);
        if (gdt != (ISMX(gx) ? MXDT(gx) : LPDT(GBF)) || gdt == 4) { fprintf(stderr, "gn_silu_bwd: gy and gx must share the storage (MX-fp8 or 16-bit; never fp4)\n"); abort(); }
        lp_gn_silu_bwd_mx(x, xdt, s, G, gamma, beta, mean, rstd, gy, gx, gdt, ds, scratch, AB);
        d2f_k<<<nblk(2 * NC, 128), 128>>>(ds, scratch, 2 * NC);
        const int zsp = zs_on();
        const float f = gn_bwd_reduce(ds, scratch, NC, s.d, ggamma, gbeta, s.n, s.c);
        gn_group_sums_k<<<nblk(s.n * G, 128), 128>>>(scratch, gamma, s.n, s.c, G, AB, f);
        lp_gn_silu_bwd_apply_mx(x, xdt, s, G, gamma, beta, mean, rstd, gy, gx, gdt, AB);
        if (!zsp) gn_param_grad_k2<<<nblk(s.c, 128), 128>>>(scratch, ggamma, gbeta, s.n, s.c);
        KCHECK();
        return;
    }
    if (GBF && g_h16) gn_silu_bwd_t<f16, f16, f16>((const f16 *)x, s, G, gamma, beta, mean, rstd, (const f16 *)gy, (f16 *)gx, ggamma, gbeta, scratch);
    else if (GBF) gn_silu_bwd_t<bf16, bf16, bf16>((const bf16 *)x, s, G, gamma, beta, mean, rstd, (const bf16 *)gy, (bf16 *)gx, ggamma, gbeta, scratch);
    else if (ABF && g_h16) gn_silu_bwd_t<f16, float, float>((const f16 *)x, s, G, gamma, beta, mean, rstd, gy, gx, ggamma, gbeta, scratch);
    else if (ABF) gn_silu_bwd_t<bf16, float, float>((const bf16 *)x, s, G, gamma, beta, mean, rstd, gy, gx, ggamma, gbeta, scratch);
    else gn_silu_bwd_t<float, float, float>(x, s, G, gamma, beta, mean, rstd, gy, gx, ggamma, gbeta, scratch);
}
