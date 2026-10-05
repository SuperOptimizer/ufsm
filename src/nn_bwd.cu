/* CUDA ops: bwd section of the former nn.cu */
#include "nn_common.cuh"
/* instantiated in nn_fwdi*.cu */
extern template int conv_fwd_tc_h<f16>(const void *x, int xbf, shape5 xs, const float *w, const float *b, int cout, void *y, int ybf, gnp_t gp, double *osum, int Go, split_t sp, const tapset_t *ts);
#ifdef UFSM_ALL_TYPES
extern template int conv_fwd_tc_h<bf16>(const void *x, int xbf, shape5 xs, const float *w, const float *b, int cout, void *y, int ybf, gnp_t gp, double *osum, int Go, split_t sp, const tapset_t *ts);
#endif
extern template int conv_fwd_tc_f16acc<f16>(const void *x, int xbf, shape5 xs, const float *w, const float *b, int cout, void *y, int ybf, gnp_t gp, double *osum, int Go, split_t sp, const tapset_t *ts);
#ifdef UFSM_ALL_TYPES
extern template int conv_fwd_tc_f16acc<bf16>(const void *x, int xbf, shape5 xs, const float *w, const float *b, int cout, void *y, int ybf, gnp_t gp, double *osum, int Go, split_t sp, const tapset_t *ts);
#endif
extern template int conv_fwd_tc_s2_h<f16>(const void *x, int xbf, shape5 xs, const float *w, const float *b, int cout, void *y, int ybf, shape5 ys, gnp_t gp);
#ifdef UFSM_ALL_TYPES
extern template int conv_fwd_tc_s2_h<bf16>(const void *x, int xbf, shape5 xs, const float *w, const float *b, int cout, void *y, int ybf, shape5 ys, gnp_t gp);
#endif

g_zs_t g_zs[8];
void (*g_split_reduce)(double *, int);
extern "C" void nn_split_cfg(int lo0, int hi0, int D0, int Dg0) { g_zs[cur_dev()] = {D0 > 0, lo0, hi0, D0, Dg0}; }
extern "C" void nn_split_set_reduce(void (*fn)(double *, int)) { g_split_reduce = fn; }
int zs_on(void) { return g_zs[cur_dev()].on; }
void zs_range(int D, int *lo, int *hi) {   /* halo planes at the low / high end of a tensor of depth D */
    const auto &z = g_zs[cur_dev()];
    *lo = z.on ? (int)((long)z.lo0 * D / z.D0) : 0; *hi = z.on ? (int)((long)z.hi0 * D / z.D0) : 0;
}
size_t zs_len(size_t len, int D) { const auto &z = g_zs[cur_dev()]; return z.on ? len / D * (size_t)((long)z.Dg0 * D / z.D0) : len; }   /* element count of the whole window */
void zs_reduce(double *b, int n) { if (zs_on() && g_split_reduce) g_split_reduce(b, n); }   /* sum over both GPUs */
split_t zs_split(split_t sp, int D) { zs_range(D, &sp.zlo, &sp.zhi); return sp; }
int gn_fused_stored(const void *y, int cout) {
    /* MX epilogues can reduce their rounded outputs directly. Ordinary outputs retain
       the independent FP64 pass; the environment switch preserves the reference path. */
    return g_gn_stored && ISMX(y) && cout >= 16 &&
        (!getenv("UFSM_FUSED_STORED_GN") || ufsm_env_on("UFSM_FUSED_STORED_GN"));
}
void *tc_wbuf(size_t n) { static void *buf[8]; static size_t cap[8]; int d = cur_dev(); if (n > cap[d]) { if (buf[d]) cudaFree(buf[d]); cudaMalloc(&buf[d], n * 2); cap[d] = n; } return buf[d]; }
int fw_tz(int MT) {
    static int e = -2;
    if (e == -2) { const char *v = getenv("UFSM_FW_TZ"); e = v ? atoi(v) : -1; }
    if (MT != 1) return FW_TZ;
    if (e == 2 || e == 4 || e == 6) return e;
    return FW_TZ_MT1;
}
size_t fw_smem(int MT, int tz) { return (size_t)(TC_CI * (tz + 2) * 180 + (MT <= 2 ? 2 : 1) * 9 * MT * 16 * TC_CI) * 2; }
float *tc_wscbuf(size_t n) { static float *buf[8]; static size_t cap[8]; int d = cur_dev(); if (n > cap[d]) { if (buf[d]) cudaFree(buf[d]); cudaMalloc(&buf[d], n * sizeof(float)); cap[d] = n; } return buf[d]; }
int conv_fwd_tc(const void *x, int xbf, shape5 xs, const float *w, const float *b, int cout, void *y, int ybf, gnp_t gp, double *osum, int Go, split_t sp, const tapset_t *ts) {
    const int pr = eff_prec();
    const int mx = ISMX(x) || ISMX(y);
    exec_prec(g_pass, !ts && mx ? xs.c <= 8 ? 2 : MXDT(x) == 4 || pr == 3 ? 3 : 2 : !ts && !sp.up && (pr == 2 || pr == 3) ? xs.c <= 8 && pr == 3 ? 2 : pr : pr == 4 ? 4 : 1);
    if (!ts && (ISMX(x) || ISMX(y))) {   /* MX activation storage (fp8 or fp4): staged from the stored rows; fp4 storage or a
                                            prec-3 policy runs the fp4 kernel, else fp8 compute (copy staging) */
        const int mdt = MXDT(x);
        if (!mdt && ISMX(y) && xbf && xs.c <= 8 && !sp.x2 && !sp.y2 && !sp.accum) return lp_conv_fwd_f8(x, LPDT(xbf), xs, w, b, cout, y, MXDT(y), gp, osum, Go, sp);   /* 16-bit network input -> MX a1 */
        if (mdt && ISMX(y) && mdt != MXDT(y) && xs.c <= 8 && !sp.x2 && !sp.y2 && !sp.up && !sp.accum)
            return lp_conv_fwd_f8(x, mdt, xs, w, b, cout, y, MXDT(y), gp, osum, Go, sp);   /* independently quantized stem input */
        if (mdt == 4 && !ISMX(y) && !sp.y2 && !ybf) { sp.wkey = conv_wkey(); return lp_conv_fwd_f4(x, 4, xs, w, b, cout, y, 0, gp, osum, Go, sp); }   /* mx4 in, fp32 out (tests) */
        if (!mdt || MXDT(y) != mdt || (sp.x2 && MXDT(sp.x2) != mdt) || (sp.y2 && MXDT(sp.y2) != mdt)) { fprintf(stderr, "conv: MX storage needs MX inputs and outputs of one format\n"); abort(); }
        sp.wkey = conv_wkey();
        if ((mdt == 4 || pr == 3) && g_pass == 1 && sr_on()) sp.sr = sr_seed();   /* fp4 backward-data: stochastic rounding of the (MX-fp8) gy operand */
        if (mdt == 4 || pr == 3) return lp_conv_fwd_f4(x, mdt, xs, w, b, cout, y, mdt, gp, osum, Go, sp);
        return lp_conv_fwd_f8(x, mdt, xs, w, b, cout, y, mdt, gp, osum, Go, sp);
    }
    if (!ts && (pr == 2 || pr == 3) && !sp.up && g_pass == 1) { if (sr_on()) sp.sr = sr_seed(); }   /* backward-data: gy is the staged operand (fp8 dither / exact e2m1 SR) */
    if (!ts && pr == 2 && !sp.up) return lp_conv_fwd_f8(x, LPDT(xbf), xs, w, b, cout, y, LPDT(ybf), gp, osum, Go, sp);   /* sp.up: 16-bit kernels only */
    if (!ts && pr == 3 && !sp.up) { sp.wkey = conv_wkey(); return lp_conv_fwd_f4(x, LPDT(xbf), xs, w, b, cout, y, LPDT(ybf), gp, osum, Go, sp); }
    if (pr == 4) return g_h16 ? conv_fwd_tc_f16acc<f16>(x, xbf, xs, w, b, cout, y, ybf, gp, osum, Go, sp, ts) : VERIFY_ONLY(conv_fwd_tc_f16acc<bf16>(x, xbf, xs, w, b, cout, y, ybf, gp, osum, Go, sp, ts));
    return g_h16 ? conv_fwd_tc_h<f16>(x, xbf, xs, w, b, cout, y, ybf, gp, osum, Go, sp, ts) : VERIFY_ONLY(conv_fwd_tc_h<bf16>(x, xbf, xs, w, b, cout, y, ybf, gp, osum, Go, sp, ts));
}
int conv_fwd_tc_s2(const void *x, int xbf, shape5 xs, const float *w, const float *b, int cout, void *y, int ybf, shape5 ys, gnp_t gp) {
    exec_prec(g_pass, ISMX(x) || ISMX(y) || ((eff_prec() == 2 || eff_prec() == 3) && xbf == ybf) ? 2 : 1);
    if (ISMX(x) || ISMX(y)) { if (MXDT(x) != MXDT(y)) { fprintf(stderr, "conv s2: MX storage needs MX input and output of one format\n"); abort(); } return lp_conv_fwd_s2_f8(x, MXDT(x), xs, w, b, cout, y, MXDT(y), ys, gp); }
    if ((eff_prec() == 2 || eff_prec() == 3) && xbf == ybf) return lp_conv_fwd_s2_f8(x, LPDT(xbf), xs, w, b, cout, y, LPDT(ybf), ys, gp);
    return g_h16 ? conv_fwd_tc_s2_h<f16>(x, xbf, xs, w, b, cout, y, ybf, ys, gp) : VERIFY_ONLY(conv_fwd_tc_s2_h<bf16>(x, xbf, xs, w, b, cout, y, ybf, ys, gp));
}
extern "C" shape5 nn_conv3d_out_shape(shape5 xs, int cout, int k, int stride) {
    shape5 o = {xs.n, cout, (xs.d + 2 * (k / 2) - k) / stride + 1, (xs.h + 2 * (k / 2) - k) / stride + 1, (xs.w + 2 * (k / 2) - k) / stride + 1};
    return o;
}
extern "C" void nn_conv3d_fwd(const float *x, shape5 xs, const float *w, const float *b, int cout, int k, int stride, float *y) {
    if (k == 1 || !g_tf32) exec_prec(g_pass, 0);
    if (k == 1 && g_tf32 && ISMX(x)) { lp_conv1_fwd_mx(x, MXDT(x), xs, w, b, cout, y, gnp_t{}); KCHECK(); return; }   /* head reading an MX tensor (fp32 output) */
    shape5 ys = nn_conv3d_out_shape(xs, cout, k, stride);
    if (k == 3 && stride == 1 && g_tf32) { gnp_t none = {nullptr, nullptr, nullptr, nullptr, 0}; split_t ns = {nullptr, 0, nullptr, 0}; conv_fwd_tc(x, ABF, xs, w, b, cout, y, ABF, none, nullptr, 0, ns); KCHECK(); return; }
    if (k == 3 && stride == 2 && g_tf32) { conv_fwd_tc_s2(x, ABF, xs, w, b, cout, y, ABF, ys); KCHECK(); return; }
    if (k == 1 && stride == 1 && ABF) { size_t S = shape_spatial(xs); if (g_h16) conv1_f_k<f16, float><<<nblk((size_t)xs.n * S, 256), 256>>>((const f16 *)x, w, b, y, xs.n, xs.c, cout, S); else conv1_f_k<bf16, float><<<nblk((size_t)xs.n * S, 256), 256>>>((const bf16 *)x, w, b, y, xs.n, xs.c, cout, S); KCHECK(); return; }
    if (k == 3 && stride == 1) {
        dim3 grid(nblk(ys.w, TX * 4), nblk(ys.h, TY), nblk(ys.d, TZ) * nblk(cout, COT) * xs.n);
        conv_fwd_k<3, 1, 4><<<grid, 256>>>(x, w, b, y, xs.n, xs.c, xs.d, xs.h, xs.w, cout, ys.d, ys.h, ys.w);
    } else if (k == 3 && stride == 2) {
        dim3 grid(nblk(ys.w, TX * 2), nblk(ys.h, TY), nblk(ys.d, TZ) * nblk(cout, COT) * xs.n);
        conv_fwd_k<3, 2, 2><<<grid, 256>>>(x, w, b, y, xs.n, xs.c, xs.d, xs.h, xs.w, cout, ys.d, ys.h, ys.w);
    } else if (k == 1 && stride == 1) {
        dim3 grid(nblk(ys.w, TX * 4), nblk(ys.h, TY), nblk(ys.d, TZ) * nblk(cout, COT) * xs.n);
        conv_fwd_k<1, 1, 4><<<grid, 256>>>(x, w, b, y, xs.n, xs.c, xs.d, xs.h, xs.w, cout, ys.d, ys.h, ys.w);
    }
    else { fprintf(stderr, "nn_conv3d_fwd: unsupported k=%d stride=%d\n", k, stride); abort(); }
    KCHECK();
}
__global__ void transpose_w_k(const float *w, float *wt, int Co, int Ci, int T) {   /* wt[ci][co][t] = w[co][ci][t] */
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)Co * Ci * T) return;
    int t = (int)(i % T), ci = (int)((i / T) % Ci), co = (int)(i / ((size_t)T * Ci));
    wt[((size_t)ci * Co + co) * T + t] = w[i];
}
__global__ void flip_w_k(const float *w, float *wt, int Co, int Ci, int T) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)Co * Ci * T) return;
    int t = (int)(i % T), ci = (int)((i / T) % Ci), co = (int)(i / ((size_t)T * Ci));
    wt[((size_t)ci * Co + co) * T + (T - 1 - t)] = w[i];
}
__global__ void dilate2_k(const float *gy, float *gd, int N, int C, int D, int H, int W, int Do, int Ho, int Wo) {
    size_t Si = (size_t)D * H * W;
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)N * C * Si) return;
    int x = (int)(i % W), y = (int)((i / W) % H), z = (int)((i / ((size_t)W * H)) % D), nc = (int)(i / Si);
    float v = 0.f;
    if (!(x & 1) && !(y & 1) && !(z & 1) && x / 2 < Wo && y / 2 < Ho && z / 2 < Do) v = gy[((size_t)nc * Do + z / 2) * Ho * Wo + (size_t)(y / 2) * Wo + x / 2];
    gd[i] = v;
}
extern "C" size_t nn_conv3d_scratch(shape5 xs, int cout, int k) {
    size_t w = (size_t)cout * xs.c * k * k * k * sizeof(float);
    return g_tf32 ? w + 256 : w + (size_t)xs.n * cout * shape_spatial(xs) * sizeof(float);
}
void bwd_data_impl(const float *gy, shape5 ys, const float *w, shape5 xs, int k, int stride, float *gx, float *scratch, int accum) {
    int save = g_pass; g_pass = 1; bwd_data_impl_(gy, ys, w, xs, k, stride, gx, scratch, accum); g_pass = save;
}
void bwd_data_impl_(const float *gy, shape5 ys, const float *w, shape5 xs, int k, int stride, float *gx, float *scratch, int accum) {
    int T = k * k * k;
    if (k == 1 || !g_tf32 || (stride == 2 && (ISMX(gy) || ISMX(gx)))) exec_prec(1, 0);
    if (stride == 1) {
        size_t nw = (size_t)ys.c * xs.c * T;
        flip_w_k<<<nblk(nw, 256), 256>>>(w, scratch, ys.c, xs.c, T);
        /* gy has shape ys; treat it as input with ys.c channels, "cout" = xs.c. Spatial sizes equal for stride 1 / pad k/2. */
        if (k == 3 && g_tf32) { gnp_t none = {nullptr, nullptr, nullptr, nullptr, 0}; split_t ns = {nullptr, 0, nullptr, 0, accum}; conv_fwd_tc(gy, GBF, ys, scratch, nullptr, xs.c, gx, GBF, none, nullptr, 0, ns); }
        else if (k == 1 && g_tf32 && ISMX(gx)) lp_conv1_to_mx(gy, ISMX(gy) ? 3 : LPDT(GBF), xs.n, ys.c, shape_spatial(xs), scratch, xs.c, gx);   /* head: logit gradient -> MX gout */
        else if (k == 1 && GBF) { size_t S = shape_spatial(xs); if (g_h16) conv1_f_k<f16, f16><<<nblk((size_t)xs.n * S, 256), 256>>>((const f16 *)gy, scratch, nullptr, (f16 *)gx, xs.n, ys.c, xs.c, S); else conv1_f_k<bf16, bf16><<<nblk((size_t)xs.n * S, 256), 256>>>((const bf16 *)gy, scratch, nullptr, (bf16 *)gx, xs.n, ys.c, xs.c, S); }
        else { int save = g_actbf; g_actbf = 0; nn_conv3d_fwd(gy, ys, scratch, nullptr, xs.c, k, 1, gx); g_actbf = save; }
    } else if (stride == 2 && k == 3 && g_tf32 && (ISMX(gy) || ISMX(gx))) {   /* MX gradients: direct voxel-major kernel, accumulate in MX */
        if (!ISMX(gy) || !ISMX(gx)) { fprintf(stderr, "bwd_data s2: MX gradient storage needs MX gy and gx\n"); abort(); }
        lp_bwd_data_s2_mx(gy, ys, w, xs, gx, accum);
    } else if (stride == 2 && k == 3 && g_tf32 && !ufsm_env_on("UFSM_S2DIL") && !(xs.d & 1) && !(xs.h & 1) && !(xs.w & 1)) {
        /* parity decomposition: gx[2m + p] = sum over the taps compatible with parity p of w . gy[m + d]; each of the
           8 parity classes is a stride-1 conv on the gy grid with 1..8 taps (27 total: no wasted MACs) */
        size_t nw = (size_t)ys.c * xs.c * T;
        transpose_w_k<<<nblk(nw, 256), 256>>>(w, scratch, ys.c, xs.c, T);
        gnp_t none = {nullptr, nullptr, nullptr, nullptr, 0};
        split_t ns = {nullptr, 0, nullptr, 0, accum};
        static int nofuse = -1;
        if (nofuse < 0) nofuse = ufsm_env_on("UFSM_S2B_NOFUSE");
        const int fprec = eff_prec();
        const bool fused = !nofuse && ys.c <= TC_CI;   /* one launch for the 8 classes (gy of <= 16 channels) */
        s2cls_t all = {};
        for (int cls = 0; cls < 8; cls++) {
            tapset_t &ts = all.c[cls];
            ts.pz = cls >> 2; ts.py = (cls >> 1) & 1; ts.px = cls & 1; ts.Dx = xs.d; ts.Hx = xs.h; ts.Wx = xs.w;
            /* per axis: even output -> tap 1 at offset 0; odd -> tap 0 at offset +1 and tap 2 at offset 0 */
            int nz = ts.pz ? 2 : 1, ny = ts.py ? 2 : 1, nx = ts.px ? 2 : 1;
            for (int a = 0; a < nz; a++) for (int bb = 0; bb < ny; bb++) for (int c = 0; c < nx; c++) {
                int tz = ts.pz ? (a ? 2 : 0) : 1, ty = ts.py ? (bb ? 2 : 0) : 1, tx = ts.px ? (c ? 2 : 0) : 1;
                int e = ts.ntap++;
                ts.dz[e] = (signed char)(ts.pz ? (a ? 0 : 1) : 0); ts.dy[e] = (signed char)(ts.py ? (bb ? 0 : 1) : 0); ts.dx[e] = (signed char)(ts.px ? (c ? 0 : 1) : 0);
                ts.wt[e] = (signed char)(tz * 9 + ty * 3 + tx);
            }
            if (!fused) conv_fwd_tc(gy, GBF, ys, scratch, nullptr, xs.c, gx, GBF, none, nullptr, 0, ns, &ts);
        }
        if (fused) {   /* stage the gy tile once, run the 8 classes from it */
            exec_prec(1, fprec == 4 ? 4 : 1);
            if (g_h16) s2b_fused<f16>(gy, ys, xs, gx, scratch, all, accum, fprec == 4);
            else s2b_fused<bf16>(gy, ys, xs, gx, scratch, all, accum, fprec == 4);
        }
    } else if (stride == 2 && k == 3 && !(xs.d & 1) && !(xs.h & 1) && !(xs.w & 1)) {
        /* even input sizes: gx = conv_s1(dilate2(gy), flip(w)) */
        size_t nw = (size_t)ys.c * xs.c * T;
        float *gd = scratch + nw;
        flip_w_k<<<nblk(nw, 256), 256>>>(w, scratch, ys.c, xs.c, T);
        shape5 ds = xs; ds.c = ys.c;
        size_t n = shape_numel(ds);
        dilate2_k<<<nblk(n, 256), 256>>>(gy, gd, xs.n, ys.c, xs.d, xs.h, xs.w, ys.d, ys.h, ys.w);
        if (g_tf32) { gnp_t none = {nullptr, nullptr, nullptr, nullptr, 0}; split_t ns = {nullptr, 0, nullptr, 0, accum}; conv_fwd_tc(gd, 0, ds, scratch, nullptr, xs.c, gx, 0, none, nullptr, 0, ns); }
        else nn_conv3d_fwd(gd, ds, scratch, nullptr, xs.c, k, 1, gx);
    } else {
        size_t n = shape_numel(xs);
        if (k == 3) conv_bwd_data_s2_k<3><<<nblk(n, 256), 256>>>(gy, w, gx, xs.n, xs.c, xs.d, xs.h, xs.w, ys.c, ys.d, ys.h, ys.w);
        else { fprintf(stderr, "nn_conv3d_bwd_data: unsupported k=%d stride=%d\n", k, stride); abort(); }
    }
    KCHECK();
}
extern "C" void nn_conv3d_bwd_data(const float *gy, shape5 ys, const float *w, shape5 xs, int k, int stride, float *gx, float *scratch) { bwd_data_impl(gy, ys, w, xs, k, stride, gx, scratch, 0); }
extern "C" int nn_conv3d_bwd_data_acc(const float *gy, shape5 ys, const float *w, shape5 xs, int k, int stride, float *gx, float *scratch) {
    if (!(g_tf32 && k == 3 && (stride == 1 || (stride == 2 && !(xs.d & 1) && !(xs.h & 1) && !(xs.w & 1))))) return -1;
    bwd_data_impl(gy, ys, w, xs, k, stride, gx, scratch, 1);
    return 0;
}
