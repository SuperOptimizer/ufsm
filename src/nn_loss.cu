/* CUDA ops: loss section of the former nn.cu */
#include "nn_common.cuh"

float g_posw = 1.f;
extern "C" void nn_set_pos_weight(float w) { g_posw = w; }
int g_tol = 0;
extern "C" void nn_set_loss_tol(int r) { g_tol = r < 0 ? 0 : r > 2 ? 2 : r; }
extern "C" int nn_get_loss_tol(void) { return g_tol; }
__global__ void loss_stats_k(const float *lg, const uint8_t *t, const uint8_t *m, const uint8_t *w, int C, size_t S, double *ds, float pw,
                             int tol, uint8_t *code, int D, int H, int W) {
    int nc = blockIdx.x, slab = blockIdx.y, n = nc / C;
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0;   /* per-thread fp32 partials, fp64 block reduction */
    if (w[nc]) {
        const float *l = lg + (size_t)nc * S; const uint8_t *tp = t + (size_t)nc * S, *mp = m + (size_t)n * S;
        uint8_t *cp = tol && nc % C == 0 ? code + (size_t)n * S : nullptr;
        size_t per = (S + KSLAB - 1) / KSLAB, lo = (size_t)slab * per, hi = lo + per < S ? lo + per : S;
        for (size_t i = lo + threadIdx.x; i < hi; i += blockDim.x) {
            if (!mp[i]) continue;
            float x = l[i], p = tp[i] * (1.f / 255.f);
            float sg = 1.f / (1.f + __expf(-x));
            float spp = fmaxf(x, 0.f) + log1pf(__expf(-fabsf(x)));   /* softplus(x) */
            float xq = x, sq = sg, spq = spp;
            if (cp && tp[i]) {
                uint8_t c8; xq = tol_max(l, mp, i, D, H, W, tol, &c8); cp[i] = c8;
                sq = 1.f / (1.f + __expf(-xq)); spq = fmaxf(xq, 0.f) + log1pf(__expf(-fabsf(xq)));
            }
            float bce = pw * p * (spq - xq) + (1.f - p) * spp;          /* softplus(-x) = softplus(x) - x */
            a0 += 1; a1 += bce; a2 += sq * p; a3 += sg; a4 += p;
        }
    }
    __shared__ double r[5][256];
    r[0][threadIdx.x] = a0; r[1][threadIdx.x] = a1; r[2][threadIdx.x] = a2; r[3][threadIdx.x] = a3; r[4][threadIdx.x] = a4;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) { if (threadIdx.x < o) for (int k = 0; k < 5; k++) r[k][threadIdx.x] += r[k][threadIdx.x + o]; __syncthreads(); }
    if (threadIdx.x == 0) for (int k = 0; k < 5; k++) atomicAdd(&ds[nc * 5 + k], r[k][0]);
}
__global__ void loss_d2f_k(const double *d, float *f, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) f[i] = (float)d[i]; }
__global__ void loss_fin_k(const float *st, const uint8_t *w, int N, int C, float *fin) {
    if (threadIdx.x || blockIdx.x) return;
    int active = 0, cnt[16] = {0};
    for (int c = 0; c < 2 * C + 2; c++) fin[c] = 0.f;
    for (int nc = 0; nc < N * C; nc++) {
        if (!w[nc] || st[nc * 5] < 1.f) continue;
        active++;
        int c = nc % C;
        fin[c] += st[nc * 5 + 1] / st[nc * 5];
        fin[C + c] += 1.f - (2.f * st[nc * 5 + 2] + 1.f) / (st[nc * 5 + 3] + st[nc * 5 + 4] + 1.f);
        cnt[c]++;
    }
    for (int c = 0; c < C; c++) if (cnt[c]) { fin[c] /= cnt[c]; fin[C + c] /= cnt[c]; }
    fin[2 * C] = (float)active;
    fin[2 * C + 1] = active ? 1.f / active : 0.f;
}
int g_loss_g16 = 0;
extern "C" void nn_set_loss_grad_h16(int on) { g_loss_g16 = on; }
extern "C" size_t nn_loss_scratch(shape5 s) { return ((size_t)5 * s.n * s.c + 2 * s.c + 2) * sizeof(float); }
extern "C" void nn_loss_async(const float *logits, const uint8_t *t, const uint8_t *m, const uint8_t *w, shape5 s, float dice_w, float *gl, float *scratch) {
    nn_loss_async_tol(logits, t, m, w, s, dice_w, gl, scratch, nullptr);
}
extern "C" void nn_loss_async_tol(const float *logits, const uint8_t *t, const uint8_t *m, const uint8_t *w, shape5 s, float dice_w, float *gl, float *scratch, uint8_t *code) {
    size_t S = shape_spatial(s);
    int NC = s.n * s.c;
    float *fin = scratch + (size_t)5 * NC;
    double *ds = gn_dsums((size_t)5 * NC);
    const int tol = code ? g_tol : 0;
    if (tol) cudaMemsetAsync(code, 255, (size_t)s.n * S);
    else code = nullptr;
    cudaMemsetAsync(ds, 0, (size_t)5 * NC * sizeof(double));
    loss_stats_k<<<dim3(NC, KSLAB), 256>>>(logits, t, m, w, s.c, S, ds, g_posw, tol, code, s.d, s.h, s.w);
    zs_reduce(ds, 5 * NC);   /* spatial split: statistics of the whole window (the halo planes are masked out by the caller) */
    loss_d2f_k<<<nblk(5 * NC, 128), 128>>>(ds, scratch, 5 * NC);
    loss_fin_k<<<1, 32>>>(scratch, w, s.n, s.c, fin);
    if (gl) {
        size_t n = shape_numel(s);
        if (!g_loss_g16) loss_grad_k<float><<<nblk(n, 256), 256>>>(logits, t, m, w, s.n, s.c, S, scratch, dice_w, fin, gl, g_posw, 1.f, tol, code, s.d, s.h, s.w);
        else if (g_h16) loss_grad_k<f16><<<nblk(n, 256), 256>>>(logits, t, m, w, s.n, s.c, S, scratch, dice_w, fin, (f16 *)gl, g_posw, g_gscale, tol, code, s.d, s.h, s.w);
        else loss_grad_k<bf16><<<nblk(n, 256), 256>>>(logits, t, m, w, s.n, s.c, S, scratch, dice_w, fin, (bf16 *)gl, g_posw, g_gscale, tol, code, s.d, s.h, s.w);
    }
    KCHECK();
}
extern "C" void nn_loss_fetch(const float *scratch, shape5 s, float *out) {
    CK(cudaMemcpy(out, scratch + (size_t)5 * s.n * s.c, (size_t)(2 * s.c + 1) * sizeof(float), cudaMemcpyDeviceToHost));
}
extern "C" void nn_loss(const float *logits, const uint8_t *t, const uint8_t *m, const uint8_t *w, shape5 s, float dice_w,
                        float *gl, float *out, float *scratch) {
    nn_loss_async(logits, t, m, w, s, dice_w, gl, scratch);
    nn_loss_fetch(scratch, s, out);
}
__global__ void sheet_gather_k(const float *lg,shape5 s,const float *xyz,size_t np,int z0,int lo,int hi,double *out) {
    size_t p=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if (p>=np) return;
    int iz=(int)floorf(xyz[3*p]),iy=(int)floorf(xyz[3*p+1]),ix=(int)floorf(xyz[3*p+2]);
    double fz=xyz[3*p]-iz,fy=xyz[3*p+1]-iy,fx=xyz[3*p+2]-ix;
    double a=0,b=0; size_t S=(size_t)s.d*s.h*s.w;
    for (int dz=0;dz<2;dz++) for (int dy=0;dy<2;dy++) for (int dx=0;dx<2;dx++) {
        int gz=iz+dz,z=gz-z0,y=iy+dy,x=ix+dx;
        if (gz<lo || gz>=hi || z<0 || z>=s.d || y<0 || y>=s.h || x<0 || x>=s.w) continue;
        double w=(dz?fz:1-fz)*(dy?fy:1-fy)*(dx?fx:1-fx);
        if (w==0) continue; size_t k=((size_t)z*s.h+y)*s.w+x;
        a+=w*lg[k]; b+=w*lg[S+k];
    }
    out[2*p]=a; out[2*p+1]=b;
}
extern "C" void nn_sheet_gather(const float *lg,shape5 s,const float *coords,size_t np,int z0,int lo,int hi,double *values) {
    if (!np) return;
    float *xyz; double *v; CK(cudaMalloc(&xyz,3*np*sizeof(float))); CK(cudaMalloc(&v,2*np*sizeof(double)));
    CK(cudaMemcpy(xyz,coords,3*np*sizeof(float),cudaMemcpyHostToDevice));
    sheet_gather_k<<<nblk(np,128),128>>>(lg,s,xyz,np,z0,lo,hi,v);
    zs_reduce(v,(int)(2*np));
    CK(cudaMemcpy(values,v,2*np*sizeof(double),cudaMemcpyDeviceToHost));
    CK(cudaFree(xyz)); CK(cudaFree(v)); KCHECK();
}
extern "C" void nn_sheet_scatter(float *gl,const uint64_t *indices,const float *values,size_t n,int h16) {
    if (!n) return; uint64_t *ix; float *v;
    CK(cudaMalloc(&ix,n*sizeof(uint64_t))); CK(cudaMalloc(&v,n*sizeof(float)));
    CK(cudaMemcpy(ix,indices,n*sizeof(uint64_t),cudaMemcpyHostToDevice)); CK(cudaMemcpy(v,values,n*sizeof(float),cudaMemcpyHostToDevice));
    if (!h16) sheet_scatter_k<float><<<nblk(n,256),256>>>(gl,ix,v,n,1.f);
    else if (g_h16) sheet_scatter_k<f16><<<nblk(n,256),256>>>((f16 *)gl,ix,v,n,g_gscale);
    else sheet_scatter_k<bf16><<<nblk(n,256),256>>>((bf16 *)gl,ix,v,n,g_gscale);
    CK(cudaFree(ix)); CK(cudaFree(v)); KCHECK();
}
extern "C" void nn_sheet_input(void *input,int W,const float *rows,float center,float scale,int h16) {
    float *r; CK(cudaMalloc(&r,4*(size_t)W*sizeof(float))); CK(cudaMemcpy(r,rows,4*(size_t)W*sizeof(float),cudaMemcpyHostToDevice));
    size_t n=(size_t)W*W*W;
    if (!h16) sheet_input_k<float><<<nblk(n,256),256>>>((float *)input,W,r,center,scale);
    else if (g_h16) sheet_input_k<f16><<<nblk(n,256),256>>>((f16 *)input,W,r,center,scale);
    else sheet_input_k<bf16><<<nblk(n,256),256>>>((bf16 *)input,W,r,center,scale);
    CK(cudaFree(r)); KCHECK();
}
__global__ void sheet_gate_k(float *r,const float *surface,const uint8_t *ct,int W,const float *rows) {
    size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;
    if (i>=(size_t)W*W*W) return;
    if (!ct[i] || surface[i]<-2.19722458f) r[i]=nanf("");
    else {
        int z=i/((size_t)W*W),y=(i/W)%W,x=i%W;
        const float *a=rows+4*z;
        r[i]+=hypotf(y+a[0],x+a[1])*a[2]+a[3];
    }
}
extern "C" void nn_sheet_gate(float *r,const float *surface,const uint8_t *ct,int W,const float *rows) {
    float *a; CK(cudaMalloc(&a,4*(size_t)W*sizeof(float))); CK(cudaMemcpy(a,rows,4*(size_t)W*sizeof(float),cudaMemcpyHostToDevice));
    sheet_gate_k<<<nblk((size_t)W*W*W,256),256>>>(r,surface,ct,W,a);
    CK(cudaFree(a)); KCHECK();
}
extern "C" void nn_peer_copy(void *dst, int dst_dev, const void *src, int src_dev, size_t bytes) {
    static char enabled[8][8];
    if (!enabled[dst_dev & 7][src_dev & 7]) {
        int can = 0; cudaDeviceCanAccessPeer(&can, dst_dev, src_dev);
        if (can) { int cur; cudaGetDevice(&cur); cudaSetDevice(dst_dev); cudaDeviceEnablePeerAccess(src_dev, 0); cudaGetLastError(); cudaSetDevice(cur); }
        enabled[dst_dev & 7][src_dev & 7] = 1;
    }
    CK(cudaMemcpyPeer(dst, dst_dev, src, src_dev, bytes));
}
int zsegs(const void *p, shape5 s, int esz, zseg_t *sg) {
    const size_t HW = (size_t)s.h * s.w, S = HW * s.d;
    if (esz) { sg[0] = {0, (size_t)s.n * s.c, HW * esz, S * esz}; return 1; }
    const int dt = nn_storage(p), bw = s.c <= 16 ? 16 : 32, nb = (s.c + bw - 1) / bw, rb = dt == 4 ? bw / 2 : bw;
    if (dt != 4 && dt != 8) { fprintf(stderr, "split: tensor %p has no MX registration\n", p); abort(); }
    sg[0] = {0, (size_t)s.n * nb, HW * rb, S * rb};
    sg[1] = {(size_t)s.n * nb * S * rb, (size_t)s.n * nb, HW, S};
    return 2;
}
size_t zsegs_plane_bytes(const zseg_t *sg, int ns) { size_t b = 0; for (int i = 0; i < ns; i++) b += sg[i].rows * sg[i].pb; return b; }
void zsegs_copy(void *t, const zseg_t *sg, int ns, int z0, int nz, void *c, int dir) {
    char *cb = (char *)c;
    for (int i = 0; i < ns; i++) {
        char *tp = (char *)t + sg[i].base + (size_t)z0 * sg[i].pb;
        const size_t w = (size_t)nz * sg[i].pb;
        if (dir) CK(cudaMemcpy2DAsync(tp, sg[i].pitch, cb, w, w, sg[i].rows, cudaMemcpyDeviceToDevice, 0));
        else CK(cudaMemcpy2DAsync(cb, w, tp, sg[i].pitch, w, sg[i].rows, cudaMemcpyDeviceToDevice, 0));
        cb += w * sg[i].rows;
    }
}
void zsegs_zero(void *t, const zseg_t *sg, int ns, int z0, int nz) {
    if (nz <= 0) return;
    for (int i = 0; i < ns; i++) CK(cudaMemset2DAsync((char *)t + sg[i].base + (size_t)z0 * sg[i].pb, sg[i].pitch, 0, (size_t)nz * sg[i].pb, sg[i].rows, 0));
}
extern "C" void nn_split_zero(const void *t, shape5 s, int esz, int lo, int hi) {
    zseg_t sg[2]; int ns = zsegs(t, s, esz, sg);
    zsegs_zero((void *)t, sg, ns, 0, lo); zsegs_zero((void *)t, sg, ns, s.d - hi, hi);
    KCHECK();
}
cudaEvent_t zs_ev(int dev, int k) {   /* per-device events for the cross-GPU ordering */
    static cudaEvent_t ev[8][4]; static int init[8];
    if (!init[dev]) { int cur; cudaGetDevice(&cur); cudaSetDevice(dev); for (int i = 0; i < 4; i++) cudaEventCreateWithFlags(&ev[dev][i], cudaEventDisableTiming); cudaSetDevice(cur); init[dev] = 1; }
    return ev[dev][k];
}
void zs_xbar(const int *dev, int k) {
    int cur; cudaGetDevice(&cur);
    for (int i = 0; i < 2; i++) { cudaSetDevice(dev[i]); CK(cudaEventRecord(zs_ev(dev[i], k), 0)); }
    for (int i = 0; i < 2; i++) { cudaSetDevice(dev[i]); CK(cudaStreamWaitEvent(0, zs_ev(dev[1 - i], k), 0)); }
    cudaSetDevice(cur);
}
extern "C" size_t nn_split_halo_bytes(shape5 s, int esz) {   /* bytes of one plane (send / receive buffer size) */
    const size_t HW = (size_t)s.h * s.w;
    if (esz) return (size_t)s.n * s.c * HW * esz;
    const int bw = s.c <= 8 ? 8 : s.c <= 16 ? 16 : 32, nb = (s.c + bw - 1) / bw;
    return (size_t)s.n * nb * HW * (bw + 1);
}
cudaStream_t zs_comm(int dev) {
    static cudaStream_t st[8];
    if (!st[dev]) { int cur; cudaGetDevice(&cur); cudaSetDevice(dev); cudaStreamCreateWithFlags(&st[dev], cudaStreamNonBlocking); cudaSetDevice(cur); }
    return st[dev];
}
cudaEvent_t zs_hev(int dev, int slot, int k) {   /* 0 packed, 1 arrived, 2 unpacked */
    static cudaEvent_t ev[8][2][3]; static int init[8];
    if (!init[dev]) { int cur; cudaGetDevice(&cur); cudaSetDevice(dev); for (int i = 0; i < 6; i++) cudaEventCreateWithFlags(&ev[dev][i / 3][i % 3], cudaEventDisableTiming); cudaSetDevice(cur); init[dev] = 1; }
    return ev[dev][slot & 1][k];
}
extern "C" void nn_split_halo_begin(void *const *t, const int *dev, shape5 s, int esz, int h, void *const *sb, void *const *rb, int slot) {
    int cur; cudaGetDevice(&cur);
    zseg_t sg[2][2]; int ns = 0;
    for (int i = 0; i < 2; i++) { cudaSetDevice(dev[i]); ns = zsegs(t[i], s, esz, sg[i]); }
    const size_t nbytes = zsegs_plane_bytes(sg[0], ns);
    const int D = s.d;
    /* side 0 sends its last own plane D - h - 1, side 1 its first own plane h; the outer halo planes are zeroed */
    const int zsend[2] = {D - h - 1, h}, zlo[2] = {D - h + 1, 0};
    for (int i = 0; i < 2; i++) {
        cudaSetDevice(dev[i]);
        CK(cudaStreamWaitEvent(0, zs_hev(dev[1 - i], slot, 1), 0));   /* the previous exchange has read this side's send buffer */
        zsegs_copy(t[i], sg[i], ns, zsend[i], 1, sb[i], 0);
        zsegs_zero(t[i], sg[i], ns, zlo[i], h - 1);
        CK(cudaEventRecord(zs_hev(dev[i], slot, 0), 0));
    }
    for (int i = 0; i < 2; i++) {
        cudaSetDevice(dev[i]);
        cudaStream_t cs = zs_comm(dev[i]);
        CK(cudaStreamWaitEvent(cs, zs_hev(dev[1 - i], slot, 0), 0));
        CK(cudaStreamWaitEvent(cs, zs_hev(dev[i], slot, 2), 0));        /* the previous exchange has unpacked this receive buffer */
        CK(cudaMemcpyPeerAsync(rb[i], dev[i], sb[1 - i], dev[1 - i], nbytes, cs));
        CK(cudaEventRecord(zs_hev(dev[i], slot, 1), cs));
    }
    cudaSetDevice(cur);
    KCHECK();
}
extern "C" void nn_split_halo_end(void *t, int side, shape5 s, int esz, int h, void *rb, int slot) {   /* current device = this side's */
    zseg_t sg[2]; int ns = zsegs(t, s, esz, sg);
    const int dev = cur_dev();
    CK(cudaStreamWaitEvent(0, zs_hev(dev, slot, 1), 0));
    zsegs_copy(t, sg, ns, side ? h - 1 : s.d - h, 1, rb, 1);
    CK(cudaEventRecord(zs_hev(dev, slot, 2), 0));
    KCHECK();
}
__global__ void zs_add_k(double *a, const double *b, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) a[i] += b[i]; }
extern "C" void nn_split_allreduce(double *const *b, const int *dev, int n, double *const *r) {
    int cur; cudaGetDevice(&cur);
    zs_xbar(dev, 2);
    for (int i = 0; i < 2; i++) { cudaSetDevice(dev[i]); CK(cudaMemcpyPeerAsync(r[i], dev[i], b[1 - i], dev[1 - i], (size_t)n * sizeof(double), 0)); }
    zs_xbar(dev, 3);   /* both copies done before either sum overwrites its source */
    for (int i = 0; i < 2; i++) { cudaSetDevice(dev[i]); zs_add_k<<<nblk(n, 128), 128>>>(b[i], r[i], n); }
    cudaSetDevice(cur);
    KCHECK();
}
