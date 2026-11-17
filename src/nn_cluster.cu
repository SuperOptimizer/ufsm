// thread-block clusters (sm_90 on; mcc lowers __cluster_dims__ to .reqnctapercluster): the self-test of tests/test_cluster.c. Each block
// of a 1 x 1 x N cluster writes (its rank, its thread) into the next rank's shared memory through mapa + st.shared::cluster, between
// two cluster barriers, and reads back what its predecessor wrote; N = 2 / 3 / 4 / 6 (the card's 3 GPCs of 12 SMs hold 36 CTAs in
// clusters of those; 8 leaves a third idle) and dynamic shared memory up to ~99 KB are the shapes the fp4 wgrad (conv_bwd_w_f4ws_k's
// co-tile CTAs sharing one staged x tile) uses. Mode 1: the same exchange by st.async with an mbarrier's transaction count (the
// receiver waits on its own barrier; try_wait in a braced asm loop), then every block arrives on each cluster member's barrier
// remotely (mbarrier.arrive.release.cluster). ARCH=86 builds it without (returns -2).
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define CLU_ON 1
#else
#define CLU_ON 0
#endif

template <int N, int M>
__global__ void __cluster_dims__(1, 1, N) __launch_bounds__(128, 1) clu_k(unsigned *out) {
    extern __shared__ unsigned dyn[];
    __shared__ __align__(16) unsigned s[128];
    __shared__ __align__(8) unsigned long long mb;
    const size_t at = ((size_t)(blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * 128 + threadIdx.x;
#if CLU_ON
    unsigned r, nr;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(nr));
    s[threadIdx.x] = 0;
    dyn[threadIdx.x] = threadIdx.x;   /* (the dynamic shared memory is there and usable beside the cluster) */
    const unsigned mba = (unsigned)__cvta_generic_to_shared(&mb);
    if (M == 1 && threadIdx.x == 0) {   /* the barrier: one arrival (this thread's expect_tx) and 128 words of transactions */
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(mba));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(mba), "r"(128u * 4u) : "memory");
    }
    asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
    unsigned ra;
    const unsigned la = (unsigned)__cvta_generic_to_shared(&s[threadIdx.x]), nx = (r + 1) % nr;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(la), "r"(nx));
    if (M == 0) {
        asm volatile("st.shared::cluster.u32 [%0], %1;" ::"r"(ra), "r"(1000u * r + threadIdx.x) : "memory");
        asm volatile("fence.acq_rel.cluster;" ::: "memory");
    } else {
        unsigned rmb;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rmb) : "r"(mba), "r"(nx));
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.u32 [%0], %1, [%2];" ::"r"(ra), "r"(1000u * r + threadIdx.x), "r"(rmb)
                     : "memory");
        asm volatile("{\n\t.reg .pred p;\n\tW: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], 0;\n\t@!p bra W;\n\t}" ::"r"(mba)
                     : "memory");
    }
    asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
    unsigned v = (s[threadIdx.x] | nr << 24) + dyn[threadIdx.x] - threadIdx.x;
    if (M == 1) {   /* phase 2: every block arrives on each member's barrier, re-armed for nr arrivals */
        if (threadIdx.x == 0) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(mba), "r"(nr));
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
        asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
        if (threadIdx.x < nr) {
            unsigned rmb;
            asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rmb) : "r"(mba), "r"(threadIdx.x));
            asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::"r"(rmb) : "memory");
        }
        asm volatile("{\n\t.reg .pred p;\n\tW: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], 0;\n\t@!p bra W;\n\t}" ::"r"(mba)
                     : "memory");
        asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
    }
    out[at] = v;
#else
    out[at] = 0xffffffffu;
#endif
}

template <int N, int M> static cudaError_t clu_launch(unsigned *d, dim3 g, size_t dsm) {
    cudaError_t e = cudaSuccess;
    if (dsm > 48 * 1024) e = cudaFuncSetAttribute(clu_k<N, M>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dsm);
    if (!e) { clu_k<N, M><<<g, 128, dsm>>>(d); e = cudaGetLastError(); }
    return e;
}

/* N = 2, 3, 4 or 6 blocks per cluster, dsm bytes of dynamic shared memory, mode 0 (st.shared::cluster) or 1 (st.async + mbarriers,
   remote arrivals): 0 and *nbad = the wrong words, -2 not built for clusters (or a card without cluster launch), else the CUDA error */
extern "C" int nn_cluster_selftest(int N, int mode, size_t dsm, int *nbad) {
    int dev = 0, can = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&can, cudaDevAttrClusterLaunch, dev);
    if (!can) return -2;
    const dim3 g(2, 3, 12);   /* (z a multiple of every N) */
    const size_t n = (size_t)g.x * g.y * g.z * 128;
    unsigned *d = nullptr, *h = (unsigned *)malloc(n * 4);
    if (!h) return 2;
    cudaError_t e = cudaMalloc(&d, n * 4);
    if (!e) e = cudaMemset(d, 0, n * 4);
    if (!e) {
        const int k = N * 2 + (mode != 0);
        e = k == 4 ? clu_launch<2, 0>(d, g, dsm) : k == 5 ? clu_launch<2, 1>(d, g, dsm) : k == 6 ? clu_launch<3, 0>(d, g, dsm)
          : k == 7 ? clu_launch<3, 1>(d, g, dsm) : k == 8 ? clu_launch<4, 0>(d, g, dsm) : k == 9 ? clu_launch<4, 1>(d, g, dsm)
          : k == 12 ? clu_launch<6, 0>(d, g, dsm) : k == 13 ? clu_launch<6, 1>(d, g, dsm) : cudaErrorInvalidValue;
    }
    if (!e) e = cudaDeviceSynchronize();
    if (!e) e = cudaMemcpy(h, d, n * 4, cudaMemcpyDeviceToHost);
    cudaFree(d);
    if (e) { free(h); return (int)e; }
    if (h[0] == 0xffffffffu) { free(h); return -2; }
    int bad = 0;
    for (unsigned z = 0; z < g.z; z++) for (unsigned y = 0; y < g.y; y++) for (unsigned x = 0; x < g.x; x++) for (unsigned t = 0; t < 128; t++) {
        const unsigned rank = z % (unsigned)N, want = (unsigned)N << 24 | (1000u * ((rank + N - 1) % N) + t);
        bad += h[((size_t)(z * g.y + y) * g.x + x) * 128 + t] != want;
    }
    free(h);
    *nbad = bad;
    return 0;
}
