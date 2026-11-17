/* thread-block clusters (src/nn_cluster.cu): a __cluster_dims__(1, 1, N) kernel launched with a plain <<<>>> runs in clusters of N blocks,
   each reading the shared memory its cluster neighbour wrote through mapa: st.shared::cluster, and st.async completing on the receiver's
   mbarrier followed by remote mbarrier arrivals; N = 2 / 3 / 4 / 6, 512 B and 90 KB of dynamic shared memory (the fp4 wgrad's shapes).
   SKIP on a build or card without clusters (ARCH=86, pre-sm_90). */
#include "nn.h"
#include <stdio.h>

int main(void) {
    static const int ns[] = {2, 3, 4, 6};
    static const size_t dsms[] = {512, 90 * 1024};
    int fail = 0;
    for (int m = 0; m < 2; m++) for (int k = 0; k < 4; k++) for (int q = 0; q < 2; q++) {
        int bad = -1;
        const int r = nn_cluster_selftest(ns[k], m, dsms[q], &bad);
        if (r == -2) { printf("SKIP: no thread-block clusters in this build / on this card\n"); return 0; }
        printf("cluster 1 x 1 x %d, %s, %zu B dynamic shared: %s (rc %d, %d wrong words)\n", ns[k], m ? "st.async + mbarrier" : "st.shared::cluster",
               dsms[q], !r && !bad ? "ok" : "FAIL", r, bad);
        fail |= r != 0 || bad != 0;
    }
    printf(fail ? "test_cluster: FAIL\n" : "test_cluster: ok\n");
    return fail;
}
