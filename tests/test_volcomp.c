/* volcomp lossless (VLL) encoder token buffers (third_party/volcomp.h VLL_BLK_TOKS): a substream of 15 DENSE blocks and one
   SPARSE block with the most nonzero residuals (2048, each VLL_NLIT, so SPARSE is the cheaper mode) is 15 x 4097 + 4098 tokens
   plus 2 decoder states, one more than the per-substream field buffer once held: a 4-byte heap overflow ("free(): invalid next
   size" in phase-raster --gs 4 on Paris4). The scratch stays on the heap (no mmap) so glibc's free checks catch an overflow
   without ASan; the chunk must round-trip. */
#define VOLCOMP_IMPLEMENTATION
#include "volcomp.h"
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    mallopt(M_MMAP_THRESHOLD, 32 << 20);
    uint8_t *chunk = malloc(VOLCOMP_CHUNK_VOXELS), *back = malloc(VOLCOMP_CHUNK_VOXELS), *enc = malloc(VOLCOMP_ENCODE_BOUND);
    if (!chunk || !back || !enc) return 1;
    memset(chunk, 7, VOLCOMP_CHUNK_VOXELS);
    for (uint32_t bi = 0; bi < VF_BLOCKS_PER_SUB; bi++) {   /* substream 0: residual 1 everywhere (DENSE), then the SPARSE block */
        uint8_t u[VF_BLKV], b[VF_BLKV];
        for (uint32_t i = 0; i < VF_BLKV; i++) u[i] = bi + 1 < VF_BLOCKS_PER_SUB ? 1 : (i & 1) ? VLL_NLIT : 0;
        vll_rebuild(b, u);
        vll_scatter(chunk, bi >> 6, (bi >> 3) & 7u, bi & 7u, b);
    }
    size_t n = 0;
    volcomp_status st = volcomp_encode(chunk, VOLCOMP_Q_LOSSLESS, enc, VOLCOMP_ENCODE_BOUND, &n);
    if (st != VOLCOMP_OK) { fprintf(stderr, "FAIL: encode: %s\n", volcomp_status_string(st)); return 1; }
    st = volcomp_decode(enc, n, back, VOLCOMP_CHUNK_VOXELS);
    if (st != VOLCOMP_OK || memcmp(chunk, back, VOLCOMP_CHUNK_VOXELS)) { fprintf(stderr, "FAIL: lossless round trip (%s)\n", volcomp_status_string(st)); return 1; }
    free(chunk); free(back); free(enc);
    printf("test_volcomp: ok (%zu bytes)\n", n);
    return 0;
}
