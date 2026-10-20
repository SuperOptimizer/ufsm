/* Local lossless CT/labels for CLI training, resume and prediction regression tests. */
#include "z3w.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    const double um = 1; const int64_t shape[3] = {128, 128, 128};
    uint8_t *data = malloc(128 * 128 * 128);
    for (int label = 0; label < 2; label++) {
        char path[1400]; snprintf(path, sizeof path, "%s/%s/1", argv[1], label ? "labels" : "ct");
        z3w *w = z3w_create(path, shape, 128, 0, label ? 255 : 0, nullptr);
        if (!w) return 1;
        for (int z = 0; z < 128; z++) for (int y = 0; y < 128; y++) for (int x = 0; x < 128; x++)
            data[(z * 128 + y) * 128 + x] = label ? ((x % 16 == 8) ? 254 : 0) : (uint8_t)(40 + (z * 3 + y * 7 + x * 11) % 180);
        if (z3w_write_shard(w, 0, 0, 0, data, 2) || z3w_close(w)) return 1;
        snprintf(path, sizeof path, "%s/%s", argv[1], label ? "labels" : "ct");
        if (z3w_write_group(path, &um, 1, label ? "labels" : "ct", nullptr)) return 1;
    }
    /* Coarse binary bands, sampled on the finer CT grid by erosion tests. */
    const int64_t coarse_shape[3] = {64,64,64}; const double coarse_um = 2;
    char path[1400]; snprintf(path, sizeof path, "%s/binary/2", argv[1]);
    z3w *w = z3w_create_mask(path, coarse_shape, 128, nullptr);
    if (!w) return 1;
    for (int z=0; z<128; z++) for (int y=0; y<128; y++) for (int x=0; x<128; x++)
        data[(z*128+y)*128+x] = x%8==3 || x%8==4 ? 255 : 0;
    if (z3w_write_shard(w,0,0,0,data,2) || z3w_close(w)) return 1;
    snprintf(path,sizeof path,"%s/binary",argv[1]);
    if (z3w_write_group(path,&coarse_um,1,"binary",nullptr)) return 1;
    /* predict --full-grid / --resume / --occupancy: a two-level CT (1 and 2 um) of 256 x 128 x 128 whose z < 136 is air */
    const int64_t ashape[3] = {256, 128, 128}, cshape[3] = {128, 64, 64}; const double aum[2] = {1, 2};
    snprintf(path, sizeof path, "%s/ctair/1", argv[1]);
    if (!(w = z3w_create(path, ashape, 128, 0, 0, nullptr))) return 1;
    memset(data, 0, 128 * 128 * 128);
    if (z3w_write_shard(w, 0, 0, 0, data, 2)) return 1;
    for (int z = 0; z < 128; z++) for (int y = 0; y < 128; y++) for (int x = 0; x < 128; x++)
        data[(z * 128 + y) * 128 + x] = z < 8 ? 0 : (uint8_t)(40 + (z * 3 + y * 7 + x * 11) % 180);   /* (z 128..135 air too: the
                                                                                    occupancy margin of one coarse voxel stays zero) */
    if (z3w_write_shard(w, 1, 0, 0, data, 2) || z3w_close(w)) return 1;
    uint8_t *c = calloc(128 * 64 * 64, 1); if (!c) return 1;
    for (int z = 68; z < 128; z++) for (int y = 0; y < 64; y++) for (int x = 0; x < 64; x++) c[(z * 64 + y) * 64 + x] = 100;
    snprintf(path, sizeof path, "%s/ctair/2", argv[1]);
    if (!(w = z3w_create(path, cshape, 128, 0, 0, nullptr))) return 1;
    memset(data, 0, 128 * 128 * 128);
    for (int z = 0; z < 128; z++) for (int y = 0; y < 64; y++) memcpy(data + ((size_t)z * 128 + y) * 128, c + ((size_t)z * 64 + y) * 64, 64);
    if (z3w_write_shard(w, 0, 0, 0, data, 2) || z3w_close(w)) return 1;
    free(c);
    snprintf(path, sizeof path, "%s/ctair", argv[1]);
    if (z3w_write_group(path, aum, 2, "ctair", nullptr)) return 1;
    free(data); return 0;
}
