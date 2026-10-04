/* Local lossless CT/labels for CLI training, resume and prediction regression tests. */
#include "z3w.h"
#include <stdio.h>
#include <stdlib.h>

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
    free(data); return 0;
}
