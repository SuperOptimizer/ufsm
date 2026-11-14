/* ufsm z3-put <array-dir> <in.raw> --shape Z,Y,X --z0 Z0 --nz NZ [--shard 128] [--q 2] [--threads N]
   Planes z0 .. z0+nz-1 of a uint8 array, given as a raw file of nz full Y x X planes (C order), written as whole shards of the
   volcomp zarr v3 array at array-dir (z3w layout; zarr.json (re)written with these parameters, other shards untouched). z0 must be
   a multiple of the shard and nz too unless the slab ends the array, so a long build can write slab by slab from separate processes
   (pub1667 7.91 um fill store, scripts/pub1667/fill791.py). Shards entirely zero are not written (z3w omits them). */
#include "sysinfo.h"
#include "z3w.h"
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

int cmd_z3put(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: ufsm z3-put <array-dir> <in.raw> --shape Z,Y,X --z0 Z0 --nz NZ [--shard 128] [--q 2] [--threads N]\n");
        return 2;
    }
    long long Z = 0, Y = 0, X = 0, z0 = -1, nz = 0;
    int shard = 128, nth = ufsm_threads(0, 32);
    double q = 2;
    for (int a = 4; a + 1 < argc; a++) {
        if (!strcmp(argv[a], "--shape")) sscanf(argv[a + 1], "%lld,%lld,%lld", &Z, &Y, &X);
        else if (!strcmp(argv[a], "--z0")) z0 = atoll(argv[a + 1]);
        else if (!strcmp(argv[a], "--nz")) nz = atoll(argv[a + 1]);
        else if (!strcmp(argv[a], "--shard")) shard = atoi(argv[a + 1]);
        else if (!strcmp(argv[a], "--q")) q = atof(argv[a + 1]);
        else if (!strcmp(argv[a], "--threads")) nth = atoi(argv[a + 1]);
    }
    if (Z <= 0 || Y <= 0 || X <= 0 || z0 < 0 || nz <= 0 || z0 + nz > Z || shard <= 0 || shard % 128 || z0 % shard ||
        (nz % shard && z0 + nz != Z)) {
        fprintf(stderr, "z3-put: need --shape, --z0 (a multiple of the shard), --nz (a multiple of the shard unless it ends the array)\n");
        return 2;
    }
    int fd = open(argv[3], O_RDONLY);
    struct stat st;
    if (fd < 0 || fstat(fd, &st) || st.st_size != (off_t)(nz * Y * X)) {
        fprintf(stderr, "z3-put: %s: expected %lld bytes\n", argv[3], nz * Y * X);
        return 1;
    }
    const uint8_t *in = mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (in == MAP_FAILED) { perror("mmap"); return 1; }
    const int64_t shape[3] = {Z, Y, X};
    z3w *w = z3w_create(argv[2], shape, shard, (float)q, 0, nullptr);
    if (!w) { fprintf(stderr, "z3-put: %s\n", z3w_error()); return 1; }
    const size_t S = (size_t)shard;
    uint8_t *buf = malloc(S * S * S);
    int rc = 0;
    long long written = 0;
    for (long long sz = z0 / shard; sz * shard < z0 + nz && !rc; sz++)
        for (long long sy = 0; sy * shard < Y && !rc; sy++)
            for (long long sx = 0; sx * shard < X && !rc; sx++) {
                int any = 0;
                memset(buf, 0, S * S * S);
                for (size_t k = 0; k < S; k++) {
                    long long z = sz * shard + (long long)k - z0;
                    if (z >= nz) break;
                    for (size_t j = 0; j < S; j++) {
                        long long y = sy * shard + (long long)j;
                        if (y >= Y) break;
                        long long n = X - sx * shard < (long long)S ? X - sx * shard : (long long)S;
                        const uint8_t *src = in + ((size_t)z * (size_t)Y + (size_t)y) * (size_t)X + (size_t)(sx * shard);
                        uint8_t *dst = buf + (k * S + j) * S;
                        memcpy(dst, src, (size_t)n);
                        for (long long i = 0; i < n && !any; i++) any = dst[i] != 0;
                    }
                }
                if (!any) continue;
                if (z3w_write_shard(w, sz, sy, sx, buf, nth)) { fprintf(stderr, "z3-put: %s\n", z3w_error()); rc = 1; }
                written++;
            }
    free(buf);
    z3w_close(w);
    munmap((void *)in, (size_t)st.st_size);
    close(fd);
    if (!rc) printf("z3-put: %s planes %lld..%lld, %lld shards written\n", argv[2], z0, z0 + nz - 1, written);
    return rc;
}
