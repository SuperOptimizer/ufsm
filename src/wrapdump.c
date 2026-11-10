/* ufsm wrap-dump <wraps-dir> <out.f32> --z z0,z1 [--yx y0,y1,x0,x1] [--every 2] [--grid-um U]
   The vertices of every wNNN.sfc wrap inside a z range (and a y / x window) as float32 records (z, y, x, id, column / width), in voxels
   of --grid-um (default: the wraps' own voxel size; voxel index = centre). Every --every-th row and column. For analysis outside the
   pipeline (tools/fibwrap.py: the label-free fibre judge against the final snapped wraps of a slab, without a whole-scroll surf-labels
   render; DESIGN.md 2026-11-10). Single thread; rows are read 64 at a time and filtered per vertex. */
#include <dirent.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "surfcomp.h"

static const char *wd_opt(int argc, char **argv, const char *k, const char *d) { for (int i = 4; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return argv[i + 1]; return d; }

int cmd_wrap_dump(int argc, char **argv) {
    double z0, z1, yx[4] = {-1e30, 1e30, -1e30, 1e30};
    if (argc < 6 || sscanf(wd_opt(argc, argv, "--z", ""), "%lf,%lf", &z0, &z1) != 2) { fprintf(stderr, "usage: ufsm wrap-dump <wraps-dir> <out.f32> --z z0,z1 [--yx y0,y1,x0,x1] [--every 2] [--grid-um U]\n"); return 2; }
    if (*wd_opt(argc, argv, "--yx", "") && sscanf(wd_opt(argc, argv, "--yx", ""), "%lf,%lf,%lf,%lf", &yx[0], &yx[1], &yx[2], &yx[3]) != 4) { fprintf(stderr, "wrap-dump: --yx y0,y1,x0,x1\n"); return 2; }
    const int ev = atoi(wd_opt(argc, argv, "--every", "2")); const double gum = atof(wd_opt(argc, argv, "--grid-um", "0"));
    if (ev < 1) { fprintf(stderr, "wrap-dump: --every >= 1\n"); return 2; }
    DIR *dd = opendir(argv[2]); if (!dd) { fprintf(stderr, "wrap-dump: cannot open %s\n", argv[2]); return 1; }
    FILE *o = fopen(argv[3], "wb"); if (!o) { fprintf(stderr, "wrap-dump: cannot write %s\n", argv[3]); return 1; }
    struct dirent *de; long long tot = 0; int nw = 0;
    while ((de = readdir(dd))) {
        int id; char tail[8] = ""; if (sscanf(de->d_name, "w%d.%7s", &id, tail) != 2 || strcmp(tail, "sfc")) continue;
        char path[2048]; snprintf(path, sizeof path, "%s/%s", argv[2], de->d_name);
        sfc_reader *r; if (sfc_open_file(path, &r)) { fprintf(stderr, "wrap-dump: cannot open %s\n", path); return 1; }
        int c = sfc_find_channel(r, "x"); if (c < 0) c = 0;
        const sfc_channel *ch = sfc_channel_info(r, (uint32_t)c); const int w = (int)ch->width, h = (int)ch->height;
        const uint64_t ml = sfc_metadata_size(r); char *meta = calloc(ml + 1, 1);
        if (ml && sfc_read_metadata(r, meta, ml)) { fprintf(stderr, "wrap-dump: no metadata in %s\n", path); return 1; }
        const char *q = strstr(meta, "\"voxel_size_um\":"); const double um = q ? atof(q + 16) : 0; free(meta);
        if (!(um > 0)) { fprintf(stderr, "wrap-dump: %s: no voxel_size_um\n", path); return 1; }
        const double f = gum > 0 ? um / gum : 1;
        sfc_cache *ca; if (sfc_cache_create(r, (size_t)((w + 63) / 64 + 1) * 3 * 64 * 64 * 12 * 2, &ca)) return 1;
        const int R = 64; float *xyz = malloc((size_t)R * w * 12), *rec = malloc((size_t)R * w * 20); uint8_t *v = malloc((size_t)R * w); long long n = 0;
        for (int r0 = 0; r0 < h; r0 += R) {
            const int nr = r0 + R <= h ? R : h - r0; memset(v, 0, (size_t)nr * w); size_t m = 0;
            if (sfc_cache_read_xyz_region(ca, (uint32_t)c, 0, (uint64_t)r0, (uint32_t)w, (uint32_t)nr, xyz, 12, (size_t)w * 12, v)) { fprintf(stderr, "wrap-dump: %s: read failed\n", path); return 1; }
            for (int rr = (ev - r0 % ev) % ev; rr < nr; rr += ev) for (int cc = 0; cc < w; cc += ev) {
                const size_t i = (size_t)rr * w + cc; if (!v[i] || !(xyz[3 * i + 2] >= 0)) continue;
                const double Z = (xyz[3 * i + 2] + 0.5) * f - 0.5, Y = (xyz[3 * i + 1] + 0.5) * f - 0.5, X = (xyz[3 * i] + 0.5) * f - 0.5;
                if (Z < z0 || Z >= z1 || Y < yx[0] || Y >= yx[1] || X < yx[2] || X >= yx[3]) continue;
                float *g = rec + 5 * m++; g[0] = (float)Z; g[1] = (float)Y; g[2] = (float)X; g[3] = (float)id; g[4] = (float)cc / (float)w;
            }
            if (m && fwrite(rec, 20, m, o) != m) { fprintf(stderr, "wrap-dump: write failed\n"); return 1; }
            n += (long long)m; sfc_cache_clear(ca);
        }
        free(xyz); free(rec); free(v); sfc_cache_destroy(ca); sfc_close(r); tot += n; nw++;
    }
    closedir(dd);
    if (fclose(o)) { fprintf(stderr, "wrap-dump: write failed\n"); return 1; }
    printf("wrap-dump: %d wraps, %lld vertices -> %s\n", nw, tot, argv[3]);
    return 0;
}
