/* ufsm predict: sliding-window inference of a checkpoint over a CT box, written as a volcomp zarr v3
   probability pyramid (uint8 p*255, q=8) in the published layout. */
#include "nn.h"
#include "pyramid.h"
#include "sources.h"
#include "store.h"
#include "unet.h"
#include "z3w.h"
#include "zarr3.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static const char *opt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return dflt;
}

int cmd_predict(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: ufsm predict <ckpt> <root> <ct-group-key> <out-dir> --um U [--level 0] [--box z,y,x,nz,ny,nx] [--window 160]\n"
                        "       [--halo 16] [--shard 512] [--gpu 0] [--cache DIR] [--axis umbilicus.json] [--levels 4] [--q 8] [--threads 16]\n");
        fprintf(stderr, "  window - 2*halo should divide the shard size (160 - 32 = 128 divides 256 and 512): tiles then cover each shard exactly.\n");
        return 2;
    }
    const char *ckpt = argv[2], *root = argv[3], *key = argv[4], *out = argv[5];
    double um = atof(opt(argc, argv, "--um", "0"));
    int level = atoi(opt(argc, argv, "--level", "0")), W = atoi(opt(argc, argv, "--window", "160")), halo = atoi(opt(argc, argv, "--halo", "16"));
    int shard = atoi(opt(argc, argv, "--shard", "512")), gpu = atoi(opt(argc, argv, "--gpu", "0")), nlev = atoi(opt(argc, argv, "--levels", "4"));
    int nthreads = atoi(opt(argc, argv, "--threads", "16"));
    float q = (float)atof(opt(argc, argv, "--q", "8"));
    const char *cache = opt(argc, argv, "--cache", nullptr), *axisf = opt(argc, argv, "--axis", nullptr);
    if (um <= 0) { fprintf(stderr, "--um required\n"); return 2; }
    if (nn_init(gpu)) { fprintf(stderr, "cannot select GPU %d\n", gpu); return 1; }
    unet_cfg cfg; int step;
    if (unet_peek(ckpt, &cfg, &step)) { fprintf(stderr, "cannot read %s\n", ckpt); return 1; }
    unet *u = unet_create(&cfg);
    if (unet_load(u, ckpt) < 0) { fprintf(stderr, "cannot load %s\n", ckpt); return 1; }
    unet_use_ema(u, 1);
    if (W % (1 << (cfg.nlev - 1))) { fprintf(stderr, "window must be divisible by %d\n", 1 << (cfg.nlev - 1)); return 2; }
    store *s = store_open(root);
    char ak[1200];
    snprintf(ak, sizeof ak, "%s/level%d", key, level);
    z3 *ct = pyramid_open_level(s, key, level, um, cache);
    if (!ct) { fprintf(stderr, "cannot open level %d of %s: %s\n", level, key, z3_error()); return 1; }
    const z3_meta *m = z3_meta_of(ct);
    int64_t bo[3] = {0, 0, 0}, bn[3] = {m->shape[0], m->shape[1], m->shape[2]};
    { const char *b = opt(argc, argv, "--box", nullptr); long long v[6]; if (b && sscanf(b, "%lld,%lld,%lld,%lld,%lld,%lld", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5]) == 6) { for (int d = 0; d < 3; d++) { bo[d] = v[d]; bn[d] = v[3 + d]; } } }
    axis ax = {0};
    if (axisf && axis_load(&ax, axisf)) { fprintf(stderr, "cannot load axis %s\n", axisf); return 1; }
    double um_l = um * (1 << level), scale = (double)(1 << level);
    /* output array covers the box, with origin recorded in attributes */
    char lv[32], ldir[1400], attrs[1024];
    z3w_level_name(um_l, lv, sizeof lv);
    snprintf(ldir, sizeof ldir, "%s/%s", out, lv);
    snprintf(attrs, sizeof attrs, "{\"ufsm\":{\"content\":\"recto probability (uint8 = round(p*255))\",\"checkpoint\":\"%s\",\"step\":%d,\"source\":\"%s/%s\",\"origin_zyx\":[%lld,%lld,%lld],\"window\":%d,\"halo\":%d}}",
             ckpt, step, root, ak, (long long)bo[0], (long long)bo[1], (long long)bo[2], W, halo);
    z3w *w = z3w_create(ldir, bn, shard, q, 0, attrs);
    if (!w) { fprintf(stderr, "%s\n", z3w_error()); return 1; }
    int stride = W - 2 * halo;
    size_t w3 = (size_t)W * W * W;
    uint8_t *ctu = malloc(w3), *sbuf = malloc((size_t)shard * shard * shard);
    float *xh = malloc(4 * w3 * sizeof(float)), *ph = malloc(w3 * sizeof(float));
    float *xd = nn_malloc(4 * w3 * 4), *pd = nn_malloc(w3 * 4);
    shape5 xs = {1, 4, W, W, W};
    int64_t ns[3];
    for (int d = 0; d < 3; d++) ns[d] = (bn[d] + shard - 1) / shard;
    double t0 = now(); long ntiles = 0, nskip = 0;
    for (int64_t sz = 0; sz < ns[0]; sz++) for (int64_t sy = 0; sy < ns[1]; sy++) for (int64_t sx = 0; sx < ns[2]; sx++) {
        int64_t so[3] = {sz * shard, sy * shard, sx * shard}, se[3];
        for (int d = 0; d < 3; d++) se[d] = so[d] + shard < bn[d] ? so[d] + shard : bn[d];
        memset(sbuf, 0, (size_t)shard * shard * shard);
        int any = 0;
        for (int64_t tz = so[0] - halo; tz + halo < se[0]; tz += stride)
        for (int64_t ty = so[1] - halo; ty + halo < se[1]; ty += stride)
        for (int64_t tx = so[2] - halo; tx + halo < se[2]; tx += stride) {
            int64_t o[3] = {bo[0] + tz, bo[1] + ty, bo[2] + tx}, n[3] = {W, W, W};
            if (z3_read(ct, o, n, ctu, nthreads)) { fprintf(stderr, "%s\n", z3_error()); return 1; }
            size_t nz = 0; double sum = 0, sq = 0;
            for (size_t k = 0; k < w3; k++) { nz += ctu[k] != 0; sum += ctu[k]; sq += (double)ctu[k] * ctu[k]; }
            if (nz == 0) { nskip++; continue; }
            double mean = sum / (double)w3, var = sq / (double)w3 - mean * mean, sd = sqrt(var > 0 ? var : 0) + 1e-3;
            for (int z = 0; z < W; z++) {
                double cy, cx;
                axis_at(&ax, (double)(o[0] + z) * scale, &cy, &cx);
                cy /= scale; cx /= scale;
                for (int y = 0; y < W; y++) {
                    double dy = (double)(o[1] + y) - cy;
                    for (int x = 0; x < W; x++) {
                        size_t k = ((size_t)z * W + y) * W + x;
                        double dx = (double)(o[2] + x) - cx, nn_ = sqrt(dy * dy + dx * dx) + 1e-6;
                        xh[k] = (float)((ctu[k] - mean) / sd);
                        xh[w3 + k] = 0;
                        xh[2 * w3 + k] = ax.n ? (float)(dy / nn_) : 0;
                        xh[3 * w3 + k] = ax.n ? (float)(dx / nn_) : 0;
                    }
                }
            }
            nn_h2d(xd, xh, 4 * w3 * 4);
            const float *lg = unet_forward(u, xd, xs, 0);
            nn_sigmoid(lg, w3, pd);      /* channel 0 = recto */
            nn_d2h(ph, pd, w3 * 4);
            const char *e = nn_check();
            if (e) { fprintf(stderr, "cuda: %s\n", e); return 1; }
            /* interior into the shard buffer */
            for (int z = halo; z < W - halo; z++) {
                int64_t gz = tz + z - so[0]; if (gz < 0 || gz >= se[0] - so[0]) continue;
                for (int y = halo; y < W - halo; y++) {
                    int64_t gy = ty + y - so[1]; if (gy < 0 || gy >= se[1] - so[1]) continue;
                    for (int x = halo; x < W - halo; x++) {
                        int64_t gx = tx + x - so[2]; if (gx < 0 || gx >= se[2] - so[2]) continue;
                        size_t k = ((size_t)z * W + y) * W + x;
                        sbuf[((size_t)gz * shard + gy) * shard + gx] = ctu[k] ? (uint8_t)(ph[k] * 255.f + 0.5f) : 0;
                    }
                }
            }
            any = 1; ntiles++;
        }
        if (any && z3w_write_shard(w, sz, sy, sx, sbuf, nthreads)) { fprintf(stderr, "%s\n", z3w_error()); return 1; }
        fprintf(stderr, "\rshard %lld/%lld  %ld tiles (%ld air)  %.0fs   ", (long long)((sz * ns[1] + sy) * ns[2] + sx + 1), (long long)(ns[0] * ns[1] * ns[2]), ntiles, nskip, now() - t0);
    }
    fprintf(stderr, "\n");
    z3w_close(w);
    for (int l = 1; l < nlev; l++) if (pyramid_build_level(out, um_l, l, bn, shard, q, 0, nthreads, attrs)) return 1;
    pyramid_write_group(out, um_l, nlev, "ufsm-recto", attrs);
    fprintf(stderr, "wrote %s (%ld tiles in %.0fs)\n", out, ntiles, now() - t0);
    z3_close(ct); store_close(s); unet_free(u);
    return 0;
}
