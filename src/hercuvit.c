/* ufsm hercuvit: HercuViT v0 inference over a box of a CT zarr (src/hercuvit.h), following hercunet/infer/engine.py
   run_pass_blend + hercuvit/infer/model.py:
   - windows of P (default 128 = the recipe window) on the volume-global grid {k S} clamped to [0, V - P], S = 2 round(P (1 - overlap) / 2)
     (overlap 0.25 -> S 96); a window whose CT max is below --air (25) is skipped (no output there);
   - per window villa normalize_robust (1 / 99 percentile clip, median, 1.4826 MAD) of its uint8 CT;
   - surface: the logit margin l1 - l0 blended with the villa Gaussian (sigma P / 8, peak at P / 2), sigmoid, uint8 round(255 p);
   - instances (not blended, as the reference's per-window meshes): each voxel takes the window whose centre is nearest per axis (the
     largest Gaussian weight); id = window index * 64 + query + 1 (window index = linear index of the window in the volume-global
     grid, so ids agree between boxes), 0 where that window's own surface p <= --thresh (0.4); score = round(255 membership), the
     softmax-over-queries probability of the owning query.
   Output: <out>/surface, <out>/ids (uint32), <out>/score: OME groups with one level named by --um, the box shape, 512^3 shards,
   attribute ufsm.origin_zyx = box origin (ufsm overlay reads it). Work per output shard tile (windows overlapping two tiles run
   twice, as the reference's tile halo). */
#include "hercuvit.h"
#include "sysinfo.h"
#include "store.h"
#include "xnet.h"
#include "z3w.h"
#include "zarr3.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static const char *opt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return dflt;
}
static int write_file(const char *dir, const char *name, const void *p, size_t n) {
    char path[1400]; snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *f = fopen(path, "wb"); if (!f) { fprintf(stderr, "hercuvit: cannot write %s\n", path); return -1; }
    const size_t w = fwrite(p, 1, n, f); fclose(f); return w == n ? 0 : -1;
}
static float h2f(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000u) << 16, e = (h >> 10) & 0x1fu, m = h & 0x3ffu, x;
    if (e == 0) { if (!m) x = s; else { while (!(m & 0x400u)) { m <<= 1; e--; } e++; m &= 0x3ffu; x = s | ((uint32_t)(e + 112) << 23) | (m << 13); } }
    else if (e == 31) x = s | 0x7f800000u | (m << 13);
    else x = s | ((e + 112) << 23) | (m << 13);
    float f; memcpy(&f, &x, 4); return f;
}
static int dump_half(const char *dir, const char *name, const void *dev, size_t n) {
    uint16_t *h = malloc(n * 2); float *f = malloc(n * 4); if (!h || !f) return -1;
    xd2h(h, dev, n * 2); for (size_t i = 0; i < n; i++) f[i] = h2f(h[i]);
    const int rc = write_file(dir, name, f, n * 4); free(h); free(f); return rc;
}

/* --raw-in: one window through the network (parity with the PyTorch reference) */
static int raw_mode(hv_net *net, int argc, char **argv) {
    const char *in = opt(argc, argv, "--raw-in", nullptr), *out = opt(argc, argv, "--raw-out", nullptr);
    const int T = atoi(opt(argc, argv, "--window", "128")), reps = atoi(opt(argc, argv, "--reps", "1"));
    if (!in || !out) { fprintf(stderr, "hercuvit --raw-in U8 --raw-out DIR [--window 128] [--skips 1]\n"); return 2; }
    const size_t nv = (size_t)T * T * T; uint8_t *u = malloc(nv);
    FILE *f = fopen(in, "rb"); if (!f || !u || fread(u, 1, nv, f) != nv) { fprintf(stderr, "hercuvit: %s is not %d^3 bytes\n", in, T); return 1; } fclose(f);
    double lo, hi, c, s; hv_robust_stats(u, nv, &lo, &hi, &c, &s);
    fprintf(stderr, "hercuvit: robust stats lo %.6f hi %.6f median %.6f scale %.6f\n", lo, hi, c, s);
    uint8_t *ud = xmalloc_dev(nv); xh2d(ud, u, nv);
    double t0 = 0, t1 = 0, t2 = 0;
    for (int r = 0; r < reps; r++) {
        xnet_sync(); t0 = now_s();
        if (hv_forward(net, ud, T, (float)lo, (float)hi, (float)c, (float)s, 0)) return 1;
        xnet_sync(); t1 = now_s();
        if (hv_forward(net, ud, T, (float)lo, (float)hi, (float)c, (float)s, 1)) return 1;
        xnet_sync(); t2 = now_s();
    }
    fprintf(stderr, "hercuvit: window %d^3 surface forward %.1f ms, with the instance decoder %.1f ms\n", T, (t1 - t0) * 1e3, (t2 - t1) * 1e3);
    { char cmd[1500]; snprintf(cmd, sizeof cmd, "mkdir -p '%s'", out); if (system(cmd)) return 1; }
    const int g = T / 8;
    if (dump_half(out, "tokens.f32", hv_tokens(net), (size_t)g * g * g * 288)) return 1;
    if (atoi(opt(argc, argv, "--skips", "0"))) for (int k = 0; k < 3; k++) { int e, ch; const void *p = hv_skip(net, k, &e, &ch); char nm[32]; snprintf(nm, sizeof nm, "skip%d.f32", k);
        if (dump_half(out, nm, p, (size_t)e * e * e * ch)) return 1; }
    float *m = malloc(nv * 4), *mb = xmalloc_dev(nv * 4), *mh = malloc(nv * 4); uint8_t *ow = xmalloc_dev(nv), *oh = malloc(nv);
    xd2h(m, hv_margin(net), nv * 4);
    if (write_file(out, "margin.f32", m, nv * 4)) return 1;
    if (write_file(out, "embed.f32", hv_embed_host(net), (size_t)hv_queries(net) * 32 * 4)) return 1;
    const int r0[3] = {0, 0, 0}, r1[3] = {T, T, T}, dd[3] = {T, T, T};
    hv_final(hv_u1(net), T, hv_embed(net), hv_queries(net), hv_margin(net), 0.f, r0, r1, r0, dd, nullptr, 0, nullptr, ow, mb);
    xd2h(oh, ow, nv); xd2h(mh, mb, nv * 4);
    if (write_file(out, "owner.u8", oh, nv) || write_file(out, "memb.f32", mh, nv * 4)) return 1;
    char st[256]; snprintf(st, sizeof st, "%.9g %.9g %.9g %.9g\n", lo, hi, c, s);
    if (write_file(out, "stats.txt", st, strlen(st))) return 1;
    free(u); free(m); free(mh); free(oh); xfree_dev(ud); xfree_dev(mb); xfree_dev(ow);
    return 0;
}

typedef struct { int n, *s; } grid1;   /* window starts along one axis (volume-global, deduplicated, ascending) */
static grid1 grid_of(int64_t V, int P, int S) {
    grid1 g = {0, malloc(sizeof(int) * (size_t)(V / S + 3))};
    for (int64_t k = 0;; k++) {
        int64_t s = k * S; if (s > V - P) s = V - P;
        if (!g.n || g.s[g.n - 1] != s) g.s[g.n++] = (int)s;
        if (s == V - P) break;
    }
    return g;
}
/* the voxels [lo, hi) of window i that are nearest its centre along this axis (ties to the lower window) */
static void core_of(const grid1 *g, int i, int P, int *lo, int *hi) {
    const int c = g->s[i] + P / 2;
    *lo = i ? (g->s[i - 1] + P / 2 + c) / 2 + 1 : g->s[i];
    *hi = i + 1 < g->n ? (c + g->s[i + 1] + P / 2) / 2 + 1 : g->s[i] + P;
}

int cmd_hercuvit(int argc, char **argv) {
    const char *ck = opt(argc, argv, "--ckpt", nullptr);
    if (!ck || (!opt(argc, argv, "--raw-in", nullptr) && argc < 5)) {
        fprintf(stderr, "usage: ufsm hercuvit <ct-root> <ct-array-key> <out-dir> --ckpt checkpoint_latest.pth --box z,y,x,nz,ny,nx\n"
                        "         [--um 9.6] [--window 128] [--overlap 0.25] [--thresh 0.4] [--air 25] [--instances 1] [--shard 512]\n"
                        "         [--gpu 0] [--gpu-mem GB] [--threads 8] [--cache DIR]\n"
                        "  HercuViT v0 (HF jimmylomro/hercuvit-v0): <out>/surface (uint8 p), <out>/ids (uint32 window * 64 + query + 1),\n"
                        "  <out>/score (uint8 owner membership); the CT level should be ~9.6 um (the model's working scale)\n"
                        "  ufsm hercuvit --ckpt PTH --raw-in WINDOW.u8 --raw-out DIR [--window 128] [--skips 1]   one window (parity)\n");
        return 2;
    }
    if (xnet_init(atoi(opt(argc, argv, "--gpu", "0")))) return 1;
    if (opt(argc, argv, "--gpu-mem", nullptr)) xnet_set_budget((size_t)(atof(opt(argc, argv, "--gpu-mem", "8")) * 1e9));
    hv_net *net = hv_load(ck); if (!net) return 1;
    if (opt(argc, argv, "--raw-in", nullptr)) { const int rc = raw_mode(net, argc, argv); hv_free(net); return rc; }

    const char *root = argv[2], *key = argv[3], *out = argv[4];
    const int P = atoi(opt(argc, argv, "--window", "128")), inst = atoi(opt(argc, argv, "--instances", "1")), air = atoi(opt(argc, argv, "--air", "25"));
    const int SH = atoi(opt(argc, argv, "--shard", "512")), nth = ufsm_threads(atoi(opt(argc, argv, "--threads", "0")), 8);
    const double ov = atof(opt(argc, argv, "--overlap", "0.25")), thr = atof(opt(argc, argv, "--thresh", "0.4")), um = atof(opt(argc, argv, "--um", "9.6"));
    const float lthr = (float)log(thr / (1.0 - thr));
    int S = 2 * (int)fmax(1.0, round(P * (1.0 - ov) / 2)); if (S > P) S = P; if (S < 2) S = 2;
    if (SH % 128 || SH < 128) { fprintf(stderr, "hercuvit: --shard must be a multiple of 128\n"); return 2; }
    store *st = store_open(root); z3 *ct = st ? z3_open(st, key, opt(argc, argv, "--cache", nullptr)) : nullptr;
    if (!ct) { fprintf(stderr, "hercuvit: %s %s: %s\n", root, key, z3_error()); return 1; }
    const int64_t *V = z3_meta_of(ct)->shape;
    long long bv[6]; const char *box = opt(argc, argv, "--box", nullptr);
    if (!box || sscanf(box, "%lld,%lld,%lld,%lld,%lld,%lld", bv, bv + 1, bv + 2, bv + 3, bv + 4, bv + 5) != 6) { fprintf(stderr, "hercuvit: --box z,y,x,nz,ny,nx\n"); return 2; }
    for (int a = 0; a < 3; a++) {
        if (bv[a] < 0 || bv[3 + a] < 1 || bv[a] + bv[3 + a] > V[a]) { fprintf(stderr, "hercuvit: the box leaves the volume (%lld %lld %lld)\n", (long long)V[0], (long long)V[1], (long long)V[2]); return 2; }
        if (V[a] < P) { fprintf(stderr, "hercuvit: the volume is smaller than the window along axis %d\n", a); return 2; }
    }
    grid1 G[3]; for (int a = 0; a < 3; a++) G[a] = grid_of(V[a], P, S);
    const uint64_t nwin_all = (uint64_t)G[0].n * G[1].n * G[2].n;
    if (inst && nwin_all * 64 + 64 >= 0xffffffffull) { fprintf(stderr, "hercuvit: %llu windows overflow uint32 ids\n", (unsigned long long)nwin_all); return 2; }
    fprintf(stderr, "hercuvit: box %lld,%lld,%lld +%lldx%lldx%lld, window %d stride %d, global grid %dx%dx%d, instances %d\n",
            bv[0], bv[1], bv[2], bv[3], bv[4], bv[5], P, S, G[0].n, G[1].n, G[2].n, inst);

    /* outputs */
    char lv[32]; z3w_level_name(um, lv, sizeof lv);
    const int64_t ash[3] = {bv[3], bv[4], bv[5]};
    char attrs[2048], dir[1400];
    z3w *zw[3] = {0};
    const char *names[3] = {"surface", "ids", "score"}, *what[3] = {"HercuViT surface probability (uint8 = round(p 255))", "HercuViT sheet instance id = window * 64 + query + 1 (window: linear index in the volume-global window grid), 0 = no surface", "HercuViT owner membership (uint8 = round(255 softmax max))"};
    for (int k = 0; k < (inst ? 3 : 1); k++) {
        snprintf(attrs, sizeof attrs, "{\"ufsm\":{\"content\":\"%s\",\"checkpoint\":\"%s\",\"source\":\"%s/%s\",\"origin_zyx\":[%lld,%lld,%lld],\"window\":%d,\"stride\":%d,\"thresh\":%g,\"grid\":[%d,%d,%d]}}",
                 what[k], ck, root, key, bv[0], bv[1], bv[2], P, S, thr, G[0].n, G[1].n, G[2].n);
        snprintf(dir, sizeof dir, "%s/%s", out, names[k]);
        if (z3w_write_group(dir, &um, 1, names[k], attrs)) { fprintf(stderr, "hercuvit: %s\n", z3w_error()); return 1; }
        snprintf(dir, sizeof dir, "%s/%s/%s", out, names[k], lv);
        zw[k] = k == 1 ? z3w_create_typed(dir, ash, SH, 3, 0.f, 0.0, attrs) : z3w_create(dir, ash, SH, 0.f, 0, attrs);
        if (!zw[k]) { fprintf(stderr, "hercuvit: %s\n", z3w_error()); return 1; }
    }

    const size_t sh3 = (size_t)SH * SH * SH;
    float *acc = xmalloc_dev(sh3 * 4), *wsum = xmalloc_dev(sh3 * 4); uint8_t *u8d = xmalloc_dev(sh3), *scd = inst ? xmalloc_dev(sh3) : nullptr;
    uint32_t *idd = inst ? xmalloc_dev(sh3 * 4) : nullptr; uint8_t *wd = xmalloc_dev((size_t)P * P * P);
    uint8_t *hb = malloc(sh3 * 4), *win = malloc((size_t)P * P * P);
    if (!acc || !wsum || !u8d || !wd || !hb || !win || (inst && (!scd || !idd))) { fprintf(stderr, "hercuvit: out of memory\n"); return 1; }
    const int nt[3] = {(int)((bv[3] + SH - 1) / SH), (int)((bv[4] + SH - 1) / SH), (int)((bv[5] + SH - 1) / SH)};
    long long nwin = 0, nair = 0; double t0 = now_s(), tfwd = 0;
    for (int tz = 0; tz < nt[0]; tz++) for (int ty = 0; ty < nt[1]; ty++) for (int tx = 0; tx < nt[2]; tx++) {
        const int ti[3] = {tz, ty, tx}; int64_t t0a[3], t1a[3]; int dd[3], wl[3], wh[3];
        for (int a = 0; a < 3; a++) {
            t0a[a] = bv[a] + (int64_t)ti[a] * SH; t1a[a] = t0a[a] + SH < bv[a] + bv[3 + a] ? t0a[a] + SH : bv[a] + bv[3 + a]; dd[a] = (int)(t1a[a] - t0a[a]);
            wl[a] = G[a].n; wh[a] = -1;
            for (int i = 0; i < G[a].n; i++) if (G[a].s[i] < t1a[a] && G[a].s[i] + P > t0a[a]) { if (i < wl[a]) wl[a] = i; wh[a] = i; }
        }
        const size_t td = (size_t)dd[0] * dd[1] * dd[2];
        hv_fill(acc, td, 0.f); hv_fill(wsum, td, 0.f);
        if (inst) { hv_fill((float *)idd, td, 0.f); hv_fill((float *)scd, (td + 3) / 4, 0.f); }
        /* the CT of all windows touching the tile */
        const int64_t ro[3] = {G[0].s[wl[0]], G[1].s[wl[1]], G[2].s[wl[2]]}, rn[3] = {G[0].s[wh[0]] + P - ro[0], G[1].s[wh[1]] + P - ro[1], G[2].s[wh[2]] + P - ro[2]};
        uint8_t *ctb = malloc((size_t)rn[0] * rn[1] * rn[2]);
        if (!ctb || z3_read(ct, ro, rn, ctb, nth)) { fprintf(stderr, "hercuvit: CT read: %s\n", z3_error()); return 1; }
        for (int iz = wl[0]; iz <= wh[0]; iz++) for (int iy = wl[1]; iy <= wh[1]; iy++) for (int ix = wl[2]; ix <= wh[2]; ix++) {
            const int ii[3] = {iz, iy, ix}; int org[3], r0[3], r1[3], off[3], hit = 1;
            for (int a = 0; a < 3; a++) {
                org[a] = (int)(G[a].s[ii[a]] - t0a[a]);
                int lo, hi; core_of(&G[a], ii[a], P, &lo, &hi);
                if (lo < t0a[a]) { lo = (int)t0a[a]; }
                if (hi > t1a[a]) { hi = (int)t1a[a]; }
                r0[a] = lo - G[a].s[ii[a]]; r1[a] = hi - G[a].s[ii[a]]; off[a] = -org[a]; hit &= hi > lo;
            }
            int mx = 0;
            for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) {
                const uint8_t *src = ctb + (((size_t)(G[0].s[iz] - ro[0] + z) * rn[1] + (G[1].s[iy] - ro[1] + y)) * rn[2] + (G[2].s[ix] - ro[2]));
                uint8_t *dst = win + ((size_t)z * P + y) * P; memcpy(dst, src, P);
                for (int x = 0; x < P; x++) if (dst[x] > mx) mx = dst[x];
            }
            if (mx < air) { nair++; continue; }
            double lo, hi, c, s; hv_robust_stats(win, (size_t)P * P * P, &lo, &hi, &c, &s);
            xh2d(wd, win, (size_t)P * P * P);
            const double tf = now_s();
            if (hv_forward(net, wd, P, (float)lo, (float)hi, (float)c, (float)s, inst && hit)) return 1;
            hv_blend(acc, wsum, dd, hv_margin(net), P, org);
            if (inst && hit) {
                const uint32_t base = (uint32_t)((((uint64_t)iz * G[1].n + iy) * G[2].n + ix) * 64);
                hv_final(hv_u1(net), P, hv_embed(net), hv_queries(net), hv_margin(net), lthr, r0, r1, off, dd, idd, base, scd, nullptr, nullptr);
            }
            if (xnet_check("hercuvit window")) return 1;
            tfwd += now_s() - tf; nwin++;
            if (nwin % 50 == 0) fprintf(stderr, "hercuvit: %lld windows (%lld air), %.1f s, %.0f ms / window\n", nwin, nair, now_s() - t0, 1e3 * tfwd / nwin);
        }
        free(ctb);
        /* write the tile's shards (the tile is one shard of the box-shaped arrays) */
        hv_finish(acc, wsum, td, u8d);
        for (int k = 0; k < (inst ? 3 : 1); k++) {
            const int es = k == 1 ? 4 : 1; const void *src = k == 0 ? (const void *)u8d : k == 1 ? (const void *)idd : (const void *)scd;
            uint8_t *tmp = malloc(td * es); xd2h(tmp, src, td * es);
            memset(hb, 0, sh3 * es);
            for (int z = 0; z < dd[0]; z++) for (int y = 0; y < dd[1]; y++)
                memcpy(hb + (((size_t)z * SH + y) * SH) * es, tmp + (((size_t)z * dd[1] + y) * dd[2]) * es, (size_t)dd[2] * es);
            free(tmp);
            if (z3w_write_shard(zw[k], tz, ty, tx, hb, nth)) { fprintf(stderr, "hercuvit: %s\n", z3w_error()); return 1; }
        }
        fprintf(stderr, "hercuvit: tile %d,%d,%d done (%lld windows, %.1f s)\n", tz, ty, tx, nwin, now_s() - t0);
    }
    for (int k = 0; k < 3; k++) if (zw[k] && z3w_close(zw[k])) { fprintf(stderr, "hercuvit: %s\n", z3w_error()); return 1; }
    const double el = now_s() - t0;
    fprintf(stderr, "hercuvit: done: %lld windows (+%lld air) in %.1f s (%.0f ms / window on the GPU path), %.2f Mvox/s\n", nwin, nair, el, nwin ? 1e3 * tfwd / nwin : 0.0,
            (double)bv[3] * bv[4] * bv[5] / el / 1e6);
    xfree_dev(acc); xfree_dev(wsum); xfree_dev(u8d); xfree_dev(scd); xfree_dev(idd); xfree_dev(wd); free(hb); free(win);
    for (int a = 0; a < 3; a++) free(G[a].s);
    hv_free(net); z3_close(ct);
    (void)errno;
    return 0;
}
