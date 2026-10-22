/* ufsm dinovol-embed: the scrollprize dinovol model (src/dinovol.h) over a CT box with embed_volume.py's lattice tiling, blending and
   volume-wide normalisation; ufsm dinovol-eval: the sheet-identity check of the dense embedding along published surfaces. */
#include "dinovol.h"
#include "xnet.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double dv_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static const char *opt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return dflt;
}
static int write_file(const char *dir, const char *name, const void *p, size_t n) {
    char path[2048]; snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *f = fopen(path, "wb"); if (!f) { fprintf(stderr, "dinovol: cannot write %s\n", path); return -1; }
    const size_t w = fwrite(p, 1, n, f); fclose(f);
    return w == n ? 0 : -1;
}
static void *read_file(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb"); if (!f) return nullptr;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    void *p = malloc(sz > 0 ? (size_t)sz : 1); if (!p || fread(p, 1, (size_t)sz, f) != (size_t)sz) { free(p); fclose(f); return nullptr; }
    fclose(f); *n = (size_t)sz; return p;
}
static float h2f(uint16_t h) {
    const uint32_t s = (uint32_t)(h & 0x8000u) << 16; int e = (h >> 10) & 0x1f; uint32_t m = h & 0x3ffu, x;
    if (e == 0) { if (!m) x = s; else { while (!(m & 0x400u)) { m <<= 1; e--; } e++; m &= 0x3ffu; x = s | ((uint32_t)(e + 112) << 23) | (m << 13); } }
    else if (e == 31) x = s | 0x7f800000u | (m << 13);
    else x = s | ((uint32_t)(e + 112) << 23) | (m << 13);
    float f; memcpy(&f, &x, 4); return f;
}
static int dump_dev_half(const char *dir, const char *name, const void *dev, size_t n) {   /* fp16 device array -> float32 file */
    uint16_t *h = malloc(n * 2); float *f = malloc(n * 4);
    xd2h(h, dev, n * 2);
    for (size_t i = 0; i < n; i++) f[i] = h2f(h[i]);
    const int rc = write_file(dir, name, f, n * 4); free(h); free(f); return rc;
}

/* --raw-in: one tile of uint8 voxels through the network, outputs as float32 files in --raw-out (parity with the PyTorch reference) */
static int raw_mode(dv_net *net, int argc, char **argv) {
    const char *in = opt(argc, argv, "--raw-in", nullptr), *out = opt(argc, argv, "--raw-out", nullptr), *norm = opt(argc, argv, "--norm", nullptr);
    const char *pts = opt(argc, argv, "--points", nullptr);
    const int T = atoi(opt(argc, argv, "--tile", "192")), reps = atoi(opt(argc, argv, "--reps", "1"));
    float c = 0, s = 1, hi = 255;
    if (!in || !out || !norm || sscanf(norm, "%f,%f,%f", &c, &s, &hi) != 3) { fprintf(stderr, "dinovol-embed --raw-in U8 --raw-out DIR --norm center,scale,hi [--tile T] [--points P.f32]\n"); return 2; }
    size_t nb = 0; uint8_t *u = read_file(in, &nb);
    if (!u || nb != (size_t)T * T * T) { fprintf(stderr, "dinovol: %s is not %d^3 bytes\n", in, T); return 1; }
    uint8_t *ud = xmalloc_dev(nb); xh2d(ud, u, nb);
    double t0 = 0, t1 = 0;
    for (int r = 0; r < reps; r++) {
        xnet_sync(); t0 = dv_now();
        if (dv_forward(net, ud, T, c, s, hi)) return 1;
        xnet_sync(); t1 = dv_now();
    }
    fprintf(stderr, "dinovol: tile %d^3 forward %.1f ms\n", T, (t1 - t0) * 1e3);
    const int g = T / 8;
    if (dump_dev_half(out, "tokens.f32", dv_tokens(net), (size_t)g * g * g * 480)) return 1;
    for (int k = 0; k < 3; k++) { int e, ch; const void *p = dv_skip(net, k, &e, &ch); char nm[32]; snprintf(nm, sizeof nm, "skip%d.f32", k);
        if (dump_dev_half(out, nm, p, (size_t)e * e * e * ch)) return 1; }
    { int e, ch; const void *p = dv_fpn(net, &e, &ch); if (dump_dev_half(out, "fpn.f32", p, (size_t)e * e * e * ch)) return 1; }
    if (pts) {
        size_t pb = 0; float *ph = read_file(pts, &pb); if (!ph || pb % 12) { fprintf(stderr, "dinovol: %s: float32 [P][3] points\n", pts); return 1; }
        const int P = (int)(pb / 12); int nd, ni; const int C = dv_dense_dims(net, &nd, &ni);
        float *pd = xmalloc_dev(pb), *acc = xmalloc_dev((size_t)P * C * 4); xh2d(pd, ph, pb);
        xnet_sync(); const double t2 = dv_now();
        if (dv_net_points(net, pd, nullptr, nullptr, P, acc)) return 1;
        xnet_sync(); fprintf(stderr, "dinovol: dense head %d points %.1f ms\n", P, (dv_now() - t2) * 1e3);
        float *o = malloc((size_t)P * C * 4); xd2h(o, acc, (size_t)P * C * 4);
        if (write_file(out, "dense_ink.f32", o, (size_t)P * C * 4)) return 1;
        free(o); free(ph); xfree_dev(pd); xfree_dev(acc);
    }
    free(u); xfree_dev(ud);
    return 0;
}

int cmd_dinovol_embed(int argc, char **argv) {
    const char *ck = opt(argc, argv, "--ckpt", nullptr);
    if (!ck) { fprintf(stderr, "usage: ufsm dinovol-embed --ckpt teacher_step_640000.pt ...\n"); return 2; }
    if (xnet_init(atoi(opt(argc, argv, "--gpu", "0")))) return 1;
    if (*opt(argc, argv, "--gpu-mem", "")) xnet_set_budget((size_t)(atof(opt(argc, argv, "--gpu-mem", "10")) * 1e9));
    dv_net *net = dv_load(ck); if (!net) return 1;
    int rc = 2;
    if (opt(argc, argv, "--raw-in", nullptr)) rc = raw_mode(net, argc, argv);
    dv_free(net);
    return rc;
}
int cmd_dinovol_eval(int argc, char **argv) { (void)argc; (void)argv; fprintf(stderr, "dinovol-eval: not built yet\n"); return 2; }
