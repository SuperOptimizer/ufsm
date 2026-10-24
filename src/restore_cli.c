/* Task restore command-line paths (src/restore.h):
     ufsm predict <restore ckpt> ...      super-resolve a CT volume (restore_predict, called by cmd_predict)
     ufsm restore-eval <ckpt> <clean.npy>...   PSNR of the net on clean crops degraded by a fixed recipe (comparison with older runs)
     ufsm restore-stats <root> <key> ...  calibration statistics of a CT box, as stored or after a scanner degradation */
#include "nn.h"
#include "checkpoint.h"
#include "pyramid.h"
#include "restore.h"
#include "sources.h"
#include "srpair.h"
#include "band.h"
#include "store.h"
#include "unet.h"
#include "z3w.h"
#include "zarr3.h"
#include "json.h"
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

/* the checkpoint's "restore" object: 1 (and *split, *fine_um) for a task restore checkpoint, 0 otherwise */
/* real-pair checkpoints (train --sr-pairs 1) store "sr_scale" (4): the output grid is the input's / scale, whatever the input's voxel size */
static double restore_checkpoint_scale(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) return 0;
    char *h = malloc(UFSM_CHECKPOINT_HEADER + 1); size_t n = fread(h, 1, UFSM_CHECKPOINT_HEADER, f); fclose(f); h[n] = 0;
    char *e = memchr(h, '\n', n); if (e) *e = 0;
    double sc = 0;
    if (n > 4 && !strncmp(h, "UFSM", 4)) {
        json *j = json_parse(h + 4, strlen(h + 4));
        const json *x = j ? json_get(j, "extra") : nullptr, *r = x ? json_get(x, "restore") : nullptr;
        sc = r ? json_num(json_get(r, "sr_scale"), 0) : 0;
        json_free(j);
    }
    free(h); return sc;
}
int restore_checkpoint(const char *path, int *split, double *fine_um) {
    FILE *f = fopen(path, "rb"); if (!f) return 0;
    char *h = malloc(UFSM_CHECKPOINT_HEADER + 1); size_t n = fread(h, 1, UFSM_CHECKPOINT_HEADER, f); fclose(f); h[n] = 0;
    char *e = memchr(h, '\n', n); if (e) *e = 0;
    int ok = 0;
    if (n > 4 && !strncmp(h, "UFSM", 4)) {
        json *j = json_parse(h + 4, strlen(h + 4));
        const json *x = j ? json_get(j, "extra") : nullptr, *r = x ? json_get(x, "restore") : nullptr;
        if (r && !strcmp(json_str(json_get(x, "task"), ""), "restore")) {
            ok = 1;
            if (split) *split = !strcmp(json_str(json_get(r, "input"), "split"), "split");
            if (fine_um) *fine_um = json_num(json_get(r, "fine_um"), 2.4);
        }
        json_free(j);
    }
    free(h);
    return ok;
}

/* the network and its inference precision as trained (the checkpoint's runtime settings) */
static unet *restore_net(const char *ckpt, int use_ema, unet_cfg *cfg) {
    checkpoint_runtime rt; int step; const int emb = checkpoint_runtime_read(ckpt, &rt);
    if (emb < 0 || unet_peek(ckpt, cfg, &step)) { fprintf(stderr, "restore: cannot read %s\n", ckpt); return nullptr; }
    nn_set_nlev(cfg->nlev);
    if (emb) {
        nn_set_prec(rt.prec); nn_set_f16(rt.f16); nn_set_gn_stored(rt.gn_stored);
        if (nn_set_prec_policy(rt.policy)) return nullptr;
        unet_set_act_mx4(rt.act_mx4); unet_set_act_mx8(rt.act_mx8); unet_set_input_mx(rt.input_mx); unet_set_input_prec(rt.input_prec);
    } else { nn_set_f16(1); unet_set_input_prec(8); unet_set_input_mx(1); }
    unet_set_grad_mx8(0);
    if (nn_get_f16()) nn_set_grad_scale(1024.f);
    unet *u = unet_create(cfg);
    if (unet_load(u, ckpt) < 0) { fprintf(stderr, "restore: cannot load %s\n", ckpt); unet_free(u); return nullptr; }
    unet_use_ema(u, use_ema);
    fprintf(stderr, "restore: %s (step %d, %s weights), prec %d act_mx4 %d act_mx8 %d input_mx %d input_prec %d\n", ckpt, step, use_ema ? "EMA" : "live", nn_get_prec(), emb ? rt.act_mx4 : 0, emb ? rt.act_mx8 : 0, unet_input_converted(), unet_input_prec());
    return u;
}

/* ---- ufsm predict on a restore checkpoint ---- */
int restore_predict(int argc, char **argv) {
    const char *ckpt = argv[2], *root = argv[3], *key = argv[4], *out = argv[5];
    int split = 1; double fine_um = 2.4;
    restore_checkpoint(ckpt, &split, &fine_um);
    const double um = atof(opt(argc, argv, "--um", "0"));
    const int level = atoi(opt(argc, argv, "--level", "0")), S = atoi(opt(argc, argv, "--shard", "512")), H = atoi(opt(argc, argv, "--halo", "32"));
    const double sr_scale = restore_checkpoint_scale(ckpt);   /* real-pair nets: x scale of whatever input voxel size */
    const double out_um = atof(opt(argc, argv, "--out-um", "0")) > 0 ? atof(opt(argc, argv, "--out-um", "0")) : sr_scale > 0 ? um * (1 << level) / sr_scale : fine_um;
    const float q = (float)atof(opt(argc, argv, "--q", "2"));
    const int nlev = atoi(opt(argc, argv, "--levels", "1")), nthreads = atoi(opt(argc, argv, "--threads", "16")), use_ema = atoi(opt(argc, argv, "--ema", "1")) != 0;
    if (!(um > 0) || S < 128 || S % 128 || H < 0 || nlev < 1 || nlev > 10 || !(q >= 0)) {
        fprintf(stderr, "usage: ufsm predict <restore ckpt> <root> <ct-group> <out-dir> --um U (the CT's level-0 voxel) [--level L] [--out-um %g] [--box z,y,x,nz,ny,nx (output voxels)]\n"
                        "       [--shard 512] [--halo 32] [--q 2] [--levels 1] [--ema 1] [--gpu 0] [--cache DIR] [--shards FILE (\"sz sy sx\" lines: only these)] [--identity 1 (trilinear only)]\n"
                        "  restores level L of the CT onto the --out-um grid (trilinear input, residual net), uint8 CT in 512^3 shards\n", fine_um);
        return 2;
    }
    if (nn_init(atoi(opt(argc, argv, "--gpu", "0")))) { fprintf(stderr, "restore: cannot select the GPU\n"); return 1; }
    unet_cfg cfg; unet *u = restore_net(ckpt, use_ema, &cfg);
    if (!u) return 1;
    const int W = S + 2 * H, ident = atoi(opt(argc, argv, "--identity", "0")) != 0;
    if (cfg.scale_cond) unet_set_cond(u, (float)log2(um * (1 << level) / out_um) - 2.f);   /* ratio-conditioned nets: log2(input / output voxel) - 2 */
    if (W % (1 << (cfg.nlev - 1))) { fprintf(stderr, "restore: shard + 2 halo (%d) must be a multiple of %d\n", W, 1 << (cfg.nlev - 1)); return 2; }
    store *st = store_open(root);
    z3 *ct = st ? pyramid_open_level(st, key, level, um, opt(argc, argv, "--cache", nullptr)) : nullptr;
    if (!ct) { fprintf(stderr, "restore: cannot open %s/%s level %d\n", root, key, level); return 1; }
    const z3_meta *m = z3_meta_of(ct);
    const double in_um = um * (1 << level); const float step = (float)(in_um / out_um);
    int64_t bo[3] = {0, 0, 0}, bn[3];
    for (int d = 0; d < 3; d++) bn[d] = (int64_t)floor(m->shape[d] * (double)step);
    const char *box = opt(argc, argv, "--box", nullptr);
    if (box) { long long v[6]; if (sscanf(box, "%lld,%lld,%lld,%lld,%lld,%lld", v, v + 1, v + 2, v + 3, v + 4, v + 5) != 6) { fprintf(stderr, "--box z,y,x,nz,ny,nx\n"); return 2; }
               for (int d = 0; d < 3; d++) { bo[d] = v[d]; bn[d] = v[3 + d]; } }
    fprintf(stderr, "restore: %s/%s level %d (%g um) -> %g um (x%.4g), box %lld,%lld,%lld + %lldx%lldx%lld, window %d (halo %d), q %g\n", root, key, level, in_um, out_um, step,
            (long long)bo[0], (long long)bo[1], (long long)bo[2], (long long)bn[0], (long long)bn[1], (long long)bn[2], W, H, (double)q);
    char lv[32], ldir[1400], attrs[1400]; z3w_level_name(out_um, lv, sizeof lv); snprintf(ldir, sizeof ldir, "%s/%s", out, lv);
    snprintf(attrs, sizeof attrs, "{\"ufsm\":{\"content\":\"restored CT (task restore, uint8)\",\"checkpoint\":\"%s\",\"source\":\"%s/%s\",\"source_level\":%d,\"source_um\":%.9g,\"origin_zyx\":[%lld,%lld,%lld],\"window\":%d,\"halo\":%d,\"ema\":%d}}",
             ckpt, root, key, level, in_um, (long long)bo[0], (long long)bo[1], (long long)bo[2], W, H, use_ema);
    z3w *w = z3w_create(ldir, bn, S, q, 0, attrs);
    if (!w) { fprintf(stderr, "%s\n", z3w_error()); return 1; }
    const size_t w3 = (size_t)W * W * W;
    const int cs = (int)ceil(W / step) + 4;   /* coarse window edge (with margins) */
    const size_t c3 = (size_t)cs * cs * cs;
    uint8_t *hc = malloc(c3), *hr = malloc(c3), *ho = nn_host_alloc(w3), *shard = malloc((size_t)S * S * S);
    uint8_t *dc = nn_malloc(c3), *dlow = nn_malloc(w3), *dout = nn_malloc(w3); void *dx = nn_malloc(w3 * 2 * cfg.cin);
    if (!hc || !hr || !ho || !shard || !dc || !dlow || !dout || !dx) { fprintf(stderr, "restore: out of memory\n"); return 1; }
    int64_t ns[3]; for (int d = 0; d < 3; d++) ns[d] = (bn[d] + S - 1) / S;
    /* --shards FILE: only the listed output shards ("sz sy sx" lines, e.g. ufsm reg render "touch"); the rest stay empty */
    uint8_t *want = nullptr; const char *shf = opt(argc, argv, "--shards", nullptr);
    if (shf) {
        FILE *f = fopen(shf, "r"); if (!f) { fprintf(stderr, "restore: cannot read %s\n", shf); return 1; }
        want = calloc((size_t)ns[0] * ns[1] * ns[2], 1); long long a, b, c; long nw = 0;
        while (fscanf(f, "%lld %lld %lld", &a, &b, &c) == 3) if (a >= 0 && b >= 0 && c >= 0 && a < ns[0] && b < ns[1] && c < ns[2]) { want[((size_t)a * ns[1] + b) * ns[2] + c] = 1; nw++; }
        fclose(f); fprintf(stderr, "restore: %ld listed shards\n", nw);
    }
    const double t0 = now(); long done = 0, air = 0, tot = ns[0] * ns[1] * ns[2];
    for (int64_t sz = 0; sz < ns[0]; sz++) for (int64_t sy = 0; sy < ns[1]; sy++) for (int64_t sx = 0; sx < ns[2]; sx++) {
        const int64_t sh[3] = {sz, sy, sx};
        if (want && !want[((size_t)sz * ns[1] + sy) * ns[2] + sx]) { air++; done++; continue; }
        int64_t c0[3], ws[3]; float ph[3];
        for (int d = 0; d < 3; d++) { ws[d] = bo[d] + sh[d] * S - H; c0[d] = (int64_t)floor((ws[d] + 0.5) / step - 0.5) - 1; ph[d] = (float)(ws[d] / (double)step - (double)c0[d]); }
        /* the coarse window, zero outside the volume */
        int64_t lo[3], hi[3]; int empty = 0;
        for (int d = 0; d < 3; d++) { lo[d] = c0[d] < 0 ? 0 : c0[d]; hi[d] = c0[d] + cs > m->shape[d] ? m->shape[d] : c0[d] + cs; if (hi[d] <= lo[d]) empty = 1; }
        memset(hc, 0, c3);
        size_t nz = 0;
        if (!empty) {
            const int64_t rn[3] = {hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]};
            if (z3_read(ct, lo, rn, hr, nthreads)) { fprintf(stderr, "restore: %s\n", z3_error()); return 1; }
            for (int64_t z = 0; z < rn[0]; z++) for (int64_t y = 0; y < rn[1]; y++) {
                const uint8_t *src = hr + ((size_t)z * rn[1] + y) * rn[2];
                memcpy(hc + ((size_t)(z + lo[0] - c0[0]) * cs + (size_t)(y + lo[1] - c0[1])) * cs + (size_t)(lo[2] - c0[2]), src, (size_t)rn[2]);
                for (int64_t x = 0; x < rn[2]; x++) nz += src[x] != 0;
            }
        }
        if (!nz) { air++; done++; continue; }   /* (an omitted shard reads as 0) */
        nn_h2d(dc, hc, c3);
        const float stp[3] = {step, step, step};
        nn_restore_upsample(dc, cs, cs, cs, dlow, W, W, W, stp, ph, nullptr);
        if (ident) nn_d2h(ho, dlow, w3);   /* --identity 1: the trilinear upsampling itself (the SR baseline on the same grid) */
        else {
            nn_restore_input(dlow, dx, w3, cfg.cin == 2, nn_get_f16());
            const float *lg = unet_forward_x(u, dx, (shape5){1, cfg.cin, W, W, W}, 0, 1);
            nn_restore_output(lg, unet_logits_h16(u), dlow, dout, w3);
            nn_d2h(ho, dout, w3);
        }
        for (int z = 0; z < S; z++) for (int y = 0; y < S; y++) memcpy(shard + ((size_t)z * S + y) * S, ho + ((size_t)(z + H) * W + (y + H)) * W + H, (size_t)S);
        if (z3w_write_shard(w, sz, sy, sx, shard, nthreads)) { fprintf(stderr, "restore: %s\n", z3w_error()); return 1; }
        done++;
        fprintf(stderr, "\rshard %ld/%ld (%ld air) %.0fs   ", done, tot, air, now() - t0);
        const char *e = nn_check(); if (e) { fprintf(stderr, "restore: %s\n", e); return 1; }
    }
    fprintf(stderr, "\n");
    if (z3w_close(w)) { fprintf(stderr, "restore: %s\n", z3w_error()); return 1; }
    for (int l = 1; l < nlev; l++) if (pyramid_build_level(out, out_um, l, bn, S, q, 0, nthreads, attrs)) return 1;
    if (pyramid_write_group(out, out_um, nlev, "ufsm-restore", attrs)) return 1;
    fprintf(stderr, "restore: %ld shards (%ld air) in %.0fs -> %s\n", tot, air, now() - t0, out);
    return 0;
}

/* ---- .npy (uint8 C-order 3-D) ---- */
static uint8_t *npy_read(const char *path, int64_t dim[3]) {
    FILE *f = fopen(path, "rb"); if (!f) return nullptr;
    char hd[10]; if (fread(hd, 1, 10, f) != 10 || memcmp(hd, "\x93NUMPY", 6)) { fclose(f); return nullptr; }
    const size_t hl = hd[6] == 1 ? (size_t)(uint8_t)hd[8] | (size_t)(uint8_t)hd[9] << 8 : 0;
    size_t hlen = hl; char *h;
    if (hd[6] != 1) { uint8_t b2[2]; if (fread(b2, 1, 2, f) != 2) { fclose(f); return nullptr; } hlen = (size_t)(uint8_t)hd[8] | (size_t)(uint8_t)hd[9] << 8 | (size_t)b2[0] << 16 | (size_t)b2[1] << 24; }
    h = malloc(hlen + 1); if (fread(h, 1, hlen, f) != hlen) { free(h); fclose(f); return nullptr; } h[hlen] = 0;
    const char *sp = strstr(h, "shape");
    long long a, b, c;
    if (!strstr(h, "|u1") || strstr(h, "'fortran_order': True") || !sp || sscanf(strchr(sp, '(') + 1, "%lld, %lld, %lld", &a, &b, &c) != 3) { free(h); fclose(f); return nullptr; }
    free(h); dim[0] = a; dim[1] = b; dim[2] = c;
    const size_t n = (size_t)a * b * c; uint8_t *v = malloc(n);
    if (!v || fread(v, 1, n, f) != n) { free(v); fclose(f); return nullptr; }
    fclose(f); return v;
}

/* ufsm restore-eval <ckpt> <clean.npy>... [--family pyramid|scanner|identity] [--q Q] [--seed S] [--ema 1]: per crop and overall the
   input / output PSNR over voxels with clean > 0, input from the clean crop by one recipe (the training geometry and codec) */
int cmd_restore_eval(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm restore-eval <ckpt> <clean.npy>... [--family pyramid|scanner|identity] [--q Q (pyramid: 2)] [--seed 1] [--ema 1] [--gpu 0]\n"); return 2; }
    const char *fam = opt(argc, argv, "--family", "pyramid");
    int f = -1; for (int k = 0; k < RF_N; k++) if (!strcmp(fam, restore_family_names[k])) f = k;
    if (f < 0) { fprintf(stderr, "--family identity|pyramid|scanner\n"); return 2; }
    if (nn_init(atoi(opt(argc, argv, "--gpu", "0")))) return 1;
    unet_cfg cfg; unet *u = restore_net(argv[2], atoi(opt(argc, argv, "--ema", "1")) != 0, &cfg);
    if (!u) return 1;
    restore_ranges g; restore_ranges_default(&g);
    double se = 0, sb = 0, sn = 0;
    for (int i = 3; i < argc; i++) {
        if (argv[i][0] == '-') { i++; continue; }
        int64_t dm[3]; uint8_t *c = npy_read(argv[i], dm);
        if (!c) { fprintf(stderr, "restore-eval: cannot read %s (uint8 3-D .npy)\n", argv[i]); return 1; }
        const int D = (int)dm[0], Hh = (int)dm[1], Wd = (int)dm[2]; const size_t n = (size_t)D * Hh * Wd;
        if (D % (1 << (cfg.nlev - 1)) || Hh % (1 << (cfg.nlev - 1)) || Wd % (1 << (cfg.nlev - 1))) { fprintf(stderr, "restore-eval: %s: dims must divide by %d\n", argv[i], 1 << (cfg.nlev - 1)); return 2; }
        restore_recipe r; restore_recipe_draw(&r, f, (uint64_t)atoll(opt(argc, argv, "--seed", "1")) * 7919 + (uint64_t)i, D, Hh, Wd, &g);
        if (*opt(argc, argv, "--q", "")) r.q = (float)atof(opt(argc, argv, "--q", "0"));
        const size_t sb_ = restore_scratch_bytes(D, Hh, Wd, 3.f);
        uint8_t *dclean = nn_malloc(n), *dlow = nn_malloc(n); void *dx = nn_malloc(n * 2 * cfg.cin), *scr = nn_malloc(sb_); double *dst = nn_malloc(8 * sizeof(double));
        nn_h2d(dclean, c, n);
        if (nn_restore_degrade(dclean, dlow, D, Hh, Wd, &r, scr, sb_, nullptr)) { fprintf(stderr, "restore-eval: scratch\n"); return 1; }
        nn_restore_input(dlow, dx, n, cfg.cin == 2, nn_get_f16());
        const float *lg = unet_forward_x(u, dx, (shape5){1, cfg.cin, D, Hh, Wd}, 0, 1);
        nn_restore_loss(lg, unet_logits_h16(u), dlow, dclean, D, Hh, Wd, 0.01f, 0, 0, nullptr, 0, 1, 1.f, dst);
        double s[8]; nn_d2h(s, dst, sizeof s);
        const double p = 10 * log10(65025.0 * s[0] / s[3]), pb = 10 * log10(65025.0 * s[0] / s[4]);
        printf("%s %s: output %.3f dB, input %.3f dB, gain %+.3f dB (mse %.3f vs %.3f, %.0f voxels)\n", argv[i], fam, p, pb, p - pb, s[3] / s[0], s[4] / s[0], s[0]);
        se += s[3]; sb += s[4]; sn += s[0];
        nn_free(dclean); nn_free(dlow); nn_free(dx); nn_free(scr); nn_free(dst); free(c);
    }
    if (sn > 0) printf("overall %s: output %.3f dB, input %.3f dB, gain %+.3f dB (mse %.3f vs %.3f)\n", fam, 10 * log10(65025.0 * sn / se), 10 * log10(65025.0 * sn / sb),
                       10 * log10(sb / se), se / sn, sb / sn);
    return 0;
}

/* calibration statistics of a uint8 volume over voxels > 0: mean, sd, a noise estimate (robust sd of v - mean of the 6 neighbours,
   / sqrt(7/6)) and the structure function D(k) = mean (v(x + k) - v(x))^2 along the axes for lags 1, 2, 3, 4, 6, 8 */
static void vol_stats(const uint8_t *v, int D, int H, int W, double out[10]) {
    double s = 0, s2 = 0, n = 0;
    const int lags[6] = {1, 2, 3, 4, 6, 8};
    double dk[6] = {0}, nk[6] = {0};
    int *hist = calloc(3061, sizeof(int)); long nh = 0;
    for (int z = 0; z < D; z++) for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        const size_t i = ((size_t)z * H + y) * W + x; const int a = v[i];
        if (!a) continue;
        s += a; s2 += (double)a * a; n++;
        for (int k = 0; k < 6; k++) {
            const int L = lags[k];
            if (x + L < W && v[i + L]) { const double d = a - v[i + L]; dk[k] += d * d; nk[k]++; }
            if (y + L < H && v[i + (size_t)L * W]) { const double d = a - v[i + (size_t)L * W]; dk[k] += d * d; nk[k]++; }
            if (z + L < D && v[i + (size_t)L * H * W]) { const double d = a - v[i + (size_t)L * H * W]; dk[k] += d * d; nk[k]++; }
        }
        if (z > 0 && z < D - 1 && y > 0 && y < H - 1 && x > 0 && x < W - 1) {
            const size_t hw = (size_t)H * W; const int nb[6] = {v[i - 1], v[i + 1], v[i - W], v[i + W], v[i - hw], v[i + hw]};
            int ok = 1, sn = 0; for (int k = 0; k < 6; k++) { ok &= nb[k] != 0; sn += nb[k]; }
            if (ok) { const int b = 6 * a - sn + 1530; hist[b]++; nh++; }   /* 6 (v - mean6), integer */
        }
    }
    double mad = 0;
    if (nh) {   /* the median of |v - mean6| (half-DN bins) */
        long half = nh / 2, acc = hist[1530]; int c = 0;
        for (int d = 1; d <= 1530 && acc < half; d++) { acc += hist[1530 + d] + hist[1530 - d]; c = d; }
        mad = c / 6.0;
    }
    free(hist);
    out[0] = n; out[1] = n ? s / n : 0; out[2] = n ? sqrt(fmax(0, s2 / n - out[1] * out[1])) : 0; out[3] = 1.4826 * mad / sqrt(7.0 / 6.0);
    for (int k = 0; k < 6; k++) out[4 + k] = nk[k] ? dk[k] / nk[k] : 0;
}
/* ufsm restore-stats <root> <key> --um U --level L --box z,y,x,nz,ny,nx [--degrade scanner|pyramid --step S --psf P --noise N --sigvar V
   --corr C --edge E --q Q --seed S] [--out coarse.raw]: one JSON line of vol_stats, of the box as stored or of its degraded coarse grid */
int cmd_restore_stats(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm restore-stats <root> <key> --um U [--level 0] --box z,y,x,nz,ny,nx [--degrade scanner|pyramid --step 3.9 --psf 0.5 --noise 2 --sigvar 0 --corr 0 --edge 0 --q 8 --seed 1] [--out coarse.raw]\n"); return 2; }
    long long v[6]; if (sscanf(opt(argc, argv, "--box", ""), "%lld,%lld,%lld,%lld,%lld,%lld", v, v + 1, v + 2, v + 3, v + 4, v + 5) != 6) { fprintf(stderr, "--box z,y,x,nz,ny,nx\n"); return 2; }
    store *st = store_open(argv[2]);
    z3 *ct = st ? pyramid_open_level(st, argv[3], atoi(opt(argc, argv, "--level", "0")), atof(opt(argc, argv, "--um", "1")), opt(argc, argv, "--cache", nullptr)) : nullptr;
    if (!ct) { fprintf(stderr, "restore-stats: cannot open %s/%s\n", argv[2], argv[3]); return 1; }
    const int64_t o[3] = {v[0], v[1], v[2]}, nn3[3] = {v[3], v[4], v[5]};
    const int D = (int)v[3], H = (int)v[4], W = (int)v[5]; const size_t n = (size_t)D * H * W;
    uint8_t *h = malloc(n);
    if (z3_read(ct, o, nn3, h, 16)) { fprintf(stderr, "restore-stats: %s\n", z3_error()); return 1; }
    const char *dg = opt(argc, argv, "--degrade", nullptr);
    if (!dg) {
        double s[10]; vol_stats(h, D, H, W, s);
        printf("{\"source\":\"%s\",\"box\":[%lld,%lld,%lld,%lld,%lld,%lld],\"degrade\":\"none\",\"grid\":[%d,%d,%d],\"n\":%.0f,\"mean\":%.3f,\"sd\":%.3f,\"noise_est\":%.3f,\"D\":[%.2f,%.2f,%.2f,%.2f,%.2f,%.2f]}\n",
               argv[3], v[0], v[1], v[2], v[3], v[4], v[5], D, H, W, s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[9]);
        return 0;
    }
    /* --psf / --noise / --corr: comma lists, every combination on the same upload */
    float pl[32], nl[32], cl[32]; int np = 0, nq = 0, nc = 0;
    { char *t = strdup(opt(argc, argv, "--psf", "0.5")); for (char *q = strtok(t, ","); q && np < 32; q = strtok(nullptr, ",")) pl[np++] = (float)atof(q); free(t); }
    { char *t = strdup(opt(argc, argv, "--noise", "0")); for (char *q = strtok(t, ","); q && nq < 32; q = strtok(nullptr, ",")) nl[nq++] = (float)atof(q); free(t); }
    { char *t = strdup(opt(argc, argv, "--corr", "0")); for (char *q = strtok(t, ","); q && nc < 32; q = strtok(nullptr, ",")) cl[nc++] = (float)atof(q); free(t); }
    if (nn_init(atoi(opt(argc, argv, "--gpu", "0")))) return 1;
    const float step = (float)atof(opt(argc, argv, "--step", "4"));
    uint8_t *dclean = nn_malloc(n), *dlow = nn_malloc(n);
    const size_t sb = restore_scratch_bytes(D, H, W, fminf(step, 4.f)); void *scr = nn_malloc(sb);
    nn_h2d(dclean, h, n);
    const char *op = opt(argc, argv, "--out", nullptr);
    for (int a1 = 0; a1 < np; a1++) for (int a2 = 0; a2 < nq; a2++) for (int a3 = 0; a3 < nc; a3++) {
        restore_recipe r; memset(&r, 0, sizeof r);
        r.family = !strcmp(dg, "pyramid") ? RF_PYRAMID : RF_SCANNER; r.gain = r.gamma = 1.f; r.seed = (unsigned)atoi(opt(argc, argv, "--seed", "1"));
        const int dim[3] = {D, H, W};
        for (int a = 0; a < 3; a++) {
            r.small[a] = r.family == RF_PYRAMID ? dim[a] / 4 : (int)lrintf(dim[a] / step); r.step[a] = (float)dim[a] / r.small[a];
            r.psf[a] = pl[a1] * r.step[a]; r.qphase[a] = (int)(r.seed * (a + 3)) & 15;
        }
        r.box = r.family == RF_PYRAMID; r.q = (float)atof(opt(argc, argv, "--q", r.family == RF_PYRAMID ? "2" : "8"));
        r.noise = nl[a2]; r.sig_var = (float)atof(opt(argc, argv, "--sigvar", "0")); r.corr = cl[a3]; r.edge = (float)atof(opt(argc, argv, "--edge", "0"));
        const int sd = r.small[0], sh = r.small[1], sw = r.small[2]; const size_t ns = (size_t)sd * sh * sw;
        uint8_t *dco = nn_malloc(ns), *res = malloc(ns);
        if (nn_restore_degrade(dclean, dlow, D, H, W, &r, scr, sb, dco)) { fprintf(stderr, "restore-stats: scratch\n"); return 1; }
        nn_d2h(res, dco, ns);
        double s[10]; vol_stats(res, sd, sh, sw, s);
        printf("{\"source\":\"%s\",\"box\":[%lld,%lld,%lld,%lld,%lld,%lld],\"degrade\":\"%s\",\"step\":%.4g,\"psf\":%.4g,\"noise\":%.4g,\"sigvar\":%.4g,\"corr\":%.4g,\"edge\":%.4g,\"q\":%.4g,"
               "\"grid\":[%d,%d,%d],\"n\":%.0f,\"mean\":%.3f,\"sd\":%.3f,\"noise_est\":%.3f,\"D\":[%.2f,%.2f,%.2f,%.2f,%.2f,%.2f]}\n",
               argv[3], v[0], v[1], v[2], v[3], v[4], v[5], dg, r.step[0], pl[a1], r.noise, r.sig_var, r.corr, r.edge, r.q, sd, sh, sw, s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[9]);
        fflush(stdout);
        if (op) { FILE *f = fopen(op, "wb"); if (f) { fwrite(res, 1, ns, f); fclose(f); } }
        nn_free(dco); free(res);
    }
    nn_free(dclean); nn_free(dlow); nn_free(scr);
    return 0;
}

/* ufsm restore-eval-pairs <ckpt> <sources.json> [--max 64] [--all 0] [--ema 1] [--gpu 0] [--csv FILE]: every validation box of the real-pair
   sources (src/srpair.h: the "test" boxes, every box of a "val" pair; --all 1: every box) at the full box size through the restore net; per
   pair and pooled: PSNR, SSIM (8^3 blocks) and gradient PSNR (face differences) of the output and of the trilinear x4 input against the
   target (after the pair's intensity map), from nn_restore_pair_loss's metric sums */
int cmd_restore_eval_pairs(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm restore-eval-pairs <ckpt> <sources.json> [--max 64] [--all 0] [--ema 1] [--gpu 0] [--csv FILE]\n"); return 2; }
    if (nn_init(atoi(opt(argc, argv, "--gpu", "0")))) return 1;
    unet_cfg cfg; unet *u = restore_net(argv[2], atoi(opt(argc, argv, "--ema", "1")) != 0, &cfg);
    if (!u) return 1;
    sources *S = sources_load(argv[3]); if (!S) return 1;
    const int mx = atoi(opt(argc, argv, "--max", "64")), all = atoi(opt(argc, argv, "--all", "0"));
    FILE *cf = *opt(argc, argv, "--csv", "") ? fopen(opt(argc, argv, "--csv", ""), "w") : nullptr;
    if (cf) fprintf(cf, "pair,boxes,voxels,psnr,input_psnr,gain,ssim,input_ssim,grad_psnr,input_grad_psnr\n");
    const restore_pair_w w = {0.01f, 0.f, 0.f, 0.f, 0.f, 0, 0.f};
    double tot[RP_NST] = {0};
#define PS(sse, n) ((n) > 0 && (sse) > 0 ? 10.0 * log10(65025.0 * (n) / (sse)) : 99.0)
    for (int si = 0; si < S->n; si++) {
        const srpair *p = S->src[si].srp; if (!p) continue;
        const int T = p->T, q = p->same ? T : T / 4; const size_t t3 = (size_t)T * T * T, q3 = (size_t)q * q * q;
        if (cfg.scale_cond) unet_set_cond(u, (float)log2(p->ratio > 0 ? p->ratio : 4) - 2.f);   /* (ratio-conditioned nets) */
        if (T % (1 << (cfg.nlev - 1))) { fprintf(stderr, "restore-eval-pairs: %s: box %d not a multiple of %d\n", S->src[si].name, T, 1 << (cfg.nlev - 1)); continue; }
        uint8_t *ht = malloc(t3), *hc = malloc(q3), *dt = nn_malloc(t3), *dc = nn_malloc(q3), *dlow = nn_malloc(t3); void *dx = nn_malloc(t3 * 2 * cfg.cin);
        double *dst = nn_malloc(RP_NST * sizeof(double)); float *rd = nn_malloc(q3 * sizeof(float) + 256);
        double acc[RP_NST] = {0}; int nb = 0;
        const int nlist = all ? p->n : p->nte;
        for (int i = 0; i < nlist && nb < mx; i++) {
            const int k = all ? i : p->te[i]; const int oc[3] = {0, 0, 0};
            if (srpair_read(p, k, oc, T, ht, hc)) { fprintf(stderr, "restore-eval-pairs: %s box %d unreadable\n", S->src[si].name, k); continue; }
            nn_h2d(dt, ht, t3); nn_h2d(dc, hc, q3);
            const float st4[3] = {4.f, 4.f, 4.f}, ph0[3] = {0.f, 0.f, 0.f};
            if (p->same) nn_d2d(dlow, dc, t3); else nn_restore_upsample(dc, q, q, q, dlow, T, T, T, st4, ph0, nullptr);
            nn_restore_input(dlow, dx, t3, cfg.cin == 2, nn_get_f16());
            const float *lg = unet_forward_x(u, dx, (shape5){1, cfg.cin, T, T, T}, 0, 1);
            nn_restore_pair_loss(lg, unet_logits_h16(u), dlow, dt, p->same ? nullptr : dc, T, T, T, &w, nullptr, 0, nn_get_f16(), 1.f, dst, rd, 1);
            double s[RP_NST]; nn_d2h(s, dst, sizeof s);
            for (int c = 0; c < RP_NST; c++) acc[c] += s[c];
            nb++;
        }
        for (int c = 0; c < RP_NST; c++) tot[c] += acc[c];
        const double ss = acc[13] > 0 ? acc[11] / acc[13] : 0, sb = acc[13] > 0 ? acc[12] / acc[13] : 0;
        printf("%-16s %3d boxes: psnr %.3f (input %.3f, gain %+.3f) ssim %.4f (input %.4f) grad psnr %.3f (input %.3f)\n", S->src[si].name, nb,
               PS(acc[3], acc[0]), PS(acc[4], acc[0]), PS(acc[3], acc[0]) - PS(acc[4], acc[0]), ss, sb, PS(acc[9], acc[5]), PS(acc[10], acc[5]));
        if (cf) fprintf(cf, "%s,%d,%.0f,%.4f,%.4f,%.4f,%.5f,%.5f,%.4f,%.4f\n", S->src[si].name, nb, acc[0], PS(acc[3], acc[0]), PS(acc[4], acc[0]), PS(acc[3], acc[0]) - PS(acc[4], acc[0]), ss, sb, PS(acc[9], acc[5]), PS(acc[10], acc[5]));
        free(ht); free(hc); nn_free(dt); nn_free(dc); nn_free(dlow); nn_free(dx); nn_free(dst); nn_free(rd);
    }
    printf("%-16s          : psnr %.3f (input %.3f, gain %+.3f) ssim %.4f (input %.4f) grad psnr %.3f (input %.3f)\n", "pooled",
           PS(tot[3], tot[0]), PS(tot[4], tot[0]), PS(tot[3], tot[0]) - PS(tot[4], tot[0]), tot[13] > 0 ? tot[11] / tot[13] : 0, tot[13] > 0 ? tot[12] / tot[13] : 0, PS(tot[9], tot[5]), PS(tot[10], tot[5]));
#undef PS
    if (cf) fclose(cf);
    sources_free(S); unet_free(u);
    return 0;
}

/* ufsm restore-packed <ckpt|none> <pair-dir> --codes ROOT,KEY --g G [--lut 1] [--spacing-um 100] [--max 64] [--all 0] [--rmax 24] [--block 16]
     [--delta 12] [--ema 1] [--gpu 0]
   Packed-region SR metrics on an sr-pairs-layout set with ground-truth sheets: --codes is a winding_mod14 code raster (labelled rectos,
   src/band.h) on a grid of G input voxels per code voxel (Paris 4 q-9p6 on the 2.4 um scan's level 2: G 1; 0139 / 0814 codes at 18.72 um on
   the native 9.36 um scans: G 2). Per box the codes around it give the winding phase and the local sheet spacing (band_field_phase_pair);
   target voxels whose spacing is below --spacing-um are packed. For the trilinear input, the SR output (ckpt; "none" skips it) and the real
   target: PSNR and gradient PSNR against the target inside packed voxels (target identical: omitted), and the gap test on packed
   --block^3 target blocks (half packed): the real target's top / bottom quartile voxels are the sheet / gap voxels; each image's mean
   over them gives the sheet-gap contrast it shows where the real scan has it (DN); per block "open" = an image's own interquartile range
   > --delta; versus the real target: open in both / only in the prediction (hallucinated) / only in the real (merged). */
int cmd_restore_packed(int argc, char **argv) {   /* (--weights FILE --boost K: write the per-slot packed fraction, for box_weights.json) */
    if (argc < 4) { fprintf(stderr, "usage: ufsm restore-packed <ckpt|none> <pair-dir> --codes ROOT,KEY --g G [--lut 1] [--spacing-um 100] [--max 64] [--all 0] [--rmax 24] [--block 16] [--delta 12]\n"); return 2; }
    const char *cs = opt(argc, argv, "--codes", nullptr); const double G = atof(opt(argc, argv, "--g", "1"));
    if (!cs || !(G > 0)) { fprintf(stderr, "restore-packed: --codes ROOT,KEY and --g\n"); return 2; }
    char cb[2048]; snprintf(cb, sizeof cb, "%s", cs); char *cm = strchr(cb, ','); if (!cm) { fprintf(stderr, "--codes ROOT,KEY\n"); return 2; } *cm = 0;
    store *cst = store_open(cb); z3 *cz = cst ? z3_open(cst, cm + 1, nullptr) : nullptr;
    if (!cz) { fprintf(stderr, "restore-packed: cannot open codes %s/%s\n", cb, cm + 1); return 1; }
    const z3_meta *cmeta = z3_meta_of(cz);
    srpair *p = srpair_open(argv[3], "val", atoi(opt(argc, argv, "--lut", "1")), 0); if (!p) return 1;
    if (p->same) { fprintf(stderr, "restore-packed: coarse-box sets only\n"); return 2; }
    const int use_net = strcmp(argv[2], "none") != 0; unet_cfg cfg; unet *u = nullptr;
    if (use_net) { if (nn_init(atoi(opt(argc, argv, "--gpu", "0")))) return 1; u = restore_net(argv[2], atoi(opt(argc, argv, "--ema", "1")) != 0, &cfg); if (!u) return 1;
                   if (cfg.scale_cond) unet_set_cond(u, (float)log2(p->ratio) - 2.f); }
    /* box origins (input-level zyx) */
    char bp[2100]; snprintf(bp, sizeof bp, "%s/boxes.json", argv[3]);
    FILE *f = fopen(bp, "rb"); if (!f) return 1; fseek(f, 0, SEEK_END); const long L = ftell(f); fseek(f, 0, SEEK_SET);
    char *t = malloc(L + 1); if (fread(t, 1, L, f) != (size_t)L) { fclose(f); return 1; } fclose(f); t[L] = 0;
    json *bj = json_parse(t, (size_t)L); free(t); const json *bx = bj ? json_get(bj, "boxes") : nullptr; if (!bx) return 1;
    const double in_um = json_num(json_get(json_get(bj, "input"), "um"), 9.362), spum = atof(opt(argc, argv, "--spacing-um", "100"));
    const int T = p->T, B = p->B, mx = atoi(opt(argc, argv, "--max", "64")), all = atoi(opt(argc, argv, "--all", "0")), BL = atoi(opt(argc, argv, "--block", "16"));
    const float rmax = (float)atof(opt(argc, argv, "--rmax", "24")); const double delta = atof(opt(argc, argv, "--delta", "12"));
    const size_t t3 = (size_t)T * T * T, q3 = (size_t)B * B * B;
    uint8_t *ht = malloc(t3), *hc = malloc(q3), *hlow = malloc(t3), *hout = malloc(t3), *pk = malloc(t3), *ph = malloc(t3);
    uint8_t *dt = use_net ? nn_malloc(t3) : nullptr, *dc = nn_malloc(q3), *dlow = nn_malloc(t3), *dout = use_net ? nn_malloc(t3) : nullptr; void *dx = use_net ? nn_malloc(t3 * 2 * cfg.cin) : nullptr;
    (void)dt;
    /* accumulators: [0 trilinear, 1 SR]: packed n, se, grad n, grad se; gap blocks: sum S-G per image [trilinear, SR, real], counts, agreement */
    double pn[2] = {0}, pse[2] = {0}, gn[2] = {0}, gse[2] = {0}, an = 0, ase[2] = {0};
    double sg[3] = {0}; long nblk = 0, agree[2][3] = {{0}};   /* agree[pred][0 both open, 1 pred only (hallucinated), 2 real only (merged)] */
    size_t npk = 0, nall = 0;
    const char *wout = opt(argc, argv, "--weights", nullptr);   /* per-slot packed fraction -> FILE ("sr_pair": {"weights": FILE}: training draws) */
    double *wbox = wout ? calloc((size_t)p->n, sizeof *wbox) : nullptr;
    const int nlist = all ? p->n : p->nte;
    for (int ii = 0, done = 0; ii < nlist && done < mx; ii++) {
        const int k = all ? ii : p->te[ii];
        const json *b = nullptr; for (size_t q = 0; q < bx->n; q++) if ((int)json_num(json_get(json_at(bx, q), "slot"), -1) == k) { b = json_at(bx, q); break; }
        if (!b) continue;
        int64_t oz[3];   /* input-level zyx origin */
        const json *ol = json_get(b, "origin_level_zyx"), *ox = json_get(b, "origin_xyz");
        if (ol) for (int d = 0; d < 3; d++) oz[d] = (int64_t)json_num(json_at(ol, (size_t)d), 0);
        else if (ox) for (int d = 0; d < 3; d++) oz[d] = (int64_t)json_num(json_at(ox, (size_t)(2 - d)), 0);
        else continue;
        const int oc[3] = {0, 0, 0};
        if (srpair_read(p, k, oc, T, ht, hc)) continue;
        /* codes crop: the box in code voxels plus rmax */
        const int m = (int)ceil(rmax) + 2; int64_t co[3], cn[3]; int n3[3];
        for (int d = 0; d < 3; d++) { co[d] = (int64_t)floor(oz[d] / G) - m; cn[d] = (int64_t)ceil((oz[d] + B) / G) + m - co[d]; n3[d] = (int)cn[d]; }
        const size_t c3 = (size_t)cn[0] * cn[1] * cn[2];
        uint8_t *cc = calloc(c3, 1), *cph = malloc(c3), *csp = malloc(c3);
        int64_t lo[3], hi[3], rn[3]; int ok = 1;
        for (int d = 0; d < 3; d++) { lo[d] = co[d] < 0 ? 0 : co[d]; hi[d] = co[d] + cn[d] > cmeta->shape[d] ? cmeta->shape[d] : co[d] + cn[d]; rn[d] = hi[d] - lo[d]; if (rn[d] <= 0) ok = 0; }
        if (ok) {
            uint8_t *r = malloc((size_t)rn[0] * rn[1] * rn[2]);
            if (z3_read(cz, lo, rn, r, 4)) ok = 0;
            else for (int64_t z = 0; z < rn[0]; z++) for (int64_t y = 0; y < rn[1]; y++) memcpy(cc + ((size_t)(z + lo[0] - co[0]) * cn[1] + (y + lo[1] - co[1])) * cn[2] + (lo[2] - co[2]), r + ((size_t)z * rn[1] + y) * rn[2], (size_t)rn[2]);
            free(r);
        }
        size_t lab = 0; for (size_t i = 0; i < c3; i++) lab += cc[i] != 0 && cc[i] != 255;
        if (!ok || lab < 64 || band_field_phase_pair(cc, n3, rmax, cph, csp)) { free(cc); free(cph); free(csp); continue; }
        {   /* labelled recto voxels carry spacing 0: take their neighbours' spacing (the sheet itself is part of the packed cell) */
            uint8_t *sp2 = malloc(c3); memcpy(sp2, csp, c3);
            for (int64_t z = 1; z < cn[0] - 1; z++) for (int64_t y = 1; y < cn[1] - 1; y++) for (int64_t x = 1; x < cn[2] - 1; x++) {
                const size_t i = ((size_t)z * cn[1] + y) * cn[2] + x; if (cph[i] != 0 || csp[i]) continue;
                const size_t nb[6] = {i - 1, i + 1, i - (size_t)cn[2], i + (size_t)cn[2], i - (size_t)cn[1] * cn[2], i + (size_t)cn[1] * cn[2]}; uint8_t mx = 0;
                for (int q = 0; q < 6; q++) if (csp[nb[q]] > mx) mx = csp[nb[q]];
                sp2[i] = mx;
            }
            memcpy(csp, sp2, c3); free(sp2);
        }
        /* per target voxel: the nearest code voxel's phase / packed flag */
        size_t np_ = 0;
        for (int z = 0; z < T; z++) for (int y = 0; y < T; y++) for (int x = 0; x < T; x++) {
            const int tv[3] = {z, y, x}; int64_t ci[3];
            for (int d = 0; d < 3; d++) { const double in = oz[d] + (tv[d] + 0.5) / 4.0 - 0.5; ci[d] = (int64_t)floor((in + 0.5) / G) - co[d]; }
            const size_t c = ((size_t)ci[0] * cn[1] + ci[1]) * cn[2] + ci[2], i = ((size_t)z * T + y) * T + x;
            const uint8_t sp = csp[c];
            ph[i] = cph[c]; pk[i] = sp > 0 && sp * G * in_um < spum && ht[i] && cph[c] != PHASE_UNKNOWN; np_ += pk[i];
        }
        free(cc); free(cph); free(csp);
        /* trilinear input and the SR output */
        nn_h2d(dc, hc, q3);
        const float st4[3] = {4.f, 4.f, 4.f}, ph0[3] = {0.f, 0.f, 0.f};
        nn_restore_upsample(dc, B, B, B, dlow, T, T, T, st4, ph0, nullptr); nn_d2h(hlow, dlow, t3);
        if (use_net) {
            nn_restore_input(dlow, dx, t3, cfg.cin == 2, nn_get_f16());
            const float *lg = unet_forward_x(u, dx, (shape5){1, cfg.cin, T, T, T}, 0, 1);
            nn_restore_output(lg, unet_logits_h16(u), dlow, dout, t3); nn_d2h(hout, dout, t3);
        }
        const uint8_t *img[3] = {hlow, use_net ? hout : nullptr, ht};
        const size_t str[3] = {(size_t)T * T, (size_t)T, 1};
        for (size_t i = 0; i < t3; i++) {
            if (!ht[i]) continue;
            nall++; for (int a = 0; a < 2; a++) if (img[a] && img[a][i]) { const double e = (double)img[a][i] - ht[i]; if (a == 0 || use_net) ase[a] += e * e; }
            an++;
            if (!pk[i]) continue;
            npk++;
            for (int a = 0; a < 2; a++) {
                if (!img[a] || !img[a][i]) continue;
                const double e = (double)img[a][i] - ht[i]; pn[a]++; pse[a] += e * e;
                for (int d = 0; d < 3; d++) { if ((int)((i / str[d]) % T) == T - 1) continue; const size_t j = i + str[d]; if (!ht[j] || !img[a][j]) continue;
                    const double g = ((double)img[a][j] - img[a][i]) - ((double)ht[j] - ht[i]); gn[a]++; gse[a] += g * g; }
            }
        }
        for (int Z = 0; Z + BL <= T; Z += BL) for (int Y = 0; Y + BL <= T; Y += BL) for (int X = 0; X + BL <= T; X += BL) {
            /* packed block: the real target's top / bottom quartile voxels are its sheet / gap voxels; each image's mean over them (the
               contrast it shows where the real scan has sheet and gap), and each image's own interquartile range (open: > delta) */
            int hr[3][256] = {{0}}; long nv = 0, npkb = 0;
            for (int z = 0; z < BL; z++) for (int y = 0; y < BL; y++) for (int x = 0; x < BL; x++) {
                const size_t i = ((size_t)(Z + z) * T + Y + y) * T + X + x; if (!ht[i] || !hlow[i] || (use_net && !hout[i])) continue;
                npkb += pk[i]; nv++; for (int a = 0; a < 3; a++) if (img[a]) hr[a][img[a][i]]++;
            }
            if (nv < BL * BL * BL / 2 || npkb < nv / 2) continue;
            int q25[3] = {0}, q75[3] = {0};
            for (int a = 0; a < 3; a++) { if (!img[a]) continue; long c = 0; int g25 = 0;
                for (int v = 0; v < 256; v++) { c += hr[a][v]; if (!g25 && c >= nv / 4) { q25[a] = v; g25 = 1; } if (c >= 3 * nv / 4) { q75[a] = v; break; } } }
            double s[3] = {0}, g[3] = {0}; long ns = 0, ng = 0;
            for (int z = 0; z < BL; z++) for (int y = 0; y < BL; y++) for (int x = 0; x < BL; x++) {
                const size_t i = ((size_t)(Z + z) * T + Y + y) * T + X + x; if (!ht[i] || !hlow[i] || (use_net && !hout[i])) continue;
                const int sheet = ht[i] >= q75[2], gap = ht[i] <= q25[2]; if (!sheet && !gap) continue;
                for (int a = 0; a < 3; a++) { if (!img[a]) continue; if (sheet) s[a] += img[a][i]; else g[a] += img[a][i]; }
                if (sheet) ns++; else ng++;
            }
            if (!ns || !ng) continue;
            nblk++;
            for (int a = 0; a < 3; a++) sg[a] += img[a] ? s[a] / ns - g[a] / ng : 0;
            const int ro = q75[2] - q25[2] > delta;
            for (int a = 0; a < 2; a++) { if (!img[a]) continue; const int po = q75[a] - q25[a] > delta;
                if (po && ro) agree[a][0]++; else if (po && !ro) agree[a][1]++; else if (!po && ro) agree[a][2]++; }
        }
        if (wbox) wbox[k] = (double)np_ / (double)t3;
        done++;
    }
#define PS(sse, n) ((n) > 0 && (sse) > 0 ? 10.0 * log10(65025.0 * (n) / (sse)) : 99.0)
    printf("%s: packed voxels %.1f%% of %zu (spacing < %.0f um)\n", argv[3], nall ? 100.0 * npk / nall : 0, nall, spum);
    const char *nm[2] = {"trilinear", "SR"};
    for (int a = 0; a < 2; a++) {
        if (a == 1 && !use_net) continue;
        const long o = agree[a][0] + agree[a][1] + agree[a][2];
        printf("  %-9s psnr all %.3f packed %.3f  grad psnr packed %.3f | sheet-gap contrast %.2f DN (real %.2f) over %ld packed blocks; IQR > delta in both %ld, %s only (hallucinated) %ld, real only (merged) %ld\n",
               nm[a], PS(ase[a], an), PS(pse[a], pn[a]), PS(gse[a], gn[a]), nblk ? sg[a] / nblk : 0, nblk ? sg[2] / nblk : 0, nblk, agree[a][0], nm[a], agree[a][1], agree[a][2]);
        (void)o;
    }
#undef PS
    if (wbox) { FILE *wf = fopen(wout, "w"); if (wf) { fprintf(wf, "{\"what\":\"packed fraction per slot (restore-packed, spacing < %g um)\",\"boost\":%s,\"w\":[", spum, opt(argc, argv, "--boost", "4"));
                for (int i = 0; i < p->n; i++) fprintf(wf, "%s%.4f", i ? "," : "", wbox[i]); fprintf(wf, "]}\n"); fclose(wf); } free(wbox); }
    json_free(bj); srpair_free(p); free(ht); free(hc); free(hlow); free(hout); free(pk); free(ph);
    return 0;
}
