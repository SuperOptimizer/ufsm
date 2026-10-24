/* Task restore command-line paths (src/restore.h):
     ufsm predict <restore ckpt> ...      super-resolve a CT volume (restore_predict, called by cmd_predict)
     ufsm restore-eval <ckpt> <clean.npy>...   PSNR of the net on clean crops degraded by a fixed recipe (comparison with older runs)
     ufsm restore-stats <root> <key> ...  calibration statistics of a CT box, as stored or after a scanner degradation */
#include "nn.h"
#include "checkpoint.h"
#include "pyramid.h"
#include "restore.h"
#include "sources.h"
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
                        "       [--shard 512] [--halo 32] [--q 2] [--levels 1] [--ema 1] [--gpu 0] [--cache DIR] [--shards FILE (\"sz sy sx\" lines: only these)]\n"
                        "  restores level L of the CT onto the --out-um grid (trilinear input, residual net), uint8 CT in 512^3 shards\n", fine_um);
        return 2;
    }
    if (nn_init(atoi(opt(argc, argv, "--gpu", "0")))) { fprintf(stderr, "restore: cannot select the GPU\n"); return 1; }
    unet_cfg cfg; unet *u = restore_net(ckpt, use_ema, &cfg);
    if (!u) return 1;
    const int W = S + 2 * H;
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
        nn_restore_input(dlow, dx, w3, cfg.cin == 2, nn_get_f16());
        const float *lg = unet_forward_x(u, dx, (shape5){1, cfg.cin, W, W, W}, 0, 1);
        nn_restore_output(lg, unet_logits_h16(u), dlow, dout, w3);
        nn_d2h(ho, dout, w3);
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
