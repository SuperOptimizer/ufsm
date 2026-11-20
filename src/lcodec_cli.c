/* ufsm codec-fit: stage 1 of the learned input codec (lcodec.h) and its pre-check against the KLT at equal real bytes. 24 mostly-material
   256^3 boxes of one CT level (latvol's sample lattice and sd_ref): the even ones train the codec, the odd ones are the evaluation, where
   every arm (the KLT ladder: C x step; the learned codec) is encoded as latvol chunks with tables fitted on the evaluation boxes themselves
   (the same rule for every arm) and scored by stream bytes per box voxel and PSNR over the cells holding a nonzero byte (as the readers
   see them: lv_inverse, air cells 0). */
#include "lcodec.h"
#include "latfit.h"
#include "latvol.h"
#include "optcheck.h"
#include "store.h"
#include "zarr3.h"
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const char *opt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], opt_n(name))) return argv[i + 1];
    return dflt;
}
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec; }
enum { E = 256, NCB = E / 4, NCELL = NCB * NCB * NCB };

/* latvol_cli.c's sample lattice (the 5^3 points by distance from the centre, mostly-material boxes, 128-aligned) */
static int sample_boxes(z3 *z, const int64_t hi[3], int nmax, uint8_t **out, int nth) {
    int ord[125]; double dist[125]; int used = 0;
    for (int k = 0; k < 125; k++) { const int a = k / 25 - 2, b = k / 5 % 5 - 2, c = k % 5 - 2; ord[k] = k; dist[k] = a * a + b * b + c * c + 1e-3 * k; }
    for (int a = 1; a < 125; a++) for (int b = a; b > 0 && dist[ord[b]] < dist[ord[b - 1]]; b--) { const int t = ord[b]; ord[b] = ord[b - 1]; ord[b - 1] = t; }
    const size_t ev = (size_t)E * E * E; int64_t seen[125][3]; int nseen = 0;
    for (int q = 0; q < 125 && used < nmax; q++) {
        const int k = ord[q], l[3] = {k / 25, k / 5 % 5, k % 5}; int64_t o[3]; const int64_t nn[3] = {E, E, E}; int ok = 1;
        for (int d = 0; d < 3; d++) { o[d] = (int64_t)((l[d] + 0.5) * (double)hi[d] / 5) / 128 * 128; if (o[d] + E > hi[d]) o[d] = (hi[d] - E) / 128 * 128; if (o[d] < 0) ok = 0; }
        if (!ok) continue;
        int dup = 0; for (int p = 0; p < nseen; p++) dup |= seen[p][0] == o[0] && seen[p][1] == o[1] && seen[p][2] == o[2];
        if (dup) continue;
        memcpy(seen[nseen++], o, sizeof o);
        if (z3_box_any(z, o, nn) != 1) continue;
        uint8_t *ct = malloc(ev); if (!ct) return -1;
        if (z3_read(z, o, nn, ct, nth)) { free(ct); return -1; }
        size_t nzv = 0; for (size_t v = 0; v < ev; v += 7) nzv += ct[v] != 0;
        if (nzv * 7 < ev / 2) { free(ct); continue; }
        out[used++] = ct;
    }
    return used;
}
/* a box's cells on the full KLT basis, x units: xg [NCELL][64]; any [NCELL] */
static void box_coeffs(const uint8_t *ct, const lat_basis *lb, float sd, float *xg, uint8_t *any) {
    for (int c = 0; c < NCELL; c++) {
        const int z = c / (NCB * NCB), y = c / NCB % NCB, x = c % NCB; float v[64]; int a = 0;
        for (int k = 0; k < 64; k++) { const uint8_t b = ct[((size_t)(4 * z + (k >> 4)) * E + 4 * y + (k >> 2 & 3)) * E + 4 * x + (k & 3)]; v[k] = b; a |= b; }
        any[c] = a != 0;
        for (int j = 0; j < 64; j++) { double s = 0; for (int k = 0; k < 64; k++) s += (double)lb->b[j][k] * v[k]; xg[(size_t)c * 64 + j] = (float)(s / sd); }
    }
}
/* a 128^3 chunk g (0..7) of a box: its 32^3 cells' indices from qbox [NCELL][C] */
static void chunk_q(const int16_t *qbox, int C, int g, int16_t *q) {
    for (int z = 0; z < 32; z++) for (int y = 0; y < 32; y++) for (int x = 0; x < 32; x++) {
        const int bz = (g >> 2) * 32 + z, by = (g >> 1 & 1) * 32 + y, bx = (g & 1) * 32 + x;
        memcpy(q + ((size_t)(z * 32 + y) * 32 + x) * C, qbox + ((size_t)(bz * NCB + by) * NCB + bx) * C, (size_t)C * 2);
    }
}
static void chunk_ct(const uint8_t *ct, int g, uint8_t *o) {
    for (int z = 0; z < 128; z++) for (int y = 0; y < 128; y++) memcpy(o + ((size_t)z * 128 + y) * 128, ct + ((size_t)((g >> 2) * 128 + z) * E + (g >> 1 & 1) * 128 + y) * E + (g & 1) * 128, 128);
}
/* score indices qb[b] [NCELL][C] of the evaluation boxes with codec rows of lc0 (tables refitted on them): bytes / voxel, PSNR */
static int score(const lv_codec *lc0, const float *rows, int nb, int16_t **qb, uint8_t **ct, double *bpv, double *psnr) {
    const int C = lc0->C; uint64_t (*hist)[LV_NSYM] = calloc(LV_MAXC, sizeof *hist); int16_t *q = malloc((size_t)32768 * C * 2);
    uint8_t *ch = malloc(1 << 21), *rb = malloc(1 << 21), *enc = malloc(LV_CHUNK_BOUND); int rc = 0;
    for (int b = 0; b < nb; b++) for (int g = 0; g < 8; g++) { chunk_q(qb[b], C, g, q); lv_hist(lc0, q, hist); }
    lv_codec lc; if (lv_codec_init(&lc, C, rows, lc0->step, lc0->sd_ref, nullptr, 0, hist)) { rc = -1; goto out; }
    double bytes = 0, se = 0, n = 0;
    for (int b = 0; b < nb; b++) for (int g = 0; g < 8; g++) {
        chunk_q(qb[b], C, g, q); size_t len = 0;
        if (lv_encode_chunk(&lc, q, enc, &len)) { rc = -1; break; }
        bytes += (double)len;
        lv_inverse(&lc, q, rb); chunk_ct(ct[b], g, ch);
        for (int cz = 0; cz < 32; cz++) for (int cy = 0; cy < 32; cy++) for (int cx = 0; cx < 32; cx++) {
            int a = 0; double s = 0;
            for (int k = 0; k < 64; k++) { const size_t vi = ((size_t)(4 * cz + (k >> 4)) * 128 + 4 * cy + (k >> 2 & 3)) * 128 + 4 * cx + (k & 3); a |= ch[vi]; const double e = (double)rb[vi] - ch[vi]; s += e * e; }
            if (a) { se += s; n += 64; }
        }
    }
    lv_codec_free(&lc);
    *bpv = bytes / ((double)nb * E * E * E); *psnr = se > 0 ? 10 * log10(255.0 * 255.0 / (se / n)) : 99;
out:
    free(hist); free(q); free(ch); free(rb); free(enc);
    return rc;
}

typedef struct { const lcodec *lc; const float *f, *x, *u; int n; double lambda; float *g; double loss, R, D; } gjob;
static void *grad_run(void *a) { gjob *j = a; j->loss = lcodec_loss_grad(j->lc, j->n, j->f, j->x, j->u, j->lambda, j->g, &j->R, &j->D); return nullptr; }

int cmd_codec_fit(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: ufsm codec-fit BASIS.lcb ROOT CT-GROUP LEVEL [--cl 20 --hid 64 --lambda L --s0 1 --steps 3000 --batch 4096 --lr 0.002 --seed 1\n"
                        "        --threads 16 --boxes 24 --out CODEC.lcd --eval CODEC.lcd --klt 13,20,29 --klt-steps 0.5,0.7,1,1.4,2,2.8 (none: -)]\n"
                        "  stage 1 of the learned input codec: trains on the even sample boxes, scores it and the KLT ladder on the odd ones\n"
                        "  (real latvol stream bytes per voxel, PSNR). --eval: score a saved codec instead of training one.\n");
        return 2;
    }
    lat_basis lb; if (lat_basis_load(argv[2], &lb)) { fprintf(stderr, "codec-fit: cannot read %s\n", argv[2]); return 1; }
    const int cl = opt_atoi("--cl", opt(argc, argv, "--cl", "20")), hid = opt_atoi("--hid", opt(argc, argv, "--hid", "64")), steps = opt_atoi("--steps", opt(argc, argv, "--steps", "3000"));
    const int batch = opt_atoi("--batch", opt(argc, argv, "--batch", "4096")), nth = opt_atoi("--threads", opt(argc, argv, "--threads", "16")), nbox = opt_atoi("--boxes", opt(argc, argv, "--boxes", "24"));
    const double lambda = opt_atof("--lambda", opt(argc, argv, "--lambda", "0.01")), s0 = opt_atof("--s0", opt(argc, argv, "--s0", "1")), lr0 = opt_atof("--lr", opt(argc, argv, "--lr", "0.002"));
    const uint64_t seed = (uint64_t)opt_atoll("--seed", opt(argc, argv, "--seed", "1"));
    const char *out = opt(argc, argv, "--out", nullptr), *evp = opt(argc, argv, "--eval", nullptr), *klt = opt(argc, argv, "--klt", "13,20,29"), *ksteps = opt(argc, argv, "--klt-steps", "0.5,0.7,1,1.4,2,2.8");
    if (nth < 1 || nth > 256 || batch < nth || nbox < 2 || nbox > 64) { fprintf(stderr, "codec-fit: bad --threads / --batch / --boxes\n"); return 2; }
    store *st = store_open(argv[3]); if (!st) { fprintf(stderr, "codec-fit: cannot open %s\n", argv[3]); return 1; }
    char key[1200]; snprintf(key, sizeof key, "%s/%s", argv[4], argv[5]);
    z3 *z = z3_open(st, key, nullptr); if (!z) { fprintf(stderr, "codec-fit: cannot open %s: %s\n", key, z3_error()); return 1; }
    const z3_meta *zm = z3_meta_of(z); const int64_t hi[3] = {zm->shape[0], zm->shape[1], zm->shape[2]};
    double t0 = now(); uint8_t *ct[64]; const int ns = sample_boxes(z, hi, nbox, ct, nth);
    if (ns < 2) { fprintf(stderr, "codec-fit: %s: %d sample boxes\n", key, ns); return 1; }
    double s1 = 0, s2 = 0, nn = 0;
    for (int k = 0; k < ns; k++) for (size_t v = 0; v < (size_t)E * E * E; v += 3) { s1 += ct[k][v]; s2 += (double)ct[k][v] * ct[k][v]; nn++; }
    double sd = sqrt(fmax(s2 / nn - (s1 / nn) * (s1 / nn), 1e-6)); if (sd < 1) sd = 1;
    float *xg[64]; uint8_t *any[64];
    for (int k = 0; k < ns; k++) { xg[k] = malloc((size_t)NCELL * 64 * 4); any[k] = malloc(NCELL); if (!xg[k] || !any[k]) { fprintf(stderr, "codec-fit: out of memory\n"); return 1; } box_coeffs(ct[k], &lb, (float)sd, xg[k], any[k]); }
    int ntr = 0, nev = 0, tr[64], ev[64]; for (int k = 0; k < ns; k++) { if (k & 1) ev[nev++] = k; else tr[ntr++] = k; }
    fprintf(stderr, "codec-fit: %s: %d boxes (%d train, %d eval), sd_ref %.2f (%.1f s)\n", key, ns, ntr, nev, sd, now() - t0);
    uint8_t *ctev[64]; for (int i = 0; i < nev; i++) ctev[i] = ct[ev[i]];
    int16_t *qb[64];
    /* the KLT ladder */
    if (strcmp(klt, "-")) for (const char *pc = klt; *pc;) {
        const int C = atoi(pc);
        for (const char *ps = ksteps; *ps;) {
            const double step = atof(ps); lv_codec lc;
            if (C >= 1 && C <= 64 && step > 0 && !lv_codec_init(&lc, C, &lb.b[0][0], (float)step, (float)sd, nullptr, 0, nullptr)) {
                int16_t *q = malloc((size_t)32768 * C * 2); uint8_t *ch = malloc(1 << 21);
                for (int i = 0; i < nev; i++) { qb[i] = malloc((size_t)NCELL * C * 2);
                    for (int g = 0; g < 8; g++) { chunk_ct(ct[ev[i]], g, ch); lv_forward(&lc, ch, q);
                        for (int zz = 0; zz < 32; zz++) for (int yy = 0; yy < 32; yy++) for (int xx = 0; xx < 32; xx++)
                            memcpy(qb[i] + ((size_t)(((g >> 2) * 32 + zz) * NCB + (g >> 1 & 1) * 32 + yy) * NCB + (g & 1) * 32 + xx) * C, q + ((size_t)(zz * 32 + yy) * 32 + xx) * C, (size_t)C * 2); } }
                double bpv, ps_; if (!score(&lc, &lb.b[0][0], nev, qb, ctev, &bpv, &ps_)) printf("klt C %2d step %.3g: %.4f B/vox, PSNR %.2f dB\n", C, step, bpv, ps_);
                for (int i = 0; i < nev; i++) free(qb[i]);
                free(q); free(ch); lv_codec_free(&lc); fflush(stdout);
            }
            ps = strchr(ps, ','); if (!ps) break; ps++;
        }
        pc = strchr(pc, ','); if (!pc) break; pc++;
    }
    /* the learned codec: trained, or read */
    lcodec lc;
    if (evp) { if (lcodec_read(&lc, evp)) { fprintf(stderr, "codec-fit: cannot read %s\n", evp); return 1; } }
    else {
        float sig[64] = {0}; double ss[64] = {0}, mc = 0; size_t nm = 0;
        size_t ncand = 0; for (int i = 0; i < ntr; i++) for (int c = 0; c < NCELL; c++) ncand += any[tr[i]][c];
        uint32_t (*cand)[2] = malloc(ncand * sizeof *cand); ncand = 0;
        for (int i = 0; i < ntr; i++) for (int c = 0; c < NCELL; c++) if (any[tr[i]][c]) { cand[ncand][0] = (uint32_t)tr[i]; cand[ncand++][1] = (uint32_t)c;
                const float *x = xg[tr[i]] + (size_t)c * 64; for (int j = 0; j < 64; j++) ss[j] += (double)x[j] * x[j]; mc += 0.125 * x[0]; nm++; }
        for (int j = 1; j < 64; j++) sig[j] = (float)sqrt(ss[j] / (double)nm);
        if (lcodec_init(&lc, cl, hid, (float)sd, (float)s0, sig, (float)(mc / (double)nm), seed)) { fprintf(stderr, "codec-fit: bad --cl / --hid / --s0\n"); return 2; }
        const int K = cl - 1; const size_t np = lc.np;
        float *f = malloc((size_t)batch * LCD_NF * 4), *x = malloc((size_t)batch * 64 * 4), *u = malloc((size_t)batch * K * 4), *gt = calloc((size_t)nth * np, 4);
        double *m1 = calloc(np, 8), *m2 = calloc(np, 8); uint64_t rs = seed * 0x9E3779B97F4A7C15ull + 7;
        pthread_t th[256]; gjob jb[256]; t0 = now(); double ema = 0, eR = 0, eD = 0;
        for (int it = 0; it < steps; it++) {
            for (int i = 0; i < batch; i++) {
                rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; const size_t ci = (size_t)(rs % ncand); const int b = (int)cand[ci][0], c = (int)cand[ci][1];
                lcodec_feat(&lc, xg[b], NCB, NCB, NCB, c / (NCB * NCB), c / NCB % NCB, c % NCB, f + (size_t)i * LCD_NF);
                memcpy(x + (size_t)i * 64, xg[b] + (size_t)c * 64, 256);
                for (int k = 0; k < K; k++) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; u[(size_t)i * K + k] = (float)((double)(rs >> 11) * 0x1p-53 - 0.5); }
            }
            memset(gt, 0, (size_t)nth * np * 4);
            for (int t = 0; t < nth; t++) { const int a = (int)((int64_t)batch * t / nth), e = (int)((int64_t)batch * (t + 1) / nth);
                jb[t] = (gjob){&lc, f + (size_t)a * LCD_NF, x + (size_t)a * 64, u + (size_t)a * K, e - a, lambda, gt + (size_t)t * np, 0, 0, 0};
                pthread_create(&th[t], nullptr, grad_run, &jb[t]); }
            double loss = 0, R = 0, D = 0;
            for (int t = 0; t < nth; t++) { pthread_join(th[t], nullptr); const double w = (double)jb[t].n / batch; loss += w * jb[t].loss; R += w * jb[t].R; D += w * jb[t].D; }
            for (int t = 1; t < nth; t++) for (size_t i = 0; i < np; i++) gt[i] += gt[(size_t)t * np + i];   /* (fixed order) */
            const double lr = lr0 * 0.5 * (1 + cos(M_PI * it / steps)), b1 = 0.9, b2 = 0.999, c1 = 1 - pow(b1, it + 1), c2 = 1 - pow(b2, it + 1);
            for (size_t i = 0; i < np; i++) { const double gi = gt[i] / nth;   /* (each thread's gradient is its slice's mean) */ m1[i] = b1 * m1[i] + (1 - b1) * gi; m2[i] = b2 * m2[i] + (1 - b2) * gi * gi; lc.p[i] -= (float)(lr * (m1[i] / c1) / (sqrt(m2[i] / c2) + 1e-8)); }
            ema = it ? 0.98 * ema + 0.02 * loss : loss; eR = it ? 0.98 * eR + 0.02 * R : R; eD = it ? 0.98 * eD + 0.02 * D : D;
            if (it % 250 == 0 || it == steps - 1) fprintf(stderr, "codec-fit: step %5d loss %.4f R %.4f bit/vox D %.3f (%.0f s)\n", it, ema, eR, eD, now() - t0);
        }
        /* the mean channel's step: the learned channels' median effective step (|S_c|), a uniform-step allocation */
        { const lcodec_off o = lcodec_offs(cl, hid); float nr[64];
          for (int c = 0; c < K; c++) { double s = 0; for (int j = 0; j < LCD_NAX; j++) s += (double)lc.p[o.S + (size_t)c * LCD_NAX + j] * lc.p[o.S + (size_t)c * LCD_NAX + j]; nr[c] = (float)sqrt(s); }
          for (int a = 1; a < K; a++) for (int b = a; b > 0 && nr[b] < nr[b - 1]; b--) { const float t = nr[b]; nr[b] = nr[b - 1]; nr[b - 1] = t; }
          lc.mstep = nr[K / 2]; }
        free(f); free(x); free(u); free(gt); free(m1); free(m2); free(cand);
        if (out && lcodec_write(&lc, out)) { fprintf(stderr, "codec-fit: cannot write %s\n", out); return 1; }
    }
    { lv_codec lv; if (lcodec_lv(&lc, &lb, nullptr, &lv)) { fprintf(stderr, "codec-fit: the codec's synthesis does not fit latvol's rows\n"); return 1; }
      float f[LCD_NF];
      for (int i = 0; i < nev; i++) { const int b = ev[i]; qb[i] = malloc((size_t)NCELL * cl * 2);
          for (int c = 0; c < NCELL; c++) { lcodec_feat(&lc, xg[b], NCB, NCB, NCB, c / (NCB * NCB), c / NCB % NCB, c % NCB, f); lcodec_encode_cell(&lc, f, xg[b] + (size_t)c * 64, any[b][c], qb[i] + (size_t)c * cl); } }
      double bpv, ps_;
      if (!score(&lv, &lv.basis[0][0], nev, qb, ctev, &bpv, &ps_))
          printf("learned cl %2d hid %d lambda %.4g: %.4f B/vox, PSNR %.2f dB (mean step %.3f x sd, hash %016llx)\n", cl, lc.hid, lambda, bpv, ps_, lc.mstep, (unsigned long long)lcodec_hash(&lc));
      for (int i = 0; i < nev; i++) free(qb[i]);
      lv_codec_free(&lv); }
    lcodec_free(&lc);
    for (int k = 0; k < ns; k++) { free(ct[k]); free(xg[k]); free(any[k]); }
    z3_close(z); store_close(st);
    return 0;
}
