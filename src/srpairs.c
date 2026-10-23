/* ufsm sr-pairs <pair-dir> <out-dir> [--n 500] [--in 64] [--spacing 128] [--holdout 0.15] [--minfrac 0.9] [--cover-mm 1.2] [--tol-um 2.4]
                 [--threads 12] [--cache DIR] [--seed 1]
   Super-resolution training pairs from one registered pair (reg run_pair output: <pair-dir>/final/reg.json, residuals, eval boxes):
   the coarse (native, 8.6-9.4 um) scan is the input, the fine (2.2-2.4 um) scan the target, x4.
   Boxes: centres on a grid of --spacing native voxels over the overlap, visited in a seeded shuffled order; a box is kept when its centre
   and 8 corners are certified (rcover: good fit box near, no failing evaluation box near), at least --minfrac of its input voxels are
   non-zero (papyrus / inside the scan mask) and at least --minfrac of its target voxels are valid; until --n boxes.
   Per box: input = the native level-0 box (in^3 voxels, uint8), target = the fine scan sampled trilinearly through the map at the input's x4
   grid ((4 in)^3; target voxel (k, j, i) at native coords origin + ((i, j, k) + 0.5) / 4 - 0.5), 0 where any stencil voxel is masked;
   mask = target valid (binary).
   Outputs (volcomp lossless zarr, boxes stacked along z, 512^3 shards): input.zarr [n in, in, in], target.zarr [n 4in, 4in, 4in],
   mask.zarr (binary, target grid); boxes.json (per box: native centre / origin, fine centre, split, valid fractions); intensity_map.json:
   a monotone (pool-adjacent-violators) map target-at-x1/4 -> input intensity over certified valid voxels (256-entry LUT, bin counts,
   correlation). split: boxes in the top --holdout fraction of the overlap's z range are "test", the rest "train". */
#include "reg.h"
#include "json.h"
#include "z3w.h"
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

int reg_load_map(const char *path, const char *cache, rvol **cv, rvol **fv, rmap *m, rfield *D);   /* src/reg_view.c */

static const char *opt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return dflt;
}

typedef struct { double c[3]; int64_t o[3]; double fc[3]; float in_frac, tg_frac; int slot, split; } sbox;
typedef struct {
    rvol *cv, *fv; rmap m; rcover cov; int B, T; double minfrac, zcut;
    double *cand; int ncand; atomic_int next, nacc; int want;
    sbox *box; uint8_t *inputs;             /* inputs: want x B^3 */
    z3w *wt, *wm; int64_t S;                 /* target / mask writers, shard 512 */
    pthread_mutex_t mu; uint8_t **tbuf, **mbuf; int *tcnt;   /* pending target / mask shard buffers (boxes per shard = S / T) */
    uint64_t hist[256][256];                 /* [target bin at x1/4][input] */
    time_t t0;
} sjob;

static void *sr_worker(void *arg) {
    sjob *j = arg; const int B = j->B, T = j->T; const size_t NT = (size_t)T * T * T, NB = (size_t)B * B * B;
    uint8_t *in = malloc(NB), *tg = malloc(NT), *mk = malloc(NT); uint64_t (*h)[256] = calloc(256, sizeof *h);
    for (;;) {
        if (atomic_load(&j->nacc) >= j->want) break;
        const int k = atomic_fetch_add(&j->next, 1); if (k >= j->ncand) break;
        const double *c = j->cand + 3 * k;
        int64_t o[3]; for (int d = 0; d < 3; d++) o[d] = (int64_t)llround(c[d]) - B / 2;   /* x y z */
        int cert = rcover_ok(&j->cov, c);
        for (int q = 0; q < 8 && cert; q++) { const double p[3] = {o[0] + (q & 1) * (B - 1), o[1] + (q >> 1 & 1) * (B - 1), o[2] + (q >> 2) * (B - 1)}; cert = rcover_ok(&j->cov, p); }
        if (!cert) continue;
        const int64_t zo[3] = {o[2], o[1], o[0]}, zn[3] = {B, B, B};
        if (rvol_read(j->cv, 0, zo, zn, in)) continue;
        size_t nz = 0; for (size_t i = 0; i < NB; i++) nz += in[i] != 0;
        if (nz < j->minfrac * NB) continue;
        /* the fine block under the target grid */
        const double org[3] = {o[0] - 0.375, o[1] - 0.375, o[2] - 0.375};   /* native coords of target voxel 0: o + 0.5 / 4 - 0.5 */
        double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
        for (int q = 0; q < 27; q++) { const int s3[3] = {q % 3, q / 3 % 3, q / 9}; double p[3], f[3];
            for (int d = 0; d < 3; d++) p[d] = org[d] + s3[d] * 0.5 * (T - 1) / 4.0;
            rmap_c2f(&j->m, p, f); for (int d = 0; d < 3; d++) { lo[d] = fmin(lo[d], f[d]); hi[d] = fmax(hi[d], f[d]); } }
        int64_t fo[3], fn[3]; for (int d = 0; d < 3; d++) { fo[2 - d] = (int64_t)floor(lo[d]) - 3; fn[2 - d] = (int64_t)ceil(hi[d]) + 3 - fo[2 - d] + 1; }
        if ((double)fn[0] * fn[1] * fn[2] > 64e6) continue;
        uint8_t *fb = malloc((size_t)fn[0] * fn[1] * fn[2]);
        if (!fb || rvol_read(j->fv, 0, fo, fn, fb)) { free(fb); continue; }
        size_t nv = 0;
        for (int z = 0; z < T; z++) for (int y = 0; y < T; y++) for (int x = 0; x < T; x++) {
            const double p[3] = {org[0] + x / 4.0, org[1] + y / 4.0, org[2] + z / 4.0}; double f[3]; rmap_c2f(&j->m, p, f);
            const double fx = f[0] - fo[2], fy = f[1] - fo[1], fz = f[2] - fo[0]; const int64_t ix = (int64_t)floor(fx), iy = (int64_t)floor(fy), iz = (int64_t)floor(fz);
            const size_t ti = ((size_t)z * T + y) * T + x; tg[ti] = 0; mk[ti] = 0;
            if (ix < 0 || iy < 0 || iz < 0 || ix + 1 >= fn[2] || iy + 1 >= fn[1] || iz + 1 >= fn[0]) continue;
            const double ux = fx - ix, uy = fy - iy, uz = fz - iz; double s = 0; int bad = 0;
            for (int q = 0; q < 8 && !bad; q++) { const uint8_t v = fb[((size_t)(iz + (q >> 2)) * fn[1] + iy + (q >> 1 & 1)) * fn[2] + ix + (q & 1)]; if (!v) bad = 1;
                s += ((q >> 2) ? uz : 1 - uz) * ((q >> 1 & 1) ? uy : 1 - uy) * ((q & 1) ? ux : 1 - ux) * v; }
            if (bad) continue;
            tg[ti] = (uint8_t)fmin(255, fmax(1, s + 0.5)); mk[ti] = 255; nv++;
        }
        free(fb);
        if (nv < j->minfrac * NT) continue;
        /* intensity pairs: target 4^3 block means (all valid) vs the input voxel */
        for (int z = 0; z < B; z++) for (int y = 0; y < B; y++) for (int x = 0; x < B; x++) {
            const uint8_t iv = in[((size_t)z * B + y) * B + x]; if (!iv) continue;
            int s = 0, ok = 1;
            for (int q = 0; q < 64 && ok; q++) { const uint8_t v = tg[((size_t)(4 * z + (q >> 4)) * T + 4 * y + (q >> 2 & 3)) * T + 4 * x + (q & 3)]; if (!v) ok = 0; s += v; }
            if (ok) h[(s + 32) / 64][iv]++;
        }
        const int slot = atomic_fetch_add(&j->nacc, 1);
        if (slot >= j->want) break;
        sbox *b = &j->box[slot]; memcpy(b->c, c, sizeof b->c); memcpy(b->o, o, sizeof b->o);
        { const double cc[3] = {c[0], c[1], c[2]}; rmap_c2f(&j->m, cc, b->fc); }
        b->in_frac = (float)nz / NB; b->tg_frac = (float)nv / NT; b->slot = slot; b->split = c[2] >= j->zcut;
        memcpy(j->inputs + (size_t)slot * NB, in, NB);
        /* the target / mask shard of this slot: boxes per shard = S / T */
        const int per = (int)(j->S / T), sh = slot / per, pos = slot % per;
        pthread_mutex_lock(&j->mu);
        if (!j->tbuf[sh]) { j->tbuf[sh] = calloc((size_t)j->S * j->S * j->S, 1); j->mbuf[sh] = calloc((size_t)j->S * j->S * j->S, 1); }
        uint8_t *tb = j->tbuf[sh], *mb = j->mbuf[sh];
        pthread_mutex_unlock(&j->mu);
        for (int z = 0; z < T; z++) for (int y = 0; y < T; y++) { memcpy(tb + ((size_t)(pos * T + z) * j->S + y) * j->S, tg + ((size_t)z * T + y) * T, T); memcpy(mb + ((size_t)(pos * T + z) * j->S + y) * j->S, mk + ((size_t)z * T + y) * T, T); }
        pthread_mutex_lock(&j->mu);
        const int full = ++j->tcnt[sh] == per || (sh == (j->want - 1) / per && j->tcnt[sh] == j->want - sh * per);
        pthread_mutex_unlock(&j->mu);
        if (full) {
            if (z3w_write_shard(j->wt, sh, 0, 0, tb, 1) || z3w_write_shard(j->wm, sh, 0, 0, mb, 1)) fprintf(stderr, "sr-pairs: shard write failed: %s\n", z3w_error());
            pthread_mutex_lock(&j->mu); free(j->tbuf[sh]); free(j->mbuf[sh]); j->tbuf[sh] = j->mbuf[sh] = nullptr; pthread_mutex_unlock(&j->mu);
        }
        if ((slot + 1) % 25 == 0) fprintf(stderr, "  sr-pairs: %d / %d boxes (%d candidates tried, %lds)\n", slot + 1, j->want, atomic_load(&j->next), (long)(time(nullptr) - j->t0));
    }
    pthread_mutex_lock(&j->mu); for (int a = 0; a < 256; a++) for (int b = 0; b < 256; b++) j->hist[a][b] += h[a][b]; pthread_mutex_unlock(&j->mu);
    free(in); free(tg); free(mk); free(h);
    return nullptr;
}

int cmd_sr_pairs(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm sr-pairs <pair-dir> <out-dir> [--n 500] [--in 64] [--spacing 128] [--holdout 0.15] [--minfrac 0.9] [--cover-mm 1.2] [--tol-um 2.4] [--threads 12] [--cache DIR] [--seed 1]\n"); return 2; }
    const char *pd = argv[2], *od = argv[3]; char mp[2100]; snprintf(mp, sizeof mp, "%s/final/reg.json", pd);
    sjob *j = calloc(1, sizeof *j); rfield D;
    if (reg_load_map(mp, opt(argc, argv, "--cache", nullptr), &j->cv, &j->fv, &j->m, &D)) return 1;
    if (D.d) j->m.D = &D;
    if (rcover_load(&j->cov, pd, atof(opt(argc, argv, "--cover-mm", "1.2")) * 1000 / j->cv->um, atof(opt(argc, argv, "--tol-um", "2.4")) / j->cv->um)) return 1;
    j->B = atoi(opt(argc, argv, "--in", "64")); j->T = 4 * j->B; j->want = atoi(opt(argc, argv, "--n", "500")); j->minfrac = atof(opt(argc, argv, "--minfrac", "0.9"));
    const double S = atof(opt(argc, argv, "--spacing", "128")), hold = atof(opt(argc, argv, "--holdout", "0.15"));
    const int nt = atoi(opt(argc, argv, "--threads", "12")); const unsigned seed = (unsigned)atoi(opt(argc, argv, "--seed", "1"));
    /* overlap (native L0) and candidate centres */
    const int64_t *fs = j->fv->shape[0], *cs = j->cv->shape[0];
    double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
    for (int k = 0; k < 8; k++) { const double p[3] = {k & 1 ? fs[2] - 1 : 0, k & 2 ? fs[1] - 1 : 0, k & 4 ? fs[0] - 1 : 0}; double q[3]; rmap_f2c(&j->m, p, q); for (int d = 0; d < 3; d++) { lo[d] = fmin(lo[d], q[d]); hi[d] = fmax(hi[d], q[d]); } }
    for (int d = 0; d < 3; d++) { lo[d] = fmax(lo[d], j->B); hi[d] = fmin(hi[d], cs[2 - d] - 1 - j->B); }
    j->zcut = hi[2] - hold * (hi[2] - lo[2]);
    int cap = 0; for (double z = lo[2] + S / 2; z <= hi[2]; z += S) for (double y = lo[1] + S / 2; y <= hi[1]; y += S) for (double x = lo[0] + S / 2; x <= hi[0]; x += S) {
        if (j->ncand == cap) j->cand = realloc(j->cand, (cap = cap ? 2 * cap : 4096) * 3 * sizeof *j->cand);
        j->cand[3 * j->ncand] = x; j->cand[3 * j->ncand + 1] = y; j->cand[3 * j->ncand + 2] = z; j->ncand++;
    }
    srand(seed);   /* Fisher-Yates over the candidates */
    for (int i = j->ncand - 1; i > 0; i--) { const int k = (int)(((uint64_t)rand() * RAND_MAX + rand()) % (uint64_t)(i + 1)); for (int d = 0; d < 3; d++) { const double t = j->cand[3 * i + d]; j->cand[3 * i + d] = j->cand[3 * k + d]; j->cand[3 * k + d] = t; } }
    fprintf(stderr, "sr-pairs: %s: input %s (%.3f um) x4 <- target %s (%.3f um); %d candidates, overlap z %.0f .. %.0f (test from %.0f)\n", pd, j->cv->name, j->cv->um, j->fv->name, j->fv->um, j->ncand, lo[2], hi[2], j->zcut);
    mkdir(od, 0755);
    char p[2200]; j->S = 512; const int per = (int)(j->S / j->T), nsh = (j->want + per - 1) / per;
    const int64_t tshape[3] = {(int64_t)j->want * j->T, j->T, j->T};
    snprintf(p, sizeof p, "%s/target.zarr", od); j->wt = z3w_create(p, tshape, (int)j->S, 0, 0, "{\"ufsm\":{\"content\":\"sr-pairs target: the fine scan through the map at the input's x4 grid, boxes stacked along z; 0 = invalid\"}}");
    snprintf(p, sizeof p, "%s/mask.zarr", od); j->wm = z3w_create_mask(p, tshape, (int)j->S, "{\"ufsm\":{\"content\":\"sr-pairs target validity (255 valid)\"}}");
    if (!j->wt || !j->wm) { fprintf(stderr, "sr-pairs: %s\n", z3w_error()); return 1; }
    j->tbuf = calloc(nsh, sizeof *j->tbuf); j->mbuf = calloc(nsh, sizeof *j->mbuf); j->tcnt = calloc(nsh, sizeof *j->tcnt);
    j->box = calloc(j->want, sizeof *j->box); j->inputs = calloc((size_t)j->want * j->B * j->B * j->B, 1);
    pthread_mutex_init(&j->mu, nullptr); j->t0 = time(nullptr);
    pthread_t th[64]; const int T = nt < 1 ? 1 : nt > 64 ? 64 : nt;
    for (int t = 0; t < T; t++) pthread_create(&th[t], nullptr, sr_worker, j);
    for (int t = 0; t < T; t++) pthread_join(th[t], nullptr);
    int n = atomic_load(&j->nacc); if (n > j->want) n = j->want;
    /* partially filled shards (fewer boxes than --n) */
    for (int sh = 0; sh < nsh; sh++) if (j->tbuf[sh]) { z3w_write_shard(j->wt, sh, 0, 0, j->tbuf[sh], 1); z3w_write_shard(j->wm, sh, 0, 0, j->mbuf[sh], 1); free(j->tbuf[sh]); free(j->mbuf[sh]); }
    z3w_close(j->wt); z3w_close(j->wm);
    { const int64_t ishape[3] = {(int64_t)j->want * j->B, j->B, j->B}; snprintf(p, sizeof p, "%s/input.zarr", od);
      z3w *wi = z3w_create(p, ishape, (int)j->S, 0, 0, "{\"ufsm\":{\"content\":\"sr-pairs input: native level-0 boxes stacked along z\"}}");
      const int pi = (int)(j->S / j->B); uint8_t *buf = calloc((size_t)j->S * j->S * j->S, 1);
      for (int sh = 0; sh * pi < j->want; sh++) {
          memset(buf, 0, (size_t)j->S * j->S * j->S);
          for (int q = 0; q < pi && sh * pi + q < j->want; q++) for (int z = 0; z < j->B; z++) for (int y = 0; y < j->B; y++)
              memcpy(buf + ((size_t)(q * j->B + z) * j->S + y) * j->S, j->inputs + (((size_t)(sh * pi + q) * j->B + z) * j->B + y) * j->B, j->B);
          z3w_write_shard(wi, sh, 0, 0, buf, nt);
      }
      free(buf); z3w_close(wi); }
    /* boxes.json */
    snprintf(p, sizeof p, "%s/boxes.json", od); FILE *f = fopen(p, "w"); int ntest = 0;
    fprintf(f, "{\"pair\":\"%s\",\"input\":{\"key\":\"%s\",\"um\":%.6g,\"box\":%d},\"target\":{\"key\":\"%s\",\"um\":%.6g,\"box\":%d,\"scale\":4},\"n\":%d,\"requested\":%d,\"candidates_tried\":%d,\"zcut_test\":%.1f,\n \"boxes\":[",
            pd, j->cv->name, j->cv->um, j->B, j->fv->name, j->fv->um, j->T, n, j->want, atomic_load(&j->next), j->zcut);
    for (int i = 0; i < n; i++) { const sbox *b = &j->box[i]; ntest += b->split;
        fprintf(f, "%s\n  {\"slot\":%d,\"centre_xyz\":[%.1f,%.1f,%.1f],\"origin_xyz\":[%lld,%lld,%lld],\"fine_centre_xyz\":[%.1f,%.1f,%.1f],\"input_frac\":%.3f,\"target_valid\":%.3f,\"split\":\"%s\"}", i ? "," : "", b->slot,
                b->c[0], b->c[1], b->c[2], (long long)b->o[0], (long long)b->o[1], (long long)b->o[2], b->fc[0], b->fc[1], b->fc[2], b->in_frac, b->tg_frac, b->split ? "test" : "train"); }
    fprintf(f, "\n ],\"train\":%d,\"test\":%d}\n", n - ntest, ntest); fclose(f);
    /* intensity map: per target bin the mean input, pool-adjacent-violators (non-decreasing), bins weighted by counts */
    double w[256], y[256], sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0, tot = 0;
    for (int a = 0; a < 256; a++) { double c = 0, s = 0; for (int b = 1; b < 256; b++) { c += j->hist[a][b]; s += (double)b * j->hist[a][b]; sx += (double)a * j->hist[a][b]; sxx += (double)a * a * j->hist[a][b]; sy += (double)b * j->hist[a][b]; syy += (double)b * b * j->hist[a][b]; sxy += (double)a * b * j->hist[a][b]; }
        w[a] = c; y[a] = c > 0 ? s / c : 0; tot += c; }
    const double r = tot > 1 ? (sxy / tot - sx / tot * sy / tot) / sqrt(fmax(1e-12, (sxx / tot - sx * sx / tot / tot) * (syy / tot - sy * sy / tot / tot))) : 0;
    int bl[256], nbk = 0; double bw[256], bv[256]; int bstart[256];
    for (int a = 0; a < 256; a++) { if (w[a] <= 0) continue; bv[nbk] = y[a]; bw[nbk] = w[a]; bstart[nbk] = a; nbk++;
        while (nbk > 1 && bv[nbk - 2] > bv[nbk - 1]) { bv[nbk - 2] = (bv[nbk - 2] * bw[nbk - 2] + bv[nbk - 1] * bw[nbk - 1]) / (bw[nbk - 2] + bw[nbk - 1]); bw[nbk - 2] += bw[nbk - 1]; nbk--; } }
    double lut[256]; int k2 = 0;
    for (int a = 0; a < 256; a++) { while (k2 + 1 < nbk && bstart[k2 + 1] <= a) k2++; lut[a] = nbk ? bv[k2] : a; (void)bl; }
    snprintf(p, sizeof p, "%s/intensity_map.json", od); f = fopen(p, "w");
    fprintf(f, "{\"maps\":\"target (fine scan, 4^3 block mean, uint8) -> input (native scan) intensity\",\"method\":\"bin means + pool adjacent violators (non-decreasing)\",\"voxels\":%.0f,\"pearson_r\":%.4f,\"lut\":[", tot, r);
    for (int a = 0; a < 256; a++) fprintf(f, "%s%.3f", a ? "," : "", lut[a]);
    fprintf(f, "],\"bin_counts\":["); for (int a = 0; a < 256; a++) fprintf(f, "%s%.0f", a ? "," : "", w[a]);
    fprintf(f, "]}\n"); fclose(f);
    fprintf(stderr, "sr-pairs: %d boxes (%d train, %d test) from %d candidates in %lds; intensity r %.3f -> %s\n", n, n - ntest, ntest, atomic_load(&j->next), (long)(time(nullptr) - j->t0), r, od);
    return 0;
}
