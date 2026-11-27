/* ufsm ink-proto (2026-11-27, inkproto): the prototype file, the label generator and predict's --proto / --proto-seed hooks (inkproto.h) */
#include "inkproto.h"
#include "nn.h"
#include "optcheck.h"
#include "pyramid.h"
#include "sources.h"
#include "store.h"
#include "sysinfo.h"
#include "unet.h"
#include "z3w.h"
#include "zarr3.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- the prototype file: "UFPROTO1", int32 C, double n, C raw sums, C unit sums (little endian) ---- */
int ink_proto_read(const char *path, ink_proto_file *P) {
    FILE *f = fopen(path, "rb"); char mg[8]; int32_t C = 0;
    if (!f) { fprintf(stderr, "ink-proto: cannot open %s\n", path); return -1; }
    int ok = fread(mg, 1, 8, f) == 8 && !memcmp(mg, "UFPROTO1", 8) && fread(&C, 4, 1, f) == 1 && C > 0 && C <= INK_PROTO_MAXC && fread(&P->n, 8, 1, f) == 1;
    if (ok) { P->C = C; ok = fread(P->raw, 8, (size_t)C, f) == (size_t)C && fread(P->unit, 8, (size_t)C, f) == (size_t)C; }
    fclose(f);
    if (!ok) { fprintf(stderr, "ink-proto: %s is not a prototype file (predict --proto-seed)\n", path); return -1; }
    return 0;
}
int ink_proto_write(const char *path, const ink_proto_file *P) {
    FILE *f = fopen(path, "wb"); const int32_t C = P->C;
    if (!f) { fprintf(stderr, "ink-proto: cannot write %s\n", path); return -1; }
    int ok = fwrite("UFPROTO1", 1, 8, f) == 8 && fwrite(&C, 4, 1, f) == 1 && fwrite(&P->n, 8, 1, f) == 1
          && fwrite(P->raw, 8, (size_t)C, f) == (size_t)C && fwrite(P->unit, 8, (size_t)C, f) == (size_t)C;
    ok &= fclose(f) == 0;
    if (!ok) fprintf(stderr, "ink-proto: write %s failed\n", path);
    return ok ? 0 : -1;
}
int ink_proto_vector(const ink_proto_file *P, int mode, float *pr) {
    if (!(P->n > 0)) return -1;
    const double *s = mode ? P->raw : P->unit; double q = 0;
    for (int c = 0; c < P->C; c++) q += s[c] * s[c];
    if (!(q > 0)) return -1;
    q = 1 / sqrt(q); for (int c = 0; c < P->C; c++) pr[c] = (float)(s[c] * q);
    return 0;
}

/* ---- one chunk of labels (inkproto.h) ---- */
long ink_proto_label_chunk(const uint8_t *t, const uint8_t *c, const uint8_t *ct, size_t n, int air, double thr, int out_mode, uint8_t *out) {
    int lo = 256, hi = -1; long nink = 0;
    for (size_t i = 0; i < n; i++) if (t[i] && c[i] && ct[i] >= air) { if (c[i] < lo) lo = c[i]; if (c[i] > hi) hi = c[i]; }
    const double rg = hi > lo ? (double)(hi - lo) : 0;
    for (size_t i = 0; i < n; i++) {
        if (!(t[i] && c[i] && ct[i] >= air)) { out[i] = out_mode ? 255 : 0; continue; }
        const double s = rg > 0 ? t[i] / 255.0 * ((c[i] - lo) / rg) : 0;
        const int ink = s > thr; nink += ink;
        out[i] = out_mode ? (uint8_t)(ink ? 254 : 0) : (uint8_t)(1 + lrint(254 * s));
    }
    return nink;
}

/* ---- predict's hooks ---- */
#define PROTO_ZS 16   /* window planes per feature slab: C x 16 x 768^2 fp16 = 0.6 GB at C 32 */
struct ink_proto_pred { int C, seed_on; char seed[1200]; float thr_logit; double *acc; float *pr; void *ft; size_t ftn; long nwin; };
ink_proto_pred *ink_proto_pred_open(int C, const char *seed, double thr, const char *proto, int mode) {
    if (C < 1 || C > INK_PROTO_MAXC) { fprintf(stderr, "predict --proto: %d trunk channels (1..%d)\n", C, INK_PROTO_MAXC); return nullptr; }
    if (!(thr > 0 && thr < 1)) { fprintf(stderr, "predict --proto-thr: 0 < thr < 1\n"); return nullptr; }
    ink_proto_pred *Q = calloc(1, sizeof *Q); if (!Q) return nullptr;
    Q->C = C; Q->thr_logit = (float)log(thr / (1 - thr));
    if (seed) { Q->seed_on = 1; snprintf(Q->seed, sizeof Q->seed, "%s", seed); Q->acc = nn_malloc((size_t)(2 * C + 1) * 8); nn_zero(Q->acc, (size_t)(2 * C + 1) * 8); }
    if (proto) {
        ink_proto_file P; float pr[INK_PROTO_MAXC];
        if (ink_proto_read(proto, &P)) { free(Q); return nullptr; }
        if (P.C != C) { fprintf(stderr, "predict --proto: %s has %d channels, this net's trunk %d\n", proto, P.C, C); free(Q); return nullptr; }
        if (ink_proto_vector(&P, mode, pr)) { fprintf(stderr, "predict --proto: %s holds no seed voxels\n", proto); free(Q); return nullptr; }
        Q->pr = nn_malloc((size_t)C * 4); nn_h2d(Q->pr, pr, (size_t)C * 4);
        fprintf(stderr, "predict --proto: the output is (1 + cos) / 2 to the %s prototype of %s (%.0f seed voxels, %d channels)\n", mode ? "raw-mean" : "unit-mean", proto, P.n, C);
    }
    return Q;
}
int ink_proto_pred_window(ink_proto_pred *Q, unet *u, void *lg, int h16, int D, int W, int hz, int halo) {
    const size_t hw = (size_t)W * W, need = (size_t)Q->C * PROTO_ZS * hw * 2;
    if (Q->ftn < need) { nn_free(Q->ft); Q->ft = nn_malloc_try(need); Q->ftn = Q->ft ? need : 0;
        if (!Q->ft) { fprintf(stderr, "predict --proto: no device memory for the feature slab (%.2f GB)\n", need / 1e9); return -1; } }
    for (int z0 = 0; z0 < D; z0 += PROTO_ZS) {
        const int z1 = z0 + PROTO_ZS < D ? z0 + PROTO_ZS : D; const size_t nv = (size_t)(z1 - z0) * hw;
        if (unet_head_feat_z(u, z0, z1, Q->ft)) { fprintf(stderr, "predict --proto: the trunk's features are not covered (unet_head_feat_z)\n"); return -1; }
        char *ls = (char *)lg + (size_t)z0 * hw * (h16 ? 2 : 4);
        if (Q->seed_on) nn_proto_acc(Q->ft, Q->C, z0, z1 - z0, D, W, W, hz, halo, halo, ls, h16, Q->thr_logit, Q->acc);
        if (Q->pr) nn_proto_cos(Q->ft, Q->C, nv, Q->pr, ls, h16);   /* (after the seed sums: they read the ink logits) */
    }
    Q->nwin++;
    return 0;
}
int ink_proto_pred_close(ink_proto_pred *Q) {
    if (!Q) return 0;
    int rc = 0;
    if (Q->seed_on) {
        double h[2 * INK_PROTO_MAXC + 1]; nn_d2h(h, Q->acc, (size_t)(2 * Q->C + 1) * 8);
        ink_proto_file P = {.C = Q->C, .n = h[2 * Q->C]};
        for (int c = 0; c < Q->C; c++) { P.raw[c] = h[c]; P.unit[c] = h[Q->C + c]; }
        double rn = 0, un = 0; for (int c = 0; c < Q->C; c++) { rn += P.raw[c] * P.raw[c]; un += P.unit[c] * P.unit[c]; }
        fprintf(stderr, "predict --proto-seed: %.0f voxels at p >= %.3f in %ld windows -> %s (|unit mean| %.3f: 1 = every seed feature in one direction)\n",
                P.n, 1 / (1 + exp(-Q->thr_logit)), Q->nwin, Q->seed, P.n > 0 ? sqrt(un) / P.n : 0.0);
        if (!(P.n > 0)) { fprintf(stderr, "predict --proto-seed: no voxel passed the threshold: no prototype written\n"); rc = 1; }
        else rc = ink_proto_write(Q->seed, &P) ? 1 : 0;
        (void)rn; nn_free(Q->acc);
    }
    nn_free(Q->pr); nn_free(Q->ft); free(Q);
    return rc;
}

/* ---- ufsm ink-proto ---- */
static int oidx(int argc, char **argv, const char *name, int nv) { for (int i = 2; i + nv < argc; i++) if (!strcmp(argv[i], opt_n(name))) return i; return -1; }
static const char *oget(int argc, char **argv, const char *name, const char *d) { const int i = oidx(argc, argv, name, 1); return i < 0 ? d : argv[i + 1]; }
static int cmd_label(int argc, char **argv) {
    const int ti = oidx(argc, argv, "--teacher", 2), ci = oidx(argc, argv, "--cos", 2), ki = oidx(argc, argv, "--ct", 2);
    const char *out = argc > 3 && strncmp(argv[3], "--", 2) ? argv[3] : nullptr;
    const int lv = opt_atoi("--level", oget(argc, argv, "--level", "-1")), air = opt_atoi("--air", oget(argc, argv, "--air", "50")), CH = opt_atoi("--chunk", oget(argc, argv, "--chunk", "128"));
    const double thr = opt_atof("--thr", oget(argc, argv, "--thr", "0.5")), um = opt_atof("--ct-um", oget(argc, argv, "--ct-um", "0"));
    const char *mode = oget(argc, argv, "--out", "score"); const int om = !strcmp(mode, "prob254") ? 1 : !strcmp(mode, "score") ? 0 : -1;
    const int nt = ufsm_threads(opt_atoi("--threads", oget(argc, argv, "--threads", "0")), 8);
    if (!out || ti < 0 || ci < 0 || ki < 0 || lv < 0 || !(um > 0) || om < 0 || CH < 8 || !(thr >= 0 && thr < 1) || air < 0 || air > 255) {
        fprintf(stderr, "usage: ufsm ink-proto label OUT --teacher ROOT KEY --cos ROOT KEY --ct ROOT KEY --ct-um U --level L [--air 50] [--thr 0.5] [--chunk 128]\n"
                        "         [--out score | prob254] [--threads 8]\n"
                        "  teacher / cos: predict outputs on the CT level's grid (--full-grid 1; cos from predict --proto); per CH^3 chunk the cosine is min-max\n"
                        "  normalised over the voxels with both outputs and CT >= air, times the teacher's probability: score = 1 + 254 s (0: masked), or\n"
                        "  prob254 = 254 where s > thr, 0 where not, 255 masked (a stored ink target). OUT/<teacher KEY> on the teacher's shards.\n");
        return 2;
    }
    store *st = store_open(argv[ti + 1]), *sc = store_open(argv[ci + 1]), *sk = store_open(argv[ki + 1]);
    z3 *zt = st ? z3_open(st, argv[ti + 2], nullptr) : nullptr, *zc = sc ? z3_open(sc, argv[ci + 2], nullptr) : nullptr;
    z3 *zk = sk ? pyramid_open_level(sk, argv[ki + 2], lv, um, nullptr) : nullptr;
    if (!zt || !zc || !zk) { fprintf(stderr, "ink-proto label: cannot open the %s: %s\n", !zt ? "teacher" : !zc ? "cosine store" : "CT level", z3_error()); return 1; }
    const z3_meta *mt = z3_meta_of(zt), *mc = z3_meta_of(zc), *mk = z3_meta_of(zk);
    for (int d = 0; d < 3; d++) if (mt->shape[d] != mc->shape[d] || mt->shard[d] != mc->shard[d] || mt->shape[d] != mk->shape[d]) {
        fprintf(stderr, "ink-proto label: teacher %lld,%lld,%lld (shard %lld), cosine %lld,%lld,%lld (shard %lld), CT level %d %lld,%lld,%lld: one grid needed (predict --full-grid 1, one --shard)\n",
                (long long)mt->shape[0], (long long)mt->shape[1], (long long)mt->shape[2], (long long)mt->shard[0], (long long)mc->shape[0], (long long)mc->shape[1], (long long)mc->shape[2],
                (long long)mc->shard[0], lv, (long long)mk->shape[0], (long long)mk->shape[1], (long long)mk->shape[2]); return 2; }
    const int S = (int)mt->shard[0];
    if (mt->shard[1] != S || mt->shard[2] != S || S % CH) { fprintf(stderr, "ink-proto label: the teacher's shard %d must be cubic and a multiple of --chunk %d\n", S, CH); return 2; }
    int64_t ns[3]; uint8_t *pm = nullptr;
    if (z3_shard_map(zt, ns, &pm) < 0) { fprintf(stderr, "ink-proto label: shard map: %s\n", z3_error()); return 1; }
    char od[1400]; snprintf(od, sizeof od, "%s/%s", out, argv[ti + 2]);
    char attrs[600]; snprintf(attrs, sizeof attrs, "{\"ufsm\":{\"content\":\"ink-proto %s (teacher x per-chunk min-max prototype cosine, air < %d masked, thr %.3g, chunk %d)\"%s}}",
                              om ? "label" : "score", air, thr, CH, om ? ",\"encoding\":\"prob254: p*254, 254 = ink, 0 = no ink, 255 = unknown\"" : "");
    z3w *w = z3w_create(od, mt->shape, S, 0, om ? 255 : 0, attrs);
    if (!w) { fprintf(stderr, "%s\n", z3w_error()); return 1; }
    const size_t sv = (size_t)S * S * S, cv = (size_t)CH * CH * CH;
    uint8_t *bt = malloc(sv), *bc = malloc(sv), *bk = malloc(sv), *bo = malloc(sv), *ct = malloc(cv * 4);
    if (!bt || !bc || !bk || !bo || !ct) { fprintf(stderr, "ink-proto label: out of memory\n"); return 1; }
    long nsh = 0; double nink = 0, nlab = 0, nmask = 0, nflat = 0, nch = 0;
    for (int64_t a = 0; a < ns[0]; a++) for (int64_t b = 0; b < ns[1]; b++) for (int64_t c = 0; c < ns[2]; c++) {
        if (!pm[(size_t)((a * ns[1] + b) * ns[2] + c)]) continue;
        const int64_t o[3] = {a * S, b * S, c * S}, n3[3] = {S, S, S};
        if (z3_read(zt, o, n3, bt, nt) || z3_read(zc, o, n3, bc, nt) || z3_read(zk, o, n3, bk, nt)) { fprintf(stderr, "ink-proto label: read: %s\n", z3_error()); return 1; }
        const int nb = S / CH;
        for (int q = 0; q < nb * nb * nb; q++) {   /* each chunk gathered, labelled, scattered back */
            const int qz = q / (nb * nb) * CH, qy = q / nb % nb * CH, qx = q % nb * CH; uint8_t *gt = ct, *gc = ct + cv, *gk = ct + 2 * cv, *go = ct + 3 * cv;
            for (int z = 0; z < CH; z++) for (int y = 0; y < CH; y++) { const size_t s0 = ((size_t)(qz + z) * S + qy + y) * S + qx, d0 = ((size_t)z * CH + y) * CH;
                memcpy(gt + d0, bt + s0, CH); memcpy(gc + d0, bc + s0, CH); memcpy(gk + d0, bk + s0, CH); }
            const long k = ink_proto_label_chunk(gt, gc, gk, cv, air, thr, om, go); long lab = 0, msk = 0;
            for (size_t i = 0; i < cv; i++) { const int m = !(gt[i] && gc[i] && gk[i] >= air); msk += m; lab += !m; }
            if (lab) { nch++; int lo = 256, hi = -1; for (size_t i = 0; i < cv; i++) if (gt[i] && gc[i] && gk[i] >= air) { if (gc[i] < lo) lo = gc[i]; if (gc[i] > hi) hi = gc[i]; } nflat += hi <= lo; }
            nink += k; nlab += lab; nmask += msk;
            for (int z = 0; z < CH; z++) for (int y = 0; y < CH; y++) memcpy(bo + ((size_t)(qz + z) * S + qy + y) * S + qx, go + ((size_t)z * CH + y) * CH, CH);
        }
        if (z3w_write_shard(w, a, b, c, bo, nt)) { fprintf(stderr, "%s\n", z3w_error()); return 1; }
        nsh++;
    }
    z3w_close(w);
    const double lum = atof(argv[ti + 2]);
    if (lum > 0) pyramid_write_group(out, lum, 1, "ink-proto", attrs);
    else fprintf(stderr, "DEFAULTED: ink-proto label: the teacher KEY %s names no level um: no group zarr.json written (the array %s is complete)\n", argv[ti + 2], od);
    fprintf(stderr, "ink-proto label: %ld shards, %.0f labelled voxels (%.0f masked), %.0f ink (%.4f of labelled); %.0f chunks, %.0f without cosine contrast (label 0) -> %s\n",
            nsh, nlab, nmask, nink, nlab > 0 ? nink / nlab : 0.0, nch, nflat, od);
    if (nflat > 0) fprintf(stderr, "DEFAULTED: ink-proto label: %.0f chunks had one cosine value over their voxels: scored 0 (no ink) there\n", nflat);
    free(bt); free(bc); free(bk); free(bo); free(ct); free(pm);
    z3_close(zt); z3_close(zc); z3_close(zk); store_close(st); store_close(sc); store_close(sk);
    return 0;
}
static int cmd_merge(int argc, char **argv) {   /* the sums of several seed files (scrolls, runs) into one */
    if (argc < 5) { fprintf(stderr, "usage: ufsm ink-proto merge OUT.proto IN.proto ...\n"); return 2; }
    ink_proto_file A = {0};
    for (int i = 4; i < argc; i++) { ink_proto_file P; if (ink_proto_read(argv[i], &P)) return 1;
        if (i > 4 && P.C != A.C) { fprintf(stderr, "ink-proto merge: %s has %d channels, the first %d\n", argv[i], P.C, A.C); return 2; }
        A.C = P.C; A.n += P.n; for (int c = 0; c < P.C; c++) { A.raw[c] += P.raw[c]; A.unit[c] += P.unit[c]; } }
    return ink_proto_write(argv[3], &A) ? 1 : 0;
}
static int cmd_show(int argc, char **argv) {   /* the counts and the cosine between the prototypes (files x files) */
    if (argc < 4) { fprintf(stderr, "usage: ufsm ink-proto show A.proto [B.proto ...]\n"); return 2; }
    const int n = argc - 3 > 16 ? 16 : argc - 3; float pr[16][INK_PROTO_MAXC]; int C[16];
    for (int i = 0; i < n; i++) { ink_proto_file P; if (ink_proto_read(argv[3 + i], &P) || ink_proto_vector(&P, 0, pr[i])) return 1; C[i] = P.C;
        float rv[INK_PROTO_MAXC]; ink_proto_vector(&P, 1, rv); double d = 0; for (int c = 0; c < P.C; c++) d += (double)rv[c] * pr[i][c];
        double un = 0; for (int c = 0; c < P.C; c++) un += P.unit[c] * P.unit[c];
        printf("%s\tC %d\tn %.0f\t|unit mean| %.4f\tcos(unit-mean, raw-mean) %.4f\n", argv[3 + i], P.C, P.n, sqrt(un) / P.n, d); }
    for (int i = 0; i < n; i++) { printf("cos %d:", i); for (int j = 0; j < n; j++) { double d = 0; if (C[i] == C[j]) for (int c = 0; c < C[i]; c++) d += (double)pr[i][c] * pr[j][c]; printf(" %.4f", d); } printf("\n"); }
    return 0;
}
int cmd_ink_proto(int argc, char **argv) {
    const char *sub = argc > 2 ? argv[2] : "";
    if (!strcmp(sub, "label")) return cmd_label(argc, argv);
    if (!strcmp(sub, "merge")) return cmd_merge(argc, argv);
    if (!strcmp(sub, "show")) return cmd_show(argc, argv);
    fprintf(stderr, "usage: ufsm ink-proto label | merge | show ...   (seeds and cosine maps: predict --proto-seed OUT.proto / --proto P.proto; src/inkproto.h)\n");
    return 2;
}
