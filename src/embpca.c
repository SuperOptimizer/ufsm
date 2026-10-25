/* ufsm emb-pca: a PCA basis of dinovol-embed vectors (tokens_unit 480-d) fitted over several embedding groups (equal weight per group),
   and its application: each vector centred, projected on the top k components, renormalised to a unit vector, stored int8 x 127 in a
   zarr v3 array of 512^3-voxel shards (64 cells of 8) -> ~0.13 B / voxel at k 64 instead of 0.94 for 480-d tokens.
     ufsm emb-pca fit BASIS.json GROUP... [--name tokens_unit] [--k 64] [--per 50000] [--threads 8]
     ufsm emb-pca apply BASIS.json GROUP OUTGROUP [--name tokens_unit] [--out-name pca64] [--threads 8]
   GROUP: a dinovol-embed output group (its zarr.json attributes give box_zyx); only the box's cells are read. */
#include "embz.h"
#include "fit_emb.h"
#include "json.h"
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static const char *popt(int argc, char **argv, const char *k, const char *d) { for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return argv[i + 1]; return d; }
static char *slurp(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb"); if (!f) return nullptr;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char *p = malloc(sz > 0 ? (size_t)sz + 1 : 1); if (!p || fread(p, 1, (size_t)sz, f) != (size_t)sz) { free(p); fclose(f); return nullptr; }
    fclose(f); p[sz] = 0; *n = (size_t)sz; return p;
}
static json *jload(const char *path) { size_t n = 0; char *t = slurp(path, &n); json *j = t ? json_parse(t, n) : nullptr; free(t); return j; }
/* the group's box (embedding-volume voxels) -> the loaded cells */
static emb_vol *open_box(const char *group, const char *name) {
    char p[1200]; snprintf(p, sizeof p, "%s/zarr.json", group); json *j = jload(p);
    const json *b = j ? json_path(j, "attributes.box_zyx") : nullptr;
    if (!b || !json_at(b, 5)) { fprintf(stderr, "emb-pca: %s: no attributes.box_zyx\n", p); json_free(j); return nullptr; }
    long long o[3], n[3]; for (int d = 0; d < 3; d++) { o[d] = (long long)json_num(json_at(b, d), 0); n[d] = (long long)json_num(json_at(b, 3 + d), 0); }
    json_free(j);
    return emb_vol_open(group, name, o, n, 1.0, 0);
}

/* Jacobi eigen-decomposition of a symmetric C x C matrix A (destroyed): eigenvalues w, eigenvectors as rows of V */
static void jacobi(double *A, int C, double *w, double *V) {
    for (int i = 0; i < C * C; i++) V[i] = 0; for (int i = 0; i < C; i++) V[(size_t)i * C + i] = 1;
    for (int sweep = 0; sweep < 60; sweep++) {
        double off = 0, dia = 0;
        for (int i = 0; i < C; i++) { dia += A[(size_t)i * C + i] * A[(size_t)i * C + i]; for (int j = i + 1; j < C; j++) off += A[(size_t)i * C + j] * A[(size_t)i * C + j]; }
        if (off <= 1e-24 * dia) break;
        for (int p = 0; p < C; p++) for (int q = p + 1; q < C; q++) {
            const double apq = A[(size_t)p * C + q]; if (fabs(apq) < 1e-300) continue;
            const double th = (A[(size_t)q * C + q] - A[(size_t)p * C + p]) / (2 * apq);
            const double t = (th >= 0 ? 1 : -1) / (fabs(th) + sqrt(th * th + 1)), c = 1 / sqrt(t * t + 1), s = t * c;
            for (int k = 0; k < C; k++) {   /* columns p, q */
                const double akp = A[(size_t)k * C + p], akq = A[(size_t)k * C + q];
                A[(size_t)k * C + p] = c * akp - s * akq; A[(size_t)k * C + q] = s * akp + c * akq;
            }
            for (int k = 0; k < C; k++) {   /* rows p, q */
                const double apk = A[(size_t)p * C + k], aqk = A[(size_t)q * C + k];
                A[(size_t)p * C + k] = c * apk - s * aqk; A[(size_t)q * C + k] = s * apk + c * aqk;
            }
            for (int k = 0; k < C; k++) {   /* V rows = eigenvectors */
                const double vp = V[(size_t)p * C + k], vq = V[(size_t)q * C + k];
                V[(size_t)p * C + k] = c * vp - s * vq; V[(size_t)q * C + k] = s * vp + c * vq;
            }
        }
    }
    for (int i = 0; i < C; i++) w[i] = A[(size_t)i * C + i];
}

typedef struct { const int8_t *v; const size_t *ids; size_t n; int C; double *S, *Q; } acc_job;
static void *acc_main(void *a) {
    acc_job *J = a; const int C = J->C; double *x = malloc(C * sizeof *x);
    for (size_t k = 0; k < J->n; k++) {
        const int8_t *s = J->v + J->ids[k] * C;
        for (int i = 0; i < C; i++) { x[i] = s[i] / 127.0; J->S[i] += x[i]; }
        for (int i = 0; i < C; i++) { const double xi = x[i]; double *q = J->Q + (size_t)i * C; for (int j = i; j < C; j++) q[j] += xi * x[j]; }
    }
    free(x); return nullptr;
}

static int pca_fit(int argc, char **argv) {
    const char *outp = argv[3], *name = popt(argc, argv, "--name", "tokens_unit");
    const int k = atoi(popt(argc, argv, "--k", "64")), nt0 = atoi(popt(argc, argv, "--threads", "8")); const size_t per = (size_t)atoll(popt(argc, argv, "--per", "50000"));
    const int nt = nt0 < 1 ? 1 : nt0 > 64 ? 64 : nt0;
    int C = 0, ng = 0; double *S = nullptr, *Q = nullptr; double wsum = 0;
    for (int a = 4; a < argc; a++) {
        if (argv[a][0] == '-' && argv[a][1] == '-') { a++; continue; }
        emb_vol *e = open_box(argv[a], name); if (!e) return 1;
        int64_t n[3], o[3]; int step; const int8_t *v = emb_vol_cells(e, n, o, &step); const int c = emb_vol_dim(e);
        if (!C) { C = c; S = calloc(C, sizeof *S); Q = calloc((size_t)C * C, sizeof *Q); }
        if (c != C) { fprintf(stderr, "emb-pca: %s has %d channels, not %d\n", argv[a], c, C); return 1; }
        const size_t nc = (size_t)n[0] * n[1] * n[2]; size_t m = 0;
        size_t *ids = malloc(nc * sizeof *ids);
        for (size_t i = 0; i < nc; i++) { const int8_t *s = v + i * C; int any = 0; for (int q = 0; q < C && !any; q++) any = s[q]; if (any) ids[m++] = i; }
        if (!m) { fprintf(stderr, "emb-pca: %s: no vectors, skipped\n", argv[a]); free(ids); emb_vol_free(e); continue; }
        const size_t stride = m > per ? (m + per - 1) / per : 1; size_t ms = 0;
        for (size_t i = 0; i < m; i += stride) ids[ms++] = ids[i];
        double *Sg = calloc((size_t)nt * C, sizeof *Sg), *Qg = calloc((size_t)nt * C * C, sizeof *Qg);
        acc_job J[64]; pthread_t th[64];
        for (int t = 0; t < nt; t++) { const size_t a0 = ms * t / nt, a1 = ms * (t + 1) / nt; J[t] = (acc_job){v, ids + a0, a1 - a0, C, Sg + (size_t)t * C, Qg + (size_t)t * C * C}; pthread_create(&th[t], nullptr, acc_main, &J[t]); }
        for (int t = 0; t < nt; t++) pthread_join(th[t], nullptr);
        const double w = 1.0 / ms;   /* equal weight per group */
        for (int t = 0; t < nt; t++) { for (int i = 0; i < C; i++) S[i] += w * Sg[(size_t)t * C + i]; for (size_t i = 0; i < (size_t)C * C; i++) Q[i] += w * Qg[(size_t)t * C * C + i]; }
        wsum += 1; ng++;
        fprintf(stderr, "emb-pca: %s: %zu vectors, %zu sampled\n", argv[a], m, ms);
        free(Sg); free(Qg); free(ids); emb_vol_free(e);
    }
    if (!ng || k < 1 || k > C) { fprintf(stderr, "emb-pca fit: no groups read (or k %d outside 1..%d)\n", k, C); return 1; }
    double *mean = malloc(C * sizeof *mean); for (int i = 0; i < C; i++) mean[i] = S[i] / wsum;
    double tot = 0;
    for (int i = 0; i < C; i++) for (int j = i; j < C; j++) { const double c = Q[(size_t)i * C + j] / wsum - mean[i] * mean[j]; Q[(size_t)i * C + j] = Q[(size_t)j * C + i] = c; if (i == j) tot += c; }
    double *w = malloc(C * sizeof *w), *V = malloc((size_t)C * C * sizeof *V); int *ord = malloc(C * sizeof *ord);
    jacobi(Q, C, w, V);
    for (int i = 0; i < C; i++) ord[i] = i;
    for (int i = 0; i < C; i++) for (int j = i + 1; j < C; j++) if (w[ord[j]] > w[ord[i]]) { const int t = ord[i]; ord[i] = ord[j]; ord[j] = t; }
    double ek = 0; for (int i = 0; i < k; i++) ek += w[ord[i]];
    FILE *f = fopen(outp, "w"); if (!f) { fprintf(stderr, "emb-pca: cannot write %s\n", outp); return 1; }
    fprintf(f, "{\"format\":\"ufsm emb-pca/1\",\"name\":\"%s\",\"C\":%d,\"k\":%d,\"groups\":%d,\"total_var\":%.9g,\"explained\":%.9g,\"eig\":[", name, C, k, ng, tot, ek / tot);
    for (int i = 0; i < k; i++) fprintf(f, "%s%.9g", i ? "," : "", w[ord[i]]);
    fprintf(f, "],\n\"mean\":["); for (int i = 0; i < C; i++) fprintf(f, "%s%.9g", i ? "," : "", mean[i]);
    fprintf(f, "],\n\"basis\":[");
    for (int r = 0; r < k; r++) { fprintf(f, "%s[", r ? ",\n" : ""); for (int i = 0; i < C; i++) fprintf(f, "%s%.9g", i ? "," : "", V[(size_t)ord[r] * C + i]); fprintf(f, "]"); }
    fprintf(f, "]}\n");
    if (fclose(f)) return 1;
    fprintf(stderr, "emb-pca: %d groups, C %d -> k %d explains %.2f%% of the variance (top eig %.4g, k-th %.4g) -> %s\n", ng, C, k, 100 * ek / tot, w[ord[0]], w[ord[k - 1]], outp);
    free(S); free(Q); free(mean); free(w); free(V); free(ord);
    return 0;
}

typedef struct { const int8_t *v; int8_t *out; const float *B, *mean; int C, k; int64_t n[3], a[3]; int64_t off[3]; int z0, z1; } prj_job;
static void *prj_main(void *p) {
    prj_job *J = p; const int C = J->C, k = J->k; float *x = malloc(C * sizeof *x), *y = malloc(k * sizeof *y);
    for (int z = J->z0; z < J->z1; z++) for (int64_t yy = 0; yy < J->n[1]; yy++) for (int64_t xx = 0; xx < J->n[2]; xx++) {
        const int8_t *s = J->v + (((size_t)z * J->n[1] + yy) * J->n[2] + xx) * C; int any = 0;
        for (int i = 0; i < C && !any; i++) any = s[i];
        if (!any) continue;
        for (int i = 0; i < C; i++) x[i] = s[i] / 127.f - J->mean[i];
        double l = 0;
        for (int r = 0; r < k; r++) { const float *b = J->B + (size_t)r * C; float d = 0; for (int i = 0; i < C; i++) d += b[i] * x[i]; y[r] = d; l += (double)d * d; }
        if (l <= 0) continue;
        const double inv = 127 / sqrt(l);
        int8_t *o = J->out + ((((size_t)(z + J->off[0]) * J->a[1]) + yy + J->off[1]) * J->a[2] + xx + J->off[2]) * k;
        for (int r = 0; r < k; r++) { long t = lround(y[r] * inv); if (!t && y[r] != 0) t = y[r] > 0 ? 1 : -1; o[r] = (int8_t)(t > 127 ? 127 : t < -127 ? -127 : t); }
    }
    free(x); free(y); return nullptr;
}

static int pca_apply(int argc, char **argv) {
    if (argc < 6) return 2;
    const char *bp = argv[3], *group = argv[4], *outg = argv[5], *name = popt(argc, argv, "--name", "tokens_unit"), *oname = popt(argc, argv, "--out-name", "pca64");
    const int nt0 = atoi(popt(argc, argv, "--threads", "8")), nt = nt0 < 1 ? 1 : nt0 > 64 ? 64 : nt0;
    json *bj = jload(bp); if (!bj) { fprintf(stderr, "emb-pca: cannot read %s\n", bp); return 1; }
    const int C = (int)json_num(json_path(bj, "C"), 0), k = (int)json_num(json_path(bj, "k"), 0);
    float *B = malloc((size_t)k * C * sizeof *B), *mean = malloc(C * sizeof *mean);
    for (int i = 0; i < C; i++) mean[i] = (float)json_num(json_at(json_path(bj, "mean"), i), NAN);
    for (int r = 0; r < k; r++) for (int i = 0; i < C; i++) B[(size_t)r * C + i] = (float)json_num(json_at(json_at(json_path(bj, "basis"), r), i), NAN);
    json_free(bj);
    for (size_t i = 0; i < (size_t)k * C; i++) if (isnan(B[i])) { fprintf(stderr, "emb-pca: %s: bad basis\n", bp); return 1; }
    emb_vol *e = open_box(group, name); if (!e) return 1;
    if (emb_vol_dim(e) != C) { fprintf(stderr, "emb-pca: %s has %d channels, the basis %d\n", group, emb_vol_dim(e), C); return 1; }
    int64_t n[3], o[3]; int step; const int8_t *v = emb_vol_cells(e, n, o, &step);
    /* the source array's whole-grid shape (cells) */
    char p[1400]; snprintf(p, sizeof p, "%s/%s/zarr.json", group, name); json *aj = jload(p);
    int64_t shp[3]; for (int d = 0; d < 3; d++) shp[d] = aj ? (int64_t)json_num(json_at(json_path(aj, "shape"), d), 0) : 0;
    json_free(aj);
    if (!shp[0]) { fprintf(stderr, "emb-pca: cannot read %s\n", p); return 1; }
    const int CH = 16, SH = 512 / step;   /* 16-cell chunks, 512-voxel shards */
    int64_t c0[3], a[3], off[3];
    for (int d = 0; d < 3; d++) { const int64_t g0 = o[d] / step; c0[d] = g0 / CH * CH; off[d] = g0 - c0[d]; a[d] = (g0 + n[d] - c0[d] + CH - 1) / CH * CH; }
    int8_t *out = calloc((size_t)a[0] * a[1] * a[2] * k, 1); if (!out) { fprintf(stderr, "emb-pca: out of memory\n"); return 1; }
    prj_job J[64]; pthread_t th[64];
    for (int t = 0; t < nt; t++) { J[t] = (prj_job){v, out, B, mean, C, k, {n[0], n[1], n[2]}, {a[0], a[1], a[2]}, {off[0], off[1], off[2]}, (int)(n[0] * t / nt), (int)(n[0] * (t + 1) / nt)}; pthread_create(&th[t], nullptr, prj_main, &J[t]); }
    for (int t = 0; t < nt; t++) pthread_join(th[t], nullptr);
    /* the output group: the source group's zarr.json + the array */
    if (mkdir(outg, 0755) && errno != EEXIST) { fprintf(stderr, "emb-pca: cannot create %s\n", outg); return 1; }
    { snprintf(p, sizeof p, "%s/zarr.json", group); size_t len = 0; char *t = slurp(p, &len); snprintf(p, sizeof p, "%s/zarr.json", outg); FILE *f = t ? fopen(p, "w") : nullptr;
      if (!f || fwrite(t, 1, len, f) != len || fclose(f)) { fprintf(stderr, "emb-pca: cannot write %s\n", p); return 1; } free(t); }
    char attrs[1600], d[1400];
    snprintf(attrs, sizeof attrs, "{\"step_zyx\":[%d,%d,%d],\"values\":\"unit vector of the top-%d PCA projection of the centred %s (basis %s)\",\"value_scale\":0.007874015748031496,\"pca_basis\":\"%s\"}",
             step, step, step, k, name, bp, bp);
    snprintf(d, sizeof d, "%s/%s", outg, oname);
    zw W; zw_nthreads = nt;
    if (zw_create(&W, d, shp, k, "int8", 1, SH, CH, attrs)) return 1;
    int rc = zw_put(&W, c0[0], c0[1], c0[2], (int)a[0], (int)a[1], (int)a[2], out); rc |= zw_flush(&W); zw_free(&W);
    fprintf(stderr, "emb-pca: %s -> %s (%lldx%lldx%lld cells, k %d, %d-cell shards)%s\n", group, d, (long long)n[0], (long long)n[1], (long long)n[2], k, SH, rc ? " FAILED" : "");
    free(out); free(B); free(mean); emb_vol_free(e);
    return rc ? 1 : 0;
}

int cmd_emb_pca(int argc, char **argv) {
    if (argc >= 5 && !strcmp(argv[2], "fit")) return pca_fit(argc, argv);
    if (argc >= 6 && !strcmp(argv[2], "apply")) return pca_apply(argc, argv);
    fprintf(stderr, "usage: ufsm emb-pca fit BASIS.json GROUP... [--name tokens_unit] [--k 64] [--per 50000] [--threads 8]\n"
                    "       ufsm emb-pca apply BASIS.json GROUP OUTGROUP [--name tokens_unit] [--out-name pca64] [--threads 8]\n");
    return 2;
}
