/* latcomp v1 stores (latfit.h): ufsm lc-encode writes a CT volume as quantised cell coefficients, ufsm lc-info reads one back.
   A store is a sharded zarr v3 int8 array [Z/4][Y/4][X/4](C + 1) (embz.h writer: zstd inner chunks): channels 0-1 the cell-mean
   coefficient's quantiser index (low byte - 128, high byte: it is >= 0 for uint8 CT), 2.. the C - 1 principal-axis indices (|q| <= 127).
   Coefficients are of the raw CT bytes with one step for all (step_raw = step x sd_ref, sd_ref a typical window sd in grey levels):
   a window's z-score (v - mean) / sd is linear, so a reader gets the trainer's normalised channels as (q step_raw - [mean channel] 8 mean) / sd
   with the window's mean / sd from the coefficients themselves. basis.lcb in the store is the transform. */
#include "embz.h"
#include "latfit.h"
#include "store.h"
#include "sysinfo.h"
#include "zarr3.h"
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static const char *opt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return dflt;
}
typedef struct { const uint8_t *ct; int8_t *out; int n[3], C, z0, z1; float step; const lat_basis *lb; double se, sv; long hist[30][256]; } enc_job;
static void *enc_run(void *arg) {
    enc_job *j = arg; const int C = j->C, cy = j->n[1] / 4, cx = j->n[2] / 4; const size_t Y = j->n[1], X = j->n[2];
    for (int z = j->z0; z < j->z1; z++) for (int y = 0; y < cy; y++) for (int x = 0; x < cx; x++) {
        float v[64]; double s2 = 0; int any = 0;
        for (int k = 0; k < 64; k++) { const uint8_t b = j->ct[((size_t)(4 * z + (k >> 4)) * Y + 4 * y + (k >> 2 & 3)) * X + 4 * x + (k & 3)]; v[k] = b; any |= b; s2 += (double)b * b; }
        int8_t *o = j->out + ((size_t)(z * cy + y) * cx + x) * (C + 1);
        if (!any) { o[0] = -128; memset(o + 1, 0, C); j->hist[0][0]++; for (int c = 1; c <= C; c++) j->hist[c][128]++; continue; }   /* air */
        double kept = 0, qe = 0;
        for (int c = 0; c < C; c++) {
            float s = 0; for (int k = 0; k < 64; k++) s += j->lb->b[c][k] * v[k];
            int q = (int)lrintf(s / j->step);
            if (c == 0) { q = q < 0 ? 0 : q > 32767 ? 32767 : q; o[0] = (int8_t)((q & 255) - 128); o[1] = (int8_t)(q >> 8); j->hist[0][q & 255]++; j->hist[1][(q >> 8) + 128]++; }
            else { q = q < -127 ? -127 : q > 127 ? 127 : q; o[c + 1] = (int8_t)q; j->hist[c + 1][q + 128]++; }
            kept += (double)s * s; const double e = s - q * j->step; qe += e * e;
        }
        j->se += fmax(s2 - kept, 0) + qe; j->sv += 64;
    }
    return nullptr;
}
static long dir_bytes(const char *d) {
    char cmd[1300]; snprintf(cmd, sizeof cmd, "du -sb '%s' 2>/dev/null", d); FILE *p = popen(cmd, "r"); long n = 0; if (p) { if (fscanf(p, "%ld", &n) != 1) n = 0; pclose(p); } return n;
}
int cmd_lc_encode(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: ufsm lc-encode BASIS.lcb ROOT KEY OUT [--box Z,Y,X,NZ,NY,NX] [--c 29] [--step 0.25] [--sd S] [--threads N] [--shard 64] [--chunk 32]\n"
                                    "  KEY: the CT array (uint8); the box in its voxels (multiples of 4 x --shard; default: the whole array rounded down); --sd: sd_ref in grey\n"
                                    "  levels (default: measured on the box); --shard / --chunk in cells\n"); return 2; }
    lat_basis *lb = malloc(sizeof *lb); if (lat_basis_load(argv[2], lb)) { fprintf(stderr, "lc-encode: cannot read the transform %s\n", argv[2]); return 1; }
    store *st = store_open(argv[3]); z3 *z = st ? z3_open(st, argv[4], nullptr) : nullptr; if (!z) { fprintf(stderr, "lc-encode: cannot open %s %s: %s\n", argv[3], argv[4], z3_error()); return 1; }
    const z3_meta *m = z3_meta_of(z);
    const int C = atoi(opt(argc, argv, "--c", "29")), SH = atoi(opt(argc, argv, "--shard", "64")), CH = atoi(opt(argc, argv, "--chunk", "32")), nth = ufsm_threads(atoi(opt(argc, argv, "--threads", "0")), 8);
    const float step = (float)atof(opt(argc, argv, "--step", "0.25"));
    if (C < 1 || C > 29 || SH % CH || !(step > 0)) { fprintf(stderr, "lc-encode: --c 1..29, --shard a multiple of --chunk, --step > 0\n"); return 2; }
    const int V = 4 * SH; int64_t box[6] = {0, 0, 0, m->shape[0] / V * V, m->shape[1] / V * V, m->shape[2] / V * V};
    if (*opt(argc, argv, "--box", "")) { long long b[6]; if (sscanf(opt(argc, argv, "--box", ""), "%lld,%lld,%lld,%lld,%lld,%lld", b, b + 1, b + 2, b + 3, b + 4, b + 5) != 6) { fprintf(stderr, "lc-encode: --box Z,Y,X,NZ,NY,NX\n"); return 2; } for (int i = 0; i < 6; i++) box[i] = b[i]; }
    for (int i = 0; i < 3; i++) if (box[i] % 4 || box[3 + i] % V || box[3 + i] <= 0 || box[i] < 0 || box[i] + box[3 + i] > m->shape[i] + V) { fprintf(stderr, "lc-encode: the box origin must be a multiple of 4 and its size of %d voxels\n", V); return 2; }
    uint8_t *ct = malloc((size_t)V * V * V); int8_t *out = malloc((size_t)SH * SH * SH * (C + 1));
    double sd = atof(opt(argc, argv, "--sd", "0"));
    if (!(sd > 0)) {   /* sd_ref: the sd over a lattice of shards (all voxels, as a window's z-score counts them) */
        double s1 = 0, s2 = 0, n = 0; const int64_t ns[3] = {box[3] / V, box[4] / V, box[5] / V}; const int64_t tot = ns[0] * ns[1] * ns[2], stp = tot / 24 > 0 ? tot / 24 : 1;
        for (int64_t i = stp / 2; i < tot; i += stp) {
            const int64_t o[3] = {box[0] + i / (ns[1] * ns[2]) * V, box[1] + i / ns[2] % ns[1] * V, box[2] + i % ns[2] * V}, nn[3] = {V, V, V};
            if (z3_read(z, o, nn, ct, nth)) { fprintf(stderr, "lc-encode: read failed: %s\n", z3_error()); return 1; }
            for (size_t k = 0; k < (size_t)V * V * V; k += 7) { s1 += ct[k]; s2 += (double)ct[k] * ct[k]; n++; }
        }
        sd = sqrt(fmax(s2 / n - (s1 / n) * (s1 / n), 1e-6)); if (sd < 1) sd = 1;
    }
    const float sraw = step * (float)sd;
    char attrs[1024]; snprintf(attrs, sizeof attrs, "{\"latcomp\":1,\"lat_c\":%d,\"step\":%.9g,\"sd_ref\":%.9g,\"step_raw\":%.9g,\"stride\":4,\"origin\":[%lld,%lld,%lld],\"ct\":\"%s\"}", C, step, sd, sraw,
                               (long long)box[0], (long long)box[1], (long long)box[2], argv[4]);
    const int64_t cshape[3] = {box[3] / 4, box[4] / 4, box[5] / 4};
    zw W; if (zw_create(&W, argv[5], cshape, C + 1, "int8", 1, SH, CH, attrs)) { fprintf(stderr, "lc-encode: cannot create %s\n", argv[5]); return 1; }
    { char bp[1200]; snprintf(bp, sizeof bp, "%s/basis.lcb", argv[5]); FILE *f = fopen(bp, "wb"); const uint32_t hd[2] = {0x424c4655u, 1}; if (!f || fwrite(hd, sizeof hd, 1, f) != 1 || fwrite(lb, sizeof *lb, 1, f) != 1 || fclose(f)) { fprintf(stderr, "lc-encode: cannot write %s\n", bp); return 1; } }
    fprintf(stderr, "lc-encode: %lld x %lld x %lld voxels -> %lld x %lld x %lld cells x %d bytes, sd_ref %.2f, step %.3g (%.2f grey levels), %d threads\n", (long long)box[3], (long long)box[4], (long long)box[5],
            (long long)cshape[0], (long long)cshape[1], (long long)cshape[2], C + 1, sd, step, sraw, nth);
    enc_job *J = calloc((size_t)nth, sizeof *J); pthread_t *T = malloc((size_t)nth * sizeof *T);
    for (int64_t sz = 0; sz < cshape[0]; sz += SH) {
        for (int64_t sy = 0; sy < cshape[1]; sy += SH) for (int64_t sx = 0; sx < cshape[2]; sx += SH) {
            const int64_t o[3] = {box[0] + 4 * sz, box[1] + 4 * sy, box[2] + 4 * sx}, nn[3] = {V, V, V};
            if (z3_read(z, o, nn, ct, nth)) { fprintf(stderr, "lc-encode: read failed: %s\n", z3_error()); return 1; }
            for (int t = 0; t < nth; t++) { J[t].ct = ct; J[t].out = out; J[t].n[0] = J[t].n[1] = J[t].n[2] = V; J[t].C = C; J[t].step = sraw; J[t].lb = lb; J[t].z0 = (int)((int64_t)SH * t / nth); J[t].z1 = (int)((int64_t)SH * (t + 1) / nth);
                                            pthread_create(&T[t], nullptr, enc_run, &J[t]); }
            for (int t = 0; t < nth; t++) pthread_join(T[t], nullptr);
            if (zw_put(&W, sz, sy, sx, SH, SH, SH, out)) { fprintf(stderr, "lc-encode: write failed\n"); return 1; }
        }
        if (zw_flush(&W)) { fprintf(stderr, "lc-encode: write failed\n"); return 1; }
        fprintf(stderr, "\rlc-encode: cell planes %lld / %lld", (long long)(sz + SH), (long long)cshape[0]);
    }
    zw_free(&W); fprintf(stderr, "\n");
    double se = 0, sv = 0, bits = 0; const double ncell = (double)cshape[0] * cshape[1] * cshape[2];
    for (int c = 0; c <= C; c++) { long h[256] = {0}; for (int t = 0; t < nth; t++) for (int k = 0; k < 256; k++) h[k] += J[t].hist[c][k]; for (int k = 0; k < 256; k++) if (h[k]) { const double p = h[k] / ncell; bits -= p * log2(p); } }
    for (int t = 0; t < nth; t++) { se += J[t].se; sv += J[t].sv; }
    const long db = dir_bytes(argv[5]); const double vox = ncell * 64;
    printf("lc-encode: %s: %.4f bytes per voxel on disk (zstd; %.1fx smaller than uint8), zeroth-order entropy %.4f bytes per voxel; material cells %.1f%%, rms error %.2f grey levels (PSNR %.1f dB of 255)\n",
           argv[5], db / vox, vox / (db > 0 ? db : 1), bits / 64 / 8, 100 * sv / vox, sv > 0 ? sqrt(se / sv) : 0, sv > 0 && se > 0 ? 10 * log10(255.0 * 255.0 / (se / sv)) : 0);
    free(J); free(T); free(ct); free(out); free(lb); z3_close(z); store_close(st);
    return 0;
}
