/* ufsm latvol: a CT pyramid group re-encoded as latent cells (src/latvol.h), level by level, in the CT group's own layout (the same
   level names, shapes and shard grid), so the result is a drop-in CT source: z3_read gives the rebuilt CT, predict takes the cells.
   ufsm latvol-check: a latvol level against its source (bytes per voxel, rebuilt-CT error on sampled boxes). */
#include "latfit.h"
#include "latvol.h"
#include "nn.h"
#include "nn_latvol.h"
#include "store.h"
#include "sysinfo.h"
#include "z3w.h"
#include "zarr3.h"
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

static const char *opt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return dflt;
}
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec; }

/* up to nmax mostly-material boxes of edge E on a 5^3 lattice over [lo, hi), nearest the centre first (a masked scan's corners are air) */
static int sample_boxes(z3 *z, const int64_t lo[3], const int64_t hi[3], int E, int nmax, uint8_t **out, int nth) {
    int ord[125]; double dist[125]; int used = 0;
    for (int k = 0; k < 125; k++) { const int a = k / 25 - 2, b = k / 5 % 5 - 2, c = k % 5 - 2; ord[k] = k; dist[k] = a * a + b * b + c * c + 1e-3 * k; }
    for (int a = 1; a < 125; a++) for (int b = a; b > 0 && dist[ord[b]] < dist[ord[b - 1]]; b--) { const int t = ord[b]; ord[b] = ord[b - 1]; ord[b - 1] = t; }
    const size_t ev = (size_t)E * E * E; uint8_t *ct = malloc(ev);
    for (int q = 0; q < 125 && used < nmax && ct; q++) {
        const int k = ord[q], l[3] = {k / 25, k / 5 % 5, k % 5}; int64_t o[3]; const int64_t nn[3] = {E, E, E}; int ok = 1;
        for (int d = 0; d < 3; d++) { o[d] = lo[d] + (int64_t)((l[d] + 0.5) * (double)(hi[d] - lo[d]) / 5) / 128 * 128; if (o[d] + E > hi[d]) o[d] = (hi[d] - E) / 128 * 128; if (o[d] < lo[d]) ok = 0; }
        if (!ok) continue;
        int dup = 0; for (int p = 0; p < q; p++) { const int kp = ord[p]; int same = 1; for (int d = 0; d < 3; d++) { const int lp = d == 0 ? kp / 25 : d == 1 ? kp / 5 % 5 : kp % 5; int64_t op = lo[d] + (int64_t)((lp + 0.5) * (double)(hi[d] - lo[d]) / 5) / 128 * 128; if (op + E > hi[d]) op = (hi[d] - E) / 128 * 128; same &= op == o[d]; } dup |= same; }
        if (dup) continue;
        if (z3_box_any(z, o, nn) != 1) continue;
        if (z3_read(z, o, nn, ct, nth)) { free(ct); return -1; }
        size_t nzv = 0; for (size_t v = 0; v < ev; v += 7) nzv += ct[v] != 0;
        if (nzv * 7 < ev / 2) continue;
        out[used] = malloc(ev); if (!out[used]) break;
        memcpy(out[used++], ct, ev);
    }
    free(ct);
    return used;
}

typedef struct {
    const lv_codec *lc; const uint8_t *ct; const int16_t *q; int SH, gpuq;   /* ct: the shard's voxels [SH]^3; q: its indices [SH/4]^3 x C (device transform) or nullptr */
    uint8_t **enc; size_t *len; atomic_int next; atomic_int failed; atomic_long nmat, bytes;
} enc_job;
static void *enc_run(void *arg) {
    enc_job *j = arg; const int C = j->lc->C, cg = j->SH / 128, cd = j->SH / 4;
    uint8_t *chunk = malloc(128 * 128 * 128), *e = malloc(LV_CHUNK_BOUND); int16_t *q = malloc((size_t)LV_CELLS * LV_CELLS * LV_CELLS * C * sizeof *q);
    for (int i; chunk && e && q && (i = atomic_fetch_add(&j->next, 1)) < cg * cg * cg;) {
        const int gz = i / (cg * cg), gy = i / cg % cg, gx = i % cg; long nmat = 0;
        if (j->q) {
            for (int z = 0; z < LV_CELLS; z++) for (int y = 0; y < LV_CELLS; y++) memcpy(q + ((size_t)z * LV_CELLS + y) * LV_CELLS * C, j->q + (((size_t)(gz * 32 + z) * cd + gy * 32 + y) * cd + gx * 32) * C, (size_t)LV_CELLS * C * sizeof *q);
            for (int k = 0; k < LV_CELLS * LV_CELLS * LV_CELLS; k++) nmat += q[(size_t)k * C] != 0;
        } else {
            for (int z = 0; z < 128; z++) for (int y = 0; y < 128; y++) memcpy(chunk + ((size_t)z * 128 + y) * 128, j->ct + ((size_t)(gz * 128 + z) * j->SH + gy * 128 + y) * j->SH + gx * 128, 128);
            nmat = lv_forward(j->lc, chunk, q);
        }
        size_t len = 0;
        if (nmat && lv_encode_chunk(j->lc, q, e, &len)) { atomic_store(&j->failed, 1); break; }
        if (len) { j->enc[i] = malloc(len); if (!j->enc[i]) { atomic_store(&j->failed, 1); break; } memcpy(j->enc[i], e, len); j->len[i] = len; }
        atomic_fetch_add(&j->nmat, nmat); atomic_fetch_add(&j->bytes, (long)len);
    }
    if (!chunk || !e || !q) atomic_store(&j->failed, 1);
    free(chunk); free(e); free(q);
    return nullptr;
}

static int encode_level(store *st, const char *group, const char *lname, const char *outg, const lat_basis *lb, int C, float step, double sd_opt, int nth, int gpu,
                        const char *cache, int resume, const long long (*boxv)[6], int nbox, const char *tables, double *tot_bytes, double *tot_vox) {
    char key[1100], odir[1200], p[1400]; snprintf(key, sizeof key, "%s/%s", group, lname); snprintf(odir, sizeof odir, "%s/%s", outg, lname);
    z3 *z = z3_open(st, key, cache); if (!z) { fprintf(stderr, "latvol: cannot open %s: %s\n", key, z3_error()); return 1; }
    const z3_meta *m = z3_meta_of(z); const int SH = m->shard[0];
    if (!m->sharded || m->shard[1] != SH || m->shard[2] != SH || SH % 128 || m->esize != 1) { fprintf(stderr, "latvol: %s: need a uint8 array in cubic shards of 128^3 chunks\n", key); return 1; }
    /* the shards to write: all, or those the boxes touch (the array keeps the level's shape: a partial store is a partial mirror) */
    const int64_t gs[3] = {(m->shape[0] + SH - 1) / SH, (m->shape[1] + SH - 1) / SH, (m->shape[2] + SH - 1) / SH};
    uint8_t *want = malloc((size_t)(gs[0] * gs[1] * gs[2])); memset(want, nbox ? 0 : 1, (size_t)(gs[0] * gs[1] * gs[2]));
    for (int b = 0; b < nbox; b++) {
        int64_t a[3], e[3]; int ok = 1;
        for (int d = 0; d < 3; d++) { a[d] = boxv[b][d] < 0 ? 0 : boxv[b][d] / SH; e[d] = (boxv[b][d] + boxv[b][3 + d] + SH - 1) / SH; if (e[d] > gs[d]) e[d] = gs[d]; ok &= e[d] > a[d]; }
        for (int64_t z1 = a[0]; ok && z1 < e[0]; z1++) for (int64_t y1 = a[1]; y1 < e[1]; y1++) for (int64_t x1 = a[2]; x1 < e[2]; x1++) want[(z1 * gs[1] + y1) * gs[2] + x1] = 1;
    }
    const int64_t lo[3] = {0, 0, 0}, hi[3] = {m->shape[0], m->shape[1], m->shape[2]};   /* (the codec's sample: the whole level, whatever is written) */
    /* the step's sd_ref, the entropy tables and the rebuilt-CT error from sampled material boxes */
    lv_codec lc; int have = 0;
    snprintf(p, sizeof p, "%s/latvol.bin", odir);
    if (resume) {   /* an existing store's codec is kept as it is: its shards were written with it */
        FILE *f = fopen(p, "rb"); static uint8_t sb[1 << 17]; const size_t sn = f ? fread(sb, 1, sizeof sb, f) : 0; if (f) fclose(f);
        if (sn && !lv_codec_parse(&lc, sb, sn)) {
            if (lc.C != C || fabsf(lc.step - step) > 1e-6f * step) { fprintf(stderr, "latvol: --resume: %s has C %d step %g, asked C %d step %g\n", odir, lc.C, lc.step, C, step); return 2; }
            for (int c = 0; c < C; c++) for (int k = 0; k < 64; k++) if (lc.basis[c][k] != lb->b[c][k]) { fprintf(stderr, "latvol: --resume: %s was written with another transform\n", odir); return 2; }
            have = 1; fprintf(stderr, "latvol: %s: resuming with the store's codec (sd_ref %.2f, step %.2f grey levels)\n", key, lc.sd_ref, lc.step_raw);
        }
    }
    enum { E = 256, NS = 24 }; uint8_t *smp[NS]; const double t0 = now(); double hbits = 0;
    if (!have) {
    const int ns = sample_boxes(z, lo, hi, E, NS, smp, nth);
    if (ns < 1) { fprintf(stderr, "latvol: %s: no mostly-material %d^3 box on the 5^3 lattice%s\n", key, E, ns < 0 ? " (read failed)" : ""); return 2; }
    double s1 = 0, s2 = 0, n = 0;
    for (int k = 0; k < ns; k++) for (size_t v = 0; v < (size_t)E * E * E; v += 3) { s1 += smp[k][v]; s2 += (double)smp[k][v] * smp[k][v]; n++; }
    double sd = sd_opt > 0 ? sd_opt : sqrt(fmax(s2 / n - (s1 / n) * (s1 / n), 1e-6)); if (sd < 1) sd = 1;
    uint64_t (*hist)[LV_NSYM] = calloc(LV_MAXC, sizeof *hist);
    uint8_t map[LV_MAXC]; int ntab = 0;   /* --tables N: channels 1.. share N - 1 tables in runs (fewer device tables; default one a channel) */
    if (tables && atoi(tables) > 1 && atoi(tables) < C) { ntab = atoi(tables); map[0] = 0; for (int c = 1; c < C; c++) map[c] = (uint8_t)(1 + (int)((int64_t)(c - 1) * (ntab - 1) / (C - 1))); }
    if (lv_codec_init(&lc, C, &lb->b[0][0], step, (float)sd, ntab ? map : nullptr, ntab, nullptr)) { fprintf(stderr, "latvol: bad codec parameters\n"); return 2; }
    int16_t *q = malloc((size_t)LV_CELLS * LV_CELLS * LV_CELLS * C * sizeof *q); uint8_t *ch = malloc(128 * 128 * 128), *rb = malloc(128 * 128 * 128);
    double se = 0, sdrop = 0, sv = 0;
    for (int k = 0; k < ns; k++) for (int g = 0; g < 8; g++) {
        for (int zz = 0; zz < 128; zz++) for (int yy = 0; yy < 128; yy++) memcpy(ch + ((size_t)zz * 128 + yy) * 128, smp[k] + ((size_t)((g >> 2) * 128 + zz) * E + (g >> 1 & 1) * 128 + yy) * E + (g & 1) * 128, 128);
        lv_forward(&lc, ch, q); lv_hist(&lc, q, hist); lv_inverse(&lc, q, rb);
        for (int i = 0; i < LV_CELLS * LV_CELLS * LV_CELLS; i++) { const int16_t *o = q + (size_t)i * C; if (!o[0]) continue;
            const int cz = i / 1024, cy = i / 32 % 32, cx = i % 32; double v2 = 0, k2 = 0;
            for (int kk = 0; kk < 64; kk++) { const size_t vi = ((size_t)(4 * cz + (kk >> 4)) * 128 + 4 * cy + (kk >> 2 & 3)) * 128 + 4 * cx + (kk & 3); const double e = (double)rb[vi] - ch[vi]; se += e * e; v2 += (double)ch[vi] * ch[vi]; }
            for (int c = 0; c < C; c++) { double s = 0; for (int kk = 0; kk < 64; kk++) s += (double)lb->b[c][kk] * ch[((size_t)(4 * cz + (kk >> 4)) * 128 + 4 * cy + (kk >> 2 & 3)) * 128 + 4 * cx + (kk & 3)]; k2 += s * s; }
            sdrop += fmax(v2 - k2, 0); sv += 64; }
    }
    for (int k = 0; k < ns; k++) free(smp[k]);
    lv_codec_free(&lc);
    if (lv_codec_init(&lc, C, &lb->b[0][0], step, (float)sd, ntab ? map : nullptr, ntab, hist)) { fprintf(stderr, "latvol: table fit failed\n"); return 1; }
    lc.drop_ms = sv > 0 ? (float)(sdrop / sv) : 0;
    for (int t = 0; t < lc.ntab; t++) { double tot = 0; for (int s = 0; s < LV_NSYM; s++) tot += (double)hist[t][s]; for (int s = 0; s < LV_NSYM; s++) if (hist[t][s]) hbits -= (double)hist[t][s] * log2((double)hist[t][s] / tot); }
    fprintf(stderr, "latvol: %s: %lld x %lld x %lld, shard %d; %d sample boxes (%.1f s): sd_ref %.2f, step %.3g = %.2f grey levels, C %d, %d tables; rebuilt CT rms error %.2f (PSNR %.1f dB), dropped axes %.2f ms\n",
            key, (long long)m->shape[0], (long long)m->shape[1], (long long)m->shape[2], SH, ns, now() - t0, sd, step, lc.step_raw, C, lc.ntab, sv > 0 ? sqrt(se / sv) : 0, sv > 0 && se > 0 ? 10 * log10(255.0 * 255.0 / (se / sv)) : 0, lc.drop_ms);
    free(hist); free(ch); free(rb); free(q);
    }
    /* the output array */
    char codec[512], attrs[1024];
    snprintf(codec, sizeof codec, "{\"name\":\"latvol\",\"configuration\":{\"c\":%d,\"step_raw\":%.9g,\"side\":\"latvol.bin\"}}", C, lc.step_raw);
    snprintf(attrs, sizeof attrs, "{\"latvol\":{\"version\":1,\"c\":%d,\"step\":%.9g,\"sd_ref\":%.9g,\"step_raw\":%.9g,\"drop_ms\":%.6g,\"cell\":4,\"source\":\"%s\"}}", C, step, (double)lc.sd_ref, lc.step_raw, lc.drop_ms, key);
    z3w *w = z3w_create_codec(odir, m->shape, SH, codec, 0, attrs); if (!w) { fprintf(stderr, "latvol: cannot create %s: %s\n", odir, z3w_error()); return 1; }
    snprintf(p, sizeof p, "%s/latvol.bin", odir); if (!have && lv_codec_write(&lc, p)) { fprintf(stderr, "latvol: cannot write %s\n", p); return 1; }
    /* shards */
    const int cg = SH / 128, nc = cg * cg * cg, cd = SH / 4; const size_t sv3 = (size_t)SH * SH * SH, ncell = (size_t)cd * cd * cd;
    uint8_t *ct = nullptr; int16_t *qh = nullptr; void *d_ct = nullptr, *d_q = nullptr; nn_lv *L = nullptr;
    if (gpu >= 0) {
        ct = nn_host_alloc(sv3); qh = nn_host_alloc(ncell * C * sizeof *qh); d_ct = nn_malloc(sv3); d_q = nn_malloc(ncell * C * sizeof *qh);
        L = nn_latvol_create(lc.dt, lc.ntab, lc.map, &lc.bi[0][0], C, lc.step_raw);
    } else ct = malloc(sv3);
    if (!ct || (gpu >= 0 && (!qh || !L))) { fprintf(stderr, "latvol: allocation failed\n"); return 1; }
    uint8_t **enc = calloc((size_t)nc, sizeof *enc); size_t *len = calloc((size_t)nc, sizeof *len); pthread_t *T = malloc((size_t)nth * sizeof *T);
    double t_read = 0, t_fwd = 0, t_enc = 0, t_wr = 0, bytes = 0, vox = 0, nmat = 0; long nsh = 0, nskip = 0, ndone = 0;
    const double tl = now();
    for (int64_t sz = 0; sz < gs[0]; sz++) for (int64_t sy = 0; sy < gs[1]; sy++) for (int64_t sx = 0; sx < gs[2]; sx++) {
        if (!want[(sz * gs[1] + sy) * gs[2] + sx]) continue;
        nsh++;
        const int64_t o[3] = {sz * SH, sy * SH, sx * SH}, nn3[3] = {SH, SH, SH};
        double ev = 1; for (int d = 0; d < 3; d++) ev *= (double)((o[d] + SH > m->shape[d] ? m->shape[d] : o[d] + SH) - o[d]);
        vox += ev;
        if (resume) { struct stat sb; snprintf(p, sizeof p, "%s/c/%lld/%lld/%lld", odir, (long long)sz, (long long)sy, (long long)sx); if (!stat(p, &sb)) { bytes += (double)sb.st_size; nskip++; continue; } }
        if (z3_shard_present(z, sz, sy, sx) != 1) continue;
        double t1 = now();
        if (z3_read(z, o, nn3, ct, nth)) { fprintf(stderr, "latvol: read failed: %s\n", z3_error()); return 1; }
        double t2 = now(); t_read += t2 - t1;
        if (L) { nn_h2d(d_ct, ct, sv3); nn_latvol_forward(L, d_ct, cd, cd, cd, d_q); nn_d2h(qh, d_q, ncell * C * sizeof *qh); }
        double t3 = now(); t_fwd += t3 - t2;
        enc_job J = {.lc = &lc, .ct = ct, .q = L ? qh : nullptr, .SH = SH, .enc = enc, .len = len};
        for (int t = 0; t < nth; t++) pthread_create(&T[t], nullptr, enc_run, &J);
        for (int t = 0; t < nth; t++) pthread_join(T[t], nullptr);
        if (atomic_load(&J.failed)) { fprintf(stderr, "latvol: encode failed\n"); return 1; }
        double t4 = now(); t_enc += t4 - t3;
        nmat += (double)atomic_load(&J.nmat);
        if (z3w_write_shard_enc(w, sz, sy, sx, enc, len)) { fprintf(stderr, "latvol: write failed: %s\n", z3w_error()); return 1; }
        t_wr += now() - t4; bytes += (double)atomic_load(&J.bytes) + (atomic_load(&J.bytes) ? nc * 16 + 4 : 0); ndone++;
        fprintf(stderr, "\rlatvol: %s shard %ld (%ld written, %ld kept), %.2f GB, %.0f s", lname, nsh, ndone, nskip, bytes / 1e9, now() - tl);
    }
    const double wall = now() - tl;
    fprintf(stderr, "\n");
    printf("latvol: %s: %.3f GB for %.1f Gvox (%.4f bytes per voxel; %.1f%% material cells, %.4f B per material voxel of the %ld shards written); %.0f s: read %.0f, transform %.0f%s, entropy %.0f, write %.0f (%.2f Gvox/s of written shards)\n",
           odir, bytes / 1e9, vox / 1e9, vox > 0 ? bytes / vox : 0, ndone ? 100 * nmat * 64 / ((double)ndone * (double)sv3) : 0, nmat > 0 ? bytes / (nmat * 64) : 0, ndone, wall, t_read, t_fwd, L ? " (GPU)" : " (in the entropy threads)", t_enc, t_wr,
           wall > 0 ? (double)ndone * (double)sv3 / 1e9 / wall : 0);
    (void)hbits; *tot_bytes += bytes; *tot_vox += vox;
    if (L) { nn_latvol_free(L); nn_free(d_ct); nn_free(d_q); nn_host_free(ct); nn_host_free(qh); } else free(ct);
    free(enc); free(len); free(T); free(want); lv_codec_free(&lc); z3w_close(w); z3_close(z);
    return 0;
}

int cmd_latvol(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: ufsm latvol BASIS.lcb ROOT CT-GROUP OUT-GROUP --levels L[,L..] [--c 29] [--step 0.25] [--sd S] [--gpu 0 | -1] [--threads N]\n"
                                    "                   [--cache DIR] [--resume 1] [--box Z,Y,X,NZ,NY,NX] [--tables N]\n"
                                    "  the CT levels (group paths, e.g. 2,3) as latent cells of the BASIS (train --lat-fit; the first --c rows, quantiser step --step x the\n"
                                    "  level's sd, measured or --sd) in OUT-GROUP, a CT group in the same layout: train / predict read it as their CT (src/latvol.h).\n"
                                    "  --gpu -1: the transform on the CPU. --box / --boxes FILE: only the shards those boxes touch (voxels of the first --levels entry; other\n"
                                    "  levels: scaled). --resume 1: shards already written are kept, with the store's own codec.\n"); return 2; }
    lat_basis *lb = malloc(sizeof *lb); if (lat_basis_load(argv[2], lb)) { fprintf(stderr, "latvol: cannot read the transform %s\n", argv[2]); return 1; }
    store *st = store_open(argv[3]); if (!st) { fprintf(stderr, "latvol: cannot open %s\n", argv[3]); return 1; }
    const char *group = argv[4], *outg = argv[5], *levels = opt(argc, argv, "--levels", nullptr), *cache = opt(argc, argv, "--cache", nullptr);
    if (!levels) { fprintf(stderr, "latvol: --levels L[,L..]\n"); return 2; }
    const int C = atoi(opt(argc, argv, "--c", "29")), nth = ufsm_threads(atoi(opt(argc, argv, "--threads", "0")), 8), gpu = atoi(opt(argc, argv, "--gpu", "0")), resume = atoi(opt(argc, argv, "--resume", "0"));
    const float step = (float)atof(opt(argc, argv, "--step", "0.25")); const double sd = atof(opt(argc, argv, "--sd", "0"));
    if (C < 1 || C > LV_MAXC || !(step > 0)) { fprintf(stderr, "latvol: --c 1..64, --step > 0\n"); return 2; }
    enum { MAXBOX = 4096 }; long long (*bx)[6] = malloc(MAXBOX * sizeof *bx), (*b2)[6] = malloc(MAXBOX * sizeof *b2); int nbox = 0;
    if (*opt(argc, argv, "--box", "")) { if (sscanf(opt(argc, argv, "--box", ""), "%lld,%lld,%lld,%lld,%lld,%lld", bx[0], bx[0] + 1, bx[0] + 2, bx[0] + 3, bx[0] + 4, bx[0] + 5) != 6) { fprintf(stderr, "latvol: --box Z,Y,X,NZ,NY,NX\n"); return 2; } nbox = 1; }
    if (*opt(argc, argv, "--boxes", "")) {   /* a file of boxes, one "z y x nz ny nx" (or comma separated) a line */
        FILE *bf = fopen(opt(argc, argv, "--boxes", ""), "r"); char ln[512]; if (!bf) { fprintf(stderr, "latvol: cannot read --boxes %s\n", opt(argc, argv, "--boxes", "")); return 2; }
        while (nbox < MAXBOX && fgets(ln, sizeof ln, bf)) { for (char *c = ln; *c; c++) if (*c == ',' || *c == '\t') *c = ' ';
            if (sscanf(ln, "%lld %lld %lld %lld %lld %lld", bx[nbox], bx[nbox] + 1, bx[nbox] + 2, bx[nbox] + 3, bx[nbox] + 4, bx[nbox] + 5) == 6) nbox++; }
        fclose(bf); if (!nbox) { fprintf(stderr, "latvol: --boxes: no box read\n"); return 2; }
    }
    {   /* the group's metadata as it is (level names, voxel sizes) */
        char gk[1100], cmd[1300], gp[1300]; snprintf(gk, sizeof gk, "%s/zarr.json", group); size_t gn = 0; uint8_t *gt = store_read_all(st, gk, &gn);
        if (!gt) { fprintf(stderr, "latvol: cannot read %s\n", gk); return 1; }
        snprintf(cmd, sizeof cmd, "mkdir -p '%s'", outg); if (system(cmd)) return 1;
        snprintf(gp, sizeof gp, "%s/zarr.json", outg); FILE *f = fopen(gp, "wb"); if (!f || fwrite(gt, 1, gn, f) != gn || fclose(f)) { fprintf(stderr, "latvol: cannot write %s\n", gp); return 1; }
        free(gt);
    }
    if (gpu >= 0) nn_init(gpu);
    double tb = 0, tv = 0; int first = -1;
    char lv[256]; snprintf(lv, sizeof lv, "%s", levels);
    for (char *tok = strtok(lv, ","); tok; tok = strtok(nullptr, ",")) {
        const int L = atoi(tok); if (first < 0) first = L;
        for (int b = 0; b < nbox; b++) { const int sh = L - first; for (int i = 0; i < 6; i++) b2[b][i] = sh >= 0 ? (bx[b][i] + (i >= 3 ? (1ll << sh) - 1 : 0)) >> sh : bx[b][i] << -sh; }
        const int rc = encode_level(st, group, tok, outg, lb, C, step, sd, nth, gpu, cache, resume, b2, nbox, opt(argc, argv, "--tables", nullptr), &tb, &tv);
        if (rc) return rc;
    }
    printf("latvol: %s: %.3f GB in all (%.4f bytes per voxel)\n", outg, tb / 1e9, tv > 0 ? tb / tv : 0);
    free(lb); store_close(st);
    return 0;
}

/* ufsm latvol-check LV-ROOT LV-KEY CT-ROOT CT-KEY [--n 8] [--threads N]: sampled 256^3 boxes of the latvol level against its source */
int cmd_latvol_check(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: ufsm latvol-check LV-ROOT LV-ARRAY CT-ROOT CT-ARRAY [--n 8] [--threads N] [--cache DIR]\n"); return 2; }
    store *sa = store_open(argv[2]), *sb = store_open(argv[4]); const char *cache = opt(argc, argv, "--cache", nullptr);
    z3 *za = sa ? z3_open(sa, argv[3], cache) : nullptr, *zb = sb ? z3_open(sb, argv[5], cache) : nullptr;
    if (!za || !zb) { fprintf(stderr, "latvol-check: cannot open the arrays: %s\n", z3_error()); return 1; }
    const lv_codec *lc = z3_latvol(za); if (!lc) { fprintf(stderr, "latvol-check: %s is not a latvol array\n", argv[3]); return 2; }
    const int nth = ufsm_threads(atoi(opt(argc, argv, "--threads", "0")), 8), nb = atoi(opt(argc, argv, "--n", "8"));
    int64_t grid[3]; uint8_t *bits = nullptr; const int64_t npres = z3_chunk_map(za, grid, &bits); if (npres < 0) { fprintf(stderr, "latvol-check: no chunk map: %s\n", z3_error()); return 1; }
    printf("latvol-check: %s: C %d, step_raw %.3f (step %.3g x sd_ref %.2f), %d tables, %lld of %lld chunks stored\n", argv[3], lc->C, lc->step_raw, lc->step, lc->sd_ref, lc->ntab, (long long)npres, (long long)(grid[0] * grid[1] * grid[2]));
    enum { E = 256 }; uint8_t *a = malloc((size_t)E * E * E), *b = malloc((size_t)E * E * E); uint64_t rs = 88172645463325252ull; double se = 0, sn = 0, ta = 0, tb = 0; long air = 0; int done = 0;
    for (int tries = 0; tries < 4000 && done < nb; tries++) {
        rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; const int64_t c = (int64_t)(rs % (uint64_t)(grid[0] * grid[1] * grid[2]));
        if (!bits[c]) continue;
        const int64_t o[3] = {c / (grid[1] * grid[2]) * 128, c / grid[2] % grid[1] * 128, c % grid[2] * 128}, nn3[3] = {E, E, E};
        double t0 = now(); if (z3_read(za, o, nn3, a, nth)) { fprintf(stderr, "latvol-check: read failed: %s\n", z3_error()); return 1; }
        double t1 = now(); if (z3_read(zb, o, nn3, b, nth)) { fprintf(stderr, "latvol-check: read failed: %s\n", z3_error()); return 1; }
        double t2 = now(); size_t nz = 0; for (size_t v = 0; v < (size_t)E * E * E; v += 5) nz += b[v] != 0;
        if (nz * 5 < (size_t)E * E * E / 2) continue;
        ta += t1 - t0; tb += t2 - t1;
        for (size_t v = 0; v < (size_t)E * E * E; v++) { if (b[v]) { const double e = (double)a[v] - b[v]; se += e * e; sn++; } air += !a[v] != !b[v]; }
        done++;
    }
    printf("latvol-check: %d material boxes of %d^3: rebuilt CT rms error %.2f grey levels (PSNR %.1f dB), %ld voxels air in one and not the other; host read %.0f Mvox/s (latvol) vs %.0f (source), %d threads\n",
           done, E, sn > 0 ? sqrt(se / sn) : 0, se > 0 ? 10 * log10(255.0 * 255.0 / (se / sn)) : 0, air, ta > 0 ? done * (double)E * E * E / 1e6 / ta : 0, tb > 0 ? done * (double)E * E * E / 1e6 / tb : 0, nth);
    free(a); free(b); free(bits); z3_close(za); z3_close(zb); store_close(sa); store_close(sb);
    return 0;
}
