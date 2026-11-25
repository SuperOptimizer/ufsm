/* ufsm dinovol-probe (lead 11-24): is ink linearly readable from the frozen scrollprize dinovol features on our coarse scans?
   feats: per ink-devcrops crop (tifxyz + labels / mask, src/inksplat.c), the dinovol features at the surface points of a pixel lattice:
          patch tokens (E, trilinear between token centres 8 k + 3.5), the dense head's unit dense (nd) and ink (ni) vectors at the point.
          Tiles of the training crop (192), the volume-wide robust normalisation of dinovol-embed; each point from the tile it is most
          central in (>= margin voxels from every face where the volume allows).
   fit:   a weighted logistic probe (soft label = the cell's ink share, weight = its supervised px) on standardised features, Newton steps
          with an L2 ridge; feature blocks by --feat.
   apply: probe logits on another feature file -> one uint8 map per crop on the crop's 256 px canvas (bilinear over the lattice, then
          the sigmoid), ink-score's input (dev/devscore.sh's scoring, scripted outside). */
#include "dinovol.h"
#include "imgops.h"
#include "json.h"
#include "optcheck.h"
#include "store.h"
#include "sysinfo.h"
#include "tiffw.h"
#include "xnet.h"
#include "zarr3.h"
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

static const char *opt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], opt_n(name))) return argv[i + 1];
    return dflt;
}
static double pnow(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static uint16_t f2h(float f) {   /* round to nearest even fp16 (finite inputs) */
    uint32_t x; memcpy(&x, &f, 4); const uint32_t s = (x >> 16) & 0x8000u; int e = (int)((x >> 23) & 0xff) - 127 + 15; uint32_t m = x & 0x7fffffu;
    if (e >= 31) return (uint16_t)(s | 0x7c00u);
    if (e <= 0) { if (e < -10) return (uint16_t)s; m |= 0x800000u; const int sh = 14 - e; uint32_t r = m >> sh; const uint32_t rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (r & 1))) r++; return (uint16_t)(s | r); }
    uint32_t r = ((uint32_t)e << 10) | (m >> 13); const uint32_t rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1))) r++;
    return (uint16_t)(s | r);
}
static float h2f(uint16_t h) {
    const uint32_t s = (uint32_t)(h & 0x8000u) << 16; int e = (h >> 10) & 0x1f; uint32_t m = h & 0x3ffu, x;
    if (!e) { if (!m) x = s; else { e = 1; while (!(m & 0x400u)) { m <<= 1; e--; } m &= 0x3ffu; x = s | ((uint32_t)(e + 112) << 23) | (m << 13); } }
    else if (e == 31) x = s | 0x7f800000u | (m << 13);
    else x = s | ((uint32_t)(e + 112) << 23) | (m << 13);
    float f; memcpy(&f, &x, 4); return f;
}

/* ---- feature file: header, crop names, rows {crop, r, c, label, weight, F fp16} ---- */
typedef struct { char magic[8]; int32_t F, E, nd, ni, ncrop, stride, win, pad; } dvf_head;
enum { DVF_NAME = 256 };

/* a crop: surface grid, labels / mask on the canvas, its CT key and voxel size */
typedef struct { int w, h, N; float *X, *Y, *Z; double scale; uint8_t *lab, *msk; char ct[512]; double um; } pcrop;
static void pcrop_free(pcrop *c) { free(c->X); free(c->Y); free(c->Z); free(c->lab); free(c->msk); }
static int pcrop_load(pcrop *c, const char *dir) {
    memset(c, 0, sizeof *c); char p[1400];
    snprintf(p, sizeof p, "%s/crop.json", dir); FILE *f = fopen(p, "rb"); if (!f) { fprintf(stderr, "dinovol-probe: cannot read %s\n", p); return -1; }
    char t[8192]; size_t n = fread(t, 1, sizeof t - 1, f); fclose(f); t[n] = 0;
    json *j = json_parse(t, n); if (!j) { fprintf(stderr, "dinovol-probe: bad %s\n", p); return -1; }
    snprintf(c->ct, sizeof c->ct, "%s", json_str(json_get(j, "ct"), "")); c->um = json_num(json_get(j, "um"), 0); c->N = (int)json_num(json_get(j, "win"), 256);
    json_free(j);
    snprintf(p, sizeof p, "%s/meta.json", dir); f = fopen(p, "rb"); if (!f) { fprintf(stderr, "dinovol-probe: cannot read %s\n", p); return -1; }
    n = fread(t, 1, sizeof t - 1, f); fclose(f); t[n] = 0;
    j = json_parse(t, n); if (!j) { fprintf(stderr, "dinovol-probe: bad %s\n", p); return -1; }
    c->scale = json_num(json_at(json_get(j, "scale"), 0), 0); json_free(j);
    static const char *nm[3] = {"x", "y", "z"}; float **A[3] = {&c->X, &c->Y, &c->Z};
    for (int k = 0; k < 3; k++) { int w, h; snprintf(p, sizeof p, "%s/%s.tif", dir, nm[k]); *A[k] = img_read_f32(p, &w, &h);
        if (!*A[k] || (k && (w != c->w || h != c->h))) { fprintf(stderr, "dinovol-probe: cannot read %s\n", p); return -1; } c->w = w; c->h = h; }
    int lw, lh, mw, mh; snprintf(p, sizeof p, "%s/labels.tif", dir); c->lab = img_read_u8(p, &lw, &lh);
    snprintf(p, sizeof p, "%s/mask.tif", dir); c->msk = img_read_u8(p, &mw, &mh);
    if (!c->lab || !c->msk || lw != c->N || lh != c->N || mw != c->N || mh != c->N || !c->ct[0] || !(c->scale > 0)) {
        fprintf(stderr, "dinovol-probe: %s: labels / mask not %d^2, or no ct / scale\n", dir, c->N); return -1; }
    return 0;
}
/* the surface point at canvas pixel (r, c) (bilinear over the grid at (c scale, r scale)); (z, y, x); 0 when the cell is valid */
static int pcrop_point(const pcrop *s, double r, double c, double P[3]) {
    const double gu = c * s->scale, gv = r * s->scale;
    if (gu < 0 || gv < 0 || gu > s->w - 1 || gv > s->h - 1) return -1;
    int c0 = (int)gu, r0 = (int)gv; if (c0 >= s->w - 1) c0 = s->w - 2; if (r0 >= s->h - 1) r0 = s->h - 2;
    const double fu = gu - c0, fv = gv - r0; const size_t i00 = (size_t)r0 * s->w + c0, i01 = i00 + 1, i10 = i00 + s->w, i11 = i10 + 1;
    if (s->Z[i00] < 0 || s->Z[i01] < 0 || s->Z[i10] < 0 || s->Z[i11] < 0) return -1;
    const float *A[3] = {s->Z, s->Y, s->X};
    for (int d = 0; d < 3; d++) P[d] = (1 - fv) * ((1 - fu) * A[d][i00] + fu * A[d][i01]) + fv * ((1 - fu) * A[d][i10] + fu * A[d][i11]);
    return 0;
}

/* per-scan normalisation statistics, cached as text files (dinovol-embed's: median, scale, p99) */
static int scan_stats(const char *dir, const char *key, z3 *ct, int nthreads, float S[3]) {
    char p[1400], k2[600]; snprintf(k2, sizeof k2, "%s", key); for (char *q = k2; *q; q++) if (*q == '/') *q = '_';
    snprintf(p, sizeof p, "%s/stats_%s.txt", dir, k2);
    FILE *f = fopen(p, "r"); double m, s, h;
    if (f) { const int ok = fscanf(f, "%lf %lf %lf", &m, &s, &h) == 3; fclose(f); if (ok) { S[0] = (float)m; S[1] = (float)s; S[2] = (float)h; return 0; } }
    const double t0 = pnow();
    if (dv_ct_stats(ct, 8, nthreads, &m, &s, &h)) return -1;
    fprintf(stderr, "dinovol-probe: %s: median %.2f scale %.3f p99 %.1f (%.0f s)\n", key, m, s, h, pnow() - t0);
    f = fopen(p, "w"); if (f) { fprintf(f, "%.17g %.17g %.17g\n", m, s, h); fclose(f); }
    S[0] = (float)m; S[1] = (float)s; S[2] = (float)h; return 0;
}

static int cmd_feats(int argc, char **argv) {
    const char *ck = opt(argc, argv, "--ckpt", nullptr), *root = opt(argc, argv, "--root", "/vesuvius/ufsm/v2/ct"), *cdir = opt(argc, argv, "--crops", nullptr),
               *list = opt(argc, argv, "--list", nullptr), *out = opt(argc, argv, "--out", nullptr), *cache = opt(argc, argv, "--cache", nullptr);
    int T = opt_atoi("--tile", opt(argc, argv, "--tile", "0")); const int M = opt_atoi("--margin", opt(argc, argv, "--margin", "32")),
              st = opt_atoi("--px-stride", opt(argc, argv, "--px-stride", "4")), onlysup = opt_atoi("--sup-only", opt(argc, argv, "--sup-only", "1")),
              maxc = opt_atoi("--max-crops", opt(argc, argv, "--max-crops", "0"));
    const int nthreads = ufsm_threads(opt_atoi("--threads", opt(argc, argv, "--threads", "0")), 6);
    if (!ck || !cdir || !out || T % 16 || M < 0 || (T && 2 * M >= T) || st < 1) {
        fprintf(stderr, "usage: ufsm dinovol-probe feats --ckpt PT --crops DIR [--list crops.tsv] --out F.dvf [--root CTROOT] [--tile T (0: the model's training crop)] [--margin 32]\n"
                        "         [--px-stride 4] [--sup-only 1] [--max-crops N] [--stats-dir DIR] [--gpu 0] [--gpu-mem 7] [--threads 6] [--cache DIR]\n");
        return 2; }
    char lp[1400]; if (!list) { snprintf(lp, sizeof lp, "%s/crops.tsv", cdir); list = lp; }
    FILE *lf = fopen(list, "r"); if (!lf) { fprintf(stderr, "dinovol-probe: cannot read %s\n", list); return 1; }
    int nc = 0, cap = 256; char (*names)[DVF_NAME] = malloc((size_t)cap * DVF_NAME); char line[4096];
    while (fgets(line, sizeof line, lf)) { if (line[0] == '#' || line[0] == '\n') continue; char *tab = strchr(line, '\t'); if (tab) *tab = 0; else line[strcspn(line, "\n")] = 0;
        if (nc == cap) { cap *= 2; names = realloc(names, (size_t)cap * DVF_NAME); } snprintf(names[nc++], DVF_NAME, "%s", line); if (maxc > 0 && nc >= maxc) break; }
    fclose(lf);
    if (!nc) { fprintf(stderr, "dinovol-probe: no crops in %s\n", list); return 1; }
    const char *sdir = opt(argc, argv, "--stats-dir", cdir);
    if (xnet_init(opt_atoi("--gpu", opt(argc, argv, "--gpu", "0")))) return 1;
    xnet_set_budget((size_t)(opt_atof("--gpu-mem", opt(argc, argv, "--gpu-mem", "7")) * 1e9));
    store_global_init();
    dv_net *net = dv_load(ck); if (!net) return 1;
    if (!T) T = dv_train_tile(net);
    if (2 * M >= T) { fprintf(stderr, "dinovol-probe: margin %d too large for tile %d\n", M, T); return 2; }
    fprintf(stderr, "dinovol-probe: tile %d, margin %d\n", T, M);
    int nd = 0, ni = 0; dv_dense_dims(net, &nd, &ni); const int E = dv_embed_dim(net), F = E + nd + ni, g = T / 8;
    FILE *fo = fopen(out, "wb"); if (!fo) { fprintf(stderr, "dinovol-probe: cannot write %s\n", out); return 1; }
    dvf_head H = {{'D', 'V', 'P', 'F', '0', '0', '0', '1'}, F, E, nd, ni, nc, st, 0, 0};
    fwrite(&H, sizeof H, 1, fo); fwrite(names, DVF_NAME, (size_t)nc, fo);
    uint8_t *u8 = malloc((size_t)T * T * T), *u8d = xmalloc_dev((size_t)T * T * T);
    uint16_t *tok = malloc((size_t)g * g * g * E * 2);
    const int NP = 65536; float *pts = malloc((size_t)NP * 3 * 4), *ptsd = xmalloc_dev((size_t)NP * 3 * 4), *acc = xmalloc_dev((size_t)NP * (nd + ni) * 4),
        *acch = malloc((size_t)NP * (nd + ni) * 4);
    uint16_t *rowf = malloc((size_t)F * 2); long nrows = 0; int rc = 0; const double t0 = pnow(); long ntiles = 0;
    char curkey[512] = ""; store *s = store_open(root); z3 *ct = nullptr; float S[3] = {0};
    for (int ci = 0; ci < nc && !rc; ci++) {
        char d[1400]; snprintf(d, sizeof d, "%s/%s", cdir, names[ci]);
        pcrop C; if (pcrop_load(&C, d)) { rc = 1; break; }
        if (strcmp(curkey, C.ct)) {
            if (ct) z3_close(ct);
            char ak[600]; snprintf(ak, sizeof ak, "%s/0", C.ct);   /* (crop.json "ct" is the scan's group: level 0) */
            ct = s ? z3_open(s, ak, cache) : nullptr;
            if (!ct || scan_stats(sdir, C.ct, ct, nthreads, S)) { fprintf(stderr, "dinovol-probe: cannot open / normalise %s %s\n", root, C.ct); pcrop_free(&C); rc = 1; break; }
            snprintf(curkey, sizeof curkey, "%s", C.ct);
        }
        const z3_meta *zm = z3_meta_of(ct);
        /* lattice points (cell centres of st x st pixel blocks) with label share / supervised weight */
        const int G = C.N / st; int np = 0; double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
        double *P = malloc(sizeof(double) * 3 * G * G); float *lab = malloc(sizeof(float) * G * G), *wg = malloc(sizeof(float) * G * G); int *rr = malloc(sizeof(int) * G * G * 2);
        for (int a = 0; a < G; a++) for (int b = 0; b < G; b++) {
            int sup = 0, ink = 0;
            for (int y = a * st; y < (a + 1) * st; y++) for (int x = b * st; x < (b + 1) * st; x++) { const size_t k = (size_t)y * C.N + x; if (C.msk[k] >= 128) { sup++; ink += C.lab[k] >= 128; } }
            if (onlysup && !sup) continue;
            double q[3]; if (pcrop_point(&C, a * st + (st - 1) * 0.5, b * st + (st - 1) * 0.5, q)) continue;
            for (int dd = 0; dd < 3; dd++) { P[3 * np + dd] = q[dd]; if (q[dd] < lo[dd]) lo[dd] = q[dd]; if (q[dd] > hi[dd]) hi[dd] = q[dd]; }
            lab[np] = sup ? (float)ink / sup : 0.f; wg[np] = (float)sup; rr[2 * np] = a; rr[2 * np + 1] = b; np++;
        }
        if (!np) { free(P); free(lab); free(wg); free(rr); pcrop_free(&C); continue; }
        /* tiles: per axis one centred tile, or evenly spaced ones covering [lo - M, hi + M]; origins clamped into the volume */
        int nt[3]; int64_t org[3][16];
        for (int dd = 0; dd < 3; dd++) {
            const double a0 = floor(lo[dd]) - M, a1 = ceil(hi[dd]) + 1 + M, ext = a1 - a0;
            nt[dd] = ext <= T ? 1 : (int)ceil((ext - T) / (double)(T - 2 * M)) + 1; if (nt[dd] > 16) nt[dd] = 16;
            for (int k = 0; k < nt[dd]; k++) {
                double o = nt[dd] == 1 ? 0.5 * (a0 + a1) - T / 2.0 : a0 + (ext - T) * k / (nt[dd] - 1);
                int64_t oi = (int64_t)floor(o); if (oi > zm->shape[dd] - T) oi = zm->shape[dd] - T; if (oi < 0) oi = 0; org[dd][k] = oi;
            }
        }
        int *own = malloc(sizeof(int) * np);   /* the tile index each point is most central in */
        for (int i = 0; i < np; i++) { double best = -1e300; own[i] = 0;
            for (int kz = 0; kz < nt[0]; kz++) for (int ky = 0; ky < nt[1]; ky++) for (int kx = 0; kx < nt[2]; kx++) {
                const int64_t o[3] = {org[0][kz], org[1][ky], org[2][kx]}; double m = 1e300;
                for (int dd = 0; dd < 3; dd++) { const double u = P[3 * i + dd] - o[dd], e = u < T - 1 - u ? u : T - 1 - u; if (e < m) m = e; }
                if (m > best) { best = m; own[i] = (kz * nt[1] + ky) * nt[2] + kx; } } }
        for (int ti = 0; ti < nt[0] * nt[1] * nt[2] && !rc; ti++) {
            int cnt = 0; for (int i = 0; i < np; i++) cnt += own[i] == ti;
            if (!cnt) continue;
            const int kz = ti / (nt[1] * nt[2]), ky = ti / nt[2] % nt[1], kx = ti % nt[2];
            const int64_t o[3] = {org[0][kz], org[1][ky], org[2][kx]}, n3[3] = {T, T, T};
            if (z3_read(ct, o, n3, u8, nthreads)) { fprintf(stderr, "dinovol-probe: CT read: %s\n", z3_error()); rc = 1; break; }
            xh2d(u8d, u8, (size_t)T * T * T);
            if (dv_forward(net, u8d, T, S[0], S[1], S[2])) { rc = 1; break; }
            ntiles++;
            xd2h(tok, dv_tokens(net), (size_t)g * g * g * E * 2);
            int done = 0;
            while (done < cnt && !rc) {   /* points of this tile, in chunks of NP */
                int m = 0; static int sel[65536];
                for (int i = 0, seen = 0; i < np && m < NP; i++) if (own[i] == ti) { if (seen++ < done) continue; sel[m] = i; for (int dd = 0; dd < 3; dd++) pts[3 * m + dd] = (float)(P[3 * i + dd] - o[dd]); m++; }
                xh2d(ptsd, pts, (size_t)m * 12);
                if (dv_net_points(net, ptsd, nullptr, nullptr, m, acc)) { rc = 1; break; }
                xd2h(acch, acc, (size_t)m * (nd + ni) * 4);
                for (int q = 0; q < m; q++) {
                    const int i = sel[q]; float w8[8]; size_t ix[8]; int c0[3]; float fr[3];
                    for (int dd = 0; dd < 3; dd++) { double u = (pts[3 * q + dd] - 3.5) / 8; if (u < 0) u = 0; if (u > g - 1) u = g - 1;
                        c0[dd] = (int)u; if (c0[dd] > g - 2) c0[dd] = g - 2; fr[dd] = (float)(u - c0[dd]); }
                    for (int k = 0; k < 8; k++) { const int dz = k >> 2, dy = (k >> 1) & 1, dx = k & 1;
                        w8[k] = (dz ? fr[0] : 1 - fr[0]) * (dy ? fr[1] : 1 - fr[1]) * (dx ? fr[2] : 1 - fr[2]);
                        ix[k] = (((size_t)(c0[0] + dz) * g + c0[1] + dy) * g + c0[2] + dx) * E; }
                    for (int e = 0; e < E; e++) { float v = 0; for (int k = 0; k < 8; k++) v += w8[k] * h2f(tok[ix[k] + e]); rowf[e] = f2h(v); }
                    for (int e = 0; e < nd + ni; e++) rowf[E + e] = f2h(acch[(size_t)q * (nd + ni) + e]);
                    const dvf_row R = {ci, (int16_t)rr[2 * i], (int16_t)rr[2 * i + 1], lab[i], wg[i]};
                    if (fwrite(&R, sizeof R, 1, fo) != 1 || fwrite(rowf, 2, (size_t)F, fo) != (size_t)F) { fprintf(stderr, "dinovol-probe: write failed\n"); rc = 1; break; }
                    nrows++;
                }
                done += m;
            }
        }
        free(own); free(P); free(lab); free(wg); free(rr); pcrop_free(&C);
        if ((ci + 1) % 10 == 0 || ci + 1 == nc) fprintf(stderr, "dinovol-probe: %d / %d crops, %ld rows, %ld tiles, %.0f s\n", ci + 1, nc, nrows, ntiles, pnow() - t0);
    }
    if (ct) z3_close(ct);
    if (fclose(fo)) rc = 1;
    free(u8); xfree_dev(u8d); free(tok); free(pts); xfree_dev(ptsd); xfree_dev(acc); free(acch); free(rowf); free(names); dv_free(net);
    return rc;
}

/* ---- reading a feature file; the selected feature columns as float ---- */
typedef struct { dvf_head H; char (*names)[DVF_NAME]; long n; dvf_row *R; float *X; int Fs; } dvf;
static int feat_cols(const dvf_head *H, const char *spec, int *cols) {   /* "tok,dense,ink" -> column indices */
    int n = 0; char b[128]; snprintf(b, sizeof b, "%s", spec);
    for (char *t = strtok(b, ","); t; t = strtok(nullptr, ",")) {
        int a = -1, e = -1;
        if (!strcmp(t, "tok")) { a = 0; e = H->E; } else if (!strcmp(t, "dense")) { a = H->E; e = H->E + H->nd; } else if (!strcmp(t, "ink")) { a = H->E + H->nd; e = H->F; }
        else { fprintf(stderr, "dinovol-probe: --feat: tok, dense, ink\n"); return -1; }
        for (int k = a; k < e; k++) cols[n++] = k;
    }
    return n;
}
static int dvf_read(dvf *D, const char *path, const char *spec) {
    memset(D, 0, sizeof *D); FILE *f = fopen(path, "rb"); if (!f) { fprintf(stderr, "dinovol-probe: cannot read %s\n", path); return -1; }
    if (fread(&D->H, sizeof D->H, 1, f) != 1 || memcmp(D->H.magic, "DVPF0001", 8)) { fprintf(stderr, "dinovol-probe: %s: not a feature file\n", path); fclose(f); return -1; }
    D->names = malloc((size_t)D->H.ncrop * DVF_NAME);
    if (fread(D->names, DVF_NAME, (size_t)D->H.ncrop, f) != (size_t)D->H.ncrop) { fclose(f); return -1; }
    const long p0 = ftell(f); fseek(f, 0, SEEK_END); const long rb = (long)sizeof(dvf_row) + 2L * D->H.F; D->n = (ftell(f) - p0) / rb; fseek(f, p0, SEEK_SET);
    int *cols = malloc(sizeof(int) * D->H.F); D->Fs = feat_cols(&D->H, spec, cols); if (D->Fs <= 0) { fclose(f); free(cols); return -1; }
    D->R = malloc(sizeof(dvf_row) * D->n); D->X = malloc(sizeof(float) * D->n * D->Fs); uint16_t *b = malloc(2L * D->H.F);
    for (long i = 0; i < D->n; i++) {
        if (fread(&D->R[i], sizeof(dvf_row), 1, f) != 1 || fread(b, 2, (size_t)D->H.F, f) != (size_t)D->H.F) { fprintf(stderr, "dinovol-probe: %s: short read\n", path); fclose(f); return -1; }
        for (int k = 0; k < D->Fs; k++) D->X[i * D->Fs + k] = h2f(b[cols[k]]);
    }
    free(b); free(cols); fclose(f); return 0;
}
static void dvf_free(dvf *D) { free(D->names); free(D->R); free(D->X); }

/* ---- the weighted logistic probe: standardised x, logit = w . z + b; Newton with ridge lam on w ---- */
typedef struct { const float *X; const dvf_row *R; long a, b; int F; const double *mu, *sd, *w; double *g, *Hm, loss; } nt_job;
static void *nt_main(void *p) {
    nt_job *J = p; const int F = J->F, K = F + 1; double *z = malloc(sizeof(double) * K);
    for (long i = J->a; i < J->b; i++) {
        const double y = J->R[i].lab, wt = J->R[i].wgt; if (!(wt > 0)) continue;
        for (int k = 0; k < F; k++) z[k] = (J->X[i * F + k] - J->mu[k]) / J->sd[k]; z[F] = 1;
        double t = 0; for (int k = 0; k < K; k++) t += J->w[k] * z[k];
        const double pr = 1 / (1 + exp(-t)), r = wt * (pr - y), h = wt * pr * (1 - pr);
        J->loss += wt * (t > 0 ? t + log1p(exp(-t)) - y * t : log1p(exp(t)) - y * t);
        for (int k = 0; k < K; k++) { J->g[k] += r * z[k]; const double hz = h * z[k]; double *Hr = J->Hm + (size_t)k * K; for (int l = 0; l <= k; l++) Hr[l] += hz * z[l]; }
    }
    free(z); return nullptr;
}
static int chol_solve(double *A, double *b, int n) {   /* A (lower) = L L^T in place; b <- A^-1 b */
    for (int j = 0; j < n; j++) { double s = A[(size_t)j * n + j]; for (int k = 0; k < j; k++) s -= A[(size_t)j * n + k] * A[(size_t)j * n + k];
        if (!(s > 0)) return -1; const double d = sqrt(s); A[(size_t)j * n + j] = d;
        for (int i = j + 1; i < n; i++) { double t = A[(size_t)i * n + j]; for (int k = 0; k < j; k++) t -= A[(size_t)i * n + k] * A[(size_t)j * n + k]; A[(size_t)i * n + j] = t / d; } }
    for (int i = 0; i < n; i++) { double t = b[i]; for (int k = 0; k < i; k++) t -= A[(size_t)i * n + k] * b[k]; b[i] = t / A[(size_t)i * n + i]; }
    for (int i = n - 1; i >= 0; i--) { double t = b[i]; for (int k = i + 1; k < n; k++) t -= A[(size_t)k * n + i] * b[k]; b[i] = t / A[(size_t)i * n + i]; }
    return 0;
}
/* weighted ROC AUC of scores s against soft labels: each row splits into an ink part (w y) and a non-ink part (w (1 - y)) */
typedef struct { double s, p, q; } auc_it;
static int auc_cmp(const void *a, const void *b) { const double x = ((const auc_it *)a)->s, y = ((const auc_it *)b)->s; return x < y ? -1 : x > y; }
double dvp_wauc(const double *s, const dvf_row *R, long n) {
    auc_it *v = malloc(sizeof *v * n); long m = 0;
    for (long i = 0; i < n; i++) if (R[i].wgt > 0) v[m++] = (auc_it){s[i], R[i].wgt * R[i].lab, R[i].wgt * (1 - R[i].lab)};
    qsort(v, m, sizeof *v, auc_cmp); double P = 0, Q = 0, A = 0, q = 0;
    for (long i = 0; i < m;) { long j = i; double dp = 0, dq = 0; while (j < m && v[j].s == v[i].s) { dp += v[j].p; dq += v[j].q; j++; }
        A += dp * (q + 0.5 * dq); q += dq; P += dp; Q += dq; i = j; }
    free(v); return P > 0 && Q > 0 ? A / (P * Q) : NAN;
}
int dvp_fit(const float *X, const dvf_row *R, long n, int F, double lam, int iters, int nthreads, double *mu, double *sd, double *w) {
    const int K = F + 1; double W = 0;
    for (int k = 0; k < F; k++) { mu[k] = 0; sd[k] = 0; }
    for (long i = 0; i < n; i++) { if (!(R[i].wgt > 0)) continue; W += R[i].wgt; for (int k = 0; k < F; k++) mu[k] += R[i].wgt * X[i * F + k]; }
    if (!(W > 0)) return -1;
    for (int k = 0; k < F; k++) mu[k] /= W;
    for (long i = 0; i < n; i++) { if (!(R[i].wgt > 0)) continue; for (int k = 0; k < F; k++) { const double d = X[i * F + k] - mu[k]; sd[k] += R[i].wgt * d * d; } }
    for (int k = 0; k < F; k++) { sd[k] = sqrt(sd[k] / W); if (!(sd[k] > 1e-8)) sd[k] = 1; }
    for (int k = 0; k < K; k++) w[k] = 0;
    if (nthreads < 1) nthreads = 1; if (nthreads > 64) nthreads = 64;
    nt_job J[64]; pthread_t th[64]; double *g = malloc(sizeof(double) * K), *Hm = malloc(sizeof(double) * K * K); double prev = 1e300;
    for (int it = 0; it < iters; it++) {
        for (int t = 0; t < nthreads; t++) { J[t] = (nt_job){X, R, n * t / nthreads, n * (t + 1) / nthreads, F, mu, sd, w, calloc(K, sizeof(double)), calloc((size_t)K * K, sizeof(double)), 0};
            pthread_create(&th[t], nullptr, nt_main, &J[t]); }
        memset(g, 0, sizeof(double) * K); memset(Hm, 0, sizeof(double) * K * K); double loss = 0;
        for (int t = 0; t < nthreads; t++) { pthread_join(th[t], nullptr); loss += J[t].loss; for (int k = 0; k < K; k++) g[k] += J[t].g[k];
            for (size_t k = 0; k < (size_t)K * K; k++) Hm[k] += J[t].Hm[k]; free(J[t].g); free(J[t].Hm); }
        loss /= W; for (int k = 0; k < K; k++) g[k] /= W; for (size_t k = 0; k < (size_t)K * K; k++) Hm[k] /= W;
        double reg = 0; for (int k = 0; k < F; k++) { reg += 0.5 * lam * w[k] * w[k]; g[k] += lam * w[k]; Hm[(size_t)k * K + k] += lam; }
        Hm[(size_t)F * K + F] += 1e-9;
        fprintf(stderr, "dinovol-probe: Newton %d: loss %.6f (+ ridge %.6f)\n", it, loss, reg);
        if (chol_solve(Hm, g, K)) { fprintf(stderr, "dinovol-probe: Hessian not positive definite\n"); free(g); free(Hm); return -1; }
        for (int k = 0; k < K; k++) w[k] -= g[k];
        if (fabs(prev - (loss + reg)) < 1e-7) break; prev = loss + reg;
    }
    free(g); free(Hm); return 0;
}
static void dvp_scores(const float *X, long n, int F, const double *mu, const double *sd, const double *w, double *s) {
    for (long i = 0; i < n; i++) { double t = w[F]; for (int k = 0; k < F; k++) t += w[k] * (X[i * F + k] - mu[k]) / sd[k]; s[i] = t; }
}

static int cmd_fit(int argc, char **argv) {
    const char *in = argc > 3 ? argv[3] : nullptr, *out = opt(argc, argv, "--out", nullptr), *spec = opt(argc, argv, "--feat", "tok,dense,ink");
    const double lam = opt_atof("--l2", opt(argc, argv, "--l2", "1e-3")); const int iters = opt_atoi("--iters", opt(argc, argv, "--iters", "12"));
    const int nthreads = ufsm_threads(opt_atoi("--threads", opt(argc, argv, "--threads", "0")), 6);
    if (!in || in[0] == '-' || !out) { fprintf(stderr, "usage: ufsm dinovol-probe fit TRAIN.dvf --out PROBE.txt [--feat tok,dense,ink] [--l2 1e-3] [--iters 12] [--threads 6]\n"); return 2; }
    dvf D; if (dvf_read(&D, in, spec)) return 1;
    const int F = D.Fs; double *mu = malloc(sizeof(double) * F), *sd = malloc(sizeof(double) * F), *w = malloc(sizeof(double) * (F + 1)), *s = malloc(sizeof(double) * D.n);
    double W = 0, Y = 0; for (long i = 0; i < D.n; i++) { W += D.R[i].wgt; Y += D.R[i].wgt * D.R[i].lab; }
    fprintf(stderr, "dinovol-probe fit: %ld rows over %d crops, %d features (%s), ink share %.3f, l2 %g\n", D.n, D.H.ncrop, F, spec, Y / W, lam);
    if (dvp_fit(D.X, D.R, D.n, F, lam, iters, nthreads, mu, sd, w)) return 1;
    dvp_scores(D.X, D.n, F, mu, sd, w, s);
    fprintf(stderr, "dinovol-probe fit: train AUC %.4f\n", dvp_wauc(s, D.R, D.n));
    FILE *f = fopen(out, "w"); if (!f) { fprintf(stderr, "dinovol-probe: cannot write %s\n", out); return 1; }
    fprintf(f, "dvprobe1 %s %d\n", spec, F); for (int k = 0; k < F; k++) fprintf(f, "%.17g %.17g %.17g\n", mu[k], sd[k], w[k]); fprintf(f, "%.17g\n", w[F]);
    const int rc = fclose(f) != 0;
    free(mu); free(sd); free(w); free(s); dvf_free(&D); return rc;
}

static int cmd_apply(int argc, char **argv) {
    const char *in = argc > 3 ? argv[3] : nullptr, *pf = opt(argc, argv, "--probe", nullptr), *out = opt(argc, argv, "--out", nullptr);
    if (!in || in[0] == '-' || !pf || !out) { fprintf(stderr, "usage: ufsm dinovol-probe apply DEV.dvf --probe PROBE.txt --out DIR   (DIR/<crop>.tif, uint8 canvas maps)\n"); return 2; }
    FILE *f = fopen(pf, "r"); char spec[128]; int F = 0;
    if (!f || fscanf(f, "dvprobe1 %127s %d", spec, &F) != 2 || F <= 0) { fprintf(stderr, "dinovol-probe: bad probe %s\n", pf); if (f) fclose(f); return 1; }
    double *mu = malloc(sizeof(double) * F), *sd = malloc(sizeof(double) * F), *w = malloc(sizeof(double) * (F + 1));
    for (int k = 0; k < F; k++) if (fscanf(f, "%lf %lf %lf", &mu[k], &sd[k], &w[k]) != 3) { fclose(f); return 1; }
    if (fscanf(f, "%lf", &w[F]) != 1) { fclose(f); return 1; } fclose(f);
    dvf D; if (dvf_read(&D, in, spec)) return 1;
    if (D.Fs != F) { fprintf(stderr, "dinovol-probe: %s has %d %s features, the probe %d\n", in, D.Fs, spec, F); return 1; }
    double *s = malloc(sizeof(double) * D.n); dvp_scores(D.X, D.n, F, mu, sd, w, s);
    mkdir(out, 0755);
    const int st = D.H.stride, N = 256, G = N / st; double *L = malloc(sizeof(double) * G * G); uint8_t *has = malloc((size_t)G * G), *map = malloc((size_t)N * N);
    int rc = 0; long i = 0;
    for (int c = 0; c < D.H.ncrop && !rc; c++) {
        memset(has, 0, (size_t)G * G);
        while (i < D.n && D.R[i].crop < c) i++;
        for (; i < D.n && D.R[i].crop == c; i++) { const int a = D.R[i].r, b = D.R[i].c; if (a < G && b < G) { L[a * G + b] = s[i]; has[a * G + b] = 1; } }
        int any = 0; for (int k = 0; k < G * G; k++) any |= has[k]; if (!any) continue;
        for (int k = 0; k < G * G; k++) if (!has[k]) {   /* (cells without a point: the nearest one's logit, for the bilinear blend) */
            const int a = k / G, b = k % G; int bd = 1 << 30; for (int q = 0; q < G * G; q++) if (has[q]) { const int d = (q / G - a) * (q / G - a) + (q % G - b) * (q % G - b); if (d < bd) { bd = d; L[k] = L[q]; } } }
        for (int y = 0; y < N; y++) for (int x = 0; x < N; x++) {
            double u = (y - (st - 1) * 0.5) / st, v = (x - (st - 1) * 0.5) / st; if (u < 0) u = 0; if (v < 0) v = 0; if (u > G - 1) u = G - 1; if (v > G - 1) v = G - 1;
            int a = (int)u, b = (int)v; if (a > G - 2) a = G - 2; if (b > G - 2) b = G - 2; const double fu = u - a, fv = v - b;
            const double t = (1 - fu) * ((1 - fv) * L[a * G + b] + fv * L[a * G + b + 1]) + fu * ((1 - fv) * L[(a + 1) * G + b] + fv * L[(a + 1) * G + b + 1]);
            map[(size_t)y * N + x] = (uint8_t)lrint(255 / (1 + exp(-t)));
        }
        char p[1400]; snprintf(p, sizeof p, "%s/%s.tif", out, D.names[c]); if (tiffw_write(p, N, N, 1, map, 1)) { fprintf(stderr, "dinovol-probe: cannot write %s\n", p); rc = 1; }
    }
    fprintf(stderr, "dinovol-probe apply: %ld rows, pooled AUC %.4f -> %s\n", D.n, dvp_wauc(s, D.R, D.n), out);
    free(L); free(has); free(map); free(s); free(mu); free(sd); free(w); dvf_free(&D); return rc;
}

/* stats: the normalisation statistics of every scan a crop list uses, CPU only (so feats does not hold a GPU while it reads them) */
static int cmd_stats(int argc, char **argv) {
    const char *root = opt(argc, argv, "--root", "/vesuvius/ufsm/v2/ct"), *cdir = opt(argc, argv, "--crops", nullptr), *list = opt(argc, argv, "--list", nullptr),
               *cache = opt(argc, argv, "--cache", nullptr);
    const int nthreads = ufsm_threads(opt_atoi("--threads", opt(argc, argv, "--threads", "0")), 6);
    if (!cdir) { fprintf(stderr, "usage: ufsm dinovol-probe stats --crops DIR [--list crops.tsv] [--stats-dir DIR] [--root CTROOT] [--threads 6] [--cache DIR]\n"); return 2; }
    const char *sdir = opt(argc, argv, "--stats-dir", cdir);
    char lp[1400]; if (!list) { snprintf(lp, sizeof lp, "%s/crops.tsv", cdir); list = lp; }
    FILE *lf = fopen(list, "r"); if (!lf) { fprintf(stderr, "dinovol-probe: cannot read %s\n", list); return 1; }
    store_global_init(); store *s = store_open(root); char line[4096], done[64][512]; int nd = 0, rc = 0;
    while (fgets(line, sizeof line, lf) && !rc) {   /* crops.tsv column 7: the CT key */
        if (line[0] == '#') continue; line[strcspn(line, "\n")] = 0; char *f = line; for (int k = 0; k < 6 && f; k++) { f = strchr(f, '\t'); if (f) f++; }
        if (!f || !*f) continue; int seen = 0; for (int k = 0; k < nd; k++) seen |= !strcmp(done[k], f);
        if (seen || nd >= 64) continue; snprintf(done[nd++], 512, "%s", f);
        char ak[600]; snprintf(ak, sizeof ak, "%s/0", f); z3 *ct = s ? z3_open(s, ak, cache) : nullptr; float S[3];
        if (!ct || scan_stats(sdir, f, ct, nthreads, S)) { fprintf(stderr, "dinovol-probe: cannot open / normalise %s %s\n", root, f); rc = 1; }
        if (ct) z3_close(ct);
    }
    fclose(lf); return rc;
}

int cmd_dinovol_probe(int argc, char **argv) {
    const char *m = argc > 2 ? argv[2] : "";
    if (!strcmp(m, "stats")) return cmd_stats(argc, argv);
    if (!strcmp(m, "feats")) return cmd_feats(argc, argv);
    if (!strcmp(m, "fit")) return cmd_fit(argc, argv);
    if (!strcmp(m, "apply")) return cmd_apply(argc, argv);
    fprintf(stderr, "usage: ufsm dinovol-probe feats|fit|apply ...   linear ink probe of frozen dinovol features on ink-devcrops crops\n");
    return 2;
}
