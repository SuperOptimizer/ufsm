/* ufsm dino: our own DINO-style embedding training from scratch (ufsm dino-train), its inference (ufsm dino-embed) and the
   view-pair data tool (ufsm halves). See DESIGN.md "Own dense embedding net (dino-train)". */
#include "dino.h"
#include "dinovol.h"
#include "store.h"
#include "zarr3.h"
#include "z3w.h"
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

static double dn_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static const char *opt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return dflt;
}

/* ---- ufsm halves: two 4x-pooled views of a fine scan from disjoint sub-voxel sets (Neighbor2Neighbor-style noise-independent pair):
   each 4^3 block of the level-0 scan splits by the parity of (dz + dy + dx) into two 32-voxel halves; a = the even half's mean, b = the
   odd half's, full = the 64-voxel mean. Written as uint8 volcomp arrays (lossless) of the box, at the 4x level's grid ---- */
int cmd_dino_halves(int argc, char **argv) {
    long long bv[6];
    if (argc < 5 || !opt(argc, argv, "--box", nullptr) || sscanf(opt(argc, argv, "--box", ""), "%lld,%lld,%lld,%lld,%lld,%lld", &bv[0], &bv[1], &bv[2], &bv[3], &bv[4], &bv[5]) != 6) {
        fprintf(stderr, "usage: ufsm halves <root> <level-0-array-key> <out-dir> --box z,y,x,nz,ny,nx (4x-level voxels) [--threads 8] [--slab 16]\n"
                        "  out-dir/{a,b,full}: uint8 arrays of the box (a / b: the even / odd (dz + dy + dx) halves of every 4^3 block, 32 voxels each;\n"
                        "  full: all 64). Attributes record the box origin. Two views of the same tissue with independent scan noise.\n");
        return 2;
    }
    const char *root = argv[2], *key = argv[3], *out = argv[4];
    const int nthreads = atoi(opt(argc, argv, "--threads", "8")), slab = atoi(opt(argc, argv, "--slab", "16"));
    store *st = store_open(root); z3 *ct = st ? z3_open(st, key, opt(argc, argv, "--cache", nullptr)) : nullptr;
    if (!ct) { fprintf(stderr, "halves: cannot open %s %s: %s\n", root, key, z3_error()); return 1; }
    const int64_t nz = bv[3], ny = bv[4], nx = bv[5];
    uint8_t *A = calloc((size_t)nz * ny * nx, 1), *B = calloc((size_t)nz * ny * nx, 1), *F = calloc((size_t)nz * ny * nx, 1);
    uint8_t *buf = malloc((size_t)slab * 4 * ny * 4 * nx * 4);
    if (!A || !B || !F || !buf) { fprintf(stderr, "halves: out of memory\n"); return 1; }
    const double t0 = dn_now();
    for (int64_t z0 = 0; z0 < nz; z0 += slab) {
        const int sz = (int)(nz - z0 < slab ? nz - z0 : slab);
        const int64_t o[3] = {(bv[0] + z0) * 4, bv[1] * 4, bv[2] * 4}, n[3] = {(int64_t)sz * 4, ny * 4, nx * 4};
        if (z3_read(ct, o, n, buf, nthreads)) { fprintf(stderr, "halves: read failed: %s\n", z3_error()); return 1; }
        for (int z = 0; z < sz; z++)
            for (int64_t y = 0; y < ny; y++)
                for (int64_t x = 0; x < nx; x++) {
                    int sa = 0, sb = 0;
                    for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++) {
                        const uint8_t *r = buf + (((size_t)(z * 4 + a) * ny * 4) + y * 4 + b) * (nx * 4) + x * 4;
                        for (int c = 0; c < 4; c++) { if ((a + b + c) & 1) sb += r[c]; else sa += r[c]; }
                    }
                    const size_t i = ((size_t)(z0 + z) * ny + y) * nx + x;
                    A[i] = (uint8_t)((sa + 16) / 32); B[i] = (uint8_t)((sb + 16) / 32); F[i] = (uint8_t)((sa + sb + 32) / 64);
                }
        fprintf(stderr, "\rhalves: %lld / %lld planes (%.0f s)   ", (long long)(z0 + sz), (long long)nz, dn_now() - t0);
    }
    fprintf(stderr, "\n");
    char attrs[512]; snprintf(attrs, sizeof attrs, "{\"ufsm\":{\"content\":\"4x-pooled sub-voxel halves of %s %s\",\"box_origin_zyx\":[%lld,%lld,%lld]}}", root, key, bv[0], bv[1], bv[2]);
    const char *names[3] = {"a", "b", "full"}; uint8_t *arr[3] = {A, B, F};
    const int64_t shp[3] = {nz, ny, nx}; const int S = 512;
    uint8_t *sh = malloc((size_t)S * S * S);
    for (int k = 0; k < 3; k++) {
        char d[1200]; snprintf(d, sizeof d, "%s/%s", out, names[k]);
        z3w *w = z3w_create(d, shp, S, 0.f, 0, attrs); if (!w) { fprintf(stderr, "halves: %s\n", z3w_error()); return 1; }
        for (int64_t sz = 0; sz * S < nz; sz++) for (int64_t sy = 0; sy * S < ny; sy++) for (int64_t sx = 0; sx * S < nx; sx++) {
            memset(sh, 0, (size_t)S * S * S);
            for (int64_t z = 0; z < S && sz * S + z < nz; z++) for (int64_t y = 0; y < S && sy * S + y < ny; y++) {
                const int64_t w0 = nx - sx * S < S ? nx - sx * S : S;
                memcpy(sh + ((size_t)z * S + y) * S, arr[k] + ((size_t)(sz * S + z) * ny + sy * S + y) * nx + sx * S, (size_t)w0);
            }
            if (z3w_write_shard(w, sz, sy, sx, sh, nthreads)) { fprintf(stderr, "halves: write: %s\n", z3w_error()); return 1; }
        }
        if (z3w_close(w)) { fprintf(stderr, "halves: close: %s\n", z3w_error()); return 1; }
    }
    /* noise check: the a - b difference is pure noise (same tissue); its std / sqrt(2) is the per-half noise */
    double s = 0, s2 = 0, d2 = 0; const size_t nv = (size_t)nz * ny * nx; size_t cnt = 0;
    for (size_t i = 0; i < nv; i++) if (F[i]) { const double d = (double)A[i] - B[i]; s += F[i]; s2 += (double)F[i] * F[i]; d2 += d * d; cnt++; }
    if (cnt) fprintf(stderr, "halves: %zu nonzero voxels, full mean %.2f std %.2f, (a - b) rms %.2f -> per-half noise ~%.2f\n", cnt, s / cnt,
                     sqrt(fmax(s2 / cnt - (s / cnt) * (s / cnt), 0)), sqrt(d2 / cnt), sqrt(d2 / cnt / 2));
    free(A); free(B); free(F); free(buf); free(sh); z3_close(ct); store_close(st);
    return 0;
}

/* =============================================================================================================================== */
/* ufsm dino-train                                                                                                                 */
/* =============================================================================================================================== */
#include "json.h"
#include "unet.h"
#include <unistd.h>
#include <signal.h>

enum { DN_D = 32, DN_MAXSRC = 64, DN_MAXBOX = 256, DN_SLOTS = 6 };
typedef struct {
    char name[64]; z3 *ca, *cb, *codes; int halves;
    int64_t org[3];                              /* added to window corners (ct array coordinates) to address the codes (global grid) */
    int nbox; int64_t box[DN_MAXBOX][6]; int nho; int64_t ho[DN_MAXBOX][6];
    double weight, bvol; float center, scale, pap; int64_t shape[3];
} dn_src;
typedef struct { int P, ngt, nssl, margin; int nsrc; dn_src src[DN_MAXSRC]; double wsum; } dn_data;
typedef struct {   /* one sample: two CT views, the geometry, the points (output-space voxel indices) */
    uint8_t *ca, *cb; int si; int64_t corner[3]; dn_geom g; int halves;
    int ngt, nssl; int *gidx, *gcode, *gxyz, *sidx;
    int ready;
} dn_sample;
typedef struct {
    dn_data *D; dn_sample slot[DN_SLOTS]; pthread_mutex_t mu; pthread_cond_t cv; long produced, consumed; int stop; uint64_t seed; int nthreads;
    pthread_t th[8]; int nworkers; long next_id;
} dn_ring;

static uint64_t rng_next(uint64_t *s) { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; return *s; }
static double rng_u(uint64_t *s) { return (rng_next(s) >> 11) * (1.0 / 9007199254740992.0); }
static int box_hit(const int64_t a[6], const int64_t o[3], int P) {   /* the window [o, o + P)^3 intersects box a */
    for (int d = 0; d < 3; d++) if (o[d] + P <= a[d] || o[d] >= a[d] + a[3 + d]) return 0;
    return 1;
}
/* output index of source voxel s under g: o[d] = flip[d] ? P - 1 - s[perm[d]] : s[perm[d]] */
static void geom_map(const dn_geom *g, int P, const int s[3], int o[3]) { for (int d = 0; d < 3; d++) o[d] = g->flip[d] ? P - 1 - s[g->perm[d]] : s[g->perm[d]]; }

static int sample_fill(dn_data *D, dn_sample *S, uint64_t *rs, int nthreads, uint8_t *codes_buf) {
    const int P = D->P; const size_t p3 = (size_t)P * P * P;
    for (int tries = 0; tries < 200; tries++) {
        double r = rng_u(rs) * D->wsum; int si = 0;
        for (; si < D->nsrc - 1; si++) { r -= D->src[si].weight; if (r < 0) break; }
        dn_src *s = &D->src[si];
        double rb = rng_u(rs) * s->bvol; int bi = 0;
        for (; bi < s->nbox - 1; bi++) { const int64_t *b = s->box[bi]; rb -= (double)b[3] * b[4] * b[5]; if (rb < 0) break; }
        const int64_t *b = s->box[bi]; int64_t o[3];
        for (int d = 0; d < 3; d++) o[d] = b[d] + (b[3 + d] > P ? (int64_t)(rng_u(rs) * (double)(b[3 + d] - P + 1)) : 0);
        int bad = 0; for (int h = 0; h < s->nho && !bad; h++) bad = box_hit(s->ho[h], o, P);
        if (bad) continue;
        const int64_t n[3] = {P, P, P};
        if (z3_read(s->ca, o, n, S->ca, nthreads)) return -1;
        if (s->halves) { if (z3_read(s->cb, o, n, S->cb, nthreads)) return -1; }
        else memcpy(S->cb, S->ca, p3);
        /* enough papyrus? */
        size_t pap = 0; for (size_t i = 0; i < p3; i += 7) pap += S->ca[i] > s->pap;
        if (pap * 7 < p3 / 50) continue;   /* < 2% papyrus: mostly air */
        S->si = si; memcpy(S->corner, o, sizeof o); S->halves = s->halves;
        /* geometry: a random axis permutation and flips */
        int perm[3] = {0, 1, 2}; for (int k = 2; k > 0; k--) { const int j = (int)(rng_next(rs) % (uint64_t)(k + 1)); const int t = perm[k]; perm[k] = perm[j]; perm[j] = t; }
        for (int d = 0; d < 3; d++) { S->g.perm[d] = perm[d]; S->g.flip[d] = (int)(rng_next(rs) & 1); }
        /* GT points: labelled voxels (codes) in the interior, reservoir-sampled */
        S->ngt = 0;
        if (s->codes) {
            const int64_t oc[3] = {o[0] + s->org[0], o[1] + s->org[1], o[2] + s->org[2]};
            if (z3_read(s->codes, oc, n, codes_buf, nthreads)) return -1;
            long seen = 0; const int m = D->margin;
            for (int z = m; z < P - m; z++) for (int y = m; y < P - m; y++) {
                const uint8_t *row = codes_buf + ((size_t)z * P + y) * P;
                for (int x = m; x < P - m; x++) {
                    const int c = row[x]; if (!c || c == 255) continue;
                    long k = seen++;
                    if (k >= D->ngt) { k = (long)(rng_next(rs) % (uint64_t)seen); if (k >= D->ngt) continue; }
                    const int sv[3] = {z, y, x}; int ov[3]; geom_map(&S->g, P, sv, ov);
                    S->gidx[k] = (ov[0] * P + ov[1]) * P + ov[2]; S->gcode[k] = c; S->gxyz[3 * k] = ov[0]; S->gxyz[3 * k + 1] = ov[1]; S->gxyz[3 * k + 2] = ov[2];
                }
            }
            S->ngt = (int)(seen < D->ngt ? seen : D->ngt);
            if (S->ngt < 64) S->ngt = 0;
        }
        /* SSL points: papyrus voxels (view A above the source's papyrus level) */
        long seen = 0; const int m = D->margin;
        for (int z = m; z < P - m; z++) for (int y = m; y < P - m; y++) {
            const uint8_t *row = S->ca + ((size_t)z * P + y) * P;
            for (int x = m; x < P - m; x++) {
                if (row[x] <= s->pap) continue;
                long k = seen++;
                if (k >= D->nssl) { k = (long)(rng_next(rs) % (uint64_t)seen); if (k >= D->nssl) continue; }
                const int sv[3] = {z, y, x}; int ov[3]; geom_map(&S->g, P, sv, ov);
                S->sidx[k] = (ov[0] * P + ov[1]) * P + ov[2];
            }
        }
        S->nssl = (int)(seen < D->nssl ? seen : D->nssl);
        return 0;
    }
    fprintf(stderr, "dino-train: no usable window in 200 tries\n");
    return -1;
}
static void *ring_worker(void *a) {
    dn_ring *R = a; uint64_t rs;
    pthread_mutex_lock(&R->mu); rs = R->seed + 0x9E3779B97F4A7C15ull * (uint64_t)(++R->next_id); pthread_mutex_unlock(&R->mu);
    const size_t p3 = (size_t)R->D->P * R->D->P * R->D->P; uint8_t *cbuf = malloc(p3);
    for (;;) {
        pthread_mutex_lock(&R->mu);
        int k = -1;
        while (!R->stop) { for (int i = 0; i < DN_SLOTS; i++) if (R->slot[i].ready == 0) { k = i; break; } if (k >= 0) break; pthread_cond_wait(&R->cv, &R->mu); }
        if (R->stop) { pthread_mutex_unlock(&R->mu); break; }
        R->slot[k].ready = 1;   /* being filled */
        pthread_mutex_unlock(&R->mu);
        const int rc = sample_fill(R->D, &R->slot[k], &rs, R->nthreads, cbuf);
        pthread_mutex_lock(&R->mu); R->slot[k].ready = rc ? -1 : 2; pthread_cond_broadcast(&R->cv); pthread_mutex_unlock(&R->mu);
        if (rc) break;
    }
    free(cbuf); return nullptr;
}
static dn_sample *ring_get(dn_ring *R) {
    pthread_mutex_lock(&R->mu);
    for (;;) {
        for (int i = 0; i < DN_SLOTS; i++) {
            if (R->slot[i].ready == -1) { pthread_mutex_unlock(&R->mu); return nullptr; }
            if (R->slot[i].ready == 2) { R->slot[i].ready = 3; pthread_mutex_unlock(&R->mu); return &R->slot[i]; }
        }
        pthread_cond_wait(&R->cv, &R->mu);
    }
}
static void ring_put(dn_ring *R, dn_sample *S) { pthread_mutex_lock(&R->mu); S->ready = 0; pthread_cond_broadcast(&R->cv); pthread_mutex_unlock(&R->mu); }

static int parse_box(const json *a, int64_t b[6]) {
    if (!a || a->type != J_ARR || a->n != 6) return -1;
    for (int d = 0; d < 6; d++) b[d] = (int64_t)json_num(json_at(a, d), 0);
    return 0;
}
/* sources JSON: {"sources":[{"name", "root", "ct" (array key at the training grid), "ct_b" (optional: a second, noise-independent view of the same
   grid, e.g. ufsm halves a / b), "cache", "codes": {"root", "key"} (winding codes on the same grid; optional), "origin": [z,y,x] (ct coords + origin =
   codes coords; default 0), "boxes": [[z,y,x,nz,ny,nx]...] (ct coords; windows lie inside), "holdout": [[...]] (windows never touch), "weight",
   "norm": [center, scale] (default: robust statistics of a box sample), "papyrus": raw CT above which a voxel counts as papyrus (default center)}]} */
static int load_sources(const char *path, dn_data *D, int nthreads) {
    FILE *f = fopen(path, "rb"); if (!f) { fprintf(stderr, "dino-train: cannot read %s\n", path); return -1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET); char *t = malloc(n + 1); if (fread(t, 1, n, f) != (size_t)n) { fclose(f); return -1; } fclose(f); t[n] = 0;
    json *j = json_parse(t, n); free(t); if (!j) { fprintf(stderr, "dino-train: %s: bad JSON\n", path); return -1; }
    const json *arr = json_get(j, "sources"); if (!arr || arr->type != J_ARR) { fprintf(stderr, "dino-train: no sources\n"); return -1; }
    for (size_t i = 0; i < arr->n && D->nsrc < DN_MAXSRC; i++) {
        const json *s = json_at(arr, i); dn_src *S = &D->src[D->nsrc]; memset(S, 0, sizeof *S);
        snprintf(S->name, sizeof S->name, "%s", json_str(json_get(s, "name"), "?"));
        const char *root = json_str(json_get(s, "root"), nullptr), *ct = json_str(json_get(s, "ct"), nullptr), *ctb = json_str(json_get(s, "ct_b"), nullptr);
        const char *cache = json_str(json_get(s, "cache"), nullptr);
        if (!root || !ct) { fprintf(stderr, "dino-train: source %zu: root / ct missing\n", i); return -1; }
        store *st = store_open(root);
        S->ca = st ? z3_open(st, ct, cache) : nullptr; if (!S->ca) { fprintf(stderr, "dino-train: %s: cannot open %s %s: %s\n", S->name, root, ct, z3_error()); return -1; }
        if (ctb) { S->cb = z3_open(st, ctb, cache); if (!S->cb) { fprintf(stderr, "dino-train: %s: cannot open %s\n", S->name, ctb); return -1; } S->halves = 1; }
        memcpy(S->shape, z3_meta_of(S->ca)->shape, sizeof S->shape);
        const json *cj = json_get(s, "codes");
        if (cj) {
            store *cs = store_open(json_str(json_get(cj, "root"), "")); S->codes = cs ? z3_open(cs, json_str(json_get(cj, "key"), ""), nullptr) : nullptr;
            if (!S->codes) { fprintf(stderr, "dino-train: %s: cannot open the codes\n", S->name); return -1; }
        }
        const json *org = json_get(s, "origin"); for (int d = 0; d < 3; d++) S->org[d] = org ? (int64_t)json_num(json_at(org, d), 0) : 0;
        const json *bx = json_get(s, "boxes");
        if (bx && bx->type == J_ARR) for (size_t k = 0; k < bx->n && S->nbox < DN_MAXBOX; k++) if (!parse_box(json_at(bx, k), S->box[S->nbox])) S->nbox++;
        if (!S->nbox) { for (int d = 0; d < 3; d++) { S->box[0][d] = 0; S->box[0][3 + d] = S->shape[d]; } S->nbox = 1; }
        for (int k = 0; k < S->nbox; k++) for (int d = 0; d < 3; d++) if (S->box[k][3 + d] < D->P) { fprintf(stderr, "dino-train: %s box %d is smaller than the window\n", S->name, k); return -1; }
        const json *ho = json_get(s, "holdout");
        if (ho && ho->type == J_ARR) for (size_t k = 0; k < ho->n && S->nho < DN_MAXBOX; k++) if (!parse_box(json_at(ho, k), S->ho[S->nho])) S->nho++;
        S->weight = json_num(json_get(s, "weight"), 1.0);
        S->bvol = 0; for (int k = 0; k < S->nbox; k++) S->bvol += (double)S->box[k][3] * S->box[k][4] * S->box[k][5];
        const json *nm = json_get(s, "norm");
        if (nm) { S->center = (float)json_num(json_at(nm, 0), 100); S->scale = (float)json_num(json_at(nm, 1), 50); }
        else {   /* robust statistics of nonzero voxels of a few 128^3 blocks of the boxes */
            int64_t hist[256] = {0}; uint8_t *b = malloc(128 * 128 * 128); uint64_t rs = 12345 + i;
            for (int r = 0; r < 8; r++) {
                const int64_t *B = S->box[(int)(rng_next(&rs) % (uint64_t)S->nbox)]; int64_t o[3];
                for (int d = 0; d < 3; d++) o[d] = B[d] + (int64_t)(rng_u(&rs) * (double)(B[3 + d] - 128 > 0 ? B[3 + d] - 128 : 0));
                const int64_t nn[3] = {128, 128, 128};
                if (!z3_read(S->ca, o, nn, b, nthreads)) for (int v = 0; v < 128 * 128 * 128; v++) hist[b[v]]++;
            }
            free(b); hist[0] = 0; int64_t tot = 0; for (int v = 1; v < 256; v++) tot += hist[v];
            int64_t acc = 0; int med = 128, q25 = 64, q75 = 192;
            for (int v = 1; v < 256; v++) { const int64_t prev = acc; acc += hist[v];
                if (prev < tot / 4 && acc >= tot / 4) q25 = v;
                if (prev < tot / 2 && acc >= tot / 2) med = v;
                if (prev < tot * 3 / 4 && acc >= tot * 3 / 4) q75 = v;
            }
            S->center = (float)med; S->scale = (float)(q75 - q25) / 1.349f; if (S->scale < 1) S->scale = 1;
        }
        S->pap = (float)json_num(json_get(s, "papyrus"), S->center);
        D->wsum += S->weight;
        fprintf(stderr, "dino-train: source %s: %d boxes (%.2f Gvox), %d holdouts, %s, codes %s, weight %g, norm %.1f / %.1f, papyrus > %.0f\n", S->name, S->nbox, S->bvol / 1e9,
                S->nho, S->halves ? "two views (ct / ct_b)" : "one CT (views by noise)", S->codes ? "yes" : "no", S->weight, S->center, S->scale, S->pap);
        D->nsrc++;
    }
    json_free(j);
    return D->nsrc ? 0 : -1;
}

/* precision / storage settings of the production trainer (tests/prof_infer.c with ~/ufsm-data/pi_prod.sh's environment) unless --fp32 1 */
static void dn_precision(int fp32) {
    if (fp32) { nn_set_tf32(0); return; }
    setenv("UFSM_ACT_MX4", "1", 0); setenv("UFSM_XIN_PREC", "8", 0); setenv("UFSM_GRAD_MX8", "1", 0); setenv("UFSM_GRAD_MX4", "1", 0);
    setenv("UFSM_F4_WGRAD", "1", 0); setenv("UFSM_DOWN_NORM", "1", 0);
}
static void dn_precision_after_init(int fp32) {
    if (fp32) return;
    nn_set_tf32(1); nn_set_f16(1);
    nn_set_prec_policy(getenv("UFSM_PREC_POLICY") ? getenv("UFSM_PREC_POLICY") : "all=fp4:fp4:fp4,enc0.c1=fp16");
    nn_set_gn_stored(1); nn_set_sr(1);
}

/* the EMA network over a box (CT array coords) -> unit int8 embeddings [nz][ny][nx][D] (host), tiles of P with halo h */
static int8_t *dn_embed_box(unet *u, z3 *ct, const int64_t box[6], int P, int h, float center, float scale, int nthreads, int ema) {
    const int D = unet_cfg_of(u)->cout, in = P - 2 * h; const size_t p3 = (size_t)P * P * P;
    int8_t *out = calloc((size_t)box[3] * box[4] * box[5] * D, 1);
    uint8_t *cth = malloc(p3), *ctd = nn_malloc(p3); float *x = nn_malloc(p3 * 4);
    int8_t *td = nn_malloc((size_t)in * in * in * D), *th = malloc((size_t)in * in * in * D);
    if (!out || !cth || !ctd || !x || !td || !th) { fprintf(stderr, "dino: embed buffers\n"); return nullptr; }
    if (ema) unet_use_ema(u, 1);
    const dn_geom g0 = {{0, 1, 2}, {0, 0, 0}}; const shape5 xs = {1, 1, P, P, P};
    for (int64_t tz = 0; tz < box[3]; tz += in) for (int64_t ty = 0; ty < box[4]; ty += in) for (int64_t tx = 0; tx < box[5]; tx += in) {
        const int64_t o[3] = {box[0] + tz - h, box[1] + ty - h, box[2] + tx - h}, nn[3] = {P, P, P};
        if (z3_read(ct, o, nn, cth, nthreads)) { fprintf(stderr, "dino: embed read: %s\n", z3_error()); return nullptr; }
        nn_h2d(ctd, cth, p3);
        dn_input(ctd, x, P, g0, center, scale, 1.f, 0.f, 0.f, 0, 0, 0.f, 0, 0.f);
        const void *lg = unet_forward(u, x, xs, 0);
        const int nz = (int)(box[3] - tz < in ? box[3] - tz : in), ny = (int)(box[4] - ty < in ? box[4] - ty : in), nx = (int)(box[5] - tx < in ? box[5] - tx : in);
        dn_tile_out(lg, unet_logits_h16(u), P, D, ctd, h, h + nz, h, h + ny, h, h + nx, td, nz, ny, nx, 0, 0, 0);
        nn_d2h(th, td, (size_t)nz * ny * nx * D);
        for (int z = 0; z < nz; z++) for (int y = 0; y < ny; y++)
            memcpy(out + ((((size_t)(tz + z) * box[4]) + ty + y) * box[5] + tx) * D, th + (((size_t)z * ny) + y) * nx * D, (size_t)nx * D);
    }
    if (ema) unet_use_ema(u, 0);
    free(cth); free(th); nn_free(ctd); nn_free(x); nn_free(td);
    return out;
}

static volatile sig_atomic_t g_dn_stop = 0;
static void dn_sig(int s) { (void)s; g_dn_stop = 1; }

int cmd_dino_train(int argc, char **argv) {
    const char *srcf = opt(argc, argv, "--sources", nullptr), *outd = opt(argc, argv, "--out", nullptr);
    if (!srcf || !outd) {
        fprintf(stderr, "usage: ufsm dino-train --sources S.json --out DIR [--P 256] [--widths 16,32,64,96,128,192] [--steps 20000] [--lr 1e-3] [--muon-lr 0.02]\n"
                        "         [--wd 0.01] [--ema 0.996] [--warmup 500] [--gt-points 2048] [--ssl-points 2048] [--tau 0.1] [--hard-w 4] [--hard-r 48]\n"
                        "         [--ssl 0.3 (weight; 0 = GT contrastive only)] [--protos 1024] [--ts 0.1] [--tt 0.04] [--center-m 0.9]\n"
                        "         [--noise 0.15] [--gain-jit 0.15] [--mask-frac 0] [--mask-cube 16] [--seed 1] [--workers 4] [--threads 4]\n"
                        "         [--eval SURF-LIST --eval-root R --eval-key K --eval-box z,y,x,nz,ny,nx --eval-every 2000 --eval-codes ROOT KEY --eval-norm c,s]\n"
                        "         [--log-every 20] [--save-every 2000] [--resume CKPT] [--fp32 1] [--gpu 0]\n"
                        "  Our dense embedding net from scratch (src/dino.h): the U-Net with D = 32 unit embedding channels; GT sheet-identity SupCon on\n"
                        "  labelled voxels (primary) + DINO prototype cross-view term on papyrus voxels (EMA teacher on the other view). STOP file in --out stops.\n");
        return 2;
    }
    const int P = atoi(opt(argc, argv, "--P", "256")), steps = atoi(opt(argc, argv, "--steps", "20000")), fp32 = atoi(opt(argc, argv, "--fp32", "0"));
    const float lr0 = (float)atof(opt(argc, argv, "--lr", "1e-3")), mlr0 = (float)atof(opt(argc, argv, "--muon-lr", "0.02")), wd = (float)atof(opt(argc, argv, "--wd", "0.01"));
    const float ema = (float)atof(opt(argc, argv, "--ema", "0.996")), clip = (float)atof(opt(argc, argv, "--clip", "5"));
    const int warm = atoi(opt(argc, argv, "--warmup", "500"));
    const float tau = (float)atof(opt(argc, argv, "--tau", "0.1")), hw = (float)atof(opt(argc, argv, "--hard-w", "4")), hr = (float)atof(opt(argc, argv, "--hard-r", "48"));
    const float wssl = (float)atof(opt(argc, argv, "--ssl", "0.3")), ts = (float)atof(opt(argc, argv, "--ts", "0.1")), tt = (float)atof(opt(argc, argv, "--tt", "0.04"));
    const float cm = (float)atof(opt(argc, argv, "--center-m", "0.9")), noise = (float)atof(opt(argc, argv, "--noise", "0.15")), gj = (float)atof(opt(argc, argv, "--gain-jit", "0.15"));
    const float mfrac = (float)atof(opt(argc, argv, "--mask-frac", "0")); const int mcube = atoi(opt(argc, argv, "--mask-cube", "16"));
    const int K = atoi(opt(argc, argv, "--protos", "1024")), log_every = atoi(opt(argc, argv, "--log-every", "20")), save_every = atoi(opt(argc, argv, "--save-every", "2000"));
    const int eval_every = atoi(opt(argc, argv, "--eval-every", "2000")), nthreads = atoi(opt(argc, argv, "--threads", "4")), nwork = atoi(opt(argc, argv, "--workers", "4"));
    const uint64_t seed = strtoull(opt(argc, argv, "--seed", "1"), nullptr, 10);
    dn_data *Dd = calloc(1, sizeof *Dd); Dd->P = P; Dd->ngt = atoi(opt(argc, argv, "--gt-points", "2048")); Dd->nssl = atoi(opt(argc, argv, "--ssl-points", "2048")); Dd->margin = 8;
    if (mkdir(outd, 0755) && errno != EEXIST) { fprintf(stderr, "dino-train: cannot create %s\n", outd); return 1; }
    dn_precision(fp32);
    if (nn_init(atoi(opt(argc, argv, "--gpu", "0")))) return 1;
    dn_precision_after_init(fp32);
    if (load_sources(srcf, Dd, nthreads)) return 1;
    unet_cfg cfg = {0}; cfg.cin = 1; cfg.cout = DN_D; cfg.G = 8; cfg.down_norm = fp32 ? 0 : 1;
    { char *t = strdup(opt(argc, argv, "--widths", "16,32,64,96,128,192")); for (char *q = strtok(t, ","); q && cfg.nlev < UNET_MAXLEV; q = strtok(nullptr, ",")) cfg.widths[cfg.nlev++] = atoi(q); free(t); }
    if (P % (1 << (cfg.nlev - 1))) { fprintf(stderr, "dino-train: P must be a multiple of %d\n", 1 << (cfg.nlev - 1)); return 2; }
    unet *u = unet_create(&cfg); unet_init(u, seed);
    if (!fp32) unet_set_fp16_logits(u, 1);
    int step0 = 0;
    const char *resume = opt(argc, argv, "--resume", nullptr);
    if (resume) { step0 = unet_load(u, resume); if (step0 < 0) { fprintf(stderr, "dino-train: cannot resume %s\n", resume); return 1; } }
    /* prototypes */
    const size_t kd = (size_t)K * DN_D;
    float *W = nn_malloc(kd * 4), *Wt = nn_malloc(kd * 4), *gW = nn_malloc(kd * 4), *Wm = nn_malloc(kd * 4), *Wv = nn_malloc(kd * 4), *cen = nn_malloc((size_t)K * 4), *tmean = nn_malloc((size_t)K * 4);
    {
        float *h = malloc(kd * 4); uint64_t rs = seed * 7919 + 3;
        for (size_t i = 0; i < kd; i++) { const double u1 = rng_u(&rs) + 1e-12, u2 = rng_u(&rs); h[i] = (float)(sqrt(-2 * log(u1)) * cos(2 * M_PI * u2)); }
        char pp[1200]; snprintf(pp, sizeof pp, "%s.protos", resume ? resume : "");
        FILE *pf = resume ? fopen(pp, "rb") : nullptr;
        if (pf) { if (fread(h, 4, kd, pf) != kd) fprintf(stderr, "dino-train: short %s\n", pp); fclose(pf); }
        nn_h2d(W, h, kd * 4); nn_h2d(Wt, h, kd * 4); free(h);
        nn_zero(Wm, kd * 4); nn_zero(Wv, kd * 4); nn_zero(cen, (size_t)K * 4);
    }
    /* device buffers */
    const size_t p3 = (size_t)P * P * P; const int NG = Dd->ngt, NS = Dd->nssl;
    uint8_t *cta = nn_malloc(p3), *ctb = nn_malloc(p3); float *x = nn_malloc(p3 * 4), *gl = nn_malloc(p3 * DN_D * 4);
    int *d_gidx = nn_malloc((size_t)NG * 4), *d_gcode = nn_malloc((size_t)NG * 4), *d_gxyz = nn_malloc((size_t)NG * 12), *d_sidx = nn_malloc((size_t)NS * 4);
    float *es_g = nn_malloc((size_t)NG * DN_D * 4), *et_g = nn_malloc((size_t)NG * DN_D * 4), *ns_g = nn_malloc((size_t)NG * 4), *nt_g = nn_malloc((size_t)NG * 4), *ga_g = nn_malloc((size_t)NG * DN_D * 4);
    float *es_s = nn_malloc((size_t)NS * DN_D * 4), *et_s = nn_malloc((size_t)NS * DN_D * 4), *ns_s = nn_malloc((size_t)NS * 4), *nt_s = nn_malloc((size_t)NS * 4), *ga_s = nn_malloc((size_t)NS * DN_D * 4);
    float *scr_sc = nn_malloc(((size_t)2 * NG + 16) * 4), *scr_dn = nn_malloc(dn_dino_scratch(NS, K)), *outs = nn_malloc(64 * 4);
    if (!gl || !x || !scr_dn) { fprintf(stderr, "dino-train: device allocation failed\n"); return 1; }
    /* data ring */
    dn_ring *R = calloc(1, sizeof *R); R->D = Dd; R->seed = seed; R->nthreads = nthreads; pthread_mutex_init(&R->mu, nullptr); pthread_cond_init(&R->cv, nullptr);
    for (int i = 0; i < DN_SLOTS; i++) {
        dn_sample *S = &R->slot[i]; S->ca = malloc(p3); S->cb = malloc(p3);
        S->gidx = malloc((size_t)NG * 4); S->gcode = malloc((size_t)NG * 4); S->gxyz = malloc((size_t)NG * 12); S->sidx = malloc((size_t)NS * 4);
    }
    R->nworkers = nwork > 8 ? 8 : nwork;
    for (int i = 0; i < R->nworkers; i++) pthread_create(&R->th[i], nullptr, ring_worker, R);
    signal(SIGTERM, dn_sig); signal(SIGINT, dn_sig);
    char logp[1200]; snprintf(logp, sizeof logp, "%s/train.log", outd); FILE *lf = fopen(logp, "a");
    fprintf(stderr, "dino-train: P %d, widths", P); for (int l = 0; l < cfg.nlev; l++) fprintf(stderr, " %d", cfg.widths[l]);
    fprintf(stderr, ", %zu params, D %d, GT %d + SSL %d points (weight %g), %d prototypes, %s\n", unet_nparams(u), DN_D, NG, NS, wssl, K, fp32 ? "fp32" : "fp4 / MX storage");
    const shape5 xs = {1, 1, P, P, P};
    double t_last = dn_now(), sums[12] = {0}; int nsum = 0; uint64_t arng = seed * 1000003 + 17;
    const char *stopf_name = "STOP"; char stopf[1200]; snprintf(stopf, sizeof stopf, "%s/%s", outd, stopf_name);
    float *hout = malloc(64 * 4);
    for (int step = step0 + 1; step <= steps && !g_dn_stop; step++) {
        dn_sample *S = ring_get(R); if (!S) { fprintf(stderr, "dino-train: data loader failed\n"); return 1; }
        const dn_src *src = &Dd->src[S->si];
        nn_h2d(cta, S->ca, p3); nn_h2d(ctb, S->cb, p3);
        if (S->ngt) { nn_h2d(d_gidx, S->gidx, (size_t)S->ngt * 4); nn_h2d(d_gcode, S->gcode, (size_t)S->ngt * 4); nn_h2d(d_gxyz, S->gxyz, (size_t)S->ngt * 12); }
        if (S->nssl) nn_h2d(d_sidx, S->sidx, (size_t)S->nssl * 4);
        const int ngt = S->ngt, nssl = S->nssl;
        ring_put(R, S);   /* (its host buffers are copied) */
        const uint8_t *cs = step & 1 ? ctb : cta, *ctt = step & 1 ? cta : ctb;   /* student / teacher views alternate */
        const float ga1 = 1.f + gj * (2.f * (float)rng_u(&arng) - 1.f), gb1 = gj * (2.f * (float)rng_u(&arng) - 1.f);
        const float ga2 = 1.f + gj * (2.f * (float)rng_u(&arng) - 1.f), gb2 = gj * (2.f * (float)rng_u(&arng) - 1.f);
        const float sg1 = noise * (float)rng_u(&arng), sg2 = noise * (float)rng_u(&arng);
        /* teacher (EMA weights), forward only */
        dn_input(ctt, x, P, S->g, src->center, src->scale, ga2, gb2, sg2, (uint32_t)(step * 2 + 1), 0, 0.f, 0, 0.f);
        unet_use_ema(u, 1);
        const void *lg = unet_forward(u, x, xs, 0); const int h16 = unet_logits_h16(u);
        if (ngt) dn_gather(lg, h16, p3, DN_D, d_gidx, ngt, et_g, nt_g);
        if (nssl) dn_gather(lg, h16, p3, DN_D, d_sidx, nssl, et_s, nt_s);
        unet_use_ema(u, 0);
        /* student */
        dn_input(cs, x, P, S->g, src->center, src->scale, ga1, gb1, sg1, (uint32_t)(step * 2), mcube, mfrac, (uint32_t)step, 0.f);
        lg = unet_forward(u, x, xs, 1);
        if (ngt) dn_gather(lg, h16, p3, DN_D, d_gidx, ngt, es_g, ns_g);
        if (nssl) dn_gather(lg, h16, p3, DN_D, d_sidx, nssl, es_s, ns_s);
        nn_zero(gl, p3 * DN_D * 4);
        nn_zero(outs, 64 * 4);
        if (ngt) {
            dn_supcon(es_g, d_gcode, d_gxyz, ngt, et_g, d_gcode, d_gxyz, ngt, tau, hw, hr, ga_g, outs, scr_sc);
            dn_scatter_grad(es_g, ga_g, ns_g, d_gidx, ngt, DN_D, p3, 1.f, gl);
            dn_emb_std(es_g, ngt, DN_D, outs + 8);
        }
        if (nssl && wssl > 0) {
            dn_dino(es_s, et_s, nssl, W, Wt, cen, K, DN_D, ts, tt, ga_s, gW, tmean, outs + 4, scr_dn);
            dn_scatter_grad(es_s, ga_s, ns_s, d_sidx, nssl, DN_D, p3, wssl, gl);
            dn_axpby(gW, gW, kd, 0.f, wssl);
        }
        if (nssl) dn_emb_std(es_s, nssl, DN_D, outs + 9);
        unet_zero_grad(u);
        unet_backward(u, gl);
        const double gn = unet_grad_norm(u);
        if (clip > 0) unet_clip_grad(u, clip);
        const float sched = step <= warm ? (float)step / warm : 0.5f * (1.f + cosf((float)M_PI * (float)(step - warm) / (float)(steps - warm > 1 ? steps - warm : 1)));
        unet_muon(u, mlr0 * sched, 0.95f, lr0 * sched, 0.9f, 0.999f, 1e-8f, wd, step);
        unet_ema(u, step < 100 ? 0.9f : ema);
        if (nssl && wssl > 0) {
            dn_adam(W, gW, Wm, Wv, kd, lr0 * sched, 0.9f, 0.999f, 1e-8f, 0.f, step);
            dn_ema(Wt, W, kd, step < 100 ? 0.9f : ema);
            dn_axpby(cen, tmean, (size_t)K, cm, 1.f - cm);
        }
        nn_d2h(hout, outs, 64 * 4);
        /* 0 supcon loss, 1 pos cos, 2 neg cos, 3 anchors, 4 dino loss, 5 batch-mean teacher entropy, 6 mean teacher entropy, 8 std gt, 9 std ssl */
        const double v[12] = {hout[0], hout[1], hout[2], hout[3], hout[4], hout[5], hout[6], hout[8], hout[9], gn, (double)ngt, (double)nssl};
        for (int k = 0; k < 12; k++) sums[k] += v[k];
        nsum++;
        if (!isfinite(hout[0]) || !isfinite(hout[4]) || !isfinite(gn)) { fprintf(stderr, "dino-train: non-finite loss / gradient at step %d\n", step); return 1; }
        if (step % log_every == 0) {
            const double now = dn_now(), dt = (now - t_last) / nsum;
            char line[512];
            snprintf(line, sizeof line, "step %d gt %.4f pos %.3f neg %.3f anchors %.0f | ssl %.4f H(mean) %.2f H %.2f | std gt %.4f ssl %.4f | gnorm %.3f | lr %.2e | %.0f ms/step %.1f Mvox/s",
                     step, sums[0] / nsum, sums[1] / nsum, sums[2] / nsum, sums[3] / nsum, sums[4] / nsum, sums[5] / nsum, sums[6] / nsum, sums[7] / nsum, sums[8] / nsum,
                     sums[9] / nsum, lr0 * sched, dt * 1e3, (double)p3 / dt / 1e6);
            fprintf(stderr, "%s\n", line); if (lf) { fprintf(lf, "%s\n", line); fflush(lf); }
            memset(sums, 0, sizeof sums); nsum = 0; t_last = dn_now();
            if (access(stopf, F_OK) == 0) { fprintf(stderr, "dino-train: STOP file\n"); g_dn_stop = 1; }
        }
        if (step % save_every == 0 || step == steps || g_dn_stop) {
            char cp[1200], ex[256]; snprintf(cp, sizeof cp, "%s/step%06d.ckpt", outd, step);
            snprintf(ex, sizeof ex, "\"task\":\"dino\",\"embed_dim\":%d,\"protos\":%d", DN_D, K);
            if (unet_save(u, cp, step, ex)) fprintf(stderr, "dino-train: cannot save %s\n", cp);
            char pp[1300]; snprintf(pp, sizeof pp, "%s.protos", cp); float *h = malloc(kd * 4); nn_d2h(h, W, kd * 4);
            FILE *pf = fopen(pp, "wb"); if (pf) { fwrite(h, 4, kd, pf); fclose(pf); } free(h);
            char lp[1200]; snprintf(lp, sizeof lp, "%s/last.ckpt", outd); unlink(lp); if (symlink(cp + strlen(outd) + 1, lp)) {}
        }
        if (opt(argc, argv, "--eval", nullptr) && (step % eval_every == 0 || step == steps)) {
            /* the EMA net over the eval box -> a dinovol-format group, then dinovol-eval against the surfaces (and codes) */
            long long eb[6]; float ec = 0, es = 0;
            if (sscanf(opt(argc, argv, "--eval-box", ""), "%lld,%lld,%lld,%lld,%lld,%lld", &eb[0], &eb[1], &eb[2], &eb[3], &eb[4], &eb[5]) == 6 &&
                sscanf(opt(argc, argv, "--eval-norm", ""), "%f,%f", &ec, &es) == 2) {
                const double te = dn_now();
                store *est = store_open(opt(argc, argv, "--eval-root", "")); z3 *ez = est ? z3_open(est, opt(argc, argv, "--eval-key", ""), opt(argc, argv, "--eval-cache", nullptr)) : nullptr;
                if (!ez) fprintf(stderr, "dino-train: eval CT: %s\n", z3_error());
                else {
                    const int64_t box[6] = {eb[0], eb[1], eb[2], eb[3], eb[4], eb[5]};
                    int8_t *emb = dn_embed_box(u, ez, box, P, 32, ec, es, nthreads, 1);
                    char gdir[1200]; snprintf(gdir, sizeof gdir, "%s/eval-emb", outd);
                    if (emb && !dv_write_dense_box(gdir, z3_meta_of(ez)->shape, box, DN_D, emb)) {
                        const double te2 = dn_now();
                        /* dinovol-eval argv: group, surfaces from the list file, options */
                        char *av[96]; int ac = 0; av[ac++] = "ufsm"; av[ac++] = "dinovol-eval"; av[ac++] = gdir;
                        FILE *sl = fopen(opt(argc, argv, "--eval", ""), "r"); char lines[32][1024]; int nl = 0;
                        while (sl && nl < 32 && fgets(lines[nl], sizeof lines[nl], sl)) { lines[nl][strcspn(lines[nl], "\n")] = 0; if (lines[nl][0]) av[ac++] = lines[nl++]; }
                        if (sl) fclose(sl);
                        av[ac++] = "--n-ref"; av[ac++] = (char *)opt(argc, argv, "--eval-refs", "300");
                        if (opt(argc, argv, "--eval-codes", nullptr)) { int ci = 0; for (int k = 2; k + 2 < argc; k++) if (!strcmp(argv[k], "--eval-codes")) ci = k;
                            av[ac++] = "--codes"; av[ac++] = argv[ci + 1]; av[ac++] = argv[ci + 2]; av[ac++] = "--codes-radius"; av[ac++] = "4"; }
                        av[ac] = nullptr;
                        fflush(stdout); fprintf(stderr, "dino-train: eval at step %d (embed %.1f s)\n", step, te2 - te);
                        if (lf) { fprintf(lf, "eval step %d\n", step); fflush(lf); }
                        int fd = dup(1); if (lf) dup2(fileno(lf), 1);
                        cmd_dinovol_eval(ac, av);
                        fflush(stdout); dup2(fd, 1); close(fd);
                        fprintf(stderr, "dino-train: eval done (%.1f s)\n", dn_now() - te2);
                    }
                    free(emb); z3_close(ez);
                }
                store_close(est);
                t_last = dn_now(); memset(sums, 0, sizeof sums); nsum = 0;
            }
        }
    }
    pthread_mutex_lock(&R->mu); R->stop = 1; pthread_cond_broadcast(&R->cv); pthread_mutex_unlock(&R->mu);
    for (int i = 0; i < R->nworkers; i++) pthread_join(R->th[i], nullptr);
    if (lf) fclose(lf);
    return 0;
}

/* ufsm dino-embed: a dino-train checkpoint over a box -> the dinovol-format group (dense, unit int8, step 1) */
int cmd_dino_embed(int argc, char **argv) {
    long long b[6]; float c = 0, s = 0;
    if (argc < 6 || sscanf(opt(argc, argv, "--box", ""), "%lld,%lld,%lld,%lld,%lld,%lld", &b[0], &b[1], &b[2], &b[3], &b[4], &b[5]) != 6 || sscanf(opt(argc, argv, "--norm", ""), "%f,%f", &c, &s) != 2) {
        fprintf(stderr, "usage: ufsm dino-embed <ckpt> <ct-root> <ct-key> <out-group> --box z,y,x,nz,ny,nx --norm center,scale [--P 256] [--halo 32] [--ema 1] [--cache DIR] [--fp32 1]\n"); return 2; }
    const int P = atoi(opt(argc, argv, "--P", "256")), fp32 = atoi(opt(argc, argv, "--fp32", "0"));
    dn_precision(fp32);
    if (nn_init(atoi(opt(argc, argv, "--gpu", "0")))) return 1;
    dn_precision_after_init(fp32);
    unet_cfg cfg; int step = 0;
    if (unet_peek(argv[2], &cfg, &step)) { fprintf(stderr, "dino-embed: cannot read %s\n", argv[2]); return 1; }
    unet *u = unet_create(&cfg); if (unet_load(u, argv[2]) < 0) return 1;
    if (!fp32) unet_set_fp16_logits(u, 1);
    store *st = store_open(argv[3]); z3 *ct = st ? z3_open(st, argv[4], opt(argc, argv, "--cache", nullptr)) : nullptr;
    if (!ct) { fprintf(stderr, "dino-embed: %s\n", z3_error()); return 1; }
    const int64_t box[6] = {b[0], b[1], b[2], b[3], b[4], b[5]};
    const double t0 = dn_now();
    int8_t *e = dn_embed_box(u, ct, box, P, atoi(opt(argc, argv, "--halo", "32")), c, s, 8, atoi(opt(argc, argv, "--ema", "1")));
    if (!e) return 1;
    const double t1 = dn_now();
    if (dv_write_dense_box(argv[5], z3_meta_of(ct)->shape, box, cfg.cout, e)) return 1;
    fprintf(stderr, "dino-embed: %lld x %lld x %lld voxels in %.1f s (%.1f Mvox/s network + read), written in %.1f s\n", b[3], b[4], b[5], t1 - t0, (double)b[3] * b[4] * b[5] / (t1 - t0) / 1e6, dn_now() - t1);
    free(e);
    return 0;
}

/* ufsm dino-boxes <codes-root> <codes-key> [--block 512] [--min 200000] [--exclude z,y,x,nz,ny,nx]... : block-aligned boxes holding at least
   --min labelled voxels (codes 1..254), as a JSON list for dino-train sources; blocks touching an --exclude box are left out */
int cmd_dino_boxes(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm dino-boxes <codes-root> <codes-key> [--block 512] [--min 200000] [--exclude z,y,x,nz,ny,nx]...\n"); return 2; }
    store *st = store_open(argv[2]); z3 *z = st ? z3_open(st, argv[3], nullptr) : nullptr;
    if (!z) { fprintf(stderr, "dino-boxes: %s\n", z3_error()); return 1; }
    const int B = atoi(opt(argc, argv, "--block", "512")); const long mn = atol(opt(argc, argv, "--min", "200000"));
    int64_t ex[32][6]; int nex = 0;
    for (int i = 2; i + 1 < argc && nex < 32; i++) if (!strcmp(argv[i], "--exclude")) { long long v[6]; if (sscanf(argv[i + 1], "%lld,%lld,%lld,%lld,%lld,%lld", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5]) == 6) { for (int d = 0; d < 6; d++) ex[nex][d] = v[d]; nex++; } }
    const z3_meta *m = z3_meta_of(z); int64_t grid[3]; uint8_t *bits = nullptr;
    if (z3_chunk_map(z, grid, &bits) < 0) { fprintf(stderr, "dino-boxes: chunk map: %s\n", z3_error()); return 1; }
    const int cpb = B / m->chunk[0];   /* chunks per block edge */
    uint8_t *buf = malloc((size_t)m->chunk[0] * m->chunk[1] * m->chunk[2]);
    int first = 1; printf("[");
    for (int64_t bz = 0; bz * B < m->shape[0]; bz++) for (int64_t by = 0; by * B < m->shape[1]; by++) for (int64_t bx = 0; bx * B < m->shape[2]; bx++) {
        const int64_t o[3] = {bz * B, by * B, bx * B};
        int bad = 0; for (int e = 0; e < nex && !bad; e++) bad = box_hit(ex[e], o, B);
        if (bad) continue;
        long cnt = 0;
        for (int a = 0; a < cpb; a++) for (int b = 0; b < cpb; b++) for (int c = 0; c < cpb; c++) {
            const int64_t cz = bz * cpb + a, cy = by * cpb + b, cx = bx * cpb + c;
            if (cz >= grid[0] || cy >= grid[1] || cx >= grid[2] || !bits[(cz * grid[1] + cy) * grid[2] + cx]) continue;
            const int64_t oo[3] = {cz * m->chunk[0], cy * m->chunk[1], cx * m->chunk[2]}, nn[3] = {m->chunk[0], m->chunk[1], m->chunk[2]};
            if (z3_read(z, oo, nn, buf, 4)) continue;
            for (size_t i = 0; i < (size_t)nn[0] * nn[1] * nn[2]; i++) cnt += buf[i] && buf[i] != 255;
        }
        if (cnt < mn) continue;
        if (o[0] + B > m->shape[0] || o[1] + B > m->shape[1] || o[2] + B > m->shape[2]) continue;
        printf("%s\n  [%lld,%lld,%lld,%d,%d,%d]", first ? "" : ",", (long long)o[0], (long long)o[1], (long long)o[2], B, B, B);
        fprintf(stderr, "block %lld,%lld,%lld: %ld labelled voxels\n", (long long)o[0], (long long)o[1], (long long)o[2], cnt);
        first = 0;
    }
    printf("\n]\n");
    free(buf); free(bits); z3_close(z); store_close(st);
    return 0;
}
