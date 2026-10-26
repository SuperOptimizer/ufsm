/* ufsm field-blend (src/fblend.h) */
#include "fblend.h"
#include "json.h"
#include "store.h"
#include "z3w.h"
#include "zarr3.h"
#include <dirent.h>
#include <pthread.h>
#include <stdatomic.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define FB_MAXIN 32
#define FB_MAXP 64
#define FB_ROWS 128   /* field rows (z and y) per pass (= the output zarr shard) */

static inline uint8_t clamp8(double v, int lo) { return (uint8_t)(v < lo ? lo : v > 255 ? 255 : lrint(v)); }

#define CW(i, j) (w[i] * (cw && cw[i] ? cw[i][j] : 1.f))
void fb_phase_cw(int n, const uint8_t *const *pc, const uint8_t *const *ps, const float *w, const float *const *cw, int conf, size_t ncell, uint8_t *opc, uint8_t *ops) {
    for (size_t j = 0; j < ncell; j++) {
        double x = 0, y = 0, ws = 0, m = 0;
        for (int i = 0; i < n; i++) if (pc[i] && ps[i]) {
            const double q = CW(i, j), vx = (pc[i][j] - 128.0) / 127.0, vy = (ps[i][j] - 128.0) / 127.0;
            x += q * vx; y += q * vy; ws += q; if (conf == FB_CONF_MEAN) m += q * hypot(vx, vy);
        }
        if (ws > 0) { x /= ws; y /= ws; m /= ws; }
        if (conf == FB_CONF_MEAN) { const double r = hypot(x, y); if (r > 0) { x *= m / r; y *= m / r; } }
        opc[j] = clamp8(128 + 127 * x, 1); ops[j] = clamp8(128 + 127 * y, 1);
    }
}
void fb_phase(int n, const uint8_t *const *pc, const uint8_t *const *ps, const float *w, size_t ncell, uint8_t *opc, uint8_t *ops) {
    fb_phase_cw(n, pc, ps, w, nullptr, FB_CONF_RESULTANT, ncell, opc, ops);
}
void fb_mean_cw(int n, const uint8_t *const *a, const float *w, const float *const *cw, size_t ncell, uint8_t *o) {
    for (size_t j = 0; j < ncell; j++) {
        double s = 0, ws = 0; for (int i = 0; i < n; i++) if (a[i]) { const double q = CW(i, j); s += q * a[i][j]; ws += q; }
        o[j] = clamp8(ws > 0 ? s / ws : 0, 0);
    }
}
void fb_mean(int n, const uint8_t *const *a, const float *w, size_t ncell, uint8_t *o) { fb_mean_cw(n, a, w, nullptr, ncell, o); }
void fb_max(int n, const uint8_t *const *a, size_t ncell, uint8_t *o) {
    for (size_t j = 0; j < ncell; j++) { uint8_t m = 0; for (int i = 0; i < n; i++) if (a[i] && a[i][j] > m) m = a[i][j]; o[j] = m; }
}
void fb_median_known_cw(int n, const uint8_t *const *a, const float *w, const float *const *cw, size_t ncell, uint8_t *o) {
    for (size_t j = 0; j < ncell; j++) {
        uint8_t v[FB_MAXIN]; float vw[FB_MAXIN]; int m = 0; double tot = 0;
        for (int i = 0; i < n; i++) if (a[i] && a[i][j] && CW(i, j) > 0) {   /* insertion sort by value */
            int k = m++; while (k > 0 && v[k - 1] > a[i][j]) { v[k] = v[k - 1]; vw[k] = vw[k - 1]; k--; }
            v[k] = a[i][j]; vw[k] = CW(i, j); tot += vw[k];
        }
        if (!m) { o[j] = 0; continue; }
        double cum = 0; int k = 0;
        for (; k < m; k++) { cum += vw[k]; if (cum >= 0.5 * tot - 1e-9 * tot) break; }
        /* exactly half the weight at or below v[k]: the midpoint with the next value (two equal inputs: their mean) */
        if (k + 1 < m && fabs(cum - 0.5 * tot) <= 1e-9 * tot) o[j] = clamp8(0.5 * (v[k] + v[k + 1]), 1);
        else o[j] = v[k];
    }
}
void fb_median_known(int n, const uint8_t *const *a, const float *w, size_t ncell, uint8_t *o) { fb_median_known_cw(n, a, w, nullptr, ncell, o); }
void fb_wind4_cw(int n, const uint8_t *const (*d)[4], const float *w, const float *const *cw, size_t ncell, uint8_t *const o[4]) {
    const double k = 2 * M_PI / 256;
    for (size_t j = 0; j < ncell; j++) {
        double rs = 0, ws = 0; for (int i = 0; i < n; i++) if (d[i][0]) ws += CW(i, j);
        for (int a = 0; a < 3; a++) {
            double x = 0, y = 0;
            for (int i = 0; i < n; i++) if (d[i][0]) { const double q = CW(i, j) * d[i][3][j] / 255.0; x += q * cos(k * d[i][a][j]); y += q * sin(k * d[i][a][j]); }
            const double r = ws > 0 ? hypot(x, y) / ws : 0;
            o[a][j] = r > 0 ? (uint8_t)((long)lrint(atan2(y, x) / k) & 255) : 0;
            rs += r;
        }
        o[3][j] = clamp8(255 * rs / 3, 0);
    }
}
void fb_wind4(int n, const uint8_t *const (*d)[4], const float *w, size_t ncell, uint8_t *const o[4]) { fb_wind4_cw(n, d, w, nullptr, ncell, o); }
#undef CW
double fb_tile_w(long long o, int h, int c, double S, double g) {
    double u = fmod((double)o + ((double)c + 0.5) * h - g, S) / S; if (u < 0) u += 1;
    const double s = sin(M_PI * u); return s * s > 1e-6 ? s * s : 1e-6;
}

/* ---- field directories ---- */
typedef struct {
    const char *dir; json *j; int zarr, shape[3], h; double um; long long origin[3], off[3], rsh[3];
    int np; char *pn[FB_MAXP];   /* the planes present */
} fb_in;
static json *read_json(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) return nullptr;
    fseek(f, 0, SEEK_END); const long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *t = malloc((size_t)n + 1); if (!t || fread(t, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(t); return nullptr; }
    fclose(f); t[n] = 0; json *j = json_parse(t, (size_t)n); free(t); return j;
}
static int fb_open(fb_in *I, const char *dir) {
    memset(I, 0, sizeof *I); I->dir = dir;
    char p[2048]; snprintf(p, sizeof p, "%s/fields.json", dir);
    if (!(I->j = read_json(p))) { fprintf(stderr, "field-blend: cannot read %s\n", p); return -1; }
    I->zarr = !strcmp(json_str(json_get(I->j, "format"), "raw"), "zarr");
    I->h = (int)json_num(json_get(I->j, "h"), 8); I->um = json_num(json_get(I->j, "um"), 2.4);
    for (int d = 0; d < 3; d++) {
        I->origin[d] = (long long)json_num(json_at(json_get(I->j, "origin"), d), 0); I->shape[d] = (int)json_num(json_at(json_get(I->j, "shape"), d), 0);
        I->off[d] = (long long)json_num(json_at(json_get(I->j, I->zarr ? "zarr_offset" : "raw_offset"), d), 0);
        I->rsh[d] = I->zarr ? I->shape[d] : (long long)json_num(json_at(json_get(I->j, "raw_shape"), d), I->shape[d]);
        if (I->shape[d] <= 0) { fprintf(stderr, "field-blend: %s: no shape\n", p); return -1; }
    }
    DIR *D = opendir(dir); if (!D) { fprintf(stderr, "field-blend: cannot list %s\n", dir); return -1; }
    struct dirent *e;
    while ((e = readdir(D)) && I->np < FB_MAXP) {   /* zarr: subdirectories with a zarr.json; raw: NAME.raw */
        const size_t L = strlen(e->d_name); struct stat st;
        if (e->d_name[0] == '.') continue;
        if (I->zarr) { snprintf(p, sizeof p, "%s/%s/zarr.json", dir, e->d_name); if (stat(p, &st)) continue; I->pn[I->np++] = strdup(e->d_name); }
        else if (L > 4 && !strcmp(e->d_name + L - 4, ".raw")) I->pn[I->np++] = strndup(e->d_name, L - 4);
    }
    closedir(D);
    for (int a = 1; a < I->np; a++) for (int b = a; b > 0 && strcmp(I->pn[b - 1], I->pn[b]) > 0; b--) { char *t = I->pn[b]; I->pn[b] = I->pn[b - 1]; I->pn[b - 1] = t; }
    return 0;
}
static int fb_has(const fb_in *I, const char *name) { for (int k = 0; k < I->np; k++) if (!strcmp(I->pn[k], name)) return 1; return 0; }
static void fb_close(fb_in *I) { json_free(I->j); for (int k = 0; k < I->np; k++) free(I->pn[k]); }

/* a plane reader: rows [z0, z0 + nz) of the input's box */
typedef struct { z3 *z; FILE *f; const fb_in *I; } fb_rd;
static int rd_open(fb_rd *r, const fb_in *I, const char *name) {
    r->I = I; r->z = nullptr; r->f = nullptr;
    char p[2048];
    if (I->zarr) { r->z = z3_open(store_open(I->dir), name, nullptr); if (!r->z) { fprintf(stderr, "field-blend: %s/%s: %s\n", I->dir, name, z3_error()); return -1; } }
    else { snprintf(p, sizeof p, "%s/%s.raw", I->dir, name); if (!(r->f = fopen(p, "rb"))) { fprintf(stderr, "field-blend: cannot open %s\n", p); return -1; } }
    return 0;
}
static int rd_rows(fb_rd *r, int z0, int nz, int y0, int H, uint8_t *out, int nthreads) {   /* rows [z0, z0 + nz) x [y0, y0 + H) */
    const fb_in *I = r->I; const int W = I->shape[2];
    if (r->z) { const int64_t o[3] = {z0 + I->off[0], y0 + I->off[1], I->off[2]}, n[3] = {nz, H, W}; return z3_read(r->z, o, n, out, nthreads) ? -1 : 0; }
    for (int z = 0; z < nz; z++) for (int y = 0; y < H; y++) {
        const off_t at = (off_t)((((size_t)(I->off[0] + z0 + z) * I->rsh[1]) + (size_t)(I->off[1] + y0 + y)) * I->rsh[2] + (size_t)I->off[2]);
        if (fseeko(r->f, at, SEEK_SET) || fread(out + ((size_t)z * H + y) * W, 1, (size_t)W, r->f) != (size_t)W) return -1;
    }
    return 0;
}
static void rd_close(fb_rd *r) { if (r->z) z3_close(r->z); if (r->f) fclose(r->f); }

/* ---- fields.json: in[0]'s, re-serialized with the format / recto / w4_d / blend keys replaced ---- */
static void jw(FILE *f, const json *j) {
    if (!j) { fputs("null", f); return; }
    switch (j->type) {
    case J_NULL: fputs("null", f); break;
    case J_BOOL: fputs(j->num ? "true" : "false", f); break;
    case J_NUM: if (j->num == floor(j->num) && fabs(j->num) < 1e15) fprintf(f, "%.0f", j->num); else fprintf(f, "%.10g", j->num); break;
    case J_STR: fputc('"', f); for (const char *c = j->str; *c; c++) { if (*c == '"' || *c == '\\') fputc('\\', f); fputc(*c, f); } fputc('"', f); break;
    case J_ARR: fputc('[', f); for (size_t i = 0; i < j->n; i++) { if (i) fputs(", ", f); jw(f, j->items[i]); } fputc(']', f); break;
    case J_OBJ: fputc('{', f); for (size_t i = 0; i < j->n; i++) { if (i) fputs(", ", f); fprintf(f, "\"%s\": ", j->keys[i]); jw(f, j->items[i]); } fputc('}', f); break;
    }
}

static int is_t(const char *s) { return s[0] == 't' && s[1] >= '0' && s[1] <= '5' && !s[2]; }
static int is_w4(const char *s) { return !strcmp(s, "dz") || !strcmp(s, "dy") || !strcmp(s, "dx") || !strcmp(s, "dq"); }

/* the work of one slab of FB_ROWS field rows, on threads: reads (plane, input), combines (plane, cell chunk), shard writes (plane, shard) */
#define FB_CHUNK ((size_t)1 << 20)
typedef struct {
    int n, np, H, W, z0, nz, y0, nchunk, nx, kps, kw4[4];   /* H: the band's rows (of 128 at y0) */ const char **pl; const float *w;
    fb_rd (*R)[FB_MAXIN]; int (*use)[FB_MAXIN]; uint8_t *(*buf)[FB_MAXIN]; uint8_t **obuf; z3w **ZW; atomic_int next, fail;
    const float *aw[3][FB_MAXIN]; int conf;   /* tile weights per input along z, y, x (aw[0][0] nullptr: none); the phase confidence rule */
} fb_slab;
typedef void (*fb_fn)(fb_slab *, int);
typedef struct { fb_slab *S; fb_fn fn; int njob; } fb_pool;
static void *fb_worker(void *a) {
    fb_pool *P = a;
    for (;;) { const int j = atomic_fetch_add(&P->S->next, 1); if (j >= P->njob) break; P->fn(P->S, j); }
    return nullptr;
}
static void fb_par(int njob, int nthreads, fb_fn fn, fb_slab *S) {
    fb_pool P = {S, fn, njob}; atomic_store(&S->next, 0);
    int nt = nthreads < njob ? nthreads : njob; if (nt > 64) nt = 64; if (nt < 1) nt = 1;
    pthread_t th[64]; int up[64] = {0};
    for (int t = 1; t < nt; t++) up[t] = pthread_create(&th[t], nullptr, fb_worker, &P) == 0;
    fb_worker(&P);
    for (int t = 1; t < nt; t++) if (up[t]) pthread_join(th[t], nullptr);
}
static void fb_job_read(fb_slab *S, int j) {
    const int k = j / S->n, i = j % S->n;
    if (S->use[k][i] && rd_rows(&S->R[k][i], S->z0, S->nz, S->y0, S->H, S->buf[k][i], 1)) atomic_store(&S->fail, 1);
}
static void fb_job_combine(fb_slab *S, int j) {
    const int k = j / S->nchunk, c = j % S->nchunk, n = S->n; const char *nm = S->pl[k];
    const size_t nc = (size_t)S->nz * S->H * S->W, c0 = (size_t)c * FB_CHUNK, m = nc - c0 < FB_CHUNK ? nc - c0 : FB_CHUNK;
    const uint8_t *src[FB_MAXIN]; for (int i = 0; i < n; i++) src[i] = S->use[k][i] ? S->buf[k][i] + c0 : nullptr;
    uint8_t *o = S->obuf[k] + c0;
    if (!strcmp(nm, "ps") || (is_w4(nm) && strcmp(nm, "dz"))) return;   /* (done with pc / dz) */
    float *cwb = nullptr; const float *cw[FB_MAXIN] = {0};   /* tile weights of this chunk's cells: wz x wy x wx per input */
    if (S->aw[0][0] && (!strcmp(nm, "pc") || !strcmp(nm, "dz") || is_t(nm) || !strcmp(nm, "r") || !strcmp(nm, "sp"))) {
        if (!(cwb = malloc((size_t)n * m * sizeof *cwb))) { atomic_store(&S->fail, 1); return; }
        for (int i = 0; i < n; i++) {
            float *q = cwb + (size_t)i * m; cw[i] = q;
            for (size_t t = 0; t < m; t++) {
                const size_t jj = c0 + t; const int x = (int)(jj % S->W), y = (int)(jj / S->W % S->H), z = (int)(jj / ((size_t)S->W * S->H));
                q[t] = S->aw[0][i][S->z0 + z] * S->aw[1][i][S->y0 + y] * S->aw[2][i][x];
            }
        }
    }
    const float *const *cwp = cwb ? cw : nullptr;
    if (!strcmp(nm, "pc")) {   /* with its ps partner */
        const int kp = S->kps; const uint8_t *s2[FB_MAXIN]; for (int i = 0; i < n; i++) s2[i] = S->use[kp][i] ? S->buf[kp][i] + c0 : nullptr;
        fb_phase_cw(n, src, s2, S->w, cwp, S->conf, m, o, S->obuf[kp] + c0);
    } else if (!strcmp(nm, "dz")) {
        const uint8_t *d4[FB_MAXIN][4]; for (int i = 0; i < n; i++) for (int a = 0; a < 4; a++) d4[i][a] = S->use[S->kw4[a]][i] ? S->buf[S->kw4[a]][i] + c0 : nullptr;
        uint8_t *const o4[4] = {S->obuf[S->kw4[0]] + c0, S->obuf[S->kw4[1]] + c0, S->obuf[S->kw4[2]] + c0, S->obuf[S->kw4[3]] + c0};
        fb_wind4_cw(n, (const uint8_t *const (*)[4])d4, S->w, cwp, m, o4);
    }
    else if (is_t(nm) || !strcmp(nm, "r")) fb_mean_cw(n, src, S->w, cwp, m, o);
    else if (!strcmp(nm, "sp")) fb_median_known_cw(n, src, S->w, cwp, m, o);
    else if (!strcmp(nm, "w")) fb_max(n, src, m, o);
    else { for (int i = 0; i < n; i++) if (src[i]) { memcpy(o, src[i], m); break; } }   /* passed through */
    free(cwb);
}
static void fb_job_write(fb_slab *S, int j) {
    const int k = j / S->nx, sx = j % S->nx, W = S->W;
    uint8_t *sb = malloc((size_t)FB_ROWS * FB_ROWS * FB_ROWS); if (!sb) { atomic_store(&S->fail, 1); return; }
    memset(sb, !strcmp(S->pl[k], "pc") || !strcmp(S->pl[k], "ps") ? 128 : 0, (size_t)FB_ROWS * FB_ROWS * FB_ROWS);
    const int nx = W - sx * FB_ROWS < FB_ROWS ? W - sx * FB_ROWS : FB_ROWS;
    for (int a = 0; a < S->nz; a++) for (int b = 0; b < S->H; b++)
        memcpy(sb + ((size_t)a * FB_ROWS + b) * FB_ROWS, S->obuf[k] + ((size_t)a * S->H + b) * W + (size_t)sx * FB_ROWS, (size_t)nx);
    if (z3w_write_shard(S->ZW[k], S->z0 / FB_ROWS, S->y0 / FB_ROWS, sx, sb, 1)) atomic_store(&S->fail, 1);
    free(sb);
}

int fblend_run(const char *out, int n, const char *const *in, const float *wt, int nthreads) { return fblend_run2(out, n, in, wt, nullptr, nthreads); }
int fblend_run2(const char *out, int n, const char *const *in, const float *wt, const fb_opts *opt, int nthreads) {
    float *awb = nullptr;
    if (n < 1 || n > FB_MAXIN) { fprintf(stderr, "field-blend: 1..%d inputs\n", FB_MAXIN); return 2; }
    float w[FB_MAXIN]; for (int i = 0; i < n; i++) { w[i] = wt ? wt[i] : 1.f; if (!(w[i] >= 0)) { fprintf(stderr, "field-blend: weights must be >= 0\n"); return 2; } }
    fb_in I[FB_MAXIN]; int rc = 1, nin = 0;
    for (; nin < n; nin++) if (fb_open(&I[nin], in[nin])) { nin++; goto done; }
    for (int i = 1; i < n; i++) for (int d = 0; d < 3; d++)
        if (I[i].shape[d] != I[0].shape[d] || fabs(I[i].h * I[i].um - I[0].h * I[0].um) > 1e-6 * I[0].h * I[0].um ||
            fabs(I[i].origin[d] * I[i].um - I[0].origin[d] * I[0].um) > 1e-6 * (1 + fabs(I[0].origin[d] * I[0].um))) {
            fprintf(stderr, "field-blend: %s and %s cover different cells (axis %d: shape %d / %d, origin %.1f / %.1f um, cell %.2f / %.2f um)\n", in[0], in[i], d,
                    I[0].shape[d], I[i].shape[d], I[0].origin[d] * I[0].um, I[i].origin[d] * I[i].um, I[0].h * I[0].um, I[i].h * I[i].um);
            rc = 2; goto done;
        }
    /* the output planes: the known ones any input has (the phase pair and the winding mod 4 quad only whole), then in[0]'s others */
    static const char *known[] = {"pc", "ps", "t0", "t1", "t2", "t3", "t4", "t5", "sp", "w", "dz", "dy", "dx", "dq", "r"};
    const char *pl[FB_MAXP]; int np = 0, w4src = -1;
    for (int k = 0; k < 15; k++) for (int i = 0; i < n; i++) {
        const char *nm = known[k];
        const int ok = !strcmp(nm, "pc") || !strcmp(nm, "ps") ? fb_has(&I[i], "pc") && fb_has(&I[i], "ps")
                     : is_w4(nm) ? fb_has(&I[i], "dz") && fb_has(&I[i], "dy") && fb_has(&I[i], "dx") && fb_has(&I[i], "dq") : fb_has(&I[i], nm);
        if (ok) { pl[np++] = nm; if (is_w4(nm) && w4src < 0) w4src = i; break; }
    }
    for (int k = 0; k < I[0].np && np < FB_MAXP; k++) { int kn = 0; for (int q = 0; q < 15; q++) if (!strcmp(I[0].pn[k], known[q])) kn = 1; if (!kn) pl[np++] = I[0].pn[k]; }
    if (!np) { fprintf(stderr, "field-blend: %s has no planes\n", in[0]); goto done; }
    const int D = I[0].shape[0], H = I[0].shape[1], W = I[0].shape[2], ozarr = I[0].zarr;
    if (mkdir(out, 0755) && errno != EEXIST) { fprintf(stderr, "field-blend: cannot create %s\n", out); goto done; }
    /* readers [plane][input] (nullptr: the input does not take part) and writers */
    fb_rd (*R)[FB_MAXIN] = calloc((size_t)np, sizeof *R); int (*use)[FB_MAXIN] = calloc((size_t)np, sizeof *use);
    z3w *ZW[FB_MAXP] = {0}; FILE *FW[FB_MAXP] = {0};
    const size_t rowc = (size_t)FB_ROWS * W;   /* a band: 128 rows of y (memory: 128 x 128 x W per plane and input) */
    uint8_t *(*buf)[FB_MAXIN] = calloc((size_t)np, sizeof *buf), *obuf[FB_MAXP] = {0};
    if (!R || !use || !buf) goto done2;
    for (int k = 0; k < np; k++) {
        int known_ = 0; for (int q = 0; q < 15; q++) if (!strcmp(pl[k], known[q])) known_ = 1;
        int pass = -1; if (!known_) for (int i = 0; i < n && pass < 0; i++) if (fb_has(&I[i], pl[k])) pass = i;   /* (in[0] has it) */
        for (int i = 0; i < n; i++) {
            const int on = known_ ? (is_w4(pl[k]) ? fb_has(&I[i], "dz") && fb_has(&I[i], "dy") && fb_has(&I[i], "dx") && fb_has(&I[i], "dq")
                                                  : !strcmp(pl[k], "pc") || !strcmp(pl[k], "ps") ? fb_has(&I[i], "pc") && fb_has(&I[i], "ps") : fb_has(&I[i], pl[k]))
                                  : i == pass;
            if (!on) continue;
            use[k][i] = 1;
            if (rd_open(&R[k][i], &I[i], pl[k])) goto done2;
            if (!(buf[k][i] = malloc((size_t)FB_ROWS * rowc))) goto done2;
        }
        if (!(obuf[k] = malloc((size_t)FB_ROWS * rowc))) goto done2;
        char p[2048];
        if (ozarr) {
            snprintf(p, sizeof p, "%s/%s", out, pl[k]);
            char at[512]; snprintf(at, sizeof at, "{\"ufsm\":{\"content\":\"winding fitter fields (src/fit.h), ufsm field-blend of %d sets\",\"h\":%d,\"origin_zyx\":[%lld,%lld,%lld]}}", n, I[0].h, I[0].origin[0], I[0].origin[1], I[0].origin[2]);
            const int64_t sh[3] = {D, H, W};
            if (!(ZW[k] = z3w_create(p, sh, FB_ROWS, 0, !strcmp(pl[k], "pc") || !strcmp(pl[k], "ps") ? 128 : 0, at))) { fprintf(stderr, "field-blend: %s\n", z3w_error()); goto done2; }
        } else {
            snprintf(p, sizeof p, "%s/%s.raw", out, pl[k]);
            if (!(FW[k] = fopen(p, "wb"))) { fprintf(stderr, "field-blend: cannot write %s\n", p); goto done2; }
        }
    }
    fb_slab S = {.n = n, .np = np, .W = W, .pl = pl, .w = w, .R = R, .use = use, .buf = buf, .obuf = obuf, .ZW = ZW};
    S.conf = opt ? opt->conf : FB_CONF_RESULTANT;
    if (opt && opt->tile) {   /* the tile weights of every input along each axis (fb_tile_w) */
        const int sh[3] = {D, H, W}; if (!(awb = malloc((size_t)n * (D + H + W) * sizeof *awb))) goto done2;
        float *q = awb;
        for (int i = 0; i < n; i++) for (int d = 0; d < 3; d++) {
            for (int c = 0; c < sh[d]; c++) q[c] = (float)fb_tile_w(I[i].origin[d], I[i].h, c, opt->stride[d], opt->grid[i][d]);
            S.aw[d][i] = q; q += sh[d];
        }
    }
    for (int k = 0; k < np; k++) {   /* the partner planes of the groups */
        if (!strcmp(pl[k], "ps")) S.kps = k;
        static const char *wn[4] = {"dz", "dy", "dx", "dq"}; for (int a = 0; a < 4; a++) if (!strcmp(pl[k], wn[a])) S.kw4[a] = k;
    }
    for (int z0 = 0; z0 < D; z0 += FB_ROWS) for (int y0 = 0; y0 < H; y0 += FB_ROWS) {   /* bands of 128 x 128 x W cells */
        S.z0 = z0; S.nz = D - z0 < FB_ROWS ? D - z0 : FB_ROWS; S.y0 = y0; S.H = H - y0 < FB_ROWS ? H - y0 : FB_ROWS;
        const size_t nc = (size_t)S.nz * S.H * W;
        atomic_store(&S.fail, 0);
        fb_par(np * n, nthreads, fb_job_read, &S);     /* (plane, input) reads */
        if (atomic_load(&S.fail)) { fprintf(stderr, "field-blend: cannot read rows z %d.., y %d..\n", z0, y0); goto done2; }
        S.nchunk = (int)((nc + FB_CHUNK - 1) / FB_CHUNK);
        fb_par(np * S.nchunk, nthreads, fb_job_combine, &S);   /* (plane, cell chunk) */
        if (atomic_load(&S.fail)) { fprintf(stderr, "field-blend: out of memory\n"); goto done2; }
        for (int k = 0; k < np; k++) if (FW[k]) for (int a = 0; a < S.nz; a++) for (int b = 0; b < S.H; b++)
            if (fseeko(FW[k], (off_t)((((size_t)(z0 + a) * H) + (size_t)(y0 + b)) * W), SEEK_SET) || fwrite(obuf[k] + ((size_t)a * S.H + b) * W, 1, (size_t)W, FW[k]) != (size_t)W) {
                fprintf(stderr, "field-blend: write failed (%s)\n", pl[k]); goto done2;
            }
        if (ozarr) {
            S.nx = (W + FB_ROWS - 1) / FB_ROWS;
            fb_par(np * S.nx, nthreads, fb_job_write, &S);   /* (plane, shard) */
            if (atomic_load(&S.fail)) { fprintf(stderr, "field-blend: %s\n", z3w_error()); goto done2; }
        }
    }
    {   /* fields.json */
        char p[2048]; snprintf(p, sizeof p, "%s/fields.json", out); FILE *f = fopen(p, "w");
        if (!f) { fprintf(stderr, "field-blend: cannot write %s\n", p); goto done2; }
        int has_r = 0; for (int k = 0; k < np; k++) if (!strcmp(pl[k], "r")) has_r = 1;
        fprintf(f, "{\"format\": \"%s\"", ozarr ? "zarr" : "raw");
        const json *j = I[0].j;
        for (size_t i = 0; i < j->n; i++) {
            const char *key = j->keys[i];
            if (!strcmp(key, "format") || !strcmp(key, "zarr_offset") || !strcmp(key, "raw_offset") || !strcmp(key, "raw_shape") || !strcmp(key, "recto") ||
                !strcmp(key, "w4_d") || !strcmp(key, "blend")) continue;
            fprintf(f, ", \"%s\": ", key); jw(f, j->items[i]);
        }
        fprintf(f, ", \"recto\": %s", has_r ? "true" : "false");
        if (w4src >= 0) fprintf(f, ", \"w4_d\": %d", (int)json_num(json_get(I[w4src].j, "w4_d"), 32));
        fprintf(f, ", \"blend\": {\"inputs\": [");
        for (int i = 0; i < n; i++) { fprintf(f, "%s\"", i ? ", " : ""); for (const char *c = in[i]; *c; c++) { if (*c == '"' || *c == '\\') fputc('\\', f); fputc(*c, f); } fputc('"', f); }
        fprintf(f, "], \"weights\": [");
        for (int i = 0; i < n; i++) fprintf(f, "%s%.6g", i ? ", " : "", w[i]);
        fprintf(f, "]");
        if (opt && opt->tile) {
            fprintf(f, ", \"tile_stride\": [%g, %g, %g], \"tile_grids\": [", opt->stride[0], opt->stride[1], opt->stride[2]);
            for (int i = 0; i < n; i++) fprintf(f, "%s[%g, %g, %g]", i ? ", " : "", opt->grid[i][0], opt->grid[i][1], opt->grid[i][2]);
            fprintf(f, "]");
        }
        if (opt && opt->conf == FB_CONF_MEAN) fprintf(f, ", \"conf\": \"mean\"");
        fprintf(f, "}}\n");
        if (fclose(f)) goto done2;
    }
    rc = 0;
    fprintf(stderr, "field-blend: %d sets, %d planes (", n, np); for (int k = 0; k < np; k++) fprintf(stderr, "%s%s", k ? " " : "", pl[k]);
    fprintf(stderr, "), %dx%dx%d cells -> %s (%s)\n", D, H, W, out, ozarr ? "zarr" : "raw");
done2:
    for (int k = 0; k < np; k++) {
        for (int i = 0; i < n; i++) { if (use && use[k][i]) rd_close(&R[k][i]); if (buf) free(buf[k][i]); }
        free(obuf[k]); if (ZW[k] && z3w_close(ZW[k])) rc = 1; if (FW[k] && fclose(FW[k])) rc = 1;
    }
    free(R); free(use); free(buf); free(awb);
done:
    for (int i = 0; i < nin; i++) fb_close(&I[i]);
    return rc;
}
