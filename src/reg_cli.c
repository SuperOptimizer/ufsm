/* ufsm reg <pair|boxes|fit|stats|overlay|init> : scan-to-scan registration commands (src/reg.h).
   A map JSON (written by reg pair, or reg fit's reg.json) names the two volumes and the mapping:
     {"name", "coarse": {"root", "key", "um"}, "fine": {"root", "key", "um"}, "A": 3x4 fine L0 xyz -> coarse L0 xyz,
      "field": {"file", "n": [nz, ny, nx], "o": [x, y, z], "h"} (optional; float32 [nz][ny][nx][3] x y z displacement, coarse L0 voxels)}
   coarse c <-> fine A^-1 (c + D(c)). */
#include "reg.h"
#include "json.h"
#include "store.h"
#include "imgops.h"
#include "zarr3.h"
#include "z3w.h"
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
static int has(int argc, char **argv, const char *name) { for (int i = 1; i < argc; i++) if (!strcmp(argv[i], name)) return 1; return 0; }
static char *slurp(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb"); if (!f) return nullptr;
    fseek(f, 0, SEEK_END); const long L = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = malloc(L + 1); if (fread(b, 1, L, f) != (size_t)L) { fclose(f); free(b); return nullptr; }
    b[L] = 0; fclose(f); if (n) *n = L; return b;
}
static json *load_json(const char *path) {
    size_t n; char *t = slurp(path, &n); if (!t) { fprintf(stderr, "reg: cannot read %s\n", path); return nullptr; }
    json *j = json_parse(t, n); free(t);
    if (!j) fprintf(stderr, "reg: bad JSON %s\n", path);
    return j;
}
static int get_aff(const json *a, raff *r) {   /* 3x4 or 4x4 rows */
    if (!a || a->type != J_ARR || a->n < 3) return -1;
    for (int i = 0; i < 3; i++) { const json *row = json_at(a, i); if (!row || row->n < 4) return -1; for (int j = 0; j < 4; j++) r->m[i][j] = json_num(json_at(row, j), 0); }
    return 0;
}
static void put_aff(FILE *f, const raff *a) {
    fprintf(f, "[[%.12g,%.12g,%.12g,%.12g],[%.12g,%.12g,%.12g,%.12g],[%.12g,%.12g,%.12g,%.12g]]", a->m[0][0], a->m[0][1], a->m[0][2], a->m[0][3],
            a->m[1][0], a->m[1][1], a->m[1][2], a->m[1][3], a->m[2][0], a->m[2][1], a->m[2][2], a->m[2][3]);
}
static void dirname_of(const char *p, char *out, size_t n) {
    snprintf(out, n, "%s", p); char *s = strrchr(out, '/'); if (s) *s = 0; else snprintf(out, n, ".");
}

/* ---------- map loading ---------- */
typedef struct { json *j; rvol *cv, *fv; rmap m; rfield D; char cache[1024]; } rctx;
static int ctx_open(rctx *x, const char *path, const char *cache, int open_vols) {
    memset(x, 0, sizeof *x);
    if (!(x->j = load_json(path))) return -1;
    if (get_aff(json_get(x->j, "A"), &x->m.A)) { fprintf(stderr, "reg: %s has no A\n", path); return -1; }
    x->m.Ainv = raff_inv(x->m.A);
    const json *fd = json_get(x->j, "field");
    if (fd) {
        char dir[1024], fp[2100]; dirname_of(path, dir, sizeof dir);
        const char *fn = json_str(json_get(fd, "file"), "field.f32");
        if (fn[0] == '/') snprintf(fp, sizeof fp, "%s", fn); else snprintf(fp, sizeof fp, "%s/%s", dir, fn);
        for (int a = 0; a < 3; a++) { x->D.n[a] = (int)json_num(json_at(json_get(fd, "n"), a), 0); x->D.o[a] = json_num(json_at(json_get(fd, "o"), a), 0); }
        x->D.h = json_num(json_get(fd, "h"), 1);
        const size_t N = (size_t)x->D.n[0] * x->D.n[1] * x->D.n[2] * 3; size_t got = 0;
        float *d = (float *)slurp(fp, &got);
        if (!d || got != N * sizeof(float)) { fprintf(stderr, "reg: field %s: %zu bytes, expected %zu\n", fp, got, N * sizeof(float)); free(d); return -1; }
        x->D.d = d; x->m.D = &x->D;
    }
    snprintf(x->cache, sizeof x->cache, "%s", cache ? cache : "");
    if (!open_vols) return 0;
    const json *c = json_get(x->j, "coarse"), *f = json_get(x->j, "fine");
    x->cv = rvol_open(json_str(json_get(c, "root"), ""), json_str(json_get(c, "key"), ""), cache, json_num(json_get(c, "um"), 0));
    x->fv = rvol_open(json_str(json_get(f, "root"), ""), json_str(json_get(f, "key"), ""), cache, json_num(json_get(f, "um"), 0));
    return x->cv && x->fv ? 0 : -1;
}
static void ctx_close(rctx *x) { rvol_close(x->cv); rvol_close(x->fv); free(x->D.d); json_free(x->j); }

/* ---------- reg pair ---------- */
/* landmark residuals of one transform file: rms / max of |M from - to| and of a least-squares affine refit (to voxels) */
static void lm_report(const json *t, const raff *M, int inv, const char *name) {
    const json *fr = json_get(t, "from_landmarks"), *to = json_get(t, "to_landmarks");
    if (!fr) { fr = json_get(t, "moving_landmarks"); to = json_get(t, "fixed_landmarks"); }
    if (!fr || !to || fr->n != to->n || fr->n == 0) { fprintf(stderr, "  %s: no landmarks\n", name); return; }
    const int n = (int)fr->n; double *c = malloc(n * 3 * sizeof *c), *v = malloc(n * 3 * sizeof *v), *w = malloc(n * sizeof *w);
    double rs = 0, mx = 0;
    for (int i = 0; i < n; i++) {
        double p[3], q[3], r[3];
        for (int d = 0; d < 3; d++) { p[d] = json_num(json_at(json_at(fr, i), d), 0); q[d] = json_num(json_at(json_at(to, i), d), 0); }
        raff_apply(M, p, r); double e = 0; for (int d = 0; d < 3; d++) e += (r[d] - q[d]) * (r[d] - q[d]);
        rs += e; mx = fmax(mx, sqrt(e));
        for (int d = 0; d < 3; d++) { c[3 * i + d] = p[d]; v[3 * i + d] = q[d]; } w[i] = 1;
    }
    double F[3][4]; double frs = -1;
    if (n >= 4 && !reg_fit_affine(n, c, v, w, 0, F)) {
        raff fa; memcpy(fa.m, F, sizeof F); frs = 0;
        for (int i = 0; i < n; i++) { double r[3]; raff_apply(&fa, c + 3 * i, r); for (int d = 0; d < 3; d++) frs += (r[d] - v[3 * i + d]) * (r[d] - v[3 * i + d]); }
        frs = sqrt(frs / n);
    }
    fprintf(stderr, "  %s%s: %d landmarks, |M from - to| rms %.3f max %.3f (to voxels); LS affine refit rms %.3f\n", name, inv ? " (inverted)" : "", n, sqrt(rs / n), mx, frs);
    free(c); free(v); free(w);
}
static void aff_describe(const raff *A, double um_f, double um_c) {
    /* polar decomposition-lite: column norms (scales) and the rotation angle of M / mean scale */
    double s[3]; for (int j = 0; j < 3; j++) s[j] = sqrt(A->m[0][j] * A->m[0][j] + A->m[1][j] * A->m[1][j] + A->m[2][j] * A->m[2][j]);
    const double ms = cbrt(fabs(raff_det(A)));
    const double tr = (A->m[0][0] + A->m[1][1] + A->m[2][2]) / ms, ang = acos(fmax(-1, fmin(1, (tr - 1) / 2))) * 180 / M_PI;
    fprintf(stderr, "  A fine->coarse: column scales %.6f %.6f %.6f (expected %.6f), det %.6g, rotation ~%.2f deg\n", s[0], s[1], s[2], um_c > 0 ? um_f / um_c : 0, raff_det(A), ang);
}
static int cmd_pair(int argc, char **argv) {
    const char *out = opt(argc, argv, "-o", nullptr);
    const char *cr = opt(argc, argv, "--coarse-root", nullptr), *ck = opt(argc, argv, "--coarse", nullptr), *fr = opt(argc, argv, "--fine-root", cr), *fk = opt(argc, argv, "--fine", nullptr);
    if (!out || !cr || !ck || !fk) { fprintf(stderr, "usage: ufsm reg pair --coarse-root R --coarse KEY [--fine-root R] --fine KEY [--coarse-um U] [--fine-um U] (--tr FILE[:inv] ...| --A a,..,l | --identity) [--name N] -o pair.json\n"
        "  --tr files compose in order, each mapping the previous result onward (first one applied to fine points); :inv inverts one; the result must map fine L0 xyz -> coarse L0 xyz\n"); return 2; }
    raff A = raff_ident();
    for (int i = 2; i + 1 < argc; i++) {
        if (strcmp(argv[i], "--tr")) continue;
        char fn[1024]; snprintf(fn, sizeof fn, "%s", argv[i + 1]); int inv = 0; char *c = strrchr(fn, ':');
        if (c && !strcmp(c, ":inv")) { *c = 0; inv = 1; }
        json *t = load_json(fn); if (!t) return 1;
        const json *mj = json_get(t, "transformation_matrix"); if (!mj) mj = json_get(t, "matrix");
        raff M; if (get_aff(mj, &M)) { fprintf(stderr, "reg pair: %s: no 3x4 matrix\n", fn); return 1; }
        lm_report(t, &M, inv, fn);
        A = raff_mul(inv ? raff_inv(M) : M, A);
        json_free(t);
    }
    const char *as = opt(argc, argv, "--A", nullptr);
    if (as) { double v[12]; if (sscanf(as, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf", v, v + 1, v + 2, v + 3, v + 4, v + 5, v + 6, v + 7, v + 8, v + 9, v + 10, v + 11) != 12) { fprintf(stderr, "reg pair: --A needs 12 numbers\n"); return 2; } for (int k = 0; k < 12; k++) A.m[k / 4][k % 4] = v[k]; }
    double cu = atof(opt(argc, argv, "--coarse-um", "0")), fu = atof(opt(argc, argv, "--fine-um", "0"));
    { rvol *c = rvol_open(cr, ck, nullptr, cu), *f = rvol_open(fr, fk, nullptr, fu); if (!c || !f) return 1;
      cu = c->um; fu = f->um;
      fprintf(stderr, "  coarse %s %.4f um %lld x %lld x %lld (%d levels); fine %s %.4f um %lld x %lld x %lld (%d levels)\n", ck, cu, (long long)c->shape[0][0], (long long)c->shape[0][1], (long long)c->shape[0][2], c->nl,
              fk, fu, (long long)f->shape[0][0], (long long)f->shape[0][1], (long long)f->shape[0][2], f->nl);
      aff_describe(&A, fu, cu);
      /* overlap: the fine volume's corners in coarse voxels */
      double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
      for (int k = 0; k < 8; k++) { const double p[3] = {k & 1 ? f->shape[0][2] - 1 : 0, k & 2 ? f->shape[0][1] - 1 : 0, k & 4 ? f->shape[0][0] - 1 : 0}; double q[3]; raff_apply(&A, p, q); for (int d = 0; d < 3; d++) { lo[d] = fmin(lo[d], q[d]); hi[d] = fmax(hi[d], q[d]); } }
      fprintf(stderr, "  fine bbox in coarse voxels: x %.0f..%.0f y %.0f..%.0f z %.0f..%.0f (coarse shape x %lld y %lld z %lld)\n", lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], (long long)c->shape[0][2], (long long)c->shape[0][1], (long long)c->shape[0][0]);
      rvol_close(c); rvol_close(f); }
    FILE *o = fopen(out, "w"); if (!o) { perror(out); return 1; }
    fprintf(o, "{\"name\":\"%s\",\n \"coarse\":{\"root\":\"%s\",\"key\":\"%s\",\"um\":%.6g},\n \"fine\":{\"root\":\"%s\",\"key\":\"%s\",\"um\":%.6g},\n \"A\":", opt(argc, argv, "--name", "pair"), cr, ck, cu, fr, fk, fu);
    put_aff(o, &A);
    fprintf(o, ",\n \"A_note\":\"fine level-0 voxel (x,y,z) -> coarse level-0 voxel (x,y,z); fine(c) = A^-1 (c + D(c))\"}\n");
    fclose(o);
    return 0;
}

/* ---------- reg boxes ---------- */
typedef struct { rmeas q; rbox *bx; int n; atomic_int next, done; time_t t0; } bjob;
static void *box_worker(void *p) {
    bjob *j = p;
    for (;;) {
        const int i = atomic_fetch_add(&j->next, 1); if (i >= j->n) break;
        reg_measure(&j->q, &j->bx[i]);
        const int d = atomic_fetch_add(&j->done, 1) + 1;
        if (d % 50 == 0) fprintf(stderr, "  boxes %d / %d (%lds)\n", d, j->n, (long)(time(nullptr) - j->t0));
    }
    return nullptr;
}
static int cmd_boxes(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: ufsm reg boxes <map.json> <out.jsonl> [--lc 0] [--B 128] [--R 4] [--spacing S (coarse L0 voxels)] [--offset 0] [--metric dog|ncc|grad] [--hp 4] [--lp 0.7]\n"
        "  [--ss auto|1|2] [--minfrac 0.95] [--occ-min 0.9] [--max 3000] [--threads 12] [--cache DIR] [--zr z0,z1 (coarse L0)] [--iters 4] [--centers FILE (x y z per line)]\n"); return 2; }
    rctx x; const char *cache = opt(argc, argv, "--cache", nullptr);
    if (ctx_open(&x, argv[3], cache, 1)) return 1;
    rmeas q = {0}; q.cv = x.cv; q.fv = x.fv; q.m = &x.m;
    q.lc = atoi(opt(argc, argv, "--lc", "0")); q.B = atoi(opt(argc, argv, "--B", "128")); q.R = atoi(opt(argc, argv, "--R", "4"));
    const char *mt = opt(argc, argv, "--metric", "dog"); q.metric = !strcmp(mt, "ncc") ? REG_NCC : !strcmp(mt, "grad") ? REG_GRAD : REG_DOG;
    q.hp = atof(opt(argc, argv, "--hp", "4")); q.lp = atof(opt(argc, argv, "--lp", "0.7")); q.minfrac = atof(opt(argc, argv, "--minfrac", "0.95")); q.iters = atoi(opt(argc, argv, "--iters", "4"));
    const double cum = x.cv->um * (1 << q.lc);
    q.lf = rvol_level_for(x.fv, cum);
    const double ratio = cum / (x.fv->um * (1 << q.lf));
    const char *ss = opt(argc, argv, "--ss", "auto"); q.ss = !strcmp(ss, "auto") ? (ratio > 1.3 ? 2 : 1) : atoi(ss);
    fprintf(stderr, "reg boxes: coarse level %d (%.3f um), fine level %d (%.3f um, ratio %.3f, ss %d), box %d, radius %d, metric %s hp %.2g lp %.2g\n", q.lc, cum, q.lf, x.fv->um * (1 << q.lf), ratio, q.ss, q.B, q.R, mt, q.hp, q.lp);
    const int nt = atoi(opt(argc, argv, "--threads", "12")), mx = atoi(opt(argc, argv, "--max", "3000"));
    const double box0 = q.B * (double)(1 << q.lc);   /* box size, coarse L0 voxels */
    const double S = atof(opt(argc, argv, "--spacing", "0")) > 0 ? atof(opt(argc, argv, "--spacing", "0")) : 2 * box0, off = atof(opt(argc, argv, "--offset", "0"));
    double *cen = nullptr; int nc = 0, cap = 0;
    const char *cf = opt(argc, argv, "--centers", nullptr);
    if (cf) {
        FILE *f = fopen(cf, "r"); double a, b, c;
        while (f && fscanf(f, "%lf %lf %lf", &a, &b, &c) == 3) { if (nc == cap) cen = realloc(cen, (cap = cap ? 2 * cap : 1024) * 3 * sizeof *cen); cen[3 * nc] = a; cen[3 * nc + 1] = b; cen[3 * nc + 2] = c; nc++; }
        if (f) fclose(f);
    } else {
        /* overlap bbox in coarse L0 */
        const int64_t *fs = x.fv->shape[0], *cs = x.cv->shape[0];
        double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
        for (int k = 0; k < 8; k++) { const double p[3] = {k & 1 ? fs[2] - 1 : 0, k & 2 ? fs[1] - 1 : 0, k & 4 ? fs[0] - 1 : 0}; double qq[3]; rmap_f2c(&x.m, p, qq); for (int d = 0; d < 3; d++) { lo[d] = fmin(lo[d], qq[d]); hi[d] = fmax(hi[d], qq[d]); } }
        for (int d = 0; d < 3; d++) { lo[d] = fmax(lo[d], 0); hi[d] = fmin(hi[d], cs[2 - d] - 1); }
        const char *zr = opt(argc, argv, "--zr", nullptr);
        if (zr) { double a, b; if (sscanf(zr, "%lf,%lf", &a, &b) == 2) { lo[2] = fmax(lo[2], a); hi[2] = fmin(hi[2], b); } }
        if (lo[0] > hi[0] || lo[1] > hi[1] || lo[2] > hi[2]) { fprintf(stderr, "reg boxes: no overlap\n"); return 1; }
        /* occupancy levels */
        int loc = 0; while (loc + 1 < x.cv->nl && (double)(1 << (loc + 1)) <= box0 / 4) loc++;
        int64_t oo[3], on[3];
        for (;;) {
            const double f = (double)(1 << loc);
            for (int d = 0; d < 3; d++) { oo[2 - d] = (int64_t)floor((lo[d] + 0.5) / f - 0.5) - 1; on[2 - d] = (int64_t)ceil((hi[d] + 0.5) / f - 0.5) + 2 - oo[2 - d]; }
            if ((double)on[0] * on[1] * on[2] < 1.2e9 || loc + 1 >= x.cv->nl) break;
            loc++;
        }
        const int lof = x.fv->nl - 1; const int64_t *fl = x.fv->shape[lof];
        fprintf(stderr, "reg boxes: overlap x %.0f..%.0f y %.0f..%.0f z %.0f..%.0f; occupancy coarse level %d (%lld x %lld x %lld), fine level %d (%lld x %lld x %lld)\n", lo[0], hi[0], lo[1], hi[1], lo[2], hi[2],
                loc, (long long)on[0], (long long)on[1], (long long)on[2], lof, (long long)fl[0], (long long)fl[1], (long long)fl[2]);
        uint8_t *co = malloc((size_t)on[0] * on[1] * on[2]), *fo = malloc((size_t)fl[0] * fl[1] * fl[2]);
        const int64_t z0[3] = {0, 0, 0};
        if (rvol_read_nt(x.cv, loc, oo, on, co, nt) || rvol_read_nt(x.fv, lof, z0, fl, fo, nt)) { fprintf(stderr, "reg boxes: occupancy read failed\n"); return 1; }
        const double om = atof(opt(argc, argv, "--occ-min", "0.9")), fcl = (double)(1 << loc), ffl = (double)(1 << lof);
        for (double cz = lo[2] + off + S / 2; cz <= hi[2]; cz += S) for (double cy = lo[1] + off + S / 2; cy <= hi[1]; cy += S) for (double cx = lo[0] + off + S / 2; cx <= hi[0]; cx += S) {
            int nco = 0, nfo = 0, tot = 0;
            for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++) for (int c = 0; c < 4; c++) {
                const double p[3] = {cx + (c - 1.5) / 4 * box0, cy + (b - 1.5) / 4 * box0, cz + (a - 1.5) / 4 * box0}; tot++;
                const int64_t iz = (int64_t)llround((p[2] + 0.5) / fcl - 0.5) - oo[0], iy = (int64_t)llround((p[1] + 0.5) / fcl - 0.5) - oo[1], ix = (int64_t)llround((p[0] + 0.5) / fcl - 0.5) - oo[2];
                if (iz >= 0 && iy >= 0 && ix >= 0 && iz < on[0] && iy < on[1] && ix < on[2] && co[((size_t)iz * on[1] + iy) * on[2] + ix]) nco++;
                double fp[3]; rmap_c2f(&x.m, p, fp);
                const int64_t jz = (int64_t)llround((fp[2] + 0.5) / ffl - 0.5), jy = (int64_t)llround((fp[1] + 0.5) / ffl - 0.5), jx = (int64_t)llround((fp[0] + 0.5) / ffl - 0.5);
                if (jz >= 0 && jy >= 0 && jx >= 0 && jz < fl[0] && jy < fl[1] && jx < fl[2] && fo[((size_t)jz * fl[1] + jy) * fl[2] + jx]) nfo++;
            }
            if (nco < om * tot || nfo < om * tot) continue;
            if (nc == cap) cen = realloc(cen, (cap = cap ? 2 * cap : 1024) * 3 * sizeof *cen);
            cen[3 * nc] = cx; cen[3 * nc + 1] = cy; cen[3 * nc + 2] = cz; nc++;
        }
        free(co); free(fo);
    }
    int n = nc;
    if (mx > 0 && n > mx) {   /* even subsample */
        for (int i = 0; i < mx; i++) { const int k = (int)((double)i * nc / mx); memmove(cen + 3 * i, cen + 3 * k, 3 * sizeof *cen); }
        n = mx;
    }
    fprintf(stderr, "reg boxes: %d candidate boxes (of %d on papyrus in the overlap)\n", n, nc);
    bjob j = {.q = q, .n = n}; j.bx = calloc(n, sizeof *j.bx); j.t0 = time(nullptr);
    for (int i = 0; i < n; i++) memcpy(j.bx[i].c, cen + 3 * i, sizeof j.bx[i].c);
    pthread_t th[64]; const int T = nt < 1 ? 1 : nt > 64 ? 64 : nt;
    for (int t = 0; t < T; t++) pthread_create(&th[t], nullptr, box_worker, &j);
    for (int t = 0; t < T; t++) pthread_join(th[t], nullptr);
    FILE *o = fopen(argv[4], "w"); if (!o) { perror(argv[4]); return 1; }
    int cnt[7] = {0};
    for (int i = 0; i < n; i++) {
        const rbox *b = &j.bx[i]; cnt[b->status]++;
        fprintf(o, "{\"i\":%d,\"c\":[%.2f,%.2f,%.2f],\"D0\":[%.4f,%.4f,%.4f],\"s\":[%.4f,%.4f,%.4f],\"ncc\":%.4f,\"ncc0\":%.4f,\"second\":%.4f,\"curv\":%.5f,\"cstd\":%.3f,\"gstd\":%.3f,\"cfrac\":%.3f,\"gfrac\":%.3f,\"status\":\"%s\",\"lc\":%d,\"um\":%.6g}\n",
                i, b->c[0], b->c[1], b->c[2], b->D0[0], b->D0[1], b->D0[2], b->s[0], b->s[1], b->s[2], b->ncc, b->ncc0, b->second, b->curv, b->cstd, b->gstd, b->cfrac, b->gfrac, reg_status_name[b->status], q.lc, x.cv->um);
    }
    fclose(o);
    fprintf(stderr, "reg boxes: %d boxes in %lds:", n, (long)(time(nullptr) - j.t0));
    for (int k = 0; k < 7; k++) if (cnt[k]) fprintf(stderr, " %s %d", reg_status_name[k], cnt[k]);
    fprintf(stderr, "\n");
    free(j.bx); free(cen); ctx_close(&x);
    return 0;
}

/* ---------- box results ---------- */
typedef struct { double c[3], D0[3], s[3]; double ncc, ncc0, second, curv; int status, i; double um; } brec;
static int read_boxes(const char *list, brec **out) {
    int n = 0, cap = 0; brec *r = nullptr; char lst[4096]; snprintf(lst, sizeof lst, "%s", list);
    for (char *tok = strtok(lst, ","); tok; tok = strtok(nullptr, ",")) {
        FILE *f = fopen(tok, "r"); if (!f) { fprintf(stderr, "reg: cannot read %s\n", tok); continue; }
        char line[4096];
        while (fgets(line, sizeof line, f)) {
            json *j = json_parse(line, strlen(line)); if (!j) continue;
            if (n == cap) r = realloc(r, (cap = cap ? 2 * cap : 1024) * sizeof *r);
            brec *b = &r[n++]; memset(b, 0, sizeof *b);
            for (int d = 0; d < 3; d++) { b->c[d] = json_num(json_at(json_get(j, "c"), d), 0); b->D0[d] = json_num(json_at(json_get(j, "D0"), d), 0); b->s[d] = json_num(json_at(json_get(j, "s"), d), 0); }
            b->ncc = json_num(json_get(j, "ncc"), 0); b->ncc0 = json_num(json_get(j, "ncc0"), 0); b->second = json_num(json_get(j, "second"), 0); b->curv = json_num(json_get(j, "curv"), 0);
            b->i = (int)json_num(json_get(j, "i"), 0); b->um = json_num(json_get(j, "um"), 1);
            const char *st = json_str(json_get(j, "status"), "read"); b->status = 5;
            for (int k = 0; k < 7; k++) if (!strcmp(st, reg_status_name[k])) b->status = k;
            json_free(j);
        }
        fclose(f);
    }
    *out = r; return n;
}
typedef struct { double ncc_min, sec_max, curv_min; } accept_t;
static int accepted(const brec *b, const accept_t *a) {
    return b->status == 0 && b->ncc >= a->ncc_min && b->second <= a->sec_max * b->ncc && b->curv >= a->curv_min;
}
static int dcmp(const void *a, const void *b) { const double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }
static void quant(double *v, int n, double q[5]) {   /* p10 p50 p90 p99 mean */
    if (!n) { for (int k = 0; k < 5; k++) q[k] = NAN; return; }
    qsort(v, n, sizeof *v, dcmp); double m = 0; for (int i = 0; i < n; i++) m += v[i];
    const double ps[4] = {0.1, 0.5, 0.9, 0.99}; for (int k = 0; k < 4; k++) q[k] = v[(int)fmin(n - 1, floor(ps[k] * (n - 1) + 0.5))];
    q[4] = m / n;
}
static void put_q(FILE *o, const char *name, double *v, int n, double um, int last) {
    double q[5]; quant(v, n, q);
    if (o) fprintf(o, "\"%s\":{\"n\":%d,\"p10_vox\":%.4f,\"p50_vox\":%.4f,\"p90_vox\":%.4f,\"p99_vox\":%.4f,\"mean_vox\":%.4f,\"p50_um\":%.3f,\"p90_um\":%.3f}%s", name, n, q[0], q[1], q[2], q[3], q[4], q[1] * um, q[2] * um, last ? "" : ",");
    fprintf(stderr, "  %-26s n %5d  |d| p10 %.3f p50 %.3f p90 %.3f p99 %.3f mean %.3f vox  (p50 %.2f um, p90 %.2f um)\n", name, n, q[0], q[1], q[2], q[3], q[4], q[1] * um, q[2] * um);
}

/* ---------- reg stats ---------- */
static int cmd_stats(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm reg stats <boxes.jsonl>[,more] [--ncc-min 0.3] [--sec-max 0.9] [--curv-min 0.002] [--json OUT]\n"); return 2; }
    brec *b; const int n = read_boxes(argv[3], &b);
    accept_t a = {atof(opt(argc, argv, "--ncc-min", "0.3")), atof(opt(argc, argv, "--sec-max", "0.9")), atof(opt(argc, argv, "--curv-min", "0.002"))};
    int cnt[7] = {0}, pap = 0, acc = 0, good = 0; double *m = malloc((n + 1) * sizeof *m), *nc = malloc((n + 1) * sizeof *nc), *n0 = malloc((n + 1) * sizeof *n0); int k = 0;
    const double um = n ? b[0].um * (1 << 0) : 1;
    for (int i = 0; i < n; i++) {
        cnt[b[i].status]++;
        if (b[i].status == 1 || b[i].status == 2 || b[i].status == 5 || b[i].status == 6) continue;
        pap++;
        if (!accepted(&b[i], &a)) continue;
        acc++; const double e = sqrt(b[i].s[0] * b[i].s[0] + b[i].s[1] * b[i].s[1] + b[i].s[2] * b[i].s[2]);
        good += e < 0.25; nc[k] = b[i].ncc; n0[k] = b[i].ncc0; m[k++] = e;
    }
    fprintf(stderr, "reg stats %s: %d boxes:", argv[3], n); for (int s = 0; s < 7; s++) if (cnt[s]) fprintf(stderr, " %s %d", reg_status_name[s], cnt[s]);
    fprintf(stderr, "\n  papyrus boxes %d, confident %d (%.1f%%), |s| < 0.25 vox: %d (%.1f%% of papyrus)\n", pap, acc, 100.0 * acc / fmax(1, pap), good, 100.0 * good / fmax(1, pap));
    FILE *o = opt(argc, argv, "--json", nullptr) ? fopen(opt(argc, argv, "--json", nullptr), "w") : nullptr;
    if (o) fprintf(o, "{\"boxes\":%d,\"papyrus\":%d,\"confident\":%d,\"under_quarter\":%d,\"coverage\":%.4f,\"um\":%.6g,", n, pap, acc, good, (double)good / fmax(1, pap), um);
    double qn[5], q0[5]; double *t = malloc((k + 1) * sizeof *t);
    memcpy(t, nc, k * sizeof *t); quant(t, k, qn); memcpy(t, n0, k * sizeof *t); quant(t, k, q0);
    fprintf(stderr, "  NCC at the peak p50 %.3f (p10 %.3f); NCC at the prior map p50 %.3f (p10 %.3f)\n", qn[1], qn[0], q0[1], q0[0]);
    put_q(o, "shift", m, k, um, 1);
    if (o) { fprintf(o, ",\"ncc_p50\":%.4f,\"ncc0_p50\":%.4f}\n", qn[1], q0[1]); fclose(o); }
    free(t); free(m); free(nc); free(n0); free(b);
    return 0;
}

/* ---------- reg fit ---------- */
static int cmd_fit(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: ufsm reg fit <map.json> <boxes.jsonl>[,more] <outdir> [--h 256 (coarse L0 voxels)] [--lam 1] [--huber 0.5] [--ncc-min 0.3] [--sec-max 0.9] [--curv-min 0.002]\n"
        "  [--holdout 5] [--affine-only] [--margin 1]\n"); return 2; }
    rctx x; if (ctx_open(&x, argv[3], nullptr, 0)) return 1;
    brec *b; const int n = read_boxes(argv[4], &b);
    const char *od = argv[5]; mkdir(od, 0755);
    accept_t a = {atof(opt(argc, argv, "--ncc-min", "0.3")), atof(opt(argc, argv, "--sec-max", "0.9")), atof(opt(argc, argv, "--curv-min", "0.002"))};
    const double h = atof(opt(argc, argv, "--h", "256")), lam = atof(opt(argc, argv, "--lam", "1")), hub = atof(opt(argc, argv, "--huber", "0.5"));
    const int hold = atoi(opt(argc, argv, "--holdout", "5")), aff_only = has(argc, argv, "--affine-only"), margin = atoi(opt(argc, argv, "--margin", "1"));
    double *c = malloc(n * 3 * sizeof *c), *v = malloc(n * 3 * sizeof *v), *w = malloc(n * sizeof *w), *wt = malloc(n * sizeof *wt);
    int na = 0; const double um = n ? b[0].um : 1;
    for (int i = 0; i < n; i++) {
        for (int d = 0; d < 3; d++) { c[3 * i + d] = b[i].c[d]; v[3 * i + d] = b[i].D0[d] + b[i].s[d]; }
        w[i] = accepted(&b[i], &a); na += w[i] > 0;
    }
    fprintf(stderr, "reg fit: %d boxes, %d accepted (ncc >= %.2f, second <= %.2f ncc, curv >= %.4f)\n", n, na, a.ncc_min, a.sec_max, a.curv_min);
    if (na < 12) { fprintf(stderr, "reg fit: too few accepted boxes\n"); return 1; }
    /* residual model: v ~ M c + t + u(c) */
    double *pred = malloc(n * 3 * sizeof *pred);
    #define MODEL_FIT(WEIGHTS, FIELD, MOUT) do { \
        if (reg_fit_affine(n, c, v, WEIGHTS, hub, MOUT)) return 1; \
        raff ma_; memcpy(ma_.m, MOUT, sizeof ma_.m); \
        double *r_ = malloc(n * 3 * sizeof *r_); \
        for (int i_ = 0; i_ < n; i_++) { double p_[3]; raff_apply(&ma_, c + 3 * i_, p_); for (int d_ = 0; d_ < 3; d_++) r_[3 * i_ + d_] = v[3 * i_ + d_] - p_[d_]; } \
        memset(FIELD, 0, sizeof *(FIELD)); \
        if (!aff_only && reg_fit_field(n, c, r_, WEIGHTS, h, lam, hub, margin, FIELD)) return 1; \
        for (int i_ = 0; i_ < n; i_++) { double p_[3], u_[3]; raff_apply(&ma_, c + 3 * i_, p_); rfield_eval(FIELD, c + 3 * i_, u_); for (int d_ = 0; d_ < 3; d_++) pred[3 * i_ + d_] = p_[d_] + u_[d_]; } \
        free(r_); } while (0)
    /* cross-validation: every hold-th accepted box held out */
    double *eb = malloc(n * sizeof *eb), *ein = malloc(n * sizeof *ein), *eho = malloc(n * sizeof *eho), *eaff = malloc(n * sizeof *eaff); int nb = 0, nin = 0, nho = 0, naf = 0;
    for (int i = 0; i < n; i++) if (w[i] > 0) eb[nb++] = sqrt(v[3 * i] * v[3 * i] + v[3 * i + 1] * v[3 * i + 1] + v[3 * i + 2] * v[3 * i + 2]);
    double Mcv[3][4]; rfield Fcv; int k = 0;
    if (hold > 1) {
        for (int i = 0; i < n; i++) wt[i] = w[i] > 0 && (k++ % hold) != hold / 2 ? w[i] : 0;
        MODEL_FIT(wt, &Fcv, Mcv);
        k = 0;
        for (int i = 0; i < n; i++) if (w[i] > 0 && (k++ % hold) == hold / 2) { double e = 0; for (int d = 0; d < 3; d++) e += (v[3 * i + d] - pred[3 * i + d]) * (v[3 * i + d] - pred[3 * i + d]); eho[nho++] = sqrt(e); }
        free(Fcv.d);
        /* affine-only holdout */
        raff ma; double Ma[3][4]; if (!reg_fit_affine(n, c, v, wt, hub, Ma)) { memcpy(ma.m, Ma, sizeof Ma); k = 0;
            for (int i = 0; i < n; i++) if (w[i] > 0 && (k++ % hold) == hold / 2) { double p[3], e = 0; raff_apply(&ma, c + 3 * i, p); for (int d = 0; d < 3; d++) e += (v[3 * i + d] - p[d]) * (v[3 * i + d] - p[d]); eaff[naf++] = sqrt(e); } }
    }
    double M[3][4]; rfield F;
    MODEL_FIT(w, &F, M);
    for (int i = 0; i < n; i++) if (w[i] > 0) { double e = 0; for (int d = 0; d < 3; d++) e += (v[3 * i + d] - pred[3 * i + d]) * (v[3 * i + d] - pred[3 * i + d]); ein[nin++] = sqrt(e); }
    /* fold the affine correction into A: A' = K^-1 A with K = [(I + M) | t]; field D' = (I + M)^-1 u */
    raff K; memcpy(K.m, M, sizeof M); for (int d = 0; d < 3; d++) K.m[d][d] += 1;
    raff Ki = raff_inv(K), An = raff_mul(Ki, x.m.A);
    raff Kl = Ki; for (int d = 0; d < 3; d++) Kl.m[d][3] = 0;
    const size_t NN = F.d ? (size_t)F.n[0] * F.n[1] * F.n[2] : 0;
    for (size_t i = 0; i < NN; i++) { double u[3] = {F.d[3 * i], F.d[3 * i + 1], F.d[3 * i + 2]}, r[3]; raff_apply(&Kl, u, r); for (int d = 0; d < 3; d++) F.d[3 * i + d] = (float)r[d]; }
    double umax = 0; for (size_t i = 0; i < NN * 3; i++) umax = fmax(umax, fabs(F.d[i]));
    char fp[2100]; snprintf(fp, sizeof fp, "%s/field.f32", od);
    if (NN) { FILE *f = fopen(fp, "wb"); fwrite(F.d, sizeof(float), NN * 3, f); fclose(f); }
    /* reg.json: the input map's volumes + the refined A + the field + stats */
    snprintf(fp, sizeof fp, "%s/reg.json", od);
    FILE *o = fopen(fp, "w");
    const json *cj = json_get(x.j, "coarse"), *fj = json_get(x.j, "fine");
    fprintf(o, "{\"name\":\"%s\",\n \"coarse\":{\"root\":\"%s\",\"key\":\"%s\",\"um\":%.6g},\n \"fine\":{\"root\":\"%s\",\"key\":\"%s\",\"um\":%.6g},\n \"A\":", json_str(json_get(x.j, "name"), "pair"),
            json_str(json_get(cj, "root"), ""), json_str(json_get(cj, "key"), ""), json_num(json_get(cj, "um"), 0), json_str(json_get(fj, "root"), ""), json_str(json_get(fj, "key"), ""), json_num(json_get(fj, "um"), 0));
    put_aff(o, &An); fprintf(o, ",\n \"A_inv\":"); { raff t = raff_inv(An); put_aff(o, &t); }
    fprintf(o, ",\n \"A_input\":"); put_aff(o, &x.m.A);
    { const json *ap = json_get(x.j, "A_published"); raff P; if (ap && !get_aff(ap, &P)) { fprintf(o, ",\n \"A_published\":"); put_aff(o, &P); } else { fprintf(o, ",\n \"A_published\":"); put_aff(o, &x.m.A); } }
    fprintf(o, ",\n \"A_note\":\"A: fine level-0 voxel (x,y,z) -> coarse level-0 voxel (x,y,z), refined affine; fine(c) = A^-1 (c + D(c)), D the field (coarse L0 voxels, trilinear, clamped outside the grid)\"");
    if (NN) fprintf(o, ",\n \"field\":{\"file\":\"field.f32\",\"n\":[%d,%d,%d],\"o\":[%.4f,%.4f,%.4f],\"h\":%.4f,\"layout\":\"float32 [nz][ny][nx][3] x,y,z displacement in coarse L0 voxels; node (k,j,i) at o + h (i,j,k)\",\"max_abs\":%.4f}",
                    F.n[0], F.n[1], F.n[2], F.o[0], F.o[1], F.o[2], F.h, umax);
    fprintf(o, ",\n \"fit\":{\"boxes\":\"%s\",\"n\":%d,\"accepted\":%d,\"h\":%.4g,\"lam\":%.4g,\"huber\":%.4g,\"ncc_min\":%.3g,\"sec_max\":%.3g,\"curv_min\":%.4g,\"holdout\":%d,\"affine_only\":%d,\"coarse_um\":%.6g,\n  \"affine_correction\":",
            argv[4], n, na, h, lam, hub, a.ncc_min, a.sec_max, a.curv_min, hold, aff_only, um);
    { raff t; memcpy(t.m, M, sizeof M); put_aff(o, &t); }
    fprintf(o, ",\n  ");
    fprintf(stderr, "reg fit: residual |d| (coarse voxels, %.4g um each)\n", um);
    put_q(o, "before", eb, nb, um, 0); fprintf(o, "\n  ");
    put_q(o, "holdout_affine", eaff, naf, um, 0); fprintf(o, "\n  ");
    put_q(o, "holdout_field", eho, nho, um, 0); fprintf(o, "\n  ");
    put_q(o, "insample_field", ein, nin, um, 1);
    fprintf(o, "}}\n"); fclose(o);
    fprintf(stderr, "reg fit: field %d x %d x %d nodes (h %.0f), max |D| %.3f vox; wrote %s\n", F.n[0], F.n[1], F.n[2], h, umax, fp);
    /* per-box residuals */
    snprintf(fp, sizeof fp, "%s/residuals.jsonl", od); o = fopen(fp, "w"); k = 0;
    for (int i = 0; i < n; i++) {
        const int ho = w[i] > 0 && hold > 1 && (k++ % hold) == hold / 2;
        fprintf(o, "{\"i\":%d,\"c\":[%.2f,%.2f,%.2f],\"v\":[%.4f,%.4f,%.4f],\"pred\":[%.4f,%.4f,%.4f],\"used\":%d,\"holdout\":%d,\"ncc\":%.4f,\"status\":\"%s\"}\n", b[i].i, c[3 * i], c[3 * i + 1], c[3 * i + 2],
                v[3 * i], v[3 * i + 1], v[3 * i + 2], pred[3 * i], pred[3 * i + 1], pred[3 * i + 2], w[i] > 0, ho, b[i].ncc, reg_status_name[b[i].status]);
    }
    fclose(o);
    free(F.d); free(c); free(v); free(w); free(wt); free(pred); free(eb); free(ein); free(eho); free(eaff); free(b); ctx_close(&x);
    return 0;
#undef MODEL_FIT
}

/* ---------- reg map: points / boxes between the two volumes ---------- */
static int cmd_map(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: ufsm reg map <map.json> [--f2c] [--um] x,y,z [x,y,z ...]   (coarse L0 voxels -> fine L0 voxels; --f2c the other way)\n"
        "       ufsm reg map <map.json> [--f2c] --box z,y,x,nz,ny,nx   (the bounding box of the mapped box, zyx like --box elsewhere)\n"); return 2; }
    rctx x; if (ctx_open(&x, argv[3], nullptr, 0)) return 1;
    const int f2c = has(argc, argv, "--f2c");
    const char *bx = opt(argc, argv, "--box", nullptr);
    if (bx) {
        double v[6]; if (sscanf(bx, "%lf,%lf,%lf,%lf,%lf,%lf", v, v + 1, v + 2, v + 3, v + 4, v + 5) != 6) { fprintf(stderr, "reg map: --box z,y,x,nz,ny,nx\n"); return 2; }
        double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
        for (int a = 0; a <= 8; a++) for (int b = 0; b <= 8; b++) for (int c = 0; c <= 8; c++) {   /* 9^3 points: the field is not affine */
            const double p[3] = {v[2] + v[5] * c / 8, v[1] + v[4] * b / 8, v[0] + v[3] * a / 8}; double q[3];
            if (f2c) rmap_f2c(&x.m, p, q); else rmap_c2f(&x.m, p, q);
            for (int d = 0; d < 3; d++) { lo[d] = fmin(lo[d], q[d]); hi[d] = fmax(hi[d], q[d]); }
        }
        printf("%lld,%lld,%lld,%lld,%lld,%lld\n", (long long)floor(lo[2]), (long long)floor(lo[1]), (long long)floor(lo[0]),
               (long long)(ceil(hi[2]) - floor(lo[2]) + 1), (long long)(ceil(hi[1]) - floor(lo[1]) + 1), (long long)(ceil(hi[0]) - floor(lo[0]) + 1));
    }
    for (int i = 3; i < argc; i++) {
        double p[3], q[3];
        if (sscanf(argv[i], "%lf,%lf,%lf", p, p + 1, p + 2) != 3 || argv[i][0] == '-') continue;
        if (i > 3 && !strcmp(argv[i - 1], "--box")) continue;
        if (f2c) rmap_f2c(&x.m, p, q); else rmap_c2f(&x.m, p, q);
        printf("%.4f,%.4f,%.4f\n", q[0], q[1], q[2]);
    }
    ctx_close(&x);
    return 0;
}

/* ---------- label warp: coarse-scan labels resampled into the fine scan's grid ----------
   ufsm reg warp <reg.json> <label-root> <label-key> <label-um> <out-dir> --level L [--boxes eval_boxes.jsonl] [--fill 255] [--cache DIR]
   The labels (uint8: winding codes, stored phase, ink prob254; on the coarse scan's grid at label-um, origin shared with coarse L0) are
   sampled nearest-neighbour at every voxel of the fine scan's level L: fine level-L voxel -> fine L0 -> coarse L0 (rmap_f2c, evaluated on
   a lattice of 8 voxels and interpolated trilinearly: the field is smooth) -> label voxel. With --boxes (the pair's eval_boxes.jsonl), a
   voxel keeps its label only when the nearest probe box (coarse L0 centre "c") is "certified"; elsewhere, outside the labels and outside
   the map: fill. Output: one array (512^3 shards, lossless) of the fine level-L shape. */
typedef struct { double c[3]; int ok; } rprobe;
static int cmd_warp(int argc, char **argv) {
    if (argc < 8) { fprintf(stderr, "usage: ufsm reg warp <reg.json> <label-root> <label-key> <label-um> <out-dir> --level L [--boxes eval_boxes.jsonl] [--box z,y,x,nz,ny,nx] [--fill 255] [--cache DIR] [--threads 8]\n"); return 2; }
    const int L = atoi(opt(argc, argv, "--level", "-1")), fill = atoi(opt(argc, argv, "--fill", "255")), nt = atoi(opt(argc, argv, "--threads", "8"));
    const double lum = atof(argv[6]);
    if (L < 0 || lum <= 0) { fprintf(stderr, "reg warp: --level L and label-um > 0\n"); return 2; }
    rctx x; if (ctx_open(&x, argv[3], opt(argc, argv, "--cache", nullptr), 1)) return 1;
    if (L >= x.fv->nl) { fprintf(stderr, "reg warp: the fine scan has %d levels\n", x.fv->nl); return 1; }
    const double cum = x.cv->um, k = lum / cum;   /* label voxel = k coarse L0 voxels */
    store *ls = store_open(argv[4]); z3 *lz = ls ? z3_open(ls, argv[5], opt(argc, argv, "--cache", nullptr)) : nullptr;
    if (!lz) { fprintf(stderr, "reg warp: cannot open the labels %s %s\n", argv[4], argv[5]); return 1; }
    const int64_t *lsh = z3_meta_of(lz)->shape;
    rprobe *pr = nullptr; int np = 0;
    const char *bf = opt(argc, argv, "--boxes", nullptr);
    if (bf) {
        FILE *f = fopen(bf, "r"); if (!f) { fprintf(stderr, "reg warp: %s\n", bf); return 1; }
        char line[4096]; int cap = 0;
        while (fgets(line, sizeof line, f)) {
            const char *c = strstr(line, "\"c\":["); if (!c) continue;
            if (np == cap) { cap = cap ? 2 * cap : 1024; pr = realloc(pr, sizeof *pr * cap); }
            if (sscanf(c + 5, "%lf,%lf,%lf", &pr[np].c[0], &pr[np].c[1], &pr[np].c[2]) != 3) continue;
            pr[np].ok = strstr(line, "\"certified\":true") != nullptr; np++;
        }
        fclose(f);
        int nok = 0; for (int i = 0; i < np; i++) nok += pr[i].ok;
        fprintf(stderr, "reg warp: %d probe boxes, %d certified\n", np, nok);
    }
    const int64_t *fs = x.fv->shape[L]; const double sc = (double)(1 << L);
    char attrs[512]; snprintf(attrs, sizeof attrs, "{\"ufsm\":{\"content\":\"labels %s %s warped by %s (nearest)\",\"fill\":%d}}", argv[4], argv[5], argv[3], fill);
    z3w *w = z3w_create(argv[7], fs, 512, 0.f, fill, attrs); if (!w) { fprintf(stderr, "reg warp: %s\n", z3w_error()); return 1; }
    const int S = 512, G = 8, NN = S / G + 1; uint8_t *buf = malloc((size_t)S * S * S); float *nd = malloc(sizeof(float) * 4 * NN * NN * NN);
    long long nset = 0;
    long long bx[6] = {0, 0, 0, fs[0], fs[1], fs[2]};   /* --box z,y,x,nz,ny,nx (fine level L): only the shards meeting it */
    if (opt(argc, argv, "--box", nullptr) && sscanf(opt(argc, argv, "--box", ""), "%lld,%lld,%lld,%lld,%lld,%lld", &bx[0], &bx[1], &bx[2], &bx[3], &bx[4], &bx[5]) != 6) { fprintf(stderr, "reg warp: --box z,y,x,nz,ny,nx\n"); return 2; }
    for (int64_t sz = 0; sz < fs[0]; sz += S) for (int64_t sy = 0; sy < fs[1]; sy += S) for (int64_t sx = 0; sx < fs[2]; sx += S) {
        if (sz + S <= bx[0] || sz >= bx[0] + bx[3] || sy + S <= bx[1] || sy >= bx[1] + bx[4] || sx + S <= bx[2] || sx >= bx[2] + bx[5]) continue;
        /* lattice: coarse L0 xyz of fine level-L points (sz + G k, ...), and its certified flag (1 / 0) */
        double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300}; int anyok = 0;
        for (int a = 0; a < NN; a++) for (int b = 0; b < NN; b++) for (int c = 0; c < NN; c++) {
            const double fz = sz + (double)G * a, fy = sy + (double)G * b, fx = sx + (double)G * c;
            const double f0[3] = {(fx + 0.5) * sc - 0.5, (fy + 0.5) * sc - 0.5, (fz + 0.5) * sc - 0.5}; double q[3];
            rmap_f2c(&x.m, f0, q);
            int ok = 1;
            if (np) { double bd = 1e300; int bi = -1; for (int i = 0; i < np; i++) { const double dx = q[0] - pr[i].c[0], dy = q[1] - pr[i].c[1], dz = q[2] - pr[i].c[2], d = dx * dx + dy * dy + dz * dz; if (d < bd) { bd = d; bi = i; } } ok = bi >= 0 && pr[bi].ok; }
            float *e = nd + 4 * (((size_t)a * NN + b) * NN + c); e[0] = (float)q[0]; e[1] = (float)q[1]; e[2] = (float)q[2]; e[3] = (float)ok;
            if (ok) { anyok = 1; for (int d = 0; d < 3; d++) { lo[d] = fmin(lo[d], q[d]); hi[d] = fmax(hi[d], q[d]); } }
        }
        if (!anyok) continue;
        /* the label block under the mapped points (label voxels, zyx), clipped to the label array */
        int64_t lo_[3], ln_[3];
        for (int d = 0; d < 3; d++) {   /* d: x y z -> label axis 2 - d */
            const int ax = 2 - d; int64_t a0 = (int64_t)floor((lo[d] + 0.5) / k - 0.5) - 2, a1 = (int64_t)ceil((hi[d] + 0.5) / k - 0.5) + 2;
            if (a0 < 0) a0 = 0;
            if (a1 > lsh[ax] - 1) a1 = lsh[ax] - 1;
            lo_[ax] = a0; ln_[ax] = a1 - a0 + 1;
        }
        if (ln_[0] <= 0 || ln_[1] <= 0 || ln_[2] <= 0) continue;
        uint8_t *lb = malloc((size_t)ln_[0] * ln_[1] * ln_[2]);
        if (!lb || z3_read(lz, lo_, ln_, lb, nt)) { fprintf(stderr, "reg warp: label read failed\n"); return 1; }
        memset(buf, fill, (size_t)S * S * S);
        for (int z = 0; z < S && sz + z < fs[0]; z++) {
            const int a = z / G; const float tz = (float)(z % G) / G;
            for (int y = 0; y < S && sy + y < fs[1]; y++) {
                const int b = y / G; const float ty = (float)(y % G) / G;
                for (int xx = 0; xx < S && sx + xx < fs[2]; xx++) {
                    const int c = xx / G; const float tx = (float)(xx % G) / G;
                    float q[4] = {0, 0, 0, 0};
                    for (int o = 0; o < 8; o++) {
                        const int oa = o >> 2, ob = (o >> 1) & 1, oc = o & 1; const float wgt = (oa ? tz : 1 - tz) * (ob ? ty : 1 - ty) * (oc ? tx : 1 - tx);
                        const float *e = nd + 4 * (((size_t)(a + oa) * NN + b + ob) * NN + c + oc); for (int d = 0; d < 4; d++) q[d] += wgt * e[d];
                    }
                    if (q[3] < 0.5f) continue;   /* (nearest probe not certified) */
                    const int64_t li[3] = {(int64_t)lrint((q[2] + 0.5) / k - 0.5) - lo_[0], (int64_t)lrint((q[1] + 0.5) / k - 0.5) - lo_[1], (int64_t)lrint((q[0] + 0.5) / k - 0.5) - lo_[2]};
                    if (li[0] < 0 || li[1] < 0 || li[2] < 0 || li[0] >= ln_[0] || li[1] >= ln_[1] || li[2] >= ln_[2]) continue;
                    const uint8_t v = lb[((size_t)li[0] * ln_[1] + li[1]) * ln_[2] + li[2]];
                    buf[((size_t)z * S + y) * S + xx] = v; nset += v != fill;
                }
            }
        }
        free(lb);
        if (z3w_write_shard(w, sz / S, sy / S, sx / S, buf, nt)) { fprintf(stderr, "reg warp: %s\n", z3w_error()); return 1; }
        fprintf(stderr, "reg warp: shard %lld,%lld,%lld done (%lld labelled voxels so far)\n", (long long)(sz / S), (long long)(sy / S), (long long)(sx / S), nset);
    }
    z3w_close(w); free(buf); free(nd); free(pr); ctx_close(&x);
    fprintf(stderr, "reg warp: %lld labelled voxels -> %s (fine level %d, shape %lld x %lld x %lld)\n", nset, argv[7], L, (long long)fs[0], (long long)fs[1], (long long)fs[2]);
    return 0;
}

int cmd_reg_overlay(int argc, char **argv);
int cmd_reg_init(int argc, char **argv);
int cmd_reg(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: ufsm reg <pair|map|warp|boxes|fit|stats|overlay|init> ...\n"); return 2; }
    store_global_init();
    const char *s = argv[2];
    if (!strcmp(s, "pair")) return cmd_pair(argc, argv);
    if (!strcmp(s, "map")) return cmd_map(argc, argv);
    if (!strcmp(s, "warp")) return cmd_warp(argc, argv);
    if (!strcmp(s, "boxes")) return cmd_boxes(argc, argv);
    if (!strcmp(s, "fit")) return cmd_fit(argc, argv);
    if (!strcmp(s, "stats")) return cmd_stats(argc, argv);
    if (!strcmp(s, "overlay")) return cmd_reg_overlay(argc, argv);
    if (!strcmp(s, "init")) return cmd_reg_init(argc, argv);
    fprintf(stderr, "reg: unknown subcommand %s\n", s);
    return 2;
}
