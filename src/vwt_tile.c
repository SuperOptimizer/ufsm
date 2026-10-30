/* vwt_tile (src/vwt_tile.h): slab tiling of a box, neighbour pairs and their overlap point sets, save / load. */
#include "vwt_tile.h"
#include "json.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { int n; double (*p)[3]; } umbt;
static int umb_load(const char *path, umbt *u) {
    FILE *f = fopen(path, "rb"); if (!f) return -1; fseek(f, 0, SEEK_END); const long L = ftell(f); fseek(f, 0, SEEK_SET);
    char *t = malloc(L + 1); if (fread(t, 1, L, f) != (size_t)L) { fclose(f); free(t); return -1; } fclose(f); t[L] = 0;
    json *J = json_parse(t, L); free(t); const json *cp = J ? json_get(J, "control_points") : nullptr; if (!cp || !cp->n) { json_free(J); return -1; }
    u->n = (int)cp->n; u->p = malloc(cp->n * sizeof *u->p);
    for (size_t i = 0; i < cp->n; i++) { const json *q = json_at(cp, i); u->p[i][0] = json_num(json_get(q, "x"), 0); u->p[i][1] = json_num(json_get(q, "y"), 0); u->p[i][2] = json_num(json_get(q, "z"), 0); }
    for (int i = 1; i < u->n; i++) for (int j = i; j > 0 && u->p[j][2] < u->p[j - 1][2]; j--) { double tt[3]; memcpy(tt, u->p[j], 24); memcpy(u->p[j], u->p[j - 1], 24); memcpy(u->p[j - 1], tt, 24); }
    json_free(J); return 0;
}
static void umb_at(const umbt *u, double z, double *x, double *y) {
    if (z <= u->p[0][2]) { *x = u->p[0][0]; *y = u->p[0][1]; return; }
    for (int i = 1; i < u->n; i++) if (z <= u->p[i][2]) {
        const double t = (z - u->p[i - 1][2]) / fmax(1e-9, u->p[i][2] - u->p[i - 1][2]);
        *x = u->p[i - 1][0] + t * (u->p[i][0] - u->p[i - 1][0]); *y = u->p[i - 1][1] + t * (u->p[i][1] - u->p[i - 1][1]); return; }
    *x = u->p[u->n - 1][0]; *y = u->p[u->n - 1][1];
}
static void cross3(const double *u, const double *v, double *w) { w[0] = u[1] * v[2] - u[2] * v[1]; w[1] = u[2] * v[0] - u[0] * v[2]; w[2] = u[0] * v[1] - u[1] * v[0]; }
/* villawind frame_of */
static void frame(vwt_slab *s, const double dir[3], const double ro[3], int T) {
    const double n = sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
    for (int c = 0; c < 3; c++) s->d[c] = dir[c] / n;
    double ref[3] = {0, 0, 0}; int m = 0; for (int c = 1; c < 3; c++) if (fabs(s->d[c]) < fabs(s->d[m])) m = c; ref[m] = 1;
    cross3(ref, s->d, s->ax); const double na = sqrt(s->ax[0] * s->ax[0] + s->ax[1] * s->ax[1] + s->ax[2] * s->ax[2]); for (int c = 0; c < 3; c++) s->ax[c] /= na;
    cross3(s->d, s->ax, s->bx);
    for (int c = 0; c < 3; c++) s->o[c] = ro[c] - 0.5 * (T - 1) * (s->ax[c] + s->bx[c]);
}
static void slab_bbox(const vwt_slab *s, int T, int L, double lo[3], double hi[3]) {   /* xyz */
    for (int e = 0; e < 3; e++) { lo[e] = 1e300; hi[e] = -1e300; }
    for (int c = 0; c < 8; c++) {
        const double i = (c & 1) ? T - 1 : 0, j = (c & 2) ? T - 1 : 0, k = (c & 4) ? L - 1 : 0;
        for (int e = 0; e < 3; e++) { const double v = s->o[e] + i * s->ax[e] + j * s->bx[e] + k * s->d[e]; lo[e] = fmin(lo[e], v); hi[e] = fmax(hi[e], v); }
    }
}
static int inside(const vwt_slab *s, const double p[3], int T, int L, double mg) {
    double u[3]; vwt_to_slab(s, p, u);
    return u[0] >= mg && u[0] <= T - 1 - mg && u[1] >= mg && u[1] <= T - 1 - mg && u[2] >= mg && u[2] <= L - 1 - mg;
}

int vwt_tiling_build(const char *umb_json, const int64_t box[6], int T, int L, double sz, double sr, double st, double r00, vwt_tiling *t) {
    memset(t, 0, sizeof *t); umbt u; if (umb_load(umb_json, &u)) { fprintf(stderr, "vwt_tile: cannot read %s\n", umb_json); return -1; }
    t->T = T; t->L = L; t->sz = sz; t->sr = sr; t->st = st; t->r00 = r00; memcpy(t->box, box, sizeof t->box);
    const double bz0 = box[0], bz1 = box[0] + box[3], by0 = box[1], by1 = box[1] + box[4], bx0 = box[2], bx1 = box[2] + box[5];
    size_t cap = 4096; t->s = malloc(cap * sizeof *t->s);
    for (int b = 0;; b++) {
        const double zc = bz0 + 0.5 * T - 8 + sz * b; if (zc - 0.5 * T >= bz1) break;
        double ux, uy; umb_at(&u, zc, &ux, &uy);
        double rmax = 0; for (int c = 0; c < 4; c++) { const double x = (c & 1) ? bx1 : bx0, y = (c & 2) ? by1 : by0; rmax = fmax(rmax, hypot(x - ux, y - uy)); }
        for (int m = 0;; m++) {
            const double r0 = r00 + sr * m; if (r0 > rmax + 8) break;
            const double rout = r0 + (L - 1); const int na = (int)ceil(2 * M_PI * rout / st);  /* step at the outer end: no wedge gaps */ const double dth = 2 * M_PI / na;
            for (int a = 0; a < na; a++) {
                vwt_slab s = {b, m, a, zc, r0, a * dth, {0}, {0}, {0}, {0}};
                const double dir[3] = {cos(s.th), sin(s.th), 0}, ro[3] = {ux + r0 * dir[0], uy + r0 * dir[1], zc};
                frame(&s, dir, ro, T);
                double lo[3], hi[3]; slab_bbox(&s, T, L, lo, hi);
                if (hi[0] < bx0 || lo[0] >= bx1 || hi[1] < by0 || lo[1] >= by1 || hi[2] < bz0 || lo[2] >= bz1) continue;
                if ((size_t)t->n == cap) { cap *= 2; t->s = realloc(t->s, cap * sizeof *t->s); }
                t->s[t->n++] = s;
            }
        }
    }
    free(u.p);
    /* neighbour pairs: candidates within band +-1, ring +-1 (slab order is band, ring, angle) */
    size_t pc = 4096; t->p = malloc(pc * sizeof *t->p);
    const double mg = 8, step = 4;
    for (int i = 0; i < t->n; i++) for (int j = i + 1; j < t->n; j++) {
        const vwt_slab *A = &t->s[i], *B = &t->s[j];
        if (B->b > A->b + 1) break;
        if (abs(B->m - A->m) > 1) continue;
        double la[3], ha[3], lb[3], hb[3]; slab_bbox(A, T, L, la, ha); slab_bbox(B, T, L, lb, hb);
        if (ha[0] < lb[0] || hb[0] < la[0] || ha[1] < lb[1] || hb[1] < la[1] || ha[2] < lb[2] || hb[2] < la[2]) continue;
        /* lattice over A's interior, kept where inside B's interior */
        /* B's corners in A's slab coordinates bound the intersection: scan only that part of A */
        double ulo[3] = {1e300, 1e300, 1e300}, uhi[3] = {-1e300, -1e300, -1e300};
        for (int c = 0; c < 8; c++) {
            const double ii = (c & 1) ? T - 1 : 0, jj = (c & 2) ? T - 1 : 0, kk = (c & 4) ? L - 1 : 0; double p[3], uu[3];
            for (int e = 0; e < 3; e++) p[e] = B->o[e] + ii * B->ax[e] + jj * B->bx[e] + kk * B->d[e];
            vwt_to_slab(A, p, uu); for (int e = 0; e < 3; e++) { ulo[e] = fmin(ulo[e], uu[e]); uhi[e] = fmax(uhi[e], uu[e]); }
        }
        const double lim[3] = {T - 1 - mg, T - 1 - mg, L - 1 - mg};
        for (int e = 0; e < 3; e++) { ulo[e] = fmax(ulo[e], mg); uhi[e] = fmin(uhi[e], lim[e]); }
        if (ulo[0] > uhi[0] || ulo[1] > uhi[1] || ulo[2] > uhi[2]) continue;
        for (int e = 0; e < 3; e++) ulo[e] = mg + step * ceil((ulo[e] - mg) / step);
        int np = 0, cap2 = 8192; float (*pp)[3] = malloc(cap2 * sizeof *pp);
        for (double i0 = ulo[0]; i0 <= uhi[0]; i0 += step) for (double j0 = ulo[1]; j0 <= uhi[1]; j0 += step) for (double k0 = ulo[2]; k0 <= uhi[2]; k0 += step) {
            double p[3]; for (int e = 0; e < 3; e++) p[e] = A->o[e] + i0 * A->ax[e] + j0 * A->bx[e] + k0 * A->d[e];
            if (!inside(B, p, T, L, mg)) continue;
            if (np == cap2) { cap2 *= 2; pp = realloc(pp, cap2 * sizeof *pp); }
            pp[np][0] = (float)p[0]; pp[np][1] = (float)p[1]; pp[np][2] = (float)p[2]; np++;
        }
        if (np >= 16) {
            if ((size_t)t->np == pc) { pc *= 2; t->p = realloc(t->p, pc * sizeof *t->p); }
            vwt_pair *q = &t->p[t->np++]; q->i = i; q->j = j; q->npts = np < VWT_PAIR_PTS ? np : VWT_PAIR_PTS;
            for (int k = 0; k < q->npts; k++) { const int src = (int)((long)k * np / q->npts); memcpy(q->pts[k], pp[src], sizeof q->pts[k]); }
        }
        free(pp);
    }
    return 0;
}
void vwt_tiling_free(vwt_tiling *t) { free(t->s); free(t->p); memset(t, 0, sizeof *t); }
int vwt_tiling_save(const vwt_tiling *t, const char *path) {
    FILE *f = fopen(path, "wb"); if (!f) return -1;
    int rc = fwrite("VWTT", 1, 4, f) != 4;
    rc |= fwrite(&t->T, sizeof t->T, 1, f) != 1; rc |= fwrite(&t->L, sizeof t->L, 1, f) != 1;
    const double g[4] = {t->sz, t->sr, t->st, t->r00}; rc |= fwrite(g, 8, 4, f) != 4; rc |= fwrite(t->box, 8, 6, f) != 6;
    rc |= fwrite(&t->n, 4, 1, f) != 1; rc |= fwrite(t->s, sizeof *t->s, t->n, f) != (size_t)t->n;
    rc |= fwrite(&t->np, 4, 1, f) != 1; rc |= fwrite(t->p, sizeof *t->p, t->np, f) != (size_t)t->np;
    rc |= fclose(f) != 0; return rc ? -1 : 0;
}
int vwt_tiling_load(const char *path, vwt_tiling *t) {
    memset(t, 0, sizeof *t); FILE *f = fopen(path, "rb"); if (!f) return -1; char mg[4]; double g[4];
    int ok = fread(mg, 1, 4, f) == 4 && !memcmp(mg, "VWTT", 4) && fread(&t->T, 4, 1, f) == 1 && fread(&t->L, 4, 1, f) == 1 && fread(g, 8, 4, f) == 4 && fread(t->box, 8, 6, f) == 6 && fread(&t->n, 4, 1, f) == 1;
    if (ok) { t->s = malloc((size_t)t->n * sizeof *t->s); ok = fread(t->s, sizeof *t->s, t->n, f) == (size_t)t->n && fread(&t->np, 4, 1, f) == 1; }
    if (ok) { t->p = malloc((size_t)t->np * sizeof *t->p); ok = fread(t->p, sizeof *t->p, t->np, f) == (size_t)t->np; }
    fclose(f); if (!ok) { vwt_tiling_free(t); return -1; }
    t->sz = g[0]; t->sr = g[1]; t->st = g[2]; t->r00 = g[3]; return 0;
}
int vwt_slab_pairs(const vwt_tiling *t, int k, int **pairs, int **side) {
    int n = 0; for (int q = 0; q < t->np; q++) n += t->p[q].i == k || t->p[q].j == k;
    *pairs = malloc((n ? n : 1) * sizeof **pairs); *side = malloc((n ? n : 1) * sizeof **side); n = 0;
    for (int q = 0; q < t->np; q++) if (t->p[q].i == k || t->p[q].j == k) { (*pairs)[n] = q; (*side)[n] = t->p[q].j == k; n++; }
    return n;
}

/* ufsm vwt-tile UMB.json OUT.bin --box z,y,x,nz,ny,nx [--T 128 --L 384 --sz 112 --sr 320 --st 112 --r0 16]: build + save a tiling */
#include <time.h>
static const char *topt(int argc, char **argv, const char *k, const char *d) { for (int i = argc - 2; i >= 1; i--) if (!strcmp(argv[i], k)) return argv[i + 1]; return d; }
int cmd_vwt_tile(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm vwt-tile UMB.json OUT.bin --box z,y,x,nz,ny,nx [--T 128 --L 384 --sz 112 --sr 320 --st 112 --r0 16]\n"); return 2; }
    int64_t box[6]; long long b[6];
    if (sscanf(topt(argc, argv, "--box", ""), "%lld,%lld,%lld,%lld,%lld,%lld", b, b + 1, b + 2, b + 3, b + 4, b + 5) != 6) { fprintf(stderr, "vwt-tile: --box z,y,x,nz,ny,nx\n"); return 2; }
    for (int i = 0; i < 6; i++) box[i] = b[i];
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    vwt_tiling t; if (vwt_tiling_build(argv[2], box, atoi(topt(argc, argv, "--T", "128")), atoi(topt(argc, argv, "--L", "384")), atof(topt(argc, argv, "--sz", "112")),
                                       atof(topt(argc, argv, "--sr", "320")), atof(topt(argc, argv, "--st", "112")), atof(topt(argc, argv, "--r0", "16")), &t)) return 1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double mp = 0; for (int q = 0; q < t.np; q++) mp += t.p[q].npts;
    printf("vwt-tile: %d slabs, %d pairs (mean %.0f points), %.1f s\n", t.n, t.np, t.np ? mp / t.np : 0, (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec));
    return vwt_tiling_save(&t, argv[3]) ? 1 : 0;
}
