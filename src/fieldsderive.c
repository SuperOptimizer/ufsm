/* ufsm fields-derive IN OUT [--normal 1] [--density 1] [--r 2] [--q 8] [--threads N]: a fit-field set whose normal planes (t0..t5)
   and / or density plane (sp) are computed from the phase planes (pc, ps) instead of the network's normal / density heads; the other
   planes are symlinks to IN's. The netsplit ablation (DESIGN 2026-11-20): can the unroll drop the 6 normal and the density channels?

   Per field cell with material (w > 0): the phase gradient g (radians per cell) by central differences of the phasor z = (pc - 128,
   ps - 128): g_a = arg(conj(z_-a) z_+a) / 2 (one-sided where a neighbour has no material or no phase); its confidence c = |z|^2 / 127^2.
   Over a (2r + 1)^3 box: T = sum c n n^T / sum c (n = g / |g|) and G = sum c |g| / sum c. The planes are encoded as predict's
   fit_finalize_k does: t0..t2 = 255 T_zz, T_yy, T_xx; t3..t5 = 255 (0.5 + T_zy, T_zx, T_yx); sp = 8 / (h rho) with rho = G / (2 pi h)
   sheets per native voxel (one phase turn per wrap), i.e. sp = 16 pi / G. A material cell without any phase in its box gets t = 0 and
   sp = 0 (as a cell the network never saw); the count is printed. */
#include "fit.h"
#include "json.h"
#include "optcheck.h"
#include "store.h"
#include "z3w.h"
#include "zarr3.h"
#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

enum { FD_S = 128 };
typedef struct {
    z3 *pc, *ps, *w; z3w *zt[6], *zsp; int D, H, W, R, dn, dd, nys;
    atomic_long nmat, nfall; atomic_int fail;
} fd_job;

static int fd_rd(z3 *z, int z0, int z1, int y0, int y1, int W, uint8_t *o) {   /* rows [z0, z1) x [y0, y1) x W, clamped by the caller */
    const int64_t org[3] = {z0, y0, 0}, n[3] = {z1 - z0, y1 - y0, W};
    return z3_read(z, org, n, o, 1);
}
/* sliding (2R + 1) box sum along one axis of a nz x ny x nx float block, in place (stride s, length n, lines at the other indices) */
static void fd_box1(float *a, size_t nl, const size_t *lo, size_t s, int n, int R, float *t) {
    for (size_t l = 0; l < nl; l++) {
        float *p = a + lo[l]; double acc = 0;
        for (int i = 0; i < n; i++) t[i] = p[(size_t)i * s];
        for (int i = 0; i <= R && i < n; i++) acc += t[i];
        for (int i = 0; i < n; i++) { p[(size_t)i * s] = (float)acc; if (i + R + 1 < n) acc += t[i + R + 1]; if (i - R >= 0) acc -= t[i - R]; }
    }
}
static void fd_box3(float *a, int nz, int ny, int nx, int R) {
    float *t = malloc(sizeof(float) * (size_t)(nz > ny ? (nz > nx ? nz : nx) : (ny > nx ? ny : nx)));
    size_t *lo = malloc(sizeof(size_t) * (size_t)nz * ny), k = 0;
    for (int z = 0; z < nz; z++) for (int y = 0; y < ny; y++) lo[k++] = ((size_t)z * ny + y) * nx;   /* x lines */
    fd_box1(a, k, lo, 1, nx, R, t); free(lo);
    lo = malloc(sizeof(size_t) * (size_t)nz * nx); k = 0;
    for (int z = 0; z < nz; z++) for (int x = 0; x < nx; x++) lo[k++] = (size_t)z * ny * nx + x;   /* y lines */
    fd_box1(a, k, lo, (size_t)nx, ny, R, t); free(lo);
    lo = malloc(sizeof(size_t) * (size_t)ny * nx); k = 0;
    for (int y = 0; y < ny; y++) for (int x = 0; x < nx; x++) lo[k++] = (size_t)y * nx + x;   /* z lines */
    fd_box1(a, k, lo, (size_t)ny * nx, nz, R, t); free(lo); free(t);
}

static void fd_task(void *ctx, int a0, int a1, int th) {
    fd_job *J = ctx; (void)th;
    for (int task = a0; task < a1 && !atomic_load(&J->fail); task++) {
        const int zs = task / J->nys, ys = task % J->nys, R = J->R, W = J->W;
        const int z0 = zs * FD_S, z1 = z0 + FD_S < J->D ? z0 + FD_S : J->D, y0 = ys * FD_S, y1 = y0 + FD_S < J->H ? y0 + FD_S : J->H;
        /* the gradient cells [gz0, gz1) x [gy0, gy1) (the box's reach), the phasors one more */
        const int gz0 = z0 - R < 0 ? 0 : z0 - R, gz1 = z1 + R > J->D ? J->D : z1 + R, gy0 = y0 - R < 0 ? 0 : y0 - R, gy1 = y1 + R > J->H ? J->H : y1 + R;
        const int pz0 = gz0 - 1 < 0 ? 0 : gz0 - 1, pz1 = gz1 + 1 > J->D ? J->D : gz1 + 1, py0 = gy0 - 1 < 0 ? 0 : gy0 - 1, py1 = gy1 + 1 > J->H ? J->H : gy1 + 1;
        const int pnz = pz1 - pz0, pny = py1 - py0, gnz = gz1 - gz0, gny = gy1 - gy0;
        const size_t pn = (size_t)pnz * pny * W, gn = (size_t)gnz * gny * W;
        uint8_t *pc = malloc(pn), *ps = malloc(pn), *w = malloc(pn);
        float *A = calloc(8 * gn, sizeof(float));   /* c n n^T (6), c |g|, c */
        uint8_t *ob = calloc((size_t)7 * FD_S * FD_S * FD_S, 1);
        if (!pc || !ps || !w || !A || !ob || fd_rd(J->pc, pz0, pz1, py0, py1, W, pc) || fd_rd(J->ps, pz0, pz1, py0, py1, W, ps) || fd_rd(J->w, pz0, pz1, py0, py1, W, w)) {
            fprintf(stderr, "fields-derive: rows %d-%d y %d-%d: %s\n", z0, z1, y0, y1, z3_error()); atomic_store(&J->fail, 1);
            free(pc); free(ps); free(w); free(A); free(ob); return;
        }
        #define PI_(z, y, x) (((size_t)((z) - pz0) * pny + ((y) - py0)) * W + (x))
        for (int z = gz0; z < gz1; z++) for (int y = gy0; y < gy1; y++) for (int x = 0; x < W; x++) {
            const size_t i = PI_(z, y, x); if (!w[i]) continue;
            const float zr = pc[i] - 128.f, zi = ps[i] - 128.f, c = (zr * zr + zi * zi) * (1.f / (127.f * 127.f)); if (c <= 0.f) continue;
            float g[3]; int ok = 1;
            for (int a = 0; a < 3 && ok; a++) {
                const int cz = z, cy = y, cx = x; int mz = cz, my = cy, mx = cx, qz = cz, qy = cy, qx = cx;
                if (a == 0) { mz--; qz++; } else if (a == 1) { my--; qy++; } else { mx--; qx++; }
                const int lo_ok = (a == 0 ? mz >= pz0 : a == 1 ? my >= py0 : mx >= 0), hi_ok = (a == 0 ? qz < pz1 : a == 1 ? qy < py1 : qx < W);
                size_t im = 0, iq = 0; int hm = 0, hq = 0;
                if (lo_ok) { im = PI_(mz, my, mx); hm = w[im] && (pc[im] != 128 || ps[im] != 128); }
                if (hi_ok) { iq = PI_(qz, qy, qx); hq = w[iq] && (pc[iq] != 128 || ps[iq] != 128); }
                const size_t u = hm ? im : i, v = hq ? iq : i; const float dv = (float)(hm + hq);
                if (dv == 0.f) { ok = 0; break; }
                const float ur = pc[u] - 128.f, ui = ps[u] - 128.f, vr = pc[v] - 128.f, vi = ps[v] - 128.f;
                g[a] = atan2f(ur * vi - ui * vr, ur * vr + ui * vi) / dv;   /* arg(conj(z_u) z_v) per cell */
            }
            if (!ok) continue;
            const float gl = sqrtf(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]); if (gl < 1e-6f) continue;
            const float n0 = g[0] / gl, n1 = g[1] / gl, n2 = g[2] / gl; const size_t k = ((size_t)(z - gz0) * gny + (y - gy0)) * W + x;
            A[k] = c * n0 * n0; A[gn + k] = c * n1 * n1; A[2 * gn + k] = c * n2 * n2; A[3 * gn + k] = c * n0 * n1; A[4 * gn + k] = c * n0 * n2;
            A[5 * gn + k] = c * n1 * n2; A[6 * gn + k] = c * gl; A[7 * gn + k] = c;
        }
        for (int q = 0; q < 8; q++) fd_box3(A + (size_t)q * gn, gnz, gny, W, R);
        long nm = 0, nf = 0;
        for (int z = z0; z < z1; z++) for (int y = y0; y < y1; y++) for (int x = 0; x < W; x++) {
            if (!w[PI_(z, y, x)]) continue;
            nm++; const size_t k = ((size_t)(z - gz0) * gny + (y - gy0)) * W + x; const float C = A[7 * gn + k];
            if (C <= 1e-6f) { nf++; continue; }
            uint8_t t[7];
            for (int q = 0; q < 6; q++) { const float v = (q < 3 ? 0.f : 0.5f) + A[(size_t)q * gn + k] / C; t[q] = (uint8_t)lroundf(255.f * fminf(1.f, fmaxf(0.f, v))); }
            const float G = A[6 * gn + k] / C, sp = G > 0.f ? 16.f * (float)M_PI / G : 0.f; t[6] = (uint8_t)(sp <= 0.f ? 0 : sp < 1.f ? 1 : sp > 255.f ? 255 : lroundf(sp));
            for (int q = 0; q < 7; q++) A[(size_t)q * gn + k] = (float)t[q];   /* (the sums are spent: the codes go to the first 7 slots) */
        }
        atomic_fetch_add(&J->nmat, nm); atomic_fetch_add(&J->nfall, nf);
        /* the shards of this (zs, ys) row: x shards one after another */
        for (int xs = 0; xs * FD_S < W && !atomic_load(&J->fail); xs++) {
            memset(ob, 0, (size_t)7 * FD_S * FD_S * FD_S);
            const int x0 = xs * FD_S, x1 = x0 + FD_S < W ? x0 + FD_S : W;
            for (int z = z0; z < z1; z++) for (int y = y0; y < y1; y++) for (int x = x0; x < x1; x++) {
                if (!w[PI_(z, y, x)]) continue;
                const size_t k = ((size_t)(z - gz0) * gny + (y - gy0)) * W + x; if (A[7 * gn + k] <= 1e-6f) continue;
                const size_t o = ((size_t)(z - z0) * FD_S + (y - y0)) * FD_S + (x - x0);
                for (int q = 0; q < 7; q++) ob[(size_t)q * FD_S * FD_S * FD_S + o] = (uint8_t)A[(size_t)q * gn + k];
            }
            for (int q = 0; q < 7; q++) {
                z3w *zw = q < 6 ? (J->dn ? J->zt[q] : nullptr) : (J->dd ? J->zsp : nullptr); if (!zw) continue;
                if (z3w_write_shard(zw, zs, ys, xs, ob + (size_t)q * FD_S * FD_S * FD_S, 1)) { fprintf(stderr, "fields-derive: write: %s\n", z3w_error()); atomic_store(&J->fail, 1); }
            }
        }
        #undef PI_
        free(pc); free(ps); free(w); free(A); free(ob);
    }
}

static const char *fd_opt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 3; i + 1 < argc; i++) if (!strcmp(argv[i], opt_n(name))) return argv[i + 1];
    return dflt;
}

int cmd_fields_derive(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: ufsm fields-derive IN OUT [--normal 1] [--density 1] [--r 2] [--q 8] [--threads N]\n"
                        "  the normal planes t0..t5 and / or the density plane sp from the phase gradient (pc, ps); other planes linked (src/fieldsderive.c)\n");
        return 2;
    }
    const char *in = argv[2], *out = argv[3];
    fd_job J = {.R = opt_atoi("--r", fd_opt(argc, argv, "--r", "2")), .dn = opt_atoi("--normal", fd_opt(argc, argv, "--normal", "1")),
                .dd = opt_atoi("--density", fd_opt(argc, argv, "--density", "1"))};
    const float q = (float)opt_atof("--q", fd_opt(argc, argv, "--q", "8"));
    fit_set_threads(opt_atoi("--threads", fd_opt(argc, argv, "--threads", "0")));
    if (!J.dn && !J.dd) { fprintf(stderr, "fields-derive: nothing to derive (--normal 0 --density 0)\n"); return 2; }
    store *s = store_open(in); if (!s) { fprintf(stderr, "fields-derive: cannot open %s\n", in); return 1; }
    if (!(J.pc = z3_open(s, "pc", nullptr)) || !(J.ps = z3_open(s, "ps", nullptr)) || !(J.w = z3_open(s, "w", nullptr))) { fprintf(stderr, "fields-derive: %s: no pc / ps / w zarr planes\n", in); return 1; }
    const z3_meta *m = z3_meta_of(J.pc); J.D = (int)m->shape[0]; J.H = (int)m->shape[1]; J.W = (int)m->shape[2];
    if (mkdir(out, 0775) && errno != EEXIST) { fprintf(stderr, "fields-derive: cannot create %s\n", out); return 1; }
    /* links to every entry of IN except the planes written here */
    DIR *d = opendir(in); if (!d) return 1; struct dirent *e;
    while ((e = readdir(d))) {
        if (e->d_name[0] == '.') continue;
        const int tpl = e->d_name[0] == 't' && e->d_name[1] >= '0' && e->d_name[1] <= '5' && !e->d_name[2];
        if ((J.dn && tpl) || (J.dd && !strcmp(e->d_name, "sp"))) continue;
        char a[2048], b[2048]; snprintf(b, sizeof b, "%s/%s", out, e->d_name);
        if (!strcmp(e->d_name, "fields.json")) {   /* a copy (consumers may rewrite theirs) */
            snprintf(a, sizeof a, "%s/%s", in, e->d_name); FILE *f = fopen(a, "rb"), *g = fopen(b, "wb"); if (!f || !g) return 1;
            char buf[65536]; size_t r; while ((r = fread(buf, 1, sizeof buf, f)) > 0) fwrite(buf, 1, r, g); fclose(f); if (fclose(g)) return 1; continue;
        }
        char *rp = realpath(in, nullptr); snprintf(a, sizeof a, "%s/%s", rp ? rp : in, e->d_name); free(rp);
        unlink(b); if (symlink(a, b)) { fprintf(stderr, "fields-derive: cannot link %s\n", b); return 1; }
    }
    closedir(d);
    const int64_t sh[3] = {J.D, J.H, J.W};
    char at[256]; snprintf(at, sizeof at, "{\"ufsm\":{\"content\":\"winding fitter fields (src/fit.h), ufsm fields-derive r %d from the phase gradient\"}}", J.R);
    for (int k = 0; k < 7; k++) {
        if ((k < 6 && !J.dn) || (k == 6 && !J.dd)) continue;
        char p[2048]; snprintf(p, sizeof p, "%s/%s", out, k < 6 ? (const char *[]){"t0", "t1", "t2", "t3", "t4", "t5"}[k] : "sp");
        z3w *zw = z3w_create(p, sh, FD_S, q, 0, at); if (!zw) { fprintf(stderr, "fields-derive: %s\n", z3w_error()); return 1; }
        if (k < 6) J.zt[k] = zw; else J.zsp = zw;
    }
    J.nys = (J.H + FD_S - 1) / FD_S; const int nt = ((J.D + FD_S - 1) / FD_S) * J.nys;
    fit_par(nt, fd_task, &J);
    int rc = atomic_load(&J.fail);
    for (int k = 0; k < 6; k++) if (J.zt[k] && z3w_close(J.zt[k])) rc = 1;
    if (J.zsp && z3w_close(J.zsp)) rc = 1;
    fprintf(stderr, "fields-derive: %s%s%s from the phase gradient (r %d): %ld material cells, %ld (%.4f) without phase in their box (t, sp 0)%s\n",
            J.dn ? "t0..t5" : "", J.dn && J.dd ? " + " : "", J.dd ? "sp" : "", J.R, (long)J.nmat, (long)J.nfall, J.nmat ? (double)J.nfall / J.nmat : 0.0, rc ? ": FAILED" : "");
    return rc;
}
