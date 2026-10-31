/* CT in the winding fitter (src/fit.h): the papyrus / void parity of the fields' mean-CT plane, the CT's own phase for the voxel
   unwrapping, sheet spacing from papyrus peaks along a normal, and ufsm ct-fields (the ct / cs planes of an existing field directory
   from a local CT zarr, so fields predicted without them need no new inference).
   Planes (uint8, on the field grid):
     ct  the mean CT over the cell's CT voxels (0 = outside the scan)
     cs  the CT sheet spacing in the units of sp (h / 8 native voxels; 0 = unknown): per cell (every --stride cells, the block filled)
         the CT sampled along the network normal (the top eigenvector of the cell's n n^T) over +-R, its papyrus peaks counted
         (fit_ct_profile_spacing), spacing = the distance between the first and last peak / (peaks - 1). It does not use the phase. */
#include "fit.h"
#include "sysinfo.h"
#include "fit_terms.h"
#include "zarr3.h"
#include "store.h"
#include <complex.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static double ct_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

/* ---- the parity: m = tanh((ct - mu) / sd) with mu / sd over a cube of half-edge r cells. The statistics are smooth, so they are
   taken on blocks of s = max(1, r / 2) cells (block sums of ct, ct^2 and counts), box-filtered there over +-ceil(r / s) blocks and
   interpolated trilinearly back to the cells: memory n / s^3, not 12 bytes per cell. ---- */
static void box1(double *a, int n0, int n1, int n2, int axis, int R, double *tmp) {   /* running sums over +-R along one axis, in place */
    const size_t st = axis == 0 ? (size_t)n1 * n2 : axis == 1 ? (size_t)n2 : 1; const int L = axis == 0 ? n0 : axis == 1 ? n1 : n2;
    const size_t nlines = (size_t)n0 * n1 * n2 / L;
    for (size_t q = 0; q < nlines; q++) {
        size_t base;
        if (axis == 0) base = q; else if (axis == 1) base = (q / n2) * (size_t)n1 * n2 + q % n2; else base = q * n2;
        double s = 0; for (int k = 0; k <= R && k < L; k++) s += a[base + k * st];
        for (int k = 0; k < L; k++) {
            tmp[k] = s;
            if (k + R + 1 < L) s += a[base + (k + R + 1) * st];
            if (k - R >= 0) s -= a[base + (k - R) * st];
        }
        for (int k = 0; k < L; k++) a[base + k * st] = tmp[k];
    }
}
void fit_ct_parity(const uint8_t *ct, int D, int H, int W, int r, uint8_t *out) {
    if (r < 1) r = 1;
    const int s = r / 2 > 1 ? r / 2 : 1, R = (r + s - 1) / s;
    const int bD = (D + s - 1) / s, bH = (H + s - 1) / s, bW = (W + s - 1) / s; const size_t nb = (size_t)bD * bH * bW;
    double *S1 = calloc(nb, sizeof(double)), *S2 = calloc(nb, sizeof(double)), *N = calloc(nb, sizeof(double));
    const int L = bD > bH ? (bD > bW ? bD : bW) : (bH > bW ? bH : bW); double *tmp = malloc((size_t)L * sizeof(double));
    for (int z = 0; z < D; z++) for (int y = 0; y < H; y++) {
        const uint8_t *row = ct + ((size_t)z * H + y) * W; const size_t bo = ((size_t)(z / s) * bH + y / s) * bW;
        for (int x = 0; x < W; x++) if (row[x]) { const double v = row[x]; S1[bo + x / s] += v; S2[bo + x / s] += v * v; N[bo + x / s] += 1; }
    }
    for (int a = 0; a < 3; a++) { box1(S1, bD, bH, bW, a, R, tmp); box1(S2, bD, bH, bW, a, R, tmp); box1(N, bD, bH, bW, a, R, tmp); }
    float *mu = malloc(nb * sizeof(float)), *sd = malloc(nb * sizeof(float));
    for (size_t k = 0; k < nb; k++) {
        if (N[k] < 1) { mu[k] = 0; sd[k] = 0; continue; }
        const double m = S1[k] / N[k], v = S2[k] / N[k] - m * m; mu[k] = (float)m; sd[k] = (float)sqrt(v > 0 ? v : 0);
    }
    free(S1); free(S2); free(N); free(tmp);
    for (int z = 0; z < D; z++) {
        const float qz = fminf(fmaxf((z + 0.5f) / s - 0.5f, 0.f), (float)(bD - 1)); const int z0 = (int)qz < bD - 1 ? (int)qz : (bD > 1 ? bD - 2 : 0); const float tz = bD > 1 ? qz - z0 : 0.f;
        for (int y = 0; y < H; y++) {
            const float qy = fminf(fmaxf((y + 0.5f) / s - 0.5f, 0.f), (float)(bH - 1)); const int y0 = (int)qy < bH - 1 ? (int)qy : (bH > 1 ? bH - 2 : 0); const float ty = bH > 1 ? qy - y0 : 0.f;
            for (int x = 0; x < W; x++) {
                const size_t i = ((size_t)z * H + y) * W + x;
                if (!ct[i]) { out[i] = 128; continue; }
                const float qx = fminf(fmaxf((x + 0.5f) / s - 0.5f, 0.f), (float)(bW - 1)); const int x0 = (int)qx < bW - 1 ? (int)qx : (bW > 1 ? bW - 2 : 0); const float tx = bW > 1 ? qx - x0 : 0.f;
                float m = 0, d = 0;
                for (int c = 0; c < 8; c++) {
                    const int a = z0 + (c >> 2 & 1), b = y0 + (c >> 1 & 1), e = x0 + (c & 1);
                    const float w = (c >> 2 & 1 ? tz : 1 - tz) * (c >> 1 & 1 ? ty : 1 - ty) * (c & 1 ? tx : 1 - tx);
                    if (w == 0.f) continue;
                    const size_t k = ((size_t)(a < bD ? a : bD - 1) * bH + (b < bH ? b : bH - 1)) * bW + (e < bW ? e : bW - 1);
                    m += w * mu[k]; d += w * sd[k];
                }
                const float v = d >= 1.f ? tanhf((ct[i] - m) / d) : 0.f;
                out[i] = (uint8_t)lroundf(128.f + 127.f * v);
            }
        }
    }
    free(mu); free(sd);
}

/* ---- the CT phase: m along the model's normal, demodulated at the local density ---- */
static float ctm_at(const uint8_t *ctm, int D, int H, int W, float z, float y, float x) {   /* trilinear parity at cell coordinates (centres at k + 0.5) */
    z -= 0.5f; y -= 0.5f; x -= 0.5f;
    if (z < 0 || y < 0 || x < 0 || z > D - 1 || y > H - 1 || x > W - 1) return 0.f;
    const int z0 = (int)z < D - 1 ? (int)z : (D > 1 ? D - 2 : 0), y0 = (int)y < H - 1 ? (int)y : (H > 1 ? H - 2 : 0), x0 = (int)x < W - 1 ? (int)x : (W > 1 ? W - 2 : 0);
    const float tz = z - z0, ty = y - y0, tx = x - x0; float s = 0;
    for (int c = 0; c < 8; c++) {
        const int a = z0 + (c >> 2 & 1), b = y0 + (c >> 1 & 1), e = x0 + (c & 1);
        const float w = (c >> 2 & 1 ? tz : 1 - tz) * (c >> 1 & 1 ? ty : 1 - ty) * (c & 1 ? tx : 1 - tx);
        if (w != 0.f) s += w * ((int)ctm[((size_t)(a < D ? a : D - 1) * H + (b < H ? b : H - 1)) * W + (e < W ? e : W - 1)] - 128) * (1.f / 127.f);
    }
    return s;
}
typedef struct { const fit_model *m; const fit_fields *f; const uint8_t *ctm; float phi; uint8_t *opc, *ops; _Atomic int next; } cph_job;
static void *cph_rows(void *a) {
    cph_job *J = a; const fit_fields *f = J->f; const double h = f->h;
    for (;;) {
        const int z = atomic_fetch_add(&J->next, 1); if (z >= f->D) break;
        for (int y = 0; y < f->H; y++) for (int x = 0; x < f->W; x++) {
            const size_t i = ((size_t)z * f->H + y) * f->W + x; J->opc[i] = J->ops[i] = 128;
            if (!f->w[i] || J->ctm[i] == 128) continue;
            const double pz = (z + 0.5) * h, py = (y + 0.5) * h, px = (x + 0.5) * h, e = 0.25 * h;
            double g[3], gl = 0;
            for (int d = 0; d < 3; d++) {
                const double dq = fit_W(J->m, pz + (d == 0 ? e : 0), py + (d == 1 ? e : 0), px + (d == 2 ? e : 0)) - fit_W(J->m, pz - (d == 0 ? e : 0), py - (d == 1 ? e : 0), px - (d == 2 ? e : 0));
                g[d] = (dq - round(dq)) / (2 * e); gl += g[d] * g[d];   /* (the theta cut: a whole turn) */
            }
            gl = sqrt(gl); if (!(gl > 1e-6)) continue;
            const double rho = f->cs && f->cs[i] ? 8.0 / (h * f->cs[i]) : gl;   /* turns per native voxel */
            const double step = 0.5 * h; int K = (int)ceil(1.5 / (rho * step)); if (K > 64) K = 64; if (K < 2) K = 2;
            double complex A = 0; double sw = 0;
            for (int k = -K; k <= K; k++) {
                const double t = k * step, hw = 0.5 + 0.5 * cos(M_PI * k / (K + 1.0));
                const float mv = ctm_at(J->ctm, f->D, f->H, f->W, (float)((pz + t * g[0] / gl) / h), (float)((py + t * g[1] / gl) / h), (float)((px + t * g[2] / gl) / h));
                A += hw * mv * cexp(-I * 2 * M_PI * rho * t); sw += hw;
            }
            const double q = fmin(1.0, 2 * cabs(A) / sw), ang = carg(A) + 2 * M_PI * J->phi;   /* m ~ cos 2 pi (W + rho t - phi): arg A = 2 pi (W - phi) */
            J->opc[i] = (uint8_t)lround(128 + 127 * q * cos(ang)); J->ops[i] = (uint8_t)lround(128 + 127 * q * sin(ang));
        }
    }
    return nullptr;
}
void fit_ct_phasor(const fit_model *m, const fit_fields *f, const uint8_t *ctm, float phi_pap, uint8_t *opc, uint8_t *ops) {
    cph_job J = {.m = m, .f = f, .ctm = ctm, .phi = phi_pap, .opc = opc, .ops = ops};
    long nt = ufsm_ncpu(); const char *e = getenv("UFSM_FIT_THREADS"); if (e && atoi(e) > 0) nt = atoi(e); if (nt > 64) nt = 64; if (nt < 1) nt = 1;
    pthread_t th[64]; int up[64] = {0};
    for (int t = 1; t < nt; t++) up[t] = pthread_create(&th[t], nullptr, cph_rows, &J) == 0;
    cph_rows(&J);
    for (int t = 1; t < nt; t++) if (up[t]) pthread_join(th[t], nullptr);
}

/* ---- sheet spacing from a CT profile ---- */
double fit_ct_profile_spacing(const float *p, int n, double step, double prom) {   /* src/fit_terms.h fit_ct_spacing on a copy */
    float q[1024]; if (n > 1024) n = 1024; if (n < 1) return 0;
    memcpy(q, p, (size_t)n * sizeof(float));
    return fit_ct_spacing(q, n, (float)step, (float)prom);
}

/* ---- ufsm ct-fields ---- */
static const char *copt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return dflt;
}
typedef struct {
    const fit_fields *f; int fz0;              /* the slab's fields (rows fz0.. of the box) */
    const uint8_t *vol; int64_t vo[3], vn[3];  /* the CT box (CT voxels) */
    double fct, um, ctum, org[3];              /* native -> CT voxel scale, native um, CT um, the box origin (native) */
    double radius_um, prom; int stride;
    uint8_t *ct, *cs;                          /* whole-box outputs */
    int H, W; _Atomic int next;
    _Atomic long long nray, npk;               /* rays cast, rays with a spacing */
} ctf_job;
static float ctf_sample(const ctf_job *J, double cz, double cy, double cx) {   /* trilinear CT at CT voxel coordinates (box-relative to vo); -1 outside */
    double q[3] = {cz - J->vo[0], cy - J->vo[1], cx - J->vo[2]}; int i0[3]; double t[3];
    for (int d = 0; d < 3; d++) { if (q[d] < 0 || q[d] > J->vn[d] - 1) return -1.f; i0[d] = (int)q[d]; if (i0[d] > J->vn[d] - 2) i0[d] = (int)J->vn[d] - 2; if (i0[d] < 0) i0[d] = 0; t[d] = q[d] - i0[d]; }
    double s = 0;
    for (int c = 0; c < 8; c++) {
        const int a = i0[0] + (c >> 2 & 1), b = i0[1] + (c >> 1 & 1), e = i0[2] + (c & 1);
        const double w = (c >> 2 & 1 ? t[0] : 1 - t[0]) * (c >> 1 & 1 ? t[1] : 1 - t[1]) * (c & 1 ? t[2] : 1 - t[2]);
        if (w != 0) s += w * J->vol[((size_t)(a < J->vn[0] ? a : J->vn[0] - 1) * J->vn[1] + (b < J->vn[1] ? b : J->vn[1] - 1)) * J->vn[2] + (e < J->vn[2] ? e : J->vn[2] - 1)];
    }
    return (float)s;
}
static void *ctf_rows(void *a) {
    ctf_job *J = a; const fit_fields *f = J->f; const double h = f->h, F = J->fct; const int st = J->stride;
    float prof[1024]; long long nray = 0, npk = 0;
    for (;;) {
        const int z = atomic_fetch_add(&J->next, 1); if (z >= f->D) break;
        const int gz = J->fz0 + z;
        for (int y = 0; y < f->H; y++) for (int x = 0; x < f->W; x++) {
            const size_t i = ((size_t)z * f->H + y) * f->W + x, go = ((size_t)gz * J->H + y) * J->W + x;
            /* the mean CT over the CT voxels whose centres fall in the cell (native [k h, (k + 1) h)): CT voxel c at native (c + 0.5) / F - 0.5 */
            int64_t lo[3], hi[3]; const int p3[3] = {gz, y, x};
            for (int d = 0; d < 3; d++) { lo[d] = (int64_t)ceil((p3[d] * h + J->org[d] + 0.5) * F - 0.5); hi[d] = (int64_t)ceil(((p3[d] + 1) * h + J->org[d] + 0.5) * F - 0.5); }
            double sum = 0; long cnt = 0;
            if (hi[0] > lo[0] && hi[1] > lo[1] && hi[2] > lo[2]) {
                for (int64_t a2 = lo[0]; a2 < hi[0]; a2++) for (int64_t b = lo[1]; b < hi[1]; b++) for (int64_t c = lo[2]; c < hi[2]; c++) {
                    const int64_t q0 = a2 - J->vo[0], q1 = b - J->vo[1], q2 = c - J->vo[2];
                    if (q0 < 0 || q1 < 0 || q2 < 0 || q0 >= J->vn[0] || q1 >= J->vn[1] || q2 >= J->vn[2]) continue;
                    sum += J->vol[((size_t)q0 * J->vn[1] + q1) * J->vn[2] + q2]; cnt++;
                }
            } else {   /* CT coarser than the cell: trilinear at the centre */
                const float v = ctf_sample(J, ((gz + 0.5) * h + J->org[0] + 0.5) * F - 0.5, ((y + 0.5) * h + J->org[1] + 0.5) * F - 0.5, ((x + 0.5) * h + J->org[2] + 0.5) * F - 0.5);
                if (v >= 0) { sum = v; cnt = 1; }
            }
            J->ct[go] = cnt ? (uint8_t)lround(sum / cnt) : 0;
            if (gz % st || y % st || x % st) continue;
            /* the sheet spacing on this stride cell's block */
            uint8_t code = 0;
            const float tzz = f->t[0][i] / 255.f, tyy = f->t[1][i] / 255.f, txx = f->t[2][i] / 255.f, tzy = f->t[3][i] / 255.f - 0.5f, tzx = f->t[4][i] / 255.f - 0.5f, tyx = f->t[5][i] / 255.f - 0.5f;
            if (f->w[i] && J->ct[go] && tzz + tyy + txx > 0.1f) {
                double v[3] = {0.31, 0.57, 0.76};   /* the top eigenvector of n n^T (power iteration) */
                for (int it = 0; it < 30; it++) {
                    const double u[3] = {tzz * v[0] + tzy * v[1] + tzx * v[2], tzy * v[0] + tyy * v[1] + tyx * v[2], tzx * v[0] + tyx * v[1] + txx * v[2]};
                    const double l = sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]); if (l < 1e-9) break; for (int d = 0; d < 3; d++) v[d] = u[d] / l;
                }
                const double spn = f->sp[i] ? f->sp[i] * h / 8.0 * J->um : 0;   /* the network spacing (um) sets the ray's reach */
                double R = spn > 0 ? 2.5 * spn : J->radius_um; if (R < 60) R = 60; if (R > 400) R = 400;
                const double stp = J->ctum; int n = 2 * (int)(R / stp) + 1; if (n > 1023) n = 1023;
                const double c0[3] = {((gz + 0.5) * h + J->org[0] + 0.5) * F - 0.5, ((y + 0.5) * h + J->org[1] + 0.5) * F - 0.5, ((x + 0.5) * h + J->org[2] + 0.5) * F - 0.5};
                int ok = 1;
                for (int k = 0; k < n && ok; k++) {
                    const double t = (k - n / 2) * 1.0;   /* CT voxels along the normal */
                    const float s = ctf_sample(J, c0[0] + t * v[0], c0[1] + t * v[1], c0[2] + t * v[2]);
                    if (s < 0) ok = 0; else prof[k] = s;
                }
                nray++;
                if (ok) {
                    const double spc = fit_ct_profile_spacing(prof, n, J->ctum / J->um, J->prom);   /* native voxels */
                    if (spc > 0) { const double cv = spc * 8.0 / h; code = (uint8_t)(cv < 1 ? 1 : cv > 255 ? 255 : lround(cv)); npk++; }
                }
            }
            for (int a2 = 0; a2 < st && gz + a2 < J->fz0 + f->D; a2++) for (int b = 0; b < st && y + b < f->H; b++) for (int c = 0; c < st && x + c < f->W; c++)
                J->cs[((size_t)(gz + a2) * J->H + y + b) * J->W + x + c] = code;
        }
    }
    atomic_fetch_add(&J->nray, nray); atomic_fetch_add(&J->npk, npk);
    return nullptr;
}
int cmd_ct_fields(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: ufsm ct-fields CTROOT KEY FIELDS-DIR --ct-um U [--radius-um 200] [--prom 1] [--stride 2] [--parity-um 150] [--threads 8] [--cache DIR]\n"
                        "  the CT planes of a fit-field directory from a CT zarr (KEY: the array, voxel size U um): ct.raw (mean CT per cell) and cs.raw\n"
                        "  (CT sheet spacing: papyrus peaks along the network normal over +-2.5 x the network spacing, 60..400 um; --radius-um where\n"
                        "  the spacing is unknown), box-shaped beside the other planes, and ct.json. Prints the papyrus phase: where the CT parity\n"
                        "  peaks relative to the fields' phase (with published-wrap fields: phi_pap for fit --ct-phi).\n");
        return 2;
    }
    const char *root = argv[2], *key = argv[3], *dir = argv[4];
    const double ctum = atof(copt(argc, argv, "--ct-um", "0")), rum = atof(copt(argc, argv, "--radius-um", "200")), prom = atof(copt(argc, argv, "--prom", "1"));
    const int stride = atoi(copt(argc, argv, "--stride", "2")), nt0 = ufsm_threads(atoi(copt(argc, argv, "--threads", "0")), 8);
    const double pum = atof(copt(argc, argv, "--parity-um", "150"));
    if (!(ctum > 0) || stride < 1) { fprintf(stderr, "ct-fields: --ct-um > 0 and --stride >= 1 needed\n"); return 2; }
    z3 *cz = z3_open(store_open(root), key, copt(argc, argv, "--cache", nullptr));
    if (!cz) { fprintf(stderr, "ct-fields: %s\n", z3_error()); return 1; }
    const z3_meta *zm = z3_meta_of(cz);
    fit_fields_buf g; if (fit_fields_load(dir, 0, 0, 0, &g)) { fprintf(stderr, "ct-fields: cannot read %s/fields.json\n", dir); return 1; }
    const int D = g.shape[0], H = g.shape[1], W = g.shape[2], h = g.h; const double um = g.um, F = um / ctum;
    const double org[3] = {(double)g.origin[0], (double)g.origin[1], (double)g.origin[2]};
    fit_fields_free(&g);
    const size_t n = (size_t)D * H * W;
    uint8_t *ct = calloc(n, 1), *cs = calloc(n, 1);
    if (!ct || !cs) { fprintf(stderr, "ct-fields: out of memory\n"); return 1; }
    const double t0 = ct_now(), reach = 400 / ctum + 2;   /* the longest ray (CT voxels) */
    /* z slabs of field rows (a multiple of the stride) whose CT box stays under ~1.5 GB */
    const double ypl = (H * h * F + 2 * reach) * (W * h * F + 2 * reach);
    int nzs = (int)((1.5e9 / ypl - 2 * reach) / (h * F)); nzs -= nzs % stride; if (nzs < stride) nzs = stride;
    long long nray = 0, npk = 0;
    for (int fz0 = 0; fz0 < D; fz0 += nzs) {
        const int fz1 = fz0 + nzs < D ? fz0 + nzs : D;
        fit_fields_buf b; if (fit_fields_load(dir, fz0, fz1, 1, &b)) { fprintf(stderr, "ct-fields: cannot load fields rows %d..%d\n", fz0, fz1); return 1; }
        ctf_job J = {.f = &b.f, .fz0 = fz0, .fct = F, .um = um, .ctum = ctum, .org = {org[0], org[1], org[2]}, .radius_um = rum, .prom = prom, .stride = stride, .ct = ct, .cs = cs, .H = H, .W = W};
        const double lo[3] = {(fz0 * h + org[0] + 0.5) * F - 0.5 - reach, (org[1] + 0.5) * F - 0.5 - reach, (org[2] + 0.5) * F - 0.5 - reach};
        const double hi[3] = {(fz1 * h + org[0] + 0.5) * F - 0.5 + reach, ((double)H * h + org[1] + 0.5) * F - 0.5 + reach, ((double)W * h + org[2] + 0.5) * F - 0.5 + reach};
        int ok = 1;
        for (int d = 0; d < 3; d++) { int64_t a = (int64_t)floor(lo[d]), e = (int64_t)ceil(hi[d]) + 1; if (a < 0) a = 0; if (e > zm->shape[d]) e = zm->shape[d]; if (e <= a) ok = 0; J.vo[d] = a; J.vn[d] = e - a; }
        uint8_t *vol = ok ? malloc((size_t)J.vn[0] * J.vn[1] * J.vn[2]) : nullptr;
        if (ok && (!vol || z3_read(cz, J.vo, J.vn, vol, nt0))) { fprintf(stderr, "ct-fields: CT read failed: %s\n", z3_error()); return 1; }
        if (ok) {
            J.vol = vol;
            const int nt = nt0 < 64 ? (nt0 > 0 ? nt0 : 1) : 64; pthread_t th[64]; int up[64] = {0};
            for (int t = 1; t < nt; t++) up[t] = pthread_create(&th[t], nullptr, ctf_rows, &J) == 0;
            ctf_rows(&J);
            for (int t = 1; t < nt; t++) if (up[t]) pthread_join(th[t], nullptr);
            nray += J.nray; npk += J.npk;
        }
        free(vol); fit_fields_free(&b);
        fprintf(stderr, "\rct-fields: rows %d / %d (%.0f s)", fz1, D, ct_now() - t0);
    }
    fprintf(stderr, "\n");
    /* the papyrus phase: the parity against the fields' phase (confident cells) and the mean parity by phase bin */
    uint8_t *ctm = malloc(n); const int pr = (int)lround(pum / (h * um)); fit_ct_parity(ct, D, H, W, pr, ctm);
    double complex A = 0; double sa = 0, bm[16] = {0}, bn[16] = {0}; size_t nct = 0, ncs = 0; double csm = 0, spm = 0, csr = 0; size_t ncr = 0;
    for (int fz0 = 0; fz0 < D; fz0 += 64) {
        const int fz1 = fz0 + 64 < D ? fz0 + 64 : D;
        fit_fields_buf b; if (fit_fields_load(dir, fz0, fz1, 1, &b)) break;
        const size_t nb = (size_t)(fz1 - fz0) * H * W, off = (size_t)fz0 * H * W;
        for (size_t i = 0; i < nb; i++) {
            const size_t gi = off + i; if (!b.f.w[i] || !ct[gi]) continue; nct++;
            if (cs[gi]) { ncs++; csm += cs[gi] * h / 8.0 * um; if (b.f.sp[i]) { spm += b.f.sp[i] * h / 8.0 * um; csr += log((double)cs[gi] / b.f.sp[i]); ncr++; } }
            if (ctm[gi] == 128) continue;
            const double mc = (b.f.pc[i] - 128) / 127.0, ms = (b.f.ps[i] - 128) / 127.0, cf = sqrt(mc * mc + ms * ms);
            if (cf < 0.5) continue;
            const double m = (ctm[gi] - 128) / 127.0, ph = atan2(ms, mc);
            A += cf * m * cexp(-I * ph); sa += cf * fabs(m);
            double fr = ph / (2 * M_PI); fr -= floor(fr); const int k = (int)(fr * 16) & 15; bm[k] += m; bn[k] += 1;
        }
        fit_fields_free(&b);
    }
    double phi = -carg(A) / (2 * M_PI); phi -= floor(phi);
    printf("ct-fields: %dx%dx%d cells (h %d, %.3f um), CT %s %s at %.3f um; %zu material cells with CT; cs on %zu (%.3f; rays %lld, %lld with >= 2 peaks): mean CT spacing %.1f um",
           D, H, W, h, um, root, key, ctum, nct, ncs, nct ? (double)ncs / nct : 0, nray, npk, ncs ? csm / ncs : 0);
    if (ncr) printf(", where both: network %.1f um, median-ish ratio cs / sp exp(mean log) %.3f", spm / ncr, exp(csr / ncr));
    printf(" (%.0f s)\n", ct_now() - t0);
    printf("ct-fields: papyrus phase phi_pap %.3f (coherence %.3f over confident cells, parity radius %d cells); mean parity by fields' phase:", phi, sa > 0 ? cabs(A) / sa : 0, pr);
    for (int k = 0; k < 16; k++) printf(" %.2f:%+.2f", (k + 0.5) / 16, bn[k] ? bm[k] / bn[k] : 0);
    printf("\n");
    char p[1400]; int bad = 0;
    snprintf(p, sizeof p, "%s/ct.raw", dir); FILE *fo = fopen(p, "wb"); bad |= !fo || fwrite(ct, 1, n, fo) != n; if (fo) bad |= fclose(fo) != 0;
    snprintf(p, sizeof p, "%s/cs.raw", dir); fo = fopen(p, "wb"); bad |= !fo || fwrite(cs, 1, n, fo) != n; if (fo) bad |= fclose(fo) != 0;
    snprintf(p, sizeof p, "%s/ct.json", dir); fo = fopen(p, "w");
    if (fo) { fprintf(fo, "{\"ct_root\": \"%s\", \"ct_key\": \"%s\", \"ct_um\": %.9g, \"radius_um\": %.9g, \"prom\": %.9g, \"stride\": %d, \"phi_pap\": %.4f, \"phi_coherence\": %.4f}\n",
                      root, key, ctum, rum, prom, stride, phi, sa > 0 ? cabs(A) / sa : 0); bad |= fclose(fo) != 0; } else bad = 1;
    free(ct); free(cs); free(ctm); z3_close(cz);
    if (bad) { fprintf(stderr, "ct-fields: cannot write the planes into %s\n", dir); return 1; }
    return 0;
}
