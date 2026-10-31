/* ufsm ct-cal FIELDS-DIR OUT.cal --ct ROOT KEY --ct-um U [--nz 32] [--nth 32] [--band-mm 1] [--sigma 1.2] [--prom 1] [--threads 16]
   The density head's turn count calibrated against the CT's papyrus peaks: along radial rays from the umbilicus (--nth angles at --nz
   z levels evenly over the box, 2.5% .. 97.5%), per radial band of --band-mm, the CT peaks inside material (wrap-check's detector:
   gaussian --sigma native voxels, prominence >= --prom x the local sd) and the density turns (the integral of |n . r| / sp over the
   same material samples). Per (z level, angle sector, band) cell k = peaks / density turns (the CT's sheets per density turn), from
   cells with >= 3 density turns, smoothed over +-1 neighbouring cells (z, angle periodic, band) by summing peaks and turns; cells
   without enough data take their ring's (z level, band) value, then the box's. count-field --ct-cal OUT.cal scales the pooled count
   targets by k (so the ref counts the CT's sheets where the density head over- or under-counts them, e.g. crushed scrolls).
   OUT.cal: text, a header line "ufsm-ct-cal nz nth nb z0 z1 band_native" then nz x nth x nb values k (z-major, then angle, then band).
   --model M: the turns are a model's W increments instead of density turns (k = CT peaks per model turn; e.g. the cf ref, whose LS
   solve counts more than its density targets); --prev CAL: the written k multiplied by that table's (ref0 -> ct-cal --model ref0 --prev
   cal0 -> count-field --ct-cal: the scale that brings ref0's count to the CT's).
   Prints k by radius and overall. CPU, the CT planes read like wrap-check (the rows covering each native z, averaged). */
#include "fit.h"
#include "zarr3.h"
#include "store.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const char *cc_opt(int argc, char **argv, const char *k, const char *d) { for (int i = 3; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return argv[i + 1]; return d; }
static int cc_idx(int argc, char **argv, const char *k) { for (int i = 3; i < argc; i++) if (!strcmp(argv[i], k)) return i; return -1; }
static double cc_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

typedef struct {
    const fit_fields *f; double h, F, cy, cx, sigma, prom, band; int nth, nb, H, W;   /* f: one field row (box-local y / x cells); cy / cx the axis (native) */
    const float *ct; int64_t co[2], cn[2]; double org[2];   /* CT plane (CT voxels), origin (CT voxels, global); the box's native y / x origin */
    double *pk, *dn;   /* [nth][nb] sums */
    const fit_set *ref; double zl, roff[3];   /* --model: the turns are the model's W increments (its box-local = ours + roff) */
} cc_job;

static double cc_ctv(const cc_job *J, double y, double x) {   /* box-local native -> bilinear CT */
    const double cy = (y + J->org[0] + 0.5) * J->F - 0.5 - J->co[0], cx = (x + J->org[1] + 0.5) * J->F - 0.5 - J->co[1];
    const int64_t y0 = (int64_t)floor(cy), x0 = (int64_t)floor(cx); const double ty = cy - y0, tx = cx - x0;
    if (y0 < 0 || x0 < 0 || y0 + 1 >= J->cn[0] || x0 + 1 >= J->cn[1]) return NAN;
    const float *p = J->ct + y0 * J->cn[1] + x0;
    return (1 - ty) * ((1 - tx) * p[0] + tx * p[1]) + ty * ((1 - tx) * p[J->cn[1]] + tx * p[J->cn[1] + 1]);
}

static void cc_rays(void *ctx, int a0, int a1, int t) {
    cc_job *J = ctx; (void)t; const fit_fields *f = J->f; const double h = J->h, st = fmin(1.0, fmax(0.25, 1 / J->F));
    const double Y = J->H * h, X = J->W * h;
    for (int j = a0; j < a1; j++) {
        const double th = 2 * M_PI * (j + 0.5) / J->nth, sn = sin(th), cs = cos(th);
        double rmax = 0; for (double r = 0;; r += 16) { const double y = J->cy + r * sn, x = J->cx + r * cs; if (r > 0 && (y < 0 || x < 0 || y >= Y || x >= X)) break; rmax = r + 16; if (r > 4 * (Y + X)) break; }
        const int n = (int)(rmax / st) + 1;
        float *C = malloc(sizeof(float) * n), *Cs = malloc(sizeof(float) * n), *dn = malloc(sizeof(float) * n), *Wm = malloc(sizeof(float) * n); uint8_t *mat = calloc(n, 1);
        for (int i = 0; i < n; i++) {
            const double r = i * st, y = J->cy + r * sn, x = J->cx + r * cs; C[i] = NAN; dn[i] = 0;
            if (y < 0 || x < 0 || y >= Y || x >= X) continue;
            const int fy = (int)floor(y / h), fx = (int)floor(x / h); if (fy < 0 || fx < 0 || fy >= J->H || fx >= J->W) continue;
            const size_t ci = (size_t)fy * J->W + fx;
            C[i] = (float)cc_ctv(J, y, x); Wm[i] = J->ref ? (float)fit_set_W(J->ref, J->zl + J->roff[0], y + J->roff[1], x + J->roff[2]) : NAN;
            if (!f->w[ci] || !f->sp[ci]) continue;
            mat[i] = 1;
            const double nyy = f->t[1][ci] / 255.0, nxx = f->t[2][ci] / 255.0, nyx = f->t[5][ci] / 255.0 - 0.5;
            dn[i] = (float)(sqrt(fmax(0, sn * sn * nyy + cs * cs * nxx + 2 * sn * cs * nyx)) / (f->sp[ci] * h / 8.0));
        }
        /* peaks as wrap-check: gaussian smoothing, prominence within +-24 native over the sd within +-12 native */
        const int R = (int)ceil(3 * J->sigma / st); double *gk = malloc(sizeof(double) * (2 * R + 1));
        for (int d = -R; d <= R; d++) gk[d + R] = exp(-0.5 * (d * st / J->sigma) * (d * st / J->sigma));
        for (int i = 0; i < n; i++) {
            double s = 0, ws = 0;
            for (int d = -R; d <= R; d++) { const int k = i + d; if (k < 0 || k >= n || !(C[k] == C[k])) continue; s += gk[d + R] * C[k]; ws += gk[d + R]; }
            Cs[i] = ws > 0 ? (float)(s / ws) : NAN;
        }
        free(gk);
        const int L = (int)lround(12 / st), P = (int)lround(24 / st);
        double *ps = calloc(n + 1, sizeof(double)), *ps2 = calloc(n + 1, sizeof(double)); int *pc = calloc(n + 1, sizeof(int));
        for (int i = 0; i < n; i++) { const int ok = Cs[i] == Cs[i]; ps[i + 1] = ps[i] + (ok ? Cs[i] : 0); ps2[i + 1] = ps2[i] + (ok ? (double)Cs[i] * Cs[i] : 0); pc[i + 1] = pc[i] + ok; }
        double *bp = J->pk + (size_t)j * J->nb, *bd = J->dn + (size_t)j * J->nb;
        for (int i = 0; i < n; i++) {
            if (!mat[i]) continue;
            const int b = (int)(i * st / J->band); if (b >= J->nb) break;
            if (!J->ref) bd[b] += dn[i] * st;
            else if (i + 1 < n && mat[i + 1] && Wm[i] == Wm[i] && Wm[i + 1] == Wm[i + 1] && fabs(Wm[i + 1] - Wm[i]) < 0.5) bd[b] += Wm[i + 1] - Wm[i];
            if (i < 1 || i + 1 >= n || !(Cs[i] == Cs[i]) || !(Cs[i] > Cs[i - 1]) || Cs[i] < Cs[i + 1]) continue;
            const int a = i - L < 0 ? 0 : i - L, e = i + L + 1 > n ? n : i + L + 1, c = pc[e] - pc[a];
            if (c < 4) continue;
            const double mu = (ps[e] - ps[a]) / c, sd = sqrt(fmax(0, (ps2[e] - ps2[a]) / c - mu * mu)); if (!(sd > 0)) continue;
            double ml = Cs[i], mr = Cs[i];
            for (int k = i - 1; k >= 0 && k >= i - P; k--) { if (!(Cs[k] == Cs[k]) || Cs[k] > Cs[i]) break; if (Cs[k] < ml) ml = Cs[k]; }
            for (int k = i + 1; k < n && k <= i + P; k++) { if (!(Cs[k] == Cs[k]) || Cs[k] >= Cs[i]) break; if (Cs[k] < mr) mr = Cs[k]; }
            if (Cs[i] - fmax(ml, mr) >= J->prom * sd) bp[b] += 1;
        }
        free(ps); free(ps2); free(pc); free(C); free(Cs); free(dn); free(Wm); free(mat);
    }
}

int cmd_ct_cal(int argc, char **argv) {
    const int ci = cc_idx(argc, argv, "--ct");
    if (argc < 4 || ci < 0 || ci + 2 >= argc) {
        fprintf(stderr, "usage: ufsm ct-cal <fields-dir> <out.cal> --ct ROOT KEY --ct-um U [--model M] [--prev CAL] [--nz 32] [--nth 32] [--band-mm 1] [--sigma 1.2] [--prom 1] [--threads 16]\n"
                        "  the density head's turn count vs the CT's papyrus peaks along radial rays: k = peaks / density turns per (z, angle, band), for count-field --ct-cal\n");
        return 2;
    }
    const double t0 = cc_now(), ctum = atof(cc_opt(argc, argv, "--ct-um", "0"));
    const int nz = atoi(cc_opt(argc, argv, "--nz", "32")), nth = atoi(cc_opt(argc, argv, "--nth", "32")), nt = atoi(cc_opt(argc, argv, "--threads", "16"));
    const double sigma = atof(cc_opt(argc, argv, "--sigma", "1.2")), prom = atof(cc_opt(argc, argv, "--prom", "1")), bmm = atof(cc_opt(argc, argv, "--band-mm", "1"));
    if (!(ctum > 0) || nz < 1 || nth < 4 || !(bmm > 0)) { fprintf(stderr, "ct-cal: --ct-um, --nz, --nth, --band-mm\n"); return 2; }
    fit_set_threads(nt); fit_fields_skip_planes(1 | 2); fit_fields_ct_planes(0);
    fit_fields_buf g; if (fit_fields_load(argv[2], 0, 0, 0, &g)) { fprintf(stderr, "ct-cal: cannot read %s\n", argv[2]); return 1; }
    const int D = g.shape[0], H = g.shape[1], W = g.shape[2], na = (int)(D * g.h) + 1; const double h = g.h, um = g.um, F = um / ctum, band = bmm * 1000 / um;
    float *ay = malloc(na * sizeof(float)), *ax = malloc(na * sizeof(float)); memcpy(ay, g.cy, na * sizeof(float)); memcpy(ax, g.cx, na * sizeof(float));
    const long long org[3] = {g.origin[0], g.origin[1], g.origin[2]}; fit_fields_free(&g);
    const int nb = (int)ceil(hypot(H * h, W * h) / band) + 1;
    z3 *cz = z3_open(store_open(argv[ci + 1]), argv[ci + 2], cc_opt(argc, argv, "--cache", nullptr)); if (!cz) { fprintf(stderr, "ct-cal: %s\n", z3_error()); return 1; }
    const z3_meta *zm = z3_meta_of(cz);
    fit_set *mref = nullptr; double mo[3] = {0, 0, 0}; fit_ct_cal *prev = nullptr;
    if (*cc_opt(argc, argv, "--model", "")) { if (!(mref = fit_set_load(cc_opt(argc, argv, "--model", "")))) { fprintf(stderr, "ct-cal: cannot load --model\n"); return 1; } for (int d = 0; d < 3; d++) mo[d] = (double)mref->global->origin[d]; }
    if (*cc_opt(argc, argv, "--prev", "") && !(prev = fit_ct_cal_load(cc_opt(argc, argv, "--prev", "")))) { fprintf(stderr, "ct-cal: cannot read --prev\n"); return 1; }
    double *PK = calloc((size_t)nz * nth * nb, sizeof(double)), *DN = calloc((size_t)nz * nth * nb, sizeof(double));
    const double z0 = 0.025 * D * h, z1 = 0.975 * D * h;
    for (int zi = 0; zi < nz; zi++) {
        const double zl = floor(z0 + (zi + 0.5) * (z1 - z0) / nz) + 0.5, zg = zl + org[0]; const int fz = (int)floor(zl / h);
        if (fz < 0 || fz >= D) continue;
        fit_fields_buf b; if (fit_fields_load(argv[2], fz, fz + 1, 1, &b)) { fprintf(stderr, "ct-cal: cannot load field row %d\n", fz); return 1; }
        int64_t c0 = (int64_t)floor((zg - 0.5) * F), c1 = (int64_t)ceil((zg + 0.5) * F); if (c1 <= c0) c1 = c0 + 1; if (c0 < 0) c0 = 0; if (c1 > zm->shape[0]) c1 = zm->shape[0];
        int64_t o3[3] = {c0, (int64_t)floor(org[1] * F) - 2, (int64_t)floor(org[2] * F) - 2}, n3[3] = {c1 - c0, (int64_t)ceil(H * h * F) + 4, (int64_t)ceil(W * h * F) + 4};
        for (int d = 1; d < 3; d++) { if (o3[d] < 0) { n3[d] += o3[d]; o3[d] = 0; } if (o3[d] + n3[d] > zm->shape[d]) n3[d] = zm->shape[d] - o3[d]; }
        if (n3[0] <= 0 || n3[1] < 2 || n3[2] < 2) { fit_fields_free(&b); continue; }
        uint8_t *raw = malloc((size_t)n3[0] * n3[1] * n3[2]); if (!raw || z3_read(cz, o3, n3, raw, nt)) { fprintf(stderr, "ct-cal: CT read failed: %s\n", z3_error()); return 1; }
        const size_t pl = (size_t)n3[1] * n3[2]; float *ctp = calloc(pl, sizeof(float));
        for (int64_t a = 0; a < n3[0]; a++) for (size_t i = 0; i < pl; i++) ctp[i] += raw[a * pl + i];
        for (size_t i = 0; i < pl; i++) ctp[i] /= (float)n3[0];
        free(raw);
        const int iz = (int)fmin(fmax(zl, 0), na - 1);
        cc_job J = {.ref = mref, .zl = zl, .roff = {(double)org[0] - mo[0], (double)org[1] - mo[1], (double)org[2] - mo[2]}, .f = &b.f, .h = h, .F = F, .cy = ay[iz], .cx = ax[iz], .sigma = sigma, .prom = prom, .band = band, .nth = nth, .nb = nb, .H = H, .W = W,
                    .ct = ctp, .co = {o3[1], o3[2]}, .cn = {n3[1], n3[2]}, .org = {(double)org[1], (double)org[2]},
                    .pk = PK + (size_t)zi * nth * nb, .dn = DN + (size_t)zi * nth * nb};
        fit_par(nth, cc_rays, &J);
        free(ctp); fit_fields_free(&b);
        fprintf(stderr, "\rct-cal: z %d / %d (%.0f s)", zi + 1, nz, cc_now() - t0);
    }
    fprintf(stderr, "\n");
    /* k per cell: sums over +-1 neighbours (z, angle periodic, band) with >= 3 density turns; else the ring's (z, band) value; else the box's */
    double tp = 0, td = 0; for (size_t i = 0; i < (size_t)nz * nth * nb; i++) { tp += PK[i]; td += DN[i]; }
    const double kall = td > 0 ? tp / td : 1;
    float *K = malloc((size_t)nz * nth * nb * sizeof(float)); long long nown = 0, nring = 0, nbox = 0;
    for (int zi = 0; zi < nz; zi++) for (int bi = 0; bi < nb; bi++) {
        double rp = 0, rd = 0; for (int a = 0; a < nth; a++) { rp += PK[((size_t)zi * nth + a) * nb + bi]; rd += DN[((size_t)zi * nth + a) * nb + bi]; }
        for (int a = 0; a < nth; a++) {
            double sp_ = 0, sd_ = 0;
            for (int dz = -1; dz <= 1; dz++) for (int da = -1; da <= 1; da++) for (int db = -1; db <= 1; db++) {
                const int z2 = zi + dz, a2 = (a + da + nth) % nth, b2 = bi + db; if (z2 < 0 || z2 >= nz || b2 < 0 || b2 >= nb) continue;
                sp_ += PK[((size_t)z2 * nth + a2) * nb + b2]; sd_ += DN[((size_t)z2 * nth + a2) * nb + b2];
            }
            double k; if (sd_ >= 3) { k = sp_ / sd_; nown++; } else if (rd >= 3) { k = rp / rd; nring++; } else { k = kall; nbox++; }
            if (prev) { const double zc = z0 + (zi + 0.5) * (z1 - z0) / nz, thc = 2 * M_PI * (a + 0.5) / nth, rc = (bi + 0.5) * band; k *= fit_ct_cal_k(prev, zc, rc * sin(thc), rc * cos(thc)); }
            K[((size_t)zi * nth + a) * nb + bi] = (float)fmin(2, fmax(0.33, k));
        }
    }
    FILE *fo = fopen(argv[3], "w"); if (!fo) { fprintf(stderr, "ct-cal: cannot write %s\n", argv[3]); return 1; }
    fprintf(fo, "ufsm-ct-cal %d %d %d %.3f %.3f %.6f\n", nz, nth, nb, z0, z1, band);
    for (size_t i = 0; i < (size_t)nz * nth * nb; i++) fprintf(fo, "%.4f\n", K[i]);
    fclose(fo);
    printf("ct-cal: %s: %d z x %d angles x %d bands of %.1f mm: CT peaks per density turn %.3f overall (%.0f peaks / %.0f turns); cells own %lld, ring %lld, box %lld (%.0f s) -> %s\n",
           argv[2], nz, nth, nb, bmm, kall, tp, td, nown, nring, nbox, cc_now() - t0, argv[3]);
    printf("  radius(mm)  density-turns  CT/density\n");
    for (int bi = 0; bi < nb; bi++) { double p = 0, d = 0; for (int i = 0; i < nz * nth; i++) { p += PK[(size_t)i * nb + bi]; d += DN[(size_t)i * nb + bi]; } if (d >= 20) printf("  %4.0f-%-4.0f  %13.0f  %10.3f\n", bi * bmm, (bi + 1) * bmm, d, p / d); }
    if (mref) { fit_set_free(mref); }
    fit_ct_cal_free(prev); free(PK); free(DN); free(K); free(ay); free(ax); z3_close(cz);
    return 0;
}

/* count-field --ct-cal: k at a box-local native point (z, y, x) about the axis (cy, cx) at that z; the table loaded by fit_ct_cal_load */
typedef struct fit_ct_cal { int nz, nth, nb; double z0, z1, band; float *k; } fit_ct_cal;
fit_ct_cal *fit_ct_cal_load(const char *path) {
    FILE *f = fopen(path, "r"); if (!f) return nullptr;
    fit_ct_cal *c = calloc(1, sizeof *c);
    if (fscanf(f, "ufsm-ct-cal %d %d %d %lf %lf %lf", &c->nz, &c->nth, &c->nb, &c->z0, &c->z1, &c->band) != 6 || c->nz < 1 || c->nth < 1 || c->nb < 1) { fclose(f); free(c); return nullptr; }
    const size_t n = (size_t)c->nz * c->nth * c->nb; c->k = malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) if (fscanf(f, "%f", &c->k[i]) != 1) { fclose(f); free(c->k); free(c); return nullptr; }
    fclose(f); return c;
}
void fit_ct_cal_free(fit_ct_cal *c) { if (c) { free(c->k); free(c); } }
double fit_ct_cal_k(const fit_ct_cal *c, double z, double dy, double dx) {   /* trilinear in z level / angle (periodic) / band centres */
    const double fz = (z - c->z0) / (c->z1 - c->z0) * c->nz - 0.5, th = atan2(dy, dx), fa = (th < 0 ? th + 2 * M_PI : th) / (2 * M_PI) * c->nth - 0.5, fb = hypot(dy, dx) / c->band - 0.5;
    const int iz = (int)floor(fz), ia = (int)floor(fa), ib = (int)floor(fb); const double tz = fz - iz, ta = fa - ia, tb = fb - ib;
    double s = 0;
    for (int q = 0; q < 8; q++) {
        int z2 = iz + (q >> 2 & 1), a2 = ia + (q >> 1 & 1), b2 = ib + (q & 1);
        z2 = z2 < 0 ? 0 : z2 >= c->nz ? c->nz - 1 : z2; a2 = ((a2 % c->nth) + c->nth) % c->nth; b2 = b2 < 0 ? 0 : b2 >= c->nb ? c->nb - 1 : b2;
        const double w = (q >> 2 & 1 ? tz : 1 - tz) * (q >> 1 & 1 ? ta : 1 - ta) * (q & 1 ? tb : 1 - tb);
        s += w * c->k[((size_t)z2 * c->nth + a2) * c->nb + b2];
    }
    return s;
}
