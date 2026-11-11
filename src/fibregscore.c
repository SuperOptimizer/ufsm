/* ufsm fiber-regscore: how well a scan-to-scan map puts published fibre polylines (AFV, traced in the map's COARSE scan) onto a fibre
   presence prediction made on the map's FINE scan. A registration check that shares nothing with the CT box matching of ufsm reg:
   fibres are thin (~200 um wide, presence falls off within ~2-3 9.6 um voxels), so the mean presence along the mapped polylines peaks
   sharply at the right offset. Per region (a cube of the prediction level around a random fibre point): the mean presence at the mapped
   points, and the per-axis shift (fine L0 voxels -> um) that maximises it (scan -S..S, parabola at the peak) = the local residual of the
   map along x, y, z. Map: --map reg.json (ufsm reg: coarse = the AFV's scan, fine = the prediction's) or --tr T.json (an affine
   transformation_matrix fine L0 xyz -> coarse L0 xyz, the published scrollprize convention). */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "afv.h"
#include "json.h"
#include "reg.h"
#include "store.h"
#include "zarr2.h"

static const char *rs_opt(int argc, char **argv, const char *k, const char *d) { for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return argv[i + 1]; return d; }
static uint64_t rs_s = 0x5eed;
static double rs_u(void) { uint64_t z = (rs_s += 0x9E3779B97F4A7C15ull); z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; z ^= z >> 31; return (z >> 11) * 0x1.0p-53; }

static int load_affine(const char *path, raff *A) {
    FILE *f = fopen(path, "rb"); if (!f) { fprintf(stderr, "fiber-regscore: cannot read %s\n", path); return -1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET); char *t = malloc(n + 1); if (fread(t, 1, n, f) != (size_t)n) n = 0; t[n] = 0; fclose(f);
    json *J = json_parse(t, n); free(t);
    const json *m = J ? json_get(J, "transformation_matrix") : nullptr; if (!m) m = J ? json_get(J, "A") : nullptr;
    if (!m) { fprintf(stderr, "fiber-regscore: %s has no transformation_matrix\n", path); json_free(J); return -1; }
    for (int r = 0; r < 3; r++) for (int c = 0; c < 4; c++) A->m[r][c] = json_num(json_at(json_at(m, r), c), NAN);
    json_free(J); return 0;
}
/* trilinear at level coords x y z of a zyx block (origin o), NaN outside */
static double blk_at(const uint8_t *v, const int64_t o[3], const int64_t n[3], double x, double y, double z) {
    x -= o[2]; y -= o[1]; z -= o[0];
    const int64_t x0 = (int64_t)floor(x), y0 = (int64_t)floor(y), z0 = (int64_t)floor(z);
    if (x0 < 0 || y0 < 0 || z0 < 0 || x0 + 1 >= n[2] || y0 + 1 >= n[1] || z0 + 1 >= n[0]) return NAN;
    const double fx = x - x0, fy = y - y0, fz = z - z0; double s = 0;
    for (int c = 0; c < 8; c++) {
        const int dx = c & 1, dy = (c >> 1) & 1, dz = c >> 2;
        s += (dx ? fx : 1 - fx) * (dy ? fy : 1 - fy) * (dz ? fz : 1 - fz) * v[((size_t)(z0 + dz) * n[1] + (y0 + dy)) * n[2] + (x0 + dx)];
    }
    return s;
}

static int rs_afv_fine = 0;   /* --afv-fine 1: the AFV is in the map's FINE scan and the volume is the COARSE scan */
static void rs_fwd(const rmap *m, const double a[3], double v[3]) { if (rs_afv_fine) rmap_f2c(m, a, v); else rmap_c2f(m, a, v); }
static void rs_bwd(const rmap *m, const double v[3], double a[3]) { if (rs_afv_fine) rmap_c2f(m, v, a); else rmap_f2c(m, v, a); }
int cmd_fiber_regscore(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm fiber-regscore <coarse-frame.afv> (--map reg.json | --tr T.json) --pred ROOT KEY [--level 2] [--regions 40] [--box 192]\n"
                                    "  [--identity 1] [--afv-fine 1 (AFV in the fine scan, volume = the coarse scan)] [--place T.json (region placement)] [--um U (volume L0 voxel)] [--smax 12 (fine L0 voxels)] [--step 1] [--seed 1] [--extent 8096,7888,14258 (coarse L0 x,y,z)] [--cache DIR] [--threads 8] [--out regions.tsv]\n"
                                    "  KEY: the volume group (zarr v3 / volcomp or v2 pyramid; level L voxel = 2^L L0 voxels): a fibre presence prediction or the CT\n"); return 2; }
    const char *mp = rs_opt(argc, argv, "--map", nullptr), *tp = rs_opt(argc, argv, "--tr", nullptr), *cache = rs_opt(argc, argv, "--cache", nullptr);
    int pi = 0; for (int i = 1; i + 2 < argc; i++) if (!strcmp(argv[i], "--pred")) pi = i;
    const int ident = !!*rs_opt(argc, argv, "--identity", "");
    rs_afv_fine = !!atoi(rs_opt(argc, argv, "--afv-fine", "0"));   /* --identity 1: no map; the volume is in the AFV's own frame (e.g. its CT) */
    if ((!mp + !tp + !ident != 2) || !pi) { fprintf(stderr, "fiber-regscore: one of --map / --tr / --identity 1, and --pred ROOT KEY\n"); return 2; }
    const int L = atoi(rs_opt(argc, argv, "--level", "2")), nreg = atoi(rs_opt(argc, argv, "--regions", "40")), B = atoi(rs_opt(argc, argv, "--box", "192")), nt = atoi(rs_opt(argc, argv, "--threads", "8"));
    const double smax = atof(rs_opt(argc, argv, "--smax", "12")), step = atof(rs_opt(argc, argv, "--step", "1")), sc = ldexp(1.0, L);
    rs_s = (uint64_t)atoll(rs_opt(argc, argv, "--seed", "1")) * 0x9E3779B97F4A7C15ull + 1;
    rmap M0 = {0}; const rmap *M = &M0; double fum = 0, cum = 0;
    if (mp) { rvol *fv; M = reg_map_open(mp, cache, &fv, &cum, &fum); if (!M) return 1; }
    else if (tp) { if (load_affine(tp, &M0.A)) return 1; M0.Ainv = raff_inv(M0.A); M0.D = nullptr; }
    else { M0.A = raff_ident(); M0.Ainv = M0.A; M0.D = nullptr; }
    rmap P0 = {0}; const rmap *PL = nullptr;   /* --place T.json: regions placed with this affine, so different maps are scored on the same regions */
    if (*rs_opt(argc, argv, "--place", "")) { if (load_affine(rs_opt(argc, argv, "--place", ""), &P0.A)) return 1; P0.Ainv = raff_inv(P0.A); PL = &P0; }
    if (*rs_opt(argc, argv, "--um", "")) fum = atof(rs_opt(argc, argv, "--um", ""));   /* the volume's L0 voxel (um), for the residuals */
    rvol *Z = rvol_open(argv[pi + 1], argv[pi + 2], cache, 1.0);   /* zarr v3 / volcomp or v2 pyramid; level L = 2^L L0 voxels */
    if (!Z || L >= Z->nl || (!Z->z[L] && !Z->z2[L])) { fprintf(stderr, "fiber-regscore: cannot open level %d of %s %s\n", L, argv[pi + 1], argv[pi + 2]); return 1; }
    const struct { int64_t shape[3]; } zm0 = {{Z->shape[L][0], Z->shape[L][1], Z->shape[L][2]}}, *zm = &zm0;
    afv *A = afv_open(argv[2]); if (!A) return 1;
    /* fibre points to seed regions: the runs in a few random coarse boxes over the AFV's extent */
    double ext[3]; if (sscanf(rs_opt(argc, argv, "--extent", "8096,7888,14258"), "%lf,%lf,%lf", &ext[0], &ext[1], &ext[2]) != 3) return 2;   /* coarse L0 x,y,z */
    FILE *of = *rs_opt(argc, argv, "--out", "") ? fopen(rs_opt(argc, argv, "--out", ""), "w") : nullptr;
    if (of) fprintf(of, "region\tz\ty\tx\tpoints\tpres0\tpresbest\tdx_um\tdy_um\tdz_um\n");
    const int ns = (int)floor(smax / step + 1e-9), nsh = 2 * ns + 1; const double fum0 = fum > 0 ? fum : 2.4;
    double sum0 = 0, sumb = 0, sd2 = 0; int nok = 0; double *res = malloc((size_t)nreg * 3 * sizeof *res);
    uint8_t *blk = malloc((size_t)B * B * B);
    for (int r = 0, tries = 0; r < nreg && tries < nreg * 400; tries++) {
        double slo[3], shi[3]; for (int d = 0; d < 3; d++) { slo[d] = rs_u() * (ext[d] - 200); shi[d] = slo[d] + 200; }
        afv_run *R0; const int n0 = afv_query(A, slo, shi, &R0); if (n0 <= 0) { if (n0 == 0) afv_runs_free(R0, 0); continue; }
        const afv_run *q = &R0[(int)(rs_u() * n0) % n0]; double c0[3]; memcpy(c0, q->xyz + 3 * (size_t)((int)(rs_u() * q->n) % q->n), sizeof c0);
        afv_runs_free(R0, n0); double f[3]; rs_fwd(PL ? PL : M, c0, f);   /* fine L0 xyz (region placement: --place, else the scored map) */
        int64_t o[3] = {(int64_t)llround((f[2] + 0.5) / sc - 0.5) - B / 2, (int64_t)llround((f[1] + 0.5) / sc - 0.5) - B / 2, (int64_t)llround((f[0] + 0.5) / sc - 0.5) - B / 2};
        const int64_t n[3] = {B, B, B};
        if (o[0] < 0 || o[1] < 0 || o[2] < 0 || o[0] + B > zm->shape[0] || o[1] + B > zm->shape[1] || o[2] + B > zm->shape[2]) continue;
        /* the coarse bbox of the region (corners mapped back), the runs in it, mapped forward */
        double clo[3] = {1e30, 1e30, 1e30}, chi[3] = {-1e30, -1e30, -1e30};
        for (int cc = 0; cc < 8; cc++) {
            const double p[3] = {((o[2] + ((cc & 1) ? B : 0)) + 0.5) * sc - 0.5, ((o[1] + ((cc & 2) ? B : 0)) + 0.5) * sc - 0.5, ((o[0] + ((cc & 4) ? B : 0)) + 0.5) * sc - 0.5}; double cq[3];
            rs_bwd(M, p, cq); for (int d = 0; d < 3; d++) { clo[d] = fmin(clo[d], cq[d]); chi[d] = fmax(chi[d], cq[d]); }
        }
        afv_run *Rr; const int nr = afv_query(A, clo, chi, &Rr); if (nr <= 0) continue;
        size_t np = 0, cap = 1 << 16; float *pts = malloc(cap * 3 * sizeof *pts);   /* level xyz */
        const double mg = smax / sc + 2;
        for (int i = 0; i < nr; i++) for (int j = 0; j < Rr[i].n; j++) {
            double ff[3]; rs_fwd(M, Rr[i].xyz + 3 * (size_t)j, ff);
            const double lx = (ff[0] + 0.5) / sc - 0.5, ly = (ff[1] + 0.5) / sc - 0.5, lz = (ff[2] + 0.5) / sc - 0.5;
            if (lx < o[2] + mg || ly < o[1] + mg || lz < o[0] + mg || lx > o[2] + B - 1 - mg || ly > o[1] + B - 1 - mg || lz > o[0] + B - 1 - mg) continue;
            if (np == cap) { cap *= 2; pts = realloc(pts, cap * 3 * sizeof *pts); }
            pts[3 * np] = (float)lx; pts[3 * np + 1] = (float)ly; pts[3 * np + 2] = (float)lz; np++;
        }
        afv_runs_free(Rr, nr);
        if (np < (size_t)atoi(rs_opt(argc, argv, "--min-pts", "800"))) { free(pts); continue; }
        if (rvol_read_nt(Z, L, o, n, blk, nt)) { fprintf(stderr, "fiber-regscore: read failed\n"); free(pts); continue; }   /* read only regions with enough points */
        double best[3], bv = -1, p0 = 0;
        for (int a = 0; a < 3; a++) {   /* per axis (x, y, z): mean presence vs shift */
            double *pr = malloc(nsh * sizeof *pr); int bi = 0;
            for (int si = 0; si < nsh; si++) {
                const double sh = (si - ns) * step / sc; double sum = 0; size_t cnt = 0;
                for (size_t i = 0; i < np; i++) { double v = blk_at(blk, o, n, pts[3 * i] + (a == 0) * sh, pts[3 * i + 1] + (a == 1) * sh, pts[3 * i + 2] + (a == 2) * sh); if (v == v) { sum += v; cnt++; } }
                pr[si] = cnt ? sum / cnt : 0; if (pr[si] > pr[bi]) bi = si;
            }
            if (a == 0) p0 = pr[ns];
            double off = bi - ns;
            if (bi > 0 && bi < nsh - 1) { const double d = pr[bi - 1] - 2 * pr[bi] + pr[bi + 1]; if (d < 0) off += 0.5 * (pr[bi - 1] - pr[bi + 1]) / d; }
            best[a] = off * step * fum0; if (pr[bi] > bv) bv = pr[bi];
            free(pr);
        }
        if (of) fprintf(of, "%d\t%lld\t%lld\t%lld\t%zu\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n", r, (long long)o[0], (long long)o[1], (long long)o[2], np, p0, bv, best[0], best[1], best[2]);
        sum0 += p0; sumb += bv; for (int a = 0; a < 3; a++) { res[3 * r + a] = best[a]; sd2 += best[a] * best[a]; }
        nok++; r++; free(pts);
        fprintf(stderr, "  region %d: %zu points, presence %.1f (best %.1f), residual x %.1f y %.1f z %.1f um\n", r, np, p0, bv, best[0], best[1], best[2]);
    }
    if (!nok) { fprintf(stderr, "fiber-regscore: no region with enough fibre points\n"); return 1; }
    double *mag = malloc(nok * sizeof *mag); for (int i = 0; i < nok; i++) mag[i] = sqrt(res[3 * i] * res[3 * i] + res[3 * i + 1] * res[3 * i + 1] + res[3 * i + 2] * res[3 * i + 2]);
    for (int i = 1; i < nok; i++) for (int j = i; j > 0 && mag[j] < mag[j - 1]; j--) { const double t = mag[j]; mag[j] = mag[j - 1]; mag[j - 1] = t; }
    printf("fiber-regscore: %d regions: mean presence at the map %.2f, at the per-axis best %.2f; residual |d| p50 %.1f um p90 %.1f um max %.1f um (rms per axis %.1f um)\n",
           nok, sum0 / nok, sumb / nok, mag[nok / 2], mag[(int)(0.9 * (nok - 1))], mag[nok - 1], sqrt(sd2 / (3.0 * nok)));
    if (of) fclose(of);
    afv_close(A); rvol_close(Z); free(blk); free(res); free(mag);
    return 0;
}
