/* ufsm field-disagree (src/fielddis.h) on synthetic fields: flat sheets along z (60 um apart) predicted by a 9.6 um model A and a
   2.4 um model B (four times finer cells, the same 614 um box): identical sheets agree after the resolution mapping; a half-turn shift in one
   quarter of the box flags pos there only; a denser sheet set (40 um) in one half flags cnt there; a one-turn winding mod 4 error in a thin slab
   flags w4 only (same baseline, and chained over the 9.6 um model's 4x longer one); against A as the GT the flags find B's errors (precision / recall); the command writes the arrays and ranks the shifted
   regions first. */
#include "fielddis.h"
#include "fit.h"
#include "store.h"
#include "zarr3.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

enum { M_SAME, M_SHIFT, M_DENSE, M_W4 };
static const double P0 = 96 * 9.6, BOX = 64 * 9.6, SLAB = 32 * 2.4;   /* the box in um; the w4 error slab: one 2.4 um baseline thick */   /* the box in um */
/* a field set of n^3 cells of um each (h 1) over the box; mode picks the error */
static int make(const char *dir, int n, double um, int mode) {
    const size_t N = (size_t)n * n * n; uint8_t *a[14];
    for (int k = 0; k < 14; k++) { a[k] = malloc(N); if (!a[k]) return -1; }
    for (int z = 0; z < n; z++) for (int y = 0; y < n; y++) for (int x = 0; x < n; x++) {
        const size_t i = ((size_t)z * n + y) * n + x; const double zu = (z + 0.5) * um, yu = (y + 0.5) * um, xu = (x + 0.5) * um;
        double sp = 60, sh = 0, w4e = 0;
        if (mode == M_SHIFT && yu < BOX / 2 && xu < BOX / 2) sh = 0.5;
        if (mode == M_DENSE && xu >= BOX / 2) sp = 40;
        if (mode == M_W4 && zu >= BOX / 2 && zu < BOX / 2 + SLAB) w4e = 1;
        const double ph = zu / sp + sh;
        a[0][i] = (uint8_t)lround(128 + 127 * 0.9 * cos(2 * M_PI * ph)); a[1][i] = (uint8_t)lround(128 + 127 * 0.9 * sin(2 * M_PI * ph));
        a[2][i] = 255; a[3][i] = 0; a[4][i] = 0; a[5][i] = a[6][i] = a[7][i] = 128;   /* n = z */
        a[8][i] = (uint8_t)lround(sp / um * 8); a[9][i] = 255;
        const double t = 32 * um / sp + w4e;   /* relative winding over w4_d = 32 native voxels along z, turns */
        a[10][i] = (uint8_t)(((long)lround(t * 64) % 256 + 256) % 256); a[11][i] = 0; a[12][i] = 0; a[13][i] = 255;
    }
    const long long o[3] = {(long long)lround(P0 / um), (long long)lround(P0 / um), (long long)lround(P0 / um)}; const int shp[3] = {n, n, n};
    float *cy = calloc((size_t)n + 1, 4), *cx = calloc((size_t)n + 1, 4);
    const int rc = fit_fields_save_w4(dir, o, shp, 1, um, a, 32, cy, cx, n + 1);
    for (int k = 0; k < 14; k++) free(a[k]);
    free(cy); free(cx);
    return rc;
}
/* shares of valid cells flagged by plane p >= thr inside / outside the region test sel (cell centres, um rel. to the box) */
static void shares(const fdis_out *o, const uint8_t *pl, int thr, int (*sel)(const double *), double *in, double *out) {
    size_t ni = 0, fi = 0, no = 0, fo = 0;
    for (int z = 0; z < o->D; z++) for (int y = 0; y < o->H; y++) for (int x = 0; x < o->W; x++) {
        const size_t i = ((size_t)z * o->H + y) * o->W + x; if (!(o->flag[i] & 1)) continue;
        double p[3]; fdis_cell_um(o, z, y, x, p); for (int d = 0; d < 3; d++) p[d] -= P0;
        if (sel(p)) { ni++; fi += pl[i] >= thr; } else { no++; fo += pl[i] >= thr; }
    }
    *in = ni ? (double)fi / ni : -1; *out = no ? (double)fo / no : -1;
}
/* the region interiors (a cell's margin from the error boundaries: the 9.6 um cells blend across them) */
static int sel_shift(const double *p) { return p[1] < BOX / 2 - 12 && p[2] < BOX / 2 - 12; }
static int sel_shift_far(const double *p) { return !(p[1] < BOX / 2 + 12 && p[2] < BOX / 2 + 12); }
static int sel_dense(const double *p) { return p[2] >= BOX / 2 + 12; }
static int sel_dense_far(const double *p) { return !(p[2] >= BOX / 2 - 12); }
static int sel_w4s(const double *p) { return p[0] >= BOX / 2 + 12 && p[0] < BOX / 2 + SLAB - 12; }   /* the same baseline: the slab */
static int sel_w4s_far(const double *p) { return !(p[0] >= BOX / 2 - 12 && p[0] < BOX / 2 + SLAB + 12); }
static int sel_w4c(const double *p) { return p[0] >= BOX / 2 - 3 * SLAB + 12 && p[0] < BOX / 2 + SLAB - 12; }   /* chained over 4 baselines: one step in the slab */
static int sel_w4c_far(const double *p) { return !(p[0] >= BOX / 2 - 3 * SLAB - 24 && p[0] < BOX / 2 + SLAB + 24); }

int main(void) {
    char root[] = "/tmp/test_fielddis.XXXXXX"; if (!mkdtemp(root)) { perror("mkdtemp"); return 1; }
    char pa[256], pb[256], ps[256], pd[256], pw[256], po[256];
    snprintf(pa, sizeof pa, "%s/a", root); snprintf(pb, sizeof pb, "%s/b", root); snprintf(ps, sizeof ps, "%s/shift", root);
    snprintf(pd, sizeof pd, "%s/dense", root); snprintf(pw, sizeof pw, "%s/w4", root); snprintf(po, sizeof po, "%s/out", root);
    CHECK(!make(pa, 64, 9.6, M_SAME) && !make(pb, 256, 2.4, M_SAME) && !make(ps, 256, 2.4, M_SHIFT) && !make(pd, 256, 2.4, M_DENSE) && !make(pw, 256, 2.4, M_W4), "make fields");
    fit_fields_buf A, B, S, Dn, W4;
    CHECK(!fit_fields_load(pa, 0, 0, 1, &A) && !fit_fields_load(pb, 0, 0, 1, &B) && !fit_fields_load(ps, 0, 0, 1, &S) && !fit_fields_load(pd, 0, 0, 1, &Dn) && !fit_fields_load(pw, 0, 0, 1, &W4), "load fields");
    if (fails) return 1;
    fit_set_threads(4);
    const fdis_cfg c = {.conf = 0.2, .gt_conf = 0.3, .ray_um = 100, .step_um = 0, .stride = 1, .metrics = 7};
    fdis_out o; double in, out;
    {   /* the same sheets at 9.6 and 2.4 um: no disagreement, on every cell */
        const fit_fields_buf *P[2] = {&A, &B};
        CHECK(!fdis_run(2, P, nullptr, &c, &o), "run same");
        size_t n = (size_t)o.D * o.H * o.W, mp = 0, mc = 0, mw = 0;
        for (size_t i = 0; i < n; i++) { if (o.pos[i] > mp) mp = o.pos[i]; if (o.cnt[i] > mc) mc = o.cnt[i]; if (o.w4[i] > mw) mw = o.w4[i]; }
        CHECK(o.nvalid == n && o.nray == n && o.nw4 > n * 9 / 10 && o.nw4 < n, "same (w4: the 2.4 um chain leaves the box in the top 230 um): valid %zu ray %zu w4 %zu of %zu", o.nvalid, o.nray, o.nw4, n);
        CHECK(o.ndis == 0 && mp < 0.03 * 510 && mc < 0.1 * 64 && mw < 0.05 * 128, "same: dis %zu, max pos %zu cnt %zu w4 %zu", o.ndis, mp, mc, mw);
        fdis_free(&o);
        /* the reference may be the finer set (every 4th cell) */
        const fit_fields_buf *Q[2] = {&B, &A}; const fdis_cfg c4 = {.conf = 0.2, .gt_conf = 0.3, .ray_um = 100, .stride = 4, .metrics = 7};
        CHECK(!fdis_run(2, Q, nullptr, &c4, &o) && o.D == 64 && o.ndis == 0 && o.nvalid == (size_t)64 * 64 * 64, "fine ref, stride 4: D %d dis %zu valid %zu", o.D, o.ndis, o.nvalid);
        fdis_free(&o);
    }
    {   /* a half-turn shift in one quarter: pos there, not elsewhere; vs A as the GT, the flags find B's errors */
        const fit_fields_buf *P[2] = {&A, &S};
        CHECK(!fdis_run(2, P, &A, &c, &o), "run shift");
        shares(&o, o.pos, 128, sel_shift, &in, &out); CHECK(in > 0.98, "shift pos inside %.3f", in);
        shares(&o, o.pos, 128, sel_shift_far, &out, &in); CHECK(out < 0.01, "shift pos far %.3f", out);
        shares(&o, o.dis, 128, sel_shift, &in, &out); CHECK(in > 0.98, "shift dis inside %.3f", in);
        shares(&o, o.dis, 128, sel_shift_far, &out, &in); CHECK(out < 0.01, "shift dis far %.3f", out);
        const fdis_gtstat *g = &o.gt; const double prec = (double)g->flag_wrong[3] / g->flag[3], rec = (double)g->flag_wrong[3] / g->nwrong;
        CHECK(g->n == o.nvalid && g->wrong_k[0] == 0 && g->nwrong == g->wrong_k[1] && g->nboth == 0, "gt tallies n %zu valid %zu wrong A %zu B %zu both %zu", g->n, o.nvalid, g->wrong_k[0], g->wrong_k[1], g->nboth);
        CHECK(prec > 0.95 && rec > 0.95 && fabs((double)g->nwrong / g->n - 0.25) < 0.06, "gt: precision %.3f recall %.3f base %.3f", prec, rec, (double)g->nwrong / g->n);
        CHECK(g->w4n > g->n / 10 && g->w4wrong * 100 < g->w4n, "gt w4: %zu of %zu cells scored (all three axes inside the box), %zu off", g->w4n, g->n, g->w4wrong);
        fdis_free(&o);
    }
    {   /* denser sheets in one half: cnt there; w4 too (the winding gradient differs), nothing far from it */
        const fit_fields_buf *P[2] = {&A, &Dn};
        CHECK(!fdis_run(2, P, nullptr, &c, &o), "run dense");
        shares(&o, o.cnt, 32, sel_dense, &in, &out); CHECK(in > 0.95, "dense cnt inside %.3f", in);
        shares(&o, o.cnt, 32, sel_dense_far, &out, &in); CHECK(out < 0.01, "dense cnt far %.3f", out);
        fdis_free(&o);
    }
    {   /* a one-turn winding mod 4 error in a slab one 2.4 um baseline thick: w4 there (same baseline: B against it; 9.6 um A: the 2.4 um
           planes chained over A's 4x longer baseline, so the cells within 3 baselines below the slab see it too), pos / cnt nowhere */
        const fit_fields_buf *P[2] = {&B, &W4}; const fdis_cfg c4 = {.conf = 0.2, .gt_conf = 0.3, .ray_um = 100, .stride = 4, .metrics = 7};
        CHECK(!fdis_run(2, P, nullptr, &c4, &o), "run w4 same baseline");
        shares(&o, o.w4, 64, sel_w4s, &in, &out); CHECK(in > 0.98, "w4 same baseline inside %.3f", in);
        shares(&o, o.w4, 64, sel_w4s_far, &out, &in); CHECK(out < 0.01, "w4 same baseline far %.3f", out);
        fdis_free(&o);
        const fit_fields_buf *Q[2] = {&A, &W4};
        CHECK(!fdis_run(2, Q, &A, &c, &o), "run w4 chained");
        shares(&o, o.w4, 64, sel_w4c, &in, &out); CHECK(in > 0.98, "w4 chained inside %.3f", in);
        shares(&o, o.w4, 64, sel_w4c_far, &out, &in); CHECK(out < 0.01, "w4 chained far %.3f", out);
        size_t np = 0, nc = 0, n = (size_t)o.D * o.H * o.W; for (size_t i = 0; i < n; i++) { np += o.pos[i] >= 128; nc += o.cnt[i] >= 32; }
        const fdis_gtstat *g = &o.gt;   /* against A's phase as the GT: W4's winding is off, A's is not, and the w4 flags find it */
        CHECK(g->w4n_k[0] == g->w4n && g->w4wrong_k[0] == 0 && g->w4wrong_k[1] > g->w4n / 4 && g->w4flag_wrong * 100 > 98 * g->w4flag && g->w4flag_wrong * 100 > 98 * g->w4wrong,
              "gt w4: scored %zu, A off %zu, W4 off %zu, flagged %zu, both %zu", g->w4n, g->w4wrong_k[0], g->w4wrong_k[1], g->w4flag, g->w4flag_wrong);
        CHECK(g->nwrong == 0, "w4 case: %zu sheet errors", g->nwrong);
        CHECK(np == 0 && nc == 0 && o.nw4 > n / 3 && o.nw4 < n, "w4 case: pos %zu cnt %zu flagged, w4 valid %zu of %zu (the chain leaves the box near the top)", np, nc, o.nw4, n);
        fdis_free(&o);
    }
    {   /* field-label: B (2.4 um) half a turn off in one quarter; the agreeing cells around the quarter decide for A near its edges,
           deep inside (no agreeing cell on the tangent ring) it abstains; every label is right against A as the GT, B is never picked.
           With the grids swapped (S the grid, A second) the picks are B's. */
        const fdl_cfg lc = {.conf = 0.2, .agree = 0.1, .max_err = 0.2, .min_margin = 0.1, .rmin = 2, .rmax = 8, .nang = 12, .min_samp = 8};
        const size_t N = (size_t)64 * 64 * 64; uint8_t *ph = malloc(N), *cf = malloc(N), *kd = malloc(N); fdl_stat st;
        CHECK(!fdis_label(&A, &S, &A, &lc, ph, cf, kd, &st), "field-label run");
        size_t nk[6] = {0}; for (size_t i = 0; i < N; i++) nk[kd[i] < 6 ? kd[i] : 0]++;
        CHECK(st.ndis > 0 && st.nlab > st.ndis / 10 && st.nlab < st.ndis && st.ncorrect == st.nlab && st.nB == 0 && nk[4] > 0, "labels: dis %zu labelled %zu correct %zu B %zu abstain %zu", st.ndis, st.nlab, st.ncorrect, st.nB, nk[4]);
        CHECK(st.nagree > N / 2 && st.agree_ok == st.nagree, "agree labels %zu ok %zu", st.nagree, st.agree_ok);
        int ok = 1; for (size_t i = 0; i < N; i++) if (kd[i] == 2 && (ph[i] > 251 || !cf[i])) ok = 0;
        CHECK(ok, "label codes");
        free(ph); free(cf); free(kd);
        const size_t M = (size_t)256 * 256 * 256; ph = malloc(M); cf = malloc(M); kd = malloc(M);
        CHECK(!fdis_label(&S, &A, &A, &lc, ph, cf, kd, &st) && st.nA == 0 && st.nB > 0 && st.ncorrect == st.nlab, "swapped: A %zu B %zu correct %zu of %zu", st.nA, st.nB, st.ncorrect, st.nlab);
        free(ph); free(cf); free(kd);
    }
    {   /* the command: arrays, gt.tsv, regions ranked (16-cell regions of the 9.6 um frame: the two shifted ones first) */
        char *argv[] = {"ufsm", "field-disagree", po, pa, ps, "--gt", pa, "--region", "16", "--threads", "4", "--tag", "t", nullptr};
        CHECK(cmd_field_disagree(13, argv) == 0, "command");
        store_global_init(); z3 *z = z3_open(store_open(po), "dis", nullptr);
        CHECK(z && z3_meta_of(z)->shape[0] == 64, "dis array");
        if (z) {
            uint8_t *v = malloc(64 * 64 * 64); const int64_t oo[3] = {0, 0, 0}, nn[3] = {64, 64, 64};
            CHECK(!z3_read(z, oo, nn, v, 2) && v[(4 * 64 + 4) * 64 + 4] >= 128 && v[(4 * 64 + 60) * 64 + 60] < 128, "dis array values");
            free(v); z3_close(z);
        }
        char p[300]; snprintf(p, sizeof p, "%s/regions.tsv", po); FILE *f = fopen(p, "r"); char line[1024]; int nr = 0, top_ok = 1;
        CHECK(f != nullptr, "regions.tsv");
        while (f && fgets(line, sizeof line, f)) {
            if (line[0] == '#' || !strncmp(line, "rank", 4)) continue;
            int rk; char tg[16]; long long rz, ry, rx; if (sscanf(line, "%d %15s %lld %lld %lld", &rk, tg, &rz, &ry, &rx) != 5) continue;
            nr++; if (rk <= 16 && !(ry < 128 && rx < 128)) top_ok = 0;
        }
        if (f) fclose(f);
        CHECK(nr == 64 && top_ok, "regions: %d rows, the shifted quarter first %d", nr, top_ok);
        snprintf(p, sizeof p, "%s/gt.tsv", po); CHECK(access(p, R_OK) == 0, "gt.tsv");
    }
    {   /* field-label + field-label-raster: the labels of a box as a 9.6 um phase raster over one listed region (16^3 voxels), unknown elsewhere */
        char pl[300], pr[300], rf[300]; snprintf(pl, sizeof pl, "%s/lab", root); snprintf(pr, sizeof pr, "%s/raster", root); snprintf(rf, sizeof rf, "%s/regions.txt", root);
        char *a1[] = {"ufsm", "field-label", pl, pa, ps, "--threads", "4", nullptr};
        CHECK(cmd_field_label(7, a1) == 0, "field-label command");
        FILE *f = fopen(rf, "w"); fprintf(f, "# z y x nz ny nx\n112 112 144 16 16 16\n"); fclose(f);
        char *a2[] = {"ufsm", "field-label-raster", pr, "--shape", "256,256,256", "--um", "9.6", "--regions", rf, "--cut", "0", "--agree-cut", "0", pl, nullptr};
        CHECK(cmd_field_label_raster(14, a2) == 0, "field-label-raster command");
        z3 *z = z3_open(store_open(pr), "9.6", nullptr); CHECK(z != nullptr, "raster array");
        if (z) {
            uint8_t *v = malloc(32 * 32 * 32); const int64_t oo[3] = {104, 104, 136}, nn[3] = {32, 32, 32};
            CHECK(!z3_read(z, oo, nn, v, 2), "raster read");
            /* the region (x 144 .. 160: A's cells 48 .. 63, outside the shifted quarter) agrees: every voxel known, its phase A's: z (120 + 0.5) 9.6 um from the box start P0 */
            const double ph = (120 + 0.5 - 96) * 9.6 / 60; const int want = (int)lround((ph - floor(ph)) * 252) % 252, got = v[(16 * 32 + 16) * 32 + 16];
            int known = 0, outside = 0; for (int i = 0; i < 32 * 32 * 32; i++) { const int zz = i / 1024, yy = i / 32 % 32, xx = i % 32;
                const int in = zz >= 8 && zz < 24 && yy >= 8 && yy < 24 && xx >= 8 && xx < 24; if (in) known += v[i] < 252; else outside += v[i] != 255; }
            CHECK(known == 16 * 16 * 16 && outside == 0 && abs(got - want) <= 3, "raster: %d known in the region, %d set outside, centre %d (want %d)", known, outside, got, want);
            free(v); z3_close(z);
        }
    }
    fit_fields_free(&A); fit_fields_free(&B); fit_fields_free(&S); fit_fields_free(&Dn); fit_fields_free(&W4);
    char cmd[300]; snprintf(cmd, sizeof cmd, "rm -rf '%s'", root); if (system(cmd)) {}
    if (fails) { fprintf(stderr, "test_fielddis: %d failures\n", fails); return 1; }
    printf("test_fielddis: ok\n");
    return 0;
}
