/* ufsm fit-join (src/fitjoin.c): two independent z-slab fits of one synthetic spiral field box (overlapping in z, the second with an
   extra integer gauge of +3 turns) joined into one fit set: the shift is found from the overlap alone and the joined set's W is one
   integer gauge from the true W over the whole box (winding ids >= 99%). Also fit_set_upload with more than 63 slabs: the GPU grid
   evaluation of a 70-slab set equals the CPU one. Coarse-to-fine: a whole-box fit as the global ref: fit-join --ref --chain-w 0 (the
   gauges from the ref alone) is one gauge too, and a slab fitted with --ref-model / --ref-snap 1 takes the ref's integers (>= 99%). */
#include "fit.h"
#include "nn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int cmd_fit_join(int argc, char **argv);
static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)

static const double H8 = 8, R0 = 384;
static double ax_y(double z) { return R0 + 0.02 * z; }
static double ax_x(double z) { return R0 - 0.01 * z; }
static double U_true(double z, double y, double x) { const double r = hypot(y - ax_y(z), x - ax_x(z)); return r / (60 + 4 * z / 512) + 0.3 * sin(atan2(y - ax_y(z), x - ax_x(z))) * r / 300; }
static double W_true(double z, double y, double x) { return U_true(z, y, x) - atan2(y - ax_y(z), x - ax_x(z)) / (2 * M_PI); }

int main(void) {
    nn_init(0);
    const int D = 64, H = 96, W = 96; const size_t n = (size_t)D * H * W;
    uint8_t *a[10]; for (int k = 0; k < 10; k++) a[k] = malloc(n);
    const int na = (int)(D * H8) + 1; float *cy = malloc(na * 4), *cx = malloc(na * 4);
    for (int z = 0; z < na; z++) { cy[z] = (float)ax_y(z); cx[z] = (float)ax_x(z); }
    for (int z = 0; z < D; z++) for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        const size_t i = ((size_t)z * H + y) * W + x; const double pz = (z + 0.5) * H8, py = (y + 0.5) * H8, px = (x + 0.5) * H8;
        const double r = hypot(py - ax_y(pz), px - ax_x(pz)), ph = 2 * M_PI * W_true(pz, py, px);
        a[9][i] = r >= 210 && r <= 370 ? 255 : 0;
        a[0][i] = (uint8_t)lround(128 + 127 * cos(ph)); a[1][i] = (uint8_t)lround(128 + 127 * sin(ph));
        double g[3]; const double e = 0.5;
        for (int d = 0; d < 3; d++) {
            double p[3] = {pz, py, px}, q[3] = {pz, py, px}; p[d] += e; q[d] -= e;
            double dth = atan2(p[1] - ax_y(p[0]), p[2] - ax_x(p[0])) - atan2(q[1] - ax_y(q[0]), q[2] - ax_x(q[0])); dth -= 2 * M_PI * floor((dth + M_PI) / (2 * M_PI));
            g[d] = (U_true(p[0], p[1], p[2]) - U_true(q[0], q[1], q[2]) - dth / (2 * M_PI)) / (2 * e);
        }
        const double rho = sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]), nz = g[0] / rho, ny = g[1] / rho, nx = g[2] / rho;
        const double tv[6] = {nz * nz, ny * ny, nx * nx, 0.5 + nz * ny, 0.5 + nz * nx, 0.5 + ny * nx};
        for (int k = 0; k < 6; k++) a[2 + k][i] = (uint8_t)lround(255 * tv[k]);
        const double s = 8 / (H8 * rho); a[8][i] = (uint8_t)(s < 1 ? 1 : s > 255 ? 255 : lround(s));
    }
    char dir[] = "/tmp/ufsm-fitjoin-XXXXXX"; if (!mkdtemp(dir)) { fprintf(stderr, "FAIL: tmp dir\n"); return 1; }
    const long long org[3] = {0, 0, 0}; const int shape[3] = {D, H, W};
    CHECK(!fit_fields_save(dir, org, shape, (int)H8, 2.4, a, cy, cx, na), "save fields");
    char m0[300], m1[300], out[300], cmd[1200];
    snprintf(m0, sizeof m0, "%s/m0", dir); snprintf(m1, sizeof m1, "%s/m1", dir); snprintf(out, sizeof out, "%s/set", dir);
    const int rows[2][2] = {{0, 40}, {24, 64}};   /* field rows: 16 rows of overlap */
    for (int s = 0; s < 2; s++) {
        fit_fields_buf b; CHECK(!fit_fields_load(dir, rows[s][0], rows[s][1], 1, &b), "load slab %d", s);
        fit_cfg c = fit_cfg_default(); c.verbose = 0; c.dir = 1; c.um = b.um;
        fit_model *m = fit_run(&b.f, &c); CHECK(m != nullptr, "fit slab %d", s); if (!m) return 1;
        for (int d = 0; d < 3; d++) m->origin[d] = b.origin[d];
        m->um = b.um;
        if (s == 1) for (int q = 0; q < m->nk; q++) m->offset[q] += 3;   /* an extra gauge the join must undo */
        CHECK(!fit_save(m, s ? m1 : m0), "save slab %d", s);
        fit_free(m); fit_fields_free(&b);
    }
    char *av[] = {"ufsm", "fit-join", out, dir, "--blend", "16", m0, "0", "256", m1, "256", "512"};
    CHECK(cmd_fit_join(12, av) == 0, "fit-join");
    fit_set *S = fit_set_load(out); CHECK(S && S->n == 2, "load the joined set");
    if (S) {   /* one integer gauge from the true W over the whole box */
        int hist[81] = {0}; size_t nd = 0, right = 0;
        for (int pass = 0; pass < 2; pass++) {
            int k0 = 0; for (int k = 1; k < 81; k++) if (hist[k] > hist[k0]) k0 = k;
            for (size_t i = 0; i < n; i += 3) {
                if (!a[9][i]) continue;
                const int z = (int)(i / ((size_t)H * W)), y = (int)((i / W) % H), x = (int)(i % W); const double pz = (z + 0.5) * H8, py = (y + 0.5) * H8, px = (x + 0.5) * H8;
                const double d = fit_set_W(S, pz, py, px) - W_true(pz, py, px);
                if (!pass) { const long k = lround(d); if (k >= -40 && k <= 40) hist[k + 40]++; nd++; }
                else right += fabs(d - (k0 - 40)) < 0.5;
            }
        }
        const double acc = nd ? (double)right / nd : 0;
        printf("fit-join: joined set winding ids right %.4f over %zu voxels\n", acc, nd);
        CHECK(acc >= 0.99, "joined set: winding ids right %.4f < 0.99", acc);
        /* 70 slabs (copies of slab 0's model) on the GPU = on the CPU */
        fit_set T = {0}; T.global = S->global; T.n = 70; T.blend = 4;
        T.slab = calloc(70, sizeof *T.slab); T.z0 = calloc(70, sizeof(double)); T.z1 = calloc(70, sizeof(double));
        for (int k = 0; k < 70; k++) { T.slab[k] = fit_load(m0); T.z0[k] = 512.0 * k / 70; T.z1[k] = 512.0 * (k + 1) / 70; }
        fit_set_dev *gd = fit_set_upload(&T); CHECK(gd != nullptr, "fit_set_upload of 70 slabs");
        if (gd) {
            const double o[3] = {0, 100, 100}; const int64_t g3[3] = {64, 8, 8}; float *wg = malloc(64 * 64 * 4), *wc = malloc(64 * 64 * 4);
            CHECK(!fit_set_grid(gd, &T, o, g3, 8, 0, wg) && !fit_set_grid_cpu(&T, o, g3, 8, 0, nullptr, wc), "grid evaluation");
            double md = 0; for (int i = 0; i < 64 * 64; i++) md = fmax(md, fabs((double)wg[i] - wc[i]));
            printf("fit-join: 70-slab set, GPU vs CPU grid max |dW| %.2e\n", md);
            CHECK(md < 1e-3, "70-slab grid GPU vs CPU differ by %.3g", md);
            fit_set_dev_free(gd); free(wg); free(wc);
        }
        for (int k = 0; k < 70; k++) fit_free(T.slab[k]);
        free(T.slab); free(T.z0); free(T.z1);
        fit_set_free(S);
    }
    {   /* coarse-to-fine: the whole box fitted as the ref */
        char ref[300], out2[300]; snprintf(ref, sizeof ref, "%s/ref", dir); snprintf(out2, sizeof out2, "%s/set2", dir);
        fit_fields_skip_planes(0); fit_fields_ct_planes(3);   /* (fit-join left its pc ps w-only plane selection set in this process) */
        fit_fields_buf b; CHECK(!fit_fields_load(dir, 0, 0, 1, &b), "load the box");
        fit_cfg c = fit_cfg_default(); c.verbose = 0; c.dir = 1; c.um = b.um;
        fit_model *m = fit_run(&b.f, &c); CHECK(m != nullptr, "fit the ref"); if (!m) return 1;
        for (int d = 0; d < 3; d++) m->origin[d] = b.origin[d];
        m->um = b.um; for (int q = 0; q < m->nk; q++) m->offset[q] -= 5;   /* (any gauge) */
        CHECK(!fit_save(m, ref), "save the ref"); fit_free(m); fit_fields_free(&b);
        char *av2[] = {"ufsm", "fit-join", out2, dir, "--ref", ref, "--chain-w", "0", m0, "0", "256", m1, "256", "512"};
        CHECK(cmd_fit_join(14, av2) == 0, "fit-join --ref");
        fit_set *S2 = fit_set_load(out2), *R = fit_set_load(ref); CHECK(S2 && R, "load the ref join / the ref");
        if (S2 && R) {   /* the joined set vs the ref: one integer everywhere */
            long hist[81] = {0}; size_t nd = 0;
            for (size_t i = 0; i < n; i += 3) {
                if (!a[9][i]) continue;
                const int z = (int)(i / ((size_t)H * W)), y = (int)((i / W) % H), x = (int)(i % W); const double pz = (z + 0.5) * H8, py = (y + 0.5) * H8, px = (x + 0.5) * H8;
                const long k = lround(fit_set_W(S2, pz, py, px) - fit_set_W(R, pz, py, px)); if (k >= -40 && k <= 40) hist[k + 40]++; nd++;
            }
            long best = 0; for (int k = 0; k < 81; k++) if (hist[k] > best) best = hist[k];
            printf("fit-join --ref --chain-w 0: share of one integer vs the ref %.4f\n", nd ? (double)best / nd : 0);
            CHECK(nd && (double)best / nd >= 0.99, "fit-join --ref: one integer on %.4f < 0.99", nd ? (double)best / nd : 0);
            /* a slab fitted with its integers snapped to the ref */
            fit_fields_skip_planes(0); fit_fields_ct_planes(3);
            fit_fields_buf sb; CHECK(!fit_fields_load(dir, rows[1][0], rows[1][1], 1, &sb), "load slab 1");
            fit_cfg cs = fit_cfg_default(); cs.verbose = 0; cs.dir = 1; cs.um = sb.um; cs.ref = R; cs.ref_snap = 1;
            for (int d = 0; d < 3; d++) cs.ref_off[d] = (double)(sb.origin[d] - R->global->origin[d]);
            fit_model *ms = fit_run(&sb.f, &cs); CHECK(ms != nullptr, "fit --ref-snap");
            if (ms) {
                size_t ns = 0, ok = 0;
                for (size_t i = 0; i < (size_t)(rows[1][1] - rows[1][0]) * H * W; i += 3) {
                    if (!sb.f.w[i]) continue;
                    const int z = (int)(i / ((size_t)H * W)), y = (int)((i / W) % H), x = (int)(i % W); const double pz = (z + 0.5) * H8, py = (y + 0.5) * H8, px = (x + 0.5) * H8;
                    ok += lround(fit_W(ms, pz, py, px) - fit_set_W(R, pz + cs.ref_off[0], py, px)) == 0; ns++;
                }
                printf("fit --ref-snap 1: the ref's integer on %.4f of the slab's voxels\n", ns ? (double)ok / ns : 0);
                CHECK(ns && (double)ok / ns >= 0.99, "fit --ref-snap: the ref's integer on %.4f < 0.99", ns ? (double)ok / ns : 0);
                fit_free(ms);
            }
            fit_fields_free(&sb);
        }
        fit_set_free(S2); fit_set_free(R);
    }
    snprintf(cmd, sizeof cmd, "rm -rf '%s'", dir); if (system(cmd)) fprintf(stderr, "(could not remove %s)\n", dir);
    if (fails) { fprintf(stderr, "test_fit_join: %d failure(s)\n", fails); return 1; }
    printf("test_fit_join: OK\n");
    return 0;
}
