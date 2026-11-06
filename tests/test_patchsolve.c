/* ufsm patch-solve / sheet-trace / patch-score on a synthetic box: 6 concentric sheets r = 18 + 8 k (k = 0..5) around a straight axis at (y, x) = (64, 64), in a
   48 x 128 x 128 prob box (Gaussian sheet profile, peak 250). Traced patches: two overlapping arcs per sheet for k = 0, 1, 3, 4, 5 (sheet
   2 is in the prob but has no patch, so the edges 1 -> 3 must count it from the ridge / prob between them), plus a short invalid (-1)
   border on every patch. GT codes: 1 + round(18 (k + theta / 2 pi)) mod 252 (spiral sense +1).
   Expect: every node in one graph comp, turn(b) - turn(a) = k_b - k_a for all node pairs, the GT score 1.0 (both vote modes), and
   the oracle diagnostic changing no pair-mode edge. */
#include "tiffw.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

int cmd_patch_solve(int argc, char **argv);
int cmd_sheet_trace(int argc, char **argv);
int cmd_patch_score(int argc, char **argv);
static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)
enum { NZ = 48, NY = 128, NX = 128 };
static double rk(int k) { return 18 + 8 * k; }

static int run(const char *root, const char *votes, const char *extra, double *score, int *oracle_changed) {
    char paths[600], prob[600], gt[600], ax[600], tsv[600], out[600];
    snprintf(paths, sizeof paths, "%s/paths", root); snprintf(prob, sizeof prob, "%s/prob.raw", root); snprintf(gt, sizeof gt, "%s/gt.raw", root);
    snprintf(ax, sizeof ax, "%s/axis.json", root); snprintf(tsv, sizeof tsv, "%s/nodes.tsv", root); snprintf(out, sizeof out, "%s/out.txt", root);
    char box[64]; snprintf(box, sizeof box, "0,0,0,%d,%d,%d", NZ, NY, NX);
    char *av[32] = {"ufsm", "patch-solve", "--paths", paths, "--prob", prob, "--gt", gt, "--box", box, "--axis", ax, "--sgn", "1",
                    "--votes", (char *)votes, "--threads", "3", "--tsv", tsv};
    int ac = 20; if (extra) { av[ac++] = (char *)extra; av[ac++] = "1"; }
    fflush(stdout); FILE *sv = fopen(out, "w"); if (!sv) return -1;
    const int fd = dup(fileno(stdout)); dup2(fileno(sv), fileno(stdout));
    const int rc = cmd_patch_solve(ac, av);
    fflush(stdout); dup2(fd, fileno(stdout)); close(fd); fclose(sv);
    FILE *f = fopen(out, "r"); char line[1024]; *score = -1; *oracle_changed = -1;
    while (f && fgets(line, sizeof line, f)) {
        double s; int c;
        if (sscanf(line, "sign +1: area on right sheet+turn %lf", &s) == 1) *score = s;
        if (sscanf(line, "oracle: %d edges changed", &c) == 1) *oracle_changed = c;
    }
    if (f) fclose(f);
    return rc;
}

int main(void) {
    char root[] = "/tmp/test_patchsolve_XXXXXX"; if (!mkdtemp(root)) return 1;
    char p[700];
    /* prob + GT */
    uint8_t *pb = calloc((size_t)NZ * NY * NX, 1), *g = calloc((size_t)NZ * NY * NX, 1);
    for (int z = 0; z < NZ; z++) for (int y = 0; y < NY; y++) for (int x = 0; x < NX; x++) {
        const double r = hypot(y - 64.0, x - 64.0), th = atan2(y - 64.0, x - 64.0); double best = 0; int kb = -1;
        for (int k = 0; k < 6; k++) { const double d = r - rk(k), v = 250 * exp(-d * d / (2 * 1.2 * 1.2)); if (v > best) { best = v; kb = k; } }
        if (fabs(th) > 2.4) best = 0;   /* arcs, not full rings: the ring GT (unlike a spiral's) jumps by a turn at theta = pi */
        const size_t i = ((size_t)z * NY + y) * NX + x; pb[i] = (uint8_t)lround(best);
        if (kb >= 0 && fabs(th) <= 2.4 && fabs(r - rk(kb)) < 0.75) g[i] = (uint8_t)(1 + ((lround(18 * (kb + th / (2 * M_PI))) % 252) + 252) % 252);
    }
    snprintf(p, sizeof p, "%s/prob.raw", root); FILE *f = fopen(p, "wb"); fwrite(pb, 1, (size_t)NZ * NY * NX, f); fclose(f);
    snprintf(p, sizeof p, "%s/gt.raw", root); f = fopen(p, "wb"); fwrite(g, 1, (size_t)NZ * NY * NX, f); fclose(f);
    free(pb); free(g);
    snprintf(p, sizeof p, "%s/axis.json", root); f = fopen(p, "w");
    fprintf(f, "{\"control_points\": [{\"x\": 64, \"y\": 64, \"z\": 0}, {\"x\": 64, \"y\": 64, \"z\": %d}]}\n", NZ - 1); fclose(f);
    /* patches: per traced sheet two arcs theta in [-0.9, 0.3] and [-0.3, 0.9], unit arc spacing, z 2..45, one invalid border column */
    snprintf(p, sizeof p, "%s/paths", root); mkdir(p, 0755);
    const int traced[5] = {0, 1, 3, 4, 5};
    for (int ti = 0; ti < 5; ti++) for (int half = 0; half < 2; half++) {
        const int k = traced[ti]; const double r = rk(k), t0 = half ? -0.3 : -0.9; const int W = (int)(1.2 * r) + 2, H = 44;
        float *X = malloc((size_t)W * H * 4), *Y = malloc((size_t)W * H * 4), *Z = malloc((size_t)W * H * 4);
        for (int j = 0; j < H; j++) for (int i = 0; i < W; i++) {
            const size_t q = (size_t)j * W + i; const double th = t0 + (i - 1) / r;
            if (i == 0) { X[q] = Y[q] = Z[q] = -1; continue; }
            X[q] = (float)(64 + r * cos(th)); Y[q] = (float)(64 + r * sin(th)); Z[q] = (float)(2 + j);
        }
        snprintf(p, sizeof p, "%s/paths/s%d_%d", root, k, half);
        if (tifxyz_write(p, W, H, X, Y, Z, 1, 9.6, -1, "test")) { fprintf(stderr, "tifxyz_write failed\n"); return 1; }
        free(X); free(Y); free(Z);
    }
    for (int m = 0; m < 2; m++) {
        const char *votes = m ? "ray" : "pair"; double sc; int oc;
        CHECK(run(root, votes, nullptr, &sc, &oc) == 0, "%s: patch-solve failed", votes);
        CHECK(fabs(sc - 1) < 1e-9, "%s: GT score %.4f, want 1", votes, sc);
        /* the node table: one graph comp, turns = sheet index + a constant */
        snprintf(p, sizeof p, "%s/nodes.tsv", root); f = fopen(p, "r"); char line[1024]; int nnode = 0, gc0 = -1, off = 1 << 30, bad = 0, gcs = 0;
        if (f && !fgets(line, sizeof line, f)) { fclose(f); f = nullptr; }
        while (f && fgets(line, sizeof line, f)) {
            char name[256]; int comp, act, gc, turn, gs, gp, gn; size_t nv;
            if (sscanf(line, "%255s %d %zu %d %d %d %d %d %d", name, &comp, &nv, &act, &gc, &turn, &gs, &gp, &gn) != 9) continue;
            int k; if (sscanf(name, "s%d_", &k) != 1) continue;
            nnode++;
            if (gc0 < 0) gc0 = gc;
            gcs += gc != gc0;
            if (off == 1 << 30) off = turn - k;
            bad += turn - k != off;
            CHECK(act == 1, "%s: node %s pruned", votes, name);
        }
        if (f) fclose(f);
        CHECK(nnode == 10, "%s: %d nodes, want 10", votes, nnode);
        CHECK(gcs == 0, "%s: %d nodes outside the first graph comp", votes, gcs);
        CHECK(bad == 0, "%s: %d nodes off the sheet index (sheet 2 untraced must be counted)", votes, bad);
        /* pair mode: every pair's mode equals the GT difference; ray mode keeps minority votes (a ray whose ridge misses sheet 2) as edges */
        CHECK(run(root, votes, "--oracle", &sc, &oc) == 0 && (m ? oc <= 2 : oc == 0), "%s: oracle changed %d edges", votes, oc);
    }
    char cmd[800];
    /* sheet-trace on the same prob (every sheet, incl. the untraced 2, is a peak ridge): 6 pure patches; tiled (--tile 8) the trace ->
       patch-solve (--cut 0 --links) chain scores 1 on the GT; patch-score recall > 0.9, purity 1 */
    {   char out[700], prob[600], gt[600], ax[600], box[64], st[700], links[800], txt[700];
        snprintf(out, sizeof out, "%s/st", root); snprintf(prob, sizeof prob, "%s/prob.raw", root); snprintf(gt, sizeof gt, "%s/gt.raw", root);
        snprintf(ax, sizeof ax, "%s/axis.json", root); snprintf(box, sizeof box, "0,0,0,%d,%d,%d", NZ, NY, NX);
        snprintf(st, sizeof st, "%s/st/paths", root); snprintf(links, sizeof links, "%s/st/links.tsv", root); snprintf(txt, sizeof txt, "%s/st.txt", root);
        for (int tiled = 0; tiled < 2; tiled++) {
            char *tv[] = {"ufsm", "sheet-trace", "--prob", prob, "--box", box, "--axis", ax, "--out", out, "--threads", "3", "--min-cells", "50", "--tile", tiled ? "8" : "0"};
            snprintf(cmd, sizeof cmd, "rm -rf %s", out); if (system(cmd)) {}
            fflush(stdout); FILE *sv = fopen(txt, "w"); const int fd = dup(fileno(stdout)); dup2(fileno(sv), fileno(stdout));
            int rc = cmd_sheet_trace((int)(sizeof tv / sizeof *tv), tv);
            char *sv2[] = {"ufsm", "patch-score", "--paths", st, "--gt", gt, "--box", box, "--axis", ax, "--sgn", "1"};
            rc |= cmd_patch_score((int)(sizeof sv2 / sizeof *sv2), sv2);
            char *pv[] = {"ufsm", "patch-solve", "--paths", st, "--prob", prob, "--gt", gt, "--box", box, "--axis", ax, "--sgn", "1", "--cut", "0", "--minv", "16", "--links", links, "--threads", "3"};
            rc |= cmd_patch_solve((int)(sizeof pv / sizeof *pv), pv);
            fflush(stdout); dup2(fd, fileno(stdout)); close(fd); fclose(sv);
            CHECK(rc == 0, "sheet-trace chain (tile %d) failed", tiled);
            FILE *f2 = fopen(txt, "r"); char line[1024]; int kept = -1, np = -1; double rec = -1, pur = -1, sc = -1;
            while (f2 && fgets(line, sizeof line, f2)) {
                const char *k = strstr(line, "patches grown, "); if (k) sscanf(k, "patches grown, %d kept", &kept);
                if (sscanf(line, "patches %d | recall %lf", &np, &rec) == 2) { const char *u = strstr(line, "matched-weighted) "); if (u) pur = atof(u + 18); }
                double v; if (sscanf(line, "sign +1: area on right sheet+turn %lf", &v) == 1) sc = v;
            }
            if (f2) fclose(f2);
            if (!tiled) CHECK(kept == 6, "sheet-trace kept %d patches, want 6 (one per sheet)", kept);
            CHECK(rec > 0.9 && pur > 0.999, "sheet-trace (tile %d): recall %.3f purity %.4f", tiled, rec, pur);
            CHECK(fabs(sc - 1) < 1e-9, "sheet-trace (tile %d) -> patch-solve: GT score %.4f, want 1", tiled, sc);
        }
    }
    snprintf(cmd, sizeof cmd, "rm -rf %s", root);
    if (getenv("KEEP")) fprintf(stderr, "kept %s\n", root); else if (system(cmd)) {}
    if (fails) { fprintf(stderr, "test_patchsolve: %d failures\n", fails); return 1; }
    printf("test_patchsolve: ok\n");
    return 0;
}
