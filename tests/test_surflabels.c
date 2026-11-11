/* ufsm surf-labels on synthetic spiral wraps: an Archimedean spiral r = 20 + 10 W around a vertical axis (W = U - dir theta / 2 pi, wrap
   k = the level set W = k, so wrap k at angle theta sits at r_k(theta) = 20 + 10 (k + dir theta / 2 pi)), wraps k = 1..7 written as .sfc
   (rows = z, columns = theta in (-pi, pi]), dir = +1 and -1. Checks, on the 9.6 um grid:
   - codes: every coded voxel's q = k + dir theta / 2 pi (mod 14) within 1/18 turn of the analytic sheet coordinate of the nearest wrap;
   - phase: between kept consecutive wraps it equals the analytic fraction (r - r_k) / (r_k+1 - r_k) (mean error < 0.03 turn) and rises
     outward (the sign), for both spiral senses (dir);
   - identical to band_field_phase_pair on the written codes (PHASE_* convention), when nothing is dropped;
   - a dropped wrap (--drop w4): no code on it, ignore = 1 and phase unknown between wraps 3 and 5, labels elsewhere kept;
   - a conflict: a wrap 8 lying on wrap 2 -> codes 255 there, written 0, ignore 2, never labelled;
   - --gt with the dir +1 codes as GT: phase agreement and pair accuracy 1.0 against itself;
   - a holdout box (--exclude) -> ignore 4;
   - --geom: defect cells / slipping z bins of geom-check -> no codes there;
   - --min-coh: CT ridges on the sheets keep the labels, ridges midway between them (labels half a turn off) -> ignore 5; --weight-coh: 0 off the kept labels;
   - --anchor A,A (union) = --anchor A;
   - wrap-anchor: wraps miscounted (+10 turns, a false extra turn after wrap 3) against the true codes as GT, votes in alternate z
     blocks: every anchored code equals the GT code, the held-out blocks included;
   - the same wraps joined into one strip.sfc (+ strip.tsv column ranges) give the same bytes as the wrap files;
   - same bytes with 1 and 4 threads; --dir auto picks -1 on the dir -1 spiral (same bytes as --dir -1). */
#include "band.h"
#include "store.h"
#include "tiffw.h"
#include "z3w.h"
#include "zarr3.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int cmd_surf_labels(int argc, char **argv);
int cmd_phase_coh(int argc, char **argv);
static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { fprintf(stderr, "FAIL: %s\n", m); fails++; } } while (0)
#define NZ 64
#define NY 256
#define NX 256
#define C0 128.0
static double rk(double W) { return 20 + 10 * W; }
static uint8_t *readall(const char *root, const char *key) {
    z3 *z = z3_open(store_open(root), key, nullptr); if (!z) return nullptr;
    const int64_t o[3] = {0, 0, 0}, n[3] = {NZ, NY, NX}; uint8_t *b = malloc((size_t)NZ * NY * NX);
    if (z3_read(z, o, n, b, 1)) { free(b); b = nullptr; } z3_close(z); return b;
}
static void wraps_ids(const char *dir, const int *ids) {   /* dir +1 spiral wraps k = 1..7 written under other ids (ids[k]): a wrong count */
    const int H = NZ / 2, Wc = 720;
    for (int k = 1; k <= 7; k++) {
        float *x = malloc(Wc * H * 4), *y = malloc(Wc * H * 4), *z = malloc(Wc * H * 4);
        for (int r = 0; r < H; r++) for (int c = 0; c < Wc; c++) {
            const double th = M_PI - (c + 0.5) * 2 * M_PI / Wc, rr = rk(k + th / (2 * M_PI));
            x[r * Wc + c] = (float)(C0 + rr * cos(th)); y[r * Wc + c] = (float)(C0 + rr * sin(th)); z[r * Wc + c] = (float)(2 * r);
        }
        char p[600]; snprintf(p, sizeof p, "%s/w%03d", dir, ids[k]);
        tifxyz_write(p, Wc, H, x, y, z, 2, 9.6, ids[k], "test");
        free(x); free(y); free(z);
    }
}
static void wraps(const char *dir, int sense, int dup) {
    const int H = NZ / 2, Wc = 720;
    for (int k = 1; k <= 7 + (dup ? 1 : 0); k++) {
        float *x = malloc(Wc * H * 4), *y = malloc(Wc * H * 4), *z = malloc(Wc * H * 4);
        const double kk = k == 8 ? 2.0 : k;   /* (dup: wrap "8" lies on wrap 2: codes 6 turns apart in the same voxels) */
        for (int r = 0; r < H; r++) for (int c = 0; c < Wc; c++) {
            const double th = M_PI - (c + 0.5) * 2 * M_PI / Wc, rr = rk(kk + sense * th / (2 * M_PI));
            x[r * Wc + c] = (float)(C0 + rr * cos(th)); y[r * Wc + c] = (float)(C0 + rr * sin(th)); z[r * Wc + c] = (float)(2 * r);
        }
        char p[600]; snprintf(p, sizeof p, "%s/w%03d", dir, k);
        tifxyz_write(p, Wc, H, x, y, z, 2, 9.6, k, "test");
        free(x); free(y); free(z);
    }
}
static void strip_of(const char *dir, int sense) {   /* the dir +1 / -1 wraps 1..7 joined side by side as ONE strip.sfc (ufsm strip's layout) + strip.tsv */
    const int H = NZ / 2, Wc = 720, W = 7 * Wc;
    float *x = malloc((size_t)W * H * 4), *y = malloc((size_t)W * H * 4), *z = malloc((size_t)W * H * 4);
    for (int k = 1; k <= 7; k++) for (int r = 0; r < H; r++) for (int c = 0; c < Wc; c++) {
        const size_t i = (size_t)r * W + (size_t)(k - 1) * Wc + c;
        const double th = M_PI - (c + 0.5) * 2 * M_PI / Wc, rr = rk(k + sense * th / (2 * M_PI));
        x[i] = (float)(C0 + rr * cos(th)); y[i] = (float)(C0 + rr * sin(th)); z[i] = (float)(2 * r);
    }
    char p[600]; snprintf(p, sizeof p, "%s/strip", dir); tifxyz_write(p, W, H, x, y, z, 2, 9.6, 0, "test");
    free(x); free(y); free(z);
    snprintf(p, sizeof p, "%s/strip.tsv", dir); FILE *f = fopen(p, "w");
    fprintf(f, "seg\twrap\tname\tcol0\tncols\twrap_col0\tseam_gap_pitch\n");
    for (int k = 7; k >= 1; k--) fprintf(f, "%d\t%d\tw%03d\t%d\t%d\t0\t0\n", 7 - k, k, k, (k - 1) * Wc, Wc);   /* (outermost first, as ufsm strip writes) */
    fclose(f);
}
static int run(const char *root, const char *wd, const char *out, const char *dir, const char *threads, const char *extra1, const char *extra2, const char *gt) {
    char ax[400], wdp[400], o[400], gtr[400]; snprintf(ax, sizeof ax, "%s/axis.json", root); snprintf(wdp, sizeof wdp, "%s/%s", root, wd); snprintf(o, sizeof o, "%s/%s", root, out);
    char *av[40]; int n = 0;
    av[n++] = "ufsm"; av[n++] = "surf-labels"; av[n++] = wdp; av[n++] = o; av[n++] = "--axis"; av[n++] = ax; av[n++] = "--axis-um"; av[n++] = "9.6";
    av[n++] = "--dir"; av[n++] = (char *)dir; av[n++] = "--grid-um"; av[n++] = "9.6"; av[n++] = "--shape"; av[n++] = "64,256,256";
    av[n++] = "--shard"; av[n++] = "128"; av[n++] = "--block"; av[n++] = "32"; av[n++] = "--radius"; av[n++] = "12"; av[n++] = "--threads"; av[n++] = (char *)threads;
    if (extra1) { av[n++] = (char *)extra1; av[n++] = (char *)extra2; }
    if (gt) { snprintf(gtr, sizeof gtr, "%s/%s", root, gt); av[n++] = "--gt"; av[n++] = gtr; av[n++] = "9.6"; }
    if (extra1 && !strcmp(extra1, "--exclude")) { av[n++] = "--exclude-ct"; av[n++] = "scan.zarr"; av[n++] = "--exclude-mm"; av[n++] = "0"; }
    return cmd_surf_labels(n, av);
}
int main(void) {
    char root[256]; snprintf(root, sizeof root, "%s/test_surflabels_%d", getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp", (int)getpid());
    char cmd[1400], p[600]; snprintf(cmd, sizeof cmd, "rm -rf %s && mkdir -p %s/wp %s/wm %s/wd %s/wx", root, root, root, root, root); if (system(cmd)) return 1;
    snprintf(p, sizeof p, "%s/axis.json", root); FILE *f = fopen(p, "w"); if (!f) return 1;
    fprintf(f, "{\"control_points\": [{\"x\": %g, \"y\": %g, \"z\": 0}, {\"x\": %g, \"y\": %g, \"z\": %d}]}\n", C0, C0, C0, C0, NZ - 1); fclose(f);
    snprintf(p, sizeof p, "%s/hold.json", root); f = fopen(p, "w"); if (!f) return 1;
    fprintf(f, "{\"boxes\": [{\"name\": \"h\", \"ct\": \"S/scan.zarr\", \"l0\": [10, 160, 160, 20, 30, 30]}, {\"name\": \"o\", \"ct\": \"T/other.zarr\", \"l0\": [0, 0, 0, 64, 64, 64]}]}\n"); fclose(f);
    for (int v = 0; v < 2; v++) {   /* CT: gaussian ridges on the dir +1 sheets (ct0) or midway between them (ct1) */
        const int64_t sh[3] = {NZ, NY, NX}; snprintf(p, sizeof p, "%s/ct%d/0", root, v); z3w *w = z3w_create(p, sh, 128, 0, 0, nullptr); if (!w) return 1;
        uint8_t *d = malloc(128 * 128 * 128);
        for (int sy = 0; sy < 2; sy++) for (int sx = 0; sx < 2; sx++) {
            for (int z = 0; z < NZ; z++) for (int y = 0; y < 128; y++) for (int x = 0; x < 128; x++) {
                const double Y = sy * 128 + y - C0, X = sx * 128 + x - C0, Wv = (hypot(X, Y) - 20) / 10 - atan2(Y, X) / (2 * M_PI) - 0.5 * v;
                const double e = (Wv - round(Wv)) * 10; d[((size_t)z * 128 + y) * 128 + x] = (uint8_t)lround(20 + 200 * exp(-e * e / 4.5));
            }
            for (int z = NZ; z < 128; z++) memset(d + (size_t)z * 128 * 128, 0, 128 * 128);
            if (z3w_write_shard(w, 0, sy, sx, d, 2)) return 1;
        }
        free(d); if (z3w_close(w)) return 1;
    }
    tifxyz_set_sfc(0.01);
    snprintf(p, sizeof p, "%s/wp", root); wraps(p, 1, 0);
    snprintf(p, sizeof p, "%s/wm", root); wraps(p, -1, 0);
    snprintf(p, sizeof p, "%s/wd", root); wraps(p, 1, 1);
    { const int ids[8] = {0, 11, 12, 13, 15, 16, 17, 18}; snprintf(p, sizeof p, "%s/wx", root); wraps_ids(p, ids); }   /* +10 turns, then a false extra turn between wraps 3 and 4 */
    snprintf(p, sizeof p, "%s/ws", root); snprintf(cmd, sizeof cmd, "mkdir -p %s", p); if (system(cmd)) return 1; strip_of(p, 1);
    tifxyz_set_sfc(0);
    CHECK(!run(root, "wp", "p1", "1", "1", nullptr, nullptr, nullptr), "surf-labels dir +1 (1 thread) failed");
    CHECK(!run(root, "wp", "p4", "1", "4", nullptr, nullptr, nullptr), "surf-labels dir +1 (4 threads) failed");
    snprintf(cmd, sizeof cmd, "cd %s && for a in codes phase ignore; do for s in $(cd p1/$a.zarr/9.6 && find c -type f); do cmp -s p1/$a.zarr/9.6/$s p4/$a.zarr/9.6/$s || exit 1; done; done", root);
    CHECK(!system(cmd), "outputs differ between 1 and 4 threads");
    /* --zband: a band holding the whole grid writes the same bytes; a band past it writes no shard */
    CHECK(!run(root, "wp", "zb", "1", "4", "--zband", "0,64", nullptr), "surf-labels --zband 0,64 failed");
    snprintf(cmd, sizeof cmd, "cd %s && for a in codes phase ignore; do for s in $(cd p4/$a.zarr/9.6 && find c -type f); do cmp -s p4/$a.zarr/9.6/$s zb/$a.zarr/9.6/$s || exit 1; done; done", root);
    CHECK(!system(cmd), "--zband over the whole grid changed the outputs");
    CHECK(!run(root, "wp", "zc", "1", "4", "--zband", "128,256", nullptr), "surf-labels --zband 128,256 failed");
    snprintf(cmd, sizeof cmd, "test -z \"$(find %s/zc/phase.zarr/9.6 -path '*/c/*' -type f 2>/dev/null)\"", root);
    CHECK(!system(cmd), "--zband past the grid still wrote phase shards");
    CHECK(!run(root, "ws/strip.sfc", "st", "1", "4", nullptr, nullptr, nullptr), "surf-labels on a strip.sfc failed");   /* one strip = its wraps (strip.tsv columns) */
    {   /* (the wraps and the strip are separate lossy .sfc encodings, tolerance 0.01 voxel: a few voxels at the sheet edges may differ) */
        static const char *arr[3] = {"codes", "phase", "ignore"}; size_t nd[3] = {0, 0, 0}, nk = 0, big = 0;
        for (int a = 0; a < 3; a++) {
            char ra[700], rb[700]; snprintf(ra, sizeof ra, "%s/p4/%s.zarr", root, arr[a]); snprintf(rb, sizeof rb, "%s/st/%s.zarr", root, arr[a]);
            uint8_t *A = readall(ra, "9.6"), *B = readall(rb, "9.6"); if (!A || !B) { CHECK(0, "cannot read the strip / wrap labels"); free(A); free(B); continue; }
            for (size_t k = 0; k < (size_t)NZ * NY * NX; k++) {
                nd[a] += A[k] != B[k];
                if (a == 1 && A[k] < 252) { nk++; if (B[k] < 252) { int d = abs((int)A[k] - (int)B[k]); if (d > 126) d = 252 - d; big += d > 3; } else big++; }
            }
            free(A); free(B);
        }
        printf("test_surflabels: strip.sfc vs wrap files: codes %zu, phase %zu, ignore %zu voxels differ; %zu of %zu known phase voxels off by > 3 codes or lost\n", nd[0], nd[1], nd[2], big, nk);
        CHECK(nk > 100000 && big * 1000 <= nk && nd[0] * 200 <= nk && nd[2] * 1000 <= (size_t)NZ * NY * NX, "labels from the strip.sfc differ from the same wraps as files");
    }
    CHECK(!run(root, "wm", "m", "-1", "4", nullptr, nullptr, nullptr), "surf-labels dir -1 failed");
    CHECK(!run(root, "wm", "ma", "auto", "2", nullptr, nullptr, nullptr), "surf-labels --dir auto (-1) failed");
    snprintf(cmd, sizeof cmd, "cd %s && for a in codes phase ignore; do for s in $(cd m/$a.zarr/9.6 && find c -type f); do cmp -s m/$a.zarr/9.6/$s ma/$a.zarr/9.6/$s || exit 1; done; done", root);
    CHECK(!system(cmd), "--dir auto did not pick -1 on the dir -1 spiral");
    CHECK(!run(root, "wp", "d", "1", "4", "--drop", "w4", nullptr), "surf-labels --drop failed");
    CHECK(!run(root, "wd", "c", "1", "4", nullptr, nullptr, nullptr), "surf-labels dup failed");
    CHECK(!run(root, "wp", "g", "1", "4", nullptr, nullptr, "p1/codes.zarr"), "surf-labels --gt failed");
    { char ct[400]; for (int v = 0; v < 2; v++) {
        snprintf(ct, sizeof ct, "%s/ct%d", root, v); char o[32]; snprintf(o, sizeof o, "coh%d", v);
        char ax[400], wdp[400], od[400]; snprintf(ax, sizeof ax, "%s/axis.json", root); snprintf(wdp, sizeof wdp, "%s/wp", root); snprintf(od, sizeof od, "%s/%s", root, o);
        char *av[] = {"ufsm", "surf-labels", wdp, od, "--axis", ax, "--axis-um", "9.6", "--dir", "1", "--grid-um", "9.6", "--shape", "64,256,256", "--shard", "128", "--block", "32",
                      "--radius", "12", "--threads", "4", "--ct", ct, "0", "--coh-cell", "16", "--min-coh", "0.3", "--weight-coh", "0.3,0.6"};
        CHECK(!cmd_surf_labels((int)(sizeof av / sizeof *av), av), "surf-labels --min-coh failed");
        char r[600]; snprintf(r, sizeof r, "%s/ignore.zarr", od); uint8_t *ig = readall(r, "9.6"); snprintf(r, sizeof r, "%s/phase.zarr", od); uint8_t *ph = readall(r, "9.6");
        long kept = 0, coh = 0; for (size_t i = 0; ig && ph && i < (size_t)NZ * NY * NX; i++) { kept += ph[i] != PHASE_UNKNOWN; coh += ig[i] == 5; }
        printf("test_surflabels: CT ridges %s: %ld kept, %ld coherence-gated\n", v ? "midway (wrong labels)" : "on the sheets", kept, coh);
        if (v == 0) CHECK(kept > 10 * coh && kept > 100000, "labels on CT-coherent sheets were gated");
        else CHECK(coh > 10 * kept && coh > 100000, "labels half a turn off the CT ridges were kept");
        snprintf(r, sizeof r, "%s/weight.zarr", od); uint8_t *wt = readall(r, "9.6"); long wbad = 0, wpos = 0;   /* --weight-coh: 0 wherever ignored / unknown, > 0 on kept coherent labels */
        for (size_t i = 0; wt && ph && i < (size_t)NZ * NY * NX; i++) { wbad += wt[i] && ph[i] == PHASE_UNKNOWN; wpos += wt[i] > 0; }
        CHECK(wt && !wbad && (v == 1 || wpos > kept / 2), "--weight-coh: weight on unknown phase or missing on kept labels");
        if (v == 0) {   /* ufsm phase-coh on the stored phase: surf-labels' weight within rounding (cells with ignored voxels may differ) */
            char pr[600], pc[600]; snprintf(pr, sizeof pr, "%s/phase.zarr", od); snprintf(pc, sizeof pc, "%s/pc", od);
            char *cv[] = {"ufsm", "phase-coh", pr, "9.6", ct, "0", pc, "--weight", "0.3,0.6", "--weight2", "0.3,0.6", "--cell", "16", "--threads", "3"};
            CHECK(!cmd_phase_coh((int)(sizeof cv / sizeof *cv), cv), "phase-coh failed");
            snprintf(r, sizeof r, "%s/pc/weight.zarr", od); uint8_t *w2 = readall(r, "9.6"); long nk = 0, nd = 0;
            for (size_t i = 0; w2 && wt && ph && i < (size_t)NZ * NY * NX; i++) if (ph[i] != PHASE_UNKNOWN) { nk++; nd += abs((int)w2[i] - (int)wt[i]) > 1; }
            printf("test_surflabels: phase-coh vs surf-labels --weight-coh: %ld of %ld known voxels differ by > 1\n", nd, nk);
            CHECK(w2 && nk > 100000 && nd * 20 < nk, "phase-coh weight differs from surf-labels' on > 5% of the known voxels");
            {   /* --box-scale: one box over the lower half in z at 0.5, 0 outside (--box-default 0) = the plain weight x 0.5 inside, 0 outside */
                char bf[600], pb[600]; snprintf(bf, sizeof bf, "%s/boxes.txt", od); snprintf(pb, sizeof pb, "%s/pcb", od);
                FILE *f = fopen(bf, "w"); CHECK(f, "boxes.txt"); fprintf(f, "# z y x nz ny nx s\n0 0 0 %d %d %d 0.5\n", NZ / 2, NY, NX); fclose(f);
                char *cb[] = {"ufsm", "phase-coh", pr, "9.6", ct, "0", pb, "--weight", "0.3,0.6", "--cell", "16", "--box-scale", bf, "--box-default", "0", "--threads", "3"};
                CHECK(!cmd_phase_coh((int)(sizeof cb / sizeof *cb), cb), "phase-coh --box-scale failed");
                snprintf(r, sizeof r, "%s/pcb/weight.zarr", od); uint8_t *wb = readall(r, "9.6"); long bad = 0, pos = 0;
                for (size_t i = 0; wb && w2 && i < (size_t)NZ * NY * NX; i++) {
                    const int want = i / ((size_t)NY * NX) < (size_t)(NZ / 2) ? (int)lround(w2[i] * 0.5) : 0; bad += abs((int)wb[i] - want) > 1; pos += wb[i] > 0; }
                printf("test_surflabels: phase-coh --box-scale: %ld voxels off the scaled weight, %ld weighted\n", bad, pos);
                CHECK(wb && !bad && pos > 1000, "phase-coh --box-scale is not the weight x the box factor");
                free(wb);
            }
            {   /* --papyrus T --pap-dilate 1 --phase-out 1: against a brute-force mask (CT < T with no CT >= T in the 3^3 cube about it) */
                char pp[600]; snprintf(pp, sizeof pp, "%s/pcp", od); const int T = 140;
                char *cp[] = {"ufsm", "phase-coh", pr, "9.6", ct, "0", pp, "--weight", "0.3,0.6", "--cell", "16", "--papyrus", "140", "--pap-dilate", "1", "--phase-out", "1", "--threads", "3"};
                CHECK(!cmd_phase_coh((int)(sizeof cp / sizeof *cp), cp), "phase-coh --papyrus failed");
                snprintf(r, sizeof r, "%s/pcp/weight.zarr", od); uint8_t *wp = readall(r, "9.6"); snprintf(r, sizeof r, "%s/pcp/phase.zarr", od); uint8_t *pq = readall(r, "9.6");
                snprintf(r, sizeof r, "%s", ct); uint8_t *cv0 = readall(r, "0"); long bad = 0, air = 0, kept = 0;
                for (int z = 0; wp && pq && cv0 && z < NZ; z++) for (int y = 0; y < NY; y++) for (int x = 0; x < NX; x++) {
                    const size_t i = ((size_t)z * NY + y) * NX + x; if (ph[i] == PHASE_UNKNOWN) { bad += pq[i] != PHASE_UNKNOWN || wp[i]; continue; }
                    int pap = 0; for (int dz = -1; dz <= 1 && !pap; dz++) for (int dy = -1; dy <= 1 && !pap; dy++) for (int dx = -1; dx <= 1 && !pap; dx++) {
                        const int zz = z + dz, yy = y + dy, xx = x + dx; if (zz < 0 || yy < 0 || xx < 0 || zz >= NZ || yy >= NY || xx >= NX) continue;
                        pap = cv0[((size_t)zz * NY + yy) * NX + xx] >= T; }
                    if (pap) { kept++; bad += pq[i] != ph[i] || abs((int)wp[i] - (int)w2[i]) > 1; } else { air++; bad += pq[i] != PHASE_UNKNOWN || wp[i]; } }
                printf("test_surflabels: phase-coh --papyrus %d dilate 1: %ld known voxels masked as air, %ld kept, %ld voxels off the brute-force mask\n", T, air, kept, bad);
                CHECK(wp && pq && cv0 && !bad && air > 1000 && kept > 1000, "phase-coh --papyrus is not the dilated CT threshold mask");
                free(wp); free(pq); free(cv0);
            }
            free(w2);
        }
        free(wt);
        free(ig); free(ph);
    } }
    {   /* --geom: geom-check defect cells covering wrap 3 (rows 0..63, every 1/64 turn), slip z bins on wrap 6 rows 16..31 */
        snprintf(p, sizeof p, "%s/geom", root); snprintf(cmd, sizeof cmd, "mkdir -p %s", p); if (system(cmd)) return 1;
        snprintf(p, sizeof p, "%s/geom/cells.tsv", root); FILE *g = fopen(p, "w"); fprintf(g, "wrap\trow0\tturn_frac\tn\n");
        for (int k = 0; k < 64; k++) fprintf(g, "w003\t0\t%.4f\t100\n", (k + 0.5) / 64);
        fclose(g);
        snprintf(p, sizeof p, "%s/geom/zbins.tsv", root); g = fopen(p, "w"); fprintf(g, "wrap\tzbin\trow0\n"); for (int k = 0; k < 4; k++) fprintf(g, "w001\t%d\t%d\n", k, 8 * k);
        fclose(g);
        snprintf(p, sizeof p, "%s/geom/slip.tsv", root); g = fopen(p, "w"); fprintf(g, "wrap\tzbin\tn\tslip_mean\tslip_abs\nw006\t2\t9\t0.9\t0.9\nw006\t1\t9\t0.05\t0.1\n"); fclose(g);
        snprintf(p, sizeof p, "%s/geom", root);
        CHECK(!run(root, "wp", "gm", "1", "4", "--geom", p, nullptr), "surf-labels --geom failed");
        char r[600]; snprintf(r, sizeof r, "%s/gm/codes.zarr", root); uint8_t *cd = readall(r, "9.6");
        long on3 = 0, on6a = 0, on6b = 0;
        for (size_t i = 0; cd && i < (size_t)NZ * NY * NX; i++) { if (!cd[i]) continue;
            const int z = (int)(i / (NY * NX)), y = (int)(i / NX % NY), x = (int)(i % NX); const double Y = y - C0, X = x - C0, Wv = (hypot(X, Y) - 20) / 10 - atan2(Y, X) / (2 * M_PI);
            if (fabs(Wv - 3) < 0.05) on3++;
            if (fabs(Wv - 6) < 0.05) { if (z >= 34 && z <= 46) on6a++; else if (z < 30 || z > 50) on6b++; } }
        printf("test_surflabels: --geom: codes on wrap 3 %ld, on wrap 6 in the slip bin (z 32..47) %ld, elsewhere %ld\n", on3, on6a, on6b);
        CHECK(cd && on3 == 0 && on6a == 0 && on6b > 1000, "--geom masks not applied");
        free(cd);
    }
    {   /* wrap-anchor: the miscounted wraps wx against the true codes (p1) as published-segment GT; votes in the even 0.15 mm z blocks, scored in the odd */
        char ax[400], wdp[400], an[400], gtr[400], od[400]; snprintf(ax, sizeof ax, "%s/axis.json", root); snprintf(wdp, sizeof wdp, "%s/wx", root);
        snprintf(an, sizeof an, "%s/wx.anchor", root); snprintf(gtr, sizeof gtr, "%s/p1/codes.zarr", root); snprintf(od, sizeof od, "%s/ax", root);
        char *av[] = {"ufsm", "wrap-anchor", wdp, an, "--axis", ax, "--axis-um", "9.6", "--dir", "1", "--grid-um", "9.6", "--shape", "64,256,256", "--shard", "128", "--block", "32",
                      "--radius", "12", "--threads", "4", "--gt", gtr, "9.6", "--gt-split-mm", "0.1536", "--gt-half", "A", "--anchor-cell", "4,16", "--anchor-min", "5"};
        CHECK(!cmd_surf_labels((int)(sizeof av / sizeof *av), av), "wrap-anchor failed");
        char *bv[] = {"ufsm", "surf-labels", wdp, od, "--axis", ax, "--axis-um", "9.6", "--dir", "1", "--grid-um", "9.6", "--shape", "64,256,256", "--shard", "128", "--block", "32",
                      "--radius", "12", "--threads", "4", "--gt", gtr, "9.6", "--gt-split-mm", "0.1536", "--gt-half", "B", "--anchor-cell", "4,16", "--anchor", an};   /* (bv[3], the last: replaced below) */
        CHECK(!cmd_surf_labels((int)(sizeof bv / sizeof *bv), bv), "surf-labels --anchor failed");
        char r[600]; snprintf(r, sizeof r, "%s/codes.zarr", od); uint8_t *ac = readall(r, "9.6"); snprintf(r, sizeof r, "%s/p1/codes.zarr", root); uint8_t *gc = readall(r, "9.6");
        snprintf(r, sizeof r, "%s/phase.zarr", od); uint8_t *ph = readall(r, "9.6");
        long both = 0, bad = 0, z1 = 0, kept = 0;
        for (size_t i = 0; ac && gc && ph && i < (size_t)NZ * NY * NX; i++) { kept += ph[i] != PHASE_UNKNOWN;
            if (!ac[i] || !gc[i] || gc[i] == 255) { continue; }
            both++; if (abs(((ac[i] - gc[i]) % 252 + 378) % 252 - 126) > 2) { bad++; if ((i / (NY * NX)) / 16 % 2) z1++; } }
        printf("test_surflabels: wrap-anchor: %ld anchored codes on GT codes, %ld off by > 1/9 turn (%ld of them in held-out z blocks); %ld phase voxels kept\n", both, bad, z1, kept);
        CHECK(both > 50000 && bad == 0 && kept > 1300000, "anchored codes do not match the GT (the +10 / +11 turn miscount not undone)");
        free(ac); free(gc); free(ph);
        /* --anchor A,B union: the file with itself = the file alone (same bytes) */
        char an2[900], od2[400]; snprintf(an2, sizeof an2, "%s,%s", an, an); snprintf(od2, sizeof od2, "%s/ax2", root);
        bv[3] = od2; bv[sizeof bv / sizeof *bv - 1] = an2;
        CHECK(!cmd_surf_labels((int)(sizeof bv / sizeof *bv), bv), "surf-labels --anchor A,A failed");
        char cmd2[1400]; snprintf(cmd2, sizeof cmd2, "cd %s && for s in $(cd ax/codes.zarr/9.6 && find c -type f); do cmp -s ax/codes.zarr/9.6/$s ax2/codes.zarr/9.6/$s || exit 1; done", root);
        CHECK(!system(cmd2), "--anchor A,A differs from --anchor A");
    }
    {   /* --agree smoke: the CT ridges on the sheets as both CT and recto prediction */
        char ax[400], wdp[400], od[400], ct[400], gtr[400]; snprintf(ax, sizeof ax, "%s/axis.json", root); snprintf(wdp, sizeof wdp, "%s/wp", root); snprintf(od, sizeof od, "%s/agree", root);
        snprintf(ct, sizeof ct, "%s/ct0", root); snprintf(gtr, sizeof gtr, "%s/p1/codes.zarr", root);
        char *av[] = {"ufsm", "surf-labels", wdp, od, "--axis", ax, "--axis-um", "9.6", "--dir", "1", "--grid-um", "9.6", "--shape", "64,256,256", "--shard", "128", "--block", "32",
                      "--radius", "12", "--threads", "4", "--ct", ct, "0", "--agree", ct, "0", "--gt", gtr, "9.6"};
        CHECK(!cmd_surf_labels((int)(sizeof av / sizeof *av), av), "surf-labels --agree failed");
    }
    snprintf(p, sizeof p, "%s/hold.json", root);
    CHECK(!run(root, "wp", "h", "1", "4", "--exclude", p, nullptr), "surf-labels --exclude failed");
    const int sense[2] = {1, -1}; const char *outs[2] = {"p1", "m"};
    for (int s = 0; s < 2; s++) {
        char r[600]; snprintf(r, sizeof r, "%s/%s/codes.zarr", root, outs[s]); uint8_t *cd = readall(r, "9.6");
        snprintf(r, sizeof r, "%s/%s/phase.zarr", root, outs[s]); uint8_t *ph = readall(r, "9.6");
        if (!cd || !ph) { CHECK(0, "cannot read outputs"); break; }
        long nc = 0, cbad = 0, np = 0, up = 0, tot = 0; double perr = 0;
        for (int z = 0; z < NZ; z++) for (int y = 0; y < NY; y++) for (int x = 0; x < NX; x++) {
            const size_t i = ((size_t)z * NY + y) * NX + x; const double Y = y - C0, X = x - C0, rr = hypot(X, Y), th = atan2(Y, X);
            const double Wv = (rr - 20) / 10 - sense[s] * th / (2 * M_PI);   /* the analytic W: wrap k where W = k */
            const double U = (rr - 20) / 10;   /* the sheet coordinate q of a point on a sheet */
            if (cd[i]) { nc++; double dq = (cd[i] - 1) / 18.0 - U; dq -= 14 * round(dq / 14); if (fabs(dq) > 1.5 / 18 + 0.05) cbad++; }
            if (ph[i] != PHASE_UNKNOWN && Wv > 1.05 && Wv < 6.95) {
                np++; double e = ph[i] / 252.0 - (Wv - floor(Wv)); e -= round(e); perr += fabs(e);
                const size_t j = i + 1; if (x + 1 < NX && ph[j] != PHASE_UNKNOWN && x > C0 + 25 && abs(y - (int)C0) < 4) { int d = ph[j] - ph[i]; if (d > 126) d -= 252; if (d < -126) d += 252; if (d) { tot++; up += d > 0; } }
            }
        }
        printf("test_surflabels: dir %+d: %ld coded voxels (%ld off), %ld phase voxels, mean |phase error| %.4f turn, outward %ld / %ld\n", sense[s], nc, cbad, np, np ? perr / np : 0, up, tot);
        CHECK(nc > 1000 && cbad <= nc / 200, "codes are not q = k + dir theta / 2 pi");
        CHECK(np > 100000 && perr / np < 0.03, "phase differs from the analytic fraction");
        CHECK(tot > 100 && up > 0.98 * tot, "phase does not rise outward");
        if (s == 0) {   /* identical to band_field_phase_pair on the codes (one block = whole array, radius 12) */
            uint8_t *ref = malloc((size_t)NZ * NY * NX); const int n3[3] = {NZ, NY, NX};
            band_field_phase_pair(cd, n3, 12, ref, nullptr);
            long diff = 0, cmpn = 0; for (int z = 15; z < NZ - 15; z++) for (int y = 15; y < NY - 15; y++) for (int x = 15; x < NX - 15; x++) { const size_t i = ((size_t)z * NY + y) * NX + x; cmpn++; diff += ref[i] != ph[i]; }
            printf("test_surflabels: vs band_field_phase_pair on the written codes: %ld of %ld differ\n", diff, cmpn);
            CHECK(diff == 0, "phase differs from band_field_phase_pair (convention)");
            free(ref);
        }
        free(cd); free(ph);
    }
    { char r[600]; snprintf(r, sizeof r, "%s/d/codes.zarr", root); uint8_t *cd = readall(r, "9.6");
      snprintf(r, sizeof r, "%s/d/phase.zarr", root); uint8_t *ph = readall(r, "9.6"); snprintf(r, sizeof r, "%s/d/ignore.zarr", root); uint8_t *ig = readall(r, "9.6");
      long on4 = 0, gap = 0, gapk = 0, gapi = 0, other = 0, otherk = 0;
      for (size_t i = 0; cd && ph && ig && i < (size_t)NZ * NY * NX; i++) {
          const int z = (int)(i / (NY * NX)), y = (int)(i / NX % NY), x = (int)(i % NX); (void)z;
          const double Y = y - C0, X = x - C0, Wv = (hypot(X, Y) - 20) / 10 - atan2(Y, X) / (2 * M_PI);
          if (fabs(Wv - 4) < 0.05 && cd[i]) on4++;
          if (Wv > 3.15 && Wv < 4.85) { gap++; gapk += ph[i] != PHASE_UNKNOWN; gapi += ig[i] == 1; }
          if ((Wv > 1.15 && Wv < 2.85) || (Wv > 5.15 && Wv < 6.85)) { other++; otherk += ph[i] != PHASE_UNKNOWN && !ig[i]; }
      }
      printf("test_surflabels: --drop w4: codes on it %ld, gap 3..5 %ld voxels: phase known %ld, ignore 1 %ld; elsewhere %ld of %ld kept\n", on4, gap, gapk, gapi, otherk, other);
      CHECK(cd && on4 == 0, "a dropped wrap has codes");
      CHECK(gap > 1000 && gapk == 0 && gapi > 0.9 * gap, "the gap around a dropped wrap is labelled");
      CHECK(other > 1000 && otherk > 0.9 * other, "labels away from the dropped wrap lost");
      free(cd); free(ph); free(ig); }
    { char r[600]; snprintf(r, sizeof r, "%s/c/codes.zarr", root); uint8_t *cd = readall(r, "9.6");
      snprintf(r, sizeof r, "%s/c/ignore.zarr", root); uint8_t *ig = readall(r, "9.6"); snprintf(r, sizeof r, "%s/c/phase.zarr", root); uint8_t *ph = readall(r, "9.6");
      long n255 = 0, nig2 = 0, nconf_lab = 0;
      for (size_t i = 0; cd && ig && ph && i < (size_t)NZ * NY * NX; i++) { n255 += cd[i] == 255; nig2 += ig[i] == 2; nconf_lab += ig[i] == 2 && ph[i] != PHASE_UNKNOWN; }
      printf("test_surflabels: wrap 8 on wrap 2: %ld conflict ignore voxels, codes 255 written %ld, labelled %ld\n", nig2, n255, nconf_lab);
      CHECK(nig2 > 0 && n255 == 0 && nconf_lab == 0, "conflicts not marked / labelled");
      free(cd); free(ig); free(ph); }
    { char r[600]; snprintf(r, sizeof r, "%s/h/ignore.zarr", root); uint8_t *ig = readall(r, "9.6"); long in = 0, n4 = 0, out4 = 0;
      for (int z = 0; ig && z < NZ; z++) for (int y = 0; y < NY; y++) for (int x = 0; x < NX; x++) {
          const int inb = z >= 10 && z < 30 && y >= 160 && y < 190 && x >= 160 && x < 190; const uint8_t v = ig[((size_t)z * NY + y) * NX + x];
          if (inb) { in++; n4 += v == 4; } else out4 += v == 4; }
      printf("test_surflabels: holdout box: %ld of %ld voxels ignore 4 (it has no wraps everywhere: labelled ones only), %ld outside\n", n4, in, out4);
      CHECK(n4 > 0 && out4 == 0, "holdout box not excluded (or the other CT's box applied)");
      free(ig); }
    if (!fails) { snprintf(cmd, sizeof cmd, "rm -rf %s", root); if (system(cmd)) {} }
    printf("test_surflabels: %s\n", fails ? "FAILED" : "ok");
    return fails ? 1 : 0;
}
