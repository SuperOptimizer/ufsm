/* ufsm ink-audit on a synthetic segment (inknorm 2026-11-18): test_ink_splat's curved surface, ink label and supervision mask, a CT
   papyrus slab d in [-6, 6] along the normal (the open face at d ~ +6 on +N) and a "prediction" that is high only at ink pixels and only
   at t in [16, 20] along the normal = [f + 10, f + 14] (beyond the face, splatted from the surface). The audit must find the face (q50 offset ~ +6), split the pixels inside a holdout box into
   HOLD, give AUC ~ 0.5 at the training band [f - 6, f] and ~ 1 once the band is shifted onto the prediction (best face shift 10 .. 20: the max over the band plateaus while it overlaps the layer;
   best mesh shift = face shift + ~6), and report the label ink share. CPU only (no CUDA call). */
#include "store.h"
#include "tiffw.h"
#include "z3w.h"
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

enum { NV = 256, GW = 40, GH = 44 };
static const double STEP = 4.0;
static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); fails++; } } while (0)
static void surf_at(double u, double v, double P[3]) { const double x = 30 + STEP * u, y = 28 + STEP * v; P[0] = 110 + 0.15 * x + 6 * sin(y / 25.0); P[1] = y; P[2] = x; }
static void normal_at(double u, double v, double N[3]) {
    const double e = 1e-3; double a[3], b[3], c[3], d[3], du[3], dv[3];
    surf_at(u - e, v, a); surf_at(u + e, v, b); surf_at(u, v - e, c); surf_at(u, v + e, d);
    for (int k = 0; k < 3; k++) { du[k] = (b[k] - a[k]) / (2 * e); dv[k] = (d[k] - c[k]) / (2 * e); }
    N[0] = du[1] * dv[2] - du[2] * dv[1]; N[1] = du[2] * dv[0] - du[0] * dv[2]; N[2] = du[0] * dv[1] - du[1] * dv[0];
    const double l = sqrt(N[0] * N[0] + N[1] * N[1] + N[2] * N[2]); for (int k = 0; k < 3; k++) N[k] /= l;
}
static int ink_at(int r, int c) { const double dr = r - 80, dc = c - 70; return dr * dr + dc * dc < 30 * 30 || (c >= 110 && c < 122); }
static int sup_at(int r, int c) { (void)c; return !(r >= 120 && r < 140); }
int cmd_ink_audit(int argc, char **argv);
int cmd_ink_rayface(int argc, char **argv);

int main(void) {
    char dir[] = "/tmp/ufsm_inkaudit_XXXXXX"; if (!mkdtemp(dir)) { perror("mkdtemp"); return 1; }
    char p[600], q[600];
    float *X = malloc(GW * GH * 4), *Y = malloc(GW * GH * 4), *Z = malloc(GW * GH * 4);
    for (int v = 0; v < GH; v++) for (int u = 0; u < GW; u++) { double P[3]; surf_at(u, v, P); X[v * GW + u] = (float)P[2]; Y[v * GW + u] = (float)P[1]; Z[v * GW + u] = (float)P[0]; }
    snprintf(p, sizeof p, "%s/seg.tifxyz", dir); if (tifxyz_write(p, GW, GH, X, Y, Z, STEP, 1.0, -1, "test")) { fprintf(stderr, "tifxyz_write failed\n"); return 1; }
    const int CW = (int)(GW * STEP), CHh = (int)(GH * STEP); uint8_t *lab = malloc((size_t)CW * CHh), *msk = malloc((size_t)CW * CHh);
    long nsup[3] = {0, 0, 0}, nink[3] = {0, 0, 0};   /* per split: HOLD (2) = surface y < 88 (the holdout box below), canvas row r < 60 */
    for (int r = 0; r < CHh; r++) for (int c = 0; c < CW; c++) { lab[r * CW + c] = ink_at(r, c) ? 255 : 0; msk[r * CW + c] = sup_at(r, c) ? 255 : 0; if (sup_at(r, c)) { nsup[r < 60 ? 2 : 0]++; nink[r < 60 ? 2 : 0] += ink_at(r, c); } }
    snprintf(p, sizeof p, "%s/inklabels.tif", dir); tiffw_write(p, CW, CHh, 1, lab, 1);
    snprintf(p, sizeof p, "%s/mask.tif", dir); tiffw_write(p, CW, CHh, 1, msk, 1);
    /* CT (papyrus 200 at |d| <= 6, else 20) and the prediction (230 at ink columns with d in [12, 20], else 30); d along the local normal */
    uint8_t *ct = malloc((size_t)NV * NV * NV), *pr = malloc((size_t)NV * NV * NV);
    for (int64_t z = 0; z < NV; z++) for (int64_t y = 0; y < NV; y++) for (int64_t x = 0; x < NV; x++) {
        const double u = (x - 30) / STEP, v = (y - 28) / STEP; double P[3], N[3]; surf_at(u, v, P); normal_at(u, v, N);
        const double d = (z - P[0]) * N[0]; const size_t k = (size_t)(z * NV + y) * NV + x; const int r = (int)(y - 28), c = (int)(x - 30);
        ct[k] = d >= -6 && d <= 6 ? 200 : 20;
        (void)r; (void)c; pr[k] = 30;
    }
    for (double r = 0; r < CHh; r += 0.25) for (double c = 0; c < CW; c += 0.25) {   /* the prediction splatted from the surface along its normal */
        if (!ink_at((int)lround(r), (int)lround(c))) continue;
        double P[3], N[3]; surf_at(c / STEP, r / STEP, P); normal_at(c / STEP, r / STEP, N);
        for (double t = 16; t <= 20; t += 0.25) { int64_t v[3]; int ok = 1; for (int k = 0; k < 3; k++) { v[k] = (int64_t)lround(P[k] + t * N[k]); ok &= v[k] >= 0 && v[k] < NV; }
            if (ok) pr[((size_t)v[0] * NV + v[1]) * NV + v[2]] = 230; }
    }
    const int64_t sh[3] = {NV, NV, NV}; const double um[1] = {1.0};
    const char *nm[2] = {"ct.zarr", "pred.zarr"}; uint8_t *vb[2] = {ct, pr};
    for (int i = 0; i < 2; i++) { snprintf(p, sizeof p, "%s/%s", dir, nm[i]); snprintf(q, sizeof q, "%s/1", p); z3w *w = z3w_create(q, sh, 256, 0, 0, nullptr);
        if (!w || z3w_write_shard(w, 0, 0, 0, vb[i], 4) || z3w_close(w)) { fprintf(stderr, "write %s failed\n", nm[i]); return 1; } z3w_write_group(p, um, 1, "t", nullptr); }
    snprintf(p, sizeof p, "%s/segs.json", dir); FILE *f = fopen(p, "w");
    fprintf(f, "{\"segments\": [{\"tifxyz\": \"%s/seg.tifxyz\", \"labels\": \"%s/inklabels.tif\", \"mask\": \"%s/mask.tif\"}]}\n", dir, dir, dir); fclose(f);
    snprintf(p, sizeof p, "%s/ho.json", dir); f = fopen(p, "w");   /* a holdout box over canvas rows 0 .. ~60 (y 28 .. 88) */
    fprintf(f, "{\"boxes\": [{\"name\": \"h\", \"ct\": \"x/ct.zarr\", \"l0\": [0, 0, 0, 256, 88, 256]}]}\n"); fclose(f);
    store_global_init();
    /* the audit in a child (its report on stdout) */
    char segs[600], ho[600], out[600], tsv[600]; snprintf(segs, sizeof segs, "%s/segs.json", dir); snprintf(ho, sizeof ho, "%s/ho.json", dir);
    snprintf(out, sizeof out, "%s/a", dir); snprintf(tsv, sizeof tsv, "%s/a.tsv", dir);
    fflush(stdout); fflush(stderr);
    const pid_t ch = fork();
    if (ch == 0) {
        const int fd = open(tsv, O_WRONLY | O_CREAT | O_TRUNC, 0644); if (fd < 0) _exit(9); dup2(fd, 1);
        char *av[] = {"ufsm", "ink-audit", segs, "--ct", dir, "ct.zarr", "--ct-um", "1", "--level", "0", "--iso", "100", "--holdouts", ho, "--pred", dir, "pred.zarr/1",
                      "--pred-level", "0", "--stride", "1", "--tag", "t", "--out", out, "--blocks", "--threads", "4", nullptr};
        _exit(cmd_ink_audit(28, av));
    }
    int st = 0; waitpid(ch, &st, 0); CHECK(WIFEXITED(st) && !WEXITSTATUS(st), "ink-audit exit %d", WIFEXITED(st) ? WEXITSTATUS(st) : -1);
    f = fopen(tsv, "r"); if (!f) { fprintf(stderr, "FAIL: no report\n"); return 1; }
    char ln[4096]; int nrow = 0, have[3] = {0, 0, 0}; long px[3] = {0, 0, 0};
    while (fgets(ln, sizeof ln, f)) {
        if (!strncmp(ln, "tag\t", 4)) continue;
        char *tok[40]; int nt = 0; for (char *t = strtok(ln, "\t\n"); t && nt < 40; t = strtok(nullptr, "\t\n")) tok[nt++] = t;
        if (nt < 22) { CHECK(0, "short row (%d fields)", nt); continue; }
        const int sp = !strcmp(tok[1], "TRAIN") ? 0 : !strcmp(tok[1], "VAL") ? 1 : 2; have[sp] = 1; px[sp] = atol(tok[2]); nrow++;
        const double share = atof(tok[3]), fi = atof(tok[4]), q50 = atof(tok[7]), a0 = atof(tok[16]), sf = atof(tok[18]), af = atof(tok[19]), sm = atof(tok[20]), am = atof(tok[21]);
        fprintf(stderr, "%s: px %s ink %.3f face found %.3f q50 %+.0f; AUC at the band %.3f, best face shift %+.0f (%.3f), mesh %+.0f (%.3f)\n", tok[1], tok[2], share, fi, q50, a0, sf, af, sm, am);
        CHECK(fi > 0.95, "%s: face found %.3f", tok[1], fi);
        CHECK(q50 >= 5 && q50 <= 7, "%s: face offset q50 %+.0f (the slab's face at +6)", tok[1], q50);
        CHECK(a0 < 0.6, "%s: AUC at the training band %.3f (the prediction lies beyond it)", tok[1], a0);
        CHECK(af > 0.97 && sf >= 10 && sf <= 20, "%s: best face shift %+.0f AUC %.3f", tok[1], sf, af);
        CHECK(am > 0.97 && fabs((sm - sf) - 6) <= 2, "%s: best mesh shift %+.0f vs face %+.0f (face ~ mesh + 6)", tok[1], sm, sf);
        CHECK(fabs(share - (double)nink[sp] / nsup[sp]) < 0.02, "%s: ink share %.3f vs labels %.3f", tok[1], share, (double)nink[sp] / nsup[sp]);
    }
    fclose(f);
    CHECK(have[0] && have[2] && !have[1], "splits TRAIN %d VAL %d HOLD %d", have[0], have[1], have[2]);
    CHECK(px[2] > 2000 && px[0] > 2000, "HOLD %ld / TRAIN %ld px", px[2], px[0]);
    snprintf(p, sizeof p, "%s/a.blocks.tsv", dir); f = fopen(p, "r"); int nb = 0; while (f && fgets(ln, sizeof ln, f)) nb++; if (f) fclose(f);
    CHECK(nb == 2, "blocks table rows %d (header + one 2048-block)", nb);
    /* ink-rayface --profile (2026-11-20): the prediction as a volume along the label pixels' normals; its ink peak lies 10 .. 14 past the
       open face (+6; the layer at t 16 .. 20), so none of its mass in the face band [f - 6, f], and the CT level as a volume equals the CT */
    snprintf(p, sizeof p, "%s/src.json", dir); f = fopen(p, "w");
    fprintf(f, "{\"sources\": [{\"name\": \"t\", \"root\": \"%s\", \"ct\": \"ct.zarr\", \"um\": 1, \"iso\": 100, \"targets\": {\"ink\": {\"encoding\": \"prob254\", \"min_level\": 0, "
               "\"onfly\": {\"segments\": \"%s\", \"level\": 0, \"iso\": 100, \"face\": [-6, 0], \"cache_dir\": \"%s/dc\"}}}}]}\n", dir, segs, dir); fclose(f);
    char srcj[600], pv[1400], ptsv[600], pout[600]; snprintf(srcj, sizeof srcj, "%s/src.json", dir); snprintf(ptsv, sizeof ptsv, "%s/p.tsv", dir); snprintf(pout, sizeof pout, "%s/p.out", dir);
    snprintf(pv, sizeof pv, "pred=%s|pred.zarr/1|1;ctv=%s|ct.zarr/1|1", dir, dir);
    fflush(stdout); fflush(stderr);
    const pid_t c2 = fork();
    if (c2 == 0) {
        const int fd = open(pout, O_WRONLY | O_CREAT | O_TRUNC, 0644); if (fd < 0) _exit(9); dup2(fd, 1);
        char *av[] = {"ufsm", "ink-rayface", srcj, "--rays", "0", "--threads", "2", "--profile", "600", "--per-tile", "64", "--pfeat", "1", "--pvol", pv, "--pout", ptsv, nullptr};
        const int r = cmd_ink_rayface(17, av); fflush(stdout); _exit(r);
    }
    st = 0; waitpid(c2, &st, 0); CHECK(WIFEXITED(st) && !WEXITSTATUS(st), "ink-rayface --profile exit %d", WIFEXITED(st) ? WEXITSTATUS(st) : -1);
    f = fopen(pout, "r"); int seen = 0; char ctm[2][4096] = {"", ""};
    while (f && fgets(ln, sizeof ln, f)) {
        fputs(ln, stderr);
        double h[13], band = -1, milb = -1; char nm2[32];
        if (sscanf(ln, "ink profile: %31s ink, face: peak t - t_f (<-20 <-12 <-8 <-6 <-3 <0 0 <=3 <=6 <=8 <=12 <=20 >20): %lf%% %lf%% %lf%% %lf%% %lf%% %lf%% %lf%% %lf%% %lf%% %lf%% %lf%% %lf%% %lf%% | mass in the face band %lf%%, in the MIL band %lf%%",
                   nm2, h, h + 1, h + 2, h + 3, h + 4, h + 5, h + 6, h + 7, h + 8, h + 9, h + 10, h + 11, h + 12, &band, &milb) == 16 && !strcmp(nm2, "pred")) {
            seen = 1; CHECK(h[10] + h[11] > 90, "profile: pred peak at t_f + 9 .. 20 in %.1f%% of ink pixels", h[10] + h[11]);
            CHECK(band < 15, "profile: pred mass in the face band %.1f%%", band);
        }
        { int bt; double ba; char rr[32];
          if (sscanf(ln, "ink profile auc: pred %31s", rr) == 1 && strstr(ln, "best t ") && sscanf(strstr(ln, "best t "), "best t %d AUC %lf", &bt, &ba) == 2) {
              if (!strcmp(rr, "face")) { seen |= 2; CHECK(bt >= 10 && bt <= 14 && ba > 0.97, "profile auc: pred face-relative best t %+d AUC %.3f (the layer at t_f + 10 .. 14)", bt, ba); }
              if (!strcmp(rr, "mesh")) { seen |= 4; CHECK(bt >= 16 && bt <= 20 && ba > 0.97, "profile auc: pred mesh-relative best t %+d AUC %.3f (the layer at 16 .. 20)", bt, ba); } } }
        if (strstr(ln, "ink profile auc: ctv.g ") && strstr(ln, " face ")) seen |= 8;   /* (--pfeat: the CT volume's derivative channel; the slab's edges at t_f, t_f - 12) */
        if (strstr(ln, "ink, face         mean")) { const int k = !strncmp(ln, "ink profile: CT ", 16) ? 0 : strstr(ln, " ctv ") ? 1 : -1; if (k >= 0) snprintf(ctm[k], sizeof ctm[k], "%s", strstr(ln, "mean")); }
    }
    if (f) fclose(f);
    CHECK(seen == 15, "profile: pred / feature lines missing (%d)", seen);
    CHECK(ctm[0][0] && !strcmp(ctm[0], ctm[1]), "profile: the CT as a volume differs from the CT channel");
    {   /* --map (2026-11-22, R11's tracker): (1) an identity reg.json == no map, the report byte for byte; (2) the mesh and the holdout box
           written in a "coarse" scan shifted by t (A = [I | t]: c = f + t, the box named by the coarse key) == the plain audit within float
           rounding; (3) --blocks-size 64: the block rows sum to the default table's counts */
        const int tz = 3, ty = 5, tx = 7;
        for (int v = 0; v < GH; v++) for (int u = 0; u < GW; u++) { double P[3]; surf_at(u, v, P); X[v * GW + u] = (float)(P[2] + tx); Y[v * GW + u] = (float)(P[1] + ty); Z[v * GW + u] = (float)(P[0] + tz); }
        snprintf(p, sizeof p, "%s/segT.tifxyz", dir); CHECK(!tifxyz_write(p, GW, GH, X, Y, Z, STEP, 1.0, -1, "testT"), "tifxyz T");
        snprintf(p, sizeof p, "%s/segsT.json", dir); f = fopen(p, "w");
        fprintf(f, "{\"segments\": [{\"tifxyz\": \"%s/segT.tifxyz\", \"labels\": \"%s/inklabels.tif\", \"mask\": \"%s/mask.tif\"}]}\n", dir, dir, dir); fclose(f);
        snprintf(p, sizeof p, "%s/hoT.json", dir); f = fopen(p, "w"); fprintf(f, "{\"boxes\": [{\"name\": \"h\", \"ct\": \"c/coarse.zarr\", \"l0\": [%d, %d, %d, 256, 88, 256]}]}\n", tz, ty, tx); fclose(f);
        const char *rj = "{\"coarse\": {\"key\": \"%s\", \"um\": 1}, \"fine\": {\"key\": \"ct.zarr\", \"um\": 1}, \"A\": [[1, 0, 0, %d], [0, 1, 0, %d], [0, 0, 1, %d]]}";
        snprintf(p, sizeof p, "%s/regI.json", dir); f = fopen(p, "w"); fprintf(f, rj, "x/ct.zarr", 0, 0, 0); fclose(f);
        snprintf(p, sizeof p, "%s/regT.json", dir); f = fopen(p, "w"); fprintf(f, rj, "c/coarse.zarr", tx, ty, tz); fclose(f);   /* (A rows x y z) */
        char rep[4][4096] = {"", "", "", ""}; long bsum[2] = {0, 0};
        for (int k = 0; k < 4; k++) {   /* (k 3: --hold-only in place of --blocks) */
            char sj[600], hj[600], mj[600], ot[600], oo[600];
            snprintf(sj, sizeof sj, "%s/%s.json", dir, k == 1 ? "segsT" : "segs"); snprintf(hj, sizeof hj, "%s/%s.json", dir, k == 1 ? "hoT" : "ho");
            snprintf(mj, sizeof mj, "%s/%s.json", dir, k == 1 ? "regT" : "regI"); snprintf(ot, sizeof ot, "%s/m%d.tsv", dir, k); snprintf(oo, sizeof oo, "%s/m%d", dir, k);
            fflush(stdout); fflush(stderr);
            const pid_t c3 = fork();
            if (c3 == 0) {
                const int fd = open(ot, O_WRONLY | O_CREAT | O_TRUNC, 0644); if (fd < 0) _exit(9); dup2(fd, 1);
                char *av[] = {"ufsm", "ink-audit", sj, "--ct", dir, "ct.zarr", "--ct-um", "1", "--level", "0", "--iso", "100", "--holdouts", hj, "--pred", dir, "pred.zarr/1",
                              "--pred-level", "0", "--stride", "1", "--tag", "t", "--out", oo, k == 3 ? "--hold-only" : "--blocks", "--threads", "4", "--map", mj, "--map-pad", "0", "--blocks-size", k == 2 ? "64" : "2048", nullptr};
                _exit(cmd_ink_audit(34, av));
            }
            st = 0; waitpid(c3, &st, 0); CHECK(WIFEXITED(st) && !WEXITSTATUS(st), "ink-audit --map %d exit %d", k, WIFEXITED(st) ? WEXITSTATUS(st) : -1);
            FILE *g = fopen(ot, "r"); size_t n = g ? fread(rep[k], 1, sizeof rep[k] - 1, g) : 0; if (g) fclose(g); rep[k][n] = 0;
            snprintf(p, sizeof p, "%s/m%d.blocks.tsv", dir, k); g = fopen(p, "r"); int rows = 0;
            while (g && fgets(ln, sizeof ln, g)) { unsigned long long b[6]; char t2[16], bx[128];
                if (sscanf(ln, "%15s %127s %llu %llu %llu %llu %llu %llu", t2, bx, b, b + 1, b + 2, b + 3, b + 4, b + 5) == 8) { rows++; if (k != 1) bsum[k == 2] += (long)(b[0] + b[1] + b[2] + b[3] + b[4] + b[5]); } }
            if (g) fclose(g);
            if (k == 2) CHECK(rows > 4, "--blocks-size 64: %d block rows", rows);
        }
        char plain[4096] = ""; { FILE *g = fopen(tsv, "r"); size_t n = g ? fread(plain, 1, sizeof plain - 1, g) : 0; if (g) fclose(g); plain[n] = 0; }
        CHECK(*plain && !strcmp(rep[0], plain), "ink-audit: an identity --map changes the report");
        CHECK(!strcmp(rep[2], plain), "ink-audit: --blocks-size changes the report");
        { const char *h0 = strstr(plain, "\tHOLD\t"), *h3 = strstr(rep[3], "\tHOLD\t"); const size_t n0 = h0 ? strcspn(h0, "\n") : 0, n3 = h3 ? strcspn(h3, "\n") : 0;
          fprintf(stderr, "--hold-only: HOLD row %s\n", h0 && h3 && n0 == n3 && !memcmp(h0, h3, n0) ? "identical" : "DIFFERS");
          CHECK(h0 && h3 && n0 == n3 && !memcmp(h0, h3, n0), "ink-audit --hold-only changes the HOLD row"); }
        CHECK(bsum[0] > 0 && bsum[0] == bsum[1], "block counts: 2048 blocks %ld vs 64 blocks %ld", bsum[0], bsum[1]);
        /* the shifted scan: rows equal up to float rounding of the shifted mesh (px within 0.5 %, AUCs within 0.01) */
        char *l0[3], *l1[3]; int n0 = 0, n1 = 0; char a0[4096], a1[4096]; snprintf(a0, sizeof a0, "%s", plain); snprintf(a1, sizeof a1, "%s", rep[1]);
        for (char *s = strtok(a0, "\n"); s && n0 < 3; s = strtok(nullptr, "\n")) l0[n0++] = s;
        for (char *s = strtok(a1, "\n"); s && n1 < 3; s = strtok(nullptr, "\n")) l1[n1++] = s;
        CHECK(n0 == n1 && n0 == 3, "shifted map: %d vs %d report lines", n1, n0);
        for (int i = 1; i < n0 && i < n1; i++) {
            double x0[32], x1[32]; int m0 = 0, m1 = 0; char sp0[16] = "", sp1[16] = "";
            for (char *t = l0[i], *e; *t && m0 < 32; t = *e ? e + 1 : e) { e = strchr(t, '\t'); if (!e) e = t + strlen(t); if (m0 == 1) snprintf(sp0, sizeof sp0, "%.*s", (int)(e - t), t); x0[m0++] = atof(t); }
            for (char *t = l1[i], *e; *t && m1 < 32; t = *e ? e + 1 : e) { e = strchr(t, '\t'); if (!e) e = t + strlen(t); if (m1 == 1) snprintf(sp1, sizeof sp1, "%.*s", (int)(e - t), t); x1[m1++] = atof(t); }
            CHECK(!strcmp(sp0, sp1) && m0 == m1, "shifted map: row %d split %s vs %s", i, sp1, sp0);
            fprintf(stderr, "map shifted %s: px %.0f vs %.0f, AUC max %.4f vs %.4f, best face %.4f vs %.4f\n", sp1, x1[2], x0[2], x1[16], x0[16], x1[19], x0[19]);
            CHECK(fabs(x1[2] - x0[2]) <= 0.005 * x0[2] && fabs(x1[16] - x0[16]) < 0.01 && fabs(x1[19] - x0[19]) < 0.01, "shifted map %s differs from the plain audit", sp0);
        }
    }
    snprintf(p, sizeof p, "rm -rf '%s'", dir); if (system(p)) {}
    free(X); free(Y); free(Z); free(lab); free(msk); free(ct); free(pr);
    if (fails) { fprintf(stderr, "test_ink_audit: %d FAILED\n", fails); return 1; }
    fprintf(stderr, "test_ink_audit: PASS\n"); return 0;
}
