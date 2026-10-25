/* ufsm emb-calib: calibration of the embedding shift vote (src/fit_emb.h) on ground truth: P(r right | conf) per partner distance.
     ufsm emb-calib <fields-dir> --model FIT --codes ROOT KEY --emb GROUP [--emb-output tokens_unit] [--n 20000] [--d 16,48,128]
                    [--rmax 2] [--phi 0] [--seed 1] [--out table.tsv]
   Points a are labelled GT voxels (winding_mod14 codes, KEY = their um); b = the labelled voxel nearest a + D t (t a random unit
   tangent of the fit's sheet at a, within 2 label voxels). Sides as the fitter's repair builds them: the fit's W, the unit gradient of W and
   1 / |grad W| as the sheet spacing; b's W is offset by a random r0 in -rmax..rmax. The right answer: r = k + r0 with k = round((W_b - W_a)
   - s (G_b - G_a)) (G the GT winding, mod-14 difference wrapped to (-7, 7], s = +-1 the fit's direction vs the GT's, the majority sign),
   i.e. the turns by which the fit's b is too high relative to a, plus the planted offset. Output: per D and conf bin, n and P(r right),
   P(|r - right| = 1) (adjacent-sheet picks), plus the share of pairs where the fit itself is off (k != 0). */
#include "fit.h"
#include "fit_emb.h"
#include "store.h"
#include "zarr3.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *copt(int argc, char **argv, const char *k, const char *d) { for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return argv[i + 1]; return d; }
static uint64_t crng;
static double cu(void) { crng ^= crng << 13; crng ^= crng >> 7; crng ^= crng << 17; return (crng >> 11) / 9007199254740992.0; }
typedef struct { int di; int r, r0; float conf; double dm, dg; } crec;

int cmd_emb_calib(int argc, char **argv) {
    const char *dir = argc > 2 ? argv[2] : nullptr, *mdir = copt(argc, argv, "--model", nullptr), *eg = copt(argc, argv, "--emb", nullptr);
    int ci = 0; for (int i = 3; i + 2 < argc; i++) if (!strcmp(argv[i], "--codes")) ci = i;
    if (!dir || !mdir || !eg || !ci) {
        fprintf(stderr, "usage: ufsm emb-calib <fields-dir> --model FIT --codes ROOT KEY --emb GROUP [--emb-output tokens_unit] [--n 20000] [--d 16,48,128] [--rmax 2] [--phi 0] [--seed 1] [--out table.tsv]\n");
        return 2;
    }
    const int N = atoi(copt(argc, argv, "--n", "20000")), rmax = atoi(copt(argc, argv, "--rmax", "2")); const double phi = atof(copt(argc, argv, "--phi", "0"));
    crng = (uint64_t)atoll(copt(argc, argv, "--seed", "1")) * 0x9E3779B97F4A7C15ull + 1;
    int D[16], nd = 0; { char *t = strdup(copt(argc, argv, "--d", "16,48,128")); for (char *q = strtok(t, ","); q && nd < 16; q = strtok(nullptr, ",")) D[nd++] = atoi(q); free(t); }
    fit_fields_buf fb; memset(&fb, 0, sizeof fb);
    if (fit_fields_load(dir, 0, 0, 0, &fb)) { fprintf(stderr, "emb-calib: cannot load %s\n", dir); return 1; }
    fit_set *S = fit_set_load(mdir); if (!S) { fprintf(stderr, "emb-calib: cannot load the fit %s\n", mdir); return 1; }
    const double cf = atof(argv[ci + 2]) / fb.um;   /* label voxel in native voxels */
    store *st = store_open(argv[ci + 1]); z3 *z = st ? z3_open(st, argv[ci + 2], nullptr) : nullptr;
    if (!z || !(cf > 0)) { fprintf(stderr, "emb-calib: cannot open the codes %s %s\n", argv[ci + 1], argv[ci + 2]); return 1; }
    const double ext[3] = {fb.shape[0] * fb.h, fb.shape[1] * fb.h, fb.shape[2] * fb.h};
    int64_t lo[3], ln[3];
    for (int d = 0; d < 3; d++) { lo[d] = (int64_t)floor(fb.origin[d] / cf); ln[d] = (int64_t)ceil((fb.origin[d] + ext[d]) / cf) - lo[d]; if (lo[d] < 0) { ln[d] += lo[d]; lo[d] = 0; } }
    const size_t nl = (size_t)ln[0] * ln[1] * ln[2]; uint8_t *codes = malloc(nl);
    if (!codes || z3_read(z, lo, ln, codes, 8)) { fprintf(stderr, "emb-calib: codes read failed\n"); return 1; }
    size_t *lab = malloc(nl * sizeof *lab), nlab = 0;
    for (size_t i = 0; i < nl; i++) if (codes[i] >= 1 && codes[i] <= 252) lab[nlab++] = i;
    fprintf(stderr, "emb-calib: %zu labelled voxels (cf %.3g) in the box\n", nlab, cf);
    if (nlab < 100) return 1;
    int pad = 0; for (int k = 0; k < nd; k++) if (D[k] > pad) pad = D[k];
    const long long en[3] = {(long long)ceil(ext[0]), (long long)ceil(ext[1]), (long long)ceil(ext[2])};
    emb_vol *E = emb_vol_open(eg, copt(argc, argv, "--emb-output", "tokens_unit"), fb.origin, en, atof(copt(argc, argv, "--emb-scale", "1")), pad + 32);
    if (!E) return 1;
    /* label voxel index -> box-local native point (the voxel's centre) */
    #define LP(i, p) do { const size_t i_ = (i); const int64_t zz = (int64_t)(i_ / ((size_t)ln[1] * ln[2])), yy = (int64_t)((i_ / ln[2]) % ln[1]), xx = (int64_t)(i_ % ln[2]); \
        p[0] = cf * (zz + lo[0] + 0.5) - fb.origin[0]; p[1] = cf * (yy + lo[1] + 0.5) - fb.origin[1]; p[2] = cf * (xx + lo[2] + 0.5) - fb.origin[2]; } while (0)
    crec *R = malloc((size_t)N * sizeof *R); int nr = 0, tries = 0;
    while (nr < N && tries < 50 * N) {
        tries++;
        const size_t ia = lab[(size_t)(cu() * nlab) % nlab]; double pa[3]; LP(ia, pa);
        double ga[3]; const double la = fit_set_gradW(S, pa[0], pa[1], pa[2], 1.0, ga); if (!(la > 1e-6)) continue;
        double na[3] = {ga[0] / la, ga[1] / la, ga[2] / la};
        /* a random unit tangent */
        double v[3] = {cu() - 0.5, cu() - 0.5, cu() - 0.5}, dp = v[0] * na[0] + v[1] * na[1] + v[2] * na[2];
        for (int d = 0; d < 3; d++) v[d] -= dp * na[d];
        const double lv = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); if (lv < 1e-6) continue;
        const int di = nr % nd; double b0[3]; for (int d = 0; d < 3; d++) b0[d] = pa[d] + D[di] * v[d] / lv;
        /* the nearest labelled voxel within 2 label voxels */
        int64_t c[3]; for (int d = 0; d < 3; d++) c[d] = (int64_t)floor((b0[d] + fb.origin[d]) / cf) - lo[d];
        double bd = 1e30; size_t ib = 0;
        for (int dz = -2; dz <= 2; dz++) for (int dy = -2; dy <= 2; dy++) for (int dx = -2; dx <= 2; dx++) {
            const int64_t q[3] = {c[0] + dz, c[1] + dy, c[2] + dx};
            if (q[0] < 0 || q[1] < 0 || q[2] < 0 || q[0] >= ln[0] || q[1] >= ln[1] || q[2] >= ln[2]) continue;
            const size_t i = ((size_t)q[0] * ln[1] + q[1]) * ln[2] + q[2]; if (codes[i] < 1 || codes[i] > 252) continue;
            const double dd = dz * dz + dy * dy + dx * dx; if (dd < bd) { bd = dd; ib = i; }
        }
        if (bd > 1e29) continue;
        double pb[3]; LP(ib, pb);
        double gb[3]; const double lb = fit_set_gradW(S, pb[0], pb[1], pb[2], 1.0, gb); if (!(lb > 1e-6)) continue;
        const double Wa = fit_set_W(S, pa[0], pa[1], pa[2]), Wb = fit_set_W(S, pb[0], pb[1], pb[2]);
        if (!(Wa == Wa) || !(Wb == Wb)) continue;
        double dg = ((int)codes[ib] - (int)codes[ia]) / 18.0; dg = fmod(dg + 7, 14); if (dg < 0) dg += 14; dg -= 7;
        const int r0 = (int)(cu() * (2 * rmax + 1)) - rmax;
        emb_side A = {{pa[0], pa[1], pa[2]}, Wa, {na[0], na[1], na[2]}, 1 / la}, B = {{pb[0], pb[1], pb[2]}, Wb + r0, {gb[0] / lb, gb[1] / lb, gb[2] / lb}, 1 / lb};
        int r; float conf;
        if (emb_shift_vote(E, &A, &B, rmax, phi, &r, &conf)) continue;
        R[nr++] = (crec){di, r, r0, conf, Wb - Wa, dg};
    }
    /* the fit's direction vs the GT's: the sign under which more pairs agree to the turn */
    int agree[2] = {0, 0};
    for (int i = 0; i < nr; i++) for (int s = 0; s < 2; s++) agree[s] += lround(R[i].dm - (s ? -1 : 1) * R[i].dg) == 0;
    const double sg = agree[1] > agree[0] ? -1 : 1;
    enum { NB = 7 }; const double edge[NB + 1] = {0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 3};
    long n[16][NB + 1] = {{0}}, ok[16][NB + 1] = {{0}}, adj[16][NB + 1] = {{0}}, off[16] = {0};
    for (int i = 0; i < nr; i++) {
        const long k = lround(R[i].dm - sg * R[i].dg), want = k + R[i].r0; int b = 0; while (b < NB - 1 && R[i].conf >= edge[b + 1]) b++;
        const int di = R[i].di; off[di] += k != 0;
        for (int bb = 0; bb < 2; bb++) { const int j = bb ? NB : b; n[di][j]++; ok[di][j] += R[i].r == want; adj[di][j] += labs(R[i].r - want) == 1; }
    }
    FILE *o = stdout; const char *op = copt(argc, argv, "--out", nullptr); if (op && !(o = fopen(op, "w"))) { fprintf(stderr, "emb-calib: cannot write %s\n", op); return 1; }
    fprintf(o, "# emb-calib %s model %s emb %s: %d pairs (%d tries), fit direction %+g vs GT (agree %d / %d)\n# D\tconf_lo\tconf_hi\tn\tP_right\tP_adjacent\n",
            dir, mdir, eg, nr, tries, sg, agree[sg < 0], agree[sg > 0]);
    for (int di = 0; di < nd; di++) {
        for (int b = 0; b <= NB; b++) if (n[di][b]) {
            if (b == NB) fprintf(o, "%d\tall\tall\t%ld\t%.4f\t%.4f\n", D[di], n[di][b], (double)ok[di][b] / n[di][b], (double)adj[di][b] / n[di][b]);
            else fprintf(o, "%d\t%.2f\t%.2f\t%ld\t%.4f\t%.4f\n", D[di], edge[b], edge[b + 1], n[di][b], (double)ok[di][b] / n[di][b], (double)adj[di][b] / n[di][b]); }
        fprintf(o, "# D %d: fit off by >= 1 turn on %.2f%% of the pairs\n", D[di], 100.0 * off[di] / (n[di][NB] ? n[di][NB] : 1));
    }
    if (op) fclose(o);
    free(R); free(codes); free(lab); emb_vol_free(E); z3_close(z); fit_fields_free(&fb);
    return 0;
}
