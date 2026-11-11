/* ufsm field-diag A_DIR [B_DIR] --phase ROOT KEY [--bins 60,100,150,250] [--tsv 1] [--label NAME]
   Where two fit-field sets (e.g. the latent net vs wind96 on one wbench box) differ against the label phase raster (PHASE_* codes:
   0..251 = phase x 252, 255 unknown; same native grid as the fields, e.g. 9.6 um for Paris 4 / 1667). Per field cell whose centre voxel
   and its +z / +y / +x neighbours are labelled: the label phase g, the label sheet spacing um / |wrapped grad g| and the label normal
   (direction of that gradient). Per spacing bin, for each field set:
     perr   mean |wrap(phase - g)| in turns, over cells with confidence >= 0.05 (0.25 = chance)
     p10    share of labelled cells with |err| < 0.1 turn and confidence >= 0.05 (cells without information count as misses)
     conf   mean phase confidence |(pc, ps) - 128| / 127
     nal    mean n^T T n (T the predicted n n^T from t0..t5; 1 = the label normal, 1/3 isotropic)
     sperr  mean |log2(predicted spacing / label spacing)| over cells with a predicted spacing
   The point: is a gap between two nets concentrated in packed sheets (small spacing: resolution) or spread over all spacings. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fit.h"
#include "store.h"
#include "zarr3.h"

static const char *fd_opt(int argc, char **argv, const char *k, const char *d) { for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return argv[i + 1]; return d; }
static inline double fd_wrap(double d) { return d - round(d); }
enum { FD_NB = 8 };
typedef struct { double n, nc, perr, p10, conf, nal, nsp, sperr; } fd_acc;

/* the per-cell statistics of one field set against the label raster lab (box-shaped over the field's native extent + 1, origin o) */
static void fdg_set(const fit_fields_buf *b, const uint8_t *lab, const int64_t ln[3], const double *edges, int nb, fd_acc *acc) {
    const fit_fields *f = &b->f; const int h = b->h; const double um = b->um;
    for (int i = 0; i < f->D; i++) for (int j = 0; j < f->H; j++) for (int k = 0; k < f->W; k++) {
        const int64_t z = (int64_t)h * i + h / 2, y = (int64_t)h * j + h / 2, x = (int64_t)h * k + h / 2;
        if (z + 1 >= ln[0] || y + 1 >= ln[1] || x + 1 >= ln[2]) continue;
        const size_t li = ((size_t)z * ln[1] + y) * ln[2] + x; const uint8_t a = lab[li];
        if (a > 251) continue;
        const uint8_t bz = lab[li + (size_t)ln[1] * ln[2]], by = lab[li + ln[2]], bx = lab[li + 1];
        if (bz > 251 || by > 251 || bx > 251) continue;
        const double gz = fd_wrap((bz - a) / 252.0), gy = fd_wrap((by - a) / 252.0), gx = fd_wrap((bx - a) / 252.0), gn = sqrt(gz * gz + gy * gy + gx * gx);
        if (gn < 1e-4 || gn > 0.45) continue;   /* flat or aliased label phase */
        const double spl = um / gn; int bi = 0; while (bi < nb && spl >= edges[bi]) bi++;
        fd_acc *A = &acc[bi]; const size_t c = ((size_t)i * f->H + j) * f->W + k;
        const double vc = (f->pc[c] - 128.0) / 127.0, vs = (f->ps[c] - 128.0) / 127.0, cf = sqrt(vc * vc + vs * vs);
        A->n += 1; A->conf += cf;
        if (cf >= 0.05) { const double e = fabs(fd_wrap(atan2(vs, vc) / (2 * M_PI) - a / 252.0)); A->nc += 1; A->perr += e; if (e < 0.1) A->p10 += 1; }
        if (f->t[0]) {
            const double nz = gz / gn, ny = gy / gn, nx = gx / gn;
            const double Tzz = f->t[0][c] / 255.0, Tyy = f->t[1][c] / 255.0, Txx = f->t[2][c] / 255.0;
            const double Tzy = f->t[3][c] / 255.0 - 0.5, Tzx = f->t[4][c] / 255.0 - 0.5, Tyx = f->t[5][c] / 255.0 - 0.5;
            const double tr = Tzz + Tyy + Txx;
            if (tr > 0.05) A->nal += (nz * nz * Tzz + ny * ny * Tyy + nx * nx * Txx + 2 * (nz * ny * Tzy + nz * nx * Tzx + ny * nx * Tyx)) / tr;
        }
        if (f->sp && f->sp[c]) { const double sp = f->sp[c] * h / 8.0 * um; A->nsp += 1; A->sperr += fabs(log2(sp / spl)); }
    }
}

int cmd_field_diag(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: ufsm field-diag A_DIR [B_DIR] --phase ROOT KEY [--bins 60,100,150,250] [--tsv 1] [--label NAME]\n"); return 2; }
    const char *dirs[2] = {argv[2], (argc > 3 && argv[3][0] != '-') ? argv[3] : nullptr};
    const char *pr = nullptr, *pk = nullptr;
    for (int i = 2; i + 2 < argc; i++) if (!strcmp(argv[i], "--phase")) { pr = argv[i + 1]; pk = argv[i + 2]; }
    if (!pr) { fprintf(stderr, "field-diag: --phase ROOT KEY\n"); return 2; }
    double edges[FD_NB]; int nb = 0; char bs[256]; snprintf(bs, sizeof bs, "%s", fd_opt(argc, argv, "--bins", "60,100,150,250"));
    for (char *t = strtok(bs, ","); t && nb < FD_NB - 1; t = strtok(nullptr, ",")) edges[nb++] = atof(t);
    const int tsv = atoi(fd_opt(argc, argv, "--tsv", "0")); const char *lbl = fd_opt(argc, argv, "--label", "-");
    fit_fields_skip_planes(1 | 2);   /* no recto / wind4 planes */
    fit_fields_ct_planes(0);
    fit_fields_buf b[2]; int ns = 0;
    for (int s = 0; s < 2 && dirs[s]; s++) { if (fit_fields_load(dirs[s], 0, 0, 1, &b[s])) { fprintf(stderr, "field-diag: cannot load %s\n", dirs[s]); return 1; } ns++; }
    for (int s = 1; s < ns; s++)
        if (b[s].f.D != b[0].f.D || b[s].f.H != b[0].f.H || b[s].f.W != b[0].f.W || b[s].h != b[0].h || memcmp(b[s].origin, b[0].origin, sizeof b[0].origin)) {
            fprintf(stderr, "field-diag: %s and %s differ in grid / origin\n", dirs[0], dirs[s]); return 1; }
    z3 *z = z3_open(store_open(pr), pk, nullptr); if (!z) { fprintf(stderr, "field-diag: %s\n", z3_error()); return 1; }
    const z3_meta *m = z3_meta_of(z);
    const int64_t o[3] = {b[0].origin[0], b[0].origin[1], b[0].origin[2]};
    int64_t ln[3] = {(int64_t)b[0].f.D * b[0].h + 1, (int64_t)b[0].f.H * b[0].h + 1, (int64_t)b[0].f.W * b[0].h + 1};
    for (int d = 0; d < 3; d++) if (o[d] + ln[d] > m->shape[d]) ln[d] = m->shape[d] - o[d];
    uint8_t *lab = malloc((size_t)ln[0] * ln[1] * ln[2]);
    if (!lab || ln[0] < 2 || ln[1] < 2 || ln[2] < 2 || z3_read(z, o, ln, lab, 8)) { fprintf(stderr, "field-diag: label read failed\n"); return 1; }
    fd_acc acc[2][FD_NB]; memset(acc, 0, sizeof acc);
    for (int s = 0; s < ns; s++) fdg_set(&b[s], lab, ln, edges, nb, acc[s]);
    if (!tsv) printf("field-diag %s: %s%s%s, label %s %s, field %d x %d x %d (h %d, %.3f um)\n", lbl, dirs[0], ns > 1 ? " vs " : "", ns > 1 ? dirs[1] : "", pr, pk,
                     b[0].f.D, b[0].f.H, b[0].f.W, b[0].h, b[0].um);
    for (int bi = 0; bi <= nb; bi++) {
        char rng[64]; if (bi == 0) snprintf(rng, sizeof rng, "<%g", edges[0]); else if (bi == nb) snprintf(rng, sizeof rng, ">=%g", edges[nb - 1]); else snprintf(rng, sizeof rng, "%g-%g", edges[bi - 1], edges[bi]);
        for (int s = 0; s < ns; s++) {
            const fd_acc *A = &acc[s][bi]; if (A->n < 1) continue;
            const double perr = A->nc ? A->perr / A->nc : NAN, p10 = A->p10 / A->n, conf = A->conf / A->n, nal = A->nal / A->n, sperr = A->nsp ? A->sperr / A->nsp : NAN;
            if (tsv) printf("%s\t%c\t%s\t%.0f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n", lbl, 'A' + s, rng, A->n, perr, p10, conf, nal, sperr);
            else printf("  spacing %-9s um  %c  n %9.0f  perr %.4f  p10 %.4f  conf %.3f  nal %.4f  sperr %.3f\n", rng, 'A' + s, A->n, perr, p10, conf, nal, sperr);
        }
    }
    free(lab); for (int s = 0; s < ns; s++) fit_fields_free(&b[s]);
    return 0;
}
