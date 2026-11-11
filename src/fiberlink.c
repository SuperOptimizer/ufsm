/* ufsm fiber-link <in.fib> <out.fib> [--gap-mm 0.3] [--um 2.4] [--angle 30] [--lat-mm 0.03] [--tan-mm 0.05] [--links rows.tsv]
   Joins traced fibre pieces (UFIB, level-0 voxels z, y, x; ufsm fiber-trace) end to end into longer fibres. Our mask traces stop early
   (DESIGN.md 2026-11-11: median 0.56 mm on Paris 4 HOLD against 1.8 mm for the official prediction traced at level 4), mostly where
   the predicted tube fades for a few voxels, so a break is usually a short gap along the same line.
   Every piece has two ends, each with an outward tangent: the direction from the point --tan-mm back along the piece to the end.
   A candidate link (end a, end b) needs: the same family, different pieces, gap g = |b - a| <= --gap-mm; the gap leads out of a
   (t_a . v >= cos(--angle) |v|) and into b (-t_b . v >= cos |v|, gaps under 2 voxels are exempt from both), the two tangents
   continue each other (t_a . -t_b >= cos(--angle)), and b lies within --lat-mm of a's tangent line (and a of b's) once the gap is
   bridged straight. Cost = g / gap + (1 - t_a . -t_b) / (1 - cos) + lateral / lat (each term in [0, 1]); links are taken by rising cost,
   each end at most once and never closing a cycle (union-find), so the result is a set of chains. Confidence of a link = 1 - cost / 3.
   Output: each chain as one run (pieces reversed as needed, the gap bridged by the straight segment), family of its pieces.
   --links: one row per link (piece a, end a, piece b, end b, gap mm, angle deg, lateral mm, confidence). CPU, one thread. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "afv.h"

static const char *fl_opt(int argc, char **argv, const char *k, const char *d) { for (int i = 4; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return argv[i + 1]; return d; }
typedef struct { double p[3], t[3]; int run, end; } fl_end;   /* end 0 = the first point, 1 = the last */
typedef struct { int a, b; double cost, g, ang, lat; } fl_link;   /* a, b: end indices (2 run + end) */
static int fl_cmp(const void *x, const void *y) { const double a = ((const fl_link *)x)->cost, b = ((const fl_link *)y)->cost; return (a > b) - (a < b); }
static int uf_find(int *u, int i) { while (u[i] != i) { u[i] = u[u[i]]; i = u[i]; } return i; }
static double dot3(const double *a, const double *b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

int cmd_fiber_link(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm fiber-link <in.fib> <out.fib> [--gap-mm 0.3] [--um 2.4] [--angle 30] [--lat-mm 0.03] [--tan-mm 0.05] [--links rows.tsv]\n"); return 2; }
    const double um = atof(fl_opt(argc, argv, "--um", "2.4")), gap = atof(fl_opt(argc, argv, "--gap-mm", "0.3")) * 1000 / um, lat = atof(fl_opt(argc, argv, "--lat-mm", "0.03")) * 1000 / um;
    const double tanl = atof(fl_opt(argc, argv, "--tan-mm", "0.05")) * 1000 / um, ca = cos(atof(fl_opt(argc, argv, "--angle", "30")) * M_PI / 180);
    if (!(um > 0) || !(gap > 0) || !(lat > 0) || !(tanl > 0) || !(ca < 1)) { fprintf(stderr, "fiber-link: bad options\n"); return 2; }
    fib_set F = {0}; if (fib_load(argv[2], &F)) { fprintf(stderr, "fiber-link: cannot read %s\n", argv[2]); return 1; }
    const int n = F.n > 0 ? F.n : 0; fl_end *E = malloc((size_t)2 * (n ? n : 1) * sizeof *E);
    for (int r = 0; r < n; r++) for (int e = 0; e < 2; e++) {
        const int a = F.off[r], b = F.off[r + 1] - 1, s = e ? b : a, dir = e ? -1 : 1; fl_end *q = &E[2 * r + e]; q->run = r; q->end = e;
        for (int c = 0; c < 3; c++) q->p[c] = F.pts[3 * (size_t)s + c];
        int k = s; double L = 0;   /* walk back along the piece --tan-mm (or to its other end) */
        while (k + dir >= a && k + dir <= b && L < tanl) { double d = 0; for (int c = 0; c < 3; c++) { const double x = F.pts[3 * (size_t)(k + dir) + c] - F.pts[3 * (size_t)k + c]; d += x * x; } L += sqrt(d); k += dir; }
        double t[3], m = 0; for (int c = 0; c < 3; c++) { t[c] = q->p[c] - F.pts[3 * (size_t)k + c]; m += t[c] * t[c]; }
        m = sqrt(m); for (int c = 0; c < 3; c++) q->t[c] = m > 0 ? t[c] / m : 0;
    }
    /* candidates via a hash grid of cell = gap */
    const int64_t HB = 1 << 20; int *head = malloc(HB * sizeof(int)), *nxt = malloc((size_t)2 * (n ? n : 1) * sizeof(int)); for (int64_t i = 0; i < HB; i++) head[i] = -1;
    #define CELL(v) ((int64_t)floor((v) / gap))
    #define HSH(z, y, x) ((uint64_t)((z) * 73856093LL ^ (y) * 19349663LL ^ (x) * 83492791LL) & (HB - 1))
    for (int i = 0; i < 2 * n; i++) { const uint64_t h = HSH(CELL(E[i].p[0]), CELL(E[i].p[1]), CELL(E[i].p[2])); nxt[i] = head[h]; head[h] = i; }
    size_t nl = 0, cap = 1 << 16; fl_link *Lk = malloc(cap * sizeof *Lk);
    for (int i = 0; i < 2 * n; i++) {
        const fl_end *A = &E[i]; const int64_t cz = CELL(A->p[0]), cy = CELL(A->p[1]), cx = CELL(A->p[2]);
        for (int64_t dz = -1; dz <= 1; dz++) for (int64_t dy = -1; dy <= 1; dy++) for (int64_t dx = -1; dx <= 1; dx++)
            for (int j = head[HSH(cz + dz, cy + dy, cx + dx)]; j >= 0; j = nxt[j]) {
                if (j <= i) continue;
                const fl_end *B = &E[j];
                if (B->run == A->run || F.fam[A->run] != F.fam[B->run]) continue;
                if (CELL(B->p[0]) != cz + dz || CELL(B->p[1]) != cy + dy || CELL(B->p[2]) != cx + dx) continue;   /* (hash collisions) */
                double v[3]; for (int c = 0; c < 3; c++) v[c] = B->p[c] - A->p[c]; const double g = sqrt(dot3(v, v)); if (g > gap) continue;
                const double cont = -dot3(A->t, B->t); if (cont < ca) continue;
                if (g >= 2 && (dot3(A->t, v) < ca * g || -dot3(B->t, v) < ca * g)) continue;
                /* lateral: distance of each end from the other's tangent line */
                const double pa = dot3(v, A->t), pb = -dot3(v, B->t), la = sqrt(fmax(0, g * g - pa * pa)), lb = sqrt(fmax(0, g * g - pb * pb)), l = fmax(la, lb);
                if (l > lat) continue;
                if (nl == cap) { cap *= 2; Lk = realloc(Lk, cap * sizeof *Lk); }
                Lk[nl++] = (fl_link){i, j, g / gap + (1 - cont) / (1 - ca) + l / lat, g, acos(fmin(1, cont)) * 180 / M_PI, l};
            }
    }
    #undef CELL
    #undef HSH
    qsort(Lk, nl, sizeof *Lk, fl_cmp);
    int *uf = malloc((size_t)(n ? n : 1) * sizeof(int)), *mate = malloc((size_t)2 * (n ? n : 1) * sizeof(int)); for (int r = 0; r < n; r++) uf[r] = r; for (int i = 0; i < 2 * n; i++) mate[i] = -1;
    FILE *lf = *fl_opt(argc, argv, "--links", "") ? fopen(fl_opt(argc, argv, "--links", ""), "w") : nullptr;
    if (lf) fprintf(lf, "run_a\tend_a\trun_b\tend_b\tgap_mm\tangle_deg\tlateral_mm\tconfidence\n");
    size_t used = 0; double gsum = 0;
    for (size_t k = 0; k < nl; k++) {
        const fl_link *l = &Lk[k];
        if (mate[l->a] >= 0 || mate[l->b] >= 0) continue;
        const int ra = uf_find(uf, E[l->a].run), rb = uf_find(uf, E[l->b].run); if (ra == rb) continue;
        uf[ra] = rb; mate[l->a] = l->b; mate[l->b] = l->a; used++; gsum += l->g;
        if (lf) fprintf(lf, "%d\t%d\t%d\t%d\t%.4f\t%.2f\t%.4f\t%.3f\n", E[l->a].run, E[l->a].end, E[l->b].run, E[l->b].end, l->g * um / 1000, l->ang, l->lat * um / 1000, 1 - l->cost / 3);
    }
    if (lf) fclose(lf);
    /* chains: start at a piece whose end 0 or end 1 is free, walk through the mates */
    fib_set O = {0}; size_t capr = (size_t)n + 1, capp = (size_t)F.npts + 1; O.off = malloc((capr + 1) * sizeof(int)); O.fam = malloc(capr + 1); O.pts = malloc(capp * 12); O.off[0] = 0;
    uint8_t *done = calloc((size_t)(n ? n : 1), 1); int maxp = 0;
    for (int r0 = 0; r0 < n; r0++) {
        if (done[r0]) continue;
        int start = -1, e0 = 0;
        if (mate[2 * r0] < 0) { start = r0; e0 = 0; } else if (mate[2 * r0 + 1] < 0) { start = r0; e0 = 1; } else continue;   /* (interior pieces are reached from a chain end) */
        int r = start, ein = e0, np = 0;   /* ein: the end we enter the piece by */
        while (r >= 0 && !done[r]) {
            done[r] = 1; const int a = F.off[r], b = F.off[r + 1] - 1, m = b - a + 1;
            for (int k = 0; k < m; k++) { const int s = ein ? b - k : a + k; for (int c = 0; c < 3; c++) O.pts[3 * (size_t)(O.npts + np) + c] = F.pts[3 * (size_t)s + c]; np++; }
            const int eo = 2 * r + (1 - ein), mt = mate[eo]; if (mt < 0) break;
            r = E[mt].run; ein = E[mt].end; maxp++;
        }
        O.fam[O.n] = F.fam[start]; O.npts += np; O.off[++O.n] = O.npts;
    }
    for (int r = 0; r < n; r++) if (!done[r]) fprintf(stderr, "fiber-link: piece %d not placed (a cycle?)\n", r);
    printf("fiber-link: %d pieces, %zu candidate links, %zu used (mean gap %.3f mm) -> %d fibres, %d points -> %s\n", n, nl, used, used ? gsum / used * um / 1000 : 0, O.n, O.npts, argv[3]);
    const int rc = fib_save(argv[3], &O) ? 1 : 0; if (rc) fprintf(stderr, "fiber-link: cannot write %s\n", argv[3]);
    fib_free(&F); fib_free(&O); free(E); free(head); free(nxt); free(Lk); free(uf); free(mate); free(done); (void)maxp;
    return rc;
}
