/* ufsm fiber-match on synthetic UFIB sets (2.4 um voxels, a 1000^3 box): the reference = three straight H lines of 3 mm (1250 voxels)
   and one V line; ours = reference line 1 cut into three 1 mm pieces shifted 5 voxels (12 um, inside --r-um 30), line 2 exactly plus
   a parallel duplicate 4 voxels off (must not cut its run), line 3 missing, a false H line far from all, the V line rotated 60 degrees
   (outside --angle 30: no match). Checks: H recall 2/3, H precision = matched / all our H arc, H runs: line 2 one 3 mm run, line 1 three
   ~1 mm runs (arc-median 3 mm, >= 2 mm share 0.5), V recall 0; a reference line leaving the box is split there. Usage: test_fiber_match DIR */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "afv.h"
int cmd_fiber_match(int argc, char **argv);
static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); fails++; } } while (0)
typedef struct { int n, np; int off[64]; char fam[64]; float pts[3 * 20000]; } fset;
static void line(fset *F, char fam, const double a[3], const double b[3], int k) {
    for (int i = 0; i <= k; i++) for (int c = 0; c < 3; c++) F->pts[3 * (F->np + i) + c] = (float)(a[c] + (b[c] - a[c]) * i / k);
    F->np += k + 1; F->fam[F->n] = fam; F->off[++F->n] = F->np;
}
static void save(const fset *F, const char *p) { fib_set f = {F->n, F->np, (int *)F->off, (char *)F->fam, (float *)F->pts}; if (fib_save(p, &f)) { fprintf(stderr, "cannot write %s\n", p); exit(1); } }
static int row(const char *tsv, char fam, const char *set, double v[10]) {
    FILE *f = fopen(tsv, "r"); if (!f) return -1; char l[512]; int ok = -1;
    while (fgets(l, sizeof l, f)) { char fm, st[16]; double x[10] = {0};
        if (sscanf(l, "%c\t%15s\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf", &fm, st, &x[0], &x[1], &x[2], &x[3], &x[4], &x[5], &x[6], &x[7], &x[8], &x[9]) >= 9 && fm == fam && !strcmp(st, set)) { memcpy(v, x, sizeof x); ok = 0; } }
    fclose(f); return ok;
}
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: test_fiber_match DIR\n"); return 2; }
    static fset R, O; char pr[1024], po[1024], pt[1024]; snprintf(pr, sizeof pr, "%s/ref.fib", argv[1]); snprintf(po, sizeof po, "%s/ours.fib", argv[1]); snprintf(pt, sizeof pt, "%s/m.tsv", argv[1]);
    /* reference: H lines along x at z 500, y 200 / 500 / 800; a V line along z; a line leaving the box (split: its outside part ignored) */
    for (int k = 0; k < 3; k++) { const double a[3] = {500, 200 + 300.0 * k, 100}, b[3] = {500, 200 + 300.0 * k, 1350}; line(&R, 'H', a, b, 1250); }
    { const double a[3] = {100, 100, 900}, b[3] = {700, 100, 900}; line(&R, 'V', a, b, 600); }
    /* ours */
    for (int p = 0; p < 3; p++) { const double a[3] = {500, 205, 100 + 417.0 * p}, b[3] = {500, 205, 100 + 417.0 * p + 410}; line(&O, 'H', a, b, 410); }
    { const double a[3] = {500, 500, 100}, b[3] = {500, 500, 1350}; line(&O, 'H', a, b, 1250); }
    { const double a[3] = {500, 504, 100}, b[3] = {500, 504, 1350}; line(&O, 'H', a, b, 1250); }   /* (a parallel duplicate) */
    { const double a[3] = {900, 900, 100}, b[3] = {900, 900, 500}; line(&O, 'H', a, b, 400); }    /* (false) */
    { const double a[3] = {100, 100, 900}, b[3] = {400, 100 + 300 * 1.732, 900}; line(&O, 'V', a, b, 600); }   /* (60 degrees off) */
    save(&R, pr); save(&O, po);
    char *av[] = {"ufsm", "fiber-match", po, pr, "--box", "0,0,0,1000,1000,1000", "--um", "2.4", "--out", pt, nullptr};
    CHECK(!cmd_fiber_match(10, av), "fiber-match failed");
    double h[10], hr[10], v[10];
    CHECK(!row(pt, 'H', "ours", h) && !row(pt, 'H', "ref", hr) && !row(pt, 'V', "ref", v), "rows missing in %s", pt);
    /* H: the reference's three lines are clipped at x 1000: 900 voxels = 2.16 mm each */
    CHECK(fabs(hr[0] - 3) < 0.5 && fabs(hr[2] - 2.16) < 0.05, "H reference: %g lines, median %g mm (want 3, 2.16)", hr[0], hr[2]);
    CHECK(fabs(hr[6] - 2.0 / 3) < 0.04, "H recall %.3f (want 2/3)", hr[6]);
    /* precision: matched = the pieces of line 1, line 2 and its duplicate; unmatched = the false line */
    const double mat = (410 + 410 + 66 + 900 + 900) * 2.4e-3, all = mat + 400 * 2.4e-3;   /* (piece 3 clipped at x 1000; the duplicate is within r) */
    CHECK(fabs(h[6] - mat / all) < 0.04, "H precision %.3f (want %.3f)", h[6], mat / all);
    /* runs: line 2 one 2.16 mm run; line 1 three ~0.98 / 0.98 / 0.2 mm runs (the third piece clipped at x 1000) */
    CHECK(fabs(h[7] - 2.16) < 0.05, "H runs arc-median %.3f mm (want 2.16: the duplicate must not cut line 2's run)", h[7]);
    CHECK(h[9] > 0.42 && h[9] < 0.6, "H runs >= 2 mm share %.3f (want ~0.5)", h[9]);
    CHECK(v[6] < 0.05, "V recall %.3f (want 0: 60 degrees off)", v[6]);
    if (fails) { fprintf(stderr, "test_fiber_match: %d failures\n", fails); return 1; }
    printf("test_fiber_match: ok (H recall %.3f precision %.3f runs arc-median %.2f mm)\n", hr[6], h[6], h[7]);
    return 0;
}
