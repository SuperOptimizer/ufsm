/* ufsm annot-export: manual winding annotations (an annotate-serve store, src/annot.h) -> inputs the pipeline reads:
     --fib OUT.fib      same-sheet runs as UFIB polylines (level-0 z y x; family 'M'): every "line", every "pair" with dw 0 and every
                        "point" with "same_as" (its point and the target point). fiber-export + fit --fibers --uw-rep-fib read UFIB as
                        same-sheet votes in the slip repair, so these annotations act like (very reliable) fibres.
     --pairs OUT.tsv    every winding relation as "za ya xa zb yb xb dw" (level-0): pairs (dw), consecutive line vertices (0),
                        same_as points (0) and pairs of winding-numbered points within --dmax level-0 voxels (dw = w_b - w_a)
     --gate OUT.boxes   region verdicts as "z y x nz ny nx s" in level --level voxels (good 1, bad 0; unknown left out), the format of
                        phase-coh --box-scale (a per-box factor on the label weights)
     --points DIR       a sparse winding-number raster on the grid of --like ROOT KEY (level --level): every winding-numbered point and
                        line vertex / segment, dilated by --radius voxels, value w mod 254; 255 elsewhere (unknown)
   ufsm annot-regions: a work queue for the annotation page ("next flagged region"): fibre-judge per-tile tables (worst persistent
   sheet-change rate first) and wrap-check verdicts (non-"real" wraps at their bbox centre) -> REGIONS.tsv (name z y x size score note,
   level-0). */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "annot.h"
#include "json.h"
#include "store.h"
#include "tiffw.h"
#include "z3w.h"
#include "zarr3.h"

static const char *eopt(int argc, char **argv, const char *k, const char *d) { for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return argv[i + 1]; return d; }
static int v3(const json *a, double o[3]) {
    if (!a || a->type != J_ARR || a->n != 3) return 0;
    for (int k = 0; k < 3; k++) o[k] = json_num(a->items[k], NAN);
    return o[0] == o[0] && o[1] == o[1] && o[2] == o[2];
}
typedef struct { int n, npts, cap, pcap; int *off; char *fam; float *pts; } ufib;
static void uf_run(ufib *u, const double (*p)[3], int n) {
    if (n < 2) return;
    if (u->n + 1 >= u->cap) { u->cap = u->cap ? 2 * u->cap : 64; u->off = realloc(u->off, (size_t)(u->cap + 1) * sizeof *u->off); u->fam = realloc(u->fam, (size_t)u->cap); }
    if (u->npts + n > u->pcap) { u->pcap = (u->npts + n) * 2; u->pts = realloc(u->pts, (size_t)u->pcap * 3 * sizeof *u->pts); }
    if (!u->n) u->off[0] = 0;
    for (int i = 0; i < n; i++) for (int k = 0; k < 3; k++) u->pts[3 * (size_t)(u->npts + i) + k] = (float)p[i][k];
    u->npts += n; u->fam[u->n] = 'M'; u->n++; u->off[u->n] = u->npts;
}
static int uf_save(const char *path, const ufib *u) {
    FILE *f = fopen(path, "wb"); if (!f) return -1;
    const int32_t h[3] = {1, u->n, u->npts}; int bad = fwrite("UFIB", 1, 4, f) != 4 || fwrite(h, 4, 3, f) != 3;
    int32_t z = 0; if (u->n) bad |= fwrite(u->off, 4, (size_t)u->n + 1, f) != (size_t)u->n + 1; else bad |= fwrite(&z, 4, 1, f) != 1;
    if (u->n) bad |= fwrite(u->fam, 1, (size_t)u->n, f) != (size_t)u->n;
    if (u->npts) bad |= fwrite(u->pts, 12, (size_t)u->npts, f) != (size_t)u->npts;
    return fclose(f) || bad ? -1 : 0;
}
typedef struct { double p[3]; int w, item; } wpt;

int cmd_annot_export(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: ufsm annot-export ANN.json [--fib OUT.fib] [--pairs OUT.tsv [--dmax 2000]] [--gate OUT.boxes --level L]\n"
                        "       [--points OUTDIR --like ROOT KEY --level L [--radius 2]]   (coordinates in the store: level-0 voxels)\n");
        return 2;
    }
    FILE *f = fopen(argv[2], "rb"); if (!f) { fprintf(stderr, "annot-export: cannot read %s\n", argv[2]); return 1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET); char *t = malloc((size_t)n + 1);
    if (fread(t, 1, (size_t)n, f) != (size_t)n) n = 0; t[n] = 0; fclose(f);
    json *J = json_parse(t, (size_t)n); free(t);
    const json *it = J ? json_get(J, "items") : nullptr; if (!it) { fprintf(stderr, "annot-export: %s has no items\n", argv[2]); json_free(J); return 1; }
    const int L = atoi(eopt(argc, argv, "--level", "0")); const double sc = ldexp(1.0, L), dmax = atof(eopt(argc, argv, "--dmax", "2000"));
    ufib U = {0}; wpt *W = nullptr; int nw = 0; FILE *pf = nullptr, *gf = nullptr;
    const char *fibp = eopt(argc, argv, "--fib", nullptr), *pairp = eopt(argc, argv, "--pairs", nullptr), *gatep = eopt(argc, argv, "--gate", nullptr), *ptsp = eopt(argc, argv, "--points", nullptr);
    if (pairp && !(pf = fopen(pairp, "w"))) { fprintf(stderr, "annot-export: cannot write %s\n", pairp); return 1; }
    if (gatep && !(gf = fopen(gatep, "w"))) { fprintf(stderr, "annot-export: cannot write %s\n", gatep); return 1; }
    if (pf) fprintf(pf, "# za ya xa zb yb xb dw (level-0 voxels; dw = winding(b) - winding(a)) from %s\n", argv[2]);
    if (gf) fprintf(gf, "# z y x nz ny nx s: manual region verdicts from %s (level %d voxels; good 1, bad 0)\n", argv[2], L);
    int npair = 0, nline = 0, nreg = 0, nsame = 0;
    for (size_t i = 0; i < it->n; i++) {
        const json *o = it->items[i]; const char *ty = json_str(json_get(o, "type"), ""); double a[3], b[3];
        if (!strcmp(ty, "pair") && v3(json_get(o, "a"), a) && v3(json_get(o, "b"), b)) {
            const int dw = (int)json_num(json_get(o, "dw"), 0); npair++;
            if (pf) fprintf(pf, "%.0f %.0f %.0f %.0f %.0f %.0f %d\n", a[0], a[1], a[2], b[0], b[1], b[2], dw);
            if (!dw) { const double q[2][3] = {{a[0], a[1], a[2]}, {b[0], b[1], b[2]}}; uf_run(&U, q, 2); }
        } else if (!strcmp(ty, "line")) {
            const json *P = json_get(o, "pts"); if (!P || P->type != J_ARR) continue;
            double (*q)[3] = malloc(P->n * sizeof *q); int m = 0;
            for (size_t k = 0; k < P->n; k++) if (v3(P->items[k], q[m])) m++;
            uf_run(&U, (const double (*)[3])q, m); nline++;
            for (int k = 0; pf && k + 1 < m; k++) fprintf(pf, "%.0f %.0f %.0f %.0f %.0f %.0f 0\n", q[k][0], q[k][1], q[k][2], q[k + 1][0], q[k + 1][1], q[k + 1][2]);
            const json *wj = json_get(o, "w");
            if (wj && wj->type == J_NUM) for (int k = 0; k < m; k++) { W = realloc(W, (size_t)(nw + 1) * sizeof *W); W[nw].p[0] = q[k][0]; W[nw].p[1] = q[k][1]; W[nw].p[2] = q[k][2]; W[nw].item = (int)i; W[nw++].w = (int)wj->num; }
            free(q);
        } else if (!strcmp(ty, "point") && v3(json_get(o, "p"), a)) {
            const json *wj = json_get(o, "w");
            if (wj && wj->type == J_NUM) { W = realloc(W, (size_t)(nw + 1) * sizeof *W); W[nw].p[0] = a[0]; W[nw].p[1] = a[1]; W[nw].p[2] = a[2]; W[nw].item = (int)i; W[nw++].w = (int)wj->num; }
            const int sid = (int)json_num(json_get(o, "same_as"), 0);
            for (size_t k = 0; sid > 0 && k < it->n; k++)
                if ((int)json_num(json_get(it->items[k], "id"), 0) == sid && v3(json_get(it->items[k], "p"), b)) {
                    const double q[2][3] = {{a[0], a[1], a[2]}, {b[0], b[1], b[2]}}; uf_run(&U, q, 2); nsame++;
                    if (pf) fprintf(pf, "%.0f %.0f %.0f %.0f %.0f %.0f 0\n", a[0], a[1], a[2], b[0], b[1], b[2]);
                }
        } else if (!strcmp(ty, "region")) {
            const json *bx = json_get(o, "box"); const char *v = json_str(json_get(o, "verdict"), "");
            if (!bx || bx->n != 6 || !strcmp(v, "unknown")) continue;
            double B[6]; for (int k = 0; k < 6; k++) B[k] = json_num(bx->items[k], 0);
            nreg++;
            if (gf) { const long long z0 = (long long)floor(B[0] / sc), y0 = (long long)floor(B[1] / sc), x0 = (long long)floor(B[2] / sc);
                fprintf(gf, "%lld %lld %lld %lld %lld %lld %d   # item %d\n", z0, y0, x0, (long long)ceil(B[3] / sc) - z0, (long long)ceil(B[4] / sc) - y0,
                        (long long)ceil(B[5] / sc) - x0, !strcmp(v, "good"), (int)json_num(json_get(o, "id"), 0)); }
        }
    }
    int nwp = 0;
    for (int i = 0; pf && i < nw; i++) for (int k = i + 1; k < nw; k++) {
        const double d = hypot(hypot(W[i].p[0] - W[k].p[0], W[i].p[1] - W[k].p[1]), W[i].p[2] - W[k].p[2]);
        if (d > dmax || d == 0 || W[i].item == W[k].item) continue;   /* (vertices of one line: already written as its chain) */
        fprintf(pf, "%.0f %.0f %.0f %.0f %.0f %.0f %d\n", W[i].p[0], W[i].p[1], W[i].p[2], W[k].p[0], W[k].p[1], W[k].p[2], W[k].w - W[i].w); nwp++;
    }
    int rc = 0;
    if (pf) fclose(pf);
    if (gf) fclose(gf);
    if (fibp && uf_save(fibp, &U)) { fprintf(stderr, "annot-export: cannot write %s\n", fibp); rc = 1; }
    if (ptsp) {   /* sparse winding raster on the --like grid */
        const char *lr = nullptr, *lk = nullptr; for (int i = 2; i + 2 < argc; i++) if (!strcmp(argv[i], "--like")) { lr = argv[i + 1]; lk = argv[i + 2]; }
        store *s = lr ? store_open(lr) : nullptr; z3 *z = s ? z3_open(s, lk, nullptr) : nullptr;
        if (!z) { fprintf(stderr, "annot-export: --points needs --like ROOT KEY (a uint8 array of level %d)\n", L); rc = 1; }
        else {
            const z3_meta *m = z3_meta_of(z); const int64_t sh[3] = {m->shape[0], m->shape[1], m->shape[2]}; const int R = atoi(eopt(argc, argv, "--radius", "2")), SH = 128;
            z3w *w = z3w_create(ptsp, sh, SH, 0.f, 255, "{\"ufsm\":{\"content\":\"manual winding annotations: w mod 254 at annotated points (dilated), 255 unknown\"}}");
            /* voxels to set: points and line segments sampled at 0.5 voxel steps, each dilated by R */
            typedef struct { int64_t v[3]; uint8_t c; } vox; vox *V = nullptr; size_t nv = 0, cv = 0;
            for (size_t i = 0; i < it->n; i++) {
                const json *o = it->items[i]; const char *ty = json_str(json_get(o, "type"), ""); const json *wj = json_get(o, "w");
                if (!wj || wj->type != J_NUM) continue;
                const uint8_t code = (uint8_t)((((int)wj->num % 254) + 254) % 254);
                double P[2][3]; int segs = 0; const json *pts = nullptr;
                if (!strcmp(ty, "point") && v3(json_get(o, "p"), P[0])) { P[1][0] = P[0][0]; P[1][1] = P[0][1]; P[1][2] = P[0][2]; segs = 1; }
                else if (!strcmp(ty, "line")) pts = json_get(o, "pts");
                const int ns = pts ? (int)pts->n - 1 : segs;
                for (int sgi = 0; sgi < ns; sgi++) {
                    if (pts && !(v3(pts->items[sgi], P[0]) && v3(pts->items[sgi + 1], P[1]))) continue;
                    const double len = hypot(hypot(P[1][0] - P[0][0], P[1][1] - P[0][1]), P[1][2] - P[0][2]) / sc; const int st = (int)ceil(len * 2) + 1;
                    for (int k = 0; k <= st; k++) { const double tt = st ? (double)k / st : 0; int64_t c[3];
                        for (int d = 0; d < 3; d++) c[d] = llround((P[0][d] + tt * (P[1][d] - P[0][d])) / sc);
                        for (int dz = -R; dz <= R; dz++) for (int dy = -R; dy <= R; dy++) for (int dx = -R; dx <= R; dx++) {
                            if (dz * dz + dy * dy + dx * dx > R * R) continue; const int64_t q[3] = {c[0] + dz, c[1] + dy, c[2] + dx};
                            if (q[0] < 0 || q[1] < 0 || q[2] < 0 || q[0] >= sh[0] || q[1] >= sh[1] || q[2] >= sh[2]) continue;
                            if (nv == cv) { cv = cv ? 2 * cv : 4096; V = realloc(V, cv * sizeof *V); }
                            V[nv++] = (vox){{q[0], q[1], q[2]}, code}; } }
                }
            }
            /* group by shard */
            uint8_t *buf = malloc((size_t)SH * SH * SH); int nsh = 0;
            for (size_t i = 0; i < nv; i++) {
                if (V[i].c == 255) continue;   /* (done) */
                const int64_t s0 = V[i].v[0] / SH, s1 = V[i].v[1] / SH, s2 = V[i].v[2] / SH; memset(buf, 255, (size_t)SH * SH * SH);
                for (size_t k = i; k < nv; k++) {
                    if (V[k].c == 255 || V[k].v[0] / SH != s0 || V[k].v[1] / SH != s1 || V[k].v[2] / SH != s2) continue;
                    buf[((V[k].v[0] - s0 * SH) * SH + (V[k].v[1] - s1 * SH)) * SH + (V[k].v[2] - s2 * SH)] = V[k].c; V[k].c = 255;
                }
                if (z3w_write_shard(w, s0, s1, s2, buf, 4)) { fprintf(stderr, "annot-export: shard write failed: %s\n", z3w_error()); rc = 1; break; }
                nsh++;
            }
            if (z3w_close(w)) { fprintf(stderr, "annot-export: %s\n", z3w_error()); rc = 1; }
            printf("annot-export: points raster %s: %zu voxels in %d shards\n", ptsp, nv, nsh); free(buf); free(V);
            z3_close(z);
        }
        if (s) store_close(s);
    }
    printf("annot-export: %zu items: %d pairs, %d lines, %d same_as, %d regions, %d winding-numbered points (%d point pairs within %.0f)\n",
           it->n, npair, nline, nsame, nreg, nw, nwp, dmax);
    if (fibp) printf("annot-export: %s: %d runs, %d points\n", fibp, U.n, U.npts);
    free(U.off); free(U.fam); free(U.pts); free(W); json_free(J); return rc;
}

/* ---- the work queue ---------------------------------------------------------------------------------------------------------- */
typedef struct { char name[64]; double z, y, x, size, score; char note[160]; } rrow;
static int rcmp(const void *a, const void *b) { const double d = ((const rrow *)b)->score - ((const rrow *)a)->score; return (d > 0) - (d < 0); }
int cmd_annot_regions(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: ufsm annot-regions OUT.tsv [--judge T.tsv ...] [--level L (the judge boxes' level, 2)] [--min-arc 0.3 (m)] [--top 200]\n"
                        "       [--wrapcheck W.tsv --wraps DIR --um0 U]   fibre-judge tiles (worst persistent sheet-change rate first) and\n"
                        "       non-'real' wrap-check verdicts (at the wrap's bbox centre) -> REGIONS.tsv for annotate-serve --regions\n");
        return 2;
    }
    const int L = atoi(eopt(argc, argv, "--level", "2")); const double sc = ldexp(1.0, L), mina = atof(eopt(argc, argv, "--min-arc", "0.3"));
    const int top = atoi(eopt(argc, argv, "--top", "200")); rrow *R = nullptr; int nr = 0;
    for (int i = 2; i + 1 < argc; i++) {
        if (strcmp(argv[i], "--judge")) continue;
        FILE *f = fopen(argv[i + 1], "r"); if (!f) { fprintf(stderr, "annot-regions: cannot read %s\n", argv[i + 1]); continue; }
        char line[4096]; int col[64], nc = 0, cH = -1, cV = -1, cHa = -1, cVa = -1, cbox = -1;
        while (fgets(line, sizeof line, f)) {
            char *tok[64]; int nt = 0; for (char *p = strtok(line, "\t\n"); p && nt < 64; p = strtok(nullptr, "\t\n")) tok[nt++] = p;
            if (!nc) { for (int k = 0; k < nt; k++) { col[k] = k; if (!strcmp(tok[k], "box")) cbox = k; if (!strcmp(tok[k], "H_pers")) cH = k; if (!strcmp(tok[k], "V_pers")) cV = k;
                    if (!strcmp(tok[k], "H_arc_m")) cHa = k; if (!strcmp(tok[k], "V_arc_m")) cVa = k; } nc = nt; continue; }
            if (cbox < 0 || cH < 0 || cV < 0 || cHa < 0 || cVa < 0 || nt <= cVa) continue;
            double b[6]; if (sscanf(tok[cbox], "%lf,%lf,%lf,%lf,%lf,%lf", &b[0], &b[1], &b[2], &b[3], &b[4], &b[5]) != 6) continue;
            const double arc = atof(tok[cHa]) + atof(tok[cVa]); if (arc < mina) continue;
            const double rate = (atof(tok[cH]) + atof(tok[cV])) / arc;
            R = realloc(R, (size_t)(nr + 1) * sizeof *R); rrow *r = &R[nr++]; memset(r, 0, sizeof *r);
            snprintf(r->name, sizeof r->name, "fj_%.0f_%.0f_%.0f", b[0], b[1], b[2]);
            r->z = (b[0] + b[3] / 2) * sc; r->y = (b[1] + b[4] / 2) * sc; r->x = (b[2] + b[5] / 2) * sc; r->size = fmax(b[4], b[5]) * sc; r->score = rate;
            snprintf(r->note, sizeof r->note, "fibre judge: %.0f persistent sheet changes /m over %.2f m of fibre", rate, arc);
        }
        (void)col; fclose(f);
    }
    const char *wc = eopt(argc, argv, "--wrapcheck", nullptr), *wd = eopt(argc, argv, "--wraps", nullptr); const double um0 = atof(eopt(argc, argv, "--um0", "0"));
    if (wc && wd && um0 > 0) {
        FILE *f = fopen(wc, "r"); char line[2048];
        while (f && fgets(line, sizeof line, f)) {
            char *tok[32]; int nt = 0; for (char *p = strtok(line, "\t\n"); p && nt < 32; p = strtok(nullptr, "\t\n")) tok[nt++] = p;
            if (nt < 2 || !strcmp(tok[0], "wrap") || !strcmp(tok[nt - 1], "real")) continue;
            char mp[4200], *meta = nullptr; snprintf(mp, sizeof mp, "%s/%s/meta.json", wd, tok[0]); meta = tfx_meta_read(mp);
            if (!meta) { snprintf(mp, sizeof mp, "%s/%s.sfc/meta.json", wd, tok[0]); meta = tfx_meta_read(mp); }
            if (!meta) continue;
            json *J = json_parse(meta, strlen(meta)); free(meta); const json *bb = json_get(J, "bbox"); const double vum = json_num(json_get(J, "voxel_size_um"), um0);
            if (bb && bb->n == 2) {
                const json *a = bb->items[0], *b = bb->items[1]; const double f2 = vum / um0;
                R = realloc(R, (size_t)(nr + 1) * sizeof *R); rrow *r = &R[nr++]; memset(r, 0, sizeof *r);
                snprintf(r->name, sizeof r->name, "wc_%s", tok[0]);
                r->x = (json_num(json_at(a, 0), 0) + json_num(json_at(b, 0), 0)) / 2 * f2; r->y = (json_num(json_at(a, 1), 0) + json_num(json_at(b, 1), 0)) / 2 * f2;
                r->z = (json_num(json_at(a, 2), 0) + json_num(json_at(b, 2), 0)) / 2 * f2; r->size = 2048; r->score = 1e6;   /* wrap-check flags first */
                snprintf(r->note, sizeof r->note, "wrap-check verdict %s for %s (bbox centre)", tok[nt - 1], tok[0]);
            }
            json_free(J);
        }
        if (f) fclose(f);
    }
    qsort(R, (size_t)nr, sizeof *R, rcmp);
    FILE *o = fopen(argv[2], "w"); if (!o) { fprintf(stderr, "annot-regions: cannot write %s\n", argv[2]); free(R); return 1; }
    fprintf(o, "# name z y x size score note (level-0 voxels; worst first)\n");
    for (int i = 0; i < nr && i < top; i++) fprintf(o, "%s\t%.0f\t%.0f\t%.0f\t%.0f\t%.6g\t%s\n", R[i].name, R[i].z, R[i].y, R[i].x, R[i].size, R[i].score, R[i].note);
    fclose(o); printf("annot-regions: %d regions (%d written) -> %s\n", nr, nr < top ? nr : top, argv[2]); free(R); return 0;
}
