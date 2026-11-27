/* ATLAS-15k (huggingface.co/datasets/Qualzz20/ATLAS-15k, CC BY-NC 4.0; 2026-11-27, atlas): 14,678 recto surface patches on 28
   Herculaneum scrolls as Parquet rows (id, scroll, scan URL, voxel_um, size_vx, centre / box, area_cm2, x / y / z float32 node grids,
   rows x columns, 2 voxels between nodes, -1 = no node). Validation of the patches as extra labels; the upstream model is not used.

   ufsm atlas-import OUT FILE.parquet... [--scroll S] [--limit N] [--threads T] [--sfc ERR] [--dump 1]
     every row -> OUT/<scroll>/<id>.tifxyz (x / y / z.tif float32, deflate; meta.json: scale 1/2 (2-voxel nodes), voxel_size_um, and
     "scroll", "scan" (the level-0 array's name, e.g. 20250521125136-8.640um-1.2m-116keV-masked.zarr), "scan_url", "atlas_id",
     "dataset"), the directory layout `meshes` onfly sources read (meshcodes.c: <dir>/*.tifxyz). A node counts only when x, y and z are
     all >= 0 (rows where one axis is -1 and another not are made invalid on all three and counted: "mixed").
     OUT/index.tsv: one line per patch (id scroll scan um rows cols valid quads area_cm2 area_cm2_atlas bbox). --dump 1: the Parquet
     metadata only (schema leaves, codecs, row groups). Exit 0 only when every row was written. */
#include "imgops.h"
#include "json.h"
#include "optcheck.h"
#include "parquet.h"
#include "store.h"
#include "zarr3.h"
#include <inttypes.h>
#include "sysinfo.h"
#include "tiffw.h"
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static const char *at_opt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], opt_n(name))) return argv[i + 1];
    return dflt;
}
static int at_is_opt(const char *a) { return a[0] == '-' && a[1] == '-'; }

/* one decoded row group: the columns the import uses */
typedef struct {
    pq_col id, scroll, scan, um, area, x, y, z, bmin, bmax;
    int lid, lscroll, lscan, lum, larea, lx, ly, lz, lbmin, lbmax;
    int64_t *sbmin, *sbmax, *vbmin, *vbmax;   /* box_min / box_max_xyz (list<int32>): record starts / value starts (checked against the nodes) */
    int64_t nrec;
    int64_t *srec[3], *sval[3];   /* x / y / z: per record the first level entry and the first value (nrec + 1 each) */
    int64_t *s_str[3];            /* id / scroll / scan: per record the value index, -1 = null */
    int64_t *s_um, *s_area;
} at_rg;

static void rg_free(at_rg *g) {
    pq_col *c[] = {&g->id, &g->scroll, &g->scan, &g->um, &g->area, &g->x, &g->y, &g->z, &g->bmin, &g->bmax};
    free(g->sbmin); free(g->sbmax); free(g->vbmin); free(g->vbmax);
    for (size_t k = 0; k < sizeof c / sizeof *c; k++) pq_col_free(c[k]);
    for (int k = 0; k < 3; k++) { free(g->srec[k]); free(g->sval[k]); free(g->s_str[k]); }
    free(g->s_um); free(g->s_area); memset(g, 0, sizeof *g);
}

/* record starts of a column (rep == 0), and the value index at each start */
static int rec_index(const pq_col *c, int max_def, int64_t nrec, int64_t **srec, int64_t **sval) {
    *srec = malloc(sizeof(int64_t) * (nrec + 1)); *sval = malloc(sizeof(int64_t) * (nrec + 1));
    if (!*srec || !*sval) return -1;
    int64_t r = 0, v = 0;
    for (int64_t i = 0; i < c->n; i++) {
        if (!c->rep[i]) { if (r >= nrec) return -1; (*srec)[r] = i; (*sval)[r] = v; r++; }
        v += c->def[i] == max_def;
    }
    if (r != nrec || v != c->nv) return -1;
    (*srec)[nrec] = c->n; (*sval)[nrec] = v; return 0;
}
/* a scalar column: the value index of each record (-1 null) */
static int scalar_index(const pq_col *c, const pq_leaf *L, int64_t nrec, int64_t **s) {
    if (L->max_rep) return -1;
    if (c->n != nrec) return -1;
    *s = malloc(sizeof(int64_t) * (nrec + 1)); if (!*s) return -1;
    int64_t v = 0; for (int64_t i = 0; i < nrec; i++) (*s)[i] = c->def[i] == L->max_def ? v++ : -1;
    return 0;
}

/* the grid of record r of a list<list<float>> column: rows / cols, values into out (nullptr: count only). -1 not rectangular */
static int grid_of(const pq_col *c, const pq_leaf *L, const int64_t *srec, const int64_t *sval, int64_t r, int *rows, int *cols, float *out) {
    if (L->max_rep != 2 || L->type != PQ_FLOAT) return -1;
    const int d_row = L->rep_def[0], d_val = L->rep_def[1];
    int64_t v = sval[r]; int nr = 0, nc = -1, cur = 0; size_t o = 0;
    for (int64_t i = srec[r]; i < srec[r + 1]; i++) {
        if (c->rep[i] <= 1 && c->def[i] >= d_row) { if (nr) { if (nc < 0) nc = cur; else if (cur != nc) return -1; } nr++; cur = 0; }
        if (c->def[i] >= d_val) {
            const float f = c->def[i] == L->max_def ? ((const float *)c->vals)[v] : -1.f;
            if (out) out[o++] = f;
            cur++;
        }
        v += c->def[i] == L->max_def;
    }
    if (nr) { if (nc < 0) nc = cur; else if (cur != nc) return -1; }
    *rows = nr; *cols = nr ? nc : 0; return 0;
}

static void str_of(const pq_col *c, int64_t k, char *dst, size_t cap) {
    if (k < 0) { snprintf(dst, cap, "%s", ""); return; }
    snprintf(dst, cap, "%.*s", (int)c->lens[k], (const char *)c->pool + c->offs[k]);
}
/* the array name of a scan URL: the last path component ending in .zarr ("…/20250521125136-8.640um-….zarr/0" -> that .zarr) */
static void scan_name(const char *url, char *dst, size_t cap) {
    char t[1024]; snprintf(t, sizeof t, "%s", url);
    size_t n = strlen(t); while (n && t[n - 1] == '/') t[--n] = 0;
    char *z = nullptr; for (char *p = strstr(t, ".zarr"); p; p = strstr(p + 1, ".zarr")) z = p;
    if (z) { z[5] = 0; char *s = strrchr(t, '/'); snprintf(dst, cap, "%s", s ? s + 1 : t); return; }
    char *s = strrchr(t, '/'); snprintf(dst, cap, "%s", s ? s + 1 : t);
}
static int name_ok(const char *s) {   /* ids / scroll names become paths: no '/', no "..", no quotes */
    if (!*s || !strcmp(s, ".") || strstr(s, "..")) return 0;
    for (const char *p = s; *p; p++) if (*p == '/' || *p == '\'' || *p == '"' || *p == '\\' || (unsigned char)*p < 32) return 0;
    return 1;
}

typedef struct {
    at_rg *g; const pq_file *f; const char *out, *scroll_only, *src; int64_t r0, r1, limit;
    int64_t *done; pthread_mutex_t *mu; FILE *idx; int rc;
    int64_t written, skipped, mixed, bad, boxok, boxbad;
} at_job;

static int write_one(at_job *J, int64_t r) {
    at_rg *g = J->g; const pq_file *f = J->f;
    char id[256], scroll[128], scanurl[1024], scan[512];
    str_of(&g->id, g->s_str[0][r], id, sizeof id); str_of(&g->scroll, g->s_str[1][r], scroll, sizeof scroll);
    str_of(&g->scan, g->s_str[2][r], scanurl, sizeof scanurl); scan_name(scanurl, scan, sizeof scan);
    if (J->scroll_only && strcmp(scroll, J->scroll_only)) { J->skipped++; return 0; }
    if (!name_ok(id) || !name_ok(scroll)) { fprintf(stderr, "atlas-import: row %lld: unusable id / scroll '%s' / '%s'\n", (long long)r, id, scroll); J->bad++; return -1; }
    const double um = g->s_um[r] >= 0 ? ((const float *)g->um.vals)[g->s_um[r]] : NAN;
    const double acm = g->s_area[r] >= 0 ? ((const float *)g->area.vals)[g->s_area[r]] : NAN;
    if (!(um > 0)) { fprintf(stderr, "atlas-import: %s: no voxel_um\n", id); J->bad++; return -1; }
    const pq_col *ax[3] = {&g->x, &g->y, &g->z}; const int la[3] = {g->lx, g->ly, g->lz};
    int rows[3], cols[3];
    for (int k = 0; k < 3; k++)
        if (grid_of(ax[k], pq_leaf_at(f, la[k]), g->srec[k], g->sval[k], r, &rows[k], &cols[k], nullptr)) { fprintf(stderr, "atlas-import: %s: axis %c not a rectangular grid\n", id, "xyz"[k]); J->bad++; return -1; }
    if (rows[1] != rows[0] || rows[2] != rows[0] || cols[1] != cols[0] || cols[2] != cols[0] || rows[0] < 1 || cols[0] < 1) {
        fprintf(stderr, "atlas-import: %s: x / y / z grids differ (%dx%d %dx%d %dx%d)\n", id, rows[0], cols[0], rows[1], cols[1], rows[2], cols[2]); J->bad++; return -1; }
    const int h = rows[0], w = cols[0]; const size_t n = (size_t)w * h;
    float *X[3] = {malloc(n * 4), malloc(n * 4), malloc(n * 4)};
    if (!X[0] || !X[1] || !X[2]) { free(X[0]); free(X[1]); free(X[2]); fprintf(stderr, "atlas-import: out of memory\n"); J->bad++; return -1; }
    for (int k = 0; k < 3; k++) { int a, b; grid_of(ax[k], pq_leaf_at(f, la[k]), g->srec[k], g->sval[k], r, &a, &b, X[k]); }
    size_t valid = 0, mixed = 0, quads = 0;
    for (size_t i = 0; i < n; i++) {
        const int ok = X[0][i] >= 0 && X[1][i] >= 0 && X[2][i] >= 0 && isfinite(X[0][i]) && isfinite(X[1][i]) && isfinite(X[2][i]);
        const int any = X[0][i] >= 0 || X[1][i] >= 0 || X[2][i] >= 0;
        if (!ok) { if (any) mixed++; X[0][i] = X[1][i] = X[2][i] = -1; } else valid++;
    }
    double lo[3] = {1e30, 1e30, 1e30}, hi[3] = {-1e30, -1e30, -1e30};
    for (int r2 = 0; r2 < h; r2++) for (int c = 0; c < w; c++) {
        const size_t i = (size_t)r2 * w + c; if (X[2][i] < 0) continue;
        for (int k = 0; k < 3; k++) { lo[k] = fmin(lo[k], X[k][i]); hi[k] = fmax(hi[k], X[k][i]); }
        if (r2 + 1 < h && c + 1 < w && X[2][i + 1] >= 0 && X[2][i + w] >= 0 && X[2][i + w + 1] >= 0) quads++;
    }
    char dir[2048], extra[2048], src[600];
    snprintf(dir, sizeof dir, "%s/%s", J->out, scroll); mkdir(dir, 0775);
    snprintf(dir, sizeof dir, "%s/%s/%s.tifxyz", J->out, scroll, id);
    snprintf(src, sizeof src, "ufsm atlas-import (ATLAS-15k, %s)", J->src);
    snprintf(extra, sizeof extra, "\"scroll\": \"%s\", \"scan\": \"%s\", \"scan_url\": \"%s\", \"atlas_id\": \"%s\", \"dataset\": \"ATLAS-15k\", \"license\": \"CC BY-NC 4.0\", \"node_step_vx\": 2",
             scroll, scan, scanurl, id);
    const int rc = tifxyz_write_ex(dir, w, h, X[0], X[1], X[2], 2.0, um, -1, src, extra);
    free(X[0]); free(X[1]); free(X[2]);
    if (rc) { fprintf(stderr, "atlas-import: cannot write %s\n", dir); J->bad++; return -1; }
    const double cm2 = (double)quads * 4.0 * um * um * 1e-8;
    if (g->vbmin && g->vbmax && valid) {   /* the dataset's own box_min / box_max_xyz vs the nodes' bbox: floor / ceil within 1 voxel */
        const int64_t a = g->vbmin[r], b = g->vbmax[r]; int ok = g->vbmin[r + 1] - a == 3 && g->vbmax[r + 1] - b == 3;
        for (int k = 0; ok && k < 3; k++) { const int32_t mn = ((const int32_t *)g->bmin.vals)[a + k], mx = ((const int32_t *)g->bmax.vals)[b + k];
                                             if (fabs(mn - floor(lo[k])) > 1 || fabs(mx - ceil(hi[k])) > 1) ok = 0; }
        if (ok) J->boxok++; else { J->boxbad++; if (J->boxbad <= 5) fprintf(stderr, "atlas-import: %s: box_min / box_max_xyz differ from the nodes' bbox\n", id); }
    }
    pthread_mutex_lock(J->mu);
    fprintf(J->idx, "%s\t%s\t%s\t%.4g\t%d\t%d\t%zu\t%zu\t%.6f\t%.6f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\n", id, scroll, scan, um, h, w, valid, quads, cm2, acm,
            valid ? lo[0] : -1, valid ? lo[1] : -1, valid ? lo[2] : -1, valid ? hi[0] : -1, valid ? hi[1] : -1, valid ? hi[2] : -1);
    pthread_mutex_unlock(J->mu);
    J->written++; J->mixed += mixed;
    return 0;
}
static void *job_main(void *a) {
    at_job *J = a;
    for (;;) {
        pthread_mutex_lock(J->mu);
        const int64_t r = *J->done < J->r1 && (J->limit < 0 || *J->done < J->limit) ? (*J->done)++ : -1;
        pthread_mutex_unlock(J->mu);
        if (r < 0) break;
        if (write_one(J, r)) J->rc = -1;
    }
    return nullptr;
}

static int col_load(pq_file *f, int rg, const char *top, pq_col *c, int *leaf) {
    *leaf = pq_find_leaf(f, top);
    if (*leaf < 0) { fprintf(stderr, "atlas-import: no column '%s'\n", top); return -1; }
    if (pq_read_col(f, rg, *leaf, c)) { fprintf(stderr, "atlas-import: column %s: %s\n", top, pq_error()); return -1; }
    return 0;
}

int cmd_atlas_import(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm atlas-import OUT FILE.parquet... [--scroll S] [--limit N] [--threads T] [--sfc ERR] [--dump 1]\n"); return 2; }
    const char *out = argv[2], *only = at_opt(argc, argv, "--scroll", nullptr);
    const int64_t limit = atoll(at_opt(argc, argv, "--limit", "-1")); const int dump = atoi(at_opt(argc, argv, "--dump", "0"));
    int nt = atoi(at_opt(argc, argv, "--threads", "0")); if (nt <= 0) nt = ufsm_ncpu(); if (nt > 64) nt = 64;
    const double sfc = atof(at_opt(argc, argv, "--sfc", "0")); if (sfc > 0) tifxyz_set_sfc(sfc);
    if (!dump && mkdir(out, 0775) && errno != EEXIST) { fprintf(stderr, "atlas-import: cannot create %s\n", out); return 1; }
    FILE *idx = nullptr;
    if (!dump) {
        char p[2048]; snprintf(p, sizeof p, "%s/index.tsv", out);
        idx = fopen(p, "w"); if (!idx) { fprintf(stderr, "atlas-import: cannot write %s\n", p); return 1; }
        fprintf(idx, "id\tscroll\tscan\tum\trows\tcols\tvalid\tquads\tarea_cm2\tarea_cm2_atlas\tx0\ty0\tz0\tx1\ty1\tz1\n");
    }
    pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    int64_t written = 0, skipped = 0, mixed = 0, bad = 0, total = 0, boxok = 0, boxbad = 0; int rc = 0;
    for (int a = 3; a < argc; a++) {
        if (at_is_opt(argv[a])) { a++; continue; }
        if (limit >= 0 && written >= limit) break;
        pq_file *f = pq_open(argv[a]);
        if (!f) { fprintf(stderr, "atlas-import: %s\n", pq_error()); rc = 1; continue; }
        const char *base = strrchr(argv[a], '/'); base = base ? base + 1 : argv[a];
        if (dump) {
            printf("%s: %lld rows, %d row groups, created by %s\n", argv[a], (long long)pq_num_rows(f), pq_num_row_groups(f), pq_created_by(f));
            for (int l = 0; l < pq_num_leaves(f); l++) {
                const pq_leaf *L = pq_leaf_at(f, l); const pq_chunkmeta *m = pq_chunk(f, 0, l);
                printf("  leaf %d %s type %d def %d rep %d rep_def %d,%d codec %s values %lld comp %lld dict %lld\n", l, L->path, L->type, L->max_def, L->max_rep,
                       L->rep_def[0], L->rep_def[1], pq_codec_name(m->codec), (long long)m->num_values, (long long)m->comp_size, (long long)m->dict_off);
            }
            for (int g = 0; g < pq_num_row_groups(f); g++) printf("  row group %d: %lld rows\n", g, (long long)pq_rg_rows(f, g));
            pq_close(f); continue;
        }
        for (int g = 0; g < pq_num_row_groups(f) && (limit < 0 || written < limit); g++) {
            at_rg G = {0}; G.nrec = pq_rg_rows(f, g); total += G.nrec;
            int e = col_load(f, g, "id", &G.id, &G.lid) || col_load(f, g, "scroll", &G.scroll, &G.lscroll) || col_load(f, g, "scan", &G.scan, &G.lscan)
                 || col_load(f, g, "voxel_um", &G.um, &G.lum) || (pq_find_leaf(f, "area_cm2") >= 0 && col_load(f, g, "area_cm2", &G.area, &G.larea))
                 || col_load(f, g, "x", &G.x, &G.lx) || col_load(f, g, "y", &G.y, &G.ly) || col_load(f, g, "z", &G.z, &G.lz);
            if (!e) {
                const pq_col *ax[3] = {&G.x, &G.y, &G.z}; const int la[3] = {G.lx, G.ly, G.lz};
                for (int k = 0; k < 3 && !e; k++) if (rec_index(ax[k], pq_leaf_at(f, la[k])->max_def, G.nrec, &G.srec[k], &G.sval[k])) { fprintf(stderr, "atlas-import: %s rg %d: column %c records do not match the row count\n", base, g, "xyz"[k]); e = 1; }
                const pq_col *sc[3] = {&G.id, &G.scroll, &G.scan}; const int ls[3] = {G.lid, G.lscroll, G.lscan};
                for (int k = 0; k < 3 && !e; k++) if (pq_leaf_at(f, ls[k])->type != PQ_BYTE_ARRAY || scalar_index(sc[k], pq_leaf_at(f, ls[k]), G.nrec, &G.s_str[k])) { fprintf(stderr, "atlas-import: %s rg %d: string column %d not a scalar\n", base, g, k); e = 1; }
                if (!e && (pq_leaf_at(f, G.lum)->type != PQ_FLOAT || scalar_index(&G.um, pq_leaf_at(f, G.lum), G.nrec, &G.s_um))) { fprintf(stderr, "atlas-import: voxel_um not a float scalar\n"); e = 1; }
                if (!e && pq_find_leaf(f, "area_cm2") < 0) { G.s_area = malloc(sizeof(int64_t) * (G.nrec + 1)); for (int64_t r = 0; r < G.nrec; r++) G.s_area[r] = -1; }   /* (only reported) */
                else if (!e && (pq_leaf_at(f, G.larea)->type != PQ_FLOAT || scalar_index(&G.area, pq_leaf_at(f, G.larea), G.nrec, &G.s_area))) { fprintf(stderr, "atlas-import: area_cm2 not a float scalar\n"); e = 1; }
            }
            if (!e && pq_find_leaf(f, "box_min_xyz") >= 0 && pq_find_leaf(f, "box_max_xyz") >= 0
                && !col_load(f, g, "box_min_xyz", &G.bmin, &G.lbmin) && !col_load(f, g, "box_max_xyz", &G.bmax, &G.lbmax)
                && pq_leaf_at(f, G.lbmin)->type == PQ_INT32 && pq_leaf_at(f, G.lbmax)->type == PQ_INT32
                && (rec_index(&G.bmin, pq_leaf_at(f, G.lbmin)->max_def, G.nrec, &G.sbmin, &G.vbmin) || rec_index(&G.bmax, pq_leaf_at(f, G.lbmax)->max_def, G.nrec, &G.sbmax, &G.vbmax))) {
                free(G.vbmin); free(G.vbmax); G.vbmin = G.vbmax = nullptr; }
            if (e) { rg_free(&G); rc = 1; continue; }
            int64_t next = 0; const int64_t lim = limit < 0 ? -1 : limit - written;
            at_job J[64]; pthread_t th[64]; const int T = nt < G.nrec ? nt : (int)(G.nrec ? G.nrec : 1);
            for (int t = 0; t < T; t++) { J[t] = (at_job){&G, f, out, only, base, 0, G.nrec, lim, &next, &mu, idx, 0, 0, 0, 0, 0, 0, 0}; pthread_create(&th[t], nullptr, job_main, &J[t]); }
            for (int t = 0; t < T; t++) { pthread_join(th[t], nullptr); written += J[t].written; skipped += J[t].skipped; mixed += J[t].mixed; bad += J[t].bad; boxok += J[t].boxok; boxbad += J[t].boxbad; if (J[t].rc) rc = 1; }
            fprintf(stderr, "atlas-import: %s rg %d: %lld rows, %lld written so far\n", base, g, (long long)G.nrec, (long long)written);
            rg_free(&G);
        }
        pq_close(f);
    }
    if (idx && fclose(idx)) rc = 1;
    if (!dump) {
        fprintf(stderr, "atlas-import: %lld rows read, %lld written, %lld skipped (--scroll), %lld failed -> %s\n", (long long)total, (long long)written, (long long)skipped, (long long)bad, out);
        if (boxok + boxbad) fprintf(stderr, "atlas-import: box_min / box_max_xyz agree with the nodes' bbox on %lld of %lld patches\n", (long long)boxok, (long long)(boxok + boxbad));
        if (mixed) fprintf(stderr, "DEFAULTED: atlas-import: %lld nodes had -1 on some axes only: made invalid on all three\n", (long long)mixed);
    }
    return rc || bad ? 1 : 0;
}

/* ---------------------------------------------------------------------------------------------------------------------------------
   ufsm atlas-audit INDEX.tsv [--ref DIR]... [--value q.tif] [--scroll S] [--radius 20] [--on 4] [--holdouts H.json]... [--out T.tsv]
                    [--threads T] [--gate-cover 0.3] [--gate-d 2] [--gate-le 0.7] [--gate-big 0.95] [--gate-sw 0.002] [--sub 1] [--tile T]
   The imported patches (INDEX.tsv from atlas-import; the tifxyz beside it) against reference sheets, CPU only:
   - each --ref DIR is one set of meshes in one level-0 frame (tifxyz directories or .sfc wraps): GT q meshes (--value q.tif: the set's
     own winding gauge, sheets differ by whole turns) or our fitted wraps (no value file: the winding id from the name's digits, w123 -> 123).
     Labels are compared only inside one set (each GT set has its own gauge).
   - per ATLAS node (every --sub-th row / column): the nearest reference vertex within --radius voxels (any set), the distance to that
     mesh's tangent plane there (normal from the reference grid's central differences), its set and label; and the nearest vertex of the
     same set on another sheet (|label difference| >= 0.5), for the two-sheet ambiguity ("amb": both within --on voxels).
   - on-sheet nodes (plane distance <= --on): grid neighbours (right / down) in one set with |label difference| >= 0.5 are sheet switches;
     the patch's sheets are the components of the other edges, joined when their median labels agree within 0.5 (gaps in the reference
     do not split a sheet). big = the largest sheet's share of the on-sheet nodes; sheets5 = sheets holding >= 5 % of them.
   - --holdouts (repeatable): boxes {"ct", "l0": [z, y, x, nz, ny, nx]} whose CT basename equals the patch's scan; the patch's bbox
     plus 16 voxels intersecting one is listed (column hold), whatever --ref says.
   Columns: id scroll nodes matched cover p50 p90 le2 le4 onsheet amb edges switches sw_rate sheets5 big set gate hold adj (on-sheet nodes whose nearest other-sheet vertex of their set lies past --on, within --radius: an adjacent sheet).
   --label-index 1: each reference mesh is its own sheet (label = its index): a patch set against itself gives amb = overlap with another patch, adj = a traced adjacent sheet.
   gate: cover >= gate-cover and p50 <= gate-d and le4 >= gate-le and big >= gate-big and sw_rate <= gate-sw; "nocover" below the cover. */

typedef struct { float p[3], n[3]; float lab; int set; } au_v;
typedef struct { uint64_t key; uint32_t i; } au_k;
typedef struct { char id[256], scroll[128], scan[512]; double lo[3], hi[3], um; } au_patch;
typedef struct { char ct[512]; char name[64]; int64_t b[6]; } au_box;

static uint64_t au_key(int64_t x, int64_t y, int64_t z) { return (uint64_t)(z & 0x1fffff) << 42 | (uint64_t)(y & 0x1fffff) << 21 | (uint64_t)(x & 0x1fffff); }
static int au_kcmp(const void *a, const void *b) { const uint64_t x = ((const au_k *)a)->key, y = ((const au_k *)b)->key; return (x > y) - (x < y); }
static int au_fcmp(const void *a, const void *b) { const float x = *(const float *)a, y = *(const float *)b; return (x > y) - (x < y); }

typedef struct {
    au_v *v; size_t nv, cap; au_k *k; double R; int nsets;
    uint8_t *occ; int64_t oc[3]; double ocell, olo[3];   /* cells (ocell voxels) within one cell of a patch node: only their vertices are kept */
} au_ref;
static int au_occ_at(const au_ref *r, const float p[3]) {
    if (!r->occ) return 1;
    int64_t c[3]; for (int a = 0; a < 3; a++) { c[a] = (int64_t)floor((p[a] - r->olo[a]) / r->ocell); if (c[a] < 0 || c[a] >= r->oc[a]) return 0; }
    const size_t i = ((size_t)c[2] * r->oc[1] + c[1]) * r->oc[0] + c[0]; return r->occ[i >> 3] >> (i & 7) & 1;
}

static int au_label_index = 0;   /* mgpu: config (set from the command line before the audit threads start). --label-index 1: every reference mesh is its own sheet (label = its index; a patch set as its own reference) */
static int au_add_mesh(au_ref *r, const char *P, const char *value, int set, const double lo[3], const double hi[3], size_t *nmesh) {
    char p[2400]; int w = 0, h = 0, w2, h2; float *X[3] = {nullptr, nullptr, nullptr}, *Q = nullptr;
    static const char *const nm[3] = {"x.tif", "y.tif", "z.tif"};
    for (int a = 0; a < 3; a++) { snprintf(p, sizeof p, "%s/%s", P, nm[a]); X[a] = tfx_axis_read(p, a ? &w2 : &w, a ? &h2 : &h, img_read_f32); if (!X[a] || (a && (w2 != w || h2 != h))) goto bad; }
    double lab = NAN;
    if (value) { snprintf(p, sizeof p, "%s/%s", P, value); if (!access(p, F_OK)) { Q = img_read_f32(p, &w2, &h2); if (!Q || w2 != w || h2 != h) goto bad; } }
    if (au_label_index) lab = (double)*nmesh;
    else if (!Q) {   /* the winding id from the name's last run of digits */
        const char *b = strrchr(P, '/'); b = b ? b + 1 : P; const char *d = nullptr;
        for (const char *q = b; *q; q++) if (isdigit((unsigned char)*q) && (q == b || !isdigit((unsigned char)q[-1]))) d = q;
        if (!d || value) { fprintf(stderr, "atlas-audit: %s: no %s and no winding id in the name\n", P, value ? value : "label"); goto bad; }
        lab = atof(d);
    }
    for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
        const size_t i = (size_t)y * w + x;
        if (!(X[2][i] > 0 && X[0][i] >= 0 && isfinite(X[0][i]) && isfinite(X[1][i]))) continue;
        if (Q && !isfinite(Q[i])) continue;
        int out = 0; for (int a = 0; a < 3; a++) if (X[a][i] < lo[a] || X[a][i] > hi[a]) out = 1;
        if (out) continue;
        { const float pp[3] = {X[0][i], X[1][i], X[2][i]}; if (!au_occ_at(r, pp)) continue; }
        au_v V = {{X[0][i], X[1][i], X[2][i]}, {0, 0, 0}, Q ? Q[i] : (float)lab, set};
        /* normal: central differences on the grid (one-sided at holes / borders) */
        double du[3] = {0}, dv[3] = {0}; int okU = 0, okV = 0;
        const int xl = x > 0 && X[2][i - 1] > 0 ? x - 1 : x, xr = x + 1 < w && X[2][i + 1] > 0 ? x + 1 : x;
        const int yl = y > 0 && X[2][i - w] > 0 ? y - 1 : y, yr = y + 1 < h && X[2][i + w] > 0 ? y + 1 : y;
        if (xr != xl) { okU = 1; for (int a = 0; a < 3; a++) du[a] = X[a][(size_t)y * w + xr] - X[a][(size_t)y * w + xl]; }
        if (yr != yl) { okV = 1; for (int a = 0; a < 3; a++) dv[a] = X[a][(size_t)yr * w + x] - X[a][(size_t)yl * w + x]; }
        if (okU && okV) {
            double n[3] = {du[1] * dv[2] - du[2] * dv[1], du[2] * dv[0] - du[0] * dv[2], du[0] * dv[1] - du[1] * dv[0]};
            const double l = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
            if (l > 0) for (int a = 0; a < 3; a++) V.n[a] = (float)(n[a] / l);
        }
        if (r->nv == r->cap) { r->cap = r->cap ? r->cap * 2 : 1 << 20; au_v *nvv = realloc(r->v, r->cap * sizeof *nvv); if (!nvv) goto bad; r->v = nvv; }
        r->v[r->nv++] = V;
    }
    for (int a = 0; a < 3; a++) free(X[a]);
    free(Q); (*nmesh)++;
    if (tfx_release_hook) tfx_release_hook();   /* (the thread's decoded .sfc copy) */
    return 0;
bad:
    for (int a = 0; a < 3; a++) free(X[a]);
    free(Q); fprintf(stderr, "atlas-audit: cannot read the mesh %s\n", P); return -1;
}
static int au_load_set(au_ref *r, const char *dir, const char *value, int set, const double lo[3], const double hi[3]) {
    DIR *d = opendir(dir); if (!d) { fprintf(stderr, "atlas-audit: cannot open %s\n", dir); return -1; }
    struct dirent *e; size_t nm = 0; int rc = 0;
    char **names = nullptr; size_t nn = 0;
    while ((e = readdir(d))) {
        const size_t l = strlen(e->d_name);
        if (e->d_name[0] == '.') continue;
        if (!(l > 7 && !strcmp(e->d_name + l - 7, ".tifxyz")) && !(l > 4 && !strcmp(e->d_name + l - 4, ".sfc"))) {
            char p[2400]; snprintf(p, sizeof p, "%s/%s/x.tif", dir, e->d_name); if (access(p, F_OK)) continue;   /* plain wrap dirs (w123/) */
        }
        char **q = realloc(names, sizeof *names * (nn + 1)); if (!q) { rc = -1; break; } names = q; names[nn++] = strdup(e->d_name);
    }
    closedir(d);
    for (size_t k = 0; k < nn && !rc; k++) {
        char P[2400]; const size_t l = strlen(names[k]);
        if (l > 4 && !strcmp(names[k] + l - 4, ".sfc")) snprintf(P, sizeof P, "%s/%.*s", dir, (int)(l - 4), names[k]);
        else snprintf(P, sizeof P, "%s/%s", dir, names[k]);
        if (au_add_mesh(r, P, value, set, lo, hi, &nm)) rc = -1;
    }
    for (size_t k = 0; k < nn; k++) free(names[k]);
    free(names);
    fprintf(stderr, "atlas-audit: set %d %s: %zu meshes, %zu vertices so far\n", set, dir, nm, r->nv);
    return rc ? -1 : nm ? 0 : (fprintf(stderr, "atlas-audit: %s: no meshes\n", dir), -1);
}
static void au_index(au_ref *r) {
    r->k = malloc(sizeof *r->k * (r->nv + 1));
    for (size_t i = 0; i < r->nv; i++) r->k[i] = (au_k){au_key((int64_t)floor(r->v[i].p[0] / r->R), (int64_t)floor(r->v[i].p[1] / r->R), (int64_t)floor(r->v[i].p[2] / r->R)), (uint32_t)i};
    qsort(r->k, r->nv, sizeof *r->k, au_kcmp);
}
static size_t au_lower(const au_ref *r, uint64_t key) {
    size_t a = 0, b = r->nv; while (a < b) { const size_t m = (a + b) / 2; if (r->k[m].key < key) a = m + 1; else b = m; } return a;
}
/* nearest vertex within R (index or -1) and, of the same set on another sheet, the nearest one's plane distance (d2) */
static int64_t au_near(const au_ref *r, const float q[3], double *dpl, double *d2pl) {
    const int64_t c[3] = {(int64_t)floor(q[0] / r->R), (int64_t)floor(q[1] / r->R), (int64_t)floor(q[2] / r->R)};
    double best = r->R * r->R; int64_t bi = -1;
    for (int dz = -1; dz <= 1; dz++) for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
        const uint64_t key = au_key(c[0] + dx, c[1] + dy, c[2] + dz);
        for (size_t j = au_lower(r, key); j < r->nv && r->k[j].key == key; j++) {
            const au_v *v = &r->v[r->k[j].i]; const double e0 = v->p[0] - q[0], e1 = v->p[1] - q[1], e2 = v->p[2] - q[2], d = e0 * e0 + e1 * e1 + e2 * e2;
            if (d < best) { best = d; bi = r->k[j].i; }
        }
    }
    *dpl = *d2pl = INFINITY;
    if (bi < 0) return -1;
    const au_v *b = &r->v[bi];
    const double nn = fabs(b->n[0]) + fabs(b->n[1]) + fabs(b->n[2]);
    *dpl = nn > 0 ? fabs((q[0] - b->p[0]) * b->n[0] + (q[1] - b->p[1]) * b->n[1] + (q[2] - b->p[2]) * b->n[2]) : sqrt(best);
    double best2 = r->R * r->R; int64_t b2 = -1;
    for (int dz = -1; dz <= 1; dz++) for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
        const uint64_t key = au_key(c[0] + dx, c[1] + dy, c[2] + dz);
        for (size_t j = au_lower(r, key); j < r->nv && r->k[j].key == key; j++) {
            const au_v *v = &r->v[r->k[j].i]; if (v->set != b->set || fabsf(v->lab - b->lab) < 0.5f) continue;
            const double e0 = v->p[0] - q[0], e1 = v->p[1] - q[1], e2 = v->p[2] - q[2], d = e0 * e0 + e1 * e1 + e2 * e2;
            if (d < best2) { best2 = d; b2 = r->k[j].i; }
        }
    }
    if (b2 >= 0) { const au_v *v = &r->v[b2]; const double n2 = fabs(v->n[0]) + fabs(v->n[1]) + fabs(v->n[2]);
                   *d2pl = n2 > 0 ? fabs((q[0] - v->p[0]) * v->n[0] + (q[1] - v->p[1]) * v->n[1] + (q[2] - v->p[2]) * v->n[2]) : sqrt(best2); }
    return bi;
}

typedef struct { const au_ref *ref; const au_patch *P; size_t np; const char *dir; const au_box *box; size_t nbox; int sub, tile; double on;
                 double gc, gd, gle, gbig, gsw; size_t *next; pthread_mutex_t *mu; FILE *out; size_t *pass, *nocover, *fail, *held;
                 double gamb; const struct au_axis *ax; int qsign; const char *qout; char **allow; size_t nallow; size_t *dropped, *written; } au_job;
/* the umbilicus (control_points [{z, y, x}] or a bare array, level-0 voxels), linear in z, clamped: sources.c axis_load / axis_at's rule
   (kept here so atlas.o does not pull the sampler's sources.o) */
typedef struct au_axis { int n; double *z, *y, *x; } au_axis;
static int au_zcmp(const void *a, const void *b) { const double x = ((const double *)a)[0], y = ((const double *)b)[0]; return (x > y) - (x < y); }
static int au_axis_load(au_axis *a, const char *path) {
    memset(a, 0, sizeof *a);
    FILE *f = fopen(path, "rb"); if (!f) return -1;
    fseek(f, 0, SEEK_END); const long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *t = n > 0 ? malloc((size_t)n + 1) : nullptr;
    if (!t || fread(t, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(t); return -1; }
    fclose(f); t[n] = 0;
    json *j = json_parse(t, (size_t)n); free(t);
    const json *cp = json_get(j, "control_points"); if (!cp) cp = j && j->type == J_ARR ? j : nullptr;
    if (!cp || !cp->n) { json_free(j); return -1; }
    double (*p)[3] = malloc(cp->n * sizeof *p);
    for (size_t i = 0; i < cp->n; i++) { const json *e = json_at(cp, i); p[i][0] = json_num(json_get(e, "z"), 0); p[i][1] = json_num(json_get(e, "y"), 0); p[i][2] = json_num(json_get(e, "x"), 0); }
    qsort(p, cp->n, sizeof *p, au_zcmp);
    a->n = (int)cp->n; a->z = malloc(cp->n * 8); a->y = malloc(cp->n * 8); a->x = malloc(cp->n * 8);
    for (size_t i = 0; i < cp->n; i++) { a->z[i] = p[i][0]; a->y[i] = p[i][1]; a->x[i] = p[i][2]; }
    free(p); json_free(j); return 0;
}
static void au_axis_at(const au_axis *a, double z, double *y, double *x) {
    if (z <= a->z[0] || a->n == 1) { *y = a->y[0]; *x = a->x[0]; return; }
    if (z >= a->z[a->n - 1]) { *y = a->y[a->n - 1]; *x = a->x[a->n - 1]; return; }
    int lo = 0, hi = a->n - 1; while (hi - lo > 1) { const int m = (lo + hi) / 2; if (a->z[m] <= z) lo = m; else hi = m; }
    const double t = a->z[hi] > a->z[lo] ? (z - a->z[lo]) / (a->z[hi] - a->z[lo]) : 0;
    *y = a->y[lo] + t * (a->y[hi] - a->y[lo]); *x = a->x[lo] + t * (a->x[hi] - a->x[lo]);
}
static int au_scmp(const void *a, const void *b) { return strcmp(*(char *const *)a, *(char *const *)b); }

static int uf_find(int *p, int a) { while (p[a] != a) { p[a] = p[p[a]]; a = p[a]; } return a; }

static void au_stats(au_job *J, const au_patch *P, const char *rid, float *X[3], int w, int h, const char *hold);
static void au_patch_one(au_job *J, const au_patch *P) {
    char d[2400], p[2600]; int w = 0, h = 0, w2 = -1, h2 = -1; float *X[3] = {nullptr, nullptr, nullptr};
    snprintf(d, sizeof d, "%s/%s/%s.tifxyz", J->dir, P->scroll, P->id);
    static const char *const nm[3] = {"x.tif", "y.tif", "z.tif"};
    for (int a = 0; a < 3; a++) { snprintf(p, sizeof p, "%s/%s", d, nm[a]); X[a] = img_read_f32(p, a ? &w2 : &w, a ? &h2 : &h); }
    if (!X[0] || !X[1] || !X[2] || w2 != w || h2 != h) { fprintf(stderr, "atlas-audit: cannot read %s\n", d); for (int a = 0; a < 3; a++) free(X[a]); return; }
    char hold[512] = "";
    for (size_t b = 0; b < J->nbox; b++) {
        const au_box *B = &J->box[b]; if (strcmp(B->ct, P->scan)) continue;
        const int64_t bz = B->b[0], by = B->b[1], bx = B->b[2], nz = B->b[3], ny = B->b[4], nx = B->b[5]; const double g = 16;
        if (P->hi[0] + g < bx || P->lo[0] - g > bx + nx || P->hi[1] + g < by || P->lo[1] - g > by + ny || P->hi[2] + g < bz || P->lo[2] - g > bz + nz) continue;
        if (strstr(hold, B->name)) continue;   /* (one name once: the holdout files overlap) */
        const size_t l = strlen(hold); if (l + strlen(B->name) + 2 < sizeof hold) snprintf(hold + l, sizeof hold - l, "%s%s", l ? "," : "", B->name);
    }
    if (J->tile <= 0) au_stats(J, P, P->id, X, w, h, hold);
    else {   /* --tile T: T x T-node tiles of the patch, one row each (id#row,col; a calibration surface at the ATLAS patch scale) */
        const int T = J->tile; float *S[3] = {malloc(sizeof(float) * T * T), malloc(sizeof(float) * T * T), malloc(sizeof(float) * T * T)};
        for (int r0 = 0; S[0] && S[1] && S[2] && r0 < h; r0 += T) for (int c0 = 0; c0 < w; c0 += T) {
            const int th = h - r0 < T ? h - r0 : T, tw = w - c0 < T ? w - c0 : T; size_t nv = 0;
            for (int a = 0; a < 3; a++) for (int r = 0; r < th; r++) for (int c = 0; c < tw; c++) { S[a][r * tw + c] = X[a][(size_t)(r0 + r) * w + c0 + c]; nv += a == 2 && S[2][r * tw + c] >= 0; }
            if (nv * 4 < (size_t)T * T) continue;   /* (tiles at least a quarter valid) */
            char rid[300]; snprintf(rid, sizeof rid, "%s#%d,%d", P->id, r0 / T, c0 / T);
            au_stats(J, P, rid, S, tw, th, hold);
        }
        for (int a = 0; a < 3; a++) free(S[a]);
    }
    for (int a = 0; a < 3; a++) free(X[a]);
}
static void au_stats(au_job *J, const au_patch *P, const char *rid, float *X[3], int w, int h, const char *hold) {
    double qK = NAN; float *theta_u = J->qout ? malloc(sizeof(float) * ((size_t)w * h + 1)) : nullptr;
    size_t adj = 0, nodes = 0, matched = 0, onsheet = 0, amb = 0, le2 = 0, le4 = 0, edges = 0, sw = 0, sheets5 = 0; double p50 = NAN, p90 = NAN, big = NAN; int dset = -1;
    if (J->ref) {
        const int S = J->sub, gw = (w + S - 1) / S, gh = (h + S - 1) / S; const size_t G = (size_t)gw * gh;
        float *dpl = malloc(G * 4), *lab = malloc(G * 4); int *set = malloc(G * sizeof(int)), *uf = malloc(G * sizeof(int));
        float *ds = malloc(G * 4); size_t nd = 0;
        if (!dpl || !lab || !set || !uf || !ds) { free(dpl); free(lab); free(set); free(uf); free(ds); fprintf(stderr, "atlas-audit: out of memory\n"); return; }
        for (int gy = 0; gy < gh; gy++) for (int gx = 0; gx < gw; gx++) {
            const size_t g = (size_t)gy * gw + gx, i = (size_t)gy * S * w + (size_t)gx * S; set[g] = -1; uf[g] = (int)g; dpl[g] = INFINITY;
            if (X[2][i] < 0) continue;
            nodes++;
            const float q[3] = {X[0][i], X[1][i], X[2][i]}; double d1, d2; const int64_t bi = au_near(J->ref, q, &d1, &d2);
            if (bi < 0) continue;
            matched++; ds[nd++] = (float)d1; le2 += d1 <= 2; le4 += d1 <= 4;
            if (d1 <= J->on) { onsheet++; set[g] = J->ref->v[bi].set; lab[g] = J->ref->v[bi].lab; dpl[g] = (float)d1; amb += d2 <= J->on; adj += d2 > J->on && isfinite(d2); }
        }
        if (nd) { qsort(ds, nd, 4, au_fcmp); p50 = ds[nd / 2]; p90 = ds[(size_t)(0.9 * (nd - 1))]; }
        for (int gy = 0; gy < gh; gy++) for (int gx = 0; gx < gw; gx++) {
            const size_t g = (size_t)gy * gw + gx; if (set[g] < 0) continue;
            const size_t nb[2] = {gx + 1 < gw ? g + 1 : (size_t)-1, gy + 1 < gh ? g + gw : (size_t)-1};
            for (int k = 0; k < 2; k++) {
                const size_t o = nb[k]; if (o == (size_t)-1 || set[o] != set[g]) continue;
                edges++;
                if (fabsf(lab[o] - lab[g]) >= 0.5f) sw++;
                else { const int a = uf_find(uf, (int)g), b = uf_find(uf, (int)o); if (a != b) uf[a] = b; }
            }
        }
        /* components -> (set, median label, count); joined when one set's medians agree within 0.5 */
        size_t nc = 0; int *root = malloc(G * sizeof(int)); float *cl = nullptr; int *cs = nullptr; size_t *cn = nullptr;
        if (root) {
            for (size_t g = 0; g < G; g++) root[g] = -1;
            for (size_t g = 0; g < G; g++) if (set[g] >= 0) { const int a = uf_find(uf, (int)g); if (root[a] < 0) root[a] = (int)nc++; }
            cl = calloc(nc + 1, 4); cs = calloc(nc + 1, sizeof(int)); cn = calloc(nc + 1, sizeof(size_t)); double *csum = calloc(nc + 1, sizeof(double));
            if (cl && cs && cn && csum) {
                for (size_t g = 0; g < G; g++) if (set[g] >= 0) { const int c = root[uf_find(uf, (int)g)]; cn[c]++; csum[c] += lab[g]; cs[c] = set[g]; }
                for (size_t c = 0; c < nc; c++) cl[c] = (float)(csum[c] / cn[c]);   /* (the mean: one sheet's labels vary < 0.5 inside a patch) */
                /* group: sort components by (set, label) and join neighbours within 0.5 */
                size_t *ord = malloc(sizeof(size_t) * (nc + 1));
                if (ord) {
                    for (size_t c = 0; c < nc; c++) ord[c] = c;
                    for (size_t a = 1; a < nc; a++) { const size_t t = ord[a]; size_t b = a; while (b > 0 && (cs[ord[b - 1]] > cs[t] || (cs[ord[b - 1]] == cs[t] && cl[ord[b - 1]] > cl[t]))) { ord[b] = ord[b - 1]; b--; } ord[b] = t; }
                    size_t best = 0, cur = 0; size_t bestset = 0; size_t *setn = calloc((size_t)J->ref->nsets + 1, sizeof(size_t));
                    for (size_t a = 0; a < nc; a++) {
                        const size_t c = ord[a];
                        if (a && cs[ord[a - 1]] == cs[c] && cl[c] - cl[ord[a - 1]] < 0.5f) cur += cn[c];
                        else { if (a && cur * 20 >= onsheet) sheets5++; cur = cn[c]; }
                        if (cur > best) { best = cur; bestset = (size_t)cs[c]; }
                        if (setn) setn[cs[c]] += cn[c];
                    }
                    if (nc && cur * 20 >= onsheet) sheets5++;
                    big = onsheet ? (double)best / onsheet : NAN; dset = nc ? (int)bestset : -1;
                    free(setn); free(ord);
                }
            }
            free(csum);
        }
        if (J->qout && J->ax && dset >= 0 && S == 1) {   /* the patch's winding offset: K = median over its main set's on-sheet nodes of label - sign theta_u / 2 pi */
            double sc = 0, cc = 0;
            for (size_t i = 0; i < G; i++) if (X[2][i] >= 0) { double cy, cx; au_axis_at(J->ax, X[2][i], &cy, &cx); const double t = atan2(X[1][i] - cy, X[0][i] - cx); sc += sin(t); cc += cos(t); }
            const double tc = atan2(sc, cc); size_t nk = 0;
            for (size_t i = 0; i < G; i++) {
                double cy, cx, t = 0; if (X[2][i] >= 0) { au_axis_at(J->ax, X[2][i], &cy, &cx); t = atan2(X[1][i] - cy, X[0][i] - cx) - tc; t -= 2 * M_PI * floor((t + M_PI) / (2 * M_PI)); t += tc; }
                theta_u[i] = (float)t;
                if (set[i] == dset) ds[nk++] = (float)(lab[i] - J->qsign * t / (2 * M_PI));
            }
            if (nk) { qsort(ds, nk, 4, au_fcmp); qK = ds[nk / 2]; }
        }
        free(root); free(cl); free(cs); free(cn); free(dpl); free(lab); free(set); free(uf); free(ds);
    }
    const double cover = nodes ? (double)matched / nodes : 0, swr = edges ? (double)sw / edges : NAN;
    int allowed = 1;
    if (J->allow) { const char *key = P->id; allowed = bsearch(&key, J->allow, J->nallow, sizeof *J->allow, au_scmp) != nullptr; }
    const char *gate = !J->ref ? "-" : *hold ? "held" : !allowed ? "dropped" : cover < J->gc ? "nocover"
                     : (p50 <= J->gd && (double)le4 / (matched ? matched : 1) >= J->gle && big >= J->gbig && swr <= J->gsw
                        && (onsheet ? (double)amb / onsheet : 1) <= J->gamb) ? "pass" : "fail";
    int wrote = 0;
    if (J->qout && !strcmp(gate, "pass") && isfinite(qK) && theta_u) {   /* the kept patch with its q.tif (turns, the reference's gauge) for `meshes` onfly sources */
        const size_t n = (size_t)w * h; float *Q = malloc(n * 4); char d[2600], qp[2700];
        if (Q) {
            for (size_t i = 0; i < n; i++) Q[i] = X[2][i] >= 0 ? (float)(qK + J->qsign * theta_u[i] / (2 * M_PI)) : NAN;
            snprintf(d, sizeof d, "%s/%s.tifxyz", J->qout, P->id); snprintf(qp, sizeof qp, "%s/q.tif", d);
            char extra[1200]; snprintf(extra, sizeof extra, "\"scroll\": \"%s\", \"scan\": \"%s\", \"atlas_id\": \"%s\", \"dataset\": \"ATLAS-15k\", \"license\": \"CC BY-NC 4.0\", \"q_K\": %.4f", P->scroll, P->scan, P->id, qK);
            wrote = !tifxyz_write_ex(d, w, h, X[0], X[1], X[2], 2.0, P->um, -1, "ufsm atlas-audit --write-q (ATLAS-15k patch, q = K + sign theta / 2 pi in the reference gauge)", extra)
                    && !tiffw_write(qp, w, h, 3, Q, 1);
            if (!wrote) fprintf(stderr, "atlas-audit: cannot write %s\n", d);
            free(Q);
        }
    }
    free(theta_u);
    pthread_mutex_lock(J->mu);
    fprintf(J->out, "%s\t%s\t%zu\t%zu\t%.4f\t%.3f\t%.3f\t%.4f\t%.4f\t%zu\t%zu\t%zu\t%zu\t%.5f\t%zu\t%.4f\t%d\t%s\t%s\t%zu\n", rid, P->scroll, nodes, matched, cover,
            p50, p90, matched ? (double)le2 / matched : 0, matched ? (double)le4 / matched : 0, onsheet, amb, edges, sw, swr, sheets5, big, dset, gate, *hold ? hold : "-", adj);
    if (!strcmp(gate, "pass")) (*J->pass)++; else if (!strcmp(gate, "nocover")) (*J->nocover)++; else if (!strcmp(gate, "fail")) (*J->fail)++;
    else if (!strcmp(gate, "dropped")) (*J->dropped)++;
    *J->written += wrote;
    if (*hold) (*J->held)++;
    pthread_mutex_unlock(J->mu);
}
static void *au_main(void *a) {
    au_job *J = a;
    for (;;) { pthread_mutex_lock(J->mu); const size_t k = *J->next < J->np ? (*J->next)++ : (size_t)-1; pthread_mutex_unlock(J->mu); if (k == (size_t)-1) break; au_patch_one(J, &J->P[k]); }
    return nullptr;
}

static int au_holdouts(const char *path, au_box **box, size_t *nbox) {
    FILE *f = fopen(path, "rb"); if (!f) { fprintf(stderr, "atlas-audit: cannot read %s\n", path); return -1; }
    fseek(f, 0, SEEK_END); const long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *t = malloc((size_t)n + 1); if (!t || fread(t, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(t); return -1; }
    fclose(f); t[n] = 0;
    json *j = json_parse(t, (size_t)n); free(t);
    if (!j) { fprintf(stderr, "atlas-audit: %s: bad json\n", path); return -1; }
    const json *b = json_get(j, "boxes"); if (!b) b = j;
    for (size_t i = 0; b && b->type == J_ARR && i < b->n; i++) {
        const json *e = json_at(b, i), *ct = json_get(e, "ct"), *l0 = json_get(e, "l0"), *nm = json_get(e, "name");
        if (!ct || !l0 || l0->type != J_ARR || l0->n != 6) continue;
        au_box *q = realloc(*box, sizeof *q * (*nbox + 1)); if (!q) { json_free(j); return -1; } *box = q;
        au_box *B = &(*box)[(*nbox)++]; const char *c = json_str(ct, ""), *s = strrchr(c, '/');
        snprintf(B->ct, sizeof B->ct, "%s", s ? s + 1 : c); snprintf(B->name, sizeof B->name, "%s", json_str(nm, "?"));
        for (int k = 0; k < 6; k++) B->b[k] = (int64_t)json_num(json_at(l0, k), 0);
    }
    json_free(j); return 0;
}

int cmd_atlas_audit(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: ufsm atlas-audit INDEX.tsv [--ref DIR]... [--value q.tif] [--scroll S] [--radius 20] [--on 4] [--holdouts H.json]... [--out T.tsv] [--sub 1]\n"); return 2; }
    const char *index = argv[2], *only = at_opt(argc, argv, "--scroll", nullptr), *value = at_opt(argc, argv, "--value", nullptr), *outp = at_opt(argc, argv, "--out", nullptr);
    const double R = atof(at_opt(argc, argv, "--radius", "20")), on = atof(at_opt(argc, argv, "--on", "4"));
    int nt = atoi(at_opt(argc, argv, "--threads", "0")); if (nt <= 0) nt = ufsm_ncpu(); if (nt > 64) nt = 64;
    au_label_index = atoi(at_opt(argc, argv, "--label-index", "0"));
    const int sub = atoi(at_opt(argc, argv, "--sub", "1")), tile = atoi(at_opt(argc, argv, "--tile", "0"));
    if (!(R > 0) || !(on > 0) || sub < 1) { fprintf(stderr, "atlas-audit: --radius, --on > 0, --sub >= 1\n"); return 2; }
    char dir[2048]; snprintf(dir, sizeof dir, "%s", index); char *sl = strrchr(dir, '/'); if (sl) *sl = 0; else snprintf(dir, sizeof dir, ".");
    FILE *f = fopen(index, "r"); if (!f) { fprintf(stderr, "atlas-audit: cannot read %s\n", index); return 1; }
    au_patch *P = nullptr; size_t np = 0; char line[4096];
    double lo[3] = {1e30, 1e30, 1e30}, hi[3] = {-1e30, -1e30, -1e30};
    while (fgets(line, sizeof line, f)) {
        if (!strncmp(line, "id\t", 3)) continue;
        au_patch Q = {0}; char *tok[16]; int nt2 = 0;
        for (char *s = strtok(line, "\t\n"); s && nt2 < 16; s = strtok(nullptr, "\t\n")) tok[nt2++] = s;
        if (nt2 < 16) continue;
        if (only && strcmp(tok[1], only)) continue;
        snprintf(Q.id, sizeof Q.id, "%s", tok[0]); snprintf(Q.scroll, sizeof Q.scroll, "%s", tok[1]); snprintf(Q.scan, sizeof Q.scan, "%s", tok[2]);
        Q.um = atof(tok[3]);
        for (int k = 0; k < 3; k++) { Q.lo[k] = atof(tok[10 + k]); Q.hi[k] = atof(tok[13 + k]); lo[k] = fmin(lo[k], Q.lo[k]); hi[k] = fmax(hi[k], Q.hi[k]); }
        au_patch *q = realloc(P, sizeof *q * (np + 1)); if (!q) { fclose(f); return 1; } P = q; P[np++] = Q;
    }
    fclose(f);
    if (!np) { fprintf(stderr, "atlas-audit: no patches%s%s in %s\n", only ? " of " : "", only ? only : "", index); free(P); return 1; }
    au_box *box = nullptr; size_t nbox = 0;
    for (int a = 3; a + 1 < argc; a++) if (!strcmp(argv[a], opt_n("--holdouts")) && au_holdouts(argv[a + 1], &box, &nbox)) return 1;
    au_ref ref = {.R = R}; int haveref = 0;
    for (int a = 3; a + 1 < argc; a++) if (!strcmp(argv[a], opt_n("--ref"))) haveref = 1;
    if (haveref) {   /* occupancy of the patches' nodes (cells of max(R, 16) voxels, dilated by one cell): whole-scroll wraps keep only nearby vertices */
        ref.ocell = fmax(R, 16);
        for (int k = 0; k < 3; k++) { ref.olo[k] = lo[k] - 2 * ref.ocell; ref.oc[k] = (int64_t)ceil((hi[k] - lo[k] + 4 * ref.ocell) / ref.ocell) + 1; }
        const size_t ncell = (size_t)ref.oc[0] * ref.oc[1] * ref.oc[2];
        uint8_t *o = calloc(ncell / 8 + 1, 1); ref.occ = calloc(ncell / 8 + 1, 1);
        if (!o || !ref.occ) { fprintf(stderr, "atlas-audit: out of memory (occupancy)\n"); return 1; }
        for (size_t k = 0; k < np; k++) {
            char d[2600]; float *X[3]; int w = 0, h = 0, w2, h2; static const char *const nm[3] = {"x.tif", "y.tif", "z.tif"};
            for (int a = 0; a < 3; a++) { snprintf(d, sizeof d, "%s/%s/%s.tifxyz/%s", dir, P[k].scroll, P[k].id, nm[a]); X[a] = img_read_f32(d, a ? &w2 : &w, a ? &h2 : &h); }
            if (X[0] && X[1] && X[2] && w2 == w && h2 == h)
                for (size_t i = 0; i < (size_t)w * h; i++) {
                    if (X[2][i] < 0) continue;
                    int64_t c[3]; for (int a = 0; a < 3; a++) c[a] = (int64_t)floor((X[a][i] - ref.olo[a]) / ref.ocell);
                    const size_t q = ((size_t)c[2] * ref.oc[1] + c[1]) * ref.oc[0] + c[0]; o[q >> 3] |= (uint8_t)(1 << (q & 7));
                }
            for (int a = 0; a < 3; a++) free(X[a]);
        }
        for (int64_t z = 1; z + 1 < ref.oc[2]; z++) for (int64_t y = 1; y + 1 < ref.oc[1]; y++) for (int64_t x = 1; x + 1 < ref.oc[0]; x++) {
            const size_t q = ((size_t)z * ref.oc[1] + y) * ref.oc[0] + x; if (!(o[q >> 3] >> (q & 7) & 1)) continue;
            for (int dz = -1; dz <= 1; dz++) for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
                const size_t t = ((size_t)(z + dz) * ref.oc[1] + (y + dy)) * ref.oc[0] + (x + dx); ref.occ[t >> 3] |= (uint8_t)(1 << (t & 7)); }
        }
        free(o); haveref = 0;
    }
    const double lo2[3] = {lo[0] - R, lo[1] - R, lo[2] - R}, hi2[3] = {hi[0] + R, hi[1] + R, hi[2] + R};
    for (int a = 3; a + 1 < argc; a++) if (!strcmp(argv[a], opt_n("--ref"))) { if (au_load_set(&ref, argv[a + 1], value, ref.nsets, lo2, hi2)) return 1; ref.nsets++; haveref = 1; }
    if (haveref) { if (ref.nv > 0xffffffffu) { fprintf(stderr, "atlas-audit: too many reference vertices\n"); return 1; } au_index(&ref); }
    FILE *out = outp ? fopen(outp, "w") : stdout; if (!out) { fprintf(stderr, "atlas-audit: cannot write %s\n", outp); return 1; }
    fprintf(out, "id\tscroll\tnodes\tmatched\tcover\tp50\tp90\tle2\tle4\tonsheet\tamb\tedges\tswitches\tsw_rate\tsheets5\tbig\tset\tgate\thold\tadj\n");
    pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER; size_t next = 0, pass = 0, nocover = 0, failn = 0, held = 0, dropped = 0, written = 0;
    const char *qout = at_opt(argc, argv, "--write-q", nullptr), *axf = at_opt(argc, argv, "--axis", nullptr), *allowf = at_opt(argc, argv, "--allow", nullptr);
    au_axis ax = {0};
    if (qout) {
        if (!axf || au_axis_load(&ax, axf)) { fprintf(stderr, "atlas-audit: --write-q needs --axis (the scroll's umbilicus json; %s)\n", axf ? "unreadable" : "missing"); return 2; }
        if (sub != 1 || tile) { fprintf(stderr, "atlas-audit: --write-q needs --sub 1 and no --tile\n"); return 2; }
        if (mkdir(qout, 0775) && errno != EEXIST) { fprintf(stderr, "atlas-audit: cannot create %s\n", qout); return 1; }
    }
    char **allow = nullptr; size_t nallow = 0;
    if (allowf) {   /* --allow FILE: patch ids (first column) that may pass at all (other gates applied beforehand, e.g. the CT alignment) */
        FILE *af = fopen(allowf, "r"); if (!af) { fprintf(stderr, "atlas-audit: cannot read %s\n", allowf); return 1; }
        char ln[1024]; while (fgets(ln, sizeof ln, af)) { char *t = strtok(ln, "\t\n "); if (!t || !*t) continue; char **q = realloc(allow, sizeof *allow * (nallow + 1)); if (!q) return 1; allow = q; allow[nallow++] = strdup(t); }
        fclose(af); qsort(allow, nallow, sizeof *allow, au_scmp);
    }
    au_job J = {haveref ? &ref : nullptr, P, np, dir, box, nbox, sub, tile, on, atof(at_opt(argc, argv, "--gate-cover", "0.3")), atof(at_opt(argc, argv, "--gate-d", "2")),
                atof(at_opt(argc, argv, "--gate-le", "0.7")), atof(at_opt(argc, argv, "--gate-big", "0.95")), atof(at_opt(argc, argv, "--gate-sw", "0.002")),
                &next, &mu, out, &pass, &nocover, &failn, &held, atof(at_opt(argc, argv, "--gate-amb", "1")), qout ? &ax : nullptr,
                atoi(at_opt(argc, argv, "--q-sign", "1")) < 0 ? -1 : 1, qout, allow, nallow, &dropped, &written};
    pthread_t th[64]; const int T = (size_t)nt < np ? nt : (int)np;
    for (int t = 0; t < T; t++) pthread_create(&th[t], nullptr, au_main, &J);
    for (int t = 0; t < T; t++) pthread_join(th[t], nullptr);
    if (outp && fclose(out)) return 1;
    fprintf(stderr, "atlas-audit: %zu patches%s%s, %zu reference vertices in %d sets: pass %zu, fail %zu, nocover %zu, held (holdout box) %zu, dropped (--allow) %zu%s",
            np, only ? " of " : "", only ? only : "", ref.nv, ref.nsets, pass, failn, nocover, held, dropped, qout ? "" : "\n");
    if (qout) fprintf(stderr, "; %zu written with q.tif -> %s\n", written, qout);
    free(ax.z); free(ax.y); free(ax.x);
    for (size_t k = 0; k < nallow; k++) free(allow[k]);
    free(allow);
    free(ref.v); free(ref.k); free(ref.occ); free(P); free(box);
    return 0;
}

/* ---------------------------------------------------------------------------------------------------------------------------------
   ufsm atlas-rvscore --rv ROOT KEY ...: a predicted recto / verso volume (predict --channel rv: 1 + 254 (1 + p_recto - p_verso) / 2, 0 = CT 0)
   scored on the CPU (2026-11-27, atlas: the A/B score while scripts/rv/eval.sh's count fields are gone). Two modes, one or both:
   - GT box: --box z,y,x,nz,ny,nx (voxels of the rv array) --phase ROOT KEY [--phase-div S] --ct ROOT KEY [--iso 64]: over the box's papyrus
     (CT >= iso) whose GT phase is known (PHASE_* codes, phase voxel = S rv voxels), GT recto = phase < 126; predicted recto = rv > 128.
     Prints recto recall, verso recall and their mean (balanced side accuracy).
   - ATLAS patches: --patches INDEX.tsv --ids FILE --axis AXIS.json [--rv-div D] [--d 2] [--every 2]: the patches named in FILE (first
     column; e.g. those inside an eval slab, never trained), every --every-th node: the grid normal oriented away from the axis, rv sampled at
     the node +- d voxels along it (rv voxel = D level-0 voxels). The recto face's outer side should read recto (> 128), its inner side not.
     Prints the node shares: outer recto, inner verso, both, and the mean outer - inner rv. Nodes outside the rv array or on CT 0 (rv 0)
     are skipped and counted. */
typedef struct { z3 *z; int64_t o[3], n[3]; uint8_t *buf; } rs_vol;
static int rs_load(rs_vol *v, const char *root, const char *key, const int64_t o[3], const int64_t n[3]) {
    store *st = store_open(root); v->z = st ? z3_open(st, key, nullptr) : nullptr;
    if (!v->z) { fprintf(stderr, "atlas-rvscore: cannot open %s %s: %s\n", root, key, z3_error()); return -1; }
    const z3_meta *m = z3_meta_of(v->z); int64_t oo[3], nn[3];
    for (int k = 0; k < 3; k++) { oo[k] = o[k] < 0 ? 0 : o[k]; const int64_t e = o[k] + n[k] > m->shape[k] ? m->shape[k] : o[k] + n[k]; nn[k] = e - oo[k]; if (nn[k] <= 0) { fprintf(stderr, "atlas-rvscore: box outside %s\n", key); return -1; } v->o[k] = oo[k]; v->n[k] = nn[k]; }
    v->buf = malloc((size_t)nn[0] * nn[1] * nn[2]);
    if (!v->buf || z3_read(v->z, oo, nn, v->buf, ufsm_ncpu())) { fprintf(stderr, "atlas-rvscore: cannot read %s: %s\n", key, z3_error()); return -1; }
    return 0;
}
static int rs_at(const rs_vol *v, int64_t z, int64_t y, int64_t x) {
    z -= v->o[0]; y -= v->o[1]; x -= v->o[2];
    if (z < 0 || y < 0 || x < 0 || z >= v->n[0] || y >= v->n[1] || x >= v->n[2]) return -1;
    return v->buf[((size_t)z * v->n[1] + y) * v->n[2] + x];
}
static void rs_free(rs_vol *v) { free(v->buf); if (v->z) z3_close(v->z); memset(v, 0, sizeof *v); }

int cmd_atlas_rvscore(int argc, char **argv) {
    const char *rvr = nullptr, *rvk = nullptr;
    for (int i = 2; i + 2 < argc; i++) if (!strcmp(argv[i], opt_n("--rv"))) { rvr = argv[i + 1]; rvk = argv[i + 2]; }
    if (!rvr) { fprintf(stderr, "usage: ufsm atlas-rvscore --rv ROOT KEY [--box z,y,x,nz,ny,nx --phase ROOT KEY [--phase-div S] --ct ROOT KEY [--iso 64]]\n"
                                "       [--patches INDEX.tsv --ids FILE --axis AXIS.json [--rv-div D] [--d 2] [--every 2]] [--tag T] [--out F]\n"); return 2; }
    const char *tag = at_opt(argc, argv, "--tag", "-"), *boxs = at_opt(argc, argv, "--box", nullptr), *idx = at_opt(argc, argv, "--patches", nullptr);
    int rc = 0, did = 0;
    const char *outp = at_opt(argc, argv, "--out", nullptr); FILE *of = outp ? fopen(outp, "a") : stdout;   /* --out F: rows appended to F */
    if (!of) { fprintf(stderr, "atlas-rvscore: cannot write %s\n", outp); return 1; }
    if (boxs) {
        int64_t b[6]; if (sscanf(boxs, "%" SCNd64 ",%" SCNd64 ",%" SCNd64 ",%" SCNd64 ",%" SCNd64 ",%" SCNd64, &b[0], &b[1], &b[2], &b[3], &b[4], &b[5]) != 6) { fprintf(stderr, "atlas-rvscore: --box z,y,x,nz,ny,nx\n"); return 2; }
        const char *pr = nullptr, *pk = nullptr, *cr = nullptr, *ck = nullptr;
        for (int i = 2; i + 2 < argc; i++) { if (!strcmp(argv[i], opt_n("--phase"))) { pr = argv[i + 1]; pk = argv[i + 2]; } if (!strcmp(argv[i], opt_n("--ct"))) { cr = argv[i + 1]; ck = argv[i + 2]; } }
        if (!pr || !cr) { fprintf(stderr, "atlas-rvscore: --box needs --phase ROOT KEY and --ct ROOT KEY\n"); return 2; }
        const int S = atoi(at_opt(argc, argv, "--phase-div", "1")), iso = atoi(at_opt(argc, argv, "--iso", "64"));
        if (S < 1) { fprintf(stderr, "atlas-rvscore: --phase-div >= 1\n"); return 2; }
        rs_vol rv = {0}, ph = {0}, ct = {0};
        const int64_t po[3] = {b[0] / S, b[1] / S, b[2] / S}, pn[3] = {(b[0] + b[3] + S - 1) / S - b[0] / S, (b[1] + b[4] + S - 1) / S - b[1] / S, (b[2] + b[5] + S - 1) / S - b[2] / S};
        if (rs_load(&rv, rvr, rvk, b, b + 3) || rs_load(&ph, pr, pk, po, pn) || rs_load(&ct, cr, ck, b, b + 3)) { rs_free(&rv); rs_free(&ph); rs_free(&ct); return 1; }
        uint64_t rr = 0, rn = 0, vv = 0, vn = 0, skip = 0;
        for (int64_t z = b[0]; z < b[0] + b[3]; z++) for (int64_t y = b[1]; y < b[1] + b[4]; y++) for (int64_t x = b[2]; x < b[2] + b[5]; x++) {
            const int c = rs_at(&ct, z, y, x), p = rs_at(&ph, z / S, y / S, x / S), r = rs_at(&rv, z, y, x);
            if (c < iso || p < 0 || p == 255) continue;
            if (r <= 0) { skip++; continue; }
            if (p < 126) { rn++; rr += r > 128; } else { vn++; vv += r <= 128; }
        }
        const double ra = rn ? (double)rr / rn : NAN, va = vn ? (double)vv / vn : NAN;
        fprintf(of, "atlas-rvscore\t%s\tbox\t%s\trecto_recall\t%.4f\tverso_recall\t%.4f\tbalanced\t%.4f\tvoxels\t%" PRIu64 "\tskipped\t%" PRIu64 "\n", tag, boxs, ra, va, (ra + va) / 2, rn + vn, skip);
        if (!(rn + vn)) rc = 1;
        rs_free(&rv); rs_free(&ph); rs_free(&ct); did = 1;
    }
    if (idx) {
        const char *idsf = at_opt(argc, argv, "--ids", nullptr), *axf = at_opt(argc, argv, "--axis", nullptr);
        const int D = atoi(at_opt(argc, argv, "--rv-div", "1")), ev = atoi(at_opt(argc, argv, "--every", "2")); const double dd = atof(at_opt(argc, argv, "--d", "2"));
        au_axis ax; if (!idsf || !axf || au_axis_load(&ax, axf) || D < 1 || ev < 1 || !(dd > 0)) { fprintf(stderr, "atlas-rvscore: --patches needs --ids FILE and a readable --axis; --rv-div, --every >= 1, --d > 0\n"); return 2; }
        char dir[2048]; snprintf(dir, sizeof dir, "%s", idx); char *sl = strrchr(dir, '/'); if (sl) *sl = 0; else snprintf(dir, sizeof dir, ".");
        FILE *f = fopen(idsf, "r"); if (!f) { fprintf(stderr, "atlas-rvscore: cannot read %s\n", idsf); return 1; }
        char **ids = nullptr; size_t ni = 0; char ln[4096];
        while (fgets(ln, sizeof ln, f)) { char *t = strtok(ln, "\t\n "); if (!t) continue; ids = realloc(ids, sizeof *ids * (ni + 1)); ids[ni++] = strdup(t); }
        fclose(f); qsort(ids, ni, sizeof *ids, au_scmp);
        f = fopen(idx, "r"); if (!f) { fprintf(stderr, "atlas-rvscore: cannot read %s\n", idx); return 1; }
        uint64_t nodes = 0, outr = 0, inv = 0, both = 0, skipped = 0; double dsum = 0; size_t np = 0;
        while (fgets(ln, sizeof ln, f)) {
            if (!strncmp(ln, "id\t", 3)) continue;
            char *tok[16]; int nt = 0; for (char *s = strtok(ln, "\t\n"); s && nt < 16; s = strtok(nullptr, "\t\n")) tok[nt++] = s;
            if (nt < 16) continue;
            const char *key = tok[0]; if (!bsearch(&key, ids, ni, sizeof *ids, au_scmp)) continue;
            char d[2600]; float *X[3]; int w = 0, h = 0, w2, h2; static const char *const nm[3] = {"x.tif", "y.tif", "z.tif"};
            for (int a = 0; a < 3; a++) { snprintf(d, sizeof d, "%s/%s/%s.tifxyz/%s", dir, tok[1], tok[0], nm[a]); X[a] = img_read_f32(d, a ? &w2 : &w, a ? &h2 : &h); }
            if (!X[0] || !X[1] || !X[2] || w2 != w || h2 != h) { fprintf(stderr, "atlas-rvscore: cannot read %s\n", tok[0]); for (int a = 0; a < 3; a++) free(X[a]); rc = 1; continue; }
            /* the rv block over this patch's bbox + d (rv grid) */
            int64_t o[3], n[3]; const double lo[3] = {atof(tok[12]), atof(tok[11]), atof(tok[10])}, hi[3] = {atof(tok[15]), atof(tok[14]), atof(tok[13])};   /* z y x */
            for (int k = 0; k < 3; k++) { o[k] = (int64_t)floor((lo[k] - dd - 2) / D); n[k] = (int64_t)ceil((hi[k] + dd + 2) / D) - o[k] + 1; }
            rs_vol rv = {0};
            if (rs_load(&rv, rvr, rvk, o, n)) { for (int a = 0; a < 3; a++) free(X[a]); rs_free(&rv); skipped++; continue; }
            np++;
            for (int r = 1; r + 1 < h; r += ev) for (int c = 1; c + 1 < w; c += ev) {
                const size_t i = (size_t)r * w + c;
                if (X[2][i] < 0 || X[2][i - 1] < 0 || X[2][i + 1] < 0 || X[2][i - w] < 0 || X[2][i + w] < 0) continue;
                double du[3], dv[3], nr[3];
                for (int a = 0; a < 3; a++) { du[a] = X[a][i + 1] - X[a][i - 1]; dv[a] = X[a][i + w] - X[a][i - w]; }
                nr[0] = du[1] * dv[2] - du[2] * dv[1]; nr[1] = du[2] * dv[0] - du[0] * dv[2]; nr[2] = du[0] * dv[1] - du[1] * dv[0];
                const double l = sqrt(nr[0] * nr[0] + nr[1] * nr[1] + nr[2] * nr[2]); if (!(l > 0)) continue;
                double cy, cx; au_axis_at(&ax, X[2][i], &cy, &cx);
                if (nr[0] * (X[0][i] - cx) + nr[1] * (X[1][i] - cy) < 0) for (int a = 0; a < 3; a++) nr[a] = -nr[a];   /* outward: away from the axis */
                int s[2];
                for (int k = 0; k < 2; k++) {
                    const double sg = k ? -dd : dd, px = X[0][i] + sg * nr[0] / l, py = X[1][i] + sg * nr[1] / l, pz = X[2][i] + sg * nr[2] / l;
                    s[k] = rs_at(&rv, (int64_t)floor(pz / D + 0.5 / D), (int64_t)floor(py / D + 0.5 / D), (int64_t)floor(px / D + 0.5 / D));
                }
                if (s[0] <= 0 || s[1] <= 0) { skipped++; continue; }
                nodes++; outr += s[0] > 128; inv += s[1] <= 128; both += s[0] > 128 && s[1] <= 128; dsum += s[0] - s[1];
            }
            rs_free(&rv); for (int a = 0; a < 3; a++) free(X[a]);
        }
        fclose(f);
        fprintf(of, "atlas-rvscore\t%s\tpatches\t%zu\tnodes\t%" PRIu64 "\touter_recto\t%.4f\tinner_verso\t%.4f\tboth\t%.4f\tmean_out_minus_in\t%.2f\tskipped\t%" PRIu64 "\n", tag, np,
               nodes, nodes ? (double)outr / nodes : NAN, nodes ? (double)inv / nodes : NAN, nodes ? (double)both / nodes : NAN, nodes ? dsum / nodes : NAN, skipped);
        if (!nodes) rc = 1;
        for (size_t k = 0; k < ni; k++) free(ids[k]);
        free(ids); free(ax.z); free(ax.y); free(ax.x); did = 1;
    }
    if (outp) fclose(of);
    if (!did) { fprintf(stderr, "atlas-rvscore: give --box ... and / or --patches ...\n"); return 2; }
    return rc;
}
