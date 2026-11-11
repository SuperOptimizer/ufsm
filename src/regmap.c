/* reg_map_load: a ufsm reg map JSON (A: fine level-0 -> coarse level-0, optional displacement field file) without opening its
   coarse / fine volumes, whose roots in reg.json may be desk paths. Same parsing as reg_cli.c ctx_open (open_vols 0). For
   fiber-raster --map on a pod (2026-11-11); kept apart from reg_cli.c so small tools link only reg.o + json.o. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "json.h"
#include "reg.h"

static char *rm_slurp(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb"); if (!f) return nullptr;
    fseek(f, 0, SEEK_END); const long L = ftell(f); fseek(f, 0, SEEK_SET); if (L < 0) { fclose(f); return nullptr; }
    char *b = malloc((size_t)L + 1); if (!b || fread(b, 1, (size_t)L, f) != (size_t)L) { fclose(f); free(b); return nullptr; }
    b[L] = 0; fclose(f); if (n) *n = (size_t)L; return b;
}
const rmap *reg_map_load(const char *path) {
    size_t n = 0; char *t = rm_slurp(path, &n); if (!t) { fprintf(stderr, "reg: cannot read %s\n", path); return nullptr; }
    json *j = json_parse(t, n); free(t); if (!j) { fprintf(stderr, "reg: bad JSON %s\n", path); return nullptr; }
    rmap *m = calloc(1, sizeof *m); const json *a = json_get(j, "A");
    if (!a || a->type != J_ARR || a->n < 3) { fprintf(stderr, "reg: %s has no A\n", path); json_free(j); free(m); return nullptr; }
    for (int r = 0; r < 3; r++) { const json *row = json_at(a, r); if (!row || row->n < 4) { fprintf(stderr, "reg: %s: bad A\n", path); json_free(j); free(m); return nullptr; }
        for (int c = 0; c < 4; c++) m->A.m[r][c] = json_num(json_at(row, c), 0); }
    m->Ainv = raff_inv(m->A);
    const json *fd = json_get(j, "field");
    if (fd) {
        rfield *D = calloc(1, sizeof *D); char dir[1024], fp[2100]; snprintf(dir, sizeof dir, "%s", path); char *s = strrchr(dir, '/'); if (s) *s = 0; else snprintf(dir, sizeof dir, ".");
        const char *fn = json_str(json_get(fd, "file"), "field.f32");
        if (fn[0] == '/') snprintf(fp, sizeof fp, "%s", fn); else snprintf(fp, sizeof fp, "%s/%s", dir, fn);
        for (int k = 0; k < 3; k++) { D->n[k] = (int)json_num(json_at(json_get(fd, "n"), k), 0); D->o[k] = json_num(json_at(json_get(fd, "o"), k), 0); }
        D->h = json_num(json_get(fd, "h"), 1);
        const size_t N = (size_t)D->n[0] * D->n[1] * D->n[2] * 3; size_t got = 0; float *d = (float *)rm_slurp(fp, &got);
        if (!d || got != N * sizeof(float)) { fprintf(stderr, "reg: field %s: %zu bytes, expected %zu\n", fp, got, N * sizeof(float)); free(d); free(D); json_free(j); free(m); return nullptr; }
        D->d = d; m->D = D;
    }
    json_free(j); return m;
}
