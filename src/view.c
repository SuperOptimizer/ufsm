/* second-scan views (src/view.h) */
#include "view.h"
#include "json.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *slurp(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb"); if (!f) return nullptr;
    fseek(f, 0, SEEK_END); const long L = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = L >= 0 ? malloc((size_t)L + 1) : nullptr;
    if (!b || fread(b, 1, (size_t)L, f) != (size_t)L) { fclose(f); free(b); return nullptr; }
    b[L] = 0; fclose(f); if (n) *n = (size_t)L; return b;
}
static int get_aff(const json *a, raff *r) {
    if (!a || a->type != J_ARR || a->n < 3) return -1;
    for (int i = 0; i < 3; i++) { const json *row = json_at(a, i); if (!row || row->n < 4) return -1; for (int j = 0; j < 4; j++) r->m[i][j] = json_num(json_at(row, j), 0); }
    return 0;
}

sview *sview_open(const char *map, int lv, double src_um, const char *cache) {
    sview *v = calloc(1, sizeof *v); if (!v) return nullptr;
    v->lv = lv < 0 ? 0 : lv;
    if (!map || !map[0]) return v;
    size_t n = 0; char *t = slurp(map, &n); json *j = t ? json_parse(t, n) : nullptr; free(t);
    if (!j) { fprintf(stderr, "view: cannot read the map %s\n", map); free(v); return nullptr; }
    int bad = get_aff(json_get(j, "A"), &v->m.A);
    if (!bad) {
        v->m.Ainv = raff_inv(v->m.A);
        const json *fd = json_get(j, "field");
        if (fd) {   /* the displacement grid next to the map (reg format: "field": {"file", "n", "o", "h"}) */
            char dir[1024], fp[2100]; snprintf(dir, sizeof dir, "%s", map); char *s = strrchr(dir, '/'); if (s) *s = 0; else snprintf(dir, sizeof dir, ".");
            const char *fn = json_str(json_get(fd, "file"), "field.f32");
            if (fn[0] == '/') snprintf(fp, sizeof fp, "%s", fn); else snprintf(fp, sizeof fp, "%s/%s", dir, fn);
            for (int a = 0; a < 3; a++) { v->D.n[a] = (int)json_num(json_at(json_get(fd, "n"), a), 0); v->D.o[a] = json_num(json_at(json_get(fd, "o"), a), 0); }
            v->D.h = json_num(json_get(fd, "h"), 1);
            const size_t N = (size_t)v->D.n[0] * v->D.n[1] * v->D.n[2] * 3; size_t got = 0;
            v->D.d = (float *)slurp(fp, &got);
            if (!v->D.d || got != N * sizeof(float)) { fprintf(stderr, "view: field %s: %zu bytes, expected %zu\n", fp, got, N * sizeof(float)); bad = 1; }
            else v->m.D = &v->D;
        }
    } else fprintf(stderr, "view: %s has no A\n", map);
    const json *c = json_get(j, "coarse"), *f = json_get(j, "fine");
    v->cum = json_num(json_get(c, "um"), 0);
    if (!bad && src_um > 0 && v->cum > 0 && fabs(v->cum / src_um - 1) > 0.02) {
        fprintf(stderr, "view: %s: the map's coarse scan is %.3f um, the source %.3f um (the source must be the map's coarse scan)\n", map, v->cum, src_um); bad = 1;
    }
    if (!bad) {
        v->fv = rvol_open(json_str(json_get(f, "root"), ""), json_str(json_get(f, "key"), ""), cache, json_num(json_get(f, "um"), 0));
        if (!v->fv) { fprintf(stderr, "view: %s: cannot open the fine scan\n", map); bad = 1; }
    }
    json_free(j);
    if (bad) { sview_free(v); return nullptr; }
    if (v->cum <= 0) v->cum = src_um;
    v->fum = v->fv->um; v->has_map = 1;
    v->lf = rvol_level_for(v->fv, v->cum * (double)(1 << v->lv));
    fprintf(stderr, "view: %s: coarse level %d (%.2f um) <- %s level %d (%.2f um)\n", map, v->lv, v->cum * (1 << v->lv), v->fv->name, v->lf, v->fum * (1 << v->lf));
    return v;
}
void sview_free(sview *v) {
    if (!v) return;
    rvol_close(v->fv); free(v->D.d); free(v);
}

int sview_read(const sview *v, z3 *self, const int64_t o[3], int Pv, uint8_t *out) {
    const size_t p3 = (size_t)Pv * Pv * Pv;
    memset(out, 0, p3);
    if (!v->has_map) {
        if (!self) return -1;
        const z3_meta *m = z3_meta_of(self); int64_t ro[3], rn[3];
        for (int d = 0; d < 3; d++) { ro[d] = o[d] < 0 ? 0 : o[d]; const int64_t e = o[d] + Pv < m->shape[d] ? o[d] + Pv : m->shape[d]; rn[d] = e - ro[d]; if (rn[d] <= 0) return 0; }
        const size_t rnn = (size_t)rn[0] * rn[1] * rn[2];
        uint8_t *tmp = malloc(rnn); if (!tmp) return -1;
        if (z3_read(self, ro, rn, tmp, 1)) { free(tmp); return -1; }
        for (int64_t z = 0; z < rn[0]; z++) for (int64_t y = 0; y < rn[1]; y++)
            memcpy(out + ((size_t)(z + ro[0] - o[0]) * Pv + (size_t)(y + ro[1] - o[1])) * Pv + (size_t)(ro[2] - o[2]), tmp + ((size_t)z * rn[1] + y) * rn[2], (size_t)rn[2]);
        free(tmp);
        return 0;
    }
    const double org[3] = {(double)o[2], (double)o[1], (double)o[0]}; const int nn[3] = {Pv, Pv, Pv};
    rblk fb; int err = 0;
    uint8_t *fbuf = reg_read_fine(v->fv, v->lf, &v->m, v->lv, org, nn, 1.0, &fb, &err);
    if (!fbuf) return err == 2 ? 0 : -1;   /* (outside the partner scan: an empty crop) */
    float *t = malloc(p3 * sizeof *t); if (!t) { free(fbuf); return -1; }
    const double sh[3] = {0, 0, 0};
    reg_resample(&fb, &v->m, v->lv, v->lf, org, nn, sh, 1, t);
    for (size_t i = 0; i < p3; i++) { const float a = t[i]; out[i] = a <= 0.f ? 0 : a >= 255.f ? 255 : (uint8_t)lrintf(a); }
    free(t); free(fbuf);
    return 0;
}
