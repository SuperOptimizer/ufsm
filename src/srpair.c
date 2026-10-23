/* Real super-resolution pairs for train --task restore (src/srpair.h) */
#include "srpair.h"
#include "json.h"
#include "store.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *slurp_file(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb"); if (!f) return nullptr;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char *t = sz >= 0 ? malloc((size_t)sz + 1) : nullptr;
    if (!t || fread(t, 1, (size_t)sz, f) != (size_t)sz) { free(t); fclose(f); return nullptr; }
    fclose(f); t[sz] = 0; *n = (size_t)sz; return t;
}
static json *load_json(const char *dir, const char *name) {
    char p[2048]; snprintf(p, sizeof p, "%s/%s", dir, name);
    size_t n; char *t = slurp_file(p, &n); if (!t) { fprintf(stderr, "sr-pair: cannot read %s\n", p); return nullptr; }
    json *j = json_parse(t, n); free(t);
    if (!j) fprintf(stderr, "sr-pair: cannot parse %s\n", p);
    return j;
}

void srpair_free(srpair *p) {
    if (!p) return;
    z3_close(p->in); z3_close(p->tg); free(p->tr); free(p->te); free(p);
}

srpair *srpair_open(const char *dir, const char *role, int lut) {
    srpair *p = calloc(1, sizeof *p);
    p->role = role && !strcmp(role, "val");
    if (role && strcmp(role, "val") && strcmp(role, "train")) { fprintf(stderr, "sr-pair: %s: role \"train\" or \"val\"\n", dir); free(p); return nullptr; }
    json *bj = load_json(dir, "boxes.json"), *ij = lut ? load_json(dir, "intensity_map.json") : nullptr;
    if (!bj || (lut && !ij)) { json_free(bj); json_free(ij); free(p); return nullptr; }
    p->B = (int)json_num(json_get(json_get(bj, "input"), "box"), 0); p->T = (int)json_num(json_get(json_get(bj, "target"), "box"), 0);
    const char *tk = json_str(json_get(json_get(bj, "target"), "key"), "");
    const char *kv = strstr(tk, "keV");
    if (kv) { const char *s = kv; while (s > tk && (s[-1] >= '0' && s[-1] <= '9')) s--; p->keV = atof(s); }
    const json *bx = json_get(bj, "boxes");
    p->n = bx ? (int)bx->n : 0;
    p->tr = malloc(((size_t)p->n + 1) * sizeof *p->tr); p->te = malloc(((size_t)p->n + 1) * sizeof *p->te);
    for (int k = 0; k < p->n; k++) {
        const json *b = json_at(bx, (size_t)k);
        const int slot = (int)json_num(json_get(b, "slot"), k), test = !strcmp(json_str(json_get(b, "split"), "train"), "test");
        if (p->role || test) p->te[p->nte++] = slot; else p->tr[p->ntr++] = slot;
    }
    json_free(bj);
    for (int v = 0; v < 256; v++) p->lut[v] = (uint8_t)v;
    if (ij) {
        const json *l = json_get(ij, "lut");
        if (!l || l->n != 256) { fprintf(stderr, "sr-pair: %s: intensity_map.json needs a 256-entry \"lut\"\n", dir); json_free(ij); srpair_free(p); return nullptr; }
        for (int v = 1; v < 256; v++) { const double t = json_num(json_at(l, (size_t)v), v); p->lut[v] = (uint8_t)(t < 1 ? 1 : t > 255 ? 255 : lrint(t)); }
        json_free(ij);
    }
    p->lut[0] = 0;
    store *st = store_open(dir);
    p->in = st ? z3_open(st, "input.zarr", nullptr) : nullptr;
    p->tg = st ? z3_open(st, "target.zarr", nullptr) : nullptr;
    if (!p->in || !p->tg || p->B <= 0 || p->T != 4 * p->B || p->n <= 0) {
        fprintf(stderr, "sr-pair: %s: cannot open input.zarr / target.zarr, or the boxes are not B^3 -> (4B)^3 (B %d T %d n %d)\n", dir, p->B, p->T, p->n);
        srpair_free(p); return nullptr;
    }
    const z3_meta *mi = z3_meta_of(p->in), *mt = z3_meta_of(p->tg);
    if (mi->shape[0] < (int64_t)p->n * p->B || mi->shape[1] != p->B || mt->shape[0] < (int64_t)p->n * p->T || mt->shape[1] != p->T) {
        fprintf(stderr, "sr-pair: %s: array shapes do not match %d boxes of %d^3 / %d^3\n", dir, p->n, p->B, p->T);
        srpair_free(p); return nullptr;
    }
    return p;
}

int srpair_read(const srpair *p, int k, const int oc[3], int P, uint8_t *tgt, uint8_t *coarse) {
    const int c = P / 4;
    if (P % 4 || c > p->B || k < 0 || k >= p->n) return -1;
    for (int d = 0; d < 3; d++) if (oc[d] < 0 || oc[d] + c > p->B) return -1;
    const int64_t ot[3] = {(int64_t)k * p->T + 4 * oc[0], 4 * oc[1], 4 * oc[2]}, nt[3] = {P, P, P};
    const int64_t oi[3] = {(int64_t)k * p->B + oc[0], oc[1], oc[2]}, ni[3] = {c, c, c};
    if (z3_read(p->tg, ot, nt, tgt, 1) || z3_read(p->in, oi, ni, coarse, 1)) return -1;
    const size_t p3 = (size_t)P * P * P;
    for (size_t i = 0; i < p3; i++) tgt[i] = p->lut[tgt[i]];
    return 0;
}
