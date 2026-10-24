/* Real super-resolution pairs for train --task restore (src/srpair.h) */
#include "srpair.h"
#include "json.h"
#include "store.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>

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
    if (p->tc) for (int k = 0; k < p->n; k++) { free(p->tc[k]); free(p->ic[k]); }
    free(p->tc); free(p->ic);
    z3_close(p->in); z3_close(p->tg); free(p->tr); free(p->te); free(p->trw); free(p);
}

srpair *srpair_open(const char *dir, const char *role, int lut, int cache) {
    srpair *p = calloc(1, sizeof *p);
    p->role = role && !strcmp(role, "val");
    if (role && strcmp(role, "val") && strcmp(role, "train")) { fprintf(stderr, "sr-pair: %s: role \"train\" or \"val\"\n", dir); free(p); return nullptr; }
    json *bj = load_json(dir, "boxes.json"), *ij = lut ? load_json(dir, "intensity_map.json") : nullptr;
    if (!bj || (lut && !ij)) { json_free(bj); json_free(ij); free(p); return nullptr; }
    p->B = (int)json_num(json_get(json_get(bj, "input"), "box"), 0); p->T = (int)json_num(json_get(json_get(bj, "target"), "box"), 0);
    p->same = !strcmp(json_str(json_get(json_get(bj, "input"), "grid"), ""), "target");
    p->ratio = json_num(json_get(bj, "ratio"), 4);
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
    if (!p->in || !p->tg || p->B <= 0 || p->T != (p->same ? p->B : 4 * p->B) || p->n <= 0) {
        fprintf(stderr, "sr-pair: %s: cannot open input.zarr / target.zarr, or the boxes are not B^3 -> (4B)^3 (B %d T %d n %d)\n", dir, p->B, p->T, p->n);
        srpair_free(p); return nullptr;
    }
    const z3_meta *mi = z3_meta_of(p->in), *mt = z3_meta_of(p->tg);
    if (mi->shape[0] < (int64_t)p->n * p->B || mi->shape[1] != p->B || mt->shape[0] < (int64_t)p->n * p->T || mt->shape[1] != p->T) {
        fprintf(stderr, "sr-pair: %s: array shapes do not match %d boxes of %d^3 / %d^3\n", dir, p->n, p->B, p->T);
        srpair_free(p); return nullptr;
    }
    if ((p->cache = cache != 0)) { p->tc = calloc((size_t)p->n, sizeof *p->tc); p->ic = calloc((size_t)p->n, sizeof *p->ic); }
    return p;
}
/* box k decoded (cache): read once, published with a compare-and-swap (a racing duplicate read is freed) */
static const uint8_t *box_of(z3 *z, _Atomic(uint8_t *) *slot, int k, int e) {
    uint8_t *b = atomic_load(slot); if (b) return b;
    const size_t e3 = (size_t)e * e * e; b = malloc(e3); if (!b) return nullptr;
    const int64_t o[3] = {(int64_t)k * e, 0, 0}, n[3] = {e, e, e};
    if (z3_read(z, o, n, b, 1)) { free(b); return nullptr; }
    uint8_t *exp = nullptr;
    if (!atomic_compare_exchange_strong(slot, &exp, b)) { free(b); return exp; }
    return b;
}

int srpair_read(const srpair *p, int k, const int oc[3], int P, uint8_t *tgt, uint8_t *coarse) {
    if (p->same) {   /* both crops P^3 at oc on the target grid */
        if (P > p->T || k < 0 || k >= p->n) return -1;
        for (int d = 0; d < 3; d++) if (oc[d] < 0 || oc[d] + P > p->T) return -1;
        const size_t p3 = (size_t)P * P * P;
        if (p->cache) {
            const uint8_t *tb = box_of(p->tg, &p->tc[k], k, p->T), *ib = box_of(p->in, &p->ic[k], k, p->T);
            if (!tb || !ib) return -1;
            for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) {
                const size_t so = ((size_t)(oc[0] + z) * p->T + oc[1] + y) * p->T + oc[2], d = ((size_t)z * P + y) * P;
                for (int x = 0; x < P; x++) tgt[d + x] = p->lut[tb[so + x]];
                memcpy(coarse + d, ib + so, (size_t)P);
            }
            return 0;
        }
        const int64_t o[3] = {(int64_t)k * p->T + oc[0], oc[1], oc[2]}, n[3] = {P, P, P};
        if (z3_read(p->tg, o, n, tgt, 1) || z3_read(p->in, o, n, coarse, 1)) return -1;
        for (size_t i = 0; i < p3; i++) tgt[i] = p->lut[tgt[i]];
        return 0;
    }
    const int c = P / 4;
    if (P % 4 || c > p->B || k < 0 || k >= p->n) return -1;
    for (int d = 0; d < 3; d++) if (oc[d] < 0 || oc[d] + c > p->B) return -1;
    const size_t p3 = (size_t)P * P * P;
    if (p->cache) {   /* crops of the RAM copies */
        const uint8_t *tb = box_of(p->tg, &p->tc[k], k, p->T), *ib = box_of(p->in, &p->ic[k], k, p->B);
        if (!tb || !ib) return -1;
        for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) {
            const uint8_t *src = tb + ((size_t)(4 * oc[0] + z) * p->T + 4 * oc[1] + y) * p->T + 4 * oc[2]; uint8_t *dst = tgt + ((size_t)z * P + y) * P;
            for (int x = 0; x < P; x++) dst[x] = p->lut[src[x]];
        }
        for (int z = 0; z < c; z++) for (int y = 0; y < c; y++) memcpy(coarse + ((size_t)z * c + y) * c, ib + ((size_t)(oc[0] + z) * p->B + oc[1] + y) * p->B + oc[2], (size_t)c);
        return 0;
    }
    const int64_t ot[3] = {(int64_t)k * p->T + 4 * oc[0], 4 * oc[1], 4 * oc[2]}, nt[3] = {P, P, P};
    const int64_t oi[3] = {(int64_t)k * p->B + oc[0], oc[1], oc[2]}, ni[3] = {c, c, c};
    if (z3_read(p->tg, ot, nt, tgt, 1) || z3_read(p->in, oi, ni, coarse, 1)) return -1;
    for (size_t i = 0; i < p3; i++) tgt[i] = p->lut[tgt[i]];
    return 0;
}

int srpair_pick_train(const srpair *p, double u) {
    if (p->ntr <= 0) return -1;
    if (!p->trw) { int i = (int)(u * p->ntr); return i < p->ntr ? i : p->ntr - 1; }
    const double t = u * p->trw[p->ntr - 1]; int lo = 0, hi = p->ntr - 1;
    while (lo < hi) { const int m = (lo + hi) / 2; if (p->trw[m] > t) hi = m; else lo = m + 1; }
    return lo;
}

int srpair_load_weights(srpair *p, const char *path) {
    size_t wn; char *wt = slurp_file(path, &wn); json *wj = wt ? json_parse(wt, wn) : nullptr; free(wt);
    const json *w = wj ? json_get(wj, "w") : nullptr;
    if (!w || (int)w->n < p->n) { fprintf(stderr, "sr-pair: cannot read %d box weights from %s\n", p->n, path); json_free(wj); return -1; }
    const double K = json_num(json_get(wj, "boost"), 4); free(p->trw); p->trw = malloc(((size_t)p->ntr + 1) * sizeof *p->trw); double acc = 0;
    for (int i = 0; i < p->ntr; i++) { const double v = json_num(json_at(w, (size_t)p->tr[i]), 0); acc += 1 + K * (v > 0 ? v : 0); p->trw[i] = acc; }
    fprintf(stderr, "sr-pair: %s: box weights (boost %g), mean draw weight %.3f\n", path, K, p->ntr ? acc / p->ntr : 0);
    json_free(wj); return 0;
}
