/* Weight files of external PyTorch models (wload.h): safetensors and torch.save zip checkpoints. */
#include "wload.h"
#include "json.h"
#include "zipr.h"
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static _Thread_local char g_err[512];
const char *wl_error(void) { return g_err; }
static void *seterr(const char *fmt, ...) {
    va_list a; va_start(a, fmt); vsnprintf(g_err, sizeof g_err, fmt, a); va_end(a); return nullptr;
}

typedef struct { char *key; int isnum; double num; char *str; } wl_meta;
struct wl_dict { int n, cap; wl_tensor *t; int nm; wl_meta *m; };

static int dict_push(wl_dict *d, const char *name, int ndim, const int64_t *shape, float *data) {
    if (d->n == d->cap) { d->cap = d->cap ? d->cap * 2 : 256; wl_tensor *g = realloc(d->t, (size_t)d->cap * sizeof *g); if (!g) return -1; d->t = g; }
    wl_tensor *t = &d->t[d->n++]; memset(t, 0, sizeof *t);
    t->name = strdup(name); t->ndim = ndim; t->numel = 1;
    for (int i = 0; i < ndim; i++) { t->shape[i] = shape[i]; t->numel *= shape[i]; }
    t->data = data;
    return t->name ? 0 : -1;
}
void wl_free(wl_dict *d) {
    if (!d) return;
    for (int i = 0; i < d->n; i++) { free(d->t[i].name); free(d->t[i].data); }
    for (int i = 0; i < d->nm; i++) { free(d->m[i].key); free(d->m[i].str); }
    free(d->t); free(d->m); free(d);
}
int wl_count(const wl_dict *d) { return d->n; }
const wl_tensor *wl_at(const wl_dict *d, int i) { return i >= 0 && i < d->n ? &d->t[i] : nullptr; }
const wl_tensor *wl_get(const wl_dict *d, const char *name) {
    for (int i = 0; i < d->n; i++) if (!strcmp(d->t[i].name, name)) return &d->t[i];
    return nullptr;
}
const wl_tensor *wl_getf(const wl_dict *d, const char *fmt, ...) {
    char b[512]; va_list a; va_start(a, fmt); vsnprintf(b, sizeof b, fmt, a); va_end(a);
    return wl_get(d, b);
}
double wl_num(const wl_dict *d, const char *key, double dflt) {
    for (int i = 0; i < d->nm; i++) if (d->m[i].isnum && !strcmp(d->m[i].key, key)) return d->m[i].num;
    return dflt;
}
const char *wl_str(const wl_dict *d, const char *key) {
    for (int i = 0; i < d->nm; i++) if (d->m[i].str && !strcmp(d->m[i].key, key)) return d->m[i].str;
    return nullptr;
}

/* ---- element conversion ---- */
enum { DT_F32, DT_F16, DT_BF16, DT_F64, DT_I64, DT_I32, DT_I16, DT_I8, DT_U8, DT_BOOL };
static const int dt_size[] = {4, 2, 2, 8, 8, 4, 2, 1, 1, 1};
static float half_to_f(uint16_t h) {
    const uint32_t s = (uint32_t)(h >> 15) << 31, e = (h >> 10) & 31, m = h & 1023; uint32_t u;
    if (e == 0) { if (!m) u = s; else { float f = ldexpf((float)m, -24); return s ? -f : f; } }
    else if (e == 31) u = s | 0x7f800000u | (m << 13);
    else u = s | ((e + 112) << 23) | (m << 13);
    float f; memcpy(&f, &u, 4); return f;
}
static float elem_f(const uint8_t *p, int dt) {
    switch (dt) {
    case DT_F32: { float f; memcpy(&f, p, 4); return f; }
    case DT_F16: { uint16_t h; memcpy(&h, p, 2); return half_to_f(h); }
    case DT_BF16: { uint16_t h; memcpy(&h, p, 2); uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
    case DT_F64: { double v; memcpy(&v, p, 8); return (float)v; }
    case DT_I64: { int64_t v; memcpy(&v, p, 8); return (float)v; }
    case DT_I32: { int32_t v; memcpy(&v, p, 4); return (float)v; }
    case DT_I16: { int16_t v; memcpy(&v, p, 2); return (float)v; }
    case DT_I8: return (float)(int8_t)p[0];
    default: return (float)p[0];
    }
}

/* ---- safetensors ---- */
static int st_dtype(const char *s) {
    static const char *nm[] = {"F32", "F16", "BF16", "F64", "I64", "I32", "I16", "I8", "U8", "BOOL"};
    for (int i = 0; i < 10; i++) if (!strcmp(s, nm[i])) return i;
    return -1;
}
static wl_dict *load_safetensors(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) return seterr("cannot open %s", path);
    uint64_t hl = 0; if (fread(&hl, 8, 1, f) != 1 || hl > (1u << 30)) { fclose(f); return seterr("%s: bad safetensors header", path); }
    char *h = malloc(hl + 1); if (!h || fread(h, 1, hl, f) != hl) { free(h); fclose(f); return seterr("%s: short header", path); }
    h[hl] = 0;
    json *j = json_parse(h, hl); free(h);
    if (!j || j->type != J_OBJ) { json_free(j); fclose(f); return seterr("%s: bad header JSON", path); }
    wl_dict *d = calloc(1, sizeof *d); const long base = 8 + (long)hl;
    for (size_t i = 0; i < j->n; i++) {
        const json *e = j->items[i]; if (!strcmp(j->keys[i], "__metadata__") || e->type != J_OBJ) continue;
        const int dt = st_dtype(json_str(json_get(e, "dtype"), "")); const json *sh = json_get(e, "shape"), *off = json_get(e, "data_offsets");
        if (dt < 0 || !sh || !off || off->n != 2 || sh->n > 8) { json_free(j); fclose(f); wl_free(d); return seterr("%s: tensor %s: unsupported entry", path, j->keys[i]); }
        int64_t shape[8], n = 1; for (size_t k = 0; k < sh->n; k++) { shape[k] = (int64_t)json_num(sh->items[k], 0); n *= shape[k]; }
        const int64_t b0 = (int64_t)json_num(off->items[0], 0), b1 = (int64_t)json_num(off->items[1], 0);
        if (b1 - b0 != n * dt_size[dt]) { json_free(j); fclose(f); wl_free(d); return seterr("%s: tensor %s: size mismatch", path, j->keys[i]); }
        uint8_t *raw = malloc((size_t)(b1 - b0) + 1); float *v = malloc((size_t)(n ? n : 1) * sizeof *v);
        if (!raw || !v || fseek(f, base + b0, SEEK_SET) || fread(raw, 1, (size_t)(b1 - b0), f) != (size_t)(b1 - b0)) {
            free(raw); free(v); json_free(j); fclose(f); wl_free(d); return seterr("%s: cannot read tensor %s", path, j->keys[i]);
        }
        for (int64_t k = 0; k < n; k++) v[k] = elem_f(raw + k * dt_size[dt], dt);
        free(raw);
        if (dict_push(d, j->keys[i], (int)sh->n, shape, v)) { free(v); json_free(j); fclose(f); wl_free(d); return seterr("out of memory"); }
    }
    json_free(j); fclose(f);
    return d;
}

/* ---- pickle VM (torch.save format) ---- */
enum { PV_NONE, PV_BOOL, PV_INT, PV_FLOAT, PV_STR, PV_BYTES, PV_TUPLE, PV_LIST, PV_DICT, PV_GLOBAL, PV_STORAGE, PV_TENSOR, PV_OBJ, PV_MARK };
typedef struct pv pv;
struct pv {
    int t; int64_t i; double f; char *s; size_t n, cap;   /* s: STR / BYTES / GLOBAL module; n: string length or item count */
    pv **items, **keys;                                    /* TUPLE / LIST items; DICT keys + items; OBJ: items[0] callable, [1] args */
    char *name;                                            /* GLOBAL name; STORAGE key */
    int dtype; int64_t numel;                              /* STORAGE */
    pv *storage; int64_t offset, shape[8], stride[8]; int ndim;   /* TENSOR */
};
typedef struct { pv **p; size_t n, cap; } pvarena;
static pv *pv_new(pvarena *A, int t) {
    if (A->n == A->cap) { A->cap = A->cap ? A->cap * 2 : 4096; pv **g = realloc(A->p, A->cap * sizeof *g); if (!g) return nullptr; A->p = g; }
    pv *v = calloc(1, sizeof *v); if (!v) return nullptr;
    v->t = t; A->p[A->n++] = v; return v;
}
static void arena_free(pvarena *A) {
    for (size_t i = 0; i < A->n; i++) { pv *v = A->p[i]; free(v->s); free(v->name); free(v->items); free(v->keys); free(v); }
    free(A->p);
}
static int pv_append(pv *c, pv *x) {
    if (c->n == c->cap) { c->cap = c->cap ? c->cap * 2 : 8; pv **g = realloc(c->items, c->cap * sizeof *g); if (!g) return -1; c->items = g; }
    c->items[c->n++] = x; return 0;
}
static int pv_setitem(pv *d, pv *k, pv *x) {
    if (d->t != PV_DICT) return 0;   /* items set on unknown objects are dropped */
    if (d->n == d->cap) {
        d->cap = d->cap ? d->cap * 2 : 8; pv **g = realloc(d->items, d->cap * sizeof *g), **h = g ? realloc(d->keys, d->cap * sizeof *h) : nullptr;
        if (!g || !h) return -1;
        d->items = g; d->keys = h;
    }
    d->keys[d->n] = k; d->items[d->n++] = x; return 0;
}
static const char *pv_s(const pv *v) { return v && (v->t == PV_STR) ? v->s : nullptr; }
static int storage_dtype(const char *name) {
    static const struct { const char *n; int dt; } T[] = {{"FloatStorage", DT_F32}, {"HalfStorage", DT_F16}, {"BFloat16Storage", DT_BF16},
        {"DoubleStorage", DT_F64}, {"LongStorage", DT_I64}, {"IntStorage", DT_I32}, {"ShortStorage", DT_I16}, {"CharStorage", DT_I8},
        {"ByteStorage", DT_U8}, {"BoolStorage", DT_BOOL}};
    for (size_t i = 0; i < sizeof T / sizeof *T; i++) if (!strcmp(name, T[i].n)) return T[i].dt;
    return -1;
}

typedef struct { const uint8_t *b; size_t n, at; } rd;
static int rd_need(rd *r, size_t k) { return r->at + k <= r->n ? 0 : -1; }
static uint64_t rd_le(rd *r, int k) { uint64_t v = 0; for (int i = 0; i < k; i++) v |= (uint64_t)r->b[r->at + i] << (8 * i); r->at += k; return v; }

/* runs the pickle; returns the top value or nullptr (g_err set) */
static pv *unpickle(const uint8_t *b, size_t n, pvarena *A) {
    rd r = {b, n, 0};
    size_t sn = 0, scap = 256; pv **st = malloc(scap * sizeof *st);
    size_t mn = 0, mcap = 1024; pv **memo = calloc(mcap, sizeof *memo);
    size_t marks[4096]; int nmarks = 0;
    pv *result = nullptr;
#define PUSH(v) do { pv *_v = (v); if (!_v) goto oom; if (sn == scap) { scap *= 2; pv **_g = realloc(st, scap * sizeof *st); if (!_g) goto oom; st = _g; } st[sn++] = _v; } while (0)
#define POP() (sn ? st[--sn] : nullptr)
#define NEED(k) do { if (rd_need(&r, (k))) { seterr("pickle: truncated"); goto fail; } } while (0)
    if (!st || !memo) goto oom;
    while (r.at < r.n) {
        const uint8_t op = b[r.at++];
        switch (op) {
        case 0x80: NEED(1); r.at++; break;                  /* PROTO */
        case 0x95: NEED(8); r.at += 8; break;               /* FRAME */
        case '.': result = POP(); goto done;                /* STOP */
        case '(': if (nmarks == 4096) { seterr("pickle: marks"); goto fail; } marks[nmarks++] = sn; break;
        case '0': (void)POP(); break;
        case '1': if (!nmarks) goto bad; sn = marks[--nmarks]; break;
        case '2': if (!sn) goto bad; PUSH(st[sn - 1]); break;
        case 'N': PUSH(pv_new(A, PV_NONE)); break;
        case 0x88: case 0x89: { pv *v = pv_new(A, PV_BOOL); if (!v) goto oom; v->i = op == 0x88; PUSH(v); break; }
        case 'J': { NEED(4); pv *v = pv_new(A, PV_INT); if (!v) goto oom; v->i = (int32_t)rd_le(&r, 4); PUSH(v); break; }
        case 'K': { NEED(1); pv *v = pv_new(A, PV_INT); if (!v) goto oom; v->i = (int64_t)rd_le(&r, 1); PUSH(v); break; }
        case 'M': { NEED(2); pv *v = pv_new(A, PV_INT); if (!v) goto oom; v->i = (int64_t)rd_le(&r, 2); PUSH(v); break; }
        case 0x8a: case 0x8b: {                             /* LONG1 / LONG4 */
            NEED(op == 0x8a ? 1 : 4); const size_t k = (size_t)rd_le(&r, op == 0x8a ? 1 : 4); NEED(k);
            if (k > 8) { seterr("pickle: long of %zu bytes", k); goto fail; }
            uint64_t u = rd_le(&r, (int)k); if (k && k < 8 && (u >> (8 * k - 1) & 1)) u |= ~0ull << (8 * k);
            pv *v = pv_new(A, PV_INT); if (!v) goto oom; v->i = (int64_t)u; PUSH(v); break;
        }
        case 'G': {                                          /* BINFLOAT: big-endian double */
            NEED(8); uint64_t u = 0; for (int i = 0; i < 8; i++) u = u << 8 | b[r.at + i]; r.at += 8;
            pv *v = pv_new(A, PV_FLOAT); if (!v) goto oom; memcpy(&v->f, &u, 8); PUSH(v); break;
        }
        case 'X': case 0x8c: case 0x8d: case 'T': case 'U': case 'B': case 'C': case 0x8e: {
            const int lb = op == 0x8c || op == 'U' || op == 'C' ? 1 : op == 0x8d || op == 0x8e ? 8 : 4;
            NEED(lb); const size_t k = (size_t)rd_le(&r, lb); NEED(k);
            pv *v = pv_new(A, op == 'X' || op == 0x8c || op == 0x8d || op == 'T' || op == 'U' ? PV_STR : PV_BYTES); if (!v) goto oom;
            v->s = malloc(k + 1); if (!v->s) goto oom; memcpy(v->s, b + r.at, k); v->s[k] = 0; v->n = k; r.at += k; PUSH(v); break;
        }
        case ')': PUSH(pv_new(A, PV_TUPLE)); break;
        case 't': case 'l': {
            if (!nmarks) goto bad;
            const size_t m = marks[--nmarks];
            pv *v = pv_new(A, op == 't' ? PV_TUPLE : PV_LIST); if (!v) goto oom;
            for (size_t i = m; i < sn; i++) if (pv_append(v, st[i])) goto oom;
            sn = m; PUSH(v); break;
        }
        case 0x85: case 0x86: case 0x87: {
            const size_t k = (size_t)(op - 0x84); if (sn < k) goto bad;
            pv *v = pv_new(A, PV_TUPLE); if (!v) goto oom;
            for (size_t i = sn - k; i < sn; i++) if (pv_append(v, st[i])) goto oom;
            sn -= k; PUSH(v); break;
        }
        case ']': PUSH(pv_new(A, PV_LIST)); break;
        case 'a': { pv *x = POP(); if (!sn || !x) goto bad; if (st[sn - 1]->t == PV_LIST && pv_append(st[sn - 1], x)) goto oom; break; }
        case 'e': {
            if (!nmarks) goto bad;
            const size_t m = marks[--nmarks]; if (m == 0) goto bad;
            pv *l = st[m - 1]; if (l->t == PV_LIST) for (size_t i = m; i < sn; i++) if (pv_append(l, st[i])) goto oom;
            sn = m; break;
        }
        case '}': PUSH(pv_new(A, PV_DICT)); break;
        case 'd': {
            if (!nmarks) goto bad;
            const size_t m = marks[--nmarks];
            pv *v = pv_new(A, PV_DICT); if (!v) goto oom;
            for (size_t i = m; i + 1 < sn; i += 2) if (pv_setitem(v, st[i], st[i + 1])) goto oom;
            sn = m; PUSH(v); break;
        }
        case 's': { pv *x = POP(), *k = POP(); if (!sn || !x || !k) goto bad; if (pv_setitem(st[sn - 1], k, x)) goto oom; break; }
        case 'u': {
            if (!nmarks) goto bad;
            const size_t m = marks[--nmarks]; if (m == 0) goto bad;
            for (size_t i = m; i + 1 < sn; i += 2) if (pv_setitem(st[m - 1], st[i], st[i + 1])) goto oom;
            sn = m; break;
        }
        case 'q': case 'r': case 0x94: {
            size_t idx;
            if (op == 0x94) idx = mn; else { const int k = op == 'q' ? 1 : 4; NEED(k); idx = (size_t)rd_le(&r, k); }
            if (!sn) goto bad;
            if (idx >= mcap) { size_t nc = mcap; while (nc <= idx) nc *= 2; pv **g = realloc(memo, nc * sizeof *g); if (!g) goto oom; memset(g + mcap, 0, (nc - mcap) * sizeof *g); memo = g; mcap = nc; }
            memo[idx] = st[sn - 1]; if (idx >= mn) mn = idx + 1; break;
        }
        case 'h': case 'j': {
            const int k = op == 'h' ? 1 : 4; NEED(k); const size_t idx = (size_t)rd_le(&r, k);
            if (idx >= mcap || !memo[idx]) goto bad;
            PUSH(memo[idx]); break;
        }
        case 'c': {                                          /* GLOBAL module\nname\n */
            const char *e1 = memchr(b + r.at, '\n', r.n - r.at); if (!e1) goto bad;
            const char *e2 = memchr(e1 + 1, '\n', r.n - (size_t)(e1 + 1 - (const char *)b)); if (!e2) goto bad;
            pv *v = pv_new(A, PV_GLOBAL); if (!v) goto oom;
            v->s = strndup((const char *)b + r.at, (size_t)(e1 - (const char *)b - r.at)); v->name = strndup(e1 + 1, (size_t)(e2 - e1 - 1));
            r.at = (size_t)(e2 + 1 - (const char *)b); PUSH(v); break;
        }
        case 0x93: {                                         /* STACK_GLOBAL */
            pv *nm = POP(), *md = POP(); if (!pv_s(nm) || !pv_s(md)) goto bad;
            pv *v = pv_new(A, PV_GLOBAL); if (!v) goto oom; v->s = strdup(md->s); v->name = strdup(nm->s); PUSH(v); break;
        }
        case 'Q': {                                          /* BINPERSID: ('storage', type, key, location, numel) */
            pv *id = POP(); if (!id || id->t != PV_TUPLE || id->n < 5 || !pv_s(id->items[0]) || strcmp(id->items[0]->s, "storage")) { seterr("pickle: unsupported persistent id"); goto fail; }
            const pv *ty = id->items[1], *key = id->items[2], *cnt = id->items[4];
            const int dt = ty->t == PV_GLOBAL ? storage_dtype(ty->name) : -1;
            if (dt < 0 || !pv_s(key) || cnt->t != PV_INT) { seterr("pickle: unsupported storage %s", ty->t == PV_GLOBAL ? ty->name : "?"); goto fail; }
            pv *v = pv_new(A, PV_STORAGE); if (!v) goto oom; v->dtype = dt; v->name = strdup(key->s); v->numel = cnt->i; PUSH(v); break;
        }
        case 'R': case 0x81: {                               /* REDUCE / NEWOBJ */
            pv *args = POP(), *fn = POP(); if (!args || !fn) goto bad;
            pv *v = nullptr;
            if (fn->t == PV_GLOBAL && !strcmp(fn->name, "_rebuild_tensor_v2") && args->t == PV_TUPLE && args->n >= 4) {
                const pv *sto = args->items[0], *off = args->items[1], *sz = args->items[2], *sd = args->items[3];
                if (sto->t != PV_STORAGE || off->t != PV_INT || sz->t != PV_TUPLE || sd->t != PV_TUPLE || sz->n > 8 || sd->n != sz->n) { seterr("pickle: bad tensor rebuild"); goto fail; }
                v = pv_new(A, PV_TENSOR); if (!v) goto oom;
                v->storage = (pv *)sto; v->offset = off->i; v->ndim = (int)sz->n;
                for (size_t i = 0; i < sz->n; i++) { v->shape[i] = sz->items[i]->i; v->stride[i] = sd->items[i]->i; }
            } else if (fn->t == PV_GLOBAL && !strcmp(fn->name, "_rebuild_parameter") && args->t == PV_TUPLE && args->n >= 1) {
                v = args->items[0];
            } else if (fn->t == PV_GLOBAL && !strcmp(fn->name, "OrderedDict")) {
                v = pv_new(A, PV_DICT);
            } else {
                v = pv_new(A, PV_OBJ); if (!v) goto oom;
                if (pv_append(v, fn) || pv_append(v, args)) goto oom;
            }
            PUSH(v); break;
        }
        case 'b': (void)POP(); break;                        /* BUILD: state dropped (tensor hooks, OrderedDict attrs) */
        case 0x8f: PUSH(pv_new(A, PV_LIST)); break;          /* EMPTY_SET as a list */
        case 0x90: { if (!nmarks) goto bad; sn = marks[--nmarks]; break; }   /* ADDITEMS dropped */
        default: seterr("pickle: unsupported opcode 0x%02x at %zu", op, r.at - 1); goto fail;
        }
    }
    seterr("pickle: no STOP");
    goto fail;
bad:
    seterr("pickle: malformed stream at %zu", r.at);
    goto fail;
oom:
    seterr("pickle: out of memory");
fail:
    result = nullptr;
done:
    free(st); free(memo);
    return result;
#undef PUSH
#undef POP
#undef NEED
}

static int is_tensor_dict(const pv *d) {
    if (!d || d->t != PV_DICT || !d->n) return 0;
    for (size_t i = 0; i < d->n; i++) if (d->items[i]->t == PV_TENSOR) return 1;
    return 0;
}

static wl_dict *load_pth(const char *path, const char *sub) {
    zipr *z = zipr_open(path); if (!z) return seterr("%s: %s", path, zipr_error());
    long pk = -1; char prefix[512] = "";
    for (size_t i = 0; i < zipr_count(z); i++) {
        zip_entry e; if (zipr_entry(z, i, &e)) continue;
        const size_t l = strlen(e.name);
        if (l >= 8 && !strcmp(e.name + l - 8, "data.pkl")) { pk = (long)i; snprintf(prefix, sizeof prefix, "%.*s", (int)(l - 8), e.name); break; }
    }
    if (pk < 0) { zipr_close(z); return seterr("%s: no data.pkl (not a torch.save zip)", path); }
    size_t plen; uint8_t *pb = zipr_read(z, (size_t)pk, &plen); if (!pb) { zipr_close(z); return seterr("%s: cannot read data.pkl", path); }
    pvarena A = {0}; pv *top = unpickle(pb, plen, &A); free(pb);
    if (!top) { arena_free(&A); zipr_close(z); char e[512]; snprintf(e, sizeof e, "%s", g_err); return seterr("%s: %s", path, e); }
    wl_dict *d = calloc(1, sizeof *d);
    const pv *sd = nullptr;
    if (top->t == PV_DICT) {
        for (size_t i = 0; i < top->n; i++) {   /* top-level numbers / strings */
            const pv *k = top->keys[i], *v = top->items[i]; if (!pv_s(k)) continue;
            if (v->t == PV_INT || v->t == PV_BOOL || v->t == PV_FLOAT || v->t == PV_STR) {
                d->m = realloc(d->m, (size_t)(d->nm + 1) * sizeof *d->m); wl_meta *m = &d->m[d->nm++]; memset(m, 0, sizeof *m);
                m->key = strdup(k->s);
                if (v->t == PV_STR) m->str = strdup(v->s); else { m->isnum = 1; m->num = v->t == PV_FLOAT ? v->f : (double)v->i; }
            }
        }
        if (sub) { for (size_t i = 0; i < top->n; i++) if (pv_s(top->keys[i]) && !strcmp(top->keys[i]->s, sub)) sd = top->items[i]; }
        else if (is_tensor_dict(top)) sd = top;
        else {
            static const char *pref[] = {"ema_model", "model", "state_dict"};
            for (int q = 0; q < 3 && !sd; q++) for (size_t i = 0; i < top->n; i++) if (pv_s(top->keys[i]) && !strcmp(top->keys[i]->s, pref[q]) && is_tensor_dict(top->items[i])) sd = top->items[i];
            for (size_t i = 0; i < top->n && !sd; i++) if (is_tensor_dict(top->items[i])) sd = top->items[i];
        }
    }
    if (!sd || sd->t != PV_DICT) { arena_free(&A); zipr_close(z); wl_free(d); return seterr("%s: no state dict%s%s", path, sub ? " named " : "", sub ? sub : ""); }
    /* storages are shared between tensors: read each data/<key> once */
    typedef struct { const char *key; uint8_t *data; size_t len; } sto_t;
    size_t ns = 0; sto_t *S = calloc(sd->n + 1, sizeof *S);
    for (size_t i = 0; i < sd->n; i++) {
        const pv *k = sd->keys[i], *v = sd->items[i];
        if (!pv_s(k) || v->t != PV_TENSOR) continue;
        const pv *sto = v->storage; size_t si = 0; while (si < ns && strcmp(S[si].key, sto->name)) si++;
        if (si == ns) {
            char en[1100]; snprintf(en, sizeof en, "%sdata/%s", prefix, sto->name);
            const long ei = zipr_find(z, en); size_t len = 0; uint8_t *data = ei >= 0 ? zipr_read(z, (size_t)ei, &len) : nullptr;
            if (!data || len < (size_t)sto->numel * dt_size[sto->dtype]) {
                free(data); for (size_t q = 0; q < ns; q++) free(S[q].data); free(S); arena_free(&A); zipr_close(z); wl_free(d);
                return seterr("%s: storage %s missing or short", path, en);
            }
            S[ns++] = (sto_t){sto->name, data, len};
        }
        int64_t numel = 1; for (int q = 0; q < v->ndim; q++) numel *= v->shape[q];
        float *out = malloc((size_t)(numel ? numel : 1) * sizeof *out); if (!out) goto oom;
        const int es = dt_size[sto->dtype]; int64_t idx[8] = {0};
        for (int64_t e = 0; e < numel; e++) {   /* strided gather in C order */
            int64_t o = v->offset; for (int q = 0; q < v->ndim; q++) o += idx[q] * v->stride[q];
            if (o < 0 || o >= sto->numel) { free(out); for (size_t q = 0; q < ns; q++) free(S[q].data); free(S); arena_free(&A); zipr_close(z); wl_free(d); return seterr("%s: tensor %s out of its storage", path, k->s); }
            out[e] = elem_f(S[si].data + (size_t)o * es, sto->dtype);
            for (int q = v->ndim - 1; q >= 0; q--) { if (++idx[q] < v->shape[q]) break; idx[q] = 0; }
        }
        if (dict_push(d, k->s, v->ndim, v->shape, out)) { free(out); goto oom; }
    }
    for (size_t q = 0; q < ns; q++) free(S[q].data);
    free(S); arena_free(&A); zipr_close(z);
    return d;
oom:
    for (size_t q = 0; q < ns; q++) free(S[q].data);
    free(S); arena_free(&A); zipr_close(z); wl_free(d);
    return seterr("out of memory");
}

wl_dict *wl_load(const char *path, const char *sub) {
    FILE *f = fopen(path, "rb"); if (!f) return seterr("cannot open %s", path);
    uint8_t m[4] = {0}; const size_t got = fread(m, 1, 4, f); fclose(f);
    if (got == 4 && m[0] == 'P' && m[1] == 'K') return load_pth(path, sub);
    return load_safetensors(path);
}
