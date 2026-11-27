/* Minimal Apache Parquet reader: see parquet.h. Format reference: apache/parquet-format (parquet.thrift, Encodings.md);
   Thrift compact protocol; snappy's format_description.txt. */
#include "parquet.h"
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <zlib.h>
#include <zstd.h>

static _Thread_local char g_err[512];
const char *pq_error(void) { return g_err; }
static int fail(const char *fmt, ...) {
    va_list a; va_start(a, fmt); vsnprintf(g_err, sizeof g_err, fmt, a); va_end(a); return -1;
}

/* ---- Thrift compact protocol ---- */
typedef struct { const uint8_t *p, *e; int err; } tc;
enum { T_STOP = 0, T_TRUE = 1, T_FALSE = 2, T_BYTE = 3, T_I16 = 4, T_I32 = 5, T_I64 = 6, T_DOUBLE = 7, T_BIN = 8, T_LIST = 9, T_SET = 10,
       T_MAP = 11, T_STRUCT = 12 };
static uint64_t tc_uvar(tc *t) {
    uint64_t v = 0; int s = 0;
    while (t->p < t->e && s < 64) { const uint8_t b = *t->p++; v |= (uint64_t)(b & 127) << s; if (!(b & 128)) return v; s += 7; }
    t->err = 1; return 0;
}
static int64_t tc_var(tc *t) { const uint64_t u = tc_uvar(t); return (int64_t)(u >> 1) ^ -(int64_t)(u & 1); }
static uint8_t tc_byte(tc *t) { if (t->p >= t->e) { t->err = 1; return 0; } return *t->p++; }
static const uint8_t *tc_bin(tc *t, size_t *n) {
    const uint64_t l = tc_uvar(t);
    if (t->err || l > (uint64_t)(t->e - t->p)) { t->err = 1; *n = 0; return nullptr; }
    const uint8_t *r = t->p; t->p += l; *n = (size_t)l; return r;
}
/* the next field of a struct: 0 at STOP. *last carries the previous field id (delta encoding). Bool fields: *type T_TRUE / T_FALSE. */
static int tc_field(tc *t, int *last, int *id, int *type) {
    const uint8_t b = tc_byte(t); if (t->err || !b) return 0;
    *type = b & 15; const int d = b >> 4;
    *id = d ? *last + d : (int)tc_var(t); *last = *id; return 1;
}
static void tc_list(tc *t, int *et, int64_t *n) {
    const uint8_t b = tc_byte(t); *et = b & 15; int64_t s = b >> 4;
    if (s == 15) s = (int64_t)tc_uvar(t);
    *n = s;
}
static void tc_skip(tc *t, int type, int in_list);
static void tc_skip_struct(tc *t) { int last = 0, id, ty; while (!t->err && tc_field(t, &last, &id, &ty)) tc_skip(t, ty, 0); }
static void tc_skip(tc *t, int type, int in_list) {
    switch (type) {
    case T_TRUE: case T_FALSE: if (in_list) tc_byte(t); break;
    case T_BYTE: tc_byte(t); break;
    case T_I16: case T_I32: case T_I64: tc_uvar(t); break;
    case T_DOUBLE: if (t->e - t->p < 8) t->err = 1; else t->p += 8; break;
    case T_BIN: { size_t n; tc_bin(t, &n); break; }
    case T_LIST: case T_SET: { int et; int64_t n; tc_list(t, &et, &n); for (int64_t i = 0; i < n && !t->err; i++) tc_skip(t, et, 1); break; }
    case T_MAP: { const int64_t n = (int64_t)tc_uvar(t); if (!n) break; const uint8_t kv = tc_byte(t);
                  for (int64_t i = 0; i < n && !t->err; i++) { tc_skip(t, kv >> 4, 1); tc_skip(t, kv & 15, 1); } break; }
    case T_STRUCT: tc_skip_struct(t); break;
    default: t->err = 1;
    }
}

/* ---- file metadata ---- */
typedef struct { int type, type_len, rep, nchild; char name[64]; } pq_sel;
struct pq_file {
    int fd; const uint8_t *map; size_t size;
    int64_t num_rows; char created_by[256];
    int nleaf; pq_leaf *leaf;
    int nrg; int64_t *rg_rows; pq_chunkmeta *chunk;   /* nrg x nleaf */
};

static int walk(pq_file *f, const pq_sel *s, int ns, int *i, int def, int rep, int *rdef, const char *path, const char *top) {
    const pq_sel *e = &s[*i]; (*i)++;
    if (e->rep == 1) def++;
    if (e->rep == 2) { def++; rep++; if (rep > 8) return fail("parquet: nesting deeper than 8"); rdef[rep - 1] = def; }
    char p[256]; snprintf(p, sizeof p, "%s%s%s", path, *path ? "." : "", e->name);
    const char *tp = *top ? top : e->name;
    if (!e->nchild) {
        pq_leaf *l = realloc(f->leaf, sizeof *l * (f->nleaf + 1)); if (!l) return fail("parquet: out of memory");
        f->leaf = l; l = &f->leaf[f->nleaf++]; memset(l, 0, sizeof *l);
        snprintf(l->path, sizeof l->path, "%s", p); snprintf(l->top, sizeof l->top, "%s", tp);
        l->type = e->type; l->type_len = e->type_len; l->max_def = def; l->max_rep = rep;
        for (int r = 0; r < rep; r++) l->rep_def[r] = rdef[r];
        return 0;
    }
    for (int c = 0; c < e->nchild; c++) {
        if (*i >= ns) return fail("parquet: schema tree truncated");
        int rd[8]; memcpy(rd, rdef, sizeof rd);
        if (walk(f, s, ns, i, def, rep, rd, p, tp)) return -1;
    }
    return 0;
}

static int parse_colmeta(tc *t, pq_chunkmeta *m) {
    int last = 0, id, ty; m->dict_off = -1;
    while (tc_field(t, &last, &id, &ty)) {
        if (id == 1 && ty == T_I32) m->type = (int)tc_var(t);
        else if (id == 4 && ty == T_I32) m->codec = (int)tc_var(t);
        else if (id == 5 && ty == T_I64) m->num_values = tc_var(t);
        else if (id == 7 && ty == T_I64) m->comp_size = tc_var(t);
        else if (id == 9 && ty == T_I64) m->data_off = tc_var(t);
        else if (id == 11 && ty == T_I64) m->dict_off = tc_var(t);
        else tc_skip(t, ty, 0);
        if (t->err) return -1;
    }
    return t->err ? -1 : 0;
}

static int parse_meta(pq_file *f, tc *t) {
    int last = 0, id, ty; pq_sel *s = nullptr; int ns = 0;
    while (tc_field(t, &last, &id, &ty)) {
        if (id == 2 && ty == T_LIST) {
            int et; int64_t n; tc_list(t, &et, &n);
            if (et != T_STRUCT || n <= 0 || n > 100000) return fail("parquet: bad schema list");
            s = calloc((size_t)n, sizeof *s); if (!s) return fail("parquet: out of memory"); ns = (int)n;
            for (int k = 0; k < ns; k++) {
                int l2 = 0, i2, t2; s[k].rep = 0;
                while (tc_field(t, &l2, &i2, &t2)) {
                    if (i2 == 1 && t2 == T_I32) s[k].type = (int)tc_var(t);
                    else if (i2 == 2 && t2 == T_I32) s[k].type_len = (int)tc_var(t);
                    else if (i2 == 3 && t2 == T_I32) s[k].rep = (int)tc_var(t);
                    else if (i2 == 4 && t2 == T_BIN) { size_t bn; const uint8_t *b = tc_bin(t, &bn); if (b) snprintf(s[k].name, sizeof s[k].name, "%.*s", (int)bn, (const char *)b); }
                    else if (i2 == 5 && t2 == T_I32) s[k].nchild = (int)tc_var(t);
                    else tc_skip(t, t2, 0);
                    if (t->err) break;
                }
            }
        } else if (id == 3 && ty == T_I64) f->num_rows = tc_var(t);
        else if (id == 4 && ty == T_LIST) {
            if (!s) return fail("parquet: row groups before the schema");
            int i = 1, rd[8] = {0}; s[0].rep = 0;
            for (int c = 0; c < s[0].nchild; c++) { if (i >= ns) { free(s); return fail("parquet: schema tree truncated"); } if (walk(f, s, ns, &i, 0, 0, rd, "", "")) { free(s); return -1; } }
            int et; int64_t n; tc_list(t, &et, &n);
            if (et != T_STRUCT || n < 0 || n > 1000000) { free(s); return fail("parquet: bad row group list"); }
            f->nrg = (int)n; f->rg_rows = calloc((size_t)n + 1, sizeof *f->rg_rows); f->chunk = calloc((size_t)n * f->nleaf + 1, sizeof *f->chunk);
            if (!f->rg_rows || !f->chunk) { free(s); return fail("parquet: out of memory"); }
            for (int g = 0; g < f->nrg; g++) {
                int l2 = 0, i2, t2;
                while (tc_field(t, &l2, &i2, &t2)) {
                    if (i2 == 1 && t2 == T_LIST) {
                        int e2; int64_t nc; tc_list(t, &e2, &nc);
                        if (e2 != T_STRUCT || nc != f->nleaf) { free(s); return fail("parquet: row group %d has %lld columns, schema %d leaves", g, (long long)nc, f->nleaf); }
                        for (int c = 0; c < f->nleaf; c++) {
                            int l3 = 0, i3, t3; int got = 0;
                            while (tc_field(t, &l3, &i3, &t3)) {
                                if (i3 == 1 && t3 == T_BIN) { size_t bn; tc_bin(t, &bn); if (bn) { free(s); return fail("parquet: column chunks in another file are not supported"); } }
                                else if (i3 == 3 && t3 == T_STRUCT) { if (parse_colmeta(t, &f->chunk[(size_t)g * f->nleaf + c])) { free(s); return fail("parquet: bad column metadata"); } got = 1; }
                                else tc_skip(t, t3, 0);
                                if (t->err) break;
                            }
                            if (!got) { free(s); return fail("parquet: column chunk without metadata"); }
                        }
                    } else if (i2 == 3 && t2 == T_I64) f->rg_rows[g] = tc_var(t);
                    else tc_skip(t, t2, 0);
                    if (t->err) break;
                }
            }
        } else if (id == 6 && ty == T_BIN) { size_t bn; const uint8_t *b = tc_bin(t, &bn); if (b) snprintf(f->created_by, sizeof f->created_by, "%.*s", (int)bn, (const char *)b); }
        else tc_skip(t, ty, 0);
        if (t->err) { free(s); return fail("parquet: footer truncated or malformed"); }
    }
    free(s);
    if (t->err) return fail("parquet: footer truncated or malformed");
    if (!f->nleaf) return fail("parquet: no columns");
    return 0;
}

pq_file *pq_open(const char *path) {
    pq_file *f = calloc(1, sizeof *f); if (!f) { fail("parquet: out of memory"); return nullptr; }
    f->fd = open(path, O_RDONLY); struct stat st;
    if (f->fd < 0 || fstat(f->fd, &st)) { fail("parquet: cannot open %s", path); if (f->fd >= 0) close(f->fd); free(f); return nullptr; }
    f->size = (size_t)st.st_size;
    if (f->size < 12) { fail("parquet: %s too small", path); close(f->fd); free(f); return nullptr; }
    f->map = mmap(nullptr, f->size, PROT_READ, MAP_PRIVATE, f->fd, 0);
    if (f->map == MAP_FAILED) { f->map = nullptr; fail("parquet: mmap %s", path); pq_close(f); return nullptr; }
    if (memcmp(f->map, "PAR1", 4) || memcmp(f->map + f->size - 4, "PAR1", 4)) { fail("parquet: %s has no PAR1 magic", path); pq_close(f); return nullptr; }
    const uint8_t *q = f->map + f->size - 8; const uint32_t ml = q[0] | q[1] << 8 | q[2] << 16 | (uint32_t)q[3] << 24;
    if ((size_t)ml + 12 > f->size) { fail("parquet: bad footer length"); pq_close(f); return nullptr; }
    tc t = {q - ml, q, 0};
    if (parse_meta(f, &t)) { pq_close(f); return nullptr; }
    return f;
}
void pq_close(pq_file *f) {
    if (!f) return;
    if (f->map) munmap((void *)f->map, f->size);
    if (f->fd >= 0) close(f->fd);
    free(f->leaf); free(f->rg_rows); free(f->chunk); free(f);
}
int64_t pq_num_rows(const pq_file *f) { return f->num_rows; }
int pq_num_row_groups(const pq_file *f) { return f->nrg; }
int64_t pq_rg_rows(const pq_file *f, int rg) { return f->rg_rows[rg]; }
int pq_num_leaves(const pq_file *f) { return f->nleaf; }
const pq_leaf *pq_leaf_at(const pq_file *f, int i) { return &f->leaf[i]; }
int pq_find_leaf(const pq_file *f, const char *top) { for (int i = 0; i < f->nleaf; i++) if (!strcmp(f->leaf[i].top, top)) return i; return -1; }
const pq_chunkmeta *pq_chunk(const pq_file *f, int rg, int leaf) { return &f->chunk[(size_t)rg * f->nleaf + leaf]; }
const char *pq_created_by(const pq_file *f) { return f->created_by; }
const char *pq_codec_name(int c) {
    static const char *const n[] = {"UNCOMPRESSED", "SNAPPY", "GZIP", "LZO", "BROTLI", "LZ4", "ZSTD", "LZ4_RAW"};
    return c >= 0 && c < 8 ? n[c] : "?";
}

/* ---- codecs ---- */
int pq_snappy_decode(const uint8_t *in, size_t n, uint8_t *out, size_t cap, size_t *outn) {
    size_t i = 0, o = 0; uint64_t len = 0; int s = 0;
    for (;;) { if (i >= n || s > 35) return -1; const uint8_t b = in[i++]; len |= (uint64_t)(b & 127) << s; if (!(b & 128)) break; s += 7; }
    if (len > cap) return -1;
    while (i < n) {
        const uint8_t tag = in[i++]; size_t l, off;
        switch (tag & 3) {
        case 0:
            l = tag >> 2;
            if (l >= 60) { const int nb = (int)l - 59; if (i + nb > n) return -1; l = 0; for (int k = 0; k < nb; k++) l |= (size_t)in[i + k] << (8 * k); i += nb; }
            l++;
            if (i + l > n || o + l > len) return -1;
            memcpy(out + o, in + i, l); i += l; o += l; continue;
        case 1: if (i >= n) return -1; l = ((tag >> 2) & 7) + 4; off = (size_t)(tag >> 5) << 8 | in[i++]; break;
        case 2: if (i + 2 > n) return -1; l = (tag >> 2) + 1; off = in[i] | (size_t)in[i + 1] << 8; i += 2; break;
        default: if (i + 4 > n) return -1; l = (tag >> 2) + 1; off = in[i] | (size_t)in[i + 1] << 8 | (size_t)in[i + 2] << 16 | (size_t)in[i + 3] << 24; i += 4; break;
        }
        if (!off || off > o || o + l > len) return -1;
        for (size_t k = 0; k < l; k++, o++) out[o] = out[o - off];   /* overlapping copies run forward */
    }
    if (o != len) return -1;
    *outn = o; return 0;
}
static int decompress(int codec, const uint8_t *in, size_t n, uint8_t *out, size_t un) {
    switch (codec) {
    case 0: if (n != un) return fail("parquet: uncompressed page size mismatch"); memcpy(out, in, n); return 0;
    case 1: { size_t o; if (pq_snappy_decode(in, n, out, un, &o) || o != un) return fail("parquet: snappy decode"); return 0; }
    case 2: {
        z_stream z; memset(&z, 0, sizeof z);
        if (inflateInit2(&z, 16 + MAX_WBITS) != Z_OK) return fail("parquet: inflateInit");
        z.next_in = (Bytef *)in; z.avail_in = (uInt)n; z.next_out = out; z.avail_out = (uInt)un;
        const int r = inflate(&z, Z_FINISH); const size_t got = z.total_out; inflateEnd(&z);
        if (r != Z_STREAM_END || got != un) return fail("parquet: gzip decode"); return 0;
    }
    case 6: { const size_t r = ZSTD_decompress(out, un, in, n); if (ZSTD_isError(r) || r != un) return fail("parquet: zstd decode"); return 0; }
    default: return fail("parquet: codec %s (%d) not supported", pq_codec_name(codec), codec);
    }
}

/* ---- RLE / bit-packed hybrid: count values into out (uint32), bit width bw; 0 ok ---- */
static int rle_decode(const uint8_t *p, const uint8_t *e, int bw, int64_t count, uint32_t *out) {
    if (bw < 0 || bw > 32) return fail("parquet: bit width %d", bw);
    int64_t o = 0; const int vb = (bw + 7) / 8; const uint64_t mask = bw == 32 ? 0xffffffffu : ((1ull << bw) - 1);
    while (o < count) {
        tc t = {p, e, 0}; const uint64_t h = tc_uvar(&t); if (t.err) return fail("parquet: levels truncated");
        p = t.p;
        if (h & 1) {
            const int64_t nv = (int64_t)(h >> 1) * 8; const size_t nb = (size_t)(h >> 1) * bw;
            if ((size_t)(e - p) < nb) return fail("parquet: bit-packed run truncated");
            for (int64_t k = 0; k < nv && o < count; k++) {
                const uint64_t bit = (uint64_t)k * bw; uint64_t v = 0;   /* bw bits from bit offset `bit`, LSB first */
                for (int b = 0; b < bw; b++) { const uint64_t q = bit + b; v |= (uint64_t)((p[q >> 3] >> (q & 7)) & 1) << b; }
                out[o++] = (uint32_t)(v & mask);
            }
            p += nb;
        } else {
            const int64_t run = (int64_t)(h >> 1); if ((e - p) < vb) return fail("parquet: RLE run truncated");
            uint32_t v = 0; for (int b = 0; b < vb; b++) v |= (uint32_t)p[b] << (8 * b);
            p += vb;
            for (int64_t k = 0; k < run && o < count; k++) out[o++] = v;
            if (!run) return fail("parquet: empty RLE run");
        }
    }
    return 0;
}
static int bits_for(int m) { int b = 0; while ((1 << b) <= m) b++; return m ? b : 0; }
static int elem_size(int type, int tl) {
    switch (type) { case PQ_BOOLEAN: return 1; case PQ_INT32: case PQ_FLOAT: return 4; case PQ_INT64: case PQ_DOUBLE: return 8; case PQ_INT96: return 12; case PQ_FLBA: return tl; default: return 0; }
}

typedef struct { uint8_t *b; size_t n, cap; } buf;
static int buf_reserve(buf *a, size_t n) {
    if (a->n + n > a->cap) { size_t c = a->cap ? a->cap : 4096; while (c < a->n + n) c *= 2; uint8_t *q = realloc(a->b, c); if (!q) return fail("parquet: out of memory"); a->b = q; a->cap = c; }
    return 0;
}
static int buf_put(buf *a, const void *p, size_t n) { if (buf_reserve(a, n)) return -1; memcpy(a->b + a->n, p, n); a->n += n; return 0; }

void pq_col_free(pq_col *c) { free(c->rep); free(c->def); free(c->vals); free(c->offs); free(c->lens); free(c->pool); memset(c, 0, sizeof *c); }

typedef struct {   /* growing output of one column chunk */
    buf rep, def, vals, offs, lens, pool;
} colbuf;

/* PLAIN values: nv of them from p..e */
static int plain_vals(const pq_leaf *L, const uint8_t *p, const uint8_t *e, int64_t nv, colbuf *o) {
    if (L->type == PQ_BYTE_ARRAY) {
        for (int64_t k = 0; k < nv; k++) {
            if (e - p < 4) return fail("parquet: BYTE_ARRAY truncated");
            const uint32_t l = p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24; p += 4;
            if ((uint64_t)(e - p) < l) return fail("parquet: BYTE_ARRAY truncated");
            const int64_t off = (int64_t)o->pool.n; const int32_t ln = (int32_t)l;
            if (buf_put(&o->pool, p, l) || buf_put(&o->offs, &off, 8) || buf_put(&o->lens, &ln, 4)) return -1;
            p += l;
        }
        return 0;
    }
    if (L->type == PQ_BOOLEAN) {
        if ((e - p) * 8 < nv) return fail("parquet: BOOLEAN truncated");
        for (int64_t k = 0; k < nv; k++) { const uint8_t v = (p[k >> 3] >> (k & 7)) & 1; if (buf_put(&o->vals, &v, 1)) return -1; }
        return 0;
    }
    const int es = elem_size(L->type, L->type_len); if (!es) return fail("parquet: type %d", L->type);
    if ((uint64_t)(e - p) < (uint64_t)nv * es) return fail("parquet: PLAIN values truncated (%lld of %d bytes)", (long long)nv, es);
    return buf_put(&o->vals, p, (size_t)nv * es);
}

int pq_read_col(pq_file *f, int rg, int leaf, pq_col *c) {
    memset(c, 0, sizeof *c);
    if (rg < 0 || rg >= f->nrg || leaf < 0 || leaf >= f->nleaf) return fail("parquet: no row group %d / leaf %d", rg, leaf);
    const pq_leaf *L = &f->leaf[leaf]; const pq_chunkmeta *m = pq_chunk(f, rg, leaf);
    const int es = elem_size(L->type, L->type_len);
    if (L->type != PQ_BYTE_ARRAY && !es) return fail("parquet: %s: physical type %d not supported", L->path, L->type);
    int64_t pos = m->dict_off >= 0 && m->dict_off < m->data_off ? m->dict_off : m->data_off;
    colbuf o = {0}; buf dict = {0}, dpool = {0}, doffs = {0}, dlens = {0}; int64_t ndict = -1, got = 0;
    uint8_t *page = nullptr; size_t pcap = 0; uint32_t *tmp = nullptr; size_t tcap = 0; int rc = -1;
    while (got < m->num_values) {
        if (pos < 0 || (size_t)pos >= f->size) { fail("parquet: %s: page offset past the file", L->path); goto out; }
        tc t = {f->map + pos, f->map + f->size, 0};
        int last = 0, id, ty, ptype = -1; int64_t usz = -1, csz = -1;
        int64_t dp_n = -1, dp_enc = -1, dp_denc = 3, dp_renc = 3, dict_n = -1, dict_enc = 0, v2_n = -1, v2_enc = -1, v2_dl = 0, v2_rl = 0; int v2_comp = 1;
        while (tc_field(&t, &last, &id, &ty)) {
            if (id == 1 && ty == T_I32) ptype = (int)tc_var(&t);
            else if (id == 2 && ty == T_I32) usz = tc_var(&t);
            else if (id == 3 && ty == T_I32) csz = tc_var(&t);
            else if ((id == 5 || id == 7 || id == 8) && ty == T_STRUCT) {
                int l2 = 0, i2, t2;
                while (tc_field(&t, &l2, &i2, &t2)) {
                    if (id == 5 && t2 == T_I32 && i2 >= 1 && i2 <= 4) { const int64_t v = tc_var(&t); if (i2 == 1) dp_n = v; else if (i2 == 2) dp_enc = v; else if (i2 == 3) dp_denc = v; else dp_renc = v; }
                    else if (id == 7 && t2 == T_I32 && i2 <= 2) { const int64_t v = tc_var(&t); if (i2 == 1) dict_n = v; else dict_enc = v; }
                    else if (id == 8 && t2 == T_I32 && i2 >= 1 && i2 <= 6) { const int64_t v = tc_var(&t); if (i2 == 1) v2_n = v; else if (i2 == 4) v2_enc = v; else if (i2 == 5) v2_dl = v; else if (i2 == 6) v2_rl = v; }
                    else if (id == 8 && i2 == 7 && (t2 == T_TRUE || t2 == T_FALSE)) v2_comp = t2 == T_TRUE;
                    else tc_skip(&t, t2, 0);
                    if (t.err) break;
                }
            } else tc_skip(&t, ty, 0);
            if (t.err) break;
        }
        if (t.err || ptype < 0 || usz < 0 || csz < 0 || (size_t)(t.e - t.p) < (size_t)csz) { fail("parquet: %s: bad page header at %lld", L->path, (long long)pos); goto out; }
        const uint8_t *body = t.p; pos = (int64_t)(t.p - f->map) + csz;
        if ((size_t)usz > pcap) { free(page); pcap = (size_t)usz + 16; page = malloc(pcap); if (!page) { fail("parquet: out of memory"); goto out; } }
        if (ptype == 2) {   /* dictionary page */
            if (dict_enc != 0 && dict_enc != 2) { fail("parquet: %s: dictionary encoding %lld", L->path, (long long)dict_enc); goto out; }
            if (decompress(m->codec, body, (size_t)csz, page, (size_t)usz)) goto out;
            colbuf d = {0};
            if (plain_vals(L, page, page + usz, dict_n, &d)) { free(d.vals.b); free(d.pool.b); free(d.offs.b); free(d.lens.b); goto out; }
            free(dict.b); free(dpool.b); free(doffs.b); free(dlens.b);
            dict = d.vals; dpool = d.pool; doffs = d.offs; dlens = d.lens; ndict = dict_n;
            continue;
        }
        if (ptype != 0 && ptype != 3) { if (ptype == 1) continue; fail("parquet: %s: page type %d", L->path, ptype); goto out; }   /* 1 = index page: skipped by design */
        const int v2 = ptype == 3; const int64_t nlev = v2 ? v2_n : dp_n; const int64_t enc = v2 ? v2_enc : dp_enc;
        if (nlev < 0 || enc < 0) { fail("parquet: %s: data page header without counts", L->path); goto out; }
        const uint8_t *lp, *le, *vp, *ve;   /* levels region (v1: inside the decompressed body), values region */
        if (!v2) {
            if (decompress(m->codec, body, (size_t)csz, page, (size_t)usz)) goto out;
            lp = page; le = vp = page; ve = page + usz;
        } else {
            if (v2_dl + v2_rl > csz || v2_dl + v2_rl > usz) { fail("parquet: %s: v2 level lengths", L->path); goto out; }
            memcpy(page, body, (size_t)(v2_rl + v2_dl));
            const size_t vcs = (size_t)(csz - v2_rl - v2_dl), vus = (size_t)(usz - v2_rl - v2_dl);
            if (v2_comp && m->codec) { if (decompress(m->codec, body + v2_rl + v2_dl, vcs, page + v2_rl + v2_dl, vus)) goto out; }
            else { if (vcs != vus) { fail("parquet: %s: v2 uncompressed size", L->path); goto out; } memcpy(page + v2_rl + v2_dl, body + v2_rl + v2_dl, vcs); }
            lp = page; le = page + v2_rl + v2_dl; vp = le; ve = page + usz;
        }
        if ((size_t)nlev > tcap) { free(tmp); tcap = (size_t)nlev + 16; tmp = malloc(tcap * 4); if (!tmp) { fail("parquet: out of memory"); goto out; } }
        const size_t r0 = o.rep.n;
        /* repetition then definition levels */
        for (int lv = 0; lv < 2; lv++) {
            const int mx = lv ? L->max_def : L->max_rep; buf *dst = lv ? &o.def : &o.rep;
            if (buf_reserve(dst, (size_t)nlev)) goto out;
            if (!mx) { memset(dst->b + dst->n, 0, (size_t)nlev); dst->n += (size_t)nlev; continue; }
            const uint8_t *a, *b;
            if (!v2) {
                if ((lv ? dp_denc : dp_renc) != 3) { fail("parquet: %s: level encoding %lld", L->path, (long long)(lv ? dp_denc : dp_renc)); goto out; }
                if (ve - vp < 4) { fail("parquet: %s: levels truncated", L->path); goto out; }
                const uint32_t ln = vp[0] | vp[1] << 8 | vp[2] << 16 | (uint32_t)vp[3] << 24;
                if ((uint64_t)(ve - vp - 4) < ln) { fail("parquet: %s: levels truncated", L->path); goto out; }
                a = vp + 4; b = a + ln; vp = b;
            } else { a = lv ? lp + v2_rl : lp; b = lv ? le : lp + v2_rl; }
            if (rle_decode(a, b, bits_for(mx), nlev, tmp)) goto out;
            for (int64_t k = 0; k < nlev; k++) { if (tmp[k] > (uint32_t)mx) { fail("parquet: %s: level %u > max %d", L->path, tmp[k], mx); goto out; } dst->b[dst->n + k] = (uint8_t)tmp[k]; }
            dst->n += (size_t)nlev;
        }
        (void)le;
        int64_t nv = 0; for (int64_t k = 0; k < nlev; k++) nv += o.def.b[r0 + k] == L->max_def;
        if (enc == 0) { if (plain_vals(L, vp, ve, nv, &o)) goto out; }
        else if (enc == 2 || enc == 8) {
            if (ndict < 0) { fail("parquet: %s: dictionary-encoded page without a dictionary", L->path); goto out; }
            if (nv) {
                if (ve - vp < 1) { fail("parquet: %s: dictionary indices truncated", L->path); goto out; }
                const int bw = *vp++;
                if ((size_t)nv > tcap) { free(tmp); tcap = (size_t)nv + 16; tmp = malloc(tcap * 4); if (!tmp) { fail("parquet: out of memory"); goto out; } }
                if (rle_decode(vp, ve, bw, nv, tmp)) goto out;
                for (int64_t k = 0; k < nv; k++) {
                    if (tmp[k] >= (uint64_t)ndict) { fail("parquet: %s: dictionary index %u of %lld", L->path, tmp[k], (long long)ndict); goto out; }
                    if (L->type == PQ_BYTE_ARRAY) {
                        const int64_t so = ((int64_t *)doffs.b)[tmp[k]]; const int32_t sl = ((int32_t *)dlens.b)[tmp[k]]; const int64_t off = (int64_t)o.pool.n;
                        if (buf_put(&o.pool, dpool.b + so, (size_t)sl) || buf_put(&o.offs, &off, 8) || buf_put(&o.lens, &sl, 4)) goto out;
                    } else if (buf_put(&o.vals, dict.b + (size_t)tmp[k] * es, (size_t)es)) goto out;
                }
            }
        } else { fail("parquet: %s: value encoding %lld not supported", L->path, (long long)enc); goto out; }
        got += nlev;
    }
    c->n = (int64_t)o.rep.n; c->rep = o.rep.b; c->def = o.def.b; o.rep.b = o.def.b = nullptr;
    c->nv = L->type == PQ_BYTE_ARRAY ? (int64_t)(o.lens.n / 4) : (int64_t)(o.vals.n / (es ? es : 1));
    c->vals = o.vals.b; c->offs = (int64_t *)o.offs.b; c->lens = (int32_t *)o.lens.b; c->pool = o.pool.b; o.vals.b = o.offs.b = o.lens.b = o.pool.b = nullptr;
    if (!c->rep) { c->rep = calloc(1, 1); c->def = calloc(1, 1); }
    rc = 0;
out:
    free(o.rep.b); free(o.def.b); free(o.vals.b); free(o.offs.b); free(o.lens.b); free(o.pool.b);
    free(dict.b); free(dpool.b); free(doffs.b); free(dlens.b); free(page); free(tmp);
    if (rc) pq_col_free(c);
    return rc;
}
