/* src/labcomp.c: label codecs (labcomp.h). */
#include "labcomp.h"
#include <stdlib.h>
#include <string.h>
#include "volcomp.h"

/* ---- binary range coder (LZMA style: 32-bit range, carry through the cache byte), 15-bit adaptive probabilities ---- */
enum { RC_BITS = 15, RC_ONE = 1 << RC_BITS, RC_SHIFT = 5, RC_TOP = 1u << 24 };
typedef uint16_t prob;
typedef struct { uint64_t low; uint32_t range; uint8_t cache; uint64_t csize; uint8_t *b; size_t n, cap; int oom; } rcenc;
typedef struct { uint32_t code, range; const uint8_t *p, *e; int bad; } rcdec;

static void re_byte(rcenc *e, uint8_t v) {
    if (e->n == e->cap) { size_t c = e->cap ? 2 * e->cap : 1 << 16; uint8_t *nb = realloc(e->b, c); if (!nb) { e->oom = 1; return; } e->b = nb; e->cap = c; }
    e->b[e->n++] = v;
}
static void re_shift(rcenc *e) {
    if ((uint32_t)e->low < 0xFF000000u || (e->low >> 32)) {
        const uint8_t carry = (uint8_t)(e->low >> 32); uint8_t t = e->cache;
        do { re_byte(e, (uint8_t)(t + carry)); t = 0xFF; } while (--e->csize);
        e->cache = (uint8_t)(e->low >> 24);
    }
    e->csize++; e->low = (e->low & 0x00FFFFFFu) << 8;
}
static void re_init(rcenc *e) { memset(e, 0, sizeof *e); e->range = 0xFFFFFFFFu; e->csize = 1; }
static inline void re_bit(rcenc *e, prob *p, int bit) {
    const uint32_t bound = (e->range >> RC_BITS) * *p;
    if (!bit) { e->range = bound; *p += (RC_ONE - *p) >> RC_SHIFT; }
    else { e->low += bound; e->range -= bound; *p -= *p >> RC_SHIFT; }
    while (e->range < RC_TOP) { e->range <<= 8; re_shift(e); }
}
static void re_flush(rcenc *e) { for (int i = 0; i < 5; i++) re_shift(e); }
static void rd_init(rcdec *d, const uint8_t *p, size_t n) {
    d->p = p; d->e = p + n; d->code = 0; d->range = 0xFFFFFFFFu; d->bad = 0;
    for (int i = 0; i < 5; i++) d->code = d->code << 8 | (d->p < d->e ? *d->p++ : (d->bad = 1, 0));
}
static inline int rd_bit(rcdec *d, prob *p) {
    const uint32_t bound = (d->range >> RC_BITS) * *p; int bit;
    if (d->code < bound) { d->range = bound; *p += (RC_ONE - *p) >> RC_SHIFT; bit = 0; }
    else { d->code -= bound; d->range -= bound; *p -= *p >> RC_SHIFT; bit = 1; }
    while (d->range < RC_TOP) { d->range <<= 8; d->code = d->code << 8 | (d->p < d->e ? *d->p++ : 0); }
    return bit;
}
static void pinit(prob *p, size_t n) { for (size_t i = 0; i < n; i++) p[i] = RC_ONE / 2; }

/* ---- stream header ---- */
enum { HDR = 8 };   /* "LBC" kind version(1) reserved(3) */
static void put_hdr(uint8_t *b, int kind) { b[0] = 'L'; b[1] = 'B'; b[2] = 'C'; b[3] = (uint8_t)kind; b[4] = 1; b[5] = b[6] = b[7] = 0; }
int lbc_kind(const void *enc, size_t n) {
    const uint8_t *b = enc;
    return n >= HDR && b[0] == 'L' && b[1] == 'B' && b[2] == 'C' && b[4] == 1 ? b[3] : -1;
}
const char *lbc_codec_json(int kind) {
    switch (kind) { case LBC_KIND_CODES: return "{\"name\":\"labcomp\",\"configuration\":{\"kind\":\"codes\"}}"; }
    return nullptr;
}

/* ---- kind 1: winding codes ---- */
enum { C = LBC_CHUNK };
typedef struct { prob eq[2][3], eq2[3], lit[2][256]; } codes_model;
static void cm_init(codes_model *m) { pinit(&m->eq[0][0], 6); pinit(m->eq2, 3); pinit(&m->lit[0][0], 512); }
/* the majority (best) and second most frequent (second) code among the nonzero causal 26-neighbours of voxel (z, y, x) of v; nv their
   count; agree: all of them equal best. -1 where absent. */
static inline void neigh(const uint8_t *v, int z, int y, int x, int *best, int *second, int *nv, int *agree) {
    int val[13], cnt[13], k = 0, n = 0;
    static const int8_t off[13][3] = {{-1, -1, -1}, {-1, -1, 0}, {-1, -1, 1}, {-1, 0, -1}, {-1, 0, 0}, {-1, 0, 1}, {-1, 1, -1}, {-1, 1, 0}, {-1, 1, 1},
                                      {0, -1, -1}, {0, -1, 0}, {0, -1, 1}, {0, 0, -1}};
    for (int i = 0; i < 13; i++) {
        const int zz = z + off[i][0], yy = y + off[i][1], xx = x + off[i][2];
        if (zz < 0 || yy < 0 || xx < 0 || yy >= C || xx >= C) continue;
        const int w = v[((size_t)zz * C + yy) * C + xx]; if (!w) continue;
        n++; int j = 0; for (; j < k; j++) if (val[j] == w) { cnt[j]++; break; }
        if (j == k) { val[k] = w; cnt[k] = 1; k++; }
    }
    *nv = n; *best = *second = -1; *agree = k == 1;
    int b = -1, s = -1;
    for (int j = 0; j < k; j++) {   /* ties: the smaller code (deterministic) */
        if (b < 0 || cnt[j] > cnt[b] || (cnt[j] == cnt[b] && val[j] < val[b])) { s = b; b = j; }
        else if (s < 0 || cnt[j] > cnt[s] || (cnt[j] == cnt[s] && val[j] < val[s])) s = j;
    }
    if (b >= 0) *best = val[b];
    if (s >= 0) *second = val[s];
}
static inline int nvb(int nv) { return nv <= 2 ? 0 : nv <= 5 ? 1 : 2; }

int lbc_codes_encode(const uint8_t *src, uint8_t **out, size_t *out_n) {
    *out = nullptr; *out_n = 0;
    uint8_t *mk = malloc(LBC_VOXELS), *mb = malloc(VOLCOMP_MASK_LL_BOUND); codes_model *m = malloc(sizeof *m);
    if (!mk || !mb || !m) { free(mk); free(mb); free(m); return -1; }
    for (size_t i = 0; i < LBC_VOXELS; i++) mk[i] = src[i] ? 255 : 0;
    size_t mn = 0;
    if (volcomp_mask_encode_lossless(mk, mb, VOLCOMP_MASK_LL_BOUND, &mn) != VOLCOMP_OK) { free(mk); free(mb); free(m); return -1; }
    free(mk);
    cm_init(m); rcenc e; re_init(&e);
    for (int z = 0; z < C; z++) for (int y = 0; y < C; y++) for (int x = 0; x < C; x++) {
        const int v = src[((size_t)z * C + y) * C + x]; if (!v) continue;
        int best, second, nv, agree; neigh(src, z, y, x, &best, &second, &nv, &agree);
        int hit = 0;
        if (best >= 0) { hit = v == best; re_bit(&e, &m->eq[agree][nvb(nv)], hit); }
        if (!hit && second >= 0) { hit = v == second; re_bit(&e, &m->eq2[nvb(nv)], hit); }
        if (!hit) { prob *t = m->lit[best >= 0]; int c = 1; for (int b = 7; b >= 0; b--) { const int bit = v >> b & 1; re_bit(&e, &t[c], bit); c = c << 1 | bit; } }
    }
    re_flush(&e); free(m);
    if (e.oom) { free(e.b); free(mb); return -1; }
    const size_t n = HDR + 4 + mn + e.n; uint8_t *o = malloc(n);
    if (!o) { free(e.b); free(mb); return -1; }
    put_hdr(o, LBC_KIND_CODES); const uint32_t m32 = (uint32_t)mn; memcpy(o + HDR, &m32, 4); memcpy(o + HDR + 4, mb, mn); memcpy(o + HDR + 4 + mn, e.b, e.n);
    free(e.b); free(mb); *out = o; *out_n = n; return 0;
}

static int codes_decode(const uint8_t *b, size_t n, uint8_t *dst) {
    if (n < HDR + 4) return -1;
    uint32_t mn; memcpy(&mn, b + HDR, 4); if ((size_t)mn > n - HDR - 4) return -1;
    if (volcomp_decode(b + HDR + 4, mn, dst, LBC_VOXELS) != VOLCOMP_OK) return -1;
    codes_model *m = malloc(sizeof *m); if (!m) return -1;
    cm_init(m); rcdec d; rd_init(&d, b + HDR + 4 + mn, n - HDR - 4 - mn);
    for (int z = 0; z < C; z++) for (int y = 0; y < C; y++) for (int x = 0; x < C; x++) {
        uint8_t *v = dst + ((size_t)z * C + y) * C + x; if (!*v) continue;
        int best, second, nv, agree; neigh(dst, z, y, x, &best, &second, &nv, &agree);
        int val = -1;
        if (best >= 0 && rd_bit(&d, &m->eq[agree][nvb(nv)])) val = best;
        if (val < 0 && second >= 0 && rd_bit(&d, &m->eq2[nvb(nv)])) val = second;
        if (val < 0) { prob *t = m->lit[best >= 0]; int c = 1; for (int k = 0; k < 8; k++) c = c << 1 | rd_bit(&d, &t[c]); val = c & 255; }
        if (!val) { free(m); return -1; }   /* a support voxel decoded as 0: corrupt */
        *v = (uint8_t)val;
    }
    free(m);
    return d.bad ? -1 : 0;
}

int lbc_decode(const void *enc, size_t n, uint8_t *dst) {
    switch (lbc_kind(enc, n)) {
    case LBC_KIND_CODES: return codes_decode(enc, n, dst);
    }
    return -1;
}
