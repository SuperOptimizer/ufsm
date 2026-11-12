/* latvol arrays read as cells: see latvol_read.h */
#include "latvol_read.h"
#include "latvol.h"
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    z3 *z; const int64_t *o, *n; uint8_t *buf; size_t cap; lv_chunk_ref *refs; int maxref;
    int64_t c0[3], cn[3]; atomic_long next; _Atomic size_t used; atomic_int nref, err;
} rb_job;
static void *rb_run(void *arg) {
    rb_job *j = arg; const long total = (long)(j->cn[0] * j->cn[1] * j->cn[2]);
    for (long k; (k = atomic_fetch_add(&j->next, 1)) < total && !atomic_load(&j->err);) {
        const int64_t cz = j->c0[0] + k / (j->cn[1] * j->cn[2]), cy = j->c0[1] + k / j->cn[2] % j->cn[1], cx = j->c0[2] + k % j->cn[2];
        uint8_t *raw = nullptr; size_t len = 0;
        const int r = z3_chunk_raw(j->z, cz, cy, cx, &raw, &len);
        if (r < 0) { atomic_store(&j->err, 1); break; }
        if (!r) continue;
        if (len < LV_HDR || (len & 3)) { free(raw); atomic_store(&j->err, 1); break; }
        /* the chunk's first cell relative to the box, the substreams that meet the box */
        const int rz = (int)((cz * 128 - j->o[0]) / 4), ry = (int)((cy * 128 - j->o[1]) / 4), rx = (int)((cx * 128 - j->o[2]) / 4);
        const int d = (int)(j->n[0] / 4), h = (int)(j->n[1] / 4), w = (int)(j->n[2] / 4);
        uint16_t words[LV_NSUB], keep[LV_NSUB]; memcpy(words, raw + 8, sizeof words); size_t need = LV_HDR, tot = LV_HDR; int any = 0;
        for (int s = 0; s < LV_NSUB; s++) {
            const int z0 = rz + 4 * (s >> 6), y0 = ry + 4 * (s >> 3 & 7), x0 = rx + 4 * (s & 7);
            const int in = z0 + 4 > 0 && y0 + 4 > 0 && x0 + 4 > 0 && z0 < d && y0 < h && x0 < w;
            keep[s] = in ? words[s] : 0; need += (size_t)keep[s] * 4; tot += (size_t)words[s] * 4; any |= keep[s] != 0;
        }
        if (tot != len) { free(raw); atomic_store(&j->err, 1); break; }
        if (!any) { free(raw); continue; }
        const size_t off = atomic_fetch_add(&j->used, need); const int ri = atomic_fetch_add(&j->nref, 1);
        if (off + need > j->cap || ri >= j->maxref) { free(raw); atomic_store(&j->err, 2); break; }
        uint8_t *dst = j->buf + off; memcpy(dst, raw, 8); memcpy(dst + 8, keep, sizeof keep);
        size_t so = LV_HDR, dp = LV_HDR;
        for (int s = 0; s < LV_NSUB; s++) { if (keep[s]) { memcpy(dst + dp, raw + so, (size_t)keep[s] * 4); dp += (size_t)keep[s] * 4; } so += (size_t)words[s] * 4; }
        j->refs[ri] = (lv_chunk_ref){(uint32_t)off, (uint32_t)need, rz, ry, rx};
        free(raw);
    }
    return nullptr;
}
int lv_read_box(z3 *z, const int64_t o[3], const int64_t n[3], uint8_t *buf, size_t cap, lv_chunk_ref *refs, int maxref, size_t *bytes, int nthreads) {
    const z3_meta *m = z3_meta_of(z);
    rb_job j = {.z = z, .o = o, .n = n, .buf = buf, .cap = cap, .refs = refs, .maxref = maxref};
    for (int d = 0; d < 3; d++) {
        if ((o[d] & 3) || (n[d] & 3) || n[d] <= 0) return -1;
        int64_t lo = o[d] < 0 ? 0 : o[d], hi = o[d] + n[d] > m->shape[d] ? m->shape[d] : o[d] + n[d];
        if (hi <= lo) { *bytes = 0; return 0; }
        j.c0[d] = lo / 128; j.cn[d] = (hi - 1) / 128 - j.c0[d] + 1;
    }
    if (cap > 0xffffffffu) j.cap = 0xfffffffcu;   /* (32-bit stream offsets) */
    const long total = (long)(j.cn[0] * j.cn[1] * j.cn[2]); int nt = nthreads < 1 ? 1 : nthreads > 64 ? 64 : nthreads; if (nt > total) nt = (int)total;
    pthread_t th[64];
    for (int t = 1; t < nt; t++) pthread_create(&th[t], nullptr, rb_run, &j);
    rb_run(&j);
    for (int t = 1; t < nt; t++) pthread_join(th[t], nullptr);
    const int e = atomic_load(&j.err);
    if (e) return e == 2 ? -2 : -1;
    *bytes = atomic_load(&j.used);
    return atomic_load(&j.nref);
}
int lv_cells_host(const void *codec, const uint8_t *buf, const lv_chunk_ref *refs, int nref, int d, int h, int w, int stride, int8_t *cells) {
    const lv_codec *lc = codec; const int C = lc->C;
    memset(cells, 0, (size_t)d * h * w * stride);
    int16_t *q = malloc((size_t)LV_CELLS * LV_CELLS * LV_CELLS * C * sizeof *q); if (!q) return -1;
    for (int i = 0; i < nref; i++) {
        if (lv_decode_chunk(lc, buf + refs[i].off, refs[i].len, q, nullptr)) { free(q); return -1; }
        for (int cz = 0; cz < LV_CELLS; cz++) for (int cy = 0; cy < LV_CELLS; cy++) for (int cx = 0; cx < LV_CELLS; cx++) {
            const int z = refs[i].cz + cz, y = refs[i].cy + cy, x = refs[i].cx + cx;
            if (z < 0 || y < 0 || x < 0 || z >= d || y >= h || x >= w) continue;
            const int16_t *o = q + ((size_t)(cz * LV_CELLS + cy) * LV_CELLS + cx) * C;
            if (o[0]) lv_pack_cells(o, 1, C, cells + (((size_t)z * h + y) * w + x) * stride, stride);
        }
    }
    free(q);
    return 0;
}
