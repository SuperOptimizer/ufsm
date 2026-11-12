/* latvol arrays read as cells: see latvol_read.h */
#include "latvol_read.h"
#include "latvol.h"
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

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

/* ---- the voxel reader's GPU path (latvol_read.h) ---- */
static _Atomic int g_lvgpu = -2;   /* -2 undecided, -1 off, >= 0 the device */
static _Atomic uint64_t g_lvboxes;
uint64_t latvol_gpu_boxes(void) { return atomic_load(&g_lvboxes); }
void latvol_gpu_config(int dev, int slots) {
    if (dev < 0 || !nn_latvol_box || !nn_latvol_box_config) { atomic_store(&g_lvgpu, -1); return; }
    nn_latvol_box_config(dev, slots); atomic_store(&g_lvgpu, dev);
}
static int lvgpu_on(void) {
    int g = atomic_load(&g_lvgpu); if (g != -2) return g >= 0;
    const char *c = getenv("UFSM_LATVOL_CPU"), *d = getenv("UFSM_LATVOL_GPU"), *sl = getenv("UFSM_LATVOL_SLOTS");
    if ((c && atoi(c)) || !d || !*d) latvol_gpu_config(-1, 0); else latvol_gpu_config(atoi(d), sl ? atoi(sl) : 2);
    return atomic_load(&g_lvgpu) >= 0;
}
int lv_box_voxels_gpu(z3 *z, const int64_t o[3], const int64_t n[3], uint8_t *out, int nthreads) {
    const lv_codec *lc = z3_latvol(z); if (!lc || !lvgpu_on()) return 1;
    const z3_meta *m = z3_meta_of(z);
    int64_t lo[3], hi[3], a[3], an[3]; int inside = 1, whole = 1;
    for (int d = 0; d < 3; d++) {
        lo[d] = o[d] < 0 ? 0 : o[d]; hi[d] = o[d] + n[d] > m->shape[d] ? m->shape[d] : o[d] + n[d]; inside &= lo[d] < hi[d];
        a[d] = lo[d] / 4 * 4; an[d] = (hi[d] + 3) / 4 * 4 - a[d]; whole &= a[d] == o[d] && an[d] == n[d];
    }
    if (!inside) return 1;
    const size_t av = (size_t)an[0] * an[1] * an[2];
    if (av < (size_t)64 * 64 * 64) return 1;   /* (a small box: the launch and the copies cost more than the host decode) */
    const int maxref = (int)((an[0] / 128 + 2) * (an[1] / 128 + 2) * (an[2] / 128 + 2));
    uint8_t *comp = malloc(av); lv_chunk_ref *refs = malloc((size_t)maxref * sizeof *refs); uint8_t *tmp = whole ? out : malloc(av); size_t bytes = 0; int rc = 1;
    if (!comp || !refs || !tmp) { rc = -1; goto done; }
    const int nr = lv_read_box(z, a, an, comp, av, refs, maxref, &bytes, nthreads);
    if (nr == -2) goto done;
    if (nr < 0) { rc = -1; goto done; }
    if (!whole) memset(out, m->fill, (size_t)n[0] * n[1] * n[2]);
    if (!nr) { if (whole) memset(out, 0, av); rc = 0; goto done; }
    const int g = nn_latvol_box(lc, lc->dt, lc->ntab, lc->map, &lc->bi[0][0], lc->C, lc->step_raw, comp, bytes, refs, nr, (int)(an[0] / 4), (int)(an[1] / 4), (int)(an[2] / 4), tmp);
    if (g) { rc = 1; goto done; }   /* (no device, or a corrupt stream: the host decoder reports it) */
    if (!whole)
        for (int64_t zz = lo[0]; zz < hi[0]; zz++) for (int64_t yy = lo[1]; yy < hi[1]; yy++)
            memcpy(out + ((size_t)(zz - o[0]) * n[1] + (size_t)(yy - o[1])) * n[2] + (size_t)(lo[2] - o[2]), tmp + ((size_t)(zz - a[0]) * an[1] + (size_t)(yy - a[1])) * an[2] + (size_t)(lo[2] - a[2]), (size_t)(hi[2] - lo[2]));
    if (!atomic_fetch_add(&g_lvboxes, 1)) fprintf(stderr, "latvol: voxel reads of latvol arrays are decoded on GPU %d\n", atomic_load(&g_lvgpu));
    rc = 0;
done:
    free(comp); free(refs); if (!whole) free(tmp);
    return rc;
}
