/* Reader of the dinovol-embed zarr v3 arrays (sharded, zstd, channel-last; int8 / float16 / float32): any region dequantised to float32
   (value_scale, and the <name>_scale per-vector array when present). Shared by dinovol-read / dinovol-eval (src/dinovol.c) and the
   fitter's embedding evidence (src/fit_emb.c). Not thread-safe per handle (one decoded-chunk cache). */
#include "embz.h"
#include "json.h"
#include <zstd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *read_file(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb"); if (!f) return nullptr;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    void *p = malloc(sz > 0 ? (size_t)sz : 1); if (!p || fread(p, 1, (size_t)sz, f) != (size_t)sz) { free(p); fclose(f); return nullptr; }
    fclose(f); *n = (size_t)sz; return p;
}
float ez_h2f(uint16_t h) {
    const uint32_t s = (uint32_t)(h & 0x8000u) << 16; int e = (h >> 10) & 0x1f; uint32_t m = h & 0x3ffu, x;
    if (e == 0) { if (!m) x = s; else { while (!(m & 0x400u)) { m <<= 1; e--; } e++; m &= 0x3ffu; x = s | ((uint32_t)(e + 112) << 23) | (m << 13); } }
    else if (e == 31) x = s | 0x7f800000u | (m << 13);
    else x = s | ((uint32_t)(e + 112) << 23) | (m << 13);
    float f; memcpy(&f, &x, 4); return f;
}
#define h2f ez_h2f

int zr_open(zr *r, const char *dir) {
    memset(r, 0, sizeof *r); snprintf(r->dir, sizeof r->dir, "%s", dir); r->ish[0] = -1; r->csh = -1;
    char p[1200]; snprintf(p, sizeof p, "%s/zarr.json", dir); size_t n = 0; char *t = read_file(p, &n);
    if (!t) return -1;
    json *j = json_parse(t, n); free(t); if (!j) return -1;
    const json *sh = json_get(j, "shape"), *dtj = json_get(j, "data_type"), *cs = json_path(j, "chunk_grid.configuration.chunk_shape"), *ic = json_path(j, "codecs.0.configuration.chunk_shape");
    if (!sh || !dtj || !cs || !ic) { json_free(j); return -1; }
    for (int d = 0; d < 3; d++) r->shape[d] = (int64_t)json_num(json_at(sh, d), 0);
    r->C = sh->n > 3 ? (int)json_num(json_at(sh, 3), 0) : 0;
    r->shard = (int)json_num(json_at(cs, 0), 0); r->chunk = (int)json_num(json_at(ic, 0), 0);
    const char *dt = json_str(dtj, ""); r->dt = !strcmp(dt, "int8") ? 0 : !strcmp(dt, "float16") ? 1 : !strcmp(dt, "float32") ? 2 : -1; r->esize = r->dt == 0 ? 1 : r->dt == 1 ? 2 : 4;
    r->vscale = json_num(json_path(j, "attributes.value_scale"), 1.0);
    json_free(j);
    if (r->dt < 0 || r->chunk <= 0 || r->shard % r->chunk) return -1;
    const size_t cb = (size_t)r->chunk * r->chunk * r->chunk * (r->C ? r->C : 1) * r->esize; r->cb = malloc(cb);
    return r->cb ? 0 : -1;
}
void zr_close(zr *r) { free(r->idx); free(r->cb); }
/* the decoded chunk holding output cell (z, y, x), or nullptr when absent (zeros) */
const uint8_t *zr_chunk(zr *r, int64_t z, int64_t y, int64_t x) {
    const int64_t s[3] = {z / r->shard, y / r->shard, x / r->shard}; const int nc1 = r->shard / r->chunk, nc = nc1 * nc1 * nc1;
    const int64_t ci = ((z % r->shard) / r->chunk * nc1 + (y % r->shard) / r->chunk) * nc1 + (x % r->shard) / r->chunk;
    char p[1300];
    if (r->C) snprintf(p, sizeof p, "%s/c/%lld/%lld/%lld/0", r->dir, (long long)s[0], (long long)s[1], (long long)s[2]);
    else snprintf(p, sizeof p, "%s/c/%lld/%lld/%lld", r->dir, (long long)s[0], (long long)s[1], (long long)s[2]);
    if (memcmp(s, r->ish, sizeof s)) {   /* shard index */
        free(r->idx); r->idx = nullptr; memcpy(r->ish, s, sizeof s); r->csh = -1;
        FILE *f = fopen(p, "rb");
        if (f) { r->idx = malloc((size_t)nc * 16); fseek(f, -(long)(nc * 16 + 4), SEEK_END); if (fread(r->idx, 16, nc, f) != (size_t)nc) { free(r->idx); r->idx = nullptr; } fclose(f); }
    }
    if (!r->idx) return nullptr;
    uint64_t off, len; memcpy(&off, r->idx + ci * 16, 8); memcpy(&len, r->idx + ci * 16 + 8, 8);
    if (off == UINT64_MAX) return nullptr;
    if (r->csh == 0 && r->cci == ci && !memcmp(r->cshard, s, sizeof s)) return r->cb;
    FILE *f = fopen(p, "rb"); if (!f) return nullptr;
    void *zb = malloc(len); fseek(f, (long)off, SEEK_SET);
    const size_t got = fread(zb, 1, len, f); fclose(f);
    const size_t cb = (size_t)r->chunk * r->chunk * r->chunk * (r->C ? r->C : 1) * r->esize;
    if (got != len || ZSTD_decompress(r->cb, cb, zb, len) != cb) { free(zb); return nullptr; }
    free(zb); r->csh = 0; r->cci = ci; memcpy(r->cshard, s, sizeof s);
    return r->cb;
}
/* float32 [n0][n1][n2](C) of the region at o (output cells); outside the array or absent: 0 */
int zr_read(zr *r, const int64_t o[3], const int64_t n[3], float *out) {
    const int C = r->C ? r->C : 1, c = r->chunk;
    for (int64_t z = 0; z < n[0]; z++) for (int64_t y = 0; y < n[1]; y++) for (int64_t x = 0; x < n[2]; x++) {
        float *dst = out + ((z * n[1] + y) * n[2] + x) * C;
        const int64_t gz = o[0] + z, gy = o[1] + y, gx = o[2] + x;
        const uint8_t *ch = gz >= 0 && gy >= 0 && gx >= 0 && gz < r->shape[0] && gy < r->shape[1] && gx < r->shape[2] ? zr_chunk(r, gz, gy, gx) : nullptr;
        if (!ch) { memset(dst, 0, C * 4); continue; }
        const size_t e = ((size_t)((gz % c) * c + gy % c) * c + gx % c) * C;
        for (int k = 0; k < C; k++) {
            if (r->dt == 0) dst[k] = (float)(((const int8_t *)ch)[e + k] * r->vscale);
            else if (r->dt == 1) { uint16_t h; memcpy(&h, ch + (e + k) * 2, 2); dst[k] = (float)(h2f(h) * r->vscale); }
            else { float v; memcpy(&v, ch + (e + k) * 4, 4); dst[k] = (float)(v * r->vscale); }
        }
    }
    return 0;
}
/* an output with its optional per-vector scale array */

int emb_open(emb *e, const char *group, const char *name) {
    char d[1200]; snprintf(d, sizeof d, "%s/%s", group, name);
    if (zr_open(&e->a, d)) { fprintf(stderr, "dinovol: cannot open %s\n", d); return -1; }
    snprintf(d, sizeof d, "%s/%s_scale", group, name); e->has_s = zr_open(&e->s, d) == 0;
    return 0;
}
void emb_close(emb *e) { zr_close(&e->a); if (e->has_s) zr_close(&e->s); }
int emb_read(emb *e, const int64_t o[3], const int64_t n[3], float *out) {
    if (zr_read(&e->a, o, n, out)) return -1;
    if (e->has_s) {
        const size_t nv = (size_t)n[0] * n[1] * n[2]; float *s = malloc(nv * 4);
        zr_read(&e->s, o, n, s);
        for (size_t i = 0; i < nv; i++) for (int k = 0; k < e->a.C; k++) out[i * e->a.C + k] *= s[i];
        free(s);
    }
    return 0;
}
