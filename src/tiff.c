#include "tiff.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>

static _Thread_local char errbuf[256];
const char *tiff_error(void) { return errbuf; }
static int fail(const char *fmt, ...) { va_list ap; va_start(ap, fmt); vsnprintf(errbuf, sizeof errbuf, fmt, ap); va_end(ap); return -1; }

typedef struct {
    tiff_page p;
    int compression, predictor, planar;
    uint32_t rows_per_strip, tile_w, tile_h;
    uint32_t nstrips;                 /* strips or tiles */
    uint64_t *offsets, *counts;
    char *description;
} page;

struct tiff {
    const uint8_t *buf;
    size_t n;
    uint8_t *own;
    int le, big;      /* big: BigTIFF (8-byte offsets, 20-byte entries) */
    int npages;
    page *pages;
};

static uint16_t rd16(const tiff *t, size_t o) { const uint8_t *p = t->buf + o; return t->le ? (uint16_t)(p[0] | p[1] << 8) : (uint16_t)(p[1] | p[0] << 8); }
static uint32_t rd32(const tiff *t, size_t o) { const uint8_t *p = t->buf + o; return t->le ? (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24 : (uint32_t)p[3] | (uint32_t)p[2] << 8 | (uint32_t)p[1] << 16 | (uint32_t)p[0] << 24; }

static const int type_size[] = {0, 1, 1, 2, 4, 8, 1, 1, 2, 4, 8, 4, 8, 4, 1, 1, 8, 8, 8};
static uint64_t rd64(const tiff *t, size_t o) { return t->le ? (uint64_t)rd32(t, o) | (uint64_t)rd32(t, o + 4) << 32 : (uint64_t)rd32(t, o) << 32 | rd32(t, o + 4); }
static uint64_t rd_off(const tiff *t, size_t o) { return t->big ? rd64(t, o) : rd32(t, o); }
static uint64_t rd_cnt(const tiff *t, size_t o) { return t->big ? rd64(t, o) : rd32(t, o); }
static size_t entry_size(const tiff *t) { return t->big ? 20 : 12; }

/* read tag value i (as uint64) from an entry at offset e */
static uint64_t tag_val(const tiff *t, size_t e, uint32_t i) {
    uint16_t type = rd16(t, e + 2);
    uint64_t count = rd_cnt(t, e + 4);
    int ts = type < 19 ? type_size[type] : 1;
    size_t inl = t->big ? 8 : 4, voff = t->big ? 12 : 8;
    size_t data = (size_t)count * ts <= inl ? e + voff : (size_t)rd_off(t, e + voff);
    size_t o = data + (size_t)i * ts;
    if (o + ts > t->n) return 0;
    switch (type) {
    case 3: return rd16(t, o);
    case 4: return rd32(t, o);
    case 1: case 2: case 6: case 7: return t->buf[o];
    case 16: return rd64(t, o);
    default: return rd32(t, o);
    }
}

static int parse_ifd(tiff *t, size_t off, page *pg) {
    size_t hdr = t->big ? 8 : 2;
    if (off + hdr > t->n) return fail("ifd offset out of range");
    uint64_t ne = t->big ? rd64(t, off) : rd16(t, off);
    if (off + hdr + ne * entry_size(t) + (t->big ? 8 : 4) > t->n) return fail("ifd truncated");
    pg->p.bits = 8; pg->p.spp = 1; pg->p.fmt = 1; pg->compression = 1; pg->predictor = 1; pg->planar = 1; pg->rows_per_strip = 0xFFFFFFFFu;
    size_t so = 0, sc = 0, to = 0, tc = 0; uint32_t nso = 0, nto = 0;
    for (uint64_t i = 0; i < ne; i++) {
        size_t e = off + hdr + (size_t)i * entry_size(t);
        uint16_t tag = rd16(t, e); uint64_t count = rd_cnt(t, e + 4);
        switch (tag) {
        case 256: pg->p.w = (int)tag_val(t, e, 0); break;
        case 257: pg->p.h = (int)tag_val(t, e, 0); break;
        case 258: pg->p.bits = (int)tag_val(t, e, 0); break;
        case 259: pg->compression = (int)tag_val(t, e, 0); break;
        case 270: { uint16_t type = rd16(t, e + 2); int ts = type < 19 ? type_size[type] : 1; size_t inl = t->big ? 8 : 4, voff = t->big ? 12 : 8; size_t data = (size_t)count * ts <= inl ? e + voff : (size_t)rd_off(t, e + voff);
                    if (data + count <= t->n) { pg->description = malloc(count + 1); memcpy(pg->description, t->buf + data, count); pg->description[count] = 0; } break; }
        case 273: so = e; nso = (uint32_t)count; break;
        case 277: pg->p.spp = (int)tag_val(t, e, 0); break;
        case 278: pg->rows_per_strip = (uint32_t)tag_val(t, e, 0); break;
        case 279: sc = e; break;
        case 284: pg->planar = (int)tag_val(t, e, 0); break;
        case 317: pg->predictor = (int)tag_val(t, e, 0); break;
        case 322: pg->tile_w = (uint32_t)tag_val(t, e, 0); break;
        case 323: pg->tile_h = (uint32_t)tag_val(t, e, 0); break;
        case 324: to = e; nto = (uint32_t)count; break;
        case 325: tc = e; break;
        case 339: pg->p.fmt = (int)tag_val(t, e, 0); break;
        }
    }
    if (pg->planar != 1) return fail("planar config %d unsupported", pg->planar);
    if (pg->p.bits != 8 && pg->p.bits != 16 && pg->p.bits != 32) return fail("bits %d unsupported", pg->p.bits);
    if (to) { pg->nstrips = nto; so = to; sc = tc; }
    else if (!so) return fail("no strip/tile offsets");
    else pg->nstrips = nso;
    if (!sc) return fail("no byte counts");
    pg->offsets = malloc(pg->nstrips * sizeof(uint64_t));
    pg->counts = malloc(pg->nstrips * sizeof(uint64_t));
    for (uint32_t i = 0; i < pg->nstrips; i++) { pg->offsets[i] = tag_val(t, so, i); pg->counts[i] = tag_val(t, sc, i); }
    return 0;
}

tiff *tiff_open_mem(const uint8_t *buf, size_t n) {
    int big = n >= 16 && (!memcmp(buf, "II+\0", 4) || !memcmp(buf, "MM\0+", 4));
    if (n < 8 || (!big && memcmp(buf, "II*\0", 4) && memcmp(buf, "MM\0*", 4))) { fail("not a TIFF"); return nullptr; }
    tiff *t = calloc(1, sizeof *t);
    t->buf = buf; t->n = n; t->le = buf[0] == 'I'; t->big = big;
    size_t off = big ? (size_t)rd64(t, 8) : rd32(t, 4);
    int cap = 8;
    t->pages = calloc((size_t)cap, sizeof(page));
    while (off && off + 2 <= n) {
        if (t->npages == cap) { cap *= 2; t->pages = realloc(t->pages, (size_t)cap * sizeof(page)); memset(t->pages + t->npages, 0, (size_t)(cap - t->npages) * sizeof(page)); }
        if (parse_ifd(t, off, &t->pages[t->npages])) { tiff_close(t); return nullptr; }
        t->npages++;
        uint64_t ne = t->big ? rd64(t, off) : rd16(t, off);
        off = (size_t)rd_off(t, off + (t->big ? 8 : 2) + (size_t)ne * entry_size(t));
        if (t->npages > 100000) break;
    }
    if (!t->npages) { fail("no pages"); tiff_close(t); return nullptr; }
    return t;
}

tiff *tiff_open_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fail("cannot open %s", path); return nullptr; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *b = malloc((size_t)n);
    if (fread(b, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(b); fail("short read %s", path); return nullptr; }
    fclose(f);
    tiff *t = tiff_open_mem(b, (size_t)n);
    if (t) t->own = b; else free(b);
    return t;
}

void tiff_close(tiff *t) {
    if (!t) return;
    for (int i = 0; i < t->npages; i++) { free(t->pages[i].offsets); free(t->pages[i].counts); free(t->pages[i].description); }
    free(t->pages); free(t->own); free(t);
}

int tiff_npages(const tiff *t) { return t->npages; }
int tiff_page_info(const tiff *t, int i, tiff_page *info) { if (i < 0 || i >= t->npages) return -1; *info = t->pages[i].p; return 0; }
const char *tiff_description(tiff *t, int i) { return i >= 0 && i < t->npages ? t->pages[i].description : nullptr; }

/* ---- LZW (TIFF flavour: MSB-first codes, 9..12 bits, early change) ---- */
static int lzw_decode(const uint8_t *in, size_t n, uint8_t *out, size_t outlen) {
    enum { CLEAR = 256, EOI = 257 };
    static _Thread_local uint16_t prefix[4096];
    static _Thread_local uint8_t suffix[4096], stack[4096];
    size_t ip = 0, op = 0;
    uint64_t bitbuf = 0; int nbits = 0, codelen = 9;
    int next = 258, prev = -1;
    while (op < outlen) {
        while (nbits < codelen) { bitbuf = bitbuf << 8 | (ip < n ? in[ip] : 0); ip++; nbits += 8; if (ip > n + 8) return fail("lzw: truncated"); }
        int code = (int)((bitbuf >> (nbits - codelen)) & ((1u << codelen) - 1));
        nbits -= codelen;
        if (code == EOI) break;
        if (code == CLEAR) { next = 258; codelen = 9; prev = -1; continue; }
        int sp = 0, c = code, first;
        if (code >= next) {                      /* KwKwK case */
            if (prev < 0) return fail("lzw: bad code");
            c = prev;
            int f = c; while (f >= 256) f = prefix[f];
            stack[sp++] = (uint8_t)f;
        }
        while (c >= 256) { stack[sp++] = suffix[c]; c = prefix[c]; if (sp >= 4096) return fail("lzw: stack"); }
        stack[sp++] = (uint8_t)c;
        first = c;
        for (int i = sp - 1; i >= 0 && op < outlen; i--) out[op++] = stack[i];
        if (prev >= 0 && next < 4096) { prefix[next] = (uint16_t)prev; suffix[next] = (uint8_t)first; next++; }
        prev = code;
        if (next + 1 >= (1 << codelen) && codelen < 12) codelen++;   /* early change */
    }
    return 0;
}

static int inflate_buf(const uint8_t *in, size_t n, uint8_t *out, size_t outlen) {
    z_stream st = {0}; st.next_in = (Bytef *)in; st.avail_in = (uInt)n; st.next_out = out; st.avail_out = (uInt)outlen;
    if (inflateInit(&st) != Z_OK) return fail("inflateInit");
    int r = inflate(&st, Z_FINISH); inflateEnd(&st);
    if (r != Z_STREAM_END && !(r == Z_OK && st.avail_out == 0)) return fail("inflate %d", r);
    return 0;
}

/* undo predictor on one decoded block of rows: w samples per row (x spp), bps bytes per sample */
static void unpredict(uint8_t *blk, int rows, int w, int spp, int bps, int predictor, int le) {
    size_t rowbytes = (size_t)w * spp * bps;
    if (predictor == 2) {
        for (int r = 0; r < rows; r++) {
            uint8_t *p = blk + r * rowbytes;
            if (bps == 1) for (size_t i = spp; i < rowbytes; i++) p[i] = (uint8_t)(p[i] + p[i - spp]);
            else if (bps == 2) for (size_t i = spp; i < (size_t)w * spp; i++) {
                uint16_t a, b; memcpy(&a, p + i * 2, 2); memcpy(&b, p + (i - spp) * 2, 2);
                if (!le) { a = (uint16_t)(a << 8 | a >> 8); b = (uint16_t)(b << 8 | b >> 8); }
                a = (uint16_t)(a + b); if (!le) a = (uint16_t)(a << 8 | a >> 8); memcpy(p + i * 2, &a, 2);
            }
        }
    } else if (predictor == 3) {
        uint8_t *tmp = malloc(rowbytes);
        size_t wc = (size_t)w * spp;
        for (int r = 0; r < rows; r++) {
            uint8_t *p = blk + r * rowbytes;
            for (size_t i = 1; i < rowbytes; i++) p[i] = (uint8_t)(p[i] + p[i - 1]);
            memcpy(tmp, p, rowbytes);
            for (size_t i = 0; i < wc; i++)
                for (int b = 0; b < bps; b++) p[i * bps + (bps - 1 - b)] = tmp[(size_t)b * wc + i];   /* planes are MSB first; host little-endian */
        }
        free(tmp);
    }
}

int tiff_read_page(tiff *t, int pi, void *vout) {
    if (pi < 0 || pi >= t->npages) return fail("no page %d", pi);
    page *pg = &t->pages[pi];
    int w = pg->p.w, h = pg->p.h, spp = pg->p.spp, bps = pg->p.bits / 8;
    uint8_t *out = vout;
    size_t rowbytes = (size_t)w * spp * bps;
    int tiled = pg->tile_w > 0;
    uint32_t tw = tiled ? pg->tile_w : (uint32_t)w, th = tiled ? pg->tile_h : (pg->rows_per_strip < (uint32_t)h ? pg->rows_per_strip : (uint32_t)h);
    uint32_t tiles_x = tiled ? (w + tw - 1) / tw : 1;
    size_t blkbytes = (size_t)tw * th * spp * bps;
    uint8_t *blk = malloc(blkbytes);
    for (uint32_t i = 0; i < pg->nstrips; i++) {
        uint64_t off = pg->offsets[i], cnt = pg->counts[i];
        if (off + cnt > t->n) { free(blk); return fail("strip %u out of range", i); }
        uint32_t ty = tiled ? i / tiles_x : i, tx = tiled ? i % tiles_x : 0;
        uint32_t y0 = ty * th, x0 = tx * tw;
        if (y0 >= (uint32_t)h) continue;
        uint32_t rows = tiled ? th : (y0 + th > (uint32_t)h ? (uint32_t)h - y0 : th);
        size_t need = (size_t)tw * rows * spp * bps;
        memset(blk, 0, need);
        int rc;
        switch (pg->compression) {
        case 1: if (cnt < need) need = cnt; memcpy(blk, t->buf + off, need); rc = 0; break;
        case 5: rc = lzw_decode(t->buf + off, cnt, blk, need); break;
        case 8: case 32946: rc = inflate_buf(t->buf + off, cnt, blk, need); break;
        default: free(blk); return fail("compression %d unsupported", pg->compression);
        }
        if (rc) { free(blk); return -1; }
        unpredict(blk, (int)rows, (int)tw, spp, bps, pg->predictor, t->le);
        /* byte order for 16-bit / float samples */
        if (!t->le && bps > 1) for (size_t k = 0; k < (size_t)tw * rows * spp; k++) { uint8_t *p = blk + k * bps; for (int a = 0; a < bps / 2; a++) { uint8_t q = p[a]; p[a] = p[bps - 1 - a]; p[bps - 1 - a] = q; } }
        uint32_t cols = x0 + tw > (uint32_t)w ? (uint32_t)w - x0 : tw;
        for (uint32_t r = 0; r < rows && y0 + r < (uint32_t)h; r++)
            memcpy(out + (size_t)(y0 + r) * rowbytes + (size_t)x0 * spp * bps, blk + (size_t)r * tw * spp * bps, (size_t)cols * spp * bps);
    }
    free(blk);
    return 0;
}
