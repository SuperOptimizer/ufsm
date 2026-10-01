#include "zipr.h"
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <zlib.h>

static _Thread_local char errbuf[256];
const char *zipr_error(void) { return errbuf; }
static int fail(const char *fmt, ...) { va_list ap; va_start(ap, fmt); vsnprintf(errbuf, sizeof errbuf, fmt, ap); va_end(ap); return -1; }

typedef struct { uint64_t size, csize, local_off; uint32_t name_off; uint16_t method; } ent;

struct zipr {
    int fd;
    uint64_t fsize;
    ent *e;
    size_t n;
    char *names;       /* arena of nul-terminated names */
};

static uint16_t u16(const uint8_t *p) { return (uint16_t)(p[0] | p[1] << 8); }
static uint32_t u32(const uint8_t *p) { return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24; }
static uint64_t u64(const uint8_t *p) { return (uint64_t)u32(p) | (uint64_t)u32(p + 4) << 32; }

static int rd(int fd, uint64_t off, void *buf, size_t n) {
    size_t got = 0;
    while (got < n) { ssize_t r = pread(fd, (uint8_t *)buf + got, n - got, (off_t)(off + got)); if (r <= 0) return -1; got += (size_t)r; }
    return 0;
}

zipr *zipr_open(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fail("open %s", path); return nullptr; }
    struct stat st; fstat(fd, &st);
    uint64_t fsize = (uint64_t)st.st_size;
    /* find EOCD in the last 64 KiB */
    size_t tail = fsize < 65536 + 22 ? (size_t)fsize : 65536 + 22;
    uint8_t *t = malloc(tail);
    if (rd(fd, fsize - tail, t, tail)) { fail("read tail"); free(t); close(fd); return nullptr; }
    long eocd = -1;
    for (long i = (long)tail - 22; i >= 0; i--) if (u32(t + i) == 0x06054b50) { eocd = i; break; }
    if (eocd < 0) { fail("no end-of-central-directory"); free(t); close(fd); return nullptr; }
    uint64_t cd_off = u32(t + eocd + 16), cd_size = u32(t + eocd + 12), nent = u16(t + eocd + 10);
    /* zip64 locator right before EOCD */
    if (eocd >= 20 && u32(t + eocd - 20) == 0x07064b50) {
        uint64_t e64 = u64(t + eocd - 20 + 8);
        uint8_t r[56];
        if (rd(fd, e64, r, 56) || u32(r) != 0x06064b50) { fail("bad zip64 EOCD"); free(t); close(fd); return nullptr; }
        nent = u64(r + 32); cd_size = u64(r + 40); cd_off = u64(r + 48);
    }
    free(t);
    uint8_t *cd = malloc(cd_size);
    if (rd(fd, cd_off, cd, cd_size)) { fail("read central directory"); free(cd); close(fd); return nullptr; }
    zipr *z = calloc(1, sizeof *z);
    z->fd = fd; z->fsize = fsize;
    z->e = malloc(nent * sizeof *z->e);
    size_t name_cap = cd_size, name_len = 0;
    z->names = malloc(name_cap);
    size_t p = 0, i = 0;
    while (i < nent && p + 46 <= cd_size) {
        if (u32(cd + p) != 0x02014b50) { fail("bad central directory entry %zu", i); break; }
        ent *e = &z->e[i];
        e->method = u16(cd + p + 10);
        uint64_t csize = u32(cd + p + 20), usize = u32(cd + p + 24), loff = u32(cd + p + 42);
        uint16_t fn = u16(cd + p + 28), ex = u16(cd + p + 30), cm = u16(cd + p + 32);
        if (p + 46 + fn + ex + cm > cd_size) { fail("truncated central directory"); break; }
        /* zip64 extra */
        const uint8_t *x = cd + p + 46 + fn, *xe = x + ex;
        while (x + 4 <= xe) {
            uint16_t id = u16(x), len = u16(x + 2);
            if (id == 1) {
                const uint8_t *q = x + 4;
                if (usize == 0xFFFFFFFFu && q + 8 <= x + 4 + len) { usize = u64(q); q += 8; }
                if (csize == 0xFFFFFFFFu && q + 8 <= x + 4 + len) { csize = u64(q); q += 8; }
                if (loff == 0xFFFFFFFFu && q + 8 <= x + 4 + len) { loff = u64(q); q += 8; }
            }
            x += 4 + len;
        }
        e->size = usize; e->csize = csize; e->local_off = loff;
        if (name_len + fn + 1 > name_cap) { name_cap = name_cap * 2 + fn + 1; z->names = realloc(z->names, name_cap); }
        e->name_off = (uint32_t)name_len;
        memcpy(z->names + name_len, cd + p + 46, fn);
        name_len += fn; z->names[name_len++] = 0;
        p += 46 + fn + ex + cm;
        i++;
    }
    z->n = i;
    free(cd);
    return z;
}

void zipr_close(zipr *z) { if (!z) return; close(z->fd); free(z->e); free(z->names); free(z); }
size_t zipr_count(const zipr *z) { return z->n; }

int zipr_entry(const zipr *z, size_t i, zip_entry *e) {
    if (i >= z->n) return -1;
    e->name = z->names + z->e[i].name_off; e->size = z->e[i].size; e->csize = z->e[i].csize; e->local_off = z->e[i].local_off; e->method = z->e[i].method;
    return 0;
}

long zipr_find(const zipr *z, const char *name) {
    for (size_t i = 0; i < z->n; i++) if (!strcmp(z->names + z->e[i].name_off, name)) return (long)i;
    return -1;
}

uint8_t *zipr_read(const zipr *z, size_t i, size_t *len) {
    if (i >= z->n) { fail("bad index"); return nullptr; }
    const ent *e = &z->e[i];
    uint8_t lh[30];
    if (rd(z->fd, e->local_off, lh, 30) || u32(lh) != 0x04034b50) { fail("bad local header for entry %zu", i); return nullptr; }
    uint64_t data = e->local_off + 30 + u16(lh + 26) + u16(lh + 28);
    if (data + e->csize > z->fsize) { fail("entry %zu beyond end of file", i); return nullptr; }
    uint8_t *c = malloc((size_t)e->csize + 1);
    if (rd(z->fd, data, c, (size_t)e->csize)) { fail("read entry %zu", i); free(c); return nullptr; }
    if (e->method == 0) { c[e->csize] = 0; *len = (size_t)e->csize; return c; }
    if (e->method != 8) { fail("entry %zu: method %d unsupported", i, e->method); free(c); return nullptr; }
    uint8_t *u = malloc((size_t)e->size + 1);
    z_stream st = {0}; st.next_in = c; st.avail_in = (uInt)e->csize; st.next_out = u; st.avail_out = (uInt)e->size;
    inflateInit2(&st, -MAX_WBITS);
    int r = inflate(&st, Z_FINISH); inflateEnd(&st);
    free(c);
    if (r != Z_STREAM_END) { fail("inflate entry %zu", i); free(u); return nullptr; }
    u[e->size] = 0; *len = (size_t)e->size;
    return u;
}
