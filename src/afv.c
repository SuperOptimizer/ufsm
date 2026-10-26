/* AFV fibre collection reader (src/afv.h): libsqlite3 through dlopen. */
#include "afv.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct sqlite3 sqlite3;
typedef struct sqlite3_stmt sqlite3_stmt;
static struct {
    void *h;
    int (*open_v2)(const char *, sqlite3 **, int, const char *);
    int (*close)(sqlite3 *);
    const char *(*errmsg)(sqlite3 *);
    int (*prepare_v2)(sqlite3 *, const char *, int, sqlite3_stmt **, const char **);
    int (*bind_double)(sqlite3_stmt *, int, double);
    int (*bind_text)(sqlite3_stmt *, int, const char *, int, void (*)(void *));
    int (*step)(sqlite3_stmt *);
    int (*finalize)(sqlite3_stmt *);
    int (*reset)(sqlite3_stmt *);
    long long (*column_int64)(sqlite3_stmt *, int);
    const void *(*column_blob)(sqlite3_stmt *, int);
    int (*column_bytes)(sqlite3_stmt *, int);
    const unsigned char *(*column_text)(sqlite3_stmt *, int);
} L;
enum { SQ_ROW = 100, SQ_DONE = 101, SQ_RO = 0x1, SQ_URI = 0x40 };

static int afv_lib(void) {
    if (L.h) return 0;
    void *h = dlopen("libsqlite3.so.0", RTLD_NOW | RTLD_LOCAL);
    if (!h) h = dlopen("libsqlite3.so", RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "afv: cannot load libsqlite3 (%s)\n", dlerror()); return -1; }
#define SYM(f, n) if (!(*(void **)&L.f = dlsym(h, n))) { fprintf(stderr, "afv: libsqlite3 lacks %s\n", n); dlclose(h); return -1; }
    SYM(open_v2, "sqlite3_open_v2") SYM(close, "sqlite3_close") SYM(errmsg, "sqlite3_errmsg") SYM(prepare_v2, "sqlite3_prepare_v2")
    SYM(bind_double, "sqlite3_bind_double") SYM(bind_text, "sqlite3_bind_text") SYM(step, "sqlite3_step") SYM(finalize, "sqlite3_finalize") SYM(reset, "sqlite3_reset")
    SYM(column_int64, "sqlite3_column_int64") SYM(column_blob, "sqlite3_column_blob") SYM(column_bytes, "sqlite3_column_bytes")
    SYM(column_text, "sqlite3_column_text")
#undef SYM
    L.h = h; return 0;
}

struct afv { sqlite3 *db; sqlite3_stmt *q; };

afv *afv_open(const char *path) {
    if (afv_lib()) return nullptr;
    /* immutable: no locking / journal lookups (the file is a published, read-only artifact) */
    size_t n = strlen(path) * 3 + 32; char *uri = malloc(n), *o = uri; o += sprintf(o, "file:");
    for (const char *p = path; *p; p++) { if (*p == '?' || *p == '#' || *p == '%') o += sprintf(o, "%%%02X", (unsigned char)*p); else *o++ = *p; }
    sprintf(o, "?immutable=1");
    afv *a = calloc(1, sizeof *a);
    if (L.open_v2(uri, &a->db, SQ_RO | SQ_URI, nullptr)) { fprintf(stderr, "afv: cannot open %s: %s\n", path, a->db ? L.errmsg(a->db) : "?"); if (a->db) L.close(a->db); free(a); free(uri); return nullptr; }
    free(uri);
    const char *sql = "SELECT b.fiber_id, b.first_segment, b.points, f.family FROM block_bounds r JOIN blocks b ON b.id = r.id JOIN fibers f ON f.id = b.fiber_id"
                      " WHERE r.max_x >= ?1 AND r.min_x <= ?2 AND r.max_y >= ?3 AND r.min_y <= ?4 AND r.max_z >= ?5 AND r.min_z <= ?6 ORDER BY b.fiber_id, b.first_segment";
    if (L.prepare_v2(a->db, sql, -1, &a->q, nullptr)) { fprintf(stderr, "afv: %s: not an AFV (%s)\n", path, L.errmsg(a->db)); L.close(a->db); free(a); return nullptr; }
    return a;
}

void afv_close(afv *a) { if (!a) return; L.finalize(a->q); L.close(a->db); free(a); }

char *afv_meta(afv *a, const char *key) {
    sqlite3_stmt *s; char *r = nullptr;
    if (L.prepare_v2(a->db, "SELECT value FROM metadata WHERE key = ?1", -1, &s, nullptr)) return nullptr;
    L.bind_text(s, 1, key, -1, (void (*)(void *))(intptr_t)-1);
    if (L.step(s) == SQ_ROW) { const unsigned char *t = L.column_text(s, 0); if (t) r = strdup((const char *)t); }
    L.finalize(s); return r;
}

void afv_runs_free(afv_run *r, int n) { if (!r) return; for (int i = 0; i < n; i++) free(r[i].xyz); free(r); }

int afv_query(afv *a, const double lo[3], const double hi[3], afv_run **out) {
    sqlite3_stmt *q = a->q; *out = nullptr;
    L.reset(q);   /* (prepared once; rebinding needs a reset) */
    for (int d = 0; d < 3; d++) { L.bind_double(q, 1 + 2 * d, lo[d]); L.bind_double(q, 2 + 2 * d, hi[d]); }
    int nr = 0, cap = 0; afv_run *R = nullptr; long long pf = -1, pseg = -1; int rc;
    while ((rc = L.step(q)) == SQ_ROW) {
        const long long fid = L.column_int64(q, 0), seg = L.column_int64(q, 1);
        const double *p = L.column_blob(q, 2); const int nb = L.column_bytes(q, 2) / 24;
        const unsigned char *fam = L.column_text(q, 3);
        if (nb < 1) continue;
        const int cont = fid == pf && seg == pseg + 256 && nr > 0;
        if (!cont) {
            if (nr == cap) { cap = cap ? 2 * cap : 256; R = realloc(R, (size_t)cap * sizeof *R); }
            R[nr++] = (afv_run){fid, fam && (fam[0] == 'H' || fam[0] == 'V') ? (char)fam[0] : '?', 0, nullptr};
        }
        afv_run *r = &R[nr - 1]; const int skip = cont ? 1 : 0;   /* the shared point */
        r->xyz = realloc(r->xyz, (size_t)(r->n + nb - skip) * 3 * sizeof(double));
        memcpy(r->xyz + 3 * r->n, p + 3 * skip, (size_t)(nb - skip) * 3 * sizeof(double)); r->n += nb - skip;
        pf = fid; pseg = seg;
    }
    if (rc != SQ_DONE) { fprintf(stderr, "afv: query failed: %s\n", L.errmsg(a->db)); afv_runs_free(R, nr); return -1; }
    *out = R; return nr;
}

int fib_save(const char *path, const fib_set *f) {
    FILE *o = fopen(path, "wb"); if (!o) return -1;
    const int32_t h[3] = {1, f->n, f->npts}; int bad = fwrite("UFIB", 1, 4, o) != 4 || fwrite(h, 4, 3, o) != 3;
    bad |= fwrite(f->off, sizeof(int), (size_t)f->n + 1, o) != (size_t)f->n + 1 || (f->n && fwrite(f->fam, 1, (size_t)f->n, o) != (size_t)f->n);
    bad |= f->npts && fwrite(f->pts, sizeof(float), 3 * (size_t)f->npts, o) != 3 * (size_t)f->npts;
    bad |= fclose(o) != 0; return bad ? -1 : 0;
}
int fib_load(const char *path, fib_set *f) {
    memset(f, 0, sizeof *f); FILE *in = fopen(path, "rb"); if (!in) return -1;
    char m[4]; int32_t h[3];
    if (fread(m, 1, 4, in) != 4 || memcmp(m, "UFIB", 4) || fread(h, 4, 3, in) != 3 || h[0] != 1 || h[1] < 0 || h[2] < 0) { fclose(in); return -1; }
    f->n = h[1]; f->npts = h[2]; f->off = malloc(((size_t)f->n + 1) * sizeof(int)); f->fam = malloc((size_t)f->n + 1); f->pts = malloc(3 * (size_t)f->npts * sizeof(float) + 4);
    int bad = !f->off || !f->fam || !f->pts || fread(f->off, sizeof(int), (size_t)f->n + 1, in) != (size_t)f->n + 1 || fread(f->fam, 1, (size_t)f->n, in) != (size_t)f->n
              || fread(f->pts, sizeof(float), 3 * (size_t)f->npts, in) != 3 * (size_t)f->npts;
    fclose(in);
    for (int r = 0; !bad && r < f->n; r++) bad = f->off[r] < 0 || f->off[r + 1] < f->off[r] || f->off[r + 1] > f->npts;
    if (bad || f->off[0] != 0) { fib_free(f); return -1; }
    return 0;
}
void fib_free(fib_set *f) { free(f->off); free(f->fam); free(f->pts); memset(f, 0, sizeof *f); }
