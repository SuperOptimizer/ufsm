/* src/afv.c: an AFV (SQLite: fibers, blocks with 256-segment float64 XYZ blobs sharing one point, rtree block_bounds) written here
   through the same run-time libsqlite3, read back by afv_query (bounding-box selection, consecutive blocks joined into one run,
   a gap in first_segment starting a new run, families); fib_save / fib_load round trip and a corrupt file rejected. */
#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "afv.h"

typedef struct sqlite3 sqlite3; typedef struct sqlite3_stmt sqlite3_stmt;
static int (*sq_open)(const char *, sqlite3 **); static int (*sq_exec)(sqlite3 *, const char *, void *, void *, char **);
static int (*sq_prep)(sqlite3 *, const char *, int, sqlite3_stmt **, const char **); static int (*sq_bind_int)(sqlite3_stmt *, int, long long);
static int (*sq_bind_dbl)(sqlite3_stmt *, int, double); static int (*sq_bind_blob)(sqlite3_stmt *, int, const void *, int, void (*)(void *));
static int (*sq_bind_text)(sqlite3_stmt *, int, const char *, int, void (*)(void *)); static int (*sq_step)(sqlite3_stmt *);
static int (*sq_fin)(sqlite3_stmt *); static int (*sq_close)(sqlite3 *);
static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); fails++; } } while (0)

/* fibre id, family, its points (n x 3), blocks from first_segment s0 (a gap: skip blocks) */
static void put_fiber(sqlite3 *db, int id, const char *fam, const double *p, int n, int skip_block, int *bid) {
    sqlite3_stmt *s; sq_prep(db, "INSERT INTO fibers VALUES(?1,'f',?2,?3,1.0,0,0,0,0,0,0,'{}')", -1, &s, nullptr);
    sq_bind_int(s, 1, id); sq_bind_text(s, 2, fam, -1, nullptr); sq_bind_int(s, 3, n); sq_step(s); sq_fin(s);
    for (int first = 0, b = 0; first < n - 1; first += 256, b++) {
        if (b == skip_block) continue;
        const int m = n - first < 257 ? n - first : 257; double lo[3] = {1e30, 1e30, 1e30}, hi[3] = {-1e30, -1e30, -1e30};
        for (int k = 0; k < m; k++) for (int a = 0; a < 3; a++) { const double v = p[3 * (first + k) + a]; if (v < lo[a]) lo[a] = v; if (v > hi[a]) hi[a] = v; }
        ++*bid;
        sq_prep(db, "INSERT INTO blocks VALUES(?1,?2,?3,?4)", -1, &s, nullptr);
        sq_bind_int(s, 1, *bid); sq_bind_int(s, 2, id); sq_bind_int(s, 3, first); sq_bind_blob(s, 4, p + 3 * first, m * 24, nullptr); sq_step(s); sq_fin(s);
        sq_prep(db, "INSERT INTO block_bounds VALUES(?1,?2,?3,?4,?5,?6,?7)", -1, &s, nullptr);
        sq_bind_int(s, 1, *bid); for (int a = 0; a < 3; a++) { sq_bind_dbl(s, 2 + 2 * a, lo[a]); sq_bind_dbl(s, 3 + 2 * a, hi[a]); }
        sq_step(s); sq_fin(s);
    }
}

int main(void) {
    void *h = dlopen("libsqlite3.so.0", RTLD_NOW);
    if (!h) { printf("test_afv: no libsqlite3 (skipped)\n"); return 0; }
    *(void **)&sq_open = dlsym(h, "sqlite3_open"); *(void **)&sq_exec = dlsym(h, "sqlite3_exec"); *(void **)&sq_prep = dlsym(h, "sqlite3_prepare_v2");
    *(void **)&sq_bind_int = dlsym(h, "sqlite3_bind_int64"); *(void **)&sq_bind_dbl = dlsym(h, "sqlite3_bind_double"); *(void **)&sq_bind_blob = dlsym(h, "sqlite3_bind_blob");
    *(void **)&sq_bind_text = dlsym(h, "sqlite3_bind_text"); *(void **)&sq_step = dlsym(h, "sqlite3_step"); *(void **)&sq_fin = dlsym(h, "sqlite3_finalize"); *(void **)&sq_close = dlsym(h, "sqlite3_close");
    char path[] = "/tmp/test_afv_XXXXXX"; const int fd = mkstemp(path); close(fd); unlink(path);
    sqlite3 *db; if (sq_open(path, &db)) { fprintf(stderr, "test_afv: cannot create %s\n", path); return 1; }
    sq_exec(db, "CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);"
                "CREATE TABLE fibers(id INTEGER PRIMARY KEY,name TEXT NOT NULL,family TEXT NOT NULL,point_count INTEGER NOT NULL,length REAL NOT NULL,"
                " min_x REAL NOT NULL,max_x REAL NOT NULL,min_y REAL NOT NULL,max_y REAL NOT NULL,min_z REAL NOT NULL,max_z REAL NOT NULL,annotation TEXT NOT NULL);"
                "CREATE TABLE blocks(id INTEGER PRIMARY KEY,fiber_id INTEGER NOT NULL,first_segment INTEGER NOT NULL,points BLOB NOT NULL);"
                "CREATE VIRTUAL TABLE block_bounds USING rtree(id,min_x,max_x,min_y,max_y,min_z,max_z);"
                "INSERT INTO metadata VALUES('complete','true');", nullptr, nullptr, nullptr);
    /* fibre 1 (H): 600 points along x at y 10 z 20 (3 blocks); fibre 2 (V): 300 points along z at x 1000 (2 blocks, far away);
       fibre 3 (H): 800 points along x at y 50 with its second block missing (two runs) */
    const int n1 = 600, n2 = 300, n3 = 800; double *p1 = malloc(n1 * 24), *p2 = malloc(n2 * 24), *p3 = malloc(n3 * 24);
    for (int k = 0; k < n1; k++) { p1[3 * k] = k * 0.5; p1[3 * k + 1] = 10; p1[3 * k + 2] = 20; }
    for (int k = 0; k < n2; k++) { p2[3 * k] = 1000; p2[3 * k + 1] = 10; p2[3 * k + 2] = k; }
    for (int k = 0; k < n3; k++) { p3[3 * k] = k * 0.25; p3[3 * k + 1] = 50; p3[3 * k + 2] = 20; }
    int bid = 0; put_fiber(db, 1, "H", p1, n1, -1, &bid); put_fiber(db, 2, "V", p2, n2, -1, &bid); put_fiber(db, 3, "H", p3, n3, 1, &bid);
    sq_close(db);

    afv *A = afv_open(path); CHECK(A, "afv_open");
    if (A) {
        char *m = afv_meta(A, "complete"); CHECK(m && !strcmp(m, "true"), "metadata complete: %s", m ? m : "null"); free(m);
        afv_run *R; const double lo[3] = {0, 0, 0}, hi[3] = {400, 100, 100};
        const int nr = afv_query(A, lo, hi, &R);
        CHECK(nr == 3, "runs %d (want 3: fibre 1 joined, fibre 3 split)", nr);
        if (nr == 3) {
            CHECK(R[0].fiber == 1 && R[0].family == 'H' && R[0].n == n1, "run 0: fibre %lld %c n %d", (long long)R[0].fiber, R[0].family, R[0].n);
            int ok = 1; for (int k = 0; k < R[0].n * 3; k++) ok &= R[0].xyz[k] == p1[k]; CHECK(ok, "fibre 1 points joined in order");
            CHECK(R[1].fiber == 3 && R[1].n == 257 && R[2].fiber == 3 && R[2].n == n3 - 512, "fibre 3 runs: %d (%lld) + %d (%lld)", R[1].n, (long long)R[1].fiber, R[2].n, (long long)R[2].fiber);
            CHECK(R[2].xyz[0] == p3[3 * 512], "fibre 3 second run starts at point 512");
        }
        afv_runs_free(R, nr);
        const double lo2[3] = {900, 0, 0}, hi2[3] = {1100, 20, 1000};
        const int nr2 = afv_query(A, lo2, hi2, &R);
        CHECK(nr2 == 1 && R[0].fiber == 2 && R[0].family == 'V' && R[0].n == n2, "second query: %d runs", nr2);
        afv_runs_free(R, nr2);
        afv_close(A);
    }
    unlink(path);
    /* fib files */
    fib_set F = {0}; F.n = 2; F.npts = 5; int off[3] = {0, 2, 5}; char fam[2] = {'H', 'V'}; float pts[15]; for (int k = 0; k < 15; k++) pts[k] = k * 1.5f;
    F.off = off; F.fam = fam; F.pts = pts;
    char fp[] = "/tmp/test_fib_XXXXXX"; const int fd2 = mkstemp(fp); close(fd2);
    CHECK(!fib_save(fp, &F), "fib_save");
    fib_set G; CHECK(!fib_load(fp, &G), "fib_load");
    CHECK(G.n == 2 && G.npts == 5 && G.off[2] == 5 && G.fam[1] == 'V' && G.pts[14] == pts[14], "fib round trip");
    fib_free(&G);
    FILE *o = fopen(fp, "r+b"); fseek(o, 16, SEEK_SET); const int bad = 99; fwrite(&bad, 4, 1, o); fclose(o);   /* off[0] corrupted */
    CHECK(fib_load(fp, &G), "corrupt fib rejected");
    unlink(fp); free(p1); free(p2); free(p3);
    if (fails) { printf("test_afv: %d failures\n", fails); return 1; }
    printf("test_afv: ok\n"); return 0;
}
