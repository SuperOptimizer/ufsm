/* test_annot: the manual annotation tool (src/annot.c): store add / validate / del / undo / reload, the wrap-plane cut on a flat and a
   tilted synthetic wrap, and the HTTP server on a fixture CT pyramid (z3w-written levels 0 and 1): /info, /slice bytes against the
   written volume, /ann/add + /ann snapshot. */
#include <arpa/inet.h>
#include <math.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
#include "annot.h"
#include "z3w.h"

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL %s:%d ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static char *http(int port, const char *method, const char *path, const char *body, size_t *n) {
    const int fd = socket(AF_INET, SOCK_STREAM, 0); struct sockaddr_in a = {.sin_family = AF_INET, .sin_port = htons((uint16_t)port)};
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK); if (connect(fd, (struct sockaddr *)&a, sizeof a)) { close(fd); return nullptr; }
    char req[1024]; const int k = snprintf(req, sizeof req, "%s %s HTTP/1.1\r\nHost: x\r\nContent-Length: %zu\r\n\r\n%s", method, path, body ? strlen(body) : 0, body ? body : "");
    if (write(fd, req, (size_t)k) != k) { close(fd); return nullptr; }
    size_t cap = 1 << 20, m = 0; char *b = malloc(cap); ssize_t r;
    while ((r = read(fd, b + m, cap - m - 1)) > 0) { m += (size_t)r; if (m + 1 >= cap) { cap *= 2; b = realloc(b, cap); } }
    close(fd); b[m] = 0; char *h = strstr(b, "\r\n\r\n"); if (!h) { free(b); return nullptr; }
    *n = m - (size_t)(h + 4 - b); char *o = malloc(*n + 1); memcpy(o, h + 4, *n); o[*n] = 0; free(b); return o;
}

int main(void) {
    char dir[] = "/tmp/test_annot_XXXXXX"; if (!mkdtemp(dir)) return 1;
    char sp[256]; snprintf(sp, sizeof sp, "%s/ann.json", dir);
    /* store */
    annot_store *s = annot_open(sp, "T", "key"); char err[256];
    const int a = annot_add(s, "{\"type\":\"point\",\"p\":[10,20,30],\"w\":5}", "me", err, sizeof err);
    const int b = annot_add(s, "{\"type\":\"pair\",\"a\":[1,2,3],\"b\":[4,5,6],\"dw\":1}", "me", err, sizeof err);
    CHECK(a == 1 && b == 2, "ids %d %d", a, b);
    CHECK(annot_add(s, "{\"type\":\"pair\",\"a\":[1,2,3]}", "me", err, sizeof err) < 0 && strstr(err, "pair"), "invalid pair accepted (%s)", err);
    CHECK(annot_add(s, "{\"type\":\"region\",\"box\":[0,0,0,1,1,1],\"verdict\":\"maybe\"}", "me", err, sizeof err) < 0, "bad verdict accepted");
    const int c = annot_add(s, "{\"type\":\"line\",\"pts\":[[0,0,0],[0,1,1]],\"w\":2,\"id\":99,\"author\":\"forged\"}", "me", err, sizeof err);
    CHECK(c == 3 && annot_count(s) == 3, "line add %d count %d", c, annot_count(s));
    char *t = annot_snapshot(s); CHECK(!strstr(t, "forged") && !strstr(t, "\"id\":99"), "client id / author must be replaced: %s", t); free(t);
    CHECK(annot_del(s, 2) == 0 && annot_count(s) == 2 && annot_del(s, 2) < 0, "del");
    CHECK(annot_undo(s) == 0 && annot_count(s) == 3, "undo del");
    CHECK(annot_undo(s) == 0 && annot_count(s) == 2, "undo add (line)");
    annot_close(s);
    s = annot_open(sp, nullptr, nullptr); CHECK(s && annot_count(s) == 2, "reload count %d", s ? annot_count(s) : -1);
    CHECK(annot_add(s, "{\"type\":\"wraperr\",\"p\":[1,1,1],\"wrap\":7,\"kind\":\"cuts\"}", "me", err, sizeof err) == 4, "ids continue after reload");
    annot_close(s);
    { char lp[300]; snprintf(lp, sizeof lp, "%s.log", sp); FILE *f = fopen(lp, "r"); int lines = 0; char l[4096]; while (f && fgets(l, sizeof l, f)) lines++; if (f) fclose(f);
      CHECK(lines == 7, "log lines %d (4 adds, 1 del, 2 undos)", lines); }
    /* wrap cut: a flat wrap z = 50 (rows along y, cols along x), cut by y = 7.5: segment(s) along x at z = 50 */
    {   const int w = 11, h = 11; float *g = malloc((size_t)w * h * 12);
        for (int r = 0; r < h; r++) for (int i = 0; i < w; i++) { float *p = g + 3 * (r * w + i); p[0] = 50; p[1] = (float)r; p[2] = (float)i; }
        float *seg = nullptr; size_t cap = 0, n = 0; const double lo[2] = {0, 0}, hi[2] = {0, 0};
        annot_cut_wrap(g, w, h, 1, 7.5, lo, hi, &seg, &cap, &n);
        double len = 0; int ok = 1;
        for (size_t k = 0; k < n; k += 4) { ok &= fabsf(seg[k] - 50) < 1e-4f && fabsf(seg[k + 2] - 50) < 1e-4f; len += hypot(seg[k + 2] - seg[k], seg[k + 3] - seg[k + 1]); }
        CHECK(n > 0 && ok && fabs(len - 10) < 1e-3, "flat cut: %zu floats, on z 50 %d, length %.4f (10)", n, ok, len);
        /* tilted: z = 40 + y; cut by z = 45 -> the line y = 5 */
        for (int r = 0; r < h; r++) for (int i = 0; i < w; i++) g[3 * (r * w + i)] = 40.f + (float)r;
        n = 0; annot_cut_wrap(g, w, h, 0, 45.25, lo, hi, &seg, &cap, &n); ok = n > 0;
        for (size_t k = 0; k < n; k += 2) ok &= fabsf(seg[k] - 5.25f) < 1e-3f;
        CHECK(ok, "tilted cut: rows at y 5.25");
        n = 0; const double lo2[2] = {0, 20}, hi2[2] = {100, 30}; annot_cut_wrap(g, w, h, 0, 45.25, lo2, hi2, &seg, &cap, &n);
        CHECK(n == 0, "region filter: %zu floats outside the box", n);
        free(g); free(seg);
    }
    /* server on a fixture pyramid: level 0 128^3 v(z,y,x) = (z + 2y + 3x) & 255, level 1 64^3 v = (z * 7) & 255 */
    {   char root[300]; snprintf(root, sizeof root, "%s/ct.zarr", dir); char l0[320], l1[320]; snprintf(l0, sizeof l0, "%s/0", root); snprintf(l1, sizeof l1, "%s/1", root);
        const int64_t sh0[3] = {128, 128, 128}, sh1[3] = {64, 64, 64}; uint8_t *v = malloc(128 * 128 * 128);
        for (int z = 0; z < 128; z++) for (int y = 0; y < 128; y++) for (int x = 0; x < 128; x++) v[(z * 128 + y) * 128 + x] = (uint8_t)((z + 2 * y + 3 * x) & 255);
        z3w *w0 = z3w_create(l0, sh0, 128, 0.f, 0, nullptr); CHECK(w0 && !z3w_write_shard(w0, 0, 0, 0, v, 1) && !z3w_close(w0), "fixture level 0");
        memset(v, 0, 128 * 128 * 128); for (int z = 0; z < 64; z++) for (int y = 0; y < 64; y++) for (int x = 0; x < 64; x++) v[(z * 128 + y) * 128 + x] = (uint8_t)((z * 7) & 255);
        z3w *w1 = z3w_create(l1, sh1, 128, 0.f, 0, nullptr); CHECK(w1 && !z3w_write_shard(w1, 0, 0, 0, v, 1) && !z3w_close(w1), "fixture level 1");
        char st2[300]; snprintf(st2, sizeof st2, "%s/srv.json", dir);
        annot_cfg c = {.ct_root = dir, .ct_key = "ct.zarr", .um0 = 1.0, .base = 0, .store = st2, .scroll = "fixture", .author = "tester", .port = 0, .threads = 2, .cache_gb = 0.1};
        const int port = annot_serve_start(&c); CHECK(port > 0, "server start");
        size_t n = 0; char *r = http(port, "GET", "/info", nullptr, &n);
        CHECK(r && strstr(r, "\"l\":0,\"shape\":[128,128,128]") && strstr(r, "\"l\":1,\"shape\":[64,64,64]"), "info: %s", r ? r : "(none)"); free(r);
        /* y plane at level-0 y = 9, rows z 10..14, cols x 20..27 */
        r = http(port, "GET", "/slice?ax=1&pos=9&l=0&r0=10&c0=20&nr=5&nc=8", nullptr, &n); int ok = r && n == 40;
        for (int i = 0; ok && i < 5; i++) for (int j = 0; j < 8; j++) ok &= (uint8_t)r[i * 8 + j] == (uint8_t)((10 + i + 18 + 3 * (20 + j)) & 255);
        CHECK(ok, "slice ax 1 bytes (%zu)", n); free(r);
        /* level 1, z plane at level-0 z = 20 (level-1 index 10), partly outside the array (cols 60..67): 0 there */
        r = http(port, "GET", "/slice?ax=0&pos=20&l=1&r0=0&c0=60&nr=2&nc=8", nullptr, &n); ok = r && n == 16;
        for (int j = 0; ok && j < 8; j++) ok &= (uint8_t)r[j] == (60 + j < 64 ? 70 : 0);
        CHECK(ok, "slice level 1 with the array edge"); free(r);
        r = http(port, "POST", "/ann/add", "{\"type\":\"point\",\"p\":[1,2,3],\"w\":4}", &n); CHECK(r && strstr(r, "\"id\":1"), "ann add: %s", r ? r : ""); free(r);
        r = http(port, "POST", "/ann/add", "{\"type\":\"nope\"}", &n); CHECK(r && strstr(r, "error"), "ann add invalid: %s", r ? r : ""); free(r);
        r = http(port, "GET", "/ann", nullptr, &n); CHECK(r && strstr(r, "\"author\":\"tester\"") && strstr(r, "\"w\":4"), "ann snapshot: %s", r ? r : ""); free(r);
        r = http(port, "GET", "/", nullptr, &n); CHECK(r && strstr(r, "<html") && strstr(r, "/ann/add"), "page served (%zu bytes)", n); free(r);
        r = http(port, "GET", "/nothing", nullptr, &n); CHECK(r && strstr(r, "not found"), "404"); free(r);
        annot_serve_stop(); free(v);
    }
    char cmd[400]; snprintf(cmd, sizeof cmd, "rm -rf %s", dir); if (system(cmd)) {}
    printf(fails ? "test_annot: %d FAILED\n" : "test_annot: ok\n", fails); return fails != 0;
}
