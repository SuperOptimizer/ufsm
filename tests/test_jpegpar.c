/* src/jpegpar.c: our banded multithreaded JPEG encoder against libjpeg-turbo's jpeg_set_defaults + jpeg_set_quality(q, TRUE) bytes:
   RGB and gray, random / smooth / constant / sparse images, sizes around every 8 / 16 boundary (1 px, odd, MCU-row padding,
   partial last MCUs), qualities 50..100, thread counts 1..7, pixel pointers and the row callback.
   test_jpegpar FILE.jpg ...: each JPEG decoded (as RGB, and its luma as gray) and re-encoded both ways (bytes compared, Mpx/s).
   test_jpegpar --bench W H: synthetic W x H RGB and gray, libjpeg vs ours on 1 and N threads.
   JPAR_DIR=dir: those two modes write real files there (timed with the write; removed after the comparison). */
#include "jpegpar.h"
#include "imgops.h"
#include <jpeglib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static int fails, cases;
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

/* the bench / file modes (JPAR_DIR set): each encode written to a real file there (timed: the encode + fclose into g_secs), then
   read back for the comparison; otherwise in memory */
static const char *g_dir; static double g_secs;
static FILE *out_open(const char *name, char **mem, size_t *ml, char *path) {
    if (!g_dir) return open_memstream(mem, ml);
    snprintf(path, 1024, "%s/%s", g_dir, name); return fopen(path, "wb");
}
static uint8_t *out_close(FILE *f, char **mem, size_t *ml, const char *path, double t0, size_t *n) {   /* (open_memstream sets mem / ml at fclose) */
    fclose(f); g_secs = now() - t0;
    if (!g_dir) { *n = *ml; return (uint8_t *)*mem; }
    FILE *r = fopen(path, "rb"); if (!r) return nullptr;
    fseek(r, 0, SEEK_END); *n = (size_t)ftell(r); fseek(r, 0, SEEK_SET); uint8_t *b = malloc(*n ? *n : 1);
    if (fread(b, 1, *n, r) != *n) { free(b); b = nullptr; }
    fclose(r); unlink(path); return b;
}
static uint8_t *lj(int w, int h, int nc, int q, const uint8_t *px, size_t *n) {
    char *mem = nullptr, path[1024]; size_t ml = 0; const double t0 = now(); FILE *f = out_open("libjpeg.jpg", &mem, &ml, path);
    struct jpeg_compress_struct ci; struct jpeg_error_mgr je;
    ci.err = jpeg_std_error(&je); jpeg_create_compress(&ci); jpeg_stdio_dest(&ci, f);
    ci.image_width = (JDIMENSION)w; ci.image_height = (JDIMENSION)h; ci.input_components = nc; ci.in_color_space = nc == 3 ? JCS_RGB : JCS_GRAYSCALE;
    jpeg_set_defaults(&ci); jpeg_set_quality(&ci, q, TRUE); jpeg_start_compress(&ci, TRUE);
    while (ci.next_scanline < ci.image_height) { JSAMPROW r = (JSAMPROW)(px + (size_t)ci.next_scanline * w * nc); jpeg_write_scanlines(&ci, &r, 1); }
    jpeg_finish_compress(&ci); jpeg_destroy_compress(&ci);
    return out_close(f, &mem, &ml, path, t0, n);
}
typedef struct { const uint8_t *px; int w, nc; } src;
static void rows_cb(void *u, int y, int x0, int n, uint8_t *dst) { const src *s = u; memcpy(dst, s->px + ((size_t)y * s->w + x0) * s->nc, (size_t)n * s->nc); }
static uint8_t *ours(int w, int h, int nc, int q, const uint8_t *px, int threads, int cb, size_t *n) {
    char *mem = nullptr, path[1024]; size_t ml = 0; const double t0 = now(); FILE *f = out_open("ours.jpg", &mem, &ml, path);
    src s = {px, w, nc};
    const int rc = jpar_encode(f, w, h, nc, q, cb ? nullptr : px, cb ? rows_cb : nullptr, &s, threads);
    if (jpar_force_own && !jpar_last_own) { fprintf(stderr, "FAIL: %dx%d nc %d q %d went to libjpeg\n", w, h, nc, q); fails++; }
    uint8_t *b = out_close(f, &mem, &ml, path, t0, n);
    if (rc) { free(b); return nullptr; }
    return b;
}
static int compare(int w, int h, int nc, int q, const uint8_t *px, int threads, int cb, const char *what) {
    size_t a, b; uint8_t *A = lj(w, h, nc, q, px, &a), *B = ours(w, h, nc, q, px, threads, cb, &b); cases++;
    int ok = B && a == b && !memcmp(A, B, a);
    if (!ok) {
        size_t i = 0; if (B) while (i < a && i < b && A[i] == B[i]) i++;
        fprintf(stderr, "FAIL: %s %dx%d nc %d q %d threads %d cb %d: libjpeg %zu bytes, ours %zu, first difference at %zu\n", what, w, h, nc, q, threads, cb, a, B ? b : 0, i);
        fails++;
    }
    free(A); free(B); return ok;
}

static uint32_t rs = 12345;
static uint32_t rnd(void) { rs = rs * 1664525u + 1013904223u; return rs >> 8; }
static void fill(uint8_t *p, int w, int h, int nc, int kind) {
    for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) for (int c = 0; c < nc; c++) {
        uint8_t v;
        switch (kind) {
        case 0: v = (uint8_t)rnd(); break;                                                    /* noise */
        case 1: v = (uint8_t)((x * 3 + y * 2 + c * 40 + ((x * y) >> 5)) & 255); break;        /* smooth ramps */
        case 2: v = (uint8_t)(c == 0 ? 255 : c == 1 ? 0 : 77); break;                          /* constant (extremes) */
        default: v = (rnd() % 23 == 0) ? (uint8_t)rnd() : (uint8_t)(c * 60); break;          /* sparse speckle (long zero runs, ZRL) */
        }
        p[((size_t)y * w + x) * nc + c] = v;
    }
}

static void bench(int w, int h) {
    const int N = jpar_threads();
    for (int nc = 3; nc >= 1; nc -= 2) {
        uint8_t *p = malloc((size_t)w * h * nc);
        for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) for (int c = 0; c < nc; c++)   /* a smooth image with noise: render-like */
            p[((size_t)y * w + x) * nc + c] = (uint8_t)(128 + 60 * __builtin_sinf(x * 0.013f + c) * __builtin_cosf(y * 0.017f) + (int)(rnd() % 24) - 12);
        const double mpx = (double)w * h * 1e-6; size_t a, b1, bn;
        uint8_t *A = lj(w, h, nc, 90, p, &a); const double tl = g_secs;
        uint8_t *B1 = ours(w, h, nc, 90, p, 1, 0, &b1); const double t1 = g_secs;
        uint8_t *BN = ours(w, h, nc, 90, p, N, 0, &bn); const double tn = g_secs;
        const int same = B1 && BN && a == b1 && a == bn && !memcmp(A, B1, a) && !memcmp(A, BN, a);
        printf("bench %s %dx%d q90 (%.1f MB): libjpeg %.2fs %.0f Mpx/s | ours 1 thread %.2fs %.0f Mpx/s | ours %d threads %.2fs %.0f Mpx/s | %s\n",
               nc == 3 ? "RGB " : "gray", w, h, a / 1e6, tl, mpx / tl, t1, mpx / t1, N, tn, mpx / tn, same ? "identical" : "DIFFERENT");
        if (!same) fails++;
        free(A); free(B1); free(BN); free(p);
    }
}
static uint8_t *read_any(const char *path, int *w, int *h, int *nc) {   /* a JPEG as gray (1 component) or RGB */
    FILE *f = fopen(path, "rb"); if (!f) return nullptr;
    struct jpeg_decompress_struct ci; struct jpeg_error_mgr je; ci.err = jpeg_std_error(&je);
    jpeg_create_decompress(&ci); jpeg_stdio_src(&ci, f); jpeg_read_header(&ci, TRUE);
    *nc = ci.num_components == 1 ? 1 : 3; ci.out_color_space = *nc == 1 ? JCS_GRAYSCALE : JCS_RGB; jpeg_start_decompress(&ci);
    *w = (int)ci.output_width; *h = (int)ci.output_height; uint8_t *p = malloc((size_t)*w * *h * *nc);
    while (ci.output_scanline < ci.output_height) { JSAMPROW r = p + (size_t)ci.output_scanline * *w * *nc; jpeg_read_scanlines(&ci, &r, 1); }
    jpeg_finish_decompress(&ci); jpeg_destroy_decompress(&ci); fclose(f); return p;
}
static void files(int n, char **paths) {
    const int N = jpar_threads();
    for (int i = 0; i < n; i++) {
        int w, h, ncf; uint8_t *rgb = nullptr, *g = read_any(paths[i], &w, &h, &ncf);   /* gray files stay gray (a 2.5 Gpx stripe) */
        if (!g) { fprintf(stderr, "FAIL: cannot read %s\n", paths[i]); fails++; continue; }
        if (ncf == 3) { rgb = g; g = malloc((size_t)w * h); for (size_t k = 0; k < (size_t)w * h; k++) g[k] = rgb[3 * k + 1]; }
        for (int nc = ncf; nc >= 1; nc -= 2) for (int qi = 0; qi < 2; qi++) {
            const int q = qi ? 88 : 95; const uint8_t *p = nc == 3 ? rgb : g; size_t a, b;
            uint8_t *A = lj(w, h, nc, q, p, &a); const double tl = g_secs;
            uint8_t *B = ours(w, h, nc, q, p, N, qi, &b); const double tn = g_secs;
            const int same = B && a == b && !memcmp(A, B, a); cases++; if (!same) fails++;
            printf("%s %dx%d %s q%d%s: %zu bytes, libjpeg %.2fs, ours (%d threads) %.2fs: %s\n", paths[i], w, h, nc == 3 ? "RGB" : "gray", q, qi ? " (row callback)" : "", a, tl, N, tn, same ? "identical" : "DIFFERENT");
            free(A); free(B);
        }
        free(rgb); free(g);
    }
}

int main(int argc, char **argv) {
    g_dir = getenv("JPAR_DIR");   /* bench / file modes: encode to real files there */
    if (argc == 4 && !strcmp(argv[1], "--bench")) { bench(atoi(argv[2]), atoi(argv[3])); return fails ? 1 : 0; }
    if (argc > 1) { files(argc - 1, argv + 1); printf("test_jpegpar files: %d/%d identical\n", cases - fails, cases); return fails ? 1 : 0; }
    jpar_force_own = 1;
    static const int sz[][2] = {{1, 1}, {1, 17}, {17, 1}, {2, 2}, {7, 9}, {8, 8}, {9, 8}, {8, 9}, {15, 15}, {16, 16}, {17, 17}, {23, 31}, {24, 40},
                                {31, 33}, {32, 32}, {33, 47}, {40, 24}, {47, 49}, {48, 48}, {63, 65}, {100, 75}, {129, 3}, {3, 129}, {200, 161}};
    static const int qs[] = {50, 85, 88, 90, 95, 100};
    for (size_t s = 0; s < sizeof sz / sizeof *sz; s++) {
        const int w = sz[s][0], h = sz[s][1];
        uint8_t *p = malloc((size_t)w * h * 3);
        for (int nc = 1; nc <= 3; nc += 2) for (int kind = 0; kind < 4; kind++) {
            fill(p, w, h, nc, kind);
            for (size_t qi = 0; qi < sizeof qs / sizeof *qs; qi++) compare(w, h, nc, qs[qi], p, 1 + (int)((s + qi + kind) % 4), (int)(qi & 1), "small");
        }
        free(p);
    }
    /* many bands per thread / per image, odd quality values */
    static const int big[][2] = {{1000, 700}, {333, 2501}, {2049, 130}, {5, 3000}, {3001, 9}};
    for (size_t s = 0; s < sizeof big / sizeof *big; s++) {
        const int w = big[s][0], h = big[s][1]; uint8_t *p = malloc((size_t)w * h * 3);
        for (int nc = 1; nc <= 3; nc += 2) for (int kind = 0; kind < 4; kind++) {
            fill(p, w, h, nc, kind);
            const int thr[] = {1, 2, 3, 7};
            for (int t = 0; t < 4; t++) compare(w, h, nc, (int[]){50, 88, 90, 100, 1, 23, 97, 75}[(t + kind * 2) & 7], p, thr[t], t & 1, "big");
        }
        free(p);
    }
    /* a tall image at 1 thread: several bands joined at every bit phase */
    { const int w = 64, h = 65000; uint8_t *p = malloc((size_t)w * h * 3); fill(p, w, h, 3, 0); compare(w, h, 3, 90, p, 1, 0, "tall"); compare(w, h, 3, 90, p, 5, 1, "tall"); fill(p, w, h, 1, 1); compare(w, h, 1, 90, p, 1, 0, "tall"); compare(w, h, 1, 90, p, 6, 0, "tall"); free(p); }
    printf("test_jpegpar: %d/%d cases byte-identical to libjpeg\n", cases - fails, cases);
    return fails ? 1 : 0;
}
