/* The looping GIF89a writer of the augmentation animations (tests/crush_anim.c, tests/scrollaug_anim.c): a fixed 256-colour palette
   (64 grays + 16 hues x 12 brightness), LZW in C (the gif.h scheme). Include after <stdio.h>, <string.h>, <math.h>. */
#pragma once

enum { NG = 64, NH = 16, NB = 12 };
static uint8_t pal[256][3];
static void make_pal(void) {
    for (int i = 0; i < NG; i++) pal[i][0] = pal[i][1] = pal[i][2] = (uint8_t)(i * 255 / (NG - 1));
    for (int h = 0; h < NH; h++) for (int b = 0; b < NB; b++) {
        const double x = h / (double)NH, c = (b + .5) * 256.0 / NB;
        const double r = .5 + .5 * cos(2 * M_PI * x), g = .5 + .5 * cos(2 * M_PI * (x - 1. / 3)), bl = .5 + .5 * cos(2 * M_PI * (x - 2. / 3));
        uint8_t *o = pal[NG + h * NB + b]; o[0] = (uint8_t)((c + 255 * r) / 2); o[1] = (uint8_t)((c + 255 * g) / 2); o[2] = (uint8_t)((c + 255 * bl) / 2);
    }
}
static uint8_t gray_ix(int c) { return (uint8_t)((c * (NG - 1) + 127) / 255); }
static uint8_t hue_ix(int c, double h) { int hi = (int)floor(h * NH + .5) % NH; if (hi < 0) hi += NH; int b = c * NB / 256; return (uint8_t)(NG + hi * NB + (b > NB - 1 ? NB - 1 : b)); }

/* ---- GIF89a: LZW (the gif.h scheme: a 4096 x 256 code tree, 12-bit codes, a clear when full) ---- */
typedef struct { FILE *f; uint32_t bits; int nbits; uint8_t blk[256]; int nb; } gbits;
static void g_byte(gbits *g, uint8_t b) { g->blk[g->nb++] = b; if (g->nb == 255) { fputc(255, g->f); fwrite(g->blk, 1, 255, g->f); g->nb = 0; } }
static void g_code(gbits *g, uint32_t code, int size) { g->bits |= code << g->nbits; g->nbits += size; while (g->nbits >= 8) { g_byte(g, (uint8_t)g->bits); g->bits >>= 8; g->nbits -= 8; } }
static void g_flush(gbits *g) { if (g->nbits) g_byte(g, (uint8_t)g->bits); g->bits = 0; g->nbits = 0; if (g->nb) { fputc(g->nb, g->f); fwrite(g->blk, 1, (size_t)g->nb, g->f); g->nb = 0; } fputc(0, g->f); }
static void gif_frame(FILE *f, const uint8_t *ix, int W, int H, int delay_cs, uint16_t *tree) {
    const uint8_t gce[8] = {0x21, 0xF9, 4, 0, (uint8_t)delay_cs, (uint8_t)(delay_cs >> 8), 0, 0}; fwrite(gce, 1, 8, f);
    const uint8_t id[10] = {0x2C, 0, 0, 0, 0, (uint8_t)W, (uint8_t)(W >> 8), (uint8_t)H, (uint8_t)(H >> 8), 0}; fwrite(id, 1, 10, f);
    const int minc = 8, clear = 256; fputc(minc, f);
    gbits g = {f, 0, 0, {0}, 0}; int size = minc + 1, maxc = clear + 1;
    memset(tree, 0, (size_t)4096 * 256 * sizeof *tree);
    g_code(&g, clear, size);
    int cur = -1;
    for (size_t p = 0; p < (size_t)W * H; p++) {
        const int c = ix[p];
        if (cur < 0) { cur = c; continue; }
        if (tree[(size_t)cur * 256 + c]) { cur = tree[(size_t)cur * 256 + c]; continue; }
        g_code(&g, (uint32_t)cur, size);
        tree[(size_t)cur * 256 + c] = (uint16_t)++maxc;
        if (maxc >= (1 << size)) size++;
        if (maxc == 4095) { g_code(&g, clear, size); memset(tree, 0, (size_t)4096 * 256 * sizeof *tree); size = minc + 1; maxc = clear + 1; }
        cur = c;
    }
    g_code(&g, (uint32_t)cur, size); g_code(&g, clear, size); g_code(&g, clear + 1, minc + 1);
    g_flush(&g);
}
static void gif_head(FILE *f, int W, int H) {
    fwrite("GIF89a", 1, 6, f);
    const uint8_t ls[7] = {(uint8_t)W, (uint8_t)(W >> 8), (uint8_t)H, (uint8_t)(H >> 8), 0xF7, 0, 0}; fwrite(ls, 1, 7, f);
    fwrite(pal, 1, 768, f);
    const uint8_t loop[19] = {0x21, 0xFF, 11, 'N', 'E', 'T', 'S', 'C', 'A', 'P', 'E', '2', '.', '0', 3, 1, 0, 0, 0}; fwrite(loop, 1, 19, f);
}
