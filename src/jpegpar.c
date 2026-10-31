/* src/jpegpar.c: see jpegpar.h. The pipeline reproduces libjpeg-turbo 2.1.5 (jccolor.c rgb_ycc_convert, jcsample.c
   fullsize / h2v2_downsample with expand_right_edge, jcprepct.c bottom padding, jccoefct.c dummy blocks, jfdctint.c
   jpeg_fdct_islow, jcdctmgr.c reciprocal quantisation = exact rounded division, jchuff.c encode_one_block + flush_bits). */
#include "jpegpar.h"
#include <jpeglib.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#ifdef __AVX2__
#include <immintrin.h>
#endif

int jpar_force_own = 0;
_Thread_local int jpar_last_own = 0;
static _Atomic int g_threads = 0;
void jpar_set_threads(int n) { atomic_store(&g_threads, n > 0 ? n : 0); }
int jpar_threads(void) {
    int n = atomic_load(&g_threads); if (n > 0) return n;
    const char *e = getenv("UFSM_JPEG_THREADS"); if (e && atoi(e) > 0) return atoi(e) > 256 ? 256 : atoi(e);
    const long c = sysconf(_SC_NPROCESSORS_ONLN); return c < 1 ? 1 : c > 32 ? 32 : (int)c;
}

static const uint8_t zz[64] = {0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,  12, 19, 26, 33, 40, 48,
                               41, 34, 27, 20, 13, 6,  7,  14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23,
                               30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63};

typedef struct {
    int w, h, nc;
    int mcuh, mcols, mrows, ybw, ybh, pw;   /* MCU height (16 / 8), MCUs per row, MCU rows, Y blocks wide / high, plane width */
    int32_t qm[2][64], qs[2][64], qh[2][64];   /* per table, natural order: floor((|c| + qh) / d) = (|c| + qh) * qm >> qs (d = 8 q, qh = d / 2) */
    uint32_t dcc[2][16], acc_[2][256]; uint8_t dcl[2][16], acl[2][256];   /* Huffman code / length per table */
    int qt[3], dt[3], at[3];   /* per component: quant / DC / AC table */
    const uint8_t *px; jpar_rows_fn fn; void *u;
    uint8_t *hdr; size_t hdrlen;
} jctx;

/* ---------------- libjpeg: headers, fallback ---------------- */
static void lj_setup(struct jpeg_compress_struct *ci, int w, int h, int nc, int q) {
    ci->image_width = (JDIMENSION)w; ci->image_height = (JDIMENSION)h; ci->input_components = nc;
    ci->in_color_space = nc == 3 ? JCS_RGB : JCS_GRAYSCALE;
    jpeg_set_defaults(ci); jpeg_set_quality(ci, q, TRUE);
}
static int lj_encode(FILE *f, int w, int h, int nc, int q, const uint8_t *px, jpar_rows_fn fn, void *u) {
    struct jpeg_compress_struct ci; struct jpeg_error_mgr je;
    ci.err = jpeg_std_error(&je); jpeg_create_compress(&ci); jpeg_stdio_dest(&ci, f);
    lj_setup(&ci, w, h, nc, q); jpeg_start_compress(&ci, TRUE);
    uint8_t *row = px ? nullptr : malloc((size_t)w * nc);
    while (ci.next_scanline < ci.image_height) {
        JSAMPROW r;
        if (px) r = (JSAMPROW)(px + (size_t)ci.next_scanline * w * nc); else { fn(u, (int)ci.next_scanline, 0, w, row); r = row; }
        jpeg_write_scanlines(&ci, &r, 1);
    }
    jpeg_finish_compress(&ci); jpeg_destroy_compress(&ci); free(row);
    return 0;
}
static void huff_build(const uint8_t *bits, const uint8_t *vals, int nv, uint32_t *code, uint8_t *len, int nsym) {
    memset(len, 0, (size_t)nsym); uint32_t c = 0; int k = 0;
    for (int l = 1; l <= 16; l++) { for (int i = 0; i < bits[l - 1]; i++, k++) if (k < nv && vals[k] < nsym) { code[vals[k]] = c++; len[vals[k]] = (uint8_t)l; } else c++; c <<= 1; }
}
/* the header (SOI .. SOS) of a 16 x 16 libjpeg encode with these parameters, SOF size patched to w x h; the tables parsed from it.
   0 ok, -1 = anything we do not reproduce */
static int ctx_header(jctx *C, int q) {
    unsigned char *mem = nullptr; unsigned long ml = 0; uint8_t tiny[16 * 16 * 3]; memset(tiny, 128, sizeof tiny);
    struct jpeg_compress_struct ci; struct jpeg_error_mgr je;
    ci.err = jpeg_std_error(&je); jpeg_create_compress(&ci); jpeg_mem_dest(&ci, &mem, &ml);
    lj_setup(&ci, 16, 16, C->nc, q);
    if (ci.restart_interval || ci.restart_in_rows || ci.optimize_coding || ci.progressive_mode || ci.arith_code || ci.dct_method != JDCT_ISLOW ||
        ci.num_components != C->nc || (C->nc == 3 && ci.jpeg_color_space != JCS_YCbCr) || ci.smoothing_factor || ci.CCIR601_sampling) { jpeg_destroy_compress(&ci); return -1; }
    jpeg_start_compress(&ci, TRUE);
    for (int y = 0; y < 16; y++) { JSAMPROW r = tiny; jpeg_write_scanlines(&ci, &r, 1); }
    jpeg_finish_compress(&ci); jpeg_destroy_compress(&ci);
    int ok = 0, have_sof = 0; size_t p = 2; uint16_t dqt[4][64]; int hq[4] = {0}; uint8_t hd[2][4] = {{0}};
    if (ml < 4 || mem[0] != 0xFF || mem[1] != 0xD8) goto done;
    while (p + 4 <= ml) {
        if (mem[p] != 0xFF) goto done;
        const int mk = mem[p + 1]; const size_t L = (size_t)mem[p + 2] << 8 | mem[p + 3], s = p + 4, e = p + 2 + L;
        if (L < 2 || e > ml) goto done;
        if (mk == 0xDB) {
            for (size_t i = s; i < e;) { const int pq = mem[i] >> 4, tq = mem[i] & 15; if (tq > 3) goto done; i++;
                for (int k = 0; k < 64; k++) { dqt[tq][zz[k]] = pq ? (uint16_t)(mem[i + 2 * k] << 8 | mem[i + 2 * k + 1]) : mem[i + k]; } i += pq ? 128 : 64; hq[tq] = 1; }
        } else if (mk == 0xC4) {
            for (size_t i = s; i < e;) { const int tc = mem[i] >> 4, th = mem[i] & 15; if (th > 1 || tc > 1) goto done;
                const uint8_t *bits = mem + i + 1; int nv = 0; for (int l = 0; l < 16; l++) nv += bits[l];
                if (tc == 0) huff_build(bits, bits + 16, nv, C->dcc[th], C->dcl[th], 16); else huff_build(bits, bits + 16, nv, C->acc_[th], C->acl[th], 256);
                hd[tc][th] = 1; i += 17 + (size_t)nv; }
        } else if (mk == 0xC0) {
            if (mem[s] != 8 || mem[s + 5] != C->nc) goto done;
            for (int c = 0; c < C->nc; c++) { const int hv = mem[s + 6 + 3 * c + 1]; C->qt[c] = mem[s + 6 + 3 * c + 2];
                if (hv != (C->nc == 3 && c == 0 ? 0x22 : 0x11) || C->qt[c] > 1) goto done; }
            mem[s + 1] = (uint8_t)(C->h >> 8); mem[s + 2] = (uint8_t)C->h; mem[s + 3] = (uint8_t)(C->w >> 8); mem[s + 4] = (uint8_t)C->w; have_sof = 1;
        } else if (mk == 0xDA) {
            if (mem[s] != C->nc || !have_sof) goto done;
            for (int c = 0; c < C->nc; c++) { C->dt[c] = mem[s + 2 + 2 * c] >> 4; C->at[c] = mem[s + 2 + 2 * c] & 15; if (C->dt[c] > 1 || C->at[c] > 1) goto done;
                if (!hd[0][C->dt[c]] || !hd[1][C->at[c]] || !hq[C->qt[c]]) goto done; }
            if (mem[e - 3] != 0 || mem[e - 2] != 63 || mem[e - 1] != 0) goto done;
            C->hdr = malloc(e); memcpy(C->hdr, mem, e); C->hdrlen = e; ok = 1; break;
        } else if (mk == 0xDD || (mk >= 0xC1 && mk <= 0xCF && mk != 0xC4)) goto done;
        p = e;
    }
    if (ok) for (int t = 0; t < 2; t++) if (hq[t]) for (int i = 0; i < 64; i++) {
        /* floor(x / d) = x * m >> s exactly for x < 2^14 (|DCT| <= 8192 plus d / 2 <= 1020): s = 14 + ceil(log2 d), m = ceil(2^s / d) */
        const uint32_t d = (uint32_t)dqt[t][i] << 3; if (!dqt[t][i]) { ok = 0; break; }
        int lg = 0; while ((1u << lg) < d) lg++;
        const int s = 14 + lg; C->qs[t][i] = s; C->qm[t][i] = (int32_t)((((uint64_t)1 << s) + d - 1) / d); C->qh[t][i] = (int32_t)(d >> 1);
    }
done:
    free(mem); return ok ? 0 : -1;
}

/* ---------------- per-band coding ---------------- */
/* unstuffed bits, MSB first: acc holds 64 - fr pending bits (in its low end); after bw_flush, nb = the bits in the last partial byte */
typedef struct { uint8_t *b; size_t len, cap; uint64_t acc; int fr, nb; } bitw;
static void bw_reserve(bitw *B, size_t n) { if (B->len + n > B->cap) { B->cap = (B->len + n) * 2 + 65536; B->b = realloc(B->b, B->cap); } }
static void bw_flush(bitw *B) {   /* whole bytes out; the < 8 remaining bits left-aligned in one more byte */
    int nb = 64 - B->fr;
    while (nb >= 8) { nb -= 8; B->b[B->len++] = (uint8_t)(B->acc >> nb); }
    B->nb = nb; B->b[B->len] = nb ? (uint8_t)(B->acc << (8 - nb)) : 0; memset(B->b + B->len + 1, 0, 16);   /* (padded for the readers) */
    B->acc = 0; B->fr = 64;
}

/* jpeg_fdct_islow (jfdctint.c) on 8 lanes at once: rows pass on columns-as-vectors, transpose, columns pass */
typedef int32_t v8i __attribute__((vector_size(32)));
typedef uint8_t v8b __attribute__((vector_size(8)));
#define CONST_BITS 13
#define PASS1_BITS 2
#define DESCALE(x, n) (((x) + (1 << ((n) - 1))) >> (n))
static inline void tr8(v8i *r) {
    const v8i t0 = __builtin_shufflevector(r[0], r[1], 0, 8, 1, 9, 4, 12, 5, 13), t1 = __builtin_shufflevector(r[0], r[1], 2, 10, 3, 11, 6, 14, 7, 15);
    const v8i t2 = __builtin_shufflevector(r[2], r[3], 0, 8, 1, 9, 4, 12, 5, 13), t3 = __builtin_shufflevector(r[2], r[3], 2, 10, 3, 11, 6, 14, 7, 15);
    const v8i t4 = __builtin_shufflevector(r[4], r[5], 0, 8, 1, 9, 4, 12, 5, 13), t5 = __builtin_shufflevector(r[4], r[5], 2, 10, 3, 11, 6, 14, 7, 15);
    const v8i t6 = __builtin_shufflevector(r[6], r[7], 0, 8, 1, 9, 4, 12, 5, 13), t7 = __builtin_shufflevector(r[6], r[7], 2, 10, 3, 11, 6, 14, 7, 15);
    const v8i u0 = __builtin_shufflevector(t0, t2, 0, 1, 8, 9, 4, 5, 12, 13), u1 = __builtin_shufflevector(t0, t2, 2, 3, 10, 11, 6, 7, 14, 15);
    const v8i u2 = __builtin_shufflevector(t1, t3, 0, 1, 8, 9, 4, 5, 12, 13), u3 = __builtin_shufflevector(t1, t3, 2, 3, 10, 11, 6, 7, 14, 15);
    const v8i u4 = __builtin_shufflevector(t4, t6, 0, 1, 8, 9, 4, 5, 12, 13), u5 = __builtin_shufflevector(t4, t6, 2, 3, 10, 11, 6, 7, 14, 15);
    const v8i u6 = __builtin_shufflevector(t5, t7, 0, 1, 8, 9, 4, 5, 12, 13), u7 = __builtin_shufflevector(t5, t7, 2, 3, 10, 11, 6, 7, 14, 15);
    r[0] = __builtin_shufflevector(u0, u4, 0, 1, 2, 3, 8, 9, 10, 11); r[4] = __builtin_shufflevector(u0, u4, 4, 5, 6, 7, 12, 13, 14, 15);
    r[1] = __builtin_shufflevector(u1, u5, 0, 1, 2, 3, 8, 9, 10, 11); r[5] = __builtin_shufflevector(u1, u5, 4, 5, 6, 7, 12, 13, 14, 15);
    r[2] = __builtin_shufflevector(u2, u6, 0, 1, 2, 3, 8, 9, 10, 11); r[6] = __builtin_shufflevector(u2, u6, 4, 5, 6, 7, 12, 13, 14, 15);
    r[3] = __builtin_shufflevector(u3, u7, 0, 1, 2, 3, 8, 9, 10, 11); r[7] = __builtin_shufflevector(u3, u7, 4, 5, 6, 7, 12, 13, 14, 15);
}
/* one 1-D pass over v[0..7] (element-wise); pass 1 (rows): shift up PASS1_BITS / descale CONST_BITS - PASS1_BITS; pass 2: descale PASS1_BITS / CONST_BITS + PASS1_BITS */
#define FDCT_PASS(v, P1)                                                                                                              \
    do {                                                                                                                              \
        const v8i tmp0 = v[0] + v[7], tmp7 = v[0] - v[7], tmp1 = v[1] + v[6], tmp6 = v[1] - v[6];                                    \
        const v8i tmp2 = v[2] + v[5], tmp5 = v[2] - v[5], tmp3 = v[3] + v[4], tmp4 = v[3] - v[4];                                    \
        const v8i tmp10 = tmp0 + tmp3, tmp13 = tmp0 - tmp3, tmp11 = tmp1 + tmp2, tmp12 = tmp1 - tmp2;                               \
        enum { SH = P1 ? CONST_BITS - PASS1_BITS : CONST_BITS + PASS1_BITS };                                                         \
        if (P1) { v[0] = (tmp10 + tmp11) << PASS1_BITS; v[4] = (tmp10 - tmp11) << PASS1_BITS; }                                       \
        else { v[0] = DESCALE(tmp10 + tmp11, PASS1_BITS); v[4] = DESCALE(tmp10 - tmp11, PASS1_BITS); }                                \
        v8i z1 = (tmp12 + tmp13) * 4433;                                                                                              \
        v[2] = DESCALE(z1 + tmp13 * 6270, SH); v[6] = DESCALE(z1 + tmp12 * -15137, SH);                                              \
        z1 = tmp4 + tmp7; v8i z2 = tmp5 + tmp6, z3 = tmp4 + tmp6, z4 = tmp5 + tmp7; const v8i z5 = (z3 + z4) * 9633;                 \
        const v8i t4 = tmp4 * 2446, t5 = tmp5 * 16819, t6 = tmp6 * 25172, t7 = tmp7 * 12299;                                         \
        z1 *= -7373; z2 *= -20995; z3 *= -16069; z4 *= -3196; z3 += z5; z4 += z5;                                                    \
        v[7] = DESCALE(t4 + z1 + z3, SH); v[5] = DESCALE(t5 + z2 + z4, SH); v[3] = DESCALE(t6 + z2 + z3, SH); v[1] = DESCALE(t7 + z1 + z4, SH); \
    } while (0)

static uint64_t zzmask_tab[8][256];   /* natural-order nonzero bits of row u (byte) -> their zigzag positions */
static int16_t zzpos[64];             /* natural index -> zigzag position */
static pthread_once_t zz_once = PTHREAD_ONCE_INIT;
static void zz_init(void) {
    for (int k = 0; k < 64; k++) zzpos[zz[k]] = (int16_t)k;
    for (int u = 0; u < 8; u++) for (int b = 0; b < 256; b++) { uint64_t m = 0; for (int c = 0; c < 8; c++) if (b >> c & 1) m |= (uint64_t)1 << zzpos[8 * u + c]; zzmask_tab[u][b] = m; }
}
static inline int lane_mask(v8i q) {   /* bit c set where lane c != 0 */
#ifdef __AVX2__
    const __m256i z = _mm256_cmpeq_epi32((__m256i)q, _mm256_setzero_si256()); return ~_mm256_movemask_ps(_mm256_castsi256_ps(z)) & 0xFF;
#else
    int m = 0; for (int c = 0; c < 8; c++) m |= (q[c] != 0) << c; return m;
#endif
}
/* the 8 x 8 block at pl (row stride ps) -> quantised coefficients (natural order); returns the zigzag nonzero mask of the AC terms */
static inline uint64_t fwd_block(const jctx *C, int t, const uint8_t *pl, size_t ps, int16_t *out) {
    v8i v[8];
    for (int r = 0; r < 8; r++) { v8b b; memcpy(&b, pl + r * ps, 8); v[r] = __builtin_convertvector(b, v8i) - 128; }
    tr8(v); FDCT_PASS(v, 1); tr8(v); FDCT_PASS(v, 0);   /* v[u] = coefficients (u, 0..7) */
    uint64_t nz = 0;
    for (int u = 0; u < 8; u++) {
        v8i m, s, h; memcpy(&m, C->qm[t] + 8 * u, 32); memcpy(&s, C->qs[t] + 8 * u, 32); memcpy(&h, C->qh[t] + 8 * u, 32);
        const v8i x = v[u], neg = x >> 31, a = (x ^ neg) - neg, q = ((a + h) * m) >> s, r = (q ^ neg) - neg;
        nz |= zzmask_tab[u][lane_mask(r)];
        for (int c = 0; c < 8; c++) out[8 * u + c] = (int16_t)r[c];
    }
    return nz & ~(uint64_t)1;
}
/* jchuff.c encode_one_block: the bit state in registers for the block */
#define PUTB(V, N)                                                                                                                      \
    do {                                                                                                                                \
        const uint32_t v_ = (V); const int n_ = (N);                                                                                    \
        if (n_ < fr) { acc = (acc << n_) | v_; fr -= n_; }                                                                              \
        else { const int rest = n_ - fr; const uint64_t w_ = __builtin_bswap64((acc << fr) | ((uint64_t)v_ >> rest)); memcpy(o, &w_, 8); o += 8; acc = v_; fr = 64 - rest; } \
    } while (0)
static inline void enc_block(const jctx *C, bitw *B, const int16_t *blk, uint64_t nz, int *last, int dt, int at) {
    uint64_t acc = B->acc; int fr = B->fr; uint8_t *o = B->b + B->len;
    const int t = blk[0] - *last; *last = blk[0];
    const int sg = t >> 31; const uint32_t a = (uint32_t)((t ^ sg) - sg); int n = 32 - __builtin_clzg(a, 32);
    PUTB((C->dcc[dt][n] << n) | ((uint32_t)(t + sg) & ((1u << n) - 1)), C->dcl[dt][n] + n);
    const uint32_t *ac = C->acc_[at]; const uint8_t *al = C->acl[at]; int prev = 0;
    while (nz) {
        const int k = __builtin_ctzll(nz); nz &= nz - 1; int r = k - prev - 1; prev = k;
        while (r > 15) { PUTB(ac[0xF0], al[0xF0]); r -= 16; }
        const int v = blk[zz[k]], s = v >> 31; const uint32_t m = (uint32_t)((v ^ s) - s);
        n = 32 - __builtin_clz(m); const int sym = (r << 4) + n;
        PUTB((ac[sym] << n) | ((uint32_t)(v + s) & ((1u << n) - 1)), al[sym] + n);
    }
    if (prev < 63) PUTB(ac[0], al[0]);
    B->acc = acc; B->fr = fr; B->len = (size_t)(o - B->b);
}

typedef struct { uint8_t *Y, *Cb, *Cr, *cf[2][2], *rowbuf; size_t rbcap; } scratch;
static void scratch_init(const jctx *C, scratch *S) {
    memset(S, 0, sizeof *S);
    S->Y = malloc((size_t)C->mcuh * C->pw);
    if (C->nc == 3) { S->Cb = malloc((size_t)8 * C->pw / 2); S->Cr = malloc((size_t)8 * C->pw / 2);
        for (int i = 0; i < 2; i++) { S->cf[i][0] = malloc((size_t)C->pw); S->cf[i][1] = malloc((size_t)C->pw); } }
}
static void scratch_free(scratch *S) { free(S->Y); free(S->Cb); free(S->Cr); for (int i = 0; i < 2; i++) { free(S->cf[i][0]); free(S->cf[i][1]); } free(S->rowbuf); }

/* source rows of MCU row my, pixels from xa on: rp[r][(x - xa) * nc] */
static void fetch_rows(const jctx *C, scratch *S, int my, int xa, const uint8_t **rp) {
    const int nc = C->nc, n = C->w - xa; const size_t rl = (size_t)n * nc;
    if (!C->px && S->rbcap < rl * C->mcuh) { S->rbcap = rl * C->mcuh; S->rowbuf = realloc(S->rowbuf, S->rbcap); }
    int prev = -1;
    for (int r = 0; r < C->mcuh; r++) {
        int y = my * C->mcuh + r; if (y > C->h - 1) y = C->h - 1;
        if (y == prev) { rp[r] = rp[r - 1]; continue; }
        prev = y;
        if (C->px) rp[r] = C->px + ((size_t)y * C->w + xa) * nc;
        else { uint8_t *d = S->rowbuf + (size_t)r * rl; C->fn(C->u, y, xa, n, d); rp[r] = d; }
    }
}
/* planes of MCU row my for MCU columns [m0, m1): Y (mcuh rows), Cb / Cr (8 rows, downsampled), x padded by edge replication */
static void make_planes(const jctx *C, scratch *S, int my, int m0, int m1) {
    const int nc = C->nc, w = C->w, pw = C->pw, xa = m0 * (nc == 3 ? 16 : 8), xb = m1 * (nc == 3 ? 16 : 8), xe = xb < w ? xb : w;
    const uint8_t *rp[16]; fetch_rows(C, S, my, xa, rp);
    if (nc == 1) {
        for (int r = 0; r < 8; r++) { uint8_t *Y = S->Y + (size_t)r * pw; memcpy(Y + xa, rp[r], (size_t)(xe - xa)); for (int x = xe; x < xb; x++) Y[x] = Y[w - 1]; }
        return;
    }
    const int hc = (C->h + 1) / 2;   /* chroma rows of the image */
    for (int r = 0; r < 16; r++) {
        uint8_t *Y = S->Y + (size_t)r * pw, *cb = S->cf[r & 1][0], *cr = S->cf[r & 1][1]; const uint8_t *s = rp[r];
        for (int x = xa; x < xe; x++) {
            const int R = s[3 * (x - xa)], G = s[3 * (x - xa) + 1], Bl = s[3 * (x - xa) + 2];
            Y[x] = (uint8_t)((19595 * R + 38470 * G + 7471 * Bl + 32768) >> 16);
            cb[x] = (uint8_t)((-11059 * R - 21709 * G + 32768 * Bl + (128 << 16) + 32767) >> 16);
            cr[x] = (uint8_t)((32768 * R - 27439 * G - 5329 * Bl + (128 << 16) + 32767) >> 16);
        }
        for (int x = xe; x < xb; x++) { Y[x] = Y[w - 1]; cb[x] = cb[w - 1]; cr[x] = cr[w - 1]; }
        if (r & 1) {
            const int j = r >> 1, cy = my * 8 + j; uint8_t *ob = S->Cb + (size_t)j * (pw / 2), *orr = S->Cr + (size_t)j * (pw / 2);
            if (cy >= hc) { memcpy(ob + xa / 2, ob - pw / 2 + xa / 2, (size_t)(xb - xa) / 2); memcpy(orr + xa / 2, orr - pw / 2 + xa / 2, (size_t)(xb - xa) / 2); continue; }
            const uint8_t *b0 = S->cf[0][0], *b1 = S->cf[1][0], *r0 = S->cf[0][1], *r1 = S->cf[1][1];
            for (int ox = xa / 2; ox < xb / 2; ox++) {
                const int bias = 1 + (ox & 1);
                ob[ox] = (uint8_t)((b0[2 * ox] + b0[2 * ox + 1] + b1[2 * ox] + b1[2 * ox + 1] + bias) >> 2);
                orr[ox] = (uint8_t)((r0[2 * ox] + r0[2 * ox + 1] + r1[2 * ox] + r1[2 * ox + 1] + bias) >> 2);
            }
        }
    }
}
/* MCU row my, MCU columns [m0, m1): coded into B (B nullptr: only the DC predictors advance) */
static void code_mcus(const jctx *C, scratch *S, int my, int m0, int m1, bitw *B, int *dc) {
    make_planes(C, S, my, m0, m1);
    int16_t blk[6][64]; uint64_t nz[6];
    if (C->nc == 1) {
        for (int m = m0; m < m1; m++) {
            nz[0] = fwd_block(C, C->qt[0], S->Y + 8 * m, (size_t)C->pw, blk[0]);
            if (B) { bw_reserve(B, 512); enc_block(C, B, blk[0], nz[0], &dc[0], C->dt[0], C->at[0]); } else dc[0] = blk[0][0];
        }
        return;
    }
    const size_t pw = (size_t)C->pw, cw = pw / 2;
    for (int m = m0; m < m1; m++) {
        for (int b = 0; b < 4; b++) {
            const int bx = 2 * m + (b & 1), by = 2 * my + (b >> 1);
            if (by >= C->ybh) { nz[b] = 0; memset(blk[b], 0, sizeof blk[b]); blk[b][0] = blk[1][0]; }   /* dummy row: DC of the block before (jccoefct.c) */
            else if (bx >= C->ybw) { nz[b] = 0; memset(blk[b], 0, sizeof blk[b]); blk[b][0] = blk[b - 1][0]; }
            else nz[b] = fwd_block(C, C->qt[0], S->Y + (size_t)(b >> 1) * 8 * pw + 8 * bx, pw, blk[b]);
        }
        nz[4] = fwd_block(C, C->qt[1], S->Cb + 8 * m, cw, blk[4]); nz[5] = fwd_block(C, C->qt[2], S->Cr + 8 * m, cw, blk[5]);
        if (B) {
            bw_reserve(B, 6 * 512);
            for (int b = 0; b < 4; b++) enc_block(C, B, blk[b], nz[b], &dc[0], C->dt[0], C->at[0]);
            enc_block(C, B, blk[4], nz[4], &dc[1], C->dt[1], C->at[1]); enc_block(C, B, blk[5], nz[5], &dc[2], C->dt[2], C->at[2]);
        } else { dc[0] = blk[3][0]; dc[1] = blk[4][0]; dc[2] = blk[5][0]; }
    }
}
static void code_band(const jctx *C, scratch *S, int r0, int r1, bitw *B) {
    int dc[3] = {0, 0, 0};
    if (r0 > 0) code_mcus(C, S, r0 - 1, C->mcols - 1, C->mcols, nullptr, dc);   /* the predictors after the previous band */
    B->len = 0; B->acc = 0; B->fr = 64; B->nb = 0;
    for (int my = r0; my < r1; my++) code_mcus(C, S, my, 0, C->mcols, B, dc);
    bw_reserve(B, 32); bw_flush(B);
}

/* ---------------- joining the bands: stuffing (on the workers), file output ---------------- */
/* Band i's bits follow phase[i] (0..7) bits carry[i] of the stream before it. Band i writes every output byte that ends inside its
   own bits (the first one starting with the carry); its last (phase[i] + bits) % 8 bits are band i + 1's carry. */
static inline uint32_t band_bits(const bitw *B, size_t pos, int L) {   /* L <= 8 bits from bit pos (buffer padded) */
    const uint32_t v = (uint32_t)B->b[pos >> 3] << 8 | B->b[(pos >> 3) + 1];
    return (v >> (16 - (int)(pos & 7) - L)) & ((1u << L) - 1);
}
static inline size_t band_nbits(const bitw *B) { return 8 * B->len + (size_t)B->nb; }
static void chain_next(const bitw *B, int p, uint32_t c, int *pn, uint32_t *cn) {   /* (phase, carry) after band B */
    const size_t nb = band_nbits(B); const int L = (int)((p + nb) & 7);
    *pn = L;
    if (!L) *cn = 0;
    else if (nb >= (size_t)L) *cn = band_bits(B, nb - (size_t)L, L);
    else *cn = (c << nb) | band_bits(B, 0, (int)nb);
}
typedef struct { uint8_t *o; size_t n, cap; } sbuf;
static inline uint8_t *stuff_byte(uint8_t *o, uint8_t v) { *o++ = v; if (v == 0xFF) *o++ = 0; return o; }
static void stuff_band(const bitw *B, int p, uint32_t c, sbuf *S) {
    const size_t nout = (p + band_nbits(B)) >> 3; const uint8_t *in = B->b;
    if (S->cap < 2 * nout + 16) { S->cap = 2 * nout + 65536; free(S->o); S->o = malloc(S->cap); }
    uint8_t *o = S->o; size_t k = 0;
    if (!p) {
        for (; k + 8 <= nout; k += 8) {
            uint64_t x; memcpy(&x, in + k, 8); const uint64_t t = ~x;
            if (!((t - 0x0101010101010101ull) & ~t & 0x8080808080808080ull)) { memcpy(o, &x, 8); o += 8; }
            else for (int j = 0; j < 8; j++) o = stuff_byte(o, in[k + j]);
        }
        for (; k < nout; k++) o = stuff_byte(o, in[k]);
    } else {
        if (nout) { o = stuff_byte(o, (uint8_t)((c << (8 - p)) | (in[0] >> p))); k = 1; }
        for (; k + 8 <= nout; k += 8) {   /* output bytes k .. k + 7 = input bits from 8 k - p */
            uint64_t x; memcpy(&x, in + k - 1, 8); x = (__builtin_bswap64(x) << (8 - p)) | (in[k + 7] >> p);
            const uint64_t t = ~x;
            if (!((t - 0x0101010101010101ull) & ~t & 0x8080808080808080ull)) { const uint64_t be = __builtin_bswap64(x); memcpy(o, &be, 8); o += 8; }
            else for (int j = 7; j >= 0; j--) o = stuff_byte(o, (uint8_t)(x >> (8 * j)));
        }
        for (; k < nout; k++) o = stuff_byte(o, (uint8_t)((in[k - 1] << (8 - p)) | (in[k] >> p)));
    }
    S->n = (size_t)(o - S->o);
}

typedef struct {
    const jctx *C; int B, nband, nslot;
    bitw *bits; sbuf *out;               /* per slot */
    uint8_t *enc, *stuffed;              /* per band */
    int *phase; uint32_t *carry; int chain;   /* phase / carry known for bands 0 .. chain */
    pthread_mutex_t mu; pthread_cond_t cv; int next, written;
} pool;
static void band_range(const pool *P, int i, int *r0, int *r1) { *r0 = i * P->B; *r1 = *r0 + P->B < P->C->mrows ? *r0 + P->B : P->C->mrows; }
static void chain_advance(pool *P) {   /* (under the lock) */
    while (P->chain < P->nband && P->enc[P->chain]) {
        const int c = P->chain; chain_next(&P->bits[c % P->nslot], P->phase[c], P->carry[c], &P->phase[c + 1], &P->carry[c + 1]); P->chain++;
    }
}
static void *worker(void *arg) {
    pool *P = arg; scratch S; scratch_init(P->C, &S);
    pthread_mutex_lock(&P->mu);
    for (;;) {
        while (P->next < P->nband && P->next >= P->written + P->nslot) pthread_cond_wait(&P->cv, &P->mu);
        if (P->next >= P->nband) break;
        const int i = P->next++; pthread_mutex_unlock(&P->mu);
        int r0, r1; band_range(P, i, &r0, &r1);
        code_band(P->C, &S, r0, r1, &P->bits[i % P->nslot]);
        pthread_mutex_lock(&P->mu); P->enc[i] = 1; chain_advance(P); pthread_cond_broadcast(&P->cv);
        while (P->chain < i) pthread_cond_wait(&P->cv, &P->mu);
        const int p = P->phase[i]; const uint32_t c = P->carry[i]; pthread_mutex_unlock(&P->mu);
        stuff_band(&P->bits[i % P->nslot], p, c, &P->out[i % P->nslot]);
        pthread_mutex_lock(&P->mu); P->stuffed[i] = 1; pthread_cond_broadcast(&P->cv);
    }
    pthread_mutex_unlock(&P->mu); scratch_free(&S);
    return nullptr;
}

static int own_encode(FILE *f, jctx *C, int threads) {
    const int nc = C->nc;
    C->mcuh = nc == 3 ? 16 : 8; C->mcols = (C->w + C->mcuh - 1) / C->mcuh; C->mrows = (C->h + C->mcuh - 1) / C->mcuh;
    C->ybw = (C->w + 7) / 8; C->ybh = (C->h + 7) / 8; C->pw = C->mcols * C->mcuh;
    /* bands of ~1 Mpx, at least 4 per thread when the image allows */
    int B = (int)((1 << 20) / ((long long)C->pw * C->mcuh)); if (B < 1) B = 1;
    if (threads > 1) { const int b2 = (C->mrows + 4 * threads - 1) / (4 * threads); if (b2 < B) B = b2 < 1 ? 1 : b2; }
    const int nband = (C->mrows + B - 1) / B; if (threads > nband) threads = nband; if (threads > 256) threads = 256;
    int err = fwrite(C->hdr, 1, C->hdrlen, f) != C->hdrlen;
    pool P = {.C = C, .B = B, .nband = nband, .nslot = threads > 1 ? 2 * threads : 1};
    P.bits = calloc((size_t)P.nslot, sizeof *P.bits); P.out = calloc((size_t)P.nslot, sizeof *P.out);
    P.enc = calloc((size_t)nband, 1); P.stuffed = calloc((size_t)nband, 1);
    P.phase = calloc((size_t)nband + 1, sizeof *P.phase); P.carry = calloc((size_t)nband + 1, sizeof *P.carry);
    if (threads <= 1) {
        scratch S; scratch_init(C, &S);
        for (int i = 0; i < nband; i++) {
            int r0, r1; band_range(&P, i, &r0, &r1); code_band(C, &S, r0, r1, &P.bits[0]);
            stuff_band(&P.bits[0], P.phase[i], P.carry[i], &P.out[0]); chain_next(&P.bits[0], P.phase[i], P.carry[i], &P.phase[i + 1], &P.carry[i + 1]);
            if (P.out[0].n && fwrite(P.out[0].o, 1, P.out[0].n, f) != P.out[0].n) err = 1;
        }
        scratch_free(&S);
    } else {
        pthread_mutex_init(&P.mu, nullptr); pthread_cond_init(&P.cv, nullptr);
        pthread_t th[256];
        for (int t = 0; t < threads; t++) pthread_create(&th[t], nullptr, worker, &P);
        for (int i = 0; i < nband; i++) {
            pthread_mutex_lock(&P.mu); while (!P.stuffed[i]) pthread_cond_wait(&P.cv, &P.mu); pthread_mutex_unlock(&P.mu);
            const sbuf *o = &P.out[i % P.nslot]; if (o->n && fwrite(o->o, 1, o->n, f) != o->n) err = 1;
            pthread_mutex_lock(&P.mu); P.written = i + 1; pthread_cond_broadcast(&P.cv); pthread_mutex_unlock(&P.mu);
        }
        for (int t = 0; t < threads; t++) pthread_join(th[t], nullptr);
        pthread_mutex_destroy(&P.mu); pthread_cond_destroy(&P.cv);
    }
    /* flush_bits: the last carry padded with 1 bits to a byte; EOI */
    uint8_t tail[4], *o = tail; const int L = P.phase[nband];
    if (L) o = stuff_byte(o, (uint8_t)((P.carry[nband] << (8 - L)) | ((1u << (8 - L)) - 1)));
    *o++ = 0xFF; *o++ = 0xD9;
    if (fwrite(tail, 1, (size_t)(o - tail), f) != (size_t)(o - tail)) err = 1;
    for (int s = 0; s < P.nslot; s++) { free(P.bits[s].b); free(P.out[s].o); }
    free(P.bits); free(P.out); free(P.enc); free(P.stuffed); free(P.phase); free(P.carry);
    return err ? -1 : 0;
}

int jpar_encode(FILE *f, int w, int h, int nc, int quality, const uint8_t *px, jpar_rows_fn fn, void *u, int threads) {
    if (w < 1 || h < 1 || (nc != 1 && nc != 3) || (!px && !fn)) return -1;
    pthread_once(&zz_once, zz_init);
    if (threads <= 0) threads = jpar_threads();
    jpar_last_own = 0;
    const int own = w <= 65500 && h <= 65500 && quality >= 1 && quality <= 100 && (jpar_force_own || (long long)w * h >= (1 << 18));
    if (own) {
        jctx C = {.w = w, .h = h, .nc = nc, .px = px, .fn = fn, .u = u};
        if (!ctx_header(&C, quality)) { jpar_last_own = 1; const int rc = own_encode(f, &C, threads); free(C.hdr); return rc; }
        free(C.hdr);
    }
    return lj_encode(f, w, h, nc, quality, px, fn, u);
}
int jpar_write(const char *path, int w, int h, int nc, int quality, const uint8_t *px, jpar_rows_fn fn, void *u, int threads) {
    FILE *f = fopen(path, "wb"); if (!f) return -1;
    const int rc = jpar_encode(f, w, h, nc, quality, px, fn, u, threads);
    return fclose(f) || rc ? -1 : 0;
}
