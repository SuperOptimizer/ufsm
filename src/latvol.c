/* latvol host codec (latvol.h): the tables, the chunk encoder / decoder and the integer transform. The device kernels
   (src/nn_latvol.cu) are checked against these byte for byte (tests/test_latvol.c). */
#include "latvol.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int hb(uint32_t v) { return 31 - __builtin_clz(v); }
static int alpha(int tab0) { return tab0 ? 19 : 9; }   /* tokens of the mean table (air + hyb of a zigzag 16-bit residual), of an axis table (hyb of <= 127) */

/* counts -> frequencies summing to LV_L, every token of the alphabet >= 1 */
static void normalise(const uint64_t *cnt, int ns, uint16_t *f) {
    uint64_t tot = 0; for (int s = 0; s < ns; s++) tot += cnt[s];
    int sum = 0, big = 0;
    for (int s = 0; s < LV_NSYM; s++) f[s] = 0;
    for (int s = 0; s < ns; s++) {
        const double p = tot ? (double)cnt[s] / (double)tot : 1.0 / ns; int v = (int)lrint(p * LV_L); if (v < 1) v = 1;
        f[s] = (uint16_t)v; sum += v; if (f[s] > f[big]) big = s;
    }
    while (sum != LV_L) {   /* the surplus / deficit from the largest entries (their cost changes least) */
        big = 0; for (int s = 1; s < ns; s++) if (f[s] > f[big]) big = s;
        const int d = sum > LV_L ? (sum - LV_L < f[big] - 1 ? sum - LV_L : f[big] - 1) : LV_L - sum;
        if (sum > LV_L) { f[big] = (uint16_t)(f[big] - d); sum -= d; } else { f[big] = (uint16_t)(f[big] + d); sum += d; }
    }
}
static int build_tables(lv_codec *lc) {
    lc->dt = malloc((size_t)lc->ntab * LV_L * sizeof *lc->dt); lc->est = malloc((size_t)lc->ntab * LV_L * sizeof *lc->est);
    lc->edn = malloc((size_t)lc->ntab * sizeof *lc->edn); lc->edf = malloc((size_t)lc->ntab * sizeof *lc->edf);
    if (!lc->dt || !lc->est || !lc->edn || !lc->edf) return -1;
    for (int t = 0; t < lc->ntab; t++) {
        const uint16_t *f = lc->freq[t]; uint8_t sym[LV_L]; int sum = 0;
        for (int s = 0; s < LV_NSYM; s++) sum += f[s];
        if (sum != LV_L) return -1;
        int pos = 0; const int step = (LV_L >> 1) + (LV_L >> 3) + 3;
        for (int s = 0; s < LV_NSYM; s++) for (int i = 0; i < f[s]; i++) { sym[pos] = (uint8_t)s; pos = (pos + step) & (LV_L - 1); }
        int next[LV_NSYM], cum[LV_NSYM], c = 0;
        for (int s = 0; s < LV_NSYM; s++) { next[s] = f[s]; cum[s] = c; c += f[s]; }
        int rank[LV_NSYM] = {0};
        for (int x = 0; x < LV_L; x++) {
            const int s = sym[x], n = next[s]++, nb = LV_TLOG - hb((uint32_t)n);
            lc->dt[(size_t)t * LV_L + x] = (uint32_t)s << 24 | (uint32_t)nb << 16 | (uint32_t)((n << nb) - LV_L);
            lc->est[(size_t)t * LV_L + cum[s] + rank[s]++] = (uint16_t)(LV_L + x);
        }
        for (int s = 0; s < LV_NSYM; s++) {
            if (!f[s]) { lc->edn[t][s] = 0; lc->edf[t][s] = 0; continue; }
            if (f[s] == 1) { lc->edn[t][s] = (LV_TLOG << 16) - LV_L; lc->edf[t][s] = cum[s] - 1; }
            else { const int mb = LV_TLOG - hb((uint32_t)f[s] - 1); lc->edn[t][s] = (mb << 16) - (f[s] << mb); lc->edf[t][s] = cum[s] - f[s]; }
        }
    }
    return 0;
}
int lv_codec_init(lv_codec *lc, int C, const float *basis, float step, float sd_ref, const uint8_t *map, int ntab, const uint64_t (*hist)[LV_NSYM]) {
    memset(lc, 0, sizeof *lc);
    if (C < 1 || C > LV_MAXC || !(step > 0) || !(sd_ref > 0)) return -1;
    lc->C = C; lc->step = step; lc->sd_ref = sd_ref; lc->step_raw = step * sd_ref;
    for (int c = 0; c < C; c++) for (int k = 0; k < 64; k++) { lc->basis[c][k] = basis[c * 64 + k]; lc->bi[c][k] = (int16_t)lrintf(basis[c * 64 + k] * (float)(1 << LV_BSHIFT)); }
    lc->ntab = map ? ntab : C;
    if (lc->ntab < 1 || lc->ntab > LV_MAXC) return -1;
    for (int c = 0; c < C; c++) { lc->map[c] = map ? map[c] : (uint8_t)c; if (lc->map[c] >= lc->ntab || (c && !lc->map[c])) return -1; }
    if (lc->map[0]) return -1;   /* table 0 is the mean's alone (its alphabet has the air token) */
    static const uint64_t flat[LV_NSYM] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    for (int t = 0; t < lc->ntab; t++) normalise(hist ? hist[t] : flat, alpha(t == 0), lc->freq[t]);
    return build_tables(lc);
}
void lv_codec_free(lv_codec *lc) { free(lc->dt); free(lc->est); free(lc->edn); free(lc->edf); lc->dt = nullptr; lc->est = nullptr; lc->edn = nullptr; lc->edf = nullptr; }

/* latvol.bin: u32 "LVT1", u32 C, u32 ntab, f32 step_raw, f32 sd_ref, f32 step, f32 drop_ms, f32 basis[C][64], u8 map[64], u16 freq[ntab][LV_NSYM] */
int lv_codec_write(const lv_codec *lc, const char *path) {
    FILE *f = fopen(path, "wb"); if (!f) return -1;
    const uint32_t hd[3] = {0x3154564cu, (uint32_t)lc->C, (uint32_t)lc->ntab}; const float fl[4] = {lc->step_raw, lc->sd_ref, lc->step, lc->drop_ms};
    int ok = fwrite(hd, sizeof hd, 1, f) == 1 && fwrite(fl, sizeof fl, 1, f) == 1 && fwrite(lc->basis, 64 * sizeof(float), (size_t)lc->C, f) == (size_t)lc->C
          && fwrite(lc->map, 1, LV_MAXC, f) == LV_MAXC && fwrite(lc->freq, sizeof lc->freq[0], (size_t)lc->ntab, f) == (size_t)lc->ntab;
    return fclose(f) || !ok ? -1 : 0;
}
int lv_codec_parse(lv_codec *lc, const uint8_t *b, size_t n) {
    memset(lc, 0, sizeof *lc);
    uint32_t hd[3]; float fl[4];
    if (n < sizeof hd + sizeof fl) return -1;
    memcpy(hd, b, sizeof hd); memcpy(fl, b + sizeof hd, sizeof fl);
    if (hd[0] != 0x3154564cu || hd[1] < 1 || hd[1] > LV_MAXC || hd[2] < 1 || hd[2] > LV_MAXC) return -1;
    lc->C = (int)hd[1]; lc->ntab = (int)hd[2]; lc->step_raw = fl[0]; lc->sd_ref = fl[1]; lc->step = fl[2]; lc->drop_ms = fl[3];
    const size_t need = sizeof hd + sizeof fl + (size_t)lc->C * 64 * sizeof(float) + LV_MAXC + (size_t)lc->ntab * sizeof lc->freq[0];
    if (n < need || !(lc->step_raw > 0)) return -1;
    const uint8_t *p = b + sizeof hd + sizeof fl;
    memcpy(lc->basis, p, (size_t)lc->C * 64 * sizeof(float)); p += (size_t)lc->C * 64 * sizeof(float);
    memcpy(lc->map, p, LV_MAXC); p += LV_MAXC;
    memcpy(lc->freq, p, (size_t)lc->ntab * sizeof lc->freq[0]);
    for (int c = 0; c < lc->C; c++) { if (lc->map[c] >= lc->ntab || (c ? !lc->map[c] : lc->map[c])) return -1;
                                      for (int k = 0; k < 64; k++) lc->bi[c][k] = (int16_t)lrintf(lc->basis[c][k] * (float)(1 << LV_BSHIFT)); }
    return build_tables(lc);
}

int lv_forward(const lv_codec *lc, const uint8_t *ct, int16_t *q) {
    const int C = lc->C; const float fs = 1.0f / ((float)(1 << LV_BSHIFT) * lc->step_raw); int nmat = 0;
    for (int z = 0; z < LV_CELLS; z++) for (int y = 0; y < LV_CELLS; y++) for (int x = 0; x < LV_CELLS; x++) {
        int16_t v[64]; int any = 0;
        for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++) { const uint8_t *r = ct + ((size_t)(4 * z + a) * 128 + 4 * y + b) * 128 + 4 * x; for (int e = 0; e < 4; e++) { v[a * 16 + b * 4 + e] = r[e]; any |= r[e]; } }
        int16_t *o = q + ((size_t)(z * LV_CELLS + y) * LV_CELLS + x) * C;
        if (!any) { memset(o, 0, (size_t)C * sizeof *o); continue; }
        nmat++;
        int nz = 0;
        for (int c = 0; c < C; c++) { int32_t acc = 0; const int16_t *b = lc->bi[c]; for (int k = 0; k < 64; k++) acc += (int32_t)b[k] * v[k]; o[c] = (int16_t)lv_fwd_index(acc, fs, c); nz |= c && o[c]; }
        o[0] = (int16_t)lv_fwd_mean(o[0], nz); nmat -= !o[0];
    }
    return nmat;
}
void lv_inverse(const lv_codec *lc, const int16_t *q, uint8_t *ct) {
    const int C = lc->C; const float rs = lc->step_raw / (float)(1 << LV_BSHIFT);
    for (int z = 0; z < LV_CELLS; z++) for (int y = 0; y < LV_CELLS; y++) for (int x = 0; x < LV_CELLS; x++) {
        const int16_t *o = q + ((size_t)(z * LV_CELLS + y) * LV_CELLS + x) * C;
        int32_t acc[64] = {0};
        if (o[0]) for (int c = 0; c < C; c++) if (o[c]) { const int32_t qc = o[c]; const int16_t *b = lc->bi[c]; for (int k = 0; k < 64; k++) acc[k] += qc * b[k]; }
        for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++) { uint8_t *r = ct + ((size_t)(4 * z + a) * 128 + 4 * y + b) * 128 + 4 * x; for (int e = 0; e < 4; e++) r[e] = o[0] ? lv_inv_byte(acc[a * 16 + b * 4 + e], rs) : 0; }
    }
}

static inline uint32_t tok_of(uint32_t a, uint32_t *k, uint32_t *raw) { if (a < 4) { *k = 0; *raw = 0; return a; } const int h = hb(a); *k = (uint32_t)h; *raw = a & ((1u << h) - 1u); return (uint32_t)h + 2u; }
void lv_hist(const lv_codec *lc, const int16_t *q, uint64_t (*hist)[LV_NSYM]) {
    const int C = lc->C;
    for (int s = 0; s < LV_NSUB; s++) {
        int pred = 0;
        for (int cell = 0; cell < LV_SUBCELLS; cell++) {
            const int z = 4 * (s >> 6) + (cell >> 4), y = 4 * (s >> 3 & 7) + (cell >> 2 & 3), x = 4 * (s & 7) + (cell & 3);
            const int16_t *o = q + ((size_t)(z * LV_CELLS + y) * LV_CELLS + x) * C; uint32_t k, raw;
            if (!o[0]) { hist[0][0]++; continue; }
            const int r = o[0] - pred; pred = o[0];
            hist[0][1 + tok_of((uint32_t)(r << 1) ^ (uint32_t)(r >> 31), &k, &raw)]++;
            for (int c = 1; c < C; c++) hist[lc->map[c]][tok_of((uint32_t)abs(o[c]), &k, &raw)]++;
        }
    }
}

/* the bit writer, prepending: the bits put last are read first; words are stored downwards from the end of the buffer */
typedef struct { uint32_t *wp; uint64_t acc; uint32_t cnt; } lv_bw;
static inline void bw_put(lv_bw *b, uint32_t v, uint32_t n) {   /* n <= 31, v < 2^n */
    b->acc = b->acc << n | v; b->cnt += n;
    if (b->cnt >= 32) { b->cnt -= 32; *--b->wp = (uint32_t)(b->acc >> b->cnt); b->acc &= (1ull << b->cnt) - 1; }
}
static inline void tans_put(const lv_codec *lc, lv_bw *b, uint32_t *X, int tab, uint32_t s) {
    const uint32_t nb = (uint32_t)((int32_t)*X + lc->edn[tab][s]) >> 16;
    bw_put(b, *X & ((1u << nb) - 1u), nb);
    *X = lc->est[(size_t)tab * LV_L + (*X >> nb) + lc->edf[tab][s]];
}
/* one substream's 64 cells (qs[cell] -> the cell's C indices) into the words ending at end; returns the word count (0: all air) */
static uint32_t encode_sub(const lv_codec *lc, const int16_t *const qs[LV_SUBCELLS], uint32_t *end) {
    const int C = lc->C; int res[LV_SUBCELLS], pred = 0, any = 0;
    for (int cell = 0; cell < LV_SUBCELLS; cell++) { const int q0 = qs[cell][0]; if (q0) { res[cell] = q0 - pred; pred = q0; any = 1; } else res[cell] = 0; }
    if (!any) return 0;
    lv_bw b = {end, 0, 0}; uint32_t X = LV_L, k, raw;
    for (int cell = LV_SUBCELLS - 1; cell >= 0; cell--) {
        const int16_t *o = qs[cell];
        if (!o[0]) { tans_put(lc, &b, &X, 0, 0); continue; }
        for (int c = C - 1; c >= 1; c--) {
            const uint32_t a = (uint32_t)abs(o[c]), t = tok_of(a, &k, &raw);
            if (a) bw_put(&b, o[c] < 0, 1);
            if (k) bw_put(&b, raw, k);
            tans_put(lc, &b, &X, lc->map[c], t);
        }
        const int r = res[cell]; const uint32_t t = tok_of((uint32_t)(r << 1) ^ (uint32_t)(r >> 31), &k, &raw);
        if (k) bw_put(&b, raw, k);
        tans_put(lc, &b, &X, 0, t + 1);
    }
    bw_put(&b, X - LV_L, LV_TLOG); bw_put(&b, 1, 1);
    if (b.cnt) bw_put(&b, 0, 32 - b.cnt);
    return (uint32_t)(end - b.wp);
}
int lv_encode_chunk(const lv_codec *lc, const int16_t *q, uint8_t *enc, size_t *len) {
    const int C = lc->C; uint32_t *tmp = malloc((size_t)LV_SUB_WORDS * 4); if (!tmp) return -1;
    size_t off = LV_HDR; int any = 0; uint16_t words[LV_NSUB];
    for (int s = 0; s < LV_NSUB; s++) {
        const int16_t *qs[LV_SUBCELLS];
        for (int cell = 0; cell < LV_SUBCELLS; cell++) {
            const int z = 4 * (s >> 6) + (cell >> 4), y = 4 * (s >> 3 & 7) + (cell >> 2 & 3), x = 4 * (s & 7) + (cell & 3);
            qs[cell] = q + ((size_t)(z * LV_CELLS + y) * LV_CELLS + x) * C;
        }
        const uint32_t nw = encode_sub(lc, qs, tmp + LV_SUB_WORDS);
        words[s] = (uint16_t)nw; any |= nw != 0;
        memcpy(enc + off, tmp + LV_SUB_WORDS - nw, (size_t)nw * 4); off += (size_t)nw * 4;
    }
    free(tmp);
    if (!any) { *len = 0; return 0; }
    const uint32_t magic = LV_MAGIC; memcpy(enc, &magic, 4); enc[4] = 1; enc[5] = (uint8_t)C; enc[6] = enc[7] = 0; memcpy(enc + 8, words, sizeof words);
    *len = off; return 0;
}
int lv_decode_chunk(const lv_codec *lc, const uint8_t *enc, size_t len, int16_t *q, const uint8_t *submask) {
    const int C = lc->C; uint32_t magic; uint16_t words[LV_NSUB];
    if (len < LV_HDR) return -1;
    memcpy(&magic, enc, 4); memcpy(words, enc + 8, sizeof words);
    if (magic != LV_MAGIC || enc[4] != 1 || enc[5] != C) return -1;
    size_t off = LV_HDR; uint32_t *w = malloc((size_t)LV_SUB_WORDS * 4); if (!w) return -1;
    int bad = 0;
    for (int s = 0; s < LV_NSUB && !bad; s++) {
        const uint32_t nw = words[s]; int16_t sq[LV_SUBCELLS * LV_MAXC];
        if (nw > LV_SUB_WORDS || off + (size_t)nw * 4 > len) { bad = 1; break; }
        memset(sq, 0, (size_t)LV_SUBCELLS * C * sizeof *sq);
        if (nw && (!submask || submask[s])) {
            memcpy(w, enc + off, (size_t)nw * 4);   /* (a copy: the stream may be unaligned) */
            lv_dec d; bad = lv_dec_init(&d, w, nw);
            for (int cell = 0; cell < LV_SUBCELLS && !bad; cell++) lv_dec_cell(&d, lc->dt, lc->map, C, sq + cell * C);
            if (!bad) bad = lv_dec_end(&d);
        }
        off += (size_t)nw * 4;
        for (int cell = 0; cell < LV_SUBCELLS; cell++) {
            const int z = 4 * (s >> 6) + (cell >> 4), y = 4 * (s >> 3 & 7) + (cell >> 2 & 3), x = 4 * (s & 7) + (cell & 3);
            memcpy(q + ((size_t)(z * LV_CELLS + y) * LV_CELLS + x) * C, sq + cell * C, (size_t)C * sizeof *sq);
        }
    }
    free(w);
    return bad || off != len ? -1 : 0;
}
int lv_decode_voxels(const lv_codec *lc, const uint8_t *enc, size_t len, uint8_t *ct, int16_t *scratch) {
    /* scratch == nullptr: the calling thread's own buffer, kept (z3_read's chunk workers: a fresh 1.9 MB malloc per chunk is an mmap /
       munmap pair, page faults and TLB shootdowns that serialised the workers: 8 threads read a P 512 box no faster than 1, 2026-11-12) */
    static thread_local int16_t *tq; static thread_local size_t tqn;
    const size_t need = (size_t)LV_CELLS * LV_CELLS * LV_CELLS * lc->C;
    int16_t *q = scratch;
    if (!q) { if (tqn < need) { free(tq); tq = malloc(need * sizeof *tq); tqn = tq ? need : 0; } q = tq; }
    if (!q) return -1;
    const int r = lv_decode_chunk(lc, enc, len, q, nullptr);
    if (!r) lv_inverse(lc, q, ct);
    return r;
}
void lv_pack_cells(const int16_t *q, size_t n, int C, int8_t *out, int stride) {
    const int na = (C < stride - 1 ? C : stride - 1);
    for (size_t i = 0; i < n; i++) {
        const int16_t *o = q + i * C; int8_t *p = out + i * stride;
        p[0] = (int8_t)(uint8_t)(o[0] & 255); p[1] = (int8_t)(o[0] >> 8);
        for (int c = 1; c < na; c++) p[c + 1] = (int8_t)o[c];
        for (int c = na < 1 ? 1 : na; c < stride - 1; c++) p[c + 1] = 0;
    }
}
