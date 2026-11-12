/* latvol: a CT volume stored as the latent cells the latent-grid networks read (train --lat 4 --lat-basis), as a zarr v3 inner-chunk
   codec ("latvol") in the usual CT layout: a uint8 [Z][Y][X] array with the CT level's shape, sharding_indexed shards of 128^3 chunks.
   A chunk holds its 32^3 cells of 4^3 voxels: per cell the quantiser indices q_c = rint(<b_c, v> / step_raw) of the cell's 64 CT bytes on
   the first C rows of the orthonormal cell basis (latfit.h: row 0 the mean direction, then the principal axes; the basis and step of the
   checkpoint's lat_basis / lat_step x sd_ref), entropy-coded. Two readers:
     cells:  the indices themselves, int8 [d][h][w](C + 1) (bytes 0-1 the mean index, low byte uint8 / high byte int8; 2.. the axes; all
             zero: an air cell: no CT, or nothing above the quantiser), which is what the trunk's input kernel takes (nn_lat_pred_input_q): entropy decode, nothing else;
     voxels: the inverse transform, 1..255 in material cells and 0 in air cells (z3_read of such an array: every other reader).
   Both are integer-exact: the basis rows are int16 (x 2^14), the sums int32, one float multiply and a rint each way, so the host code
   here and the device kernels (src/nn_latvol.cu, built without fast math) give the same bytes.

   Chunk stream (little endian): u32 magic "LVL1", u8 version 1, u8 C, u16 0; u16 words[512]; then the 512 substreams, each words[s]
   32-bit words (0: all 64 cells air). Substream s = (sz 8 + sy) 8 + sx holds cells [4 sz, 4 sz + 4) x [4 sy, ..) x [4 sx, ..) in z, y, x
   order: 64 independent serial decodes of <= 29 symbols a cell, one device thread each, and a 16-voxel halo is one substream layer.
   A substream is one bit stream read LSB first: zero padding, a 1 bit, the 10-bit tANS state, then per cell
     the mean: token t of table tab[0]; t = 0: air cell (no more symbols, the predictor keeps its value); else u = hyb(t - 1) and
               q_0 = pred + unzigzag(u), pred = q_0 (pred starts at 0 in every substream; q_0 in 1..32767);
     axis c = 1 .. C - 1: token t of table tab[c]; a = hyb(t); a > 0: one sign bit (1: negative); |q_c| <= 127.
   hyb(t) (volcomp's hybrid integer): t < 4: the value t; else k = t - 2 raw bits e, value (1 << k) | e. After a token the decoder
   reads the tANS state's nb bits, then the token's raw bits, then the sign. The tANS (FSE construction, 1024 states) tables are the
   array's (latvol.bin beside zarr.json: one normalised frequency row of LV_NSYM per table), not per chunk. */
#ifndef UFSM_LATVOL_H
#define UFSM_LATVOL_H
#include <stdint.h>
#include <stddef.h>

#define LV_MAGIC 0x314c564cu   /* "LVL1" */
enum { LV_CELLS = 32, LV_SUB = 4, LV_NSUB = 512, LV_SUBCELLS = 64, LV_HDR = 8 + 2 * 512, LV_TLOG = 10, LV_L = 1024, LV_NSYM = 20, LV_MAXC = 64,
       LV_BSHIFT = 14, LV_SUB_WORDS = 2400,   /* a substream's bound: 64 cells x (27 + 17 x 63 bits) + 43 bits < 2400 words */
       LV_CHUNK_BOUND = LV_HDR + 512 * LV_SUB_WORDS * 4 };

#ifdef __CUDACC__
#define LV_FN __device__ __forceinline__
#define LV_RINT(x) rintf(x)
#else
#define LV_FN static inline
#define LV_RINT(x) __builtin_rintf(x)
#endif

/* one decode entry: bits 0-15 the next state's base, 16-23 the bits to read, 24-31 the token */
typedef uint32_t lv_dent;

/* the bit reader over 32-bit words, LSB first (a two-word window; words past the stream's end read as 0, so a corrupt stream never
   reads outside its nwords) */
typedef struct { const uint32_t *w, *e; uint32_t lo, hi, off; } lv_br;
LV_FN void lv_br_init(lv_br *r, const uint32_t *w, uint32_t nwords) { r->w = w + 2; r->e = w + nwords; r->lo = w[0]; r->hi = nwords > 1u ? w[1] : 0u; r->off = 0; }
LV_FN uint32_t lv_br_get(lv_br *r, uint32_t n) {   /* n <= 31 */
    const uint32_t v = (r->off ? (r->lo >> r->off) | (r->hi << (32u - r->off)) : r->lo) & ((1u << n) - 1u);
    r->off += n;
    if (r->off >= 32u) { r->off -= 32u; r->lo = r->hi; r->hi = r->w < r->e ? *r->w : 0u; r->w++; }
    return v;
}
LV_FN uint32_t lv_hyb(lv_br *r, uint32_t t) { if (t < 4u) return t; const uint32_t k = t - 2u; return (1u << k) | lv_br_get(r, k); }

/* the decoder's state over one substream */
typedef struct { lv_br r; uint32_t x, used, end; int pred, bad; } lv_dec;
/* start a substream of nwords > 0 words: 0, or 1 when its start marker is missing */
LV_FN int lv_dec_init(lv_dec *d, const uint32_t *w, uint32_t nwords) {
    lv_br_init(&d->r, w, nwords); d->pred = 0; d->bad = 0; d->end = nwords * 32u; d->x = 0; d->used = 0;
    if (!d->r.lo) return 1;
    uint32_t pad = 0; while (!(d->r.lo >> pad & 1u)) pad++;
    lv_br_get(&d->r, pad); lv_br_get(&d->r, 1);
    d->x = lv_br_get(&d->r, LV_TLOG); d->used = pad + 1u + LV_TLOG;
    return 0;
}
/* the next cell's C indices into qc (untouched for an air cell); returns 1 for a material cell, 0 for air. dt: the tables [ntab][LV_L],
   map[c] the table of channel c. Range errors set d->bad. */
LV_FN int lv_dec_cell(lv_dec *d, const lv_dent *dt, const uint8_t *map, int C, int16_t *qc) {
    lv_dent e = dt[(size_t)map[0] * LV_L + d->x]; uint32_t t = e >> 24, nb = e >> 16 & 255u;
    d->x = (e & 0xffffu) + lv_br_get(&d->r, nb); d->used += nb;
    if (!t) return 0;
    { const uint32_t k = t - 1u < 4u ? 0u : t - 3u; const uint32_t u = lv_hyb(&d->r, t - 1u); d->used += k;
      d->pred += (int)(u >> 1) ^ -(int)(u & 1u); if (d->pred < 1 || d->pred > 32767) d->bad = 1; qc[0] = (int16_t)d->pred; }
    for (int c = 1; c < C; c++) {
        e = dt[(size_t)map[c] * LV_L + d->x]; t = e >> 24; nb = e >> 16 & 255u;
        d->x = (e & 0xffffu) + lv_br_get(&d->r, nb); d->used += nb;
        int v = 0;
        if (t) { const uint32_t k = t < 4u ? 0u : t - 2u; const uint32_t a = lv_hyb(&d->r, t), sg = lv_br_get(&d->r, 1); d->used += k + 1u;
                 d->bad |= a > 127u; v = sg ? -(int)a : (int)a; }
        qc[c] = (int16_t)v;
    }
    if (d->used > d->end) d->bad = 1;
    return 1;
}
/* after the 64th cell: 0 when the stream ended exactly, in the encoder's start state, with no range error */
LV_FN int lv_dec_end(const lv_dec *d) { return d->bad || d->x != 0u || d->used != d->end; }

/* the transform, integer-exact. bi: the basis rows x 2^14 as int16 [C][64] (sub-voxel k = (z 4 + y) 4 + x). */
LV_FN int lv_fwd_index(int32_t acc, float fs, int c) {   /* acc = sum_k bi[c][k] v[k]; fs = 1 / (2^14 step_raw) */
    const float f = (float)acc * fs; int q = (int)LV_RINT(f);
    if (c) { if (q > 127) q = 127; if (q < -127) q = -127; } else { if (q > 32767) q = 32767; if (q < 0) q = 0; }
    return q;
}
/* a cell's indices after lv_fwd_index of every channel: a mean index of 0 is the air cell's mark, so a cell whose mean rounds to 0 keeps
   its axes with the mean 1 (nz: any axis index nonzero), and a cell whose indices are all 0 is air (its few grey levels are below the step) */
LV_FN int lv_fwd_mean(int q0, int nz) { return q0 ? q0 : nz ? 1 : 0; }
LV_FN uint8_t lv_inv_byte(int32_t acc, float rs) {   /* acc = sum_c q_c bi[c][k]; rs = step_raw / 2^14: a material cell's byte */
    const float f = (float)acc * rs; int v = (int)LV_RINT(f);
    return (uint8_t)(v < 1 ? 1 : v > 255 ? 255 : v);
}

#ifndef __CUDACC__
/* ---- host side (src/latvol.c) ---- */
typedef struct {
    int C, ntab;                       /* channels kept (1..64), tables */
    float step_raw, sd_ref, step;      /* the quantiser step in grey levels = step x sd_ref */
    float drop_ms;                     /* the dropped axes' energy per material voxel (grey levels^2): a window's variance from the rebuilt CT is low by it */
    float basis[LV_MAXC][64];          /* the transform's float rows (what a checkpoint's lat_basis is compared with) */
    int16_t bi[LV_MAXC][64];           /* rint(basis x 2^14) */
    uint8_t map[LV_MAXC];              /* channel -> table */
    uint16_t freq[LV_MAXC][LV_NSYM];   /* normalised frequencies (sum LV_L; every token of the table's alphabet >= 1) */
    lv_dent *dt;                       /* [ntab][LV_L] */
    uint16_t *est; int32_t (*edn)[LV_NSYM], (*edf)[LV_NSYM];   /* encoder: state table [ntab][LV_L], deltaNbBits, deltaFindState */
} lv_codec;

/* a codec from the transform (rows [C][64]) and token histograms hist[ntab][LV_NSYM] (nullptr: flat tables); map nullptr: one table a
   channel. lv_codec_free releases its tables. */
int lv_codec_init(lv_codec *lc, int C, const float *basis, float step, float sd_ref, const uint8_t *map, int ntab, const uint64_t (*hist)[LV_NSYM]);
void lv_codec_free(lv_codec *lc);
/* latvol.bin: the codec's side file (written beside zarr.json); lv_codec_parse reads its bytes. 0 on success */
int lv_codec_write(const lv_codec *lc, const char *path);
int lv_codec_parse(lv_codec *lc, const uint8_t *b, size_t n);
/* a 128^3 CT chunk (z-major) -> its 32^3 cells' indices q [32768][C] int16 (cell z-major); returns the number of material cells */
int lv_forward(const lv_codec *lc, const uint8_t *ct, int16_t *q);
/* the token histograms of a chunk's indices added into hist[ntab][LV_NSYM] */
void lv_hist(const lv_codec *lc, const int16_t *q, uint64_t (*hist)[LV_NSYM]);
/* indices -> the chunk stream (enc: LV_CHUNK_BOUND bytes); *len 0 when every cell is air (the chunk is omitted) */
int lv_encode_chunk(const lv_codec *lc, const int16_t *q, uint8_t *enc, size_t *len);
/* the chunk stream -> indices q [32768][C] (submask nullptr: all substreams; else only substreams s with submask[s], others 0). 0 ok */
int lv_decode_chunk(const lv_codec *lc, const uint8_t *enc, size_t len, int16_t *q, const uint8_t *submask);
/* indices -> the 128^3 chunk's voxels (0 in air cells) */
void lv_inverse(const lv_codec *lc, const int16_t *q, uint8_t *ct);
/* the chunk stream -> voxels (scratch: 32768 C int16, or nullptr: allocated) */
int lv_decode_voxels(const lv_codec *lc, const uint8_t *enc, size_t len, uint8_t *ct, int16_t *scratch);
/* indices [n][C] int16 -> the device / v1 container int8 [n][stride] (bytes 0-1 the mean, 2.. the axes c < min(C, stride - 1)) */
void lv_pack_cells(const int16_t *q, size_t n, int C, int8_t *out, int stride);
#endif
#endif
