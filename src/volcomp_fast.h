/* volcomp_fast.h - volcomp lossy (mode 0) decoding, byte-identical to the reference decoder, ~10% faster on the desk.
 *
 * Include after third_party/volcomp.h. vfx_decode_sub(p, s, dst) = vf_decode_sub(p, s, dst, -1, NULL); vfx_decode_subs(p, mask,
 * dst) does every s in mask; vfx_decode() = volcomp_decode. Same tokens, the same float operations in the same order (dequantise,
 * scale, inverse transform, round), so the same bytes (tests/test_volcomp_fast.c; A/B over real chunks in the commit notes).
 * Changes against the reference loop: the two tANS lanes in fixed registers (DC and level tokens use lane 0, run tokens and the
 * EOB lane 1: a block is DC, (run, level)*, EOB), per scan position tables (natural index, step radius, DCT scale) instead of
 * recomputing them per coefficient, a level's extra bits and its sign in one read, and with AVX-512 (the desk's Zen 5) the y and z
 * transform passes on 16 lanes. Measured and dropped (slower on Zen 5 or no gain): eager branchless bit-reader refills, branchless
 * HybridUint selects, two substreams interleaved in one loop (separate coefficient lists, transforms after). The entropy decode is
 * a serial chain (~80% of a q 2 chunk); the reference was already close to what one core does. */
#ifndef UFSM_VOLCOMP_FAST_H
#define UFSM_VOLCOMP_FAST_H
#include <pthread.h>

/* per scan position: natural index | radius << 12 | zero-axis count << 18 | band << 20 */
static uint32_t vfx_pos_tab[VF_BLKV];
static const float vfx_scale_tab[4] = {0.0441941738f, 0.03125f, 0.0220970869f, 0.015625f};   /* = vf_dct_scale by zero-axis count */
static void vfx_tabs_init(void) {
    for (uint32_t pos = 0; pos < VF_BLKV; pos++) {
        uint32_t nat = VF_SCAN16[pos];
        uint32_t z0 = (uint32_t)((nat & 15u) == 0) + (uint32_t)(((nat >> 4) & 15u) == 0) + (uint32_t)(((nat >> 8) & 15u) == 0);
        vfx_pos_tab[pos] = nat | vf_radius(nat) << 12 | z0 << 18 | vf_band_of(pos) << 20;
    }
}
static pthread_once_t vfx_once = PTHREAD_ONCE_INIT;

/* ---- AVX-512 inverse transform (a build with -mavx512f, e.g. -march=native on the desk's Zen 5): vf_dct16_inv_avx2 with the
   y and z passes on all 16 x lanes per instruction. Each lane runs the same float operations in the same order as the AVX2 kernel,
   so the voxels are the same bits. */
#if defined(__AVX512F__) && VF_HAVE_AVX2 && VF_AVX2_STATIC
__attribute__((always_inline)) static inline void vfx_dct16_v16_inv(const __m512 in[16], __m512 out[16]) { VF_DCT16_INV_BODY(__m512) }
__attribute__((noinline)) static void vfx_dct16_inv_avx512(float blk[VF_BLKV], uint32_t zmask, uint32_t hmask, float out[VF_BLKV]) {
    __m512 in[16], o[16];
    for (int z = 0; z < 16; z++) {
        if (!(zmask >> z & 1u)) continue;
        float *pl = blk + z * 256;
        vf_dct16_x8(pl, true);
        if (hmask >> z & 1u) vf_dct16_x8(pl + 128, true);
        for (int t = 0; t < 16; t++) in[t] = _mm512_loadu_ps(pl + t * 16);   /* y lines: 16 x lanes */
        vfx_dct16_v16_inv(in, o);
        for (int t = 0; t < 16; t++) _mm512_storeu_ps(pl + t * 16, o[t]);
    }
    for (int y = 0; y < 16; y++) {   /* z lines */
        for (int t = 0; t < 16; t++) in[t] = zmask >> t & 1u ? _mm512_loadu_ps(blk + y * 16 + t * 256) : _mm512_setzero_ps();
        vfx_dct16_v16_inv(in, o);
        for (int t = 0; t < 16; t++) _mm512_storeu_ps(out + y * 16 + t * 256, o[t]);
    }
}
#define VFX_INV(blk, zm, hm, vox) vfx_dct16_inv_avx512(blk, zm, hm, vox)
#else
#define VFX_INV(blk, zm, hm, vox) vf_dct16_inv(blk, zm, hm, vox)
#endif

/* = vf_decode_sub(p, s, dst, -1, NULL) */
static volcomp_status vfx_decode_sub(const vf_parsed *restrict p, uint32_t s, uint8_t *restrict dst) {
    pthread_once(&vfx_once, vfx_tabs_init);
    const uint8_t *rb = p->payload + p->off[s];
    vf_rdec rd;
    if (!vf_rdec_init(&rd, rb, p->tok_n[s])) return VOLCOMP_ERR_CORRUPT;
    vf_bitr br;
    vf_br_init(&br, rb + p->tok_n[s], p->byp_n[s]);
    uint32_t x0 = rd.x[0], x1 = rd.x[1];
    int64_t prev_dc = 0;
    float blk[VF_BLKV] __attribute__((aligned(64))) = {0}, vox[VF_BLKV] __attribute__((aligned(64)));
    const vf_model *M = p->models;
    for (uint32_t bi = s * VF_BLOCKS_PER_SUB; bi < (s + 1) * VF_BLOCKS_PER_SUB; bi++) {
        const vf_dentry *e = &M[VF_DC_CTX].dt[x0];
        uint32_t t = e->sym, b;
        if (!vf_br_get(&rd.br, e->nb, &b)) return VOLCOMP_ERR_CORRUPT;
        x0 = e->ns + b;
        if (t > VF_TOKMAX_DC) return VOLCOMP_ERR_CORRUPT;
        uint32_t u;
        if (!vf_hyb_read(&br, t, &u)) return VOLCOMP_ERR_CORRUPT;
        int64_t dc = prev_dc + vf_unzigzag(u);
        if (dc < -VF_DC_ABS_MAX || dc > VF_DC_ABS_MAX) return VOLCOMP_ERR_CORRUPT;
        prev_dc = dc;
        if (dc != 0) {
            float mag = vf_dequant_dc((uint32_t)(dc < 0 ? -dc : dc), p->step[0]) * 0.015625f;
            blk[0] = dc < 0 ? -mag : mag;
        }
        uint32_t pos = 1, nnz_ac = 0, zmask = 1u, hmask = 0;
        for (;;) {
            e = &M[vf_run_ctx(pos)].dt[x1];
            t = e->sym;
            if (!vf_br_get(&rd.br, e->nb, &b)) return VOLCOMP_ERR_CORRUPT;
            x1 = e->ns + b;
            if (t == VF_TOK_EOB) break;
            if (t > VF_TOKMAX_RUN) return VOLCOMP_ERR_CORRUPT;
            uint32_t run;
            if (!vf_hyb_read(&br, t, &run)) return VOLCOMP_ERR_CORRUPT;
            pos += run;
            if (pos >= VF_BLKV) return VOLCOMP_ERR_CORRUPT;
            uint32_t pt = vfx_pos_tab[pos];
            e = &M[3u + (pt >> 20) * 2u + (run == 0u ? 1u : 0u)].dt[x0];
            uint32_t lt = e->sym;
            if (!vf_br_get(&rd.br, e->nb, &b)) return VOLCOMP_ERR_CORRUPT;
            x0 = e->ns + b;
            if (lt > VF_TOKMAX_LVL) return VOLCOMP_ERR_CORRUPT;
            uint32_t mag1, sign;
            if (lt < 4u) { mag1 = lt; if (!vf_br_get(&br, 1, &sign)) return VOLCOMP_ERR_CORRUPT; }
            else {
                uint32_t k = lt - 2u, bits;
                if (!vf_br_get(&br, k + 1u, &bits)) return VOLCOMP_ERR_CORRUPT;
                mag1 = (1u << k) | (bits & ((1u << k) - 1u)); sign = bits >> k;
            }
            if (mag1 + 1u > VF_AC_MAG_MAX) return VOLCOMP_ERR_CORRUPT;
            uint32_t nat = pt & 4095u;
            float stp = p->step[(pt >> 12) & 63u];
            float v = vf_dequant_ac(mag1 + 1u, stp) * vfx_scale_tab[(pt >> 18) & 3u];
            blk[nat] = sign ? -v : v;
            zmask |= 1u << (nat >> 8);
            hmask |= (nat >> 7 & 1u) << (nat >> 8);
            nnz_ac++;
            pos++;
        }
        uint32_t bz = bi >> 6, by = (bi >> 3) & 7u, bx = bi & 7u;
        if (nnz_ac == 0) {
            uint8_t bv = vf_flat_value(blk[0]);
            blk[0] = 0.0f;
            for (uint32_t z = 0; z < 16; z++)
                for (uint32_t y = 0; y < 16; y++) memset(dst + vf_row_off(bz, by, bx, z, y), bv, 16);
            continue;
        }
        if (p->strict) vf_dct16_inv_strict(blk, zmask, hmask, vox);
        else VFX_INV(blk, zmask, hmask, vox);
        vf_clear_planes(blk, zmask);
        vf_scatter_block(dst + vf_row_off(bz, by, bx, 0, 0), (size_t)VOLCOMP_CHUNK_DIM * VOLCOMP_CHUNK_DIM, VOLCOMP_CHUNK_DIM, vox);
    }
    rd.x[0] = x0; rd.x[1] = x1;
    if (!vf_rdec_finished(&rd) || !vf_br_finished(&br)) return VOLCOMP_ERR_CORRUPT;
    return VOLCOMP_OK;
}
/* = vf_decode_sub for every s in mask; a corrupt substream re-runs the reference, so status and bytes match it */
static inline volcomp_status vfx_decode_subs(const vf_parsed *restrict p, uint32_t mask, uint8_t *restrict dst) {
    for (uint32_t s = 0; s < VF_NSUB; s++)
        if (mask >> s & 1u && vfx_decode_sub(p, s, dst) != VOLCOMP_OK) {
            volcomp_status st = vf_decode_sub(p, s, dst, -1, NULL);
            if (st != VOLCOMP_OK) return st;
        }
    return VOLCOMP_OK;
}
/* = volcomp_decode */
static inline volcomp_status vfx_decode(const void *restrict enc, size_t enc_n, uint8_t *restrict dst, size_t dst_cap) {
    uint32_t mode;
    if (!enc || !dst || dst_cap < VOLCOMP_CHUNK_VOXELS || vll_chunk_mode(enc, enc_n, &mode) != VOLCOMP_OK || mode != VLL_MODE_LOSSY)
        return volcomp_decode(enc, enc_n, dst, dst_cap);
    vf_parsed p;
    volcomp_status st = vf_parse(enc, enc_n, &p);
    return st == VOLCOMP_OK ? vfx_decode_subs(&p, 0xffffffffu, dst) : st;
}
#endif
