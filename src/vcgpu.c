/* vcgpu.h: volcomp lossy encoding from GPU-computed levels (the tail of volcomp.h's vf_encode_lossy) */
#include "vcgpu.h"
#include "../third_party/volcomp.h"
static_assert(VCG_ENCODE_BOUND == VOLCOMP_ENCODE_BOUND);

int vcg_tables(float q, float *rstep, uint16_t *scan) {
    const uint32_t qraw = vf_q_raw(q); const float qe = (float)qraw * (1.0f / 256.0f);
    float step[VF_NRADII], inv[VF_NRADII];
    vf_steps(qe, step);
    for (uint32_t r = 0; r < VF_NRADII; r++) inv[r] = 1.0f / step[r];
    for (uint32_t c = 0; c < VF_BLKV; c++) rstep[c] = inv[vf_radius(c)] * vf_dct_scale(c);
    memcpy(scan, VF_SCAN16, sizeof VF_SCAN16);
    return vf_use_avx2();
}

int vcg_encode_levels(float q, const int32_t *dc, const uint32_t *n, const uint32_t *off, const int32_t *lv, const uint16_t *pos,
                      uint8_t *dst, size_t dst_cap, size_t *out_n) {
    if (!(q >= VOLCOMP_Q_MIN && q <= VOLCOMP_Q_MAX)) return VOLCOMP_ERR_ARG;
    *out_n = 0;
    const uint32_t qraw = vf_q_raw(q);
    const size_t sz_toks = (size_t)VF_BLOCKS * 8192u * sizeof(uint16_t), sz_byp = (size_t)VF_SUB_BYP_MAX * VF_NSUB, sz_stage = VF_SUB_TOK_MAX;
    const size_t sz_es = sizeof(vf_etab) * VF_NMODELS, sz_fields = ((size_t)VF_BLOCKS_PER_SUB * 8192u + 2u) * sizeof(uint32_t);
    uint8_t *mem = (uint8_t *)VOLCOMP_MALLOC(sz_toks + sz_byp + sz_stage + sz_es + sz_fields);
    if (!mem) return VOLCOMP_ERR_NOMEM;
    vf_enc e;
    memset(e.counts, 0, sizeof e.counts);
    e.toks = (uint16_t *)(void *)mem; e.byp = mem + sz_toks; e.stage = e.byp + sz_byp; e.rstep = nullptr;
    vf_etab *es = (vf_etab *)(void *)(e.stage + sz_stage);
    uint32_t *fields = (uint32_t *)(void *)(e.stage + sz_stage + sz_es);
    volcomp_status st = VOLCOMP_ERR_CORRUPT;
    /* phase A without the analysis: tokenise the given levels, substream by substream */
    uint16_t *tp = e.toks; size_t byp_pos = 0;
    int32_t lvl[VF_BLKV + 1];
    for (uint32_t s = 0; s < VF_NSUB; s++) {
        e.tok_off[s] = (size_t)(tp - e.toks); e.byp_off[s] = byp_pos;
        vf_bitw bw; vf_bw_init(&bw, e.byp + byp_pos, VF_SUB_BYP_MAX);
        int64_t prev_dc = 0;
        for (uint32_t bi = s * VF_BLOCKS_PER_SUB; bi < (s + 1) * VF_BLOCKS_PER_SUB; bi++) {
            const uint32_t nn = n[bi]; if (nn > VF_BLKV - 1) goto out;
            lvl[0] = dc[bi]; memcpy(lvl + 1, lv + off[bi], (size_t)nn * sizeof(int32_t));
            if (!vf_tokenize_block(&e, &tp, &bw, lvl, pos + off[bi], nn, &prev_dc)) goto out;
        }
        size_t bn;
        if (!vf_bw_flush(&bw, &bn)) goto out;
        byp_pos += bn;
    }
    e.tok_off[VF_NSUB] = (size_t)(tp - e.toks); e.byp_off[VF_NSUB] = byp_pos;
    /* phases B and C as vf_encode_lossy */
    {
        vf_model models[VF_NMODELS];
        for (uint32_t m = 0; m < VF_NMODELS; m++) {
            bool any = false;
            for (uint32_t t = 0; t < VF_NTOK; t++) any |= e.counts[m][t] != 0;
            if (!any) e.counts[m][0] = 1;
            if (!vf_model_build(&models[m], e.counts[m])) goto out;
        }
        vf_etabs_init(models, es, VF_NMODELS);
        uint8_t tables[VF_TABLES_MAX_BYTES];
        size_t tn = vf_tables_write(models, tables, VF_NMODELS), p = VF_HDR_BYTES + tn + VF_DIR_BYTES;
        if (dst_cap < p) { st = VOLCOMP_ERR_SHORT_BUF; goto out; }
        dst[0] = 'V'; dst[1] = 'O'; dst[2] = 'L'; dst[3] = 'C'; dst[4] = VF_VERSION; dst[5] = (uint8_t)VLL_MODE_LOSSY;
        vf_wr_u16(dst + 6, qraw);
        memcpy(dst + VF_HDR_BYTES, tables, tn);
        uint8_t *dir = dst + VF_HDR_BYTES + tn;
        for (uint32_t s = 0; s < VF_NSUB; s++) {
            size_t ntok = e.tok_off[s + 1] - e.tok_off[s];
            size_t rn = vf_tans_encode2(es, e.toks + e.tok_off[s], ntok, fields, e.stage, sz_stage);
            if (rn == 0) goto out;
            size_t bn = e.byp_off[s + 1] - e.byp_off[s];
            if (dst_cap - p < rn + bn) { st = VOLCOMP_ERR_SHORT_BUF; goto out; }
            memcpy(dst + p, e.stage, rn); p += rn;
            memcpy(dst + p, e.byp + e.byp_off[s], bn); p += bn;
            vf_wr_u32(dir + s * 8, (uint32_t)rn); vf_wr_u32(dir + s * 8 + 4, (uint32_t)bn);
        }
        *out_n = p; st = VOLCOMP_OK;
    }
out:
    VOLCOMP_FREE(mem);
    return st;
}
