/* train --split z for latent-grid nets (train --lat 4, unet_cfg.half_res 3, --lat-basis): one window's cell trunk on two GPUs (each its z half
   of the CT plus h0 = 2^(L-1) halo cells = 4 h0 voxels) against the same window on one GPU. The input is the full-resolution 16-bit CT
   channel pooled on the device (nn_input_latc) with the window's radial record; each side's record carries the global window depth (h[2])
   and its first plane (h[7]), so the radial channels are those of the global voxel under every cube symmetry. Compared: the owned cells'
   logits and every parameter gradient under a fixed cell-logit gradient (zero on the halo cells), for
   - MX-fp8 activations with round-to-nearest (the same math up to the GroupNorm / gradient summation order across the halves);
     reference: the single-GPU run on a 1e-6-perturbed CT (an MX rounding flipped by a last-bit difference is amplified downstream);
   - the latent recipe's train --fp4 2 preset (all=fp4:fp4:fp4, MX-fp4 activations / gradients / input, stored GN statistics, stochastic
     rounding, chunked, lean 2, shared encoder a1); reference: the
     single-GPU difference between two rounding seeds.
   Symmetries: identity, z flip, z <-> y (the radial vector of a permuted z reads the record's global axis centres).
   With one GPU both halves run on it (UFSM_SPLIT_ONE_GPU=1 forces it). env: UFSM_P (128 voxels), UFSM_ONLY=case. */
#include "split.h"
#include "unet.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int P, Pc, Dl, H0, LC = 29, DEV[2] = {0, 1};   /* P voxels (Pc = P / 4 cells); Dl, H0: this side's cells, the halo cells */
static _Float16 *hct; static uint8_t *hrec; static _Float16 *hg; static float hbas[29 * 64], hgain[29];
static size_t RB;   /* record bytes (global depth P) */

typedef struct { unet *u; void *ct, *rec, *gl; } side_t;
static void side_alloc(side_t *d, const unet_cfg *cfg, int dcells, int side) {
    const size_t nv = (size_t)P * P * 4 * dcells, nc = (size_t)Pc * Pc * dcells;
    d->u = unet_create(cfg); unet_init(d->u, 7); unet_set_lat_basis(d->u, hbas, hgain, LC, 0.f);
    d->ct = nn_malloc(nv * 2); d->rec = nn_malloc(RB); d->gl = nn_malloc((size_t)cfg->cout * nc * 2);
    const int z0c = side < 0 ? 0 : side ? Pc / 2 - H0 : 0, z0v = 4 * z0c;   /* (the first cell / voxel plane of this side in the window) */
    nn_h2d(d->ct, hct + (size_t)z0v * P * P, nv * 2);
    uint8_t *r = malloc(RB); memcpy(r, hrec, RB); int32_t z7 = z0v; memcpy(r + 28, &z7, 4); nn_h2d(d->rec, r, RB); free(r);
    unet_set_input_rad(d->u, d->rec);
    _Float16 *g = malloc((size_t)cfg->cout * nc * 2);
    for (int c = 0; c < cfg->cout; c++) {
        memcpy(g + (size_t)c * nc, hg + ((size_t)c * Pc + z0c) * Pc * Pc, nc * 2);
        if (side >= 0) memset(g + (size_t)c * nc + (size_t)(side ? 0 : dcells - H0) * Pc * Pc, 0, (size_t)H0 * Pc * Pc * 2);   /* (halo cells: no gradient) */
    }
    nn_h2d(d->gl, g, (size_t)cfg->cout * nc * 2); free(g);
}
static void side_free(side_t *d) { unet_free(d->u); nn_free(d->ct); nn_free(d->rec); nn_free(d->gl); }
static void logits_d2h(side_t *d, const float *lg, int cout, int dcells, int z0, int nz, float *out, int oz) {   /* cells [z0, z0 + nz) -> out planes from oz */
    const size_t pl = (size_t)Pc * Pc, nc = pl * dcells; const int h16 = unet_logits_h16(d->u);
    for (int c = 0; c < cout; c++) {
        if (h16) { _Float16 *t = malloc(nz * pl * 2); nn_d2h(t, (const char *)lg + ((size_t)c * nc + z0 * pl) * 2, nz * pl * 2); for (size_t k = 0; k < nz * pl; k++) out[((size_t)c * Pc + oz) * pl + k] = (float)t[k]; free(t); }
        else nn_d2h(out + ((size_t)c * Pc + oz) * pl, lg + (size_t)c * nc + z0 * pl, nz * pl * 4);
    }
}
/* the streamed head (unet_head_fwd_z) on cell plane z: fp16 [cout][plane] to the host (0: not covered) */
static int head_plane(side_t *d, int cout, int z, _Float16 *out) {
    const size_t pl = (size_t)Pc * Pc; void *y = nn_malloc((size_t)cout * pl * 2);
    const int r = unet_head_fwd_z(d->u, z, z + 1, y); nn_sync();
    if (!r) nn_d2h(out, y, (size_t)cout * pl * 2);
    nn_free(y); return !r;
}
/* forward (the owned cells' logits to lo right away: lean layouts reuse the logit buffer in the backward), the halo cell next to the cut
   through the streamed head after unet_split_head_halo (side >= 0; the single run: both cells next to the cut), then the backward */
static float *g_lo; static _Float16 *g_hh[2]; static int g_hok[2];
static void step(side_t *d, int cin, int cout, int dcells, int side) {
    const shape5 xs = {1, cin, dcells, Pc, Pc};
    const float *lg = unet_forward_x(d->u, d->ct, xs, 1, 1); nn_sync();
    const size_t pl = (size_t)Pc * Pc;
    if (side < 0) { logits_d2h(d, lg, cout, dcells, 0, Pc, g_lo, 0); for (int k = 0; k < 2; k++) g_hok[k] = head_plane(d, cout, Pc / 2 - 1 + k, g_hh[k]); }
    else {
        logits_d2h(d, lg, cout, dcells, side ? H0 : 0, Pc / 2, g_lo, side ? Pc / 2 : 0);
        unet_split_head_halo(d->u);   /* (a collective) */
        g_hok[side] = head_plane(d, cout, side ? H0 - 1 : dcells - H0, g_hh[side]);   /* side 0's halo cell Pc / 2, side 1's Pc / 2 - 1 */
    }
    (void)pl;
    unet_zero_grad(d->u); unet_backward_x(d->u, d->gl, 1);
}
static void single(const unet_cfg *cfg, unsigned sr, float *g, float *lo, _Float16 *hh0, _Float16 *hh1, int *hok) {
    nn_init(DEV[0]); nn_set_sr_step(sr);
    side_t d; side_alloc(&d, cfg, Pc, -1);
    g_lo = lo; g_hh[0] = hh0; g_hh[1] = hh1; step(&d, cfg->cin, cfg->cout, Pc, -1); *hok = g_hok[0] && g_hok[1]; nn_sync();
    unet_grad_d2h(d.u, g); side_free(&d);
}
static side_t S2[2]; static const unet_cfg *g_cfg;
static void job(int side, void *a) { (void)a; step(&S2[side], g_cfg->cin, g_cfg->cout, Dl, side); }
/* hh0 / hh1: the head logits of cells Pc / 2 - 1 (side 1's halo) and Pc / 2 (side 0's halo) */
static void split2(split_ctx *ctx, const unet_cfg *cfg, unsigned sr, float *g, float *lo, _Float16 *hh0, _Float16 *hh1, int *hok, size_t np) {
    g_cfg = cfg; g_lo = lo; g_hh[1] = hh0; g_hh[0] = hh1;
    for (int s = 0; s < 2; s++) { nn_init(DEV[s]); nn_set_sr_step(sr); side_alloc(&S2[s], cfg, Dl, s); unet_set_split(S2[s].u, s, H0, split_halo);
                                  if (!getenv("UFSM_SPLIT_SYNC")) unet_set_split_async(S2[s].u, split_halo_begin, split_halo_end); }
    split_run(ctx, job, nullptr);
    *hok = g_hok[0] && g_hok[1];
    float *g1 = malloc(np * 4);
    for (int s = 0; s < 2; s++) { nn_init(DEV[s]); nn_sync(); unet_grad_d2h(S2[s].u, s ? g1 : g); }
    for (size_t i = 0; i < np; i++) g[i] += g1[i];
    for (int s = 0; s < 2; s++) { nn_init(DEV[s]); side_free(&S2[s]); }
    free(g1);
}
static double rel(const float *a, const float *b, size_t n) {
    double d2 = 0, r2 = 0; for (size_t i = 0; i < n; i++) { const double d = (double)a[i] - b[i]; d2 += d * d; r2 += (double)b[i] * b[i]; }
    return sqrt(d2 / (r2 + 1e-300));
}
static uint64_t rs = 0x9e3779b97f4a7c15ull;
static double ru(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (double)(rs >> 11) * (1.0 / 9007199254740992.0); }

int main(void) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (nn_init(0)) { printf("test_split_lat: no GPU, skipped\n"); return 0; }
    if (ufsm_env_on("UFSM_SPLIT_ONE_GPU") || nn_init(1)) { DEV[1] = 0; nn_init(0); nn_check(); }
    P = getenv("UFSM_P") ? atoi(getenv("UFSM_P")) : 128;
    unet_cfg cfg = {4, {32, 32, 64, 64}, LC + 3, 16, 8, 1}; cfg.half_res = 3;
    H0 = 1 << (cfg.nlev - 1); Pc = P / 4; Dl = Pc / 2 + H0;
    if (P % (8 * H0)) { fprintf(stderr, "P must be a multiple of %d voxels\n", 8 * H0); return 2; }
    const size_t p3 = (size_t)P * P * P, c3 = (size_t)Pc * Pc * Pc;
    RB = NN_RAD_REC(P);
    hct = malloc(p3 * 2); hrec = calloc(1, RB); hg = malloc((size_t)cfg.cout * c3 * 2);
    {   /* a smooth field with sheet-like structure along z and noise, z-scored-ish */
        double k[3][3], ph[3]; for (int j = 0; j < 3; j++) { for (int a = 0; a < 3; a++) k[j][a] = ru() * 0.25; ph[j] = ru() * 6.28; }
        for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) for (int x = 0; x < P; x++) {
            double v = 0; for (int j = 0; j < 3; j++) v += sin(k[j][0] * z + k[j][1] * y + k[j][2] * x + ph[j]);
            hct[((size_t)z * P + y) * P + x] = (_Float16)(0.6 * v + 0.4 * (2 * ru() - 1));
        }
    }
    for (int c = 0; c < LC; c++) { hgain[c] = 1.f; double n = 0; for (int k = 0; k < 64; k++) { hbas[c * 64 + k] = c ? (float)(2 * ru() - 1) : 0.125f; n += (double)hbas[c * 64 + k] * hbas[c * 64 + k]; }
                                   for (int k = 0; k < 64; k++) hbas[c * 64 + k] = (float)(hbas[c * 64 + k] / sqrt(n)); }
    for (int c = 0; c < cfg.cout; c++) {   /* a smooth cell-logit gradient (x grad scale 1024: ~1e-3 per cell) with 20 % noise: MX blocks of coherent values, as a loss gives */
        double k[3], ph = ru() * 6.28; for (int a = 0; a < 3; a++) k[a] = ru() * 0.5;
        for (int z = 0; z < Pc; z++) for (int y = 0; y < Pc; y++) for (int x = 0; x < Pc; x++)
            hg[(((size_t)c * Pc + z) * Pc + y) * Pc + x] = (_Float16)(sin(k[0] * z + k[1] * y + k[2] * x + ph) + 0.2 * (2 * ru() - 1));
    }
    nn_set_f16(1); nn_set_grad_scale(1024.f); nn_set_loss_grad_h16(1);
    setenv("UFSM_F4_WGRAD", "1", 0);
    split_ctx *ctx = split_create(DEV[0], DEV[1]);
    size_t np; { unet *pr = unet_create(&cfg); np = unet_nparams(pr); unet_free(pr); }
    float *g0 = malloc(np * 4), *g1 = malloc(np * 4), *g2 = malloc(np * 4);
    const size_t nl = (size_t)cfg.cout * c3; float *l0 = malloc(nl * 4), *l1 = malloc(nl * 4), *l2 = malloc(nl * 4);
    const size_t hn = (size_t)cfg.cout * Pc * Pc; _Float16 *h0a = malloc(hn * 2), *h0b = malloc(hn * 2), *h1a = malloc(hn * 2), *h1b = malloc(hn * 2), *h2a = malloc(hn * 2), *h2b = malloc(hn * 2);
    printf("test_split_lat: P %d voxels (%d cells), level-0 halo %d cells, local depth %d cells%s, %zu params\n", P, Pc, H0, Dl, DEV[0] == DEV[1] ? ", both halves on one GPU" : "", np);
    int fails = 0, ci = 0;
    static const int syms[3] = {0 | 1 << 2 | 2 << 4, 0 | 1 << 2 | 2 << 4 | 1 << 8, 1 | 0 << 2 | 2 << 4};   /* identity, z flip, z <-> y */
    static const char *const sname[3] = {"identity", "z flip", "z<->y"};
    for (int mode = 0; mode < 3; mode++) for (int sy = 0; sy < 3; sy++, ci++) {   /* 0 MX-fp8 round to nearest, 1 the fp4 preset round to nearest, 2 the fp4 preset (stochastic) */
        if (getenv("UFSM_ONLY") && ci != atoi(getenv("UFSM_ONLY"))) continue;
        {   /* the window's record: corner (o1, o2), depth P, the axis centre per source z drifting across the window */
            int32_t h[8] = {P / 3, -P / 5, P, P, 1, syms[sy], 0, 0}; memcpy(hrec, h, sizeof h);
            float *cyz = (float *)(hrec + 32), *cxz = cyz + P;
            for (int z = 0; z < P; z++) { cyz[z] = (float)(P * 0.4 + 0.3 * z); cxz[z] = (float)(-P * 0.1 - 0.2 * z); }
        }
        if (mode == 0) { unet_set_act_mx4(0); unet_set_act_mx8(1); nn_set_sr(0); nn_set_prec_policy(""); unet_set_chunk_up(1); unet_set_grad_mx8(0); unet_set_lean(0);
                         unet_set_grad_mx4(0); unet_set_share_enc_a1(0); unet_set_recompute(1); }
        else { unet_set_act_mx4(1); nn_set_sr(mode == 2); if (nn_set_prec_policy("all=fp4:fp4:fp4")) return 2; unet_set_chunk_up(2); unet_set_grad_mx8(1); unet_set_lean(2);
               unet_set_grad_mx4(mode == 2); unet_set_share_enc_a1(1); unet_set_recompute(1); }   /* (MX-fp4 gradients need SR: the rn mode keeps MX-fp8 ones) */
        unet_set_input_mx(1); unet_set_input_prec(mode ? 4 : 0); nn_set_gn_stored(mode);   /* (the lat recipe's precision.txt: input_prec 4, gn_stored 1) */
        int ok0, ok1, ok2; single(&cfg, 1, g0, l0, h0a, h0b, &ok0); split2(ctx, &cfg, 1, g1, l1, h1a, h1b, &ok1, np);
        const double dl = rel(l1, l0, nl), dg = rel(g1, g0, np);
        double dh = -1;   /* the halo cells next to the cut after the exchange, against the single run's (the streamed head, fp16) */
        if (ok0 && ok1) { size_t nd = 0; for (size_t k = 0; k < hn; k++) nd += memcmp(&h0a[k], &h1a[k], 2) != 0 || memcmp(&h0b[k], &h1b[k], 2) != 0; dh = (double)nd / (2.0 * hn); }
        double rl, rg; const char *what;
        if (mode < 2) {   /* reference: a last-bit-perturbed CT on one GPU */
            _Float16 *keep = malloc(p3 * 2); memcpy(keep, hct, p3 * 2);
            for (size_t i = 0; i < p3; i++) { const float v = (float)hct[i]; if (i % 7 == 0) hct[i] = (_Float16)(v * (1.f + 1e-3f)); }
            single(&cfg, 1, g2, l2, h2a, h2b, &ok2); memcpy(hct, keep, p3 * 2); free(keep); what = "perturbed";
        } else { single(&cfg, 2, g2, l2, h2a, h2b, &ok2); what = "seeds"; }
        rl = rel(l2, l0, nl); rg = rel(g2, g0, np);
        const int ok = mode < 2 ? dl < 2 * rl + 1e-4 && dg < 2 * rg + 1e-3 : dl < 1.5 * rl + 1e-3 && dg < 1.5 * rg + 1e-3;
        const int okh = dh == 0.0 || (dh < 0 && mode);   /* (bit-identical; not covered only where the streamed head is not: fp4 modes may lack it) */
        printf("  %d %s %-8s: owned logits rel %.2e (%s %.2e), cut halo cells %s, grad rel %.2e (%s %.2e)  %s\n", ci, mode == 2 ? "fp4 sr lean 2" : mode ? "fp4 rn lean 2" : "MX-fp8 rn    ",
               sname[sy], dl, what, rl, dh < 0 ? "not covered" : dh == 0 ? "identical" : "DIFFER", dg, what, rg, ok && okh ? "ok" : "FAIL");
        fails += !okh;
        fails += !ok;
    }
    split_free(ctx);
    printf("test_split_lat: %s\n", fails ? "FAIL" : "ok");
    return fails != 0;
}
