/* predict --surf (DESIGN 2026-11-20 "inksurf"): the surface ink head on unrolled wraps (tifxyz). Each wrap's map grid (one pixel per window
   voxel along the surface, the training patch's spacing) is cut into Hp x Hp patches with a halo; a patch runs in the first prediction
   window whose interior holds its points (+- the K layers); per window: the trunk's level-0 features (unet_head_ext_fwd with identity
   rows), each assigned patch resampled, the 2-D head, the core's probabilities into the wrap's map (uint8 1..255, 0 = no surface / no
   window). DIR/<wrap>.tif, the ink-wrapscore --maps format. */
#include "surfpred.h"
#include "inksurf.h"
#include "imgops.h"
#include "tiffw.h"
#include "nn.h"
#include "json.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double tifxyz_meta_scale(const char *p) {   /* meta.json scale[0] (grid vertices per native voxel); <= 0 on failure */
    FILE *f = fopen(p, "rb"); if (!f) return -1; fseek(f, 0, SEEK_END); const long l = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = malloc((size_t)l + 1); const int ok = b && fread(b, 1, (size_t)l, f) == (size_t)l; fclose(f); if (!ok) { free(b); return -1; }
    json *j = json_parse(b, (size_t)l); free(b); if (!j) return -1; const double s = json_num(json_at(json_get(j, "scale"), 0), -1); json_free(j); return s;
}
typedef struct { char dir[1024], name[256]; int w, h, mw, mh, na, nb; double sc; float *X, *Y, *Z; uint8_t *map; long *pid; float **blk; int bw, bh; } spw;   /* blk: --surf-overlap's
   feathered sums, per SURF_BLK^2 block of the map (lazily: covered blocks only), [2][SURF_BLK^2] = sum w p, sum w */
typedef struct { int wr, a0, b0, pa, pb; int64_t lo[3], hi[3]; long tile; } spp;
enum { SURF_ZS = 32, SURF_BLK = 128 };   /* the feature slab's planes (training's head slabs) */
struct surfp {
    spw *W; int nw; spp *P; long np; int Hp, halo, core, K, C0; double s; isf_dev dev; isf_win win; float *idw; void *feat; size_t featc; int whole; double pv; int stride, ramp, blend;   /* stride: the patch grid's step (core: hard tiling; < core: --surf-overlap, feathered over ramp px, blend 0 linear 1 cosine); whole: the whole-window feature fallback (DEFAULTED) */
    long *tp, *toff; long ntile, nrun, nfit; uint8_t *tdone;
};
static int grid_pt(const spw *w, double gu, double gv, double P[3]) {   /* the wrap's grid at (col gu, row gv), bilinear; 0: an invalid corner */
    int c0 = (int)floor(gu), r0 = (int)floor(gv); if (c0 < 0 || r0 < 0 || c0 >= w->w - 1 || r0 >= w->h - 1) return 0;
    const double fu = gu - c0, fv = gv - r0; const size_t i00 = (size_t)r0 * w->w + c0, i01 = i00 + 1, i10 = i00 + w->w, i11 = i10 + 1;
    if (!(w->Z[i00] >= 0 && w->Z[i01] >= 0 && w->Z[i10] >= 0 && w->Z[i11] >= 0)) return 0;
    const float *A[3] = {w->Z, w->Y, w->X};
    for (int d = 0; d < 3; d++) { const double a = A[d][i00], b = A[d][i01], c = A[d][i10], e = A[d][i11]; P[d] = (1 - fv) * ((1 - fu) * a + fu * b) + fv * ((1 - fu) * c + fu * e); }
    return 1;
}
/* map pixel (row a, col b) -> the level point (z, y, x) and unit normal; 0: off the surface */
static int map_pt(const surfp *S, const spw *w, int a, int b, float q[6]) {
    const double g = S->s * S->pv * w->sc;   /* grid px per map px (pv window voxels per map px) */
    double P[3], A[3], B[3], C[3], D[3];
    if (!grid_pt(w, b * g, a * g, P) || !grid_pt(w, (b + 1) * g, a * g, A) || !grid_pt(w, (b - 1) * g, a * g, B) || !grid_pt(w, b * g, (a + 1) * g, C) || !grid_pt(w, b * g, (a - 1) * g, D)) return 0;
    const double u[3] = {A[0] - B[0], A[1] - B[1], A[2] - B[2]}, v[3] = {C[0] - D[0], C[1] - D[1], C[2] - D[2]};
    double n[3] = {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]}; const double l = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    if (!(l > 1e-9)) return 0;
    for (int d = 0; d < 3; d++) { q[d] = (float)((P[d] - 0.5 * (S->s - 1)) / S->s); q[3 + d] = (float)(n[d] / l); }
    return 1;
}
surfp *surfp_open(const char *wraps, const char *isf, int C0, int Hp, int halo, int K, double s, double pv, double ov, int blend) {
    if (Hp < 32 || Hp % 16 || halo < 0 || 2 * halo >= Hp) { fprintf(stderr, "predict --surf: --surf-P a multiple of 16 >= 32, --surf-halo < P / 2\n"); return nullptr; }
    if (!(ov >= 0 && ov <= 0.75)) { fprintf(stderr, "predict --surf: --surf-overlap in [0, 0.75]\n"); return nullptr; }
    surfp *S = calloc(1, sizeof *S); S->Hp = Hp; S->halo = halo; S->core = Hp - 2 * halo; S->K = K; S->C0 = C0; S->s = s; S->pv = pv > 0 ? pv : 1;
    /* --surf-overlap F (2026-11-21, lead: S2-avg's 1447 maps showed hard patch seams, column jumps 8.7x the median; each patch's trunk
       features come from its own window's GroupNorm statistics): patches every core (1 - F) px, each core pixel's probability weighted by
       a feather over the ramp (the overlap width) to the core's edge, the map = sum w p / sum w. F 0: hard tiling, the old bytes */
    S->ramp = (int)lround(S->core * ov); S->stride = S->core - S->ramp; if (S->stride < 1) S->stride = 1; S->blend = blend;
    char *cp = strdup(wraps); int cap = 0;
    for (char *t = strtok(cp, ","); t; t = strtok(nullptr, ",")) {
        if (S->nw == cap) { cap = cap ? 2 * cap : 16; S->W = realloc(S->W, (size_t)cap * sizeof *S->W); }
        spw *w = &S->W[S->nw]; memset(w, 0, sizeof *w); snprintf(w->dir, sizeof w->dir, "%s", t);
        size_t L = strlen(w->dir); while (L > 1 && w->dir[L - 1] == '/') w->dir[--L] = 0;
        const char *bn = strrchr(w->dir, '/'); snprintf(w->name, sizeof w->name, "%s", bn ? bn + 1 : w->dir);
        char p[1300]; int w2, h2, w3, h3;
        snprintf(p, sizeof p, "%s/x.tif", w->dir); w->X = img_read_f32(p, &w->w, &w->h);
        snprintf(p, sizeof p, "%s/y.tif", w->dir); w->Y = img_read_f32(p, &w2, &h2);
        snprintf(p, sizeof p, "%s/z.tif", w->dir); w->Z = img_read_f32(p, &w3, &h3);
        snprintf(p, sizeof p, "%s/meta.json", w->dir); w->sc = tifxyz_meta_scale(p);
        if (!w->X || !w->Y || !w->Z || w2 != w->w || w3 != w->w || h2 != w->h || h3 != w->h || !(w->sc > 0)) { fprintf(stderr, "predict --surf: cannot read the tifxyz %s\n", w->dir); free(cp); surfp_free(S); return nullptr; }
        w->mw = (int)ceil(w->w / w->sc / (s * S->pv)); w->mh = (int)ceil(w->h / w->sc / (s * S->pv)); w->map = calloc((size_t)w->mw * w->mh, 1);
        S->nw++;
    }
    free(cp);
    int dep = 4; if (isf_peek(isf, nullptr, &dep, nullptr)) dep = 4;   /* (the head's depth from its file: S3 heads are depth 5) */
    if (Hp % (1 << dep)) { fprintf(stderr, "predict --surf: --surf-P %d must be a multiple of 2^%d (the head's depth)\n", Hp, dep); surfp_free(S); return nullptr; }
    if (isf_dev_init_d(&S->dev, C0, Hp, K, dep, 1)) { fprintf(stderr, "predict --surf: out of GPU memory for the head\n"); surfp_free(S); return nullptr; }
    int st = 0; if (isf_load(&S->dev, isf, &st)) { fprintf(stderr, "predict --surf: cannot load the surface head %s (its trunk channels x layers must be %d x %d)\n", isf, C0, K); surfp_free(S); return nullptr; }
    if (isf_win_alloc(&S->win, Hp, K, 1.f)) { surfp_free(S); return nullptr; }
    float *h = calloc((size_t)C0 * (C0 + 1), 4); for (int c = 0; c < C0; c++) h[(size_t)c * C0 + c] = 1.f;
    S->idw = nn_malloc((size_t)C0 * (C0 + 1) * 4); nn_h2d(S->idw, h, (size_t)C0 * (C0 + 1) * 4); free(h);
    /* the patches: core tiles of every wrap's map, each with its level bbox (+ the layers' reach + 2) */
    long pcap = 0; const float ext = 0.5f * (K - 1) + 2;
    for (int k = 0; k < S->nw; k++) {
        spw *w = &S->W[k]; const int st = S->stride;
        w->na = w->mh > S->core ? (w->mh - S->core + st - 1) / st + 1 : 1; w->nb = w->mw > S->core ? (w->mw - S->core + st - 1) / st + 1 : 1;   /* (st = core: ceil(m / core) as before) */
        if (st == S->core) { w->na = (w->mh + S->core - 1) / S->core; w->nb = (w->mw + S->core - 1) / S->core; }
        w->pid = malloc((size_t)w->na * w->nb * sizeof *w->pid);
        if (S->ramp) { w->bw = (w->mw + SURF_BLK - 1) / SURF_BLK; w->bh = (w->mh + SURF_BLK - 1) / SURF_BLK; w->blk = calloc((size_t)w->bw * w->bh, sizeof *w->blk); }
        for (size_t q = 0; q < (size_t)w->na * w->nb; q++) w->pid[q] = -1;
        for (int pa = 0; pa < w->na; pa++) for (int pb = 0; pb < w->nb; pb++) {
            const int a0 = pa * S->stride - halo, b0 = pb * S->stride - halo;
            spp p = {k, a0, b0, pa, pb, {INT64_MAX, INT64_MAX, INT64_MAX}, {INT64_MIN, INT64_MIN, INT64_MIN}, -1}; int any = 0;
            for (int i = 0; i < Hp; i++) for (int j = 0; j < Hp; j++) { const int a = a0 + i, b = b0 + j; if (a < 0 || b < 0 || a >= w->mh || b >= w->mw) continue;
                float q[6]; if (!map_pt(S, w, a, b, q)) continue;
                const int core = i >= halo && i < Hp - halo && j >= halo && j < Hp - halo; any |= core;
                for (int d = 0; d < 3; d++) { const int64_t lo = (int64_t)floor(q[d] - ext), hi = (int64_t)ceil(q[d] + ext); if (lo < p.lo[d]) p.lo[d] = lo; if (hi > p.hi[d]) p.hi[d] = hi; } }
            if (!any) continue;
            if (S->np == pcap) { pcap = pcap ? 2 * pcap : 1024; S->P = realloc(S->P, (size_t)pcap * sizeof *S->P); }
            w->pid[(size_t)pa * w->nb + pb] = S->np; S->P[S->np++] = p;
        }
    }
    fprintf(stderr, "predict --surf: %d wraps, %ld patches of %d px (core %d, map px = %.0f level-0 voxels), head from %s (step %d)\n", S->nw, S->np, Hp, S->core, s, isf, st);
    if (S->ramp) fprintf(stderr, "predict --surf: --surf-overlap %.2f: patches every %d px, %s feather over %d px\n", ov, S->stride, blend ? "cosine" : "linear", S->ramp);
    return S;
}
void surfp_free(surfp *S) {
    if (!S) return;
    for (int k = 0; k < S->nw; k++) { free(S->W[k].X); free(S->W[k].Y); free(S->W[k].Z); free(S->W[k].map); free(S->W[k].pid);
        if (S->W[k].blk) { for (size_t b = 0; b < (size_t)S->W[k].bw * S->W[k].bh; b++) free(S->W[k].blk[b]); free(S->W[k].blk); } }
    free(S->W); free(S->P); free(S->tp); free(S->toff); free(S->tdone); isf_dev_free(&S->dev); isf_win_free(&S->win); nn_free(S->idw); nn_free(S->feat); free(S);
}
static int fits(const spp *q, const int64_t o[3], const int64_t n[3], const int m[3]) {
    for (int d = 0; d < 3; d++) if (q->lo[d] < o[d] + m[d] || q->hi[d] >= o[d] + n[d] - m[d]) return 0;
    return 1;
}
/* the windows (2026-11-21, lead: 1667's 18375 patch-centred windows were ~30x the voxel scorer's 601 grid windows): greedy set cover on the
   half-step lattice of the box's grid (origins g[0] + k * h, h = the interior / 2 per axis; the even k are the voxel scorer's windows): every
   patch fits a few lattice windows' interiors (any wrap); repeatedly the window that takes the most unplaced patches (ties: the lower
   index). The rest (wider than h along an axis): a window centred on each still unplaced patch, taking every unplaced patch that fits.
   *o: malloc'd origins [3 per window]; returns the window count */
long surfp_plan(surfp *S, const int64_t n[3], const int m[3], const int64_t *g, long ng, int64_t **o) {
    long cap = 256, nt = 0, fit = 0, nlat = 0; *o = malloc((size_t)cap * 3 * sizeof **o);
    int64_t h[3], g0[3], k0[3], K[3]; for (int d = 0; d < 3; d++) { h[d] = (n[d] - 2 * m[d]) / 2; if (h[d] < 1) h[d] = 1; g0[d] = ng ? g[d] : -m[d]; }
    /* per axis, the lattice k with g0 + k h + m <= lo and hi < g0 + k h + n - m */
    #define KLO(q, d) ((int64_t)ceil((double)((q)->hi[d] - n[d] + m[d] + 1 - g0[d]) / (double)h[d]))
    #define KHI(q, d) ((int64_t)floor((double)((q)->lo[d] - m[d] - g0[d]) / (double)h[d]))
    for (int d = 0; d < 3; d++) { int64_t a = INT64_MAX, b = INT64_MIN; for (long p = 0; p < S->np; p++) { const int64_t x = KLO(&S->P[p], d), y = KHI(&S->P[p], d); if (x <= y) { if (x < a) a = x; if (y > b) b = y; } }
                                  k0[d] = a <= b ? a : 0; K[d] = a <= b ? b - a + 1 : 0; }
    const size_t nc = (size_t)K[0] * K[1] * K[2];
    long *cnt = calloc(nc ? nc : 1, sizeof *cnt), *poff = calloc((size_t)S->np + 1, sizeof *poff); long *pc = nullptr, *coff = nullptr, *cp = nullptr;
    for (long p = 0; p < S->np; p++) { const spp *q = &S->P[p]; long c = 1; for (int d = 0; d < 3; d++) { const int64_t x = KLO(q, d), y = KHI(q, d); c *= x <= y ? y - x + 1 : 0; } poff[p + 1] = poff[p] + c; }
    pc = malloc((size_t)(poff[S->np] ? poff[S->np] : 1) * sizeof *pc); coff = calloc(nc + 1, sizeof *coff);
    for (long p = 0; p < S->np; p++) { const spp *q = &S->P[p]; long e = poff[p]; if (e == poff[p + 1]) continue;
        for (int64_t a = KLO(q, 0); a <= KHI(q, 0); a++) for (int64_t b = KLO(q, 1); b <= KHI(q, 1); b++) for (int64_t c = KLO(q, 2); c <= KHI(q, 2); c++) {
            const long id = (long)(((a - k0[0]) * K[1] + (b - k0[1])) * K[2] + (c - k0[2])); pc[e++] = id; coff[id + 1]++; cnt[id]++; } }
    for (size_t c = 0; c < nc; c++) coff[c + 1] += coff[c];
    cp = malloc((size_t)(coff[nc] ? coff[nc] : 1) * sizeof *cp); { long *w = malloc((nc + 1) * sizeof *w); memcpy(w, coff, (nc + 1) * sizeof *w);
        for (long p = 0; p < S->np; p++) for (long e = poff[p]; e < poff[p + 1]; e++) cp[w[pc[e]]++] = p; free(w); }
    for (;;) {
        long best = -1, bc = 0; for (size_t c = 0; c < nc; c++) if (cnt[c] > bc) { bc = cnt[c]; best = (long)c; }
        if (best < 0) break;
        const long kx = best % K[2], ky = (best / K[2]) % K[1], kz = best / (K[1] * K[2]);
        const int64_t c3[3] = {g0[0] + (k0[0] + kz) * h[0], g0[1] + (k0[1] + ky) * h[1], g0[2] + (k0[2] + kx) * h[2]};
        if (nt == cap) { cap *= 2; *o = realloc(*o, (size_t)cap * 3 * sizeof **o); }
        memcpy(*o + 3 * nt, c3, sizeof c3);
        for (long e = coff[best]; e < coff[best + 1]; e++) { const long p = cp[e]; if (S->P[p].tile >= 0) continue; S->P[p].tile = nt; fit++;
            for (long f = poff[p]; f < poff[p + 1]; f++) cnt[pc[f]]--; }
        nt++; nlat++;
    }
    #undef KLO
    #undef KHI
    free(cnt); free(poff); free(pc); free(coff); free(cp);
    for (long p = 0; p < S->np; p++) {
        spp *q = &S->P[p]; if (q->tile >= 0) continue;
        int64_t c[3]; for (int d = 0; d < 3; d++) c[d] = (q->lo[d] + q->hi[d]) / 2 - n[d] / 2;
        if (!fits(q, c, n, m)) continue;
        if (nt == cap) { cap *= 2; *o = realloc(*o, (size_t)cap * 3 * sizeof **o); }
        memcpy(*o + 3 * nt, c, sizeof c);
        for (long r = p; r < S->np; r++) if (S->P[r].tile < 0 && fits(&S->P[r], c, n, m)) { S->P[r].tile = nt; fit++; }
        nt++;
    }
    S->nfit = fit; S->ntile = nt;
    free(S->toff); free(S->tp); S->toff = calloc((size_t)nt + 1, sizeof *S->toff); S->tp = malloc((size_t)(fit ? fit : 1) * sizeof *S->tp);
    for (long p = 0; p < S->np; p++) if (S->P[p].tile >= 0) S->toff[S->P[p].tile + 1]++;
    for (long t = 0; t < nt; t++) S->toff[t + 1] += S->toff[t];
    long *pos = malloc((size_t)(nt + 1) * sizeof *pos); memcpy(pos, S->toff, (size_t)(nt + 1) * sizeof *pos);
    for (long p = 0; p < S->np; p++) if (S->P[p].tile >= 0) S->tp[pos[S->P[p].tile]++] = p;
    free(pos);
    if (fit < S->np) fprintf(stderr, "DEFAULTED: predict --surf: %ld of %ld patches fit no window's interior (a patch curved past the window's interior): their pixels stay 0 (a smaller --surf-P or a larger --window)\n", S->np - fit, S->np);
    fprintf(stderr, "predict --surf: %ld patches in %ld windows (%ld on the grid's half-step lattice, %ld centred on patches; the box's grid has %ld)\n", fit, nt, nlat, nt - nlat, ng);
    return nt;
}
int surfp_window(surfp *S, unet *u, long t, const int64_t o[3], int D, int H, int W) {
    if (t < 0 || t >= S->ntile || S->toff[t] == S->toff[t + 1]) return 0;
    const size_t V = (size_t)D * H * W, hw = (size_t)H * W;
    /* (2026-11-21: the trunk features in z-slabs of SURF_ZS planes, the training path's unet_head_feat_z + isf_dev_take: 32 ch x 544^3 fp16
       whole was 10.3 GB, a 32-plane slab 0.6 GB. The whole-window copy only as a DEFAULTED fallback when the slab path is not covered.) */
    if (!S->whole && getenv("UFSM_SURF_WHOLE") && atoi(getenv("UFSM_SURF_WHOLE"))) { S->whole = 1; fprintf(stderr, "DEFAULTED: predict --surf: UFSM_SURF_WHOLE=1: the whole window's features at once (%.2f GB)\n", S->C0 * V * 2e-9); }   /* (the slab path's reference, tests) */
    int whole = S->whole;
    if (whole && S->featc < (size_t)S->C0 * V * 2) { nn_free(S->feat); S->featc = (size_t)S->C0 * V * 2; S->feat = nn_malloc(S->featc); if (!S->feat) { S->featc = 0; fprintf(stderr, "predict --surf: out of GPU memory for the window's features\n"); return -1; } }
    if (whole && unet_head_ext_fwd(u, S->idw, S->idw + (size_t)S->C0 * S->C0, S->C0, S->feat)) { fprintf(stderr, "predict --surf: the trunk's features are not covered (unet_head_ext_fwd)\n"); return -1; }
    const int Hp = S->Hp; const size_t n = (size_t)Hp * Hp;
    for (long k = S->toff[t]; k < S->toff[t + 1]; k++) {
        const spp *p = &S->P[S->tp[k]]; spw *w = &S->W[p->wr]; isf_win *iw = &S->win; iw->np = 0;
        for (int i = 0; i < Hp; i++) for (int j = 0; j < Hp; j++) {
            const size_t q = (size_t)i * Hp + j; float *pn = iw->pn + 6 * q; float v[6];
            iw->lab[q] = 0; iw->wt[q] = 0; pn[0] = pn[1] = pn[2] = -1e6f; pn[3] = 1; pn[4] = pn[5] = 0;
            const int a = p->a0 + i, b = p->b0 + j; if (a < 0 || b < 0 || a >= w->mh || b >= w->mw || !map_pt(S, w, a, b, v)) continue;
            for (int d = 0; d < 3; d++) { pn[d] = v[d] - (float)o[d]; pn[3 + d] = v[3 + d]; } iw->wt[q] = 1; iw->np++;
        }
        if (isf_win_lists(iw, D, H, W) || isf_dev_upload(&S->dev, iw)) { fprintf(stderr, "predict --surf: the patch's lists failed (memory)\n"); return -1; }
        if (!S->dev.nv) continue;
        if (!whole) for (int z0 = 0; z0 < D; z0 += SURF_ZS)
            if (isf_dev_take(&S->dev, u, iw, z0, z0 + SURF_ZS < D ? z0 + SURF_ZS : D, hw)) {
                fprintf(stderr, "DEFAULTED: predict --surf: the slab features are not covered (unet_head_feat_z): the whole window's features at once (%.2f GB)\n", S->C0 * V * 2e-9);
                S->whole = 1; return surfp_window(S, u, t, o, D, H, W); }
        if (whole) nn_isf_take(S->feat, S->C0, V, 0, S->dev.vox, 0, S->dev.nv, S->dev.fc, S->dev.nv);
        isf_dev_step(&S->dev, iw, 1.f, 0, 1.f, nullptr, nullptr);
        for (int i = S->halo; i < Hp - S->halo; i++) for (int j = S->halo; j < Hp - S->halo; j++) {
            const size_t q = (size_t)i * Hp + j; const int a = p->a0 + i, b = p->b0 + j; if (!(iw->wt[q] > 0) || a < 0 || b < 0 || a >= w->mh || b >= w->mw) continue;
            const double pr = 1 / (1 + exp(-(double)S->dev.zh[q]));
            if (!S->ramp) { w->map[(size_t)a * w->mw + b] = (uint8_t)(1 + lrint(pr * 254)); continue; }
            const int ci = i - S->halo, cj = j - S->halo; double f[2];   /* the feather: 1 inside, falling over the ramp to the core's edges */
            for (int e = 0; e < 2; e++) { const int c = e ? cj : ci; double t = fmin(c + 0.5, S->core - c - 0.5) / S->ramp; t = t > 1 ? 1 : t;
                f[e] = S->blend ? 0.5 - 0.5 * cos(3.14159265358979323846 * t) : t; }
            const double wt = f[0] * f[1] + 1e-6;   /* (> 0: a pixel only one patch reached keeps its value) */
            const size_t bi = (size_t)(a / SURF_BLK) * w->bw + b / SURF_BLK;
            if (!w->blk[bi] && !(w->blk[bi] = calloc(2 * SURF_BLK * SURF_BLK, sizeof(float)))) { fprintf(stderr, "predict --surf: out of memory for the blend sums\n"); return -1; }
            float *B = w->blk[bi]; const size_t o = (size_t)(a % SURF_BLK) * SURF_BLK + b % SURF_BLK;
            B[o] += (float)(wt * pr); B[SURF_BLK * SURF_BLK + o] += (float)wt; }
        S->nrun++;
    }
    (void)n;
    return 0;
}
int surfp_write(surfp *S, const char *dir) {
    char p[1600]; int rc = 0;
    /* (2026-11-21) the maps at the mesh's canvas resolution (1 px = 1 level-0 voxel along the surface: what render / ink-score /
       ink-wrapscore read), not the window-voxel grid (s level-0 voxels per px): at level 2 the 4x coarser map was read as canvas px, i.e.
       scaled 4x about the corner (0841 AUC 0.03, P4 0.30). Canvas px x samples map coordinate x / s, bilinear over the run pixels (0 = no
       patch: left out of the weights). UFSM_SURF_NATIVE=1: the window-voxel grid as before. */
    const int native = getenv("UFSM_SURF_NATIVE") && atoi(getenv("UFSM_SURF_NATIVE"));
    for (int k = 0; S->ramp && k < S->nw; k++) { spw *w = &S->W[k];   /* --surf-overlap: the feathered means into the uint8 map */
        for (int a = 0; a < w->mh; a++) for (int b = 0; b < w->mw; b++) { const float *B = w->blk[(size_t)(a / SURF_BLK) * w->bw + b / SURF_BLK]; if (!B) continue;
            const size_t o = (size_t)(a % SURF_BLK) * SURF_BLK + b % SURF_BLK; const float sw = B[SURF_BLK * SURF_BLK + o];
            if (sw > 0) w->map[(size_t)a * w->mw + b] = (uint8_t)(1 + lrint((double)B[o] / sw * 254)); } }
    for (int k = 0; k < S->nw; k++) { const spw *w = &S->W[k]; snprintf(p, sizeof p, "%s/%s.tif", dir, w->name);
        int ow = w->mw, oh = w->mh; uint8_t *out = w->map;
        const double up = S->s * S->pv;   /* canvas px per map px */
        if (!native && up != 1) {
            ow = (int)ceil(w->w / w->sc); oh = (int)ceil(w->h / w->sc); out = calloc((size_t)ow * oh, 1);
            if (!out) { fprintf(stderr, "predict --surf: out of memory for the canvas map\n"); return -1; }
            for (int y = 0; y < oh; y++) for (int x = 0; x < ow; x++) {
                const double my = y / up, mx = x / up; const int y0 = (int)floor(my), x0 = (int)floor(mx); const double fy = my - y0, fx = mx - x0;
                double sv = 0, sw = 0;
                for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++) { const int yy = y0 + dy, xx = x0 + dx; if (yy < 0 || xx < 0 || yy >= w->mh || xx >= w->mw) continue;
                    const uint8_t v = w->map[(size_t)yy * w->mw + xx]; if (!v) continue; const double wt = (dy ? fy : 1 - fy) * (dx ? fx : 1 - fx); sv += wt * v; sw += wt; }
                if (sw > 1e-9) { const long v = lround(sv / sw); out[(size_t)y * ow + x] = (uint8_t)(v < 1 ? 1 : v > 255 ? 255 : v); }
            }
        }
        if (tiffw_write(p, ow, oh, 1, out, 1)) { fprintf(stderr, "predict --surf: cannot write %s\n", p); rc = -1; }
        if (out != w->map) free(out); }
    fprintf(stderr, "predict --surf: %ld of %ld patches run, %d maps in %s\n", S->nrun, S->np, S->nw, dir);
    if (S->nrun < S->nfit) fprintf(stderr, "DEFAULTED: predict --surf: %ld planned patches did not run (their windows skipped: absent CT, or no touched voxel): their pixels stay 0\n", S->nfit - S->nrun);
    return rc;
}
