/* render --fill (render_wraps_chunked_fill) and the cosmetic push-pull inpaint (img_pushpull_fill):
   - two flat sheets in a masked CT (z 18..20 CT 200 ink 100; z 25..27 CT 220 ink 40; 0 elsewhere) and a surface that sits in the gap
     between them: at z 22 the nearest papyrus is sheet A at 2 voxels (flag +2 along the grid normal, -z here), CT and ink both taken
     there; at z 22.5 both sheets are 2.5 voxels away and the samples at +-2 tie: the larger CT (sheet B, flag 50 + 2) by default,
     sheet A with --fill-prefer 1; with n = 1 nothing is in reach (255, the plain samples); rows on sheet A are flag 0 and equal the
     plain render; invalid vertices stay RFILL_NONE; n = 0 flags only, images byte-identical to the render without --fill;
   - img_pushpull_fill: a 3 x 3 hole in a flat region takes its value (flag 250), a hole wider than 2^levels keeps its interior empty,
     no-surface pixels are never filled;
   - PatchMatch: a striped texture hole restored far better than push-pull, nothing else changed;
   - tifxyz_fill_holes (--fill-grid): an enclosed grid hole bridged and rendered on the sheet (+ RFILL_GRID), border / oversize holes kept. */
#include "imgops.h"
#include "store.h"
#include "z3w.h"
#include "zarr3.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)

int main(void) {
    char dir[] = "/tmp/test_render_fill_XXXXXX"; CHECK(mkdtemp(dir) != nullptr, "mkdtemp");
    char p[800], q[800];
    const int S = 128; const int64_t shape[3] = {48, 64, 64};
    uint8_t *buf = calloc((size_t)S * S * S, 1); z3 *zz[2];
    for (int v = 0; v < 2; v++) {   /* v 0: ink, v 1: masked CT */
        memset(buf, 0, (size_t)S * S * S);
        for (int z = 0; z < shape[0]; z++) for (int y = 0; y < shape[1]; y++) for (int x = 0; x < shape[2]; x++) {
            const int a = z >= 18 && z <= 20, b = z >= 25 && z <= 27;
            buf[((size_t)z * S + y) * S + x] = (uint8_t)(a ? (v ? 200 : 100) : b ? (v ? 220 : 40) : 0);
        }
        snprintf(p, sizeof p, "%s/v%d.zarr/0", dir, v);
        z3w *w = z3w_create(p, shape, S, 0, 0, "{}"); CHECK(w != nullptr, "z3w_create %s", z3w_error());
        CHECK(!z3w_write_shard(w, 0, 0, 0, buf, 2), "write shard %s", z3w_error());
        CHECK(!z3w_close(w), "close");
        snprintf(q, sizeof q, "v%d.zarr/0", v);
        zz[v] = z3_open(store_open(dir), q, nullptr); CHECK(zz[v] != nullptr, "z3_open %s", z3_error());
    }
    /* grid: columns along x (2 voxels per step), rows along y; normal du x dv = -z. Rows 0..7: columns 0..9 at z 22, 10..19 at z 22.5;
       rows 8..15 at z 19 (on sheet A); vertices (15, 0..3) invalid */
    const int w = 20, h = 16; float X[20 * 16], Y[20 * 16], Z[20 * 16];
    for (int r = 0; r < h; r++) for (int c = 0; c < w; c++) {
        const int i = r * w + c; X[i] = 10.f + 2 * c; Y[i] = 10.f + 2 * r; Z[i] = r >= 8 ? 19.f : c < 10 ? 22.f : 22.5f;
        if (r == 15 && c < 4) X[i] = Y[i] = Z[i] = -1.f;
    }
    const double gs = 2; const int ow = render_out_px(w, gs), oh = render_out_px(h, gs); const size_t no = (size_t)ow * oh;
    const float *const Xs[1] = {X}, *const Ys[1] = {Y}, *const Zs[1] = {Z}; const int dp[2] = {1, 0};
    uint8_t *ref[2] = {calloc(no, 1), calloc(no, 1)}, *im[2] = {calloc(no, 1), calloc(no, 1)}, *fl = malloc(no); uint8_t *const fls[1] = {fl};
    CHECK(!render_wraps_chunked(zz, 2, dp, 1, Xs, Ys, Zs, w, h, gs, ref, ow, oh, nullptr, (size_t)1 << 30, 2, 2, 0), "plain render");
#define PX(img, c, r) (img)[(size_t)(r) * ow + (c)]
    const int ca = 6, cb = 30, rg = 6, rs = 24;   /* pixels: column band z 22 / z 22.5, gap rows / sheet rows */
    CHECK(PX(ref[1], ca, rg) == 0 && PX(ref[1], cb, rg) == 0 && PX(ref[1], ca, rs) == 200, "plain CT: void in the gap (%d %d), sheet %d", PX(ref[1], ca, rg), PX(ref[1], cb, rg), PX(ref[1], ca, rs));
    struct { int n, prefer; int fa, fb, cta, ctb, inka, inkb; } cs[] = {
        {4, 0, 2, RFILL_NEG + 2, 200, 110, 100, 40},   /* z 22 -> z 20 (sheet A); z 22.5 tie -> larger CT: z 24.5 (sheet B edge) */
        {4, 1, 2, 2, 200, 100, 100, 100},              /* tie -> +normal side: z 20.5 */
        {4, -1, 2, RFILL_NEG + 2, 200, 110, 100, 40},
        {1, 0, RFILL_EMPTY, RFILL_EMPTY, 0, 0, 0, 0},  /* out of reach */
    };
    for (size_t k = 0; k < sizeof cs / sizeof *cs; k++) {
        for (int v = 0; v < 2; v++) memset(im[v], 0, no);
        memset(fl, RFILL_NONE, no);
        const render_fill F = {cs[k].n, 1, cs[k].prefer, fls, nullptr};
        CHECK(!render_wraps_chunked_fill(zz, 2, dp, 1, Xs, Ys, Zs, w, h, gs, im, ow, oh, nullptr, (size_t)1 << 30, 2, 2, 0, &F), "fill render %zu", k);
        for (int r = 0; r < rg + 4; r++) for (int c = 0; c < 17; c++) {   /* the z 22 band (columns 0..16 avoid the transition) */
            CHECK(PX(fl, c, r) == cs[k].fa && PX(im[1], c, r) == cs[k].cta && PX(im[0], c, r) == cs[k].inka, "case %zu z 22 px %d,%d: flag %d ct %d ink %d", k, c, r, PX(fl, c, r), PX(im[1], c, r), PX(im[0], c, r));
        }
        for (int r = 0; r < rg + 4; r++) for (int c = 21; c < ow; c++) {   /* the z 22.5 band */
            CHECK(PX(fl, c, r) == cs[k].fb && PX(im[1], c, r) == cs[k].ctb && PX(im[0], c, r) == cs[k].inkb, "case %zu z 22.5 px %d,%d: flag %d ct %d ink %d", k, c, r, PX(fl, c, r), PX(im[1], c, r), PX(im[0], c, r));
        }
        size_t on = 0, none = 0, sameon = 1;
        for (size_t i = 0; i < no; i++) {
            if (fl[i] == RFILL_ON) { on++; if (im[0][i] != ref[0][i] || im[1][i] != ref[1][i]) sameon = 0; }
            if (fl[i] == RFILL_NONE) { none++; CHECK(!im[0][i] && !im[1][i], "no-surface pixel rendered"); }
        }
        CHECK(on > 100 && sameon, "case %zu: %zu on-surface pixels, equal to the plain render: %d", k, on, (int)sameon);
        CHECK(none > 0 && PX(fl, 2, oh - 1) == RFILL_NONE, "case %zu: invalid vertices -> RFILL_NONE (%zu)", k, none);
        CHECK(PX(fl, ca, rs) == RFILL_ON, "case %zu: sheet row on the surface", k);
    }
    {   /* n = 0: flags only, the images of the plain render */
        for (int v = 0; v < 2; v++) memset(im[v], 0, no);
        memset(fl, RFILL_NONE, no);
        const render_fill F = {0, 1, 0, fls, nullptr};
        CHECK(!render_wraps_chunked_fill(zz, 2, dp, 1, Xs, Ys, Zs, w, h, gs, im, ow, oh, nullptr, (size_t)1 << 30, 2, 2, 0, &F), "flags-only render");
        CHECK(!memcmp(im[0], ref[0], no) && !memcmp(im[1], ref[1], no), "n 0: images byte-identical to the plain render");
        CHECK(PX(fl, ca, rg) == RFILL_EMPTY && PX(fl, ca, rs) == RFILL_ON, "n 0 flags");
        const render_fill Fb = {4, 1, 0, fls, nullptr};
        CHECK(render_wraps_chunked_fill(zz, 1, dp, 1, Xs, Ys, Zs, w, h, gs, im, ow, oh, nullptr, (size_t)1 << 30, 2, 2, 0, &Fb) == -1, "fill without the CT volume refused");
    }
    /* push-pull */
    {
        const int W = 1100, H = 700; uint8_t *a = malloc((size_t)W * H), *b = malloc((size_t)W * H), *f = malloc((size_t)W * H); uint8_t *const ims[2] = {a, b};
        for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) { const size_t i = (size_t)y * W + x; a[i] = 90; b[i] = 160; f[i] = RFILL_ON; }
        for (int y = 0; y < H; y++) for (int x = 1000; x < W; x++) f[(size_t)y * W + x] = RFILL_NONE;   /* off the surface */
        for (int y = 510; y < 513; y++) for (int x = 510; x < 513; x++) { a[(size_t)y * W + x] = b[(size_t)y * W + x] = 0; f[(size_t)y * W + x] = RFILL_EMPTY; }   /* across the tile seam at 512 */
        for (int y = 100; y < 300; y++) for (int x = 100; x < 300; x++) { a[(size_t)y * W + x] = b[(size_t)y * W + x] = 0; f[(size_t)y * W + x] = RFILL_EMPTY; }
        for (int y = 0; y < 50; y++) f[(size_t)y * W + 1050] = RFILL_EMPTY;   /* inside the no-surface band: nothing to pull from nearby */
        const size_t nf = img_pushpull_fill(ims, 2, f, W, H, 3, 3);
        CHECK(f[(size_t)511 * W + 511] == RFILL_INPAINT && a[(size_t)511 * W + 511] == 90 && b[(size_t)511 * W + 511] == 160, "small hole filled: flag %d a %d b %d", f[(size_t)511 * W + 511], a[(size_t)511 * W + 511], b[(size_t)511 * W + 511]);
        CHECK(f[(size_t)200 * W + 200] == RFILL_EMPTY && !a[(size_t)200 * W + 200], "large hole interior stays empty");
        CHECK(f[(size_t)100 * W + 100] == RFILL_INPAINT && a[(size_t)100 * W + 100] == 90, "large hole rim filled");
        CHECK(f[(size_t)10 * W + 1050] == RFILL_EMPTY, "far from any surface pixel: empty");
        size_t cnt = 0; for (size_t i = 0; i < (size_t)W * H; i++) cnt += f[i] == RFILL_INPAINT;
        CHECK(cnt == nf && nf > 9, "filled count %zu vs %zu", nf, cnt);
        free(a); free(b); free(f);
    }
    {   /* PatchMatch: a periodic texture (period 6 in x, 2 channels in phase) with a 9 x 9 hole: push-pull blurs it to the mean,
           PatchMatch copies the stripes back (error well below push-pull's), every hole pixel stays flagged, nothing else changes */
        const int W = 300, H = 300; uint8_t *a = malloc((size_t)W * H), *b = malloc((size_t)W * H), *f = malloc((size_t)W * H), *a0 = malloc((size_t)W * H); uint8_t *const ims[2] = {a, b};
        for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) { const size_t i = (size_t)y * W + x; a[i] = (uint8_t)(x % 6 < 3 ? 60 : 200); b[i] = (uint8_t)(x % 6 < 3 ? 20 : 120); f[i] = RFILL_ON; }
        memcpy(a0, a, (size_t)W * H);
        for (int y = 140; y < 149; y++) for (int x = 140; x < 149; x++) { a[(size_t)y * W + x] = b[(size_t)y * W + x] = 0; f[(size_t)y * W + x] = RFILL_EMPTY; }
        CHECK(img_pushpull_fill(ims, 2, f, W, H, 3, 2) == 81, "push-pull fills the 9 x 9 hole");
        double epp = 0, epm = 0; for (int y = 140; y < 149; y++) for (int x = 140; x < 149; x++) epp += abs(a[(size_t)y * W + x] - a0[(size_t)y * W + x]);
        uint8_t *a1 = malloc((size_t)W * H); memcpy(a1, a, (size_t)W * H);
        CHECK(!img_patchmatch_fill(ims, 2, f, W, H, 2), "patchmatch");
        int same = 1; for (size_t i = 0; i < (size_t)W * H; i++) if (f[i] != RFILL_INPAINT && a[i] != a1[i]) same = 0;
        for (int y = 140; y < 149; y++) for (int x = 140; x < 149; x++) epm += abs(a[(size_t)y * W + x] - a0[(size_t)y * W + x]);
        CHECK(same, "PatchMatch changes only the inpainted pixels");
        CHECK(epm < 0.25 * epp, "PatchMatch texture error %.0f vs push-pull %.0f", epm, epp);
        free(a); free(b); free(f); free(a0); free(a1);
    }
    {   /* tifxyz_fill_holes: a curved grid z = 19 + sin; an enclosed hole -> interpolated near the true surface and flagged; a hole on the
           border and one larger than maxv untouched; rendered: the hole's pixels are on sheet A (+ RFILL_GRID), no RFILL_NONE inside it */
        const int gw = 24, gh = 16; float GX[24 * 16], GY[24 * 16], GZ[24 * 16], T[24 * 16]; uint8_t vg[24 * 16];
        for (int r = 0; r < gh; r++) for (int c = 0; c < gw; c++) { const int i = r * gw + c; GX[i] = 8.f + 2 * c; GY[i] = 10.f + 2 * r; GZ[i] = T[i] = 19.f + 0.6f * sinf(c * 0.3f) * cosf(r * 0.25f); }
        for (int r = 5; r < 10; r++) for (int c = 6; c < 12; c++) GX[r * gw + c] = GY[r * gw + c] = GZ[r * gw + c] = -1.f;   /* enclosed, 30 vertices */
        for (int r = 0; r < 3; r++) for (int c = 18; c < 21; c++) GZ[r * gw + c] = -1.f;                                        /* touches the border */
        for (int r = 11; r < 15; r++) for (int c = 14; c < 22; c++) GZ[r * gw + c] = -1.f;                                      /* enclosed, 32 > maxv */
        size_t nh = 0, nb = 0; const size_t nf = tifxyz_fill_holes(GX, GY, GZ, gw, gh, 31, vg, &nh, &nb);
        CHECK(nf == 30 && nh == 1 && nb == 1, "grid holes: %zu filled, %zu holes, %zu too big", nf, nh, nb);
        float em = 0; for (int r = 5; r < 10; r++) for (int c = 6; c < 12; c++) { const int i = r * gw + c; CHECK(vg[i], "filled flag"); const float e = fabsf(GZ[i] - T[i]) + fabsf(GX[i] - (8.f + 2 * c)) + fabsf(GY[i] - (10.f + 2 * r)); if (e > em) em = e; }
        CHECK(em < 0.7f, "interpolated vertices off the true grid by %.2f", em);
        CHECK(GZ[1 * gw + 19] < 0 && GZ[12 * gw + 16] < 0 && !vg[12 * gw + 16], "border / large holes untouched");
        const int gow = render_out_px(gw, 2), goh = render_out_px(gh, 2); const size_t gno = (size_t)gow * goh;
        uint8_t *gi[2] = {calloc(gno, 1), calloc(gno, 1)}, *gf = malloc(gno); uint8_t *const gfs[1] = {gf}; const uint8_t *const vgs[1] = {vg};
        memset(gf, RFILL_NONE, gno);
        const float *const GXs[1] = {GX}, *const GYs[1] = {GY}, *const GZs[1] = {GZ};
        const render_fill F = {4, 1, 0, gfs, vgs};
        CHECK(!render_wraps_chunked_fill(zz, 2, dp, 1, GXs, GYs, GZs, gw, gh, 2, gi, gow, goh, nullptr, (size_t)1 << 30, 2, 2, 0, &F), "grid-filled render");
        for (int py = 12; py <= 16; py++) for (int px = 14; px <= 20; px++) {
            const uint8_t f = gf[(size_t)py * gow + px];
            CHECK(f >= RFILL_GRID && f < RFILL_INPAINT && gi[1][(size_t)py * gow + px] >= 150, "hole pixel %d,%d: flag %d ct %d", px, py, f, gi[1][(size_t)py * gow + px]);
        }
        CHECK(gf[(size_t)2 * gow + 2] == RFILL_ON, "valid pixel flag %d", gf[(size_t)2 * gow + 2]);
        free(gi[0]); free(gi[1]); free(gf);
    }
    for (int v = 0; v < 2; v++) { z3_close(zz[v]); free(ref[v]); free(im[v]); }
    free(fl); free(buf);
    snprintf(p, sizeof p, "rm -r '%s'", dir); if (system(p)) {}
    if (fails) { fprintf(stderr, "test_render_fill: %d failures\n", fails); return 1; }
    printf("test_render_fill: ok\n");
    return 0;
}
