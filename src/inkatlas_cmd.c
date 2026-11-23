/* ufsm ink-atlas SURF-DIR MESH-DIR OUT-DIR --ct ROOT KEY (2026-11-23, atlasink): one First Letters ink-atlas surface (HF dataset
   SharpMeow/first-letters-ink-atlas: meta.json, ct_mid.png, ensemble/ORDER.png, ensemble-rev/ORDER.png, maps/MODEL/c+0_s1_ORDER.png)
   on its own traced mesh (meta.json mesh.base) -> confidence-gated pseudo ink labels in ink-teach's window form.

   1. Registration: the mesh's CT rendered through our render path (render_wrap_chunked, depth 0, the canvas rule: output px = render_scale
      / mesh scale px per grid step, px c at grid c / gs) onto the atlas canvas, against ct_mid.png: Pearson r over the px both cover, for
      the 8 grid symmetries (the transposes only on a square canvas) x shifts +-4 px (every 3rd px); the best is applied to every atlas map.
      Dropped (exit 3) when the best r < --min-r (0.5).
   2. Reading side: the atlas page's face_votes (each alive model pair's face-check consistent_face); kept only when the winner has
      >= --min-votes (3) and >= --vote-ratio (2) x the loser, else dropped (exit 3): the labels would sit on an unknown face. forward:
      ensemble/forward.png, the c+0 forward model maps and meta ensemble.ensemble.stats; reverse: ensemble-rev/reverse.png, the reverse maps,
      ensemble.ensemble_reverse_reading.stats. The atlas forward order runs along --fwd-sign (-1, measured by ink-atlas-check) x our
      +N = du x dv; when the reading face is -N the mesh and the label rasters are written TRANSPOSED (du x dv flips: +N, the face the
      onfly ink splat labels, becomes the reading face).
   3. Gate (inkatlas.h ia_gate) over the px both CT renders cover: sure ink z >= --z-ink (2.5) with >= --agree (3) alive models at z_m >=
      --model-z (2); sure background z <= --z-bg (0.5); the rest unlabelled. Dropped (exit 3) above --max-ink (0.15) ink among the labelled px.
   Out: OUT/<id> (the mesh as tifxyz, with voxel_size_um), OUT/teach.zarr + teachmask.zarr (255 / 0, mask 255 = supervised, label_scale = canvas um /
   voxel um), OUT/segs-teach.json (one ink-teach window), OUT/ct_ours.jpg; one "ink-atlas-row" line on stdout (id, r, symmetry, shift,
   order, votes, px, ink %, mm2). */
#include "inkatlas.h"
#include "imgops.h"
#include "json.h"
#include "optcheck.h"
#include "store.h"
#include "tiff.h"
#include "tiffw.h"
#include "zarr3.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <zstd.h>

static const char *aopt(int argc, char **argv, const char *k, const char *d) { for (int i = 5; i + 1 < argc; i++) if (!strcmp(argv[i], opt_n(k))) return argv[i + 1]; return d; }
static int a_mkdir_p(const char *path) {
    char p[1400]; snprintf(p, sizeof p, "%s", path);
    for (char *q = p + 1; *q; q++) if (*q == '/') { *q = 0; if (mkdir(p, 0775) && errno != EEXIST) return -1; *q = '/'; }
    return mkdir(p, 0775) && errno != EEXIST ? -1 : 0;
}
static json *a_json_file(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) return nullptr;
    fseek(f, 0, SEEK_END); const long n = ftell(f); fseek(f, 0, SEEK_SET); char *t = malloc((size_t)n + 1); const size_t rd = fread(t, 1, (size_t)n, f); fclose(f); t[rd] = 0;
    json *j = json_parse(t, rd); free(t); return j;
}
static uint8_t *a_png_gray(const char *path, int *w, int *h) {   /* channel 0 of an 8-bit PNG */
    uint8_t *rgb = png_read_rgb(path, w, h); if (!rgb) return nullptr;
    const size_t n = (size_t)*w * *h; uint8_t *g = malloc(n); for (size_t i = 0; i < n; i++) g[i] = rgb[3 * i]; free(rgb); return g;
}
/* a 2-D uint8 zarr v3 (1024^2 zstd chunks, all-zero chunks not written), OME pixel size um: ink-teach's it_write_zarr */
static int a_write_zarr(const char *dir, const uint8_t *a, int64_t H, int64_t W, double um) {
    enum { CK = 1024 }; char p[1400];
    snprintf(p, sizeof p, "%s/0", dir); if (a_mkdir_p(p)) return -1;
    snprintf(p, sizeof p, "%s/zarr.json", dir); FILE *f = fopen(p, "wb"); if (!f) return -1;
    fprintf(f, "{\"zarr_format\": 3, \"node_type\": \"group\", \"attributes\": {\"ufsm\": {\"content\": \"ink-atlas\"}, \"ome\": {\"version\": \"0.5\", \"multiscales\": [{\"axes\": "
               "[{\"name\": \"y\", \"type\": \"space\", \"unit\": \"micrometer\"}, {\"name\": \"x\", \"type\": \"space\", \"unit\": \"micrometer\"}], \"datasets\": [{\"path\": \"0\", "
               "\"coordinateTransformations\": [{\"type\": \"scale\", \"scale\": [%.9g, %.9g]}]}]}]}}}\n", um, um); fclose(f);
    snprintf(p, sizeof p, "%s/0/zarr.json", dir); f = fopen(p, "wb"); if (!f) return -1;
    fprintf(f, "{\"zarr_format\": 3, \"node_type\": \"array\", \"shape\": [%lld, %lld], \"data_type\": \"uint8\", \"chunk_grid\": {\"name\": \"regular\", \"configuration\": "
               "{\"chunk_shape\": [%d, %d]}}, \"chunk_key_encoding\": {\"name\": \"default\", \"configuration\": {\"separator\": \"/\"}}, \"fill_value\": 0, \"codecs\": "
               "[{\"name\": \"bytes\"}, {\"name\": \"zstd\", \"configuration\": {\"level\": 5, \"checksum\": false}}]}\n", (long long)H, (long long)W, CK, CK); fclose(f);
    uint8_t *c = malloc((size_t)CK * CK); const size_t cb = ZSTD_compressBound((size_t)CK * CK); uint8_t *z = malloc(cb); int rc = 0;
    for (int64_t r0 = 0; r0 < H && !rc; r0 += CK) for (int64_t c0 = 0; c0 < W && !rc; c0 += CK) {
        memset(c, 0, (size_t)CK * CK); int any = 0; const int64_t n = c0 + CK <= W ? CK : W - c0;
        for (int64_t r = 0; r < CK && r0 + r < H; r++) { memcpy(c + (size_t)r * CK, a + (size_t)(r0 + r) * W + c0, (size_t)n); for (int64_t q = 0; q < n && !any; q++) any = c[(size_t)r * CK + q] != 0; }
        if (!any) continue;
        const size_t zn = ZSTD_compress(z, cb, c, (size_t)CK * CK, 5); if (ZSTD_isError(zn)) { rc = -1; break; }
        snprintf(p, sizeof p, "%s/0/c/%lld", dir, (long long)(r0 / CK)); if (a_mkdir_p(p)) { rc = -1; break; }
        snprintf(p, sizeof p, "%s/0/c/%lld/%lld", dir, (long long)(r0 / CK), (long long)(c0 / CK)); f = fopen(p, "wb");
        if (!f || fwrite(z, 1, zn, f) != zn) rc = -1;
        if (f) fclose(f);
    }
    free(c); free(z); return rc;
}
/* in place: an atlas raster (W x H) through symmetry t and shift (dy, dx) onto our canvas */
static uint8_t *a_register(uint8_t *in, int W, int H, int t, int dy, int dx) {
    uint8_t *x = malloc((size_t)W * H), *o = malloc((size_t)W * H); int w2, h2; ia_xform(in, W, H, t, x, &w2, &h2); ia_shift(x, W, H, dy, dx, o); free(x); free(in); return o;
}
static void a_transpose(uint8_t **a, int W, int H) { uint8_t *o = malloc((size_t)W * H); int w2, h2; ia_xform(*a, W, H, 4, o, &w2, &h2); free(*a); *a = o; }

int cmd_ink_atlas(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: ufsm ink-atlas SURF-DIR MESH-DIR OUT-DIR --ct ROOT KEY [--z-ink 2.5] [--z-bg 0.5] [--model-z 2] [--agree 3] [--min-r 0.5]\n"
                        "         [--min-votes 3] [--vote-ratio 2] [--max-ink 0.15] [--order auto|forward|reverse] [--fwd-sign -1] [--threads 6] [--cache DIR]\n"
                        "  one First Letters ink-atlas surface -> gated pseudo ink labels (ink-teach window form; exit 3: dropped)\n");
        return 2; }
    const char *sd = argv[2], *md = argv[3], *out = argv[4]; const char *root = nullptr, *key = nullptr;
    for (int i = 5; i + 2 < argc; i++) if (!strcmp(argv[i], opt_n("--ct"))) { root = argv[i + 1]; key = argv[i + 2]; }
    if (!root) { fprintf(stderr, "ink-atlas: --ct ROOT KEY (the scan's level-0 array) needed\n"); return 2; }
    ia_gate_cfg G = {opt_atof("--z-ink", aopt(argc, argv, "--z-ink", "2.5")), opt_atof("--z-bg", aopt(argc, argv, "--z-bg", "0.5")),
                     opt_atof("--model-z", aopt(argc, argv, "--model-z", "2")), opt_atoi("--agree", aopt(argc, argv, "--agree", "3"))};
    const double min_r = opt_atof("--min-r", aopt(argc, argv, "--min-r", "0.5")), vratio = opt_atof("--vote-ratio", aopt(argc, argv, "--vote-ratio", "2")),
                 max_ink = opt_atof("--max-ink", aopt(argc, argv, "--max-ink", "0.15"));
    /* --fwd-sign: the atlas forward order runs along fwd_sign x our +N. -1 by ink-atlas-check (2026-11-23): our reader3 maps (ink-ext,
       along +N) vs the atlas reader-v3 maps where the meshes meet, the forward - reverse margin with anti-parallel minus parallel normals:
       6 of 9 surfaces say -N, the two clearest by far (0343_5911 anti-parallel r 0.88 forward vs 0.52 reverse, parallel 0.32 reverse vs
       -0.05; 0343_4853 margin +0.31 vs -0.19), the 3 others within 0.11 */
    const int fsign = opt_atoi("--fwd-sign", aopt(argc, argv, "--fwd-sign", "-1"));
    const int min_votes = opt_atoi("--min-votes", aopt(argc, argv, "--min-votes", "3")), nth = opt_atoi("--threads", aopt(argc, argv, "--threads", "6"));
    const char *ord_opt = aopt(argc, argv, "--order", "auto"), *cache = aopt(argc, argv, "--cache", nullptr);
    char p[1400];
    snprintf(p, sizeof p, "%s/meta.json", sd); json *M = a_json_file(p); if (!M) { fprintf(stderr, "ink-atlas: cannot read %s\n", p); return 1; }
    const char *id = json_str(json_get(M, "id"), "?");
    const double vum = json_num(json_get(M, "voxel_um"), 0), cum = json_num(json_get(M, "canvas_um"), 0), rs = json_num(json_path(M, "render.render_scale"), 0);
    const json *hw = json_path(M, "render.canvas_hw"); const int CH = (int)json_num(json_at(hw, 0), 0), CW = (int)json_num(json_at(hw, 1), 0);
    if (!(vum > 0) || !(cum > 0) || !(rs > 0) || CH <= 0 || CW <= 0) { fprintf(stderr, "ink-atlas: %s: meta.json lacks voxel_um / canvas_um / render_scale / canvas_hw\n", id); json_free(M); return 1; }
    /* face votes */
    int vf = 0, vr = 0, vt = 0; { const json *pr = json_path(M, "face_check.pairs");
        for (size_t i = 0; pr && i < pr->n; i++) { const char *cf = json_str(json_get(pr->items[i], "consistent_face"), "tie"); vf += !strcmp(cf, "forward"); vr += !strcmp(cf, "reverse"); vt += !strcmp(cf, "forward") == 0 && !strcmp(cf, "reverse") == 0; } }
    int order = !strcmp(ord_opt, "forward") ? 1 : !strcmp(ord_opt, "reverse") ? -1 : ia_face_order(vf, vr, min_votes, vratio);
    const char *on = order > 0 ? "forward" : "reverse";
    /* the mesh */
    float *XYZ[3] = {nullptr, nullptr, nullptr}; int mw = 0, mh = 0; double msc = 0;
    { const char *nm[3] = {"x", "y", "z"};
      for (int k = 0; k < 3; k++) { int w, h; snprintf(p, sizeof p, "%s/%s.tif", md, nm[k]); XYZ[k] = img_read_f32(p, &w, &h);
          if (!XYZ[k] || (k && (w != mw || h != mh))) { fprintf(stderr, "ink-atlas: %s: cannot read %s\n", id, p); return 1; } mw = w; mh = h; }
      snprintf(p, sizeof p, "%s/meta.json", md); json *mm = a_json_file(p); msc = json_num(json_at(json_get(mm, "scale"), 0), 0); json_free(mm);
      if (!(msc > 0)) { fprintf(stderr, "ink-atlas: %s: no scale in %s\n", id, p); return 1; } }
    const double gs = rs / msc;   /* canvas px per grid step */
    fprintf(stderr, "ink-atlas: %s: mesh %d x %d (scale %.6g), canvas %d x %d at %.4g um (grid -> %d x %d px), votes forward %d reverse %d tie %d -> %s\n", id, mw, mh, msc, CW, CH, cum,
            render_out_px(mw, gs), render_out_px(mh, gs), vf, vr, vt, order ? on : "undecided");
    /* 1. our CT render on the canvas */
    store *st = store_open(root); z3 *ct = st ? z3_open(st, key, cache) : nullptr;
    if (!ct) { fprintf(stderr, "ink-atlas: %s: cannot open the CT %s %s: %s\n", id, root, key, z3_error()); return 1; }
    const size_t n = (size_t)CW * CH; uint8_t *ours = calloc(n, 1); const int dep = 0;
    if (render_wrap_chunked(&ct, 1, &dep, XYZ[0], XYZ[1], XYZ[2], mw, mh, gs, &ours, CW, CH, nullptr, (size_t)2 << 30, nth, nth, 0)) { fprintf(stderr, "ink-atlas: %s: render failed\n", id); return 1; }
    z3_close(ct);
    snprintf(p, sizeof p, "%s", out); if (a_mkdir_p(p)) { fprintf(stderr, "ink-atlas: cannot create %s\n", out); return 1; }
    snprintf(p, sizeof p, "%s/ct_ours.jpg", out); jpeg_write_gray(p, CW, CH, ours, 85);
    int aw, ah; snprintf(p, sizeof p, "%s/ct_mid.png", sd); uint8_t *ctm = a_png_gray(p, &aw, &ah);
    if (!ctm || aw != CW || ah != CH) { fprintf(stderr, "ink-atlas: %s: %s missing or not %d x %d\n", id, p, CW, CH); return 1; }
    double best = -2, r_id = NAN; int bt = 0, bdy = 0, bdx = 0; size_t bn = 0;
    { uint8_t *b = malloc(n);
      for (int t = 0; t < 8; t++) { if ((t & 4) && CW != CH) continue; int w2, h2; ia_xform(ctm, CW, CH, t, b, &w2, &h2);
          for (int dy = -4; dy <= 4; dy++) for (int dx = -4; dx <= 4; dx++) { size_t m; const double r = ia_pearson(ours, b, CW, CH, dy, dx, 3, &m);
              if (!t && !dy && !dx) r_id = r;
              if (r > best) { best = r; bt = t; bdy = dy; bdx = dx; bn = m; } } }
      free(b); }
    fprintf(stderr, "ink-atlas: %s: registration r %.3f as given, best %.3f at symmetry %d shift (%d, %d) over %zu px\n", id, r_id, best, bt, bdy, bdx, bn);
    const double mm2px = cum * cum * 1e-6;
#define ROW(st_, ni_, nb_) printf("ink-atlas-row\t%s\t%s\t%.3f\t%.3f\t%d\t%d,%d\t%s\t%d/%d/%d\t%zu\t%zu\t%.2f\t%.2f\t%.2f\n", json_str(json_get(M, "scroll"), "?"), id, r_id, best, bt, bdy, bdx, \
        order ? on : "undecided", vf, vr, vt, (size_t)(ni_), (size_t)(nb_), (ni_) + (nb_) ? 100.0 * (ni_) / ((ni_) + (nb_)) : 0, (ni_) * mm2px, (nb_) * mm2px); printf("ink-atlas-status\t%s\t%s\n", id, st_)
    if (!(best >= min_r)) { ROW("drop: registration", 0, 0); fprintf(stderr, "ink-atlas: %s: dropped (registration r %.3f < %.2f)\n", id, best, min_r); return 3; }
    if (!order) { ROW("drop: face undecided", 0, 0); fprintf(stderr, "ink-atlas: %s: dropped (face votes %d / %d / %d tie undecided)\n", id, vf, vr, vt); return 3; }
    ctm = a_register(ctm, CW, CH, bt, bdy, bdx);
    /* 2-3. the reading side's ensemble and model maps, gated */
    int w, h; snprintf(p, sizeof p, order > 0 ? "%s/ensemble/forward.png" : "%s/ensemble-rev/reverse.png", sd); uint8_t *ens = a_png_gray(p, &w, &h);
    if (!ens || w != CW || h != CH) { fprintf(stderr, "ink-atlas: %s: %s missing or the wrong size\n", id, p); return 1; }
    ens = a_register(ens, CW, CH, bt, bdy, bdx);
    const json *al = json_path(M, "face_check.alive"), *sts = json_path(M, order > 0 ? "ensemble.ensemble.stats" : "ensemble.ensemble_reverse_reading.stats");
    const int nm = al ? (int)al->n : 0; uint8_t **mod = calloc(nm ? nm : 1, sizeof *mod); double *mu = calloc(nm ? nm : 1, sizeof *mu), *sg = calloc(nm ? nm : 1, sizeof *sg);
    for (int m = 0; m < nm; m++) {
        const char *mn = json_str(json_at(al, m), "?"); const json *s = json_get(sts, mn); mu[m] = json_num(json_get(s, "mean"), 0); sg[m] = json_num(json_get(s, "sd"), 0);
        snprintf(p, sizeof p, "%s/maps/%s/c+0_s1_%s.png", sd, mn, on); mod[m] = a_png_gray(p, &w, &h);
        if (!mod[m] || w != CW || h != CH || !(sg[m] > 0)) { fprintf(stderr, "ink-atlas: %s: model %s: %s missing / wrong size or no stats\n", id, mn, p); return 1; }
        mod[m] = a_register(mod[m], CW, CH, bt, bdy, bdx);
    }
    uint8_t *cov = malloc(n), *lab = malloc(n), *msk = malloc(n); for (size_t k = 0; k < n; k++) cov[k] = ours[k] > 0 && ctm[k] > 0;
    size_t ni, nb; ia_gate(ens, (const uint8_t *const *)mod, mu, sg, nm, cov, n, &G, lab, msk, &ni, &nb);
    const double fi = ni + nb ? (double)ni / (ni + nb) : 0;
    fprintf(stderr, "ink-atlas: %s: %d alive models, %zu sure ink + %zu sure background px (%.2f %% ink) = %.1f + %.1f mm2\n", id, nm, ni, nb, 100 * fi, ni * mm2px, nb * mm2px);
    if (fi > max_ink) { ROW("drop: ink fraction", ni, nb); fprintf(stderr, "ink-atlas: %s: dropped (%.1f %% ink > %.0f %%: noise or the wrong face)\n", id, 100 * fi, 100 * max_ink); return 3; }
    if (!ni) { ROW("drop: no ink", ni, nb); return 3; }
    /* out: the mesh (transposed for the reverse side), the rasters, the window */
    int LW = CW, LH = CH, ow = mw, oh = mh;
    const int flip = order * fsign < 0;   /* the reading face is -N: transpose so it becomes +N */
    if (flip) {
        a_transpose(&lab, CW, CH); a_transpose(&msk, CW, CH); LW = CH; LH = CW; ow = mh; oh = mw;
        for (int k = 0; k < 3; k++) { float *t = malloc((size_t)mw * mh * sizeof *t);
            for (int r = 0; r < mw; r++) for (int c = 0; c < mh; c++) t[(size_t)r * mh + c] = XYZ[k][(size_t)c * mw + r];
            free(XYZ[k]); XYZ[k] = t; }
    }
    char mdir[1400]; snprintf(mdir, sizeof mdir, "%s/%s", out, id);   /* (the tifxyz named after the surface: ink logs name meshes by their dir) */
    if (tifxyz_write(mdir, ow, oh, XYZ[0], XYZ[1], XYZ[2], 1 / msc, vum, -1, flip ? "ink-atlas: the atlas mesh transposed (reading face -N)" : "ink-atlas: the atlas mesh")) {
        fprintf(stderr, "ink-atlas: %s: cannot write %s\n", id, mdir); return 1; }
    snprintf(p, sizeof p, "%s/teach.zarr", out); if (a_write_zarr(p, lab, LH, LW, cum)) { fprintf(stderr, "ink-atlas: cannot write %s\n", p); return 1; }
    snprintf(p, sizeof p, "%s/teachmask.zarr", out); if (a_write_zarr(p, msk, LH, LW, cum)) { fprintf(stderr, "ink-atlas: cannot write %s\n", p); return 1; }
    snprintf(p, sizeof p, "%s/segs-teach.json", out); FILE *jf = fopen(p, "wb"); if (!jf) { fprintf(stderr, "ink-atlas: cannot write %s\n", p); return 1; }
    fprintf(jf, "{\"note\": \"ink-atlas %s: %s reading side (face votes %d / %d / %d tie), registration r %.3f (symmetry %d shift %d,%d), gate z >= %g with >= %d models at z >= %g | z <= %g\",\n"
                " \"segments\": [\n  {\"name\": \"%s-atlas\", \"tifxyz\": \"%s\", \"labels\": \"%s/teach.zarr/0\", \"mask\": \"%s/teachmask.zarr/0\", \"label_scale\": %.9g, \"crop\": [0, 0, %d, %d], "
                "\"teach_px\": %zu, \"teach_ink_px\": %zu}\n ]}\n", id, on, vf, vr, vt, best, bt, bdy, bdx, G.z_ink, G.agree, G.model_z, G.z_bg, id, mdir, out, out, 1 / rs, LW, LH, ni + nb, ni);
    fclose(jf);
    ROW("kept", ni, nb);
    for (int m = 0; m < nm; m++) free(mod[m]);
    free(mod); free(mu); free(sg); free(cov); free(lab); free(msk); free(ens); free(ctm); free(ours); for (int k = 0; k < 3; k++) free(XYZ[k]); json_free(M);
    return 0;
#undef ROW
}

/* ufsm ink-atlas-check SURF-DIR MESH-DIR --shift DY,DX --ref WRAP:MAP.tif:DU .. (2026-11-23, atlasink, lead: verify the atlas "forward"
   order = our +N): the atlas reader-v3 c+0 maps of both orders against our own reader3 map (ink-ext, read along +N of OUR wrap: R1E's
   teacher maps; map px (x, y) at wrap grid (x DU, y DU)) where the two meshes trace the same papyrus. Every 2nd atlas canvas px (registered
   by --shift as ink-atlas found it) is matched to the nearest wrap map px within --tol (3) voxels whose normal is within 37 deg of +-the
   atlas normal; pairs split by the sign of the normals' dot. If the atlas forward order reads along +N: with parallel normals our map
   correlates with "forward", with anti-parallel ones with "reverse". Prints the Pearson r of each (sign, order). */
typedef struct { const float *X, *Y, *Z; int w, h; } ck_grid;
static int ck_pn(const ck_grid *g, double gu, double gv, double P[3], double N[3]) {   /* (z, y, x), N = du x dv as ssurf_pn / ink-ext */
    if (gu < 0 || gv < 0 || gu > g->w - 1 || gv > g->h - 1) return -1;
    int c0 = (int)gu, r0 = (int)gv; if (c0 >= g->w - 1) c0 = g->w - 2; if (r0 >= g->h - 1) r0 = g->h - 2;
    const double fu = gu - c0, fv = gv - r0; const size_t i00 = (size_t)r0 * g->w + c0, i01 = i00 + 1, i10 = i00 + g->w, i11 = i10 + 1;
    if (g->Z[i00] < 0 || g->Z[i01] < 0 || g->Z[i10] < 0 || g->Z[i11] < 0) return -1;
    const float *A[3] = {g->Z, g->Y, g->X}; double du[3], dv[3];
    for (int d = 0; d < 3; d++) { const double a = A[d][i00], b = A[d][i01], c = A[d][i10], e = A[d][i11];
        P[d] = (1 - fv) * ((1 - fu) * a + fu * b) + fv * ((1 - fu) * c + fu * e); du[d] = (1 - fv) * (b - a) + fv * (e - c); dv[d] = (1 - fu) * (c - a) + fu * (e - b); }
    N[0] = du[1] * dv[2] - du[2] * dv[1]; N[1] = du[2] * dv[0] - du[0] * dv[2]; N[2] = du[0] * dv[1] - du[1] * dv[0];
    const double l = sqrt(N[0] * N[0] + N[1] * N[1] + N[2] * N[2]); if (l < 1e-12) return -1;
    for (int d = 0; d < 3; d++) N[d] /= l;
    return 0;
}
static int ck_load(const char *dir, float *G[3], int *w, int *h) {
    const char *nm[3] = {"x", "y", "z"}; char p[1400];
    for (int k = 0; k < 3; k++) { int a, b; snprintf(p, sizeof p, "%s/%s.tif", dir, nm[k]); G[k] = img_read_f32(p, &a, &b); if (!G[k] || (k && (a != *w || b != *h))) return -1; *w = a; *h = b; }
    return 0;
}
typedef struct { float p[3], n[3]; uint8_t v; } ck_pt;
typedef struct { double s[2][2][5]; size_t n[2][2]; } ck_acc;   /* [sign][order] sums x, y, xx, yy, xy */
int cmd_ink_atlas_check(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm ink-atlas-check SURF-DIR MESH-DIR [--shift DY,DX] [--tol 3] [--center Y,X] [--ref WRAP:MAP.tif:DU ..] [--self MAP.tif:DU]\n"); return 2; }
    const char *sd = argv[2], *md = argv[3]; int sdy = 0, sdx = 0; double tol = 3;
    for (int i = 4; i + 1 < argc; i++) { if (!strcmp(argv[i], opt_n("--shift"))) sscanf(argv[i + 1], "%d,%d", &sdy, &sdx); if (!strcmp(argv[i], opt_n("--tol"))) tol = opt_atof("--tol", argv[i + 1]); }
    char p[1400]; snprintf(p, sizeof p, "%s/meta.json", sd); json *M = a_json_file(p); if (!M) { fprintf(stderr, "ink-atlas-check: cannot read %s\n", p); return 1; }
    const double rs = json_num(json_path(M, "render.render_scale"), 0); const json *hw = json_path(M, "render.canvas_hw");
    const int CH = (int)json_num(json_at(hw, 0), 0), CW = (int)json_num(json_at(hw, 1), 0);
    float *AG[3]; int aw = 0, ah = 0; if (ck_load(md, AG, &aw, &ah)) { fprintf(stderr, "ink-atlas-check: cannot read the mesh %s\n", md); return 1; }
    snprintf(p, sizeof p, "%s/meta.json", md); json *mm = a_json_file(p); const double msc = json_num(json_at(json_get(mm, "scale"), 0), 0); json_free(mm);
    const double gs = rs / msc; ck_grid A = {AG[0], AG[1], AG[2], aw, ah};
    double cy = 0, cx = 0; for (int i = 4; i + 1 < argc; i++) if (!strcmp(argv[i], opt_n("--center"))) sscanf(argv[i + 1], "%lf,%lf", &cy, &cx);
    if (cx > 0) {   /* the mesh's +N about the scroll axis (y, x): outward share */
        size_t no = 0, nn = 0; for (int r = 0; r + 1 < ah; r += 2) for (int c = 0; c + 1 < aw; c += 2) { double P[3], N[3]; if (ck_pn(&A, c + 0.5, r + 0.5, P, N)) continue;
            const double ry = P[1] - cy, rx = P[2] - cx; nn++; no += N[1] * ry + N[2] * rx > 0; }
        printf("ink-atlas-check\t%s\tradial\t+N outward share %.3f of %zu\n", json_str(json_get(M, "id"), "?"), nn ? (double)no / nn : 0, nn); }
    uint8_t *mp[2]; const char *on[2] = {"forward", "reverse"};
    for (int o = 0; o < 2; o++) { int w, h; snprintf(p, sizeof p, "%s/maps/reader-v3/c+0_s1_%s.png", sd, on[o]); uint8_t *g = a_png_gray(p, &w, &h);
        if (!g || w != CW || h != CH) { fprintf(stderr, "ink-atlas-check: %s missing\n", p); return 1; } mp[o] = a_register(g, CW, CH, 0, sdy, sdx); }
    double lo[3] = {1e30, 1e30, 1e30}, hi[3] = {-1e30, -1e30, -1e30};
    for (size_t i = 0; i < (size_t)aw * ah; i++) if (AG[2][i] >= 0) { const double q[3] = {AG[2][i], AG[1][i], AG[0][i]}; for (int d = 0; d < 3; d++) { if (q[d] < lo[d]) lo[d] = q[d]; if (q[d] > hi[d]) hi[d] = q[d]; } }
    for (int d = 0; d < 3; d++) { lo[d] -= 30; hi[d] += 30; }
    ck_acc acc; memset(&acc, 0, sizeof acc);
    for (int a = 4; a + 1 < argc; a++) {
        if (strcmp(argv[a], opt_n("--ref"))) continue;
        char spec[1400]; snprintf(spec, sizeof spec, "%s", argv[a + 1]); char *c1 = strchr(spec, ':'), *c2 = c1 ? strchr(c1 + 1, ':') : nullptr;
        if (!c2) { fprintf(stderr, "ink-atlas-check: --ref WRAP:MAP.tif:DU\n"); return 2; }
        *c1 = 0; *c2 = 0; const double du = atof(c2 + 1);
        float *RG[3]; int rw = 0, rh = 0; if (ck_load(spec, RG, &rw, &rh)) { fprintf(stderr, "ink-atlas-check: cannot read %s\n", spec); return 1; }
        ck_grid R = {RG[0], RG[1], RG[2], rw, rh};
        tiff *t = tiff_open_file(c1 + 1); tiff_page pg; if (!t || tiff_page_info(t, 0, &pg) || pg.bits != 8 || pg.spp != 1) { fprintf(stderr, "ink-atlas-check: %s not a uint8 TIFF\n", c1 + 1); return 1; }
        uint8_t *mv = malloc((size_t)pg.w * pg.h); if (tiff_read_page(t, 0, mv)) return 1; tiff_close(t);
        /* the wrap's map px inside the atlas box, hashed into 4-voxel cells */
        enum { HB = 1 << 22 }; const double CS = 4; int *head = malloc(HB * sizeof *head); for (int i = 0; i < HB; i++) head[i] = -1;
        size_t np = 0, cap = 1 << 20; ck_pt *pt = malloc(cap * sizeof *pt); int *nx = malloc(cap * sizeof *nx);
        for (int y = 0; y < pg.h; y++) for (int x = 0; x < pg.w; x++) {
            const uint8_t v = mv[(size_t)y * pg.w + x]; if (!v) continue;
            const double gu = x * du, gv = y * du; const int c0 = (int)gu, r0 = (int)gv; if (c0 >= rw || r0 >= rh) continue;
            const float z0 = RG[2][(size_t)r0 * rw + c0]; if (z0 < lo[0] || z0 > hi[0]) continue;
            double P[3], N[3]; if (ck_pn(&R, gu, gv, P, N)) continue;
            if (P[1] < lo[1] || P[1] > hi[1] || P[2] < lo[2] || P[2] > hi[2]) continue;
            if (np == cap) { cap *= 2; pt = realloc(pt, cap * sizeof *pt); nx = realloc(nx, cap * sizeof *nx); }
            for (int d = 0; d < 3; d++) { pt[np].p[d] = (float)P[d]; pt[np].n[d] = (float)N[d]; } pt[np].v = v;
            const uint64_t k = ((uint64_t)(int64_t)floor(P[0] / CS) * 73856093u ^ (uint64_t)(int64_t)floor(P[1] / CS) * 19349663u ^ (uint64_t)(int64_t)floor(P[2] / CS) * 83492791u) & (HB - 1);
            nx[np] = head[k]; head[k] = (int)np; np++;
        }
        size_t used[2] = {0, 0};
        for (int r = 0; r < CH; r += 2) for (int c = 0; c < CW; c += 2) {
            double P[3], N[3]; if (ck_pn(&A, c / gs, r / gs, P, N)) continue;
            int best = -1; double bd = tol * tol;
            for (int dz = -1; dz <= 1; dz++) for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
                const int64_t kz = (int64_t)floor(P[0] / CS) + dz, ky = (int64_t)floor(P[1] / CS) + dy, kx = (int64_t)floor(P[2] / CS) + dx;
                const uint64_t k = ((uint64_t)kz * 73856093u ^ (uint64_t)ky * 19349663u ^ (uint64_t)kx * 83492791u) & (HB - 1);
                for (int q = head[k]; q >= 0; q = nx[q]) { double d2 = 0; for (int d = 0; d < 3; d++) d2 += (pt[q].p[d] - P[d]) * (pt[q].p[d] - P[d]);
                    if (d2 < bd) { const double dt = pt[q].n[0] * N[0] + pt[q].n[1] * N[1] + pt[q].n[2] * N[2]; if (fabs(dt) >= 0.8) { bd = d2; best = q; } } } }
            if (best < 0) continue;
            const double dt = pt[best].n[0] * N[0] + pt[best].n[1] * N[1] + pt[best].n[2] * N[2]; const int s = dt > 0 ? 0 : 1; used[s]++;
            for (int o = 0; o < 2; o++) { const double xv = pt[best].v, yv = mp[o][(size_t)r * CW + c]; if (!yv) continue; double *S = acc.s[s][o];
                S[0] += xv; S[1] += yv; S[2] += xv * xv; S[3] += yv * yv; S[4] += xv * yv; acc.n[s][o]++; }
        }
        fprintf(stderr, "ink-atlas-check: %s: %zu map px in the atlas box, %zu atlas px matched with parallel normals, %zu anti-parallel\n", spec, np, used[0], used[1]);
        free(head); free(pt); free(nx); free(mv); for (int k = 0; k < 3; k++) free(RG[k]);
    }
    /* --self MAP.tif:DU: our reader3 map of THIS mesh (ink-ext along +N; map px (x, y) at grid (x DU, y DU)), each canvas px against the
       map px nearest its grid point: the same mesh, so the order that reads +N correlates (lead 11-23: per-surface sign, the cross-mesh
       votes above were mixed) */
    for (int a = 4; a + 1 < argc; a++) {
        if (strcmp(argv[a], opt_n("--self"))) continue;
        char spec[1400]; snprintf(spec, sizeof spec, "%s", argv[a + 1]); char *c1 = strrchr(spec, ':'); if (!c1) { fprintf(stderr, "ink-atlas-check: --self MAP.tif:DU\n"); return 2; }
        *c1 = 0; const double du = atof(c1 + 1);
        tiff *t = tiff_open_file(spec); tiff_page pg; if (!t || tiff_page_info(t, 0, &pg) || pg.bits != 8 || pg.spp != 1) { fprintf(stderr, "ink-atlas-check: %s not a uint8 TIFF\n", spec); return 1; }
        uint8_t *mv = malloc((size_t)pg.w * pg.h); if (tiff_read_page(t, 0, mv)) return 1; tiff_close(t);
        double S[2][5] = {{0}}; size_t n[2] = {0, 0};
        for (int r = 0; r < CH; r++) for (int c = 0; c < CW; c++) {
            const int x = (int)lround(c / gs / du), y = (int)lround(r / gs / du); if (x < 0 || y < 0 || x >= pg.w || y >= pg.h) continue;
            const double xv = mv[(size_t)y * pg.w + x]; if (!xv) continue;
            for (int o = 0; o < 2; o++) { const double yv = mp[o][(size_t)r * CW + c]; if (!yv) continue; S[o][0] += xv; S[o][1] += yv; S[o][2] += xv * xv; S[o][3] += yv * yv; S[o][4] += xv * yv; n[o]++; }
        }
        double rr[2]; for (int o = 0; o < 2; o++) { const double va = S[o][2] - S[o][0] * S[o][0] / n[o], vb = S[o][3] - S[o][1] * S[o][1] / n[o];
            rr[o] = n[o] > 100 && va > 0 && vb > 0 ? (S[o][4] - S[o][0] * S[o][1] / n[o]) / sqrt(va * vb) : NAN; }
        printf("ink-atlas-self\t%s\t%zu\t%.4f\t%.4f\n", json_str(json_get(M, "id"), "?"), n[0], rr[0], rr[1]);
        free(mv);
    }
    for (int s = 0; s < 2; s++) for (int o = 0; o < 2; o++) { const double *S = acc.s[s][o]; const double n = (double)acc.n[s][o];
        const double va = S[2] - S[0] * S[0] / n, vb = S[3] - S[1] * S[1] / n, r = n > 100 && va > 0 && vb > 0 ? (S[4] - S[0] * S[1] / n) / sqrt(va * vb) : NAN;
        printf("ink-atlas-check\t%s\t%s normals\tatlas %s\tpairs %zu\tr %.4f\n", json_str(json_get(M, "id"), "?"), s ? "anti-parallel" : "parallel", on[o], acc.n[s][o], r); }
    for (int o = 0; o < 2; o++) free(mp[o]);
    for (int k = 0; k < 3; k++) free(AG[k]);
    json_free(M); return 0;
}
