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
      ensemble.ensemble_reverse_reading.stats, and the mesh and the label rasters written TRANSPOSED (du x dv flips: +N, the face the onfly
      ink splat labels, becomes the reading face). Assumes the atlas forward order runs along +N = du x dv (our ink-ext / ufsm render
      normal, vc_render_tifxyz's forward order; R1E's teacher maps use it).
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
                        "         [--min-votes 3] [--vote-ratio 2] [--max-ink 0.15] [--order auto|forward|reverse] [--threads 6] [--cache DIR]\n"
                        "  one First Letters ink-atlas surface -> gated pseudo ink labels (ink-teach window form; exit 3: dropped)\n");
        return 2; }
    const char *sd = argv[2], *md = argv[3], *out = argv[4]; const char *root = nullptr, *key = nullptr;
    for (int i = 5; i + 2 < argc; i++) if (!strcmp(argv[i], opt_n("--ct"))) { root = argv[i + 1]; key = argv[i + 2]; }
    if (!root) { fprintf(stderr, "ink-atlas: --ct ROOT KEY (the scan's level-0 array) needed\n"); return 2; }
    ia_gate_cfg G = {opt_atof("--z-ink", aopt(argc, argv, "--z-ink", "2.5")), opt_atof("--z-bg", aopt(argc, argv, "--z-bg", "0.5")),
                     opt_atof("--model-z", aopt(argc, argv, "--model-z", "2")), opt_atoi("--agree", aopt(argc, argv, "--agree", "3"))};
    const double min_r = opt_atof("--min-r", aopt(argc, argv, "--min-r", "0.5")), vratio = opt_atof("--vote-ratio", aopt(argc, argv, "--vote-ratio", "2")),
                 max_ink = opt_atof("--max-ink", aopt(argc, argv, "--max-ink", "0.15"));
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
    if (order < 0) {
        a_transpose(&lab, CW, CH); a_transpose(&msk, CW, CH); LW = CH; LH = CW; ow = mh; oh = mw;
        for (int k = 0; k < 3; k++) { float *t = malloc((size_t)mw * mh * sizeof *t);
            for (int r = 0; r < mw; r++) for (int c = 0; c < mh; c++) t[(size_t)r * mh + c] = XYZ[k][(size_t)c * mw + r];
            free(XYZ[k]); XYZ[k] = t; }
    }
    char mdir[1400]; snprintf(mdir, sizeof mdir, "%s/%s", out, id);   /* (the tifxyz named after the surface: ink logs name meshes by their dir) */
    if (tifxyz_write(mdir, ow, oh, XYZ[0], XYZ[1], XYZ[2], 1 / msc, vum, -1, order < 0 ? "ink-atlas: the atlas mesh transposed (reverse reading side)" : "ink-atlas: the atlas mesh")) {
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
