/* ufsm CLI. */
#include "afv.h"
#include "sysinfo.h"
#include "sample.h"
#include "sources.h"
#include "band.h"
#include "pairacc.h"
#include "votes.h"
#include "wind.h"
#include "phase_check.h"
#include "fuse.h"
#include "edt.h"
#include "store.h"
#include "zarr3.h"
#include "fit.h"
#include "l1pot.h"
#include "fit_emb.h"
#include "unet.h"
#include "checkpoint.h"
#include "fblend.h"
#include "json.h"
#include "nn.h"
#include "tiffw.h"
#include "z3w.h"
#include "papyrus.h"
#include <zlib.h>
#include "tiff.h"
#include "imgops.h"
#include "scanprep.h"
#include <dirent.h>
#include <errno.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#include <pthread.h>
#include <stdatomic.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int cmd_ingest_labels(int argc, char **argv);
int cmd_z3_recode(int argc, char **argv);   /* src/ingest.c */
int cmd_ink_blend(int argc, char **argv);   /* src/ingest.c */
int cmd_ink_seam(int argc, char **argv);   /* src/inkseam.c */
int cmd_ingest_kaggle(int argc, char **argv);
int cmd_ingest_mesh(int argc, char **argv);
int cmd_ls(int argc, char **argv);
int cmd_ingest_zip(int argc, char **argv);
int cmd_raster(int argc, char **argv);
int cmd_train(int argc, char **argv);
int cmd_restore_eval(int argc, char **argv);
int cmd_restore_stats(int argc, char **argv);
int cmd_predict(int argc, char **argv);
int cmd_eval(int argc, char **argv);
int cmd_refine(int argc, char **argv);       /* src/surf.c */
int cmd_surf_stats(int argc, char **argv);
int cmd_self_gate(int argc, char **argv);
int cmd_stitch(int argc, char **argv);   /* src/stitch.c */
int cmd_coreline(int argc, char **argv);   /* src/coreline.c */
int cmd_coreline_check(int argc, char **argv);   /* src/coreline_check.c */
int cmd_render_sheets(int argc, char **argv);   /* src/render.c */
int cmd_overlay_red(int argc, char **argv);
int cmd_tifxyz_scale(int argc, char **argv);
int cmd_ink_compare(int argc, char **argv);
int cmd_jpeg_fit(int argc, char **argv);
int cmd_ink_ext(int argc, char **argv);   /* src/inkext.c */
int cmd_ink_ensemble(int argc, char **argv);
int cmd_wl_dump(int argc, char **argv);
int cmd_dinovol_embed(int argc, char **argv);   /* src/dinovol.c */
int cmd_hercuvit(int argc, char **argv);   /* src/hercuvit.c */
int cmd_villawind(int argc, char **argv);   /* src/villawind.c */
int cmd_vwt_infer(int argc, char **argv);   /* src/vwt_infer.c */
int cmd_wtrunk_bench(int argc, char **argv);   /* src/wtrunk.c */
int cmd_vwt_train(int argc, char **argv);   /* src/vwt_train.c */
int cmd_vwt_tile(int argc, char **argv);   /* src/vwt_tile.c */
int cmd_vwt_stitch(int argc, char **argv);   /* src/vwt_stitch.c */
int cmd_phase_vis(int argc, char **argv);   /* src/phasevis.c */
int cmd_ink3d_ext(int argc, char **argv);   /* src/ink3dext.c */
int cmd_vol_cmp(int argc, char **argv);
int cmd_dinovol_eval(int argc, char **argv);
int cmd_dinovol_read(int argc, char **argv);
int cmd_emb_pca(int argc, char **argv);
int cmd_emb_calib(int argc, char **argv);
int cmd_dinovol_calib(int argc, char **argv);
int cmd_dino_headfit(int argc, char **argv);
int cmd_surf_eval(int argc, char **argv);   /* src/surfeval.c */
int cmd_dino_seamprobe(int argc, char **argv);   /* src/dino.c */
int cmd_dino_train(int argc, char **argv);   /* src/dino.c */
int cmd_dino_embed(int argc, char **argv);
int cmd_dino_halves(int argc, char **argv);
int cmd_dino_boxes(int argc, char **argv);
int cmd_tifxyz_map(int argc, char **argv);
int cmd_tifxyz_orient(int argc, char **argv);
int cmd_tifxyz_bbox(int argc, char **argv);
int cmd_tifxyz_shards(int argc, char **argv);
int cmd_png2jpg(int argc, char **argv);
int cmd_tifxyz_sfc(int argc, char **argv);
int cmd_sheet_cut(int argc, char **argv);   /* src/sheetcut.c */
int cmd_phase_score(int argc, char **argv);
int cmd_ink_splat(int argc, char **argv);   /* src/inksplat.c */
int cmd_ink_score(int argc, char **argv);   /* src/inksplat.c */

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static int arg_index(int argc, char **argv, const char *k) { for (int i = 2; i < argc; i++) if (!strcmp(argv[i], k)) return i; return -1; }

static int usage(void) {
    fprintf(stderr,
        "ufsm — ultra fast scroll model\n"
        "  ufsm sysinfo [--ncpu | --avail-gb | --limit-gb] [--disk PATH]   real cgroup CPU / RAM limits, GPUs (MIG), CUDA, free disk\n"
        "  ufsm info    <root> <array-or-group-key> [--cache DIR]\n"
        "  ufsm read    <root> <array-key> z y x nz ny nx <out.raw> [--cache DIR] [--threads N]\n"
        "  ufsm slice   <root> <array-key> z <out.pgm> [--cache DIR] [--threads N]     (whole z slice)\n"
        "  ufsm regions <root> <regions-dir-key>                 -> JSON list of region origins\n"
        "  ufsm mirror  <root> <array-key> <out-dir> [--threads 4] [--mbps 0] [--link-from DIR] [--verify N]  local copy of one array's shard files\n"
        "  ufsm scanprep <root> <group-key> --level L --out PREFIX [...]   per-scan object mask + CT statistics (src/scanprep.h)\n"
        "  ufsm axis    <root> <ct-group-key> <out.json> [--cache DIR]  -> per-z centroid axis (umbilicus stand-in)\n"
        "  ufsm ingest-labels <root> <zarr-key> <out-dir> --um U   HF/S3 zarr v2 labels -> volcomp pyramid\n"
        "  ufsm z3-recode <group-dir> <out-group> --um U --lut FILE [--hist 1]   a local prob254 pyramid through a LUT (255 kept), value histogram\n"
        "  ufsm ink-blend <out> --um U --tile S,S,S --grids z,y,x:... <in1> <in2> ...   overlapping uint8 predicts (predict --grid-origin) blended with sin^2 tile weights\n"
        "  ufsm ink-seam <zarr> --um U [--period 512] [--offset 0] [--n 24]   window-seam jump of a uint8 predict (border vs interior |dv|)\n"
        "  ufsm ingest-zip <labels.zip> <zarr-name> <out-dir> --um U  label zarr from the HF archive -> volcomp pyramid\n"
        "  ufsm raster <out-dir> --shape Z,Y,X --um U [--level L] [--T 3] <mesh.sfc|tifxyz>...  meshes -> label pyramid\n"
        "  ufsm ingest-kaggle <hf-root> <out-dir> [--n N]          Kaggle cubes -> volcomp images + labels\n"
        "  ufsm ingest-mesh <root> <tifxyz-key> <out.sfc>           tifxyz -> surfcomp\n"
        "  ufsm train  <sources.json> --out DIR [--P 512 --B 1 --steps N --gpus 0 ...]   train the recto model\n"
        "  ufsm predict <ckpt> <root> <ct-group> <out-dir> --um U [--box ...]      sliding-window inference -> volcomp pyramid\n"
        "  ufsm eval <pred-root> <pred-group> <label-root> <label-group> --um U [--box ...]   precision/recall/dice\n"
        "  ufsm sample  <sources.json> [--P 128] [--B 2] [--n 4] [--seed S] [--workers W] [--out DIR] [--noaug] [--bench N (loader throughput, train-like formats)]\n"
        "  ufsm prefetch <sources.json> [--levels 1] [--threads 32] [--fraction 1] [--P 128] [--dry 1]   warm the CT chunk cache for the labelled cells\n"
        "  ufsm phase-raster <codes-root> <key> <out-dir> [--box ...]          band phase precomputed for the sampler\n"
        "  ufsm phase-check <phase-root> <key> --axis A --scan-um U [--count 1] a stored phase: share rising outward, labelled volume\n"
        "  ufsm downsample <root> <key> <out-group> <out-um> --mode max|phase|votes  one array halved per axis (labels, phase, side votes)\n"
        "  ufsm side-votes <out-group> <out-um> <root> <key> [<root> <key> ...]   several surface sets' side labels added into one vote array\n"
        "  ufsm wind-assign <ref-root> <ref-key> <out-dir> <tifxyz>...   other authors' surfaces get winding q in a reference raster's gauge\n"
        "  ufsm wind-seed <out-dir> <tifxyz>... [--k K]   a surface set's own winding gauge (K from the wNNN name, dir from the inner end)\n"
        "  ufsm phase-fuse <out-group> <out-um> <root> <key>...   surface sets' phases fused per voxel (circular mean within --tol, else unknown)\n"
        "  ufsm recto-fuse <out-group> <out-um> <ct-root> <ct-key> <root> <key>...   one recto line per block (the brightest-CT set)\n"
        "  ufsm codes-merge <out-group> <out-um> <root> <key> ...   winding code rasters united (same wrap within 1/4 turn, else 255)\n"
        "  ufsm restore-eval <ckpt> <clean.npy>... [--family pyramid]   task restore: PSNR on clean crops degraded by one recipe\n"
        "  ufsm restore-stats <root> <key> --um U --box ... [--degrade scanner ...]   task restore: calibration statistics of a CT box\n"
        "  ufsm ct-hist <out.txt> <root> <key> z,y,x,n [...]  CT histograms of boxes (targets of train --aug-hist, the hist op)\n"
        "  ufsm fit <fields-dir> [--slab Z] [--out MODEL] [--winding ZARR] [--codes ROOT KEY]   winding fitter (src/fit.h)\n"
        "  ufsm convert-unwrap2 <unwrap.ckpt> <out.ckpt>   a task unwrap checkpoint as task unwrap2 (heads 2-10 -> 0-8, winding mod 4 fresh; no training step)\n"
        "  ufsm repack <in-array-dir> <out-array-dir> <shard> [--threads N]   lossless re-sharding of a z3w uint8 zarr (copies the compressed chunks)\n"
        "  ufsm wraps <model-dir> <out-dir> [--step 8] [--fields DIR]          one tifxyz per wrap (level sets of W)\n"
        "  ufsm coreline detect|box|sdist|check ...   a crushed scroll's flattened umbilicus per z (fit fields), half-box fit dirs behind it, check renders\n"
        "  ufsm render-sheets <tifxyz-dir> <out-dir> --ct ROOT KEY --ink ROOT KEY ...   every sheet -> CT + red ink 5-panel JPEGs, one batch\n"
        "  ufsm overlay-red <ct-dir> <ink-dir> <out-dir> [ink2 [ink3]]   the 5-panel overlay of existing renders\n"
        "  ufsm tifxyz-scale <src> <dst> <div> [--offset z,y,x]   tifxyz coordinates / div (e.g. 2.4 -> 9.6 um: 4)\n"
        "  ufsm ink-compare <out.jpg> --ct CT.tif --ink LABEL=INK.tif ..   CT | ink panels (half-res JPEG) + ink statistics per render\n"
        "  ufsm jpeg-fit IN.jpg OUT.jpg --div 2 | --max-px N                  a finished JPEG shrunk (OUT may be IN)\n"
        "  ufsm tifxyz-sfc <src-dir> <dst-dir> [--error 0.25] [--every K]   tifxyz wraps -> surface-compressor .sfc (readers take .sfc wraps directly)\n"
        "  ufsm sheet-cut <wraps-dir> <out-dir> [--tol 0.3] [--min-cm2 0.25] [--max 8]   square flat sheets: largest isometric rectangles of each wrap\n"
        "  ufsm ink-ext --model v8in|hecate24|hecate96|reader3|ink9um --weights W --surface TIFXYZ --ct ROOT KEY --out DIR   external ink networks on a surface\n"
        "  ufsm ink-ensemble <out.tif> --ink [W:]INK.tif ..   per-model percentile-stretched weighted mean of ink renders\n"
        "  ufsm ink-splat <out-group> --surface TIFXYZ --labels L [--mask M] --ct ROOT KEY --ct-um U --iso I [--band -4,4] ..   published 2D ink labels -> sparse 3D label zarr\n"
        "  ufsm ink-score --surface TIFXYZ --labels L [--mask M] --map NAME=MAP.tif ..   F1 / AUC of 2D ink maps vs a segment's published labels\n"
        "  ufsm wl-dump <weights> [key]   the tensors of a safetensors / torch zip weight file\n"
        "  ufsm dinovol-embed ROOT KEY OUT --ckpt PT --box z,y,x,nz,ny,nx   scrollprize dinovol tokens / dense / ink embeddings (zarr v3)\n"
        "  ufsm hercuvit ROOT KEY OUT --ckpt PTH --box z,y,x,nz,ny,nx   HercuViT v0 sheet surface probability + per-window instance ids\n"
        "  ufsm villawind ray-check ROOT KEY WINDINGS.json --ckpt PTH   villa winding model (scrollprize/winding_model_9um) turn counts on human rays\n"
        "  ufsm ink3d-ext --weights W --ct ROOT KEY --box z,y,x,nz,ny,nx --out GROUP [--out-um 9.6,4.8]   published 3D ink net (gambiarra) on a 2.4 um CT box -> prob254 zarr\n"
        "  ufsm vol-cmp A_ROOT A_KEY B_ROOT B_KEY --box z,y,x,nz,ny,nx   two uint8 volumes (zarr v3 / v2): Pearson / Spearman + value mapping\n"
        "  ufsm dinovol-read GROUP NAME z,y,x,nz,ny,nx OUT.f32   a region of a dinovol embedding as float32\n"
        "  ufsm dinovol-eval EMB SURF.tifxyz ...   sheet-identity check of the dense embedding along surfaces\n"
        "  ufsm surf-eval --cases LIST --ckpt PT --out DIR   the upstream kaggle_surfaces frozen-encoder surface decoder evaluation\n"
        "  ufsm dinovol-calib --ckpt PT --sources S.json   the upstream teacher's DINO / iBOT head entropy and agreement on our crops\n"
        "  ufsm dino-train --sources S.json --out DIR   our dense embedding net from scratch (GT sheet identity + DINO-style SSL)\n"
        "  ufsm dino-embed CKPT ROOT KEY OUT --box ... --norm c,s   a dino-train net over a box (dinovol-eval format)\n"
        "  ufsm halves ROOT KEY-L0 OUT --box ...   two 4x-pooled views of a fine scan from disjoint sub-voxel halves\n"
        "  ufsm png2jpg <in.png|in.jpg> <out.jpg> [--half] [--max-px N]   PNG / JPEG -> JPEG q90 (half: Pillow BOX halved; max-px: longest side)\n"
        "  ufsm phase-score <pc-dir> <ps-dir> [<sheets-dir>]   on-recto score of pc / ps renders on surfaces (per seam class with sheets)\n"
        "  ufsm stitch align|solve|field|sheets|info ...   overlapping box fits -> one gauge, a global winding field, stitched sheets\n"
        "  ufsm render <wraps-dir> <out-dir> --volume ROOT KEY [--ct ROOT KEY]  ink / CT along each wrap (tiled TIFF)\n"
        "  ufsm flatten <wraps-dir> <out-dir> [--wraps a,b] [--decim 1] [--out-step S]   SLIM-flattened wraps (isometric uv, largest valid rectangle)\n"
        "  ufsm faces <wraps-dir> <out-dir> --ct ROOT KEY --ct-um U [--umb JSON --umb-um U]   recto (inner) / verso (outer) faces of the sheet under each wrap\n"
        "  ufsm flatten-warp <in.jpg|tif> <flat-wrap-dir> <out.jpg|tif> --src-px P [--scale S]   a render of the original wrap onto the flat one\n"
        "  ufsm strip <wraps-dir> <out-dir> [--first K] [--last K] [--every E]   the wraps joined at the theta cut: ONE continuous spiral surface (strip.sfc)\n"
        "  ufsm strip-flatten <strip-dir> [--decim D] [--stride S] [--stripe-px 60000] [--flip-rows 1]   the whole strip flattened (chunk SLIM + blend) -> render stripes\n"
        "  ufsm strip-mosaic <flat-dir> <render-dir> <out-dir> [--ct-level 80]   stripe JPEGs, overview, on-papyrus share along the strip\n"
        "  ufsm blend <out-group> <out-um> --shape Z,Y,X --interior S <root> <key> <oz,oy,ox>...   shifted predict runs -> one seamless sin^2-weighted logit average\n"
        "  ufsm wrapdist <wraps-dir> <labels-root> <labels-key>              wrap vertices vs the nearest labelled sheet\n"
        "  ufsm wrapswitch <wraps-dir> <codes-root> <codes-key>            how often the wraps switch sheets (per cm)\n"
        "  ufsm wrap-check <model> <fields-dir> [--ct ROOT KEY --ct-um U] [--codes ROOT KEY]   duplicate / merged wraps: CT peaks between adjacent wraps along radial rays\n"
        "  ufsm surf-snap <wraps-in> <wraps-out> --volume ROOT KEY --vol-um U   tifxyz vertices onto the nearest papyrus ridge\n"
        "  ufsm refine <wraps-in> <wraps-out> --fields DIR [--range 0.33] [--smooth-um 40]   vertices onto the fields' sheet phase\n"
        "  ufsm surf-stats <wraps-dir> --fields DIR [--ct ROOT KEY --ct-um U]   label-free surface quality (phase, switches, CT)\n"
        "  ufsm self-gate <wraps-dir> <out-dir> --fields DIR --model FIT [--ct ..] [--nb FIT].. [--exclude ..]   strict stretches of a box fit -> label tifxyz + q.tif\n"
        "  ufsm ct-fields <ct-root> <key> <fields-dir> --ct-um U   CT planes ct / cs of a fit-field directory (fit --w-ct / --w-cspacing)\n"
        "  ufsm fit-join <out-set> <fields-dir> <slab-model> <z0> <z1> ...   independent z-slab fits -> one fit set, gauges matched on the overlaps\n"
        "  ufsm fields-oracle <fields-dir> <phase-root> <phase-key> <out-dir>   fit fields with the phase planes from a label phase raster (oracle / swap)\n"
        "  ufsm count-field <in-fields> <out-fields> [--method ls|ray] [--R 3] [--model DIR]   winding from the density counts, the phase only sub-turn (src/countfield.c)\n"
        "  ufsm cf-compare <model-a> <model-b> [fields] [--step 4]   two count-field models: delta and snap-integer agreement (src/countfield.c)\n"
        "  ufsm vwt-sample SOURCES OUT N   ray-model training slabs + targets (im / va / ph / comp / pw raw) and the sampler throughput\n"
        "  ufsm ray-check WINDINGS.json FIELDS   turn counts of a field set between human relative-winding ray points (phase / density)\n"
        "  ufsm patch-raster LIST OUT --shape Z,Y,X   surface patches (tifxyz) as phase-0 shell training targets (512^3 shards)\n"
        "  ufsm phase-gate <in-fields> <out-fields> --ref MODEL [--mode gate|ref]   the network phase replaced by a ref model's where it does not rotate with the sheets\n"
        "  ufsm hires-phase <ct-root> <key> <fields-dir> <out-dir> --ct-um U   fit fields with the sheet phase measured in a finer CT (valleys along the normal)\n"
        "  ufsm phase-packing <phase-root> <phase-key> [--box ..] [--map OUT.raw]   how packed the labelled sheets are (box stats, per-chunk map for the sampler focus)\n"
        "  ufsm wbench-stats A.tsv [B.tsv] [--metric ids,p1x] [--set clean|all]   winding bench means, stratified bootstrap CIs, paired A-B, MDE (--scroll a,b; --tsv 1)\n"
        "  ufsm ckpt-avg OUT.ckpt IN1.ckpt IN2.ckpt ... [--arrays ema|both]   uniform weight average of snapshots of one run (optimiser state of the last)\n"
        "  ufsm phase-conflict <a-root> <a-key> <b-root> <b-key> <out>   where two label phase sets disagree (> --tol turn): the sampler's phase ignore mask\n"
        "  ufsm fit-compare <reference-fields> <fields>                      fit fields vs a reference (phase, spacing, normals)\n"
        "  ufsm fiber-check <afv> --tr T.json --boxes B.tsv --names A,B --codes R K --phase R K   do published fibre polylines link points of one GT sheet?\n"
        "  ufsm fiber-export <afv> <fields-dir> <out.fib> --tr T.json    fibre polylines in a fit box's coordinates for fit --fibers\n"
        "  ufsm field-blend <out> <fields1> <fields2> ... [--weights a,b,..] [--tile SZ,SY,SX --grids z,y,x:..] [--conf mean]  fit fields averaged (phase: confidence-weighted complex mean; --tile: overlapping predicts, centre-weighted)\n"
        "  ufsm wind-score <sweep-dir> <codes-root> <codes-key>             the fit --codes score of a predict --cond-sweep winding raster\n"
        "  ufsm store-bench <root> <array-key> [--box 128,256,512] [--n 8] [--par 1] [--threads 16]   HTTP zarr fetch benchmark\n"
        "  ufsm phase-pack ROOT KEY OUT --row R [--level 19] [--threads 8] / phase-unpack IN DIR / phase-cmp RA KA RB KB --row R   6-bit phase-label transfer\n"
        "  ufsm papyrus-labels <ct-root> <ct-key> <out-group> --box z,y,x,nz,ny,nx   papyrus / void training labels of a CT box (gap labels, no cutoff)\n"
        "  ufsm overlay <out.png> <ct-root> <ct-key> --plane z,y,x,h,w --labels G1,G2   a CT plane beside label panels (void blue, ignored yellow)\n"
        "root: local dir or https://.../volcomp ; keys are paths under root\n");
    return 2;
}

static const char *opt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return dflt;
}
static int flag(int argc, char **argv, const char *name) {
    for (int i = 1; i < argc; i++) if (!strcmp(argv[i], name)) return 1;
    return 0;
}

static int write_pgm(const char *path, const uint8_t *img, int64_t w, int64_t h) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    fprintf(f, "P5\n%lld %lld\n255\n", (long long)w, (long long)h);
    fwrite(img, 1, (size_t)(w * h), f);
    fclose(f);
    return 0;
}

/* ---- regions: list region_<z>_<y>_<x>.zarr under a directory (local readdir or nginx index html) ---- */
static int cmd_regions(store *s, const char *key) {
    int n = 0;
    printf("[");
    if (store_is_local(s)) {
        char p[1200];
        snprintf(p, sizeof p, "%s/%s", store_root(s), key);
        DIR *d = opendir(p);
        if (!d) { fprintf(stderr, "cannot open %s\n", p); return 1; }
        struct dirent *e;
        while ((e = readdir(d))) {
            long long z, y, x;
            if (sscanf(e->d_name, "region_%lld_%lld_%lld.zarr", &z, &y, &x) == 3 && !strstr(e->d_name, ".part"))
                printf("%s[%lld,%lld,%lld]", n++ ? ",\n " : "", z, y, x);
        }
        closedir(d);
    } else {
        size_t len;
        char k[1200];
        snprintf(k, sizeof k, "%s/", key);
        uint8_t *html = store_read_all(s, k, &len);
        if (!html) { fprintf(stderr, "cannot list %s\n", key); return 1; }
        for (char *p = (char *)html; (p = strstr(p, "href=\"region_")); p += 6) {
            long long z, y, x;
            char tail[16] = "";
            if (sscanf(p + 6, "region_%lld_%lld_%lld.zarr%15[^\"]", &z, &y, &x, tail) >= 3 && !strcmp(tail, "/"))
                printf("%s[%lld,%lld,%lld]", n++ ? ",\n " : "", z, y, x);
        }
        free(html);
    }
    printf("]\n");
    fprintf(stderr, "%d regions\n", n);
    return 0;
}

/* ---- axis: per-z centroid of the masked CT at a coarse level, written as umbilicus control points ----
   The coarsest EXISTING level whose slices are still >= 128 wide (levels may be missing: the u0 copies hold L and L+2..), read one
   inner-chunk row (128 z rows) at a time: memory ~ 128 x Y x X of that level (a full-resolution shard row was ~70 GB). */
static int cmd_axis(store *s, const char *key, const char *out, const char *cache) {
    z3 *z = nullptr;
    int level = -1;
    for (int l = 0; l < MAXLEV; l++) {   /* numeric level paths, factor 2 each; absent ones skipped */
        char ak[1200];
        snprintf(ak, sizeof ak, "%s/%d", key, l);
        z3 *t = z3_open(s, ak, cache);
        if (!t) continue;
        if (z && z3_meta_of(t)->shape[1] < 128) { z3_close(t); break; }
        z3_close(z);
        z = t;
        level = l;
    }
    if (!z) { fprintf(stderr, "axis: no readable level under %s\n", key); return 1; }
    const z3_meta *m = z3_meta_of(z);
    int64_t Z = m->shape[0], Y = m->shape[1], X = m->shape[2];
    const int64_t R = m->chunk[0] > 0 ? m->chunk[0] : 128;
    fprintf(stderr, "axis: level %d, %lld x %lld x %lld, %lld rows per read\n", level, (long long)Z, (long long)Y, (long long)X, (long long)R);
    uint8_t *slab = malloc((size_t)R * Y * X);
    if (!slab) { fprintf(stderr, "axis: no memory for %lld rows\n", (long long)R); return 1; }
    FILE *f = fopen(out, "w");
    if (!f) { fprintf(stderr, "cannot write %s\n", out); return 1; }
    fprintf(f, "{\"control_points\":[");
    int n = 0;
    double scale = (double)(1 << level);
    for (int64_t z0 = 0; z0 < Z; z0 += R) {
        int64_t nz = z0 + R > Z ? Z - z0 : R;
        int64_t o[3] = {z0, 0, 0}, cnt[3] = {nz, Y, X};
        if (z3_read(z, o, cnt, slab, 0)) { fprintf(stderr, "axis: %s\n", z3_error()); return 1; }
        for (int64_t zi = 0; zi < nz; zi += 4) {
            const uint8_t *sl = slab + (size_t)zi * Y * X;
            double sy = 0, sx = 0; int64_t k = 0;
            for (int64_t y = 0; y < Y; y++)
                for (int64_t x = 0; x < X; x++)
                    if (sl[y * X + x]) { sy += (double)y; sx += (double)x; k++; }
            if (k < Y * X / 100) continue;
            fprintf(f, "%s{\"z\":%.1f,\"y\":%.1f,\"x\":%.1f,\"score\":%lld}", n++ ? "," : "", (double)(z0 + zi) * scale,
                    sy / (double)k * scale, sx / (double)k * scale, (long long)k);
        }
    }
    fprintf(f, "],\"method\":\"ufsm axis: per-z centroid of CT>0 at level %d\"}\n", level);
    fclose(f);
    free(slab);
    z3_close(z);
    fprintf(stderr, "axis: %d control points -> %s\n", n, out);
    return 0;
}

/* ---- sample: dump montages of drawn patches ---- */
/* the winding fitter's input fields (src/fit.h) from the label-grid phase and spacing of a native box b (oracle fields): per field
   voxel of h native voxels (h / 2 label voxels per axis) the mean unit phasor of the known phase (pc, ps: 128 + 127 * mean),
   the mean n n^T of the wrap-aware phase gradient (central differences over one label voxel), the mean spacing 1 / |grad phase|
   (units of h / 8: 2 * label spacing * 8 / h) and the known fraction (w); fields.json records the box, h and the axis table. */
static int band_fit_fields(const uint8_t *ph, const uint8_t *spc, const int n[3], int halo, const long long b[6], int h, const axis *ax, const char *dir) {
    const int bl = h / 2, D = (int)(b[3] / h), H = (int)(b[4] / h), W = (int)(b[5] / h);
    const size_t nf = (size_t)D * H * W, pz = (size_t)n[1] * n[2];
    uint8_t *a[10]; for (int k = 0; k < 10; k++) a[k] = calloc(nf, 1);
    for (int fz = 0; fz < D; fz++) for (int fy = 0; fy < H; fy++) for (int fx = 0; fx < W; fx++) {
        double sc = 0, ss = 0, t[6] = {0}, sp = 0; int kn = 0, kt = 0, ksp = 0;
        for (int dz = 0; dz < bl; dz++) for (int dy = 0; dy < bl; dy++) for (int dx = 0; dx < bl; dx++) {
            const int z = halo + fz * bl + dz, y = halo + fy * bl + dy, x = halo + fx * bl + dx;
            const size_t i = (size_t)z * pz + (size_t)y * n[2] + x;
            if (ph[i] >= PHASE_PERIOD) continue;
            const double q = 2 * M_PI * ph[i] / PHASE_PERIOD; sc += cos(q); ss += sin(q); kn++;
            const uint8_t nb[6] = {ph[i + pz], ph[i - pz], ph[i + n[2]], ph[i - n[2]], ph[i + 1], ph[i - 1]};
            int ok = 1; for (int k = 0; k < 6; k++) ok &= nb[k] < PHASE_PERIOD;
            if (!ok) continue;
            double g[3];
            for (int d = 0; d < 3; d++) { int df = (int)nb[2 * d] - (int)nb[2 * d + 1]; df -= PHASE_PERIOD * (df > PHASE_PERIOD / 2); df += PHASE_PERIOD * (df <= -PHASE_PERIOD / 2); g[d] = df; }
            const double gn = sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
            if (gn < 1e-9) continue;
            sp += 2.0 * PHASE_PERIOD / gn; ksp++;   /* spacing (label voxels per turn) from the gradient: 2 * 252 / |central difference| */
            for (int d = 0; d < 3; d++) g[d] /= gn;
            t[0] += g[0] * g[0]; t[1] += g[1] * g[1]; t[2] += g[2] * g[2]; t[3] += g[0] * g[1]; t[4] += g[0] * g[2]; t[5] += g[1] * g[2]; kt++;
        }
        const size_t fi = ((size_t)fz * H + fy) * W + fx;
        if (!kn) { a[0][fi] = a[1][fi] = 128; continue; }
        a[0][fi] = (uint8_t)lround(128 + 127 * sc / kn); a[1][fi] = (uint8_t)lround(128 + 127 * ss / kn);
        if (kt) for (int k = 0; k < 6; k++) a[2 + k][fi] = (uint8_t)lround(255 * fmin(1, fmax(0, (k < 3 ? 0 : 0.5) + t[k] / kt)));
        if (ksp) { const double v = 2.0 * sp / ksp * 8 / h; a[8][fi] = (uint8_t)(v < 1 ? 1 : v > 255 ? 255 : lround(v)); }
        a[9][fi] = (uint8_t)lround(255.0 * kn / ((double)bl * bl * bl));
    }
    const int na = D * h + 1; float *cy = malloc(na * sizeof(float)), *cx = malloc(na * sizeof(float));
    for (int z = 0; z < na; z++) { double y, x; axis_at(ax, (double)(b[0] + z), &y, &x); cy[z] = (float)(y - b[1]); cx[z] = (float)(x - b[2]); }
    const long long org[3] = {b[0], b[1], b[2]}; const int shp[3] = {D, H, W};
    const int rc = fit_fields_save(dir, org, shp, h, 2.4, a, cy, cx, na);   /* the band field: 2.4 um native (4.8 um labels) */
    for (int k = 0; k < 10; k++) free(a[k]);
    free(cy); free(cx);
    return rc;
}

/* ufsm phase-raster <codes-root> <key> <out-dir> [--axis A] [--radius 80] [--span 75] [--box z,y,x,nz,ny,nx] [--threads 4] [--block 256] [--shards 1]
   [--pair 1|2]
   --pair 2: recto / verso side labels instead (band_field_side_pair: 0 unknown, 1 verso, 2 recto; between consecutive wraps k and
   k + 1 the half nearer k recto; halo 2 radius), fill 0, for train sources with "encoding": "side".
   The band field's winding phase (src/band.h PHASE_*, what the sampler computes per window for --side 2 / task unwrap) of a whole
   winding_mod14 raster, on its label grid: written as <out-dir>/4.8 (uint8, lossless volcomp, fill PHASE_UNKNOWN) plus the group;
   a source then reads it with "band": {..., "phase": {"root": <out-dir>, "key": "4.8"}}. Blocks of --block label voxels plus the
   band halo, in parallel, shard by shard (512 label voxels); blocks without codes stay unknown. --box (native) limits it.
   --pool 2 writes only the 2x circular-mean pooling, <out-dir>/9.6 (a whole scroll at 4.8 um is ~25 MB per shard, 9.6 um
   ~5 MB); --resume 1 skips shards already written (long runs). --winding 1 writes the winding mod 14 turns (band_winding, same
   encoding and pooling: 1/18 turn steps) instead of the 1-turn phase, for targets beyond one turn. --gs 4 computes the field on the
   codes pooled 2x (praster_read: circular mean, conflicts kept) into <out-dir>/9.6: 35x faster (block B 7.5 s vs 4.4 min); against the
   4.8 um phase pooled to 9.6 on B: mean 0.016 turn apart, 1% over 0.1 turn. A source reads it for 2.4 um windows with "phase":
   {"key": "9.6", "level": 2} (sample.c phase_window_interp: circular trilinear 4x).
   --codes-grid 1 with --gs 4: the codes were rastered on the output grid itself (raster --level 2: 9.6 um on a 2.4 um scan), read
   as they are (no 2x pooling): 8x fewer code bytes (Paris 4 June: ~11 GB instead of 43). --mask-ref ROOT KEY [--mask-tol 0.1]:
   a phase on the output grid (another unroll of the same scroll): where it is known and the computed phase differs by more than
   --mask-tol turns, the output is unknown (two separate sources, their conflicts masked, nothing fused); the |difference|
   histogram is reported. The known voxels written are counted (labelled volume, mm^3). */
typedef struct { z3 *codes; const int64_t *shape; const axis *ax; band_params bp; int halo, blk; int64_t s0[3]; uint8_t *shard; atomic_int next; int nblk[3]; size_t nonempty; pthread_mutex_t mu; int winding; int pf; const int64_t *fshape; int pair; } praster_job;   /* pf 2: the grid is the 4.8 um codes (fshape) pooled 2x */
/* the codes over the grid box ro / rn: read directly (pf 1), or from the 2x finer codes with every 2^3 block pooled (pf 2): empty when
   all are empty, 255 (conflict) when any is or two lie more than a quarter turn apart, else their circular mean */
static int praster_read(const praster_job *J, const int64_t ro[3], const int64_t rn[3], uint8_t *out) {
    if (J->pf == 1) return z3_read(J->codes, ro, rn, out, 1);
    static float cs[BAND_PERIOD], sn[BAND_PERIOD]; static _Atomic int init;
    if (!atomic_load(&init)) { for (int k = 0; k < BAND_PERIOD; k++) { cs[k] = (float)cos(2 * M_PI * k / BAND_PERIOD); sn[k] = (float)sin(2 * M_PI * k / BAND_PERIOD); } atomic_store(&init, 1); }
    int64_t fo[3], fn[3];
    for (int d = 0; d < 3; d++) { fo[d] = 2 * ro[d]; const int64_t e = 2 * (ro[d] + rn[d]) < J->fshape[d] ? 2 * (ro[d] + rn[d]) : J->fshape[d]; fn[d] = e - fo[d]; }
    memset(out, 0, (size_t)rn[0] * rn[1] * rn[2]);
    if (fn[0] <= 0 || fn[1] <= 0 || fn[2] <= 0) return 0;
    uint8_t *f = malloc((size_t)fn[0] * fn[1] * fn[2]);
    if (!f || z3_read(J->codes, fo, fn, f, 1)) { free(f); return -1; }
    for (int64_t a = 0; a < rn[0]; a++) for (int64_t b = 0; b < rn[1]; b++) for (int64_t c = 0; c < rn[2]; c++) {
        int first = -1, conflict = 0, k = 0; float x = 0, y = 0;
        for (int q = 0; q < 8 && !conflict; q++) {
            const int64_t z = 2 * a + (q >> 2 & 1), yy = 2 * b + (q >> 1 & 1), xx = 2 * c + (q & 1);
            if (z >= fn[0] || yy >= fn[1] || xx >= fn[2]) continue;
            const int v = f[((size_t)z * fn[1] + yy) * fn[2] + xx];
            if (!v) continue;
            if (v == 255 || (first >= 0 && 2 * abs(band_diff(v - 1, first)) > BAND_STEPS / 2)) { conflict = 1; break; }
            if (first < 0) first = v - 1;
            x += cs[v - 1]; y += sn[v - 1]; k++;
        }
        uint8_t o = 0;
        if (conflict) o = 255;
        else if (k) { double t = atan2(y, x) / (2 * M_PI); if (t < 0) t += 1; o = (uint8_t)(1 + (int)lround(t * BAND_PERIOD) % BAND_PERIOD); }
        out[((size_t)a * rn[1] + b) * rn[2] + c] = o;
    }
    free(f);
    return 0;
}
static void *praster_worker(void *arg) {
    praster_job *J = arg; const int B = J->blk, Hh = J->halo, nb = J->nblk[0] * J->nblk[1] * J->nblk[2];
    for (;;) {
        const int t = atomic_fetch_add(&J->next, 1); if (t >= nb) break;
        const int bz = t / (J->nblk[1] * J->nblk[2]), by = (t / J->nblk[2]) % J->nblk[1], bx = t % J->nblk[2];
        int64_t o[3], n[3], ro[3], rn[3]; int ni[3]; int ok = 1;
        const int64_t b0[3] = {J->s0[0] + (int64_t)bz * B, J->s0[1] + (int64_t)by * B, J->s0[2] + (int64_t)bx * B};
        for (int d = 0; d < 3; d++) {
            if (b0[d] >= J->shape[d]) ok = 0;
            o[d] = b0[d] - Hh; n[d] = B + 2 * Hh; ni[d] = (int)n[d];
            ro[d] = o[d] < 0 ? 0 : o[d]; const int64_t e = o[d] + n[d] < J->shape[d] ? o[d] + n[d] : J->shape[d]; rn[d] = e - ro[d];
            if (rn[d] <= 0) ok = 0;
        }
        if (!ok) continue;
        const size_t N = (size_t)n[0] * n[1] * n[2];
        uint8_t *codes = calloc(N, 1), *tmp = malloc((size_t)rn[0] * rn[1] * rn[2]);
        if (!codes || !tmp || praster_read(J, ro, rn, tmp)) { free(codes); free(tmp); fprintf(stderr, "phase-raster: read failed: %s\n", z3_error()); continue; }
        size_t nz = 0; for (size_t k = 0; k < (size_t)rn[0] * rn[1] * rn[2]; k++) nz += tmp[k] != 0;
        if (!nz) { free(codes); free(tmp); continue; }
        for (int64_t zz = 0; zz < rn[0]; zz++) for (int64_t yy = 0; yy < rn[1]; yy++)
            memcpy(codes + ((size_t)(zz + ro[0] - o[0]) * n[1] + (yy + ro[1] - o[1])) * n[2] + (ro[2] - o[2]), tmp + ((size_t)zz * rn[1] + yy) * rn[2], (size_t)rn[2]);
        free(tmp);
        double *cy = malloc(n[0] * sizeof(double)), *cx = malloc(n[0] * sizeof(double));
        const double g = band_gs(J->bp); for (int64_t zz = 0; zz < n[0]; zz++) axis_at(J->ax, g * (zz + o[0]) + 0.5 * (g - 1), &cy[zz], &cx[zz]);
        uint8_t *band = malloc(N), *phase = malloc(N);
        if (!band || !phase || (J->pair == 2 ? band_field_side_pair(codes, ni, J->bp.radius / band_gs(J->bp), Hh, phase) : J->pair ? band_field_phase_pair(codes, ni, J->bp.radius / band_gs(J->bp), phase, nullptr) : band_field_phase(codes, ni, o, cy, cx, J->bp, band, phase, nullptr))) { fprintf(stderr, "phase-raster: out of memory\n"); free(codes); free(band); free(phase); free(cy); free(cx); continue; }
        if (J->winding) for (size_t k = 0; k < N; k++) phase[k] = band_winding(band[k], phase[k]);   /* --winding: mod 14 turns */
        for (int zz = 0; zz < B; zz++) for (int yy = 0; yy < B; yy++)   /* the block interior into the shard buffer (512^3) */
            memcpy(J->shard + ((size_t)(bz * B + zz) * 512 + by * B + yy) * 512 + bx * B, phase + ((size_t)(zz + Hh) * n[1] + yy + Hh) * n[2] + Hh, (size_t)B);
        pthread_mutex_lock(&J->mu); J->nonempty++; pthread_mutex_unlock(&J->mu);
        free(codes); free(band); free(phase); free(cy); free(cx);
    }
    return nullptr;
}
/* the 2x-pooled band phase of a 512^3 label-grid shard into 256^3: per 2^3 block the circular mean of the known codes,
   PHASE_UNKNOWN where none is known or their resultant is under half their count (as ufsm downsample --mode phase) */
static void phase_pool2(const uint8_t *src, uint8_t *dst) {
    static float cs[PHASE_PERIOD], sn[PHASE_PERIOD]; static int init;
    if (!init) { for (int k = 0; k < PHASE_PERIOD; k++) { cs[k] = (float)cos(2 * M_PI * k / PHASE_PERIOD); sn[k] = (float)sin(2 * M_PI * k / PHASE_PERIOD); } init = 1; }
    for (int z = 0; z < 256; z++) for (int y = 0; y < 256; y++) for (int x = 0; x < 256; x++) {
        float c = 0, sv = 0; int known = 0;
        for (int k = 0; k < 8; k++) {
            const int v = src[((size_t)(2 * z + (k >> 2 & 1)) * 512 + 2 * y + (k >> 1 & 1)) * 512 + 2 * x + (k & 1)];
            if (v < PHASE_PERIOD) { c += cs[v]; sv += sn[v]; known++; }
        }
        uint8_t o = PHASE_UNKNOWN;
        if (known && c * c + sv * sv >= 0.25f * known * known) { double t = atan2(sv, c) / (2 * M_PI); if (t < 0) t += 1; o = (uint8_t)((int)lround(t * PHASE_PERIOD) % PHASE_PERIOD); }
        dst[((size_t)z * 256 + y) * 256 + x] = o;
    }
}
/* phase-raster: shards taken from list in turn, each by nt block threads (praster_worker) into its own buffer, then (pooled) written */
typedef struct { z3 *z; const int64_t *shape; const axis *ax; band_params bp; int halo, blk, winding, pf, pair; const int64_t *fshape; z3w *w; int pool, nt;
                 const int64_t (*list)[3]; size_t nlist; atomic_size_t next; size_t nonempty, done; double t0; pthread_mutex_t mu; atomic_int fail;
                 z3 *mref; double mtol; size_t mhist[PHASE_AGREE_BINS], mmasked, mboth, known; } praster_run;   /* --mask-ref: a phase on the output grid; known: voxels written known */
static void *praster_shards(void *arg) {
    praster_run *R = arg;
    uint8_t *shard = malloc((size_t)512 * 512 * 512), *pooled = R->pool == 2 ? malloc((size_t)256 * 256 * 256) : nullptr, *refbuf = nullptr;
    if (!shard || (R->pool == 2 && !pooled)) { atomic_store(&R->fail, 1); free(shard); free(pooled); return nullptr; }
    for (;;) {
        const size_t t = atomic_fetch_add(&R->next, 1); if (t >= R->nlist || atomic_load(&R->fail)) break;
        const int64_t sz = R->list[t][0], sy = R->list[t][1], sx = R->list[t][2];
        memset(shard, R->pair == 2 ? RV_UNKNOWN : PHASE_UNKNOWN, (size_t)512 * 512 * 512);
        praster_job J = {R->z, R->shape, R->ax, R->bp, R->halo, R->blk, {sz * 512, sy * 512, sx * 512}, shard, 0, {512 / R->blk, 512 / R->blk, 512 / R->blk}, 0,
                         PTHREAD_MUTEX_INITIALIZER, R->winding, R->pf, R->fshape, R->pair};
        pthread_t th[64];
        for (int k = 0; k < R->nt; k++) pthread_create(&th[k], nullptr, praster_worker, &J);
        for (int k = 0; k < R->nt; k++) pthread_join(th[k], nullptr);
        if (J.nonempty && pooled) phase_pool2(shard, pooled);
        uint8_t *outp = pooled ? pooled : shard; const int64_t S = 512 / R->pool; const size_t S3 = (size_t)S * S * S;
        size_t mh[PHASE_AGREE_BINS] = {0}, mm = 0, mb = 0, kn = 0;
        if (J.nonempty && R->mref && R->pair != 2) {   /* --mask-ref: voxels the reference phase knows and disagrees with by > tol: unknown */
            if (!refbuf && !(refbuf = malloc(S3))) { atomic_store(&R->fail, 1); break; }
            const int64_t o[3] = {sz * S, sy * S, sx * S}, n[3] = {S, S, S};
            if (z3_read(R->mref, o, n, refbuf, 1)) { fprintf(stderr, "phase-raster: --mask-ref read failed: %s\n", z3_error()); atomic_store(&R->fail, 1); break; }
            mb = phase_mask_ref(outp, refbuf, S3, R->mtol, mh, &mm);
        }
        if (J.nonempty && R->pair != 2) for (size_t k = 0; k < S3; k++) kn += outp[k] < PHASE_PERIOD;
        if (J.nonempty && z3w_write_shard(R->w, sz, sy, sx, outp, R->nt)) { fprintf(stderr, "%s\n", z3w_error()); atomic_store(&R->fail, 1); break; }
        pthread_mutex_lock(&R->mu); R->nonempty += J.nonempty; R->done++; R->known += kn; R->mmasked += mm; R->mboth += mb; for (int b = 0; b < PHASE_AGREE_BINS; b++) R->mhist[b] += mh[b];
        fprintf(stderr, "\rshard %lld,%lld,%lld (%zu of %zu): %zu blocks with codes so far, %.0fs   ", (long long)sz, (long long)sy, (long long)sx, R->done, R->nlist, R->nonempty, now() - R->t0);
        pthread_mutex_unlock(&R->mu);
    }
    free(shard); free(pooled); free(refbuf); return nullptr;
}
static int cmd_phase_raster(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: ufsm phase-raster <codes-root> <key> <out-dir> [--axis A] [--radius 80] [--span 75] [--box z,y,x,nz,ny,nx] [--threads 4] [--block 256] [--pool 2] [--resume 1] [--winding 1] [--gs 4] [--codes-grid 1] [--mask-ref ROOT KEY] [--mask-tol 0.1]\n"); return 2; }
    const int pool = atoi(opt(argc, argv, "--pool", "1")), resume = atoi(opt(argc, argv, "--resume", "0"));   /* --pool 2: write only the 9.6 um (2x circular-mean) phase; --resume: skip written shards */
    const int winding = atoi(opt(argc, argv, "--winding", "0"));   /* the winding mod 14 turns (band_winding) instead of the 1-turn phase */
    if (pool != 1 && pool != 2) { fprintf(stderr, "phase-raster: --pool 1|2\n"); return 2; }
    const int codes_grid = atoi(opt(argc, argv, "--codes-grid", "0"));   /* 1: the codes are already on the output grid (raster --level 2 for --gs 4: no pooling) */
    const int gs = atoi(opt(argc, argv, "--gs", "2")), pf = codes_grid ? 1 : gs / 2;   /* native voxels per output grid voxel: 2 the codes' own 4.8 um grid, 4 the codes pooled 2x (9.6 um) */
    if (gs != 2 && gs != 4) { fprintf(stderr, "phase-raster: --gs 2|4\n"); return 2; }
    if (gs == 4 && pool != 1) { fprintf(stderr, "phase-raster: --gs 4 writes the 9.6 um grid itself (no --pool)\n"); return 2; }
    band_params bp = {(float)atof(opt(argc, argv, "--radius", "80")), (float)atof(opt(argc, argv, "--span", "75")), (float)gs};
    axis ax = {0};
    if (axis_load(&ax, opt(argc, argv, "--axis", "/vesuvius/ufsm/umbilicus/PHercParis4/umbilicus-full-resolution.json"))) { fprintf(stderr, "phase-raster: cannot load axis\n"); return 1; }
    z3 *z = z3_open(store_open(argv[2]), argv[3], nullptr); if (!z) { fprintf(stderr, "%s\n", z3_error()); return 1; }
    const z3_meta *m = z3_meta_of(z); const int64_t fshape[3] = {m->shape[0], m->shape[1], m->shape[2]};
    const int64_t shape[3] = {(fshape[0] + pf - 1) / pf, (fshape[1] + pf - 1) / pf, (fshape[2] + pf - 1) / pf};   /* the output grid */
    const int blk = atoi(opt(argc, argv, "--block", "256")), nth = ufsm_threads(atoi(opt(argc, argv, "--threads", "0")), 4);
    if (512 % blk) { fprintf(stderr, "phase-raster: --block must divide 512\n"); return 2; }
    int64_t lo[3] = {0, 0, 0}, hi[3] = {shape[0], shape[1], shape[2]};
    const char *bx = opt(argc, argv, "--box", nullptr);
    if (bx) { long long v[6]; if (sscanf(bx, "%lld,%lld,%lld,%lld,%lld,%lld", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5]) != 6) return usage(); for (int d = 0; d < 3; d++) { lo[d] = v[d] / gs; hi[d] = (v[d] + v[3 + d] + gs - 1) / gs; } }
    const double um = 2.4 * gs * pool; char lv[32]; z3w_level_name(um, lv, sizeof lv);
    char dir[1400]; snprintf(dir, sizeof dir, "%s/%s", argv[4], lv);
    const int pair_attr = atoi(opt(argc, argv, "--pair", "0"));
    if (pair_attr == 2 && (pool != 1 || winding)) { fprintf(stderr, "phase-raster: --pair 2 writes side labels on the computed grid (no --pool 2 / --winding)\n"); return 2; }
    char attrs[700]; snprintf(attrs, sizeof attrs, "{\"ufsm\":{\"content\":\"%s%s%s\",\"codes\":\"%s/%s\",\"radius\":%g,\"span\":%g}}",
                                   pair_attr == 2 ? "recto / verso side (src/band.h band_field_side_pair: 0 unknown, 1 verso, 2 recto; wrap k .. k + 0.5 recto, k + 0.5 .. k + 1 verso)"
                                   : winding ? "band field winding mod 14 turns x 18 (src/band.h band_winding: 0..251, 255 unknown)" : "band field winding phase (src/band.h PHASE_*: 0..251, 255 unknown)",
                                   pool == 2 ? ", pooled 2x (circular mean of the known codes)" : gs == 4 ? ", computed on the codes pooled 2x (9.6 um grid)" : "",
                                   pair_attr == 1 ? ", pair rule (band_field_phase_pair: the nearest sheets of opposite winding parity)" : "", argv[2], argv[3], bp.radius, bp.span);
    const int64_t oshape[3] = {(shape[0] + pool - 1) / pool, (shape[1] + pool - 1) / pool, (shape[2] + pool - 1) / pool};
    z3w *w = z3w_create(dir, oshape, 512 / pool, 0, pair_attr == 2 ? RV_UNKNOWN : PHASE_UNKNOWN, attrs);   /* a 4.8 shard of 512 -> one output shard */
    if (!w) { fprintf(stderr, "%s\n", z3w_error()); return 1; }
    if (z3w_write_group(argv[4], &um, 1, "ufsm-phase", attrs)) { fprintf(stderr, "%s\n", z3w_error()); return 1; }
    /* the shards to compute (--resume skips written ones), --shards S of them at a time, each over --threads block threads: within one
       512^3 shard at most 8 blocks of 256 run, so one shard at a time left most cores idle (Paris4 at 9.6 um: 54 min on the desk) */
    const int nsh = atoi(opt(argc, argv, "--shards", "1")) < 1 ? 1 : atoi(opt(argc, argv, "--shards", "1")) > 64 ? 64 : atoi(opt(argc, argv, "--shards", "1"));
    size_t skipped = 0, nlist = 0, cap = 0; int64_t (*list)[3] = nullptr;
    for (int64_t sz = lo[0] / 512; sz * 512 < hi[0]; sz++) for (int64_t sy = lo[1] / 512; sy * 512 < hi[1]; sy++) for (int64_t sx = lo[2] / 512; sx * 512 < hi[2]; sx++) {
        if (resume) { char sp[1600]; snprintf(sp, sizeof sp, "%s/c/%lld/%lld/%lld", dir, (long long)sz, (long long)sy, (long long)sx); if (access(sp, F_OK) == 0) { skipped++; continue; } }
        if (nlist == cap) { cap = cap ? 2 * cap : 256; list = realloc(list, cap * sizeof *list); }
        list[nlist][0] = sz; list[nlist][1] = sy; list[nlist][2] = sx; nlist++;
    }
    /* --pair 1: band_field_phase_pair (the two nearest sheets of opposite winding parity; no radial inside / outside, no span) */
    const int pair = atoi(opt(argc, argv, "--pair", "0"));
    if (pair && winding) { fprintf(stderr, "phase-raster: --pair writes the 1-turn phase (no --winding)\n"); return 2; }
    z3 *mref = nullptr; const double mtol = atof(opt(argc, argv, "--mask-tol", "0.1"));
    { const int mi = arg_index(argc, argv, "--mask-ref");
      if (mi > 0) {
          if (mi + 2 >= argc || pool != 1) { fprintf(stderr, "phase-raster: --mask-ref ROOT KEY (a phase on the output grid; no --pool 2)\n"); return 2; }
          mref = z3_open(store_open(argv[mi + 1]), argv[mi + 2], nullptr);
          if (!mref || memcmp(z3_meta_of(mref)->shape, shape, sizeof shape)) { fprintf(stderr, "phase-raster: --mask-ref %s %s: cannot open or its shape differs from the output grid\n", argv[mi + 1], argv[mi + 2]); return 1; }
      } }
    praster_run R = {.mref = mref, .mtol = mtol, .z = z, .shape = shape, .ax = &ax, .bp = bp, .halo = (int)ceilf((pair == 2 ? 2 * bp.radius : (pair ? 0 : bp.span) + bp.radius) / (float)gs) + 2, .blk = blk, .winding = winding, .pf = pf, .pair = pair,
                     .fshape = fshape, .w = w, .pool = pool, .nt = nth < 1 ? 1 : nth > 64 ? 64 : nth, .list = list, .nlist = nlist, .t0 = now(), .mu = PTHREAD_MUTEX_INITIALIZER};
    pthread_t sth[64];
    for (int k = 1; k < nsh; k++) pthread_create(&sth[k], nullptr, praster_shards, &R);
    praster_shards(&R);
    for (int k = 1; k < nsh; k++) pthread_join(sth[k], nullptr);
    free(list);
    if (R.fail) return 1;
    const size_t nonempty = R.nonempty, nshards = nlist;
    if (z3w_close(w)) { fprintf(stderr, "%s\n", z3w_error()); return 1; }
    fprintf(stderr, "\nphase-raster: %zu shards (%zu already written), %zu blocks with codes, %.0fs -> %s\n", nshards, skipped, nonempty, now() - R.t0, dir);
    if (pair_attr != 2) { const double vu = atof(opt(argc, argv, "--vol-um", "2.4")) * gs * pool;   /* (the key names the grid as if the scan were 2.4 um) */
        fprintf(stderr, "phase-raster: known %zu voxels written (%.1f mm^3 at %g um voxels, --vol-um; this run's shards only)\n", R.known, (double)R.known * pow(vu * 1e-3, 3), vu); }
    if (mref) {
        const double b = R.mboth ? (double)R.mboth : 1;
        fprintf(stderr, "phase-raster: --mask-ref: %zu voxels known in both, |difference| < 0.05 turn %.3f, 0.05-0.1 %.3f, 0.1-0.2 %.3f, 0.2-0.4 %.3f, >= 0.4 %.3f; %zu (%.3f) over %.2f turn set unknown\n",
                R.mboth, R.mhist[0] / b, R.mhist[1] / b, R.mhist[2] / b, R.mhist[3] / b, R.mhist[4] / b, R.mmasked, R.mmasked / b, mtol);
        z3_close(mref);
    }
    z3_close(z);
    return 0;
}

/* ufsm phase-check <phase-root> <key> --axis A --scan-um U [--phase-um K] [--boxes 48] [--min 0.9] [--count 1] [--threads 8] [--sign 1]
   [--min-known 0.05] [--box z,y,x,nz,ny,nx (scan voxels)]
   The guard of a stored band phase (src/phase_check.h; 2026-10-19: the multi-scroll labels of Paris 4 June / July and PHerc0500P2
   rose inward and trained that way): over --boxes of its stored 128^3 chunks (with --count: the ones at least --min-known known,
   spread over the array; sparse labels: any with >= 2000), the share of voxels whose phase gradient points away from the axis (phase_outward_count). --count 1 also
   decodes every stored chunk and counts the known voxels (the labelled volume, mm^3). The axis is in scan voxels (--scan-um), the
   phase grid --phase-um (default: the key). --sign -1 reads it negated (a sources "phase_sign": -1). --box: only chunks meeting a box
   of scan voxels. One line on stdout: "phase-check <root>/<key> outward F boxes B pairs N ... known V voxels M mm3 chunks C densest
   z,y,x pass|FAIL" (densest: the centre of the chunk with the most known voxels, scan voxels, with --count).
   Exit 0 when F >= --min, 3 below it, 4 without gradient data. */
typedef struct { z3 *z; const int64_t *list; size_t n; atomic_size_t next; uint32_t *cnt; atomic_int fail; } pcheck_job;
static void *pcheck_count(void *arg) {
    pcheck_job *J = arg; const z3_meta *m = z3_meta_of(J->z); const int c0 = m->chunk[0], c1 = m->chunk[1], c2 = m->chunk[2];
    int64_t g[3] = {(m->shape[0] + c0 - 1) / c0, (m->shape[1] + c1 - 1) / c1, (m->shape[2] + c2 - 1) / c2};
    uint8_t *b = malloc((size_t)c0 * c1 * c2); if (!b) { atomic_store(&J->fail, 1); return nullptr; }
    for (;;) {
        const size_t t = atomic_fetch_add(&J->next, 1); if (t >= J->n) break;
        const int64_t k = J->list[t], cz = k / (g[1] * g[2]), cy = (k / g[2]) % g[1], cx = k % g[2];
        const int64_t o[3] = {cz * c0, cy * c1, cx * c2}, n[3] = {c0, c1, c2};
        if (z3_read(J->z, o, n, b, 1)) { atomic_store(&J->fail, 1); break; }
        uint32_t c = 0; for (size_t i = 0; i < (size_t)c0 * c1 * c2; i++) c += b[i] < PHASE_PERIOD;
        J->cnt[t] = c;
    }
    free(b); return nullptr;
}
static int cmd_phase_check(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm phase-check <phase-root> <key> --axis A --scan-um U [--phase-um K] [--boxes 48] [--min 0.9] [--count 1] [--threads 8] [--sign 1] [--min-known 0.05] [--box z,y,x,nz,ny,nx]\n"); return 2; }
    const char *ap = opt(argc, argv, "--axis", nullptr); const double scan_um = atof(opt(argc, argv, "--scan-um", "0")), phase_um = atof(opt(argc, argv, "--phase-um", argv[3]));
    const int nbox = atoi(opt(argc, argv, "--boxes", "48")), count = atoi(opt(argc, argv, "--count", "1")), nt = ufsm_threads(atoi(opt(argc, argv, "--threads", "0")), 8), sign = atoi(opt(argc, argv, "--sign", "1"));
    const double minf = atof(opt(argc, argv, "--min", "0.9")), mink = atof(opt(argc, argv, "--min-known", "0.05"));
    if (!ap || !(scan_um > 0) || !(phase_um > 0) || nbox < 1 || (sign != 1 && sign != -1)) { fprintf(stderr, "phase-check: --axis, --scan-um > 0, a numeric key or --phase-um, --boxes >= 1, --sign +-1\n"); return 2; }
    axis ax = {0}; if (axis_load(&ax, ap)) { fprintf(stderr, "phase-check: cannot load the axis %s\n", ap); return 1; }
    z3 *z = z3_open(store_open(argv[2]), argv[3], nullptr); if (!z) { fprintf(stderr, "phase-check: %s %s: %s\n", argv[2], argv[3], z3_error()); return 1; }
    const z3_meta *m = z3_meta_of(z); const double f = phase_um / scan_um;   /* scan voxels per phase voxel */
    int64_t g[3]; uint8_t *bits = nullptr; const int64_t np = z3_chunk_map(z, g, &bits);
    if (np < 0) { fprintf(stderr, "phase-check: no chunk map (an unsharded array?)\n"); return 1; }
    long long bx[6]; const char *bs = opt(argc, argv, "--box", nullptr); int hb = 0;
    if (bs) { if (sscanf(bs, "%lld,%lld,%lld,%lld,%lld,%lld", &bx[0], &bx[1], &bx[2], &bx[3], &bx[4], &bx[5]) != 6) return usage(); hb = 1; }
    int64_t *list = malloc((size_t)(np ? np : 1) * sizeof *list); size_t nl = 0;
    for (int64_t k = 0; k < g[0] * g[1] * g[2]; k++) {
        if (!bits[k]) continue;
        if (hb) { const int64_t c[3] = {k / (g[1] * g[2]), (k / g[2]) % g[1], k % g[2]}; int in = 1;
            for (int d = 0; d < 3; d++) { const double lo = c[d] * m->chunk[d] * f, hi = (c[d] + 1) * m->chunk[d] * f; if (hi <= bx[d] || lo >= bx[d] + bx[3 + d]) in = 0; }
            if (!in) continue; }
        list[nl++] = k;
    }
    free(bits);
    const size_t c3 = (size_t)m->chunk[0] * m->chunk[1] * m->chunk[2];
    uint32_t *cnt = calloc(nl ? nl : 1, sizeof *cnt); size_t known = 0;
    if (count && nl) {
        pcheck_job J = {.z = z, .list = list, .n = nl, .cnt = cnt}; pthread_t th[64]; const int T = nt < 1 ? 1 : nt > 64 ? 64 : nt;
        for (int k = 0; k < T; k++) pthread_create(&th[k], nullptr, pcheck_count, &J);
        for (int k = 0; k < T; k++) pthread_join(th[k], nullptr);
        if (J.fail) { fprintf(stderr, "phase-check: read failed: %s\n", z3_error()); return 1; }
        for (size_t i = 0; i < nl; i++) known += cnt[i];
    }
    /* the boxes: chunks spread evenly over the stored list (with counts: only those >= --min-known known) */
    size_t nc = 0; int64_t *cand = malloc((nl ? nl : 1) * sizeof *cand);
    for (size_t i = 0; i < nl; i++) if (!count || cnt[i] >= mink * (double)c3) cand[nc++] = list[i];
    if (count && nc < (size_t)nbox) { nc = 0; for (size_t i = 0; i < nl; i++) if (cnt[i] >= 2000) cand[nc++] = list[i]; }   /* sparse labels: any chunk with some phase */
    uint8_t *b = malloc(c3); double *cy = malloc(m->chunk[0] * sizeof *cy), *cx = malloc(m->chunk[0] * sizeof *cx);
    size_t up = 0, tot = 0; int used = 0; double fmin = 1, fmax = 0;
    const size_t want = (size_t)nbox, stride = nc > want ? nc / want : 1;
    for (size_t i = stride / 2; i < nc && used < nbox; i += stride) {
        const int64_t k = cand[i], c[3] = {k / (g[1] * g[2]), (k / g[2]) % g[1], k % g[2]};
        const int64_t o[3] = {c[0] * m->chunk[0], c[1] * m->chunk[1], c[2] * m->chunk[2]}, n[3] = {m->chunk[0], m->chunk[1], m->chunk[2]};
        if (z3_read(z, o, n, b, 1)) { fprintf(stderr, "phase-check: read failed: %s\n", z3_error()); return 1; }
        if (sign < 0) for (size_t q = 0; q < c3; q++) if (b[q] < PHASE_PERIOD && b[q]) b[q] = (uint8_t)(PHASE_PERIOD - b[q]);
        for (int zz = 0; zz < m->chunk[0]; zz++) { double y, x; axis_at(&ax, (o[0] + zz) * f, &y, &x); cy[zz] = y / f; cx[zz] = x / f; }
        size_t u = 0, t = 0; const int ni[3] = {m->chunk[0], m->chunk[1], m->chunk[2]};
        phase_outward_count(b, ni, o[1], o[2], cy, cx, &u, &t);
        if (t < 200) continue;
        up += u; tot += t; used++; const double fr = (double)u / (double)t; if (fr < fmin) fmin = fr; if (fr > fmax) fmax = fr;
        if (atoi(opt(argc, argv, "--verbose", "0"))) fprintf(stderr, "phase-check: chunk %lld,%lld,%lld (scan z %.0f, r %.0f): outward %.3f of %zu\n", (long long)c[0], (long long)c[1], (long long)c[2],
                                                              (o[0] + m->chunk[0] / 2) * f, hypot((double)(o[1] + m->chunk[1] / 2) - cy[m->chunk[0] / 2], (double)(o[2] + m->chunk[2] / 2) - cx[m->chunk[0] / 2]) * f, fr, t);
    }
    long long dc[3] = {-1, -1, -1};   /* the densest stored chunk's centre, scan voxels (a holdout spot) */
    if (count && nl) { size_t bi = 0; for (size_t i = 1; i < nl; i++) if (cnt[i] > cnt[bi]) bi = i;
        const int64_t k = list[bi], c[3] = {k / (g[1] * g[2]), (k / g[2]) % g[1], k % g[2]};
        for (int d = 0; d < 3; d++) dc[d] = llround(((double)c[d] + 0.5) * m->chunk[d] * f); }
    const double F = tot ? (double)up / (double)tot : NAN, mm3 = (double)known * pow(phase_um * 1e-3, 3);
    const char *verdict = !tot ? "NODATA" : F >= minf ? "pass" : "FAIL";
    printf("phase-check %s/%s outward %.4f boxes %d pairs %zu box-range %.3f..%.3f known %s%zu voxels %.1f mm3 chunks %zu densest %lld,%lld,%lld %s\n", argv[2], argv[3], F, used, tot, used ? fmin : NAN, used ? fmax : NAN,
           count ? "" : "(not counted) ", known, mm3, nl, dc[0], dc[1], dc[2], verdict);
    free(b); free(cy); free(cx); free(cand); free(cnt); free(list); z3_close(z);
    return !tot ? 4 : F >= minf ? 0 : 3;
}

/* ufsm band <codes-root> <key> z,y,x,nz,ny,nx <out.raw> [--axis umbilicus.json] [--radius 80] [--span 75] [--threads 8] [--side side.raw]
   [--phase phase.raw]
   Band field (src/band.h) of a native-voxel box (even origin and size) from a winding_mod14 raster level (label grid, 2 native
   voxels per label voxel): writes the band per label voxel, (nz/2) x (ny/2) x (nx/2) uint8, 0..251 or 255 unknown. */
static size_t mem_available(void) { return ufsm_mem_available(); }   /* (src/sysinfo.h: cgroup aware) */
static int cmd_band(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: ufsm band <codes-root> <key> z,y,x,nz,ny,nx <out.raw> [--axis A] [--radius 80] [--span 75]\n"); return 2; }
    long long b[6];
    if (sscanf(argv[4], "%lld,%lld,%lld,%lld,%lld,%lld", &b[0], &b[1], &b[2], &b[3], &b[4], &b[5]) != 6) return usage();
    for (int d = 0; d < 6; d++) if (b[d] % 2 || (d >= 3 && b[d] <= 0)) { fprintf(stderr, "band: box must be even and positive\n"); return 2; }
    band_params bp = {(float)atof(opt(argc, argv, "--radius", "80")), (float)atof(opt(argc, argv, "--span", "75")), 0};
    axis ax = {0};
    if (axis_load(&ax, opt(argc, argv, "--axis", "/vesuvius/ufsm/umbilicus/PHercParis4/umbilicus-full-resolution.json"))) { fprintf(stderr, "band: cannot load axis\n"); return 1; }
    const int halo = (int)ceilf((bp.radius + bp.span) / 2.f) + 2;
    int64_t o[3], n[3]; int ni[3];
    for (int d = 0; d < 3; d++) { o[d] = b[d] / 2 - halo; n[d] = b[3 + d] / 2 + 2 * halo; ni[d] = (int)n[d]; }
    store *s = store_open(argv[2]);
    z3 *z = z3_open(s, argv[3], nullptr);
    if (!z) { fprintf(stderr, "%s\n", z3_error()); return 1; }
    const z3_meta *m = z3_meta_of(z);
    const size_t N = (size_t)n[0] * n[1] * n[2];
    const char *side_path = opt(argc, argv, "--side", nullptr);   /* also the recto / verso side per label voxel (band.h SIDE_*) */
    const char *phase_path = opt(argc, argv, "--phase", nullptr);   /* and / or the winding phase (band.h PHASE_*) */
    const char *ff_dir = opt(argc, argv, "--fit-fields", nullptr);   /* and / or the winding fitter's input fields (oracle) */
    const int pri = arg_index(argc, argv, "--phase-raster");   /* with --fit-fields: read the phase from ufsm phase-raster ROOT KEY (large boxes) */
    if (ff_dir && pri > 0 && pri + 2 < argc) {
        const int ffh2 = atoi(opt(argc, argv, "--fit-h", "8"));
        if (ffh2 < 2 || ffh2 % 2 || b[3] % ffh2 || b[4] % ffh2 || b[5] % ffh2) { fprintf(stderr, "band: --fit-h must be even and divide the box\n"); return 2; }
        z3 *pz = z3_open(store_open(argv[pri + 1]), argv[pri + 2], nullptr); if (!pz) { fprintf(stderr, "%s\n", z3_error()); return 1; }
        const int ph = 1; int pn[3]; int64_t po[3], pnn[3];   /* the label box with a one-voxel halo (central differences) */
        for (int d = 0; d < 3; d++) { po[d] = b[d] / 2 - ph; pnn[d] = b[3 + d] / 2 + 2 * ph; pn[d] = (int)pnn[d]; }
        const size_t PN = (size_t)pn[0] * pn[1] * pn[2]; uint8_t *phs = malloc(PN);
        memset(phs, PHASE_UNKNOWN, PN);
        int64_t ro2[3], rn2[3]; const z3_meta *pm = z3_meta_of(pz);
        for (int d = 0; d < 3; d++) { ro2[d] = po[d] < 0 ? 0 : po[d]; const int64_t e = po[d] + pnn[d] < pm->shape[d] ? po[d] + pnn[d] : pm->shape[d]; rn2[d] = e - ro2[d]; }
        if (ro2[0] == po[0] && ro2[1] == po[1] && ro2[2] == po[2] && rn2[0] == pnn[0] && rn2[1] == pnn[1] && rn2[2] == pnn[2]) {
            /* the haloed box inside the raster: read in place (a staging copy doubled the peak: a 3.9 cm slab needed ~70 GB) */
            if (z3_read(pz, ro2, rn2, phs, 8)) { fprintf(stderr, "band: phase read failed: %s\n", z3_error()); return 1; }
        } else {
            uint8_t *tmp = malloc((size_t)rn2[0] * rn2[1] * rn2[2]);
            if (z3_read(pz, ro2, rn2, tmp, 8)) { fprintf(stderr, "band: phase read failed: %s\n", z3_error()); return 1; }
            for (int64_t zz = 0; zz < rn2[0]; zz++) for (int64_t yy = 0; yy < rn2[1]; yy++)
                memcpy(phs + ((size_t)(zz + ro2[0] - po[0]) * pnn[1] + (yy + ro2[1] - po[1])) * pnn[2] + (ro2[2] - po[2]), tmp + ((size_t)zz * rn2[1] + yy) * rn2[2], (size_t)rn2[2]);
            free(tmp);
        }
        z3_close(pz);
        axis axr = {0};
        if (axis_load(&axr, opt(argc, argv, "--axis", "/vesuvius/ufsm/umbilicus/PHercParis4/umbilicus-full-resolution.json"))) { fprintf(stderr, "band: cannot load axis\n"); return 1; }
        const double t0p = now();
        if (band_fit_fields(phs, nullptr, pn, ph, b, ffh2, &axr, ff_dir)) { fprintf(stderr, "band: cannot write the fit fields to %s\n", ff_dir); return 1; }
        fprintf(stderr, "band: fit fields from the stored phase in %.1fs -> %s\n", now() - t0p, ff_dir);
        free(phs);
        return 0;
    }
    const int ffh = atoi(opt(argc, argv, "--fit-h", "8"));   /* field voxel edge, native voxels (even) */
    if (ff_dir && (ffh < 2 || ffh % 2 || b[3] % ffh || b[4] % ffh || b[5] % ffh)) { fprintf(stderr, "band: --fit-h must be even and divide the box\n"); return 2; }
    {   /* the whole haloed box in memory: ~24 bytes per label voxel (5 arrays + the band field's distance maps); refuse rather than
           drive the 24 GB WSL VM into swap and a restart (block B, 1.2e9 voxels: 27 GB). Big boxes: ufsm phase-raster (blockwise),
           then band --fit-fields --phase-raster */
        const double need = 24.0 * (double)N; const size_t av = mem_available();
        if (av && need > 0.8 * (double)av) { fprintf(stderr, "band: the %.2fe9-voxel box needs ~%.1f GB, %.1f GB available; use a smaller box, or ufsm phase-raster (blockwise) and band --fit-fields --phase-raster\n", N / 1e9, need / 1e9, av / 1e9); return 2; }
    }
    uint8_t *codes = calloc(N, 1), *band = malloc(N), *side = side_path ? malloc(N) : nullptr, *phase = phase_path || ff_dir ? malloc(N) : nullptr,
            *spacing = ff_dir ? malloc(N) : nullptr;
    int64_t ro[3], rn[3];   /* the in-bounds part of the haloed box; outside stays empty */
    for (int d = 0; d < 3; d++) { ro[d] = o[d] < 0 ? 0 : o[d]; int64_t e = o[d] + n[d] < m->shape[d] ? o[d] + n[d] : m->shape[d]; rn[d] = e - ro[d]; }
    if (rn[0] > 0 && rn[1] > 0 && rn[2] > 0) {
        uint8_t *tmp = malloc((size_t)rn[0] * rn[1] * rn[2]);
        if (z3_read(z, ro, rn, tmp, ufsm_threads(atoi(opt(argc, argv, "--threads", "0")), 8))) { fprintf(stderr, "band: read failed: %s\n", z3_error()); return 1; }
        for (int64_t zz = 0; zz < rn[0]; zz++) for (int64_t yy = 0; yy < rn[1]; yy++)
            memcpy(codes + ((size_t)(zz + ro[0] - o[0]) * n[1] + (yy + ro[1] - o[1])) * n[2] + (ro[2] - o[2]), tmp + ((size_t)zz * rn[1] + yy) * rn[2], (size_t)rn[2]);
        free(tmp);
    }
    double *cy = malloc(n[0] * sizeof(double)), *cx = malloc(n[0] * sizeof(double));
    for (int64_t zz = 0; zz < n[0]; zz++) axis_at(&ax, 2.0 * (zz + o[0]) + 0.5, &cy[zz], &cx[zz]);
    double t0 = now();
    if (band_field_side(codes, ni, o, cy, cx, bp, band, side) || (phase && band_field_phase(codes, ni, o, cy, cx, bp, band, phase, spacing))) { fprintf(stderr, "band: out of memory\n"); return 1; }
    if (ff_dir && band_fit_fields(phase, spacing, ni, halo, b, ffh, &ax, ff_dir)) { fprintf(stderr, "band: cannot write the fit fields to %s\n", ff_dir); return 1; }
    const int64_t on[3] = {b[3] / 2, b[4] / 2, b[5] / 2};
    FILE *f = fopen(argv[5], "wb");
    size_t unknown = 0;
    for (int64_t zz = 0; zz < on[0]; zz++) for (int64_t yy = 0; yy < on[1]; yy++) {
        const uint8_t *row = band + ((size_t)(zz + halo) * n[1] + yy + halo) * n[2] + halo;
        for (int64_t xx = 0; xx < on[2]; xx++) unknown += row[xx] == BAND_UNKNOWN;
        if (!f || fwrite(row, 1, (size_t)on[2], f) != (size_t)on[2]) { fprintf(stderr, "band: write failed\n"); return 1; }
    }
    fclose(f);
    if (side) {
        FILE *fs = fopen(side_path, "wb");
        for (int64_t zz = 0; zz < on[0]; zz++) for (int64_t yy = 0; yy < on[1]; yy++)
            if (!fs || fwrite(side + ((size_t)(zz + halo) * n[1] + yy + halo) * n[2] + halo, 1, (size_t)on[2], fs) != (size_t)on[2]) { fprintf(stderr, "band: side write failed\n"); return 1; }
        fclose(fs); free(side);
    }
    if (phase && phase_path) {
        FILE *fp = fopen(phase_path, "wb");
        for (int64_t zz = 0; zz < on[0]; zz++) for (int64_t yy = 0; yy < on[1]; yy++)
            if (!fp || fwrite(phase + ((size_t)(zz + halo) * n[1] + yy + halo) * n[2] + halo, 1, (size_t)on[2], fp) != (size_t)on[2]) { fprintf(stderr, "band: phase write failed\n"); return 1; }
        fclose(fp);
    }
    free(phase); free(spacing);
    fprintf(stderr, "band field %lldx%lldx%lld (halo %d) in %.1fs, unknown %.1f%%\n", (long long)on[0], (long long)on[1], (long long)on[2], halo,
            now() - t0, 100.0 * unknown / ((double)on[0] * on[1] * on[2]));
    free(codes); free(band); free(cy); free(cx); z3_close(z);
    return 0;
}

/* ufsm fit <fields-dir> [--codes ROOT KEY] [--dir 0|1|-1] [--lattice 512,256,128,64,64,32] [--steps 300,...] [--phase-stage 4]
   [--knot-dz 1024] [--quiet] [--out MODEL-DIR] [--winding ZARR-DIR [--winding-step 4] [--um 2.4]] [--slab Z [--overlap 256]]
   [--model MODEL-DIR] [--axis-normal D] [--cpu [--threads N]] [--phase-robust S] [--conf-pow P] [--bend B] [--w-phase W --w-normal W
   --w-density W --w-smooth W]
   The winding fitter (src/fit.h) on a directory of fit fields (ufsm band --fit-fields, predict --fit-fields / --fit-zarr). --slab:
   a global coarse fit plus z-slabs of Z native voxels (a fit set; --out writes it, --model reads a set or a model). --codes: score
   against a winding_mod14 band raster (label grid, 2 native voxels per label voxel, code = 1 + round(18 q) mod 252 with q the mesh
   coordinate, q = U for dir +1): on every labelled voxel of the box U_fit - q must be one constant mod 14 turns. --axis-normal D: a
   local box away from the axis (crushed scrolls): the axis replaced by a vertical line D native voxels from the box centre along the
   mean sheet normal (D < 0: the other side). --cpu: the CPU backend (no GPU; src/fit_terms.h holds the terms both backends fit).
   --phase-robust S: a Cauchy phase residual of scale S turns; --conf-pow P: phase weight conf^P; --bend B: curvature-adaptive
   smoothness; --w-pgrad W [--pgrad-delta D]: phase-gradient matching (src/fit_terms.h); --unwrap-voxels 1 / 0 / -1: the phase
   stage's integer init by voxels / lattice nodes / none (fit_cfg). --wind4-weight W (default 0): the winding mod 4 term when the fields
   carry dz dy dx dq planes (task unwrap2; 0 off). --w-recto W (default 0): recto anchors r^2 (1 - cos 2 pi W) from the fields' r plane
   (predict --fit-fields / --fit-zarr write it). --out also writes MODEL/fit-config.json (the weights and options of the fit).
   CT (fields with ct / cs planes, ufsm ct-fields / predict --fit-ct): --w-ct W papyrus / void parity c m (1 - cos 2 pi (W - --ct-phi)) weighted
   (1 - phase confidence)^--ct-pow (>= --ct-floor), the parity m the cell's CT contrast to its +---ct-radius-um neighbourhood; with it
   --ct-flood 1 (default) lets low-confidence voxels join the voxel unwrapping with the CT's own phase. --w-cspacing W: the density term
   against the CT sheet spacing where the network's is unknown or differs by more than --cs-tol in log.
   Local slip repair (fit_cfg uw_repair, src/l1pot.h): --uw-repair T splits the flooded voxels of each T^3 tile into pieces whose integers
   agree and re-solves their integer shifts as the least-cost (weighted L1) assignment of all their votes; evidence --uw-rep-w4 W (the
   winding mod 4 planes), --uw-rep-geo W (the gradient stages' W over --uw-rep-gd field voxels), --uw-rep-cs W (the CT sheet spacing);
   --uw-rep-dump PREFIX writes the per-voxel shifts and seams. Vote weights (lower confidence)^--uw-rep-qpow (3) x the phase-stretch
   factor --uw-rep-stretch (0.5; 0 off) over a (2 --uw-rep-sr + 1)^3 box (1). Recommended: --uw-repair 8 (Paris 4 hold box, wind96
   fields over 9 predict windows: mean ids 0.827 -> 0.893).
   --fibers FILE --w-fib W: published fibre polylines (ufsm fiber-export for this fields dir) as a pair term W(b) - W(a) = the network phase
   change along the fibre, on point pairs 16..128 native voxels apart whose phase stays within half a turn (src/fit.c fib_pairs).
   --phase-stretch S [--phase-stretch-r R]: the phase confidence x exp(-log2(s)^2 / 2 S^2), s the phase stretch (|grad phase| x spacing
   over a (2R+1)^3 box, R default 1), before the fit: where the phase skips sheets the density sets the turn count (0 off). */
/* U (the unwrapped radius in turns; sheets where U - dir theta / 2 pi is an integer) for the scorer: fill(ctx, og, n3, step, codes, U)
   writes U at og + i step (native voxels relative to the box origin) over the n3 grid, NaN where unknown (it may skip the points
   whose code is not 1..252); returns nonzero on failure */
typedef int (*score_fill_fn)(void *ctx, const double og[3], const int64_t n3[3], double step, const uint8_t *codes, float *U);
/* Score U against the band codes (winding_mod14) of a box, streamed in label z-chunks: the gauge (the mode of U - q mod 14), the
   fraction of winding ids right (|e| < 0.5 turn), |e| < 0.08 turn, median / p90 |e|; detail: integer errors by z octile and by radius
   (cy, cx: the axis per native z, box-local, (int)(shp[0] h) + 1 entries; um: the native voxel size); dump: box-local z y x and U - q
   per voxel. cf: a label voxel in native voxels. Returns 0, or 1 on a read failure. */
/* score_codes' per-voxel tally of one block over threads (contiguous z ranges; counts summed and the detail records concatenated in
   thread order: the same totals and record order as one pass) */
typedef struct { float d; uint8_t z, r; } score_rec;
typedef struct {
    const uint8_t *codes; const float *U; int64_t n3[3], o3[3]; double cf, h, um; const long long *org; const int *shp; const float *cy, *cx; int na_ax, detail;
    size_t *hist[64], *fine[64], nd[64], nan[64]; score_rec *dd[64]; size_t ndet[64], dcap[64]; int nt;
    pa_buf *pa;   /* per thread (fit --score-pairs), or null */
} score_job;
static int g_score_pairs;   /* fit --score-pairs 1: pair winding accuracy (src/pairacc.h) after the box score */
enum { SC_NB = 14 * 72, SC_NF = 14 * 1024 };
static void *score_rows(void *arg) {
    score_job *J = ((void **)arg)[0]; const int t = (int)(intptr_t)((void **)arg)[1];
    const int64_t z0 = J->n3[0] * t / J->nt, z1 = J->n3[0] * (t + 1) / J->nt; const double cf = J->cf, h = J->h; const long long *org = J->org; const int *shp = J->shp;
    size_t *hist = J->hist[t], *fine = J->fine[t], nd = 0, nan = 0;
    /* the thread's detail records and pair buffer as locals (written back at the end): the per-thread slots of J share cache lines */
    score_rec *ddt = J->dd[t]; size_t ndt = J->ndet[t], dct = J->dcap[t]; pa_buf pat = {0}; if (J->pa) pat = J->pa[t];
    for (int64_t zz = z0; zz < z1; zz++) for (int64_t yy = 0; yy < J->n3[1]; yy++) for (int64_t xx = 0; xx < J->n3[2]; xx++) {
        const size_t i = ((size_t)zz * J->n3[1] + yy) * J->n3[2] + xx; const uint8_t cd = J->codes[i];
        if (cd < 1 || cd > 252) continue;
        const double pz = cf * (zz + J->o3[0] + 0.25) - org[0], py = cf * (yy + J->o3[1] + 0.25) - org[1], px = cf * (xx + J->o3[2] + 0.25) - org[2];
        if (pz < 0 || py < 0 || px < 0 || pz > shp[0] * h || py > shp[1] * h || px > shp[2] * h) continue;
        if (!(J->U[i] == J->U[i])) { nan++; continue; }
        double d = fmod(J->U[i] - (cd - 1) / 18.0, 14.0); if (d < 0) d += 14; if (d >= 14) d = 0;
        hist[(int)(d / 14 * SC_NB) % SC_NB]++; fine[(int)(d / 14 * SC_NF) % SC_NF]++; nd++;
        if (J->pa) pa_add(&pat, (float)pz, (float)py, (float)px, (float)d, cd, ((uint64_t)(zz + J->o3[0]) << 42) ^ ((uint64_t)(yy + J->o3[1]) << 21) ^ (uint64_t)(xx + J->o3[2]));
        if (J->detail) {
            if (ndt == dct) { dct = dct ? 2 * dct : 1 << 16; ddt = realloc(ddt, dct * sizeof(score_rec)); if (!ddt) abort(); }
            const int zi = (int)fmin(7, fmax(0, pz / (shp[0] * h) * 8));
            const int ai = (int)fmin(J->na_ax - 1, fmax(0, pz)); const double rr = hypot(py - J->cy[ai], px - J->cx[ai]) * J->um;
            ddt[ndt++] = (score_rec){(float)d, (uint8_t)zi, (uint8_t)fmin(255, rr / 250)};
        }
    }
    J->nd[t] += nd; J->nan[t] += nan;   /* (summed over the blocks) */
    J->dd[t] = ddt; J->ndet[t] = ndt; J->dcap[t] = dct; if (J->pa) J->pa[t] = pat;
    return nullptr;
}
typedef struct { z3 *z; int64_t o3[3], n3[3]; uint8_t *buf; int rc; size_t any; } sc_read;   /* one block's codes (a reader thread) */
static void *sc_read_run(void *a) {
    sc_read *r = a; r->any = 0; r->rc = z3_read(r->z, r->o3, r->n3, r->buf, 16);
    const size_t nb3 = (size_t)r->n3[0] * r->n3[1] * r->n3[2];
    if (!r->rc) for (size_t i = 0; i < nb3 && !r->any; i++) r->any = r->buf[i] >= 1 && r->buf[i] <= 252;
    return nullptr;
}
static int score_codes(z3 *z, double cf, const long long org[3], const int shp[3], double h, double um, const float *cy, const float *cx, int detail, FILE *dump,
                       score_fill_fn fill, void *ctx) {
    int64_t lo[3], ln[3]; for (int d = 0; d < 3; d++) { lo[d] = (int64_t)floor(org[d] / cf); ln[d] = (int64_t)(shp[d] * h / cf); }
    enum { NB = SC_NB, NF = SC_NF }; size_t *hist = calloc(NB, sizeof(size_t)), *fine = calloc(NF, sizeof(size_t)), nd = 0, nan = 0;   /* d = U - q mod 14: 1/72 and 1/1024 turn bins */
    score_rec *dd = nullptr; size_t ndet = 0, dcap = 0;
    const int na_ax = (int)(shp[0] * h) + 1;
    pa_buf *pab = nullptr; if (g_score_pairs && !dump) { pab = calloc(16, sizeof *pab); for (int t = 0; t < 16; t++) pa_init(&pab[t], (size_t)1 << 19); }
    /* blocks of up to 128 x 1024 x 1024 label voxels aligned to the codes' 128^3 chunks (z planes one at a time decoded every chunk
       128 times: minutes on a 3.9 cm slab) */
    enum { SBZ = 128, SBY = 1024, SBX = 1024 };
    uint8_t *cbuf[2] = {malloc((size_t)SBZ * SBY * SBX), malloc((size_t)SBZ * SBY * SBX)}; float *U = malloc((size_t)SBZ * SBY * SBX * 4);
    const int tms = getenv("UFSM_FIT_TIMES") != nullptr; double tr = 0, tf = 0, tw = 0, tq = 0;   /* UFSM_FIT_TIMES: code reads (waits), U fill, tally */
    score_job J = {.cf = cf, .h = h, .um = um, .org = org, .shp = shp, .cy = cy, .cx = cx, .na_ax = na_ax, .detail = detail, .pa = pab};   /* (the threaded tally) */
    /* the blocks in order; block b + 1's codes decode on a thread while block b is filled and tallied */
    size_t nblk = 0, cblk = 0; int64_t (*blk)[6] = nullptr;
    for (int64_t za = lo[0]; za < lo[0] + ln[0]; za = (za / SBZ + 1) * SBZ)
    for (int64_t ya = lo[1]; ya < lo[1] + ln[1]; ya = (ya / SBY + 1) * SBY)
    for (int64_t xa = lo[2]; xa < lo[2] + ln[2]; xa = (xa / SBX + 1) * SBX) {
        const int64_t ze = (za / SBZ + 1) * SBZ < lo[0] + ln[0] ? (za / SBZ + 1) * SBZ : lo[0] + ln[0];
        const int64_t ye = (ya / SBY + 1) * SBY < lo[1] + ln[1] ? (ya / SBY + 1) * SBY : lo[1] + ln[1];
        const int64_t xe = (xa / SBX + 1) * SBX < lo[2] + ln[2] ? (xa / SBX + 1) * SBX : lo[2] + ln[2];
        if (nblk == cblk) { cblk = cblk ? 2 * cblk : 64; blk = realloc(blk, cblk * sizeof *blk); }
        blk[nblk][0] = za; blk[nblk][1] = ya; blk[nblk][2] = xa; blk[nblk][3] = ze - za; blk[nblk][4] = ye - ya; blk[nblk][5] = xe - xa; nblk++;
    }
    sc_read rd[2] = {{.z = z, .buf = cbuf[0]}, {.z = z, .buf = cbuf[1]}}; pthread_t rth[2]; int rup[2] = {0, 0};
    if (nblk) { memcpy(rd[0].o3, blk[0], 3 * sizeof(int64_t)); memcpy(rd[0].n3, blk[0] + 3, 3 * sizeof(int64_t)); rup[0] = pthread_create(&rth[0], nullptr, sc_read_run, &rd[0]) == 0; if (!rup[0]) sc_read_run(&rd[0]); }
    for (size_t bi = 0; bi < nblk; bi++) {
        const int s = (int)(bi & 1); sc_read *R = &rd[s];
        tq = now(); if (rup[s]) { pthread_join(rth[s], nullptr); rup[s] = 0; } tr += now() - tq;
        if (bi + 1 < nblk) { sc_read *N = &rd[s ^ 1]; memcpy(N->o3, blk[bi + 1], 3 * sizeof(int64_t)); memcpy(N->n3, blk[bi + 1] + 3, 3 * sizeof(int64_t));
                             rup[s ^ 1] = pthread_create(&rth[s ^ 1], nullptr, sc_read_run, N) == 0; if (!rup[s ^ 1]) sc_read_run(N); }
        uint8_t *codes = R->buf; const int64_t *o3 = R->o3, *n3 = R->n3;
        if (R->rc) { fprintf(stderr, "score: code read failed: %s\n", z3_error()); for (int q = 0; q < 2; q++) if (rup[q]) pthread_join(rth[q], nullptr); free(cbuf[0]); free(cbuf[1]); free(U); free(hist); free(fine); free(dd); free(blk); return 1; }
        tq = now();
        if (!R->any) continue;
        const double og[3] = {cf * o3[0] - 0.25 * cf - org[0], cf * o3[1] - 0.25 * cf - org[1], cf * o3[2] - 0.25 * cf - org[2]};   /* label voxel i at cf (i + lo + 1 / 4) - org */
        if (fill(ctx, og, n3, cf, codes, U)) { fprintf(stderr, "score: cannot evaluate U on the score grid\n"); for (int q = 0; q < 2; q++) if (rup[q]) pthread_join(rth[q], nullptr); free(cbuf[0]); free(cbuf[1]); free(U); free(hist); free(fine); free(dd); free(blk); return 1; }
        tf += now() - tq; tq = now();
        if (!dump) {   /* threads (the dump keeps the serial voxel order); per-thread tallies and detail records kept over all blocks */
            J.codes = codes; J.U = U; for (int d = 0; d < 3; d++) { J.n3[d] = n3[d]; J.o3[d] = o3[d]; }
            long ncpu = ufsm_ncpu(); const char *te = getenv("UFSM_FIT_THREADS"); if (te && atoi(te) > 0) ncpu = atoi(te);   /* (the fit's threads) */
            J.nt = (int)(ncpu < 1 ? 1 : ncpu > 16 ? 16 : ncpu); if (J.nt > n3[0]) J.nt = (int)n3[0];
            pthread_t th[64]; void *args[64][2];
            for (int t = 0; t < J.nt; t++) {
                if (!J.hist[t]) { J.hist[t] = calloc(NB, sizeof(size_t)); J.fine[t] = calloc(NF, sizeof(size_t)); }
                args[t][0] = &J; args[t][1] = (void *)(intptr_t)t; pthread_create(&th[t], nullptr, score_rows, args[t]);
            }
            for (int t = 0; t < J.nt; t++) pthread_join(th[t], nullptr);
            tw += now() - tq;
            continue;
        }
        for (int64_t zz = 0; zz < n3[0]; zz++) for (int64_t yy = 0; yy < n3[1]; yy++) for (int64_t xx = 0; xx < n3[2]; xx++) {
            const size_t i = ((size_t)zz * n3[1] + yy) * n3[2] + xx; const uint8_t cd = codes[i];
            if (cd < 1 || cd > 252) continue;
            const double pz = cf * (zz + o3[0] + 0.25) - org[0], py = cf * (yy + o3[1] + 0.25) - org[1], px = cf * (xx + o3[2] + 0.25) - org[2];
            if (pz < 0 || py < 0 || px < 0 || pz > shp[0] * h || py > shp[1] * h || px > shp[2] * h) continue;
            if (!(U[i] == U[i])) { nan++; continue; }   /* no U here (a raster's unreached voxel) */
            double d = fmod(U[i] - (cd - 1) / 18.0, 14.0); if (d < 0) d += 14; if (d >= 14) d = 0;
            if (dump) { const float rec[4] = {(float)pz, (float)py, (float)px, (float)d}; if (fwrite(rec, 4, 4, dump) != 4) dump = nullptr; }
            hist[(int)(d / 14 * NB) % NB]++; fine[(int)(d / 14 * NF) % NF]++; nd++;
            if (detail) {   /* d and the voxel's z / radius bins for the breakdown after the gauge is known */
                if (ndet == dcap) { dcap = dcap ? 2 * dcap : 1 << 20; dd = realloc(dd, dcap * sizeof *dd); }
                const int zi = (int)fmin(7, fmax(0, pz / (shp[0] * h) * 8));
                const int ai = (int)fmin(na_ax - 1, fmax(0, pz)); const double rr = hypot(py - cy[ai], px - cx[ai]) * um;
                dd[ndet++] = (score_rec){(float)d, (uint8_t)zi, (uint8_t)fmin(255, rr / 250)};   /* radius in 250 um bins */
            }
        }
    }
    free(cbuf[0]); free(cbuf[1]); free(U); free(blk);
    const size_t ndet_serial = ndet;   /* (dd: the dump path's records; the threads keep theirs in J.dd) */
    for (int t = 0; t < 64; t++) {   /* the threads' tallies (integer counts: any order) */
        if (J.hist[t]) { for (int k = 0; k < NB; k++) hist[k] += J.hist[t][k]; for (int k = 0; k < NF; k++) fine[k] += J.fine[t][k]; }
        nd += J.nd[t]; nan += J.nan[t]; ndet += J.ndet[t]; free(J.hist[t]); free(J.fine[t]);
    }
    if (tms) fprintf(stderr, "fit times: score code reads %.2fs, U fill %.2fs, tally %.2fs\n", tr, tf, tw);
    size_t best = 0; int bi = 0;
    for (int k = 0; k < NB; k++) { size_t sacc = 0; for (int e = -18; e <= 18; e++) sacc += hist[(k + e + NB) % NB]; if (sacc > best) { best = sacc; bi = k; } }   /* +-0.25 turn window */
    const double mode = (bi + 0.5) * 14.0 / NB;
    size_t right = 0, close = 0; double mean = 0;
    for (int k = 0; k < NF; k++) { double e = (k + 0.5) * 14.0 / NF - mode; e -= 14 * round(e / 14); if (fabs(e) < 0.5) { right += fine[k]; mean += e * fine[k]; } }
    mean = right ? mean / right : 0;   /* the gauge refined to the mean error of the right voxels */
    size_t *ah = calloc(NF / 28 + 1, sizeof(size_t));   /* |e| < 0.5 in 1/1024 turn bins */
    for (int k = 0; k < NF; k++) { double e = (k + 0.5) * 14.0 / NF - mode - mean; e -= 14 * round(e / 14); close += fabs(e) < 0.08 ? fine[k] : 0; if (fabs(e) < 0.5) ah[(int)(fabs(e) * 1024)] += fine[k]; }
    size_t na2 = 0; for (int k = 0; k <= NF / 28; k++) na2 += ah[k];
    double med = 0, p90 = 0; { size_t c = 0; int got = 0; for (int k = 0; k <= NF / 28; k++) { c += ah[k]; if (!got && c * 2 >= na2) { med = (k + 0.5) / 1024; got = 1; } if (c * 10 >= na2 * 9) { p90 = (k + 0.5) / 1024; break; } } }
    printf("fit score: %zu labelled voxels, gauge %.3f, winding ids right %.4f, |error| < 0.08 turn %.4f, |error| median %.3f p90 %.3f turn\n", nd, mode + mean,
           nd ? (double)right / nd : 0, nd ? (double)close / nd : 0, na2 ? med : 0, na2 ? p90 : 0);
    if (nan) printf("  %zu labelled voxels without U (%.4f of those in the box), not scored\n", nan, (double)nan / (nan + nd));
    if (detail && ndet) {
        size_t ie[8][9] = {{0}}, re[256][9] = {{0}};   /* integer error -4..4 by z octile / by 250 um radius bin (up to 64 mm) */
        for (int s = -1; s < 64; s++) {   /* the serial (dump) records, then each thread's */
            const score_rec *r = s < 0 ? dd : J.dd[s]; const size_t cnt = s < 0 ? ndet_serial : J.ndet[s];
            for (size_t i = 0; i < cnt; i++) {
                double e = r[i].d - mode - mean; e -= 14 * round(e / 14); const int k = (int)lround(e);
                if (k < -4 || k > 4) continue;
                ie[r[i].z][k + 4]++; re[r[i].r][k + 4]++;
            }
        }
        printf("  integer error -2 -1 0 +1 +2 (fractions):\n");
        for (int zb = 0; zb < 8; zb++) { size_t t = 0; for (int k = 0; k < 9; k++) t += ie[zb][k]; if (!t) continue;
            printf("    z octile %d %9zu voxels: %.3f %.3f %.3f %.3f %.3f\n", zb, t, (double)ie[zb][2] / t, (double)ie[zb][3] / t, (double)ie[zb][4] / t, (double)ie[zb][5] / t, (double)ie[zb][6] / t); }
        for (int rb = 0; rb < 256; rb += 2) { size_t t = 0, c[9] = {0}; for (int k = 0; k < 9; k++) { c[k] = re[rb][k] + (rb + 1 < 256 ? re[rb + 1][k] : 0); t += c[k]; } if (t < 1000) continue;
            printf("    radius %5.1f-%4.1f mm %9zu voxels: %.3f %.3f %.3f %.3f %.3f\n", rb * 0.25, rb * 0.25 + 0.5, t, (double)c[2] / t, (double)c[3] / t, (double)c[4] / t, (double)c[5] / t, (double)c[6] / t); }
    }
    if (pab) {   /* pair winding accuracy (gauge-free): 0.5 / 1 / 2 / 4 mm */
        pa_buf all; pa_init(&all, 1 << 20); for (int t = 0; t < 16; t++) { pa_merge(&all, &pab[t]); pa_free(&pab[t]); } free(pab); pab = nullptr;
        const double mm[4] = {0.5, 1, 2, 4}; double dv[4], a[4], ac[4]; size_t na[4], nc[4]; for (int k = 0; k < 4; k++) dv[k] = mm[k] * 1000 / um;
        tq = now(); pa_score(&all, dv, 4, 200000, 0x5eed, a, ac, na, nc);
        if (tms) fprintf(stderr, "fit times: pair score %.2fs\n", now() - tq);
        printf("pair winding right (%zu records):", all.n);
        for (int k = 0; k < 4; k++) printf(" %g mm %.4f cross %.4f (%zu / %zu)%s", mm[k], a[k], ac[k], na[k], nc[k], k < 3 ? " |" : "\n");
        pa_free(&all);
    }
    free(dd); for (int t = 0; t < 64; t++) free(J.dd[t]);
    free(ah); free(hist); free(fine);
    return 0;
}
typedef struct { fit_set_dev *gd; const fit_set *fs; } fit_fill_ctx;   /* gd null: the CPU */
static int fit_fill(void *c, const double og[3], const int64_t n3[3], double step, const uint8_t *codes, float *U) {
    fit_fill_ctx *f = c; return f->gd ? fit_set_grid(f->gd, f->fs, og, n3, step, 1, U) : fit_set_grid_cpu(f->fs, og, n3, step, 1, codes, U);
}

/* ufsm convert-unwrap2 IN OUT: a task unwrap checkpoint (recto 0, ink 1, phase 2-3, normal 4-9, density 10) as a task unwrap2 one
   (phase 0-1, normal 2-7, density 8, winding mod 4 9-10 fresh) without a training step: the live and EMA weights copied, the head rows
   2-10 moved to 0-8 (unet_init_from_map, as train --task unwrap2 --init-from does), the header's runtime / augmentation extras kept and
   its "unwrap" / "phase" objects rewritten for the new layout. The winding mod 4 rows are untrained: the result predicts the donor's
   fit fields byte for byte (predict --fit-fields / --fit-zarr) and is a warm start for train --task unwrap2 --init-from. */
static int json_obj_end(const char *s) {   /* s at '{': the length through its matching '}' (strings skipped), or -1 */
    int d = 0, q = 0;
    for (int i = 0; s[i]; i++) {
        if (q) { if (s[i] == '\\' && s[i + 1]) i++; else if (s[i] == '"') q = 0; continue; }
        if (s[i] == '"') q = 1; else if (s[i] == '{') d++; else if (s[i] == '}' && --d == 0) return i + 1;
    }
    return -1;
}
static int cmd_convert_unwrap2(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm convert-unwrap2 <task-unwrap.ckpt> <out.ckpt> [--gpu 0]\n"); return 2; }
    const char *in = argv[2], *out = argv[3];
    unet_cfg cfg; int step = 0;
    if (unet_peek(in, &cfg, &step) || checkpoint_unwrap_version(in) != 1 || cfg.cout < 11) { fprintf(stderr, "convert-unwrap2: %s is not a task unwrap checkpoint (11+ outputs)\n", in); return 2; }
    char *hdr = checkpoint_header_line(in);
    const char *ex = hdr ? strstr(hdr, "\"extra\":{") : nullptr;
    if (!ex) { fprintf(stderr, "convert-unwrap2: no extra object in %s\n", in); free(hdr); return 1; }
    ex += 8;
    const int el = json_obj_end(ex);
    if (el < 0) { fprintf(stderr, "convert-unwrap2: malformed header\n"); free(hdr); return 1; }
    /* the extra object without its "unwrap" and "phase" members, then the unwrap2 ones */
    char *extra = calloc((size_t)el + 4096, 1); size_t n = 0;
    for (int i = 0; i < el - 1;) {
        const char *keys[2] = {"\"unwrap\":{", "\"phase\":{"}; int skip = 0;
        for (int k = 0; k < 2 && !skip; k++) if (!strncmp(ex + i, keys[k], strlen(keys[k])) && (ex[i - 1] == ',' || ex[i - 1] == '{')) {
            const int ol = json_obj_end(ex + i + strlen(keys[k]) - 1); if (ol < 0) { free(hdr); free(extra); return 1; }
            skip = (int)strlen(keys[k]) - 1 + ol;
            if (ex[i + skip] == ',') skip++; else if (n && extra[n - 1] == ',') n--;
        }
        if (skip) { i += skip; continue; }
        extra[n++] = ex[i++];
    }
    double ds = 32, nl = 1, pw = 0, pu = 120; int dc = 0, nst = 4, air = 0;
    { const char *u = strstr(hdr, "\"unwrap\":{"), *q;
      if ((q = strstr(u, "\"density_scale\":"))) ds = atof(q + 16);
      if ((q = strstr(u, "\"density_code\":"))) dc = atoi(q + 15);
      if ((q = strstr(u, "\"packed_weight\":"))) pw = atof(q + 16);
      if ((q = strstr(u, "\"packed_um\":"))) pu = atof(q + 12);
      if ((q = strstr(u, "\"normal_lambda\":"))) nl = atof(q + 16);
      if ((q = strstr(u, "\"normal_step\":"))) nst = atoi(q + 14);
      const char *p = strstr(hdr, "\"phase\":{\"version\":1"); if (p && (q = strstr(p, "\"air\":"))) air = atoi(q + 6); }
    n += (size_t)snprintf(extra + n, 4096, "%s\"phase\":{\"version\":1,\"channel\":0,\"channels\":[\"0.5+0.5cos(2pi phase)\",\"0.5+0.5sin(2pi phase)\"],\"phase\":\"fraction of the way outward from a recto to the next\",\"air\":%d,\"lambda\":1,\"dense\":0}"
                          ",\"unwrap\":{\"version\":2,\"channels\":{\"phase\":[0,1],\"normal\":[2,3,4,5,6,7],\"density\":8,\"wind4\":[9,10]},\"normal\":\"n_z^2, n_y^2, n_x^2, 0.5+n_z n_y, 0.5+n_z n_x, 0.5+n_y n_x\",\"density\":\"%s, rho = |grad phase| turns per voxel\","
                          "\"wind4\":\"0.5+0.5cos / sin(2pi W/4), W the stored phase unwrapped over the window, gauge-free\",\"density_scale\":%.9g,\"density_code\":%d,\"packed_weight\":%.9g,\"packed_um\":%.9g,\"normal_lambda\":%.9g,\"normal_step\":%d,"
                          "\"wind4_lambda\":0,\"converted_from\":\"task unwrap rows 2-10; winding mod 4 untrained\"}}",
                          n > 1 ? "," : "", air, dc ? "x / (1 + x), x = density_scale rho" : "min(1, density_scale rho)", ds, dc, pw, pu, nl, nst);
    free(hdr);
    if (nn_init(atoi(opt(argc, argv, "--gpu", "0")))) { free(extra); return 1; }
    cfg.cout = 11;
    unet *u = unet_create(&cfg); unet_init(u, 1);
    int rows[11]; for (int c = 0; c < 11; c++) rows[c] = c < 9 ? c + 2 : -1;
    const int nt = unet_init_from_map(u, in, rows);
    if (nt < 0 || unet_save(u, out, step, extra)) { fprintf(stderr, "convert-unwrap2: cannot convert %s to %s\n", in, out); unet_free(u); free(extra); return 1; }
    fprintf(stderr, "convert-unwrap2: %d tensors of %s (step %d) -> %s (task unwrap2: rows 2-10 -> 0-8, winding mod 4 rows fresh)\n", nt, in, step, out);
    unet_free(u); free(extra);
    return 0;
}

static pthread_t g_fit_gth; static int g_fit_gthr, g_fit_gdev;   /* fit: the GPU context comes up on a thread while the fields load */
static void *fit_gpu_init_thread(void *a) { (void)a; nn_init(g_fit_gdev); return nullptr; }
static void fit_gpu_join(void) {   /* the init thread done; the device selected on this thread too */
    if (!g_fit_gthr) return;
    const double tj = now(); pthread_join(g_fit_gth, nullptr); g_fit_gthr = 0; nn_init(g_fit_gdev);
    if (getenv("UFSM_FIT_TIMES")) fprintf(stderr, "fit times: gpu init wait %.2fs\n", now() - tj);
}
/* fit --sweep: the streamed fields of the last run, kept for the next (same directory, optional planes and pooling factors) */
static int g_fit_sweep;
static struct { int valid, kskip, kct, nfp, fps[8]; char dir[1400]; fit_fields_buf fb; fit_stream fst; } g_fcache;
static int cmd_fit(int argc, char **argv);
/* ufsm fit FIELDS [common options] --sweep FILE: one process, the fields loaded once (GPU fits: device fields and pooled stages
   shared), then one fit per line of FILE: the line's options (whitespace-separated, '#' comments and blank lines skipped) come first, so
   they override the common ones; each line needs its own --out (if any). Each run prints "fit sweep: run i/n: <line>" and its usual log;
   a run's result equals the same options as a single fit (deterministic GPU sums). Options that change the loaded fields (--w-recto,
   --wind4-weight, --uw-rep-w4, --w-ct, --w-cspacing, --lattice) reload them when they differ from the previous run. */
static int cmd_fit_sweep(int argc, char **argv, const char *file) {
    FILE *f = fopen(file, "r"); if (!f) { fprintf(stderr, "fit: cannot read --sweep %s\n", file); return 1; }
    char *line = nullptr; size_t cap = 0; char **lines = nullptr; int nl = 0;
    while (getline(&line, &cap, f) > 0) {
        char *p = line; while (*p == ' ' || *p == '\t') p++;
        if (!*p || *p == '\n' || *p == '#') continue;
        p[strcspn(p, "\r\n")] = 0; lines = realloc(lines, (nl + 1) * sizeof *lines); lines[nl++] = strdup(p);
    }
    free(line); fclose(f);
    g_fit_sweep = 1; int bad = 0; const double t0 = now();
    for (int r = 0; r < nl; r++) {
        char *buf = strdup(lines[r]); char **av = malloc((argc + 256) * sizeof *av); int ac = 0;
        av[ac++] = argv[0]; av[ac++] = argv[1]; av[ac++] = argv[2];
        for (char *q = strtok(buf, " \t"); q && ac < argc + 250; q = strtok(nullptr, " \t")) av[ac++] = q;
        for (int i = 3; i < argc; i++) { if (!strcmp(argv[i], "--sweep") && i + 1 < argc) { i++; continue; } av[ac++] = argv[i]; }
        av[ac] = nullptr;
        printf("fit sweep: run %d/%d: %s\n", r + 1, nl, lines[r]); fflush(stdout); fprintf(stderr, "fit sweep: run %d/%d: %s\n", r + 1, nl, lines[r]);
        const double tr = now(); const int rc = cmd_fit(ac, av); fflush(stdout);
        fprintf(stderr, "fit sweep: run %d/%d %s in %.1fs\n", r + 1, nl, rc ? "FAILED" : "done", now() - tr);
        bad += rc != 0; free(av); free(buf); free(lines[r]);
    }
    free(lines);
    if (g_fcache.valid) { fit_stream_free(&g_fcache.fst); fit_fields_free(&g_fcache.fb); g_fcache.valid = 0; }
    fprintf(stderr, "fit sweep: %d runs (%d failed) in %.1fs\n", nl, bad, now() - t0);
    return bad ? 1 : 0;
}
/* fit --ref-model: the ref (a whole-scroll model: ~0.9 GB of lattice) loads on a thread while the fields load */
static struct { pthread_t th; int on; const char *dir; fit_set *s; } g_fit_ref;
static void *fit_ref_thread(void *a) { (void)a; g_fit_ref.s = fit_set_load(g_fit_ref.dir); return nullptr; }
static int cmd_fit(int argc, char **argv) {
    if (!g_fit_sweep && *opt(argc, argv, "--sweep", "")) return cmd_fit_sweep(argc, argv, opt(argc, argv, "--sweep", ""));
    const char *dir = argv[2];
    fit_cfg c = fit_cfg_default();
    c.verbose = arg_index(argc, argv, "--quiet") < 0;
    c.dir = atoi(opt(argc, argv, "--dir", "0")); c.unwrap_voxels = atoi(opt(argc, argv, "--unwrap-voxels", "1"));
    c.cpu = arg_index(argc, argv, "--cpu") > 0; c.threads = atoi(opt(argc, argv, "--threads", "0")); fit_set_threads(c.threads);
    c.phase_robust = (float)atof(opt(argc, argv, "--phase-robust", "0")); c.conf_pow = (float)atof(opt(argc, argv, "--conf-pow", "1"));
    c.bend = (float)atof(opt(argc, argv, "--bend", "0"));
    c.w_pgrad = (float)atof(opt(argc, argv, "--w-pgrad", "0")); c.pgrad_delta = (float)atof(opt(argc, argv, "--pgrad-delta", "0.25"));
    c.w_phase = (float)atof(opt(argc, argv, "--w-phase", "1")); c.w_normal = (float)atof(opt(argc, argv, "--w-normal", "3"));
    c.w_density = (float)atof(opt(argc, argv, "--w-density", "1")); c.w_smooth = (float)atof(opt(argc, argv, "--w-smooth", "0.1"));
    c.w_wind4 = (float)atof(opt(argc, argv, "--wind4-weight", "0"));   /* the winding mod 4 term of task unwrap2 fields (dz dy dx dq planes; 0 off: weight 4 cost fit-val, 0.5-1 neutral) */
    c.w_recto = (float)atof(opt(argc, argv, "--w-recto", "0"));        /* recto anchors (fields with an r plane; 0 off) */
    if (!(c.w_wind4 >= 0) || !(c.w_recto >= 0)) { fprintf(stderr, "fit: --wind4-weight and --w-recto must be >= 0\n"); return 2; }
    /* CT in the fitter (fields with ct / cs planes: ufsm ct-fields, predict --fit-ct): papyrus / void parity and the CT sheet spacing */
    c.w_ct = (float)atof(opt(argc, argv, "--w-ct", "0")); c.ct_pow = (float)atof(opt(argc, argv, "--ct-pow", "1")); c.ct_phi = (float)atof(opt(argc, argv, "--ct-phi", "0"));
    c.ct_floor = (float)atof(opt(argc, argv, "--ct-floor", "0")); c.ct_radius_um = atof(opt(argc, argv, "--ct-radius-um", "150")); c.ct_flood = atoi(opt(argc, argv, "--ct-flood", "0"));
    c.w_cspacing = (float)atof(opt(argc, argv, "--w-cspacing", "0")); c.cs_tol = (float)atof(opt(argc, argv, "--cs-tol", "0.25"));
    fit_fields_ct_planes((c.w_ct > 0 ? 1 : 0) | (c.w_cspacing > 0 ? 2 : 0));   /* load only the CT planes the terms read */
    if (!(c.w_ct >= 0) || !(c.ct_pow >= 0) || !(c.ct_floor >= 0) || !(c.ct_radius_um > 0) || !(c.w_cspacing >= 0) || !(c.cs_tol > 0)) { fprintf(stderr, "fit: --w-ct, --ct-pow, --ct-floor, --w-cspacing >= 0, --ct-radius-um and --cs-tol > 0\n"); return 2; }
    if (!(c.w_pgrad >= 0) || !(c.pgrad_delta > 0) || c.unwrap_voxels < -1 || c.unwrap_voxels > 1 || !(c.phase_robust >= 0) || !(c.conf_pow > 0) || !(c.bend >= 0) || !(c.w_phase >= 0) || !(c.w_normal >= 0) || !(c.w_density >= 0) || !(c.w_smooth >= 0)) {
        fprintf(stderr, "fit: --phase-robust, --bend and the weights must be >= 0, --conf-pow and --pgrad-delta > 0, --unwrap-voxels -1, 0 or 1\n"); return 2;
    }
    c.uw_split = atof(opt(argc, argv, "--uw-split", "0")); c.uw_tile = atoi(opt(argc, argv, "--uw-tile", "256")); c.uw_pdv = atof(opt(argc, argv, "--uw-pdv", "0")); c.uw_min_conf = atof(opt(argc, argv, "--uw-min-conf", "0.3"));
    { const char *uf = opt(argc, argv, "--uw-flood", "serial"); c.uw_msf = !strcmp(uf, "msf"); if (!c.uw_msf && strcmp(uf, "serial")) { fprintf(stderr, "fit: --uw-flood serial|msf\n"); return 2; } }   /* the voxel unwrapping: the serial flood or the parallel spanning forest (src/fit.c uw_msf) */
    /* local slip repair (fit_cfg uw_repair): pieces of uw_repair^3 tiles, least-cost integer shifts from their votes and the evidence */
    c.uw_repair = atoi(opt(argc, argv, "--uw-repair", "0")); c.uw_rep_w4 = (float)atof(opt(argc, argv, "--uw-rep-w4", "0"));
    c.uw_rep_geo = (float)atof(opt(argc, argv, "--uw-rep-geo", "0")); c.uw_rep_gd = atoi(opt(argc, argv, "--uw-rep-gd", "8"));
    c.uw_rep_cs = (float)atof(opt(argc, argv, "--uw-rep-cs", "0")); c.uw_rep_dump = opt(argc, argv, "--uw-rep-dump", nullptr);
    static fib_set fibs;   /* --fibers FILE (ufsm fiber-export for this fields dir) --w-fib W: same-fibre pair term in the fit stages */
    c.w_fib = (float)atof(opt(argc, argv, "--w-fib", "0"));
    if (*opt(argc, argv, "--fibers", "")) {
        if (fib_load(opt(argc, argv, "--fibers", ""), &fibs)) { fprintf(stderr, "fit: cannot read --fibers %s\n", opt(argc, argv, "--fibers", "")); return 1; }
        c.fib = fibs.pts; c.fib_off = fibs.off; c.fib_n = fibs.n;
        if (!(c.w_fib > 0)) fprintf(stderr, "fit: --fibers without --w-fib W > 0 (unused)\n");
    }
    c.uw_rep_qpow = (float)atof(opt(argc, argv, "--uw-rep-qpow", "3")); c.uw_rep_stretch = (float)atof(opt(argc, argv, "--uw-rep-stretch", "0.5"));
    c.uw_rep_sr = atoi(opt(argc, argv, "--uw-rep-sr", "1")); c.uw_rep_seed = (float)atof(opt(argc, argv, "--uw-rep-seed", *opt(argc, argv, "--seed-model", "") ? "16" : "0"));   /* (fit --slab chaining: off unless given) */ c.uw_rep_geo_init = atoi(opt(argc, argv, "--uw-rep-geo-init", "0")); c.uw_rep_off = atoi(opt(argc, argv, "--uw-rep-off", "0")); c.uw_rep_cons = atoi(opt(argc, argv, "--uw-rep-cons", "1")); c.phase_stretch = (float)atof(opt(argc, argv, "--phase-stretch", "0")); c.phase_stretch_r = atoi(opt(argc, argv, "--phase-stretch-r", "1"));
    if (!(c.phase_stretch >= 0) || c.phase_stretch_r < 0 || c.phase_stretch_r > 4) { fprintf(stderr, "fit: --phase-stretch >= 0, --phase-stretch-r 0..4\n"); return 2; }
    /* embedding evidence for the slip repair (src/fit_emb.h): --emb GROUP (a dinovol-embed output group at the fields' scan level, or
       --emb-scale S embedding voxels per native voxel) [--emb-output tokens_unit] --uw-rep-emb W [--uw-rep-emb-d 16,48 native voxels]
       [--uw-rep-emb-m0 0.3 cosine margin for full weight] [--uw-rep-emb-rmax 2] [--uw-rep-emb-sub 2 field voxels] [--emb-phi 0]
       [--emb-ct ROOT KEY --emb-papyrus L [--cache DIR]]: the papyrus gate, sheet centres snapped onto CT >= L (CT at the fields' native
       level; no gain on the 10-25 emb-calib boxes, src/fit_emb.h) */
    c.uw_rep_emb = (float)atof(opt(argc, argv, "--uw-rep-emb", "0")); c.uw_rep_emb_m0 = (float)atof(opt(argc, argv, "--uw-rep-emb-m0", "0.3"));
    c.uw_rep_emb_rmax = atoi(opt(argc, argv, "--uw-rep-emb-rmax", "2")); c.uw_rep_emb_sub = atoi(opt(argc, argv, "--uw-rep-emb-sub", "2")); c.emb_phi = (float)atof(opt(argc, argv, "--emb-phi", "0"));
    if (*opt(argc, argv, "--uw-rep-emb-d", "")) { c.uw_rep_emb_nd = 0; char *t = strdup(opt(argc, argv, "--uw-rep-emb-d", "")); for (char *q = strtok(t, ","); q && c.uw_rep_emb_nd < 4; q = strtok(nullptr, ",")) c.uw_rep_emb_d[c.uw_rep_emb_nd++] = atoi(q); free(t); }
    if (!(c.uw_rep_emb >= 0) || !(c.uw_rep_emb_m0 > 0) || c.uw_rep_emb_rmax < 1 || c.uw_rep_emb_rmax > 8 || c.uw_rep_emb_sub < 1) { fprintf(stderr, "fit: --uw-rep-emb >= 0, --uw-rep-emb-m0 > 0, --uw-rep-emb-rmax 1..8, --uw-rep-emb-sub >= 1\n"); return 2; }
    if (c.uw_repair < 0 || !(c.uw_rep_w4 >= 0) || !(c.uw_rep_geo >= 0) || !(c.uw_rep_cs >= 0) || c.uw_rep_gd < 1 || !(c.uw_rep_qpow > 0) || !(c.uw_rep_stretch >= 0) || c.uw_rep_sr < 0 || c.uw_rep_sr > 4 || !(c.uw_rep_seed >= 0)) { fprintf(stderr, "fit: --uw-repair >= 0, --uw-rep-* weights >= 0, --uw-rep-gd >= 1\n"); return 2; }
    c.core = (float)atof(opt(argc, argv, "--core", "3"));   /* no data terms within core x pitch of the axis */
    c.w_wind4 = (float)atof(opt(argc, argv, "--wind4-weight", "0"));   /* the winding mod 4 term of task unwrap2 fields (dz dy dx dq planes; 0 off: weight 4 cost fit-val, 0.5-1 neutral) */
    c.analytic = atoi(opt(argc, argv, "--analytic", "1")); c.packed_um = atof(opt(argc, argv, "--packed-um", "0")); c.batch = atoi(opt(argc, argv, "--batch", "2000000")); c.knot_dz = atoi(opt(argc, argv, "--knot-dz", "1024")); c.phase_stage = atoi(opt(argc, argv, "--phase-stage", "4"));
    if (*opt(argc, argv, "--lattice", "")) { c.nstage = 0; char *t = strdup(opt(argc, argv, "--lattice", "")); for (char *q = strtok(t, ","); q && c.nstage < 8; q = strtok(nullptr, ",")) c.lattice[c.nstage++] = atoi(q); free(t); }
    if (*opt(argc, argv, "--steps", "")) { int k = 0; char *t = strdup(opt(argc, argv, "--steps", "")); for (char *q = strtok(t, ","); q && k < 8; q = strtok(nullptr, ",")) c.steps[k++] = atoi(q); for (; k < c.nstage; k++) c.steps[k] = c.steps[k - 1]; free(t); }
    fit_fields_skip_planes((c.w_recto > 0 ? 0 : 1) | (c.w_wind4 > 0 || c.uw_rep_w4 > 0 ? 0 : 2));   /* load only the optional planes the terms read */
    /* the GPU context (~2 s) comes up on a thread while the fields load; the main thread selects the device again after the join */
    g_fit_gdev = atoi(opt(argc, argv, "--gpu", "0"));
    if (!c.cpu) { g_fit_gthr = pthread_create(&g_fit_gth, nullptr, fit_gpu_init_thread, nullptr) == 0; if (!g_fit_gthr) nn_init(g_fit_gdev); }
    if (*opt(argc, argv, "--ref-model", "")) { if (g_fit_ref.on) { pthread_join(g_fit_ref.th, nullptr); g_fit_ref.on = 0; } g_fit_ref.dir = opt(argc, argv, "--ref-model", ""); g_fit_ref.s = nullptr; g_fit_ref.on = pthread_create(&g_fit_ref.th, nullptr, fit_ref_thread, nullptr) == 0; }
    const double t0 = now();
    const char *min = opt(argc, argv, "--model", nullptr);   /* score / export a saved model or set instead of fitting */
    const int slab = atoi(opt(argc, argv, "--slab", "0")), overlap = atoi(opt(argc, argv, "--overlap", "256"));
    fit_fields_buf fb; memset(&fb, 0, sizeof fb);
    {   /* the native voxel size first: default lattices and knots are physical, given in 2.4 um voxels, scaled to the fields' native voxels */
        fit_fields_buf g0; if (fit_fields_load(dir, 0, 0, 0, &g0)) { fprintf(stderr, "fit: cannot load the fields in %s\n", dir); return 1; }
        c.um = g0.um; fb.h = g0.h; fit_fields_free(&g0);
        if (!*opt(argc, argv, "--knot-dz", "")) c.knot_dz = fit_scale_L(1024, 2.4 / c.um);
        if (!*opt(argc, argv, "--lattice", "")) for (int k = 0; k < c.nstage; k++) c.lattice[k] = fit_scale_L(c.lattice[k], 2.4 / c.um);
    }
    /* GPU fits stream the fields: the normals straight to the device, the gradient stages' pooled fields made while loading, ~4 B per cell
       on the host instead of 10 (UFSM_FIT_NOSTREAM=1: off; the options whose host code reads the normals or alters the phase planes
       before the upload fall back to the whole host fields) */
    fit_stream fst; memset(&fst, 0, sizeof fst);
    /* --coarse-pool P (> 1): the coarse stage of a coarse-to-fine unwrap: the fields max / mean pooled by P per axis (pooled phase carries no
       information) and only the gradient terms (normal, density: the integrated turn count; no phase stage), e.g. a whole scroll in one fit
       whose model is the --ref-model of the blocks' fine fits */
    const int cpool = atoi(opt(argc, argv, "--coarse-pool", "0"));
    if (cpool > 1) {
        if (min || slab > 0) { fprintf(stderr, "fit: --coarse-pool goes without --model / --slab\n"); return 2; }
        c.phase_stage = c.nstage; c.unwrap_voxels = -1;
    }
    const int use_stream = cpool <= 1 && !c.cpu && !min && slab <= 0 && !ufsm_env_on("UFSM_FIT_NOSTREAM") && !(c.w_pgrad > 0) && !(c.w_cspacing > 0) && !(c.bend > 0) &&
                           !(c.phase_stretch > 0) && !(c.w_ct > 0 && c.ct_flood) && !*opt(argc, argv, "--axis-normal", "");
    struct timespec tl0, tl1; clock_gettime(CLOCK_MONOTONIC, &tl0);
    int cached = 0;   /* fit --sweep: the streamed fields of an earlier run of this process (same directory, planes and pooling factors) */
    if (use_stream) {
        int fps[8]; const int nfp = fit_stream_fps(&c, fb.h, fps);
        const int kskip = (c.w_recto > 0 ? 0 : 1) | (c.w_wind4 > 0 || c.uw_rep_w4 > 0 ? 0 : 2), kct = (c.w_ct > 0 ? 1 : 0) | (c.w_cspacing > 0 ? 2 : 0);
        if (g_fit_sweep && g_fcache.valid && !strcmp(g_fcache.dir, dir) && g_fcache.kskip == kskip && g_fcache.kct == kct && g_fcache.nfp == nfp && !memcmp(g_fcache.fps, fps, nfp * sizeof *fps)) {
            fb = g_fcache.fb; fst = g_fcache.fst; cached = 1; fit_gpu_join();
            if (c.verbose) fprintf(stderr, "fit: the fields of the previous sweep run reused\n");
        } else {
            if (g_fcache.valid) { fit_stream_free(&g_fcache.fst); fit_fields_free(&g_fcache.fb); g_fcache.valid = 0; }
            if (fit_fields_load_stream(dir, &fb, &fst, fps, nfp, fit_gpu_join)) { fprintf(stderr, "fit: cannot load the fields in %s\n", dir); return 1; }
            if (g_fit_sweep) {
                snprintf(g_fcache.dir, sizeof g_fcache.dir, "%s", dir); g_fcache.kskip = kskip; g_fcache.kct = kct; g_fcache.nfp = nfp; memcpy(g_fcache.fps, fps, sizeof fps);
                g_fcache.fb = fb; g_fcache.fst = fst; g_fcache.valid = 1; cached = 1;
            }
        }
        fst.drop_pooled = c.dir != 0 && !cached;   /* (one direction: the pooled planes are not needed after the gradient stages; a sweep keeps them) */
        if (ufsm_env_on("UFSM_FIT_STREAM_VERIFY")) {   /* (tests: the streamed device / pooled planes against the whole host load) */
            fit_fields_buf full; if (fit_fields_load(dir, 0, 0, 1, &full)) return 1;
            const int bad = fit_stream_verify(&fst, &full.f); fit_fields_free(&full);
            fprintf(stderr, "fit: stream verify: %s (%d pooling factors)\n", bad ? "MISMATCH" : "identical", fst.nfp);
            if (bad) return 1;
        }
        c.stream = &fst;
    } else if (fit_fields_load(dir, 0, 0, min || slab > 0 ? 0 : cpool > 1 ? cpool : 1, &fb)) { fprintf(stderr, "fit: cannot load the fields in %s\n", dir); return 1; }
    clock_gettime(CLOCK_MONOTONIC, &tl1);
    fit_gpu_join();
    if (c.verbose) fprintf(stderr, "fit: fields %dx%dx%d loaded in %.1fs\n", fb.shape[0], fb.shape[1], fb.shape[2], (double)(tl1.tv_sec - tl0.tv_sec) + (tl1.tv_nsec - tl0.tv_nsec) * 1e-9);   /* (slab runs load per slab; a given model needs the geometry only) */
    emb_vol *embv = nullptr;
    if (opt(argc, argv, "--emb", nullptr) && c.uw_rep_emb > 0) {   /* the embedding over the fields' box (+ the farthest partner distance) */
        int pad = 0; for (int k = 0; k < c.uw_rep_emb_nd; k++) if (c.uw_rep_emb_d[k] > pad) pad = c.uw_rep_emb_d[k];
        const long long en[3] = {(long long)ceil(fb.shape[0] * fb.h), (long long)ceil(fb.shape[1] * fb.h), (long long)ceil(fb.shape[2] * fb.h)};
        embv = emb_vol_open(opt(argc, argv, "--emb", ""), opt(argc, argv, "--emb-output", "tokens_unit"), fb.origin, en, atof(opt(argc, argv, "--emb-scale", "1")), pad + 32);
        if (!embv) return 1;
        for (int i = 2; i + 2 < argc; i++) if (!strcmp(argv[i], "--emb-ct")) {   /* the papyrus gate on the votes (fit_emb.h): CT at the fields' native level */
            const int lv = atoi(opt(argc, argv, "--emb-papyrus", "0")); store *cs = store_open(argv[i + 1]); z3 *cz = cs ? z3_open(cs, argv[i + 2], opt(argc, argv, "--cache", nullptr)) : nullptr;
            if (!cz || lv <= 0 || emb_vol_papyrus_zarr(embv, cz, fb.origin, en, pad + 32, lv)) { fprintf(stderr, "fit: --emb-ct ROOT KEY needs --emb-papyrus L > 0 and a readable CT\n"); return 1; }
            z3_close(cz);
        }
        c.emb = embv;
    } else if (opt(argc, argv, "--emb", nullptr)) fprintf(stderr, "fit: --emb without --uw-rep-emb W > 0 (unused)\n");
    if (*opt(argc, argv, "--axis-normal", "") && !min && !slab) {
        /* local boxes away from the scroll's axis (crushed scrolls): replace the axis by a vertical line D native voxels from the box centre
           along the box's mean sheet normal (top eigenvector of the material-weighted mean n n^T), so the spiral term is nearly flat */
        const double D = atof(opt(argc, argv, "--axis-normal", "0")); const fit_fields *f = &fb.f; const size_t n = (size_t)f->D * f->H * f->W;
        double T[6] = {0}, sw = 0;   /* zz yy xx zy zx yx */
        for (size_t i = 0; i < n; i++) if (f->w[i]) {
            const double w = f->w[i] / 255.0; sw += w;
            for (int k = 0; k < 3; k++) T[k] += w * f->t[k][i] / 255.0;
            for (int k = 3; k < 6; k++) T[k] += w * (f->t[k][i] / 255.0 - 0.5);
        }
        if (sw <= 0) { fprintf(stderr, "fit: --axis-normal: no material in the box\n"); return 1; }
        const double M[3][3] = {{T[0], T[3], T[4]}, {T[3], T[1], T[5]}, {T[4], T[5], T[2]}};
        double v[3] = {0.3, 0.7, 0.6};
        for (int it = 0; it < 100; it++) {
            double u[3] = {0, 0, 0}; for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) u[a] += M[a][b] * v[b];
            const double l = sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]); if (l < 1e-12) break; for (int a = 0; a < 3; a++) v[a] = u[a] / l;
        }
        double ny = v[1], nx = v[2]; const double lyx = sqrt(ny * ny + nx * nx);
        if (lyx < 0.3) fprintf(stderr, "fit: --axis-normal: the mean normal (%.2f %.2f %.2f) is mostly along z\n", v[0], v[1], v[2]);
        if (lyx > 1e-6) { ny /= lyx; nx /= lyx; } else { ny = 1; nx = 0; }
        const double cy = f->H * f->h / 2, cx = f->W * f->h / 2; const int na = (int)(f->D * f->h) + 1;
        for (int z = 0; z < na; z++) { fb.cy[z] = (float)(cy - D * ny); fb.cx[z] = (float)(cx - D * nx); }
        fprintf(stderr, "fit: --axis-normal: mean normal z %.3f y %.3f x %.3f; axis at box-local (y %.1f, x %.1f), %.0f native voxels away\n",
                v[0], v[1], v[2], cy - D * ny, cx - D * nx, D);
    }
    fit_model *seedm = nullptr, *initm = nullptr;   /* --seed-model / --init-model: a neighbouring (separately fitted) box's model */
    if (*opt(argc, argv, "--seed-model", "")) {
        if (!(seedm = fit_load(opt(argc, argv, "--seed-model", "")))) { fprintf(stderr, "fit: cannot load --seed-model %s\n", opt(argc, argv, "--seed-model", "")); return 1; }
        if (seedm->origin[1] != fb.origin[1] || seedm->origin[2] != fb.origin[2]) { fprintf(stderr, "fit: --seed-model must share the fields' y / x origin (z slabs)\n"); return 2; }
        double z0 = -1e30, z1 = 1e30; const char *zs = opt(argc, argv, "--seed-z", "");
        if (*zs && sscanf(zs, "%lf,%lf", &z0, &z1) != 2) { fprintf(stderr, "fit: --seed-z Z0,Z1\n"); return 2; }
        c.seed = seedm; c.seed_off = (double)(fb.origin[0] - seedm->origin[0]); c.seed_z0 = z0; c.seed_z = z1;
        if (c.dir == 0) c.dir = seedm->dir;
        fprintf(stderr, "fit: integers on box-local native z [%.0f, %.0f) from the seed model (z offset %.0f), dir %+d\n", z0, z1, c.seed_off, c.dir);
    }
    if (*opt(argc, argv, "--init-model", "")) {
        if (!(initm = fit_load(opt(argc, argv, "--init-model", "")))) { fprintf(stderr, "fit: cannot load --init-model %s\n", opt(argc, argv, "--init-model", "")); return 1; }
        c.init = initm; for (int d = 0; d < 3; d++) c.init_off[d] = (double)(fb.origin[d] - initm->origin[d]);
        if (c.dir == 0) c.dir = initm->dir;
    }
    fit_set *refm = nullptr;   /* --ref-model: a global coarse model / set (coarse-to-fine: the integers tied to it) */
    if (*opt(argc, argv, "--ref-model", "")) {
        if (g_fit_ref.on) { pthread_join(g_fit_ref.th, nullptr); g_fit_ref.on = 0; refm = g_fit_ref.s; } else refm = fit_set_load(opt(argc, argv, "--ref-model", ""));
        if (!refm) { fprintf(stderr, "fit: cannot load --ref-model %s\n", opt(argc, argv, "--ref-model", "")); return 1; }
        if (seedm || slab > 0) { fprintf(stderr, "fit: --ref-model goes without --seed-model / --slab\n"); return 2; }
        c.ref = refm; for (int d = 0; d < 3; d++) c.ref_off[d] = (double)(fb.origin[d] - refm->global->origin[d]);
        c.ref_snap = atoi(opt(argc, argv, "--ref-snap", "0")); c.uw_rep_ref = (float)atof(opt(argc, argv, "--uw-rep-ref", "0"));
        if (!(c.uw_rep_ref >= 0)) { fprintf(stderr, "fit: --uw-rep-ref >= 0\n"); return 2; }
        if (c.dir == 0) c.dir = refm->global->dir;
        else if (c.dir != refm->global->dir) { fprintf(stderr, "fit: --dir %+d differs from the --ref-model's %+d\n", c.dir, refm->global->dir); return 2; }
        fprintf(stderr, "fit: ref model %s (box offset %.0f %.0f %.0f native): %s%s\n", opt(argc, argv, "--ref-model", ""), c.ref_off[0], c.ref_off[1], c.ref_off[2],
                c.ref_snap ? "integers snapped to it" : "flood seed from it", c.uw_rep_ref > 0 ? ", repair pieces anchored to it" : "");
    }
    /* --phase-gate 0|1|2 (with --ref-model; default 1 with --ref-snap 1; src/phasegate.c): the network phase replaced by the ref's where it does not rotate with the
       density's sheets (1: gated, --pg-lo 0.3 --pg-hi 0.6 --pg-R 6 --pg-conf 0.5; 2: the ref's phase everywhere). Crushed scrolls. */
    const int pgm = atoi(opt(argc, argv, "--phase-gate", refm && c.ref_snap == 1 && !min ? "1" : "0"));   /* (default on for the c2f snap: --ref-model --ref-snap 1) */
    if (pgm > 0) {
        if (!refm || min) { fprintf(stderr, "fit: --phase-gate needs --ref-model (and no --model)\n"); return 2; }
        if (cached) { fprintf(stderr, "fit: --phase-gate goes without --sweep\n"); return 2; }
        struct timespec tg0, tg1; clock_gettime(CLOCK_MONOTONIC, &tg0);
        const fit_pgate_cfg pg = {.mode = pgm == 2, .R = atoi(opt(argc, argv, "--pg-R", "6")), .lo = atof(opt(argc, argv, "--pg-lo", "0.3")), .hi = atof(opt(argc, argv, "--pg-hi", "0.6")),
                                  .rc = atof(opt(argc, argv, "--pg-conf", "0.5"))};
        const int D = fb.shape[0], H = fb.shape[1], W = fb.shape[2], na = (int)(D * fb.h) + 1; const size_t plane = (size_t)H * W, n = (size_t)D * plane;
        const int BZ = 16, dev = use_stream && !fb.f.t[1];
        uint8_t *tb = dev ? malloc((size_t)3 * BZ * plane) : nullptr; double st[4] = {0}; int bad = dev && !tb;
        fit_set_dev *gd = c.cpu ? nullptr : fit_set_upload(refm);   /* (W_ref on the GPU; null: the CPU grid) */
        for (int z0 = 0; z0 < D && !bad; z0 += BZ) {
            const int nz = z0 + BZ <= D ? BZ : D - z0; const size_t off = (size_t)z0 * plane;
            fit_fields blk = fb.f; blk.D = nz; blk.pc = fb.a[0] + off; blk.ps = fb.a[1] + off; blk.sp = fb.a[8] + off; blk.w = fb.a[9] + off;
            for (int k = 0; k < 6; k++) blk.t[k] = fb.f.t[k] ? fb.f.t[k] + off : nullptr;
            if (dev) { const int tk[3] = {1, 2, 5}; for (int q = 0; q < 3; q++) { nn_d2h(tb + (size_t)q * BZ * plane, fst.dfield + (size_t)(2 + tk[q]) * n + off, (size_t)nz * plane); blk.t[tk[q]] = tb + (size_t)q * BZ * plane; } }
            if (fit_phase_gate(&blk, z0, fb.f.cy, fb.f.cx, na, refm, gd, c.ref_off, &pg, fb.a[0] + off, fb.a[1] + off, nullptr, st)) bad = 1;
            if (!bad && dev) { nn_h2d(fst.dfield + off, fb.a[0] + off, (size_t)nz * plane); nn_h2d(fst.dfield + n + off, fb.a[1] + off, (size_t)nz * plane); }
        }
        free(tb); if (gd) fit_set_dev_free(gd);
        if (bad) { fprintf(stderr, "fit: --phase-gate failed (memory / ref grid)\n"); return 1; }
        clock_gettime(CLOCK_MONOTONIC, &tg1);
        fprintf(stderr, "fit: phase gate (%s): mean rotation q %.3f, the ref's phase on %.3f of %.0f material cells (%.1f s)\n", pg.mode ? "ref everywhere" : "gated",
                st[3] ? st[2] / st[3] : 0, st[0] ? st[1] / st[0] : 0, st[0], (tg1.tv_sec - tg0.tv_sec) + 1e-9 * (tg1.tv_nsec - tg0.tv_nsec));
    }
    fit_set *fs = nullptr;
    if (min) fs = fit_set_load(min);
    else if (slab > 0) fs = fit_run_slabs(dir, &c, slab, overlap);
    else { fs = calloc(1, sizeof *fs); fs->global = fit_run(&fb.f, &c); if (!fs->global) { free(fs); fs = nullptr; } }
    if (!fs) { fprintf(stderr, "fit: no model%s%s\n", min ? " in " : "", min ? min : ""); return 1; }
    const fit_model *m = fs->global;
    if (!min && !slab) for (int d = 0; d < 3; d++) fs->global->origin[d] = fb.origin[d];
    if (!min) { fs->global->um = fb.um; for (int k = 0; k < fs->n; k++) fs->slab[k]->um = fb.um; }
    const long long org[3] = {m->origin[0], m->origin[1], m->origin[2]};
    const int shp[3] = {fb.shape[0], fb.shape[1], fb.shape[2]}; const double h = fb.h;
    c.stream = nullptr;   /* the fields are done with: the exports and the score read the model and the axis (--winding-gate reloads them) */
    if (!cached) { fit_stream_free(&fst); for (int k = 0; k < 17; k++) { free(fb.a[k]); fb.a[k] = nullptr; } }
    else for (int k = 0; k < 17; k++) fb.a[k] = nullptr;   /* (the sweep cache owns them and the axis) */
    fprintf(stderr, "fit: dir %+d, %d slab model(s), global lattice %d (%dx%dx%d), loss %.5f in %.1fs\n", m->dir, fs->n, m->L, m->lz, m->ly, m->lx, m->loss[0], now() - t0);
    const char *mout = opt(argc, argv, "--out", nullptr);
    if (mout && (fs->n ? fit_set_save(fs, mout) : fit_save(fs->global, mout))) { fprintf(stderr, "fit: cannot save %s\n", mout); return 1; }
    if (mout && !min) {   /* the fit's options beside the model (reproducibility) */
        char cp[1400]; snprintf(cp, sizeof cp, "%s/fit-config.json", mout); FILE *cf = fopen(cp, "w");
        if (!cf) { fprintf(stderr, "fit: cannot save %s\n", cp); return 1; }
        fprintf(cf, "{\"version\":1,\"w_phase\":%.9g,\"w_normal\":%.9g,\"w_density\":%.9g,\"w_smooth\":%.9g,\"w_pgrad\":%.9g,\"pgrad_delta\":%.9g,\"w_wind4\":%.9g,\"w_recto\":%.9g,"
                    "\"w_ct\":%.9g,\"ct_pow\":%.9g,\"ct_phi\":%.9g,\"ct_floor\":%.9g,\"ct_radius_um\":%.9g,\"ct_flood\":%d,\"w_cspacing\":%.9g,\"cs_tol\":%.9g,"
                    "\"phase_robust_turns\":%.9g,\"conf_pow\":%.9g,\"bend\":%.9g,\"unwrap_voxels\":%d,\"uw_min_conf\":%.9g,\"uw_repair\":%d,\"uw_rep_w4\":%.9g,\"uw_rep_geo\":%.9g,\"uw_rep_gd\":%d,\"uw_rep_cs\":%.9g,\"uw_rep_qpow\":%.9g,\"uw_rep_stretch\":%.9g,\"uw_rep_sr\":%d,\"uw_rep_emb\":%.9g,\"phase_stretch\":%.9g,\"phase_stretch_r\":%d,\"phase_stage\":%d,\"slab\":%d,\"cpu\":%d,\"native_um\":%.9g,\"lattice\":[",
                c.w_phase, c.w_normal, c.w_density, c.w_smooth, c.w_pgrad, c.pgrad_delta, c.w_wind4, c.w_recto,
                c.w_ct, c.ct_pow, c.ct_phi, c.ct_floor, c.ct_radius_um, c.ct_flood, c.w_cspacing, c.cs_tol, c.phase_robust, c.conf_pow, c.bend, c.unwrap_voxels, c.uw_min_conf,
                c.uw_repair, c.uw_rep_w4, c.uw_rep_geo, c.uw_rep_gd, c.uw_rep_cs, c.uw_rep_qpow, c.uw_rep_stretch, c.uw_rep_sr, (double)c.uw_rep_emb, (double)c.phase_stretch, c.phase_stretch_r, c.phase_stage, slab, c.cpu, c.um);
        for (int k = 0; k < c.nstage; k++) fprintf(cf, "%s%d", k ? "," : "", c.lattice[k]);
        fprintf(cf, "]%s}\n", c.uw_msf ? ",\"uw_flood\":\"msf\"" : ""); if (fclose(cf)) { fprintf(stderr, "fit: cannot save %s\n", cp); return 1; }
    }
    const char *wout = opt(argc, argv, "--winding", nullptr);
    if (wout) {   /* the per-voxel winding number W * 256 (uint16, exact volcomp typed chunks) on a grid of --winding-step native voxels;
                     --winding-mod14 1: instead uint8 round(18 U) mod 252 (U = W + dir theta / 2 pi, the codes' q; 255 unknown), the dense winding raster kind of
                     phase-raster --winding that wind-assign --kind winding reads (a fit as the reference for other surfaces: fit QA) */
        const int ws = atoi(opt(argc, argv, "--winding-step", "4")); const double um = atof(opt(argc, argv, "--um", "0")) > 0 ? atof(opt(argc, argv, "--um", "0")) : fb.um;
        const int m14 = atoi(opt(argc, argv, "--winding-mod14", "0"));   /* 2: band codes 1 + round(18 U) mod 252 (dino-train codes; 0 / 255 unlabelled);
                                                                            3: the stored-phase codes round(252 frac(W)) mod 252 (src/band.h PHASE_*: sheets at 0; 255 unknown),
                                                                               --phase-sign -1 negates W (train sources "phase" pseudo labels) */
        const int psg = atoi(opt(argc, argv, "--phase-sign", "1")) < 0 ? -1 : 1;
        const int64_t gs[3] = {(int64_t)ceil(shp[0] * h / ws), (int64_t)ceil(shp[1] * h / ws), (int64_t)ceil(shp[2] * h / ws)};
        /* --winding-global Z,Y,X (uint8 kinds): write into an array of that shape on the global grid of --winding-step native voxels, 512^3
           shards (the box origin must sit on the shard grid: org % (512 step) == 0); several box fits fill one per-scan label zarr */
        int64_t gsh[3] = {0, 0, 0}, so[3] = {0, 0, 0}; int BK = 128; const char *wgl = opt(argc, argv, "--winding-global", nullptr);
        if (wgl) {
            long long v[3]; if (!m14 || sscanf(wgl, "%lld,%lld,%lld", &v[0], &v[1], &v[2]) != 3) { fprintf(stderr, "fit: --winding-global Z,Y,X needs --winding-mod14 1|2|3\n"); return 2; }
            BK = 512;
            for (int d = 0; d < 3; d++) {
                gsh[d] = v[d];
                if (org[d] < 0 || org[d] % (512LL * ws)) { fprintf(stderr, "fit: --winding-global: box origin %lld (axis %d) not on the 512 x %d native-voxel shard grid\n", (long long)org[d], d, ws); return 2; }
                so[d] = org[d] / (512LL * ws);
            }
        }
        char lv[32], ld[1400], at[600]; z3w_level_name(um * ws, lv, sizeof lv); snprintf(ld, sizeof ld, "%s/%s", wout, lv);
        snprintf(at, sizeof at, "{\"ufsm\":{\"content\":\"%s\",\"dir\":%d,\"origin_zyx\":[%lld,%lld,%lld],\"step\":%d}}",
                 m14 == 3 ? "stored phase: round(252 frac(W)) mod 252 (uint8; 255 unknown; sheets at integer W; src/band.h PHASE_*)" : m14 ? "winding q mod 14: round(18 U) mod 252 (uint8; 255 unknown; U = W + dir theta / 2 pi, the band codes' q)" : "winding coordinate W * 256 (uint16; W = U - dir theta / 2 pi, sheets at integer W)",
                 m->dir, wgl ? 0 : org[0], wgl ? 0 : org[1], wgl ? 0 : org[2], ws);
        z3w *zw = m14 ? z3w_create(ld, wgl ? gsh : gs, BK, 0.f, 255, at) : z3w_create_typed(ld, gs, 128, 1, 0.f, 0.0, at);
        if (!zw || z3w_write_group(wout, (const double[]){um * ws}, 1, "ufsm-winding", at)) { fprintf(stderr, "fit: %s\n", z3w_error()); return 1; }
        /* --winding-gate CONF,ERR,MAT (mod14 codes only; pseudo labels): a voxel keeps its code only where the field cell's network phase
           confidence >= CONF, the fit's W agrees with that phase within ERR turns and the material fraction >= MAT; else 255 (unknown) */
        double gc = 0, ge = 0, gm = 0; const int gate = m14 && *opt(argc, argv, "--winding-gate", "") && sscanf(opt(argc, argv, "--winding-gate", ""), "%lf,%lf,%lf", &gc, &ge, &gm) == 3;
        fit_fields_buf gb; memset(&gb, 0, sizeof gb); const fit_fields_buf *gf = &fb;
        if (gate && !fb.a[0]) { if (fit_fields_load(dir, 0, 0, 1, &gb)) { fprintf(stderr, "fit: --winding-gate: cannot load the fields\n"); return 1; } gf = &gb; }
        fit_fields_buf ab; memset(&ab, 0, sizeof ab); fit_fields alt;
        if (gate && getenv("UFSM_WCHECK_PCPS") && *getenv("UFSM_WCHECK_PCPS")) {   /* (check only: the phase planes pc / ps of another fields dir on the same cells, e.g. count-field output) */
            if (fit_fields_load(getenv("UFSM_WCHECK_PCPS"), 0, 0, 1, &ab) || ab.f.D != gf->f.D || ab.f.H != gf->f.H || ab.f.W != gf->f.W) { fprintf(stderr, "fit: UFSM_WCHECK_PCPS: cannot load or shape differs\n"); return 1; }
            alt = gf->f; alt.pc = ab.f.pc; alt.ps = ab.f.ps; ((fit_fields_buf *)gf)->f = alt;
        }
        float *wg = gate || m14 == 3 ? malloc((size_t)BK * BK * BK * 4) : nullptr; long long gn = 0, gk = 0, gskip = 0; double gcs = 0, gcw = 0;
        /* --winding-check ROOT KEY UM (with the gate): the gated codes against band codes (1 + round(18 q) mod 252) of UM um voxels on the
           global grid: the code difference histogram of the gated voxels and of all voxels with a band code, best gauge, share within 0.5 turn */
        const int wci = arg_index(argc, argv, "--winding-check"); z3 *gtz = nullptr; double gtr = 0; long long hk[252] = {0}, ha[252] = {0}; uint8_t *gtb = nullptr; long long lbest[2] = {0, 0}, ltot[2] = {0, 0};
        /* sub-turn histograms (18 codes per turn) vs the band codes, [packed sp < 10.4 native voxels (100 um at 9.6)][kind]: kind 0 the label
           phase frac(W) on gated voxels, 1 the network phase on the same voxels, 2 the network phase on every material voxel with a band code */
        long long hph[2][3][18]; memset(hph, 0, sizeof hph);
        const int gtph = getenv("UFSM_WCHECK_PHASE") && atoi(getenv("UFSM_WCHECK_PHASE")) > 0;
        if (gate) fprintf(stderr, "fit: --winding: model origin %lld %lld %lld, gate fields origin %lld %lld %lld (%dx%dx%d cells, h %d)\n", org[0], org[1], org[2],
                          gf->origin[0], gf->origin[1], gf->origin[2], gf->f.D, gf->f.H, gf->f.W, gf->h);
        double dOff[3] = {0, 0, 0};   /* the gate's fields a sub-box of the model's: the grid sits on the fields (offsets in native voxels) */
        if (gate && getenv("UFSM_WCHECK_SUBBOX")) for (int d = 0; d < 3; d++) dOff[d] = (double)(gf->origin[d] - org[d]);
        /* --winding-skip ROOT KEY UM (up to 2, mod14 codes): voxels where that raster (a GT stored phase, PHASE_*) is known (!= 255) stay unknown:
           pseudo labels only where no GT label is */
        z3 *skz[2] = {nullptr, nullptr}; double skr[2] = {0, 0}; uint8_t *skb[2] = {nullptr, nullptr}; int64_t sk0[2][3], skn[2][3]; int nsk = 0;
        for (int k = 1; k + 3 < argc && nsk < 2; k++) if (!strcmp(argv[k], "--winding-skip")) {
            skz[nsk] = z3_open(store_open(argv[k + 1]), argv[k + 2], nullptr); skr[nsk] = fb.um / atof(argv[k + 3]);
            if (!skz[nsk] || !(skr[nsk] > 0) || !m14) { fprintf(stderr, "fit: --winding-skip: cannot open %s %s (or no --winding-mod14)\n", argv[k + 1], argv[k + 2]); return 1; }
            nsk++;
        }
        if (gate && wci > 0 && wci + 3 < argc) {
            gtz = z3_open(store_open(argv[wci + 1]), argv[wci + 2], nullptr); gtr = fb.um / atof(argv[wci + 3]);
            if (!gtz || !(gtr > 0)) { fprintf(stderr, "fit: --winding-check: cannot open %s %s\n", argv[wci + 1], argv[wci + 2]); return 1; }
        }
        uint16_t *sb = malloc((size_t)BK * BK * BK * 2); float *wv = malloc((size_t)BK * BK * BK * 4);
        fit_set_dev *gd = nullptr; if (!c.cpu && !(gd = fit_set_upload(fs))) return 1;
        for (int64_t sz = 0; sz * BK < gs[0]; sz++) for (int64_t sy = 0; sy * BK < gs[1]; sy++) for (int64_t sx = 0; sx * BK < gs[2]; sx++) {
            const int64_t n3[3] = {gs[0] - sz * BK < BK ? gs[0] - sz * BK : BK, gs[1] - sy * BK < BK ? gs[1] - sy * BK : BK, gs[2] - sx * BK < BK ? gs[2] - sx * BK : BK};
            const double o3[3] = {(double)sz * BK * ws + dOff[0], (double)sy * BK * ws + dOff[1], (double)sx * BK * ws + dOff[2]};
            if (gd ? fit_set_grid(gd, fs, o3, n3, ws, m14, wv) : fit_set_grid_cpu(fs, o3, n3, ws, m14, nullptr, wv)) { fprintf(stderr, "fit: no device memory for the winding grid\n"); return 1; }
            if ((gate || m14 == 3) && (gd ? fit_set_grid(gd, fs, o3, n3, ws, 0, wg) : fit_set_grid_cpu(fs, o3, n3, ws, 0, nullptr, wg))) { fprintf(stderr, "fit: no device memory for the winding grid\n"); return 1; }
            int64_t g0[3] = {0, 0, 0}, gn3[3] = {0, 0, 0};
            if (gtz) {
                for (int d = 0; d < 3; d++) { g0[d] = (int64_t)floor((org[d] + o3[d]) * gtr); gn3[d] = (int64_t)ceil((org[d] + o3[d] + n3[d] * ws) * gtr) - g0[d] + 1; }
                gtb = realloc(gtb, (size_t)gn3[0] * gn3[1] * gn3[2]); if (z3_read(gtz, g0, gn3, gtb, 0)) { fprintf(stderr, "fit: --winding-check: read failed\n"); return 1; }
            }
            for (int k = 0; k < nsk; k++) {
                for (int d = 0; d < 3; d++) { sk0[k][d] = (int64_t)floor((org[d] + o3[d]) * skr[k]); skn[k][d] = (int64_t)ceil((org[d] + o3[d] + n3[d] * ws) * skr[k]) - sk0[k][d] + 1; }
                skb[k] = realloc(skb[k], (size_t)skn[k][0] * skn[k][1] * skn[k][2]); if (z3_read(skz[k], sk0[k], skn[k], skb[k], 0)) { fprintf(stderr, "fit: --winding-skip: read failed\n"); return 1; }
            }
            long long lha[252] = {0}, lhk[252] = {0};   /* this block's histograms: the local (crop-sized block) gauge */
            memset(sb, m14 ? 255 : 0, (size_t)BK * BK * BK * 2);
            for (int64_t a = 0; a < n3[0]; a++) for (int64_t b = 0; b < n3[1]; b++) for (int64_t c2 = 0; c2 < n3[2]; c2++) {
                const double W = (double)wv[((size_t)a * n3[1] + b) * n3[2] + c2]; const size_t o = ((size_t)a * BK + b) * BK + c2;
                if (m14 && gate) {
                    const fit_fields *f = &gf->f; gn++; if (wind_dense_code(W) == 255) continue;
                    const int fz = (int)floor((org[0] - gf->origin[0] + o3[0] + (a + 0.5) * ws) / f->h), fy = (int)floor((org[1] - gf->origin[1] + o3[1] + (b + 0.5) * ws) / f->h),
                              fx = (int)floor((org[2] - gf->origin[2] + o3[2] + (c2 + 0.5) * ws) / f->h);   /* (the fields may be a sub-box of the model's) */
                    int gtc = 0;
                    if (gtz) {
                        const int64_t qz = (int64_t)floor((org[0] + o3[0] + (a + 0.5) * ws) * gtr) - g0[0], qy = (int64_t)floor((org[1] + o3[1] + (b + 0.5) * ws) * gtr) - g0[1],
                                      qx = (int64_t)floor((org[2] + o3[2] + (c2 + 0.5) * ws) * gtr) - g0[2];
                        gtc = gtb[((size_t)qz * gn3[1] + qy) * gn3[2] + qx]; if (gtc > 252) gtc = 0;
                        if (gtph) {   /* UFSM_WCHECK_PHASE=1: the check array is a stored phase raster (PHASE_*), sub-turn histograms only */
                            const int pc = gtc ? gtc : gtb[((size_t)qz * gn3[1] + qy) * gn3[2] + qx] == 0 ? 0 : -1; gtc = 0;
                            if (pc >= 0 && fz >= 0 && fy >= 0 && fx >= 0 && fz < gf->f.D && fy < gf->f.H && fx < gf->f.W) {
                                const fit_fields *f = &gf->f; const size_t i = ((size_t)fz * f->H + fy) * f->W + fx;
                                const double mc = (f->pc[i] - 128) / 127.0, ms = (f->ps[i] - 128) / 127.0, cf = sqrt(mc * mc + ms * ms), Wg = wg[((size_t)a * n3[1] + b) * n3[2] + c2];
                                const int pk = f->sp && f->sp[i] && f->sp[i] * f->h / 8.0 < 10.4;
                                const double g = psg * pc / 252.0; double dl = Wg - g, dn = atan2(ms, mc) / (2 * M_PI) - g; dl -= floor(dl); dn -= floor(dn);
                                const int bl = (int)(dl * 18 + 0.5) % 18, bn = (int)(dn * 18 + 0.5) % 18;
                                double e = Wg - atan2(ms, mc) / (2 * M_PI); e -= floor(e + 0.5);
                                if (f->w[i] >= gm * 255 && cf > 0.05) hph[pk][2][bn]++;
                                if (cf >= gc && f->w[i] >= gm * 255 && fabs(e) <= ge) { hph[pk][0][bl]++; hph[pk][1][bn]++; }
                            }
                        }
                        if (gtc) { ha[(wind_dense_code(W) - (gtc - 1) + 252) % 252]++; lha[(wind_dense_code(W) - (gtc - 1) + 252) % 252]++; }
                    }
                    if (fz < 0 || fy < 0 || fx < 0 || fz >= f->D || fy >= f->H || fx >= f->W) continue;
                    const size_t i = ((size_t)fz * f->H + fy) * f->W + fx;
                    const double mc = (f->pc[i] - 128) / 127.0, ms = (f->ps[i] - 128) / 127.0, cf = sqrt(mc * mc + ms * ms), Wg = wg[((size_t)a * n3[1] + b) * n3[2] + c2];
                    const int pk = f->sp && f->sp[i] && f->sp[i] * f->h / 8.0 < 10.4;
                    double e = Wg - atan2(ms, mc) / (2 * M_PI); e -= floor(e + 0.5);
                    const int dW = gtc ? ((wind_dense_code(W) - (gtc - 1)) % 18 + 18) % 18 : 0, dN = gtc ? ((int)lround(dW - 18 * e) % 18 + 18) % 18 : 0;
                    if (gtc && f->w[i] >= gm * 255 && cf > 0.05) hph[pk][2][dN]++;
                    if (cf < gc || f->w[i] < gm * 255) continue;
                    gcs += cf * cos(2 * M_PI * e); gcw += cf;
                    if (fabs(e) > ge) continue;
                    {   int known = 0;
                        for (int k = 0; k < nsk && !known; k++) {
                            const int64_t qz = (int64_t)floor((org[0] + o3[0] + (a + 0.5) * ws) * skr[k]) - sk0[k][0], qy = (int64_t)floor((org[1] + o3[1] + (b + 0.5) * ws) * skr[k]) - sk0[k][1],
                                          qx = (int64_t)floor((org[2] + o3[2] + (c2 + 0.5) * ws) * skr[k]) - sk0[k][2];
                            known = skb[k][((size_t)qz * skn[k][1] + qy) * skn[k][2] + qx] != 255;
                        }
                        if (known) { gskip++; continue; } }
                    ((uint8_t *)sb)[o] = m14 == 3 ? fit_phase_code(Wg, psg) : (uint8_t)(wind_dense_code(W) + (m14 == 2)); gk++;
                    if (gtc) { hk[(wind_dense_code(W) - (gtc - 1) + 252) % 252]++; lhk[(wind_dense_code(W) - (gtc - 1) + 252) % 252]++; hph[pk][0][dW]++; hph[pk][1][dN]++; }
                    continue;
                }
                if (m14 == 3) { ((uint8_t *)sb)[o] = fit_phase_code(wg[((size_t)a * n3[1] + b) * n3[2] + c2], psg); continue; }
                if (m14) { const uint8_t q = wind_dense_code(W); ((uint8_t *)sb)[o] = q == 255 ? 255 : (uint8_t)(q + (m14 == 2)); continue; }
                const double v = W * 256; sb[o] = (uint16_t)(v < 0 ? 0 : v > 65535 ? 65535 : lround(v));
            }
            if (gtz) for (int pass = 0; pass < 2; pass++) {
                const long long *hh = pass ? lhk : lha; long long best = 0, tot = 0;
                for (int k = 0; k < 252; k++) { tot += hh[k]; long long sw = 0; for (int j = -9; j <= 8; j++) sw += hh[(k + j + 252) % 252]; if (sw > best) best = sw; }
                lbest[pass] += best; ltot[pass] += tot;
                if (pass && getenv("UFSM_WCHECK_BLOCKS") && tot > 0) {   /* (debug: each block's own gauge: radial count drift vs slips) */
                    int bo = 0; long long bb = -1; for (int k = 0; k < 252; k++) { long long sw = 0; for (int j = -9; j <= 8; j++) sw += hh[(k + j + 252) % 252]; if (sw > bb) { bb = sw; bo = k; } }
                    const double zc = o3[0] + n3[0] * ws / 2.0, yc = o3[1] + n3[1] * ws / 2.0, xc = o3[2] + n3[2] * ws / 2.0; const int iz = (int)zc;
                    const double rc = fb.cy ? hypot(yc - fb.cy[iz], xc - fb.cx[iz]) : 0.0;
                    fprintf(stderr, "fit: --winding-check block %lld %lld %lld: %lld gated, gauge %+d codes, right %.3f, r %.0f\n", (long long)sz, (long long)sy, (long long)sx, tot, bo > 126 ? bo - 252 : bo, (double)best / tot, rc);
                }
            }
            if (z3w_write_shard(zw, sz + so[0], sy + so[1], sx + so[2], (const uint8_t *)sb, 8)) { fprintf(stderr, "fit: %s\n", z3w_error()); return 1; }
        }
        fit_set_dev_free(gd); free(wv); free(wg);
        if (gate) { fprintf(stderr, "fit: --winding-gate %g,%g,%g: %lld of %lld voxels labelled (%.1f%%); conf-weighted mean cos 2 pi (W - phase) %.3f over the confident material\n",
                            gc, ge, gm, gk, gn, gn ? 100.0 * gk / gn : 0.0, gcw > 0 ? gcs / gcw : 0.0); if (gf == &gb) fit_fields_free(&gb);
                    if (nsk) fprintf(stderr, "fit: --winding-skip: %lld gated voxels left unknown (GT labelled)\n", gskip); }
        for (int k = 0; k < nsk; k++) { z3_close(skz[k]); free(skb[k]); }
        if (gtz) {   /* best gauge (code offset) of each histogram, share within +-9 codes (0.5 turn) and within +-2 (0.11 turn) */
            for (int pass = 0; pass < 2; pass++) {
                const long long *hh = pass ? hk : ha; long long tot = 0, best = -1; int bo = 0;
                for (int k = 0; k < 252; k++) tot += hh[k];
                for (int k = 0; k < 252; k++) { long long sw = 0; for (int j = -9; j <= 8; j++) sw += hh[(k + j + 252) % 252]; if (sw > best) { best = sw; bo = k; } }
                long long s2 = 0; for (int j = -2; j <= 2; j++) s2 += hh[(bo + j + 252) % 252];
                fprintf(stderr, "fit: --winding-check %s: %lld voxels with a band code; gauge %+d codes (%.2f turns); right (|dq| < 0.5) %.4f, |dq| < 0.12 %.4f\n",
                        pass ? "gated" : "all  ", tot, bo > 126 ? bo - 252 : bo, (bo > 126 ? bo - 252 : bo) / 18.0, tot ? (double)best / tot : 0.0, tot ? (double)s2 / tot : 0.0);
            }
            fprintf(stderr, "fit: --winding-check per %d^3 grid block (own gauge): right (|dq| < 0.5) all %.4f, gated %.4f\n", BK, ltot[0] ? (double)lbest[0] / ltot[0] : 0.0, ltot[1] ? (double)lbest[1] / ltot[1] : 0.0);
            for (int pk = 0; pk < 2; pk++) for (int k = 0; k < 3; k++) {   /* share within 1/4 turn (+-4 codes of 18) at offset 0 and at the best offset */
                long long tot = 0, s0 = 0, sb = 0; int bo = 0;
                for (int j = 0; j < 18; j++) tot += hph[pk][k][j];
                for (int o = 0; o < 18; o++) { long long sw = 0; for (int j = -4; j <= 4; j++) sw += hph[pk][k][(o + j + 18) % 18]; if (!o) s0 = sw; if (sw > sb) { sb = sw; bo = o; } }
                fprintf(stderr, "fit: --winding-check phase %s %s: %lld voxels, within 1/4 turn %.4f (offset 0), %.4f (best offset %+d/18)\n", pk ? "packed" : "sparse",
                        k == 0 ? "label frac(W), gated" : k == 1 ? "network phase, gated" : "network phase, all  ", tot, tot ? (double)s0 / tot : 0.0, tot ? (double)sb / tot : 0.0, bo > 9 ? bo - 18 : bo);
            }
            z3_close(gtz); free(gtb);
        }
        z3w_close(zw); free(sb);
        fprintf(stderr, "fit: winding %lldx%lldx%lld (step %d) written to %s\n", (long long)gs[0], (long long)gs[1], (long long)gs[2], ws, wout);
    }
    const int ci = arg_index(argc, argv, "--codes");
    if (ci > 0 && ci + 2 < argc) {   /* score against the band codes of the box, streamed in label z-chunks (U on the GPU) */
        z3 *z = z3_open(store_open(argv[ci + 1]), argv[ci + 2], nullptr);
        if (!z) { fprintf(stderr, "%s\n", z3_error()); return 1; }
        const double cf = atof(argv[ci + 2]) > 0 ? atof(argv[ci + 2]) / fb.um : 2;   /* label voxel in native voxels (the codes' key is its um: 4.8) */
        FILE *dump = *opt(argc, argv, "--score-dump", "") ? fopen(opt(argc, argv, "--score-dump", ""), "wb") : nullptr;   /* box-local z y x, U - q mod 14 per voxel */
        fit_set_dev *gd = nullptr; if (!c.cpu && !(gd = fit_set_upload(fs))) return 1;
        fit_fill_ctx fc = {gd, fs};
        g_score_pairs = atoi(opt(argc, argv, "--score-pairs", "0")) > 0;
        const int rc = score_codes(z, cf, org, shp, h, fb.um, fb.cy, fb.cx, arg_index(argc, argv, "--score-detail") > 0, dump, fit_fill, &fc);
        fit_set_dev_free(gd); z3_close(z); if (dump) fclose(dump);
        if (rc) return 1;
    }
    fit_set_free(fs); if (!cached) fit_fields_free(&fb);
    return 0;
}

/* ufsm wind-score <sweep-dir> <codes-root> <codes-key> [--score-detail 1] [--score-dump FILE]
   The fitter's --codes score (score_codes) of a winding raster written by predict --cond-sweep: <sweep-dir>/U.f32 (float32 [z][y][x]
   over the box, NaN where the sweep did not reach) and sweep.json (shape, origin_zyx and um of the box's level grid, scale = level
   voxels per native 2.4 um voxel, axis). U is read at the voxel holding each label voxel; the same gauge, ids-right, |error| and
   by-z / by-radius integer errors as fit --codes, plus the labelled voxels the sweep did not reach. */
typedef struct { const float *U; int64_t n[3]; } raster_fill_ctx;
static int raster_fill(void *c, const double og[3], const int64_t n3[3], double step, const uint8_t *codes, float *U) {
    const raster_fill_ctx *r = c;
    for (int64_t a = 0; a < n3[0]; a++) for (int64_t b = 0; b < n3[1]; b++) for (int64_t x = 0; x < n3[2]; x++) {
        const int64_t iz = (int64_t)floor(og[0] + a * step), iy = (int64_t)floor(og[1] + b * step), ix = (int64_t)floor(og[2] + x * step);
        U[((size_t)a * n3[1] + b) * n3[2] + x] = iz < 0 || iy < 0 || ix < 0 || iz >= r->n[0] || iy >= r->n[1] || ix >= r->n[2] ? NAN : r->U[((size_t)iz * r->n[1] + iy) * r->n[2] + ix];
    }
    return 0;
}
static int cmd_wind_score(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: ufsm wind-score <sweep-dir> <codes-root> <codes-key> [--score-detail 1] [--score-dump FILE]\n"); return 2; }
    char p[1400]; snprintf(p, sizeof p, "%s/sweep.json", argv[2]);
    FILE *f = fopen(p, "rb"); if (!f) { fprintf(stderr, "wind-score: cannot read %s\n", p); return 1; }
    char jb[65536]; const size_t jn = fread(jb, 1, sizeof jb - 1, f); fclose(f); jb[jn] = 0;
    json *j = json_parse(jb, jn); if (!j) { fprintf(stderr, "wind-score: bad %s\n", p); return 1; }
    long long org[3]; int shp[3];
    for (int d = 0; d < 3; d++) { org[d] = (long long)json_num(json_at(json_get(j, "origin_zyx"), d), 0); shp[d] = (int)json_num(json_at(json_get(j, "shape"), d), 0); }
    const double um = json_num(json_get(j, "um"), 9.6), scale = json_num(json_get(j, "scale"), 4);
    axis ax = {0};
    if (axis_load(&ax, json_str(json_get(j, "axis"), ""))) { fprintf(stderr, "wind-score: cannot load the axis named in %s\n", p); json_free(j); return 1; }
    json_free(j);
    const size_t nv = (size_t)shp[0] * shp[1] * shp[2]; float *Ur = malloc(nv * 4);
    snprintf(p, sizeof p, "%s/U.f32", argv[2]); f = fopen(p, "rb");
    if (!Ur || !f || fread(Ur, 4, nv, f) != nv) { fprintf(stderr, "wind-score: cannot read %s (%zu values)\n", p, nv); if (f) fclose(f); return 1; }
    fclose(f);
    const int na = shp[0] + 1; float *cy = malloc((size_t)na * 4), *cx = malloc((size_t)na * 4);   /* the axis per box z, box-local level voxels */
    for (int z = 0; z < na; z++) { double y, x; axis_at(&ax, (org[0] + z) * scale, &y, &x); cy[z] = (float)(y / scale - org[1]); cx[z] = (float)(x / scale - org[2]); }
    z3 *zc = z3_open(store_open(argv[3]), argv[4], nullptr); if (!zc) { fprintf(stderr, "wind-score: %s\n", z3_error()); return 1; }
    const double cf = atof(argv[4]) > 0 ? atof(argv[4]) / um : 2;
    FILE *dump = *opt(argc, argv, "--score-dump", "") ? fopen(opt(argc, argv, "--score-dump", ""), "wb") : nullptr;
    raster_fill_ctx rc = {Ur, {shp[0], shp[1], shp[2]}};
    g_score_pairs = atoi(opt(argc, argv, "--score-pairs", "0")) > 0;
    const int r = score_codes(zc, cf, org, shp, 1.0, um, cy, cx, atoi(opt(argc, argv, "--score-detail", "0")) > 0, dump, raster_fill, &rc);
    z3_close(zc); if (dump) fclose(dump); free(Ur); free(cy); free(cx);
    return r;
}

/* ufsm wraps <model-dir> <out-dir> [--step 8] [--first K] [--last K] [--fields DIR] [--um 2.4] [--threads 16]
   [--snap ROOT KEY [--snap-range 0.35] [--snap-min 0.5] [--snap-origin z,y,x] | --snap-fields DIR] [--out-um U]
   One tifxyz per wrap of a fitted winding model (ufsm fit --out): wrap k is the level set W = k. Rows every step native voxels in
   z, columns every step / r_k radians of theta over the box's angular range (r_k: the wrap's radius at the box centre), each
   vertex the root of W = k along the ray from the umbilicus (src/fit.h fit_ray_root); vertices outside the box, without a root,
   or (--fields) in a field voxel without data are -1. Coordinates are global native voxels. */
static float trilin(const uint8_t *v, const int64_t n[3], double z, double y, double x);
/* ufsm wraps --snap: each vertex moved along its ray from the axis to the recto maximum within +- range pitches, read from a
   recto probability array on the model's native grid (array voxel = native - ro), when that maximum is >= pmin; then the
   displacements are median-filtered over 5 x 5 vertices so a stray peak cannot tear the surface. Tiles of rows x columns read
   only their own box (plus the search range); the vertices of a tile run on threads. */
typedef struct { const fit_model *m; const uint8_t *vol; int64_t o[3], n[3]; const float *px, *py, *pz; float *disp; int cols, r0, r1, c0, c1;
                 double range, pmin; _Atomic int next; double ih, sub; int nearest; } snap_job;   /* array index = (native - o) ih - sub;
                 nearest: the local maximum nearest the vertex (else the highest within the range) */
static double model_pitch(const fit_model *m, double z) {
    double k = z / m->knot_dz; if (k < 0) k = 0; if (k > m->nk - 1) k = m->nk - 1;
    int k0 = (int)k; if (k0 > m->nk - 2) k0 = m->nk - 2 < 0 ? 0 : m->nk - 2;
    return m->nk > 1 ? m->pitch[k0] + (k - k0) * (m->pitch[k0 + 1] - m->pitch[k0]) : m->pitch[0];
}
static void *snap_rows(void *arg) {
    snap_job *J = arg; const fit_model *m = J->m; const int na = (int)(m->D * m->h) + 1;
    for (int r; (r = J->r0 + atomic_fetch_add(&J->next, 1)) < J->r1;) for (int c = J->c0; c < J->c1; c++) {
        const size_t i = (size_t)r * J->cols + c; if (J->pz[i] < 0) continue;
        const double zl = J->pz[i] - m->origin[0], zi = fmin(fmax(zl, 0), na - 1);
        const double cy = m->cy[(int)zi] + m->origin[1], cx = m->cx[(int)zi] + m->origin[2];
        double uy = J->py[i] - cy, ux = J->px[i] - cx; const double ul = hypot(uy, ux); if (ul < 1e-6) continue; uy /= ul; ux /= ul;
        const double R = J->range * model_pitch(m, zl);
#define SNAPV(t_) trilin(J->vol, J->n, (J->pz[i] - J->o[0]) * J->ih - J->sub, (J->py[i] + (t_) * uy - J->o[1]) * J->ih - J->sub, (J->px[i] + (t_) * ux - J->o[2]) * J->ih - J->sub)
        if (J->nearest) {   /* the local maximum nearest the vertex, >= pmin (the sheet the fit is on, not a brighter neighbour) */
            const double st = 0.25; double bd = 1e300, bt = 0; float vm = SNAPV(-R - st), v0 = SNAPV(-R);
            for (double t = -R; t <= R; t += st) {
                const float vp = SNAPV(t + st);
                if (v0 >= vm && v0 > vp && v0 >= J->pmin * 255 && fabs(t) < bd) { bd = fabs(t); bt = t; }
                vm = v0; v0 = vp;
            }
            if (bd < 1e300) J->disp[i] = (float)bt;
            continue;
        }
        float best = -1; double bt = 0;
        for (double t = -R; t <= R; t += 0.5) {
            const float v = SNAPV(t);
            if (v > best) { best = v; bt = t; }
        }
#undef SNAPV
        if (best >= J->pmin * 255) J->disp[i] = (float)bt;
    }
    return nullptr;
}
static int cmp_fl(const void *a, const void *b) { const float x = *(const float *)a, y = *(const float *)b; return (x > y) - (x < y); }
/* mem (with --snap-fields): an in-memory array of shape mn on a grid of cell mh native voxels from mo (cell centres), searched for
   the nearest local maximum instead of a zarr read per tile */
static size_t wraps_snap(z3 *rz, const int64_t ro[3], const fit_model *m, int rows, int cols, float *px, float *py, float *pz, double range, double pmin, int nthreads,
                         const uint8_t *mem, const int mn[3], const long long mo[3], double mh) {
    const size_t nv = (size_t)rows * cols; float *disp = malloc(nv * 4); for (size_t i = 0; i < nv; i++) disp[i] = NAN;
    if (mem) {
        snap_job J = {m, mem, {mo[0], mo[1], mo[2]}, {mn[0], mn[1], mn[2]}, px, py, pz, disp, cols, 0, rows, 0, cols, range, pmin, 0, 1 / mh, 0.5, 1};
        pthread_t th[64]; const int nt = nthreads < 64 ? (nthreads > 0 ? nthreads : 1) : 64;
        for (int t = 0; t < nt; t++) pthread_create(&th[t], nullptr, snap_rows, &J);
        for (int t = 0; t < nt; t++) pthread_join(th[t], nullptr);
    }
    const z3_meta *zm = rz ? z3_meta_of(rz) : nullptr; double pmax = 0; for (int k = 0; k < m->nk; k++) if (m->pitch[k] > pmax) pmax = m->pitch[k];
    const int TR = 64, TC = 512; const double mar = range * pmax + 3;
    for (int r0 = 0; r0 < (mem ? 0 : rows); r0 += TR) for (int c0 = 0; c0 < cols; c0 += TC) {
        const int r1 = r0 + TR < rows ? r0 + TR : rows, c1 = c0 + TC < cols ? c0 + TC : cols;
        double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
        for (int r = r0; r < r1; r++) for (int c = c0; c < c1; c++) {
            const size_t i = (size_t)r * cols + c; if (pz[i] < 0) continue;
            const double v[3] = {pz[i], py[i], px[i]}; for (int d = 0; d < 3; d++) { if (v[d] < lo[d]) lo[d] = v[d]; if (v[d] > hi[d]) hi[d] = v[d]; }
        }
        if (lo[0] > hi[0]) continue;
        snap_job J = {m, nullptr, {0, 0, 0}, {0, 0, 0}, px, py, pz, disp, cols, r0, r1, c0, c1, range, pmin, 0, 1, 0, 0};
        int ok = 1;
        for (int d = 0; d < 3; d++) {
            const double g = d ? mar : 2;   /* rays are horizontal: the search moves y / x only */
            int64_t a = (int64_t)floor(lo[d] - g) - ro[d], b = (int64_t)ceil(hi[d] + g) + 1 - ro[d];
            if (a < 0) a = 0;
            if (b > zm->shape[d]) b = zm->shape[d];
            if (b <= a) ok = 0;
            J.o[d] = a + ro[d]; J.n[d] = b - a;
        }
        if (!ok) continue;
        uint8_t *vol = malloc((size_t)J.n[0] * J.n[1] * J.n[2]);
        const int64_t ao[3] = {J.o[0] - ro[0], J.o[1] - ro[1], J.o[2] - ro[2]};
        if (!vol || z3_read(rz, ao, J.n, vol, nthreads)) { free(vol); continue; }
        J.vol = vol;
        pthread_t th[64]; const int nt = nthreads < 64 ? (nthreads > 0 ? nthreads : 1) : 64;
        for (int t = 0; t < nt; t++) pthread_create(&th[t], nullptr, snap_rows, &J);
        for (int t = 0; t < nt; t++) pthread_join(th[t], nullptr);
        free(vol);
    }
    size_t moved = 0; float *med = malloc(nv * 4); float win[25];
    for (int r = 0; r < rows; r++) for (int c = 0; c < cols; c++) {
        const size_t i = (size_t)r * cols + c; med[i] = NAN; if (pz[i] < 0) continue;
        int k = 0;
        for (int a = r - 2; a <= r + 2; a++) for (int b = c - 2; b <= c + 2; b++)
            if (a >= 0 && a < rows && b >= 0 && b < cols && !isnan(disp[(size_t)a * cols + b])) win[k++] = disp[(size_t)a * cols + b];
        if (k >= 5) { qsort(win, k, sizeof(float), cmp_fl); med[i] = win[k / 2]; }
    }
    const int na = (int)(m->D * m->h) + 1;
    for (size_t i = 0; i < nv; i++) {
        if (isnan(med[i]) || med[i] == 0) continue;
        const double zl = pz[i] - m->origin[0], zi = fmin(fmax(zl, 0), na - 1);
        const double cy = m->cy[(int)zi] + m->origin[1], cx = m->cx[(int)zi] + m->origin[2];
        double uy = py[i] - cy, ux = px[i] - cx; const double ul = hypot(uy, ux); if (ul < 1e-6) continue;
        py[i] = (float)(py[i] + med[i] * uy / ul); px[i] = (float)(px[i] + med[i] * ux / ul); moved++;
    }
    free(disp); free(med);
    return moved;
}
/* ufsm wraps --fields: per vertex row (z) and angle bin, the radius range [rmin, rmax] of the points around the row's axis point
   whose data-weight cell can be non-zero (4 x 4 cell blocks with a non-zero cell or neighbour, padded by a native voxel; the bin
   and its neighbours). A vertex outside it is dropped anyway, so ray roots outside are not searched for (fit_set_ray_root_b): on a
   2 cm half box ~93% of the rays found no kept vertex (in the masked half, in the void, or off the wrap) at ~30-180 W evaluations
   each. The kept vertices are the same. */
typedef struct { int rows, nb; double t0, bw; float *rmin, *rmax; } wr_polar;
typedef struct { wr_polar *P; const fit_model *m; const uint8_t *wf; int wsh[3]; double wh, step, Z; int na; _Atomic int next; double zlo; } wr_polar_job;
static void *wr_polar_rows(void *arg) {
    wr_polar_job *J = arg; wr_polar *P = J->P; const fit_model *m = J->m; const int B = 4, H = J->wsh[1], W = J->wsh[2];
    const int Hb = (H + B - 1) / B, Wb = (W + B - 1) / B; uint8_t *blk = malloc((size_t)Hb * Wb);
    for (int r; (r = atomic_fetch_add(&J->next, 1)) < P->rows;) {
        const double z = fmin(J->zlo + r * J->step, J->Z), cy = m->cy[(int)fmin(z, J->na - 1)], cx = m->cx[(int)fmin(z, J->na - 1)];
        const int fz = (int)fmin(z / J->wh, J->wsh[0] - 1); const uint8_t *pl = J->wf + (size_t)fz * H * W;
        float *rn = P->rmin + (size_t)r * P->nb, *rx = P->rmax + (size_t)r * P->nb;
        for (int q = 0; q < P->nb; q++) { rn[q] = 1e30f; rx[q] = -1e30f; }
        memset(blk, 0, (size_t)Hb * Wb);
        for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) if (pl[(size_t)y * W + x])   /* the cell and its neighbours' blocks */
            for (int a = y - 1; a <= y + 1; a++) for (int b = x - 1; b <= x + 1; b++) if (a >= 0 && b >= 0 && a < H && b < W) blk[(size_t)(a / B) * Wb + b / B] = 1;
        for (int by = 0; by < Hb; by++) for (int bx = 0; bx < Wb; bx++) {
            if (!blk[(size_t)by * Wb + bx]) continue;
            const double y0 = by * B * J->wh - 1 - cy, y1 = (by + 1) * B * J->wh + 1 - cy, x0 = bx * B * J->wh - 1 - cx, x1 = (bx + 1) * B * J->wh + 1 - cx;
            const double ny = y0 > 0 ? y0 : y1 < 0 ? -y1 : 0, nx = x0 > 0 ? x0 : x1 < 0 ? -x1 : 0;
            const double rlo = sqrt(ny * ny + nx * nx) - 1, rhi = sqrt(fmax(y0 * y0, y1 * y1) + fmax(x0 * x0, x1 * x1)) + 1;
            int q0 = 0, q1 = P->nb - 1, full = rlo <= 2;
            double alo = 0, ahi = 0;
            if (!full) {   /* the block's angles around the axis point: its corners' relative to its centre's */
                const double ac = atan2(0.5 * (y0 + y1), 0.5 * (x0 + x1)); double dmin = 0, dmax = 0;
                const double cyv[2] = {y0, y1}, cxv[2] = {x0, x1};
                for (int c = 0; c < 4; c++) { double d = atan2(cyv[c >> 1], cxv[c & 1]) - ac; d -= 2 * M_PI * floor((d + M_PI) / (2 * M_PI)); if (d < dmin) dmin = d; if (d > dmax) dmax = d; }
                alo = ac + dmin; ahi = ac + dmax;
            }
            for (int sh = -1; sh <= 1; sh++) {
                if (!full) {
                    const double lo = (alo + 2 * M_PI * sh - P->t0) / P->bw, hi = (ahi + 2 * M_PI * sh - P->t0) / P->bw;
                    if (hi < 0 || lo > P->nb - 1) continue;
                    q0 = lo < 0 ? 0 : (int)floor(lo); q1 = hi > P->nb - 1 ? P->nb - 1 : (int)floor(hi);
                } else if (sh) continue;
                for (int q = q0; q <= q1; q++) { if (rlo < rn[q]) rn[q] = (float)fmax(rlo, 0); if (rhi > rx[q]) rx[q] = (float)rhi; }
            }
        }
    }
    free(blk);
    return nullptr;
}
static void wr_polar_bounds(const wr_polar *P, int r, double th, double *ua, double *ub) {   /* the bins at th and their neighbours */
    *ua = -1e300; *ub = 1e300; if (!P) return;
    const double t = (th - P->t0) / P->bw; if (!(t >= 1) || t >= P->nb - 1) return;   /* (outside the table: unbounded) */
    const int q = (int)floor(t); double a = 1e300, b = -1e300;
    for (int k = q - 1; k <= q + 1; k++) { const float lo = P->rmin[(size_t)r * P->nb + k], hi = P->rmax[(size_t)r * P->nb + k]; if (lo <= hi) { if (lo < a) a = lo; if (hi > b) b = hi; } }
    *ua = a; *ub = b;   /* (no block: a > b, no root wanted) */
}
typedef struct {   /* ufsm wraps: one wrap's rows over threads */
    const fit_set *fs; const fit_model *m; const uint8_t *wf; int wsh[3]; double wh;
    double step, Z; int na; double th0, dth; int rows, cols, k; double rk;
    float *px, *py, *pz; _Atomic int next; _Atomic size_t valid; const wr_polar *P; double zlo;
} wraps_job;
static void *wraps_rows(void *arg) {
    wraps_job *J = arg; const fit_model *m = J->m;
    for (int r; (r = atomic_fetch_add(&J->next, 1)) < J->rows;) {
        const double z = fmin(J->zlo + r * J->step, J->Z), cy = m->cy[(int)fmin(z, J->na - 1)], cx = m->cx[(int)fmin(z, J->na - 1)];
        double rprev = J->rk; size_t valid = 0;
        for (int c = 0; c < J->cols; c++) {
            const size_t i = (size_t)r * J->cols + (J->cols - 1 - c);   /* column 0 = the largest theta side (upstream: outermost first) */
            const double th = J->th0 + c * J->dth; double rr;
            J->px[i] = J->py[i] = J->pz[i] = -1;
            double ua, ub; wr_polar_bounds(J->P, r, th, &ua, &ub);
            if (fit_set_ray_root_b(J->fs, z, th, J->k, rprev, ua, ub, &rr)) continue;
            const double y = cy + rr * sin(th), x = cx + rr * cos(th);
            if (J->wf) {
                const int fz = (int)fmin(z / J->wh, J->wsh[0] - 1), fy = (int)fmin(y / J->wh, J->wsh[1] - 1), fx = (int)fmin(x / J->wh, J->wsh[2] - 1);
                if (fy < 0 || fx < 0 || !J->wf[((size_t)fz * J->wsh[1] + fy) * J->wsh[2] + fx]) continue;
            }
            J->px[i] = (float)(x + m->origin[2]); J->py[i] = (float)(y + m->origin[1]); J->pz[i] = (float)(z + m->origin[0]); valid++; rprev = rr;
        }
        atomic_fetch_add(&J->valid, valid);
    }
    return nullptr;
}
typedef struct {
    const fit_set *fs; const fit_model *m; const uint8_t *wf; int wsh[3]; double wh, step, Z; int na; double th0, th1; int rows, k0, k1, nthreads; double um, out_um;
    const char *out; z3 *snapz; const int64_t *snapo; double snap_range, snap_min; const uint8_t *sfa; int sfn[3]; long long sfo[3]; double sfh; int verbose_snap; double Y, X;
    const wr_polar *P; _Atomic int next; _Atomic int nwr; _Atomic size_t nvalid; _Atomic int fail;
    double zlo, zhi, colr;   /* --zr: box-local native z of the first / last row; --col-r: < 0 centre, 0 p90, > 0 that radius */
    const int *ks; int nk; double min_cm2; _Atomic int nsmall;   /* the wrap ids to compute (largest first); --min-cm2; wraps dropped by it */
} wraps_all;
static int wr_cmp_d(const void *a, const void *b) { const double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }
static int wrap_one(wraps_all *A, int k, int nthreads) {   /* wrap k: its rows on nthreads threads, snapped, written */
    const fit_model *m = A->m; const double Z = A->Z, Y = A->Y, X = A->X, step = A->step; const int na = A->na, rows = A->rows;
    /* the wrap's radius at the box centre: the root along the central ray */
    const double zc = 0.5 * (A->zlo + A->zhi), cyc = m->cy[(int)fmin(zc, na - 1)], cxc = m->cx[(int)fmin(zc, na - 1)];
    const double thc = atan2(Y / 2 - cyc, X / 2 - cxc); double rk;
    if (fit_set_ray_root(A->fs, zc, thc, k, hypot(Y / 2 - cyc, X / 2 - cxc), &rk)) rk = hypot(Y / 2 - cyc, X / 2 - cxc);
    if (A->colr > 0) rk = A->colr;
    else if (A->colr == 0) {   /* --col-r p90: the 90th percentile of the wrap's radius over 33 z x 16 angles of the rows' z range (a wrap
                                  that leaves the axis near the box centre's z, e.g. a core wrap, has a tiny root on the central ray: its
                                  columns would be far apart at the radii it reaches elsewhere) */
        double rr[33 * 16]; int nr = 0;
        for (int a = 0; a <= 32; a++) {
            const double z = A->zlo + (A->zhi - A->zlo) * a / 32;
            for (int b = 0; b < 16; b++) { double r; if (!fit_set_ray_root(A->fs, z, -M_PI + 2 * M_PI * (b + 0.5) / 16, k, rk, &r)) rr[nr++] = r; }
        }
        if (nr) { qsort(rr, nr, sizeof(double), wr_cmp_d); rk = rr[(int)(0.9 * (nr - 1))]; }
    }
    const double dth = step / fmax(rk, step);
    const int cols = (int)ceil((A->th1 - A->th0) / dth) + 1;
    const size_t nv = (size_t)rows * cols;
    float *px = malloc(nv * 4), *py = malloc(nv * 4), *pz = malloc(nv * 4);
    wraps_job J = {A->fs, m, A->wf, {A->wsh[0], A->wsh[1], A->wsh[2]}, A->wh, step, Z, na, A->th0, dth, rows, cols, k, rk, px, py, pz, 0, 0, A->P, A->zlo};
    if (nthreads > 1) {
        pthread_t th[64]; const int nt = nthreads < 64 ? nthreads : 64;
        for (int t = 0; t < nt; t++) pthread_create(&th[t], nullptr, wraps_rows, &J);
        for (int t = 0; t < nt; t++) pthread_join(th[t], nullptr);
    } else wraps_rows(&J);
    const size_t valid = atomic_load(&J.valid);
    if ((A->snapz || A->sfa) && valid) { const size_t mv = wraps_snap(A->snapz, A->snapo, m, rows, cols, px, py, pz, A->snap_range, A->snap_min, nthreads, A->sfa, A->sfn, A->sfo, A->sfh); if (A->verbose_snap) fprintf(stderr, "wraps: w%03d: %zu of %zu vertices snapped\n", k, mv, valid); }
    int rc = 0;
    const double vcm2 = A->step * A->um * 1e-4 * A->step * A->um * 1e-4;   /* (cm2 per vertex at the column spacing --step) */
    if (valid > 16 && valid * vcm2 < A->min_cm2) atomic_fetch_add(&A->nsmall, 1);
    else if (valid > 16) {
        char wd[1400]; snprintf(wd, sizeof wd, "%s/w%03d", A->out, k);
        double wstep = step, wum = A->um;
        if (A->out_um > 0 && fabs(A->out_um - A->um) > 1e-9) {   /* --out-um: coordinates on another grid of the same scan (voxel centres to voxel centres) */
            const double f = A->um / A->out_um;
            for (size_t i = 0; i < nv; i++) if (pz[i] >= 0) { px[i] = (float)((px[i] + 0.5) * f - 0.5); py[i] = (float)((py[i] + 0.5) * f - 0.5); pz[i] = (float)((pz[i] + 0.5) * f - 0.5); }
            wstep = step * f; wum = A->out_um;
        }
        if (tifxyz_write(wd, cols, rows, px, py, pz, wstep, wum, k, "ufsm wraps (fitted winding level set)")) { fprintf(stderr, "wraps: cannot write %s\n", wd); rc = -1; }
        else { atomic_fetch_add(&A->nwr, 1); atomic_fetch_add(&A->nvalid, valid); }
    }
    free(px); free(py); free(pz);
    return rc;
}
static void *wraps_worker(void *arg) {
    wraps_all *A = arg;
    for (int i; (i = atomic_fetch_add(&A->next, 1)) < A->nk;) if (wrap_one(A, A->ks[i], 1)) atomic_store(&A->fail, 1);
    return nullptr;
}
/* the wrap ids near data: W at the centre of every data-weight cell (wf != 0) in the rows' z range; id k is kept when some cell has
   |W - k| <= margin (1 turn + the cell's half diagonal at a 4-voxel pitch, so a vertex in a data cell, which needs |W - k| = 0 inside it, is
   never missed); counts per id (cells within the margin) order the wraps largest first. Empty ids cost ~30-60 CPU-s each (a root search
   on every ray): Paris 4 sampled 16..580 and wrote 257, the 308 ids above the outermost data took 10.7 of the 20.7 min. */
typedef struct { const fit_set *fs; const uint8_t *wf; int wsh[3]; double wh, zlo, zhi, margin; int k0, nk; _Atomic long long *cnt; _Atomic int next; } wr_prune_job;
static void *wr_prune_rows(void *arg) {
    wr_prune_job *J = arg; const int H = J->wsh[1], W = J->wsh[2]; long long *loc = calloc((size_t)J->nk, sizeof *loc);
    for (int fz; (fz = atomic_fetch_add(&J->next, 1)) < J->wsh[0];) {
        const double z = (fz + 0.5) * J->wh; if (z < J->zlo - J->wh || z > J->zhi + J->wh) continue;
        const double zc = fmin(fmax(z, J->zlo), J->zhi); const uint8_t *pl = J->wf + (size_t)fz * H * W;
        for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
            if (!pl[(size_t)y * W + x]) continue;
            const double w = fit_set_W(J->fs, zc, (y + 0.5) * J->wh, (x + 0.5) * J->wh); if (!isfinite(w)) continue;
            int a = (int)ceil(w - J->margin) - J->k0, b = (int)floor(w + J->margin) - J->k0; if (a < 0) a = 0; if (b > J->nk - 1) b = J->nk - 1;
            for (int k = a; k <= b; k++) loc[k]++;
        }
    }
    for (int k = 0; k < J->nk; k++) if (loc[k]) atomic_fetch_add(&J->cnt[k], loc[k]);
    free(loc);
    return nullptr;
}
static const long long *g_wr_cnt; static int g_wr_k0;
static int wr_cmp_cnt(const void *a, const void *b) {   /* larger count first, then the lower id */
    const int x = *(const int *)a, y = *(const int *)b; const long long cx = g_wr_cnt[x - g_wr_k0], cy = g_wr_cnt[y - g_wr_k0];
    return cx != cy ? (cx < cy) - (cx > cy) : (x > y) - (x < y);
}
static int cmd_wraps(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm wraps <model-dir> <out-dir> [--step 8] [--first K] [--last K] [--fields DIR] [--um 2.4] [--threads 16] [--sfc ERR] [--zr Z0,Z1] [--col-r centre|p90|R]\n"
                                    "  --sfc ERR: each wrap as <out-dir>/wNNN.sfc (surface-compressor, joint XYZ within ERR voxels, meta.json inside) instead of a tifxyz dir\n"
                                    "  --zr Z0,Z1: only the rows at global native z in [Z0, Z1] (the model's grid; also the z range the wrap ids are sampled in)\n"
                                    "  --first / --last K: the wrap ids exactly (default: those the box samples meet); wraps with <= 16 vertices are not written\n"
                                    "  --col-r: the radius (native voxels) whose arc the column spacing --step is: centre = the wrap's root on the box centre's ray (default),\n"
                                    "     p90 = the 90th percentile of its roots over the rows' z range (default; wraps that leave the axis, the core), or a number\n"
                                    "  --min-cm2 C: wraps under C cm2 (vertices x (step x um)^2) are not written (default 0.5: end-cap / air fragments; 0 keeps all)\n"
                                    "  with --fields: only wrap ids within ~1 turn of a data cell are computed, largest first (UFSM_WRAPS_NOPRUNE=1: every id)\n"); return 2; }
    if (atof(opt(argc, argv, "--sfc", "0")) > 0) tifxyz_set_sfc(atof(opt(argc, argv, "--sfc", "0")));
    fit_set *fs = fit_set_load(argv[2]); if (!fs) { fprintf(stderr, "wraps: cannot load %s\n", argv[2]); return 1; }
    const fit_model *m = fs->global;   /* the box, the axis; rows dispatch to the slab model of their z */
    const double step = atof(opt(argc, argv, "--step", "8")), um = atof(opt(argc, argv, "--um", "0")) > 0 ? atof(opt(argc, argv, "--um", "0")) : fs->global->um;
    const int nthreads = ufsm_threads(atoi(opt(argc, argv, "--threads", "0")), 16); fit_set_threads(nthreads);   /* (the field loads' core masks) */
    const double out_um = atof(opt(argc, argv, "--out-um", "0"));   /* e.g. 2.4: a 9.6 um fit's wraps on the 2.4 um grid (renders with 2.4 um ink) */
    const int si = arg_index(argc, argv, "--snap");   /* --snap ROOT KEY: a recto probability array on the model's grid (predict output) */
    z3 *snapz = nullptr; int64_t snapo[3] = {0, 0, 0}; const double snap_range = atof(opt(argc, argv, "--snap-range", "0.35")), snap_min = atof(opt(argc, argv, "--snap-min", "0.5"));
    const int verbose_snap = arg_index(argc, argv, "--quiet") < 0;
    if (si > 0) {
        if (si + 2 >= argc) { fprintf(stderr, "wraps: --snap ROOT KEY\n"); return 2; }
        snapz = z3_open(store_open(argv[si + 1]), argv[si + 2], opt(argc, argv, "--cache", nullptr));
        if (!snapz) { fprintf(stderr, "wraps: --snap: %s\n", z3_error()); return 1; }
        const char *so = opt(argc, argv, "--snap-origin", nullptr);   /* else the array's ufsm.origin_zyx (predict writes it) */
        if (so) { long long v[3]; if (sscanf(so, "%lld,%lld,%lld", v, v + 1, v + 2) != 3) { fprintf(stderr, "wraps: --snap-origin z,y,x\n"); return 2; } for (int d = 0; d < 3; d++) snapo[d] = v[d]; }
        else {
            char jp[1400]; snprintf(jp, sizeof jp, "%s/%s/zarr.json", argv[si + 1], argv[si + 2]); FILE *jf = fopen(jp, "rb"); char jb[8192] = {0};
            if (jf) { size_t nr = fread(jb, 1, sizeof jb - 1, jf); (void)nr; fclose(jf); }
            const char *q = strstr(jb, "\"origin_zyx\":"); long long v[3];
            if (!q || sscanf(q + 13, " [%lld,%lld,%lld]", v, v + 1, v + 2) != 3) { fprintf(stderr, "wraps: --snap: no origin_zyx in %s (give --snap-origin)\n", jp); return 2; }
            for (int d = 0; d < 3; d++) snapo[d] = v[d];
        }
    }
    /* --snap-fields DIR: snap to the sheet centres of a fit-field set (its pc = 128 + 127 conf cos(phase) peaks at phase 0, the sheet,
       weighted by the confidence): each vertex to the nearest local maximum >= --snap-min along its ray (the level set of the fitted
       W is smoother than the sheets and crosses the gaps between them) */
    uint8_t *sfa = nullptr; int sfn[3] = {0, 0, 0}; long long sfo[3] = {0, 0, 0}; double sfh = 1;
    const char *sfd = opt(argc, argv, "--snap-fields", nullptr);
    if (sfd) {
        if (snapz) { fprintf(stderr, "wraps: --snap or --snap-fields, not both\n"); return 2; }
        fit_fields_buf g; if (fit_fields_load(sfd, 0, 0, 0, &g)) { fprintf(stderr, "wraps: cannot read the fields in %s\n", sfd); return 1; }
        for (int d = 0; d < 3; d++) { sfo[d] = g.origin[d]; }
        sfh = g.h; fit_fields_free(&g);
        if (!(sfa = fit_field_array(sfd, "pc", 1, sfn))) { fprintf(stderr, "wraps: cannot read pc in %s\n", sfd); return 1; }
    }
    const double Z = m->D * m->h, Y = m->H * m->h, X = m->W * m->h;
    uint8_t *wf = nullptr; int wsh[3] = {0, 0, 0}; double wh = 0;   /* optional: the fields' data weight (max-pooled to <= ~1 GB) */
    const char *fd = opt(argc, argv, "--fields", nullptr);
    if (fd) {
        fit_fields_buf g; if (fit_fields_load(fd, 0, 0, 0, &g)) { fprintf(stderr, "wraps: cannot read the fields in %s\n", fd); return 1; }
        int pool = 1; while ((double)g.shape[0] * g.shape[1] * g.shape[2] / ((double)pool * pool * pool) > 1e9) pool *= 2;
        wh = (double)g.h * pool; fit_fields_free(&g);
        if (!(wf = fit_field_array(fd, "w", pool, wsh))) { fprintf(stderr, "wraps: cannot read the data weight in %s\n", fd); return 1; }
    }
    double zlo = 0, zhi = Z;   /* --zr: the rows' global native z range, box-local */
    if (*opt(argc, argv, "--zr", "")) {
        double a, b; if (sscanf(opt(argc, argv, "--zr", ""), "%lf,%lf", &a, &b) != 2 || !(b >= a)) { fprintf(stderr, "wraps: --zr Z0,Z1 (global native z, Z0 <= Z1)\n"); return 2; }
        zlo = fmax(0, a - m->origin[0]); zhi = fmin(Z, b - m->origin[0]);
        if (!(zhi >= zlo)) { fprintf(stderr, "wraps: --zr %s misses the box (global z %lld..%.0f)\n", opt(argc, argv, "--zr", ""), m->origin[0], m->origin[0] + Z); return 2; }
    }
    double wmin = 1e300, wmax = -1e300;   /* the wraps that meet the box (its rows' z range) */
    for (int a = 0; a <= 16; a++) for (int b = 0; b <= 16; b++) for (int c = 0; c <= 16; c++) {
        const double w = fit_set_W(fs, zlo + (zhi - zlo) * a / 16, Y * b / 16, X * c / 16); if (w < wmin) wmin = w; if (w > wmax) wmax = w;
    }
    int k0 = (int)ceil(wmin), k1 = (int)floor(wmax);   /* (the samples can miss W near the axis: --first / --last are taken as given) */
    if (*opt(argc, argv, "--first", "")) k0 = atoi(opt(argc, argv, "--first", ""));
    if (*opt(argc, argv, "--last", "")) k1 = atoi(opt(argc, argv, "--last", ""));
    const char *cr = opt(argc, argv, "--col-r", "p90"); const double colr = !strcmp(cr, "centre") ? -1 : !strcmp(cr, "p90") ? 0 : atof(cr);
    if (strcmp(cr, "centre") && strcmp(cr, "p90") && !(colr > 0)) { fprintf(stderr, "wraps: --col-r centre|p90|R (R > 0 native voxels)\n"); return 2; }
    const int na = (int)Z + 1;
    double th0 = 1e300, th1 = -1e300; int full = 0;   /* the box's angular range around the axis */
    for (int zi = 0; zi <= 8; zi++) {
        const double z = Z * zi / 8, cy = m->cy[(int)fmin(z, na - 1)], cx = m->cx[(int)fmin(z, na - 1)];
        if (cy >= 0 && cy <= Y && cx >= 0 && cx <= X) full = 1;
        for (int c = 0; c < 4; c++) { const double t = atan2((c & 1 ? Y : 0) - cy, (c & 2 ? X : 0) - cx); if (t < th0) th0 = t; if (t > th1) th1 = t; }
    }
    if (!full && th1 - th0 > M_PI) {   /* the range straddles the -pi / pi cut: shift to [0, 2 pi) */
        th0 = 1e300; th1 = -1e300;
        for (int zi = 0; zi <= 8; zi++) {
            const double z = Z * zi / 8, cy = m->cy[(int)fmin(z, na - 1)], cx = m->cx[(int)fmin(z, na - 1)];
            for (int c = 0; c < 4; c++) { double t = atan2((c & 1 ? Y : 0) - cy, (c & 2 ? X : 0) - cx); if (t < 0) t += 2 * M_PI; if (t < th0) th0 = t; if (t > th1) th1 = t; }
        }
    }
    if (full) { th0 = -M_PI; th1 = M_PI; }
    const int rows = (int)floor((zhi - zlo) / step) + 1;
    /* the wraps on threads, each wrap's rows on one (a wrap's rows on all threads ran ~1.7 of 6 cores on a small box: per-wrap thread
       start-up, the serial tifxyz write); with snapping, one wrap at a time over all threads as before. Each wrap is computed as before. */
    wraps_all A = {fs, m, wf, {wsh[0], wsh[1], wsh[2]}, wh, step, Z, na, th0, th1, rows, k0, k1, nthreads, um, out_um, argv[3], snapz, snapo, snap_range, snap_min, sfa,
                   {sfn[0], sfn[1], sfn[2]}, {sfo[0], sfo[1], sfo[2]}, sfh, verbose_snap, Y, X, nullptr, 0, 0, 0, 0, zlo, zhi, colr};
    wr_polar P = {0};
    if (wf && !getenv("UFSM_WRAPS_NOBOUND")) {   /* the rays' radius bounds (the table spans the box's angles, bins ~1e-3 rad) */
        P.rows = rows; P.t0 = th0 - 0.01; P.nb = (int)fmin(8192, fmax(64, ceil((th1 - th0 + 0.02) / 1e-3))); P.bw = (th1 - th0 + 0.02) / P.nb;
        P.rmin = malloc((size_t)rows * P.nb * sizeof(float)); P.rmax = malloc((size_t)rows * P.nb * sizeof(float));
        wr_polar_job PJ = {&P, m, wf, {wsh[0], wsh[1], wsh[2]}, wh, step, Z, na, 0, zlo};
        pthread_t th[64]; const int nt = nthreads < 64 ? (nthreads > 0 ? nthreads : 1) : 64;
        for (int t = 0; t < nt; t++) pthread_create(&th[t], nullptr, wr_polar_rows, &PJ);
        for (int t = 0; t < nt; t++) pthread_join(th[t], nullptr);
        A.P = &P;
    }
    const int nall = k1 >= k0 ? k1 - k0 + 1 : 0; int *ks = malloc(sizeof(int) * (nall > 0 ? nall : 1)), nk = 0;
    _Atomic long long *cnt = calloc(nall > 0 ? nall : 1, sizeof *cnt);
    if (wf && nall > 0 && !ufsm_env_on("UFSM_WRAPS_NOPRUNE")) {
        wr_prune_job PJ = {fs, wf, {wsh[0], wsh[1], wsh[2]}, wh, zlo, zhi, 1 + 0.5 * sqrt(3.0) * wh / 4, k0, nall, cnt, 0};
        pthread_t th[64]; const int nt = nthreads < 64 ? (nthreads > 0 ? nthreads : 1) : 64;
        for (int t = 0; t < nt; t++) pthread_create(&th[t], nullptr, wr_prune_rows, &PJ);
        for (int t = 0; t < nt; t++) pthread_join(th[t], nullptr);
        for (int k = k0; k <= k1; k++) if (cnt[k - k0]) ks[nk++] = k;
        g_wr_cnt = (const long long *)cnt; g_wr_k0 = k0; qsort(ks, nk, sizeof(int), wr_cmp_cnt);
        fprintf(stderr, "wraps: %d of %d wrap ids (%d..%d) meet the data, largest first\n", nk, nall, k0, k1);
    } else for (int k = k0; k <= k1; k++) ks[nk++] = k;
    A.ks = ks; A.nk = nk; A.min_cm2 = atof(opt(argc, argv, "--min-cm2", "0.5"));
    tfx_set_threads(nthreads);   /* (the .sfc encoders: shared out among the wraps being written at that moment) */
    const int par = !(snapz || sfa) && nthreads > 1;
    if (par) {
        pthread_t th[64]; const int nt = nthreads < 64 ? nthreads : 64;
        for (int t = 0; t < nt; t++) pthread_create(&th[t], nullptr, wraps_worker, &A);
        for (int t = 0; t < nt; t++) pthread_join(th[t], nullptr);
    } else for (int i = 0; i < nk; i++) if (wrap_one(&A, ks[i], nthreads)) A.fail = 1;
    free(P.rmin); free(P.rmax); free(ks); free(cnt);
    if (A.fail) return 1;
    const int nwr = A.nwr; const size_t nvalid_all = A.nvalid;
    fprintf(stderr, "wraps: %d wraps (%d..%d) with %zu vertices written to %s", nwr, k0, k1, nvalid_all, argv[3]);
    if (A.nsmall) fprintf(stderr, " (%d under --min-cm2 %g not written)", (int)A.nsmall, A.min_cm2);
    fputc('\n', stderr);
    fit_set_free(fs); free(wf); free(sfa); if (snapz) z3_close(snapz);
    return 0;
}

/* ufsm render <wraps-dir> <out-dir> --volume ROOT KEY [--ct ROOT KEY] [--cache DIR] [--scale 1] [--depth 8] [--tile 512] [--threads 16]
   Per wrap (each <wraps-dir>/wNNN tifxyz): the vertex grid bilinearly interpolated to scale pixels per native voxel, the
   surface normal from the grid derivatives, the volume (an ink zarr level, uint8) sampled trilinearly at t = -depth..depth native
   voxels along it and max-composited into <out-dir>/wNNN.tif (uint8); --ct also the CT at t = 0 into wNNN_ct.tif. Pixels whose
   cell has an invalid corner are 0. Rendered in output tiles of --tile pixels, each reading only the volume around its own
   vertices (+- depth), its pixel rows over --threads: any wrap size in bounded memory. */
static float trilin(const uint8_t *v, const int64_t n[3], double z, double y, double x) {
    if (z < 0 || y < 0 || x < 0 || z > n[0] - 1 || y > n[1] - 1 || x > n[2] - 1) return 0.f;
    int z0 = (int)z, y0 = (int)y, x0 = (int)x;
    if (z0 >= n[0] - 1) z0 = (int)n[0] - 2;
    if (y0 >= n[1] - 1) y0 = (int)n[1] - 2;
    if (x0 >= n[2] - 1) x0 = (int)n[2] - 2;
    z0 = z0 < 0 ? 0 : z0; y0 = y0 < 0 ? 0 : y0; x0 = x0 < 0 ? 0 : x0;
    const double fz = z - z0, fy = y - y0, fx = x - x0; double s = 0;
    for (int c = 0; c < 8; c++) {
        const int a = z0 + (c >> 2 & 1), b = y0 + (c >> 1 & 1), e = x0 + (c & 1);
        if (a >= n[0] || b >= n[1] || e >= n[2]) continue;
        s += (c >> 2 & 1 ? fz : 1 - fz) * (c >> 1 & 1 ? fy : 1 - fy) * (c & 1 ? fx : 1 - fx) * v[((size_t)a * n[1] + b) * n[2] + e];
    }
    return (float)s;
}
static float *read_tif_f32(const char *path, int *w, int *h) {
    if (tfx_axis_hook && access(path, F_OK)) return tfx_axis_hook(path, w, h);   /* a .sfc-backed wrap (src/tfx.c) */
    tiff *t = tiff_open_file(path); tiff_page pg;
    if (!t || tiff_page_info(t, 0, &pg) || pg.fmt != 3 || pg.bits != 32 || pg.spp != 1) { if (t) tiff_close(t); return nullptr; }
    float *d = malloc((size_t)pg.w * pg.h * 4);
    if (tiff_read_page(t, 0, d)) { free(d); tiff_close(t); return nullptr; }
    *w = pg.w; *h = pg.h; tiff_close(t); return d;
}
/* ufsm downsample <root> <key> <out-group> <out-um> --mode max|phase [--box z,y,x,nz,ny,nx] [--shard 512] [--cache DIR] [--threads 16]
   One uint8 array halved per axis (lossless volcomp) into <out-group>/<out-um> (an OME group of that one level): max (binary
   labels: any positive) or phase (the band phase codes, src/band.h: the circular mean of the known codes of the 2^3 block,
   PHASE_UNKNOWN where none is known or their resultant is under half their count). --box (source voxels) limits the work to
   the output shards it meets; others stay missing (the fill value). */
/* ufsm ct-hist <out.txt> <root> <key> z,y,x,n [z,y,x,n ...] [--cache DIR]
   The CT histogram of each box (n^3 at array key): one line of 256 counts per box (index 0 = masked air), boxes under 10% material
   skipped. The scanner augmentation's hist op (train / predict --aug-hist FILE --aug-hist-p P) maps a training window's material
   histogram onto one of these lines: boxes of the scans a model must generalise to (e.g. other scrolls). */
static int cmd_ct_hist(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: ufsm ct-hist <out.txt> <root> <key> z,y,x,n [z,y,x,n ...] [--cache DIR]\n"); return 2; }
    z3 *z = z3_open(store_open(argv[3]), argv[4], opt(argc, argv, "--cache", nullptr));
    if (!z) { fprintf(stderr, "ct-hist: %s\n", z3_error()); return 1; }
    FILE *f = fopen(argv[2], "w"); if (!f) { fprintf(stderr, "ct-hist: cannot write %s\n", argv[2]); return 1; }
    int nw = 0;
    for (int a = 5; a < argc; a++) {
        if (argv[a][0] == '-') { a++; continue; }
        long long v[4]; if (sscanf(argv[a], "%lld,%lld,%lld,%lld", v, v + 1, v + 2, v + 3) != 4 || v[3] < 1 || v[3] > 1024) { fprintf(stderr, "ct-hist: box %s: z,y,x,n (n <= 1024)\n", argv[a]); fclose(f); return 2; }
        const int64_t o[3] = {v[0], v[1], v[2]}, n[3] = {v[3], v[3], v[3]}; const size_t T = (size_t)v[3] * v[3] * v[3];
        uint8_t *b = malloc(T); if (!b) { fclose(f); return 1; }
        if (z3_read(z, o, n, b, 16)) { fprintf(stderr, "ct-hist: %s\n", z3_error()); free(b); fclose(f); return 1; }
        size_t h[256] = {0}; for (size_t k = 0; k < T; k++) h[b[k]]++;
        free(b);
        if (T - h[0] < T / 10) { fprintf(stderr, "ct-hist: box %s: %.1f%% material, skipped\n", argv[a], 100.0 * (T - h[0]) / T); continue; }
        for (int k = 0; k < 256; k++) fprintf(f, "%zu%c", h[k], k == 255 ? '\n' : ' ');
        nw++;
    }
    fclose(f); z3_close(z);
    fprintf(stderr, "ct-hist: %d histograms written to %s\n", nw, argv[2]);
    return nw ? 0 : 1;
}
/* cmd_downsample's pooling of one source slab (n[0] rows, even, from output row 0 of dst): output rows split over threads; phase
   mode with cos / sin tables (the same doubles as cos / sin of each code) */
static double g_ds_cs[PHASE_PERIOD], g_ds_sn[PHASE_PERIOD];
static void ds_tables(void) { for (int k = 0; k < PHASE_PERIOD; k++) { g_ds_cs[k] = cos(2 * M_PI * k / PHASE_PERIOD); g_ds_sn[k] = sin(2 * M_PI * k / PHASE_PERIOD); } }
static uint8_t g_ds_vr[256], g_ds_vv[256];   /* --mode votes: vote code -> recto / verso counts (src/votes.h) */
static void ds_vote_tables(void) { for (int k = 0; k < 256; k++) { int r, v; vote_split((uint8_t)k, &r, &v); g_ds_vr[k] = (uint8_t)r; g_ds_vv[k] = (uint8_t)v; } }
typedef struct { const uint8_t *src; int64_t n[3]; uint8_t *dst; int S; int64_t ny, nx, na; int phase; atomic_long next; } ds_job;   /* phase: 1 phase, 2 votes */
static void *ds_worker(void *arg) {
    ds_job *jb = arg; const int64_t *n = jb->n; const int S = jb->S;
    for (int64_t a; (a = atomic_fetch_add(&jb->next, 1)) < jb->na;) {
        const uint8_t *r0 = jb->src + (size_t)(2 * a) * n[1] * n[2], *r1 = 2 * a + 1 < n[0] ? r0 + (size_t)n[1] * n[2] : nullptr;
        for (int64_t b = 0; b < jb->ny; b++) {
            uint8_t *out = jb->dst + ((size_t)a * S + b) * S;
            const int64_t y0 = 2 * b, y1 = 2 * b + 1 < n[1] ? 2 * b + 1 : -1;
            for (int64_t c = 0; c < jb->nx; c++) {
                const int64_t x0 = 2 * c, x1 = 2 * c + 1 < n[2] ? 2 * c + 1 : -1;
                int v[8], nv = 0;
                for (int k = 0; k < 8; k++) {   /* the old order: z, y, x bits */
                    const uint8_t *r = k >> 2 & 1 ? r1 : r0; const int64_t yy = k >> 1 & 1 ? y1 : y0, xx = k & 1 ? x1 : x0;
                    if (r && yy >= 0 && xx >= 0) v[nv++] = r[(size_t)yy * n[2] + xx];
                }
                if (!jb->phase) { int mx = 0; for (int k = 0; k < nv; k++) mx = v[k] > mx ? v[k] : mx; out[c] = (uint8_t)mx; continue; }
                if (jb->phase == 2) { int r = 0, vv = 0; for (int k = 0; k < nv; k++) { r += g_ds_vr[v[k]]; vv += g_ds_vv[v[k]]; } out[c] = vote_code(r, vv); continue; }
                double cs = 0, sn = 0; int known = 0;
                for (int k = 0; k < nv; k++) if (v[k] < PHASE_PERIOD) { cs += g_ds_cs[v[k]]; sn += g_ds_sn[v[k]]; known++; }
                uint8_t o = PHASE_UNKNOWN;
                if (known && hypot(cs, sn) >= 0.5 * known) { double t = atan2(sn, cs) / (2 * M_PI); if (t < 0) t += 1; o = (uint8_t)((int)lround(t * PHASE_PERIOD) % PHASE_PERIOD); }
                out[c] = o;
            }
        }
    }
    return nullptr;
}
static void ds_run(ds_job *jb, int nt) {
    atomic_store(&jb->next, 0);
    nt = nt < 1 ? 1 : nt > 64 ? 64 : nt;
    pthread_t th[64]; int k = 0;
    for (; k < nt - 1 && pthread_create(&th[k], nullptr, ds_worker, jb) == 0; k++) {}
    ds_worker(jb);
    for (int i = 0; i < k; i++) pthread_join(th[i], nullptr);
}
static int cmd_downsample(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: ufsm downsample <root> <key> <out-group> <out-um> --mode max|phase|votes [--box z,y,x,nz,ny,nx] [--shard 512] [--cache DIR] [--threads 16]\n"); return 2; }
    const char *mode = opt(argc, argv, "--mode", "max"); const int votes = !strcmp(mode, "votes"), phase = !strcmp(mode, "phase") ? 1 : votes ? 2 : 0;
    if (!phase && strcmp(mode, "max")) { fprintf(stderr, "downsample: --mode max|phase|votes\n"); return 2; }
    if (votes) ds_vote_tables();
    const int S = atoi(opt(argc, argv, "--shard", "512")), nt = ufsm_threads(atoi(opt(argc, argv, "--threads", "0")), 16); const double um = atof(argv[5]);
    z3 *z = z3_open(store_open(argv[2]), argv[3], opt(argc, argv, "--cache", nullptr));
    if (!z) { fprintf(stderr, "downsample: %s\n", z3_error()); return 1; }
    const z3_meta *m = z3_meta_of(z); const int fill = phase == 1 ? PHASE_UNKNOWN : m->fill;   /* votes: 0 = no votes */
    int64_t bo[3] = {0, 0, 0}, bn[3] = {m->shape[0], m->shape[1], m->shape[2]};
    if (*opt(argc, argv, "--box", "")) { long long v[6]; if (sscanf(opt(argc, argv, "--box", ""), "%lld,%lld,%lld,%lld,%lld,%lld", v, v + 1, v + 2, v + 3, v + 4, v + 5) != 6) { fprintf(stderr, "downsample: --box z,y,x,nz,ny,nx\n"); return 2; } for (int d = 0; d < 3; d++) { bo[d] = v[d]; bn[d] = v[3 + d]; } }
    const int64_t osh[3] = {(m->shape[0] + 1) / 2, (m->shape[1] + 1) / 2, (m->shape[2] + 1) / 2};
    char lv[32], od[1400], at[400]; z3w_level_name(um, lv, sizeof lv); snprintf(od, sizeof od, "%s/%s", argv[4], lv);
    snprintf(at, sizeof at, "{\"ufsm\":{\"content\":\"%s halved (%s)\"%s}}", argv[3], mode, m->label_binary ? ",\"encoding\":\"binary\"" : votes ? ",\"encoding\":\"side-votes\",\"code\":\"src/votes.h: the votes of each 2^3 block added\"" : "");
    z3w *w = z3w_create(od, osh, S, 0, fill, at);
    double lum[16]; int nl = 0;   /* the group lists every level already in it (each a number-named array), sorted, plus this one */
    { DIR *gd = opendir(argv[4]); struct dirent *ge;
      while (gd && (ge = readdir(gd)) && nl < 15) { char *e; const double v = strtod(ge->d_name, &e); char zj[1400]; snprintf(zj, sizeof zj, "%s/%s/zarr.json", argv[4], ge->d_name);
          if (v > 0 && !*e && fabs(v - um) > 1e-6 && access(zj, R_OK) == 0) lum[nl++] = v; }
      if (gd) closedir(gd); }
    lum[nl++] = um;
    for (int a = 1; a < nl; a++) for (int b = a; b > 0 && lum[b] < lum[b - 1]; b--) { const double t = lum[b]; lum[b] = lum[b - 1]; lum[b - 1] = t; }
    if (!w || z3w_write_group(argv[4], lum, nl, "ufsm-downsample", nullptr)) { fprintf(stderr, "downsample: %s\n", z3w_error()); return 1; }
    /* per output shard: its 2S source cube read in slabs of the source chunk depth (each inner chunk decoded once; reading two planes
       at a time decoded every chunk it met once per plane pair), the slab pooled on nt threads; output shards whose source shards are
       all absent are skipped (all fill: z3w writes nothing for them either) */
    int64_t sd = m->chunk[0] > 0 ? m->chunk[0] : 128; if (sd % 2) sd *= 2; if (sd > 2 * (int64_t)S) sd = 2 * (int64_t)S;   /* source planes per slab */
    uint8_t *sb = malloc((size_t)S * S * S), *src = malloc((size_t)sd * 2 * S * 2 * S);
    if (!sb || !src) { fprintf(stderr, "downsample: out of memory\n"); return 1; }
    ds_tables();
    int nsh = 0, nskip = 0;
    for (int64_t sz = (bo[0] / 2) / S; sz * S < osh[0] && sz * S * 2 < bo[0] + bn[0]; sz++)
    for (int64_t sy = (bo[1] / 2) / S; sy * S < osh[1] && sy * S * 2 < bo[1] + bn[1]; sy++)
    for (int64_t sx = (bo[2] / 2) / S; sx * S < osh[2] && sx * S * 2 < bo[2] + bn[2]; sx++) {
        const int64_t so[3] = {sz, sy, sx}; int present = 0;   /* the source shards under this output shard */
        int64_t q0[3], q1[3];
        for (int d = 0; d < 3; d++) {
            const int64_t a = 2 * so[d] * S, e = 2 * (so[d] + 1) * S < m->shape[d] ? 2 * (so[d] + 1) * S : m->shape[d];
            q0[d] = a / m->shard[d]; q1[d] = (e - 1) / m->shard[d];
        }
        for (int64_t a = q0[0]; a <= q1[0] && !present; a++) for (int64_t b = q0[1]; b <= q1[1] && !present; b++) for (int64_t c = q0[2]; c <= q1[2] && !present; c++)
            present = z3_shard_present(z, a, b, c) != 0;   /* (-1, an error: read it and let the read report) */
        if (!present) { nskip++; continue; }
        memset(sb, fill, (size_t)S * S * S);
        const int64_t ny = osh[1] - sy * S < S ? osh[1] - sy * S : S, nx = osh[2] - sx * S < S ? osh[2] - sx * S : S;
        for (int64_t a0 = 0; a0 < S && sz * S + a0 < osh[0]; a0 += sd / 2) {   /* output rows a0 .. from source rows 2 a0 .. */
            const int64_t o[3] = {2 * (sz * S + a0), 2 * sy * S, 2 * sx * S};
            const int64_t n[3] = {m->shape[0] - o[0] < sd ? m->shape[0] - o[0] : sd, m->shape[1] - o[1] < 2 * ny ? m->shape[1] - o[1] : 2 * ny, m->shape[2] - o[2] < 2 * nx ? m->shape[2] - o[2] : 2 * nx};
            if (z3_read(z, o, n, src, nt)) { fprintf(stderr, "downsample: %s\n", z3_error()); return 1; }
            const size_t sn_ = (size_t)n[0] * n[1] * n[2]; size_t k = 0;
            while (k < sn_ && src[k] == (uint8_t)m->fill) k++;
            if (k == sn_ && fill == m->fill) continue;   /* all fill in, all fill out (max of fills; phase: unknown) */
            int64_t na = (n[0] + 1) / 2; if (a0 + na > S) na = S - a0; if (sz * S + a0 + na > osh[0]) na = osh[0] - sz * S - a0;
            ds_job jb = {src, {n[0], n[1], n[2]}, sb + (size_t)a0 * S * S, S, ny, nx, na, phase, 0};
            ds_run(&jb, nt);
        }
        if (z3w_write_shard(w, sz, sy, sx, sb, nt)) { fprintf(stderr, "downsample: %s\n", z3w_error()); return 1; }
        nsh++;
    }
    if (nskip) fprintf(stderr, "downsample: %d output shards skipped (no source shard)\n", nskip);
    z3w_close(w); free(sb); free(src); z3_close(z);
    fprintf(stderr, "downsample: %d shards of %s written to %s\n", nsh, lv, od);
    return 0;
}

/* ufsm side-votes <out-group> <out-um> <root> <key> [<root> <key> ...] [--shard 512] [--threads 16] [--cache DIR]
   Recto / verso side labels of several surface sets of one scroll (each phase-raster --pair 2 on one set's codes, or an earlier
   side-votes array) added into one vote array (src/votes.h: per voxel the sets calling it recto side and verso side, 15 at most),
   lossless volcomp in <out-group>/<out-um> (an OME group of that level; downsample --mode votes for the coarser ones). All inputs on
   one grid (equal shapes). Sources read it with "encoding": "side-votes" (unanimous / majority / soft).
   ufsm codes-merge (same arguments): winding_mod14 code rasters (the published band codes, raster --value of wind-assign's q.tif)
   united instead: codes within 4 steps are the same wrap (the first kept), else 255; phase-raster --pair 2 then labels between
   wraps from any of them. */
static int cmd_side_votes(int argc, char **argv) {
    const int codes = !strcmp(argv[1], "codes-merge");
    if (argc < 6) { fprintf(stderr, "usage: ufsm %s <out-group> <out-um> <root> <key> [<root> <key> ...] [--shard 512] [--threads 16] [--cache DIR]\n", argv[1]); return 2; }
    const int S = atoi(opt(argc, argv, "--shard", "512")), nt = ufsm_threads(atoi(opt(argc, argv, "--threads", "0")), 16); const double um = atof(argv[3]);
    if (S < 128 || S % 128 || !(um > 0)) { fprintf(stderr, "side-votes: out-um > 0, --shard a multiple of 128\n"); return 2; }
    z3 *in[64]; const char *names[64]; char nb[64][1200]; int nin = 0;
    for (int a = 4; a + 1 < argc && nin < 64; a += 2) {
        if (argv[a][0] == '-') break;
        in[nin] = z3_open(store_open(argv[a]), argv[a + 1], opt(argc, argv, "--cache", nullptr));
        if (!in[nin]) { fprintf(stderr, "side-votes: %s %s: %s\n", argv[a], argv[a + 1], z3_error()); return 1; }
        snprintf(nb[nin], sizeof nb[nin], "%s/%s", argv[a], argv[a + 1]); names[nin] = nb[nin]; nin++;
    }
    size_t st[2] = {0, 0}; int nsh = 0;
    const int rc = (codes ? codes_merge : side_votes_merge)(argv[2], um, in, names, nin, S, nt, &nsh, st);
    for (int i = 0; i < nin; i++) z3_close(in[i]);
    if (rc) { fprintf(stderr, "%s: %s\n", argv[1], side_votes_error()); return 1; }
    if (codes) fprintf(stderr, "codes-merge: %d inputs, %d shards -> %s; %zu coded voxels, %.2f%% of them conflicts (255)\n", nin, nsh, argv[2], st[0], st[0] ? 100.0 * st[1] / st[0] : 0.0);
    else fprintf(stderr, "side-votes: %d inputs, %d shards -> %s; %zu voxels with votes, %.2f%% of them disputed (recto and verso votes)\n", nin, nsh, argv[2], st[0], st[0] ? 100.0 * st[1] / st[0] : 0.0);
    return 0;
}

/* ufsm wind-assign <ref-root> <ref-key> <out-dir> <tifxyz>... [--ref-um 4.8] [--vol-um 2.4] [--kind codes|winding] [--tol 6]
   [--step 4] [--dir 0] [--axis JSON] [--axis-um U] [--drop-duplicates 0] [--threads 4]
   Winding ids for surfaces of other authors / runs (src/wind.h): each tifxyz gets q = K + dir theta / 2 pi per vertex in the gauge of
   a reference winding raster (codes: raster --value / the published band codes; winding: phase-raster --winding 1, dense), written
   as <out-dir>/<name> (x / y / z copied, q.tif float32, NaN where dropped or unassigned) for `ufsm raster --value q.tif`; one line
   per surface on stdout and in <out-dir>/assign.jsonl. --tol: label voxels from the reference that count as on it; --vol-um: the
   surfaces' scan voxel; --axis-um: the axis JSON's voxel (default --vol-um); --out-um: the voxel of the written coordinates (default
   --vol-um; the reference's --ref-um puts them on its grid for raster --value and codes-merge). */
typedef struct { z3 *z; int kind; int nt; } wa_ref;
typedef struct { int64_t key; size_t i; } wa_pt;
static int cmp_wa_pt(const void *a, const void *b) { const int64_t x = ((const wa_pt *)a)->key, y = ((const wa_pt *)b)->key; return x < y ? -1 : x > y; }
typedef struct { const wa_ref *R; const float *pts; const wa_pt *ord; const size_t *g; size_t ng; float tol; float *dist, *qref; atomic_size_t next; } wa_job;
static void *wa_blocks(void *arg) {   /* the point groups (one 128^3 block each) taken in turn: one haloed read per block */
    wa_job *J = arg; const wa_ref *R = J->R; const float *pts = J->pts; const int H = (int)ceilf(J->tol), B = 128, S = B + 2 * H; const z3_meta *m = z3_meta_of(R->z);
    uint8_t *blk = malloc((size_t)S * S * S);
    for (size_t gi; (gi = atomic_fetch_add(&J->next, 1)) < J->ng;) {
        const size_t a = J->g[gi], b = J->g[gi + 1], i0 = J->ord[a].i;
        const int64_t o[3] = {(int64_t)floorf(pts[3 * i0] / B) * B - H, (int64_t)floorf(pts[3 * i0 + 1] / B) * B - H, (int64_t)floorf(pts[3 * i0 + 2] / B) * B - H}, nn[3] = {S, S, S};
        int inside = 1; for (int d = 0; d < 3; d++) inside &= o[d] + S > 0 && o[d] < m->shape[d];
        const int ok = blk && inside && !z3_read(R->z, o, nn, blk, 1);
        for (size_t t = a; t < b; t++) {
            const size_t i = J->ord[t].i; float *dist = J->dist, *qref = J->qref; dist[i] = INFINITY; qref[i] = NAN;
            if (!ok) continue;
            const int z0 = (int)lroundf(pts[3 * i]) - (int)o[0], y0 = (int)lroundf(pts[3 * i + 1]) - (int)o[1], x0 = (int)lroundf(pts[3 * i + 2]) - (int)o[2];
            for (int z = z0 - H; z <= z0 + H; z++) for (int y = y0 - H; y <= y0 + H; y++) for (int x = x0 - H; x <= x0 + H; x++) {
                if (z < 0 || z >= S || y < 0 || y >= S || x < 0 || x >= S) continue;
                const float q = wind_code_q(blk[((size_t)z * S + y) * S + x], R->kind);
                if (isnan(q)) continue;
                const float dz = z + o[0] - pts[3 * i], dy = y + o[1] - pts[3 * i + 1], dx = x + o[2] - pts[3 * i + 2], d = sqrtf(dz * dz + dy * dy + dx * dx);
                if (d < dist[i]) { dist[i] = d; qref[i] = q; }
            }
        }
    }
    free(blk); return nullptr;
}
static void wa_lookup(void *ud, const float *pts, size_t n, float tol, float *dist, float *qref) {   /* per 128^3 block of points: one haloed read, R->nt threads */
    const wa_ref *R = ud; const int B = 128;
    wa_pt *ord = malloc(n * sizeof *ord); size_t *g = malloc((n + 1) * sizeof *g);
    if (!ord || !g) { free(ord); free(g); for (size_t i = 0; i < n; i++) { dist[i] = INFINITY; qref[i] = NAN; } return; }
    for (size_t i = 0; i < n; i++) {
        const int64_t bz = (int64_t)floorf(pts[3 * i] / B), by = (int64_t)floorf(pts[3 * i + 1] / B), bx = (int64_t)floorf(pts[3 * i + 2] / B);
        ord[i] = (wa_pt){((bz + 4096) << 40) | ((by + 4096) << 20) | (bx + 4096), i};
    }
    qsort(ord, n, sizeof *ord, cmp_wa_pt);
    size_t ng = 0; for (size_t a = 0; a < n; a++) if (!a || ord[a].key != ord[a - 1].key) g[ng++] = a;
    g[ng] = n;
    wa_job J = {R, pts, ord, g, ng, tol, dist, qref, 0};
    const int nt = R->nt < 1 ? 1 : R->nt > 64 ? 64 : R->nt; pthread_t th[64]; int k = 0;
    for (; k < nt - 1 && pthread_create(&th[k], nullptr, wa_blocks, &J) == 0; k++) {}
    wa_blocks(&J);
    for (int t = 0; t < k; t++) pthread_join(th[t], nullptr);
    free(ord); free(g);
}
typedef struct { axis ax; double f; } wa_axis;
static void wa_axis_fn(void *ud, double z, double *cy, double *cx) { const wa_axis *A = ud; axis_at(&A->ax, z / A->f, cy, cx); *cy *= A->f; *cx *= A->f; }
static float *wa_read_f32(const char *dir, const char *name, int *w, int *h) {
    char p[1400]; snprintf(p, sizeof p, "%s/%s", dir, name);
    if (tfx_axis_hook && access(p, F_OK)) {   /* a .sfc-backed wrap (src/tfx.c) */
        int w2 = 0, h2 = 0; float *a = tfx_axis_hook(p, &w2, &h2);
        if (a && *w && (w2 != *w || h2 != *h)) { free(a); return nullptr; }
        if (a) { *w = w2; *h = h2; }
        return a;
    }
    tiff *t = tiff_open_file(p); tiff_page pg;
    if (!t || tiff_page_info(t, 0, &pg) || pg.bits != 32 || pg.fmt != 3 || pg.spp != 1 || (*w && (pg.w != *w || pg.h != *h))) { if (t) tiff_close(t); return nullptr; }
    float *d = malloc((size_t)pg.w * pg.h * 4);
    if (!d || tiff_read_page(t, 0, d)) { free(d); tiff_close(t); return nullptr; }
    *w = pg.w; *h = pg.h; tiff_close(t); return d;
}
static int cmd_wind_assign(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: ufsm wind-assign <ref-root> <ref-key> <out-dir> <tifxyz>... [--ref-um 4.8] [--vol-um 2.4] [--kind codes|winding] [--tol 6] [--step 4] [--dir 0] [--axis JSON] [--axis-um U] [--drop-duplicates 0]\n"); return 2; }
    const double ref_um = atof(opt(argc, argv, "--ref-um", "4.8")), vol_um = atof(opt(argc, argv, "--vol-um", "2.4")), axis_um = atof(opt(argc, argv, "--axis-um", opt(argc, argv, "--vol-um", "2.4")));
    const char *kind = opt(argc, argv, "--kind", "codes");
    wind_params p = {vol_um / ref_um, (float)atof(opt(argc, argv, "--tol", "6")), atoi(opt(argc, argv, "--step", "4")), atoi(opt(argc, argv, "--dir", "0")),
                     atoi(opt(argc, argv, "--drop-duplicates", "0")), !strcmp(kind, "winding")};
    if (!(ref_um > 0 && vol_um > 0 && axis_um > 0) || (strcmp(kind, "codes") && strcmp(kind, "winding")) || !(p.tol >= 0 && p.tol <= 32) || p.step < 1 || (p.dir != 0 && p.dir != 1 && p.dir != -1)) {
        fprintf(stderr, "wind-assign: --ref-um / --vol-um / --axis-um > 0, --kind codes|winding, --tol 0..32, --step >= 1, --dir -1|0|1\n"); return 2;
    }
    wa_ref R = {z3_open(store_open(argv[2]), argv[3], opt(argc, argv, "--cache", nullptr)), p.dense ? WIND_WINDING : WIND_CODES, ufsm_threads(atoi(opt(argc, argv, "--threads", "0")), 4)};
    if (!R.z) { fprintf(stderr, "wind-assign: %s\n", z3_error()); return 1; }
    wa_axis A = {.f = axis_um / vol_um};
    if (axis_load(&A.ax, opt(argc, argv, "--axis", "/vesuvius/ufsm/umbilicus/PHercParis4/umbilicus-full-resolution.json"))) { fprintf(stderr, "wind-assign: cannot load the axis\n"); return 1; }
    char lp[1400]; snprintf(lp, sizeof lp, "%s/assign.jsonl", argv[4]);
    if (mkdir(argv[4], 0755) && errno != EEXIST) { fprintf(stderr, "wind-assign: mkdir %s\n", argv[4]); return 1; }
    FILE *log = fopen(lp, "a"); if (!log) { fprintf(stderr, "wind-assign: cannot write %s\n", lp); return 1; }
    int nsurf = 0, rc = 0;
    for (int a = 5; a < argc; a++) {
        if (argv[a][0] == '-') { a++; continue; }
        int w = 0, h = 0; float *X = wa_read_f32(argv[a], "x.tif", &w, &h), *Y = X ? wa_read_f32(argv[a], "y.tif", &w, &h) : nullptr, *Z = Y ? wa_read_f32(argv[a], "z.tif", &w, &h) : nullptr;
        if (!Z) { fprintf(stderr, "wind-assign: %s: x / y / z.tif must be matching float32 images\n", argv[a]); free(X); free(Y); rc = 1; continue; }
        const size_t N = (size_t)w * h; float *xyz = malloc(N * 3 * sizeof *xyz); uint8_t *valid = malloc(N);
        for (size_t i = 0; i < N; i++) {
            xyz[3 * i] = X[i]; xyz[3 * i + 1] = Y[i]; xyz[3 * i + 2] = Z[i];
            valid[i] = isfinite(X[i]) && isfinite(Y[i]) && isfinite(Z[i]) && Z[i] > 0 && X[i] >= 0;   /* (load_mesh's rule; -1 = invalid) */
        }
        wind_result r;
        if (wind_assign(xyz, valid, w, h, wa_axis_fn, &A, wa_lookup, &R, p, &r)) { fprintf(stderr, "wind-assign: out of memory\n"); return 1; }
        const char *base = strrchr(argv[a], '/'); base = base && base[1] ? base + 1 : argv[a];
        char od[1400], qp[1500]; snprintf(od, sizeof od, "%s/%s", argv[4], base);
        const double out_um = atof(opt(argc, argv, "--out-um", opt(argc, argv, "--vol-um", "2.4"))), f = vol_um / (out_um > 0 ? out_um : vol_um);   /* --out-um: coordinates in that voxel (e.g. the reference grid's, for raster --value) */
        if (f != 1.0) for (size_t i = 0; i < N; i++) if (valid[i]) { X[i] = (float)(X[i] * f); Y[i] = (float)(Y[i] * f); Z[i] = (float)(Z[i] * f); }
        if (tifxyz_write(od, w, h, X, Y, Z, 1.0, out_um > 0 ? out_um : vol_um, -1, "ufsm wind-assign (x / y / z of the input; q.tif: the winding in the reference's gauge)")) { fprintf(stderr, "wind-assign: cannot write %s\n", od); rc = 1; }
        snprintf(qp, sizeof qp, "%s/q.tif", od);
        if (tiffw_write(qp, w, h, 3, r.q, 1)) { fprintf(stderr, "wind-assign: cannot write %s\n", qp); rc = 1; }
        size_t nq = 0, nf[5] = {0}; for (size_t i = 0; i < N; i++) { nq += !isnan(r.q[i]); nf[r.flag[i]]++; }
        for (int c = 0; c < r.ncomp; c++) {
            const wind_comp *k = &r.comp[c];
            printf("%s comp %d (%zu vertices): %s dir %+d K %d (mod 14), %zu of %zu samples on the reference, %zu agree, median dist %.2f, |resid| %.3f turn\n",
                   base, c, k->nvalid, wind_verdict_name(k->verdict), k->dir, k->K, k->nmatch, k->nsample, k->nagree, k->median_dist, k->mean_abs_resid);
            fprintf(log, "{\"surface\":\"%s\",\"comp\":%d,\"vertices\":%zu,\"verdict\":\"%s\",\"dir\":%d,\"K\":%d,\"samples\":%zu,\"matched\":%zu,\"agree\":%zu,\"median_dist\":%.3f,\"resid\":%.4f,\"q_vertices\":%zu,\"dropped_conflict\":%zu,\"dropped_duplicate\":%zu,\"unassigned\":%zu}\n",
                    base, c, k->nvalid, wind_verdict_name(k->verdict), k->dir, k->K, k->nsample, k->nmatch, k->nagree, isfinite(k->median_dist) ? k->median_dist : -1.0, k->mean_abs_resid, nq, nf[2], nf[3], nf[4]);
        }
        wind_result_free(&r); free(xyz); free(valid); free(X); free(Y); free(Z); nsurf++;
    }
    fclose(log); z3_close(R.z);
    fprintf(stderr, "wind-assign: %d surfaces -> %s (then: ufsm raster OUT --value q.tif --binary 0 --levels 1 ... %s/*)\n", nsurf, argv[4], argv[4]);
    return rc;
}

/* ufsm wind-seed <out-dir> <tifxyz>... [--k K] [--axis JSON] [--axis-um U] [--vol-um 2.4] [--out-um U] [--sense set|surface|+1|-1]
   [--order auto|+1|-1] [--seam set|start] [--min-pairs 50] [--min-score 0.75] [--points 400000] [--dry 1] [--pitch-um-lo 50] [--pitch-um-hi 1000]
   [--min-surface-score 0.5]
   A surface set's own winding gauge (src/wind.h wind_seed), for the first set of a scroll (no reference raster): every tifxyz gets
   q on its largest component, K from --k or the segment name in its path (-wNNN_ / -wrapNN / /wNNN), written like wind-assign
   (<out-dir>/<name>: x / y / z (scaled to --out-um), q.tif) for `ufsm raster --value q.tif`; one line per surface on stdout and in
   <out-dir>/seed.jsonl, the set's choice in <out-dir>/set.json.
   --sense set (default): one spiral sense for the whole set, and --order auto the order of its names (inward-counting names get
   K -> -K), both from the set's geometry (wind_seed_decide: vertex pairs on one ray from the axis at one height, the outer one must
   carry the higher q; pairs inside a surface spanning over a turn and, with names, across surfaces). A surface whose own pairs
   contradict the choice (agree share under --min-surface-score, default 0.5; >= --min-pairs votes) is DROPPED: reported, not written. surface: each surface its
   own pairs' sense (the radius rule where it has none); +1 / -1: forced (contradicting surfaces still dropped). The old end-gap
   radius vote (wind_seed_set) is only the tie-break without any pair. Sets of single wraps: the sense only moves q within each
   wrap's turn [K, K + 1) (the seam rule), which neither the pair-rule phase (the radial position between the nearest sheets of
   opposite parity, rising toward the higher winding) nor the winding mod 4 unwrapped from it sees (PHerc0172 seeded -1 and +1:
   identical stored phase); there the names' order alone sets the phase's direction. Exit 3 when the set's agreement score is under
   --min-score (>= 100 votes): its q would not rise outward; exit 4 when the sense is undetermined (margin over the other sense under
   0.02) while a surface spans >= 1.2 turns (its integer windings, hence the phase, follow the sense) and not forced
   (--allow-undetermined 1 writes anyway); nothing is written either way. --dry 1: the
   decision and each surface's votes only (set.json). A median pitch of the one-turn pairs outside --pitch-um-lo .. hi means they are
   not sheets (phi unwrapping jumps of partial segments): no evidence.
   --seam set (default): the turn boundary at the circular mean of the start azimuths (wraps of one numbering share it), q = K on
   the turn holding a short surface's median; start: q = K at each surface's start. Reported: names out of radial order (consecutive
   K whose mean radii decrease), repeated K. */
static int seed_k_from_path(const char *path, double *K) {
    for (const char *p = path; *p; p++) {
        if ((*p == '-' || *p == '_' || *p == '/') && p[1] == 'w') {
            const char *d = p + 2; if (!strncmp(d, "rap", 3)) d += 3;
            if (*d >= '0' && *d <= '9') { *K = atof(d); return 0; }
        }
    }
    return -1;
}
typedef struct { float *X, *Y, *Z, *xyz; uint8_t *valid; int w, h; } ws_surf;
static int ws_read(const char *path, ws_surf *S) {
    memset(S, 0, sizeof *S);
    S->X = wa_read_f32(path, "x.tif", &S->w, &S->h); S->Y = S->X ? wa_read_f32(path, "y.tif", &S->w, &S->h) : nullptr; S->Z = S->Y ? wa_read_f32(path, "z.tif", &S->w, &S->h) : nullptr;
    if (!S->Z) { free(S->X); free(S->Y); return -1; }
    const size_t N = (size_t)S->w * S->h; S->xyz = malloc(N * 3 * sizeof *S->xyz); S->valid = malloc(N);
    if (!S->xyz || !S->valid) return -1;
    for (size_t i = 0; i < N; i++) {
        S->xyz[3 * i] = S->X[i]; S->xyz[3 * i + 1] = S->Y[i]; S->xyz[3 * i + 2] = S->Z[i];
        S->valid[i] = isfinite(S->X[i]) && isfinite(S->Y[i]) && isfinite(S->Z[i]) && S->Z[i] > 0 && S->X[i] >= 0;
    }
    return 0;
}
static void ws_free(ws_surf *S) { free(S->X); free(S->Y); free(S->Z); free(S->xyz); free(S->valid); }
typedef struct { double K, rmean; int idx; } ws_kr;
static int cmp_ws_kr(const void *a, const void *b) { const ws_kr *x = a, *y = b; return x->K < y->K ? -1 : x->K > y->K ? 1 : x->idx - y->idx; }
static int cmd_wind_seed(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm wind-seed <out-dir> <tifxyz>... [--k K] [--axis JSON] [--axis-um U] [--vol-um 2.4] [--out-um U] [--sense set|surface|+1|-1] [--order auto|+1|-1] [--seam set|start] [--min-pairs 50] [--min-score 0.75]\n"); return 2; }
    const double vol_um = atof(opt(argc, argv, "--vol-um", "2.4")), axis_um = atof(opt(argc, argv, "--axis-um", opt(argc, argv, "--vol-um", "2.4")));
    const double out_um = atof(opt(argc, argv, "--out-um", opt(argc, argv, "--vol-um", "2.4")));
    const char *sense_o = opt(argc, argv, "--sense", "set"), *seam_o = opt(argc, argv, "--seam", "set"), *order_o = opt(argc, argv, "--order", "auto");
    const int sense_set = !strcmp(sense_o, "set"), sense_surf = !strcmp(sense_o, "surface"), seam_set = !strcmp(seam_o, "set");
    const size_t min_pairs = (size_t)atol(opt(argc, argv, "--min-pairs", "50")), npts = (size_t)atol(opt(argc, argv, "--points", "400000"));
    const double min_score = atof(opt(argc, argv, "--min-score", "0.75"));
    if (!(vol_um > 0 && axis_um > 0 && out_um > 0)) { fprintf(stderr, "wind-seed: --vol-um / --axis-um / --out-um > 0\n"); return 2; }
    if (!sense_set && !sense_surf && strcmp(sense_o, "+1") && strcmp(sense_o, "-1") && strcmp(sense_o, "1")) { fprintf(stderr, "wind-seed: --sense set|surface|+1|-1\n"); return 2; }
    if (!seam_set && strcmp(seam_o, "start")) { fprintf(stderr, "wind-seed: --seam set|start\n"); return 2; }
    if (strcmp(order_o, "auto") && strcmp(order_o, "+1") && strcmp(order_o, "1") && strcmp(order_o, "-1")) { fprintf(stderr, "wind-seed: --order auto|+1|-1\n"); return 2; }
    const int force_sense = sense_set || sense_surf ? 0 : atoi(sense_o) > 0 ? 1 : -1, force_order = !strcmp(order_o, "auto") ? 0 : atoi(order_o) > 0 ? 1 : -1;
    wa_axis A = {.f = axis_um / vol_um};
    if (axis_load(&A.ax, opt(argc, argv, "--axis", "/vesuvius/ufsm/umbilicus/PHercParis4/umbilicus-full-resolution.json"))) { fprintf(stderr, "wind-seed: cannot load the axis\n"); return 1; }
    char lp[1400]; snprintf(lp, sizeof lp, "%s/seed.jsonl", argv[2]);
    if (mkdir(argv[2], 0755) && errno != EEXIST) { fprintf(stderr, "wind-seed: mkdir %s\n", argv[2]); return 1; }
    const char *kopt = opt(argc, argv, "--k", nullptr);
    /* pass 1: each surface's geometry (sense 0: its radius rule, start azimuths) */
    int ns = 0, *idx = malloc(argc * sizeof *idx); double *Ks = malloc(argc * sizeof *Ks); wind_seed_info *i1 = calloc(argc, sizeof *i1);
    int rc = 0;
    for (int a = 3; a < argc; a++) {
        if (argv[a][0] == '-') { a++; continue; }
        double K = 0;
        if (kopt) K = atof(kopt);
        else if (seed_k_from_path(argv[a], &K)) { fprintf(stderr, "wind-seed: %s: no wNNN / wrapNN in the path (give --k)\n", argv[a]); rc = 1; continue; }
        ws_surf S; if (ws_read(argv[a], &S)) { fprintf(stderr, "wind-seed: %s: x / y / z.tif must be matching float32 images\n", argv[a]); ws_free(&S); rc = 1; continue; }
        wind_result r; wind_seed_info in;
        const int e = wind_seed(S.xyz, S.valid, S.w, S.h, wa_axis_fn, &A, K, 0, NAN, &in, &r);
        if (e == -1) { fprintf(stderr, "wind-seed: out of memory\n"); return 1; }
        ws_free(&S); if (e) { fprintf(stderr, "wind-seed: %s: no valid vertex\n", argv[a]); continue; }
        wind_result_free(&r);
        idx[ns] = a; Ks[ns] = K; i1[ns++] = in;
    }
    int vsense = 1; double phi_v = 0, vote[2] = {0, 0}, res = 0; wind_seed_set(i1, ns, &vsense, &phi_v, vote, &res);
    double phis[2] = {NAN, NAN}, ress[2] = {0, 0};   /* the seam under sense +1 / -1 */
    if (seam_set && ns) for (int si = 0; si < 2; si++) wind_seed_seam(i1, ns, si ? -1 : 1, &phis[si], &ress[si]);
    /* pass 1b: q under both senses, a subsample of every surface -> the set's sense and order (wind_seed_decide) */
    wind_pt *pts = nullptr; size_t np = 0;
    for (int k = 0; k < ns; k++) {
        ws_surf S; if (ws_read(argv[idx[k]], &S)) { ws_free(&S); rc = 1; continue; }
        wind_result r[2]; wind_seed_info in;
        int e = 0; for (int si = 0; si < 2 && !e; si++) e = wind_seed(S.xyz, S.valid, S.w, S.h, wa_axis_fn, &A, Ks[k], si ? -1 : 1, phis[si], &in, &r[si]);
        if (e) { fprintf(stderr, "wind-seed: out of memory\n"); return 1; }
        size_t m = 0; wind_pt *p = wind_seed_points(S.xyz, S.valid, S.w, S.h, wa_axis_fn, &A, r[0].q, r[1].q, k, npts, &m);
        if (m) { wind_pt *t = realloc(pts, (np + m) * sizeof *pts); if (!t) { fprintf(stderr, "wind-seed: out of memory\n"); return 1; } pts = t; memcpy(pts + np, p, m * sizeof *p); np += m; }
        free(p); wind_result_free(&r[0]); wind_result_free(&r[1]); ws_free(&S);
    }
    wind_min_surface_score = atof(opt(argc, argv, "--min-surface-score", "0.5"));
    wind_pitch_lo = atof(opt(argc, argv, "--pitch-um-lo", "50")) / vol_um; wind_pitch_hi = atof(opt(argc, argv, "--pitch-um-hi", "1000")) / vol_um;   /* papyrus sheets: ~150-400 um apart */
    wind_choice ch; wind_surf_vote *pv = calloc(ns ? ns : 1, sizeof *pv);
    if (!pv || wind_seed_decide(pts, np, Ks, ns, !kopt, force_sense, force_order, vsense, min_pairs, &ch, pv)) { fprintf(stderr, "wind-seed: out of memory\n"); return 1; }
    free(pts);
    const int sense = ch.sense, order = ch.order; const double phi0 = seam_set && ns ? phis[sense > 0 ? 0 : 1] : NAN;
    fprintf(stderr, "wind-seed: candidates (agree share / votes): sense +1 order +1 %.3f / %zu, +1 -1 %.3f / %zu, -1 +1 %.3f / %zu, -1 -1 %.3f / %zu\n",
            ch.cand[0][0], ch.cand_n[0][0], ch.cand[0][1], ch.cand_n[0][1], ch.cand[1][0], ch.cand_n[1][0], ch.cand[1][1], ch.cand_n[1][1]);
    const int undetermined = !force_sense && !sense_surf && !(ch.evidence && ch.sense_margin >= 0.02);
    fprintf(stderr, "wind-seed: spiral sense %+d, names %s (%s; agree %zu, disagree %zu, score %.3f, margin over the other sense %.3f / the other order %.3f, pitch %.1f scan voxels = %.0f um%s, %d surface(s) dropped; old end-gap radius vote: + %.0f, - %.0f scan voxels)\n",
            sense, order > 0 ? "counting outward" : "counting inward (K -> -K)", force_sense ? "forced" : undetermined ? "the sense UNDETERMINED by the geometry" : "from vertex pairs on rays from the axis",
            ch.agree, ch.dis, ch.score, ch.sense_margin, ch.order_margin, ch.pitch, ch.pitch * vol_um, ch.pitch_ok ? "" : " NOT a papyrus pitch: votes discarded", ch.ndropped, vote[0], vote[1]);
    if (seam_set && ns) fprintf(stderr, "wind-seed: seam at azimuth %.3f turn (start azimuths' resultant length %.2f)\n", phi0, ress[sense > 0 ? 0 : 1]);
    { char sp[1400]; snprintf(sp, sizeof sp, "%s/set.json", argv[2]); FILE *f = fopen(sp, "w");
      if (f) { fprintf(f, "{\"sense\":%d,\"order\":%d,\"forced\":%d,\"sense_determined\":%d,\"score\":%.4f,\"sense_margin\":%.4f,\"order_margin\":%.4f,\"pitch\":%.2f,\"agree\":%zu,\"disagree\":%zu,\"evidence\":%d,\"dropped\":%d,\"surfaces\":%d,\"seam\":%.4f,\"endgap_vote\":[%.1f,%.1f]}\n",
                       sense, order, force_sense != 0, !undetermined, isfinite(ch.score) ? ch.score : -1.0, isfinite(ch.sense_margin) ? ch.sense_margin : -1.0, isfinite(ch.order_margin) ? ch.order_margin : -1.0,
                       ch.pitch, ch.agree, ch.dis, ch.evidence, ch.ndropped, ns, isfinite(phi0) ? phi0 : -1.0, vote[0], vote[1]); fclose(f); } }
    double maxspan = 0; for (int k = 0; k < ns; k++) if (!pv[k].dropped && i1[k].span > maxspan) maxspan = i1[k].span;
    if (undetermined && maxspan < 1.2) fprintf(stderr, "wind-seed: the sense is undetermined but every surface spans under 1.2 turns (%.2f): it moves q only within each turn, not the phase\n", maxspan);
    else if (undetermined && !atoi(opt(argc, argv, "--allow-undetermined", "0"))) {
        fprintf(stderr, "wind-seed: FAILED: the geometry does not tell the spiral sense (margin %.3f) and a surface spans %.2f turns (its windings follow the sense): nothing written (--sense +1|-1 from elsewhere, or --allow-undetermined 1)\n", ch.sense_margin, maxspan);
        free(pv); free(idx); free(Ks); free(i1); return 4;
    }
    if (ch.evidence && ch.agree + ch.dis >= 100 && !(ch.score >= min_score)) {
        fprintf(stderr, "wind-seed: FAILED: the best (sense, order) puts the higher q outward on only %.3f of %zu pairs (--min-score %.2f): nothing written\n", ch.score, ch.agree + ch.dis, min_score);
        free(pv); free(idx); free(Ks); free(i1); return 3;
    }
    if (atoi(opt(argc, argv, "--dry", "0"))) {   /* --dry 1: the decision and every surface's votes, nothing written but set.json */
        for (int k = 0; k < ns; k++) { const char *base = strrchr(argv[idx[k]], '/'); base = base && base[1] ? base + 1 : argv[idx[k]];
            printf("%s K %g%s span %.2f end-gap rule %+d own pairs + %zu - %zu (own %+d) votes agree %zu disagree %zu\n", base, Ks[k], pv[k].dropped ? " DROPPED" : "", i1[k].span, i1[k].radial_dir,
                   pv[k].own_pos, pv[k].own_neg, pv[k].own_dir, pv[k].agree, pv[k].dis); }
        free(pv); free(idx); free(Ks); free(i1); return 0;
    }
    FILE *log = fopen(lp, "a"); if (!log) { fprintf(stderr, "wind-seed: cannot write %s\n", lp); return 1; }
    int nsurf = 0, ndis = 0, nown = 0; ws_kr *kr = malloc((ns + 1) * sizeof *kr);
    for (int k = 0; k < ns; k++) {   /* pass 2: the set's sense, order and seam */
        const int a = idx[k]; const double K = order * Ks[k];
        const char *base = strrchr(argv[a], '/'); base = base && base[1] ? base + 1 : argv[a];
        const wind_surf_vote *v = &pv[k];
        if (v->dropped) {
            printf("%s K %g DROPPED: its pairs contradict the set's sense %+d / order %+d (agree %zu, disagree %zu; own pairs + %zu - %zu)\n", base, Ks[k], sense, order, v->agree, v->dis, v->own_pos, v->own_neg);
            fprintf(log, "{\"surface\":\"%s\",\"K\":%g,\"dropped\":1,\"agree\":%zu,\"disagree\":%zu,\"own_pos\":%zu,\"own_neg\":%zu,\"own_dir\":%d}\n", base, Ks[k], v->agree, v->dis, v->own_pos, v->own_neg, v->own_dir);
            continue;
        }
        const int sdir = sense_surf ? (v->own_dir ? v->own_dir : i1[k].radial_dir) : sense;
        ws_surf S; if (ws_read(argv[a], &S)) { ws_free(&S); rc = 1; continue; }
        const int w = S.w, h = S.h; const size_t N = (size_t)w * h;
        wind_result r; wind_seed_info in;
        const int e = wind_seed(S.xyz, S.valid, w, h, wa_axis_fn, &A, K, sdir, sense_surf ? NAN : phi0, &in, &r);
        if (e == -1) { fprintf(stderr, "wind-seed: out of memory\n"); return 1; }
        if (e == -2) { ws_free(&S); continue; }
        char od[1400], qp[1500]; snprintf(od, sizeof od, "%s/%s", argv[2], base);
        const double f = vol_um / out_um; float *X = S.X, *Y = S.Y, *Z = S.Z;
        if (f != 1.0) for (size_t i = 0; i < N; i++) if (S.valid[i]) { X[i] = (float)(X[i] * f); Y[i] = (float)(Y[i] * f); Z[i] = (float)(Z[i] * f); }
        if (tifxyz_write(od, w, h, X, Y, Z, 1.0, out_um, -1, "ufsm wind-seed (x / y / z of the input; q.tif: the set's own winding gauge)")) { fprintf(stderr, "wind-seed: cannot write %s\n", od); rc = 1; }
        snprintf(qp, sizeof qp, "%s/q.tif", od);
        if (tiffw_write(qp, w, h, 3, r.q, 1)) { fprintf(stderr, "wind-seed: cannot write %s\n", qp); rc = 1; }
        const int dis = in.radial_dir != sdir; ndis += dis; nown += v->own_dir && v->own_dir != sdir;
        printf("%s K %g%s dir %+d%s span %.2f turns, q start %.3f, r start %.0f end %.0f mean %.0f (scan voxels), pairs agree %zu disagree %zu (own + %zu - %zu), %zu of %zu valid vertices assigned\n",
               base, Ks[k], order < 0 ? " (as -K)" : "", in.dir, dis ? " (end-gap radius rule: other end inner)" : "", in.span, in.q_start, in.r_start, in.r_end, in.r_mean, v->agree, v->dis, v->own_pos, v->own_neg, in.nassigned, in.nvalid);
        fprintf(log, "{\"surface\":\"%s\",\"K\":%g,\"K_used\":%g,\"dir\":%d,\"radial_dir\":%d,\"own_dir\":%d,\"agree\":%zu,\"disagree\":%zu,\"own_pos\":%zu,\"own_neg\":%zu,\"span\":%.4f,\"q_start\":%.4f,\"r_start\":%.2f,\"r_end\":%.2f,\"r_mean\":%.2f,\"assigned\":%zu,\"valid\":%zu}\n",
                base, Ks[k], K, in.dir, in.radial_dir, v->own_dir, v->agree, v->dis, v->own_pos, v->own_neg, in.span, in.q_start, in.r_start, in.r_end, in.r_mean, in.nassigned, in.nvalid);
        kr[nsurf] = (ws_kr){K, in.r_mean, a};
        wind_result_free(&r); ws_free(&S); nsurf++;
    }
    fprintf(stderr, "wind-seed: %d of %d written surfaces' end-gap radius rule disagrees with their sense (not used), %d their own pairs' sense; %d dropped\n", ndis, nsurf, nown, ch.ndropped);
    if (!kopt && nsurf > 1) {   /* the names (as used) against the radial order */
        qsort(kr, nsurf, sizeof *kr, cmp_ws_kr); int inv = 0, rep = 0;
        for (int k = 1; k < nsurf; k++) {
            const char *b0 = strrchr(argv[kr[k - 1].idx], '/'), *b1 = strrchr(argv[kr[k].idx], '/');
            if (kr[k].K == kr[k - 1].K) { rep++; fprintf(stderr, "wind-seed: K %g named twice: %s, %s\n", kr[k].K, b0 ? b0 + 1 : "", b1 ? b1 + 1 : ""); }
            else if (kr[k].rmean < kr[k - 1].rmean) { inv++; fprintf(stderr, "wind-seed: out of radial order (mean radius, axis-sensitive): K %g (mean r %.0f) inside K %g (mean r %.0f)\n", kr[k].K, kr[k].rmean, kr[k - 1].K, kr[k - 1].rmean); }
        }
        fprintf(stderr, "wind-seed: names vs mean radii: %d inversions between consecutive K, %d repeated K\n", inv, rep);
    }
    free(kr); free(idx); free(Ks); free(i1); free(pv);
    fclose(log);
    fprintf(stderr, "wind-seed: %d surfaces -> %s\n", nsurf, argv[2]);
    return rc;
}

/* ufsm phase-fuse <out-group> <out-um> <root> <key> [<root> <key>]... [--tol 0.1] [--shard 512] [--threads 8] [--cache DIR]
   (up to FUSE_MAX sets; per shard only those with data around it are read; --box z,y,x,nz,ny,nx in output voxels: only the shards meeting it)
   ufsm recto-fuse <out-group> <out-um> <ct-root> <ct-key> <root> <key> [<root> <key>]... [--block 32] [--dup 3] [--shard 512] [--cache DIR]
   Several surface sets of one scroll, rasterised separately on one grid (src/fuse.h): phase-fuse writes the per-voxel circular mean
   where the sets agree within --tol turns, unknown (255) where they disagree; recto-fuse one recto line per --block, the set on the
   brightest CT (ct on the same grid), plus other sets' voxels farther than --dup voxels from it (other sheets). */
static int cmd_fuse(int argc, char **argv) {
    const int recto = !strcmp(argv[1], "recto-fuse"), first = recto ? 6 : 4;
    if (argc < first + 2) { fprintf(stderr, "usage: ufsm %s <out-group> <out-um> %s<root> <key> [<root> <key>]... %s\n", argv[1], recto ? "<ct-root> <ct-key> " : "", recto ? "[--block 32] [--dup 3]" : "[--tol 0.1]"); return 2; }
    int S = atoi(opt(argc, argv, "--shard", "0")); const int nt = ufsm_threads(atoi(opt(argc, argv, "--threads", "0")), 8); const double um = atof(argv[3]);
    const float tol = (float)atof(opt(argc, argv, "--tol", "0.1")); const int block = atoi(opt(argc, argv, "--block", "32")), dup = atoi(opt(argc, argv, "--dup", "3"));
    const char *cache = opt(argc, argv, "--cache", nullptr);
    z3 *in[FUSE_MAX]; int n = 0;
    for (int a = first; a + 1 < argc && n < FUSE_MAX; a += 2) {
        if (argv[a][0] == '-') break;
        in[n] = z3_open(store_open(argv[a]), argv[a + 1], cache); if (!in[n]) { fprintf(stderr, "%s: %s %s: %s\n", argv[1], argv[a], argv[a + 1], z3_error()); return 1; }
        n++;
    }
    if (!S) S = 512;
    long long bx[6] = {0, 0, 0, 0, 0, 0};   /* --box z,y,x,nz,ny,nx (output grid voxels): only the shards meeting it */
    if (*opt(argc, argv, "--box", "") && sscanf(opt(argc, argv, "--box", ""), "%lld,%lld,%lld,%lld,%lld,%lld", bx, bx + 1, bx + 2, bx + 3, bx + 4, bx + 5) != 6) { fprintf(stderr, "%s: --box z,y,x,nz,ny,nx\n", argv[1]); return 2; }
    z3 *ct = recto ? z3_open(store_open(argv[4]), argv[5], cache) : nullptr;
    if (!n || (recto && !ct) || !(um > 0) || S < 64 || block < 4 || dup < 0 || !(tol >= 0)) { fprintf(stderr, "%s: bad arguments or cannot open the CT\n", argv[1]); return 2; }
    const z3_meta *m0 = z3_meta_of(in[0]); int64_t shape[3] = {m0->shape[0], m0->shape[1], m0->shape[2]};
    for (int k = 1; k < n; k++) { const z3_meta *mk = z3_meta_of(in[k]); for (int d = 0; d < 3; d++) if (mk->shape[d] != shape[d]) { fprintf(stderr, "%s: input %d has another shape\n", argv[1], k); return 2; } }
    if (recto) { const z3_meta *mc = z3_meta_of(ct); for (int d = 0; d < 3; d++) if (mc->shape[d] < shape[d]) shape[d] = mc->shape[d]; }
    char lv[32], od[1400], at[600]; z3w_level_name(um, lv, sizeof lv); snprintf(od, sizeof od, "%s/%s", argv[2], lv);
    snprintf(at, sizeof at, recto ? "{\"ufsm\":{\"content\":\"recto lines of %d surface sets, one per %g^3 block (the brightest-CT set; others only > %d voxels away)\",\"encoding\":\"binary\"}}"
                                  : "{\"ufsm\":{\"content\":\"band field winding phase (src/band.h PHASE_*) of %d surface sets: circular mean where they agree within %g turn, else 255\"}}", n, recto ? (double)block : (double)tol, dup);
    if (!recto) snprintf(at, sizeof at, "{\"ufsm\":{\"content\":\"band field winding phase (src/band.h PHASE_*: 0..251, 255 unknown) of %d surface sets: circular mean where they agree within %g turn, else 255\"}}", n, (double)tol);
    z3w *w = recto ? z3w_create_mask(od, shape, S, at) : z3w_create(od, shape, S, 0, PHASE_UNKNOWN, at);
    if (!w || z3w_write_group(argv[2], &um, 1, recto ? "ufsm-recto-fuse" : "ufsm-phase-fuse", nullptr)) { fprintf(stderr, "%s: %s\n", argv[1], z3w_error()); return 1; }
    const int H = recto ? dup : 0; const int64_t E = S + 2 * H; const size_t L = (size_t)E * E * E;
    /* per shard only the sets with data around it are read: one (shard + 2 halo)^3 buffer per set present there (allocated as
       needed), so many small surface sets fit at 512^3 shards */
    uint8_t *pool[FUSE_MAX] = {0}, *cb = recto ? malloc(L) : nullptr, *ob = malloc(L), *sh = malloc((size_t)S * S * S); const uint8_t *act[FUSE_MAX]; int npool = 0;
    if ((recto && !cb) || !ob || !sh) { fprintf(stderr, "%s: out of memory\n", argv[1]); return 1; }
    size_t st[4] = {0, 0, 0, 0}, hist[PHASE_PERIOD / 2 + 1] = {0}; int nsh = 0, maxact = 0, nct = 0;
    for (int64_t sz = 0; sz * S < shape[0]; sz++) for (int64_t sy = 0; sy * S < shape[1]; sy++) for (int64_t sx = 0; sx * S < shape[2]; sx++) {
        const int64_t o[3] = {sz * S - H, sy * S - H, sx * S - H}, nn[3] = {E, E, E};
        if (bx[3] && ((sz + 1) * S <= bx[0] || sz * S >= bx[0] + bx[3] || (sy + 1) * S <= bx[1] || sy * S >= bx[1] + bx[4] || (sx + 1) * S <= bx[2] || sx * S >= bx[2] + bx[5])) continue;
        int na = 0;
        for (int k = 0; k < n; k++) {
            const z3_meta *mk = z3_meta_of(in[k]); int64_t lo[3], hi[3]; int present = 0;
            for (int d = 0; d < 3; d++) { lo[d] = (o[d] < 0 ? 0 : o[d]) / mk->shard[d]; const int64_t e = o[d] + E < shape[d] ? o[d] + E : shape[d]; hi[d] = (e - 1) / mk->shard[d]; }
            for (int64_t a = lo[0]; a <= hi[0] && !present; a++) for (int64_t b = lo[1]; b <= hi[1] && !present; b++) for (int64_t c = lo[2]; c <= hi[2] && !present; c++)
                present = z3_shard_present(in[k], a, b, c) != 0;
            if (!present) continue;
            if (na == npool) { if (!(pool[npool] = malloc(L))) { fprintf(stderr, "%s: out of memory (%d sets around one shard)\n", argv[1], na + 1); return 1; } npool++; }
            if (z3_read(in[k], o, nn, pool[na], nt)) { fprintf(stderr, "%s: %s\n", argv[1], z3_error()); return 1; }
            act[na] = pool[na]; na++;
        }
        if (!na) continue;
        if (na > maxact) maxact = na;
        if (recto) { const int need = na > 1 && fuse_recto_overlap(act, na, E, E, E, block, dup);   /* the CT only where two sets' traces meet */
                     if (need && z3_read(ct, o, nn, cb, nt)) { fprintf(stderr, "%s: %s\n", argv[1], z3_error()); return 1; }
                     nct += need; fuse_recto(act, na, need ? cb : nullptr, E, E, E, block, dup, ob, st, nt); }
        else fuse_phase(act, na, L, tol, ob, st, hist);
        for (int64_t z = 0; z < S; z++) for (int64_t y = 0; y < S; y++) memcpy(sh + ((size_t)z * S + y) * S, ob + ((size_t)(z + H) * E + y + H) * E + H, (size_t)S);
        if (z3w_write_shard(w, sz, sy, sx, sh, nt)) { fprintf(stderr, "%s: %s\n", argv[1], z3w_error()); return 1; }
        nsh++;
    }
    fprintf(stderr, "%s: at most %d of %d sets around one shard%s", argv[1], maxact, n, recto ? "" : "\n");
    if (recto) fprintf(stderr, ", CT read for %d of %d shards (traces of two sets close)\n", nct, nsh);
    z3w_close(w);
    if (recto) fprintf(stderr, "recto-fuse: %d shards -> %s; primary voxels %zu, other-sheet voxels added %zu, duplicate voxels dropped %zu (halo voxels counted too)\n", nsh, od, st[0], st[1], st[2]);
    else {
        fprintf(stderr, "phase-fuse: %d shards -> %s; voxels from one set %zu, fused %zu, conflicts (unknown) %zu\n", nsh, od, st[0], st[1], st[2]);
        size_t tot = 0, c = 0; for (int k = 0; k <= PHASE_PERIOD / 2; k++) tot += hist[k];
        if (tot) {   /* the sets' offsets where they overlap: quantiles of the largest pairwise difference, the share within some tolerances */
            fprintf(stderr, "phase-fuse: overlap %zu voxels, largest pairwise difference (turns):", tot);
            const double q[] = {.5, .75, .9, .95, .99}; int qi = 0;
            for (int k = 0; k <= PHASE_PERIOD / 2 && qi < 5; k++) { c += hist[k]; while (qi < 5 && c >= q[qi] * tot) fprintf(stderr, " p%g %.3f", 100 * q[qi++], (double)k / PHASE_PERIOD); }
            const double t[] = {.025, .05, .1, .15, .2, .3}; fprintf(stderr, "; within");
            for (int j = 0; j < 6; j++) { c = 0; for (int k = 0; k <= PHASE_PERIOD / 2 && k <= t[j] * PHASE_PERIOD; k++) c += hist[k]; fprintf(stderr, " %g: %.3f", t[j], (double)c / tot); }
            fprintf(stderr, "\n");
        }
    }
    for (int k = 0; k < npool; k++) free(pool[k]);
    for (int k = 0; k < n; k++) z3_close(in[k]);
    free(cb); free(ob); free(sh); if (ct) z3_close(ct);
    return 0;
}

/* ufsm wrapdist <wraps-dir> <labels-root> <labels-key> [--max 32] [--chunk 128] [--threads 16] [--per-wrap]
   How far the wraps' vertices (every <wraps-dir>/wNNN tifxyz) lie from the nearest labelled sheet voxel (label value != 0) of
   a label array whose key is its voxel size in um (e.g. 4.8; the wraps' voxel_size_um from meta.json scales the vertices onto
   it). Exact distance transform (src/edt.c) over the vertices' bounding box in z-chunks of --chunk label rows with --max rows
   of overlap: distances up to --max label voxels are exact, larger ones reported as beyond it. Memory ~5 bytes per voxel of
   one chunk. */
static float *wd_read(const char *path, int *w, int *h);
static int cmd_wrapdist(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: ufsm wrapdist <wraps-dir> <labels-root> <labels-key> [--max 32] [--chunk 128] [--threads 16] [--per-wrap] [--at T1,T2..] [--cache DIR]\n"); return 2; }
    const int M = atoi(opt(argc, argv, "--max", "32")), C = atoi(opt(argc, argv, "--chunk", "128")), nt = ufsm_threads(atoi(opt(argc, argv, "--threads", "0")), 16);
    const int per = arg_index(argc, argv, "--per-wrap") > 0; const double lum = atof(argv[4]);
    if (!(lum > 0) || M < 1 || C < 1) { fprintf(stderr, "wrapdist: the labels key must be its voxel size in um (e.g. 4.8); --max, --chunk >= 1\n"); return 2; }
    z3 *z = z3_open(store_open(argv[3]), argv[4], opt(argc, argv, "--cache", nullptr)); if (!z) { fprintf(stderr, "wrapdist: %s\n", z3_error()); return 1; }
    const z3_meta *zm = z3_meta_of(z);
    /* vertices of every wrap, in label voxels */
    size_t nv = 0, cap = 1 << 20; float (*v)[3] = malloc(cap * sizeof *v); int *wid = malloc(cap * sizeof(int)); int nw = 0, ids[4096];
    DIR *dd = opendir(argv[2]); struct dirent *de; if (!dd) { fprintf(stderr, "wrapdist: cannot open %s\n", argv[2]); return 1; }
    while ((de = readdir(dd)) && nw < 4096) {
        if (de->d_name[0] != 'w') continue;
        char p[1400]; int w, h, w2, h2, w3, h3;
        snprintf(p, sizeof p, "%s/%s/x.tif", argv[2], de->d_name); float *X = wd_read(p, &w, &h);
        snprintf(p, sizeof p, "%s/%s/y.tif", argv[2], de->d_name); float *Y = wd_read(p, &w2, &h2);
        snprintf(p, sizeof p, "%s/%s/z.tif", argv[2], de->d_name); float *Z = wd_read(p, &w3, &h3);
        snprintf(p, sizeof p, "%s/%s/meta.json", argv[2], de->d_name); double wum = 0;
        { char *b = tfx_meta_read(p); const char *q = b ? strstr(b, "\"voxel_size_um\":") : nullptr; if (q) wum = atof(q + 16); free(b); }
        if (!X || !Y || !Z || w != w2 || w != w3 || h != h2 || h != h3 || !(wum > 0)) { free(X); free(Y); free(Z); continue; }
        const double sc = wum / lum;
        for (size_t i = 0; i < (size_t)w * h; i++) {
            if (Z[i] < 0) continue;
            if (nv == cap) { cap *= 2; v = realloc(v, cap * sizeof *v); wid = realloc(wid, cap * sizeof(int)); }
            v[nv][0] = (float)(Z[i] * sc); v[nv][1] = (float)(Y[i] * sc); v[nv][2] = (float)(X[i] * sc); wid[nv++] = nw;
        }
        ids[nw++] = atoi(de->d_name + 1); free(X); free(Y); free(Z);
    }
    closedir(dd);
    if (!nv) { fprintf(stderr, "wrapdist: no vertices in %s\n", argv[2]); return 1; }
    double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
    for (size_t i = 0; i < nv; i++) for (int d = 0; d < 3; d++) { if (v[i][d] < lo[d]) lo[d] = v[i][d]; if (v[i][d] > hi[d]) hi[d] = v[i][d]; }
    int64_t bo[3], bn[3];
    for (int d = 0; d < 3; d++) { bo[d] = (int64_t)floor(lo[d]) - M; if (bo[d] < 0) bo[d] = 0; int64_t e = (int64_t)ceil(hi[d]) + M + 1; if (e > zm->shape[d]) e = zm->shape[d]; bn[d] = e - bo[d]; }
    float *dist = malloc(nv * sizeof(float)); for (size_t i = 0; i < nv; i++) dist[i] = INFINITY;
    const size_t plane = (size_t)bn[1] * bn[2];
    uint8_t *lab = malloc((size_t)(C + 2 * M) * plane); float *dsq = malloc((size_t)(C + 2 * M) * plane * sizeof(float));
    if (!lab || !dsq) { fprintf(stderr, "wrapdist: no memory for a chunk of %lld rows of %zu voxels (lower --chunk)\n", (long long)(C + 2 * M), plane); return 1; }
    for (int64_t z0 = bo[0]; z0 < bo[0] + bn[0]; z0 += C) {
        const int64_t a = z0 - M < bo[0] ? bo[0] : z0 - M, b = z0 + C + M > bo[0] + bn[0] ? bo[0] + bn[0] : z0 + C + M;
        int any = 0; for (size_t i = 0; i < nv && !any; i++) any = v[i][0] >= z0 - 0.5 && v[i][0] < z0 + C - 0.5;
        if (!any) continue;
        const int64_t o[3] = {a, bo[1], bo[2]}, n[3] = {b - a, bn[1], bn[2]};
        if (z3_read(z, o, n, lab, nt)) { fprintf(stderr, "wrapdist: %s\n", z3_error()); return 1; }
        if (edt_sq(lab, n, dsq, nt)) { fprintf(stderr, "wrapdist: out of memory\n"); return 1; }
        for (size_t i = 0; i < nv; i++) {
            const int64_t q0 = (int64_t)lround(v[i][0]); if (q0 < z0 || q0 >= z0 + C) continue;
            const int64_t q1 = (int64_t)lround(v[i][1]) - bo[1], q2 = (int64_t)lround(v[i][2]) - bo[2];
            if (q1 < 0 || q2 < 0 || q1 >= bn[1] || q2 >= bn[2]) continue;
            const float s = dsq[((size_t)(q0 - a) * bn[1] + q1) * bn[2] + q2];
            dist[i] = s >= EDT_INF ? (float)(M + 1) : fminf(sqrtf(s), (float)(M + 1));
        }
    }
    free(lab); free(dsq); z3_close(z);
    /* stats in label voxels and um; per wrap optionally */
    for (int w = -1; w < (per ? nw : 0); w++) {   /* all, then each wrap */
        size_t k = 0; float *a = malloc(nv * sizeof(float));
        for (size_t i = 0; i < nv; i++) if (isfinite(dist[i]) && (w < 0 || wid[i] == w)) a[k++] = dist[i];
        if (k) {
            qsort(a, k, sizeof(float), cmp_fl);
            size_t c1 = 0, c2 = 0, c4 = 0, cm = 0; for (size_t i = 0; i < k; i++) { c1 += a[i] * lum <= 9.6; c2 += a[i] * lum <= 19.2; c4 += a[i] * lum > 38.4; cm += a[i] > M; }
            if (w < 0) printf("wrapdist: %zu vertices of %d wraps (%zu outside the labels' extent); distance to the nearest labelled voxel (label voxel %.3g um):\n", k, nw, nv - k, lum);
            char tag[16]; if (w < 0) snprintf(tag, sizeof tag, "all"); else snprintf(tag, sizeof tag, "w%03d", ids[w]);
            printf("  %-5s %8zu vertices: median %.1f um, p75 %.1f, p90 %.1f; <= 9.6 um %.3f, <= 19.2 um %.3f, > 38.4 um %.3f, beyond --max (%.0f um) %.3f",
                   tag, k, a[k / 2] * lum, a[k * 3 / 4] * lum, a[k * 9 / 10] * lum, (double)c1 / k, (double)c2 / k, (double)c4 / k, M * lum, (double)cm / k);
            {   /* --at T1,T2,..: the share within each distance (um) too (a finer label grid resolves smaller ones) */
                const char *at = opt(argc, argv, "--at", ""); char *t = strdup(at), *sv = nullptr;
                for (char *q = strtok_r(t, ",", &sv); q; q = strtok_r(nullptr, ",", &sv)) { const double th = atof(q); size_t c = 0; for (size_t i = 0; i < k; i++) c += a[i] * lum <= th; printf("; <= %g um %.3f", th, (double)c / k); }
                free(t);
            }
            printf("\n");
        }
        free(a);
    }
    free(v); free(wid); free(dist);
    return 0;
}
static float *wd_read(const char *path, int *w, int *h) { return read_tif_f32(path, w, h); }

/* ufsm wrapswitch <wraps-dir> <codes-root> <codes-key> [--range 30] [--gap 1] [--min-run 100] [--chunk 64] [--threads 16] [--per-wrap] [--cache DIR]
   How often the wraps (every <wraps-dir>/wNNN tifxyz) switch sheets. Each vertex takes the code of the nearest labelled voxel along
   its surface normal within +- --range um (band codes: 1 + round(18 q) mod 252, q the published wrap's unwrapped turn coordinate:
   continuous along a sheet, one turn apart between neighbouring sheets). Walking each grid row (around the axis) and each column
   (along z), consecutive coded vertices less than --gap mm apart whose q differ by more than half a turn (mod 14) are a switch to
   another sheet, once the new sheet holds for --min-run um (shorter runs: a vertex between two sheets flickering between their
   labels). Being a whole turn off everywhere costs nothing here: only changes count. Reports switches per cm of coded path
   along theta and z, the coded share of the vertices and the mean switch-free run, over all wraps (and per wrap). The codes are read
   in z chunks of --chunk label rows over the vertices' bounding box. */
typedef struct { int w, h, id; float *X, *Y, *Z; uint8_t *code; float *dist; } ws_wrap;   /* dist: um along the normal to the code */
typedef struct {
    ws_wrap *W; int nw; long *rs; long nrows; double sc, R; const uint8_t *lab; int64_t o[3], n[3], z0, z1; _Atomic long next; double lum;
    const axis *ax; double s0;   /* with --axis: the normals turned away from the axis, dist signed (+: the code lies outward) */
} ws_job;
static void *ws_rows(void *arg) {
    ws_job *J = arg;
    for (long g; (g = atomic_fetch_add(&J->next, 1)) < J->nrows;) {
        int lo = 0, hi = J->nw - 1; while (lo < hi) { const int mid = (lo + hi + 1) / 2; if (J->rs[mid] <= g) lo = mid; else hi = mid - 1; }
        ws_wrap *w = &J->W[lo]; const int r = (int)(g - J->rs[lo]), W = w->w, H = w->h;
        for (int c = 0; c < W; c++) {
            const size_t i = (size_t)r * W + c; if (w->Z[i] < 0 || w->code[i]) continue;
            const double P[3] = {w->Z[i] * J->sc, w->Y[i] * J->sc, w->X[i] * J->sc};
            const int64_t q0 = (int64_t)llround(P[0]); if (q0 < J->z0 || q0 >= J->z1) continue;   /* this chunk's vertices */
            const int cl = c > 0 && w->Z[i - 1] >= 0 ? c - 1 : c, cr = c + 1 < W && w->Z[i + 1] >= 0 ? c + 1 : c;
            const int ru = r > 0 && w->Z[i - W] >= 0 ? r - 1 : r, rd = r + 1 < H && w->Z[i + W] >= 0 ? r + 1 : r;
            if (cl == cr || ru == rd) continue;
            const size_t a = (size_t)r * W + cl, b = (size_t)r * W + cr, u = (size_t)ru * W + c, d = (size_t)rd * W + c;
            const double du[3] = {w->Z[b] - w->Z[a], w->Y[b] - w->Y[a], w->X[b] - w->X[a]}, dv[3] = {w->Z[d] - w->Z[u], w->Y[d] - w->Y[u], w->X[d] - w->X[u]};
            double N[3] = {du[1] * dv[2] - du[2] * dv[1], du[2] * dv[0] - du[0] * dv[2], du[0] * dv[1] - du[1] * dv[0]};
            const double nl = sqrt(N[0] * N[0] + N[1] * N[1] + N[2] * N[2]); if (!(nl > 0)) continue;
            for (int k = 0; k < 3; k++) N[k] /= nl;
            if (J->ax) {   /* outward: away from the axis (the published codes' winding grows outward) */
                double cy, cx; axis_at(J->ax, w->Z[i] * J->s0, &cy, &cx);
                if (N[1] * (w->Y[i] * J->s0 - cy) + N[2] * (w->X[i] * J->s0 - cx) < 0) for (int k = 0; k < 3; k++) N[k] = -N[k];
            }
            for (int s = 0; s <= 4 * (int)ceil(J->R); s++) {   /* t = 0, +.5, -.5, +1, -1, ...: the nearest labelled voxel first */
                const double t = (s + 1) / 2 * 0.5 * (s % 2 ? 1 : -1); if (fabs(t) > J->R) break;
                int64_t p[3]; int in = 1;
                for (int k = 0; k < 3; k++) { p[k] = (int64_t)llround(P[k] + t * N[k]) - J->o[k]; in &= p[k] >= 0 && p[k] < J->n[k]; }
                if (!in) continue;
                const uint8_t v = J->lab[((size_t)p[0] * J->n[1] + p[1]) * J->n[2] + p[2]];
                if (v >= 1 && v <= 252) { w->code[i] = v; w->dist[i] = (float)((J->ax ? t : fabs(t)) * J->lum); break; }
            }
        }
    }
    return nullptr;
}
typedef struct { double path, sw, runs; } ws_stat;
/* one direction over the grid: per line, coded vertices joined into paths (consecutive ones less than gap_um apart); along a path
   the sheet index unwraps the codes (s += round(dq), dq the code change mod 14 turns); runs of one sheet shorter than min_um
   (a vertex between two sheets flickering between their labels) are dropped, and a switch is a change between the remaining runs */
static void ws_walk(const ws_wrap *w, int along_z, double um, double gap_um, double min_um, ws_stat *s, int *sb, double *lb) {
    const int L = along_z ? w->w : w->h, M = along_z ? w->h : w->w;
    for (int l = 0; l < L; l++) {
        int n = 0, last = -1;
        for (int m = 0; m <= M; m++) {
            const size_t i = m < M ? (along_z ? (size_t)m * w->w + l : (size_t)l * w->w + m) : 0;
            const int coded = m < M && w->code[i];
            double dl = 0;
            if (coded && last >= 0) {
                const size_t j = along_z ? (size_t)last * w->w + l : (size_t)l * w->w + last;
                const double dz = w->Z[i] - w->Z[j], dy = w->Y[i] - w->Y[j], dx = w->X[i] - w->X[j]; dl = sqrt(dz * dz + dy * dy + dx * dx) * um;
            }
            if (n > 0 && (!coded ? m == M : dl > gap_um)) {   /* the path ends: its runs */
                double run = 0; int cur = 0, have = 0; double path = 0;
                for (int k = 0; k < n;) {
                    int e = k; double len = 0; while (e < n && sb[e] == sb[k]) { len += lb[e]; e++; }
                    path += len;
                    if (len >= min_um || (k == 0 && e == n)) {
                        if (have && sb[k] != cur) { s->sw += 1; s->runs += 1; run = 0; }
                        cur = sb[k]; have = 1; run += len;
                    } else run += len;
                    k = e;
                }
                s->path += path; if (run > 0) s->runs += 1;
                n = 0;
            }
            if (!coded) continue;
            if (n == 0) { sb[0] = 0; lb[0] = 0; n = 1; }
            else { const size_t j = along_z ? (size_t)last * w->w + l : (size_t)l * w->w + last;
                   double dq = (w->code[i] - w->code[j]) / 18.0; dq = fmod(dq + 7 + 14, 14) - 7;
                   sb[n] = sb[n - 1] + (int)lround(dq); lb[n] = dl; n++; }
            last = m;
        }
    }
}
static int cmd_wrapswitch(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: ufsm wrapswitch <wraps-dir> <codes-root> <codes-key> [--range 30] [--gap 1] [--min-run 100] [--axis FILE [--dir 1]] [--chunk 64] [--threads 16] [--per-wrap] [--cache DIR]\n"); return 2; }
    const double lum = atof(argv[4]), range_um = atof(opt(argc, argv, "--range", "30")), gap_mm = atof(opt(argc, argv, "--gap", "1")), min_um = atof(opt(argc, argv, "--min-run", "100"));
    const int C = atoi(opt(argc, argv, "--chunk", "64")), nt = ufsm_threads(atoi(opt(argc, argv, "--threads", "0")), 16), per = arg_index(argc, argv, "--per-wrap") > 0;
    if (!(lum > 0) || !(range_um > 0) || !(gap_mm > 0) || C < 1 || nt < 1) { fprintf(stderr, "wrapswitch: the codes key is its voxel size in um (e.g. 4.8); --range, --gap, --chunk > 0\n"); return 2; }
    z3 *z = z3_open(store_open(argv[3]), argv[4], opt(argc, argv, "--cache", nullptr)); if (!z) { fprintf(stderr, "wrapswitch: %s\n", z3_error()); return 1; }
    const z3_meta *zm = z3_meta_of(z);
    int nw = 0, capw = 64; ws_wrap *Wr = calloc(capw, sizeof *Wr); double wum = 0;
    DIR *dd = opendir(argv[2]); struct dirent *de; if (!dd) { fprintf(stderr, "wrapswitch: cannot open %s\n", argv[2]); return 1; }
    while ((de = readdir(dd))) {
        if (de->d_name[0] != 'w') continue;
        char p[1400]; int w, h, w2, h2, w3, h3; double um = 0;
        snprintf(p, sizeof p, "%s/%s/meta.json", argv[2], de->d_name);
        { char *b = tfx_meta_read(p); const char *q = b ? strstr(b, "\"voxel_size_um\":") : nullptr; if (q) um = atof(q + 16); free(b); }
        snprintf(p, sizeof p, "%s/%s/x.tif", argv[2], de->d_name); float *X = wd_read(p, &w, &h);
        snprintf(p, sizeof p, "%s/%s/y.tif", argv[2], de->d_name); float *Y = wd_read(p, &w2, &h2);
        snprintf(p, sizeof p, "%s/%s/z.tif", argv[2], de->d_name); float *Z = wd_read(p, &w3, &h3);
        if (!X || !Y || !Z || w != w2 || w != w3 || h != h2 || h != h3 || !(um > 0) || (wum > 0 && fabs(um - wum) > 1e-9)) { free(X); free(Y); free(Z); continue; }
        wum = um;
        if (nw == capw) { capw *= 2; Wr = realloc(Wr, capw * sizeof *Wr); }
        Wr[nw++] = (ws_wrap){w, h, atoi(de->d_name + 1), X, Y, Z, calloc((size_t)w * h, 1), calloc((size_t)w * h, sizeof(float))};
    }
    closedir(dd);
    if (!nw) { fprintf(stderr, "wrapswitch: no wraps in %s\n", argv[2]); return 1; }
    const double sc = wum / lum, R = range_um / lum;
    double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
    for (int k = 0; k < nw; k++) for (size_t i = 0; i < (size_t)Wr[k].w * Wr[k].h; i++) {
        if (Wr[k].Z[i] < 0) continue;
        const double v[3] = {Wr[k].Z[i] * sc, Wr[k].Y[i] * sc, Wr[k].X[i] * sc};
        for (int d = 0; d < 3; d++) { if (v[d] < lo[d]) lo[d] = v[d]; if (v[d] > hi[d]) hi[d] = v[d]; }
    }
    axis ax = {0}; const char *axp = opt(argc, argv, "--axis", nullptr); const int dir = atoi(opt(argc, argv, "--dir", "1"));
    if (axp && axis_load(&ax, axp)) { fprintf(stderr, "wrapswitch: cannot load the axis %s\n", axp); return 1; }
    const double s0 = wum / 2.4;   /* wrap voxels -> level-0 (axis) voxels */
    const int Ri = (int)ceil(R) + 1; int64_t bo[3], bn[3];
    for (int d = 0; d < 3; d++) { bo[d] = (int64_t)floor(lo[d]) - Ri; if (bo[d] < 0) bo[d] = 0; int64_t e = (int64_t)ceil(hi[d]) + Ri + 1; if (e > zm->shape[d]) e = zm->shape[d]; bn[d] = e - bo[d]; }
    long *rs = malloc((nw + 1) * sizeof *rs); rs[0] = 0; for (int k = 0; k < nw; k++) rs[k + 1] = rs[k] + Wr[k].h;
    const size_t plane = (size_t)bn[1] * bn[2]; uint8_t *lab = malloc((size_t)(C + 2 * Ri) * plane);
    if (!lab) { fprintf(stderr, "wrapswitch: no memory for %d rows of %zu voxels (lower --chunk)\n", C + 2 * Ri, plane); return 1; }
    for (int64_t z0 = bo[0]; z0 < bo[0] + bn[0]; z0 += C) {
        const int64_t a = z0 - Ri < bo[0] ? bo[0] : z0 - Ri, b = z0 + C + Ri > bo[0] + bn[0] ? bo[0] + bn[0] : z0 + C + Ri;
        ws_job J = {Wr, nw, rs, rs[nw], sc, R, lab, {a, bo[1], bo[2]}, {b - a, bn[1], bn[2]}, z0, z0 + C, 0, lum, axp ? &ax : nullptr, s0};
        if (z3_read(z, J.o, J.n, lab, nt)) { fprintf(stderr, "wrapswitch: %s\n", z3_error()); return 1; }
        pthread_t th[64]; const int k = nt < 64 ? nt : 64; for (int t = 0; t < k; t++) pthread_create(&th[t], nullptr, ws_rows, &J);
        for (int t = 0; t < k; t++) pthread_join(th[t], nullptr);
    }
    free(lab); z3_close(z);
    /* --axis FILE [--dir 1]: is each coded vertex on its wrap's sheet? A wrap k is U = k + dir theta / 2 pi; a code's q is the published
       sheet's U (+ one constant), so e = q - dir theta / 2 pi - k is the same for every vertex of every wrap traced on the right
       sheet (up to one global gauge), and off by whole turns elsewhere. Gauge: the circular mean of e's fraction, then the most
       common whole turn; right = |e - gauge| < 0.5 turn (mod 14). Over all wraps with one gauge, and per wrap with its own (on its
       majority sheet: coherent though maybe a turn off) */
    double gre = 0, gim = 0; size_t ghist[14] = {0}, nright = 0, nown = 0, ncod = 0;
    float *dall = malloc(sizeof(float) * 1024); size_t nd = 0, capd = 1024; size_t d10 = 0, d20 = 0, sdpos = 0, sdneg = 0; double sdsum = 0;
    if (axp) for (int k = 0; k < nw; k++) for (size_t i = 0; i < (size_t)Wr[k].w * Wr[k].h; i++) {   /* the global fraction */
        if (!Wr[k].code[i]) continue;
        double cy, cx; axis_at(&ax, Wr[k].Z[i] * s0, &cy, &cx);
        const double e = (Wr[k].code[i] - 1) / 18.0 - dir * atan2(Wr[k].Y[i] * s0 - cy, Wr[k].X[i] * s0 - cx) / (2 * M_PI) - Wr[k].id;
        gre += cos(2 * M_PI * e); gim += sin(2 * M_PI * e);
    }
    const double gfr = atan2(gim, gre) / (2 * M_PI);
    if (axp) for (int k = 0; k < nw; k++) for (size_t i = 0; i < (size_t)Wr[k].w * Wr[k].h; i++) {
        if (!Wr[k].code[i]) continue;
        double cy, cx; axis_at(&ax, Wr[k].Z[i] * s0, &cy, &cx);
        const double e = (Wr[k].code[i] - 1) / 18.0 - dir * atan2(Wr[k].Y[i] * s0 - cy, Wr[k].X[i] * s0 - cx) / (2 * M_PI) - Wr[k].id;
        ghist[(((int)lround(e - gfr)) % 14 + 14) % 14]++;
    }
    int gm = 0; for (int b = 1; b < 14; b++) if (ghist[b] > ghist[gm]) gm = b;
    ws_stat A[2] = {{0}}; size_t coded = 0, valid = 0;
    if (per) printf("  wrap   coded   theta: sw/cm  run mm   z: sw/cm  run mm%s\n", axp ? "   right  own-sheet" : "");
    for (int k = 0; k < nw; k++) {
        ws_stat s[2] = {{0}}; size_t cd = 0, vl = 0;
        size_t wr = 0, wo = 0, wc = 0, h14[14] = {0};
        for (size_t i = 0; i < (size_t)Wr[k].w * Wr[k].h; i++) {
            vl += Wr[k].Z[i] >= 0; cd += Wr[k].code[i] > 0; if (!Wr[k].code[i]) continue;
            if (nd == capd) { capd *= 2; dall = realloc(dall, capd * sizeof *dall); }
            const float ad = fabsf(Wr[k].dist[i]); sdsum += Wr[k].dist[i]; sdpos += Wr[k].dist[i] > 0; sdneg += Wr[k].dist[i] < 0;
            dall[nd++] = ad; d10 += ad <= 10; d20 += ad <= 20;
            if (!axp) continue;
            double cy, cx; axis_at(&ax, Wr[k].Z[i] * s0, &cy, &cx);
            const double e = (Wr[k].code[i] - 1) / 18.0 - dir * atan2(Wr[k].Y[i] * s0 - cy, Wr[k].X[i] * s0 - cx) / (2 * M_PI) - Wr[k].id;
            const int b = (((int)lround(e - gfr)) % 14 + 14) % 14; h14[b]++; wc++;
            double dg = fmod(e - gfr - gm + 7 + 1400, 14) - 7; wr += fabs(dg) < 0.5;
        }
        if (axp && wc) { int m = 0; for (int b = 1; b < 14; b++) if (h14[b] > h14[m]) m = b; wo = h14[m]; nright += wr; nown += wo; ncod += wc; }
        const int mx = Wr[k].w > Wr[k].h ? Wr[k].w : Wr[k].h; int *sb = malloc((mx + 1) * sizeof *sb); double *lb = malloc((mx + 1) * sizeof *lb);
        for (int d = 0; d < 2; d++) { ws_walk(&Wr[k], d, wum, gap_mm * 1000, min_um, &s[d], sb, lb); A[d].path += s[d].path; A[d].sw += s[d].sw; A[d].runs += s[d].runs; }
        free(sb); free(lb); coded += cd; valid += vl;
        if (per && s[0].path > 0) printf("  w%03d  %5.3f   %10.2f  %6.2f   %8.2f  %6.2f   %5.3f  %5.3f\n", Wr[k].id, vl ? (double)cd / vl : 0, s[0].sw / (s[0].path * 1e-4),
                                          s[0].runs ? s[0].path / s[0].runs * 1e-3 : 0, s[1].path > 0 ? s[1].sw / (s[1].path * 1e-4) : 0, s[1].runs ? s[1].path / s[1].runs * 1e-3 : 0,
                                          wc ? (double)wr / wc : 0, wc ? (double)wo / wc : 0);
        free(Wr[k].dist);
        free(Wr[k].X); free(Wr[k].Y); free(Wr[k].Z); free(Wr[k].code);
    }
    printf("wrapswitch: %d wraps, %zu vertices, %.3f coded (nearest label within %.0f um along the normal); %.1f cm of coded path around "
           "the axis, %.0f switches: %.2f per cm, mean switch-free run %.2f mm; along z %.1f cm, %.0f switches: %.2f per cm, run %.2f mm\n",
           nw, valid, valid ? (double)coded / valid : 0, range_um, A[0].path * 1e-4, A[0].sw, A[0].path > 0 ? A[0].sw / (A[0].path * 1e-4) : 0,
           A[0].runs ? A[0].path / A[0].runs * 1e-3 : 0, A[1].path * 1e-4, A[1].sw, A[1].path > 0 ? A[1].sw / (A[1].path * 1e-4) : 0, A[1].runs ? A[1].path / A[1].runs * 1e-3 : 0);
    if (nd) { qsort(dall, nd, sizeof(float), cmp_fl);
        printf("wrapswitch: distance along the normal to the nearest published sheet (coded vertices): median %.1f um, p90 %.1f; of all vertices <= 10 um %.3f, <= 20 um %.3f\n",
               dall[nd / 2], dall[nd * 9 / 10], valid ? (double)d10 / valid : 0, valid ? (double)d20 / valid : 0);
        if (axp) printf("wrapswitch: signed (+: the published sheet lies outward of the vertex, away from the axis): mean %+.2f um, outward %.3f, inward %.3f, at 0 %.3f\n",
                        sdsum / nd, (double)sdpos / nd, (double)sdneg / nd, (double)(nd - sdpos - sdneg) / nd); }
    if (axp && ncod) printf("wrapswitch: on the right sheet (one global gauge %.3f: q = k + dir theta / 2 pi + gauge, mod 14) %.3f of coded vertices; on their wrap's majority sheet %.3f\n", gfr + gm, (double)nright / ncod, (double)nown / ncod);
    free(dall); free(Wr); free(rs);
    return 0;
}

/* ufsm surf-snap <wraps-in> <wraps-out> --volume ROOT KEY --vol-um U [--range 40] [--min 100] [--median 2] [--measure T] [--threads 16] [--cache DIR]
   Refines tifxyz surfaces (every <wraps-in>/wNNN) onto the papyrus. A fitted winding's level set W = k is smooth at the fit's lattice
   scale (77 um at 19.2 um) and crosses the gaps of wavy sheets: on a 3.9 cm Paris4 slab ~half its 2.4 um render samples were void,
   for a fit to the ground-truth phase too. Each vertex moves along its surface normal (from the grid) to the nearest local maximum
   >= --min of a volume (CT, or a recto probability) within +- --range um, sampled trilinearly every half volume voxel; the
   displacements are median-filtered over (2 median + 1)^2 vertices (a stray peak cannot tear the surface; a vertex without a
   maximum takes its neighbours' median). Vertices are in the wraps' voxel units (meta.json voxel_size_um), the volume at --vol-um.
   Tiles of vertices read only their own box. --measure T: only report the volume at the vertices (mean, share below T; e.g. CT
   < 90: void), writing nothing (<wraps-out> unused). */
typedef struct { const float *X, *Y, *Z, *N; float *disp; int w, r0, r1, c0, c1; const uint8_t *vol; int64_t o[3], n[3]; double s, R, vmin; _Atomic int next; int measure; } ssn_job;
static void *ssn_rows(void *arg) {
    ssn_job *J = arg;
    for (int r; (r = J->r0 + atomic_fetch_add(&J->next, 1)) < J->r1;) for (int c = J->c0; c < J->c1; c++) {
        const size_t i = (size_t)r * J->w + c; if (J->Z[i] < 0) continue;
        const double P[3] = {J->Z[i] * J->s - J->o[0], J->Y[i] * J->s - J->o[1], J->X[i] * J->s - J->o[2]};
        if (J->measure) { J->disp[i] = trilin(J->vol, J->n, P[0], P[1], P[2]); continue; }   /* --measure: the value at the vertex */
        if (isnan(J->N[3 * i])) continue;
        const float *N = &J->N[3 * i]; const double st = 0.5;
#define SSV(t_) trilin(J->vol, J->n, P[0] + (t_) * N[0], P[1] + (t_) * N[1], P[2] + (t_) * N[2])
        double bd = 1e300, bt = 0; float vm = SSV(-J->R - st), v0 = SSV(-J->R);
        for (double t = -J->R; t <= J->R; t += st) {
            const float vp = SSV(t + st);
            if (v0 >= vm && v0 > vp && v0 >= J->vmin && fabs(t) < bd) { bd = fabs(t); bt = t; }
            vm = v0; v0 = vp;
        }
#undef SSV
        if (bd < 1e300) J->disp[i] = (float)(bt / J->s);   /* in the wraps' voxels */
    }
    return nullptr;
}
static int cmd_surfsnap(int argc, char **argv) {
    const int vi = arg_index(argc, argv, "--volume");
    if (argc < 4 || vi < 0 || vi + 2 >= argc) { fprintf(stderr, "usage: ufsm surf-snap <wraps-in> <wraps-out> --volume ROOT KEY --vol-um U [--range 40] [--min 100] [--median 2] [--threads 16] [--cache DIR]\n"); return 2; }
    const double vum = atof(opt(argc, argv, "--vol-um", "0")), range_um = atof(opt(argc, argv, "--range", "40")), vmin = atof(opt(argc, argv, "--min", "100"));
    const int nt = ufsm_threads(atoi(opt(argc, argv, "--threads", "0")), 16), MR = atoi(opt(argc, argv, "--median", "2"));
    if (!(vum > 0) || !(range_um > 0) || MR < 0 || MR > 5 || nt < 1) { fprintf(stderr, "surf-snap: --vol-um U (the volume's voxel um) > 0, --range > 0, --median 0..5\n"); return 2; }
    z3 *vz = z3_open(store_open(argv[vi + 1]), argv[vi + 2], opt(argc, argv, "--cache", nullptr)); if (!vz) { fprintf(stderr, "surf-snap: %s\n", z3_error()); return 1; }
    const z3_meta *zm = z3_meta_of(vz);
    char cmd[1400]; snprintf(cmd, sizeof cmd, "mkdir -p '%s'", argv[3]); if (atof(opt(argc, argv, "--measure", "0")) <= 0 && system(cmd)) return 1;
    DIR *dd = opendir(argv[2]); if (!dd) { fprintf(stderr, "surf-snap: cannot open %s\n", argv[2]); return 1; }
    struct dirent *de; int nw = 0; size_t nmoved = 0, nvert = 0, mlow = 0, mn = 0; double sabs = 0, msum = 0;
    const double measure = atof(opt(argc, argv, "--measure", "0"));   /* > 0: report the share of vertices where the volume is below it */
    while ((de = readdir(dd))) {
        if (de->d_name[0] != 'w') continue;
        char p[1400]; int w, h, w2, h2, w3, h3; double wum = 0, scale = 0; int winding = 0;
        snprintf(p, sizeof p, "%s/%s/meta.json", argv[2], de->d_name);
        { char *bm = tfx_meta_read(p); const char *b = bm ? bm : "";
          const char *q = strstr(b, "\"voxel_size_um\":"); if (q) wum = atof(q + 16);
          q = strstr(b, "\"scale\":"); if (q) { q = strchr(q, '['); if (q) scale = atof(q + 1); }
          q = strstr(b, "\"winding\":"); if (q) winding = atoi(q + 10); free(bm); }
        snprintf(p, sizeof p, "%s/%s/x.tif", argv[2], de->d_name); float *X = read_tif_f32(p, &w, &h);
        snprintf(p, sizeof p, "%s/%s/y.tif", argv[2], de->d_name); float *Y = read_tif_f32(p, &w2, &h2);
        snprintf(p, sizeof p, "%s/%s/z.tif", argv[2], de->d_name); float *Z = read_tif_f32(p, &w3, &h3);
        if (!X || !Y || !Z || w != w2 || w != w3 || h != h2 || h != h3 || !(wum > 0) || !(scale > 0) || w < 3 || h < 3) { free(X); free(Y); free(Z); continue; }
        const size_t nv = (size_t)w * h; const double s = wum / vum, R = range_um / vum;
        float *N = malloc(nv * 3 * sizeof(float)), *disp = malloc(nv * 4), *med = malloc(nv * 4);
        for (int r = 0; r < h; r++) for (int c = 0; c < w; c++) {   /* unit normals (central differences where the neighbours exist) */
            const size_t i = (size_t)r * w + c; float *n3 = &N[3 * i]; n3[0] = n3[1] = n3[2] = NAN; disp[i] = NAN; if (Z[i] < 0) continue;
            const int cl = c > 0 && Z[i - 1] >= 0 ? c - 1 : c, cr = c + 1 < w && Z[i + 1] >= 0 ? c + 1 : c;
            const int ru = r > 0 && Z[i - w] >= 0 ? r - 1 : r, rd = r + 1 < h && Z[i + w] >= 0 ? r + 1 : r;
            if (cl == cr || ru == rd) continue;
            const size_t a = (size_t)r * w + cl, b = (size_t)r * w + cr, u = (size_t)ru * w + c, d = (size_t)rd * w + c;
            const double du[3] = {Z[b] - Z[a], Y[b] - Y[a], X[b] - X[a]}, dv[3] = {Z[d] - Z[u], Y[d] - Y[u], X[d] - X[u]};
            const double q[3] = {du[1] * dv[2] - du[2] * dv[1], du[2] * dv[0] - du[0] * dv[2], du[0] * dv[1] - du[1] * dv[0]}, ql = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]);
            if (ql > 0) for (int k = 0; k < 3; k++) n3[k] = (float)(q[k] / ql);
        }
        const int TR = 128, TC = 512;
        for (int r0 = 0; r0 < h; r0 += TR) for (int c0 = 0; c0 < w; c0 += TC) {
            const int r1 = r0 + TR < h ? r0 + TR : h, c1 = c0 + TC < w ? c0 + TC : w;
            double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
            for (int r = r0; r < r1; r++) for (int c = c0; c < c1; c++) {
                const size_t i = (size_t)r * w + c; if (Z[i] < 0) continue;
                const double v[3] = {Z[i] * s, Y[i] * s, X[i] * s}; for (int d = 0; d < 3; d++) { if (v[d] < lo[d]) lo[d] = v[d]; if (v[d] > hi[d]) hi[d] = v[d]; }
            }
            if (lo[0] > hi[0]) continue;
            ssn_job J = {X, Y, Z, N, disp, w, r0, r1, c0, c1, nullptr, {0, 0, 0}, {0, 0, 0}, s, R, vmin, 0, measure > 0}; int ok = 1;
            for (int d = 0; d < 3; d++) {
                int64_t a = (int64_t)floor(lo[d] - R - 2), b = (int64_t)ceil(hi[d] + R + 2) + 1;
                if (a < 0) a = 0;
                if (b > zm->shape[d]) b = zm->shape[d];
                if (b <= a) ok = 0;
                J.o[d] = a; J.n[d] = b - a;
            }
            const size_t bytes = (size_t)J.n[0] * J.n[1] * J.n[2];
            if (!ok || bytes > ((size_t)4 << 30)) { if (ok) fprintf(stderr, "surf-snap: %s tile %d,%d spans %lldx%lldx%lld voxels (a fold?): skipped\n", de->d_name, r0, c0, (long long)J.n[0], (long long)J.n[1], (long long)J.n[2]); continue; }
            uint8_t *vol = malloc(bytes);
            if (!vol || z3_read(vz, J.o, J.n, vol, nt)) { free(vol); fprintf(stderr, "surf-snap: %s\n", z3_error()); continue; }
            J.vol = vol;
            pthread_t th[64]; const int k = nt < 64 ? nt : 64;
            for (int t = 0; t < k; t++) pthread_create(&th[t], nullptr, ssn_rows, &J);
            for (int t = 0; t < k; t++) pthread_join(th[t], nullptr);
            free(vol);
        }
        if (measure > 0) {   /* --measure T: the volume at the vertices, no output */
            for (size_t i = 0; i < nv; i++) if (Z[i] >= 0 && !isnan(disp[i])) { msum += disp[i]; mlow += disp[i] < measure; mn++; }
            nw++; free(X); free(Y); free(Z); free(N); free(disp); free(med); continue;
        }
        float win[121]; size_t moved = 0, valid = 0;
        for (int r = 0; r < h; r++) for (int c = 0; c < w; c++) {
            const size_t i = (size_t)r * w + c; med[i] = NAN; if (Z[i] < 0) continue;
            valid++; int k = 0;
            for (int a = r - MR; a <= r + MR; a++) for (int b = c - MR; b <= c + MR; b++)
                if (a >= 0 && a < h && b >= 0 && b < w && !isnan(disp[(size_t)a * w + b])) win[k++] = disp[(size_t)a * w + b];
            if (k >= (2 * MR + 1) * (2 * MR + 1) / 5 + 1) { qsort(win, k, sizeof(float), cmp_fl); med[i] = win[k / 2]; }
        }
        for (size_t i = 0; i < nv; i++) {
            if (Z[i] < 0 || isnan(med[i]) || isnan(N[3 * i])) continue;
            Z[i] += med[i] * N[3 * i]; Y[i] += med[i] * N[3 * i + 1]; X[i] += med[i] * N[3 * i + 2]; moved++; sabs += fabs(med[i]) * wum;
        }
        snprintf(p, sizeof p, "%s/%s", argv[3], de->d_name);
        if (tifxyz_write(p, w, h, X, Y, Z, 1 / scale, wum, winding, "ufsm surf-snap (wraps refined onto the papyrus)")) { fprintf(stderr, "surf-snap: cannot write %s\n", p); return 1; }
        nw++; nmoved += moved; nvert += valid;
        free(X); free(Y); free(Z); free(N); free(disp); free(med);
    }
    closedir(dd); z3_close(vz);
    if (measure > 0) { printf("surf-snap --measure: %d wraps, %zu vertices: mean %.1f, below %g %.3f\n", nw, mn, mn ? msum / mn : 0, measure, mn ? (double)mlow / mn : 0); return nw ? 0 : 1; }
    fprintf(stderr, "surf-snap: %d wraps, %zu of %zu vertices moved (mean |shift| %.1f um) -> %s\n", nw, nmoved, nvert, nmoved ? sabs / nmoved : 0, argv[3]);
    return nw ? 0 : 1;
}

/* ufsm field-blend OUT IN1 IN2 ... [--weights a,b,...] [--tile SZ,SY,SX --grids z,y,x:z,y,x:...] [--conf resultant|mean] [--threads N]:
   fit-field sets of several models on one grid blended into OUT (src/fblend.h: the phase as the confidence-weighted complex mean,
   its length the new confidence). --tile: the inputs are overlapping predicts of one model on tile interiors of SZ x SY x SX level
   voxels (predict --window - 2 --halo; z: --window-z - 2 --halo-z) anchored at each input's --grid-origin (--grids, in input order):
   per-cell weights sin^2 of the cell's place in its interior (fb_tile_w). --conf mean: the mean input confidence as the length. */
static int cmd_field_blend(int argc, char **argv) {
    const char *in[32]; float w[32]; int n = 0; const char *ws = nullptr, *out = nullptr, *tile = nullptr, *grids = nullptr, *conf = "resultant"; int nt = ufsm_threads(0, 8);
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--weights") && i + 1 < argc) { ws = argv[++i]; continue; }
        if (!strcmp(argv[i], "--threads") && i + 1 < argc) { nt = atoi(argv[++i]); continue; }
        if (!strcmp(argv[i], "--tile") && i + 1 < argc) { tile = argv[++i]; continue; }
        if (!strcmp(argv[i], "--grids") && i + 1 < argc) { grids = argv[++i]; continue; }
        if (!strcmp(argv[i], "--conf") && i + 1 < argc) { conf = argv[++i]; continue; }
        if (!strncmp(argv[i], "--", 2)) { fprintf(stderr, "field-blend: unknown option %s\n", argv[i]); return 2; }
        if (!out) out = argv[i]; else if (n < 32) in[n++] = argv[i]; else { fprintf(stderr, "field-blend: at most 32 inputs\n"); return 2; }
    }
    if (!out || n < 1) { fprintf(stderr, "usage: ufsm field-blend <out> <fields1> [<fields2> ...] [--weights a,b,...] [--tile SZ,SY,SX --grids z,y,x:...] [--conf resultant|mean] [--threads N]\n"); return 2; }
    if (ws) {
        int k = 0; char *t = strdup(ws);
        for (char *q = strtok(t, ","); q; q = strtok(nullptr, ",")) { if (k < 32) w[k] = (float)atof(q); k++; }
        free(t);
        if (k != n) { fprintf(stderr, "field-blend: %d weights for %d inputs\n", k, n); return 2; }
    }
    static fb_opts o; memset(&o, 0, sizeof o);
    if (!strcmp(conf, "mean")) o.conf = FB_CONF_MEAN; else if (strcmp(conf, "resultant")) { fprintf(stderr, "field-blend: --conf resultant|mean\n"); return 2; }
    if (!tile != !grids) { fprintf(stderr, "field-blend: --tile and --grids go together\n"); return 2; }
    if (tile) {
        if (sscanf(tile, "%lf,%lf,%lf", &o.stride[0], &o.stride[1], &o.stride[2]) != 3 || !(o.stride[0] > 0 && o.stride[1] > 0 && o.stride[2] > 0)) { fprintf(stderr, "field-blend: --tile SZ,SY,SX (> 0)\n"); return 2; }
        int k = 0; char *t = strdup(grids);
        for (char *q = strtok(t, ":"); q; q = strtok(nullptr, ":")) { if (k < 32 && sscanf(q, "%lf,%lf,%lf", &o.grid[k][0], &o.grid[k][1], &o.grid[k][2]) != 3) { free(t); fprintf(stderr, "field-blend: --grids z,y,x:z,y,x:...\n"); return 2; } k++; }
        free(t);
        if (k != n) { fprintf(stderr, "field-blend: %d grids for %d inputs\n", k, n); return 2; }
        o.tile = 1;
    }
    return fblend_run2(out, n, in, ws ? w : nullptr, &o, nt < 1 ? 1 : nt);
}
/* ufsm fit-compare <reference-fields> <fields>: how a field set (e.g. a network's, predict --fit-zarr) agrees with a reference
   (e.g. the oracle, band --fit-fields) cell by cell; both must cover the same physical cells (origin x um, h x um equal), at any
   native voxel sizes. Over cells where the reference phase confidence is > 0.3 and both have material: phase |error| (turns),
   phase confidence, sheet spacing ratio (um / um), normal angle (principal eigenvectors of n n^T), and the share of gross
   phase errors (> 1/4 turn) by reference spacing and by the fields' confidence. Histograms, no per-cell arrays. */
static void fc_axis(const fit_fields *f, size_t i, double v[3]) {   /* principal eigenvector of the cell's n n^T (power iteration) */
    double T[3][3]; const double t[6] = {f->t[0][i] / 255.0, f->t[1][i] / 255.0, f->t[2][i] / 255.0, f->t[3][i] / 255.0 - 0.5, f->t[4][i] / 255.0 - 0.5, f->t[5][i] / 255.0 - 0.5};
    T[0][0] = t[0]; T[1][1] = t[1]; T[2][2] = t[2]; T[0][1] = T[1][0] = t[3]; T[0][2] = T[2][0] = t[4]; T[1][2] = T[2][1] = t[5];
    int k = T[0][0] >= T[1][1] && T[0][0] >= T[2][2] ? 0 : T[1][1] >= T[2][2] ? 1 : 2; v[0] = v[1] = v[2] = 0; v[k] = 1;   /* start: the largest diagonal axis */
    for (int it = 0; it < 24; it++) {
        double w[3] = {0, 0, 0}; for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) w[a] += T[a][b] * v[b];
        const double l = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]); if (l < 1e-12) return; for (int a = 0; a < 3; a++) v[a] = w[a] / l;
    }
}
static double hist_q(const size_t *h, int nb, double lo, double w, double q) {   /* quantile of a histogram (bin centres) */
    size_t n = 0; for (int k = 0; k < nb; k++) n += h[k];
    size_t c = 0; for (int k = 0; k < nb; k++) { c += h[k]; if (n && (double)c >= q * n) return lo + (k + 0.5) * w; }
    return lo + nb * w;
}
static int fc_surf_counts(int argc, char **argv, const fit_fields_buf *A);
/* ufsm l1-bench FILE [--iters 256]: the slip repair's L1 solve on a dumped vote set (fit with UFSM_L1_DUMP=FILE): time, moves, energy
   and a hash of the shifts (bit-identity checks between solver versions) */
static int cmd_l1_bench(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: ufsm l1-bench FILE [--iters 256]\n"); return 2; }
    FILE *f = fopen(argv[2], "rb"); uint64_t n = 0, nv = 0;
    if (!f || fread(&n, 8, 1, f) != 1 || fread(&nv, 8, 1, f) != 1) { fprintf(stderr, "l1-bench: cannot read %s\n", argv[2]); return 1; }
    l1_vote *v = malloc((nv ? nv : 1) * sizeof *v); int32_t *s = calloc(n ? n : 1, sizeof *s);
    if (!v || !s || fread(v, sizeof *v, nv, f) != nv) { fprintf(stderr, "l1-bench: short read\n"); return 1; }
    fclose(f);
    {   /* the votes' offsets (a huge offset makes the +-1 moves crawl) */
        size_t h[5] = {0}; int64_t dmax = 0; size_t imax = 0;
        for (uint64_t k = 0; k < nv; k++) { const int64_t a = v[k].d < 0 ? -(int64_t)v[k].d : v[k].d; h[a == 0 ? 0 : a == 1 ? 1 : a <= 4 ? 2 : a <= 64 ? 3 : 4]++; if (a > dmax) { dmax = a; imax = k; } }
        printf("l1-bench: vote offsets 0: %zu, 1: %zu, 2-4: %zu, 5-64: %zu, > 64: %zu; max %lld (nodes %u %u, weight %lld)\n", h[0], h[1], h[2], h[3], h[4],
               (long long)dmax, nv ? v[imax].a : 0, nv ? v[imax].b : 0, nv ? (long long)v[imax].w : 0);
    }
    l1_stats st = {0};
    if (l1_solve((uint32_t)n, v, nv, s, atoi(opt(argc, argv, "--iters", "256")), &st)) { fprintf(stderr, "l1-bench: solve failed\n"); return 1; }
    uint64_t hsh = 1469598103934665603ull; size_t nz = 0;
    for (uint64_t k = 0; k < n; k++) { hsh = (hsh ^ (uint32_t)s[k]) * 1099511628211ull; nz += s[k] != 0; }
    printf("l1-bench: %llu nodes, %llu votes, %zu edges: E %lld -> %lld in %d moves (%d max flows), %zu nodes shifted, %.2f s, work %llu%s, hash %016llx\n",
           (unsigned long long)n, (unsigned long long)nv, st.nedge, (long long)st.e0, (long long)st.e1, st.moves, st.iters, nz, st.secs, (unsigned long long)st.work, st.capped ? " (budget reached)" : "", (unsigned long long)hsh);
    free(v); free(s);
    return 0;
}
/* fit-compare on threads (contiguous z ranges): integer histograms / counts summed; the floating sums (densities by spacing, the
   segments' turn errors) added afterwards in the serial loop's order, so the output is the serial one's bit for bit */
enum { FC_NP = 500, FC_NC = 100, FC_NR = 200, FC_NA = 900, FC_NS = 5, FC_NK = 4, FC_NE = 400 };
typedef struct {
    size_t hp[FC_NP], hca[FC_NC], hcb[FC_NC], hr[FC_NR], ha[FC_NA], sn[FC_NS], sg[FC_NS], kn[FC_NK], kg[FC_NK], m, lt01, gt25, hrs[FC_NS][FC_NR];
    size_t hep[FC_NE], hed[FC_NE], nseg, slp, sld, pk_n, pk_sp, pk_sd; double *seg; size_t nsg, csg;   /* seg: ep, ed, ref per segment */
} fc_part;
typedef struct { const fit_fields_buf *A, *B; int k; uint8_t *cls; fc_part *P; } fc_job;
static const double fc_sb[FC_NS + 1] = {0, 100, 140, 180, 240, 1e30}, fc_cb[FC_NK + 1] = {0, 0.2, 0.4, 0.6, 1.01};
static void fc_cells(void *a, int z0, int z1, int t) {
    const fc_job *J = a; const fit_fields_buf *A = J->A, *B = J->B; const int k = J->k; fc_part *P = &J->P[t];
    const size_t HW = (size_t)A->shape[1] * A->shape[2];
    for (size_t i = (size_t)z0 * HW; i < (size_t)z1 * HW; i++) {
        const size_t az = i / HW, ay = i / A->shape[2] % A->shape[1], ax = i % A->shape[2];
        const size_t j = ((az / k) * B->shape[1] + ay / k) * B->shape[2] + ax / k;   /* the candidate cell */
        if (!A->f.w[i] || !B->f.w[j]) continue;
        const double ac = (A->f.pc[i] - 128) / 127.0, as = (A->f.ps[i] - 128) / 127.0, bc = (B->f.pc[j] - 128) / 127.0, bs = (B->f.ps[j] - 128) / 127.0;
        const double ca = hypot(ac, as), cbv = hypot(bc, bs); if (ca <= 0.3) continue;
        P->m++;
        double e = (atan2(bs, bc) - atan2(as, ac)) / (2 * M_PI); e -= round(e); e = fabs(e);
        P->hp[(int)fmin(FC_NP - 1, e / 0.5 * FC_NP)]++; P->lt01 += e < 0.1; P->gt25 += e > 0.25;
        P->hca[(int)fmin(FC_NC - 1, ca * FC_NC)]++; P->hcb[(int)fmin(FC_NC - 1, cbv * FC_NC)]++;
        const double spa = A->f.sp[i] * A->h / 8.0 * A->um, spb = B->f.sp[j] * B->h / 8.0 * B->um;   /* spacing in um (sp = 8 / (h rho)) */
        if (spa > 0 && spb > 0) P->hr[(int)fmin(FC_NR - 1, fmax(0, (log2(spb / spa) + 2) / 4 * FC_NR))]++;   /* log2 ratio in [-2, 2) */
        if (spa > 0 && spb > 0) for (int q = 0; q < FC_NS; q++) if (spa >= fc_sb[q] && spa < fc_sb[q + 1]) { P->hrs[q][(int)fmin(FC_NR - 1, fmax(0, (log2(spb / spa) + 2) / 4 * FC_NR))]++; J->cls[i] = (uint8_t)(q + 1); }
        for (int q = 0; q < FC_NS; q++) if (spa >= fc_sb[q] && spa < fc_sb[q + 1]) { P->sn[q]++; P->sg[q] += e > 0.25; }
        for (int q = 0; q < FC_NK; q++) if (cbv >= fc_cb[q] && cbv < fc_cb[q + 1]) { P->kn[q]++; P->kg[q] += e > 0.25; }
        double va[3], vb[3]; fc_axis(&A->f, i, va); fc_axis(&B->f, j, vb);
        const double ang = acos(fmin(1, fabs(va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2]))) * 180 / M_PI;
        P->ha[(int)fmin(FC_NA - 1, ang / 90 * FC_NA)]++;
    }
}
/* the turn count without the fitter: 0.5 mm segments along the radial direction (away from the axis, in z planes) from every 4th
   reference cell; the reference crosses n turns (its phase unwrapped along the segment, every step known with confidence > 0.3); the
   candidate's count from its phase unwrapped the same way, and from its density integrated along the segment (|n . r| / spacing, n
   the principal axis of its n n^T): a field too coarse for its phase to resolve the sheets can still count them by density. Error =
   candidate - reference turns (negative: sheets missed). Threads over the segment z planes (z = 4 a). */
static void fc_segs(void *a, int a0, int a1, int t) {
    const fc_job *J = a; const fit_fields_buf *A = J->A, *B = J->B; fc_part *P = &J->P[t];
    const double L = 500.0, ds = 0.5 * A->h * A->um;   /* segment length and step (um) */
    const int ns = (int)(L / ds);
    const double cu = A->h * A->um, ccu = B->h * B->um;   /* reference / candidate cell edge (um) */
    for (int zi = a0; zi < a1; zi++) for (int y = 0; y < A->shape[1]; y += 4) for (int x = 0; x < A->shape[2]; x += 4) {
        const int z = 4 * zi;
        const double nz = A->h * (z + 0.5);   /* native z of the cell centre (reference's native voxels, box-local) */
        const int az = (int)fmin(fmax(nz, 0), A->shape[0] * A->h); const double cyv = A->f.cy[az], cxv = A->f.cx[az];
        double py = A->h * (y + 0.5), px = A->h * (x + 0.5);
        double ry = py - cyv, rx = px - cxv; const double rl = hypot(ry, rx); if (rl < 1) continue; ry /= rl; rx /= rl;
        double pa = 0, pb = 0, ta = 0, tb = 0, td = 0, la = 0, lb = 0, spsum = 0; int ok = 1, hb = 0;
        for (int k = 0; k <= ns && ok; k++) {
            const double qy = py + ry * k * ds / A->um, qx = px + rx * k * ds / A->um;   /* native voxels */
            const int iy = (int)(qy / A->h), ix = (int)(qx / A->h);
            if (iy < 0 || ix < 0 || iy >= A->shape[1] || ix >= A->shape[2]) { ok = 0; break; }
            const size_t i = ((size_t)z * A->shape[1] + iy) * A->shape[2] + ix;
            const double ac = (A->f.pc[i] - 128) / 127.0, as = (A->f.ps[i] - 128) / 127.0;
            if (!A->f.w[i] || hypot(ac, as) <= 0.3) { ok = 0; break; }
            const size_t j = (((size_t)(nz * A->um / ccu)) * B->shape[1] + (size_t)(qy * A->um / ccu)) * B->shape[2] + (size_t)(qx * A->um / ccu);
            const double bc = (B->f.pc[j] - 128) / 127.0, bs = (B->f.ps[j] - 128) / 127.0;
            pa = atan2(as, ac) / (2 * M_PI); spsum += A->f.sp[i] * cu / 8.0;
            if (k) { double d = pa - la; d -= round(d); ta += d; }
            la = pa;
            if (hypot(bc, bs) > 1e-3) { pb = atan2(bs, bc) / (2 * M_PI); if (hb) { double d = pb - lb; d -= round(d); tb += d; } lb = pb; hb = 1; }
            if (k < ns && B->f.sp[j] && B->f.w[j]) {   /* density: |n . r| / spacing over the step */
                double v[3]; fc_axis(&B->f, j, v);
                td += fabs(v[1] * ry + v[2] * rx) * ds / (B->f.sp[j] * ccu / 8.0);
            }
        }
        if (!ok) continue;
        const double ref = fabs(ta); if (ref < 0.5) continue;   /* a segment with under half a turn says nothing about counting */
        const double ep = fabs(tb) - ref, ed = td - ref;
        P->hep[(int)fmin(FC_NE - 1, fmax(0, (ep + 4) / 8 * FC_NE))]++; P->hed[(int)fmin(FC_NE - 1, fmax(0, (ed + 4) / 8 * FC_NE))]++;
        P->nseg++; P->slp += fabs(ep) >= 0.5; P->sld += fabs(ed) >= 0.5;
        if (P->nsg == P->csg) { P->csg = P->csg ? 2 * P->csg : 4096; P->seg = realloc(P->seg, P->csg * 3 * sizeof *P->seg); if (!P->seg) abort(); }
        P->seg[3 * P->nsg] = ep; P->seg[3 * P->nsg + 1] = ed; P->seg[3 * P->nsg + 2] = ref; P->nsg++;
        if (spsum / (ns + 1) < 100) { P->pk_n++; P->pk_sp += fabs(ep) >= 0.5; P->pk_sd += fabs(ed) >= 0.5; }
    }
}
static int cmd_fit_compare(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm fit-compare <reference-fields> <fields> [--surf ROOT KEY [--surf-hi 0.5] [--surf-lo 0.3]]\n"
                                    "  --surf: also count sheet crossings as peaks of a uint8 surface probability array (e.g. ufsm hercuvit) on the turn-count segments\n"); return 2; }
    fit_fields_buf A, B;
    fit_fields_skip_planes(3);   /* (neither the recto nor the winding mod 4 planes are compared) */
    if (fit_fields_load(argv[2], 0, 0, 1, &A) || fit_fields_load(argv[3], 0, 0, 1, &B)) { fprintf(stderr, "fit-compare: cannot load the fields\n"); return 1; }
    /* the candidate's cells may be k times the reference's (a coarser net, e.g. 38.4 um cells against the 19.2 um oracle): each
       reference cell reads the candidate cell that contains it */
    const int k = (int)lround(B.h * B.um / (A.h * A.um));
    for (int d = 0; d < 3; d++) {
        if (k < 1 || fabs(k * A.h * A.um - B.h * B.um) > 1e-6 * B.h * B.um || (A.shape[d] + k - 1) / k != B.shape[d] ||
            fabs(A.origin[d] * A.um - B.origin[d] * B.um) > 0.5 * A.h * A.um) {
            fprintf(stderr, "fit-compare: the field sets cover different cells (shape %d/%d, origin %.1f/%.1f um, cell %.2f/%.2f um on axis %d)\n", A.shape[d], B.shape[d], A.origin[d] * A.um, B.origin[d] * B.um, A.h * A.um, B.h * B.um, d); return 2;
        }
    }
    const size_t n = (size_t)A.shape[0] * A.shape[1] * A.shape[2];
    enum { NP = FC_NP, NC = FC_NC, NR = FC_NR, NA = FC_NA, NS = FC_NS, NK = FC_NK, NE = FC_NE };
    const double *sb = fc_sb, *cb = fc_cb;
    if (k > 1) printf("fit-compare: candidate cells %dx the reference's\n", k);
    const int ntm = fit_par_threads(1 << 30);
    fc_part *P = calloc(ntm, sizeof *P); uint8_t *cls = calloc(n ? n : 1, 1);   /* cls: the spacing class + 1 of the cells in the density sums */
    if (!P || !cls) { fprintf(stderr, "fit-compare: out of memory\n"); return 1; }
    fc_job J = {&A, &B, k, cls, P};
    int nt = fit_par(A.shape[0], fc_cells, &J);
    fc_part S = {0};
    for (int t = 0; t < nt; t++) {
        for (int q = 0; q < NP; q++) S.hp[q] += P[t].hp[q];
        for (int q = 0; q < NC; q++) { S.hca[q] += P[t].hca[q]; S.hcb[q] += P[t].hcb[q]; }
        for (int q = 0; q < NR; q++) S.hr[q] += P[t].hr[q];
        for (int q = 0; q < NA; q++) S.ha[q] += P[t].ha[q];
        for (int q = 0; q < NS; q++) { S.sn[q] += P[t].sn[q]; S.sg[q] += P[t].sg[q]; for (int r = 0; r < NR; r++) S.hrs[q][r] += P[t].hrs[q][r]; }
        for (int q = 0; q < NK; q++) { S.kn[q] += P[t].kn[q]; S.kg[q] += P[t].kg[q]; }
        S.m += P[t].m; S.lt01 += P[t].lt01; S.gt25 += P[t].gt25;
    }
    double dra[NS] = {0}, drb[NS] = {0};   /* by reference spacing: summed densities (1 / um), in cell order */
    for (size_t i = 0; i < n; i++) {
        if (!cls[i]) continue;
        const size_t az = i / ((size_t)A.shape[1] * A.shape[2]), ay = i / A.shape[2] % A.shape[1], ax = i % A.shape[2];
        const size_t j = ((az / k) * B.shape[1] + ay / k) * B.shape[2] + ax / k;
        const double spa = A.f.sp[i] * A.h / 8.0 * A.um, spb = B.f.sp[j] * B.h / 8.0 * B.um;
        dra[cls[i] - 1] += 1 / spa; drb[cls[i] - 1] += 1 / spb;
    }
    free(cls);
    const size_t m = S.m;
    if (!m) { fprintf(stderr, "fit-compare: no cells with reference phase\n"); free(P); return 1; }
    printf("fit-compare: %zu cells with reference phase (confidence > 0.3) and material in both\n", m);
    printf("  phase |error| (turns): median %.3f p75 %.3f p90 %.3f; < 0.1 %.3f, > 0.25 %.3f\n", hist_q(S.hp, NP, 0, 0.5 / NP, 0.5), hist_q(S.hp, NP, 0, 0.5 / NP, 0.75), hist_q(S.hp, NP, 0, 0.5 / NP, 0.9), (double)S.lt01 / m, (double)S.gt25 / m);
    printf("  phase confidence: median %.2f (reference %.2f)\n", hist_q(S.hcb, NC, 0, 1.0 / NC, 0.5), hist_q(S.hca, NC, 0, 1.0 / NC, 0.5));
    printf("  spacing ratio: median %.3f p10 %.3f p90 %.3f\n", exp2(hist_q(S.hr, NR, -2, 4.0 / NR, 0.5)), exp2(hist_q(S.hr, NR, -2, 4.0 / NR, 0.1)), exp2(hist_q(S.hr, NR, -2, 4.0 / NR, 0.9)));
    printf("  density ratio (candidate / reference, the sum over cells: the count ratio; median of the per-cell ratio) by reference spacing:");
    for (int q = 0; q < NS; q++) if (S.sn[q]) printf("  %s%.0f um %.3f (%.3f)", q == NS - 1 ? ">= " : "< ", q == NS - 1 ? sb[q] : sb[q + 1], dra[q] > 0 ? drb[q] / dra[q] : 0, 1 / exp2(hist_q(S.hrs[q], NR, -2, 4.0 / NR, 0.5)));
    printf("\n");
    printf("  normal angle (degrees): median %.1f p90 %.1f\n", hist_q(S.ha, NA, 0, 90.0 / NA, 0.5), hist_q(S.ha, NA, 0, 90.0 / NA, 0.9));
    printf("  gross phase errors by reference spacing:");
    for (int q = 0; q < NS; q++) printf("  %s%.0f um %.3f (%zu)", q == NS - 1 ? ">= " : "< ", q == NS - 1 ? sb[q] : sb[q + 1], S.sn[q] ? (double)S.sg[q] / S.sn[q] : 0, S.sn[q]);
    printf("\n  gross phase errors by confidence:");
    for (int q = 0; q < NK; q++) printf("  %.1f-%.1f %.3f (%zu)", cb[q], fmin(1, cb[q + 1]), S.kn[q] ? (double)S.kg[q] / S.kn[q] : 0, S.kn[q]);
    printf("\n");
    {
        nt = fit_par((A.shape[0] + 3) / 4, fc_segs, &J);
        double bp = 0, bd = 0, rt = 0;
        for (int t = 0; t < nt; t++) {
            for (int q = 0; q < NE; q++) { S.hep[q] += P[t].hep[q]; S.hed[q] += P[t].hed[q]; }
            S.nseg += P[t].nseg; S.slp += P[t].slp; S.sld += P[t].sld; S.pk_n += P[t].pk_n; S.pk_sp += P[t].pk_sp; S.pk_sd += P[t].pk_sd;
            for (size_t s = 0; s < P[t].nsg; s++) { bp += P[t].seg[3 * s]; bd += P[t].seg[3 * s + 1]; rt += P[t].seg[3 * s + 2]; }   /* (segment order) */
            free(P[t].seg);
        }
        const size_t nseg = S.nseg;
        if (nseg) {
            printf("  turn counts on %zu radial 0.5 mm segments (reference %.2f turns each on average):\n", nseg, rt / nseg);
            printf("    by phase:   error median %+.2f, mean %+.2f turns, off by >= 0.5 turn %.3f (packed < 100 um: %.3f of %zu)\n",
                   hist_q(S.hep, NE, -4, 8.0 / NE, 0.5), bp / nseg, (double)S.slp / nseg, S.pk_n ? (double)S.pk_sp / S.pk_n : 0, S.pk_n);
            printf("    by density: error median %+.2f, mean %+.2f turns, off by >= 0.5 turn %.3f (packed < 100 um: %.3f)\n",
                   hist_q(S.hed, NE, -4, 8.0 / NE, 0.5), bd / nseg, (double)S.sld / nseg, S.pk_n ? (double)S.pk_sd / S.pk_n : 0);
        }
    }
    free(P);
    if (arg_index(argc, argv, "--surf") > 0 && fc_surf_counts(argc, argv, &A)) { fit_fields_free(&A); fit_fields_free(&B); return 1; }
    fit_fields_free(&A); fit_fields_free(&B);
    return 0;
}

/* fit-compare --surf ROOT KEY [--surf-hi 0.5] [--surf-lo 0.3]: sheet crossings counted as peaks of a surface probability array
   (uint8, e.g. ufsm hercuvit surface; on the reference fields' native voxel grid, placed by its ufsm.origin_zyx) along the same radial
   0.5 mm segments as the turn counts, stepping one native voxel: hysteresis runs (on at p >= hi, off below lo), one peak (the run's
   maximum) per run inside the segment. Reference: the crossings of the unwrapped reference phase through phi0, the reference phase
   where the peaks sit (circular mean over all peaks, a first pass), so the count compares integer sheet crossings with integer peaks.
   Placement: per peak the reference phase offset from phi0 (turns) times the reference spacing there (um), |.| median / p90. */
static int fc_surf_counts(int argc, char **argv, const fit_fields_buf *A) {
    const int si = arg_index(argc, argv, "--surf");
    if (si + 2 >= argc) { fprintf(stderr, "fit-compare: --surf ROOT KEY\n"); return -1; }
    const double hi = atof(opt(argc, argv, "--surf-hi", "0.5")) * 255, lo = atof(opt(argc, argv, "--surf-lo", "0.3")) * 255;
    z3 *sz = z3_open(store_open(argv[si + 1]), argv[si + 2], nullptr); if (!sz) { fprintf(stderr, "fit-compare: --surf: %s\n", z3_error()); return -1; }
    long long so[3] = {0, 0, 0};
    {   char jp[1400]; snprintf(jp, sizeof jp, "%s/%s/zarr.json", argv[si + 1], argv[si + 2]); FILE *jf = fopen(jp, "rb"); char jb[8192] = {0};
        if (jf) { size_t nr = fread(jb, 1, sizeof jb - 1, jf); (void)nr; fclose(jf); }
        const char *q = strstr(jb, "\"origin_zyx\":"); if (q) sscanf(q + 13, " [%lld,%lld,%lld]", so, so + 1, so + 2); }
    const int64_t *ssh = z3_meta_of(sz)->shape;
    const int64_t nb[3] = {(int64_t)A->shape[0] * A->h, (int64_t)A->shape[1] * A->h, (int64_t)A->shape[2] * A->h};   /* the fields' native box */
    int64_t ro[3], rn[3];
    for (int d = 0; d < 3; d++) { ro[d] = A->origin[d] - so[d]; rn[d] = nb[d]; if (ro[d] < 0 || ro[d] + rn[d] > ssh[d]) { fprintf(stderr, "fit-compare: --surf does not cover the fields' box (axis %d)\n", d); z3_close(sz); return -1; } }
    uint8_t *S = malloc((size_t)rn[0] * rn[1] * rn[2]); if (!S || z3_read(sz, ro, rn, S, 8)) { fprintf(stderr, "fit-compare: --surf read: %s\n", z3_error()); z3_close(sz); free(S); return -1; }
    z3_close(sz);
    const double L = 500.0, cu = A->h * A->um; const int ns = (int)(L / A->um);   /* one native voxel per step */
    double *ph = malloc(sizeof(double) * (ns + 1)), *spv = malloc(sizeof(double) * (ns + 1)); int *pv = malloc(sizeof(int) * (ns + 1));
    enum { NE = 400, ND = 1000 };
    double phi0 = 0;
    for (int pass = 0; pass < 2; pass++) {
        double sc = 0, ss = 0; size_t h10[10] = {0}, n0 = 0, nseg = 0, off = 0, pkn = 0, pko = 0, npk = 0, hep[NE] = {0}, hd[ND] = {0}; double eb = 0, rt = 0;
        for (int z = 0; z < A->shape[0]; z += 4) for (int y = 0; y < A->shape[1]; y += 4) for (int x = 0; x < A->shape[2]; x += 4) {
            const double nz = A->h * (z + 0.5); const int az = (int)fmin(fmax(nz, 0), A->shape[0] * A->h); const double cyv = A->f.cy[az], cxv = A->f.cx[az];
            const double py = A->h * (y + 0.5), px = A->h * (x + 0.5);
            double ry = py - cyv, rx = px - cxv; const double rl = hypot(ry, rx); if (rl < 1) continue; ry /= rl; rx /= rl;
            int ok = 1; double u = 0, la = 0, spsum = 0;
            for (int k = 0; k <= ns && ok; k++) {
                const double qy = py + ry * k, qx = px + rx * k;   /* native voxels, box-local */
                const double cy = qy / A->h - 0.5, cx = qx / A->h - 0.5; const int iy = (int)floor(cy), ix = (int)floor(cx);
                if (iy < 0 || ix < 0 || iy + 1 >= A->shape[1] || ix + 1 >= A->shape[2] || (int)qy >= nb[1] || (int)qx >= nb[2]) { ok = 0; break; }
                const double ty = cy - iy, tx = cx - ix; double pc = 0, ps = 0, sp = 0;   /* bilinear reference phasor */
                for (int b = 0; b < 4 && ok; b++) {
                    const size_t i = ((size_t)z * A->shape[1] + iy + (b >> 1)) * A->shape[2] + ix + (b & 1); const double wgt = (b >> 1 ? ty : 1 - ty) * (b & 1 ? tx : 1 - tx);
                    const double ac = (A->f.pc[i] - 128) / 127.0, as = (A->f.ps[i] - 128) / 127.0;
                    if (!A->f.w[i] || hypot(ac, as) <= 0.3) { ok = 0; break; }
                    pc += wgt * ac; ps += wgt * as; sp += wgt * A->f.sp[i] * cu / 8.0;
                }
                if (!ok) break;
                const double a = atan2(ps, pc) / (2 * M_PI);
                if (k) { double d = a - la; d -= round(d); u += d; } else u = a;
                la = a; ph[k] = u; spv[k] = sp; spsum += sp;
                pv[k] = S[((size_t)(int)nz * rn[1] + (size_t)qy) * rn[2] + (size_t)qx];
            }
            if (!ok) continue;
            if (fabs(ph[ns] - ph[0]) < 0.5) continue;   /* as the turn counts: under half a turn says nothing */
            int npeaks = 0, on = 0, best = -1;
            for (int k = 0; k <= ns + 1; k++) {
                const int v = k <= ns ? pv[k] : -1;
                if (!on && v >= hi) { on = 1; best = k; }
                else if (on && v >= 0 && v > pv[best]) best = k;
                if (on && (v < lo || k > ns)) {   /* run ends: its peak */
                    on = 0; if (best <= 0 || best >= ns) continue;   /* a run cut by the segment end: no interior peak */
                    npeaks++;
                    double d = ph[best] - phi0; d -= round(d);   /* the sheet sits at one reference phase value (mod 1) */
                    if (pass == 0) { sc += cos(2 * M_PI * ph[best]); ss += sin(2 * M_PI * ph[best]); n0++; double f = ph[best] - floor(ph[best]); h10[(int)fmin(9, f * 10)]++; }
                    else { hd[(int)fmin(ND - 1, fabs(d) * spv[best] / 0.5)]++; npk++; }
                }
            }
            if (pass == 0) continue;
            const double ref = fabs(floor(ph[ns] - phi0) - floor(ph[0] - phi0));   /* integer crossings of phi0 */
            const double e = npeaks - ref;
            hep[(int)fmin(NE - 1, fmax(0, (e + 4) / 8 * NE))]++; nseg++; off += fabs(e) >= 0.5; eb += e; rt += ref;
            if (spsum / (ns + 1) < 100) { pkn++; pko += fabs(e) >= 0.5; }
        }
        if (pass == 0) { phi0 = atan2(ss, sc) / (2 * M_PI); printf("  surface peaks (--surf, hysteresis %.2f / %.2f): reference phase at the peaks phi0 %+.3f turn (concentration %.2f over %zu peaks)\n", hi / 255, lo / 255, phi0, n0 ? hypot(sc, ss) / n0 : 0.0, n0);
                         printf("    reference phase histogram at the peaks (10 bins of 0.1 turn):"); for (int b = 0; b < 10; b++) printf(" %.3f", n0 ? (double)h10[b] / n0 : 0.0); printf("\n"); continue; }
        if (!nseg) { printf("  surface peaks: no segments\n"); break; }
        printf("  surface peak counts on %zu segments (reference %.2f phi0 crossings each): error median %+.2f, mean %+.2f, off by >= 0.5 %.3f (packed < 100 um: %.3f of %zu)\n",
               nseg, rt / nseg, hist_q(hep, NE, -4, 8.0 / NE, 0.5), eb / nseg, (double)off / nseg, pkn ? (double)pko / pkn : 0, pkn);
        printf("  surface peak placement (%zu peaks, |phase offset from phi0| x reference spacing): median %.1f um, p75 %.1f, p90 %.1f\n",
               npk, hist_q(hd, ND, 0, 0.5, 0.5), hist_q(hd, ND, 0, 0.5, 0.75), hist_q(hd, ND, 0, 0.5, 0.9));
    }
    free(S); free(ph); free(spv); free(pv);
    return 0;
}

/* ufsm gap-compare <array-root> <array-key> <reference-fields> [--net FIELDS] [--R-um 115] [--thr 128] [--min-run 2] [--origin z,y,x]
   Sheet density from gaps in a uint8 array (a papyrus / void probability from predict, or a raw CT level; the key is its voxel
   size in um, positions mapped physically; origin from --origin, ufsm.origin_zyx, else 0): per reference field cell with phase
   (confidence > 0.3) the radial ray through the cell centre at its z, +-R um, binarised at --thr, void -> papyrus entries counted (runs shorter than --min-run ignored) per
   2R voxels; against the reference's sheet density (1 / spacing), by reference spacing; --net: the same for a field set's
   density (e.g. a winding network's, any native voxel size covering the same cells). Medians / quantiles of the ratios. */
static double axis_lerp_tab(const float *v, int n, double z) {   /* a per-native-z axis table at fractional z (clamped) */
    if (z <= 0) return v[0];
    if (z >= n - 1) return v[n - 1];
    const int i = (int)z; const double t = z - i; return v[i] + t * (v[i + 1] - v[i]);
}
typedef struct { const uint8_t *pl; int64_t H, W; const fit_fields_buf *A, *Bn; int fz, R, thr, minrun; double s; long long po[3]; size_t (*hg)[5][200], (*hn)[5][200]; pthread_mutex_t *mu; _Atomic int next; } gap_job;
static void *gap_rows(void *arg) {
    gap_job *J = arg; const fit_fields *f = &J->A->f; const int na = (int)(f->D * f->h) + 1;
    size_t lg[5][200] = {{0}}, ln[5][200] = {{0}};
    for (int fy; (fy = atomic_fetch_add(&J->next, 1)) < f->H;) for (int fx = 0; fx < f->W; fx++) {
        const size_t i = ((size_t)J->fz * f->H + fy) * f->W + fx;
        if (!f->w[i] || !f->sp[i]) continue;
        const double mc = (f->pc[i] - 128) / 127.0, ms = (f->ps[i] - 128) / 127.0; if (hypot(mc, ms) <= 0.3) continue;
        const double pz = (J->fz + 0.5) * f->h, py = (fy + 0.5) * f->h, px = (fx + 0.5) * f->h;
        const double cy = axis_lerp_tab(f->cy, na, pz), cx = axis_lerp_tab(f->cx, na, pz);
        double uy = py - cy, ux = px - cx; const double ul = hypot(uy, ux); if (ul < 1) continue; uy /= ul; ux /= ul;
        const double yp = (py + J->A->origin[1]) * J->s - J->po[1], xp = (px + J->A->origin[2]) * J->s - J->po[2];   /* on the array's grid */
        int prev = -1, run = 0, entries = 0, lastv = -1, len = 0;
        for (int t = -J->R; t <= J->R; t++) {
            const int64_t yy = (int64_t)lround(yp + t * uy), xx = (int64_t)lround(xp + t * ux);
            if (yy < 0 || xx < 0 || yy >= J->H || xx >= J->W) { prev = -1; run = 0; continue; }
            len++;
            const int v = J->pl[(size_t)yy * J->W + xx] >= J->thr;
            if (v == prev) { run++; if (run == J->minrun && v == 1 && lastv == 0) entries++; if (run >= J->minrun) lastv = v; }
            else { prev = v; run = 1; if (J->minrun <= 1) { if (v == 1 && lastv == 0) entries++; lastv = v; } }
        }
        if (len < J->R) continue;
        const double spa = f->sp[i] * f->h / 8.0 * J->A->um, rho_ref = 1.0 / spa;   /* sheets per um */
        const double rho_gap = entries / (len * J->A->um / J->s);
        int sbin = spa < 100 ? 0 : spa < 140 ? 1 : spa < 180 ? 2 : spa < 240 ? 3 : 4;
        if (rho_gap > 0) lg[sbin][(int)fmin(199, fmax(0, (log2(rho_gap / rho_ref) + 2) / 4 * 200))]++; else lg[sbin][0]++;
        if (J->Bn && J->Bn->f.sp[i]) { const double spb = J->Bn->f.sp[i] * J->Bn->h / 8.0 * J->Bn->um; ln[sbin][(int)fmin(199, fmax(0, (log2(spa / spb) + 2) / 4 * 200))]++; }
    }
    pthread_mutex_lock(J->mu);
    for (int a = 0; a < 5; a++) for (int b = 0; b < 200; b++) { (*J->hg)[a][b] += lg[a][b]; (*J->hn)[a][b] += ln[a][b]; }
    pthread_mutex_unlock(J->mu);
    return nullptr;
}
static int cmd_gap_compare(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: ufsm gap-compare <array-root> <array-key> <reference-fields> [--net FIELDS] [--R-um 115] [--thr 128] [--min-run 2] [--origin z,y,x] [--um U] [--cache DIR] [--threads 16]\n"); return 2; }
    const double aum = atof(opt(argc, argv, "--um", "0")) > 0 ? atof(opt(argc, argv, "--um", "0")) : atof(argv[3]), Rum = atof(opt(argc, argv, "--R-um", "115"));   /* --um: when the key is not the voxel size */
    const int thr = atoi(opt(argc, argv, "--thr", "128")), minrun = atoi(opt(argc, argv, "--min-run", "2")), nt = ufsm_threads(atoi(opt(argc, argv, "--threads", "0")), 16);
    if (!(aum > 0)) { fprintf(stderr, "gap-compare: the array key must be its voxel size in um (e.g. 2.4)\n"); return 2; }
    fit_fields_buf A, Bn; const char *net = opt(argc, argv, "--net", nullptr);
    if (fit_fields_load(argv[4], 0, 0, 1, &A) || (net && fit_fields_load(net, 0, 0, 1, &Bn))) { fprintf(stderr, "gap-compare: cannot load the fields\n"); return 1; }
    z3 *pz = z3_open(store_open(argv[2]), argv[3], opt(argc, argv, "--cache", nullptr)); if (!pz) { fprintf(stderr, "gap-compare: %s\n", z3_error()); return 1; }
    long long po[3] = {0, 0, 0};   /* the array's origin on its own grid: --origin, else ufsm.origin_zyx (predict outputs), else 0 (whole volumes) */
    if (*opt(argc, argv, "--origin", "")) sscanf(opt(argc, argv, "--origin", ""), "%lld,%lld,%lld", po, po + 1, po + 2);
    else { char jp[1400]; snprintf(jp, sizeof jp, "%s/%s/zarr.json", argv[2], argv[3]); FILE *jf = fopen(jp, "rb"); char jb[8192] = {0};
           if (jf) { size_t nr = fread(jb, 1, sizeof jb - 1, jf); (void)nr; fclose(jf); }
           const char *q = strstr(jb, "\"origin_zyx\":"); if (q) sscanf(q + 13, " [%lld,%lld,%lld]", po, po + 1, po + 2); }
    const z3_meta *zm = z3_meta_of(pz); const double s = A.um / aum; const int R = (int)lround(Rum / aum);
    /* the planes of the array covering the fields' y / x extent */
    long long lo[2], hi[2];
    for (int d = 1; d < 3; d++) {
        lo[d - 1] = (long long)floor(A.origin[d] * s) - po[d] - R - 2; hi[d - 1] = (long long)ceil((A.origin[d] + A.shape[d] * A.h) * s) - po[d] + R + 2;
        if (lo[d - 1] < 0) lo[d - 1] = 0;
        if (hi[d - 1] > zm->shape[d]) hi[d - 1] = zm->shape[d];
    }
    const int64_t H = hi[0] - lo[0], W = hi[1] - lo[1];
    uint8_t *pl = malloc((size_t)H * W); size_t hg[5][200] = {{0}}, hn[5][200] = {{0}}; pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    const long long pol[3] = {po[0], po[1] + lo[0], po[2] + lo[1]};
    for (int fz = 0; fz < A.shape[0]; fz++) {
        const int64_t z = (int64_t)lround(((fz + 0.5) * A.h + A.origin[0]) * s) - po[0]; if (z < 0 || z >= zm->shape[0]) continue;
        const int64_t o[3] = {z, lo[0], lo[1]}, n[3] = {1, H, W};
        if (z3_read(pz, o, n, pl, nt)) { fprintf(stderr, "gap-compare: %s\n", z3_error()); return 1; }
        gap_job J = {pl, H, W, &A, net ? &Bn : nullptr, fz, R, thr, minrun, s, {pol[0], pol[1], pol[2]}, &hg, &hn, &mu, 0};
        pthread_t th[64]; const int k = nt < 1 ? 1 : nt > 64 ? 64 : nt;
        for (int t = 0; t < k; t++) pthread_create(&th[t], nullptr, gap_rows, &J);
        for (int t = 0; t < k; t++) pthread_join(th[t], nullptr);
    }
    static const char *bn[5] = {"< 100 um", "100-140", "140-180", "180-240", ">= 240"};
    printf("gap-compare: density ratio to the reference (median, p10, p90) by reference spacing; R %.0f um, threshold %d, min run %d\n", Rum, thr, minrun);
    for (int a = 0; a < 5; a++) {
        size_t t = 0; for (int b = 0; b < 200; b++) t += hg[a][b]; if (!t) continue;
        printf("  %-9s %8zu cells: gaps %.2f (%.2f-%.2f)", bn[a], t, exp2(hist_q(hg[a], 200, -2, 4.0 / 200, 0.5)), exp2(hist_q(hg[a], 200, -2, 4.0 / 200, 0.1)), exp2(hist_q(hg[a], 200, -2, 4.0 / 200, 0.9)));
        size_t u = 0; for (int b = 0; b < 200; b++) u += hn[a][b];
        if (u) printf("   net density %.2f (%.2f-%.2f)", exp2(hist_q(hn[a], 200, -2, 4.0 / 200, 0.5)), exp2(hist_q(hn[a], 200, -2, 4.0 / 200, 0.1)), exp2(hist_q(hn[a], 200, -2, 4.0 / 200, 0.9)));
        printf("\n");
    }
    free(pl); z3_close(pz); fit_fields_free(&A); if (net) fit_fields_free(&Bn);
    return 0;
}

/* ufsm gap-density <papyrus-root> <papyrus-key> <fields-in> <fields-out> [--fuse-um 150] [--R-um 115] [--thr 128] [--min-run 2]
   A field set (any native voxel size) with its sheet spacing fused with the papyrus-gap spacing: per cell the radial ray through
   its centre at its z, +-R um, over the papyrus / void probability (task papyrus predict output; its key is its voxel size in um,
   origin_zyx from its attributes), void -> papyrus entries counted (runs shorter than --min-run voxels ignored). Where the
   fields' spacing is under --fuse-um and the gaps' is smaller, the gaps' replaces it: networks under-count packed sheets (block B:
   0.72 of the true density under 100 um) while papyrus gaps count them (1.04) but over-count loose sheets (each sheet's own
   layers and cracks: 1.4-2.2). Writes fields-out as raw fit fields. */
static int cmd_gap_density(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: ufsm gap-density <papyrus-root> <papyrus-key> <fields-in> <fields-out> [--fuse-um 150] [--R-um 115] [--thr 128] [--min-run 2]\n"); return 2; }
    const double fuse = atof(opt(argc, argv, "--fuse-um", "150")), Rum = atof(opt(argc, argv, "--R-um", "115")), ump = atof(argv[3]);
    const int thr = atoi(opt(argc, argv, "--thr", "128")), minrun = atoi(opt(argc, argv, "--min-run", "2")), nt = ufsm_threads(atoi(opt(argc, argv, "--threads", "0")), 16);
    if (!(ump > 0)) { fprintf(stderr, "gap-density: the papyrus key must be its voxel size in um\n"); return 2; }
    fit_fields_buf A; if (fit_fields_load(argv[4], 0, 0, 1, &A)) { fprintf(stderr, "gap-density: cannot load %s\n", argv[4]); return 1; }
    z3 *pz = z3_open(store_open(argv[2]), argv[3], nullptr); if (!pz) { fprintf(stderr, "gap-density: %s\n", z3_error()); return 1; }
    long long po[3] = {0, 0, 0};
    { char jp[1400]; snprintf(jp, sizeof jp, "%s/%s/zarr.json", argv[2], argv[3]); FILE *jf = fopen(jp, "rb"); char jb[8192] = {0};
      if (jf) { size_t nr = fread(jb, 1, sizeof jb - 1, jf); (void)nr; fclose(jf); }
      const char *q = strstr(jb, "\"origin_zyx\":"); if (!q || sscanf(q + 13, " [%lld,%lld,%lld]", po, po + 1, po + 2) != 3) { fprintf(stderr, "gap-density: no origin_zyx in %s\n", jp); return 2; } }
    const z3_meta *zm = z3_meta_of(pz); const int64_t PH = zm->shape[1], PW = zm->shape[2];
    const double s = A.um / ump;   /* fields' native voxel -> papyrus voxels */
    const int R = (int)lround(Rum / ump); const int na = (int)(A.f.D * A.f.h) + 1;
    uint8_t *pl = malloc((size_t)PH * PW), *sp = malloc((size_t)A.shape[0] * A.shape[1] * A.shape[2]);
    size_t fused = 0, cells = 0;
    for (int fz = 0; fz < A.shape[0]; fz++) {
        const double zf = (fz + 0.5) * A.h; const int64_t zp = (int64_t)lround((zf + A.origin[0]) * s) - po[0];
        const int have = zp >= 0 && zp < zm->shape[0];
        if (have) { const int64_t o[3] = {zp, 0, 0}, n[3] = {1, PH, PW}; if (z3_read(pz, o, n, pl, nt)) { fprintf(stderr, "gap-density: %s\n", z3_error()); return 1; } }
        const double cy = axis_lerp_tab(A.f.cy, na, zf), cx = axis_lerp_tab(A.f.cx, na, zf);
        for (int fy = 0; fy < A.shape[1]; fy++) for (int fx = 0; fx < A.shape[2]; fx++) {
            const size_t i = ((size_t)fz * A.shape[1] + fy) * A.shape[2] + fx; sp[i] = A.f.sp[i];
            if (!have || !A.f.w[i] || !A.f.sp[i]) continue;
            const double spn = A.f.sp[i] * A.h / 8.0 * A.um; if (spn >= fuse) continue;
            const double yf = (fy + 0.5) * A.h, xf = (fx + 0.5) * A.h; double uy = yf - cy, ux = xf - cx; const double ul = hypot(uy, ux); if (ul < 1) continue; uy /= ul; ux /= ul;
            const double yp = (yf + A.origin[1]) * s - po[1], xp = (xf + A.origin[2]) * s - po[2];
            int prev = -1, run = 0, entries = 0, lastv = -1, len = 0;
            for (int t = -R; t <= R; t++) {
                const int64_t yy = (int64_t)lround(yp + t * uy), xx = (int64_t)lround(xp + t * ux);
                if (yy < 0 || xx < 0 || yy >= PH || xx >= PW) { prev = -1; run = 0; continue; }
                len++;
                const int v = pl[(size_t)yy * PW + xx] >= thr;
                if (v == prev) { run++; if (run == minrun && v == 1 && lastv == 0) entries++; if (run >= minrun) lastv = v; }
                else { prev = v; run = 1; if (minrun <= 1) { if (v == 1 && lastv == 0) entries++; lastv = v; } }
            }
            cells++;
            if (!entries || len < R) continue;
            const double spg = len * ump / entries;   /* um per sheet */
            if (spg < spn) { const double v = spg / A.um / A.h * 8.0; sp[i] = (uint8_t)(v < 1 ? 1 : v > 255 ? 255 : lround(v)); fused++; }
        }
    }
    uint8_t *a[10] = {A.a[0], A.a[1], A.a[2], A.a[3], A.a[4], A.a[5], A.a[6], A.a[7], sp, A.a[9]};
    const long long org[3] = {A.origin[0], A.origin[1], A.origin[2]};
    if (fit_fields_save(argv[5], org, A.shape, A.h, A.um, a, A.cy, A.cx, na)) { fprintf(stderr, "gap-density: cannot write %s\n", argv[5]); return 1; }
    fprintf(stderr, "gap-density: %zu of %zu packed cells took the papyrus-gap spacing -> %s\n", fused, cells, argv[5]);
    free(pl); free(sp); z3_close(pz); fit_fields_free(&A);
    return 0;
}

/* ufsm void-pool <pred-root> <pred-key> <out-group> <out-um> --vol-shape z,y,x [--factor 4] [--thr 128] [--shard 512]
   The aux input for a coarse winding net (train --aux-input): per output voxel the fraction of its factor^3 source voxels that are
   void (papyrus probability < --thr) x 255, from a task papyrus predict output (box-local, ufsm.origin_zyx; on the level-0 grid
   of --vol-shape) into <out-group>/<out-um>, a whole-volume array (shape ceil(vol-shape / factor)) shared by many boxes: each
   output shard the box meets is read, the box's part replaced, and written back. Thin gaps between packed sheets survive as
   void fractions where a 9.6 um CT averages them away. */
static int cmd_void_pool(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: ufsm void-pool <pred-root> <pred-key> <out-group> <out-um> --vol-shape z,y,x [--factor 4] [--thr 128] [--shard 512]\n"); return 2; }
    const int F = atoi(opt(argc, argv, "--factor", "4")), thr = atoi(opt(argc, argv, "--thr", "128")), S = atoi(opt(argc, argv, "--shard", "512")), nt = 16;
    long long vs[3]; if (sscanf(opt(argc, argv, "--vol-shape", ""), "%lld,%lld,%lld", vs, vs + 1, vs + 2) != 3 || F < 1 || S < 1) { fprintf(stderr, "void-pool: --vol-shape z,y,x (level 0)\n"); return 2; }
    z3 *pz = z3_open(store_open(argv[2]), argv[3], nullptr); if (!pz) { fprintf(stderr, "void-pool: %s\n", z3_error()); return 1; }
    long long po[3] = {0, 0, 0};
    { char jp[1400]; snprintf(jp, sizeof jp, "%s/%s/zarr.json", argv[2], argv[3]); FILE *jf = fopen(jp, "rb"); char jb[8192] = {0};
      if (jf) { size_t nr = fread(jb, 1, sizeof jb - 1, jf); (void)nr; fclose(jf); }
      const char *q = strstr(jb, "\"origin_zyx\":"); if (!q || sscanf(q + 13, " [%lld,%lld,%lld]", po, po + 1, po + 2) != 3) { fprintf(stderr, "void-pool: no origin_zyx in %s\n", jp); return 2; } }
    const z3_meta *pm = z3_meta_of(pz); const double um = atof(argv[5]);
    const int64_t osh[3] = {(vs[0] + F - 1) / F, (vs[1] + F - 1) / F, (vs[2] + F - 1) / F};
    char lv[32], od[1400]; z3w_level_name(um, lv, sizeof lv); snprintf(od, sizeof od, "%s/%s", argv[4], lv);
    char at[300]; snprintf(at, sizeof at, "{\"ufsm\":{\"content\":\"papyrus void fraction x 255 over %d^3 voxels (probability < %d)\",\"factor\":%d}}", F, thr, F);
    z3w *w = z3w_create(od, osh, S, 0, 0, at); if (!w || z3w_write_group(argv[4], &um, 1, "ufsm-void-fraction", nullptr)) { fprintf(stderr, "void-pool: %s\n", z3w_error()); return 1; }
    z3 *oz = z3_open(store_open(argv[4]), lv, nullptr);   /* existing shard contents (other boxes) */
    int64_t lo[3], hi[3]; for (int d = 0; d < 3; d++) { lo[d] = po[d] / F; hi[d] = (po[d] + pm->shape[d] + F - 1) / F; if (hi[d] > osh[d]) hi[d] = osh[d]; }
    uint8_t *sb = malloc((size_t)S * S * S), *src = nullptr; size_t srccap = 0; int nsh = 0;
    for (int64_t sz = lo[0] / S; sz * S < hi[0]; sz++) for (int64_t sy = lo[1] / S; sy * S < hi[1]; sy++) for (int64_t sx = lo[2] / S; sx * S < hi[2]; sx++) {
        const int64_t so[3] = {sz * S, sy * S, sx * S}, sn[3] = {osh[0] - so[0] < S ? osh[0] - so[0] : S, osh[1] - so[1] < S ? osh[1] - so[1] : S, osh[2] - so[2] < S ? osh[2] - so[2] : S};
        memset(sb, 0, (size_t)S * S * S);
        if (oz) { uint8_t *t = malloc((size_t)sn[0] * sn[1] * sn[2]); if (t && !z3_read(oz, so, sn, t, nt)) for (int64_t a = 0; a < sn[0]; a++) for (int64_t b = 0; b < sn[1]; b++) memcpy(sb + ((size_t)a * S + b) * S, t + ((size_t)a * sn[1] + b) * sn[2], (size_t)sn[2]); free(t); }
        int64_t c0[3], c1[3];   /* this shard's output voxels inside the box */
        for (int d = 0; d < 3; d++) { c0[d] = lo[d] > so[d] ? lo[d] : so[d]; c1[d] = hi[d] < so[d] + sn[d] ? hi[d] : so[d] + sn[d]; }
        if (c1[0] <= c0[0] || c1[1] <= c0[1] || c1[2] <= c0[2]) continue;
        for (int64_t a0 = c0[0]; a0 < c1[0]; a0 += 32) {   /* 32 output rows at a time: a ~0.5 GB source slab (a whole 2048^3 box: 8.7 GB) */
            const int64_t a1 = a0 + 32 < c1[0] ? a0 + 32 : c1[0], cl[3] = {a0, c0[1], c0[2]}, ch[3] = {a1, c1[1], c1[2]};
            int64_t ro[3], rn[3];   /* the source voxels behind them (box-local) */
            for (int d = 0; d < 3; d++) { ro[d] = cl[d] * F - po[d]; int64_t e = ch[d] * F - po[d]; if (ro[d] < 0) ro[d] = 0; if (e > pm->shape[d]) e = pm->shape[d]; rn[d] = e - ro[d]; }
            if (rn[0] <= 0 || rn[1] <= 0 || rn[2] <= 0) continue;
            const size_t need = (size_t)rn[0] * rn[1] * rn[2]; if (need > srccap) { free(src); src = malloc(need); srccap = need; }
            if (!src || z3_read(pz, ro, rn, src, nt)) { fprintf(stderr, "void-pool: %s\n", z3_error()); return 1; }
            for (int64_t a = a0; a < a1; a++) for (int64_t b = c0[1]; b < c1[1]; b++) for (int64_t c = c0[2]; c < c1[2]; c++) {
                int tot = 0, vd = 0;
                for (int64_t za = a * F - po[0]; za < (a + 1) * F - po[0]; za++) { if (za < ro[0] || za >= ro[0] + rn[0]) continue;
                    for (int64_t ya = b * F - po[1]; ya < (b + 1) * F - po[1]; ya++) { if (ya < ro[1] || ya >= ro[1] + rn[1]) continue;
                        for (int64_t xa = c * F - po[2]; xa < (c + 1) * F - po[2]; xa++) { if (xa < ro[2] || xa >= ro[2] + rn[2]) continue;
                            tot++; vd += src[((size_t)(za - ro[0]) * rn[1] + (ya - ro[1])) * rn[2] + (xa - ro[2])] < thr; } } }
                if (tot) sb[((size_t)(a - so[0]) * S + (b - so[1])) * S + (c - so[2])] = (uint8_t)((vd * 255 + tot / 2) / tot);
            }
        }
        if (z3w_write_shard(w, sz, sy, sx, sb, nt)) { fprintf(stderr, "void-pool: %s\n", z3w_error()); return 1; }
        nsh++;
    }
    z3w_close(w); if (oz) z3_close(oz); z3_close(pz); free(sb); free(src);
    fprintf(stderr, "void-pool: %d shards of %s updated from %s\n", nsh, od, argv[2]);
    return 0;
}

/* ufsm papyrus-labels <ct-root> <ct-key> <out-group> --box z,y,x,nz,ny,nx [--um 2.4] [--local 8] [--local-sigma 1.5] [--depth 30]
   [--margin 0.1] [--sigma 1] [--cut 50,128] [--band 0] [--cutoff T] [--threads 8] [--cache DIR]
   The task papyrus training labels (src/papyrus.h) of a CT box as a box-local uint8 array <out-group>/<um> (ufsm.origin_zyx, so
   gap-compare and the overlays place it): 255 papyrus, 0 void or outside the scroll, 128 ignored. Gap labels by default (--local
   R, papyrus_local); --cutoff T: the old global cutoff at T instead. Per 512^3 output shard the CT is read once with a halo covering
   the smoothing and the closing window, then labelled in 256^3 tiles, one per thread. */
typedef struct { const uint8_t *ct; int64_t cn[3]; const papyrus_cfg *c; float t; int T, H, P; uint8_t *out; int64_t on[3], oo[3]; _Atomic int next; } plab_job;
static void *plab_tiles(void *arg) {
    plab_job *J = arg; const int P = J->P, T = J->T, H = J->H; const size_t p3 = (size_t)P * P * P;
    uint8_t *ct = malloc(p3), *tg = malloc(p3), *ig = malloc(p3), *fg = malloc(p3), *sc = malloc(8 * p3);
    if (!ct || !tg || !ig || !fg || !sc) abort();
    for (int k; (k = atomic_fetch_add(&J->next, 1)) < 8;) {
        const int64_t t0[3] = {(k >> 2 & 1) * (int64_t)T, (k >> 1 & 1) * (int64_t)T, (k & 1) * (int64_t)T};   /* tile origin in the shard */
        if (t0[0] >= J->on[0] || t0[1] >= J->on[1] || t0[2] >= J->on[2]) continue;
        for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) {   /* the haloed tile from the shard's CT (zero outside it) */
            const int64_t a = t0[0] + z, b = t0[1] + y; uint8_t *d = ct + ((size_t)z * P + y) * P;
            if (a >= J->cn[0] || b >= J->cn[1]) { memset(d, 0, P); continue; }
            const int64_t nx = J->cn[2] - t0[2] < P ? J->cn[2] - t0[2] : P;
            memcpy(d, J->ct + ((size_t)a * J->cn[1] + b) * J->cn[2] + t0[2], (size_t)nx); if (nx < P) memset(d + nx, 0, (size_t)(P - nx));
        }
        memset(ig, 0, p3);
        papyrus_labels(ct, P, J->c, J->t, 0.0, tg, ig, fg, sc);
        for (int z = 0; z < T && t0[0] + z < J->on[0]; z++) for (int y = 0; y < T && t0[1] + y < J->on[1]; y++) {
            const size_t s = ((size_t)(z + H) * P + y + H) * P + H; uint8_t *o = J->out + ((size_t)(t0[0] + z) * J->on[1] + t0[1] + y) * J->on[2] + t0[2];
            for (int x = 0; x < T && t0[2] + x < J->on[2]; x++) o[x] = ig[s + x] ? 128 : tg[s + x];
        }
    }
    free(ct); free(tg); free(ig); free(fg); free(sc);
    return nullptr;
}
static int cmd_papyrus_labels(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: ufsm papyrus-labels <ct-root> <ct-key> <out-group> --box z,y,x,nz,ny,nx [--um 2.4] [--local 8] [--local-sigma 1.5] [--depth 30] [--margin 0.1] [--sigma 1] [--cut 50,128] [--band 0] [--cutoff T] [--threads 8] [--cache DIR]\n"); return 2; }
    long long bx[6]; papyrus_cfg c; memset(&c, 0, sizeof c);
    if (sscanf(opt(argc, argv, "--box", ""), "%lld,%lld,%lld,%lld,%lld,%lld", bx, bx + 1, bx + 2, bx + 3, bx + 4, bx + 5) != 6) { fprintf(stderr, "papyrus-labels: --box z,y,x,nz,ny,nx\n"); return 2; }
    const float cutoff = (float)atof(opt(argc, argv, "--cutoff", "-1"));
    c.local = cutoff >= 0 ? 0 : atoi(opt(argc, argv, "--local", "8")); c.local_sigma = (float)atof(opt(argc, argv, "--local-sigma", "1.5"));
    c.local_depth = (float)atof(opt(argc, argv, "--depth", "30")); c.local_margin = (float)atof(opt(argc, argv, "--margin", "0.1"));
    c.sigma = (float)atof(opt(argc, argv, "--sigma", "1")); c.band = (float)atof(opt(argc, argv, "--band", "0"));
    if (sscanf(opt(argc, argv, "--cut", "50,128"), "%f,%f", &c.t_lo, &c.t_hi) != 2 || !(c.sigma >= 0 && c.sigma <= PAPYRUS_RMAX / 3.f) ||
        !(c.local_sigma >= 0 && c.local_sigma <= PAPYRUS_RMAX / 3.f) || (cutoff < 0 && c.local < 1)) { fprintf(stderr, "papyrus-labels: bad label options\n"); return 2; }
    const int nt = ufsm_threads(atoi(opt(argc, argv, "--threads", "0")), 8), S = 512, T = S / 2;
    float w[2 * PAPYRUS_RMAX + 1]; const int rs = papyrus_taps(c.sigma, w), rl = papyrus_taps(c.local_sigma, w);
    const int H = cutoff >= 0 ? rs : (rs > 2 * c.local + rl ? rs : 2 * c.local + rl), P = T + 2 * H;   /* what a core voxel's label can see */
    z3 *cz = z3_open(store_open(argv[2]), argv[3], opt(argc, argv, "--cache", nullptr)); if (!cz) { fprintf(stderr, "papyrus-labels: %s\n", z3_error()); return 1; }
    const z3_meta *cm = z3_meta_of(cz); const double um = atof(opt(argc, argv, "--um", "2.4"));
    const int64_t shp[3] = {bx[3], bx[4], bx[5]};
    char lv[32], od[1400]; z3w_level_name(um, lv, sizeof lv); snprintf(od, sizeof od, "%s/%s", argv[4], lv);
    char at[600];
    if (cutoff >= 0) snprintf(at, sizeof at, "{\"ufsm\":{\"content\":\"papyrus labels at cutoff %g (255 papyrus, 0 void, 128 ignored)\",\"origin_zyx\":[%lld,%lld,%lld],\"sigma\":%g,\"band\":%g}}", (double)cutoff, bx[0], bx[1], bx[2], (double)c.sigma, (double)c.band);
    else snprintf(at, sizeof at, "{\"ufsm\":{\"content\":\"papyrus gap labels (255 papyrus, 0 void, 128 ignored)\",\"origin_zyx\":[%lld,%lld,%lld],\"sigma\":%g,\"cut\":[%g,%g],\"local\":%d,\"local_sigma\":%g,\"local_depth\":%g,\"local_margin\":%g,\"band\":%g}}",
                  bx[0], bx[1], bx[2], (double)c.sigma, (double)c.t_lo, (double)c.t_hi, c.local, (double)c.local_sigma, (double)c.local_depth, (double)c.local_margin, (double)c.band);
    z3w *ow = z3w_create(od, shp, S, 0, 0, at); if (!ow || z3w_write_group(argv[4], &um, 1, "ufsm-papyrus-labels", nullptr)) { fprintf(stderr, "papyrus-labels: %s\n", z3w_error()); return 1; }
    const size_t cap = (size_t)(S + 2 * H) * (S + 2 * H) * (S + 2 * H);
    uint8_t *cbuf = malloc(cap), *obuf = malloc((size_t)S * S * S); if (!cbuf || !obuf) { fprintf(stderr, "papyrus-labels: out of memory\n"); return 1; }
    const double t0 = now(); int nsh = 0; const int64_t ns[3] = {(shp[0] + S - 1) / S, (shp[1] + S - 1) / S, (shp[2] + S - 1) / S};
    for (int64_t sz = 0; sz < ns[0]; sz++) for (int64_t sy = 0; sy < ns[1]; sy++) for (int64_t sx = 0; sx < ns[2]; sx++) {
        const int64_t so[3] = {sz * S, sy * S, sx * S}, on[3] = {shp[0] - so[0] < S ? shp[0] - so[0] : S, shp[1] - so[1] < S ? shp[1] - so[1] : S, shp[2] - so[2] < S ? shp[2] - so[2] : S};
        int64_t want0[3], cn[3], ro[3], rn[3];   /* the haloed shard in volume coordinates, and the part inside the volume */
        for (int d = 0; d < 3; d++) { want0[d] = bx[d] + so[d] - H; cn[d] = on[d] + 2 * H; ro[d] = want0[d] < 0 ? 0 : want0[d]; int64_t e = want0[d] + cn[d]; if (e > cm->shape[d]) e = cm->shape[d]; rn[d] = e - ro[d]; }
        memset(cbuf, 0, (size_t)cn[0] * cn[1] * cn[2]);
        if (rn[0] > 0 && rn[1] > 0 && rn[2] > 0) {
            uint8_t *tmp = malloc((size_t)rn[0] * rn[1] * rn[2]);
            if (!tmp || z3_read(cz, ro, rn, tmp, nt)) { fprintf(stderr, "papyrus-labels: %s\n", z3_error()); return 1; }
            for (int64_t a = 0; a < rn[0]; a++) for (int64_t b = 0; b < rn[1]; b++)
                memcpy(cbuf + ((size_t)(ro[0] - want0[0] + a) * cn[1] + (ro[1] - want0[1] + b)) * cn[2] + (ro[2] - want0[2]), tmp + ((size_t)a * rn[1] + b) * rn[2], (size_t)rn[2]);
            free(tmp);
        }
        memset(obuf, 0, (size_t)S * S * S);
        plab_job J = {cbuf, {cn[0], cn[1], cn[2]}, &c, cutoff, T, H, P, obuf, {on[0], on[1], on[2]}, {0, 0, 0}, 0};
        /* the job writes rows of on[2] voxels; the shard writer wants S^3 */
        uint8_t *dense = malloc((size_t)on[0] * on[1] * on[2]); if (!dense) abort(); J.out = dense;
        pthread_t th[8]; const int k = nt < 1 ? 1 : nt > 8 ? 8 : nt;
        for (int t = 0; t < k; t++) pthread_create(&th[t], nullptr, plab_tiles, &J);
        for (int t = 0; t < k; t++) pthread_join(th[t], nullptr);
        for (int64_t a = 0; a < on[0]; a++) for (int64_t b = 0; b < on[1]; b++) memcpy(obuf + ((size_t)a * S + b) * S, dense + ((size_t)a * on[1] + b) * on[2], (size_t)on[2]);
        free(dense);
        if (z3w_write_shard(ow, sz, sy, sx, obuf, nt)) { fprintf(stderr, "papyrus-labels: %s\n", z3w_error()); return 1; }
        fprintf(stderr, "\rpapyrus-labels: shard %d / %lld  %.0fs", ++nsh, (long long)(ns[0] * ns[1] * ns[2]), now() - t0);
    }
    z3w_close(ow); z3_close(cz); free(cbuf); free(obuf);
    fprintf(stderr, "\npapyrus-labels: %s (%s labels, halo %d) in %.0fs\n", od, cutoff >= 0 ? "cutoff" : "gap", H, now() - t0);
    return 0;
}

/* An 8-bit RGB PNG (zlib deflate, filter 0 rows) */
static void png_chunk(FILE *f, const char *type, const uint8_t *d, uint32_t n) {
    const uint8_t be[4] = {(uint8_t)(n >> 24), (uint8_t)(n >> 16), (uint8_t)(n >> 8), (uint8_t)n}; fwrite(be, 1, 4, f); fwrite(type, 1, 4, f);
    if (n) fwrite(d, 1, n, f);
    uLong c = crc32(0L, Z_NULL, 0); c = crc32(c, (const Bytef *)type, 4); if (n) c = crc32(c, d, n);
    const uint8_t cb[4] = {(uint8_t)(c >> 24), (uint8_t)(c >> 16), (uint8_t)(c >> 8), (uint8_t)c}; fwrite(cb, 1, 4, f);
}
static int png_write_rgb(const char *path, int w, int h, const uint8_t *rgb) {
    const size_t rs = (size_t)w * 3 + 1, raw = rs * (size_t)h; uint8_t *r = malloc(raw); uLongf zn = compressBound(raw); uint8_t *z = malloc(zn);
    if (!r || !z) { free(r); free(z); return 1; }
    for (int y = 0; y < h; y++) { r[(size_t)y * rs] = 0; memcpy(r + (size_t)y * rs + 1, rgb + (size_t)y * w * 3, (size_t)w * 3); }
    FILE *f = compress2(z, &zn, r, raw, 6) == Z_OK ? fopen(path, "wb") : nullptr;
    if (f) {
        static const uint8_t sig[8] = {137, 80, 78, 71, 13, 10, 26, 10}; fwrite(sig, 1, 8, f);
        const uint8_t ih[13] = {(uint8_t)(w >> 24), (uint8_t)(w >> 16), (uint8_t)(w >> 8), (uint8_t)w, (uint8_t)(h >> 24), (uint8_t)(h >> 16), (uint8_t)(h >> 8), (uint8_t)h, 8, 2, 0, 0, 0};
        png_chunk(f, "IHDR", ih, 13); png_chunk(f, "IDAT", z, (uint32_t)zn); png_chunk(f, "IEND", nullptr, 0);
    }
    free(r); free(z);
    return f ? fclose(f) != 0 : 1;
}
/* ufsm overlay <out.png> <ct-root> <ct-key> --plane z,y,x,h,w [--labels G1,G2,..] [--um 2.4] [--zoom 1] [--cache DIR]
   One CT plane (volume coordinates) beside the same plane of each label group (papyrus-labels output: key = --um level, placed by
   its ufsm.origin_zyx): void (0 inside the scroll) tinted blue, ignored (128) yellow, papyrus left grey. Panels left to right in
   --labels order after the plain CT, 4 white pixels apart; --zoom N repeats pixels. */
static int cmd_overlay(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: ufsm overlay <out.png> <ct-root> <ct-key> --plane z,y,x,h,w [--axis z|y|x] [--labels G1,G2,..] [--prob G1,G2,..] [--ids G1,..] [--um 2.4] [--zoom 1] [--cache DIR]\n"
                                  "  --labels: papyrus label groups (void blue, ignored yellow); --prob: uint8 probability groups (predict output, binary\n"
                                  "  labels), red with alpha p / 255; --ids: integer instance groups (any width; e.g. ufsm hercuvit ids), one hashed colour per id\n"
                                  "  over the CT (0 = none); each group's level is named by --um, its origin_zyx attribute places it. --axis y / x: the plane is\n"
                                  "  normal to y / x (h runs along z, w along x / y)\n"); return 2; }
    long long pz[5]; if (sscanf(opt(argc, argv, "--plane", ""), "%lld,%lld,%lld,%lld,%lld", pz, pz + 1, pz + 2, pz + 3, pz + 4) != 5 || pz[3] < 1 || pz[4] < 1) { fprintf(stderr, "overlay: --plane z,y,x,h,w\n"); return 2; }
    const int H = (int)pz[3], W = (int)pz[4], Z = atoi(opt(argc, argv, "--zoom", "1")) < 1 ? 1 : atoi(opt(argc, argv, "--zoom", "1"));
    char lv[32]; z3w_level_name(atof(opt(argc, argv, "--um", "2.4")), lv, sizeof lv);
    char lab[4096]; snprintf(lab, sizeof lab, "%s", opt(argc, argv, "--labels", "")); const char *names[16]; int nl = 0;
    for (char *s = lab, *tok; nl < 16 && (tok = strtok(s, ",")); s = nullptr) names[nl++] = tok;
    char prb[4096]; snprintf(prb, sizeof prb, "%s", opt(argc, argv, "--prob", "")); const int nlab = nl;   /* panels [0, nlab) labels, then probabilities */
    for (char *s = prb, *tok; nl < 16 && (tok = strtok(s, ",")); s = nullptr) names[nl++] = tok;
    char idg[4096]; snprintf(idg, sizeof idg, "%s", opt(argc, argv, "--ids", "")); const int nprob = nl;   /* then instance-id panels [nprob, nl) */
    for (char *s = idg, *tok; nl < 16 && (tok = strtok(s, ",")); s = nullptr) names[nl++] = tok;
    const size_t n2 = (size_t)H * W; uint8_t *ct = malloc(n2), *lb = malloc(n2 * (nl ? nl : 1)); uint32_t *idv = malloc(n2 * 4 * (nl - nprob ? nl - nprob : 1)); uint8_t *tb = malloc(n2 * 8);
    z3 *cz = z3_open(store_open(argv[3]), argv[4], opt(argc, argv, "--cache", nullptr)); if (!cz || !ct || !lb || !idv || !tb) { fprintf(stderr, "overlay: %s\n", z3_error()); return 1; }
    const char ax = opt(argc, argv, "--axis", "z")[0];
    const int64_t o[3] = {pz[0], pz[1], pz[2]}, n[3] = {ax == 'z' ? 1 : H, ax == 'z' ? H : ax == 'y' ? 1 : W, ax == 'x' ? 1 : W};
    if (z3_read(cz, o, n, ct, 8)) { fprintf(stderr, "overlay: %s\n", z3_error()); return 1; }
    for (int i = 0; i < nl; i++) {
        long long po[3] = {0, 0, 0}; char jp[1400]; snprintf(jp, sizeof jp, "%s/%s/zarr.json", names[i], lv); FILE *jf = fopen(jp, "rb"); char jb[8192] = {0};
        if (jf) { size_t nr = fread(jb, 1, sizeof jb - 1, jf); (void)nr; fclose(jf); }
        const char *q = strstr(jb, "\"origin_zyx\":"); if (q) sscanf(q + 13, " [%lld,%lld,%lld]", po, po + 1, po + 2);
        z3 *lz = i < nprob ? z3_open(store_open(names[i]), lv, nullptr) : z3_open_typed(store_open(names[i]), lv, nullptr); if (!lz) { fprintf(stderr, "overlay: %s: %s\n", names[i], z3_error()); return 1; }
        const int64_t lo[3] = {pz[0] - po[0], pz[1] - po[1], pz[2] - po[2]};
        if (i < nprob) { if (z3_read(lz, lo, n, lb + n2 * i, 8)) { fprintf(stderr, "overlay: %s: %s\n", names[i], z3_error()); return 1; } }
        else {   /* instance ids: any integer width, as uint32 */
            const int es = z3_meta_of(lz)->esize; if (es > 8 || z3_read(lz, lo, n, tb, 8)) { fprintf(stderr, "overlay: %s: %s\n", names[i], z3_error()); return 1; }
            uint32_t *dst = idv + n2 * (i - nprob);
            for (size_t k = 0; k < n2; k++) { uint64_t v = 0; memcpy(&v, tb + k * es, es); dst[k] = (uint32_t)v; }
        }
        z3_close(lz);
    }
    const int gap = 4, OW = (W * Z + gap) * (nl + 1) - gap, OH = H * Z; uint8_t *img = malloc((size_t)OW * OH * 3); if (!img) return 1;
    memset(img, 255, (size_t)OW * OH * 3);
    for (int p = 0; p <= nl; p++) for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        const uint8_t g = ct[(size_t)y * W + x]; int r = g, gg = g, b = g;
        if (p && p - 1 >= nprob) { const uint32_t v = idv[n2 * (p - 1 - nprob) + (size_t)y * W + x];   /* instance: hashed colour, alpha 0.6 */
                 if (v) { uint32_t hsh = v * 2654435761u; hsh ^= hsh >> 15; hsh *= 2246822519u; hsh ^= hsh >> 13;
                          const int cr = 60 + (int)(hsh & 0xff) * 195 / 255, cg = 60 + (int)((hsh >> 8) & 0xff) * 195 / 255, cb = 60 + (int)((hsh >> 16) & 0xff) * 195 / 255;
                          r = (2 * g + 3 * cr) / 5; gg = (2 * g + 3 * cg) / 5; b = (2 * g + 3 * cb) / 5; } }
        else if (p && p - 1 >= nlab) { const int a = lb[n2 * (p - 1) + (size_t)y * W + x];   /* probability: red over the CT */
                 r = (g * (255 - a) + 255 * a) / 255; gg = (g * (255 - a) + 30 * a) / 255; b = (g * (255 - a) + 20 * a) / 255; }
        else if (p) { const uint8_t v = lb[n2 * (p - 1) + (size_t)y * W + x];
                 if (g && v == 0) { r = g / 2; gg = g / 2 + 30; b = g / 2 + 120; } else if (v == 128) { r = g / 2 + 120; gg = g / 2 + 100; b = g / 2; } }
        for (int dy = 0; dy < Z; dy++) for (int dx = 0; dx < Z; dx++) {
            uint8_t *px = img + (((size_t)y * Z + dy) * OW + (size_t)p * (W * Z + gap) + (size_t)x * Z + dx) * 3;
            px[0] = (uint8_t)(r > 255 ? 255 : r); px[1] = (uint8_t)(gg > 255 ? 255 : gg); px[2] = (uint8_t)(b > 255 ? 255 : b);
        }
    }
    const int rc = png_write_rgb(argv[2], OW, OH, img);
    if (rc) fprintf(stderr, "overlay: cannot write %s\n", argv[2]); else fprintf(stderr, "overlay: %s (%d x %d, %d panels)\n", argv[2], OW, OH, nl + 1);
    z3_close(cz); free(ct); free(lb); free(idv); free(tb); free(img);
    return rc;
}

typedef struct {   /* ufsm render: one output tile's pixel rows over threads */
    const float *X, *Y, *Z; int w, h; double gs;   /* the vertex grid; output pixels per grid step */
    const uint8_t *vol, *ct; int64_t o[3], n[3]; int depth;
    uint8_t *img, *cimg; int ow, px0, px1, py0, py1; _Atomic int next;
} render_job;
static void *render_rows(void *arg) {
    render_job *J = arg; const int w = J->w, h = J->h;
    for (int py; (py = J->py0 + atomic_fetch_add(&J->next, 1)) < J->py1;) for (int px = J->px0; px < J->px1; px++) {
        const double gu = px / J->gs, gv = py / J->gs;
        int c0 = (int)gu, r0 = (int)gv; if (c0 >= w - 1) c0 = w - 2; if (r0 >= h - 1) r0 = h - 2; if (c0 < 0 || r0 < 0) continue;
        const double fu = gu - c0, fv = gv - r0;
        const size_t i00 = (size_t)r0 * w + c0, i01 = i00 + 1, i10 = i00 + w, i11 = i10 + 1;
        if (!(J->Z[i00] >= 0 && J->Z[i01] >= 0 && J->Z[i10] >= 0 && J->Z[i11] >= 0)) continue;   /* invalid (-1) or NaN corner */
        const float *A[3] = {J->Z, J->Y, J->X}; double P[3], du[3], dv[3];
        for (int d = 0; d < 3; d++) {
            const double a = A[d][i00], b = A[d][i01], c = A[d][i10], e = A[d][i11];
            P[d] = (1 - fv) * ((1 - fu) * a + fu * b) + fv * ((1 - fu) * c + fu * e);
            du[d] = (1 - fv) * (b - a) + fv * (e - c); dv[d] = (1 - fu) * (c - a) + fu * (e - b);
        }
        double N[3] = {du[1] * dv[2] - du[2] * dv[1], du[2] * dv[0] - du[0] * dv[2], du[0] * dv[1] - du[1] * dv[0]};
        const double nl = sqrt(N[0] * N[0] + N[1] * N[1] + N[2] * N[2]); if (nl < 1e-12) continue;
        for (int d = 0; d < 3; d++) N[d] /= nl;
        float m = 0;
        for (int t = -J->depth; t <= J->depth; t++) { const float v = trilin(J->vol, J->n, P[0] + t * N[0] - J->o[0], P[1] + t * N[1] - J->o[1], P[2] + t * N[2] - J->o[2]); if (v > m) m = v; }
        J->img[(size_t)py * J->ow + px] = (uint8_t)lroundf(m);
        if (J->cimg) J->cimg[(size_t)py * J->ow + px] = (uint8_t)lroundf(trilin(J->ct, J->n, P[0] - J->o[0], P[1] - J->o[1], P[2] - J->o[2]));
    }
    return nullptr;
}
/* ufsm blend: several predict runs of one volume whose tile grids are shifted against each other (each run a --box with its own origin,
   interiors anchored at that origin, period = window - 2 halo) -> one probability array: per voxel the logits averaged with weights
   prod_d sin^2(pi (q_d + 0.5) / S), q = the voxel's offset inside its run's tile interior, S the interior size. Runs shifted by S / 2
   on each axis (2^3 runs) give weights summing to 1 (sin^2 + cos^2), so every voxel is a smooth average over 8 windows that gives
   each window's interior edges ~0 weight (no seams); elsewhere the weights are normalised. Output: uint8 round(p 255), 512^3 shards. */
typedef struct {
    int n; z3 **in; int64_t (*org)[3]; int64_t (*shp)[3]; int S, sh; int64_t shape[3]; z3w *w; int nthreads;
    _Atomic int64_t next; int64_t nsh[3]; _Atomic int fail; _Atomic int64_t written;
} blend_job;
static void *blend_worker(void *arg) {
    blend_job *J = arg; const int sh = J->sh; const size_t n3 = (size_t)sh * sh * sh;
    uint8_t *buf = malloc(n3), *out = malloc(n3); float *acc = malloc(n3 * 4), *ws = malloc(n3 * 4);
    float lut[256]; for (int u = 0; u < 256; u++) { const double p = (u + 0.5) / 256.0; lut[u] = (float)log(p / (1 - p)); }
    float *wt[3]; for (int d = 0; d < 3; d++) wt[d] = malloc((size_t)J->S * 4);
    for (int d = 0; d < 3; d++) for (int q = 0; q < J->S; q++) { const double s = sin(M_PI * (q + 0.5) / J->S); wt[d][q] = (float)(s * s); }
    const int64_t tot = J->nsh[0] * J->nsh[1] * J->nsh[2];
    for (int64_t t; (t = atomic_fetch_add(&J->next, 1)) < tot && !atomic_load(&J->fail);) {
        const int64_t sz = t / (J->nsh[1] * J->nsh[2]), sy = t / J->nsh[2] % J->nsh[1], sx = t % J->nsh[2];
        const int64_t o[3] = {sz * sh, sy * sh, sx * sh};
        int64_t e[3]; for (int d = 0; d < 3; d++) e[d] = o[d] + sh < J->shape[d] ? sh : J->shape[d] - o[d];
        memset(acc, 0, n3 * 4); memset(ws, 0, n3 * 4); int any = 0;
        for (int k = 0; k < J->n; k++) {
            int64_t lo[3], hi[3], skip = 0;   /* the part of this shard the run covers, in volume coordinates */
            for (int d = 0; d < 3; d++) { lo[d] = o[d] > J->org[k][d] ? o[d] : J->org[k][d]; hi[d] = o[d] + e[d] < J->org[k][d] + J->shp[k][d] ? o[d] + e[d] : J->org[k][d] + J->shp[k][d]; if (hi[d] <= lo[d]) skip = 1; }
            if (skip) continue;
            const int64_t ro[3] = {lo[0] - J->org[k][0], lo[1] - J->org[k][1], lo[2] - J->org[k][2]}, rn[3] = {hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]};
            if (z3_read(J->in[k], ro, rn, buf, 1)) { fprintf(stderr, "blend: %s\n", z3_error()); atomic_store(&J->fail, 1); break; }
            for (int64_t z = 0; z < rn[0]; z++) {
                const float wz = wt[0][(ro[0] + z) % J->S];
                for (int64_t y = 0; y < rn[1]; y++) {
                    const float wzy = wz * wt[1][(ro[1] + y) % J->S];
                    const uint8_t *b = buf + ((size_t)z * rn[1] + y) * rn[2];
                    const size_t ob = ((size_t)(lo[0] - o[0] + z) * sh + (lo[1] - o[1] + y)) * sh + (lo[2] - o[2]);
                    for (int64_t x = 0; x < rn[2]; x++) {
                        const float w = wzy * wt[2][(ro[2] + x) % J->S];
                        if (b[x]) any = 1;
                        acc[ob + x] += w * lut[b[x]]; ws[ob + x] += w;
                    }
                }
            }
        }
        if (atomic_load(&J->fail)) break;
        if (!any) continue;   /* all runs 0 here: the output fill */
        for (size_t i = 0; i < n3; i++) { const float l = ws[i] > 0 ? acc[i] / ws[i] : -30.f; const int v = (int)lrintf(255.f / (1.f + expf(-l)));
                                          out[i] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v); }
        if (z3w_write_shard(J->w, sz, sy, sx, out, 1)) { fprintf(stderr, "blend: %s\n", z3w_error()); atomic_store(&J->fail, 1); break; }
        atomic_fetch_add(&J->written, 1);
    }
    free(buf); free(out); free(acc); free(ws); for (int d = 0; d < 3; d++) free(wt[d]);
    return nullptr;
}
static int cmd_blend(int argc, char **argv) {
    /* ufsm blend <out-group> <out-um> --shape Z,Y,X --interior S [--q 8] [--threads 8] <root> <key> <oz,oy,ox> [<root> <key> <oz,oy,ox> ...] */
    if (argc < 9) { fprintf(stderr, "usage: ufsm blend <out-group> <out-um> --shape Z,Y,X --interior S [--q 8] [--threads 8] <root> <key> <oz,oy,ox> ...\n"); return 2; }
    blend_job J; memset(&J, 0, sizeof J);
    long long s3[3]; if (sscanf(opt(argc, argv, "--shape", ""), "%lld,%lld,%lld", s3, s3 + 1, s3 + 2) != 3) { fprintf(stderr, "blend: --shape Z,Y,X\n"); return 2; }
    for (int d = 0; d < 3; d++) J.shape[d] = s3[d];
    J.S = atoi(opt(argc, argv, "--interior", "0")); J.sh = 512; J.nthreads = ufsm_threads(atoi(opt(argc, argv, "--threads", "0")), 8);
    const float q = (float)atof(opt(argc, argv, "--q", "8"));
    if (J.S <= 0) { fprintf(stderr, "blend: --interior S (window - 2 halo of the runs)\n"); return 2; }
    J.in = calloc(argc, sizeof *J.in); J.org = calloc(argc, sizeof *J.org); J.shp = calloc(argc, sizeof *J.shp);
    for (int i = 4; i + 2 < argc; i++) {
        if (argv[i][0] == '-') { i++; continue; }
        long long v[3]; if (sscanf(argv[i + 2], "%lld,%lld,%lld", v, v + 1, v + 2) != 3) { fprintf(stderr, "blend: bad origin %s\n", argv[i + 2]); return 2; }
        z3 *z = z3_open(store_open(argv[i]), argv[i + 1], nullptr); if (!z) { fprintf(stderr, "blend: %s\n", z3_error()); return 1; }
        J.in[J.n] = z; for (int d = 0; d < 3; d++) { J.org[J.n][d] = v[d]; J.shp[J.n][d] = z3_meta_of(z)->shape[d]; }
        fprintf(stderr, "blend: run %d %s/%s origin %lld,%lld,%lld shape %lldx%lldx%lld\n", J.n, argv[i], argv[i + 1], v[0], v[1], v[2],
                (long long)J.shp[J.n][0], (long long)J.shp[J.n][1], (long long)J.shp[J.n][2]);
        J.n++; i += 2;
    }
    if (!J.n) { fprintf(stderr, "blend: no runs\n"); return 2; }
    char lv[32], dir[1400], at[512]; z3w_level_name(atof(argv[3]), lv, sizeof lv); snprintf(dir, sizeof dir, "%s/%s", argv[2], lv);
    snprintf(at, sizeof at, "{\"ufsm\":{\"content\":\"blend of %d shifted predict runs (sin^2 weights over tile interiors of %d, logit average; uint8 = round(p*255))\"}}", J.n, J.S);
    J.w = z3w_create(dir, J.shape, J.sh, q, 0, at); if (!J.w) { fprintf(stderr, "blend: %s\n", z3w_error()); return 1; }
    for (int d = 0; d < 3; d++) J.nsh[d] = (J.shape[d] + J.sh - 1) / J.sh;
    const double t0 = now(); pthread_t th[64]; const int nt = J.nthreads < 1 ? 1 : J.nthreads > 64 ? 64 : J.nthreads;
    for (int t = 0; t < nt; t++) pthread_create(&th[t], nullptr, blend_worker, &J);
    for (int t = 0; t < nt; t++) pthread_join(th[t], nullptr);
    if (z3w_close(J.w) || atomic_load(&J.fail)) { fprintf(stderr, "blend: failed\n"); return 1; }
    fprintf(stderr, "blend: %lld of %lld shards written to %s in %.1fs\n", (long long)atomic_load(&J.written), (long long)(J.nsh[0] * J.nsh[1] * J.nsh[2]), dir, now() - t0);
    return 0;
}
/* render --fill: the share of surface pixels per class (stderr) and the cosmetic inpaint of the rest (--fill-inpaint L) */
static void render_fill_report(const char *name, uint8_t *img, uint8_t *cimg, uint8_t *flag, int ow, int oh, int inpaint, int pm, int nthreads) {
    size_t nin = 0;
    if (inpaint > 0) {
        uint8_t *const im[2] = {img, cimg}; const double t0 = now(); nin = img_pushpull_fill(im, 2, flag, ow, oh, inpaint, nthreads); const double t1 = now();
        if (pm && img_patchmatch_fill(im, 2, flag, ow, oh, nthreads)) fprintf(stderr, "render: %s: PatchMatch out of memory (push-pull kept)\n", name);
        fprintf(stderr, "render: %s inpaint: push-pull %.1fs, PatchMatch %.1fs\n", name, t1 - t0, pm ? now() - t1 : 0.0);
    }
    size_t ns = 0, non = 0, nfill = 0, nemp = 0, ngrid = 0, nnone = 0;   /* pixels of the original surface; grid-filled ones apart */
    for (size_t i = 0; i < (size_t)ow * oh; i++) { const uint8_t f = flag[i]; if (f == RFILL_NONE) { nnone++; continue; }
        if (f >= RFILL_GRID && f < RFILL_INPAINT) { ngrid++; continue; }
        ns++; if (f == RFILL_ON) non++; else if (f == RFILL_EMPTY) nemp++; else if (f != RFILL_INPAINT) nfill++; }
    const double d = ns ? 100.0 / ns : 0;
    fprintf(stderr, "render: %s fill: %zu surface pixels: void on the surface %.2f%%, filled along the normal %.2f%%, inpainted %.2f%% (incl. grid-hole pixels), still empty %.2f%%; "
                    "grid holes filled: %zu pixels (%.2f%% of the surface); no surface: %zu pixels\n",
            name, ns, (ns - non) * d, nfill * d, nin * d, nemp * d, ngrid, ngrid * d, nnone);
}
static int cmd_render(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm render <wraps-dir> <out-dir> --volume ROOT KEY [--ct ROOT KEY] [--cache DIR] [--scale 1] [--depth 8] [--tile 512] [--threads 16] [--crop x,y,w,h] [--joint 1]\n"
"         [--fill N] [--fill-ct 1] [--fill-prefer 0] [--fill-inpaint L] [--fill-pm 1] [--fill-grid V]\n"
"  --joint 1: all wraps (one grid size) in one chunk-ordered pass, shared chunks decoded once\n"
"  --fill N (needs --ct): a pixel whose CT on the surface is < --fill-ct (masked CT: 0 off the papyrus) takes the nearest papyrus\n"
"    sample along the surface normal within +-N voxels for the volume AND the CT (same offset); ties: --fill-prefer 1 / -1 the\n"
"    +- normal side, 0 the larger CT. Writes <wrap>_fill.tif: 0 on the surface, k filled from +k, 50 + k from -k, 250 inpainted,\n"
"    254 no surface, 255 still empty; prints the shares. --fill 0: flags only. --fill-inpaint L: push-pull over L levels (holes up\n"
"    to ~2^L pixels) of the still-empty pixels in both renders, then PatchMatch texture (--fill-pm 1, default; 0: push-pull only);\n"
"    cosmetic, off by default; flag 250\n"
"  --fill-grid V (with --fill): enclosed holes of the grid (invalid vertices, at most V per hole) bridged by interpolated vertices\n"
"    (push-pull + Laplace), rendered like the rest; their pixels' flags + 100\n"
                                     "  --crop: only that rectangle of each wrap's output pixels is rendered and written (e.g. a labelled region)\n"); return 2; }
    const int vi = arg_index(argc, argv, "--volume"), ci = arg_index(argc, argv, "--ct");
    int crop[4] = {0, 0, 0, 0}; const char *crs = opt(argc, argv, "--crop", nullptr);
    if (crs && (sscanf(crs, "%d,%d,%d,%d", &crop[0], &crop[1], &crop[2], &crop[3]) != 4 || crop[0] < 0 || crop[1] < 0 || crop[2] < 1 || crop[3] < 1)) { fprintf(stderr, "render: --crop x,y,w,h\n"); return 2; }
    if (vi < 0 || vi + 2 >= argc) { fprintf(stderr, "render: --volume ROOT KEY\n"); return 2; }
    const double scale = atof(opt(argc, argv, "--scale", "1")); const int depth = atoi(opt(argc, argv, "--depth", "8"));
    const int tile = atoi(opt(argc, argv, "--tile", "512")), nthreads = ufsm_threads(atoi(opt(argc, argv, "--threads", "0")), 16);
    const char *cache = opt(argc, argv, "--cache", nullptr);
    const int fill_n = atoi(opt(argc, argv, "--fill", "-1")), fill_inp = atoi(opt(argc, argv, "--fill-inpaint", "0"));
    const int fill_thr = atoi(opt(argc, argv, "--fill-ct", "1")), fill_pref = atoi(opt(argc, argv, "--fill-prefer", "0")), fill_pm = atoi(opt(argc, argv, "--fill-pm", "1"));
    const long fill_grid = atol(opt(argc, argv, "--fill-grid", "0"));
    if (fill_n > 49 || (fill_grid && fill_n < 0) || fill_inp < 0 || fill_inp > 8 || ((fill_n >= 0 || fill_inp) && (ci < 0 || ci + 2 >= argc)) || (fill_inp && fill_n < 0) || (fill_n >= 0 && getenv("UFSM_RENDER_TILED"))) {
        fprintf(stderr, "render: --fill N (0..49) needs --ct (and not UFSM_RENDER_TILED); --fill-inpaint L (1..8) needs --fill\n"); return 2; }
    store *vs = store_open(argv[vi + 1]), *cs = ci > 0 && ci + 2 < argc ? store_open(argv[ci + 1]) : nullptr;
    const int remote = (vs && !store_is_local(vs)) || (cs && !store_is_local(cs));
    z3 *vz = vs ? z3_open(vs, argv[vi + 2], cache) : nullptr, *cz = cs ? z3_open(cs, argv[ci + 2], cache) : nullptr;
    if (!vz || (ci > 0 && !cz)) { fprintf(stderr, "render: %s\n", z3_error()); return 1; }
    char cmd[1400]; snprintf(cmd, sizeof cmd, "mkdir -p '%s'", argv[3]); if (system(cmd)) return 1;
    DIR *dd = opendir(argv[2]); if (!dd) { fprintf(stderr, "render: cannot open %s\n", argv[2]); return 1; }
    struct dirent *de; int nw = 0;
    /* the wraps' union box (+- depth): when it fits UFSM_RENDER_MEM GB (default 8) the volume (and CT) are read once and every tile samples
       them (wraps are rings around the axis: per-tile boxes are nearly the whole cross-section, re-decoded per wrap: 127 wraps of a 3.9 cm
       slab took 226 s) */
    uint8_t *rvol = nullptr, *rct = nullptr; int64_t ro[3] = {0, 0, 0}, rn[3] = {0, 0, 0};
    /* otherwise the wraps render chunk-ordered (render_wrap_chunked: every chunk decoded once, ink and CT in one pass; UFSM_RENDER_MEM
       GB of decoded chunks); UFSM_RENDER_TILED=1: the former per-tile reads; UFSM_RENDER_CHUNKED=1: chunk-ordered even when the box fits */
    int chunked = 0; double cbudget = 0;
    {
        double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
        while ((de = readdir(dd))) {
            if (de->d_name[0] != 'w') continue;
            char p[1400]; int w, h; const char *ax[3] = {"z", "y", "x"};
            for (int d = 0; d < 3; d++) {
                snprintf(p, sizeof p, "%s/%s/%s.tif", argv[2], de->d_name, ax[d]); float *A = read_tif_f32(p, &w, &h); if (!A) continue;
                for (size_t i = 0; i < (size_t)w * h; i++) if (A[i] >= 0) { if (A[i] < lo[d]) lo[d] = A[i]; if (A[i] > hi[d]) hi[d] = A[i]; }
                free(A);
            }
        }
        rewinddir(dd);
        double budget = atof(getenv("UFSM_RENDER_MEM") ? getenv("UFSM_RENDER_MEM") : "8") * (1u << 30);
        { const double av = (double)ufsm_mem_available();   /* never above half of what this process may still allocate (cgroup aware) */
          if (av > 0 && budget > 0.5 * av) { fprintf(stderr, "render: UFSM_RENDER_MEM lowered to %.1f GB (half of the %.1f GB available)\n", 0.5 * av / (1u << 30), av / (1u << 30)); budget = 0.5 * av; } }
        chunked = getenv("UFSM_RENDER_TILED") ? 0 : getenv("UFSM_RENDER_CHUNKED") || fill_n >= 0 ? 1 : 0;   /* --fill: the chunk-ordered renderer only */
        cbudget = budget;
        if (lo[0] <= hi[0] && lo[1] <= hi[1] && lo[2] <= hi[2]) {
            for (int d = 0; d < 3; d++) { ro[d] = (int64_t)floor(lo[d]) - depth - 2; if (ro[d] < 0) ro[d] = 0; rn[d] = (int64_t)ceil(hi[d]) + depth + 3 - ro[d]; }
            const double nv = (double)rn[0] * rn[1] * rn[2];
            if (nv * (cz ? 2 : 1) <= budget && !chunked) {
                rvol = malloc((size_t)nv); rct = cz ? malloc((size_t)nv) : nullptr;
                if (!rvol || (cz && !rct) || z3_read(vz, ro, rn, rvol, 32) || (cz && z3_read(cz, ro, rn, rct, 32))) { fprintf(stderr, "render: read failed: %s\n", z3_error()); return 1; }
                fprintf(stderr, "render: the wraps' box %lldx%lldx%lld read once\n", (long long)rn[0], (long long)rn[1], (long long)rn[2]);
            }
        }
    }
    /* --joint 1: every wrap of the directory (same grid size and step, e.g. one wrap's recto and verso faces on one flat uv) in one
       chunk-ordered pass, so the chunks they share are decoded once; the outputs are those of separate renders, byte for byte */
    if (atoi(opt(argc, argv, "--joint", "0")) && !rvol && !getenv("UFSM_RENDER_TILED")) {
        enum { JMAX = 8 }; char *nm[JMAX]; float *JX[JMAX], *JY[JMAX], *JZ[JMAX]; uint8_t *VG[JMAX] = {0}; int nj = 0, jw = 0, jh = 0, ok = 1; double jstep = 0;
        while ((de = readdir(dd)) && ok) {
            if (de->d_name[0] != 'w') continue;
            if (nj == JMAX) { fprintf(stderr, "render: --joint takes at most %d wraps\n", JMAX); ok = 0; break; }
            char p[1400]; int w, h, w2, h2, w3, h3;
            snprintf(p, sizeof p, "%s/%s/x.tif", argv[2], de->d_name); float *X = read_tif_f32(p, &w, &h);
            snprintf(p, sizeof p, "%s/%s/y.tif", argv[2], de->d_name); float *Y = read_tif_f32(p, &w2, &h2);
            snprintf(p, sizeof p, "%s/%s/z.tif", argv[2], de->d_name); float *Z = read_tif_f32(p, &w3, &h3);
            if (!X || !Y || !Z || w != w2 || w != w3 || h != h2 || h != h3 || w < 2 || h < 2) { free(X); free(Y); free(Z); continue; }
            snprintf(p, sizeof p, "%s/%s/meta.json", argv[2], de->d_name);
            double step = 1; { char *b = tfx_meta_read(p); const char *q = b ? strstr(b, "\"scale\":") : nullptr; if (q && (q = strchr(q, '['))) { const double sc = atof(q + 1); if (sc > 0) step = 1 / sc; } free(b); }
            if (nj && (w != jw || h != jh || step != jstep)) { fprintf(stderr, "render: --joint: %s is %dx%d step %g, not %dx%d step %g\n", de->d_name, w, h, step, jw, jh, jstep); free(X); free(Y); free(Z); ok = 0; break; }
            jw = w; jh = h; jstep = step; nm[nj] = strdup(de->d_name); JX[nj] = X; JY[nj] = Y; JZ[nj] = Z;
            if (fill_grid > 0) { size_t nh, nbg; VG[nj] = malloc((size_t)w * h); const size_t nf = tifxyz_fill_holes(X, Y, Z, w, h, (size_t)fill_grid, VG[nj], &nh, &nbg);
                                 fprintf(stderr, "render: %s grid: %zu holes, %zu vertices interpolated, %zu holes over %ld vertices left\n", de->d_name, nh, nf, nbg, fill_grid); }
            nj++;
        }
        if (ok && nj) {
            const double gs = jstep * scale; const int ow = render_out_px(jw, gs), oh = render_out_px(jh, gs), nvol = cz ? 2 : 1;
            uint8_t *im[2 * JMAX] = {0}, *fl[JMAX] = {0};
            for (int k = 0; k < nj * nvol && ok; k++) if (!(im[k] = calloc((size_t)ow * oh, 1))) { fprintf(stderr, "render: out of memory\n"); ok = 0; }
            for (int k = 0; k < nj && ok && fill_n >= 0; k++) { if (!(fl[k] = malloc((size_t)ow * oh))) { fprintf(stderr, "render: out of memory\n"); ok = 0; } else memset(fl[k], RFILL_NONE, (size_t)ow * oh); }
            const render_fill RF = {fill_n, fill_thr, fill_pref, fl, fill_grid > 0 ? (const uint8_t *const *)VG : nullptr};
            z3 *zs[2] = {vz, cz}; const int dp[2] = {depth, 0};
            const size_t imgb = (size_t)ow * oh * nvol * nj, cb = cbudget > imgb + ((size_t)1 << 30) ? (size_t)cbudget - imgb : (size_t)1 << 30;
            const int io = remote ? (nthreads > 32 ? nthreads : 32) : nthreads;
            if (ok && render_wraps_chunked_fill(zs, nvol, dp, nj, (const float *const *)JX, (const float *const *)JY, (const float *const *)JZ, jw, jh, gs, im, ow, oh,
                                                crs ? crop : nullptr, cb, nthreads, io, 1, fill_n >= 0 ? &RF : nullptr)) { fprintf(stderr, "render: joint render failed\n"); ok = 0; }
            for (int k = 0; k < nj && ok && fill_n >= 0; k++) {
                render_fill_report(nm[k], im[k * nvol], im[k * nvol + 1], fl[k], ow, oh, fill_inp, fill_pm, nthreads);
                uint8_t *f = fl[k]; int fw = ow, fh = oh;
                if (crs) { uint8_t *c2 = img_crop_u8(f, ow, oh, crop[0], crop[1], crop[2], crop[3]); if (!c2) { ok = 0; break; } free(f); fl[k] = f = c2; fw = crop[2]; fh = crop[3]; }
                char p[1400]; snprintf(p, sizeof p, "%s/%s_fill.tif", argv[3], nm[k]);
                if (tiffw_write(p, fw, fh, 1, f, 1)) { fprintf(stderr, "render: cannot write %s\n", p); ok = 0; }
            }
            for (int k = 0; k < nj; k++) free(fl[k]);
            for (int k = 0; k < nj && ok; k++) for (int v = 0; v < nvol && ok; v++) {
                uint8_t *img = im[k * nvol + v]; int fw = ow, fh = oh;
                if (crs) { uint8_t *c2 = img_crop_u8(img, ow, oh, crop[0], crop[1], crop[2], crop[3]); if (!c2) { ok = 0; break; } free(img); im[k * nvol + v] = img = c2; fw = crop[2]; fh = crop[3]; }
                char p[1400]; snprintf(p, sizeof p, "%s/%s%s.tif", argv[3], nm[k], v ? "_ct" : "");
                if (tiffw_write(p, fw, fh, 1, img, 1)) { fprintf(stderr, "render: cannot write %s\n", p); ok = 0; }
            }
            for (int k = 0; k < nj * nvol; k++) free(im[k]);
        }
        for (int k = 0; k < nj; k++) { free(nm[k]); free(JX[k]); free(JY[k]); free(JZ[k]); free(VG[k]); }
        closedir(dd); free(rvol); free(rct);
        if (!ok) return 1;
        fprintf(stderr, "render: %d wraps written to %s (one pass)\n", nj, argv[3]);
        return 0;
    }
    while ((de = readdir(dd))) {
        if (de->d_name[0] != 'w') continue;
        char p[1400]; int w, h, w2, h2, w3, h3;
        snprintf(p, sizeof p, "%s/%s/x.tif", argv[2], de->d_name); float *X = read_tif_f32(p, &w, &h);
        snprintf(p, sizeof p, "%s/%s/y.tif", argv[2], de->d_name); float *Y = read_tif_f32(p, &w2, &h2);
        snprintf(p, sizeof p, "%s/%s/z.tif", argv[2], de->d_name); float *Z = read_tif_f32(p, &w3, &h3);
        if (!X || !Y || !Z || w != w2 || w != w3 || h != h2 || h != h3 || w < 2 || h < 2) { free(X); free(Y); free(Z); continue; }
        snprintf(p, sizeof p, "%s/%s/meta.json", argv[2], de->d_name);
        double step = 1; { char *b = tfx_meta_read(p); const char *q = b ? strstr(b, "\"scale\":") : nullptr; if (q && (q = strchr(q, '['))) { const double sc = atof(q + 1); if (sc > 0) step = 1 / sc; } free(b); }
        const double gs = step * scale;   /* output pixels per grid step */
        const int ow = render_out_px(w, gs), oh = render_out_px(h, gs);
        uint8_t *vgrid = nullptr;
        if (fill_grid > 0) { size_t nh, nbg; vgrid = malloc((size_t)w * h); const size_t nf = tifxyz_fill_holes(X, Y, Z, w, h, (size_t)fill_grid, vgrid, &nh, &nbg);
                             fprintf(stderr, "render: %s grid: %zu holes, %zu vertices interpolated, %zu holes over %ld vertices left\n", de->d_name, nh, nf, nbg, fill_grid); }
        uint8_t *img = calloc((size_t)ow * oh, 1), *cimg = cz ? calloc((size_t)ow * oh, 1) : nullptr, *flag = fill_n >= 0 ? malloc((size_t)ow * oh) : nullptr;
        if (!img || (cz && !cimg) || (fill_n >= 0 && !flag)) { fprintf(stderr, "render: out of memory\n"); return 1; }
        if (flag) memset(flag, RFILL_NONE, (size_t)ow * oh);
        if (!rvol && !getenv("UFSM_RENDER_TILED")) {
            z3 *zs[2] = {vz, cz}; const int dp[2] = {depth, 0}; uint8_t *const im[2] = {img, cimg};
            const size_t imgb = (size_t)ow * oh * (cz ? 2 : 1), cb = cbudget > imgb + ((size_t)1 << 30) ? (size_t)cbudget - imgb : (size_t)1 << 30;
            const int io = remote ? (nthreads > 32 ? nthreads : 32) : nthreads;
            const float *const Xs[1] = {X}, *const Ys[1] = {Y}, *const Zs[1] = {Z}; uint8_t *const fls[1] = {flag}; const uint8_t *const vgs[1] = {vgrid};
            const render_fill RF = {fill_n, fill_thr, fill_pref, fls, vgrid ? vgs : nullptr};
            if (render_wraps_chunked_fill(zs, cz ? 2 : 1, dp, 1, Xs, Ys, Zs, w, h, gs, im, ow, oh, crs ? crop : nullptr, cb, nthreads, io, 1, flag ? &RF : nullptr)) { fprintf(stderr, "render: %s: render failed\n", de->d_name); return 1; }
            if (flag) render_fill_report(de->d_name, img, cimg, flag, ow, oh, fill_inp, fill_pm, nthreads);
        } else
        /* output tiles: each reads only the volume around its own vertices (+- depth), so a wrap of any size renders in bounded memory */
        for (int ty = 0; ty < oh; ty += tile) for (int tx = 0; tx < ow; tx += tile) {
            const int py1 = ty + tile < oh ? ty + tile : oh, px1 = tx + tile < ow ? tx + tile : ow;
            if (crs && (px1 <= crop[0] || tx >= crop[0] + crop[2] || py1 <= crop[1] || ty >= crop[1] + crop[3])) continue;   /* outside --crop */
            const int r0 = (int)(ty / gs), r1 = (int)fmin(h - 1, ceil(py1 / gs) + 1), c0 = (int)(tx / gs), c1 = (int)fmin(w - 1, ceil(px1 / gs) + 1);
            double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
            for (int r = r0; r <= r1; r++) for (int c = c0; c <= c1; c++) {
                const size_t i = (size_t)r * w + c; if (Z[i] < 0) continue;
                const double v[3] = {Z[i], Y[i], X[i]}; for (int d = 0; d < 3; d++) { if (v[d] < lo[d]) lo[d] = v[d]; if (v[d] > hi[d]) hi[d] = v[d]; }
            }
            if (lo[0] > hi[0]) continue;
            render_job J = {X, Y, Z, w, h, gs, nullptr, nullptr, {0, 0, 0}, {0, 0, 0}, depth, img, cimg, ow, tx, px1, ty, py1, 0};
            if (rvol) {   /* the region read once */
                J.vol = rvol; J.ct = rct; for (int d = 0; d < 3; d++) { J.o[d] = ro[d]; J.n[d] = rn[d]; }
                pthread_t th[64]; const int nt = nthreads < 64 ? nthreads : 64;
                for (int t = 0; t < nt; t++) pthread_create(&th[t], nullptr, render_rows, &J);
                for (int t = 0; t < nt; t++) pthread_join(th[t], nullptr);
                continue;
            }
            for (int d = 0; d < 3; d++) { J.o[d] = (int64_t)floor(lo[d]) - depth - 2; if (J.o[d] < 0) J.o[d] = 0; J.n[d] = (int64_t)ceil(hi[d]) + depth + 3 - J.o[d]; }
            const size_t nv = (size_t)J.n[0] * J.n[1] * J.n[2];
            if (nv > ((size_t)4 << 30)) { fprintf(stderr, "render: %s tile %d,%d spans %lldx%lldx%lld voxels (a fold?): skipped; use a smaller --tile\n", de->d_name, ty, tx, (long long)J.n[0], (long long)J.n[1], (long long)J.n[2]); continue; }
            uint8_t *vol = malloc(nv), *ct = cz ? malloc(nv) : nullptr;
            if (!vol || (cz && !ct) || z3_read(vz, J.o, J.n, vol, 16) || (cz && z3_read(cz, J.o, J.n, ct, 16))) { fprintf(stderr, "render: read failed: %s\n", z3_error()); return 1; }
            J.vol = vol; J.ct = ct;
            pthread_t th[64]; const int nt = nthreads < 64 ? nthreads : 64;
            for (int t = 0; t < nt; t++) pthread_create(&th[t], nullptr, render_rows, &J);
            for (int t = 0; t < nt; t++) pthread_join(th[t], nullptr);
            free(vol); free(ct);
        }
        int fw = ow, fh = oh;
        if (crs) {   /* --crop: the rectangle only (0 where it reaches outside the render) */
            uint8_t *ci2 = img_crop_u8(img, ow, oh, crop[0], crop[1], crop[2], crop[3]), *cc2 = cimg ? img_crop_u8(cimg, ow, oh, crop[0], crop[1], crop[2], crop[3]) : nullptr;
            if (!ci2 || (cimg && !cc2)) { fprintf(stderr, "render: out of memory\n"); return 1; }
            free(img); free(cimg); img = ci2; cimg = cc2; fw = crop[2]; fh = crop[3];
            if (flag) { uint8_t *cf = img_crop_u8(flag, ow, oh, crop[0], crop[1], crop[2], crop[3]); if (!cf) return 1; free(flag); flag = cf; }
        }
        if (flag) { snprintf(p, sizeof p, "%s/%s_fill.tif", argv[3], de->d_name); if (tiffw_write(p, fw, fh, 1, flag, 1)) { fprintf(stderr, "render: cannot write %s\n", p); return 1; } free(flag); }
        snprintf(p, sizeof p, "%s/%s.tif", argv[3], de->d_name);
        if (tiffw_write(p, fw, fh, 1, img, 1)) { fprintf(stderr, "render: cannot write %s\n", p); return 1; }
        if (cimg) { snprintf(p, sizeof p, "%s/%s_ct.tif", argv[3], de->d_name); if (tiffw_write(p, fw, fh, 1, cimg, 1)) return 1; }
        free(X); free(Y); free(Z); free(img); free(cimg); free(vgrid); nw++;
    }
    closedir(dd); free(rvol); free(rct);
    fprintf(stderr, "render: %d wraps written to %s\n", nw, argv[3]);
    return 0;
}

static int cmd_sample(int argc, char **argv) {
    const char *path = argv[2];
    sample_cfg cfg = sample_cfg_default();
    cfg.P = atoi(opt(argc, argv, "--P", "128"));
    cfg.B = atoi(opt(argc, argv, "--B", "2"));
    cfg.seed = (uint64_t)atoll(opt(argc, argv, "--seed", "0"));
    cfg.nworkers = atoi(opt(argc, argv, "--workers", "4"));
    cfg.augment = !flag(argc, argv, "--noaug");
    cfg.shell = atoi(opt(argc, argv, "--shell", "0"));
    cfg.iso = atoi(opt(argc, argv, "--iso", "0")); cfg.iso_void = atoi(opt(argc, argv, "--iso-void", "0")); cfg.iso_pap = atoi(opt(argc, argv, "--iso-pap", "2")); cfg.iso_lift = atoi(opt(argc, argv, "--iso-lift", "0"));
    cfg.dist_sigma = (float)atof(opt(argc, argv, "--dist-head", "0"));
    int nb = atoi(opt(argc, argv, "--n", "4"));
    const char *out = opt(argc, argv, "--out", ".");
    /* --levels W0,W1,...: level weights as train; --side 2 --wind4 1: the unwrap2 phase / winding mod 4 targets; --dump 1: each
       window's phase (and w4 / comp) as raw P^3 bytes, <out>/phase_BB_I.raw (w4_BB_I.raw, wcomp_BB_I.raw): e.g. with UFSM_SAMPLE_AT
       the same corners from two sources files (a stored phase's sign, "phase_sign") */
    { const char *lv = opt(argc, argv, "--levels", nullptr); if (lv) { char *t = strdup(lv); int l = 0; memset(cfg.level_p, 0, sizeof cfg.level_p); for (char *q = strtok(t, ","); q && l < MAXLEV; q = strtok(nullptr, ",")) cfg.level_p[l++] = atof(q); free(t); } }
    cfg.side = atoi(opt(argc, argv, "--side", "0")); cfg.wind4 = atoi(opt(argc, argv, "--wind4", "0"));
    const int dump = atoi(opt(argc, argv, "--dump", "0"));
    /* --bench N: the trainer's batch formats for a CPU-only loader throughput test (no images; with UFSM_SAMPLER_NOPIN=1 no CUDA at all):
       N batches after a warm-up of --warm W, samples/s, cpu time and peak RSS. Train-like knobs: --Pz, --zfix 1, --x16 1 (fp16 batches),
       --gpu-input 1, --pack 1, --slots, --det, --min-phase, --wind4-tile / -min / -step, --symmetry-p, --axis-jitter */
    const int bench = atoi(opt(argc, argv, "--bench", "0"));
    if (bench > 0) {
        cfg.Pz = atoi(opt(argc, argv, "--Pz", "0")); if (cfg.Pz == cfg.P) cfg.Pz = 0;
        if (atoi(opt(argc, argv, "--zfix", "0"))) cfg.augment = 3;
        if (sample_aug_preset(&cfg, "none")) return 2;
        cfg.xfmt = atoi(opt(argc, argv, "--x16", "1")); cfg.gpu_input = atoi(opt(argc, argv, "--gpu-input", "1")); cfg.pack_tm = atoi(opt(argc, argv, "--pack", "1"));
        cfg.nbuf = atoi(opt(argc, argv, "--slots", "0")) > 0 ? atoi(opt(argc, argv, "--slots", "0")) : cfg.nworkers + 2;
        cfg.deterministic = atoi(opt(argc, argv, "--det", "1")); cfg.pmask = cfg.side == 2;
        cfg.min_phase = atof(opt(argc, argv, "--min-phase", cfg.wind4 ? "0.001" : "0"));
        cfg.wind4_step = atoi(opt(argc, argv, "--wind4-step", "80")); cfg.wind4_min = atoi(opt(argc, argv, "--wind4-min", "512"));
        { const int t = atoi(opt(argc, argv, "--wind4-tile", "64")); cfg.wind4_tile = t > 0 ? t : -1; }
        cfg.symmetry_p = (float)atof(opt(argc, argv, "--symmetry-p", "1")); cfg.axis_jitter = (float)atof(opt(argc, argv, "--axis-jitter", "0"));
        if (cfg.gpu_input && !sample_gpu_input_ok(&cfg)) { fprintf(stderr, "sample --bench: --gpu-input not available with these settings\n"); return 2; }
    }
    sources *S = sources_load(path);
    if (!S) return 1;
    fprintf(stderr, "%d sources\n", S->n);
    double t0 = now();
    sampler *sp = sampler_start(S, &cfg);
    if (!sp) return 1;
    if (bench > 0) {
        const int warm = atoi(opt(argc, argv, "--warm", "4"));
        struct rusage ru0, ru1; double ta = 0; size_t nv = (size_t)(cfg.Pz ? cfg.Pz : cfg.P) * cfg.P * cfg.P * cfg.B;
        for (int bi = 0; bi < warm + bench; bi++) {
            if (bi == warm) { ta = now(); getrusage(RUSAGE_SELF, &ru0); }
            batch *b = sampler_next(sp);
            if (!b) { fprintf(stderr, "sampler stopped\n"); return 1; }
            sampler_release(sp, b);
        }
        const double dt = now() - ta; getrusage(RUSAGE_SELF, &ru1);
        const double cpu = (ru1.ru_utime.tv_sec - ru0.ru_utime.tv_sec) + (ru1.ru_stime.tv_sec - ru0.ru_stime.tv_sec) + 1e-6 * ((ru1.ru_utime.tv_usec - ru0.ru_utime.tv_usec) + (ru1.ru_stime.tv_usec - ru0.ru_stime.tv_usec));
        uint64_t prod, rej; sampler_stats(sp, &prod, &rej);
        fprintf(stderr, "bench: %d batches of %d x %dx%dx%d in %.1f s: %.3f samples/s (%.0f Mvox/s), %.1f cpu s per sample, %d workers; "
                "%llu produced, %llu rejected; peak RSS %.2f GB; start %.1f s\n", bench, cfg.B, cfg.Pz ? cfg.Pz : cfg.P, cfg.P, cfg.P, dt,
                bench * cfg.B / dt, bench * (double)nv / dt / 1e6, cpu / (bench * cfg.B), cfg.nworkers, (unsigned long long)prod, (unsigned long long)rej,
                ru1.ru_maxrss / 1e6, ta - t0);
        sampler_prof_print(sp);
        sampler_stop(sp); sources_free(S);
        return 0;
    }
    int P = cfg.P;
    size_t p3 = (size_t)P * P * P;
    /* montage: one row per patch: CT | recto | sheet | mask | radial-y | radial-x (mid z slice) */
    int cols = 6;
    uint8_t *img = malloc((size_t)cols * P * P);
    for (int bi = 0; bi < nb; bi++) {
        batch *b = sampler_next(sp);
        if (!b) { fprintf(stderr, "sampler stopped\n"); break; }
        for (int i = 0; i < cfg.B; i++) {
            const float *x = b->x + (size_t)i * 4 * p3 + (size_t)(P / 2) * P * P;
            for (int c = 0; c < cols; c++)
                for (int y = 0; y < P; y++)
                    for (int xx = 0; xx < P; xx++) {
                        size_t k = (size_t)y * P + xx;
                        double v;
                        if (c == 0) v = 128 + 40 * x[k];
                        else if (c == 1) v = b->t[(size_t)i * NCH * p3 + 0 * p3 + (size_t)(P / 2) * P * P + k];
                        else if (c == 2) v = b->t[(size_t)i * NCH * p3 + 1 * p3 + (size_t)(P / 2) * P * P + k];
                        else if (c == 3) v = 255 * b->m[(size_t)i * p3 + (size_t)(P / 2) * P * P + k];
                        else v = 128 + 127 * x[(size_t)(c - 2) * p3 + k];
                        img[(size_t)y * cols * P + c * P + xx] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
                    }
            char fn[1200];
            snprintf(fn, sizeof fn, "%s/patch_%02d_%d.pgm", out, bi, i);
            write_pgm(fn, img, (int64_t)cols * P, P);
            if (dump && cfg.side) {
                const char *nm[3] = {"phase", "w4", "wcomp"}; uint8_t *bf[3] = {b->side, cfg.wind4 ? b->w4 : nullptr, cfg.wind4 ? b->wcomp : nullptr};
                for (int k = 0; k < 3; k++) if (bf[k]) {
                    snprintf(fn, sizeof fn, "%s/%s_%02d_%d.raw", out, nm[k], bi, i); FILE *f = fopen(fn, "wb");
                    if (!f || fwrite(bf[k] + (size_t)i * p3, 1, p3, f) != p3) { fprintf(stderr, "sample: cannot write %s\n", fn); if (f) fclose(f); return 1; }
                    fclose(f);
                }
                snprintf(fn, sizeof fn, "%s/patch_%02d_%d.pgm", out, bi, i);
            }
            fprintf(stderr, "%s: src %s level %d corner %lld,%lld,%lld w=[%d,%d]\n", fn, S->src[b->src[i]].name, b->level[i],
                    (long long)b->corner[i][0], (long long)b->corner[i][1], (long long)b->corner[i][2], b->w[i * NCH], b->w[i * NCH + 1]);
            if (dump) {   /* label check: the montage as JPEG, and the input CT (z-scored) under the recto / ink target's classes over the
                             whole window: a misplaced or rescaled label set puts its positives on void (CT near the window's low end) */
                uint8_t *rgb = malloc((size_t)cols * P * P * 3);
                if (rgb) { for (size_t k = 0; k < (size_t)cols * P * P; k++) rgb[3 * k] = rgb[3 * k + 1] = rgb[3 * k + 2] = img[k];
                           char jn[1200]; snprintf(jn, sizeof jn, "%s/patch_%02d_%d.jpg", out, bi, i);
                           if (jpeg_write_rgb(jn, cols * P, P, rgb, 90)) fprintf(stderr, "sample: cannot write %s\n", jn);
                           free(rgb); }
                const float *x0 = b->x + (size_t)i * 4 * p3; const uint8_t *t0 = b->t + (size_t)i * NCH * p3, *m0 = b->m + (size_t)i * p3;
                double s[3] = {0, 0, 0}, all = 0; size_t c[3] = {0, 0, 0};   /* (t = p * 255; m = 0: ignored) */
                for (size_t k = 0; k < p3; k++) { const int cl = !m0[k] ? 2 : t0[k] >= 128 ? 0 : 1; s[cl] += x0[k]; c[cl]++; all += x0[k]; }
                fprintf(stderr, "  label check: ink %zu (CT z %.2f), no-ink %zu (CT z %.2f), unknown %zu (CT z %.2f), window CT z %.2f\n",
                        c[0], c[0] ? s[0] / c[0] : 0., c[1], c[1] ? s[1] / c[1] : 0., c[2], c[2] ? s[2] / c[2] : 0., all / p3);
            }
        }
        sampler_release(sp, b);
    }
    uint64_t prod, rej;
    sampler_stats(sp, &prod, &rej);
    fprintf(stderr, "%llu patches produced, %llu rejected, %.1fs\n", (unsigned long long)prod, (unsigned long long)rej, now() - t0);
    sampler_stop(sp);
    sources_free(S);
    free(img);
    return 0;
}

int cmd_repack(int argc, char **argv);   /* src/repack.c */
int cmd_fit_join(int argc, char **argv);   /* src/fitjoin.c */
int cmd_wrap_check(int argc, char **argv);   /* src/wrapcheck.c */
int cmd_fields_oracle(int argc, char **argv);   /* src/fieldsoracle.c */
int cmd_flatten(int argc, char **argv);         /* src/flatten.c */
int cmd_faces(int argc, char **argv);           /* src/faces_cli.c */
int cmd_flatten_warp(int argc, char **argv);
int cmd_strip(int argc, char **argv);           /* src/strip.c */
int cmd_strip_flatten(int argc, char **argv);
int cmd_strip_mosaic(int argc, char **argv);
int cmd_strip_crop(int argc, char **argv);
int cmd_fiber_check(int argc, char **argv);    /* src/fibercheck.c */
int cmd_fiber_export(int argc, char **argv);   /* src/fibercheck.c */
int cmd_count_field(int argc, char **argv);   /* src/countfield.c */
int cmd_hires_phase(int argc, char **argv);   /* src/hiresphase.c */
int cmd_phase_gate(int argc, char **argv);   /* src/phasegate.c */
int cmd_patch_raster(int argc, char **argv);   /* src/patchraster.c */
int cmd_ray_check(int argc, char **argv);   /* src/raycheck.c */
int cmd_vwt_sample(int argc, char **argv);   /* src/vwt_sample.c */
int cmd_hires_count(int argc, char **argv);
int cmd_spiral_fit(int argc, char **argv);    /* src/countfield.c */
int cmd_cf_compare(int argc, char **argv);    /* src/countfield.c */
int cmd_phase_packing(int argc, char **argv);   /* src/packmap.c */
int cmd_wbench_stats(int argc, char **argv);   /* src/wbench.c */
int cmd_ckpt_avg(int argc, char **argv);   /* src/wbench.c */
int cmd_phase_conflict(int argc, char **argv);   /* src/packmap.c */
int cmd_reg(int argc, char **argv);   /* src/reg_cli.c */
int cmd_sr_pairs(int argc, char **argv);   /* src/srpairs.c */
int main(int argc, char **argv) {
    if (argc >= 2 && !strcmp(argv[1], "sysinfo")) return cmd_sysinfo(argc, argv);
    if (argc == 2 && !strcmp(argv[1], "train")) return cmd_train(argc, argv);
    if (argc < 3) return usage();
    const char *cmd = argv[1];
    if (!strcmp(cmd, "sample")) return cmd_sample(argc, argv);
    if (!strcmp(cmd, "prefetch")) {   /* ufsm prefetch <sources.json> [--levels 1] [--threads 32] [--fraction 1.0] [--P 128] [--dry 0] */
        if (argc < 3) { fprintf(stderr, "usage: ufsm prefetch <sources.json> [--levels L] [--threads N] [--fraction F] [--P window] [--dry 1 (count only)]\n"); return 2; }
        sources *S = sources_load(argv[2]); if (!S) return 1;
        return sources_prefetch(S, atoi(opt(argc, argv, "--levels", "1")), ufsm_threads(atoi(opt(argc, argv, "--threads", "0")), 32), atof(opt(argc, argv, "--fraction", "1.0")),
                                atoi(opt(argc, argv, "--P", "128")), atoi(opt(argc, argv, "--dry", "0")));
    }
    if (!strcmp(cmd, "ingest-labels")) return cmd_ingest_labels(argc, argv);
    if (!strcmp(cmd, "z3-recode")) return cmd_z3_recode(argc, argv);
    if (!strcmp(cmd, "ink-blend")) return cmd_ink_blend(argc, argv);
    if (!strcmp(cmd, "ink-seam")) return cmd_ink_seam(argc, argv);
    if (!strcmp(cmd, "ingest-kaggle")) return cmd_ingest_kaggle(argc, argv);
    if (!strcmp(cmd, "ingest-mesh")) return cmd_ingest_mesh(argc, argv);
    if (!strcmp(cmd, "ls")) return cmd_ls(argc, argv);
    if (!strcmp(cmd, "mirror")) return cmd_mirror(argc, argv);     /* src/scanprep.c */
    if (!strcmp(cmd, "scanprep")) return cmd_scanprep(argc, argv);
    if (!strcmp(cmd, "ingest-zip")) return cmd_ingest_zip(argc, argv);
    if (!strcmp(cmd, "raster")) return cmd_raster(argc, argv);
    if (!strcmp(cmd, "train")) return cmd_train(argc, argv);
    if (!strcmp(cmd, "predict")) return cmd_predict(argc, argv);
    if (!strcmp(cmd, "eval")) return cmd_eval(argc, argv);
    if (!strcmp(cmd, "band")) return cmd_band(argc, argv);
    if (!strcmp(cmd, "fit")) return cmd_fit(argc, argv);
    if (!strcmp(cmd, "fit-join")) return cmd_fit_join(argc, argv);
    if (!strcmp(cmd, "wrap-check")) return cmd_wrap_check(argc, argv);
    if (!strcmp(cmd, "fields-oracle")) return cmd_fields_oracle(argc, argv);
    if (!strcmp(cmd, "flatten")) return cmd_flatten(argc, argv);
    if (!strcmp(cmd, "faces")) return cmd_faces(argc, argv);
    if (!strcmp(cmd, "flatten-warp")) return cmd_flatten_warp(argc, argv);
    if (!strcmp(cmd, "strip")) return cmd_strip(argc, argv);
    if (!strcmp(cmd, "strip-flatten")) return cmd_strip_flatten(argc, argv);
    if (!strcmp(cmd, "strip-mosaic")) return cmd_strip_mosaic(argc, argv);
    if (!strcmp(cmd, "strip-crop")) return cmd_strip_crop(argc, argv);
    if (!strcmp(cmd, "fiber-check")) return cmd_fiber_check(argc, argv);
    if (!strcmp(cmd, "fiber-export")) return cmd_fiber_export(argc, argv);
    if (!strcmp(cmd, "count-field")) return cmd_count_field(argc, argv);
    if (!strcmp(cmd, "hires-phase")) return cmd_hires_phase(argc, argv);
    if (!strcmp(cmd, "phase-gate")) return cmd_phase_gate(argc, argv);
    if (!strcmp(cmd, "patch-raster")) return cmd_patch_raster(argc, argv);
    if (!strcmp(cmd, "ray-check")) return cmd_ray_check(argc, argv);
    if (!strcmp(cmd, "vwt-sample")) return cmd_vwt_sample(argc, argv);
    if (!strcmp(cmd, "hires-count")) return cmd_hires_count(argc, argv);
    if (!strcmp(cmd, "spiral-fit")) return cmd_spiral_fit(argc, argv);
    if (!strcmp(cmd, "cf-compare")) return cmd_cf_compare(argc, argv);
    if (!strcmp(cmd, "phase-packing")) return cmd_phase_packing(argc, argv);
    if (!strcmp(cmd, "wbench-stats")) return cmd_wbench_stats(argc, argv);
    if (!strcmp(cmd, "ckpt-avg")) return cmd_ckpt_avg(argc, argv);
    if (!strcmp(cmd, "phase-conflict")) return cmd_phase_conflict(argc, argv);
    if (!strcmp(cmd, "reg")) return cmd_reg(argc, argv);
    if (!strcmp(cmd, "sr-pairs")) return cmd_sr_pairs(argc, argv);
    if (!strcmp(cmd, "convert-unwrap2")) return cmd_convert_unwrap2(argc, argv);
    if (!strcmp(cmd, "repack")) return cmd_repack(argc, argv);
    if (!strcmp(cmd, "phase-raster")) return cmd_phase_raster(argc, argv);
    if (!strcmp(cmd, "wraps")) return cmd_wraps(argc, argv);
    if (!strcmp(cmd, "stitch")) return cmd_stitch(argc, argv);
    if (!strcmp(cmd, "coreline") && argc > 2 && !strcmp(argv[2], "check")) return cmd_coreline_check(argc, argv);
    if (!strcmp(cmd, "coreline")) return cmd_coreline(argc, argv);
    if (!strcmp(cmd, "render-sheets")) return cmd_render_sheets(argc, argv);
    if (!strcmp(cmd, "overlay-red")) return cmd_overlay_red(argc, argv);
    if (!strcmp(cmd, "tifxyz-scale")) return cmd_tifxyz_scale(argc, argv);
    if (!strcmp(cmd, "ink-compare")) return cmd_ink_compare(argc, argv);
    if (!strcmp(cmd, "jpeg-fit")) return cmd_jpeg_fit(argc, argv);
    if (!strcmp(cmd, "ink-ext")) return cmd_ink_ext(argc, argv);
    if (!strcmp(cmd, "ink-splat")) return cmd_ink_splat(argc, argv);
    if (!strcmp(cmd, "ink-score")) return cmd_ink_score(argc, argv);
    if (!strcmp(cmd, "ink-ensemble")) return cmd_ink_ensemble(argc, argv);
    if (!strcmp(cmd, "wl-dump")) return cmd_wl_dump(argc, argv);
    if (!strcmp(cmd, "dinovol-embed")) return cmd_dinovol_embed(argc, argv);
    if (!strcmp(cmd, "hercuvit")) return cmd_hercuvit(argc, argv);
    if (!strcmp(cmd, "villawind")) return cmd_villawind(argc, argv);
    if (!strcmp(cmd, "vwt-train")) return cmd_vwt_train(argc, argv);
    if (!strcmp(cmd, "vwt-tile")) return cmd_vwt_tile(argc, argv);
    if (!strcmp(cmd, "vwt-infer")) return cmd_vwt_infer(argc, argv);
    if (!strcmp(cmd, "wtrunk-bench")) return cmd_wtrunk_bench(argc, argv);
    if (!strcmp(cmd, "vwt-stitch")) return cmd_vwt_stitch(argc, argv);
    if (!strcmp(cmd, "phase-vis")) return cmd_phase_vis(argc, argv);
    if (!strcmp(cmd, "ink3d-ext")) return cmd_ink3d_ext(argc, argv);
    if (!strcmp(cmd, "vol-cmp")) return cmd_vol_cmp(argc, argv);
    if (!strcmp(cmd, "dinovol-eval")) return cmd_dinovol_eval(argc, argv);
    if (!strcmp(cmd, "dinovol-read")) return cmd_dinovol_read(argc, argv);
    if (!strcmp(cmd, "emb-pca")) return cmd_emb_pca(argc, argv);
    if (!strcmp(cmd, "emb-calib")) return cmd_emb_calib(argc, argv);
    if (!strcmp(cmd, "dinovol-calib")) return cmd_dinovol_calib(argc, argv);
    if (!strcmp(cmd, "dino-headfit")) return cmd_dino_headfit(argc, argv);
    if (!strcmp(cmd, "surf-eval")) return cmd_surf_eval(argc, argv);
    if (!strcmp(cmd, "dino-train")) return cmd_dino_train(argc, argv);
    if (!strcmp(cmd, "dino-embed")) return cmd_dino_embed(argc, argv);
    if (!strcmp(cmd, "dino-seamprobe")) return cmd_dino_seamprobe(argc, argv);
    if (!strcmp(cmd, "halves")) return cmd_dino_halves(argc, argv);
    if (!strcmp(cmd, "dino-boxes")) return cmd_dino_boxes(argc, argv);
    if (!strcmp(cmd, "tifxyz-map")) return cmd_tifxyz_map(argc, argv);
    if (!strcmp(cmd, "tifxyz-orient")) return cmd_tifxyz_orient(argc, argv);
    if (!strcmp(cmd, "tifxyz-bbox")) return cmd_tifxyz_bbox(argc, argv);
    if (!strcmp(cmd, "tifxyz-shards")) return cmd_tifxyz_shards(argc, argv);
    if (!strcmp(cmd, "png2jpg")) return cmd_png2jpg(argc, argv);
    if (!strcmp(cmd, "tifxyz-sfc")) return cmd_tifxyz_sfc(argc, argv);
    if (!strcmp(cmd, "sheet-cut")) return cmd_sheet_cut(argc, argv);
    if (!strcmp(cmd, "phase-score")) return cmd_phase_score(argc, argv);
    if (!strcmp(cmd, "render")) return cmd_render(argc, argv);
    if (!strcmp(cmd, "blend")) return cmd_blend(argc, argv);
    if (!strcmp(cmd, "downsample")) return cmd_downsample(argc, argv);
    if (!strcmp(cmd, "side-votes") || !strcmp(cmd, "codes-merge")) return cmd_side_votes(argc, argv);
    if (!strcmp(cmd, "wind-assign")) return cmd_wind_assign(argc, argv);
    if (!strcmp(cmd, "wind-seed")) return cmd_wind_seed(argc, argv);
    if (!strcmp(cmd, "phase-check")) return cmd_phase_check(argc, argv);
    if (!strcmp(cmd, "phase-fuse") || !strcmp(cmd, "recto-fuse")) return cmd_fuse(argc, argv);
    if (!strcmp(cmd, "ct-hist")) return cmd_ct_hist(argc, argv);
    if (!strcmp(cmd, "restore-eval")) return cmd_restore_eval(argc, argv);
    if (!strcmp(cmd, "restore-stats")) return cmd_restore_stats(argc, argv);
    if (!strcmp(cmd, "wrapdist")) return cmd_wrapdist(argc, argv);
    if (!strcmp(cmd, "wrapswitch")) return cmd_wrapswitch(argc, argv);
    if (!strcmp(cmd, "surf-snap")) return cmd_surfsnap(argc, argv);
    if (!strcmp(cmd, "refine")) return cmd_refine(argc, argv);
    if (!strcmp(cmd, "surf-stats")) return cmd_surf_stats(argc, argv);
    if (!strcmp(cmd, "self-gate")) return cmd_self_gate(argc, argv);
    if (!strcmp(cmd, "ct-fields")) return cmd_ct_fields(argc, argv);
    if (!strcmp(cmd, "fit-compare")) return cmd_fit_compare(argc, argv);
    if (!strcmp(cmd, "l1-bench")) return cmd_l1_bench(argc, argv);
    if (!strcmp(cmd, "field-blend")) return cmd_field_blend(argc, argv);
    if (!strcmp(cmd, "gap-compare")) return cmd_gap_compare(argc, argv);
    if (!strcmp(cmd, "gap-density")) return cmd_gap_density(argc, argv);
    if (!strcmp(cmd, "void-pool")) return cmd_void_pool(argc, argv);
    if (!strcmp(cmd, "papyrus-labels")) return cmd_papyrus_labels(argc, argv);
    if (!strcmp(cmd, "overlay")) return cmd_overlay(argc, argv);
    if (!strcmp(cmd, "wind-score")) return cmd_wind_score(argc, argv);
    if (!strcmp(cmd, "store-bench")) { int cmd_store_bench(int, char **); return cmd_store_bench(argc, argv); }
    if (!strcmp(cmd, "phase-pack") || !strcmp(cmd, "phase-unpack") || !strcmp(cmd, "phase-cmp") || !strcmp(cmd, "phase-drop")) { int cmd_phase_pack(int, char **); return cmd_phase_pack(argc, argv); }
    if (argc < 4) return usage();
    const char *root = argv[2], *key = argv[3];
    const char *cache = opt(argc, argv, "--cache", nullptr);
    int threads = atoi(opt(argc, argv, "--threads", "0"));
    store *s = store_open(root);
    if (!strcmp(cmd, "regions")) return cmd_regions(s, key);
    if (!strcmp(cmd, "axis")) { if (argc < 5) return usage(); return cmd_axis(s, key, argv[4], cache); }
    if (!strcmp(cmd, "info")) {
        z3_level lv[16];
        int nl = z3_group_levels(s, key, lv, 16);
        if (nl > 0) {
            printf("group %s: %d levels\n", key, nl);
            for (int i = 0; i < nl; i++) {
                char ak[2048];
                snprintf(ak, sizeof ak, "%s/%s", key, lv[i].path);
                z3 *z = z3_open(s, ak, cache);
                if (!z) { printf("  %-10s %.4g um  (unreadable: %s)\n", lv[i].path, lv[i].um, z3_error()); continue; }
                const z3_meta *m = z3_meta_of(z);
                printf("  %-10s %.4g um  shape %lld x %lld x %lld  shard %d chunk %d  %s q=%g%s\n", lv[i].path, lv[i].um,
                       (long long)m->shape[0], (long long)m->shape[1], (long long)m->shape[2], m->shard[0], m->chunk[0],
                       m->q >= 0 ? "volcomp" : m->raw ? "raw" : "?", m->q, m->zstd ? "+zstd" : "");
                z3_close(z);
            }
            return 0;
        }
        z3 *z = z3_open(s, key, cache);
        if (!z) { fprintf(stderr, "%s\n", z3_error()); return 1; }
        const z3_meta *m = z3_meta_of(z);
        printf("array %s\n  shape %lld x %lld x %lld  shard %d x %d x %d  chunk %d x %d x %d  sep '%c'\n  codec %s q=%g%s  scale %.4g um\n",
               key, (long long)m->shape[0], (long long)m->shape[1], (long long)m->shape[2], m->shard[0], m->shard[1], m->shard[2],
               m->chunk[0], m->chunk[1], m->chunk[2], m->sep, m->q >= 0 ? "volcomp" : m->raw ? "raw" : "?", m->q,
               m->zstd ? "+zstd" : "", m->scale_um);
        z3_close(z);
        return 0;
    }
    if (!strcmp(cmd, "read") || !strcmp(cmd, "slice")) {
        z3 *z = !strcmp(cmd, "read") ? z3_open_typed(s, key, cache) : z3_open(s, key, cache);   /* read: typed arrays as raw elements */
        if (!z) { fprintf(stderr, "%s\n", z3_error()); return 1; }
        const z3_meta *m = z3_meta_of(z);
        int64_t o[3], n[3];
        const char *out;
        if (!strcmp(cmd, "read")) {
            if (argc < 11) return usage();
            for (int i = 0; i < 3; i++) { o[i] = atoll(argv[4 + i]); n[i] = atoll(argv[7 + i]); }
            out = argv[10];
        } else {
            if (argc < 6) return usage();
            o[0] = atoll(argv[4]); o[1] = o[2] = 0;
            n[0] = 1; n[1] = m->shape[1]; n[2] = m->shape[2];
            out = argv[5];
        }
        size_t nv = (size_t)n[0] * n[1] * n[2]; const int es = m->esize, dt_ = m->dtype;
        uint8_t *buf = malloc(nv * es);
        double t0 = now();
        int rc = z3_read(z, o, n, buf, threads);
        double dt = now() - t0;
        if (rc) { fprintf(stderr, "read failed: %s\n", z3_error()); return 1; }
        size_t nz = 0; double sum = 0;
        for (size_t i = 0; i < nv; i++) {
            double v;
            switch (dt_) {
            case 1: { uint16_t u; memcpy(&u, buf + 2 * i, 2); v = u; break; }
            case 2: { int16_t u; memcpy(&u, buf + 2 * i, 2); v = u; break; }
            case 3: { uint32_t u; memcpy(&u, buf + 4 * i, 4); v = u; break; }
            case 4: { int32_t u; memcpy(&u, buf + 4 * i, 4); v = u; break; }
            case 5: { float u; memcpy(&u, buf + 4 * i, 4); v = u; break; }
            default: v = buf[i];
            }
            nz += v != 0; sum += v;
        }
        fprintf(stderr, "read %zu voxels in %.2fs (%.1f Mvox/s), nonzero %.1f%%, mean %.2f\n", nv, dt, nv / dt / 1e6,
                100.0 * nz / nv, sum / nv);
        if (!strcmp(cmd, "slice")) rc = write_pgm(out, buf, n[2], n[1]);
        else { FILE *f = fopen(out, "wb"); rc = f ? (fwrite(buf, es, nv, f) == nv ? 0 : -1) : -1; if (f) fclose(f); }
        free(buf);
        z3_close(z);
        return rc ? 1 : 0;
    }
    return usage();
}
