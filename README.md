# ufsm — ultra fast scroll model

C23 + CUDA, no PyTorch, no cuDNN. Trains a ~1 M-parameter 3-D convnet for the recto papyrus face of
Vesuvius Challenge scrolls from the released ground truth only (HuggingFace `scrollprize/datasets` and
the AWS open-data bucket), after re-exporting that data into volcomp volumes and surfcomp surfaces.
Design, data inventory and milestones: [DESIGN.md](DESIGN.md).

```sh
make && make test        # gcc -std=c23 + nvcc; needs libcurl, libzstd, libblosc, zlib, OpenSSL libcrypto, CUDA 13 (sm_120)
make ARCH=86             # Ampere (RTX 3090) into build86/: software MX-fp4 / fp8 (bit-exact conversions, emulated block-scaled MMAs)
                         # (on by default there; UFSM_MX_EMU=0: fp8 / fp4 as bf16, 16-bit storage); PTX=1: compute_86 PTX
                         # only (build86p/), so a newer GPU runs the sm_86 code (tests); BDIR=build puts it in build/
```

## Commands

```sh
ufsm info|read|slice <root> <key> ...                 # inspect / read volcomp zarr v3 pyramids (local or HTTPS, cached)
ufsm ls <root> <key>                                   # list HF / S3 / local trees
ufsm ingest-zip  labels.zip <zarr-name> <out> --um U   # HF label zarr (from the archive) -> volcomp label pyramid
ufsm ingest-labels <root> <zarr-key> <out> --um U      # same, chunk by chunk over HTTP (slow: HF rate limits)
ufsm ingest-kaggle <hf-root> <out>                     # Kaggle cubes -> images.zarr + labels.zarr
ufsm ingest-mesh <s3-root> <tifxyz-key> <out.sfc>      # segment mesh -> surfcomp
ufsm raster <out> --shape Z,Y,X --um U meshes...       # meshes -> label pyramid (surface band / background / ignore)
ufsm axis <root> <ct-group> <out.json>                 # per-z centroid axis when no umbilicus exists
ufsm sample <sources.json> --out DIR                   # dump patch montages to eyeball the sampler
ufsm train <sources.json> --out runs/r1 --gpus 0 --P 512 --B 1 --steps 20000
ufsm predict <ckpt> <root> <ct-group> <out> --um U [--box z,y,x,nz,ny,nx]
ufsm eval <pred-root> <pred-group> <label-root> <label-group> --um U [--box ...]
ufsm train <sources.json> --task papyrus --pap-cond 2 --pap-tau 2,8 --dice 0 ...   # papyrus vs void, labels from the CT itself
ufsm predict <papyrus.ckpt> <root> <ct-group> <out> --um U --papyrus-cut 80        # papyrus where the smoothed CT exceeds 80
```

Papyrus / void network (DESIGN.md "Papyrus / void network"): per patch a random cutoff t, labels = the clean CT smoothed
by a Gaussian above t, scanner augmentation on the input only, the cutoff fed as the input's zero, so one checkpoint serves
every cutoff.

## Training: crush augmentation

Sheet-aware crush (`src/crush.h`, DESIGN.md "Sheet-aware crush augmentation"): a window is pulled through an invertible
map (positive Jacobian) that squashes it in the sheet plane, bends / kinks it along the sheet normal and collapses the
voids between wraps while papyrus keeps its thickness; the stored phase is pulled through the same map, so phase, winding
mod 4, normals and density stay exact. Applies to stored-phase windows (task unwrap / unwrap2, side 2, cube windows) and
needs `--geometry-aug 1`. **Every flag defaults off**: without `--aug-crush` the batches are bit-identical (tested), so
existing recipes are unchanged.

| flag | default | meaning |
|---|---|---|
| `--aug-crush P` | 0 | probability a window is crushed |
| `--aug-crush-void LO,HI` | 0.15,0.5 | void ratio a gap keeps (per window, varying over it) |
| `--aug-crush-squash P` / `--aug-crush-k LO,HI` | 0.5 / 1.5,3 | in-plane squash probability / factor |
| `--aug-crush-bend P` / `--aug-crush-bend-amp A` | 0.5 / 0.06 | low-frequency bends along the normal (amplitude x P) |
| `--aug-crush-kink P` | 0.25 | a fold line |
| `--aug-crush-rot DEG` | 10 | the crush map's own rotation |
| `--aug-crush-fine N` | 0 | CT box mean from N levels finer (too slow for the loader) |
| `--aug-crush-warp 0/1` | 0 | 1: crushed windows also take the ordinary geometry warp (rotation / elastic / affine / zoom, `--rotate-p`, `--elastic-p`, `--aug-affine-p`, `--zoom-p`) after the crush map; labels pulled through both |

Realistic setting (the desk crush arm `ctrain.sh`; at the defaults above gaps collapse into touching sheets with the phase
kept smooth, which is the crushed-scroll look):

```sh
--geometry-aug 1 --aug-crush 0.3 --aug-crush-warp 1 --aug-crush-bend 0.7 --aug-crush-kink 0.35 \
  --rotate-p 0.3 --elastic-p 0.5 --aug-affine-p 0.4 --aug-aniso 0.1 --aug-shear 0.06 --zoom-p 0.3 --zoom-min 0.85
```

CPU cost: a crushed P512 window takes 18-31 CPU-s on the sampler (tile decodes through the map's footprint; P256 6-17 s),
against well under 1 s for a plain window. Budget workers for it: at P512 `--aug-crush 1` needs ~16-24 sampler workers to
feed one GPU and stays loader-bound on the laptop; 0.3 is the comfortable rate for normal training. `--aug-crush-warp 1`
adds only the ordinary warp's cost.

Debugging: `build/crush_render <sources.json> <out-prefix> [P] [windows] [seed] [level] [fine] [box] [--aug-crush-* flags]`
renders CT | phase | crushed CT | crushed phase panels. `UFSM_DUMP_WIN=PREFIX` (with `UFSM_SYNC_UPLOAD=1`, `--gpu-input 0`,
cube windows) makes `train` write the first `UFSM_DUMP_N` (8) host windows as PREFIX-NN.jpg (columns
CT | phase hue over CT | winding mod 4 hue over CT; rows z-mid / y-mid / x-mid).

Ground-truth export scripts: `tools/pget.sh` (parallel ranged download), `tools/ingest_all.sh` (all label zarrs),
`tools/ingest_segments.sh`, `tools/raster_all.sh`, `tools/surfcomp_all.sh`.

## Runtime notes

Fresh training computes GroupNorm statistics from rounded stored activations;
resume and prediction preserve each checkpoint's contract (`--gn-stats stored|legacy`).
`--input-prec 4|8` selects stem-input quantization independently of body storage and is saved in
the checkpoint.

`predict --grid-origin 0,0,0` anchors tile interiors to CT coordinates at the selected
level. Changing the requested output box or shard layout therefore preserves the neural windows
covering each shared voxel. Two native 1024-cubed regions preserve every common-ROI decoded byte
under all three tested axis shifts; lossless fixture tests also cover arbitrary crop origins,
host/device placement and multiple GPU workers. Without this flag, the historical box/shard-relative
tiling remains available. Aligned 528/halo8/shard512 serving retains eight tiles per 1024-cubed box.
Off-grid requests can repeat windows across output shards: the tested one-axis-expanded boxes use
20 forwards instead of 12 unique windows. The aligned full-volume throughput does not apply to
those requests. Lossy output encoding can also depend on codec-block placement; use `--q 0` for
arbitrary-crop byte comparisons.

Evaluation grids with three or more cutoffs now filter prediction probabilities once and collect
byte histograms instead of dilating each threshold separately. Complete ten-cutoff scoring is
4.9–5.5x faster on the tested native dense and partial boxes, with byte-identical reports.
One/two-cutoff evaluation retains the direct scan. This accelerates calibration and scoring,
not training updates or CNN inference.

Remote sharded Zarr reads combine nearby uncached chunks into bounded requests and fetch uncached
shard indexes concurrently when a disk cache is configured. Both paths are enabled by default;
`UFSM_Z3_COALESCE=0` and `UFSM_Z3_INDEX_PARALLEL=0` disable them for comparisons.
Grouped requests span at most 2 MiB with at most 64 KiB gaps, and index fetching uses at most
16 threads. Index checksums and ranges are checked; HTTP permission failures are errors, while
genuinely missing shards retain the array's fill value. Cold native 1024-cubed trials on two scans
showed about 1.3–1.4x whole-process inference speed, with identical prediction and cache bytes.
Cached/local inference, GPU computation and training have no established speedup from this change.

Reader workers now return idle HTTP handles to a bounded pool, retaining connections between read
jobs. `UFSM_HTTP_REUSE=0` disables this for comparisons. The pool retains at most 32 handles with
two cached connections per handle; request options and authorization headers reset before reuse,
and global cleanup closes the pool after callers join. Additional cold native trials take about
5.3 s versus 7.4–8.4 s on two 1024-cubed regions. A longer dense region gains 1.43x, and a region
with 48 empty tiles out of 64 gains 1.60x. Prediction stores and cold caches match exactly;
cached/local inference stays within 1%. These are bounded cold-input gains, not GPU-kernel or
whole-volume speedups.

## Local surface fitting with recto guidance

The unwrap checkpoint predicts recto (channel 0), winding phase (2–3), normals (4–9),
and density (10). `predict --fit-fields` and `--fit-zarr` now retain the recto maximum
of each field cell as `r.raw` or the `r` array, alongside the existing ten fields.
Old field stores still load. `gap-density` preserves the optional recto field.
For local positioning, `--fit-h 1` retains the recto head's native resolution;
larger cells broaden its peaks through maximum pooling.

An experimental local fit can use recto peaks as independent integer-level-set
anchors, with normals and density constraining sheet orientation and spacing:

```sh
build/ufsm fit fields --out local-fit --core 0 --axis-normal 2000 \
  --w-recto 8 --w-phase 0.25 --phase-robust 0.10 --unwrap-voxels -1
build/ufsm wraps local-fit local-tifxyz --fields fields --step 1
```

`--axis-normal` is a local initialization: its signed distance is in the fields'
native voxels. Compare its two signs when selecting a local box. Recto contributes
`r² (1 - cos(2πW))` in the phase stages and in the initial pitch/offset search.
Density and normal losses retain their defaults; `--w-density` and `--w-normal`
set their relative strengths. `--phase-robust S` applies a Cauchy penalty to the
circular phase residual with scale S turns. `--unwrap-voxels -1` skips the phase
flood, which can otherwise commit to bad phase predictions before optimization.
`--w-phase 0` fits positions from recto, normals and density alone. Recto anchors
do not uniquely establish winding integers across disconnected regions.

These options are opt-in. The synthetic local-stack test includes a confident
0.35-turn phase error, a missing recto patch and false recto detections. Mean
geometry error falls from 0.148 to approximately 0.007 turns, with positive sheet
spacing. Existing spiral tests and finite-difference checks of both anchor kernels
pass. A matched 2.46 mm PHerc1299 crop modestly improves CT alignment (samples below
CT 40: 11.1% to 10.4%); this is a proxy without surface ground truth. Finer lattices
increase roughness on this crop without a useful additional CT improvement.
The existing larger-box tifxyz surfaces score 10.5% on the same crop. Rotating the
predicted phase by 0.35 turns in its middle third changes the fitted coordinate by
0.132 turns on average with the old objective and 0.013 with recto guidance, measured
against each method's uncorrupted fit after integer gauge alignment. Approximately
1% of samples still change by at least half a turn in the guided fit, so integer
stability needs further work.
Whole-scroll topology and accuracy are not established by these local checks.

## Flattening wraps

Wraps are parameterized by column = angle and row = z, so their renders are sheared and stretched
(Paris 4 w072: σmax/σmin median 1.6). `ufsm flatten` SLIM-flattens each wrap
(symmetric Dirichlet, multilevel, flip-free line search; `src/flatten.h`). It then resamples
the wrap onto an isometric uv grid and crops it to the largest fully valid rectangle; enclosed
tears of up to `--fill-holes` vertices are Laplace-filled first and shown orange in
`<out>/<wrap>_mask.jpg`. `ufsm render` of the output is the flattened render.
`flatten-warp` moves an existing render of the original wrap onto the flat grid, through the
`gu.tif` and `gv.tif` source-grid maps:

```sh
build/ufsm flatten wraps flat --wraps w072 --threads 16      # distortion before / after on stdout
build/ufsm render flat r --volume ink.zarr 9.6 --ct CT 2 --depth 1 --scale 2
build/ufsm flatten-warp old/w072_published.jpg flat/w072 r/w072_published.tif --src-px 8 --scale 2
```

When the wraps' union box does not fit `UFSM_RENDER_MEM` GB, `render` goes chunk-ordered: blocks
sorted by chunk row, every chunk of the ink and CT zarrs decoded once (decoder threads run ahead of
the renderers, 32 for HTTP stores), chunks freed after their last use. `ink-compare` streams the
renders strip by strip (histograms, then the band statistic, then panels -> BOX half -> BILINEAR fit
row by row). Both write the same bytes as before; a whole Paris 3 wrap (338 cm², 1.6 Gpixel
renders) took render 5:28 -> 1:37 and ink-compare 1:37 / 51.6 GB -> 0:19 / 0.47 GB.
`UFSM_RENDER_TILED=1` and `UFSM_INKCMP_MEM=1` select the old paths.

## Recto and verso faces

A wrap is a level set W = k of the smooth winding field. It follows the sheet but can sit anywhere
across the sheet or in the gap next to it. `ufsm faces` (`src/faces.h`) finds the two faces of the
papyrus sheet under each vertex. It samples the CT along the outward normal and splits the profile
into bands (persistence-simplified extrema, so a shallow fibre-layer dip does not split a sheet).
Each band's edges sit at the half level between its peak and the minimum on that side; a high
valley between touching sheets is an edge. The pick is made consistent over the surface with
median windows, then median filtered, and holes are filled by push-pull. Output per wrap, on the
input grid: `<w>_recto` (inner face, toward the umbilicus) and `<w>_verso` (outer face), plus
`conf.tif` / `thick.tif` in `<w>_recto` and `<w>_faces.txt`.

Recto is the inner face because of the published-label side rule (`band_field_side_pair`): the
papyrus outward of wrap k is labelled recto side. Run it on the flattened grid, so both faces share
one uv and render pixel-aligned. `ink-compare --rgb-agree recto,verso` writes the agreement image:
recto ink is magenta, verso ink is green, and both together are white. `--diff-agree recto,verso`
writes |recto - verso| through viridis instead (0 = agree, 255 = disagree; `--diff-gain G` scales it).

```sh
build/ufsm faces flat faces --ct CT 2 --ct-um 9.6 --umb umbilicus.json --umb-um 2.4 [--gt codes.zarr 9.6 --gt-um 9.6]
build/ufsm tifxyz-scale faces up 1 --flip-rows 1
build/ufsm render up r --volume ink.zarr 9.6 --ct CT 2 --depth 1 --scale 2
build/ufsm ink-compare w_rgb.jpg --ct r/w_recto_ct.tif --ink recto=r/w_recto.tif --ink verso=r/w_verso.tif --rgb-agree recto,verso --stretch --px-um 4.8
```

## Prediction input transform

`predict --input-gain G --input-bias B` exposes the standardized CT transform
`G * zscore(CT) + B`, matching training's intensity jitter. Applying an affine
change before per-window standardization largely cancels it. These flags change
only the CT input channel; radial inputs and CT material masking are preserved.
Defaults preserve existing predictions. Split inference, surface winding,
conditioned sweeps, and cutoff-conditioned papyrus models reject nonidentity
input transforms.

## Label encoding (our exported stores)

Legacy targets: uint8, volcomp lossless, `fill_value` 255: `0` background, `254` surface, `255` ignore. Coarser pyramid
levels hold `round(254 * surface fraction)` of the non-ignore children, or 255 when at least half are
ignore. The sampler turns this into a soft target and a loss mask (CT > 0 and not ignore).

The Paris 4 all-surfaces target instead uses binary `0`/`255`, `fill_value` 0, the lossless mask
codec, and a finest level of 4.8 micrometers. Its zeros are supervised background wherever CT > 0.

Third-party code: `third_party/volcomp.h` and `third_party/surfcomp/` (MIT, SuperOptimizer).
