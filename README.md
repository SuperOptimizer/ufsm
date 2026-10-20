# ufsm — ultra fast scroll model

C23 + CUDA, no PyTorch, no cuDNN. Trains a ~1 M-parameter 3-D convnet for the recto papyrus face of
Vesuvius Challenge scrolls from the released ground truth only (HuggingFace `scrollprize/datasets` and
the AWS open-data bucket), after re-exporting that data into volcomp volumes and surfcomp surfaces.
Design, data inventory and milestones: [DESIGN.md](DESIGN.md).

```sh
make && make test        # gcc -std=c23 + nvcc; needs libcurl, libzstd, libblosc, zlib, OpenSSL libcrypto, CUDA 13 (sm_120)
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
