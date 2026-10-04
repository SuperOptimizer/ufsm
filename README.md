# ufsm — ultra fast scroll model

C23 + CUDA, no PyTorch, no cuDNN. Trains a ~1 M-parameter 3-D convnet for the recto papyrus face of
Vesuvius Challenge scrolls from the released ground truth only (HuggingFace `scrollprize/datasets` and
the AWS open-data bucket), after re-exporting that data into volcomp volumes and surfcomp surfaces.
Design, data inventory and milestones: [DESIGN.md](DESIGN.md).
The experimental surface-confidence and continuous-winding pipeline, including
matched ablations and constrained sheet reconstruction, is documented in
[SHEET_PIPELINE.md](SHEET_PIPELINE.md). It retains the native backbone; its optional
geometry/reconstruction tools use SciPy and PyTorch.

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
```

Pipeline scripts: `tools/pget.sh` (parallel ranged download), `tools/ingest_all.sh` (all label zarrs),
`tools/ingest_segments.sh`, `tools/raster_all.sh`, `tools/make_sources.py` (writes `configs/all.json`).

## One Paris 4 volume containing all AWS surfaces

`tools/build_surface_store.py` downloads every segment's tifxyz coordinates registered to the
Paris 4 scan `20260411134726` at 2.4 micrometers. The AWS inventory on 2026-10-02 contains 81
segment directories with this registration. Other registrations of the same surface belong to
different CT coordinate systems. A missing or ambiguous registration stops the build.

That full inventory includes overlapping revisions of the same wraps. The current curated
Paris 4 target uses exactly the 25 directories in `configs/paris4-segments-20260623.txt`
(wrap groups w010–w123). Selection is by exact directory name, so another revision of the
same wrap group does not enter the union. Missing or duplicate requested IDs stop the build.

```sh
python3 tools/build_surface_store.py \
  --segments-file configs/paris4-segments-20260623.txt \
  --download-cache /vesuvius/ufsm/gt/paris4-all-surfaces \
  --work /vesuvius/ufsm/gt/paris4-selected-20260623 \
  --out /vesuvius/ufsm/gt/paris4-selected-20260623/labels.zarr \
  --sources-out configs/paris4-selected-20260623.json --threads 12
python3 tools/verify_surface_store.py \
  --work /vesuvius/ufsm/gt/paris4-selected-20260623 \
  --sources configs/paris4-selected-20260623.json \
  --out /vesuvius/ufsm/gt/paris4-selected-20260623/sample-check
python3 tools/prepare_surface_training.py \
  --work /vesuvius/ufsm/gt/paris4-selected-20260623 \
  --out /vesuvius/ufsm/gt/paris4-selected-20260623/training-4p8.zarr \
  --sources-out /vesuvius/ufsm/gt/paris4-selected-20260623/training-sources.json \
  --source-template configs/paris4-selected-20260623.json
```

The cache flag reuses previously downloaded AWS files after comparing their sizes and
ETags with the selected inventory; it also supports an empty cache. The resulting provenance
records every selected directory, the selection-file hash, and input receipts. The mask
geometry, binary codec, held-out box, and native nearest-neighbor upsampling are the same as
below. The curated store has its own path, so an existing training job keeps consistent labels.
Wraps omitted from this list contribute no positive labels. The background supervision
assumption described below still applies.

The 2026-10-03 selected build's finest mask occupies 4.86 GiB and rasterized in 728 seconds;
the complete six-level pyramid occupies 6.40 GiB.
Three real native 704-cubed training samples passed finite-input and target checks. In the
fixed 192-cubed held-out preview, positive labels fell from 2,649,811 to 1,344,133 voxels
(49.3% fewer), with no added positives and the same surface-band expansion. These are
data-selection measurements, not accuracy scores for a model trained on the new labels.

The earlier full-inventory build is reproducible with:

```sh
make build/ufsm build/check_surface_samples
python3 tools/build_surface_store.py --threads 12
python3 tools/verify_surface_store.py \
  --work /vesuvius/ufsm/gt/paris4-all-surfaces \
  --sources configs/paris4-all-surfaces.json \
  --out /vesuvius/ufsm/gt/paris4-all-surfaces/sample-check
```

The result is **one sparse binary mask pyramid**, `/vesuvius/ufsm/gt/paris4-all-surfaces/labels.zarr`,
whose finest stored level is **4.8 micrometers** (`--level 1`). There is no 2.4-micrometer label
array. All meshes contribute to the same union: 255 means expanded surface and 0 means no surface.
The band includes each raster seed and its 18 face/edge neighbours (chamfer distance <= 4),
about three stored voxels thick for an axis-aligned surface. Each cube can contain multiple sheets.
All six levels remain binary; coarser levels use any-positive pooling to preserve thin surfaces.

Chunks use the **lossless 128-cubed mask codec** in `volcomp.h`; its built-in lossy 2x pooling
mode is not used. Empty chunks remain implicit zero bits. The source config declares
`encoding: "binary", min_level: 1` on the target, while the CT stays at native 2.4 micrometers.
The sampler reads the smallest matching mask region and upsamples finer requests with nearest
voxel centres in global coordinates (ties toward increasing coordinates), including odd cube origins.
Stored 255 is converted to the trainer's surface value before the legacy ignore handling.
Optional `--soft 3` generates continuous targets at sampling time without storing fractional values.

This source has **no ignore class or trust band**. Every CT-positive voxel outside the expanded
surface mask contributes background supervision; air remains excluded. This assumes that regions
without a released mesh are negative. Including all 81 released surfaces does not establish that
every physical sheet was annotated, so omitted sheets would receive negative supervision.
The existing partially annotated sources retain their original background/ignore rules.

AWS files occupy about 18.8 GiB. Downloads retain ETags, sizes and SHA256 receipts for reuse.
Rasterization uses a spatial mesh index and a direct 19-point binary expansion, checked against
the original distance transform. A `.building` directory is published only
after every pyramid level succeeds; the script then writes the single-source configuration.
An interrupted raster leaves its staging directory for inspection; raster restart currently requires
moving that directory aside before rerunning. `--fetch-only` only prepares the AWS inputs.

The verification command draws three real native 704-cubed training samples on the CPU, checks
finite inputs, surface and soft-target coverage, and writes orthogonal CT/target/mask montages
plus JSON counts. It supports `--wait` while the build is running and `--builder-unit` to detect
failure of a systemd user build service. After this verification succeeds, train directly from
`configs/paris4-all-surfaces.json` with the desired window and `--soft 3`.
The existing Paris 4 held-out box remains excluded from training. Raw surfaces, labels, montages
and build outputs stay outside Git; the repository contains the reproducible code and config.

To start native training after the 4.8-micrometer render finishes, while coarser levels
are still building, freeze that completed level into a stable view:

```sh
python3 tools/prepare_surface_training.py \
  --work /vesuvius/ufsm/gt/paris4-all-surfaces \
  --out /vesuvius/ufsm/gt/paris4-all-surfaces/training-4p8.zarr \
  --sources-out /vesuvius/ufsm/gt/paris4-all-surfaces/training-sources.json \
  --source-template configs/paris4-all-surfaces.json
ufsm train /vesuvius/ufsm/gt/paris4-all-surfaces/training-sources.json \
  --out runs/paris4 --P 704 --B 1 --gpus 0,1 --split z --levels 1,0,0,0
```

The view uses hard links on the same filesystem, so it preserves the finished mask through
the builder's final rename without duplicating its payload. An exact coarse occupancy array
is built from nonempty inner chunks to guide sampling; native targets come from the 4.8 level.
Its indexes are checksum-checked. The view's background and held-out box rules match the
completed store. Optional `--resume` starts from an existing checkpoint.

## One complete pass over the selected Paris 4 labels

The cleaned June 23 labels can be trained with one finite, shuffled cover instead of random
sampling. Prepare the stable 4.8-micrometer training view as above, then build the cover:

```sh
python3 tools/build_training_cover.py \
  --sources /vesuvius/ufsm/gt/paris4-selected-20260623/training-sources.json \
  --out /vesuvius/ufsm/gt/paris4-selected-20260623/cover704.json --P 704 --seed 2
python3 tools/production.py train --recipe configs/paris4-selected-cover704.json \
  --out runs/paris4-selected-cover --gpus 0,1 --split z --mem auto \
  --resume runs/previous/large/last.ckpt
```

The current plan contains **39,453 distinct 704-cubed tiles**. It covers every occupied coarse
label-cell box within native bounds `[27904,5376,5888]` to `[68096,29696,30464]`, excluding the
held-out box. Six slabs around that box and clamped final tiles avoid coverage holes. Tiles
near slab/tail boundaries overlap; each planned tile is visited once. This is a coverage pass,
not a promise that every individual voxel is visited exactly once. Unlabeled scan ends are
excluded. At the previous 0.378 cubes/s, GPU training would take about 29 hours; augmentation
and uncached reads still need measured throughput.

The network is unchanged: widths `16,32,64,80`, 1,172,050 parameters, one native 2.4-micrometer
window split across both GPUs. Binary targets are nearest-neighbor upsampled from 4.8 micrometers.
The original recipe fixes the hard label band and soft sigma 3. The recipe retains the FP4 body, FP8 stem input
and activation-gradient storage, FP32 master/optimizer/reduction state, and existing automatic
memory modes.

Augmentation retains gain, offset and Gaussian noise. On half the samples it additionally draws
from the eight z-preserving XY rotations/reflections, including identity. CT, targets, validity and
radial vector components transform together. The optional reconstructed-CT appearance module uses
mild gamma 0.9–1.1 (p 0.15), bounded linear shading (p 0.15), weak smoothing (p 0.10) or unsharp
filtering (p 0.10), and correlated Gaussian noise (p 0.15). At most two of gamma/shading/filtering
combine. Auxiliary axis uncertainty is applied on 10% of samples, bounded by 32 native voxels per
XY component and a 2-degree radial direction change throughout the cube. Label dilation/erosion,
CT/label misregistration, arbitrary z-axis swaps and crop-origin jitter are disabled. These
appearance transforms approximate reconstruction sharpness/noise variation; they are not a
simulation of the scanner's acquisition physics or a measured accuracy improvement.

For thinner surface targets, `--erode 1 --soft 2 --levels 1,0,0,0` erodes the binary
core by one native voxel using its six face neighbours, then adds the sigma-2 soft
falloff. Erosion runs after nearest-neighbour upsampling from 4.8 to 2.4 micrometers.
A one-voxel halo is read from the labels so crop faces do not cause false erosion.
The option supports native-level binary pyramid targets, defaults to zero, and is
applied to both training and validation. Source labels and coverage plans remain
immutable. Every checkpoint records the target settings; use the same flags when
resuming. Compare development F1/AUC across target changes, since BCE and Dice losses
then use different targets. Erosion can remove thin bridges and thin valid bands;
it does not certify that all touching sheets are separated.

The runner freezes the cover, binary, source configuration, axis and recipe. Its SHA256, total
count, committed cursor and original schedule base are saved in every checkpoint. A continuation
using the same frozen cover and recipe resumes at the next tile and retains the original final
step and LR schedule. Prefetched but uncommitted samples can be regenerated after interruption.
Missing teachers, read failures or nonfinite updates cannot silently consume a planned tile.
Keep augmentation settings and seed unchanged when resuming. Target-width annealing is rejected
for finite covers because asynchronous prefetch would otherwise make it timing dependent.

Use `--trial-seconds 60` for a bounded production smoke check. This retains the complete-pass
step schedule, marks the resulting run **partial**, and allows resuming its checkpoint for the
remaining tiles. Completed cover checkpoints reject another pass under the same plan. The CPU
checks are `build/test_cover`, `build/test_ct_augment`, `build/test_sampler_safety`,
`tests/test_training_cover.py` and `tests/test_production_cover.py`; the real single/split GPU
exhaustion and resume regression is `python3 tests/test_cover_training.py`.

## Reproducible training and deployment

`tools/production.py` freezes the executable, sources configuration, axis files, recipe and evaluation
script. Its run manifest records commands, checkpoint hashes and completed stages. New checkpoints
embed their precision policy and storage settings; moving the checkpoint preserves inference defaults.
The runner removes inherited `UFSM_*` settings and uses the recipe's explicit environment instead.

The current [candidate recipe](configs/production-candidate.json) uses a 128-cubed warmup followed by
512-cubed batch-1 training on one GPU, with 528-cubed / halo-8 inference. Staging closed the short-run
accuracy gap on MANBp. The corrected short multi-source confirmation is complete, but dense-box
accuracy remains weak and the aggregate score is distorted by partially labelled segments.
Two matched seeds, each continued for 160 updates, show no consistent aggregate accuracy gain
from FP8 weight gradients; the recipe retains the faster FP4 path. The pipeline is qualified for
a full training run after bounded precision, calibration and global-grid checks plus all 40 test
commands. The resulting model still needs post-training acceptance; the short-trial dense scores
remain weak.
Fresh training computes GroupNorm statistics from rounded stored activations;
resume and prediction preserve each checkpoint's contract (`--gn-stats stored|legacy`).
`--input-prec 4|8` selects stem-input quantization independently of body storage and is saved in
the checkpoint. Cooperative gradient loading and fused stored-output statistics bring the FP8-stem
candidate to about 71 Mvox/s at 512. The recipe's wider finest gradient buffer brings this to
73.7–74.2 Mvox/s, with 14796 MiB observed device use and about 1.5 GiB spare. Cached native
MANBp inference gains about 12% with matched FP4; its decoded output is identical across the tested
1024-cubed box. Nearby 496/528 shapes previously had comparable per-voxel speed; no production
precision choice is final.
The large MX4-input/MX8-gradient weight-gradient path now uses deeper persistent tiles, retaining
the existing depth for other layouts. A frozen four-run comparison gives 74.5–74.9 Mvox/s versus
72.5–74.0 controls (2.0% by paired-run means), with unchanged 14785 MiB observed device use.
Clocks and temperatures drift across the sweep; even the slower candidate exceeds the faster
control by 0.7%. The arithmetic, rounding keys and compiled GPU kernels are unchanged;
FP32 reduction grouping differs. `UFSM_F4W_ZC=12` retains the old depth for comparisons.
Pure forward inference at 528 cubed takes 0.424–0.429 s on one RTX 5060 Ti 16 GB:
343–347 million window voxels/s, or 313–317 million useful voxels/s after the halo-8 crop to 512.
This restores the actual checkpoint and measures warm whole-network forwards on real device input;
input preparation, transfers, reads, output encoding and writes are outside that timing.
Cached/local complete predictions previously measured about 240–260 million useful voxels/s.
An exact forward-operand cache was tested and rejected: complete cached/local predictions took
7–8% longer and used another 3664 MiB, despite identical output bytes. The major measured kernel
and IO candidates have now been qualified or rejected; another 2x gain is not established.
All throughput figures above are per GPU. Kaggle's 320-cubed regions participate in the warmup and
are explicitly excluded at 512. No long run has started. The full recipe uses 780 seconds of warmup
and 20,000 large-window updates, approximately ten hours of training at the measured throughput,
plus validation, checkpoint and final scoring time. Launch it with:

```sh
python3 tools/production.py train --out runs/production512 --gpu 0
```

The production runner also supports both GPUs. Data parallelism trains a separate 512-cubed window
on each GPU and averages their parameter gradients; `B=1` then means effective batch 2. Spatial
parallelism splits each window along z, exchanges boundary activations and gradients, and reduces
normalization/loss statistics over both GPUs; effective batch remains 1. Both modes preserve staged
resume and export, and acquire each device's lease separately.

A fresh bounded comparison using the same checkpoint and 24 updates per mode measures 75.6 Mvox/s
on one GPU, 149.7 total with data parallelism, and 141.7 / 144.4 total with spatial split in wide / auto
memory mode. Five four-update intervals exclude startup and final validation. Split auto observes
11236 / 11219 MiB total card use, leaving about 5 GiB per card. It completes about 1.076 updates/s,
versus 0.563 on one GPU. The same 20,000 large-window updates therefore take approximately 5.2 hours
plus warmup, validation and scoring; these short timings do not guarantee sustained clocks or convergence.
For this batch-1 run, use:

```sh
python3 tools/production.py train --out runs/production512-split --gpus 0,1 --split z --mem auto
```

With the production FP8 stem and FP4 body, a bounded split-window sweep also trains 640, 704 and
720 cubed at batch 1; 736 and 768 run out of memory. Window sizes must be multiples of 16.
640 measures 138.9 Mvox/s total with about 3.3 GiB spare per card. The original 704 mode measures
126.5 Mvox/s with 1.8 GiB spare; 720 fits with only 0.8 GiB spare. These include real optimizer
steps, final validation and checkpoint writing, rather than allocation-only checks.

The low-memory recompute mode now reuses the final decoder's surviving first activation on its
first backward, preserving repeated-backward behavior. At 704, a control/reuse/control comparison
gives 125.1–126.0 versus 132.7 Mvox/s with unchanged memory. Optional
`UFSM_RC_KEEP_COARSE=1` retains the smaller decoder activations too. A final 24-update confirmation
measures 135.9 Mvox/s total and 14964/14947 MiB sampled card peaks, leaving about 1.3 GiB per card.
This recovers approximately half of the voxel-throughput gap to 512, with unchanged quantization
and FP32 accumulation. The full suite checks fresh/repeated gradients and spatial split with retention.

To select 704, copy the recipe, change the large stage's `P` to 704 and use `--mem auto` with
`--gpus 0,1 --split z`. Enable decoder retention in the copied recipe's `environment` as
`"UFSM_RC_KEEP_COARSE": "1"`; the production runner freezes that environment and clears ambient
UFSM variables. Retention is optional and uses another 454 MiB per card in the measured 704 run.
The standard recipe remains 512/528. Larger windows need their own serving-window calibration and
quality acceptance. At the measured rate, 20,000 updates at 704 take approximately 14.3 hours plus
warmup/validation/scoring; each window contains 2.6 times as many voxels as 512.

For data parallelism, use `--gpus 0,1` without `--split z`. Keeping 20,000 updates processes twice as
many windows and retains approximately the single-GPU duration. Halving the updates matches the
original voxel budget but changes the optimizer-step count; that is a different training recipe.

These commands run a **limited pipeline trial**, calibrate and score precision profiles,
and export a self-contained candidate bundle:

```sh
python3 tools/production.py plan
python3 tools/production.py train --out runs/production-trial --gpu 0 --trial-seconds 60
python3 tools/production.py evaluate runs/production-trial --gpu 0 --predictions /vesuvius/ufsm/eval
python3 tools/production.py export runs/production-trial --profile matched --out runs/model-bundle
python3 tools/production.py predict runs/model-bundle --root <ct-root> --ct <ct-group> --um <microns> \
  --out <prediction-dir> --box z,y,x,nz,ny,nx --axis <axis.json> --cache <cache-dir> --gpu 0
```

Each training stage gets the specified trial budget. Without `--trial-seconds`, `train` runs the full
recipe. `--resume <checkpoint>` continues weights and optimizer state in a new run directory. Stage
learning-rate schedules restart without resetting the global optimizer step. All precision profiles
must return complete held-out scores; changed inputs are rejected before export. Reports distinguish
prediction time from scoring time and leave throughput empty when cached predictions are reused.
The candidate recipe scores dense labels at native resolution and partial labels at level 1
(or their minimum available level). It fits one global cutoff per serving profile by maximizing
geometric mean F1 on MANBp and Paris4, then reports the six remaining dense boxes separately as
acceptance sources. Relative source gains receive equal weight; exact zeros have no score floor.
Dense and partial groups include their constant-foreground baselines. Exported bundles retain the selected
profile's calibrated cutoff. Custom source configurations must provide matching evaluation groups
and roles in their recipe; recipes without a calibration split retain their fixed report cutoff,
and calibration without an aggregation setting retains arithmetic mean F1.
The existing holdouts have been inspected during development, so this split supports threshold
qualification and is not an untouched final test. On the frozen eight-dense-box diagnostic,
geometric calibration selects 0.525 / 0.500 / 0.525 for FP4/FP8/FP16 and gives mean acceptance F1
0.0736 / 0.0730 / 0.0729, versus constant foreground 0.0566. The four two-seed training comparisons
retain their original cutoffs and conclusions. Seven historical arithmetic reports remain exact;
changing acceptance or partial scores cannot affect the fitted cutoff. These bounded checks
qualify the serving choice, not the eventual production model's accuracy.

The recipe sets `--grid-origin 0,0,0`, anchoring tile interiors to CT coordinates at the selected
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

Prediction uses a staging directory and replaces an earlier output only after success. Repeated calls
reuse outputs only when hashes, source arguments, axis and settings match. The bundle is checked before
every prediction. GPU leases prevent overlapping commands from cooperating production runners; direct
`ufsm` invocations must still be scheduled by the caller. Model-bundle status means an evaluated
candidate; it does not certify a globally optimal precision choice or production model quality.

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

## Label encoding (our exported stores)

Legacy targets: uint8, volcomp lossless, `fill_value` 255: `0` background, `254` surface, `255` ignore. Coarser pyramid
levels hold `round(254 * surface fraction)` of the non-ignore children, or 255 when at least half are
ignore. The sampler turns this into a soft target and a loss mask (CT > 0 and not ignore).

The Paris 4 all-surfaces target instead uses binary `0`/`255`, `fill_value` 0, the lossless mask
codec, and a finest level of 4.8 micrometers. Its zeros are supervised background wherever CT > 0.

Third-party code: `third_party/volcomp.h` and `third_party/surfcomp/` (MIT, SuperOptimizer).
