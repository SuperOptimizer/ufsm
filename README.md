# ufsm — ultra fast scroll model

C23 + CUDA, no PyTorch, no cuDNN. Trains a ~1 M-parameter 3-D convnet for the recto papyrus face of
Vesuvius Challenge scrolls from the released ground truth only (HuggingFace `scrollprize/datasets` and
the AWS open-data bucket), after re-exporting that data into volcomp volumes and surfcomp surfaces.
Design, data inventory and milestones: [DESIGN.md](DESIGN.md).

```sh
make && make test        # gcc -std=c23 + nvcc; needs libcurl, libzstd, libblosc, zlib, CUDA 13 (sm_120)
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

uint8, volcomp lossless, `fill_value` 255: `0` background, `254` surface, `255` ignore. Coarser pyramid
levels hold `round(254 * surface fraction)` of the non-ignore children, or 255 when at least half are
ignore. The sampler turns this into a soft target and a loss mask (CT > 0 and not ignore).

Third-party code: `third_party/volcomp.h` and `third_party/surfcomp/` (MIT, SuperOptimizer).
