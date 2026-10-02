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
Matched precision trials and production accuracy qualification remain. A bounded
trial of corrected normalization and an FP8 stem has finished; dense-box scores remain weak.
Fresh training computes GroupNorm statistics from rounded stored activations;
resume and prediction preserve each checkpoint's contract (`--gn-stats stored|legacy`).
`--input-prec 4|8` selects stem-input quantization independently of body storage and is saved in
the checkpoint. Cooperative gradient loading and fused stored-output statistics bring the FP8-stem
candidate to about 71 Mvox/s at 512. The recipe's wider finest gradient buffer brings this to
73.7–74.2 Mvox/s, with 14796 MiB observed device use and about 1.5 GiB spare. Cached native
MANBp inference gains about 12% with matched FP4; its decoded output is identical across the tested
1024-cubed box. Nearby 496/528 shapes previously had comparable per-voxel speed; no production
precision choice is final.
Kaggle's 320-cubed regions participate in the warmup and are explicitly excluded at 512. The recipe's
long run is on hold while those checks and remaining performance headroom are assessed.

These commands run a **limited pipeline trial**, compare precision profiles at the same fixed threshold,
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
(or their minimum available level). It fits one global cutoff per serving profile on MANBp and
Paris4, then reports the six remaining dense boxes separately as acceptance sources. Dense and
partial groups include their constant-foreground baselines. Exported bundles retain the selected
profile's calibrated cutoff. Custom source configurations must provide matching evaluation groups
and roles in their recipe; recipes without a calibration split retain their fixed report cutoff.
The existing holdouts have been inspected during development, so this split supports threshold
qualification and is not an untouched final test. A current native-resolution diagnostic found
poor cutoff transfer to PHerc0500; precision and production-quality acceptance remain pending.

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
