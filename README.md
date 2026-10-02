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
accuracy gap on MANBp; multi-source confirmation, threshold calibration and seam validation remain.
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
The default report threshold, 0.6, is provisional and is not independently calibrated.

Prediction uses a staging directory and replaces an earlier output only after success. Repeated calls
reuse outputs only when hashes, source arguments, axis and settings match. The bundle is checked before
every prediction. GPU leases prevent overlapping commands from cooperating production runners; direct
`ufsm` invocations must still be scheduled by the caller. Model-bundle status means an evaluated
candidate; it does not certify a globally optimal precision choice or production model quality.

## Label encoding (our exported stores)

uint8, volcomp lossless, `fill_value` 255: `0` background, `254` surface, `255` ignore. Coarser pyramid
levels hold `round(254 * surface fraction)` of the non-ignore children, or 255 when at least half are
ignore. The sampler turns this into a soft target and a loss mask (CT > 0 and not ignore).

Third-party code: `third_party/volcomp.h` and `third_party/surfcomp/` (MIT, SuperOptimizer).
