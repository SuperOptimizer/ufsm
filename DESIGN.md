# ufsm — design

Ultra fast scroll model: a C23 + CUDA pipeline that trains a ~1 M-parameter 3-D convnet to predict the
recto papyrus face in Vesuvius Challenge micro-CT, from scratch, using only the released ground truth:
the gated HuggingFace `scrollprize/datasets` bucket and the AWS `vesuvius-challenge-open-data` bucket.
No teacher models, no distillation. All upstream data is re-exported once into the user's own codecs
(volcomp volumes, surfcomp surfaces) so training reads one compact local store.

## Constraints
- Host code is C23 (`gcc -std=c23`). GPU kernels are `.cu` files compiled by nvcc and linked into the
  same binary. No cuDNN, no cuBLAS: every kernel is ours.
- Third-party: libc/libm/pthreads, libcurl (HTTPS), libzstd, libblosc + zlib (only to read upstream
  zarr v2 / TIFF during ingest), CUDA runtime, and the vendored codecs `third_party/volcomp.h` and
  `third_party/surfcomp/` (both MIT, SuperOptimizer).
- Hardware: 2x RTX 5060 Ti 16 GB (sm_120), 32 cores, 182 GB RAM, `/vesuvius` nvme (≈400 GB free).

## Ground truth
| source | what | native form |
|---|---|---|
| HF `surfaces/2um_032726/*.zarr` | rasterized human-verified sheet meshes for PHerc0500P2, 0343P, MANBp, 1667, Paris4 (+3 winding-range zarrs, volume tbd); values 0 / surface / 2 = ignore | zarr v2, 128^3 u8 chunks, blosc-zstd / blosc-lz4 |
| HF `surfaces/kaggle/` | 500 CT+label 320^3 cubes (labels 0/1/2) | LZW TIFF stacks |
| HF `ink/<scroll>/<segment>/` | ink labels + supervision masks on tifxyz segments (for a later ink model) | TIFF / zarr |
| AWS `<scroll>/segments/*/mesh/.../tifxyz*` | 188 segment meshes over 12 scrolls, each in its original scan's frame | tifxyz: float32 tiled TIFF, deflate, fp predictor |
| `community-uploads/forrest/surfcomp/` | the same segments re-registered onto every scan of their scroll (880 surfaces; e.g. all 81 Paris4 segments on the 2.4 µm and 1.129 µm scans) | surfcomp `.sfc` |
| AWS `<scroll>/volumes/*.zarr` | CT | zarr v2 blosc; we read the user's volcomp mirrors instead |

Ingest (`ufsm ingest`) converts each label zarr into a volcomp **lossless** (q=0) zarr v3 sharded
pyramid in the volcomp tree layout (`<scroll>/representations/labels/<name>.zarr/<um>/`), the Kaggle
cubes into one volcomp array per cube with its CT, and every tifxyz into an `.sfc` via surfcomp.
Meshes become voxel labels by rasterization (`ufsm raster`): for each mesh, a band of ±1 voxel around
the surface = recto, a ±T voxel shell (T ≈ 6) with no other mesh = trusted background, everything else
ignore — the usrm `udf/valid` convention, stored as one u8 label pyramid per volume.

Recto needs a sense of "inside": the model gets the unit radial vector away from the scroll axis as
input channels (usrm2 convention, `(0, dy, dx)/|d|`). Axis control points come from the user's
`umbilicus-full-resolution.json` where one exists, else `ufsm axis` (per-z centroid of the masked CT).

## Model
usrm2's tiny U-Net, sized to ≈1 M parameters: 4 levels, widths (16, 32, 64, 80), two 3^3 convs +
GroupNorm(8) + SiLU per level, stride-2 3^3 conv down, trilinear 2x up + concat, 1^3 head.
Inputs: [z-scored CT, radial(3)] = 4 channels. Output: 1 recto logit per voxel (a second `sheet`
channel slot exists for whole-sheet labels such as the Kaggle set). Loss = BCE + soft Dice over voxels
whose label is not ignore and whose CT is nonzero.

Resolution: one model across voxel sizes, rung k = 0.6 x 2^k um; labels are pooled 2x per level.

## Pipeline
1. `zarr3` reader (done): local or HTTPS, cached, threaded decode; bit-exact vs volcomp 1.3.0.
2. `zarr2` reader + `zarr3` writer + TIFF reader; `ufsm ingest` / `ufsm raster` produce the local
   ground-truth store.
3. `sources` + `sample` (done for pyramids/regions): threaded patch sampler with the 48 cube symmetries,
   intensity jitter, radial channels; targets become hard labels with an ignore mask.
4. `nn.cu`: conv3d 3^3 fwd/bwd, GroupNorm, SiLU, stride-2 conv, trilinear up, concat, BCE+Dice, AdamW,
   EMA; each op has a CPU reference and a finite-difference gradient test.
5. `train`: LR warmup + cosine, raw-binary checkpoints with a JSON header, CSV log, held-out boxes.
6. `predict`: sliding-window inference with Gaussian blend, writes a volcomp zarr v3 pyramid in the
   published layout.
7. `eval`: recall/precision/offset against held-out meshes and label boxes.

## Performance notes (2026-09-30 night)
- All 3^3 convolutions (stride 1 and 2, forward, backward-data, weight gradient) run on the tensor cores as
  hand-rolled `mma.sync.m16n8k16` BF16 x BF16 -> fp32 implicit GEMMs. The input tile is staged in shared memory
  channel-contiguous, so for every kernel tap the B operand is just a shifted view (odd offsets via a funnel
  shift, stride 2 via parity-split tiles); no im2col matrix is ever built. `wmma` cannot be used for this
  because shifted views are not 32-byte aligned.
- GroupNorm statistics are reduced across 32 slabs per group; GroupNorm-apply + SiLU is fused into the
  following conv's staging in forward and into the weight-gradient staging in backward, so the normalised
  activations are never materialised (`nn_conv3d_fwd_gn`, `nn_conv3d_bwd_weight_gn`, `nn_silu_bwd_gn`).
- Stride-2 backward-data = zero insertion + the stride-1 tensor-core conv.
- `nn_set_tf32(0)` (env `UFSM_FP32=1` in the tools) selects the exact fp32 CUDA-core kernels; the
  finite-difference tests run on those and `test_unet` reports the tensor-core/fp32 logit agreement (~1%).
- Loss statistics, bias gradients (summed from the staged gradient tile inside the weight-gradient kernels),
  upsample index math and the GroupNorm backward (SiLU' recomputed on the fly, group sums precomputed) are
  all parallelised/fused; `ufsm train --gpus 0,1` runs data-parallel with an asynchronous loss and gradient
  averaging over peer copies (1.9x on two cards).
- The decoder concatenation is gone: the decoder's first conv reads the upsampled tensor and the encoder skip
  as two inputs with a channel split, and its backward-data writes the two gradients straight to their
  destinations. Upsample forward/backward are shared-memory tiled.
- Reference point: the same network in PyTorch 2.14 + cuDNN 9.2 (`tools/torch_bench.py`, bf16 autocast,
  cudnn.benchmark) does 96.6 ms/step at 96^3 batch 2 (20.7 samples/s, 3.75 GB) on the same card.
- Second round (2026-10-01): activations are stored as bf16 in tensor-core mode (`nn_set_act_bf16`, env
  `UFSM_ACTF32=1` turns it off; gradients, statistics and parameters stay fp32); the weight-gradient kernel was
  rewritten (9 warps = one (kz,ky) pair each sharing one row of loads for its three kx taps, padded row stride so
  every fragment is a 64-bit load, row-wise vector staging) and went from 13 to 19-23 TFLOP/s; the forward kernel
  stages rows with vector loads (11 instead of 25 instructions per MMA) and double-buffers its weight groups; the
  GroupNorm backward uses fp32 partials (fp64 is 1/64 rate here), slab grids and 4-wide loads; the stride-2
  backward-data accumulates straight into the skip gradient (prologue read: an epilogue read was 5x slower) and is
  parity-decomposed (8 classes, 27 taps total) instead of zero-insertion; nvcc `-use_fast_math` (+4%).
- A gradient check of the whole network (tensor-core path vs fp32, now in `test_unet`) caught a real bug: the m-tile
  choice did not divide 48 output channels, so the decoder's level-0 skip gradient missed 16 channels. Fixed; the
  whole-network gradient now agrees to 1.9% (bf16 rounding). `tests/test_fused.c` covers the fused gn+silu staging,
  the channel-split inputs and the bf16 storage against the fp32 kernels.
- Activation gradients are bf16 too (`nn_set_grad_bf16`, env `UFSM_GRADF32=1`): the whole-network gradient still
  agrees with fp32 to 2.0%, activations+gradients take 1.33 GB at 96^3 batch 2. The trainer uploads batches from
  pinned sampler buffers on a separate copy stream into double-buffered device inputs, so the upload of step k+1
  overlaps the compute of step k (events order the two streams).
- The forward kernel computes 4 output rows per warp (128-thread blocks, 3 per SM): fewer fragment loads per MMA.
  The stride-2 forward and weight-gradient kernels stage rows with vector loads like the stride-1 ones. The upsample
  forward and backward are separable (x, y, z blends in shared memory) with paired stores / sector-aligned loads;
  the head's weight gradient reads its inputs once (all 16 x 2 products per thread, 4 voxels per load).
- Result at 96^3 batch 2 on an idle RTX 5060 Ti: 0.56 s/step at the start of the work -> 62 ms (first round) ->
  36 ms (55 samples/s in the benchmark, 53 samples/s in `ufsm train`, 102 samples/s on both GPUs; 2.7x PyTorch+cuDNN). Per step: forward
  convs 8 ms, backward-data 8, weight gradients 10, GroupNorm 5, upsample 2, stride-2 convs 3, rest 2. Against the
  card's peak (36 SMs at 3.21 GHz: ~59 TFLOP/s BF16-tensor with fp32 accumulate, ~448 GB/s): the step sustains
  ~16 TFLOP/s (27%); forward kernels 25-36 TFLOP/s (tensor pipe active 52%), weight gradients 19-23 (38-41%; the
  remaining stalls are barriers between the staging and MMA phases and the shared-memory queue), the memory-bound
  kernels run at 55-75% of DRAM bandwidth. Batch 4 and 8 give the same samples/s as batch 2; voxel throughput is
  flat from 96^3 to 192^3 (29-30 Mvoxel/s at the 42 ms stage), so patch size and batch are modeling choices (192^3
  batch 1 needs 11.8 GB with fp32 gradients). Tried and reverted: 4-plane forward tile with 512 threads (spills),
  register prefetch of the next channel chunk (no gain, spills), warp-per-(kz,ky) with 288 threads in the old
  layout (slower), a 10th staging-only warp in the weight gradient (spills, no gain), double-buffered input slabs in the weight
  gradient (2 blocks/SM instead of 3: slower), accumulate read in the epilogue (5x slower than in the prologue). `build/bench_conv` (`UFSM_LAYER=i` for one layer) and `UFSM_PROF=1`
  give per-kernel numbers; Nsight Compute works with `sudo ncu`, nsys with `TMPDIR` set to a writable directory.

- FP8 / FP4 (opt-in, `ufsm train --prec 2`, `UFSM_PREC=2` in the tests; `src/nn_fp8.cu`, written by a second agent, report in
  `FP8_REPORT.md` of the agent's tree copy): e4m3 operands with MX block scales (ue8m0 per 32-element block, no amax pass,
  no loss scaling) on the block-scaled `mma.sync kind::mxf8f6f4`, which GeForce Blackwell runs at full rate with fp32
  accumulation (215 TFLOP/s measured vs 54 for bf16). Needs the `sm_120a` target. Per-conv kernels reach 30-77 TFLOP/s
  forward and 24-37 weight gradient; the step drops to 31 ms (65 samples/s; `ufsm train` 62 samples/s on one GPU, 120 on two). Cost: every conv has a 4% relative error
  (bf16: 0.2%), and the whole-network gradient against fp32 differs by 21% (bf16: 2%); on the agent's synthetic
  training task fp8 reached the same held-out loss as fp32, but this has not been checked on the real data, so bf16
  stays the default. FP4 (prec 3) is implemented but brings no step-time gain (the kernels are staging-bound).
- Vector staging paths require aligned rows: a bottom level of 6 or 10 voxels (patch 48 or 80) made 8- and 16-byte
  loads misaligned and silently zeroed the gradients; every vector path now checks `W % 4` (`% 8` for 16-byte loads) and
  `make test` runs `test_unet` at patch 48 as well.

- Patch size: voxel throughput is flat from 96^3 to 192^3 (49-51 Mvoxel/s per GPU for a train step), so the training
  default moved to 128^3 batch 2 on both GPUs (`tools/train_r1.sh`; 3.2 GB per GPU, 45 samples/s = 94 Mvoxel/s) and
  192^3 batch 1 (4.9 GB) is available for more context; after dropping the unused zero-insertion scratch in tensor-core
  mode, 256^3 batch 1 trains on the 16 GB card (2.8 samples/s = 47 Mvoxel/s). Inference prefers the largest window: with a 16-voxel halo
  the useful fraction rises from 30% at 96^3 to 58% at 192^3, so `ufsm predict` and `tools/eval_holdouts.py` now
  default to a 160^3 window: its 128^3 interior divides the 256/512 shards exactly (a 256^3 box predicts in 0.50 s
  instead of 0.80 s with 96^3 windows; 192 and 288 waste tiles on the shard grid, 288 also exhausts memory in training).

- Precision work (2026-10-01): the 16-bit type used for activation/gradient storage and for the MMA operands is now
  selectable (`nn_set_f16`, env `UFSM_F16=1`, `ufsm train --f16 1`, the training default): fp16 has 10 mantissa bits
  against bf16's 7 at the same tensor-core rate, so the step stays at 36 ms while the per-conv error drops from
  2.3e-3 to 3e-4 and the whole-network gradient error against fp32 from 2.0% to 0.24% (logit max error 0.06 -> 0.008).
  fp16 gradient storage keeps the activation gradients scaled (`--gscale`, default 1024; the network divides the
  parameter gradients by it); on a non-finite gradient norm the trainer skips the step and halves the scale.
  Per-layer precision: `nn_set_layer`/`nn_set_prec_policy` ("enc0=1,dec0=2,..." or positional, `--policy`,
  `UFSM_PREC_POLICY`) let fp8/fp4 be mixed per conv with bf16; measured on random inputs, any single fp8 layer costs
  6-13% gradient error for at most 4% speed, and all-fp8 costs 21% for 18%, so no mixed fp8 policy beats fp16 at
  equal speed. The fp8/fp4 kernels still require bf16 storage (they force it). On the synthetic sheet task (`build/train_lp`, 300
  steps) fp32, bf16 and fp16 reach the same held-out loss (0.2487 / 0.2492 / 0.2485) and the fp16 loss curve tracks fp32
  step for step; at 600 steps with two seeds, fp32 / bf16 / fp16 / fp8 reach 0.159-0.162 / 0.158-0.160 / 0.158-0.162 /
  0.158-0.159, i.e. indistinguishable within seed noise, so the gradient-error metric is a much stricter yardstick than
  this task; the real data will decide whether fp8's 21% gradient error matters.

- Quantization-aware and low-precision-weight training (2026-10-01): `--qat 2|3` runs forward and backward-data at fp8
  / fp4 with 16-bit weight gradients (`nn_set_prec_wgrad`); `--wq 8|4` keeps the 3^3 conv weights in packed e4m3 / e2m1
  storage with one ue8m0 scale per 32 input channels, and AdamW and the EMA update the packed values directly with
  stochastic rounding (`nn_wq_adamw`, `nn_wq_ema`); the fp32 arrays are dequantized shadows for the kernels, so the
  checkpoint holds exact grid values. `--sparse24 STEP` prunes every group of 4 input channels to its 2 largest weights
  (the hardware 2:4 pattern) with SR-STE (`--srste`); the dense kernels run on the masked copy because the convs are
  staging-bound and a sparse MMA would not help today. Synthetic sheet task, 600 steps, two seeds (dense 0.156 / 0.165):
  QAT fp8 0.159 / 0.161, plain fp8 0.157 / 0.165, QAT fp4 0.169 / 0.166, 2:4 sparse 0.179 / 0.167, fp8 weights
  (stochastic rounding) 0.153-0.163, fp8 weights + QAT fp8 0.156 / 0.158. Direct fp4 weights with plain stochastic
  rounding do not train (one mantissa bit: the rounding noise dominates the updates). With an fp8 error-feedback residual
  per weight (`w = q4 s + r8 sr`, the kernels see only q4; residual and scales live on the optimizer side) they train to
  0.221 / 0.223 (QAT fp4 on top: 0.226 / 0.232), i.e. fp4 weights cost ~0.06 held-out loss on this task where fp8 weights
  cost nothing; fp4 nibbles are stored block-contiguously (16 bytes per block of 32) so one thread owns each byte.

- First real run (2026-10-01): run r1 on the MANBp HF labels + Kaggle cubes plateaued at validation 0.76 because the
  sampler drew MANBp positions uniformly over a 17148 x 12577^2 masked volume and rejected 98% of them as air, so 98% of
  the patches were Kaggle cubes while validation came from the MANBp holdout. The sampler now reads the coarsest
  recto level of every pyramid source once (MANBp: 0.3 M of 83 M level-5 cells contain surface) and draws positions
  around random cells that contain surface; the mix is then ~50/50 and run r2 trains at bce 0.28 instead of 0.37 after
  600 steps. Label/CT alignment was verified: CT is 30-40 gray levels brighter on labeled-surface voxels than average.
  Sampling MANBp at level 0 over HTTPS is CPU/decode bound with the exports running (GPU waits ~50% with 24 workers).
- Run r2 (fixed mix) still sat at the constant prior (bce 0.23 = entropy of the 5% positive rate). Diagnostics, in
  order: a fixed batch is memorized in 100 steps (pipeline and gradients fine); 8 fixed batches are learned, 32 fixed
  batches escape the plateau only after ~900 steps, streaming never does within 1200; precision, augmentation, the
  asynchronous upload, learning rate, weight decay, positive weighting and level choice change nothing; 2-D and 3-D
  cross-correlation of labels against CT shows no consistent global offset; zoomed overlays of raw cubes show the HF
  "surface" label is the recto FACE of a sheet (the papyrus/air edge on the side facing the axis), aligned with the
  CT, with the labels' own ignore region on the far side. So the task needs the scroll-axis channels (MANBp had no
  umbilicus: `ufsm axis` now provides a centroid axis under gt/axis/, picked up by make_sources) and is an edge task,
  not a sheet mask. New sampler option `trust_band` (R voxels: background counts only near an annotated surface) for
  partially annotated sources; new trainer diagnostics `--overfit N` (cycle N fixed batches), `--noaug`,
  `UFSM_SYNC_UPLOAD`, `UFSM_DUMP_BATCH`, `--pos-weight`, `--dilate D` (thicker target band as a curriculum).
- Learnability: a logistic regression on hand-made local features (smoothed intensity, gradient along the radial
  direction, gradient magnitude) trained on two raw cubes and tested on two others reaches only ~2x chance precision,
  the same as the network after 4000 streaming steps, so the recto-face target is genuinely hard at the patch level and
  needs the long schedule (nnU-Net-style); short runs cannot show progress. Run r3 (all five exported sources, computed
  axes for every scan, trust band 8 on the HF labels, 128^3 batch 2) stayed at validation 0.83 for 10k steps.
- What finally moved: a soft ridge target (`--soft 3`: background near the surface gets 254 exp(-(d/3)^2/2) with d the
  chamfer distance, so every voxel of the valid band carries gradient) AND no symmetry augmentation. With both, a
  2000-step MANBp run reaches training dice 0.63 (still falling) and holdout precision 0.14 at recall 0.27 (band
  precision 0.34, 3x chance); with the 48-symmetry augmentation the same run stays at 0.10. Reflections flip the
  handedness that distinguishes recto from verso, but the 24 proper rotations hurt just as much (0.08), so the
  spatial symmetry augmentation is off for now (`--noaug`); z-fixed symmetries also cost (0.13 at recall 0.18) while
  intensity jitter alone is harmless or better (`--intonly`: 0.18 at recall 0.17, F1 0.175 at 2000 steps), so the next
  run uses intensity-only augmentation. Without it, 4000
  steps on MANBp reach holdout precision 0.17 at recall 0.34 (F1 0.23, band precision 0.39) and are still improving. `ufsm prefetch` warms the CT chunk cache for every labelled cell (5.2 GB for the four pyramid sources at
  levels 0-1, 32 connections, ~27 MB/s), after which training is decode-bound instead of network-bound. Run r4 (r3 settings + soft target + no augmentation, 30k steps) reached
  validation 0.827 (from 0.93) and holdout F1 0.19 (precision 0.17 at recall 0.20; band precision 0.39 at recall 0.29),
  about 4x chance; the curve was still descending when the cosine schedule ended. Run r5: eight sources (adds PHerc1667
  and the three PHerc0139 winding-range zarrs), intensity-only augmentation, 40k steps.

### Memory for larger windows (2026-10-01)
- Recompute level 1 is the default (`UFSM_RECOMPUTE=0` restores the stored block outputs): 0.87 -> 0.60 GB at
  96^3 batch 2, same 0.25% gradient error, ~20% slower per step on a shared GPU (measurements contaminated by the
  concurrent run; redo on an idle GPU). Level 2 (shared a1, conv1 re-run in the backward) 0.50 GB.
- Trainer-side buffers were a fifth of the level-0 bytes: the sampler now emits the batch in the network's 16-bit
  storage type (`sample_cfg.xfmt`, `batch.x16`, F16C on the host) and the trainer uploads it straight in as the
  input (`unet_forward_x(.., x_h16 = 1)`: no fp32 double-buffered upload, no device-side conversion copy); the loss
  kernel writes the logit gradient as 16-bit scaled by the gradient scale (`nn_set_loss_grad_h16`,
  `unet_backward_x(.., g_h16 = 1)`). `UFSM_X32=1` restores the fp32 path; on a fixed batch both give the same losses
  to run-to-run noise. Per-process peak (fp16 mode, recompute 1, batch 1): 192^3 2.8 GB, 256^3 6.4 -> 5.9 GB,
  320^3 12.3 GB before the trim. Per-voxel accounting: see "Per-voxel memory" below.
- Agent round 3 (merged 9b6f54a, 53abc70): decoder conv1 reads the upsampled part straight from the coarse block output
  (trilinear upsample fused into the 16-bit conv staging, `UFSM_FUSED_UP`), and in training the up-part input gradient
  goes through B in w[i]-channel chunks that are upsample-backwarded straight into the coarse gradient
  (`UFSM_CHUNK_UP`). 96^3 B2 fp16: training 0.596 -> 0.540 GB at 33.8 ms (recompute 2: 0.447 GB at 41 ms),
  inference 0.341 -> 0.228 GB, gradient error unchanged (0.25%). Decoder conv1 always runs the 16-bit kernels now, even
  under an fp8 policy (per-voxel accounting below).
- Intermittent sampler stall (1 run in ~10 on the kaggle source, two workers at 100% CPU, the trainer waiting
  forever): the lazy per-level opens in `sources.c` (`source_ct`, `source_tgt`, `source_region`) were called from every
  worker without a lock; two workers racing on the same level both opened it and a failed open under the race marked
  the level absent for the run, after which every draw was impossible. One mutex around the opens: 0 stalls in 40 runs.
- Inference: the CT window goes up as uint8 and the input channels (z-score, radial unit vector) are built on the
  device in the 16-bit storage type (`nn_pred_input`, read in place by `unet_forward_x`); the recto probability comes
  back as uint8 (`nn_pred_output`). Output identical to the host path. Default window 288 (halo 16: 70% of the voxels
  are interior against 51% at 160; 512^3 box at level 1: 5.4 -> 3.8 s on the shared GPU; metrics equal within noise).
- Inference tile reads run on a reader thread (3-window ring) overlapping the network: a 1024^3 box at level 0
  (58 tiles of 288^3) takes 19 s on the shared GPU against 21 s, i.e. close to the ~16 s of pure network time at
  ~88 Mvoxel/s; a 512^3 box at level 1 is dominated by start-up and writes (3.7 s).
- A patch must fit the sources: the sampler now warns after 4M consecutive impossible draws (region, holdout box
  or level too small for P) instead of spinning silently, and stops after 20 consecutive read failures (the
  failure counter was reset on every error before, so an I/O error retried forever).

### Per-voxel memory (master b1862ff, 2026-10-01)
Bytes per full-resolution input voxel per sample, model (16,32,64,80), 16-bit storage (fp16 or bf16), recompute 1,
fused upsample, chunked B, NCH = 2 output channels. A level-l tensor of C channels costs 2C / 8^l B. down_norm only
adds per-group statistics. Checked against `bench_mem` at 96^3 B2: training 0.540 GB = 305 B/voxel, inference
0.228 GB = 129 B/voxel (bench_mem feeds fp32 input and NCH = 1, so it adds the unet's own 16-bit input copy, 8 B,
and logit-gradient copy, 2 B, and has 4 B less logits).

Training (unet + trainer):

| level (C) | tensor | B/voxel | note |
|---|---|---|---|
| 0 (16) | enc0 a1, a2; dec0 a1, a2 | 4 x 32 = 128 | conv outputs before GN; kept for backward |
| 0 | gradient A, B, gout[0] (= gskip[0]) | 3 x 32 = 96 | shared across levels, sized at level 0 |
| 0 | logits (fp32, 2 ch) | 8 | |
| 0 | trainer: input x16 (4 ch, double-buffered) | 16 | `gpu_state.xb[2]` |
| 0 | trainer: targets (2 x uint8) + mask, double-buffered; logit gradient (16-bit, 2 ch) | 6 + 4 | |
| 1 (32) | enc1, dec1 a1/a2 (4 x 8); dec1 s2 (upsample source) 8; down0 out (16 ch) 4; gout[1] 8 | 52 | |
| 2 (64) | enc2, dec2 a1/a2 (4 x 2); dec2 s2 2; down1 out 1; gout[2] 2 | 13 | |
| 3 (80) | enc3 a1/a2, s2, down2 out, gout[3] | 1.5 | |
| | total | ~325 | plus fixed: CUDA context, params + EMA + Adam (5 x 4.7 MB), conv scratch |

The lead's per-process peak at 256^3 B1 was 5874 MiB (6.16 GB). The table gives 5.45 GB, so the fixed part is about
0.7 GB at that size.

Inference (one shared-buffer build, same storage):

| tensor | B/voxel | note |
|---|---|---|
| T1: every a1 | 32 | sized at level 0 |
| T2: the a2 that are not kept (decoder) and the down-conv outputs | 32 | |
| enc0 a2 (skip; GN applied in the decoder's staging) | 32 | |
| enc1 / enc2 / enc3 a2 | 8 + 2 + 0.3 | |
| kept upsample sources: dec1 s2, dec2 s2, enc3 s2 | 8 + 2 + 0.3 | |
| input (16-bit, 4 ch) | 8 | |
| logits fp32 | 4 x NCH | `predict` turns them into uint8 on the device |
| total | ~125 + 4 NCH | |

Where the rest would come from:
- **Recompute 2:** saves 52 B/voxel in training (0.447 GB at 96^3 B2) for +7 ms per step. Every a1 shares one
  32 B buffer, and the backward re-runs conv1.
- **The level-0 gradient buffers (96 B):** A, B and gout[0] each hold 16 channels. They are live at the same time
  during dec0's backward, so going lower needs a fused GN-backward + conv-backward-data or chunked channels.
- **MX-fp8 storage:** about halves every activation term. That is 1.03 B per value for C >= 32 and 1.06 for C = 16,
  against 2. It is opt-in: about −0.008 holdout F1 at 6000 steps; MX gradients cost −0.018.
- **Logits:** 16-bit logits would save 4 B per output channel.
- **Trainer input buffer:** going single-buffered saves 8 B/voxel, at the cost of upload/compute overlap.

### Low-precision storage on real data (agent round 2, 2026-10-01)
- Yardstick: 2000 steps on MANBp (`--soft 3 --noaug 1 --P 64 --B 8`), holdout F1 at threshold 0.5. fp16 twice:
  0.215 / 0.218; MX-fp8 activations twice: 0.216 / 0.218; MX activations + gradients (after the split-output fix
  907cb33): 0.220 (0.210 before the fix); fp8 compute with 16-bit storage 0.221; simulated NVFP4 0.218; simulated
  MXFP6 0.215. Every correct mode sits inside the fp16 noise band, so 2000 steps cannot separate them. At 6000
  steps (one run each) a gap opens: fp16 F1 0.269, MX activations 0.261, MX activations + gradients 0.251
  (band-tolerant 0.393 / 0.381 / 0.363; recall at 0.7 down 0.05 with MX gradients). Decision: fp16 storage +
  recompute 1 stays the default; MX activations (`UFSM_ACT_MX8`, 0.46 GB at 96^3 B2) are an opt-in memory trade
  worth ~3% F1; MX gradients (`UFSM_GRAD_MX8`, 0.29 GB) stay off. The agent no longer recommends MXFP6
  (saves ~0.05 GB; MX gradients save more and exist already); NVFP4 storage was clearly worse on the synthetic task.

### fp16 activation overflow (2026-10-01, run r6)
- r6 started skipping steps at 16k with "non-finite gradient norm" and halving the gradient scale down to 1, which was the
  wrong reflex: the loss parts were NaN, i.e. the forward itself was non-finite. The trainer now tells the two apart
  (`non-finite forward` keeps the scale, dumps the batch to `<out>/nan_step<N>.bin`); `tests/fwd_nan.c` replays such
  a dump through a checkpoint and prints per-block GroupNorm statistics and stored-activation extremes
  (`unet_debug_stats`, `unet_debug_acts`).
- Cause: two un-normalised convs in series (stride-2 down conv, then the next block's conv1) amplify the O(1) block
  output to O(10^3..10^4) before the GroupNorm; at the deepest level with r6's weights 14 voxels of one sample
  exceeded 65504, the fp16 finite maximum, the stored activation became inf and the sample's statistics NaN. The
  exact fp32 kernels were finite; r5's weights left 4x more headroom (a1 ~250 at enc3 against ~1000).
- Fix: the tensor-core epilogue saturates fp16 activation stores that feed a GroupNorm (the statistics use the stored
  value); gradient stores are untouched so the overflow detection keeps working. GroupNorm is scale-invariant, so the
  clamped voxels only lose a little precision. The replayed batch now matches the exact kernels. A GroupNorm after
  each down conv would remove the amplification altogether but changes the model (candidate for a later run).

### fp8 training on real data (agent, 2026-10-01 afternoon)
- 6000-step MANBp yardstick (AdamW, down_norm 0, seed 0), F1 at 0.5 / 0.7: all-fp8 GEMMs (`--prec 2`) 0.247 / 0.257,
  fp8 weight gradient only 0.241 / 0.232, fp8 forward + backward-data (`--qat 2`) 0.248 / 0.236; fp16 0.269 / 0.282
  on an earlier binary and 0.253 / 0.225 for two seeds on the current one; validation loss 0.845 for every fp8 mode
  and for fp16. The fp8 modes sit inside the fp16 seed band (+-0.02), so single runs cannot resolve a 0.005 target.
- Kernel work (branch fp8train): stochastic rounding for every fp8 quantisation of a gradient operand (per-element
  hash, per-step seed; `UFSM_SR=1`), e5m2 gradient operands (`UFSM_GFMT=e5m2`). One-step error: e5m2 is WORSE than
  e4m3 under the per-32 block scale (the scale already covers the range, e5m2 only loses a mantissa bit): fp8 weight
  gradient 2.92% e4m3 vs 4.29% e5m2; SR removes part of the bias (2.47% averaged over 8 seeds). The forward pass
  dominates the all-fp8 error (23% vs 9% with a 16-bit forward). Decision: e4m3 + SR; next a long paired run
  (identical batch order, fp16 vs all-fp8+SR, Muon) after r10.

### Sampler throughput, fp4 inference, Muon (2026-10-01 afternoon)
- The trainer had become data-loader bound (r9 fell to 8-14 samples/s with the sampler waiting 75-90%). Per-stage
  profile (`UFSM_SAMPLER_PROF=1`, `tests/bench_sampler.c`), ms of worker time per 128^3 patch on the loaded box: CT read
  337, intensity noise 105 (Box-Muller per voxel), soft-target chamfer 85, trust-band max filter 82, 16-bit convert 14,
  z-score 13. Fixes: Gaussian noise from a 4096-entry per-worker table (105 -> 29), trust band derived from the same
  3-4-5 chamfer distance as the soft target (82 -> 4, chamfer computed once per channel), windows snapped to the CT
  chunk grid (`sample_cfg.snap`, on by default: the CT chunks are 128^3, so a 128^3 window decodes 1 chunk instead of up
  to 8). 16 workers: 44 -> 87 samples/s under load. The remaining read cost was cache misses: 41% of chunk reads went
  to the network (`z3_io_stats`), because the old prefetch only cached chunks containing surface at levels 0-1. New
  prefetch (`z3_prefetch_chunk`, no decode): per level, the chunk set covering every window around every occupied cell
  (P/2 offset, snapped, neighbours) plus the level+3 probe windows, deduplicated in a bitmap; `--levels 3` covers the
  training levels. A cached, aligned 128^3 window decodes in 4.6 ms (`tests/bench_read.c`).
- fp4 inference: the fp16 fine-tuned MANBp model scores F1 0.297 (0.5) / 0.277 (0.7) at fp16 inference and 0.295 /
  0.271 with `predict --prec 3` (e2m1 activations and weights, block scales): fp4 inference of an fp16-trained model
  costs 0.002 F1. Training with packed fp4 weights (`--qat 3 --wq 4`) costs 0.05 F1 (0.242) and is not needed for fp4
  deployment, since prec 3 quantises the weights per block on the fly to the same values.
- Muon (modded-nanogpt): `--opt muon` runs nesterov momentum + 5 Newton-Schulz iterations on each 3^3 conv weight
  viewed as [Co][Ci*27] (`nn_muon`, `tests/test_muon.c`), AdamW on biases, GroupNorm and the head, lr 0.02 following
  the AdamW schedule shape. Kaggle smoke run, 600 steps: train loss 0.41 vs 0.59, validation 0.54 vs 0.63 for AdamW,
  at ~15% more step time (naive small-matrix kernels). Real-data yardstick (MANBp, P64 B8, 6000 steps, down_norm,
  same seed): peak F1 0.296 at threshold 0.6 (band F1 0.476 / recall 0.584) for Muon against 0.199 at 0.6 for AdamW
  (0.362 / 0.366); validation loss 0.872 vs 0.898. Adopted for new runs (`--opt muon`); lr sweep 0.01 / 0.02 / 0.05
  done: peak F1 0.307 (lr 0.01), 0.296 (0.02), 0.290 (0.05); validation loss 0.870 / 0.872 / 0.866. Note the AdamW
  arm scored below the agent's earlier AdamW reference (0.262 at 0.5), so seed variance is large at 6000 steps, but
  the Muon margin is far outside it. Step cost: +11-15% at 64^3-96^3 (255 small launches per step), ~5% at 128^3.
- r10 (2026-10-01 13:10): configs/all2.json (17 sources, full chunk prefetch, 1.1 um scan from level 2), down_norm,
  Muon lr 0.01, `--sched wsd --cooldown 0.2`, 200k steps at 128^3, effective batch 2 on both GPUs. Scored against
  the r8 baseline on all 17 held-out boxes when done.
- Prefetch and training levels: scans finer than 1.8 um (PHerc0139 1.129 um) get `min_level` 1 from make_sources
  (their level 0 alone would be 1.5M chunks); the sampler and the prefetch honour it.

### Accuracy diagnosis (2026-10-01 morning)
- Held-out scores of r5 (8 sources, soft sigma 3, intensity-only augmentation, 17k of 40k steps) against r4: F1 at
  threshold 0.3 per source 0.18/0.06/0.17/0.17/0.065/0.045/0.78 vs 0.19/0.05/0.16/0.16/0.06/0.045/0.77 (level 1);
  level 0 gives the same picture (MANBp 0.20 at 0.5). `ufsm eval --dump` (CT | prediction | label slice) shows why:
  the prediction sits on the labelled sheets (86% of the voxels above 0.5 are within 5 voxels of a label line, 62%
  within 2) but as a wide, faint ridge (mean probability 0.33 on label voxels, ridge ~4 voxels wide against a
  1-voxel label line at level 1), plus a visible tile grid from the per-window GroupNorm statistics. Window 288
  instead of 160 changes F1 by < 0.002, so the tiles are not the main loss; the softness is.
- Hypothesis tested (run r6, 40k steps, sigma annealed 3 -> 1, resumed at 16.6k after the fp16 overflow fix): the
  softness is NOT the limit. At the standard thresholds r6 is worse everywhere (precision up, recall collapsed: the
  sharper target shifts the calibration down); at each run's best threshold (`ufsm eval --thr 0.05,...`) it is a
  wash: F1 MANBp 0.183 vs 0.160, 0343P 0.159 vs 0.180, 1667 0.170 vs 0.166, 0500P2 0.059 vs 0.060 (r6 vs r5). The
  plateau (~0.17 F1, ~0.4 band F1) stands with this model and these labels.
- r7 = r5's config with `--down-norm 1` (GroupNorm + SiLU after each stride-2 down conv, agent commit cf65220), on
  both GPUs at effective batch 2 (30 min for 40k steps, 44 samples/s). Best validation loss 0.862 against r5's
  0.881 on the same validation target, no overflow risk (enc3 a1 peaks ~30 instead of ~250..65000). Held-out F1 at
  0.3 / 0.5 (r7 vs r5): MANBp 0.180 / 0.213 vs 0.160 / 0.192, 0343P 0.172 / 0.043 vs 0.180 / 0.037, 1667 0.160 /
  0.040 vs 0.166 / 0.053, 0500P2 0.062 vs 0.060, PHerc0139 0.066 / 0.073 vs 0.071 / 0.076: a clear gain only on
  MANBp (the agent's yardstick source), a wash elsewhere. down_norm is adopted for new runs on stability and
  validation loss; it is not the accuracy breakthrough. r7's validation loss was still falling at 33k, so r8 = r7
  with 120k steps tests whether the plateau is simply under-training.
- r8 (120k steps, 1.7 h on both GPUs, data-loader bound at ~38 samples/s): IT WAS under-training. Best validation
  loss 0.805 (r7 0.862, r5 0.881). Held-out F1 at 0.3 / 0.5, r8 vs r7: MANBp 0.226 / 0.286 vs 0.180 / 0.213
  (band F1 at 0.5 ~0.50), 0343P 0.201 / 0.121 vs 0.172 / 0.043, 1667 0.189 / 0.055 vs 0.160 / 0.040, 0500P2 0.071
  vs 0.062, PHerc0139 0.075 / 0.084 vs 0.066 / 0.073 and 0.043 vs 0.044, PHerc0009B unchanged. Every HF source
  improves; precision and recall both rise. Next: r9 = r8 with 300k steps (same cosine schedule stretched), and the
  sampler must get faster (the trainer waits ~50% of the time at 2 GPUs). Its validation batches keep sigma 3 (they are materialised at start), so the validation
  loss drifts up as the training target sharpens and `best.ckpt` is an early checkpoint: score `last.ckpt`.

## Status (2026-09-30 night)
- M0 reader + CLI + sampler: done, tested (bit-exact zarr3 reads, sampler montages checked by eye).
- M1 ingest: done — `ingest-zip` (label archive), `ingest-kaggle`, `ingest-mesh`, `raster`, writer, zip and
  TIFF readers all tested; exports running under `/vesuvius/ufsm/gt` (`tools/ingest_all.sh`,
  `tools/surfcomp_all.sh`). Labels.zip is the only practical path: HF rate-limits per-file requests.
- M2 CUDA ops + UNet: done — every op and the whole network pass finite-difference checks
  (`make test`); the 1.17 M model trains at 0.41 s/step for 96^3 on the shared GPU (1.4 GB).
- M3 training loop: done and smoke-tested; held-out boxes per source feed validation. First real run
  starts once the first label pyramids finish (`tools/train_r1.sh`).
- M4 predict + M5 eval: implemented and exercised on the Kaggle smoke model; the real evaluation
  waits for a trained checkpoint.
- Both GPUs are shared with other jobs on this box (≈2.5 GB free), hence P=96, B=1.
