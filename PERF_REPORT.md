# Latent-grid training and inference: performance report (2026-11-09)

Branch `perf-20261109` (from `latcomp-20261109` at 31df896), worktree `/home/forrest/ufsm-perf`. One RTX 5060 Ti 16 GB (GPU 1).
Recipe: `train --lat 4 --task unwrap2 --rv 1`, widths 32,64,128,192,256, MX-fp4 activations, 11 output channels, `--gpu-aug 2`,
label planes streamed per head slab; `predict --channel rv --shard 128 --window 640 --halo 64 --threads 8`.

How to read the numbers:
- **measured, idle GPU**: GPU 1 had no other process. Only the first two training columns below.
- **measured, shared GPU**: a production trainer (k768) held GPU 1 for the rest of the session and the machine ran at load 20-35.
  These numbers are 1.3-2x slower than idle and noisy; they are used for ratios, or the best 5-step window is quoted.
- **estimated**: scaled or derived, never timed.

## Result in short

| | before | after | how |
|---|---|---|---|
| train P768, one repeated batch, GPU side | 1307 ms/step, 347 Mvox/s | 602 ms/step, 752 Mvox/s | measured, idle GPU, without the upload pipeline |
| same, with the slab upload pipeline | | <= 487 ms/step, >= 930 Mvox/s | best 5-step windows on the shared GPU (930, 932, 917, 914) |
| same, median on the shared GPU | 409 Mvox/s (pipeline off) | 540 Mvox/s (pipeline on) | measured, shared GPU, alternating runs |
| predict, box 1024 x 3072 x 3072 (56 windows of 640^3) | 66.0 s | 16.5 s and 17.8 s | measured back to back, shared GPU, load 20-24 |
| same at a quieter moment | 68.1 s (not quiet) | 7.55 s | measured, different load: not a clean pair |
| predict device memory per window voxel | ~19 B + trunk | ~1.2 B + trunk | estimated from the buffers removed |

Training with the real loader is **not** faster in practice yet: the sampler supplies about 17 Mvox/s per worker core, so the GPU
side is no longer the limit (see "The loader").

## 1. Training: per-phase time of one step

P768, `UFSM_BENCH_SAME=1 UFSM_PTICK=1` (one batch repeated, the device synchronised after every phase), ms per step, measured on the
idle GPU. "after" is commit ec679e1 (separable upsampling and adjoint), before the upload pipeline.

| phase | before | after | P960 est. (x1.95) | P1280 est. (x4.63) |
|---|---|---|---|---|
| input pooling (`nn_input_lat`) | 16.2 | 16.2 | 32 | 75 |
| trunk forward | 30.1 | 30.2 | 59 | 140 |
| pass 0: label slab uploads | 74.8 | 73.0 | 143 | 338 |
| pass 0: upsampling | 302.5 | 29.0 | 57 | 134 |
| pass 0: targets | 41.8 | 41.7 | 81 | 193 |
| pass 0: loss statistics | 40.0 | 39.9 | 78 | 185 |
| pass 1: label slab uploads | 73.1 | 84.1 | 164 | 389 |
| pass 1: upsampling | 302.0 | 29.3 | 57 | 136 |
| pass 1: targets | 41.8 | 41.7 | 81 | 193 |
| pass 1: loss gradients | 78.7 | 78.4 | 153 | 363 |
| pass 1: adjoint gather | 220.0 | 52.3 | 102 | 242 |
| trunk backward | 72.7 | 73.1 | 143 | 338 |
| optimiser | 11.8 | 11.8 | 12 | 12 |
| other | 1.4 | 1.4 | 1 | 1 |
| **total** | **1306.9** | **602.1** | **1163** | **2739** |
| Mvox/s | 347 | 752 | 761 | 766 |

The P960 / P1280 columns are the "after" column scaled by the voxel count (every phase but the optimiser works per voxel); they are
estimates. The phase timer cannot show the upload pipeline: its synchronisation waits for the next slab's upload too.

Per-batch upload work, which a repeated batch skips (measured on the shared GPU with `UFSM_SYNC_UPLOAD=1`, P768, ms per batch; the
other phases of that run were ~1.5x their idle times, so the idle estimates are about two thirds):

| phase | measured, shared GPU | idle estimate |
|---|---|---|
| label planes: upload, device warp, download to the pinned host planes | 247 | ~165 |
| CT upload + gpu-aug (`nn_aug_ct`) | 237 | ~160 |
| other uploads | 1.0 | 1 |

In a normal run these go through the copy stream under the step, but their kernels and PCIe traffic still cost GPU and bus time,
so the real step is longer than the repeated-batch 487-602 ms.

Where the time is now (after, 602 ms): label slab uploads 157 (hidden by the pipeline when the bus is free), loss kernels 118,
trunk 131, targets 83, upsampling 58, adjoint 52. Every remaining full-resolution kernel is bound by memory traffic: a P768 pass
writes 22 B a voxel of fp16 logits (10 GB in 29 ms), the gradient pass reads and rewrites them plus 14 B of targets.

## 2. Implemented

| commit | change | effect | verification |
|---|---|---|---|
| ec679e1 | `nn_lat_up_slab`: 3 taps per axis, one thread per coarse cell, separable, 8-byte row stores | 302 -> 29 ms per pass (idle) | test_lat; `UFSM_LAT_CHECK=1`: 6.4e3 of 1.48e9 fp16 logits differ by one code |
| ec679e1 | `nn_lat_up_adj`: gather in two separable stages through the slab's normal-target buffer | 220 -> 52 ms (idle) | test_lat (adjoint and dot products); check: rms difference 1e-7 of rms |
| ec679e1 | `UFSM_PTICK=1` phase timer (train, predict), `UFSM_LAT_CHECK=1`, `UFSM_LAT_SLOW=1` | | |
| e5e2e59 | label slab uploads one slab ahead on the aux stream, second buffer set (`UFSM_LAT_PIPE=0` off) | 409 -> 540 Mvox/s median, best windows 930 (shared GPU) | `UFSM_LAT_STREAM=check`: 0 of 8.7e9 bytes; losses identical on and off (25 steps, one worker) |
| 268a84e | predict: `nn_lat_place_rv`, cell logits straight to shard bytes | no upsampling (25-28 ms), fuse (14-16 ms), fp32 volume (8 B/voxel) per 640^3 window (shared GPU) | 1139 of 4.6e8 nonzero bytes differ by 1, masks equal |
| 268a84e | predict (all nets): place kernels over the interior-shard intersection | 64 launches over 512^3 voxels -> over 128^3 each | output files identical |
| 268a84e | predict: up to 8 writer threads when a shard has fewer chunks than `--threads` (`UFSM_PRED_WRITERS`) | 29-window box 8.1 -> 4.7 s (shared GPU) | output files identical |
| 8931e9d | predict: `nn_lat_pred_input`, cell-grid input straight from the CT bytes | input build + pooling 30 -> 3 ms per window (shared GPU); no 8 + 2 B/voxel | output bytes identical |
| 517a5cc | headers, test_lat predict cases, DESIGN.md, this report | | test_lat 118 cases, test_gpu_input, test_gpu_aug pass on GPU 1 |

Losses on a repeated batch: step 1 matches to the printed digits (1.6863 both, gn 7.14 against 7.12) between the old and new slab
kernels; later steps drift (1.6860 against 1.6889 at step 2). The kernels themselves agree to one fp16 code and 1e-7 (the direct
check above), so the drift is the trunk amplifying rounding, not a different computation. The upload pipeline changes nothing: the
loss lines are identical.

Not run: the fast suite (`make test`). It takes the machine-wide GPU locks and GPU 0 held a production run. It has to pass before
a merge. The predict baseline-vs-new comparison on the large box: 1170 of 4.9e8 nonzero bytes differ by 1 over 128 planes.

## 3. Predict: per-phase time of one 640^3 window

ms per window, `UFSM_PTICK=1`, 29-window box (128 x 3136 x 3232), all measured on the shared GPU. Last column: one run of the
56-window box when GPU 1 was nearly idle.

| phase | before | after | after, quiet moment |
|---|---|---|---|
| upload + CT statistics | 18.8 | 22-34 | 10.5 |
| input build (`nn_pred_input_d`) | 14.5 | 0.8 | 0.02 |
| input pooling | 14.1 | 1.8 | 0.7 |
| trunk forward | 41.7 | 34-41 | 20.0 |
| upsampling to fp32 | 27.5 | 0 | 0 |
| rv fuse | 15.8 | 0 | 0 |
| place kernels | 33.3 (16 shards; already over the intersection) | 15-28 (16 shards) | 9.7 (64 shards) |
| shard download | 3.6 | 3-5 | 7.4 (64 shards) |
| waiting for the writer | 74-112 | 0.2-5 | 81 (64 shards per window) |

The original binary was not phase-timed: its place kernels, shard downloads and writer waits were one lump of 100-140 ms per window in
the first instrumented build.

Read + decode is the reader thread's (8 threads): 3.8-7.4 s for the 29-window box depending on machine load, 6-17 s for the
56-window box. In the quiet run the GPU work was 48 ms per window (5.4 Gvox/s of window volume, 2.8 Gvox/s of interior); the run
was bound by the 8 writer threads (volcomp q 0: 147 MB/s per thread when the machine is quiet, 68-78 MB/s under load) and the reader.

## 4. Evaluated and not adopted

- **One fused target kernel** (phase, side, packed masks, verso weight, normals in one pass): 14-17 ms against 14 ms per P512 pass,
  no gain within noise; the six normal planes differed by one code in 2e-4 of the voxels. Removed.
- **Larger slabs** (`UFSM_HEAD_SLAB_Z=64`): medians 562 against 548 Mvox/s on the shared GPU (+2.5%, within noise), +0.6 GB at P768.
- **Skipping all-masked slabs**: 0 of 128 slabs in 8 real windows had no supervision; 51-94% of the voxels carry the side loss,
  22-92% the phase, 19-75% the normals. Nothing to skip in this recipe.
- **Fusing the two slab passes**: the BCE heads need only mask counts before the gradient (label-only), but the winding mod 4 head
  picks its gauge per component from the full statistics, so a pass over two logit channels and all label planes stays. Not built.
- **Larger default predict window**: tried k = 8 shards (1088^3). The run is writer- and reader-bound, and big windows skip less
  air (4608 shards compressed against 3584 for the same box). Left at k <= 4; `--window 1152 --halo 64` works when wanted.
- **`--window-z` for latent checkpoints**: not done. It needs the per-shard (non super-shard) tiling path; it only matters for thin
  test boxes, whole volumes use cubes.

## 5. The loader (measured, not changed: sample.c is the host-memory work's area)

- Steady state (perf attach after step 30, P512, 4 workers, machine load ~25): 7.7 CPU-seconds per P512 window, i.e. ~17 Mvox/s per
  worker core. Stages: augment 4.57 s, ctread 1.05, scan 0.72, targets 0.55, zscore 0.39, ctstats 0.14, x16 0.14, encode 0.14.
  Symbols: `draw_` 18%, memmove + memset 20%, volcomp decode 15.5%, `band_wind4*` 17%, `sym_u8` 7%, `copy_out` 3.7%,
  `sample_pack_tm` 3.6%, `phase_air` 2.9%; kernel 4%.
- 32 cores therefore cap the loader near 550 Mvox/s for both GPUs together (estimate), below one GPU's 750-930 Mvox/s.
- Trainer start: 86% of the CPU samples of the first steps were in the kernel (TLB-shootdown IPIs 32%, page zeroing 7%, page
  migration and compaction ~15%): THP direct compaction on the first touch of the worker scratch (`wbig_alloc` requests
  `MADV_HUGEPAGE`; the system has `defrag = defer+madvise`; free memory was ~7 GB with 112 GB of page cache). A P768 run made
  6-20 Mvox/s over its first 10 steps and 39 CPU-seconds per window over its first 40. Every restart pays it.

## 6. Remaining ideas, ranked

1. **Sampler CPU cost** (the limit now). The winding unwrap (`band_wind4*`, 17%), the window copies (memmove / memset / `sym_u8`,
   27%) and the inline loops of `draw_` (18%). Expected: every 10% saved is 10% more training throughput until ~750 Mvox/s per GPU.
2. **THP compaction at start**: skip `MADV_HUGEPAGE` in `wbig_alloc` when memory is tight, or behind an env switch. Expected: the
   first ~10 steps of each run at normal speed. One line, not measured.
3. **Label round trip per batch** (~165 ms idle estimate at P768): unwarped windows go host -> device -> host only to be uploaded
   again per slab. Streaming their slabs straight from the batch's pinned planes (the batch held until the step ends) removes
   1.8 GB up and 1.8 GB down per unwarped batch. Estimated, depends on the warped share.
4. **Fused slab kernels without materialised logits**: one kernel per pass per coarse cell (logits in registers, targets from the
   label bytes, the gradient reduced inside the cell to 27 partial sums per channel, then a gather). Traffic per voxel falls from
   ~116 B to ~41 B in the gradient pass and to ~4 B in the statistics pass; estimate 278 -> ~90 ms of full-resolution work at P768
   (step ~220 ms). Large: it has to reproduce six target and loss kernels; only worth it once the loader can feed it.
5. **Statistics pass without the 9 BCE channels**: upsample only the 2 winding channels in pass 0, take the BCE normalisers from
   mask counts and the logged BCE sums from the gradient kernel. Estimate -50 to -65 ms at P768.
6. **predict writer**: volcomp q 0 at ~147 MB/s per thread bounds the run; more writer threads (the cap is 8) or skipping the
   compression of all-zero shards. **predict reader**: each 640^3 window re-reads its halo (1.95 voxels read per voxel written).
7. **predict place**: one kernel launch and one download per window instead of one per shard (64 launches at window 640).

## Reproduce

Scripts in `/tmp/claude-1000/-home-forrest-ufsm/5e877b34-3bb5-4c5a-8003-dd19cde6a779/scratchpad/perf/` (session scratch): `rep.sh P`
(the training recipe; `UFSM_BENCH_SAME=1 UFSM_PTICK=1 STEPS=15 W=1`), `pred.sh NAME` (`UFSM_PRED_PROF=1 UFSM_PTICK=1`, `BOX=`,
`WIN=`), `zdiff.sh A B` (voxel differences of two predict outputs through `ufsm read`). Old paths for A/B: `UFSM_LAT_SLOW=1`,
`UFSM_LAT_PIPE=0`, `UFSM_PRED_LAT_DIRECT=0`, `UFSM_PRED_LAT_INPUT=0`, `UFSM_PRED_WRITERS=1`.
