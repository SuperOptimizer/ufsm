# Inference memory plan (`ufsm predict`)

Analysis and implementation plan for cutting the device memory of the inference forward. Written 2026-10-11 against
`origin/master` 2bdcd49; line numbers refer to that commit. Target implementer: a 16 GB RTX 5080 (sm_120, PCIe 5.0 x16).
Training memory work (desk mode, 78.5 B/voxel) is out of scope; this is the inference build only.

## Why it matters

- **Where the time goes.** Production prediction tiles each 512^3 output shard with one window of W = 512 + 2*halo.
  Only the interior is kept, so compute per output voxel is (W / 512)^3.
- **Seams.** Window seams are an edge-context effect of the model, not GroupNorm (DESIGN.md D1). Halo 16 shows clear
  seams; halo 64 roughly halves them.
- **Example.** Measured on the Paris4 ink test block with the 26k-step ink checkpoint:
  - seam ratio (mean |Δ| across the x = 511 tile boundary / elsewhere) 6.2 → 3.25;
  - F1 vs the published ink 0.798 → 0.811.

  | window / halo | compute per output voxel | fits on |
  |---|---|---|
  | 544 / 16 | 1.20x | 16 GB |
  | 640 / 64 | 1.95x | 16 GB (largest that fits; 704 OOMs) |
  | 768 / 128 | 3.38x | 32 GB (5090; 896 OOMs) |

- **What limits the window.** Activation memory grows with W^3.
  - On 16 GB, 640 is the ceiling, so better seams cannot be bought at any price.
  - Turning memory into speed needs a window interior larger than one shard (see "Converting memory into speed").

## Current inference memory (measured)

**Setup.**
- Network: the production 6-level net (widths 32,64,96,128,160,192, cin 4, G 8, down_norm 1).
  - Checkpoints: runs/verso5 recto, runs/ink3d; identical numbers for both.
  - Runtime: MX-fp4 activations, input_prec 8, gn_stats stored, recompute on, reuse_skip on.
- Method: dry inference build (`unet_infer_bytes`, src/unet.c:816) with `UFSM_MEM_REPORT=1`.
- Result: **56.45 B per window voxel at every W**:

| W | 288 | 512 | 544 | 640 | 704 | 768 | 896 | 1024 |
|---|---|---|---|---|---|---|---|---|
| GB | 1.35 | 7.58 | 9.09 | 14.80 | 19.70 | 25.57 | 40.61 | 60.62 |

**Per allocation at W = 640, in bytes per level-0 voxel.** MX-fp4 with 32-channel blocks is 17 B/voxel per 32
channels: 16 B nibbles + 1 B ue8m0 scale. The layout is `data[n][blk][voxel][16]`, then the scale plane `sc[n][blk][voxel]`
(src/lp_common.cuh:80-86).

| buffer | B/voxel | what | lifetime in the forward |
|---|---|---|---|
| T1 | 17.00 | shared a1 scratch, sized for level 0 (32 ch) | every block's a1; also predict's 16-bit input `xd` (unet_input_scratch, src/unet.c:802) before the stem |
| T2 | 4.25 | down outputs and non-skip a2, sized for level 1 (64 ch at 1/8) | reuse_skip makes level 1 its largest user |
| enc[0].a2 | 17.00 | level-0 skip; dec[0].c2 writes its output here (reuse_skip, src/unet.c:530-540) | enc0.c2 → head |
| enc[1..5].a2 | 4.25 + 0.80 + 0.13 + 0.02 + 0.003 | coarse skips | encoder → matching decoder |
| logits | 4.00 | fp32, one channel (`unet_set_head_channels`) | head → `pred_place` |
| xin | 9.00 | MX-fp8 stem input: 4 channels padded to an 8-channel block, 8 + 1 B (`nn_mx8_bytes`, nn_core.cu:192) | conversion → enc0.c1 only |
| GN stats, misc | ~0 | | |
| **total** | **56.45** | | |

**predict.c allocates on top of this** (src/predict.c:305-318 at 2bdcd49):
- `ctd`, the uint8 CT window: 1 B/voxel.
- `dsh`, the uint8 output shard: shard^3, about 0.5 B/voxel at W = 640.
- Two 4·W floats.

Host side are pinned rings: 3 CT windows (W^3) and 2 output shards (shard^3), a few GB. Host RAM is not a constraint.

**Check against hardware.** 640^3 x (56.45 + 1.5) = 15.2 GB, and nvidia-smi shows 14.9 GiB in use on the 5060 Ti.

**Liveness (the real peak).** The allocation is the sum of static buffers. What must actually be live at once is
much smaller:

| point in the forward | live level-0 tensors | B/voxel |
|---|---|---|
| input conversion | ctd 1 + xd 8 (in T1) → xin 9 | 18 |
| enc0.c1 | xin 9 → a1 17 (T1) | 26 |
| enc0.c2 | a1 17 → a2 17 (skip) | 34 |
| coarse levels | skip 17 + level-1 tensors (~4.25 x 2 + 2.1) + deeper (~1) | ~29 |
| dec0.c1 | skip 17 + up source dec1 output 4.25 (T2) → a1 17 (T1) | 38.25 |
| dec0.c2 | a1 17 → a2 17 (over the skip) | 34 |
| head | a2 17 → logits 4 | 21 |

The live peak is ~38 B/voxel, against 56.45 allocated. Getting below ~34 needs a level-0 conv that does not hold its
full input and output at once: Phase 2.

**Where the time goes.** UFSM_PROF=layers, one 512^3 window, RTX 5090, shared with another job, so ratios matter more
than the absolute ms:

| level | convs | ms | share |
|---|---|---|---|
| 0 | enc0.c1 40.8, enc0.c2 36.3, down0 38.5, dec0.c1 91.9, dec0.c2 29.6, head 7.0 | 244 | 68% |
| 1 | enc1 20.9, down1 8.1, dec1.c1 36.2, dec1.c2 10.8 | 76 | 21% |
| 2+ | | 39 | 11% |

Side observations for later speed work, not memory:
- enc0.c1 (4 → 32 channels) costs as much as enc0.c2 (32 → 32) despite 8x fewer MACs.
- down0 costs as much as a full-resolution conv.

Both look like input-staging or bandwidth bound kernels.

## Plan

Every phase must leave the default output unchanged: byte-identical where marked "exact", else within the tolerance in
"Verification". Each phase sits behind an env switch while it is validated, as the training offload modes are
(`UFSM_*=0` falls back), and becomes the default once green.

### Phase 1: aliasing in the inference build (exact, small)

Pure buffer placement in `build_acts` (src/unet.c:509) for `!train && share && reuse_skip`. No kernel changes.

1. **xin into enc[0].a2's buffer (−9 B).**
   - The stem input is dead after enc0.c1, and enc0.a2 is first written by enc0.c2. It fits: 9 ≤ 17.
   - Reuse the training mechanism: `u->xin_shared`, plus the `nn_set_storage` re-registration done at
     src/unet.c:1106 and 1168.
   - The registry must say "MX8 input" while the stem reads, then "MX4 activation" before enc0.c2 writes. Training does
     exactly this dance for `dec[0].a2` (src/unet.c:775-784, 1029-1039).
2. **logits into T1 (−4 B).**
   - At the head, T1 holds dec0.a1, dead after dec0.c2. The head reads dec0.a2, which lives in enc0.a2's buffer, and can
     write fp32 logits into T1 (4 ≤ 17).
   - Register T1 as plain storage for the plane-major fp32 writes, as lean mode does for A (src/unet.c:633-637).
   - `unet_input_scratch` must keep working: it returns enc[0].a1 == T1, which is free at the window start.
3. **predict's ctd into enc[0].a2's buffer (−1 B).**
   - ctd is read by the stats pass and `nn_pred_input` (→ xd in T1), then is dead before the forward converts xd → xin.
   - Expose `unet_input_scratch2(u, bytes)` returning enc[0].a2's buffer at the window start.
   - predict uploads the CT there instead of `nn_malloc(w3)`. The H2D is on the main stream before the forward, so it is
     already ordered.
   - Keep a fallback to the separate buffer if the scratch is null.
4. **uint8 output in the head epilogue (optional, removes the fp32 logits entirely).**
   - `pred_place` (nn_elem.cu) converts fp32 logits to uint8 p·255 and clips them into the shard.
   - Fusing sigmoid, round and clip into the head kernel writes 1 B/voxel. Only worth it if T1 cannot hold the logits
     for some configuration: wider dec_widths or a cout > 4 head without head_nc.

**Result:** 56.45 → **42.45 B/voxel**, plus predict's ~0.5 for the shard.
- 16 GB holds W ≈ 704; the 5090 holds W ≈ 896.
- Effort: about a day. Mostly registry bookkeeping and tests.

### Phase 2: in-place z-slab convolutions at levels 0 and 1 (exact up to GN-sum order, medium-large)

**Goal:** keep only ONE full level-0 tensor (17 B) on the device. Each level-0 3^3 conv then runs as a sequence of
z-slab launches that overwrite their own input.

**Why it is safe.** Output plane z depends on input planes z-1..z+1. Process slabs in increasing z:
1. Compute out[z0, z1) into a small temp slab (s planes).
2. Copy the temp into the tensor's planes [z0, z1).

At that point input planes < z1-1 are dead, so the copy only overwrites dead input. The next slab needs input planes
from z1-1, which are still intact. The one-plane shift between "computed" and "safe to overwrite" is absorbed by the
temp: write slab k only after slab k+1 is computed, or keep a two-slab ring.

**Storage.** The MX layout keeps data and scales as two voxel-major regions, so z-planes are contiguous inside each
region. The in-place copy is two memcpys per slab, one per region, for 32-channel tensors (one block).

**Chain per window:**
- **enc0.c1:** CT/xin → X. Not in place, since the footprints differ. Simplest is to keep xin separate (9 B). Better:
  a fused stem (Phase 2b).
- **enc0.c2:** X → X in place. X then holds the skip.
- **down0:** X → level-1 buffer. Unchanged; reads the whole of X.
- **Levels 1..5:** the same in-place treatment at level 1 removes one of its two 4.25 B tensors. Deeper levels are
  negligible.
- **dec0.c1:** [up2(dec1 out) | skip X] → X, in place over the skip. The skip plane z is dead once out plane z+1 is
  computed.
- **dec0.c2:** X → X in place.
- **head:** X → uint8 output, slab by slab, straight into the shard buffer (Phase 1.4).

**GroupNorm.**
- Each conv's output statistics are accumulated in double sums by the fused epilogue (`gn_dsums`, `gn_finalize_k`,
  src/nn_wgrad.cu:73-95).
- For slab launches:
  - memset the sums once;
  - launch every slab without finalizing;
  - finalize after the last slab.
- The consumer conv normalizes in its staging with the finished statistics, so the result is the same as today except
  for the summation order of the double sums. Expect ≤ 1 ulp differences in the statistics and rare ±1 uint8 output flips.
- With `--gn-frozen` the statistics are given and nothing changes.

**Kernel interface: reuse the split-z machinery.**
- `split_t` / `zs_split` / `zs_len` / `zs_reduce` (nn_wgrad.cu) already run a conv on a z-slab of a window with halo
  planes, for `--split z` across two GPUs.
- A slab launch is the same thing: a z-range of the output plus one halo plane each side read from the same tensor,
  zero padding only at the window's real z faces, statistics partial sums kept for a later finalize.
- Add a z-range variant (`nn_conv3d_fwd_gn_stats_zr` / `_split_zr`) rather than new kernels. The tile loops already
  index by output plane.
- The fused up2 in dec0.c1 reads coarse plane z/2 and needs no change beyond the range.

**Slab size.** Large enough to keep the GPU full: ≥ 32 planes at W ≥ 640. Each extra launch costs a few µs, and the
temp is s·W^2·17 B (32 planes at W = 1024: 0.57 GB).

**Phase 2b: fused stem (removes xin 9 B and xd 8 B).**
- The stem's four input channels are:
  - z-scored CT: `(ct - mean) * inv_sd`, from the window statistics or `--norm`;
  - three radial unit-vector components computed from the axis row offsets (`nn_pred_input`).
- enc0.c1's staging can build them from the uint8 CT window and the per-plane axis rows directly. The input then costs
  1 B/voxel (the CT) instead of 8 + 9.
- The MX-fp8 quantization of the input must be reproduced exactly in staging to stay bit-compatible. That is a
  requirement, not an option, because checkpoints are trained with input_prec 8.
- Side benefit: enc0.c1 currently takes as long as enc0.c2 (see profile); a fused stem probably speeds it up.

**Result, in B per level-0 voxel:**
- Level-0 phases: X 17 + up source 4.25 at dec0.c1 + ctd 1 + slab temps (~0.5 at W = 1024) ≈ 23.
- Coarse phase: X 17 + level 1 ~6.4 + deeper ~1 ≈ 24.5.
- Peak ≈ **25 B/voxel**, 2.3x below today.
- 16 GB holds W ≈ 840; 32 GB holds W ≈ 1060.

### Phase 3 (optional): level-0 skip to the host during the coarse levels

- After down0 consumes X, X is dead on the device until dec0.c1.
- Copy it to pinned host memory, overlapped with the coarse levels; `nn_offload_copy` and the event pattern already
  exist (src/unet.c:1115, 1145). Copy it back before dec0.c1.
- Traffic: 2 x 17 B per voxel per window. At W = 1024 that is 36 GB, about 0.8 s on PCIe 5.0 x16. A 1024^3 window
  takes ~2.5-3.5 s on a 5080, so it can be overlapped.
- Peak becomes max(level-0 phases ≈ 23, coarse phase ≈ 8) ≈ **23 B/voxel**.
- Small gain over Phase 2. Do it only if Phase 2 lands just short of a target window.

### Not recommended

- **Streaming every level-0 tensor through host memory** (layer-by-layer, with live GroupNorm).
  - Each GN'd tensor (enc0.a1, enc0.a2, dec0.a1, dec0.a2) must be complete before its consumer normalizes. So all four
    round-trip PCIe: about 136 B/voxel per window.
  - At W = 1152 that is ~210 GB, about 4 s at ~50 GB/s, against ~3.4 s of compute. PCIe-bound.
- **Two-pass statistics by recomputing convs.** Level 0 is 68% of the time, so this costs +40-60% compute.
- **Frozen GroupNorm on current checkpoints** to make the net purely local and stream it freely.
  - D1 measured −0.009 F1 at 288 and 544.
  - It is only an option for checkpoints trained or fine-tuned with frozen statistics. That is a training-side
    research item and would also allow seam-free sliding inference; recorded here, not planned.

## Converting memory into speed (predict tiling)

predict enumerates tiles per output shard and clips each tile's interior to its shard (src/predict.c:322-340). A window
whose interior exceeds the shard computes voxels it throws away. Memory therefore buys only more halo (fewer seams) at
interior 512, not speed.

To buy speed, let one tile cover k^3 whole shards:
- interior = k·shard;
- the tile writes all k^3 shards through the writer thread;
- the shard order (z-major) is unchanged.

| target (shard 512) | W needed | B/voxel available at 16 GB / 32 GB | reachable after |
|---|---|---|---|
| interior 512, halo 64 (today) | 640 | 58 / 117 | today |
| interior 512, halo 96 | 704 | 43 / 87 | Phase 1 (16 GB) |
| interior 512, halo 128 | 768 | 32 / 66 | Phase 2 (16 GB), Phase 1 (32 GB) |
| interior 512, halo 192 | 896 | 20 / 41 | Phase 1 (32 GB) |
| interior 1024, halo 64 (1.42x vs 1.95x today) | 1152 | 10 / 20 | not reachable with Phase 2 or 3 |

(B/voxel available = (memory − ~1.5 GB context and predict buffers) / W^3.)

**Conclusion for the 16 GB 5080:**
- The realistic payoff of Phases 1-2 is **quality per unit time**: halo 96-128 on 16 GB instead of 64, the dominant
  seam lever.
- It is not raw speed: interior 1024 would need ~10 B/voxel. The 32 GB 5090 gets halo 192 with Phase 1 alone.
- If raw speed is the goal, the next lever is kernel efficiency at level 0: the stem and down0 observations above, and
  `--dec-widths` (DESIGN.md: −14% step time at equal F1) for new checkpoints.

## Verification and acceptance

**Unit.** Extend tests/test_unet.c, or add tests/test_infer_mem.c, building the production config at W = 64 and 128
(must divide 32):
- forward logits of the new placement equal the old one;
  - exact for Phase 1;
  - for Phase 2, ≤ 1e-3 absolute in logits and identical GN statistics to 1e-6 relative;
- each switch on and off;
- head_nc 1 and 0;
- scale_cond on and off;
- input_prec 8.

**Dry report.** `UFSM_MEM_REPORT=1` shows the new per-buffer table. `unet_infer_bytes` must equal the real peak:
`nvidia-smi` and `cudaMemGetInfo` before and after build.

**End to end.**
- Run `ufsm predict` on the Paris4 test box 37888,17408,16384,1024,1024,1024 with the ink checkpoint, comparing old and
  new binaries at W = 640 / halo 64.
  - Phase 1: identical uint8 output (`cmp`).
  - Phase 2: ≤ 0.01% of voxels differ by 1, none by more.
- Same for the recto checkpoint (verso5 final, `--axis`, `--channel 0`) on box 48512,17408,16384,96,1024,1024, scored by
  tools/score_surface.py (F1/AUC unchanged to 3 decimals).

**Speed.** Tiles per second at W = 640 must not drop by more than 2% (Phase 1) or 5% (Phase 2; slab launch overhead).
Report `UFSM_PRED_PROF=1` and `UFSM_PROF=layers` before and after.

**Headroom.** On the 5080, the largest W that runs: 704 after Phase 1, 832-840 after Phase 2. Record the seam ratio
(as in "Why it matters") and the ink F1 at the largest window with interior 512.

## Notes for the implementer

- **The planner.** predict picks the default window with `unet_infer_bytes(...) + 2*W^3 + shard^3 + 1 GiB`
  (src/predict.c:273). Update its extra terms when ctd, xd and the logits move into network scratch.
- **The storage registry** (`nn_set_storage`) decides how kernels read a buffer. Every alias above must re-register the
  buffer before its new user runs. The training paths `xin_shared` and `a0_share` show the pattern and its ordering
  hazards: the waits on `ev_*` events before dec[0].c2.
- **Split-z.** Keep `--split z` (two GPUs on one window) working: a slab launch inside one split side is the same range
  logic with the side's halo planes.
- **Default checkpoints** use input_prec 8, so the stem input is MX-fp8 with 4 channels padded to an 8-channel block.
  Any fused-stem path must reproduce that quantization exactly.
- **Out of scope.** This repo's working copy on the training machine also has local predict changes (`--resume`,
  `--occupancy`, `--norm`, `UFSM_TCP_MSS` for black-holed TLS paths). They do not interact with this plan; expect them to
  land separately.
