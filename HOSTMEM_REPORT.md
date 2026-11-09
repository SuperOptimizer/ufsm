# Host RAM of `ufsm train --lat 4` at large windows (2026-11-09, branch hostmem-20261109)

Recipe: `/tmp/.../scratchpad/rep.sh` (task unwrap2, --rv 1 --rv-iso 64, --gpu-aug 2, --input-prec 8, --mem auto, --val-store disk,
--val-batches 0, B 1), GPU 1, every run under `memcap -m 60G`. Units: **B/vox** = bytes per window voxel (P^3); P 960 = 0.885 Gvox,
P 1024 = 1.074, P 1152 = 1.529, P 1280 = 2.097.

How it was measured: `UFSM_HOSTMEM=1` (new, a `hostmem:` line at every log line) splits RSS into pinned memory (every
`nn_host_alloc*` block with its size and caller), the batch slots, and per worker the fixed scratch (`big` 10 P^3, `ttmp` 2 P^3)
and the window pool (`tp_get`), with residency read by `mincore` (only touched pages count); plus `mallinfo2`. `UFSM_TP_LOG=1`
logs worker 0's pool calls by source line and its scratch residency per P^3 slice; `UFSM_BATCH_HASH=1` prints one hash per batch
plane (byte-identity checks with `--det 1`). RSS was also sampled from /proc every 2 s and smaps saved at the peak.

## 1. Where the bytes are (measured)

| owner | before | after | what |
|---|---|---|---|
| batch slot (pinned), per slot | **7 B/vox** | **5 B/vox** | ct8, side, w4, wcomp, ink + (tm, pm: never read by the trainer in this mode) |
| worker, `big` (10 P^3 virtual) | 4 B/vox resident | 3 | CT, region/scan-air scratch, pmask temp (slice 2), loss mask (slice 9) |
| worker, `ttmp` | 2 | 2 | recto / verso targets (they feed the ignore mask -> ink plane, and the empty-window test) |
| worker, pool (`tp_get`) | 4 (peak, + transient 1) | 0 | stored-phase block: phase, winding input, winding mod 4, component |
| worker, ink buffer (thread-local malloc) | 1 | 0 | --ink-head / --rv plane |
| **per worker total** | **11 B/vox** | **5 B/vox** | |
| pinned, not slots | 0.2 GB (P640), 0.67 GB (P960) | same | U-Net offloads (shared encoder a1 0.31, 0.235, 0.124 GB at P960), g_zeros, head weights |
| pageable fixed | ~0.5 GB | same | CUDA libraries, malloc arena 0.18 GB (fragmentation is not an issue: the arena holds 0.17 GB, everything big is mmapped and returned) |

Not present in this recipe: zcache / zshm (UFSM_ZCACHE_GB unset), validation store (--val-batches 0), decoded-chunk caches in RAM.

Model (fits every run below within a few %, once all workers have drawn a window):
- before: RSS = 1.2 GB + P^3 (11 W + 7 S)
- after:  RSS = 1.2 GB + P^3 (5 W + 5 S)

Runs (measured; P 640 = 0.262 Gvox, 25-30 steps):

| run | before RSS | after RSS | after: wait |
|---|---|---|---|
| P512 W4 S3 (det, 30 batches) | 8.81 GB | 5.07 GB | |
| P640 W4 S3 | 15.65 | 8.10 | 46% |
| P640 W8 S3 | 14.0 (6 of 8 workers warm) | 13.04 | 38% |
| P640 W4 S6 | 21.77 | 12.99 | 0% |
| P640 W4 S4 | | 10.54 | 7% |
| P640 W4 S5 | | 11.75 | 0% (306-311 Mvox/s) |
| P640 W6 S8 | | 17.88 | 0% |
| P768 W5 S7 | | 26.24 | 0% (323 Mvox/s) |
| P768 W4 S5 | | 19.92 | 4% |
| P960 W8 S3 (the one P960 run) | (lead: ~88 GB) | 34.2 at step 25 (model, all warm: 50) | 12-32% |

The old model gives 97.7 GB for P960 W8 S3 (lead saw ~88 with the pool not yet at its peak) and 213 GB for W18 S6 (OOM at >175).

### The second finding: workers beyond the free slots do nothing but hold scratch (measured)
A worker builds its window straight into a slot it claims (`state FILLING`), and the trainer holds about one slot per GPU, so at
most S - 1 windows are built at once. W8 S3 builds 2 at a time: the lead's W8 S3 run (26-29% wait) and W18 S6 were slot-bound, and
every worker beyond S - 1 cost 11 B/vox for nothing. At P640: W4 S3 46% wait, W8 S3 38% (no better), W4 S4 7%, W4 S5 0%.
`train` now prints `sampler: N workers but S slots: at most K windows are built at once (--sampler-slots X ...)`.

Loader rate (measured): CPU per P960 window 14.6 s summed over stages (ctread 3.5, targets 2.5, encode 1.2, scan 2.0, augment 4.9)
= ~60 Mvox/s per worker; wall rate per worker at P768 with W5 S7 >= 65 Mvox/s (the GPU, 310-323 Mvox/s, was the limit). The machine
was shared with lat1's 8 workers during all of this.

## 2. Changes (commits on hostmem-20261109, all pushed)
- `ce55322` accounting: UFSM_HOSTMEM, UFSM_TP_LOG, UFSM_BATCH_HASH, nn_host_bytes / nn_host_report.
- `dc7b950` slot_skip (`sample_cfg.slot_skip`): batch.tm (unwrap2 head slabs, g_tm_unused) and batch.pm (unwrap2 without --pap-head) are
  neither allocated nor filled for the training sampler (the validation sampler keeps them: its stored batches are unchanged).
  -2 B/vox per slot.
- `216e8e0`, `23e0e54` stored-phase block in dead worker scratch: for unwarped and device-warped windows without a sheet prior (`ph_own`), phase / ignore /
  weight / winding input / winding mod 4 / component live in ttmp (+4 KB), big + P^3 and big + 9 P^3, which nothing reads again for
  those windows (write_spatial_tm and the sheet prior are the only later readers, and these windows skip both). Host-warped windows
  (--gpu-aug 0/1) keep the pool. -4 B/vox per worker.
- `216e8e0` ink plane built in the batch's own plane (the symmetry / host warp move it through big + P^3): -1 B/vox per worker.
- `7fb8bf9` `train` warning for workers > free slots.

Verification:
- `--det 1` batch hashes, P512, 30 batches: ct8, aug records, radial records, side, w4, wcomp, ink, um, src, w identical
  (300 plane hashes, 0 differ) after each step.
- fast suite (`make test SUITE_FLAGS="-g 1 -j 4"`): 175 pass, 1 skip, 2 fail: `test_cover_training` (gpu2: "cannot select GPU 1",
  it needs two GPUs and the run was limited to GPU 1) and `test_pack_tm` part 1, which only packs synthetic bytes and unpacks them on
  the device (no sampler code); rerun by hand it passed 2 of 3 times: flaky on the copy-stream path, not caused by these changes.
  test_gpu_aug, test_unwrap2_sampler, test_wind_sampler, test_sampler_safety, test_rv, test_unwrap2_ink pass.
- Speed: CPU per window not worse (P640 W4: 2.35 s vs 2.56 s summed stage time; the ink / tm / pm copies are gone); the
  measured wait went down because more workers can be afforded, not because a worker got faster.

## 3. Recommended settings under a 110 GB cap (estimates from the model, not run)
S = W + 2 (one slot with the trainer, one ready, W filling). RSS = 1.2 + P^3 (5 W + 5 S).

| P | W / S | expected RSS | expected loader | expected wait |
|---|---|---|---|---|
| 960 | 8 / 10 | 81 GB | 480-620 Mvox/s, GPU-bound ~310 | 0% |
| 960 | 6 / 8 (leaner) | 63 GB | 360-470 | 0% |
| 1024 | 7 / 9 | 87 GB | 420-550 | 0% |
| 1152 | 5 / 7 | 93 GB | 300-390 | 0-5% |
| 1280 | 4 / 6 | 106 GB (tight) | 240-310 | 0-20% |
| 1280 | 4 / 5 | 96 GB | 240-310 (one slot less queue) | 0-25% |

Loader = W x 60-78 Mvox/s (measured per worker at P640-P960); the GPU consumes ~310-320 Mvox/s at P640/P768 (measured), assumed
per-voxel constant at larger P. lat1's own 8 workers share the CPU; on an idle machine the per-worker rate should be higher.

## 4. Remaining ideas, by expected gain
1. **Build windows with several threads each, fewer in flight** (memory scales with windows in flight, W + S, not with loader
   throughput): e.g. 3 windows x 4 threads instead of 8 x 1 halves the worker side again. Large change (every stage).
2. **Do not enable the rare augmentations at large P before restructuring them**: crush / curv / xscan / papyrus labels / region
   targets / host scan aug use big + P^3 .. 9 P^3 (8 B/vox more per worker, 7 GB per worker at P960, since `big` keeps them once
   touched); and --gpu-aug 0/1 (host warps) still uses the pool (+4-6 B/vox per worker). Measure with UFSM_HOSTMEM before turning
   any of them on.
3. rv ink plane from the device: with --rv-iso it is (CT >= iso ? 254 : 0) except unknown voxels; one bit of unknown per voxel
   (P^3/8) instead of a byte: -0.9 B/vox per slot (~9 GB at P960 S10).
4. CT read straight into the slot (ct8): -1 B/vox per worker; needs the post-scan 0/1 `ctu` elsewhere for scan-augmented windows (a
   bit plane, or a free slice when the phase block moves).
5. Idle worker scratch: MADV_DONTNEED / free while a worker waits for a slot (only helps if W > S - 1 is kept on purpose).
6. ttmp (2 B/vox): the targets matter only for the ignore mask and the empty test in this mode; computing them as one byte (recto
   code + unknown bit) would save 1 B/vox per worker.
