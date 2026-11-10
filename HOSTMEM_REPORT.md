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

# Round 2 (2026-11-09): loader CPU, RAM, disk / IO

Commits on hostmem-20261109 after merging latcomp-20261109 (575a58b): 74adc4a, 0d114c9, 616256a, 82ac47e (all pushed; each one was
checked with `--det 1` batch hashes, every plane identical, and with the sampler / zarr / scan-aug / wind4 / gpu-aug / rv / unwrap2 tests).

## Tools added
- `UFSM_SAMPLER_ONLY=N ufsm train <recipe>`: the recipe's training sampler alone, with no GPU (`CUDA_VISIBLE_DEVICES=` works).
  It takes N batches as fast as they come, then prints the loader rate, the per-stage profile, host memory, per-root store IO and
  /proc/self/io. Every number in this section comes from it. Production GPUs were not used, except for the GPU-tagged tests.
- `UFSM_SAMPLER_PROF=2`: stages in thread CPU time instead of wall time. Wall time counts IO and contention, so it is not a CPU measure.
  The old "augment" stage is now split into sympack / phread / wind4 / phout.
- `store_io_report`: reads, bytes and open+pread wall time per store root.
- `UFSM_SAMPLER_RT=T`: threads per window (see below).
- `UFSM_SAMPLER_THP=0/1`.

## CPU (measured)
Setup: P512, `--det 1` (the same 24 windows each run), 4 workers, the instrumented round-2 baseline (575a58b) vs 82ac47e, two
alternating runs each. The machine was busy with the two production loaders, so absolute times drift by ±10% between runs.

| per window (thread CPU ms) | before | after |
|---|---|---|
| ctread (CT decode) | 777 | 825 |
| targets | 512 | 401 |
| scan (plan, ct8 copy, air mask) | 756 | 395 |
| sympack (dead tm pack) | 238 | 0 |
| phread (phase + gate decode) | 1037 | 1064 |
| wind4 (band_wind4_tiled) | 1173 | 1246 |
| phout (air test, side output) | 283 | 203 |
| augment (rest: ink, w4 fill) | 361 | 293 |
| zscore / encode / ctstats / x16 / probe | 785 | 818 |
| **total** | **5920** | **5245 (-11%)** |

- perf stat: user instructions are unchanged (356.6 vs 353.8 G for 24 windows), user cycles went down 9% (764 / 814 -> 696 / 745 G),
  and user time went down 9% (164 / 173 -> 148 / 158 s).
- What changed:
  - fused passes: two-channel ignore/encode, mask+ink, in-place not-air;
  - dead work skipped: the unwarped-window tm pack, zeroing of target-less channels, and the target-max loop once it reaches 254;
  - z3_read fills only absent chunks;
  - an ignore window with no stored chunk is no longer read;
  - phase, winding and component go straight into their batch planes for device-warped and identity-symmetry windows (no copies).
- Where the time goes now (perf): IPC is about 0.46 and each voxel costs ~100 instructions and ~200 cycles. CT + phase decode
  (vfx / vll, copy_out) is ~25%, the winding mod 4 union-find ~25%, and the rest is ~30 passes over P^3. While both production
  loaders run, the vectorised byte loops reach only 2-3 GB/s per core: the box is DRAM-bandwidth-bound.
- Per core: 4.5-5.5 user-s per 134 Mvox window, i.e. ~25-30 Mvox/s per busy core (det mix).
- Production check: m768 (W8 S10, P768) logs 0.42 samples/s = 190 Mvox/s with 80 s of wait per 50 steps (~67%). That is
  ~24 Mvox/s per worker while k768 shares the CPU. **The loader is CPU-bound by a wide margin.** A GPU at ~570-750 Mvox/s would
  need 20-30 cores per GPU at today's cost per voxel.
- **Threads per window** (`UFSM_SAMPLER_RT=T`): CT, phase and target reads decode their chunks in parallel, and the winding tiles run
  in parallel (`band_wind4_tiled_mt`). The bytes are the same for any T; test_wind4 checks the mt variant on 2 and 5 threads.
  Measured: W2 RT2 gave 68-78 Mvox/s against 99-101 for W4 RT1. So half the windows in flight (half the worker scratch, two slots
  fewer) give ~70-80% of the throughput, at an efficiency of ~75% per thread. Use it when RAM binds, i.e. at P1152-P1280.
- **THP (measured):** under system memory pressure (MemAvailable ~5 GB, compact_fail 76k), MADV_HUGEPAGE on the worker scratch made
  first touches compact memory synchronously. One run took 493 s system vs 125 s user for 24 P512 windows, with a 57-163 s first
  batch. With huge pages on vs off, the --gpu-aug 2 recipe showed no cycle difference (695 / 638 vs 690 / 654 G), so madvise is now
  off by default under --gpu-aug 2. The machine-wide fix is `defrag=defer+madvise`, which needs root.

## RAM (measured)
- Per worker: 4 B/vox (was 5 after round 1, 11 before). Measured `big` residency is 2 P^3 per worker plus ttmp 2 P^3; big + P^3 is
  no longer touched in this recipe.
- Per slot: 5 B/vox.
- Model: **RSS = 1.2 GB + P^3 (4 W + 5 S)**, with S = W + 2.
- Validation: the production runs (P768, W8 S10, still 5 B/vox per worker) show VmRSS 42.6 / 42.7 GB; the round-1 model gives 42.0 GB.
- Floor in this recipe: 4 planes are live at once in the target and phase stages (CT, mask, two of ttmp). Going lower needs the ideas below.

## Disk / IO (measured)
- **Where:** everything the loader reads is on /vesuvius (NVMe). /mnt/scratch (HDD) holds only axis.json files, read at start.
- **Store sizes:** the phase labels are far larger than the CT. Phase stores (volcomp q0, lossless):
  - P4 phaseA 86 GB, phaseB 57 GB
  - good/0175A 49 GB, good/0139 46 GB, good/0009B 5 GB
  - 0139full 8.5 GB
  - CT levels used: 60 GB in total.
- **Bytes per window** (non-det recipe, P512, 40 windows): 1.63 GB of phase vs 0.35 GB of CT. That is ~37 MB of phase and ~8 MB of
  CT per 134 Mvox window, ~0.33 B/vox, 80% of it labels. Per root over the 40 windows:

  | root | MB read |
  |---|---|
  | good/0175A | 477 |
  | good/0139 | 409 |
  | ct | 351 |
  | p4full/phaseA | 330 |
  | p0139full | 299 |
  | p4full/phaseB | 75 |
  | good/0009B | 54 |
  | gates | <1 |

  pread runs at 220-300 MB/s per thread, 7.5 s of 46 s × 4 workers: ~4% of worker time. The ~15k small reads are shard indices at startup.
- **Page cache hit rate ≈ 0:**
  - cold run: rchar 2.00 GB, read_bytes 2.10 GB;
  - production after 40 min: rchar 176 GB, read_bytes 190 GB (8% readahead overshoot), ~73 MB/s per run.
  - The working set (~310 GB) is larger than RAM, so the cache helps nothing. Yet each production scope holds ~10 GB of file cache
    under its 50 GB cap and keeps reclaiming it (pgscan 44M pages, workingset_refault_file 11.8M).
  - The scope's "shmem" (~25 GB) is the pinned memory: cudaMallocHost is shared anonymous memory.
- **`UFSM_STORE_NOCACHE=1`** (already in store.c: fadvise DONTNEED after each read): two alternating pairs of runs showed no
  throughput cost (107 / 113 Mvox/s cached vs 118 / 120 with nocache) and the cgroup file charge drops from 1.92 GB to 0.
  **Recommended for every training run.**
- **No decoded-window cache**: nothing is reused.
- **No pre-decoded levels**: raw phase is 1 B/vox, 3-4x the bytes. Decode is ~15% of CPU, so NVMe could carry it (~750 MB/s per
  run at the GPU rate), but storage would need ~1 TB+.

## Recommended settings
Estimates from the model. S = W + 2. Set UFSM_STORE_NOCACHE=1 and keep THP off (default now).

| P | W / S (RT) | RSS (model) | loader estimate |
|---|---|---|---|
| 960 | 10 / 12 | 90 GB | ~250-300 Mvox/s if 10 cores are free |
| 1024 | 8 / 10 | 89 GB | ~200-240 |
| 1152 | 6 / 8 | 99 GB | ~150-180 |
| 1152 | 4 / 6, RT 2 | 72 GB | ~170-200 on 8 threads |
| 1280 | 4 / 6, RT 2 | 98 GB | ~170-200 on 8 threads |
| 1280 | 3 / 5, RT 3 | 79 GB | ~150-200 on 9 threads |

**At every P the loader, not RAM, limits throughput:** ~25-30 Mvox/s per core.

## Remaining ideas (ranked by expected gain)
1. **The winding mod 4 (25% of CPU):** band_wind4's pass0 / pass1 are scalar union-find at ~8 ns/voxel. A run-length front end
   (runs of known phase per row, joined row to row) or a coarser tile pass would cut the per-voxel work.
2. **Decode straight into the window (copy_out ~3-5%) and a faster lossless phase code.** vll q0 runs at ~3.4 ns/voxel and the
   phase is 80% of the bytes read. This is codec work.
3. **Ink plane from ct8 plus a 1-bit known mask** (on the device, or at upload): -0.875 B/vox per slot, 11 GB at P1280 S6. It is
   exact for this recipe because no seam op shifts the CT; with seam the ink plane is not shifted today anyway.
4. **Fold the remaining elementwise passes into cache-sized z-slabs** (zscore / encode / ctstats / ink / mask / u8_stats, ~20% of
   CPU). The box is bandwidth-bound, so fewer DRAM passes pay directly.
5. **xoshiro jump-ahead instead of burning P^3/4 draws** to keep the CPU path's stream: ~2.4% of instructions.
6. **CT straight into the slot** (saves the ct8 memcpy). RAM gain only if the post-scan 0/1 mask finds a free plane.

## Round 2b: window replay, system time, THP (2026-11-09)
Commits: 8f628d2 (`--replay`), 790e62b (band_wind4 arrays, NOCACHE default), 489ce05 (THP heuristic).

### Window replay (`train --replay K`, default 1)
What a replay is:
- Each built window is delivered K times. Replays 2..K are prepared on one sampler thread from the window's own slot.
- The device-side records are redrawn the way the sampler draws them: a new cube symmetry (augment mode, symmetry_p) and a warp
  record (rotation is forced if none came up, because the labels need the device warp); elastic / zoom / affine at their usual
  probabilities; a new scanner plan; new intensity jitter and noise seed; the axis record and um follow the new warp.
- In the new scanner plan, the label-touching ops (seam, air, dropped slices, cutouts, chunk dropouts) are never drawn anew. The
  window's own air / drop / cutout / chunk boxes are carried over, so the labels (ink unknown on dropped slices, the CT > 0 air
  test) stay exact.
- The label planes of an unwarped window with a non-identity symmetry are put back on the source grid first (sym_inverse,
  self-tested on all 48 symmetries at sampler start). The CT bytes (ct8) are on the source grid already.
- No voxel work is redone: no read, decode, targets or phase block.

Measured:
- Replay 0 batches are byte-identical to K=1 (`--det 1`, 12 windows, every plane).
- Delivered loader rate (UFSM_SAMPLER_ONLY, P512, 4 workers, non-det): K=1 131 Mvox/s, K=2 245, K=3 375 (1.9x / 2.9x). The replay
  thread is not the bottleneck.
- A 20-step GPU run at P384 with K=2 trains normally: finite losses in the same range as K=1.

Caveats:
- Every replay is warped (rotation <= --rotate-deg).
- A replay keeps the original window's source, position, level, axis jitter and dropout / cutout / chunk boxes.
- The K replays of a window arrive back to back, so K consecutive steps are correlated: start with K=2.
- Configurations: `sampler_replay_ok` (--gpu-aug 2 cubes; no sheet / band / gap / condition / distill / view / cover / chain, pm
  only when skipped).

### System time
- In production (P768, W8 S10) each run's scope sat at its memory.max with ~10 GB of page cache.
- Every page read was paid for by a direct reclaim in the reading worker: pgscan_direct ~15k pages/s per run, matching the read
  rate; kswapd did nothing.
- Train now sets **UFSM_STORE_NOCACHE=1** by default (`=0` keeps the cache). The cgroup file charge goes to 0 and throughput is
  unchanged (measured earlier).
- Page faults (perf -e page-faults on a production run for 15 s): 90% came from band_wind4's union-find arrays, which grew by
  realloc per tile and were freed after it; the other 10% from the read path copying into fresh buffers. The arrays now stay with
  the thread (byte-identical; test_wind4 OK).
- THP: worker-scratch MADV_HUGEPAGE is now used by default only with host warps (--gpu-aug < 2) and MemFree > 16 GB at sampler
  start. UFSM_SAMPLER_THP=0/1 forces it either way.

## Round 3: replay interleave, decode into the window, fewer passes (2026-11-09)
Commits: 001274f (replay interleave), fbda89d / 491b3e9 / 72c5d5e / 1068d87 / 3948381 (pass cuts), 145368d (z3 window sink),
f089678 (rng jump-ahead). origin/latcomp-20261109 is merged in. At every step the batches are byte-identical (`--det 1` plane
hashes, 24 P512 windows; also with --min-phase 0.3, which rejected 5 windows, and with air / cutout / dropslice at p 0.4, compared
against the round-2b binary).

### Window replay, interleaved (`--replay K --replay-gap G`)
- A window that has replays left keeps its slot. Its next use is handed out G batches after its previous one (FIFO over the
  released windows).
- Non-det: a replay can come earlier only when no new window is ready. `--det 1`: a replay is also forced when its slot holds the
  next batch index. Every decision depends only on the trainer's call order, so runs are deterministic.
- Default G = (slots - workers) K / (K - 1), clamped to 2 .. slots. A held window keeps its slot for ~(K-1) G / K batches per new
  window, and this G leaves every worker a slot to build in. For more spacing, add slots: `--sampler-slots workers + 2 + extra`.
  With K=3 about 1/3 of the workers are needed, so trading workers (4 B/vox each) for slots (5 B/vox each) is about RAM-neutral.
- Measured: two `--det 1 --replay 3` runs are byte-identical; K=1 is unchanged. P512, 4 workers, 6 slots, K=3: the mean distance
  between uses of a window is 3.3-4.5 batches, and 0-4 of ~40 uses are adjacent. Mvox/s matches the back-to-back order within noise.
  A 40-step GPU-1 run at P512 (K=3, 3 workers, 8 slots, G=7) trains normally.
- Delivered loader rate (P512, 4 workers, 6 slots, non-det, load avg ~8): K=1 428 Mvox/s, K=2 773, K=3 985. The round-2b numbers
  (131 / 245 / 375) were taken at load ~25; the box's load moves these by 2-3x.

What stays the same across the K uses (everything done on the CPU before upload):
- source, level and window position
- the CT bytes (ct8) and every label plane (side, w4, wcomp, ink, tm)
- the CPU warps (--aug-crush / tear / delam / curv) and the zoom-out read
- the axis jitter (the umbilicus estimate)
- the label-touching scanner ops: seam, air, dropped slices, cutouts, chunk dropouts
- the --ct-aug noise sigma

What is new on each use (on the device):
- the cube symmetry
- the rotation / elastic / zoom-in / affine warp (a rotation is forced when none is drawn)
- the scanner plan of the intensity ops (blur, noise, paganin, tone, ring, quant, bias, cupping, lowres, sharpen, ...)
- the intensity jitter and the noise seed

### Loader CPU
| change | effect |
|---|---|
| z3 window sink (145368d): lossy, lossless-coded and constant volcomp chunks decode straight into the request box. There is no 128^3 chunk buffer and no copy_chunk; blocks outside the box are entropy-decoded but skip the IDCT / rebuild. Not used for z3_read_sub, device sinks, the shared cache or the GPU decoder. | single-thread 512^3 reads: phase -12.5% CPU, CT -2.6% |
| xoshiro jump-ahead (f089678): the P^3/4 skipped draws become one GF(2) matrix T^n (built once per count, self-tested at sampler start) | -1.6% instructions |
| the ink-channel zeroing skipped when unread; the ignore mask written by the first labelled channel, or lazily zeroed (only papyrus labels, the sheet prior, the host scanner path or ignore-writing scanner ops need it); the CT->ct8 copy fused into scan_plan_notair; the --rv papyrus plane built from ct8 in the mask pass; the loss mask not written when nothing reads it (tdead); the --min-phase count taken inside phase_air | ~5 P^3 passes / streams fewer per window |

Total, A/B against the round-2b binary (P512 --det 1, 24 windows, alternating runs, user+sys CPU per Gvox):
- load ~8: 13.08 -> 11.81 (-9.7%), new faster in 3 of 4 pairs
- earlier, load ~8-10: 12.86 -> 12.17 (-5.4%), new faster in 4 of 5 pairs

So about -5 to -10%. The machine's load (production loaders) moves single runs by +-20%, so only paired runs are meaningful.

Profile now (P512, perf):
- volcomp decode ~30% (the lossless phase's entropy decode ~19%, lossy CT ~11%)
- band_wind4 + w4_tiled 17%
- phase_air 7%, sym_u8 6%, scan_plan_notair 6%
- the mask / ink pass, which is DRAM-bound (four streams)
- memset 4.5%: the w4 / wcomp 255 fill of windows without a stored phase, plus fill_part

### Not done
- **band_wind4:** waiting for the perf agent to say who takes it (it was asked to move band_wind4 / symmetry to the GPU). band.c
  is untouched.
- **Faster lossless phase codec:** a format change, so every phase store would need re-encoding. Not started.
- **Ink plane from ct8 plus a 1-bit known mask** (-0.875 B/vox per slot): needs the trainer's upload paths (lab_upload, label
  streaming) and a device kernel. Not started, pending the lead's call; replay lowers the worker count, which already cuts RAM.
- **The w4 / wcomp unknown fill (2 P^3 memsets) as a per-window device fill:** a trainer change.
- **Tests:** the CPU fast tier plus the GPU / sampler tests on GPU 1 pass (15/15). With `-g none`, 10 tests that need CUDA but
  have no `gpu` tag fail on master too: test_gpu_input, test_unwrap2_sampler, test_l0_stream, test_gpu_aug, test_fit*,
  test_wtrunk, test_nn.
