# ufsm — design

Ultra fast scroll model: a C23 + CUDA pipeline that trains a ~1 M-parameter 3-D convnet to predict the
recto papyrus face in Vesuvius Challenge micro-CT, from scratch, using only the released ground truth:
the gated HuggingFace `scrollprize/datasets` bucket and the AWS `vesuvius-challenge-open-data` bucket.
No teacher models, no distillation. All upstream data is re-exported once into the user's own codecs
(volcomp volumes, surfcomp surfaces) so training reads one compact local store.

## The human ink targets on the fly, the derived-array chunk cache, and the villa ink recipe (2026-11-11, branch inkfly-20261111)

**On the fly.** The stored splats of the section below (inkgt/gt) are now only the reference. Training reads the same bytes from a
derived array: `inkfly_open(ct, segs.json, cfg, ...)` (inksplat.c), a sources target `"ink": {"onfly": {"segments", "level",
"iso", "face", "holdout_file", "cache_mb", "cache_dir"}}`.
- **Open.** Every label window's mask is read once. The 0/1 -> 0/255 rule is decided per window (as inksplat_run does), and each
  supervised 64^2 tile's 3-D box (reach 21 voxels) goes into a 1024^3 grid.
- **Per 128^3 chunk:**
  - the candidate tiles from the grid;
  - per window, the label rectangle and the supervised pixels whose surface point can reach the chunk;
  - the CT box around those points only, through a zcache;
  - face search and splat with ink-splat's arithmetic (rect offsets split exactly);
  - known-max merge over windows, with the held-out boxes and CT >= iso applied as in the stored splat.
- **Level 2** is known-max 2x pooling of the level-1 array, as in the stored pyramid.
- **Draw cells.** The sampler draws from the label tiles' boxes (inkfly_cells), not from the CT's chunk map. Ink-only sources therefore
  draw only where labels reach; univA confirmed ink-only sources train with the winding losses at 0.

**The chunk cache (dcache.h, shared).** `z3_open_cached(inner, dir, key, cell, mem_mb)` wraps any array, typically a derived one, so
the derive signature does not change.
- Reads are served in cell^3 cells. Each is looked up in a memory LRU, then in DIR/<sha256(key)>/<cz>/<cy>.<cx>; only then is inner
  read, with an aligned cell box.
- Concurrent readers of a cell wait for the first one's compute. Processes sharing DIR race only on atomic renames. Deleting DIR is
  always safe.
- DIR/<hash>/inputs.txt holds the key. For ink the key is a code-version string, the CT, the cfg, the held-out boxes, the size and
  mtime of every mesh / label raster, and the segments json, so edited inputs miss the cache.
- DIR is `"cache_dir"`, else $UFSM_DERIVED_CACHE ("off" = memory only), else ~/.cache/ufsm/derived.
- Phase and fibre can wrap their derived arrays the same way.

**Gate and cost** (`ufsm ink-onfly SEGS --box .. --ref <stored splat> --dcache DIR`, 8 threads, desk):

| box (4.8 um voxels) | known / ink | cold: cells (labelled) | cold CPU | from disk | result |
|---|---|---|---|---|---|
| 1667 around crop w029_0, 863 x 905 x 663 | 0.98 M / 0.26 M | 392 (55) | 6.1 s (CT 3.3 s) | 0.07 s | 0 of 518 M voxels differ |
| Paris 4 around crop w02_0 (a held-out box inside), 1280 x 1240 x 796 | 4.55 M / 0.90 M | 847 (234) | 25.6 s (CT 7.1 s) | 0.17 s | 0 of 1.26 G voxels differ |
| MAN5 (before the cache) | 1.2 M / - | | | | 0 differ |

- About 110 ms of CPU per labelled cell computed, against 0.2 ms per cell read from disk and 0.4-1.1 ms per warm 256^3 window.
  Cells that no label can reach cost nothing (612 of 847 on Paris 4).
- Disk use is about 12 KB per labelled cell; constant cells take 16 bytes.
- A whole scan is computed once: about the 40 CPU-min of the stored splats, spread over the first epoch's reads.
- What remains per process is the index open: Paris 4 33 s -> 10 s, 1667 19 s -> ~6 s, after a row memcpy in lsrc_read (the
  per-byte copy was half of it) and the windows on 8 threads. Paris 4 is opened by three sources (june / july / june48). The rest
  is the serial mesh TIFF loads.
- The index (tile boxes, 0/1 flags) is cached beside the chunks as DIR/<hash>/index.bin, under the same inputs key: Paris 4 opens in 0.7 s from it, against 10.8 s cold.
- test_ink_splat: on the fly == stored over two merged crops with a held-out box, on odd boxes, cold, from memory, and from disk
  through a reopened array (0 cells recomputed).

**The GPU port (nn_inkfly.cu, inkflygpu.h).** A cold chunk's compute now runs on the device; the host keeps the I/O.
- **Host:** the candidate tiles, the label rectangles (lsrc_read), a mesh patch per unit (the rectangle's grid cells +- 1 label px)
  and the CT box.
- **Device, two stages on the calling thread's own stream and buffers** (the sampler's workers run concurrently):
  - near: one thread per label pixel of the candidate tiles computes ssurf_pn and the near test, and appends near pixels with
    their P / N and the level-0 box their samples reach (atomic min / max). With no near pixel, the chunk is unknown and no CT is read.
  - finish: per unit, the face (27 trilinear samples, [1 2 1] in float, face_of), then the splat (one thread per pixel x sub-sample,
    atomic counts and max; the CPU's saturating uint8 counts are min(count, 255)), then the finalize merged by known max.
- **Exactness:** -fmad=false, IEEE sqrt / division, the CPU's expression order. The bytes match the CPU's: 0 differing voxels on the
  1667 and Paris 4 gate boxes and in test_ink_splat (open 2, now tagged gpu).
- **Fallback:** a mesh patch that falls short returns 1 and the CPU code runs. A CUDA failure turns the GPU off for the process.
- **Device choice:** train uses `--onfly-gpu` (as phase / fibre); ink-onfly uses $UFSM_INKFLY_GPU (default 0); $UFSM_INKFLY_CPU=1
  forces the CPU.

1667 gate box (55 labelled of 392 cells, 90 with near pixels; desk, both GPUs at 100 % with u1 training):

| | 1 thread: wall / CT / device | 8 threads: wall |
|---|---|---|
| CPU | 4.7 s / 1.3 s / - | 1.5 s |
| GPU | 3.5 s / 2.1 s / 0.9 s (10 ms per chunk) | 1.0 s |
| GPU + UFSM_GPU_DECODE=1 | 4.3 s / 2.8 s / 1.0 s | 1.9 s |

On the Paris 4 gate box at 8 threads: GPU 8.7 s wall, CPU 15.9 s (the desk was loaded by other suites).
- The face + splat is now ~10 ms per chunk on a shared GPU, against ~90 ms of CPU.
- The CT decode (~23 ms of CPU per chunk, decoded chunks shared through the zcache) is now the larger part.
- nn_gpudec through z3_read (UFSM_GPU_DECODE=1) is byte-identical but slower while the GPUs train: small batches queue behind u1's
  kernels. CT decode therefore stays on the CPU by default; the decoded CT feeds the device box through one H2D copy.
- With the disk cache, each chunk is paid once per dataset in any case.

**MIL building blocks (adopt item 1; the loss wiring is univA's head_slab_pass).**
- **Rays:** `inkfly_rays(handle, o, n, mode, stride, R, ..)` gives the rays of a level-L window: per supervised label pixel whose mesh
  point lies in the window, the point and unit normal in window voxels, the 2-D label and a weight. Mode 0 takes every pixel; mode 1
  only those without an open face (the voxel target's unknowns, by the same face search). Pixels within R + 1 voxels of a held-out box
  are left out.
- **How many lack a face** (`ink-onfly --rays`, R 10):
  - 1667 gate box: 1.34 M supervised pixels, 410 k without a face (31 %, 24.6 % ink vs 25.4 % overall);
  - Paris 4 gate box: 6.53 M, 2.72 M without a face (42 %, 26.6 % ink).
  The unfaced pixels carry the same ink share, so MIL on them adds unbiased supervision.
- **Ray cost:** about 3 us of CPU per ray including the face search, so ~0.15 s for a 256^3 window's ~43 k rays; stride 2 gives a
  quarter of that.
- **Loss:** `ink_mil_loss_dev` (nn_inkmil.cu, inkmil.h). One thread per ray takes K = 2R / step + 1 trilinear samples of the
  full-resolution ink logit along the normal and pools them: soft max tau log mean exp(z / tau), or the hard max at tau 0. Then BCE
  with label smoothing eps, the gradient spread by atomics over the samples' trilinear corners.
- **test_ink_mil:** the host reference matches central differences (max rel err 1.5e-7 soft, 2e-8 hard), and the GPU matches the host
  (loss to 1e-6, gradient error 5e-9 of 3e-3).

**The villa ink recipe vs ours.** Read-only survey of ScrollPrize villa: vesuvius/ink_detection on main (the merged
merge-ink-pipelines configs), plus the paris4-3d-gambiarra and distilled_3d_ink branches.

| | villa | ufsm |
|---|---|---|
| labels | 2-D per segment (inklabels / supervision / validation masks); ignore = supervision 0; components cleaned to {0, 255}; several self-training relabel rounds (mean teacher, DINO-guided v1-v3) | the same 2-D human sets (20260918), unchanged; prob254 with 255 unknown |
| 3-D placement | full_3d: label projected along the normal, ±1 voxel (±3 in the teacher), on the mesh; normal_pooled_3d: max of the 3-D logits over ±10 voxels along the normal, then the 2-D loss (MIL); no notion of which side is the recto | per pixel at the open recto face found in the CT (+N), band [f - 6, f]; no face = unknown (17-48 % of pixels) |
| input | ~9.6 um isotropic, 17 of 21 surface-parallel slices (z jitter ±2), patch 17 x 128 x 128; or 3-D 256^3 at 2.4 um (DINO teacher) | the scan grid itself (4.8 / 9.6 um levels), 3-D windows; no flattening |
| model | 3-D stem + depth attention/max pool -> 2-D nnU-Net (output 2-D); full 3-D U-Net; DINOv2 / dinovol encoders; gambiarra 24.5 M 3-D residual U-Net | ink3dext teacher (3-D); the universal net's 3-D ink head on the shared trunk |
| loss | BCE (label smoothing 0.5 in hybrid, 0.1 in the teacher) + 0.25 dice; SGD 0.01 Nesterov, 78 k iterations, batch 64 with fixed scroll quotas | ink BCE per known voxel (nn_loss_stats over the ink mask), no smoothing, no dice |
| augmentation | XY mirror, rot90 about z only, intensity (gamma, invert, illumination, noise / blur, cutout); no elastic / scale (normals) | full 3-D spatial + CT augmentation (ct_augment, scan_augment, spatial_augment, crush) |
| split | validation_mask regions inside trained segments (interpolation), or a whole dataset held out | crop regions + 32-voxel margin held out (24 crops on 0139 / 1667 / Paris 4) plus every wbench box |
| scoring | val balanced accuracy, val BCE unsmoothed (2-D) | 3-D AUC / F1 at known voxels (ink-vscore) + 2-D render + ink-score |

What we adopt, in this order (lead, 2026-11-11):
1. **The normal-pooled MIL mode as a second ink target.** Per label pixel, keep the mesh point and normal. In the loss, take the max
   (or soft-max) of the 3-D ink logits over ±R voxels along the normal and apply BCE to the 2-D label. This recovers the pixels the
   face search leaves unknown; face labels stay where a face is found. The rays come from the inkfly index; the loss sits in
   head_slab_pass (univA). Measure both modes on the 24 held-out crops.
2. Label smoothing 0.1 on the ink BCE.
3. The published validation_mask_v2 regions as extra holdouts where they exist, so our numbers compare with theirs.
4. BCE + 0.25 dice as the starting ink loss.

Skipped for now: the scroll quotas and the self-training chains.

Item 2 is in as `train --ink-smooth EPS` (nn_ink_targets_smooth; default 0 so running recipes do not change; test_ink_mil checks it).

## B1 4.8 um phase rasters: what was built before the switch to on-the-fly phase (2026-11-11, branch fibd-20261111)

Workstream B1 rasterised the band phase at 4.8 um (`phase-raster --gs 2 --pair 1 --radius 160`, GPU, byte-identical to the CPU path;
32 shards in 17 s, ~5x the 9.6 bytes). Done: PHerc1667 A / B over the whole scroll (77 / 64 min, 46 / 50 GB). Paris 4 June A was
limited to the packed shards (spacing < 150 um share >= 0.625: 549 of the 9.6 phaseA shards, 339 boxes) and ran on pod 2's GPU:
the first run outpaced the desk pull and filled the pod disk (130 boxes lost to ENOSPC); with a box ledger and a disk cap it reached
280 / 339 boxes before the pod was terminated, ~500 shards still in flight. On the desk: 2998 4.8 shards, 107 GB; 316 of the 549
packed 9.6 shards are complete. Then the user decided that the sampler computes the winding phase from the codes on the fly (univA),
so no further stored phase sets are built and the missing boxes are not planned. `scripts/univ/mksources_univ.py` keeps the partial
Paris 4 set as `PHercParis4-june48` with `"partial": true`, weight 0.2 x 316/549, and a holdout file (`p4A48.holdouts.json`: the wbench
boxes first, then the 233 incomplete 9.6 shards as 2048^3 level-0 boxes) so no window touches a missing region; the ledger is
`scripts/univ/p4A48.boxes.ok`. All stored 4.8 entries are replaced by the "onfly" source key when it lands.

Update (same day): onfly is on master; 1667 A48 / B48 and Paris 4 June 48 (codesA48 rebuilt on the desk in 1352 s, 43 GB; phase on a
check tile byte-identical to the pod's stored shards, 8 / 8) are onfly entries, Paris 4 over the whole scroll. The fibre-only entries
use the onfly fibre planes (afv + the mklabels2 / mklabels3 recipes), so the fibhv-d rasters are not rebuilt.

## The fibre H / V planes on the fly from the polylines (2026-11-11, branch univ-20261111)

Item 5 of the on-the-fly list: fiber-raster's H / V prob254 planes (labels/ 55 GB + labels2/ 6 GB on the desk) computed per read box
from the AFV / UFIB polylines, so a fibre source needs the CT and the tracing only.
- fiber-raster's shard body is now `fr_box` (any box: the polyline query and steps over the box + the reach radius + guard + ring,
  the stamp, the CT air test, the combine); a voxel depends only on the steps within the reach and its own CT byte, so any box gives
  the shard raster's bytes. `fiber_onfly_open` (fiberraster.h) wraps it as two derived arrays (zarr3.h `z3_open_derived`) over a CT
  level, H and V from one computation (a reading thread keeps its last box, so the V read after the H read is free).
- Sources: a target `{"fibre": {"afv", "plane": "H" | "V", "radius", "guard", "ring", "iso", "extra", "map" | "tr"}}` on any channel
  (fibre_h / fibre_v, or recto / sheet for the older fibre runs); computed on the CT level of the read, so every level has native
  planes (radius etc. in that level's voxels, as fiber-raster --level). Channels with the same options share one context.
- GPU (`nn_fibre.cu`, `--onfly-gpu`, buffers for (P + 64)^3 reserved before the planner): the steps (host: from the polylines,
  compacted to those whose reach cube meets the box) are binned on the device to 4 x 4 x 8 voxel blocks (count, cub scan, fill), one
  thread per voxel classes each step's code (<= radius, <= radius + guard, stamped) in float with a 1e-2 margin and falls back to
  fr_stamp's double arithmetic (no FMA, IEEE sqrt) only near a class boundary; the planes depend on the codes only through these
  classes. CPU fallback (UFSM_FIBRE_CPU=1 or no nn_fibre.o).
- Byte identity: test_fiberraster (fixture: plain, --ring --extra, --ct; whole / partial / outside boxes; CPU and GPU), test_fibre_onfly
  (fiber-onfly vs fiber-raster on curved crossing fibres over the wind fixture, GPU and CPU; three --det --fib-head runs, stored /
  onfly GPU / onfly CPU: identical losses), and PHerc1447 (labels/, radius 2 guard 3 iso 64): 26 windows of 512^3 and 896^3, 0
  differing bytes against the stored planes.

Per 896^3 PHerc1447 window with fibres (1.6-1.9M steps), desk 5060 Ti (GPU 1):

| stage | CPU fr_box (before) | GPU, 8^3 blocks, double | + int cubes, float classes | + 4x4x8 blocks, device bins |
|---|---|---|---|---|
| polylines (sqlite query + steps), CPU | 0.07-0.15 s | same | same | same |
| stamp + combine | ~13 s (512^3: 2.4-3.9 s) | 0.8-0.9 s kernel | 0.45 s | 52-117 ms kernel |
| bins / compaction (CPU) | | 0.13-0.2 s | 0.1-0.2 s (8^3), 0.8-1.0 s (4x4x8) | 0.06-0.12 s (compaction only) |
| H2D + D2H (pinned planes) | | ~0.4 s | ~0.4 s | ~0.1-0.2 s |

Empty windows (air / no tracing): ~0.1-0.15 s GPU (the planes' copies). The CT air test reads the window's CT again (the CLI: 0.7-1.3 s
on one thread, no cache; in training through the shared decoded-CT cache the sampler's own read warmed). Still on the CPU, and why:
the sqlite polyline query and the step generation (~0.1 s, irregular), the step compaction (~0.1 s), the CT read for the air test
(the window's CT is on the device for the GPU input build, but the label planes are built on the host: the device-resident label
plane step removes both copies).
## 3-D ink targets from the human labels: inventory, the open face, held-out crops (2026-11-11, branch inkgt-20261111)

Goal (universal net U4): train the ink head on human ink ground truth instead of only teacher rasters, and score it against the
same labels on held-out regions.

**Inventory.** Every published human ink label set is 2-D: per-segment `inklabels` + `supervision` (+ some `validation`) on the
tifxyz canvas (label px = mesh px). Two copies exist: HF scrollprize/datasets `ink/<scroll>/<seg>` (TIFF + zarr) and open-data
`<scroll>/segments/<id>/ink-labels/<um>-volume-<scan>/20260918/` (OME zarr), each next to a `tifxyz-transformed` mesh on the
same scan. They are one label set (20260918). The open-data metadata.json names each set's target scan. Licence CC BY-NC 4.0
(open-data LICENSE.txt). HF `ink_9um` holds the same labels pooled to 9.6 um; `ink/unused` holds model predictions only.

| scroll | segs | scan (level 0) | supervised mm^2 | ink |
|---|---|---|---|---|
| PHerc Paris 4 | 8 | 20260411134726 2.400 um | 8230 | 26 % |
| PHerc1667 | 6 | 20251217075048 2.399 um | 2868 | 24 % |
| PHerc0139 | 8 + 2 HF-only (w016, w030_2026) | 20260102150214 2.399 um | 981 | 32 % |
| PHerc0841 | 3 | 20260319124803 2.403 um | 340 | 25 % |
| PHerc0009B | 1 | 20250820154339 2.401 um | 258 | 22 % |
| PHerc0500P2 | 1 | 20250526151718 2.215 um | 220 | 21 % |
| PHercMAN5 | 1 (HF only) | 20260311104824 2.399 um | 65 | 29 % |
| PHerc0814 | 1 | 20260309142202 2.399 um | 54 | 31 % |

All 8 scans are on the volcomp mirror. HF's 841 and 814 meshes are not on those scans: their CT depth profile is flat. The open-data
copies are aligned and are the ones used (scripts/inkgt: segs.tsv rows = the 28 open-data sets plus the 3 HF-only ones, checked
not to overlap the open-data meshes).

**Where the label sits in depth.** CT depth profiles (`ink-splat --profile`, 2048^2 labelled window per segment, all 31) show the
same two things everywhere. First, +N of ink-splat's normal (du x dv) points to the umbilicus: N . (axis - P) > 0 on 94-100 % of
the vertices of every Paris 4 / 0139 / 1667 segment. Second, the open face is on +N (papyrus bulk on -N). The mesh's distance from
that face is not fixed: the papyrus-fraction edge sits at t = +2 .. +16 voxels (2.4 um) depending on the segment. This is the
sheet-pbr finding (labels 1-8 voxels outward of the tracked recto) seen from the label side. A fixed band around the mesh therefore
labels a varying slice of sheet interior.

`ink-splat --face A,B` places the band per label pixel relative to the open face instead of the mesh:
- Along sign N, t = --face-search -6 .. 20 level-0 voxels, it samples the output level's CT (trilinear, [1 2 1] smoothed).
- The face f is the end of the first papyrus run (>= 3 samples >= iso) that ends at t >= -4 and starts at t <= 8, provided >= 2
  samples below iso follow.
- The label goes to t in [f + A, f + B]. A pixel without such a face stays unknown.

With A,B = -6,0 (the outermost ~14 um of papyrus at the recto face), iso 64 (the CT valley is 54-70 on these scans) and 4.8 um
output, the face is found on 52-83 % of supervised pixels per scan (841 / 0009B / 0500P2 ~53 %: packed sheets; MAN5 83 %). Face
offsets p10..p90 are -3 .. +17 voxels. Face-relative profiles have papyrus fraction 0.94-1.00 over t = -6 .. 0 and 0.00 at +2,
for ink and no-ink pixels alike. The ink - no-ink CT difference in that band is small and of either sign (-8 .. +8), so the labels
are not a density signal at this resolution (as in sheet-pbr).

**Targets.** /vesuvius/ufsm/inkgt/gt/<scroll>.zarr: prob254 (254 ink, 0 no ink, 255 unknown) on the scan's level-1 grid (4.8 um)
+ known-max pyramid, 512^3 shards, 3-74 MB per scan. Each scan is the merge of every segment's 16384^2 label windows
(scripts/inkgt/windows.py, splat_all.sh: --face -6,0 --iso 64 --level 1, about 40 CPU-min for 2.2 G supervised px). They are
stored, not splatted in the sampler, because a per-window splat would need the meshes and 2-D labels inside the loader. Held out
(--exclude, and the sources' holdout_file /vesuvius/ufsm/inkgt/holdouts-ink.json):
- every wbench box, with the 9.362 um boxes of 0139 / 0814 / 0500P2 mapped into the fine scans by the xres reg maps
  (`ufsm reg map --box`);
- the 16 Paris 4 ink eval crops (wbench/inkeval/crops);
- 8 new crops on PHerc0139 (w040, w044) and PHerc1667 (w029, w013) in the same layout (scripts/inkgt/mkcrops.py ->
  /vesuvius/ufsm/inkgt/evcrops; box = the crop surface's bbox +- 32 voxels).
The held-out crops are also splatted alone (splat_eval.sh -> inkgt/evgt/<crop>.zarr) as their 3-D references.

**Scoring.** `ufsm ink-vscore --gt R K --pred R K --box ...`: at the voxels known in both, AUC (tie-aware), best F1, F1 / P / R at
0.5. scripts/inkgt/baseline.sh PRED runs it on the 24 crops (3-D), plus render + ink-score (2-D, the latres column) for networks.
Floor, the CT itself as the score: 3-D AUC 0.513, best F1 0.443 (= the all-ink F1 at ~30 % ink) over the 24 crops.
(`ink-score` had lost its dispatch in main.c in 129afd3; restored.)

## The winding phase on the fly from the codes (2026-11-11, branch univ-20261111)

User: the phase computed from the codes per window, "on the gpu", "so that we dont have so many artifacts". A 4.8 um phase set is ~10x
its codes (PHerc1667 A: phase 46 GB vs codes 4.5 GB; volcomp q0, the ramp varies everywhere) and a deterministic function of them.

- `z3_open_derived` (zarr3.h): an array whose `z3_read` computes its values from a base array; meta, chunk map, `z3_box_any` and shard
  presence come from the base, so every phase reader (phase_window direct / upsampled / interpolated, crush layers, occupancy) works
  unchanged.
- `band_onfly.h`: the pair rule (band_field_phase_pair) over the read box plus a halo of ceil(rmax) + 2, in tiles that fit a device
  budget, on the GPU (band_pair_gpu, slots reserved before the trainer's planner; `--onfly-gpu / --onfly-slots / --onfly-mb`), CPU
  fallback. Sources: `"phase": {"onfly": {"root", "key", "rmax"}, "level": L}` (rmax = phase-raster --radius / --gs, codes voxels).
- Byte identity: the phase at a voxel depends only on labelled voxels within rmax of it. Measured: PHerc1667 A48 / B48, 3 + 3 windows
  of 896^3, 0 differing voxels against phase-raster's stored sets; test_phase_onfly (fixture: CPU, GPU, 64 MB tiles with seams, and
  two --det training runs on the stored vs the onfly phase: identical losses).
- Paris 4 p4full (--gs 4) was computed on the 4.8 um codes pooled 2x inside phase-raster; `ufsm codes-pool2` writes those pooled codes
  (praster_read, byte-exact) so the onfly phase over them matches p4full. (Pooling on the GPU cannot match: CUDA's atan2 differs from
  glibc's at the exact half-step ties the circular mean produces.)

Cost (per 896^3 window, rmax 80 at 4.8 um / 82 at 9.36 um):

| where | tiles (voxels / window) | pair rule | codes read (CPU decode) | total |
|---|---|---|---|---|
| desk 5060 Ti (shared with u1), 1 GB tiles | 64 x 436^3 (4.7x) | 8.5 s | 1.2 s | 8.8 s |
| + warp-per-line x pass (12x on that kernel) and envelope chunks sized to the tile (1.7x) | 512^3 test: 8 tiles | 0.62 s for 8 x 420^3 (was 1.2 s) | | |
| RTX PRO 6000 pod, one 1064^3 tile (16 GB budget), before the kernel changes | 1 (1.67x) | 1.5 s | 2-4 s | 4.7 s |

Then (same day): the codes decoded on the GPU (volcomp lossless mode 1: k_ll_sub one thread per substream with the chunk's 8 tANS
tables in shared memory, k_ll_build the zigzag / prefix rebuild and the scatter into the box; the CPU only fetches the raw chunks and
packs tables + aligned substreams; test_lldec: byte-identical to z3_read on SPARSE / DENSE / RAW / CONST blocks, seams, partial edge
chunks), only the tile interior copied down (pinned), the envelope scratch without the F array:

| PRO 6000 pod, 896^3 window, 9.36 um, rmax 82 | decode | parity 0 + 1 | pair | D2H | wall per window |
|---|---|---|---|---|---|
| one 1064^3 tile, codes decoded on the CPU | 2-3 s (CPU, 16 threads) | 0.47 s | 14 ms | 1.0 s pageable | 4.7 s |
| one 1064^3 tile, GPU decode | 15-20 ms | 0.40-0.48 s | 14 ms | 14 ms (crop) | 0.73-0.81 s |
| 8 tiles of 616^3 (8 GB budget) | 5-19 ms each | 37-73 ms each | 3 ms | 3 ms | 0.86-1.1 s (GPU ~0.53 s) |

Training on the pod (u1 recipe, P 896, --mem fast, replay 3), an onfly PHerc0139 source vs a stored-phase 0500P2 source: 1.21 vs
1.55-1.68 samp/s, loader wait 0% in both: the pair rule's ~0.5 s of GPU per window, shared over the window's 3 uses (~0.17 s per
sample), on the same GPU as the training. Remaining CPU work: the raw chunk fetch / pack (~10-40 ms), the window memset and the copy
of the interior into the sampler's host label plane (~0.25 s of one worker thread).

Stored artifacts the universal sources still need, and why (2026-11-11):
- CT (volcomp pyramids, the mirror): the input.
- codes (winding_mod14 rasters of the q-assigned surfaces): the phase's input; the next step derives them from the meshes.
- stored phase for the fused sets (PHerc0139, 0500P2, the 1667 9.6 um "phase"): their phase is the circular mean of two code sets
  where they agree; onfly computes one set. Paris 4 p4full: until its 2x-pooled codes are written (codes-pool2) or it is relabelled.
- recto / verso, normal, density, wind4 targets: already derived per window in the sampler from the phase (no stored rasters).
- fibre H / V rasters: on the fly from the AFV / UFIB tracings since the section above (sources "fibre" targets); the stored
  labels/ and labels2/ can go once no sources file references them.
- teacher rasters, registered-twin warps: stored; on-the-fly versions are the later items of the list.

nsys on a 420^3 tile (desk): x pass 45 -> 1.8 ms, the y / z envelopes 99 -> 57 ms (90% of what remains), pair kernel 4-8 ms, copies
13 ms. Rastering a whole set once is ~1.6T voxel-work (1667 A: 77 min on the desk GPU); one u1-length run at ~16% onfly windows
recomputes ~15T. What remains on the CPU and why: the codes decode (volcomp lossless mode 1: nn_gpudec only decodes the lossy mode 0),
the padded-box copy and the D2H of the phase into the sampler's label planes (the sampler builds planes on the host). Next: GPU decode
of the lossless codes chunks, the envelope passes in shared memory, and the phase kept on the device for the GPU input build.

## Universal net: the fibre head in unwrap2 (--fib-head L) (2026-11-11, branch univ-20261111)

Why: `--fib-dir` (fibhv-c / fibhv-d: the H / V fibre masks plus the tube direction) ran as its own task. It used the two batch target
channels, needed a whole-window loss, and excluded every unwrap2 head. The universal net (U5) needs it as one more head on the
shared trunk.

How:
- **Targets:** two new target slots, "fibre_h" and "fibre_v" (`CH_FIBH` / `CH_FIBV`, NTG 5), each a prob254 tube raster with
  255 = unknown. A builder maps fibhv data's recto to fibre_h and sheet to fibre_v.
- **Planes:** they become two more aux label planes (roles `AUX_FIBH` / `AUX_FIBV`):
  - The registry now holds up to 5 aux planes (sampler `batch.aux[0..3]`, `aux_ch[j]` = the plane's target slot; `ink_ch` when
    the fibre H plane is plane 0).
  - Every per-plane stream array grows to `NLPL` = 8.
  - The two fibre planes are streamed with ±`g_fd_r` halo planes per head slab (`lhalo`); the slab upload, pointer and stream-check
    code is now one loop over the planes.
- **Head (`--fib-head L`):** channels after rv: H mask, V mask, then the direction t·tᵀ (6) and the family bit (V).
  - Masks: soft BCE over the known voxels (`nn_ink_targets`).
  - Direction: per slab from the two planes (`nn_fib_head_dir`: `fib_dir_k` with 255-unknown planes), as a 7-BCE without
    --pos-weight, like --fib-dir.
  - Weights: λ × the source's "fibre" head weight. The logged loss is λ (mask bce + dir bce); the step line shows
    "fib bce … dir …".
- **Padding:** more than 16 outputs are padded to a multiple of 16 (outputs `g_pad_ch..`, zero gradient). The 1³ MX head kernels
  run in passes of 16, and 20 or 22 outputs had no head backward.
- **Init and header:**
  - `--init-from` a net with the same layout without the head (u1: cout 13) copies rows 0..12; the fibre rows start fresh
    (bias `--fib-bias` −2 on the masks).
  - The runtime header carries the --fib-dir channel map at its offset ("fib_dir": {"head": 1, "channels": {"H": 13, …}}).
- **Not yet:** with `--lat-guide 3`, the sub-cell layout still caps the interpolated outputs at 16, so the fibre head runs with
  `--lat 4` alone or with voxel trunks. Lifting that cap is the sub-cell rv / ink step.

Tests:
- `test_fibhead`: on random tubes, every 24-plane slab's direction targets equal `nn_fib_dir_targets` over the whole window
  (13962 targets). With a third of H unknown, none of the unknown voxels gets a target.
- `test_fibhead_train` (wind fixture tube rasters `fibh` / `fibv`, sources-fibhv.json):
  - The voxel run logs the fibre losses on 100% of windows.
  - The latent `--rv 2` run's stream check finds 0 of 26.7 M slab bytes differing (the halos included), and it logs identically to
    a run with the planes kept on the device.
  - Init-from rows 0–12 and resume both work, and the checkpoint header has the channel map.

## Data-parallel training: one host thread per GPU (2026-11-11, branch univ-20261111)

Why: u1 (P 896 latent, two desk GPUs) ran at 0.54 samples/s with one host thread launching both GPUs. A step queues more kernels
than the driver's launch queue holds, so the thread blocked in GPU 0's backward launches while GPU 1 sat idle (perf: host time in
the launch path; each GPU ~75 % busy, idle out of phase). The first fix, phased launches (every GPU's forward before any backward,
d1924ea), gave 0.62 samples/s (+13 %). This step gives each GPU its own persistent host thread, so each GPU's launches block only
that GPU. The target is a step close to the slower GPU's compute alone.

How:
- `train.c` gets a pool of persistent workers (`gpool_post` / `gpool_wait`), one per GPU, started on first use. They are kept
  alive because their thread-local device scratch must outlive a step.
- Each step is `gstep_fwd` + `gstep_bwd`, run by the GPU's worker. The main thread keeps the GPU-ordered `sampler_next` /
  upload / release, the gradient exchange, the losses and the optimizer.
- The schedules: `UFSM_MGPU_THREADS=0` gives one thread with phased launches, and `UFSM_MGPU_PHASED=0` as well gives forward +
  backward per GPU in turn. All three draw the same numbers:
  - The SR counter is per GPU (base `g << 24`; GPU 0 gets what one GPU always had) and is carried from the forward to the backward
    in `gpu_state.sr_ctr`.
  - The `--lat-step-jitter` generator is per GPU (`lat_js`).
  - `gpu_state G[8]` is now zeroed, because the fields added later relied on garbage.
- nn's mutable per-call state is `thread_local`: the layer / sub / pass / block ids, the SR counter, pos weight, the logit /
  head / loss fp16 flags, `gn_const`, `g_actbf`, `g_gyrec_off`, `g_f4w_gy4`, and `unet.c`'s `g_in_bwd`. A worker starts from the
  main thread's values (`nn_tls_capture` / `nn_tls_install`). The storage registry takes a mutex.
- `split.c` hands the same `nn_tls` on with the turn, so `--split z`'s two alternating threads still see one sequence of it.
  Without this, `test_cover_training` (split) went non-finite.

The races the gate found (each one only changed bits in some runs):
- **`g_gyrec_off`:** `unet_backward_x` sets it around decoder blocks. One thread's restore could leave the other thread's records
  off.
- **`g_f4w_gy4`:** a "the gy holds MX-fp4 rows" flag that `lp_bwd_w_f4` sets around one call. It aborted ("need the gy
  pre-pass") or decoded the wrong rows. It was found by bisection: `UFSM_GRAD_MX4=0` gave 0 / 30 differing runs, against 25 / 30
  with the flag shared.
- **The fine-stem backward's float `atomicAdd`s:** these made `--det 1` runs differ in the last bits, and a discrete choice later
  amplified the difference into visible loss changes. The fix is per-block partial rows, added in block order by `fstem_red_k`.
  test_fstem is unchanged: 3.5e-4 relative against the host.

Gate: `test_univ2` now also runs the u1 recipe (`--det 1`, 30 steps, latent, rv, rungs, FiLM, fstem, uw, grad-cos 3, replay 3) under
threads, phased and in turn, and requires identical step / validation / grad-cos lines.
- Before the fixes: 7–10 of 10 threaded runs differed.
- After: 0 / 12 (full recipe), 0 / 12 (replay 1), 0 / 8 (no fstem).
- Phased and in-turn runs agree with each other in every repeat.
- One GPU is identical to the previous binary.
- The parameter gradients still differ in ~700 of 394 k entries between two identical phased runs (FiLM / stem / a few GN
  affines: ~3e-8 relative). These come from other float atomics in the fp16 head-gradient path. The MX-fp4 trunk gradients absorb
  them, and they do not show in the logged lines.

## Universal net A6: the fine voxel stem (--fstem C, --fstem-drop P) (2026-11-11, branch univ-20261111)

Why: the latbeat verdict above. The latent net's phase deficit against wind96 is resolution: 38.4 um cells hold 1.5-4 cells per sheet
period at 60-150 um spacing. **`--fstem 16`** (latent grid, `--lat 4 --lat-guide 3`, task unwrap2) adds a second path into the head
beside the trunk. It reads the CT channel itself at stride 2: per 2^3-voxel block a 4^3 conv (taps 2 i - 1 .. 2 i + 2 per axis, zero
outside the window), 16 silu features (19.2 um at 9.6 um voxels), and a 16 -> nout projection added to every head output that the
head slabs hold as voxel logits. That is every output except the fused normal / density group: phase, wind4, rv, ink (7 with --rv 2
--ink-head 1). It is added after the upsampling + sub-cell shuffle and before the losses, and in the backward the parameter gradients
are read straight from the slab's gradient planes before the adjoints. Parameters (W1 16 x 64, b1, W2 7 x 16) are appended last;
W1 is He-initialised and b1 / W2 are zero, so the net computes its donor's function (step 1 logs the stem-less run's losses exactly,
test_heads (7)). AdamW (non-matrix). Header key "fstem":[C,nout] only when on; `--init-from` copies a stem with the same shape; a
resume must repeat --fstem. **Level dropout** `--fstem-drop 0.25`: a training window runs without the stem (forward and backward),
drawn per GPU and step, so the net keeps working without the fine level. Validation and predict always use it. `predict` adds it in
the fit-field (11 planes) and `--channel rv` / side-pair paths. With a stem it falls back from the latent fast paths (lat_lean,
direct rv, latin, fit-field z-slabs), which read no CT; those are to be ported when U2 needs the speed.

Kernels (`src/nn_fstem.cu`): row tiles of up to 128 blocks of one (z, y) block row, the 16 input rows staged in shared memory once.
The forward does the 64 -> 16 product on the tensor cores (mma m16n8k16 fp16, W1 in fp16). The backward recomputes the same way and
does the parameter-gradient products dW1 = sum dpre x taps and dW2 = sum go x h in tf32 MMA (fp32 range: the pre-activation gradients
are ~1e-5 and would be fp16 subnormals), in half tiles (22 KB of shared memory, 4 blocks an SM). P 256 window, RTX 5060 Ti:

| version | forward (x2 a step: statistics + gradient pass) | backward |
|---|---|---|
| per-thread scalar (taps from global) | 2.1 ms | 9.6 ms |
| + register-blocked outer products, 128-block tiles | 2.1 | 3.4 |
| + staged rows (row tiles) | 2.2 | 4.3 |
| + tensor-core forward / recompute | 0.74 | 4.0 |
| + half tiles (occupancy 17 -> 33 %) | 0.74 | 3.1 |
| + tf32 parameter products | 0.74 | 2.8 |
| + gradient gather on all threads | 0.75 | 2.5 |

ncu on the last backward: long-scoreboard (global loads) and barrier stalls dominate. Next levers: double-buffered staging, the
gather through one aligned 4-byte pass. Training step, P 256, latent, all heads, `UFSM_BENCH_SAME=1`, 40 steps, no dropout:
widths 32..128: 26.2 -> 30.9 ms (+18 %); widths 96,128,160,192: 45.6 -> 50.4 ms (+10.6 %; ~8 % at the default dropout 0.25). The
stem scales with voxels, the trunk with voxels x width^2, so wider trunks pay less.

Tests: `test_fstem` (new): the kernels against a double host reference on a 10 x 12 x 14 window in two slabs. The reference's gradient
is taken by central differences over all 1152 parameters. Forward worst |error| / (1 + |want|) 1.8e-4; gradient 3.5e-4 relative to
the largest (fp16 W1 in the tensor-core products, tf32 sums; the fp32 scalar kernel gave 1.6e-7). The other channels stay
untouched, and an odd slab start is refused. `test_heads` (7): step 1 with --fstem 16 logs the stem-less run's losses; header
[16,7]; resume; a resume without --fstem is refused; predict --channel rv runs. Fixture, 8 steps, all heads, latent, P 64: loss
3.268 without -> 3.256 with the stem (from step 2 on, every step lower). That is a fixture number; whether the stem closes the
packed-sheet gap is U2's measurement (wbench Paris 4 packed boxes).

## Universal net A7: learned task weights (--uw 1) and the head-gradient cosines (--grad-cos N) (2026-11-11, branch univ-20261111)

**`--uw 1`** (Kendall, Gal and Cipolla 2018): one learned log variance s_h per head (phase, normal + density shared, winding mod 4,
rv, ink; the fibre-normal term is a regulariser, not a likelihood, and keeps its lambda). Each head's gradient is scaled by
exp(-s_h) through the same per-source head multiplier as A2's "heads" key (`src_head_w`), so no new device path. The objective is
sum_h exp(-s_h) L_h + s_h, with s_h stepped on the host by Adam from each step's logged head loss (dL/ds = 1 - exp(-s) L).
The rate is --uw-lr (1e-3), and s is clamped to [-4, 4]. The s_h and the Adam step count are written to the checkpoint header
("uw":{"t":..,"s":[..]}, only with --uw) and resumed. The step line shows the weights in effect. Without --uw every multiplier is
the float 1, so the old layouts are unchanged (test_heads).

**`--grad-cos N`**: every N steps, the cosines between the heads' gradients at the head input (the trunk's shared level-0 features).
The head is a 1^3 conv W (cout x C), so the input gradient of task a summed over the cells is W^T g_a. The pairwise dot products are
sum_{i in a, j in b} (W W^T)_ij S_ij, with S the Gram of the logit-gradient planes. S is accumulated per head slab with `nn_gram16`
inside the step's single backward: one K(K+1)/2-pair reduction per slab, versus a backward per task. Channels map to tasks as
follows: interpolated outputs by head; latent guided / sub-cell outputs (16-143) to the phase. Cost: zero on steps that do not log.
Gradients of the deeper trunk are not separated (they would need a backward per task); the head input is where all heads meet.

Tests: `test_gram` (new): fp16 / bf16, K 1 / 7 / 13, strided rows, two accumulating calls against a double host sum; worst error
relative to |x_i||x_j| is 1.2e-7. `test_heads` (6): latent + all heads with --uw 1 --uw-lr 0.05 --grad-cos 2; the cosine lines
name every pair, the header holds t 4 and a resume continues to t 6.
Fixture numbers (6 steps, P 64, all heads): |cos| <= 0.17 between every pair, i.e. the heads barely interfere at the head input on
this fixture. The largest values are normal/wind4 (+0.16..0.17 at step 4) and normal/rv (-0.11..-0.17). After 6 steps at lr 0.05
the weights moved phase 1.02, normal 0.80, wind4 1.31, rv 1.35, ink 1.26. These are numbers from a 6-step fixture run; a
real run's cosines and the weights' end point are for U1 / U2 to measure.

## Universal net A4 + A5: levels by physical voxel size, per-level scale FiLM (2026-11-11, branch univ-20261111)

**A4, `--um-rungs U[:W],..`** (sample.h `nrung`): instead of `--levels` (a probability per pyramid index, the same index for a 2.4 and a
9.4 um scan), each window draws a physical rung U (weight W, default 1) among the rungs its source reaches and reads the source's level
nearest U (|log2(level um / U)| <= `--rung-tol`, 0.5 octave; levels whose labels are stored coarser than the level are not a rung of
that source, so a global rung list over mixed scans needs no per-source levels; `label_res_check` still refuses them under `--levels`).
`--rung-shift-p P` moves the drawn rung to a list neighbour (scale augmentation over whole octaves, labels read at that level as now;
the continuous `--zoom-*` warps stay for in-between scales). Validation keeps the rungs without the shift. The start-up log lists per
source which level serves which rung. Deliberately not done: resampling a scan to the exact rung (a 7.9 um scan at the 9.6 rung reads
7.9 um voxels; the scale FiLM below gets the true size) and a window fixed in mm (`--fov-mm`): with fixed-size device buffers a fixed
physical field of view needs every window resampled to one grid, which for a 9.4 um scan on the 2.4 um voxel grid of the L9.6 trunk
is a 4x upsampling that reads a 4^3 = 64x larger cube than it uses (or a read path that reads the central part and resamples, touching
every label reader). Both can come later if the cross-resolution gate (U1: one checkpoint within 0.01 of itself at 4.8 / 9.6 / 19.2
feeds) needs them; the feeds themselves are native levels, which this trains.

**A5, `--film K`** (unet_cfg.film, K 1 or 2): every encoder and decoder block's output GroupNorm affine is FiLM'd,
gamma (1 + sum_k c_k A_k), beta + sum_k c_k D_k, with c = log2(um / 4.8) of the window's grid (b.um: level x zoom / affine) and, for
K 2, of the level read (b.level's voxel size, no zoom). A / D are per channel and code, appended last in `unet_create`, zero-initialised
(a donor's function is unchanged; `--init-from` copies a FiLM donor's pairs), header key `"film":K` only when on; old checkpoints load
unchanged. Implementation: the forward copies the live weights into a forward copy (as `--l0-norm ws` does) and writes the effective
affines there (`src/nn_film.cu`, one tiny kernel per norm, 2 L - 1 of them), so every GN reader (fused conv + GN statistics, recompute,
the slab paths, the next block's input norm) sees them without new code paths; the backward turns the effective affines' gradient
into gamma's, A's and D's (exact: the gradient is zeroed before each backward, as for `scale_cond`). The old `--scale-cond` stem bias
stays. B 1 (one code vector per forward; every unwrap task is B 1). `predict` sets the codes from the CT level's voxel size
(`--scale-um` / `--film-data-um` override). Cost: one device copy of the weights (4 B per parameter) and 2 L - 1 launches per forward.

Tests: `test_unet~FILM=2` (FD over every FiLM parameter, 112 of them, + 60 random others, random A / D, codes 1 / -0.6): worst relative
error 5.3e-5; the tensor-core path against fp32 on the big net with FiLM: gradient rel L2 0.020 (0.018 without). `test_rungs` (new):
two scans at 2.4 and 7.9 um, rungs 9.6 -> levels 2 / 0 only; rungs 4.8,9.6 -> A both levels about evenly, B level 0 only; shift 1 on
2.4,4.8 -> A both levels, B (no rung) excluded; rung 38.4 -> no sampler. `test_heads` (5): two fixture sources whose labels allow one
level each train together under `--um-rungs 19.2,38.4 --film 2` (latent, `--rv 2`), header `"film":2`, resume. The seven old layouts
still log the reference losses.

## Universal net A1 + A2: the head registry, rv on its own channels, per-source head weights (2026-11-11, branch univ-20261111)

Step A1 / A2 of the universal latent net plan (one net with every head). Before this, ink (`--ink-head`), the rv papyrus (`--rv`) and the
fibre tubes (`--fib-normal-lambda`) shared the single ink plane `IPLANE`, so they excluded each other, and rv took over the phase slots 0-1.

**Aux planes.** `train` now keeps a list of aux label planes, one per enabled head that reads one, in a fixed order: papyrus (rv), ink,
fibre (`aux_setup`, `g_aux_of[role]`). Plane 0 is the sampler's `batch.ink` (stream plane 3) exactly as before; planes 1-2 are the new
`batch.aux[0..1]` (stream planes 4-5), read from the source's "ink" (CH_SHEET) or new "fibre" (CH_FIB, alias "fiber") target slot, made
unknown where the loss mask is off, pulled through the same symmetry / host warp (`sample_warp_ink`), the `--replay` back-to-source-grid
pass and the device warps (`lab_upload` kind 3), streamed per head slab (`lab_stream`, `LSLAB`) or kept on the device (`ldev`, the papyrus
plane in 2 bits, the others bytes). With one head on, every buffer, upload, stream plane and planner term is the old one. The fibre
plane reads the source's "fibre" target, else (no ink head) its "sheet" / "ink" one, so the old fibre sources work unchanged. The only
exclusion left between the three is `--rv 1` + `--phase-trunc` / `--pgrad-lambda` (its side slot is not a phase). Source target slots
grew to NTG = 3 (`CH_FIB`, never a batch channel); `label_res_check` and `level_ok` see it.

**rv on its own pair.** `--rv 2`: recto / verso papyrus on two new output channels after the ink channel (unwrap2: phase 0-1, normal 2-7,
density 8, winding mod 4 9-10, ink 11, rv 12-13; without ink rv 11-12), the phase trained as without --rv; BCE x `--rv-lambda` (logged
as `rv bce` next to `phase bce`). Own slab targets `rvt` / `rvm`, loss sums at `hs_ds + 64` (the slab sums grew to 128 doubles), its own
scratch. Checkpoint header `"rv":{"version":2,"recto":12,"verso":13,...}`; `--rv 1` keeps version 1 and the old layout. `--init-from` a
donor without the pair resets rows 12-13 (zero weights, `--rv-bias`, default 0), and a plain unwrap2 donor (11 outputs) maps rows 0-10
for `--rv 2` as it did for `--ink-head`. `predict --channel rv` on a version-2 checkpoint fuses channels 12-13 (interpolated, never the
guided / sub-cell pair; the direct latent placement paths are off for it); `--fit-*` works on it (the phase is in 0-1). The latent
fused gradient (`g_lat_fuse`) stays on with `--rv 2` (rv in the slab chain).

**Per-source heads (A2).** A source's `"heads": {"phase": 1, "rv": 0, "ink": {"on": 1, "lambda": 0.5}, ...}` (phase, normal, density,
wind4, rv, ink, fibre; a number is a multiplier, 0 off) scales that head's logit gradient on the source's windows (`src_head_w`, folded
into the existing lambdas, so a weight 1 is the same float product); unknown names are refused. Absent labels keep mask 0 as before.

Tests: `test_heads` (new, fast tier, 12 s). (1) the seven old single-head layouts (plain unwrap2, `--ink-head`, `--rv 1`,
`--fib-normal-lambda`, and `--lat 4 --lat-guide 3` plain / rv / ink) log exactly the losses and gradient norms (4 decimals, every step
and the validation) of the binary before the registry (`tests/heads_ref.txt`, per GPU name; the weights are not bit-reproducible run to
run under `--det 1`, 24 873 bytes of 6.2 MB differed between two master runs, the logged losses are). Checked for both builds: VERIFY
(the suite's) and the production fp16 build, whose latent runs take the fused gradient: 7 / 7 equal to master (60ff701) in each.
(2) `--rv 2 --ink-head 1 --fib-normal-lambda 0.1` with the fibre plane from a "fibre" target, whole-window and latent: finite phase /
rv / ink / fibn losses, header names rv 12-13 and ink 11; the latent stream check (`UFSM_LAT_STREAM=check`) 0 of 16 M slab bytes differ,
the device-kept planes (`UFSM_LAT_LDEV=63`) give the streamed run's losses. (3) `predict --channel rv` on both against its channels
12 / 13 (lossless stores): worst 1.0 code. (4) a source with every head at 0 trains with gradient norm 0.00; a misspelt head is refused.
`--rv 2` latent with and without `UFSM_LAT_FUSE` (fp16 build): identical losses. Still open from A1: `--fib-dir` (the H / V fibre mask
task, a surface task with its own whole-window path) is not yet a head of unwrap2, and the `--lat-guide 3` sub-cell head refines only
the phase pair (rv / ink are interpolated).

## Fibre-normal arms: the seed-repeat noise floor and an untested source-mix confound (2026-11-11, branch fibnote-20261111)

Closes the fibre-in-the-winding-net arms (section "Fibres in the winding network", below); the work moves to the universal latent net.
All numbers: wbench core, WIN 384 HALO 64, the 39 clean boxes (ceil >= 0.8), step 4000, paired A - B with 95 % CI (`wbench-stats`).

Noise floor, measured: vc384c-r2 is vc384c's recipe with seed 4 instead of 3. vc384c-r2 - vc384c: p05x +0.002 [-0.008, +0.013], ids
-0.009 [-0.033, +0.011], p1x +0.003 [-0.008, +0.014]. Per set the four Paris 4 boxes alone move +0.020 p05x on the seed. So on these
39 boxes a p05x difference under ~0.01 and an ids difference under ~0.02-0.03 is not a result; per-set (4-box) deltas under ~0.02 are noise.

| A - B (p05x) | all | note |
|---|---|---|
| vcfibA2 (lambda 0.1, map-registered P4 labels) - vc384c | -0.015 [-0.027, -0.003] | P4 -0.028, July -0.022, 1667 +0.021 (ids +0.033) |
| vcfibU (term only where the normal BCE mask is off) - vc384c | -0.016 [-0.027, -0.005] | July -0.025, 1667 +0.012 |
| vc384c-r2 - vcfibA2 / - vcfibU | +0.018 [+0.007, +0.027] / +0.019 [+0.009, +0.028] | the loss stands outside seed noise |
| vcfibA2 - vcfibA (old affine labels) | +0.014 [+0.001, +0.029] | the registration map helped |

Absolute p05x: vc384c 0.668, vc384c-r2 0.671, vcfibA2 0.653, vcfibU 0.652, vcfibA 0.640.

Untested confound: every fibre arm also changed the source list (`sources-wind-gt-fibp4r`: the Paris 4 June / July sources carry the
fibre raster as their "sheet" target, which also changes how those windows are drawn - the level-5 CT papyrus occupancy fallback in
`sample.c` - against vc384c's `sources-wind-gt-nvme`). So the Paris 4 / July loss may be the data mix, not the term. The control
(vcfib0: vcfibA2 exactly with lambda 1e-9 - lambda 0 switches `IPLANE` off and changes the data path) was prepared in
`runs/vcfib.launch.sh` and never run; run it before drawing a conclusion about the term. Not run either: lambda 0.03, the pair term.

## Latent winding net vs wind96: where the gap is (2026-11-11, branch latbeat-20261111)

wbench core, 39 clean boxes (paired, 95 % CI): wind96 (voxel, WIN 640) p05x 0.677; best latent wz896-12000 (wz768 8000 + 12000 steps,
P 896, WIN 640) 0.624 (-0.035 vs wind96 at WIN 384, lat-wind96-ct 0.662: -0.038 [-0.057, -0.017]); wzlo2-8000 (wind96 init-shift 2,
`--init-lr-mult 0.2`) 0.589, equal to wzlo-8000 (`--trunk-lr-mult 0.2`) +0.003 [-0.010, +0.016]: the inherited-layer lr is not the lever;
wz896's extra training is.

New: `ufsm field-diag A_DIR [B_DIR] --phase ROOT KEY [--bins 60,100,150,250] [--tsv 1]` (src/fielddiag.c): two fit-field sets against
the label phase raster per label sheet-spacing bin (um / |wrapped grad phase|): phase error |wrap(phase - label)| (confident cells), p10
(share of labelled cells within 0.1 turn and confident), confidence, normal alignment n^T T n / tr T against the label-phase gradient,
|log2| spacing error. A = wz896-12000 (WIN 640), B = wind96 at WIN 384 (lat-wind96-ct: the only wind96 core field set kept), all 39 clean
boxes, cells weighted equally:

| set | spacing um | share | p10 A / B (A-B) | conf A / B | nal A / B (A-B) |
|---|---|---|---|---|---|
| all | <60 | 0.026 | .184 / .195 (-.011) | .177 / .196 | .782 / .779 (+.003) |
| all | 60-100 | 0.166 | .290 / .319 (-.029) | .246 / .285 | .879 / .881 (-.002) |
| all | 100-150 | 0.329 | .405 / .425 (-.020) | .327 / .364 | .910 / .916 (-.006) |
| all | 150-250 | 0.333 | .432 / .443 (-.011) | .412 / .425 | .906 / .918 (-.012) |
| all | >=250 | 0.146 | .503 / .517 (-.014) | .520 / .510 | .838 / .857 (-.019) |
| paris4 (4) | 60-100 / 100-150 | .34 / .37 | -.032 / -.035 | .214/.246, .314/.364 | -.000 / -.009 |
| paris4j (28) | 60-100 / 100-150 | .14 / .34 | -.030 / -.020 | .254/.298, .318/.353 | -.003 / -.006 |
| 1667 (7) | 60-100 / 100-150 | .05 / .14 | +.028 / +.041 | .415/.435, .557/.560 | -.007 / -.007 |

Per box: corr(share of cells with spacing < 100 um, p05x A - B) = -0.31; the 5 most packed boxes (share >= 0.25) lose -0.101 p05x,
the 25 middle ones -0.027, the 9 least packed -0.033.

Verdict: mostly resolution. The latent net's phase deficit is largest at 60-150 um sheet spacing (6-15 label voxels, 1.5-4 latent
cells of 38.4 um per sheet period) and on Paris 4, where most sheets are that packed; its confidence there is 10-15 % lower; on 1667
(wide sheets) its phase is better than wind96's. A smaller second deficit is the normal head at wide spacing (-0.012..-0.05 nal), where
the latent spacing head is also worse (sperr .54 vs .39 at 150-250 um) while it is better in packed sheets. Next (queued): a stride-2
voxel stem into the cell-shuffle head (resolution), larger windows (P >= 1152, context), distillation from wind96, more gated data.

## fibhv-d: the fibre mask model on every published fibre set (2026-11-11, branch fibd-20261111)

Why: fibhv-a..c trained on two scrolls only (PHerc1447 + PHerc0175A `*_1mm_v2`, 8.64 um) and were judged on Paris 4 HOLD, a scroll and
scanner they never saw; traced lines stayed short (below). fibhv-d = the fibhv-c recipe (H / V masks + direction head `--fib-dir 1`,
P 512, MX-fp4, ring labels radius 2 / guard 2 / ring 3) initialised from fibhv-c step 21000, on every set we can rasterise:

| source | fibres | positives | ring veto (`--extra`) | grid | metres | weight |
|---|---|---|---|---|---|---|
| PHerc1447-fib2 (labels2) | PHerc1447_1mm_v2 | > 1 mm tracings | PHerc1447_all_v2 (all lengths) | 8.64 um L0 | 5065 | 0.277 |
| PHerc0175A-fib2 (labels2) | PHerc0175A_1mm_v2 | > 1 mm | 0175A_SMALLMORE (all lengths) | 8.64 um L0 | 3915 | 0.214 |
| PHercParis4-fib3 | Paris4_1mm_v2 (2023 frame) through the ufsm reg map | > 1 mm | Paris4_all_v2 @map + Paris4_v1 (native) | 9.6 um (2.4 L2), z 6656..11776 | 3528 | 0.193 |
| PHercParis4-spiral | open-data spiral dataset `fibers.afv` (931,104 H / V, mean 3.6 mm, native 2.4 um) | all | Paris4_v1 | 9.6 um, z 7000..17000 | 3332 | 0.182 |
| PHerc0125-fib3 | PHerc0125_3mm_v2 | > 3 mm | none published | 9.362 um L0, z 9216..12288 | 197 | 0.030 |
| PHerc0139-hf / PHercParis4-hf / PHerc1451-hf / PHerc0332-hf | public human fibre lines (HF `spiral/<scroll>/fibers`) | all | none | 9.6 um (2.4 L2) | 30.7 / 12.9 / 7.2 / 2.5 | 0.03 / 0.03 / 0.03 / 0.015 |

Choices:
- Unfiltered tracings (`*_all_v2`, `0175A_SMALLMORE`: 6.5 M / 5.8 M fibres, ~84 % of them < 1 mm) are vetoes, not positives: they are
  the same tracer's output before the publisher's > 1 mm cut, so as positives they would add mostly fragment tubes of unknown quality;
  as vetoes they keep untraced short fibres out of the negative ring (the fibhv-b design). `*_5mm_v1` (older tracer, same fibres) is not
  used: its axes sit a few voxels off the v2 ones and would blur the tubes.
- Paris4_v1 (native 2.4 um chains) has no H / V families, so it can only veto.
- The spiral-dataset `eval_fibers/*.json` are left out (evaluation fibres of that dataset).
- Weights: by traced metres, a floor of 0.03 per source (0.015 for the 2.5 m 0332 set). No air negatives (`--ct`) for the new rasters:
  the sampler draws windows around papyrus inside stored label chunks (3d84ada), so no CT is read while rasterising.
- Holdouts: the wbench boxes (incl. Paris 4 HOLD, level-0 49152,16384,14336 + 2048,2304,3072) plus 1447-HOLD (5120,3840,3072 +
  512,768,768, the densest 1447_1mm_v2 cell). fibhv-a..c trained on 1447-HOLD, so it is a cross-check only.

Code: `fiber-raster --extra A[@map],B..` (up to 4 AFVs, each native or through `--map`); `reg_map_load` (src/regmap.c, the map
without opening its volumes). Paris4_1mm_v2 through `reg_map_load`: one shard row byte-identical to the l2r raster (40 of 40 shards).
Scripts: scripts/fibers/mklabels3.sh, mksources3.py, fibhv-d.launch.sh, evald.sh (pod: predict + trace both boxes, 1447 mask vs
labels), evald_desk.sh (desk: GT judge, lengths), fibre_lens.py.

**Measuring continuity.** `fiber-trace --min-len 48` drops lines shorter than 48 prediction voxels = 0.46 mm at 9.6 um, so the plain
median of a traced set sits just above that cut whatever the model does (fibhv-c: 0.55-0.57 at every step). fibre_lens.py adds the
arc-weighted median and the arc share in lines >= 1 / 2 mm. fibhv-c on HOLD with those (step 3000 -> 12000):

| family / trace | lines | arc m | arc-median mm | >= 1 mm | >= 2 mm |
|---|---|---|---|---|---|
| H tube | 5335 -> 6988 | 3.18 -> 4.40 | 0.58 -> 0.61 | 0.045 -> 0.095 | 0.002 -> 0.007 |
| H dyad | 2771 -> 4218 | 1.61 -> 2.48 | 0.56 -> 0.57 | 0.029 -> 0.043 | 0.000 -> 0.000 |
| V tube | 6074 -> 7259 | 3.73 -> 4.56 | 0.60 -> 0.61 | 0.074 -> 0.087 | 0.002 -> 0.002 |
| V dyad | 642 -> 7509 | 0.34 -> 4.64 | 0.52 -> 0.60 | 0.000 -> 0.070 | 0.000 -> 0.000 |

Launch (pod 1, RTX PRO 4000): 9 sources, init 59 layer tensors from fibhv-c 21000; step 100 at 199 Mvox/s, loader wait 21 % (the
Paris 4 / human-line CT streams from the mirror into the cache), 217 Mvox/s and 0 % wait by step 11600. Evals at steps 3000, 9000,
15000, 21000, 27000 (pod 2 predicts and traces, the desk judges).

HOLD (Paris 4, now in-domain), fibhv-c 12000 -> fibhv-d 3000 -> 9000 (lines, arc m, arc-median mm, >= 1 mm, >= 2 mm; pairs at 0.6 mm):

| trace | fibhv-c 12000 | fibhv-d 3000 | fibhv-d 9000 |
|---|---|---|---|
| H tube | 6988, 4.40, 0.61, 9.5 %, 0.7 %; .955 | 9899, 6.48, 0.64, 13.5 %, 0.4 %; .951 | 11041, 7.41, 0.66, 15.9 %, 0.9 %; .941 |
| H dyad | 4218, 2.48, 0.57, 4.3 %, 0.0 %; .932 | 7232, 4.42, 0.60, 7.0 %, 0.0 %; .948 | 6597, 3.93, 0.58, 5.3 %, 0.0 %; .950 |
| V tube | 7259, 4.56, 0.61, 8.7 %, 0.2 %; .864 | 10380, 6.76, 0.63, 13.4 %, 0.6 %; .900 | 10578, 7.00, 0.64, 13.9 %, 0.8 %; .886 |
| V dyad | 7509, 4.64, 0.60, 7.0 %, 0.0 %; .904 | 11791, 7.65, 0.63, 11.5 %, 0.2 %; .887 | 12857, 8.44, 0.64, 13.0 %, 0.3 %; .894 |

More traced fibre (+68 % arc on H tube at 9000) and a larger share in lines >= 1 mm at about the same sheet fidelity; lines >= 2 mm
stay under 1 %. 1447-HOLD (cross-check) stays flat between 3000 and 9000 (H tube arc-median 0.59, V tube 0.64). The 1447 mask-vs-label
score in evald.sh does not work yet (`ufsm eval` reads 0 valid voxels from the copied label shards). The first desk orchestrator never
started the pod eval: its `pgrep -f` guard matched its own ssh command line; evald_desk.sh now uses a pid file.

## phase-raster's pair rule lean and on the GPU, byte-identical (2026-11-11, branch labelfast-20261111)

The Paris 4 phase labels (`phase-raster --gs 4 --codes-grid 1 --pair 1 --radius 160`, the p4full flags, 3,549 shards in the box) ran
on pod 2 at ~8 shards / min on 4 threads, ~7 h per set; 4 threads because each block thread held ~0.8 GB (a 256^3 block plus the
42-voxel halo, 340^3 = 39 M voxels at ~22 B: codes, a band buffer the pair rule never reads, two seed masks, two float distance arrays,
two int32 feature-index arrays) under the pod's 31 GB cap.

Profile (desk, 8 dense shards 17-18 / 8-9 / 8-9 of June, perf): Felzenszwalb-Huttenlocher envelopes 40 %, the EDT gather / scatter
20 %, the pair loop 22 % (index decodes), the zarr read / volcomp encode ~5 %. All compute; the CPU path had ~3.6 cores busy.

Changes (src/band.c, src/main.c, src/nn_band.cu):
- **Lean EDT**: `edt_feature_cap(..., d2 = nullptr, idx)` keeps only the feature indices; each pass rebuilds the squared distance from
  the index (after the x pass the feature lies on the voxel's x line, after y in its z plane). Those distances are exact integers
  (< 2^24) and so equal the stored floats; the envelopes see the same values, the indices come out identical. `band_field_phase_pair`
  and `band_field_side_pair` use it; the pair rule reuses one seed mask and skips the band buffer: ~10 B / block voxel instead of 22.
- **GPU pair rule** (`--gpu D [--gpu-slots K]`): the same x sweeps, y / z envelopes (one thread per line, consecutive x per warp, the
  envelope in [q][line] scratch, 32 k lines per chunk) and pair kernel, built `-fmad=false` without fast math (IEEE div / sqrt,
  `lrintf`) like nn_gpudec / nn_cf, so every float rounds as in band.c (gcc -std=c23 does not contract). A pool of K device contexts
  (~0.5 GB each at these flags) shared by the block threads; any CUDA failure falls back to the CPU. Weak symbols: a binary without
  nn_band.o runs the CPU code.

| 8 dense June shards (desk, 5060 Ti) | wall | peak RSS |
|---|---|---|
| before, `--threads 4 --shards 1` (the pod's setting) | 54.5 s | 3.4 GB |
| before, `--threads 8 --shards 3` | 21.5 s | 18.6 GB |
| lean CPU, `--threads 4 --shards 1` | 62.9 s | 2.0 GB |
| lean CPU, `--threads 8 --shards 3` | 14.8 s | 10.5 GB |
| GPU, `--threads 8 --shards 3 --gpu 1 --gpu-slots 1 / 2 / 4` | 4.6 / 3.7 / 3.6 s | 2.4 GB |

GPU: 15x the pod's old setting at 0.7x its RAM; 64 dense shards (z 18-21, y / x 6-9) in 38.5 s (0.6 s a shard). The lean CPU loses 15 %
per thread to the index decodes but halves the RAM, so the same 31 GB holds twice the threads. Now the CPU side (read, pool, encode)
bounds it: more slots buy nothing past 2.

Byte identity: the 8 and the 64 shards equal the desk `v2/rv/gt/p4full/phaseA` shards byte for byte; `tests/test_band_pair.c` (suite,
gpu tag) checks `band_field_phase_pair` (lean CPU and GPU) and `band_field_side_pair` against the previous band.c code kept verbatim
in the test, on synthetic spirals with holes, conflicts and a missing wrap, five box shapes (non-cubic and an empty one), two radii:
0 differing voxels. `scripts/pod/p4kit.sh`: `PS=3 PR="--gpu 0 --gpu-slots 2"` for the phase steps (resume keeps the shards written).

## Manual winding annotation: ufsm annotate-serve / annot-export / annot-regions (2026-11-11, branch annot-20261111)

Why: the automatic judges disagree with the eye (CT coherence ranks 1218 / 1299 low, the fibre judge sees 1299 cross sheets in some
tiles); the user wants to enter winding labels and verdicts by hand. A local tool, no Python: `ufsm annotate-serve CT-ROOT CT-KEY --um U0
--store ANN.json [--level L] [--wraps DIR] [--labels ROOT KEY] [--fibers F.fib] [--regions REGIONS.tsv]` serves one page (src/annot_page.html,
embedded with #embed) on 127.0.0.1 only. Launcher: `scripts/annot/serve.sh PHerc1299|PHerc1218|PHercParis4 [PORT]`.

- Viewer: three planes (z, y, x) through a shared cursor, CT pyramid levels >= L as zoom (shift+wheel), wheel steps the slice, drag pans.
  Overlays: wrap cuts (each wrap's vertex grid, stride 2, cut by the plane per request; colour and number = wrap index), the label array
  (phase / codes, base level only; 255 = unknown), fibres within 1.5 voxels of the plane (UFIB, H yellow / V cyan), annotations.
- Speed (desk, PHerc1299 level 2): a 600 x 400 slice 0.13 s cold (decoded 128^3 chunks are kept in an LRU, --cache-gb 2: neighbouring
  slices and pans decode nothing new); wrap cuts of 109 snapped wraps over a 2048^2 region 0.12 s; fibre points (5.7 M loaded) 0.01 s.
  The 109 wraps load in a background thread (the .sfc decoder is not thread-safe) in about a minute; the page polls and redraws.
- Annotation types (level-0 voxel coordinates): point (p, optional winding w, optional same_as), pair (a, b, dw = w(b) - w(a)),
  line (pts, optional w: one sheet followed by clicks), region (box, verdict good / bad / unknown), wraperr (p, wrap, kind cuts / skips
  / doubles / other). The server validates each item, assigns id / time / author (client values are replaced), writes the snapshot
  (ANN.json, atomic rename) and appends every operation to ANN.json.log (add / del / undo; never rewritten). Undo reverts the newest
  add or del of the session.
- Work queue: `ufsm annot-regions OUT.tsv --judge T.tsv ... [--wrapcheck W --wraps D --um0 U]` ranks fibre-judge tiles by persistent
  sheet changes per metre of fibre (tiles under --min-arc 0.3 m dropped) and adds non-"real" wrap-check verdicts at the wrap's bbox
  centre (whole-wrap flags: the centre is only a starting point). The page's "next flagged region" button walks it.
- Consumers: `ufsm annot-export ANN.json`
  - `--fib OUT.fib`: same-sheet runs (lines, dw 0 pairs, same_as points) as UFIB (family 'M', level-0): `fiber-export` into a fit box and
    `fit --fibers F --uw-rep-fib W` (branch fibers-20261110) use them as slip-repair votes today.
  - `--pairs OUT.tsv`: every relation "za ya xa zb yb xb dw" incl. dw != 0 and pairs of winding-numbered points within --dmax. A fitter
    hook for dw != 0 (fit --annot / --w-annot next to uwr_fib) is not written yet: uwr_fib lives on fibers-20261110, not on this branch.
  - `--gate OUT.boxes --level L`: region verdicts in the `phase-coh --box-scale` format (good 1, bad 0; unknown omitted).
  - `--points DIR --like ROOT KEY --level L [--radius 2]`: sparse winding raster (w mod 254, 255 unknown) on a label grid.
- Test: test_annot (store validate / del / undo / reload / log lines, wrap cut flat + tilted + region filter, server on a z3w fixture
  pyramid: /info, /slice bytes on two levels incl. the array edge, /ann add / invalid / snapshot, page, 404).

## Physically based sheet rendering: what ink looks like on the surface without an ink model (2026-11-11, branch pbr-20261111)
`ufsm sheet-pbr <tifxyz-dir> <out-dir> --ct ROOT KEY --crop U0,V0,N [--ss S]` renders the papyrus surface of a published segment crop
as a physical object (src/sheetpbr.c). Surface: the tifxyz grid through Catmull-Rom splines (C1, no 48 um grid-cell creases), outward
normal by the crop's convexity (recto = inner face, faces.h convention). Profiles along the normal every 0.25 voxel (trilinear CT,
`--interp cubic` tricubic). Recto face: the steepest rise (air -> papyrus) of the smoothed profile, first on an 8-px lattice within
[-15, +5] voxels of the mesh, robustly smoothed (Gaussian 3 cells, points > 3 voxels off dropped, 4 rounds), then as ONE continuous
surface by dynamic programming (row Viterbi then column Viterbi over the offset within +-6 voxels of that, transition cost 0.5 |dt|,
|dt| <= 1 voxel per sample, column pass tied to the row result), refined sub-voxel (parabola through the derivative) within +-0.5
voxel of the DP surface. Verso: the steepest fall around its neighbourhood median. Shading of the recto height field seen from the
recto side: Lambert + GGX (alpha 0.35, F0 0.04), 12 raking lights at 15 deg with cast shadows, horizon AO, exposure 2 / 98 % + gamma.
Diagnostics: near-surface CT (first 4.8 um inside), thickness, high-pass / roughness / curvature of the height, `--xsec` cross-sections
with the faces drawn, `--layers` CT on parallel layers, `--profiles`.

Data: Paris 4 segments 20260701183125-w028-037, ...26-w038-045, ...27-w046-052 (open-data bucket, the 2.4 um tifxyz on
20260411134726, CT level 0 from the volcomp mirror, q 8). Crops (2048^2 px = 4.9 x 4.9 mm) chosen where the published 2D ink
prediction covers ~35 % of a fully surfaced 2 x 2-tile block. The reference is that published ink PREDICTION
(new_canon_autoresearch_recipe, a model output): no human ink labels exist for these 2026 segments; the HF bucket's
ink/phercparis4/wNN_<2023 segment> sets are on the 7.91 um 2023 scan and need the 2023 -> 2026 registration.

Findings:
- The 2.4 um meshes are tilted / offset relative to the sheets (cross-sections: the bands cross t = 0 at ~2 deg, tens of voxels over
  a crop) — consistent with segments carried over from the 7.91 um scan. Sheets in the inner wraps are packed: no clean void, gaps are
  8-10 voxel dips, ~20 voxels between interfaces (fibre layers / sheets), so a per-sample interface finder hops between interfaces
  (first draft: piecewise-flat plateaus). After the DP: per-sample steepest rise more than 1 voxel off the continuous surface at
  35.5 / 37.8 / 39.0 % of samples (w046 / w038 / w028); runner-up edge >= 0.7 of the best (ambiguous) at 7.5 / 9.3 / 9.3 %; faces
  found at 86-87 %; mean recto-verso distance 20-25 voxels (48-60 um).
- The shaded surface shows the papyrus itself (horizontal / vertical fibre strips, relief, crackle in the near-surface CT); fine
  diagonal hatching in the raking lights is a sampling artefact (0.25-voxel DP offsets), not texture.
- AUC against the ink prediction (pos >= 160, neg <= 64; raw / detrended by a 0.7 mm Gaussian; tie-aware ranks):

| map | w028-037 | w038-045 | w046-052 |
|---|---|---|---|
| flat CT at the mesh | 0.577 / 0.539 | 0.530 / 0.526 | 0.433 / 0.420 |
| max CT +-3 | 0.586 / 0.539 | 0.553 / 0.546 | 0.426 / 0.412 |
| near-recto CT | 0.567 / 0.527 | 0.567 / 0.558 | 0.462 / 0.463 |
| thickness | 0.479 / 0.529 | 0.462 / 0.451 | 0.538 / 0.576 |
| recto height | 0.568 / 0.605 | 0.558 / 0.522 | 0.518 / 0.517 |
| height high-pass | 0.486 / 0.484 | 0.493 / 0.492 | 0.495 / 0.494 |
| roughness | 0.548 / 0.644 | 0.494 / 0.423 | 0.550 / 0.458 |
| curvature | 0.510 / 0.507 | 0.503 / 0.501 | 0.505 / 0.505 |
| RTI min (12 lights) | 0.489 / 0.438 | 0.503 / 0.532 | 0.466 / 0.499 |
| PBR (GGX + shadows + AO) | 0.457 / 0.445 | 0.497 / 0.502 | 0.471 / 0.485 |
| best single CT layer | t -35: 0.553 / 0.582 | t -2: 0.532 / 0.529 | t +2: 0.433 / 0.419 |

  No physical surface map separates the predicted ink: every AUC is within 0.40-0.64 and the sign flips between crops (flat CT 0.58
  on w028, 0.43 on w046). Two readings, not yet separated: carbon ink leaves no relief / density signature our sampling resolves at
  2.4 um with q 8 compression; or the reference itself (a model prediction, on a mesh that does not sit on one face) marks letters at
  a layer our tracked recto is not on. Next: human-labelled segments (the 2023-scan labels through the new registration), the 1.129 um
  scan (meshes exist, on-20260608103018), --ss 2 / 4 on sub-crops, and the face on the side the ink model's input layers sit.
1.129 um (`--reg MAP.json`): the mesh stays on the 2.4 um scan, every sample point goes through the published ufsm reg map
PHercParis4/transforms/20260608103018_to_20260411134726/reg.json (affine + field, held-out p50 0.85 um) into the 1.129 um scan
20260608103018, the normal through the map's local Jacobian; offsets and all voxel parameters in 1.129 um voxels (ratio 2.126),
sampled at --ss 2 (1.13 um along the surface). Coverage: the 1.129 um scan spans 2.4 um z ~28000-56600; crop w028-037 (z ~63000) is
outside it and w038-045's crop lies in its masked-out zeros, so 1024-px sub-crops were used: w046 (centre of the w046 crop) and two
substitutes picked the same way inside the coverage, w028s (u 103424, v 27648) and w038s (u 90112, v 44032). The registration holds
at the surface: cracks, inclusions and fibre strips coincide in the 2.4 and 1.129 um renders (out/cmp_*.png, the same 1 x 1 mm).
At 1.129 um the recto edge is less decided: runner-up edge >= 0.7 of the best at 26-31 % of samples (2.4 um: 8-9 %), the DP surface
departs from the per-sample steepest rise at 42-56 % (2.4 um: 23-36 %): more noise per voxel, and fine structure (fibre bundles,
crackle) makes several real edges. AUC vs the ink prediction, 2.4 um | 1.129 um, raw / detrended:

| map | w046 | w028s | w038s |
|---|---|---|---|
| flat CT | 0.307 / 0.372 \| 0.331 / 0.411 | 0.424 / 0.425 \| 0.449 / 0.465 | 0.650 / 0.613 \| 0.629 / 0.558 |
| near-recto CT | 0.434 / 0.502 \| 0.457 / 0.510 | 0.544 / 0.548 \| 0.538 / 0.546 | 0.520 / 0.551 \| 0.582 / 0.566 |
| thickness | 0.630 / 0.827 \| 0.650 / 0.583 | 0.399 / 0.402 \| 0.431 / 0.465 | 0.590 / 0.505 \| 0.564 / 0.452 |
| recto height | 0.546 / 0.574 \| 0.562 / 0.555 | 0.459 / 0.460 \| 0.481 / 0.492 | 0.567 / 0.635 \| 0.578 / 0.599 |
| roughness | 0.515 / 0.524 \| 0.493 / 0.516 | 0.574 / 0.365 \| 0.593 / 0.415 | 0.495 / 0.383 \| 0.506 / 0.377 |
| RTI min | 0.506 / 0.392 \| 0.507 / 0.350 | 0.455 / 0.513 \| 0.470 / 0.656 | 0.532 / 0.353 \| 0.515 / 0.291 |
| PBR | 0.487 / 0.461 \| 0.509 / 0.451 | 0.450 / 0.480 \| 0.441 / 0.525 | 0.515 / 0.514 \| 0.506 / 0.506 |

  The finer scan does not change the picture: no map separates the predicted ink consistently, the signs flip between crops
  (flat CT 0.31 on w046, 0.65 on w038s), and 1.129 um numbers track the 2.4 um ones within a few hundredths.
Outputs: /vesuvius/ufsm/pbr/out/<crop>/{contact.png, png/*.png, *.f32, map.json, auc.json}.

### Addendum: against HUMAN ink labels (2026-11-11)

- The public human labels (HF scrollprize/datasets ink/phercparis4, 8 segments w00..w09) are already on the 2.4 um
  20260411134726 scan (their meshes reach z ~74000, x ~26000; the 2023 scan's shape is 14258 x 7888 x 8096). So no 2023 -> 2026
  map was needed. Pipeline: ink-splat each label segment (label + supervision mask) into one sparse prob254 zarr (band +-4
  voxels, 255 = unknown), then `sheet-pbr --labels`. For each output pixel the sampler walks the normal +-D around our tracked
  recto. It picks the labelled run nearest the face and returns its majority label (human_ink) and its offset from the face
  (human_off). Voxels outside the label box count as unknown, not as no ink.
- Picking crops by geometry alone gave 0 % labelled on 9 crops: the label segments run near our sheets but are unsupervised
  there. So crops were re-picked by distance to supervised label pixels (tools/pbr hcand2.txt; >= 60 % of the mesh vertices
  within 12 voxels). That gave L1-L5 plus w028 and hC. w028s is 7.4 % labelled and its label surface sits -28 voxels from our
  face, i.e. on the neighbouring sheet, so it is excluded.

| crop (seg, u, v, px) | labelled | ink | label - recto mean / mean abs (2.4 \| 1.13) |
|---|---|---|---|
| w028 (w028-037 122880 10240, 2048) | 99.2 % | 36.5 % | +4.6 / 9.8 \| - |
| hC (w028-037 61440 21504, 1024) | 26.6 % | 50.6 % | +7.3 / 11.3 \| +6.9 / 11.5 |
| L1 (w028-037 77824 19456) | 100 % | 38.9 % | +5.5 / 7.9 \| +5.8 / 8.0 |
| L2 (w028-037 54272 26624) | 99.6 % | 47.6 % | +3.4 / 6.5 \| +4.0 / 6.7 |
| L3 (w010-027 62464 23552) | 100 % | 40.3 % | +3.1 / 6.6 \| +5.3 / 8.4 |
| L4 (w010-027 72704 37888) | 99.5 % | 42.9 % | +8.1 / 9.9 \| +7.9 / 9.6 |
| L5 (w038-045 167936 24576) | 100 % | 44.2 % | +0.6 / 7.1 \| +1.7 / 7.3 |

  The label surfaces are on the same sheet as our face. They sit on average 1-8 mesh voxels outward of it (|d| 6.5-10 voxels,
  i.e. 16-24 um, about one sheet thickness of jitter), never a whole wrap off, except w028s.

AUC vs HUMAN labels, raw / detrended (2.4 um; 1.129 um in parentheses):

| map | w028 | hC | L1 | L2 | L3 | L4 | L5 | pooled (z-scored per crop) |
|---|---|---|---|---|---|---|---|---|
| recto offset | 0.41/0.38 | 0.73/0.72 (0.70) | 0.65/0.56 (0.63) | 0.57/0.56 (0.50) | 0.46/0.46 (0.45) | 0.53/0.52 (0.55) | 0.39/0.45 (0.36) | 0.493 (0.529) |
| flat CT | 0.58/0.56 | 0.37/0.38 (0.39) | 0.44/0.47 (0.42) | 0.44/0.47 (0.42) | 0.53/0.52 (0.52) | 0.53/0.52 (0.54) | 0.53/0.50 (0.53) | 0.525 (0.481) |
| near-recto CT | 0.56/0.53 | 0.54/0.56 (0.45) | 0.53/0.52 (0.46) | 0.50/0.52 (0.47) | 0.53/0.53 (0.54) | 0.50/0.50 (0.53) | 0.52/0.53 (0.51) | 0.534 (0.499) |
| thickness | 0.48/0.52 | 0.55/0.51 (0.57) | 0.39/0.44 (0.43) | 0.46/0.44 (0.51) | 0.46/0.45 (0.43) | 0.54/0.54 (0.49) | 0.57/0.56 (0.59) | 0.485 (0.497) |
| recto height | 0.60/0.62 | 0.27/0.28 (0.30) | 0.35/0.43 (0.38) | 0.43/0.44 (0.50) | 0.54/0.54 (0.55) | 0.47/0.48 (0.45) | 0.61/0.55 (0.64) | 0.507 (0.471) |
| roughness | 0.55/0.58 | 0.54/0.63 (0.56) | 0.55/0.44 (0.52) | 0.65/0.38 (0.57) | 0.50/0.49 (0.51) | 0.53/0.45 (0.52) | 0.54/0.50 (0.54) | 0.511 (0.517) |
| PBR | 0.47/0.47 | 0.46/0.45 (0.45) | 0.48/0.45 (0.49) | 0.41/0.47 (0.47) | 0.50/0.51 (0.50) | 0.48/0.49 (0.49) | 0.47/0.49 (0.46) | 0.462 (0.478) |
| RTI min | 0.49/0.47 | 0.46/0.41 (0.49) | 0.48/0.39 (0.50) | 0.42/0.53 (0.52) | 0.49/0.51 (0.50) | 0.48/0.54 (0.49) | 0.48/0.56 (0.48) | 0.467 (0.503) |
| AO | 0.46/0.46 | 0.46/0.46 (0.45) | 0.44/0.43 (0.47) | 0.37/0.41 (0.45) | 0.51/0.51 (0.50) | 0.47/0.48 (0.48) | 0.46/0.49 (0.46) | 0.454 (0.472) |

- Result: against human labels, as against the predictions, no label-free surface map separates ink. Pooled AUCs are
  0.45-0.53. Per-crop values reach 0.27 / 0.73 (hC height, recto offset), but the sign flips between crops: recto offset is
  0.73 on hC and 0.39 on L5, and height is 0.27 on hC and 0.61 on L5.
- Caveat: the human labels are coarse polygons at letter scale. A 1024-px crop (2.46 mm) holds only one or two label
  boundaries (see cmph_L1.png: the crop splits into one ink half and one no-ink half). So a per-crop AUC mostly measures
  "which half of the crop", and detrending at sigma 300 px is close to the crop size. A test with real power needs >= 8 mm
  crops spanning many letters, scored at stroke edges.
- Images: /vesuvius/ufsm/pbr/out/cmph_{L1,L4,hC}.png (human label | flat | near-recto | PBR | raking light | recto offset,
  2.4 um row over 1.129 um row), out/h_{s24,s113}_<crop>/contact_human.png, auc_human.txt.

## fiber-trace on the GPU: the per-voxel axis, byte-identical (2026-11-11, branch gtrace-20261111)

A `--mask` trace is almost all per-voxel work: on the Paris 4 HOLD box (512 x 576 x 768 at 9.6 um, fibhv-a step-10000 H mask) the CPU
run takes 128 s, and perf puts 97 % of it in the axis (the presence's gradient structure tensor over 5^3 voxels and two 40-step power
iterations per voxel, `ft_ridge_axis`); the streamline integration with its sequential "earlier seed wins" occupancy is ~3 %. So only the
axis moved to the GPU; the integration stays on the CPU and its seed order is unchanged, and no parallel-integration approximation of
the occupancy rule is needed.

- `src/ftaxis.c` holds the CPU axis (moved out of fibertrace.c unchanged, so tests link it alone); `src/nn_ftrace.cu` the GPU copy, one
  thread per (y, x) column marching z in 32-plane launches, the in-plane (2R+1)^2 sums per plane in a ring and a sliding z sum.
- Byte-identical, not just close: the gradient products and their sums are exact integers on both sides (the CPU's float partial sums
  stay below 2^24 for --st-r <= 6; the GPU path refuses larger R), and the double power iterations run the CPU's operations in its order
  (nn_ftrace.o: -fmad=false, no fast math, IEEE division / sqrt; gcc -std=c23 does not contract).
- `ufsm fiber-trace ... --gpu G` (default -1 = CPU); on any CUDA failure it says so and runs the CPU axis.
- `test_ftrace_gpu` (gpu, fast tier): random tubes in noise on an odd-sized box (37 x 45 x 53, every face clamp), a flat region,
  --st-r 1 / 2 / 3 x prior none / H / V: 9 of 9 byte-identical.
- HOLD box, desk GPU 1 (shared with a training run): H 128 s CPU -> 6.6 s with --gpu (19x), V 6.4 s; the H and V .fib files are
  byte-identical to the CPU's (cmp), so the judge numbers are those of the CPU traces (H 0.956 / V 0.914 at 0.6 mm, DESIGN 2026-11-10).
  What remains: the sequential tracing (ft_tensor / ft_axis ~35 % of the 6.6 s), the zarr read, CUDA start and the copies. A 512^3
  tile (134 Mvox, 0.59 of HOLD) should take ~4 s instead of ~4 min per pod core.
- Not ported: `--blur` (CPU, off by default), `--dyad` (CPU, already a cheap per-voxel decode), the published-prediction path (no
  derived axis). A GPU integration would only pay once the axis is no longer the cost, i.e. not now.

## Paris 4: the 2023 7.91 um scan registered to our 2.4 um scan (2026-11-11, branch p4reg-20261111)

Why: the published AFV fibre tracings (Paris4_1mm_v2 etc.) sit in the 2023 scan (PHercParis4 20230205180739, scrollprize
`full-scrolls/Scroll1/PHercParis4.volpkg/volumes_zarr_standardized/54keV_7.91um_Scroll1A.zarr`, zarr v2), and the only published
link to our 20260411134726 scan is the open-data affine `PHercParis4/transforms/20260411134726-to-20230205180739-r1` (24 landmarks,
LS rms 2.45 coarse voxels = 19 um, max 6.4 = 51 um; it is the matrix we had in /vesuvius/ufsm/fibers/meta/p4_2026_to_2023.json).
The fibre-normal training arm vcfibA lost on Paris 4 and gained on 1667 with fibre labels brought through that affine.

Method: the existing `ufsm reg` pipeline (pair / boxes / fit / stats), coarse = 2023 (7.91 um), fine = ours. Gaps filled:
- `rvol_open` read zarr v3 only and needed level 0: it now reads zarr v2 pyramids (levels "0", "1", ...) and stores holding only
  some levels (our local Paris 4 copy starts at level 2; the official fibre predictions hold only 3 and 4); `rvol_level_for` skips
  absent levels. test_regz2.
- `reg boxes --maskfill 1` (masked CT 0 -> the box mean before the band-pass). Not used in the final map (see below).
- `fiber-export --map reg.json`: AFV fibres through a reg map with its field (affine-only map = --tr, 6e-5 voxel).
- `ufsm fiber-regscore`: an independent check (no CT matching): per region (a cube around a random fibre point, placed with one
  fixed affine so all maps see the same regions), mean fibre presence of the official prediction on our scan along the mapped AFV
  fibres, and the per-axis shift maximising it (parabola) = the local residual. `--identity` / `--afv-fine` score fibres on a CT
  in their own frame or the other way round.

Checks before fitting: AFV v2 on the 2023 CT with no map: per-axis bias <= 0.7 um at 2023 levels 0 / 1 / 2 (the AFV frame is
Scroll1A's, although the AFV declares a 14258-slice base and Scroll1A has 14376). The 2023 CT is masked inside the scroll too
(median box coverage 0.84), so the default coverage gate 0.95 kept 8 % of the boxes; 0.75 keeps 82 %.

Passes (coarse level 1 = 15.8 um boxes of 96^3, fine level 2 = 9.6 um, DoG NCC, sub-voxel):
| pass | boxes | CT hold-out p50 / p90 (um) | fibres (100 regions) presence, residual p50 / p90 (um), bias z |
|---|---|---|---|
| published affine | | (eval boxes, below) | 99.8, 17.7 / 48.0, -4.9 |
| A: 334 boxes (mirror fine CT, gate 0.95), affine + field h 768 | 334 | 8.2 / 17.4 | 98.2, 18.3 / 46.3, +11.4 |
| C: from A, local fine CT, gate 0.75, spacing 288, h 768 | 6461 | 5.8 / 17.4 | 124.1, 6.5 / 18.9, +0.6 |
| C, h 512 | 6461 | 4.8 / 13.7 | |
| C, h 384 | 6461 | 4.2 / 12.2 | 127.7, 4.9 / 12.4, +0.5 |
| **C, h 256 (final)** | 6461 | **3.7 / 10.5** | **129.6, 4.3 / 10.4, +0.4** |
| C, h 192 | 6461 | 4.5 / 12.3 | 128.4, 4.7 / 11.9, +0.6 |
| D: --maskfill 1 from the published affine, h 256 | 2439 | 5.2 / 13.9 | 128.1, 5.1 / 14.5, +0.6 |

- Pass A looked good by CT hold-out (20.5 -> 8.2 um) and was not better by fibres: 334 boxes are too few for the affine + field
  (the fit moved z by ~16 um). The dense pass C agrees on both measures. Mask filling (D) gave no gain over C: the mask edges
  were not the problem.
- Evaluation boxes midway between the fit boxes (offset 144, never fitted), final map: 2440 confident, residual p50 0.369 voxel
  = 2.9 um, p90 1.23 voxel = 9.7 um, 31.7 % under 1/4 voxel; NCC p50 0.909. Published affine on the same boxes: 2430 confident, p50 2.278 voxel = 18.0 um, p90 4.54 = 35.9 um, 0.0 % under 1/4 voxel.
- Field 57 x 30 x 33 nodes, spacing 256 coarse voxels (2.0 mm), max |D| 6.0 coarse voxels (48 um).

Output: /vesuvius/ufsm/reg/PParis4_2026to2023/ (reg.json for `ufsm reg map` / `fiber-export --map`, transform.json in the published
schema with the refined affine, field.f32 + field.json, eval_boxes.jsonl, residuals.jsonl, the two fibre-region tables). Work dir:
/vesuvius/ufsm/reg/P4_2023 (passes, logs, scripts runC.sh / runD.sh / rs.sh).

## Fibre votes in fit-join (stitching): `--fibers --join-fib`, measured at the four Paris 4 slab sites (2026-11-11, branch fibers-20261110)

`ufsm fit-join ... --fibers F.fib --join-fib W [--join-fib-arc 0.6] [--join-fib-gap 0]`: a fibre lies in one sheet, so over a short
arc its two ends have the same W up to the network phase change along it. For each slab boundary, every fibre stretch that crosses it
(resampled at 1 native voxel; the phase unwrapped along it as `fit.c` fib_pairs: confidence >= 0.2, gaps <= 2 bridged, a fast change
cuts) gives pairs (a below, b above, <= arc mm apart, both ends >= gap mm from the boundary, a in slab k-1's box and b in slab k's) that
vote round(W_k(b) - W_k-1(a) - dP); |dP| >= 0.5 (a hop the phase sees) drops the pair, residues > 0.3 turn are dropped. The votes'
histogram adds W x -log P_fib(d) to the overlap's chain cost (chain mode and the `--ref` DP; refused with `--global`). Inputs: our own
fibres, `scripts/fibers/wbfib_sites.sh` (fibhv-a step 10000, V family only: vertical fibres cross z boundaries; a +-72 native-voxel
band around the boundary over the whole cross-section, 4 y tiles; tracing ~15 min per 2 Gvox tile), exported into the joined crop's
frame (`fiber-export` now takes UFIB inputs). Test: test_fit_join, `--chain-w 0` (fibres alone) recovers the +3 gauge between two
slabs of the synthetic spiral with 60 of 240 fibres hopping a sheet at the boundary (the unwrap drops them), ids 1.0000.

Measured with `scripts/fibers/wbjoin_fib.sh` (wbench SLABS=1 protocol: wind96 fields, slabs A [0, 96) / B [64, 160) field cells,
interiors [32, 160) / [160, 288) native, one boundary per site; slab fits and joined scores with the pinned wbench fitter b29b1bda):

| site | overlap share(mode) | fibre votes (gap 0.3 mm, arc 1.2 mm) | fibre share(mode) | offset: overlap / fibres alone | joined ids, every variant |
|---|---|---|---|---|---|
| 4624 | 0.490 | 235,485 | 0.546 | +1 / +1 | 0.2403 |
| 5392 | 0.521 | 235,334 | 0.495 | +37 / +37 | 0.3268 |
| 5904 | 0.469 | 321,669 | 0.546 | +101 / +101 | 0.2093 |
| 6672 | 0.388 | 351,522 | 0.325 | -6 / -6 | 0.2207 |

- The fibres alone pick the overlap's integer at all four sites, so `--join-fib 1` and `4` (with or without the gap) change nothing:
  joined ids and p4x are identical to the plain join (difference exactly 0 on 4 of 4 sites; no interval to give).
- Without a gap the votes sample W at fibre points next to the boundary, i.e. the overlap again (shares within 0.03 of the overlap's);
  with both ends >= 0.3 mm from the boundary, beyond the overlap's edges, the shares are the same. Fibre votes are an independent
  confirmation of a whole-cross-section slab offset, not a correction: here the offset is never wrong.
- The joined scores are low (0.21-0.33) because the slab fits are (the A / B fits alone: 0.24-0.37); fit-join has one integer per
  slab pair, so a mis-join that is wrong in one region and right elsewhere (the 1218 / 1299 story) is invisible to it. Fibre votes
  can help stitching only at the piece level: the slip repair inside a fit (`--uw-rep-fib`, +0.021 ids on 15 boxes, 2026-11-05) or a
  per-region join. Next step if wanted: `fit-join` with per-region offsets (the overlap and fibre histograms per y / x tile), checked
  first with the per-tile overlap modes, which this measurement does not log.
- Unrelated, found here: this branch's `ufsm fit` stops with "an illegal memory access" after stage 4 on these 96 x 3328 x 3520 crops
  (sites 4624, 5904, 6672; also on an idle GPU), where the pinned b29b1bda fitter runs (1218 s per slab fit). Not investigated.

## Fibre retrain: ring labels, a direction head, the smoothed-axis check (2026-11-11, branch fibers-20261110)

Follow-up of the section below (our masks mark fibre layers, not tubes).

**Fibre spacing** (nearest other fibre axis, voxels of 8.64 um, 4 random 384^3 boxes per AFV, every 7th resampled axis point):

| AFV | same family p5 / p25 / median / p75 | any family p5 / median | share < 5 / 8 voxels (any) |
|---|---|---|---|
| PHerc1447_1mm_v2 | 9.1 / 12.3 / 14.7 / 17.2 | 6.3 / 12.3 | 0.01 / 0.15 |
| PHerc0175A_1mm_v2 | 8.5 / 11.8 / 14.3 / 16.8 | 6.4 / 12.3 | 0.01 / 0.15 |
| 0175A_SMALLMORE (all lengths) | 0.2 / 0.4 / 0.9 / 11.7 | 0.2 / 6.9 | 0.46 / 0.54 |

The all-lengths tracing is mostly duplicate traces of the same fibres (half its points lie within one voxel of another trace).

**Ring labels** (`fiber-raster --ring K --extra ALL.afv`, `scripts/fibers/mklabels2.sh` -> labels2): positive within radius 2 of an
axis as before, unknown in a guard out to 4 voxels, then 0 in both planes ("no fibre axis here") out to 7 voxels (35..60 um), unless any
fibre of the all-lengths tracing lies within 4 voxels; unknown elsewhere in papyrus as before; air 0. The ring ends at 7 < the 9-voxel
5th percentile of same-family spacing, so a neighbour's tube rarely reaches it. `--resume 1` skips shards already written.
test_fiberraster covers ring / guard / veto.

**Direction head** (`train --fib-dir L [--fib-dir-r 4]`): 7 more outputs after the two masks: the axis dyad a a^T in the normal head's
coding (zz, yy, xx, 0.5 + zy, 0.5 + zx, 0.5 + yx; BCE, unit positive weight) and the family (V 1). Choice: a sign-free dyad, not the
official model's (nx, ny) on the z >= 0 hemisphere: that encoding flips sign exactly at horizontal tangents, i.e. along every H fibre,
a discontinuity a per-voxel regression cannot fit; the dyad is continuous and the tracer already sums dyads (ft_tensor). Targets are
computed on the GPU from the augmented label planes (`nn_fib_dir_targets`: the second moment of the voxel's own family tube voxels in
a 9^3 cube, largest eigenvector, kept where lambda_1 / trace >= 0.6 and >= 9 tube voxels; crossings, unknown and non-tube voxels
masked), so every augmentation (symmetries, rotation, elastic) is consistent by construction and no label plane changes. Whole-window
loss path only (the H / V mask task, B 1; refused with other tasks). test_fib_dir (4 synthetic tubes, |a . axis| >= 0.98 inside).
30-step smoke at P 256: dir bce 1.00 -> 0.87, mask bce / dice as in the same smoke without the head; the checkpoint records
`fib_dir` (channels, coding).

**fibhv-b** (pod, `scripts/fibers/POD.md`): fibhv-a's recipe (P 512, 7 levels, MX-fp4, dice 0.5, pos-weight 4, 30000 steps) on the ring
labels (0175A + 1447, weights by shard rows 0.35 / 0.65 instead of labelled bytes 0.45 / 0.55), 1.7 samp/s at 0-10 % loader wait.
Paris 4 HOLD at step 5000 against fibhv-a at step 5000 (same eval_hold.sh; persistent = net sheet changes per metre, lower is better):

| run, family | papyrus marked (>= 128) | lines | total m | median mm | p90 mm | persistent /m | pairs 0.3 / 0.6 / 1.2 mm |
|---|---|---|---|---|---|---|---|
| fibhv-a H | 0.40 (step 10000) | 12223 | 7.63 | 0.57 | 0.83 | 47.0 | .971 / .957 / .991 |
| fibhv-b H | 0.07 | 8525 | 5.42 | 0.58 | 0.87 | 60.1 | .966 / .926 / 1.0 (37) |
| fibhv-a V | 0.47 (step 10000) | 10323 | 6.40 | 0.56 | 0.83 | 101.2 | .945 / .937 / .992 |
| fibhv-b V | 0.07 | 9213 | 5.77 | 0.58 | 0.85 | 149.6 | .909 / .882 / .983 (57) |

At step 10000 (fibhv-b): H 5103 lines, 3.12 m, median 0.56 mm, p90 0.82, persistent 51.8 /m, pairs .968 / .954 / .889 (9);
V 6853 lines, 4.24 m, median 0.57, p90 0.84, persistent 142.0 /m, pairs .917 / .903 / .964 (28). No longer lines, fewer of them; the
change rate stays above fibhv-a's (47.0 H / 101.2 V). fibhv-b was stopped at 10000 and fibhv-c started from it: `--fib-dir 1
--init-from fibhv-b/snap/10000 --head-map 0,1,-1,...` (the mask head kept, the 7 direction rows fresh; 59 tensors copied), same
recipe, snapshots every 3000; at step 50 mask bce 0.109 (fibhv-b 0.105), dir bce 1.45, 1.6 samp/s, 9.7 GB.

The ring did what it was meant to: the masks now mark 7 % of the papyrus (tubes) instead of 40-47 % (layers). It did not make the traced
lines longer at step 5000 (line ends: support 12657, turn 2276, coherence 1470 for H: the tubes fade, they no longer wander), and
the lines change sheet more often. Step 10000 / 30000 still to come; the direction head (fibhv-c) is the next lever.

**Smoothed axis (`fiber-trace --st-r 3`) as default: no.** Persistent sheet changes per metre (lower is better), base -> st-r 3:
HOLD H 47.5 -> 41.0 (better), HOLD V 90.6 -> 93.7, PH1 H 272.9 -> 282.0, PH1 V 319.3 -> 343.3, PH2 H 204.3 -> 211.9, PH2 V 196.1 -> 212.5,
PH3 H 335.4 -> 354.6, PH3 V 265.5 -> 281.5; the 0.6 mm pair share falls in all 7 of those. Defaults unchanged.

## Fibre continuity: why our traced fibres are short, `fiber-link`, tracer options (2026-11-11, branch fibers-20261110)

Question: our fibres (fibhv-a step 10000 masks traced with `fiber-trace --mask`) are short; can tracing or joining make them longer
without making them change sheet? Measured on Paris 4 HOLD (H family unless noted; masks at 9.6 um, scaledown 2; judge = `fiber-judge
--phase` against GT phaseA as in `scripts/fibers/eval_hold.sh`; "persistent" = net sheet changes per metre of fibre; lengths in mm,
float64 sums: an earlier float32 cumulative sum overstated totals by up to 16 %).

| variant | lines | total m | median | mean | p90 | share >= 2 mm | persistent /m | pairs 0.3 / 0.6 / 1.2 / 2.4 mm |
|---|---|---|---|---|---|---|---|---|
| base (defaults) | 11675 | 7.33 | 0.57 | 0.63 | 0.84 | 0.01 | 47.5 | .969 / .956 / .944 (54) / 1.0 (5) |
| --support 0.1 | 18182 | 11.86 | 0.58 | 0.65 | 0.90 | 0.01 | 54.2 | .969 / .949 / .981 / 1.0 |
| --turn 50 | 17547 | 11.53 | 0.59 | 0.66 | 0.90 | 0.02 | 48.7 | .970 / .956 / .984 / 1.0 |
| --st-r 3 (7^3 structure tensor) | 13983 | 8.55 | 0.56 | 0.61 | 0.83 | 0.00 | 41.0 | .975 / .941 / 1.0 / 1.0 |
| --blur 1 | 53575 | 47.59 | 0.73 | 0.89 | 1.47 | 0.11 | 70.6 | .962 / .932 / .917 / .895 |
| --blur 1 --st-r 3 | 70892 | 76.02 | 0.86 | 1.07 | 1.88 | 0.22 | 61.4 | .968 / .943 / .920 / .864 |
| --blur 1 --st-r 3 --thr 128 --support 0.3 | 57328 | 54.00 | 0.78 | 0.94 | 1.58 | 0.13 | 59.3 | .969 / .943 / .928 / .927 |
| --bridge 5 | 15466 | 9.82 | 0.57 | 0.63 | 0.86 | 0.01 | 54.8 | .967 / .947 / .944 / 1.0 |
| --bridge 10 --st-r 3 | 24920 | 15.82 | 0.58 | 0.63 | 0.87 | 0.00 | 58.0 | .968 / .929 / .954 / 1.0 |
| --axis-prior 1 (family direction in the layer plane) | 11368 | 6.76 | 0.55 | 0.59 | 0.78 | 0.00 | 96.7 | .951 / .880 / .923 / - |
| base + fiber-link 0.1 mm | 11617 | 7.33 | 0.57 | 0.63 | 0.84 | 0.01 | 47.7 | .969 / .955 / .952 / 1.0 |
| base + fiber-link 0.3 mm | 11581 | 7.34 | 0.57 | 0.63 | 0.85 | 0.01 | 48.1 | .969 / .954 / .954 / 1.0 |
| base + overlap merge (Python prototype, 25 um) | 9142 | 6.07 | 0.59 | 0.66 | 0.98 | 0.02 | 53.2 | .968 / .947 / .935 / 1.0 |
| official prediction, direction channels (reference) | 2204 H+V | 6.01 | 2.34 | 2.73 | 4.65 | 0.79 | H 54.7 / V 176.8 | H .969 / .966 / .961 / .952 |

- Stop reasons of the base trace (line ends of kept lines): support 51 %, turn 31 %, coherence 15 %, box 3 %. Along the extrapolated
  tangent beyond an end the mask presence falls from a median 117 to 53 within 13 voxels (0.12 mm).
- The cause is the mask, not the tracer: on HOLD the H mask is >= 128 on 40 % of the papyrus (CT >= 64) and V on 47 %; one of the two
  on 84 %, both on 3 %. The labels (`fiber-raster`) are positive within 2 voxels of a traced > 1 mm fibre and UNKNOWN elsewhere in
  papyrus, so the net learned the H / V fibre LAYERS (a recto / verso-like segmentation), not individual fibres. A layer has no axis of
  its own: the tube structure tensor is ambiguous inside it, and the family-direction prior (axis = H: normal x z, V: z projected into the
  layer) does not help because the layer itself is blotchy along the fibre direction (lines still 0.55 mm, now ended by support 87 %).
- Joining does not recover length: only 1 % of ends have a same-family end within 0.3 mm that continues them (gap, angle <= 30 deg,
  lateral <= 30 um); a line's end is typically 33 um (median) from ANOTHER line running alongside, and 21 % of ends are continued by an
  aligned neighbour within 25 um, but merging onto those (overlap prototype) adds little length and raises the change rate. On the official
  prediction's traces linking finds 4-7 links of 2204: those lines end at real ends, not at gaps.
- Defaults: unchanged. `fiber-link` (default --gap-mm 0.3) is safe to run (no measurable cost in the pair shares or change rate) but gives
  ~1 % more length on today's traces; the tracer options (--st-r, --blur, --bridge, --axis-prior) stay off: each either changes nothing or
  buys length with more sheet changes. --st-r 3 alone lowers the change rate (41.0 vs 47.5 /m) at equal length: a candidate default once
  measured on V and on more boxes.
- What would make long fibres: the predictor. (a) direction channels as targets (the tangent dyad at tube voxels, like the official
  model's nx / ny): with direction the official prediction traces 4x longer lines than from its presence alone (2026-11-10: 74 vs 31
  voxels); (b) negatives between fibres: a 0 shell around each traced fibre beyond its tube, with unknown only where the all-lengths AFV
  (`*_SMALLMORE`, `*_all_v2`) has another trace, so the mask becomes tubular; (c) train at 19.2 um (tube 2 voxels = 38 um), closer to
  the official model's scale. (b) is a label change only (`fiber-raster`), (a) needs a 3-channel head.

New: `ufsm fiber-link IN.fib OUT.fib [--gap-mm 0.3] [--angle 30] [--lat-mm 0.03] [--tan-mm 0.05] [--links rows.tsv]` (`src/fiberlink.c`,
`test_fiber_link`: a broken line rejoins in order, a crossing V line, a parallel and a perpendicular piece stay apart, a cut loop joins
once); `fiber-trace` prints its stop reasons and takes `--st-r R`, `--blur R`, `--bridge K`, `--axis-prior 1` (mask mode).

## Fibre judge in C, streamlines from the official prediction, PHerc1299, the public human fibre annotations, H / V labels (2026-11-10, branch fibers-20261110)

Follow-up of the section below. New commands, each with a fast-tier test on synthetic data: `ufsm fiber-judge` (`src/fiberjudge.c`,
`test_fiberjudge`, which also covers `wrap-dump` and `fit-phase-box`), `ufsm fiber-trace` (`src/fibertrace.c`, `test_fibertrace`),
`ufsm fiber-raster` (`src/fiberraster.c`, `test_fiberraster`). All CPU. Scratch: `/vesuvius/ufsm/fibers/work`; drivers `tools/trace_1299.sh`,
`scripts/fibers/`.

**`fiber-judge`** is `tools/fibjudge.py`'s measure: fibres (an AFV, optionally through a published transform, or a UFIB file of level-0
points) against our winding from the wraps (nearest vertex per wrap-id parity, phase = d1 / (d1 + d2) between the two sheets), a band-phase
store, or a joined fit set. Per 512^2 x NZ tile and family: labelled arc, excursions (hysteresis 0.75 turn after a 9-sample median),
PERSISTENT sheet changes (|sum of the signed excursions| per labelled stretch; excursions - persistent = there-and-back), pairs within half
a turn at each distance. Against the Python on the same 0175A tile: same 214,718 samples, arc 0.9414 / 0.9096 m (0.9405 / 0.9095), excursions
185 / 478 (183 / 476), pairs at 64 voxels 4592 / 5378 (4586 / 5372) - the nearest-vertex search is exact here, the KD-tree's was not;
38 s for the tile against minutes. Its "persistent" is not the Python's "net" (that one rounded end - start of 8-sample medians: 127 / 257
on this tile against 99 / 224): numbers of the two must not be mixed. Paris4_v1 against GT reproduces: 0.9376 at 0.61 mm, 122.8
excursions / m (Python 0.937, 125).

**`fiber-trace`** integrates streamlines through the official prediction (presence, nx, ny). The encoding is not documented with the
data; measured against the uploader's Paris4_v1 streamlines in two level-4 chunks (2538 + 2554 samples): the axis is stored on the z >= 0
hemisphere, t = ((nx - 128) / 127, (ny - 128) / 127, +sqrt(1 - x^2 - y^2)) in (x, y, z): |cos| median 0.996 / 0.997 at presence >= 64 (the
three other axis assignments 0.25 .. 0.95). Tracer = the community description of v1 (seed = presence maximum per 8^3 cell at >= 64,
trilinear sum of presence-weighted dyads, principal axis by power iteration, support 0.2, coherence lambda_1 / trace 0.65 (my reading of
"coherence"), 35 degrees a step, midpoint, 256 steps each way), plus a rule v1 does not describe: a seed within 1 voxel of an earlier line
is skipped (without it every voxel of a fibre's cross-section starts its own parallel line: 142,715 lines instead of 86,597 on the Paris 4
box, same judge numbers). Family: V when the mean squared axial tangent component is > 1/2.
Validation on Paris 4 against GT, slab z 11328 + 512 at 9.6 um, 92 tiles (pairs within half a turn at 0.6 mm; excursions / persistent per m):

| fibres | arc m | H pairs | V pairs | H exc / pers | V exc / pers |
|---|---|---|---|---|---|
| mine, level 4 (38.4 um) | 139.9 | 0.9249 | 0.9084 | 142 / 111 | 201 / 149 |
| Paris4_v1 (no family) | 44.5 | 0.9376 | | 123 / 98 | |
| mine, level 3 vs level 4, the same 16 tiles | 255 / 52 | 0.9135 / 0.9159 | 0.8707 / 0.8668 | 155 / 155 | 267 / 274 |

So my tracer is slightly below v1 (longer lines: it has no length cap by family) and level 3 (19.2 um) buys nothing over level 4 for this
judge: against the wraps the two levels also agree (H 0.762 / 0.760, V 0.681 / 0.675). 19.2 um is 5 voxels per sheet pitch (median gap
of wrapcheck.tsv: 1299 98 um, 1218 93, Paris 4 114, 1447 / 0175A / 0125 101): too coarse to place a line inside a sheet, enough to follow
it, and the judge only asks which sheet the line is in.

**PHerc1299** (official prediction on scan 20260309130042 = our CT; presence aligned with our CT: correlation 0.451 at level 4, 0.09 .. 0.13
for the flipped / transposed frames; presence >= 64 in 16.6 % of papyrus). Slab z 10944 + 512 at 9.596 um (the same slab size as the other
rows), 26 tiles, my level-3 streamlines (66 m labelled):

| scroll, fibres -> basis | H pairs 0.6 mm | V pairs 0.6 mm | H exc / pers per m | V exc / pers per m |
|---|---|---|---|---|
| Paris 4, v2 AFV -> wraps (section below) | 0.761 | 0.709 | 400 | 497 |
| Paris 4, my streamlines -> wraps | 0.775 | 0.728 | 342 / 206 | 417 / 199 |
| 1447, v2 AFV -> wraps | 0.764 | 0.735 | 407 | 477 |
| 0175A, v2 AFV -> wraps | 0.746 | 0.689 | 473 | 528 |
| 0125, v2 AFV -> wraps | 0.588 | 0.570 | 845 | 875 |
| **1299, my streamlines -> wraps (unsnapped, 88)** | **0.596** | **0.674** | 748 / 637 | 506 / 339 |
| 1299, my streamlines -> surf-labels (`latcal/labels/PHerc1299`) | 0.684 | 0.739 | 697 / 660 | 453 / 378 |

By this judge 1299 sits with 0125, below Paris 4 / 1447 / 0175A, and its excursions are 85 % persistent on H (Paris 4: 60 %). What this
does and does not show: the Paris 4 rows say my streamlines judge like the v2 fibres do there; nothing validates the streamlines ON 1299
(no GT, no human fibres published for it), so the 1299 row is an upper bound on our labels' error, not a measurement of it. A second,
local measure does not separate the scrolls: fibre tangent against the normal of the labels (gradient of the phase raster, 128-slice
box): 1299 |t . n| median 0.138 H / 0.134 V (7.9 / 7.7 degrees out of the sheet), Paris 4 against GT 0.120 / 0.159 (6.9 / 9.2). A persistent
rate of 600 / m at a 100 um pitch is a mean slope of 0.06 = 3.5 degrees, inside what the tangent test cannot resolve. The user's own
reading of 1299 is "quite good"; the judge's row does not overturn that, it says the fibre evidence on 1299 is not clean enough to decide.
PHerc1218 has no official fibre prediction (only lasagna and surfaces) and no AFV: see the routes below.

**The official model, its code and its training annotations are public.** Weights: Hugging Face `scrollprize/lasagna-fiber`, ungated,
three checkpoints (`5scroll_json_eligible_refinement_128_sd1_16x8_ema_20260915_212757/snapshots/best_ema_0p999.pt` 220 MB = the model of
the open-data predictions; `s1a_128_1_..._best91_5k.pt` and `s1a_128_2_..._best_25_9k.pt` 344 MB each). Its card documents the two S1A
checkpoints only (U-Net depth 5, base 16, pixel-shuffle decoder, BatchNorm, z-score input, 128^3 patches, 7 raw outputs = 6 direction
values + presence, bf16; "trained on S1A NML fibers registered to the PHercParis4 2.4 um volume"); no licence field. Code: GitHub
`ScrollPrize/villa`, `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/` (`infer`, `train`, annotations, evaluation; revision
9bcb5938 recorded in the predictions, "dirty"). Annotations: the Hugging Face storage bucket `scrollprize/datasets` (listable and
downloadable without credentials today, although this file's first paragraph calls it gated), `spiral/<scroll>/fibers/*.json`, type
`vc3d_fiber`: control points, the traced line (`line_points`, level-0 xyz of the 2.4 um scan, 8 voxels apart), per segment the tracing mode
(trace through a fibre prediction / cspline / lasagna), an automatic H / V tag, branches, review tags. Downloaded (289 MB,
`/vesuvius/ufsm/fibers/raw/hf_fibers`):

| scroll | files | line m (H / V) | control points | tags |
|---|---|---|---|---|
| PHercParis4 | 136 | 12.93 (7.26 / 5.67) | 5,639 | all "reviewed", 4 "problems" |
| PHerc0139 | 411 | 30.66 (15.78 / 14.87) | 13,570 | none |
| PHerc0332 | 77 | 2.48 (2.00 / 0.48) | 13,394 (12,870 cspline) | 7 "reviewed" |
| PHerc1451 | 72 | 7.17 (5.03 / 2.14) | 6,240 | none |

53 m in 696 files on four scrolls; the run name says five and the fifth is not in the bucket (I infer PHerc1299, the fifth scroll with
a prediction; not verified). Also public, not downloaded: the open-data `PHercParis4/representations/spiral/20260411134726-spiral-dataset-20261019084150/`
holds `fibers.afv` (11.1 GB), `fiber_directions.npz` (2.7 GB) and `eval_fibers/*.json`.
The human lines sit on the predicted fibres (Paris 4, level 4: presence mean 146 on the line, 32 two voxels beside it; |cos| to the
predicted axis median 0.997), which is how they were made: mostly traced THROUGH a fibre prediction between human control points.
Against GT they are not cleaner than the automatic fibres. Whole GT band (z 8704 + 3584, 54 tiles, 2.74 m labelled): H pairs 0.850 at
0.6 mm, 340 excursions / m (243 persistent); V 0.797, 500 (323). In the 33 tiles of the 11328 slab that hold both: human 0.793, 446 / m
(295 persistent); my streamlines in the same tiles weighted by the human arc 0.867, 261 (194); Paris4_v1 0.900, 199 (155). Human lines
run on where the automatic tracers stop (through low coherence), and those are the places where the GT phase is least certain too; I
did not separate the two causes.

**Same winding along a fibre as a training constraint: the number.** Persistent sheet changes per metre against GT on Paris 4: Paris4_v1
98, my streamlines 111 H / 149 V, human-reviewed lines 243 H / 323 V; the codes-only floor of the section below, on its 10 good tiles,
was <= 14. One change per 7 .. 10 mm on the automatic fibres, with the GT's own errors inside that figure. A hard constraint is not
supported by any of these sets; a soft pair term over <= 0.3 mm (pairs within half a turn 0.93 .. 0.95) is, and so is the vote use that
already measured positive (`--uw-rep-fib`, section below).

**`fiber-raster` and the H / V mask model (not trained).** `fiber-raster` writes two prob254 volumes on the level-0 CT grid: plane H is
254 within 2 voxels of an H fibre axis, 0 within 2 voxels of a V axis and in air (CT < 64), 255 elsewhere (the guard ring, crossings,
fibres without family, and all papyrus no tracing covers: the AFVs hold fibres > 1 mm only); plane V the reverse. One 512-slice band of
0175A: 171 M H and 178 M V fibre voxels, 4.3 min at 2 threads, 880 MB. `scripts/fibers/mklabels.sh` (1447 whole, 0175A z 1024..11776;
running at the time of writing), `mksources.sh` (targets "recto" = H, "sheet" = V, `encoding: prob254`, wbench holdouts),
`train_hv.sh NAME STEPS` = `scripts/rv/train.sh` with the default two-channel mask task (`--dice 0.5 --pos-weight 4`), P 512, the 7-level
net 16,16,32,64,128,192,256, MX-fp4, `--mem auto`. Expected memory, inferred from the voxel net's recorded 61.8 B per window voxel at
P576 (the latent-grid section of 2026-11-09): about 8.3 GB of activations at P 512, 11.8 GB at P 576 on a 16 GB card, with
2 head channels instead of 11; not measured for this task - the first launch should be `UFSM_MEM_DRY=1` and a 200-step smoke run.
Nothing here has run on a GPU.

**3b, design only: the normal head should be orthogonal to fibres.** With N = the predicted sign-free normal dyad (6 channels at
`g_norm_ch`, `sigmoid` outputs coding n n^T as nn_unwrap_targets2 writes it) and t a fibre tangent, the term is L = lambda t^T N t at fibre
voxels (0 when the fibre lies in the predicted sheet, 1 when it is along the normal). It needs no winding label, so it trains the normal
head where we have fibres and no spiral. Sites: (1) a label plane: the sampler's batch gets `fib` = B x P^3 x 2 bytes (the tangent on the
z >= 0 hemisphere, the published nx / ny encoding, 128,128 with a zero third byte = none), rasterised like `fiber-raster` from a sources
entry `"fibers": {root, group}` (two more uint8 volumes, a `fiber-raster` option still to be written), rotated by the spatial augmentations as a
direction (the normal targets avoid this by being derived from the warped phase; the tangent plane cannot, so `spatial_augment.c` has to
apply its local rotation to it - the one new piece of sampler work); (2) `head_slab_pass` (`src/train.c:846`):
after `nn_unwrap_targets2_z` (line 952) builds `d->nt / d->nm`, a kernel `nn_fib_ortho(LGC(g_norm_ch), fib + vb, ns, lambda, grad)` adds
dL/dN_k = lambda c_k t_i t_j into the six normal gradient channels after `nn_loss_grad_only` (line 999) has written them, and its sum into
`d->hs_ds` for the log; in pass 0 only the statistic. The whole-window path (line 1306..1315) gets the same call; the fused latent path
(`nn_lat_fuse_grad`, line 1021) does not and the flag must refuse `--lat` until it does. (3) flag `--fib-lambda` (default 0: checkpoints
and old runs unchanged; no new parameters). Test: central differences of the kernel against a host reference, as for the other loss
kernels. Buffers: the plane is uint8 x 2 per voxel on the host, uploaded per 32-plane slab with the other label planes; no fp4 question
arises (it is a label, not an activation). Expected size of the effect is unknown; the tangent-vs-normal medians above (7 .. 9 degrees
against GT normals) say the fibres are consistent with the sheets to that accuracy, which is the ceiling of what this term can teach.

**4, design only: a soft same-fibre pair term for the phase and winding-mod-4 heads.** A fibre lies in one sheet, so two points of one
fibre <= 0.3 mm apart should get the same winding: for the phase head (unit vector u = (cos, sin) of the two phase channels at `g_side_ch`
in task unwrap2) and for the w4 head (`g_w4_ch`, the same form with the angle 2 pi w / 4) the term is L = lambda (1 - u_a . u_b) over
pairs (a, b), capped per pair (the measured 5 .. 7 % of pairs beyond half a turn at 0.3 mm are outliers by construction and must not pull).
It uses no winding label, so it is the one winding signal available on scrolls without a trusted spiral; alone it has the constant
solution, so it only ever runs beside the supervised phase loss, lambda small. What the loss path dictates: logits exist one 32-plane
z-slab at a time (`head_slab_pass`), so both points of a pair must lie in the same slab - 32 planes at 9.6 um is 0.31 mm, the same range
the measurements support, and H fibres run inside a slab anyway; and the loss gradients are half floats without an atomic add, so the
sampler emits DISJOINT pairs (a voxel in at most one pair), which makes the scatter two plain writes per pair. Sampler: per source a
prebuilt pair-point file (UFIB polylines + a 64^3 cell index, built once by a `fiber-raster`-like pass from the AFV or from traced
predictions), per window up to 4096 pairs = 32 KB of int32 voxel indices, uploaded with the label planes of the slab they fall in. The
points are moved by the window's own spatial transform; flips, the 5 degree rotations and the axis jitter are affine and cost a 3 x 4
product per point (microseconds per window, nothing next to the CT decode); the elastic warp is stored as output -> input and has no
cheap forward map, so windows with it (p 0.15) carry no pairs. Kernel `nn_pair_cos(LGC(ch), pairs, n, lambda, cap, grad, stats)` after
the phase / w4 gradients are written (the block around `src/train.c:976..1021`), statistic = mean 1 - cos into `d->hs_ds`; the fused
latent path refuses the flag as for 3b; `--fibpair-lambda 0` default, no parameters, old checkpoints unchanged. Test: central differences
against a host reference. The measured ceiling: 0.93 .. 0.95 of such pairs agree in the GT's own phase, so the term can remove gross
sheet switches along fibres, not refine a good phase.

**Centring, snapped wraps, and masks without directions (later the same day).** The judge is already free of centring: every statistic
is a difference of the unwrapped phase along one fibre (pairs: phase at s + d minus phase at s; excursions: against the stretch's own
start), so a wrap that sits a constant few voxels off its sheet changes nothing; only an offset that varies by half a pitch along a
fibre would. Measured: hostmem's snapped 1299 wraps (`latcal/wraps_snap/PHerc1299`, mean |shift| 4.15 um) give H 0.5957 / V 0.6722 at
0.6 mm and 751 / 509 excursions per m, against 0.5961 / 0.6738 and 748 / 506 unsnapped. So the 1299 row is not centring. Per tile
(19 tiles with >= 0.3 m) the persistent rate is 210 .. 1278 per m, median 374, against Paris 4's 67 .. 576, median 189: the excess is
spread over the whole section, with one tile (y 3584, x 2048) far worse.
`fiber-trace --presence-only 1 / --mask ROOT KEY` takes the axis from the presence itself (smallest eigenvector of its gradient structure
tensor over 5^3 voxels), so lines can be traced through a plain mask - our own H / V predictions. On the official Paris 4 presence at
38.4 um, where a fibre is about one voxel wide, it is clearly weaker than the direction channels: 25,864 lines of mean length 31 voxels
against 86,597 of 74, pairs at 0.6 mm 0.861 H / 0.874 V against 0.925 / 0.908. `scripts/fibers/eval_hold.sh RUN` is the validation of
the H / V model built on it: predict both channels on the Paris 4 HOLD box (another scanner setting, never trained on), trace each mask,
judge against the GT phase. Reference rows on HOLD (6 tiles of 256^2 x 512, pairs at 0.6 mm H / V): published v2 fibres 0.952 / 0.902
(3.5 m), official prediction with directions 0.966 / 0.876 (2.3 m), presence only 0.888 / 0.758 (0.24 m: too little to rank on, so
the script also prints the traced length). The predict flags in it are untested until a GPU is free.

**Pictures and launch checks (evening).** `tools/fibview.py` / `tools/view1299.sh` -> `/vesuvius/ufsm/fibers/work/view1299/` (20 PNG +
index.txt): native-resolution CT crops (768 px) with the wraps, the traced fibres and the judge's sheet changes, for the worst / median /
best 1299 tile and two Paris 4 tiles. In the worst 1299 tile the wraps cut across the visible sheets at an angle while the fibres follow
the sheets; in the Paris 4 tiles the wraps lie on the sheets. Labels: 0175A 17 GB, 1447 21 GB. `mksources.sh`: weights 0.446 / 0.554.
Dry planner (GPU 1, planning for 15 GB free): P 512 -> 5.67 GB (network 4.43, 33.0 B per input voxel; MX gradients, chunked, lean 2),
P 640 -> 10.44 GB with 7.81 GB of host offloads each way per step; P 768: no plan. Sampler (`ufsm sample --bench 10`, 5 workers, no
CUDA): 3.62 windows / s at P 512 (486 Mvox / s, 1.4 cpu s per window); 16 windows: known voxels 28 % of the mid slice (8.7 .. 51 %), H
and V positives 1.05 % each, one window without positives, none without labels.

**PHerc1218 (no fibres published): two routes.** (a) One-off inference with the public model, outside the pipeline: clone villa, its
bootstrap venv (PyTorch cu128), `python -m vesuvius.neural_tracing.fiber_trace_3d.infer` with the SD2 checkpoint on level 0 of the 8.64 um
scan (the card's own recipe for a 9.36 um scan, "transfer use"). For the judge one 640-slice slab is enough: 7593^2 x 640 voxels = 1,152
tiles of 320^3 net (512 with overlap 96); the whole scroll is 42,048. Needs a free GPU; I have no throughput figure for that model and did
not run it. (b) Our own H / V predictor from the labels above, then `ufsm predict` on 1218: 1447, 0175A and 1218 are the same acquisition
(8.64 um, 1.2 m, 116 keV), so no transfer gap, but it needs the training run first and gives masks, not directions (streamlines would
need the direction planes of 3b). (a) is the quicker way to a 1218 row; (b) is the one that stays in the pipeline.

## Published automated fibre volumes: what they are, how good, fibres as a judge of our spirals, plan (2026-11-10, branch fibers-20261110)

Source: `https://dl.ash2txt.org/community-uploads/qual/AutomatedFiberVolumes/` (community upload, 11 `.afv` files + one PNG set). Local copy
(6 files, 11.4 GB, sha256 verified for the three with published sums): `/vesuvius/ufsm/fibers/raw`; scratch results `/vesuvius/ufsm/fibers/work`;
analysis scripts `tools/*.py` (Python, analysis only, run with `python3 -I`; nothing here is in the pipeline). The repo already had an AFV
reader and fibre terms from 10-26 / 11-05 (`src/afv.c`, `ufsm fiber-check`, `fiber-export`, `fit --fibers --w-fib --uw-rep-fib`; commits
6a0e2d1, d397bf4: fibre votes in the slip repair ids +0.021 [+0.006, +0.040] on 15 Paris 4 boxes) that never got a section; this one covers the data.

**Format.** SQLite 3 (application id "VCFC"): `metadata(key, value)`; `fibers(id, name, family 'H' | 'V' | 'unknown', point_count, length, bbox,
annotation JSON)`; `blocks(fiber_id, first_segment, points)` = float64 LE XYZ, 256 segments per block, neighbouring blocks share a point;
`block_bounds` = an rtree over the blocks. Coordinates are level-0 voxels of the scan named in `metadata.root`, integer = voxel centre
(so level L: `(v + 0.5) / 2^L - 0.5`). Annotations: the H / V family (key "manual_tag", but `classification_source` =
"model_prediction_family_locked_tracing"), length, two control points; 0125 adds per-join evidence (`joins`: overlap seams with scores);
the v1 5 mm files add `estimated_widths` (piecewise width in voxels). No confidence anywhere.

| file | scan it is native to | fibres | metres | H / V | length median / p95 / max mm | vertex spacing |
|---|---|---|---|---|---|---|
| Paris4_1mm_v2 | PHercParis4/20230205180739, 7.91 um (the 2023 scan) | 1,038,191 | 3528 | 495,817 / 542,374 | 1.97 / 10.4 / 257 | 65 um |
| Paris4_1mm_v2_sample | same, z 5129..9128 | 364,708 | 1174 | 166,275 / 198,433 | 1.91 / 9.9 / 194 | 64 um |
| Paris4_v1 | PHercParis4/20260411134726, 2.4 um (ours) | 512,698 | 1726 | none | 2.57 / 8.5 / 44.7 | 363 um |
| PHerc1447_1mm_v2 | PHerc1447/20250521151220, 8.64 um (ours) | 1,228,256 | 5065 | 602,140 / 626,116 | 2.12 / 14.0 / 155 | 79 um |
| PHerc0175A_1mm_v2 | PHerc0175A/20250521115057, 8.64 um (ours), z 1490..11485 | 934,424 | 3915 | 443,739 / 490,685 | 2.15 / 14.1 / 314 | 71 um |
| PHerc0125_3mm_v2 | PHerc0125/20250821151825, 9.362 um (ours), z 9425..11985 only | 42,299 | 197 | 21,692 / 20,607 | 4.12 / 8.0 / 30.2 | 4 um |

Not downloaded (heads read by HTTP range): `Paris4_all_v2` (6.7 GB) / `PHerc1447_all_v2` (8.2 GB) = the same tracings without the length
filter (Paris 4: 6,486,903 fibres before the > 1 mm cut); `0175A_SMALLMORE` (7.2 GB, 5,797,793 fibres, all lengths); `1447_5mm_v1` (80,762
fibres) and `0175A_5mm_v1` (61,300), an older tracing cut at 5 mm, with widths. `1447_5mm_v1_png/`: per fibre of `1447_5mm_v1` a straightened
129-row "face" image (row 64 = centreline, the across direction from the published Lasagna frames), the exact sampling maps (6 float64 per
column) and automatic width pseudo-labels: 80,762 fibres, 70.1 M columns = 606 m, 333,473 measured width segments, width median 24.5 px =
212 um (p5 75, p25 114, p75 254, p95 300 um), H 39,248 / V 41,514. 8.6 GB of PNG, 3.4 GB of maps; we fetched the index, one map and one PNG.

**Where they come from** (the files' provenance metadata). The v2 files of Paris 4, 1447 and 0175A are one unpublished model of the uploader
("ARGON", checkpoint step 29220, one sha for all three) traced on the CT downsampled 2x (15.8 / 17.3 um), "family locked" (an H trace stays H),
cleaned near the scroll exterior, cut at 1 mm, simplified (Douglas-Peucker 0.25 voxel). 0125 is a different tracer ("NEWALGO": fragments joined
at overlap seams). No model, tracer or code is in the upload. `Paris4_v1` is something else: streamlines integrated at 38.4 um through the
OFFICIAL fibre prediction `PHercParis4/representations/predictions/fibers/20260411134726-fibers-20260915212757-L1/` of the open-data bucket
("estimated overlapping streamlines, not unique physical fibres", its own words). That official product exists for PHercParis4, PHerc1299,
PHerc0139, PHerc0332 and PHerc1451 (presence + nx + ny, uint8 OME-zarr v2 at levels 3 and 4 of a 2.4 um scan = 19.2 / 38.4 um; model
"fiber3d/unet", 128^3 patches, 7 output channels, stride 4, trained on human fibre line annotations of 5 scrolls), and its weights are public,
ungated: Hugging Face `scrollprize/lasagna-fiber` (220 MB checkpoint), beside `fiber_hz_vt`, `fiber_tracer_large`, `fiber_single_small` and
others. Our PHerc1299 CT is the scan that prediction is on.

**Geometry against our CT.** (a) Exact: one 1447 face PNG (129 x 618) re-sampled from our local volcomp level 0 (trilinear, integer = voxel
centre): all 79,722 pixels within 1 grey level (they truncate; correlation 0.99995); a +-0.5 voxel shift gives a mean error of 7.1. The
uploader read our own volcomp upload (source path in the AFV). (b) Ridge check, mean CT along the centrelines vs on a ring of radius r voxels
about them, 384^3 boxes (`tools/fibgeo.py`): 0125 131.7 at the centre, 115.7 at r = 4 (box sd 28.7); 1447 126.8 / 110.4; 0175A 127.2 / 99.4;
a 3-D translation search peaks at (0, 0, 0) for 0125 and 0175A and within 1 voxel for 1447 (flat top). Paris 4 v2 must go through the
published affine `transforms/20260411134726-to-20230205180739-r1` (inverted; 24 landmarks): centre 114.0, r = 6 94.9 on our level 2; the
residual, from a CT-only search per 256^3 block (`fiber-check --align`), is within 1 level-2 voxel in HOLD, up to 2.3 in PH2, 2-3 in PH1,
3-5 in PH3: 10-30 um typically, 50 um at worst, against sheets ~125 um apart. Paris 4 v1 does not sit on fibres (102.5 vs 101.0 at r = 8,
best shift 6 voxels away). H fibres are circumferential (|t_z| < 0.3 on 97 %), V axial (|t_z| > 0.7 on 97 %).

**Coverage** (`tools/fibcover.py`, one 384^3 box per scroll, papyrus = above the box's Otsu threshold): Paris 4 v2 20 fibres and 0.028 m per
mm^3 of papyrus, 1447 27 / 0.029, 0175A 29 / 0.043, 0125 (>= 3 mm only) 7 / 0.016. Share of papyrus voxels within 1 / 2 / 4 / 8 voxels of a
centreline: Paris 4 0.013 / 0.035 / 0.117 / 0.361, 1447 0.012 / 0.032 / 0.114 / 0.375, 0175A 0.016 / 0.042 / 0.132 / 0.394, 0125 0.006 /
0.015 / 0.049 / 0.165. So the > 1 mm sets label about an eighth of the papyrus at a 4-voxel tube and are far from exhaustive (the length cut
alone drops 84 % of the Paris 4 traces by count): papyrus away from a tube is unknown, not fibre-free. 93-99 % of centreline points are on papyrus.

**Does a fibre stay in one sheet? Paris 4 against the human GT** (`tools/fibgt.py`, `fibjudge.py`: the GT band phase unwrapped along each
fibre; a labelled stretch ends at an unknown voxel). Two full cross-section slabs (z 8832 and 11328 of level 2, 179 tiles of 512^3, 279.5 m of
labelled fibre arc). Same-fibre pairs within half a turn of each other:

| arc distance | 0.15 mm | 0.31 mm | 0.61 mm | 1.2 mm | 2.5 mm |
|---|---|---|---|---|---|
| pairs agreeing / pairs | 5,546,475 / 5,676,882 | 2,010,111 / 2,163,740 | 596,360 / 659,344 | 123,984 / 137,969 | 15,126 / 16,886 |
| share | 0.977 | 0.929 | 0.905 | 0.899 | 0.896 |

H 0.913 (289,011 / 316,537), V 0.897 (307,349 / 342,807) at 0.61 mm; half-turn excursions 187 / m (H), 254 / m (V). The share is flat from
0.6 to 2.5 mm: not a random walk of switches but a fixed tenth of the arc at odds with the label, and it follows the tile: in the 4 tiles
where our spiral and the GT agree completely (score >= 0.999) the share against the GT is 0.983-0.995 at 0.31 mm and fibres cross our
labels 14-65 times per metre; the gt-gate mask
(`rv/gt/p4gate`) lifts a bad tile (49 % of its GT gated) from 0.85 to 0.89 (H) and 0.76 to 0.80 (V), so part of it is the GT. On the
labelled surfaces only (`tools/fibcodes.py`: fibre samples within 2 voxels of a GT code voxel, no interpolated phase), H fibres, the 10
tiles where our spiral matches the GT: 0.61 mm 38,597 / 40,151 = 0.961, 1.2 mm 0.951, 2.5 mm 13,247 / 14,157 = 0.936, 4.9 mm 1,983 / 2,091 =
0.948. The slope from 0.61 to 2.5 mm puts the persistent switch rate of H fibres at <= 14 per metre (one per 7 cm); the 4 % floor does not
grow with distance (label and registration noise). V fibres cannot be tested this way: they lie on the far side of the sheet from the labelled
recto face (only 27 % of V samples are within 2 voxels of a code, and the nearest code is as often the next sheet's). By fibre length (HOLD,
4.2 m): excursions per metre H 87 / 123 / 88 / 80 / 83 and V 218 / 197 / 217 / 240 / 285 for 0-2 / 2-4 / 4-8 / 8-16 / > 16 mm fibres: no
length class is cleaner. The fibre tangent leaves the GT sheet plane by 6.2 deg (H) / 8.4 deg (V) at the median, > 30 deg on 1.5 % / 2.4 %.

**H and V layers = recto / verso?** On Paris 4 (HOLD, registration within a voxel) H fibres sit at GT phase 0.11 (circular concentration
0.39), V at 0.40 (0.22): V outward of H, the recto face towards the umbilicus, but the two phase histograms overlap (best threshold 66 %).
On the natively registered scrolls (`tools/fibhv.py`: nearest V point within 10 voxels of an H point, radial offset about the axis) the
offset is bimodal at about +-5 voxels (45 um) with a dip at zero on 0175A and 1447, V outward in 0.64 / 0.57 of pairs (the other mode is the
inner neighbour sheet's V layer), flat on 0125 and blurred on Paris 4 (0.56). So the layers are resolved at 8.64 um; they become a recto /
verso label only together with the sheet (which mode is the same sheet), i.e. as a constraint on our side labels, not as a label of their own.

**Fibres as a label-free judge of our wraps.** A fibre lies in one sheet, so its samples should keep one winding. Three ways to get our
winding along a fibre, all reduced to the same statistic (band phase unwrapped along the fibre): (i) a `surf-labels` phase raster
(`tools/fibjudge.py --ours`; hostmem's `/mnt/scratch/lat/cal/p4june` has one for Paris 4, with its ignore gates applied); (ii) the joined fit
set, `ufsm fit-phase-box SET z y x nz ny nx out.raw` (new: frac(W) of the set over a box, CPU); (iii) the final snapped wraps,
`ufsm wrap-dump WRAPS out.f32 --z a,b` (new: the wrap vertices of a slab) + `tools/fibwrap.py` (phase between the two nearest wrap sheets
from two KD-trees, even and odd wrap ids). (ii) and (iii) agree per tile on 0175A (64 tiles: Spearman 0.964, mean difference 0.011): the
snap does not change the verdict, and (ii) runs on any scroll with a joined set. One slab of 512 slices per scroll, basis (iii), every wrap
and every voxel (pairs within half a turn / pairs; excursions per metre):

| scroll (slab) | fibre arc | H 0.3 mm | H 0.6 mm | H 1.2 mm | H 2.4 mm | V 0.6 mm | excursions H / V per m | tile median H (p10..p90) |
|---|---|---|---|---|---|---|---|---|
| Paris 4 (z 11328 L2) | 168 m | 0.849 | 292,237 / 384,143 = 0.761 | 0.679 | 0.588 | 0.709 | 400 / 497 | 0.764 (0.629..0.883) |
| 1447 (z 12160) | 112 m | 0.883 | 237,281 / 310,385 = 0.764 | 0.622 | 0.464 | 0.735 | 407 / 477 | 0.763 (0.561..0.878) |
| 0175A (z 6400) | 134 m | 0.853 | 276,767 / 371,207 = 0.746 | 0.631 | 0.484 | 0.689 | 473 / 528 | 0.763 (0.606..0.875) |
| 0125 (z 10624; other tracer, >= 3 mm) | 35 m | 0.725 | 53,719 / 91,387 = 0.588 | 0.441 | 0.291 | 0.570 | 845 / 875 | 0.633 (0.271..0.724) |

Against the wraps the share falls with distance (true crossings), unlike against the GT. Paris 4 on basis (i), i.e. only where surf-labels
keeps the label (0.47 of the known voxels; the CT coherence gate drops most): 111,780 / 126,545 = 0.883 at 0.61 mm, flat, 206 excursions per
metre, against the GT's 0.905 / 220 on the same tiles: the existing gate already removes the places where fibres cross our wraps.

Validity on Paris 4, per 512^3 tile, against a GT score of our labels (`fiblib.offset_clusters`: at GT code voxels the offset between our
winding and the GT's; the share of the largest integer cluster, an ids-like number; median 0.50, quartiles 0.38 / 0.66; hostmem's log of the
same labels: pair winding right 0.72 at 0.5 mm, 0.28 at 4 mm): fibre excursions per metre vs that score, Spearman -0.61 (basis iii, 74
tiles, p 8e-9), -0.53 (basis i, 150 tiles in two slabs, p 3e-12; -0.51 and -0.54 per slab); 10-26 had -0.62 on 11 boxes against bench ids.
Two cautions. The fibres' disagreement with the GT in a tile predicts their disagreement with our wraps in that tile (Spearman 0.93), so
most of the tile ranking is how hard the tile is for everything; with the GT excursion rate partialled out the judge still correlates
-0.52 with the score. And a fibre only sees a wrap that cuts across sheets: a skipped or doubled sheet is invisible to it, which is why a
spiral whose pair accuracy is 0.72 / 0.28 can agree with fibres almost as well as the GT does. It is a judge of "follows the sheets",
complementary to count statistics, not a replacement.

Extending it past the four scrolls needs fibres elsewhere, and the published official prediction is enough for that. `Paris4_v1`
(streamlines through it at 38.4 um, off the fibre ridges) agrees with the GT better than v2 on the same slab (z 11328: 111,701 / 119,262 =
0.937 at 0.61 mm and 125 excursions per metre; v2 there 229,494 / 257,926 = 0.890 and 264: smooth streamlines wander less than ridge
traces), and as a judge of our labels its excursion rate correlates -0.40 with the GT score over 64 tiles (v2 on that slab -0.51), from
3.3x less arc (20.2 vs 67.5 m). So PHerc1299 can be judged from the published prediction on our own scan with a streamline integrator
(`fiber-trace`, to write: seeds at presence maxima, midpoint integration of the direction dyad; the uploader's thresholds are in the v1
metadata); PHerc1451 and PHerc0139 have predictions on other scans of theirs (the transforms exist); PHerc1218 and the rest need the
published net run (weights public) or our own predictor.

**Plan.** (a) Our own fibre predictor. Labels: `ufsm fiber-raster AFV OUT --like ROOT KEY [--tr T] [--radius-um 40] [--all AFV2]` (to write):
per family a prob254 volcomp pyramid on the CT grid, 254 inside the tube, 255 (unknown) in a shell to twice the radius and wherever the
all-lengths AFV has a trace the > 1 mm set lacks, 0 only in air and in papyrus far from every trace; plus the unoriented direction as the
dyad t t^T in the encoding of our normal planes (t0..t5). Widths: the 1 mm files have none; the v1 5 mm files and the PNG index give
per-segment widths for 80,762 (1447) and 61,300 (0175A) fibres, enough to set the tube radius by family and to check it, not a per-fibre
label for the v2 sets. Training needs no new code for the masks: two sources per scroll (H, V) with `"targets": {"recto": {"encoding":
"prob254"}}` and `"head": 0 | 1`, `ufsm train --heads 2`; the direction dyad is a second step (the unwrap task's normal channels). Volume:
1447 5065 m, 0175A 3915 m natively registered (a 40 um tube is ~5 mm^3 per metre: ~25,000 and ~20,000 mm^3 of positives), Paris 4 3528 m
with 10-30 um registration error (use it at 19.2 um or keep it for validation), 0125 197 m of another tracer (validation). Split: by scroll
plus z blocks with a 2 mm gap inside the training scrolls (H fibres stay in a z range, V fibres are cut by the gap; "no chain across the
split" holds because labels are voxels and the gap is unlabelled). Score: Dice on the tube with the unknown shell, and the traced
prediction's same-sheet rate on Paris 4 HOLD against GT codes (the 0.96 at 0.6 mm above is the reference). Cheaper first step than training:
port the published `scrollprize/lasagna-fiber` net (`fiber3d-ext`, like `ink3d-ext` / `surfrecto-predict`) or just read its published
prediction volumes; they are 19.2 um presence + direction, enough for a judge and an auxiliary target, not for 9 um centrelines.
(b) Fibres in the latent winding net, cheapest first: (1) no training change: the judge as a label gate (tiles or cells where fibres cross
our wraps become phase-ignore) on the scrolls that have fibres, and as the acceptance test of every new spiral there; (2) a local loss on
the normal head, t^T N t -> 0 at fibre voxels (the differential form of "phase constant along a fibre"; one label plane, one term in the
head slab pass, no pair sampling), which also supervises where the spiral label is ignored (53 % of Paris 4's known voxels); (3) the pair
form on the w4 / phase heads (two points of one fibre in a window: equal w4 class, equal phase) needs pair lists per window from the
sampler; (4) fibre mask as an auxiliary head or as `--aux-input`: only after (a) exists for all 33 scrolls, since an input channel must
exist at inference. Measure on wbench against wind96 (p05x, ids; `BOXES=core`): Paris 4 boxes are the only bench boxes with AFV fibres, so
(2) and (3) are first tried with Paris 4 fibres at 19.2 um cells where the 10-30 um registration error is under a fifth of a cell.

## The 10-26 / 11-05 fibre work: fiber-check, fiber-export, fibre terms in the fit (recovered 2026-11-10 from commit messages)

This section was not written at the time. Everything in it is copied from the messages of commits 6a0e2d1 (2026-10-26) and d397bf4
(2026-11-05); the numbers are as recorded there and were not re-measured.

- 6a0e2d1: `ufsm fiber-check` (published AFV fibre polylines against GT sheets: same-fibre pairs, tangent / straight / rand3d baselines,
  a path-unwrapped network-phase filter, H / V layer order, a CT translation search, label facing); `ufsm fiber-export` (box-local
  polylines); `fit --fibers F --w-fib W` (pair loss W(b) - W(a) = the phase change along the fibre, in every fit stage); `src/afv.c` reads
  the AFV SQLite through a dlopen'ed libsqlite3; `test_afv`.
- d397bf4: `--uw-rep-fib W` (with `--fibers`, `--uw-repair`): same-fibre pairs vote integer shifts between repair pieces in the L1 solve
  (`src/fit.c` uwr_fib). Paris 4, 15 boxes with AFV fibres: W 16 ids +0.021 [+0.006, +0.040], P 0.999, p05x flat (J34 +0.14, J22 / J23 / J26
  +0.05, HOLD -0.03); W >= 64 overrides the phase (J34 -0.14 at 256).
- d397bf4: fibre loss terms `--fib-spans`, `--w-fib-cons` / `--fib-cons-len` (per-fibre consensus, winding-field style): at W 0.1 .. 0.3 they
  move ids < 0.001, at 1 they hurt.
- d397bf4: fibre trace knobs `--fib-conf` / `--fib-gap` / `--fib-dmax` and the hop gate `--uw-rep-fib-hop` / `--uw-rep-fib-min`, all off by
  default: the stricter trace drops the fibres that fix slips, the hop gate flags ~0 fibres.
- d397bf4: `fiber-check --fit-pat`: a GT-free fit score (fibre drift / separation); drift against ids Spearman -0.62 over 11 J boxes.
- Reading it with today's numbers: the vote use gains because a vote needs only the majority of fibres to be right over a short span;
  the loss terms ask every fibre to hold its sheet, and one persistent change per 7 .. 10 mm (section above) is why they do not pay.

## Fibres in the winding network: the normal head orthogonal to fibres (2026-11-11, branch fibnet-20261111)

`train --fib-normal-lambda L` adds L x the mean of t^T N t over a window's fibre voxels: N the sign-free normal dyad of the six normal
logits exactly as `nn_unwrap_targets2` codes it (p = sigmoid: N_zz, N_yy, N_xx = p0..p2, the off-diagonals p3..p5 - 1/2), t the fibre
tangent. A fibre lies in its sheet, so (n . t)^2 = 0 is a target that needs no winding label (`src/nn_loss.cu` nn_fibn / nn_fibn_apply).

Where the tangent comes from (a deviation from the 11-10 design, which rotated a stored tangent plane through the spatial augmentation):
the fibre input is a prob254 tube raster (`ufsm fiber-raster`, radius 2 voxels; 255 unknown) given as a source's "sheet" / "ink" target,
and it rides the existing ink plane (`IPLANE`): every symmetry, warp, replay and `--gpu-aug` path the ink plane already has applies, and
nothing new happens in the sampler. On the device the tangent is read off the transformed tube: the gradient structure tensor S of the
plane over 5^3 voxels is c (I - t t^T) for a tube, so T = I - 2 S / tr S ~ t t^T and t is its leading eigenvector (three power steps).
Voxels whose T is not near rank one (leading eigenvalue < 0.75, or the other two with squares summing over 0.5: blobs, sheets, crossings,
tube ends) are skipped. A direction is never transformed, so the 48 symmetries and the warps cannot get it wrong; the cost is that t is
known only where the raster is a clean tube, and only in the tube's 5-voxel core (the tube raster has no sub-voxel position).

Loss path: head slabs only (`head_slab_pass`). The statistics pass counts the window's valid voxels and sums t^T N t (the logged `fibn`);
the gradient pass collects each slab's per-voxel d(t^T N t) / dlogit = c_k p_k (1 - p_k) (c = t_z^2, t_y^2, t_x^2, 2 t_z t_y, 2 t_z t_x,
2 t_y t_x) into a list before the in-place normal BCE gradient overwrites the logits, and adds L / count x the list after the
--normal-lambda scaling. Only voxels whose stencil (3 planes) lies inside their own slab count: a streamed slab holds only its planes
(12.5 % of fibre voxels lost at 32-plane slabs). The fused latent gradient (`g_lat_fuse`) is switched off with the flag (the unfused
latent chain works); the whole-window loss path, `--ink-head` and `--rv` (both own the ink plane) are refused; task unwrap needs
`--ink-lambda 0` (its CH_SHEET channel is the plane now). No parameters, default 0: checkpoints and old runs unchanged.

Test `test_fibn`: a radius-2 tube along the 48 symmetric images of (1, 2, 3) / sqrt 14 (the device sees only the transformed tube): mean
|t . d| 0.9978, worst 0.9929 over 1220 axis voxels; kernel count / mean t^T N t equal to a host reference; the gradient against central
differences of the host loss (worst 8e-9 at a largest gradient 4e-4); two 16-plane slabs with the planes outside each slab poisoned
count exactly the host's per-slab voxels. First real steps (wind96 from vc384.init.ckpt, P 384, Paris 4 / 0139 / 0500P2 winding GT +
1447 / 0175A H-fibre sources at 25 % of the weight, lambda 0.1): fibn 0.06 .. 0.10 (wind96's normals already make (n . t)^2 small:
sin^2 of 14 .. 18 degrees), 260 000 .. 315 000 valid voxels per fibre window, but only 5 of 60 windows came from the fibre sources (8 %
against their 25 % weight: the sampler rejects more of their draws; not investigated). Planner: 6.43 GB at P 384 (network 4.57). With the
Paris 4 raster on the two Paris 4 sources (vcfibA's list) 24 % of windows carry fibre voxels (48 of 135 sampled windows had known fibre
labels; z band 6656..11776 only), fibn 0.051 at step 50; Paris 4 raster: 1.05 G H-tube voxels, 8.6 GB.

Arms (`/vesuvius/ufsm/v2/lat/runs/vcfib.launch.sh`, not started): vcfibA = vc384c exactly (wind96, lr 1e-4 / muon 0.002, task unwrap,
P 384, 4000 steps) with lambda 0.1 and the Paris 4 H-fibre raster as the "sheet" target of the two Paris 4 sources (the same windows
as vc384c; only the loss differs); vcfibB adds the 1447 / 0175A fibre sources. Paris 4 raster: `fiber-raster Paris4_1mm_v2.afv --tr
p4_2026_to_2023.json --level 2 --zband 6656,11776` into /vesuvius/ufsm/fibers/labels/PHercParis4-l2.

**Same-fibre pair term (design only).** The 11-10 design needs sampler pair lists moved by the window's transform. With the tangent on
the device it does not: pair a voxel v of a clean tube with v + round(d t) (d <= 0.3 mm, 32 planes at 9.6 um) when that voxel is also a
clean tube voxel with |t_a . t_b| > 0.9; the pair term lambda (1 - u_a . u_b) on the phase (and w4) unit vectors, capped per pair, then
needs no sampler change either. Collisions in the gradient scatter (two pairs sharing a voxel) are avoided by letting only voxels with
even (z + y + x) / d parity start a pair, or by the same list-then-add scheme as nn_fibn. Both points must lie in the slab, which a 32-plane
slab allows for H fibres (they run in the sheet, mostly across z only at small slopes).

## Inference from the stored latent volume, without the CT (2026-11-10, branch latinf-20261109)

`predict --latents STORE` (an `ufsm lc-encode` store of the CT level) reads no CT voxel; the CT store is opened for its metadata only
(3 metadata reads per run). The reader thread reads each window's Wz/4 x W/4 x W/4 cells of C + 1 bytes (`zr_read_i8_mt`: the window's
chunks decoded by `--threads` store handles; one thread made a 1.36 Gvox p1667 row 0.29 Gvox/s, 10-16 threads 1.4), and all-air windows are
dropped before upload. On the device `nn_lat_synth_ct` rebuilds the window's CT (inverse transform; 1..255 in material cells, 0 in air
cells), so the statistics (the z-score), the "0 where CT 0" masks (cell granularity in air) and the guided heads' CT terms run unchanged on it;
the network input comes from the stored coefficients directly (`nn_lat_pred_input_q`: no second quantiser; radial channels from the axis
offsets). Occupancy: the store's shard indices (absent or all-zero chunks; a 32-voxel grid). Windows must lie on the store's cell grid (box,
grid origin, halo and stride multiples of 4 from its origin; refused otherwise). Tests: test_lat (rebuilt CT = host inverse transform within
1 code; input within half a stored step of the CT path's unquantised channels, radial channels bit-equal), tests/test_latents.sh.

Gate, p1667 recto-ruler (scripts/rv/eval.sh, `LATENTS=/mnt/scratch/lat/p1667-l2.lc`), g768 snap 10000 (guided head, fp8), `--window 384`,
24 shard rows of 128 (32.7 Gvox), turns per GT turn / share off >= 0.5 turn, all radii:

| source | rv.pkw | rv.pk | rv.run |
|---|---|---|---|
| CT (level 2) | 0.852 / 0.494 | 0.799 / 0.447 | 1.136 / 0.429 |
| latents (KLT29, step 0.25) | 0.830 / 0.496 | 0.805 / 0.434 | 1.109 / 0.424 |

Equal within noise (rv.pk and rv.run slightly better from the store, rv.pkw slightly worse). Rate on GPU 1 beside the wz768 trainer, warm
page cache, the same three rows alternating: CT 0.49 / 0.38 / 0.35, latents 0.58 / 0.47 / 0.37 Gvox/s (wall per predict process). Bytes read:
CT store 0.82 GB per box Gvox (26.8 GB), latent store 0.97 GB per box Gvox (31.8 GB), both including the windows' halos and the 3x z overread
of 128-row boxes with 384-voxel windows; in material regions the latent store (0.13 B/vox on disk for the fixture) is not smaller than the
volcomp CT, so the store is not an IO saving at this level: the gain is the cheaper window input.

Fit fields (2026-11-10): `--fit-zarr` / `--fit-ct` from the store read the rebuilt CT. tests/test_latents.sh bounds the CT-vs-store
correlation per group of planes, each plane standardised and the group pooled: single components of 6-step nets swing between training
seeds (t0 0.64-0.95, t2 0.62-0.95 over seeds 5-10 on latcomp 6dea200; one failed a 0.7 bound after an unrelated planner change), the
groups do not (min / max over the 6 seeds, guided head 2 | sub-cell head 3: phase .93/.96 | .79/.83, normal .82/.88 | .85/.89, spacing
.78/.86 | .72/.86, material .944, winding mod 4 .78/.87 | .83/.85, its directions .58/.62 | .58/.68, --fit-ct ct .9995, cs .81/.96);
bounds are those minima less ~0.1. With 30 steps the same seed reaches phase .96 / .91, normal .92 / .96, directions .78 / .75: the
disagreement is the young nets' sensitivity, not the store. lc-encode's sd_ref now comes from mostly-material shards (the masked Paris 4
level 2 had starved the old stride sample: sd 1, a 125 GB store at PSNR 25 dB); a box without material is refused.

## Label planes kept on the device where the plan has room; what --det 1 fixes and what it does not (2026-11-10, branch perf-20261109)

**Labels.** The latent nets' label stream builds each plane on the device, copies it back to pinned host memory and uploads it again per
head slab in both head passes (rv P1024: 178 ms of copy-back per batch, 380 ms of slab uploads inside the step). Now the planes the
memory plan has room for stay on the device (`d->ldev`), the rest stream as before:

- Order: the `--rv` papyrus plane as 2-bit classes (unknown / papyrus / void: all `rv_tg_k` reads; an `--ink-head` plane stays bytes),
  then phase, winding, component as bytes. The spare is the plan's free bytes minus its need, with 1 GB left. P1024 rv: everything.
  P1152 rv (plan need 12.22 of 16.20 GB): papyrus + phase, 1.9 GB.
- A byte plane is read in place by the target kernels; the packed plane is expanded per slab into the slab buffer (`nn_lab_unpack`,
  on the aux stream in the upload pipeline).
- The component is not 4 bits: the tiled winding numbers its components per tile, ids pass 14. Packed so, 6% of its slab bytes were
  wrong in the stream check.
- `UFSM_LAT_LDEV=mask` forces a set (bit k: phase, winding, component, papyrus), `0` is the stream alone.
- Checks: `UFSM_LAT_STREAM=check` (which keeps reference byte planes) reports 0 differing slab bytes for rv (all four planes, 12
  steps, P256) and for the winding net; `--det 1` step lines are identical with the planes on the device, partly, and streamed
  (rv and winding, 12 steps). `test_gpu_aug` covers the pack / unpack kernels. The check mode died without a papyrus plane (it read a
  null plane); fixed.
- Real loader, rv P768, quiet GPU 1, 2 workers / 3 slots, `--det 1` (the same batches), all four planes on the device against
  `UFSM_LAT_LDEV=0`: 1.123 -> 1.223 samples/s, the mean of steps 4-14 (+9%; loader wait 0%). In a profiled step (`UFSM_PTICK=2`,
  synchronous upload, which also counts copies that otherwise overlap) the copy-back fell from 69 to 18 ms and the slab uploads from
  156 to 9 ms, the step from 896 to 688 ms. P512 beside a training run: 3.60 -> 3.92. **Not measured:** P1024 / P1152 (the dev
  memory slice had no room for the loader).

**Phase and winding planes at 6 and 4 bits** (`UFSM_LAB_QBITS`: the training batches' planes quantised to 2^n cyclic levels on the
device, validation on the stored bytes; P256, `--det 1`, GPU 1 beside a training run, one seed each). Validation after the last step:

| arm | net | phase bce | normal bce | wind4 bce | turn acc |
|---|---|---|---|---|---|
| 600 steps from scratch, 8 bits | winding | 0.69164 | 0.54111 | 0.67356 | 0.3507 |
| 6 bits | | 0.69127 | 0.53992 | 0.67389 | 0.3484 |
| 4 bits | | 0.68629 | 0.55046 | 0.67359 | 0.3478 |
| 600 steps from scratch, 8 bits | rv | 0.65439 | 0.61485 | 0.81120 | 0.3570 |
| 6 bits | | 0.65032 | 0.61675 | 0.80235 | 0.3561 |
| 4 bits | | 0.65462 | 0.61704 | 0.80636 | 0.3564 |
| 300 steps on from wz768's step 2000, 8 bits | winding | 0.67876 | 0.51487 | 0.65902 | 0.3999 |
| 6 bits | | 0.67870 | 0.51491 | 0.65897 | 0.4006 |
| 4 bits | | 0.67864 | 0.51514 | 0.65848 | 0.4023 |
| 300 steps on from g768's step 9000, 8 bits | rv | 0.55067 | 0.51373 | 0.65902 | 0.3738 |
| 6 bits | | 0.55066 | 0.51374 | 0.65899 | 0.3737 |
| 4 bits | | 0.55058 | 0.51388 | 0.65893 | 0.3736 |

6 bits: no difference beyond the arms' noise. 4 bits: the normal bce is higher in three of four comparisons (+0.0002, +0.0003, +0.009
from scratch on the winding net; the normal targets are a difference stencil on the phase), nothing else moves. Weak evidence: the
from-scratch runs had not learned phase or winding yet (bce ~0.69), and the resumed runs sat in the schedule's cooldown (the 8-bit
arm itself moved by < 1e-4). A decision for 4 bits needs a longer run at a working learning rate.

**A label mix-up with two device buffers.** The stream has one set of host label planes. Under `--mem fast` (two device buffers) the
next batch's upload ran during the step and overwrote them while the head passes still uploaded slabs: each step trained on a mix of
two batches' labels (`--det 1`: 13 outcomes in 14 runs, from step 4, losses 2.0810 against 2.0562). The upload now waits for the step,
as under lean; 10 of 10 runs agree and equal a run with device planes. Lean runs never overlapped.

**The label builds of the batch upload (2026-11-10).** With the planes kept on the device the builds were the largest upload item
(P1024: phase 265-275, winding 320-340, papyrus 120-150 ms). Four changes, all with the same bytes:
(1) the warp kernels' phase code took a double-precision `atan2` per voxel (double runs at 1 / 64 of the single rate on these
cards): the single-precision one now decides wherever its value is more than 2e-3 from a rounding boundary, the double one only
there (1 voxel in 250); voxel indices are split with 32-bit divisions. (2) `--gpu-aug 3` uploaded the source-grid phase twice (phase,
then the winding rebuilt from it) and the CT bytes twice (papyrus bits, then the CT channel): the staging buffer now remembers
which host plane it holds (`aug_dev.key`, reset per batch by `nn_aug_stage_reset`), and the papyrus kernels read the CT bytes and
the known bits directly instead of writing the plane over the staged bytes first. (3) Planes kept as bytes are built in place
instead of in the input buffer and copied. Tests: fast against reference kernels (`nn_label_warp_ref`) byte for byte;
`nn_w4runs_label` (untested before) against the host-rebuilt planes, alone and from the staged phase; the CT channel after
`nn_ink_bits` equal to the one on its own. rv guided P512, `--gpu-aug 3`, GPU 1 shared with a P896 training run (contended, every
profiler sync point costs a time slice): label builds + copies 87 -> 45 ms per step, CT upload 29 -> 23 ms; real loader 1.845 ->
1.905 samp/s; 20-step loss identical (2.4004), `UFSM_LAT_STREAM=check` 0 bytes differ, host-stream path and the winding
head equal too. Not measured at P1024 (no quiet card); by voxel count the saving there is ~350 ms of a ~2.3 s step.

**The CT byte sums.** `aug_sums_k` (a voxel a thread, two 64-bit atomics a warp) is now 32 bytes a thread and one pair of atomics
a block; `aug_final_k` one pair a block; every per-voxel index split in nn_aug.cu uses 32-bit divisions (`lw_zyx`). Same sums, same
channel (`TGA_CTBENCH=1 test_gpu_aug 512`, hashes equal). `nn_aug_ct` with its upload, P512, shared GPU 1: no scanner op 16.9 ->
11.5 ms (symmetry), 22-24 -> 16 ms (warped); with a scanner op (load + final + write) 22-30 ms unchanged within the noise.
Real loader with the label changes above: 1.832 -> 1.908 samp/s (+4%, contended both).

**A weight gradient that changed when other processes shared the card (fixed).** `test_lat_head~FP8` failed in ~1 run in 10 when
three copies ran on one GPU and never alone (0 of 55): one conv's weight gradient off by 1e-4..5e-3, everything else equal to 5e-8.
`UFSM_WG_TRACE=1` (hashes of what `lp_bwd_w_f8` reads, a random projection of what it adds) caught calls with byte-identical
activations, gradients, GroupNorm parameters and seed whose output differed, at 96 -> 96, 128 -> 128 and 160 -> 96. Cause: the MX-input
path of `conv_bwd_w_f8_k` keeps per-channel plane maxima in two alternating shared-memory buffers; after a plane's barrier warp 0
reset the *next* plane's buffer, with no barrier before the other warps' reductions into it. A warp that got there first lost its
maximum, the plane's scale came out too small and its e4m3 bytes saturated. Alone the reset practically always wins; with the GPU
time-sliced between short-lived processes it did not. Three buffers now (reset two passes ahead, as `lp_s2.cu`'s kernels already
did): old kernel 18 failing lines in 120 runs, new 0, interleaved under the same load. Not removed by blocking launches, and not
triggered by a steady training run as the neighbour (0 of 40) or by CPU load (0 of 30). Exposure: every `--fp8 1` run beside
benchmark or eval processes (all convs), and in fp4 mode the layers whose weight gradient still runs this kernel on an MX input
(the first conv); the fp4-mode test showed 0 failing lines in 252 runs, so the size there is unknown. `lp_f8fwd.cuh` has the
two-buffer pattern too, but with a barrier between the reset and the next reductions.

**--det 1.** Runs with the same seed printed different lines in 20-70% of cases, always first at step 26 (rv P256), in three
outcomes. The batches (hashes, slots), the input and the label planes are identical; the weights are not: after the first update 321
of 12.6 M parameters differ by one float unit, mostly conv biases. The bias and weight gradients are summed across blocks with float
atomicAdd (`lp_f4wgrad.cuh`, `lp_f8wgrad.cuh`, `lp_s2.cu`, `lp_mx_conv1.cu`, `lp_mx_gnbwd.cu`), so the order varies. The fp4 weight
quantisation hides the difference until a weight crosses a boundary, which is where the runs split. Left as is (decision: no
order-independent sums now): `--det 1` fixes batches and seeds, not the last bit. `UFSM_DET_TRACE=1` prints per-step hashes of input,
label planes and weights; `UFSM_DET_PDUMP` / `UFSM_DET_PSTEP` dump one step's weights.

## The batch upload measured part by part; cheaper scan-augmentation kernels; the wide stride-2 backward on tensor cores (2026-11-10, branch perf-20261109)

**Why.** The P1152 guided rv run made 370-420 Mvox/s against a same-batch figure of 786: the same-batch mode (`UFSM_BENCH_SAME`) skips
the loader and the upload. Under lean 2 there is one input buffer, so the next batch's upload runs after the step, serial to it.

**The fused head on a quiet card** (GPU 0, same batch, free-running, one binary with the env switches):

| samples/s | slab chain (`UFSM_LAT_FUSE=0`) | fused | fused + light pass 0 |
|---|---|---|---|
| rv, lat_guide 5, P1024 | 0.74 | 0.83 | 0.91 (980 Mvox/s) |
| rv, P512 | 5.26 | 5.47 | 6.12 |
| winding, lat_guide 64, P1024 | 0.40 | | 0.42 |
| winding, P512 | 3.09 | | 3.20 |

**The upload of one rv P1024 batch** (quiet card, `UFSM_PTICK=2`, `UFSM_SYNC_UPLOAD=1`; one batch, with blur, correlated noise and gin all
on; 1.7 s against 1.1 s of step compute):

| part | ms | note |
|---|---|---|
| correlated noise | 577 | p 0.1 |
| load + blur ops | 311 | p 0.7 |
| gin | 102 | p 0.15 |
| other scan ops | 31 | |
| CT sums + warp / write | 110 | every batch |
| CT bytes up | 38 | every batch |
| label builds on the device (papyrus 185, side 168, w4 5) | 358 | every batch; mostly the 1 GB source plane going up |
| label planes back to the host | 178 | every batch (the label stream) |
| per-slab label upload in the two head passes | 380 | inside the step, on the aux stream |

Weighted by the ops' probabilities: labels ~0.5 s outside the step, scan augmentation ~0.32 s, CT ~0.15 s, per average step. An earlier
split at P512 beside another process (copies ~11% of the upload) does not hold at P1024.

**No overlap without memory.** With guided heads the full-resolution input is read until the end of the gradient pass (the guide), and
the label builds use the input buffer as their scratch, so uploading batch N+1 during step N needs a second input buffer or 2 P^3 bytes of
label scratch: 2-3 GB at P1152, which the window does not have. Not done.

**Correlated noise in chunks of planes.** It ran two kernels per z plane (2 P launches of P threads each). Now three kernels per 64 MB chunk
of planes: the x recurrences of the chunk's rows, y per (plane, column), z per (y, x) with the carried plane. Byte-identical to the
per-plane kernels (`test_gpu_aug`, chunks of 1 / 5 / P / default planes). P512, op on, beside another process: 316 -> 26 ms.

**Separable operators (blur, lowres).** Micro-benchmark `TGA_BENCH=P build-verify/test_gpu_aug` (GPU 1 beside a training run, so
indicative), ms per pass:

| P, taps | z / y shared-memory | z / y line | x, one row a block | x, rows + weight table |
|---|---|---|---|---|
| 768, 13 | 17.5 | 11.8 | 56.8 | 12.7 |
| 1024, 5 | | 22.8 | 67.1 | 25.2 |
| 1024, 13 | | 27.7 | 137.2 | 30.1 |
| 1024, 21 | | 37.5 | 193.7 | 36.8 |

- z / y: one thread per line with a ring of the last 64 original values (in place, coalesced across x) instead of P x 32 halves of shared
  memory per block (64 KB at P1024: one block per SM). A host check (`bop_ring_ok`) falls back to the shared-memory kernel when a row's
  taps would leave the ring.
- x: the old kernel loaded a different row of the weight table per thread (8 ms per tap and Gvoxel against 1 in z / y, where a warp
  shares its row). Now R rows per block in shared memory and the weights from a table of the operator's distinct rows, built on the
  device (`aug_bop_uniq_k`; more than 64 distinct rows: the plan's own rows). A thread-per-row ring variant measured 80-90 ms flat.
- Byte-identical to the first kernels (`test_gpu_aug`: a 5-tap, an irregular 16-tap and a 9-tap operator, and the table's fallback).

**The 96-wide net's stride-2 convs.** `UFSM_PROF=layers`, winding P512, 10 steps, beside another process: the trunk's convs are 75% of
the profile and scale with the products (dec0.c1, 224 -> 96: 560 ms; enc0.c2 and dec0.c2, 96 -> 96: ~230 ms each), no slow path there.
The down convs were: the fp4 tensor-core stride-2 backward-data existed for 16 / 32 / 64 channels only, wider layers ran the dilated fp8
conv through MX-fp8 copies (a full-resolution conv's work for an eighth of the products). `bwd_data_s2_f4_k` now takes 96 and 128
channels: two K units per tap (the tap's gy blocks 2 j, 2 j + 1; 96: a zero fourth block), 85 KB of shared memory at 128.

| ms per 10 steps | before | after |
|---|---|---|
| down0 (96) bwd_data | 134.1 | 22.7 |
| down1 (128) bwd_data | 31.4 | 4.5 |
| all conv bwd_data | 524 | 327 |

`test_grad_mx4`: 96 and 128 against the fp32 reference like 64 (one draw 0.166, mean of 16 0.042). 160 channels (down2) stay on the
fp8 route (three K units per tap do not fit the shared memory). The stride-2 forward and weight gradient stay fp8 at these widths: the
general fp4 forward (`UFSM_S2_F4G=1`, off for its unexplained drift in test_split) measured 62 against 81 ms, and a fit over the two
nets (17 ms at 32 -> 32, 83 at 96 -> 96) puts ~35 ms of a 96-channel level-0 conv in reading and transforming its input, which a
stride-2 conv pays in full.

**Not measured:** the new augmentation kernels and the 96 / 128 backward in a real run on a quiet card.

## The latent head's full-resolution passes fused per coarse cell; an uncontended step profile (2026-11-10, branch perf-20261109)

Profile first (GPU 0 alone for 5.5 minutes, fp4, streamed head, `--gpu-aug 3`, same batch, `UFSM_PTICK=2`: the target / loss / upsample
kernels ticked one by one; a tick costs ~0.05 ms on a quiet card and ~2.3 ms beside another process, which made an earlier shared-card
table worthless for the per-slab rows). rv = 32,64,128,192,256 with lat_guide 5, P 1024, free-running 0.74 samp/s = 1351 ms:

| phase | ms |
|---|---|
| head forward per slab + upsample (both passes) | 214 |
| targets (both passes): phase 28, rv 31, packed mask rv 44 / side 32, normal 55 | 189 |
| trunk backward (with the per-slab head backward) | 182 |
| upsample adjoint | 178 |
| loss gradients: normal 111, side 42, wind4 20 | 174 |
| guide forward (both passes) / adjoint | 111 / 48 |
| loss statistics: normal 54, side 21, wind4 12 | 87 |
| trunk forward / input pool / optimizer | 71 / 64 / 12 |
| (label slab uploads, both passes: overlap the kernels when free-running) | 315 |

The head path is 1001 of 1330 ms. Winding (96,128,160,192, lat_guide 64) at P 1024, 2500 ms: trunk backward 1134, trunk forward 380,
head forward + upsample 296, loss gradients 164, adjoint 123, targets 115, shuffle 101, statistics 82: there the 96-wide trunk is 60%.
Same-batch mode has no upload: the first batch's upload measured 1150 ms (CT + gpu-aug) + 500 ms (labels) at P 1024, and the P 1152
production run makes 420 - 430 Mvox/s against 786 same-batch. A real-loader P 512 run (shared card) splits the upload into scan
augmentation kernels 41%, label builds on the device 38%, the label planes' copy back to the host 11%, CT sums + write 6%, CT bytes 4%:
kernels, not the bus. In lean 2 the upload runs after the step (one buffer), so it adds to it. Labels kept on the device instead of
streamed: +1.7% same-batch at P 512, not worth 4 - 10 B per voxel.

The fused passes (`nn_lat_fuse_grad`, `nn_lat_fuse_stats`, default on; `UFSM_LAT_FUSE=0` off): one thread per coarse cell computes its 4^3
fine logits (lat_up4_k's arithmetic, each rounded to fp16 as the slab stored it), the guided head's CT terms, the BCE gradient
(loss_grad_v, rounded to fp16 and scaled as the slab chain left it) and the transposed upsampling onto the cell's 3^3 neighbours, for a
loss group at a time: the normal / density channels (every latent head), and the guided side pair with its 5 coefficient channels
(lat_guide 5). 27 fp32 partials per cell and channel go to a scratch (`fpart`, 378 P^2 bytes: 0.5 GB at P 1152, in the planner);
a second kernel gathers each cell's partials in a fixed order (no atomics: two runs give the same bits). The winding mod 4 pair and the
other heads' side channels keep the slab chain on their channel ranges. The statistics pass is fused the same way, and in a training
step it only counts the groups' mask weights (all the gradient reads: weight sums and the active count; `UFSM_LAT_LIGHT0=0` off):
the logged loss sums are taken in the gradient pass and finalised after it (fetch_loss runs after the backward).

Tests: test_lat_fuse (24 cases: coarse gradient against the slab chain max 1.4e-7 of the largest value, statistics 8e-8, mask-only +
gradient-pass sums 8e-8, two fused runs bit-equal). A real rv step: 58 of 14.2 M fp16 codes of the coarse gradient differ by one code
against the chain, 0 between fused and fused + light; winding losses equal to 4 digits over 10 steps; rv losses equal at step 1 and
then differ in the third decimal, as two identical rv runs sometimes do.

| free-running, quiet card | baseline | fused (both passes) |
|---|---|---|
| rv P 1024 | 0.74 samp/s (1351 ms) | 0.84 (1190 ms) |
| rv P 512 | 5.28 (189 ms) | 5.70 (175 ms) |
| winding P 1024 / P 512 | 0.40 / 3.11 | 0.42 / 3.27 |

The fused kernels are compute-bound (fused statistics 29 ms at P 512 against 12 ms for the slab kernels they replace; the gain is the
skipped upsample and slab traffic). The light statistics pass is not timed yet (no quiet card since).

## Real device peaks of the latent nets, the planner's task term, the wide stride-2 backward in chunks (2026-11-09, branch perf-20261109)

Measured in a 5-minute window on an empty GPU 1 (16 GB), per-process peak from nvidia-smi, 3 steps each (1 MiB = 1.049e-3 GB; the
planner prints 1e9-byte GB). rv = 32,64,128,192,256 with lat_guide 5; winding = 96,128,160,192 with lat_guide 64.

| run | real peak | planner then | planner now |
|---|---|---|---|
| P 768 resume of g768 (rv, fp8), streamed / stored head | 4.89 / 5.74 GB | 9.61 / 10.48 | 5.49 streamed |
| P 768 resume of wz768 (winding, fp8), streamed / stored | 8.84 / 13.03 GB | 11.73 / 16.07 | 8.68 streamed |
| P 896 fp4 streamed, rv / winding | 6.15 / 11.54 GB | 12.82 / 13.35 | 6.13 / 8.53 (chunked kernel, below) |
| P 960 fp4 streamed, winding | 13.94 GB | 16.15 | 10.12 (chunked) |
| P 1024 | refused by the planner (18.61 / 19.34) | | 8.55 / 11.92 |

The resumes' losses are equal streamed and stored (wz768 1.1792 1.1791 1.1791; g768 1.2592 1.2590 1.2589). The rv P 960 run made no
step and logged no error (not explained). P 512 beside the production run: winding 2.46 GB real / 2.67 planned, rv 1.88 / 1.91.

Second window (2026-11-10 00:21 - 00:30, after both fixes below; fp4, streamed head, `--mem auto` picked mode 4 without an
override every time; one sampler slot and 2 workers to stay under a 20 GB host cap; 3 - 6 steps, same batch):

| run | real peak | planner | rate |
|---|---|---|---|
| rv P 960 / 1024 / 1088 / 1152 / 1216 | 7.28 / 8.49 / 9.96 / 11.52 / 13.25 GB | 7.41 / 8.72 / 10.20 / 11.84 / 13.66 | 0.87 / 0.74 / 0.60 / 0.51 / 0.43 samp/s (770 - 795 Mvox/s) |
| winding P 960 / 1024 / 1088 | 9.99 / 11.75 / 13.83 GB | 10.33 / 12.16 / 14.22 | 0.48 / 0.40 / 0.33 samp/s (425 - 435 Mvox/s) |

The planner is 2 - 3.5% above the real peak over the whole range. The first window's rv P 960 run with 0 steps was the host cap,
not the device: the kernel log has the cgroup OOM kill at 23:05:17 with 20.0 GB of shared memory (three sampler slots at P 960).

Two planner errors and one kernel:

- The label-stream correction of the task term (the label planes stay on the host: 10 B per window voxel less) never applied: its
  condition read g_xs == 1, which gpu_input_settle sets after the planner. It now reads the request (g_gpu_want). With it the count
  ran 3 - 5% under the real peaks (source not found: at rv P 512 the logged allocations plus the context come to 1.86 GB against 1.82 planned and 1.88 real), so the stream
  branch adds 1 / 14. A run without the label stream after a plan that assumed it prints one line.
- The wide MX-fp4 stride-2 backward-data (more than 64 channels: no direct tensor-core kernel) went through whole-layer MX-fp8
  copies: gx (99 B per level-0 voxel at 96 channels), the dilated gy (132) and the accumulate temp (99), kept by the lp_buf slots:
  5.2 B per window voxel of a 96-wide latent net, 3.7 GB at P 896, in no planner term. It now runs in chunks of fine planes of
  ~96 MB (lp_set_s2b_cz / UFSM_S2B_CZ; 0 = whole): fine planes [z0, z1) read coarse planes z0 / 2 .. z1 / 2, the chunk computes
  one plane further and does not store it. Same bytes as the whole-layer call (test_grad_mx4: 96 -> 128 and 128 -> 160, 2 / 4 / 6
  planes, with and without accumulate: 0 bytes differ); 3-step losses equal; P 512 peak 2658 -> 2346 MiB; step time unchanged
  (contended). The planner counts 3 chunks.

P 512 uncontended, streamed vs stored head: winding 429.4 vs 409.5 ms (synchronised sums), 3.10 vs 3.23 samp/s free-running; rv
542.3 vs 529.7 ms, 5.29 vs 5.33 samp/s. The streamed head costs 1 - 5% of the step for 0.2 - 1.1 GB at P 512 (4 GB at P 768).

## Latent nets with learned cell channels under fp4 activations; the guided heads' adjoint (2026-11-09, branch perf-20261109)

The winding latent net (w768: widths 96,128,160,192, 29 learned cell channels + 3 radial means, `--lat-guide 2`, `--fp8 1`) took
3.2 s a P768 step against 1.3 s for the 32-wide rv net. Same-batch profile at P512 on an idle GPU 1, synchronised phases: 841 ms =
trunk forward 84, trunk backward 310, slab adjoint 206, slab upsampling 50, label upload 37, targets 34, loss kernels 35, CT upload
+ gpu-aug 50, input 8, optimizer 9.
- **Adjoint.** `nn_lat_up_adj` took the per-voxel gather whenever C > 28 (its scratch, the normal targets' 7 B a voxel, holds 28
  channels of the intermediate): the guide-2 head has 32. It now runs the separable pair in channel groups of 28: 206 -> 60 ms,
  step 841 -> 687 ms. `UFSM_LAT_CHECK`: rms difference / rms 1.3e-7 against the gather.
- **`--fp8 1` was the rest.** Per conv, ms a step at P512: fp8 forward 83 / backward-data 58 / weight gradient 234 / GN 14; the same
  net in fp4 (4 inputs): 48 / 46 / 57 / 5. `--fp8 1` was passed only because the first conv (32 MX-fp8 inputs) could not write
  MX-fp4 activations: both mixed paths existed for <= 8 input channels only (the 4-channel stem's small kernel and its FAST weight
  gradient).
  - forward: the general fp8 kernel is now instantiated for MX-fp8 in / MX-fp4 out (it was templated on both; only the stem's
    small kernel was compiled for mixed types), reached for any channel count from `conv3d` and `lp_conv_fwd_f8`.
  - weight gradient: no kernel stages MX-fp4 gradients for a general fp8 conv, so `lp_bwd_w_f8` runs the MX-fp8 / MX-fp8 kernel on
    z chunks (`bwd_w_f8_mixed`): the chunk's gradient rows re-encoded as MX-fp8 (exact: e2m1 values under their block scale are
    e4m3 values) with a zero plane on either side, the input's planes with their halo copied; <= 96 MB of copies whatever the
    window. fp8 compute on the exact operands, as the stem's FAST path does. (The fp4 weight-gradient kernel also accepts it with
    `--policy all=fp4:fp4:fp4`, but rounds the cell channels to e2m1 in the gradient.)
  - `tests/test_lat_conv0`: forward against a host fp32 convolution (rms error / rms 0.038 for fp8 rows, 0.119 for fp4 rows =
    the e2m1 rounding, worst error 0.26 of the row's largest value) and the chunked weight gradient against the host sum and the
    whole-tensor fp8 kernel (differences <= 2e-8 of the rms for 1, 3 and 6 chunks; the bias sums exact).
  A `--lat-basis` run without `--fp8 1` now trains (default `--fp4 2` policy, `enc0.c1=fp16` = the fp8 compute above). Same batch,
  P512, idle GPU: 687 -> 483 ms a step (measured with the fp4 weight gradient for conv0; the chunked fp8 one was built after the
  GPU was taken and is timed only functionally: conv0's weight gradient is ~3 ms of either). Trunk forward 83 -> 48, backward
  306 -> 124. Losses after 4 same-batch steps at P256: 2.4380 (fp4 trunk) against 2.4396 (`--fp8 1`).
  **This is a precision change, not a rounding-level one**: the trunk's activations and gradients are stored and multiplied in fp4
  as in every 4-input net; weights, optimizer state and the checkpoint format are untouched, and a `--fp8 1` checkpoint resumes
  under fp4 (tried on a 6-step one). `--fp8 1` runs exactly as before.
- **Default since the evening: every conv of a `--lat-basis` net is fp4** (`all=fp4:fp4:fp4`, no `enc0.c1=fp16` exception: there is
  no 16-bit stem). Without `--input-prec` the 32 input channels are MX-fp4 too and conv0 is an ordinary fp4 conv; with
  `--input-prec 8` its forward is the mixed fp8 kernel and its weight gradient the fp4 kernel (the cell channels rounded to e2m1
  there); the chunked fp8 weight gradient remains for `--policy ...,enc0.c1=fp16` (15 instead of 3 ms a P512 step).
  300 deterministic steps (`--det 1`, P256, the winding net with the sub-cell head, one seed each; mean loss of steps 1 / 1-50 /
  51-100 / 101-200 / 201-300):

  | | step 1 | 1-50 | 51-100 | 101-200 | 201-300 |
  |---|---|---|---|---|---|
  | `--fp8 1` (as w768 ran) | 2.6173 | 2.6960 | 2.1196 | 1.5390 | 1.3608 |
  | fp4 trunk, fp8 input, conv0 gradient fp8 (chunked) | 2.6315 | 2.7326 | 2.1845 | 1.5868 | 1.3780 |
  | fp4 trunk, fp8 input, conv0 gradient fp4 | 2.6315 | 2.7334 | 2.1827 | 1.5859 | 1.3771 |
  | fp4 trunk, fp4 input (`--input-prec 4`) | 2.6364 | 2.7314 | 2.1830 | 1.5857 | 1.3819 |
  | the new default (no `--input-prec`: fp4 input) | 2.6364 | 2.7335 | 2.1853 | 1.5872 | 1.3799 |

  fp4 against fp8: +1.3% at 201-300 (one seed; no noise estimate). The input's and conv0's precision make no visible difference
  inside fp4 (1.377-1.382).
- **With the sub-cell head (`--lat-guide 3`), P512, idle GPU, same batch**: winding net (96,128,160,192) `--fp8 1` 624 ms, fp4
  trunk 402 ms (trunk forward 89 -> 54, backward 330 -> 143); rv net (32,...,256) 330 -> 287 ms. What is left in the fp4 steps:
  winding: trunk 197 (of it the 96 -> 144 head 37: forward 8, data gradient 14, weight gradient 15), slab section 123, CT upload +
  gpu-aug 50, label upload 15, input 8, optimizer 9; rv: slab section 143 (label upload 38, targets 35, loss kernels 36, upsampling
  15, adjoint 19), uploads 72, trunk 51 (head 12), optimizer 12. For the rv net the full-resolution passes, not the trunk, are the step.
- **What is not fp4 in a latent step, and whether it could be** (sub-cell head, bytes per window voxel): the cell logits `lg` (144
  channels, fp16, 4.5 B = 2.0 GB at P768) and their gradient `latgl` (the same), `latg` (16 channels of fp32 sums, 1.0 B: sums of
  ~100 slab terms, kept), the slab logits `hs_lg` (fp16, one slab), GroupNorm statistics / weights / optimizer state (fp32, 0.12 GB).
  Experiment switches `UFSM_LAT_LG_MX` / `UFSM_LAT_GL_MX` = 8 | 4 round `lg` / `latgl` through MX rows in place (32-channel blocks per
  cell, round to nearest; no kernel reads MX yet). The same 300 steps as above (run-to-run spread of the baseline: 1.3792 / 1.3799,
  the first ~20 steps identical):

  | | step 1 | 1-50 | 51-100 | 101-200 | 201-300 |
  |---|---|---|---|---|---|
  | fp16 both (default) | 2.6364 | 2.7342 | 2.1850 | 1.5853 | 1.3792 |
  | logits MX-fp8 | 2.6347 | 2.7313 | 2.1832 | 1.5851 | 1.3799 |
  | logits MX-fp4 | 2.6103 | 2.7081 | 2.1714 | 1.5859 | 1.3837 |
  | gradient MX-fp8 | 2.6364 | 2.7333 | 2.1815 | 1.5857 | 1.3820 |
  | gradient MX-fp4 | 2.6364 | 2.7325 | 2.1837 | 1.5870 | 1.3801 |
  | logits fp8 + gradient fp8 | 2.6347 | 2.7312 | 2.1828 | 1.5853 | 1.3805 |
  | logits fp8 + gradient fp4 | 2.6347 | 2.7332 | 2.1830 | 1.5857 | 1.3806 |
  | logits fp4 + gradient fp4 | 2.6103 | 2.7077 | 2.1716 | 1.5870 | 1.3820 |

  Nothing separates from the baseline's own spread at 300 steps. fp4 logits move the step-1 loss by 1% (the same weights, another
  rounding of the outputs), fp8 logits by 0.06%. Real MX storage (logits fp8 2.3 B, gradient fp4 1.2 B: -5.5 B a voxel, 2.5 GB at
  P768) needs the head forward writing MX rows, `nn_lat_up_slab` / `nn_lat_shuffle` reading them, the adjoints' final store in MX and
  the head's weight gradient reading an MX gy: not written. It saves memory only, no step time.
- **The streamed head (default for `--lat-guide 0 / 3`; `UFSM_LAT_HEAD_STREAM=0` off): neither `lg` nor `latgl` nor the whole `latg`
  exists.** The network is built with head slabs (`unet_set_head_ext`): `head_slab_pass` runs the head on the cell planes a slab reads
  (`unet_head_fwd_z` on planes z0/4 - 1 .. z1/4, both passes; `nn_lat_up_slab_c` / `nn_lat_shuffle_c` take a chunk of cell planes), and
  the gradient pass moved into the backward's head callback: each slab adds into a rolling buffer of hs_z / 4 + 2 planes (fp32 sums of
  the 16 interpolated outputs, fp16 of the sub-cell ones), the planes no later slab reaches go through `unet_head_bwd_z` at once
  (`nn_lat_grad_fin`), the last two are carried over (`nn_lat_grad_carry`). No rounding is added: the logits are the whole head's bit
  for bit, the gradients equal up to the weight gradient's summation order (tests/test_lat_head: 32 / 96 -> 144 per 3 cell planes
  against the whole head, logits 0 differ, gradients 2e-8 / 4e-8). Bytes of cell logits and gradients: P256 38 MB instead of 170 MB;
  the planner's bytes per voxel 71.4 -> 63.4 (winding net), 71.7 -> 63.7 (rv); at P768 4.6 GB -> 0.33 GB. Training: the rv net
  (32 wide) gives the same loss to 4 decimals through 40 same-batch steps (1.9671 ... 1.8012 both ways); the winding net (96 wide)
  the same through step 4 and 1.6316 / 1.6322 at step 40, which is that net's own run-to-run spread (below).
  Cost: the head forward runs in both passes with 2 halo planes per 8, and the head backward runs per range; the range weight
  gradient used 8192 voxels a block (9 blocks for 8 planes at P384: each of its 54 passes as long as the whole grid's), now >= 128
  blocks (`lp_bwd_w1_mx_b_r`). Timings so far only with ws768 on the same GPU (P384, winding net, synchronised phases, two runs
  each): 466 / 424 ms streamed against 404 / 439 whole, i.e. inside the contention noise; an idle-GPU number is owed.
  `nn_head_bwd_z` takes the general path for more than 16 outputs (the fused kernel is <= 16; 32 inputs -> 144 was "not covered").
- Planner (`UFSM_MEM_DRY=1`, smallest mode, GB; streamed / whole head): P768 rv 8.73 / 13.06, winding 8.80 / 13.13; P896 13.22 / 20.18
  and 13.35 / 20.31; P960 15.99 / 24.58 and 16.15 / 24.75; P1024 19.13 / 29.61 and 19.34 / 29.81. Of the 19.13 GB at P1024 the
  network is 1.48 and the "task" term 16.87 (15.7 B a voxel): it still counts the label planes that `lab_stream` keeps on the host.
  One real run to compare (P384 rv, ws768 on the same card): process peak 1.20 GB by nvidia-smi at 1 Hz (CUDA context included)
  against the planner's 1.83. Real P896-1024 steps are owed (a free GPU).
- Later the same evening: every head streams (the guided ones too: their coefficient planes are interpolated outputs, all through the
  fp32 sums; no sub-cell buffer). 40 same-batch steps, streamed / stored: `--lat-guide 1` rv net 2.7170, 2.6549, 2.5046 both;
  `--lat-guide 2` winding net 2.3987 both at step 4, 2.1172 / 2.1187 at step 40 (the 96-wide spread); plain head 1.9676, 1.9132
  both, 1.7992 / 1.7994 at step 40. tests/test_lat: the chunk kernels and the rolling gradient against the whole arrays (slab
  values bit for bit, coarse gradient within one fp16 code). No checkpoint field changes: running fp8 checkpoints resume as before.
  The planner's task term now leaves out the label planes the label stream keeps on the host (`UFSM_PLAN_LABELS_DEV=1`: the old
  count).
- About the 96-wide irreproducibility below, measured after writing it: one net repeats its own backward exactly only for 32-wide
  nets; a 96-wide net's second backward on the same build differs from its first (enc[0] 0.375, the rest 0.15) while a second net
  of the same configuration reproduces the first net's first backward outside enc[0]. Filling every new device buffer with a byte
  (`UFSM_MALLOC_FILL`) changes nothing, so no unwritten memory is read. K repeated backwards without a rebuild stay unbiased
  (enc[0] 0.153 at 24, 0.080 at 96): the draws are valid, their assignment differs between the first backward after a build and
  the later ones, and between layouts. The op is not located.
- **Resolved as far as production goes (22:30): the irreproducibility below needs level 1 narrower than level 0.** It appears only
  with the test's first widths {96, 64, 96}; {96, 128, 160}, {96, 96, 96} and {32, 32, 32} reproduce every tensor to 1e-9 (one net
  twice, two equal nets, whole head against per range), and so does {96, 64, 96} with `--input-prec 8`. On {96, 64, 96} each
  backward of one net differs from the last in enc[0] and down0 (0.25-0.5), with identical logits and seed counts; it goes away
  with `UFSM_GN_DEFER=0` or `UFSM_COARSE_GRAD_SHARE=0`. So a size assumption of the coarse-gradient share (gout[1]'s home in
  dec[0].a2's buffer beside the 32-channel fp4 input) fails when w[1] < w[0]; the production nets (96,128,160,192 and
  32,64,128,192,256) are not that case. test_lat_head now uses (32,64,96) and (96,128,160) and holds enc[0] to the same bound;
  `TLH_W1=64 TLH_W2=96 TLH_REPRO=1` shows the defect. Not fixed.
- 22:40: the streamed head under `--fp8 1` aborted ("the head backward is not covered per slab") when a g768 / wz768 checkpoint was
  resumed: the per-range head backward's entries wanted storage code 4 / 8 while nn's MXDT names MX-fp8 3. Now the general path for
  MX-fp8 x or gx (as the whole path's fused kernel declines it). Copies of g768/snap/9000 (guide 5) and wz768/snap/2000 (guide 64)
  resumed at P256, 4 steps at the runs' learning rates, streamed and stored: 1.2750, 1.0221, 1.1934, 0.8154 and 1.1716, 1.2295,
  1.3785, 1.1394 both ways. `test_lat_head~FP8` (TLH_FP8=1) runs the head test in that mode. Two `--det 1` runs of 40 steps
  give identical step lines for the rv and the winding net (streamed, fp4).
- Coarse-gradient share: off when widths[1] < widths[0] (one stderr line); the narrow net then repeats its backward exactly, but
  still fails against the 8-output net and under 16-bit gradients (`TLH_NARROW_ALL=1`): open.
- **Found on the way, not fixed: with a 96-wide level 0 under the lean-2 sharing modes (share_enc_a1 + a0_share) enc[0]'s parameter
  gradients are not reproduced between two equal backwards** (two nets with equal parameters, or the head whole against per range:
  rel 0.17-0.26 on enc[0], every other tensor equal to 1e-9; it depends on the allocation layout; 32-wide nets reproduce exactly;
  synchronous offload copies, `UFSM_OFFLOAD_SYNC=1`, change nothing; the storage registry shows no stale entry, `nn_storage_dump`).
  The gradients are valid draws: their mean over K backwards approaches the 16-bit-gradient reference as 1 / sqrt(K) (0.153 at 24,
  0.080 at 96, as the 32-wide net's 0.189 / 0.095). So some SR key or rounding input on the way to enc[0] differs per run for 3-block
  rows; this is probably why two `--det 1` runs of the winding net part after ~20 steps. test_lat_head compares enc[0] apart for
  that reason.
- The slab loop stays two passes: the side / normal gradients divide by the window's mask sums and the count of active channels, the
  winding term by its gauges, all known only after every slab's targets (and the winding logits) were seen.
- Left in the slab section (P512, ms): upsampling 50 and adjoint 60 still cover all 32 planes of the guide-2 head although 13 are
  unused; label upload 37 is hidden by the upload pipeline outside the synchronised profile.

## The winding mod 4 labels on the GPU: train --gpu-aug 3 (2026-11-09, branch perf-20261109)

`band_wind4_tiled` was ~25% of the loader's CPU (hostmem round 2). Its bytes depend on the order of the unions (which run ends up
the root fixes the component's gauge; the conflict count of an inconsistent component depends on the spanning tree; union by size
reads the sizes at that moment), so a parallel connected-components pass cannot reproduce them. The split that keeps the bytes
(`src/nn_w4.cu`, `band_w4dev.h`):
- device, per strip of one tile in z and y (<= T^2 W voxels, 4 B each + the events): the runs along x of each tile row (one thread a
  row; per voxel the run's ordinal in the row and its turns since the run's start), then per run its length and its *events*: the
  first voxel that joins a new y / z neighbour run (position in the run, relative turn), plus the count and 4-bin turn histogram of
  the run's further voxels joining that same run. A count pass, the host's prefix sums (tiles in raster order, rows in scan order =
  the CPU's run numbering), an emit pass.
- host: band.c's union-find replayed over the events in the same order (a run's voxels up to an event are added to its
  component's size before the union, as `w4_pass0` flushes them; the histogram gives the edges and conflicts of the repeated joins),
  the same component choice.
- device: pass 1 from one root offset and component byte per run; the two planes come back by DMA into the (pinned) batch planes.
A P512 window of the P4 phase has 6.0 M runs (16-19 voxels a run) and 14.4 M events (16 B each), against 134 M voxels.

The sampler's workers call `band_wind4_dev` directly (own streams, two contexts per trainer GPU behind mutexes, polling waits: a
stream sync would spin a core behind the trainer's kernels); on any CUDA error it returns -2 and the worker takes the CPU call from
then on. `--gpu-aug 3` = `--gpu-aug 2` + `sample_cfg.w4_dev`; sample.c changes only at its two winding calls.

Checks: `tests/test_w4dev` (72 cases: spirals with holes, vortex lines whose loops gain a turn (10.8 M voxels in components dropped
for conflicts), random phases, windows off the tile grid, one tile with 15 components, 1-voxel tiles) byte-identical to
`band_wind4_tiled` in w4, the component plane, the count and the statistics; two stored P512 phase windows identical;
`UFSM_BATCH_HASH` of 16 `--det 1` P512 batches identical between `--gpu-aug 2` and `3`; a 6-step deterministic training run prints
the same losses.

Numbers (thread CPU, machine shared with two production trainers and their loaders):
- one stored P512 window, tile 86: CPU path 0.65-0.75 CPU-s, device path 0.155 CPU-s (0.125 of it the replay), 0.37-0.41 s wall
  behind a trainer on the same GPU.
- sampler alone (`UFSM_SAMPLER_ONLY=16 UFSM_SAMPLER_PROF=2`, P512, `--det 1`, 4 workers, two alternating runs each): the wind4 stage
  473 / 529 -> 281 / 210 ms a window (the stage also holds phase_air and the side output): -51%. All stages, second pair only (the
  first pair's full profiles were not kept): 1.79 -> 1.38 CPU-s a window, of which the wind4 stage gives 0.32; the rest of that
  difference is the machine's run-to-run drift (ctread equal, augment / phout lower), so the supported figure is about -0.25 to
  -0.3 CPU-s a window of this mix (~-15%).

## Latent-grid training and inference speed (2026-11-09, branch perf-20261109)

**Update (evening): after merging latcomp (lat basis, --latents, guided heads, fit fields from latent nets).** The predict fast
paths by checkpoint kind, each compared with the latcomp binary (6-step checkpoints, box 256 x 768 x 768, window 384, halo 64):
- plain and `--lat-basis` rv: direct input (`nn_lat_pred_input`, basis: `nn_lat_pred_input_c` = the projections of the cell's 64
  z-scored bytes, fp16 storage only) and direct rv bytes (`nn_lat_place_rv`): 351 / 371 of 151 M bytes differ, all by 1.
- guided head (`lat_guide` 5), with or without basis: `nn_lat_place_rvg` upsamples the 2 side logits and 5 coefficient planes per
  voxel and takes the CT terms from a byte copy of the window (the forward overwrites the shared one; 1 B per voxel), with the fp16
  roundings of the slab path: 0 bytes differ. No 4-channel full-resolution input, no [2][W^3] fp16 logits.
- `--latents`, `lat_guide` 8 rv, fit fields with `--fit-ct` / `--tta` / without `--no-out 1`: the latcomp path unchanged.
- `--fit-zarr --no-out 1` from a latent net (`fzs` in predict.c; `UFSM_PRED_FIT_SLAB=0` off): the heads are upsampled (and guided,
  `lat_guide` 8: x for the slab from the bytes, `nn_lat_ct_x`) only over the planes of each z chunk of field cells, the accumulators
  run on the slab as a window of its own (z offset, mask pointer offset; the winding planes in a 2-channel slab with their +-d
  neighbours). Chunks are capped at 768 MB of head planes. All 141 field files byte-identical to latcomp's for a phase, a
  basis + guide-2 and a guide-2 checkpoint. A 640 window (guide 2, 19 planes) took 2.9 GB of the device beside an 11 GB trainer;
  the whole-window buffer alone was 10 GB.
Numbers: one RTX 5060 Ti 16 GB, `train --lat 4 --task unwrap2 --rv 1` (widths 32,64,128,192,256, 11 output channels), P 768, one batch
repeated (`UFSM_BENCH_SAME=1`), phases timed with `UFSM_PTICK=1` (nn_ptick: device synchronised after every phase). PERF_REPORT.md
has the full tables and what was measured on a busy GPU.
- The profile before: 1307 ms a step (347 Mvox/s). The cell-grid trunk (pool + forward + backward + optimiser) was 131 ms; the
  full-resolution head slabs took the rest: upsampling 605 ms (two passes), adjoint 220, label slab uploads 148, loss kernels 119,
  target kernels 84.
- `nn_lat_up_slab`: per axis the pool4 chain (two trilinear 2x steps) is a 3-tap filter of the cell grid (fine 4 c + r reads c - 1, c,
  c + 1 with weights {.375 .625 0}, {.1875 .75 .0625}, {.0625 .75 .1875}, {0 .625 .375}; the mid grid's clamps only occur where
  they fold onto the same edge cell). One thread per coarse cell writes its 4^3 voxels of every channel separably from 27 loads, each
  fine row as one 8-byte store: 302 -> 29 ms a pass (now bound by writing 22 bytes a voxel). `nn_lat_up_adj`: stage 1 gathers each
  fine plane onto the coarse (y, x) grid (10 x 10 loads per coarse cell and plane, weights from a table), stage 2 adds the planes
  along z; the scratch is the slab's normal-target buffer (free after the losses): 220 -> 52 ms. Slabs that are not whole coarse
  planes keep the per-voxel kernels (`UFSM_LAT_SLOW=1` forces them). `UFSM_LAT_CHECK=1` runs both in the trainer: 6.4e3 of 1.48e9 fp16
  logits differ by one code, the coarse gradient by 1e-7 of its rms. Step 1 of a repeated batch has the same loss to the printed
  digits and gn 7.14 against 7.12; later steps drift apart (the MX-fp4 trunk amplifies one-code differences).
- Label slab uploads (`UFSM_LAT_STREAM`): slab j + 1 goes up on the aux stream into a second buffer set while slab j computes
  (`UFSM_LAT_PIPE=0`: in the compute stream as before; +0.08 GB at P768, skipped when the device is full). Stream check 0 differing
  bytes, losses identical with it on and off. Best 5-step windows on a shared GPU: 930 Mvox/s with, 564-636 without.
- After: 602 ms a step measured without the upload pipeline (752 Mvox/s), <= 487 ms with it. Not adopted: one fused target kernel
  (same time within noise: the targets are bound by their 14 written bytes a voxel; its normal targets differed by one code in 2e-4
  of the voxels), 64-plane slabs (+2.5%, within noise), skipping unsupervised slabs (0 of 128 slabs in 8 windows; 51-94% of the
  voxels carry the side loss).
- The sampler now limits training: ~7.7 CPU-seconds per P512 window in steady state (17 Mvox/s per worker), and at start 86% of the
  trainer's CPU samples sat in THP direct compaction (wbig_alloc's MADV_HUGEPAGE with defrag = defer+madvise on a full page cache).
- predict `--channel rv` on a latent checkpoint: `nn_lat_pred_input` builds the cell-grid input from the CT bytes (the values of
  nn_pred_input_d + nn_lowres_input, output bytes identical), `nn_lat_place_rv` writes each shard's bytes from the cell logits (the
  upsampling above, nn_rv_fuse and nn_pred_place in one kernel: 1139 of 4.6e8 bytes differ by 1); no full-resolution input (8 B a
  voxel), half-resolution intermediate (2 B) or fp32 logits (8 B): `--window 1152` fits beside a P768 trainer. The place kernels of
  every predict run one thread per voxel of the interior-shard intersection (they ran over the whole interior per shard). With
  `--shard 128` a shard is one 128^3 chunk, so `--threads` had no effect on the single writer thread (144 MB/s): up to 8 writer
  threads now (`UFSM_PRED_WRITERS`). Box 1024 x 3072^2 of PHerc1667 at window 640, back to back at machine load 20: 66.0 -> 16.5 s
  (7.6 s at a quieter moment; the writers and the CT reader bound it). tests/test_lat.c covers both new
  kernels against the full-resolution chain.
## Latent-grid networks and latcomp v1: train --lat 4, learned cell channels, latent stores (2026-11-09)
- Why: at the production voxel net level 0 is ~86% of the activations (61.8 B per window voxel at P576). `train --lat 4` (task unwrap2 /
  --rv; unet_cfg.half_res 3) runs the U-Net on the 4^3 cells of the window (widths 32,64,128,192,256, 5 levels); the losses stay at
  full resolution: head_slab_pass upsamples the coarse logits one 32-plane slab at a time (nn_lat_up_slab: the pool4 chain of
  nn_lowres_output, per axis a 3-tap filter of the cell grid; separable per-cell kernel since perf-20261109) and gathers each slab's
  gradient back onto the cells (nn_lat_up_adj, the exact adjoint; tests/test_lat.c). The input is built on the device from the
  augmented CT channel and the radial records (nn_input_lat: cell means).
- Memory: label planes (phase, winding mod 4 + component, papyrus) live in pinned host memory and reach the device per head slab
  (lab_stream: warped on the device as before, scratch = the input buffer, D2H; UFSM_LAT_STREAM=check: 0 of 8.7e9 slab bytes differ).
  Device peak at P960: 10.2 -> 7.7 GB (one 5060 Ti; the voxel net fits P576). Host RAM is the limit above that (HOSTMEM_REPORT.md).
  Two size limits of the scanner augmentation showed above P768: the Paganin x pass's shared memory (rows per block now follow 96 KB)
  and the radial table (SA_MAXTAB 16 x SA_MAXP; SA_MAXP 1536).
- Speed (perf-20261109, same batch, P768): the full-resolution slab work was 90% of a step (upsampling 605 ms of 1307); with the
  separable upsampling and the two-stage adjoint a step is 602 ms (752 Mvox/s on the GPU side). The loader is the bottleneck:
  ~60-78 Mvox/s per worker, and at most (slots - 1) windows are built at once (use --sampler-slots = --workers + 2).
- latcomp v1, the learned cell channels (src/latfit.c, `train --lat-fit FILE`, `--lat-basis FILE --lat-c 13|29 --lat-step S
  [--lat-step-jitter J]`): a cell's 64 z-scored CT values on an orthonormal basis fitted on real augmented training windows: row 0 the
  cell mean, then the principal axes of the mean-removed cells. 24 windows of the rv recipe (778k cells): the mean carries 86% of the
  voxel variance; of the rest 75% is in 3 axes, 96% in 13, 98.5% in 29. Coefficients are quantised with one step (orthonormal: the
  rate-optimal allocation), |q| <= 127 (the mean: 16 bits), scaled by 1 / max(sigma, step) (the mean by 1/8) and fed with the 3 radial
  cell means as C + 3 = 32 inputs (nn_input_latc in the trainer, nn_lat_input_x in predict: byte-identical, test_gpu_input). Rate /
  distortion (zeroth-order entropy, z-scored CT): C 29 step 0.25: 0.89 bit per voxel, 23.9 dB; C 13 step 0.5: 0.47 bit, 20.5 dB. The
  transform rides in the checkpoint header (lat_c / lat_step / lat_gain / lat_basis; UFSM_CHECKPOINT_HEADER 131072). The 32-input first
  conv needs one MX format for its input and output: `--fp8 1` for now (the 4-input stem kernels take <= 8 channels).
- Latent stores (`ufsm lc-encode BASIS ROOT KEY OUT`, src/lc_cli.c): sharded zarr v3 int8 [Z/4][Y/4][X/4](C + 1) through the embz
  writer (zstd): the mean index as two bytes, then the axes. Coefficients are of the raw CT bytes with step_raw = step x sd_ref: a
  window's z-score is linear (all voxels count, air included), so a reader gets the trainer's channels as
  (q step_raw - [mean] 8 mean_w) / sd_w with mean_w / sd_w from the coefficients themselves (58.664 / 33.01 against the CT's
  58.664 / 33.29 on a P1667 window). `predict --latents STORE` feeds a --lat-basis checkpoint from a store (nn_lat_input_q; the CT is
  still read for the air mask and the radial channels). PHerc1667 level 2, a dense 768 x 1024^2 box: 0.175 B per voxel on disk
  (5.7x under uint8; entropy 0.16), rms error 4.8 grey levels (34.5 dB).
- Open: accuracy of the pair m768 (cell means) / k768 (learned channels) against rv1d2 on p1667 (FINDINGS.md when done); a tANS
  container instead of zstd; a first conv for 32 fp8 inputs into fp4; predict without the CT read; compressed outputs.

## Host RAM of train --lat at large windows (2026-11-09)
- Accounting: `UFSM_HOSTMEM=1` (a `hostmem:` line per log line: pinned by caller, slots, per worker `big` / `ttmp` / pool residency via
  mincore, mallinfo2), `UFSM_TP_LOG=1` (worker 0's pool calls by line), `UFSM_BATCH_HASH=1` (per-plane batch hashes, with --det 1).
- The lat recipe (unwrap2 --rv, --gpu-aug 2) was 7 B/vox per slot + 11 B/vox per worker (P960 W18 S6: 213 GB, the OOMs). Now 5 + 5:
  tm / pm slot planes skipped (sample_cfg.slot_skip), the stored-phase block's 4 planes in dead worker scratch (ttmp, big + P^3,
  big + 9 P^3) for unwarped / device-warped windows, the ink plane built in the slot. Batches byte-identical (30 --det batches).
- A worker builds into a slot it claims, so at most S - 1 windows are built at once: W8 S3 is slot-bound (P640: 38% wait; W4 S5 0%).
  Use S = W + 2. P960 W8 S10 ~81 GB, P1280 W4 S6 ~106 GB (model 1.2 GB + P^3 (5 W + 5 S)). Details: HOSTMEM_REPORT.md.
- Round 2: UFSM_SAMPLER_ONLY (GPU-free loader benchmark), UFSM_SAMPLER_PROF=2 (thread CPU per stage), UFSM_SAMPLER_RT (threads per
  window, byte-identical). Loader CPU -11% per window (fused / skipped P^3 passes; P512 --det: 5.92 -> 5.25 thread-s, cycles -9%); per
  worker 4 B/vox (phase built in place). The loader is CPU / DRAM-bandwidth bound at ~25-30 Mvox/s per core (decode 25%, wind4 25%);
  reads ~0.33 B/vox, 80% phase labels, page-cache hit ~0: UFSM_STORE_NOCACHE=1; THP madvise off with --gpu-aug 2 (compaction stalls).
- Round 3: `--replay K` interleaved (`--replay-gap G`, FIFO, default (S - W) K / (K - 1); --det 1 byte-identical between runs). P512 W4
  S6 at load ~8: K=1 428, K=2 773, K=3 985 Mvox/s delivered. z3_read decodes volcomp chunks straight into the request box (window sink,
  UFSM_Z3_SINK=0 off; tests/test_z3_sink): phase reads -12.5% CPU. xoshiro GF(2) jump-ahead for skipped draws; five more P^3 passes /
  streams cut (the ignore and loss masks only when read, the rv plane from ct8). Loader -5..-10% CPU per Gvox vs round 2b (paired A/B),
  batches byte-identical.

## Own dense embedding net (dino-train) and the dinovol port (2026-10-21/22)
- `ufsm dinovol-embed` runs the scrollprize dinovol ViT (HF scrollprize/dinovol_8um_ps192_102126) in C/CUDA with embed_volume.py's
  lattice / blending / normalisation; `dinovol-read`, `dinovol-eval` (the sheet-identity check; `--codes` classifies misses by the winding
  code raster: same wrap off-surface / adjacent wrap / farther / unlabelled). Parity: tests/test_dinovol.py (one tile vs the official
  PyTorch model), tests/test_dinovol_embed.py (the whole tool vs the official RegionEmbedder).
- `ufsm dino-train` (src/dino.c, kernels src/nn_dino.cu): our U-Net from scratch with D = 16 unit embedding channels at full resolution
  (the MX head takes <= 16 outputs). Losses on sampled voxels only: GT sheet-identity SupCon over labelled voxels (winding codes,
  positives round(dq) == 0 anywhere in the window, k +- 1 within --hard-r weighted --hard-w), keys = the student's points plus the EMA
  teacher's at the same voxels of the other view; a variance hinge (--var) against the collapse SupCon falls into while features cannot yet
  tell wraps apart (collapse is a stationary point: positives' mean = all keys' mean); optional DINO prototype cross-view term (--ssl) on
  papyrus voxels with teacher centring / sharpening. Views: the same window with independent noise / gain / bias (ct_b: a second scan or
  sub-voxel halves). `ufsm halves` builds 4x-pooled views from disjoint 2.4 um sub-voxel halves, but on Paris 4 the halves differ by
  only ~0.3 grey levels rms (both our lossy volcomp level 0 and the raw S3 zarr are smooth at 2.4 um): no independent noise there.
  `ufsm dino-embed` writes the dinovol-eval format; `dino-boxes` picks labelled (codes) or papyrus-rich (coarse CT) training boxes.
  tests/test_dino_loss.c checks the loss gradients against double CPU references (D 16 and 32).

## Hybrid conv + attention U-Net: the bottleneck transformer stage (2026-10-22)
- `unet_cfg.attn_blocks` (train / dino-train `--attn-blocks N --attn-pe peg|rope|none --attn-reg K --attn-mlp F`; default 0 = the
  plain U-Net, bit for bit). The bottleneck level's raw output x (its last conv / residual block a2, before the GroupNorm the decoder
  applies) goes to fp32 token rows [n][registers + D H W][C], then x + PEG(x) (a zero-initialised depthwise 3^3 conv on the token grid;
  `rope`: axial 3D RoPE on q / k instead, 8 frequency pairs per axis, base 100; `none`), then N pre-LN blocks: LayerNorm -> fused QKV ->
  QK-norm (RMS over each head's 48 dims, a gain per channel) -> softmax attention (heads of 48, all tokens of the window) -> out
  projection; LayerNorm -> SwiGLU (hidden 8/3 C) -> projection; fp32 residual stream, zero-initialised out projections (a fresh stage is
  the identity). The result is written back as a new a2 in the activation storage (MX-fp4 in production) with its own GroupNorm
  statistics; the GroupNorm parameters stay the bottleneck's n2 (stage_out() returns the stage's output block). Width % 48 == 0
  (dino-train's 192: 4 heads). Parameters appended last (older checkpoints load unchanged); the four linears per block are Muon
  matrices (for_each_mat: the 3^3 convs + the linears), norms / biases / PEG / registers AdamW; Muon-C / Hyperball refuse the stage.
- Kernels src/nn_attn.cu (src/attn.h): fp32 throughout. Linears are cuBLASLt fp32 GEMMs (UFSM_ATTN_TF32=1: TF32); attention is a
  flash-style SIMT kernel (64 queries per block, 32-key shared tiles, online softmax in log2 units) that stores only the per-row
  log-sum-exp; the backward recomputes the probabilities tile by tile in two kernels (dK / dV per key, dQ per query: no atomics, no
  S x S matrix). Per-block recomputation: the stage keeps one [T][C] row set per block input; a block's backward re-runs its forward
  into one shared scratch. Bottleneck tokens: P 256 -> 512, P 384 -> 1728, P 512 -> 4096, P 768 -> 13824 per window (6 levels).
- Backward hook: d/d(silu(gn(stage a2))) -> the stage's GroupNorm backward -> fp32 -> blocks -> PEG -> d/d x in the gradient storage,
  handed to the bottleneck block's backward as an already-normalised gradient (block_bwd pre = 1). Gradients stay in the backward's
  scaled domain (the flat parameter gradient is divided by the gradient scale at the end).
- Tests: tests/test_attn.c (block input / every parameter gradient vs central differences of a double CPU forward at 66 and 512
  tokens, 2 and 4 heads, with / without RoPE and registers; attention alone at 4100 tokens vs the double analytic backward; PEG and the
  layouts), tests/test_hybrid_identity.c (production MX storage: a fresh stage on a conv net's trunk reproduces its logits / trunk
  gradients up to re-quantisation; a random stage: every stage tensor gets a gradient; checkpoint round trip).
- Scroll-coordinate attention (2026-10-24, default off: `unet_cfg.attn_axis`, `attn_pe` 3, `attn_repeat`; train `--attn-axis full|radial
  --attn-pe scroll --attn-repeat K`). Each window carries its scroll-axis record (sample.c axis_record: the axis centre per window plane
  along the window dim nearest the source z direction, through the cube symmetry and the warp's forward map; predict: the axis rows of
  the window or its TTA transform; unet_set_attn_axis). The stage averages it onto the bottleneck planes and plans rays (src/nn_attn_ray.cu,
  host): per plane `UFSM_ATTN_RAY_MULT` (2) x Pmax rays leave the axis centre over the angles that see the plane (all of them when the
  axis is inside), unit-spaced samples centred on each ray's chord (at most Nr = the plane diagonal + 1). `radial`: each block's q / k / v
  are gathered onto the ray samples (bilinear, edge-clamped), attention runs per ray (sheet crossings in order; keys past a ray's length
  masked in the fp32 and fp16 cores; MX-fp8 falls back to fp16), and the output is scattered back normalised (o[v] = sum w o_s / sum w);
  the backward is the exact adjoint. `scroll` RoPE: rotary on each token's continuous (r, arc = theta x rc, plane index), 16 head dims
  per axis, on the rays or (full attention) on the grid tokens. `attn_repeat` K runs the blocks K times with the same parameters
  (gradients summed). Costs (laptop 5080, one block, two forwards + backward, fp16 core, MX-fp4 linears, kept forwards): 16^3 tokens
  (P512, 6 levels) C 192 full 2.23 ms / radial 1.70-1.76 ms (3.0x ray slots, 35-53 percent used), scratch 96 -> 232 MB; 24^3 (P768,
  13.8k tokens) 17.3 / 6.5-7.3 ms, 323 -> 767 MB; C 288: 16^3 3.34 / 2.5-2.7 ms, 24^3 26.0 / 9.6-10.5 ms. Grid coverage (tokens some
  ray sample touches; the rest keep their residual only) 0.998 with the axis inside the window at mult 2, 1.000 outside or at mult 3.
  Tests: tests/test_attn_ray.c (gather / scatter vs a double stencil and their adjoints; radial and scroll-RoPE blocks vs finite
  differences of a double forward, cores 0 and 1), tests/test_hybrid_identity.c (fresh radial / scroll / repeated stages are the
  identity; a tied 1 x 2 stage equals the untied 2-block stage with its gradient the sum; checkpoint keeps the flags),
  tests/test_gpu_input.c (the records' directions = the radial input channels for every symmetry and slabs, warped within 0.2 rad).
  The fp16 core's dK / dV kernel zeroes P for masked keys: a padded ray slot (k = 0) scored above very negative valid scores made
  exp2(s - lse) inf and the masked row's 0 x inf NaN (the k-gain gradients; 14-28 skipped steps per 500 before the fix; test_attn_ray
  test_masked_keys). Smoke (laptop, unwrap2 wind96 widths, P256, 2 blocks radial + scroll, 500 steps): no non-finite steps, 4.83
  samples/s vs 4.99 without the stage (~3 percent); P512 one-batch step 1.39 s none / 1.41 full / 1.41 radial.
- Window memory tokens (2026-10-24, default off: `unet_cfg.attn_mem` M; train `--attn-mem M --chain-p p --chain-len L --chain-step s`;
  Transformer-XL style, no extra batch memory). M token rows follow the registers in every attention sequence; row j = learned row +
  gate (x) mem_in[j] + W_off (dz, dy, dx, valid) (gate and W_off zero-initialised: a fresh stage stays the identity). After the last block
  the rows are the window's memory output (unet_attn_mem_out). The sampler draws window chains (sample_chain_next: with probability p a
  fresh draw starts a chain; the slot's next L - 1 draws walk s x P from it along the scroll axis or around it, same source / level /
  symmetry, no zoom-out / spatial warp; batch chid / chidx / choff). train keeps a per-GPU bank (64 entries) of each chain's latest
  memory output (device rows, a constant: stop-gradient) and feeds the successor its predecessor's rows with the offset (window frame,
  / P) and valid 1; chain starts, misses and validation get zero memory with valid 0. predict carries the memory from the previous window in its
  walk order when that lies within a window size (else zero); TTA passes run without memory. Needs full attention (not radial). Tests: test_attn_ray test_mem
  (add / gate / W_off gradients / out vs double), test_hybrid_identity (identity with a random memory, a random stage: memory changes the
  logits, none = zero memory bit for bit, gate / W_off gradients nonzero, the carried rows change the next window, checkpoint),
  test_wind_sampler (chain ids / indices / offsets per slot). The stage builds lazily at the first forward: the build no longer resets
  the set memory (it made the first window's memory zero).
  Smoke (laptop, unwrap2 wind96 widths, P256, 2 full blocks, 300 steps, 3 workers; sampler-bound): 4.40 samples/s vs 4.66 without
  memory, every continuation (215) got its predecessor's memory, no non-finite steps.
- Scan fusion as joint attention (2026-10-24, default off: `unet_cfg.view_P`, `view_patch`; train `--view-P Pv --view-patch k` (8);
  sources `"view": {"level": L, "map": REG.json}`; predict `--view-level L [--view-map REG.json]`). Per window a Pv^3 uint8 crop of a second
  scan centred on the window (src/view.h sview_origin: level-L voxels of the source's grid, L <= the window level), either the scan itself
  at a finer level (no map: e.g. the 2.4 um detail around a 9.6 um window) or a registered partner scan through a ufsm reg map
  (fine(c) = A^-1 (c + D(c)), reg_read_fine + reg_resample, trilinear from the partner level nearest the voxel size; e.g. Paris 4
  137 keV on the 78 keV grid: ~/ufsm-data/reg/pairs/PParis4_2to2/final/reg.json). The crop takes the window's cube symmetry (sym_u8);
  windows with a view skip zoom-out and the spatial warp. In the stage the crop is cut into (Pv / k)^3 patches of k^3 voxels, each a token
  row after the registers / memory rows: learned row (its position) + W patch / 255 + b; the grid tokens attend to them in every block
  (the rows also get the blocks' updates), so a fresh stage is still the identity. No view (or a source without one) = zero patches.
  Needs full attention. Tests: test_attn_ray test_view (patch / add / gather / bias gradient vs host), test_hybrid_identity (identity,
  random stage: the view changes the logits, none = zero view bit for bit, W / b gradients nonzero, checkpoint), test_wind_sampler (the
  crop = the CT box centred on the window).
  Smoke (same setup, view_P 32 / patch 8 = 64 view rows beside 512 grid tokens): self view 4.40 samples/s vs 4.66; the Paris 4
  137 keV scan through PParis4_2to2 (mirror reads) 4.02. Partner crops match the 78 keV crop where the partner exists (mean 45.9 vs
  45.7) and are zero beyond its z range (coarse level-0 z ~42100..48700).
- Deformable attention (2026-10-24, default off: `unet_cfg.attn_deform` K, `attn_dlev` L (2, at most 3); train `--attn-deform K
  [--attn-dlev L]`; src/nn_attn_dfm.cu). Multi-scale 3D Deformable-DETR in place of each block's self-attention (subsumes the planned
  normal rays / anisotropic bias / spiral RoPE): per token a linear layer (H L K x 4 rows: per head, level and point a 3-D offset in
  level-l voxels, then the H x L x K logits; zero weights, the offsets' bias a grid prior of Fibonacci-sphere directions at radius
  (1 + k % 4) / 2 bottleneck voxels, logits 0) and per head a softmax over the L K logits; the head's values are trilinearly sampled
  (zero outside the level's grid) at the token's centre on level l plus the offset and summed. Level 0 = the bottleneck's tokens (value
  projection C x C, the q / k / v slot), levels 1..L-1 = the next finer encoder outputs (2^l x the grid, their raw a2 LayerNorm'd
  without affine; a value projection C x C_l each, no gradient into the trunk from them). Then the out projection (zero init: a fresh
  stage is the identity) and the SwiGLU MLP as before. No registers / memory / view / radial / RoPE. The offset / logit and finer-level
  projections train with Adam (Muon: the value, out and MLP matrices); the finer-level weight gradient runs in fp32 cuBLAS (MX-fp4 was
  4x slower at its long K). Kernels: one warp per (token, head), lanes on the head's 48 channels; the backward scatters dV with atomics
  and reduces the offset / logit gradients over the warp. Tests: test_attn_dfm (kernels: o, every level's dV, d offsets / d logits vs
  double central differences with points just inside voxel boundaries and partly / wholly outside the grids; whole blocks with 1 / 2 /
  3 levels vs a double forward incl. the finer projections, the kept forward = the recompute, zero-init identity bit for bit),
  test_hybrid_identity (fresh stage identity, random stage: all 20 block tensors nonzero finite gradients, checkpoint, inference build
  = training build).
  Bench (laptop 5080, C 192 / 4 heads, F 512, one block, teacher fwd + student fwd + bwd, kept forward; ms and block scratch):
  P512 (16^3 tokens): full 2.42 / 96 MB; K8 L2 2.71 / 136 MB; K16 L2 3.56 / 146 MB; K32 L2 5.75 / 165 MB; K8 L3 6.46 / 544 MB;
  K16 L3 8.52 / 558 MB. P768 (24^3): full 17.2 / 323 MB; K8 L2 9.17 / 460 MB; K16 L2 13.2 / 492 MB; K32 L2 20.4 / 556 MB; K8 L3 22.8 /
  1835 MB. Flash attention keeps no S x S matrix, so at these token counts deformable costs MORE memory (the finer levels' fp32
  values: 8^l x the grid x C per block) and is only faster from ~P768 (K <= 16, L 2). Smoke (P256, 2 blocks, K16): L2 4.31-4.55
  samples/s, L3 4.62-4.70 (sampler-bound; full attention 4.66), finite.
- Deformable attention for the winding fine-tune (2026-10-24, all default off; wdef = wind96 + 2 blocks K16 L3 lost HOLD 0.93 -> 0.79,
  the conv trunk drifting). (1) train `--trunk-lr-mult F` (unet_set_trunk_lr): every parameter before the attention stage (Muon convs,
  their AdamW tail, packed MX convs) at F x the LR, the stage (its tail [tb_p0, np)) at the base LR; Muon runs two batched calls.
  (2) `--attn-deform-init normal` (`attn_dinit` 1): per-token frames (normal = radial from the scroll axis, tangent, axis; from the
  unet_set_attn_axis records, identity without an axis; at_dfm_frame) rotate the learned offsets (p = c + sum_j o_j R_j); the prior
  spreads the K points along the normal (+-2 voxels x 2^l) with a small per-head lateral twist. train / predict pass axis records when
  set (unet_cfg_axis). (3) `--attn-dcoarse 1` (`attn_dco` 1): one more level after the finer ones = the 2x2x2 mean-pooled bottleneck
  values (no own projection; gradient unpooled into level 0), its own offset / logit rows. Warm starts from another deformable layout
  copy tensor by tensor (dfm_copy_block: value / out / MLP / finer projections and the common levels' logit rows; offset rows only with
  the same frame mode; new rows keep their init). Tests: test_attn_dfm (coarse level and frames vs double central differences, frame
  identity, frame kernel and normal prior geometry), test_hybrid_identity (coarse + frames identity, gradients, axis dependence,
  checkpoint, warm-start remap, trunk LR x 0.5 under Muon and AdamW with the stage unchanged and 1 = default bit for bit).
  Laptop smoke (5080, wdef snap 12001 + all three + --aug-crush 0.5, trunk x 0.25, P384 level 2 of the local Paris 4 box, 80 steps):
  --mem fast fits (planner 12.38 of 15.53 GB; P512 does not: lean 2 fallback), GPU ~634 ms/step (UFSM_PROF sum) = ~89 Mvox/s, finite
  losses (1.50 -> 1.46-1.51), 72 crushed windows; loader-bound at 6 workers with crush (0.2-0.4 samples/s).
- dino-train for large windows: `--head-slab Z` runs the 1^3 head per z-slab of Z planes (unet head slabs): no whole-window logits
  (P 512 at D 32: 8.6 GB of fp16 logits) and no dense fp32 logit gradient (17 GB) -- the points' embeddings are gathered per slab, and
  in the backward's head step their gradients are scattered into each slab's fp16 logit gradient. `--mem lean` sets train's smallest
  recompute-1 mode (chunked MX gradients, lean 2, shared encoder a1); `UFSM_MEM_DRY=1` prints the dry-build bytes and exits.
  `--minutes M` sets the step count from the speed of steps 21..200 (matched wall-time A/Bs). dino-embed `--norm auto`: the box's
  median / IQR statistics as dino-train computes them for a source.

## Cleanup 2026-10-20: the dead Python tools removed

Removed (in git history before this commit): every Python tool in `tools/` (the sheet / surface-store / finite-cover
pipeline: band_field, build_*, sheet_*, score_*, surface tools, production.py, eval_holdouts, make_sources, label_grid,
papyrus_eval, torch_bench, roofline, killdl, gen_lp_launch (its src/lp_*.cu output is committed; the Makefile never ran it)),
the shell helpers that drove them (confirm_run.sh, train_r1.sh, perf_baseline.sh), tools/surface_dedup.c and
tools/check_surface_samples.c, their tests (test_production*, test_sheet_{cli,geometry,native,pipeline,watch},
test_surface_store, test_training_cover, test_score_sheets, test_eval_holdouts, test_evaluation_plan, test_gpu_leases,
test_extend_surface_training, torch_compare.py), SHEET_PIPELINE.md, requirements-sheet.txt, the production / sheet / cover
recipes in configs/, and scripts/podbatch (the 2026-10-16 RunPod fit-field batch). The dated sections below are history:
their references to these files point into git history. The C side of the sheet / surface-winding task and the finite
cover (src/sheet.c, src/cover.c, train --geometry / --cover) stays: it is wired into train / sample / predict.
Two-GPU tests (test_sheet_gpu, test_cover_training.py, test_pipeline_cli.py) skip their second-GPU parts on one GPU.

## Integration 2026-10-20: every local branch on master

origin/master was 7ae7a47 (10-16); everything since lived on local branches. Merged in this order (first-parent chain of master):
1. training chain `wind-sense-20261019` (contains warp-speed / wind2-run, loader2, the unwrap2 worktree with the 2-GPU leak fix,
   phase-sign, multi-scroll-20261016); 2. SR `train --task restore`; 3. `fitter-p4` (fitter, fitter-p2 / p3, coreline-1299 and the
   stitch workers), `fitter-p1`; 4. the stopped WIP branches `fitter-speed`, `render-speed`, `wind24-20261020`; 5. `podbatch-fitq`;
   6. worker D, X (`--val-P`), K1 + R, S2, S3, S4; 7. the other 10-15 / 10-16 worker branches; 8. the main checkout's C work.
D, K1, R and the remaining 10-15 / 10-16 workers had been cherry-picked into master before 7ae7a47 (patch-equivalent, or their added
lines present; the rest later edited in place): recorded with `merge -s ours` (no tree change) so no branch holds unmerged history.

Non-trivial resolutions:
- `fit.c` / `fit.h` / `nn_fit.cu` / `predict.c` / `checkpoint.c` / `tests/test_fit_w4.c` (training chain vs fitter): the fitter
  versions. The training chain's only change to them (9a3d51e: unwrap2 fit planes and the winding mod 4 term) had already been
  ported by the fitter into `fit_terms.h` and the variable plane list (`FFZ_MAXP`; r, dz dy dx dq, ct cs).
- `train.c` (unwrap2 vs restore vs `--val-P`): one task flag block (unwrap2 implies unwrap; restore beside it), the loss dispatch
  `restore -> unwrap2 -> unwrap`, planner bytes with both the unwrap2 and the restore terms, the step launch keeps the NaN hooks /
  sync checks around `restore_train_recipe(), run_batch()`; validation runs the `--val-P` copy, else the existing path with the
  restore per-family pass and the unwrap2 metrics; `--val-P` also refuses task restore; the planner keeps both
  `UFSM_MEM_FREE_GB` and the `--val-P` reserve.
- `sample.c` / `sample.h`: `draw_restore` beside the wind4 warps.
- `fit.c` (fitter-speed vs fitter-p2): the threaded row pooler keeps the pooled CT spacing `cs`; `fit_fields_load_ex` keeps
  `fit_fields_ct_planes`; the slab-to-slab row reuse compares all 17 planes (was 15: ct / cs postdate fitter-speed's base).
- `predict.c` (podbatch vs fitter): the ring of 3 field-slab rows and `--fit-q` / `--fit-q-phase` / `--no-out` on the variable
  plane list (`ffz_np` planes, `FFZ_MAXP` staging, `ffacc4` freed).
- `lp_mx_s2bwd_tc.cu` (S2 vs K1-era one-pass-ahead loads): the prefetching loop starts at the z window,
  `tz_beg = tz0 + zc ZC`, `tz_end = min(nzt, tz0 + (zc + 1) ZC)`.
- Makefile: one SRC list, the union of the test-all prerequisites and runs; duplicate `test_fit_w4` rules dropped.
- Main checkout (uncommitted on 7ae7a47): kept `ufsm blend`, `predict --input-gain / --input-bias`, README sections and two of its
  test_fit cases (ported to the fitter API); its recto term, r plane, flags and fit-config.json were already in the fitter.
  Dropped the old Python restoration / winding-ensemble / ink-TTA code (never tracked; `train --task restore` replaces it).

Determinism / `test_share_enc_a1`: with S2 merged it failed 8 of 10 runs (MX-fp8 mode 1, "fresh" / "repeated" gradients 1e-4..7e-4
apart instead of ~1e-7). The cause is the size of `split_t` alone: the pre-S2 tree with 8 padding bytes in `split_t` (136 bytes)
fails 8 of 12, without them 0 of 12. Fixed-order GroupNorm partial sums in the forward epilogues (warps committing in turn, Phase B1)
did not change it (7 of 12) and were not kept. The kept fix: the int z fields of `split_t` (zi0, Di, o_pad, zw0, zw1) are int16 and
a `static_assert` holds `split_t` to 128 bytes. The underlying run-to-run sensitivity (float atomic sums in weight gradients / GN
backward, Phase B2 / B3) remains; `~/ufsm-perf-det` holds an uncommitted, unverified attempt at B1-B3 by the stopped perf agent.

## Spiral-fit heads only: train --task unwrap2, a gauge-free winding mod 4 head (2026-10-18)

User (10-18): drop the recto and ink heads from the winding model; add only heads whose training data follows from our ground truth.
Kept: what `ufsm fit` reads, plus one new head for counting turns.

- Layout (11 outputs; `runtime_extra.unwrap` `"version":2`, `checkpoint_unwrap_version()`): 0-1 phase (0.5 + 0.5 cos / sin 2 pi
  phase, soft BCE, as task unwrap's 2-3), 2-7 n n^T, 8 density (as 4-10: `--normal-*`, `--density-*`, `--packed-*`, count loss),
  9-10 relative winding mod 4: 0.5 + 0.5 cos / sin (2 pi W / 4). Task unwrap (11 heads with recto / ink) is unchanged; predict
  and fit read both. `--cond` / `--pap-head` / ink flags are refused with unwrap2.
- Winding target (src/band.c `band_wind4`, `band_wind4_tiled`). W = k + phase, but the band codes' k is per label set (Paris4 June
  and July unrolls are separate sources with their own numbering; the ~9 um "each"-mode scrolls seed every segment separately and
  their codes are the first set only), so the sampler never reads k: it unwraps the window's stored phase. Voxels join their
  6-neighbours where both phases are known and differ by <= `--wind4-step` (80 of 252 codes); the phase ramps 0 -> 1 between
  consecutive labelled windings and wraps at each recto, so along joins W = phase + integer is continuous. Runs along x, a
  union-find over runs with offsets in turns mod 4 (only W mod 4 is needed; W = phase + 252 (j + O) codes). A component whose joins
  do not close (> 8 + 0.2% inconsistent loops: labels that disagree, or a loop around the umbilicus) is dropped. The phase is read
  before the air test (gaps join the windings; CT 0 does not); w4 = W mod 4 in 63 codes per turn, comp = its gauge id.
  Each component's integer is arbitrary: one gauge per component, never compared with another. So every source contributes the
  same way: wherever its fused stored phase is known and consistent (Paris4 sets, multi-scroll named and each-mode sets), nothing
  from the codes' numbering enters.
- Tiles: on PHerc0139 (named mode, 18.72 um phase) whole windows lost nearly all voxels to inconsistent loops (P384 at 9.4 um, 2x
  upsampled phase: kept 0%; tiles of 64: 80%; on the 18.72 grid tiles of 32 / 64 / 96 kept 90 / 75 / 49%). So the gauge is per tile
  of `--wind4-tile` 64 window voxels (at least P / 6, <= 216 tiles; the largest component of each; up to 255 gauges per window);
  `--wind4-tile 0`: per window (15 largest components). The net learns consistency over a tile (~600 um at 9.6 um, several sheets),
  more than the fitter's offset (`--fit-w4-d` 32 voxels). Warped samples: `warp_w4` pulls w4 as a period-4 phasor within the
  nearest voxel's component (gauges never mix). Cost: ~0.4 s per P384 window on one sampler thread (run union-find, two passes).
- Loss (src/nn_loss.cu `nn_wind4_*`): for component c and gauge g (W + g turns = a quarter-turn rotation of the target),
  bce_g = sum m [sp(a) + sp(b) - 0.5 (a + b) - 0.5 R_g] with R_0 = a cos t + b sin t, R_1 = b cos t - a sin t, R_2 = -R_0, R_3 = -R_1;
  loss = sum_c min_g bce_g / (2 sum m) times `--wind4-lambda` (mask m = the phase loss mask, incl. packed weights and the split halo).
  So per component only N, S0, H, X, Y are summed (plus 4 accuracy counts); the gauge g* = argmax of {X, Y, -X, -Y}; the gradient is
  the soft-BCE gradient against the target rotated by g* (clean: piecewise the phase loss's gradient, no gradient through the
  argmax). Head slabs: pass 0 sums over the slabs, then the gauges; pass 1 the gradient per slab. Logged: wind4 bce, the worst-gauge
  bce (a gauge-blind net has both equal) and the turn accuracy (predicted angle within 1/8 turn mod 4 of the target at g*, i.e. the
  right integer); `val unwrap2:` line per validation. tests/test_wind4: targets on planar / spiral fields (one integer per component,
  <= 1/63 turn), tiles around the umbilicus, loss vs a double reference (1e-6), whole-turn target shifts per component change
  neither loss nor gradient (2.6e-9), gradient vs central differences (5e-8 of 2e-3), fp16 gradients.
- Warm start: `train --task unwrap2 --init-from <task unwrap ckpt>` maps head rows 2-10 -> 0-8 (rows 9-10 fresh), same architecture.
  `ufsm convert-unwrap2 IN OUT` writes that conversion without a training step (header extras kept, unwrap / phase objects
  rewritten). wind96-wk1 vs its conversion, predict on a PHerc0139 box (256 x 512 x 512, window 192 / halo 32, fit-h 2):
  --fit-fields and --fit-zarr pc ps t0..t5 sp w byte-identical (zarr: every shard; only the "checkpoint" attribute differs).
  (A conversion by one lr-0 training step was not: the step changes the weights.)
- Fit fields (predict): unwrap2 checkpoints pool phase / normal / density from channels 0 / 2 / 8 (no recto max) into the same 10
  planes, plus 4: dz dy dx = the predicted W(x + d e_a) - W(x) mod 4 over d = `--fit-w4-d` (32, clamped to the halos) voxels, from
  the cell's material-voxel sum of p(v + d e_a) conj p(v) (p = 2 sigma - 1 of channels 9-10) inside one window (one gauge), byte b
  = b / 64 turns; dq = 255 x the mean over the axes of |sum| / pairs. Per-cell cos / sin planes would not do: each window predicts
  its own gauge, so pooled cells would change gauge at every window seam. fields.json `"w4_d"`; zarr arrays dz dy dx dq beside the
  10 (128^3 field shards like the others). tests/test_fit_w4: the unwrap and unwrap2 pooling of permuted logits byte-identical,
  the relative planes vs a host reference (0 byte difference), save / load round trip.
- Fit term (`ufsm fit`, nn_fit_loss2, `--wind4-weight` default 0 since 2026-10-20 (4 cost fit-val, 0.5-1 neutral); on when the planes exist and the weight > 0, 10-plane fields unchanged): per field voxel and
  axis dq (1 - cos(2 pi / 4 (W(x + d e_a) - W(x) - delta_a))), theta difference wrapped, both points' lattice corners get gradient;
  phase stages only, ramped like the phase; `--analytic 0` ignores it; loss part `wind4` (parts[6]). On test_fit's spiral with the
  phase (and density) under-counting a packed sector by 40%: winding ids 0.876 -> 1.000 with the term at weight >= 3 and d of 1-2
  pitches (d 16 voxels = 0.27 pitch: no gain, a sub-sheet offset cannot tell a missing turn from the phase's own drift).
- Blending (`ufsm blend` of 8 runs): average the 10 planes as now; dz dy dx are angles mod 4 turns: blend as unit vectors weighted
  by dq (atan2 of the weighted sums), dq as the resultant length. Not implemented (cmd_blend is not on this branch's base 7ae7a47).
- Smokes (laptop, PHerc0139 at 9.36 um level 0, phase 18.72 upsampled, holdout box, P128): widths 32,32,64,64,96,128 6k steps from
  scratch (head slabs): val wind4 0.630 -> 0.608, worst gauge ~0.93, turn acc 0.52 -> 0.59 at 1.8k then 0.55 (small holdout);
  phase 0.684 -> 0.654, normal 0.675 -> 0.486. Widths 16,16,32,64,128 (whole-window head and losses) 3k steps: wind4 0.621 -> 0.610,
  turn acc 0.563 -> 0.593, normal 0.759 -> 0.621. Widths 16,16,32,64,128,192,256 P192 --mem fast (the desk layout, small): runs,
  losses fall. Warm start from wind96 (1.5k steps): normal starts at 0.54 (scratch 0.88).
- Open: Paris4 labels are not on the laptop (consistency / kept fraction there unmeasured); the sampler cost of band_wind4 (one
  thread); predict's main output for unwrap2 is channel 0 = phase cos; fit-zarr field shards stay 128^3.
- The desk's 2-GPU wind2-wk1 run (P512, 16..256, --mem fast) died with an illegal memory access (and a non-finite gradient in the
  same step) ~1211 steps after every start; the logged windows were PHerc0172's (the heaviest source, phase-free windows) but the
  windows replayed on the laptop (ASan sampler at P512, GPU at P128 / P256 with compute-sanitizer memcheck) were clean. Cause: a
  device memory leak that only two GPUs on one host thread trigger. The GroupNorm+SiLU backward keeps its fp4 weight-gradient gy
  records (lp_gn_silu_bwd_apply_rec; levels with a multiple of 32 channels) in one thread_local struct and reset the pointers
  without freeing them at every GPU switch, so each GPU lost one record buffer per step (level 2 of this net at P512: 35.6 MB) until
  its card was full; then the records fell back and the unchecked workspace allocations (the records' bias sums, lp_buf, tc_wbuf)
  failed and their kernels wrote through invalid pointers. Earlier 2-GPU runs (prod96, wind96) probably ran with full cards too
  (records off after the first steps: slower, but no workspace growth later).
  Fix: lp_gyrec_select parks each GPU's buffers per device (tests/test_gyrec: 200 switches, no memory lost); the allocations are
  checked. Also: --min-phase (unwrap2 default 0.001) redraws windows without phase (they supervise nothing); UFSM_SYNC_CHECK=1
  counts non-finite logits / gradients per step, prints device memory free per GPU and locates a CUDA fault with its window;
  UFSM_SAMPLE_AT replays given windows; tests/test_unwrap2_sampler (a non-cubic band + phase-only source).
## Task restore: CT super-resolution in `ufsm train --task restore` (2026-10-19)

One residual net restores coarse CT onto the 2.4 um grid (x4): the 9.6 um pyramid rung of 2.4 um scans and native 8.64 / 9.36 um
scans. Replaces the Python orchestration (`tools/restoration*.py` + `ufsm restore-worker`, uncommitted in the main checkout); the
GPU loss and degradation ideas of `src/nn_restore.cu` were kept and rewritten (`src/restore.h`, `src/nn_restore.cu`, `src/restore.c`,
`src/restore_cli.c`).

- **Pairing.** The sampler (`sample_cfg.restore`, `draw_restore`) reads one clean window at the source's min_level (2.4 um / 2.215 um
  scans only) inside a source `"boxes"` entry (the locally present 1024^3 shards) or, for validation, inside the `"holdout"` box;
  tone LUT `255 gain (v / 255)^gamma` (gain, gamma `exp(U(+-0.18))`, 0 kept) and a cube symmetry; `batch.t` = the clean window, no
  host input, no mask (mask = clean > 0). The trainer uploads 1 B per voxel.
- **Degradation on the GPU** (`nn_restore_degrade`, recipe drawn per step from `(seed, step, GPU)`): separable per-axis resampling
  (exact 4x block mean, or a Gaussian PSF sampled at the coarse centres `(i + 0.5 - phase) step - 0.5`), the coarse grid as stored
  (scanner: edge term, tone, correlated signal-dependent noise; uint8 rounding), the volcomp codec round trip on the coarse grid
  (`nn_codec_roundtrip`: 16^3 orthonormal DCT, the encoder's dead zone and the decoder's reconstruction offsets; = volcomp to +-1 DN,
  >99.9% exact, tests/test_restore_task.c), trilinear back onto the fine grid masked by clean > 0. Scratch ~1.5 B per fine voxel.
  Families (`--restore-mix`, default identity 0.1 / pyramid 0.45 / scanner 0.45):
  - *pyramid*: 4x block mean + volcomp **q 2**. The published pyramids store level 0 at q 8, level 1 at q 4, level 2 (9.6 um) at q 2,
    level 3 at q 1 (every volume checked), so the 9.6 um rung is q 2, not q 8. On a PHerc1299 1024^3 box the 4x mean of the stored
    level 0 differs from the stored level 2 by 4.27 DN rmse (one- vs two-stage rounding: 4.273 / 4.276; from the stored level 1:
    4.24): the codec noise of the published levels, which no emulation removes.
  - *scanner*: coarse step U(3.4, 4.4) fine voxels, PSF sigma U(0.45, 0.75) coarse voxels (x step; per-axis factor exp(U(+-0.15))),
    random sub-voxel phase, gain / gamma exp(U(+-0.06)), bias U(+-3) DN, noise U(0, 6) DN, signal variance U(0, 0.02) DN^2/DN, 6-neighbour
    correlation U(0, 0.4), edge term U(-0.1, 0.25) with p 0.3, volcomp **q 8** (native level 0).
- **Calibration** (`ufsm restore-stats`, ~/ufsm-data/sr/calib, calib_native.sh / calib_fit.py): six scrolls with both a native ~9 um
  and a 2.4 um scan (PHerc0009B, 0814, 0841, 0846A, 1203, 1451); shape statistics only (the scans differ in energy and contrast):
  structure function ratios D(k) / D(8) (k = 1, 2, 3, 4, 6 coarse voxels; spectrum and edge width) and noise / sd. Native D1/D8
  0.091-0.108, D2/D8 0.28-0.35, noise / sd 0.04-0.11. Best fits per scroll: PSF 0.7 / 0.6 / 0.6 / 0.5 / 0.7 / 0.5 coarse voxels; the
  noise term is nearly flat over 0-6 DN (the 2.4 um scan's own noise, pooled, already matches; fits 1-6 DN), correlation weakly
  preferred at 0.3. Range used: PSF 0.45-0.75, noise 0-6. (The other session's validation-scanner-v4 / quality-20261018-codec held
  no calibration: RESTORATION.md says the ranges were never calibrated.)
- **Network / loss.** Input `(x - 128) / 64` split exactly into two channels `hi = k / 4`, `lo = (x - 128 - 16 k) / 16`
  (`k = round((x - 128) / 16)`), both exact in MX-fp8 e4m3, so the MX-fp8 stem input (forced `--input-prec 8`) loses nothing (one
  channel would round to 3 mantissa bits, up to +-4 DN); `--restore-input plain` keeps one channel. Residual head `out = input + 64 logit`,
  zero-initialised (`unet_zero_head`: output == input at step 0). Loss over clean > 0: Charbonnier (eps 0.01) + 0.5 edge
  (Charbonnier of face-neighbour error differences) + 0.1 MSE; fp16 logits, separate 16-bit logit gradient (the edge term reads
  neighbours: never in place). Checkpoint `"task":"restore"` with the input layout, normalisation, `fine_um` and the recipe ranges.
- **Validation.** The trainer's holdout windows (PHerc1299: two boxes incl. the sr512-highres-v3 crops; Paris 4 2.4 um: a 2048^3 block),
  each under every family with fixed recipes: PSNR of output and input, gain per family, `restore_val.csv`; `best.ckpt` on the mean
  gain of the degraded families.
- **Predict.** `ufsm predict <restore ckpt> <root> <ct-group> <out> --um U [--level L]`: the CT level upsampled trilinear onto
  `--out-um` (default the checkpoint's fine grid) with the training geometry, window = shard + 2 halo (512 + 64), uint8 CT in 512^3
  shards (volcomp `--q 2` default), optional pyramid. `ufsm restore-eval` scores a checkpoint on .npy crops under one recipe.
- **Data** (~/ufsm-data/sr/make_sources.sh -> sources-sr.json): 20 training scans (2.4 / 2.215 um, 600 level-0 shards hard-linked
  from the old restoration whole-shard mirror into ~/ufsm-data/vol: no new disk), holdouts fetched: PHerc1299 5 shards (67 MB),
  Paris 4 8 shards (129 MB).
- **Window / widths (laptop 5080, `--mem fast`, MX-fp4).** Planner: P384 [32,48,64,96,128,192] lean, 15.4 / 15.6 GB, 75 Mvox/s;
  P512 [16,32,64,96,128,192] lean 2, 14.4 GB, **184 Mvox/s (1.37 steps/s)**; P256 [48,..,192] 32 Mvox/s; P448 [32,48,..] recompute 2,
  71 Mvox/s; P512 with 32 at level 0 and P640 do not fit. The 48-channel levels miss the fp4 kernels (multiples of 32 / 8 / 16 only).
  Chosen: **P512, 16,32,64,96,128,192**, the largest and fastest.
- **MX-fp4 vs MX-fp8 activations** (2000 steps, P384 16..192, same seed; fp8 does not fit P512): mean gain +1.555 vs +1.623 dB
  (pyramid +2.191 / +2.274, scanner +0.919 / +0.971), 177 vs 94 Mvox/s: within 0.1 dB, so fp4.
- **500-step smoke** (chosen config): validation gain pyramid +1.31 / 1.40 / 1.42 / 1.50 / 1.60 dB, scanner +0.66 / 0.67 / 0.68 /
  0.71 / 0.73 dB at steps 100..500. Identity output PSNR ~34 dB early (the net still perturbs sharp input by ~5 DN rmse; 55-57 dB by
  step 2000 in the A/B runs).
- **Against sr512-highres-v3** (its frozen crops, 4x block mean without codec; restore-eval of the 2000-step P384 fp4 A/B checkpoint):
  output MSE crop 0 111.9 / crop 2 53.0 (PHerc1299 2.4 um) vs 155.7 / 80.9 for the old best (x4_nearest, step 8400): +1.4 / +1.8 dB
  absolute; the Paris 4 1.129 um crops 1 / 3: 287 / 251 vs 394 / 365. Gain over the trilinear input +2.31 dB overall (+2.03 with q 2).
- **Long run**: ~/ufsm-data/sr/sr100k.sh (100k steps, Muon 0.01 / AdamW 1e-3, WSD cooldown 0.2, warmup 500, EMA 0.999, ckpt / val
  every 2000, restart-safe `--resume last.ckpt`): ~20.5 h at 1.37 steps/s.

## The two biggest level-0 kernels: conv_fwd_f4p_k and conv_bwd_w_f4ws_k (2026-10-16)
Desk P512 trainer (P2's profile): conv_fwd_f4p_k (16-channel forward and backward-data) ~149 ms / step over 17 calls,
conv_bwd_w_f4ws_k (fp4 weight gradient, warp-specialised) ~114 ms over 66, of ~597 ms of kernels.
- Floors, 16 -> 16 at P512 (134 M voxels, 9 B / voxel in and out): DRAM 2.4 GB = 5.4 ms at 448 GB/s; MMA 1.24 T MAC as issued
  (9 tap rows of K 64, the kx3 quarter zero) = ~6.5 ms at the dense fp4 rate (0.93 T MAC useful). Measured (bench_f4conv, desk
  GPU 1): fwd 20.5 ms, bdata 18.3 ms before this work -- ~3.5x either floor; ncu: DRAM ~26% of peak, tensor pipe ~29-36%, 0.6
  issue / SMSP / cycle, 2 blocks / SM (registers 128 and shared memory both limit), stalls mostly fixed-latency waits.
- f4p phase split (F4R_PROF, thread 0 of a block): fwd staging 38% / MMA 24% / epilogue 35%; bdata 22 / 30 / 46; the 16 + 16
  split output 14 / 29 / 55. Instructions (ncu source counters, fwd): staging 41%, epilogue 25% (GN statistics ~9%, the SR
  quantisation of bdata outputs ~28% of bdata's), MMA loop 12% at only 24% OMMA density (the rest ring-slot % 6, addresses,
  scale loads).
- Done (conv_fwd_f4p_k, every output hash unchanged): the ring slots once per tile and the tap-row loop fully unrolled for
  MT <= 2 (kz constant: OMMA density 24 -> 38%); MT 1 loads a tile's rows right after the previous tile's staging (they
  arrive during its MMA and epilogue; MT 2 spilled); GN pre-halved (fmaf(v, a / 2, b / 2), same bits); the staging amax as one
  FMNMX.NAN per value; MT 1 scale words interleaved for one 8-byte load per row; MT > 1 GN pairs in shared memory. fwd 16 -> 16
  20.5 -> 19.0 ms (-7.3%), bdata 16 -> 16 18.3 -> 17.0 ms (-7.0%), bdata 16 -> 16 + 16 32.3 -> 31.2 ms (-3.3%).
- Tried and dropped: an 8-slot (power-of-two) ring for MT 1 (+0.8%: the extra shared memory comes out of L1); TZ 2 (+15%).
- Then z chunks of up to 64 tiles while >= 2048 blocks remain (was 8: a block's first tile stages 2 extra planes and loads the
  weights): P512 fwd 18.1 ms (-11.9% vs the start), bdata 16.4 (-10.3%), the 16 + 16 split 30.5 (-5.7%); P256 -8 to -10%.
- Slab-local GroupNorm (--l0-norm slab with the kernel path): both kernels take per-slab statistics -- conv_fwd_f4p_k MT 1 reloads its
  (pre-halved) register pairs when a staged position lies in another slab (gn_row / gn_ab), MT > 1 with a GN input routes around
  it (its pairs live in shared memory, shared by threads in different slabs); conv_bwd_w_f4ws_k's LY 1 lanes take chan_ab of
  their plane.
- conv_bwd_w_f4ws_k is producer-bound (ncu: the consumers' first load after the FULL barrier holds 23% of the samples), at the
  80-register cap of 672 threads. Tried, all reverted: the FMNMX.NAN amax in the LY 1 lanes (+0.4%), one cursor advance per pass
  (spills 16 B), the MX-fp4 decode through a bf16x2 byte table + one fma.rn.bf16x2 per pair with 8-byte LY 1 loads (+6%, a 16 B
  spill) -- every change that moves its register allocation lost more than its instructions saved.
- Left: the epilogue (fwd_epi_mx16, lp_common.cuh: 35-55% of f4p) -- GN statistics summed per lane in registers instead of the
  tb re-read, a cheaper exact SR nibble; the f4p staging index math (li / 180 etc., ~8% of its instructions); f4ws needs a
  producer restructure that frees registers first.
- Follow-up (P6, desk GPU 1, every bench_f4conv output hash unchanged at P384; |gw| / statistics differ only in the 9th digit,
  as between two base runs):
  - conv_bwd_w_f4ws_k (16 -> 16, P512, one record slab as the trainer): ncu showed it issue-bound (IPC 2.47 of 4, 5.07e9 warp
    instructions; even at IPC 4 the instructions alone are 11.4 ms), the producers ~80% of them, ~40% of theirs per-pass index
    math (load addresses, record rows, cursor copies, LY 1 lane setup). Hoisted to per-thread / per-item constants, the three
    7-int cursors replaced by pass counters (pass p of an item: planes 2 ztb - 1 + 2 p, z-step max(p - 1, 0)), the ring slot
    as & 7, the LY 1 amax as one FMNMX.NAN.XORSIGN (amax_xs: max(|a|, |v|); mx_exp ignores the sign; amax_nan's fabs had become
    a separate FADD): 5.07e9 -> 4.42e9 instructions, 20.6 -> 19.5 ms (-5.5%), still 80 registers, no stack (base: 8 B).
  - Its time split (diagnostic builds): with the x row loads and the gy record copies off 15.1 ms, with the loads alone off
    16.3 ms -- the DRAM latency of the rows, loaded one pass ahead, is ~3 ms. Tried and reverted: rows two passes ahead (a
    second register set: spills, 22.8 ms), the gy records one z-step ahead in a third buffer (cursor registers: spills, 20.3
    ms), 15 producer warps (24 warps cost the 80-register allocation of 21; IPC fell, 19.8 ms). Every deeper prefetch needs the
    registers a producer restructure would free (decode and LY 1 lanes on different warps, or the raw rows staged by cp.async
    into shared memory: MT 1 has ~5 KB left).
  - conv_fwd_f4p_k staging: 16-channel inputs (block-uniform Ci >= 16) without the k < Ci masks (32 instructions per position),
    the amax as amax_xs: fwd 16 -> 16 18.45 -> 18.1 ms (-1.9%).
  - fwd_epi_mx16's exact SR (the bdata outputs): sr_e2m1_nibf rounds on the float bits -- one mantissa bit kept, the uniform
    added at bits 6..21 of the 22 dropped ones, [0, 1) shifted to [1, 2) by + 1: up exactly when frac + u >= 1, as
    sr_e2m1_nib's floor(t + u) (which rounds t + u in fp32), without the float sum and conversion (~3 instructions per value);
    NaN blocks keep sr_e2m1_nib (its 0 keeps the NaN under the 0xFF scale). bdata 16 -> 16 16.8 -> 16.55 ms, 16 + 16 31.9 ->
    31.4 ms; output hashes unchanged, test_grad_mx4's bdata 16 -> 16 mean of 16 0.0404 (as before).
  - Tried and reverted: the epilogue's zero-bias FADDs skipped and its row index hoisted (+0.5-1%: register allocation again).
  - prof_infer P512 (trainer-like: lean 2, three level-0 buffers, interleaved x 3): step min 577.7 -> 568.8 ms (-1.5%).

## Many surface sets: winding ids, merged codes, side votes (2026-10-16)
Goal (user): train eventually on every published surface of every scroll. Surfaces of different authors / runs overlap (the same
wrap traced twice, a voxel or two apart) and carry no common winding numbering, while side labels need the order of consecutive
wraps (k .. k + 0.5 recto, k + 0.5 .. k + 1 verso). Three layers, each its own command, all lossless volcomp on one grid:
1. `ufsm wind-assign <ref-root> <ref-key> <out> <tifxyz>...` (src/wind.h): every surface gets q = K + dir theta / 2 pi per vertex
   in the gauge of a reference winding raster (the published band codes / any raster --value codes, or a dense phase-raster
   --winding raster): theta around the umbilicus unwrapped along the mesh grid, one K (mod 14) and dir per connected component,
   voted by the samples within --tol label voxels of a reference voxel; the dir whose agreeing votes sit nearer integer turns wins
   (the wrong dir's residuals are uniform). Verdicts per component: duplicate (re-traces the reference), new (agrees where it
   overlaps), conflict (< 80% of matches agree: a sheet switch; the disagreeing step x step cells are dropped), unassigned
   (nothing anchors it: needs a dense winding reference, e.g. a fit, or another set first). --drop-duplicates keeps only new
   coverage (the reference wins where both exist). Output: <out>/<name> tifxyz (--out-um: coordinates on the reference grid) +
   q.tif, assign.jsonl.
   Paris4 check (laptop codes box, 9 of our 19.2 um fitted wraps from slab-w-p19v2-w640): every component dir +1; 35% of their
   area duplicate (86-100% of matches agree, median 1.6-2.4 label voxels from published sheets, K = the wrap's own winding mod
   14: the fitter's gauge is the codes'), 65% conflict (41-76% agree: the fitted wraps switch between neighbouring windings,
   as wrapswitch reports).
2. `ufsm raster OUT --value q.tif --binary 0 --levels 1 <out>/*` writes one codes raster per surface set (codes of surfaces that
   agree within 4 steps (1/4 turn) are the same wrap, otherwise 255); `ufsm codes-merge <out-group> <um> <root> <key> ...` unites
   such rasters with the published codes by the same rule. phase-raster --pair 2 of the merged codes labels between wraps from any
   of the sets (two copies of one wrap a voxel or two apart carry the same code: band_field_side_pair treats them as one sheet, codes
   within 5 steps). Paris4: 3 assigned fitted wraps rasterised (17 s) and merged into the published codes box: 5.16 G coded voxels,
   0.03% conflicts.
3. Side votes (src/votes.h): where sets disagree, or to keep each set's own ordering, every set's side labels (phase-raster --pair 2
   of its own codes) are ADDED: `ufsm side-votes <out-group> <um> <root> <key> ...` stores per voxel (r recto votes, v verso
   votes), 0..15 each, in a diagonal code ((0,0) 0 | (0,1) 1, (1,0) 2 | (0,2) 3 ...), so a one-set side array IS its vote array
   byte for byte; `downsample --mode votes` adds the votes of each 2^3 block. Sources: "encoding": "side-votes" with "votes":
   {"rule": "unanimous" (default; any disagreement ignored) | "majority", "min": N (votes a label needs), "soft": 1 (recto-side
   papyrus r / (r + v): soft targets, not packed)}; hard rules decode on read, so warps, packed targets and the rest of the side
   path are unchanged. Codec (bench_votes, 4 Paris4 cubes of 256^3 at 4.8 um, the labels + two displaced partial copies, 3.85% of
   labelled voxels disputed): one set 0.151 bits / voxel, votes in the diagonal code 0.343, as nibbles (r << 4 | v) 0.469 (+37%:
   recto and verso 15 apart cost bypass bits at every face), decode ~800 MB/s; no volcomp change needed.
What real multi-author data needs: tifxyz (or .sfc) per surface in a known scan's voxel frame (voxel_size_um, which scan / volume
id), the scroll's umbilicus, provenance (author / run) to form sets; a reference gauge per scroll: the published band codes where
they exist, else the largest consistent set or our fit's dense winding raster (unassigned surfaces wait for it); per-vertex q from
authors that publish it can replace wind-assign's (then wind-assign is the consistency check). Open: surfaces of another scan of
the same scroll must be registered onto the reference scan first (surfcomp re-registrations exist for Paris4's 81 segments).
## Training window P768 vs P512 on the same validation voxels: train --val-P N (2026-10-16)
`train --val-P N` (N <= P, a multiple of 2^(nlev-1)) validates on N^3 windows whatever the training window: an inference-only
copy of the network on GPU 0 (its own N^3 build; the planner keeps its bytes free) reads the training net's current weights
(`unet_fwd_params`) and scores the windows through GPU 0's batch buffers. The startup line prints a hash of the validation
windows' bytes: P128, P256 `--val-P 128` and P128 through the copy (`UFSM_VALP_FORCE=1`) print the same hash; the copy's loss
equals the plain pass to 2e-4 (the inference build's arithmetic). The copy takes ~19 B per N^3 voxel (1.08 GB at 384^3), so a
P768 trainer (13.8 GB) fits a 16 GB card with `--val-P 384`, not 512. Plain cube windows and losses only.

A/B on the desk (GPU 0 alone, production recipe: sources-rv24 2.4 um, 7 levels 16..256, Muon, --det 1, seed 1, --aug none,
all three scored on the same 8 windows of 384^3, `--val-P 384 --val-seed 1`; EMA-weight validation loss):

| arm | P | steps | wall s | Mvox/s | GPU 0 peak | val 500 | 1000 | 1500 | 2000 | end |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 512 | 2000 | 1251 | 219 | 5.99 GB | .69053 | .67182 | .65939 | .65242 | .65242 |
| B | 768 | 2000 | 4172 | 222 | 15.52 GB | .69247 | .66986 | .65530 | .64717 | .64717 |
| C | 512 | 6670 (B's wall clock) | 4099 | 220 | 5.99 GB | .69521 | .67113 | .65919 | .65255 | .63354 |

- Matched steps (B vs A at 2000): P768 better by 0.005 with 3.4x the voxels per step (about the run-to-run noise, ~0.003).
- Matched compute (B vs C, ~4100 s, ~900 G voxels each): P512 for 3.3x the steps better by **0.014**. At this length the
  window is not what limits the side task: more optimizer steps beat more voxels per step (windows of one scroll region are
  highly correlated, so a larger window acts like a larger batch). The receptive field (~1000 voxels) exceeds both windows.
- Caveats: one seed; evaluation on 384^3 windows (a P768-trained net's use of longer context at inference windows is not
  measured); 2000-step runs. Same throughput (~220 Mvox/s) either way.

## Level-0 kernels outside the 16-channel convs; why test_l3_sched broke under a fused GN-backward sum (2026-10-16)
Desk 5060 Ti, P512 production net (`tests/bench_l0misc`: per-call times and output hashes; every change below is bit for bit):
| kernel | before | after | change |
|---|---|---|---|
| dec[0] conv1, fused up (`conv_fwd_f4u_k`) | 40.2 ms | 37.2 ms | stage only the coarse rows the blend reads |
| network input (`input_rad_k`), plain / warped record | 5.0 / 8.0 ms | 3.6 / 7.4 ms | 3-D grid: no 64-bit divisions per voxel |
| loss statistics (`loss_stats_k`) | 5.9 ms | 2.9 ms | four voxels' loads first, then each in the thread's order |
| loss gradient (`loss_grad_k`) | 8.0 ms | 5.7 ms | 2-D grid (n c, voxel blocks), four voxels per thread, loads first |
Per trainer step: dec0.c1 once, the input twice (forward, the stem's weight gradient), the loss once. Real trainer (sources-rv24,
P512, `--aug none`, two interleaved runs each, steps 60-140): 1.61-1.62 -> 1.63-1.65 windows/s (~619 -> ~607 ms per step).
- `conv_fwd_f4u_k` staged 6 coarse planes x [8 rows][12 columns] per z tile through GN+SiLU; the blend reads at most 3-5 planes x
  6 x 10 (`up_read_range`: the in-bounds fine coordinates and their trilinear neighbours). Its coarse stage is still ~25-30% of the
  kernel: latency (2 blocks of 256 threads per SM at the 128-register cap), not work; prefetching the next tile's coarse rows into
  registers during the MMA changed nothing (the phases are serialised per block). A warp-specialised producer / consumer split is
  the next step there.
- The loss and input kernels were issue-bound on index arithmetic (64-bit `/` and `%` per element) or latency-bound (one dependent
  byte / half load per loop iteration); the arithmetic of every element and the order of every partial sum are unchanged.

test_l3_sched failed intermittently (~50% of runs, always the same wrong enc0 / down0 gradients) when the GroupNorm backward sums of
dec[0]'s GN1 were fused into conv2's backward-data epilogue (worker P2's experiment, not merged). Not a race in the three-buffer
schedule: the wrong result appears with the copy schedule off, with `CUDA_LAUNCH_BLOCKING=1` and every offload copy synchronous,
under memcheck / initcheck (clean), and only dec[0]'s d/d a1 after GN1 differs between good and bad runs. The fused epilogue added
its warps' float partials with shared-memory `atomicAdd` (`fwd_epi_mx16`'s `cs[]`): the heavily cancelled GN-backward sums varied
~1e-5 relative run to run, a GN coefficient occasionally landed one float ulp off, and the MX-fp4 SR chain through the coarse levels
amplifies one changed fp4 code into O(SR noise) gradients: with the unfused (deterministic) sums, nudging one dec[0] GN1 gamma by one
ulp moves enc0's gradient L1 norm by 2-15% at P128. Adding the warps' partials in warp order made the sums identical across runs and
the test pass 30 / 30. Rule: anything feeding a level-0 GroupNorm coefficient must reduce in a fixed order (fp64 atomics of fixed
per-block partials are fine: their order noise stays below a float ulp); the bit-for-bit schedule tests catch any nondeterminism
upstream of the coarse levels. The forward GN statistics in the same epilogue (float shared atomics) and the fp32 weight-gradient
atomics are order-dependent too: the likely reason `--det 1` runs of one recipe drift apart after ~40 steps (workers F / H; not
traced further).

## Sheet-aware crush augmentation: train --aug-crush P (2026-10-24)
Goal: teach the winding nets compressed / crushed scroll regions (PHerc1299-like) from Paris 4's clean labels. src/crush.[ch], used by
sample.c `crush_draw` on stored-phase windows (task unwrap / unwrap2, side 2, no target channels, no band/cond/sheet/papyrus/aux, cube
windows); default p 0 = bit-identical (test). A crushed window replaces the plain read and the geometry warp (it carries its own rotation);
`--aug-crush-warp 1` (2026-10-26) applies the ordinary geometry warp after the crush map as well (labels through both, tested).
- Map u -> x (window voxel -> source voxel), C1 and invertible, det = k F' > 0:
  A. squash 1.5-3x (p .5) along a random direction IN the sheet plane (squashing along the normal thinned the sheets), + rotation 10 deg;
  B. bends (p .5, amp .06) and a logcosh kink (p .25) along the sheet normal n (normal from the stored phase gradient over the window);
  C. gap collapse along n: per column (quadratic B-spline nodes, h = max(12, P/20)) sigma = F(s), F = g^-1, g' = rho =
     1 - (1 - j) v, j ~ U(.15, .5) in void, 1 in papyrus; voidness v = soft Otsu-midpoint ramp of the blurred CT read from level l+1;
     F a monotone Hermite. Closed gaps blend the CT; the phase is pulled as a circular value (CRUSH_PHASE, unknown kept),
     so labels of adjacent wraps stay separate.
- Normals of targets come from the pulled phase (= J^T grad W, tested); unwrap2's n n^T, density, wind4 are derived after the pull.
- Cost: map on a 2-voxel lattice (trilinear between, cell dets checked > 0), 128^3 tile LRU in the sampler's arena (8 P^3), Morton
  block order, polynomial atan2. P512 Paris 4 level 2 on one laptop core: 15-29 s per crushed window (561-2227 tiles, ~50% decode);
  P256: 6-17 s. At p .5 the desk loader (24 workers, ~28 core-s slack per sample) absorbs ~8-15 core-s per sample.
- `--aug-crush-fine N` / sources `"crush_fine": N`: CT box-mean of 8 sub-pulls from N levels finer (e.g. the 2.4 um level 0/1 for a
  9.6 um window), default 0. Too slow for the loader (P256 N=2: 40-184 s, 10-58k tiles) and the desk holds only levels 2-9 of the
  Paris 4 2.4 um zarr; 1.1 um would need its nonlinear registration (not done).
- Not in dino-train (its loader has no stored phase for the sheet normal).
- Tests tests/test_crush.c (analytic spiral: invertibility, det vs FD, C1, footprint, void vs papyrus compression, normals; fixture
  windows: phase = frac(W(pull)), CT agreement per class, blocked == point pull, lattice error; sampler: default-off bit identity,
  p 1 crushes deterministically). Renders: build/crush_render (4 panels: CT | phase | crushed CT | crushed phase).
- Animations: build/crush_anim (`crush_map_scaled`: squash k^t, bends / kink x t, sigma = s' + t (F - s'), each frame invertible,
  tested) sweeps t 0 -> 1 -> 0 over one window, slice across the sheets, CT | phase hue, own LZW GIF89a writer (64 grays + 16 x 12
  hue / brightness palette, looping, 12 fps). ~10-15 s for 48 frames at P 384 (point pulls). Examples ~/ufsm-data/crush/anim.

## Cross-scan CT swap: train --aug-xscan P (2026-10-24)
Goal: scanner / energy / resolution robustness from one set of labels: replace a window's CT by a registered partner scan of the same
scroll warped into the label frame (Paris 4: 78 keV 2.4 um labels; partners 137 keV 2.4 um and 78 keV 1.1 um). Labels unchanged.
- sources.json `"xscan": [{"map": "<ufsm reg map>"} ...]` (up to 4; plain strings accepted). The map's coarse scan is the source's
  scan (`sview_open` checks the um within 2%); map JSON + field.f32 as written by `ufsm reg` (fine(c) = A^-1 (c + D(c))).
- `sview_read_level` (src/view.c): the partner level nearest the window's voxel (rvol_level_for), box mean of 2^3 sub-samples when that
  level is < 0.75 x finer, z slabs of <= 48M fine voxels; a block entirely outside the partner reads 0.
- sample.c `xscan_draw`: random partner; needs >= 50% of the window's scan voxels covered, else the window is kept; quantile LUT
  partner -> window values over voxels where both are nonzero (keeps the source's intensity scale for norm / iso); partner 0 -> CT 0 ->
  labels unknown through the existing air tests. Applied after the min_fg check; not for holdout, validation, crush, chained, view
  or cover windows. Default p 0 = bit-identical (test). Counts per run: train prints the aug counters at the end.
- Tests tests/test_scrollaug.c (fixture spiral, translation maps: bit-exact read, 0 outside, default-off identity, p 1 swaps all
  windows deterministically, a shifted partner keeps >= 99% of the known labels). Renders build/xscan_render (CT | each partner,
  phase hue below); ~/ufsm-data/scrollaug/xscan/h-00.jpg: both partners line up with the labels.
- Cost (laptop, one core, warm disk cache, Paris 4 level 2 P512): 137 keV partner 3.5 s, 1.1 um (its level 3, 9.03 um) 6.8 s per
  swapped window; ~60% volcomp decode, ~20% trilinear resample + field. Expected loader cost ~= P x 5 s worker time per window.
- Desk: scripts/xscan/desk-maps.sh copies the maps with local roots (/vesuvius/ufsm/volcomp coarse, /vesuvius/ufsm/u0/vol partners:
  137 keV has level 2, 1.1 um has level 3). The 137 keV partner covers L0 z 42250-48750 only (level 2 z 10562-12187).
- Animation: build/scrollaug_anim ... xscan (cross-fade into each partner and back, labels fixed); ~/ufsm-data/scrollaug/anim/xscan.gif.

## Tears / slips: train --aug-tear P [--aug-tear-um 480] (2026-10-24)
Goal: the fitter's slip failure: sheets cut and offset by about a turn, where the CT looks nearly continuous but the windings jump.
- Same windows as --aug-crush (stored phase, side 2, no target channels, cube, no zoom-out; exclusive with crush, xscan skips torn windows).
- src/scrollaug.[ch]: `sa_phase_normal` (structure tensor of the circular phase gradient: normal + turn spacing from the median
  gradient), `sa_tear_make` (cut normal within 30 deg of perpendicular to the sheets, plane + 3 sines (wavelength P/2 .. 2P, amplitude
  <= P/24 each); shift 0.3 .. 1.3 turns along the normal, half the time also -1 .. 1 turns along the sheet, capped at tear_um; each side
  shifted by half, integer voxels: exact copies, no resampling; seam none / void (1-3 voxels, CT = 5% quantile, labels unknown) /
  blend (2-6 voxels, CT mixed, labels unknown), 1/3 each), `sa_tear_apply`.
- sample.c `tear_draw`: a padded read (margin tear_um / 2 + 2), the labels prepared on the padded box as the stored-phase path does
  (sign, ignore mask, air, band_wind4_tiled with <= 6^3 tiles) and pulled with the cut, so the winding mod 4 keeps each side's true
  windings (it jumps across the cut); g_crush carries the phase / w4 / component into the ordinary label path (then the usual warp).
- Tests (tests/test_scrollaug.c): normal + spacing on the fixture spiral (11.9 vs 12 voxels, |n . radial| 0.997); apply is bit-exact
  per side, seams void / blended with unknown labels; default off bit-identical; p 1 deterministic; neighbour w4 steps agree with the
  phase steps mod 1 turn (100%), >= 0.3 turn w4 steps only in torn windows (1652 vs 15).
- Paris 4 at 9.6 um: spacing 15.7 voxels / turn (151 um); drawn shifts 16-22 voxels. Cost P512 one core: +3 s per torn window (padded
  read + padded unwrap; plain window ~7.4 s on the local box). Memory: 4 (P + 2M)^3 buffers transient (~0.7 GB at P512).
- Renders: build/aug_render (sampler, plain vs augmented rows), build/scrollaug_anim ... tear (the shift swept 0 -> 1 -> 0);
  ~/ufsm-data/scrollaug/anim/tear-{none,void,blend}.gif.

## Delamination and holes: train --aug-delam P [--aug-delam-hole 0.5] (2026-10-24)
Goal: sheets that split into layers (a void opening inside a sheet, the outer layer pushed out) and missing papyrus (holes, ash).
- Same windows as --aug-tear (stored phase, side 2, cube); exclusive with crush and tear; xscan skips damaged windows.
- src/scrollaug.[ch]: `sa_delam_make` draws 1-3 sites on papyrus (CT >= iso, known phase), each with its own normal from
  `sa_phase_normal_box` over +-2 turns (oriented so the phase grows along it); peel or hole (p --aug-delam-hole).
  - Peel: a lens b = t g0 (1 - r^2)^2 (g0 2 .. min(6, .4 spacing) voxels, R 1-4 turns), tapered off the plane between |a| .4 and
    .6 turns; the two halves of the sheet move apart by up to b / 2 each, within their own half turn. Distances come from a per-site
    profile (geometric distance along the normal through c of each phase offset 0 .. 126 codes, ahead and behind): the stored phase is
    far from linear in space (papyrus spans about half a turn over a few voxels), and measuring the gap in phase units made it half as
    thick as the pull, so the pull refilled it (a dotted 1-voxel line on Paris 4). Distance d < b / 2 from the split is the gap (void CT,
    phase phc); beyond, the CT is pulled by (b / 2) w(d) away from the split (w linear from 1 at the gap edge to 0 at the half-turn
    point; each half moves at most half way there). The phase is the profile inverted at d - (b / 2) w, a monotone function of the
    phase itself, so the unwrap gains no turn (resampling the phase at the pulled point gave nearest-neighbour jitter that dropped
    whole w4 tiles as conflicts on Paris 4). Voxels of unknown phase stay unknown. On Paris 4 the gap is a 3-4 voxel lens tapering
    along the sheet; the scattered unlabelled specks in the renders are the stored phase's own unknown voxels (present at t 0).
  - Hole: a ragged ellipse (R, R2 .5-2.5 turns, 3 harmonics) in the sheet plane, restricted to the centre's sheet (|a| <= .4 turn and
    within .22 turn of the sheet's CT-weighted mean phase on the normal); papyrus -> void + ash (0-.2) x speckle x (CT - void), soft
    rims; labels unchanged (the net infers the missing sheet).
- sample.c `delam_draw`: padded read (margin 8), labels prepared on the padded box (sign, ignore, air), the damage applied, then the
  ordinary unwrap of the damaged window (continuous, so w4 is smooth inside components).
- Tests (tests/test_scrollaug.c): t 0 identity; the peel gap is void at the split phase; 16 peels on the fixture unwrap with 0
  conflict voxels (w4 kept 1.0000 of the undamaged box); holes only darken the CT, labels unchanged; default off bit-identical, p 1
  deterministic; damaged windows' w4 smooth inside components.
- Cost P512 one core (4 fixed Paris 4 windows): no net cost (ctread +1.6 s for the padded read, augment -1.8 s: 116.6 vs 132.1 s for
  24 windows). Render: build/scrollaug_anim ... delam (strength swept 0 -> 1 -> 0, w4 component borders drawn);
  ~/ufsm-data/scrollaug/anim/delam.gif.

## Inclusions: train --aug-inclusions P (2026-10-24)
Goal: the net looks through mineral grains, bubbles and debris (other scrolls have more of them; the labels are of the sheets).
- Any augmented training window (no phase needed), after density; CT only, labels unchanged.
- src/scrollaug.[ch] `sa_incl_make` / `sa_incl_apply`: per 128^3 a uniform 0-6 grains, 0-4 bubbles, 0-3 debris (a fraction of an item
  is the chance of one), each placed on a voxel of its kind: grains anywhere in the scan (rough ellipsoids 0.8-3 voxels, CT 190-255,
  never darkening, a thin dark phase-contrast fringe out to 1.8 radii), bubbles on papyrus (1.5-5 voxels, void inside, a faint bright
  rim), debris in gaps (elongated rough clumps 2-8 voxels, iso + 10-70% of the range above it, speckled). Soft ~1 voxel edges; CT 0
  kept.
- Tests: t 0 identity, CT 0 kept; all three kinds, bubbles on papyrus and debris in gaps; local (0.2% of the voxels at 4x the count),
  brighter and darker voxels; default off, p 1 deterministic.
- Cost P512 one core: +0.3 s per window. Render: scrollaug_anim ... incl [--dens 1]; ~/ufsm-data/scrollaug/anim/incl.gif (--dens 5,
  stills incl-t0/t1(-crop).jpg).

## Curvature: train --aug-curv P (2026-10-24)
Goal: the tight inner core and near-flat outer wraps from any wrap's windows (the training boxes see few radii).
- Same windows as --aug-crush / tear / delam (stored phase, side 2, cube, no target channels); exclusive with them; xscan skips bent
  windows; the ordinary spatial warp still follows.
- src/crush.[ch] stage R (`crush_map_rebend`), pulled through the crush tile cache (`crush_window`: CT trilinear, phase as the
  phasor mean, ignore mask nearest), so the footprint can be wider than the window at no RAM cost. In the plane normal to the axis A
  an output point at radius rho' and angle th' about the moved axis (c - r1 U) pulls from radius rho' - r1 + r0 and angle th' r1 / r0
  about the source centre (c - r0 U); the coordinate along A is kept. Radial spacing (the wrap pitch) and arc length along the centre
  sheet are kept, the curvature there goes 1 / r0 -> 1 / r1; smooth and invertible for |th'| < pi.
- sample.c `curv_map` / `curv_draw`: r0 the distance to the source axis (its tilt from +-4 windows: hand-placed control points
  wiggle ~30 deg over one window on Paris 4), U the window's phase normal oriented outward (the axis direction alone was ~30 deg off
  the sheet normal on Paris 4 and turned the rings into spirals); skipped when the axis is within 0.75 P. Without an axis the sheets
  are taken as flat (r0 = inf, U the phase normal, either sign). Tighter 2 in 3: r1 log-uniform 0.6 P .. 0.9 r0 (flat: .. 8 P);
  else flatter: r0 .. 4 r0. Below r1 0.6 P the whole core of rings maps onto a short source arc and any off-centre sheet becomes a
  spiral into the axis (tried 0.25 P on Paris 4). The seam behind an output axis inside the window (|th'| > 0.85 pi): CT 0, phase
  unknown. The radial input channels follow the moved axis (g_crush.cax overrides cyz / cxz).
- Tests (tests/test_scrollaug.c): r1 = r0 identity, the centre fixed; rings about the moved axis -> rings about the source axis with
  the same offset (flat: planes), arc length kept, det > 0 off the seam, the radial line through the centre unmoved, the seam only
  behind the axis; default off, p 1 deterministic, w4 smooth inside components (P 32 windows: the fixture's scroll is small).
- Cost P512 one core (4 Paris 4 windows, p 1): ctread 0.95 -> 5.6 s per window (tile decodes over the footprint, the pull); --aug-crush
  is 20 s there. Render: scrollaug_anim ... curv [--tight 3 | 0 flattening]; ~/ufsm-data/scrollaug/anim/curv.gif, curv-flat.gif.

## Sheet density: train --aug-sheet-density P (2026-10-24)
Goal: wraps of different brightness (carbonisation, compression, papyrus quality vary per wrap) and slow along-sheet variation.
- Windows with a stored phase (level shift 1, or 0 direct), not zoomed out, augment on; applied after crush / tear / delam / xscan
  (it re-lights their output, from their pulled phase) and before the CT statistics. CT only: labels unchanged.
- src/scrollaug.[ch]: `sa_density_make` takes the window's mean sheet normal (`sa_phase_normal`), void (5% CT quantile), iso and
  the reference phase phc (circular mean on papyrus), and the offset c0 so that kc = a / spacing - psi - c0 (a: distance along the
  normal, psi: stored phase in turns) is an integer on the wrap centres. Per wrap k a hashed gain (+-0.10..0.35) and offset
  (+-0..0.15 of iso - void); flat within +-.2 of each integer, smoothstep between (no step inside papyrus). Slow field: 3 plane
  waves of wavelength .7-2.5 window, amplitude 0..0.3 total. out = void + (CT - void) gain slow + offset m (m: papyrus fraction of
  CT between void and iso); CT 0 (masked air) kept. Unknown-phase voxels take the nearest known kc along the row within 6 voxels.
- sample.c `density_draw`: the window's phase is read once into g_crush.ph (pre 0, so the conflict mask still applies) and the
  label path reuses it.
- Tests: wrap spread between wraps .105 vs within .010 on the fixture; t 0 identity; CT 0 kept; default off bit-identical, p 1
  deterministic.
- Cost P512 one core: apply ~0.8 s + make 0.08 s per window (the phase read moves forward, not added); ~+12% on a ~7 s window.
  Render: scrollaug_anim ... density; ~/ufsm-data/scrollaug/anim/density.gif.

## Augmentation presets: train --aug all (default) | none (2026-10-16)
User: "I want all the augs we can get" (robustness to other scrolls, scanners, resolutions and damage from one scroll's labels).
- `--aug all` is the default for the dense-target tasks on cube windows (recto / side targets); `none` (the old defaults: 48 cube
  symmetries + intensity jitter) for papyrus, sheet, band, unwrap, condition, aux inputs and slabs, which have their own recipes.
  The preset only sets fields (sample.c `sample_aug_preset`); every flag still overrides them; `--aug-p-scale X` scales all of the
  preset's probabilities (the A/B dial); the runtime metadata records `"aug":{"preset":...}`. Resume a run started before the
  presets with `--aug none` to keep its data.
- all: the scanner ops of "Generalising to other scrolls" at strength 1 with p seam .05, air .05, lowres .25, blur .3, sharpen .05,
  bias .15, ring .1, cupping .1, slice .1, noise .15, cnoise .1, tone .25, quant .1, dropslice .05, cutout .1, chunk .05,
  paganin .15, gin .1, motion .1 (+ hist .15 when `--aug-hist FILE` is given); cutouts and dropped slices keep their labels
  (`drop_ignore 0`: the net infers occluded papyrus from context; masked air and chunk dropouts stay unsupervised through CT 0);
  rotation +/-10 deg p .25, elastic p .15, zoom in to 0.8 p .2, zoom out to 1.25 p .15, affine p .1 (aniso .1, shear .06); axis
  jitter 800 native voxels p .1 (<= 2 deg; an estimated umbilicus). About 1 window in 20 sees no scanner op, 2 in 5 no warp.
  `--ct-aug` stays off (its ops are scanner ops here).
- New: `motion` (sample motion during the scan along a source axis: 1-2 ghost echoes or a linear motion blur, in the per-axis
  operator chain before the coarser sampling grid); zoom-out (`--zoom-max`, `--zoom-out-p`: the next coarser level, of which the
  window shows the central 1/2 z: the effective voxel size x z from 2x coarser data, i.e. a coarser scan); packed targets with
  warps (hard targets take the nearest source voxel's code, f > 0.5 per axis: side classes are never blended); `--gpu-input`
  with warps (the radial record, `NN_RAD_REC` = 8 + 8 D + 32 words, carries the warp; `nn_input_rad` builds write_spatial's
  transported radial vector with the host's float expressions: test_gpu_input 512 warped cases bit for bit).
- Sampler cost (cpu-s per P512 window, 4.8 um rv48 data, idle laptop, 6 workers): none 1.6 (4.5 windows/s), all 6.2 before the
  speed work (0.9 windows/s). Fixed: the warp writer (per-voxel loop: 3.9 s per warped window, 36.8 s on the loaded desk) is
  cache-blocked and AVX2 for the trainer's case (hard targets, CT-only, P % 16 == 0: 17-38 ns per voxel, ~2x), GIN 12 -> 2.2 s
  (row-vectorised), paganin 8.6 -> 1.3 s (axes 0 / 1 by rows), all bit for bit (write_spatial_ref, gin / exp_smooth loops in the
  tests). Estimated all now ~4.5-5 cpu-s per P512 window: a laptop P512 trainer (~1.5 windows/s) wants 8+ workers; the desk's
  12 are enough at ~1 window/s. P768: x3.4. Left: the scanner ops' common float pass (~0.6 s), the noise pass, AVX2 for the
  other writer cases.
- In-domain cost (desk GPU 0 beside another run, production recipe: sources-rv24 2.4 um, P512, 2000 steps, --det 1, seed 1, 12
  sampler workers): holdout val 0.6914 / 0.6807 / 0.6721 / 0.6639 at 500 / 1000 / 1500 / 2000 against 0.6539 at 2000 for the same
  recipe without augmentation (the same validation windows): +0.010 (+1.5%) at this short length, the price of the harder data;
  the gain it buys (other scrolls / scanners) needs an evaluation on another scan. 0.64-0.74 samples/s, sampler wait 0% in 88 of 93
  log lines (<= 9% early): 12 workers keep a desk P512 trainer fed.

## Level-0 normalisation alternatives: train --l0-norm gn|none|ema|ws|slab (2026-10-15)
Every GroupNorm normalises over the whole window, so each level-0 norm is a whole-window barrier for a z-slab pipeline (level-0 slab
recompute stops at ~25 B / voxel). `--l0-norm` replaces the four level-0 norms (enc[0] n1 n2, dec[0] n1 n2 = the head's input; down
norms stay GroupNorm); the mode (and its state) is in the checkpoint header, predict follows it. gn (default) is bit-identical.
- none: gamma, beta + SiLU, statistics mean 0 / rstd 1 (the convs skip their statistics); the GroupNorm backwards drop the statistics'
  terms (nn_set_gn_const: the group sums scaled by 0).
- ema: BatchNorm-style running statistics (--l0-mom 0.01), live GroupNorm through a warm-up (--l0-warm 100), then constants in forward
  and backward. The EMA weights need their own: unet_ema's first-order blend is not enough (val 0.770 at 2000 vs 0.669 recalibrated),
  so the trainer recalibrates them before each validation from --l0-recal 4 training windows (own deterministic sampler).
- ws: none + scaled weight standardisation of the four level-0 3^3 convs per output channel over its fan-in, from the fp32 master
  weights every forward (into a copy the kernels read; their fp4 / fp8 prep as before), the convs reading SiLU outputs with the
  NF-net gain 1 / std(silu(z)) = 1.67, the stem 1; the backward maps the gradient through the standardisation.
- slab: GroupNorm statistics per z-slab of --l0-slab planes (64). A prototype without kernel changes: each level-0 tensor is
  normalised in place with its slabs' statistics (MX: a pseudo-sample shape, one channel block, so (sample, slab) is a sample of the
  same layout; lp_mx_affine), the readers apply gamma, beta + SiLU as none; the backward runs over the pseudo shape, then multiplies
  by the slab's rstd. It costs an extra rounding of each normalised tensor and of its gradient, extra passes, and the three-buffer
  layout (its head-recomputing GN backward wants per-sample statistics).
Tests: test_l0_norm; UFSM_L0_NORM=1..4 test_unet (finite differences through each mode), 1..3 test_grad_mx4.
A/B on the desk (production recipe: 2.4 um side labels, 7 levels 16..256, P512, Muon, 2000 steps, --det 1, the same 8 holdout
windows: --val-seed 1; val = loss, EMA weights as the trainer reports it / live weights from the checkpoints, resumed for one lr-0 step):

| arm | val 500 / 1000 / 1500 / 2000 (EMA) | live 1000 / 2000 |
|---|---|---|
| gn seed 1 | .6909 / .6681 / .6576 / .6502 | .6512 / .6441 |
| gn seed 2 | .7054 / .6817 / .6631 / .6559 | .6562 / .6471 |
| none | .8493 / .6944 / .6753 / .6734 | .6708 / .6710 |
| ema (raw first-order EMA-weight statistics) | .9831 / .9652 / .8612 / .7698 | .6677 / .6625 |
| ema, recalibrated (4 / 16 windows) | 1000: .6967 / .7081, 2000: .6687 / .6667 | |
| ws seed 1 | .7818 / .6842 / .6687 / .6634 | .6659 / .6520 |
| ws seed 2 | .7070 / .6637 / .6734 / .6594 | .6618 / .6600 |
| slab, 64 planes (prototype) | .6901 / .6707 / .6625 / .6553 | .6519 / .6492 |
| slabfull, one 512-plane slab (= GroupNorm + the prototype's roundings: the control) | .7014 / .6749 / .6583 / .6540 | .6570 / .6491 |
Seed spread (gn): 0.006 (EMA) / 0.003 (live) at 2000. Every statistics-free mode loses by several spreads: none +0.023 / +0.027,
ema +0.017 recalibrated / +0.018 live, ws +0.008 / +0.010 (two seeds each side). Slab statistics cost nothing measurable: slab vs its
control +0.001 / 0.000; the prototype's extra roundings ~+0.004 / +0.005 vs gn (the control; about one seed spread). Device peak per
process 4562 MiB (gn, none, ema), 4612 (ws), 5716 (slab: four level-0 buffers, planner 46.1 vs 36.5 B / voxel); samp/s were measured
two arms per GPU only (gn pair 0.70 each, slab pair 0.62 each).
Slab-local GroupNorm as a production design (if its arm holds): fold it into the kernels instead of the prototype's extra passes --
the statistics-producing epilogues (stem fp8, fp4 16-channel convs, the up conv) sum per (sample, group, z / S) (4-plane tiles never
cross a slab), the readers' staging takes the affine per (channel, input plane) (a tile's 6 input planes span at most 2 slabs: two
coefficient sets per channel), the GroupNorm backwards (and the head-recomputing one) reduce per slab. With S equal to the pipeline's
slab, a level-0 slab pipeline needs only a one-slab lag (the halo plane of slab k + 1 is final once that slab's conv1 ran) and the
backward streams per slab (only parameter gradients reduce over the window): the barrier behind ~25 B / voxel is gone. Seams at slab
borders (statistics jump) are the risk to watch (a random z-offset of the slab grid per training window would hide the grid).

Kernel path, stage S1 in progress (2026-10-16; plan: /home/forrest/.claude/plans/level0-slab-streaming.md). `UFSM_L0_SLAB_KERN=1` with
`--l0-norm slab` (MX tensor cores, uniform slabs): the level-0 blocks' m1 / r1 / m2 / r2 hold [n][slab][G] statistics, registered
with `lp_gn_slab_reg` (nn.h); `gnp_mk` / `to_gnp` give every gnp_t built from a registered array its slab geometry (gnp_t zs, zo,
ns, zhw; slab of plane z = (z + zo) / zs), and a reader applies each plane's slab (lp_slab.cuh gn_row / gn_ab: the arithmetic of
make_chan / row_gn, so one slab is GroupNorm bit for bit). No normalised copy, no extra rounding (the prototype stays the default and
the reference). Producer statistics: one pass of nn_gn_stats over the pseudo-sample shape after the conv (S2 folds them into the
epilogues); GroupNorm backwards: nn_gn_silu_bwd over the pseudo-sample shape on the raw tensor with the true statistics (exact).
Slab-aware so far: the head (lp_mx_conv1.cu: forward, the weight gradients, the fused backward; two coefficient banks per block, a
block's voxels capped to one slab so they span <= 2) and the skip half of the 16-wide decoder conv1 (lp_f4up.cu: the bank of each
staged plane, reloaded at most once per tile). tests/test_l0_slab.c: these against fp32 with the right slab per plane (slabs with
very different statistics: a reader on another slab's is ~100% off; head 2e-4, f4u at the per-sample kernel's fp4 error), and the
unet kernel path with one slab = GroupNorm (logits bit for bit, gradients at the run-to-run atomic noise).

S1 continued (2026-10-16): every MX reader applies slabs, so several slabs train (the multi-slab stop is gone; slabs must be multiples
of 8 planes, and with one slab nothing is registered: the statistics are the sample's and every reader runs its GroupNorm kernel).
- make_chan takes the plane whose statistics apply (zg; lp_common.cuh chan_ab: the same affine alone). Readers with one coefficient
  table per staged z range keep two banks: bank 0 = the range's first plane's slab, bank 1 (gn_bank2: at the end of the dynamic
  shared memory, its bytes added by the launcher only for slab statistics) from plane zb = gn_zb (a range of <= 9 staged planes
  spans <= 2 slabs of >= 8; gn_zb3 traps otherwise): conv_fwd_f4_k, conv_fwd_f4r_k (rebuilt per z tile when the slab pair changes),
  conv_fwd_f8_k, conv_fwd_s2_f8_k, conv_fwd_s2_f4_k. Readers with per-thread coefficients reload them when the thread's staged plane
  changes slab (each thread's planes ascend): conv_fwd_s2_f4p_k (down0), conv_fwd_s2_f8_k P16, conv_fwd_f8p_k, the weight gradients
  conv_bwd_w_s2_f8_k (FAST: per plane before xdec, incl. the paired passes), conv_bwd_w_f8_k (FAST, tile and per-element paths),
  conv_bwd_w_f4_k (LY 1 lanes, per-task channels); conv_fwd_f8s_k per element.
- Not yet slab-aware, and not taken with slab statistics: conv_fwd_f4p_k (the packed 16-channel forward: fwd_f4_t routes a
  16-channel GN input with gnp_t ns to the chunk kernels f4r / f4) and conv_bwd_w_f4ws_k (bwd_w_f4_t takes conv_bwd_w_f4_k; same
  values). Both are a few lines (the s2_f4p / conv_bwd_w_f4_k pattern) once their current rewrite lands; until then the slab path
  runs those layers on the slower kernels. The 16-bit / fp32 readers (nn_common.cuh gn_coef / gn_silu_at) trap on slab statistics
  (the slab path needs MX tensors anyway).
- The head-recomputing GN backward (three level-0 buffers) runs over the pseudo-sample shape like the other GN backwards:
  nn_gn_silu_bwd_head_ns indexes the per-sample fp16 logit gradient from (sample, slab).
- tests/test_l0_slab.c: every reader family (stride 1 / 2 forward at fp4 / fp8, 16 / 32 channels, the 4-channel input kernel;
  weight gradients fp4 / fp8, with and without SR, stride 1 / 2) against fp32 with each plane's slab: the error equals the same
  kernel's against per-sample statistics (its fp4 / fp8 baseline), and ~0.8-0.9 from the per-sample reference; the unet with 4 / 2
  slabs (MX-fp8, no SR) against the prototype: 0.059 logits / 0.28 gradients, as with one slab (0.063 / 0.31), and 0.24 / 1.0 from
  GroupNorm. test_grad_mx4 with UFSM_L0_NORM=4 UFSM_L0_SLAB=8 UFSM_L0_SLAB_KERN=1: three level-0 buffers vs four with 4 slabs, logits
  equal, gradients 1e-9. bench_f4conv at its default P384: every output value and statistic equal to the base.
- Training smoke (laptop 5080, production net, P512, rv48 sources, 60 steps, `--l0-norm slab --l0-slab 32` = 16 slabs,
  UFSM_L0_SLAB_KERN=1): runs with the three level-0 buffers at 36.5 B per voxel / 4.89 GB like GroupNorm (the prototype: 46.1 B),
  losses tracking GroupNorm's (step 60: 0.629 vs 0.621, val 0.741 vs 0.736 on 2 batches); 1.49 vs 1.77 samp/s (-16%): the chunk
  forwards and conv_bwd_w_f4_k in place of conv_fwd_f4p_k / conv_bwd_w_f4ws_k, and the producers' statistics pass (S2 folds it into
  the epilogues).

S1 result (2026-10-16, laptop 5080, rv48 sources `--levels 0,1`, P512, 2000 steps, --det 1, --val-seed 1, --aug none; binary 9ae7eea,
UFSM_L0_SLAB_KERN=1 `--l0-norm slab --l0-slab 32`, two arms sharing the GPU at 0.63 samp/s each): the kernel path equals GroupNorm.

| arm | val 500 | 1000 | 1500 | 2000 |
|---|---|---|---|---|
| slab kernel 32, seed 1 | 0.6885 | 0.6622 | 0.6511 | 0.6436 |
| gn, seed 1 (worker S's arm) | 0.6892 | 0.6635 | 0.6523 | 0.6436 |
| slab kernel 32, seed 2 | 0.7102 | 0.6730 | 0.6581 | 0.6460 |
| gn, seed 2 | 0.7099 | 0.6731 | 0.6584 | 0.6481 |

Mean at 2000: 0.6448 vs 0.6458 (the prototype's slab 32 arms: 0.6508, its +0.0049 rounding penalty gone with the extra passes).

Stage S2, z-window launches (2026-10-16): `nn_set_zwin(zw0, zw1, D0, first, last)` (nn_bwd.cu, per host thread) makes the next ops
compute only output planes [zw0, zw1) of their WHOLE output tensors, the window given in planes of a depth-D0 reference (the level-0
depth) and scaled to each output (zw_win: zw0 D / D0, so down0's level-1 output gets [zw0 / 2, zw1 / 2)); first / last zero /
finalise output GroupNorm sums (zr_zero / zr_final). Every kernel keeps its whole-launch tiles, staging, rings and stochastic-rounding
keys (all already keyed by global voxel position), so windows covering a tensor give one launch's bits, given the same seeds: the op's
seed counter replayed per window with nn_sr_set_ctr (the test does; the streamed schedule, S3, must draw each level-0 op's seed once).
Windows lie on the kernels' z tiles (2 or 4 planes; 8 for the stride-2 backward-data's gx), or end at the tensor's depth (partial
last windows). Covered (default paths unchanged, split_t gets zw0 / zw1, the other kernels take a tile range):
- conv_fwd_tc (forward and backward-data): conv_fwd_f4_k / conv_fwd_f4r_k (the 16 | 16 split output too), conv_fwd_f8_k /
  conv_fwd_f8p_k, the stem conv_fwd_f8s_k (its 4-channel MX input and 16-bit inputs);
- stride 2: conv_fwd_s2_f8_k / s2_f4_k / s2_f4p_k (down0 forward), conv_bwd_w_s2_f8_k (down0 weight gradient), bwd_data_s2_f4q_k /
  bwd_data_s2_f4_k (down0 backward-data, set or accumulated into the skip gradient);
- weight gradients: lp_bwd_w_f4 clips its gy pre-pass slabs to the window (conv_bwd_w_f4_k and the warp-specialised kernel already
  take a z-step range; the GN apply's whole-layer gy records are not used under a window: their layout starts at z-step 0);
  conv_bwd_w_f8_k (the stem's FAST path);
- slab-local GroupNorm backward (MX-fp4: stats, the lane-pair apply, the head-recomputing pair): on the [slabs] pseudo-sample shape
  of one sample a window of whole slabs is a pseudo-sample range (gn_nwin: kernels take a first sample n0; other rows' partial sums
  zeroed, so ggamma / gbeta accumulate over windows; the records-writing apply is skipped: same gx bits);
- the 2x upsample (up2_mx_tile_k: dec[0]'s up transient) and its backward (up2_bwd_mx_tile_k: coarse planes [zw0 / 2, zw1 / 2),
  reading the fine gy planes around them); the GPU-built network input (input_rad_k: planes [zw0, zw1)).
Under a window the routing avoids conv_fwd_f4p_k (packed 16-channel) and the fused-upsample f4u, which are not window-aware yet
(the chunk kernels give the same bits for 16 channels: the test's no-window comparison is 0); kernels without window support abort
with a message (16-bit tensor-core convs, the per-voxel / slice upsample backward, the bf16 stride-2 backward-data, the non-MX-fp4
GN backward, the fp4 weight gradient without its gy pre-pass). Halos for S3: a 3^3 conv's window reads one plane beyond each end,
down0's backward-data and the upsample backward read the fine planes around their coarse window, down0 the two level-0 planes
below / above. tests/test_zwin.c (test-all): every family at 16 / 32 channels, windows of 8 and 16 planes, depths 32 and 36 (a
partial last window): outputs byte for byte, statistics at double-sum noise, weight / GN-parameter gradients summing to the one
launch's (accumulated windows: a window ignoring its range would add the whole gradient each time), lone windows writing exactly
their planes, and no-window launches equal.

Stage S3, the streamed schedule (2026-10-16; `UFSM_L0_STREAM=1|2` or `unet_set_l0_stream`, unet.c `l0st_*`): on the slab kernel
path with several slabs the level-0 ops run per slab through S2's windows on the WHOLE tensors, in a skewed schedule.
- F1 (ascending): the stem at slab t (its statistics by `nn_gn_stats_n`: one slab of the pseudo-sample shape with the whole
  launch's block partition, so the slab pass's bits), enc[0]'s conv2 and down0 at t - 1 (conv2's window reads a1's first plane
  of slab t, whose statistics have just closed); down_norm's level-1 statistics after. Coarse levels unchanged.
- F2: dec[0]'s conv1 at t, conv2 at t - 1 (stream 2: enc[0]'s a1 at t and the skip at t - 1 recomputed first, conv1 at t - 2).
  The head stays whole (S3).
- B1 (descending, after the whole head backward): GN2 + conv2's weight gradient at slab t; conv2's backward-data, GN1 and conv1's
  weight gradient at t + 1; conv1's backward-data (up part | skip part) at t + 2; the upsample backward into gout[1] at t + 3
  (its coarse window reads the fine up-part gradient's planes around it). dec[0]'s up transient is built once (whole), at
  the first conv1 weight gradient. Stream 2 recomputes enc[0]'s a1, the skip, dec[0]'s a1 and a2 three, two, one and zero
  slabs ahead.
- B2' (ascending, from d/d downo[0] after enc[1]'s backward): down0's weight gradient and its backward-data into the skip
  gradient at slab t, enc[0]'s GN2 and conv2's weight gradient at t, conv2's backward-data, GN1 and the stem's weight gradient
  at t - 1 (stream 2: enc[0]'s a1 one slab ahead, the skip at t).
- Seeds: each op's first draw in the unstreamed order is replayed for each of its windows (`nn_sr_set_ctr`), the draws per
  window asserted equal; recomputes replay the forward op's record (`l0st_rc / l0st_rk`) and keep the forward's statistics.
- Buffers: the levels share one A and one B, and d/d downo[0] is level 1's B, so dec[0]'s and enc[0]'s d/d a2 (and dec[0]'s
  up transient) get the stream's own buffers (unstreamed, down0's backward finishes before enc[0]'s writes B).
- Supported: one sample, the plain layout (no spatial split / three level-0 buffers / offloads / lean 2 / head slabs /
  chunked up part), recompute 1, MX activations and gradients, two-conv level-0 blocks (the stream aborts with the reason).
tests/test_l0_stream (test-all, plain and UFSM_L0_STREAM_POISON=1): the production chain, 7 levels 16..256, slabs 16 / 32,
down_norm 0 / 1, two training steps and a validation forward, unstreamed vs stream 1 vs stream 2: logits, gout[1], dec[0]'s
skip gradient, d/d downo[0] and the final skip gradient bit for bit, parameter gradients <= 4e-7 relative (6e-7 at P256: the
weight gradients' atomic order), the same seed counts, at P128 and P256. Poison: dead transient slabs (dec[0]'s d/d a2, A,
the up-part gradient; enc[0]'s d/d a2, A) are overwritten with NaN scales after their last reader, and in stream 2 every
forward activation (enc[0]'s a1 / a2, dec[0]'s a1 / a2) at each phase's entry: still bit for bit. Controls fail as they must:
POISON=2 (a transient one slab early), POISON=3 (no skip recompute). Laptop 5080 P256 step (forward + backward, shared GPU,
noisy): unstreamed 141-151 ms, stream 1 157-177 (more, smaller launches; slabs of 16 cost most), stream 2 183-201 (+21-42%:
the plan's +32% recompute estimate).
What S4 (rings / mapped chunks) still needs, given this schedule (live slabs per tensor, whole = W):
- level-0 activations: F1 a1 3 slabs, the skip 2; F2 (stream 2) a1 3, the skip 3, dec[0]'s a1 3, a2 1; B1 enc[0]'s a1 3,
  the skip 5 (recomputed at t - 2, read by conv1's weight gradient around t + 1), dec[0]'s a1 3, a2 1; B2' a1 3, the skip 2;
- level-0 gradients: B1 d/d a2 3, A 3-4, the up-part gradient 3; the skip gradient W from B1's conv1 backward-data until B2'
  (the plan's device / host / replay dial); B2' d/d a2 3, A 1;
- still whole in S3 and to be streamed: the network input X (rebuild per slab from the CT: input_rad's window, S2 part 7, for
  the stem's forward, its recomputes and its weight gradient), the head and loss (F2 head per slab with the loss's first pass,
  B1's head backward per slab before GN2 of the same slab: train.c head_slab_pass), gout[0], dec[0]'s up transient (per-slab up2
  windows); level 1 (downo[0], gout[1], d/d downo[0], dec[1]'s output) stays whole.

CUDA virtual memory management under WSL2 (probed 2026-10-16, laptop 5080): cuMemAddressReserve of 64 GB on the 16 GB card,
cuMemCreate / cuMemMap / cuMemSetAccess of 256 MB chunks at 2 MB granularity, kernels addressing the mapped ranges correctly;
map + access + unmap of 256 MB ~2 ms of host time. So S4 can keep whole-window tensor addresses and map physical slab chunks under
them (MX data and scale planes of 8 planes at P >= 512 sit on 2 MB boundaries), instead of slab buffers with a zkey base in every
kernel: S2's windows on whole tensors are then all the kernels need. The host map / unmap traffic (hundreds per P1024 step) belongs
in a helper thread ahead of the compute stream, unmaps after the last reader's event.

Stage S4, slab rings (2026-10-16; `UFSM_L0_RING=1` with `UFSM_L0_STREAM=2`, unet_set_l0_ring; worker S4):
- Rings without any map / unmap traffic: nn_ring_alloc (nn_core.cu, the driver's VMM through cudaGetDriverEntryPointByVersion, no
  -lcuda) reserves a tensor's whole virtual range and maps slab s of each region (an MX tensor's data planes, then its scale planes)
  onto physical chunk s mod R once, at the build -- one cuMemCreate per slab slot (cuMemMap maps whole allocations: a sub-range of one
  handle returned CUDA_ERROR_NOT_SUPPORTED under WSL2), the same handles mapped R times apart. Kernels keep the whole-tensor
  addresses: S2's windows and S3's schedule unchanged; writing slab s + R reuses slab s's memory, so the schedule's liveness is the
  only contract (verified by the bits: a ring one slab short changes them). Slabs must lie on the 2 MB mapping granularity:
  l0_slab h w a multiple of 2 MB (16-wide level 0: P a multiple of 256 at 32-plane slabs, P 512 / 1024 at 16).
- Rings (depth; each one shorter fails test_l0_stream): enc[0].a1 3, the skip 5, dec[0].a1 3, dec[0].a2 1, the stream's d/d a2 3,
  level 0's A 3, its B (the up-part gradient) 3, dec[0]'s up transient 3 (built per slab by up2 windows from silu(gn(dec[1].a2))
  computed once into gout[1], free until B1's upsample backward writes it top down), gout[0] 1, the network input 3 (with
  --gpu-input: rebuilt per slab from the CT channel by input_rad windows before each stem window, recompute and weight gradient),
  the skip gradient 1 (host mode, below). UFSM_L0_RING_R overrides in that order. The shared A / B are sized for levels >= 1.
- The stream head (unet_set_l0_head(fwd, bwd, ctx)): no whole-window logits -- F2 runs the 1^3 head on each slab as its statistics
  close and hands the fp16 slab to fwd (the loss statistics); B1's dec[0] step at slab t computes the slab's logits again, bwd writes
  the fp16 logit gradient over them, the head's backward writes gout[0]'s rows of the slab before GroupNorm 2's backward. The 16-wide
  head has no fused kernel: per range as the whole path (nn_head_bwd_x's declined draw, conv_bwd_w1_tc_k over a voxel range,
  conv1_to_mx_t_k over a row range with its own draw), so gout[0]'s bits and the seed count equal one whole call's.
- The skip gradient's host mode (UFSM_L0_RING_SKIP=1, default; 0: whole on the device): a 1-slab ring with a pinned host copy, each
  slab out as dec[0]'s conv1 backward-data completes it (B1) and back before down0's backward-data accumulates into it (B2'), on the
  compute stream (1.2 GB each way per P512 step; not yet overlapped).
- Tests: test_l0_stream arms 3 (rings, whole head), 4 (stream head on whole tensors: against the unstreamed run bit for bit but the
  fp16 logits, within 0.004 of 10), 5 (stream head in rings: against 4), 7 (--gpu-input with the input ring: against 6, the whole
  input), all bit for bit in logits, gout[1], dec[0]'s skip gradient (the whole 16-channel tensor), d/d downo[0], the final skip
  gradient, seeds; every parameter-gradient part < 1e-5 (the atomic order). Plain and poison legs, P 128 / 256.
- Memory (laptop 5080, tests/bench_l0_ring: the production chain in the plain layout the stream needs, --gpu-input; device peak incl.
  the CT channel, weights, optimizer, context; another process shared the GPU, so the times are rough):

| | dry B / voxel | device peak | step |
|---|---|---|---|
| P512 plain (unstreamed) | 85.8 | 12.82 GB | 852 ms |
| P512 stream 2, whole tensors (S3) | 85.8 | 15.58 GB | 1193 ms |
| P512 rings, slab 32 | 28.6 | 4.62 GB | 1111 ms |
| P512 rings, slab 16 | 20.8 | 3.56 GB | 1157 ms |
| P512 rings, slab 16, skip gradient on the device | 29.5 | 4.73 GB | 1282 ms |
| P768 rings, slab 32 | 23.3 | 12.36 GB | 4226 ms (107 Mvox/s) |
| P1024 rings, slab 16 / 32 (dry only) | 16.8 / 20.7 | 18.1 / 22.3 GB | |

  The production three-buffer layout (worker C, same laptop): P512 4.63 GB, P768 13.83 GB at ~231 Mvox/s. The rings at P1024 / slab
  16 leave the coarse levels as the bulk: at P1024 ~13 B per voxel of the 16.8 (encoder 4.3, decoder 3.0, the shared A / B 3.2 --
  B holds dec[1]'s unchunked 32-channel up-part gradient, the stream refuses the chunked one --, coarse gout / gskip 2.5), all
  resident in the plain layout (no coarse skip / down-output offloads). P1024 on 16 GB needs those down to ~7 B (stage S5).
- Time: the rings cost nothing over the stream on whole tensors (1111 vs 1193 ms); the stream itself (S3's recompute and windowed
  launches) and the plain layout's paths against the production layout's are the cost (S5 / perf).

## Trainer host RAM: packed targets, validation windows on disk (2026-10-15)
At P768 the laptop trainer (production net, 6 workers, 7 slots, 4.8 um rv48 side labels) was OOM-killed by the 37.6 GiB slice cap
before its first step (38.3 GB RSS): host RAM, not the GPU, bound the window. Per voxel the host held 5 B per sampler slot (CT fp16,
2 target bytes, the mask), 5 B per stored validation window, ~4 B of scratch per worker and ~19 B of pinned offload buffers.
- `train --pack-targets 1` (default where it applies): hard two-class targets (every source's recto target side-encoded, no warps,
  --heads, --split, papyrus task, --dist-head, sheet prior, --overfit) travel as one code byte per voxel (sample.h SAMPLE_TM_*: bit 0
  target 0, bit 1 target 1, bit 2 mask) through the slots, the validation windows and PCIe; nn_unpack_tm rebuilds the target and mask
  bytes on the device (copy stream), so every device consumer is unchanged. The sampler refuses (aborts) a soft value.
  tests/test_pack_tm.c: every code, both streams, the refusals; with a sources.json, deterministic windows with and without packing
  equal byte for byte after the device unpack (rv48, P128: ok).
- The validation windows come from a deterministic sampler (same windows every run; they were the first 8 a 4-worker sampler
  happened to finish). `--val-store disk` (default): written once to an unlinked file in the run directory (space returns at exit)
  and read back through one host batch at each pass (POSIX_FADV_DONTNEED keeps them out of the page cache; on tmpfs they would be
  RAM); `ram`: the former copies; `regen`: drawn again at each pass. The three modes and --pack-targets 0 give the same validation
  loss on a fixed model (P256, --lr 0: 1.11942 in all four).
Laptop RTX 5080, P512: host VmHWM 16.5 -> 12.3 GB, device 4626 MiB (planner 4.89 GB), 1.53 samp/s either way. P768: killed at
38.3 GB -> runs at 32.9 GB VmHWM (31.5 GB steady), device peak 13.8 GB (planner 15.13 of 15.51 GB), 0.46 samp/s = 208 Mvox/s
(P512: 205), sampler wait 5-7% with 6 workers (UFSM_SAMPLER_PROF: 7.3 cpu-s per P768 window, 48% of it the x16 CT stage).
- The x16 stage of --gpu-input (the CT channel alone) gathered each output row along the window's symmetry: for 2 of 3 symmetries a
  strided source axis, one cache line (often a page) per voxel. write_x_ct gathers 16 rows whose source rows are adjacent at once
  (16 output planes when output z runs along source x, their noise rows drawn first in write_x's order and kept as table indices),
  the same draws and float operations: test_gpu_input's 48 symmetries stay bit-identical, bench_sampler hashes unchanged
  (UFSM_WRITE_X_ROWS=1: the row path). P768: x16 3492 -> 1050 ms per window, the window 7.3 -> 4.6 cpu-s, sampler wait 5% -> 0%.
## Planner dial: --mem fast is PCIe-aware; the pick prints B / voxel and copy bytes (2026-10-15)
`--mem auto` (default) still takes the leanest mode that fits. `--mem fast` takes the fastest and first tries every mode with no
pinned host offloads (unet_set_host_offloads(0): encoder a1, input, level-0 skip / A share, down outputs, coarse skips) and no
shared encoder a1 unless the mode is one; only if none fits, the modes as auto builds them (the three level-0 buffers only there:
they rely on their host copies). unet counts the offload bytes of every build, dry included (unet_last_offload_bytes); the planner
prints the mode, its device bytes per voxel, the batch upload and the offload bytes per step. Production net, laptop RTX 5080:
| P | --mem | mode | device need | host offloads each way per step | measured |
|---|---|---|---|---|---|
| 512 | auto | lean 2, shared a1, three level-0 buffers | 4.89 GB, 36.5 B/voxel | 3.66 GB (27.3 B/voxel) | 4626 MiB, 1.53 samp/s, host VmHWM 12.3 GB |
| 512 | fast | MX gradients, no offloads | 13.68 GB, 101.9 B/voxel | 0 | 12546 MiB, 1.72 samp/s (+12%), host 6.5 GB |
| 768 | auto / fast | lean 2, shared a1, three level-0 buffers | 15.13 GB, 33.4 B/voxel | 12.35 GB | 13.8 GB, 0.46 samp/s |
| 1024 | auto | none fits: the smallest (three level-0 buffers) needs 35.05 GB; the no-fit fallback now tries it (it tried recompute 2, 85.8 GB) | | | |
The batch upload is 0.40 GB per P512 step (CT fp16 + packed targets). At P768 the offloads move ~12 GB each way per ~2.2 s step
(5.6 GB/s); at the same B / voxel they would bind near 800 Mvox/s, so a faster GPU wants --mem fast where it fits.
## Training input built on the GPU; the planner counts the separate input (2026-10-15)
The network input is z-scored CT plus the radial unit vector (3 channels, from the window's corner, cube symmetry and the axis centre
per source z). The sampler wrote all 4 as 16-bit per voxel (8 B: pinned slot, PCIe, a device buffer dead after the forward's MX input
conversion, and the validation copies). `train --gpu-input 1` (default where possible: 16-bit -> MX input conversion, whole windows,
the plain 4 inputs, no warps / sheet prior / slab appearance plan; settled after the memory planner because the conversion follows
the MX-gradient mode): batches carry the CT channel alone (2 B) plus a 32-byte record and the axis centres per sample; nn_input_rad
(lp_mx_cvt.cu) builds the MX input straight into unet's xin, bit for bit sample.c write_x + nn_h16_to_mx (IEEE intrinsics: the
build's -use_fast_math would contract / approximate; tests/test_gpu_input.c, 896 cases incl. all 48 symmetries, slabs, axis-on-voxel
zeros, fp16 / bf16, MX-fp8 / fp4, 2 samples; the trainer's UFSM_XIN_HASH input hashes equal over deterministic runs). The radial
pass is cheap on the CPU (vectorised), so the sampler's x16 stage only drops ~15%; the gains are memory. Desk 5060 Ti, P512, 7 levels
16..256, fp4 gradients: peak GPU 6816 -> 5772 MiB (the input now fits gout[0]: "batch buffers and input inside the gradient
buffers"), trainer host RSS 41.1 -> 24.2 GB (14 slots + 8 validation windows), 1.03 -> 1.06 windows/s. Planner: lean 2 with MX
gradients now counts the input when it cannot sit in gout[0] (it assumed it inside: 6.19 GB planned vs a 7.15 GB nvidia-smi peak,
CUDA context included, with the host-built input; now 7.26 GB; 6.19 GB vs 6.05 GB with the GPU-built input).
## MX-fp4 gradients for 8- and 16-channel levels (2026-10-15)
User: as much fp4 as possible across the net; the kernels follow the widths, never the reverse (a 32-wide level costs 4x compute for
no memory: 32 ch x 4 bit = 16 ch x 8 bit). The MX layout already had 8 / 16-channel blocks; the gradient paths assumed 32:
- upsample backward: the tiled kernel templated on the block width (BW / 8 threads per voxel); per-voxel kernel with exact SR;
- stride-2 backward-data: 16 channels on the bf16 tensor-core kernel (4 nibbles per lane); 8 (and any narrow fp4 layer) on the
  parity-major kernel templated on fp4 rows (it replaced a fallback through two full-size MX-fp8 copies and a double rounding);
- stride-1 backward-data with <= 16-channel fp4 gy: the 32-padded chunk path (rows copied with their own scale) -- the p16
  kernel re-rounded paired voxels to their shared scale to nearest and small_f8 used one scale per tile (biased gradients);
- the shared MX output epilogue (lp_common.cuh, lp_s2.cu): `mm < bw / 16` stored no values for 8-channel blocks -- 8-channel MX
  outputs (activations too) never worked; now a per-element predicate (bit-identical at 16 / 32: 26 bench_f4conv hashes);
- 4-byte rows (8-channel fp4) in the fp4 conv staging, GN backward fp8 loads, mx_load16, the FAST stem / stride-2 weight
  gradients (one narrow block in the tile's slot, zero-filled), split halo block widths (zsegs);
- planning: up-part chunks start at a whole MX block (mx_chunk_min); no gy-record reserve below 32-channel level 0;
- trainer: no in-place logit gradient when the fp16 logits share the head's input-gradient buffer (unet_logits_in_gx: the
  in-place gradient made the head's backward-data write gout[0] as 16-bit, 32 B into a 9 B buffer; latent for any 2-output
  lean MX-fp4 layout).
Tests (test_grad_mx4): GN backward 8..64; backward-data with weights exact in every format (8->8, 16->8, 8->16, 16->16, 32->16,
16->32: the mean of 16 SR runs must reach 1.5 / sqrt(K) of one run -- the old staging failed it); s2 backward-data 8 / 16 (+
accumulate); up2 backward 8 / 16; end to end {16,32,64}, {16,16,32}, {8,16,32}: unbiased vs 16-bit, fp16 logits, training bytes
-22..-33%. Opt-in UFSM_G4_NARROW=1 until a paired training check; fp4 gradients also need P % 256 == 0 for 7 levels.
Measured (P512, 7 levels): 16-wide 69.6 -> 39.0 B/voxel planner, desk 9.7 -> 6.8 GB, 1.01 -> 1.03 windows/s; 8-wide trains
(laptop P256 smoke) but is NOT faster (desk 913 vs 917 ms per step): 8-channel convs run on the 32-padded fp4 path or the generic
fp8 small kernel (8->8 forward 58 ms vs 39 for 16->16; down0 backward-data 42 vs 20 ms). Speed at 8 channels needs dedicated
narrow tensor-core kernels. Level-0 16->16 convs run ~7x above the memory floor (39 ms vs ~5 ms of traffic at 448 GB/s).

## Recto / verso side labels from the published wraps (2026-10-15)
Every voxel between two consecutive labelled wraps k and k + 1 (windings increase outward) is recto side in the half nearer k and verso
side in the half nearer k + 1: the published wraps mark recto faces, the label leaves each wrap on its outer side only. band_field_side_pair:
feature transforms of the labelled voxels by winding mod 4; the nearest wrap is k, the nearest of the other classes in the opposite
direction ends the voxel's interval (k + 1 recto, k - 1 verso, k +- 2 a missing wrap: unknown); codes within 5 steps of k's are k's own
sheet across a turn boundary of the codes. The pair-rule phase (fc02990) decided sides by distance alone and left 16-24% of the papyrus
unknown in strips beside wraps whose neighbouring gaps differ; this rule labels 94-97% of it (crops), 99.6% agreeing with a Python flood. User: every papyrus voxel recto or verso, so the
unbounded cases fall back: no wrap on the voxel's own side within 2 rmax (holes / unsegmented sheets in the published meshes; ~5% of
a damaged window's papyrus) takes the side of its nearest wrap away from that wrap's neighbours, only within half the local spacing (k to that neighbour; farther
the unseen next wrap may be the nearer one: unknown, user: "if it lies too far from the surface labels we don't want to be randomly
picking"); a gap of m windings (missing wraps)
is interpolated (winding position k + m d_k / (d_k + d_b), recto in the first half of each winding). Box crops: 95.2% (damaged) to
99.99% of papyrus labelled; the scroll centre inside the innermost published wrap (winding 10) stays unknown (74% near the axis). The laptop's q-4p8.zarr was a partial copy (445 shards of
the desk's 13056): label boxes need the desk's codes over the box plus halo.
phase-raster --pair 2 writes the raster once (lossless volcomp, 0 unknown, 1 verso, 2 recto); downsample --mode max gives the levels the
sampler places windows with. Sources: "encoding": "side" on the recto target with the source's "iso": channel 0 recto-side papyrus,
channel 1 verso-side papyrus, void 0 in both, papyrus of unknown side and CT 0 ignored; no erosion, softening or label morphing.
The feature transform's y / z passes run on tiles of 64 x lines (bit-identical; 1.4x), the x pass by two sweeps, distances past the
bound limit dropped between passes, classification on the block interior only: 12.9 -> 9.1 s per 384^3 block (one thread). First box
(12 x 20 x 20 mm around the Paris4 holdout, 4.8 um): 24 min on 16 laptop threads. Void cutoff at 4.8 um: 60 (CT histogram valley 62-66).

## Several recto teachers at 9.6 um: one head per teacher (2026-10-15)
Teachers: the published surfaces (labels), the upstream ps256 3D-recto mask and the upstream m7 surface probabilities. At 9.6 um the
bands are 2.5-2.7 voxels (labels), 4.1-4.7 (ps256), 2.7 (m7 >= 0.5) thick (4 x mean inside EDT - 2, three Paris4 crops); the radial CT
profile across each band's centre voxels peaks at the same place for all three with papyrus inward, and m7's peak sits at the label
centre (median offset 0), so m7 is a recto model too. Sources: per-source "erode" / "dilate" (binary targets) equalise widths (ps256
erode 1); m7 is a probability target on the recto channel, its noise floor zeroed in deep void with --iso-void 1.
One head would learn the teachers' average where they disagree (partial probabilities: haze). `train --heads H`: the surface net gets H
recto heads (output channels 0 and 2..H, the sheet channel stays 1) and every source names the head its recto target trains ("head": k).
The upload writes each sample's recto target into its head's channel (device targets B x cout x P^3) with weight 0 on the other heads, so
the shared trunk sees every teacher and each 1x1 head only its own; validation prints the loss per head over its own sources' windows;
the checkpoint records the head channels and sources (extra "heads"); predict --channel picks a head. --tol and --recall act on channel 0.
Sampler throughput: a worker draws only into a free batch slot, so concurrency is min(workers, slots - the trainer's 1-2). At P 512 a
slot is ~1.6 GB pinned and a worker's scratch ~1.35 GB: 12 workers with 4 slots drew ~2.5 at a time (0.8 samp/s, 60-75% wait at
4.2 cpu-s per patch); 7 workers with 8 slots cost the same memory.

## Sheet extraction: thickness, recto/verso sides, scoring (2026-10-09)

Plan (research by 7 agents + codebase exploration, user-approved): extract SHEETS, not a recto probability. No vision
transformers or Mamba (they lose or tie well-tuned CNN U-Nets in rigorous 3-D benchmarks; the Kaggle surface-detection
top 10 were nnU-Net ResEnc ensembles); levers in order: target representation, long training, post-processing, then
ResEnc-style network changes (residual coarse blocks, lighter decoder, frozen or per-voxel normalisation).

- Scoring harness `tools/score_sheets.py` (laptop wrapper ~/ufsm-data/score_local.sh; unit test tests/test_score_sheets.py)
  on the Paris4 holdout boxes against the thin reference (paris4-sheet-20261003) and the winding band reference: pixel
  F1/AUC, F1 at tolerance 0/1/2 label voxels raw and after radial non-maximum suppression, the radial profile through
  label centres (peak offset, FWHM, double peaks), band-instance VOI/ARAND/splits/merges over known and over material
  voxels (4.8 um CT >= Otsu tau), and sheet pieces from a recto/verso side channel (`--side-channel`).
  `tools/label_grid.py` is the trainer's native <-> 4.8 um mapping (label j covers natives 2j-1, 2j); score_bands,
  sheet_connectivity, score_surface and score_open used {2j, 2j+1} pooling or floor indices before (F1 moves ~0.003).
- Baseline (P528 W544/16 EMA; FWHM in 2.4 um voxels; labels themselves measure ~3.8):

  | checkpoint | box | F1 | AUC | F1 tol 0/1/2 | FWHM | VOI (merge) |
  |---|---|---|---|---|---|---|
  | verso3 step 5000 | 48128 | 0.242 | 0.777 | 0.239/0.409/0.573 | 21.1 | 1.48 (1.21) |
  | verso3 teacher | 48128 | 0.242 | 0.778 | 0.239/0.408/0.574 | 22.0 | 1.49 (1.22) |
  | start-E (4 levels) | 48128 | 0.192 | 0.725 | 0.191/0.337/0.490 | 25.4 | 1.98 (1.10) |
  | verso3 step 5000 | 48512 | 0.259 | 0.796 | 0.256/0.434/0.612 | 23.0 | 1.43 (1.19) |
  | verso3 teacher | 48512 | 0.261 | 0.793 | 0.256/0.432/0.612 | 23.9 | 1.27 (0.98) |
  | start-E | 48512 | 0.197 | 0.733 | 0.195/0.345/0.512 | 27.2 | 1.73 (1.32) |

  The recto output is a plateau ~5x wider than the labels (63% of profiles never fall below half maximum within +-16
  voxels; peak offset spread +-5.6 voxels); radial thinning does not help (no ridge inside the plateau); merges dominate the
  instance error. A Gaussian target around jittered traces makes the jitter-blurred bump the BCE optimum (width^2 =
  sigma^2 + jitter^2); a class boundary or a signed distance keeps its crossing under jitter.
- Recto/verso side labels: `band_field_side` (src/band.c) returns, for every voxel the band field knows, whether it lies
  outward of its nearest labelled recto (SIDE_RECTO) or inward (SIDE_VERSO). The recto faces the scroll axis, so going
  outward the side turns verso -> recto exactly on each recto face and recto -> verso midway to the next recto; unknown
  where the band is (beyond the flood radius, conflicts, missing wraps). tests/test_side.c (concentric shells). `ufsm band
  --side` writes it for a box; the sampler fills batch.side (sample.c band_window, CT <= --side-air unknown); `train --side 1
  [--side-lambda L --side-air T]` (task band_affinity) adds the last output channel, BCE over its own mask
  (nn_side_targets), split-z halo planes masked; `--head-map` maps donor head rows for --init-from.
- Extraction from sides (score_sheets.side_pieces): cut the recto-side voxels on verso -> recto boundaries (a verso-side
  material 6-neighbour, and the smoothed recto share increasing outward along the radial direction), 6-connected
  components of material minus cuts, small pieces dropped, the rest to the nearest piece. On the reference sides of box
  48128 (oracle) it recovers the band instances: VOI 0.001 (23 pieces, 23 bands, ARAND 1.000) over all CT, 0.024 over CT >=
  80 (vs 1.48 from the recto channel). Checking only the radial neighbour leaked through oblique faces; smoothing that
  counted unknown voxels as verso flipped the orientation at the edges of the known region (normalised smoothing fixes it).
- Frozen GroupNorm statistics (plan D1): `unet_gn_nstats / unet_gn_stats_get / unet_gn_accumulate / unet_gn_pooled /
  unet_gn_freeze` (inference forwards replay the stores without reducing statistics), `predict --gn-save FILE` (pooled over
  every window: mean of window means, within-window variance + variance of the means) and `--gn-frozen FILE`. A window
  replayed with its own frozen statistics is byte-identical (tests/test_gn_frozen.c, predict on a 256^3 box).

- D1 result (verso3 step 5000, box 48128, calibration: 8 windows of 544 in the training block below the holdout): frozen
  statistics cost F1 at every window and do not remove the window effect, which is context, not GroupNorm:

  | window | statistics | F1 | best cutoff | AUC | F1 at 115/255 |
  |---|---|---|---|---|---|
  | 288 | live | 0.233 | 0.573 | 0.769 | 0.217 |
  | 288 | frozen | 0.224 | 0.592 | 0.758 | 0.214 |
  | 544 | live | 0.242 | 0.584 | 0.777 | 0.222 |
  | 544 | frozen | 0.233 | 0.604 | 0.766 | 0.218 |

  Not adopted (the options stay for other checkpoints); the per-voxel channel norm (plan D2) is not pursued: on current
  checkpoints the best cutoff moves by ~0.01 between windows.
- Narrower coarse decoder (plan B1): `unet_cfg.dec_widths` / `train --dec-widths 0,32` (checkpoint header "dec_widths" only
  when set; level 0 keeps the encoder width so the level-0 memory modes are unchanged). dec[i].c1 reads dec_widths[i+1] +
  widths[i]; a level whose decoder is narrower than its encoder cannot hold the encoder's gradient in dec[i].a2's buffer
  (a0_share), so its A is the shared one. Desk widths, level 1 at 32: step -14% (laptop, GPU shared: 1001-1030 -> 861-872 ms;
  dec0.c1 222 -> 157 ms, dec1.c1 88 -> 40 ms), memory 78.5 -> 80.6 B/voxel, 10.96M -> 10.71M parameters. test_unet
  (finite differences, UFSM_DEC_WIDTHS / UFSM_BIG_DEC_WIDTHS) and test_split case 36 (desk mode, levels 1/2 at 32/64) pass.
  Accuracy A/B (laptop, from scratch on the Paris4 band block, cover x13 = 7048 steps, P256, Muon 0.01 WSD, seeds 1-2;
  F1 / AUC per holdout box 48128 | 48512): control 0.242 / 0.239 | 0.277 / 0.278 (AUC 0.769-0.775 | 0.806-0.808), dec_widths
  0,32 0.250 / 0.241 | 0.278 / 0.273 (AUC 0.768-0.775 | 0.805-0.808); FWHM equal (16.3-16.5 | 17.9-18.4); band VOI noisy
  (control 2.48 / 2.48 | 1.29 / 1.42, narrow 2.41 / 1.90 | 1.07 / 1.67). Mean dF1 +0.005 | -0.002: within the seed spread at
  -14% step time -> recommended for new runs (`--dec-widths 0,32`; a fresh decoder, not on resume). Side note: a net trained
  from scratch on this one block next to the holdout matches the 37-scroll desk model there (locality, not missing data).

- Side A/B (laptop, Paris4 band block cover x5 = 2738 steps, P256, warm start verso3 step 5000, Muon 0.005 / lr 5e-4 WSD;
  control = recto + affinities, side = + `--side 1 --side-air 80`), seed 1, box 48128: recto unchanged (F1 0.265 vs 0.261,
  FWHM 16.9 vs 17.1); fine-tuning itself helped (verso3 0.242 -> control 0.265; VOI 1.48 -> 1.32). The side channel reaches
  72% accuracy on known voxels (48% within one voxel of a recto face, ~77% far from boundaries) and the cut-based pieces
  merge almost everything (VOI 4.01); components of its confident recto-side stripes (P >= 0.7) give VOI 1.97 — both worse
  than the recto-cutoff pieces (1.32). Not adopted at this training length. Diagnosis: along the radial line through the
  traced recto points of this box only 25% show a clear CT edge (contrast > 30; edge offset std 2.9 voxels), 57% almost
  none (compressed sheets: the trace is an interpolation); the face position is not visible where the side boundary must
  be placed, the same limit that keeps the recto ridge wide. Laptop runs were sampler-bound (6 batches in flight with the
  band-field EDT per sample): `train --sampler-slots N`.
- Continued (another 2738 steps, fresh optimizer, side loss x2): side accuracy 72.0 -> 75.4% (box 48128), 79.4% (box 48512);
  pieces from it still merge (VOI 3.73 / 3.20); recto unchanged vs control (F1 0.266 / 0.289 vs 0.271 / 0.292). The recto F1
  plateaus with more fine-tuning on the one band block (0.265 -> 0.271, 0.293 -> 0.292) while recto-cutoff VOI is noisy
  (1.32 -> 1.92, 1.01 -> 1.23: one ridge gap anywhere merges whole windings). Side channel in this form: shelved.
- Other extractions on control seed 1 (box 48128, VOI): winding affinities (offset 1 merges everything, 4.0; offset 8
  "different" > 0.3: 1.91), seeded watershed on the recto ridges (2.4-4.2), recto cutoff 0.10 (1.34) plus a closing of the
  ridge mask along z and the in-plane tangent (perpendicular to the radial direction) by 2 label voxels: 1.24 (merge 1.08 ->
  0.85, ARAND 0.50 -> 0.63; 4 and 8 voxels over-split).

- Residual coarse blocks (plan A): `unet_cfg.enc_blocks` / `train --enc-blocks 0,0,2,3,3,3` adds pre-activation residual blocks
  after encoder level i >= 1: x_k = x_{k-1} + conv2(silu(gn1(conv1(silu(gn(x_{k-1})))))), gn(x_{k-1}) the previous block's
  output normalisation applied in conv1's staging (the down_norm input path); a2 holds the sum, its statistics taken after
  `nn_add_into` (new MX / 16-bit / fp32 elementwise add, fp4 with exact SR). The last block's output is the level's skip,
  down-conv input and bottom. Backward: block_bwd(pre, gid, save): the tensor-core chain backpropagates each block's input
  GroupNorm inside it, so the block before takes d/d a2 directly (pre) and the saved identity gradient (gres per level) is
  added after; the fp32 path adds it after the GroupNorm backward instead. Parameters appended last (header "enc_blocks");
  fresh blocks start as the identity (zero conv2) and take the level's output normalisation on --init-from: a net warm-started
  from verso3 with 7 blocks gives byte-identical logits. The coarse-skip offload is off with residual blocks. Desk mode
  (512^3): [0,0,1,1,1,1] 86.6 B/voxel / 15.7M params, [0,0,2,3,3,3] 88.8 / 24.7M, [0,0,3,5,5,5] 91.0 / 33.8M (78.5 / 11.0M
  without). test_unet (UFSM_ENC_BLOCKS, UFSM_RB_NONZERO: finite differences with non-zero residual weights; UFSM_BIG_ENC_BLOCKS
  tensor-core agreement), test_split case 37 (desk mode, levels 1-3 at 1/2/2). Found on the way: the fp4 weight memo keyed
  convs by (layer, conv, pass) and compared a scratch pointer on the backward-data path, so convs sharing a layer id read each
  other's prepared weights (here: enc[i].c2's backward used the zero residual conv2: zero gradients below every level);
  the key now carries a block id (nn_set_block, WMEMO_N 4096).
- Residual A/B (laptop, warm start from verso3 on the Paris4 band block, 3000 steps, cover x5, seeds 1 / 2, both arms the
  same recipe; `--enc-blocks 0,0,2,3,3,3` vs none): F1 box 48128 0.260 / 0.271 vs control 0.263 / 0.271, box 48512 0.293 /
  0.295 vs 0.294 / 0.294 (mean -0.001; the plan's bar was +0.01); FWHM 0.1-0.4 voxels wider; recto-cutoff band VOI lower in
  all four pairs (box 48128 1.255 / 1.265 vs 1.402 / 1.614, box 48512 0.839 / 0.986 vs 0.855 / 1.108: fewer merges);
  3.46 vs 3.75 samples/s (+8.5% step). Not adopted for the recto; the merge reduction is worth re-testing with a longer run.
- Residual blocks on the 2026-10-15 production chain (MX-fp4 gradients, three level-0 buffers): they had silently turned off the
  three-buffer layout and the coarse skip offload (both excluded any residual block: ~+9 B per level-0 voxel); the skip offload and
  the l3b coarse homes now hold each level's stage output (the last block's a2), only level-0 blocks stay excluded. A stale-record
  bug: the residual backward adds the identity gradient into d/d a2 after the GN backward wrote the fp4 weight gradient's gy records
  for it (on by default at >= 32 channels since the warp-specialised wgrad), so the next block's conv2 weight gradient missed the
  identity term (mean-of-16 error 0.91 vs ~0.2); lp_add_mx invalidates the records of its target. The forward sum rounds to nearest
  (nn_add_into_act; it was the only stochastically rounded forward store). P512, 7 levels 16..256 (trainer planner / device peak):
  none 12.7M params 36.5 B per voxel 4.56 GB; 0,0,1,1,1,1,1 19.4M 37.6 / 4.92; 0,1,2,3,3,3,3 32.8M 42.2 / 5.90;
  0,2,3,5,5,5,5 (ResEnc-L counts) 46.1M 45.5 / 6.73.
- ResEnc A/B on that chain (desk, production recipe: 2.4 um side labels, P512, 2000 steps, seed 1, --det 1; EMA validation loss at
  500 / 1000 / 1500 / 2000): none 0.6944 / 0.6698 / 0.6577 / 0.6505; 0,1,2,3,3,3,3 0.6946 / 0.6704 / 0.6591 / 0.6525;
  0,2,3,5,5,5,5 0.6917 / 0.6695 / 0.6580 / 0.6527. A tie: the same recipe's baseline run by another worker (identical arguments,
  the same losses to step 40, apart from step 60: atomic-order reductions) ended at 0.6539, so run-to-run noise alone is ~0.003.
  Not adopted (+16-25% memory, 2.6-3.6x the parameters).
- Winding phase (after upstream lasagna's cos target, made signed): `band_field_phase` (src/band.c) gives every voxel between
  two labelled consecutive rectos the fraction of the way outward from the recto it lies outward of to the next one (x252;
  0 on a recto, 126 midway, wrapping to 0 at the next recto). The midpoints are the neighbour pairs on opposite sides whose
  band is (nearly) equal (at a recto face the band jumps by a turn); with ds the distance to the nearest recto and dm to the
  nearest midpoint (+ half a voxel), phase = 0.5 ds / (ds + dm) on the recto side and 1 - that on the verso side (linear
  between parallel rectos); known only if ds + dm is within the flood radius and the nearest recto and midpoint lie on
  opposite sides of the voxel (not beyond the outermost / innermost labelled recto or past a trace's end). Concentric shells
  (tests/test_side.c): all voxels between the shells known, error 0.015 turns mean / 0.057 max. Holdout boxes: known on
  62% / 77% of CT voxels (the side: 65% / 79%), its half agrees with the side everywhere; cutting the reference phase at its
  wraps (6-neighbours > half a turn apart) within known voxels recovers the band instances (VOI 0.001 / 0.006); unknown
  voxels in the domain bridge windings (VOI 4.1), so predicted pieces are cut within confident voxels. ~10% on top of the
  band field. `train --side 2` (task band_affinity, B 1): two channels, soft BCE targets 0.5 + 0.5 cos / sin (2 pi phase)
  (nn_phase_targets), mask = known phase (CT <= --side-air unknown, default only CT 0: gaps are supervised); `ufsm band
  --phase`; score_sheets `--phase-channel C`: circular error, side accuracy of phase < 0.5, confidence, pieces at the wraps.
  The MX head kernels (forward fp32 / fp16 logits, weight gradient, backward data) now take up to 16 outputs (recto + 6
  affinities + 2 = 9): staged weights padded to 16, the tensor-core / CUDA-core weight gradients run once per 8 outputs
  (test_mx4: 9 and 16 outputs). The 16-bit (non-MX) 1^3 weight gradient still takes at most 8 outputs.
- Phase A/B (laptop, the residual A/B's recipe and seeds; control = its control arms; `--side 2 --head-map
  0,2,3,4,5,6,7,-1,-1`; 2.7-2.9 vs 3.75 samples/s, sampler-bound): recto unchanged (F1 box 48128 0.266 / 0.271 vs 0.263 /
  0.271, box 48512 0.291 / 0.292 vs 0.294 / 0.294; FWHM equal). The phase is learned: median error 0.071-0.076 / 0.063-0.066
  turns (mean 0.115-0.119 / 0.100-0.101), 60-68% of known voxels within 0.1 turn, its side (phase < 0.5) 76-77% / 79-80%
  correct (the side channel: 72% / 79% at this length); the training loss plateaus at ~0.56 from step 1000 (0.69 = no
  information, 0.39 = the targets' entropy). Sheet pieces, VOI box 48128 / 48512, mean of the two seeds (cutoffs chosen
  per box, as for every method): control recto cutoff 1.51 / 0.98; phase arm recto cutoff 1.68 / 1.02; phase wraps alone
  1.55 / 1.25 (merges where the predicted phase has defects: a stripe that ends lets the windings meet around the cut's
  end); recto ridge OR phase wraps within confidence >= 0.6 (score_sheets pieces_combo) 1.28 / 1.01; quality-guided
  unwrapping of the phase (confidence order, piece = floor of the unwrapped phase; prototype
  ~/ufsm-data/scratch-20261009/unwrap) 1.32 / 1.00 with the fewest merges (0.70-0.78 / 0.54-0.62) but more splits. Box 48128
  -15%, box 48512 +2%: short of the plan's -15% on both boxes. Not adopted yet; the unwrapping's splits (offset errors
  around defects) are the next target.
- Where the phase fails (seed 1): 4.5-6% of confident known voxels are > 0.35 turn off; there the reference windings are
  packed (spacing median 28-30 label voxels vs 42-46 overall) and the prediction cycles slower (median 0.62-0.68 of the
  reference rate); over the boxes it advances 12-15% less phase per voxel than the reference: it under-counts windings
  where sheets are compressed (the CT often shows no gap there), which is where pieces merge. Extraction variants on the
  same predictions all land at VOI ~1.28 / ~0.97-1.0: fragments (confident voxels minus recto >= 0.2 and wraps) labelled by
  the unwrapping's majority winding 1.28 / 0.97; Gaussian smoothing of the (cos, sin) vector before cutting / unwrapping
  (sigma 1-3) no better (unwrap 1.26-1.32 / 1.03-1.04; the combined cut loses confident voxels and gets worse).
- `train --phase-dense S` (lasagna's density weight as supervision thinning: a phase voxel whose windings are more than S
  label voxels apart is kept with probability S / spacing, position hash, no RNG; band_field_phase's optional spacing
  output): S 30 keeps ~65-70% of the phase voxels. Seed 1: phase error 0.119 / 0.102 turns, side 76.7% / 80.0%, F1 0.266 /
  0.291, combined-cut VOI 1.29 / 1.01, phase advance ratio 0.855 / 0.883 - identical to the unweighted arm. The packed
  windings are not under-weighted; they are not visible to the net in a 256^3 window.

## Unwrap pipeline: one network + our own winding fitter (2026-10-11)

Goal (user): our own single model does the whole unwrap from the published Paris4 wraps alone (no human annotations), with the
published 3-D ink as the ink teacher. Per scroll: a 3-D ink zarr, per-voxel winding numbers, one tifxyz per wrap, and ink
rendered on each wrap. 2.4 um first, then a 9.6 um version for the 8-9 um scans. Plan: ~/.claude/plans/bright-forging-seahorse.md.
The upstream villa spiral fit (spiral-fitting/, ~826M-parameter torch flow fit) works from sparse tracks plus normals, and in its
default mode from human patches / outer shell / winding-model rays; its winding inputs are relative (it counts from the
umbilicus = winding 0). Ours fits one scalar winding field to the network's dense fields (level sets never cross).

- `train --task unwrap` (B 1): output channels 0 recto (wrap labels, the usual loss), 1 ink (a source target named "ink",
  prob encoding, target slot 1; BCE over all papyrus, `--ink-lambda`, `--ink-dice`), 2-3 the winding phase (the --side 2 path),
  4-9 a sign-free normal n n^T and 10 the sheet density min(1, 32 rho), both from the phase gradient over +-4 voxels on the GPU
  (`nn_unwrap_targets`: the phase is symmetry-transformed as a scalar, so the gradient is in output coordinates; BCE,
  `--normal-lambda`, `--normal-step`). The sampler adds a papyrus mask (`batch.pm`, CT > 0) for the ink loss. Checkpoint
  `runtime_extra.unwrap` records the layout. tests/test_unwrap_targets (planar and radial phase fields: max target error 0.03).
  Smoke run (Paris4 smoke block, cover x13 = 7048 steps, P256, the dec_widths recipe + scale-cond, seed 1; 52 min,
  band-field bound at ~2 samples/s): final train losses recto bce 0.42 / dice 0.43, phase 0.51, ink 0.13, normal 0.55. Recto
  F1 / AUC on the holdout boxes 0.246 / 0.776 | 0.277 / 0.812 vs the recto-only control's 0.250 / 0.774 | 0.273 / 0.804: the
  extra heads cost nothing. `ufsm phase-raster` precomputes the band field phase over a box (blocks of 256 label voxels plus
  the band halo, in parallel, lossless volcomp; the smoke region in 8.5 min on 3 threads) and a band target's
  `"phase": {"root", "key"}` makes the sampler read it: batches byte-identical to the per-window computation (bench_sampler
  UFSM_BENCH_SIDE=2 hashes), 2.4 -> 13.8 samples/s (12 workers).
- Winding fitter `ufsm fit` (src/fit.c, src/nn_fit.cu): U = r / pitch(z) + offset(z) + delta (trilinear lattice), W = U -
  dir theta / 2 pi. Losses at field voxels: phase 1 - cos 2 pi (W - phi) (weight x phasor confidence), normal 1 - g^T T g /
  (|g|^2 tr T), density (log |g| - log rho)^2, non-degeneracy, lattice smoothness; grad W by central differences with the theta
  difference wrapped (no branch cut). Schedule: lattices 512 / 256 / 128 / 64 native with the gradient terms only (they carry
  no integer ambiguity and integrate the turn count), then a phase-unwrapping initialisation (per lattice node the residual
  phasor, corrections unwrapped node to node in coherence order) and the phase term at 64 / 32. Prior: pitch from the spacing
  field refined by sector coherence; direction from the unwrapped sector offsets per radial ring (they change by -dir per turn);
  when that vote is ambiguous both directions are fitted and the lower loss wins (wrong direction: 40x the loss).
  tests/test_fit (Archimedean spiral, tilted axis, z-varying pitch, a half-turn wobble, a 1.6x packed sector, a hole, dir +-1):
  100% winding ids, |error| < 0.08 turn everywhere; with the phase under-counting the packed sector by 15%: 99.3% ids.
- Oracle fit (`ufsm band --fit-fields`: the fields from the wrap labels' band field; Paris4 smoke sub-block 512^3 native at
  47616,16896,14848, fields at h = 8 native, --dir 1): winding ids right 99.2% of the 796k labelled voxels (band codes q mod 14
  turns), |error| median 0.029 turn (~1.5 native voxels), p90 0.082 (the codes' own quantisation is 1/18 turn). 18 s.
- `predict --fit-fields DIR` (unwrap checkpoints): every head in one pass, each window's interior pooled on the GPU into the
  fitter's fields (nn_fit_accum: material-weighted means of the phase vector, n n^T, density, material fraction, max recto).
  Network fields of the step-4000 smoke checkpoint on the same box: median spacing 58 vs the labels' 50 (the known
  under-count), phase confidence 0.69; fitted: winding ids right 64%, |error| median 0.083 turn. The final (step 7048)
  checkpoint: winding ids right 96.9%, |error| median 0.050 p90 0.154 turn (in-sample: the box lies in the training block).
  `ufsm fit --winding DIR` writes W * 256 as a uint16 zarr (volcomp typed chunks, exact; 128^3 at step 4: 800 KB).
- Whole smoke block (1792 x 1856 x 2048 native, fields 224 x 232 x 256 at h 8): the gradient-only stages now read fields pooled
  to ~4 field voxels per lattice cell (stages 0-3: 300 s -> 6 s; the fit 6.4 -> 1.5 min), and the phase stage starts from a
  voxel-level quality-guided unwrapping of U mod 1 = frac(phi + dir theta / 2 pi) (flooded in confidence order, half-turn
  continuity between 6-neighbours) instead of the lattice-node unwrapping. Oracle fields (`ufsm band --fit-fields
  --phase-raster`): 94.5% winding ids, limited by label coverage (known phase 24-49% of the field voxels in the end z-slabs vs
  81-83% in the middle; the wrong ids sit at large radius in the end slabs, where the fit bridges unlabelled stretches).
  Network fields (step-7048 checkpoint, predict --fit-fields, 75 s): 73.4% with the node unwrapping -> 90.6% with the voxel
  unwrapping; the wrong ids are mostly -1 (the fit under-counts, as the network's density does) and at the largest radii.
  `ufsm fit --model DIR` scores / exports a saved model, `--score-dump F` writes the per-voxel errors.
  The data terms now take grad W analytically at the voxel centre (trilinear delta gradient, the prior's and theta's z / y / x
  derivatives with the axis slope) on a random ~2M-voxel subset per step (nn_fit_loss2; knots fixed after the prior; the 7-point
  finite-difference kernel stays as --analytic 0): network fields 90.9% in 16 s (was 132 s), oracle 93.0% in 8 s (was 91 s).
- `ufsm wraps` (one tifxyz per wrap: rows every step native in z, columns every step / r_k in theta, each vertex the root of
  W = k along the ray from the umbilicus; tifxyz per upstream save_tifxyz: x/y/z float32, -1 invalid, meta.json scale / bbox /
  area / winding) and `ufsm render` (vertex grid to 1 px per native voxel, ink zarr max-composited over +-depth along the
  normal, plus the CT at t = 0) on src/tiffw.c (TIFF writer: uint8 / float32, deflate; tests/test_tiffw round trip). Oracle
  wraps 42-56 of the sub-block: vertices on the labelled sheets (median distance 0-2 native, p90 <= 3.5, 96-100% within 4
  voxels except the sparse edge wrap); the CT renders show papyrus fibre texture.
- Whole-scroll scaling. `predict --fit-zarr DIR` writes the same fields as 10 uint8 zarr arrays (lossless volcomp, shards of
  128 field voxels) plus fields.json (`"format": "zarr"`): each window's whole interior cells are accumulated in a small device
  buffer, finalized on the host into a slab of 128 field rows, and the slab is flushed when the windows leave it (needs the
  window stride, the shard and 128 h aligned to h; one window per shard). Smoke block: byte-identical to --fit-fields, 61 s (was
  75 s), 0.74 GB peak RSS, 72 MB (raw 127 MB). `nn_fit_accum` takes field cells up to 32^3 voxels (--fit-h 2..32).
  `ufsm fit DIR --slab Z [--overlap 256]` (src/fit.c fit_run_slabs): the prior from full-resolution slabs (knots on the global
  knot grid, offsets made continuous, the direction from the summed ring votes), a global gradient-only fit (lattices 512 /
  256 / 128) on fields pooled to ~64 native, then per z-slab of Z native (+- overlap) the voxel unwrapping and the phase stages
  at 64 / 32 started from the global model (delta = the global U minus the slab's base at the nodes). The slab's flood starts
  from every confident voxel below its interior, with the integers of the previous slab's model, so the unwrapping continues
  across slabs and the gauge carries over (a slab flooded from its own seed: 80.6% in its range, 3 turns off the gauge). The
  set (dir/global, dir/slabNNN, set.json) blends adjacent slab models linearly over +- overlap / 2; `fit --model`, `--winding`
  and `ufsm wraps` read sets (fit_set_W, fit_set_ray_root). Network fields of the smoke block: one box 90.88%, 2 slabs of 1024
  90.85%, 4 slabs of 512 (knots every 512) 90.93% vs 90.87% for one box at the same knots; wraps 40-52 from the 4-slab set have
  no seams (row-to-row steps at the slab boundaries: median 4.5 / p99 11.6 native vs 4.4 / 13.3 elsewhere). The fitter only
  loads one slab's full-resolution fields at a time (`fit_fields_load` rows [fz0, fz1), pooled, or geometry only); 1.6 GB peak.
- Grid evaluation of a set on the GPU (`fit_set_upload` / `fit_set_grid`, nn_fit_grid: per z row model a, optionally blended
  with b, as fit_set_W): `fit --winding` writes shard by shard and `fit --codes` streams label z-chunks (~32M voxels) through U on
  the GPU into 1/72- and 1/1024-turn histograms (mode, refined gauge, |error| quantiles; no per-voxel arrays, no whole-box code
  read). Smoke block, 4-slab set: score + winding 13.7 s -> 5.4 s, 1.4 -> 0.34 GB; winding volume vs the CPU writer: 0.05% of
  voxels differ, by 1/256 turn; score unchanged (0.9093). `ufsm wraps` runs rows on --threads (16) and finds each vertex's root
  scanning outward from the previous vertex's radius (the first ring of pitch / 4 intervals with a crossing; 24 bisections)
  instead of along the whole ray: wraps 40-52 6.8 -> 0.79 s (1 thread 4.0 s); neighbour steps > 3 grid steps 0.13% before and
  after. `--fields` reads raw or zarr fields, the data weight max-pooled to <= ~1 GB. `ufsm render` works in output tiles
  (--tile 512 px), each reading only the volume around its own vertices (+- depth), pixel rows over --threads: renders of the
  oracle wraps byte-identical with 64-px tiles, 43 MB peak. Whole-scroll estimates: wraps ~640k vertices/s (Paris4's ~1.4e9
  in ~40 min on 16 threads); render bound by decoding the ink zarr around each wrap.
- Field cell size: h 16 fields are 8x smaller (smoke block 9.1 vs 72 MB) but the fit drops to 44.9% winding ids (h 8: 90.9%):
  a 16-voxel cell spans ~1/3 turn in packed regions, blurring the phase and leaving neighbouring cells near the half-turn
  ambiguity. h 8 stays. A model-guided flood (neighbours stepped by the current model's predicted U difference instead of none)
  changes nothing (identical scores): the network's -1 errors are not unwrapping slips but turns missing from its phase in
  packed regions, so the lever is the network (longer / larger training: the desk run).
- 9.6 um version (Stage 6, smoke). `ufsm downsample <root> <key> <out-group> <out-um> --mode max|phase [--box]` halves an
  array (max: binary labels; phase: the circular mean of the known band phase codes of each 2^3 block, unknown where the
  resultant is under half their count) into an OME group that lists every level present. The smoke region's labels at 9.6 /
  19.2 / 38.4 / 76.8 (the sampler's occupancy index needs a level >= 3 under 4.3e9 cells) and phase at 9.6 (29 s). Sources:
  the stored phase takes `"level": L` (its CT-level grid; samples of level L read it directly, of L - 1 upsampled; without
  "level" the old behaviour: unwrap batches byte-identical, bench_sampler hash 1eccaa34d8e977fb); the unwrap density target is
  min(1, s rho) with `--density-scale s` (32 at 2.4 um, 8 at 9.6 um: the same physical code; recorded in runtime_extra.unwrap,
  read by predict's pooling). predict --fit-fields / --fit-zarr run at any --level (fields.json "um", the axis scaled to the
  level); the fitter's default lattices, knots and slab pooling are physical (2.4 um values x 2.4 / um); models save "um";
  fit --codes scales the label grid (code um / native um); wraps / winding default to the fields' or model's um.
  Smoke run (CT level 2, P128 = 1.2 mm, random draws at level 2, --normal-step 1 --density-scale 8, 6000 steps, 4 min, 25
  samples/s): predict --level 2 --fit-zarr --fit-h 2 (19.2 um cells) of the whole smoke block 5 s (2.4 um: 61 s); fit 94.6%
  winding ids, |error| median 0.045 p90 0.144 turn (2.4 um net: 90.9%, 0.058 / 0.207; both in-sample, the 9.6 net trained on
  the smoke block only); 2 slabs of 256: 94.5%. Wraps 40-52 (step 2) and ink renders (ink 9.6, CT level 2, depth 2) 2.3 s; the
  CT renders show papyrus fibres. The coarser net sees 4x the physical context per patch, which plausibly helps counting
  sheets in packed regions; whether winding fields from 9.6 beat 2.4 out of sample is open (needs band codes beyond the smoke
  block: on the desk).
- Out of sample (block B: native 49408,16576,14336 + 1792x1856x2048, the block above the smoke block in z; 8 band-code shards
  copied read-only from the desk, its stored phase 203 s on 4 threads, 6 GB): the oracle fit (band fields) 98.5% winding ids,
  |error| median 0.029 turn, so block B is a fair test; the smoke-trained nets 43.5% (2.4 um) and 43.7% (9.6 um): both
  overfit their single training block (their in-sample 90.9 / 94.6% say little). The networks need many blocks (the desk run on
  all of Paris4); a laptop check at 9.6 um trains on 8 more blocks spread over the scroll (prep_blocks_9p6.sh).
  8 blocks of 2x2x2 band-code shards (2048^3 native each, shard indices multiples of 4 in distinct z groups so no two blocks
  share a shard at any level; codes 457 MB copied read-only; stored phase ~185 s per block on 4 threads; 9.6 phase / labels by
  ufsm downsample) + the smoke block, 9.6 um, P128: 12k steps (8 min) 33.3% on B, 45.7% on A; 48k steps (33 min) 83.7% on B
  (out of sample), 54.2% on A (A's oracle is 94.5%: thin labels). Training length is the lever. The phase BCE has an entropy
  floor of 0.386 nats (soft cos / sin targets), so the training phase loss (0.40-0.45) says little; judge by fits. Network vs
  oracle fields on B (12k net): phase |error| median 0.087 turn, 14% of cells > 1/4 turn off (35% where the sheets are < 100 um
  apart, 8-12% elsewhere; 32% at network phase confidence < 0.2, 4% above 0.6), spacing ratio median 1.04 (p10-p90 0.73-1.51),
  normals median 5 degrees. Fitter knobs do not rescue bad phase: gradient-only fits 27% (oracle fields: 53%, so the phase is
  essential), node unwrapping 41%, flood confidence thresholds 0.2-0.6 no gain, a model-referenced phase-consistency quality
  map +3 to +7 points (not kept).
- Wrap placement on block B (9-block 48k 9.6 um net, out of sample): vertex distance to the nearest labelled sheet median 2.0
  native voxels, p75 4.5, p90 11.5; 72.7% within 4, 84.9% within 8, 6.8% beyond 16 (labels cover 25 segment groups, so part of
  the tail is unlabelled sheet). `ufsm wraps --snap ROOT KEY [--snap-range 0.35] [--snap-min 0.5]` moves each vertex along its
  ray to the recto maximum within +- range pitches (the predict output; origin_zyx from its attributes; tiles of 64 x 512
  vertices read only their box) and median-filters the displacements over 5 x 5 vertices: no gain with the 9.6 um recto
  (within 4: 71.7%); kept opt-in for 2.4 um recto. `ufsm wrapdist <wraps> <labels-root> <labels-key>` measures this (the
  labels' key is their um; z-chunks with --max rows of overlap, exact up to --max) on src/edt.c, an exact threaded distance
  transform (Felzenszwalb-Huttenlocher, float32 squared distances, 4 B/voxel; tests/test_edt vs brute force; 256^3 in 0.15 s):
  block B 12.7 s, 1.35 GB, the same numbers as the scipy analysis it replaces (30+ min, up to 23 GB; heavy compute stays out
  of Python).
- Laptop memory: three WSL VM restarts (2026-10-10/11/12) were OOM kills of my own jobs with RAM and the 16 GB swap full (a
  scipy EDT over 851M voxels, `ufsm band` on a whole block with 16 threads (27 GB), a 23 GB ufsm job); Windows then restarts the
  24 GB VM. Heavy laptop jobs now run in a systemd user slice capped at 18 GB + 2 GB swap (~/ufsm-data/capped.sh), and
  `ufsm band` refuses boxes needing more than 80% of the memory it can get (24 B per label voxel vs MemAvailable and the cgroup
  limit), pointing at phase-raster + band --phase-raster.
- 2.4 um vs 9.6 um winding nets on the laptop budget: a 2.4 um net on the 8 blocks (P256, ink head off, 30k steps, 2.0 h,
  4.2 samples/s) 51.2% on B / 46.9% on A, |error| median 0.19 turn, still far from converged (a P256 patch at 2.4 um holds 1/8
  of the physical volume of a P128 patch at 9.6 um); the 9.6 um net (9 blocks, 48k steps, 33 min) 83.7% / 54.2%. Winding
  fields come from a 9.6 um net; recto and ink stay at 2.4 um. `ufsm wraps --out-um 2.4` writes a 9.6 um fit's wraps on the
  2.4 um grid (voxel centres to voxel centres, grid step and voxel_size_um scaled) so they render with the 2.4 um ink
  (wrapdist unchanged: 72.2 vs 72.7% within 9.6 um).
- `ufsm fit-compare <reference-fields> <fields>` (any native voxel sizes covering the same physical cells; histograms; block B
  in 9 s, 0.33 GB): vs the oracle on B, the 9.6 um net (9 blocks, 48k) phase |error| median 0.069 turn, 12.1% > 1/4 turn
  (29% where sheets are < 100 um apart), confidence 0.73; the 2.4 um net (8 blocks, 30k) 0.207 turn, 41% > 1/4, confidence
  0.04 (its phase head still outputs ~0.5 everywhere); both: spacing ratio ~1.0 (p10-p90 ~0.7-1.4), normals ~5 degrees.
- test_split's "memory layout vs separate buffers" check failed twice under GPU contention (cases 34, 35; grad rel ~0.3) and
  passed alone. It took the best of three tries, assuming run-to-run noise is rare; it now also repeats the separate-buffer
  runs and passes a try whose layout difference is within 1.5x of that repeat difference (a layout error differs where
  repeats agree). Open: case 35 (lean 1, fp4 grads, offloads) is two-state: identical runs, split or single GPU, sometimes
  differ by grad rel 0.29-0.35 (other cases repeat to 1e-8). Either a flipped stochastic rounding spreading into an equally
  valid MX-fp4 sample, or a race in that path: compare both outcomes with an fp32 reference gradient.
- 24 blocks (8 + 16 more, same selection rules), 9.6 um, 96k steps (66 min, resumed once after the laptop's standby): fields
  as good as the 9-block net's (phase gross errors 12.3%, normals 4.5 degrees, spacing ratio p10-p90 0.72-1.39) but B 53.5%,
  A 46.7%. `fit --score-detail` (integer error -2..+2 by z octile and by 0.5 mm radius ring) shows why: the errors are radial
  and systematic, +1 / +2 turns inside ~2.5 mm and -1 / -2 beyond ~5.5 mm for both nets (the 9-block net's correct band is
  just wider): the network phase misses ~3-4 turns across the block's 7 mm radial span, mostly in packed regions (29% gross
  phase errors where the sheets are < 100 um apart, 5-10 voxels per turn at 9.6 um), and the fit follows it. A region-vote
  unwrapping (regions along model-checked reliable steps, offsets decided by weighted votes over whole region boundaries,
  merged in order of decisiveness) changed nothing (82.2 / 53.6%; not kept): the errors are not path slips.
  A 4.8 um winding net (24 blocks, P128 = 614 um, level-1 draws, normal step 2, density scale 16, 96k steps, 66 min): B 43.9%,
  A 41.8%, phase gross errors 25.5%, confidence 0.14: undertrained like the 2.4 um net, with the same radial under-count. Down-
  weighting the phase where the fields' spacing is packed (ramp to 0.1 below a threshold, those voxels out of the integer
  unwrapping, so density and normals count the turns there): no gain at 100 / 140 um, worse at 180 um (B 83.7 -> 53.2% for the
  9-block net; not kept): the network density under-counts in the same places. Three fitter-side remedies failed, so the
  winding accuracy is bounded by the network's view of packed sheets; the lever is the network (desk-scale data and steps,
  and losses that weight packed regions).
- Papyrus / void net as a sheet counter (`ufsm gap-compare <papyrus> <key> <reference-fields> [--net FIELDS]`: void -> papyrus
  entries along each cell's radial ray, +-115 um, vs the reference density, by reference spacing; native, 17 s): on block B
  with the full-res papyrus net (2.4 um, cutoff 80) the gap count is 1.04 x the true density where sheets are < 100 um apart,
  where the 9.6 um winding net is at 0.72 (its missing turns); but 1.44-2.16 x in loose regions (each sheet's own layers and
  cracks read as gaps), where the net is 0.94-1.48. Cutoff 110 is similar. `ufsm gap-density` fuses them (the gaps' spacing
  where the fields' is under --fuse-um and the gaps' is smaller; packed 0.72 -> 1.01 at 150 um) and `fit --packed-um T`
  lowers the phase weight (to 0.1 at 0.6 T) where the fields' spacing is packed and keeps those voxels out of the integer
  unwrapping: together they halve the radial turn drift on B (+1 at 0.5-1 mm 56 -> 34%, -2 beyond 6 mm 10-23% -> 0) but the
  score moves only 83.7 -> 84.05% and some settings flip the whole-block alignment (~54%). Better use: the papyrus net's
  2.4 um gaps as an input channel of the 9.6 um winding net, which sees CT averaged over 4^3 voxels where packed gaps vanish.
  gap-compare on raw CT (--um, physical mapping to any grid): packed-sheet gap counts 1.24 at 2.4 um (noise adds gaps), 0.82 at
  9.6 um, vs the papyrus net's 1.02: the 9.6 um CT has lost about a fifth of the packed gaps before any network sees it.
- Aux input channel (`train --aux-input 1`, default off; default batches byte-identical, bench hash 1eccaa34d8e977fb): a fifth
  network input from each source's `"aux": {"root", "key", "level"}` volume, written by the sampler with the CT's symmetry
  (v / 127.5 - 1), cin 5 in the checkpoint; `predict --aux ROOT KEY` reads it per window (5-input checkpoints need it; one GPU, no
  split; the fused CT stem needs cin 4, so the generic input path). `ufsm void-pool` builds it: per 9.6 um voxel the fraction of
  its 4^3 2.4 um voxels the papyrus net calls void, merged box by box into a whole-scroll array (32-row source slabs, 0.85 GB,
  10 s per 2048^3 block). The papyrus net (F-full, cutoff 80) over the 24 training blocks + A + B: 1.5 GB at 9.6 um; voxels 44%
  nearly all papyrus, 40% nearly all void, 16% partly void (the sub-voxel gaps). A 96k-step 9.6 um run on it is the test; it is
  not deployable for 9 um-only scans (it needs a 2.4 um scan), so it measures how much a 9.6 um student could gain from 2.4 um
  information (the plan: a 2.4 um teacher on the desk, distilled to 9.6 um).
- Aux input result (24 blocks, 96k steps, papyrus void fraction as the fifth input): B 42.1%, A 69.8% winding ids (baseline
  53.5% / 46.7%; the fitter flips between two states, so these move with it), but packed-region gross phase errors are unchanged
  (30.2% where sheets are < 100 um apart, baseline 29%) and so is the radial drift: the void-fraction input does not fix the phase.
- Conditioned inference (2026-10-12, user: keep to in-window relative reasoning, no cascade or whole-scroll net). A window knows
  winding only up to an integer; the fitter stitches windows globally and drifts. Instead windows run in sequence and each gets
  the earlier windows' winding where they reached it, so one gauge is carried across the box.
  - Ground truth beyond one turn: the published band codes are q mod 14 at 1/18 turn (q = sheet + theta / 2 pi, the mesh
    coordinate); per voxel q mod 14 = band - 9 + phase / 14 steps (band.h band_winding; `phase-raster --winding 1`). On B it
    agrees with the stored 1-turn phase as frac(q - phase - theta / 2 pi) ~ 0: mean 0.047 turn (the codes' 1/18-turn steps).
  - `train --cond 1` (task unwrap): inputs 4-6 = cos, sin of 2 pi q' / 14 and a mask (the stem takes at most 8 inputs);
    outputs 11-14 = cos / sin of 2 pi q' / 14 and 2 pi q' / 2, q' = q + a random integer gauge per sample. The sampler draws the
    condition in the output frame: none (10%, the gauge heads unsupervised), a 4-12 voxel seed cube (15%), else face slabs
    (each axis with p 0.5, thickness P/8..P/2, low face 75%), dropout cubes, smooth 0.03-turn angle noise. Sources: "winding"
    next to the stored phase. Warm start: `--init-from --head-map` now grows the inputs (unet_init_from_map: the stem's new
    input weights zero, so the net computes the donor's function; test_checkpoint).
  - `predict --cond-sweep DIR` (src/sweep.c): windows in one z, y, x raster (W 128, halo 32 at 9.6 um), a device canvas of the
    box (14-turn phasor, 1-turn phase, recto, done: 6 B / voxel), the condition gathered from it, interiors placed into it. Seed:
    the first window's most confident papyrus voxel, a 4^3 cube at gauge 0 from its phase + theta / 2 pi, then that window again,
    conditioned. Then q mod 14 = the 14-turn phasor snapped to phase + dir theta / 2 pi, flood-unwrapped from the seed (period 14,
    16 confidence buckets) into U.f32. `ufsm wind-score` scores it exactly as `fit --codes` (the scoring is now one function,
    score_codes, fed by the fitter or by a raster; byte-identical fit output). test_sweep: a 26-turn spiral with +-0.35 turn of
    noise on the 14-turn phasor unwraps to within 0.0014 turn.
  - Result (desk, 24 blocks, warm start from the 96k laptop net, 24k steps, ~24 samples/s per 5060 Ti): punted. The sweep scored
    chance on B and A (7.4% / 7.5% ids right, integer errors uniform; within a window only 24% of voxels share one gauge), the
    fitter path unchanged (B 53.1% vs the control's 53.2%). The 14-turn head alone (`--cond-raw`) is chance too: supervised
    inside the condition as well, it learned to copy the input there (its loss fell to 0.377, like the phase head's) and never
    to count; its turn accuracy outside the condition was 0.10-0.14 (chance 0.07). With `--cond-loss outside` (supervision only
    where the condition does not reach; the trainer logs that turn accuracy as "acc") two runs stayed at chance through 13k
    steps (0.06-0.08) while the loss fell (the head hedges towards 0.5): counting sheets 3-6 turns from a condition is not
    learned in this form at 9.6 um. Untried: accuracy binned by distance from the condition (is even near-boundary copying
    learned?) and thin-shell extension (each window adds about one sheet, sweeps with small strides).
- Papyrus gap labels, no cutoff (`train --task papyrus --pap-local R`, default off; src/papyrus.h papyrus_local). A random
  cutoff without conditioning only teaches P(s > t) averaged over t, a smoothed CT (one effective cutoff, ~89); conditioning
  makes the cutoff a dial. The user: a voxel is papyrus or not, the cutoffs only exist because the CT is not bimodal. Rule:
  void where the CT smoothed by sigma is <= t_lo (open void is dark), or in the lower half of a gap: the closing (min of the
  max over a (2R+1)^3 window of the CT smoothed by local_sigma) fills only dark features with brighter material around them,
  so where closing - floor >= local_depth the threshold is the gap's half depth (blur puts a boundary at half the step),
  clamped to [t_lo, t_hi]; a step or a faint plateau (bright on one side only) is not filled and keeps t_lo. A plain local
  max / min (first try) called faint papyrus beside a bright sheet void. Ignored: local_margin x depth around the half depth
  in gaps, and +-band around t. Running max / min: van Herk / Gil-Werman, separable, 3 comparisons per voxel; tests against
  brute force. Block B (6.8 Gvox, `ufsm papyrus-labels`, 2 min on 8 threads), R 8, sigmas 1 / 1.5, depth 30, margin 0.1,
  cut 55,128, band 5: gap-compare packed (< 100 um) density 1.08 (cutoff 55: 0.72, 80: 1.04), loose over-count 1.5-2.35
  (cutoff 55: 1.0-1.55): the gap labels also find splits and pores inside sheets, right for a material reading, wrong for
  counting sheets. With t_lo 50 open air (up to ~50 here) left papyrus specks; 55 plus band 5 leaves them ignored. 28% of the
  recto label band on B lies in void by these labels (the band straddles the face). `ufsm overlay` draws a CT plane beside
  label panels as a PNG (zlib). Training A/B against the cutoff-80 net: not run yet (user hold).
- Papyrus head in the unified model (`train --task unwrap --pap-head 1`, default off): one more output channel after the others
  (11 without --cond), the gap labels of each window's clean CT computed by the sampler before the scanner augmentation (B's
  settings by default: cut 55,128, band 5, local 8, depth 30, margin 0.1), the CT's symmetry; BCE over voxels inside the scroll
  (CT > 0) not ignored, times --pap-lambda. The net gets no cutoff and gives one papyrus / void reading. Laptop smoke (2.4 um, 8
  blocks, warm start from the 30k net, head fresh): papyrus bce 1.37 -> 0.10 in 400 steps, the other heads unchanged.
  predict --channel 11 writes it; --fit-fields accepts any unwrap checkpoint with 11+ outputs.
- `ufsm phase-raster --pool 2` writes only the 2x circular-mean pooled phase (<out>/9.6, shards of 256: one per 4.8 um shard
  of 512); `--resume 1` skips written shards. A whole-scroll Paris4 phase at 4.8 um would be ~326 GB (25 MB per shard, 13054
  shards; the desk has 317 GB free), at 9.6 um ~65 GB. Block B: identical known / unknown masks to phase-raster + downsample,
  1.2% of voxels differing by exactly 1 code (float lookup tables vs double trigonometry at rounding boundaries).
- Slab windows (2026-10-13; user: "what if we didnt do cubes"): the windings change across y / x, while z mostly repeats the same
  wrap, so `train --Pz N` draws N x P x P windows (N <= P, a multiple of 2^(nlev-1): 32 for 6 levels, the coarsest level then 1
  plane deep) and `predict --window-z N --halo-z H` (default N / 4) tiles with them. The convolutions already take any depth (the
  --split z halves); the sampler, trainer and the predict input / placement / fit-pooling kernels gained the depth. Slabs: pyramid
  targets at levels >= 1 with a stored phase, --zfix 1 symmetries, no regions, cover, band affinities, aux, cond, papyrus labels,
  dilation, CT / scanner / geometry augmentation, --split or --half-res; predict on one GPU's plain path. Cube windows unchanged:
  bench_sampler hashes (augment 0 / 1 / 3, phase, 16-bit chamfer, default recipe) and predict's fit fields and shards byte-identical.
  Laptop: 32 x 256 x 256 at 9.6 um, a 1.5M net (widths 32,32,32,32,64,64), 36 samples/s. A 6-level net sees ~480 voxels (9.6 um
  ~4.6 mm, 19.2 um ~9 mm), so wider slabs need xy-only pooling levels (not built). A net trained on slabs is run on slabs: its
  coarsest z taps only ever saw padding.
- Counting turns (2026-10-13): `fit-compare` also counts turns on 0.5 mm radial segments without the fitter (whose winding-id
  score swung 5-20 points between inference windows of one weak net). On block B every net so far (2.4 um teacher, 9.6 / 19.2 um,
  cubes, slabs, 7 levels, 1.5M to 10.7M) misses 0.4-0.6 of ~3.4 turns per segment and gets 94-99% of the packed (< 100 um)
  segments wrong; best: 19.2 um input, 10.7M, 60k (30.7% of segments off by >= 1/2 turn; 9.6 um 33.1%). The density head could
  not help: its target min(1, scale rho) clips every spacing under ~77 um. `train --density-code 1` targets x / (1 + x) (x = scale
  rho; recorded as "density_code", decoded by predict) and `--packed-weight W --packed-um S` weighs the phase and normal / density
  losses 1 at a spacing of S or more up to W at S / 2 (the loss mask byte is a per-voxel weight: 0/1 masks are unchanged).
- Generalising to other scrolls (2026-10-13; the model must work from 8-9 um scans only). The targets (0343 / 0175A / 1447 at
  8.64 um, 116 keV, 1.2 m; 0139 / 0813 at 9.36 um, 113 keV) are ~2x smoother than Paris4 at 9.6 um: per box, relative to the
  material sd, high-frequency noise 0.19 vs 0.35, Laplacian energy 0.68 vs 1.39, no phase-contrast halos (the 2.4 um Paris4
  reconstruction ends with an unsharp mask). Train's blur + low-res at strength 0.5 on Paris4 matches them (0.18 / 0.63). Scanner
  augmentation, same training otherwise (+140k steps from a 60k net), on block B degraded at strength 2: miscounted 0.5 mm segments
  68.3% -> 54.6%, phase median error 0.115 -> 0.088; clean unchanged (31.5 / 31.9%). Added: per-op strength (`--aug-OP-s`); ops
  `paganin` (per-axis two-sided exponential low-pass, the 1 / (1 + c f^2) phase-retrieval form), `gin` (random shallow conv
  remap, positive-biased, rescaled to the window's material mean / sd, blended) and `hist` (the material histogram mapped onto a
  target scroll's, `--aug-hist FILE` from `ufsm ct-hist`); geometric augmentation (zoom, rotation, elastic, affine) for stored-phase
  unwrap samples: the phase raster is pulled through the same warp with phasor interpolation (an analytic ramp: < 0.004 turn).
  Existing recipes give byte-identical batches. Recipe: blur .6 (strength 1.5), lowres .4, paganin .3, tone .3, gin .15, hist .15,
  bias .15, noise .15, cnoise / ring / slice / quant / cupping .1, sharpen .03; --geometry-aug 1, zoom .3 down to 0.85, rotate .2,
  elastic .15. Inference on 8.64 / 9.36 um scans: area-resample to the net grid.
- Left: BigTIFF for renders over 4 GB compressed; 9.6 um training on all of Paris4 (desk band codes); the desk runs.

## Half-resolution networks (2026-10-11)

- `train --half-res pool|pack|pool4|pack4` (task surface; checkpoint header `"half_res":1-4`, absent = full resolution):
  the U-Net runs on 2^3 (pool, pack) or 4^3 (pool4, pack4) blocks of the window, the loss stays at full resolution
  (fp32 logits mapped up, their gradient mapped back by the exact adjoint). pool: input = the block means of the 4 input
  channels, logits upsampled 2x trilinear (fine voxel f samples coarse (f + 0.5) / 2 - 0.5, edge-clamped). pack: 8 input
  channels = the 4 means + the CT block's Haar details along z, y, x ((upper half - lower half) / 8) and its mean absolute
  deviation (the stem kernels take at most 8 inputs, so not all 16 sub-voxel values), and 8 logits per output channel, one
  per sub-voxel (channel 8c + k, k = 4dz + 2dy + dx; no interpolation). pool4 / pack4 put a 2x pool in front of the pool /
  pack stage (pack4's sub-voxel logits are at 2x, then upsampled 2x trilinear). Kernels nn_half_input / nn_half_output /
  nn_half_grad and the chained nn_lowres_* (src/nn_elem.cu); predict runs the net on W / div (the window must be divisible
  by div * 2^(levels-1); no --split, sheet task or host path for these nets). The MX head weight gradient now takes any
  multiple of 32 inputs in blocks of 32 (a 64-wide level 0), and the stem's cooperative weight gradient any multiple of 32
  outputs. Tests: tests/test_half.c (layouts, host trilinear reference, adjoints, quarter chains, checkpoint), test_mx4
  (head 16-96 inputs, stem 4 / 8 -> 64 / 96).
- A/B (laptop, the dec_widths A/B's recipe: Paris4 band block cover x13 = 7048 steps, P256 B1, Muon 0.01 WSD, soft 7 um,
  seeds 1-2; control = 32-192 with `--dec-widths 0,32`; pool / pack = 64-192 at 4.8 um; pool4 / pack4 = 96-192 at 9.6 um).
  F1 (seeds 1 / 2, mean) and AUC per holdout box 48128 | 48512; ridge F1 at tolerance 1 (label grid, after NMS); FWHM in
  2.4 um voxels; train wall time (laptop, sampler incl.):

  | arm | F1 48128 | F1 48512 | AUC | ridge F1 tol 1 | FWHM | train |
  |---|---|---|---|---|---|---|
  | control | 0.252 / 0.247 (0.250) | 0.272 / 0.273 (0.273) | 0.771-0.776 / 0.803-0.804 | 0.386 / 0.424 | 16.2 / 18.0 | 1557 s |
  | pool | 0.254 / 0.242 (0.248) | 0.277 / 0.278 (0.278) | 0.769-0.779 / 0.804-0.806 | 0.389 / 0.440 | 16.2 / 17.9 | 700 s |
  | pack | 0.250 / 0.250 (0.250) | 0.271 / 0.281 (0.276) | 0.770-0.780 / 0.803-0.807 | 0.388 / 0.440 | 16.2 / 18.1 | 705 s |
  | pool4 | 0.244 / 0.243 (0.244) | 0.273 / 0.264 (0.269) | 0.772-0.773 / 0.796-0.805 | 0.381 / 0.425 | 16.4 / 18.1 | 325 s |
  | pack4 | 0.255 / 0.236 (0.246) | 0.272 / 0.268 (0.270) | 0.758-0.781 / 0.801 | 0.388 / 0.428 | 16.4 / 18.1 | 327 s |

  At 2x nothing is lost (mean dF1 -0.002 / +0.005 pool, 0.000 / +0.004 pack; ridge F1 tol 1 at or above the control; the
  ridge width and double-peak rate (0.037-0.039 / 0.021-0.022) equal: the sub-voxel head shows no block pattern), at ~2.2x
  the training speed. At 4x the mean F1 drops 0.003-0.006 on both boxes (pool4 -0.006 / -0.004, pack4 -0.004 / -0.003),
  at ~4.8x; the 4x runs were sampler-bound on the laptop (40% wait, ~21 samples/s; now fed, see Sampler CPU). pack's
  sub-voxel CT detail is not measurably better than pool at 2x or 4x. The ridge width is set by the labels (jittered traces,
  soft 7 um targets), not by the network's grid. GPU ms per training step (UFSM_PROF, idle laptop GPU, 100 steps without a
  cover) control / pool / pack / pool4 / pack4: P256 211 / 99 / 99 / 39 / 38, P384 673 / 297 / 305 / 96 / 98 (2.1-2.3x and
  5.4-7x); wall samples/s P256 4.7 / 9.9 / 9.9 / 24 / 23, P384 1.42 / 3.3 / 3.2 / 7-9 / 7.5-8.5 (the 4x nets outran the
  sampler at ~400 M voxels/s; fixed, Sampler CPU below). predict W544 halo 16, forward GPU ms per window: 540 / 225 / 225 /
  65 / 63.
- Recommendation: `--half-res pool --widths 64,96,128,160,192` for new runs (fresh net; not on resume of a full-resolution
  checkpoint). 4x only where speed matters more than ~0.005 F1. Not deployed on the desk (the user's call).

## Sampler CPU (2026-10-11)

The 4x nets outran the sampler (pool4 cover recipe: ~21.5 samples/s, 40% wait). CPU per P256 patch (bench_sampler, cover
recipe, 16 workers, ms summed over workers) 586 -> ~295, standalone 28.4 -> 55.8 samples/s, batches byte-identical:
- label read (`z3_read_label_grid`): an x index map, and the 2^shift output rows / planes that read one label row are copies;
  target_erode1 branch-free; the empty-patch maximum skipped under a cover. Decoded volcomp mask chunks (binary labels: a
  serial arithmetic coder, ~10 ms per 128^3 chunk, 8 chunks per window) are kept in memory as bits, process-wide, least
  recently used dropped past `UFSM_Z3_MASK_CACHE_MB` (default 1024 = 4096 chunks; 0 off). targets 170 -> ~30 ms.
- soft targets: the 3-4-5 chamfer in bytes saturated at 255 (exact below 255; used when the soft cutoff and the trust band
  are below 255, else the 16-bit map) with saturating byte adds and log-step row scans: 125 -> ~65 ms.
- input channels (`write_x`): per output row in vectorised loops (no math errno for sqrtf) and F16C conversion (the
  _Float16 cast loop stayed scalar); write_x_aug only for the appearance plan / sheet prior: 154 -> ~61 ms.
- CT statistics as integer partial sums, encode / trust / mask loops branch-free: 55 -> ~25 ms together.
Byte identity: tests/bench_sampler `UFSM_BENCH_HASH=1` against the previous build (cover recipe 404 batches; training recipe
with augment 0-3; the default recipe deterministic; soft 80 um (16-bit chamfer) and no soft). Training (laptop): pool4 /
pack4 at P256 and P384 now wait 0% (cover recipe 26-27 samples/s; P384 pool4 7-9 -> 9.45 samples/s). Left: soft ~65, input
~61, CT read ~56 (whole 128^3 chunks decoded for a window on the 32-voxel grid; volcomp substreams would allow less).

## Papyrus / void network (2026-10-11)

User request: a small conv3d net that separates papyrus (foreground) from void / air inside the scroll (the outside is masked
to 0 already). Low CT means void, but there is no exact cutoff (Paris4 ~80), so train on random cutoffs t in [50, 128].
- `train --task papyrus` (src/papyrus.h, header-only; hooks in sample.c / train.c / predict.c, existing tasks byte-identical:
  bench_sampler hashes of the x13 cover's training recipe, augment 1, no soft and the default recipe). The sampler reads no
  targets: per native patch it draws t (`--pap-cut 50,128`; `--pap-rel LO,HI --pap-rel-p P`: Otsu of the patch x U(LO, HI)),
  smooths the CLEAN CT (`--pap-sigma 1`, separable Gaussian streamed through a ring of 7 planes: ~70 ms at P256) and writes
  channel 0 = 255 sigmoid((s - t) / tau), tau ~ U(`--pap-tau 2,8`) raw units (hard s > t without --pap-tau; `--pap-band B`
  ignores |s - t| < B; `--pap-clean N` flips 6-connected specks / holes below N voxels). Then the scanner augmentation
  corrupts only the input (recipe: noise .3, cnoise .15, blur .2, lowres .15, sharpen .1, ring .15, slice .15, quant .1;
  no tone / bias / cupping, which would move the cutoff). CT 0 stays out of the loss; predict writes 0 there.
- Conditioning on t (`--pap-cond`): 2 (recommended) z-scores the CT channel around the cutoff, (ct - t) / window sd (no
  offset jitter), so the cutoff is the input's zero; predict `--papyrus-cut T` passes T as the window "mean" (the CT stem is
  unchanged). 1 adds (t - mean) / sd to the stem bias through the --scale-cond vector (`unet_set_cond`): learned too slowly
  (train loss 0.33-0.36 flat from step 200, val 0.445 at 1000 = the unconditioned arm's; stopped). 0 trains the marginal
  P(s > t) over t, read at p > (t - 50) / 78. Checkpoint tag `"task":"papyrus","papyrus":{..."cond":2}`.
- Data (holdout sub-box 256x512x512): material mean 87, sd 36, percentiles 5/25/50/75/95 = 37/49/90/117/141, void mode 40-49,
  histogram minimum 60-69, Otsu 84; CT - G1(CT) residual sd 3.65 (the scan is clean); BCE floor of the soft targets 0.14
  (an ideal unconditioned predictor: 0.36). Cleanup flips < 0.04% (N 64) / < 0.15% (N 512) of the voxels at t 50-110: off.
- Evaluation without human labels (`tools/papyrus_eval.py`, holdout boxes 48128 384x1024^2 and 48512 512x1024^2, predict
  window 416 / halo 16): IoU with the reference label (G1 CT > t), papyrus fraction, recall of the training-4p8 recto voxels,
  CT 0 called void (1.000 by construction), per-block fraction (2x4x4 blocks) vs the reference, mid-plane PNG grids.
  Arms (16,32,64,96 = 1.3M params full resolution, or --half-res pool 32,64,96,128 = 3.1M at 4.8 um; Muon 0.01 WSD,
  P256 B1, soft tau 2-8, sigma 1, --dice 0, MX-fp4 body, 16-bit or MX-fp8 stem input), IoU box 48128 at t = 50 / 65 / 80 /
  95 / 110 / 128 (box 48512 within 0.01):

  | arm | steps | val loss | IoU | papyrus fraction at t 80 (ref 0.587) |
  |---|---|---|---|---|
  | raw CT > t (no smoothing) | - | - | 0.972 / 0.973 / 0.964 / 0.942 / 0.903 / 0.810 | 0.575 |
  | B marginal, read at (t - 50) / 78 | 2738 | 0.434 | 0.763 / 0.943 / 0.883 / 0.832 / 0.791 / 0.000 | 0.521 |
  | A2 cond 2, full res | 2738 | 0.158 | 0.972 / 0.991 / 0.977 / 0.977 / 0.958 / 0.936 | 0.573 |
  | E cond 2, half-res pool | 2738 | 0.157 | 0.966 / 0.986 / 0.983 / 0.974 / 0.949 / 0.910 | 0.578 |
  | F-pool cond 2, half-res pool | 7048 | 0.153 | 0.981 / 0.992 / 0.991 / 0.984 / 0.973 / 0.944 | 0.585 |
  | F-full cond 2, full res | 7048 | 0.153 | 0.984 / 0.993 / 0.993 / 0.990 / 0.986 / 0.971 | 0.584 |

  Recto-label recall follows the reference's within 0.012 (F-pool box 48128 t 65 / 80 / 95: 0.768 / 0.676 / 0.542; the label
  band straddles the face, so the reference itself reaches 0.771 / 0.678 / 0.545; F-full 0.767 / 0.675 / 0.543); per block (2x4x4) the papyrus fraction is
  within 0.0014-0.0034 of the reference's at t >= 65 (0.011-0.014 at 50); the block-to-block spread (sd 0.017-0.099) is the
  reference's own, i.e. real variation, not instability. Predict (idle laptop GPU, window 544 / halo 16, 512^3 shards, a
  shard-aligned 512x1024^2 box = 537 M voxels, one window per shard): F-full 1.33 s GPU (~400 M voxels/s), 1.65 s wall;
  F-pool 0.43 s GPU (~1.25 G voxels/s), ~0.8 s wall (reads dominate); Paris4's ~15 T occupied voxels ~10 h vs ~3.3 h of GPU.
  Training (x13 cover, shared GPU): F-full 1539 s (5.7 samples/s, GPU-bound, 6 workers), F-pool 689 s (10-11, sampler-bound,
  8 workers). The papyrus sampler costs ~95 ms per P256 patch for the labels (smoothing ~70); batches are deterministic
  across worker counts (bench_sampler UFSM_BENCH_PAPYRUS hash, 4 and 10 workers).
- Recommendation: F-pool (`--half-res pool --widths 32,64,96,128 --pap-cond 2 --pap-tau 2,8 --pap-sigma 1 --dice 0`) for
  whole-scroll runs: 3x faster inference, within 0.007 IoU of full resolution at cutoffs 50-95 (-0.013 at 110, -0.027 at 128);
  F-full where aggressive cutoffs (>= 110) matter. Not measured: transfer to other scans (the Otsu-relative cutoffs
  `--pap-rel` exist for that), robustness on actually noisier scans, `--pap-band` (soft targets used instead). Checkpoints
  ~/ufsm-data/runs/papyrus-20261011/F-{pool,full}-x13/last.ckpt (laptop). Paris4 cutoffs: 80 (default: the user's estimate, Otsu 84), 65 (conservative: the
  histogram minimum, keeps faint papyrus), 95-110 (aggressive: only dense papyrus; specks at 110+ come from the label itself).

## Inference memory (2026-10-11)

INFERENCE_MEMORY_PLAN.md phases 1, 2 and 2b (status table there): the inference build of the production net (one head
channel) goes 56.45 -> 43.45 B per window voxel by aliasing (the stem input in enc[0].a2's buffer, the logits in T1, predict's
CT window in the skip buffer with a bit-plane output mask; `UFSM_INFER_ALIAS`) and -> 26-28 B by in-place z-slab
convolutions at levels 0 and 1 (`UFSM_INFER_SLAB`, default 64 planes): one buffer per level, slab launches through an input
z-range in the MX forward kernels (`split_t.zi0 / Di`, `nn_set_zrange`) into a temp copied back with a one-plane carry, the
GroupNorm sums accumulated over the slabs, and the stem built per slab from the uint8 CT (`unet_forward_ct`). Output
byte-identical to the previous layout; on the 16 GB laptop GPU the largest fast window goes from < 640 to 800 (interior 512
with halo 144), at +2.5% forward time (W 640). predict computes one window per k^3 shards when the interior is k whole
shards.

Phase 3 and the split (same day): with the coarse offload (`UFSM_INFER_OFFLOAD`, default 1) the coarse levels live in the tail
of the level-0 buffer X, whose last planes wait in pinned host memory from enc[0]'s end to dec[0]'s conv1 (~2.8 B per voxel
each way, overlapped with down0 and dec[0]'s conv1); the level-1 buffer Y holds the level-0 slab temps and the stem input while
it is free, and down0's output, which enc[1]'s conv1 now reads in place; dec[0]'s conv1 takes its temp from R, then from Y's
consumed coarse planes. 26.8 -> 21.3 B per window voxel (X 17 + Y 4.25), logits byte-identical, forward time unchanged
(W 640 +0.5%, noise); largest fast window on the laptop 800 -> 896. In-place slabs now run under the spatial split (z-range
aware owned-plane statistics, per-thread z-range), and `predict --split z --gpus 0,1` runs each window on both GPUs (a half
plus 32 halo planes each; CT statistics and GroupNorm sums reduced across them): 21.4 B per held voxel, so a 1024^3 window
needs ~12.2 GB per GPU. Byte-identical to one GPU on the Paris4 box (window 576; 320 with 128 shards, both tilings), halves
emulated on the laptop's one GPU. Two real GPUs not yet run.

### The 16-wide production net (2026-10-16)

The slabs, the level-1 slabs and the coarse offload were gated to level widths that are multiples of 32, so the production net
(7 levels 16, 16, 32, 64, 128, 192, 256; MX-fp4) ran the phase-1 layout (20.6 B per window voxel) and predict had no working default
window (288 and 544 are not multiples of 64). Now: 16- and 8-channel levels take the slabs (whole MX blocks; the offload's layout uses
each tensor's own block width), the CT stem builds an MX-fp4 input as these nets are trained (UFSM_XIN_PREC=4), and with one fp16
logit channel the head runs per z-slab with its logits over X's first data rows (lg_x: the 16-wide Y holds 1.1 of their 2 B per voxel).
10.1 B per window voxel, logits byte-identical to the phase-1 layout (test_infer_mem: fp8 / fp4 stem input, fp32 / fp16 logits,
offload 0 / 1 / 2, the split).

Laptop RTX 5080 (16 GB), a 1000-step checkpoint, Paris4 2.4 um, the cached 1024^3 box at 49664, 20992, 20992:
- 576 / 32 (8 windows): 1.66 s of GPU time without slabs, 1.79 s with (the slab plane copies), outputs identical.
- 1088 / 32, one window per 2^3 shards, now the default on 16 GB cards: 13.8 GB, ~1.5 s of GPU per window in a run of 8 (871 Mvox/s
  computed, ~700 Mvox/s of interior); the shards compress at ~3.8 GB/s on a ring of 2 k^3 pinned buffers (not a bottleneck); the
  reader (0.2 s per window) and the CT upload (0.06 s) are minor.
- GPU time per 1088 window: the 16-channel convs (f4p) 31%, dec[0]'s conv1 with the fused upsample (f4u) 25%, the slab plane copies
  13% (D2D at ~300 GB/s copy rate, ~60% of the laptop's DRAM bandwidth; a single-launch copy kernel was no faster), the stem 9%,
  the 64-wide convs 6%, down0 5%, the head 3%.
- Halo: against a 1088 / 288 reference (interior 512, every voxel 288+ from its window's edge), 1088 / 32 differs by a mean
  4.9 / 255 (2.2% of 0.5 decisions flip) and 576 / 32 by 7.1 / 255 (3.9%), both nearly flat in the distance to the window edge
  (5.3 at 0-31, ~4.0 beyond 256): the halo matters little, the window does: per-window GroupNorm statistics make the predictions
  depend on the window's size and placement (frozen statistics, predict --gn-save / --gn-frozen, remove that).
- Paris4 level 0 is 75784 x 32693 x 32693 (81 Tvox, 20.8 Tvox nonzero after the mask); 22,505 windows of interior 1024 hold CT:
  ~10 h on the laptop GPU at 1.57 s per window (GPU-bound; a CT read over HTTP aside); not measured on a 5060 Ti (estimate:
  ~2x per GPU at half the DRAM bandwidth, so ~10 h with the desk's two, predict --gpus 0,1).

## Memory and inference defaults (2026-10-08)

- Training stores the MX activation gradients as MX-fp4 with exact stochastic rounding by default when every width is a
  multiple of 32 and P is divisible by 2^levels (`UFSM_GRAD_MX4=0` keeps MX-fp8). Desk mode (lean 2, chunk 2,
  recompute 1) at 512^3: 177.7 -> 139.8 B per level-0 voxel (-2.9 GB per GPU of the split slab), same step time
  (laptop P 384: 985 vs 984 ms of kernels). Paired 500-step runs at P 256 (two seeds): validation loss +0.1..0.8%.
  Against 16-bit gradients over 16 rounding seeds the fp4 parameter gradients are unbiased; MX-fp8's
  round-to-nearest stores are not (mean residual 0.14). A 16 GB GPU now trains a whole 512^3 window (0.43 samples/s).
- The decoder conv1 weight gradient runs over 32-channel up slices, so gout[0] holds one slice; with fp4 gradients the
  fp16 logits sit at the start of gout[0], the logit gradient in A, the batch at gout[0]'s end.
- Shared encoder a1 is on by default (recompute 1; `UFSM_RC_ENC_A1=0` off): the decoders reuse the encoders' a1 buffers and
  each encoder a1 goes to pinned host memory once its block's forward is done and comes back after the same-level
  decoder's backward (`UFSM_A1_OFFLOAD=0`: re-run the encoder conv1 instead, +2.5%). The training input copy lives in
  dec[0].a2's buffer, free before the decoder's forward and after its backward; host copy from the stem's forward to
  the stem's weight gradient (`UFSM_XIN_OFFLOAD=0` off). Copies at ~28 GB/s on their own stream; a repeated backward
  re-runs dec[0]'s conv2. Fixed-batch steps at P 384: 811 -> 817 ms (+0.7%). Desk mode: 139.8 -> 108.6 B per level-0
  voxel (split slab ~8.2 GB per GPU, 13.4 GB before fp4 gradients); a whole 512^3 window trains on a 16 GB GPU at
  recompute 1 (15.2 GB peak). If the host cannot pin the buffers, the level re-runs conv1 / keeps a separate input.
- Level-0 gradient buffer A in dec[0].a2's buffer (lean 2, MX-fp4 activations and gradients; `UFSM_A0_SHARE=0` off):
  dec[0].a2 is dead after the first op of dec[0]'s backward. The skip enc[0].a2 (idle from dec[0]'s conv1 to that conv's
  weight gradient) gets a pinned host copy after the encoders' a1 copies; its buffer takes the 16-bit logit gradient
  from the loss to the head's backward, the skip comes back before dec[0]'s conv1 weight gradient, and the input returns
  into it once enc[0]'s backward has read a2. Every coarser decoder level's A is likewise its dec[i].a2's buffer (no
  copies: nothing else holds it), so the separate A only serves the bottom level; a repeated backward re-runs the
  decoders' conv2 bottom-up. Desk mode 108.6 -> 91.6 B per level-0 voxel (12.3 GB at 512^3, ~6.9 GB per desk GPU's
  split slab); +0.4% step (the skip's copy queued before the encoders' a1 copies stalled dec[1]'s forward: +3%).
  `--mem auto` now runs a 512^3 window at recompute 1 on a 16 GB GPU (13.6 GB peak).
- The coarse levels' gradient buffers gout[1..] (up-part and skip gradients) live at the end of dec[0].a2's buffer, dead from
  the end of dec[0]'s backward to enc[0]'s conv2 backward-data; gout[1], also written in dec[0]'s backward (the sliced weight
  gradient's coarse silu(gn(a2)), the up-part gradient), sits at the end of dec[0].a1's buffer there (dead after dec[0]'s GN1
  backward) and is copied over before that buffer takes the encoder's a1 back (desk slab 0.32 GB of device copy). Same
  gradients as own buffers (test_grad_mx4, test_split), no measurable step time (P 384 medians within 0.5 ms either order).
  Desk mode 91.6 -> 86.4 B per level-0 voxel: 512^3 12.30 -> 11.60 GB, desk split slab 6.92 -> 6.53 GB.
  `UFSM_COARSE_GRAD_SHARE=0` off.
- The down-conv outputs (read by the next encoder's conv1 and its backward) sit below them in dec[0].a2's buffer (above the
  input it holds first), with a pinned host copy from that conv1 to dec[0]'s backward (desk slab 0.21 GB each way; back ahead
  of the encoders' a1). Same gradients, fresh and repeated backward; P 384 step +0.2% (ABBA medians). Desk mode 86.4 -> 83.7 B
  per level-0 voxel: 512^3 11.23 GB, desk split slab 6.32 GB. `UFSM_DOWN_OFFLOAD=0` off.
- The coarse skips enc[1 .. L-2].a2 (written by their encoder's conv2, read by the down conv and their decoder's conv1, idle
  until that decoder's backward) sit at the start of dec[0].a2's buffer (enc[1]'s conv2 waits for the input's copy out), with
  a pinned host copy from their decoder's conv1 to dec[0]'s backward (desk slab 0.39 GB each way). The copies are waited for
  where the data is used: dec[0]'s conv2 (not its conv1, which the last skip's copy out overlaps) and the decoder's conv1
  weight gradient (not its backward's start): +10 ms -> +0.6 ms per P 384 step (ABBA medians, 0.08%). Same gradients, fresh
  and repeated backward. Desk mode 83.7 -> 78.5 B per level-0 voxel: 512^3 10.53 GB, desk split slab 5.93 GB.
  `UFSM_SKIP_OFFLOAD=0` off.
- Three level-0 buffers instead of four (unet.l3b; the planner's first `--mem auto` mode where it applies: MX-fp4 activations and
  gradients, the chain above, an 8- or 16-channel level 0 the decoder above fits, UPW(1) <= w[0]; `UFSM_L0_3BUF=0` off).
  X1 = enc[0].a1 = dec[0].a1 = level 0's gout / gskip / B (dec[0]'s up transient; the input in enc[0]'s backward), X2 = enc[0].a2
  = dec[0].a2 (the batch head; the coarse backward homes; the encoder's a1 in enc[0]'s backward), X3 = level 0's A = the input,
  then the coarse skips / down outputs (forward homes), the fp16 logits with their in-place gradient, the batch tail at its end,
  the skip again for down0 / enc[0]'s backward. The head's input gradient is never stored: dec[0]'s GroupNorm backward recomputes
  it per voxel from the logit gradient (nn_gn_silu_bwd_head: the head backward-data's FMA chain and exact SR row, the same seeds),
  writing d/d a2 in place. The skip comes back into X2 during GN1's backward and dec[0]'s up transient (the coarse temp's bytes
  after it), into X3 after the decoders; enc[0]'s conv2 backward-data runs before its weight gradient so the encoder's a1 comes
  back into X2 meanwhile (seed draw replayed); with the GPU-built input the CT (2 B per voxel) goes to the host instead of the
  input and nn_input_rad rebuilds the input in X1. Logits bit for bit, gradients at the atomic-sum noise, the same seed sequence
  (test_grad_mx4: 16- and 8-channel level 0, down_norm, validation forwards, two steps, the GPU-built input). P512 16..256 on the
  laptop: 39.0 -> 30.0 B per level-0 voxel (5.24 -> 4.03 GB; planner need 6.19 -> 4.89 GB), trainer steps 845.5 -> 852.0 ms
  (+0.8%: ~16 ms of the skip copy outruns GN1 + the up transient, ~5 ms its second return; copies 28.3 GB/s). Not for the
  8-wide net (UPW(1) 16 > 8: X1 would hold a 16-channel up transient; ~2 B per voxel at stake). `UFSM_L0_3BUF_SPLIT=1`: dec[0]'s
  conv1 weight gradient in an up and a skip part around the skip copy (hides it, but the two 16-channel weight gradients cost
  ~19 ms more on the laptop).
- Three level-0 buffers, the copy schedule (unet l3_sched, `UFSM_L3_SCHED=0` off; latched at the build). The laptop under WSL runs
  every host copy one at a time, H2D and D2H included, and a small copy on another stream does not pass a big one queued before it
  (measured: a 20 MB D2H on a second stream finished after a 1.2 GB one; 1.2 GB D2H + 1.2 GB H2D on two streams 89 ms, no
  overlap), so the order the copies are issued in is the order they run. nsys of a P512 trainer step (628 ms, kernels ~540):
  ~40 ms of the forward waited on copies (the coarse decoders for their a1 behind level 0's first half, dec[0]'s conv2 for the
  skip queued after the encoders), ~25 ms of the backward (down0's weight gradient for the skip into X3 issued after the decoders,
  enc[0]'s for its a1 issued after down0's backward). Now: the encoder's a1 goes out as soon as the stem wrote it, the skip as
  soon as enc[0] finished, then the down outputs and coarse skips, whose X3 homes only the logits overwrite (the head waits for
  them, not dec[0]'s conv2); after dec[0]'s backward the coarse tensors come back, then the skip into X3 and the encoder's a1
  into X2 but for the coarse backward homes' bytes (l3_x2_free: between the coarse skips at X2's start and the down outputs /
  coarse gradients at its end), the rest after down0's backward. The coarse encoders (levels >= 1, ~0.2 GB at P512) re-run
  conv1 instead of their a1 copies (the same kernel and rounding keys; ~3 ms). Laptop P512 trainer, interleaved runs: 628.8 ->
  590.0 ms per step (1.59 -> 1.70 samp/s). test_l3_sched: the production chain at P128 / P512, schedule on vs off, logits bit for
  bit, enc[0] / down0 / dec[0] / head / all gradients at the atomic-sum noise, the same seeds, the batch tail kept.
- Offload copy order (`UFSM_A1_OUT_SPLIT=0` off): level 0's encoder a1 goes out in two halves, the second after the encoders, the
  down-conv outputs after the encoders, and level 0's a1 comes back after the decoders. The one copy stream had queued the coarse
  encoders' a1 copies (their decoders' conv1 waits for them) behind the 1.2 GB level-0 one: dec[L-2]'s conv1 waited ~27 ms per
  P512 forward, enc[1]'s backward ~6 ms. Four buffers, P512 prof_infer: 796 -> 765 ms.
- test_split runs on one GPU when there is only one (both halves take turns on the host anyway; the halo streams / events,
  the split config and the double-sum scratch are kept per half). New cases with the trainer's defaults (MX-fp4 gradients,
  the a1 / input / skip offloads, the shared gradient buffers): split vs one GPU at the rounding-seed noise, and the memory
  layout vs separate buffers with a fixed logit gradient within 1e-8, split and one GPU. MX-fp4 gradients are not bitwise
  reproducible: now and then (~1 in 7 identical steps here) a last-bit difference of an atomic sum flips a stochastic rounding
  and the change spreads to the seed-noise level; both outcomes are equally far from MX-fp8 gradients (0.078 rel).
- train: the MX-fp4 gradient default needs --fp4 2 and no --policy (the fp8 weight gradients of --fp4 1 took no MX-fp4 gy and
  aborted).
- Before / after the memory work (ac690ae vs now, laptop, same profiler): training steps at P 384 in the desk mode
  ~823 -> ~815 ms (fp8 -> fp4 gradients included); inference forward at 288^3 unchanged (~93 ms); predict over 8
  shards 6.35 -> 5.28 s of GPU time, wall 6.7 -> 5.9 s.
- predict computes only the written logit channel and keeps its 16-bit input in the network's shared a1 scratch
  (inference 80.5 -> ~57 B per window voxel). Boxes of whole shards default to one window per shard (shard + 32,
  halo 16) when it fits the GPU, else 288 / halo 8: laptop, 8 shards of 512, 16% less GPU time (5.20 vs 6.21 s),
  9.4 GB peak, and twice the halo. `--window` / `--halo` override.

## Earlier production audit (2026-10-02)

The pipeline is qualified to launch full training: major measured performance hypotheses are bounded,
all 38 final Make commands pass, and calibration and global-grid checks have completed. No long run
has started. The recipe selects 512/B1 on one 16 GB GPU and matched 528/halo8/shard512 serving.
Another 2x gain is not established; globally optimal kernels and final model accuracy are not claimed.
Historical timings below include other batch sizes, two-GPU runs, shared GPUs, and earlier kernels;
they are not directly comparable to this audit. References to benchmark files describe local evidence
under `runs/recovery512/`; generated artifacts and checkpoints are not stored in Git.

Geometric-mean-F1 calibration replaces the candidate's arithmetic mean, which permitted gains on
MANBp to offset Paris4 collapse. It uses only the same two calibration sources and the unchanged
threshold grid, with exact zeros and lower-cutoff tie breaking. Across the complete native dense
diagnostic, matched/FP8/FP16 cutoffs become 0.525 / 0.500 / 0.525 and six-source acceptance F1 is
0.073558 / 0.072976 / 0.072921, versus constant foreground 0.056580. Acceptance band-F1 is
0.201834 / 0.201611 / 0.192906. The four two-seed 160-update raw-weight comparisons retain cutoff
0.60 and their original precision conclusions. All seven old arithmetic reports remain exact,
and adversarial acceptance/partial-score changes cannot affect calibration. Frozen old recipes
preserve their original objective (`calibration-geometric-plan.json`,
`calibration-geometric-summary.json`). These are bounded, previously inspected holdouts.

Serving now optionally anchors tile interiors to selected-level CT coordinates with
`--grid-origin z,y,x`; the recipe chooses zero. The reader loads complete anchored windows beyond
the requested ROI, then clips only output placement. Two native dense scans pass legacy full-store
replays, aligned probability-store comparisons, and all three axis shifts with every decoded byte
in their common 1024-cubed ROI preserved (`global-grid-study.json`). CLI tests additionally cover
nonzero anchors, arbitrary lossless crops, CT edges, shard dimensions that do not divide the tile
core, host/device placement and multiple workers. All sixteen evaluation origins are 512-aligned
at their configured levels, so serving work and historical aligned quality scores are unchanged.
Off-grid shard boundaries can repeat network windows: the one-axis-expanded qualification requests
perform 20 forwards for 12 unique windows. This is a remaining crop-specific optimization opportunity,
not a gain available in the aligned production benchmark. Lossy codec placement can change decoded
bytes for arbitrary voxel shifts; the arbitrary-crop regression uses lossless output.

The full recipe is 780 seconds at 128/B4 followed by 20,000 updates at 512/B1, with an FP8 stem,
MX4 body storage, MX8 activation gradients, predominantly FP4 operands and FP32 accumulation.
Master weights, parameter gradients, optimizer state and EMA remain FP32. Training is approximately
75 Mvox/s per GPU with about 1.5 GiB spare; warm pure inference is about 345 raw / 315 useful Mvox/s,
and prior cached/local full predictions are about 240–260 useful Mvox/s. Training and serving GPU
objects are unchanged by the calibration/grid commit; final tests also cover calibrated exports
and relocated bundles (`github-publish-regressions.json`). A full model still requires scoring at
its own calibrated cutoffs, dense/partial/constant reports and acceptance of its actual quality.
Whole-sheet and verso labelling remain unstarted separate work.

The production wrapper now exposes `train --gpus 0,1`, `--split z`, and a memory-mode override,
with separate nonblocking leases for every selected device. Cross-process tests cover collisions
with either single GPU, canonical ID aliases, rollback after a partial acquisition, and release.
Real fixture trials exercise both-stage resume, calibrated evaluation and export in data-parallel,
spatial-wide and spatial-auto modes; manifests record each stage's effective batch and parallelism.
One same-checkpoint, 24-update benchmark per mode gives single-wide / dual-wide / split-wide /
split-auto 75.62 / 149.71 / 141.73 / 144.36 Mvox/s total. The sampled card peaks are dual-wide
14812/14795 MiB, split-wide 7878/7861 MiB, and split-auto 11236/11219 MiB. The recovered single-GPU
CSV remains unchanged, but its transient GPU samples are unavailable after a benchmark-driver
interval-count assertion; earlier single-GPU headroom measurements remain separate evidence.
Five four-update intervals exclude initial partial and final validation intervals. Data parallel
has effective batch 2 and 0.558 updates/s; split auto retains batch 1 at 1.076 updates/s, roughly
1.91x single-GPU update throughput. This supports an approximately 5.2-hour large stage for 20,000
updates, not a long-run convergence or sustained-clock guarantee (`dual-gpu-bench.json`). The GPU
kernels and precision policy are unchanged; the new wrapper arguments reproduce the qualified
split-auto flags. Both-GPU use is an optional execution mode; the single-GPU recipe remains available.

Current larger-window split capacity and recovery (2026-10-02): same frozen production binary,
FP8 stem, stored GroupNorm, FP4 body, MX8 gradients, B1, two idle 16 GB cards, and the same step-43
warmup checkpoint. Twelve-update runs at 640, 704 and 720 finish optimizer steps, final validation
and portable checkpoints; 736 and 768 fail activation allocation. Their sampled total card peaks
are 12890/12873, 14510/14493 and 15496/15479 MiB. Two four-update intervals excluding startup/final
validation give 138.94, 126.48 and 125.22 Mvox/s total. The capacity sweep predates the recovery
change below; 720 is the largest tested fit, with about 0.8 GiB spare. Split windows are multiples
of 16 (`large-split/results.json`). This supersedes the older 704 maximum under `--fp4 1` below.

Recompute 2 shared a1 across every block and reran every first convolution during backward.
The last decoder's a1 survives the head and loss, so its first backward can reuse that buffer
without allocation. A per-model live flag is cleared on consumption/free and rearmed only by a
new training forward; repeated backwards still reconstruct it. A frozen 24-update
control/reuse/control comparison gives 126.03 / 132.73 / 125.05 Mvox/s, a 5.73% gain against the
control mean, with identical 14510/14493 MiB sampled card peaks (`large-split/recover/results.json`).
`UFSM_RC_REDO_LAST=1` retains the original rerun for comparisons.

Optional `UFSM_RC_KEEP_COARSE=1` allocates private a1 buffers only for decoder levels above zero,
so their backwards avoid their first-convolution reruns too. Allocation counts include these buffers
in the existing dry memory planner, and the mode key forces a rebuild if retention changes.
The flag defaults off; put it in the production recipe's frozen environment to reproduce the mode.
An initial decoder-only trial gives 135.73 Mvox/s. A 24-update run of the final source confirms
135.94 over five four-update intervals and 14964/14947 MiB sampled card peaks, about 1.3 GiB spare
per card (`large-split-final/results.json`). Quantization, rounding seeds and accumulation precision
are unchanged. Against the repeated original-mode control mean, this is an 8.28% gain and recovers
about half of the gap to the 144.36 Mvox/s split-512 result. The same 20,000 updates would take
about 14.3 hours before warmup, validation and scoring; context grows by 2.60x per window.

The broader encoder-retention experiment was rejected after a preliminary stored-GN/FP8-stem
split-gradient check returned NaN. Its driver had already started a bounded larger trial when
stopped; that interrupted run has no completed checkpoint and is excluded from performance evidence.
The decoder-only trial's original GPU samples are intact; a later 15284/15267 MiB observation belongs
to the broader experiment and is excluded. Encoder retention is absent from the final source.
The final suite covers 24 fresh/repeated-backward combinations across 16-bit/MX8/MX4 storage and
lean modes, with and without decoder retention, plus the full spatial-split matrix with retention.
The standard production training/serving recipe remains 512/528; larger-context training needs its
own calibrated serving and quality acceptance. No long training run has started.

### Earlier qualification stages

The entries below retain the preceding experiments and launch holds; the completed audit above
supersedes their pending calibration and grid decisions.

Evaluation now declares dense/partial label groups, calibration sources and acceptance sources in
the candidate recipe. Dense boxes are scored at native resolution; partial boxes use level 1 or
their minimum available level. Each serving profile fits a single global cutoff on MANBp/Paris4,
and the remaining six dense boxes are reported separately at that cutoff. Acceptance and partial
scores cannot affect threshold fitting. Groups must cover all holdouts exactly once, calibration
and acceptance must partition the declared dense group, and source-level overrides are validated.
Export preserves the selected profile's fitted cutoff. Frozen old recipes retain their original
fixed-threshold behavior. Historical inspection of these holdouts is disclosed; the new split is
not claimed as an untouched final test. All 37 Make test commands pass, including adversarial
cutoff-selection tests and calibrated train/evaluate/export/moved-bundle integration. The current
scoring build repeats all 37 commands successfully (`eval-histogram-final-regressions.json`).

Ten-cutoff scoring now uses one grayscale maximum filter and cumulative byte histograms, preserving
exact binary threshold/dilation semantics. Predictions at ignored voxels still participate in the
filter; ignored labels never enter counters. The one/two-cutoff path retains its direct scan.
Twenty quiet whole-process controls on native 1024-cubed dense boxes and a partial-label box show
4.94x / 5.49x / 4.94x ten-cutoff gains; single-cutoff controls improve 2.9% / 6.1%.
All JSON reports and console scores match byte for byte (`eval-histogram-v3-quiet-bench.json`).
The integrated evaluation object's executable text matches the qualified private object exactly;
five additional real-volume CLI pairs match across tolerance 0–3, one/two/three/sixteen cutoffs,
repeated/unsorted cutoffs, endpoints and seam masks. Independent scatter tests cover 6484 grayscale
filter cases and 46080 histogram/direct counter cases over every byte cutoff, as well as the prior
6480 binary dilation cases (`eval-histogram-integrated-cli.json`, `eval-histogram-integrated-fixture.log`).
This is an evaluation/calibration gain, not a training-step or CNN-inference gain.

The two-seed weight-gradient study is complete (`wgrad-precision-summary.json`). Both seeds resume
the same checkpoint for 160 updates; only FP4 versus FP8 weight-gradient computation differs.
Reports use raw final weights so EMA history does not mask this short continuation. Each of four
models is served with matched FP4 and scored on all sixteen boxes, with native dense support and
separate partial-label support. All calibration fits select 0.60. Six-box acceptance mean F1 is
FP4/FP8 0.066569/0.065946 for seed 2 and 0.057726/0.057998 for seed 3; two-seed means are
0.062148/0.061972 versus constant foreground 0.056580. Both band-F1 comparisons favor FP4.
FP8 improves PHerc0500 and PHerc1667 at the fitted cutoff in both seeds, but the aggregate effect
reverses with seed. The slower FP8 path therefore has no established aggregate benefit, and FP4
remains provisional. These bounded diagnostics do not qualify production quality or cutoff transfer.

Warm forward-only inference with the actual restored checkpoint takes 0.424–0.429 s at 528 cubed
on one RTX 5060 Ti 16 GB: 343–347 million window voxels/s, or 313–317 million useful voxels/s
after cropping halo 8 to 512 cubed. Four runs on two real inputs each include three warmup and
twelve synchronized whole-network forwards, with the other GPU idle. Input construction, uploads,
probability placement, downloads, reads/writes and startup are excluded. Each complete output store
still matches its frozen production reference (`pure-inference-bench.json`). The earlier profiler's
`gpu(fwd+d2h)` label is broader than a forward timer and, on the default device-placement path,
does not include shard downloads charged to writer time. Use the dedicated benchmark for pure
inference claims; do not combine its timing with instrumented whole-process duration.

Forward operand reuse has now been bounded with an inference-only cache of the exact existing
16-channel pair-scaled FP4 operands. A separate single-position encoding preserves the final
shared-tile column's scale, so this does not add another activation quantization. All 29 small
batched/edge/scale cases and six repeated 528-cubed kernel cases match output bytes exactly.
Cached convolution alone drops from 52.89 to 41.97 ms at 16 output channels, but preparation raises
the total to 72.40 ms; the 32-output case rises from 70.26 to 88.01 ms. Eight complete native
1024-cubed predictions use isolated 0/1/1/0 controls on cached MANBp and local PHerc1667.
Complete wall time regresses 7.0% / 8.2%, while every store byte and metadata entry remains exact.
Observed peak device use rises from 7315 to 10979 MiB. The prototype is rejected and remains private
(`forward-pair-cache-summary.json`). A C16 FP32 normalized-value cache would need another
8984 MiB, leaving only about 12 MiB at this whole-pipeline peak; it fails the memory-headroom
requirement. This closes these specific cache hypotheses, not a claim of globally optimal kernels.
The current profiling loop has no qualified remaining large gain. Serving/grid/cutoff quality
qualification is the next launch gate; the full training run remains on hold.

A suffix-index read prototype removes HEAD requests before uncached shard indexes. A local fixture
passes 94 reader cases and 27 direct tail cases, reducing its shard request count from 29 to 22.
Twenty-three complete predictions preserve all prediction-store and cold-cache bytes. Whole-process
gains are only about 0.8–1.5% on two cold dense regions and 0.9% on the air-heavy region, while
cached/local results remain within 0.4% (`http-suffix-bench.json`). These runs share the environment
with quality evaluation on the other GPU, so such small changes do not qualify a default change.
The prototype remains private; production retains the existing coalesced reader and HTTP handle pool.

Increasing FP4 weight-gradient depth from 12 to 48/96 amortizes repeated halo setup and parameter
atomics. Three full-size isolated kernel cases improve roughly 7–8%, with relative gradient changes
below 3.2e-7 and worst absolute changes below 4.6e-6 of the largest reference gradient, within the
repeated-control reduction noise (`wgrad-depth-probe-summary.json`). A quiet six-run, 40-update
whole-training sweep at 512 and `--mem wide` gives 73.954 / 74.759 / 74.491 / 74.357 / 73.954 /
72.209 Mvox/s for depths 12 / 48 / 96 / 96 / 48 / 12, at unchanged 14785 MiB peak device use.
GPU temperature rises from 47–72 C in the initial control to 84 C, and active mean SM clock falls
from 2868 to 2780 MHz. The hot depth-48 run and final depth-12 control have essentially equal
clocks, with 2.4% greater whole-step throughput; hot depth-96 is about 3.0% faster at a slightly
higher clock. These support a small tuning opportunity, not another 2x gain
(`wgrad-depth-quiet-train-bench.json`).

The guarded automatic choice is now integrated: depth up to 96 only for non-Hadamard row-pair
MX4 inputs with MX8 activation gradients and at least 64 two-plane z tiles; smaller volumes and
other layouts retain depth up to 12. The existing 72-block occupancy floor and explicit
`UFSM_F4W_ZC` override remain. Four complete 40-update controls without a depth override give
73.954 / 74.893 / 74.491 / 72.478 Mvox/s for reference / candidate / candidate / reference,
at identical 14785 MiB peak use and identical executed precision manifests. Candidate/control
mean throughput improves 2.0%; the slower candidate still exceeds the faster control by 0.7%.
The frozen comparison records device clocks and temperature; these are bounded whole-step
measurements, not steady full-run promises (`wgrad-depth-guarded-summary.json`).
Twenty gradient comparisons cover 128/512 shapes, normalized inputs, decoder upsampling and
three rounding seeds/modes. Relative changes remain <=2.6e-7, with worst element error <=5.1e-6
of the largest reference gradient. Ten seed comparisons still vary by at least 16.5%. The CUDA
fatbin is byte-identical: only the host's launch depth changes, altering FP32 reduction grouping.
The regular staging test now exercises the larger automatic-depth geometry as well as its
existing boundary, precision and underflow cases. Nine additional batched edge/upsampling comparisons
at 128x130x132 match depth-12 gradients exactly in RN and two SR seeds. A real bounded recipe trial
completes 43 warmup updates at 128/B4 and resumes to step 45 at 512/B1 across the full source
configuration; embedded settings and executed manifests preserve the same stem/body/gradient
precision. Its five-second budget applies to each stage; it is not the long production run.
Two native 1024-cubed serving replays at 528/halo8 preserve every prediction-store byte and metadata
entry. The integrated executable is byte-identical to the qualified candidate; all 37 Make commands
pass on it (`wgrad-depth-guarded-inputs/numerics.json`,
`wgrad-depth-guarded-inputs/batched-numerics.json`, `depth-pipeline-trial/run.json`,
`wgrad-depth-integrated-serving.json`, `wgrad-depth-guarded-final-regressions.json`).

A frozen stored-GN/wide-gradient checkpoint has completed 24 native prediction/score cases
(`tile-contract-audit.json`): three serving precisions on two calibration scans, followed by
two other scans at the fitted cutoffs with z/y/x tile shifts, halos 8/16/32, precision comparisons,
and repeat controls. FP4 calibration selects 0.60; FP8/FP16 select 0.55. Their two-box acceptance
mean F1 is 0.05825 / 0.06812 / 0.06700, against constant foreground 0.06524. Cutoff transfer is
weak on PHerc0500. This does not show that FP4 destroys prediction detail: post-hoc fixed-cutoff
diagnosis at 0.50 gives mean F1 0.08135 / 0.08134 / 0.08120, and FP4 also leads at 0.55. Those
diagnostic cutoffs are not substituted into the calibrated acceptance report
(`tile-contract-threshold-diagnostic.json`). The matched weight-gradient comparison above now bounds
that precision choice over two seeds; serving precision, phase and cutoff transfer remain concerns.

On PHerc1667, shifting one tile axis by 256 changes FP4 F1 from 0.11230 to 0.12265 / 0.12452 /
0.09759. Halo 16/32 gives 0.11299 / 0.11178, at 4.45 / 5.19 s versus 4.11 s for halo 8.
PHerc0500 likewise shows no useful halo improvement. Repeat controls have byte-identical full
prediction stores and identical metrics. Exact common-ROI comparisons independently reproduce
label support and true-positive counts; prediction changes also occur away from seam bands
(`tile-contract-stability.json`). These are bounds on one diagnostic checkpoint, not a production
seam acceptance or a recommendation to pay for larger halos. Window/grid sensitivity and global
cutoff generalization remain explicit quality concerns; production training stays on hold.

The serving comparison now covers all eight native dense boxes at phase zero, using the same
frozen stored-GN diagnostic checkpoint (`serving-dense-summary.json`). All 24 cases complete;
twelve repeat earlier cases with byte-identical full stores and identical shared-cutoff counters.
Observed manifests confirm FP8 stem/down and FP4 body for matched serving, FP8 convolutions for
the FP8 profile, FP16 convolutions for the FP16 profile, and an FP32 head throughout.
The existing arithmetic-mean-F1 calibration selects 0.60 / 0.55 / 0.55 for matched/FP8/FP16.
Six-source acceptance mean F1 is 0.054022 / 0.071068 / 0.070368, against constant foreground
0.056580. At a common diagnostic cutoff of 0.50, means are 0.072182 / 0.072976 / 0.072842.
This locates much of the apparent serving-quality gap in cutoff selection. Fitting band-F1 only
on the declared calibration sources selects matched cutoff 0.55 and gives acceptance F1 0.071841
and band-F1 0.183754; these remain diagnostics and do not alter the production objective.
Prediction averages are 4.46 / 5.60 / 9.11 s; one recovered matched case has no retained timing
and is excluded from that average, while every case remains in quality reporting. The next
step is to validate a calibration correction on bounded reference studies while preserving the
calibration/acceptance roles, then qualify the supported serving grid. No long run has started.

The stored-tensor normalization and independently selectable stem precision pass the complete
`make -j8 test` suite (`runs/recovery512/stem-gn-full-tests.log`). Fresh training's convolution-output
GroupNorm statistics now describe rounded stored values, including MX4/MX8; legacy checkpoints
retain their historical contract on resume and prediction. Independent FP64 reference tests cover
constant/varied groups, gradients, split inputs and FP16 saturation replay. Mixed MX8-to-MX4 and
MX4-to-MX8 stem tests compare forward/weight-gradient operations with decoded operands. The input
buffer is included in the dry memory planner; `--input-prec 4|8` and `--gn-stats stored|legacy` are
portable checkpoint settings. A bounded multi-source trial is complete; it does not establish
production quality.

Frozen candidate `ufsm-stem-gn-candidate` (SHA256
`5a0ab3077a95ba56de69e4133ae090ee77c1222ba9b34f0970b7b632cf1edcac`) gave these short, single-GPU,
batch-1 diagnostics (`stem-gn-bench.json`; approximately 35 training seconds per case):

| window / stem bits / GN | samples/s | Mvox/s | peak MiB |
|---|---:|---:|---:|
| 512 / 4 / legacy | 0.517 | 69.39 | 14272 |
| 512 / 8 / legacy | 0.515 | 69.12 | 12624 |
| 512 / 4 / stored | 0.499 | 66.97 | 14272 |
| 512 / 8 / stored | 0.499 | 66.97 | 12624 |
| 496 / 8 / stored | 0.544 | 66.38 | 13476 |
| 528 / 8 / stored | 0.449 | 66.09 | 13842 |

512 with an FP8 stem selects lean 2, leaving 3687 MiB (3.6 GiB) on the 16311 MiB card. Correct stored
statistics cost about 3–4%. The five steady-step
profile is 41.7% weight gradient, 24.3% backward-data, 22.0% forward, 7.2% normalization and 4.1%
upsampling (`stem-gn-p512-i8-stored.log`). Final decoder 48-to-16 alone accounts for 26.8% of the step.
The extra output-statistics pass is included in forward timing. Prior tile/register tuning did not
improve the valid baseline; the next work targets operand preparation/reuse and correct fusion.

Cooperative MX8 gradient loading reuses the decoded-input shared tile after X packing, retaining
the FP4 operand grids and stochastic-rounding keys. It is selected only for output widths <=16;
the 32-channel experiment slowed down. Tiny gradient-scale blocks retain the original load path
to preserve underflow behavior. Paired 40-step stored-GN/FP8-stem runs at 512 give 0.515/0.516
samples/s versus bracketing controls 0.499/0.497 (about +3.5%); disabled in the same candidate,
the result is 0.501. All peaks remain 12624 MiB (`wgrad-coop-step-bench.json`). Final decoder
weight-gradient preparation was the first target. The candidate and final binaries contain
byte-identical CUDA fatbins; the final host option lookup is read-only across GPU threads.
All 30 commands of the Make test recipe pass, including both split presets
(`wgrad-regressions.json`). The recorded Makefile and executable hashes match the current files;
the command multiset covers the entire recipe. The option `UFSM_F4W_COOP_GY=0` retains the original
gathering.

Corrected native-resolution inference on the same cached MANBp 1024-cubed box now has three-run
medians of 5.109 / 6.179 / 9.228 s for matching FP4 / FP8 / FP16, with peak memory 7202 / 10094 /
14324 MiB (`stored-infer-bench.json`). These end-to-end diagnostics use a very short checkpoint,
with a bounded trial on the other GPU, and do not select production precision. The fresh stored-GN,
FP8-stem 17-source trial has completed 780-second warmup/large/control training and all 16 holdouts
(`production17-stored-summary.json`). On eight dense boxes the best sampled global-cutoff mean F1
is 0.07729 / 0.07794 / 0.07797 for matched FP4 / FP8 / FP16, versus 0.07052 for the small-window
control, all at cutoff 0.5. This cutoff selection uses the evaluated holdouts and is diagnostic,
not calibration or independent acceptance. At the provisional cutoff 0.6, dense means are
0.03481 / 0.02680 / 0.02820 versus 0.02614; the sixteen-box aggregate 0.28540 versus 0.31334 is
distorted by the partial labels. Dense-box quality remains weak. Prediction-only throughput is
177.8 / 152.4 / 106.3 Mvox/s (`production17-stored-group-diagnostics.json`). The quality trial
uses the frozen stored-GN baseline, preceding cooperative weight-gradient loading. Only bounded
trials have been launched.

Nsight of the first stored-MX4 statistics pass at 528 measures 18.32 ms: SM throughput 77.65%,
DRAM 16.46%, L1/TEX 66.07%, 64 registers/thread and 62.92% achieved occupancy. The launch has
574992 blocks, each 256 threads (`stored-gn.ncu-rep`). This pass is a concrete fusion target,
provided partials describe rounded output values and are combined over complete GN groups.
Its measured cost does not prove a 2x whole-pipeline opportunity.

Stored-MX output statistics are now fused into the convolution epilogue: it decodes the exact
codes and scales it writes, then reduces those rounded values with the bias already included.
Ordinary FP16/BF16/FP32 outputs retain the independent FP64 statistics pass; legacy checkpoints
retain their original contract. `UFSM_FUSED_STORED_GN=0` selects the separate reference pass.
At 512, paired 40-update runs improve 0.512/0.516 samples/s to 0.528/0.529 (about +2.8%,
70.9 Mvox/s); the disabled candidate gives 0.513. All peaks in this comparison are 12750 MiB,
including the additional idle-device allocation relative to the preceding benchmark.
On the same cached native MANBp 1024-cubed box and fresh staged checkpoint, matched FP4
prediction takes 4.479/4.515 s versus controls 4.953/5.113 s (+11.9% throughput); FP8 takes
5.661/5.686 s versus 6.040/6.108 s (+7.1%). Memory is unchanged within each comparison
(`gn-fusion-step-bench.json`, `gn-fusion-infer-bench.json`). These are isolated-box diagnostics.
The FP4 decoded byte outputs are identical over all 1073741824 voxels. FP8 changes 9358 bytes,
with maximum error 2 and 270/68 classification flips at byte thresholds 128/153; its two
unfused controls are identical. Independent FP64 checks now cover output widths 16–80,
groups crossing packed blocks, upsampling, owned planes and all three public convolution APIs.
All 33 commands of the expanded Make test recipe pass, including both stored-GN/FP8-stem
split presets (`gn-fusion-regressions.json`), with command coverage and current hashes verified.
The measured candidate and tested final executable have identical CUDA fatbins. The production
candidate recipe explicitly selects the trial's FP8 stem and stored normalization; dense quality
and calibration remain unqualified.

A fresh profile of the cooperative final-decoder weight-gradient kernel at 512 shows 44.6% SM
throughput, 8.6% DRAM throughput, 33.9% L1/TEX throughput, 96 registers/thread and 37.3%
achieved occupancy; long/short-scoreboard stalls are 17.5%/7.4% (`wgrad-coop-profile-summary.json`).
This isolated decoder probe supports investigating operand reuse and occupancy; it does not
establish a corresponding whole-step speedup.
Rechecking an eight-input-channel weight-gradient tile after cooperative gathering still loses:
the final decoder takes 347.2 ms versus bracketing 216.4/217.1 ms with sixteen input channels,
and the 16-to-16 layer takes 110.8 ms versus 67.9/67.5 ms. Every gradient comparison passes
(`wgrad-coop-nt-sweep.json`). Smaller tiles remain rejected. A once-per-convolution gradient-prepacking prototype also loses
(`wgrad-prepack-experiment.json`; implementation and candidate binary archived). All 24 small
cases exactly match the frozen CUDA implementation, and full-size gradients pass unchanged
tolerances. At 512 the fused-up decoder is 219.4 ms versus 221.1 ms in the candidate's disabled
path, but a frozen control is 216.3 ms; the 16-to-16 layer regresses from 69.0 to 79.9 ms.
The materialized-up / skip-GN layout used by lean training is 197.4 ms versus 197.1 ms disabled
and 193.9 ms frozen. Nsight separates about 11.35 ms of preparation from 208.1 ms of consumption
for the fused-up probe. The preparation buffer adds 1344 MiB at 512. The prototype is removed;
production keeps the validated cooperative in-block gathering. No whole-step speedup is claimed.
The next target is the 16-channel FP4 convolution's operand-staging register footprint: the
existing 528 profile reports 128 registers/thread and a two-block register limit
(`infer528-p16-counters.json`). A two-pass staging prototype exactly matches all 37 frozen
small outputs and the full-size packed outputs, but retaining fewer values alone still compiles
to 128 registers and loses at 512: normalized MX4 input is 70.8 versus 48.1 ms, and unnormalized
MX8 input is 60.4 versus 55.2 ms (16 outputs) / 71.3 versus 70.4 ms (32 outputs). An isolated
three-block launch bound produces 80 registers: the 16-output gradient shape improves to
52.7 versus 55.1 ms, but normalization loses and 32 outputs spill and regress to 123.4 ms.
With two z planes, all three large shapes lose. The broad replacement is removed; no production
or whole-step speedup is established (`f4p-reload-experiment.json`).
A frozen-kernel precision probe also finds FP8 computation within about 2% of FP4 on these
16-input-channel shapes; output bytes differ as expected. This is a speed diagnostic and does
not establish gradient/model-quality equivalence (`f4p-vs-f8p-512.log`).

Warp-distributed staging also fails to improve the 16-input-channel FP4 convolution. Five private
builds distribute a position over one, two, four or eight lanes, with two-/three-block launch
bounds. All 185 small comparisons and all fifteen full-size comparisons match every packed output
byte; pair scales, rounding keys and MMA order are retained. Three-block builds use 79–80 registers,
but the 32-output instantiations spill. The best full-size version (one lane per position) takes
55.0 versus bracketing 48.10 / 48.09 ms for normalized MX4 input, 65.7 versus 55.12 / 55.13 ms
for unnormalized MX8 input with 16 outputs, and 121.9 versus 70.16 / 70.19 ms with 32 outputs.
Two/four/eight-lane versions are slower still. Lower register counts alone have not yielded a
throughput gain. These standalone prototypes are rejected without production integration or a
whole-step claim; source, object, binary, compiler reports and exact-output logs are archived in
`f4p-warp-experiment.json`. The tested production executable remains unchanged.

A register-only accumulator probe on the RTX 5060 Ti checks the current block-scaled MMA forms
against unscaled FP8/FP4 and FP16 operands (`accumulator-probe-summary.json`). PTX defines the
current `mxf4` and `mxf8f6f4` block-scale forms with FP32 accumulators only. Two repetitions give
approximately 425 TFLOP/s for MXFP4/FP32, 213 for MXFP8/FP32, 213 for ordinary FP8/FP16, and
108 for the alternate unscaled FP4/FP16 form. Ordinary FP16 doubles from about 54 to 107 TFLOP/s
when its accumulator format changes, but those instructions are not the dominant FP4 body path.
Each mode checks every thread's result against an exact uniform-input sum; this is not a training
accuracy qualification. The test excludes operand staging and external scale application, and
does not rule out a shape-specific register benefit. It establishes no arithmetic-throughput
advantage from replacing the current block-scaled FP4/FP8 instructions with these FP16-accumulator
alternatives. Production retains the tested instructions and FP32 parameter reductions.

FP8 weight-gradient computation is slower in the same frozen packed-input probes: at 512 the
materialized-up / skip-GN decoder is 237.9 versus 193.4 ms, and 16-to-16 is 86.5 versus 67.9 ms;
32-to-32 at 256 is 35.9 versus 33.2 ms. Gradient relative differences are 3.1–8.6%, as expected
from changing precision; no accuracy acceptance is inferred (`wgrad-f4-vs-f8-512.log`).
The validated FP4 weight-gradient path remains the throughput candidate.
The post-fusion nearby-window sweep (`gn-fused-window-sweep.json`) gives 71.2–71.9 Mvox/s
at 512 (12750 MiB), 70.8 at 496 (13602 MiB), 70.2 at 528 (13968 MiB), and 64.1 at 544
(13024 MiB, recompute 2). These are short single-GPU diagnostics, with 512 bracketing the
other shapes; 512 remains the training candidate on speed and headroom. A profile using the
actual lean-training input layout (materialized up segment, skip GN) measures 39.8% SM,
14.3% DRAM, 34.9% L1/TEX, 96 registers and 37.3% occupancy, with 21.2% long-scoreboard
and 8.1% short-scoreboard stalls (`wgrad-materialized-profile-summary.json`). The earlier
cooperative profile above used fused-up staging. Both are isolated probes.

A decoded-X row-padding probe retains the LY 1 quantization grid, gradient gathering and rounding
keys. Strides of 26 and 28 BF16 elements pass all 24 small edge/split/upsample/scale cases and
the full-size comparisons. The materialized 48-to-16 decoder improves from about 193.5 ms to
191.8 ms (26) or 192.6 ms (28); 16-to-16 improves about 0.6%. An isolated unpadded rebuild
matches the frozen CUDA control. These gains are below 1% of the isolated kernel, and no
whole-step gain is established, so production remains unchanged (`wgrad-padding-experiment.json`).

An optional larger final-decoder up-gradient buffer retains lean 2's deferred skip write, but
allows its 32-channel up part to run in one backward-data launch. Paired 40-update runs give
0.546–0.549 samples/s against controls 0.528–0.532 (about +3%), at 14796 versus 12748 MiB;
the 16311 MiB card retains about 1.5 GiB. Widening every decoder gives similar throughput
at 15374 MiB and is rejected in favor of the finest level (`wide-up-step-bench.json`,
`wide-up-finest-step-bench.json`). These are isolated-device performance comparisons.
The buffer option defaults off. `UFSM_WIDE_UP_GRAD=1` or `unet_set_wide_up_grad(1)` changes
only the finest-level buffer under lean 2 / MX8 gradients. The memory planner counts the extra
allocation; automatic planning conservatively reserves additional workspace. `--mem wide`
explicitly selects chunk 2 / recompute 1 / MX8 gradients / lean 2 and the larger buffer, requiring
tensor cores and MX activation storage. The candidate recipe selects this mode for its large
stage while preserving automatic planning for the small-window warmup.
Changing the option rebuilds existing model buffers. Whole-network forward/gradient tests
cover dry allocation accounting, repeated runs, rounding modes and cached-buffer transitions.
Round-to-nearest gradients agree within FP32 reduction noise; stochastic differences remain
below baseline seed variation, which does not establish training quality. All 35 Make test
commands pass with coverage and current artifact hashes recorded (`wide-up-regressions.json`).
The unchanged CUDA objects are byte-identical to the measured frozen baseline. A bounded,
equal-update 17-source continuation from the same stored-GN checkpoint completed 160 updates
per side and scored all 16 boxes. At cutoff 0.5, dense mean F1 is 0.07676 (narrow) versus
0.07699 (wide); at 0.6 it is 0.03911 versus 0.03929. No obvious regression appears, but this
one-seed diagnostic does not qualify the weak dense accuracy or independently calibrate a
production threshold (`wide-up-quality-summary.json`). All 35 commands pass again after
adding the public memory mode and exercising it through staged train/evaluate/export/predict
(`wide-up-mode-regressions.json`); the full training run remains on hold.
The final public-mode sweep gives 73.7–74.2 Mvox/s at 512 (14796 MiB), versus 73.0 at
496 (13496 MiB), so the recipe retains 512 (`wide-up-mode-window-sweep.json`). The two
512 controls bracket 496; all three runs use the fully tested final executable.

A selective-compute experiment keeps the FP8 stem, packed MX4 body activations and MX8 gradients.
The experimental `UFSM_MX4_FP8=1` dispatch allows a requested FP8 forward operation to use the
existing FP8 kernel on MX4 input/output; it defaults off and has not been promoted to production.
Legacy MX4 dispatch continues to select FP4 computation. A six-case isolated-GPU sweep at 512,
batch 1 and `--mem wide` uses 45 training seconds per case and bracketing FP4 controls:

| compute change from FP4 body | Mvox/s | throughput change | peak MiB |
|---|---:|---:|---:|
| controls, start / end | 74.09 / 73.55 | reference | 14796 |
| FP8 forward | 68.45 | -7.3% | 14796 |
| FP8 backward-data | 73.95 | +0.2% | 14796 |
| FP8 weight gradient | 69.52 | -5.8% | 14796 |
| FP8 all three passes | 65.23 | -11.6% | 14796 |

These are speed diagnostics, not a precision acceptance (`mixed-fp8-step-bench.json`). Observed
manifests confirm the selected arithmetic; all six runs finish without nonfinite steps. Middle,
complete five-update profiles put the current FP4 controls at approximately 374 ms forward,
439 ms backward-data, 740 ms weight gradient, 141 ms normalization and 61 ms upsampling per step
(`mixed-fp8-steady-profile.json`). Forward FP8 adds about 143 ms; FP8 weight gradients add about
106 ms. Neither is a throughput improvement. FP8 backward-data is already available through
`--policy all=fp4:fp8:fp4,enc0.c1=fp16`; this setting merits a bounded learning comparison.

Both experimental dispatch states pass thirteen forward cases and eight backward cases,
including split/upsampled inputs and independent decoded-output statistics.
Twelve additional body cases match independently decoded inputs, FP8 arithmetic and a separate
MX4 output converter exactly, with normalization enabled/disabled and widths 16/32/48.
Repeated whole-model backward-data tests pass with rounding enabled/disabled and with the
legacy dispatch retained.
All-compute FP8 occasionally differs by about 4.1% in a reused model's gradients; one forward-only
stochastic test also fails its same-seed repeat tolerance. Those modes remain unqualified while
the cause is investigated. Initialization checking reports zero errors, and a targeted check of
two FP8/MX8 backward-data launches reports zero shared-memory hazards; neither proves the full
network discrepancy harmless. The broader racecheck was deliberately stopped to narrow its
scope. Numerical logs, hashes and limits are recorded in `mixed-fp8-experiment.json`.
The equal-update FP4 / FP8-backward-data pair has completed 160 updates from the same checkpoint
and all sixteen held-out boxes. Eight dense boxes average F1 0.076782 / 0.076816 at cutoff 0.5,
with band F1 0.292076 / 0.292597. This provides no meaningful evidence of a quality advantage;
at cutoff 0.6 the respective means are 0.036526 / 0.039963, illustrating cutoff sensitivity
(`mixed-fp8-backward-quality-paired-summary.json`). The forward-FP8 continuation also completed
160 updates and all sixteen boxes under matched and FP4 inference. Dense F1 at the diagnostic cutoff 0.5 is 0.077387 matched and 0.077166
served with FP4, versus the control 0.076782. At cutoff 0.6 the forward-FP8 means are
0.034870 / 0.039333, versus control 0.036526. These small one-seed changes do not establish
a learning advantage or justify the measured 7.3% training throughput loss
(`mixed-fp8-quality-summary.json`). The production recipe and long-training hold remain unchanged.

A predecoded FP32 activation-slab probe preserves the current staging values and quantization
rather than introducing BF16 rounding after GN/SiLU. The private dry allocator confirms a
2 GiB unused tail while the wider finest gradient buffer holds its MX4 upsample; a 34-plane,
48-channel FP32 slab would require 1.59 GiB. Actual buffer-tail aliasing was not implemented.
The isolated 32-by-512-by-512 probe loses: all-FP32 preparation/consumption takes 25.07 ms
versus packed controls 12.53 / 12.59 ms; preparing only the normalized skip takes 18.14 ms
(3.65 ms preparation, 14.51 ms consumption). The normalized 16-to-16 probe takes 9.18 ms
versus 4.45 / 4.49. Six small cases and both large probes match every gradient exactly, but
these measured versions are rejected before full-step integration (`wgrad-fp32-slab-experiment.json`).

A separate packed-layout prototype reduces the LY 1 row pitch from 24 to 16 bytes and the
per-plane scale slots from 32 to 10 elements, retaining the same values, row pairings and
rounding keys. With the existing two-block launch bound it gives no kernel gain. A three-block
bound uses 72 registers and measures 183.24 versus 193.59 / 194.05 ms for the final decoder,
and 62.89 versus 67.89 / 68.10 ms for 16-to-16 at 512. Twenty-eight small comparisons include
nineteen selected compact-kernel cases plus fallback checks; full-size gradients remain within
the frozen-reference tolerances. Whole-model rounding modes and reused-buffer transitions pass.
In complete 45-second training diagnostics, two candidate runs give 0.553 samples/s versus
bracketing controls 0.550 / 0.547, about +0.8%; the separate frozen control gives 0.552.
All peaks are 14796 MiB and no run has nonfinite steps. Complete middle profiles show weight
gradients reduced from about 742 to 722 ms per update, with smaller gains in the whole step.
The prototype remains unpromoted: the complete regression suite has not been run with this
implementation, and these short timings do not establish a substantial pipeline gain
(`wgrad-compact-experiment.json`).


An inference I/O audit separates cold remote access from cached GPU execution
(`infer-io-audit.json`). On a 512-cubed MANBp box, a fresh compressed cache takes 4.07 s
versus 1.39 s warm; the reader accounts for 2.68 versus 0.03 s. Cached native 1024-cubed
MANBp and local PHerc1667 boxes take 4.43 / 4.12 s, with about 3.57 / 3.59 s in GPU
input/forward/placement and only 0.03 s waiting for the reader. Profiling adds explicit
synchronization; process times include startup and final writing, while the printed main-loop
wall excludes part of that work. Reader and GPU times overlap and must not be added together.

A bounded HTTP-read prototype coalesces nearby uncached chunk ranges (at most 64 KiB gaps,
2 MiB spans) and fetches missing shard indices in parallel. On independent fresh caches of the
same MANBp box, controls take 3.75 / 3.52 s versus 2.26 / 2.20 s for the candidate: about
1.63x whole-process speed, with reader time falling from 2.30 / 2.22 to 0.97 / 0.92 s.
The grouped plan has 77 tasks for 216 chunks and about 6.01 MB of range spans for 5.88 MB
of requested chunk payload. Each cache contains the same 194 files and 5,945,604 bytes as
the frozen reader; every cached encoded byte and every produced prediction-store byte is
identical. Cached 1024-cubed prediction remains about 4.39 s, and its full store matches the
frozen reader exactly. This is a promising cold-access gain on one box, with no demonstrated
cached-GPU, training or whole-volume gain. HTTP error cases, connection/resource lifetime,
other cold volumes and the full regression suite remain before promotion
(`infer-coalesce-experiment.json`). That experiment was initially kept outside `src/`.

The reader has now been integrated and qualified (`io-qualified-summary.json`). In bracketed,
independent cold caches of native 1024-cubed regions, MANBp takes 10.57 / 9.62 s with the frozen
reader versus 7.59 / 7.66 s with the integrated candidate (1.32x); PHerc0500P2 takes 11.29 / 10.49
versus 8.11 / 7.73 s (1.38x). All prediction-store bytes and all 743 / 978 cache files match
exactly. Cached MANBp stays about 4.43 versus 4.46 s, and local PHerc1667 about 4.11 versus
4.12 s. The final planner also removes absent chunks before grouping; this lowers fixture payload
requests from 18 to 14 (original control 37), but its additional whole-process speedup is not
established: both real scans remain within 1% of the initial integrated candidate. Its independently
populated cache and prediction stores also match. This is cold-access qualification on two bounded
regions, not a whole-volume, cached-GPU or training speedup.

Production defaults enable bounded coalescing and parallel index fetching; either can be disabled
with `UFSM_Z3_COALESCE=0` / `UFSM_Z3_INDEX_PARALLEL=0`. Parallel metadata fetching requires the
disk cache and uses at most 16 threads; grouped payload spans remain at most 2 MiB with 64 KiB
maximum gaps. The reader now validates index CRC32C and payload ranges, verifies HTTP Content-Range
and exact response length, and propagates worker errors. Only true absence becomes sparse fill;
403 responses fail rather than poisoning the missing-shard cache. Thread-specific curl handles and
headers are destroyed when workers exit, cache temporary names include process and thread IDs, and
writer CRC table initialization is thread-safe. Array-edge chunks are clipped to the declared shape,
fixing the previous overwrite of fill values outside the array. A deterministic local HTTP fixture
checks 89 reads against independent voxel bytes, sparse/unsharded/single-chunk shards, cold/hot/partial
caches, redirects, retries, malformed responses, corrupt indexes, request bounds, concurrent writers,
and repeated reads/global cleanup; file descriptors remain 4 -> 4. All 36 Make test commands pass on
the final executable (`io-qualified-v2-regressions.json`). Its CUDA fatbin is identical to the frozen
baseline. Precision and model-quality acceptance remain open, and full training is still on hold.

A 17-run concurrency diagnostic retains the current executable and changes only prediction reader
threads (`pipeline-headroom-summary.json`). Fresh MANBp controls take 8.12 / 7.65 s around a
32-thread run at 7.37 s; PHerc0500P2 takes 8.06 / 12.69 s around 7.56 s. The latter control is a
large network outlier, so averaging it into a claimed gain would exaggerate the result. The initial
8/16/32/64 sweep also has drifting controls. All cold cache files and prediction shards match.
Cached MANBp takes 4.67 s at 32 threads versus 4.52 / 4.45 s controls; local PHerc1667 takes 4.16
versus 4.09 / 4.11 s. Cached/local comparisons check prediction shards only. Higher concurrency
is a modest cold-read candidate, not a universal improvement; the default remains 16. Cold readers
still leave seconds of main-thread wait, while cached readers leave about 0.03 s. Reusing HTTP
connections across reader jobs merits investigation; the current short-lived workers close their
handles on exit. Existing tile read-ahead and asynchronous shard writing already overlap CPU work.

That connection-reuse candidate is now integrated (`http-reuse-summary.json`). Workers return
reset curl handles to a mutex-protected idle pool of at most 32; a handle belongs to exactly one
caller, and each caches at most two connections. Reset detaches request callbacks and headers while
retaining connection/DNS/TLS caches. Global cleanup drains the pool after callers join, and
`UFSM_HTTP_REUSE=0` retains the close-on-worker-exit control. The local fixture verifies 60 fresh
workers use one connection on one origin or two on two origins, against 60 connections with reuse
disabled. Three 64-worker rounds retain exactly 32 connections between rounds, with stable 68 file
descriptors and cleanup back to 4. Mixed HEAD/range/whole/absolute-URL requests verify method,
range, callback and bearer reset; closed idle connections and an interrupted response recover.

Thirty-eight isolated prediction comparisons cover two native dense regions, cached and local
reads, a longer dense region, a coarse-level region and an air-heavy region. MANBp takes
8.06 / 7.35 s with the frozen reader versus 5.27 / 5.29 s with reuse (1.46x); PHerc0500P2 takes
8.37 / 7.66 versus 5.26 / 5.29 s (1.52x). A 2048-by-1024-by-1024 MANBp region takes
12.48 / 12.82 versus 8.85 / 8.82 s (1.43x). A 1024-by-8192-by-1024 region skips 48 of 64 tiles
and takes 17.57 / 17.16 versus 10.81 / 10.92 s (1.60x). Reuse-disabled candidates return to the
control timings. The coarse-level region gains only 1.12x and has no wholly empty tiles, despite
the historical `air-cold` artifact name. Cached/local trials remain within 1%. Every full prediction
store, including metadata, and every independently populated cold cache match exactly. Warm cache
bytes are not compared. The final rebuilt executable independently reproduces the MANBp gain
(7.83 / 7.58 versus 5.36 / 5.44 s, 1.43x) and retains the identical CUDA fatbin. These are bounded
cold-I/O gains; no GPU-kernel, training or whole-volume improvement is inferred.
All 36 Make test commands pass on the final binary (`http-reuse-regressions.json`). Four bracketed
512-cubed training checks give 74.02 / 73.62 Mvox/s controls and 73.75 / 73.69 Mvox/s candidates,
all at 14796 MiB with identical executed precision manifests. These cached-sampler checks establish
no material training regression and no training gain (`http-reuse-train-bench.json`). Precision,
calibration and seam acceptance remain open; long production training stays on hold.

Complete FP4 steady-profile controls put weight gradients at 41.9%, backward-data at 24.9%, forward
convolutions at 21.2%, normalization at 8.0%, upsampling at 3.5%, and upload/loss/optimizer at 0.7%
of the profiled time. Halving both gradient-convolution components would give a hypothetical 1.50x
step speedup; halving all three convolution components would give 1.78x. These are component-budget
calculations, not achieved gains or forecasts. Substantial training and cached-inference gains need
less convolution operand preparation/repacking and better reuse; the accumulator probe provides no
faster arithmetic replacement for the current block-scaled FP4/FP8 path. No further 2x pipeline gain
has been demonstrated, and the broader precision/quality/seam qualification remains incomplete.

The corrected legacy-contract 17-source paired trial has finished (`production17-v2-summary.json`
and `production17-v2-group-diagnostics.json`). At the provisional cutoff 0.6, staged FP4 mean F1 is
0.2969 versus control 0.3304, but eight partial segment boxes have 55–65% positives within their
labelled support and uninformative all-foreground F1 of about 0.70–0.79. On the eight dense HF boxes,
best sampled global-cutoff mean F1 is only 0.0770 / 0.0803 / 0.0801 for FP4 / FP8 / FP16, versus
0.0787 for the control. These held-out cutoff optima are diagnostics, not independently calibrated
acceptance. Prediction-only throughput is 197.2 / 162.5 / 129.1 Mvox/s across all 16 boxes, using the
legacy numerical contract; it must not be attributed to the new stored-GN/FP8-stem candidate.

The recovered work and benchmark artifacts are in `runs/recovery512/`. Dirty experimental branches
were archived before changes were integrated. The stochastic-rounding seed was missing from the MX
activation / FP8 weight-gradient dispatch; the fix restores step-dependent rounding and has a regression
test. Prediction and scoring errors now fail the confirmation instead of producing incomplete score rows.
Scoring respects each source's minimum available label resolution.

The first 17-source staged confirmation is complete (`production17-summary.json`): 780 s of
128-cubed warmup, then 780 s of 512-cubed training versus continuing at 128 from the same
checkpoint. The large stage made 398 updates at 0.51 samples/s; the control made 6470.
At the provisional threshold 0.6, mean F1 was 0.3074 for large-stage FP4 inference and 0.3320
for the control. Large-stage FP8 scored 0.3081 and FP16 0.3367. Prediction-only throughput
over all 16 boxes was approximately 200 / 165 / 130 Mvox/s for FP4 / FP8 / FP16. These are
short, single-seed diagnostics, not a production acceptance. The segment-label scores need
comparison with their constant-foreground baseline; their high surface fraction can make
unselective predictions look good.

That confirmation used the frozen earlier binary. Subsequent audit fixed overlapping optimizer
ownership: Muon/ANVIL conv weights also received AdamW decay, and stale Adam moments could
change them. Packed AdamW and EMA also traversed nonmonotonic offsets and updated overlapping
ranges. Each parameter now has exactly one optimizer owner. Validation sampling no longer
snaps outside its holdout; training excludes the union of holdouts for sources sharing a CT;
read failures discard incomplete batches and fail the run. New checks fail on the earlier code
and pass on the fixes. New checkpoints also capture stem-input MX quantization separately;
prediction restores it without enabling gradient buffers. Older embedded headers infer this
setting from the saved gradient mode. Previous "matched" inference did not restore it, so
the precision comparison must be repeated before choosing the production profile.

The complete `make -j8 test` passes, including optimizer ownership, input-format transitions,
portable checkpoint/resume, model-buffer reuse, sampler failures, both spatial-split presets,
and the production runner. Evaluation's common radius-2 dilation uses separable passes;
6480 thin-volume and boundary cases match the scatter reference exactly. Soft Dice is computed
once per volume. Structured scores retain exact thresholds and annotation support, and
`eval --seam-core 512 --seam-band 16 --scores report.json` separates internal tile-boundary
scores from interior scores. Full training remains held pending corrected precision/accuracy
confirmation, calibration, real-volume seam validation and remaining performance checks.
On an actual 1024-cubed segment holdout with six thresholds, two timings per implementation
measured 26.27 / 26.01 s before and 18.78 / 18.59 s after: 1.40x faster scoring with identical
printed metric tables (`eval-radius2-bench.json`). This gain is in CPU scoring, not GPU inference.

The native-resolution MANBp halo check is complete (`seam17-summary.json`, 1024 cubed,
eight tiles, the earlier short 17-source checkpoint). Matched-input inference with a 512-cubed
core took 4.51 / 4.84 / 5.58 / 7.34 s for halos 8 / 16 / 32 / 64 (windows 528 / 544 / 576 / 640).
Diagnostic best F1 over the sampled cutoffs was 0.1048 / 0.1012 / 0.1030 / 0.1073.
Halo 8 had seam/interior F1 0.1067 / 0.1046 at its best sampled cutoff; the larger halos
did not establish a quality gain worth their extra prediction cost on this box. This one
undertrained model is insufficient to qualify production seams. The corrected legacy-contract
17-source confirmation completed as `production17-v2-*`; its large stage and paired control use
seed 2 to avoid restarting the warmup's sample streams. Results are recorded above. No long
production model run has started.

Single-GPU batch-1 measurements with the recovered memory changes:

| training window | steady samples/s | million voxels/s | observed peak GPU memory, MiB |
|---|---:|---:|---:|
| 480 | 0.623 | 68.9 | 11826 |
| 496 | 0.560 | 68.3 | 13010 |
| 512 | 0.515 | 69.1 | 14272 |
| 528 | 0.461 | 67.9 | 13280 |

These are short runs, not accuracy comparisons. 512 leaves about 2 GiB unused on the 16311 MiB card.
Dropping the saved coarse decoder SiLU buffers and packing four-channel MX inputs into eight-wide
rows recovered that margin. 528 selects lean 2 and has more margin but no throughput advantage.
The first kernel sweep's apparent improvements with shorter weight-gradient chunks were invalid:
their spatial launch grid exceeded grid.z's 65535-block limit, so some gradients were never computed.
Low-precision kernel launch errors had been cleared into a separate buffer which the trainer did not
check. The correction joins that buffer into `nn_check` and puts the spatial grid in grid.x, which has
a larger limit. Those benchmark runs must not be used as accuracy or speed evidence. Weight gradients
accounted for about 42% of the valid original step; forward and backward-data also need examination
to establish remaining speed headroom.

After the grid fix, the valid paired 24-step timings at 512 are 0.514 samples/s with FP4 stride-1
weight gradients and 0.505 with FP8 weight gradients. ZC=4 takes 0.482 and ZC=1 takes 0.345, so
the existing ZC=12 remains the choice. Both complete all weight-gradient operations. The full test
suite passes, including both split precision presets and the new large-grid/error-propagation tests.
The execution manifest now reports actual dispatch separately from requested policy: the input conv
is FP8, most stride-1 convolutions use FP4, downsampling uses FP8 forward/weight-gradient and FP32
backward-data arithmetic, and the head uses FP32. Storage and FP32 accumulation are separate from
the operand precision; master weights and optimizer state remain FP32 unless explicitly quantized.

Inference at window 528 / halo 8, on the same cached MANBp 1024-cubed output and checkpoint, took a
median 4.97 seconds with the previous host pipeline and head, 4.35 with device statistics and shard
placement, and 4.28 with the new head. Both changes produced byte-identical output across all
1,073,741,824 voxels. These end-to-end measurements include startup and writing. The GPU statistics
and clipped placement also have CPU-reference tests. Sparse boxes can favor smaller windows by
skipping more air; window 288 / halo 16 took 3.49 seconds here. A model trained at 512 still needs an
accuracy comparison across inference sizes because its GroupNorm statistics depend on the window.
With profiling and error propagation enabled, the same FP4 inference takes about 4.43 seconds versus
5.37 with FP8 compute/storage. FP16 input gives 4.29 and 5.40 seconds respectively (two repetitions).
These numbers are preliminary throughput measurements, not an accuracy selection for changed storage.
Forcing three resident blocks on the register-heavy 16-channel FP4 convolution preserved the output
but made inference about 5% and training about 4% slower; that variant was rejected. Batching complete
staging rows reduced live registers without recomputing the input and was also byte-identical, but
inference was about 3% and training about 1.5% slower. Neither experiment is integrated; the measured
stock kernels remain the choice. Their source and results are retained under `runs/recovery512/`.

The recovered 17-source confirmation now has scores for all 16 held-out boxes. Its FP4-minus-FP16
mean best-threshold hard F1 is -0.0025, with a worst difference of -0.019 on MANBp. Both models were
trained at 64 and scored at 528, and eight segment boxes have uninformative all-foreground optima.
This is diagnostic evidence, not a production precision acceptance test. Large-window paired runs,
threshold calibration, seam checks, and a final precision choice remain necessary. `train --seconds`
and `--warmup-seconds` now allow comparisons using an actual wall-clock budget and time-based LR
scheduling instead of guessed step counts.
Accuracy trials can also score `predict --ema 0` (current weights) as well as the default EMA weights.
EMA's horizon is measured in optimizer steps, so large-window trials with few steps can otherwise
look worse merely because their EMA has not caught up. The output metadata records which view was used.
Held-out prediction reuse now checks hashes of the checkpoint, binary, precision manifest and axis,
plus the prediction arguments and UFSM environment settings. Changed settings trigger a new prediction
in a staging directory; the previous output is replaced only after success. `--score-only` explicitly
scores an existing prediction without regenerating it. CPU regression tests cover reuse, invalidation,
failed replacements, legacy outputs and minimum label resolution.

The equal-time MANBp trials have completed (780 seconds each, batch 1 at 512 versus batch 4 at 128).
Both process roughly 69 million voxels/s, but finish 398 versus 6467 optimizer updates. Best-threshold
held-out hard F1 is:

| training window | prediction window / halo | current weights | EMA weights |
|---|---|---:|---:|
| 512 | 528 / 8 | 0.2415 | 0.2076 |
| 512 | 288 / 16 | 0.2181 | 0.2042 |
| 128 | 528 / 8 | 0.3771 | 0.3772 |
| 128 | 288 / 16 | 0.3769 | 0.3679 |

The larger prediction window helps the 512-trained model on this box, but removing EMA lag does not
close the training-window gap. These are one-seed, one-box diagnostic optima, not calibrated production
scores. The production run remains on hold. Four times the learning rate did not close the gap:
current-weight F1 was 0.2336 with FP4 weight gradients and 0.1717 with FP8 (EMA 0.2092 / 0.1812).
The FP4 trial completed 398 updates, the FP8 trial 381; both were finite and free of CUDA errors.
These results do not establish a precision ranking across sources, but provide no reason to switch.
The corrected tile sweep completed all eight runs. Controls at both ends were 0.501 samples/s;
alternatives were 0.383–0.501. None improved speed, so the existing tile choices remain in use.

Checkpoint resume now preserves loaded weights (the former conditional reinitialized them unless
`--finetune 1` was specified). New checkpoints also include Muon momentum; older checkpoints load
with zero Muon momentum. Regression tests verify that the next AdamW/Muon update after save/load is
identical to uninterrupted training, reject missing declared momentum, and preserve old-format support.
A real CLI test preserves trained parameters at zero learning rate and reproduces the reinitialization
with the previous binary. The full suite passes. The staged follow-up has completed: both models start
from the same 13-minute 128 checkpoint and get another 13 minutes. At prediction 528 / halo 8, switching
to 512 gives current-weight best-threshold F1 0.3942 versus 0.3803 for continuing at 128. EMA scores are
0.3850 versus 0.3849. At prediction 288 / halo 16, the corresponding current-weight scores are 0.3823
and 0.3792. Thus the warmup closes the short-run large-window gap on this one seed/box, while 528 helps
the staged 512 model. Both trials resumed a legacy checkpoint with zero initial Muon momentum; new
checkpoints save it. This does not establish the recipe across sources or calibrate a production threshold.

`tools/production.py` now freezes the binary, recipe, sources/axes and scoring script, performs staged
training with continuous optimizer steps and fresh stage LR schedules, compares matched/FP4/FP8/FP16
inference at a fixed threshold, and exports a hashed model bundle. Deployment verifies its artifacts
and replaces predictions only after success. GPU leases cover cooperating runners. Local real-CLI tests
exercise training, resume, the four inference profiles, bundle relocation, reuse, failed replacement,
and changed-artifact rejection. Checkpoints embed inference storage and requested precision defaults;
legacy sidecars remain supported. The default threshold 0.6 is provisional. The candidate recipe and
its long training run remain under validation; no full production run has started.

The sampler now filters source/level combinations that cannot fit the window, validation box or region,
and fails immediately when none are eligible. In particular, Kaggle's 320 cubes participate in the 128
warmup and are explicitly excluded from 512 training instead of repeatedly drawing impossible patches.
The full suite, including the real CLI impossible-source case, passes.

Inference now reuses consumed encoder skips for final decoder outputs and kept coarse SiLU buffers.
At 528 / halo 8 on the cached 1024-cubed benchmark, FP4 peak memory falls from 7017 to 6053 MiB and
FP8 from 10753 to 8945 MiB. Both outputs are byte-identical to separate buffers across 1,073,741,824
voxels. FP16 previously failed to allocate a 1.18 GB buffer; it now completes at 14439 MiB. Three-run
median end-to-end times are FP4 4.34 -> 4.44 seconds and FP8 5.57 -> 5.53; this is a memory change,
not a demonstrated speed gain. FP16 takes 7.49 seconds. Repeated-forward tests cover two/four levels,
recompute 0/1/2 and all three storage modes. FP16 group-accumulator tests use an unchanged-repeat
reference because atomic GN reduction order itself can change logits by about 0.1%. The full suite
passes. `UFSM_INFER_SCRATCH_OLD=1` retains separate buffers for comparison.

The legacy-contract 17-source confirmations completed through the production runner. The fresh
stored-value contract/FP8-stem confirmation uses 13 minutes at 128 followed by 13 minutes at 512,
with the same warmup checkpoint continued at 128 as a control. Muon state is preserved in both.
It scores all 16 configured hold-outs and matching/FP8/FP16 profiles. Artifacts are under
`runs/recovery512/production17-stored-*`; this is a bounded trial, not the full model run.

## Constraints
- Host code is C23 (`gcc -std=c23`). GPU kernels are `.cu` files compiled by nvcc and linked into the
  same binary. No cuDNN, no cuBLAS: every kernel is ours.
- Third-party: libc/libm/pthreads, libcurl (HTTPS), libzstd, libblosc + zlib (only to read upstream
  zarr v2 / TIFF during ingest), CUDA runtime, and the vendored codecs `third_party/volcomp.h` and
  `third_party/surfcomp/` (both MIT, SuperOptimizer).
- Hardware: 2x RTX 5060 Ti 16 GB (sm_120), 32 cores, 182 GB RAM, `/vesuvius` nvme (≈790 GB free).
  Both GPUs are under this session's exclusive control during the current audit.

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
6. `predict`: sliding-window inference with halo cropping into output shards, writes a volcomp zarr v3 pyramid in the
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

### Performance program toward the hardware ceiling (goal set 2026-10-01 afternoon)
Roofline per level-0 voxel from the layer shapes: forward 139 kFLOP (enc0 3.5k + 13.8k, dec0 41.5k + 13.8k, head
0.06k, down0 3.5k; level 1 /8: 43k; level 2 /64: 18k; level 3 /512: 0.6k), training ~417 kFLOP. 96^3 B2 = 1.77M
voxels: training 738 GFLOP in 33.8 ms = 22 TFLOP/s, inference 246 GFLOP in 9.6 ms = 26 TFLOP/s. CORRECTION from
the agent's per-conv profile (tests/prof_infer): on this GeForce card the fp16 MMA with fp32 accumulation peaks at
54 TFLOP/s (108 measured with fp16 accumulation, prec 4; fp8 ~215), so the 16-bit convs with >= 32 channels already
run at 80-95% of their ceiling and the level-0 16-channel ones at ~75%; the gap to "2x" is NOT in the 16-bit path.
Floors for inference at 96^3 B2: MMA 4.3 ms (16-bit, fp32 acc), 1.1 ms fp8, 0.54 ms fp4; memory 1.9 ms (16-bit
storage), 1.0 ms MX. Realistic ceilings: ~6 ms 16-bit (4.5 with prec 4), ~3 ms fp8 + MX (3.2x over today's 9.6).
Measured today on an idle GPU: 9.6 ms fp16, 6.4 ms MX storage. Decision (user: "just as fast as possible"): MX
activation storage is the inference default (`predict --act-mx8 1`, free in accuracy: r8 0.2670 vs 0.2669); training
moves to all-fp8 compute + stochastic rounding (+0.011 F1 vs fp16 in paired 6000-step runs) with fp16 accumulation
where anything stays 16-bit, behind a paired-run accuracy guard (within 0.005 F1) and the long paired run.
Second correction (agent, prec 4 test): the 16-bit convs are NOT MMA-bound either. Doubling the MMA peak with fp16
accumulation leaves the level-0 16-channel convs unchanged (0.87 ms) and gains only 15-30% at levels >= 1, so they
are issue / staging / ldmatrix-bound at ~41 TFLOP/s, 2.4x above their memory floor. Fix in progress: a conv kernel
family with M = output positions, N = output channels, K = input channels, the weights resident in registers across
all 27 taps (one activation ldmatrix_x4 and two MMAs per tap per 16 positions, no weight loads), deeper z per block;
16-bit first, then fp8 m16n8k32 with two taps per K block. Target ~2x on enc0.c2, dec0.c2, dec0.c1 (122 of 232 GF).
First cut (d6c8438): 4 output planes per block for the 16-channel convs (halo restaging 2.8x -> 2.1x, weights loaded
once per 512 outputs): enc0.c2 / dec0.c2 0.87 -> 0.75 ms, dec0.c1 2.86 -> 2.58, training step 66.4 -> 63.0 ms on the
shared GPU, gradient error unchanged. Diagnosis with staging-only / MMA-only builds: the two halves (0.58 / 0.54 ms)
each sit near their own limit and barely overlap (block stages the whole tile, syncs, then multiplies), so the next
kernel is a persistent double-buffered producer / consumer (producer warps stage + GN + SiLU + convert tile i+1
while consumer warps run the 27-tap MMAs on tile i, fp16 accumulation, weights resident per block); target
~0.45-0.5 ms per 16-channel conv, the fused block kernel reuses the structure.
Pipelined producer/consumer kernel tried and dropped (0.83 vs 0.75 ms: producer- and bandwidth-bound; a plain copy of
the conv's I/O takes 0.32 ms, the conv 0.75, i.e. 2.4x from the 2.1x halo re-read at the largest tiles that fit
smem). x-shift packing for enc0.c1 (K = 16 = 4 kx shifts x 4 channels, 9 tap rows instead of 27 padded): 0.62 ->
0.46 ms (19f3eab). 16-bit conv time 9.65-10.2 -> 8.84-9.04 ms contended at 96^3 B2; the rest of the 16-bit items are
3-6% each, so the effort moves to the MX/fp8 path (dec0.c1 fused upsample in MX staging, 16-channel MX convs with
two taps per k) where 1.5-3x remains.
Plan: (1) per-op profile (agent, done: fp8 on 16-channel convs is padding/staging-bound, Ci = 16 pads to the fp8
K = 32; enc0.c1 pads Ci = 4 to 16; stride-2 convs latency-bound at 4-10 TFLOP/s; head and GN small); (2) agent's
kernel changes in payoff order: dec0.c1 on the MX path without the transient (-14% at 96^3), K-packing of two taps
per k for 16-channel fp8 / MX convs (-12%), stride-2 kernels (-0.4 ms), then the fused block kernel (~-0.5 ms);
(3) fp8 activation storage with down_norm and inference-only mode (agent, done: 67dcaa4);
(4) inference pipeline: writer-thread overlap, window-size sweep for L2 residency (lead); (5) training: batch 2 per
GPU at 128^3, tiled Muon / ANVIL matmuls (lead); (6) wider level-0 channels once the kernels are efficient.

- Done so far on the lead's side: (4) predict shard writes on a writer thread with a double shard buffer (output
  identical); window-size / batch sweep on the shared GPU shows flat per-voxel throughput from 96^3 to 288^3 and no
  gain from batch 2 per GPU, so there is no cheap win in window or batch choice; (5) Muon / ANVIL cascades use
  shared-memory tiled products (tile_xxt_k, tile_bx_k; identical losses, ~11% faster than the per-conv path on a
  shared GPU). fp8 activation storage at inference is free on an fp16-trained model (r5: F1 0.197 vs 0.192).

### Throughput in voxels per second (user's metric, 2026-10-01 evening)
- Useful throughput = raw voxel rate x interior fraction of the window (halo 16): 160^3 51%, 288^3 70%, 416^3 78%,
  544^3 83%. Raw per-voxel rate is flat with window size (sweep above), so bigger windows pay through the halo only;
  inference memory (~133 B/voxel fp16, about half with MX) bounds the window: 544^3 needs MX. Per GPU: fp16 today
  184 Mvoxel/s raw (129 useful at 288), MX storage 276 (193), the ~3 ms ceiling ~590 (413 at 288, 490 at 544);
  two GPUs double it, i.e. ~1 Gvoxel/s useful at the ceiling (a 2.7-Tvoxel scroll at 2.4 um in ~45 min).
- `predict --gpus 0,1`: one worker process per GPU over the shards of one output store (fork before CUDA init, the
  writer has no global state), the parent builds the pyramid. 1024^3 box at level 1: 13.8 s -> 6.6 s on two shared
  GPUs, output identical. Window 416 is slow (stride 384 does not divide the 512 shard); window 544 (stride 512,
  shard 1024) aborts in the MX path ("MX-fp8 storage needs MX inputs and outputs", 544/8 = 68): reported to the agent.
- fp4 everywhere (user): inference-only packed e2m1 activation storage first (the level-0 convs are bandwidth/halo
  bound, so bytes are the lever), then fp4 forward in training with fp8 + SR backward, then the NVFP4 recipe for the
  backward operands (random Hadamard, SR, 2D weight scales, wide first/last layers), each step behind the paired
  guard (within 0.005 F1). Earlier fp4 numbers (QAT fine-tune 0.242 vs 0.297, simulated NVFP4 62-66% error) were
  without those ingredients.

### fp4 storage study (step 0 of the fp4 plan, 2026-10-01 evening)
Paired 6000-step yardstick (MANBp, det, Muon, down_norm), fake MXFP4 (e2m1, ue8m0/32, RN) applied in training:

| arm | F1 0.5 | F1 0.7 | val |
|---|---|---|---|
| fp16 reference | 0.212 | 0.300 | 0.907 |
| raw pre-GN a1/a2 in fp4 (`UFSM_FAKEQ_WHERE=0`) | 0.227 | 0.287 | 0.918 |
| post-GN+SiLU operand s2 in fp4 (1) | 0.247 | 0.294 | 0.913 |
| pre-GN a1/a2 after the previous-step affine (2) | 0.217 | 0.314 | 0.917 |
| activation GRADIENTS in fp4 (`UFSM_FAKEQ_GRAD`) | 0.089 | 0.082 | 1.119 |

Every activation-storage variant is inside the 0.02 band (the earlier "62-66%" was a one-step gradient error, not a
training outcome; the network adapts to the grid); fp4 gradient storage destroys training. Decisions: phase 2 ON
(training a1/a2 and down outputs become packed fp4 as well, raw first, the delayed-stats affine only if the real
kernels show a gap); gradients stay MX-fp8 / fp16; fp4 in the backward is restricted to the GEMM operands.

Real kernels (c5f70d0, every activation stored as packed e2m1 + ue8m0, `UFSM_ACT_MX4=1`), same paired yardstick:
F1 0.214 / 0.291 at 0.5 / 0.7, peak 0.293, val 0.909, 0 skips, against fp16 0.212 / 0.300, peak 0.301, val 0.907.
Inference with mx4 storage on the fp16-trained r8 model: peak 0.277 vs 0.284 (calibration shifts down; models
trained with mx4 do not need that). Speed at 96^3 B2 (agent, quiet GPU): inference 8.91 fp16 / 7.47 mx8 / 7.36 mx4
ms, memory 0.228 / 0.205 / 0.111 GB; training 32.8 / 33.3 / 33.6 ms (the 16-channel fp8-path convs and the MX GN
backward are the regressions being fixed), memory 0.540 / 0.458 / 0.363 GB.

### Whole-box scanning throughput (2026-10-01 evening, master 3a8ddc9, both GPUs idle)
`predict --gpus 0,1 --halo 8`, MANBp level-1 box 2048^3 at (3000, 2000, 2000), r8 model: fp16 window 272 19.5 s
(441 Mvoxel/s useful), mx8 272 13.5 s (637), mx8 528 14.3 s (599), mx4 528 13.7 s (626). The 528 window does not
pay here: tiles are 8x larger, so fewer of them are pure air (40 of 64 run vs 244 of 512 at 272), which cancels the
better interior fraction; on a compact mask like Paris 4 (27% vs 28% non-empty at 128 vs 512) it would. fp4 storage
is not faster end to end yet (the 16-channel fp4 kernel is step 5b). At 637 Mvoxel/s Paris 4's ~15.3 Tvoxel of
occupied tiles take ~6.7 h.

### GroupNorm backward (2026-10-01 evening)
Timeline of a fp16 training step (nsys): conv forward/backward-data 47%, conv weight gradient 29%, GroupNorm backward
17%, the rest 7%. tests/bench_gn vs a device copy of the same bytes: level 0 (96^3 x 16) and level 1 (48^3 x 32) run
at the DRAM rate already; levels 2-3 and the down-conv norms were launch-bound (a fixed 32 slabs per channel gave
blocks of ~50 elements). Slab count now adapts to >= 8k elements per block (b83f932): 24^3x64 0.127 -> 0.019 ms,
12^3x80 0.154 -> 0.014 ms; whole step (prof_infer, same GPU, A/B builds): GN 4.20 -> 3.28 ms, step 33.16 -> 32.23 ms.
A fused one-block-per-group kernel was slower except at 12^3 and was dropped; the same change to the conv bias
reduction made no measurable difference. What remains of the GN backward is at the memory floor; the next step there
is fusing its two passes into the conv epilogue / staging (kernel agent).

### fp4 training stairs (2026-10-01 evening; 6000 steps, MANBp, det, Muon lr 0.01, down_norm, mx4 storage, SR on
gradient operands; peak F1 over cutoffs; fp16 references 0.291 (seed 0) / 0.312 (seed 1))

| stair | policy | seed 0 | seed 1 |
|---|---|---|---|
| S1 fp4 forward | all=fp4:fp8:fp8 | 0.304 | 0.305 |
| S1 + fp4 weight gradient (Hadamard + SR on x) | all=fp4:fp8:fp4 | 0.294 | |
| S1 + fp4 weight gradient (plain) | all=fp4:fp8:fp4 | 0.287 | |
| S2 fp4 backward-data | all=fp4:fp4:fp8 | 0.299 | 0.303 |
| all fp4 | all=fp4:fp4:fp4 | 0.281 | 0.302 |

(enc0.c1 stays 16-bit in every policy.) Presets: `train --fp4 1` = S2 (fastest that passed: mx4 storage, fp4 forward
and backward-data, fp8 weight gradient, SR), `--fp4 2` = all fp4; `predict --fp4 1` = mx4 storage + fp4 compute. One-step gradient errors (0.7-0.9 whole-net with mx4 storage, 14% for the fp4
weight gradient alone vs 3% fp8) do not predict the outcome: every stair so far trains within 0.02 of fp16. Step time
at 96^3 B2 (agent, idle GPU): fp16 32.7, S1 29.8, S2 29.5 ms; the fp4 weight-gradient kernel is slower than fp8 (step
32.0 / 33.3 ms with Hadamard) and is being sped up before it can be a default. One failed fp16 seed-1 reference
(peak 0.100, memorised a narrow set) traced to a transient remote open failure silently dropping a level under the
parallel eager open (fixed in 970ce89 with retries); its rerun on the same build trained normally.

### Run lengths (2026-10-01 evening)
- Under Muon the 6000-step yardstick reaches the F1 that AdamW needed 120k steps for, and r10's validation loss has
  been flat (0.94-0.99) since step 50k of its constant-lr phase. Decisions therefore use the 6000-step paired
  yardstick (6 min); long-run confirmations are 40k steps with `--sched wsd --cooldown 0.2` (~30 min on two GPUs,
  the cooldown is where the accuracy lands), not 120k-200k. r10 (200k) runs to completion only because its cooldown
  and 17-box scores are the baseline; r11 onwards are 40k.

### 40k-step confirmations r11a/b/c (2026-10-01 afternoon, configs/all2_cached.json, 9 sources, P64 B4 x 2 GPUs,
WSD cooldown 0.2; F1 at each box's best cutoff, 7 held-out boxes)

| run | config | mean dF1 | worst box | vs | samp/s | best val |
|---|---|---|---|---|---|---|
| r11a | fp16 | +0.110 | -0.092 (MANBp) | r8 (120k AdamW) | 366 | 0.854 |
| r11b | `--fp4 1` (S2) | +0.002 | -0.016 (Paris4) | r11a | 364 | 0.863 |
| r11c | `--fp4 2` (all fp4) | +0.009 | -0.008 (0500P2) | r11a | 340 | 0.860 |

Both fp4 presets pass the confirmation gate (mean within -0.02, no box below -0.05). The two seg boxes are degenerate
(recall 1.0, same F1 for every model). End to end at P64 B4 the presets do not run faster than fp16: the 64^3 step is
not compute-bound (launches, sampler, small grids), and `--fp4 2` is 7% slower because the fp4 weight-gradient kernel
is still slower than fp8. The fp4 speed shows at large windows / batches (prof_infer 96^3 B2) and in whole-box
inference; the training default becomes `--fp4 1`, `--fp4 2` waits for the faster weight-gradient kernel.

End-to-end training at P128 B4 x 2 GPUs (400 steps, all2_cached, sampler wait 0, master 8f30cde): fp16 51.9,
`--fp4 1` 60.8 (+17%), `--fp4 2` 55.6 samples/s = 0.109 / 0.127 / 0.117 Gvoxel/s. At P64 B4 the presets tie
(fixed per-step costs dominate). `ufsm train` now defaults to `--fp4 1`; `predict` defaults to `--fp4 1` when the
checkpoint's precision.txt says act_mx4 1.

fp4 weight-gradient layouts (6000-step MANBp stair, all fp4, peak F1; fp16 0.291 / 0.312): layout 2 (x scale per
(ci, plane), 5-9% faster than fp8 per conv) 0.292 / 0.201 -- fails seed 1 (0.202 also under MX8 inference, so not
calibration); layout 1 (per-32 x scales, ~10% slower than fp8) seed 1 0.310; `--fp4 1` on the same master seed 1
0.308. Default layout is 1; `--fp4 2` stays opt-in (slower than `--fp4 1`).

Equal voxel budget window study (50.3 Gvox, MANBp, `--fp4 1`, seed 0, peak F1 / band F1): 64^3 x 32 0.316 / 0.554,
128^3 x 4 0.343 / 0.586, 256^3 x 1 (3000 steps) 0.295 / 0.519.

MX-fp8 gradient storage (UFSM_GRAD_MX8, 6000-step MANBp stair under `--fp4 1`, peak F1, seeds 0/1/2): 0.307 / 0.284 /
0.288 (mean 0.293) vs `--fp4 1` 0.299 / 0.308 / 0.276 (mean 0.294): passes; `train --mem auto` (default) may pick it.
Memory per level-0 voxel under `--fp4 1`: 16-bit gradients 212 B, MX-fp8 gradients 156 B (step time equal).

Largest training windows (master after the 512^3 merge, `--fp4 1 --mem auto`, B1 per GPU, 2 GPUs data parallel,
all2_cached): planner order is by step cost (default, MX-fp8 gradients, + chunked, + recompute 2, + lean; the 16-bit
chunked modes only under `--mem auto16`). 448^3 picks MX-fp8 gradients (15.4 of 16.4 GB), 1.39 samples/s = 0.125
Gvoxel/s; 512^3 picks MX-fp8 + chunked + lean (15.8 GB, peak 15.85 measured), 0.81 samples/s = 0.109 Gvoxel/s (lean
loses the upload overlap). 16-bit gradients top out at ~448^3.

Lean 2 merged (no gradient buffer B): 2 GPUs data parallel, B1 per GPU, `--mem auto`, all2_cached, 20 steps:
512^3 MX-fp8 chunked lean 0.94 samples/s = 0.126 Gvoxel/s; 544^3 + lean 2 0.81 = 0.130; 576^3 + recompute 2 + lean 2
0.62 = 0.118. Per-voxel rate is flat up to 544^3 (single GPU: 66 Mvoxel/s at 448-544, 59 at 576).

Scoring correction (2026-10-01 night): the 6000-step stairs scored best.ckpt, chosen by an 8-batch validation loss;
several runs picked a half-trained checkpoint (f2s0 step 2500: 0.234; its rerun with the same flags picked 6000: 0.308).
Re-scored with last.ckpt (end of the cosine schedule): fp4 wgrad layout 2 seed 1 0.203 (still fails); `--fp4 2`
(layout 1, MX-fp8 gradients) 0.299 / 0.284 vs `--fp4 1` 0.307 / 0.284 (passes; with 16-bit gradients seed 0 0.297);
MX-fp8 gradients seeds 0-2 0.307 / 0.284 / 0.284 vs `--fp4 1` seed 2 0.285 (passes); windows at equal voxel budget
64^3 0.300, 128^3 0.347, 256^3 0.292. Stairs and confirm_run.sh now score last.ckpt. `ufsm train` defaults to
`--fp4 2` (fp4 weight gradients, per-32 x scales; 1-4% faster per step than `--fp4 1`).
### Spatial split of one window across the two GPUs (`train --split z`, 2026-10-01 night)
- Each GPU holds its z half of the window plus 2^(L-1-l) halo planes at level l on the side facing the other GPU (8 at
  level 0 for 4 levels): the local depth P/2 + 8 keeps every stride-2 parity and the U-Net shapes, so the unchanged
  kernels run on it (P must be a multiple of 2^L = 16). Every stencil consumer (3^3 conv, stride-2 conv, upsample, fused
  upsample) only needs the innermost halo plane: each produced tensor (a1, a2, down outputs, upsample transients, the
  gradients before a backward-data or upsample backward) gets it from the other GPU and has its outer halo planes zeroed;
  a gradient that feeds a reduction (GroupNorm backward, weight / bias gradient) gets all its halo planes zeroed first.
  GroupNorm statistics skip the halo planes inside the conv epilogues / sum kernels (split_t.zlo / zhi) and, like the
  GroupNorm backward sums (the parameter gradients stay per-GPU, their sum is in the gradient allreduce) and the loss
  statistics, are summed across the GPUs before use; the loss mask is zeroed on the halo planes. Parameter gradients are
  summed (not averaged) across the two halves.
- No P2P on these GeForce cards: exchanges are staged peer copies (14-17 GB/s, ~10 us small). Host side (src/split.c):
  two threads, one per GPU, take turns (nn / unet keep global state); the second to reach a collective issues it for
  both GPUs. Backward gradient halos are exchanged on a per-device communication stream while the weight gradient runs
  (UFSM_SPLIT_SYNC=1 turns that off; +1% at 256^3). Remaining cost at 256^3: the 8 halo planes (+6% compute, +3% at
  512^3) and ~35 halo exchanges / 35 GroupNorm sums per step (GPUs ~92% busy).
- Equality (tests/test_split, in make test, skipped with one GPU): 96^3 (and 128^3 B2) split vs one GPU, recompute
  0/1/2, down_norm 0/1, the --mem auto modes (chunked, MX-fp8 gradients, lean 1/2). fp16: loss within 3e-6, gradient
  relative L2 1.5-2.5e-4 = the single-GPU rerun difference (atomics). --fp4 1 with round-to-nearest: loss within 1.5e-8,
  gradient 3e-7..2.4e-2 against 0.10-0.13 for a 1e-6 input perturbation (flipped fp4 roundings amplify); --fp4 1 with
  stochastic rounding: gradient 0.068-0.080 against 0.065-0.076 between two rounding seeds of one GPU.
- Largest windows, B1, `--fp4 1 --mem auto` (P multiple of 16): 640^3 (MX-fp8 gradients, chunked, lean; 15.3 GB peak),
  704^3 (+ recompute 2, lean 2; 14.9 GB peak); 720^3 runs out of memory. Data parallel tops out at 576^3.
- Throughput (all2_cached, B1, idle GPUs, samples/s = windows/s; Gvoxel/s): 256^3 data parallel 8.88 (0.149), split 7.70
  (0.129); 384^3 data parallel 2.67 (0.151), split 2.43 (0.138); split 512^3 1.05 (0.141; data parallel with lean 0.94),
  576^3 0.715 (0.137), 640^3 0.51 (0.134), 704^3 0.35 (0.122). Per-buffer upload events (the next batch uploads during
  the step) made data parallel 384^3 2.58 -> 2.67.

### Sampler on the 17-source set (2026-10-01 evening)
- On configs/all2.json the sampler collapsed to 34 patches/s (16 workers) and got slower with 32: 331 lazy store
  opens (one remote metadata fetch each, 70-500 ms) serialised on the open mutex, paid by every worker's first draws
  (a 6000-step yardstick spent its first minute there; r10 paid it once). Fix (a1f95fa): the open runs outside the
  lock (the lock only publishes the pointer; a racing duplicate is closed; a failed level clears ct_present so it is
  not retried per draw), and sampler_start opens every (source, level, target) store eagerly on 32 threads: 22 s
  summed -> 3.7 s wall. Steady state under load: 143 patches/s at 16 workers, 188 at 32, with 53-58% of draws
  rejected by the coarse occupancy probe (14 ms each) on the sparse new sources. Two GPUs at the fp4 training
  ceiling (~20 ms per 128^3 step) need ~200 patches/s, so the sampler is no longer the limit on an idle box.

### Depth and window yardstick (2026-10-01 evening)
6000 steps, MANBp, det, Muon lr 0.01, down_norm, fp16; peak F1 over thresholds (band P/R):

| arm | voxels/step | peak F1 |
|---|---|---|
| 4 levels, P64 B8 | 2.1M | **0.301** (0.47/0.63) |
| 4 levels, P128 B1 | 2.1M | 0.175 |
| 5 levels, P64 B8 | 2.1M | 0.226 |
| 5 levels, P128 B1 | 2.1M | 0.274 |
| 6 levels, P128 B1 | 2.1M | 0.263 |

At equal voxels per step, eight 64^3 patches beat one 128^3 patch by a wide margin (the long runs r8-r10 used P128
B1 per GPU, i.e. batch 2: probably the wrong trade). Depth helps when the window is large enough to use it (P128: 4 ->
5 levels 0.175 -> 0.274) and hurts at P64. Batch is the confound; P128 B8 at 4 and 5 levels is queued to separate
window from batch. Batch held at 8: 4 levels at P128 B8 peaks at 0.393 (band 0.66 / 0.63) against 0.301 at P64 B8,
so larger windows do help detection once the batch is not starved, at 8x the voxels and ~3.3x the wall time per
step (1868 s vs 570 s for 6000 steps on a shared GPU); a compute-matched comparison (P64 B8 for ~8x the steps) is
the open question. Bug found: MX-fp8 GroupNorm statistics fail for 96/112 channels (12/14 channels per group do not
divide the 32-channel MX block); scored with fp16 storage meanwhile.

Scanning window (same trained model, only the inference window changes; MANBp box, halo 8, fp16 storage), peak F1:
model trained at 64^3: window 96 0.306, 160 0.306, 272 0.301; model trained at 128^3 (5 levels): 96 0.269, 160 0.273,
272 0.274. The GroupNorm-over-the-tile mismatch costs at most 0.005 and leans toward the training size; the scanning
window is a speed choice, not an accuracy choice. 528^3 needs MX storage (out of memory at fp16).
With 528/544 working (3a8ddc9, the old abort was a silent out-of-memory on the dec0.c1 upsample transient), r8 model,
standard scoring: peak F1 272/mx8 0.285 (best cutoff 0.6), 528/mx8 0.289 (0.5), 544/mx8 0.286 (0.5), 528/mx4 0.286
(0.4). Peak detection is unchanged; the probabilities shift down with larger tiles (GroupNorm normalises over the
tile) and again with fp4 storage, so a fixed 0.5 cutoff degrades while the best cutoff does not. Production output
must carry the calibrated cutoff of its configuration (or the model must be trained in that configuration).

### Largest training window, batch 1 per GPU (2026-10-01 evening)
Measured with `ufsm train` (Muon, down_norm, fp16 storage, recompute 1), peak device memory: 256^3 5.3 GB,
320^3 10.3 GB, 352^3 13.6 GB = 328 B per voxel, matching the per-voxel accounting (bench_mem overstates: it also
builds the inference network and an fp32 input). On the 16 GB cards the practical maximum is 352^3 at fp16 storage
(384^3 would need 18.6 GB). With fp4 activation storage (~190 B/voxel with fp16 gradients, ~145 with MX-fp8
gradients) it becomes 416^3 and 448^3 (both multiples of 32, so valid for a 6-level net). Two GPUs are data
parallel (batch 1 each), so the per-GPU limit is the window limit.

### Receptive field vs window (2026-10-01 evening)
- Per-voxel inference rate is flat with window size, so a larger window buys only the halo fraction (288: 70%,
  544: 83%, 800: 88%); 544 is the practical maximum (memory ~65 B/voxel with MX-fp8 storage, ~40 with fp4). The window
  is stride + 2 halo, so a 512 stride (one shard) needs 544 at halo 16 or 528 at halo 8. Halo on the r8 model, MANBp
  box, F1 at 0.5 / 0.7: 16 -> 0.2858 / 0.2277, 8 -> 0.2852 / 0.2294, 4 -> 0.2844 / 0.2261: halo 8 is free (inside
  noise), so it is the predict default (interior 89% at 528, 84% at 272 vs 70% at 288/16: +20% useful voxels at no
  cost). eval_holdouts uses 272/8.
- The 4-level net's theoretical field is ~150 voxels at level 0; the user wants the field to match the window, so
  depth is the lever (width bought nothing today): 5 levels ~300, 6 levels ~600 (fits the 544 window and 256^3
  training patches) for < 5% more FLOPs. Yardstick arms queued: P64 control, P128, 5 levels (P64 and P128),
  6 levels (widths 16/32/64/80/96/112, ~1.9M params, P128). The winner joins the fp4 forward in r11.
- Paris 4 at 2.4 um (54 Tvoxel, ~25% occupied): inference of the occupied ~15 Tvoxel takes ~9 h on two GPUs
  today (MX storage, window 544), ~4 h at the fp8/fp4 kernel ceiling, ~2 h at the hardware floor; skipping empty
  and interior-of-sheet volume with a coarse pre-pass is worth another 2-3x. Training is step-bound, not
  volume-bound: a full-quality Muon run is ~25-50k steps.
- Measured on Paris 4 (level 5, whole volume read in 0.9 s): 25.8% of voxels are inside the scroll mask, and the
  non-empty fraction of level-0 tiles is 27.3% at 128^3, 28.2% at 512^3, 31.7% at 1024^3. The mask is compact, so
  predict's existing per-tile air skip at stride 512 already removes 72% of the volume (15.3 Tvoxel left); smaller
  tiles buy almost nothing. Further reduction needs a papyrus-vs-air criterion inside the mask (sheet gaps), not
  better tiling.

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
- Merged with the agent's sampler work (7247eec: fused single-pass input write straight into the 16-bit batch, 65536-entry
  noise table, row-wise chamfer 6.3 ms per pass pair; fe35893: `--det 1` deterministic batch order for paired runs,
  170 vs 222 samples/s at 32 workers; tests/test_sample_ops in make test). MANBp, 16 workers under load:
  164 samples/s (was 44 at the start of the day); per patch: probe 26 ms, CT read 11, soft 17, x16 22, augment 5.
  The "store reads" on MANBp were the label pyramid (a local store without a cache layer), not misses.
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
- Muon variants (2026-10-15; `train --opt muon --muon-c 1 --hyperball 1`, either alone or both; `nn_muon_ext_batch`,
  `tests/test_muonc.c`; both 0 = Muon bit for bit, the old code path):
  - Muon-C (Qing & Li, arXiv 2609.09676): the Nesterov momentum of each 3^3 kernel goes through the orthonormal 3^3 DFT
    over the kernel axes (the paper's critical grid, extended to 3D: one Co x Ci channel-transfer block per kernel
    frequency, a bijection on the 27 taps, so the inverse lands exactly on the kernel support); each block is
    polarised on its own by complex Newton-Schulz (5 iterations, Muon's coefficients, X0 = B / (|B|_F + 1e-8), blocks
    with more rows than columns as their conjugate transpose), and the inverse DFT gives a real update. Conjugate
    symmetry leaves 14 of the 27 frequencies to polarise. Blocks are stored as interleaved complex and iterated by complex
    versions of Muon's tiled products (`gemm_nsc_k`), every frequency of every conv in one launch per stage (one
    Newton-Schulz entry per conv and frequency).
    Routing as the paper (every kernel larger than 1x1, strided ones included: the stem, the down convs). Scale:
    the paper's matched applied-update RMS against unfolded Muon (the RMS of exact-polar Muon for that conv,
    sqrt(min(Co, K) / (Co K)) for Co <= K), so `--muon-lr` keeps its meaning and the A/B compares geometry only.
    No new state. Cost (`bench_muonc`, production net, idle desk 5060 Ti): Muon 10.8 ms per optimizer step, Muon-C
    16.9 ms, Hyperball +0.6 ms (shared with three trainings: 48 / 72 ms; the first version, real embedding and one
    frequency per launch, was 145 ms there); scratch ~185 MB for 14 complex blocks per conv.
  - A/B (2026-10-15/16, desk, production recipe of l0pair.sh: 2.4 um side labels, P512, 2000 steps, --det 1, Muon lr
    0.02; val loss at 500 / 1000 / 1500 / 2000; seed 2 arms share seed-2 validation windows):

    | arm | seed | val 500 | 1000 | 1500 | 2000 |
    |---|---|---|---|---|---|
    | Muon | 1 | 0.6961 | 0.6744 | 0.6605 | 0.6539 |
    | Hyperball | 1 | 0.6948 | 0.6753 | 0.6634 | 0.6564 |
    | Muon-C | 1 | 0.7095 | 0.6674 | 0.6543 | 0.6488 |
    | Muon-C + Hyperball | 1 | 0.7047 | 0.6771 | 0.6616 | 0.6551 |
    | Muon-C, lr 0.012 | 1 | 0.7278 | 0.6719 | 0.6554 | 0.6504 |
    | Muon | 2 | 0.6753 | 0.6467 | 0.6296 | 0.6202 |
    | Muon-C | 2 | 0.6908 | 0.6457 | 0.6285 | 0.6189 |

    Muon-C is ahead from step 1000 in both seeds (-0.0051 and -0.0013 at 2000) and behind at 500 in both; the margin is
    under the noise floor (gn vs gn2 0.006; two Muon runs of one recipe on two trees, nondeterministic after step 20:
    0.0034). The paper's lower Muon-C lr (~0.6x) does not transfer (0.012: +0.0017 vs 0.02). Hyperball does not help
    at 2000 steps (+0.0025, +0.0063 on Muon-C); the paper's gains grow with run length, untested here. Both stay opt-in.
  - Hyperball (Wen, Dang, Lyu, Ma & Liang, arXiv 2606.16899): W <- R Normalize(W - eta R Normalize(u)) with R the
    conv's |W|_F at its first Hyperball step (the initialisation) and u the Muon or Muon-C direction; no weight decay
    on those convs. Applied to the convs whose output goes through a GroupNorm (the paper's scale-invariant
    matrices): all Muon convs except the down convs of a net without down_norm. eta = `--hb-lr-mult` x the
    scheduled Muon lr; the default 1/sqrt(2) makes the first step Muon's relative length (Kaiming init: |W|_F =
    sqrt(2 Co), Muon's step lr sqrt(Co) for Co <= K), while Muon's later steps shrink as |W| grows and Hyperball's do
    not. The radii are checkpointed (`"hb_R"` in the header) so a resumed run continues on the same spheres.
    With a level-0 norm that is not a live GroupNorm (`--l0-norm none / ema`) the level-0 convs are not scale
    invariant: exempt them too when that work is merged.
- r10 (2026-10-01 13:10): configs/all2.json (17 sources, full chunk prefetch, 1.1 um scan from level 2), down_norm,
  Muon lr 0.01, `--sched wsd --cooldown 0.2`, 200k steps at 128^3, effective batch 2 on both GPUs. Scored against
  the r8 baseline on all 17 held-out boxes when done.
- ANVIL II (modded-nanogpt record #90 / hyperstition.cc; `--opt anvil`, nn_anvil_batch): twin-rail velocity (fast
  rail beta 0.85 -> 0.93 over 240 steps, then 0.85 with the slow 0.98 rail blended 0.4385 / 0.5615 from step 514),
  Nesterov lookahead 0.95, normalisation by 1.05 x Frobenius, six quintic spectral maps with the record's
  coefficients, per-row energy equalisation at constant norm (lane EMA 0.9), sign-aligned weight decay (default
  2.25 x lr), the same sqrt(max(1, Co/K)) multiplier; AdamW on biases, norms and the head. The 600-step kaggle smoke
  run trails Muon (val 0.609 vs 0.548), expected before the slow rail engages; 6000-step yardsticks at (lr, wd) =
  (0.023, 2.25), (0.01, 2.25), (0.01, 0.1) against Muon's 0.307: peak F1 0.204 / 0.231 / 0.329 (validation loss
  0.914 / 0.924 / 0.887). The record's heavy decay is wrong for this network; with wd 0.1 ANVIL edges Muon
  (0.329 vs 0.290-0.307 over three Muon arms). Confirmation: wd 0.01 -> 0.310; seed 1: ANVIL wd 0.1 0.302 vs
  Muon 0.303. The one-seed ANVIL lead was noise; the two are equal within it and Muon stays the default (simpler,
  one fewer state buffer).
- Efficiency review (2026-10-15; modded-nanogpt, hyperstition.cc, a 3-D segmentation literature survey): ANVIL III is
  proprietary (no public rule); ANVIL II already equals Muon here. New, all off by default except the decay exemption:
  `--lr-floor F` (schedules end at F x peak; modded-nanogpt 0.3 with a 0.8 cooldown), `predict --ema-blend A` (A x final +
  (1 - A) x EMA, their 0.35), no weight decay on GroupNorm / biases / head under Muon and ANVIL (`--wd-nonconv 1` restores
  it; the gains decayed toward 0), `--recall W` (recto soft-recall term: the skeleton-recall loss of the Vesuvius recto model,
  whose thin labels are their own skeleton), `--shell R` (background within R level-0 voxels of a label ignored),
  `--dist-head SIGMA` (target channel 1 = a distance ridge of channel 0), and deep supervision `--deep-sup K` (1^3 heads on
  decoder levels 1..K, params last, header "ds_levels"; loss on max-pooled targets / any-pooled masks, weight
  `--ds-weight` w^j annealed to 0 over the run unless `--ds-anneal 0`; the head's input gradient is written in gout[j]'s
  storage and added before dec[j]'s backward; task surface only; test_unet UFSM_DS=1). Literature verdict: residual CNN
  U-Nets win compute-matched (nnU-Net Revisited 2024); transformers / Mamba / large kernels do not; the 2025 Vesuvius
  surface winners were nnU-Net ResEnc ensembles (blocks 1,3,4,6,6,6,6, widths to 320, scSE, skeleton recall, 3.5k+ epochs).
  Not built: scSE attention (per-voxel gating inside the fused GN + SiLU staging of every conv).
- Width: widths 32/64/128/160 (4.68M params, 4x the FLOPs) with Muon lr 0.01 scores 0.301 at 6000 steps against
  0.307 for 16/32/64/80: no accuracy per step from width at this length, so the 1.17M network stays.
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
