# ufsm — ultra fast scroll model

C23 + CUDA, no PyTorch, no cuDNN, no Python in the pipeline (Python only in `tests/`). One binary, `ufsm`, covers the whole
chain for Vesuvius Challenge scrolls: volcomp / surfcomp ingest of the released data, training and inference of our 3-D
U-Nets (winding fields, ink, papyrus), the winding fitter, wrap extraction, flattening, rendering and ink panels.
Production models: `wind96` (task unwrap, 9.6 um: phase, normals, density, recto) for the fit fields and `ift3-24000`
(ink96 fine-tuned on tight human labels) for ink. Design notes, experiments and numbers: [DESIGN.md](DESIGN.md),
[FINDINGS.md](FINDINGS.md). `ufsm` without arguments prints the command list; each command prints its usage on a missing argument.

## Build and test

```sh
make -j && make test     # gcc -std=c23 + nvcc; needs libcurl, libzstd, libblosc, zlib, libjpeg, xxhash (header), OpenSSL libcrypto, CUDA 13 (sm_120)
make ARCH=86             # Ampere (RTX 3090) into build86/: software MX-fp4 / fp8 (bit-exact conversions, emulated block-scaled MMAs)
                         # (on by default there; UFSM_MX_EMU=0: fp8 / fp4 as bf16, 16-bit storage); PTX=1: compute_86 PTX
                         # only (build86p/), so a newer GPU runs the sm_86 code (tests); BDIR=build puts it in build/
make -j test             # fast tier (the merge gate); make test-slow: the slow tier; make test-all: both
scripts/test/desk_suite.sh BRANCH   # the same on the desk (push over ssh, build, run, failing logs), one suite at a time
make doctor              # the toolchain the Makefile picked + `ufsm sysinfo` (real cgroup CPUs / RAM, GPUs / MIG, CUDA, disk)
```

The Makefile finds its toolchain itself: the first of gcc / gcc-15 / gcc-14 that accepts `-std=c23` (HOSTCC=, CC= override),
CUDA in /usr/local/cuda, else the newest /usr/local/cuda-*, else nvcc's prefix on PATH (CUDA= overrides), nvcc `-ccbin g++-N`
for gcc-N, and `-lstdc++` in CUDALIBS (gcc-14 + CUDA 13.2 failed to link without it). Default thread counts follow the cgroup
(`cpu.max` quota, affinity; `UFSM_NCPU` overrides), not nproc, and memory budgets (count-field `--mem-gb`, `UFSM_RENDER_MEM`)
are lowered to what the cgroup still allows (src/sysinfo.h). ccache is shared across worktrees (`CCACHE_BASEDIR` = the tree,
`CCACHE_NOHASHDIR`), so a fresh worktree or desk suite dir rebuilds from the cache.

The tests are listed in `tests/suite.txt` (`NAME | TAGS | EST_SECONDS | COMMAND`) and run by `tests/run_suite.sh`. `make test`
builds `build/ufsm`, the fixtures the Python CLI tests use and every suite binary on the VERIFY=1 build (`build-verify/`), then runs
the fast tier in parallel: the runner sets the job count from the CPUs this process may use (cgroup-aware, `-j`, `SUITE_J`), caps the
threads of each test (`UFSM_NCPU` / `UFSM_FIT_THREADS` / `UFSM_TIFF_THREADS` / `OMP_NUM_THREADS`, `-T`), gives each `gpu` test
one of 2 slots per GPU (`-G`; machine-wide locks in `/tmp/ufsm-suite-locks`, shared by concurrent suites), the GPU with the most free memory first, runs the
longest first, retries `flaky` tests once and prints each failing test's log tail plus a summary (`results.tsv` holds wall / CPU /
max RSS per test). Runner options go through `SUITE_FLAGS`, e.g. `make test SUITE_FLAGS="-j 4 test_fit"` (names matching a regex),
`-l` lists the tests, `-P` profiles (marks the tests that open the GPU, samples their GPU memory). One test by hand:
`make VERIFY=1 build-verify/test_strip && build-verify/test_strip`. A new test: its binary as a prerequisite of `suite-bins` in the
Makefile and a line in `tests/suite.txt` (tag `gpu` if it touches the GPU, `slow` if it takes more than ~30 s). The pipeline
stages below have their own tests: `test_countfield`, `test_fit_join`, `test_wraps_core`, `test_flatten`, `test_flatten_gpu`,
`test_faces`, `test_faces_gpu`, `test_strip`, `test_render_stream` (render / ink-compare / jpeg-fit, byte-identical streamed vs in-memory),
`test_sheetcut`, `test_ink_blend.py`. On the laptop run builds and tests through `~/ufsm-data/capped.sh` (with nice).

`scripts/test/desk_suite.sh [-t fast|slow|all] REF [REGEX...]` pushes the commit into `/vesuvius/ufsm/suite/repo.git` on the desk,
exports it to `/vesuvius/ufsm/suite/<sha>`, builds it (ccache shared across the sha dirs) and runs the suite at nice 10 under a lock
(one desk suite at a time). It waits for the desk to have MemAvailable >= 40 GB and a 1-min load <= 32 (`-m` / `-l`, at most `-w` 60 min)
before building and before testing, and uses the GPUs with >= 3 GB free. Logs stay in `/vesuvius/ufsm/suite/<sha>/out`.

## Unroll a scroll

The Paris 4 recipe (9.6 um, CT level 2 of the 2.4 um scan), as run by the drivers in `scripts/`. Coarse to fine: the integer
winding is decided once for the whole scroll by a global count-field reference, the slab fits only add sub-turn detail.

```sh
U=build/ufsm; R=/vesuvius/ufsm/volcomp; K=PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr
CK=wind96-wk1-best.ckpt; INKCK=ift3-24000.ckpt

# 1. fit fields (wind96): 10 planes on a 2-voxel cell grid, written straight into a fit-field set; repack to 512^3 shards
$U predict $CK $R $K fields.tmp --um 2.4 --level 2 --box Z,Y,X,NZ,NY,NX --window auto --halo 64 --shard 512 \
   --grid-origin 0,0,0 --ema 1 --q 8 --axis axis.json --levels 1 --fit-zarr fields --fit-h 2 --fit-q 8 --fit-q-phase 4 --no-out 1 --occupancy 5
for a in fields/*/; do $U repack $a ${a%/}.512 512 --threads 12 && rm -r $a && mv ${a%/}.512 $a; done
#    (--window auto: the cheapest window per interior voxel the GPU holds: 16 GB 384 x 640^2 (the old production window), 24-46 GB 640^3,
#     ~96 GB 1152^3; --feather ink: 576 up to ~30 GB, 1088 from ~46 GB. Direct slab outputs when there is room (~73 B per window voxel, no
#     plane copies). Bigger windows change the outputs slightly (per-window GroupNorm, context): allowed without a gate (user 10-31);
#     scripts/pod/wingate.sh is the gate. UFSM_PRED_WINDOW=prod: the old windows; UFSM_SLAB_DIRECT=0: the copy path.)

# 2. ink (ift3-24000): frozen GroupNorm from a calibration band, two passes offset by half a shard, blended
$U predict $INKCK $R $K gncal.zarr --um 2.4 --level 2 --box CALBOX --shard 512 --window 640 --halo 64 --ema 1 --full-grid 1 --occupancy 5 --levels 1 --no-out 1 --gn-save gn.bin
$U predict $INKCK $R $K ink.zarr  --um 2.4 --level 2 --box BOX  --shard 512 --window 640 --halo 64 --ema 1 --q 32 --full-grid 1 --resume 1 --occupancy 5 --levels 4 --gn-frozen gn.bin
$U predict $INKCK $R $K inkb.zarr --um 2.4 --level 2 --box BOXB --shard 512 --window 640 --halo 64 --ema 1 --q 32 --resume 1 --occupancy 5 --levels 1 --gn-frozen gn.bin
$U ink-blend inkbl.zarr --um 9.6 --tile 512,512,512 --grids 0,0,0:256,256,256 --origins 0,0,0:BZ,BY,BX --levels 4 --shard 512 --q 32 ink.zarr inkb.zarr

# 3. coarse: one global count-field reference model (no output fields; ~3 h for Paris 4 on the desk CPU)
$U count-field fields cf.unused --R 0 --pool 8 --phase-w 4 --iters 1200 --model ref --model-only 1 --model-stride 4 --threads 20 --mem-gb 24

# 4. fine: slab fits (64 field-cell interiors + 16 overlap, full cross-section) with every integer snapped to the ref
scripts/wbench/mkcrop.sh fields s000/f A 0 0 NZ NY NX                     # a fit-field crop of cells [A, A+NZ) (symlinks, no copy)
$U fit s000/f --gpu 0 --out s000/fit --dir $(jq .dir ref/model.json) --ref-model ref --ref-snap 1 --uw-repair 0

# 5. join: per-slab integer gauges from the ref (interiors in box-local native z = 2 x field cells)
$U fit-join joined fields --blend 16 --ref ref s000/fit 0 128 s001/fit 128 256 ...      # [--global 1]: joint robust solve
$U wrap-check joined fields --ct $R $K/2 --ct-um 9.6 [--codes Q 9.6]                   # QA: CT papyrus peaks per turn, dup / merged wraps

# 6. wraps: one .sfc per integer level set of W
$U wraps joined wraps --step 4 --out-um 9.6 --sfc 0.25 --fields fields --threads 16
$U wraps joined core  --step 4 --out-um 9.6 --sfc 0.25 --fields fields --first 70 --last 105 --zr 384,18176 --col-r p90   # core wraps

# 7. per wrap: flatten, render upright, panels (scripts: /vesuvius/ufsm/p4full/p4full3.sh on the desk)
$U tifxyz-scale wraps orig 1 --wraps w120
$U flatten orig flat --wraps w120 --decim D --threads 4 --crop none --gpu 1          # D: finest SLIM mesh <= 1.5M vertices
$U tifxyz-scale flat up 1 --wraps w120 --flip-rows 1
UFSM_RENDER_MEM=16 UFSM_RENDER_CHUNKED=1 $U render up r --volume inkbl.zarr 9.6 --ct $R $K/2 --depth 1 --scale 1 --threads 8
$U ink-compare img/w120_full_side.jpg --ct r/w120_ct.tif --ink ift3=r/w120.tif --px-um 9.6 --stretch --stretch-ct --max-px 65000 \
   --then prev/w120_side.jpg --max-px 8000

# 8. recto / verso faces of the sheet under each flat wrap (scripts/faces/faces_p4c.sh)
$U faces flat faces --wraps w120 --ct $R $K/2 --ct-um 9.6 --umb umbilicus.json --umb-um 2.4
$U tifxyz-scale faces fup 1 --wraps w120_recto,w120_verso --flip-rows 1
$U render fup rf --volume inkbl.zarr 9.6 --ct $R $K/2 --depth 0 --scale 2 --joint 1
$U ink-compare w120_rv_rgb.jpg --ct rf/w120_recto_ct.tif --ink recto=rf/w120_recto.tif --ink verso=rf/w120_verso.tif --px-um 4.8 \
   --stretch --stretch-ct --rgb-agree recto,verso --then w120_rv_side.jpg --then w120_rv_viridis.jpg --cmap viridis --no-ct

# 9. the whole scroll as one strip (scripts/strip/strip_run.sh OUT WRAPS)
$U strip wraps s/strip --threads 12 [--first K --last K]
$U strip-flatten s/strip --decim 4 --threads 12 --flip-rows 1 --cg 60 --tol 3e-4
for st in s/strip/flat/stripes/ws*; do ... render <dir holding only $st> s/render --volume inkbl.zarr 9.6 --ct $R $K/2 --depth 1 --scale 1; done
$U strip-mosaic s/strip/flat s/render s/img --ct-level 80 --threads 12
```

Drivers (resumable per stage; `STOP` files; MemAvailable gates):
- `scripts/unroll/unroll.sh SCROLL`: steps 1-9 on one host as one resumable run. Stages: fields (one predict per GPU on z parts
  of the box, joined by `fields_cat.sh`: shard files moved, byte-identical to one predict), ink on GPU `INKGPU` (1) beside the ref,
  ref, slabs (one lane per GPU; the ink GPU joins once the ink is done), join, wraps, strip, strip-flatten, stripe renders + mosaic,
  optional per-wrap renders (`PERWRAP=1`) and faces (`FACES=1`), and the upload (`UPLOAD=1` only, never for prize-eligible
  scrolls). Settings per scroll (box, wraps `--zr`, strip flags) come from `scripts/unroll/scrolls.tsv`, else `scripts/corpus/list.tsv`.
  `state/<stage>.done` holds the wall time, CPU seconds and max RSS. `DRY=1` prints the plan; `touch $O/STOP` stops between stages.
  The render, faces and ink-compare stages use the shared decoded-CT cache (`UFSM_ZCACHE_GB`, admit 0.6). Its size is `ZCACHE_GB`:
  by default 1/4 of the available RAM, at most 30 GB (the desk 30; a 62 GB pod ~15), and off on the laptop. The run removes
  `/dev/shm/ufsm-zcache-<uid>` when it ends, also on STOP. strip-mosaic gets `UFSM_JPEG_THREADS` = cores / 2.
  Env: `U CK INKCK FIELDGPUS FITGPUS INKGPU BOX TH PTH ITH MINMEM CTROOT CTKEY TAG INKCMD REFCMD ZCACHE_GB ZCACHE_ADMIT`.
- `scripts/corpus/`: the three-host pipelined corpus unroll (`common.sh` describes the split): `desk.sh` prep (mask box from
  scanprep, `ufsm axis` when there is no umbilicus), wind96 fields, repack, push; `laptop.sh` / `inklib.sh` ift3 ink over HTTP
  from the mirror (steps 2); `fd.sh` ref, snapped slab fits, join, wraps (steps 3-6); `desk.sh` then surf-stats, flatten and
  renders. `list.tsv`: one ~9 um scan per scroll (root, key, um, level, axis or `auto`).
- `scripts/cfit/`: the Paris 4 coarse-to-fine scripts with GT scoring (`refs.sh`, `slabs.sh`, `join.sh`; `cropref.sh` the crop A/B).
- `scripts/faces/faces_p4c.sh`, `scripts/strip/strip_run.sh`.

## Pod runs

The user rents the pods. These scripts make a fresh one ready and run a whole-scroll unroll within its limits; the scripts
need nothing from the desk. Typical limits are 13-16 real CPUs, 62 GB RAM, a 48 GB MIG slice and 500 GB disk. nproc and free
report the host (256 CPUs, 1 TB), so ufsm reads the cgroup instead: `ufsm sysinfo`.

```sh
# on the pod, as root (re-run any time: every step skips what is done)
curl -sfO https://raw.githubusercontent.com/SuperOptimizer/ufsm/master/scripts/pod/setup.sh
bash setup.sh PHerc1299        # limits, apt deps, clone + build master, smoke test, checkpoints, the scroll's CT from the mirror
setsid nohup bash /root/pod/ufsm/scripts/pod/unroll.sh PHerc1299 > /root/pod/unroll-PHerc1299.out 2>&1 < /dev/null &
tail -f /root/pod/unroll/PHerc1299/progress.log        # per-stage wall / CPU / max RSS in state/*.done; stop: touch .../STOP
```

`setup.sh` fetches everything from the public mirror, never via the desk. The mirror gives about 40 MB/s on a pod; desk -> pod
rsync manages only about 1.6 MB/s.
- Checkpoints: `ufsm/checkpoints/{wind96-wk1-best,ift3-24000}`.
- The scan's CT levels LV..LV+3: `ufsm mirror`, resumable and verified.
- Any published `ufsm-c2f-20261028` box / axis / ink of the scroll.
- When no box is published, `ufsm scanprep --box-out` makes one: the corpus prep box (PHerc1299 reproduces the published box).
- The scan comes from `scripts/unroll/scrolls.tsv` or `scripts/corpus/list.tsv`.

Env: POD (/root/pod), BRANCH, NOAPT / NOBUILD / NOSMOKE, LEVELS, MTH (mirror threads, 12: the mirror answers 429 past 64
requests in flight per IP). ARCH=86 is picked by itself on sm_80/86/89 GPUs (software MX).

`unroll.sh` runs `scripts/unroll/unroll.sh` (the stages above, resumable) on the pod's local CT, on one GPU or MIG slice, with:
- `TH` = the cgroup CPUs;
- the MemAvailable gate from `ufsm sysinfo --avail-gb`;
- count-field `--mem-gb` = min(24, 40% of the limit);
- `UFSM_RENDER_MEM` = min(16, 25% of the limit);
- no per-wrap renders (`PERWRAP=1` adds them).

Every variable can be overridden from the environment. Disk for a Paris 4-size scroll: CT level 2 42 GB, fields ~5 GB (50 GB with the old lossless phase; deleted
once the wraps exist), joined 10 GB, ink 1.5 GB.

Fields copied home: predict writes the phase planes lossy (`--fit-q-phase 4`; they were lossless, ~90% of the store). PHerc0490B
21.6 -> 2.85 GB (7.6x), PHerc0483B 30.4 -> 3.47 GB (8.8x). A store written before that shrinks in place with
`scripts/pod/packfields.sh FIELDS` (`ufsm fields-pack`, byte-identical to predicting at q 4): 21.6 GB in 160 s at 8 threads on the
loaded desk, IO-bound on a slow disk.

Peak RSS per stage. Measured with `scripts/pod/unroll.sh` on a Paris 4 crop: 1024 level-2 z of the papyrus box y 1024+7150,
x 512+7424, so slab fits and predicts see the whole-scroll cross-section. The crop ran on the desk in
`systemd-run --scope -p CPUQuota=1400% -p MemoryMax=60G`. The whole-scroll column holds Paris 4 runs where they exist
(logs named), else estimates scaled by the strip's valid vertices (x18.5).

| stage | command | crop | whole Paris 4 | grows with | cap / lever |
|---|---|---|---|---|---|
| fields | `predict --fit-zarr` (wind96) | 9.5 GB | ~9.5 GB | cross-section (one z slab of windows) | `--window` |
| ink | `predict` x3 (ift3, 2 passes) | 1.8 GB | ~1.8 GB | window | |
| ink blend | `ink-blend` | 1.2 GB | ~1.2 GB | tile | `--tile` |
| ref | `count-field --model-only` | 23.2 GB | 24.5 GB (cfit/whole/ref-cf, 3 h) | `--mem-gb` (24) | `--mem-gb`, now <= 60% of what the cgroup allows |
| slab fits | `fit --ref-model` (96 cells, full section) | 9.6 GB host; 12.7 GB GPU | same per slab | cross-section | GPU: full 8174^2 box ~16 GB OOMs a 16 GB card: crop to the papyrus box |
| join | `fit-join --ref` | 2.7 GB | 11.6 GB (JDETAIL=0) | scroll | `--score-detail` costs 89 GB: off |
| wraps | `wraps` | 2.1 GB | 11.6 GB (p4core) | scroll | `--first/--last`, `--zr` |
| strip | `strip` | 1.9 GB | 33.6 GB (e2e validation) | valid vertices (~28 B each) | `--every 2` (4x less), `--first/--last` |
| strip flatten | `strip-flatten` | 0.7 GB | ~13 GB (est.) | vertices | `--decim` |
| stripe render | `render` (union box) | 15.2 GB | 29.3 GB (Paris 3, 60000 x 22361 px stripes) | `UFSM_RENDER_MEM` + stripe px (~19 B/px) | `UFSM_RENDER_MEM` (now <= 50% available), strip-flatten `--stripe-px` |
| mosaic | `strip-mosaic` | 9.5 GB (2 P4 stripes, 2 jobs) | ~5 GB per stripe in flight (3.3 B/stripe px) + overview sums | stripe jobs x stripe px | `--stripe-jobs` (2) capped by memory (70% available); `--threads` work inside a stripe |
| per wrap | `render` / `ink-compare` / `flatten` | 0.8 / 0.25 / 0.43 GB | render <= `UFSM_RENDER_MEM` + output; ink-compare streamed 0.5 GB | wrap area | `UFSM_RENDER_MEM`, `--decim` |

On a pod only ink (GPU) and ref (CPU) overlap, about 26 GB together. Every other stage runs alone. The largest are strip
(34 GB) and a stripe render (~30 GB), both under 50 GB at Paris 4 size. For a scroll 1.3x taller (PHerc1447) strip would
reach ~44 GB: use `strip --every 2` there.

The crop is 1/18.5 of Paris 4. With 14 CPUs and one 16 GB GPU, its stage wall times were:
- fields 4.5 min;
- ink beside ref 7.6 min;
- 8 slabs 4.5 min;
- join + wraps 2.4 min;
- strip + flatten + stripe images 5 min.

Scaled up, whole Paris 4 takes about 7 h on such a pod. The global ref alone is ~3 h and does not split.

## Command reference

Only the flags that matter for each stage; the usage strings list the rest.

### Data

```sh
ufsm info|read|slice <root> <key> ...                 # inspect / read volcomp zarr v3 pyramids (local or HTTPS, cached)
ufsm ls <root> <key>                                   # list HF / S3 / local trees
ufsm mirror <root> <array-key> <out-dir>               # local copy of one array's shard files
ufsm scanprep <root> <group-key> --level L --out PREFIX   # per-scan object mask + CT statistics (Otsu level)
ufsm axis <root> <ct-group> <out.json>                 # per-z centroid axis when no umbilicus exists
ufsm ingest-zip  labels.zip <zarr-name> <out> --um U   # HF label zarr (from the archive) -> volcomp label pyramid
ufsm ingest-labels <root> <zarr-key> <out> --um U      # same, chunk by chunk over HTTP (slow: HF rate limits)
ufsm ingest-kaggle <hf-root> <out>                     # Kaggle cubes -> images.zarr + labels.zarr
ufsm ingest-mesh <s3-root> <tifxyz-key> <out.sfc>      # segment mesh -> surfcomp
ufsm raster <out> --shape Z,Y,X --um U meshes...       # meshes -> label pyramid (surface band / background / ignore)
ufsm repack <in-array> <out-array> 512 [--threads N]   # lossless re-sharding of a z3w uint8 array (copies compressed chunks)
ufsm fields-pack IN OUT [--q-phase 4] [--q -1] [--threads N]   # a fit-field store smaller: phase planes lossy (old stores: lossless pc / ps = 90%), other planes copied; readers unchanged
ufsm z3-recode <group> <out-group> --um U --lut FILE [--hist 1]   # a prob254 pyramid through a LUT (255 kept), value histogram
```

Ground-truth export scripts: `tools/pget.sh` (parallel ranged download), `tools/ingest_all.sh` (all label zarrs),
`tools/ingest_segments.sh`, `tools/raster_all.sh`, `tools/surfcomp_all.sh`.

### Training and evaluation

```sh
ufsm sample <sources.json> --out DIR                   # dump patch montages to eyeball the sampler
ufsm prefetch <sources.json> [--levels 1]              # warm the CT chunk cache for the labelled cells
ufsm train <sources.json> --out runs/r1 --gpus 0 --P 512 --B 1 --steps 20000
ufsm eval <pred-root> <pred-group> <label-root> <label-group> --um U [--box ...]
ufsm train <sources.json> --task papyrus --pap-cond 2 --pap-tau 2,8 --dice 0 ...   # papyrus vs void, labels from the CT itself
ufsm predict <papyrus.ckpt> <root> <ct-group> <out> --um U --papyrus-cut 80        # papyrus where the smoothed CT exceeds 80
ufsm ckpt-avg ...                                      # checkpoint averaging
```

Papyrus / void network (DESIGN.md "Papyrus / void network"): per patch a random cutoff t, labels = the clean CT smoothed
by a Gaussian above t, scanner augmentation on the input only, the cutoff fed as the input's zero, so one checkpoint serves
every cutoff.

#### Crush augmentation

Sheet-aware crush (`src/crush.h`, DESIGN.md "Sheet-aware crush augmentation"): a window is pulled through an invertible
map (positive Jacobian) that squashes it in the sheet plane, bends / kinks it along the sheet normal and collapses the
voids between wraps while papyrus keeps its thickness; the stored phase is pulled through the same map, so phase, winding
mod 4, normals and density stay exact. Applies to stored-phase windows (task unwrap / unwrap2, side 2, cube windows) and
needs `--geometry-aug 1`. **Every flag defaults off**: without `--aug-crush` the batches are bit-identical (tested), so
existing recipes are unchanged.

| flag | default | meaning |
|---|---|---|
| `--aug-crush P` | 0 | probability a window is crushed |
| `--aug-crush-void LO,HI` | 0.15,0.5 | void ratio a gap keeps (per window, varying over it) |
| `--aug-crush-squash P` / `--aug-crush-k LO,HI` | 0.5 / 1.5,3 | in-plane squash probability / factor |
| `--aug-crush-bend P` / `--aug-crush-bend-amp A` | 0.5 / 0.06 | low-frequency bends along the normal (amplitude x P) |
| `--aug-crush-kink P` | 0.25 | a fold line |
| `--aug-crush-rot DEG` | 10 | the crush map's own rotation |
| `--aug-crush-fine N` | 0 | CT box mean from N levels finer (too slow for the loader) |
| `--aug-crush-warp 0/1` | 0 | 1: crushed windows also take the ordinary geometry warp (rotation / elastic / affine / zoom, `--rotate-p`, `--elastic-p`, `--aug-affine-p`, `--zoom-p`) after the crush map; labels pulled through both |

Realistic setting (the desk crush arm `ctrain.sh`; at the defaults above gaps collapse into touching sheets with the phase
kept smooth, which is the crushed-scroll look):

```sh
--geometry-aug 1 --aug-crush 0.3 --aug-crush-warp 1 --aug-crush-bend 0.7 --aug-crush-kink 0.35 \
  --rotate-p 0.3 --elastic-p 0.5 --aug-affine-p 0.4 --aug-aniso 0.1 --aug-shear 0.06 --zoom-p 0.3 --zoom-min 0.85
```

CPU cost: a crushed P512 window takes 18-31 CPU-s on the sampler (P256 6-17 s), against well under 1 s for a plain window.
At P512 `--aug-crush 1` needs ~16-24 sampler workers to feed one GPU; 0.3 is the comfortable rate. `--aug-crush-warp 1`
adds only the ordinary warp's cost.

Debugging: `build/crush_render <sources.json> <out-prefix> [P] [windows] [seed] [level] [fine] [box] [--aug-crush-* flags]`
renders CT | phase | crushed CT | crushed phase panels. `UFSM_DUMP_WIN=PREFIX` (with `UFSM_SYNC_UPLOAD=1`, `--gpu-input 0`,
cube windows) makes `train` write the first `UFSM_DUMP_N` (8) host windows as PREFIX-NN.jpg (columns
CT | phase hue over CT | winding mod 4 hue over CT; rows z-mid / y-mid / x-mid).

### Prediction

`ufsm predict <ckpt> <root> <ct-group> <out> --um U [--level L] [--box z,y,x,nz,ny,nx]` writes a volcomp pyramid of 512^3
(`--shard`) shards. Flags used in production:

| flag | meaning |
|---|---|
| `--window W --halo H` / `--window-z --halo-z` | tile size and discarded border (thin-z windows for the fields) |
| `--grid-origin z,y,x` | tile interiors anchored to CT coordinates: a voxel's windows do not depend on the box / shard layout |
| `--full-grid 1` | every tile of the grid (ink); `--occupancy L`: skip shards, and windows whose interior, that are zero on the coarser CT level L (one coarse voxel of margin; outputs and fit fields identical except at isolated low CT specks whose coarse voxels round to 0, e.g. 6 of 4.2M P3 coarse voxels; also with `--fit-zarr`; reads of air windows saved: P4 61% / P3 31% of the shards at L 5; `UFSM_OCC_TILES=0`: shards only); `--resume 1`: skip written shards |
| `--fit-zarr DIR --fit-h 2 --fit-q 8 --fit-q-phase 4 --no-out 1 --axis JSON` | the unwrap heads straight into a fit-field set (no prediction pyramid) |
| `--gn-save FILE` / `--gn-frozen FILE` | GroupNorm statistics from a calibration box, then frozen for the whole volume (no per-tile brightness seams) |
| `--ema 1`, `--q Q`, `--levels N` | EMA weights, volcomp quality (0 lossless), pyramid levels written |
| `--tta`, `--input-gain G --input-bias B` | test-time augmentation; the standardized CT input transform `G * zscore(CT) + B` (defaults: identity) |

Runtime notes:
- Fresh training computes GroupNorm statistics from rounded stored activations; resume and prediction keep each checkpoint's
  contract (`--gn-stats stored|legacy`). `--input-prec 4|8` selects stem-input quantization and is saved in the checkpoint.
- Without `--grid-origin` the historical box / shard-relative tiling is used. Off-grid requests can repeat windows across
  output shards; lossy encoding depends on codec-block placement, so use `--q 0` for arbitrary-crop byte comparisons.
- Evaluation with three or more cutoffs filters once and collects byte histograms (4.9-5.5x faster, same reports).
- Remote zarr reads coalesce nearby chunks (<= 2 MiB, <= 64 KiB gaps) and fetch shard indexes on up to 16 threads; HTTP
  handles are pooled (32). `UFSM_Z3_COALESCE=0`, `UFSM_Z3_INDEX_PARALLEL=0`, `UFSM_HTTP_REUSE=0` disable them for comparisons.
  HTTP permission failures are errors; genuinely missing shards read as the fill value.
- The input transform is rejected by split inference, surface winding, conditioned sweeps and cutoff-conditioned papyrus models.
- `--feather B` (B <= halo; window - 2 halo = k shards on the shard grid): seam-free single pass. The lower B voxels of every
  window are a sin^2 blend of its own and the earlier neighbours' halo predictions (normalized over the windows present), only
  the earlier windows' upper slabs are held (a window layer + a row on the host), the blend runs on its own thread. `--resume`
  recomputes done windows a remaining neighbour needs; `--gpus a,b` deals contiguous window layers. Byte-identical away from the
  blend bands and across resume / worker splits (test_predict_feather). Paris 4 ift3 ink: border step (`ink-seam --block 16`)
  20.2 (halo 0) / 6.3 (halo 64) -> 0.0 at window 576 halo 32 `--feather 32`, as the old 2-pass + ink-blend, in one pass.
- `z3_read` decodes only the volcomp substreams (16 z x 32 y slabs of a 128^3 chunk) under the request box: same bytes, -33%
  read time on 576^3 windows with 32-voxel halos (`UFSM_Z3_AUTOSUB=0` off; test_z3_autosub).

### Ink

```sh
ufsm ink-blend <out> --um U --tile SZ,SY,SX --grids z,y,x:z,y,x:... <in1> <in2> ... [--origins z,y,x:...] [--levels 4] [--shard 512] [--q 32]
ufsm ink-seam <zarr> --um U [--period 512] [--offset 0] [--n 24] [--block 16]   # window-seam jump (border vs interior |dv|; --block 16: vs the codec-block faces)
ufsm ink-ext --model v8in|hecate24|hecate96|reader3|ink9um --weights W --surface TIFXYZ --ct ROOT KEY --out DIR   # external 2.5D ink nets
ufsm ink3d-ext --weights W --ct ROOT KEY --box ... --out GROUP       # the published 3D ink net (gambiarra) on a 2.4 um CT box
ufsm ink-splat <out-group> --surface TIFXYZ --labels L --ct ROOT KEY --ct-um U --iso I [--band -4,4]   # 2D ink labels -> 3D label zarr
ufsm ink-score --surface TIFXYZ --labels L --map NAME=MAP.tif ..     # F1 / AUC of 2D ink maps vs a segment's labels
```

`ink-blend` averages overlapping uint8 predictions voxel by voxel with sin^2 tile weights. `--grids` gives each input's tile
grid origin (as `predict --grid-origin`); `--origins` places inputs predicted on boxes (no `--full-grid`) at their origin,
weight 0 outside. Offset passes as box-local predicts avoid the misaligned-window cost of `--full-grid` on a shifted grid.
Paris 4 deliverable `paris4/ink/ift3-24000-ov8.zarr`: 8 passes (window 1024, halo 0, frozen GN, origins 0 / 512 per axis at
level 2), `ink-blend --um 9.6 --tile 1024,1024,1024`; window-border jump 30 -> 6 (a single pass with halo 64 + frozen GN: 9.4).

### Fitting

```sh
ufsm count-field <in-fields> <out-fields> --R 0 --pool 8 --phase-w 4 --iters 1200 --model DIR --model-only 1 [--model-stride 4] [--mem-gb 24]
ufsm fit <fields-dir> --gpu G --out MODEL [--dir 1|-1] [--ref-model REF --ref-snap 1 --uw-repair 0] [--codes ROOT KEY --score-pairs 1]
ufsm fit <fields-dir> --model MODEL --codes ROOT KEY [--score-pairs 1]   # score a saved model or set
ufsm fit-join <out-set> <fields-dir> --blend 16 --ref REF [--global 1 [--cap 3] [--min-cells 20000]] <slab-model> <z0> <z1> ...
ufsm wrap-check <model-or-set> <fields-dir> --ct ROOT KEY --ct-um U [--codes ROOT KEY] [--zr Z0,Z1] [--tsv OUT]
ufsm surf-stats <wraps-dir> --fields DIR [--model FIT] [--ct ROOT KEY --ct-um U [--ct-level 80]] [--ct-only 1] [--per-wrap]
ufsm field-blend <out> <fields1> [<fields2> ...] [--tile SZ,SY,SX --grids z,y,x:...]
ufsm spiral-fit <in-fields> --model DIR [--init-model REF]   # = count-field --method spiral: villa diffeomorphic spiral (opt-in, null on P4)
ufsm cf-compare <model-a> <model-b> [fields] [--step 4]   # two count-field refs: lattice |delta|, snap-integer agreement
```

- `count-field --model-only 1 --model DIR`: winding integrated from the network's density x normals (weighted LS, coarse to
  fine), the network phase only sub-turn; with `--R 0` one global LS gauge over the whole box. Writes only the model (no
  output fields). `--model-stride K` takes the gauge from every K-th field chunk (Paris 4: 15% faster, same scores at 4).
  `--mem-gb` is the field chunk budget: small budgets re-read the 128-row zarr shards many times. The model's `dir` is the
  scroll's handedness (PHerc1447 is -1); pass it to the slab fits.
- `count-field --cf-gpu 1` (default; the CPU code without a device, and per chunk / CG level when the device is short of
  memory, e.g. shared): pass 1's pooling, the CG solves and the `--model-only --R 0` model pass on the GPU (`src/nn_cf.cu`);
  the global gauge pass and the model pass become one read of the strided chunks; the next chunk decodes on a thread
  meanwhile (`--prefetch`, on when two chunks fit in half of MemAvailable; `--read-threads N` the zarr decode threads).
  The pooled sums are bit-identical to the CPU (no FMA contraction, IEEE division, glibc atan2 / hypot of the 65536 phase
  bytes as a table); the CG sums in another order, a difference the size of another `--threads`; runs are deterministic.
  Paris 4 whole ref (desk, 8 threads, loaded): 3:02 h -> 32 min; snap integers round(W - phi) agree 0.999986 with the
  10-26 CPU ref (CPU 8 vs 6 threads on a cut: the same level). `ufsm cf-compare A B [FIELDS]` measures this. The result
  depends on `--mem-gb` (the chunk rows: chunk-edge rows skip the phase term, `--model-stride` picks chunks): `--mem-gb 13`
  reproduces the 88-row chunks of the 10-26 Paris 4 ref.
- `fit --ref-model REF --ref-snap 1`: every voxel's integer = round(U_ref - bias - u); no flood, use with `--uw-repair 0`.
  A full-cross-section slab needs ~11 GB of GPU memory (one fit per 16 GB GPU). An air slab at the box ends fails with "no data
  for the prior"; leave it out of the join. `UFSM_FIT_GPU_LOCK=FILE` (default in `scripts/cfit/slabs.sh`: /tmp/ufsm-fit-gpuN.lock)
  holds a lock from the device allocation to exit, so two lanes per GPU overlap one fit's field decoding with the other's device
  stages. Zarr fields decode only the 16-row volcomp substreams the slab touches (`UFSM_FIT_FULLDECODE=1`: whole chunks).
- `fit-join --ref`: per-slab integer gauges by DP over the ref and the overlaps; prints the ref share per slab and the integer
  agreement on the overlaps. `--global 1`: joint robust solve over the whole overlap graph (ties the DP on Paris 4 and Paris 3).
  Slab interiors are box-local native z (`2 x` field cells at `--fit-h 2`). All sample requests run in one z sweep through a
  two-chunk-row cache (each chunk row decoded once; ~9 GB RSS on a full cross-section).
- `wrap-check`: along radial rays, CT papyrus peaks between consecutive wraps (defaults `--sigma 1.2 --prom 1`, calibrated on
  Paris 4 GT): per-wrap dup / merged / real verdicts, radius and z tables. ~1.0 peak per turn is right; `--w-scale 2` is the
  doubled-count control.
- Scoring a whole scroll with `--score-detail 1` needs ~89 GB; leave it off.

Paris 4 numbers (q-9p6 codes): ref alone ids .1605; snapped slabs + `fit-join --ref` .1626, pairs .5/4 mm .681/.303, overlap
agreement .918 (the old chained join .0851).

#### Recto-guided local fits (experimental)

The unwrap checkpoint predicts recto (channel 0), winding phase (2-3), normals (4-9) and density (10). `predict --fit-fields`
/ `--fit-zarr` keep the recto maximum of each field cell as `r`. A local fit can use recto peaks as integer-level-set anchors:

```sh
ufsm fit fields --out local-fit --core 0 --axis-normal 2000 --w-recto 8 --w-phase 0.25 --phase-robust 0.10 --unwrap-voxels -1
ufsm wraps local-fit local-tifxyz --fields fields --step 1
```

`--axis-normal` is a local initialization (signed distance in the fields' native voxels; compare both signs).
`--phase-robust S` is a Cauchy penalty on the circular phase residual (scale S turns); `--unwrap-voxels -1` skips the phase
flood; `--w-phase 0` fits from recto, normals and density alone; `--w-density` / `--w-normal` set those terms. Recto anchors
do not fix winding integers across disconnected regions; whole-scroll accuracy is not established.

### Surfaces

```sh
ufsm wraps <model> <out> [--step 8] [--out-um U] [--sfc ERR] [--fields DIR] [--first K] [--last K] [--zr Z0,Z1] [--col-r centre|p90|R] [--min-cm2 0.5]
ufsm tifxyz-scale <src> <dst> <div> [--wraps a,b] [--offset z,y,x] [--flip-rows 1]   # coordinates / div; div 1 = .sfc -> tifxyz copy
ufsm tifxyz-sfc <src> <dst> [--error 0.25] [--every K] [--wraps a,b]                 # tifxyz -> .sfc
ufsm flatten <wraps> <out> [--wraps a,b] [--decim 1] [--crop rect|none] [--max-aspect A] [--rects N] [--min-cm2 5] [--fill-holes 5000] [--gpu 1] [--sfc ERR]
ufsm flatten-warp <in.jpg|tif> <flat-wrap-dir> <out.jpg|tif> --src-px P [--scale S]
ufsm faces <wraps> <out> --ct ROOT KEY --ct-um U [--wraps a,b] [--umb JSON --umb-um U] [--thick 15,200] [--gt ROOT KEY --gt-um U] [--gpu 1]
ufsm sheet-cut <wraps> <out> [--tol 0.3] [--min-cm2 0.25] [--max 8]                  # square flat sheets from the raw grids
ufsm wrapswitch <wraps> <codes-root> <codes-key> [--per-wrap]                          # sheet switches per cm vs GT codes
```

**wraps.** Wrap k is the level set W = k of the fit model, sampled on rays: column = angle at arc step `--step` (native
voxels), row = z. `--out-um 9.6` writes coordinates on that grid (a 9.6 um fit's wraps on the 2.4 um grid with 2.4), `--sfc
0.25` one `wNNN.sfc` per wrap instead of tifxyz dirs. The column spacing is the arc of the wrap's root on the ray through the
box centre at mid z; a core wrap leaves the axis near that z, so its root is tiny and its columns land far apart (Paris 4 w085:
29 columns). `--col-r p90` (default) uses the 90th percentile of the wrap's roots over the rows' z range instead (w085: 267
columns); `--col-r centre` is the old spacing. `--zr Z0,Z1` keeps rows at global native z in [Z0, Z1] and samples the wrap ids there
(drop a broken end cap). `--first` / `--last` are exact. Wraps with <= 16 vertices, or under `--min-cm2` (default 0.5; vertices x
(step x um)^2), are not written: on Paris 4 these are the 62 end-cap fragments w016-w077 of 0.07-0.09 cm2. With `--fields`, only
wrap ids within about one turn of a data cell are computed, largest first. Each empty id cost 30-60 CPU-s of failed root searches:
Paris 4 sampled 16..580, wrote 257, and the empty ids above the data took 10.7 of 20.7 min. `UFSM_WRAPS_NOPRUNE=1` computes every id.
The `.sfc` files (here and in `ufsm strip`) are encoded on all threads, byte-identical to the serial writer (P4 strip.sfc 876 -> 170 s
at 4 threads). The encoder itself runs ~1.8x faster with the same bytes: the vendored surfcomp's DCTs and quantizer on 8-wide vectors
(the same multiply-add order per lane; the build has no FP contraction), each candidate tokenized once, tANS sizes before bytes, the
index entries buffered (`test_tfx` checks golden hashes of the original encoder). The writer's workers run a ring ahead and append
in order themselves (no thread woken per patch). `--fields` reads the data weight chunk by chunk
on threads (PHerc0490B's 80 G cells at pool 8: 190 -> 6 s), the slab models load on threads, and a row's slab choice is made once.
PHerc0490B wraps + strip and a Paris 4 crop: every file byte-identical to master.

**flatten.** SLIM (symmetric Dirichlet, multilevel, flip-free line search; `src/flatten.h`) on the wrap's own vertex grid,
then an isometric uv resample, Laplace fill of enclosed tears up to `--fill-holes` vertices (orange in `<w>_mask.jpg`; they
are bridges, not papyrus) and the largest fully valid rectangle (`--crop none`: the whole flat grid; `--max-aspect` /
`--rects N`: aspect-capped, greedy non-overlapping rectangles indexed in `<w>/rects.tsv`). Writes tifxyz plus `gu.tif` /
`gv.tif` (source grid coordinates) and `flatten.txt` (distortion before / after). `--gpu 1` (default) runs the SLIM
iterations on the GPU (fp64, deterministic, ~1e-6 voxel from the CPU solver; CPU fallback without a device): Paris 4 w120
whole wrap 87 -> 17 s. Keep the finest mesh <= ~1.5M vertices with `--decim`. Paris 4 aniso (sigma1/sigma2) p50 1.6 -> 1.02.
`ufsm render` of the flat tifxyz is the flattened render; `flatten-warp` moves an existing render of the original wrap onto it.

**faces.** A wrap is a level set of a smooth field: it follows the sheet but can sit anywhere across it or in the gap next to
it. `faces` samples the CT along the outward normal (`--range` um), splits the profile into bands (persistence-simplified
extrema; half-level edges; a high valley between touching sheets is an edge), makes the pick consistent over the surface
(median windows, median filter, push-pull hole fill), and writes `<w>_recto` (inner face, toward the umbilicus) and
`<w>_verso` (outer face) on the input grid, with `conf.tif` / `thick.tif` in `<w>_recto` and `<w>_faces.txt`; one table
line per wrap on stdout. Recto is the inner face by the published-label side rule (papyrus outward of wrap k is recto side).
Run it on the flat grid so both faces share one uv and render pixel-aligned; render faces with `--depth 0` (the sheet is
~4 voxels at 9.6 um). The CT cache decodes only the volcomp substreams under the samples (`UFSM_FACES_FULLDECODE=1`: whole
chunks; same output). `--gpu 1` (default) samples the profiles, cuts the bands and runs the median windows on the GPU from a
device mirror of the decoded chunks (decode list = the exact substreams the samples read; `src/nn_faces.cu`, no fast math,
no FMA): byte-identical outputs, CPU fallback without a device or when the device pool fills. Paris 4 w178: sampling 35 -> 3 s,
medians 15 -> 2.5 s; the CPU volcomp decode is then ~75% of the wall. `UFSM_FACES_PROF=1` prints per-stage times. With
`UFSM_GPU_DECODE=1` (below) the CT substreams are decoded on the GPU straight into that device mirror (no host round trip;
if the GPU path fails mid-run the substreams only the device had are decoded again on the host; with it faces marks + decodes
128 rows per fill instead of 32 (`UFSM_FACES_FILL_ROWS`): bigger decode batches, w178 CT fill 79-82 -> 55-58 s on a loaded laptop).

**GPU volcomp decoding** (`UFSM_GPU_DECODE=1`, opt-in; `src/nn_gpudec.cu`): every lossy (mode 0) volcomp substream that
z3_read / z3_read_sub decodes goes to the GPU instead of volcomp_fast.h, byte-identical (one thread per substream for the tANS
/ bypass decode, one CUDA block per 16^3 block for the dequantisation and the reference's inverse DCT network; no fast math,
`-fmad=false`; corrupt or truncated streams are decoded again on the CPU, which gives the reference's status). A service
thread batches the requests of all decode threads (render and faces run `UFSM_GPU_DECODE_THREADS`, default 32, decoders
when it is on); a substream is a ~3-5 ms serial chain on the GPU, so throughput comes from many substreams in flight.
Requests under `UFSM_GPU_DECODE_MIN_BYTES` (2048) compressed bytes per substream (ink, air: flat blocks) stay on the CPU;
(not when decoding straight into device memory) `UFSM_GPU_DECODE_DEV` picks the device, `UFSM_GPU_DECODE_STATS=1` prints substreams / batches at exit; any device error
turns it off for the rest of the process. A GPU that another job saturates starves it (time slicing: next to a fit at 99% SM
it crawled), so a round trip over `UFSM_GPU_DECODE_MAX_MS` (60) sends the requests to the CPU for `UFSM_GPU_DECODE_COOL_S` (2 s),
then the GPU is tried again; requests on the CPU path run at most `UFSM_GPU_DECODE_CPU_THREADS` (default: the CPUs this process
may use) at once, so the extra decode threads do not oversubscribe the machine. `tests/test_gpudec.c` (`--real ARRAY cz cy cx nz ny nx [threads] [density]`:
identity + MB/s on real chunks).

### Whole-scroll strip

Why it works: W = U - dir theta / 2 pi with U smooth and single valued, so wrap k's theta = -pi edge is wrap k - dir's
theta = +pi edge; the wraps in order are one continuous sheet. The junction order is measured (both pairings tried).

```sh
ufsm strip <wraps> <out> [--first K] [--last K] [--every E] [--err 0.25] [--threads 8]
ufsm strip-flatten <strip-dir> [--out DIR] [--decim D] [--stride S] [--iters 30] [--cg 200] [--tol T] [--ostep S] [--stripe-px 60000] [--px-scale 1] [--flip-rows 1] [--max-edge 3] [--reuse 1] [--threads 16] [--gpu 1] [--jobs J] [--stripe-jobs 3] [--chunks J0,J1]
ufsm strip-mosaic <flat-dir> <render-dir> <out-dir> [--strip DIR] [--ct-level 80] [--overview-px 16000] [--rows 10] [--tile-cm 1] [--quality 88] [--stripes 1] [--threads 8] [--stripe-jobs 2] [--papyrus GAIN] [--papyrus-dir DIR [--papyrus-gain 1] [--papyrus-overview-px PX]] [--fill-maps DIR]
ufsm strip-crop <in.tif> <out.jpg> x,y,w,h [--stretch 1]
```

- `strip`: `<out>/strip.sfc` (rows z, columns the unrolled angle, region-readable 64 x 64 blocks), `strip.tsv` (segments:
  wrap, first column, columns, seam gap), `strip.json`. Paris 4 (257 wraps): 584284 x 4673, 0.9e9 vertices, 823 MB, 15 min.
- `strip-flatten`: an arc-length frame shared by every row, chunks of 2 S columns every S (`--stride`, default half the
  height) SLIM-flattened and rigidly aligned onto it, blended by a partition of unity; columns without surface take 0.02 step.
  Output `<strip>/flat`: `flat.json`, `stripes.tsv`, `stripes/wsNNN` (tifxyz + `gu.tif` / `gv.tif`, each at most
  `--stripe-px` pixels at render `--scale --px-scale`, under the JPEG limit), `chunks.tsv`, `segments.tsv`, `flatten.txt`.
  `--flip-rows` defaults to 0; Paris 4 needs 1 to read upright. Paris 4: 1757 cm x 20.6 cm, aniso p50/p95 1.053/1.166,
  flips 0.0045% of the area, 2.6 h at `--decim 4` (CPU).
- Speed: `--gpu 1` (default when a device answers) runs each chunk's SLIM iterations on the GPU (`flatten --gpu`, fp64,
  deterministic); `--jobs J` chunks run at once (default 3 per GPU, device = job % devices, `--threads / J` CPU threads each)
  and are blended strictly in chunk order, so the lattice does not depend on J. `--stripe-jobs` (3) writes stripes
  concurrently (stripes.tsv rows in order). `--chunks J0,J1`: a benchmark of those chunks only (stats over their span, no
  stripes). Desk (2x RTX 5060 Ti): Paris 4 chunk 8.5 s alone vs 88-97 s CPU, 2.5 s per chunk at `--jobs 6`; Paris 3
  (254 chunks, decim 2 on step 8) 605 s for the chunks, 705 s in all. The GPUs (fp64) are then the limit, not the cores.
- Render each stripe dir alone (`render` takes a directory: link one stripe into its own dir, as `strip_run.sh` does).
- `strip-mosaic`: global contrast stretch over all stripes, then `<out>/wsNNN_ct_ink.jpg` (CT over ink; `--stripes 0`: none),
  `overview.jpg` (the length folded into `--rows` rows, with an on-papyrus bar: green on, red off, 10 cm ticks),
  `strip_long.jpg`, `strip_papyrus.tsv` (per `--tile-cm`: valid cm², share with CT >= `--ct-level` within 3 x 3 px, wrap range)
  and `papyrus_wraps.tsv`. `--papyrus GAIN`: the papyrus look instead (`<s>_papyrus.jpg`, one-band overview / long strip);
  `--papyrus-dir DIR` writes that papyrus mosaic as well, from the same reads, into DIR (the bytes of a separate `--papyrus` run;
  `--papyrus-gain`, `--papyrus-overview-px`). `--stripe-jobs` (2, memory capped) stripes in flight, each worked on by `--threads`
  threads (strip-parallel TIFF decode, mask rows, pixel row bands with private integer sums merged under a lock: the same bytes
  for any thread / job count); stripe JPEGs are streamed row by row through lookup tables. Desk (loaded), real P4 stripe
  60000 x 21452 px at 4 threads: per stripe TIFF decode 14.8 -> 3.9 s, mask/fill/gu 13 -> 2.5, pixels 45 -> 3.3; a 2-stripe run
  grey 81 -> 14 s, papyrus 71 -> 11 s, both from one read 17 s; 6 stripes grey 189 -> 73 s. Off-papyrus surface is reported, never cut.
- `strip-crop`: a rectangle of a stripe-size render TIFF (60000 px, past most image tools) as a stretched JPEG.

### Rendering and panels

```sh
ufsm render <wraps> <out> --volume ROOT KEY [--ct ROOT KEY] [--scale 1] [--depth 8] [--threads 16] [--crop x,y,w,h] [--joint 1]
ufsm ink-compare <out.jpg> --ct CT.tif --ink LABEL=INK.tif [--ink ..] [--px-um 2.4] [--stretch | --fixed LO,HI] [--stretch-ct] [--max-px N] [output options] [--then OUT2 [output options]] ...
ufsm jpeg-fit IN.jpg OUT.jpg --div D | --max-px N [--quality 90]   # a finished JPEG shrunk (antialiased bilinear; OUT may be IN)
ufsm png2jpg <in> <out.jpg> [--half] [--max-px N]
```

**render** samples the volume (a uint8 ink level) trilinearly at -depth..depth native voxels along the surface normal and
max-composites it into `<w>.tif`; `--ct` adds the CT at t = 0 as `<w>_ct.tif`; `--scale` pixels per wrap-grid voxel. When the
wraps' union box (+- depth) fits `UFSM_RENDER_MEM` GB (default 8) the volumes are read once; otherwise the render goes
chunk-ordered: blocks sorted by chunk row, every chunk decoded once (decoders ahead of the renderers, 32 for HTTP stores),
freed after its last use, and only the volcomp substreams (16 z x 32 y x 128 x slabs) some block touches are decoded.
`UFSM_RENDER_CHUNKED=1` sends small wraps down that path too; `--joint 1` renders every wrap of the directory (one grid
size, e.g. recto + verso) in one chunk pass. Same bytes on every path (`UFSM_RENDER_TILED=1`, `UFSM_RENDER_FULLDECODE=1`:
the old paths). A whole Paris 3 wrap (338 cm², 1.6 Gpixel): 5:28 -> 1:37.

`render --gpu 1` (the default; `--gpu 0`: the CPU path; `src/nn_render.cu`; same bytes, every `--fill*` / `--depth` / `--joint` / `--crop` mode) runs
the chunk-ordered renderer on the GPU. Decoded chunks stay on the device. The decoder threads have gpudec decode the lossy
substreams straight into a device chunk pool (device-sink requests ignore `UFSM_GPU_DECODE_MIN_BYTES`). `--gpu 1` sets `UFSM_GPU_DECODE=1` (unless set) when the device has room for the GPU path,
and runs 128 decoders (`UFSM_RENDER_GPU_DECODERS`). Anything decoded on the host is uploaded from pinned staging
buffers. The kernel evaluates `src/render_core.h` `rc_pixel`, the per-pixel arithmetic shared with the CPU renderer (built
with `-fmad=false`, no fast math). Its outputs go to mapped pinned memory, two launches in flight. The pool is sized from the
device's free memory: at most min(50% of it, 8 GB) by default so a later slab fit on a shared GPU keeps room (`UFSM_RENDER_GPU_MEM` GB overrides, higher or lower). When a live chunk set exceeds the device (P3
stripes peak ~29 GB), mapped pinned host slots hold the rest. Any device error, or a pool too full for the earliest chunk row,
re-renders on the CPU. `UFSM_RENDER_PROF=1` prints decoder and driver times. Back to back, --gpu 0 -> 1: w178 per-wrap render on
the idle laptop (5080, 8 threads): render 37.1 -> 9.6 s, wall 40.6 -> 13.7 s; Paris 4 stripe ws010 on the desk (5060 Ti, load ~90): wall
43 -> 22 s, CPU 700 -> 195 s; next to a compute-saturating GPU job: 43 -> 24 s; on a full device the render falls back to the CPU (GPU
decoding stays off then), same speed as --gpu 0. The hole fills around the samples (`--fill-grid` grid bridging: its Laplace pass one hole per thread, same bytes; `--fill-inpaint` /
`--fill-pm`) stay on the CPU (a few seconds per stripe).

**ink-compare** prints per-ink statistics (mean, frac50, p50 / p95 / p99, band contrast at `--rs-um` / `--rl-um`, pairwise
correlation) and writes CT | ink panels, Pillow-BOX-halved (`--full`: full size) and fitted to `--max-px`. It streams the
renders strip by strip (Paris 3 wrap: 51.6 GB -> 0.47 GB; `UFSM_INKCMP_MEM=1`: the in-memory version, same bytes). Under 5% of
the CT pixels with data: no JPEG, exit 3. Input options: `--crop x,y,w,h`, `--crop-frac fx,fy,w,h`, `--best-crop N`,
`--threads`. Output options (each `--then OUT2` starts a new output with its own; inputs, crop and statistics are shared, read once):

| option | output |
|---|---|
| `--max-px N`, `--quality Q`, `--full` | size and JPEG quality |
| `--each PREFIX [--each-suffix S] [--each-ct]` | one JPEG per ink (and the CT): PREFIX<label>S.jpg |
| `--no-ct` | ink panels only |
| `--cmap viridis [--rgb-ct-min 1]` | ink panels through viridis; off-papyrus (CT below the floor) dark grey |
| `--rgb-agree RECTO,VERSO` | one panel: recto ink magenta, verso ink green, both white |
| `--diff-agree RECTO,VERSO [--diff-gain G]` | viridis of \|recto - verso\| (0 agree, 255 disagree) |
| `--papyrus INK [--papyrus-gain G] [--papyrus-floor median\|V]` | one panel that looks like papyrus: CT through a dark brown .. warm tan ramp, the ink as carbon black by its density; the floor raises CT below the CT median (or V in stretched units) and colours off-papyrus pixels |

### Other commands

Training / research tools still in the binary (usage via `ufsm <cmd>`): `band`, `phase-raster`, `phase-check`, `downsample`,
`side-votes`, `codes-merge`, `wind-assign`, `wind-seed`, `phase-fuse`, `recto-fuse`, `restore-eval`, `restore-stats`, `ct-hist`,
`convert-unwrap2`, `coreline`, `stitch`, `blend`, `refine`, `surf-snap`, `wrapdist`, `fiber-check`, `fiber-export`, `hires-phase`,
`hires-count`, `patch-raster`, `ray-check`, `reg`, `sr-pairs`, `dinovol-*`, `dino-*`, `hercuvit`, `villawind`, `vwt-train`,
`vwt-tile`, `vwt-infer`, `vwt-stitch`, `render-sheets`, `overlay-red`, `phase-vis`, `vol-cmp`, `store-bench`, `wl-dump`.

## Outputs and file formats

- **Volumes**: zarr v3 with volcomp codecs, one group per pyramid, arrays keyed by level index (CT) or by voxel size in um
  (our predictions: `ink.zarr/9.6`). Write and publish with 512^3 (or 1024^3) shards; `predict --fit-zarr` writes 128^3 shards,
  `ufsm repack` re-shards losslessly. Mirror CT has 16^3 codec blocks (seams in inputs and targets).
- **Predictions**: uint8 prob254 (`round(254 p)`), 255 = unknown / ignore.
- **Fit-field set**: `fields.json` (format, origin, shape, cell `h`, um, axis table) + one uint8 array per plane: `pc ps`
  (confidence x cos / sin of the phase, 128 = no information), `t0..t5` (normal tensor n n^T), `sp` (sheet spacing from the
  density), `w` (material fraction), optional `dz dy dx dq` (unwrap2 winding mod 4) and `r` (recto max).
  `scripts/wbench/mkcrop.sh` makes a crop as symlinks + offsets.
- **Fit model / set**: a directory with `model.json` (+ lattices); a joined set carries the slab models and their gauges.
- **tifxyz**: a directory `wNNN/` with `x.tif y.tif z.tif` (float32 grid, -1 invalid) and `meta.json` (`scale`,
  `voxel_size_um`, `winding`). **.sfc**: the surface-compressor file (`third_party/surfcomp`, format 4) holding x / y / z as a
  joint XYZ triple within `--sfc` / `--error` voxels plus the meta.json bytes; every tifxyz reader takes `wNNN.sfc` directly.
  Flat wraps add `gu.tif` / `gv.tif`, faces add `conf.tif` / `thick.tif`.
- **Renders**: `<w>.tif` (ink, uint8) and `<w>_ct.tif`, tiled / strip TIFF; panels are JPEG (at most ~65500 px a side).
- **Labels** (our exported stores): legacy targets uint8, volcomp lossless, `fill_value` 255: `0` background, `254` surface,
  `255` ignore; coarser levels hold `round(254 * surface fraction)` of the non-ignore children, or 255 when at least half are
  ignore. The sampler turns this into a soft target and a loss mask (CT > 0 and not ignore). The Paris 4 all-surfaces target
  uses binary `0`/`255`, `fill_value` 0, the lossless mask codec and a 4.8 um finest level; its zeros are supervised
  background wherever CT > 0.

## Uploads

Public products go to `dl.ash2txt.org/community-uploads/forrest/` (SFTP / lftp from the desk; credentials stay out of the repo):

```
volcomp/<SCROLL>/volumes/<scan>.zarr                                     # CT mirror (volcomp, the pipelines read it over HTTP)
volcomp/<SCROLL>/representations/predictions/fit-fields/<scan>-fitfields-20261016-ufsm-prod96-h2.zarr
volcomp/<SCROLL>/representations/ufsm-c2f-20261028/{axis.json,box.txt,ref,joined,wraps,ink.zarr[,fields]}   # scripts/corpus/upcorpus.sh
volcomp/<SCROLL>/segments/ufsm-c2f-20261028/{README.txt,sheets.tsv,segments/<wNNN>/ (flat tifxyz + rects.tsv),renders/}
```

Only scrolls that are not eligible for a First Letters / Grand Prize are uploaded (PHerc1447, 1667, ParisMAN5, ParisMANB,
Paris3, 0139, 0814, 0841, 1451, 1299, 0009B, 0332, 0343); ask before uploading anything else.

## Known caveats

- **The winding gauge is arbitrary.** Wrap ids are not turns from the centre. Paris 4: W at the umbilicus is 78-93 over z,
  the GT codes are winding mod 14 (w = code - 8 only mod 14), outer wraps end ~w200.
- **Junk inner wraps at the end caps.** Paris 4 w000-w077 exist only at global z 18192-18832, the frayed top cap (the top
  slabs and the ref have W -2..150 across the section there); every wrap below ~w150 carries a junk fragment there, and the
  bottom slab (z 256-384) is broken the same way. The real core is w078-w100. Extract with `--zr 384,18176 --col-r p90` and
  exact `--first`; the p4full wraps and the strip built from them (w016-w272) still contain the junk.
- **Spurious outermost wraps.** Paris 4 w216+ and Paris 3 w422+ are flagged dup by wrap-check or lie mostly off the CT
  (ink-compare exits 3). The interior counts match the CT (1.0-1.05 peaks per turn).
- Wraps (level sets) can sit in the gap beside the sheet; use `faces` for the papyrus surfaces. 9.6 um ink smears across the
  ~4 voxel sheet: recto vs verso ink correlates 0.89-0.91.
- Flattened filled holes are bridges; `strip-flatten` keeps a few flipped triangles (0.0045% of the area on Paris 4).
- Single-pass halo-0 ink has window seams (border jump ~30 uint8); use the frozen-GN overlapped passes + `ink-blend`.
- No `ufsm fit` on the laptop (2 cm box fits OOM-killed WSL); fits run on the desk / forlindesk, one per GPU.
- Do not cache whole published volumes locally (the chunk cache filled the desk disk); whole-scroll fit fields are 100s of GB
  (`surf-stats --ct-only 1` reads only their geometry).
- JPEGs are limited to ~65500 px a side: stripes are cut at 60000 px, panels capped with `--max-px 65000`.

Third-party code: `third_party/volcomp.h` and `third_party/surfcomp/` (MIT, SuperOptimizer).
