# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

ufsm ("ultra fast scroll model"): C23 host code + CUDA kernels, no PyTorch / cuDNN, no Python in the pipeline (Python only in
`tests/` as parity references). One binary, `build/ufsm`, covers the whole Vesuvius Challenge chain: ingest into our own codecs
(volcomp volumes, surfcomp surfaces), training and inference of 3-D U-Nets (winding fields, ink, papyrus, recto/verso), the
winding fitter, wrap extraction, flattening, rendering. `ufsm` without arguments lists the commands; each prints its usage on a
missing argument. README.md holds the build notes and the full unroll recipe; DESIGN.md (newest sections first, dated) holds the
design decisions, experiments and numbers; FINDINGS.md the results.

## Build and test

```sh
scripts/toolchain/install_mcc.sh                              # once per machine / pin: microc (mcc + mcudahw) at cmake/mcc.version, ~/.local/mcc
cmake --preset release && cmake --build build -j 8            # build/ufsm by mcc (the only supported compiler since 2026-11-15; PTX, mcudahw)
scripts/toolchain/jit_warm.sh build-verify/test_unet 8        # fill mcudahw's kernel cache before suites / training (cold JIT times tests out)
cmake --preset verify && cmake --build build-verify --target test   # fast tier: the merge gate. test-slow / test-all: slow tier / both
SUITE_FLAGS="-j 4 test_fit" cmake --build build-verify --target test # runner options: only tests whose name matches the regex
cmake --build build-verify --target test_lat && build-verify/test_lat   # one test by hand
cmake --preset arch86 && cmake --build build86                  # Ampere (software MX; sm_86 SASS + PTX); arch86-verify for its suite
cmake --build build --target doctor                             # the toolchain picked + machine limits (ufsm sysinfo)
cmake --preset llvm-verify && cmake --build build-llvm-verify # clang + nvcc, kept building for comparison / bisecting (llvm-release too)
scripts/test/desk_suite.sh BRANCH   # push to the desk, build and run the suite there (one suite at a time)
```

- `cmake --build build-verify` (the default target) does not rebuild the suite's test binaries: after kernel or library changes run
  suites through `--target test` (or build the test target), or stale binaries fail with "invalid device function".
- Build dirs are configurations that never invalidate each other: `build/` (release, the fp16 / MX kernels) and `build-verify/`
  (`UFSM_VERIFY=ON`, adds the bf16 / fp32 instantiations); `build86*/` for `UFSM_ARCH=86`. The suite binaries run from
  `build-verify/`; `test` also builds the release-dir helpers the Python tests use (`build/ufsm`, `make_pipeline_fixture` ...).
  On the desk: `~/bin/lowprio`, `-j 8` at most. ccache is shared across worktrees.
- Toolchain: microc (`cmake/mcc.cmake`, pin `cmake/mcc.version`): mcc compiles the host C and every .cu unit to PTX in the objects;
  mcudahw JIT-compiles it at first launch (cached in ~/.cache/mcudahw). `UFSM_MCC` / `MCC_HOME` / `~/.local/mcc` locate it. The PTX
  is packed (MPY, ~10x); verify builds keep each unit's in `OBJ.ptxz` beside its object (`UFSM_PTX_FILE`, mcc --cuda-ptx-file), so
  build-verify is ~1.8 GB and its binaries run only where they were built (`MCUDA_PTXDIR` if moved). The clang + nvcc presets (`llvm-release`, `llvm-verify`, cmake/llvm.cmake)
  stay buildable for comparison: there nvcc's host compiler is probed at configure time (clang++ with `-allow-unsupported-compiler`,
  else the newest g++), host .c units and the link clang / lld.
- A new test: one `ufsm_exe(test_x SUITE tests/test_x.c OBJS ... LIBS ...)` line in `tests/targets.cmake` (OBJS `NN LP` = every
  nn_* / lp_* CUDA object, `CORE` = every src/*.c object but main, or named objects like `unet json`; LIBS `LDLIBS`, `CUDA`, `m`) and
  a line in `tests/suite.txt` (`NAME | TAGS | EST_SECONDS | COMMAND`; tag `gpu` if it touches the GPU, `gpu2` for two, `slow` above
  ~30 s; `UFSM_TEST_QUICK=1` for a fast variant with `NAME~full` in the slow tier). The `suite-check` target rejects either half alone.
- `tests/run_suite.sh` takes machine-wide GPU locks (`/tmp/ufsm-suite-locks`: `gpu<i>.excl` shared plus a slot), so concurrent suites
  share the GPUs; a trainer takes `gpu<i>.excl` exclusively. A card held by a trainer or a foreign process: its tests go BUSY after `-W`
  seconds. `-g none` runs only the CPU tests. Exit 0 pass, 1 fail, 3 INCOMPLETE (GPU tests not run: NOGPU / BUSY / NEED2): not a merge
  gate. The GPU half of a merge: `scripts/test/pod_gate.sh REF` (the perf pod, `run_suite -c <merge base>` under /root/gpu.lock).
- One GPU, 2-GPU tests: `train --gpus 0,0` runs two data-parallel ranks as logical devices on one card (src/ufsm_dev.cuh); gpu2 tests
  tagged `l2` run that way on one-GPU machines (UFSM_TEST_GPUS=0,0).
- Training runs: `scripts/run/install_health.sh RUN` (the health watcher service); `scripts/run/replay_ab.sh RUN OLD NEW` before a
  binary swap; `--ckpt-ring` / `--snap-every` (src/ckring.c). A result-changing fallback prints `DEFAULTED:` (src/defaulted.h).
  /vesuvius scratch retention: `scripts/run/retain.py` (dry run; `--apply`; a `KEEP` file pins an entry).
- Sources are globbed: a new `src/*.c`, `src/nn_*.cu` or `src/lp_*.cu` is picked up at the next build (no list to edit). CUDA units
  that must be byte-identical to their CPU versions go into `UFSM_STRICT_UNITS` in CMakeLists.txt (no fast math, no FMA).
- Useful env: `UFSM_MEM_REPORT=1..3` (allocation report; 3 logs each allocation with its caller), `UFSM_MEM_DRY=1` (the planner
  only, no training), `UFSM_NCPU` (thread cap).

## Architecture

- **CLI**: `src/main.c` dispatches the commands; most have their own file (`train.c`, `predict.c`, `fit.c`, `render.c`, ...).
  Options are read with `opt(argc, argv, "--name", "default")`.
- **GPU ops** (`src/nn.h`, the C API): `src/nn_*.cu` are the fp16 / fp32 ops by section (shared helpers in `nn_common.cuh`),
  one object each (`NN` in targets.cmake). `src/lp_*.cu` are the low-precision MX-fp8 / MX-fp4 block-scaled kernels
  (`lp_common.cuh`, API `nn_lp.h`), one translation unit per kernel family and instantiation (`lp_f4fwd_i3.cu` ...; `LP`); on
  sm_120a they use the block-scaled tensor-core MMA, on ARCH=86 software emulation. `nn_gpudec.cu`, `nn_cf.cu` and the other
  `UFSM_STRICT_UNITS` are built without fast math because they must be byte-identical to their CPU versions.
- **Network** (`src/unet.c`, `unet.h`): the U-Net, its memory planner (`build_acts`: which activations are kept, recomputed,
  offloaded to host; "lean" modes), forward / backward, optimizers (Muon + AdamW), checkpoints (`checkpoint.c`). Activations
  are stored in MX-fp4 / fp8; level 0 dominates memory. Checkpoint compatibility rule: new header keys are written only when
  non-zero and new parameters are appended last, so old checkpoints load unchanged.
- **Training** (`src/train.c`): tasks (`--task unwrap | unwrap2 | restore | ...`), the per-GPU state, the window planner, and the
  loss path. Full-resolution logits are never held whole: the head runs in 32-plane z-slabs (`head_slab_pass`), each slab's
  loss gradient is fed straight back. Inputs are built on the GPU (`--gpu-input`, gpu-aug in `nn_aug.cu`).
- **Data** (`src/sample.c`, `band.c`, `sources.c`): the sampler builds training windows and full-resolution label planes
  (phase, side, winding, papyrus) from sources json files; labels are always full resolution (`label_res_check`).
  Augmentations: `ct_augment.c`, `scan_augment.c`, `spatial_augment.c`, `crush.c`, `scrollaug.c`.
- **Storage**: `store.c` (local / http roots), `zarr3.c` + `z3w.c` (read / sharded write), `third_party/volcomp.h` (the CT
  codec), `third_party/surfcomp` (surfaces), `embz.c` (channel-last int8 stores), `zcache.c` (shared decoded-CT cache).
- **Inference** (`src/predict.c`): windowed prediction with halos into zarr pyramids or fit-field sets.
- **Unroll pipeline**: `predict` (fit fields) -> `count-field` (global integer reference) -> `fit` slabs (`fit.c`, `fit_ct.c`)
  -> `fit-join` -> `wraps` -> `flatten` -> `render` -> `ink-compare`; drivers in `scripts/unroll/`, per-topic launchers in
  `scripts/<topic>/`.

## Conventions

- Wide lines (~140 columns), dense C23 (`nullptr`, `constexpr` style), comments explain why and cite measurements.
- Kernels get a test against a host reference (byte-exact where possible) or central-difference gradients before use.
- Every non-trivial feature or experiment gets a dated DESIGN.md section with its numbers.
- Work happens on dated branches (`topic-YYYYMMDD`), often in separate worktrees; the fast suite must pass before merging to
  master. Commit and push after each passing step.
- Shell is zsh: unquoted variables are not word-split, so pass flag strings through `bash -c` or arrays.
- Before deleting data directories, check for symlinks pointing into them from live trees.
- Rented pods (containers): `nproc`, `/proc/cpuinfo` and `free` show the HOST, not the pod. The real limits are the cgroup's:
  CPUs = `/sys/fs/cgroup/cpu.max` quota / period, RAM = `/sys/fs/cgroup/memory.max` (v1: `cpu.cfs_quota_us`,
  `memory.limit_in_bytes`). Size `-j`, sampler `--workers` / slots and `UFSM_NCPU` from those. A 2x5090 pod showed 256 CPUs /
  755 GB but had ~62 CPUs / 200 GB. Pods cost money: queue their work before asking for one, and report as soon as the queue is
  empty.
- NEVER copy label data from the desk (forlindesk2) to a pod; its uplink is ~3 MB/s. Pods read CT from dl.ash2txt (volcomp),
  fibres from dl.ash2txt community-uploads/qual/AutomatedFiberVolumes, and the ground truth from dl.ash2txt, and compute every
  label on the fly during training (the sampler's on-the-fly paths plus the derived-label cache). Only tiny files (sources json,
  axis jsons, scripts) may be copied.

## Team rules (agents; 2026-11-20)

- **Concise.** Messages and reports: lead with the result or the decision needed, numbers in a short table, ≤ ~10 lines. No
  restating plans, no repeating an earlier report, no "waiting" notes: write only when something changed or you need an answer.
- **Master constantly.** Merge the day a change passes its touched tests; merge origin/master in every few hours and before building
  any run binary. After a merge delete the branch (`--force-with-lease`) and its worktree. One worktree per agent. Unmerged work
  older than a day gets merged or deleted. `scripts/dev/cleanup.sh` lists stale branches / worktrees / build dirs.
- **No silent fallbacks.** Any CPU fallback, precision drop or skipped path prints `DEFAULTED:`. Gates treat a missing or empty
  score as "scoring failed" (alert), never as a fail / STOP.
- **Check a new run within 15 min:** device placement lines (CT decode, ink / fibre labels on the GPU), no DEFAULTED, GPU busy from
  `nvidia-smi dmon` (60 s mean; the log's "wait" alone is not proof), Mvox/s. Device indices are the ones inside the process's
  CUDA_VISIBLE_DEVICES.
- **Desk CPU belongs to the trainers.** Nothing but trainers in ufsm-train.slice; other jobs in ufsm-dev.slice, ≤ 6 threads, one
  heavy job per agent. Card-1 GPU tests go into the shared batched slot (inksdk schedules it), scoring runs one job at a time.
- **Widths** are multiples of 32, or exactly 8 / 16 (else gradients drop to MX-fp8 and the lean level-0 modes switch off).
- **Disk:** outputs to /vesuvius or the NAS, never ~; delete or move to the NAS what no live run or DESIGN section needs.
- **No approval prompts.** A prompt from a subagent never reaches the user's web UI: the agent hangs. So never send a command the
  checker can't verify. Banned in a Bash call: `rm` / `mv` / `sed -i` / `chmod` / `kill` inside loops, `case`, `if`, `&&` chains,
  pipelines, `$(...)`, `bash -c` / `sh -c` / `zsh -c`, `find -delete` / `-exec rm`, `xargs rm`, globbed deletes. Instead:
  - edit files with the Edit / Write tools;
  - delete or move with one plain command per call on literal paths (`rm -r /vesuvius/ufsm/x/old`);
  - multi-step work: Write a script file under /vesuvius/ufsm/<agent>/ first, then run it as `bash /vesuvius/ufsm/<agent>/x.sh`;
  - read-only inspection (grep, ls, cat, du, ps, git log / status) may be compound.
  Scripts reference stable links (e.g. /vesuvius/ufsm/bin/ufsm-<agent> → current binary), never worktree paths, so repointing is one
  `ln -sfn`. If a command of yours ever waits on approval, tell the lead at once.
