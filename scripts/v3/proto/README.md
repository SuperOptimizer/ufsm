# Unroll v3 prototypes: tracer + patch turn solver on PHerc1667 boxes (2026-11-06)

This directory holds the Python and shell prototypes behind the surface-first redesign. The full plan, P0–P7, is in `docs/unroll_v3_plan.md`. The scripts live and run on the desk, in `/mnt/scratch/v2/trace` on forlindesk2; this directory is a copy. Each box keeps its data in `<BOX>/` there: `gt.raw`, `prob.raw`, `ridge.raw`, `ctbox.raw`, `ours.raw`, and `paths/` with the VC3D patches.

Notes for running them:
- The remote login shell is zsh, so wrap `for b in "C2 ..."; do set -- $b` loops in `bash -c`.
- On the desk, the Python interpreter is `~/miniconda3/bin/python3`.

## Test bed

Use PHerc1667 bench boxes. The 1667 GT follows the CT sheets, and the scroll is never trained on. The grid is 9.596 µm, boxes are 512³, and the axis file is `/mnt/scratch/pods/PHerc1667/unroll/axis.json`. Its control points are in 2.4 µm voxels, so scale them by 0.25. GT spiral sense is −1, ours is +1.

| box | z y x | packed share |
|---|---|---|
| C2 | 3328 1152 512 | 18.5% |
| C3 | 5120 1024 768 | 11% |
| C7 | 3840 2304 1152 | 4% |

Do **not** use the Paris 4 GT to judge voxel placement or purity in packed boxes (PH1–3). Those GT sheets are smoothed, evenly spaced lines that do not follow the CT. HOLD is fine.

## Pipeline as prototyped

1. **`box1667.sh NAME Z Y X 512 512 512`** reads the GT codes and the m7 L2 prob (dl.ash2txt). It writes `ridge.raw`, the radial non-max suppression of the prob at p ≥ 64 along the direction from the axis, then prints detector recall and precision.
2. **`trace_box2.sh NAME Z Y X 512 512 512 64 6`** runs the VC3D tracer, image `localhost/vc-local:20260828` (a podman build of the villa source). It needs env `VS=9.596 AXIS=... SGN=-1`.
   - Input: the ridge as a max-pooled zarr v2 OME pyramid plus `meta.json`.
   - It runs `vc_gen_normalgrids` first; these are required, or patches ignore the data.
   - Seeding: 64 seeds, step_size 5. The run ends by calling `score_patches.py`.
3. **`cutpatch.py`** cuts each patch at vertices with no ridge support (ridge within 1 voxel) into 4-connected components, and reports purity before and after.
4. **`psolve4.py BOX Z0 Y0 X0 512 AXIS 0.25 -1`** is the turn solver (best version).
   - **Nodes:** the cut components with at least 100 vertices.
   - **Rays:** bins of 4 voxels in z by 4 voxels of arc around the axis.
   - **Votes:** consecutive crossings a→b along r vote `turn_b − turn_a = 1 + (sheets between)`. The sheets between are counted two ways and both votes are pooled: ridge runs, and m7 prob peaks with prominence ≥ 32.
   - **Edges:** the per-pair mode, with at least 3 votes.
   - **Solve:** a weighted L1 integer solve (least squares, then weighted-median sweeps).
   - **Pruning rounds:** drop the top 5% of components by violated-vote share (threshold .25), then re-solve.
   - **Scoring:** against GT sheet ids (mod 14) in the largest graph component.
5. **`basespiral.py`** scores the current spiral pipeline (codes in `/mnt/scratch/v2/cal/PHerc1667/codes.zarr`) with the same metric.

Diagnostics:
- `normdev.py`: patch normal vs data normal (it does not flag sheet jumps).
- `edgediag.py`: edge vote vs GT turn difference.
- `cntdiag.py`: gap-counter comparison.
- `dbgjump.py`: slice render of the worst patch.

## Results

**Detectors (m7 mask ≥ 128, recall / precision at ±1 voxel)**

| box | recall | precision |
|---|---|---|
| C2 | .61 | .61 |
| C3 | .66 | .82 |
| C7 | .77 | .78 |

The thin ridge has about .1–.15 lower recall and a slightly higher precision.

**VC3D patch purity**

| box | uncut | after support cut | area kept |
|---|---|---|---|
| C2 | .90 | .91 | 83% |
| C3 | .84 | .95 | 77% |
| C7 | .974 | .997 | 83% |

Recall is about .3 per 64 seeds. On HOLD (Paris 4, open) purity is .97.

**Right sheet and right turn (share of GT-matched area)**

| | C7 | C3 | C2 |
|---|---|---|---|
| current spiral pipeline | .63 | .68 | .43 |
| psolve (ridge counts) | .969 | .845 | .525 |
| + pruning (psolve2) | .997 | .91 | .56 |
| prob-peak counts only (psolve3) | .976 | .53 | .70 |
| **pooled counts + pruning (psolve4)** | **.997** | **.89** | **.70** |

The current-pipeline baseline is scored on GT voxels covered by our spiral, about a third of GT. The pruned rows also drop 4–14% of matched area.

## Findings

- **Open regions are close to solved:** trace, cut at unsupported vertices, then solve the turns from radial gap counts.
- **In packed regions, the C2 jumps lie on strong m7 response.** The detector itself merges touching sheets, and normals don't flag it.
- **C2's solver errors are mostly gap miscounts,** not impure patches. Of the edge weight between pure patches, 9.5% undercounts by 1 because a sheet was missed between them.
- **CT intensity peaks are useless for counting:** they pick up fibre layers and overcount 60–100% of gaps.
- **The greedy pruning and median-sweep solve are fragile.** Which group of patches becomes the largest graph component can flip, as in psolve3 on C3.

## Next steps (P4 of the plan)

1. **Port the solver to C** as `src/patchsolve.c` (`ufsm patch-solve`), with a test.
   - Use the existing exact L1 solver `src/l1pot.h` (`l1_votes_merge`, `l1_solve`, PUMA graph cuts) in place of the median sweeps.
   - Read patches with `src/tiff.h` and write test patches with `src/tiffw.h` (`tifxyz_write`). Use `axis_load` / `axis_at` from `src/sources.h`, and read the prob box with `z3_read`.
   - Compute the ridge internally from the prob.
   - Optionally compute a `--gt` score for parity with psolve4.
2. **Make the edge model more robust:** per-ray votes into L1 directly (not per-pair modes), and a confidence-weighted choice between the two counters.
3. **Widen the test:** many 1667 boxes, more seeds for coverage, then tiles across a slab, then whole-scroll (P5, pods; ask the user before any spend).
4. **Packed sheets:** a side-aware (recto/verso) detector or tracer to separate touching sheets (P3b / P6). This is the real blocker for C2-like regions, and for Paris 4 PH1–3.
