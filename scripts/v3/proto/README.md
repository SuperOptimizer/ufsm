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

## C port: `ufsm patch-solve` (src/patchsolve.c, 2026-11-06)

`ufsm patch-solve --paths BOX/paths --prob BOX/prob.raw --gt BOX/gt.raw --box Z,Y,X,512,512,512 --axis AXIS --axis-scale 0.25 --sgn -1`
does psolve4 end to end in C (ridge computed internally, 0.3 s per box on the desk) and solves with the exact L1 solver (`l1pot`).
Test: `tests/test_patchsolve.c` (concentric sheets, one untraced sheet that must be counted).

- **Parity:** identical graph to psolve4 (C2 79 nodes / 1012 edges / agreement .896; C3 87 / 981 / .917; C7 65 / 799).
- **Scores (right sheet + turn):** C7 .997, C3 .92 (psolve4 .89), C2 .52 (psolve4 .71). The exact solve reaches a far lower energy on C2
  (violated weight .065 vs .141). psolve4's .71 was a lucky local minimum (its rounds jump .55 -> .69), so the C2 votes themselves are wrong.
- **Oracle (`--oracle 1`, GT turn differences as edge values):** C2 .912 and C3 .950, which equal the purity ceilings. The solver
  and the patches are fine; the whole gap is gap counts (23.5% of C2's edge weight is wrong, 12% of C3's).
- **Vote models tried (`--votes pair|ray`, `--counter ridge|prob|pitch|maxrp`, lists allowed):**
  - Every variant leaves C2 at .52-.56.
  - `ray` votes give C3 .93; `maxrp` (max of the ridge count and dr / box pitch) gives C2 .71 but breaks C7 (.50).
  - Defaults stay pair + ridge,prob.
- **Why (`--cross-tsv` dump):** in C2, 29% of GT gap-2 crossings and 39% of GT gap-3 crossings are counted one short. The miscounted
  gap-2 crossings have the same dr (8-16 voxels) as true gap-1 crossings, so no counter built on the current detector can fix them.
  Same-sheet duplicate traces also always vote d >= 1 (C7: 646 crossings at dr ~4); they are a small effect.
- **Conclusion:** step 4 below (a detector that separates touching sheets) is the blocker. Step 2 is done and gives no gain on C2.

## Own tracer: `ufsm sheet-trace` (src/sheettrace.c, 2026-11-06)

VC3D cannot run at corpus scale: its CUDA path is commented out (`GrowPatch.cpp`), so each grower is CPU Ceres at about 0.25 mm²/s.
On C7 with 256 seeds it reached recall .57 / purity .98 in 36 min.

`sheet-trace` builds a column grid, (z, u) every `--step` voxels, with rays from the axis or parallel to a fixed direction. Prob peaks
along each ray (≥ `--thr`, sub-voxel) become height fields r(u, z). A step to a neighbouring column is taken only when exactly one peak
lies within `--tol` of the extrapolated r. On C7 it reaches the same recall at purity .995 in about 1 s per 512³ box.

- **`--tile T`** writes each patch as T×T-cell tiles. **links.tsv** records the same-sheet links: seams between tiles of one patch
  (0.5% wrong on C7, 9% on C3) and contacts. Contacts are 45–58% wrong, from weak leftover patches, so `patch-solve` uses seams only.
  `patch-solve --cut 0 --links links.tsv --frame frame.txt` then cuts staircase patches (sheet switches through merges) where the gap votes
  disagree.
- **`--dir auto`** picks the ray frame from the structure tensor: parallel rays along the dominant normal when the radial alignment is < .7,
  the coherence is ≥ .5 and the axis is outside the box. Crushed scrolls need it, because radial rays run along their sheets: 0125's traced
  share went .40 → .98, 0009B .42 → .98, Paris 4 .88 → .97.
- **`--regions R`** gives an R×R xy grid of regions, each with its own frame (folds). A shared 2-voxel cell grid stops duplicates, and
  rseam links join regions. It is neutral on the GT boxes and gave 0191 .69 → .76.
- **`--extra CODES`** adds second sheet evidence, e.g. an older spiral raster. Its peaks join the columns where m7 has none within
  `--extra-sep` voxels; `--extra-gap` g > 0 limits them to gaps ≥ g × the column pitch.
  - Paris 4 HOLD, with our c2f surfaces: recall .70 → .79, solve .53 → .88, effective .37 → .70.
  - 1667, whose old spiral has false sheets: C7 solve .93 → .34.
  - The GT-free violated share rises in both cases, so it cannot be chosen automatically. It is off by default; use it only with an extra
    source known to be good.
- **`--nms R`** (default 5 voxels): along each ray, keep only the strongest peak within ±R. m7 often peaks on both faces of one sheet
  (HOLD: about 20k same-sheet crossings at Δr ≈ 7).
  - On 0139, recall rises on every box: 139-1 .36 → .42, 139-5 .748 → .756.
  - On 1667 and HOLD it is neutral to slightly positive. R = 3 is worse on C3.
- **Recall is saturated against m7:** lower thresholds, smaller tiles or step 1 move recall by ≤ .015, and over 99% of peaks are owned.
  The remaining misses are detector misses.
- **`--jpg`** writes a QA image: the mid-z prob with traced peaks coloured by tile; red = untraced. `ufsm raw2jpg` draws any slice.

**GT boxes** (`scripts/v3/ev_box.sh`; effective = recall × solve accuracy):

| box | recall | tile purity | solve | effective | VC3D-cut effective |
|---|---|---|---|---|---|
| C7 | .57 | .997 | .93 | .54 | .23 |
| C3 | .54 | .96 | .77 (.50 on another run) | .42 | .21 |
| C2 | .44 | .96 | .56 | .25 | .10 |
| HOLD (P4) | .70 | .988 | .53 | .37 | – |
| 139-5 | .64 | .997 | .71 | .46 | – |
| 139-1 (48% packed) | .34 | .96 | .27 | .09 | – |

The solve is the limit, not the tracer. With GT edge values (`--oracle 1`) HOLD reaches .97 and C3 .94, but the gap counts are wrong
where m7 misses or merges sheets. The single-gauge score is unstable: C3 gives .77 or .50 on near-identical traces.

**Corpus** (`scripts/v3/corpus_trace.sh`: one 512³ box per scroll at half the papyrus extent, m7 prob from the mirror; GT-free QA
in `/mnt/scratch/v2/trace/corpus/summary.tsv`; `corpus_cmp.sh` compares runs):
- 31 of 33 scrolls traced in 6–15 s per box. Paris 3 has no m7 prediction; 0175B's m7 is empty at the box.
- Traced share and vote agreement are both > .9 on 0009B, 0125, 0139, 0211, 0257, 0306B, 0483A, 0800, 0813, 0841, 1218, 1447, 1451,
  1545, 1667 and Paris 4.
- Weak: 0268 (.56), 0826 (.65), 0358 (.65). Their m7 prediction has no coherent sheets there (QA image), so the detector is the problem,
  not the tracer.

## Next steps (P4 of the plan)

1. **Port the solver to C** as `src/patchsolve.c` (`ufsm patch-solve`), with a test.
   - Use the existing exact L1 solver `src/l1pot.h` (`l1_votes_merge`, `l1_solve`, PUMA graph cuts) in place of the median sweeps.
   - Read patches with `src/tiff.h` and write test patches with `src/tiffw.h` (`tifxyz_write`). Use `axis_load` / `axis_at` from `src/sources.h`, and read the prob box with `z3_read`.
   - Compute the ridge internally from the prob.
   - Optionally compute a `--gt` score for parity with psolve4.
2. **Make the edge model more robust:** per-ray votes into L1 directly (not per-pair modes), and a confidence-weighted choice between the two counters.
3. **Widen the test:** many 1667 boxes, more seeds for coverage, then tiles across a slab, then whole-scroll (P5, pods; ask the user before any spend).
4. **Packed sheets:** a side-aware (recto/verso) detector or tracer to separate touching sheets (P3b / P6). This is the real blocker for C2-like regions, and for Paris 4 PH1–3.
