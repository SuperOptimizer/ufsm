# wbench: the winding + fitter bench

One command scores a winding checkpoint (or a fitter change) on the clean held-out boxes (default `BOXES=clean`: the boxes
of `boxes.tsv` not tagged `label-noise`) plus 4 full-cross-section Paris 4 slab sites. It prints rows that `ufsm wbench-stats` turns into means with bootstrap CIs and paired differences.

    # desk, your own GPU (B = 0, C = 1); the pinned binary is $WB/bin/ufsm
    SITE=desk GPU=1 bash scripts/wbench/wbench.sh /path/ckpt.ckpt mytag
    # compare with the baseline (rows in $WB/res)
    ufsm wbench-stats $WB/res/mytag.tsv $WB/res/wind96.tsv --metric p05x,ids,p1x,gross100,frac
    # a fitter change on fixed fields (no predict): FIELDS names the run whose fields are reused
    SITE=desk GPU=1 FIELDS=wind96 FIT="--uw-repair 12" bash scripts/wbench/wbench.sh /path/ckpt.ckpt wind96-rep12

`WB` is `/vesuvius/ufsm/wbench` on the desk and `~/ufsm-data/wbench` on the laptop (`site-*.env`).

## Box set (frozen 2026-10-25: `boxes.tsv`, `holdouts.json`)

| scroll | boxes | grid | notes |
|---|---|---|---|
| Paris 4 | PH1 PH2 PH3 HOLD | level 2 (9.6 um) | HOLD is clean for everything; PH1-3 are in-sample for wind96 and dino/full |
| PHerc1667 | C1..C70 | level 2 (9.6 um) | eval-only scroll: never train on any of it; C25-C70 added 10-25 (38 clean) |
| PHerc0139 | 139-1..139-8 | level 0 (9.362 um) | 139-1 = the dino/full holdout |
| PHerc0814 | 814-1..814-8 | level 0 (9.362 um) | 814-1 = the dino/full holdout |
| PHerc0500P2 | 500P2-1 | level 0 (9.362 um) | the labels are sparse |
| Paris 4 July-only | J1..J34 (28 clean; J18 J21 J29 J31 J33 ceil < 0.8, J28 air) | level 2 (9.6 um) | July labels where June has none (z 6656..6912, 17024..17408); codes from `gtjuly/build.sh` (`P4JQ`); in-sample for d1 before its p4resume relaunch |

The 31 boxes (plus J18 J21 J28 J29 J31 J33) whose wind96 label ceiling is under 0.8 (all 0814, 139-1..4 and 6..8, C1 C5 C6 C8 C13 C17 C20 C23, C25 C41 C46 C48 C53 C65 C67 C69) carry the
note `label-noise` and are skipped unless `BOXES=all`: their labels, not the model, limit the score (e.g. 139-2 ids 0.78 at
ceil 0.55). They stay held out of training. `BT=file` runs another box list (e.g. `cand-1667.tsv`).

`BOXES=core` (44 boxes, ~50 min on the laptop: the clean paris4 + paris4j boxes and 12 z-spread 1667 boxes tagged `core`) is the
routine snapshot eval; the full clean set (88, ~1.7 h) is for milestone students and fitter arms.

**Training must avoid all of them.** Add this to every source entry of 0139, 0814, 0500P2, Paris 4 and 1667 (the train sampler
and `dino-train` both read it):

    "holdout_file": "/vesuvius/ufsm/wbench/holdouts.json"

The file matches boxes by the source's CT basename. `"holdout"` also accepts a list of boxes `[[z,y,x,nz,ny,nx], ...]`; the
first box is the validation box.

## What one run does

1. **Predict.** Production fit fields per box (`predict --fit-zarr --fit-h 2 --fit-q 8 --fit-q-phase 0`, window 384 x 640^2,
   halo 64, EMA weights). The fields are cached in `$WB/fields/TAG/BOX`; `KEEP=0` deletes them after the run.
2. **Fit.** The production fitter (`fit --dir 1 --uw-repair 8 $FIT`) is scored against the published-wrap codes:
   - `ids0` is the box score;
   - `ids_a` and `ids_b` are fits on 16-cell shifted sub-domains of the same fields (`REP=2`);
   - `ids` is the mean of the three, and `ids_sd` is their spread (fit noise);
   - `e08` is the share of voxels whose winding is right within 0.08 turn.
3. **Pair accuracy.** Pairs of labelled voxels about 0.5 / 1 / 2 / 4 mm apart; a pair counts as right if the fitted winding
   difference is right (columns `p05 p1 p2 p4`). The `x` columns (`p1x` etc.) keep only pairs on different sheets. Pair
   accuracy has no gauge, so one slip costs only the pairs that cross it. `ids` instead jumps to the larger piece's share.
4. **Network phase vs the label phase** (`fields-oracle` + `fit-compare`; needs no fit):
   - `pmed`: median phase error;
   - `gross`: share of cells more than 1/4 turn off;
   - `gross100`: the same on packed cells (label spacing under 100 um);
   - `conf`: median confidence.
   - Turn counts without the fitter (fit-compare): 0.5 mm radial segments from every 4th labelled cell; the label phase
     unwrapped along the segment gives the reference turns, the network phase unwrapped the same way gives `cnt_bias` (median
     error in turns; negative = sheets missed) and `cnt_off` (share of segments >= 0.5 turn off; `cnt_offp` on packed segments,
     label spacing < 100 um). `dcnt_*` are the same counts from the density head (|n . r| / spacing integrated along the segment).
     wind96: PH1-3 cnt_bias -1.2 to -1.9 (the phase skips packed sheets), HOLD -0.05, 1667 -0.10; its density is unbiased.
     Rows written before these columns existed get them with `addcnt.sh RES.tsv` (reads runs/TAG/BOX/cmp.log).
5. **Ceiling.** The same fitter on label phase (with network planes where there are no labels) gives `ceil` and `ceil_p1x`.
   It is cached per fitter config (binary hash + FIT flags) in `$WB/ceil/`. `frac` = ids / ceil. With `--set clean` (the
   default), `wbench-stats` drops boxes with ceil < 0.8 (the labels, not the model, limit them: all 0814 boxes so far) from the mean and prints them on a separate "label-noise" line.
6. **Slabs** (`SLABS=1`, the default). Run them for every fitter arm and every milestone student; routine snapshot evals
   use `SLABS=0`. Four Paris 4 full-cross-section sites at z cells 4624, 5392, 5904 and 6672. Each site is fitted as two
   96-cell slabs with a 32-cell overlap (the production slab size), then joined with `fit-join`. The run reports:
   - per-slab ids and pair accuracy;
   - the **join share** (how far adjacent slabs agree);
   - the joined site score;
   - the radius profile (`TAG-radius.tsv`: integer error fractions by radius, which show radial drift).

   **The slabs are in-sample for every Paris 4-trained model.** They measure whole-scroll behaviour, not generalization.

Rows go to `$WB/res/TAG.tsv`, `TAG-slabs.tsv` and `TAG-radius.tsv`; logs go to `$WB/runs/TAG/`. A rerun skips boxes that
already have rows, so delete the rows (or use a new TAG) to redo them. The run takes the site GPU lock with flock (desk:
`$WB/gpu$GPU.lock`).

## Time per checkpoint

Measured on the wind96 baseline (2026-10-25):

| part | laptop 5080 | desk 5060 Ti (GPU 0, CPU shared with B) |
|---|---|---|
| 45 boxes, first run (predict + 3 fits + phase compare + ceiling) | ~50 min | ~55 min |
| 45 boxes, fixed fields or cached ceilings | ~35 min | ~40 min |
| 4 slab sites | 22-40 min (6-12 min per site) | ~25 min |

So one checkpoint takes ~1-1.5 h with slabs; `SLABS=0` brings it under 1 h. Per box: predict 6-12 s, one fit 4-13 s
(4.5 s of it is pair scoring on the Paris 4 boxes), the ceiling ~25 s (first run only). Laptop and desk agree per box (same fields,
same fitter): HOLD 0.9253 on both, the old desk eval 0.9255.

## Statistics

`ufsm wbench-stats A.tsv [B.tsv] --metric ids,p1x --set clean|all --n 10000`

- It averages replicate rows per box.
- Bootstrap: boxes are resampled within each scroll, so every scroll keeps its box count.
- It prints:
  - A's mean with a 95 % CI;
  - with B: the paired difference A - B and its CI;
  - P(A > B);
  - the MDE (2.8 x the bootstrap SE of the paired difference);
  - per-scroll means (`--boxes 1` adds per-box values).
- **Judge per scroll: the paris4 and paris4j lines carry the phase-skip failure (wind96 cnt_off 0.65 / 0.56 vs 1667 0.11); the
  pooled mean is mostly 1667.**
- **A difference smaller than the MDE is noise. The headline is p05x (lowest paired variance: MDE 0.046 vs 0.083 for ids on
  22 clean boxes), with ids beside it, over the clean set.**

## Whole-scroll Paris 4 unwrap (the milestone; scheduled by the coordinator, about 9 h on one desk GPU)

On the desk, in /vesuvius/ufsm/paris4 (scripts there). These are the steps that produced the wind96 whole-scroll result
(joined ids 0.112; the label-phase oracle gets 0.917):

1. `run_pred.sh`: whole-scroll fit fields at 9.6 um (about 1.4 h). The checkpoint, output dir and GPU are set at the top; use
   the same predict flags as the bench.
2. `run_fit.sh TAG`: 146 slabs of 64 interior cells + 16 overlap, `fit --dir 1 --uw-repair 8` with codes scoring (about 7.5 h).
3. `post_slabs.sh TAG`:
   - z x radius slip map;
   - fit-join of all slabs;
   - the whole-scroll codes score (`slabs-TAG/joined-score.log`);
   - wraps and wrapswitch per wrap.

Compare the joined ids, the slip map and the wrapswitch "right" share with `slabs-base`.
