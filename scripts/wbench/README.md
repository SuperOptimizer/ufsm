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

## wgate: the decision for a winding checkpoint (use this to promote / drop arms)

    # laptop: one GPU block (gpuq priority eval), ~30 min; deployed copy in $WB/bin/wb8
    ~/ufsm-data/capped.sh bash $WB/bin/wb8/wgate.sh CKPT TAG        # -> $WB/gate/TAG/table.txt
    bash $WB/bin/wb8/wgate-table.sh TAG                            # the table again (or ROWS=$WB/res/old.tsv for old wbench rows)

**What it measures.** The 44 core boxes and the 4 Paris 4 slab sites, against wind96. The wind96 rows and ceilings are cached
and never re-run: fits are deterministic and identical across binaries, and wind96 predicted with a new build reproduces its old ids.

| set | boxes | role |
|---|---|---|
| primary | clean paris4 + paris4j: PH1-3, HOLD, 28 July boxes | the decision |
| guard | 12 PHerc1667 core boxes | must not get worse |
| slabs | sites 4624 / 5392 / 5904 / 6672 | joined ids: the direct scale-failure metric (oracle ceiling .93-.98, wind96 .21-.33) |

The paris4j boxes are held out of every current arm. PH1-3 are in-sample for wind96, which biases the primary set toward wind96
(conservative). Never gate on PH1-3 / HOLD / C9 alone: that 5-box read was selected on wind96-best and failed wcomb1, which ties on the full set.

**The table.**
- Paired per-box differences with stratified bootstrap 95 % CIs, P(A > B) and the MDE (`ufsm wbench-stats --scroll .. --tsv 1`).
- p05x is the headline (lowest paired variance), with ids and p1x beside it.
- A paris4j-only line.
- Per-site slab joined ids, join share and the share of the oracle gap closed.
- The radius profile (share of slab voxels with the right winding, by radius band).
- Diagnostic turn counts (cnt_off, cnt_bias): these take no part in the rule. cnt_off improved for pg1 while its fits got worse.

**PASS rule (pre-registered 2026-10-26; do not change it after seeing a result):**
- **R1 primary:** the p05x paired-difference 95 % CI is entirely > 0, AND the ids difference is >= 0.
- **R2 slabs:** the mean joined ids over all 4 sites is > wind96's.
- **R3 guard 1667:** not worse beyond its CI: the upper CI bound of the p05x and of the ids difference are both >= 0.

PASS = R1 and R2 and R3, which promotes the checkpoint to the p4unwrap milestone. The other verdicts:
- FAIL: R3 fails, or the primary p05x CI is entirely < 0, or there is no gain.
- INCONCLUSIVE: the p05x difference is positive but inside the noise. There is no promotion; gate a later or averaged checkpoint.

Typical MDE on the primary set: p05x ~0.03, ids ~0.05 (one slip moves a box's ids by ~0.3).

**Fair checkpoints.** Gate only a cooled snapshot (the end of a schedule's cooldown) or a weight average of the last snapshots
of the run. Never gate a single peak-LR snapshot. The trainer keeps EMA weights (`--ema 0.999`, read by predict `--ema 1`), but
at peak LR that is only a ~1k-step average.

    ufsm ckpt-avg run-avg.ckpt snap-6000.ckpt snap-8000.ckpt   # uniform average of the EMA + live weights; optimiser state of the last

**Speed.** One GPU block (gpuq eval). The GPU lane only predicts: the 4 sites first (~2 min each), then the boxes (~11 s each).
- 2 box lanes fit each box as soon as its fields land. The box fit and its 2 shifted fits run concurrently, followed by the
  CPU-only fields-oracle and fit-compare (diagnostic columns; `DIAG=0` skips them).
- 1 slab lane fits each site as its fields land. It is the critical path.
- There are no ceiling fits: the clean set uses wind96's ceilings.
- Fields are kept in `$WB/fields/TAG`, and a rerun skips everything that already has rows.

Env: `REF` (wind96), `BOXES`, `SITES` ("" = boxes only, ~10 min), `FL` (box lanes), `UP` (predict binary, default
`$WB/bin/ufsm-gate`), `U` (fitter pin).

## gpuq: the laptop GPU queue (`scripts/gpuq.sh`; deployed copy `~/ufsm-data/gpuq.sh`)

The GPU is still one exclusive flock on `~/ufsm-data/gpu.lock`, and gpuq decides who takes it next. Priorities:

| priority | for | behaviour |
|---|---|---|
| `now` (0) | short checks: pcheck, one quick test | runs at the next box boundary of any bg job |
| `eval` (1) | suites: wgate, a fulltest GPU block | one block, never yields |
| `bg` (2) | long loops: wbench core benches, slab ceilings, p4unwrap steps | yields at every box boundary, round robin among bg jobs |

Order is FIFO inside a priority, and tickets of dead processes are dropped.

    ~/ufsm-data/gpuq.sh run now mycheck cmd args...     # instead of: flock ~/ufsm-data/gpu.lock cmd args...
    ~/ufsm-data/gpuq.sh status                          # holder + waiting tickets
    . ~/ufsm-data/gpuq.sh; gpuq_acquire bg mytag; for b in ...; do gpuq_yield; work $b; done; gpuq_release

- wbench.sh (PRIO bg by default), pcheck.sh (now) and p4unwrap.sh (bg) use it.
- Commands started under a held lock inherit it, so a nested acquire returns at once.
- Old scripts that call plain `flock gpu.lock` still exclude and are excluded, but they have no priority. Convert them with `gpuq.sh run`.

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

## Whole-Paris-4 unwrap milestone (p4unwrap.sh)
`SITE=desk GPUS="0" bash scripts/wbench/p4unwrap.sh CKPT TAG` = the 10-21 run_pred + run_fit + post_slabs for any checkpoint:
whole-scroll fit fields, 147 full-section slab fits (64-cell interiors + 16 overlap, `--uw-repair 8`), fit-join, whole-scroll
`--codes` score -> `$WB/p4unwrap/TAG/summary.txt` (joined ids vs wind96's 0.1117, join share, pair accuracy, radius rows).
Cost on one 5060 Ti (wind96): predict 1.4 h + fits 4.7 GPU-h + join/score ~1.6 h; ~21 GB host per slab fit, 50 GB fields.
`GPUS="0 1"` splits the slabs over two lanes; `FIELDS=dir` skips the predict; `WRAPS=1` adds wraps + wrapswitch; restart-safe, `touch $O/STOP`.
Only for milestone candidates that won BOXES=core on paris4 + paris4j.
