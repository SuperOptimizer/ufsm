# Plan: end-to-end redesign for fully automated, near-perfect scroll unwrapping

## Context
**Goal:** every sheet of every scroll found, numbered with the right turn, meshed, flattened and rendered, with no human in the loop.

**Pass rule (user, 11-05):**
- Whole-scroll GT score **≥ .95** on Paris 4, PHerc1667 and PHerc0139: the share of GT sheet voxels on the right sheet with the right turn.
- GT-free checks clean on all 39 scrolls:
  - CT sheets per fitted turn 1.00 ± .02
  - no duplicated or merged wraps
  - on-papyrus ≥ .9

**Where we are:**
- **Whole-scroll scores are ~.21** on P4, ~.4 on 1667 slabs, and ~.2 on 0139 (4 mm pairs).
- **Running our fitter on the GT phase reaches .92.** So the numbering machinery can work, but its input, the wind96 phase field, is wrong in packed regions.
  - It skips sheets (cnt_bias −1.2..−1.9 on PH1–3).
  - Each skip shifts the turn count of everything outside it.
- **Our own spirals are not trustworthy teachers yet.**
  - Pairs within 0.5 mm: .73 on 1667 and .61 on 0139 (wind96 itself gets .83 and .86).
  - Recto/verso side agreement is only .69.
  - The CT-coherence gate only beats wind96 at 7.5% kept.
- **The current pipeline asks one field to do two jobs:**
  - (A) find each sheet (geometry);
  - (B) give each sheet its turn number (topology).

**Redesign:** make **surfaces** the primary object.
1. Fuse several independent sheet detectors.
2. Trace explicit sheet patches.
3. Solve turn numbers per patch as a constrained global problem, with our phase, the count-field reference and human segments as votes and anchors.
4. Then mesh, flatten and render.
5. Close the loop by retraining the detectors on multi-signal-agreed labels.

### Assets we already have (reuse, do not rebuild)

**Detectors**
- **m7 recto surface net:** CUDA port with parity |d| .002 (`src/surfrecto.c`, `surfrecto-predict`). m7 predictions are on the mirror for every scroll except Paris3.
- **wind96:** phase, normal and density heads.
- **Papyrus/void net:** IoU .98–.99 (`train --task papyrus`).
- **rv side model:** `v2/rv/runs/rv1`, 5k–35k snapshots, **never scored**.
- **4× CT super-resolution:** `train --task restore`, +2.3 dB.
- **Higher-resolution scans:** ≤2.4 µm for 17 scrolls, 1.13 µm for 0139, 0814, 1667 and P4.

**Sheet, side and patch tools**
- `patch-id` (sheet-mask atoms → union-find patches, `--gt` purity; `src/patchid.c`)
- `side-face`, `side-gate` (`src/sideface.c`, `src/sidegate.c`)
- `wrap-snap`, `refine`, `self-gate` (`src/wrapsnap.c`, `src/surf.c`)
- `stitch` (`src/stitch.c`)

**Numbering**
- `count-field` (`src/countfield.c`)
- `fit` (L1 integer shift solver `l1_vote`, slip repair; `src/fit.c`)
- `fit-join --ref` (`src/fitjoin.c`)
- `wind-assign`, `wind-seed`, `codes-merge`, `raster`

**Labels and evaluation**
- `surf-labels` (strip input, `--gt`, `--min-coh`, `--agree`, `--anchor`, `--weight-coh`)
- `wrap-anchor`, `recto-ruler`, `wrap-check`, `geom-check`, `wbench` + `wbench-stats`

**Training**
- `train --task unwrap2`, with scope local and per-voxel phase weight (branch `v2strip-20261105`)

**Post-processing**
- `wraps`, `strip`, `strip-flatten`, `strip-mosaic`, `render`, ink models
- `scripts/unroll/unroll.sh`

**GT**
- **Winding codes:** P4 (June 28 + July 56 segments, 9.6 µm), 1667 (21 segments), 0139 (39 segments, 18.7 µm), 0814 (noisy), 0500P2 (sparse).
- **Bench boxes:** frozen (`wbench/boxes.tsv`, `holdouts.json`).
- **Not found on disk:** the PHerc0172 labels; recover them from the mirror.

### Gaps (new work)
- a surface tracer
- a patch graph + global turn solver
- sheet-level metrics: GT sheet recall, sheet switches and merges
- a fused evidence volume
- a v3 driver
- an evaluation of rv1

## Target architecture (unroll v3)
```
CT (native + finest scan, optional SR)
 └─► L1 evidence: m7 recto | rv recto/verso | papyrus | wind96 phase+normal+density | CT itself      (per scan, cached zarrs)
      └─► L2 fused sheet field: P(recto face), P(verso face), normal, sheet-centre distance, confidence
           └─► L3 tracing: seeds → grown sheet patches (ported villa/VC3D grower → own grower) ─► patch set + adjacency
                └─► L4 topology: patch graph (same-sheet links, radial-order links, crossings forbidden)
                     │   priors: count-field ref, wind96 phase W, density counts, fibres; anchors: human segments
                     └─► integer turn per patch (L1/robust solve, reusing fit.c l1_vote) + conflict report
                          └─► L5 surfaces: per-turn meshes from patches + gap fill (wrap-snap/refine) → spiral strip
                               └─► L6 flatten/render/ink (existing) → L7 QA scorecard (GT-free) → outputs
learning loop: L4/L5 outputs where ≥ 3 signals agree → labels (surf-labels, scope local unless anchored) → retrain L1 nets
```

## Phases (each with a go/no-go before the next spends GPU time)

### P0: evaluation foundation (week 1; CPU)
- Freeze the contract in `wbench/README` + `holdouts.json`, as now.
- Add **sheet metrics** to a new `ufsm sheet-score <surfaces> --gt CODES` (reuse the `surf-labels --gt` and pairacc code):
  - GT sheet-area recall (within 1 voxel)
  - precision (our area on a GT sheet)
  - switches (a patch spanning two GT sheets)
  - merges and duplicates
  - turn-id accuracy
- Whole-scroll GT scoring already exists (joined score; `eval.sh p4`). Add the same for 1667 and 0139.
- **Baseline table:** wind96 pipeline on boxes, slabs and the three whole scrolls. Every later phase reports against it.

### P1: evidence audit (week 1–2; ~1 GPU-day)
Score every detector on the same GT (P4 incl. PH1–3 packed, 1667, 0139), separately for packed (< 100 µm spacing) and open regions:
- **sheet recall / precision** for each: m7 prob, `surfrecto-predict` on the finest scan, wind96 phase level sets, papyrus net, CT peaks;
- **side accuracy:** rv1 snapshots (`predict --channel rv`), m7 + side-face;
- **turn counts per GT interval:** `wrap-check` source columns already do this, e.g. `v2/trace/rc/p4_m7.log`.

Also:
- Resolution test: does a 2.4 µm or SR input recover packed gaps (gap count .82 → ~1.0)?
- Finish the `surf-labels --gt` calibrations, adding `--agree m7`.

**Go:** pick the evidence set for L2. **Kill:** detectors that add nothing in packed regions.

### P2: fused sheet field (week 2–3)
- New `ufsm sheet-fuse` (`src/sheetfuse.c`): per voxel, P(recto), P(verso), the normal, signed distance to the sheet centre, and a confidence.
  - Start rule-based: calibrated logistic fusion fit on GT boxes.
  - Later: a small learned fusion head (`train --task fuse`), with the detectors' outputs as inputs.
- Output on the training grid, chunked like the fields; predict per GPU / per pod.
- **Exit:** in packed regions, sheet recall and precision both beat the best single detector by ≥ 5 points.

### P3: tracer (week 3–6). Port first, own successor second (user's choice)
- **P3a port:** the villa/VC3D patch-growing tracer.
  - Seeds on high-confidence recto voxels.
  - Quad-mesh growth along the predicted surface, with straightness, spacing and normal constraints.
  - Rewrite it in C (`src/trace.c`, `ufsm trace`). It reads the L2 field or plain m7. It writes tifxyz/.sfc patches plus an adjacency list.
  - Parity test against the reference on a crop, as was done for m7.
- **P3b own tracer** (`src/trace2.c`):
  - **side-aware growth:** a front never crosses from a recto face to the next sheet's verso face. This is the packed-sheet separator.
  - stops at low confidence, so patches stay pure rather than large;
  - GPU candidate scoring.
- **Metric:** `sheet-score` on boxes (PH1–3, HOLD, J-boxes, 1667 C-boxes, 0139):
  - recall ≥ .9 of GT sheet area;
  - switch rate < 0.5% of patch area;
  - P3b replaces P3a only if better on both.
- **Go/no-go:** the traced patches must find packed sheets that the wind96 phase misses. Check this on PH1–3 with the cnt_bias diagnostics.

### P4: patch graph + turn solver (week 5–8)
New `ufsm patch-solve` (`src/patchsolve.c`). It reuses the `fit.c` `l1_vote` integer solver and the `patch-id` union-find.

**Nodes:** patches (from the tracer, or `patch-id` atoms as a fallback).

**Edges:**
- **same-sheet:** shared or adjacent boundary, normals agree → Δturn 0 (hard when confident);
- **radial:** along rays from the axis (axis file or `ufsm axis`), consecutive patch crossings → Δturn +1. Weighted, because a missed sheet between them is possible. Use the L2 field and the CT-peak count between the two crossings to decide +1 versus +2.
- **crossing-forbidden:** two patches in one voxel column at the same turn = a conflict.

**Priors and anchors:**
- soft unary votes from the count-field reference (`count-field --model`), wind96 W, density integrals and fibres (`--fibers` AFV);
- hard anchors from human segments (`wrap-anchor` / `wind-assign` codes), where present and not held out.

**Output:** turn per patch, a conflict list (patches dropped or split), and the per-voxel turn field via the existing `raster` / `stitch field`.

**Metric:** box ids and pairs (`wbench`) and whole-scroll GT score.
- **Exit 1:** beat the wind96 pipeline on the pooled bench by ≥ the MDE.
- **Exit 2:** whole-P4 ≥ .5.

### P5: whole-scroll scale + surfaces (week 7–10)
- Tile the tracer and solver over the scroll. Patches cross tile seams by overlap merge (reuse `stitch align` logic). Run one global solve per scroll (patches ~1e5–1e6; sparse L1).
- **Per-turn surfaces:** union of each turn's patches, holes filled by interpolation along the turn, then `wrap-snap` / `refine`. Then the existing `strip` → `strip-flatten` → `render` / ink.
- **Driver:** `scripts/unroll/unroll3.sh`. Same stage/resume/STOP/DRY conventions as `unroll.sh`: stages ev → fuse → trace → solve → surf → strip → flat → img → qa. Runs on pods per scroll.
- **Exit:** whole-scroll GT ≥ .8 on P4, 1667 and 0139; GT-free scorecard better than v2 on ≥ 30 of 39 scrolls.

### P6: learning loop (week 9+; the earlier self-training plan, re-based)
- Harvest labels only where the v3 surfaces agree with ≥ 3 independent signals: m7 recto, rv side, CT peak, wind96 phase. Use `surf-labels` `--agree`, `--min-coh`, `--weight-coh`.
- Train with **scope local**, except anchored → global. Confidence-weighted (implemented, branch `v2strip-20261105`).
- **Retrain the detectors:**
  - rv side model (recto/verso from traced patches);
  - a wind96 successor (phase/normal/density);
  - optionally an m7-style recto net at native resolution.
- Holdouts never harvested: the 1667, 0139, 1299 and 0846A whole scrolls, plus the bench boxes. GT keeps a fixed share.
- **Accept rule** (from the earlier plan): pooled bench p05x ≥ MDE with P ≥ .95, and no scroll line worse than 2 × MDE.
- **Rounds:**
  - pods re-run `unroll3` with the new detectors;
  - stop after 2 rounds without gain;
  - ask before each pod spend.

### P7: hardening to "done" (ongoing)
- **All 39 scrolls through `unroll3`:** the QA scorecard flags defects, and defect regions get a second pass.
  - Second pass: finest scan + SR, stricter tracer, more anchors.
- **Readability check:** ink continuity across turns (render + ink-compare).
- **Pass rule:** ≥ .95 whole-scroll on the three GT scrolls, clean scorecards elsewhere. Push past .95 by targeted fixes from the conflict lists.
- **Uploads stay manual**; prize scrolls private.

## Parallel tracks & compute
- **Desk GPU 0:** P1/P6 training. **Desk GPU 1:** evals.
- **CPU:** labels, solver, metrics.
- **Pods:** whole-scroll evidence + `unroll3` runs, with the user's OK on count and spend each time.
- **Disk:** /vesuvius 366 GB free, /mnt/scratch 1 TB free. Fused fields are ~1 B/voxel per channel at 9.6 µm, so keep per-scroll caches on scratch and delete them after the QA stage.

## Critical files
- **New:**
  - `src/sheetscore.c`, `src/sheetfuse.c`, `src/trace.c` (port), `src/trace2.c` (own), `src/patchsolve.c`
  - their tests in `tests/` + `tests/suite.txt`
  - `scripts/unroll/unroll3.sh`
- **Extend:**
  - `src/patchid.c` (atoms as solver fallback, `--gt` purity)
  - `src/fit.c` (expose `l1_vote` as a library function)
  - `src/surflabels.c` (multi-signal agreement gate)
  - `src/train.c` / `src/sample.c` (rv / fuse tasks as needed)
  - `scripts/v2/mksources.sh`, `scripts/wbench/wbench.sh` (sheet metrics columns)
- **Reuse unchanged:**
  - `src/surfrecto.c`, `src/countfield.c`, `src/fitjoin.c`, `src/wrapsnap.c`, `src/stitch.c`
  - strip / flatten / render, `recto-ruler`, `wrap-check`

## Verification
1. **Unit tests per new tool** (synthetic spirals as in `tests/test_surflabels.c`):
   - tracer: recovers each sheet without switches;
   - solver: recovers the turns with a missing-sheet case and a mis-anchored segment;
   - fusion: monotone in its inputs.
   The fast suite stays green.
2. **Phase gates** use `sheet-score` and `wbench-stats` against the P0 baseline, recorded in memory and `DESIGN.md` per phase.
3. **Whole-scroll:** P4, 1667 and 0139 joined GT scores, the slip map and wrap-check per run. Renders sent at ≤ 8000 px.
4. **Final:** the pass rule above on the three GT scrolls, plus the scorecard on all 39.
