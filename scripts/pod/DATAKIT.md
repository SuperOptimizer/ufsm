# Pod data kit (2026-11-11)

Goal: run the latent winding trainer and the winding bench on a rented pod without uploading label data from the desk (desk uplink
~4-5 MB/s; dl.ash2txt.org and the open-data bucket download at 20+ MB/s). `scripts/pod/datakit.sh K` rebuilds the labels from the
published segment meshes; CT is read over HTTP from the volcomp mirror.

## What is public, what is copied

| input | source | size / time on pod 2 |
|---|---|---|
| CT (Paris 4 level 2, 0139, 0500P2, 1667) | volcomp mirror `<scroll>/volumes/<scan>` over HTTP + local `--cache` | sampled shards byte-identical to the desk's `v2/ct` (md5 of Paris 4 `2/c/11/5/4` and 0139 `0/c/8/4/5`) |
| published segments (Paris 4 June / July, 0139, 0500P2, 1667) | open-data bucket, `fetchsegs.sh` (16 parallel curls) | Paris 4 39 GB, 0139 6.8 GB, 1667 6.3 GB, 0500P2 0.4 GB; ~30 min together |
| PHerc0500P2 phase (9.362, level 0) | `scripts/cal/gt.sh` NATIVE=1 | 14 / 14 shards byte-identical to desk `v2/rv/gt/p0500p2full/phase` |
| PHerc0139 phase (9.362, level 0) | `k0139.sh` from the q-assigned surfaces qA / qB | 91 / 91 sampled shards byte-identical to desk `v2/rv/gt/p0139full/phase`; 17,035,432,659 known voxels both |
| Paris 4 codes A / B (9.6) | `p4kit.sh` (= `scripts/wbench/gtjuly/build.sh`) | 85 / 85 and 123 / 123 sampled shards byte-identical to desk `wbench/gtjuly/codesA`, `codesB`; June seed 6 min, rasters 10 / 14 min, July assign 8 min at T=4 |
| Paris 4 phase A (9.6) | `p4kit.sh` phase-raster with the p4full flags (gs 4, codes grid, radius 160, the p4full box) | the first 316 of 3549 shards byte-identical to desk `v2/rv/gt/p4full/phaseA` (~8 shards / min at T=4: ~7 h for all) |
| wbench Paris 4 codes q-9p6 | 4.8 um raster halved (max) | NOT identical to desk `gt/paris4-q9p6`: the same sheets one turn (18 codes) apart, plus some voxels labelled on one side only (Spearman 0.94 on a shard). The 10-19 original came from an older wind-seed gauge. The fit score takes the gauge as the mode of U - q, so the offset is harmless, but per-voxel coverage differs: for numbers comparable with the desk's tables, copy the desk q-9p6 shards of the boxes |
| copied from the desk (205 MB) | axis / umbilicus JSONs, `klt1.lcb`, `wind96-wk1-best.ckpt`, `boxes.tsv`, `holdouts.json`; and `gt/p0139-q` (55 MB) | the 0139 q-surfaces come from the 10-16 wind-seed / wind-assign, which predates the 10-19 sense rule, so a fresh seed is not guaranteed to reproduce them |

## Run notes (pod 2: RTX PRO 4000 24 GB, ~27 CPUs, **31 GB cgroup memory**, not the 125 GB `free` reports)

- The Paris 4 rasters (28 June / 28 July meshes over 75784 x 32693 x 32693) were OOM-killed at T=10 beside a 10 GB fibre trace and a
  12 GB trainer; T=4 fits. Everything built on the incomplete codes (reference, July assign, phase) was redone.
- Latent winding smoke (`wzsmoke.sh`: the wzlo2 recipe, wind96 warm start, `--init-lr-mult 0.2`; sources 0139 + 0500P2 while Paris 4
  rebuilt): at P 896 the trainer took 12.6 GB host RSS (stopped for memory); at P 512, 4 workers: 330-363 Mvox/s, loader wait 35-40 %
  (CPU-bound beside the raster and the trace), 9,959,904 of 10,057,104 parameters inherited.
- Check `/sys/fs/cgroup/memory.max` and `cpu.max` on a new pod; `nproc` and `free` show the host.

## Disk

Paris 4 phase A + B are 86 + 57 GB on the desk, codes 28 GB, q-9p6 10 GB, the segments 39 GB (delete after the codes), the 4.8 um codes
behind q-9p6 43 GB (delete after q-9p6). With other agents' caches on the 250 GB pod, phase B did not fit; a pod for the full Paris 4
set needs >= 400 GB, or train on June (A) only.

Pod 2 state (2026-11-11, after the disk filled to 100 % beside the 4.8 um relabel): `gt/paris4/codesB` (17 GB, the July codes) and
`gt/p0139full/phaseA` (8.5 GB, the k0139 intermediate; `p0139full/phase` is fused and kept) are deleted, as are `seg/PHerc0139`,
`seg/PHerc1667`, `seg/PHerc0500P2` (public, refetch with fetchsegs.sh). Phase B was never built, so `sources-wind-gt-pod.json` has
June, 0139 and 0500P2 only (mksources.sh adds July only when `gt/paris4/phaseB` exists). Rebuilding July needs `p4kit.sh` from the
codes step (~14 min at T=4) and ~60 GB more disk.

Pods 1 and 2 terminated (2026-11-11 evening; replaced by one RTX PRO 6000 96 GB pod, 500 GB disk). Saved on the desk: fibhv-b / c / d
runs (all snapshots, logs, sources) under `/vesuvius/ufsm/fibers/runs`; pod 2's fibre work under `/vesuvius/ufsm/fibers/pod2`
(fibp4 complete, fibd evals, the pod scripts and logs); the B1 Paris 4 4.8 phase, partial (below). **Lost, rebuild on the new pod:**
- the kit itself (`datakit.sh`: segments, seeds, codes, phase), and all fibre label rasters (fibhv-d's 79 GB: `scripts/fibers/mklabels3.sh`
  from the .afv files, which are public downloads);
- the Paris 4 official-prediction trace `trp4` (`trp4b.sh`, 192-slice chunks): only 11 of its 50 chunks (z 0..2111) had been written
  before pod 2's disk filled (c2304..c9408 failed with ENOSPC); those 11 are on the desk in `fibers/pod2/trp4`. Rerun the rest with
  `fiber-trace --gpu` when the fibre teacher needs it;
- the Paris 4 June 4.8 um codes (`codesA48`, 43 GB, `raster --level 1`; scripts/univ/b1_pod_p4A.sh up to the codes step).
- Stored 4.8 um phase sets are retired (the sampler computes the phase from the codes on the fly, univA): the Paris 4 June set stays at
  316 of its 549 packed 9.6 shards complete on the desk; mksources_univ.py holds out the other 233.

## Not covered

- PHerc1667 GT (only the wbench 1667 boxes need it; its 10-16 codes at 4.798 um and phase came from a pre-10-19 pipeline). Copy the
  box shards instead (12 core boxes: 462 code shards 1.5 GB + 128 phase shards 3.7 GB), or run the bench on Paris 4 / 0139 boxes only.
- PHerc0814 (never a training source).
