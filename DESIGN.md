# ufsm — design

Ultra fast scroll model: a C23 + CUDA pipeline that trains a ~1 M-parameter 3-D convnet to predict the
recto papyrus face in Vesuvius Challenge micro-CT, from scratch, using only the released ground truth:
the gated HuggingFace `scrollprize/datasets` bucket and the AWS `vesuvius-challenge-open-data` bucket.
No teacher models, no distillation. All upstream data is re-exported once into the user's own codecs
(volcomp volumes, surfcomp surfaces) so training reads one compact local store.

## Memory and inference defaults (2026-10-08)

- Training stores the MX activation gradients as MX-fp4 with exact stochastic rounding by default when every width is a
  multiple of 32 and P is divisible by 2^levels (`UFSM_GRAD_MX4=0` keeps MX-fp8). Desk mode (lean 2, chunk 2,
  recompute 1) at 512^3: 177.7 -> 139.8 B per level-0 voxel (-2.9 GB per GPU of the split slab), same step time
  (laptop P 384: 985 vs 984 ms of kernels). Paired 500-step runs at P 256 (two seeds): validation loss +0.1..0.8%.
  Against 16-bit gradients over 16 rounding seeds the fp4 parameter gradients are unbiased; MX-fp8's
  round-to-nearest stores are not (mean residual 0.14). A 16 GB GPU now trains a whole 512^3 window (0.43 samples/s).
- The decoder conv1 weight gradient runs over 32-channel up slices, so gout[0] holds one slice; with fp4 gradients the
  fp16 logits sit at the start of gout[0], the logit gradient in A, the batch at gout[0]'s end.
- Shared encoder a1 (`UFSM_RC_ENC_A1=1`, -1.66 GB, ~2.3% slower) stays a planner choice for when lean 2 does not fit.
- predict computes only the written logit channel and keeps its 16-bit input in the network's shared a1 scratch
  (inference 80.5 -> ~57 B per window voxel). Boxes of whole shards default to one window per shard (shard + 32,
  halo 16) when it fits the GPU, else 288 / halo 8: laptop, 8 shards of 512, 16% less GPU time (5.20 vs 6.21 s),
  9.4 GB peak, and twice the halo. `--window` / `--halo` override.

## Continuous winding experiment (2026-10-03)

The experimental surface/winding pipeline is implemented without changing the
1,172,050-parameter backbone. It adds sparse coordinate, continuity, relative
ordering, path-support and gap losses; floating winding output; constrained sheet
reconstruction; and matched development/final-test evaluation. See
[SHEET_PIPELINE.md](SHEET_PIPELINE.md) for commands, precision contracts and limits.
The complete selected Paris4 geometry dataset is built and four 2,000-update
ablations are prepared. Small native single-/split-GPU integration tests pass;
large-window capacity, throughput and quality remain unmeasured for this task.
The existing selected-surface coverage production run continues with its frozen
binary. Its recipe is not replaced by the experimental winding task.

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
