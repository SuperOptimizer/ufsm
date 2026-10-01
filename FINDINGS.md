# ufsm audit findings

## Adversarial re-review — 2026-10-01

This update is a four-reviewer, source-only audit: a coordinating reviewer and three agents independently examined model/backward/checkpoint correctness, low-precision CUDA/dispatch/performance, and sampling/storage/prediction/evaluation. **No code was executed in this re-review:** no builds, compiler probes, tests, benchmarks, training, inference, project scripts, or reproducer programs. Read-only source inspection and documentation edits were the only audit actions. Numerical counterexamples below are algebraic, unexecuted examples. No implementation files were changed by the reviewers.

The earlier F001–F106 entries are retained below as historical evidence. Their original priorities and line references do not make them current defects. Use the status register in this update before acting on an old entry. New findings start at F107; closely related historical defects are updated in the register rather than renumbered as new discoveries.

### Source cuts and confidence

The primary cut is Git commit **`6b0390a963043c3eef345766ba52b2a0f32325a6`**, captured in `/tmp/ufsm-review-20261001-WhnCIo` while the working tree was clean. It covers all committed work through prediction precision selection, fine-tuning, and the new source configurations. The secondary cut, `/tmp/ufsm-review-latest-20261001-m1hfpC`, captures intermediate sampler/storage edits and new sampler/read benchmarks. The final cut is commit **`51125cd5e32f598c4ab5dcf967585db19b89fb1d`**, captured in `/tmp/ufsm-review-final-20261001-ten1DI`; all three agents reviewed the relevant delta, including Muon, WSD, later prefetch correction, and profiling/float preprocessing.

The closing committed source delta through **`c64dab9fb14d2cd0444480d83a2aeadf41a2dde3`** is captured in `/tmp/ufsm-review-close-20261001-s05xyu`. It adds source minimum-level filtering, source-generation/configuration changes, and cache-backed-only I/O counters, plus reported experiment notes. A final uncommitted batched-Muon delta based on **`6fd40ac0f71e2374fecaeebc15ad072e92808ff8`** is captured in `/tmp/ufsm-review-batched-20261001-8XXlTL`; the intervening commit changes experiment notes only. Both model and CUDA agents reviewed that final source delta. These changes do not correct the core model/optimizer/norm defects. The update adds **48 entries, F107–F154**, including one explicitly resolved intermediate defect (F128); historical F001–F106 remain with a current-status register.

Source references without a cut label in F107–F142 use primary line numbers; their workspace links are navigation aids, and later edits can shift those lines. Use the frozen primary tree for exact evidence. **Secondary** explicitly identifies the intermediate tree. F143 onward and final-status annotations use the final tree. This is a review of these captured sources, not a guarantee about later edits. In particular, F128's sparse-bitmap hang was fixed after the intermediate review and is retained only as a resolved intermediate finding.

| Primary file | SHA-256 |
| --- | --- |
| `src/nn.cu` | `e25fd0d67cd933eb187714d3e73a7874337ed53f15cc230acf24af6d80759188` |
| `src/nn_fp8.cu` | `013f22f4d2948786e36b2f96ca9ecc0656c44e8419d0c5614badf2b61f5991c4` |
| `src/unet.c` | `ab906cb08414db38fd16ce802b5249a3ea119c03eb8c97eceb8383ce3f8a6f27` |
| `src/train.c` | `573d7400383a97713774c8602123f532feb14f54703ef3394efeb228ba70cfaf` |
| `src/predict.c` | `4e83c6f31ca188ba23e21b0ae671c5843d2b610681300cb9de350df7d5d85bab` |
| `src/sample.c` | `5117ed0a6ddf33c192e968413949aa5ba118725b0e33637f15c7e4e3f8582ec6` |
| `src/sources.c` | `9e56cf0e28b4447fad9b73606eb75f575b98726fc271b53ce08d9ccd757983b2` |
| `Makefile` | `4dd330e1e1720c41cf5f0c05e849e8863d7e99243ec473ef4b93b99ee108f706` |

| Changed final-cut file | SHA-256 |
| --- | --- |
| `src/nn.cu` | `e34128b23c6fe317b18637d51a0ea3280b4c4ef1561aba33d20fec00e897565b` |
| `src/unet.c` | `d4c94034c28850faa45eb91be2610c15ebb3c4c12c68991457f7402baacd8323` |
| `src/train.c` | `2eceafd66b18bd8ec9562819a4acc2ba0a7ec8417d3b29bceef2dd1b5f6e4581` |
| `src/sample.c` | `7f49605b4b116eec7a4856051f70105b82717cc3cb0a960150f14f962da710b8` |
| `src/zarr3.c` | `5dffee5fe17a040cc3e5e3eaf742b133a3a712f835a31cb1256e0e0f30f05005` |
| `tests/test_muon.c` | `1f8e282e297f948184f6f998ded2977d92375056d59d5b64272fae87d8d9a551` |
| `Makefile` | `e7dedeb34bed85e6e27e4bbc4fef18183d14a3b954afba43450fc18a30416d14` |

Final batched-Muon snapshot fingerprints: `src/nn.cu` = `e9d29999389eaee05af2f565bb1adab8c36308f226ef0b95a7a4d6e4c487771d`; `src/nn.h` = `7c864e851cea23b79ba72794ca0659b376133f6fa11238f680d43921ec0f0662`; `src/unet.c` = `c4d800b2fdb95a4010048c968b5d52a4151f75c37e797057f124d5429bb7ee86`. Later concurrent edits are outside the captured review boundary.

Priorities retain the original meaning: **P1** serious correctness, memory safety, data integrity, or a blocking failure; **P2** conditional correctness/reliability or material performance concern; **P3** lower-impact diagnostics/coverage/optimization. A source-confirmed conditional defect proves what happens when its stated trigger occurs; it does not claim that a particular dataset or run has hit that trigger. Algorithmic performance findings establish extra work or dependencies, not a measured speed penalty.

### Current review verdict

Substantial handwritten low-precision and memory-planning work exists. Acceptance of ordinary checkpoint resume and packed-weight training is blocked by direct control-flow and parameter-ownership defects. Default fused normalization computes statistics for a different tensor from the one it normalizes. The latest sampler retains validation leakage and unsafe partial-batch publication. Muon integration assigns convolution weights to two optimizers and omits required state. Reported speed, memory, test-pass, and F1 results in `DESIGN.md` and `FP16_MIXED_REPORT.md` are author-reported results; this audit did not reproduce or independently validate them.

| Resolve first | Findings | Why it blocks trust |
| --- | --- | --- |
| Resume and packed updates | F107–F111 | Resume resets loaded weights; packed AdamW/EMA touch overlapping ranges; format transitions and saved state are incomplete |
| Normalization and replay | F112–F114 | Statistics use unrounded values; replay loses saturation; down normalization remains unguarded |
| CUDA failure handling and unsupported paths | F119–F125 | Launch failures can be hidden; fallback/split/conversion paths can corrupt or omit results |
| Latest sampler and validation | F129–F132 | Snapped validation escapes holdouts; incomplete batches can be published; target/mask semantics depend on channel count |
| Muon integration and state | F144–F149 | Double decay/updates, incomplete momentum lifecycle, unstable norm and a test that passes NaNs |
| Historical unresolved data integrity | F004, F018–F024, F047–F050, F070–F081, F083, F087, F089, F091–F092 | Coordinates, cache identity, checkpoint validation, and storage/format integrity still need correction |

The proposed precision engine's contract and acceptance sequence have been reconciled with this source in [PRECISION_PLAN.md](PRECISION_PLAN.md). Implementation presence, a small primitive comparison, synthetic training loss, deployment precision quality, and complete pipeline throughput are separate kinds of evidence.

### New findings: training state and numerical semantics

#### F107 — P1: Ordinary resume immediately reinitializes the checkpoint it just loaded

Locations: [src/train.c:178](/home/forrest/ufsm/src/train.c:178), [src/unet.c:153](/home/forrest/ufsm/src/unet.c:153).

Trigger: `train ... --resume checkpoint` without `--finetune 1`. Loading sets `step0` and restores the arrays. The next `if` checks `resume && finetune`; its `else` calls `unet_init`. That call replaces live parameters and EMA and clears Adam moments. The saved step survives, and the log still says the run resumed. Training therefore starts from random weights at a late learning-rate/bias-correction step rather than continuing the checkpoint. For a packed checkpoint, initialization also leaves old packed arrays alive, adding the state disagreement in F110.

Required correction: make fresh initialization, exact resume, and fine-tuning distinct branches with explicit state semantics. Future validation must compare the first post-resume forward and update with an uninterrupted run. No such comparison was executed here.

#### F108 — P1: Packed AdamW and EMA update overlapping parameter ranges multiple times

Locations: [src/unet.c:107](/home/forrest/ufsm/src/unet.c:107), [src/unet.c:730](/home/forrest/ufsm/src/unet.c:730), [src/unet.c:737](/home/forrest/ufsm/src/unet.c:737).

Trigger: packed weights (`--wq 4` or `--wq 8`) with at least three levels, including the default four-level graph. Parameter construction stores all down convolutions first, then decoder blocks in descending level order. Packed enumeration instead interleaves `down[j], dec[j].c1, dec[j].c2` in ascending level order. The optimizer and EMA assume this enumeration is sorted by parameter offset: gaps and the final tail receive ordinary FP32 updates.

In a three-level example, moving from `down0` to `dec0` first treats intervening `down1` and `dec1` weights as FP32-only gaps. Later visiting `down1` moves `pos` backward; the final tail includes already processed decoder parameters. Thus weights can receive both dense and packed updates, moments and affine/bias parameters can update repeatedly, and unpacked shadows can be modified after their packed update. Two-level layouts happen to avoid this ordering defect. This is an interval-ownership error, not a harmless choice of iteration order.

Required correction: walk disjoint, sorted parameter intervals or explicitly enumerate FP32-only intervals excluding every packed weight range. Validate AdamW and EMA independently on a multilevel graph with distinct gradients per parameter segment.

#### F109 — P1: Changing packed-weight precision keeps incompatible buffers and offsets

Locations: [src/unet.c:702](/home/forrest/ufsm/src/unet.c:702), [src/unet.c:711](/home/forrest/ufsm/src/unet.c:711), [src/train.c:178](/home/forrest/ufsm/src/train.c:178).

Trigger: `--resume packed4.ckpt --finetune 1 --wq 8`, the reverse transition, or repeated public `unet_set_wq` calls. The setter changes `u->wq` immediately, but allocates, computes offsets, and packs only when `wq_q` is null. Existing FP4 data is interpreted as FP8 or vice versa. A `Cin=80` output/tap uses 48 packed FP4 bytes versus 80 FP8 bytes; changing the mode can overwrite following regions and exceed the allocation. FP8→FP4 also changes padding and fails to create residual arrays. Nonzero unsupported bit counts are not rejected by this setter.

Required correction: validate the mode and transactionally rebuild all dependent arrays/offsets on a transition, or reject unsupported transitions before changing state. An inference dispatcher cannot safely treat this setter as a runtime precision switch.

#### F110 — P2: Loading or initializing an existing packed model retains stale packed state and execution flags

Locations: [src/unet.c:153](/home/forrest/ufsm/src/unet.c:153), [src/unet.c:705](/home/forrest/ufsm/src/unet.c:705), [src/unet.c:869](/home/forrest/ufsm/src/unet.c:869).

Trigger: load another checkpoint into an already packed object, initialize that object again, or disable packed mode, update densely, then reenable it. Only the FP32 parameter/EMA/moment arrays are replaced. The packed setter skips repacking existing arrays, so a subsequent packed optimizer/EMA update consumes old grid/residual values and overwrites the newly loaded shadows. Loading a dense checkpoint also fails to disable existing sparse/packed modes because only nonzero saved flags invoke setters. A fresh object avoids this particular reuse defect.

Required correction: reconcile all modes exactly with the loaded checkpoint and regenerate execution state from the newly installed parameters. Define initialization and load as complete state transitions, not partial array assignments.

#### F111 — P2: FP4 checkpoints omit state required for faithful optimizer and EMA continuation

Locations: [src/unet.c:58](/home/forrest/ufsm/src/unet.c:58), [src/unet.c:753](/home/forrest/ufsm/src/unet.c:753), [src/unet.c:824](/home/forrest/ufsm/src/unet.c:824), [src/nn.cu:2341](/home/forrest/ufsm/src/nn.cu:2341).

FP4 updates reconstruct optimizer values using packed grid values plus FP8 error-feedback residuals; EMA has its own residuals and stochastic-rounding counter. Save writes only FP32 grid shadows, EMA shadows, and Adam moments. Load recreates residuals from already rounded shadows, losing accumulated sub-grid information, and does not restore `ema_step`. This affects exact continuation even in a two-level graph that avoids F108. Quantizer/policy state, RNG/sampler progress, and adaptive gradient scale also remain absent from the checkpoint schema.

Required correction: version and persist all non-disposable training state. Alternatively expose a clearly named lossy warm start distinct from exact resume. Inference alone does not need optimizer residuals; training does.

#### F112 — P1: Fused GroupNorm statistics describe unrounded values while normalization reads rounded values

Locations: [src/nn.cu:697](/home/forrest/ufsm/src/nn.cu:697), [src/nn.cu:2025](/home/forrest/ufsm/src/nn.cu:2025), [src/nn.cu:1915](/home/forrest/ufsm/src/nn.cu:1915), [src/nn_fp8.cu:299](/home/forrest/ufsm/src/nn_fp8.cu:299).

Ordinary tensor-core blocks store convolution results through BF16/FP16 conversion, but accumulate output sums/squares from the original float values. Saved mean/rstd therefore describe a different tensor from stored `a1/a2`. The epilogue comment says statistics use stored values; applying saturation before the statistics does not also apply rounding. MX epilogues explicitly use pre-quantization statistics while subsequent normalization reads dequantized MX values, creating the same contract problem unless a different hybrid operator is deliberately specified.

Unexecuted algebraic witness: zero input/weights, bias `257/256`, one element per group, BF16 storage, epsilon `1e-5`. The stored value rounds to `1`, saved mean remains `257/256`, and saved rstd is approximately `316.228`. The constant singleton group is normalized to about `-1.235` instead of zero. The ordinary GN backward formula assumes centered normalized inputs; that invariant is broken and can produce a spurious gradient. The discrepancy can be especially large for constant or nearly constant groups.

Required correction: compute statistics from the representable stored values, or specify a different forward operator and a consistent surrogate derivative. Check constant groups, near-zero variance, singleton groups, and each storage format. Existing MX statistics comparisons against raw FP32 convolution outputs do not establish stored-tensor GroupNorm correctness.

#### F113 — P1: Recompute level 2 loses FP16 saturation during backward replay

Locations: [src/unet.c:535](/home/forrest/ufsm/src/unet.c:535), [src/nn.cu:697](/home/forrest/ufsm/src/nn.cu:697).

Trigger: FP16 activations, `UFSM_RECOMPUTE=2`, and a finite conv1 result greater than `65504`. Original forward requests output GN statistics, so `osum` is non-null and the FP16 epilogue clamps the value to a finite representable range. Backward replay requests no statistics (`G_out=0`), so the same epilogue skips the clamp and stores infinity. GN backward then reads replayed infinity with finite statistics from the original forward. This changes the activation being differentiated and can create nonfinite gradients.

Required correction: make storage conversion/saturation independent of whether statistics are requested, and replay the complete original conversion contract. This finding is about replaying a different value, not merely whether a clamp derivative should use an STE.

#### F114 — P2: Stride-2 stores feeding new down normalization have no FP16 overflow guard

Locations: [src/nn.cu:967](/home/forrest/ufsm/src/nn.cu:967), [src/nn_fp8.cu:1216](/home/forrest/ufsm/src/nn_fp8.cu:1216), [src/unet.c:485](/home/forrest/ufsm/src/unet.c:485), [FP16_MIXED_REPORT.md:201](/home/forrest/ufsm/FP16_MIXED_REPORT.md:201).

Trigger: `down_norm`, FP16 output storage, and a finite stride-2 convolution result outside FP16's finite range. Both 16-bit and FP8 stride-2 epilogues store without the guard added to stride-1 epilogues. GN statistics are then computed from the stored down output, so infinity reaches normalization. The broad report claim that FP16 stores feeding GN are guarded does not cover this later path. No observed overflow frequency or task-quality effect is claimed.

Required correction: define and enforce the same finite-store policy for every normalization input, including downsample outputs, and cover the conditional path explicitly.

#### F115 — P2: FP16 backward rescales earlier accumulated parameter gradients again

Locations: [src/unet.h:32](/home/forrest/ufsm/src/unet.h:32), [src/unet.c:618](/home/forrest/ufsm/src/unet.c:618), [src/unet.c:663](/home/forrest/ufsm/src/unet.c:663).

Trigger: multiple backward calls without `unet_zero_grad`, with 16-bit gradient storage and scale `S != 1`. Each backward accumulates scaled contributions into the flat parameter gradient and then divides the entire array by `S`. After calls with unscaled contributions `g1,g2`, the result is `g1/S + g2`, rather than the API's promised `g1+g2`. The ordinary trainer zeroes each iteration and avoids this public microbatch-accumulation defect.

Required correction: accumulate each call's unscaled contribution or maintain one consistently scaled buffer and unscale once at an explicit accumulation boundary.

#### F116 — P2: Device selection resets the supposedly adaptive FP16 gradient scale

Locations: [src/nn.cu:120](/home/forrest/ufsm/src/nn.cu:120), [src/train.c:226](/home/forrest/ufsm/src/train.c:226), [src/train.c:318](/home/forrest/ufsm/src/train.c:318).

Trigger: the documented environment path `UFSM_F16=1`, with or without `UFSM_GSCALE`. Every `nn_init(device)` resets `g_gscale` to the configured value or 1024. Training repeatedly calls it to select devices. After an overflow halves the scale, a later device selection restores the old value; with multiple GPUs that can happen within the same attempted step. Repeated overflowing batches therefore do not benefit from the advertised adaptive reduction. CLI-only `--f16 1` without the environment variable avoids this specific reset.

Required correction: initialize numerical settings once and keep device selection separate from mutable optimizer/scale state. Preserve and checkpoint adaptive scale state.

#### F117 — P2: Adam bias correction counts attempted steps rather than actual moment updates

Locations: [src/train.c:179](/home/forrest/ufsm/src/train.c:179), [src/train.c:221](/home/forrest/ufsm/src/train.c:221), [src/train.c:315](/home/forrest/ufsm/src/train.c:315), [src/train.c:325](/home/forrest/ufsm/src/train.c:325).

Overflow/nonfinite steps skip AdamW but still advance the loop's `step`; the next successful update uses that attempted step for Adam bias correction. If attempt one is skipped, the first moment update is corrected as update two, although the moments have advanced only once. Fine-tuning also resets `step0` to zero while keeping loaded moments, combining moments of one age with bias correction of another. A fresh learning-rate schedule can be intentional; changing the moment counter without resetting or correctly continuing moments needs its own semantics.

Required correction: separate data/schedule attempt count from successful optimizer update count and define whether fine-tuning preserves or resets moments. Save both counters when applicable.

#### F118 — P2: Default recomputation accepts custom networks that cannot complete backward

Locations: [src/unet.c:627](/home/forrest/ufsm/src/unet.c:627), [src/nn.cu:1869](/home/forrest/ufsm/src/nn.cu:1869), [src/nn_fp8.cu:1579](/home/forrest/ufsm/src/nn_fp8.cu:1579), [src/nn_fp8.cu:1611](/home/forrest/ufsm/src/nn_fp8.cu:1611).

The recomputed non-MX head weight-gradient path supports only `Cin*Cout<=64` and `Cout<=8`. A valid widths `{64,128,256}`, two-output network can construct and run forward, then return `-1` and abort at its first backward under default recomputation. The materialized head path has broader coverage. MX head and normalization kernels impose additional limits (`Cout<=8` forward, `Cin*Cout<=64` / `Cout<=4` head wgrad, `C<=160` GN backward). Guards prevent the obvious local array overruns but do not validate a complete training graph before work starts.

Required correction: validate the selected architecture, formats, and passes together at construction, or supply explicit supported fallbacks. Tiny/large model-size sweeps must not assume every width sequence is supported.

### New findings: low-precision CUDA and state ownership

#### F119 — P1: FP8/MX launch errors are consumed into an error collector normal callers never read

Locations: [src/nn_fp8.cu:30](/home/forrest/ufsm/src/nn_fp8.cu:30), [src/nn_fp8.cu:899](/home/forrest/ufsm/src/nn_fp8.cu:899), [src/nn.cu:121](/home/forrest/ufsm/src/nn.cu:121).

`LPCK` consumes `cudaGetLastError` and stores failure in private `g_lp_err`. Low-precision wrappers generally return success. The caller's `KCHECK` and public `nn_check` then see only their own collector and the already cleared CUDA last error; they do not consult `lp_check`. No normal source/test caller of that collector was found. An immediate unsupported-kernel/configuration/resource failure can therefore leave output unwritten while normal checks report success. A later synchronization might detect some asynchronous failures; that does not fix the consumed immediate error.

Related allocation failure: `lp_buf` at line 597 and 16-bit staging caches ignore allocation/free results and record requested capacity even after failure. Subsequent same-size requests need not retry, leaving a persistent null/stale cache pointer. This amplifies the reporting gap.

Required correction: share one error propagation mechanism, check allocation state transitions, and distinguish enqueue success from completed execution. Validate injected launch/allocation failures in a future execution phase.

#### F120 — P1: `UFSM_S2DIL` selects a float fallback with undersized scratch and 16-bit destinations

Locations: [src/nn.cu:1173](/home/forrest/ufsm/src/nn.cu:1173), [src/nn.cu:1210](/home/forrest/ufsm/src/nn.cu:1210), [src/nn.cu:1239](/home/forrest/ufsm/src/nn.cu:1239).

Trigger: tensor-core mode, non-MX gradient storage, an even-shaped stride-2 3³ dgrad, and `UFSM_S2DIL` present. Registered MX gradients enter an earlier dedicated branch and avoid this forced dilation path. Tensor-mode scratch sizing assumes parity kernels and returns flipped weights plus only 256 bytes. The forced dilation path writes a full `N*Cout*input_volume` float tensor after those weights. It also reads `gy` as float and explicitly writes float `gx`, even when both are allocated as 16-bit gradient storage. The path can overrun scratch and output and misdecode input. This debug switch can affect otherwise ordinary supported U-Net shapes.

Required correction: size workspace from the actual selected path and dispatch typed fallback kernels, or reject this combination before launch.

#### F121 — P2: Odd-shaped stride-2 dgrad falls through to a float-only kernel for 16-bit buffers

Locations: [src/nn.cu:1210](/home/forrest/ufsm/src/nn.cu:1210), [src/nn.cu:1249](/home/forrest/ufsm/src/nn.cu:1249).

Trigger: public stride-2 primitive with any odd input spatial dimension and 16-bit gradient storage. Parity dispatch requires all-even dimensions; the remaining gather kernel accepts float pointers without respecting `GBF`. It reads incorrectly encoded gradients and writes float results into a 16-bit output allocation. Standard multilevel U-Net divisibility checks avoid this particular odd-shape case. The public primitive contract does not reject it.

Required correction: implement a typed odd-shape adjoint or enforce the actual shape/storage capability before launch. This is distinct from F120's forced even-shape dilation workspace defect.

#### F122 — P2: Unvalidated FP8 tuning values break grid selection or divide by zero

Locations: [src/nn_fp8.cu:661](/home/forrest/ufsm/src/nn_fp8.cu:661), [src/nn_fp8.cu:359](/home/forrest/ufsm/src/nn_fp8.cu:359), [src/nn_fp8.cu:863](/home/forrest/ufsm/src/nn_fp8.cu:863).

General FP8 forward accepts arbitrary `UFSM_F8_TZ`. With `Ci>=32`, `Cout=16` or 32, and `TZ=3`, grid arithmetic uses the requested tuple, but the switch default launches specialization `<4,2>`. Its `BM=64` makes `Cop/BM=0`, used in division/remainder, or produces wrong coverage for larger widths. Weight-gradient `UFSM_F8_NT=0` / `MT=0` divide before the supported-tuple check; `UFSM_F8_ZC=0` also divides by zero.

Required correction: validate tuning values before arithmetic and derive the grid exclusively from the specialization that will launch. Treat these environment variables as inputs, not trusted compile-time constants.

#### F123 — P2: Later positional policy entries do not override earlier named entries as documented

Locations: [src/nn.cu:66](/home/forrest/ufsm/src/nn.cu:66), [src/nn.cu:77](/home/forrest/ufsm/src/nn.cu:77), [src/nn.cu:47](/home/forrest/ufsm/src/nn.cu:47).

The accepted policy `enc0.c1=fp8,fp16` leaves encoder 0 conv1 in FP8, although the later positional entry assigns encoder 0 FP16 and the parser promises later entries override earlier ones. Positional assignment changes only `g_lprec`; old finer `g_lprec3` settings still win during dispatch. Invalid policies also mutate/reset prior state before failure, and strings above 2047 bytes are silently truncated; CLI abort-on-error limits the former mainly to reusable API callers.

Required correction: define one precedence rule, apply it consistently, parse transactionally, and expose the effective per-pass plan so accepted configuration cannot silently differ from execution.

#### F124 — P2, compact/API case: The MX split-output correction misses the small-input FP8 specialization

Locations: [src/nn_fp8.cu:612](/home/forrest/ufsm/src/nn_fp8.cu:612), [src/nn_fp8.cu:639](/home/forrest/ufsm/src/nn_fp8.cu:639), [src/nn_fp8.cu:251](/home/forrest/ufsm/src/nn_fp8.cu:251), [src/nn.cu:1745](/home/forrest/ufsm/src/nn.cu:1745).

General MX weight preparation now pads/remaps each split output separately. The small path, selected for MX input with at most eight channels and no split input, does not exclude split output. It pads only total `Cout` and treats padded weight rows as real rows. The shared epilogue expects the second segment at `pad32(o_split)`.

Unexecuted indexing witness: MX `gy` with four or eight channels, dgrad to 32 input channels split into two separate 16-channel MX tensors. `Cop=32` emits offsets 0 and 16. The epilogue skips offset 16 as padding of segment one; segment two starts at 32, which is never emitted. `gx2` remains unwritten. For 80+64 outputs the small preparation also selects wrong real weight rows after the padded boundary. The reported general 80+64 correction and its wide-input regression do not cover this specialization. The default wide network avoids the small-input trigger; compact/custom models and primitive callers do not universally avoid it.

Required correction: use identical segment mapping in every specialization or exclude the unsupported combination before dispatch.

#### F125 — P2, API case: MX 1×1 dgrad dispatches an input format its conversion kernel decodes as float

Locations: [src/nn.cu:1204](/home/forrest/ufsm/src/nn.cu:1204), [src/nn_fp8.cu:1709](/home/forrest/ufsm/src/nn_fp8.cu:1709), [src/nn_fp8.cu:1729](/home/forrest/ufsm/src/nn_fp8.cu:1729).

For registered MX `gy` and `gx`, the caller passes dtype 3 to `lp_conv1_to_mx`. That function recognizes dtype 1/2 but treats all others as plane-major float. It therefore reads MX payload/scale bytes with float indexing and can read beyond the packed input allocation. The normal head uses a plane-major logit gradient and avoids this path.

Required correction: implement the input MX decoder or explicitly reject dtype 3 before calling the float conversion path.

#### F126 — P2: Small FP8 convolutions requantize stored weights onto a different grid

Locations: [src/nn.cu:2231](/home/forrest/ufsm/src/nn.cu:2231), [src/nn_fp8.cu:473](/home/forrest/ufsm/src/nn_fp8.cu:473), [src/nn_fp8.cu:489](/home/forrest/ufsm/src/nn_fp8.cu:489).

Stored quantized weights use separate scales per `(output channel,tap,32 input channels)`. Small-channel forward combines several taps in a K32 block and chooses a common scale. The source assertion that stored-grid forward quantization is exact is therefore false for this backend.

Unexecuted arithmetic witness: `Ci=4`, one weight `2^-20` in tap 0 and weight 1 in tap 1, others zero. Per-tap E4M3 grids represent both exactly (payload 256 with scales `2^-28` and `2^-8`). A shared small-forward block chooses `2^-8`; the tiny normalized value `2^-12` rounds to zero on the E4M3 subnormal grid. The deployed operator differs from the stored grid, although the task-quality impact is unmeasured.

Required correction: align quantization block definitions or make this extra conversion explicit in the model/QAT contract and its reference.

#### F127 — P2: Model destruction leaks newly added sparse and packed parameter buffers

Locations: [src/unet.c:128](/home/forrest/ufsm/src/unet.c:128), [src/unet.c:690](/home/forrest/ufsm/src/unet.c:690), [src/unet.c:713](/home/forrest/ufsm/src/unet.c:713).

`unet_free` frees ordinary state and activations but omits `wm` and all eight packed parameter/scale/residual arrays. Repeated model sweeps, checkpoint evaluations, or public-library create/destroy cycles retain device allocations after objects are destroyed. This is separate from the removed historical debug-buffer leak F052.

Final-cut extension: `muon_mom` (`unet.c:692`, `4*np` bytes) and `muon_work` (`unet.c:686`, largest matrix workspace) are also omitted from the destructor at lines 129–134. Muon model destruction has the same ownership defect.

Batched-Muon extension: the new descriptor allocation and summed scratch pool are also absent from cleanup. The pool is now the sum of matrix scratch regions rather than only the largest single-convolution workspace; account for both layouts separately.

Required correction: include every owned optional allocation in destruction and account for it in total model memory. No runtime leak measurement was performed.

### New findings: sampler, storage, and pipeline

#### F128 — P1 in secondary, fixed in final: Prefetch waited forever for unmarked bitmap entries

Locations: [secondary src/sample.c:507](/tmp/ufsm-review-latest-20261001-m1hfpC/src/sample.c:507), [secondary src/sample.c:551](/tmp/ufsm-review-latest-20261001-m1hfpC/src/sample.c:551), [secondary src/sample.c:555](/tmp/ufsm-review-latest-20261001-m1hfpC/src/sample.c:555).

Workers skip unmarked grid entries before incrementing `done`. After all workers finish, `done=nmark`, the number of marked entries. The coordinator waits until `done=total`, the size of the entire grid. Any `nmark<total` leaves all workers finished while the coordinator loops forever. Ordinary localized occupancy and fractional prefetch can produce this condition; it does not require an I/O failure. Zero worker count, accepted by the CLI, and unchecked worker creation failures can independently leave completion unreachable.

Required correction: use an explicit worker-completion condition or a counter whose denominator matches what workers count, validate positive thread counts, and unwind failed thread creation. Fixing fetch performance cannot resolve this control-flow deadlock.

**Final status:** commit `146b440`, included in the final cut, changes the wait to `done<nmark` ([final src/sample.c:562](/tmp/ufsm-review-final-20261001-ten1DI/src/sample.c:562)). The marked-versus-total defect is fixed by source inspection. Zero workers and unchecked failed `pthread_create` still make completion impossible when marked work remains; those numeric/resource failures remain within F003/F099. No execution verified this correction.

#### F129 — P1, secondary: Default chunk snapping moves validation patches outside held-out bounds

Locations: [secondary src/sample.c:30](/tmp/ufsm-review-latest-20261001-m1hfpC/src/sample.c:30), [secondary src/sample.c:229](/tmp/ufsm-review-latest-20261001-m1hfpC/src/sample.c:229), [secondary src/sample.c:247](/tmp/ufsm-review-latest-20261001-m1hfpC/src/sample.c:247).

`snap` defaults to one and applies to non-region sources in both training and validation. Validation first chooses an origin inside a holdout, then rounds it down to the CT chunk grid. Only training checks holdout intersection afterward. Unexecuted coordinate witness using accepted patch **96**: level 0, CT chunk width 64, holdout `[100,196)` on all axes. The only original valid validation origin is 100; snapping changes it to 64, producing `[64,160)³`. A training patch at origin zero produces `[0,96)³` and passes holdout rejection because it ends before 100. Its overlap with validation is `[64,96)³`. Thus accepted geometry can share training/validation voxels, provided the remaining foreground/target checks accept the patches.

Required correction: disable snapping for holdout validation or choose/revalidate an aligned window wholly inside the held-out extent. This regression is separate from the retained region-branch and physical-volume split problems F019/F020/F035.

Final cut still selects holdout origins at `sample.c:231–233`, then snaps at `249–254` and only protects training at `257`.

#### F130 — P1: Stop and fatal-read paths publish incompletely filled sampler slots as READY

Locations: [primary src/sample.c:407](/tmp/ufsm-review-20261001-WhnCIo/src/sample.c:407), [primary src/sample.c:516](/tmp/ufsm-review-20261001-WhnCIo/src/sample.c:516), [secondary src/sample.c:459](/tmp/ufsm-review-latest-20261001-m1hfpC/src/sample.c:459), [secondary src/sample.c:598](/tmp/ufsm-review-latest-20261001-m1hfpC/src/sample.c:598).

The improved filling loop exits when stop is set, including after its consecutive-read-failure limit or another worker's failure. It unconditionally marks the whole slot READY even if fewer than `B` examples were filled. `sampler_next` selects READY before checking stop, so a consumer can receive uninitialized examples from first use or stale examples left by a previous batch. The improvement to F014's retry counter does not make failure publication safe.

Required correction: publish only fully filled slots; represent terminal failure separately and wake consumers without offering a partial batch. Do not infer batch validity merely from a READY state.

Final cut retains the same unconditional publication at `sample.c:464–471`.

#### F131 — P2: Softening can make neighboring background stronger than its fractional surface seed

Locations: [primary src/sample.c:275](/tmp/ufsm-review-20261001-WhnCIo/src/sample.c:275), [secondary src/sample.c:166](/tmp/ufsm-review-latest-20261001-m1hfpC/src/sample.c:166), [secondary src/sample.c:342](/tmp/ufsm-review-latest-20261001-m1hfpC/src/sample.c:342).

All nonzero, non-ignore labels are treated as full-strength zero-distance surface seeds. Softening updates only target values equal to zero, leaving fractional pooled positives unchanged. For example, a seed value 31 remains 31 while adjacent zero background receives approximately 203 at sigma 1.5. The resulting ridge can peak off the original positive voxel. The secondary lookup-table optimization preserves this inconsistency. Whether fractional positives should remain fractional or become a full ridge center must be defined explicitly; both assumptions cannot be used simultaneously.

Required correction: specify how pooled probabilities contribute to ridge strength and apply that rule consistently to seeds and neighbors. Compare actual target fields across levels before attributing quality changes solely to precision.

#### F132 — P2, secondary: The soft/trust distance cache changes seed semantics when another channel is active

Locations: [secondary src/sample.c:330](/tmp/ufsm-review-latest-20261001-m1hfpC/src/sample.c:330), [secondary src/sample.c:340](/tmp/ufsm-review-latest-20261001-m1hfpC/src/sample.c:340), [secondary src/sample.c:352](/tmp/ufsm-review-latest-20261001-m1hfpC/src/sample.c:352).

Trigger: soft targets, a positive trust band, and multiple active channels. The soft loop leaves the distance map for its last channel. The trust loop starts again at the first active channel, so its cache miss recomputes distances from already softened `ttmp`. Soft positives satisfy the hard-surface seed predicate. The next channel also recomputes after the previous channel replaces the cache. With one active channel the original hard map is reused. Thus adding unrelated supervision changes the mask for an existing channel, and the comment claiming measurement from hard surfaces is false. Dilation also precedes distance generation, so original annotation versus dilated-band seed semantics remain distinct choices.

Required correction: preserve immutable hard seed masks/maps or finish soft/trust operations for each channel before changing the cached map. Validate mask invariance to adding a supervised channel and to soft-target annealing.

Final cut retains this cache/seed behavior at `sample.c:335,345,357`; the missing body-diagonal chamfer neighbors have been corrected separately.

#### F133 — P2: Occupancy setup repeats large whole-source scans and has unchecked large allocations

Locations: [primary src/sample.c:428](/tmp/ufsm-review-20261001-WhnCIo/src/sample.c:428), [secondary src/sample.c:480](/tmp/ufsm-review-latest-20261001-m1hfpC/src/sample.c:480), [secondary src/sample.c:490](/tmp/ufsm-review-latest-20261001-m1hfpC/src/sample.c:490), [src/train.c:197](/home/forrest/ufsm/src/train.c:197).

Every sampler startup scans each source's coarse recto volume, including zero-weight or holdout-ineligible sources. Training and validation startup repeat the scan/index construction sequentially; validation stops and frees its indexes before training starts. The accepted limit of 300 million cells permits about 300 MB of temporary labels and up to 1.2 GB of retained indices per source in a sampler, without an aggregate budget. The large index allocation is written without checking success, a concrete new exposure of F099. These are source-derived allocation/work estimates, not measured RSS or startup time.

Required correction: skip ineligible sources, share immutable indexes where appropriate, check allocations, and bound aggregate memory. Treat occupancy construction as pipeline setup cost rather than excluding it from all throughput claims.

#### F134 — P2: Prefetch can report success after failed reads or failed cache publication

Locations: [primary src/sample.c:477](/tmp/ufsm-review-20261001-WhnCIo/src/sample.c:477), [secondary src/sample.c:510](/tmp/ufsm-review-latest-20261001-m1hfpC/src/sample.c:510), [secondary src/sample.c:562](/tmp/ufsm-review-latest-20261001-m1hfpC/src/sample.c:562), [secondary src/zarr3.c:328](/tmp/ufsm-review-latest-20261001-m1hfpC/src/zarr3.c:328).

The command prints/counts fetch failures but returns zero. The new unsharded helper treats any failed `store_read_all` as absent/cached return zero, including permissions, transport, and local read errors. `cache_put` supplies no publication status, so a chunk can count as fetched even though it was not cached. Fixing F128's wait condition leaves these false-success paths intact. The underlying missing-versus-error problems F071/F072 are still present.

Required correction: distinguish missing data from failed reads, propagate cache publication failure, and report a nonzero outcome for incomplete requested warming.

#### F135 — P2, secondary: Fractional prefetch warms a spatial prefix rather than representative occupied cells

Locations: [secondary src/sample.c:498](/tmp/ufsm-review-latest-20261001-m1hfpC/src/sample.c:498), [secondary src/sample.c:528](/tmp/ufsm-review-latest-20261001-m1hfpC/src/sample.c:528), [secondary src/sample.c:539](/tmp/ufsm-review-latest-20261001-m1hfpC/src/sample.c:539).

The new bitmap path selects `o.idx[k]` from the first fraction of a raster-ordered occupancy list. Its comment still describes shuffled coverage. A small fraction warms the earliest occupied z/y/x region while training samples uniformly from the full list, so cache warming is spatially biased. No measured cache-hit penalty is claimed.

Required correction: use a reproducible distributed sample/permutation if fractional warming is intended to represent training coverage; report selected cells and actual published chunks separately.

#### F136 — P2: Prediction error exits abandon a reader thread using stack-owned state

Locations: [src/predict.c:37](/home/forrest/ufsm/src/predict.c:37), [src/predict.c:130](/home/forrest/ufsm/src/predict.c:130), [src/predict.c:137](/home/forrest/ufsm/src/predict.c:137), [src/predict.c:185](/home/forrest/ufsm/src/predict.c:185).

The threaded read-ahead implementation has no cancellation predicate in the full-ring wait. Main-thread reader/writer/CUDA failure returns occur before signaling cancellation and joining. The reader can remain blocked or continue accessing stack-owned `reader_t rd` and box state after `cmd_predict` returns. Normal successful ring reuse is safe in the reviewed order because the slot is released after the synchronous output transfer. The normal CLI exits after an error, limiting persistent exposure, but the function has invalid error-path lifetime for reuse in a longer-lived caller. Reader errors also store only a failed flag; printing the consumer's thread-local `z3_error()` loses the originating message (F098).

Required correction: use stop-aware waits and one cleanup path that signals, joins, and frees on every outcome; transfer the actual reader error across threads.

#### F137 — P2, unmeasured performance: Next-batch upload waits for the current forward and backward to finish

Locations: [src/train.c:51](/home/forrest/ufsm/src/train.c:51), [src/train.c:279](/home/forrest/ufsm/src/train.c:279), [src/train.c:291](/home/forrest/ufsm/src/train.c:291).

The trainer records `ev_done` after the current forward/backward and makes the next-buffer copy stream wait on that same event. Even though the other device buffer is available, the next batch cannot upload during this step's forward/backward. It may overlap later reductions/optimizer work. The comment promising upload while this step computes therefore overstates the implemented overlap. This dependency is visible in source; its time cost was not measured.

Required correction: track each buffer's last use separately and overlap only after that buffer's actual prior consumer finishes. Verify timeline overlap and end-to-end step latency in a future measurement phase.

#### F138 — P2: Multi-GPU averaging changes the global active-target loss when local counts differ

Locations: [src/train.c:299](/home/forrest/ufsm/src/train.c:299), [src/nn.cu:2642](/home/forrest/ufsm/src/nn.cu:2642), [src/nn.cu:2668](/home/forrest/ufsm/src/nn.cu:2668).

The loss normalizes each GPU's gradients by its own number of active `(sample,channel)` pairs. The trainer then averages these local gradients with weight `1/ng`. This equals a global per-active-pair objective only when all local active counts match. With one GPU having two active pairs and another one, global weighting should be 2:1; the implementation uses 1:1. A GPU with no active targets still dilutes another GPU's gradient. Mixed supervision makes unequal counts realistic. This is independent of F008's scalar logging mismatch.

Required correction: declare the global objective and aggregate/reweight active counts consistently with it before deriving or averaging gradients. Do not fix logging alone and assume multi-GPU optimization is equivalent.

#### F139 — P2: The all-GPU profiler reuses events created on the first device

Locations: [src/nn.cu:168](/home/forrest/ufsm/src/nn.cu:168), [src/nn.cu:171](/home/forrest/ufsm/src/nn.cu:171), [src/train.c:353](/home/forrest/ufsm/src/train.c:353).

Trigger: profiling a multi-GPU training run. All event pairs are lazily created while the first GPU is current, then reused for later GPUs' default streams. Event recording requires the event and recording stream to belong to the same device context. Creation/record/elapsed-time returns are ignored, so invalid later-device recordings can silently omit or invalidate categories printed as “all GPUs.” Collection synchronizes only the current device. Skipped optimizer steps also bypass the matching `nn_prof_end` after beginning category 6, weakening interval pairing.

Required correction: own events and completion per device, check profiler API results, and balance intervals on skip/error paths. The old missing-backward-wrapper issue F063 is substantially fixed; this is a distinct current measurement defect.

#### F140 — P3: New precision benchmark lists can overwrite fixed stack arrays

Locations: [tests/bench_lp.c:87](/home/forrest/ufsm/tests/bench_lp.c:87), [tests/prec_sweep.c:110](/home/forrest/ufsm/tests/prec_sweep.c:110).

`bench_lp` appends every digit of `UFSM_PRECS` to `precs[4]` without a bound; the natural five-mode string `01234` already exceeds it. `prec_sweep` copies every character of `UFSM_SWEEP_PRECS` into `pl[8]` and later indexes eight-entry result tables with the unchecked length. These are diagnostic executable inputs, not production model inputs, but can corrupt the benchmark responsible for selecting precision policies.

Required correction: validate lengths and legal modes before modifying arrays, and return failure for unsupported diagnostic inputs.

#### F141 — P3, secondary: Per-worker Gaussian-noise tables are never freed

Locations: [secondary src/sample.c:446](/tmp/ufsm-review-latest-20261001-m1hfpC/src/sample.c:446), [secondary src/sample.c:470](/tmp/ufsm-review-latest-20261001-m1hfpC/src/sample.c:470).

The new worker allocates 4,096 floats, fills the table, and omits it from cleanup. Sampler restart/shutdown leaks 16 KiB per worker. This is bounded per startup and lower priority than the batch publication and prefetch defects.

Required correction: include the table in worker ownership/cleanup. Its finite reused noise distribution is also a changed augmentation recipe and should be recorded when comparing runs.

#### F142 — P2, acceptance gap: Current test and quality claims do not certify the deployed mixed-precision graph

Locations: [Makefile:72](/tmp/ufsm-review-20261001-WhnCIo/Makefile:72), [tests/test_rc.c:26](/home/forrest/ufsm/tests/test_rc.c:26), [tests/train_lp.c:145](/home/forrest/ufsm/tests/train_lp.c:145), [tools/eval_holdouts.py:41](/home/forrest/ufsm/tools/eval_holdouts.py:41).

The expanded test target includes useful MX primitives and recomputation checks, but does not explicitly run full-network FP4, packed optimizer/EMA, exact packed resume, per-pass policy transitions, or down-normalization combinations. `test_rc` deliberately uses FP32 activation/gradient storage, missing rounded GN statistics, FP16 clamp replay, and scaled gradient accumulation. Whole-network error checks and EMA assertions still have F066/F067 gaps. `train_lp` selects FP32 kernels for held-out evaluation, prints results, and returns zero even after a detected CUDA error or a skipped invalid policy. Its synthetic held-out loss is therefore not a direct measurement of the deployment arithmetic, topology, or a pass/fail quality budget.

The real-data holdout tool can request precision, but accepts any existing prediction metadata as a cache hit. Checkpoint, architecture, policy, input, and completed-output identity are absent from cache validation (F018), so changing `--prec` or `UFSM_PREDICT_ARGS` does not guarantee a new prediction. Failures are printed/skipped without a failing overall status (extension of F036). F1 at selected thresholds does not alone establish sheet merge/split topology or cross-shard consistency. Reported sanitizer/test passes on finite ordinary cases cannot dismiss source witnesses outside those cases.

Closing-cut provenance discrepancy: `tools/make_sources.py:154` and the applicable `all2.json` entry set `min_level=2`, while `DESIGN.md:342` describes 1. Region/Kaggle exemptions are present. Experiment notes need the actual resolution recipe, rather than the stale documentation value.

Required acceptance evidence, after execution is authorized: finite-value assertions; independent converter and optimizer references; default and compact shape matrices; complete backward under each storage/recompute/down-norm mode; exact resume where promised; explicitly fresh, identity-bound predictions under the deployment policy; trusted split/coordinate/mask rules; and complete latency/memory measurements. No tests were added or executed in this review.

### Final-cut findings: Muon, scheduling, and later pipeline changes

#### F143 — P2: Accepted WSD/Muon schedule inputs can produce nonfinite learning rates

Locations: [final src/train.c:136](/tmp/ufsm-review-final-20261001-ten1DI/src/train.c:136), [final src/train.c:314](/tmp/ufsm-review-final-20261001-ten1DI/src/train.c:314), [final src/train.c:333](/tmp/ufsm-review-final-20261001-ten1DI/src/train.c:333).

With `--sched wsd --cooldown 0` and a final step after warmup, `cd0=steps` and the final cooldown expression is `0/0`. The resulting NaN learning rate reaches parameter updates without a finite-rate check, despite finite loss/gradients. Small fractions that round `cd0` to `steps` have the same risk. Separately, `--opt muon --lr 0` computes `muon_lr*(lr/lr0)`, again `0/0`; a request to disable the base rate can corrupt parameters instead. Invalid optimizer/schedule names silently select default branches, and hyperparameters/ranges remain unchecked.

Required correction: validate schedule/optimizer names and ranges, define zero-cooldown and zero-rate behavior, and reject nonfinite effective rates before applying updates. This extends F003 with new algorithm-specific arithmetic.

#### F144 — P2: Fresh Muon convolution updates receive a second weight-decay operation from AdamW

Locations: [final src/unet.c:680](/tmp/ufsm-review-final-20261001-ten1DI/src/unet.c:680), [final src/unet.c:695](/tmp/ufsm-review-final-20261001-ten1DI/src/unet.c:695), [final src/nn.cu:2755](/tmp/ufsm-review-final-20261001-ten1DI/src/nn.cu:2755).

Muon updates each 3³ convolution, including its decay, then zeroes that convolution's gradient. AdamW subsequently visits the entire parameter array with nonzero decay. Zero current gradient does not exclude a parameter from AdamW. With zero initial Adam moments, the resulting weight is

`p_final = (1-lr_adam*wd) * ((1-lr_muon*wd)*p_old - lr_muon*scale*O)`.

The additional AdamW factor decays both the parameter and the freshly applied Muon contribution. This occurs in ordinary fresh `--opt muon` training with default nonzero weight decay. The comment claiming the zeroed gradients leave these weights alone is false.

Required correction: give Muon convolution weights and AdamW-only parameters disjoint update intervals. A zero gradient is not an optimizer ownership mask.

#### F145 — P1: Fine-tuning an AdamW checkpoint with Muon applies both optimizer directions to convolution weights

Locations: [final src/train.c:183](/tmp/ufsm-review-final-20261001-ten1DI/src/train.c:183), [final src/unet.c:687](/tmp/ufsm-review-final-20261001-ten1DI/src/unet.c:687), [final src/nn.cu:2718](/tmp/ufsm-review-final-20261001-ten1DI/src/nn.cu:2718).

Trigger: `--resume adamw.ckpt --finetune 1 --opt muon`. This path preserves loaded Adam moments. After Muon updates and zeroes conv gradients, the full-array AdamW pass decays the nonzero moments and applies their bias-corrected update as well. Convolution weights receive two optimizer directions, independently of F144's double decay. The fine-tune step reset can amplify the old-moment correction (F117). This is a real accepted CLI path that avoids F107's ordinary-resume reinitialization branch.

Required correction: exclude Muon-owned intervals from AdamW and define optimizer-family transitions explicitly, including which prior state is discarded or converted.

#### F146 — P2: Muon checkpoints omit momentum and optimizer identity

Locations: [final src/unet.c:692](/tmp/ufsm-review-final-20261001-ten1DI/src/unet.c:692), [final src/unet.c:836](/tmp/ufsm-review-final-20261001-ten1DI/src/unet.c:836), [final src/train.c:132](/tmp/ufsm-review-final-20261001-ten1DI/src/train.c:132).

Muon momentum is persistent input to every update, but save/load still serialize only parameters, EMA, Adam m, and Adam v. A fresh loaded object initializes Muon momentum to zero on its next update, so it cannot continue the previous trajectory. Header metadata also omits optimizer identity/settings; resuming without repeating `--opt muon` selects AdamW. Fixing ordinary resume initialization does not fix this independent state omission.

Required correction: version optimizer identity/settings and persist Muon momentum. Rebuild workspace, but restore non-disposable state for exact continuation.

#### F147 — P2: Reinitializing or loading a reused Muon model leaves unrelated old momentum active

Locations: [final src/unet.c:154](/tmp/ufsm-review-final-20261001-ten1DI/src/unet.c:154), [final src/unet.c:692](/tmp/ufsm-review-final-20261001-ten1DI/src/unet.c:692), [final src/unet.c:881](/tmp/ufsm-review-final-20261001-ten1DI/src/unet.c:881).

Once allocated, `muon_mom` is zeroed only during first allocation. `unet_init` resets weights/Adam moments without clearing it; `unet_load` replaces checkpoint arrays without restoring or clearing it. Reusing an object therefore carries momentum from another initialization/checkpoint into the new model. Fresh objects avoid stale momentum but have F146's lost-momentum continuation problem. The same buffers also leak at destruction (F127 final extension).

Required correction: include Muon state in every complete initialization/load transition and destructor. Distinguish fresh start, warm start, optimizer change and exact resume.

#### F148 — P2, public numerical case: Muon squares finite gradients in float and can normalize them to zero

Locations: [final src/nn.cu:2749](/tmp/ufsm-review-final-20261001-ten1DI/src/nn.cu:2749), [final src/nn.cu:2754](/tmp/ufsm-review-final-20261001-ten1DI/src/nn.cu:2754).

The sum destination is double, but squares, thread sums and shared reductions are float. Conversion to double occurs after possible overflow. Unexecuted witness: `Co=K=1`, beta/decay zero, parameter/momentum zero, finite gradient `1e20`. Squaring overflows, the inverse norm becomes zero, and all polynomial iterations and the update remain zero. A rescaled gradient 1 instead produces a nonzero update. Thus finite magnitude alone can collapse a normalized update. Default trainer clipping mitigates this trigger; the public primitive has no enforced clipping prerequisite. F055 describes the related existing gradient-norm problem.

Required correction: use wide or scaled norm reduction and validate finite normalized state before committing an update.

#### F149 — P2: The new Muon test passes all-NaN output and does not validate optimizer integration

Locations: [final tests/test_muon.c:18](/tmp/ufsm-review-final-20261001-ten1DI/tests/test_muon.c:18), [final Makefile:45](/tmp/ufsm-review-final-20261001-ten1DI/Makefile:45), [final Makefile:78](/tmp/ufsm-review-final-20261001-ten1DI/Makefile:78).

NaN output makes each Gram entry NaN; `fmax(current,fabs(NaN))` retains the finite current value, initially zero. Both reported errors remain zero and the test succeeds. It also tests only one wide random matrix with zero parameters/momentum, beta zero and decay zero, directly calling `nn_muon`. It cannot detect F144/F145, momentum evolution, state continuation, or even an unrelated fixed orthogonal output with the same Gram properties. The aggregate test target neither depends on nor runs the new target.

Required correction: reject nonfinite values, compare the intended multi-step polynomial/update to an independent reference, and test complete model partition/state behavior. Rank-deficient inputs need reference behavior; five iterations cannot guarantee identity Gram for arbitrary ranks/conditioning. This is a concrete new instance of F066/F067, not evidence that tests were run here.

The final batched-Muon delta adds no batched-versus-unbatched or model-integration assertion; the existing test still calls the standalone primitive only.

#### F150 — P2, unmeasured performance: Muon adds scalar matrix products and many launches for each convolution

Locations: [final src/nn.cu:2724](/tmp/ufsm-review-final-20261001-ten1DI/src/nn.cu:2724), [final src/nn.cu:2767](/tmp/ufsm-review-final-20261001-ten1DI/src/nn.cu:2767).

Each matrix output uses a serial scalar dot-product loop; there is no tiled/shared-memory or tensor-core GEMM. Five iterations perform approximately `5*(2*Co²*K + Co³)` multiply-add terms per convolution, excluding normalization and application. At Co80/K2160 this is **140.8 million terms per convolution per step**, with 19 kernel launches plus a memset. These are analytical work counts, not measured latency. Tall matrices also use Co×Co Gram storage instead of a potentially cheaper K×K orientation.

Required correction/measurement: include optimizer work in complete step comparisons, then consider tiled products and the smaller orientation where useful. Computing `P(XXᵀ)X` is mathematically compatible with `X P(XᵀX)`; lack of transpose is a cost concern, not a proven orientation error.

**Latest status:** the captured uncommitted batched path ([nn.cu:2816](/tmp/ufsm-review-batched-20261001-8XXlTL/src/nn.cu:2816)) reduces the numerical-kernel launch count to 18 for all convolutions together, plus the norm memset. It does not remove the scalar dot-product work or float norm overflow. Model integration still issues individual gradient-zeroing operations afterward. The 19-per-matrix statement describes the retained unbatched path, not the latest default.

#### F151 — P2: Latest prefetch warms whole coarse CT volumes independently of requested coverage

Locations: [final src/sample.c:535](/tmp/ufsm-review-final-20261001-ten1DI/src/sample.c:535), [final src/sample.c:538](/tmp/ufsm-review-final-20261001-ten1DI/src/sample.c:538), [final src/sample.c:485](/tmp/ufsm-review-final-20261001-ten1DI/src/sample.c:485).

The `whole=l>=4` path marks every CT chunk at all available levels 4 and above, regardless of requested max level, needed probe level or fraction. Its justification says occupancy scans the coarse volume in full, but `source_occupancy` scans the recto **target**, not CT. This extra CT warming cannot eliminate that target scan, and even a tiny fractional request can fetch all coarse CT chunks and indexes. This is a scope/resource concern; no amount of transferred data or startup time was measured.

Required correction: make whole-CT warming explicit, restrict it to requested needs, and warm/share the actual target occupancy data if that is the purpose. Report actual chosen coverage and cache publication.

#### F152 — P2: New stat-only cache lookup treats unusable paths as cached chunks

Locations: [final src/zarr3.c:319](/tmp/ufsm-review-final-20261001-ten1DI/src/zarr3.c:319), [final src/zarr3.c:327](/tmp/ufsm-review-final-20261001-ten1DI/src/zarr3.c:327), [final src/zarr3.c:338](/tmp/ufsm-review-final-20261001-ten1DI/src/zarr3.c:338).

`cache_has` accepts successful stat without verifying regular-file type or read access. A directory or unreadable file at a cache key is reported as already cached, so prefetch does not repair/report it; training later fails to read or fetches it again. The old read-based lookup at least attempted to open/read the entry. This does not establish a newly introduced general corruption/checksum defect; both paths lack broader cache content validation.

Required correction: define usable cache entries and distinguish absent, valid, unreadable and corrupt states without mistaking mere filesystem existence for successful warming.

#### F153 — P3: Sampler stage profiling omits work spent on rejected foreground probes

Locations: [final src/sample.c:268](/tmp/ufsm-review-final-20261001-ten1DI/src/sample.c:268), [final src/sample.c:279](/tmp/ufsm-review-final-20261001-ten1DI/src/sample.c:279).

Coarse CT reads that reject occupancy return before the probe timing mark; fine CT statistics that reject foreground return before the stats mark. Frequent rejections therefore consume CPU/I/O work absent from those stage totals, obscuring the bottleneck. Other recorded stages can include rejected attempts, making the mixed accounting harder to interpret. No measured profiler discrepancy was reproduced.

Required correction: record time on every outcome and declare whether totals cover all attempts or only produced patches. Separate startup, attempted-draw work, completed production and queue consumption.

#### F154 — P2, mode-switch/API case: Switching Muon batching can invalidate the cached descriptor pool

Locations: [batched src/unet.c:687](/tmp/ufsm-review-batched-20261001-8XXlTL/src/unet.c:687), [batched src/unet.c:705](/tmp/ufsm-review-batched-20261001-8XXlTL/src/unet.c:705), [batched src/unet.c:715](/tmp/ufsm-review-batched-20261001-8XXlTL/src/unet.c:715).

Trigger: reuse one model while changing the per-call `UFSM_MUON_UNBATCHED` setting from absent to present and then absent. The batched path stores its descriptor pool in `muon_work` but leaves `muon_work_n` zero. The first unbatched call sees a positive workspace requirement, frees that pool, and replaces it with single-matrix scratch. Existing device descriptors still contain pointers into the freed pool. Returning to batched mode skips descriptor construction because `muon_descs` is non-null and launches with dangling scratch pointers. Conversely, starting unbatched and then building the batched pool overwrites the old workspace owner without freeing it. Fixed-mode CLI runs avoid this switch trigger; mutable library/diagnostic mode use does not.

Required correction: give single-matrix scratch and the batched pool separate ownership, or invalidate/rebuild descriptors transactionally whenever their pool changes. The new batching reduces launches but does not make cached pointer lifetimes safe under the exposed per-call mode selection.

### Precision and performance claims reconciled with source

| Claim or label | What the reviewed source actually does | Consequence for the plan |
| --- | --- | --- |
| Native FP4 | MXFP4 E2M1 / UE8M0 blocks of 32 (`nn_fp8.cu:197`); NVFP4 appears in simulation | Preserve and validate MXFP4 first; native NVFP4 is still a separate implementation |
| `all=fp4` | Small-Ci, stride-2, and wgrad fall back to FP8; fused upsampling uses 16-bit kernels; head remains float accumulation | Report effective per-operator/per-pass arithmetic, not one network-wide label |
| FP16 policy 4 | FP16 group/partial accumulators are periodically folded into FP32 with local scales | Distinguish it from FP16 operands with FP32 accumulation and validate the folding error |
| Packed weights save model memory | FP32 live/EMA shadows remain; packed arrays and FP4 residuals are additional; convolution repacks execution operands | Count all persistent copies and repack/scratch traffic before claiming a memory or speed win |
| 2:4 sparsity enables sparse TOPS | Weights are copied/masked and passed to dense MMA; no native sparse metadata/instruction path was found | Treat as a pruning/training experiment; sparse peak throughput is inapplicable |
| All MX dgrad uses tensor cores | MX stride-2 dgrad uses scalar decoded gathers (`nn_fp8.cu:1776`) | Measure actual kernels, including repeated per-CTA weight staging |
| Complete dynamic precision is present | Host policy selection exists, but storage/configuration/cache state is global and transitions have gaps | Require validated graph-boundary transitions and explicit ownership before promising reusable dynamic execution |
| New sampler/prediction timings describe full pipeline | Reader overlap exists; sampler benchmarks omit setup, consume preproduced queue contents, and read benchmarks do not establish a warm cache | Separate initialization, queue drain, steady-state production, decoding, transfers, compute, and publication |
| Latest minimum-level and I/O-counter changes preserve the old recipe | Applicable fine-resolution pyramid sources now exclude levels below 2; counters omit non-cache-backed handles | Record physical sampling resolution and counter scope; do not interpret either change as an isolated precision result |

No hardware peak, source FLOP count, sanitizer claim, or synthetic learning curve establishes an end-to-end quality/latency improvement. Full useful work includes normalization, conversion, staging, replay, gradient exchange, optimizer state, data preparation, and persistence.

### Historical finding status register

**Fixed** below means the original offending source pattern was removed or repaired by inspection. It does not mean tests were run by this review. **Partial** means the exact improvement and remaining trigger are identified. **Open/conditional** retains the original scope qualification. Storage byte identity is evidence that old code paths persist, not a runtime reproduction.

| Historical IDs | 2026-10-01 status / current evidence |
| --- | --- |
| F001 | **Fixed:** validation passes null gradient; loss launches gradient work only for a non-null destination (`train.c:98`, `nn.cu:2684`). |
| F002 | **Fixed:** formats target now includes both CUDA objects and CUDA libraries (`Makefile:63–64`). |
| F003 | **Open:** numeric options remain unchecked; unsafe intervals, halo/window geometry, levels, worker counts, and products persist. New tuning/list examples are F122/F128/F140. |
| F004–F007 | **Open:** evaluation still ignores recorded origin; validity differs across band matching; hard tile interiors remain; axis code still assumes numeric OME paths. No blend implementation was found. |
| F008–F013 | **Open:** scalar objective weighting, nested validation cadence, ignored publication failures, JSON interpolation, shell-based mkdir, and truncated integer threshold remain. Multi-GPU weighting is separately F138. |
| F014 | **Partial:** fill loop now observes stop and counts I/O failures correctly. Impossible/rejected draws merely warn at `2^22` and never terminate; READY publication is unsafe (F130). Closing `min_level` filtering can also remove all eligible levels (minimum beyond MAXLEV, incompatible region level, or probability only below minimum), with no preflight eligibility error. |
| F015 | **Partial:** writes/duplicate opens are serialized, but `sources.c:189–236` reads pointers/presence outside the mutex while first publication writes them inside; the C data race remains. |
| F016–F017 | **Open:** anisotropic ingest and TIFF/mesh integration paths were not corrected; ingest's observed format change only adds ZIP u1 dtype spellings. |
| F018 | **Partial:** generator/all2 names improve known collisions; script still defaults to duplicate names in `all.json` and validates cache by existence only. Different checkpoints/policies can reuse stale output. |
| F019–F025 | **Open or originally conditional:** region branch still bypasses holdout placement; split identity is per source, not physical CT; shared ignore mask and last-region-channel selection remain; cache override/presence/scale assumptions remain. Actual cross-source label overlap was not established. |
| F026–F037 | **Open or originally conditional:** ingestion error/geometry assumptions, static-catalog generator exclusion, legacy teacher encoding, direct TIFF masks/pages, nonaligned origins, shell failure statuses, and download identity remain. ZIP u1 acceptance does not fix order/filter validation. |
| F038 | **Partial, final:** per-voxel Box–Muller log/cos is replaced by table sampling and preprocessing now uses float. Remaining passes and startup work still require measurement. Table cleanup is F141; rejected-work accounting is F153. |
| F039–F042 | **Open, unmeasured:** repeated mesh/shard scans, large raster/pyramid memory, repeated record search, and morphology work remain. Multi-threshold evaluation does not establish an asymptotic optimization. |
| F043 | **Superseded in original form:** prediction now has threaded read-ahead and device preprocessing. Normal slot reuse appears ordered; new error-path lifetime issue is F136. |
| F044 | **Open:** `min(G,C)` still need not divide C; width 10 / G8 can compute group indexes beyond statistics (`unet.c:83`, `nn.cu:1905`). |
| F045 | **Partial:** tensor-core shared B sizing includes prior width; FP32 bottom/per-level sizing still omits it. Decreasing widths `{64,8}` and FP32 down-norm paths remain unsafe (`unet.c:335–339,605–607`). |
| F046 | **Open:** one-level models build no decoder but head still reads `dec[0]` (`unet.c:518–519,624–629`). |
| F047–F050 | **Open:** header parsing lacks validation; compatibility checks only `np`; missing/truncated moments can succeed; long header text can be read as weights (`unet.c:833–877`). |
| F051 | **Open:** GPU loss finalization still has `cnt[16]` indexed by arbitrary C (`nn.cu:2642–2652`). Default two outputs avoid it. |
| F052 | **Fixed/superseded:** old debug activation fields/copies removed. New optional parameter leaks are F127. |
| F053 | **Open:** fixed 1 MiB scratch; fused GN needs `2*N*C+2*N*G` floats (`nn.cu:1978–1980`), exceeding it when G=C and N*C>65536. |
| F054–F055 | **Open:** unchecked checkpoint close precedes rename; norm reduction still casts double partials to float (`unet.c:829`, `nn.cu:2731–2747`). |
| F056 | **Substantially superseded:** header now documents BF16/default and precision modes; stale TF32 naming and “bf16 tensor cores” trainer log still misdescribe some modes. It is no longer the original undisclosed-backend finding. |
| F057 | **Partial/API risk:** scratch is per device, fixing ordinary sequential two-GPU selection. Device keys use `device &7` and can alias IDs 0/8; global policy/registry/staging remain unsafe for concurrent callers. |
| F058–F060 | **Open/API risks:** synchronous header promise differs from enqueue/check semantics; input channels are not validated; workspace maximum omits head. Wide/one-level/shape capability gaps remain. |
| F061 | **Fixed structurally:** loss statistics now use multiple slabs (`nn.cu:2681`). No speed improvement was measured by reviewers. |
| F062 | **Partial:** fused tensor-core GN computes group sums once; FP32 GN still repeats the group loop per voxel (`nn.cu:2127`). |
| F063 | **Substantially reworked:** backward/replay operations have profiler wrappers. Current device/event defect is F139. |
| F064 | **Partial:** tensor-core skip reads in place; FP32 retains a separate copy for each batch item (`unet.c:510–511`). |
| F065 | **Partial:** fresh tensor-core inference shares temporaries and has no training-only allocation. A training→inference call retains its training build; this may intentionally avoid reallocating during validation. |
| F066 | **Partial but open:** MX/recompute helpers explicitly reject nonfinite error results. `test_nn`/`test_fused` max comparisons still ignore NaNs; new `test_unet` L2 check prints FAIL for NaN but `NaN>=0.05` does not increment failure (`test_unet.c:158–162`). Finite-difference comparisons have the same gap; latest Muon adds F149. |
| F067 | **Improved/incomplete:** whole-model gradient assertion added, subject to F066. EMA remains print-only. Packed updates, transition/resume and complete precision graph coverage are missing (F142). |
| F068 | **Partial:** tensor correctness path zeroes wgrad/bias buffers; FP32 benchmark skips that initialization, then accumulates into them. Init/errors can still return success (`bench_conv.c:26,44,51,58–59`). |
| F069 | **Open, reduced:** ordinary network test still includes full-model comparisons and training/inference timing, now defaulting to P96. |
| F070–F082 | **Open:** primary store/Zarr/writer code is byte-identical to the initial cut; secondary Zarr only adds prefetch/counters. Range/status/cache/geometry/index/decode/flush/CRC initialization defects remain in the original paths. |
| F083 | **Narrowed/API concurrency risk:** `json_path` still uses shared `strtok` state. The new global source-open mutex serializes the original sampler metadata-open path, so that specific old interleaving claim is superseded. Other concurrent library callers are not protected. |
| F084–F101 | **Open or originally conditional:** HTTP/CURL lifecycle, ZIP/TIFF validation, JSON boundaries/grammar/nesting, lost errors, unchecked resources, marker leaks and writer padding paths are unchanged. F127/F133/F136 add concrete new exposures. |
| F102–F104 | **Open, unmeasured:** shard-index preparation is serial; per-read workers/decoder scratch are still recreated; V2 metadata-size request behavior remains. |
| F105–F106 | **Open:** format fixture/open failures can count as skips; presence of `UFSM_NET=0` still activates network tests. Relevant test files are unchanged. |

### Adversarial checks that did not establish another defect

The reviewers challenged the following paths and rejected the initial suspicion or narrowed it instead of counting it as a confirmed bug:

- Default-stream ordering consumes incoming gradients before shared A/B or skip/gout storage is overwritten. Chunked output indexing uses the total channel stride and destination offset. No direct default-path alias/chunk-offset defect was established.
- Shared fresh-inference temporaries appear dead before overwrite in the reviewed order. Retaining a training build for frequent validation is a memory tradeoff, not automatically an incorrect lifetime.
- General MX split output now uses padded destination rows and correct real-row mapping. F124 identifies the uncovered small specialization; it does not reinstate the fixed general defect.
- MX normal exponent decoding and zero-block scales are internally consistent. Sixteen-channel row loads zero their upper lanes. FP4's final padded tap and small FP8's final padded tap have zero weights, making the reused input value harmless for those padded products.
- Reviewed vectorized loads obey width divisibility/alignment and paired stores fall back for odd widths. The MX GN backward fixed arrays have an explicit channel guard. These observations do not certify all shapes or instruction fragment mappings.
- The normal finite-scale FP16 partial-accumulation product bound is internally consistent. No off-by-one overflow in that stated bound was found; storage overflow/replay are separately F113/F114.
- Fused upsampling intentionally uses 16-bit kernels under FP8/FP4 policy. This is an effective-policy exception that must be reported, rather than a newly proven accidental selection bug.
- General QAT/saturation STE choices were not treated as automatically incorrect. F112/F113 show concrete operator/replay inconsistencies beyond disagreement about which surrogate derivative to choose.
- Primary missing chamfer body-diagonal neighbors and primary high-level prefetch coordinate conversion were corrected in the secondary cut. They are not active findings in that cut.
- The secondary sparse-bitmap wait mismatch was corrected in the final cut (F128). Zero-worker/thread-failure cases remain. Float preprocessing and opt-in storage trace changes did not yield another proven normal-input defect.
- Muon's workspace regions and X/Y swaps are correctly partitioned for positive supported shapes. Zero momentum/gradient remains zero safely; rank deficiency is preserved as expected. Classic Nesterov recurrence was not found incorrect merely because it differs from a lerp expression.
- Batched Muon uses per-descriptor bounds for ordinary matrix tails, and the five-step swap selects the final Y buffer. No new default fixed-mode indexing/swap defect was established; the mode-switch lifetime problem is F154.
- Vendored volcomp/surfcomp integration contains mode/length/capacity and, for surfcomp files, CRC/range checks. No new concrete transform/entropy-codec defect was established; those algorithms were not formally proved or executed. Existing container/storage integrity findings still apply.

This review did not inspect full external label arrays, prove actual cross-source held-out voxel overlap, measure failure frequency, establish convergence/topology, or validate hardware throughput. Its strongest conclusions are source control-flow, interval ownership, indexing, buffer-size/type, and state-transition proofs. All implementation remedies remain unimplemented by the audit team.

## Historical audit — 2026-09-30

The following original entries are preserved for traceability. Their scope, execution disclosure, and snapshot references apply to that earlier review only. The new 2026-10-01 review above executed no code.

Date: 2026-09-30. Reviewers: coordinating agent plus three audit agents covering CUDA/model code, formats/storage, and sampling/ingestion/tools. No implementation changes were made by the audit team.

## Scope and evidence

This is a source review of the C/CUDA pipeline, build rules, tests, scripts, configurations, and relevant vendored-code integration paths. Findings explain the triggering inputs and control flow; they are not runtime reproductions. Performance observations are estimates from algorithms, allocations, and synchronization, not benchmark results.

The workspace had no Git commits and was being edited concurrently by another process. Most findings refer to the frozen source snapshot at `/tmp/ufsm-audit-8_vefo68`; its file hashes are recorded in [snapshot-hashes.json](/tmp/ufsm-audit-8_vefo68/snapshot-hashes.json). The frozen `src/nn.cu` SHA-256 is `72f2b41bdf371a8b737ccfeedab346c0af5c3ef3428484b458dc808d2df18ab9`. Subsequent source comparisons identified external changes in `src/nn.cu`, `src/nn.h`, `src/unet.c`, `tests/bench_conv.c`, `tests/test_nn.c`, and `tests/test_unet.c`. The later CUDA rewrite is explicitly distinguished below; it was inspected selectively and did not receive a complete second audit. Line numbers refer to the snapshot unless labeled **live source**; links to unchanged files use the workspace path. Snapshot findings describe that captured version and must be rechecked against later revisions.

Before the user clarified “don't run any code / source only review,” the coordinator attempted a build in the isolated snapshot, the CUDA reviewer compiled a temporary object and inspected its resource output, and the formats reviewer compiled a temporary library and probed dependency availability. These had finished when the clarification arrived. No project tests, inference/training runs, reproducer harnesses, or benchmarks were executed by the audit team. All subsequent work consisted of source inspection and writing this document.

Priorities: **P1** = serious correctness, safety, data-integrity, or build blocker; **P2** = conditional correctness/reliability issue or material performance concern; **P3** = lower-impact defect, coverage gap, or optimization opportunity. These are review priorities, not measured failure rates.

There are 106 numbered review entries, including conditional API risks, unmeasured performance observations, and test-harness issues. The first issues to resolve before relying on training/evaluation outputs are:

| Area | Findings | Main consequence |
| --- | --- | --- |
| Validation memory | F001 | Default validation writes millions of bytes into 104 bytes of scratch |
| Sampler reliability/concurrency | F014, F015, F083 | Permanent hangs and races during normal threaded sampling |
| Evaluation identity/alignment | F004, F018, F019, F020 | Scores can use wrong coordinates, stale predictions, or training regions |
| Ingestion geometry | F016, F017, F034 | Misplaced labels and out-of-bounds reads/writes |
| Checkpoint loading/publication | F047–F050, F054 | Unsafe headers, incompatible loads, incomplete resumes, or lost checkpoints |
| Storage integrity | F070–F081 | Wrong HTTP ranges, suppressed failures, mixed caches, and stale output voxels |
| Format safety | F087, F089, F091, F092 | Invalid buffers or silent corruption from incomplete/malformed inputs |
| Clean test build | F002 | Formats target has unresolved neural-network symbols |

Representative snapshot fingerprints, retained here even if the temporary snapshot is later removed:

| File | SHA-256 |
| --- | --- |
| `src/train.c` | `96a0dab3cf5fb58d57e6ac7769bba17a78257a2831b95ae4318c1e0474961789` |
| `src/unet.c` | `6cd6cf07a0eb6a781a4504be3481a1f396ed64ca212755ceb5c665a78bced750` |
| `src/nn.cu` | `72f2b41bdf371a8b737ccfeedab346c0af5c3ef3428484b458dc808d2df18ab9` |
| `Makefile` | `ce2b2a40dc9f5cecc810922f58f9e0b071afd919ac410dd088a7f2ff76df48a3` |

## Training, prediction, evaluation, and CLI

### F001 — P1: Validation writes a full gradient tensor into a tiny scratch allocation

Locations: [src/train.c:40](/home/forrest/ufsm/src/train.c:40), [src/train.c:83](/home/forrest/ufsm/src/train.c:83), [src/train.c:134](/home/forrest/ufsm/src/train.c:134), [snapshot src/nn.cu:704](/tmp/ufsm-audit-8_vefo68/src/nn.cu:704), [snapshot src/nn.cu:728](/tmp/ufsm-audit-8_vefo68/src/nn.cu:728).

Validation calls `run_batch` with `gl == NULL`, which passes `scratch` as the loss gradient destination and `scratch + 8` as loss statistics. `nn_loss` always launches a kernel that writes `B * NCH * P^3` floats into that destination. With the default `B=1`, `NCH=2`, `P=96`, the allocation is only `40 + 64 = 104` bytes, but gradient writes require **7,077,888 bytes**. The destination also overlaps the statistics that the kernel reads. The first validation can corrupt device memory or fail with an illegal access; reported validation values cannot be trusted after this call.

Suggested remedy: provide a correctly sized, separate gradient destination or a loss-only API that skips the gradient kernel. Keep statistics and gradients disjoint.

### F002 — P1: The formats test target cannot link with its declared dependencies

Locations: [Makefile:11](/home/forrest/ufsm/Makefile:11), [Makefile:40](/home/forrest/ufsm/Makefile:40), [Makefile:49](/home/forrest/ufsm/Makefile:49).

`build/test_formats` links all `$(OBJ)`, including `sample.o`, `train.o`, `unet.o`, and `predict.o`, which call the `nn_*` API. The target includes neither `build/nn.o` nor `$(CUDALIBS)`. A clean `make test` therefore has unresolved neural-network symbols before the tests can run. The isolated build attempted before the source-only restriction independently produced those linker errors.

Suggested remedy: link only the formats dependencies, or explicitly include the CUDA implementation and runtime. Verify the clean-build dependency graph after a future fix.

### F003 — P1: CLI numeric options reach unsafe arithmetic without validation

Locations: [src/train.c:55](/home/forrest/ufsm/src/train.c:55), [src/train.c:65](/home/forrest/ufsm/src/train.c:65), [src/train.c:128](/home/forrest/ufsm/src/train.c:128), [src/train.c:145](/home/forrest/ufsm/src/train.c:145), [src/predict.c:30](/home/forrest/ufsm/src/predict.c:30), [src/predict.c:62](/home/forrest/ufsm/src/predict.c:62), [src/main.c:267](/home/forrest/ufsm/src/main.c:267), [src/eval.c:50](/home/forrest/ufsm/src/eval.c:50), [src/ingest.c:128](/home/forrest/ufsm/src/ingest.c:128).

Examples established by inspection:

- `--log-every 0`, `--val-every 0`, or `--ckpt-every 0` reaches integer remainder by zero during training.
- `predict --window 96 --halo 48` makes the tile stride zero and leaves the tile loops non-advancing; larger halos make them move backward. Negative halos also permit negative window indices in the output-copy loops.
- Zero/negative patch, batch, or box dimensions are accepted into products, allocations, reads, and GPU launches. Large products can overflow before allocation or indexing.
- Empty widths or negative/excessive level values reach shifts such as `1 << (nlev - 1)` and `1 << level`; excessive pyramid counts overrun the fixed `double um[MAXLEV_PYR]` array.

Suggested remedy: validate dimensions, option ranges, positive intervals, `0 <= halo < window/2`, level bounds, and checked allocation/index products before constructing any model or output store. Model-specific width constraints are detailed separately below.

### F004 — P1: Evaluation ignores the prediction origin recorded by `predict`

Locations: [src/predict.c:58](/home/forrest/ufsm/src/predict.c:58), [src/eval.c:34](/home/forrest/ufsm/src/eval.c:34), [src/eval.c:46](/home/forrest/ufsm/src/eval.c:46), [src/eval.c:53](/home/forrest/ufsm/src/eval.c:53).

`predict --box` creates an array whose local origin is zero and records the CT crop origin in `ufsm.origin_zyx`. Evaluation's usage text suggests that attribute is used, but evaluation initializes the origin to zero and only changes it when `--pred-origin` is supplied. It never reads the attribute. Evaluating a cropped prediction without the manual option silently compares it to labels at the wrong coordinates and can print plausible, incorrect scores. The holdout script supplies the option, so that particular caller avoids this omission.

Suggested remedy: read and validate prediction origin/scale metadata by default, define coordinate units explicitly, and allow an intentional override.

### F005 — P2: Ignored predictions can inflate band-tolerant recall

Locations: [src/eval.c:69](/home/forrest/ufsm/src/eval.c:69), [src/eval.c:74](/home/forrest/ufsm/src/eval.c:74), [src/eval.c:76](/home/forrest/ufsm/src/eval.c:76).

`pr[i]` is populated before the `l[i] == 255` validity check. Dilation subsequently includes every positive prediction, including predictions in ignored voxels. A prediction exclusively in an ignored voxel within tolerance of a valid surface is omitted from precision accounting but can credit that surface for recall. The band metrics consequently use different validity rules on their two sides.

Suggested remedy: define whether ignored predictions participate in matching, then apply that rule consistently to both dilations and denominators. Under the current “valid voxels” contract, mask predictions before dilation.

### F006 — P2: Prediction uses hard tile interiors instead of the documented Gaussian blend

Locations: [DESIGN.md:56](/home/forrest/ufsm/DESIGN.md:56), [src/predict.c:62](/home/forrest/ufsm/src/predict.c:62), [src/predict.c:76](/home/forrest/ufsm/src/predict.c:76), [src/predict.c:84](/home/forrest/ufsm/src/predict.c:84), [src/predict.c:107](/home/forrest/ufsm/src/predict.c:107).

Each tile is normalized independently, run through GroupNorm, and copied directly into its non-overlapping interior. There is no Gaussian weighting or accumulation buffer. A finite halo does not remove the global dependence introduced by per-window normalization and GroupNorm, so neighboring interiors can disagree across tile boundaries. Tile origins also restart at each output shard. This is a source-confirmed discrepancy from the design; the magnitude of seams was not measured.

Suggested remedy: implement the stated overlap/blend contract or document the hard-interior behavior and qualify it with boundary-consistency tests. Use a volume-wide tile grid if output should be independent of shard partitioning.

### F007 — P2: Axis generation cannot open the project's physically named OME levels

Locations: [src/main.c:102](/home/forrest/ufsm/src/main.c:102), [src/main.c:107](/home/forrest/ufsm/src/main.c:107), [src/z3w.c:45](/home/forrest/ufsm/src/z3w.c:45), [src/sources.c:166](/home/forrest/ufsm/src/sources.c:166).

`cmd_axis` reads the OME level list but ignores each dataset's path and attempts `group/0`, `group/1`, etc. The writer and published layout name levels by voxel size, such as `2.399` and `4.798`. An otherwise readable group therefore fails axis generation with “no readable level.” The `2^level` coordinate scaling also assumes the list index corresponds to a native-resolution power-of-two level.

Suggested remedy: use the OME paths and declared scales, retaining integer-path fallback only for groups that use that layout.

### F008 — P2: Logged/validated loss can differ from the objective being differentiated

Locations: [src/train.c:42](/home/forrest/ufsm/src/train.c:42), [snapshot src/nn.cu:717](/tmp/ufsm-audit-8_vefo68/src/nn.cu:717), [snapshot src/nn.cu:725](/tmp/ufsm-audit-8_vefo68/src/nn.cu:725), [snapshot tests/test_nn.c:235](/tmp/ufsm-audit-8_vefo68/tests/test_nn.c:235).

The loss kernel normalizes gradients over active `(sample, channel)` pairs. `run_batch` reconstructs its scalar by equally averaging active per-channel means, without weighting by the number of active samples in each channel. For `B>1` and channel counts of two and one, training differentiates `(2*L0 + L1)/3`, while the logged/validation value is `(L0 + L1)/2`. The existing loss test explicitly applies the sample counts when reconstructing its scalar, illustrating the mismatch. This can change which checkpoint is considered best.

Suggested remedy: return the actual scalar objective, or expose active counts per channel and use the same weighting in training and validation.

### F009 — P2: Validation cadence is accidentally coupled to logging cadence

Locations: [src/train.c:128](/home/forrest/ufsm/src/train.c:128), [src/train.c:132](/home/forrest/ufsm/src/train.c:132).

Validation is nested inside the logging condition. With `--log-every 20 --val-every 30`, validation occurs at multiples of 60 rather than 30, plus the final step. Other combinations can postpone validation and best-checkpoint selection far beyond the requested interval. The default intervals happen to be compatible.

Suggested remedy: schedule validation and logging independently, or reject/document incompatible intervals.

### F010 — P1: Training/output callers discard persistence failures

Locations: [src/train.c:138](/home/forrest/ufsm/src/train.c:138), [src/train.c:147](/home/forrest/ufsm/src/train.c:147), [src/predict.c:127](/home/forrest/ufsm/src/predict.c:127), [src/ingest.c:725](/home/forrest/ufsm/src/ingest.c:725), [src/main.c:59](/home/forrest/ufsm/src/main.c:59).

Training ignores `unet_save` return values, and advances `best_val` before knowing the best checkpoint was saved. Prediction, label/ZIP/Kaggle ingest, and rasterization ignore final group metadata failures; Kaggle ingest also tolerates failure to create the `cubes.json` index needed by source generation ([src/ingest.c:343](/home/forrest/ufsm/src/ingest.c:343)). PGM writers likewise ignore short writes and close errors. Full disks or failed renames can leave missing/stale artifacts while the command reports success. Lower-level close/flush handling is another persistence issue documented in the formats findings.

Suggested remedy: propagate and report write/close/rename failures; update best-checkpoint state only after a successful save and return a failure status for incomplete outputs.

### F011 — P2: Metadata strings are interpolated into JSON without escaping or truncation checks

Locations: [src/predict.c:55](/home/forrest/ufsm/src/predict.c:55), [src/predict.c:58](/home/forrest/ufsm/src/predict.c:58), [src/z3w.c:157](/home/forrest/ufsm/src/z3w.c:157).

Checkpoint paths, source paths, and names are written through `%s` inside JSON strings. Quotes, backslashes, or control characters in an otherwise valid filesystem path produce invalid JSON. Prediction's fixed 1,024-byte attributes buffer can also truncate the closing JSON when paths are long, and that invalid string is inserted into array and group metadata. The recorded `source` additionally uses an invented `key/levelN` path rather than the array path actually opened. The writer's textual attributes merge ([src/z3w.c:164](/home/forrest/ufsm/src/z3w.c:164)) only checks the first two characters: valid empty attributes `{ }` are treated as nonempty and produce an invalid trailing comma.

Suggested remedy: serialize escaped strings through a JSON writer, size output dynamically or reject truncation, and record the actual selected array key and coordinate level.

### F012 — P2: Training output-directory creation treats a filesystem path as shell syntax

Location: [src/train.c:67](/home/forrest/ufsm/src/train.c:67).

`system("mkdir -p '%s'")` wraps `--out` in single quotes without escaping embedded single quotes. An output directory such as `runs/forrest's-run` breaks directory creation; crafted text can execute additional shell commands. No shell is needed for this operation.

Suggested remedy: create directories using filesystem calls, as the Zarr writer already does.

### F013 — P3: Evaluation thresholds are rounded downward

Locations: [src/eval.c:66](/home/forrest/ufsm/src/eval.c:66), [src/eval.c:69](/home/forrest/ufsm/src/eval.c:69).

Casting `threshold * 255` to `uint8_t` truncates. At the reported threshold `0.50`, a stored value of 127 (`127/255 ≈ 0.4980`) is counted positive. The correct integer boundary for `p/255 >= 0.5` is 128. The same downward bias affects 0.30 and 0.70.

Suggested remedy: compare normalized probabilities or use the ceiling of the scaled threshold, and document the treatment of quantized values.

## Sampling, ingestion, configurations, and tools

### F014 — P1: Sampler retry and shutdown loops can hang permanently

Locations: [src/sample.c:304](/home/forrest/ufsm/src/sample.c:304), [src/sample.c:307](/home/forrest/ufsm/src/sample.c:307), [src/sample.c:342](/home/forrest/ufsm/src/sample.c:342), [src/sample.c:364](/home/forrest/ufsm/src/sample.c:364).

The inner batch-fill loop never checks `stop`. Each I/O failure increments `fail` and immediately resets it to zero, making `fail > 20` unreachable. Persistent read errors, impossible patch geometry, or a distribution where every patch is rejected can spin indefinitely. `sampler_next` waits forever, and `sampler_stop` cannot finish joining a worker even after setting the stop flag.

Suggested remedy: check cancellation within the draw loop, maintain real error/rejection budgets, surface terminal errors, and never publish an incomplete batch.

### F015 — P1: Lazy source handles and presence flags are shared without synchronization

Locations: [src/sources.c:181](/home/forrest/ufsm/src/sources.c:181), [src/sources.c:190](/home/forrest/ufsm/src/sources.c:190), [src/sources.c:201](/home/forrest/ufsm/src/sources.c:201), [src/sample.c:334](/home/forrest/ufsm/src/sample.c:334).

Workers concurrently read/write each source's CT/target handles, presence flags, region handles, and shared region array. There is no locking, atomic publication, or full preload before the workers start. Concurrent first accesses can overwrite opened handles, leak objects, race success against failed probes, and invoke undefined behavior in C. JSON lookup has a separate concurrency defect below.

Suggested remedy: initialize immutable source state before worker creation or synchronize each lazy initialization and publication.

### F016 — P1: Anisotropic chunks cause shard misgrouping and destination underflow during ingest

Locations: [src/ingest.c:165](/home/forrest/ufsm/src/ingest.c:165), [src/ingest.c:201](/home/forrest/ufsm/src/ingest.c:201), [src/ingest.c:220](/home/forrest/ufsm/src/ingest.c:220), [src/ingest.c:464](/home/forrest/ufsm/src/ingest.c:464), [src/ingest.c:528](/home/forrest/ufsm/src/ingest.c:528).

Input validation allows each chunk dimension to divide the shard size, but both label ingest and ZIP ingest use `shard / chunk[0]` for grouping on every axis. For chunks `[128,32,32]`, shard 128, and chunk coordinate `[0,0,1]`, grouping chooses output shard x=1 although the chunk starts at x=32. The copy computes `gx=32-128=-96` and writes before the shard allocation because negative destination x is not rejected. Other anisotropic arrangements silently drop or misplace data.

Suggested remedy: calculate grouping independently per axis and clip/check all destination coordinates before pointer arithmetic. This finding is destination underflow/misgrouping, not a claim of selection-list overflow for unique chunk coordinates.

### F017 — P1: Direct TIFF mesh loading trusts incompatible dimensions/types and failed decodes

Location: [src/ingest.c:580](/home/forrest/ufsm/src/ingest.c:580).

`load_mesh` takes the grid size from x.tif, independently allocates y/z arrays from their own dimensions, then indexes all three using x's count. Smaller y/z pages cause out-of-bounds reads. Four bytes per pixel are allocated without requiring float32 and one sample per pixel, so larger layouts can overflow that allocation or be misinterpreted. Metadata/decode return values are ignored.

Suggested remedy: require matching float32 single-channel page layouts and successful reads before constructing any mesh coordinates.

### F018 — P1: Holdout prediction reuse ignores checkpoint, volume, crop, level, and settings

Locations: [tools/eval_holdouts.py:20](/home/forrest/ufsm/tools/eval_holdouts.py:20), [tools/eval_holdouts.py:29](/home/forrest/ufsm/tools/eval_holdouts.py:29), [tools/eval_holdouts.py:38](/home/forrest/ufsm/tools/eval_holdouts.py:38), [tools/make_sources.py:26](/home/forrest/ufsm/tools/make_sources.py:26), [tools/make_sources.py:100](/home/forrest/ufsm/tools/make_sources.py:100), [tools/make_sources.py:124](/home/forrest/ufsm/tools/make_sources.py:124).

The default output identifies only the checkpoint parent directory, and each prediction directory identifies only the source name. Any existing `zarr.json` skips prediction without checking provenance. Changing `last.ckpt` to `best.ckpt`, replacing a checkpoint, changing level/box/CT, or changing settings silently reuses stale predictions. The generator also gives three PHerc0139 HF sources the same name and gives different segment volumes of one scroll the same `<scroll>-seg` name, creating collisions within a single run.

Suggested remedy: assign unique source identities and bind reuse to verified checkpoint content/version, CT/target identity, crop, scale, and prediction settings.

### F019 — P1: Region-source validation does not enforce its holdout

Locations: [src/sample.c:151](/home/forrest/ufsm/src/sample.c:151), [src/sample.c:160](/home/forrest/ufsm/src/sample.c:160), [src/sample.c:173](/home/forrest/ufsm/src/sample.c:173), [src/sample.h:17](/home/forrest/ufsm/src/sample.h:17).

The region coordinate-selection branch precedes the holdout branch. A region source with a configured holdout therefore validates anywhere in its regions, while training excludes only the holdout box. Validation can draw coordinates also used for training, contradicting the sampler's documented holdout contract.

Suggested remedy: select validation corners from the intersection of each region and its holdout, and reject impossible intersections promptly.

### F020 — P2, conditional: Physical holdouts are not shared across duplicate CT sources

Locations: [tools/make_sources.py:26](/home/forrest/ufsm/tools/make_sources.py:26), [tools/make_sources.py:100](/home/forrest/ufsm/tools/make_sources.py:100), [src/sample.c:173](/home/forrest/ufsm/src/sample.c:173).

Three generated PHerc0139 sources point at the same CT but have separate targets and holdouts. Training excludes only the currently chosen source's box. A voxel held out for source A can be drawn through source B. Actual supervised leakage depends on target overlap/ignore coverage, which was not inspected.

Suggested remedy: apply the union of held-out regions to every source sharing the same physical CT identity.

### F021 — P2: One channel's ignore label suppresses valid supervision in other channels

Locations: [src/sample.c:224](/home/forrest/ufsm/src/sample.c:224), [src/sample.c:250](/home/forrest/ufsm/src/sample.c:250), [src/sample.h:23](/home/forrest/ufsm/src/sample.h:23), [snapshot src/nn.h:72](/tmp/ufsm-audit-8_vefo68/src/nn.h:72).

The sampler ORs ignore labels across active channels into a single `[B,S]` mask. With CT>0, recto=254, sheet=255, and both channel weights enabled, the valid recto target loses supervision. Disjoint channel coverage can remove every supervised voxel. The loader accepts independent target stores and does not enforce identical ignore support. Single-target sources avoid this issue.

Suggested remedy: use per-channel masks, sample channels independently, or explicitly validate a restricted common-mask contract.

### F022 — P2: Only the last configured region channel is sampled

Locations: [src/sample.c:153](/home/forrest/ufsm/src/sample.c:153), [src/sample.c:199](/home/forrest/ufsm/src/sample.c:199), [src/sources.c:97](/home/forrest/ufsm/src/sources.c:97).

When both recto and sheet use regions, the channel-selection loop overwrites `region_ch` with sheet. The recto region is never read/activated; it falls through to an empty target because it has no pyramid key. The configuration is accepted without warning.

Suggested remedy: support all compatible region targets or reject unsupported multi-region combinations explicitly.

### F023 — P2: Global cache configuration borrows freed memory and defeats its documented override

Locations: [src/sources.c:8](/home/forrest/ufsm/src/sources.c:8), [src/sources.c:83](/home/forrest/ufsm/src/sources.c:83), [src/sources.c:161](/home/forrest/ufsm/src/sources.c:161), [src/sources.c:218](/home/forrest/ufsm/src/sources.c:218), [src/sources.h:72](/home/forrest/ufsm/src/sources.h:72).

`g_cache` borrows `S->cache`, but freeing the collection does not clear the global. A later collection without a cache or subsequent source opening can use a dangling pointer. Loading a file with a cache also replaces an earlier `sources_set_cache` override despite the header instructing callers to set that override before loading. Multiple collections cannot independently own cache settings.

Suggested remedy: keep owned cache state with each collection/source and define override precedence explicitly.

### F024 — P2: Transient metadata errors permanently remove source levels/channels

Locations: [src/sources.c:181](/home/forrest/ufsm/src/sources.c:181), [src/sources.c:190](/home/forrest/ufsm/src/sources.c:190), [src/sample.c:126](/home/forrest/ufsm/src/sample.c:126), [src/sample.c:213](/home/forrest/ufsm/src/sample.c:213).

Every failed level open sets its presence flag to zero, including transport, permission, parsing, and resource failures. That level is never retried; a failed target probe can silently remove a channel from training. Excluding all candidates combines with F014 to hang the sampler.

Suggested remedy: distinguish confirmed absence from operational failure, propagate errors, and use deliberate retry policies instead of changing the sampling distribution silently.

### F025 — P2: Voxel-size inference cannot discover physically named native levels

Locations: [src/sources.c:126](/home/forrest/ufsm/src/sources.c:126), [src/sources.c:166](/home/forrest/ufsm/src/sources.c:166), [src/z3w.c:45](/home/forrest/ufsm/src/z3w.c:45).

When a source omits `um`, loading asks `source_ct(s,0)` to infer it. With unknown `um0=0`, `pyramid_open_level` bypasses OME path matching and tries `<group>/0`, even when metadata identifies a native array named `2.399`. Valid project-exported pyramids therefore fail the advertised inference path.

Suggested remedy: select the native/first declared dataset to discover base voxel size before matching requested physical levels.

### F026 — P2: Ingest ignores listing failures and can export only a partial chunk set

Locations: [src/ingest.c:145](/home/forrest/ufsm/src/ingest.c:145), [src/ingest.c:204](/home/forrest/ufsm/src/ingest.c:204), [src/ingest.c:242](/home/forrest/ufsm/src/ingest.c:242).

Top-level and recursive `store_list` results are ignored. A listing that fails after yielding some chunks leaves a nonempty list; ingest exports it, builds pyramids, and reports success. Omitted labels become ignore fill and omitted CT becomes zero fill.

Suggested remedy: require successful completion of every enumeration before publishing completed output metadata.

### F027 — P2: Pyramid building treats an input-shard error as absence

Locations: [src/ingest.c:90](/home/forrest/ufsm/src/ingest.c:90), [src/ingest.c:124](/home/forrest/ufsm/src/ingest.c:124), [src/zarr3.h:34](/home/forrest/ufsm/src/zarr3.h:34).

`z3_shard_present` returns -1 on error, but `pres <= 0` skips the shard without setting failure. Existing unreadable data can silently become fill in a successfully reported pyramid.

Suggested remedy: skip only a confirmed absent shard and propagate negative error results.

### F028 — P2: ZIP ingest ignores storage order and filters

Locations: [src/ingest.c:496](/home/forrest/ufsm/src/ingest.c:496), [src/ingest.c:455](/home/forrest/ufsm/src/ingest.c:455), [src/zarr2.c:56](/home/forrest/ufsm/src/zarr2.c:56).

ZIP ingest validates dtype/compressor but neither implements nor rejects `order:"F"` or nonempty filters. It interprets all decoded bytes as ordinary C-order uint8. The ordinary V2 reader at least rejects filters, so the two ingestion paths differ.

Suggested remedy: validate the entire relevant `.zarray` contract and reject unsupported layouts/filters before conversion.

### F029 — P3: S3 listing assigns the previous object's size to later keys

Location: [src/hf.c:96](/home/forrest/ufsm/src/hf.c:96).

After reading a key, `p` advances past the key but remains before its size. On the next iteration the next key is found, while the size search still finds the preceding object's size. The first size is correct; later reported sizes lag one object.

Suggested remedy: parse each `<Contents>` record as a unit and associate its key and size within that boundary.

### F030 — P2, conditional: S3 query and XML text handling fails for escaped/significant characters

Locations: [src/hf.c:70](/home/forrest/ufsm/src/hf.c:70), [src/hf.c:89](/home/forrest/ufsm/src/hf.c:89), [src/hf.c:120](/home/forrest/ufsm/src/hf.c:120).

Continuation tokens and prefixes are interpolated into URL queries without percent encoding, and XML text is returned without decoding entities. Tokens with query-significant characters or escaped object keys can break pagination or identify the wrong object. Actual server responses were not fetched.

Suggested remedy: encode query values and decode XML text with bounded, format-aware parsing.

### F031 — P2: Source generation excludes raster data based on HF entries that were never included

Locations: [tools/make_sources.py:91](/home/forrest/ufsm/tools/make_sources.py:91), [tools/make_sources.py:106](/home/forrest/ufsm/tools/make_sources.py:106), [tools/make_sources.py:114](/home/forrest/ufsm/tools/make_sources.py:114).

`hf_vols` uses the entire static HF catalog, including exports skipped because labels or CT pairing were missing. A usable raster source for such a scan is then excluded even though no HF source was added.

Suggested remedy: derive the exclusion set from HF sources successfully added to the generated configuration.

### F032 — P2: The development configuration contradicts the GT-only design and may use incompatible targets

Locations: [configs/dev.json:1](/home/forrest/ufsm/configs/dev.json:1), [DESIGN.md:4](/home/forrest/ufsm/DESIGN.md:4), [src/sample.c:222](/home/forrest/ufsm/src/sample.c:222).

The shipped development config references `teacher_regions` and an `m7` probability prediction store, while the design excludes teacher models/distillation. The sampler assumes 254=surface and 255=ignore for every target. If that referenced probability store uses ordinary probability-times-255 encoding, confidence 1.0 is discarded as ignore. Store contents were not inspected, so the encoding mismatch is a conditional risk; this review does not assert that an actual training run used the config.

Suggested remedy: replace or explicitly label the legacy development config and require declared/validated target encodings.

### F033 — P2, conditional: Direct tifxyz rasterization omits the mask applied by mesh conversion

Locations: [src/ingest.c:577](/home/forrest/ufsm/src/ingest.c:577), [src/ingest.c:399](/home/forrest/ufsm/src/ingest.c:399), [src/ingest.c:425](/home/forrest/ufsm/src/ingest.c:425).

The direct tifxyz loader never reads `mask.tif`, whereas `ingest-mesh` uses it to construct SFC validity. For a segment with a mask, the two supported paths can rasterize different points and create different training labels.

Suggested remedy: apply the same mask/validity rules to direct TIFF and converted SFC input.

### F034 — P2: Cube ingest validates only the first TIFF page

Location: [src/ingest.c:282](/home/forrest/ufsm/src/ingest.c:282).

`read_cube` validates page 0's dimensions/bit depth/sample count, then decodes every page at offsets sized for that layout. A later page with larger geometry or sample width can overrun cube storage; smaller/incompatible pages can silently corrupt the layout.

Suggested remedy: validate every page against the expected cube layout before decoding it.

### F035 — P2, conditional: Coarse sampling can round a patch below a region/holdout boundary

Locations: [src/sample.c:166](/home/forrest/ufsm/src/sample.c:166), [src/sample.c:169](/home/forrest/ufsm/src/sample.c:169), [src/sample.c:208](/home/forrest/ufsm/src/sample.c:208).

Corners are chosen in level-0 coordinates and then right-shifted. An unaligned lower boundary can round downward: a region starting at 1 at level 1 can produce native patch origin 0 and region-local read origin -1. The shipped example region origins are aligned, so this is a generic-input defect.

Suggested remedy: choose corners on the level's grid, rounding the allowed lower bound upward and last valid corner downward.

### F036 — P2: Batch pipeline scripts finish successfully after failed operations

Locations: [tools/ingest_all.sh:14](/home/forrest/ufsm/tools/ingest_all.sh:14), [tools/ingest_segments.sh:10](/home/forrest/ufsm/tools/ingest_segments.sh:10), [tools/raster_all.sh:25](/home/forrest/ufsm/tools/raster_all.sh:25), [tools/surfcomp_all.sh:24](/home/forrest/ufsm/tools/surfcomp_all.sh:24).

Failed commands print `FAILED`, continue, and end with a successful unconditional `echo`. An orchestration caller cannot distinguish complete success from partial export/raster failures through exit status.

Suggested remedy: retain best-effort processing if desired, collect failures, and return a nonzero final status when required work failed.

### F037 — P2, conditional: Download resume validates total size but not object identity or each range

Locations: [tools/pget.sh:7](/home/forrest/ufsm/tools/pget.sh:7), [tools/pget.sh:15](/home/forrest/ufsm/tools/pget.sh:15), [tools/pget.sh:21](/home/forrest/ufsm/tools/pget.sh:21), [tools/pget.sh:29](/home/forrest/ufsm/tools/pget.sh:29).

Existing part files are reused by size alone. If an object changes between attempts, the final file can contain old and new ranges while passing the total-size check. A successful curl exit is also treated as completion without explicit HTTP status/Content-Range or expected part-length validation. No download was attempted during this review.

Suggested remedy: bind resume files to object identity/version/ETag and verify conditional range responses plus exact part lengths before assembly.

## Performance observations from the pipeline

All entries in this section are unmeasured. They identify work/allocation patterns to profile, not guaranteed speedups.

### F038 — P3: Sampler augmentation performs expensive scalar work for every voxel

Locations: [src/sample.c:22](/home/forrest/ufsm/src/sample.c:22), [src/sample.c:242](/home/forrest/ufsm/src/sample.c:242), [src/sample.c:266](/home/forrest/ufsm/src/sample.c:266).

A P=96 patch generates 884,736 normals through separate Box–Muller `log/sqrt/cos` calls, discarding the second normal. Radial normalization is also computed for every voxel even when no axis is configured. Paired/vectorized noise generation, avoiding absent-axis calculations, and fusing symmetry/vector copies are candidates for profiling.

### F039 — P2: Rasterization rescans complete meshes for each intersecting shard

Location: [src/ingest.c:645](/home/forrest/ufsm/src/ingest.c:645).

Mesh bounding boxes prune whole meshes, but every remaining quad is scanned again for each shard. Work grows with intersecting shard count times mesh quad count. Indexing quads or mesh tiles by spatial bounds would reduce repeated scanning; benefit depends on actual geometry.

### F040 — P2: Default raster/pyramid scratch memory is large and unbudgeted

Locations: [src/ingest.c:83](/home/forrest/ufsm/src/ingest.c:83), [src/ingest.c:119](/home/forrest/ufsm/src/ingest.c:119), [src/ingest.c:631](/home/forrest/ufsm/src/ingest.c:631), [src/ingest.c:679](/home/forrest/ufsm/src/ingest.c:679), [src/ingest.c:699](/home/forrest/ufsm/src/ingest.c:699).

At shard 1024 and default T=3, the expanded raster distance grid plus output buffer requires approximately 2.04 GiB per worker, or 16.3 GiB for eight workers before meshes/writer resources. Pyramid building allocates `out + in + piece` of 2.125 GiB per shard worker, up to 17 GiB for eight workers before decoder buffers. These are allocation estimates, not measured RSS. Unchecked allocations also make memory pressure a correctness risk.

Suggested remedy: budget concurrency against available memory, check allocations, reuse buffers, and consider smaller processing tiles.

### F041 — P3: Ingest repeatedly scans a z-band to find each output shard's chunks

Locations: [src/ingest.c:227](/home/forrest/ufsm/src/ingest.c:227), [src/ingest.c:535](/home/forrest/ufsm/src/ingest.c:535).

Both label and ZIP ingestion scan records in a z-band again for each y/x shard. Bucketing/sorting records by complete output-shard identity would avoid the repeated search as the shard grid grows.

### F042 — P3: Evaluation repeats full-volume work and uses a per-positive dilation loop

Locations: [src/eval.c:19](/home/forrest/ufsm/src/eval.c:19), [src/eval.c:74](/home/forrest/ufsm/src/eval.c:74), [src/eval.c:79](/home/forrest/ufsm/src/eval.c:79).

Dilation scans the box and loops over `(2*tol+1)^3` candidates for each positive voxel. It runs once for ground truth and once per prediction threshold. Soft Dice, which is threshold-independent, is recomputed three times. Whole-box allocation uses six byte-per-voxel arrays after both stores are loaded. Compute Dice once and consider bounded-memory/streamed evaluation and a more efficient distance/dilation method for larger boxes/tolerances.

### F043 — P2: Prediction serializes reading, preprocessing, transfers, and one-window inference

Locations: [src/predict.c:76](/home/forrest/ufsm/src/predict.c:76), [src/predict.c:80](/home/forrest/ufsm/src/predict.c:80), [src/predict.c:101](/home/forrest/ufsm/src/predict.c:101), [src/predict.c:121](/home/forrest/ufsm/src/predict.c:121).

Each window fully reads CT, preprocesses on the CPU, uploads, runs batch-one inference, downloads probabilities, and copies the interior before the next window starts. Shard encoding also finishes before advancing. There is no input prefetch/batch pipeline in this command. The default halo makes 96^3 inference retain only 64^3 output voxels, an interior compute ratio of 3.375 before boundary effects. Batching/prefetch and bounded asynchronous stages deserve profiling; halo cost must be balanced against prediction quality.

## Model, CUDA, and checkpoints

F001 also belongs to this section. The model/CUDA references here deliberately link to the frozen copy because these files changed externally during the review.

### F044 — P1: Accepted widths can make GroupNorm read beyond its statistics

Locations: [snapshot src/unet.c:52](/tmp/ufsm-audit-8_vefo68/src/unet.c:52), [snapshot src/unet.c:157](/tmp/ufsm-audit-8_vefo68/src/unet.c:157), [snapshot src/nn.cu:470](/tmp/ufsm-audit-8_vefo68/src/nn.cu:470), [src/train.c:64](/home/forrest/ufsm/src/train.c:64).

`G_of` chooses `min(G,C)` without requiring divisibility. Kernels then calculate `cpg=C/G` and group index `c/cpg`. With width 10 and G=8, channels 8 and 9 access statistics indices 8 and 9 even though only eight groups were allocated per sample. In larger batches this first reads another sample's statistics, then goes out of bounds on the final sample. Arbitrary `--widths`, including `10,16`, are accepted.

Suggested remedy: require positive groups dividing each width or choose a valid divisor bounded by the requested count.

### F045 — P1: Decreasing encoder widths overflow backward buffers

Locations: [snapshot src/unet.c:169](/tmp/ufsm-audit-8_vefo68/src/unet.c:169), [snapshot src/unet.c:190](/tmp/ufsm-audit-8_vefo68/src/unet.c:190), [snapshot src/unet.c:279](/tmp/ufsm-audit-8_vefo68/src/unet.c:279).

A level's gradient buffers are sized to its own width plus the next width, or only its own width at the bottom. Encoder input gradients instead have the preceding level's width. For a two-level `{64,8}` model, the bottom buffer holds eight channels but encoder backward writes 64 channels into it. These widths satisfy the default group divisibility and are accepted by training.

Suggested remedy: size workspace from every actual input/output shape that uses it, or explicitly constrain supported width sequences.

### F046 — P1: One-level networks use an uninitialized decoder for the head

Locations: [snapshot src/unet.c:177](/tmp/ufsm-audit-8_vefo68/src/unet.c:177), [snapshot src/unet.c:252](/tmp/ufsm-audit-8_vefo68/src/unet.c:252), [snapshot src/unet.c:288](/tmp/ufsm-audit-8_vefo68/src/unet.c:288).

With `nlev=1`, no decoder is built. The head still uses the zeroed `dec[0].ys` shape, and backward uses `dec[0].s2 == NULL`. A single width is accepted at the CLI, so this reaches invalid launches/pointers rather than a useful model or clear rejection.

Suggested remedy: attach the head to the encoder for this case or reject networks with fewer than two levels.

### F047 — P1: Checkpoint headers are accepted without required-field or range validation

Locations: [snapshot src/unet.c:341](/tmp/ufsm-audit-8_vefo68/src/unet.c:341), [snapshot src/unet.c:358](/tmp/ufsm-audit-8_vefo68/src/unet.c:358), [snapshot src/unet.c:70](/tmp/ufsm-audit-8_vefo68/src/unet.c:70), [src/predict.c:38](/home/forrest/ufsm/src/predict.c:38).

`read_header` reports success whenever the magic matches and `fgets` returns some text. It does not require valid JSON, required fields, a complete newline, a matching width count, or valid ranges. `UFSM{}\n` is accepted by `unet_peek` with zero dimensions/groups; `nlev > UNET_MAXLEV` reaches construction loops past fixed arrays. Width parsing can continue through commas outside the width array into other fields.

Suggested remedy: parse and validate a complete versioned header before returning a configuration or allocating a model.

### F048 — P1: Checkpoint compatibility checks only parameter count

Location: [snapshot src/unet.c:370](/tmp/ufsm-audit-8_vefo68/src/unet.c:370).

The parsed configuration is not compared with the receiving model. Changing G from 4 to 8 leaves parameter count unchanged, so weights load successfully but produce different normalization and predictions. Other count collisions can reinterpret parameter layouts.

Suggested remedy: compare all architecture/group fields and widths, plus format/version and parameter count, before applying weights.

### F049 — P1: Missing/truncated optimizer arrays are accepted as a successful resume

Locations: [snapshot src/unet.c:374](/tmp/ufsm-audit-8_vefo68/src/unet.c:374), [src/train.c:76](/home/forrest/ufsm/src/train.c:76), [src/train.c:123](/home/forrest/ufsm/src/train.c:123).

Short parameter/EMA reads fail, but short Adam m/v reads simply break and return the saved step. A checkpoint ending after EMA resumes at that step with old or zero optimizer moments and inappropriate bias correction. Earlier read failures can also partially mutate a model before returning failure.

Suggested remedy: validate all required arrays before mutating the model and fail an incomplete resume; explicitly version any supported optimizer-free format.

### F050 — P2: A long saved checkpoint header is read as binary parameters

Locations: [snapshot src/unet.c:332](/tmp/ufsm-audit-8_vefo68/src/unet.c:332), [snapshot src/unet.c:344](/tmp/ufsm-audit-8_vefo68/src/unet.c:344), [snapshot src/unet.c:374](/tmp/ufsm-audit-8_vefo68/src/unet.c:374).

Save accepts unrestricted `extra` JSON, but load reads only 4095 bytes and never verifies the newline. A valid header longer than that leaves text where binary parameters are expected. The early parameter-count field can pass, and the remaining file can be long enough for all binary reads to appear successful while weights are corrupted.

Suggested remedy: read the complete bounded header, reject oversized/incomplete metadata, and locate the binary payload explicitly.

### F051 — P2: Loss has an undocumented 16-channel stack limit

Locations: [snapshot src/nn.cu:716](/tmp/ufsm-audit-8_vefo68/src/nn.cu:716), [snapshot src/nn.cu:723](/tmp/ufsm-audit-8_vefo68/src/nn.cu:723), [snapshot src/nn.h:72](/tmp/ufsm-audit-8_vefo68/src/nn.h:72).

`cnt[16]` is indexed and read using arbitrary `s.c`. At C=17, the final loop reads beyond it, and an active channel 16 also writes beyond it. The public API specifies no channel limit. Default two-channel training avoids this.

Suggested remedy: size counts from the supplied channel count or validate/document a supported bound.

### F052 — P2, snapshot only; later removed: Debug buffers survive resizing and are not freed

Locations: [snapshot src/unet.c:142](/tmp/ufsm-audit-8_vefo68/src/unet.c:142), [snapshot src/unet.c:222](/tmp/ufsm-audit-8_vefo68/src/unet.c:222), [snapshot src/unet.c:96](/tmp/ufsm-audit-8_vefo68/src/unet.c:96).

`dbg_s1` is allocated only when null and is omitted from `free_acts`. With `UFSM_DEBUG`, a larger subsequent forward copies larger activations into the old allocation. Destruction also leaks it. Debug mode can therefore introduce device corruption precisely while investigating model problems.

Suggested remedy: include debug buffers in activation sizing, resizing, and cleanup.

Later-source status: the externally revised `unet.c` removed these debug buffer fields and copy paths. This particular snapshot issue is superseded by that removal; the audit team made no such change.

### F053 — P2: GroupNorm backward workspace is a fixed 1 MiB

Locations: [snapshot src/unet.c:90](/tmp/ufsm-audit-8_vefo68/src/unet.c:90), [snapshot src/nn.cu:543](/tmp/ufsm-audit-8_vefo68/src/nn.cu:543), [snapshot src/nn.cu:557](/tmp/ufsm-audit-8_vefo68/src/nn.cu:557).

The backend writes `2*N*C*sizeof(float)` bytes, but the model never sizes this scratch from batch/channel dimensions. A level with `N*C > 131072` exceeds the allocation; small spatial shapes can permit that batch/channel combination without otherwise exhausting device memory.

Suggested remedy: allocate the largest actual `nn_gn_scratch` requirement when activation shapes are built.

### F054 — P1: Checkpoint save can replace a good checkpoint despite flush failure

Location: [snapshot src/unet.c:330](/tmp/ufsm-audit-8_vefo68/src/unet.c:330).

Header writes and `fclose` are unchecked. Buffered parameter writes can succeed until close, after which the temporary file is still renamed over the destination. Disk exhaustion or delayed I/O failure can publish an incomplete checkpoint as successful, independently of the caller failures in F010.

Suggested remedy: verify writes and successful flush/close before replacing the destination, and preserve the previous good checkpoint on failure.

### F055 — P2: Double-return reductions cast block partials to float

Locations: [snapshot src/nn.cu:752](/tmp/ufsm-audit-8_vefo68/src/nn.cu:752), [snapshot src/nn.cu:758](/tmp/ufsm-audit-8_vefo68/src/nn.cu:758), [snapshot src/unet.c:315](/tmp/ufsm-audit-8_vefo68/src/unet.c:315).

Block sums are accumulated in double but stored in float before the final double sum. A finite input around `1e20` has a finite double square around `1e40`, but the float partial becomes infinity. Gradient norms and clipping can therefore be wrong for large finite values despite the public double result type.

Suggested remedy: retain double partials or calculate norms using a scaled algorithm that avoids overflow.

### F056 — P1, later live source: The advertised TF32 mode uses BF16 products

Locations in the later live version read during review: `src/nn.cu`, functions `prep_w_k`, `mma16816`, `conv_fwd_tc_k`, and `nn_conv3d_fwd`; compare [snapshot src/nn.h:30](/tmp/ufsm-audit-8_vefo68/src/nn.h:30) and [snapshot tests/test_nn.c:67](/tmp/ufsm-audit-8_vefo68/tests/test_nn.c:67).

The live rewrite converts weights and staged inputs with `__float2bfloat16` and uses an MMA instruction with BF16 operands, while the public/default mode remains named TF32. An analytic, **unexecuted** one-voxel example has center weight 1, all other taps 0, no bias, and input `257/256 = 1.00390625`: TF32 represents it exactly, but BF16 rounds it to 1, an error of 0.00390625. This precision distinction exceeds the snapshot's `1e-4` forward tolerance and changes default numerical behavior, including backward-data convolutions routed through forward. No model-accuracy or speed claim is established by this example.

Suggested remedy: restore the promised precision or expose/document BF16 as an explicit mode with appropriate numerical tests and checkpoint/run provenance. This supplement covers the inspected precision change, not the correctness of the entire evolving CUDA rewrite.

### F057 — P2, API risk: Persistent CUDA scratch is global rather than per-device/per-caller

Locations: [snapshot src/nn.cu:190](/tmp/ufsm-audit-8_vefo68/src/nn.cu:190), [snapshot src/nn.cu:204](/tmp/ufsm-audit-8_vefo68/src/nn.cu:204), [snapshot src/nn.cu:459](/tmp/ufsm-audit-8_vefo68/src/nn.cu:459).

Tensor-core weight scratch, GroupNorm sums, and kernel attribute flags are static process state without device identity or host-thread synchronization. Selecting another device can reuse pointers belonging to the previous device, and concurrent callers can race on reallocation/use. The ordinary CLI uses one GPU per process, which limits current exposure; equivalent global scratch remains in the later CUDA version inspected.

Suggested remedy: provide per-device/per-context ownership or explicitly restrict and enforce the API's supported device/thread lifecycle.

### F058 — P2: CUDA completion/error behavior contradicts the synchronous API description

Locations: [snapshot src/nn.h:1](/tmp/ufsm-audit-8_vefo68/src/nn.h:1), [snapshot src/nn.cu:15](/tmp/ufsm-audit-8_vefo68/src/nn.cu:15), [snapshot src/nn.cu:19](/tmp/ufsm-audit-8_vefo68/src/nn.cu:19).

The header describes every operation as synchronous. Kernel wrappers enqueue work and query launch errors; `nn_check` does not synchronize execution. A successful check does not establish completed, successful device work, and asynchronous failures can surface at a later transfer/synchronization. The shared `g_err` also has no host-thread protection.

Suggested remedy: document the actual stream/completion semantics and make required completion/error boundaries explicit, or implement the promised synchronous behavior.

### F059 — P2, API risk: Forward does not validate supplied input channels

Locations: [snapshot src/unet.c:169](/tmp/ufsm-audit-8_vefo68/src/unet.c:169), [snapshot src/unet.c:227](/tmp/ufsm-audit-8_vefo68/src/unet.c:227).

Forward checks spatial divisibility but substitutes `cfg.cin` for supplied `xs.c` when allocating/using encoder shapes. A caller supplying fewer channels can cause reads beyond its input allocation; inconsistent batch/spatial dimensions also lack complete validation.

Suggested remedy: require positive representable shapes and an input channel count matching the model before any allocation/launch.

### F060 — P2, API risk: Convolution scratch sizing omits the output head

Locations: [snapshot src/unet.c:197](/tmp/ufsm-audit-8_vefo68/src/unet.c:197), [snapshot src/unet.c:291](/tmp/ufsm-audit-8_vefo68/src/unet.c:291), [snapshot src/nn.cu:290](/tmp/ufsm-audit-8_vefo68/src/nn.cu:290).

The maximum workspace is calculated from encoder/decoder convolutions, omitting the head. The head's backward-data operation writes flipped head weights there. A sufficiently large public `cfg.cout` with a narrow model can exceed that allocation; default two-channel training does not trigger it.

Suggested remedy: include every actual backward-data operation, including the head, in workspace sizing.

## CUDA/model performance and instrumentation

These are source-derived, unmeasured observations of the snapshot. The later rewrite changed several kernels and fusion paths, so they require rechecking before any optimization work.

### F061 — P2: Loss statistics use only one block per sample/channel volume

Locations: [snapshot src/nn.cu:670](/tmp/ufsm-audit-8_vefo68/src/nn.cu:670), [snapshot src/nn.cu:709](/tmp/ufsm-audit-8_vefo68/src/nn.cu:709).

Default B=1, C=2 exposes only two blocks to scan the volume; each of 256 threads processes thousands of voxels at P=96. Spatial slabs followed by a reduction would expose more parallelism. The current amount of wall-clock cost was not measured.

### F062 — P2: GroupNorm backward rebuilds the same group sums for every voxel

Location: [snapshot src/nn.cu:520](/tmp/ufsm-audit-8_vefo68/src/nn.cu:520).

Each output element loops over C/G channels to construct the same two group sums. Work scales as `N*C*S*(C/G)` rather than a group reduction followed by a linear application pass. Precompute those sums per sample/group; repeated loads/arithmetic were also visible in the later code read, but no timing was collected.

### F063 — P2, snapshot; later reworked: Profiling percentages omit substantial backward work

Locations: [snapshot src/unet.c:12](/tmp/ufsm-audit-8_vefo68/src/unet.c:12), [snapshot src/unet.c:268](/tmp/ufsm-audit-8_vefo68/src/unet.c:268), [snapshot src/unet.c:274](/tmp/ufsm-audit-8_vefo68/src/unet.c:274).

Block backward's GroupNorm, recomputation, SiLU, and convolution backward-data execute outside `PROF`, while weight-gradient calls are timed. Synchronizing before starting each timer excludes preceding untimed work, and percentages divide by the incomplete measured total. The report can misidentify what dominates training.

Suggested remedy: time all relevant operations consistently and report unaccounted time, or clearly identify the report as partial coverage.

Later-source status: the external revision added profiling around the previously omitted backward operations and replaced host timing with CUDA event profiling. The specific snapshot omission was addressed structurally; the new profiler's complete behavior was not audited or run.

### F064 — P2: Decoder skip copies make separate runtime calls for each batch item

Locations: [snapshot src/unet.c:245](/tmp/ufsm-audit-8_vefo68/src/unet.c:245), [snapshot src/nn.cu:30](/tmp/ufsm-audit-8_vefo68/src/nn.cu:30).

After upsampling, a host loop issues one `cudaMemcpyDeviceToDevice` per sample/decoder level. A batched device copy into concat channels could reduce host/runtime calls and make overlap easier. No slowdown or synchronization cost was measured.

### F065 — P2: Inference retains activations and training workspace beyond their needed lifetime

Locations: [snapshot src/unet.c:153](/tmp/ufsm-audit-8_vefo68/src/unet.c:153), [snapshot src/unet.c:186](/tmp/ufsm-audit-8_vefo68/src/unet.c:186), [snapshot src/unet.c:230](/tmp/ufsm-audit-8_vefo68/src/unet.c:230).

Inference retains a1/a2/statistics for every block after that block is finished. A training-built model switched to inference also keeps gradient/recompute scratch; rebuilding occurs for inference-to-training, not the reverse. This is an optimization opportunity that can limit batch/patch size, not a demonstrated output error. Reuse forward temporary workspace and expose deliberate release of training resources for sustained inference.

## Numerical tests and benchmark quality

### F066 — P2: Numerical error checks can pass NaNs

Locations: [snapshot tests/test_nn.c:15](/tmp/ufsm-audit-8_vefo68/tests/test_nn.c:15), [snapshot tests/test_nn.c:37](/tmp/ufsm-audit-8_vefo68/tests/test_nn.c:37), [snapshot tests/test_nn.c:262](/tmp/ufsm-audit-8_vefo68/tests/test_nn.c:262), [snapshot tests/test_unet.c:85](/tmp/ufsm-audit-8_vefo68/tests/test_unet.c:85).

`NaN > worst` is false, so comparison loops can leave maximum error at zero. `fmax(existing,NaN)` can similarly keep an existing finite maximum, and network finite differences can retain `bad==0`. Forward/gradient/optimizer failures with nonfinite values can therefore appear to pass.

Suggested remedy: assert finiteness before error/tolerance calculations and for outputs, gradients, losses, and optimizer state.

### F067 — P2: EMA and important backend branches lack meaningful assertions

Locations: [snapshot tests/test_unet.c:92](/tmp/ufsm-audit-8_vefo68/tests/test_unet.c:92), [snapshot tests/test_nn.c:268](/tmp/ufsm-audit-8_vefo68/tests/test_nn.c:268).

EMA/live loss is printed without an assertion; the tests never establish different live/EMA weights or check a known EMA update. Primitive convolution checks use only the default precision mode. Listed stride-2 inputs contain an odd dimension, so they bypass the all-even backward-data branch used for ordinary U-Net shapes. No assertions cover invalid groups/widths, one-level behavior, malformed/truncated checkpoints, or resume compatibility.

Suggested remedy: add explicit checks for each precision/backend branch, known EMA state transitions, and the failure contracts identified above. These are future validation proposals; no tests were written or run during this audit.

Later-source status: primitive/network finite-difference tests now explicitly select fp32, and the network benchmark prints a tensor-core-versus-fp32 comparison. This improves path separation, but the printed comparison still has no acceptance assertion and does not establish default reduced-precision correctness.

### F068 — P2: The convolution benchmark uses uninitialized accumulation buffers and returns success on errors

Locations: [snapshot tests/bench_conv.c:12](/tmp/ufsm-audit-8_vefo68/tests/bench_conv.c:12), [snapshot tests/bench_conv.c:32](/tmp/ufsm-audit-8_vefo68/tests/bench_conv.c:32), [snapshot tests/bench_conv.c:39](/tmp/ufsm-audit-8_vefo68/tests/bench_conv.c:39), [snapshot tests/bench_conv.c:49](/tmp/ufsm-audit-8_vefo68/tests/bench_conv.c:49).

gw/gb are not zeroed before an API that accumulates gradients. Initialization failure is ignored, numerical differences are only printed, and the executable returns zero even when `nn_check` reports an error. A failed setup/kernel can still be presented as a successful benchmark.

Suggested remedy: initialize accumulation buffers, enforce correctness/initialization checks, and return failure for CUDA errors.

### F069 — P3: The ordinary network test includes a substantial default benchmark

Location: [snapshot tests/test_unet.c:100](/tmp/ufsm-audit-8_vefo68/tests/test_unet.c:100).

After tiny correctness checks, `test_unet` allocates the full model at B=2, P=128 and runs timing loops by default. Thus `make test` also requests a sizeable GPU workload; on a shared GPU it can fail for memory/occupancy reasons unrelated to tiny correctness tests. Separate correctness tests from opt-in performance benchmarks and make resource assumptions explicit.

Later-source status: external changes reduced the default benchmark to B=1, P=96, but retained the benchmark and added full-model precision-comparison forwards to the ordinary test executable.

## Storage, Zarr, TIFF, ZIP, and JSON

### F070 — P1: HTTP range reads can return the object prefix as successful requested data

Locations: [src/store.c:131](/home/forrest/ufsm/src/store.c:131), [src/store.c:182](/home/forrest/ufsm/src/store.c:182), [src/store.c:279](/home/forrest/ufsm/src/store.c:279).

`store_read` accepts HTTP 200 as well as 206. If the server ignores Range, the fixed sink copies the first requested-length bytes of the full object without skipping the requested offset, then reports success. A request for offset 4, length 3 in `abcdefghij` would return `abc` instead of `efg`. The callback also reports consuming the rest of the full response, so the whole object continues downloading. Shard indexes and payload ranges can be corrupted.

Suggested remedy: require a matching 206/Content-Range response or implement a bounded, explicitly validated whole-object fallback that selects the requested offset.

### F071 — P1: HTTP authorization failures become missing data and persistent negative cache entries

Locations: [src/store.c:221](/home/forrest/ufsm/src/store.c:221), [src/store.c:283](/home/forrest/ufsm/src/store.c:283), [src/zarr2.c:92](/home/forrest/ufsm/src/zarr2.c:92), [src/zarr3.c:234](/home/forrest/ufsm/src/zarr3.c:234).

HTTP 403 maps to the same missing-object result as 404. Zarr readers return fill, and HEAD-based paths record persistent missing markers. Denied access or expired authorization can silently remove real CT/labels from subsequent reads, even after access is restored.

Suggested remedy: distinguish authentication/permission failures from confirmed absence and cache only genuine absent-object responses.

### F072 — P1: Unsharded V3 reads suppress all fetch failures as missing chunks

Locations: [src/zarr3.c:333](/home/forrest/ufsm/src/zarr3.c:333), [src/zarr3.c:433](/home/forrest/ufsm/src/zarr3.c:433), [src/store.c:288](/home/forrest/ufsm/src/store.c:288).

Any null `store_read_all` result causes the worker to continue without setting failure. Permission, transport, local I/O, and allocation failures all yield a successful read containing fill. This silently substitutes data rather than surfacing an operational problem.

Suggested remedy: expose explicit store result/status information and propagate every failure other than confirmed absence.

### F073 — P1: Disk caches omit the store root from object identity

Locations: [src/zarr2.c:80](/home/forrest/ufsm/src/zarr2.c:80), [src/zarr3.c:180](/home/forrest/ufsm/src/zarr3.c:180).

Cache paths use the cache directory and relative key but omit `store_root`. Distinct roots with the same array/key can therefore reuse each other's chunks, indexes, or missing markers. The source format supports multiple roots sharing one cache, so this affects legitimate configurations.

Suggested remedy: namespace caches by normalized store identity, and include version/ETag where mutable sources are supported.

### F074 — P1: Edge-chunk copying overwrites outside-array fill with stored padding

Locations: [src/zarr2.c:148](/home/forrest/ufsm/src/zarr2.c:148), [src/zarr2.c:170](/home/forrest/ufsm/src/zarr2.c:170), [src/zarr3.c:275](/home/forrest/ufsm/src/zarr3.c:275), [src/zarr3.c:376](/home/forrest/ufsm/src/zarr3.c:376).

Scheduling clips to the array shape, but each copy clips only to the requested box and full chunk bounds. For shape `[1,1,1]`, chunk `[2,2,2]`, and a two-cubed request at zero, all eight stored bytes are copied even though seven lie outside the array. Padded inference windows and pyramid pooling receive those padding bytes instead of the specified outside-array fill.

Suggested remedy: include the array shape in each actual copy's upper bounds and preserve fill outside it.

### F075 — P1: V2 accepts Fortran-order arrays and interprets them as C order

Locations: [src/zarr2.c:42](/home/forrest/ufsm/src/zarr2.c:42), [src/zarr2.c:158](/home/forrest/ufsm/src/zarr2.c:158).

The reader neither inspects nor stores `order`; all source offsets use C order. Valid `order:"F"` data is accepted with permuted voxel coordinates. ZIP ingest has the related metadata omission in F028.

Suggested remedy: reject unsupported order or use appropriate source strides.

### F076 — P1: Format geometry and work-count products are insufficiently validated

Locations: [src/zarr2.c:44](/home/forrest/ufsm/src/zarr2.c:44), [src/zarr2.c:168](/home/forrest/ufsm/src/zarr2.c:168), [src/zarr3.c:99](/home/forrest/ufsm/src/zarr3.c:99), [src/zarr3.c:119](/home/forrest/ufsm/src/zarr3.c:119), [src/z3w.c:105](/home/forrest/ufsm/src/z3w.c:105), [src/ingest.c:503](/home/forrest/ufsm/src/ingest.c:503).

V2 permits zero/negative chunk dimensions before division. V3 validates positive inner chunks but accepts divisible zero/negative outer shard dimensions; an outer shard `[0,1,1]` with inner `[1,1,1]` later divides by zero. Three-member objects can pass count checks intended for arrays, then produce zero defaults through `json_at`. Grid/chunk/work counts and writer cubic chunk counts use unchecked products and narrowing to `int`. ZIP ingest also defaults missing chunk sizes to zero before modulo/division. These can cause invalid allocations, indexing, or arithmetic.

Suggested remedy: validate actual JSON array types, finite integral representable dimensions, positive chunks/shards, nonnegative shapes, and every size/work-count product. CLI validation in F003 is additionally needed.

### F077 — P1: V3 shard indexes lack checksum, codec, and payload-range validation

Locations: [src/zarr3.c:106](/home/forrest/ufsm/src/zarr3.c:106), [src/zarr3.c:228](/home/forrest/ufsm/src/zarr3.c:228), [src/zarr3.c:337](/home/forrest/ufsm/src/zarr3.c:337), [src/zarr3.c:414](/home/forrest/ufsm/src/zarr3.c:414).

The reader assumes a `16*nc+4` index without validating index codecs or its CRC32C. It copies offsets/lengths in host endianness, does not reject inconsistent missing sentinels, and does not bound ranges to the payload before the index/object end or check representability. A damaged index can redirect a read to different valid chunk data; huge lengths drive unbounded allocations. Invalid indexes may also be cached.

Suggested remedy: validate the supported codec chain and checksum before caching; decode declared endianness; check paired sentinels and bounded, representable payload ranges.

### F078 — P1: Blosc decompression ignores the actual compressed-buffer length

Location: [src/zarr2.c:119](/home/forrest/ufsm/src/zarr2.c:119); local dependency contract: [blosc.h:430](/usr/include/blosc.h:430).

`z2_decode` passes the input pointer directly to `blosc_decompress_ctx` without validating its supplied byte length. That API does not receive the actual allocation length. The installed dependency header provides `blosc_cbuffer_validate(cbuffer,cbytes,...)` specifically to establish that attempting decompression is safe. Truncated HTTP/cache buffers reach the decoder without that validation.

Suggested remedy: validate the compressed buffer using its real length and ensure the advertised decoded size matches the expected chunk before decompression.

### F079 — P1: Fill-only rewrites preserve older non-fill shards

Locations: [src/z3w.c:115](/home/forrest/ufsm/src/z3w.c:115), [src/z3w.c:118](/home/forrest/ufsm/src/z3w.c:118), [src/z3w.c:47](/home/forrest/ufsm/src/z3w.c:47), [src/predict.c:121](/home/forrest/ufsm/src/predict.c:121).

When every chunk is fill, `z3w_write_shard` does nothing to the existing path and returns success. Replacing non-fill data with fill therefore leaves the old data readable. Creating an array in an existing output directory also does not clear the previous shard tree; prediction skips all-air shards entirely. Re-running export/prediction can retain stale voxels.

Suggested remedy: define rebuild/overwrite semantics, delete or replace existing fill-only shards, and prevent stale shard trees from surviving array recreation.

### F080 — P1: Zarr writer reports success before confirming buffered writes flushed

Locations: [src/z3w.c:54](/home/forrest/ufsm/src/z3w.c:54), [src/z3w.c:133](/home/forrest/ufsm/src/z3w.c:133), [src/z3w.c:139](/home/forrest/ufsm/src/z3w.c:139), [src/z3w.c:157](/home/forrest/ufsm/src/z3w.c:157).

Metadata `fprintf` and all close results are ignored. A buffered shard write can fail at close but still be renamed to its final location and reported successful. Metadata creation can likewise return a valid handle/zero after a delayed failure. This compounds caller error suppression in F010.

Suggested remedy: verify all writes and successful flush/close before publication; remove failed temporary artifacts and preserve previous complete outputs.

### F081 — P1: Failed V2 cache writes can turn real data into cached absence

Locations: [src/zarr2.c:83](/home/forrest/ufsm/src/zarr2.c:83), [src/zarr2.c:101](/home/forrest/ufsm/src/zarr2.c:101).

Cache writes/close/rename results are ignored. An empty positive cache file from a failed write is subsequently interpreted as a missing marker and returns fill. Truncated cached reads are used rather than invalidated/refetched, so cache storage failures can silently alter the dataset or feed malformed compressed data to decoders.

Suggested remedy: distinguish deliberate missing markers from cached data; publish only complete, successfully closed data and refetch invalid entries.

### F082 — P2: CRC32C's lazy table initialization races on parallel first writes

Locations: [src/z3w.c:17](/home/forrest/ufsm/src/z3w.c:17), [src/z3w.c:135](/home/forrest/ufsm/src/z3w.c:135), [src/ingest.c:119](/home/forrest/ufsm/src/ingest.c:119).

Plain static `init` and `table` are written/read without synchronization. Pyramid workers can call the first CRC simultaneously; identical intended values still constitute a C data race.

Suggested remedy: use a constant table or thread-safe one-time initialization.

### F083 — P1: JSON path lookup shares `strtok` state across metadata threads

Locations: [src/json.c:149](/home/forrest/ufsm/src/json.c:149), [src/sources.c:181](/home/forrest/ufsm/src/sources.c:181), [src/sample.c:334](/home/forrest/ufsm/src/sample.c:334).

Concurrent lazy metadata opens invoke `json_path`, whose `strtok` continuation state is shared. One traversal can consume another's tokens or point into another thread's stack buffer, returning incorrect/missing metadata or causing undefined behavior. Fixing handle publication alone in F015 does not fix this independent tokenizer race.

Suggested remedy: tokenize locally or use reentrant tokenization.

### F084 — P1: HTTP header parsers scan beyond their supplied length

Locations: [src/store.c:38](/home/forrest/ufsm/src/store.c:38), [src/store.c:231](/home/forrest/ufsm/src/store.c:231).

Callbacks receive a pointer and length, but rate-limit/Link parsing uses `strstr`, `strchr`, `atof`, and open-ended scans as if every buffer were NUL-terminated. Missing delimiters can cause reads beyond the callback's valid bytes.

Suggested remedy: parse only within the supplied range or copy into a checked NUL-terminated buffer first.

### F085 — P2: Repeated HTTP reader threads leak CURL state and lose connection reuse

Locations: [src/store.c:51](/home/forrest/ufsm/src/store.c:51), [src/store.c:149](/home/forrest/ufsm/src/store.c:149), [src/zarr2.c:185](/home/forrest/ufsm/src/zarr2.c:185), [src/zarr3.c:429](/home/forrest/ufsm/src/zarr3.c:429).

Thread-local CURL handles/header lists are allocated without cleanup/destructors. Zarr calls repeatedly create and destroy workers, losing those pointers on exit. Allocations/connection resources can accumulate, and successive workers cannot reuse prior connections.

Suggested remedy: use bounded persistent workers or TLS ownership with destructors and explicit shutdown.

### F086 — P2: Store global initialization and cleanup have unsafe lifecycle assumptions

Locations: [src/store.c:55](/home/forrest/ufsm/src/store.c:55), [src/store.h:31](/home/forrest/ufsm/src/store.h:31).

The plain static initialization flag races between concurrent opens. Cleanup neither coordinates live handles nor resets the flag, so a later open skips initialization after cleanup.

Suggested remedy: define a synchronized process/store lifecycle and coordinate teardown with all handle owners.

### F087 — P1: ZIP64 extra lengths can permit reads past the central-directory buffer

Location: [src/zipr.c:72](/home/forrest/ufsm/src/zipr.c:72).

An extra field's declared `len` is used as its bound without checking that it fits the enclosing extra area. A four-byte ZIP64 header advertising eight bytes, combined with an extended-size sentinel, can make `u64` read absent bytes beyond that area and potentially beyond the directory allocation.

Suggested remedy: validate each extra field against its actual enclosing range before reading fields; use overflow-safe subtraction bounds.

### F088 — P2: ZIP parsing returns partial archives after malformed/truncated directories

Locations: [src/zipr.c:45](/home/forrest/ufsm/src/zipr.c:45), [src/zipr.c:64](/home/forrest/ufsm/src/zipr.c:64), [src/zipr.c:91](/home/forrest/ufsm/src/zipr.c:91).

Bad entry signatures and truncated entries only break the parser loop; it still returns a handle with fewer entries than advertised. Running out of directory bytes is similarly accepted. EOCD selection does not validate comment length/end placement, so a signature inside a comment can be mistaken for the actual record. Ingest can omit remaining chunks without a terminal archive error.

Suggested remedy: reject incomplete/inconsistent directory traversal and validate EOCD position, counts, and all ranges.

### F089 — P1: ZIP reads omit CRC/exact-size validation and can return uninitialized tail bytes

Locations: [src/zipr.c:15](/home/forrest/ufsm/src/zipr.c:15), [src/zipr.c:67](/home/forrest/ufsm/src/zipr.c:67), [src/zipr.c:119](/home/forrest/ufsm/src/zipr.c:119).

Entry CRC is never retained or checked. Stored entries accept compressed/uncompressed size mismatch. Deflate only needs `Z_STREAM_END`; a stream expanding to fewer bytes than advertised is returned with the advertised length and an uninitialized tail. Corrupt/inconsistent archive data can therefore become apparent valid labels/images.

Suggested remedy: verify CRC and exact cumulative decoded size/input consumption, enforce stored-size equality, and reject unsupported encryption explicitly.

### F090 — P2: Deflated ZIP64 entries are decoded through 32-bit length fields in one call

Location: [src/zipr.c:122](/home/forrest/ufsm/src/zipr.c:122).

64-bit entry lengths are cast to zlib `uInt` availability counts, followed by a single inflate call. Deflated entries beyond those availability limits cannot be decoded correctly despite ZIP64 metadata support.

Suggested remedy: stream through bounded pieces while tracking and validating 64-bit cumulative counts.

### F091 — P1: TIFF tag/bounds/geometry validation can reach invalid reads and arithmetic

Locations: [src/tiff.c:40](/home/forrest/ufsm/src/tiff.c:40), [src/tiff.c:57](/home/forrest/ufsm/src/tiff.c:57), [src/tiff.c:68](/home/forrest/ufsm/src/tiff.c:68), [src/tiff.c:211](/home/forrest/ufsm/src/tiff.c:211).

Additive/multiplicative checks can overflow, especially for BigTIFF's 64-bit offsets/counts. An unknown tag type can have a one-byte checked size but fall through to a four-byte `rd32`, allowing a read beyond the file buffer. Image/sample/tile dimensions and block products are insufficiently validated; zero/overflowed tile geometry can cause division by zero or invalid access.

Suggested remedy: reject unsupported types, validate actual access widths and positive representable geometry, and check all counts/ranges/products before decoding.

### F092 — P1: Incomplete TIFF image data can be reported as successfully decoded

Locations: [src/tiff.c:149](/home/forrest/ufsm/src/tiff.c:149), [src/tiff.c:173](/home/forrest/ufsm/src/tiff.c:173), [src/tiff.c:219](/home/forrest/ufsm/src/tiff.c:219).

Short raw strips are padded with zeros; LZW can return success at early EOI without checking output length; inflate accepts stream end without exact output-size checks. Too few strips can leave rows untouched while returning success. Truncated/corrupt inputs can become plausible training data containing zero padding, prior contents, or uninitialized caller-buffer bytes.

Suggested remedy: require exact decoded block sizes and full expected image coverage; reject truncation rather than silently filling it.

### F093 — P2: TIFF predictors are accepted beyond implemented combinations

Locations: [src/tiff.c:87](/home/forrest/ufsm/src/tiff.c:87), [src/tiff.c:184](/home/forrest/ufsm/src/tiff.c:184), [src/tiff.c:238](/home/forrest/ufsm/src/tiff.c:238).

Predictor 2 reconstruction has only 8/16-bit branches, but 32-bit predictor-2 samples are accepted and returned still differenced. Unknown predictor values are also accepted and ignored.

Suggested remedy: implement supported combinations or reject them during metadata parsing. Predictor-3 concerns needing separate format-reference verification are listed below.

### F094 — P2: Failed TIFF IFD construction leaks allocations from the uncounted page

Locations: [src/tiff.c:72](/home/forrest/ufsm/src/tiff.c:72), [src/tiff.c:109](/home/forrest/ufsm/src/tiff.c:109), [src/tiff.c:133](/home/forrest/ufsm/src/tiff.c:133).

An IFD can allocate description/state and fail before incrementing `npages`. Cleanup only visits counted pages, omitting the failing page's owned allocations.

Suggested remedy: explicitly clean the in-progress page on failure or include it in construction-time ownership accounting.

### F095 — P2, API risk: JSON numbers can read past the declared input boundary

Locations: [src/json.c:111](/home/forrest/ufsm/src/json.c:111), [src/json.h:17](/home/forrest/ufsm/src/json.h:17).

`json_parse(text,len)` supplies a length, but numeric parsing delegates to unbounded `strtod`. A number ending at a non-NUL-terminated allocation boundary can make conversion read beyond it before cursor checks. Common store/ZIP callers add a terminator; the public bounded-input contract still does not guarantee this safety.

Suggested remedy: lex numbers within the explicit end pointer and convert only a bounded token.

### F096 — P2: JSON grammar is permissive and non-ASCII escaped strings lose information

Locations: [src/json.c:32](/home/forrest/ufsm/src/json.c:32), [src/json.c:111](/home/forrest/ufsm/src/json.c:111).

The numeric converter admits C spellings/forms outside JSON grammar, including nonfinite values, hex, leading plus, and leading zeros. Unknown escapes/control characters and malformed Unicode digits are not strictly rejected. Non-ASCII Unicode escapes become `?`, changing valid names/paths; nonfinite values can flow into unsafe metadata integer conversions.

Suggested remedy: enforce JSON lexical rules, validate finite metadata numbers, and decode Unicode/surrogates correctly or reject unsupported strings explicitly.

### F097 — P2: JSON parsing/freeing has no nesting bound

Locations: [src/json.c:61](/home/forrest/ufsm/src/json.c:61), [src/json.c:128](/home/forrest/ufsm/src/json.c:128).

Recursive arrays/objects and recursive freeing have no depth limit. Deep input can exhaust the process stack even at a manageable byte size.

Suggested remedy: enforce a documented nesting bound or use iterative traversal.

### F098 — P2: Worker failures leave callers with empty/stale error strings

Locations: [src/zarr2.c:17](/home/forrest/ufsm/src/zarr2.c:17), [src/zarr2.c:185](/home/forrest/ufsm/src/zarr2.c:185), [src/zarr3.c:341](/home/forrest/ufsm/src/zarr3.c:341), [src/z3w.c:95](/home/forrest/ufsm/src/z3w.c:95).

Workers set thread-local error text and a shared failed flag. After joining, `z2_error`, `z3_error`, or `z3w_error` on the calling thread reads a different TLS buffer, which may be empty or contain an earlier error. Operational failures are consequently difficult to diagnose correctly.

Suggested remedy: store the first failure in job-owned shared state and publish it to the caller after workers finish.

### F099 — P2: Resource failures are widely unchecked

Locations: [src/zarr2.c:177](/home/forrest/ufsm/src/zarr2.c:177), [src/zarr3.c:319](/home/forrest/ufsm/src/zarr3.c:319), [src/zarr3.c:429](/home/forrest/ufsm/src/zarr3.c:429), [src/z3w.c:79](/home/forrest/ufsm/src/z3w.c:79), [src/zipr.c:56](/home/forrest/ufsm/src/zipr.c:56), [src/sample.c:337](/home/forrest/ufsm/src/sample.c:337).

Failed allocations often proceed into pointer writes/copies. Thread-creation failures are ignored, followed by joins using thread IDs that were never successfully initialized. Large geometry or ordinary resource pressure can turn recoverable failures into crashes or invalid behavior. Related defaults are quantified in F040.

Suggested remedy: check allocation/creation results, preserve existing pointers across failed realloc, join only successfully started workers, and return explicit resource errors.

### F100 — P3: Missing-shard cache checks leak their allocated marker buffer

Location: [src/zarr3.c:232](/home/forrest/ufsm/src/zarr3.c:232).

`cache_get(...,"missing",...)` returns allocated memory even for a zero-byte marker. The pointer is used only as a condition and is never freed, leaking one allocation on every such check.

Suggested remedy: perform a non-allocating existence check or free the retrieved marker.

### F101 — P2: The writer encodes bytes outside the declared array despite its contract

Locations: [src/z3w.h:13](/home/forrest/ufsm/src/z3w.h:13), [src/z3w.c:76](/home/forrest/ufsm/src/z3w.c:76), [src/z3w.c:104](/home/forrest/ufsm/src/z3w.c:104).

The encoder worker receives neither shard coordinates nor array shape. It encodes every shard byte, including padding, and accepts wholly outside-array shard coordinates. This wastes work/storage and makes edge encoding depend on caller padding; a lossy block transform can also couple padding to retained edge samples. The actual edge distortion was not measured.

Suggested remedy: validate coordinates and normalize outside-array positions according to a documented padding rule before omission/encoding.

## Storage performance observations

### F102 — P2: Cold shard-index reads finish serially before chunk workers start

Locations: [src/zarr3.c:234](/home/forrest/ufsm/src/zarr3.c:234), [src/zarr3.c:387](/home/forrest/ufsm/src/zarr3.c:387), [src/zarr3.c:429](/home/forrest/ufsm/src/zarr3.c:429).

The caller performs each overlapping shard's HEAD/index-range read serially before starting chunk workers. Increasing decode threads does not parallelize this cold latency, and the first chunk waits for all indexes. Bounded shard tasks or parallel index loading are profiling candidates; no network timing was collected.

### F103 — P2: Each read recreates workers/decoder state and allocates unused large scratch

Locations: [src/zarr3.c:318](/home/forrest/ufsm/src/zarr3.c:318), [src/zarr3.c:424](/home/forrest/ufsm/src/zarr3.c:424), [src/zarr2.c:180](/home/forrest/ufsm/src/zarr2.c:180), [third_party/volcomp.h:86](/home/forrest/ufsm/third_party/volcomp.h:86).

Calls create fresh threads and decoder contexts. Each V3 worker allocates at least `VOLCOMP_ENCODE_BOUND = 14,681,264` bytes of scratch even when no Zstd stage exists and that buffer is unused. At 256 workers this alone requests roughly 3.5 GiB of address space, **not a measured resident-memory amount**. Use bounded reusable pools/contexts and allocate workspace only for stages that need it. HTTP lifetime issues are separately covered by F085.

### F104 — P3: Uncached V2 chunks normally require HEAD followed by GET

Location: [src/zarr2.c:92](/home/forrest/ufsm/src/zarr2.c:92).

Every present uncached chunk uses a HEAD before whole-object GET, consuming two requests on paced stores. The existing optional `assume_present` path avoids the preliminary request. An explicit-result GET path could make the extra HEAD unnecessary. Latency/throughput impact was not measured.

## Format-test gaps and additional issues

### F105 — P2: Fixture failures can be reported as successful skips

Locations: [tests/test_formats.c:29](/home/forrest/ufsm/tests/test_formats.c:29), [tests/test_formats.sh:9](/home/forrest/ufsm/tests/test_formats.sh:9), [tests/test_zarr.sh:10](/home/forrest/ufsm/tests/test_zarr.sh:10), [tests/test_zarr.sh:36](/home/forrest/ufsm/tests/test_zarr.sh:36).

The TIFF test skips whenever opening the TIFF fails, even if the input and reference exist; a decoder regression can therefore become a passing skip. `test_zarr.sh` can perform no reads at all when its hardcoded local dataset is absent and networking is disabled, yet still prints `zarr ok`. Fixtures depend heavily on private/external data, so a green run need not establish the advertised format coverage.

Suggested remedy: distinguish absent fixtures from failed decoding and report which checks actually ran; provide deterministic local format fixtures.

### F106 — P2: `UFSM_NET=0` still enables the formats network-test branch

Locations: [tests/test_formats.sh:26](/home/forrest/ufsm/tests/test_formats.sh:26), [tests/test_formats.c:118](/home/forrest/ufsm/tests/test_formats.c:118).

The shell exports the variable with value `0`, but C checks only whether it exists. If cached reference bytes and the token file are present, the supposedly disabled network path still runs.

Suggested remedy: parse the flag value consistently and make offline tests independent of prior cached fixtures/token availability.

Additional source-review concerns requiring format-reference verification:

- **TIFF predictor 3:** [src/tiff.c:199](/home/forrest/ufsm/src/tiff.c:199) always uses a one-byte inverse-difference stride rather than samples-per-pixel; unshuffle writes little-endian host bytes and is followed by an additional swap for big-endian TIFF. Multi-sample and big-endian correctness is suspected, not established here. Independently generated fixtures/reference review are needed.
- **V3 codec pipeline interpretation:** [src/zarr3.c:39](/home/forrest/ufsm/src/zarr3.c:39) reduces codecs to flags; ordering/repetition, missing serialization, and codecs after an initial sharding stage are not fully validated. Define the supported pipeline contract and reject unsupported combinations explicitly. Not every valid/invalid codec chain was classified in this audit.

The reviewed tests do not establish the following behavior:

- Sampler shutdown/retry behavior, per-channel masking, region holdouts, duplicate-CT exclusions, cache ownership, or anisotropic ingest.
- Crop-origin evaluation, ignored-voxel band metrics, prediction artifact identity, blend/seam behavior, or failure propagation from final output publication.
- Independent writer/reader conformance: the writer round trip uses the project's own reader and can pass matching bugs in both. It lacks fill-only overwrite, checksum corruption, outside-shape padding, flush-failure, and first-use concurrency coverage.
- A deterministic V2 raw/zlib/gzip/Zstd/Blosc matrix, Fortran order, cache-root isolation/corruption, permission-versus-absence handling, and invalid geometry.
- Deterministic TIFF malformed/compression/tile-edge/endian/multi-sample/predictor fixtures; ZIP reader tests; strict/Unicode/deep/bounded/concurrent JSON cases.
- Local HTTP fixtures for ignored ranges, bad Content-Range, interrupted bodies, forbidden responses, rate-limit/Link headers, retries, and cache concurrency.

These are proposed validation needs only; no tests were added or executed.

## Superseded observation and limits

The frozen TF32 forward used WMMA operand pointers at arbitrary float voxel/tap offsets ([snapshot src/nn.cu:163](/tmp/ufsm-audit-8_vefo68/src/nn.cu:163)), creating an alignment-contract concern. The later live rewrite loads packed BF16 registers directly and superseded that specific concern. It is **not counted as an active finding in the later version**. Its manual forward fragment mapping and tile bounds appeared internally consistent during selective review, but were not executed or independently validated.

The audit did not inspect actual external datasets, measure speed/RSS/accuracy, prove mesh-to-scan registration, or independently verify the complete vendored volcomp/surfcomp transforms and entropy codecs. Vendored code was considered at the integration/trust-boundary level. The evolving CUDA/model/test revisions require a fresh review against a stable revision before treating snapshot findings as either fixed or still present. All suggested remedies remain unimplemented.
