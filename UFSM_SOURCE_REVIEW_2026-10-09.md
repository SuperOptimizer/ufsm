# UFSM source review and implementation handoff

The highest priority repairs concern FP4 nonfinite propagation, prepared-weight cache validity, gradient accumulation, and faithful checkpoint continuation. Several additional supported configurations can cause invalid memory accesses or launches. These findings have specific triggers; they do not establish that every training or inference run is incorrect. The usual divisible patch sizes, default channel widths, and one backward per zeroing avoid some of the operator and API defects below.

This is the single consolidated handoff from a coordinating reviewer and three specialist agents, working in two rounds with six specialist passes. The review covered numerical forward and backward kernels, model integration, optimizer and checkpoint state, inference, data and storage handling, tests, performance tooling, and consistency. No implementation files were changed. No compilation, tests, benchmarks, training, inference, GPU operations, project scripts, or reproducer programs were run. Suggested checks below are future work.

## Source identity and confidence

- Review date: 2026-10-09.
- Repository: `/home/forrest/ufsm`.
- HEAD: `baa9b90ed40aff680eb42903c29ab4258485fa35`.
- The reviewed tree includes preexisting uncommitted changes and untracked source files. HEAD alone does not reproduce it.
- SHA256 of `git diff --binary`: `cdd30f076dbf561b101d60c7ef144195ed978d4830da9d828c0a67fa7f601cc1`.
- SHA256 of the sorted per-file SHA256 listing for files returned by `rg --files src tests tools configs`: `5df7a930c3bf7546ba474073019471eea125aa3550689cc4332b499eec4d4ca8`. Listing order used `LC_ALL=C sort`.

Preexisting tracked changes were in `Makefile`; `src/lp_mx_s2bwd_tc.cu`, `lp_mx_up2bwd.cu`, `nn.h`, `nn_common.cuh`, `nn_elem.cu`, `nn_loss.cu`, `predict.c`, `sample.c`, `sample.h`, `sources.c`, `sources.h`, `spatial_augment.c`, `spatial_augment.h`, `train.c`, `unet.c`, `unet.h`; and `tests/test_sampler_safety.c`, `test_spatial_augment.c`, `test_unet.c`. Untracked source included scanner augmentation and related tests/tools, plus surface-building/scoring tools. Existing build artifacts were excluded from execution and numerical evidence.

Locations refer to this working tree. Existing `FINDINGS.md`, `PRECISION_PLAN.md`, `FP16_MIXED_REPORT.md`, and README measurements were treated as context and checked against current implementation where relevant. Their historical performance and test results were not reproduced. Third-party codecs and all auxiliary Python geometry algorithms were not exhaustively audited line by line.

**Confidence:** Findings R01–R37 describe source-supported defects or explicitly identified measurement/contract limitations. Mathematical examples are deductions from the code, not executed experiments. Performance opportunities in their own section describe visible work and allocations; their speed benefit remains unmeasured.

**Severity:** P1 means silent numerical/state/data corruption, memory corruption, or a substantial execution failure under the stated trigger. P2 means a narrower mode/API defect, validation weakness, or materially misleading instrumentation. P3 means a limited API or measurement issue. No unconditional P0 defect was established.

## Actual mixed precision behavior

Precision selection combines global arithmetic, per-convolution/per-pass policy, activation storage, gradient storage, stem conversion, and memory-planner choices. A label such as “FP4 training” does not specify the entire graph.

| Path | Multiplication and accumulation | Storage and important qualifications |
| --- | --- | --- |
| Global exact mode with tensor cores disabled | Direct FP32 multiplication and FP32 accumulation | Ordinary FP32 tensors; intended numerical reference |
| Precision 1 with FP16 enabled | FP16 tensor-core operands, FP32 accumulation | FP16 or FP32 ordinary storage; FP32 weights are converted for MMA |
| Precision 1 with FP16 disabled | BF16 tensor-core operands, FP32 accumulation | BF16 or FP32 ordinary storage; verification build required for those instantiations |
| Precision 2 stride 1 without an ordinary fused-up fallback | E4M3 FP8 operands with local UE8M0 power-of-two scales, FP32 accumulation | Supported ordinary or MX storage combinations |
| Precision 3 stride 1 with more than eight input channels, without an ordinary fused-up fallback | E2M1 FP4 operands with scales per 32 reduction elements, FP32 accumulation | MX4/MX8 or supported ordinary storage; this is the existing MXFP4 path |
| Precision 3 with eight or fewer input channels | FP8 fallback | The stem can execute FP8 arithmetic despite a requested FP4 policy |
| MX4 activation storage | Selects FP4 stride-1 arithmetic except small-input fallbacks | Requesting an FP16 policy does not itself promote MX4 storage to an ordinary FP16 graph |
| Stride 2 forward and weight gradient under FP8/FP4/MX modes | FP8 arithmetic and FP32 accumulation | MX4 inputs can be recoded to the exact FP8 grid before MMA; FP4 policy does not imply FP4 down-convolution arithmetic |
| Stride 2 MX backward data | BF16 tensor-core arithmetic with FP32 accumulation for supported narrow C16/32/64 shapes; FP8 dilation or FP32 fallback for other shapes | MX8/MX4 gradient storage; arithmetic depends on the actual backward-data dispatcher |
| Ordinary fused-up stride-1 forward under requested FP8/FP4 | FP16/BF16 tensor-core arithmetic with FP32 accumulation | The ordinary `sp.up` fallback uses 16-bit arithmetic; MX fused-up paths retain their low-precision families |
| Precision 4 ordinary stride-1 forward and backward data | Scaled FP16 operands; FP16 accumulation in groups of at most 144 products, then folding into FP32 | Weight gradients retain FP32 accumulation; stride-2 forward uses ordinary tensor-core accumulation. Different numerical behavior from precision 1, even with FP16 storage |
| MX 1 by 1 output head | Decoded MX inputs with FP32 products and accumulation | FP32 logits, optionally rounded into FP16 logits by lean training |
| Parameters and optimizer state | Ordinary master parameters and parameter gradients in FP32; Adam moments in FP32 | True packed weights via `--wq` have different ownership and residual state from merely using FP4 convolution operands |
| GroupNorm and loss reductions | FP32 or FP64 local partial sums, depending on path, with several FP64 block/global reductions | Ordinary forward GN has double local sums; fused output/MX GN and loss include float local partials. A double buffer does not imply all arithmetic was FP64 |

Evidence: `src/nn_bwd.cu:49-77`; `src/nn_wgrad.cu:23-25`; `src/lp_common.cuh:534-546`; `src/nn_common.cuh:445-448,565-574,595-684,1785`; `src/lp_mx_conv1.cu:26-59`; `src/lp_mx_s2bwd_tc.cu:112-114,166-186`; `src/lp_mx_s2bwd.cu:70-109,124-159`; `src/unet.c:269-308`; `src/nn_loss.cu:13-40`.

The current trainer defaults to `--fp4 2` unless explicit precision/weight options change that default, requests `all=fp4:fp4:fp4,enc0.c1=fp16`, and enables MX4 activations: `src/train.c:455-468`. Gradient formats and retention/offload choices also depend on memory planning and environment settings. The default widths `16,32,64,80` fail the all-multiples-of-32 gate for MX4 activation gradients at `src/train.c:672-676`; they must not be described as automatically using the most aggressive FP4-gradient recipe. Inference restores more embedded runtime settings than training resume currently does.

The precision-4 group bound is internally coherent: the implementation scales operands to at most 16 after rounding, limits a group to 144 products, and folds that group's half-precision sum into FP32. The nominal group bound is at most 36,864, under the largest finite FP16 value 65,504. This intentional accumulation scheme is not itself an established overflow defect.

## Priority repair map

| Priority area | Findings | Why it comes first |
| --- | --- | --- |
| Nonfinite detection and trustworthy assertions | R01, R28 | Quantization and test predicates can conceal invalid results |
| Weight identity and numerical continuation | R03, R10–R15 | A successful forward/resume can use the wrong weights, precision, or optimizer state |
| Gradient semantics and tensor bounds | R02, R04–R09, R16, R18 | Supported calls can return wrong gradients or access invalid storage |
| Durable and correctly identified data | R19–R27 | Training/inference can use unrelated, stale, incomplete, or silently disabled data |
| Reliable performance and optimization contracts | R17, R29–R37 | Profiling, benchmarks, freezing, scheduling, and loss reporting need explicit semantics |

## Numerical kernels and gradient semantics

### R01 P1 FP4 convolution stores can conceal NaNs

**Evidence:** `src/lp_common.cuh:598-619,710-752,793-839`; `src/lp_s2.cu:202-234`; FP4 weight preparation at `src/lp_f4fwd.cu:51-57,66-75,82-92`. The intended NaN-preserving helper already exists at `src/lp_common.cuh:383-391` and in `src/lp_mxops.cuh:13-25`.

**Trigger and consequence:** A NaN accumulator/bias is written as MX4, or a NaN FP32 weight is prepared for FP4. These paths reduce magnitudes with `fmaxf`, which drops NaNs. E2M1 has no NaN payload, so conversion with a finite scale can yield a finite value. Decoded stored-output statistics can consequently also become finite. The normal production FP4 path is affected; loss nonfinite checks can miss the original failure. FP8 payloads preserve NaN independently, so the same maximum pattern is not equivalent evidence of FP8 concealment.

**Repair:** Use the existing NaN-propagating magnitude reduction in every FP4 preparation/output path, including shuffle reductions, and preserve its sentinel through decoded statistics.

**Future check:** Inject NaN separately into input, bias, and one weight across packed/general/ring forward kernels, regular and irregular widths, stride 2, and stored GN statistics. Require a detectable nonfinite decoded result/statistic/loss. The existing standalone conversion probe at `tests/test_mx4.c:139-157` does not cover these epilogues.

### R02 P1 Repeated backward calls corrupt accumulated gradients

**Evidence:** Accumulation is promised at `src/unet.h:42-47`. New logit gradients are scaled at `src/unet.c:1250-1253`, but the entire existing parameter-gradient array is divided by the scale at `1364` after each backward.

**Trigger and consequence:** Two backwards without `unet_zero_grad`, with scale `s != 1`, produce approximately `G1/s + G2` rather than `G1 + G2`. At the trainer's usual scale 1024, earlier contributions nearly vanish. Current ordinary training zeroes before each backward, so the public API and microbatch accumulation are the primary triggers.

Scale conditioning has a separate accumulation error even at scale 1: `src/unet.c:1366-1367` uses the cumulative stem-bias gradient to construct each new conditioning-vector contribution. With scale codes `c1,c2`, it adds `c1*b1 + c2*(b1+b2)` instead of `c1*b1+c2*b2`.

**Repair:** Merge unscaled per-backward contributions into persistent gradients, or keep a consistently scaled accumulator and unscale once at finalization. Compute conditioning gradients from this backward's bias increment.

**Future check:** Compare two accumulated backwards with the sum of independently zeroed backwards at scales 1/1024, FP16/MX8/MX4, identical/different conditioning codes, and changing gradient scales. Existing “repeated backward” tests zero between calls and do not test this contract.

### R03 P1 Prepared FP4 weights can belong to the wrong parameter set

**Evidence:** Memo validity uses pointer, padded shape, step, and generation at `src/lp_f4fwd.cuh:307-325`. Sparse preparation rewrites the same `u->wm` buffer at `src/unet.c:1516-1520`. Parameter swaps at `1497-1501` and mutations at `236-247,1691-1695,1725-1736` do not invalidate it. No owning mutation caller invokes `nn_wmemo_clear`.

**Trigger and consequence:** With FP4 plus sparse24, the frozen teacher forward at `src/train.c:196-198` can seed the memo, then the student uses that teacher's quantized convolution body. Same-step EMA validation after updates at `1040-1062` can instead reuse pre-update live body weights. Normalization/head parameters come from the newly selected set, creating a hybrid model. Dense swaps change pointers and avoid this specific sparse-buffer trigger. Loading/reinitializing an already-used model in place without advancing the step is another trigger; initial startup load before any forward is safe.

**Repair:** Version underlying parameter contents and sparse shadows, and invalidate inside every parameter-mutating API. Cache validity must follow weight content, not only the address of a reusable buffer.

**Future check:** Compare memo enabled/disabled for sparse teacher→student, live→EMA→live, update→validation, reload, and reinitialization using deliberately distinct body weights.

### R04 P1 Odd stride 2 backward treats ordinary 16 bit tensors as FP32

**Evidence:** Optimized ordinary paths require even input dimensions at `src/nn_bwd.cu:145,175`. The fallback at `185-187` selects `conv_bwd_data_s2_k<3>`, whose pointers and stores are FP32 at `src/nn_common.cuh:1164-1186`.

**Trigger and consequence:** `k=3`, stride 2, ordinary FP16/BF16 gradient storage, and any odd input dimension, for example `xs={1,16,5,7,9}`. Packed halves are interpreted as floats, and four bytes are written per two-byte output element. This is wrong mathematics and memory corruption. Usual divisible U-Net patches avoid it; the public operator API accepts it. MX storage selects a separate earlier branch.

**Repair:** Make the fallback storage-aware or extend parity decomposition to odd shapes.

**Future check:** FP16/BF16 mixed-parity Conv3D adjoints against FP32, with output canaries and future memory checking.

### R05 P1 Forced stride 2 dilation overruns scratch and misreads storage

**Evidence:** `src/nn_bwd.cu:124-126` allocates weights plus only 256 scratch bytes in tensor-core mode. `UFSM_S2DIL` bypasses parity at `145`; the fallback at `175-184` writes an entire fine-resolution FP32 tensor after those weights. `dilate2_k` is float-only at `115-122`.

**Trigger and consequence:** An ordinary even-shape stride-2 backward with tensor cores enabled and `UFSM_S2DIL=1`, using the advertised scratch size. Scratch overflows even for FP32 gradients. FP16/BF16 gradients are also misread, and the fallback selects FP32 output into a potentially 16-bit allocation.

**Repair:** Size scratch for the resolved path and implement its actual storage formats, or explicitly reject incompatible settings before launch.

**Future check:** Forced dilation for FP32/FP16/BF16 with scratch/output canaries and equivalence to parity computation.

### R06 P1 Large FP8 stride 2 weight gradients exceed the launch limit

**Evidence:** Spatial tiles and batch items occupy `grid.z` at `src/lp_s2.cu:755-763`; the kernel decodes that dimension at `324-329`.

**Trigger and consequence:** For `xs={2,32,832,832,832}`, `ys={2,32,416,416,416}`, the selected `ZC=8` yields `grid.z=26*52*26*2=70,304`, exceeding 65,535. The launch fails and weight gradients are not computed. This includes FP4 training's FP8 down-convolution fallback, as well as FP8/MX modes. Smaller deployed shapes can stay below the limit.

**Repair:** Flatten spatial/sample work into grid.x with bounded channel dimensions in y/z, or chunk launches, as already done for stride-1 families.

**Future check:** Extend `tests/test_wgrad_grid.c:40-45`, which currently selects precision 1, to FP8/MX stride 2. A low-channel/high-batch fixture can cross the limit with less memory.

### R07 P2 Narrow MX convolution outputs retain old or uninitialized bytes

**Evidence:** `mx_bw(C<=8)=8` at `src/lp_common.cuh:112-114`; MX epilogues use `MBt=bwt/16` at `700-716,738-764,790-827`. Stride 2 has the same assumption at `src/lp_s2.cu:191-234`. Dispatch does not reject these outputs.

**Trigger and consequence:** A 3D convolution outputs 1–8 MX4/MX8 channels, including a narrow split segment. `MBt` becomes zero, so the fallback writes scales but no payload; a shared-memory store can publish uninitialized payload. Default widths avoid it; custom operators/widths can reach it.

**Repair:** Implement eight-channel partial output tiles with correct layout/bounds, or reject that storage/shape combination consistently.

**Future check:** MX4/MX8 16→4, 16→8, and 4→4 at both strides and regular/irregular widths. Initialize outputs with different sentinels and require identical decoded results.

### R08 P2 Low precision tile tuning can select incompatible specializations

**Evidence:** `src/lp_f4fwd.cuh:481-493,515-541`; `src/lp_f8fwd.cuh:419-431,474-484`.

**Trigger and consequence:** Unsupported values such as `UFSM_F4_TZ=3` or `UFSM_F8_TZ=3` fall through to a `<4,2>` specialization after grid/padding were selected for different parameters. Small padded output widths can then give zero channel-tile counts or incompatible prepared-weight bounds; other widths can produce incorrect coverage. Related unvalidated stride-2 tuning values appear at `src/lp_s2.cu:756-768`.

**Repair:** Validate supported tile tuples before allocation/grid construction; dispatch from a shared typed configuration with a compatible fallback.

**Future check:** Invalid values must reject clearly or resolve safely. Assert complete output coverage for every supported tuple.

### R09 P2 Device selection resets adaptive FP16 gradient scaling

**Evidence:** `src/nn_core.cu:167` reassigns `g_gscale` on every `nn_init` when `UFSM_F16` is enabled. The trainer repeatedly selects devices at `src/train.c:921,1022` and halves the scale after overflow at `1032`.

**Trigger and consequence:** Environment-enabled FP16 training experiences an overflow or uses a CLI scale different from the environment/default. The next device selection restores `UFSM_GSCALE` or 1024, defeating recovery and overriding runtime configuration. CLI FP16 without that environment switch avoids this reset.

**Repair:** Apply environment defaults once and separate device selection from numerical configuration.

**Future check:** Set a reduced scale, reselect the same/another device, and require it to persist through the next backward.

## Checkpoints model contracts and optimizer state

### R10 P1 Flagless training resume changes precision or optimizer

**Evidence:** `src/train.c:413-424` reads runtime metadata but restores only GN semantics and nonzero stem input precision. Global precision, FP16, and optimizer receive CLI defaults. `455-468` independently applies the current FP4/policy defaults. Saved fields are at `783-790`; `src/checkpoint.h:7-8` lists the runtime schema.

**Trigger and consequence:** Resume without repeating original flags. For example, `--fp4 0 --opt muon` can resume with MX4 activations and AdamW. Restoring the stem setting alone does not preserve its meaning, because input conversion depends on the newly selected MX mode at `src/unet.c:303-304`. A wrapper that repeats the recipe can protect its usual path; the direct resume command does not.

**Repair:** Use checkpoint settings as defaults, then explicit CLI overrides. Version and persist omitted training-critical settings such as gradient scale, MX4 gradient mode, stochastic-rounding configuration/state, and required optimizer hyperparameters. Distinguish continuation from deliberate warm start.

**Future check:** Uninterrupted versus flagless resume for each precision preset, custom policy, Muon, ANVIL, and memory mode; compare resolved execution plus the next update.

### R11 P1 Missing Adam payloads are accepted as a successful resume

**Evidence:** `src/unet.c:1692-1695` rejects truncated weights/EMA but merely breaks on a short `m` or `v` array and returns the saved step. `src/train.c:610-612` treats that result as a successful resume.

**Trigger and consequence:** A checkpoint contains both weight arrays but incomplete/absent Adam moments and no required Muon trailer. A fresh object substitutes zero moments; a reused object can retain unrelated moments. Bias correction still uses the old large step. A partially loaded first moment creates a hybrid state.

**Repair:** Validate all declared arrays before mutating the model. Separate strict training restore from explicitly supported weights-only/legacy inference loads.

**Future check:** Truncate every payload array at several offsets. Strict resume must reject without partially changing live state. Current tests reject missing declared Muon momentum but do not establish this Adam contract.

### R12 P1 True FP4 weight checkpoints discard error feedback

**Evidence:** Residual allocations and construction at `src/unet.c:1536,1542-1547`; consumption at `1563,1571` and `src/nn_elem.cu:241-255,265-274`. `src/unet.c:1633-1647` saves snapped shadows, common moments, and optional Muon momentum, omitting live/EMA residuals and scales.

**Trigger and consequence:** Save/resume with true packed weights `--wq 4`. Sub-FP4 update history is lost. Reload repacks already snapped shadows, which cannot recover the missing error feedback or effective EMA state. This is separate from ordinary FP4 operand computation with FP32 masters.

**Repair:** Version the packed payload and store packed values/scales plus live/EMA residuals/scales and any required rounding counter.

**Future check:** Round-trip next update and EMA with updates below a quantization step; cover packed FP8 as well as FP4.

### R13 P1 Changing packed weight precision reuses an incompatible layout

**Evidence:** `src/unet.c:1524-1549` updates `u->wq` immediately but initializes offsets/allocations/packing only when `wq_q` is null. The trainer loads before applying explicit `--wq` at `src/train.c:610,629`.

**Trigger and consequence:** Resume FP4 packed weights with `--wq 8`, change 8→4, disable packing and update ordinary masters before re-enabling it, or load new packed weights into an already-packed object. The old offsets, byte capacity, payload format, and residual state remain. A 4→8 transition can exceed allocation bounds; reverse transitions decode incompatible bytes and omit required residual initialization. An unchanged off/on toggle alone does not establish corruption.

**Repair:** Reject unsupported transitions or explicitly rebuild/repack all state from a defined source. Validate bits centrally and make repeated setters well-defined.

**Future check:** Every format transition and reuse with two distinct packed checkpoints, including allocation canaries and next-update equivalence.

### R14 P1 ANVIL checkpoints omit persistent rails and lane energy

**Evidence:** `src/unet.c:1467-1477` allocates `anvil_v1` and per-convolution energy `E`; updates consume them at `1484`. Save/load at `1633-1647,1684-1700` preserves `muon_mom` but neither additional state.

**Trigger and consequence:** ANVIL resume, even with the correct optimizer flag repeated. The fast rail resumes, slow rail restarts at zero, and energy restarts at one. This changes the next update, especially after the slow rail engages at step 514 (`1480-1482`).

**Repair:** Serialize every persistent ANVIL component with a versioned optimizer identity/layout; exclude only transient matrix scratch.

**Future check:** Next-update equivalence before/after step 514 and after multiple saved/resumed updates.

### R15 P1 Checkpoint architecture and required header fields are not validated

**Evidence:** `src/unet.c:1653-1672` parses with `strstr`/`atoi` and returns success with missing fields. `1687-1689` compares only parameter count. Construction at `179-201` trusts configuration. The stricter runtime JSON reader does not validate the model architecture fields.

**Trigger and consequence:** Same-size but semantically different architectures, or malformed headers. Changing GN group count keeps the parameter count identical and is accepted. Missing/invalid widths, channels, groups, or level counts can reach invalid allocations/indexing. Training rebuilds its own widths/groups and only copies selected structural fields at `src/train.c:478-483`.

**Repair:** Parse/validate the complete header with required typed fields, range/product checks, and full architecture comparison. Keep adaptation in explicit warm-start APIs. Restore should be transactional.

**Future check:** Reject identical-parameter-count/different-G loads and malformed/missing/negative/oversized fields before any allocation or state change.

### R16 P1 Graph constraints are accepted without safe implementation

**Evidence:** `src/train.c:483-484`; `src/unet.c:179-201,1007-1012,1099-1102`; `src/nn_elem.cu:18-24,72-89`; `src/nn_common.cuh:1681-1695`.

The following cases require one central graph/shape validator:

| Case | Source consequence | Suggested disposition |
| --- | --- | --- |
| Ordinary FP16/FP32 GroupNorm with width 17 and G=8 | `cpg=C/G=2`; channel 16 uses group 8, outside the eight mean/rstd entries | Require divisibility by effective group count, or implement uneven groups explicitly |
| Zero/negative width, invalid/empty level list, nonpositive G | Unchecked allocation products/divisors/shifts reach model setup | Reject before parameter planning |
| One-level network, e.g. `--widths 32` | No decoder is constructed, but head forward/backward reads `dec[0]`; prediction accepts nlev=1 at `src/predict.c:173` | Implement head/backward/planner against enc0 or require nlev≥2 |
| Caller input channels differ from cfg.cin | Forward checks spatial divisibility but not channel equality | Enforce the documented input shape contract |

Some MX helpers reject invalid groups earlier; that does not make ordinary storage safe. Default widths/groups avoid the demonstrated nondivisible-width case.

**Future check:** Reject invalid configurations before allocation; cover all valid smallest topologies and storage modes. The one-level case is a topology defect, not a quantization approximation.

### R17 P2 Head only training does not freeze old optimizer updates

**Evidence:** `src/unet.c:1490-1494` zeros disallowed gradients; `src/train.c:1037-1042` then runs the full optimizer. Adam still applies old moments at `src/nn_opt.cu:7-11`; Muon/ANVIL retain momentum/rails too.

**Trigger and consequence:** Resume a trained model with `--train-head-only N --wd 0`. Backbone weights continue moving from old optimizer state despite zero gradients. Nonzero decay also moves frozen parameters on fresh starts. The source comment's suggestion of zero weight decay addresses only decay.

**Repair:** Restrict optimizer ownership/update intervals to selected parameters. Define how frozen optimizer and EMA state is handled.

**Future check:** Start with nonzero optimizer state and require bit-identical frozen parameters across multiple steps for every optimizer.

## Inference data and storage correctness

### R18 P1 Winding prediction channel selection accesses beyond logits

**Evidence:** `src/predict.c:175` accepts any valid head row. `394` shifts the logits pointer by that row, but `395-397` always accesses another full-window row for winding.

**Trigger and consequence:** A two-output winding checkpoint with `--channel 1`. The winding gate writes and the copy reads one window beyond the logits allocation.

**Repair:** Validate the task's channel contract before GPU initialization; address surface and winding rows explicitly, independently of exported channel selection.

**Future check:** Reject incompatible winding channel selections and verify accepted mappings.

### R19 P1 Remote cache identity omits the store root

**Evidence:** `src/zarr3.c:82-94,171-186,254-261,495-497`.

**Trigger and consequence:** Two HTTP roots use the same relative array key and cache directory. Cache paths contain only relative key/shard/item, so the second root can read the first's bytes, shard index, or missing marker. Matching shape/codec makes this silent numerical-data corruption. Replacement under an unchanged remote identity also requires a revision invalidation policy.

**Repair:** Namespace by canonical store identity and dataset/revision identity. Bind negative-cache entries to that identity as well.

**Future check:** Two distinct remote roots with identical keys must remain isolated, including missing-shard caching and source replacement.

### R20 P1 Buffered output failures can publish corrupt shards

**Evidence:** `src/z3w.c:146-153` checks writes but ignores `fclose` before rename. Metadata paths at `59-68,170-180` similarly omit close/write-result checks.

**Trigger and consequence:** A buffered write fails while flushing/closing, such as disk exhaustion or an I/O error. Truncated data can replace a completed shard and still report success. Higher-level staging cannot protect against false command success.

**Repair:** Check stream/close errors before publication; write metadata through checked temporary files and propagate failures.

**Future check:** Inject write/flush/close failures and require a failed operation that preserves previous completed output.

### R21 P1 All fill rewrites and prediction reruns retain stale data

**Evidence:** `src/z3w.c:129-155` performs filesystem writes only when some encoded data is non-fill. `z3w_create` allows existing directories. Prediction can skip all-air shards at `src/predict.c:351,375,439`.

**Trigger and consequence:** Rewrite an existing nonzero shard as all fill, or predict into an existing output where a formerly populated shard becomes air. The command can succeed while readers still observe old labels/probabilities.

**Repair:** Define replacement semantics: remove old all-fill shards and stage complete output atomically, or reject nonempty destinations. Preserve wrappers that already stage predictions (`tools/eval_holdouts.py:33-60`).

**Future check:** Non-fill→fill and fill→non-fill readback, plus a prediction rerun that makes old populated regions empty.

### R22 P2 One ignored label suppresses every target channel

**Evidence:** `src/sample.c:633-652,670-673` unions channel-specific ignore flags into one spatial mask, which the loss receives at `src/train.c:275`.

**Trigger and consequence:** Two active target channels have different valid/trusted regions. A valid recto label can disappear because the other channel is ignored, and vice versa. Disjoint trust bands can remove most support.

**Repair:** Keep per-channel label validity, intersected with shared CT/geometry validity. If complete-case supervision is intended, document/enforce it and quantify discarded valid labels.

**Future check:** Independently valid channels with disjoint ignore regions must retain their own loss/gradient support.

### R23 P2 Standalone evaluation ignores cropped prediction origin

**Evidence:** `src/eval.c:173,194-202` claims metadata handling but starts from zero and changes origin only from `--pred-origin`. Prediction writes origin at `src/predict.c:273-274`.

**Trigger and consequence:** Score a nonzero-origin crop without repeating the origin. It is compared to labels at another location and can produce plausible wrong scores. The holdout wrapper supplies an explicit native origin at `tools/eval_holdouts.py:129-130` and avoids the common trigger.

**Repair:** Read validated metadata by default. Define units across input CT level and derived pyramid levels: stored prediction `bo` is in selected input-level voxels, while current evaluation CLI origin is native-level and shifted by evaluation level. Also correct the help's subtraction/addition description.

**Future check:** Perfect cropped predictions should score identically with automatic metadata and explicit override at source levels 0/1 and derived levels.

### R24 P2 Cache configuration is a dangling process global

**Evidence:** `src/sources.c:12,87-88,189,219,224,354,361`.

**Trigger and consequence:** Free a source set owning `S->cache`, then open another set lacking a cache, or interleave sets with different cache roots. Global `g_cache` borrows the freed allocation or points to another set's configuration. New opens can use freed memory or the wrong cache.

**Repair:** Own cache configuration per source/source set and pass it explicitly; any retained override must have owned lifetime.

**Future check:** Interleave loading, opening, freeing, and reloading two source sets with different cache roots.

### R25 P2 Temporary metadata failures permanently disable supervision

**Evidence:** `src/sources.c:219-229,259-275`; eligibility caching at `src/sample.c:197-220`; missing target fallback at `575-577`.

**Trigger and consequence:** A listed remote level remains unavailable through startup retries but later recovers. The code marks it absent and never retries; a required target can lose supervision while another target keeps the source eligible. The sampled source/scale distribution changes for the remainder of training.

**Repair:** Distinguish authoritative absence from transport/access/decode failure. Required listed levels should fail setup or follow an explicit retry policy; only authoritative missing data should be cached as absent.

**Future check:** Simulate a recovering listed level and require recovery or a clear startup failure, never silent permanent omission.

### R26 P2 Prefetch success does not establish cached availability

**Evidence:** `src/sample.c:867-869,918-922` counts failures but returns zero. `src/zarr3.c:213-230` discards cache-publication errors; `383,396-397` counts fetched chunks anyway. Without a cache, `374` returns without work.

**Trigger and consequence:** Failed remote reads, unwritable/full cache, failed publication, or no cache. Automation can regard prefetch as complete even though later training still requires the network or fails.

**Repair:** Return aggregate failure; distinguish optional caching during reads from required persistence during prefetch, and count published cache entries.

**Future check:** Fail fetch and cache writes independently; a successful prefetch must support later reads with remote access unavailable.

### R27 P2 Invalid prefetch worker counts can hang forever

**Evidence:** `src/main.c:259-262` forwards unchecked thread counts. `src/sample.c:912-915` starts that many workers, ignores thread-creation failures, and waits for every marked chunk to increment progress.

**Trigger and consequence:** `prefetch --threads 0` with work remaining has no worker and never finishes. Negative counts request invalid allocations; worker creation failures can leave unprocessed ranges and the same hang.

**Repair:** Validate positive bounded counts, check thread creation, and run work on the caller or propagate startup failure.

**Future check:** Invalid counts reject immediately; injected thread failures terminate predictably.

## Tests performance tools and consistency

### R28 P1 Some numerical assertions accept NaN results

**Evidence:** `tests/test_nn.c:16-17,36-49,241-247,465-472`; `tests/test_unet.c:101-118,174-178`; standalone Muon Gram check at `tests/test_muon.c:57-64`.

**Consequence:** `NaN > worst` and `NaN >= tolerance` are false; `fmax` can discard NaNs. Entirely invalid gradients can leave a worst error of zero. The whole-network test can print `FAIL` for NaN while failing to increment its failure count.

**Repair:** Require finite operands/results and positive acceptance predicates such as `isfinite(error) && error < tolerance`. Check the actual tensors, not merely a maximum computed with NaN-dropping operators.

**Future check:** Exercise comparison helpers with one/all NaNs, infinities, and zero references. Preserve newer MX comparators at `tests/test_mx.c:17-25`, `test_mx4.c:23-33`, and `test_grad_mx4.c:33-37`, which correctly reject NaN aggregate errors.

### R29 P2 Precision sweep crashes and can report errors as success

**Evidence:** `tests/prec_sweep.c:18-20` sizes `lname` to `UNET_NSLOT` (46) but initializes only 22 entries; loops at `87-88,127-128` dereference all entries. `196-197` prints CUDA errors and returns zero; the skip path at `105` returns without its final check.

**Trigger and consequence:** Normal sweep or SHOW reaches a null layer name at slot 22. Independently, failed measurements can be accepted by automation through exit status zero. The candidate loop also unconditionally excludes FP4 weight gradients at `131`, so the tool does not evaluate the current default's full policy search space.

**Repair:** Derive active graph slots from model metadata, check candidate bounds, and fail on runtime/nonfinite measurement errors. Explicitly describe excluded candidate families.

**Future check:** Short normal/SHOW/SKIP sweeps enumerate only active slots, return valid resolved policies, and propagate failures.

### R30 P2 All GPU profiling uses invalid device events and mismatched regions

**Evidence:** One global event array/initialization/cursor at `src/nn_core.cu:223-235`; advertised all-GPU collection at `src/train.c:1094`. Split handoff at `src/split.c:79-91` can occur inside profiled normalization/reduction calls, including `src/unet.c:975-976` and `src/nn_fusedgn.cu:49`.

**Trigger and consequence:** After events are created on one device, profiling another GPU records those device-owned events on the wrong device. CUDA statuses are ignored. Also, begin does not reserve a slot; another split side can overwrite its start or advance the shared cursor before its end. Collection synchronizes only the current device. Resulting per-layer/all-GPU timing cannot be trusted.

**Repair:** Use per-device pools, explicit region tokens or per-thread active IDs, checked event operations, and collection on every participating device.

**Future check:** Sequential two-device profiles and split profiles containing reductions must have correctly paired event counts and valid nonzero durations.

### R31 P2 FP4 benchmarks measure different cache and statistics lifecycles

**Evidence:** `tests/bench_mem.c:22,32-39` uses zero external gradients and zero decay, and never advances `nn_set_sr_step`, unlike production at `src/train.c:890`. `tests/bench_f4conv.c:31-36` leaves `sp.wkey=0` and `sp.stored_stats=0`; production sets layer identity and can compute decoded stored statistics.

**Consequence:** The memory harness reuses prepared FP4 weights across all pseudo-training iterations, omitting production's per-step preparation. Its zero-gradient workload is not a full loss-derived training step. Conversely, the isolated forward benchmark prepares weights every call and times legacy accumulator statistics, which can mis-rank epilogue/preparation optimizations. Because the memory harness's weights do not substantively update, it is not evidence of stale changed-weight outputs.

**Repair:** Separate cold preparation, same-step reuse, inference reuse, and complete training modes. Print resolved storage/statistics/cache settings and use representative gradients.

**Future check:** Count preparation calls per step, check outputs against each matching reference graph, and compare with end-to-end trainer rates.

### R32 P2 Performance baseline converts failed stages into success

**Evidence:** `tools/perf_baseline.sh:12,19-23,31-35` converts test/training/profile/inference failures to successful echo/diagnostic paths.

**Consequence:** A failed required test or benchmark can still produce summary artifacts and a successful final exit. `set -e` does not enforce failure in those explicitly handled branches.

**Repair:** Collect diagnostics if useful, but track failures, invalidate dependent summaries, and return failure when a required stage fails.

**Future check:** Stub failures for test/train/predict/roofline and require failed-stage metadata plus a nonzero final status.

### R33 P2 Core training numbers lack validation before arithmetic

**Evidence:** Numeric parsing at `src/train.c:352-355,372-373,423-437`; arithmetic at `1003-1016,1040-1042,1056,1060,1097`.

**Concrete triggers:** Zero logging/validation/checkpoint intervals reach modulo by zero. `cooldown=0` reaches a zero denominator when the final WSD step enters decay after warmup; a warmup spanning the whole run avoids that branch. Nonfinite learning rates/scales and unknown optimizer/schedule names lack a common rejecting gate; unknown optimizer names fall through to AdamW, and unknown schedules fall through to cosine. The finite forward/gradient check precedes the update and does not validate the newly computed LR.

**Repair:** Centralize typed option validation, require finite appropriate bounds, reject unknown enum values, and define zero-warmup/cooldown semantics without undefined arithmetic. Keep schedule validation consistent for step and wall-time modes.

**Future check:** Invalid values reject before samplers/GPU allocations; boundary schedules produce finite intended rates through their final update.

### R34 P2 Loss reporting and data parallel weighting change the objective

**Evidence:** The documented objective is mean over active `(n,c)` at `src/nn.h:223-224`. `src/nn_loss.cu:43-57` emits per-channel means plus total active count; `src/train.c:313-317` averages those channel means equally. Data-parallel gradients are averaged equally by rank at `990-995` after each local loss has already normalized by its local active count.

**Trigger and consequence:** Channels have unequal active sample counts, or GPUs have unequal active target counts. Logged/validation loss is a mean of channel means rather than the differentiated active-pair mean. Equal rank weighting is also different from evaluating the combined batch under that active-pair objective; a rank with fewer labels can be overweighted. Spatial split uses a distinct global-statistics/summed-gradient scheme and must be evaluated separately.

**Repair:** Choose and document one objective. Preserve per-channel counts/raw loss sums for matching reporting, and weight data-parallel contributions by global active counts when reproducing the combined-batch active-pair mean.

**Future check:** Imbalanced labels across channels/ranks, including a rank with no active supervision, must match the chosen single-device combined-batch reference and its reported loss.

### R35 P2 Adam bias correction counts skipped attempts as updates

**Evidence:** The loop step advances at `src/train.c:885`; nonfinite steps skip optimizer updates at `1029-1035`; later updates still receive that attempted step at `1040-1042`. Adam correction uses the supplied step at `src/nn_opt.cu:163-165` and `src/nn_elem.cu:299` for packed Adam.

**Trigger and consequence:** An overflow/nonfinite attempt is skipped and training later recovers. Moments have seen fewer updates than the bias-correction exponent assumes. This changes update scale; repeated skips amplify the mismatch.

**Repair:** Track and checkpoint successful optimizer updates separately from schedule/data-attempt steps. Define deliberate fine-tuning resets independently from loading old moments.

**Future check:** Insert a skipped update, then compare resumed updates with a reference optimizer given the same successful gradients and update count.

### R36 P2 Validation cadence is nested inside logging cadence

**Evidence:** `src/train.c:1056-1060` checks validation only after entering the logging branch.

**Trigger and consequence:** Logging every 20 steps and validation every 30 steps validates at common multiples such as 60, rather than every requested 30 steps. Best-checkpoint selection and experiment monitoring consequently differ from the CLI setting. Common production intervals that divide evenly avoid it.

**Repair:** Evaluate validation cadence independently, then include its result in logging; preserve final/stop behavior.

**Future check:** Coprime or nondividing log/validation intervals must each execute at their own requested steps.

### R37 P3 FP4 memo shape identity omits real channel counts

**Evidence:** `src/lp_f4fwd.cuh:307-326,474-478,496-511` stores padded channel dimensions; ordinary plane-major calls use sentinel split dimensions and do not identify actual Ci/Co.

**Trigger and consequence:** A custom caller reuses pointer/key/step/generation with different actual channels that share padding, for example Ci=33→48, and compatible allocation capacity. Prepared weights depend on actual layout/counts, but the memo can treat them as identical. Fixed-channel U-Net layers avoid this narrow trigger.

**Repair:** Include actual Ci/Co and preparation layout/mode in the key.

**Future check:** Same-padding shape changes with the same identity must agree with memo-disabled preparation.

## Source supported performance opportunities

These are implementation leads, not measured speedups. Fix R28/R30/R31/R32 before using timing data to choose a replacement.

| Opportunity | Evidence and visible cost | Implementation direction and required future measurement |
| --- | --- | --- |
| Avoid a second gradient-norm reduction when clipping | Trainer already gets norm at `src/train.c:1024`, then `unet_clip_grad` recomputes it at `src/unet.c:1376-1378`. Reduction launches 1024 blocks and synchronously copies 1024 floats at `src/nn_opt.cu:179-185` | Reuse the known norm and scale directly; compare complete-step latency when clipping activates |
| Cache immutable FP8 prepared weights safely | Repeated preparation at `src/lp_f8fwd.cuh:273-278,414-416,468-471` and `src/lp_s2.cu:270-283`; the FP4 recipe still uses FP8 stem/down operations | Add content-versioned inference/within-step caching after R03; measure preparation share and cache memory |
| Extend direct parity stride-2 backward to wide MX layers | `src/lp_mx_s2bwd.cu:70-109` dilates gradients to the fine grid and performs roughly eight times parity's MACs; narrow direct paths exist | Generalize direct parity for supported wide channels; compare both temporary bytes and whole-step throughput |
| Cache FP4 persistent launch choices | `src/lp_f4wgrad.cuh:981-989` scans every `k=1..nitem` on each eligible call/slab | Cache by shape/config or evaluate mathematically sufficient breakpoints; measure host launch overhead |
| Budget host scratch and pinned queues by patch size | Each worker allocates `10*P^3` scratch plus target buffers at `src/sample.c:766-771`; queues pin `B*(8+NCH+1)*P^3` bytes with FP16 input at `126-130`. Trainer uses 12 workers and `4*ng+2` slots at `src/train.c:355,500` | At P704, worker `big` alone is about 3.25 GiB: 12 workers need about 39 GiB before other buffers. Ten FP16 B1 queue slots at NCH=2 pin about 35.75 GiB. Validation/offload add more. Plan host memory explicitly and tune workers/queue depth from measured producer demand |
| Restrict eager metadata opens and reuse occupancy | Every sampler startup opens all levels/channels at `src/sample.c:948` and `src/sources.c:240-253`. Training and validation can separately scan occupancy at `src/sample.c:817-853,979-981` | Open requested levels plus target/probe/occupancy dependencies; share immutable validated occupancy and avoid constructing it for holdout draws that do not use it |
| Account for retained global CUDA workspaces | Maximum-size per-device buffers remain at `src/lp_common.cuh:901-910`; memo allocations remain at `src/lp_f4fwd.cuh:319-322` beyond model teardown | Expose allocated bytes, reserve before timing, and provide a safe release policy for model/shape changes. This is a reuse tradeoff, not an established default-model leak |
| Derive roofline traffic from actual storage | `tools/roofline.py:17,41,85-87` assumes 4.25-bit activations/8.25-bit gradients and only the first observed arithmetic path | Use shape/block padding, stem/head formats, and resolved gradient format. MX4 C16 costs 4.5 bits/channel; MX8 C4 with an eight-channel row costs 18 bits/real channel. MX4 gradients can halve assumed gradient traffic. The current model is not always a valid lower bound |

The upload loop at `src/train.c:963-986` visibly uses separate copy-stream events and double buffering. It should be profiled for real overlap after event instrumentation is fixed. The previous blanket claim that every next upload waits for all current computation was not re-established for the current source.

The API header still describes all tensors as FP32 and operations as synchronous (`src/nn.h:1-2`), while current registered MX/16-bit tensors and kernel launches have different contracts. Also, “TF32” names refer to enabling a mixed tensor-core backend whose default products are FP16/BF16. Update terminology and buffer/stream contracts alongside implementation repairs so callers can reason about actual precision and completion.

## Future numerical acceptance matrix

This matrix is intended for the implementation agent's later validation phase. Nothing in it was executed during this review.

| Order | Checks | Required evidence |
| --- | --- | --- |
| 1 | Make all numerical comparators reject unexpected NaN/Inf; check FP4 nonfinite propagation through every epilogue | Injected bad values fail reliably; unaffected finite blocks retain correct values |
| 2 | Conv3D forward/adjoints/wgrad with exactly representable operands and separately staged/dequantized references | Distinguish indexing/layout/scaling bugs from expected quantization error |
| 2 | Ordinary FP16/BF16 odd stride-2 and forced dilation; narrow MX outputs; invalid tiles; FP8 stride-2 grid boundary | Correct bounds or explicit rejection, with storage/layout-aware references |
| 3 | Accumulation at scales 1/1024 and changing conditioning/scale codes | Accumulated GPU result equals summed independent backwards |
| 3 | Strict checkpoint payload/header validation and next-update continuation for AdamW/Muon/ANVIL/packed FP8/FP4 | Matching parameters, moments, rails, energy, residuals, EMA, update counters, and relevant rounding state |
| 3 | Sparse teacher/live/EMA swaps and load/reinit with memo enabled/disabled | Every forward uses the selected parameter generation |
| 4 | Full-network finite logits and per-tensor/parameter-gradient bounds for resolved production presets | Finite outputs plus agreed absolute/relative/cosine criteria, including small gradients; print executed arithmetic and storage |
| 4 | Recompute/offload/lean/fused upsample configurations, fresh/repeated backwards, and actual two-GPU split | Match each configuration's documented numerical graph; test event lifetimes and aliasing |
| 4 | Task/channel contracts, imbalanced supervision, head freezing, cropped evaluation, cache isolation, writer failure/rewrite | Chosen semantics reproduced through public CLI/API paths |
| 5 | Complete-step training and deployed inference, including transfer, preparation, optimizer, reductions, I/O and host memory | Checked successful runs, valid per-device timing, resolved manifests, memory accounting, and workload identity |
| 6 | Matched real-data quality across precision presets using fixed data/targets/calibration and acceptance splits | Measure task quality/topology tradeoffs independently of primitive numerical tolerances |

Additional concrete coverage gap: the stochastic/subnormal cooperative weight-gradient staging comparison at `tests/test_wgrad_staging.c:34-38,96-99` leaves the gy pre-pass enabled. That pre-pass is selected at `src/lp_f4wgrad.cuh:953-956`, while cooperative staging requires `!gpre` at `424`. Its SR/subnormal comparisons therefore bypass the intended path. Disable the pre-pass explicitly for that check and assert executed path identity; test pre-pass equivalence separately.

Useful existing tests should be extended, not replaced indiscriminately. `tests/test_mx4.c:503-574` covers exact-grid FP4 wgrad with several storage formats and ragged/split inputs; `637-660` checks a matching-scale adjoint; `723-743` compares gy pre-pass and in-kernel rounding. `tests/test_split.c:155-230` covers several precision/recompute/offload modes. `tests/test_checkpoint.c:38-79` already verifies AdamW/Muon continuation and declared momentum handling. The main Makefile runs most numerical coverage through `VERIFY=1`; add a compact numerical smoke matrix against the deployed `VERIFY=0` binary as well.

Approximation-only considerations remain separate from defects: FP8 quantization before fused coarse interpolation (`src/lp_common.cuh:478-502`), approximate tanh-based SiLU, stochastic gradient rounding, sliced gradient requantization, and FP16 saturation before GN intentionally change numerical behavior. Stress large-offset/low-variance normalization and finite extremes, but do not label these indexing defects without a reference matched to their documented computation.

## Earlier findings already addressed or not reestablished

| Earlier concern | Current source assessment |
| --- | --- |
| Resume immediately reinitializes loaded weights | Addressed: initialization is confined to the non-resume branch at `src/train.c:608-615` |
| Packed/Muon convolution weights receive overlapping ordinary optimizer updates | Sorted ownership intervals and nonconvolution updates are present at `src/unet.c:1384-1451`; that earlier double-owner defect was not reestablished. R13/R17 concern different transitions/contracts |
| Low-precision launch errors never reach `nn_check` | Addressed: `src/nn_core.cu:168-173` reads `lp_check`; corresponding propagation coverage exists. R29/R32 concern callers that still discard failure |
| Every fused GN always normalizes rounded values using unrounded statistics | Current stored-statistics mode has decoded-output handling. Legacy mode still exists and requires explicit interpretation; R01 concerns concealed nonfinites, not blanket GN mismatch |
| FP16 recompute universally drops forward saturation | Current replay preserves the relevant output-group argument and saturation policy in reviewed paths; no new blanket replay defect established |
| Shutdown publishes partially filled sampler batches | Current `src/sample.c:798-807` tracks filled count and only publishes complete batches |
| Validation snapping moves patches outside the heldout box | Current `src/sample.c:527` restricts snapping to non-holdout/non-region training draws; bounds use upward-rounded starts at `149-167` |
| CRC table initialization races | Current `src/z3w.c:26-27` uses `pthread_once` |
| New MX shared-memory attributes are initialized only for one GPU | Dirty stride-2/up-backward changes track initialization per device in the reviewed paths |
| Scanner augmentation scratch overlaps live ignore data | No such defect established: five-cube-byte scratch begins at `big+p3` and ends before ignore data at `big+9*p3` |

The new scanner augmentation silently becomes identity for P>1024 (`src/scan_augment.c:120`) without a matching setup rejection. Treat this as a lower-priority option-contract issue if such patches are intended to be supported.

## Suggested implementation sequence

1. Repair nonfinite propagation and numerical assertions together so subsequent validation can detect failures.
2. Fix prepared-weight invalidation, accumulated gradients, unsafe stride-2/narrow/tile paths, and central graph validation.
3. Make exact resume transactional and complete, including precision defaults, optimizer identity/state, packed residuals, and successful-update counters.
4. Repair task/channel bounds, data identity, durable publication/replacement, validity masks, and prefetch/setup failure handling.
5. Fix profiler and benchmark fidelity, then measure the performance opportunities on the actual resolved training/inference graph.
6. Reconcile API documentation/manifests and run matched numerical, continuation, multi-GPU, and task-quality checks before declaring a precision recipe qualified.

The document is the only artifact added by this review. Existing implementation changes and reports were left untouched.
