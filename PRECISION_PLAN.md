# Handwritten mixed precision Conv3d training and inference plan

Updated: 2026-10-01. Target: UFSM on RTX 5060 Ti 16 GB. This document replaces the supplied plan and now reconciles it with the implemented engine, including the latest sampler, Muon, and schedule changes. The source review covers commit `6b0390a963043c3eef345766ba52b2a0f32325a6`, an intermediate sampler/storage cut, the detailed final cut `51125cd5e32f598c4ab5dcf967585db19b89fb1d`, and the small closing delta through `c64dab9fb14d2cd0444480d83a2aeadf41a2dde3`. See [FINDINGS.md](FINDINGS.md) for source cuts and individual proofs. No project code, compiler probes, tests, training, inference, benchmarks, or reproducer scripts were executed during this re-review, and no implementation files were changed.

The final captured uncommitted delta based on `6fd40ac0f71e2374fecaeebc15ad072e92808ff8` adds batched Muon. Its fingerprints and separate review boundary are recorded in the findings document. Changes after that captured boundary are outside this analysis.

Build one model graph with explicit precision policies for each operation and tensor. Establish FP32 correctness, qualify BF16/FP16 mixed training, then validate the implemented FP8/MXFP8 and MXFP4 paths with quantization-aware training. Add native NVFP4 as a separate comparison. Keep reductions, optimizer state, and sensitive operations in FP32. Promote individual operations when the lower precision exceeds the accuracy budget. FP4 backward computation is a separate research phase.

The objective is the fastest complete training and inference pipeline that preserves surface quality on held-out scrolls. Parameter count, packed tensor size, and advertised tensor throughput are inputs to that decision, not acceptance criteria by themselves. No evidence in the supplied conversation establishes a minimum useful parameter count or predicts FP4 accuracy for this task.

## Analysis of the supplied plan

The proposed order of development is broadly useful, but several numerical claims and implementation assumptions would produce an unreliable design if adopted literally.

| Supplied claim or assumption | Assessment and replacement |
| --- | --- |
| A usable surface model needs approximately 90k parameters or 60–90 voxels of context | These are unvalidated task hypotheses. Derive parameter counts and convolutional receptive fields from the exact architecture, then compare quality on held-out difficult regions. |
| Four levels means four downsampling operations | A four-level encoder normally has three transitions. Specify levels and downsampling operations separately. UFSM currently has four levels and three downsampling operations. |
| Distillation is the path to a tiny network | UFSM's current design explicitly uses released ground truth and excludes teacher models. Train from random initialization and use ground-truth QAT in this plan. |
| Wider channels always make FP4 worthwhile | Wider layers may fill tensor tiles better, but also increase work and activation memory. Compare complete architectures; do not widen the full-resolution layers merely to match a hardware tile. |
| FP4 costs approximately 1.125 bytes per channel including scales | A normally packed NVFP4 tensor costs 0.5625 bytes per value before padding and alignment. 1.125 bytes is the combined input-plus-output cost for two equal-sized tensors. |
| FP8 and BF16 peaks can be read from the FP4 peak by halving repeatedly | Accumulator precision matters on GeForce. Distinguish FP16 and FP32 accumulation and distinguish architectural estimates from measured rates for a particular instruction family. |
| All Blackwell instructions apply to this GPU | Use the SM 12.0 warp-MMA path. Do not design around the datacenter TMEM and tcgen05 execution model. Validate each optional instruction against its target requirements. |
| sm_120a is the only possible feature target | It is a suitable initial target for this GPU. Current documentation also describes family targets. Pin the actual compiler/PTX combination and enable only the features it documents and accepts. |
| Every precision is an instantiation of the same convolution template | Share convolution geometry, dispatch, and testing. Use distinct SIMT, unscaled tensor-MMA, and block-scaled tensor-MMA implementations where packing and scale handling differ. |
| Every FP4 or FP8 tensor needs channels padded to 64 | Instruction K size does not automatically impose a global channel width. Compare packing across taps, channel tails, and fallback kernels. Keep logical and physical dimensions distinct. |
| A halo loaded once means each input is read from DRAM once | The reuse is within a CTA and channel stage. Neighboring output bricks and separate output-channel CTAs reload overlapping inputs; caches may reduce actual DRAM traffic. |
| TMA gives free convolution padding and arbitrary shifted access | Packed data, scales, descriptors, alignment, and synchronization still need explicit handling. Start with ordinary cooperative loads; treat TMA as an optimization. |
| One lane-quad shuffle handles every 16-channel output scale | Reduction ownership depends on the accumulator mapping, N tiling, and warp partitioning. Make complete output quantization blocks owned by a defined CTA and reduce across the actual participating lanes. |
| GroupNorm can be completed in the convolution epilogue | GroupNorm needs statistics over the logical group and the full spatial extent of each sample. An ordinary CTA cannot normalize its output using incomplete global statistics. |
| Convolution activations never need a higher-precision DRAM representation | GroupNorm, loss, backward recomputation, and runtime fallback may need one. Specify the retained tensors separately for training and inference. |
| Changing downsampling to 2×2×2 and upsampling to transposed convolution is just a kernel choice | Those changes alter the model. First preserve UFSM's 3×3×3 stride-2 downsampling and trilinear upsampling. Compare alternatives as separate architectures. |
| Backward-data is always forward convolution with flipped weights | Stride, padding, output shape, and quantization orientation matter. Derive the adjoint from the forward coordinates; use a separate gather or parity-based implementation for stride 2. |
| Weight gradients are covered by the forward tile layout | Their reduction axis is batch and output space, not input channels. They need different staging, scale blocks, and reduction scheduling. |
| Quantized outputs should match an FP64 convolution reference bit for bit | Format conversions and packing can be bit-exact. Different valid accumulation orders can cross quantization thresholds, so whole-convolution outputs require a numerical contract and tolerances. |
| Finite differences validate QAT | Rounding is discontinuous and the straight-through estimator is an intentionally chosen surrogate derivative. Use finite differences for the smooth FP32 model and separate checks for the surrogate backward rule. |
| A uniform CTA branch is enough for tile-dependent precision | It also needs recoverable higher-precision inputs, compatible output storage, scale metadata, synchronization, and a measured dispatch benefit. Begin with host-selected policies at graph boundaries. |
| Ordinary 2:4 pruning automatically enables native sparse FP4 | The selected native FP4 sparse instruction has its own paired sparsity and metadata requirements. Treat sparsity as a separate, instruction-specific project. |

The roofline expression of approximately 48C FLOP/byte can be retained only as an ideal equal-channel NVFP4 calculation. It does not justify the original unconditional conclusion that every layer with C above about 18 is compute-bound. The detailed traffic calculation appears below.

## Scope and relationship to UFSM

The production engine remains handwritten C/CUDA with a C ABI. Use nvcc, its CUDA headers and toolchain components, a supported host compiler, CUDA runtime/driver APIs, and standard host facilities. Implement convolution, normalization, interpolation, reductions, conversion, loss, optimization, and any gradient exchange directly. Do not depend on cuDNN, cuBLAS, CUTLASS, TensorRT, Transformer Engine, Triton, or prebuilt CUB/Thrust numerical kernels.

“Using nvcc” still entails its host compiler, assembler, device linker, runtime, and driver. It does not mean the complete application can be built with no other compiler or system library. A future PyTorch binding may expose the handwritten operators for experiments, but PyTorch is optional and is not the production implementation or the ground-truth reference.

There are two meanings of “from scratch” here: all numerical kernels are ours, and the model can train from random initialization using released labels. Later QAT may continue from our own BF16 checkpoint. A separate teacher is unnecessary.

The source now has FP32, BF16/FP16, FP8/MXFP8, and MXFP4 compute paths, with mixed storage, recomputation, and per-convolution/per-pass policies. The public boundary still uses `float*` for several buffers whose actual storage can be 16-bit or registered MX bytes; ordinary logical indexing remains NCDHW, while MX storage uses channel-blocked payload/scale rows. This is not a typed NDHWC engine. The CLI trainer defaults to FP16 storage via `--f16 1`; the low-level initial mode is BF16 unless settings/environment change it. Compute policy, storage type, accumulator, and effective fallback must be reported independently. See [nn.h](src/nn.h), [nn.cu](src/nn.cu), [nn_fp8.cu](src/nn_fp8.cu), and [unet.c](src/unet.c).

Evolve the handwritten backend through an explicit execution/context boundary. Preserve a versioned model, mask, coordinate, interpolation, and checkpoint contract while correcting the identified defects. Audit entries in [FINDINGS.md](FINDINGS.md) distinguish fixed historical findings from current source defects and conditional API cases. They are source evidence, not claims that this review ran a failing program.

## Implemented work and remaining acceptance gates

The project is no longer starting with only a BF16 staging path. Reuse the existing work where its contract is correct; do not write another complete backend merely to satisfy the old phase names. The immediate objective is a trustworthy mixed model whose actual arithmetic/storage are explicit, followed by additional native formats where complete quality and performance evidence warrants them.

| Area | Present in reviewed source | Still required before acceptance |
| --- | --- | --- |
| Baselines | Direct FP32 kernels; 16-bit tensor kernels with FP32 accumulators; FP16 partial/group accumulation policy 4 | Independent full-graph numerical checks, finite-value assertions, shape/storage capability validation |
| FP8 | Block-scaled E4M3 convolution/dgrad/wgrad; MX activation and gradient storage; higher-precision pass overrides | Consistent normalization contract, small split-output correction, complete backward and deployment-policy quality |
| FP4 | Native MXFP4 forward/selected dgrad; FP8 fallback paths; fake-format experiments | Explicit fallback map, consistent grids and QAT reference, corrected packed updates/resume; native NVFP4 remains separate |
| Memory | Default recompute 1; optional recompute 2; shared inference temporaries; fused upsampling and chunked gradients | Identical replay conversions, complete allocation ownership/accounting, capability limits and peak-memory evidence |
| Model | Four-input surface graph; optional GroupNorm/SiLU after downsampling | Treat down_norm as a separate architecture and guard every FP16 normalization input |
| Optimizers | AdamW; packed grid/error-feedback updates; latest Muon plus AdamW partition | Disjoint update intervals, one decay/update owner per parameter, correct counters and complete optimizer checkpoints |
| Pipeline | Occupancy sampling, soft/trust targets, pinned batch copies, prediction read-ahead/device preprocessing; latest chunk snapping/prefetch | Holdout-safe sampling, complete-batch publication, immutable hard seeds, failure propagation, actual transfer overlap |
| Policy | Per-layer/conv/pass host selection and setters | Transactional parsing/transitions, context-owned state, immutable effective execution manifest and safe weight-cache rebuilds |
| Evidence | Primitive harnesses and author-reported synthetic/real-data runs | Assertions covering failing contracts, fresh identity-bound holdouts, sheet topology, complete latency/memory and failure handling |

Current source blockers are ordinary resume resetting loaded state; nonmonotonic packed optimizer/EMA intervals; statistics computed from a different tensor than stored normalization inputs; saturation lost during recompute 2; hidden low-precision launch errors; and sampler holdout/partial-batch defects. Latest Muon adds parameter-partition, optimizer-state, and test gaps. The intermediate prefetch bitmap hang was corrected in the final cut; other prefetch status/budget issues remain. Use the individual statuses in [FINDINGS.md](FINDINGS.md), not the earlier blanket “done/tested” project milestones.

The effective precision manifest must be generated from the selected kernels and storage, not just the requested name. In current source, `all=fp4` can execute FP8 for small-Ci/stride-2/wgrad, 16-bit fused decoder upsampling, and a float-accumulating head; registered MX inputs can force FP8 despite another requested policy. Some exceptions are intentional and useful. Each must appear in the plan and benchmark attribution. No all-FP4 network is established by that policy label.

Packed weights retain FP32 live/EMA shadows plus packed payloads/scales and, for FP4, FP8 residual state. They are a different optimizer recipe from FP32-master QAT, not proof of lower total memory. Convolution prepares execution weights again from shadows. Masked 2:4 weights still use dense MMA; this is a pruning experiment without native sparse instruction throughput. MX stride-2 dgrad currently uses scalar gathers. Measure these actual operations before assigning peak-format FLOPs to them.

Keep FP32-master AdamW with BF16/FP16 compute as the initial comparison recipe. Evaluate packed weights, Muon, down_norm, soft-target curricula, changed augmentation, and WSD scheduling as distinct experiments. Changing several simultaneously prevents attributing quality to precision. Muon and packed-weight modes are currently incompatible in the dispatcher: requesting Muon with packed weights falls back to AdamW. A supported fallback must be explicit in run provenance.

The closing source delta adds per-source `min_level` and makes the generator select level 2 or coarser for applicable fine-resolution pyramid sources. It changes physical sampling resolution and removes some possible levels; record that recipe and confirm at least one eligible level/window remains. Validation/policy comparisons must hold this setting fixed. Storage I/O counters now count only cache-backed handles, so they do not represent all process I/O.

## Source contract before further optimization

Resolve the following contracts before accepting more precision combinations:

1. **Parameters and state:** fresh initialization, exact resume, and fine-tuning are separate operations. Optimizer parameter intervals are disjoint; each weight receives one intended update and decay. An exact resume restores packed residuals, Muon momentum where used, moments, EMA/stochastic counters, policy, loss scale, and successful-update count.
2. **Normalization:** define whether GN statistics describe the stored rounded tensor or a deliberately different operator. For the baseline, use the stored tensor. Backward and replay consume that same value and saved statistics. Constant groups must normalize to the affine offset. Clamping, rounding, scale selection, and fake quantization occur at the same boundaries in original forward and replay.
3. **Typed execution:** every edge and scratch region has a format, shape, allocation size, lifetime, and context. No float-only fallback receives 16-bit/MX buffers. Validate all selected forward/backward/head/norm capabilities before allocating a graph.
4. **Failure:** launch, allocation, completion, I/O, cache publication, checkpoint close, and thread-start failures reach the caller. An error cannot become an unwritten output, an incomplete READY batch, an empty successful volume, or a passing diagnostic.
5. **Data/evaluation identity:** holdouts are enforced after all origin transforms; physical CT identity and per-channel validity are explicit; prediction caches include checkpoint, architecture, source, crop, level, axis, precision and completion identity. Quantization quality uses new predictions under the effective deployment policy.
6. **Counters and contexts:** gradient accumulation is consistently scaled until one unscale boundary; Adam bias correction follows successful moment updates. Device selection cannot reset adaptive state. Configurations, scratch, storage metadata and events have one explicit owner per model/device context.

These are implementation/validation requirements for later authorized work. This review changes documents only.

## Hardware and toolchain contract

NVIDIA lists the RTX 5060 Ti with compute capability 12.0, 16 GB or 8 GB GDDR7, 448 GB/s bandwidth, and 759 AI TOPS. The product page alone does not specify the performance of every MMA variant. [NVIDIA RTX 5060 family specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5060-family/)

The documentation checked for the original 2026-09-30 rewrite identified CUDA 13.4 Update 1 and PTX ISA 9.4. Those are candidate pinned versions, with an archived documentation set, a supported host compiler, and a compatible driver. The source-only re-review did not query installed tools/drivers or reverify release availability. Future upgrades require the capability and numerical checks. [CUDA release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/)

Initially build the specialized low-precision backend for `sm_120a` and retain an ordinary `sm_120` baseline. Architecture and family suffixes have different compatibility scopes; keep target-specific code separate and dispatch according to the device and the compiled capability manifest. Do not assume an architecture-specific binary is portable to a later GPU. [CUDA compiler target compatibility](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#feature-availability)

Record actual object-specific build flags and fast-math settings in that manifest. The current Makefile has distinct CUDA objects and target settings; its presence is not a compiler acceptance result from this review. Separate a deterministic correctness configuration from tuned arithmetic/reduction behavior.

Plan against the documented 99 KiB per-block shared-memory ceiling. Allocations above 48 KiB require dynamic shared memory and explicit opt-in. The current tuning guide and programming guide describe the SM/cache capacities differently, so the eventual implementation must query its actual device limits and occupancy rather than relying on one quoted per-SM number. [Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/), [CUDA compute-capability specifications](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)

### Supported arithmetic families

The initial instruction choices below are a capability shortlist, not compiled wrapper code. Consult the pinned PTX instruction's legal types, layouts, registers, scale selectors, and target requirements when implementing it.

| Mode | Initial arithmetic path | Accumulator |
| --- | --- | --- |
| FP32 | SIMT FP32 FMA | FP32 |
| TF32 | Warp MMA m16n8k8 | FP32 |
| FP16 | Warp MMA m16n8k16 | FP32 |
| BF16 | Warp MMA m16n8k16 | FP32 |
| FP8 | E4M3/E5M2 warp MMA m16n8k32 | FP32 |
| MXFP8 | mxf8f6f4, k32, 1X, UE8M0 | FP32 |
| NVFP4 | mxf4nvf4, k64, 4X, UE4M3 | FP32 |
| MXFP4 | mxf4, k64, 2X, UE8M0 | FP32 |
| W4A8 experiment | Unscaled f8f6f4 mixed E2M1/E4M3, k32 | FP32 |

Use warp-level `mma.sync` on this target, not TMEM/tcgen05. W4A8 is not the native block-scaled NVFP4 pair; its scale treatment needs separate validation. Arbitrary FP4/FP16/FP32 operand pairs are not promised. Unsupported pairs convert explicitly to a supported compute pair. [PTX MMA specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma)

FP16 accumulation is an optional later experiment where the instruction supports it. BF16 and the selected block-scaled paths retain FP32 accumulation. A policy that requests an unavailable accumulator must fail validation rather than silently changing arithmetic.

### Throughput planning estimates

NVIDIA's RTX Blackwell whitepaper distinguishes FP8 and FP16 throughput by accumulator type, with lower FP32-accumulation rates, and distinguishes dense from effective sparse rates. These distinctions should already influence planning. They do not need to wait for a benchmark to be acknowledged. [RTX Blackwell architecture whitepaper, specification tables](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)

The following are **architectural estimates**, obtained by applying those published GeForce ratios to the rounded 759 AI TOPS figure. They are not separately published RTX 5060 Ti instruction measurements and are not expected kernel throughput.

| Arithmetic | Approximate dense planning ceiling in TFLOP/s |
| --- | ---: |
| FP4 with FP32 accumulation | 379.5 |
| Unscaled FP8 with FP16 accumulation | 189.8 |
| Unscaled FP8 with FP32 accumulation | 94.9 |
| FP16 with FP16 accumulation | 94.9 |
| FP16 or BF16 with FP32 accumulation | 47.4 |
| TF32 with FP32 accumulation | 23.7 |
| SIMT FP32 | About 23.7 from CUDA cores and listed boost clock |

Do not assign the unscaled FP8 rate to MXFP8, or claim that all E2M1 instruction families reach the native FP4 ceiling. The capability phase must measure each selected family and packing path. All sparse numbers require the correct structured representation and metadata; count dense useful FLOPs separately from sparse effective FLOPs.

## Model size and architecture choices

Retain a named version of UFSM's four-level graph: widths `(16, 32, 64, 80)`, two 3×3×3 convolution–GroupNorm–SiLU stages per encoder/decoder block, 3×3×3 stride-2 convolutions between encoder levels, trilinear 2× upsampling, logical skip concatenation, and a 1×1×1 head. The source uses four input channels, comprising CT and radial direction, and two training output slots. The new `down_norm` adds GroupNorm/SiLU after each down convolution; compare it as a separate model version rather than folding it into a precision experiment. [Model construction](src/unet.c), [model configuration](src/unet.h)

The parameter count is architecture-specific. For the construction in `src/unet.c`, a block mapping `Ci` to `Co` contributes `27*Ci*Co + 27*Co*Co + 6*Co` parameters, including biases and two affine GroupNorms. Each downsampling convolution contributes `27*C*C + C`; the head contributes `C0*Cout + Cout`.

For four levels with widths `(b, 2b, 4b, 8b)`, four input channels and two output channels, this gives:

`parameters = 6264*b*b + 251*b + 2`

| Widths | Parameters derived from that exact graph | Intended comparison |
| --- | ---: | --- |
| 1, 2, 4, 8 | 6,517 | Very small capacity baseline |
| 2, 4, 8, 16 | 25,560 | Small capacity baseline |
| 4, 8, 16, 32 | 101,230 | Approximately 100k candidate |
| 8, 16, 32, 64 | 402,906 | Intermediate candidate |
| 16, 32, 64, 128 | 1,607,602 | Wider bottleneck candidate |
| 16, 32, 64, 80 | 1,171,826 | Current graph with two output slots |

These counts are algebraic source analysis, not results from running the model. Change the input count, head count, downsampling, normalization, or decoder and the counts change. Tiny widths also require valid normalization group counts. No row carries a predicted quality guarantee.

For `down_norm`, add `2*sum(widths[0:nlev-1])` affine parameters. The `(16,32,64,80)` two-output graph therefore has **1,172,050** parameters with down_norm, versus 1,171,826 without it. The geometric `(b,2b,4b,8b)` formula becomes `6264*b*b + 265*b + 2`. These are source-derived counts. Actual accepted widths must also satisfy group divisibility, head/recompute capacity, MX normalization limits, and storage/layout constraints; parameter count alone does not establish a runnable architecture.

A two-parameter intensity threshold is a useful control experiment, but its ability to represent complex sheet separation is an empirical question. There is no defensible “one parameter is enough” conclusion from the supplied material.

Keep three separate comparisons: model capacity, arithmetic precision on the same graph, and architectural changes such as normalization removal or learned upsampling. Maintain held-out splits while searching. Prefer an accuracy–latency–memory Pareto set, meaning candidates that are not worse on every objective, over one alleged universally optimal parameter count.

Convolutional context can be calculated using `r_next = r + (k-1)*j` and `j_next = j*stride`. The current encoder's longest convolutional path reaches nominal receptive-field width 75 at the bottleneck before considering normalization. GroupNorm and patch-wide CT normalization couple distant voxels within a patch, so the complete model is not bounded by that local convolutional receptive field. Patch seams and normalization must be evaluated explicitly; a fixed convolution halo does not guarantee tile-independent predictions.

## Tensor representation and numerical rules

### Tensor metadata

Introduce a typed tensor/view descriptor rather than extending a bare `float*` with an ambiguous mode flag. It carries:

- Logical `N,D,H,W,C`, physical extents, strides in bytes, device, alignment, allocation size, and layout identifier.
- Storage encoding, nibble/byte packing order, scale encoding, block length, block axis, scale strides, and any tensor-level multiplier.
- Whether a view begins at a block boundary and whether it owns complete quantization blocks.
- Quantizer version, rounding/saturation policy, scale-selection policy, and scale-state generation.
- Buffer ownership, lifetime, aliasing permissions, readiness on a stream, and permitted access by other streams.

Use NDHWC as the initial internal candidate for low-precision convolution. Retain a canonical host/reference layout and explicit boundary conversions. Do not assume the current NCDHW interface becomes NDHWC merely because a kernel stages channels contiguously.

Physical padding is an implementation choice. Padded channels are zeroed and excluded from GroupNorm, losses, valid-voxel counts, parameter updates, and logical output stores. Affine normalization must not turn padded zeros into live channels. Compute sizes with checked wide arithmetic before allocation.

### Storage costs

NVFP4 uses E2M1 values, an E4M3-compatible nonnegative byte scale for each 16-value block, and a tensor-level FP32 multiplier. MXFP4 uses a power-of-two scale per 32 values. NVFP4's different granularity/scales motivate a later native comparison against the implemented MXFP4 path; they do not prove Vesuvius accuracy or justify discarding existing work. [NVIDIA NVFP4 format explanation](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)

Ignoring alignment, padded dimensions, scale-layout swizzles, and global scalars:

| Representation | Bytes per stored value | One 128³ tensor with 64 channels |
| --- | ---: | ---: |
| FP32 | 4 | 512 MiB |
| BF16 or FP16 | 2 | 256 MiB |
| FP8 with a tensor scalar | 1 | 128 MiB |
| MXFP8 with one byte scale per 32 values | 1.03125 | 132 MiB |
| NVFP4 with one byte scale per 16 values | 0.5625 | 72 MiB |
| MXFP4 with one byte scale per 32 values | 0.53125 | 68 MiB |

Budget the actual physical allocation and scratch, not just this table. For the current roughly 1.17M-parameter network, an FP32 parameter copy is only about 4.7 MB. High-resolution activation tensors can therefore dominate memory even though optimizer state has several parameter copies. FP4 forward storage does not remove FP32 master weights, moments, gradients, or tensors required for backward.

### Quantizer contract

For NVFP4, define dequantization as `x_hat = g * s_block * decode_E2M1(q)`. `g` is the tensor multiplier; `s_block` is the nonnegative encoded block scale. The MMA applies the block scales. Apply `g_input*g_weight` to the accumulated dot product before bias, normalization, residual addition, or activation. Mixing this order changes the model.

For the first quantizer version, use deterministic round-to-nearest-even for payloads with finite saturation. Define scale selection separately and give the full scheme a versioned name. Use the following proposed deterministic, maximum-based NVFP4 baseline:

1. Reduce the maximum absolute finite value over logical tensor elements. For an entirely zero tensor, select `g=1`, positive unit block scales, and zero payload.
2. Otherwise select `g = max(amax/(6*448), 2^-126)`, rounded to FP32. The constants are the E2M1 and nonnegative E4M3 finite maxima; the lower bound is an explicit choice to avoid a subnormal tensor multiplier.
3. For each nonzero block, select `s = clamp(round_scale(block_amax/(6*g)), 2^-9, 448)`. The lower bound is the smallest positive E4M3 scale. Compute payloads as `round_E2M1(x/(g*s))` with finite saturation. Zero blocks use the canonical rule below.
4. Specify FP32 arithmetic and rounding at every division/product in the reference, then reproduce that schedule in converters. Count payload saturation, scale clamping, and small-value loss. Scale rounding can create clipping even when `g` covers the tensor maximum.

This is a chosen quantization recipe, not a claim that every NVFP4 framework uses this exact scale-selection algorithm. Current, calibrated, delayed, upward-rounded, or error-minimizing scales are separately versioned policies. Their frozen codes and scales must be visible to the reference and backward path.

The canonical all-zero block has zero payload and a fixed positive finite scale, so decode and division never depend on a zero scale. Tail lanes contain zero and do not affect maximum or error statistics. Preserve or canonicalize signed zero consistently. Treat nonfinite input as an explicit numerical fault at the engine level; never rely on a saturating FP4 conversion to preserve NaNs. NVIDIA's conversion API documents saturation and a non-NaN result for NaN inputs. [CUDA FP4 conversion semantics](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP4__MISC.html)

For MXFP8 and MXFP4, begin with a range-preserving power-of-two scale: choose the smallest encodable power of two at least `block_amax/max_payload`, clamp to the supported finite exponent range, and record clamping. Zero blocks use the canonical positive scale. This scale-selection policy is also versioned. For ordinary FP8, initially use a positive FP32 tensor multiplier with independently selected scales for each operand. Per-channel or per-group scaling is a later variant only where its algebra maps correctly to the reduction.

A low-precision cast between different scale blocks is a dequantize–requantize operation, not a byte copy. FP4 to BF16 conversion cannot recover values lost to FP4 quantization. Direct storage-pair conversion kernels can be optimized later; initially convert through a defined FP32 numerical value.

### Scale updates

Support three named scale policies:

1. Current scaling: reduce the tensor maximum, select `g`, then quantize. This is the initial correctness path and generally needs a pass or an existing full-tensor reduction.
2. Calibrated scaling: freeze the tensor multiplier from representative inference data while retaining the declared block-scale rule. Record calibration data identity and clipping statistics.
3. Delayed scaling: consume the previous declared observation/history and collect the current maximum for a future update. Initialize the history, handle abrupt range changes, and save it in checkpoints.

Record saturation counts, small-value loss, quantization error, scale range failures, and nonfinite counts. Maximum magnitude alone is not an accuracy estimator. A delayed scale must not be presented as equivalent to current scaling. Training loss scaling is a separate mechanism and must not be confused with quantization scaling.

## Precision policy and execution planning

Separate storage decisions from arithmetic decisions. Each layer has policies for forward, backward-data, and backward-weight; each edge has a storage/quantizer contract. Specify operand encodings, accumulator, output encoding, scale orientation, and fallback for each pass. FP32 master weights and optimizer state are model-level training requirements.

The planner validates policies against a capability registry keyed by GPU target, instruction family, legal operand pair, accumulation type, convolution geometry, tensor layout, scale scheme, and alignment/tail support. It inserts explicit conversions and plans scratch. An illegal request produces an error or an explicitly configured fallback, with the actual selected arithmetic reported to the caller.

Quantized weight caches are versioned by master-weight generation, live-versus-EMA selection, device, format, quantizer state, orientation, and packing variant. Regenerate only needed caches after an optimizer update or policy change. Do not eagerly retain every possible precision of every weight tensor, and do not reuse a cache after EMA swaps or scale-policy updates.

Execution state belongs to an engine/context on a specific device and stream, rather than process-global scratch. Declare asynchronous completion and buffer lifetimes accurately. Give callers a distinct way to check launch failures and completion failures. Event dependencies protect conversion, quantization, convolution, and cache reuse.

Plan tensor lifetimes separately for inference and training. Reuse allocations only after all consumers and backward requirements finish. Reserve workspaces during planning; keep allocation, descriptor setup, and weight repacking out of the steady-state hot path where possible.

### Initial mixed model

This is a proposed starting policy, not a final accuracy result:

| Operation | Initial compute/storage choice | Reason and promotion rule |
| --- | --- | --- |
| CT/radial preprocessing | FP32 arithmetic; FP32 or BF16 boundary buffer | Preserve coordinate and normalization semantics |
| First 4-channel convolution | BF16 operands, FP32 accumulation, or FP32 SIMT | Avoid costly forced padding; compare both |
| Internal dense convolutions | BF16 first, FP8 next, selected NVFP4 after validation | Reduce only where complete latency and surface quality improve |
| Raw convolution output for GroupNorm | Initially FP32; later BF16 if validated | Global statistics and saved backward values need a defined representation |
| GroupNorm statistics and affine parameters | FP32 | Preserve reductions and small denominators |
| SiLU/interpolation/residual arithmetic | FP32 initially, with explicitly rounded storage | Add lower precision independently when useful |
| Skip tensors | BF16 first; FP8/NVFP4 only with a declared consumer contract | Long-lived activations may offer memory savings |
| Final logits, sigmoid, loss, blend accumulators | FP32 | Preserve thresholds, probabilities, and reductions |
| Backward convolution operands | BF16 initially; selected FP8 later | FP4 backward is an independent research gate |
| Weight-gradient accumulation, clipping, AdamW, EMA | FP32 | Preserve update accuracy and resumability |

The same checkpoint's logical graph can therefore contain all four bit widths simultaneously: FP32 reductions and state, BF16 boundaries, FP8 sensitive internal paths, and FP4 selected dense paths. This does not require every operation to implement every precision pair.

## Forward convolution design

### Geometry before optimization

Define the operation as cross-correlation with explicit kernel, stride, dilation, left/right padding, logical channel counts, and output dimensions. For each axis:

`out = floor((in + pad_left + pad_right - dilation*(kernel-1) - 1)/stride) + 1`

Start with kernel sizes 1 and 3, stride 1 and 2, dilation 1, and the existing model's padding. Reject unsupported geometry. Support non-cubic test tensors and partial output tiles even if the first production patch is 128³.

Maintain a simple, independently indexed FP32 direct convolution and a CPU FP64 reference. They establish coordinate, tail, padding, and bias correctness. Add separate optimized families after that contract is stable.

### Dense tensor path

Use an implicit GEMM: flatten batch and output space into M, use output channels for N, and use kernel taps times input channels for K. Choose whether activations or weights occupy operand A according to the actual packing and instruction mapping. Define that orientation once for each kernel family.

Stage a spatial input brick including its halo, plus a bounded input-channel slice, in shared memory. Reuse it across taps and the assigned output-channel tile. Stream the corresponding weights. Begin with cooperative vector loads, explicit boundary masking, and ordinary synchronization. Add asynchronous staging only after the single-stage kernel is correct.

A useful NVFP4 layout stores blocks along channels within a voxel, and weight blocks along input channels within a tap and output channel. Reuse an input block scale whenever that same voxel/channel block is used by another tap. Each tap loads the scale of its actual shifted source voxel, not the output voxel's scale. This works directly only when the gathered K block respects the stored block boundaries.

Do not require 64 input channels for k64. For example, with 16-channel NVFP4 blocks, four taps with 16 channels can fill a k64 step, carrying four distinct block scales. With 32 channels, two taps can fill the step. Compare this gather complexity with padding and a higher-precision kernel. MXFP4's 32-value blocks need their own treatment when logical channels are below 32; a 16-channel stored block cannot simply supply one scale for a 32-value gathered block.

Compile a small initial family of tiles for narrow, medium, and wide channels. Add variants because actual model shapes need them, not to form the Cartesian product of every tile and precision. Shape dispatch includes the first layer, head, narrow layers, channel tails, and volume boundaries.

### Shared memory and registers

For an 8×8×4 output brick with a 3×3×3 stride-1 convolution, the input halo is 10×10×6. At 64 channels, one normally packed NVFP4 halo contains 21,600 bytes; two such stages require 43,200 bytes before swizzling, weights, barriers, and other scratch.

Keeping all weights for 27 taps and a 64×64 channel tile would add 62,208 bytes in the same nominal representation. Together these exceed 99 KiB, so the proposed halo cannot also keep every weight resident under that budget. Stream a smaller set of taps/channels or reduce the tile. A double-buffered BF16 halo of that size alone requires 153,600 bytes and does not fit.

A 256-voxel by 64-output-channel tile also carries 16,384 FP32 accumulator values. At 256 threads, that is an average of 64 accumulator registers per thread before addresses, fragments, and pipeline state. Shared memory and registers both affect occupancy. These arithmetic budgets justify measuring multiple tile shapes; they do not establish that this brick is optimal.

### TMA and packing

Treat TMA as an optional loader implementation. Keep payload and scale descriptors separate unless the chosen storage format has a proven combined descriptor. Verify packed-byte coordinates, scale-block coordinates, shared-memory swizzles, alignment, barrier completion, and border handling for each supported variant. A zero-filled payload must have valid associated scales. Do not import datacenter-only descriptor or multicast modes without checking target support.

Weights need a canonical layout for checkpoints and separately versioned execution layouts. Preserve the existing logical `[Cout][Cin][Kd][Kh][Kw]` indexing at the model boundary; a candidate packed forward layout is `[Cout][tap][Cin_padded]` with a separately indexed scale array. An “offline” packing step applies to frozen inference weights; training needs packing after updates. Favor stable tile layouts where possible so tile changes do not require another full weight copy.

For the optional unscaled W4A8 experiment, start with one activation multiplier and one weight multiplier per output channel, so the multiplier can be restored after the K reduction. Native NVFP4 block scales varying along K cannot simply be restored by one epilogue factor. Supporting those scales requires separately rescaled partial sums or conversion to another supported operand format. Include the unscaled instruction's register packing and any nibble expansion in the cost; packed FP4 storage does not imply native FP4 arithmetic throughput.

### Epilogues

Fuse operations that depend only on the finished output tile: restoring tensor multipliers, adding bias, a local activation where the graph allows it, and quantizing complete output blocks when their tensor multiplier is already known. Residual inputs require explicit dequantization into the same numerical units before addition.

Assign full 16- or 32-channel output quantization blocks to one CTA, or use a separate quantization pass. Reducing over incomplete eight-channel MMA tiles would choose the wrong scale. Do not let different CTAs race to write different nibbles of the same byte.

Current-tensor global scaling prevents a general one-pass final quantized store: the multiplier is unavailable until all output maxima are reduced. Calibrated or delayed scaling can remove that dependency, with different numerical behavior. Fold the scale observation into an existing reduction where useful, but include its cost in latency.

## Normalization interpolation and concatenation

### GroupNorm

Initially implement three stages: produce the raw convolution tensor, reduce mean/variance over each sample's logical group, then apply normalization, affine transformation, SiLU, and optional output quantization. Use a stable FP32 reduction, such as a staged Welford algorithm, and save the statistics required by backward.

Convolution epilogues may write partial statistics, but final statistics still require a reduction and a dependency boundary. GroupNorm cannot generally be folded into fixed convolution weights because its statistics depend on the current input. BatchNorm folding is an option only for a separately defined architecture with fixed inference statistics.

UFSM already explores applying saved GroupNorm/SiLU during the next convolution's input staging. Preserve this as a candidate after the statistics pass. Compare recomputing normalization and quantization on overlapping halos with materializing a reusable quantized tensor once; either can win depending on reuse and conversion cost. Save enough state that backward uses the declared forward values.

### Upsampling

Keep trilinear interpolation with the existing `align_corners=false` convention and defined edge behavior. Initially interpolate in FP32 and store BF16 or FP32. A later quantized-output interpolation kernel must dequantize neighbor values before interpolation, then requantize the interpolated output using newly selected scales. Copying input codes/scales is not trilinear upsampling.

A stride-2 transposed convolution with kernel 2, padding 0, and no overlap is a separate possible model: each low-resolution voxel generates eight output positions. The simple GEMM-and-scatter description applies to that geometry only; overlapping transposed convolutions require accumulation. Do not substitute it into the baseline silently.

### Concatenation and skips

Begin with explicit concatenation or a consumer supporting two logical inputs. Writing producers directly into a future concatenation allocation is an optional optimization with ownership and lifetime constraints; a retained encoder skip must remain usable until its decoder consumer.

For packed concatenation, channel boundaries, tensor multipliers, and scale blocks must be compatible. Different source multipliers cannot be represented by one destination multiplier through a byte copy. Reblock/requantize where needed, or let the consuming convolution load two independently scaled operands. Padded gaps must stay outside the logical convolution and normalization dimensions.

Two-convolution fusion is a later targeted experiment. Neighboring intermediate outputs, larger effective halos, normalization dependencies, registers, and backward saves limit fusion. A whole UNet is not automatically a suitable persistent kernel. Start with local fusions and graph launch reuse.

## Backward computation and training

### Backward-data

Derive input gradients from the exact forward index relation. For each input coordinate and tap, accept an output-gradient contribution only if the stride divisibility and output bounds hold. A gather formulation avoids floating-point atomic writes and gives each input-gradient element one owner.

At stride 1, reuse appropriate tensor-MMA machinery with the correct weight orientation and padding. At stride 2, first use the direct gather baseline, then compare parity/phase-based tensor tiles. Materialized zero insertion is allowed as a correctness implementation, with its full allocation and traffic included; it is not the assumed fast path.

Distinguish the mathematical transpose from the quantization transpose. Forward blocks along input channels do not become valid backward blocks along output channels just by swapping metadata. In QAT, the baseline surrogate gradient uses the dequantized weights actually used by forward, transposed into the backward layout. A newly quantized backward orientation introduces additional gradient approximation and must be evaluated as such.

### Backward-weight and bias

Compute each kernel tap's weight gradient as a reduction over batch and output voxels, using the correct strided input coordinates. This has different M/N/K roles from forward. Quantization over the reduction dimension needs scales for those gathered spatial/batch sequences, not reused forward channel scales.

Start with BF16 operands and FP32 partial sums. Use split reduction where necessary, followed by a specified FP32 reduction of partials. Keep the order deterministic in the correctness mode. Avoid a full-volume im2col allocation. Bias gradients are separately reduced in FP32.

Specify SET versus ACCUMULATE semantics for every gradient destination. Gradient accumulation over microbatches, skip connections, branches, and devices must initialize buffers exactly once and add all required contributions. Derive checked workspace sizes from the complete shape and reduction plan.

### Smooth baseline training

Implement and verify FP32 forward/backward first, then BF16 operands with FP32 accumulators. Retain FP32 master parameters, accumulated parameter gradients, AdamW moments, affine-normalization parameters, EMA, clipping reductions, and loss calculations. Saved convolution tensors may move to BF16 only under a declared rounding contract.

Use a stable BCE-with-logits formulation and soft Dice reductions in FP32, with explicit rules for empty supervision, ignored voxels, and channel-specific labels. Keep validation loss separate from the gradient-writing path. Invalid masks, targets, and active-channel denominators must not silently change when precision changes.

FP16 training gets explicit loss scaling, unscale-before-clipping, nonfinite detection, and skip-update rules. BF16 does not require FP16's limited exponent-range remedy, but still requires checks for nonfinite values. Specify exactly when an unsuccessful optimizer step advances the step counter, EMA, caches, and quantizer histories.

### FP8 training

Add FP8 forward first while retaining BF16 backward. Then independently enable FP8 backward-data and backward-weight where the complete training comparison passes. E4M3 is an initial forward candidate; E5M2 is an initial gradient candidate because their range/precision tradeoffs differ, not an unconditional rule that every gradient should use E5M2.

Give forward activations, backward activations, weights, and output gradients independent scale state and orientation. Apply operand tensor multipliers to accumulated results at the correct point. Monitor gradient direction, norm, update differences, loss trajectory, and held-out quality against the same BF16 recipe. Do not interpret a forward-output tolerance as training equivalence.

### FP4 QAT

First validate native MXFP4 forward with higher-precision backward and FP32 optimizer state, using the existing kernels and their actual fallback map. Then implement native NVFP4 forward as a separate format comparison using the contract below. These are mixed-precision QAT recipes, not fully FP4 training. An FP16 fake-NVFP4 experiment does not establish native NVFP4 arithmetic or packed storage behavior.

Specify the straight-through estimator: initially stop gradients through scale selection and use an identity derivative through finite quantize/dequantize values. This chosen derivative is biased and does not equal the derivative of rounding. A clipped or learned-scale surrogate is a separately named experiment. Apply the chosen rule to both activation and master-weight quantizers.

The backward convolution consumes the dequantized forward operands according to the saved/recomputed forward contract. Save or reproducibly regenerate everything required for GroupNorm, SiLU, interpolation, and quantization boundaries. If a value was rounded before normalization, reconstruct that same rounded value rather than substituting an unrounded master value during backward.

Support a slow fake-quant path built from our converters and higher-precision convolution. Use it to check quantization placement and surrogate gradients. Compare native FP4 forward kernels with the same scales, codes, policy, and block ownership. Small-channel tap grouping and backward orientation can create additional requantization; represent it in the reference instead of assuming a stored grid makes every forward conversion exact. Native tensor accumulation may differ numerically from the reference, so require the declared tolerance rather than universal byte-identical convolution outputs.

QAT can either start from an internally trained BF16 checkpoint or run from random initialization with a specified precision schedule. Compare these as separate training recipes. Quantization warmup, scaling-history warmup, and promotion rules must be reproducible and checkpointed. No teacher/distillation dependency is introduced.

Keep packed-grid optimization separate from FP32-master QAT. The current packed-grid recipe uses stochastic rounding and FP4 error-feedback residuals; residuals are optimizer state, not disposable packing scratch. Correct parameter interval ownership, shadow/cache coherence, residual persistence, and mode transitions before comparing it to the baseline. A combined Muon/packed recipe requires its own implementation; silently executing AdamW after requesting Muon is not that recipe.

### Optimizer and schedule experiments

The latest Muon implementation views each 3³ convolution as `[Cout, Cin*27]`, applies Nesterov momentum and five Newton–Schulz iterations, and intends AdamW for biases, GN affine parameters and the head. Give those two optimizers disjoint parameter lists. Zero gradients do not exclude a parameter from AdamW decay or from updates driven by existing moments. Save optimizer identity, momentum and settings; rebuilding numerical scratch is sufficient, but rebuilding zero momentum is not exact continuation.

The present Muon matrix products are scalar CUDA dot products, not tensor-core GEMMs. Include their work and scratch in full-step comparisons before claiming a faster optimizer. Its standalone test uses zero parameters/momentum, beta zero and decay zero, and does not exercise model partitioning or checkpoint lifecycle. Validate multi-step updates, nonzero decay/momentum, optimizer transitions, finite-value rejection, and shape/rank edge cases separately.

The latest captured batched variant groups the stage launches across convolutions, reducing numerical-kernel launches to 18 for the graph plus a memset. Its scalar product count is unchanged, and it retains all matrices' scratch simultaneously. Validate batched versus unbatched updates and own their workspaces independently: toggling the current per-call batching setting can free the pool still referenced by cached descriptors. Fixed-mode speed measurements cannot establish transition safety.

Keep cosine and WSD (warmup, stable, cooldown) as named, checkpointed schedules with validated endpoints. Define warmup/stable/cooldown durations and behavior when extending a run. `cooldown=0` needs an explicit no-cooldown rule rather than a zero denominator at the final step. A zero base learning rate must not become a `0/0` multiplier for the Muon rate. Record schedule attempts and successful optimizer updates independently.

### Optional FP4 backward research

Only after BF16 and FP8 backward are trustworthy, consider NVFP4 backward operands, stochastic rounding, weight scales compatible with transposition, and transformations of backward-weight inputs. A transform requires matching algebra on both operands or an inverse where appropriate; adding a Hadamard transform to a tensor arbitrarily changes the network.

NVIDIA's published NVFP4 training recipes include ingredients such as 2D weight scaling, stochastic rounding, and transformations of backward-weight inputs. Those results concern their tested model/hardware recipes, not this Conv3d task or this GeForce implementation. Use them as research leads, not promised convergence or speed. [NVIDIA NVFP4 training recipe](https://developer.nvidia.com/blog/train-models-faster-with-jax-and-maxtext-using-nvfp4-on-nvidia-blackwell/)

## Dynamic precision in one model

### Static layer policies

This is the first production mode. Train and export a graph whose forward/backward and edge formats are explicit. Use FP32 or BF16 where needed, FP8 for intermediate candidates, and FP4 for selected dense layers. Conversion and scale-update nodes are visible in the execution plan and performance report.

Measure candidate layers in isolation as a useful sensitivity screen, then evaluate complete policy combinations. Errors can interact through skip connections and normalization; a greedy search is not sufficient evidence. Fine-tune with the final policy and report the held-out outcome, including difficult compressed regions and patch boundaries.

### Runtime policies

Support selecting from a small set of prevalidated complete policies at inference-request, patch, or training-step boundaries. Compile ahead of time, prepare needed weights, validate layouts and memory, and finish dependencies before switching.

Initially retain canonical BF16/FP32 activations at boundaries where runtime fallback needs the original values. Different compute paths may produce a common output storage format. This makes switching simpler but reduces the storage benefit; include that cost honestly. A graph that stores an activation only in FP4 must promote its producer and recompute, or retain a higher-precision copy, to recover a higher-precision fallback input.

Training switches use an explicit schedule or a prevalidated rule. Changing quantizer placement changes the effective forward operator and surrogate gradient; the switch is part of the training recipe. Save policy history and scale states for resume.

### Tile policies

Defer arbitrary per-tile FP4/FP8 storage. A uniform branch alone does not solve format metadata, neighboring halo gathers, shared normalization, output-block ownership, or gradient reproduction.

The first possible tile experiment keeps one common input/output representation and selects only arithmetic precision within the CTA. It has a defined fallback source and one output quantizer. Measure the extra quantization, error estimation, register use, and code size before claiming a benefit. A maximum-magnitude threshold alone cannot promise better sheet topology.

Mixed storage by tile needs typed tile metadata, gathers capable of decoding all neighboring tile formats, stable scale ownership, checkpoint/export definitions, and backward replay. Treat that as a separate extension after ordinary runtime policies succeed.

## Performance model and measurement plan

### Corrected roofline

For one stride-1 3×3×3 convolution with `V` output voxels and equal logical channel counts `C`, useful dense work is `F = 54*V*C*C` FLOPs, counting an FMA as two. With ideal one-time NVFP4 input and output traffic, ignoring weights, `B = 2*V*C*0.5625` bytes and `F/B = 48*C` FLOP/byte.

Using the provisional dense FP4 ceiling of 379.5 TFLOP/s and 448 GB/s gives an ideal ridge near 847 FLOP/byte. Thus the ideal equal-channel formula crosses it near C=18. This is an optimistic model, not proof that an actual C=32 convolution is compute-bound.

For the proposed 8×8×4 brick, the halo/output voxel ratio is `600/256 = 2.34375`. If each CTA handles all output channels and the halo is not effectively reused through cache, nominal input-plus-output intensity falls to approximately `28.7*C`. If two separate output-channel CTAs each reload that halo, it falls further to approximately `16.9*C`. Weights, output format, extra reductions, input conversions, padded work, and cache reuse change these numbers again.

In particular, if a nominal FP4 convolution must store its raw result in FP32 for normalization, the output term is 4 bytes/value rather than 0.5625. The ideal equal-channel intensity becomes approximately `11.84*C` even before the normalization passes. That is why complete graph traffic matters more than a standalone packed-convolution claim.

One 64→64 convolution at 128³ has 463,856,467,968 useful FLOPs, approximately 0.464 TFLOP. Dividing by the provisional 379.5 TFLOP/s peak gives an arithmetic-only lower bound of about 1.22 ms. The supplied 2–3 ms prediction is unverified. Do not use it as a milestone or latency promise.

### Future measurements

Once execution is authorized for implementation, collect:

- Instruction throughput for each operand/accumulator tuple and scale family, plus conversion throughput and achieved memory bandwidth.
- Per-layer forward, backward-data, backward-weight, normalization, interpolation, quantization, and optimizer times at the actual shapes.
- Complete forward latency, full training-step latency, peak allocated memory, and useful output voxels per second.
- Data decoding, preprocessing, transfers, blending, and output writing, with overlap and queue occupancy visible.
- Logical FLOPs, issued padded FLOPs, scale bytes, conversion/repack traffic, launch counts, registers, spills, and shared-memory occupancy.

Use device events for asynchronous kernel intervals and a synchronized end-to-end wall-clock measurement for complete requests. Separate setup/warmup from steady state. Report single-patch latency, batched throughput, training throughput, and pipeline throughput separately. On the shared GPUs, report concurrent load, clocks, and variance; compare policies under comparable conditions.

Measure the complete `conv → GroupNorm → SiLU → quantize` path and the complete model, not only MMA time. A lower arithmetic precision stays enabled only if it improves the chosen production objective within the quality and memory budget.

### Optimization order

After each numerical gate, prioritize eliminating repeated weight packing, unnecessary conversions, allocation churn, and unused saved tensors. Then overlap staging and computation, tune shape-specific tiles, fuse local operations, and consider CUDA graphs for stable execution plans. Plans with different precision policies may require different graph instances and workspaces.

Persistence, multiple-convolution fusion, TMA, and structured sparsity are optional branches, each justified by a demonstrated bottleneck. Optimizing a sampler or transfer can matter more than doubling convolution peak when that stage limits the complete pipeline.

## Validation contract

### Reference hierarchy

Maintain four independent layers of evidence:

1. A CPU FP64 reference establishes convolution coordinates and a high-accuracy mathematical result.
2. Exact format encoders/decoders establish representable values, scale selection, packing, and conversion behavior.
3. A quantized mathematical reference dequantizes the exact operands and computes their convolution in high precision. It separates quantization error from kernel indexing and accumulation error.
4. A declared FP32/SIMT accumulation reference and numerical tolerances assess GPU arithmetic, with a separate reference for each permitted surrogate backward policy.

A high-precision sum is not an emulator of every valid tensor-core accumulation order. NVIDIA leaves aspects of MMA accumulation order and rounding unspecified. Test conversion bytes exactly; test convolution and final codes with a stated numerical/threshold contract. [PTX floating-point MMA semantics](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma)

Exhaustively enumerate FP4/FP8 encoded values and conversion boundaries where practical. The space of whole scale blocks and FP32 source tensors is not exhaustively covered merely because the low-precision payload alphabet is small. Include midpoints, neighboring FP32 values, tiny values, scale limits, saturation, signed zero, and nonfinite handling.

### Operator cases

Cover kernel sizes 1 and 3; stride 1 and 2; odd/non-cubic spatial shapes; tiny tensors; partial bricks; small and non-tile-aligned channels; different batch sizes; every volume border; concatenation boundaries; quantization block boundaries; all-zero blocks; large and tiny values; and independently scaled operands. Use impulse and distinct-coordinate inputs to detect indexing or packing mistakes.

Check gradient accumulation across branches and microbatches, GroupNorm with constant input and small groups, interpolation boundaries and its adjoint, masked loss with no valid voxels, and live/EMA weight swaps. Test context/device isolation, allocation-size checks, launch/completion errors, and checkpoint truncation or incompatible metadata.

### Numerical checks

Derive absolute/relative and norm-based tolerances from operand rounding, reduction length, and expected magnitude. Track maximum error, RMS error, bias, cosine agreement for gradients, nonfinite counts, and update differences. Reject NaNs explicitly before computing an error summary; a maximum-error loop must not accidentally pass NaNs.

For fused quantized output, distinguish a kernel defect from a quantization threshold crossing caused by valid accumulation error. Away from thresholds require exact codes under a known reference schedule; near thresholds permit only codes consistent with the accumulator error interval and separately track their rate. Keep a deterministic debug path if exact packing replay is needed.

Use finite differences on small smooth FP32 cases for convolution, normalization, interpolation, and loss derivatives. Check FP8/FP4 STE against the declared surrogate gradient using the same quantized operands and stopped scale derivatives. Do not finite-difference the rounded forward function and claim that validates an identity STE.

### Task quality

Use trusted held-out scrolls/regions with a fixed coordinate system, model identity, and valid-label masks. Report the relevant official task score where its protocol is available, along with surface distance, recall/precision, Dice, connected-sheet merges/splits, and patch-boundary errors. Mesh or volume labels may not uniquely specify sheet identity everywhere; restrict topology claims to regions where the labels support them and define any proxy explicitly.

Do not tune on the final test split. Keep training labels, QAT calibration, policy search, and final held-out evaluation separate. Compare precision policies using the same weights where applicable, and compare training recipes over multiple seeds once feasible. A generic “FP4 loses less than 1%” statement from another domain is not a task-specific accuracy budget.

Before the policy search, record the acceptable loss in task score and merge/split rate, the target latency or throughput, and the maximum memory footprint. Until those are specified, produce a Pareto comparison rather than declaring one model accepted. FP4 is optional per layer and is removed where no acceptable advantage exists.

## Multi GPU checkpoint and inference behavior

Start with one GPU. For the two-card setup, first implement correct independent contexts and FP32 gradient exchange using explicit host staging/reduction if necessary. Query peer-access support in the future implementation before selecting a device-to-device path. NVLink availability, peer copies, and communication performance are not assumed.

Match the global loss definition across devices. Unequal counts of valid voxels or active targets need the correct weighting; averaging local gradients blindly can change the objective. If Dice is defined using global batch statistics, reduce those statistics before deriving gradients. GroupNorm remains per sample and does not need cross-device statistics.

Version training checkpoints with model geometry, logical channels, normalization groups, canonical FP32 parameters, moments, EMA, optimizer step, seed/RNG state, precision policy, quantizer version, histories, loss-scaling state, and relevant data/split identity. Validate lengths, ranges, checksums, and compatibility before allocating or loading. Quantized execution caches are disposable and rebuilt, or exported with a complete packing/layout manifest for frozen inference.

Inference uses a separate memory plan without backward tensors. Calibrate using representative CT and radial inputs, including compressed regions and each supported resolution. Preserve patch preprocessing and coordinate conventions. Accumulate overlap/blend sums and weights in FP32, use the declared blend rule, and test origin/border handling. Include conversion, skip storage, and output serialization in the pipeline measurement.

## Implementation phases and exit criteria

These phases are **acceptance gates for the existing and extended implementation**. Presence in source does not satisfy an exit criterion. Preserve correct existing kernels, repair their contracts, and implement missing formats only after the earlier gate. None of these executable checks was performed during this review.

| Phase | Deliverable | Exit criterion |
| --- | --- | --- |
| P0 Contracts and source blockers | Freeze reviewed revision, architecture/data identity, effective per-pass precision, parameter ownership, failure and checkpoint semantics | Current resume/packed/norm/replay/error/sampler blockers are corrected and reviewed; rejected configurations fail before work starts |
| P1 Reproducible references | Pin toolchain/driver/flags; independent CPU geometry and conversion references; format/capability/scratch manifests | Legal tuples later compile and pass finite, tail, border, split and failure checks; artifacts identify their exact source/toolchain |
| P2 FP32 graph and training | Audit existing forward/backward, loss-only evaluation, per-channel masks, optimizer/EMA, typed contexts and checkpoints | Smooth derivative checks, branch/microbatch accumulation, exact continuation and invalid-input rejection pass |
| P3 BF16/FP16 baseline | Audit tensor passes/storage, GN/replay rounding, FP16 scale lifecycle, head/down-norm capabilities, fresh inference memory | Whole-graph and multi-step checks pass for the chosen architecture; held-out baseline and full-step/forward memory/latency are recorded |
| P4 FP8/MXFP8 graph | Audit existing scale/packing, split and 1×1 paths; independently select activation/gradient storage and pass arithmetic | Full backward and actual deployment precision meet the recorded quality budget; all effective fallback/conversion costs are reported |
| P5 MXFP4 and QAT | Validate implemented native MXFP4, fake-quant contract, higher-precision backward; correct optional packed optimizer state | Packing/reference checks and multi-step/continuation tests pass; complete policies satisfy quality and useful latency/memory gates |
| P6 Mixed runtime policies | Transactional policy parsing, validated graph-boundary switches, cache generations, export/provenance, sensitivity search | Every allowed switch preserves shapes/ownership/state; effective manifests match dispatch; complete combinations show a useful Pareto improvement |
| P7 Native NVFP4 | Implement 16-value UE4M3 blocks, tensor multipliers, native load/MMA/output conversion and matching QAT | Native/reference and deployment comparisons pass; benefit over accepted MXFP4/FP8 is demonstrated at the same model and quality budget |
| P8 Optimizer/data experiments | Independently qualify Muon, packed updates, down_norm, soft/trust recipes, augmentation and WSD | One-owner updates/decay, complete state persistence, heldout invariants and controlled quality comparisons pass |
| P9 Targeted optimization | Fix buffer-specific transfer dependencies; tune tiles/staging/conversions/scalar dgrad and Muon; consider graph/TMA/local fusion | Timelines and complete requests demonstrate improvement including initialization/decoding/persistence; numerics and failure contracts remain satisfied |
| P10 Optional research | W4A8, FP4 backward, tile arithmetic/storage policies and instruction-specific sparsity | Each extension has its own capability, correctness, training/topology and complete-performance evidence |

An accepted P5 mixed MXFP4 forward with higher-precision backward can be useful without fully FP4 backward. To call a specific recipe FP8 training, its actual backward-data and backward-weight choices must pass P4's complete training gate; forward alone does not establish that. P7 and P10 are optional extensions, not prerequisites to a trustworthy mixed FP32/16/8/4 model using existing supported formats.

For a future sparse branch, the selected native FP4 warp path uses paired 4:8 sparsity and ordered metadata rather than unconstrained scalar 2:4 pruning. Choose operand orientation and pruning rules together. [PTX sparse FP4 definition](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-sparse-mma)

## Concrete design decisions

Keep full FP32 as the correctness/debug model, and qualify the existing BF16/FP16 compute with FP32 accumulated parameter/optimizer state as the practical baseline. Validate existing FP8/MXFP8 and MXFP4 before adding native NVFP4. Begin accepted FP4 recipes with higher-precision backward. Preserve named GroupNorm/interpolation and data-target semantics during precision comparisons. Make storage, accumulator/folding, scale orientation, effective fallback, optimizer identity and checkpoint state explicit in every execution plan.

Evaluate compact model widths independently of hardware-friendly widening. Prefer fewer real FLOPs and smaller physical tensors when they win complete latency. Promote sensitive or poorly tiled layers instead of padding every layer to 64 channels. Deliver a trustworthy mixed model before pursuing whole-network persistence, arbitrary tile storage, or sparse peak throughput.

The immediate unresolved work is correctness and state coherence identified by the source audit. Then resolve real instruction/kernel rates, best layouts at actual narrow channels, FP4 boundary sensitivity, multi-step training stability, and the task-quality budget. Author-reported timings and F1 results are useful experiment leads; they do not replace these acceptance gates or establish that all documented modes are correct.
