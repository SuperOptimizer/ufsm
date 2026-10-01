# Low-precision round 2: mixed fp16 / fp8 / fp4 compute and narrow activation storage

> **Status on master:** the last section, "Round 3", supersedes parts of this report. Recompute 1 is now the default. The upsample is fused into the decoder conv and B is chunked, so default fp16/bf16 training is 0.540 GB and inference 0.228 GB at 96^3 B2. The MX-gradient error of 64% below came from a bug fixed in 907cb33 (now 33.5%). On real data, MX storage loses F1 at 6000 steps, so the defaults stay 16-bit.

Branch `lowprec-r2-live` in /home/forrest/ufsm-lp4. It is rebased on live HEAD 25d1322, conflicts are resolved, and `make test` passes.
Branch `lowprec-r2` holds the same work on c0bafb5.

All numbers are for the (16,32,64,80) U-Net at 96^3 with batch 2 on GPU 0 (RTX 5060 Ti, sm_120). Measurement method:

- **Memory** is `unet_activation_bytes`: activations plus activation gradients plus conv scratch. Parameters and Adam moments are not included. The device-level growth measured with `cudaMemGetInfo` is within 0.03 GB of it.
- **Time** is the mean of 20 steps (tests/bench_mem).
- **Error** is the whole-network parameter-gradient error against the exact fp32 kernels, using the test_unet metric.

## Headline

**Training memory falls from 1.22 GB to 0.50 GB with fp16 accuracy unchanged.** That is fp16 storage with recompute level 2 (`UFSM_F16=1 UFSM_RECOMPUTE=2`). The step takes 40 ms, against 37 ms for the old build and 34.5 ms for the fp16 build with the same tree.

| setup | train GB | train ms | inference GB | inference ms | grad error |
|---|---|---|---|---|---|
| live HEAD c0bafb5/25d1322, bf16 or fp16 | 1.22 | 37 | n/a | 11 | 1.96% / 0.25% |
| this branch, bf16 (default) | 0.873 | 34.4 | 0.397 | 10.1 | 1.97% |
| fp16 storage | 0.873 | 34.5 | 0.397 | 10.1 | 0.25% |
| fp16 + prec 4 (`all=fp16`) | 0.873 | 33.4 | 0.397 | 9.5 | 0.38% |
| fp16 + recompute 1 | 0.596 | 34.9 | 0.341 | 9.7 | 0.25% |
| fp16 + recompute 2 | 0.504 | 40.4 | 0.341 | 9.7 | 0.25% |
| fp16 + prec 4 + recompute 2 | 0.504 | 39.0 | 0.341 | 9.1 | 0.38% |
| fp8 compute, bf16 storage (`all=fp8`) | 0.873 | 29.6 | 0.397 | 8.9 | 25% |
| MX-fp8 activations | 0.603 | 28.9 | 0.234 | 6.4 | 30% |
| MX-fp8 activations + recompute 1 | 0.458 | 30.6 | 0.205 | 7.4 | 30% |
| MX-fp8 activations + recompute 2 | 0.410 | 34.2 | 0.205 | 7.4 | 30% |
| MX-fp8 activations + gradients | 0.487 | 33.2 | 0.234 | 6.4 | 64% (33.5% after the 907cb33 fix) |
| MX-fp8 activations + gradients + recompute 2 | **0.293** | 38.4 | 0.205 | 7.4 | 64% (33.5% after the fix) |

Notes on the table:

- **Gradients.** These are 0.248 GB of every 16-bit row and 0.132 GB with MX gradients.
- **Default bf16 numerics are unchanged.** The gradient error is still 1.8–2.0%. Inference logits are bit-identical between the shared-buffer inference build and the training build.
- **The fp16 inference difference is noise.** fp16 logits differ by up to 0.006 between inference and training builds. Two training-build forwards differ by the same amount: the fp16 forward is not deterministic run to run, and that predates this branch.

## What saves the memory, ranked by bytes

1. **Gradient buffer plan: −0.35 GB of 1.22 in training.** This was on by default in the earlier round.
   - The per-level A and B buffers become one pair sized for the largest level.
   - B is sized for the decoder up part rather than the whole concat.
   - gskip is aliased with gout.
   - The unused enc0 input gradient is skipped.
2. **Recompute level 1: −0.28 GB (fp16) or −0.15 GB (MX).** Enable it with `UFSM_RECOMPUTE=1`.
   - No full-resolution block outputs `s2 = silu(gn(a2))` are stored.
   - The down conv, the decoder skip segment and the head apply GN+SiLU while staging their input.
   - The blocks whose outputs get upsampled keep `s2` because they are small: 18 MB in fp16.
   - The decoder's upsampled input is rebuilt into the gradient buffer B before its forward and its weight-gradient use. B is free at both points.
   - Gradient error is unchanged in every mode.
3. **Inference buffer sharing: −0.23 GB for 16-bit inference and −0.12 GB for MX.** It is on by default for inference builds.
   - Every a1 shares one buffer.
   - The down-conv outputs and the a2 that are not skips share a second buffer.
   - The upsampled inputs share a third.
4. **MX-fp8 activation storage: −0.27 GB versus fp16, about 2× on activations.** Enable it with `UFSM_ACT_MX8=1`.
   - The format is channel-blocked e4m3 with a ue8m0 scale per 32 channels, or per 16 when C ≤ 16. That is 8.25 bits per value.
   - Staging a 32-channel chunk is a straight copy.
   - The convs that read MX tensors run in fp8.
5. **MX-fp8 activation gradients: −0.12 GB.** Enable it with `UFSM_GRAD_MX8=1`.
6. **Recompute level 2: −0.09 GB (fp16) or −0.05 GB (MX), at a cost of 4–6 ms per step.** Enable it with `UFSM_RECOMPUTE=2`.
   - No a1 is stored.
   - All blocks share one a1 buffer, and backward re-runs conv1 into it with the same kernel and precision.

## Accuracy of narrow storage

The one-step gradient-error metric is harsh on any 8-bit storage. Training on the synthetic sheet task (train_lp, 64^3, batch 2, 2000 steps) shows that MX-fp8 activations and gradients train as well as fp16:

| run | held-out loss |
|---|---|
| fp16, run 1 | 0.0821 |
| fp16, run 2 | 0.0815 |
| MX-fp8 activations + recompute 2 | 0.0810 |
| MX-fp8 activations + gradients + recompute 2 | 0.0814 |
| simulated MXFP6 e2m3 activations (fp16 compute) | 0.0827 |
| simulated NVFP4 activations (fp16 compute) | **0.0863**, clearly worse |

### fp4 and fp6 storage were measured by simulation, not built

I used `UFSM_FAKEQ`, which rounds every stored activation in place to the format right after it is produced. The simulated runs use fp16 storage and fp16 compute.

| storage format | bits/value | grad error (recompute 0 / 1) | memory vs MX-fp8 activations |
|---|---|---|---|
| MXFP8 e4m3 | 8.25 | 21% / 19% | 1.0× |
| MXFP6 e2m3 | 6.25 | 23% / 20% | 0.76× |
| MXFP6 e3m2 | 6.25 | 42% / 38% | 0.76× |
| MXFP4 e2m1 | 4.25 | 79% / 74% | 0.52× |
| NVFP4 (e2m1, e4m3 per 16, fp32 tensor scale) | 4.5 | 66% / 62% | 0.55× |

Conclusions:

- **NVFP4 activation storage costs measurable quality.** Held-out loss is 6% worse at 2000 steps, and the gradient error is 2–3× that of fp8.
- **Its memory payoff would be small.** In the 0.29 GB MX configuration, activations are about 0.16 GB. NVFP4 would save about 0.07 GB, roughly 25% of the total.
- **I did not build real NVFP4 storage, so I have no step time for it.** I recommend against it for this model.
- **MXFP6 e2m3 is the better narrow format.** It matches fp8 accuracy at 24% fewer bytes. It would be the next step if more is needed: kind::f8f6f4 takes e2m3 operands at the same 215 TF/s rate. It needs the same scope of kernel work as MX-fp8.

## Compute precision results

These come from the per-pass precision policy and prec_sweep, with fp16 storage and recompute 1.

| policy | error | step |
|---|---|---|
| `all=fp16` (prec 4: fp16 operands, fp16 group accumulation) | 0.38% | 34.1 ms |
| `all=fp16:fp16:fp8` (fp16 forward and backward-data, fp8 weight gradient) | 3.7% | 32.5 ms |
| `all=fp8` | 24.5% | 32.9 ms |
| `all=fp4` | 68% | 33.6 ms |

- **prec 4 against the fp16-storage calibration point (0.25% at 34.5 ms).** prec 4 is 3% faster at 0.38% error, which is inside the 1% bar. It is the only compute change that meets the bar.
- **Single fp8 layers cost 2–5% each.** So the greedy search under a 5% or 10% budget picks only fp16 layers: 0.29% at 34.8 ms.

## Recommended configurations

- **Accuracy first (≤ 1%):** `UFSM_F16=1 UFSM_RECOMPUTE=1 --policy all=fp16`. That is 0.596 GB, about 34 ms and 0.38%. Use `UFSM_RECOMPUTE=2` for 0.504 GB at about 39 ms.
- **Under 5% error:** the same plus `--policy all=fp16:fp16:fp8`. That is 3.7% at 32.5 ms.
- **Fastest training with lower memory:** `UFSM_ACT_MX8=1 UFSM_RECOMPUTE=1 --policy all=fp8`. That is 0.458 GB at 30.6 ms. It is faster than fp16 and trains equally on the synthetic task.
- **Minimum memory:** `UFSM_ACT_MX8=1 UFSM_GRAD_MX8=1 UFSM_RECOMPUTE=2 --policy all=fp8`. That is 0.293 GB at 38.4 ms, 4.2× less than live HEAD. It trains equally on the synthetic task.
- **Inference:** MX-fp8 activations need 0.234 GB at 6.4 ms, against 0.397 GB at 10.1 ms for bf16. Use `UFSM_RECOMPUTE=1` for 0.205 GB at 7.4 ms.

Validate any MX configuration on the real data before relying on it.

## Validation

- **`make test` passes on the rebased branch.** The default bf16 gradient error is unchanged at 1.8–2.0%. `make test` now also runs test_mx, test_rc, and test_unet with fp16 storage and recompute 2.
- **tests/test_mx** checks every MX-storage op against fp32 kernels fed the dequantized inputs. That covers GN apply and backward, upsample and its backward, the forward conv with GN input, split forward, split backward-data, the weight gradients, stride-2 forward, backward-data and its accumulate mode, and the head. Errors are 2.6–4.7%, which is fp8 rounding.
- **tests/test_rc** checks the input-recompute ops against materialized inputs at prec 1, 4 and 2. Errors are at most 2e-5.
- **compute-sanitizer** found nothing. memcheck and racecheck are clean on test_mx and test_rc. memcheck is clean on bench_mem at 32^3 for fp16 + recompute 2, MX activations + gradients + recompute 2, MX activations + recompute 1, and fake NVFP4. memcheck is also clean on test_fused: the C=4 out-of-bounds read I saw earlier is fixed by live c0a27b8.
- **Whole-network gradient error is unchanged by recompute** in every mode: bf16 1.99%, fp16 0.25%, fp8 25.2%, MX 29.7%, MX + gradients 64%.

## Environment and API

- **Environment variables:**
  - `UFSM_ACT_MX8=1` turns on MX-fp8 activation storage.
  - `UFSM_GRAD_MX8=1` turns on MX-fp8 gradient storage and requires the previous variable.
  - `UFSM_RECOMPUTE=1|2` selects the recompute level.
  - `UFSM_FAKEQ=nvfp4|mxfp4|mxfp6e2m3|mxfp6e3m2|mxfp8` runs the storage simulation study.
- **Runtime setters:** `unet_set_act_mx8`, `unet_set_grad_mx8`, `unet_set_recompute` and `unet_grad_bytes`.
- **nn API:**
  - `nn_set_storage`, `nn_storage` and `nn_storage_forget` form the per-tensor registry.
  - `nn_mx8_bytes`, `nn_f32_to_act` and `nn_fake_quant`.
  - `nn_conv3d_fwd_x` and `nn_conv3d_bwd_weight_x` apply GN+SiLU on the input in staging, including per-segment GroupNorms for split inputs.
  - `nn_up2_fwd_gn_into`.
  - Per-pass precision: `nn_set_conv_prec`, and policies such as `dec0.c1=fp16:fp16:fp8` or `all=...`.

## Files (`git diff --stat 25d1322..lowprec-r2-live`)

| file | lines changed | content |
|---|---|---|
| src/nn.cu | ~700 | per-pass precision policy, prec 4 kernels, fused stride-2 backward-data, storage registry and MX dispatch, input-recompute entry points, GN-fused upsample, fake quant |
| src/nn_fp8.cu | ~1070 | fp16 storage specializations, MX-fp8 format, MX staging and epilogues, MX elementwise kernels, MX gradient kernels, per-segment GN in staging, stride-2 GN input |
| src/nn_lp.h, src/nn.h | ~60 | declarations; `split_t` gains `gp2` |
| src/unet.c, src/unet.h | ~280 | storage modes, gradient buffer plan, recompute levels 1 and 2, inference buffer sharing, per-conv profiler, fake-quant hooks |
| tests | new and changed | new: bench_mem.c, test_mx.c, test_rc.c, prec_sweep.c. Changed: bench_lp.c, train_lp.c (policy runs) |
| Makefile | — | new targets; `make test` runs test_mx, test_rc and test_unet with recompute 2 |
| src/train.c | — | usage string |

Commits on lowprec-r2-live:

- 300398e: precision policy and prec 4
- 14bf0d8: MX-fp8 activations
- a994072: MX-fp8 gradients
- 05ac069: recompute 1
- e3f32f9: inference sharing, recompute 2 and the simulation study

## Round 3 (on master)

Same model and measurement as above: 96^3, batch 2, GPU 0. Memory is activations plus gradients plus scratch. Error is the whole-network gradient error against the exact fp32 kernels.

### Current numbers on master

| setup | train GB | grads GB | train ms | inference GB | inference ms | grad err |
|---|---|---|---|---|---|---|
| live HEAD before round 2 (c0bafb5) | 1.22 | – | 37 | – | 11 | 1.96% / 0.25% |
| default: fp16 or bf16, recompute 1, fused upsample, chunked B | **0.540** | 0.192 | 33.8 | **0.228** | 9.6 | 0.25% / 1.95% |
| fp16, recompute 2 | 0.447 | 0.192 | 41.3 | 0.228 | 9.6 | 0.25% |
| fp16, `all=fp16:fp16:fp8` | 0.540 | 0.192 | 32.1 | 0.228 | 9.1 | 2.9% |
| bf16, `all=fp8` | 0.540 | 0.192 | 31.5 | 0.228 | 9.7 | 24% |
| MX-fp8 activations (opt-in) | 0.458 | 0.248 | 30.6 | 0.205 | 7.4 | 30% |
| MX-fp8 activations + gradients | 0.342 | 0.132 | 34.6 | 0.205 | 7.4 | 33.5% |

### What changed

1. **Recompute 1 is the default (77b776a).** `UFSM_RECOMPUTE=0` restores the stored block outputs.
2. **Fused upsample in decoder conv1 (9b6f54a).**
   - The 16-bit forward and weight-gradient staging read the coarse block output and form its trilinear 2× upsample on the fly. These are the `nn_up2_fwd_into` values.
   - The coarse rows are loaded with vector loads (4 rows × 10 columns).
   - No full-resolution upsampled tensor exists any more.
   - Inference: 0.341 → 0.228 GB at the same speed (9.7 → 9.6 ms). Training: 34.9 → 33.9 ms, because the upsample kernel is gone.
   - fp8 and MX convs keep a transient copy, allocated on demand. A first fp8 version of the fused read slowed every fp8 kernel by up to 1.7×, even without upsampling, so it was not kept.
   - Env: `UFSM_FUSED_UP=0|1|2` means never, inference only, or always (default).
3. **Chunked up-part gradient (53abc70).**
   - Decoder conv1's backward-data produces the skip gradient first, then the up-part gradient in w[i]-channel chunks.
   - Each chunk goes through B and is upsample-backwarded straight into its slice of gout[i+1]. The new calls are `nn_conv3d_bwd_data_range` and `nn_up2_bwd_into`.
   - B shrinks from w[i+1] to w[i] channels. Training: 0.596 → 0.540 GB at the same step time.
   - The fused-upsample decoder conv1 always runs the 16-bit kernels, even under an fp8 or fp4 policy. Otherwise it would need the 32-channel transient again (0.653 GB). As a side effect, `dec0.c1=fp8` error drops from 6.6% to 4.3%.
   - `UFSM_CHUNK_UP=0` restores the old sizing.
4. **MX split-output bug fixed (907cb33).**
   - The fp8 weight prep wrote output rows at the real channel instead of the padded row. That broke split outputs whose first segment is not a multiple of 32 channels: dec2, 80 + 64. Padding rows also wrote one row before the buffer.
   - Only MX gradient storage was affected.
   - Error: 64% → 33.5%. A fake-quant simulation of the same storage gives 32%, and the per-tensor errors now match it.
5. **fp16 overflow guards.**
   - Fixes: dc08a7a saturates fp16 stores that feed a GroupNorm in the 16-bit conv epilogue, and d6395fa does the same in the fp8 epilogue.
   - 720c5a2 (`sat_h16`) keeps NaN: `fminf`/`fmaxf` had turned NaN into ±65504 and hidden non-finite forwards from the trainer.
   - Synthetic check: an r6 checkpoint with enc3 conv1 weights × 64, replayed on the r6 dump.
     - Without the fp8 guard, all=fp8 gave about 1e5 non-finite enc3 a1 voxels, and the NaN reached the logits.
     - With the guards, the logits matched fp16 and exact fp32.
6. **down_norm (cf65220).**
   - Optional GroupNorm + SiLU after each stride-2 down conv: `unet_cfg.down_norm`, `train --down-norm 1`, and a checkpoint header field.
   - The down-conv output is stored pre-norm with its statistics, and the next conv1 applies the transform while staging.
   - Cost: +0.4 ms per training step (+1.2%), +0.15 ms per inference, no extra activation memory.
   - tc vs fp32 gradients agree to 0.30%.

### Real-data yardstick

The setup is MANBp, `--soft 3 --noaug 1 --P 64 --B 8 --f16 1`, then `tools/eval_holdouts.py` on last.ckpt. Values are F1 at thresholds 0.5 / 0.7.

| run | 2000 steps | 6000 steps |
|---|---|---|
| fp16 | 0.215 / 0.221 and 0.218 / 0.225 (2 runs) | 0.269 / 0.282 |
| MX-fp8 activations | 0.216 / 0.221 and 0.218 / 0.217 | 0.261 / 0.276 |
| MX-fp8 activations + gradients (fixed) | 0.220 / 0.220 | 0.251 / 0.253 |
| simulated NVFP4 / MXFP6 activations | 0.218 / 0.220 and 0.215 / 0.214 | – |
| down_norm 0, seeds 0 and 1 | – | 0.253 / 0.257 and 0.225 / 0.228 |
| down_norm 1, seeds 0 and 1 | – | 0.262 / 0.257 and 0.259 / 0.255 |

- **The 2000-step runs cannot separate the modes.** All of them sit inside the fp16 noise.
- **At 6000 steps, MX storage loses F1.** Activations alone cost −0.008, and activations plus gradients cost −0.018.
- **down_norm:** mean F1 at 0.5 is 0.260 against 0.239 without it, and the seed spread is 0.003 against 0.028.
- **Activation scale under down_norm:** enc3 a1 maxima on the r6 dump are 27–61 with down_norm against 153–255 without it. enc3 GroupNorm means are 2–4 against 9–24.

### Recommendations (supersede "Recommended configurations" above)

- **Training:** keep the default fp16 + recompute 1 (0.540 GB, about 34 ms). Use `UFSM_RECOMPUTE=2` for 0.447 GB at +7 ms.
- **down_norm:** use `--down-norm 1` for new runs.
- **MX-fp8 activations:** keep them opt-in, for memory-bound cases that can accept about 3% F1. Keep MX gradients off.
- **Inference:** the default needs 0.228 GB at 9.6 ms. MX-fp8 activations need 0.205 GB at 7.4 ms.
- **Narrower formats:** NVFP4 or MXFP6 storage is not worth building now. MXFP6 would save about 0.05 GB.
