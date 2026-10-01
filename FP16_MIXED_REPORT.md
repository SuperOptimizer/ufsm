# Low-precision round 2: mixed fp16 / fp8 / fp4 compute and narrow activation storage

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
| MX-fp8 activations + gradients | 0.487 | 33.2 | 0.234 | 6.4 | 64% |
| MX-fp8 activations + gradients + recompute 2 | **0.293** | 38.4 | 0.205 | 7.4 | 64% |

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
