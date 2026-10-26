# ufsm — C23 host code (gcc), CUDA kernels (nvcc) added later.
# Default build: the fp16 / MX kernels used by training and inference. VERIFY=1 builds the bf16 and fp32 tensor-core
# instantiations as well (verification, legacy --f16 0) into build-verify/, so the two builds never invalidate each other.
VERIFY ?= 0
# GPU target. ARCH=120 (default): GeForce Blackwell, nn_* for sm_120 and the MX-fp4 / fp8 kernels (lp_*) for sm_120a with the
# block-scaled mma. ARCH=86: Ampere (RTX 3090): everything for sm_86, the lp_* kernels with their software MX paths
# (lp_common.cuh: e2m1 / e4m3 / ue8m0 conversions in integer ops, the block-scaled mma as int8 / f16 mma + per-block scaling).
# PTX=1 embeds compute_$(ARCH) PTX only (no SASS): a newer GPU JIT-compiles it, so the sm_86 code paths run (and are tested) on
# the 5080. Build dirs: build / build-verify for ARCH=120; build86 / build86p (PTX) + -verify otherwise; BDIR= overrides the stem
# (a suite clone of the sm_86 build uses BDIR=build: the python tests run build/ufsm).
ARCH ?= 120
PTX  ?= 0
ifeq ($(ARCH),120)
BDIR    ?= build
NVARCH  := -arch=sm_120
LPARCH  := -gencode arch=compute_120a,code=sm_120a
else ifeq ($(PTX),1)
BDIR    ?= build$(ARCH)p
NVARCH  := -gencode arch=compute_$(ARCH),code=compute_$(ARCH)
LPARCH  := $(NVARCH)
else
BDIR    ?= build$(ARCH)
NVARCH  := -gencode arch=compute_$(ARCH),code=sm_$(ARCH)
LPARCH  := $(NVARCH)
endif
ifeq ($(VERIFY),1)
B := $(BDIR)-verify
TYPEFLAGS := -DUFSM_ALL_TYPES
else
B := $(BDIR)
TYPEFLAGS :=
endif

# ccache (when installed) makes recompiling unchanged sources free: switching VERIFY, make clean, other worktrees.
# CCACHE= disables it.
CCACHE  ?= $(shell command -v ccache 2>/dev/null)
CC      := $(CCACHE) gcc
NVCC    ?= $(CCACHE) nvcc
CFLAGS  ?= -std=c23 -O3 -march=native -g -Wall -Wextra -Wno-unused-parameter -Wno-format-truncation -pthread
CPPFLAGS = -D_GNU_SOURCE -Isrc -Ithird_party -Ithird_party/surfcomp $(shell pkg-config --cflags libcurl libzstd blosc zlib libcrypto)
LDLIBS   = $(shell pkg-config --libs libcurl libzstd blosc zlib libcrypto) -ljpeg -lm -lpthread
CUDA    ?= /usr/local/cuda
NVFLAGS ?= -O3 $(NVARCH) -DUFSM_ARCH=$(ARCH) -use_fast_math -Xcompiler -fno-threadsafe-statics -Isrc   # fast math: +4% step time, FD tests still pass
CUDALIBS = -L$(CUDA)/lib64 -lcudart -lcublasLt

SRC  = src/json.c src/checkpoint.c src/sheet.c src/store.c src/zarr3.c src/sources.c src/sample.c src/band.c src/ct_augment.c src/scan_augment.c src/spatial_augment.c src/crush.c src/scrollaug.c src/cover.c src/zarr2.c src/tiff.c src/z3w.c src/hf.c src/ingest.c src/zipr.c src/train.c src/restore.c src/restore_cli.c src/unet.c src/split.c src/predict.c src/sweep.c src/eval.c src/fit.c src/tiffw.c src/edt.c src/votes.c src/wind.c src/fuse.c src/phase_check.c
SRC += src/surf.c src/stitch.c src/coreline.c src/coreline_check.c src/fit_ct.c src/l1pot.c
SRC += src/fblend.c
SRC += src/render.c src/imgops.c
SRC += src/repack.c
SRC += src/wload.c src/inkext.c src/inkext_net.c src/dinovol.c src/dinovol_net.c src/dino.c src/surfeval.c
SRC += src/inksplat.c
SRC += src/tfx.c
SRC += src/embz.c src/fit_emb.c src/embpca.c src/embcalib.c src/sample_emb.c
SRC += src/hiresphase.c src/fitjoin.c src/fieldsoracle.c src/packmap.c src/scanprep.c src/pairacc.c src/wbench.c
SRC += src/view.c src/reg.c src/reg_cli.c src/reg_view.c src/reg_init.c src/reg_render.c src/reg_ink.c src/srpairs.c
SRC += src/srpair.c
SRC += src/storebench.c src/phasepack.c
OBJ  = $(patsubst src/%.c,$(B)/%.o,$(SRC)) $(B)/surfcomp.o

all: $(B)/ufsm

# make test: the default binaries (used by the python CLI / sheet tests), then the whole suite on the VERIFY=1 build
test:
	$(MAKE) VERIFY=0 build/ufsm build/test_sheet build/make_pipeline_fixture build/test_http_reader build/test_sampler_safety build/test_formats
	$(MAKE) VERIFY=1 test-all

$(B)/%.o: src/%.c src/*.h third_party/volcomp.h | $(B)
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

$(B)/surfcomp.o: third_party/surfcomp/surfcomp.c third_party/surfcomp/*.h | $(B)
	$(CC) $(CFLAGS) $(CPPFLAGS) -Wno-all -Wno-extra -c $< -o $@

$(B)/ufsm: $(OBJ) $(B)/main.o $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS) $(CUDALIBS)

# The CUDA ops are split by section (src/nn_*.cu, shared declarations / device helpers / templates in src/nn_common.cuh)
# so make -j compiles them in parallel; $(B)/nn.o is their relocatable link.
NNSRC = $(wildcard src/nn_*.cu)
NNOBJ = $(patsubst src/%.cu,$(B)/%.o,$(NNSRC))
$(B)/nn_%.o: src/nn_%.cu | $(B)
	$(NVCC) $(NVFLAGS) $(TYPEFLAGS) -Xfatbin=-compress-all -MMD -MP -c $< -o $@
$(B)/nn.o: $(NNOBJ)
	ld -r -o $@ $^
# FP8 / FP4 block-scaled mma (kind::mxf8f6f4 / kind::mxf4) needs the arch-specific sm_120a target
# The low-precision kernels are split into one translation unit per kernel family (src/lp_*.cu, shared helpers in
# src/lp_common.cuh) so make -j compiles them in parallel; $(B)/nn_fp8.o is their relocatable link, so every target
# that links $(B)/nn_fp8.o is unchanged.
LPSRC = $(wildcard src/lp_*.cu)
LPOBJ = $(patsubst src/%.cu,$(B)/%.o,$(LPSRC))
$(B)/lp_%.o: src/lp_%.cu | $(B)
	$(NVCC) -O3 $(LPARCH) -DUFSM_ARCH=$(ARCH) -use_fast_math -Xcompiler -fno-threadsafe-statics $(TYPEFLAGS) -Xfatbin=-compress-all -MMD -MP -Isrc -c $< -o $@
$(B)/nn_fp8.o: $(LPOBJ)
	ld -r -o $@ $^

$(B)/unet.o: src/unet.c src/unet.h src/nn.h | $(B)
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

$(B)/prof_infer: tests/prof_infer.c $(B)/unet.o $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm

$(B)/test_nn: tests/test_nn.c $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm

$(B)/bench_conv: tests/bench_conv.c $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm

$(B)/prec_sweep: tests/prec_sweep.c $(B)/unet.o $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm

$(B)/test_rc: tests/test_rc.c $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
$(B)/test_muon: tests/test_muon.c $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
$(B)/test_muonc: tests/test_muonc.c $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
$(B)/bench_muonc: tests/bench_muonc.c $(B)/unet.o $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
$(B)/test_checkpoint: tests/test_checkpoint.c $(B)/unet.o $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
$(B)/test_optimizer_owners: tests/test_optimizer_owners.c src/unet.c src/*.h $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) tests/test_optimizer_owners.c $(B)/nn.o $(B)/nn_fp8.o -o $@ $(CUDALIBS) -lm

$(B)/test_wide_up_grad: tests/test_wide_up_grad.c src/unet.c src/*.h $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) tests/test_wide_up_grad.c $(B)/nn.o $(B)/nn_fp8.o -o $@ $(CUDALIBS) -lm
$(B)/test_gn_contract: tests/test_gn_contract.c $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
$(B)/test_stem_precision: tests/test_stem_precision.c $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
$(B)/test_wgrad_staging: tests/test_wgrad_staging.c $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
$(B)/test_infer_buffers: tests/test_infer_buffers.c $(B)/unet.o $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
$(B)/test_recompute_live: tests/test_recompute_live.c $(B)/unet.o $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
$(B)/test_share_enc_a1: tests/test_share_enc_a1.c $(B)/unet.o $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
$(B)/bench_f4conv: tests/bench_f4conv.c $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
$(B)/bench_l0misc: tests/bench_l0misc.c $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
$(B)/bench_head: tests/bench_head.c $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
$(B)/bench_head2: tests/bench_head2.c $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
$(B)/bench_gn: tests/bench_gn.c $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm

$(B)/bench_mem: tests/bench_mem.c $(B)/unet.o $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm

$(B)/test_mx: tests/test_mx.c $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
$(B)/test_mx_sw: tests/test_mx_sw.c $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
test-all: $(B)/test_mx_sw
$(B)/test_mx4: tests/test_mx4.c $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
$(B)/test_grad_mx4: tests/test_grad_mx4.c $(B)/unet.o $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
$(B)/test_l3_sched: tests/test_l3_sched.c $(B)/unet.o $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm

$(B)/test_unet: tests/test_unet.c $(B)/unet.o $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
$(B)/test_gn_frozen: tests/test_gn_frozen.c $(B)/unet.o $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
$(B)/test_l0_norm: tests/test_l0_norm.c $(B)/unet.o $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
$(B)/test_l0_slab: tests/test_l0_slab.c $(B)/unet.o $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
$(B)/test_zwin: tests/test_zwin.c $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
$(B)/test_l0_stream: tests/test_l0_stream.c $(B)/unet.o $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
$(B)/bench_l0_ring: tests/bench_l0_ring.c $(B)/unet.o $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
$(B)/test_half: tests/test_half.c $(B)/unet.o $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
$(B)/test_unwrap_targets: tests/test_unwrap_targets.c $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
$(B)/test_fit: tests/test_fit.c $(B)/fit.o $(B)/fit_emb.o $(B)/embz.o $(B)/z3w.o $(B)/fit_ct.o $(B)/l1pot.o $(B)/coreline.o $(B)/json.o $(B)/zarr3.o $(B)/store.o $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(LDLIBS) $(CUDALIBS)
$(B)/test_surf: tests/test_surf.c $(B)/surf.o $(B)/stitch.o $(B)/z3w.o $(B)/fit.o $(B)/fit_emb.o $(B)/embz.o $(B)/fit_ct.o $(B)/l1pot.o $(B)/coreline.o $(B)/tiffw.o $(B)/tiff.o $(B)/json.o $(B)/zarr3.o $(B)/store.o $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(LDLIBS) $(CUDALIBS)
$(B)/test_predict_unwrap2: tests/test_predict_unwrap2.c $(B)/unet.o $(B)/fit.o $(B)/fit_emb.o $(B)/embz.o $(B)/fit_ct.o $(B)/l1pot.o $(B)/coreline.o $(B)/json.o $(B)/zarr3.o $(B)/store.o $(B)/z3w.o $(B)/nn.o $(B)/nn_fp8.o | $(B)/ufsm
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(LDLIBS) $(CUDALIBS)
test-all: $(B)/test_predict_unwrap2
$(B)/test_tta: tests/test_tta.c $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(LDLIBS) $(CUDALIBS)
test-all: $(B)/test_tta
$(B)/test_fblend: tests/test_fblend.c $(B)/fblend.o $(B)/fit.o $(B)/fit_emb.o $(B)/embz.o $(B)/fit_ct.o $(B)/l1pot.o $(B)/coreline.o $(B)/json.o $(B)/zarr3.o $(B)/store.o $(B)/z3w.o $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(LDLIBS) $(CUDALIBS)
test-all: $(B)/test_fblend
$(B)/test_stitch: tests/test_stitch.c $(B)/stitch.o $(B)/fit.o $(B)/fit_emb.o $(B)/embz.o $(B)/fit_ct.o $(B)/l1pot.o $(B)/coreline.o $(B)/json.o $(B)/zarr3.o $(B)/store.o $(B)/z3w.o $(B)/tiffw.o $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(LDLIBS) $(CUDALIBS)

test-all: $(B)/test_stitch
$(B)/test_coreline: tests/test_coreline.c $(B)/coreline.o $(B)/z3w.o $(B)/fit.o $(B)/fit_emb.o $(B)/embz.o $(B)/fit_ct.o $(B)/l1pot.o $(B)/json.o $(B)/zarr3.o $(B)/store.o $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(LDLIBS) $(CUDALIBS)
test-all: $(B)/test_coreline
$(B)/test_pyports: tests/test_pyports.c tests/test_pyports_expected.h $(B)/coreline_check.o $(B)/coreline.o $(B)/imgops.o $(B)/tiff.o $(B)/tiffw.o $(B)/fit.o $(B)/fit_emb.o $(B)/embz.o $(B)/z3w.o $(B)/fit_ct.o $(B)/l1pot.o $(B)/json.o $(B)/zarr3.o $(B)/store.o $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) tests/test_pyports.c $(filter %.o,$^) -o $@ $(LDLIBS) $(CUDALIBS)
test-all: $(B)/test_pyports
$(B)/test_fit_w4: tests/test_fit_w4.c $(B)/fit.o $(B)/fit_emb.o $(B)/embz.o $(B)/z3w.o $(B)/fit_ct.o $(B)/l1pot.o $(B)/coreline.o $(B)/json.o $(B)/zarr3.o $(B)/store.o $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(LDLIBS) $(CUDALIBS)
test-all: $(B)/test_fit_w4
$(B)/test_repair: tests/test_repair.c $(B)/fit.o $(B)/fit_emb.o $(B)/embz.o $(B)/z3w.o $(B)/fit_ct.o $(B)/l1pot.o $(B)/coreline.o $(B)/json.o $(B)/zarr3.o $(B)/store.o $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(LDLIBS) $(CUDALIBS)
test-all: $(B)/test_repair
$(B)/test_fit_emb: tests/test_fit_emb.c $(B)/fit_emb.o $(B)/embz.o $(B)/z3w.o $(B)/fit.o $(B)/fit_ct.o $(B)/l1pot.o $(B)/coreline.o $(B)/json.o $(B)/zarr3.o $(B)/store.o $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(LDLIBS) $(CUDALIBS)
test-all: $(B)/test_fit_emb
$(B)/test_distill: tests/test_distill.c $(B)/sample_emb.o $(B)/embz.o $(B)/z3w.o $(B)/spatial_augment.o $(B)/json.o $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(LDLIBS) $(CUDALIBS)
test-all: $(B)/test_distill

$(B)/test_ink_splat: tests/test_ink_splat.c $(OBJ) $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(LDLIBS) $(CUDALIBS)
test-all: $(B)/test_ink_splat

$(B)/test_imgops: tests/test_imgops.c tests/test_imgops_expected.h $(B)/imgops.o $(B)/tiffw.o $(B)/tiff.o
	$(CC) $(CFLAGS) $(CPPFLAGS) tests/test_imgops.c $(B)/imgops.o $(B)/tiffw.o $(B)/tiff.o -o $@ $(LDLIBS)
test-all: $(B)/test_imgops
$(B)/test_head32: tests/test_head32.c $(B)/unet.o $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
$(B)/test_sslstat: tests/test_sslstat.c src/sslstat.h | $(B)
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -o $@ -lm
test-all: $(B)/test_sslstat
test-all: $(B)/test_head32
$(B)/test_dino_loss: tests/test_dino_loss.c $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) tests/test_dino_loss.c $(B)/nn.o $(B)/nn_fp8.o -o $@ $(LDLIBS) $(CUDALIBS)
test-all: $(B)/test_dino_loss
$(B)/test_attn: tests/test_attn.c $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) tests/test_attn.c $(B)/nn.o $(B)/nn_fp8.o -o $@ $(LDLIBS) $(CUDALIBS)
test-all: $(B)/test_attn
$(B)/test_attn_ray: tests/test_attn_ray.c $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) tests/test_attn_ray.c $(B)/nn.o $(B)/nn_fp8.o -o $@ $(LDLIBS) $(CUDALIBS)
test-all: $(B)/test_attn_ray
$(B)/test_attn_dfm: tests/test_attn_dfm.c $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) tests/test_attn_dfm.c $(B)/nn.o $(B)/nn_fp8.o -o $@ $(LDLIBS) $(CUDALIBS)
test-all: $(B)/test_attn_dfm
$(B)/bench_attn: tests/bench_attn.c $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) tests/bench_attn.c $(B)/nn.o $(B)/nn_fp8.o -o $@ $(LDLIBS) $(CUDALIBS)
$(B)/test_hybrid_identity: tests/test_hybrid_identity.c $(B)/unet.o $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) tests/test_hybrid_identity.c $(B)/unet.o $(B)/nn.o $(B)/nn_fp8.o -o $@ $(LDLIBS) $(CUDALIBS)
test-all: $(B)/test_hybrid_identity
$(B)/test_dino_pts: tests/test_dino_pts.c src/dino_pts.h
	$(CC) $(CFLAGS) $(CPPFLAGS) tests/test_dino_pts.c -o $@ $(LDLIBS)
test-all: $(B)/test_dino_pts
$(B)/test_dinor: tests/test_dinor.c $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) tests/test_dinor.c $(B)/nn.o $(B)/nn_fp8.o -o $@ $(LDLIBS) $(CUDALIBS)
test-all: $(B)/test_dinor
$(B)/test_tfx: tests/test_tfx.c $(B)/tfx.o $(B)/imgops.o $(B)/tiffw.o $(B)/tiff.o $(B)/surfcomp.o
	$(CC) $(CFLAGS) $(CPPFLAGS) tests/test_tfx.c $(B)/tfx.o $(B)/imgops.o $(B)/tiffw.o $(B)/tiff.o $(B)/surfcomp.o -o $@ $(LDLIBS)
test-all: $(B)/test_tfx
$(B)/test_fit_join: tests/test_fit_join.c $(B)/fitjoin.o $(B)/fit.o $(B)/fit_emb.o $(B)/embz.o $(B)/z3w.o $(B)/fit_ct.o $(B)/l1pot.o $(B)/coreline.o $(B)/json.o $(B)/zarr3.o $(B)/store.o $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(LDLIBS) $(CUDALIBS)
test-all: $(B)/test_fit_join
$(B)/test_pairacc: tests/test_pairacc.c $(B)/pairacc.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ -lm
test-all: $(B)/test_pairacc
$(B)/test_robust: tests/test_robust.c $(B)/packmap.o $(B)/z3w.o $(B)/zarr3.o $(B)/store.o $(B)/json.o $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(LDLIBS) $(CUDALIBS)
test-all: $(B)/test_robust
$(B)/test_tiffw: tests/test_tiffw.c $(B)/tiffw.o $(B)/tiff.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(LDLIBS)
$(B)/test_wind: tests/test_wind.c src/wind.h $(B)/wind.o
	$(CC) $(CFLAGS) $(CPPFLAGS) tests/test_wind.c $(B)/wind.o -o $@ -lm
$(B)/test_phase_check: tests/test_phase_check.c src/phase_check.h $(B)/phase_check.o
	$(CC) $(CFLAGS) $(CPPFLAGS) tests/test_phase_check.c $(B)/phase_check.o -o $@ -lm
$(B)/test_fuse: tests/test_fuse.c src/fuse.c src/fuse.h src/band.h | $(B)
	$(CC) $(CFLAGS) $(CPPFLAGS) tests/test_fuse.c src/fuse.c -o $@ -lm
$(B)/bench_votes: tests/bench_votes.c src/votes.h $(B)/zarr3.o $(B)/store.o $(B)/json.o
	$(CC) $(CFLAGS) $(CPPFLAGS) tests/bench_votes.c $(B)/zarr3.o $(B)/store.o $(B)/json.o -o $@ $(LDLIBS)
$(B)/test_volcomp: tests/test_volcomp.c third_party/volcomp.h
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -o $@ -lm
$(B)/test_edt: tests/test_edt.c $(B)/edt.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ -lpthread -lm
$(B)/test_infer_mem: tests/test_infer_mem.c $(B)/unet.o $(B)/split.o $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
$(B)/test_split: tests/test_split.c $(B)/unet.o $(B)/split.o $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
SAMPLE_TEST_OBJ = $(B)/edt.o $(B)/band.o $(B)/sheet.o $(B)/ct_augment.o $(B)/scan_augment.o $(B)/spatial_augment.o $(B)/crush.o $(B)/scrollaug.o $(B)/cover.o $(B)/json.o $(B)/store.o $(B)/zarr3.o $(B)/sources.o $(B)/srpair.o $(B)/zarr2.o $(B)/tiff.o $(B)/z3w.o $(B)/hf.o $(B)/zipr.o $(B)/view.o $(B)/reg.o $(B)/sample_emb.o $(B)/embz.o
$(B)/test_gpu_aug: tests/test_gpu_aug.c src/sample.c src/*.h $(SAMPLE_TEST_OBJ) $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) tests/test_gpu_aug.c $(SAMPLE_TEST_OBJ) $(B)/nn.o $(B)/nn_fp8.o -o $@ $(LDLIBS) $(CUDALIBS)
$(B)/test_votes: tests/test_votes.c src/votes.h $(B)/votes.o $(SAMPLE_TEST_OBJ)
	$(CC) $(CFLAGS) $(CPPFLAGS) tests/test_votes.c $(B)/votes.o $(SAMPLE_TEST_OBJ) -o $@ $(LDLIBS)
$(B)/test_sample_ops: tests/test_sample_ops.c src/sample.c src/*.h $(SAMPLE_TEST_OBJ)
	$(CC) $(CFLAGS) $(CPPFLAGS) tests/test_sample_ops.c $(SAMPLE_TEST_OBJ) -o $@ $(LDLIBS)
$(B)/test_aug_preset: tests/test_aug_preset.c src/sample.c src/*.h $(SAMPLE_TEST_OBJ)
	$(CC) $(CFLAGS) $(CPPFLAGS) tests/test_aug_preset.c $(SAMPLE_TEST_OBJ) -o $@ $(LDLIBS)
$(B)/test_sampler_safety: tests/test_sampler_safety.c src/sample.c src/*.h $(SAMPLE_TEST_OBJ)
	$(CC) $(CFLAGS) $(CPPFLAGS) tests/test_sampler_safety.c $(SAMPLE_TEST_OBJ) -o $@ $(LDLIBS)
$(B)/test_spatial_augment: tests/test_spatial_augment.c src/sample.c src/*.h $(SAMPLE_TEST_OBJ)
	$(CC) $(CFLAGS) $(CPPFLAGS) tests/test_spatial_augment.c $(SAMPLE_TEST_OBJ) -o $@ $(LDLIBS)
$(B)/bench_warp: tests/bench_warp.c src/sample.c src/*.h $(SAMPLE_TEST_OBJ)
	$(CC) $(CFLAGS) $(CPPFLAGS) tests/bench_warp.c $(SAMPLE_TEST_OBJ) -o $@ $(LDLIBS)
$(B)/make_wind_fixture: tests/make_wind_fixture.c $(B)/z3w.o
	$(CC) $(CFLAGS) $(CPPFLAGS) tests/make_wind_fixture.c $(B)/z3w.o -o $@ $(LDLIBS)
$(B)/test_wind_sampler: tests/test_wind_sampler.c tests/make_wind_fixture.c src/sample.c src/*.h $(SAMPLE_TEST_OBJ)
	$(CC) $(CFLAGS) $(CPPFLAGS) tests/test_wind_sampler.c $(SAMPLE_TEST_OBJ) -o $@ $(LDLIBS)
$(B)/crush_render: tests/crush_render.c src/sample.c src/*.h $(SAMPLE_TEST_OBJ) $(B)/imgops.o $(B)/tiffw.o $(B)/tfx.o $(B)/surfcomp.o
	$(CC) $(CFLAGS) $(CPPFLAGS) tests/crush_render.c $(SAMPLE_TEST_OBJ) $(B)/imgops.o $(B)/tiffw.o $(B)/tfx.o $(B)/surfcomp.o -o $@ $(LDLIBS)
$(B)/xscan_render: tests/xscan_render.c src/sample.c src/*.h $(SAMPLE_TEST_OBJ) $(B)/imgops.o $(B)/tiffw.o $(B)/tfx.o $(B)/surfcomp.o
	$(CC) $(CFLAGS) $(CPPFLAGS) tests/xscan_render.c $(SAMPLE_TEST_OBJ) $(B)/imgops.o $(B)/tiffw.o $(B)/tfx.o $(B)/surfcomp.o -o $@ $(LDLIBS)

$(B)/scrollaug_anim: tests/scrollaug_anim.c tests/gif_anim.h src/sample.c src/*.h $(SAMPLE_TEST_OBJ) $(B)/imgops.o $(B)/tiffw.o $(B)/tfx.o $(B)/surfcomp.o
	$(CC) $(CFLAGS) $(CPPFLAGS) tests/scrollaug_anim.c $(SAMPLE_TEST_OBJ) $(B)/imgops.o $(B)/tiffw.o $(B)/tfx.o $(B)/surfcomp.o -o $@ $(LDLIBS)
$(B)/aug_render: tests/aug_render.c src/sample.c src/*.h $(SAMPLE_TEST_OBJ) $(B)/imgops.o $(B)/tiffw.o $(B)/tfx.o $(B)/surfcomp.o
	$(CC) $(CFLAGS) $(CPPFLAGS) tests/aug_render.c $(SAMPLE_TEST_OBJ) $(B)/imgops.o $(B)/tiffw.o $(B)/tfx.o $(B)/surfcomp.o -o $@ $(LDLIBS)
$(B)/crush_anim: tests/crush_anim.c src/sample.c src/*.h $(SAMPLE_TEST_OBJ) $(B)/imgops.o $(B)/tiffw.o $(B)/tfx.o $(B)/surfcomp.o
	$(CC) $(CFLAGS) $(CPPFLAGS) tests/crush_anim.c $(SAMPLE_TEST_OBJ) $(B)/imgops.o $(B)/tiffw.o $(B)/tfx.o $(B)/surfcomp.o -o $@ $(LDLIBS)
$(B)/test_scrollaug: tests/test_scrollaug.c tests/make_wind_fixture.c src/sample.c src/*.h $(SAMPLE_TEST_OBJ)
	$(CC) $(CFLAGS) $(CPPFLAGS) tests/test_scrollaug.c $(SAMPLE_TEST_OBJ) -o $@ $(LDLIBS)
$(B)/test_crush: tests/test_crush.c tests/make_wind_fixture.c src/sample.c src/*.h $(SAMPLE_TEST_OBJ)
	$(CC) $(CFLAGS) $(CPPFLAGS) tests/test_crush.c $(SAMPLE_TEST_OBJ) -o $@ $(LDLIBS)

$(B)/test_raster: tests/test_raster.c src/ingest.c src/*.h $(SAMPLE_TEST_OBJ) $(B)/surfcomp.o
	$(CC) $(CFLAGS) $(CPPFLAGS) tests/test_raster.c $(SAMPLE_TEST_OBJ) $(B)/surfcomp.o -o $@ $(LDLIBS)

$(B)/bench_sampler: tests/bench_sampler.c $(B)/sample.o $(B)/sample_emb.o $(B)/embz.o $(B)/edt.o $(B)/band.o $(B)/sheet.o $(B)/ct_augment.o $(B)/scan_augment.o $(B)/spatial_augment.o $(B)/crush.o $(B)/scrollaug.o $(B)/cover.o $(B)/sources.o $(B)/srpair.o $(B)/view.o $(B)/reg.o $(B)/zarr3.o $(B)/zarr2.o $(B)/store.o $(B)/json.o $(B)/z3w.o $(B)/tiff.o $(B)/zipr.o $(B)/hf.o $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm -lcurl -lzstd -lblosc -lz -lcrypto
$(B)/test_gpu_input: tests/test_gpu_input.c $(B)/sample.o $(B)/sample_emb.o $(B)/embz.o $(B)/edt.o $(B)/band.o $(B)/sheet.o $(B)/ct_augment.o $(B)/scan_augment.o $(B)/spatial_augment.o $(B)/crush.o $(B)/scrollaug.o $(B)/cover.o $(B)/sources.o $(B)/srpair.o $(B)/view.o $(B)/reg.o $(B)/zarr3.o $(B)/zarr2.o $(B)/store.o $(B)/json.o $(B)/z3w.o $(B)/tiff.o $(B)/zipr.o $(B)/hf.o $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm -lcurl -lzstd -lblosc -lz -lcrypto
$(B)/test_pack_tm: tests/test_pack_tm.c $(B)/sample.o $(B)/sample_emb.o $(B)/embz.o $(B)/edt.o $(B)/band.o $(B)/sheet.o $(B)/ct_augment.o $(B)/scan_augment.o $(B)/spatial_augment.o $(B)/crush.o $(B)/scrollaug.o $(B)/cover.o $(B)/sources.o $(B)/srpair.o $(B)/view.o $(B)/reg.o $(B)/zarr3.o $(B)/zarr2.o $(B)/store.o $(B)/json.o $(B)/z3w.o $(B)/tiff.o $(B)/zipr.o $(B)/hf.o $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm -lcurl -lzstd -lblosc -lz -lcrypto
$(B)/test_gyrec: tests/test_gyrec.c $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
$(B)/test_pgrad: tests/test_pgrad.c $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
$(B)/test_wind4: tests/test_wind4.c $(B)/band.o $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
$(B)/test_unwrap2_sampler: tests/test_unwrap2_sampler.c $(B)/sample.o $(B)/sample_emb.o $(B)/embz.o $(B)/edt.o $(B)/band.o $(B)/sheet.o $(B)/ct_augment.o $(B)/scan_augment.o $(B)/spatial_augment.o $(B)/crush.o $(B)/scrollaug.o $(B)/cover.o $(B)/sources.o $(B)/srpair.o $(B)/view.o $(B)/reg.o $(B)/zarr3.o $(B)/zarr2.o $(B)/store.o $(B)/json.o $(B)/z3w.o $(B)/tiff.o $(B)/zipr.o $(B)/hf.o $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm -lcurl -lzstd -lblosc -lz -lcrypto
$(B)/test_phase_interp: tests/test_phase_interp.c $(B)/sample.o $(B)/sample_emb.o $(B)/embz.o $(B)/edt.o $(B)/band.o $(B)/sheet.o $(B)/ct_augment.o $(B)/scan_augment.o $(B)/spatial_augment.o $(B)/crush.o $(B)/scrollaug.o $(B)/cover.o $(B)/sources.o $(B)/srpair.o $(B)/view.o $(B)/reg.o $(B)/zarr3.o $(B)/zarr2.o $(B)/store.o $(B)/json.o $(B)/z3w.o $(B)/tiff.o $(B)/zipr.o $(B)/hf.o $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm -lcurl -lzstd -lblosc -lz -lcrypto
$(B)/bench_read: tests/bench_read.c $(B)/sample.o $(B)/sample_emb.o $(B)/embz.o $(B)/edt.o $(B)/band.o $(B)/sheet.o $(B)/ct_augment.o $(B)/scan_augment.o $(B)/spatial_augment.o $(B)/crush.o $(B)/scrollaug.o $(B)/cover.o $(B)/sources.o $(B)/srpair.o $(B)/view.o $(B)/reg.o $(B)/zarr3.o $(B)/zarr2.o $(B)/store.o $(B)/json.o $(B)/z3w.o $(B)/tiff.o $(B)/zipr.o $(B)/hf.o $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm -lcurl -lzstd -lblosc -lz -lcrypto
$(B)/fwd_nan: tests/fwd_nan.c $(B)/unet.o $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
$(B)/test_fused: tests/test_fused.c $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
$(B)/bench_lp: tests/bench_lp.c $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
$(B)/train_lp: tests/train_lp.c $(B)/unet.o $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm

$(B)/test_formats: tests/test_formats.c $(OBJ) $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(LDLIBS) $(CUDALIBS)
$(B)/test_typed_zarr: tests/test_typed_zarr.c $(B)/z3w.o $(B)/zarr3.o $(B)/store.o $(B)/json.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(LDLIBS)

$(B)/test_ct_augment: tests/test_ct_augment.c $(B)/ct_augment.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ -lm

$(B)/test_side: tests/test_side.c $(B)/band.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ -lm

$(B)/test_scan_augment: tests/test_scan_augment.c src/scan_augment.c src/scan_augment.h $(B)/ct_augment.o
	$(CC) $(CFLAGS) $(CPPFLAGS) tests/test_scan_augment.c $(B)/ct_augment.o -o $@ -lm

$(B)/aug_preview: tools/aug_preview.c $(SAMPLE_TEST_OBJ)
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(LDLIBS)

$(B)/test_target_erode: tests/test_target_erode.c src/target_erode.h | $(B)
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -o $@

test-all: $(B)/test_target_erode

$(B)/test_papyrus: tests/test_papyrus.c src/papyrus.h | $(B)
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -o $@ -lm

test-all: $(B)/test_papyrus

$(B)/test_cover: tests/test_cover.c $(B)/cover.o $(B)/json.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ -lcrypto -lm

$(B)/test_sheet: tests/test_sheet.c $(B)/sheet.o $(B)/json.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ -lcrypto -lm

$(B)/test_sheet_gpu: tests/test_sheet_gpu.c $(B)/split.o $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm

$(B)/test_wgrad_grid: tests/test_wgrad_grid.c $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm

test-sheet: $(B)/test_sheet $(B)/test_spatial_augment
	./$(B)/test_spatial_augment

test-sheet-gpu: $(B)/ufsm $(B)/test_sheet_gpu $(B)/test_wgrad_grid $(B)/make_pipeline_fixture
	./$(B)/test_wgrad_grid
	./$(B)/test_sheet_gpu

$(B)/test_json: tests/test_json.c $(B)/json.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(LDLIBS)

# store-bench without the CUDA build (laptop / desk / pods): HTTP fetch throughput, see src/storebench.c
$(B)/store_bench: src/storebench.c $(B)/store.o $(B)/zarr3.o $(B)/json.o
	$(CC) $(CFLAGS) $(CPPFLAGS) -DSTORE_BENCH_MAIN $^ -o $@ $(LDLIBS)


# phase-pack without the CUDA build (desk / pods): compact phase-label transfer, see src/phasepack.c
$(B)/phase_pack: src/phasepack.c $(B)/store.o $(B)/zarr3.o $(B)/json.o $(B)/z3w.o
	$(CC) $(CFLAGS) $(CPPFLAGS) -DPHASE_PACK_MAIN $^ -o $@ $(LDLIBS)

$(B)/test_phasepack: tests/test_phasepack.c $(B)/phasepack.o $(B)/store.o $(B)/zarr3.o $(B)/json.o $(B)/z3w.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(LDLIBS)
$(B)/test_http_reader: tests/test_http_reader.c $(B)/store.o $(B)/zarr3.o $(B)/json.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(LDLIBS)

$(B)/test_checkpoint_runtime: tests/test_checkpoint_runtime.c $(B)/checkpoint.o $(B)/json.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ -lm

$(B)/make_pipeline_fixture: tests/make_pipeline_fixture.c $(B)/z3w.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(LDLIBS)

$(B)/test_eval: tests/test_eval.c src/eval.c $(SAMPLE_TEST_OBJ)
	$(CC) $(CFLAGS) $(CPPFLAGS) tests/test_eval.c $(SAMPLE_TEST_OBJ) -o $@ $(LDLIBS)

test-all: $(B)/test_wgrad_grid

$(B):
	mkdir -p $(B)

test-all: $(B)/test_gpu_aug $(B)/test_sheet $(B)/test_sheet_gpu $(B)/test_cover $(B)/test_ct_augment $(B)/test_scan_augment $(B)/test_side $(B)/test_votes $(B)/test_wind $(B)/test_phase_check $(B)/test_fuse $(B)/test_raster $(B)/test_http_reader $(B)/test_wide_up_grad $(B)/test_wgrad_staging $(B)/test_stem_precision $(B)/test_gn_contract $(B)/test_sampler_safety $(B)/test_optimizer_owners $(B)/test_infer_buffers $(B)/test_recompute_live $(B)/test_share_enc_a1 $(B)/test_checkpoint_runtime $(B)/make_pipeline_fixture $(B)/test_checkpoint $(B)/test_eval $(B)/test_sample_ops $(B)/test_json $(B)/test_nn $(B)/test_unet $(B)/test_fused $(B)/test_formats $(B)/ufsm $(B)/test_mx $(B)/test_mx4 $(B)/test_grad_mx4 $(B)/test_rc $(B)/test_split $(B)/test_muon $(B)/test_muonc $(B)/test_gn_frozen $(B)/test_infer_mem $(B)/test_half $(B)/test_unwrap_targets $(B)/test_fit $(B)/test_surf $(B)/test_tiffw $(B)/test_edt $(B)/test_volcomp $(B)/test_typed_zarr $(B)/test_aug_preset $(B)/test_l0_norm $(B)/test_l0_slab $(B)/test_wind_sampler $(B)/test_crush $(B)/test_scrollaug $(B)/test_zwin $(B)/test_l0_stream $(B)/test_phasepack
	./$(B)/test_target_erode
	./$(B)/test_wind_sampler
	./$(B)/test_crush
	./$(B)/test_scrollaug
	./$(B)/test_papyrus
	./$(B)/test_cond
	./$(B)/test_scanprep
	./$(B)/test_sweep
	./$(B)/test_gpu_input
	./$(B)/test_pack_tm
	./$(B)/test_wind4
	./$(B)/test_pgrad
	./$(B)/test_gyrec
	./$(B)/test_unwrap2_sampler $(B)/unwrap2_fixture
	./$(B)/test_restore_task
	./$(B)/test_phase_interp
	$(MAKE) test-sheet
	./$(B)/test_wgrad_grid
	./$(B)/test_sheet_gpu
	./$(B)/test_cover
	./$(B)/test_ct_augment
	./$(B)/test_scan_augment
	./$(B)/test_side
	./$(B)/test_votes
	./$(B)/test_wind
	./$(B)/test_phase_check
	./$(B)/test_fuse
	./$(B)/test_gn_frozen
	./$(B)/test_l0_norm
	./$(B)/test_l0_slab
	./$(B)/test_zwin
	./$(B)/test_l0_stream
	UFSM_L0_STREAM_POISON=1 ./$(B)/test_l0_stream
	./$(B)/test_infer_mem
	./$(B)/test_half
	./$(B)/test_unwrap_targets
	./$(B)/test_gpu_aug
	./$(B)/test_fit
	./$(B)/test_fit_join
	./$(B)/test_pairacc
	bash tests/test_wbench_stats.sh $(B)/ufsm
	bash tests/test_gpuq.sh
	./$(B)/test_robust
	./$(B)/test_reg
	./$(B)/test_reg_render
	./$(B)/test_surf
	./$(B)/test_stitch
	./$(B)/test_fblend
	./$(B)/test_tta
	./$(B)/test_predict_unwrap2 ./$(B)/ufsm
	./$(B)/test_coreline
	./$(B)/test_fit_w4
	./$(B)/test_repair
	./$(B)/test_fit_emb
	./$(B)/test_distill
	./$(B)/test_tiffw
	./$(B)/test_imgops
	./$(B)/test_tfx
	./$(B)/test_dino_loss
	UFSM_TEST_D=16 ./$(B)/test_dino_loss
	./$(B)/test_head32
	./$(B)/test_sslstat
	./$(B)/test_attn
	./$(B)/test_attn_ray
	./$(B)/test_attn_dfm
	./$(B)/test_hybrid_identity
	./$(B)/test_dinor
	./$(B)/test_dino_pts
	./$(B)/test_ink_splat
	./$(B)/test_pyports
	./$(B)/test_edt
	./$(B)/test_volcomp
	./$(B)/test_typed_zarr
	python3 tests/test_http_reader.py
	./$(B)/test_checkpoint_runtime
	./$(B)/test_infer_buffers
	./$(B)/test_recompute_live
	./$(B)/test_share_enc_a1
	python3 tests/test_pipeline_cli.py
	python3 tests/test_cover_training.py
	./$(B)/test_checkpoint
	./$(B)/test_optimizer_owners
	./$(B)/test_wide_up_grad
	UFSM_TEST_SR=0 ./$(B)/test_wide_up_grad
	./$(B)/test_gn_contract
	UFSM_FUSED_STORED_GN=0 ./$(B)/test_gn_contract
	./$(B)/test_stem_precision
	./$(B)/test_wgrad_staging
	./$(B)/test_eval
	./$(B)/test_sample_ops
	./$(B)/test_aug_preset
	./$(B)/test_raster
	python3 tests/test_sampler_safety.py
	python3 tests/test_ink_ext_v8in.py
	python3 tests/test_ink_ext_hecate.py
	python3 tests/test_ink_ext_reader3.py
	python3 tests/test_ink_ext_tiling.py
	python3 tests/test_dinovol.py
	python3 tests/test_dinovol_embed.py
	./$(B)/test_mx_sw
	./$(B)/test_mx
	./$(B)/test_mx4
	./$(B)/test_grad_mx4
	./$(B)/test_l3_sched
	UFSM_L0_NORM=1 ./$(B)/test_grad_mx4
	UFSM_L0_NORM=2 ./$(B)/test_grad_mx4
	UFSM_L0_NORM=3 ./$(B)/test_grad_mx4
	UFSM_L0_NORM=4 UFSM_L0_SLAB=8 UFSM_L0_SLAB_KERN=1 ./$(B)/test_grad_mx4
	UFSM_F4W_LAYOUT=1 ./$(B)/test_mx4
	./$(B)/test_rc
	./$(B)/test_muon
	./$(B)/test_muonc
	UFSM_FUSED_UP=0 UFSM_F16=1 ./$(B)/test_unet
	UFSM_RECOMPUTE=2 UFSM_F16=1 ./$(B)/test_unet
	./$(B)/test_json
	./$(B)/test_phasepack
	./$(B)/test_nn
	./$(B)/test_unet
	UFSM_SCALE_COND=1 ./$(B)/test_unet
	UFSM_DS=1 ./$(B)/test_unet
	UFSM_EMB=3 ./$(B)/test_unet
	UFSM_EMB=3 UFSM_DS=1 UFSM_RECOMPUTE=2 ./$(B)/test_unet
	UFSM_ENC_BLOCKS=0,1,2 UFSM_RB_NONZERO=1 UFSM_BIG_ENC_BLOCKS=0,1,2,2 UFSM_F16=1 ./$(B)/test_unet
	UFSM_L0_NORM=1 ./$(B)/test_unet
	UFSM_L0_NORM=2 ./$(B)/test_unet
	UFSM_L0_NORM=3 ./$(B)/test_unet
	UFSM_L0_NORM=4 ./$(B)/test_unet
	UFSM_P=48 UFSM_B=1 ./$(B)/test_unet
	UFSM_F16=1 ./$(B)/test_unet
	./$(B)/test_fused
	UFSM_F16=1 ./$(B)/test_fused
	./$(B)/test_split
	UFSM_TEST_POLICY=all=fp4:fp4:fp8,enc0.c1=fp16 ./$(B)/test_split
	UFSM_TEST_GN_STORED=1 UFSM_TEST_INPUT_PREC=8 ./$(B)/test_split
	UFSM_TEST_GN_STORED=1 UFSM_TEST_INPUT_PREC=8 UFSM_RC_KEEP_COARSE=1 ./$(B)/test_split
	UFSM_TEST_GN_STORED=1 UFSM_TEST_INPUT_PREC=8 UFSM_TEST_POLICY=all=fp4:fp4:fp8,enc0.c1=fp16 ./$(B)/test_split
	./tests/test_formats.sh
	./tests/test_zarr.sh

clean:
	rm -rf build build-verify build86 build86-verify build86p build86p-verify

.PHONY: all test test-all clean

$(B)/bench_wgrad4: tests/bench_wgrad4.c $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm

# header dependencies of the CUDA units (nvcc -MMD -MP)
-include $(wildcard $(B)/nn_*.d $(B)/lp_*.d)

$(B)/test_cond: tests/test_cond.c src/sample.c src/*.h $(SAMPLE_TEST_OBJ)
	$(CC) $(CFLAGS) $(CPPFLAGS) tests/test_cond.c $(SAMPLE_TEST_OBJ) -o $@ $(LDLIBS)

test-all: $(B)/test_cond

$(B)/test_scanprep: tests/test_scanprep.c src/sample.c src/*.h $(SAMPLE_TEST_OBJ) $(B)/scanprep.o
	$(CC) $(CFLAGS) $(CPPFLAGS) tests/test_scanprep.c $(SAMPLE_TEST_OBJ) $(B)/scanprep.o -o $@ $(LDLIBS)
test-all: $(B)/test_scanprep

$(B)/test_sweep: tests/test_sweep.c $(OBJ) $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(LDLIBS) $(CUDALIBS)

test-all: $(B)/test_sweep
$(B)/test_restore_task: tests/test_restore_task.c $(OBJ) $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(LDLIBS) $(CUDALIBS)
test-all: $(B)/test_restore_task
test-all: $(B)/test_gpu_input
test-all: $(B)/test_pack_tm
test-all: $(B)/test_wind4
test-all: $(B)/test_pgrad
test-all: $(B)/test_gyrec
test-all: $(B)/test_unwrap2_sampler
test-all: $(B)/test_phase_interp
test-all: $(B)/test_l3_sched
$(B)/test_reg: tests/test_reg.c $(B)/reg.o $(B)/reg_init.o $(B)/zarr3.o $(B)/store.o $(B)/json.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(LDLIBS)
test-all: $(B)/test_reg
$(B)/test_reg_render: tests/test_reg_render.c $(OBJ) $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(LDLIBS) $(CUDALIBS)
test-all: $(B)/test_reg_render
