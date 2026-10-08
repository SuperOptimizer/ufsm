# ufsm — C23 host code (gcc), CUDA kernels (nvcc) added later.
# Default build: the fp16 / MX kernels used by training and inference. VERIFY=1 builds the bf16 and fp32 tensor-core
# instantiations as well (verification, legacy --f16 0) into build-verify/, so the two builds never invalidate each other.
VERIFY ?= 0
ifeq ($(VERIFY),1)
B := build-verify
TYPEFLAGS := -DUFSM_ALL_TYPES
else
B := build
TYPEFLAGS :=
endif

# ccache (when installed) makes recompiling unchanged sources free: switching VERIFY, make clean, other worktrees.
# CCACHE= disables it.
CCACHE  ?= $(shell command -v ccache 2>/dev/null)
CC      := $(CCACHE) gcc
NVCC    ?= $(CCACHE) nvcc
CFLAGS  ?= -std=c23 -O3 -march=native -g -Wall -Wextra -Wno-unused-parameter -Wno-format-truncation -pthread
CPPFLAGS = -D_GNU_SOURCE -Isrc -Ithird_party -Ithird_party/surfcomp $(shell pkg-config --cflags libcurl libzstd blosc zlib libcrypto)
LDLIBS   = $(shell pkg-config --libs libcurl libzstd blosc zlib libcrypto) -lm -lpthread
CUDA    ?= /usr/local/cuda
NVFLAGS ?= -O3 -arch=sm_120 -use_fast_math -Xcompiler -fno-threadsafe-statics -Isrc   # fast math: +4% step time, FD tests still pass
CUDALIBS = -L$(CUDA)/lib64 -lcudart

SRC  = src/json.c src/checkpoint.c src/sheet.c src/store.c src/zarr3.c src/sources.c src/sample.c src/band.c src/ct_augment.c src/spatial_augment.c src/cover.c src/zarr2.c src/tiff.c src/z3w.c src/hf.c src/ingest.c src/zipr.c src/train.c src/unet.c src/split.c src/predict.c src/eval.c
OBJ  = $(patsubst src/%.c,$(B)/%.o,$(SRC)) $(B)/surfcomp.o

all: $(B)/ufsm

# make test: the default binaries (used by the python CLI / sheet tests), then the whole suite on the VERIFY=1 build
test:
	$(MAKE) VERIFY=0 build/ufsm build/test_sheet build/make_pipeline_fixture build/test_http_reader build/test_sampler_safety build/check_surface_samples build/test_formats
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
	$(NVCC) -O3 -gencode arch=compute_120a,code=sm_120a -use_fast_math -Xcompiler -fno-threadsafe-statics $(TYPEFLAGS) -Xfatbin=-compress-all -MMD -MP -Isrc -c $< -o $@
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
$(B)/bench_head: tests/bench_head.c $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
$(B)/bench_gn: tests/bench_gn.c $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm

$(B)/bench_mem: tests/bench_mem.c $(B)/unet.o $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm

$(B)/test_mx: tests/test_mx.c $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
$(B)/test_mx4: tests/test_mx4.c $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm

$(B)/test_unet: tests/test_unet.c $(B)/unet.o $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
$(B)/test_split: tests/test_split.c $(B)/unet.o $(B)/split.o $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
SAMPLE_TEST_OBJ = $(B)/band.o $(B)/sheet.o $(B)/ct_augment.o $(B)/spatial_augment.o $(B)/cover.o $(B)/json.o $(B)/store.o $(B)/zarr3.o $(B)/sources.o $(B)/zarr2.o $(B)/tiff.o $(B)/z3w.o $(B)/hf.o $(B)/zipr.o
$(B)/test_sample_ops: tests/test_sample_ops.c src/sample.c src/*.h $(SAMPLE_TEST_OBJ)
	$(CC) $(CFLAGS) $(CPPFLAGS) tests/test_sample_ops.c $(SAMPLE_TEST_OBJ) -o $@ $(LDLIBS)
$(B)/test_sampler_safety: tests/test_sampler_safety.c src/sample.c src/*.h $(SAMPLE_TEST_OBJ)
	$(CC) $(CFLAGS) $(CPPFLAGS) tests/test_sampler_safety.c $(SAMPLE_TEST_OBJ) -o $@ $(LDLIBS)
$(B)/test_spatial_augment: tests/test_spatial_augment.c src/sample.c src/*.h $(SAMPLE_TEST_OBJ)
	$(CC) $(CFLAGS) $(CPPFLAGS) tests/test_spatial_augment.c $(SAMPLE_TEST_OBJ) -o $@ $(LDLIBS)

$(B)/test_raster: tests/test_raster.c src/ingest.c src/*.h $(SAMPLE_TEST_OBJ) $(B)/surfcomp.o
	$(CC) $(CFLAGS) $(CPPFLAGS) tests/test_raster.c $(SAMPLE_TEST_OBJ) $(B)/surfcomp.o -o $@ $(LDLIBS)

$(B)/check_surface_samples: tools/check_surface_samples.c $(B)/sample.o $(SAMPLE_TEST_OBJ)
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(LDLIBS)

$(B)/bench_sampler: tests/bench_sampler.c $(B)/sample.o $(B)/band.o $(B)/sheet.o $(B)/ct_augment.o $(B)/spatial_augment.o $(B)/cover.o $(B)/sources.o $(B)/zarr3.o $(B)/zarr2.o $(B)/store.o $(B)/json.o $(B)/z3w.o $(B)/tiff.o $(B)/zipr.o $(B)/hf.o $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm -lcurl -lzstd -lblosc -lz -lcrypto
$(B)/bench_read: tests/bench_read.c $(B)/sample.o $(B)/band.o $(B)/sheet.o $(B)/ct_augment.o $(B)/spatial_augment.o $(B)/cover.o $(B)/sources.o $(B)/zarr3.o $(B)/zarr2.o $(B)/store.o $(B)/json.o $(B)/z3w.o $(B)/tiff.o $(B)/zipr.o $(B)/hf.o $(B)/nn.o $(B)/nn_fp8.o
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

$(B)/test_ct_augment: tests/test_ct_augment.c $(B)/ct_augment.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ -lm

$(B)/test_target_erode: tests/test_target_erode.c src/target_erode.h | $(B)
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -o $@

test-all: $(B)/test_target_erode

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
	python3 tests/test_sheet_geometry.py
	python3 tests/test_sheet_native.py
	python3 tests/test_sheet_pipeline.py
	python3 tests/test_sheet_watch.py
	python3 tests/test_extend_surface_training.py

test-sheet-gpu: $(B)/ufsm $(B)/test_sheet_gpu $(B)/test_wgrad_grid $(B)/make_pipeline_fixture
	./$(B)/test_wgrad_grid
	./$(B)/test_sheet_gpu
	python3 tests/test_sheet_cli.py

$(B)/test_json: tests/test_json.c $(B)/json.o
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

test-all: $(B)/test_sheet $(B)/test_sheet_gpu $(B)/test_cover $(B)/test_ct_augment $(B)/check_surface_samples $(B)/test_raster $(B)/test_http_reader $(B)/test_wide_up_grad $(B)/test_wgrad_staging $(B)/test_stem_precision $(B)/test_gn_contract $(B)/test_sampler_safety $(B)/test_optimizer_owners $(B)/test_infer_buffers $(B)/test_recompute_live $(B)/test_share_enc_a1 $(B)/test_checkpoint_runtime $(B)/make_pipeline_fixture $(B)/test_checkpoint $(B)/test_eval $(B)/test_sample_ops $(B)/test_json $(B)/test_nn $(B)/test_unet $(B)/test_fused $(B)/test_formats $(B)/ufsm $(B)/test_mx $(B)/test_mx4 $(B)/test_rc $(B)/test_split $(B)/test_muon
	./$(B)/test_target_erode
	$(MAKE) test-sheet
	./$(B)/test_wgrad_grid
	./$(B)/test_sheet_gpu
	python3 tests/test_sheet_cli.py
	./$(B)/test_cover
	./$(B)/test_ct_augment
	python3 tests/test_training_cover.py
	python3 tests/test_production_cover.py
	python3 tests/test_http_reader.py
	python3 tests/test_eval_holdouts.py
	python3 tests/test_evaluation_plan.py
	python3 tests/test_gpu_leases.py
	./$(B)/test_checkpoint_runtime
	./$(B)/test_infer_buffers
	./$(B)/test_recompute_live
	./$(B)/test_share_enc_a1
	python3 tests/test_pipeline_cli.py
	python3 tests/test_cover_training.py
	python3 tests/test_production.py
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
	./$(B)/test_raster
	python3 tests/test_surface_store.py
	python3 tests/test_sampler_safety.py
	./$(B)/test_mx
	./$(B)/test_mx4
	UFSM_F4W_LAYOUT=1 ./$(B)/test_mx4
	./$(B)/test_rc
	./$(B)/test_muon
	UFSM_FUSED_UP=0 UFSM_F16=1 ./$(B)/test_unet
	UFSM_RECOMPUTE=2 UFSM_F16=1 ./$(B)/test_unet
	./$(B)/test_json
	./$(B)/test_nn
	./$(B)/test_unet
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
	rm -rf build build-verify

.PHONY: all test test-all clean

$(B)/bench_wgrad4: tests/bench_wgrad4.c $(B)/nn.o $(B)/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm

# header dependencies of the CUDA units (nvcc -MMD -MP)
-include $(wildcard $(B)/nn_*.d $(B)/lp_*.d)
