# ufsm — C23 host code (gcc), CUDA kernels (nvcc) added later.
CC      ?= gcc
NVCC    ?= nvcc
CFLAGS  ?= -std=c23 -O3 -march=native -g -Wall -Wextra -Wno-unused-parameter -Wno-format-truncation -pthread
CPPFLAGS = -D_GNU_SOURCE -Isrc -Ithird_party -Ithird_party/surfcomp $(shell pkg-config --cflags libcurl libzstd blosc zlib libcrypto)
LDLIBS   = $(shell pkg-config --libs libcurl libzstd blosc zlib libcrypto) -lm -lpthread
CUDA    ?= /usr/local/cuda
NVFLAGS ?= -O3 -arch=sm_120 -use_fast_math -Xcompiler -fno-threadsafe-statics -Isrc   # fast math: +4% step time, FD tests still pass
CUDALIBS = -L$(CUDA)/lib64 -lcudart

SRC  = src/json.c src/checkpoint.c src/sheet.c src/store.c src/zarr3.c src/sources.c src/sample.c src/ct_augment.c src/spatial_augment.c src/cover.c src/zarr2.c src/tiff.c src/z3w.c src/hf.c src/ingest.c src/zipr.c src/train.c src/unet.c src/split.c src/predict.c src/eval.c
OBJ  = $(patsubst src/%.c,build/%.o,$(SRC)) build/surfcomp.o

all: build/ufsm

build/%.o: src/%.c src/*.h third_party/volcomp.h | build
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

build/surfcomp.o: third_party/surfcomp/surfcomp.c third_party/surfcomp/*.h | build
	$(CC) $(CFLAGS) $(CPPFLAGS) -Wno-all -Wno-extra -c $< -o $@

build/ufsm: $(OBJ) build/main.o build/nn.o build/nn_fp8.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS) $(CUDALIBS)

build/nn.o: src/nn.cu src/nn.h src/nn_lp.h | build
	$(NVCC) $(NVFLAGS) -c $< -o $@
# FP8 / FP4 block-scaled mma (kind::mxf8f6f4 / kind::mxf4) needs the arch-specific sm_120a target
build/nn_fp8.o: src/nn_fp8.cu src/nn.h src/nn_lp.h | build
	$(NVCC) -O3 -gencode arch=compute_120a,code=sm_120a -use_fast_math -Xcompiler -fno-threadsafe-statics -Isrc -c $< -o $@

build/unet.o: src/unet.c src/unet.h src/nn.h | build
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

build/prof_infer: tests/prof_infer.c build/unet.o build/nn.o build/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm

build/test_nn: tests/test_nn.c build/nn.o build/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm

build/bench_conv: tests/bench_conv.c build/nn.o build/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm

build/prec_sweep: tests/prec_sweep.c build/unet.o build/nn.o build/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm

build/test_rc: tests/test_rc.c build/nn.o build/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
build/test_muon: tests/test_muon.c build/nn.o build/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
build/test_checkpoint: tests/test_checkpoint.c build/unet.o build/nn.o build/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
build/test_optimizer_owners: tests/test_optimizer_owners.c src/unet.c src/*.h build/nn.o build/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) tests/test_optimizer_owners.c build/nn.o build/nn_fp8.o -o $@ $(CUDALIBS) -lm

build/test_wide_up_grad: tests/test_wide_up_grad.c src/unet.c src/*.h build/nn.o build/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) tests/test_wide_up_grad.c build/nn.o build/nn_fp8.o -o $@ $(CUDALIBS) -lm
build/test_gn_contract: tests/test_gn_contract.c build/nn.o build/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
build/test_stem_precision: tests/test_stem_precision.c build/nn.o build/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
build/test_wgrad_staging: tests/test_wgrad_staging.c build/nn.o build/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
build/test_infer_buffers: tests/test_infer_buffers.c build/unet.o build/nn.o build/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
build/test_recompute_live: tests/test_recompute_live.c build/unet.o build/nn.o build/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
build/bench_gn: tests/bench_gn.c build/nn.o build/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm

build/bench_mem: tests/bench_mem.c build/unet.o build/nn.o build/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm

build/test_mx: tests/test_mx.c build/nn.o build/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
build/test_mx4: tests/test_mx4.c build/nn.o build/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm

build/test_unet: tests/test_unet.c build/unet.o build/nn.o build/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
build/test_split: tests/test_split.c build/unet.o build/split.o build/nn.o build/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
SAMPLE_TEST_OBJ = build/sheet.o build/ct_augment.o build/spatial_augment.o build/cover.o build/json.o build/store.o build/zarr3.o build/sources.o build/zarr2.o build/tiff.o build/z3w.o build/hf.o build/zipr.o
build/test_sample_ops: tests/test_sample_ops.c src/sample.c src/*.h $(SAMPLE_TEST_OBJ)
	$(CC) $(CFLAGS) $(CPPFLAGS) tests/test_sample_ops.c $(SAMPLE_TEST_OBJ) -o $@ $(LDLIBS)
build/test_sampler_safety: tests/test_sampler_safety.c src/sample.c src/*.h $(SAMPLE_TEST_OBJ)
	$(CC) $(CFLAGS) $(CPPFLAGS) tests/test_sampler_safety.c $(SAMPLE_TEST_OBJ) -o $@ $(LDLIBS)
build/test_spatial_augment: tests/test_spatial_augment.c src/sample.c src/*.h $(SAMPLE_TEST_OBJ)
	$(CC) $(CFLAGS) $(CPPFLAGS) tests/test_spatial_augment.c $(SAMPLE_TEST_OBJ) -o $@ $(LDLIBS)

build/test_raster: tests/test_raster.c src/ingest.c src/*.h $(SAMPLE_TEST_OBJ) build/surfcomp.o
	$(CC) $(CFLAGS) $(CPPFLAGS) tests/test_raster.c $(SAMPLE_TEST_OBJ) build/surfcomp.o -o $@ $(LDLIBS)

build/check_surface_samples: tools/check_surface_samples.c build/sample.o $(SAMPLE_TEST_OBJ)
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(LDLIBS)

build/bench_sampler: tests/bench_sampler.c build/sample.o build/sheet.o build/ct_augment.o build/spatial_augment.o build/cover.o build/sources.o build/zarr3.o build/zarr2.o build/store.o build/json.o build/z3w.o build/tiff.o build/zipr.o build/hf.o build/nn.o build/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm -lcurl -lzstd -lblosc -lz -lcrypto
build/bench_read: tests/bench_read.c build/sample.o build/sheet.o build/ct_augment.o build/spatial_augment.o build/cover.o build/sources.o build/zarr3.o build/zarr2.o build/store.o build/json.o build/z3w.o build/tiff.o build/zipr.o build/hf.o build/nn.o build/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm -lcurl -lzstd -lblosc -lz -lcrypto
build/fwd_nan: tests/fwd_nan.c build/unet.o build/nn.o build/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
build/test_fused: tests/test_fused.c build/nn.o build/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
build/bench_lp: tests/bench_lp.c build/nn.o build/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
build/train_lp: tests/train_lp.c build/unet.o build/nn.o build/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm

build/test_formats: tests/test_formats.c $(OBJ) build/nn.o build/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(LDLIBS) $(CUDALIBS)

build/test_ct_augment: tests/test_ct_augment.c build/ct_augment.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ -lm

build/test_cover: tests/test_cover.c build/cover.o build/json.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ -lcrypto -lm

build/test_sheet: tests/test_sheet.c build/sheet.o build/json.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ -lcrypto -lm

build/test_sheet_gpu: tests/test_sheet_gpu.c build/split.o build/nn.o build/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm

build/test_wgrad_grid: tests/test_wgrad_grid.c build/nn.o build/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm

test-sheet: build/test_sheet build/test_spatial_augment
	./build/test_spatial_augment
	python3 tests/test_sheet_geometry.py
	python3 tests/test_sheet_native.py
	python3 tests/test_sheet_pipeline.py
	python3 tests/test_sheet_watch.py

test-sheet-gpu: build/ufsm build/test_sheet_gpu build/test_wgrad_grid build/make_pipeline_fixture
	./build/test_wgrad_grid
	./build/test_sheet_gpu
	python3 tests/test_sheet_cli.py

build/test_json: tests/test_json.c build/json.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(LDLIBS)

build/test_http_reader: tests/test_http_reader.c build/store.o build/zarr3.o build/json.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(LDLIBS)

build/test_checkpoint_runtime: tests/test_checkpoint_runtime.c build/checkpoint.o build/json.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ -lm

build/make_pipeline_fixture: tests/make_pipeline_fixture.c build/z3w.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(LDLIBS)

build/test_eval: tests/test_eval.c src/eval.c $(SAMPLE_TEST_OBJ)
	$(CC) $(CFLAGS) $(CPPFLAGS) tests/test_eval.c $(SAMPLE_TEST_OBJ) -o $@ $(LDLIBS)

test: build/test_wgrad_grid

build:
	mkdir -p build

test: build/test_sheet build/test_sheet_gpu build/test_cover build/test_ct_augment build/check_surface_samples build/test_raster build/test_http_reader build/test_wide_up_grad build/test_wgrad_staging build/test_stem_precision build/test_gn_contract build/test_sampler_safety build/test_optimizer_owners build/test_infer_buffers build/test_recompute_live build/test_checkpoint_runtime build/make_pipeline_fixture build/test_checkpoint build/test_eval build/test_sample_ops build/test_json build/test_nn build/test_unet build/test_fused build/test_formats build/ufsm build/test_mx build/test_mx4 build/test_rc build/test_split
	$(MAKE) test-sheet
	./build/test_wgrad_grid
	./build/test_sheet_gpu
	python3 tests/test_sheet_cli.py
	./build/test_cover
	./build/test_ct_augment
	python3 tests/test_training_cover.py
	python3 tests/test_production_cover.py
	python3 tests/test_http_reader.py
	python3 tests/test_eval_holdouts.py
	python3 tests/test_evaluation_plan.py
	python3 tests/test_gpu_leases.py
	./build/test_checkpoint_runtime
	./build/test_infer_buffers
	./build/test_recompute_live
	python3 tests/test_pipeline_cli.py
	python3 tests/test_cover_training.py
	python3 tests/test_production.py
	./build/test_checkpoint
	./build/test_optimizer_owners
	./build/test_wide_up_grad
	UFSM_TEST_SR=0 ./build/test_wide_up_grad
	./build/test_gn_contract
	UFSM_FUSED_STORED_GN=0 ./build/test_gn_contract
	./build/test_stem_precision
	./build/test_wgrad_staging
	./build/test_eval
	./build/test_sample_ops
	./build/test_raster
	python3 tests/test_surface_store.py
	python3 tests/test_sampler_safety.py
	./build/test_mx
	./build/test_mx4
	UFSM_F4W_LAYOUT=1 ./build/test_mx4
	./build/test_rc
	UFSM_FUSED_UP=0 UFSM_F16=1 ./build/test_unet
	UFSM_RECOMPUTE=2 UFSM_F16=1 ./build/test_unet
	./build/test_json
	./build/test_nn
	./build/test_unet
	UFSM_P=48 UFSM_B=1 ./build/test_unet
	UFSM_F16=1 ./build/test_unet
	./build/test_fused
	UFSM_F16=1 ./build/test_fused
	./build/test_split
	UFSM_TEST_POLICY=all=fp4:fp4:fp8,enc0.c1=fp16 ./build/test_split
	UFSM_TEST_GN_STORED=1 UFSM_TEST_INPUT_PREC=8 ./build/test_split
	UFSM_TEST_GN_STORED=1 UFSM_TEST_INPUT_PREC=8 UFSM_RC_KEEP_COARSE=1 ./build/test_split
	UFSM_TEST_GN_STORED=1 UFSM_TEST_INPUT_PREC=8 UFSM_TEST_POLICY=all=fp4:fp4:fp8,enc0.c1=fp16 ./build/test_split
	./tests/test_formats.sh
	./tests/test_zarr.sh

clean:
	rm -rf build

.PHONY: all test clean

build/bench_wgrad4: tests/bench_wgrad4.c build/nn.o build/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
