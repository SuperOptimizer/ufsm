# ufsm — C23 host code (gcc), CUDA kernels (nvcc) added later.
CC      ?= gcc
NVCC    ?= nvcc
CFLAGS  ?= -std=c23 -O3 -march=native -g -Wall -Wextra -Wno-unused-parameter -Wno-format-truncation -pthread
CPPFLAGS = -D_GNU_SOURCE -Isrc -Ithird_party -Ithird_party/surfcomp $(shell pkg-config --cflags libcurl libzstd blosc zlib)
LDLIBS   = $(shell pkg-config --libs libcurl libzstd blosc zlib) -lm -lpthread
CUDA    ?= /usr/local/cuda
NVFLAGS ?= -O3 -arch=sm_120 -use_fast_math -Xcompiler -fno-threadsafe-statics -Isrc   # fast math: +4% step time, FD tests still pass
CUDALIBS = -L$(CUDA)/lib64 -lcudart

SRC  = src/json.c src/store.c src/zarr3.c src/sources.c src/sample.c src/zarr2.c src/tiff.c src/z3w.c src/hf.c src/ingest.c src/zipr.c src/train.c src/unet.c src/predict.c src/eval.c
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

build/test_nn: tests/test_nn.c build/nn.o build/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm

build/bench_conv: tests/bench_conv.c build/nn.o build/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm

build/prec_sweep: tests/prec_sweep.c build/unet.o build/nn.o build/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm

build/test_rc: tests/test_rc.c build/nn.o build/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm

build/bench_mem: tests/bench_mem.c build/unet.o build/nn.o build/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm

build/test_mx: tests/test_mx.c build/nn.o build/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm

build/test_unet: tests/test_unet.c build/unet.o build/nn.o build/nn_fp8.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(CUDALIBS) -lm
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

build/test_json: tests/test_json.c build/json.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(LDLIBS)

build:
	mkdir -p build

test: build/test_json build/test_nn build/test_unet build/test_fused build/test_formats build/ufsm build/test_mx build/test_rc
	./build/test_mx
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
	./tests/test_formats.sh
	./tests/test_zarr.sh

clean:
	rm -rf build

.PHONY: all test clean
