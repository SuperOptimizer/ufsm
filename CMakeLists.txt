# ufsm: C23 host code + CUDA kernels, all compiled by mcc (microc, cmake/mcc.cmake: host C and the .cu units -> PTX, mcudahw runtime;
# scripts/toolchain/install_mcc.sh). The previous toolchain stays as presets llvm-release / llvm-verify for bisecting:
# (cmake/llvm.cmake: clang, clang -x cuda, lld, llvm-ar, compiler-rt,
# libunwind, libc++). Ninja. Configurations are build directories (CMakePresets.json):
#   build/            release: the fp16 / MX kernels training and inference use                (cmake --preset release)
#   build-verify/     UFSM_VERIFY=ON: + the bf16 / fp32 tensor-core instantiations; the suite runs from here (--preset verify)
#   build86/, build86-verify/, build86p/   UFSM_ARCH=86: Ampere, the lp_* kernels with their software MX paths; build86p: SASS + PTX
# Targets: ufsm (default), suite-bins, test / test-slow / test-all (tests/run_suite.sh, runner options in the SUITE_FLAGS env var),
# suite-check, doctor, every test binary by name (cmake --build build-verify --target test_lat). The binaries land in the build
# directory itself (build/ufsm, build-verify/test_lat), as with the old Makefile.
cmake_minimum_required(VERSION 3.28)
if(NOT CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/mcc.cmake)   # (since 2026-11-14; cmake/llvm.cmake: --preset llvm-release)
endif()
project(ufsm LANGUAGES C CXX)

option(UFSM_VERIFY "bf16 / fp32 tensor-core instantiations as well (-DUFSM_ALL_TYPES; the suite build)" OFF)
set(UFSM_ARCH auto CACHE STRING "GPU kernel family: auto (from the GPU present), 120 (Blackwell 12.x, lp_* on the block-scaled MMA) or 86 (software MX)")
set(UFSM_GPU_SM auto CACHE STRING "SASS target: auto (compute capability of the GPU present, e.g. 120 / 86 / 89), a number, or portable (sm_120 + PTX / sm_86 + PTX)")
# Native device code (2026-11-14): SASS for the GPU actually present, no PTX (nothing to JIT). The binary then runs only on that
# compute capability (and newer minors of the same major for non-'a' targets): every machine, pods included, builds its own
# (scripts/pod/setup.sh). Without a visible GPU (a build box, CUDA_VISIBLE_DEVICES=) auto falls back to the portable targets.
if(UFSM_ARCH STREQUAL "auto" OR UFSM_GPU_SM STREQUAL "auto")
  execute_process(COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader OUTPUT_VARIABLE _cc ERROR_QUIET)
  string(REGEX MATCH "[0-9]+\\.[0-9]+" _cc "${_cc}")
  string(REPLACE "." "" _cc "${_cc}")
endif()
if(UFSM_ARCH STREQUAL "auto")
  if(_cc MATCHES "^12[0-9]$" OR NOT _cc)
    set(UFSM_ARCH 120)
  else()
    set(UFSM_ARCH 86)
  endif()
endif()
if(UFSM_GPU_SM STREQUAL "auto")
  if(_cc)
    set(UFSM_GPU_SM ${_cc})
  else()
    set(UFSM_GPU_SM portable)
  endif()
endif()
message(STATUS "ufsm: GPU kernels ARCH=${UFSM_ARCH}, SASS ${UFSM_GPU_SM}")
# $BUILD/native_id: the CPU / GPU this build is native to (scripts/pod/setup.sh refuses a foreign binary)
execute_process(COMMAND sh ${CMAKE_SOURCE_DIR}/cmake/native_id.sh OUTPUT_FILE ${CMAKE_BINARY_DIR}/native_id)
option(UFSM_LLVM_RUNTIME "compiler-rt builtins and crt files instead of libgcc.a / gcc crtbegin" ON)
option(UFSM_CCACHE "compile through ccache when it is installed" ON)
option(UFSM_STRICT_MATH "IEEE host C (no contraction) and IEEE / no-FMA UFSM_STRICT_UNITS: the bit-exact GPU-vs-CPU build (pre 2026-11-14 default)" OFF)
set(UFSM_CUDA_COMPILER nvcc CACHE STRING "device compiler: nvcc (nvcc -ccbin clang++, the default: clang -x cuda kernels ran 1.35x slower on sm_120, DESIGN 2026-11-13) or clang (clang -x cuda)")
set(UFSM_CUDA "" CACHE PATH "CUDA toolkit root (default: /usr/local/cuda, else the newest /usr/local/cuda-*, else nvcc's prefix)")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)   # (the flags below are explicit; Release only keeps CMake from adding its own)
endif()
set(CMAKE_C_FLAGS_RELEASE "")
set(CMAKE_CXX_FLAGS_RELEASE "")
# CMake seeds CMAKE_<LANG>_FLAGS from the environment (CFLAGS, CXXFLAGS, CUDAFLAGS) at the first configure and appends them to every
# compile: a shell with CFLAGS=-ffast-math exported would change every object silently. Extra flags go through UFSM_EXTRA_CFLAGS.
set(CMAKE_C_FLAGS "")
set(CMAKE_CXX_FLAGS "")

# --- toolchain -------------------------------------------------------------------------------------------------------------
# UFSM_TOOLCHAIN: llvm (cmake/llvm.cmake: clang + nvcc / clang -x cuda) or mcc (cmake/mcc.cmake: mcc for the host C and the CUDA
# units, mcudahw for the runtime; --preset mcc). The mcc mode needs no CUDA toolkit: UFSM_CUDA is mcudahw's directory.
set(UFSM_TOOLCHAIN llvm CACHE STRING "llvm (cmake/llvm.cmake) or mcc (cmake/mcc.cmake)")
if(UFSM_TOOLCHAIN STREQUAL "mcc")
  set(UFSM_CUDA_COMPILER mcc CACHE STRING "" FORCE)
  set(UFSM_CUDA ${UFSM_MCC_CUDA} CACHE PATH "" FORCE)
  set(UFSM_CUDA_VERSION mcc)
  set(UFSM_LLVM_RUNTIME OFF CACHE BOOL "" FORCE)
  set(UFSM_CCACHE OFF CACHE BOOL "" FORCE)   # (mcc compiles a unit in less time than ccache takes to hash it)
  # A build dir keeps the compiler of its first configure (CMakeFiles/<ver>/CMakeCCompiler.cmake overrides the toolchain file), and
  # UFSM_MCC_CUDA is cached from it: a later -DUFSM_MCC= only changed the cache entry while every object was still built by the old
  # mcc ("no work to do"; the 2026-11-16 e52bb39 pin round gated 9f63cdd binaries that way).
  file(REAL_PATH ${UFSM_MCC} _mcc_want)
  file(REAL_PATH ${CMAKE_C_COMPILER} _mcc_have)
  if(NOT _mcc_want STREQUAL _mcc_have)
    message(FATAL_ERROR "mcc: this build dir was configured with ${CMAKE_C_COMPILER}, UFSM_MCC is now ${UFSM_MCC}: CMake keeps the first "
                        "compiler, so delete ${CMAKE_BINARY_DIR} and configure again")
  endif()
  if(NOT EXISTS ${UFSM_CUDA}/include/mcuda_device.h OR NOT EXISTS ${UFSM_CUDA}/lib64/libcudart.a)
    message(FATAL_ERROR "mcc: no mcudahw at ${UFSM_CUDA} (in the microc tree: sh mkcuda.sh MCC DIR hw; or -DUFSM_MCC_CUDA=)")
  endif()
elseif(NOT CMAKE_C_COMPILER_ID STREQUAL "Clang")
  message(FATAL_ERROR "ufsm builds with clang (cmake/llvm.cmake) or mcc (cmake/mcc.cmake); got ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER}")
endif()
if(UFSM_TOOLCHAIN STREQUAL "mcc")
  # (no toolkit, no ptxas: the driver compiles mcc's PTX at run time)
elseif(NOT UFSM_CUDA)
  if(EXISTS /usr/local/cuda/bin/ptxas)
    set(UFSM_CUDA /usr/local/cuda)
  else()
    file(GLOB _cudas /usr/local/cuda-*/bin/ptxas)
    list(SORT _cudas COMPARE NATURAL ORDER DESCENDING)
    if(_cudas)
      list(GET _cudas 0 _p)
      cmake_path(GET _p PARENT_PATH _p)
      cmake_path(GET _p PARENT_PATH UFSM_CUDA)
    else()
      find_program(_nvcc nvcc)
      if(_nvcc)
        cmake_path(GET _nvcc PARENT_PATH _p)
        cmake_path(GET _p PARENT_PATH UFSM_CUDA)
      else()
        message(FATAL_ERROR "no CUDA toolkit found (set -DUFSM_CUDA=/usr/local/cuda-13.x)")
      endif()
    endif()
  endif()
  set(UFSM_CUDA ${UFSM_CUDA} CACHE PATH "" FORCE)
endif()
if(NOT UFSM_TOOLCHAIN STREQUAL "mcc")
  execute_process(COMMAND ${UFSM_CUDA}/bin/ptxas --version OUTPUT_VARIABLE _pv ERROR_QUIET)
  string(REGEX MATCH "release ([0-9.]+), V([0-9.]+)" _m "${_pv}")
  set(UFSM_CUDA_VERSION ${CMAKE_MATCH_2})
  if(NOT UFSM_CUDA_VERSION)
    message(FATAL_ERROR "${UFSM_CUDA}/bin/ptxas --version: no version (${_pv})")
  endif()
endif()
if(UFSM_CCACHE)
  find_program(UFSM_CCACHE_EXE ccache)
  if(UFSM_CCACHE_EXE)
    # shared across worktrees and suite dirs: paths relative to the tree, the cwd not hashed (-g would key every object on it)
    set(_launch ${CMAKE_COMMAND} -E env CCACHE_BASEDIR=${CMAKE_SOURCE_DIR} CCACHE_NOHASHDIR=1 ${UFSM_CCACHE_EXE})
    set(CMAKE_C_COMPILER_LAUNCHER ${_launch})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${_launch})
  endif()
endif()
find_package(PkgConfig REQUIRED)
pkg_check_modules(UFSM_DEPS REQUIRED IMPORTED_TARGET libcurl libzstd blosc zlib libcrypto)

# --- flags -----------------------------------------------------------------------------------------------------------------
# Every optimisation, every unit (2026-11-14, DESIGN "Fast math everywhere ..."), unless UFSM_STRICT_MATH=ON.
# Host C: -O3 -march=native -mtune=native (UFSM_MARCH; binaries are not portable: every machine, pods included, builds its own;
# -DUFSM_MARCH=x86-64-v3 / znver2 ... is the explicit override for a binary run elsewhere, else SIGILL), -ffast-math
# (-funsafe-math-optimizations: reassociation, reciprocal division, no signed zeros, no trapping math; approximate libm, no errno)
# with -ffp-contract=fast (FMA across statements), and FTZ / DAZ in MXCSR at startup (src/fpenv.c, linked into every binary; the
# GPU already flushes). -fno-plt, -fno-semantic-interposition, -fomit-frame-pointer; -g stays (line tables for perf / gdb).
# The one exception: -fno-finite-math-only. Not a tuning: ~226 isnan / isfinite / NaN-fill uses (the loss's NaN guards, fwd_nan,
# zarr fill values, volcomp's typed-chunk checks) are folded to false under finite-math-only, i.e. the guards silently deleted --
# a correctness bug, not a speed gain. Nothing in src/ depends on signed zeros (the copysignf sites take the sign of nonzero
# data) or on errno from libm (the errno reads follow strtod / I/O). volcomp's surface-chunk base (whose entropy contexts are read
# from its own reconstruction, so encoder and decoder must agree to the bit) is fenced by its own `#pragma clang fp
# contract(off) reassociate(off)` (third_party/volcomp.h VF_STRICT_BEGIN), only + and *, so it stays bit-exact.
# UFSM_STRICT_MATH=ON: the old IEEE host flags, no FTZ / DAZ. -ffp-contract=off: gcc -std=c23 (ISO mode) never contracted a*b+c
# into an FMA, clang's default (on) would, and the byte-exact host references (volcomp, count-field, the GPU-vs-CPU tests) are
# written against uncontracted arithmetic. -fno-math-errno: value-neutral, lets sqrtf / fabsf vectorise without the errno call-out.
if(UFSM_STRICT_MATH)
  set(UFSM_FP_CFLAGS -ffp-contract=off -fno-math-errno)
else()
  set(UFSM_FP_CFLAGS -ffast-math -fno-finite-math-only -ffp-contract=fast)
endif()
set(UFSM_MARCH native CACHE STRING "host -march: native (default), or a portable level for binaries run elsewhere (x86-64-v3, znver2 ...)")
if(UFSM_MARCH STREQUAL "native")
  set(UFSM_MARCH_FLAGS -march=native -mtune=native)
else()
  set(UFSM_MARCH_FLAGS -march=${UFSM_MARCH})
endif()
set(UFSM_CFLAGS -std=c23 -O3 ${UFSM_MARCH_FLAGS} -g -Wall -Wextra -Werror=comment -Werror=return-type -Werror=implicit-function-declaration
    -Werror=incompatible-pointer-types -Werror=int-conversion -Werror=format -Werror=uninitialized
    -Wno-error=incompatible-pointer-types-discards-qualifiers -Wno-unused-parameter -pthread
    -fno-plt -fno-semantic-interposition -fomit-frame-pointer ${UFSM_FP_CFLAGS})
# Link-time optimisation of the host C (lld): full LTO by default in the release configurations; off under UFSM_VERIFY (the suite
# links ~250 test executables against CORE, each a whole-program LTO link) -- so the suite tests the non-LTO host code; thin is
# the cheaper middle. The bitcode carries -march / fast-math per function; the link repeats -O3 -march=native for the codegen.
if(UFSM_VERIFY)
  set(UFSM_LTO off CACHE STRING "host LTO: full | thin | off")
else()
  set(UFSM_LTO full CACHE STRING "host LTO: full | thin | off")
endif()
set(UFSM_LTO_JOBS 8 CACHE STRING "ThinLTO backend threads per link (desk rule: 8)")
# (mcc, toolchain-mcc branch: -flto fails under -mglibc and its PGO is unmeasured -- microc 45156e7: both off there)
if(UFSM_TOOLCHAIN STREQUAL "mcc")
  set(UFSM_LTO off)
  set(UFSM_PGO "")
endif()
if(NOT UFSM_LTO STREQUAL "off")
  list(APPEND UFSM_CFLAGS -flto=${UFSM_LTO})
  add_link_options(-flto=${UFSM_LTO} -O3 ${UFSM_MARCH_FLAGS} -Wl,--lto-O3)
  if(UFSM_LTO STREQUAL "thin")
    add_link_options(-Wl,--thinlto-jobs=${UFSM_LTO_JOBS})
  endif()
endif()
# PGO (clang instrumentation): UFSM_PGO=gen builds an instrumented host (profiles to UFSM_PGO_DIR/%m-%p.profraw);
# scripts/toolchain/pgo_collect.sh runs the workload and merges them into UFSM_PGO_DIR/ufsm.profdata, which UFSM_PGO=use reads.
# Host C only (the device code has no PGO). A profile from older sources is used where it still matches (warnings off).
set(UFSM_PGO "" CACHE STRING "profile-guided optimisation of the host C: '' | gen | use")
set(UFSM_PGO_DIR ${CMAKE_SOURCE_DIR}/pgo CACHE PATH "PGO profile directory (gen writes *.profraw there, use reads ufsm.profdata)")
if(UFSM_PGO STREQUAL "gen")
  list(APPEND UFSM_CFLAGS -fprofile-generate=${UFSM_PGO_DIR})
  add_link_options(-fprofile-generate=${UFSM_PGO_DIR})
elseif(UFSM_PGO STREQUAL "use")
  if(NOT EXISTS ${UFSM_PGO_DIR}/ufsm.profdata)
    message(FATAL_ERROR "UFSM_PGO=use: no ${UFSM_PGO_DIR}/ufsm.profdata (scripts/toolchain/pgo_collect.sh makes it)")
  endif()
  list(APPEND UFSM_CFLAGS -fprofile-use=${UFSM_PGO_DIR}/ufsm.profdata -Wno-profile-instr-unprofiled -Wno-profile-instr-out-of-date
       -Wno-profile-instr-missing -Wno-backend-plugin)
elseif(NOT UFSM_PGO STREQUAL "")
  message(FATAL_ERROR "UFSM_PGO must be '', gen or use (got ${UFSM_PGO})")
endif()
set(UFSM_EXTRA_CFLAGS "" CACHE STRING "extra host C flags, appended last (experiments: -O0, -fsanitize=...)")
separate_arguments(_extra UNIX_COMMAND "${UFSM_EXTRA_CFLAGS}")
list(APPEND UFSM_CFLAGS ${_extra})
set(UFSM_DEFS _GNU_SOURCE)
if(UFSM_TOOLCHAIN STREQUAL "mcc")
  # -mglibc: the machine's GNU C library (libcurl, zstd, blosc and the CUDA driver are its shared libraries). mcc takes and ignores
  # -W and most -f. -march=native (user 11-14: every machine builds its own binary): mcc reads cpuid (+ XGETBV) and defines exactly
  # what this CPU has; __AVX2__ / __FMA__ / __F16C__ / __BMI2__ switch ufsm's and volcomp's intrinsic paths on (mcc's <immintrin.h>
  # is compiler builtins in VEX form since microc ebdc34c, 97 intrinsics byte-identical to clang). AVX-512 stays undefined until
  # mcc's EVEX lands. A binary built here is for this machine only: pods build their own (scripts/pod/setup.sh).
  # -O3: mcc's loop vectoriser. -fno-math-errno: sqrt in line, as with clang. -g: DWARF 5 lines, variables, CFI.
  # Fast math (DESIGN 2026-11-14 "Fast math everywhere ..."): -ffast-math -fno-finite-math-only, which mcc parses (microc 08f32ee).
  # Real effect today: reassociation (opt_vec vectorises float sums / dots: dot 0.467 -> 0.043 ns/elem at 256-bit); no host FMA
  # contraction, reciprocal or no-signed-zeros transforms in mcc yet (gaps), so results differ from strict by sum order only. Float
  # max / min reductions and the select -> min / max rewrite need -ffinite-math-only and stay off: NaN results stay exact.
  # UFSM_STRICT_MATH=ON: the list without -ffast-math (sums in source order).
  if(UFSM_STRICT_MATH)
    set(UFSM_MCC_FP_CFLAGS)
  else()
    set(UFSM_MCC_FP_CFLAGS -ffast-math -fno-finite-math-only)
  endif()
  set(UFSM_CFLAGS -mglibc -std=c23 -O3 -march=native -fno-math-errno -g -pthread ${UFSM_MCC_FP_CFLAGS})
  separate_arguments(_extra UNIX_COMMAND "${UFSM_EXTRA_CFLAGS}")
  list(APPEND UFSM_CFLAGS ${_extra})
endif()
if(UFSM_STRICT_MATH)
  list(APPEND UFSM_DEFS UFSM_STRICT_MATH)   # tests/fm_tol.h: the GPU-vs-CPU / two-path tolerances collapse to exact
endif()
# for the shell tests (tests/fm_tol.sh reads $B/strict_math): 1 under UFSM_STRICT_MATH, else 0
if(UFSM_STRICT_MATH)
  file(WRITE ${CMAKE_BINARY_DIR}/strict_math "1\n")
else()
  file(WRITE ${CMAKE_BINARY_DIR}/strict_math "0\n")
endif()
set(UFSM_INC ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/third_party ${CMAKE_SOURCE_DIR}/third_party/surfcomp)
# Runtime: compiler-rt (builtins, crtbegin / crtend) instead of libgcc.a and gcc's crt files. The C++ library stays libstdc++ (with
# libgcc_s for unwinding), shared as before: CUDA's headers refuse libc++ on x86 (crt/host_defines.h: "libc++ is not supported on
# x86 system") and libcudart / libcublasLt load libstdc++ anyway.
if(UFSM_LLVM_RUNTIME)
  set(UFSM_RT_LINK --rtlib=compiler-rt --unwindlib=libgcc)
else()
  set(UFSM_RT_LINK)
endif()
set(UFSM_CXX_STDLIB)
add_link_options(${UFSM_RT_LINK})

# CUDA units: clang -x cuda (device: LLVM NVPTX -> ptxas from UFSM_CUDA). nn_* for sm_120 (SASS + compute_120 PTX, as nvcc
# -arch=sm_120), lp_* for sm_120a (the block-scaled mma; SASS only, as nvcc -gencode code=sm_120a).
# (clang embeds no PTX unless asked: --cuda-include-ptx)
if(NOT UFSM_GPU_SM STREQUAL "portable")
  # native: SASS only, the 'a' target on Blackwell 12.x (lp_* need it; nn_* lose nothing by it)
  if(UFSM_ARCH STREQUAL "120")
    set(UFSM_NN_ARCH --cuda-gpu-arch=sm_${UFSM_GPU_SM}a)
  else()
    set(UFSM_NN_ARCH --cuda-gpu-arch=sm_${UFSM_GPU_SM})
  endif()
  set(UFSM_LP_ARCH ${UFSM_NN_ARCH})
elseif(UFSM_ARCH STREQUAL "120")
  set(UFSM_NN_ARCH --cuda-gpu-arch=sm_120 --cuda-include-ptx=sm_120)
  set(UFSM_LP_ARCH --cuda-gpu-arch=sm_120a)
else()
  # Ampere: SASS + PTX for sm_${UFSM_ARCH}; a newer GPU JIT-compiles the PTX, so the sm_86 paths run on Blackwell too (the old PTX=1)
  set(UFSM_NN_ARCH --cuda-gpu-arch=sm_${UFSM_ARCH} --cuda-include-ptx=sm_${UFSM_ARCH})
  set(UFSM_LP_ARCH --cuda-gpu-arch=sm_${UFSM_ARCH} --cuda-include-ptx=sm_${UFSM_ARCH})
endif()
# The CUDA 13 headers clang does not find itself: CCCL (cub / thrust / cuda::std) moved to include/cccl (nvcc adds it).
# clang_cuda_compat.cuh: nvcc's host min / max overloads. --ptxas-path: the PTX ISA version bump (cmake/ptxas.sh.in), generated
# into each build directory and named relative to it (ninja runs the compiles there), so the command line is the same in build/ and
# build-verify/ and across worktrees: ccache shares the objects whose preprocessed source does not depend on UFSM_ALL_TYPES.
# -D_GNU_SOURCE: g++ (nvcc's host compiler) always defines it (CLOCK_MONOTONIC in nn_aug.cu). UFSM_CUDA_VERSION keys ccache on
# the toolkit and UFSM_PTX_ISA on the PTX version (ptxas is not hashed).
set(UFSM_PTX_ISA 9.1 CACHE STRING "PTX ISA version the CUDA units are assembled as (cmake/ptxas.sh.in)")
configure_file(cmake/ptxas.sh.in ${CMAKE_BINARY_DIR}/ptxas @ONLY FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ
               GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
set(UFSM_CU_FLAGS -x cuda --cuda-path=${UFSM_CUDA} --ptxas-path=./ptxas -Wno-unknown-cuda-version -Wno-cuda-compat -O3 -std=c++17
    -fno-threadsafe-statics ${UFSM_CXX_STDLIB} "SHELL:-Xcuda-fatbinary -compress-all" "SHELL:-isystem ${UFSM_CUDA}/include/cccl"
    "SHELL:-include ${CMAKE_SOURCE_DIR}/src/clang_cuda_compat.cuh" "SHELL:-include ${CMAKE_SOURCE_DIR}/src/ufsm_dev.cuh" -D_GNU_SOURCE -DUFSM_CUDA_VERSION=${UFSM_CUDA_VERSION}
    -DUFSM_PTX_ISA=${UFSM_PTX_ISA})
# ptxas --split-compile=N: ptxas optimises the kernels of one unit on N threads. The big lp_* units hold up to 9 kernel
# instantiations; lp_f4fwd_i0p.cu 41 s -> 22 s with 0 (all cores), 25 s with 4 (desk, 2026-11-13). It only shortens the tail of a
# build (the rest is -j bound), but adds threads beyond -j, hence off by default (the desk's 8-job rule); 4 on dedicated machines.
# The SASS is identical with and without it (lp_f4fwd_i0g, cuobjdump -sass).
set(UFSM_PTXAS_SPLIT 0 CACHE STRING "ptxas --split-compile threads per CUDA unit (0 / 1: off)")
if(UFSM_PTXAS_SPLIT GREATER 1)
  list(APPEND UFSM_CU_FLAGS "SHELL:-Xcuda-ptxas --split-compile=${UFSM_PTXAS_SPLIT}")
endif()
# Device fast math (clang -x cuda; nvcc: -use_fast_math below), every unit: flush denormals, approximate division / sqrt /
# transcendentals, FMA contraction "fast" (across statements, as nvcc's fmad). Under UFSM_STRICT_MATH contraction is "on" (within
# one expression): with fast, LLVM fused nn_loss_pack and nn_loss_async_tol + nn_half_grad differently (tests/test_half.c: 546 of
# 6144 gradients 1 ulp apart, the BCE sum too; with on they agree, as under nvcc; forlindesk sm_86, 2026-11-13); test_half
# allows 1 ulp outside strict. NaN / inf stay honoured (no -ffinite-math-only: the NaN guards in the loss and fwd_nan paths must
# survive). Checked 2026-11-13: the float instruction kinds in the PTX match nvcc -use_fast_math (div.approx.ftz,
# sqrt.approx.ftz, ex2.approx.ftz, fma.rn.ftz). ptxas -O3 (its default, made explicit); no LTO of device code (each unit is
# whole-program, no -rdc / device link step: nothing for device LTO to work on).
# ("SHELL:" groups: CMake de-duplicates repeated options, which would drop the second -Xarch_device)
# UFSM_DEVICE_FAST_MATH: the rest of nvcc's fast-math intrinsic mapping (expf -> __expf ..., src/clang_cuda_compat.cuh)
if(UFSM_STRICT_MATH)
  set(_contract on)
else()
  set(_contract fast)
endif()
set(UFSM_CU_FAST -DUFSM_DEVICE_FAST_MATH -fgpu-flush-denormals-to-zero -fgpu-approx-transcendentals "SHELL:-Xarch_device -fapprox-func"
    "SHELL:-Xarch_device -ffp-contract=${_contract}")
list(APPEND UFSM_CU_FLAGS "SHELL:-Xcuda-ptxas -O3")
foreach(f ${UFSM_MARCH_FLAGS})
  list(APPEND UFSM_CU_FLAGS "SHELL:-Xarch_host ${f}")
endforeach()
# UFSM_STRICT_UNITS: the units that, under UFSM_STRICT_MATH=ON, are byte-identical to their CPU versions: IEEE division / sqrt, no
# FMA contraction, no flush (nvcc -fmad=false, no fast math). By default (2026-11-14) they get UFSM_CU_FAST like every other unit
# and the host is fast math too, so GPU and CPU agree within tolerances (tests/fm_tol.h, DESIGN "Fast math everywhere") instead.
# clang leaves libdevice's __CUDA_PREC_SQRT reflection at 0, so sqrtf would be sqrt.approx.f32 even without fast math (nvcc:
# sqrt.rn.f32); the reflection is set explicitly (-mllvm reaches the host pass too, where it does nothing). Checked 2026-11-13: the
# PTX of every strict unit uses the same float instruction kinds as nvcc -fmad=false (div.rn, sqrt.rn, no fma / approx / ftz).
set(UFSM_CU_STRICT "SHELL:-Xarch_device -ffp-contract=off" "SHELL:-mllvm -nvvm-reflect-add=__CUDA_PREC_SQRT=1")
# the host half of every .cu unit contracts as the host C does: fast by default, off under UFSM_STRICT_MATH (the strict units hold
# CPU-side code too: nn_band, nn_render, nn_gpudec ... host functions, compared bit for bit against the .c references under strict;
# under nvcc + g++ -std=c++17 they never fused)
if(UFSM_STRICT_MATH)
  list(APPEND UFSM_CU_FLAGS "SHELL:-Xarch_host -ffp-contract=off")
else()
  list(APPEND UFSM_CU_FLAGS "SHELL:-Xarch_host -ffp-contract=fast")
endif()
set(UFSM_STRICT_UNITS nn_gpudec nn_cf nn_ftrace nn_latvol nn_band nn_fibre nn_inkfly nn_gpuenc nn_mcodes nn_faces nn_render nn_hgs)
set(UFSM_TYPEDEFS UFSM_ARCH=${UFSM_ARCH})
if(UFSM_VERIFY)
  list(APPEND UFSM_TYPEDEFS UFSM_ALL_TYPES)
endif()

# --- objects: one OBJECT library per source (obj_<name>), so a test links exactly the objects it lists ------------------------
add_library(ufsm_ldlibs INTERFACE)
target_link_libraries(ufsm_ldlibs INTERFACE PkgConfig::UFSM_DEPS jpeg m pthread)
add_library(ufsm_cudalibs INTERFACE)
target_link_directories(ufsm_cudalibs INTERFACE ${UFSM_CUDA}/lib64)
if(UFSM_TOOLCHAIN STREQUAL "mcc")
  # mcudahw's static libcudart.a (it dlopens libcuda.so.1 and libcublasLt.so); its libcublasLt.a / libstdc++.a hold nothing
  target_link_libraries(ufsm_cudalibs INTERFACE cudart cublasLt stdc++ dl)
else()
  target_link_libraries(ufsm_cudalibs INTERFACE cudart cublasLt stdc++)
  target_link_options(ufsm_cudalibs INTERFACE -Wl,-rpath,${UFSM_CUDA}/lib64)
endif()

function(ufsm_host_obj name src)
  add_library(obj_${name} OBJECT ${src})
  target_compile_options(obj_${name} PRIVATE ${UFSM_CFLAGS} ${ARGN})
  target_compile_definitions(obj_${name} PRIVATE ${UFSM_DEFS})
  target_include_directories(obj_${name} PRIVATE ${UFSM_INC} ${UFSM_DEPS_INCLUDE_DIRS})
  target_include_directories(obj_${name} SYSTEM PRIVATE ${UFSM_CUDA}/include)
endfunction()

# UFSM_EXACT_HOST / UFSM_EXACT_CU: IEEE in every build, fast math or not (2026-11-15). The volcomp decoders: zarr3.c (the host reader:
# volcomp.h + volcomp_fast.h) and nn_gpudec.cu (the GPU decoder) must give the same bytes as each other and as the decoders the stores
# were checked with, because the nets amplify a +-1 CT voxel: fast math moved 5-8 voxels per 160^3 window by 1 (the host IDCT's
# signed zeros: -fsigned-zeros on zarr3.c alone restored it) and predict's rv then differed on 8 % of the bytes (max 120) of the
# windows holding them, per-window GroupNorm statistics and MX-fp4 block scales spreading it (fm round 11-15, test_predict_heads
# (1d): GPU vs host decode 16 %, max 123). The decode is ~80 % the serial tANS chain: IEEE costs it nothing measurable.
set(UFSM_EXACT_HOST zarr3)
set(UFSM_EXACT_CU nn_gpudec)
set(UFSM_IEEE_CFLAGS -fno-fast-math -ffp-contract=off -fno-math-errno)
file(GLOB UFSM_C_SRC CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/*.c)
set(UFSM_CORE_OBJS)
foreach(f ${UFSM_C_SRC})
  cmake_path(GET f STEM n)
  if(n IN_LIST UFSM_EXACT_HOST)
    ufsm_host_obj(${n} ${f} ${UFSM_IEEE_CFLAGS})
  else()
    ufsm_host_obj(${n} ${f})
  endif()
  if(NOT n MATCHES "^(main|sysinfo_cli)$")
    list(APPEND UFSM_CORE_OBJS obj_${n})
  endif()
endforeach()
ufsm_host_obj(surfcomp ${CMAKE_SOURCE_DIR}/third_party/surfcomp/surfcomp.c -w)
list(APPEND UFSM_CORE_OBJS obj_surfcomp)

if(UFSM_CUDA_COMPILER STREQUAL "nvcc")
  # nvcc for the device code with clang++ as its host compiler (still no gcc): the old Makefile's flags. nvcc 13.x accepts clang
  # < 22 only, hence -allow-unsupported-compiler for clang 22 (the host side of the .cu files is plain C++ glue).
  # The host compiler is probed: nvcc -allow-unsupported-compiler -ccbin clang++ on a one-kernel unit that includes a CUDA runtime
  # header (2026-11-13: the desk's 13.3 and the pod's 13.2 both take clang 22 so; a plain nvcc 13.2 refuses clang 22). If clang++
  # fails, the newest g++ (g++-15 .. g++) is nvcc's host compiler instead; the host C / C++ units stay clang either way.
  # UFSM_CUDA_HOST_COMPILER pins one.
  set(UFSM_CUDA_HOST_COMPILER "" CACHE FILEPATH "nvcc's host compiler (default: clang++ if nvcc accepts it, else the newest g++)")
  if(NOT UFSM_CUDA_HOST_COMPILER)
    set(_probe ${CMAKE_BINARY_DIR}/CMakeFiles/ufsm_nvcc_probe)
    file(MAKE_DIRECTORY ${_probe})
    file(WRITE ${_probe}/p.cu "#include <cuda_runtime.h>\n#include <cstdio>\n__global__ void k(float *x) { x[threadIdx.x] *= 2.f; }\nint f(float *x) { k<<<1, 32>>>(x); return printf(\"%d\\n\", (int)cudaGetLastError()); }\n")
    set(_cands ${CMAKE_CXX_COMPILER})
    foreach(v RANGE 15 11 -1)
      find_program(_gxx NAMES g++-${v} NO_CACHE)
      if(_gxx)
        list(APPEND _cands ${_gxx})
      endif()
      unset(_gxx)
    endforeach()
    find_program(_gxx NAMES g++ NO_CACHE)
    if(_gxx)
      list(APPEND _cands ${_gxx})
    endif()
    foreach(c ${_cands})
      execute_process(COMMAND ${UFSM_CUDA}/bin/nvcc -allow-unsupported-compiler -ccbin ${c} -std=c++17 -c p.cu -o p.o
                      WORKING_DIRECTORY ${_probe} RESULT_VARIABLE _rc OUTPUT_QUIET ERROR_VARIABLE _err)
      if(_rc EQUAL 0)
        set(UFSM_CUDA_HOST_COMPILER ${c})
        break()
      endif()
      message(STATUS "ufsm: nvcc ${UFSM_CUDA_VERSION} does not take ${c} as host compiler")
    endforeach()
    if(NOT UFSM_CUDA_HOST_COMPILER)
      message(FATAL_ERROR "ufsm: nvcc ${UFSM_CUDA_VERSION} accepts none of ${_cands} as host compiler (last error: ${_err}); set -DUFSM_CUDA_HOST_COMPILER=")
    endif()
    set(UFSM_CUDA_HOST_COMPILER ${UFSM_CUDA_HOST_COMPILER} CACHE FILEPATH "" FORCE)
  endif()
  message(STATUS "ufsm: nvcc host compiler ${UFSM_CUDA_HOST_COMPILER}")
  set(CMAKE_CUDA_COMPILER ${UFSM_CUDA}/bin/nvcc)
  set(CMAKE_CUDA_HOST_COMPILER ${UFSM_CUDA_HOST_COMPILER})
  set(CMAKE_CUDA_FLAGS_INIT "-allow-unsupported-compiler")
  set(CMAKE_CUDA_ARCHITECTURES OFF)
  if(UFSM_CCACHE_EXE)
    set(CMAKE_CUDA_COMPILER_LAUNCHER ${_launch})
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_FLAGS_RELEASE "")
  set(CMAKE_CUDA_FLAGS "-allow-unsupported-compiler")   # (not the environment's CUDAFLAGS, see CMAKE_C_FLAGS)
  if(NOT UFSM_GPU_SM STREQUAL "portable")
    if(UFSM_ARCH STREQUAL "120")
      set(UFSM_NN_ARCH "SHELL:-gencode arch=compute_${UFSM_GPU_SM}a,code=sm_${UFSM_GPU_SM}a")
    else()
      set(UFSM_NN_ARCH "SHELL:-gencode arch=compute_${UFSM_GPU_SM},code=sm_${UFSM_GPU_SM}")
    endif()
    set(UFSM_LP_ARCH ${UFSM_NN_ARCH})
  elseif(UFSM_ARCH STREQUAL "120")
    set(UFSM_NN_ARCH -arch=sm_120)
    set(UFSM_LP_ARCH "SHELL:-gencode arch=compute_120a,code=sm_120a")
  else()
    set(UFSM_NN_ARCH "SHELL:-gencode arch=compute_${UFSM_ARCH},code=[sm_${UFSM_ARCH},compute_${UFSM_ARCH}]")
    set(UFSM_LP_ARCH ${UFSM_NN_ARCH})
  endif()
  # -Xptxas -O3 (ptxas' default, explicit), --extra-device-vectorization (nvvm's extra vectoriser), the host side native.
  # No -dlto: device LTO needs -rdc and an nvcc device-link step; the units are whole-program and linked by clang / lld.
  # (ufsm_dev.cuh: logical devices, force-included like clang's -include above; ops audit #8)
  # The host half of every unit as the host C: contraction fast, or off under UFSM_STRICT_MATH (clang++ contracts by default; the
  # old g++ -std=c++17 host did not)
  if(UFSM_STRICT_MATH)
    set(_hcontract off)
  else()
    set(_hcontract fast)
  endif()
  string(REPLACE ";" "," _hmarch "${UFSM_MARCH_FLAGS}")
  set(UFSM_CU_FLAGS -O3 -Xcompiler=-fno-threadsafe-statics,${_hmarch} -Xcompiler=-ffp-contract=${_hcontract} -Xfatbin=-compress-all
      "SHELL:-Xptxas -O3" --extra-device-vectorization "SHELL:--pre-include ${CMAKE_SOURCE_DIR}/src/ufsm_dev.cuh")
  set(UFSM_CU_FAST -use_fast_math)
  set(UFSM_CU_STRICT -fmad=false)
  if(UFSM_PTXAS_SPLIT GREATER 1)
    list(APPEND UFSM_CU_FLAGS "SHELL:-Xptxas --split-compile=${UFSM_PTXAS_SPLIT}")
  endif()
elseif(UFSM_CUDA_COMPILER STREQUAL "mcc")
  # mcc reads a .cu as CUDA C++ (by its suffix) and writes the device side as PTX into the object; mcudahw links and JIT-compiles
  # it through the driver at the first launch (cached in ~/.cache/mcudahw; MCUDA_PRECOMPILE=N warms the cache). --cuda-arch: the
  # PTX .target and __CUDA_ARCH__ (1200a: the block-scaled mma of the lp_* units). mcuda_device.h: what nvcc knows without a header.
  if(UFSM_ARCH STREQUAL "120")
    # nn_* too on 1200a, as the native clang / nvcc build (SASS for the 'a' target): mcudahw JITs the PTX on the GPU present, so the
    # 'a' target costs nothing there, and nn_xnet's MX-fp4 path (the block-scaled mma) failed ptxas on .target sm_120 (11-15, test_xmx)
    set(UFSM_NN_ARCH --cuda-arch=1200a)
    set(UFSM_LP_ARCH --cuda-arch=1200a)
  else()
    set(UFSM_NN_ARCH --cuda-arch=${UFSM_ARCH}0)
    set(UFSM_LP_ARCH --cuda-arch=${UFSM_ARCH}0)
  endif()
  set(UFSM_CU_FLAGS -mglibc -O3 -D_GNU_SOURCE -DUFSM_CUDA_VERSION=mcc "SHELL:-isystem ${UFSM_CUDA}/include"
      "SHELL:-include ${UFSM_CUDA}/include/mcuda_device.h"
      "SHELL:-include ${CMAKE_SOURCE_DIR}/src/ufsm_dev.cuh")   # (the logical-device map, as the clang / nvcc paths: else cudaSetDevice is unwrapped and ufsm_dev_phys an implicit extern)
  set(UFSM_CU_FAST --cuda-fast-math)   # (nvcc's -use_fast_math)
  set(UFSM_CU_STRICT -fmad=false)
  # how mcc packs each unit's PTX into its object (mcuda_pack.h): 1 MPZ, 2 MPY greedy, 3 MPY optimal; empty: mcc's default
  set(UFSM_PTX_PACK "" CACHE STRING "mcc --cuda-ptx-pack level (empty: mcc's default)")
  if(NOT UFSM_PTX_PACK STREQUAL "")
    list(APPEND UFSM_CU_FLAGS --cuda-ptx-pack=${UFSM_PTX_PACK})
  endif()
  # verify builds: each unit's packed PTX in OBJ.ptxz beside its object, named by absolute path (mcc --cuda-ptx-file), so the ~240
  # suite binaries do not carry ~50 MB each (a binary from build-verify/ runs where it was built; MCUDA_PTXDIR if the dir moved).
  # Release binaries stay self-contained.
  option(UFSM_PTX_FILE "mcc: unit PTX in sidecar files (default: on in verify builds)" ${UFSM_VERIFY})
  if(UFSM_PTX_FILE)
    list(APPEND UFSM_CU_FLAGS --cuda-ptx-file)
  endif()
endif()
function(ufsm_cuda_obj name src arch)
  add_library(obj_${name} OBJECT ${src})
  if(UFSM_CUDA_COMPILER STREQUAL "mcc")
    set_source_files_properties(${src} PROPERTIES LANGUAGE C)   # (mcc: a .cu is CUDA C++ by its suffix)
  elseif(NOT UFSM_CUDA_COMPILER STREQUAL "nvcc")
    set_source_files_properties(${src} PROPERTIES LANGUAGE CXX)
  endif()
  if((UFSM_STRICT_MATH AND name IN_LIST UFSM_STRICT_UNITS) OR name IN_LIST UFSM_EXACT_CU)
    set(fp ${UFSM_CU_STRICT})
  else()
    set(fp ${UFSM_CU_FAST})
  endif()
  target_compile_options(obj_${name} PRIVATE ${UFSM_CU_FLAGS} ${arch} ${fp})
  target_compile_definitions(obj_${name} PRIVATE ${UFSM_TYPEDEFS})
  target_include_directories(obj_${name} PRIVATE ${CMAKE_SOURCE_DIR}/src)
endfunction()
file(GLOB UFSM_NN_SRC CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/nn_*.cu)
file(GLOB UFSM_LP_SRC CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/lp_*.cu)
set(UFSM_NN_OBJS)
foreach(f ${UFSM_NN_SRC})
  cmake_path(GET f STEM n)
  ufsm_cuda_obj(${n} ${f} "${UFSM_NN_ARCH}")
  list(APPEND UFSM_NN_OBJS obj_${n})
endforeach()
set(UFSM_LP_OBJS)
foreach(f ${UFSM_LP_SRC})
  cmake_path(GET f STEM n)
  ufsm_cuda_obj(${n} ${f} "${UFSM_LP_ARCH}")
  list(APPEND UFSM_LP_OBJS obj_${n})
endforeach()
# concurrent strip-flatten chunks on one device (nvcc --default-stream per-thread)
if(UFSM_CUDA_COMPILER STREQUAL "nvcc")
  target_compile_options(obj_nn_flat PRIVATE "SHELL:--default-stream per-thread")
else()
  target_compile_definitions(obj_nn_flat PRIVATE CUDA_API_PER_THREAD_DEFAULT_STREAM=1)
endif()

# --- executables -----------------------------------------------------------------------------------------------------------
# ufsm_exe(NAME [SUITE] SOURCES... [OBJS obj... | NN | LP] [LIBS LDLIBS | CUDA | lib...] [DEFS d...] [INC CUDA | BUILD]
#          [DEPS target...] [ALL])
# OBJS: the src objects it links (NN = every nn_*.cu object, LP = every lp_*.cu object: the old nn.o / nn_fp8.o), LIBS: LDLIBS =
# libcurl libzstd blosc zlib libcrypto libjpeg libm pthread, CUDA = cudart + cublasLt, anything else as -l<name>. DEPS: built
# first (the binaries a test runs). SUITE: part of suite-bins. Not in the default target unless ALL.
set_property(GLOBAL PROPERTY UFSM_SUITE_BINS "")
function(ufsm_exe name)
  cmake_parse_arguments(A "SUITE;ALL" "" "OBJS;LIBS;DEFS;INC;DEPS" ${ARGN})
  if(A_ALL)
    add_executable(${name} ${A_UNPARSED_ARGUMENTS})
  else()
    add_executable(${name} EXCLUDE_FROM_ALL ${A_UNPARSED_ARGUMENTS})
  endif()
  target_compile_options(${name} PRIVATE ${UFSM_CFLAGS})
  target_compile_definitions(${name} PRIVATE ${UFSM_DEFS} ${A_DEFS})
  target_include_directories(${name} PRIVATE ${UFSM_INC} ${UFSM_DEPS_INCLUDE_DIRS})
  set(objs)
  set(cuda OFF)
  foreach(o ${A_OBJS})
    if(o STREQUAL "NN")
      list(APPEND objs ${UFSM_NN_OBJS})
      set(cuda ON)
    elseif(o STREQUAL "LP")
      list(APPEND objs ${UFSM_LP_OBJS})
      set(cuda ON)
    elseif(o STREQUAL "CORE")
      list(APPEND objs ${UFSM_CORE_OBJS})
    else()
      if(NOT TARGET obj_${o})
        message(FATAL_ERROR "ufsm_exe(${name}): no source for object ${o}")
      endif()
      list(APPEND objs obj_${o})
      if(o MATCHES "^(nn|lp)_")
        set(cuda ON)
      endif()
    endif()
  endforeach()
  list(APPEND objs obj_fpenv)   # FTZ / DAZ at startup in every executable (src/fpenv.c; empty under UFSM_STRICT_MATH)
  list(APPEND objs obj_optcheck)   # (every binary: the src objects call its opt_n / opt_ato*, src/optcheck.h)
  list(REMOVE_DUPLICATES objs)
  set(libs)
  foreach(l ${A_LIBS})
    if(l STREQUAL "LDLIBS")
      list(APPEND libs ufsm_ldlibs)
    elseif(l STREQUAL "CUDA")
      list(APPEND libs ufsm_cudalibs)
    else()
      list(APPEND libs ${l})
    endif()
  endforeach()
  target_link_libraries(${name} PRIVATE ${objs} ${libs})
  if(cuda AND NOT UFSM_TOOLCHAIN STREQUAL "mcc")   # (mcc links the CUDA objects as C: no C++ runtime in them)
    set_target_properties(${name} PROPERTIES LINKER_LANGUAGE CXX)
  endif()
  foreach(i ${A_INC})
    if(i STREQUAL "CUDA")
      target_include_directories(${name} SYSTEM PRIVATE ${UFSM_CUDA}/include)
    elseif(i STREQUAL "BUILD")
      target_include_directories(${name} PRIVATE ${CMAKE_BINARY_DIR})
    endif()
  endforeach()
  if(A_DEPS)
    add_dependencies(${name} ${A_DEPS})
  endif()
  if(A_SUITE)
    set_property(GLOBAL APPEND PROPERTY UFSM_SUITE_BINS ${name})
  endif()
endfunction()

ufsm_exe(ufsm ALL SUITE src/main.c src/sysinfo_cli.c OBJS CORE NN LP LIBS LDLIBS CUDA)
target_link_libraries(ufsm PRIVATE ufsm_cudalibs)   # (main.c / sysinfo_cli.c are C; the CUDA objects make it a C++ link)
include(tests/targets.cmake)

# --- suite targets ---------------------------------------------------------------------------------------------------------
get_property(_suite GLOBAL PROPERTY UFSM_SUITE_BINS)
add_custom_target(suite-bins DEPENDS ${_suite})
# the release binaries the Python CLI / sheet tests run from build/ (tests/*.py use build/ufsm): built in the release directory
# next to this one, the suite binaries in the -verify one (configured on first use with this directory's settings)
set(UFSM_TEST_PREP_REL ufsm test_sheet make_pipeline_fixture test_http_reader test_sampler_safety test_formats)
cmake_path(GET CMAKE_BINARY_DIR FILENAME _bname)
cmake_path(GET CMAKE_BINARY_DIR PARENT_PATH _bparent)
if(UFSM_VERIFY)
  string(REGEX REPLACE "-verify$" "" _rel ${_bname})
  set(UFSM_REL_DIR ${_bparent}/${_rel})
  set(UFSM_VERIFY_DIR ${CMAKE_BINARY_DIR})
else()
  set(UFSM_REL_DIR ${CMAKE_BINARY_DIR})
  set(UFSM_VERIFY_DIR ${CMAKE_BINARY_DIR}-verify)
endif()
# jobs of the nested release / verify build: min(8 (the desk rule), the cgroup's cpu.max quota, nproc); UFSM_JOBS overrides
cmake_host_system_information(RESULT _ncpu QUERY NUMBER_OF_LOGICAL_CORES)
set(_jobs 8)
if(_ncpu LESS _jobs)
  set(_jobs ${_ncpu})
endif()
if(EXISTS /sys/fs/cgroup/cpu.max)
  file(READ /sys/fs/cgroup/cpu.max _cm)
  if(_cm MATCHES "^([0-9]+) ([0-9]+)")
    math(EXPR _q "(${CMAKE_MATCH_1} + ${CMAKE_MATCH_2} - 1) / ${CMAKE_MATCH_2}")   # (ceil: a 0.5-CPU share is 1 job, not 0)
    if(_q GREATER 0 AND _q LESS _jobs)
      set(_jobs ${_q})
    endif()
  endif()
endif()
set(UFSM_JOBS ${_jobs} CACHE STRING "parallel jobs of the nested release / verify build (test-prep)")
# the sibling dir is (re)configured with this dir's settings every time (a configure of an existing dir is ~1 s), so the release
# helpers the python tests run are built like the suite binaries (a sanitizer in UFSM_EXTRA_CFLAGS, the device compiler, -march)
set(_cfg_common -G Ninja -S ${CMAKE_SOURCE_DIR} -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE} -DUFSM_ARCH=${UFSM_ARCH} -DUFSM_GPU_SM=${UFSM_GPU_SM}
    -DUFSM_CUDA=${UFSM_CUDA} -DUFSM_LLVM_RUNTIME=${UFSM_LLVM_RUNTIME} -DUFSM_CCACHE=${UFSM_CCACHE} -DUFSM_MARCH=${UFSM_MARCH}
    -DUFSM_EXTRA_CFLAGS=${UFSM_EXTRA_CFLAGS} -DUFSM_CUDA_COMPILER=${UFSM_CUDA_COMPILER} -DUFSM_CUDA_HOST_COMPILER=${UFSM_CUDA_HOST_COMPILER}
    -DUFSM_PTX_ISA=${UFSM_PTX_ISA} -DUFSM_PTXAS_SPLIT=${UFSM_PTXAS_SPLIT} -DUFSM_STRICT_MATH=${UFSM_STRICT_MATH}
    -DUFSM_PGO=${UFSM_PGO} -DUFSM_PGO_DIR=${UFSM_PGO_DIR})
if(UFSM_VERIFY)
  add_custom_target(test-prep
    COMMAND ${CMAKE_COMMAND} ${_cfg_common} -B ${UFSM_REL_DIR} -DUFSM_VERIFY=OFF
    COMMAND ${CMAKE_COMMAND} --build ${UFSM_REL_DIR} -j ${UFSM_JOBS} --target ${UFSM_TEST_PREP_REL}
    DEPENDS suite-check suite-bins USES_TERMINAL VERBATIM)
else()
  add_custom_target(test-prep
    COMMAND ${CMAKE_COMMAND} ${_cfg_common} -B ${UFSM_VERIFY_DIR} -DUFSM_VERIFY=ON
    COMMAND ${CMAKE_COMMAND} --build ${UFSM_VERIFY_DIR} -j ${UFSM_JOBS} --target suite-bins
    DEPENDS suite-check ${UFSM_TEST_PREP_REL} USES_TERMINAL VERBATIM)
endif()
# the manifest runs ./$B/test_x from the repo root: $B relative to it (a build dir outside the tree must be given as a relative path too)
cmake_path(RELATIVE_PATH UFSM_VERIFY_DIR BASE_DIRECTORY ${CMAKE_SOURCE_DIR} OUTPUT_VARIABLE _suite_b)
# test / test-slow / test-all: the tiers of tests/suite.txt, runner options from the environment: SUITE_FLAGS="-j 4 test_fit"
foreach(t fast slow all)
  if(t STREQUAL "fast")
    set(tn test)
  else()
    set(tn test-${t})
  endif()
  add_custom_target(${tn}
    COMMAND bash -c "exec tests/run_suite.sh -B '${_suite_b}' -t ${t} $SUITE_FLAGS"
    DEPENDS test-prep WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} USES_TERMINAL VERBATIM)
endforeach()
# every suite binary needs a line in tests/suite.txt and every tests/suite.txt binary a target (a merge that added only one half)
add_custom_target(suite-check
  COMMAND bash ${CMAKE_SOURCE_DIR}/cmake/suite_check.sh ${CMAKE_SOURCE_DIR} "${_suite}"
  USES_TERMINAL VERBATIM)
# jit-warm (mcc): fill mcudahw's kernel cache for this build's ufsm on the first visible GPU (scripts/toolchain/jit_warm.sh: every kernel
# the cache lacks through the driver's ptxas, cgroup CPUs at a time), so no kernel JIT-compiles inside a step. Needs a GPU
if(UFSM_TOOLCHAIN STREQUAL "mcc")
  add_custom_target(jit-warm
    COMMAND bash ${CMAKE_SOURCE_DIR}/scripts/toolchain/jit_warm.sh $<TARGET_FILE:ufsm>
    DEPENDS ufsm USES_TERMINAL VERBATIM)
endif()
# doctor: the toolchain picked, then the machine's real limits (cgroup CPUs / RAM, GPUs / MIG, CUDA, disk)
add_custom_target(doctor
  COMMAND ${CMAKE_COMMAND} -E echo "cmake: toolchain ${UFSM_TOOLCHAIN}, CC=${CMAKE_C_COMPILER} (${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}), CUDA=${UFSM_CUDA} (${UFSM_CUDA_VERSION}), ARCH=${UFSM_ARCH}, VERIFY=${UFSM_VERIFY}, runtime=$<IF:$<BOOL:${UFSM_LLVM_RUNTIME}>,compiler-rt,libgcc>+libstdc++, ccache=${UFSM_CCACHE_EXE}, build dir ${CMAKE_BINARY_DIR}"
  COMMAND ${CMAKE_COMMAND} -E echo "cmake: libs: libcurl=${UFSM_DEPS_libcurl_VERSION} libzstd=${UFSM_DEPS_libzstd_VERSION} blosc=${UFSM_DEPS_blosc_VERSION} zlib=${UFSM_DEPS_zlib_VERSION} libcrypto=${UFSM_DEPS_libcrypto_VERSION}"
  COMMAND ${CMAKE_BINARY_DIR}/ufsm sysinfo --disk ${CMAKE_SOURCE_DIR} --disk /tmp
  DEPENDS ufsm USES_TERMINAL VERBATIM)
