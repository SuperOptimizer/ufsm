# ufsm: C23 host code + CUDA kernels, all compiled by LLVM (cmake/llvm.cmake: clang, clang -x cuda, lld, llvm-ar, compiler-rt,
# libunwind, libc++). Ninja. Configurations are build directories (CMakePresets.json):
#   build/            release: the fp16 / MX kernels training and inference use                (cmake --preset release)
#   build-verify/     UFSM_VERIFY=ON: + the bf16 / fp32 tensor-core instantiations; the suite runs from here (--preset verify)
#   build86/, build86-verify/, build86p/   UFSM_ARCH=86: Ampere, the lp_* kernels with their software MX paths; build86p: SASS + PTX
# Targets: ufsm (default), suite-bins, test / test-slow / test-all (tests/run_suite.sh, runner options in the SUITE_FLAGS env var),
# suite-check, doctor, every test binary by name (cmake --build build-verify --target test_lat). The binaries land in the build
# directory itself (build/ufsm, build-verify/test_lat), as with the old Makefile.
cmake_minimum_required(VERSION 3.28)
if(NOT CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/llvm.cmake)
endif()
project(ufsm LANGUAGES C CXX)

option(UFSM_VERIFY "bf16 / fp32 tensor-core instantiations as well (-DUFSM_ALL_TYPES; the suite build)" OFF)
set(UFSM_ARCH 120 CACHE STRING "GPU target: 120 (GeForce / RTX PRO Blackwell, lp_* on sm_120a) or 86 (Ampere, software MX)")
option(UFSM_LLVM_RUNTIME "compiler-rt builtins and crt files instead of libgcc.a / gcc crtbegin" ON)
option(UFSM_CCACHE "compile through ccache when it is installed" ON)
set(UFSM_CUDA "" CACHE PATH "CUDA toolkit root (default: /usr/local/cuda, else the newest /usr/local/cuda-*, else nvcc's prefix)")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)   # (the flags below are explicit; Release only keeps CMake from adding its own)
endif()
set(CMAKE_C_FLAGS_RELEASE "")
set(CMAKE_CXX_FLAGS_RELEASE "")

# --- toolchain -------------------------------------------------------------------------------------------------------------
if(NOT CMAKE_C_COMPILER_ID STREQUAL "Clang")
  message(FATAL_ERROR "ufsm builds with clang only (cmake/llvm.cmake); got ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER}")
endif()
if(NOT UFSM_CUDA)
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
execute_process(COMMAND ${UFSM_CUDA}/bin/ptxas --version OUTPUT_VARIABLE _pv ERROR_QUIET)
string(REGEX MATCH "release ([0-9.]+), V([0-9.]+)" _m "${_pv}")
set(UFSM_CUDA_VERSION ${CMAKE_MATCH_2})
if(NOT UFSM_CUDA_VERSION)
  message(FATAL_ERROR "${UFSM_CUDA}/bin/ptxas --version: no version (${_pv})")
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
# Host C. -ffp-contract=off: gcc -std=c23 (ISO mode) never contracted a*b+c into an FMA, clang's default (on) would, and the
# byte-exact host references (volcomp, count-field, the GPU-vs-CPU tests) are written against uncontracted arithmetic.
# -fno-math-errno: value-neutral (only errno), lets sqrtf / fabsf vectorise without the errno call-out (sample.c relied on a
# gcc-only optimize("no-math-errno") attribute for that).
set(UFSM_CFLAGS -std=c23 -O3 -march=native -g -Wall -Wextra -Werror=comment -Wno-unused-parameter -pthread
    -ffp-contract=off -fno-math-errno)
set(UFSM_DEFS _GNU_SOURCE)
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
if(UFSM_ARCH STREQUAL "120")
  set(UFSM_NN_ARCH --cuda-gpu-arch=sm_120)
  set(UFSM_LP_ARCH --cuda-gpu-arch=sm_120a --no-cuda-include-ptx=sm_120a)
else()
  # Ampere: SASS + PTX for sm_${UFSM_ARCH}; a newer GPU JIT-compiles the PTX, so the sm_86 paths run on Blackwell too (the old PTX=1)
  set(UFSM_NN_ARCH --cuda-gpu-arch=sm_${UFSM_ARCH})
  set(UFSM_LP_ARCH --cuda-gpu-arch=sm_${UFSM_ARCH})
endif()
# The CUDA 13 headers clang does not find itself: CCCL (cub / thrust / cuda::std) moved to include/cccl (nvcc adds it).
# clang_cuda_compat.cuh: nvcc's host min / max overloads. --ptxas-path: the PTX ISA version bump (cmake/ptxas.sh.in).
# -D_GNU_SOURCE: g++ (nvcc's host compiler) always defines it (CLOCK_MONOTONIC in nn_aug.cu). UFSM_CUDA_VERSION keys ccache on
# the toolkit (ptxas is not hashed).
set(UFSM_PTX_ISA 9.1 CACHE STRING "PTX ISA version the CUDA units are assembled as (cmake/ptxas.sh.in)")
configure_file(cmake/ptxas.sh.in ${CMAKE_BINARY_DIR}/ptxas @ONLY FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ
               GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
set(UFSM_CU_FLAGS -x cuda --cuda-path=${UFSM_CUDA} --ptxas-path=${CMAKE_BINARY_DIR}/ptxas -Wno-unknown-cuda-version -O3 -std=c++17
    -fno-threadsafe-statics ${UFSM_CXX_STDLIB} -Xcuda-fatbinary -compress-all -isystem ${UFSM_CUDA}/include/cccl
    -include ${CMAKE_SOURCE_DIR}/src/clang_cuda_compat.cuh -D_GNU_SOURCE -DUFSM_CUDA_VERSION=${UFSM_CUDA_VERSION})
# nvcc -use_fast_math on the device side only: flush denormals, approximate division / sqrt / transcendentals, FMA contraction.
# NaN / inf stay honoured (no -ffinite-math-only: the NaN guards in the loss and fwd_nan paths must survive).
set(UFSM_CU_FAST -fgpu-flush-denormals-to-zero -fgpu-approx-transcendentals -Xarch_device -fapprox-func
    -Xarch_device -ffp-contract=fast)
# byte-identical to their CPU versions: IEEE division / sqrt, no FMA contraction, no flush (nvcc -fmad=false, no fast math)
set(UFSM_CU_STRICT -Xarch_device -ffp-contract=off)
set(UFSM_STRICT_UNITS nn_gpudec nn_cf nn_ftrace nn_latvol nn_band nn_fibre nn_inkfly nn_gpuenc nn_mcodes nn_faces nn_render)
set(UFSM_TYPEDEFS UFSM_ARCH=${UFSM_ARCH})
if(UFSM_VERIFY)
  list(APPEND UFSM_TYPEDEFS UFSM_ALL_TYPES)
endif()

# --- objects: one OBJECT library per source (obj_<name>), so a test links exactly the objects it lists ------------------------
add_library(ufsm_ldlibs INTERFACE)
target_link_libraries(ufsm_ldlibs INTERFACE PkgConfig::UFSM_DEPS jpeg m pthread)
add_library(ufsm_cudalibs INTERFACE)
target_link_directories(ufsm_cudalibs INTERFACE ${UFSM_CUDA}/lib64)
target_link_libraries(ufsm_cudalibs INTERFACE cudart cublasLt stdc++)
target_link_options(ufsm_cudalibs INTERFACE -Wl,-rpath,${UFSM_CUDA}/lib64)

function(ufsm_host_obj name src)
  add_library(obj_${name} OBJECT ${src})
  target_compile_options(obj_${name} PRIVATE ${UFSM_CFLAGS} ${ARGN})
  target_compile_definitions(obj_${name} PRIVATE ${UFSM_DEFS})
  target_include_directories(obj_${name} PRIVATE ${UFSM_INC} ${UFSM_DEPS_INCLUDE_DIRS})
  target_include_directories(obj_${name} SYSTEM PRIVATE ${UFSM_CUDA}/include)
endfunction()

file(GLOB UFSM_C_SRC CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/*.c)
set(UFSM_CORE_OBJS)
foreach(f ${UFSM_C_SRC})
  cmake_path(GET f STEM n)
  ufsm_host_obj(${n} ${f})
  if(NOT n MATCHES "^(main|sysinfo_cli)$")
    list(APPEND UFSM_CORE_OBJS obj_${n})
  endif()
endforeach()
ufsm_host_obj(surfcomp ${CMAKE_SOURCE_DIR}/third_party/surfcomp/surfcomp.c -w)
list(APPEND UFSM_CORE_OBJS obj_surfcomp)

function(ufsm_cuda_obj name src arch)
  add_library(obj_${name} OBJECT ${src})
  set_source_files_properties(${src} PROPERTIES LANGUAGE CXX)
  if(name IN_LIST UFSM_STRICT_UNITS)
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
target_compile_definitions(obj_nn_flat PRIVATE CUDA_API_PER_THREAD_DEFAULT_STREAM=1)

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
  if(cuda)
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
    math(EXPR _q "${CMAKE_MATCH_1} / ${CMAKE_MATCH_2}")
    if(_q GREATER 0 AND _q LESS _jobs)
      set(_jobs ${_q})
    endif()
  endif()
endif()
set(UFSM_JOBS ${_jobs} CACHE STRING "parallel jobs of the nested release / verify build (test-prep)")
set(_cfg_common -G Ninja -S ${CMAKE_SOURCE_DIR} -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE} -DUFSM_ARCH=${UFSM_ARCH}
    -DUFSM_CUDA=${UFSM_CUDA} -DUFSM_LLVM_RUNTIME=${UFSM_LLVM_RUNTIME} -DUFSM_CCACHE=${UFSM_CCACHE})
if(UFSM_VERIFY)
  add_custom_target(test-prep
    COMMAND test -f ${UFSM_REL_DIR}/build.ninja || ${CMAKE_COMMAND} ${_cfg_common} -B ${UFSM_REL_DIR} -DUFSM_VERIFY=OFF
    COMMAND ${CMAKE_COMMAND} --build ${UFSM_REL_DIR} -j ${UFSM_JOBS} --target ${UFSM_TEST_PREP_REL}
    DEPENDS suite-check suite-bins USES_TERMINAL VERBATIM)
else()
  add_custom_target(test-prep
    COMMAND test -f ${UFSM_VERIFY_DIR}/build.ninja || ${CMAKE_COMMAND} ${_cfg_common} -B ${UFSM_VERIFY_DIR} -DUFSM_VERIFY=ON
    COMMAND ${CMAKE_COMMAND} --build ${UFSM_VERIFY_DIR} -j ${UFSM_JOBS} --target suite-bins
    DEPENDS suite-check ${UFSM_TEST_PREP_REL} USES_TERMINAL VERBATIM)
endif()
# test / test-slow / test-all: the tiers of tests/suite.txt, runner options from the environment: SUITE_FLAGS="-j 4 test_fit"
foreach(t fast slow all)
  if(t STREQUAL "fast")
    set(tn test)
  else()
    set(tn test-${t})
  endif()
  add_custom_target(${tn}
    COMMAND bash -c "exec tests/run_suite.sh -B '${UFSM_VERIFY_DIR}' -t ${t} $SUITE_FLAGS"
    DEPENDS test-prep WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} USES_TERMINAL VERBATIM)
endforeach()
# every suite binary needs a line in tests/suite.txt and every tests/suite.txt binary a target (a merge that added only one half)
add_custom_target(suite-check
  COMMAND bash ${CMAKE_SOURCE_DIR}/cmake/suite_check.sh ${CMAKE_SOURCE_DIR} "${_suite}"
  USES_TERMINAL VERBATIM)
# doctor: the toolchain picked, then the machine's real limits (cgroup CPUs / RAM, GPUs / MIG, CUDA, disk)
add_custom_target(doctor
  COMMAND ${CMAKE_COMMAND} -E echo "cmake: CC=${CMAKE_C_COMPILER} (clang ${CMAKE_C_COMPILER_VERSION}), CUDA=${UFSM_CUDA} (${UFSM_CUDA_VERSION}), ARCH=${UFSM_ARCH}, VERIFY=${UFSM_VERIFY}, runtime=$<IF:$<BOOL:${UFSM_LLVM_RUNTIME}>,compiler-rt,libgcc>+libstdc++, ccache=${UFSM_CCACHE_EXE}, build dir ${CMAKE_BINARY_DIR}"
  COMMAND ${CMAKE_COMMAND} -E echo "cmake: libs: libcurl=${UFSM_DEPS_libcurl_VERSION} libzstd=${UFSM_DEPS_libzstd_VERSION} blosc=${UFSM_DEPS_blosc_VERSION} zlib=${UFSM_DEPS_zlib_VERSION} libcrypto=${UFSM_DEPS_libcrypto_VERSION}"
  COMMAND ${CMAKE_BINARY_DIR}/ufsm sysinfo --disk ${CMAKE_SOURCE_DIR} --disk /tmp
  DEPENDS ufsm USES_TERMINAL VERBATIM)
