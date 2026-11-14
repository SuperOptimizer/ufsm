# The microc toolchain (github.com/SuperOptimizer/microc): mcc for the host C and for the CUDA units (mcc's own C++ subset ->
# PTX, kept in the objects), mcudahw for the CUDA runtime (it hands the PTX to the driver, which JIT-compiles and caches it; it
# forwards cuBLASLt to the real libcublasLt.so). No CUDA toolkit, no clang / gcc at compile time; the final link of a glibc
# program is still mcc driving the system cc -no-pie. CMakePresets.json: --preset mcc (build-mcc/), mcc-verify.
#   UFSM_MCC       the mcc binary (default: mcc on PATH, else $MCC_HOME/mcc)
#   UFSM_MCC_CUDA  mcudahw as mkcuda.sh MCC DIR hw installs it: DIR/cudahw/{include,lib64} (default: <mcc's directory>/cudahw)
set(UFSM_TOOLCHAIN mcc CACHE STRING "host / device toolchain: llvm (cmake/llvm.cmake) or mcc (this file)" FORCE)
if(NOT UFSM_MCC)
  find_program(UFSM_MCC NAMES mcc HINTS $ENV{MCC_HOME} NO_CACHE)
  if(NOT UFSM_MCC)
    message(FATAL_ERROR "mcc.cmake: no mcc found (put it on PATH, set MCC_HOME, or -DUFSM_MCC=/path/to/mcc)")
  endif()
  set(UFSM_MCC ${UFSM_MCC} CACHE FILEPATH "the mcc binary")
endif()
cmake_path(GET UFSM_MCC PARENT_PATH _mccdir)
set(UFSM_MCC_CUDA ${_mccdir}/cudahw CACHE PATH "mcudahw (mkcuda.sh MCC DIR hw): include/ and lib64/")
set(CMAKE_C_COMPILER ${UFSM_MCC})
# mcc identifies as GNU 5.1 (__GNUC__ 5) to CMake. C++ is not used by the mcc build (the .cu units are compiled as C sources by
# mcc, which reads .cu as CUDA C++); the system c++ only satisfies project(... CXX)
set(CMAKE_C_COMPILER_ID_RUN ON)
set(CMAKE_C_COMPILER_ID GNU)
set(CMAKE_C_COMPILER_VERSION 5.1.0)
set(CMAKE_C_COMPILER_FORCED ON)
set(CMAKE_C_STANDARD_COMPUTED_DEFAULT 23)
set(CMAKE_C_COMPILE_FEATURES c_std_90;c_std_99;c_std_11;c_std_17;c_std_23)
find_program(_cxx NAMES c++ g++ clang++ NO_CACHE)
set(CMAKE_CXX_COMPILER ${_cxx})
# (no archiver: ufsm builds OBJECT libraries and links them straight into each binary)
