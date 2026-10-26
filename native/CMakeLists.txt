# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

cmake_minimum_required(VERSION 3.23...4.4)

# scikit-build-core passes the repository's version (shared with solprop); a direct
# CMake configure, such as a preset's, has none and uses 0.0.0.
if(DEFINED SKBUILD_PROJECT_VERSION)
  set(SOLPROP_NATIVE_PROJECT_VERSION "${SKBUILD_PROJECT_VERSION}")
else()
  set(SOLPROP_NATIVE_PROJECT_VERSION "0.0.0")
endif()
project(solprop_native VERSION ${SOLPROP_NATIVE_PROJECT_VERSION} LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# No C++20 modules: without this, CMake adds GCC's module-scanning flags (GCC 14 and
# newer), which clang-tidy's parser does not accept.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# These options must be visible before the compiler-family guard below.  In
# particular, HIP uses ROCm Clang and CUDA's nvcc_wrapper uses GNU C++.
option(SOLPROP_NATIVE_ENABLE_CUDA "Use CUDA in Kokkos and cuFFT" OFF)
option(SOLPROP_NATIVE_ENABLE_HIP "Use HIP in Kokkos and rocFFT" OFF)
option(SOLPROP_NATIVE_ENABLE_COVERAGE
       "Instrument project CPU sources with Clang source-based atomic coverage" OFF)
set(SOLPROP_NATIVE_COVERAGE_SOURCE_HASH "")
if(SOLPROP_NATIVE_ENABLE_COVERAGE)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "^(Clang|AppleClang)$" OR
     SOLPROP_NATIVE_ENABLE_CUDA OR SOLPROP_NATIVE_ENABLE_HIP OR WIN32)
    message(FATAL_ERROR "Coverage requires a separate Clang CPU build on Linux or macOS")
  endif()
  # A separate O1 configuration keeps branch maps useful without changing the
  # production build. Atomic counters are required for OpenMP kernel execution.
  set(SOLPROP_NATIVE_COVERAGE_FLAGS
      -O1 -g -fprofile-instr-generate -fcoverage-mapping -fprofile-update=atomic)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE OFF)
  file(GLOB_RECURSE _coverage_sources CONFIGURE_DEPENDS RELATIVE "${PROJECT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}/src/*.cpp" "${PROJECT_SOURCE_DIR}/include/*.hpp")
  list(SORT _coverage_sources)
  set(_coverage_source_record "")
  foreach(_source IN LISTS _coverage_sources)
    file(SHA256 "${PROJECT_SOURCE_DIR}/${_source}" _source_hash)
    string(APPEND _coverage_source_record "${_source}:${_source_hash}\n")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
      "${PROJECT_SOURCE_DIR}/${_source}")
  endforeach()
  string(SHA256 SOLPROP_NATIVE_COVERAGE_SOURCE_HASH "${_coverage_source_record}")
endif()

if(APPLE)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "^(AppleClang|Clang)$")
    message(FATAL_ERROR
      "The macOS build requires Clang (Homebrew LLVM is recommended); "
      "found ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}.")
  endif()
  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND
     CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14)
    message(FATAL_ERROR
      "The macOS build requires Clang 14 or newer; "
      "found ${CMAKE_CXX_COMPILER_VERSION}.")
  endif()
elseif(SOLPROP_NATIVE_ENABLE_HIP AND
       NOT CMAKE_CXX_COMPILER_ID MATCHES "^(Clang|AMDClang)$")
  message(FATAL_ERROR
    "The HIP validation build requires the ROCm Clang compiler; "
    "found ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}.")
elseif(SOLPROP_NATIVE_ENABLE_CUDA AND
       NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  message(FATAL_ERROR
    "The NVIDIA validation build requires GNU C++; "
    "found ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}.")
elseif(NOT SOLPROP_NATIVE_ENABLE_HIP AND
       NOT CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|IntelLLVM|Clang|AMDClang)$")
  message(FATAL_ERROR "Linux CPU builds require GNU C++, IntelLLVM, or Clang/AOCC.")
elseif(SOLPROP_NATIVE_ENABLE_CUDA AND
       CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14)
  # The CPU reference compiler below, for the host pass of nvcc_wrapper too: GCC 13
  # also refuses the device build under -Werror (a false maybe-uninitialized report).
  # GCC 14 needs CUDA 12.8 or newer, checked where nvcc is found.
  message(FATAL_ERROR
    "CUDA builds require GCC 14 or newer as the host compiler; found "
    "${CMAKE_CXX_COMPILER_VERSION}. On Ubuntu 24.04 install g++-14 and build with "
    "CXX=g++-14 NVCC_WRAPPER_DEFAULT_COMPILER=g++-14.")
elseif(NOT SOLPROP_NATIVE_ENABLE_CUDA AND
       CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
       CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14)
  # GCC 13 compiles the out-of-line strict helpers (include/solprop_native/fp.hpp)
  # through the stack, stalling every call on a failed store-to-load forward: coupled
  # Multistep steps took 1.25 to 1.45 times as long as with GCC 14 (Cascade Lake,
  # Zen 5).
  message(FATAL_ERROR
    "Linux CPU builds require GCC 14 or newer; found ${CMAKE_CXX_COMPILER_VERSION}. "
    "On Ubuntu 24.04 or Debian, install it with 'sudo apt install g++-14' and build "
    "with CXX=g++-14 (a CMake cache that recorded another compiler must be removed).")
endif()

if(NOT APPLE AND NOT SOLPROP_NATIVE_ENABLE_CUDA AND
   NOT SOLPROP_NATIVE_ENABLE_HIP AND
   CMAKE_CXX_COMPILER_ID MATCHES "^(Clang|AMDClang)$")
  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND
     CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14)
    message(FATAL_ERROR "Linux Clang CPU builds require Clang 14 or newer.")
  endif()
  # Vendor compiler defaults must not permit reassociation of the error estimate.
  add_compile_options(-fno-fast-math)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 2022)
    message(FATAL_ERROR "Kokkos CPU builds require IntelLLVM 2022 or newer.")
  endif()
  # icpx defaults to fast floating-point transformations. The adaptive error
  # estimate requires precise arithmetic in project and dependency templates.
  add_compile_options(-fp-model=precise)
  # CMake can find the distribution's older LLVM archiver even when icpx is
  # newer. LTO objects require the archive tools shipped with that compiler.
  foreach(_tool IN ITEMS ar ranlib)
    execute_process(COMMAND "${CMAKE_CXX_COMPILER}" "-print-prog-name=llvm-${_tool}"
      OUTPUT_VARIABLE _intel_tool OUTPUT_STRIP_TRAILING_WHITESPACE
      COMMAND_ERROR_IS_FATAL ANY)
    if(NOT EXISTS "${_intel_tool}")
      message(FATAL_ERROR "icpx could not locate its llvm-${_tool}: ${_intel_tool}")
    endif()
    string(TOUPPER "${_tool}" _tool_upper)
    set(CMAKE_CXX_COMPILER_${_tool_upper} "${_intel_tool}" CACHE FILEPATH
        "Archive tool matching IntelLLVM bitcode" FORCE)
    set(CMAKE_CXX_COMPILER_${_tool_upper} "${_intel_tool}")
  endforeach()
  # CMake 3.28 expands these paths during project(), before the correction above.
  set(CMAKE_CXX_ARCHIVE_CREATE_IPO
      "\"${CMAKE_CXX_COMPILER_AR}\" cr <TARGET> <LINK_FLAGS> <OBJECTS>")
  set(CMAKE_CXX_ARCHIVE_APPEND_IPO
      "\"${CMAKE_CXX_COMPILER_AR}\" r <TARGET> <LINK_FLAGS> <OBJECTS>")
  set(CMAKE_CXX_ARCHIVE_FINISH_IPO
      "\"${CMAKE_CXX_COMPILER_RANLIB}\" <TARGET>")
endif()

# solprop SEP 25: floating-point contraction (fused multiply-add) is on by default for
# CPU builds, and the few protected blocks opt out (include/solprop_native/fp.hpp).
# Clang's plain `-ffp-contract=fast` ignores `#pragma clang fp contract(off)` and
# `#pragma STDC FP_CONTRACT OFF`; `fast-honor-pragmas` contracts across statements and
# keeps the protected blocks strict. GCC ignores both pragmas: its protected functions
# carry `optimize("fp-contract=off")` instead. icpx is Clang-based and takes the Clang
# form after `-fp-model=precise`, which alone contracts within a statement only. The
# flag is added after `-fno-fast-math` and `-fp-model=precise`, so neither resets it.
# Device builds keep their compilers' defaults (nvcc and HIP Clang already contract).
set(SOLPROP_NATIVE_FP_CONTRACT_FLAG "")
if(NOT SOLPROP_NATIVE_ENABLE_CUDA AND NOT SOLPROP_NATIVE_ENABLE_HIP)
  if(CMAKE_CXX_COMPILER_ID MATCHES "^(Clang|AppleClang|AMDClang|IntelLLVM)$")
    set(SOLPROP_NATIVE_FP_CONTRACT_FLAG "-ffp-contract=fast-honor-pragmas")
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(SOLPROP_NATIVE_FP_CONTRACT_FLAG "-ffp-contract=fast")
  endif()
endif()
if(SOLPROP_NATIVE_FP_CONTRACT_FLAG)
  add_compile_options("${SOLPROP_NATIVE_FP_CONTRACT_FLAG}")
  # icpx reports the deliberate override of `-fp-model=precise`'s contraction as a
  # warning, which `-Werror` would turn into a failed build.
  if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    add_compile_options(-Wno-overriding-option)
  endif()
endif()

if(WIN32)
  # The C runtime's "use the _s variant" deprecations (getenv) and windows.h's min and
  # max macros, which break std::min and std::max.
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
  # Kokkos Kernels adds clang-cl's /EHsc for any Clang on the MSVC ABI. The GNU-style
  # clang++ driver this build uses rejects it, and enables C++ exceptions by default.
  if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU")
    set(KK_WINDOWS_EHSC_DONE TRUE)
  endif()
endif()

include(FetchContent)

find_package(Python 3.14 EXACT REQUIRED COMPONENTS Interpreter Development.Module)

# Query the nanobind package installed in the active uv environment. This is
# reliable for both scikit-build-core and direct CMake invocations.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_VARIABLE nanobind_ROOT
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
find_package(nanobind CONFIG REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
          "import importlib.metadata; print(importlib.metadata.version('nanobind'))"
  OUTPUT_VARIABLE SOLPROP_NATIVE_NANOBIND_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)

execute_process(
  COMMAND git rev-parse HEAD
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  OUTPUT_VARIABLE SOLPROP_NATIVE_BACKEND_COMMIT
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
execute_process(
  COMMAND git diff --quiet --ignore-submodules HEAD
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  RESULT_VARIABLE SOLPROP_NATIVE_TREE_DIRTY
)
if(SOLPROP_NATIVE_TREE_DIRTY)
  string(APPEND SOLPROP_NATIVE_BACKEND_COMMIT "+dirty")
endif()
# solprop and solprop-native are released together from one repository at one
# version, so the solprop version this build belongs to is its own. The record is
# provenance: check_build() compares the kernel-contract schema and looks the driver
# members it relies on up by name, which still catches mismatched halves (an editable
# checkout beside a stale extension, a local device build).
if(DEFINED SKBUILD_PROJECT_VERSION_FULL)
  set(SOLPROP_NATIVE_SOLPROP_BUILT_AGAINST "${SKBUILD_PROJECT_VERSION_FULL}")
else()
  set(SOLPROP_NATIVE_SOLPROP_BUILT_AGAINST "unknown")
endif()

# On Homebrew, FFTW is keg-only. Linux package managers install it in a normal
# system prefix, so no platform-specific value is needed there.
if(APPLE AND NOT FFTW_ROOT)
  find_path(SOLPROP_NATIVE_FFTW_ROOT include/fftw3.h
            PATHS /opt/homebrew/opt/fftw /usr/local/opt/fftw
            NO_DEFAULT_PATH)
  if(SOLPROP_NATIVE_FFTW_ROOT)
    set(FFTW_ROOT "${SOLPROP_NATIVE_FFTW_ROOT}" CACHE PATH
        "Homebrew FFTW installation")
  endif()
endif()

# Kokkos FFT maps the OpenMP host execution space to FFTW's threaded library and
# plans with the concurrency of the execution-space instance it is given. The Python
# module initialises Kokkos with SOLPROP_NATIVE_THREADS within the process's thread
# budget (solprop ADR-052: the affinity mask, the cgroup quota and SOLPROP_THREADS) and
# sizes the kernel, FFT and Raman FFT instances from the optional
# SOLPROP_NATIVE_*_THREADS pools, all read once at import; OMP_NUM_THREADS does not
# set these counts.
option(SOLPROP_NATIVE_ENABLE_OPENMP "Use OpenMP in Kokkos and FFTW" ON)
# Kokkos Threads (C++ threads) is the host space of a build that must add no OpenMP
# runtime to the process: a fallback for Windows builds (solprop SEP 29). FFTW then uses its own
# threads library, Win32 threads on Windows and POSIX threads elsewhere. It replaces
# OpenMP, so it needs SOLPROP_NATIVE_ENABLE_OPENMP=OFF.
option(SOLPROP_NATIVE_ENABLE_THREADS "Use Kokkos Threads and FFTW's threads library" OFF)
if(SOLPROP_NATIVE_ENABLE_THREADS AND SOLPROP_NATIVE_ENABLE_OPENMP)
  message(FATAL_ERROR
    "SOLPROP_NATIVE_ENABLE_THREADS replaces OpenMP; also set "
    "SOLPROP_NATIVE_ENABLE_OPENMP=OFF.")
endif()
# Three build profiles. LOCAL (the default) is the fast build for the machine it is
# built on: a vendor BLAS, that machine's own architecture and checked IPO.
# DISTRIBUTION is the build for wheels that run on other machines: a documented CPU
# baseline (x86-64-v3 on x86-64, apple-m1 on Apple silicon) and a BLAS every such
# machine has (Accelerate on macOS; on Linux and in the Windows wheel an OpenMP
# OpenBLAS on the extension's own runtime, else on Windows the scipy-openblas32
# OpenBLAS; each selects its kernels at run time and is carried in the wheel).
# REFERENCE is the gate's tidy and sanitizer build: no vendor BLAS and no
# machine-specific code.
# cmake/HostLibraries.cmake resolves the AUTO values below and reports the choice.
set(SOLPROP_NATIVE_PROFILE "LOCAL" CACHE STRING
    "Build profile: LOCAL, DISTRIBUTION or REFERENCE")
set_property(CACHE SOLPROP_NATIVE_PROFILE PROPERTY STRINGS LOCAL DISTRIBUTION REFERENCE)
set(SOLPROP_NATIVE_HOST_ARCH "AUTO" CACHE STRING
    "Optimize CPU code for the build machine: AUTO (ON for a LOCAL CPU build), ON or OFF")
set_property(CACHE SOLPROP_NATIVE_HOST_ARCH PROPERTY STRINGS AUTO ON OFF)
set(SOLPROP_NATIVE_CPU_BASELINE "AUTO" CACHE STRING
    "Minimum CPU of a non-native build: AUTO, NONE, or an -march (x86-64) or -mcpu (Arm) value")
option(SOLPROP_NATIVE_ENABLE_AGGRESSIVE_VECTORIZATION
       "Assert independent Kokkos loop iterations to the vectorizer" ON)
option(SOLPROP_NATIVE_ENABLE_IPO
       "Enable checked interprocedural/link-time optimization for CPU builds" ON)
option(SOLPROP_NATIVE_WARNINGS_AS_ERRORS
       "Treat diagnostics in project-owned C++ sources as errors" ON)
option(SOLPROP_NATIVE_ENABLE_CLANG_TIDY
       "Run clang-tidy while compiling project-owned C++ sources" OFF)
option(SOLPROP_NATIVE_ENABLE_SANITIZERS
       "Instrument project targets with AddressSanitizer and UBSan" OFF)
option(SOLPROP_NATIVE_BUILD_BENCHMARKS
       "Build local native microbenchmarks (never installed)" OFF)
set(SOLPROP_NATIVE_GEMM_PROVIDER "HYBRID" CACHE STRING
    "Dense transform provider: CUSTOM, KOKKOSBLAS_NATIVE, KOKKOSBLAS_TPL, or HYBRID")
set_property(CACHE SOLPROP_NATIVE_GEMM_PROVIDER PROPERTY STRINGS
             CUSTOM KOKKOSBLAS_NATIVE KOKKOSBLAS_TPL HYBRID)
set(SOLPROP_NATIVE_ENABLE_TPL_BLAS "AUTO" CACHE STRING
    "Dispatch compatible GEMMs to vendor BLAS: AUTO (the profile's BLAS), ON or OFF")
set_property(CACHE SOLPROP_NATIVE_ENABLE_TPL_BLAS PROPERTY STRINGS AUTO ON OFF)
set(SOLPROP_NATIVE_BLAS_ROOT "" CACHE PATH
    "Optional prefix containing the local BLAS provider")
set(SOLPROP_NATIVE_BLAS_LIBRARIES "" CACHE STRING
    "Optional explicit BLAS libraries for a LOCAL KokkosKernels build")
# The automatic BLAS search below looks here first.
if(SOLPROP_NATIVE_BLAS_ROOT)
  list(PREPEND CMAKE_PREFIX_PATH "${SOLPROP_NATIVE_BLAS_ROOT}")
endif()
include(cmake/HostLibraries.cmake)
if(SOLPROP_NATIVE_CPU_BASELINE_FLAG)
  # Every target from here on, the fetched Kokkos libraries included, is compiled and
  # link-time optimised for the baseline.
  add_compile_options("${SOLPROP_NATIVE_CPU_BASELINE_FLAG}")
  add_link_options("${SOLPROP_NATIVE_CPU_BASELINE_FLAG}")
endif()
set(SOLPROP_NATIVE_CUDA_ARCH "HOPPER90" CACHE STRING
    "Kokkos CUDA architecture suffix, e.g. AMPERE80, ADA89 or HOPPER90")
set(SOLPROP_NATIVE_HIP_ARCH "AMD_GFX942" CACHE STRING
    "Kokkos HIP architecture suffix, e.g. AMD_GFX90A or AMD_GFX942")
set(SOLPROP_NATIVE_DEVICE_TEAM_SIZE "256" CACHE STRING
    "Maximum accelerator response and scan team width")
set(SOLPROP_NATIVE_DEVICE_REDUCTION_SIZE "512" CACHE STRING
    "Fixed accelerator deterministic reduction block width")
# solprop SEP 18's band rule (include/solprop_native/plasma.hpp). The staging changes
# only how a device team reads a filter's halo, through the same arithmetic (Q8). Its
# default is the faster read measured on each vendor's device: on an MI325X (HIP)
# staging the row in team scratch, 1.25 times the trapezoid's RDW window against 1.36
# reading the halo directly; on an H200 (CUDA) direct reads, 1.10 against 1.21. Every
# build runs the band rule's team column (solprop SEP 25).
if(SOLPROP_NATIVE_ENABLE_HIP)
  set(SOLPROP_NATIVE_BAND_STAGING_DEFAULT ON)
else()
  set(SOLPROP_NATIVE_BAND_STAGING_DEFAULT OFF)
endif()
option(SOLPROP_NATIVE_BAND_STAGING
       "Stage the band rule's filter rows in team scratch (solprop SEP 18 Q8)"
       ${SOLPROP_NATIVE_BAND_STAGING_DEFAULT})

# The precision policies the native core instantiates (solprop SEP 36 section 6.3,
# include/solprop_native/precision.hpp), each in its own translation unit. Binary64 is
# the default arithmetic, so every build instantiates `double`. `single` is built on
# CPU builds by default; a device build's binary32 paths are qualified in SEP 36's
# phase 4, so it builds `double` alone until then. `mixed` is refused until it is built.
if(SOLPROP_NATIVE_ENABLE_CUDA OR SOLPROP_NATIVE_ENABLE_HIP)
  set(SOLPROP_NATIVE_PRECISIONS_DEFAULT "double")
else()
  set(SOLPROP_NATIVE_PRECISIONS_DEFAULT "double;single")
endif()
set(SOLPROP_NATIVE_PRECISIONS "${SOLPROP_NATIVE_PRECISIONS_DEFAULT}" CACHE STRING
    "Semicolon-separated precision policies to instantiate: double (required), single")
set(SOLPROP_NATIVE_PRECISION_DEFINITIONS)
set(SOLPROP_NATIVE_PRECISION_SOURCES)
foreach(SOLPROP_NATIVE_PRECISION IN LISTS SOLPROP_NATIVE_PRECISIONS)
  if(SOLPROP_NATIVE_PRECISION STREQUAL "double")
    list(APPEND SOLPROP_NATIVE_PRECISION_DEFINITIONS SOLPROP_NATIVE_PRECISION_DOUBLE=1)
  elseif(SOLPROP_NATIVE_PRECISION STREQUAL "single")
    if(SOLPROP_NATIVE_ENABLE_CUDA OR SOLPROP_NATIVE_ENABLE_HIP)
      message(FATAL_ERROR
        "SOLPROP_NATIVE_PRECISIONS: the single precision policy runs on CPU builds "
        "only until its device paths are qualified (solprop SEP 36 phase 4); use "
        "\"double\".")
    endif()
    list(APPEND SOLPROP_NATIVE_PRECISION_DEFINITIONS SOLPROP_NATIVE_PRECISION_SINGLE=1)
    list(APPEND SOLPROP_NATIVE_PRECISION_SOURCES src/general_single.cpp)
  elseif(SOLPROP_NATIVE_PRECISION STREQUAL "mixed")
    message(FATAL_ERROR
      "SOLPROP_NATIVE_PRECISIONS: the mixed precision policy is not built yet "
      "(solprop SEP 36); use \"double\" and \"single\".")
  else()
    message(FATAL_ERROR
      "SOLPROP_NATIVE_PRECISIONS must list double, single or mixed; got "
      "\"${SOLPROP_NATIVE_PRECISION}\".")
  endif()
endforeach()
list(REMOVE_DUPLICATES SOLPROP_NATIVE_PRECISION_DEFINITIONS)
list(REMOVE_DUPLICATES SOLPROP_NATIVE_PRECISION_SOURCES)
if(NOT "SOLPROP_NATIVE_PRECISION_DOUBLE=1" IN_LIST SOLPROP_NATIVE_PRECISION_DEFINITIONS)
  message(FATAL_ERROR
    "SOLPROP_NATIVE_PRECISIONS must include double, the default arithmetic; got "
    "\"${SOLPROP_NATIVE_PRECISIONS}\".")
endif()

foreach(SOLPROP_NATIVE_DEVICE_SIZE IN ITEMS
        SOLPROP_NATIVE_DEVICE_TEAM_SIZE SOLPROP_NATIVE_DEVICE_REDUCTION_SIZE)
  if(NOT "${${SOLPROP_NATIVE_DEVICE_SIZE}}" MATCHES
         "^(64|128|256|512|1024)$")
    message(FATAL_ERROR
      "${SOLPROP_NATIVE_DEVICE_SIZE} must be one of 64, 128, 256, 512, or 1024; "
      "got ${${SOLPROP_NATIVE_DEVICE_SIZE}}.")
  endif()
endforeach()

string(TOUPPER "${SOLPROP_NATIVE_GEMM_PROVIDER}"
       SOLPROP_NATIVE_GEMM_PROVIDER)
set(SOLPROP_NATIVE_GEMM_PROVIDERS
    CUSTOM KOKKOSBLAS_NATIVE KOKKOSBLAS_TPL HYBRID)
if(NOT SOLPROP_NATIVE_GEMM_PROVIDER IN_LIST SOLPROP_NATIVE_GEMM_PROVIDERS)
  message(FATAL_ERROR
    "SOLPROP_NATIVE_GEMM_PROVIDER must be CUSTOM, KOKKOSBLAS_NATIVE, "
    "KOKKOSBLAS_TPL, or HYBRID; got ${SOLPROP_NATIVE_GEMM_PROVIDER}.")
endif()
if(SOLPROP_NATIVE_GEMM_PROVIDER STREQUAL "KOKKOSBLAS_TPL" AND
   NOT SOLPROP_NATIVE_ENABLE_TPL_BLAS)
  message(FATAL_ERROR
    "KOKKOSBLAS_TPL requires SOLPROP_NATIVE_ENABLE_TPL_BLAS=ON.")
endif()
if(SOLPROP_NATIVE_ENABLE_CUDA AND SOLPROP_NATIVE_ENABLE_HIP)
  message(FATAL_ERROR
    "CUDA and HIP are separate build variants and cannot both be enabled.")
endif()
if((SOLPROP_NATIVE_ENABLE_OPENMP OR SOLPROP_NATIVE_ENABLE_THREADS) AND
   (SOLPROP_NATIVE_ENABLE_CUDA OR SOLPROP_NATIVE_ENABLE_HIP))
  message(FATAL_ERROR
    "Accelerator variants use a Serial host execution space; disable OpenMP.")
endif()

if(SOLPROP_NATIVE_ENABLE_SANITIZERS AND
   (SOLPROP_NATIVE_ENABLE_CUDA OR SOLPROP_NATIVE_ENABLE_HIP))
  message(FATAL_ERROR
    "SOLPROP_NATIVE_ENABLE_SANITIZERS is the CPU ASan+UBSan configuration; "
    "use the accelerator vendor's memory tools for device builds.")
endif()

if(SOLPROP_NATIVE_ENABLE_CUDA)
  set(SOLPROP_NATIVE_DEVICE_BACKEND "cuda")
  set(SOLPROP_NATIVE_DEVICE_ARCH "${SOLPROP_NATIVE_CUDA_ARCH}")
elseif(SOLPROP_NATIVE_ENABLE_HIP)
  set(SOLPROP_NATIVE_DEVICE_BACKEND "hip")
  set(SOLPROP_NATIVE_DEVICE_ARCH "${SOLPROP_NATIVE_HIP_ARCH}")
else()
  set(SOLPROP_NATIVE_DEVICE_BACKEND "none")
  set(SOLPROP_NATIVE_DEVICE_ARCH "host")
endif()
string(TOUPPER "${CMAKE_BUILD_TYPE}" SOLPROP_NATIVE_BUILD_TYPE_UPPER)
set(SOLPROP_NATIVE_CONFIGURATION_FLAGS
    "${CMAKE_CXX_FLAGS_${SOLPROP_NATIVE_BUILD_TYPE_UPPER}}")
string(CONCAT SOLPROP_NATIVE_COMPILER_FLAGS
  "${CMAKE_CXX_FLAGS} ${SOLPROP_NATIVE_CONFIGURATION_FLAGS}")
if(SOLPROP_NATIVE_CPU_BASELINE_FLAG)
  string(APPEND SOLPROP_NATIVE_COMPILER_FLAGS " ${SOLPROP_NATIVE_CPU_BASELINE_FLAG}")
endif()
string(STRIP "${SOLPROP_NATIVE_COMPILER_FLAGS}" SOLPROP_NATIVE_COMPILER_FLAGS)
string(APPEND SOLPROP_NATIVE_COMPILER_FLAGS
  " -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow")
if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
  string(APPEND SOLPROP_NATIVE_COMPILER_FLAGS " -fp-model=precise")
endif()
if(SOLPROP_NATIVE_FP_CONTRACT_FLAG)
  string(APPEND SOLPROP_NATIVE_COMPILER_FLAGS " ${SOLPROP_NATIVE_FP_CONTRACT_FLAG}")
endif()
# The band rule's staging option changes the compiled column; the record says so.
if(SOLPROP_NATIVE_BAND_STAGING)
  string(APPEND SOLPROP_NATIVE_COMPILER_FLAGS " -DSOLPROP_NATIVE_BAND_STAGING=1")
endif()
if(SOLPROP_NATIVE_ENABLE_COVERAGE)
  list(JOIN SOLPROP_NATIVE_COVERAGE_FLAGS " " _coverage_flags_record)
  string(APPEND SOLPROP_NATIVE_COMPILER_FLAGS " ${_coverage_flags_record}")
endif()

if(SOLPROP_NATIVE_ENABLE_CLANG_TIDY)
  find_program(SOLPROP_NATIVE_CLANG_TIDY clang-tidy
               HINTS /opt/homebrew/opt/llvm/bin
                     /usr/local/opt/llvm/bin
               REQUIRED)
  set(SOLPROP_NATIVE_CLANG_TIDY_COMMAND
      "${SOLPROP_NATIVE_CLANG_TIDY}"
      "--config-file=${PROJECT_SOURCE_DIR}/.clang-tidy"
      "--header-filter=^${PROJECT_SOURCE_DIR}/(include|src|tests)/.*")
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    execute_process(
      COMMAND "${CMAKE_CXX_COMPILER}" -print-file-name=include
      OUTPUT_VARIABLE SOLPROP_NATIVE_GNU_INCLUDE_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      COMMAND_ERROR_IS_FATAL ANY)
    if(EXISTS "${SOLPROP_NATIVE_GNU_INCLUDE_DIR}/omp.h")
      # Parse with the compiling GCC's own libstdc++ rather than whichever GCC clang
      # would find first: a newer GCC's omp.h includes libstdc++ internals that an
      # older system GCC's headers do not have.
      get_filename_component(SOLPROP_NATIVE_GNU_INSTALL_DIR
                             "${SOLPROP_NATIVE_GNU_INCLUDE_DIR}" DIRECTORY)
      list(APPEND SOLPROP_NATIVE_CLANG_TIDY_COMMAND
           "--extra-arg=--gcc-install-dir=${SOLPROP_NATIVE_GNU_INSTALL_DIR}"
           # After clang's own resource headers, so that tidy finds GCC's omp.h but
           # parses clang's <immintrin.h>, not GCC's (clang 18 cannot parse GCC 13's).
           "--extra-arg=-idirafter${SOLPROP_NATIVE_GNU_INCLUDE_DIR}"
           # Clang 18 does not parse GCC 13's cleanup-function argument to
           # the malloc attribute. This affects tidy's parser only; GNU C++
           # still sees the unmodified OpenMP header and compilation flags.
           "--extra-arg=-D__malloc__(...)=__malloc__")
    endif()
  endif()
endif()

function(solprop_native_configure_project_target target)
  target_compile_features(${target} PUBLIC cxx_std_20)
  if(SOLPROP_NATIVE_ENABLE_COVERAGE)
    target_compile_options(${target} PRIVATE ${SOLPROP_NATIVE_COVERAGE_FLAGS})
    target_link_options(${target} PRIVATE -fprofile-instr-generate)
  endif()
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang,AMDClang,IntelLLVM>:-Wall>
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang,AMDClang,IntelLLVM>:-Wextra>
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang,AMDClang,IntelLLVM>:-Wpedantic>
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang,AMDClang,IntelLLVM>:-Wconversion>
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang,AMDClang,IntelLLVM>:-Wsign-conversion>
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang,AMDClang,IntelLLVM>:-Wshadow>
    $<$<AND:$<BOOL:${SOLPROP_NATIVE_WARNINGS_AS_ERRORS}>,$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang,AMDClang,IntelLLVM>>:-Werror>
  )
  if(SOLPROP_NATIVE_ENABLE_CLANG_TIDY)
    set_property(TARGET ${target} PROPERTY CXX_CLANG_TIDY
                 "${SOLPROP_NATIVE_CLANG_TIDY_COMMAND}")
  endif()
  if(SOLPROP_NATIVE_ENABLE_SANITIZERS)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang,AMDClang,IntelLLVM>:-fsanitize=address,undefined>
      $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang,AMDClang,IntelLLVM>:-fno-omit-frame-pointer>
    )
    target_link_options(${target} PRIVATE
      $<$<LINK_LANG_AND_ID:CXX,GNU,Clang,AppleClang,AMDClang,IntelLLVM>:-fsanitize=address,undefined>
    )
  endif()
endfunction()

# Homebrew LLVM includes an OpenMP runtime, but Homebrew FFTW's OpenMP library
# depends on the separate keg-only libomp formula. Force every native target to
# that one runtime: loading LLVM's bundled libomp and FFTW's formula libomp in
# the same Python process aborts when the second runtime initializes.
# AppleClang also needs the external header and frontend flag because Xcode
# does not ship OpenMP support itself.
if(APPLE AND SOLPROP_NATIVE_ENABLE_OPENMP)
  find_path(SOLPROP_NATIVE_LIBOMP_INCLUDE_DIR omp.h
            PATHS /opt/homebrew/opt/libomp/include
                  /usr/local/opt/libomp/include
            NO_DEFAULT_PATH REQUIRED)
  find_library(SOLPROP_NATIVE_LIBOMP_LIBRARY omp
               PATHS /opt/homebrew/opt/libomp/lib
                     /usr/local/opt/libomp/lib
               NO_DEFAULT_PATH REQUIRED)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    set(SOLPROP_NATIVE_OPENMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
  else()
    set(SOLPROP_NATIVE_OPENMP_CXX_FLAGS "-fopenmp=libomp")
  endif()
  set(OpenMP_CXX_FLAGS "${SOLPROP_NATIVE_OPENMP_CXX_FLAGS}" CACHE STRING
      "macOS OpenMP frontend flags" FORCE)
  set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING
      "macOS OpenMP runtime" FORCE)
  set(OpenMP_omp_LIBRARY "${SOLPROP_NATIVE_LIBOMP_LIBRARY}" CACHE FILEPATH
      "Homebrew libomp runtime" FORCE)
  set(OpenMP_CXX_INCLUDE_DIR "${SOLPROP_NATIVE_LIBOMP_INCLUDE_DIR}" CACHE PATH
      "Homebrew libomp headers" FORCE)
endif()

if(SOLPROP_NATIVE_ENABLE_CUDA AND APPLE)
  message(FATAL_ERROR "The CUDA build path is supported on Linux only.")
endif()

if(SOLPROP_NATIVE_ENABLE_HIP AND APPLE)
  message(FATAL_ERROR "The HIP build path is supported on Linux only.")
endif()

if(SOLPROP_NATIVE_ENABLE_CUDA)
  if(NOT SOLPROP_NATIVE_CUDA_ARCH MATCHES "^[A-Z0-9_]+$")
    message(FATAL_ERROR
      "SOLPROP_NATIVE_CUDA_ARCH must be a Kokkos architecture suffix, "
      "for example HOPPER90.")
  endif()

  # nvcc_wrapper uses CUDA_ROOT when it is exported, and otherwise finds nvcc
  # on PATH. CUDAToolkit_ROOT/CUDA_ROOT also guide Kokkos FFT's
  # find_package(CUDAToolkit), which supplies CUDA::cufft.
  if(CUDAToolkit_ROOT AND NOT CUDA_ROOT)
    set(CUDA_ROOT "${CUDAToolkit_ROOT}" CACHE PATH "CUDA Toolkit root")
  endif()
  find_program(SOLPROP_NATIVE_NVCC nvcc
               HINTS "${CUDA_ROOT}/bin" "$ENV{CUDA_ROOT}/bin"
                     /opt/cuda/bin /usr/local/cuda/bin
               REQUIRED)
  # CUDA 12.8 is the first toolkit whose host-compiler table admits GCC 14 (12.6 stops
  # at GCC 13.2); the qualified reference is CUDA 13 with GCC 14.
  execute_process(COMMAND "${SOLPROP_NATIVE_NVCC}" --version
                  OUTPUT_VARIABLE _solprop_nvcc_version RESULT_VARIABLE _solprop_nvcc_result)
  string(REGEX MATCH "release ([0-9]+\\.[0-9]+)" _solprop_nvcc_match
         "${_solprop_nvcc_version}")
  if(NOT _solprop_nvcc_result EQUAL 0 OR NOT _solprop_nvcc_match)
    message(FATAL_ERROR "Could not read the CUDA toolkit version from ${SOLPROP_NATIVE_NVCC}.")
  endif()
  set(SOLPROP_NATIVE_CUDA_TOOLKIT_VERSION "${CMAKE_MATCH_1}")
  if(SOLPROP_NATIVE_CUDA_TOOLKIT_VERSION VERSION_LESS 12.8)
    message(FATAL_ERROR
      "CUDA builds require CUDA 12.8 or newer (the first to support GCC 14 as the host "
      "compiler); found ${SOLPROP_NATIVE_CUDA_TOOLKIT_VERSION} at ${SOLPROP_NATIVE_NVCC}.")
  endif()
  if(NOT CUDA_ROOT)
    get_filename_component(SOLPROP_NATIVE_CUDA_BIN "${SOLPROP_NATIVE_NVCC}" DIRECTORY)
    get_filename_component(SOLPROP_NATIVE_CUDA_ROOT "${SOLPROP_NATIVE_CUDA_BIN}" DIRECTORY)
    set(CUDA_ROOT "${SOLPROP_NATIVE_CUDA_ROOT}" CACHE PATH "CUDA Toolkit root")
  endif()
  set(CUDAToolkit_ROOT "${CUDA_ROOT}" CACHE PATH "CUDA Toolkit root" FORCE)
  set("Kokkos_ARCH_${SOLPROP_NATIVE_CUDA_ARCH}" ON CACHE BOOL
      "Target CUDA architecture" FORCE)
  message(STATUS
    "CUDA backend enabled: nvcc=${SOLPROP_NATIVE_NVCC}, "
    "architecture=${SOLPROP_NATIVE_CUDA_ARCH}")
endif()

if(SOLPROP_NATIVE_ENABLE_HIP)
  if(NOT SOLPROP_NATIVE_HIP_ARCH MATCHES "^[A-Z0-9_]+$")
    message(FATAL_ERROR
      "SOLPROP_NATIVE_HIP_ARCH must be a Kokkos architecture suffix, "
      "for example AMD_GFX942.")
  endif()
  set("Kokkos_ARCH_${SOLPROP_NATIVE_HIP_ARCH}" ON CACHE BOOL
      "Target HIP architecture" FORCE)
  message(STATUS "HIP backend enabled: architecture=${SOLPROP_NATIVE_HIP_ARCH}")
endif()

if(SOLPROP_NATIVE_ENABLE_OPENMP)
  set(Kokkos_ENABLE_SERIAL OFF CACHE BOOL "Disable the Serial backend" FORCE)
  set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "Enable the OpenMP backend" FORCE)
elseif(SOLPROP_NATIVE_ENABLE_THREADS)
  set(Kokkos_ENABLE_SERIAL OFF CACHE BOOL "Disable the Serial backend" FORCE)
  set(Kokkos_ENABLE_OPENMP OFF CACHE BOOL "Disable the OpenMP backend" FORCE)
else()
  set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "Enable the Serial backend" FORCE)
  set(Kokkos_ENABLE_OPENMP OFF CACHE BOOL "Disable the OpenMP backend" FORCE)
endif()
set(Kokkos_ARCH_NATIVE ${SOLPROP_NATIVE_HOST_ARCH} CACHE BOOL
    "Optimize Kokkos host code for the build machine" FORCE)
set(Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION
    ${SOLPROP_NATIVE_ENABLE_AGGRESSIVE_VECTORIZATION} CACHE BOOL
    "Enable aggressive vectorization of independent Kokkos loops" FORCE)
set(Kokkos_ENABLE_THREADS ${SOLPROP_NATIVE_ENABLE_THREADS} CACHE BOOL
    "Enable the Threads backend" FORCE)
set(Kokkos_ENABLE_CUDA ${SOLPROP_NATIVE_ENABLE_CUDA} CACHE BOOL
    "Enable the CUDA backend" FORCE)
set(Kokkos_ENABLE_HIP ${SOLPROP_NATIVE_ENABLE_HIP} CACHE BOOL
    "Enable the HIP backend" FORCE)
set(Kokkos_ENABLE_SYCL OFF CACHE BOOL "SYCL is outside the supported backends" FORCE)
set(Kokkos_ENABLE_TESTS OFF CACHE BOOL "Do not build Kokkos tests" FORCE)

# CPU IPO is checked rather than assuming that the compiler, archiver, and
# linker all support LTO. The CUDA path already compiles each device kernel in
# one translation unit and uses an explicit GPU architecture; generic C++ IPO
# flags are intentionally not sent through nvcc_wrapper.
set(SOLPROP_NATIVE_IPO_AVAILABLE OFF)
if(SOLPROP_NATIVE_ENABLE_IPO AND NOT SOLPROP_NATIVE_ENABLE_COVERAGE AND
   NOT SOLPROP_NATIVE_ENABLE_CUDA AND
   NOT SOLPROP_NATIVE_ENABLE_HIP)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT SOLPROP_NATIVE_IPO_SUPPORTED
                      OUTPUT SOLPROP_NATIVE_IPO_ERROR LANGUAGES CXX)
  if(SOLPROP_NATIVE_IPO_SUPPORTED)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
    set(SOLPROP_NATIVE_IPO_AVAILABLE ON)
  else()
    message(WARNING "IPO/LTO is unavailable: ${SOLPROP_NATIVE_IPO_ERROR}")
  endif()
endif()

# Kokkos's host waits back off with sub-millisecond sleeps, which Windows rounds up to its
# timer resolution; cmake/KokkosWindowsHostWait.cmake makes them yield there.
FetchContent_Declare(
  kokkos
  GIT_REPOSITORY https://github.com/kokkos/kokkos.git
  GIT_TAG 5.2.1
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE
  PATCH_COMMAND "${CMAKE_COMMAND}" -P "${PROJECT_SOURCE_DIR}/cmake/KokkosWindowsHostWait.cmake"
  UPDATE_DISCONNECTED TRUE
)
FetchContent_MakeAvailable(kokkos)
# FetchContent's EXCLUDE_FROM_ALL argument was only added in CMake 3.28.
# Setting the directory property retains CMake 3.23 compatibility and keeps
# dependency install rules out of the Python wheel.
set_property(DIRECTORY "${kokkos_SOURCE_DIR}" PROPERTY EXCLUDE_FROM_ALL TRUE)

# The modal transforms use the ordinary dense GEMM interface at solprop's skinny-wide
# transform shapes.  Only BLAS is enabled; sparse, graph, batched, LAPACK, tests,
# examples, and performance tests are deliberately absent from this package.
set(KokkosKernels_ENABLE_ALL_COMPONENTS OFF CACHE BOOL
    "Build only selected KokkosKernels components" FORCE)
set(KokkosKernels_ENABLE_COMPONENT_BLAS ON CACHE BOOL
    "Build the KokkosKernels BLAS component" FORCE)
set(KokkosKernels_ENABLE_COMPONENT_BATCHED OFF CACHE BOOL
    "No batched dense kernels are used" FORCE)
set(KokkosKernels_ENABLE_COMPONENT_SPARSE OFF CACHE BOOL
    "No sparse kernels are used" FORCE)
set(KokkosKernels_ENABLE_COMPONENT_GRAPH OFF CACHE BOOL
    "No graph kernels are used" FORCE)
set(KokkosKernels_ENABLE_COMPONENT_LAPACK OFF CACHE BOOL
    "No LAPACK kernels are used" FORCE)
set(KokkosKernels_ENABLE_TESTS OFF CACHE BOOL
    "Do not build KokkosKernels tests" FORCE)
set(KokkosKernels_ENABLE_EXAMPLES OFF CACHE BOOL
    "Do not build KokkosKernels examples" FORCE)
set(KokkosKernels_ENABLE_PERFTESTS OFF CACHE BOOL
    "Do not build KokkosKernels benchmarks" FORCE)
set(KokkosKernels_NO_DEFAULT_CUDA_TPLS ON CACHE BOOL
    "Enable only the explicitly requested CUDA TPL" FORCE)
set(KokkosKernels_NO_DEFAULT_ROCM_TPLS ON CACHE BOOL
    "Enable only the explicitly requested ROCm TPL" FORCE)
if(SOLPROP_NATIVE_ENABLE_CUDA)
  set(KokkosKernels_ENABLE_TPL_BLAS OFF CACHE BOOL
      "Host BLAS cannot access CUDA device memory" FORCE)
  set(KokkosKernels_ENABLE_TPL_CUBLAS ${SOLPROP_NATIVE_ENABLE_TPL_BLAS}
      CACHE BOOL "Enable cuBLAS for eligible CUDA GEMMs" FORCE)
  set(KokkosKernels_ENABLE_TPL_ROCBLAS OFF CACHE BOOL
      "rocBLAS is unavailable in a CUDA build" FORCE)
elseif(SOLPROP_NATIVE_ENABLE_HIP)
  set(KokkosKernels_ENABLE_TPL_BLAS OFF CACHE BOOL
      "Host BLAS cannot access HIP device memory" FORCE)
  set(KokkosKernels_ENABLE_TPL_CUBLAS OFF CACHE BOOL
      "cuBLAS is unavailable in a HIP build" FORCE)
  set(KokkosKernels_ENABLE_TPL_ROCBLAS ${SOLPROP_NATIVE_ENABLE_TPL_BLAS}
      CACHE BOOL "Enable rocBLAS for eligible HIP GEMMs" FORCE)
else()
  set(KokkosKernels_ENABLE_TPL_BLAS ${SOLPROP_NATIVE_ENABLE_TPL_BLAS}
      CACHE BOOL "Enable the selected host BLAS" FORCE)
  set(KokkosKernels_ENABLE_TPL_CUBLAS OFF CACHE BOOL
      "cuBLAS is unavailable in a CPU build" FORCE)
  set(KokkosKernels_ENABLE_TPL_ROCBLAS OFF CACHE BOOL
      "rocBLAS is unavailable in a CPU build" FORCE)
endif()
set(KokkosKernels_ENABLE_TPL_LAPACK OFF CACHE BOOL
    "BLAS must not pull in unused LAPACK" FORCE)
if(SOLPROP_NATIVE_BLAS_LIBRARIES)
  set(SOLPROP_NATIVE_BLAS_NAMES)
  set(SOLPROP_NATIVE_BLAS_LIBRARY_DIRS)
  foreach(SOLPROP_NATIVE_BLAS_LIBRARY IN LISTS SOLPROP_NATIVE_BLAS_LIBRARIES)
    if(IS_ABSOLUTE "${SOLPROP_NATIVE_BLAS_LIBRARY}")
      get_filename_component(SOLPROP_NATIVE_BLAS_LIBRARY_DIR
                             "${SOLPROP_NATIVE_BLAS_LIBRARY}" DIRECTORY)
      get_filename_component(SOLPROP_NATIVE_BLAS_NAME
                             "${SOLPROP_NATIVE_BLAS_LIBRARY}" NAME)
      # find_library adds a lib prefix back on POSIX only; Windows names stay whole.
      if(NOT WIN32)
        string(REGEX REPLACE "^lib" "" SOLPROP_NATIVE_BLAS_NAME
                             "${SOLPROP_NATIVE_BLAS_NAME}")
      endif()
      string(REGEX REPLACE "\\.(a|lib|dylib|framework|so(\\.[0-9.]+)?)$" ""
                           SOLPROP_NATIVE_BLAS_NAME
                           "${SOLPROP_NATIVE_BLAS_NAME}")
      list(APPEND SOLPROP_NATIVE_BLAS_LIBRARY_DIRS
           "${SOLPROP_NATIVE_BLAS_LIBRARY_DIR}")
      list(APPEND SOLPROP_NATIVE_BLAS_NAMES "${SOLPROP_NATIVE_BLAS_NAME}")
    else()
      list(APPEND SOLPROP_NATIVE_BLAS_NAMES
           "${SOLPROP_NATIVE_BLAS_LIBRARY}")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES SOLPROP_NATIVE_BLAS_LIBRARY_DIRS)
  set(BLAS_LIBRARIES "${SOLPROP_NATIVE_BLAS_NAMES}" CACHE STRING
      "Explicit BLAS library names" FORCE)
  if(SOLPROP_NATIVE_BLAS_LIBRARY_DIRS)
    set(BLAS_LIBRARY_DIRS "${SOLPROP_NATIVE_BLAS_LIBRARY_DIRS}" CACHE STRING
        "Explicit BLAS library directories" FORCE)
  endif()
endif()

FetchContent_Declare(
  kokkoskernels
  GIT_REPOSITORY https://github.com/kokkos/kokkos-kernels.git
  GIT_TAG 5.2.0
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(kokkoskernels)
set_property(DIRECTORY "${kokkoskernels_SOURCE_DIR}" PROPERTY EXCLUDE_FROM_ALL TRUE)

set(SOLPROP_NATIVE_BLAS_BACKEND "none")
set(SOLPROP_NATIVE_BLAS_THREAD_POLICY "disabled")
if(SOLPROP_NATIVE_ENABLE_TPL_BLAS)
  if(SOLPROP_NATIVE_ENABLE_CUDA)
    set(SOLPROP_NATIVE_BLAS_BACKEND "cuBLAS")
    set(SOLPROP_NATIVE_BLAS_THREAD_POLICY "CUDA execution-space stream")
  elseif(SOLPROP_NATIVE_ENABLE_HIP)
    set(SOLPROP_NATIVE_BLAS_BACKEND "rocBLAS")
    set(SOLPROP_NATIVE_BLAS_THREAD_POLICY "HIP execution-space stream")
  elseif(SOLPROP_NATIVE_BLAS_VENDOR STREQUAL "MKL")
    set(SOLPROP_NATIVE_BLAS_BACKEND "MKL")
    set(SOLPROP_NATIVE_BLAS_THREAD_POLICY "MKL_DOMAIN_NUM_THREADS at process start")
  elseif(SOLPROP_NATIVE_BLAS_VENDOR STREQUAL "OPENBLAS" OR BLAS_openblas_LIBRARY OR
         "${BLAS_LIBRARIES}" MATCHES "[Oo]pen[Bb][Ll][Aa][Ss]")
    set(SOLPROP_NATIVE_BLAS_BACKEND "OpenBLAS")
    set(SOLPROP_NATIVE_BLAS_THREAD_POLICY "OPENBLAS_NUM_THREADS at process start")
  elseif(APPLE AND NOT SOLPROP_NATIVE_BLAS_LIBRARIES)
    set(SOLPROP_NATIVE_BLAS_BACKEND "Accelerate")
    set(SOLPROP_NATIVE_BLAS_THREAD_POLICY "system-managed")
  else()
    set(SOLPROP_NATIVE_BLAS_BACKEND "host BLAS")
    set(SOLPROP_NATIVE_BLAS_THREAD_POLICY "provider environment at process start")
  endif()
endif()
message(STATUS
  "solprop-native build profile: ${SOLPROP_NATIVE_BUILD_PROFILE}; "
  "BLAS ${SOLPROP_NATIVE_BLAS_BACKEND} (${SOLPROP_NATIVE_BLAS_SELECTION}); "
  "GEMM provider ${SOLPROP_NATIVE_GEMM_PROVIDER}; "
  "native architecture ${SOLPROP_NATIVE_HOST_ARCH}; "
  "CPU baseline ${SOLPROP_NATIVE_CPU_BASELINE}; "
  "IPO ${SOLPROP_NATIVE_IPO_AVAILABLE} (Release builds)")

if(SOLPROP_NATIVE_ENABLE_CUDA)
  set(KokkosFFT_ENABLE_FFTW OFF CACHE BOOL
      "Do not require FFTW for a CUDA-only FFT path" FORCE)
  set(KokkosFFT_ENABLE_CUFFT ON CACHE BOOL "Use cuFFT" FORCE)
  set(KokkosFFT_ENABLE_ROCFFT OFF CACHE BOOL "Disable rocFFT" FORCE)
  set(KokkosFFT_ENABLE_HIPFFT OFF CACHE BOOL "Disable hipFFT" FORCE)
elseif(SOLPROP_NATIVE_ENABLE_HIP)
  set(KokkosFFT_ENABLE_FFTW OFF CACHE BOOL
      "Do not require FFTW for a HIP-only FFT path" FORCE)
  set(KokkosFFT_ENABLE_CUFFT OFF CACHE BOOL "Disable cuFFT" FORCE)
  set(KokkosFFT_ENABLE_ROCFFT ON CACHE BOOL "Use rocFFT" FORCE)
  set(KokkosFFT_ENABLE_HIPFFT OFF CACHE BOOL "Use rocFFT rather than hipFFT" FORCE)
else()
  set(KokkosFFT_ENABLE_FFTW ON CACHE BOOL
      "Use FFTW as the host FFT backend" FORCE)
  set(KokkosFFT_ENABLE_CUFFT OFF CACHE BOOL "Disable cuFFT" FORCE)
  set(KokkosFFT_ENABLE_ROCFFT OFF CACHE BOOL "Disable rocFFT" FORCE)
  set(KokkosFFT_ENABLE_HIPFFT OFF CACHE BOOL "Disable hipFFT" FORCE)
endif()
set(KokkosFFT_ENABLE_EXAMPLES OFF CACHE BOOL "Do not build Kokkos FFT examples" FORCE)
set(KokkosFFT_ENABLE_TESTS OFF CACHE BOOL "Do not build Kokkos FFT tests" FORCE)
set(KokkosFFT_ENABLE_BENCHMARK OFF CACHE BOOL "Do not build Kokkos FFT benchmarks" FORCE)
set(KokkosFFT_ENABLE_DISTRIBUTED OFF CACHE BOOL "Disable distributed FFTs" FORCE)

FetchContent_Declare(
  kokkosfft
  GIT_REPOSITORY https://github.com/kokkos/kokkos-fft.git
  GIT_TAG v2.0.0
  GIT_SHALLOW TRUE
  GIT_SUBMODULES ""
  GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(kokkosfft)
set_property(DIRECTORY "${kokkosfft_SOURCE_DIR}" PROPERTY EXCLUDE_FROM_ALL TRUE)

# nvcc_wrapper loses spaces embedded in quoted preprocessor definitions. Encode
# them for the compiler command and restore them while forming the build record.
string(REPLACE " " "__SOLPROP_SPACE__" SOLPROP_NATIVE_COMPILER_FLAGS_ENCODED
  "${SOLPROP_NATIVE_COMPILER_FLAGS}")
string(REPLACE " " "__SOLPROP_SPACE__" SOLPROP_NATIVE_BLAS_THREAD_POLICY_ENCODED
  "${SOLPROP_NATIVE_BLAS_THREAD_POLICY}")
string(REPLACE " " "__SOLPROP_SPACE__" SOLPROP_NATIVE_BLAS_SELECTION_ENCODED
  "${SOLPROP_NATIVE_BLAS_SELECTION}")

function(solprop_native_mark_dependency_headers_system target)
  target_include_directories(${target} SYSTEM PRIVATE
    "${kokkos_BINARY_DIR}"
    "${kokkos_BINARY_DIR}/core/src"
    "${kokkos_SOURCE_DIR}/core/src"
    "${kokkos_BINARY_DIR}/containers/src"
    "${kokkos_SOURCE_DIR}/containers/src"
    "${kokkos_BINARY_DIR}/algorithms/src"
    "${kokkos_SOURCE_DIR}/algorithms/src"
    "${kokkos_BINARY_DIR}/simd/src"
    "${kokkos_SOURCE_DIR}/simd/src"
    "${kokkosfft_BINARY_DIR}/fft/src"
    "${kokkosfft_SOURCE_DIR}/fft/src"
    "${kokkosfft_SOURCE_DIR}/common/src"
    "${kokkoskernels_BINARY_DIR}"
    "${kokkoskernels_SOURCE_DIR}/blas/src"
    "${kokkoskernels_SOURCE_DIR}/blas/impl"
    "${kokkoskernels_SOURCE_DIR}/blas/tpls"
    "${kokkoskernels_SOURCE_DIR}/common/src"
    "${kokkoskernels_SOURCE_DIR}/common/impl"
  )
endfunction()

if(SOLPROP_NATIVE_ENABLE_COVERAGE)
  # Keep label collection in a separate, uninstrumented Tools module. The runner
  # loads it only for a separate execution-label pass, never a timing sample.
  add_library(solprop_native_labels MODULE src/coverage_labels.cpp)
  target_include_directories(solprop_native_labels SYSTEM PRIVATE
    "${kokkos_SOURCE_DIR}/core/src")
  target_compile_options(solprop_native_labels PRIVATE
    -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Werror)
  set_target_properties(solprop_native_labels PROPERTIES
    PREFIX "" CXX_VISIBILITY_PRESET default)
  install(TARGETS solprop_native_labels LIBRARY DESTINATION solprop_native)
endif()

add_library(solprop_native_core STATIC src/core.cpp src/general.cpp
  ${SOLPROP_NATIVE_PRECISION_SOURCES}
  src/general_modal.cpp src/coupled_flow.cpp src/dispersive.cpp src/far_field.cpp)
# A binary32 pass computes in binary32: any binary64 promotion in the single-precision
# translation unit is an error (solprop SEP 36 section 6.3).
if(SOLPROP_NATIVE_PRECISION_SOURCES)
  set_source_files_properties(${SOLPROP_NATIVE_PRECISION_SOURCES} PROPERTIES
    COMPILE_OPTIONS "-Wdouble-promotion")
endif()
target_include_directories(solprop_native_core PUBLIC
  "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
)
target_link_libraries(solprop_native_core PUBLIC
  ${CMAKE_DL_LIBS}
  Kokkos::kokkos
  KokkosFFT::fft
  Kokkos::kokkoskernels
  ${SOLPROP_NATIVE_MKL_FFT_LIBRARIES}
)
target_compile_definitions(solprop_native_core PUBLIC
  SOLPROP_NATIVE_GEMM_PROVIDER_${SOLPROP_NATIVE_GEMM_PROVIDER}=1
  SOLPROP_NATIVE_DEVICE_TEAM_SIZE=${SOLPROP_NATIVE_DEVICE_TEAM_SIZE}
  SOLPROP_NATIVE_DEVICE_REDUCTION_SIZE=${SOLPROP_NATIVE_DEVICE_REDUCTION_SIZE}
  SOLPROP_NATIVE_BAND_STAGING=$<BOOL:${SOLPROP_NATIVE_BAND_STAGING}>
  ${SOLPROP_NATIVE_PRECISION_DEFINITIONS}
)
if(SOLPROP_NATIVE_BLAS_BACKEND STREQUAL "OpenBLAS")
  target_compile_definitions(solprop_native_core PUBLIC
    SOLPROP_NATIVE_TPL_BLAS_OPENBLAS=1
  )
endif()
if(SOLPROP_NATIVE_BLAS_BACKEND STREQUAL "cuBLAS")
  target_compile_definitions(solprop_native_core PUBLIC
    SOLPROP_NATIVE_TPL_BLAS_CUBLAS=1
  )
elseif(SOLPROP_NATIVE_BLAS_BACKEND STREQUAL "rocBLAS")
  target_compile_definitions(solprop_native_core PUBLIC
    SOLPROP_NATIVE_TPL_BLAS_ROCBLAS=1
  )
endif()
if(SOLPROP_NATIVE_RAMAN_FFT_PROVIDER STREQUAL "ACCELERATE")
  target_sources(solprop_native_core PRIVATE src/raman_fft.cpp)
  target_compile_definitions(solprop_native_core PUBLIC SOLPROP_NATIVE_RAMAN_ACCELERATE=1)
  target_link_libraries(solprop_native_core PUBLIC "-framework Accelerate")
endif()
solprop_native_mark_dependency_headers_system(solprop_native_core)
solprop_native_configure_project_target(solprop_native_core)

nanobind_add_module(_core NOMINSIZE NB_SUPPRESS_WARNINGS src/bindings.cpp
  src/helpers.cpp)
target_link_libraries(_core PRIVATE solprop_native_core ${CMAKE_DL_LIBS})
solprop_native_mark_dependency_headers_system(_core)
solprop_native_configure_project_target(_core)
if(WIN32 AND SOLPROP_NATIVE_BLAS_STATIC)
  # A static OpenBLAS is invisible to the module walk that finds a DLL's symbols, so
  # the extension exports the three calls its BLAS thread fit and build record make.
  target_link_options(_core PRIVATE
    "LINKER:/EXPORT:openblas_get_parallel"
    "LINKER:/EXPORT:openblas_get_num_threads"
    "LINKER:/EXPORT:openblas_set_num_threads")
endif()
target_compile_definitions(_core PRIVATE
  SOLPROP_NATIVE_BACKEND_COMMIT="${SOLPROP_NATIVE_BACKEND_COMMIT}"
  SOLPROP_NATIVE_SOLPROP_BUILT_AGAINST="${SOLPROP_NATIVE_SOLPROP_BUILT_AGAINST}"
  SOLPROP_NATIVE_KOKKOS_VERSION="5.2.1"
  SOLPROP_NATIVE_KOKKOSFFT_VERSION="2.0.0"
  SOLPROP_NATIVE_KOKKOSKERNELS_VERSION="5.2.0"
  SOLPROP_NATIVE_GEMM_PROVIDER="${SOLPROP_NATIVE_GEMM_PROVIDER}"
  SOLPROP_NATIVE_TPL_BLAS_ENABLED=$<BOOL:${SOLPROP_NATIVE_ENABLE_TPL_BLAS}>
  SOLPROP_NATIVE_BLAS_BACKEND="${SOLPROP_NATIVE_BLAS_BACKEND}"
  SOLPROP_NATIVE_BLAS_THREAD_POLICY="${SOLPROP_NATIVE_BLAS_THREAD_POLICY_ENCODED}"
  SOLPROP_NATIVE_BLAS_SELECTION="${SOLPROP_NATIVE_BLAS_SELECTION_ENCODED}"
  SOLPROP_NATIVE_BUILD_PROFILE="${SOLPROP_NATIVE_BUILD_PROFILE}"
  SOLPROP_NATIVE_CPU_BASELINE="${SOLPROP_NATIVE_CPU_BASELINE}"
  SOLPROP_NATIVE_BLAS_SYMBOL_PREFIX="${SOLPROP_NATIVE_BLAS_SYMBOL_PREFIX}"
  SOLPROP_NATIVE_RAMAN_FFT_PROVIDER="${SOLPROP_NATIVE_RAMAN_FFT_PROVIDER}"
  SOLPROP_NATIVE_HOST_FFT_BACKEND="${SOLPROP_NATIVE_HOST_FFT_BACKEND}"
  SOLPROP_NATIVE_HOST_FFT_PLANNER="${SOLPROP_NATIVE_HOST_FFT_PLANNER}"
  SOLPROP_NATIVE_NANOBIND_VERSION="${SOLPROP_NATIVE_NANOBIND_VERSION}"
  SOLPROP_NATIVE_DEVICE_BACKEND="${SOLPROP_NATIVE_DEVICE_BACKEND}"
  SOLPROP_NATIVE_DEVICE_ARCH="${SOLPROP_NATIVE_DEVICE_ARCH}"
  SOLPROP_NATIVE_DEVICE_TEAM_SIZE_RECORD=${SOLPROP_NATIVE_DEVICE_TEAM_SIZE}
  SOLPROP_NATIVE_DEVICE_REDUCTION_SIZE_RECORD=${SOLPROP_NATIVE_DEVICE_REDUCTION_SIZE}
  SOLPROP_NATIVE_COMPILER_FLAGS="${SOLPROP_NATIVE_COMPILER_FLAGS_ENCODED}"
  SOLPROP_NATIVE_BUILD_TYPE="$<CONFIG>"
  SOLPROP_NATIVE_COVERAGE_INSTRUMENTED=$<BOOL:${SOLPROP_NATIVE_ENABLE_COVERAGE}>
  SOLPROP_NATIVE_COVERAGE_SOURCE_HASH="${SOLPROP_NATIVE_COVERAGE_SOURCE_HASH}"
  SOLPROP_NATIVE_FP_CONTRACT_RECORD="${SOLPROP_NATIVE_FP_CONTRACT_FLAG}"
  SOLPROP_NATIVE_HOST_ARCH_ENABLED=$<BOOL:${SOLPROP_NATIVE_HOST_ARCH}>
  SOLPROP_NATIVE_AGGRESSIVE_VECTORIZATION_ENABLED=$<BOOL:${SOLPROP_NATIVE_ENABLE_AGGRESSIVE_VECTORIZATION}>
  SOLPROP_NATIVE_IPO_ENABLED=$<AND:$<BOOL:${SOLPROP_NATIVE_IPO_AVAILABLE}>,$<CONFIG:Release>>
)

if(SOLPROP_NATIVE_BUILD_BENCHMARKS)
  add_executable(solprop_native_gemm_benchmark src/benchmark_gemm.cpp)
  target_link_libraries(solprop_native_gemm_benchmark PRIVATE solprop_native_core)
  solprop_native_mark_dependency_headers_system(solprop_native_gemm_benchmark)
  solprop_native_configure_project_target(solprop_native_gemm_benchmark)
endif()

if(BUILD_TESTING)
  set(INSTALL_GTEST OFF CACHE BOOL "Do not install the test-only GoogleTest" FORCE)
  set(gtest_force_shared_crt ON CACHE BOOL
      "Use the parent MSVC runtime if that platform is added" FORCE)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.17.0
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
  )
  FetchContent_MakeAvailable(googletest)
  set_property(DIRECTORY "${googletest_SOURCE_DIR}" PROPERTY EXCLUDE_FROM_ALL TRUE)

  add_executable(solprop_native_tests
    tests/native/main.cpp
    tests/native/test_core.cpp
    tests/native/test_envelope_plasma.cpp
    tests/native/test_fft.cpp
    tests/native/test_fp.cpp
    tests/native/test_precision.cpp
    tests/native/test_propagator.cpp
    tests/native/test_general_modal.cpp
    tests/native/test_helpers.cpp
    tests/native/test_z_table.cpp
    tests/native/test_coupled_flow.cpp
    tests/native/test_dispersive.cpp
    tests/native/test_aperture.cpp
  )
  target_link_libraries(solprop_native_tests PRIVATE
    solprop_native_core
    GTest::gtest
  )
  target_include_directories(solprop_native_tests SYSTEM PRIVATE
    "${googletest_SOURCE_DIR}/googletest/include"
  )
  solprop_native_mark_dependency_headers_system(solprop_native_tests)
  solprop_native_configure_project_target(solprop_native_tests)

  if(SOLPROP_NATIVE_ENABLE_SANITIZERS AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Ubuntu's high-entropy ASLR can place a PIE mapping inside ASan's fixed shadow
    # range before the test process starts. Keep the instrumented binary unchanged
    # and disable randomisation only for this CTest invocation.
    find_program(SOLPROP_NATIVE_SETARCH setarch REQUIRED)
    add_test(NAME native_core
             COMMAND "${SOLPROP_NATIVE_SETARCH}" "${CMAKE_SYSTEM_PROCESSOR}"
                     -R "$<TARGET_FILE:solprop_native_tests>")
  else()
    add_test(NAME native_core COMMAND solprop_native_tests)
  endif()
  set_tests_properties(native_core PROPERTIES
    ENVIRONMENT "OMP_PROC_BIND=false"
  )
endif()

if(SKBUILD)
  # A local build's vendor BLAS (oneAPI's MKL, a private OpenBLAS) can live outside
  # the loader's default path; the installed module keeps the directories it linked
  # against, so it loads without the environment it was built in. The explicit BLAS
  # directories are listed too: sourcing oneAPI puts MKL's directory on LIBRARY_PATH,
  # which makes CMake treat it as implicit and leave it out of the link path.
  set(SOLPROP_NATIVE_INSTALL_RPATH)
  foreach(SOLPROP_NATIVE_RPATH_DIR IN LISTS SOLPROP_NATIVE_BLAS_LIBRARY_DIRS)
    if(NOT SOLPROP_NATIVE_RPATH_DIR MATCHES "^/(usr/)?lib(64)?(/|$)")
      list(APPEND SOLPROP_NATIVE_INSTALL_RPATH "${SOLPROP_NATIVE_RPATH_DIR}")
    endif()
  endforeach()
  set(SOLPROP_NATIVE_INSTALL_LINK_PATH TRUE)
  if(SOLPROP_NATIVE_VENDORED_LIBRARIES)
    # A distribution build carries its BLAS beside the module: the build
    # environment it linked against is gone once an isolated build finishes.
    if(APPLE)
      set(SOLPROP_NATIVE_INSTALL_RPATH "@loader_path")
    else()
      set(SOLPROP_NATIVE_INSTALL_RPATH "$ORIGIN")
    endif()
    set(SOLPROP_NATIVE_INSTALL_LINK_PATH FALSE)
    # On Windows the wheel repair (delvewheel) copies the DLLs under unique names
    # instead, so that they cannot be confused with another package's copies.
    if(NOT WIN32)
      install(FILES ${SOLPROP_NATIVE_VENDORED_LIBRARIES} DESTINATION solprop_native)
    endif()
  endif()
  # The BLAS's licence travels with it, copied or (Windows) statically linked.
  if(SOLPROP_NATIVE_VENDORED_LICENCE)
    install(FILES "${SOLPROP_NATIVE_VENDORED_LICENCE}" DESTINATION solprop_native
            RENAME ${SOLPROP_NATIVE_VENDORED_LICENCE_NAME})
  endif()
  set_target_properties(_core PROPERTIES
    INSTALL_RPATH "${SOLPROP_NATIVE_INSTALL_RPATH}"
    INSTALL_RPATH_USE_LINK_PATH ${SOLPROP_NATIVE_INSTALL_LINK_PATH})
  install(TARGETS _core LIBRARY DESTINATION solprop_native)
  # The package checks the processor against this before it loads the extension.
  file(WRITE "${PROJECT_BINARY_DIR}/cpu_baseline.txt" "${SOLPROP_NATIVE_CPU_BASELINE}\n")
  install(FILES "${PROJECT_BINARY_DIR}/cpu_baseline.txt" DESTINATION solprop_native
          RENAME _cpu_baseline.txt)
  # Kokkos, Kokkos Kernels, Kokkos-FFT, nanobind and nanobind's robin_map are compiled
  # into the extension, so the wheel carries their notices beside it. Kokkos-FFT is
  # MIT OR Apache-2.0 WITH LLVM-exception; its MIT terms are the ones taken here.
  install(FILES "${kokkos_SOURCE_DIR}/LICENSE" DESTINATION solprop_native
          RENAME LICENSE-kokkos.txt)
  install(FILES "${kokkoskernels_SOURCE_DIR}/LICENSE" DESTINATION solprop_native
          RENAME LICENSE-kokkos-kernels.txt)
  install(FILES "${kokkosfft_SOURCE_DIR}/LICENSES/MIT.txt" DESTINATION solprop_native
          RENAME LICENSE-kokkos-fft.txt)
  # nanobind's licence is in its distribution metadata; robin_map ships no licence
  # file inside nanobind, and its MIT notice is the leading comment of its header.
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c [=[
import importlib.metadata as metadata, pathlib, sys
out = pathlib.Path(sys.argv[1])
licence = next(f for f in metadata.files("nanobind") if f.name == "LICENSE")
(out / "LICENSE-nanobind.txt").write_text(pathlib.Path(licence.locate()).read_text())
header = pathlib.Path(sys.argv[2]).read_text().split("*/", 1)[0]
lines = [line.strip().removeprefix("/**").removeprefix("*").removeprefix(" ")
         for line in header.splitlines()]
(out / "LICENSE-robin-map.txt").write_text("\n".join(lines).strip() + "\n")
]=] "${PROJECT_BINARY_DIR}" "${nanobind_ROOT}/../ext/robin_map/include/tsl/robin_map.h"
    COMMAND_ERROR_IS_FATAL ANY
  )
  install(FILES "${PROJECT_BINARY_DIR}/LICENSE-nanobind.txt"
                "${PROJECT_BINARY_DIR}/LICENSE-robin-map.txt"
          DESTINATION solprop_native)
  # The wheel repair (auditwheel, delocate, delvewheel) copies FFTW and the OpenMP
  # runtime into the wheel but not their licences; the release recipe's dependency
  # scripts (native/ci/) collect those as LICENSE-*.txt in one directory, named here.
  set(SOLPROP_NATIVE_BUNDLED_LICENCES "$ENV{SOLPROP_NATIVE_BUNDLED_LICENCES}" CACHE PATH
      "Directory of LICENSE-*.txt files for the libraries the wheel repair bundles")
  if(SOLPROP_NATIVE_BUNDLED_LICENCES)
    file(GLOB _bundled_licences "${SOLPROP_NATIVE_BUNDLED_LICENCES}/LICENSE-*.txt")
    if(NOT _bundled_licences)
      message(FATAL_ERROR
        "SOLPROP_NATIVE_BUNDLED_LICENCES=${SOLPROP_NATIVE_BUNDLED_LICENCES} holds no LICENSE-*.txt")
    endif()
    install(FILES ${_bundled_licences} DESTINATION solprop_native)
  endif()
endif()
