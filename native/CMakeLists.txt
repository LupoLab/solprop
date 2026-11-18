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
# The source digest's patterns, which the coverage hash below shares.
include(cmake/SourceDigest.cmake)
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
  # The source digest's C++ set (cmake/SourceDigest.cmake), src/general/'s headers included.
  list(TRANSFORM SOLPROP_NATIVE_DIGEST_GLOBS PREPEND "${PROJECT_SOURCE_DIR}/"
       OUTPUT_VARIABLE _coverage_globs)
  file(GLOB_RECURSE _coverage_sources CONFIGURE_DEPENDS RELATIVE "${PROJECT_SOURCE_DIR}"
    ${_coverage_globs})
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
       NOT CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AMDClang)$")
  message(FATAL_ERROR "Linux CPU builds require GNU C++ or Clang/AOCC.")
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
  # GCC 13 compiles the out-of-line strict helpers (include/solprop_native/runtime/fp.hpp)
  # through the stack, stalling every call on a failed store-to-load forward: coupled
  # Multistep steps took 1.25 to 1.45 times as long as with GCC 14 (Cascade Lake,
  # Zen 5).
  message(FATAL_ERROR
    "Linux CPU builds require GCC 14 or newer; found ${CMAKE_CXX_COMPILER_VERSION}. "
    "On Ubuntu 24.04 or Debian, install it with 'sudo apt install g++-14' and build "
    "with CXX=g++-14 (a CMake cache that recorded another compiler must be removed). "
    "In a checkout, a contributor with no compiler can run "
    "'uv run --no-project tools/setup_light.py' instead.")
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

# solprop SEP 25: floating-point contraction (fused multiply-add) is on by default for
# CPU builds, and the few protected blocks opt out (include/solprop_native/runtime/fp.hpp).
# Clang's plain `-ffp-contract=fast` ignores `#pragma clang fp contract(off)` and
# `#pragma STDC FP_CONTRACT OFF`; `fast-honor-pragmas` contracts across statements and
# keeps the protected blocks strict. GCC ignores both pragmas: its protected functions
# carry `optimize("fp-contract=off")` instead. The flag is added after
# `-fno-fast-math`, so that does not reset it.
# Device builds keep their compilers' defaults (nvcc and HIP Clang already contract).
set(SOLPROP_NATIVE_FP_CONTRACT_FLAG "")
if(NOT SOLPROP_NATIVE_ENABLE_CUDA AND NOT SOLPROP_NATIVE_ENABLE_HIP)
  if(CMAKE_CXX_COMPILER_ID MATCHES "^(Clang|AppleClang|AMDClang)$")
    set(SOLPROP_NATIVE_FP_CONTRACT_FLAG "-ffp-contract=fast-honor-pragmas")
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(SOLPROP_NATIVE_FP_CONTRACT_FLAG "-ffp-contract=fast")
  endif()
endif()
if(SOLPROP_NATIVE_FP_CONTRACT_FLAG)
  add_compile_options("${SOLPROP_NATIVE_FP_CONTRACT_FLAG}")
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

find_package(Python 3.14 REQUIRED COMPONENTS Interpreter Development.Module Development.SABIModule)

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
# The benchmark history is what a tracking run writes, not what it times: one checkout
# may record several runs, and a modified history must not stamp the build dirty.
execute_process(
  COMMAND git diff --quiet --ignore-submodules HEAD -- ":/" ":(top,exclude)benchmarks/history"
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  RESULT_VARIABLE SOLPROP_NATIVE_TREE_DIRTY
)
if(SOLPROP_NATIVE_TREE_DIRTY)
  string(APPEND SOLPROP_NATIVE_BACKEND_COMMIT "+dirty")
endif()
# solprop and solprop-native are released together from one repository at one
# version, so the solprop version this build belongs to is its own. The record is
# provenance, which nothing compares: check_build() compares only an editable
# checkout's sources with the extension by digest (a stale extension beside newer
# sources), and an installed pair is held to one version by solprop's exact
# requirement on solprop-native (hatch_build.py).
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
# A build with its tests is development, and CI, and stops on a warning. A user's build
# from source isn't: one warning from a newer compiler shouldn't stop it.
option(SOLPROP_NATIVE_WARNINGS_AS_ERRORS
       "Treat diagnostics in project-owned C++ sources as errors" ${BUILD_TESTING})
option(SOLPROP_NATIVE_ENABLE_CLANG_TIDY
       "Run clang-tidy while compiling project-owned C++ sources" OFF)
option(SOLPROP_NATIVE_ENABLE_SANITIZERS
       "Instrument project targets with AddressSanitizer and UBSan" OFF)
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
# solprop SEP 18's band rule (include/solprop_native/response/plasma.hpp). The staging changes
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
# A device build replays an Adams trial's source launches as a device graph
# (graph.hpp, `graph::Region`); host builds have no graph.
option(SOLPROP_NATIVE_DEVICE_GRAPH
       "Replay an Adams trial's source launches as a CUDA or HIP graph" ON)

foreach(SOLPROP_NATIVE_DEVICE_SIZE IN ITEMS
        SOLPROP_NATIVE_DEVICE_TEAM_SIZE SOLPROP_NATIVE_DEVICE_REDUCTION_SIZE)
  if(NOT "${${SOLPROP_NATIVE_DEVICE_SIZE}}" MATCHES
         "^(64|128|256|512|1024)$")
    message(FATAL_ERROR
      "${SOLPROP_NATIVE_DEVICE_SIZE} must be one of 64, 128, 256, 512, or 1024; "
      "got ${${SOLPROP_NATIVE_DEVICE_SIZE}}.")
  endif()
endforeach()

if(SOLPROP_NATIVE_ENABLE_CUDA AND SOLPROP_NATIVE_ENABLE_HIP)
  message(FATAL_ERROR
    "CUDA and HIP are separate build variants and cannot both be enabled.")
endif()
if(SOLPROP_NATIVE_ENABLE_OPENMP AND
   (SOLPROP_NATIVE_ENABLE_CUDA OR SOLPROP_NATIVE_ENABLE_HIP))
  message(FATAL_ERROR
    "Accelerator variants use a Serial host execution space; disable OpenMP.")
endif()
if(NOT SOLPROP_NATIVE_ENABLE_OPENMP AND
   NOT (SOLPROP_NATIVE_ENABLE_CUDA OR SOLPROP_NATIVE_ENABLE_HIP))
  message(FATAL_ERROR
    "A CPU build runs the OpenMP host execution space, and the backend refuses a "
    "CPU build without it; enable SOLPROP_NATIVE_ENABLE_OPENMP, or enable "
    "SOLPROP_NATIVE_ENABLE_CUDA or SOLPROP_NATIVE_ENABLE_HIP for a device build.")
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
# The project's own sources' warnings, defined once: the targets compile with them and the
# build record lists them. MSVC is not a supported compiler (Windows builds use clang++).
set(SOLPROP_NATIVE_WARNING_FLAGS
    -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Wundef
    -Wnon-virtual-dtor -Woverloaded-virtual)
if(CMAKE_CXX_COMPILER_ID MATCHES "^(Clang|AppleClang|AMDClang)$")
  list(APPEND SOLPROP_NATIVE_WARNING_FLAGS -Wimplicit-fallthrough)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  list(APPEND SOLPROP_NATIVE_WARNING_FLAGS
       -Wduplicated-cond -Wduplicated-branches -Wlogical-op)
endif()
list(JOIN SOLPROP_NATIVE_WARNING_FLAGS " " _warning_flags_record)
string(APPEND SOLPROP_NATIVE_COMPILER_FLAGS " ${_warning_flags_record}")
if(SOLPROP_NATIVE_FP_CONTRACT_FLAG)
  string(APPEND SOLPROP_NATIVE_COMPILER_FLAGS " ${SOLPROP_NATIVE_FP_CONTRACT_FLAG}")
endif()
# The band rule's staging option changes the compiled column; the record says so.
if(SOLPROP_NATIVE_BAND_STAGING)
  string(APPEND SOLPROP_NATIVE_COMPILER_FLAGS " -DSOLPROP_NATIVE_BAND_STAGING=1")
endif()
if(NOT SOLPROP_NATIVE_DEVICE_GRAPH AND
   (SOLPROP_NATIVE_ENABLE_CUDA OR SOLPROP_NATIVE_ENABLE_HIP))
  string(APPEND SOLPROP_NATIVE_COMPILER_FLAGS " -DSOLPROP_NATIVE_DEVICE_GRAPH=0")
endif()
if(SOLPROP_NATIVE_ENABLE_COVERAGE)
  list(JOIN SOLPROP_NATIVE_COVERAGE_FLAGS " " _coverage_flags_record)
  string(APPEND SOLPROP_NATIVE_COMPILER_FLAGS " ${_coverage_flags_record}")
endif()

# clang-tidy while compiling, when SOLPROP_NATIVE_ENABLE_CLANG_TIDY asks for it.
include(cmake/ClangTidy.cmake)

function(solprop_native_configure_project_target target)
  target_compile_features(${target} PUBLIC cxx_std_20)
  if(SOLPROP_NATIVE_ENABLE_COVERAGE)
    target_compile_options(${target} PRIVATE ${SOLPROP_NATIVE_COVERAGE_FLAGS})
    target_link_options(${target} PRIVATE -fprofile-instr-generate)
  endif()
  target_compile_options(${target} PRIVATE
    "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang,AMDClang>:${SOLPROP_NATIVE_WARNING_FLAGS}>"
    $<$<AND:$<BOOL:${SOLPROP_NATIVE_WARNINGS_AS_ERRORS}>,$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang,AMDClang>>:-Werror>
  )
  if(SOLPROP_NATIVE_ENABLE_CLANG_TIDY)
    set_property(TARGET ${target} PROPERTY CXX_CLANG_TIDY
                 "${SOLPROP_NATIVE_CLANG_TIDY_COMMAND}")
  endif()
  if(SOLPROP_NATIVE_ENABLE_SANITIZERS)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang,AMDClang>:-fsanitize=address,undefined>
      # A report stops the test there, whatever UBSAN_OPTIONS says.
      $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang,AMDClang>:-fno-sanitize-recover=all>
      $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang,AMDClang>:-fno-omit-frame-pointer>
    )
    target_link_options(${target} PRIVATE
      $<$<LINK_LANG_AND_ID:CXX,GNU,Clang,AppleClang,AMDClang>:-fsanitize=address,undefined>
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

# The fetched dependencies, configured for the build: Kokkos (with the CUDA or HIP
# toolchain of a device build), Kokkos Kernels and its BLAS, Kokkos-FFT, vector maths.
include(cmake/Dependencies.cmake)

if(SOLPROP_NATIVE_ENABLE_COVERAGE)
  # Keep label collection in a separate, uninstrumented Tools module. The runner
  # loads it only for a separate execution-label pass, never a timing sample.
  add_library(solprop_native_labels MODULE src/runtime/coverage_labels.cpp)
  target_include_directories(solprop_native_labels SYSTEM PRIVATE
    "${kokkos_SOURCE_DIR}/core/src")
  target_compile_options(solprop_native_labels PRIVATE
    ${SOLPROP_NATIVE_WARNING_FLAGS} -Werror)
  set_target_properties(solprop_native_labels PROPERTIES
    PREFIX "" CXX_VISIBILITY_PRESET default)
  install(TARGETS solprop_native_labels LIBRARY DESTINATION solprop_native)
endif()

add_library(solprop_native_core STATIC src/runtime/cpu_execution.cpp src/runtime/crossing.cpp
  src/general/checks.cpp src/general/setup.cpp src/general/trial.cpp
  src/general/statistics.cpp src/general/state.cpp src/general/memory.cpp src/response/general_modal.cpp src/response/coupled_flow.cpp src/response/dispersive.cpp src/response/far_field.cpp
  src/march/adams_march.cpp src/session/watch.cpp src/march/rk_march.cpp src/march/engines.cpp src/session/session.cpp
  src/session/statistics_plan.cpp src/session/session_widths.cpp src/march/low_storage.cpp src/session/transverse.cpp
  src/session/session_edge.cpp src/march/dense.cpp src/problem/problem.cpp
  src/workspace/workspace.cpp src/workspace/run_records.cpp
  src/transforms/reference_timing.cpp src/transforms/fftw_wisdom.cpp)
# solprop SEP 39, SEP 45 and SEP 47: the Adams march's decisions, the boundary watch's
# predictions, the Runge-Kutta march's positions and norm, the low-storage replay's
# positions and table keys, the native session's scheduling, export, statistics
# finishing, fluence widths and dense-output weights and the transverse edge's
# judgement round as solprop's Python does, so their
# translation units take no contraction on any build (the flag follows the global one,
# and the last wins). MSVC's /fp:precise does not contract by default.
# SOLPROP_NATIVE_STRICT_TU marks them, so that the general workspace's contracting
# headers (src/general/) refuse to be included there.
set(SOLPROP_NATIVE_STRICT_SOURCES src/march/adams_march.cpp src/session/watch.cpp src/march/rk_march.cpp
  src/session/session.cpp src/session/statistics_plan.cpp src/session/session_widths.cpp src/march/low_storage.cpp
  src/session/transverse.cpp src/session/session_edge.cpp src/march/dense.cpp)
set_source_files_properties(${SOLPROP_NATIVE_STRICT_SOURCES}
  PROPERTIES COMPILE_DEFINITIONS SOLPROP_NATIVE_STRICT_TU)
if(CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang|AMDClang)$")
  set_source_files_properties(${SOLPROP_NATIVE_STRICT_SOURCES}
    PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
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
  SOLPROP_NATIVE_DEVICE_TEAM_SIZE=${SOLPROP_NATIVE_DEVICE_TEAM_SIZE}
  SOLPROP_NATIVE_DEVICE_REDUCTION_SIZE=${SOLPROP_NATIVE_DEVICE_REDUCTION_SIZE}
  SOLPROP_NATIVE_BAND_STAGING=$<BOOL:${SOLPROP_NATIVE_BAND_STAGING}>
)
if(SOLPROP_NATIVE_ENABLE_CUDA OR SOLPROP_NATIVE_ENABLE_HIP)
  target_compile_definitions(solprop_native_core PUBLIC
    SOLPROP_NATIVE_DEVICE_GRAPH=$<BOOL:${SOLPROP_NATIVE_DEVICE_GRAPH}>
  )
endif()
if(SOLPROP_NATIVE_BLAS_BACKEND STREQUAL "OpenBLAS")
  target_compile_definitions(solprop_native_core PUBLIC
    SOLPROP_NATIVE_TPL_BLAS_OPENBLAS=1
  )
endif()
# Compact problems' transforms take a scoped team through FFTW's loop callback
# (cpu_execution.cpp, `fftw_team_loops`), which only the OpenMP FFTW runs on the OpenMP team.
if(SOLPROP_NATIVE_HOST_FFT_BACKEND STREQUAL "FFTW_OPENMP")
  target_compile_definitions(solprop_native_core PUBLIC SOLPROP_NATIVE_FFTW_OPENMP=1)
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
  target_sources(solprop_native_core PRIVATE src/transforms/raman_fft.cpp)
  target_compile_definitions(solprop_native_core PUBLIC SOLPROP_NATIVE_RAMAN_ACCELERATE=1)
  target_link_libraries(solprop_native_core PUBLIC "-framework Accelerate")
endif()
solprop_native_link_vector_math(solprop_native_core)
solprop_native_mark_dependency_headers_system(solprop_native_core)
solprop_native_configure_project_target(solprop_native_core)

# What the extension is compiled from: this file, its cmake/ modules and the sources the
# digest's patterns match (cmake/SourceDigest.cmake says why a custom command and not a
# configure step).
list(TRANSFORM SOLPROP_NATIVE_DIGEST_GLOBS PREPEND "${PROJECT_SOURCE_DIR}/"
     OUTPUT_VARIABLE _source_digest_globs)
file(GLOB_RECURSE _source_digest_inputs CONFIGURE_DEPENDS ${_source_digest_globs})
list(TRANSFORM SOLPROP_NATIVE_DIGEST_MODULE_GLOBS PREPEND "${PROJECT_SOURCE_DIR}/"
     OUTPUT_VARIABLE _source_digest_module_globs)
file(GLOB _source_digest_modules CONFIGURE_DEPENDS LIST_DIRECTORIES false
     ${_source_digest_module_globs})
set(SOLPROP_NATIVE_SOURCE_DIGEST_CPP "${CMAKE_CURRENT_BINARY_DIR}/source_digest.cpp")
add_custom_command(
  OUTPUT "${SOLPROP_NATIVE_SOURCE_DIGEST_CPP}"
  COMMAND "${CMAKE_COMMAND}" "-DSOURCE_ROOT=${PROJECT_SOURCE_DIR}"
          "-DOUTPUT=${SOLPROP_NATIVE_SOURCE_DIGEST_CPP}"
          -P "${PROJECT_SOURCE_DIR}/cmake/SourceDigest.cmake"
  DEPENDS "${PROJECT_SOURCE_DIR}/CMakeLists.txt"
          "${PROJECT_SOURCE_DIR}/cmake/SourceDigest.cmake" ${_source_digest_inputs}
          ${_source_digest_modules}
  COMMENT "Hashing the native sources"
  VERBATIM
)

nanobind_add_module(_core STABLE_ABI NOMINSIZE NB_SUPPRESS_WARNINGS src/bindings/module.cpp
  src/bindings/problem.cpp src/bindings/march.cpp src/bindings/runtime.cpp
  src/bindings/testing.cpp src/bindings/transverse.cpp src/bindings/helpers.cpp
  src/bindings/records.cpp
  "${SOLPROP_NATIVE_SOURCE_DIGEST_CPP}")
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

# The build record, compiled into the extension as definitions.
include(cmake/BuildRecord.cmake)

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
    tests/native/test_adams_march.cpp
    tests/native/test_watch.cpp
    tests/native/test_rk_march.cpp
    tests/native/test_session.cpp
    tests/native/test_session_monitor.cpp
    tests/native/test_low_storage.cpp
    tests/native/test_transverse.cpp
    tests/native/test_session_transverse.cpp
    tests/native/test_adams_phi.cpp
    tests/native/test_cpu_execution.cpp
    tests/native/test_gemm.cpp
    tests/native/test_general_state.cpp
    tests/native/test_plasma_scan.cpp
    tests/native/test_crossing.cpp
    tests/native/test_envelope_plasma.cpp
    tests/native/test_fft.cpp
    tests/native/test_fp.cpp
    tests/native/test_general_modal.cpp
    tests/native/test_chebyshev.cpp
    tests/native/test_z_table.cpp
    tests/native/test_stage_tables.cpp
    tests/native/test_coupled_flow.cpp
    tests/native/test_dispersive.cpp
    tests/native/test_far_field.cpp
    tests/native/test_vmath.cpp
    tests/native/test_workspace.cpp
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

# The wheel's install rules, for a scikit-build-core build.
include(cmake/Install.cmake)
