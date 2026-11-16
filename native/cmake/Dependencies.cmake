# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# The fetched dependencies. Refuses an unsupported device configuration, finds a CUDA
# build's toolkit, selects Kokkos's execution spaces and architecture, fetches Kokkos,
# Kokkos Kernels (the BLAS component, with the BLAS it dispatches to) and Kokkos-FFT, and
# includes cmake/VectorMath.cmake. Defines the kokkos, kokkoskernels and kokkosfft
# directories, SOLPROP_NATIVE_BLAS_BACKEND, SOLPROP_NATIVE_BLAS_THREAD_POLICY,
# SOLPROP_NATIVE_BLAS_LIBRARY_DIRS and solprop_native_mark_dependency_headers_system().
# Expects the options, cmake/HostLibraries.cmake's choices and the IPO check.

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
else()
  set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "Enable the Serial backend" FORCE)
  set(Kokkos_ENABLE_OPENMP OFF CACHE BOOL "Disable the OpenMP backend" FORCE)
endif()
set(Kokkos_ARCH_NATIVE ${SOLPROP_NATIVE_HOST_ARCH} CACHE BOOL
    "Optimize Kokkos host code for the build machine" FORCE)
set(Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION
    ${SOLPROP_NATIVE_ENABLE_AGGRESSIVE_VECTORIZATION} CACHE BOOL
    "Enable aggressive vectorization of independent Kokkos loops" FORCE)
set(Kokkos_ENABLE_THREADS OFF CACHE BOOL "OpenMP is the CPU host space" FORCE)
set(Kokkos_ENABLE_CUDA ${SOLPROP_NATIVE_ENABLE_CUDA} CACHE BOOL
    "Enable the CUDA backend" FORCE)
set(Kokkos_ENABLE_HIP ${SOLPROP_NATIVE_ENABLE_HIP} CACHE BOOL
    "Enable the HIP backend" FORCE)
set(Kokkos_ENABLE_SYCL OFF CACHE BOOL "SYCL is outside the supported backends" FORCE)
set(Kokkos_ENABLE_TESTS OFF CACHE BOOL "Do not build Kokkos tests" FORCE)

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
  PATCH_COMMAND "${CMAKE_COMMAND}" -P "${PROJECT_SOURCE_DIR}/cmake/KokkosKernelsWindowsArm.cmake"
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

include(cmake/VectorMath.cmake)

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
