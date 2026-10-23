# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# The distribution build's Linux BLAS: OpenBLAS built with OpenMP by the same compiler
# as the extension, so that it runs on the extension's own OpenMP runtime instead of
# keeping a second thread pool beside Kokkos's. A pthreads OpenBLAS beside the Kokkos
# pool oversubscribes the cores: on a 32-core Zen 5 the modal and wall cases ran 10 to
# 75 times slower than with this build, which needs no settings and is level with a
# one-thread BLAS at one thread (solprop SEP 29, phase 0).
#
# DYNAMIC_ARCH selects the kernels at run time; the common code is compiled for a
# processor at or below the wheel's CPU baseline (Nehalem, x86-64-v2, on x86-64, below
# the wheel's x86-64-v3, as qualified in solprop SEP 29; ARMv8 on aarch64). The
# symbols and the library carry a solprop_ prefix so that the copy never collides with
# NumPy's or a system OpenBLAS in the same process. LAPACK is OpenBLAS's C translation:
# no Fortran compiler or runtime is needed.
#
# SOLPROP_NATIVE_OPENBLAS_ROOT names a prefix that already holds such a build (lanes
# and CI cache it); otherwise the pinned source is downloaded, checked and built once
# into the build tree.
set(SOLPROP_NATIVE_OPENBLAS_VERSION "0.3.34")
set(SOLPROP_NATIVE_OPENBLAS_SHA256
    "cd7e129868320cc2d033afa920e31202dfe0b8066a5b66661900ccc0f197dfed")
set(SOLPROP_NATIVE_OPENBLAS_ROOT "$ENV{SOLPROP_NATIVE_OPENBLAS_ROOT}" CACHE PATH
    "Prefix of a prebuilt OpenMP OpenBLAS (solprop_ prefix) for the distribution build")

# Windows: OpenBLAS's make build does not run there, and its CMake build's DLL exports
# no symbols under Clang, so the library is a static OpenMP build on the same LLVM
# libomp as the extension (native/ci/build-deps-windows.ps1), linked into the
# extension and named by SOLPROP_NATIVE_OPENBLAS_ROOT: lib/openblas.lib and LICENSE.
# Its symbols stay unprefixed: a static copy is invisible to other modules.
function(_solprop_native_openmp_openblas_windows)
  set(_root "${SOLPROP_NATIVE_OPENBLAS_ROOT}")
  if(NOT _root OR NOT EXISTS "${_root}/lib/openblas.lib" OR NOT EXISTS "${_root}/LICENSE")
    message(FATAL_ERROR
      "SOLPROP_NATIVE_DISTRIBUTION_BLAS=OPENMP_OPENBLAS on Windows needs "
      "SOLPROP_NATIVE_OPENBLAS_ROOT to name a static OpenMP OpenBLAS "
      "(lib/openblas.lib and LICENSE; native/ci/build-deps-windows.ps1 builds one).")
  endif()
  file(TO_CMAKE_PATH "${_root}" _root)
  set(SOLPROP_NATIVE_BLAS_LIBRARIES "${_root}/lib/openblas.lib" PARENT_SCOPE)
  set(SOLPROP_NATIVE_VENDORED_LIBRARIES "" PARENT_SCOPE)
  set(SOLPROP_NATIVE_VENDORED_LICENCE "${_root}/LICENSE" PARENT_SCOPE)
  set(SOLPROP_NATIVE_VENDORED_LICENCE_NAME "LICENSE-openblas.txt" PARENT_SCOPE)
endfunction()

function(_solprop_native_openmp_openblas)
  set(_root "${SOLPROP_NATIVE_OPENBLAS_ROOT}")
  set(_source_dir "${PROJECT_BINARY_DIR}/_openblas/OpenBLAS-${SOLPROP_NATIVE_OPENBLAS_VERSION}")
  if(NOT _root)
    set(_root "${PROJECT_BINARY_DIR}/_openblas/install")
  endif()
  set(_library "${_root}/lib/libsolprop_openblas${CMAKE_SHARED_LIBRARY_SUFFIX}")
  if(NOT EXISTS "${_library}")
    if(SOLPROP_NATIVE_OPENBLAS_ROOT)
      message(FATAL_ERROR
        "SOLPROP_NATIVE_OPENBLAS_ROOT=${SOLPROP_NATIVE_OPENBLAS_ROOT} holds no "
        "lib/libsolprop_openblas${CMAKE_SHARED_LIBRARY_SUFFIX}.")
    endif()
    set(_archive "${PROJECT_BINARY_DIR}/_openblas/OpenBLAS-${SOLPROP_NATIVE_OPENBLAS_VERSION}.tar.gz")
    file(DOWNLOAD
      "https://github.com/OpenMathLib/OpenBLAS/releases/download/v${SOLPROP_NATIVE_OPENBLAS_VERSION}/OpenBLAS-${SOLPROP_NATIVE_OPENBLAS_VERSION}.tar.gz"
      "${_archive}" EXPECTED_HASH SHA256=${SOLPROP_NATIVE_OPENBLAS_SHA256}
      TLS_VERIFY ON STATUS _status)
    list(GET _status 0 _code)
    if(NOT _code EQUAL 0)
      message(FATAL_ERROR
        "Downloading OpenBLAS ${SOLPROP_NATIVE_OPENBLAS_VERSION} failed (${_status}). "
        "Build it once and name it with SOLPROP_NATIVE_OPENBLAS_ROOT, or build offline "
        "with SOLPROP_NATIVE_DISTRIBUTION_BLAS=SCIPY_OPENBLAS.")
    endif()
    file(ARCHIVE_EXTRACT INPUT "${_archive}" DESTINATION "${PROJECT_BINARY_DIR}/_openblas")
    # The C compiler must be the C++ compiler's family, so that both use one OpenMP
    # runtime (GNU libgomp, or LLVM libomp).
    if(NOT CMAKE_C_COMPILER_ID STREQUAL CMAKE_CXX_COMPILER_ID)
      message(FATAL_ERROR
        "The OpenMP OpenBLAS needs a C compiler of the C++ compiler's family "
        "(${CMAKE_CXX_COMPILER_ID}); CMake found ${CMAKE_C_COMPILER_ID} "
        "${CMAKE_C_COMPILER}. Set CC to match CXX.")
    endif()
    set(_cc "${CMAKE_C_COMPILER}")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
      set(_target NEHALEM)
    else()
      set(_target ARMV8)
    endif()
    cmake_host_system_information(RESULT _jobs QUERY NUMBER_OF_PHYSICAL_CORES)
    set(_make_arguments
      CC=${_cc} USE_OPENMP=1 DYNAMIC_ARCH=1 TARGET=${_target} NUM_THREADS=512
      NOFORTRAN=1 C_LAPACK=1 NO_STATIC=1 BUILD_TESTING=0
      LIBNAMEPREFIX=solprop_ SYMBOLPREFIX=solprop_)
    message(STATUS "solprop-native: building OpenMP OpenBLAS ${SOLPROP_NATIVE_OPENBLAS_VERSION} "
                   "with ${_cc} (once per build tree)")
    execute_process(
      COMMAND make -j${_jobs} ${_make_arguments} libs netlib shared
      WORKING_DIRECTORY "${_source_dir}"
      OUTPUT_FILE "${PROJECT_BINARY_DIR}/_openblas/build.log"
      ERROR_FILE "${PROJECT_BINARY_DIR}/_openblas/build.log"
      RESULT_VARIABLE _result)
    if(_result EQUAL 0)
      execute_process(
        COMMAND make ${_make_arguments} PREFIX=${_root} install
        WORKING_DIRECTORY "${_source_dir}"
        OUTPUT_FILE "${PROJECT_BINARY_DIR}/_openblas/install.log"
        ERROR_FILE "${PROJECT_BINARY_DIR}/_openblas/install.log"
        RESULT_VARIABLE _result)
    endif()
    if(NOT _result EQUAL 0 OR NOT EXISTS "${_library}")
      message(FATAL_ERROR
        "Building OpenBLAS failed; see ${PROJECT_BINARY_DIR}/_openblas/build.log "
        "and install.log.")
    endif()
    file(COPY_FILE "${_source_dir}/LICENSE" "${_root}/LICENSE")
  endif()
  if(NOT EXISTS "${_root}/LICENSE")
    message(FATAL_ERROR
      "${_root} lacks OpenBLAS's LICENSE, which the wheel must carry; copy it there.")
  endif()
  # Copy the library under its soname, the name the extension records, for the wheel.
  file(REAL_PATH "${_library}" _real)
  set(_staging "${PROJECT_BINARY_DIR}/_openblas/vendored")
  file(MAKE_DIRECTORY "${_staging}")
  file(COPY_FILE "${_real}" "${_staging}/libsolprop_openblas.so.0" ONLY_IF_DIFFERENT)
  set(_licence "${_root}/LICENSE")
  set(SOLPROP_NATIVE_BLAS_LIBRARIES "${_library}" PARENT_SCOPE)
  set(SOLPROP_NATIVE_VENDORED_LIBRARIES "${_staging}/libsolprop_openblas.so.0" PARENT_SCOPE)
  set(SOLPROP_NATIVE_VENDORED_LICENCE "${_licence}" PARENT_SCOPE)
  set(SOLPROP_NATIVE_VENDORED_LICENCE_NAME "LICENSE-openblas.txt" PARENT_SCOPE)
endfunction()
