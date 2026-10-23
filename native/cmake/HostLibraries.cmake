# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Host library and build-profile choices. A LOCAL build (the default) links a vendor
# BLAS and compiles for the build machine. A DISTRIBUTION build compiles for a CPU
# baseline and links a BLAS every machine of its platform has: Accelerate on macOS, the
# OpenBLAS on Linux built with OpenMP on the extension's own runtime (cmake/OpenMPOpenBLAS.cmake),
# and on Windows the same OpenBLAS prebuilt static (the wheel) or the scipy-openblas32
# OpenBLAS; each is carried in the wheel. A REFERENCE build has neither a vendor BLAS
# nor machine-specific code.
# The automatic LOCAL order is Accelerate on macOS; on Linux MKL when oneAPI's
# environment names it (MKLROOT, or SOLPROP_NATIVE_MKL_ROOT), otherwise OpenBLAS.
# SOLPROP_NATIVE_BLAS_VENDOR or SOLPROP_NATIVE_BLAS_LIBRARIES overrides the order, and
# SOLPROP_NATIVE_ENABLE_TPL_BLAS=OFF keeps the native-order GEMM in a LOCAL build. An
# automatic search that finds nothing falls back to that GEMM with a warning; an
# explicit request that finds nothing fails. FFT choices are unaffected.
set(SOLPROP_NATIVE_BLAS_VENDOR "AUTO" CACHE STRING
    "Host BLAS selection: AUTO, OPENBLAS, MKL, or ACCELERATE")
set_property(CACHE SOLPROP_NATIVE_BLAS_VENDOR PROPERTY STRINGS
             AUTO OPENBLAS MKL ACCELERATE)
set(SOLPROP_NATIVE_FFT_PROVIDER "FFTW" CACHE STRING
    "Host FFT implementation: FFTW or MKL_FFTW")
set_property(CACHE SOLPROP_NATIVE_FFT_PROVIDER PROPERTY STRINGS FFTW MKL_FFTW)
set(SOLPROP_NATIVE_FFTW_THREADING "OPENMP" CACHE STRING
    "FFTW thread library: OPENMP or PTHREADS")
set_property(CACHE SOLPROP_NATIVE_FFTW_THREADING PROPERTY STRINGS OPENMP PTHREADS)
set(_solprop_raman_default "KOKKOSFFT")
if(APPLE AND NOT SOLPROP_NATIVE_ENABLE_CUDA AND NOT SOLPROP_NATIVE_ENABLE_HIP)
  # vDSP handles the large-field and parallel Raman workloads. Short serial
  # transforms use measured FFTW in the adapter; keep the explicit build override.
  set(_solprop_raman_default "ACCELERATE")
endif()
set(SOLPROP_NATIVE_RAMAN_FFT_PROVIDER "${_solprop_raman_default}" CACHE STRING
    "Raman FFT selection: KOKKOSFFT or macOS ACCELERATE")
set_property(CACHE SOLPROP_NATIVE_RAMAN_FFT_PROVIDER PROPERTY STRINGS KOKKOSFFT ACCELERATE)
if(NOT SOLPROP_NATIVE_RAMAN_FFT_PROVIDER MATCHES "^(KOKKOSFFT|ACCELERATE)$")
  message(FATAL_ERROR "Invalid SOLPROP_NATIVE_RAMAN_FFT_PROVIDER.")
endif()
if(SOLPROP_NATIVE_RAMAN_FFT_PROVIDER STREQUAL "ACCELERATE" AND
   (NOT APPLE OR SOLPROP_NATIVE_ENABLE_CUDA OR SOLPROP_NATIVE_ENABLE_HIP))
  message(FATAL_ERROR "Accelerate Raman FFT requires a macOS CPU build.")
endif()
set(SOLPROP_NATIVE_MKL_ROOT "$ENV{MKLROOT}" CACHE PATH "Optional oneMKL root")
set(SOLPROP_NATIVE_DISTRIBUTION_BLAS "AUTO" CACHE STRING
    "Distribution BLAS on Linux and Windows: AUTO, OPENMP_OPENBLAS or SCIPY_OPENBLAS")
set_property(CACHE SOLPROP_NATIVE_DISTRIBUTION_BLAS PROPERTY STRINGS
             AUTO OPENMP_OPENBLAS SCIPY_OPENBLAS)
if(NOT SOLPROP_NATIVE_DISTRIBUTION_BLAS MATCHES "^(AUTO|OPENMP_OPENBLAS|SCIPY_OPENBLAS)$")
  message(FATAL_ERROR "Invalid SOLPROP_NATIVE_DISTRIBUTION_BLAS.")
endif()

if(NOT SOLPROP_NATIVE_BLAS_VENDOR MATCHES "^(AUTO|OPENBLAS|MKL|ACCELERATE)$")
  message(FATAL_ERROR "Invalid SOLPROP_NATIVE_BLAS_VENDOR.")
endif()
foreach(_switch IN ITEMS SOLPROP_NATIVE_ENABLE_TPL_BLAS SOLPROP_NATIVE_HOST_ARCH)
  string(TOUPPER "${${_switch}}" _value)
  if(_value MATCHES "^(1|TRUE|YES|Y)$")
    set(_value ON)
  elseif(_value MATCHES "^(0|FALSE|NO|N)$")
    set(_value OFF)
  endif()
  if(NOT _value MATCHES "^(AUTO|ON|OFF)$")
    message(FATAL_ERROR "${_switch} must be AUTO, ON or OFF; got ${${_switch}}.")
  endif()
  set(_solprop_request_${_switch} "${_value}")
endforeach()
set(_tpl_request "${_solprop_request_SOLPROP_NATIVE_ENABLE_TPL_BLAS}")
set(_native_request "${_solprop_request_SOLPROP_NATIVE_HOST_ARCH}")
# The resolved values below are normal variables shadowing the cache entries for the
# rest of this project's configuration.
string(TOUPPER "${SOLPROP_NATIVE_PROFILE}" SOLPROP_NATIVE_PROFILE)
if(NOT SOLPROP_NATIVE_PROFILE MATCHES "^(LOCAL|DISTRIBUTION|REFERENCE)$")
  message(FATAL_ERROR
    "SOLPROP_NATIVE_PROFILE must be LOCAL, DISTRIBUTION or REFERENCE; "
    "got ${SOLPROP_NATIVE_PROFILE}.")
endif()
string(TOLOWER "${SOLPROP_NATIVE_PROFILE}" SOLPROP_NATIVE_BUILD_PROFILE)
set(_device_build OFF)
if(SOLPROP_NATIVE_ENABLE_CUDA OR SOLPROP_NATIVE_ENABLE_HIP)
  set(_device_build ON)
endif()
if(_device_build AND SOLPROP_NATIVE_PROFILE STREQUAL "DISTRIBUTION")
  message(FATAL_ERROR
    "SOLPROP_NATIVE_PROFILE=DISTRIBUTION is a CPU profile; a LOCAL device build "
    "already links the device vendor BLAS and keeps its host code generic.")
endif()
if(NOT SOLPROP_NATIVE_PROFILE STREQUAL "LOCAL")
  if(_native_request STREQUAL "ON")
    message(FATAL_ERROR
      "A ${SOLPROP_NATIVE_BUILD_PROFILE} build runs on other machines; it cannot be "
      "combined with ENABLE_NATIVE_ARCH=ON.")
  endif()
  if(NOT SOLPROP_NATIVE_BLAS_VENDOR STREQUAL "AUTO" OR SOLPROP_NATIVE_BLAS_LIBRARIES)
    message(FATAL_ERROR
      "A ${SOLPROP_NATIVE_BUILD_PROFILE} build chooses its own BLAS; "
      "unset BLAS_VENDOR/BLAS_LIBRARIES.")
  endif()
endif()
if(SOLPROP_NATIVE_PROFILE STREQUAL "REFERENCE" AND _tpl_request STREQUAL "ON")
  message(FATAL_ERROR
    "SOLPROP_NATIVE_PROFILE=REFERENCE builds without a vendor BLAS; it cannot be "
    "combined with ENABLE_TPL_BLAS=ON.")
endif()
if(SOLPROP_NATIVE_PROFILE STREQUAL "DISTRIBUTION" AND _tpl_request STREQUAL "OFF")
  message(FATAL_ERROR
    "SOLPROP_NATIVE_PROFILE=DISTRIBUTION always links its BLAS; it cannot be "
    "combined with ENABLE_TPL_BLAS=OFF.")
endif()
if(_native_request STREQUAL "AUTO")
  # Device builds keep their host code generic, as their qualified presets did.
  if(SOLPROP_NATIVE_PROFILE STREQUAL "LOCAL" AND NOT _device_build)
    set(SOLPROP_NATIVE_HOST_ARCH ON)
  else()
    set(SOLPROP_NATIVE_HOST_ARCH OFF)
  endif()
else()
  set(SOLPROP_NATIVE_HOST_ARCH "${_native_request}")
endif()

# The CPU baseline is the oldest processor a non-native build runs on. The
# distribution defaults are measured choices. x86-64 takes x86-64-v3 (AVX2, FMA, BMI1/2:
# Intel Haswell, 2013, and AMD Excavator, 2015, or newer). An x86-64-v2 build was up to
# 2 to 4 percent slower on the benchmark core and, on the general modal route, 23 to 24
# percent slower at one thread and 7 to 12 percent at full width (an
# 18-core-per-socket AVX-512 server; solprop SEP 29). v3 leaves out older cluster nodes
# and the low-power Atom, Celeron and Pentium chips without AVX, where solprop's
# automatic selection falls back to numba_cpu; SOLPROP_NATIVE_CPU_BASELINE=x86-64-v2
# builds for them from source. FFTW and OpenBLAS select their kernels at run time
# either way. Every Apple silicon Mac is at least an M1. Linux aarch64 takes Neoverse
# N1 (Armv8.2-A with the dot-product and half-precision extensions: Graviton2, Ampere
# Altra and every later server core; solprop SEP 29). Other platforms keep the
# compiler's default. The package refuses to import on a CPU below its baseline.
set(_x86_processor OFF)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
  set(_x86_processor ON)
endif()
string(STRIP "${SOLPROP_NATIVE_CPU_BASELINE}" _baseline_request)
string(TOUPPER "${_baseline_request}" _baseline_upper)
if(_baseline_upper STREQUAL "AUTO")
  set(SOLPROP_NATIVE_CPU_BASELINE "none")
  if(SOLPROP_NATIVE_PROFILE STREQUAL "DISTRIBUTION")
    if(_x86_processor)
      set(SOLPROP_NATIVE_CPU_BASELINE "x86-64-v3")
    elseif(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
      set(SOLPROP_NATIVE_CPU_BASELINE "apple-m1")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
           CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
      set(SOLPROP_NATIVE_CPU_BASELINE "neoverse-n1")
    endif()
  endif()
elseif(_baseline_upper STREQUAL "NONE" OR _baseline_request STREQUAL "")
  set(SOLPROP_NATIVE_CPU_BASELINE "none")
else()
  if(NOT _baseline_request MATCHES "^[A-Za-z0-9._+-]+$")
    message(FATAL_ERROR
      "SOLPROP_NATIVE_CPU_BASELINE must be AUTO, NONE or an architecture name; "
      "got '${_baseline_request}'.")
  endif()
  if(SOLPROP_NATIVE_PROFILE STREQUAL "REFERENCE" OR _device_build)
    message(FATAL_ERROR
      "SOLPROP_NATIVE_CPU_BASELINE applies to LOCAL and DISTRIBUTION CPU builds.")
  endif()
  set(SOLPROP_NATIVE_CPU_BASELINE "${_baseline_request}")
endif()
set(SOLPROP_NATIVE_CPU_BASELINE_FLAG "")
if(NOT SOLPROP_NATIVE_CPU_BASELINE STREQUAL "none")
  if(SOLPROP_NATIVE_HOST_ARCH)
    message(FATAL_ERROR
      "SOLPROP_NATIVE_CPU_BASELINE=${SOLPROP_NATIVE_CPU_BASELINE} and "
      "ENABLE_NATIVE_ARCH=ON both choose the architecture; set one.")
  endif()
  if(_x86_processor)
    set(SOLPROP_NATIVE_CPU_BASELINE_FLAG "-march=${SOLPROP_NATIVE_CPU_BASELINE}")
  else()
    set(SOLPROP_NATIVE_CPU_BASELINE_FLAG "-mcpu=${SOLPROP_NATIVE_CPU_BASELINE}")
  endif()
  include(CheckCXXCompilerFlag)
  string(MAKE_C_IDENTIFIER "${SOLPROP_NATIVE_CPU_BASELINE_FLAG}" _baseline_id)
  check_cxx_compiler_flag("${SOLPROP_NATIVE_CPU_BASELINE_FLAG}"
                          SOLPROP_NATIVE_COMPILER_ACCEPTS${_baseline_id})
  if(NOT SOLPROP_NATIVE_COMPILER_ACCEPTS${_baseline_id})
    message(FATAL_ERROR
      "The compiler does not accept the CPU baseline ${SOLPROP_NATIVE_CPU_BASELINE_FLAG}.")
  endif()
endif()
if(NOT SOLPROP_NATIVE_FFT_PROVIDER MATCHES "^(FFTW|MKL_FFTW)$")
  message(FATAL_ERROR "Invalid SOLPROP_NATIVE_FFT_PROVIDER.")
endif()
if(NOT SOLPROP_NATIVE_FFTW_THREADING MATCHES "^(OPENMP|PTHREADS)$")
  message(FATAL_ERROR "Invalid SOLPROP_NATIVE_FFTW_THREADING.")
endif()
if(SOLPROP_NATIVE_ENABLE_CUDA OR SOLPROP_NATIVE_ENABLE_HIP)
  if(NOT SOLPROP_NATIVE_BLAS_VENDOR STREQUAL "AUTO" OR
     NOT SOLPROP_NATIVE_FFT_PROVIDER STREQUAL "FFTW")
    message(FATAL_ERROR "Host BLAS/FFT selections cannot be used in device builds.")
  endif()
  # The device's vendor BLAS (cuBLAS, rocBLAS) is the local default; KokkosKernels
  # finds it with the toolkit and a missing one fails there, as before.
  if(_tpl_request STREQUAL "AUTO")
    if(SOLPROP_NATIVE_PROFILE STREQUAL "REFERENCE")
      set(SOLPROP_NATIVE_ENABLE_TPL_BLAS OFF)
      set(SOLPROP_NATIVE_BLAS_SELECTION "disabled: reference build")
    else()
      set(SOLPROP_NATIVE_ENABLE_TPL_BLAS ON)
      set(SOLPROP_NATIVE_BLAS_SELECTION "automatic: the device vendor BLAS")
    endif()
  elseif(_tpl_request STREQUAL "ON")
    set(SOLPROP_NATIVE_ENABLE_TPL_BLAS ON)
    set(SOLPROP_NATIVE_BLAS_SELECTION "explicit: SOLPROP_NATIVE_ENABLE_TPL_BLAS=ON")
  else()
    set(SOLPROP_NATIVE_ENABLE_TPL_BLAS OFF)
    set(SOLPROP_NATIVE_BLAS_SELECTION "disabled: SOLPROP_NATIVE_ENABLE_TPL_BLAS=OFF")
  endif()
  return()
endif()
if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM" AND
   SOLPROP_NATIVE_ENABLE_OPENMP AND
   SOLPROP_NATIVE_FFT_PROVIDER STREQUAL "FFTW" AND
   SOLPROP_NATIVE_FFTW_THREADING STREQUAL "OPENMP")
  message(FATAL_ERROR
    "IntelLLVM/FFTW CPU builds require FFTW_THREADING=PTHREADS. "
    "Distribution OpenMP FFTW libraries link GNU OpenMP, not Intel's runtime.")
endif()
if(_tpl_request STREQUAL "OFF" AND
   (NOT SOLPROP_NATIVE_BLAS_VENDOR STREQUAL "AUTO" OR SOLPROP_NATIVE_BLAS_LIBRARIES))
  message(FATAL_ERROR
    "An explicit BLAS_VENDOR or BLAS_LIBRARIES requires ENABLE_TPL_BLAS=AUTO or ON.")
endif()
if(NOT SOLPROP_NATIVE_BLAS_VENDOR STREQUAL "AUTO" AND
   SOLPROP_NATIVE_BLAS_LIBRARIES)
  message(FATAL_ERROR "Select BLAS_VENDOR or explicit BLAS_LIBRARIES, not both.")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
  set(_mkl_thread mkl_intel_thread)
else()
  set(_mkl_thread mkl_gnu_thread)
endif()
if(NOT SOLPROP_NATIVE_ENABLE_OPENMP)
  set(_mkl_thread mkl_sequential)
endif()
set(_mkl_components mkl_intel_lp64 ${_mkl_thread} mkl_core)

# Find the LP64 MKL libraries; REQUIRED fails the configuration, otherwise the
# result says whether all three were found.
function(_solprop_native_find_mkl required result)
  set(_found ON)
  set(_libraries)
  foreach(_component IN LISTS _mkl_components)
    if(required)
      find_library(SOLPROP_NATIVE_${_component}_LIBRARY NAMES ${_component}
        HINTS "${SOLPROP_NATIVE_MKL_ROOT}" PATH_SUFFIXES lib lib/intel64 REQUIRED)
    else()
      find_library(SOLPROP_NATIVE_${_component}_LIBRARY NAMES ${_component}
        HINTS "${SOLPROP_NATIVE_MKL_ROOT}" PATH_SUFFIXES lib lib/intel64)
    endif()
    if(NOT SOLPROP_NATIVE_${_component}_LIBRARY)
      set(_found OFF)
    endif()
    list(APPEND _libraries "${SOLPROP_NATIVE_${_component}_LIBRARY}")
  endforeach()
  set(${result} ${_found} PARENT_SCOPE)
  set(_mkl_libraries "${_libraries}" PARENT_SCOPE)
endfunction()

# Probe one FindBLAS vendor in a function scope, so its result variables do not
# reach KokkosKernels; its cached library entries are what KokkosKernels finds.
function(_solprop_native_probe_blas vendor result)
  set(BLA_VENDOR ${vendor})
  set(BLA_SIZEOF_INTEGER 4)
  find_package(BLAS QUIET)
  set(${result} ${BLAS_FOUND} PARENT_SCOPE)
endfunction()

# Locate the scipy-openblas32 wheel in the build's Python: its LP64 OpenBLAS selects
# its kernels at run time, and its symbols carry a scipy_ prefix so that it never
# collides with NumPy's own copy. The distribution build links it and copies the
# library directory's shared objects (OpenBLAS and its Fortran runtime, which it finds
# through its $ORIGIN runpath) and its licence into the package.
function(_solprop_native_find_scipy_openblas)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c [=[
import importlib.metadata as metadata
import scipy_openblas32 as blas
distribution = metadata.distribution("scipy-openblas32")
licences = [f for f in distribution.files or () if f.name.upper().startswith("LICENSE")]
print(blas.get_lib_dir())
print(blas.get_library())
print(distribution.version)
print(distribution.locate_file(licences[0]) if licences else "(no licence file)")
]=]
    OUTPUT_VARIABLE _query
    ERROR_VARIABLE _query_error
    RESULT_VARIABLE _query_result
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT _query_result EQUAL 0)
    message(FATAL_ERROR
      "SOLPROP_NATIVE_PROFILE=DISTRIBUTION on Linux and Windows links the scipy-openblas32 "
      "OpenBLAS, which ${Python_EXECUTABLE} cannot import. Install scipy-openblas32 "
      "in the build environment (an isolated build installs it from the package's "
      "build requirements).\n${_query_error}")
  endif()
  string(REPLACE "\n" ";" _query "${_query}")
  list(GET _query 0 _directory)
  list(GET _query 1 _name)
  list(GET _query 2 _version)
  list(GET _query 3 _licence)
  # Windows paths come back with backslashes, which CMake's install code reads as escapes.
  file(TO_CMAKE_PATH "${_directory}" _directory)
  file(TO_CMAKE_PATH "${_licence}" _licence)
  if(WIN32)
    # The Windows wheel names its import library after the DLL (libscipy_openblas).
    set(_library "${_directory}/${_name}.lib")
    file(GLOB _vendored LIST_DIRECTORIES false "${_directory}/*.dll")
  else()
    set(_library "${_directory}/lib${_name}${CMAKE_SHARED_LIBRARY_SUFFIX}")
    file(GLOB _vendored LIST_DIRECTORIES false "${_directory}/*.so*")
  endif()
  if(NOT EXISTS "${_library}" OR NOT EXISTS "${_licence}")
    message(FATAL_ERROR
      "scipy-openblas32 ${_version} lacks ${_library} or its licence file.")
  endif()
  set(SOLPROP_NATIVE_BLAS_LIBRARIES "${_library}" PARENT_SCOPE)
  set(SOLPROP_NATIVE_VENDORED_LIBRARIES "${_vendored}" PARENT_SCOPE)
  set(SOLPROP_NATIVE_VENDORED_LICENCE "${_licence}" PARENT_SCOPE)
  set(_scipy_openblas_version "${_version}" PARENT_SCOPE)
endfunction()

set(SOLPROP_NATIVE_ENABLE_TPL_BLAS OFF)
set(SOLPROP_NATIVE_BLAS_SYMBOL_PREFIX "")
set(SOLPROP_NATIVE_VENDORED_LIBRARIES)
set(SOLPROP_NATIVE_BLAS_STATIC OFF)
set(SOLPROP_NATIVE_VENDORED_LICENCE "")
set(SOLPROP_NATIVE_VENDORED_LICENCE_NAME "")
if(SOLPROP_NATIVE_PROFILE STREQUAL "REFERENCE")
  set(SOLPROP_NATIVE_BLAS_SELECTION "disabled: reference build")
elseif(SOLPROP_NATIVE_PROFILE STREQUAL "DISTRIBUTION")
  set(SOLPROP_NATIVE_ENABLE_TPL_BLAS ON)
  if(APPLE)
    _solprop_native_probe_blas(Apple _blas_found)
    if(NOT _blas_found)
      message(FATAL_ERROR "The distribution build could not find Accelerate.")
    endif()
    set(SOLPROP_NATIVE_BLAS_VENDOR ACCELERATE)
    # Accelerate's legacy BLAS symbols are the Fortran ones (dgemm_). Declared here so
    # that Kokkos Kernels does not probe a Fortran compiler for the convention
    # (kokkoskernels_tpls.cmake), which a build machine need not have.
    set(F77_BLAS_MANGLE "(name,NAME) name ## _")
    set(SOLPROP_NATIVE_BLAS_SELECTION "distribution: Accelerate (macOS)")
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" OR WIN32)
    set(_distribution_blas "${SOLPROP_NATIVE_DISTRIBUTION_BLAS}")
    if(_distribution_blas STREQUAL "AUTO")
      # On Windows AUTO keeps scipy-openblas32, which the package holds to one thread
      # beside the Kokkos pool; the wheel recipe selects OPENMP_OPENBLAS explicitly,
      # with a prebuilt static library (native/ci/build-deps-windows.ps1).
      if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND SOLPROP_NATIVE_ENABLE_OPENMP)
        set(_distribution_blas OPENMP_OPENBLAS)
      else()
        set(_distribution_blas SCIPY_OPENBLAS)
      endif()
    endif()
    if(_distribution_blas STREQUAL "OPENMP_OPENBLAS")
      if(NOT SOLPROP_NATIVE_ENABLE_OPENMP)
        message(FATAL_ERROR
          "SOLPROP_NATIVE_DISTRIBUTION_BLAS=OPENMP_OPENBLAS needs an OpenMP build.")
      endif()
      include(${CMAKE_CURRENT_LIST_DIR}/OpenMPOpenBLAS.cmake)
      if(WIN32)
        _solprop_native_openmp_openblas_windows()
        set(F77_BLAS_MANGLE "(name,NAME) name ## _")
        set(SOLPROP_NATIVE_BLAS_STATIC ON)
        set(SOLPROP_NATIVE_BLAS_SELECTION
            "distribution: static OpenMP OpenBLAS ${SOLPROP_NATIVE_OPENBLAS_VERSION} on the extension's OpenMP runtime, linked into the extension")
      else()
        enable_language(C)
        _solprop_native_openmp_openblas()
        # Kokkos Kernels declares its BLAS prototypes through this macro; a preset value
        # takes precedence over the Fortran-interface probe (kokkoskernels_tpls.cmake).
        set(F77_BLAS_MANGLE "(name,NAME) solprop_ ## name ## _")
        set(SOLPROP_NATIVE_BLAS_SYMBOL_PREFIX "solprop_")
        set(SOLPROP_NATIVE_BLAS_SELECTION
            "distribution: OpenMP OpenBLAS ${SOLPROP_NATIVE_OPENBLAS_VERSION} on the extension's OpenMP runtime, copied into the package")
      endif()
    else()
      _solprop_native_find_scipy_openblas()
      set(F77_BLAS_MANGLE "(name,NAME) scipy_ ## name ## _")
      set(SOLPROP_NATIVE_BLAS_SYMBOL_PREFIX "scipy_")
      set(SOLPROP_NATIVE_VENDORED_LICENCE_NAME "LICENSE-scipy-openblas32.txt")
      set(SOLPROP_NATIVE_BLAS_SELECTION
          "distribution: scipy-openblas32 ${_scipy_openblas_version}, copied into the package")
    endif()
  else()
    message(FATAL_ERROR "The distribution build supports Linux, macOS and Windows.")
  endif()
elseif(_tpl_request STREQUAL "OFF")
  set(SOLPROP_NATIVE_BLAS_SELECTION "disabled: SOLPROP_NATIVE_ENABLE_TPL_BLAS=OFF")
elseif(SOLPROP_NATIVE_BLAS_LIBRARIES)
  set(SOLPROP_NATIVE_ENABLE_TPL_BLAS ON)
  set(SOLPROP_NATIVE_BLAS_SELECTION "explicit: SOLPROP_NATIVE_BLAS_LIBRARIES")
elseif(NOT SOLPROP_NATIVE_BLAS_VENDOR STREQUAL "AUTO")
  if(SOLPROP_NATIVE_BLAS_VENDOR STREQUAL "ACCELERATE" AND NOT APPLE)
    message(FATAL_ERROR "Accelerate requires macOS.")
  endif()
  set(SOLPROP_NATIVE_ENABLE_TPL_BLAS ON)
  set(SOLPROP_NATIVE_BLAS_SELECTION
      "explicit: SOLPROP_NATIVE_BLAS_VENDOR=${SOLPROP_NATIVE_BLAS_VENDOR}")
else()
  set(_blas_found OFF)
  if(APPLE)
    _solprop_native_probe_blas(Apple _blas_found)
    set(_blas_vendor ACCELERATE)
    set(_blas_reason "Accelerate (macOS)")
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(SOLPROP_NATIVE_MKL_ROOT)
      _solprop_native_find_mkl(OFF _blas_found)
      set(_blas_vendor MKL)
      set(_blas_reason "MKL (SOLPROP_NATIVE_MKL_ROOT=${SOLPROP_NATIVE_MKL_ROOT})")
    endif()
    if(NOT _blas_found)
      _solprop_native_probe_blas(OpenBLAS _blas_found)
      set(_blas_vendor OPENBLAS)
      set(_blas_reason "OpenBLAS")
    endif()
  endif()
  if(_blas_found)
    set(SOLPROP_NATIVE_ENABLE_TPL_BLAS ON)
    set(SOLPROP_NATIVE_BLAS_VENDOR "${_blas_vendor}")
    set(SOLPROP_NATIVE_BLAS_SELECTION "automatic: ${_blas_reason}")
  elseif(_tpl_request STREQUAL "ON")
    message(FATAL_ERROR
      "SOLPROP_NATIVE_ENABLE_TPL_BLAS=ON found no vendor BLAS (Accelerate on macOS; "
      "MKL with MKLROOT set, or OpenBLAS, on Linux). Name one with "
      "SOLPROP_NATIVE_BLAS_VENDOR, SOLPROP_NATIVE_BLAS_ROOT or SOLPROP_NATIVE_BLAS_LIBRARIES.")
  else()
    set(SOLPROP_NATIVE_BLAS_SELECTION "automatic: none found, native-order GEMM")
    message(WARNING
      "solprop-native: no vendor BLAS was found (Accelerate on macOS; MKL with "
      "MKLROOT set, or OpenBLAS, on Linux). This local build falls back to the "
      "native-order GEMM, which is slower on multimode transforms. Install "
      "OpenBLAS or source oneAPI, or name a provider with SOLPROP_NATIVE_BLAS_VENDOR, "
      "SOLPROP_NATIVE_BLAS_ROOT or SOLPROP_NATIVE_BLAS_LIBRARIES.")
  endif()
endif()

if((SOLPROP_NATIVE_ENABLE_TPL_BLAS AND SOLPROP_NATIVE_BLAS_VENDOR STREQUAL "MKL") OR
   SOLPROP_NATIVE_FFT_PROVIDER STREQUAL "MKL_FFTW")
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "The optional MKL configuration requires Linux.")
  endif()
  _solprop_native_find_mkl(ON _mkl_found)
endif()

if(SOLPROP_NATIVE_ENABLE_TPL_BLAS)
  if(DEFINED BLA_SIZEOF_INTEGER AND NOT BLA_SIZEOF_INTEGER STREQUAL "4")
    message(FATAL_ERROR "Host BLAS requires LP64 (BLA_SIZEOF_INTEGER=4).")
  endif()
  set(BLA_SIZEOF_INTEGER 4)
  if(SOLPROP_NATIVE_BLAS_VENDOR STREQUAL "MKL")
    set(SOLPROP_NATIVE_BLAS_LIBRARIES "${_mkl_libraries}")
  elseif(SOLPROP_NATIVE_BLAS_VENDOR STREQUAL "OPENBLAS")
    set(BLA_VENDOR OpenBLAS)
  elseif(SOLPROP_NATIVE_BLAS_VENDOR STREQUAL "ACCELERATE")
    set(BLA_VENDOR Apple)
  endif()
endif()

set(SOLPROP_NATIVE_HOST_FFT_PLANNER "FFTW_ESTIMATE")
if(SOLPROP_NATIVE_FFT_PROVIDER STREQUAL "MKL_FFTW")
  find_path(SOLPROP_NATIVE_MKL_FFT_INCLUDE fftw3.h
    HINTS "${SOLPROP_NATIVE_MKL_ROOT}" PATH_SUFFIXES include/fftw REQUIRED)
  set(FFTW_INCLUDE_DIRS "${SOLPROP_NATIVE_MKL_FFT_INCLUDE}" CACHE PATH "FFTW API headers" FORCE)
  # Kokkos FFT uses the same FFTW C API for both thread implementations.
  # MKL exports both precisions and the planning-thread controls from its LP64
  # interface; its dependency libraries are linked after the FFT/BLAS targets.
  foreach(_component IN ITEMS DOUBLE_LIB FLOAT_LIB DOUBLE_OPENMP_LIB FLOAT_OPENMP_LIB)
    set(FFTW_${_component} "${SOLPROP_NATIVE_mkl_intel_lp64_LIBRARY}"
        CACHE FILEPATH "MKL FFTW compatibility library" FORCE)
  endforeach()
  set(SOLPROP_NATIVE_MKL_FFT_LIBRARIES "${_mkl_libraries}")
  set(SOLPROP_NATIVE_HOST_FFT_BACKEND "MKL_FFTW")
  set(SOLPROP_NATIVE_HOST_FFT_PLANNER "MKL_ignores_FFTW_planning_flags")
elseif(SOLPROP_NATIVE_ENABLE_THREADS)
  # Kokkos FFT links FFTW's threads library for the Threads host space.
  set(SOLPROP_NATIVE_HOST_FFT_BACKEND "FFTW_THREADS")
elseif(SOLPROP_NATIVE_ENABLE_OPENMP)
  if(SOLPROP_NATIVE_FFTW_THREADING STREQUAL "PTHREADS")
    find_library(SOLPROP_NATIVE_FFTW_DOUBLE_PTHREADS fftw3_threads
      HINTS "${FFTW_ROOT}" PATH_SUFFIXES lib lib64 REQUIRED)
    find_library(SOLPROP_NATIVE_FFTW_FLOAT_PTHREADS fftw3f_threads
      HINTS "${FFTW_ROOT}" PATH_SUFFIXES lib lib64 REQUIRED)
    set(FFTW_DOUBLE_OPENMP_LIB "${SOLPROP_NATIVE_FFTW_DOUBLE_PTHREADS}"
        CACHE FILEPATH "FFTW pthread implementation of the thread API" FORCE)
    set(FFTW_FLOAT_OPENMP_LIB "${SOLPROP_NATIVE_FFTW_FLOAT_PTHREADS}"
        CACHE FILEPATH "FFTW pthread implementation of the thread API" FORCE)
    set(SOLPROP_NATIVE_HOST_FFT_BACKEND "FFTW_PTHREADS")
  else()
    set(SOLPROP_NATIVE_HOST_FFT_BACKEND "FFTW_OPENMP")
  endif()
else()
  set(SOLPROP_NATIVE_HOST_FFT_BACKEND "FFTW_SERIAL")
endif()
