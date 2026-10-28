# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Host vector maths for the CPU kernels' transcendental passes (solprop ADR-059,
# include/solprop_native/vmath.hpp): SLEEF's vector exp, log and sincos, its 1.0 ulp
# (u10) functions, called over contiguous blocks. glibc's scalar exp, cos and sin cost a
# complex exponential 10 to 30 ns on an AMD EPYC 7763 and 9.8 ns on an Intel Xeon Gold
# 6240, and SLEEF's AVX2 and AVX-512 functions take it 3 to 3.6 times faster at the same
# worst error on the kernels' own arguments; on Apple silicon the system libm is already
# about 3 ns and SLEEF's AdvSIMD functions gain nothing.
#
# AUTO takes SLEEF on x86-64 CPU builds and the system libm on Arm (Apple silicon, where
# it was measured, and Linux aarch64, not yet measured) and in device builds, whose
# kernels keep their own functions. SLEEF requests it on Arm too (its AdvSIMD
# functions); SCALAR keeps the system libm everywhere. The variant follows the code's
# target: AVX-512F where a local build targets it, AVX2 with FMA (the distribution's
# x86-64-v3), else SSE2; AdvSIMD on aarch64.
#
# SLEEF 3.9.0 is fetched like Kokkos (FETCHCONTENT_SOURCE_DIR_SLEEF names a local copy
# for an offline build) and linked static.
set(SOLPROP_NATIVE_SLEEF_VERSION "3.9.0")
set(SOLPROP_NATIVE_VECTOR_MATH "AUTO" CACHE STRING
    "Host vector maths for the transcendental passes: AUTO, SLEEF or SCALAR")
set_property(CACHE SOLPROP_NATIVE_VECTOR_MATH PROPERTY STRINGS AUTO SLEEF SCALAR)
string(TOUPPER "${SOLPROP_NATIVE_VECTOR_MATH}" _vector_math_request)
if(NOT _vector_math_request MATCHES "^(AUTO|SLEEF|SCALAR)$")
  message(FATAL_ERROR
    "SOLPROP_NATIVE_VECTOR_MATH must be AUTO, SLEEF or SCALAR; "
    "got ${SOLPROP_NATIVE_VECTOR_MATH}.")
endif()
set(_vector_math_arm OFF)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
  set(_vector_math_arm ON)
endif()
if(_device_build)
  if(_vector_math_request STREQUAL "SLEEF")
    message(FATAL_ERROR
      "SOLPROP_NATIVE_VECTOR_MATH=SLEEF is for CPU builds; device kernels keep their "
      "own functions.")
  endif()
  set(SOLPROP_NATIVE_VECTOR_MATH "SCALAR")
elseif(_vector_math_request STREQUAL "AUTO")
  if(_x86_processor)
    set(SOLPROP_NATIVE_VECTOR_MATH "SLEEF")
  else()
    set(SOLPROP_NATIVE_VECTOR_MATH "SCALAR")
  endif()
elseif(_vector_math_request STREQUAL "SLEEF" AND NOT _x86_processor AND
       NOT _vector_math_arm)
  message(FATAL_ERROR
    "SOLPROP_NATIVE_VECTOR_MATH=SLEEF supports x86-64 and aarch64; this is "
    "${CMAKE_SYSTEM_PROCESSOR}.")
else()
  set(SOLPROP_NATIVE_VECTOR_MATH "${_vector_math_request}")
endif()

# SLEEF builds as its own CMake project, at build time: its generic target names
# (`common`, `headers`) would collide with Kokkos-FFT's in this one, and its flags stay
# its own. It compiles itself with -ffp-contract=off, which its double-double steps
# need (every product and sum rounded separately), and each variant with that
# variant's instruction set; this project's contraction flags, IPO and sanitizers
# never reach it. FetchContent only fetches the pinned source (SOURCE_SUBDIR names a
# directory without a CMakeLists.txt, so nothing is added here).
function(_solprop_native_add_sleef)
  FetchContent_Declare(
    sleef
    GIT_REPOSITORY https://github.com/shibatch/sleef.git
    GIT_TAG ${SOLPROP_NATIVE_SLEEF_VERSION}
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
    SOURCE_SUBDIR solprop-native-builds-this-separately
  )
  FetchContent_MakeAvailable(sleef)
  enable_language(C)
  include(ExternalProject)
  set(_binary "${PROJECT_BINARY_DIR}/_sleef")
  set(_library
      "${_binary}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}sleef${CMAKE_STATIC_LIBRARY_SUFFIX}")
  set(_arguments
    -DCMAKE_BUILD_TYPE=Release
    "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
    "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    # Only libsleef, with the variants vmath.hpp can call: none of its tests, DFT, quad
    # or GNU ABI libraries, and no TLFloat, which it would otherwise download.
    -DSLEEF_BUILD_SHARED_LIBS=OFF -DSLEEF_BUILD_TESTS=OFF -DSLEEF_BUILD_GNUABI_LIBS=OFF
    -DSLEEF_BUILD_DFT=OFF -DSLEEF_BUILD_QUAD=OFF -DSLEEF_BUILD_SCALAR_LIB=OFF
    -DSLEEF_BUILD_INLINE_HEADERS=OFF -DSLEEF_ENABLE_TLFLOAT=OFF
    -DSLEEF_ENABLE_TESTER4=OFF -DSLEEF_DISABLE_SSE4=ON -DSLEEF_DISABLE_AVX=ON
    -DSLEEF_DISABLE_FMA4=ON -DSLEEF_DISABLE_SVE=ON -DSLEEF_DISABLE_MPFR=ON
    -DSLEEF_DISABLE_SSL=ON -DSLEEF_DISABLE_FFTW=ON -DSLEEF_SHOW_CONFIG=OFF)
  # The CPU baseline reaches SLEEF's common code; each variant adds its own flags.
  if(SOLPROP_NATIVE_CPU_BASELINE_FLAG)
    list(APPEND _arguments "-DCMAKE_C_FLAGS=${SOLPROP_NATIVE_CPU_BASELINE_FLAG}")
  endif()
  if(APPLE)
    list(APPEND _arguments
      "-DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}"
      "-DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}")
  endif()
  ExternalProject_Add(
    solprop_native_sleef_build
    SOURCE_DIR "${sleef_SOURCE_DIR}"
    BINARY_DIR "${_binary}"
    CMAKE_ARGS ${_arguments}
    BUILD_COMMAND "${CMAKE_COMMAND}" --build <BINARY_DIR> --target sleef
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS "${_library}"
    EXCLUDE_FROM_ALL TRUE
  )
  # The header directory must exist when this project is generated.
  file(MAKE_DIRECTORY "${_binary}/include")
  add_library(solprop_native_sleef STATIC IMPORTED GLOBAL)
  set_target_properties(solprop_native_sleef PROPERTIES
    IMPORTED_LOCATION "${_library}"
    INTERFACE_INCLUDE_DIRECTORIES "${_binary}/include")
  add_dependencies(solprop_native_sleef solprop_native_sleef_build)
  set(sleef_SOURCE_DIR "${sleef_SOURCE_DIR}" PARENT_SCOPE)
endfunction()

set(SOLPROP_NATIVE_VECTOR_MATH_RECORD "scalar libm")
if(SOLPROP_NATIVE_VECTOR_MATH STREQUAL "SLEEF")
  _solprop_native_add_sleef()
  set(SOLPROP_NATIVE_VECTOR_MATH_RECORD "SLEEF ${SOLPROP_NATIVE_SLEEF_VERSION}")
endif()
message(STATUS "solprop-native vector maths: ${SOLPROP_NATIVE_VECTOR_MATH_RECORD}")

# Links the layer into a target: SLEEF's static library and its generated header.
function(solprop_native_link_vector_math target)
  if(SOLPROP_NATIVE_VECTOR_MATH STREQUAL "SLEEF")
    target_link_libraries(${target} PUBLIC solprop_native_sleef)
    target_compile_definitions(${target} PUBLIC SOLPROP_NATIVE_VECTOR_MATH_SLEEF=1)
  endif()
endfunction()
