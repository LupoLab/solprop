# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# The wheel's install rules, for a scikit-build-core build (SKBUILD; nothing otherwise):
# the extension and its run path, the BLAS and the licences the wheel carries, the CPU
# baseline (_cpu_baseline.txt) and the build's configuration (_build.json). Defines the
# SOLPROP_NATIVE_BUNDLED_LICENCES cache path. Expects the _core target, the fetched
# dependencies' source directories, nanobind_ROOT and cmake/HostLibraries.cmake's
# vendored-library variables.

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
  # solprop SEP 46: the build's configuration, which a calibration's fingerprint reads
  # without loading the extension (and its OpenMP runtime). A variant wheel's builds
  # share it; the fingerprint also digests the chosen extension itself.
  set(_build_record "{}")
  foreach(_field IN ITEMS
      "compiler_id;${CMAKE_CXX_COMPILER_ID}"
      "compiler_version;${CMAKE_CXX_COMPILER_VERSION}"
      "openmp_libraries;${OpenMP_CXX_LIB_NAMES}"
      "fft_backend;${SOLPROP_NATIVE_HOST_FFT_BACKEND}"
      "blas_backend;${SOLPROP_NATIVE_BLAS_BACKEND}"
      # One GEMM provider remains (the vendor GEMM for an eligible transform, the
      # custom kernels for the rest); the field stays for the records that hold it.
      "gemm_provider;HYBRID"
      "raman_fft_provider;${SOLPROP_NATIVE_RAMAN_FFT_PROVIDER}"
      "vector_math;${SOLPROP_NATIVE_VECTOR_MATH}"
      "build_profile;${SOLPROP_NATIVE_BUILD_PROFILE}"
      "device_backend;${SOLPROP_NATIVE_DEVICE_BACKEND}"
      "device_arch;${SOLPROP_NATIVE_DEVICE_ARCH}")
    list(POP_FRONT _field _key)
    string(REPLACE ";" "," _value "${_field}")
    string(REPLACE "\\" "\\\\" _value "${_value}")
    string(REPLACE "\"" "\\\"" _value "${_value}")
    string(JSON _build_record SET "${_build_record}" "${_key}" "\"${_value}\"")
  endforeach()
  file(WRITE "${PROJECT_BINARY_DIR}/build_record.json" "${_build_record}\n")
  install(FILES "${PROJECT_BINARY_DIR}/build_record.json" DESTINATION solprop_native
          RENAME _build.json)
  # Kokkos, Kokkos Kernels, Kokkos-FFT, SLEEF, nanobind and nanobind's robin_map are
  # compiled into the extension, so the wheel carries their notices beside it. Kokkos-FFT is
  # MIT OR Apache-2.0 WITH LLVM-exception; its MIT terms are the ones taken here.
  install(FILES "${kokkos_SOURCE_DIR}/LICENSE" DESTINATION solprop_native
          RENAME LICENSE-kokkos.txt)
  install(FILES "${kokkoskernels_SOURCE_DIR}/LICENSE" DESTINATION solprop_native
          RENAME LICENSE-kokkos-kernels.txt)
  install(FILES "${kokkosfft_SOURCE_DIR}/LICENSES/MIT.txt" DESTINATION solprop_native
          RENAME LICENSE-kokkos-fft.txt)
  # SLEEF (the host vector maths, linked static) is under the Boost Software License.
  if(SOLPROP_NATIVE_VECTOR_MATH STREQUAL "SLEEF")
    install(FILES "${sleef_SOURCE_DIR}/LICENSE.txt" DESTINATION solprop_native
            RENAME LICENSE-sleef.txt)
  endif()
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
