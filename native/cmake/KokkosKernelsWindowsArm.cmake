# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Patch the fetched Kokkos Kernels: like Kokkos, it gives every Clang on Windows -mcx16,
# an x86 option that Clang refuses for Arm. Run by FetchContent's PATCH_COMMAND from the
# Kokkos Kernels source directory. The patch is idempotent.

set(_file cmake/kokkoskernels_windows.cmake)
set(_old "add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-mcx16>)")
set(_new "if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES \"^(ARM64|aarch64)$\")\n    ${_old}\n  endif()")
file(READ "${_file}" _text)
string(FIND "${_text}" "${_new}" _done)
if(_done EQUAL -1)
  string(FIND "${_text}" "${_old}" _at)
  if(_at EQUAL -1)
    message(FATAL_ERROR "Kokkos Kernels Windows Arm patch: pattern not found in ${_file}")
  endif()
  string(REPLACE "${_old}" "${_new}" _text "${_text}")
  file(WRITE "${_file}" "${_text}")
endif()
