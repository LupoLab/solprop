# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# clang-tidy while compiling (SOLPROP_NATIVE_ENABLE_CLANG_TIDY; nothing when it is off).
# Finds the development group's pinned clang-tidy beside the build's Python, defines
# SOLPROP_NATIVE_CLANG_TIDY_COMMAND, which solprop_native_configure_project_target()
# sets on each project target, and writes it to clang_tidy_command.txt for
# native/scripts/tidy.sh. Expects Python_EXECUTABLE and the C++ compiler.

if(SOLPROP_NATIVE_ENABLE_CLANG_TIDY)
  # The development group's pinned clang-tidy, in the environment beside the Python
  # this build uses, before any machine's LLVM: each LLVM release adds checks.
  get_filename_component(_python_bin "${Python_EXECUTABLE}" DIRECTORY)
  find_program(SOLPROP_NATIVE_CLANG_TIDY clang-tidy
               HINTS "${_python_bin}"
                     /opt/homebrew/opt/llvm/bin
                     /usr/local/opt/llvm/bin
               REQUIRED)
  execute_process(COMMAND "${SOLPROP_NATIVE_CLANG_TIDY}" --version
                  OUTPUT_VARIABLE _clang_tidy_version OUTPUT_STRIP_TRAILING_WHITESPACE)
  string(REGEX MATCH "version [0-9.]+" _clang_tidy_version "${_clang_tidy_version}")
  message(STATUS "clang-tidy: ${SOLPROP_NATIVE_CLANG_TIDY} (${_clang_tidy_version})")
  set(SOLPROP_NATIVE_CLANG_TIDY_COMMAND
      "${SOLPROP_NATIVE_CLANG_TIDY}"
      "--config-file=${PROJECT_SOURCE_DIR}/.clang-tidy"
      "--header-filter=^${PROJECT_SOURCE_DIR}/(include|src|tests)/.*")
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # The GCC-only warnings in the compile commands are unknown to clang-tidy's parser.
    list(APPEND SOLPROP_NATIVE_CLANG_TIDY_COMMAND "--extra-arg=-Wno-unknown-warning-option")
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
  if(APPLE)
    # Homebrew's Clang finds the macOS SDK from its own configuration file, which the
    # PyPI clang-tidy does not read.
    set(_tidy_sysroot "${CMAKE_OSX_SYSROOT}")
    if(NOT _tidy_sysroot)
      execute_process(COMMAND xcrun --show-sdk-path OUTPUT_VARIABLE _tidy_sysroot
                      OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
    endif()
    list(APPEND SOLPROP_NATIVE_CLANG_TIDY_COMMAND "--extra-arg=-isysroot${_tidy_sysroot}")
  endif()
  # The command the build runs, the resolved binary first, one argument per line, for
  # native/scripts/tidy.sh (`poe tidy-native`) and the weekly include-cleaner report.
  list(JOIN SOLPROP_NATIVE_CLANG_TIDY_COMMAND "\n" _tidy_command)
  file(WRITE "${PROJECT_BINARY_DIR}/clang_tidy_command.txt" "${_tidy_command}\n")
endif()
