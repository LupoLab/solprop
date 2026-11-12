# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Sourced by check.sh and tidy.sh: the C++ compiler and Raman FFT provider of the native
# gate's builds, chosen in one place.

# The compiler: CXX on Linux (CPU builds need GCC 14 or newer; Ubuntu 24.04's g++ is 13),
# Homebrew's Clang on macOS (Apple silicon, then Intel), else the clang++ on PATH.
native_cxx_compiler() {
  if [[ $(uname -s) == Linux ]]; then
    command -v "${CXX:-g++}"
  elif [[ -x /opt/homebrew/opt/llvm/bin/clang++ ]]; then
    echo /opt/homebrew/opt/llvm/bin/clang++
  elif [[ -x /usr/local/opt/llvm/bin/clang++ ]]; then
    echo /usr/local/opt/llvm/bin/clang++
  else
    command -v clang++
  fi
}

# The Raman FFT provider: Accelerate on macOS, Kokkos-FFT elsewhere.
native_raman_provider() {
  if [[ $(uname -s) == Darwin ]]; then
    echo ACCELERATE
  else
    echo KOKKOSFFT
  fi
}
