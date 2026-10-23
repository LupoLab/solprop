#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Build LLVM's OpenMP runtime (libomp) for the macOS wheel's deployment target.
#
# The wheel bundles the libomp its extension and FFTW link against (delocate copies it
# into the package). Homebrew's libomp is built for the build machine's macOS, so a
# wheel delocated from it would refuse every older system; this builds the same
# release for MACOSX_DEPLOYMENT_TARGET instead. The extension's compiler must not be
# newer than this runtime (a newer compiler can emit entry points an older libomp
# lacks), so the script checks Homebrew's Clang, the compiler native/pyproject.toml
# selects, against LIBOMP_VERSION.
#
# Usage: MACOSX_DEPLOYMENT_TARGET=14.0 native/ci/build-libomp.sh PREFIX
# Installs PREFIX/lib/libomp.dylib (install name PREFIX/lib/libomp.dylib),
# PREFIX/include/omp.h and PREFIX/licences/LICENSE-libomp.txt. Sources and build trees
# go to SOLPROP_CI_WORK (default PREFIX/_work).
set -euo pipefail

prefix=${1:?usage: build-libomp.sh PREFIX}
: "${MACOSX_DEPLOYMENT_TARGET:?set MACOSX_DEPLOYMENT_TARGET to the wheel deployment target}"
# The release Homebrew's libomp formula builds, from the same archive and checksum.
version=${LIBOMP_VERSION:-23.1.3}
sha256=${LIBOMP_SHA256:-c44186a7762ed28954be72e5ff6df9808e0779d4f1bf014ecc4e7e211d31ee34}
work=${SOLPROP_CI_WORK:-$prefix/_work}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=native/ci/lib.sh
source "$here/lib.sh"

compiler=${SOLPROP_CI_CXX:-/opt/homebrew/opt/llvm/bin/clang++}
if [[ -x "$compiler" ]]; then
  compiler_major=$("$compiler" --version | sed -n 's/.*clang version \([0-9]*\).*/\1/p' | head -1)
  if [[ -n "$compiler_major" && "$compiler_major" -gt "${version%%.*}" ]]; then
    echo "$compiler is Clang $compiler_major, newer than libomp $version: raise LIBOMP_VERSION" >&2
    exit 1
  fi
fi

archive=$(fetch "https://github.com/llvm/llvm-project/releases/download/llvmorg-$version/llvm-project-$version.src.tar.xz" "$sha256" "$work")
source_dir="$work/llvm-project-$version.src"
if [[ ! -d "$source_dir/runtimes" ]]; then
  # The runtimes build needs only these parts of the monorepo.
  tar -xJf "$archive" -C "$work" \
    "llvm-project-$version.src/cmake" "llvm-project-$version.src/llvm/cmake" \
    "llvm-project-$version.src/llvm/utils/llvm-lit" "llvm-project-$version.src/openmp" \
    "llvm-project-$version.src/runtimes" "llvm-project-$version.src/third-party"
fi

# The options are Homebrew's: no libgomp aliases (they would shadow GCC's), no OMPT
# tools and no offloading. An absolute install name lets delocate find the library.
cmake -S "$source_dir/runtimes" -B "$work/libomp-build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$prefix" \
  -DCMAKE_INSTALL_NAME_DIR="$prefix/lib" \
  -DCMAKE_OSX_ARCHITECTURES="$(uname -m)" \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="$MACOSX_DEPLOYMENT_TARGET" \
  -DLLVM_ENABLE_RUNTIMES=openmp \
  -DLIBOMP_INSTALL_ALIASES=OFF \
  -DOPENMP_ENABLE_OMPT_TOOLS=OFF \
  -DOPENMP_ENABLE_LIBOMPTARGET=OFF
cmake --build "$work/libomp-build" --parallel "$(jobs)"
cmake --install "$work/libomp-build"
# The wheel carries the runtime's licence (CMake's SOLPROP_NATIVE_BUNDLED_LICENCES).
mkdir -p "$prefix/licences"
cp "$source_dir/openmp/LICENSE.TXT" "$prefix/licences/LICENSE-libomp.txt"
otool -l "$prefix/lib/libomp.dylib" | grep -A3 LC_BUILD_VERSION | grep minos
