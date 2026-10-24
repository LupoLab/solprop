#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Build FFTW for a Linux or macOS wheel: double and single precision, shared, with
# its OpenMP and POSIX-threads libraries.
#
# The wheel repair (auditwheel, delocate) copies these libraries into the package.
# The common code is compiled for the wheel's CPU baseline, except on x86-64: there it
# stays at x86-64-v2, below the wheel's x86-64-v3, as qualified in solprop SEP 29 (the
# speed is in the codelets, and no FMA contraction enters FFTW's scalar code). On
# x86-64 FFTW's SSE2 and AVX codelets are chosen at run time (AVX2 and AVX-512 codelets were not a
# uniform gain when measured, so they are left out). On Arm the cycle counter is
# enabled, so that measured planning (FFTW_MEASURE) works; without one FFTW plans by
# estimate only. Apple silicon also enables FFTW's explicit NEON codelets in both
# precisions; compiler auto-vectorisation does not enable these implementations.
#
# OpenMP: on Linux the compiler's own runtime (GCC's libgomp in the manylinux image);
# on macOS the libomp of LIBOMP_PREFIX (native/ci/build-libomp.sh), the runtime the
# extension links, so that the process loads one OpenMP runtime; there the compiler is
# SOLPROP_CI_CC (default Homebrew's Clang).
#
# Usage: native/ci/build-fftw.sh PREFIX
#   macOS: MACOSX_DEPLOYMENT_TARGET and LIBOMP_PREFIX must be set.
# Sources and build trees go to SOLPROP_CI_WORK (default PREFIX/_work).
set -euo pipefail

prefix=${1:?usage: build-fftw.sh PREFIX}
version=3.3.11
sha256=5630c24cdeb33b131612f7eb4b1a9934234754f9f388ff8617458d0be6f239a1
work=${SOLPROP_CI_WORK:-$prefix/_work}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=native/ci/lib.sh
source "$here/lib.sh"

configure_args=(--prefix="$prefix" --enable-shared --disable-static --enable-threads
                --enable-openmp --disable-fortran --disable-doc)
case "$(uname -s)/$(uname -m)" in
  Linux/x86_64)
    cflags="-O3 -march=x86-64-v2 -mtune=generic"
    simd=(--enable-sse2 --enable-avx)
    ;;
  Linux/aarch64)
    cflags="-O3 -mcpu=neoverse-n1"
    simd=(--enable-armv8-cntvct-el0)
    ;;
  Darwin/arm64)
    : "${MACOSX_DEPLOYMENT_TARGET:?set MACOSX_DEPLOYMENT_TARGET}"
    : "${LIBOMP_PREFIX:?set LIBOMP_PREFIX to the libomp built by build-libomp.sh}"
    cflags="-O3 -mcpu=apple-m1 -mmacosx-version-min=$MACOSX_DEPLOYMENT_TARGET"
    simd=(--enable-armv8-cntvct-el0 --enable-neon)
    # The extension's compiler, Homebrew's Clang (native/pyproject.toml), which takes
    # -fopenmp and links the libomp found first on the library path: this one.
    # Dead-stripping keeps libomp out of the libraries that do not use it.
    export CC=${SOLPROP_CI_CC:-/opt/homebrew/opt/llvm/bin/clang}
    export ax_cv_c_openmp=-fopenmp
    export CPPFLAGS="-I$LIBOMP_PREFIX/include ${CPPFLAGS:-}"
    export LDFLAGS="-L$LIBOMP_PREFIX/lib -lomp -Wl,-dead_strip_dylibs ${LDFLAGS:-}"
    ;;
  *)
    echo "no FFTW recipe for $(uname -s)/$(uname -m)" >&2
    exit 1
    ;;
esac

archive=$(fetch "https://www.fftw.org/fftw-$version.tar.gz" "$sha256" "$work")
rm -rf "$work/fftw-$version"
tar -xzf "$archive" -C "$work"
for precision in double single; do
  build="$work/fftw-build-$precision"
  rm -rf "$build"
  mkdir -p "$build"
  precision_args=(--disable-single)
  [[ $precision == single ]] && precision_args=(--enable-single)
  (cd "$build" && CFLAGS="$cflags" "$work/fftw-$version/configure" \
     "${configure_args[@]}" "${simd[@]}" "${precision_args[@]}" >configure.log)
  make -C "$build" -j"$(jobs)" >"$build/make.log"
  make -C "$build" install >"$build/install.log"
done
# The wheel carries FFTW's licence (CMake's SOLPROP_NATIVE_BUNDLED_LICENCES).
mkdir -p "$prefix/licences"
cp "$work/fftw-$version/COPYING" "$prefix/licences/LICENSE-fftw.txt"
ls "$prefix/lib"
