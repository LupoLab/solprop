#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Build and repair the macOS arm64 wheel locally, with the steps and settings of
# cibuildwheel's macOS build (native/pyproject.toml, [tool.cibuildwheel.macos]):
# cibuildwheel itself needs python.org's framework Python, which it installs only on CI.
# Keep the two in step.
#
# Needs Homebrew's llvm, cmake and ninja, and uv. Run from anywhere:
#   native/ci/build-wheel-macos.sh OUT_DIR [DEPS_PREFIX]
# The repaired wheel lands in OUT_DIR; the dependencies are built once into
# DEPS_PREFIX (default OUT_DIR/deps) and reused while their libraries are there.
set -euo pipefail

out=$(mkdir -p "${1:?usage: build-wheel-macos.sh OUT_DIR [DEPS_PREFIX]}" && cd "$1" && pwd)
deps=${2:-$out/deps}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET:-14.0}

if [[ ! -f "$deps/lib/libfftw3_omp.dylib" || ! -f "$deps/licences/LICENSE-fftw.txt" ]]; then
  "$here/build-deps.sh" "$deps"
fi
export CMAKE_ARGS="-DFFTW_ROOT=$deps -DSOLPROP_NATIVE_LIBOMP_LIBRARY=$deps/lib/libomp.dylib -DSOLPROP_NATIVE_LIBOMP_INCLUDE_DIR=$deps/include -DSOLPROP_NATIVE_BUNDLED_LICENCES=$deps/licences"
rm -rf "$out/raw"
(cd "$here/.." && uv build --wheel --python 3.14 \
  -Ccmake.define.SOLPROP_NATIVE_PROFILE=DISTRIBUTION \
  "-Cbuild-dir=build/cibw-{wheel_tag}" --out-dir "$out/raw")
uvx --from delocate delocate-wheel --require-archs arm64 \
  --require-target-macos-version "$MACOSX_DEPLOYMENT_TARGET" -w "$out" -v "$out"/raw/*.whl
ls "$out"/*.whl
