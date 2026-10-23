#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Build the libraries a Linux or macOS wheel bundles into PREFIX: FFTW, and on macOS
# first the libomp it and the extension share; their licences go to PREFIX/licences.
# cibuildwheel runs this as before-all (native/pyproject.toml); it runs the same way by
# hand. Windows has its own script, build-deps-windows.ps1.
#
# Usage: native/ci/build-deps.sh PREFIX   (macOS: with MACOSX_DEPLOYMENT_TARGET set)
set -euo pipefail

prefix=${1:?usage: build-deps.sh PREFIX}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ "$(uname -s)" == Darwin ]]; then
  "$here/build-libomp.sh" "$prefix"
  LIBOMP_PREFIX="$prefix" "$here/build-fftw.sh" "$prefix"
else
  "$here/build-fftw.sh" "$prefix"
  # auditwheel copies GCC's libgomp into the wheel: its licence, GPL-3.0 with the GCC
  # Runtime Library Exception, goes with it (the smoke test checks that it is there).
  exception=$(find /opt/rh /usr/share/licenses /usr/share/doc -name COPYING.RUNTIME \
                2>/dev/null | head -1 || true)
  if [[ -n "$exception" && -f "$(dirname "$exception")/COPYING3" ]]; then
    cat "$exception" "$(dirname "$exception")/COPYING3" >"$prefix/licences/LICENSE-libgomp.txt"
  else
    echo "warning: GCC's COPYING.RUNTIME and COPYING3 not found; no LICENSE-libgomp.txt" >&2
  fi
fi
