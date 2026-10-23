#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Run the native package's CPU quality gate (`poe check-native` from the repository
# root; `poe check-all` runs solprop's gate first). The native package is a member of
# the repository's uv workspace and shares its environment, `../.venv`.

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
workspace_root=$(cd "$repo_root/.." && pwd)
cd "$repo_root"
raman_default=KOKKOSFFT
if [[ $(uname -s) == Darwin ]]; then
  raman_default=ACCELERATE
fi
raman_provider=${SOLPROP_NATIVE_CHECK_RAMAN_FFT_PROVIDER:-$raman_default}
export CMAKE_ARGS="${CMAKE_ARGS:-} -DSOLPROP_NATIVE_RAMAN_FFT_PROVIDER=$raman_provider"

if [[ $(uname -s) == "Linux" ]]; then
  # CXX when set (Linux CPU builds need GCC 14 or newer; Ubuntu 24.04's g++ is 13).
  cxx_compiler=$(command -v "${CXX:-g++}")
elif [[ -x /opt/homebrew/opt/llvm/bin/clang++ ]]; then
  cxx_compiler=/opt/homebrew/opt/llvm/bin/clang++
elif [[ -x /usr/local/opt/llvm/bin/clang++ ]]; then
  cxx_compiler=/usr/local/opt/llvm/bin/clang++
else
  cxx_compiler=$(command -v clang++)
fi
cpp_sources=(
  benchmarks/cpu_timer.cpp
  include/solprop_native/chebyshev.hpp
  include/solprop_native/core.hpp
  include/solprop_native/coupled_flow.hpp
  include/solprop_native/dispersive.hpp
  include/solprop_native/envelope_plasma.hpp
  include/solprop_native/far_field.hpp
  include/solprop_native/fft_plan.hpp
  include/solprop_native/fp.hpp
  include/solprop_native/general_modal.hpp
  include/solprop_native/raman_fft.hpp
  include/solprop_native/gemm.hpp
  include/solprop_native/lagrange_basis.hpp
  include/solprop_native/plasma.hpp
  include/solprop_native/slab.hpp
  include/solprop_native/z_table.hpp
  src/bindings.cpp
  src/coverage_labels.cpp
  src/core.cpp
  src/coupled_flow.cpp
  src/dispersive.cpp
  src/far_field.cpp
  src/general.cpp
  src/general_modal.cpp
  src/helpers.cpp
  src/helpers.hpp
  src/raman_fft.cpp
  src/benchmark_gemm.cpp
  tests/native/bounds.hpp
  tests/native/main.cpp
  tests/native/test_aperture.cpp
  tests/native/test_core.cpp
  tests/native/test_coupled_flow.cpp
  tests/native/test_dispersive.cpp
  tests/native/test_envelope_plasma.cpp
  tests/native/test_fft.cpp
  tests/native/test_fp.cpp
  tests/native/test_general_modal.cpp
  tests/native/test_helpers.cpp
  tests/native/test_z_table.cpp
)

qualification_python=${SOLPROP_NATIVE_QUALIFICATION_PYTHON:-}
if [[ -n "$qualification_python" ]]; then
  if [[ ! -x "$qualification_python" ]]; then
    echo "SOLPROP_NATIVE_QUALIFICATION_PYTHON must name an executable Python" >&2
    exit 1
  fi
  qualification_python=$("$qualification_python" -c 'import sys; print(sys.executable)')
fi
# The workspace's locked environment, without removing a native build already in it;
# the editable native package is then built without isolation, from this tree. Later
# commands must not sync, which would remove it again.
python_executable="$workspace_root/.venv/bin/python"
uv sync --frozen --inexact --python 3.14 --project "$workspace_root"
export UV_NO_SYNC=1
# An editable install records the version of the commit it was installed at, and a
# sync considers it satisfied after the commit moves: reinstall solprop so its metadata
# names the same commit as the native package built below (one version for both).
uv pip install --python "$python_executable" --no-deps --reinstall --editable "$workspace_root"
if [[ -z "$qualification_python" ]]; then
  uv pip install --python "$python_executable" --no-deps --no-build-isolation --editable .
fi
# The repository's version, as hatch-vcs gives solprop and setuptools-scm gives this
# package: the native wheel built below must carry it (one version for both).
expected_version=$("$python_executable" -c 'import sys; from setuptools_scm import get_version; print(get_version(root=sys.argv[1], fallback_version="0.0.0"))' "$workspace_root")
uv run ruff format --check .
uv run ruff check .
mkdir -p build
uv run python - "$repo_root" "$workspace_root" <<'PY'
import json
import sys
import tomllib
from pathlib import Path

root, solprop = map(Path, sys.argv[1:])
config = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["pyright"]
for key in ("include", "exclude"):
    config[key] = [str(Path("..") / path) for path in config[key]]
config["extraPaths"] = [str(root / "src"), str(solprop / "src"), str(solprop / "tests/data")]
config["venvPath"] = str(solprop)
config["executionEnvironments"] = [{"root": ".."}]
(root / "build/pyright-worktree.json").write_text(json.dumps(config, indent=2) + "\n")
PY
npx --yes pyright@1.1.413 --project build/pyright-worktree.json
# The formatter is the dev group's pinned clang-format, never the machine's LLVM.
uv run clang-format --dry-run --Werror "${cpp_sources[@]}"

cmake --fresh --preset cpu-tidy \
  -DSOLPROP_NATIVE_RAMAN_FFT_PROVIDER="$raman_provider" \
  -DCMAKE_CXX_COMPILER="$cxx_compiler" \
  -DPython_EXECUTABLE="$python_executable"
cmake --build --preset cpu-tidy --parallel
ctest --test-dir build/presets/cpu-tidy --output-on-failure

cmake --fresh --preset cpu-sanitize \
  -DSOLPROP_NATIVE_RAMAN_FFT_PROVIDER="$raman_provider" \
  -DCMAKE_CXX_COMPILER="$cxx_compiler" \
  -DPython_EXECUTABLE="$python_executable"
cmake --build --preset cpu-sanitize --target solprop_native_tests --parallel
ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=1:strict_string_checks=1}" \
UBSAN_OPTIONS="${UBSAN_OPTIONS:-print_stacktrace=1:halt_on_error=1}" \
  ctest --test-dir build/presets/cpu-sanitize --output-on-failure

if [[ -z "$qualification_python" ]]; then
  uv pip install --python "$python_executable" --no-deps --no-build-isolation -e .
  OMP_PROC_BIND="${OMP_PROC_BIND:-spread}" \
  OMP_PLACES="${OMP_PLACES:-cores}" \
  SOLPROP_NATIVE_THREADS="${SOLPROP_NATIVE_THREADS:-4}" \
    uv run pytest
fi

# The editable build above is the local default (vendor BLAS, native architecture);
# the wheel is the distribution build published binaries use (a CPU baseline and the
# platform's BLAS, copied into the wheel on Linux), in its own build tree. The tidy
# and sanitizer presets are the reference profile.
quality_temp=$(mktemp -d "${TMPDIR:-/tmp}/solprop-native-check.XXXXXX")
trap 'rm -rf "$quality_temp"' EXIT
distribution_wheel=(-Ccmake.define.SOLPROP_NATIVE_PROFILE=DISTRIBUTION "-Cbuild-dir=build/distribution-{wheel_tag}")
if [[ -n "$qualification_python" ]]; then
  uv build --wheel --no-build-isolation "${distribution_wheel[@]}" --out-dir "$quality_temp/wheel"
else
  uv build --wheel "${distribution_wheel[@]}" --out-dir "$quality_temp/wheel"
fi

wheels=("$quality_temp"/wheel/*.whl)
if [[ ${#wheels[@]} -ne 1 || ! -f "${wheels[0]}" ]]; then
  echo "expected exactly one built wheel" >&2
  exit 1
fi

if [[ -n "$qualification_python" ]]; then
  uv pip install --python "$qualification_python" --force-reinstall --no-deps "${wheels[0]}"
  OMP_PROC_BIND="${OMP_PROC_BIND:-spread}" \
  OMP_PLACES="${OMP_PLACES:-cores}" \
  SOLPROP_NATIVE_THREADS="${SOLPROP_NATIVE_THREADS:-4}" \
    "$qualification_python" -m pytest
else
  uv venv "$quality_temp/venv" --python 3.14
  uv pip install --python "$quality_temp/venv/bin/python" "${wheels[0]}"
  qualification_python="$quality_temp/venv/bin/python"
fi
# Isolated mode confirms that the smoke test imports the installed wheel, with no
# source checkout added through the working directory or PYTHONPATH. On Linux the
# wheel's OpenBLAS must load from the package itself, not from the build environment.
OMP_PROC_BIND=false "$qualification_python" -I - "$expected_version" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

import solprop_native as sk

info = sk.build_info()
assert sk.__version__ == sys.argv[1], (sk.__version__, sys.argv[1])
assert info.build_profile == "distribution", info.build_profile
if sys.platform == "linux":
    package = Path(sk.__file__).resolve().parent
    assert Path(info.blas_library).resolve().parent == package, info.blas_library
    assert info.blas_threads is not None
json.dumps(info.__dict__)
subprocess.run([sys.executable, "-I", "-m", "solprop_native"], check=True, capture_output=True, text=True)
PY
