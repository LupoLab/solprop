# shellcheck shell=bash
# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Shared helpers for the wheel dependency scripts in this directory (sourced, not run).

# fetch URL SHA256 DIRECTORY: download URL into DIRECTORY once, check its SHA-256 and
# print the archive's path. A mismatch removes the file and fails: every source the
# wheels bundle is pinned.
fetch() {
  local url=$1 sha256=$2 directory=$3
  local archive="$directory/${url##*/}"
  mkdir -p "$directory"
  if [[ ! -f "$archive" ]]; then
    curl --fail --location --silent --show-error --retry 3 --output "$archive.part" "$url"
    mv "$archive.part" "$archive"
  fi
  local actual
  if command -v sha256sum >/dev/null; then
    actual=$(sha256sum "$archive" | cut -d' ' -f1)
  else
    actual=$(shasum -a 256 "$archive" | cut -d' ' -f1)
  fi
  if [[ "$actual" != "$sha256" ]]; then
    rm -f "$archive"
    echo "checksum mismatch for $url: expected $sha256, got $actual" >&2
    return 1
  fi
  echo "$archive"
}

# The number of parallel build jobs: CMAKE_BUILD_PARALLEL_LEVEL when set, else every CPU.
jobs() {
  if [[ -n "${CMAKE_BUILD_PARALLEL_LEVEL:-}" ]]; then
    echo "$CMAKE_BUILD_PARALLEL_LEVEL"
  elif command -v nproc >/dev/null; then
    nproc
  else
    sysctl -n hw.ncpu
  fi
}
