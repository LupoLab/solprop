// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// The host allocator queries of the general workspace's memory ledger: host code, run
// once around planning (setup.cpp's FFT plans, statistics_monitor.inc's edge plans),
// so that the ledger can record what the vendor FFT libraries allocate out of sight.
// Functions: heap_bytes (the process allocator's bytes in use) and device_free_bytes
// (the device's free memory), both defined in memory.cpp.

#include <cstddef>

namespace solprop_native::general {

// Net process allocator growth includes opaque vendor FFT plan/workspace
// allocations. It is measured separately from Kokkos Views and is not a peak bound.
std::size_t heap_bytes();
/** The device's free memory in bytes; zero on a host build. */
std::size_t device_free_bytes();

} // namespace solprop_native::general
