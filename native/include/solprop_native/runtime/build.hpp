// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// The build's facts as constants -- `cuda`, `hip`, `device`, `host` and `openmp` in
// `solprop_native::build` -- for `if constexpr` where both branches compile in every
// build (CPU, CUDA and HIP); code that names a device or host-only facility keeps its
// `#if`.

#include <Kokkos_Macros.hpp>

namespace solprop_native::build {

#ifdef KOKKOS_ENABLE_CUDA
inline constexpr bool cuda = true;
#else
inline constexpr bool cuda = false;
#endif

#ifdef KOKKOS_ENABLE_HIP
inline constexpr bool hip = true;
#else
inline constexpr bool hip = false;
#endif

/** A CUDA or HIP build: the kernels' execution space is a device. */
inline constexpr bool device = cuda || hip;

/** A host build: the kernels run on the CPU. */
inline constexpr bool host = !device;

#ifdef KOKKOS_ENABLE_OPENMP
inline constexpr bool openmp = true;
#else
inline constexpr bool openmp = false;
#endif

} // namespace solprop_native::build
