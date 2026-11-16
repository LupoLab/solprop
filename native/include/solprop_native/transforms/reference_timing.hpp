// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <complex>
#include <cstddef>
#include <vector>

namespace solprop_native {

/** solprop SEP 28 section 4.2's per-evaluation reference, independent of any
    workspace: the seconds of each timed forward-then-backward pair of an in-place,
    unnormalised rank-3 complex transform of `values` (row-major
    `(planes, n_y, n_x)`, copied in once) through Kokkos-FFT on the FFT execution
    space, after planning and `warm` untimed pairs. */
[[nodiscard]] std::vector<double>
reference_fft3_pairs(const std::complex<double>* values, std::size_t planes,
                     std::size_t n_y, std::size_t n_x, std::size_t warm,
                     std::size_t pairs);

/** The same reference's streaming volume-pass: the seconds of each timed scaled copy
    `b = s a` between two complex volumes of `elements` first touched in parallel on
    the kernels' execution space, after `warm` untimed copies. */
[[nodiscard]] std::vector<double>
reference_stream_copies(std::size_t elements, std::size_t warm, std::size_t repeats);

} // namespace solprop_native
