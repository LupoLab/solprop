// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: GPL-3.0-or-later
// Standalone scheduling probe: all plans operate on the actual temporal lengths.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fftw3.h>
#include <iostream>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
int main(int argc, char** argv) {
  const int threads = argc > 1 ? std::stoi(argv[1]) : 1;
  const int repeats = 40;
  omp_set_dynamic(0);
  if (!fftw_init_threads())
    return 2;
  std::cout << "samples,rows,threads,policy,planning_seconds,pair_seconds,error\n";
  for (int n : {10240, 20480, 40960})
    for (int rows : {1, 3, 4, 8, 9, 11, 12, 33}) {
      const int bins = (n / 2 + 4) / 4 * 4;
      double* time = fftw_alloc_real(std::size_t(n) * rows);
      fftw_complex* frequency = fftw_alloc_complex(std::size_t(bins) * rows);
      if (!time || !frequency)
        throw std::bad_alloc();
      for (int policy = 0; policy < 3; ++policy) {
        // Distributed serial rows, or internally threaded batch ESTIMATE/MEASURE.
        fftw_plan_with_nthreads(policy ? threads : 1);
        const unsigned flags =
            policy == 1 && threads > 1 ? FFTW_ESTIMATE : FFTW_MEASURE;
        const auto planning = Clock::now();
        fftw_plan forward =
            policy ? fftw_plan_many_dft_r2c(1, &n, rows, time, nullptr, 1, n, frequency,
                                            nullptr, 1, bins, flags)
                   : fftw_plan_dft_r2c_1d(n, time, frequency, flags);
        fftw_plan inverse =
            policy ? fftw_plan_many_dft_c2r(1, &n, rows, frequency, nullptr, 1, bins,
                                            time, nullptr, 1, n, flags)
                   : fftw_plan_dft_c2r_1d(n, frequency, time, flags);
        const double construction =
            std::chrono::duration<double>(Clock::now() - planning).count();
        if (!forward || !inverse)
          throw std::runtime_error("plan creation failed");
        for (int row = 0; row < rows; ++row)
          for (int i = 0; i < n; ++i)
            time[std::size_t(row) * n + i] = std::sin(.03 * i) + .2 * std::cos(.1 * i);
        auto execute = [&] {
          if (policy) {
            fftw_execute(forward);
            fftw_execute(inverse);
            for (std::size_t i = 0; i < std::size_t(n) * rows; ++i)
              time[i] /= n;
          } else {
            auto row_pair = [&](int row) {
              double* t = time + std::size_t(row) * n;
              fftw_complex* f = frequency + std::size_t(row) * bins;
              fftw_execute_dft_r2c(forward, t, f);
              fftw_execute_dft_c2r(inverse, f, t);
              for (int i = 0; i < n; ++i)
                t[i] /= n;
            };
            if (threads == 1) {
              for (int row = 0; row < rows; ++row)
                row_pair(row);
            } else {
#pragma omp parallel for num_threads(threads) schedule(static)
              for (int row = 0; row < rows; ++row)
                row_pair(row);
            }
          }
        };
        execute();
        const auto start = Clock::now();
        for (int i = 0; i < repeats; ++i)
          execute();
        const double elapsed =
            std::chrono::duration<double>(Clock::now() - start).count() / repeats;
        double error = 0;
        for (int row = 0; row < rows; ++row)
          for (int i = 0; i < n; ++i)
            error =
                std::max(error, std::abs(time[std::size_t(row) * n + i] -
                                         std::sin(.03 * i) - .2 * std::cos(.1 * i)));
        // Each round trip contributes O(log(n)*eps); the allowance covers 41 pairs.
        if (error > 128 * (repeats + 1) * std::log2(n) * 2.220446049250313e-16)
          return 3;
        std::cout << n << ',' << rows << ',' << threads << ',' << policy << ','
                  << construction << ',' << elapsed << ',' << error << '\n';
        fftw_destroy_plan(forward);
        fftw_destroy_plan(inverse);
      }
      fftw_free(time);
      fftw_free(frequency);
    }
}
