// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: GPL-3.0-or-later
// Compare explicit block offsets with OpenMP's scan operation. An inscan
// reduction expresses the dependency; an ordinary parallel-for would be wrong.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <omp.h>
#include <string>
#include <utility>
#include <vector>
int main(int argc, char** argv) {
  const int threads = argc > 1 ? std::stoi(argv[1]) : 1;
  constexpr int block = 512, repeats = 100;
  omp_set_dynamic(0);
  std::cout << "samples,rows,threads,oscillating,policy,seconds,scaled_error\n";
  for (const auto dimensions : {std::pair{1728, 128},
                                {6400, 33},
                                {10240, 11},
                                {10240, 33},
                                {20480, 11},
                                {20480, 33}}) {
    const int n = dimensions.first, rows = dimensions.second;
    for (bool oscillating : {false, true}) {
      const int blocks = (n + block - 1) / block;
      std::vector<double> input(rows * n), output(rows * n), sums(rows * blocks);
      std::vector<long double> reference(rows * n);
      long double absolute_work = 0;
      for (int row = 0; row < rows; ++row) {
        long double sum = 0, absolute = 0;
        for (int i = 0; i < n; ++i) {
          const double x = i ? std::exp(-std::pow((i - n * .5) / (n * .1), 2)) *
                                   (oscillating ? std::sin(i * .1) : 1.)
                             : 0.;
          input[row * n + i] = x;
          sum += x;
          absolute += std::abs(x);
          reference[row * n + i] = sum;
        }
        absolute_work = std::max(absolute_work, absolute);
      }
      for (int policy = 0; policy < 4; ++policy) {
        auto execute = [&] {
          if (!policy) {
#pragma omp parallel for num_threads(threads) if (threads > 1) schedule(static)
            for (int row = 0; row < rows; ++row) {
              double sum = 0;
              for (int i = 0; i < n; ++i)
                output[row * n + i] = sum += input[row * n + i];
            }
          } else if (policy == 2) {
#pragma omp parallel for num_threads(threads) if (threads > 1) schedule(static)
            for (int row = 0; row < rows; ++row) {
              double sum = 0;
#pragma omp simd reduction(inscan, + : sum)
              for (int i = 0; i < n; ++i) {
                sum += input[row * n + i];
#pragma omp scan inclusive(sum)
                output[row * n + i] = sum;
              }
            }
          } else if (policy == 3) {
            // This gives a long column the full team, at the cost of repeated
            // synchronization. Measure it; do not assume more parallelism wins.
            for (int row = 0; row < rows; ++row) {
              double sum = 0;
#pragma omp parallel for simd num_threads(threads) reduction(inscan, + : sum)
              for (int i = 0; i < n; ++i) {
                sum += input[row * n + i];
#pragma omp scan inclusive(sum)
                output[row * n + i] = sum;
              }
            }
          } else {
#pragma omp parallel num_threads(threads) if (threads > 1)
            {
#pragma omp for schedule(static)
              for (int b = 0; b < rows * blocks; ++b) {
                const int row = b / blocks, first = (b % blocks) * block;
                double sum = 0;
                for (int i = first; i < std::min(n, first + block); ++i)
                  output[row * n + i] = sum += input[row * n + i];
                sums[b] = sum;
              }
#pragma omp for schedule(static)
              for (int row = 0; row < rows; ++row) {
                double sum = 0;
                for (int b = 0; b < blocks; ++b) {
                  const double value = sums[row * blocks + b];
                  sums[row * blocks + b] = sum;
                  sum += value;
                }
              }
#pragma omp for schedule(static)
              for (int b = 0; b < rows * blocks; ++b) {
                const int row = b / blocks, first = (b % blocks) * block;
                const double offset = sums[b];
#pragma omp simd
                for (int i = first; i < std::min(n, first + block); ++i)
                  output[row * n + i] += offset;
              }
            }
          }
        };
        execute();
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < repeats; ++i)
          execute();
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
                .count() /
            repeats;
        long double error = 0;
        for (std::size_t i = 0; i < output.size(); ++i)
          error = std::max(error, std::abs(reference[i] - output[i]));
        // Standard forward bound for n sequential additions, independent of
        // cancellation.
        const double bound = n * std::numeric_limits<double>::epsilon();
        if (error > bound * absolute_work)
          return 2;
        std::cout << n << ',' << rows << ',' << threads << ',' << oscillating << ','
                  << policy << ',' << seconds << ',' << double(error / absolute_work)
                  << '\n';
      }
    }
  }
}
