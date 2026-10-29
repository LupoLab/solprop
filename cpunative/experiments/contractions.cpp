// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: GPL-3.0-or-later
// Standalone provider experiment; no provider selected by benchmark name.
#include <Eigen/Core>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

void blocked(int rows, int inner, int columns, const double* __restrict a,
             const double* __restrict b, double* __restrict out) {
  constexpr int block = 128;
  for (int row = 0; row < rows; ++row)
    for (int first = 0; first < columns; first += block) {
      double sum[block]{};
      const int count = std::min(block, columns - first);
      for (int k = 0; k < inner; ++k) {
        const double weight = a[row * inner + k];
        const double* __restrict input = b + k * columns + first;
#pragma omp simd
        for (int t = 0; t < count; ++t)
          sum[t] += weight * input[t];
      }
      std::copy_n(sum, count, out + row * columns + first);
    }
}
int main() {
  using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  for (int n : {10240, 20480})
    for (int modes : {3, 4})
      for (int nodes : {1, 11, 22, 33})
        for (bool projection : {false, true}) {
          int rows = projection ? modes : nodes, inner = projection ? nodes : modes;
          std::vector<double> a(rows * inner), b(inner * n), out(rows * n),
              reference(rows * n);
          for (std::size_t i = 0; i < a.size(); ++i)
            a[i] = std::sin(double(i));
          for (std::size_t i = 0; i < b.size(); ++i)
            b[i] = std::cos(double(i));
          blocked(rows, inner, n, a.data(), b.data(), reference.data());
          for (int provider = 0; provider < 3; ++provider) {
            auto execute = [&] {
              if (provider == 0)
                blocked(rows, inner, n, a.data(), b.data(), out.data());
              else if (provider == 1)
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, n, inner,
                            1., a.data(), inner, b.data(), n, 0., out.data(), n);
              else {
                Eigen::Map<const Matrix> left(a.data(), rows, inner),
                    right(b.data(), inner, n);
                Eigen::Map<Matrix>(out.data(), rows, n).noalias() = left * right;
              }
            };
            execute();
            double error = 0, norm = 0;
            for (std::size_t i = 0; i < out.size(); ++i) {
              error += std::pow(out[i] - reference[i], 2);
              norm += reference[i] * reference[i];
            }
            if (std::sqrt(error / norm) > 1e-12)
              return 2;
            auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < 80; ++i)
              execute();
            const double elapsed =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
                    .count() /
                80;
            std::cout << n << ',' << modes << ',' << nodes << ',' << projection << ','
                      << provider << ',' << elapsed << '\n';
          }
        }
}
