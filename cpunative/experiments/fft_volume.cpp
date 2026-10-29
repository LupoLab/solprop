// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: GPL-3.0-or-later
// Isolate full-volume FFT planning and execution from propagation arithmetic.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fftw3.h>
#include <iostream>
#include <omp.h>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
  if (argc != 3 && argc != 4)
    return 1;
  const int threads = std::stoi(argv[1]);
  const bool measure = std::string(argv[2]) == "measure";
  const int block = argc == 4 ? std::stoi(argv[3]) : 0;
  constexpr int nx = 384, ny = 384, nt = 750;
  constexpr std::size_t size = std::size_t(nx) * ny * nt;
  omp_set_dynamic(0);
  omp_set_num_threads(threads);
  auto* input = fftw_alloc_complex(size);
  auto* output = fftw_alloc_complex(size);
  if (!input || !output || !fftw_init_threads())
    throw std::runtime_error("FFT allocation/initialisation failed");
#pragma omp parallel for schedule(static)
  for (std::size_t i = 0; i < size; ++i) {
    input[i][0] = input[i][1] = 0;
    output[i][0] = output[i][1] = 0;
  }
  fftw_plan_with_nthreads(block ? 1 : threads);
  // Bound the exploratory planner independently of execution time.
  fftw_set_timelimit(20.);
  using Clock = std::chrono::steady_clock;
  const auto start = Clock::now();
  const unsigned flags = measure ? FFTW_MEASURE : FFTW_ESTIMATE;
  auto inverse = block ? fftw_plan_dft_2d(nx, nt, input, output, FFTW_BACKWARD, flags)
                       : fftw_plan_dft_3d(ny, nx, nt, input, output, FFTW_BACKWARD, flags);
  auto forward = block ? fftw_plan_dft_2d(nx, nt, output, output, FFTW_FORWARD, flags)
                       : fftw_plan_dft_3d(ny, nx, nt, output, output, FFTW_FORWARD, flags);
  fftw_plan y_inverse{}, y_forward{}, tail_inverse{}, tail_forward{};
  constexpr int plane = nx * nt;
  const int tail = block ? plane % block : 0;
  if (block) {
    y_inverse = fftw_plan_many_dft(1, &ny, block, output, nullptr, plane, 1,
                                   output, nullptr, plane, 1, FFTW_BACKWARD, flags);
    y_forward = fftw_plan_many_dft(1, &ny, block, output, nullptr, plane, 1,
                                   output, nullptr, plane, 1, FFTW_FORWARD, flags);
    if (tail) {
      tail_inverse = fftw_plan_many_dft(1, &ny, tail, output, nullptr, plane, 1,
                                        output, nullptr, plane, 1, FFTW_BACKWARD, flags);
      tail_forward = fftw_plan_many_dft(1, &ny, tail, output, nullptr, plane, 1,
                                        output, nullptr, plane, 1, FFTW_FORWARD, flags);
    }
  }
  if (!inverse || !forward)
    throw std::runtime_error("FFT plan construction failed");
  std::cout << "threads=" << threads << " effort=" << argv[2]
            << " split_block=" << block
            << " planning_seconds="
            << std::chrono::duration<double>(Clock::now() - start).count() << '\n';
  fftw_print_plan(inverse);
  std::cout << '\n';
  fftw_print_plan(forward);
  if (block) {
    std::cout << '\n';
    fftw_print_plan(y_inverse);
    std::cout << '\n';
    fftw_print_plan(y_forward);
  }
  std::cout << '\n' << std::flush;
#pragma omp parallel for schedule(static)
  for (std::size_t i = 0; i < size; ++i) {
    input[i][0] = std::sin(.01 * (i % nt));
    input[i][1] = std::cos(.02 * (i % nx));
  }
  for (int repeat = 0; repeat < 3; ++repeat) {
    auto previous = Clock::now();
    if (block) {
#pragma omp parallel for schedule(static)
      for (int y = 0; y < ny; ++y)
        fftw_execute_dft(inverse, input + std::size_t(y) * plane,
                          output + std::size_t(y) * plane);
#pragma omp parallel for schedule(static)
      for (int column = 0; column < plane; column += block)
        fftw_execute_dft(plane - column < block ? tail_inverse : y_inverse,
                          output + column, output + column);
    } else {
      fftw_execute(inverse);
    }
    const double backward = std::chrono::duration<double>(Clock::now() - previous).count();
    previous = Clock::now();
    if (block) {
#pragma omp parallel for schedule(static)
      for (int column = 0; column < plane; column += block)
        fftw_execute_dft(plane - column < block ? tail_forward : y_forward,
                          output + column, output + column);
#pragma omp parallel for schedule(static)
      for (int y = 0; y < ny; ++y)
        fftw_execute_dft(forward, output + std::size_t(y) * plane,
                          output + std::size_t(y) * plane);
    } else {
      fftw_execute(forward);
    }
    const double ahead = std::chrono::duration<double>(Clock::now() - previous).count();
    double error = 0;
#pragma omp parallel for reduction(max : error) schedule(static)
    for (std::size_t i = 0; i < size; ++i)
      error = std::max(error, std::hypot(output[i][0] / size - input[i][0],
                                       output[i][1] / size - input[i][1]));
    std::cout << "repeat=" << repeat << " inverse_seconds=" << backward
              << " forward_seconds=" << ahead << " error=" << error << '\n'
              << std::flush;
    if (!(error < 1e-11))
      return 3;
  }
  fftw_destroy_plan(inverse);
  fftw_destroy_plan(forward);
  for (auto plan : {y_inverse, y_forward, tail_inverse, tail_forward})
    if (plan)
      fftw_destroy_plan(plan);
  fftw_free(input);
  fftw_free(output);
}
