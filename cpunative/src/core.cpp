// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#include "solprop_cpunative/core.hpp"
#include "solprop_cpunative/fp.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>

#ifdef CPUNATIVE_BLAS
#ifdef CPUNATIVE_MKL_BLAS
#include <mkl_cblas.h>
#elif defined(__APPLE__)
// The umbrella header includes unrelated vector intrinsics that GCC rejects.
// Select the same modern Accelerate ABI with the narrow CBLAS declaration.
#include <vecLib/cblas_new.h>
#else
#include <cblas.h>
#endif
#endif
#ifdef CPUNATIVE_MKL_FFT
#include <fftw3_mkl.h>
#endif

namespace solprop_cpunative {
std::mutex& fftw_planner_mutex() {
  static std::mutex mutex;
  return mutex;
}
namespace {
fftw_complex* fft(Complex* p) { return reinterpret_cast<fftw_complex*>(p); }

// omega is finite and strictly positive on active plasma bins. Writing this
// special denominator explicitly avoids a general complex-division helper and
// its scaling operations on the tiny current tails; real divisions stay IEEE.
inline Complex divide_iomega(Complex value, double omega) {
  return {value.imag() / omega, -value.real() / omega};
}

// The serial branch never enters the OpenMP runtime. Each instantiation is
// inlined at its call site, so the compiler still sees the contiguous loop.
template <class Body> inline void for_rows(int threads, std::size_t count, Body body) {
  if (!count)
    return;
  // Keep the team width stable between small FFT batches and flat state loops.
  // Repeated team resizing dominated the developed-field GCC measurements.
  if (threads == 1 || count == 1) {
    for (std::size_t i = 0; i < count; ++i)
      body(i);
  } else {
#pragma omp parallel for num_threads(threads) schedule(static)
    for (std::size_t i = 0; i < count; ++i)
      body(i);
  }
}

// One cache line per worker; reduction ownership is fixed at construction.
// Serial execution calls the same arithmetic without an OpenMP region.
template <class Body>
std::array<double, 3> reduce_rows(int threads, std::size_t count, double* scratch,
                                  Body body, bool maximum = false) {
  const double initial = maximum ? -std::numeric_limits<double>::infinity() : 0.;
  for_rows(threads, threads, [&](std::size_t worker) {
    double* partial = scratch + 8 * worker;
    std::fill_n(partial, 3, initial);
    for (std::size_t i = count * worker / threads; i < count * (worker + 1) / threads;
         ++i)
      body(i, partial);
  });
  std::array<double, 3> total{initial, initial, initial};
  for (int worker = 0; worker < threads; ++worker)
    for (int k = 0; k < 3; ++k)
      if (maximum)
        total[k] = std::max(total[k], scratch[8 * worker + k]);
      else
        total[k] += scratch[8 * worker + k];
  return total;
}

// Optional wall-clock phase attribution: no clock calls when disabled.
struct PhaseTimer {
  bool enabled;
  std::array<double, 10>& totals;
  std::chrono::steady_clock::time_point previous;
  PhaseTimer(bool enabled, std::array<double, 10>& totals)
      : enabled(enabled), totals(totals) {
    if (enabled)
      previous = std::chrono::steady_clock::now();
  }
  void mark(int phase) {
    if (!enabled)
      return;
    const auto now = std::chrono::steady_clock::now();
    totals[phase] += std::chrono::duration<double>(now - previous).count();
    previous = now;
  }
};

Config validated(Config c) {
  if (c.modes < 1 || c.nodes < 1 || c.chunk < 1 || c.chunk > c.nodes || c.nt < 2 ||
      c.nt % 2 || c.nf < c.nt || c.nf % 2 || c.nw != c.nt / 2 + 1 || c.slots < 3 ||
      c.slots > 9 || c.threads < 1)
    throw std::invalid_argument("invalid workspace dimensions");
  const auto size = std::size_t(c.modes) * c.nw;
  const auto maps = std::size_t(c.nodes) * c.modes;
  const auto nf = std::size_t(c.nf), nw = std::size_t(c.nw);
  if (c.phi.empty() || c.phi.size() % size || c.prefactor.empty() ||
      c.prefactor.size() % size || c.synthesis.size() != maps ||
      c.projection.size() != maps || (!c.coarse.empty() && c.coarse.size() != maps) ||
      c.window.size() != nf || c.window_derivative.size() != nf ||
      c.omega.size() != nw || c.omega_window.size() != nw || c.active.size() != nw ||
      c.norm.size() != nw || c.crop.size() != nw ||
      (!c.absorber.empty() && c.absorber.size() != nf) ||
      (c.split && c.absorber_coarse.size() != std::size_t(c.nt)) ||
      c.undamped.size() % (2 * nf) || c.solution.size() != 7 || c.error.size() != 7 ||
      c.tableau.size() != 49)
    throw std::invalid_argument("inconsistent workspace tables");
  if (!(c.dt > 0) || !(c.log_step > 0) || c.taps.empty() || c.taps.size() > nf ||
      (c.plasma && (c.rate.empty() || c.rate.size() % 4)) ||
      (c.planning != "measure" && c.planning != "estimate"))
    throw std::invalid_argument("invalid response or planning tables");
  for (std::size_t w = 0; w < nw; ++w)
    if (c.crop[w] > nf / 2 || (c.plasma && c.active[w] && !(c.omega[w] > 0)))
      throw std::invalid_argument("invalid crop index or active plasma frequency");
  if (c.contraction != "auto" && c.contraction != "blas" && c.contraction != "hand")
    throw std::invalid_argument("contraction provider must be auto, blas or hand");
  if (!c.admittance.empty() &&
      (c.admittance.size() != size || c.spatial_axis.size() != std::size_t(c.modes) ||
       c.spectral_weights.size() != std::size_t(c.modes)))
    throw std::invalid_argument("inconsistent spatial statistics tables");
#ifndef CPUNATIVE_BLAS
  if (c.contraction == "blas")
    throw std::invalid_argument("BLAS provider not built");
  c.contraction = "hand";
#else
  if (c.contraction == "auto") {
    c.contraction = "blas";
#if defined(__linux__) && defined(__x86_64__) && !defined(CPUNATIVE_MKL_BLAS)
    // Source and ordinary-driver trials favour flat products for a single
    // mode at wider teams. Keep vendor kernels for serial and multimode work.
    if (c.modes == 1 && c.threads >= 8)
      c.contraction = "hand";
#endif
  }
#endif
  return c;
}

// The same stable phi recurrence as the independent exponential method:
// evaluate near zero by a scaled Taylor series, then double, rather than
// subtracting 1.
SOLPROP_CPUNATIVE_STRICT_FUNCTION
void phi_values(Complex x, Complex exp_x, int count, Complex* out) {
  SOLPROP_CPUNATIVE_STRICT_BLOCK
  double reciprocal[18]{1.0};
  for (int k = 1; k < 18; ++k)
    reciprocal[k] = reciprocal[k - 1] / k;
  const double magnitude = std::abs(x);
  if (magnitude == 0) {
    for (int k = 0; k <= count; ++k)
      out[k] = reciprocal[k];
    return;
  }
  if (magnitude > 3.0) {
    Complex inverse = strict_divide(Complex(1), x);
    out[0] = exp_x;
    for (int k = 1; k <= count; ++k)
      out[k] = strict_mul(strict_sub(out[k - 1], Complex(reciprocal[k - 1])), inverse);
    return;
  }
  int squarings = magnitude > .5 ? int(std::ceil(std::log2(magnitude / .5))) : 0;
  Complex scaled = strict_scale(std::pow(.5, squarings), x);
  Complex term = reciprocal[count], total = term;
  for (int k = 1; k < 24; ++k) {
    term = strict_scale(1.0 / (k + count), strict_mul(term, scaled));
    total = strict_add(total, term);
    if (std::abs(term) <=
        .25 * std::numeric_limits<double>::epsilon() * std::abs(total))
      break;
  }
  out[count] = total;
  for (int k = count - 1; k > 0; --k)
    out[k] = strict_add(reciprocal[k], strict_mul(scaled, out[k + 1]));
  out[0] = std::exp(scaled);
  for (int pass = 0; pass < squarings; ++pass) {
    Complex old[18];
    std::copy_n(out, count + 1, old);
    out[0] = strict_mul(old[0], old[0]);
    for (int k = 1; k <= count; ++k) {
      Complex sum = strict_mul(old[0], old[k]);
      for (int i = 1; i <= k; ++i)
        sum = strict_accumulate(sum, reciprocal[k - i], old[i]);
      out[k] = strict_scale(std::pow(.5, k), sum);
    }
  }
}
} // namespace

State::State(Config config)
    : c_(validated(std::move(config))), size_(std::size_t(c_.modes) * c_.nw),
      fine_size_(std::size_t(c_.modes) * c_.nf),
      fine_bins_(((c_.nf / 2 + 1) + 3) / 4 * 4), streams_(c_.plasma ? 3 : 1),
      raman_bins_((c_.nf + 4) / 4 * 4), y_(size_, c_.threads),
      trial_(size_, c_.threads), input_(size_, c_.threads), source_(size_, c_.threads),
      stages_(size_ * c_.solution.size(), c_.threads), fsal_(size_, c_.threads),
      transform_(std::size_t(c_.modes) * fine_bins_, c_.threads),
      projected_spectrum_(std::size_t(streams_) * c_.modes * fine_bins_, c_.threads),
      coarse_spectrum_(projected_spectrum_.size(), c_.threads),
      history_(size_ * c_.slots, c_.threads),
      history_state_(size_ * c_.slots, c_.threads),
      history_rate_(size_ * c_.slots, c_.threads),
      moments_(size_ * (c_.slots + 1), c_.threads),
      compensation_(size_ * c_.slots, c_.threads), endpoint_(size_, c_.threads),
      modal_(fine_size_, c_.threads), node_(std::size_t(c_.chunk) * c_.nf, c_.threads),
      modal_quadrature_(c_.analytic_kerr ? fine_size_ : 0, c_.threads),
      node_quadrature_(c_.analytic_kerr ? node_.size() : 0, c_.threads),
      polarisation_(node_.size(), c_.threads), current_(node_.size(), c_.threads),
      edge_(node_.size(), c_.threads), rate_(node_.size(), c_.threads),
      fraction_(node_.size(), c_.threads), scan_(node_.size(), c_.threads),
      temporary_(node_.size(), c_.threads),
      projected_(std::size_t(streams_) * fine_size_, c_.threads),
      coarse_projected_(projected_.size(), c_.threads),
      raman_time_(c_.undamped.empty() ? 0 : 2 * node_.size(), c_.threads),
      raman_kernel_time_(c_.undamped.empty() ? 0 : 2 * c_.nf, c_.threads),
      raman_frequency_(c_.undamped.empty() ? 0 : std::size_t(c_.chunk) * raman_bins_,
                       c_.threads),
      raman_kernel_(c_.undamped.empty() ? 0 : raman_bins_, c_.threads),
      statistics_time_(std::size_t(2) * c_.modes * c_.nt, c_.threads),
      statistics_nodes_(c_.admittance.empty() ? 0 : std::size_t(2) * c_.chunk * c_.nt,
                        c_.threads),
      statistics_axis_(c_.admittance.empty() ? 0 : 2 * c_.nt, c_.threads),
      statistics_power_(c_.nt, c_.threads), reductions_(8 * c_.threads, c_.threads),
      last_rates_(c_.undamped.size() / (2 * c_.nf),
                  std::numeric_limits<double>::quiet_NaN()),
      node_populations_(std::size_t(c_.nodes) * 8),
      chunk_populations_(std::size_t(c_.chunk) * 8) {
  const auto pairs = std::size_t(c_.modes) * (std::size_t(c_.modes) + 1) / 2;
  if (!c_.undamped.empty() && pairs < std::size_t(c_.nodes) &&
      pairs <= std::size_t(c_.chunk))
    raman_pairs_ = int(pairs);
  unsigned flags = c_.planning == "estimate" ? FFTW_ESTIMATE : FFTW_MEASURE;
  if (c_.nf % 8)
    flags |= FFTW_UNALIGNED;
  std::lock_guard lock(fftw_planner_mutex());
  const auto planning_start = std::chrono::steady_clock::now();
  if (!fftw_init_threads())
    throw std::runtime_error("FFTW thread initialization failed");
  // Another FFTW consumer may have selected an internally threaded plan.
  // Our plans run on independent rows inside our own OpenMP regions.
  fftw_plan_with_nthreads(1);
#ifdef CPUNATIVE_MKL_FFT
  // MKL's FFTW3 interface supports shared plans; declare the external team so
  // its descriptor can avoid internal locking. Intel's FFTW3 user-thread guide.
  fftw3_mkl.number_of_user_threads = c_.threads;
#endif
  inverse_ = fftw_plan_dft_c2r_1d(c_.nf, fft(transform_.data()), modal_.data(), flags);
  forward_ = fftw_plan_dft_r2c_1d(c_.nf, modal_.data(), fft(transform_.data()), flags);
  analytic_inverse_ = fftw_plan_dft_1d(c_.nt, fft(statistics_time_.data()),
                                       fft(statistics_time_.data()), FFTW_BACKWARD,
                                       flags | (c_.nt % 4 ? FFTW_UNALIGNED : 0));
  if (c_.split) {
    coarse_inverse_ =
        fftw_plan_dft_c2r_1d(c_.nt, fft(transform_.data()), modal_.data(), flags);
    coarse_forward_ =
        fftw_plan_dft_r2c_1d(c_.nt, modal_.data(), fft(transform_.data()), flags);
  }
  if (!c_.undamped.empty()) {
    raman_forward_ = fftw_plan_dft_r2c_1d(2 * c_.nf, raman_time_.data(),
                                          fft(raman_frequency_.data()), flags);
    raman_inverse_ = fftw_plan_dft_c2r_1d(2 * c_.nf, fft(raman_frequency_.data()),
                                          raman_time_.data(), flags);
  }
  if (!inverse_ || !forward_ || !analytic_inverse_ ||
      (c_.split && (!coarse_inverse_ || !coarse_forward_)) ||
      (!c_.undamped.empty() && (!raman_forward_ || !raman_inverse_))) {
    for (auto plan : {inverse_, forward_, analytic_inverse_, coarse_inverse_,
                      coarse_forward_, raman_forward_, raman_inverse_})
      if (plan)
        fftw_destroy_plan(plan);
    throw std::runtime_error("FFTW plan creation failed");
  }
  planning_seconds_ =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - planning_start)
          .count();
}

State::~State() {
  std::lock_guard lock(fftw_planner_mutex());
  for (fftw_plan plan : {inverse_, forward_, raman_forward_, raman_inverse_,
                         analytic_inverse_, coarse_inverse_, coarse_forward_})
    if (plan)
      fftw_destroy_plan(plan);
  // FFTW cleanup is process-wide; this workspace never owns it.
}

void State::validate_stage(const Stage& stage) const {
  if (stage.delta.size() != c_.phi.size() / size_ ||
      stage.prefactor.size() != c_.prefactor.size() / size_ ||
      stage.raman_rates.size() != last_rates_.size() || !(stage.taper > 0))
    throw std::invalid_argument("stage table shape mismatch");
}
void State::validate_rate(std::span<const double> rate) const {
  if (rate.size() != c_.phi.size() / size_)
    throw std::invalid_argument("linear weight shape mismatch");
}

Complex State::combination(const std::vector<Complex>& terms,
                           std::span<const double> weights, std::size_t i) const {
  Complex result{};
  for (std::size_t k = 0; k < weights.size(); ++k)
    result += weights[k] * terms[k * size_ + i];
  return result;
}

void State::load(std::span<const Complex> field) {
  if (field.size() != size_)
    throw std::invalid_argument("initial field shape mismatch");
  for (std::size_t i = 0; i < size_; ++i)
    if (!std::isfinite(field[i].real()) || !std::isfinite(field[i].imag()))
      throw std::invalid_argument("nonfinite initial field");
  for (std::size_t i = 0; i < size_; ++i) {
    y_[i] = c_.active[i % c_.nw] ? field[i] : Complex{};
  }
  fsal_ready_ = false;
  evaluations_ = 0;
  quadrature_ = std::numeric_limits<double>::quiet_NaN();
}

void State::contract(int rows, int inner, int leading, const double* __restrict a,
                     const double* __restrict b, double* __restrict out, double scale,
                     bool add, int samples) {
  const int n = samples ? samples : c_.nf;
#ifdef CPUNATIVE_BLAS
  if (c_.contraction != "hand") {
    // Vendor calls are outside any OpenMP region; no nested worker pools.
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, n, inner, scale, a,
                leading, b, n, add ? 1. : 0., out, n);
    return;
  }
#endif
  constexpr int block = 128;
  const int blocks = (n + block - 1) / block;
  for_rows(c_.threads, std::size_t(rows) * blocks, [&](std::size_t index) {
    const int row = int(index / blocks), first = int(index % blocks) * block;
    const int count = std::min(block, n - first);
    double sum[block]{};
    for (int k = 0; k < inner; ++k) {
      const double weight = a[row * leading + k];
      const double* __restrict input = b + std::size_t(k) * n + first;
#pragma omp simd
      for (int t = 0; t < count; ++t)
        sum[t] += weight * input[t];
    }
    double* __restrict output = out + std::size_t(row) * n + first;
    if (add) {
#pragma omp simd
      for (int t = 0; t < count; ++t)
        output[t] += scale * sum[t];
    } else {
#pragma omp simd
      for (int t = 0; t < count; ++t)
        output[t] = scale * sum[t];
    }
  });
}

void State::to_time(const Complex* field) {
  const int n = c_.nf, bins = fine_bins_, nw = c_.nw;
  for_rows(c_.threads, c_.modes, [&](int m) {
    Complex* f = transform_.data() + std::size_t(m) * bins;
    double* t = modal_.data() + std::size_t(m) * n;
    std::fill_n(f, bins, Complex{});
    for (int w = 0; w < nw; ++w)
      f[c_.crop[w]] = field[std::size_t(m) * nw + w];
    fftw_execute_dft_c2r(inverse_, fft(f), t);
#pragma omp simd
    for (int i = 0; i < n; ++i)
      t[i] /= n;
    if (c_.analytic_kerr) {
      // Real synthesis commutes with the temporal Hilbert transform. Form its
      // quadrature once per spectral column, before the node contractions.
      std::fill_n(f, bins, Complex{});
      for (int w = 0; w < nw; ++w) {
        const auto k = c_.crop[w];
        if (k != 0 && k != std::size_t(n / 2)) {
          const Complex value = field[std::size_t(m) * nw + w];
          f[k] = {value.imag(), -value.real()};
        }
      }
      double* q = modal_quadrature_.data() + std::size_t(m) * n;
      fftw_execute_dft_c2r(inverse_, fft(f), q);
#pragma omp simd
      for (int i = 0; i < n; ++i)
        q[i] /= n;
    }
  });
}

SOLPROP_CPUNATIVE_STRICT_FUNCTION
void State::scan(const double* __restrict values, double* __restrict filtered,
                 double* __restrict output) const {
  SOLPROP_CPUNATIVE_STRICT_BLOCK
  const int n = c_.nf, reach = int(c_.taps.size()) - 1;
  const double* __restrict taps = c_.taps.data();
  auto boundary = [&](int t) {
    double sum = taps[0] * values[t];
    for (int k = 1; k <= reach; ++k) {
      double left = t >= k ? values[t - k] : 2 * values[0] - values[k - t];
      double right =
          t + k < n ? values[t + k] : 2 * values[n - 1] - values[2 * (n - 1) - t - k];
      sum += taps[k] * (left + right);
    }
    filtered[t] = sum;
  };
  for (int t = 0; t < reach; ++t)
    boundary(t);
  for (int t = std::max(reach, n - reach); t < n; ++t)
    boundary(t);
  // Accumulate each sample in exactly the stencil's tap order, but vectorise
  // independent samples inside a cache-resident block. No boundary branches in
  // this loop, and no SIMD annotation on the following causal recurrence.
  constexpr int block = 128;
  for (int first = reach; first < n - reach; first += block) {
    const int count = std::min(block, n - reach - first);
    double sums[block];
#pragma omp simd
    for (int t = 0; t < count; ++t)
      sums[t] = taps[0] * values[first + t];
    for (int k = 1; k <= reach; ++k) {
      const double coefficient = taps[k];
#pragma omp simd
      for (int t = 0; t < count; ++t)
        sums[t] += coefficient * (values[first + t - k] + values[first + t + k]);
    }
    std::copy_n(sums, count, filtered + first);
  }
  output[0] = 0;
  // A recurrence: each column is sequential; independent columns run in
  // parallel.
  for (int t = 1; t < n; ++t)
    output[t] = output[t - 1] + c_.dt * (filtered[t] + filtered[t - 1]) / 2;
}

void State::update_raman(const Stage& stage) {
  if (c_.undamped.empty() || stage.raman_rates == last_rates_)
    return;
  const int n = 2 * c_.nf;
  if (stage.raman_rates.size() * n != c_.undamped.size())
    throw std::invalid_argument("Raman rate shape mismatch");
  for_rows(c_.threads, n, [&](int t) {
    double sum = 0;
    for (std::size_t k = 0; k < stage.raman_rates.size(); ++k)
      sum += c_.undamped[k * n + t] * std::exp(-stage.raman_rates[k] * t * c_.dt);
    raman_kernel_time_[t] = c_.raman_fraction * sum;
  });
  fftw_execute_dft_r2c(raman_forward_, raman_kernel_time_.data(),
                       fft(raman_kernel_.data()));
  last_rates_ = stage.raman_rates;
}

void State::prepare_raman_pairs() {
  if (!raman_pairs_)
    return;
  // For real static maps E_j = sum_m A_jm E_m, the linear convolution of
  // E_j^2 is a sum of convolved modal pairs. Compute each unordered pair once;
  // cross terms carry two. This uses the already allocated Raman row scratch.
  const int n = c_.nf;
  for_rows(c_.threads, raman_pairs_, [&](int pair) {
    int first = 0, second = pair;
    while (second >= c_.modes - first) {
      second -= c_.modes - first;
      ++first;
    }
    second += first;
    const auto* a = modal_.data() + std::size_t(first) * n;
    const auto* b = modal_.data() + std::size_t(second) * n;
    auto* time = raman_time_.data() + std::size_t(pair) * 2 * n;
    auto* frequency = raman_frequency_.data() + std::size_t(pair) * raman_bins_;
    const double multiplier = first == second ? 1 : 2;
#pragma omp simd
    for (int t = 0; t < n; ++t) {
      time[t] = multiplier * a[t] * b[t];
      time[t + n] = 0;
    }
    fftw_execute_dft_r2c(raman_forward_, time, fft(frequency));
    for (int w = 0; w < raman_bins_; ++w)
      frequency[w] *= raman_kernel_[w];
    fftw_execute_dft_c2r(raman_inverse_, fft(frequency), time);
  });
}

SOLPROP_CPUNATIVE_STRICT_FUNCTION
void State::response(int row, const Stage& stage, bool diagnostic, int node) {
  SOLPROP_CPUNATIVE_STRICT_BLOCK
  const bool profile = c_.profile && !diagnostic;
  auto previous = profile ? std::chrono::steady_clock::now()
                          : std::chrono::steady_clock::time_point{};
  auto mark = [&](int phase) {
    if (profile) {
      const auto now = std::chrono::steady_clock::now();
      chunk_populations_[8 * row + phase] =
          std::chrono::duration<double>(now - previous).count();
      previous = now;
    }
  };
  const int n = c_.nf;
  const auto offset = std::size_t(row) * n;
  const double* __restrict e = node_.data() + offset;
  double* __restrict p = polarisation_.data() + offset;
  double* __restrict j = current_.data() + offset;
  double* __restrict edge = edge_.data() + offset;
  double* __restrict rate = rate_.data() + offset;
  double* __restrict fraction = fraction_.data() + offset;
  double* __restrict integrated = scan_.data() + offset;
  double* __restrict temporary = temporary_.data() + offset;
  if (!diagnostic && c_.analytic_kerr) {
    const double* q = node_quadrature_.data() + offset;
#pragma omp simd
    for (int t = 0; t < n; ++t)
      p[t] = c_.kerr * stage.kerr_density * (e[t] * e[t] + q[t] * q[t]) * e[t];
  } else {
#pragma omp simd
    for (int t = 0; t < (diagnostic ? 0 : n); ++t)
      p[t] = c_.kerr * stage.kerr_density * e[t] * e[t] * e[t];
  }
  mark(0);
  if (!diagnostic && !c_.undamped.empty()) {
    const double* convolution;
    if (raman_pairs_ && node >= 0) {
      std::fill_n(temporary, n, 0.);
      const auto* map = c_.synthesis.data() + std::size_t(node) * c_.modes;
      int pair = 0;
      for (int first = 0; first < c_.modes; ++first)
        for (int second = first; second < c_.modes; ++second, ++pair) {
          const double coefficient =
              stage.taper * stage.taper * map[first] * map[second];
          const auto* time = raman_time_.data() + std::size_t(pair) * 2 * n;
#pragma omp simd
          for (int t = 0; t < n; ++t)
            temporary[t] += coefficient * time[t];
        }
      convolution = temporary;
    } else {
      double* time = raman_time_.data() + 2 * offset;
      Complex* frequency = raman_frequency_.data() + std::size_t(row) * raman_bins_;
#pragma omp simd
      for (int t = 0; t < n; ++t) {
        time[t] = e[t] * e[t];
        time[t + n] = 0;
      }
      fftw_execute_dft_r2c(raman_forward_, time, fft(frequency));
      for (int w = 0; w < raman_bins_; ++w)
        frequency[w] *= raman_kernel_[w];
      fftw_execute_dft_c2r(raman_inverse_, fft(frequency), time);
      convolution = time;
    }
#pragma omp simd
    for (int t = 0; t < n; ++t)
      p[t] +=
          c_.epsilon0 * stage.raman_density * e[t] * convolution[t] * c_.dt / (2 * n);
  }
#pragma omp simd
  for (int t = 0; t < n; ++t)
    p[t] *= c_.window[t];
  mark(1);
  if (!c_.plasma) {
    if (profile)
      std::fill_n(chunk_populations_.data() + 8 * row + 2, 5, 0.);
    return;
  }
  ionisation_rates(e, rate);
  mark(2);
  scan(rate, temporary, integrated);
  mark(3);
  for (int t = 0; t < n; ++t) {
    fraction[t] =
        1 - (1 - stage.preionisation) *
                std::exp(-(c_.band ? std::max(0.0, integrated[t]) : integrated[t]));
    j[t] = fraction[t] * c_.charge * e[t];
  }
  mark(4);
  if (diagnostic) {
    // Per-node records are reduced after the worker region by the caller.
    double* parts = chunk_populations_.data() + std::size_t(row) * 8;
    parts[0] = fraction[n - 1];
    parts[1] = *std::max_element(rate, rate + n);
    double dip = 0, rise = 0, paid = 0, early = 0, peak = 0;
    for (int t = 0; t < n; ++t) {
      const double fe = std::abs(fraction[t] * e[t]);
      peak = std::max(peak, fe);
      if (t < 2 * (int(c_.taps.size()) - 1))
        early = std::max(early, fe);
      if (t) {
        double increment = integrated[t] - integrated[t - 1];
        dip = std::max(dip, -increment);
        rise = std::max(rise, increment);
        paid += c_.dt *
                (rate[t] * (1 - fraction[t]) + rate[t - 1] * (1 - fraction[t - 1])) / 2;
      }
    }
    const double freed = fraction[n - 1] - fraction[0];
    parts[2] = dip;
    parts[3] = rise;
    parts[4] = std::abs(paid - freed);
    parts[5] = freed;
    parts[6] = early;
    parts[7] = peak;
    return;
  }
  scan(j, temporary, integrated);
  mark(5);
#pragma omp simd
  for (int t = 0; t < n; ++t) {
    const double squared = e[t] * e[t];
    const double loss = rate[t] > 0 && squared > 0
                            ? c_.potential * rate[t] * (1 - fraction[t]) / squared
                            : 0;
    const double value = stage.plasma_density * (integrated[t] + loss * e[t]);
    j[t] = c_.window[t] * value;
    edge[t] = c_.window_derivative[t] * value;
  }
  mark(6);
}

void State::spectral_source(const Complex* field, Complex* output, const Stage& stage,
                            bool diagnostic) {
  PhaseTimer timer(c_.profile, timings_);
  const int n = c_.nf, nw = c_.nw, modes = c_.modes, nodes = c_.nodes;
  // Real static maps commute with temporal transforms. The unused coarse
  // spectrum buffer owns aligned complex storage for the packed node spectra.
  auto* packed = coarse_spectrum_.data();
  std::fill_n(projected_spectrum_.data(), std::size_t(streams_) * size_, Complex{});
  diagnostics_.fill(0);
  if (!diagnostic)
    update_raman(stage);
  timer.mark(1);
  for (int begin = 0; begin < nodes; begin += c_.chunk) {
    const int count = std::min(c_.chunk, nodes - begin);
    contract(count, modes, modes, c_.synthesis.data() + begin * modes,
             reinterpret_cast<const double*>(field), reinterpret_cast<double*>(packed),
             stage.taper, false, 2 * nw);
    timer.mark(2);
    for_rows(c_.threads, count, [&](int row) {
      const Complex* spectrum = packed + std::size_t(row) * nw;
      Complex* fine = transform_.data() + std::size_t(row) * fine_bins_;
      double* time = node_.data() + std::size_t(row) * n;
      std::fill_n(fine, fine_bins_, Complex{});
      for (int w = 0; w < nw; ++w)
        fine[c_.crop[w]] = spectrum[w];
      fftw_execute_dft_c2r(inverse_, fft(fine), time);
#pragma omp simd
      for (int t = 0; t < n; ++t)
        time[t] /= n;
      if (c_.analytic_kerr && !diagnostic) {
        std::fill_n(fine, fine_bins_, Complex{});
        for (int w = 0; w < nw; ++w) {
          const auto k = c_.crop[w];
          if (k != 0 && k != std::size_t(n / 2))
            fine[k] = {spectrum[w].imag(), -spectrum[w].real()};
        }
        double* quadrature = node_quadrature_.data() + std::size_t(row) * n;
        fftw_execute_dft_c2r(inverse_, fft(fine), quadrature);
#pragma omp simd
        for (int t = 0; t < n; ++t)
          quadrature[t] /= n;
      }
    });
    timer.mark(0);
    for_rows(c_.threads, count,
             [&](std::size_t row) { response(int(row), stage, diagnostic); });
    if (c_.profile && !diagnostic)
      for (int row = 0; row < count; ++row)
        for (int phase = 0; phase < 7; ++phase)
          response_timings_[phase] += chunk_populations_[8 * row + phase];
    timer.mark(2);
    if (diagnostic) {
      for (int row = count - 1; row >= 0; --row) {
        const auto from = std::size_t(row) * 8, to = std::size_t(begin + row) * 8;
        for (int k = 0; k < 6; ++k)
          diagnostics_[k] = std::max(diagnostics_[k], chunk_populations_[from + 2 + k]);
        std::copy_n(chunk_populations_.data() + from, 8, node_populations_.data() + to);
      }
      continue;
    }
    for (int stream = 0; stream < streams_; ++stream) {
      double* values = stream == 0   ? polarisation_.data()
                       : stream == 1 ? current_.data()
                                     : edge_.data();
      for_rows(c_.threads, count, [&](int row) {
        Complex* fine = transform_.data() + std::size_t(row) * fine_bins_;
        fftw_execute_dft_r2c(forward_, values + std::size_t(row) * n, fft(fine));
        Complex* spectrum = packed + std::size_t(row) * nw;
        for (int w = 0; w < nw; ++w)
          spectrum[w] = fine[c_.crop[w]];
      });
      timer.mark(4);
      contract(modes, count, nodes, c_.projection.data() + begin,
               reinterpret_cast<double*>(packed),
               reinterpret_cast<double*>(projected_spectrum_.data() + stream * size_),
               1 / stage.taper, true, 2 * nw);
      timer.mark(3);
    }
  }
  if (diagnostic)
    return;
  if (!c_.split && !c_.absorber.empty()) {
    to_time(field);
    for_rows(c_.threads, modes, [&](int m) {
      double* time = modal_.data() + std::size_t(m) * n;
#pragma omp simd
      for (int t = 0; t < n; ++t)
        time[t] *= -.5 * c_.absorber[t];
      fftw_execute_dft_r2c(forward_, time,
                           fft(transform_.data() + std::size_t(m) * fine_bins_));
    });
  }
  timer.mark(5);
  for_rows(c_.threads, size_, [&](std::size_t i) {
    const int w = int(i % nw), m = int(i / nw);
    if (!c_.active[w]) {
      output[i] = {};
      return;
    }
    Complex p = projected_spectrum_[i];
    if (c_.plasma)
      p += divide_iomega(projected_spectrum_[size_ + i], c_.omega[w]) +
           projected_spectrum_[2 * size_ + i] / (c_.omega[w] * c_.omega[w]);
    Complex value =
        combination(c_.prefactor, stage.prefactor, i) * c_.omega_window[w] * p;
    if (!c_.split && !c_.absorber.empty())
      value += transform_[std::size_t(m) * fine_bins_ + c_.crop[w]];
    output[i] = value;
  });
  timer.mark(6);
}

void State::source(const Complex* field, Complex* output, const Stage& stage,
                   bool diagnostic, bool score) {
  validate_stage(stage);
  // On a square spatial map, moving contractions across the FFTs leaves their
  // count unchanged while reducing the matrix products to retained frequencies.
  if (!c_.admittance.empty() && c_.nodes == c_.modes && c_.nf >= 2 * c_.nw &&
      c_.coarse.empty()) {
    spectral_source(field, output, stage, diagnostic);
    return;
  }
  PhaseTimer timer(c_.profile, timings_);
  to_time(field);
  timer.mark(0);
  if (!diagnostic) {
    update_raman(stage);
    prepare_raman_pairs();
  }
  timer.mark(1);
  const int n = c_.nf, modes = c_.modes, nodes = c_.nodes;
  std::fill_n(projected_.data(), projected_.size(), 0.0);
  std::fill_n(coarse_projected_.data(), coarse_projected_.size(), 0.0);
  diagnostics_.fill(0);
  // Only the configured node chunk is resident. Tail rows use the same
  // new-array FFTW plans and disjoint, equally aligned arrays.
  for (int begin = 0; begin < nodes; begin += c_.chunk) {
    const int count = std::min(c_.chunk, nodes - begin);
    contract(count, modes, modes, c_.synthesis.data() + begin * modes, modal_.data(),
             node_.data(), stage.taper, false);
    if (c_.analytic_kerr && !diagnostic)
      contract(count, modes, modes, c_.synthesis.data() + begin * modes,
               modal_quadrature_.data(), node_quadrature_.data(), stage.taper, false);
    for_rows(c_.threads, count, [&](std::size_t row) {
      response(int(row), stage, diagnostic, begin + int(row));
    });
    if (c_.profile && !diagnostic)
      for (int row = 0; row < count; ++row)
        for (int phase = 0; phase < 7; ++phase)
          response_timings_[phase] += chunk_populations_[8 * row + phase];
    timer.mark(2);
    if (diagnostic) {
      // Move backwards so a later chunk cannot overwrite its own packed rows.
      for (int row = count - 1; row >= 0; --row) {
        const auto from = std::size_t(row) * 8, to = std::size_t(begin + row) * 8;
        for (int k = 0; k < 6; ++k)
          diagnostics_[k] = std::max(diagnostics_[k], chunk_populations_[from + 2 + k]);
        std::copy_n(chunk_populations_.data() + from, 8, node_populations_.data() + to);
      }
      continue;
    }
    for (int stream = 0; stream < streams_; ++stream) {
      const double* values = stream == 0   ? polarisation_.data()
                             : stream == 1 ? current_.data()
                                           : edge_.data();
      contract(modes, count, nodes, c_.projection.data() + begin, values,
               projected_.data() + stream * fine_size_, 1 / stage.taper, true);
      if (score && !c_.coarse.empty())
        contract(modes, count, nodes, c_.coarse.data() + begin, values,
                 coarse_projected_.data() + stream * fine_size_, 1 / stage.taper, true);
    }
    timer.mark(3);
  }
  for_rows(c_.threads, (diagnostic ? 0 : streams_ * modes), [&](int row) {
    fftw_execute_dft_r2c(
        forward_, projected_.data() + std::size_t(row) * n,
        fft(projected_spectrum_.data() + std::size_t(row) * fine_bins_));
  });
  timer.mark(4);
  if (diagnostic)
    return;
  if (!c_.split && !c_.absorber.empty()) {
    for_rows(c_.threads, modes, [&](int m) {
      double* time = modal_.data() + std::size_t(m) * n;
      for (int t = 0; t < n; ++t)
        time[t] *= -.5 * c_.absorber[t];
      fftw_execute_dft_r2c(forward_, time,
                           fft(transform_.data() + std::size_t(m) * fine_bins_));
    });
  }
  timer.mark(5);
  for_rows(c_.threads, size_, [&](std::size_t i) {
    const int w = int(i % c_.nw), m = int(i / c_.nw);
    if (!c_.active[w]) {
      output[i] = {};
      return;
    }
    const auto bin = std::size_t(m) * fine_bins_ + c_.crop[w];
    Complex p = projected_spectrum_[bin];
    if (c_.plasma) {
      const auto stride = std::size_t(modes) * fine_bins_;
      p += divide_iomega(projected_spectrum_[stride + bin], c_.omega[w]) +
           projected_spectrum_[2 * stride + bin] / (c_.omega[w] * c_.omega[w]);
    }
    Complex value =
        combination(c_.prefactor, stage.prefactor, i) * c_.omega_window[w] * p;
    if (!c_.split && !c_.absorber.empty())
      value += transform_[bin];
    output[i] = value;
  });
  timer.mark(6);
  if (score && !c_.coarse.empty()) {
    for_rows(c_.threads, streams_ * modes, [&](int row) {
      fftw_execute_dft_r2c(
          forward_, coarse_projected_.data() + std::size_t(row) * n,
          fft(coarse_spectrum_.data() + std::size_t(row) * fine_bins_));
    });
    double difference = 0, total = 0;
    {
      const auto reduced = reduce_rows(
          c_.threads, size_, reductions_.data(),
          [&](std::size_t i, double* partial) {
            double& difference = partial[0];
            double& total = partial[1];

            const int w = int(i % c_.nw), m = int(i / c_.nw);
            if (!c_.active[w])
              return;
            const auto bin = std::size_t(m) * fine_bins_ + c_.crop[w],
                       stride = std::size_t(modes) * fine_bins_;
            Complex fine = projected_spectrum_[bin], coarse = coarse_spectrum_[bin];
            if (c_.plasma) {
              fine +=
                  divide_iomega(projected_spectrum_[stride + bin], c_.omega[w]) +
                  projected_spectrum_[2 * stride + bin] / (c_.omega[w] * c_.omega[w]);
              coarse +=
                  divide_iomega(coarse_spectrum_[stride + bin], c_.omega[w]) +
                  coarse_spectrum_[2 * stride + bin] / (c_.omega[w] * c_.omega[w]);
            }
            const Complex weight =
                combination(c_.prefactor, stage.prefactor, i) * c_.omega_window[w];
            difference += std::norm(weight * coarse - weight * fine);
            total += std::norm(weight * fine);
          },
          false);
      difference = reduced[0];
      total = reduced[1];
    }
    pending_quadrature_ = total > 0 ? std::sqrt(difference / total)
                                    : std::numeric_limits<double>::quiet_NaN();
  }
  timer.mark(7);
}

// Private diagnostic probes use the actual workspace and plans. They are not
// used by the stepping path and return owning host copies for independent
// tests.
std::vector<double> State::temporal_probe() {
  to_time(y_.data());
  return {modal_.data(), modal_.data() + fine_size_};
}
std::vector<Complex> State::roundtrip_probe() {
  to_time(y_.data());
  std::vector<Complex> out(size_);
  for (int m = 0; m < c_.modes; ++m) {
    auto* frequency = transform_.data() + std::size_t(m) * fine_bins_;
    fftw_execute_dft_r2c(forward_, modal_.data() + std::size_t(m) * c_.nf,
                         fft(frequency));
    for (int w = 0; w < c_.nw; ++w)
      out[std::size_t(m) * c_.nw + w] = frequency[c_.crop[w]];
  }
  return out;
}
std::vector<double> State::response_probe(std::span<const double> field,
                                          const Stage& stage) {
  validate_stage(stage);
  if (field.size() != std::size_t(c_.nf))
    throw std::invalid_argument("response probe shape mismatch");
  update_raman(stage);
  std::copy(field.begin(), field.end(), node_.data());
  response(0, stage, false);
  std::vector<double> out(5 * c_.nf);
  int row = 0;
  for (const auto* buffer : {&polarisation_, &current_, &edge_, &rate_, &fraction_}) {
    std::copy_n(buffer->data(), c_.nf, out.data() + row * c_.nf);
    ++row;
  }
  return out;
}

std::vector<Complex> State::evaluate(const Stage& stage) {
  source(y_.data(), source_.data(), stage);
  ++evaluations_;
  return {source_.data(), source_.data() + size_};
}

std::array<double, 3> State::trial(double dz, const std::vector<Stage>& stages,
                                   const std::vector<double>& whole) {
  PhaseTimer timer(c_.profile, timings_);
  if (stages.size() != c_.solution.size())
    throw std::invalid_argument("stage count mismatch");
  validate_rate(whole);
  for (const auto& stage : stages)
    validate_stage(stage);
  const auto count = stages.size();
  for (std::size_t s = 0; s < count; ++s) {
    Complex* derivative = stages_.data() + s * size_;
    if (s == 0 && fsal_ready_) {
      std::copy_n(fsal_.data(), size_, derivative);
      continue;
    }
    for_rows(c_.threads, size_, [&](std::size_t i) {
      Complex sum{};
      for (std::size_t j = 0; j < s; ++j)
        sum += c_.tableau[s * count + j] * stages_[j * size_ + i];
      input_[i] =
          std::exp(combination(c_.phi, stages[s].delta, i)) * (y_[i] + dz * sum);
    });
    source(input_.data(), derivative, stages[s], false, s + 1 == count);
    ++evaluations_;
    for_rows(c_.threads, size_, [&](std::size_t i) {
      derivative[i] *= std::exp(-combination(c_.phi, stages[s].delta, i));
    });
  }
  double old = 0, next = 0, error = 0;
  {
    const auto reduced = reduce_rows(
        c_.threads, size_, reductions_.data(),
        [&](std::size_t i, double* partial) {
          double& old = partial[0];
          double& next = partial[1];
          double& error = partial[2];

          Complex sum{}, defect{};
          for (std::size_t s = 0; s < count; ++s) {
            sum += c_.solution[s] * stages_[s * size_ + i];
            defect += c_.error[s] * stages_[s * size_ + i];
          }
          Complex flow = std::exp(combination(c_.phi, whole, i));
          trial_[i] = flow * (y_[i] + dz * sum);
          if (c_.norm[i % c_.nw]) {
            old += std::norm(y_[i]);
            next += std::norm(trial_[i]);
            error += std::norm(dz * flow * defect);
          }
          // Recycle in the next step's laboratory frame only after acceptance.
          source_[i] = flow * stages_[(count - 1) * size_ + i];
        },
        false);
    old = reduced[0];
    next = reduced[1];
    error = reduced[2];
  }
  timer.mark(9);
  return {old, next, error};
}

void State::apply_finish(double dz) {
  quadrature_ = pending_quadrature_;
  if (c_.split) {
    for_rows(c_.threads, c_.modes, [&](int m) {
      auto* time = modal_.data() + std::size_t(m) * c_.nf;
      auto* spectrum = transform_.data() + std::size_t(m) * fine_bins_;
      for (int w = 0; w < c_.nw; ++w)
        spectrum[w] = c_.active[w] ? trial_[std::size_t(m) * c_.nw + w] : Complex{};
      fftw_execute_dft_c2r(coarse_inverse_, fft(spectrum), time);
      for (int t = 0; t < c_.nt; ++t)
        time[t] *= std::exp(-.5 * dz * c_.absorber_coarse[t]) / c_.nt;
      fftw_execute_dft_r2c(coarse_forward_, time, fft(spectrum));
      for (int w = 0; w < c_.nw; ++w)
        trial_[std::size_t(m) * c_.nw + w] = spectrum[w];
    });
  }
  for_rows(c_.threads, size_, [&](std::size_t i) {
    y_[i] = c_.active[i % c_.nw] ? trial_[i] : Complex{};
  });
}

void State::accept(double dz) {
  apply_finish(dz);
  std::copy_n(source_.data(), size_, fsal_.data());
  fsal_ready_ = !c_.split;
}

void State::adams_start(const Stage& stage, const std::vector<double>& rate) {
  validate_rate(rate);
  source(y_.data(), history_.data(), stage);
  ++evaluations_;
  std::copy_n(y_.data(), size_, history_state_.data());
  for (std::size_t i = 0; i < size_; ++i)
    history_rate_[i] = combination(c_.phi, rate, i);
}

SOLPROP_CPUNATIVE_STRICT_FUNCTION
std::vector<double> State::adams_trial(double dz, const Stage& step,
                                       const Stage& sample, const AdamsTables& table) {
  SOLPROP_CPUNATIVE_STRICT_BLOCK
  PhaseTimer timer(c_.profile, timings_);
  const int width = table.width, count = table.count, order = table.order;
  if (count < 1 || count > c_.slots + 1 || count > 16 || width < 2 ||
      width > c_.slots || order < 1 || order >= width || table.rows < 1 ||
      table.rows > 3)
    throw std::invalid_argument("Adams table exceeds workspace");
  validate_stage(step);
  validate_stage(sample);
  validate_rate(table.rate);
  const bool compensated = !table.spans.empty();
  if (table.predictor.size() != std::size_t(order * order) ||
      table.corrector.size() != std::size_t(width * width) ||
      table.defects.size() != std::size_t(table.rows * (width + count)) ||
      table.slots.size() < std::size_t(std::max(order, width - 1)) ||
      (compensated && (table.spans.size() != std::size_t(width - 1) ||
                       table.carried.size() > std::size_t(width - 1) ||
                       table.endpoint.size() < std::size_t(order + 1) ||
                       table.comp_defects.size() != std::size_t(table.rows * width))))
    throw std::invalid_argument("Adams table shape mismatch");
  for (auto slot : table.slots)
    if (slot >= std::size_t(c_.slots))
      throw std::invalid_argument("Adams history slot outside workspace");
  double magnitude = 0;
  {
    const auto reduced = reduce_rows(
        c_.threads, size_, reductions_.data(),
        [&](std::size_t i, double* partial) {
          double& magnitude = partial[0];

          Complex x = combination(c_.phi, step.delta, i), flow = std::exp(x), phi[18];
          phi_values(x, flow, count, phi);
          double factorial = 1;
          for (int k = 0; k < count; ++k) {
            if (k)
              factorial *= k;
            moments_[std::size_t(k) * size_ + i] = strict_scale(factorial, phi[k + 1]);
          }
          Complex predicted = flow * y_[i], sum{};
          for (int k = 0; k < order; ++k) {
            Complex coefficient{};
            for (int t = 0; t < order; ++t)
              coefficient +=
                  table.predictor[t * order + k] * history_[table.slots[t] * size_ + i];
            sum += moments_[std::size_t(k) * size_ + i] * coefficient;
          }
          predicted += dz * sum;
          if (compensated) {
            Complex correction{}, chord = strict_divide(x, dz);
            for (int t = 0; t < width - 1; ++t) {
              const auto slot = table.slots[t] * size_ + i;
              Complex carried =
                  strict_mul(std::exp(table.spans[t] * x),
                             strict_mul(strict_sub(history_rate_[slot], chord),
                                        history_state_[slot]));
              compensation_[std::size_t(t) * size_ + i] = carried;
              if (t < int(table.carried.size()))
                correction = strict_accumulate(correction, table.carried[t], carried);
            }
            predicted = strict_accumulate(predicted, dz, correction);
          }
          input_[i] = predicted;
          if (c_.norm[i % c_.nw])
            magnitude += std::norm(y_[i]);
        },
        false);
    magnitude = reduced[0];
  }
  source(input_.data(), source_.data(), sample, false, true);
  ++evaluations_;
  const double divisor = c_.atol + c_.rtol * std::sqrt(magnitude);
  double norms[3]{};
  {
    const auto reduced = reduce_rows(
        c_.threads, size_, reductions_.data(),
        [&](std::size_t i, double* partial) {
          double* norms = partial;

          Complex x = combination(c_.phi, step.delta, i), sum{};
          for (int k = 0; k <= order; ++k) {
            Complex coefficient = table.corrector[k] * source_[i];
            for (int t = 1; t <= order; ++t)
              coefficient += table.corrector[t * width + k] *
                             history_[table.slots[t - 1] * size_ + i];
            sum += moments_[std::size_t(k) * size_ + i] * coefficient;
          }
          Complex corrected = std::exp(x) * y_[i] + dz * sum;
          if (compensated) {
            Complex endpoint_rate{};
            for (std::size_t k = 0; k < table.rate.size(); ++k)
              endpoint_rate = strict_accumulate(endpoint_rate, table.rate[k],
                                                c_.phi[k * size_ + i]);
            endpoint_[i] =
                strict_mul(strict_sub(endpoint_rate, strict_divide(x, dz)), input_[i]);
            Complex correction = strict_scale(table.endpoint[0], endpoint_[i]);
            for (int t = 0; t < order; ++t)
              correction = strict_accumulate(correction, table.endpoint[t + 1],
                                             compensation_[std::size_t(t) * size_ + i]);
            corrected = strict_accumulate(corrected, dz, correction);
          }
          trial_[i] = corrected;
          if (!c_.norm[i % c_.nw])
            return;
          for (int r = 0; r < table.rows; ++r) {
            const double* row = table.defects.data() + r * (width + count);
            Complex difference = strict_scale(row[0], source_[i]), weight{};
            for (int t = 1; t < width; ++t)
              difference = strict_accumulate(difference, row[t],
                                             history_[table.slots[t - 1] * size_ + i]);
            for (int k = 0; k < count; ++k)
              weight = strict_accumulate(weight, row[width + k],
                                         moments_[std::size_t(k) * size_ + i]);
            Complex defect = strict_scale(dz, strict_mul(difference, weight));
            if (compensated) {
              const double* cr = table.comp_defects.data() + r * width;
              Complex correction = strict_scale(cr[0], endpoint_[i]);
              for (int t = 0; t < width - 1; ++t)
                correction = strict_accumulate(
                    correction, cr[t + 1], compensation_[std::size_t(t) * size_ + i]);
              defect = strict_accumulate(defect, dz, correction);
            }
            norms[r] += std::norm(strict_divide(defect, divisor));
          }
        },
        false);
    std::copy_n(reduced.data(), 3, norms);
  }
  std::vector<double> result(table.rows);
  for (int r = 0; r < table.rows; ++r)
    result[r] = std::sqrt(norms[r]);
  timer.mark(9);
  return result;
}

void State::adams_accept(double dz, std::size_t slot, const std::vector<double>& rate) {
  validate_rate(rate);
  if (slot >= std::size_t(c_.slots))
    throw std::invalid_argument("Adams slot outside history");
  std::copy_n(source_.data(), size_, history_.data() + slot * size_);
  std::copy_n(input_.data(), size_, history_state_.data() + slot * size_);
  for (std::size_t i = 0; i < size_; ++i)
    history_rate_[slot * size_ + i] = combination(c_.phi, rate, i);
  apply_finish(dz);
  fsal_ready_ = false;
}

std::span<const Complex> State::history(const std::string& name) const {
  const auto* buffer = name == "store"    ? &history_
                       : name == "states" ? &history_state_
                       : name == "rates"  ? &history_rate_
                                          : nullptr;
  if (!buffer)
    throw std::invalid_argument("unknown Adams history row");
  return {buffer->data(), buffer->size()};
}

void State::load_history(const std::string& name, std::span<const Complex> values) {
  auto* buffer = name == "store"    ? &history_
                 : name == "states" ? &history_state_
                 : name == "rates"  ? &history_rate_
                                    : nullptr;
  if (!buffer || buffer->size() != values.size())
    throw std::invalid_argument("Adams history mismatch");
  std::copy(values.begin(), values.end(), buffer->data());
  fsal_ready_ = false;
}

std::vector<double> State::populations(const Stage& stage) {
  source(y_.data(), source_.data(), stage, true);
  std::vector<double> values(std::size_t(c_.nodes) * 2);
  for (int n = 0; n < c_.nodes; ++n) {
    values[2 * n] = node_populations_[8 * n];
    values[2 * n + 1] = node_populations_[8 * n + 1];
  }
  return values;
}

std::vector<double> State::statistics(std::span<const double> weight,
                                      std::span<const double> axis,
                                      double intensity_factor) {
  PhaseTimer timer(c_.profile, timings_);
  if (weight.size() != size_ ||
      (!axis.empty() && axis.size() != std::size_t(2 * c_.modes)))
    throw std::invalid_argument("statistics weight or axis shape mismatch");
  const int modes = c_.modes, n = c_.nt, nw = c_.nw;
  const double dt = c_.dt * c_.nf / n;
  std::vector<double> out(3 * modes + 4);
  // Analytic electric amplitude and its admittance-weighted partner. Their real
  // cross product is the cycle-averaged Poynting power, not |sqrt(W) E|^2.
  for_rows(c_.threads, 2 * modes, [&](int row) {
    const int m = row % modes;
    Complex* time = statistics_time_.data() + std::size_t(row) * n;
    std::fill_n(time, n, Complex{});
    for (int w = 0; w < nw; ++w) {
      const auto i = std::size_t(m) * nw + w;
      const double factor = (w == 0 || w == nw - 1) ? 1 : 2;
      time[w] = y_[i] * (factor / n) * (row < modes ? 1 : weight[i]);
    }
    fftw_execute_dft(analytic_inverse_, fft(time), fft(time));
  });
  for_rows(c_.threads, modes, [&](int m) {
    double total = 0, first = 0, energy = 0,
           peak = -std::numeric_limits<double>::infinity();
    for (int w = 0; w < nw; ++w) {
      const auto i = std::size_t(m) * nw + w;
      const double density = std::norm(y_[i]) * weight[i];
      total += density;
      first += density * c_.omega[w];
      energy += density * ((w == 0 || w == nw - 1) ? 1 : 4);
    }
    for (int t = 0; t < n; ++t) {
      const auto i = std::size_t(m) * n + t;
      peak = std::max(
          peak, std::real(statistics_time_[i] *
                          std::conj(statistics_time_[std::size_t(modes) * n + i])));
    }
    out[m] = total > 0 ? first / total : std::numeric_limits<double>::quiet_NaN();
    out[modes + m] = dt / n * energy;
    out[2 * modes + m] = peak;
  });
  double peak = -std::numeric_limits<double>::infinity(), intensity = peak;
  {
    const auto reduced = reduce_rows(
        c_.threads, n, reductions_.data(),
        [&](int t, double* partial) {
          double& peak = partial[0];
          double& intensity = partial[1];

          double power = 0;
          Complex electric[2]{}, magnetic[2]{};
          for (int m = 0; m < modes; ++m) {
            const auto i = std::size_t(m) * n + t;
            const Complex a = statistics_time_[i],
                          b = statistics_time_[std::size_t(modes) * n + i];
            power += std::real(a * std::conj(b));
            if (!axis.empty())
              for (int pol = 0; pol < 2; ++pol) {
                electric[pol] += axis[pol * modes + m] * a;
                magnetic[pol] += axis[pol * modes + m] * b;
              }
          }
          statistics_power_[t] = power;
          peak = std::max(peak, power);
          intensity =
              std::max(intensity, std::real(electric[0] * std::conj(magnetic[0]) +
                                            electric[1] * std::conj(magnetic[1])));
        },
        true);
    peak = reduced[0];
    intensity = reduced[1];
  }
  auto width = [&](bool outer) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (peak <= 0)
      return nan;
    const double level = peak / 2;
    const auto* p = statistics_power_.data();
    int left, right;
    if (outer) {
      left = 0;
      while (left < n && p[left] <= level)
        ++left;
      --left;
      right = n - 1;
      while (right >= 0 && p[right] <= level)
        --right;
      ++right;
    } else {
      int apex = int(std::max_element(p, p + n) - p);
      left = apex;
      while (left >= 0 && p[left] >= level)
        --left;
      right = apex;
      while (right < n && p[right] >= level)
        ++right;
    }
    if (left < 0 || right >= n)
      return nan;
    const double lo = left + (level - p[left]) / (p[left + 1] - p[left]);
    const double hi = right - (level - p[right]) / (p[right - 1] - p[right]);
    return (hi - lo) * dt;
  };
  out[3 * modes] = peak;
  out[3 * modes + 1] = width(false);
  out[3 * modes + 2] = width(true);
  out[3 * modes + 3] = intensity_factor * intensity;
  timer.mark(8);
  return out;
}

std::vector<double> State::spatial_statistics(double intensity_factor) {
  if (c_.admittance.empty())
    throw std::invalid_argument("spatial statistics require a plane measure");
  PhaseTimer timer(c_.profile, timings_);
  const int modes = c_.modes, nodes = c_.nodes, n = c_.nt, nw = c_.nw;
  const double dt = c_.dt * c_.nf / n;
  const double domega = 2 * std::acos(-1.0) / (n * dt);
  std::vector<double> out(4 + nw + 2 * nodes);
  const auto energy = reduce_rows(
      c_.threads, size_, reductions_.data(), [&](std::size_t i, double* partial) {
        const int w = int(i % nw), m = int(i / nw);
        const double multiplicity = w == 0 || w == nw - 1 ? 1 : 4;
        partial[0] += c_.spectral_weights[m] * c_.admittance[i].real() * multiplicity *
                      std::norm(y_[i]);
      });
  out[0] = intensity_factor * dt / n * energy[0];
  // Each pair is the analytic electric field and its magnetic closure. The
  // complex axial admittance, including evanescent bins, comes from translation.
  for_rows(c_.threads, 2 * modes, [&](int row) {
    const int m = row % modes;
    Complex* time = statistics_time_.data() + std::size_t(row) * n;
    std::fill_n(time, n, Complex{});
    for (int w = 0; w < nw; ++w) {
      const auto i = std::size_t(m) * nw + w;
      const double multiplicity = w == 0 || w == nw - 1 ? 1 : 2;
      time[w] = (multiplicity / n) * y_[i] *
                (row < modes ? Complex(1) : std::conj(c_.admittance[i]));
    }
    fftw_execute_dft(analytic_inverse_, fft(time), fft(time));
  });
  std::fill_n(statistics_axis_.data(), statistics_axis_.size(), Complex{});
  for_rows(c_.threads, nw, [&](int w) {
    Complex electric{}, magnetic{};
    for (int m = 0; m < modes; ++m) {
      const auto i = std::size_t(m) * nw + w;
      const Complex value = c_.spatial_axis[m] * y_[i];
      electric += value;
      magnetic += value * std::conj(c_.admittance[i]);
    }
    const double multiplicity = w == 0 || w == nw - 1 ? 1 : 2;
    out[4 + w] = intensity_factor * dt / (n * domega) * multiplicity * multiplicity *
                 std::real(electric * std::conj(magnetic));
    statistics_axis_[w] = (multiplicity / n) * electric;
    statistics_axis_[n + w] = (multiplicity / n) * magnetic;
  });
  for (int row = 0; row < 2; ++row) {
    Complex* time = statistics_axis_.data() + row * n;
    fftw_execute_dft(analytic_inverse_, fft(time), fft(time));
  }
  out[1] = -std::numeric_limits<double>::infinity();
  for (int t = 0; t < n; ++t)
    out[1] = std::max(out[1],
                      intensity_factor * std::real(statistics_axis_[t] *
                                                   std::conj(statistics_axis_[n + t])));
  for (int w = 0; w < nw; ++w)
    out[3] += domega * out[4 + w];
  out[2] = -std::numeric_limits<double>::infinity();
  for (int begin = 0; begin < nodes; begin += c_.chunk) {
    const int count = std::min(c_.chunk, nodes - begin);
    // A real map times interleaved complex samples is one real matrix product
    // with twice as many columns. Remove the fine-grid scale for observations.
    for (int partner = 0; partner < 2; ++partner)
      contract(count, modes, modes, c_.synthesis.data() + begin * modes,
               reinterpret_cast<const double*>(statistics_time_.data() +
                                               std::size_t(partner) * modes * n),
               reinterpret_cast<double*>(statistics_nodes_.data() +
                                         std::size_t(partner) * c_.chunk * n),
               double(n) / c_.nf, false, 2 * n);
    for_rows(c_.threads, count, [&](int row) {
      const Complex* e = statistics_nodes_.data() + std::size_t(row) * n;
      const Complex* g = e + std::size_t(c_.chunk) * n;
      double fluence = 0, magnitude = 0,
             peak = -std::numeric_limits<double>::infinity();
      for (int t = 0; t < n; ++t) {
        const double flux = std::real(e[t] * std::conj(g[t]));
        fluence += flux;
        magnitude += std::abs(e[t]) * std::abs(g[t]);
        peak = std::max(peak, flux);
      }
      out[4 + nw + begin + row] = intensity_factor * dt * fluence;
      out[4 + nw + nodes + begin + row] = intensity_factor * dt * magnitude;
      chunk_populations_[8 * row] = intensity_factor * peak;
    });
    for (int row = 0; row < count; ++row)
      out[2] = std::max(out[2], chunk_populations_[8 * row]);
  }
  timer.mark(8);
  return out;
}

std::size_t State::memory_bytes() const {
  std::size_t tables = 16 * (c_.phi.size() + c_.prefactor.size() +
                             c_.admittance.size() + c_.spatial_axis.size()) +
                       sizeof(std::size_t) * c_.crop.size() + c_.active.size() +
                       c_.norm.size();
  for (const auto* row :
       {&c_.synthesis, &c_.projection, &c_.coarse, &c_.omega, &c_.window,
        &c_.window_derivative, &c_.omega_window, &c_.absorber, &c_.absorber_coarse,
        &c_.taps, &c_.rate, &c_.undamped, &c_.tableau, &c_.solution, &c_.error,
        &last_rates_, &node_populations_, &chunk_populations_, &c_.spectral_weights})
    tables += 8 * row->size();
  return tables + y_.bytes() + trial_.bytes() + input_.bytes() + source_.bytes() +
         stages_.bytes() + fsal_.bytes() + transform_.bytes() +
         projected_spectrum_.bytes() + coarse_spectrum_.bytes() + history_.bytes() +
         history_state_.bytes() + history_rate_.bytes() + moments_.bytes() +
         compensation_.bytes() + endpoint_.bytes() + modal_.bytes() + node_.bytes() +
         modal_quadrature_.bytes() + node_quadrature_.bytes() + polarisation_.bytes() +
         current_.bytes() + edge_.bytes() + rate_.bytes() + fraction_.bytes() +
         scan_.bytes() + temporary_.bytes() + projected_.bytes() +
         coarse_projected_.bytes() + raman_time_.bytes() + raman_kernel_time_.bytes() +
         raman_frequency_.bytes() + raman_kernel_.bytes() + statistics_time_.bytes() +
         statistics_power_.bytes() + reductions_.bytes() + statistics_nodes_.bytes() +
         statistics_axis_.bytes();
}
} // namespace solprop_cpunative
