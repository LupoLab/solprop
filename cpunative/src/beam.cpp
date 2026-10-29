// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "solprop_cpunative/beam.hpp"
#include <chrono>
#include <cmath>
#include <numbers>
#ifdef CPUNATIVE_MKL_FFT
#include <fftw3_mkl.h>
#endif

namespace solprop_cpunative {
namespace {
fftw_complex* fft(Complex* p) { return reinterpret_cast<fftw_complex*>(p); }
template <class Body> void each(int threads, std::size_t count, Body body) {
  if (threads == 1) {
    for (std::size_t i = 0; i < count; ++i)
      body(i);
  } else {
#pragma omp parallel for num_threads(threads) schedule(static)
    for (std::size_t i = 0; i < count; ++i)
      body(i);
  }
}
BeamConfig validate(BeamConfig c) {
  if (c.nx < 2 || c.ny < 1 || c.nt < 2 || c.threads < 1 || !(c.dt > 0))
    throw std::invalid_argument("invalid Cartesian workspace dimensions");
  // Plane transforms are independent; retain internally threaded volume plans
  // when there are too few planes to occupy the requested team.
  if (c.fft_schedule == "auto")
    c.fft_schedule = c.ny >= c.threads ? "split" : "direct";
  if (c.fft_schedule != "direct" && c.fft_schedule != "split")
    throw std::invalid_argument("Cartesian FFT schedule must be direct or split");
  if (c.fft_schedule == "split" &&
      std::size_t(c.nx) * c.nt > std::numeric_limits<int>::max())
    throw std::invalid_argument("Cartesian split FFT plane exceeds supported strides");
  const auto columns = std::size_t(c.nx) * c.ny, nt = std::size_t(c.nt);
  if (c.k_squared.size() != nt || c.k_squared_low.size() != nt ||
      c.numerator.size() != nt || c.clamped.size() != nt || c.frame.size() != nt ||
      c.spectral_rate.size() != nt || c.omega.size() != nt || c.window.size() != nt ||
      c.omega_window.size() != nt || c.active.size() != nt || c.norm.size() != nt ||
      c.kperp.size() != columns || c.kperp_low.size() != columns ||
      c.transverse_rate.size() != columns || c.axis.size() != columns ||
      c.spectral_weights.size() != columns ||
      (!c.absorber.empty() && c.absorber.size() != nt))
    throw std::invalid_argument("inconsistent Cartesian workspace tables");
  return c;
}
} // namespace

BeamState::BeamState(BeamConfig config)
    : c_(validate(std::move(config))), columns_(std::size_t(c_.nx) * c_.ny),
      size_(columns_ * c_.nt), y_(size_, c_.threads), b_(size_, c_.threads),
      stage_(size_, c_.threads), source_(size_, c_.threads), flow_(size_, c_.threads),
      axis_(2 * c_.nt), absorber_factor_(c_.nt) {
  real_dispersion_ = std::all_of(c_.k_squared.begin(), c_.k_squared.end(),
                                 [](Complex value) { return value.imag() == 0; }) &&
                     std::all_of(c_.k_squared_low.begin(), c_.k_squared_low.end(),
                                 [](Complex value) { return value.imag() == 0; });
  const auto start = std::chrono::steady_clock::now();
  std::lock_guard lock(fftw_planner_mutex());
  if (!fftw_init_threads())
    throw std::runtime_error("FFTW thread initialization failed");
  const bool split_fft = c_.fft_schedule == "split";
  fftw_plan_with_nthreads(split_fft ? 1 : c_.threads);
#ifdef CPUNATIVE_MKL_FFT
  fftw3_mkl.number_of_user_threads = split_fft ? c_.threads : 1;
#endif
  const unsigned flags =
      c_.planning == "measure" && c_.threads == 1 ? FFTW_MEASURE : FFTW_ESTIMATE;
  if (split_fft) {
    const int plane = c_.nx * c_.nt;
    unsigned serial_flags = c_.planning == "measure" ? FFTW_MEASURE : FFTW_ESTIMATE;
    if (fftw_alignment_of(reinterpret_cast<double*>(y_.data())) !=
        fftw_alignment_of(reinterpret_cast<double*>(y_.data() + plane)))
      serial_flags |= FFTW_UNALIGNED;
    inverse_ = fftw_plan_dft_2d(c_.nx, c_.nt, fft(y_.data()), fft(source_.data()),
                                FFTW_BACKWARD, serial_flags);
    forward_ = fftw_plan_dft_2d(c_.nx, c_.nt, fft(source_.data()), fft(source_.data()),
                                FFTW_FORWARD, serial_flags);
    auto across_y = [&](int count, int sign) {
      return fftw_plan_many_dft(1, &c_.ny, count, fft(source_.data()), nullptr,
                                 plane, 1, fft(source_.data()), nullptr, plane, 1,
                                 sign, serial_flags);
    };
    y_inverse_ = across_y(std::min(plane, fft_block_), FFTW_BACKWARD);
    y_forward_ = across_y(std::min(plane, fft_block_), FFTW_FORWARD);
    if (plane > fft_block_ && plane % fft_block_) {
      tail_inverse_ = across_y(plane % fft_block_, FFTW_BACKWARD);
      tail_forward_ = across_y(plane % fft_block_, FFTW_FORWARD);
    }
  } else {
    inverse_ = fftw_plan_dft_3d(c_.ny, c_.nx, c_.nt, fft(y_.data()), fft(source_.data()),
                                FFTW_BACKWARD, flags);
    forward_ = fftw_plan_dft_3d(c_.ny, c_.nx, c_.nt, fft(source_.data()),
                                fft(source_.data()), FFTW_FORWARD, flags);
  }
  fftw_plan_with_nthreads(1);
#ifdef CPUNATIVE_MKL_FFT
  fftw3_mkl.number_of_user_threads = c_.threads;
#endif
  // Rows can start at different SIMD alignments when nt is not a multiple of
  // the allocator alignment. New-array execution must match the plan's class.
  const unsigned temporal_flags =
      flags | (fftw_alignment_of(reinterpret_cast<double*>(b_.data())) !=
                       fftw_alignment_of(reinterpret_cast<double*>(b_.data() + c_.nt))
                   ? FFTW_UNALIGNED
                   : 0);
  temporal_inverse_ = fftw_plan_dft_1d(c_.nt, fft(b_.data()), fft(b_.data()),
                                       FFTW_BACKWARD, temporal_flags);
  temporal_forward_ = fftw_plan_dft_1d(c_.nt, fft(b_.data()), fft(b_.data()),
                                       FFTW_FORWARD, temporal_flags);
  const std::size_t plane = std::size_t(c_.nx) * c_.nt;
  const int tail = int(plane % fft_block_);
  if (!inverse_ || !forward_ || !temporal_inverse_ || !temporal_forward_ ||
      (split_fft && (!y_inverse_ || !y_forward_ ||
                     (plane > fft_block_ && tail &&
                      (!tail_inverse_ || !tail_forward_))))) {
    for (auto plan : {inverse_, forward_, temporal_inverse_, temporal_forward_,
                      y_inverse_, y_forward_, tail_inverse_, tail_forward_})
      if (plan)
        fftw_destroy_plan(plan);
    throw std::runtime_error("Cartesian FFTW plan construction failed");
  }
  planning_seconds_ =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}
BeamState::~BeamState() {
  std::lock_guard lock(fftw_planner_mutex());
  for (auto plan : {inverse_, forward_, temporal_inverse_, temporal_forward_,
                    y_inverse_, y_forward_, tail_inverse_, tail_forward_})
    if (plan)
      fftw_destroy_plan(plan);
}
void BeamState::inverse_volume(const Complex* input, Complex* output) {
  if (!y_inverse_) {
    fftw_execute_dft(inverse_, fft(const_cast<Complex*>(input)), fft(output));
    return;
  }
  const std::size_t plane = std::size_t(c_.nx) * c_.nt;
  // Each row/column batch is disjoint. Serial FFTW plans avoid the deeply
  // nested worker trees observed in large internally threaded volume plans.
  each(c_.threads, c_.ny, [&](std::size_t y) {
    fftw_execute_dft(inverse_, fft(const_cast<Complex*>(input) + y * plane),
                      fft(output + y * plane));
  });
  each(c_.threads, (plane + fft_block_ - 1) / fft_block_, [&](std::size_t block) {
    const auto first = block * fft_block_;
    const auto plan = plane - first < fft_block_ && tail_inverse_ ? tail_inverse_
                                                                : y_inverse_;
    fftw_execute_dft(plan, fft(output + first), fft(output + first));
  });
}
void BeamState::forward_volume(Complex* field) {
  if (!y_forward_) {
    fftw_execute_dft(forward_, fft(field), fft(field));
    return;
  }
  const std::size_t plane = std::size_t(c_.nx) * c_.nt;
  each(c_.threads, (plane + fft_block_ - 1) / fft_block_, [&](std::size_t block) {
    const auto first = block * fft_block_;
    const auto plan = plane - first < fft_block_ && tail_forward_ ? tail_forward_
                                                                : y_forward_;
    fftw_execute_dft(plan, fft(field + first), fft(field + first));
  });
  each(c_.threads, c_.ny, [&](std::size_t y) {
    fftw_execute_dft(forward_, fft(field + y * plane), fft(field + y * plane));
  });
}
Complex BeamState::root(std::size_t column, int w) const {
  // Preserve the translated high/low subtraction near the propagating cutoff.
  return std::sqrt((c_.k_squared[w] - c_.kperp[column]) +
                   (c_.k_squared_low[w] - c_.kperp_low[column]));
}
Complex BeamState::prefactor(std::size_t column, int w) const {
  const auto square = (c_.k_squared[w] - c_.kperp[column]) +
                      (c_.k_squared_low[w] - c_.kperp_low[column]);
  return square.real() > 0 ? c_.numerator[w] / std::sqrt(square).real() : c_.clamped[w];
}
Complex BeamState::admittance(std::size_t column, int w) const {
  const auto square = (c_.k_squared[w] - c_.kperp[column]) +
                      (c_.k_squared_low[w] - c_.kperp_low[column]);
  return c_.active[w] && c_.omega[w] > 0 && square.real() > 0
             ? c_.light_speed * std::sqrt(square) / c_.omega[w]
             : Complex{};
}
void BeamState::load(std::span<const Complex> field) {
  if (field.size() != size_)
    throw std::invalid_argument("Cartesian field shape mismatch");
  for (auto value : field)
    if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
      throw std::invalid_argument("nonfinite Cartesian input");
  each(c_.threads, columns_, [&](std::size_t m) {
    for (int w = 0; w < c_.nt; ++w)
      y_[m * c_.nt + w] = c_.active[w] ? field[m * c_.nt + w] : Complex{};
  });
  flow_key_ = -1;
  evaluations_ = 0;
}
void BeamState::source(const Complex* input, double density) {
  auto start = c_.profile ? std::chrono::steady_clock::now()
                          : std::chrono::steady_clock::time_point{};
  auto mark = [&](int phase) {
    if (c_.profile) {
      const auto now = std::chrono::steady_clock::now();
      timings_[phase] += std::chrono::duration<double>(now - start).count();
      start = now;
    }
  };
  inverse_volume(input, source_.data());
  mark(0);
  const double inverse_size = 1.0 / size_, coefficient = c_.kerr * density;
  each(c_.threads, columns_, [&](std::size_t m) {
    auto* row = source_.data() + m * c_.nt;
#pragma omp simd
    for (int t = 0; t < c_.nt; ++t) {
      const auto field = row[t] * inverse_size;
      row[t] = (coefficient * std::norm(field) * c_.window[t]) * field;
    }
  });
  mark(1);
  forward_volume(source_.data());
  mark(2);
  if (real_dispersion_) {
    // A real positive squared wave number has a real square root. Avoid the
    // general complex square-root helper while retaining the translated clamp.
    each(c_.threads, columns_, [&](std::size_t m) {
      auto* __restrict row = source_.data() + m * c_.nt;
#pragma omp simd
      for (int w = 0; w < c_.nt; ++w) {
        const double square = (c_.k_squared[w].real() - c_.kperp[m]) +
                              (c_.k_squared_low[w].real() - c_.kperp_low[m]);
        const Complex factor =
            square > 0 ? c_.numerator[w] / std::sqrt(square) : c_.clamped[w];
        row[w] = c_.active[w] ? c_.omega_window[w] * factor * row[w] : Complex{};
      }
    });
  } else {
    each(c_.threads, columns_, [&](std::size_t m) {
      auto* row = source_.data() + m * c_.nt;
      for (int w = 0; w < c_.nt; ++w)
        row[w] =
            c_.active[w] ? c_.omega_window[w] * prefactor(m, w) * row[w] : Complex{};
    });
  }
  mark(3);
  ++evaluations_;
}
std::vector<Complex> BeamState::evaluate(double density) {
  source(y_.data(), density);
  return {source_.data(), source_.data() + size_};
}
void BeamState::plan_flow(double key) {
  if (key == flow_key_)
    return;
  each(c_.threads, columns_, [&](std::size_t m) {
    for (int w = 0; w < c_.nt; ++w) {
      const auto square =
          (c_.k_squared[w] - c_.kperp[m]) + (c_.k_squared_low[w] - c_.kperp_low[m]);
      const auto kz = std::sqrt(square);
      const double attenuation =
          square.real() > 0 ? kz.imag()
                            : std::min(std::abs(kz.imag()), c_.evanescent_rate);
      auto rate = c_.active[w] ? Complex(-attenuation - .5 * c_.transverse_rate[m],
                                         -(kz.real() - c_.frame[w]))
                               : Complex{};
      rate -= .5 * c_.spectral_rate[w];
      flow_[m * c_.nt + w] = std::exp((.5 * key) * rate);
    }
  });
  flow_key_ = key;
}
double BeamState::step(double dz, double key, std::span<const double> densities) {
  if (!(dz > 0) || !(key > 0) || densities.size() != 3)
    throw std::invalid_argument("invalid RK4IP step");
  plan_flow(key);
  source(y_.data(), densities[0]);
  each(c_.threads, size_, [&](std::size_t i) {
    y_[i] *= flow_[i];
    const auto g = flow_[i] * source_[i];
    b_[i] = y_[i] + (dz / 6) * g;
    stage_[i] = y_[i] + (.5 * dz) * g;
  });
  source(stage_.data(), densities[1]);
  each(c_.threads, size_, [&](std::size_t i) {
    b_[i] += (dz / 3) * source_[i];
    stage_[i] = y_[i] + (.5 * dz) * source_[i];
  });
  source(stage_.data(), densities[1]);
  each(c_.threads, size_, [&](std::size_t i) {
    b_[i] += (dz / 3) * source_[i];
    stage_[i] = flow_[i] * (y_[i] + dz * source_[i]);
  });
  source(stage_.data(), densities[2]);
  double norm = 0;
  if (c_.threads == 1) {
    for (std::size_t i = 0; i < size_; ++i) {
      y_[i] = flow_[i] * b_[i] + (dz / 6) * source_[i];
      if (c_.norm[i % c_.nt])
        norm += std::norm(y_[i]);
    }
  } else {
#pragma omp parallel for simd num_threads(c_.threads) schedule(static)                 \
    reduction(+ : norm)
    for (std::size_t i = 0; i < size_; ++i) {
      y_[i] = flow_[i] * b_[i] + (dz / 6) * source_[i];
      if (c_.norm[i % c_.nt])
        norm += std::norm(y_[i]);
    }
  }
  return norm;
}
void BeamState::finish(double dz) {
  const auto start = std::chrono::steady_clock::now();
  if (c_.split && !c_.absorber.empty() && dz != absorber_key_) {
    for (int t = 0; t < c_.nt; ++t)
      absorber_factor_[t] = std::exp(-.5 * dz * c_.absorber[t]) / c_.nt;
    absorber_key_ = dz;
  }
  each(c_.threads, columns_, [&](std::size_t m) {
    auto* row = y_.data() + m * c_.nt;
    for (int w = 0; w < c_.nt; ++w)
      if (!c_.active[w])
        row[w] = 0;
    if (c_.split && !c_.absorber.empty()) {
      fftw_execute_dft(temporal_inverse_, fft(row), fft(row));
      for (int t = 0; t < c_.nt; ++t)
        row[t] *= absorber_factor_[t];
      fftw_execute_dft(temporal_forward_, fft(row), fft(row));
      for (int w = 0; w < c_.nt; ++w)
        if (!c_.active[w])
          row[w] = 0;
    }
  });
  if (c_.profile)
    timings_[4] +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}
std::vector<double> BeamState::statistics(double factor) {
  const auto start = std::chrono::steady_clock::now();
  const auto nt = c_.nt;
  std::vector<double> result(4 + nt + 2 * columns_, 0.);
  each(c_.threads, columns_, [&](std::size_t m) {
    for (int w = 0; w < nt; ++w)
      source_[m * nt + w] = std::conj(admittance(m, w)) * y_[m * nt + w];
  });
  for (std::size_t m = 0; m < columns_; ++m)
    for (int w = 0; w < nt; ++w)
      result[0] += c_.spectral_weights[m] *
                   std::real(y_[m * nt + w] * std::conj(source_[m * nt + w]));
  result[0] *= factor * c_.dt / nt;
  each(c_.threads, nt, [&](std::size_t w) {
    Complex e{}, g{};
    for (std::size_t m = 0; m < columns_; ++m) {
      e += c_.axis[m] * y_[m * nt + w];
      g += c_.axis[m] * source_[m * nt + w];
    }
    axis_[w] = e;
    axis_[nt + w] = g;
    result[4 + w] =
        factor * c_.dt * c_.dt / (2 * std::numbers::pi) * std::real(e * std::conj(g));
  });
  fftw_execute_dft(temporal_inverse_, fft(axis_.data()), fft(axis_.data()));
  fftw_execute_dft(temporal_inverse_, fft(axis_.data() + nt), fft(axis_.data() + nt));
  result[1] = -std::numeric_limits<double>::infinity();
  for (int t = 0; t < nt; ++t)
    result[1] = std::max(result[1], factor / (double(nt) * nt) *
                                        std::real(axis_[t] * std::conj(axis_[nt + t])));
  inverse_volume(y_.data(), b_.data());
  inverse_volume(source_.data(), stage_.data());
  const double scale = factor / (double(size_) * size_);
  // The stage registers are idle between steps and own the observation scratch.
  each(c_.threads, columns_, [&](std::size_t m) {
    double fluence = 0, magnitude = 0, peak = -std::numeric_limits<double>::infinity();
    for (int t = 0; t < nt; ++t) {
      const auto e = b_[m * nt + t], g = stage_[m * nt + t];
      const double intensity = scale * std::real(e * std::conj(g));
      fluence += intensity;
      magnitude += scale * std::abs(e) * std::abs(g);
      peak = std::max(peak, intensity);
    }
    result[4 + nt + m] = c_.dt * fluence;
    result[4 + nt + columns_ + m] = c_.dt * magnitude;
    b_[m * nt] = peak;
  });
  // Do not reuse a stage row while another worker still reads it.
  result[2] = -std::numeric_limits<double>::infinity();
  for (std::size_t m = 0; m < columns_; ++m)
    result[2] = std::max(result[2], b_[m * nt].real());
  for (int w = 0; w < nt; ++w)
    result[3] += 2 * std::numbers::pi / (nt * c_.dt) * result[4 + w];
  if (c_.profile)
    timings_[5] +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  return result;
}
std::size_t BeamState::memory_bytes() const {
  std::size_t bytes = 5 * y_.bytes() + axis_.bytes() + absorber_factor_.bytes();
  for (const auto* v :
       {&c_.k_squared, &c_.k_squared_low, &c_.numerator, &c_.clamped, &c_.axis})
    bytes += v->size() * sizeof(Complex);
  for (const auto* v :
       {&c_.kperp, &c_.kperp_low, &c_.frame, &c_.spectral_rate, &c_.transverse_rate,
        &c_.omega, &c_.window, &c_.omega_window, &c_.absorber, &c_.spectral_weights})
    bytes += v->size() * sizeof(double);
  return bytes + c_.active.size() + c_.norm.size();
}
} // namespace solprop_cpunative
