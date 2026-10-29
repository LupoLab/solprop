// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "solprop_cpunative/core.hpp"
#include <mutex>

namespace solprop_cpunative {
std::mutex& fftw_planner_mutex();

struct BeamConfig {
  int nx{}, ny{}, nt{}, threads{1};
  double dt{}, kerr{}, light_speed{}, evanescent_rate{};
  bool split{}, profile{};
  std::string planning = "measure";
  std::string fft_schedule = "auto";
  std::vector<Complex> k_squared, k_squared_low, numerator, clamped, axis;
  std::vector<double> kperp, kperp_low, frame, spectral_rate, transverse_rate;
  std::vector<double> omega, window, omega_window, absorber, spectral_weights;
  std::vector<unsigned char> active, norm;
};

/** Low-storage Cartesian envelope experiment; five resident complex volumes. */
class BeamState {
  BeamConfig c_;
  std::size_t columns_, size_;
  Buffer<Complex> y_, b_, stage_, source_, flow_, axis_;
  Buffer<double> absorber_factor_;
  fftw_plan inverse_{}, forward_{}, temporal_inverse_{}, temporal_forward_{};
  fftw_plan y_inverse_{}, y_forward_{}, tail_inverse_{}, tail_forward_{};
  static constexpr int fft_block_ = 32;
  double flow_key_ = -1, planning_seconds_ = 0;
  double absorber_key_ = -1;
  bool real_dispersion_ = false;
  std::size_t evaluations_ = 0;
  std::array<double, 6> timings_{};
  Complex root(std::size_t column, int w) const;
  Complex prefactor(std::size_t column, int w) const;
  Complex admittance(std::size_t column, int w) const;
  void source(const Complex* input, double density);
  void plan_flow(double key);
  void inverse_volume(const Complex* input, Complex* output);
  void forward_volume(Complex* field);

public:
  explicit BeamState(BeamConfig config);
  ~BeamState();
  BeamState(const BeamState&) = delete;
  BeamState& operator=(const BeamState&) = delete;
  void load(std::span<const Complex> field);
  std::span<const Complex> field() const { return {y_.data(), size_}; }
  std::vector<Complex> evaluate(double density);
  double step(double dz, double key, std::span<const double> densities);
  void finish(double dz);
  std::vector<double> statistics(double factor);
  std::size_t memory_bytes() const;
  std::size_t evaluations() const { return evaluations_; }
  int threads() const { return c_.threads; }
  std::string fft_schedule() const { return c_.fft_schedule; }
  double planning_seconds() const { return planning_seconds_; }
  std::array<double, 6> timings() const { return timings_; }
};
} // namespace solprop_cpunative
