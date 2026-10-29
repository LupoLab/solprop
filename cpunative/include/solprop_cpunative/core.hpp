// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <fftw3.h>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace solprop_cpunative {
using Complex = std::complex<double>;

/** FFTW-aligned storage; sizes and lifetimes are fixed by the owning workspace.
 */
template <class T> class Buffer {
  struct Free {
    void operator()(T* p) const { fftw_free(p); }
  };
  std::unique_ptr<T[], Free> data_;
  std::size_t size_ = 0;

public:
  Buffer() = default;
  explicit Buffer(std::size_t n, int threads = 1) : size_(n) {
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
      throw std::length_error("workspace allocation overflow");
    data_.reset(static_cast<T*>(fftw_malloc(n * sizeof(T))));
    if (n && !data_)
      throw std::bad_alloc();
    if (threads == 1 || n < 65536)
      std::fill_n(data_.get(), n, T{});
    else {
      // First-touch large resident buffers in contiguous worker partitions.
#pragma omp parallel for num_threads(threads) schedule(static)
      for (std::size_t i = 0; i < n; ++i)
        data_[i] = T{};
    }
  }
  T* data() { return data_.get(); }
  const T* data() const { return data_.get(); }
  T& operator[](std::size_t i) { return data_[i]; }
  const T& operator[](std::size_t i) const { return data_[i]; }
  std::size_t size() const { return size_; }
  std::size_t bytes() const { return size_ * sizeof(T); }
};

struct Config {
  int modes{}, nodes{}, chunk{}, nt{}, nf{}, nw{}, threads{1}, slots{7};
  bool plasma{}, band{}, split{}, profile{}, analytic_kerr{};
  double dt{}, kerr{}, charge{}, potential{}, field_min{}, field_max{}, rate_max{},
      log_min{}, log_step{};
  double raman_fraction{}, epsilon0{}, rtol{}, atol{};
  std::vector<Complex> phi, prefactor, admittance, spatial_axis;
  std::vector<double> spectral_weights;
  std::vector<double> synthesis, projection, coarse, omega, window, window_derivative;
  std::vector<double> omega_window, absorber, absorber_coarse, taps, rate, undamped;
  std::vector<double> tableau, solution, error;
  std::vector<std::size_t> crop;
  std::vector<unsigned char> active, norm;
  std::string planning = "measure", contraction = "auto";
};

/** Small scalar bundle. No field-sized objects cross the stepping boundary. */
struct Stage {
  std::vector<double> delta, prefactor, raman_rates;
  double kerr_density{}, plasma_density{}, preionisation{}, raman_density{}, taper{1};
};

struct AdamsTables {
  int order{}, width{}, count{}, rows{};
  std::vector<std::size_t> slots;
  std::vector<double> predictor, corrector, defects, spans, carried, endpoint,
      comp_defects, rate;
};

class State {
  Config c_;
  std::array<double, 10> timings_{};
  std::array<double, 7> response_timings_{};
  double planning_seconds_ = 0;
  std::size_t size_, fine_size_;
  int fine_bins_, streams_, raman_bins_, raman_pairs_ = 0;
  Buffer<Complex> y_, trial_, input_, source_, stages_, fsal_, transform_,
      projected_spectrum_, coarse_spectrum_;
  Buffer<Complex> history_, history_state_, history_rate_, moments_, compensation_,
      endpoint_;
  Buffer<double> modal_, node_, modal_quadrature_, node_quadrature_, polarisation_,
      current_, edge_, rate_, fraction_, scan_, temporary_;
  Buffer<double> projected_, coarse_projected_, raman_time_, raman_kernel_time_;
  Buffer<Complex> raman_frequency_, raman_kernel_;
  Buffer<Complex> statistics_time_, statistics_nodes_, statistics_axis_;
  Buffer<double> statistics_power_, reductions_;
  std::vector<double> last_rates_, node_populations_, chunk_populations_;
  std::array<double, 6> diagnostics_{};
  fftw_plan inverse_{}, forward_{}, raman_forward_{}, raman_inverse_{},
      analytic_inverse_{}, coarse_inverse_{}, coarse_forward_{};
  bool fsal_ready_ = false;
  std::size_t evaluations_ = 0;
  double quadrature_ = std::numeric_limits<double>::quiet_NaN();
  double pending_quadrature_ = quadrature_;
  Complex combination(const std::vector<Complex>& terms,
                      std::span<const double> weights, std::size_t i) const;
  void validate_stage(const Stage& stage) const;
  void validate_rate(std::span<const double> rate) const;
  void contract(int rows, int inner, int leading, const double* a, const double* b,
                double* out, double scale, bool add, int samples = 0);
  void to_time(const Complex* field);
  void update_raman(const Stage& stage);
  void prepare_raman_pairs();
  void response(int row, const Stage& stage, bool diagnostic, int node = -1);
  void scan(const double* input, double* filtered, double* output) const;
  double ionisation_rate(double field) const;
  void ionisation_rates(const double* field, double* rate) const;
  void source(const Complex* field, Complex* output, const Stage& stage,
              bool diagnostic = false, bool score = false);
  void spectral_source(const Complex* field, Complex* output, const Stage& stage,
                       bool diagnostic);
  void apply_finish(double dz);

public:
  explicit State(Config config);
  ~State();
  State(const State&) = delete;
  State& operator=(const State&) = delete;
  void load(std::span<const Complex> field);
  std::span<const Complex> field() const { return {y_.data(), size_}; }
  std::vector<Complex> evaluate(const Stage& stage);
  std::vector<double> temporal_probe();
  std::vector<Complex> roundtrip_probe();
  std::vector<double> response_probe(std::span<const double> field, const Stage& stage);
  std::array<double, 3> trial(double dz, const std::vector<Stage>& stages,
                              const std::vector<double>& whole);
  void accept(double dz);
  void clear_fsal() { fsal_ready_ = false; }
  void adams_start(const Stage& stage, const std::vector<double>& rate);
  std::vector<double> adams_trial(double dz, const Stage& step, const Stage& sample,
                                  const AdamsTables& tables);
  void adams_accept(double dz, std::size_t slot, const std::vector<double>& rate);
  std::span<const Complex> history(const std::string& name) const;
  void load_history(const std::string& name, std::span<const Complex> values);
  std::vector<double> populations(const Stage& stage);
  std::vector<double> statistics(std::span<const double> weight,
                                 std::span<const double> axis, double intensity_factor);
  std::vector<double> spatial_statistics(double intensity_factor);
  std::array<double, 6> diagnostics() const { return diagnostics_; }
  std::size_t evaluations() const { return evaluations_; }
  double quadrature() const { return quadrature_; }
  void reset_quadrature() { quadrature_ = std::numeric_limits<double>::quiet_NaN(); }
  std::size_t memory_bytes() const;
  int threads() const { return c_.threads; }
  std::string contraction() const { return c_.contraction; }
  double planning_seconds() const { return planning_seconds_; }
  std::array<double, 10> timings() const { return timings_; }
  std::array<double, 7> response_timings() const { return response_timings_; }
};
} // namespace solprop_cpunative
