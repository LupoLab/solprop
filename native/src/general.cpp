// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "general_impl.hpp"

namespace solprop_native {

// The Double policy is instantiated here; each further policy this build selects
// (`SOLPROP_NATIVE_PRECISIONS`, precision.hpp) in its own translation unit
// (general_single.cpp), which instantiates the members a state of that policy runs.
template class GeneralWorkspace<Double>;
template class GeneralBatchWorkspace<Double>;
#if SOLPROP_NATIVE_SINGLE_BUILT
extern template class GeneralWorkspace<Single>;
extern template class GeneralBatchWorkspace<Single>;
#endif

// A state runs on the workspace of its problem's policy; a call names it once.
#if SOLPROP_NATIVE_SINGLE_BUILT
#define SOLPROP_NATIVE_EACH(call) (single_ ? single_->call : workspace_->call)
#else
#define SOLPROP_NATIVE_EACH(call) (workspace_->call)
#endif

GeneralState::GeneralState(Problem problem) {
  if (problem.precision == PrecisionName::single) {
#if SOLPROP_NATIVE_SINGLE_BUILT
    single_ = std::make_unique<GeneralWorkspace<Single>>(std::move(problem));
#else
    throw std::invalid_argument(
        "this solprop-native build has no single precision: build it with "
        "SOLPROP_NATIVE_PRECISIONS including \"single\" (solprop SEP 36)");
#endif
  } else {
    workspace_ = std::make_unique<GeneralWorkspace<Double>>(std::move(problem));
  }
}
bool GeneralState::engaged() const noexcept {
  return static_cast<bool>(workspace_) || static_cast<bool>(single_);
}
GeneralWorkspace<Double>* GeneralState::binary64(const char* what) const {
  if (!workspace_)
    throw std::invalid_argument(std::string("the native backend runs ") + what +
                                " in binary64 only, not under single precision");
  return workspace_.get();
}
GeneralState::~GeneralState() { close(); }
void GeneralState::set_initial(const std::vector<Complex>& field,
                               const std::vector<Complex>& noise) {
  SOLPROP_NATIVE_EACH(set_initial(field, noise));
}
std::vector<double> GeneralState::trial_step(const double dz,
                                             const TrialScalars& scalars) {
  return SOLPROP_NATIVE_EACH(trial_step(dz, scalars));
}
void GeneralState::prepare_trial(const double dz, const TrialScalars& scalars) {
  SOLPROP_NATIVE_EACH(prepare_trial(dz, scalars));
}
std::vector<Complex> GeneralState::adams_source(const std::vector<Complex>& input,
                                                const TrialScalars& scalars) {
  return binary64("the Adams march")->adams_source(input, scalars);
}
void GeneralState::adopt(const std::vector<Complex>& field, const double dz) {
  binary64("the Adams march")->adopt(field, dz);
}
double GeneralState::low_storage_step(const double dz, const TrialScalars& scalars,
                                      const bool rebuild_table) {
  return SOLPROP_NATIVE_EACH(low_storage_step(dz, scalars, rebuild_table));
}
void GeneralState::hold_plane() { SOLPROP_NATIVE_EACH(hold_plane()); }
std::vector<Complex> GeneralState::held_to_host() const {
  return SOLPROP_NATIVE_EACH(held_to_host());
}
void GeneralState::adams_allocate(const std::size_t slots, const bool compensated,
                                  const bool interpolate) {
  binary64("the Adams march")->adams_allocate(slots, compensated, interpolate);
}
void GeneralState::adams_compensation_start(const std::size_t slot,
                                            const std::vector<double>& rate) {
  binary64("the Adams march")->adams_compensation_start(slot, rate);
}
void GeneralState::adams_start(const std::size_t slot, const TrialScalars& scalars) {
  binary64("the Adams march")->adams_start(slot, scalars);
}
double GeneralState::adams_magnitude() {
  return binary64("the Adams march")->adams_magnitude();
}
void GeneralState::adams_scale(const double rtol, const double atol,
                               const std::vector<double>& weights) {
  binary64("the Adams march")->adams_scale(rtol, atol, weights);
}
std::vector<double> GeneralState::adams_fused_trial(const double dz,
                                                    const AdamsWeights& weights,
                                                    const std::vector<double>& divisor,
                                                    const TrialScalars& scalars) {
  return binary64("the Adams march")->adams_fused_trial(dz, weights, divisor, scalars);
}
void GeneralState::adams_fused_accept(const double dz, const AdamsWeights& keep,
                                      const std::size_t free,
                                      const AdamsWeights& dense) {
  binary64("the Adams march")->adams_fused_accept(dz, keep, free, dense);
}
void GeneralState::adams_hold() { binary64("the Adams march")->adams_hold(); }
void GeneralState::adams_restore() { binary64("the Adams march")->adams_restore(); }
void GeneralState::table_upload(const TableKind kind, std::vector<double> edges,
                                const double lower, const double upper,
                                const std::complex<double>* coefficients,
                                const std::size_t n_intervals,
                                const std::size_t n_terms, const std::size_t n_values) {
  SOLPROP_NATIVE_EACH(table_upload(kind, std::move(edges), lower, upper, coefficients,
                                   n_intervals, n_terms, n_values));
}
void GeneralState::table_evaluate(const std::vector<std::size_t>& slots,
                                  const std::vector<double>& positions) {
  SOLPROP_NATIVE_EACH(table_evaluate(slots, positions));
}
void GeneralState::table_rate(const std::size_t slot, const double z,
                              const bool before) {
  SOLPROP_NATIVE_EACH(table_rate(slot, z, before));
}
std::vector<Complex> GeneralState::table_terms_to_host(const TableKind kind,
                                                       const std::size_t slot) const {
  return SOLPROP_NATIVE_EACH(table_terms_to_host(kind, slot));
}
std::vector<Complex>
GeneralState::phase_sum_to_host(const std::vector<double>& weights) {
  return SOLPROP_NATIVE_EACH(phase_sum_to_host(weights));
}
std::vector<std::string> GeneralState::adams_row_names() const {
  return binary64("the Adams march")->adams_row_names();
}
std::array<std::size_t, 3>
GeneralState::adams_row_extents(const std::string& name) const {
  return binary64("the Adams march")->adams_row_extents(name);
}
std::vector<Complex> GeneralState::adams_rows_to_host(const std::string& name) const {
  return binary64("the Adams march")->adams_rows_to_host(name);
}
void GeneralState::adams_rows_to_host(const std::string& name,
                                      std::complex<double>* destination,
                                      const std::size_t size) const {
  binary64("the Adams march")->adams_rows_to_host(name, destination, size);
}
void GeneralState::adams_rows_from_host(const std::string& name,
                                        const std::vector<Complex>& values) {
  binary64("the Adams march")->adams_rows_from_host(name, values);
}
void GeneralState::adams_observe(const AdamsWeights& weights, const bool hermite,
                                 const TrialScalars& scalars) {
  binary64("the Adams march")->adams_observe(weights, hermite, scalars);
}
std::vector<double> GeneralState::adams_exponential_trial(
    const double dz, const ExponentialWeights& weights,
    const std::vector<double>& divisor, const TrialScalars& step,
    const TrialScalars& sample, const std::vector<double>& rate) {
  return binary64("the Adams march")
      ->adams_exponential_trial(dz, weights, divisor, step, sample, rate);
}
void GeneralState::adams_exponential_accept(const double dz, const std::size_t free,
                                            const AdamsWeights& dense) {
  binary64("the Adams march")->adams_exponential_accept(dz, free, dense);
}
void GeneralState::adams_exponential_observe(const double xi, const double dz,
                                             const ExponentialWeights& weights) {
  binary64("the Adams march")->adams_exponential_observe(xi, dz, weights);
}
std::vector<double> GeneralState::weighted_error(const std::vector<double>& scale) {
  return SOLPROP_NATIVE_EACH(weighted_error(scale));
}
void GeneralState::set_flow_actions(const std::vector<Complex>& forward,
                                    const std::vector<Complex>& inverse,
                                    const std::vector<unsigned char>& identity) {
  binary64("a coupled flow")->set_flow_actions(forward, inverse, identity);
}
std::vector<Complex> GeneralState::flow_actions_to_host(const bool inverse) const {
  return binary64("a coupled flow")->flow_actions_to_host(inverse);
}
std::vector<double>
GeneralState::form_flow_actions(const std::size_t section, const double dz,
                                const std::vector<double>& fractions) {
  return binary64("a coupled flow")->form_flow_actions(section, dz, fractions);
}
void GeneralState::prepare_raman(const std::vector<double>& rates,
                                 const std::vector<double>& densities) {
  SOLPROP_NATIVE_EACH(prepare_raman(rates, densities));
}
std::vector<double>
GeneralState::statistics(const std::uint64_t flags, const std::vector<double>& windows,
                         const std::vector<double>& on_axis, const double density,
                         const double taper, const double preionisation,
                         const double intensity_factor,
                         const std::vector<double>& channel_preionisation,
                         const std::vector<double>& power_weight) {
  return SOLPROP_NATIVE_EACH(statistics(flags, windows, on_axis, density, taper,
                                        preionisation, intensity_factor,
                                        channel_preionisation, power_weight));
}
std::vector<double> GeneralState::spatial_statistics(const std::uint64_t flags,
                                                     const double intensity_factor,
                                                     const double z) {
  return binary64("the spatial statistics")
      ->spatial_statistics(flags, intensity_factor, z);
}
std::vector<double> GeneralState::aperture_statistics(
    const std::uint64_t flags, const std::vector<double>& apertures,
    const int profile_code, const int exponent, const std::vector<double>& regions,
    const double intensity_factor) {
  return binary64("the far-field statistics")
      ->aperture_statistics(flags, apertures, profile_code, exponent, regions,
                            intensity_factor);
}
void GeneralState::finish(const double dz) { SOLPROP_NATIVE_EACH(finish(dz)); }
void GeneralState::promote_fsal() { SOLPROP_NATIVE_EACH(promote_fsal()); }
void GeneralState::clear_fsal() { SOLPROP_NATIVE_EACH(clear_fsal()); }
double GeneralState::quadrature_error() const {
  return engaged() ? SOLPROP_NATIVE_EACH(quadrature_error())
                   : Kokkos::Experimental::quiet_NaN_v<double>;
}
double GeneralState::quadrature_peak() {
  return engaged() ? SOLPROP_NATIVE_EACH(quadrature_peak())
                   : Kokkos::Experimental::quiet_NaN_v<double>;
}
std::string GeneralState::synthesis_provider() const {
  return engaged() ? SOLPROP_NATIVE_EACH(synthesis_provider()) : "closed";
}
std::string GeneralState::projection_provider() const {
  return engaged() ? SOLPROP_NATIVE_EACH(projection_provider()) : "closed";
}
std::string GeneralState::gemm_shape_policy() const {
  return GeneralWorkspace<Double>::gemm_shape_policy();
}
std::size_t GeneralState::response_team_size() const noexcept {
  return engaged() ? SOLPROP_NATIVE_EACH(response_team_size()) : 0;
}
std::vector<Complex> GeneralState::field_to_host() const {
  return SOLPROP_NATIVE_EACH(field_to_host());
}
std::vector<Complex> GeneralState::stage_to_host(const std::size_t stage) const {
  return SOLPROP_NATIVE_EACH(stage_to_host(stage));
}
std::vector<Complex> GeneralState::projected_to_host(const bool coarse) {
  return SOLPROP_NATIVE_EACH(projected_to_host(coarse));
}
std::map<std::string, bool> GeneralState::dispersive_path() const {
  return SOLPROP_NATIVE_EACH(dispersive_path());
}
std::map<std::string, std::size_t> GeneralState::memory_ledger() const {
  return SOLPROP_NATIVE_EACH(memory_ledger());
}
CpuSizing GeneralState::execution_sizing() const {
  return SOLPROP_NATIVE_EACH(execution_sizing());
}
double GeneralState::planning_seconds() const noexcept {
  return engaged() ? SOLPROP_NATIVE_EACH(planning_seconds()) : 0.0;
}
double GeneralState::raman_planning_seconds() const noexcept {
  return engaged() ? SOLPROP_NATIVE_EACH(raman_planning_seconds()) : 0.0;
}
std::string GeneralState::raman_fft_provider() const {
  return engaged() ? SOLPROP_NATIVE_EACH(raman_fft_provider()) : "none";
}
std::map<std::string, int> GeneralState::raman_fft_tasks() const {
  return engaged() ? SOLPROP_NATIVE_EACH(raman_fft_tasks())
                   : std::map<std::string, int>{};
}
std::size_t GeneralState::raman_kernel_transforms() const noexcept {
  return engaged() ? SOLPROP_NATIVE_EACH(raman_kernel_transforms()) : 0;
}
bool GeneralState::fsal_ready() const noexcept {
  return engaged() && SOLPROP_NATIVE_EACH(fsal_ready());
}
std::size_t GeneralState::rk_source_evaluations() const noexcept {
  return engaged() ? SOLPROP_NATIVE_EACH(rk_source_evaluations()) : 0;
}
bool GeneralState::closed() const noexcept {
  return !engaged() || SOLPROP_NATIVE_EACH(closed());
}
void GeneralState::close() noexcept {
  if (workspace_)
    workspace_->close();
#if SOLPROP_NATIVE_SINGLE_BUILT
  if (single_)
    single_->close();
#endif
}

GeneralBatch::GeneralBatch(std::vector<std::shared_ptr<GeneralState>> members) {
  // Members of one precision only (solprop SEP 36 section 6.6); the batch workspace
  // refuses a member of another.
  const bool single =
      !members.empty() && members.front() && members.front()->single_ != nullptr;
  if (single) {
#if SOLPROP_NATIVE_SINGLE_BUILT
    single_ = std::make_unique<GeneralBatchWorkspace<Single>>(std::move(members));
#endif
  } else {
    workspace_ = std::make_unique<GeneralBatchWorkspace<Double>>(std::move(members));
  }
}
GeneralBatch::~GeneralBatch() { close(); }
std::vector<double> GeneralBatch::accept(const double dz) {
  return SOLPROP_NATIVE_EACH(accept(dz));
}
std::map<std::string, std::size_t> GeneralBatch::memory_ledger() const {
  return SOLPROP_NATIVE_EACH(memory_ledger());
}
CpuSizing GeneralBatch::execution_sizing() const {
  return SOLPROP_NATIVE_EACH(execution_sizing());
}
void GeneralBatch::close() noexcept { SOLPROP_NATIVE_EACH(close()); }

#undef SOLPROP_NATIVE_EACH

} // namespace solprop_native
