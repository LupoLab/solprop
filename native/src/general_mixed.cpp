// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// The Mixed policy's general engine (solprop SEP 36 sections 4.2 and 6.3, phase 3): a
// binary64 state, stored rows and propagator with a binary32 source, in its own
// translation unit so that the policies build in parallel. Only the members a
// mixed-precision state runs are instantiated: the routes SEP 36 refuses under a
// reduced precision are refused by name before anything is allocated
// (`require_policy_route`, `GeneralState::binary64`), and their members are never
// compiled for a binary32 source. This translation unit is compiled with
// `-Wdouble-promotion`, so each widening of a source value is written out.

#include "general_impl.hpp"

namespace solprop_native {

template GeneralWorkspace<Mixed>::GeneralWorkspace(Problem problem);
template void GeneralWorkspace<Mixed>::set_initial(const std::vector<Complex>&,
                                                   const std::vector<Complex>&);
template std::vector<double> GeneralWorkspace<Mixed>::trial_step(double,
                                                                 const TrialScalars&);
template void GeneralWorkspace<Mixed>::prepare_trial(double, const TrialScalars&);
template std::vector<double> GeneralWorkspace<Mixed>::collect_trial();
template double GeneralWorkspace<Mixed>::low_storage_step(double, const TrialScalars&,
                                                          bool);
template void GeneralWorkspace<Mixed>::hold_plane();
template std::vector<Complex> GeneralWorkspace<Mixed>::held_to_host() const;
template void GeneralWorkspace<Mixed>::table_upload(TableKind, std::vector<double>,
                                                    double, double,
                                                    const std::complex<double>*,
                                                    std::size_t, std::size_t,
                                                    std::size_t);
template void GeneralWorkspace<Mixed>::table_evaluate(const std::vector<std::size_t>&,
                                                      const std::vector<double>&);
template void GeneralWorkspace<Mixed>::table_rate(std::size_t, double, bool);
template std::vector<Complex>
GeneralWorkspace<Mixed>::table_terms_to_host(TableKind, std::size_t) const;
template std::vector<Complex>
GeneralWorkspace<Mixed>::phase_sum_to_host(const std::vector<double>&);
template std::vector<double>
GeneralWorkspace<Mixed>::weighted_error(const std::vector<double>&);
template void GeneralWorkspace<Mixed>::prepare_raman(const std::vector<double>&,
                                                     const std::vector<double>&);
template std::vector<double> GeneralWorkspace<Mixed>::statistics(
    std::uint64_t, const std::vector<double>&, const std::vector<double>&, double,
    double, double, double, const std::vector<double>&, const std::vector<double>&);
template void GeneralWorkspace<Mixed>::finish(double);
template void GeneralWorkspace<Mixed>::promote_fsal();
template void GeneralWorkspace<Mixed>::clear_fsal();
template double GeneralWorkspace<Mixed>::quadrature_error() const;
template double GeneralWorkspace<Mixed>::quadrature_peak();
template std::string GeneralWorkspace<Mixed>::synthesis_provider() const;
template std::string GeneralWorkspace<Mixed>::projection_provider() const;
template std::size_t GeneralWorkspace<Mixed>::response_team_size() const noexcept;
template std::vector<Complex> GeneralWorkspace<Mixed>::field_to_host() const;
template std::vector<Complex> GeneralWorkspace<Mixed>::stage_to_host(std::size_t) const;
template std::vector<Complex> GeneralWorkspace<Mixed>::projected_to_host(bool);
template std::map<std::string, bool> GeneralWorkspace<Mixed>::dispersive_path() const;
template std::map<std::string, std::size_t>
GeneralWorkspace<Mixed>::memory_ledger() const;
template CpuSizing GeneralWorkspace<Mixed>::execution_sizing() const;
template double GeneralWorkspace<Mixed>::planning_seconds() const noexcept;
template double GeneralWorkspace<Mixed>::raman_planning_seconds() const noexcept;
template std::string GeneralWorkspace<Mixed>::raman_fft_provider() const;
template std::map<std::string, int> GeneralWorkspace<Mixed>::raman_fft_tasks() const;
template std::size_t GeneralWorkspace<Mixed>::raman_kernel_transforms() const noexcept;
template bool GeneralWorkspace<Mixed>::fsal_ready() const noexcept;
template std::size_t GeneralWorkspace<Mixed>::rk_source_evaluations() const noexcept;
template bool GeneralWorkspace<Mixed>::closed() const noexcept;
template void GeneralWorkspace<Mixed>::close() noexcept;
template class GeneralBatchWorkspace<Mixed>;

// The Adams march: binary64 history and compensation, the source binary32 (SEP 36
// section 4.6), admitted by solprop's study admission until phase 5.
template std::vector<Complex>
GeneralWorkspace<Mixed>::adams_source(const std::vector<Complex>&, const TrialScalars&);
template void GeneralWorkspace<Mixed>::adopt(const std::vector<Complex>&, double);
template void GeneralWorkspace<Mixed>::adams_allocate(std::size_t, bool, bool);
template void
GeneralWorkspace<Mixed>::adams_compensation_start(std::size_t,
                                                  const std::vector<double>&);
template void GeneralWorkspace<Mixed>::adams_start(std::size_t, const TrialScalars&);
template double GeneralWorkspace<Mixed>::adams_magnitude();
template void GeneralWorkspace<Mixed>::adams_scale(double, double,
                                                   const std::vector<double>&);
template std::vector<double> GeneralWorkspace<Mixed>::adams_fused_trial(
    double, const AdamsWeights&, const std::vector<double>&, const TrialScalars&);
template void GeneralWorkspace<Mixed>::adams_fused_accept(double, const AdamsWeights&,
                                                          std::size_t,
                                                          const AdamsWeights&);
template void GeneralWorkspace<Mixed>::adams_hold();
template void GeneralWorkspace<Mixed>::adams_restore();
template std::vector<std::string> GeneralWorkspace<Mixed>::adams_row_names() const;
template std::array<std::size_t, 3>
GeneralWorkspace<Mixed>::adams_row_extents(const std::string&) const;
template std::vector<Complex>
GeneralWorkspace<Mixed>::adams_rows_to_host(const std::string&) const;
template void GeneralWorkspace<Mixed>::adams_rows_to_host(const std::string&,
                                                          std::complex<double>*,
                                                          std::size_t) const;
template void
GeneralWorkspace<Mixed>::adams_rows_from_host(const std::string&,
                                              const std::vector<Complex>&);
template void GeneralWorkspace<Mixed>::adams_observe(const AdamsWeights&, bool,
                                                     const TrialScalars&);
template std::vector<double> GeneralWorkspace<Mixed>::adams_exponential_trial(
    double, const ExponentialWeights&, const std::vector<double>&, const TrialScalars&,
    const TrialScalars&, const std::vector<double>&);
template void GeneralWorkspace<Mixed>::adams_exponential_accept(double, std::size_t,
                                                                const AdamsWeights&);
template void
GeneralWorkspace<Mixed>::adams_exponential_observe(double, double,
                                                   const ExponentialWeights&);

} // namespace solprop_native
