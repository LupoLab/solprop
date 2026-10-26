// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// The Single policy's general engine (solprop SEP 36 section 6.3, phase 2), in its own
// translation unit so that the policies build in parallel. Only the members a
// single-precision state runs are instantiated: the routes SEP 36 has not admitted
// under single precision are refused by name before anything is allocated
// (`require_policy_route`, `GeneralState::binary64`), and their members are never
// compiled for binary32. This translation unit is compiled with `-Wdouble-promotion`.

#include "general_impl.hpp"

namespace solprop_native {

template GeneralWorkspace<Single>::GeneralWorkspace(Problem problem);
template void GeneralWorkspace<Single>::set_initial(const std::vector<Complex>&,
                                                    const std::vector<Complex>&);
template std::vector<double> GeneralWorkspace<Single>::trial_step(double,
                                                                  const TrialScalars&);
template void GeneralWorkspace<Single>::prepare_trial(double, const TrialScalars&);
template std::vector<double> GeneralWorkspace<Single>::collect_trial();
template double GeneralWorkspace<Single>::low_storage_step(double, const TrialScalars&,
                                                           bool);
template void GeneralWorkspace<Single>::hold_plane();
template std::vector<Complex> GeneralWorkspace<Single>::held_to_host() const;
template void GeneralWorkspace<Single>::table_upload(TableKind, std::vector<double>,
                                                     double, double,
                                                     const std::complex<double>*,
                                                     std::size_t, std::size_t,
                                                     std::size_t);
template void GeneralWorkspace<Single>::table_evaluate(const std::vector<std::size_t>&,
                                                       const std::vector<double>&);
template void GeneralWorkspace<Single>::table_rate(std::size_t, double, bool);
template std::vector<Complex>
GeneralWorkspace<Single>::table_terms_to_host(TableKind, std::size_t) const;
template std::vector<Complex>
GeneralWorkspace<Single>::phase_sum_to_host(const std::vector<double>&);
template std::vector<double>
GeneralWorkspace<Single>::weighted_error(const std::vector<double>&);
template void GeneralWorkspace<Single>::prepare_raman(const std::vector<double>&,
                                                      const std::vector<double>&);
template std::vector<double> GeneralWorkspace<Single>::statistics(
    std::uint64_t, const std::vector<double>&, const std::vector<double>&, double,
    double, double, double, const std::vector<double>&, const std::vector<double>&);
template std::vector<double>
GeneralWorkspace<Single>::spatial_statistics(std::uint64_t, double, double);
template void GeneralWorkspace<Single>::finish(double);
template void GeneralWorkspace<Single>::promote_fsal();
template void GeneralWorkspace<Single>::clear_fsal();
template double GeneralWorkspace<Single>::quadrature_error() const;
template double GeneralWorkspace<Single>::quadrature_peak();
template std::string GeneralWorkspace<Single>::synthesis_provider() const;
template std::string GeneralWorkspace<Single>::projection_provider() const;
template std::size_t GeneralWorkspace<Single>::response_team_size() const noexcept;
template std::vector<Complex> GeneralWorkspace<Single>::field_to_host() const;
template std::vector<Complex>
GeneralWorkspace<Single>::stage_to_host(std::size_t) const;
template std::vector<Complex> GeneralWorkspace<Single>::projected_to_host(bool);
template std::map<std::string, bool> GeneralWorkspace<Single>::dispersive_path() const;
template std::map<std::string, std::size_t>
GeneralWorkspace<Single>::memory_ledger() const;
template CpuSizing GeneralWorkspace<Single>::execution_sizing() const;
template double GeneralWorkspace<Single>::planning_seconds() const noexcept;
template double GeneralWorkspace<Single>::raman_planning_seconds() const noexcept;
template std::string GeneralWorkspace<Single>::raman_fft_provider() const;
template std::map<std::string, int> GeneralWorkspace<Single>::raman_fft_tasks() const;
template std::size_t GeneralWorkspace<Single>::raman_kernel_transforms() const noexcept;
template bool GeneralWorkspace<Single>::fsal_ready() const noexcept;
template std::size_t GeneralWorkspace<Single>::rk_source_evaluations() const noexcept;
template bool GeneralWorkspace<Single>::closed() const noexcept;
template double GeneralWorkspace<Single>::gemm_probe_error() const noexcept;
template void GeneralWorkspace<Single>::close() noexcept;
template class GeneralBatchWorkspace<Single>;

} // namespace solprop_native
