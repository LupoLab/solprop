# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The reduced precisions on a device build: the device lane's collection (SEP 36 4).

The native single- and mixed-precision suites hold the installed build against the
oracles whatever its execution space, so on a CUDA or HIP build they qualify the device
forms: this module collects their class R comparisons, witnesses and targeted tests,
and the arithmetic probes, under the ``gpu`` marker, and skips them on a CPU build.
The ledger tests stay with the CPU suites: a device's ledger adds its own rows.
"""

import pytest

import backends.test_native_mixed_precision as mixed
import backends.test_native_reduced_arithmetic as arithmetic
import backends.test_native_single_precision as single

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def device_build() -> None:
    """Skip unless the installed native build runs on a device."""
    native = pytest.importorskip("solprop_native")
    if native.build_info().device_backend == "none":
        pytest.skip("the installed solprop-native build has no accelerator backend")


# The arithmetic probes; the negative test's fixture comes with it.
reduced_math_mode = arithmetic.reduced_math_mode
test_gemm_probe = arithmetic.test_the_gemm_probe_measures_binary32_products
test_probe_without_products = arithmetic.test_routes_without_products_probe_nothing
test_reduced_math_mode_is_refused = arithmetic.test_a_reduced_math_mode_is_refused
test_binary32_transforms = arithmetic.test_binary32_transforms_are_within_their_bound

# Single precision: class R, witnesses and targeted tests.
test_single_class_r = single.test_native_agrees_with_the_oracle_within_both_sides_bounds
test_single_adaptive = single.test_an_adaptive_run_keeps_its_accuracy_envelope
test_single_tensor_raman = single.test_tensor_raman_agrees_with_the_oracle
test_single_batch = single.test_a_batch_member_is_its_solo_run
test_single_scaling = single.test_scaling_by_a_power_of_two_is_exact
test_single_launch_scale = (
    single.test_the_state_is_held_at_the_launch_scale_and_observed_exactly
)
test_single_molecular_raman = single.test_a_molecular_raman_response_survives_narrowing
test_single_populations = single.test_the_plasma_populations_agree_with_the_oracle
test_single_diagnostics = (
    single.test_the_plasma_integration_diagnostics_are_formed_in_binary64
)
test_single_drift = single.test_the_propagator_drift_is_linear_in_the_step_count
test_single_non_finite = single.test_a_non_finite_state_stops_every_march
test_single_spatial_routes = single.test_the_spatial_routes_agree_with_the_oracle
test_single_plasma_beam = single.test_a_plasma_beam_agrees_with_the_oracle
test_single_spatial_statistics = (
    single.test_the_spatial_statistics_are_binary64_reductions_of_the_state
)
test_single_envelope_plasma = (
    single.test_the_envelope_plasma_agrees_within_its_class_t_envelope
)

# Mixed precision: class R, the linear-part, damping and scale witnesses, targeted.
test_mixed_fibre = mixed.test_the_fibre_agrees_with_the_oracle
test_mixed_cubic_routes = mixed.test_the_cubic_routes_agree_with_the_oracle
test_mixed_field_plasma = mixed.test_the_field_grid_plasma_agrees_with_the_oracle
test_mixed_in_stage_collar = (
    mixed.test_an_in_stage_temporal_collar_agrees_with_the_oracle
)
test_mixed_tensor_raman = mixed.test_tensor_raman_agrees_with_the_oracle
test_mixed_batch = mixed.test_a_batch_member_is_its_solo_run
test_mixed_source_free = (
    mixed.test_a_source_free_run_agrees_with_binary64_within_binary64_bounds
)
test_mixed_damped_bin = mixed.test_a_strongly_damped_bin_runs_and_agrees_with_the_oracle
test_mixed_scaling = mixed.test_scaling_by_a_power_of_two_is_exact
test_mixed_scale_witness = (
    mixed.test_a_launch_scale_of_two_to_the_thirty_gives_the_physical_source
)
test_mixed_physical_state = mixed.test_the_state_is_physical_and_observed_exactly
test_mixed_adams = (
    mixed.test_an_adams_run_agrees_with_binary64_adams_under_the_study_admission
)
test_mixed_adams_snapshot = mixed.test_an_adams_snapshot_round_trip_is_exact
test_mixed_non_finite = mixed.test_a_non_finite_state_stops_every_march
test_mixed_envelope_plasma = (
    mixed.test_the_envelope_plasma_agrees_within_its_class_t_envelope
)
test_mixed_spatial_routes = mixed.test_the_spatial_routes_agree_with_the_oracle
test_mixed_spatial_collars = mixed.test_in_stage_spatial_collars_agree_with_the_oracle
test_mixed_source_free_cartesian = (
    mixed.test_a_source_free_cartesian_run_agrees_with_binary64_within_binary64_bounds
)
