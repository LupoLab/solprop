// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// The far-field pass of solprop SEP 23 against a straightforward serial host reduction
// of the same definitions: the beam and axial-flux spectra, the absorbed channels, the
// regions, the apertures' collected and on-axis spectra and the on-axis spectrum, on a
// plane and on a line of several blocks, on field and envelope grids. The two sum the
// same terms in different associations, so they agree within a rounding bound fixed
// before running (solprop SEP 25), never by bits; hard-edge membership is shared.

#include "bounds.hpp"
#include "solprop_native/core.hpp"
#include "solprop_native/far_field.hpp"

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

namespace native = solprop_native;
using native::Complex;
using solprop_native::test::gamma;

constexpr double kEpsilon = std::numeric_limits<double>::epsilon();
constexpr double kDistance = 0.25;
constexpr std::uint64_t kEvery = native::far_field_flags;

struct Case {
  native::Problem problem;
  std::vector<Complex> field, on_axis, square, square_low;
  std::vector<double> weights, transverse, transverse_low, spectral_rate,
      transverse_rate, omega;
  std::vector<unsigned char> active;
  std::vector<double> apertures, regions;
  double evanescent{2e4};
};

// A transverse grid of spacing `dk` rad/m in FFT order, `count` samples.
std::vector<double> fft_axis(const std::size_t count, const double dk) {
  std::vector<double> axis(count);
  for (std::size_t index = 0; index < count; ++index) {
    const auto signed_index = index < (count + 1) / 2 ? static_cast<double>(index)
                                                      : static_cast<double>(index) -
                                                            static_cast<double>(count);
    axis[index] = dk * signed_index;
  }
  return axis;
}

Case make_case(const std::size_t n_x, const std::size_t n_y, const std::size_t n_pol,
               const bool envelope, const int profile) {
  Case data;
  auto& problem = data.problem;
  const std::size_t n = 9;
  problem.n_x = n_x;
  problem.n_y = n_y;
  problem.n_omega = n;
  problem.n_t = envelope ? n : 2 * (n - 1);
  problem.n_pol_state = n_pol;
  problem.n_modes = n_x * n_y;
  problem.dt = 1.5e-15;
  problem.family =
      envelope ? native::FieldFamily::envelope : native::FieldFamily::field_real;
  problem.spatial_k_x = fft_axis(n_x, n_y > 1 ? 1.5e6 : 1.0e5);
  problem.spatial_k_y = n_y > 1 ? fft_axis(n_y, 1.2e6) : std::vector<double>{0.0};
  const std::size_t columns = n_x * n_y;
  std::mt19937_64 random(7);
  std::normal_distribution<double> normal;
  data.field.resize(columns * n_pol * n);
  for (auto& value : data.field)
    value = Complex(1e3 * normal(random), 1e3 * normal(random));
  for (std::size_t column = 0; column < columns; ++column) {
    data.weights.push_back(1e-9 * (1.0 + 0.01 * static_cast<double>(column % 7)));
    const double phase = 0.3 * static_cast<double>(column);
    data.on_axis.emplace_back(std::cos(phase) / static_cast<double>(columns),
                              std::sin(phase) / static_cast<double>(columns));
    const double kx = problem.spatial_k_x[column % n_x];
    const double ky = problem.spatial_k_y[column / n_x];
    data.transverse.push_back(kx * kx + ky * ky);
    data.transverse_low.push_back(1e-3 * static_cast<double>(column % 3));
    data.transverse_rate.push_back(column % 2 == 0 ? 0.0 : 3.0);
  }
  for (std::size_t bin = 0; bin < n; ++bin) {
    // Frequencies from zero (a field grid's first bin) upwards, some bins inactive
    // and a medium of complex index whose wavenumber is 0.4 omega / c, so every
    // channel and the zero-frequency rule take part and the outer columns are
    // evanescent; the images in the mask plane follow the vacuum mapping.
    const double omega = envelope ? 2.0e15 + 1e14 * static_cast<double>(bin)
                                  : 2.5e14 * static_cast<double>(bin);
    data.omega.push_back(omega);
    const double k = 0.4 * omega / native::far_field_speed_of_light;
    data.square.emplace_back(k * k, 1e-6 * k * k);
    data.square_low.emplace_back(1e-4 * static_cast<double>(bin), 0.0);
    data.spectral_rate.push_back(bin + 1 == n ? 5.0 : 0.0);
    data.active.push_back(bin == 1 ? 0U : 1U);
  }
  // Three apertures: code `profile`, the scale row z c / omega (zero at omega <= 0).
  const double centres[3][2] = {{0.0, 0.0}, {0.056, -0.045}, {-0.056, 0.045}};
  const double diameters[3] = {0.12, 0.06, 0.09};
  for (std::size_t index = 0; index < 3; ++index) {
    const double half_width = diameters[index] / 2.0;
    const double width = diameters[index] / 8.0;
    const double edge = profile == 3 ? 4.0 : profile == 0 ? 0.0 : width;
    const double inverse = profile == 1 || profile == 2 ? 1.0 / width : 0.0;
    const double y = n_y > 1 ? centres[index][1] : 0.0;
    for (const double value :
         {centres[index][0], y, half_width, inverse, edge, kDistance})
      data.apertures.push_back(value);
    for (std::size_t bin = 0; bin < n; ++bin)
      data.apertures.push_back(data.omega[bin] > 0.0
                                   ? kDistance * native::far_field_speed_of_light /
                                         data.omega[bin]
                                   : 0.0);
  }
  const double inf = std::numeric_limits<double>::infinity();
  data.regions = {-inf, 0.0, -inf, 0.0, 0.0, inf, -inf, 0.0,
                  -inf, 0.0, 0.0,  inf, 0.0, inf, 0.0,  inf};
  return data;
}

template <class T> auto device(const char* label, const std::vector<T>& values) {
  Kokkos::View<T*, native::Layout, native::MemorySpace> view(label, values.size());
  const auto host = Kokkos::create_mirror_view(view);
  for (std::size_t index = 0; index < values.size(); ++index)
    host(index) = values[index];
  Kokkos::deep_copy(view, host);
  return view;
}

std::vector<double> run(Case& data, const std::uint64_t flags, const int profile,
                        const int exponent) {
  const auto n = data.problem.n_omega;
  const auto rows = data.problem.n_x * data.problem.n_y * data.problem.n_pol_state;
  native::FarFieldPass::ComplexView2 field("test.field", rows, n);
  const auto host = Kokkos::create_mirror_view(field);
  for (std::size_t row = 0; row < rows; ++row)
    for (std::size_t bin = 0; bin < n; ++bin)
      host(row, bin) = data.field[row * n + bin];
  Kokkos::deep_copy(field, host);
  native::FarFieldPass pass(data.problem, native::ExecutionSpace{});
  const native::FarFieldPass::Inputs inputs{
      device("test.weights", data.weights),
      device("test.on_axis", data.on_axis),
      device("test.square", data.square),
      device("test.square_low", data.square_low),
      device("test.transverse", data.transverse),
      device("test.transverse_low", data.transverse_low),
      device("test.spectral_rate", data.spectral_rate),
      device("test.transverse_rate", data.transverse_rate),
      device("test.omega", data.omega),
      device("test.active", data.active),
      data.evanescent};
  auto values = pass.run(flags, data.apertures, profile, exponent, data.regions, 0.5,
                         field, inputs);
  pass.close();
  return values;
}

// The serial host reduction: each definition written out directly, every sum over
// columns in one plain loop. `magnitude` forms the same sums of |terms|, the scale of
// the rounding bound.
std::vector<double> reference(const Case& data, const int profile, const int exponent,
                              const bool magnitude) {
  const auto& problem = data.problem;
  const auto n = problem.n_omega;
  const auto n_pol = problem.n_pol_state;
  const auto columns = problem.n_x * problem.n_y;
  const bool envelope = problem.family == native::FieldFamily::envelope;
  const double dt = problem.dt;
  const double density =
      envelope ? dt * dt / (2.0 * std::numbers::pi) : 2.0 * dt * dt / std::numbers::pi;
  const std::size_t row = native::far_field_aperture_constants + n;
  const std::size_t n_apertures = data.apertures.size() / row;
  const std::size_t n_regions = data.regions.size() / 4;
  std::vector<double> beam(n), flux(n), absorbed(4), axis(n);
  std::vector<std::vector<double>> regions(n_regions, std::vector<double>(n));
  std::vector<std::vector<double>> collected(n_apertures, std::vector<double>(n));
  std::vector<std::vector<double>> through(n_apertures, std::vector<double>(n));
  for (std::size_t bin = 0; bin < n; ++bin) {
    std::vector<std::complex<double>> amplitude(n_pol), axial(n_apertures * n_pol);
    for (std::size_t column = 0; column < columns; ++column) {
      const double kx = problem.spatial_k_x[column % problem.n_x];
      const double ky = problem.spatial_k_y[column / problem.n_x];
      double power = 0.0;
      for (std::size_t pol = 0; pol < n_pol; ++pol)
        power += std::norm(
            std::complex<double>(data.field[(column * n_pol + pol) * n + bin].real(),
                                 data.field[(column * n_pol + pol) * n + bin].imag()));
      const double weighted = data.weights[column] * power;
      beam[bin] += weighted;
      const std::complex<double> longitudinal =
          (std::complex<double>(data.square[bin].real(), data.square[bin].imag()) -
           data.transverse[column]) +
          (std::complex<double>(data.square_low[bin].real(),
                                data.square_low[bin].imag()) -
           data.transverse_low[column]);
      const auto root = std::sqrt(longitudinal);
      const bool propagating = longitudinal.real() > 0.0;
      const double attenuation =
          propagating ? root.imag() : std::min(std::abs(root.imag()), data.evanescent);
      const bool live = data.active[bin] != 0U;
      if (live && propagating && data.omega[bin] > 0.0)
        flux[bin] +=
            weighted * native::far_field_speed_of_light * root.real() / data.omega[bin];
      const double energy = (envelope || bin == 0 || bin + 1 == n ? 1.0 : 4.0) * dt /
                            static_cast<double>(problem.n_t);
      absorbed[0] += energy * weighted * data.spectral_rate[bin];
      if (live) {
        absorbed[1] += energy * weighted * data.transverse_rate[column];
        absorbed[propagating ? 3 : 2] +=
            energy * weighted * 2.0 * (magnitude ? std::abs(attenuation) : attenuation);
      }
      for (std::size_t region = 0; region < n_regions; ++region)
        if (native::far_field_member(
                kx, ky, data.regions[4 * region], data.regions[4 * region + 1],
                data.regions[4 * region + 2], data.regions[4 * region + 3]))
          regions[region][bin] += weighted;
      const std::complex<double> alpha(data.on_axis[column].real(),
                                       data.on_axis[column].imag());
      for (std::size_t pol = 0; pol < n_pol; ++pol) {
        const auto& stored = data.field[(column * n_pol + pol) * n + bin];
        const std::complex<double> value(stored.real(), stored.imag());
        amplitude[pol] += magnitude ? std::abs(alpha) * std::abs(value) : alpha * value;
      }
      for (std::size_t index = 0; index < n_apertures; ++index) {
        const double* constants = &data.apertures[index * row];
        const double scale = constants[native::far_field_aperture_constants + bin];
        const double radius =
            native::far_field_radius(scale, kx, ky, constants[0], constants[1]);
        const double origin = native::far_field_radius(scale, kx, ky, 0.0, 0.0);
        const double weight =
            native::far_field_weight(radius, origin, scale, constants[2], constants[3],
                                     constants[4], constants[5], profile);
        collected[index][bin] += weighted * (exponent == 1 ? weight : weight * weight);
        for (std::size_t pol = 0; pol < n_pol; ++pol) {
          const auto& stored = data.field[(column * n_pol + pol) * n + bin];
          const std::complex<double> value(stored.real(), stored.imag());
          axial[index * n_pol + pol] += magnitude
                                            ? std::abs(alpha) * weight * std::abs(value)
                                            : alpha * (weight * value);
        }
      }
    }
    for (std::size_t pol = 0; pol < n_pol; ++pol)
      axis[bin] += density * std::norm(amplitude[pol]);
    for (std::size_t index = 0; index < n_apertures; ++index)
      for (std::size_t pol = 0; pol < n_pol; ++pol)
        through[index][bin] += density * std::norm(axial[index * n_pol + pol]);
  }
  std::vector<double> packed;
  for (const double value : beam)
    packed.push_back(0.5 * density * value);
  for (const double value : flux)
    packed.push_back(0.5 * density * value);
  for (const double value : absorbed)
    packed.push_back(0.5 * value);
  for (const auto& rows : {regions, collected})
    for (const auto& one : rows)
      for (const double value : one)
        packed.push_back(0.5 * density * value);
  for (const auto& one : through)
    for (const double value : one)
      packed.push_back(0.5 * value);
  for (const double value : axis)
    packed.push_back(0.5 * value);
  return packed;
}

// Each value is a sum over the columns (and bands) of terms of up to eight rounded
// operations; a squared modulus of a complex sum at most triples the relative bound,
// and an aperture's weight carries a transcendental routine's absolute error of a few
// ulps of one: 3 gamma(8 m + 64) of the sum of the magnitudes, plus 8 eps of it.
void expect_matches(const std::vector<double>& actual,
                    const std::vector<double>& expected,
                    const std::vector<double>& magnitude, const std::size_t columns) {
  ASSERT_EQ(actual.size(), expected.size());
  const double relative = 3.0 * gamma(8 * columns + 64) + 8.0 * kEpsilon;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const double bound = relative * std::abs(magnitude[index]);
    // A sum of no terms (a hole no image reaches at a bin) is an exact zero.
    if (magnitude[index] == 0.0) {
      EXPECT_EQ(actual[index], 0.0) << index;
      continue;
    }
    EXPECT_LE(std::abs(actual[index] - expected[index]), bound) << index;
  }
}

struct Variant {
  std::size_t n_x, n_y, n_pol;
  bool envelope;
  int profile, exponent;
};

class FarFieldPassAgainstHost : public ::testing::TestWithParam<Variant> {};

TEST_P(FarFieldPassAgainstHost, EveryStatisticMatchesTheSerialReduction) {
  const auto variant = GetParam();
  auto data = make_case(variant.n_x, variant.n_y, variant.n_pol, variant.envelope,
                        variant.profile);
  const auto actual = run(data, kEvery, variant.profile, variant.exponent);
  const auto expected = reference(data, variant.profile, variant.exponent, false);
  const auto magnitude = reference(data, variant.profile, variant.exponent, true);
  expect_matches(actual, expected, magnitude, variant.n_x * variant.n_y);
  // Every channel and aperture takes part on this fixture.
  const auto n = data.problem.n_omega;
  for (std::size_t channel = 0; channel < 4; ++channel)
    EXPECT_NE(actual[2 * n + channel], 0.0) << channel;
  // On a line the boxes excluding k_y = 0 select nothing (SEP 23 section 4.10).
  for (std::size_t row = variant.n_y == 1 ? 2 : 0; row < 4 + 3; ++row) {
    double total = 0.0;
    for (std::size_t bin = 0; bin < n; ++bin)
      total += actual[2 * n + 4 + row * n + bin];
    EXPECT_GT(total, 0.0) << row;
  }
}

INSTANTIATE_TEST_SUITE_P(Grids, FarFieldPassAgainstHost,
                         ::testing::Values(Variant{5, 4, 1, true, 1, 2},
                                           Variant{5, 4, 2, false, 2, 1},
                                           Variant{6, 3, 1, false, 3, 2},
                                           Variant{5, 4, 2, true, 0, 2},
                                           Variant{130, 1, 1, true, 2, 1}));

TEST(FarFieldPass, EveryRadiusPathMatchesTheSerialReduction) {
  // One distance and centre (both radii shared), one distance (the origin's shared)
  // and a second distance (neither): each against the host reduction.
  for (const int path : {0, 1, 2}) {
    auto data = make_case(5, 4, 2, true, 1);
    const std::size_t row = native::far_field_aperture_constants + data.problem.n_omega;
    for (std::size_t index = 1; index < 3; ++index) {
      if (path == 0) {
        data.apertures[index * row] = data.apertures[0];
        data.apertures[index * row + 1] = data.apertures[1];
      }
      if (path == 2 && index == 2) {
        data.apertures[index * row + 5] = 0.3;
        for (std::size_t bin = 0; bin < data.problem.n_omega; ++bin)
          data.apertures[index * row + 6 + bin] =
              0.3 * native::far_field_speed_of_light / data.omega[bin];
      }
    }
    const auto actual = run(data, kEvery, 1, 1);
    expect_matches(actual, reference(data, 1, 1, false), reference(data, 1, 1, true),
                   20);
  }
}

TEST(FarFieldPass, APartialRequestPacksOnlyItsRows) {
  auto data = make_case(5, 4, 1, true, 1);
  const auto every = run(data, kEvery, 1, 2);
  const auto n = data.problem.n_omega;
  // The axial flux and the apertures' on-axis spectra alone: the same values as the
  // full request's rows, within the bound (another slot count, the same sums).
  const auto partial =
      run(data, native::far_field_poynting | native::far_field_amplitude, 1, 2);
  ASSERT_EQ(partial.size(), 4 * n);
  const auto magnitude = reference(data, 1, 2, true);
  const double relative = 3.0 * gamma(8 * 20 + 64) + 8.0 * kEpsilon;
  for (std::size_t bin = 0; bin < n; ++bin) {
    EXPECT_LE(std::abs(partial[bin] - every[n + bin]), relative * magnitude[n + bin]);
    for (std::size_t index = 0; index < 3; ++index) {
      const auto full = 2 * n + 4 + (4 + 3) * n + index * n + bin;
      EXPECT_LE(std::abs(partial[n + index * n + bin] - every[full]),
                relative * magnitude[full]);
    }
  }
  // A region request without regions forms nothing.
  data.regions.clear();
  EXPECT_TRUE(run(data, native::far_field_regions, 1, 2).empty());
}

TEST(FarFieldPass, DeclaredAperturesCostNothingUnlessAsked) {
  // A periodic beam spectrum or region sample with apertures declared: the aperture
  // loop is gated on the layout's aperture slot, so the rows are those of a request
  // that carries no apertures, within the bound (the same sums, no aperture term).
  auto data = make_case(5, 4, 2, true, 2);
  const std::uint64_t flags = native::far_field_beam | native::far_field_regions |
                              native::far_field_absorbed | native::far_field_axis;
  const auto declared = run(data, flags, 2, 1);
  const auto magnitude = reference(data, 2, 1, true);
  data.apertures.clear();
  const auto bare = run(data, flags, 2, 1);
  ASSERT_EQ(declared.size(), bare.size());
  const auto n = data.problem.n_omega;
  ASSERT_EQ(bare.size(), (1 + 4 + 1) * n + 4);
  const double relative = 3.0 * gamma(8 * 20 + 64) + 8.0 * kEpsilon;
  for (std::size_t index = 0; index < bare.size(); ++index) {
    // Packed rows: the beam, the absorbed powers, the regions, the axis; the full
    // request also packs the flux row and three rows of each aperture statistic.
    const std::size_t full = index < n           ? index
                             : index < n + 4     ? index + n
                             : index < 5 * n + 4 ? index + n
                                                 : index + (1 + 3 + 3) * n;
    EXPECT_LE(std::abs(declared[index] - bare[index]), relative * magnitude[full])
        << index;
  }
}

TEST(FarFieldPass, TheLayoutAndBandsAreSolpropsOwn) {
  // solprop's far_field_layout: every quantity, three apertures, four regions, two
  // polarisations.
  const auto layout = native::far_field_layout(kEvery, 3, 4, 2);
  EXPECT_EQ(layout.beam, 0);
  EXPECT_EQ(layout.poynting, 1);
  EXPECT_EQ(layout.absorbed, 2);
  EXPECT_EQ(layout.regions, 6);
  EXPECT_EQ(layout.apertures, 10);
  EXPECT_EQ(layout.stride, 5);
  EXPECT_EQ(layout.collected, 0);
  EXPECT_EQ(layout.amplitude, 1);
  EXPECT_EQ(layout.axis, 25);
  EXPECT_EQ(layout.n_slots, 29U);
  const auto alone = native::far_field_layout(native::far_field_amplitude, 2, 0, 1);
  EXPECT_EQ(alone.apertures, 0);
  EXPECT_EQ(alone.stride, 2);
  EXPECT_EQ(alone.collected, -1);
  EXPECT_EQ(alone.amplitude, 0);
  EXPECT_EQ(native::far_field_layout(native::far_field_regions, 0, 0, 1).n_slots, 0U);
  // solprop's far_field_bands: a row per band on a plane, 64-column blocks on a line,
  // the last block short.
  EXPECT_EQ(native::far_field_band_count(5, 7, 3, 10), 7U);
  EXPECT_EQ(native::far_field_band_count(130, 1, 2, 8), 3U);
  EXPECT_EQ(native::far_field_band_edge(1, 3, 130, 1), 64U);
  EXPECT_EQ(native::far_field_band_edge(3, 3, 130, 1), 130U);
  EXPECT_EQ(native::far_field_band_edge(2, 7, 5, 7), 10U);
  // The budget caps the bands: 512 rows of 4096 slots by 64 bins admit 64.
  EXPECT_EQ(native::far_field_band_count(8, 512, 4096, 64), 64U);
  EXPECT_EQ(native::far_field_band_edge(1, 64, 8, 512), 64U);
  EXPECT_THROW(static_cast<void>(native::far_field_band_count(5, 7, 0, 10)),
               std::invalid_argument);
}

TEST(FarFieldPass, TheElementFunctionsHoldTheirContracts) {
  // The radius as written; the weight's zeros; the raised cosine's compact support.
  const double radius = native::far_field_radius(2.0, 3.0, -1.0, 0.5, 0.25);
  EXPECT_EQ(radius, std::sqrt(6.5 * 6.5 + 1.75 * 1.75));
  EXPECT_EQ(native::far_field_weight(0.5, 0.0, 0.0, 1.0, 8.0, 0.0, 10.0, 1), 0.0);
  EXPECT_EQ(native::far_field_weight(0.5, 10.0, 1.0, 1.0, 8.0, 0.0, 10.0, 1), 0.0);
  EXPECT_EQ(native::far_field_weight(1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 10.0, 0), 1.0);
  EXPECT_EQ(native::far_field_weight(std::nextafter(1.0, 2.0), 0.0, 1.0, 1.0, 0.0, 0.0,
                                     10.0, 0),
            0.0);
  EXPECT_EQ(native::far_field_weight(0.875, 0.0, 1.0, 1.0, 8.0, 0.0, 10.0, 2), 1.0);
  EXPECT_EQ(native::far_field_weight(1.125, 0.0, 1.0, 1.0, 8.0, 0.0, 10.0, 2), 0.0);
  EXPECT_NEAR(native::far_field_weight(1.0, 0.0, 1.0, 1.0, 8.0, 0.0, 10.0, 1), 0.5,
              2 * kEpsilon);
  EXPECT_NEAR(native::far_field_weight(1.0, 0.0, 1.0, 1.0, 0.0, 4.0, 10.0, 3),
              std::exp(-1.0), 4 * kEpsilon);
  // tanh's saturation: the skipped transcendental's value, exactly.
  for (const double t : {22.0, 30.0, 1e3}) {
    EXPECT_EQ(std::tanh(t), 1.0);
    EXPECT_EQ(native::far_field_weight(1.0 + t / 8.0, 0.0, 1.0, 1.0, 8.0, 0.0, 10.0, 1),
              0.0);
    EXPECT_EQ(native::far_field_weight(0.0, 0.0, 1.0, t / 8.0, 8.0, 0.0, 1e9, 1), 1.0);
  }
  EXPECT_EQ(
      0.5 * (1.0 - std::tanh(21.9)),
      native::far_field_weight(1.0 + 21.9 / 8.0, 0.0, 1.0, 1.0, 8.0, 0.0, 10.0, 1));
  EXPECT_TRUE(native::far_field_member(-0.0, -1.0, 0.0, 1.0, 0.0, 2.0));
  EXPECT_FALSE(native::far_field_member(0.0, -2.0, 0.0, 1.0, 0.0, 2.0));
}

TEST(FarFieldPass, InvalidRequestsAreRefused) {
  auto data = make_case(5, 4, 1, true, 1);
  EXPECT_THROW(static_cast<void>(run(data, 16, 1, 2)), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(run(data, 0, 1, 2)), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(run(data, kEvery, 4, 2)), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(run(data, kEvery, 1, 3)), std::invalid_argument);
  auto short_table = data;
  short_table.apertures.pop_back();
  EXPECT_THROW(static_cast<void>(run(short_table, kEvery, 1, 2)),
               std::invalid_argument);
  auto negative = data;
  negative.apertures[native::far_field_aperture_constants + 2] = -1.0;
  EXPECT_THROW(static_cast<void>(run(negative, kEvery, 1, 2)), std::invalid_argument);
  auto undefined = data;
  undefined.regions[0] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(static_cast<void>(run(undefined, kEvery, 1, 2)), std::invalid_argument);
  auto bare = data;
  bare.apertures.clear();
  EXPECT_THROW(static_cast<void>(run(bare, native::far_field_collected, 1, 2)),
               std::invalid_argument);
  auto radial = data;
  radial.problem.n_x = 0;
  EXPECT_THROW(static_cast<void>(run(radial, kEvery, 1, 2)), std::invalid_argument);
}

} // namespace
