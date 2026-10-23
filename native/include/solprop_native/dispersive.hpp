// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "solprop_native/core.hpp"
#include "solprop_native/fft_plan.hpp"

#include <KokkosFFT.hpp>

#include <array>
#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace solprop_native {

// solprop SEP 6's frequency-dependent electronic response (`responses.dispersive_kerr`)
// on the engine's fused node path: the legacy transform's real synthesis and
// projection, the same at every frequency, and a temporal collar the same at every
// node. A spectral multiplier and the collar commute with both, so the response's
// multipliers act on the modal columns rather than at the nodes (solprop
// `backends/numba_cpu/dispersive.py`, whose structure this is):
//
// 1. the input multipliers act on the padded modal spectrum before the stage's inverse
//    transform, and one batched inverse transform gives every filtered modal field;
// 2. the filtered fields are synthesised at a chunk's nodes by the same real matrix as
//    the bare field, and the products are formed there in the leaf's sector, the
//    collar multiplying each (`G[w (G E)^3]`, decided 2026-10-08);
// 3. each product is projected onto the modal columns (and onto the embedded rule's);
// 4. one batched forward transform, and the products' stage multipliers -- the output
//    multipliers times the channels' `eps0 gamma0 rho` -- land them in the stage's own
//    polarisation spectrum. The embedded rule's are transformed and landed only when a
//    step is scored, from the last stage's products (`finish_coarse`).
//
// Alone (no constant Kerr term, plasma, Raman or envelope plasma beside it) a symmetric
// response is one product, which reads the stage's node buffer: with no in-stage
// absorber reading the bare modal field it filters the stage's spectrum in place, and
// when its product has the polarisation's type it is the stage's polarisation, whose
// own projections and transforms then serve it -- the constant cubic's transforms.
//
// The other routes take the same response per chunk (solprop
// `backends/numba_cpu/dispersive_spatial.py` and `dispersive_general.py`, SEP 6 Phase
// 3b): on the Cartesian free-space and line routes the chunk's temporal rows are its
// nodes, on the hybrid planar route a block's (p, y) rows are synthesised at its
// (x, y) nodes by the real slab rule in x (`SeparableDispersive`); on the general
// modal route the input multipliers act on each chunk's synthesised spectrum and the
// products are formed at each channel's own region's nodes, with no collar
// (`GeneralDispersive`). Every route weighs its channels at a stage by one rule
// (`DispersiveWeights`).

/** The product form of a grid and a sector: solprop `dispersive_form`. */
enum class DispersiveForm : int {
  /** A field grid's full cubic: the real product of real fields. */
  real = 0,
  /** The number-conserving group of analytic signals or envelopes. */
  conserving = 1,
  /** An envelope's full cubic: the conserving group and both harmonic groups. */
  harmonic = 2,
};

/**
 * Refuse a dispersive problem whose layout this response cannot execute.
 *
 * Called before any allocation; a problem without the response passes.
 */
void require_dispersive(const Problem& problem);

// --- the sample bodies --------------------------------------------------------------
//
// solprop `backends/bodies.py`'s dispersive bodies, operation for operation: the
// isotropic Kleinman contraction `T_ijkl x_j y_k z_l` of two-component fields, its
// number-conserving and harmonic groups, and the closed forms of the products
// `(u, u, u)`, `(u, u, f)` and `(u, f, f)` of one component.

/** `[(x . y) z + (x . z) y + (y . z) x] / 3`, without conjugation. */
template <class T>
KOKKOS_INLINE_FUNCTION void dispersive_isotropic(const T x0, const T x1, const T y0,
                                                 const T y1, const T z0, const T z1,
                                                 T& out0, T& out1) {
  const T xy = x0 * y0 + x1 * y1;
  const T xz = x0 * z0 + x1 * z1;
  const T yz = y0 * z0 + y1 * z1;
  out0 = (xy * z0 + xz * y0 + yz * x0) / 3.0;
  out1 = (xy * z1 + xz * y1 + yz * x1) / 3.0;
}

/** `T(x, y, z*) + T(x, y*, z) + T(x*, y, z)`. */
KOKKOS_INLINE_FUNCTION void dispersive_conserving(const Complex x0, const Complex x1,
                                                  const Complex y0, const Complex y1,
                                                  const Complex z0, const Complex z1,
                                                  Complex& out0, Complex& out1) {
  Complex a0, a1, b0, b1, c0, c1;
  dispersive_isotropic(x0, x1, y0, y1, Kokkos::conj(z0), Kokkos::conj(z1), a0, a1);
  dispersive_isotropic(x0, x1, Kokkos::conj(y0), Kokkos::conj(y1), z0, z1, b0, b1);
  dispersive_isotropic(Kokkos::conj(x0), Kokkos::conj(x1), y0, y1, z0, z1, c0, c1);
  out0 = a0 + b0 + c0;
  out1 = a1 + b1 + c1;
}

/** `c T(x, y, z) + c* [T(x, y*, z*) + T(x*, y, z*) + T(x*, y*, z)]`. */
KOKKOS_INLINE_FUNCTION void dispersive_harmonic(const Complex x0, const Complex x1,
                                                const Complex y0, const Complex y1,
                                                const Complex z0, const Complex z1,
                                                const Complex carrier, Complex& out0,
                                                Complex& out1) {
  Complex u0, u1, a0, a1, b0, b1, c0, c1;
  dispersive_isotropic(x0, x1, y0, y1, z0, z1, u0, u1);
  const Complex x0c = Kokkos::conj(x0), x1c = Kokkos::conj(x1);
  const Complex y0c = Kokkos::conj(y0), y1c = Kokkos::conj(y1);
  const Complex z0c = Kokkos::conj(z0), z1c = Kokkos::conj(z1);
  dispersive_isotropic(x0, x1, y0c, y1c, z0c, z1c, a0, a1);
  dispersive_isotropic(x0c, x1c, y0, y1, z0c, z1c, b0, b1);
  dispersive_isotropic(x0c, x1c, y0c, y1c, z0, z1, c0, c1);
  const Complex down = Kokkos::conj(carrier);
  out0 = carrier * u0 + down * (a0 + b0 + c0);
  out1 = carrier * u1 + down * (a1 + b1 + c1);
}

/** `x^3`: `(x x) x`. */
KOKKOS_INLINE_FUNCTION double dispersive_real_self(const double x) { return x * x * x; }
/** `x^2 f`: `(x x) f`. */
KOKKOS_INLINE_FUNCTION double dispersive_real_pair(const double x, const double f) {
  return x * x * f;
}
/** `x f^2`: `(x f) f`. */
KOKKOS_INLINE_FUNCTION double dispersive_real_cross(const double x, const double f) {
  return x * f * f;
}
/** `3 |u|^2 u`. */
KOKKOS_INLINE_FUNCTION Complex dispersive_conserving_self(const Complex u) {
  const double weight = 3.0 * (u.real() * u.real() + u.imag() * u.imag());
  return {weight * u.real(), weight * u.imag()};
}
/** `u^2 f* + 2 |u|^2 f`. */
KOKKOS_INLINE_FUNCTION Complex dispersive_conserving_pair(const Complex u,
                                                          const Complex f) {
  const double modulus = u.real() * u.real() + u.imag() * u.imag();
  return u * u * Kokkos::conj(f) + (2.0 * modulus) * f;
}
/** `2 |f|^2 u + f^2 u*`. */
KOKKOS_INLINE_FUNCTION Complex dispersive_conserving_cross(const Complex u,
                                                           const Complex f) {
  const double modulus = f.real() * f.real() + f.imag() * f.imag();
  return (2.0 * modulus) * u + f * f * Kokkos::conj(u);
}
/** `c u^3 + c* 3 |u|^2 u*`. */
KOKKOS_INLINE_FUNCTION Complex dispersive_harmonic_self(const Complex u,
                                                        const Complex carrier) {
  const double weight = 3.0 * (u.real() * u.real() + u.imag() * u.imag());
  return carrier * (u * u * u) + Kokkos::conj(carrier) * (weight * Kokkos::conj(u));
}
/** `c u^2 f + c* (2 |u|^2 f* + u*^2 f)`. */
KOKKOS_INLINE_FUNCTION Complex dispersive_harmonic_pair(const Complex u,
                                                        const Complex f,
                                                        const Complex carrier) {
  const double modulus = u.real() * u.real() + u.imag() * u.imag();
  const Complex conjugate = Kokkos::conj(u);
  return carrier * (u * u * f) +
         Kokkos::conj(carrier) *
             ((2.0 * modulus) * Kokkos::conj(f) + conjugate * conjugate * f);
}
/** `c u f^2 + c* (u f*^2 + 2 |f|^2 u*)`. */
KOKKOS_INLINE_FUNCTION Complex dispersive_harmonic_cross(const Complex u,
                                                         const Complex f,
                                                         const Complex carrier) {
  const double modulus = f.real() * f.real() + f.imag() * f.imag();
  const Complex conjugate = Kokkos::conj(f);
  return carrier * (u * f * f) +
         Kokkos::conj(carrier) *
             (u * conjugate * conjugate + (2.0 * modulus) * Kokkos::conj(u));
}

/**
 * The isotropic contraction of three `Components`-component fields, componentwise.
 *
 * `[(x . y) z + (x . z) y + (y . z) x] / 3` without conjugation, each dot product
 * summed over the components in order from the first: solprop `dispersive_isotropic3`
 * for the general modal route's three Cartesian components, whose fewer components are
 * the three with the rest zero.
 */
template <class T, std::size_t Components>
KOKKOS_INLINE_FUNCTION void
dispersive_isotropic_n(const T (&x)[Components], const T (&y)[Components],
                       const T (&z)[Components], T (&out)[Components]) {
  T xy = x[0] * y[0];
  T xz = x[0] * z[0];
  T yz = y[0] * z[0];
  for (std::size_t c = 1; c < Components; ++c) {
    xy = xy + x[c] * y[c];
    xz = xz + x[c] * z[c];
    yz = yz + y[c] * z[c];
  }
  for (std::size_t c = 0; c < Components; ++c)
    out[c] = (xy * z[c] + xz * y[c] + yz * x[c]) / 3.0;
}

/** `T(x, y, z*) + T(x, y*, z) + T(x*, y, z)` of `Components`-component fields. */
template <std::size_t Components>
KOKKOS_INLINE_FUNCTION void
dispersive_conserving_n(const Complex (&x)[Components], const Complex (&y)[Components],
                        const Complex (&z)[Components], Complex (&out)[Components]) {
  Complex xc[Components], yc[Components], zc[Components];
  for (std::size_t c = 0; c < Components; ++c) {
    xc[c] = Kokkos::conj(x[c]);
    yc[c] = Kokkos::conj(y[c]);
    zc[c] = Kokkos::conj(z[c]);
  }
  Complex a[Components], b[Components], d[Components];
  dispersive_isotropic_n(x, y, zc, a);
  dispersive_isotropic_n(x, yc, z, b);
  dispersive_isotropic_n(xc, y, z, d);
  for (std::size_t c = 0; c < Components; ++c)
    out[c] = a[c] + b[c] + d[c];
}

/**
 * Write a symmetric mixture's filter rows at one stage.
 *
 * solprop `nonlinear.responses.mixture_filter`'s arithmetic, in its order (decided
 * 2026-10-08, SEP 6 Phase 2b open question 3): with `f_s = factors(members(m))` the
 * channels' `eps0 gamma3_0 rho` and `K_s = quartics(s, k) = (g_s^2)^2`,
 * `g_mix = sqrt(sqrt(sum_s f_s K_s / total))`, the species summed in member order
 * from zero; `inputs(k) = g_mix input_scale(k)` and `outputs(k) = g_mix
 * output_scale(k)`. At a vacuum (`total` not positive) the filter is one. `total` is
 * the members' factors summed in member order, which the caller forms on the host.
 * Allocates nothing.
 */
void dispersive_mixture_rows(
    const ExecutionSpace& space,
    const Kokkos::View<const double**, Layout, MemorySpace>& quartics,
    const Kokkos::View<const std::size_t*, Layout, MemorySpace>& members,
    const Kokkos::View<const double*, Layout, MemorySpace>& factors, double total,
    const Kokkos::View<const double*, Layout, MemorySpace>& input_scale,
    const Kokkos::View<const double*, Layout, MemorySpace>& output_scale,
    const Kokkos::View<double*, Layout, MemorySpace>& inputs,
    const Kokkos::View<double*, Layout, MemorySpace>& outputs);

/**
 * The response's layout and its weights at one stage, on every route.
 *
 * solprop `numba_cpu`'s `DispersiveTerms` and `StageWeights`: the input multipliers
 * (one row per filtered field), the output terms, the channels' coefficients and the
 * polynomial's pair constants, uploaded once; at a stage, `eps0 gamma0 rho` per
 * channel, each symmetric mixture's coefficient and filter rows, each polynomial
 * group's pair constants and one multiplier per product, rebuilt only when the
 * stage's densities differ from the last rebuild's (compared on the host). The
 * products are each symmetric group's, or for the polynomial one cube per distinct
 * group key (a general modal region; one key elsewhere) followed by each group's two
 * pairs. The factors cross through a ring of four host staging buffers, so a rebuild
 * allocates nothing and needs no fence.
 */
class DispersiveWeights {
public:
  using RealView2 = Kokkos::View<double**, Layout, MemorySpace>;
  using IndexView = Kokkos::View<std::size_t*, Layout, MemorySpace>;

  DispersiveWeights(const Problem& problem, const ExecutionSpace& kernels);
  ~DispersiveWeights() = default;
  DispersiveWeights(const DispersiveWeights&) = delete;
  DispersiveWeights& operator=(const DispersiveWeights&) = delete;
  DispersiveWeights(DispersiveWeights&&) = delete;
  DispersiveWeights& operator=(DispersiveWeights&&) = delete;

  /** Refuse a trial's densities that `set_densities` would not take. */
  void require_densities(const std::vector<double>& densities) const;
  /**
   * Keep a trial's channel densities, `(n_stages, n_channels)` row-major, m^-3.
   *
   * Copied into a host buffer held from construction; nothing is allocated.
   */
  void set_densities(const std::vector<double>& densities);

  [[nodiscard]] bool symmetric() const noexcept { return symmetric_; }
  [[nodiscard]] std::size_t bins() const noexcept { return bins_; }
  [[nodiscard]] std::size_t groups() const noexcept { return groups_; }
  [[nodiscard]] std::size_t fields() const noexcept { return fields_; }
  [[nodiscard]] std::size_t products() const noexcept { return products_; }
  [[nodiscard]] std::size_t cubes() const noexcept { return cubes_; }
  /** A group's key: its general modal region, zero elsewhere. */
  [[nodiscard]] std::size_t key(std::size_t group) const { return keys_.at(group); }
  /** A polynomial group's cube product. */
  [[nodiscard]] std::size_t cube(std::size_t group) const { return cube_of_.at(group); }
  /** The key of a cube product: the region its nodes are. */
  [[nodiscard]] std::size_t cube_key(std::size_t cube) const {
    return cube_keys_.at(cube);
  }
  /** A polynomial group's `c0` (`which` 0) and `c1` (1) at the last rebuild. */
  [[nodiscard]] double pair(std::size_t group, std::size_t which) const {
    return pairs_.at(2 * group + which);
  }
  /** The input multipliers at the stage, `(fields, bins)`. */
  [[nodiscard]] const RealView2& inputs() const noexcept { return inputs_; }
  /** One multiplier per product at the stage, `(products, bins)`. */
  [[nodiscard]] const RealView2& weights() const noexcept { return weights_; }
  /** The bytes of every table and staging buffer, the ledger's multipliers row. */
  [[nodiscard]] std::size_t bytes() const;
  void close() noexcept;

  // NVCC extended lambdas require their enclosing member functions to be public.
  /** Weigh the channels at a stage; a no-op when its densities are the last ones. */
  void update(std::size_t stage);

private:
  std::size_t channels_{}, bins_{}, groups_{}, fields_{}, products_{}, cubes_{};
  std::size_t mixtures_{}, n_stages_{};
  bool symmetric_{}, weighted_{};
  ExecutionSpace kernels_;
  RealView2 static_inputs_, inputs_, term_multipliers_, outputs_, weights_, quartics_;
  RealView input_scale_, output_scale_, factors_;
  IndexView term_rows_, term_channels_, members_;
  std::array<RealView::host_mirror_type, 4> host_factors_;
  std::size_t next_factors_{};
  std::vector<double> coefficients_, pair_constants_, pairs_, densities_, last_;
  std::vector<std::size_t> group_offsets_, group_members_, mixture_groups_;
  std::vector<std::size_t> keys_, cube_of_, cube_keys_;
};

/**
 * The response on the fused node path of one engine state: its buffers, plans and
 * per-stage passes.
 *
 * Every View and plan is allocated at construction (`plan` creates the plans); the
 * stage passes allocate, plan and log nothing. The engine's own buffers it reads or
 * borrows are handed over in `StageViews`.
 */
class DispersiveResponse {
public:
  using ComplexView2 = Kokkos::View<Complex**, Layout, MemorySpace>;
  using RealView2 = Kokkos::View<double**, Layout, MemorySpace>;

  /** The engine's views the response reads, writes or borrows. */
  struct StageViews {
    ComplexView2 fine_spectrum, projected_spectrum, projected_coarse_spectrum;
    RealView2 node_real, polarisation_real;
    ComplexView2 node_complex, polarisation_complex;
    RealView2 scaled_synthesis, scaled_projection, scaled_projection_coarse;
    RealView time_window;
    ComplexView carrier;
  };

  DispersiveResponse(const Problem& problem, CpuExecutionPool& pool, StageViews views);
  ~DispersiveResponse();
  DispersiveResponse(const DispersiveResponse&) = delete;
  DispersiveResponse& operator=(const DispersiveResponse&) = delete;
  DispersiveResponse(DispersiveResponse&&) = delete;
  DispersiveResponse& operator=(DispersiveResponse&&) = delete;

  /**
   * Create the persistent plans: a transform of fewer than `threshold` real-side
   * samples per execution on `serial`, as the engine's own small transforms, else on
   * `threaded`.
   */
  void plan(CpuExecutionPool& transforms);

  /** The layout and the stage's weights. */
  [[nodiscard]] DispersiveWeights& weights() noexcept { return weights_; }
  /** Whether no other response runs: no column, projection or stage transform. */
  [[nodiscard]] bool alone() const noexcept { return alone_; }
  /** Whether the stage's own inverse transform gives the filtered field. */
  [[nodiscard]] bool in_place() const noexcept { return in_place_; }
  /** Whether the product is the stage's polarisation, projected by the stage. */
  [[nodiscard]] bool borrowed() const noexcept { return borrowed_; }
  /** Whether the response writes the node buffer the stage would synthesise. */
  [[nodiscard]] bool synthesises_nodes() const noexcept { return lone_ && !in_place_; }
  /** Whether the response writes the stage's spectrum in place of its transform. */
  [[nodiscard]] bool replaces_transform() const noexcept {
    return alone_ && !borrowed_;
  }

  /**
   * Weigh the channels at a stage and filter the padded modal spectrum.
   *
   * Before the stage's inverse transform, which destroys the spectrum. The weights
   * and a mixture's filter are rebuilt only when the stage's densities differ from
   * the last rebuild's. Out of place, the filtered fields are transformed here.
   */
  void prepare(std::size_t stage);
  /** Pack the stage's scaled projections per chunk, after the engine scaled them. */
  void begin();
  /** Synthesise, form and project the products of one node chunk. */
  void respond(std::size_t chunk, std::size_t first_node, std::size_t node_count);
  /** Transform the fine rule's products, after the last chunk. */
  void finish();
  /**
   * Land the products' weighted spectra in the stage's polarisation spectrum.
   *
   * After the stage's forward transforms, or in their place when
   * `replaces_transform()`; a borrowed product's are weighed in place.
   */
  void combine();
  /**
   * Transform and land the embedded rule's products of the last evaluated stage.
   *
   * Once per evaluation, when its projection is scored; no effect otherwise.
   */
  void finish_coarse();

  /** Persistent allocations, by role, in bytes. */
  [[nodiscard]] std::map<std::string, std::size_t> memory_ledger() const;
  /** The same rows, all zero, for a problem without the response. */
  [[nodiscard]] static std::map<std::string, std::size_t> empty_ledger();
  /** Release plans before Views; repeated calls have no effect. */
  void close() noexcept;

  // NVCC extended lambdas require their enclosing member functions to be public.
  void synthesise(std::size_t field, std::size_t first_node, std::size_t node_count,
                  bool complex_fields, double* target);
  void project(std::size_t chunk, std::size_t node_count, std::size_t product,
               const double* source, bool first);
  void offsets(std::size_t first_node, std::size_t node_count);

private:
  using InverseRealPlan = KokkosFFT::Plan<ExecutionSpace, ComplexView2, RealView2>;
  using ForwardRealPlan = KokkosFFT::Plan<ExecutionSpace, RealView2, ComplexView2>;
  using ComplexPlan = KokkosFFT::Plan<ExecutionSpace, ComplexView2, ComplexView2>;

  std::size_t n_modes_{}, n_pol_state_{}, n_pol_transform_{}, components_{};
  std::size_t columns_{}, n_nodes_{}, node_chunk_{}, n_chunks_{}, n_to_{};
  std::size_t spectrum_bins_{};
  DispersiveForm form_{DispersiveForm::real};
  bool analytic_{}, embedded_{}, alone_{}, lone_{}, in_place_{};
  bool borrowed_{}, shifted_{}, coarse_pending_{}, closed_{};
  ExecutionSpace kernels_, response_, tail_response_, dense_, tail_dense_, matrix_copy_;
  std::size_t response_chunk_{};
  const ExecutionSpace& response_for(std::size_t count) const {
    return count == response_chunk_ ? response_ : tail_response_;
  }
  const ExecutionSpace& dense_for(std::size_t count) const {
    return count == response_chunk_ ? dense_ : tail_dense_;
  }
  StageViews stage_;
  DispersiveWeights weights_;
  // The response's own buffers; borrowed ones are the engine's rows.
  ComplexView2 filtered_spectrum_, spectra_, filtered_complex_, slots_complex_;
  ComplexView2 products_complex_, coarse_complex_;
  RealView2 filtered_real_, slots_real_, products_real_, coarse_real_;
  RealView packed_projection_, packed_coarse_;
  ComplexView dc_, offset_;
  std::unique_ptr<InverseRealPlan> inverse_real_;
  std::unique_ptr<ComplexPlan> inverse_complex_, forward_complex_, coarse_complex_plan_;
  std::unique_ptr<ForwardRealPlan> forward_real_, coarse_real_plan_;
};

/**
 * The response on the Cartesian free-space and line routes and the hybrid planar
 * route, chunk by chunk (solprop `numba_cpu`'s `SpatialDispersive`).
 *
 * Each chunk of the separable stage goes to time and back on its own: a free-space
 * node chunk's temporal rows are its nodes, a planar block's `(p, y)` rows are
 * synthesised at its `(x, y)` nodes by the real slab rule in x. Per chunk the response
 * filters the packed spectrum before the stage's inverse transform (`prepare`), forms
 * its products at the nodes under the collar after the stage put the bare field there,
 * projecting them back to the rows on the planar route (`respond`), transforms them
 * (`finish`) and lands their weighted spectra in the chunk's polarisation spectrum
 * after the stage's forward transform, or in its place when alone (`combine`).
 *
 * A lone symmetric response -- one product, alone -- always filters the chunk's
 * spectrum in place: the stage's in-stage absorber is a pass of its own on these
 * routes, which packs the chunk afresh, so nothing reads the bare field after the
 * material. Its product is then the stage's polarisation unless it is a field grid's
 * analytic signal. Free space forms each product over its own filtered field, which
 * the inverse transform writes straight into the product's block.
 */
class SeparableDispersive {
public:
  using ComplexView2 = Kokkos::View<Complex**, Layout, MemorySpace>;
  using RealView2 = Kokkos::View<double**, Layout, MemorySpace>;

  /** The engine's views the response reads, writes or borrows. */
  struct StageViews {
    ComplexView2 fine_spectrum, projected_spectrum;
    RealView2 node_real, polarisation_real;
    ComplexView2 node_complex, polarisation_complex;
    /** The planar kind's slab rule in x, `(n_x, n_p)` and `(n_p, n_x)`. */
    RealView2 synthesis, projection;
    RealView time_window;
    ComplexView carrier;
  };

  SeparableDispersive(const Problem& problem, CpuExecutionPool& pool, StageViews views);
  ~SeparableDispersive();
  SeparableDispersive(const SeparableDispersive&) = delete;
  SeparableDispersive& operator=(const SeparableDispersive&) = delete;
  SeparableDispersive(SeparableDispersive&&) = delete;
  SeparableDispersive& operator=(SeparableDispersive&&) = delete;

  /** Create the persistent plans, as `DispersiveResponse::plan`. */
  void plan(CpuExecutionPool& transforms);

  [[nodiscard]] DispersiveWeights& weights() noexcept { return weights_; }
  [[nodiscard]] bool alone() const noexcept { return alone_; }
  [[nodiscard]] bool in_place() const noexcept { return in_place_; }
  [[nodiscard]] bool borrowed() const noexcept { return borrowed_; }
  /** Whether the response writes the chunk's spectrum in place of its transform. */
  [[nodiscard]] bool replaces_transform() const noexcept {
    return alone_ && !borrowed_;
  }

  /** Weigh the channels at a stage, once for all its chunks. */
  void begin(std::size_t stage);
  /**
   * Filter the chunk's packed spectrum, before the stage's inverse transform; out of
   * place, transform the filtered fields.
   */
  void prepare(std::size_t node_count);
  /**
   * Form the chunk's products at its `node_count` nodes, the bare field (or, in
   * place, the filtered one) in the stage's node buffer; on the planar route project
   * them onto its rows. A borrowed product is written to the stage's polarisation.
   */
  void respond(std::size_t node_count);
  /** Transform the chunk's products. */
  void finish();
  /** Land or weigh the products' spectra in the chunk's polarisation spectrum. */
  void combine();

  [[nodiscard]] std::map<std::string, std::size_t> memory_ledger() const;
  void close() noexcept;

  // NVCC extended lambdas require their enclosing member functions to be public.
  void slab_offsets(std::size_t units);

private:
  using InverseRealPlan = KokkosFFT::Plan<ExecutionSpace, ComplexView2, RealView2>;
  using ForwardRealPlan = KokkosFFT::Plan<ExecutionSpace, RealView2, ComplexView2>;
  using ComplexPlan = KokkosFFT::Plan<ExecutionSpace, ComplexView2, ComplexView2>;

  std::size_t rows_{}, node_rows_{}, components_{}, n_to_{}, spectrum_bins_{};
  std::size_t n_x_{}, n_p_{};
  DispersiveForm form_{DispersiveForm::real};
  bool slab_{}, analytic_{}, alone_{}, lone_{}, in_place_{}, borrowed_{}, shifted_{};
  bool closed_{};
  ExecutionSpace kernels_, tail_kernels_, response_, tail_response_, dense_,
      tail_dense_;
  std::size_t tail_rows_{}, active_rows_{};
  bool active_tail_{};
  std::size_t response_chunk_{};
  const ExecutionSpace& response_for(std::size_t count) const {
    return count == response_chunk_ ? response_ : tail_response_;
  }
  const ExecutionSpace& dense_for(std::size_t count) const {
    return count == response_chunk_ ? dense_ : tail_dense_;
  }
  StageViews stage_;
  DispersiveWeights weights_;
  ComplexView2 filtered_spectrum_, spectra_, filtered_complex_, slots_complex_;
  ComplexView2 products_complex_;
  RealView2 filtered_real_, slots_real_, products_real_;
  ComplexView dc_, offset_;
  std::unique_ptr<InverseRealPlan> inverse_real_;
  std::unique_ptr<ComplexPlan> inverse_complex_, forward_complex_;
  std::unique_ptr<ForwardRealPlan> forward_real_;
  TailRowPlans<ExecutionSpace, ComplexView2, RealView2> tail_inverse_real_;
  TailRowPlans<ExecutionSpace, ComplexView2, ComplexView2> tail_inverse_complex_,
      tail_forward_complex_;
  TailRowPlans<ExecutionSpace, RealView2, ComplexView2> tail_forward_real_;
};

/**
 * The response on the general modal route (solprop `numba_cpu`'s
 * `GeneralDispersive`).
 *
 * The route synthesises each chunk's Cartesian components in frequency through maps
 * that depend on frequency, so the input multipliers act in synthesis: on the chunk's
 * synthesised spectrum before its inverse transform (`prepare`), one batched inverse
 * transform giving every filtered analytic signal. The products are formed at the
 * nodes of each channel's own region -- a gas region's fill -- and zero at the others,
 * on the real field (the full sector, beside the route's real cubic) or on the
 * analytic signals (the number-conserving sector), contracted through the isotropic
 * Kleinman tensor (SEP 6, decision Q3); one batched forward transform (`respond`), and
 * their weighted spectra land in the chunk's polarisation spectrum after the route's
 * own forward transform, or in its place when no regional tensor or delayed response
 * makes one (`combine`). The route has no collar. The route's synthesis keeps its
 * inverse transform's `1 / n_to` for the cubic, so the response applies it to what it
 * reads: a power of two, which moves no bit.
 */
class GeneralDispersive {
public:
  using ComplexView2 = Kokkos::View<Complex**, Layout, MemorySpace>;
  using RealView2 = Kokkos::View<double**, Layout, MemorySpace>;
  using IndexView = Kokkos::View<std::size_t*, Layout, MemorySpace>;

  /** The route's chunk buffers: the synthesised spectrum, its signal, the source. */
  struct StageViews {
    ComplexView2 spectrum, field, source;
  };

  GeneralDispersive(const Problem& problem, CpuExecutionPool& pool, StageViews views);
  ~GeneralDispersive();
  GeneralDispersive(const GeneralDispersive&) = delete;
  GeneralDispersive& operator=(const GeneralDispersive&) = delete;
  GeneralDispersive(GeneralDispersive&&) = delete;
  GeneralDispersive& operator=(GeneralDispersive&&) = delete;

  /** Create the persistent plans on the route's transform space. */
  void plan(CpuExecutionPool& transforms);

  [[nodiscard]] DispersiveWeights& weights() noexcept { return weights_; }
  /** Whether the polynomial reads the bare signal, which the route transforms. */
  [[nodiscard]] bool reads_bare() const noexcept { return !weights_.symmetric(); }
  /** Whether a chunk has a node of any channel's region. */
  [[nodiscard]] bool live(std::size_t chunk) const { return live_.at(chunk) != 0U; }

  /** Weigh the channels at a stage, once for all its chunks. */
  void begin(std::size_t stage);
  /** Filter a live chunk's synthesised spectrum and transform the filtered fields. */
  void prepare(std::size_t chunk);
  /** Form and transform a live chunk's products after the route's inverse transform. */
  void respond(std::size_t chunk, std::size_t first, std::size_t count);
  /** Land the products' spectra in the chunk's source: written, or added to. */
  void combine(bool overwrite);

  [[nodiscard]] std::map<std::string, std::size_t> memory_ledger() const;
  void close() noexcept;

  // NVCC extended lambdas require their enclosing member functions to be public.
  template <bool Real, std::size_t Components>
  void products(std::size_t first, std::size_t count);

private:
  using ComplexPlan = KokkosFFT::Plan<ExecutionSpace, ComplexView2, ComplexView2>;
  using ForwardRealPlan = KokkosFFT::Plan<ExecutionSpace, RealView2, ComplexView2>;

  std::size_t rows_{}, components_{}, n_to_{}, spectrum_bins_{};
  bool real_{}, shifted_{}, closed_{};
  ExecutionSpace kernels_, tail_kernels_, response_, tail_response_;
  std::size_t tail_rows_{}, active_rows_{}, last_chunk_{};
  bool active_tail_{};
  std::size_t response_chunk_{};
  const ExecutionSpace& response_for(std::size_t count) const {
    return count == response_chunk_ ? response_ : tail_response_;
  }
  StageViews stage_;
  DispersiveWeights weights_;
  IndexView node_region_;
  std::vector<unsigned char> live_;
  ComplexView2 filtered_spectrum_, filtered_, products_complex_, spectra_;
  RealView2 products_real_;
  ComplexView dc_;
  std::unique_ptr<ComplexPlan> inverse_, forward_complex_;
  std::unique_ptr<ForwardRealPlan> forward_real_;
  TailRowPlans<ExecutionSpace, ComplexView2, ComplexView2> tail_inverse_,
      tail_forward_complex_;
  TailRowPlans<ExecutionSpace, RealView2, ComplexView2> tail_forward_real_;
};

} // namespace solprop_native
