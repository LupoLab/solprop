// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#include "helpers.hpp"
#include "solprop_cpunative/beam.hpp"
#include "solprop_cpunative/core.hpp"
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <omp.h>
#include <type_traits>
#if defined(CPUNATIVE_MKL_FFT) || defined(CPUNATIVE_MKL_BLAS)
#include <mkl_service.h>
#endif

namespace nb = nanobind;
using namespace solprop_cpunative;
template <class T>
using Array = nb::ndarray<nb::numpy, const T, nb::c_contig, nb::device::cpu>;
template <class T> std::vector<T> values(nb::handle h) {
  auto a = nb::cast<Array<T>>(h);
  return {a.data(), a.data() + a.size()};
}
template <class T> std::vector<T> values(const nb::dict& d, const char* key) {
  return values<T>(d[key]);
}
template <class T> T scalar(const nb::dict& d, const char* key) {
  return nb::cast<T>(d[key]);
}
template <class T> auto array(std::span<const T> source, int threads = 1) {
  static_assert(std::is_trivially_copyable_v<T>);
  // The observation owns its storage. Avoid serial zero-initialisation of a
  // complex volume before overwriting it, and first-touch large copies in parallel.
  std::unique_ptr<void, decltype(&std::free)> storage(
      std::malloc(std::max(std::size_t(1), source.size()) * sizeof(T)), &std::free);
  if (!storage)
    throw std::bad_alloc();
  auto* destination = static_cast<T*>(storage.get());
  {
    nb::gil_scoped_release release;
    if (threads > 1 && source.size() >= 262144) {
#pragma omp parallel for schedule(static) num_threads(threads)
      for (int part = 0; part < threads; ++part) {
        const auto first = source.size() * std::size_t(part) / std::size_t(threads);
        const auto last = source.size() * std::size_t(part + 1) / std::size_t(threads);
        std::memcpy(destination + first, source.data() + first,
                    (last - first) * sizeof(T));
      }
    } else if (!source.empty()) {
      std::memcpy(destination, source.data(), source.size_bytes());
    }
  }
  nb::capsule owner(storage.get(), [](void* ptr) noexcept { std::free(ptr); });
  storage.release();
  return nb::ndarray<nb::numpy, T>(destination, {source.size()}, owner);
}
template <class T> auto array(const std::vector<T>& source) {
  return array<T>(std::span<const T>(source));
}

Stage stage(const nb::dict& d) {
  Stage s;
  s.delta = values<double>(d, "delta");
  s.prefactor = values<double>(d, "prefactor");
  s.raman_rates = values<double>(d, "raman_rates");
  s.kerr_density = scalar<double>(d, "kerr_density");
  s.plasma_density = scalar<double>(d, "plasma_density");
  s.preionisation = scalar<double>(d, "preionisation");
  s.raman_density = scalar<double>(d, "raman_density");
  s.taper = scalar<double>(d, "taper");
  return s;
}
Config config(const nb::dict& d) {
  Config c;
#define INT(key) c.key = scalar<int>(d, #key)
  INT(modes);
  INT(nodes);
  INT(chunk);
  INT(nt);
  INT(nf);
  INT(nw);
  INT(threads);
  INT(slots);
#undef INT
#define BOOL(key) c.key = scalar<bool>(d, #key)
  BOOL(plasma);
  BOOL(band);
  BOOL(split);
  BOOL(profile);
  BOOL(analytic_kerr);
#undef BOOL
#define REAL(key) c.key = scalar<double>(d, #key)
  REAL(dt);
  REAL(kerr);
  REAL(charge);
  REAL(potential);
  REAL(field_min);
  REAL(field_max);
  REAL(rate_max);
  REAL(log_min);
  REAL(log_step);
  REAL(raman_fraction);
  REAL(epsilon0);
  REAL(rtol);
  REAL(atol);
#undef REAL
#define VECTOR(key) c.key = values<double>(d, #key)
  VECTOR(synthesis);
  VECTOR(projection);
  VECTOR(coarse);
  VECTOR(omega);
  VECTOR(window);
  VECTOR(window_derivative);
  VECTOR(omega_window);
  VECTOR(absorber);
  VECTOR(absorber_coarse);
  VECTOR(taps);
  VECTOR(rate);
  VECTOR(undamped);
  VECTOR(tableau);
  VECTOR(solution);
  VECTOR(error);
  VECTOR(spectral_weights);
#undef VECTOR
  c.phi = values<Complex>(d, "phi");
  c.prefactor = values<Complex>(d, "prefactor");
  c.admittance = values<Complex>(d, "admittance");
  c.spatial_axis = values<Complex>(d, "spatial_axis");
  c.crop = values<std::size_t>(d, "crop");
  c.active = values<unsigned char>(d, "active");
  c.norm = values<unsigned char>(d, "norm");
  c.planning = scalar<std::string>(d, "planning");
  c.contraction = scalar<std::string>(d, "contraction");
  return c;
}
AdamsTables tables(const nb::dict& d) {
  AdamsTables t;
  t.order = scalar<int>(d, "order");
  t.width = scalar<int>(d, "width");
  t.count = scalar<int>(d, "count");
  t.rows = scalar<int>(d, "rows");
  t.slots = values<std::size_t>(d, "slots");
#define VECTOR(key) t.key = values<double>(d, #key)
  VECTOR(predictor);
  VECTOR(corrector);
  VECTOR(defects);
  VECTOR(spans);
  VECTOR(carried);
  VECTOR(endpoint);
  VECTOR(comp_defects);
  VECTOR(rate);
#undef VECTOR
  return t;
}

NB_MODULE(_core, m) {
  // LLVM/Intel may defer the importing thread's binding until the first runtime
  // query. Complete that initialisation before Python records the post-load mask.
  // Otherwise later provenance queries can make a core place look like the grant.
  static_cast<void>(omp_get_max_threads());
  static_cast<void>(omp_get_num_places());
  static_cast<void>(omp_get_place_num());
  solprop_cpunative::bindings::bind_helpers(m);
  nb::class_<BeamState>(m, "BeamState")
      .def("__init__",
           [](BeamState* self, const nb::dict& d) {
             BeamConfig c;
#define GET(type, key) c.key = scalar<type>(d, #key)
             GET(int, nx);
             GET(int, ny);
             GET(int, nt);
             GET(int, threads);
             GET(double, dt);
             GET(double, kerr);
             GET(double, light_speed);
             GET(double, evanescent_rate);
             GET(bool, split);
             GET(bool, profile);
             GET(std::string, planning);
             GET(std::string, fft_schedule);
#undef GET
#define VECTOR(type, key) c.key = values<type>(d, #key)
             VECTOR(Complex, k_squared);
             VECTOR(Complex, k_squared_low);
             VECTOR(Complex, numerator);
             VECTOR(Complex, clamped);
             VECTOR(Complex, axis);
             VECTOR(double, kperp);
             VECTOR(double, kperp_low);
             VECTOR(double, frame);
             VECTOR(double, spectral_rate);
             VECTOR(double, transverse_rate);
             VECTOR(double, omega);
             VECTOR(double, window);
             VECTOR(double, omega_window);
             VECTOR(double, absorber);
             VECTOR(double, spectral_weights);
             VECTOR(unsigned char, active);
             VECTOR(unsigned char, norm);
#undef VECTOR
             new (self) BeamState(std::move(c));
           })
      .def("load",
           [](BeamState& s, Array<Complex> a) {
             nb::gil_scoped_release release;
             s.load({a.data(), a.size()});
           })
      .def("field", [](BeamState& s) { return array(s.field(), s.threads()); })
      .def("evaluate",
           [](BeamState& s, double density) {
             std::vector<Complex> result;
             {
               nb::gil_scoped_release release;
               result = s.evaluate(density);
             }
             return array(result);
           })
      .def("step",
           [](BeamState& s, double dz, double key, Array<double> density) {
             nb::gil_scoped_release release;
             return s.step(dz, key, {density.data(), density.size()});
           })
      .def("finish", &BeamState::finish, nb::call_guard<nb::gil_scoped_release>())
      .def("statistics",
           [](BeamState& s, double factor) {
             std::vector<double> result;
             {
               nb::gil_scoped_release release;
               result = s.statistics(factor);
             }
             return array(result);
           })
      .def_prop_ro("memory_bytes", &BeamState::memory_bytes)
      .def_prop_ro("threads", &BeamState::threads)
      .def_prop_ro("fft_schedule", &BeamState::fft_schedule)
      .def_prop_ro("evaluations", &BeamState::evaluations)
      .def_prop_ro("timings", &BeamState::timings)
      .def_prop_ro("planning_seconds", &BeamState::planning_seconds);
  nb::class_<State>(m, "State")
      .def("__init__",
           [](State* self, const nb::dict& d) { new (self) State(config(d)); })
      .def("load",
           [](State& s, Array<Complex> a) {
             nb::gil_scoped_release release;
             s.load({a.data(), a.size()});
           })
      .def("field", [](State& s) { return array(s.field(), s.threads()); })
      .def("_temporal",
           [](State& s) {
             std::vector<double> result;
             {
               nb::gil_scoped_release release;
               result = s.temporal_probe();
             }
             return array(result);
           })
      .def("_roundtrip",
           [](State& s) {
             std::vector<Complex> result;
             {
               nb::gil_scoped_release release;
               result = s.roundtrip_probe();
             }
             return array(result);
           })
      .def("_response",
           [](State& s, Array<double> field, const nb::dict& d) {
             auto st = stage(d);
             std::vector<double> result;
             {
               nb::gil_scoped_release release;
               result = s.response_probe({field.data(), field.size()}, st);
             }
             return array(result);
           })
      .def("evaluate",
           [](State& s, const nb::dict& d) {
             auto st = stage(d);
             std::vector<Complex> out;
             {
               nb::gil_scoped_release release;
               out = s.evaluate(st);
             }
             return array(out);
           })
      .def("trial",
           [](State& s, double dz, const nb::list& rows, Array<double> whole) {
             std::vector<Stage> stages;
             for (auto row : rows)
               stages.push_back(stage(nb::cast<nb::dict>(row)));
             std::vector<double> w(whole.data(), whole.data() + whole.size());
             nb::gil_scoped_release release;
             return s.trial(dz, stages, w);
           })
      .def("accept", &State::accept, nb::call_guard<nb::gil_scoped_release>())
      .def("clear_fsal", &State::clear_fsal)
      .def("adams_start",
           [](State& s, const nb::dict& d, Array<double> rate) {
             auto st = stage(d);
             std::vector<double> r(rate.data(), rate.data() + rate.size());
             nb::gil_scoped_release release;
             s.adams_start(st, r);
           })
      .def("adams_trial",
           [](State& s, double dz, const nb::dict& step, const nb::dict& sample,
              const nb::dict& table) {
             auto st = stage(step), sa = stage(sample);
             auto t = tables(table);
             std::vector<double> out;
             {
               nb::gil_scoped_release release;
               out = s.adams_trial(dz, st, sa, t);
             }
             return array(out);
           })
      .def("adams_accept",
           [](State& s, double dz, std::size_t slot, Array<double> rate) {
             std::vector<double> r(rate.data(), rate.data() + rate.size());
             nb::gil_scoped_release release;
             s.adams_accept(dz, slot, r);
           })
      .def("history",
           [](State& s, const std::string& name) { return array(s.history(name)); })
      .def("load_history",
           [](State& s, const std::string& name, Array<Complex> values) {
             nb::gil_scoped_release release;
             s.load_history(name, {values.data(), values.size()});
           })
      .def("populations",
           [](State& s, const nb::dict& d) {
             auto st = stage(d);
             std::vector<double> out;
             {
               nb::gil_scoped_release release;
               out = s.populations(st);
             }
             return array(out);
           })
      .def("diagnostics", &State::diagnostics)
      .def("statistics",
           [](State& s, Array<double> weight, Array<double> axis, double factor) {
             std::vector<double> result;
             {
               nb::gil_scoped_release release;
               result = s.statistics({weight.data(), weight.size()},
                                     {axis.data(), axis.size()}, factor);
             }
             return array(result);
           })
      .def("spatial_statistics",
           [](State& s, double factor) {
             std::vector<double> result;
             {
               nb::gil_scoped_release release;
               result = s.spatial_statistics(factor);
             }
             return array(result);
           })
      .def_prop_ro("timings", &State::timings)
      .def_prop_ro("response_timings", &State::response_timings)
      .def_prop_ro("evaluations", &State::evaluations)
      .def_prop_ro("quadrature", &State::quadrature)
      .def("reset_quadrature", &State::reset_quadrature)
      .def_prop_ro("memory_bytes", &State::memory_bytes)
      .def_prop_ro("planning_seconds", &State::planning_seconds)
      .def_prop_ro("contraction", &State::contraction)
      .def_prop_ro("threads", &State::threads);
  m.def("build_info", [] {
    nb::dict d;
    d["compiler"] = CPUNATIVE_COMPILER;
    d["profile"] = CPUNATIVE_PROFILE;
    d["math_profile"] = CPUNATIVE_MATH_PROFILE;
    d["rate_math"] = CPUNATIVE_RATE_MATH;
    d["fftw_threading"] = CPUNATIVE_FFTW_THREADING;
    d["target"] = CPUNATIVE_TARGET;
    d["commit"] = CPUNATIVE_COMMIT;
    d["source_hash"] = CPUNATIVE_SOURCE_HASH;
    d["ipo"] = bool(CPUNATIVE_IPO);
    d["fast_math"] = false;
    d["fft_policy"] = "OpenMP-distributed serial FFTW rows";
#ifdef CPUNATIVE_BLAS
    d["blas_available"] = true;
#else
    d["blas_available"] = false;
#endif
#ifdef CPUNATIVE_MKL_FFT
    char version[256]{};
    mkl_get_version_string(version, sizeof(version));
    d["fftw"] = std::string(version);
    d["fft_provider"] = "MKL FFTW3 interface";
#else
    d["fftw"] = std::string(fftw_version);
    d["fft_provider"] = "FFTW";
#endif
#ifdef CPUNATIVE_MKL_BLAS
    d["blas_provider"] = "MKL";
#else
    d["blas_provider"] = "platform";
#endif
    d["openmp"] = _OPENMP;
    d["precision"] = "double";
    d["schema_version"] = 37;
    d["cxx_standard"] = 23;
    Dl_info fft_info{}, omp_info{}, fft_thread_info{};
    if (dladdr(reinterpret_cast<void*>(&fftw_execute), &fft_info))
      d["fft_library"] = std::string(fft_info.dli_fname);
    if (dladdr(reinterpret_cast<void*>(&omp_get_max_threads), &omp_info))
      d["openmp_library"] = std::string(omp_info.dli_fname);
    if (dladdr(reinterpret_cast<void*>(&fftw_init_threads), &fft_thread_info))
      d["fft_thread_library"] = std::string(fft_thread_info.dli_fname);
    return d;
  });
}
