// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "bindings_transverse.hpp"

#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace solprop_native::bindings {
namespace {

namespace nb = nanobind;
using namespace nb::literals;

using RealArray = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using IndexArray =
    nb::ndarray<const std::uint64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

std::vector<double> reals(const nb::handle value) {
  const auto array = nb::cast<RealArray>(value, false);
  return {array.data(), array.data() + array.shape(0)};
}

std::array<double, 4> four(const nb::handle value) {
  const auto values = reals(value);
  if (values.size() != 4)
    throw std::invalid_argument("a transverse reading's region sums are four");
  return {values[0], values[1], values[2], values[3]};
}

/** A dictionary's entry that may be absent or `None`. */
std::optional<nb::handle> present(const nb::dict values, const char* name) {
  if (!values.contains(name))
    return std::nullopt;
  nb::handle value = values[name];
  if (value.is_none())
    return std::nullopt;
  return value;
}

transverse::Reading reading_from(const nb::dict values) {
  transverse::Reading reading;
  try {
    if (const auto faces = present(values, "faces")) {
      reading.faces = four(*faces);
      const auto columns = present(values, "columns");
      if (!columns)
        throw std::invalid_argument("a transverse reading's faces come with columns");
      reading.columns = reals(*columns);
    }
    if (const auto clip = present(values, "clip"))
      reading.clip = four(*clip);
    if (const auto totals = present(values, "totals"))
      reading.totals = four(*totals);
    if (const auto strip = present(values, "strip"))
      reading.strip = reals(*strip);
    if (const auto current = present(values, "current"))
      reading.current = nb::cast<double>(*current);
  } catch (const nb::cast_error&) {
    throw std::invalid_argument("a transverse reading has the wrong types");
  }
  return reading;
}

nb::tuple pair(const double first, const double second) {
  return nb::make_tuple(first, second);
}

} // namespace

transverse::Noise transverse_noise_from(const nb::dict values) {
  transverse::Noise noise;
  try {
    noise.columns = reals(values["columns"]);
    noise.energy = nb::cast<double>(values["energy"]);
    noise.ramp = nb::cast<double>(values["ramp"]);
    noise.strip = reals(values["strip"]);
    if (const auto clean = present(values, "clean"))
      noise.clean = reals(*clean);
  } catch (const nb::cast_error&) {
    throw std::invalid_argument("a transverse launch's noise has the wrong types");
  } catch (const nb::python_error&) {
    throw std::invalid_argument("a transverse launch's noise lacks a field");
  }
  return noise;
}

transverse::Config transverse_config_from(const nb::dict values) {
  transverse::Config config;
  try {
    config.faces = nb::cast<std::vector<std::string>>(values["faces"]);
    config.warn = nb::cast<double>(values["warn"]);
    config.stop = nb::cast<double>(values["stop"]);
    config.length = nb::cast<double>(values["length"]);
    config.guard = nb::cast<double>(values["guard"]);
    config.k_min = nb::cast<double>(values["k_min"]);
    config.k_floor = nb::cast<double>(values["k_floor"]);
    config.reads = nb::cast<double>(values["reads"]);
    config.backstop_reads = nb::cast<double>(values["backstop_reads"]);
    config.reach_growth = nb::cast<double>(values["reach_growth"]);
    config.noise_factor = nb::cast<double>(values["noise_factor"]);
    config.columns = reals(values["columns"]);
    config.kappa = reals(values["kappa"]);
    const auto order = nb::cast<IndexArray>(values["kappa_order"], false);
    config.kappa_order.reserve(order.shape(0));
    for (std::size_t index = 0; index < order.shape(0); ++index)
      config.kappa_order.push_back(static_cast<std::size_t>(order.data()[index]));
    config.design = nb::cast<bool>(values["design"]);
    config.ramp_rate = reals(values["ramp_rate"]);
    config.collar = nb::cast<bool>(values["collar"]);
    config.collar_rate = nb::cast<bool>(values["collar_rate"]);
    config.declared = nb::cast<bool>(values["declared"]);
    config.reads_edge = nb::cast<bool>(values["reads_edge"]);
    if (const auto longest = present(values, "longest_step"))
      config.longest_step = nb::cast<double>(*longest);
    config.noise_share = nb::cast<double>(values["noise_share"]);
    config.noise_modes = nb::cast<double>(values["noise_modes"]);
  } catch (const nb::cast_error&) {
    throw std::invalid_argument(
        "the transverse edge's configuration has the wrong types");
  } catch (const nb::python_error&) {
    throw std::invalid_argument("the transverse edge's configuration lacks a field");
  }
  return config;
}

transverse::Resumption transverse_resumption_from(const nb::dict values) {
  transverse::Resumption record;
  const auto number = [&values](const char* name) -> std::optional<double> {
    if (const auto value = present(values, name))
      return nb::cast<double>(*value);
    return std::nullopt;
  };
  const auto list = [&values](const char* name) -> std::optional<std::vector<double>> {
    if (const auto value = present(values, name))
      return nb::cast<std::vector<double>>(*value);
    return std::nullopt;
  };
  try {
    record.record = nb::cast<bool>(values["record"]);
    record.launch_energy = number("launch_energy");
    record.edge_baseline = list("edge_baseline");
    record.edge_max = list("edge_max");
    record.edge_max_raw = list("edge_max_raw");
    record.noise_allowance = number("noise_allowance");
    record.raw_ramp_removed = number("raw_ramp_removed");
    record.noise_ramp_removed = number("noise_ramp_removed");
    record.noise_share_in_ramp = number("noise_share_in_ramp");
    record.declared_share_in_ramp = number("declared_share_in_ramp");
    record.spatial_removed = number("spatial_removed");
    record.current = number("current");
    record.current_baseline = number("current_baseline");
    record.strip_baseline = number("strip_baseline");
    record.strip_floor = number("strip_floor");
    record.strip_max = number("strip_max");
    record.returned_max = number("returned_max");
    if (const auto reads = present(values, "reads"))
      record.reads = nb::cast<std::int64_t>(*reads);
    record.largest_gap = number("largest_gap");
    record.warned_at = number("warned_at");
    if (const auto status = present(values, "transverse_status"))
      record.status = nb::cast<std::string>(*status);
  } catch (const nb::cast_error&) {
    throw std::invalid_argument("a resumed transverse record has the wrong types");
  } catch (const nb::python_error&) {
    throw std::invalid_argument("a resumed transverse record lacks a field");
  }
  return record;
}

nb::dict transverse_record_to(const transverse::Record& record,
                              const transverse::Finish& finish) {
  nb::dict out;
  out["launch_energy"] = record.launch_energy;
  out["noise_ramp_removed"] = record.noise_ramp_removed;
  out["noise_share_in_ramp"] = record.noise_share_in_ramp;
  out["declared_share_in_ramp"] = record.declared_share_in_ramp;
  out["transverse_status"] = static_cast<int>(record.status);
  out["transverse_reason"] = static_cast<int>(record.reason);
  out["edge_baseline"] = record.edge_baseline;
  out["edge_max"] = record.edge_max;
  out["edge_max_raw"] = record.edge_max_raw;
  out["noise_allowance"] = record.noise_allowance;
  out["ramp_removed"] = record.ramp_removed;
  out["raw_ramp_removed"] = record.raw_ramp_removed;
  out["spatial_removed"] = record.spatial_removed;
  out["reads"] = record.reads;
  out["returned_max"] = record.returned_max;
  out["strip_max"] = record.strip_max;
  out["strip_baseline"] = record.strip_baseline;
  out["strip_floor"] = record.strip_floor;
  out["current"] = record.current;
  out["current_baseline"] = record.current_baseline;
  out["largest_gap"] = record.largest_gap;
  out["warned_at"] = record.warned_at;
  out["resumed"] = record.resumed;
  nb::dict end;
  end["status"] = static_cast<int>(finish.status);
  end["largest"] = finish.largest;
  end["ramp_removed"] = finish.ramp_removed;
  end["reads"] = finish.reads;
  out["finish"] = end;
  return out;
}

nb::dict transverse_notice_to(const transverse::Notice& notice) {
  nb::dict out;
  out["kind"] = static_cast<int>(notice.kind);
  out["stop"] = notice.stop;
  out["face"] = notice.face;
  out["value"] = notice.value;
  out["z"] = notice.z;
  out["spacing"] = notice.spacing;
  out["warned_at"] = notice.warned_at;
  return out;
}

void bind_transverse(nb::module_& module_) {
  // solprop SEP 47 section 6.4: the transverse edge a native session owns, for the
  // recorded-readings harness that holds it to solprop's `TransverseEdge`.
  nb::class_<transverse::Edge>(module_, "TransverseEdge")
      .def(
          "__init__",
          [](transverse::Edge* self, const nb::dict config) {
            new (self) transverse::Edge(transverse_config_from(config));
          },
          "config"_a)
      .def(
          "resume",
          [](transverse::Edge& self, const nb::dict record) {
            self.resume(transverse_resumption_from(record));
          },
          "record"_a)
      .def(
          "launch",
          [](transverse::Edge& self, const nb::dict reading, const double z,
             const std::optional<std::pair<double, double>> sums,
             const std::optional<nb::dict>& noise) {
            std::optional<transverse::LaunchSums> held;
            if (sums)
              held = transverse::LaunchSums{.total = sums->first, .ramp = sums->second};
            std::optional<transverse::Noise> carried;
            if (noise)
              carried = transverse_noise_from(*noise);
            return transverse_notice_to(self.launch(reading_from(reading), z,
                                                    held ? &*held : nullptr,
                                                    carried ? &*carried : nullptr));
          },
          "reading"_a, "z"_a, "sums"_a = nb::none(), "noise"_a = nb::none())
      .def(
          "read",
          [](transverse::Edge& self, const nb::dict reading, const double z,
             const std::optional<std::pair<double, double>> sums) {
            std::optional<transverse::ReadSums> held;
            if (sums)
              held = transverse::ReadSums{.raw = sums->first, .noise = sums->second};
            return transverse_notice_to(
                self.read(reading_from(reading), z, held ? &*held : nullptr));
          },
          "reading"_a, "z"_a, "sums"_a = nb::none())
      .def("proceed",
           [](transverse::Edge& self) { return transverse_notice_to(self.proceed()); })
      .def("step", &transverse::Edge::step, "current"_a, "z"_a)
      .def("ceiling", &transverse::Edge::ceiling)
      .def("due", &transverse::Edge::due, "z"_a, "proposal"_a)
      .def("wants_read", &transverse::Edge::wants_read, "z"_a)
      .def(
          "launch_sums",
          [](const transverse::Edge& self, const RealArray columns) {
            const auto sums = self.launch_sums(
                std::vector<double>(columns.data(), columns.data() + columns.shape(0)));
            return pair(sums.total, sums.ramp);
          },
          "columns"_a.noconvert())
      .def(
          "read_sums",
          [](const transverse::Edge& self, const double gap) {
            const auto sums = self.read_sums(gap);
            return pair(sums.raw, sums.noise);
          },
          "gap"_a)
      .def("record",
           [](const transverse::Edge& self) {
             return transverse_record_to(self.record(), self.finish());
           })
      .def_prop_ro("next_read", &transverse::Edge::next_read)
      .def_prop_ro("every_step", &transverse::Edge::every_step)
      .def_prop_ro("resumed", &transverse::Edge::resumed)
      .def_prop_ro("pending", &transverse::Edge::pending)
      .def_prop_ro("status", [](const transverse::Edge& self) {
        return static_cast<int>(self.status());
      });
}

} // namespace solprop_native::bindings
