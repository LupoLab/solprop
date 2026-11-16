// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "bind.hpp"
#include "records.hpp"
#include "solprop_native/problem/problem.hpp"
#include "solprop_native/problem/workspace.hpp"
#include "solprop_native/session/statistics_plan.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

/*
 * The classes module.cpp binds that the private `_testing` entry points (testing.cpp)
 * also take: the workspace and the statistics plan. A session, a probe and a host
 * source are module.cpp's alone, so the workspace members that make them are defined
 * there.
 */
namespace solprop_native::bindings {

class BoundSession;
class BoundProbe;

// A statistics evaluation lowered once from the problem (solprop SEP 47 section 6.4),
// read by its constructor (module.cpp).
class BoundStatisticsPlan {
public:
  explicit BoundStatisticsPlan(nb::dict values);
  [[nodiscard]] const std::shared_ptr<const native::statistics::Plan>& plan() const {
    return plan_;
  }
  [[nodiscard]] std::size_t packed_size() const { return plan_->packed_size(); }

private:
  std::shared_ptr<const native::statistics::Plan> plan_;
};

// solprop SEP 52 section 5.3: one problem's workspace, its set-once data applied when
// it is built, reused across the runs of a scan.
class BoundWorkspace final : public Owned {
public:
  explicit BoundWorkspace(native::Problem problem)
      : raman_terms_(problem.raman.n_terms), raman_(problem.raman.enabled),
        workspace_(std::make_unique<native::Workspace>(std::move(problem))) {
    register_owned(this);
  }
  ~BoundWorkspace() override {
    close();
    forget_owned(this);
  }
  BoundWorkspace(const BoundWorkspace&) = delete;
  BoundWorkspace& operator=(const BoundWorkspace&) = delete;
  BoundWorkspace(BoundWorkspace&&) = delete;
  BoundWorkspace& operator=(BoundWorkspace&&) = delete;

  [[nodiscard]] native::Workspace& workspace() const { return *workspace_; }

  std::unique_ptr<BoundSession> start(nb::handle run, const nb::object& host);
  std::unique_ptr<BoundProbe> probe(nb::handle run, const nb::object& host);
  // Everything `NativeBackend.metadata` assembles from the state's facts.
  [[nodiscard]] nb::dict diagnostics() const {
    const auto& state = workspace_->built();
    nb::dict out;
    out["memory_ledger"] = state.memory_ledger();
    out["execution_sizing"] = state.execution_sizing();
    out["cpu_execution"] = cpu_execution_dict(state.cpu_execution());
    out["fft_planning"] = fft_planning_dict(state.fft_planning());
    out["planning_seconds"] = state.planning_seconds();
    out["raman_planning_seconds"] = state.raman_planning_seconds();
    out["raman_fft_provider"] = state.raman_fft_provider();
    out["raman_fft_tasks"] = state.raman_fft_tasks();
    out["raman_kernel_transforms"] = state.raman_kernel_transforms();
    out["fsal_ready"] = state.fsal_ready();
    out["rk_source_evaluations"] = state.rk_source_evaluations();
    out["axis_scans"] = state.axis_scans();
    out["synthesis_provider"] = state.synthesis_provider();
    out["projection_provider"] = state.projection_provider();
    out["gemm_shape_policy"] = state.gemm_shape_policy();
    out["response_team_size"] = state.response_team_size();
    out["dispersive_path"] = state.dispersive_path();
    const auto crossed = state.crossed_bytes();
    out["crossed_bytes"] = nb::make_tuple(crossed.to_host, crossed.from_host);
    out["table_evaluations"] = workspace_->z_slots().evaluations;
    static constexpr std::array<const char*, 3> kinds{"adams", "runge_kutta",
                                                      "low_storage"};
    const auto kind = workspace_->kind();
    out["method"] = kinds[static_cast<std::size_t>(kind)];
    out["history_rows"] = kind == native::MethodKind::adams
                              ? state.adams_row_names()
                              : std::vector<std::string>{};
    return out;
  }
  // The live counters alone, cheap enough to read at every run's start and close and
  // every plane's metadata: the on-axis row scans, the Runge-Kutta stage evaluations,
  // the z-table positions, the Raman kernel's transforms and the recycled stage.
  [[nodiscard]] nb::dict counters() const {
    const auto& state = workspace_->built();
    nb::dict out;
    out["axis_scans"] = state.axis_scans();
    out["rk_source_evaluations"] = state.rk_source_evaluations();
    out["table_evaluations"] = workspace_->z_slots().evaluations;
    out["raman_kernel_transforms"] = state.raman_kernel_transforms();
    out["fsal_ready"] = state.fsal_ready();
    return out;
  }
  [[nodiscard]] bool closed() const noexcept { return workspace_->closed(); }
  // Closes a session or probe that leases the workspace first. Python-free: the runtime
  // calls it at shutdown, after the interpreter.
  void close() noexcept override { workspace_->close(); }
  // `close` from Python: without the interpreter lock (a session's close waits for an
  // advance in hand), then the host source dropped with it once the state is gone,
  // since the state's reference to it is invisible to the garbage collector.
  void close_from_python() noexcept {
    {
      const nb::gil_scoped_release release;
      close();
    }
    if (workspace_->closed())
      host_owner_ = nb::object();
  }
  [[nodiscard]] std::size_t raman_terms() const noexcept { return raman_terms_; }
  [[nodiscard]] bool raman() const noexcept { return raman_; }

private:
  // Bind `host` as the next run's host source (solprop SEP 49), or unbind with None.
  void bind_host(const nb::object& host);

  std::size_t raman_terms_{};
  bool raman_{};
  std::unique_ptr<native::Workspace> workspace_;
  nb::object host_owner_;
};

} // namespace solprop_native::bindings

// The statistics plans a run's start names cross as the bound `StatisticsPlan`.
template <>
struct solprop_native::records::Extern<
    std::shared_ptr<const solprop_native::statistics::Plan>> {
  static constexpr const char* kind = "StatisticsPlan";
  static void read(const nb::handle value,
                   std::shared_ptr<const solprop_native::statistics::Plan>* out,
                   const std::string& path) {
    if (value.is_none()) {
      if (out != nullptr)
        out->reset();
      return;
    }
    const bindings::BoundStatisticsPlan* plan = nullptr;
    if (!nb::try_cast<const bindings::BoundStatisticsPlan*>(value, plan) ||
        plan == nullptr)
      solprop_native::records::type_refusal(path, "must be a StatisticsPlan or None");
    if (out != nullptr)
      *out = plan->plan();
  }
  static nb::object
  write(const std::shared_ptr<const solprop_native::statistics::Plan>& plan) {
    return plan ? nb::cast(plan->packed_size()) : nb::none();
  }
};
