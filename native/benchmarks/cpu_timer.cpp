// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// CPU-only attribution tool for the pinned Kokkos profiling callback ABI.
// OpenMP kernels finish synchronously; disabling tool fences leaves their order intact.
#include <Kokkos_Profiling_C_Interface.h>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
struct Event {
  std::string name;
  Clock::time_point start;
};
struct Total {
  std::uint64_t calls{};
  double seconds{};
};
bool enabled{};
std::vector<Event> events;
std::vector<std::uint64_t> regions;
std::map<std::string, Total> totals;
void begin(const char* name, std::uint64_t* id) {
  *id = 0;
  if (enabled) {
    events.push_back({name, Clock::now()});
    *id = events.size();
  }
}
void end(std::uint64_t id) {
  if (id != 0 && enabled) {
    const auto& event = events.at(id - 1);
    auto& total = totals[event.name];
    total.seconds += std::chrono::duration<double>(Clock::now() - event.start).count();
    ++total.calls;
  }
}
} // namespace
extern "C" {
void kokkosp_request_tool_settings(std::uint32_t, Kokkos_Tools_ToolSettings* settings) {
  settings->requires_global_fencing = false;
}
void kokkosp_begin_parallel_for(const char* name, std::uint32_t, std::uint64_t* id) {
  begin(name, id);
}
void kokkosp_end_parallel_for(std::uint64_t id) { end(id); }
void kokkosp_begin_parallel_reduce(const char* name, std::uint32_t, std::uint64_t* id) {
  begin(name, id);
}
void kokkosp_end_parallel_reduce(std::uint64_t id) { end(id); }
void kokkosp_begin_parallel_scan(const char* name, std::uint32_t, std::uint64_t* id) {
  begin(name, id);
}
void kokkosp_end_parallel_scan(std::uint64_t id) { end(id); }
void kokkosp_push_profile_region(const char* name) {
  std::uint64_t id{};
  begin(name, &id);
  regions.push_back(id);
}
void kokkosp_pop_profile_region() {
  end(regions.back());
  regions.pop_back();
}
void solprop_timer_start() {
  events.clear();
  totals.clear();
  enabled = true;
}
void solprop_timer_stop(const char* path) {
  enabled = false;
  std::ofstream output(path);
  output.precision(17);
  output << "name\tcalls\tseconds\n";
  for (const auto& [name, total] : totals)
    output << name << '\t' << total.calls << '\t' << total.seconds << '\n';
}
}
