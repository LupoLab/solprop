// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// A host-only Kokkos Tools plug-in for coverage campaigns. Counts are execution
// evidence, not timers; it neither instruments device kernels nor asks for fences.
#include <impl/Kokkos_Profiling_C_Interface.h>

#include <cstdlib>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <utility>

namespace {
std::mutex label_mutex;
std::map<std::pair<std::string, std::string>, std::uint64_t> labels;
std::uint64_t next_id = 0;

void record(const char* kind, const char* label, std::uint64_t* id = nullptr) {
  const std::lock_guard<std::mutex> guard(label_mutex);
  ++labels[{kind, label == nullptr ? "" : label}];
  if (id != nullptr) {
    *id = ++next_id;
  }
}

std::string quoted(const std::string& value) {
  constexpr char hex[] = "0123456789abcdef";
  std::string result = "\"";
  for (const char character : value) {
    const auto item = static_cast<unsigned char>(character);
    if (item == '"' || item == '\\') {
      result += '\\';
      result += static_cast<char>(item);
    } else if (item < 32) {
      result += "\\u00";
      result += hex[item / 16];
      result += hex[item % 16];
    } else {
      result += static_cast<char>(item);
    }
  }
  return result + '"';
}
} // namespace

extern "C" {
void kokkosp_init_library(int, std::uint64_t, std::uint32_t,
                          Kokkos_Profiling_KokkosPDeviceInfo*) {}

void kokkosp_request_tool_settings(std::uint32_t, Kokkos_Tools_ToolSettings* settings) {
  settings->requires_global_fencing = false;
}

void kokkosp_begin_parallel_for(const char* label, std::uint32_t, std::uint64_t* id) {
  record("parallel_for", label, id);
}
void kokkosp_end_parallel_for(std::uint64_t) {}

void kokkosp_begin_parallel_reduce(const char* label, std::uint32_t,
                                   std::uint64_t* id) {
  record("parallel_reduce", label, id);
}
void kokkosp_end_parallel_reduce(std::uint64_t) {}

void kokkosp_begin_parallel_scan(const char* label, std::uint32_t, std::uint64_t* id) {
  record("parallel_scan", label, id);
}
void kokkosp_end_parallel_scan(std::uint64_t) {}

void kokkosp_push_profile_region(const char* label) { record("region", label); }
void kokkosp_pop_profile_region() {}

void kokkosp_finalize_library() {
  const char* path = std::getenv("SOLPROP_NATIVE_LABEL_FILE");
  if (path == nullptr) {
    return;
  }
  const std::lock_guard<std::mutex> guard(label_mutex);
  std::ofstream output(path);
  for (const auto& [key, count] : labels) {
    output << "{\"kind\":" << quoted(key.first) << ",\"label\":" << quoted(key.second)
           << ",\"count\":" << count << "}\n";
  }
}
}
