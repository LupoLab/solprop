# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Patch the fetched Kokkos so that its host waits never sleep on Windows. Run by
# FetchContent's PATCH_COMMAND from the Kokkos source directory.
#
# Kokkos's HostBarrier backs off with std::this_thread::sleep_for of 256 ns to 4 us, and
# the Threads spin-wait with microsecond sleeps. Windows rounds any sub-millisecond sleep
# up to the system timer resolution, 15.6 ms by default, so every backed-off team barrier
# costs milliseconds: on an 8-core Windows host the reduced comb-to-continuum case took
# 70 s instead of 1.4 s, and its ten-step full-size window over fifteen minutes instead
# of 3.2 s (solprop SEP 29 phase 0). Under _WIN32 both waits yield instead; other
# platforms compile the original code. The patch is idempotent.

function(_solprop_native_patch file old new)
  file(READ "${file}" _text)
  string(FIND "${_text}" "${new}" _done)
  if(NOT _done EQUAL -1)
    return()
  endif()
  string(FIND "${_text}" "${old}" _at)
  if(_at EQUAL -1)
    message(FATAL_ERROR "Kokkos Windows host-wait patch: pattern not found in ${file}")
  endif()
  string(REPLACE "${old}" "${new}" _text "${_text}")
  file(WRITE "${file}" "${_text}")
endfunction()

_solprop_native_patch(core/src/impl/Kokkos_HostBarrier.cpp
"      std::this_thread::sleep_for(
          std::chrono::nanoseconds(c < 16 ? 256 * c : 4096));"
"#if defined(_WIN32)
      // Windows rounds a sub-millisecond sleep up to the timer resolution.
      std::this_thread::yield();
#else
      std::this_thread::sleep_for(
          std::chrono::nanoseconds(c < 16 ? 256 * c : 4096));
#endif")

_solprop_native_patch(core/src/Threads/Kokkos_Threads_Spinwait.cpp
"      std::this_thread::sleep_for(std::chrono::microseconds(c));"
"#if !defined(_WIN32)  // Windows rounds a sub-millisecond sleep up to the timer resolution.
      std::this_thread::sleep_for(std::chrono::microseconds(c));
#endif")
