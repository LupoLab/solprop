// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <cstdio>
#include <exception>

int main(int argc, char** argv) {
  try {
    testing::InitGoogleTest(&argc, argv);
    const Kokkos::ScopeGuard kokkos_scope(argc, argv);
    return RUN_ALL_TESTS();
  } catch (const std::exception& error) {
    std::fprintf(stderr, "native test initialization failed: %s\n", error.what());
  } catch (...) {
    std::fputs("native test initialization failed with an unknown error\n", stderr);
  }
  return 1;
}
