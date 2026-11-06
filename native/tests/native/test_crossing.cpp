// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "solprop_native/crossing.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

namespace {

using solprop_native::crossing::chunk_bytes;
using solprop_native::crossing::host_buffer;
using solprop_native::crossing::Ring;
using solprop_native::crossing::ring_threshold;
using solprop_native::crossing::takes_ring;

std::vector<std::uint8_t> counting(const std::size_t size) {
  std::vector<std::uint8_t> values(size);
  std::iota(values.begin(), values.end(), std::uint8_t{1});
  return values;
}

// A lossless copy is an exact structural contract (SEP 25), so these compare bits. On a
// host build the workers' copies are plain ones; a device's run in the device tests.
class RingSizes : public ::testing::TestWithParam<std::size_t> {};

TEST_P(RingSizes, CrossesEveryByteBothWays) {
  const auto source = counting(GetParam());
  std::vector<std::uint8_t> down(source.size(), 0), up(source.size(), 0);
  Ring ring;
  ring.to_host(down.data(), source.data(), source.size());
  ring.from_host(up.data(), down.data(), down.size());
  EXPECT_EQ(down, source);
  EXPECT_EQ(up, source);
}

INSTANTIATE_TEST_SUITE_P(AroundAChunk, RingSizes,
                         ::testing::Values(std::size_t{1}, chunk_bytes - 1, chunk_bytes,
                                           chunk_bytes + 1, 3 * chunk_bytes + 17));

TEST(Ring, HoldsNothingUntilItsFirstCrossingAndAfterItsRelease) {
  Ring ring;
  EXPECT_EQ(ring.bytes(), 0U);
  const auto source = counting(16);
  std::vector<std::uint8_t> copy(16);
  ring.to_host(copy.data(), source.data(), source.size());
  const auto held = ring.bytes();
  EXPECT_GE(held, chunk_bytes);
  EXPECT_EQ(held % chunk_bytes, 0U);
  ring.to_host(copy.data(), source.data(), source.size());
  EXPECT_EQ(ring.bytes(), held);
  ring.release();
  EXPECT_EQ(ring.bytes(), 0U);
}

TEST(Ring, TakesNoTurnOnAHostBuildOrBelowItsThreshold) {
  EXPECT_FALSE(takes_ring(ring_threshold - 1));
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  EXPECT_TRUE(takes_ring(ring_threshold));
#else
  EXPECT_FALSE(takes_ring(ring_threshold));
#endif
}

TEST(HostBuffer, GivesWritableMemoryOfEitherSize) {
  for (const std::size_t count : {std::size_t{1}, ring_threshold / sizeof(double)}) {
    auto buffer = host_buffer(count);
    ASSERT_NE(buffer.get(), nullptr);
    buffer[0] = 1.0;
    buffer[count - 1] = 2.0;
    EXPECT_EQ(buffer[count - 1], 2.0);
  }
}

} // namespace
