// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Tests of the stage tables (`march/stage_tables.hpp`): the scalar-table loader reads
// a one-value table and refuses a multi-valued one by name, before any read.

#include "solprop_native/march/stage_tables.hpp"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

namespace {

namespace native = solprop_native;

// A scalar stage table holds one value: the loader refuses one of two by name, before
// any read could write past the single value it returns.
TEST(StageTable, LoaderRefusesAMultiValuedScalarTableByName) {
  native::stage::Table table;
  table.edges = {0.0, 1.0};
  table.lower = 0.0;
  table.upper = 1.0;
  table.n_terms = 1;
  table.n_values = 1;
  table.coefficients = {2.5};
  native::stage::Tables tables;
  tables.density = table;
  EXPECT_NO_THROW(native::stage::require_scalar_tables(tables));
  EXPECT_EQ(tables.density.value(0.5), 2.5);
  auto wide = table;
  wide.n_values = 2;
  wide.coefficients = {2.5, -1.0};
  tables.plasma_densities = {table, wide};
  try {
    native::stage::require_scalar_tables(tables);
    FAIL() << "a two-value plasma density table was accepted";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string(error.what()).find("plasma_densities[1]"), std::string::npos)
        << error.what();
  }
}

} // namespace
