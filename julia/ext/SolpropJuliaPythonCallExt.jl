# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

module SolpropJuliaPythonCallExt

import PythonCall: Py, PyArray
import SolpropJulia: copy_preparation

"""
    copy_preparation(values::Py)

Copy a Python numerical array into independently owned Julia storage.

# Arguments
- `values`: A Python array or buffer with fixed-size numerical elements. Shape, element
  type and the translated input's units are preserved, including strided input arrays.

# Returns
A Julia `Array` that does not retain a Python array, buffer or owner.

# Throws
- `ArgumentError`: If the element type is not a fixed-size subtype of `Number`.
- `PythonCall.PyException`: If accessing the Python array's buffer raises an exception.
"""
copy_preparation(values::Py) = copy_preparation(PyArray(values; copy = false))

end
