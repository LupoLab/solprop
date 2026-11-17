An editable solprop-native build's `source_digest` now also covers the build's CMake
modules (`native/cmake/*.cmake`), so `check_build()` refuses an extension built before a
change to them, as it does for `CMakeLists.txt` and the C++ sources.
