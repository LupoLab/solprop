`solprop-native` wheels for Windows on Arm (`win-arm64`), and an x86-64-v2 build beside the
x86-64-v3 one in the Linux and Windows x86-64 wheels, chosen at import, so processors
without AVX2 now run the native backend. The v2 build runs about 20 to 25 % longer by
geometric mean. `SOLPROP_NATIVE_CPU_VARIANT=x86-64-v2` asks for it.
