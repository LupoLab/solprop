# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Build the Windows wheel's FFTW and OpenBLAS on one OpenMP runtime, upstream LLVM's
# libomp (Microsoft's libomp140 is not redistributable).
#
# The recipe proved in solprop SEP 29 phase 0: Clang with the GNU driver on the MSVC
# ABI (MSVC STL and CRT, release /MD), with FFTW and OpenBLAS built static with OpenMP
# on LLVM's libomp, so that Kokkos, FFTW and OpenBLAS share one thread runtime in the
# process. Static, because neither library's Windows DLL exports what the other side
# needs under Clang. delvewheel later copies libomp.dll into the wheel under a mangled
# name. FFTW keeps SSE2 and AVX codelets chosen at run time (x86-64-v2 common code, as
# on Linux: native/ci/build-fftw.sh). OpenBLAS is built for one fixed target, Haswell,
# the wheel's x86-64-v3 baseline, with no run-time kernel selection: built by Clang for
# Windows, its run-time selection (DYNAMIC_ARCH) gave wrong complex results with the
# AVX-512 kernels at one thread (the coupled TE/TM flow off by 5e-4 against rounding)
# and an access violation in the radial route's real GEMM with the Haswell and Zen
# kernels, or whenever a kernel family was forced; the fixed Haswell build (SEP 29
# phase 0's) passes. Newer Clang also does not compile some AVX-512 kernels at all.
#
# Needs LLVM (clang, clang++, libomp; preinstalled on GitHub's Windows images), the
# Visual Studio build tools in the environment (vcvars64, or the MSVC developer
# shell), CMake and Ninja.
#
# Usage: native/ci/build-deps-windows.ps1 -Prefix C:\solprop-deps
# Installs Prefix\libomp\{lib\libomp.lib,bin\libomp.dll}, Prefix\fftw\{include,lib},
# Prefix\openblas\{lib\openblas.lib,LICENSE} and Prefix\licences\LICENSE-*.txt.
param([Parameter(Mandatory = $true)][string]$Prefix)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
# Every download is retried on a transient failure (GitHub answers 429 at times), as
# lib.sh's curl --retry 3 is on Linux and macOS.
$PSDefaultParameterValues['Invoke-WebRequest:MaximumRetryCount'] = 3
$PSDefaultParameterValues['Invoke-WebRequest:RetryIntervalSec'] = 10

$fftwVersion = '3.3.11'
$fftwSha256 = '5630c24cdeb33b131612f7eb4b1a9934234754f9f388ff8617458d0be6f239a1'
$openblasVersion = '0.3.34'  # native/cmake/OpenMPOpenBLAS.cmake pins the same release
$openblasSha256 = 'cd7e129868320cc2d033afa920e31202dfe0b8066a5b66661900ccc0f197dfed'

$work = Join-Path $Prefix '_work'
New-Item -ItemType Directory -Force $work | Out-Null

function Get-Source([string]$Url, [string]$Sha256) {
    $archive = Join-Path $work ($Url -split '/')[-1]
    if (-not (Test-Path $archive)) { Invoke-WebRequest -UseBasicParsing $Url -OutFile $archive }
    $actual = (Get-FileHash $archive -Algorithm SHA256).Hash.ToLower()
    if ($actual -ne $Sha256) { Remove-Item $archive; throw "checksum mismatch for ${Url}: $actual" }
    tar -xzf $archive -C $work
    if ($LASTEXITCODE) { throw "cannot unpack $archive" }
}

function Invoke-Native([string]$What) {
    # Native tools report through their exit code, not through PowerShell errors.
    if ($LASTEXITCODE) { throw "$What failed with exit code $LASTEXITCODE" }
}

# LLVM's libomp, copied beside the other dependencies so that later paths hold no spaces.
$clang = (Get-Command clang).Source
$llvm = Split-Path (Split-Path $clang)
$clangVersion = (& $clang --version | Select-Object -First 1)
if ($clangVersion -notmatch 'clang version (\d+\.\d+\.\d+)') { throw "unknown Clang: $clangVersion" }
$llvmVersion = $Matches[1]
"LLVM $llvmVersion at $llvm"
$omp = Join-Path $Prefix 'libomp'
New-Item -ItemType Directory -Force "$omp\lib", "$omp\bin" | Out-Null
Copy-Item "$llvm\lib\libomp.lib" "$omp\lib\libomp.lib"
Copy-Item "$llvm\bin\libomp.dll" "$omp\bin\libomp.dll"
# The LLVM installer carries no per-runtime licence file; the runtime's, from the
# matching release, travels with the wheel.
# CMake's SOLPROP_NATIVE_BUNDLED_LICENCES installs Prefix\licences into the wheel.
$licences = Join-Path $Prefix 'licences'
New-Item -ItemType Directory -Force $licences | Out-Null
Invoke-WebRequest -UseBasicParsing `
    "https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-$llvmVersion/openmp/LICENSE.TXT" `
    -OutFile "$licences\LICENSE-libomp.txt"
$ompLibrary = ($omp -replace '\\', '/') + '/lib/libomp.lib'
$openmp = @('-DOpenMP_C_FLAGS=-fopenmp', '-DOpenMP_C_LIB_NAMES=libomp',
            "-DOpenMP_libomp_LIBRARY=$ompLibrary")

# FFTW, double and single precision, static, with OpenMP.
Get-Source "https://www.fftw.org/fftw-$fftwVersion.tar.gz" $fftwSha256
foreach ($precision in @('double', 'float')) {
    $build = Join-Path $work "fftw-build-$precision"
    $flags = @('-S', "$work\fftw-$fftwVersion", '-B', $build, '-G', 'Ninja',
        '-DCMAKE_C_COMPILER=clang', '-DCMAKE_BUILD_TYPE=Release',
        '-DCMAKE_POLICY_VERSION_MINIMUM=3.5', '-DCMAKE_C_FLAGS=-march=x86-64-v2',
        '-DBUILD_SHARED_LIBS=OFF', '-DBUILD_TESTS=OFF', '-DENABLE_OPENMP=ON',
        '-DENABLE_THREADS=OFF', '-DENABLE_SSE2=ON', '-DENABLE_AVX=ON',
        "-DCMAKE_INSTALL_PREFIX=$Prefix\fftw") + $openmp
    if ($precision -eq 'float') { $flags += '-DENABLE_FLOAT=ON' }
    cmake @flags; Invoke-Native "FFTW $precision configure"
    cmake --build $build --parallel; Invoke-Native "FFTW $precision build"
    cmake --install $build; Invoke-Native "FFTW $precision install"
}
Copy-Item "$work\fftw-$fftwVersion\COPYING" "$licences\LICENSE-fftw.txt"

# OpenBLAS, static, BLAS plus its C LAPACK translation (Kokkos Kernels needs crot_ and
# zrot_), no Fortran, with OpenMP.
Get-Source "https://github.com/OpenMathLib/OpenBLAS/releases/download/v$openblasVersion/OpenBLAS-$openblasVersion.tar.gz" $openblasSha256
$build = Join-Path $work 'openblas-build'
$flags = @('-S', "$work\OpenBLAS-$openblasVersion", '-B', $build, '-G', 'Ninja',
    '-DCMAKE_C_COMPILER=clang', '-DCMAKE_BUILD_TYPE=Release',
    '-DCMAKE_POLICY_VERSION_MINIMUM=3.5', '-DBUILD_SHARED_LIBS=OFF',
    '-DBUILD_STATIC_LIBS=ON', '-DNOFORTRAN=ON', '-DC_LAPACK=ON',
    '-DBUILD_WITHOUT_LAPACK=OFF', '-DBUILD_TESTING=OFF', '-DTARGET=HASWELL',
    '-DDYNAMIC_ARCH=OFF', '-DUSE_OPENMP=ON', '-DNUM_THREADS=512') + $openmp
cmake @flags; Invoke-Native 'OpenBLAS configure'
cmake --build $build --parallel --target openblas_static; Invoke-Native 'OpenBLAS build'
New-Item -ItemType Directory -Force "$Prefix\openblas\lib" | Out-Null
$library = Get-ChildItem "$build\lib\*.lib" | Sort-Object Length -Descending | Select-Object -First 1
Copy-Item $library.FullName "$Prefix\openblas\lib\openblas.lib"
Copy-Item "$work\OpenBLAS-$openblasVersion\LICENSE" "$Prefix\openblas\LICENSE"
Get-ChildItem "$Prefix\fftw\lib", "$Prefix\openblas\lib", "$omp\lib", "$omp\bin" |
    Select-Object FullName, Length
