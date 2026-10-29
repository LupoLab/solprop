# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compile and execute host OpenMP feature probes; acceptance is not a speed claim."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

# Exact integer-valued computations establish functionality before any floating
# scan, loop transformation or task schedule is tested in a numerical kernel.
FEATURES = {
    "loop": """
        #pragma omp parallel num_threads(4)
        {
          #pragma omp loop bind(parallel)
          for (int i = 0; i < n; ++i) a[i] = i;
        }
    """,
    "scan": """
        int sum = 0;
        #pragma omp parallel for num_threads(4) reduction(inscan, +:sum)
        for (int i = 0; i < n; ++i) {
          sum += 1;
          #pragma omp scan inclusive(sum)
          a[i] = sum - 1;
        }
    """,
    "simd_scan": """
        int sum = 0;
        #pragma omp simd reduction(inscan, +:sum)
        for (int i = 0; i < n; ++i) {
          sum += 1;
          #pragma omp scan inclusive(sum)
          a[i] = sum - 1;
        }
    """,
    "tile": """
        #pragma omp parallel for num_threads(4)
        #pragma omp tile sizes(32)
        for (int i = 0; i < n; ++i) a[i] = i;
    """,
    "unroll": """
        #pragma omp parallel for num_threads(4)
        #pragma omp unroll partial(4)
        for (int i = 0; i < n; ++i) a[i] = i;
    """,
    "taskloop_strict": """
        #pragma omp parallel num_threads(4)
        #pragma omp single
        {
          #pragma omp taskloop grainsize(strict:32)
          for (int i = 0; i < n; ++i) a[i] = i;
        }
    """,
    "metadirective": """
        #pragma omp metadirective when(user={condition(1)}: parallel for) default(simd)
        for (int i = 0; i < n; ++i) a[i] = i;
    """,
    "scope_reduction": """
        int sum = 0;
        #pragma omp parallel num_threads(4)
        {
          #pragma omp scope reduction(+:sum)
          {
            #pragma omp for
            for (int i = 0; i < n; ++i) sum += i;
          }
        }
        if (sum != n * (n - 1) / 2) return 3;
        for (int i = 0; i < n; ++i) a[i] = i;
    """,
    "aligned_allocator": """
        omp_alloctrait_t traits[] = {{omp_atk_alignment, 64}};
        auto allocator = omp_init_allocator(omp_default_mem_space, 1, traits);
        auto* p = static_cast<int*>(omp_alloc(n * sizeof(int), allocator));
        if (!p || reinterpret_cast<std::uintptr_t>(p) % 64) return 3;
        for (int i = 0; i < n; ++i) p[i] = i;
        for (int i = 0; i < n; ++i) a[i] = p[i];
        omp_free(p, allocator);
        omp_destroy_allocator(allocator);
    """,
    "assume_no_openmp_constructs": """
        #pragma omp assume no_openmp_constructs
        {
          for (int i = 0; i < n; ++i) a[i] = i;
        }
    """,
    "reverse": """
        #pragma omp reverse
        for (int i = 0; i < n; ++i) a[i] = i;
    """,
    "stripe": """
        #pragma omp stripe sizes(4)
        for (int i = 0; i < n; ++i) a[i] = i;
    """,
    "interchange": """
        #pragma omp interchange permutation(2, 1)
        for (int j = 0; j < 32; ++j)
          for (int i = 0; i < 32; ++i) a[j * 32 + i] = j * 32 + i;
    """,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--flag", action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    version = subprocess.run(
        [args.compiler, "--version"], capture_output=True, text=True, check=True
    ).stdout
    results = {}
    for name, body in FEATURES.items():
        source = args.out / f"{name}.cpp"
        source.write_text(
            "#include <omp.h>\n#include <cstdint>\n"
            "int main() { constexpr int n=1024; int a[n]{};\n"
            + body
            + "\nfor(int i=0;i<n;++i) if(a[i]!=i) return 2; return 0;}\n",
            encoding="utf-8",
        )
        binary = args.out / name
        command = [
            args.compiler,
            "-std=c++23",
            "-O3",
            "-fno-fast-math",
            "-fopenmp",
            "-Werror=unknown-pragmas",
            *args.flag,
            str(source),
            "-o",
            str(binary),
        ]
        compiled = subprocess.run(command, capture_output=True, text=True, check=False)
        run = None
        if compiled.returncode == 0:
            run = subprocess.run(
                [str(binary.absolute())],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        results[name] = dict(
            command=command,
            compiled=compiled.returncode == 0,
            executed=run is not None,
            passed=run is not None and run.returncode == 0,
            diagnostics=compiled.stdout + compiled.stderr,
            runtime_output="" if run is None else run.stdout + run.stderr,
        )
    (args.out / "results.json").write_text(
        json.dumps(dict(compiler=version, features=results), indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({name: record["passed"] for name, record in results.items()}))


if __name__ == "__main__":
    main()
