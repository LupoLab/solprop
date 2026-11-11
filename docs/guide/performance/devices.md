# Execution devices

Choose a device in `SimulationSettings.device`. It controls where a backend executes;
it does not change the translated scientific problem. The names are `"cpu"`,
`"cuda"`, `"rocm"` and `"metal"`. The default, `None`, accepts the backend's default
target. Unsupported choices raise a named error before numerical preparation.

| Backend | `device=None` | Explicit device |
|---|---|---|
| `native` or `auto` | The installed native build's target | Must match that build: `cpu`, `cuda`, or `rocm` for a HIP build |
| `numpy_ref` | CPU | `cpu` |
| `julia` | CPU | Currently `cpu`; GPU execution remains refused until each extension is qualified |

`auto` still selects native, without fallback. Setting `device="cpu"` on a CUDA
native build refuses; it does not switch the build or select another backend. Metal
requires `precision="single"` (binary32); this requirement alone does not enable its
currently unqualified Julia execution path. Backend setup and precision admission are
separate checks. See [native builds](building-native.md), [Julia](julia.md) and
[single precision](single-precision.md) for the qualified numerical routes.

```python
from solprop.driver import SimulationSettings

settings = SimulationSettings(
    save_planes=[0.001], dz_max=0.0001, dz_init=0.00001,
    backend="numpy_ref", device="cpu",
)
assert settings.execution_options().device == "cpu"
```

Device and `backend_options` travel with the existing settings through runs, reusable
runners and spawned scan workers. Each options mapping is copied. For example, a
Julia setup identity is supplied as `backend_options={"setup_id": "..."}`. The chosen
backend validates its options; another backend does not silently ignore them.

These execution choices may change after translation. They stay outside scientific,
configuration and restart hashes, so changing a device does not change which physical
calculation a file describes. Execution metadata records the effective device. A runner
retains its preparation's owned execution choices: changing them requires a new runner.
Scan workers rebuild a cached runner when a later point selects different execution
choices, including a different setup identity or node budget.

## Recipes

Use the same execution table on every route:

```toml
[execution]
device = "cpu"
backend_options = { setup_id = "published-setup-identity" }
```

The mapping in this example is a Julia option; keep it empty for a backend that does
not accept it. A recipe writes `device = "none"` for the default selection. Python
export writes Python `None` and preserves the options mapping. A recipe's scan builder
passes these settings to each point without starting a backend in the parent process.

The older object-route spelling `[build.settings].backend_options` remains accepted.
If both spellings appear they must be equal; conflicting mappings refuse by name.
Resolved recipes write the current mapping only in `[execution]`, and changing these
choices retains the recipe and its scientific identity.

## Require a GPU in a job

`SOLPROP_REQUIRE_DEVICE` is a guard on a selected target, not a device selector:

| Value | Required effective target |
|---|---|
| `cuda` | Native or Julia CUDA |
| `hip` | Native HIP or Julia ROCm, both named `rocm` by settings |
| `metal` | Julia Metal |
| `any` | Any of these GPU targets |

An unset or empty value imposes no GPU requirement. The guard never redirects a CPU
request, loads a GPU package on its own, or enables an unqualified extension. A native
GPU build must also report a visible device. Leave the variable unset for CPU runs.

This execution control follows [SEP 48](../../dev/seps/sep-0048-julia-backend.md).
