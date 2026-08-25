# Cost of publishing the callconv entry point under an alias

Measures what the const-route alias lowering costs relative to `declare_function` plus
`add_linking_library`. Two trees are compared: the branch tip and the alias prototype, each
editable-installed into its own venv so neither shadows the other.

Setup, per tree:

    /home/erik/.local/bin/uv venv --python 3.12 <tree>/.venv
    /home/erik/.local/bin/uv pip install --python <tree>/.venv/bin/python -e <tree> pytest numba==0.65.1 llvmlite==0.47.0

Every script is run with `python -P` and an explicit `NUMBA_CACHE_DIR`, because `sys.path[0]`
otherwise wins over the installed package. `gen_bindings.py <dir>` writes the synthetic bindings
the runtime and compile benchmarks import; the generated modules go next to `mods/`.

| script | what it answers |
|---|---|
| `drive_runtime.py` | ns per call, constant route vs argument route, small body vs large |
| `bench_libm.py` | the same for real shipped libm bindings |
| `bench_array.py` | elementwise array map, plain and under fastmath |
| `bench_work.py` | a `Work` node whose derive is a binding's `.as_func` |
| `drive_compile.py` | cold and warm compile time, and cache file sizes |
| `bench_many.py` | a caller reaching twenty bindings as constants |
| `bench_guard.py` | what the stale-alias guard costs on a const-route cache load |
| `drive_import.py` | per-module import wall time for every shipped binding module |
| `bench_alias_cost.py` | the per-decoration work the alias adds, timed directly |
| `bench_cres_import.py` | whether minting a `cres` derive grows the import graph |
| `bench_pin.py` | resident memory held by the alias registry |
| `bench_foreign.py` | cacheability of a caller of a `rewrap_derive`-upgraded wrapper |
| `enum_aliases.py`, `diff_aliases.py` | whether any shipped binding's alias value moved |
| `dump_asm.py` | the caller's own machine code, both trees (`asm_base.s`, `asm_fix.s`) |
