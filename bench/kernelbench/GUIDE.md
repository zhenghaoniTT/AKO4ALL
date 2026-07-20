# Tenstorrent KernelBench Default Benchmark

Self-contained evaluation script for AKO4ALL, targeting **Tenstorrent** (tt-metal / TT-NN). It keeps the KernelBench-format contract (`class Model` + `get_inputs`) but runs the solution on a Tenstorrent device and scores it with PCC (Pearson Correlation Coefficient) against a CPU PyTorch golden. Core logic is inlined so no external KernelBench dependency is needed. Requires a working `ttnn` install and a reachable Tenstorrent device (only `--self-test` runs without either).

## Setup

When `bench/` is empty (default bench mode), use `bench/kernelbench/bench.py` as the benchmark.

### Inputs to assemble

Three pieces must be reachable at eval time:

| Piece | Source priority | Required |
|-------|-----------------|----------|
| `class Model(nn.Module)` with `forward()` | `--ref` file | Yes |
| `get_inputs()` returning sample input tensors | `--inputs` file > `--ref` file | Yes |
| `get_init_inputs()` returning constructor args | `--inputs` file > `--ref` file > defaults to `[]` | No |

The agent's job during Setup is to assemble these from whatever the user provided — usually by pointing `--ref` / `--inputs` at the files in their existing locations, or (only when a new helper file is genuinely needed, e.g. wrapping raw `.npz` data) by writing into `source/`. Don't move or copy existing user files just to canonicalize paths.

### The device contract (how the solution runs on Tenstorrent)

The **reference** `class Model` is plain PyTorch and runs on the **CPU** — it is the numerical golden, nothing else. The **solution** `class Model` (renamed to `ModelNew`) runs on the **Tenstorrent device**. The contract:

- `forward(self, *inputs)` receives **torch CPU tensors** (the same `get_inputs()` tensors the reference sees) and must return a **torch tensor** (or a `ttnn.Tensor`, which the bench converts with `ttnn.to_torch`).
- Inside `forward`, the solution does the device round-trip itself: `ttnn.from_torch(x, dtype=..., layout=ttnn.TILE_LAYOUT, device=DEVICE) → ttnn ops (or a tt-metal program) → ttnn.to_torch(out)`.
- The opened TT device is injected as the module-level global **`DEVICE`** (and set as the ttnn default device where that API exists). Do **not** call `ttnn.open_device` inside the solution — use `DEVICE`.
- `__init__(self, *init_inputs)` may build persistent `ttnn` weight tensors on `DEVICE` from the (CPU) init inputs.

RUNTIME is the solution's end-to-end device latency (including the host↔device transfer), timed with a host wall-clock bracketed by `ttnn.synchronize_device(DEVICE)`, with the first (compile) run discarded.

### Common assembly patterns

These are typical paths, not rules.

- **TT-NN op** — the solution is a Python file whose `forward` calls `ttnn` ops. Reference is the equivalent PyTorch op. Use the ref directly as `--ref`; the solution as `--solution`.
- **tt-metal Tensix kernel** — the solution is a Python `forward` that drives C++ kernels (reader / compute / writer `.cpp`, circular buffers). The raw Metalium host API (`CreateKernel`/`Program`) is C++-only, so from Python use `ttnn.generic_op` with `ProgramDescriptor`/`CBDescriptor`/`KernelDescriptor`, the PyKernel API, or a compiled pybind / `torch.utils.cpp_extension` module that exposes the program. Keep the `.cpp` kernel files **next to** the solution `.py` (the bench loads the solution from a temp file in its own directory so `os.path.dirname(__file__)`-relative kernel paths still resolve).
- **Kernel + separate input data** — user provides a kernel plus data in a non-Python format (`.npz`, `.pt`, `.bin`, shape lists). Keep the reference in `--ref`, then write `source/inputs.py` whose `get_inputs()` loads the data (`np.load`, `torch.load`, custom parser) and returns torch tensors. Pass it via `--inputs`.
- **Kernel referenced by external path** — `--ref` / `--inputs` accept arbitrary paths (e.g. into a model repo); no need to copy files into `source/`.

### Pitfall: `get_inputs()` must produce fresh data per call

The bench calls `get_inputs()` once per correctness trial (5 by default) and again for timing. If it returns a module-level cached tensor, every trial sees identical data — the PCC check passes trivially and timing reflects cache-warm behavior. Regenerate each call (`torch.randn`, re-`np.load`, etc.). Fresh-input PCC is the anti-cheat: a constant / precomputed output fails.

### Bench command

```
python3 bench/kernelbench/bench.py --ref <ref-path>.py --solution solution/<kernel>.py [--inputs <inputs-path>.py] --pcc 0.99 --verbose
```

If the file passed to `--inputs` also defines `get_init_inputs()`, it overrides the ref's version too. The solution file keeps `class Model` — the bench transparently renames it to `class ModelNew` before evaluation.

### Fast iteration on an expensive reference

The reference is re-run for every correctness trial *and* re-timed for the speedup denominator, but it is **invariant across solution edits** — so an expensive reference slows every iteration without helping you *rank* two solutions. For the loop, rank by the solution's own `RUNTIME` and skip the reference; pay for it only at the verdict:

```
# signal (fast): rank by RUNTIME, no reference timing
python3 bench/kernelbench/bench.py --ref <ref>.py --solution solution/<k>.py --no-ref --num-perf-trials 20
# verdict (before declaring a winner): full run, real SPEEDUP + full PCC re-check
python3 bench/kernelbench/bench.py --ref <ref>.py --solution solution/<k>.py
```

`--no-ref` leaves `REF_RUNTIME`/`SPEEDUP` at -1; correctness still runs `--num-correct-trials` PCC trials, so trim that too (keep it ≥1) if the reference is the bottleneck. Never swap the comparison target to speed up the bench — only reduce how often the reference is paid for.

## Output Format

Each run prints structured lines (parsed by the agent):

```
COMPILED: True
CORRECT: True
BACKEND: ttnn (auto)
PCC: 0.999123
RUNTIME: 0.4523
REF_RUNTIME: 1.2301
SPEEDUP: 2.7197x
```

- **COMPILED** — solution imported, `ModelNew` instantiated, and the kernels built (JIT for tt-metal, program build for ttnn) without throwing.
- **CORRECT** — PCC ≥ threshold on every correctness trial (against the CPU golden).
- **BACKEND** — `ttnn` or `tt-metal`, and whether it was `auto`-detected or `explicit`.
- **PCC** — the minimum Pearson Correlation Coefficient across correctness trials (the accuracy margin).
- **RUNTIME** — solution mean device latency in milliseconds (end-to-end, incl. host↔device transfer).
- **REF_RUNTIME** — reference mean latency in milliseconds, on **CPU** (torch).
- **SPEEDUP** — `REF_RUNTIME / RUNTIME`.

Under `--no-ref`, `REF_RUNTIME` and `SPEEDUP` print as `-1` (reference not timed); `COMPILED`, `CORRECT`, `PCC`, and `RUNTIME` are unaffected.

> **Note on SPEEDUP:** the reference runs on CPU, so `SPEEDUP` compares TT-device time to CPU time — a coarse figure. The meaningful per-iteration signal is the solution's own **RUNTIME** (lower is better) and its improvement across iterations, exactly as the loop uses it.

Exit code: `0` = correct, `1` = incorrect or failed.

## CLI Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--ref` | (required) | Path to reference kernel (defines `class Model`, the CPU golden) |
| `--solution` | (required) | Path to optimized kernel (defines `class Model`, runs on the TT device) |
| `--inputs` | (none) | Optional file defining `get_inputs()` (required) and `get_init_inputs()` (optional); overrides definitions in `--ref` |
| `--pcc` | `0.99` | PCC correctness threshold. Use `0.999` / `0.9999` for fp32/bf16-strict ops |
| `--rel-tol` | `0.1` | Relative-L2 magnitude tolerance, checked **alongside** PCC. Catches scale/bias errors PCC misses (e.g. output×2) while tolerating bf16/bfloat8_b noise; loosen for bfloat4_b |
| `--backend` | auto-detected | `ttnn`, `tt-metal` (auto-detected from solution source; pass explicitly to override). Informational — both load the same way |
| `--device-id` | `0` | Tenstorrent device id to open |
| `--num-correct-trials` | `5` | Number of PCC correctness trials |
| `--num-perf-trials` | `100` | Number of performance timing trials |
| `--no-ref` | off | Skip reference (CPU) timing: emit `COMPILED`/`CORRECT`/`PCC`/`RUNTIME` but set `REF_RUNTIME`/`SPEEDUP` to -1. Fast iteration — rank by the solution's own `RUNTIME` |
| `--verbose` | off | Print detailed debug info |
| `--self-test` | off | Run source-transformation / PCC self-test and exit (no ttnn / device needed) |

### Backend selection (auto-detected)

`bench.py` picks a backend label by sniffing the solution source: tt-metal host-API / kernel markers (`CreateKernel`, `CreateProgram`, `EnqueueProgram`, `CircularBufferConfig`, `tt_metal`, `ttnn.experimental`, `program_factory`, …) → `tt-metal`; otherwise `import ttnn` / `ttnn.` → `ttnn`; default `ttnn`. Pass `--backend <name>` only to override the sniff. The label is **informational** — both backends load the same way (a Python file whose `forward` runs on the device), so it does not change execution; it's recorded for the log. The chosen backend prints as `BACKEND: <name> (auto|explicit)`.

| Solution shape | Backend label |
|----------------|---------------|
| TT-NN ops (`ttnn.matmul`, `ttnn.add`, …) | `ttnn` |
| tt-metal host wrapper launching C++ Tensix kernels | `tt-metal` |
| ttnn custom op via `ttnn.experimental` | `tt-metal` |

## Solution File Requirements

- The solution file must contain `class Model(nn.Module)` with a `forward()` matching the reference's signature. It is renamed to `ModelNew` transparently — **do not** rename it yourself.
- `forward` takes torch CPU tensors and returns a torch tensor (or `ttnn.Tensor`); it does the `ttnn.from_torch → ops → ttnn.to_torch` round-trip internally, using the injected `DEVICE` global.
- Do not include `get_inputs()` / `get_init_inputs()` in the solution. The bench removes any top-level `get_inputs` / `get_init_inputs` definitions as an anti-cheat boundary (test inputs come from the reference / `--inputs`, never the solution) — but it **keeps** other post-class helper functions and constants, so you may define those freely.

## Correctness — PCC, not tight allclose

TT kernels run in low-precision formats (bfloat16 has 7 mantissa bits ≈ 2–3 decimal digits; bfloat8_b / bfloat4_b fewer), so element-wise agreement with an fp32 golden to fp32-grade tolerance (`rtol=1e-5, atol=1e-8`) is physically impossible even for a *correct* kernel. The TT-standard metric is PCC (Pearson Correlation Coefficient), which measures whole-tensor correlation.

| Precision of the kernel | Typical PCC threshold |
|-------------------------|-----------------------|
| float32 / bfloat16 (accurate) | 0.999 – 0.9999 |
| bfloat16 (default) | 0.99 – 0.999 |
| bfloat8_b / bfloat4_b (aggressive) | 0.99 (or looser, if the user allows) |

The bench's `comp_pcc` masks each tensor's own NaN/Inf to zero, handles constant tensors by comparing their value with a loose tolerance, and returns 1.0 for identical tensors. **PCC blind spot:** a global scale/bias (e.g. output ×2) can still yield PCC ≈ 1 — so correctness is gated on PCC **and** a relative-L2 magnitude check (`--rel-tol`, default 0.1), which rejects a 2× scale (~1.0 error) while tolerating bf16/bfloat8_b rounding (a few %). Both must pass for `CORRECT: True`.

## Timing

`bench.py` times with a **host wall-clock** around `forward`, calling `ttnn.synchronize_device(device)` before and after each trial (TT dispatch is asynchronous — without the sync you'd time only host enqueue). Warmup runs plus a discarded first trial absorb the JIT compile + program-cache population, so reported numbers are steady-state. This measures **end-to-end** latency including host↔device transfer.

For **per-kernel device time** and bottleneck metrics (the deeper, `ncu`-style analysis — `DEVICE KERNEL DURATION [ns]`, per-RISC breakdown, MATH utilization, thread stall rates), use the tt-metal **device profiler (Tracy)** separately: build with the profiler, set `TT_METAL_DEVICE_PROFILER=1`, and run `./tools/tracy/profile_this.py -n <name> -c "pytest <test>"` → `generated/profiler/reports/ops_perf_results_<ts>.csv`. See `SKILL.md` ("Device profiler") and `knowledge/tenstorrent.md`.
