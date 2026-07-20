<h1 align="center">AKO4ALL · Tenstorrent</h1>
<p align="center"><b>Agentic Kernel Optimization for Tenstorrent</b></p>

<p align="center">
  <a href="https://github.com/tenstorrent/tt-metal"><img src="https://img.shields.io/badge/Hardware-Tenstorrent-8A2BE2" alt="Tenstorrent"></a>
  <a href="https://github.com/tenstorrent/tt-metal/tree/main/tech_reports"><img src="https://img.shields.io/badge/tt--metal-tech%20reports-blue" alt="tt-metal tech reports"></a>
  <a href="https://tongminglaic.github.io/AKO"><img src="https://img.shields.io/badge/Upstream-AKO-lightgrey" alt="Upstream AKO"></a>
</p>

<p align="center"><b>A Tenstorrent adaptation of <a href="https://tongminglaic.github.io/AKO">AKO4ALL</a> — retargeted from NVIDIA/CUDA to tt-metal / TT-NN.</b></p>

<p align="center">
  <img src="assets/hero.png" alt="A cartoon robot agent at a single workshop bench iterates on a glowing kernel cube — with iteration logbook, profiler, timer, and commit stamp at hand, and a speedup chart on the chalkboard — illustrating AKO4ALL's single-session, drop-in kernel optimization loop." width="780" />
  <br/>
  <i>Illustration of the single-session, drop-in optimization loop. (Original artwork from the upstream AKO project.)</i>
</p>

## News

- ✨ **Tenstorrent adaptation** — AKO4ALL now targets Tenstorrent hardware (Wormhole / Blackhole) and kernels (tt-metal Tensix kernels + TT-NN ops). Correctness is PCC-based; profiling uses the tt-metal device profiler (Tracy). Knowledge distilled from the [tt-metal tech reports](https://github.com/tenstorrent/tt-metal/tree/main/tech_reports).
- ✨ AKO4ALL is a single drop-in [Claude Code](https://docs.anthropic.com/en/docs/claude-code) skill — invoke it in any working directory.

**Table of Contents**

- [What is AKO4ALL?](#what-is-ako4all)
- [What You Provide](#what-you-provide)
- [Install](#install)
- [How to Use](#how-to-use)
- [Requirements](#requirements)
- [How It Works](#how-it-works)
- [Results](#results)
- [Agent Behavior](#agent-behavior)
- [Permissions](#permissions)
- [Repo Layout](#repo-layout)
- [Example: optimize a TT-NN op](#example-optimize-a-tt-nn-op)
- [Anti-Cheat](#anti-cheat)
- [FAQ](#faq)
- [Acknowledgments](#acknowledgments)

## What is AKO4ALL?

**AKO4ALL is automated Tenstorrent kernel optimization, packaged as a single [Claude Code](https://docs.anthropic.com/en/docs/claude-code) skill.** Drop a kernel into your working directory, invoke the skill, and the agent bootstraps a workspace in place and iteratively rewrites the kernel for maximum performance on a Tenstorrent device — profile, edit, benchmark, repeat, until the kernel stops getting faster. The skill is a single `SKILL.md` protocol document; other coding agents can drive the same loop by following it directly.

It targets both layers of the Tenstorrent stack:

- **TT-NN (ttnn)** — the PyTorch-like Python op library (op choice, dtype, layout, memory config / sharding, core grid, math fidelity, Metal Trace).
- **tt-metal** — low-level C++ Tensix kernels (reader / compute / writer, circular buffers, matmul tiling, NoC data movement).

The reference is a plain PyTorch `class Model` run on the **CPU** as the numerical golden; the kernel under optimization runs on the **Tenstorrent device** and is scored by **PCC** (Pearson Correlation Coefficient) — the standard TT correctness metric — not a tight `torch.allclose`.

## What You Provide

A kernel and at least one set of test inputs are required. Everything else is optional.

- **Kernel** (required) — The kernel to optimize. A TT-NN solution (a Python file whose `forward` runs `ttnn` ops) or a tt-metal solution (a Python host wrapper launching C++ Tensix kernels).
- **Reference implementation** — A plain-PyTorch `class Model` the evaluator runs on **CPU** as the fp32 golden. Required for the built-in evaluator; a ttnn / tt-metal kernel can't double as the CPU golden, so express the intended computation in PyTorch (a plain-PyTorch kernel can be its own reference).
- **Inputs** (at least one set required) — In any form: hardcoded in the kernel/reference, a `get_inputs()` function, or raw data files (`.npz`, `.pt`, `.bin`, shape lists, etc.) the agent wires up itself.
- **Benchmark script** (optional) — Your own benchmark script; the agent reads it to figure out how to run it. If none is provided, the built-in Tenstorrent evaluator is used automatically.
- **Knowledge** (optional) — Reference materials: algorithm descriptions, papers, design docs. The skill ships `knowledge/tenstorrent.md`, a TT stack cheat-sheet the agent reads.
- **Hints** (optional) — Directives for the agent: optimization constraints, PCC floor, focus areas, behavior controls.

## Install

Clone the repo directly into Claude Code's skills directory:

```bash
git clone <this-repo> ~/.claude/skills/ako4all
```

Or, if you already have it cloned somewhere else, symlink it:

```bash
ln -s /path/to/AKO4ALL ~/.claude/skills/ako4all
```

## How to Use

### Mode 1 — Interactive (default)

Open Claude Code in a directory containing your kernel, then ask it to optimize the kernel. Files in the working directory are inspected automatically; you can also point at external paths. Optional constraints — iteration cap, PCC floor, dtype preference, dependency policy, etc. — included in the prompt will be merged into `HINTS.md`.

The skill presents a resolved plan, asks if anything is genuinely ambiguous, and starts iterating. You can interrupt at any point to redirect.

<details>
<summary>Optional: pre-organize your workspace</summary>

If you want the conventional layout the skill recognizes:

```
<workspace>/
├── source/                      # Kernel + optional reference and inputs
│   ├── kernel.py                #   TT-NN op or tt-metal host wrapper
│   ├── kernels/*.cpp            #   (tt-metal) reader/compute/writer kernels
│   ├── reference.py             # Optional — PyTorch golden (runs on CPU)
│   └── inputs.py / data.npz ... # Optional — input data, any format
├── bench/                       # Your own benchmark script (optional)
│   └── ...
├── knowledge/                   # Reference materials (optional)
└── HINTS.md                     # Agent directives (optional)
```

These are conventions, not requirements. If your files are organized differently, just point Claude at them.

</details>

### Mode 2 — Batch via subagents (advanced)

For optimizing several related kernels in parallel, in Claude Code the main agent can spawn one subagent per kernel via the `Task` tool.

> ⚠️ **Subagents can't ask back.** In Claude Code, subagents lack `AskUserQuestion`. The main agent composes each subagent's prompt from your instructions plus what's in each kernel's directory; any decision you don't pin down there will fall to `SKILL.md` defaults. So include the things you care about (per-kernel paths, optional reference / inputs locations, backend, PCC floor, iteration cap) — and let the rest default.

## Requirements

- A coding agent (e.g., [Claude Code](https://docs.anthropic.com/en/docs/claude-code))
- A **Tenstorrent device** (Wormhole or Blackhole) with a working [tt-metal / TT-NN](https://github.com/tenstorrent/tt-metal) install (`ttnn` importable). Set `TT_METAL_HOME`, `PYTHONPATH`, and `ARCH_NAME` (`wormhole_b0` / `blackhole`).
- PyTorch (CPU) for the reference golden and the built-in evaluator; Python **>= 3.10**
- The tt-metal **device profiler (Tracy)** — recommended for per-kernel profiling (built by default; enable at runtime with `TT_METAL_DEVICE_PROFILER=1`). If unavailable, the loop still proceeds (no device profiling — the agent reasons from runtime stats and roofline analysis instead).
- Git

> **Note:** During the loop the agent may run `pip install` etc. to fill in missing dependencies. A container or virtual environment is recommended for isolation; to forbid installs, add a directive to `HINTS.md` (see [Agent Behavior](#agent-behavior)) or restrict via [Permissions](#permissions).

## How It Works

The agent creates an `opt/<kernel>` branch, copies your kernel to `solution/`, and generates `scripts/bench.sh` to wrap your benchmark. After verifying the baseline is correct (PCC ≥ threshold) — and profiling it with the tt-metal device profiler if available — it iterates: edit `solution/` → benchmark → log → commit. After 3 stagnant iterations it re-profiles, plans the next direction, and continues. Defaults like the stall threshold live in `HINTS.md`; iteration history lives in `ITERATIONS.md`. See [`SKILL.md`](SKILL.md) for the full protocol.

The built-in evaluator ([`bench/kernelbench/`](bench/kernelbench/)) opens one TT device, runs the solution's `forward` on it (the solution does `ttnn.from_torch → ops → ttnn.to_torch` itself, using the injected `DEVICE` global), and scores it against the CPU golden with PCC. Timing is a host wall-clock bracketed by `ttnn.synchronize_device`, with the first (compile) run discarded.

## Results

Speedups on Tenstorrent depend heavily on the operator, the shapes, the data format, and the device generation (Wormhole vs Blackhole), so this adaptation ships **no fixed benchmark table** — the numbers you get are the numbers your op and device produce, reported per run as `RUNTIME` (solution device latency), `PCC` (accuracy), and `SPEEDUP`.

What the loop reports, and how it decides it is done:

- **Per iteration:** the solution's own `RUNTIME` (lower is better) is the ranking signal; `PCC` gates correctness.
- **Physical floors** (a legitimate stop): near the DRAM bandwidth ceiling for memory-bound kernels (Wormhole ≈ 288–336 GB/s, Blackhole ≈ 512 GB/s), or near the matrix-engine peak **for the dtype in use** for compute-bound kernels (data-format/fidelity dependent — e.g. Wormhole ≈ 50 TFLOPS bf16 / ≈ 190 bfp4, Blackhole ≈ 332 bf16 / 664 bfp8). Core counts vary with harvesting — query `device.compute_with_storage_grid_size()` at runtime. See [`knowledge/tenstorrent.md`](knowledge/tenstorrent.md).

The original NVIDIA/CUDA study and its FlashInfer-expert results live in the [upstream AKO project](https://tongminglaic.github.io/AKO).

## Agent Behavior

A few defaults worth knowing:

- **Rewriting, not just tuning** — the agent may change the data format / math fidelity (bf16 → bfloat8_b/bfloat4_b), restructure memory (L1 vs DRAM, sharding), retile, add Metal Trace, or drop a TT-NN op down to a hand-written tt-metal Tensix kernel to chase performance.
- **Web search** — enabled; the agent searches for TT-specific optimization ideas after consecutive stagnant iterations.
- **Profiling** — the tt-metal device profiler runs on the baseline before iter-1 and again after 3 stagnant iterations (when available).

To adjust any of this — or to add your own directives (PCC floor, dtype constraints, dependency policies, iteration caps, ...) — edit `HINTS.md` in your workspace. The file is self-documenting.

## Permissions

The optimization loop involves running shell commands (building, benchmarking, profiling). By default, most coding agents prompt for approval on each command. To run fully unattended, grant the necessary permissions through your agent's configuration.

For Claude Code, the simplest option is to bypass all permission checks:

```bash
claude --dangerously-skip-permissions
```

<details>
<summary>Granular control via settings file</summary>

Create `.claude/settings.local.json`:

```json
{
  "permissions": {
    "allow": [
      "Bash(*)", "Read(*)", "Write(*)", "Edit(*)",
      "Glob(*)", "Grep(*)", "Agent(*)",
      "WebFetch(*)", "WebSearch(*)"
    ]
  }
}
```

</details>

For other agents, consult their documentation on permission / auto-approve settings.

## Repo Layout

```
AKO4ALL/
├── SKILL.md                  # Entry point — loaded by Claude Code when the skill triggers
├── HINTS.md                  # Scaffold: agent behavior defaults
├── ITERATIONS.md             # Scaffold: iteration log template
├── bench-wrapper.sh          # Scaffold: bench script template (TT env checks)
├── bench/kernelbench/        # Scaffold: built-in Tenstorrent evaluator (PCC + host timing)
└── knowledge/tenstorrent.md  # TT stack cheat-sheet the agent reads
```

Scaffold files are copied into each workspace on first bootstrap, never overwriting existing files. `README.md`, `LICENSE`, and `assets/` are project metadata — not installed anywhere.

## Example: optimize a TT-NN op

Point the skill at a kernel and a PyTorch reference; it wires up the built-in evaluator and iterates.

1. Install the skill (see [Install](#install)) and make sure `ttnn` imports and a device is reachable:

   ```bash
   export TT_METAL_HOME=/path/to/tt-metal
   export PYTHONPATH="$TT_METAL_HOME:$PYTHONPATH"
   export ARCH_NAME=wormhole_b0
   python3 -c "import ttnn; d=ttnn.open_device(device_id=0); ttnn.close_device(d); print('ok')"
   ```

2. Open Claude Code in a directory with your kernel + reference and give it a prompt:

   ```
   Optimize the ttnn matmul kernel in source/matmul.py against the PyTorch
   reference in source/reference.py. Keep PCC >= 0.99. Optimize for up to
   20 iterations; stop early only if a physical floor is reached.
   ```

The skill resolves the paths, generates `scripts/bench.sh` (which runs `bench/kernelbench/bench.py`), verifies the baseline (PCC), profiles it with the device profiler, and starts iterating — trying bfloat8_b, L1 sharding, larger core grids, Metal Trace, and so on.

## Anti-Cheat

Protection against the agent gaming the metric instead of achieving real speedup:

1. **Agent rules.** The skill instructs the agent to pursue genuine latency reduction — no returning constant / precomputed / uninitialized tensors, no skipping the device round-trip, no monkey-patching the benchmark. Full list in [`SKILL.md`](SKILL.md) "Gotchas".

2. **Evaluator checks** (built-in). Correctness is PCC on **fresh random inputs each trial**, so a solution that doesn't actually compute the op fails. The evaluator strips any input-generating code (`get_inputs` / `get_init_inputs`) from the solution file before evaluation — so the solution can't choose what it's tested against.

3. **Stricter enforcement** (optional). Provide a custom bench script with additional static analysis or a tighter PCC threshold to reject solutions before they are timed.

## FAQ

**1. Which Tenstorrent devices are supported?**
Wormhole and Blackhole (set `ARCH_NAME=wormhole_b0` / `blackhole`). Grayskull software support is discontinued upstream — don't target it.

**2. TT-NN or tt-metal — which does it optimize?**
Both. The evaluator auto-detects the backend from the solution source (`ttnn` vs tt-metal C++ kernel markers). The agent may start at the ttnn op level and drop to a hand-written tt-metal kernel when that's where the speedup is.

**3. Why PCC instead of `allclose`?**
TT kernels run in bf16 / bfloat8_b, whose few mantissa bits make element-wise agreement with an fp32 golden to tight tolerance physically impossible. PCC (Pearson Correlation Coefficient) is the TT-standard correctness metric. Default gate 0.99; tighten to 0.999 / 0.9999 for fp32-accumulated ops.

**4. The device profiler isn't set up. Does the loop still work?**
Yes. Profiling is best-effort. Without it the agent reasons from the bench's runtime stats and roofline analysis (achieved GB/s vs DRAM ceiling, TFLOPS vs matrix-engine peak).

**5. How do I cap iterations?**
By default there is no cap — the agent decides when to stop. To enforce a limit, add a directive to `HINTS.md`, or include it in your prompt (e.g., `Optimize for up to 20 iterations`).

**6. My bench runs on a remote / lab machine with the device. Does that work?**
Yes, as long as your bench script runs from the command line and prints results to stdout. Edit `scripts/bench.sh` to `ssh`/activate the environment where the device lives.

**7. Can I intervene during optimization?**
Yes. Interrupt to redirect, or to manually edit files in `solution/` — then tell the agent to continue.

## Acknowledgments

This project is a Tenstorrent adaptation of **AKO4ALL**. Thanks to:

- [AKO / AKO4ALL](https://tongminglaic.github.io/AKO) — the upstream agentic kernel optimization project and protocol this adapts.
- [KernelBench](https://github.com/ScalingIntelligence/KernelBench) — the benchmark and evaluation format the built-in evaluator derives from.
- [tt-metal](https://github.com/tenstorrent/tt-metal) and its [tech reports](https://github.com/tenstorrent/tt-metal/tree/main/tech_reports) — the source of the Tenstorrent profiling, correctness, data-format, layout, memory, and optimization knowledge encoded here.
