---
name: ako4all
description: Drive an agentic loop that iteratively optimizes a Tenstorrent kernel for maximum speedup. Use this skill whenever the user wants to optimize / speed up / benchmark a Tenstorrent kernel — a TT-NN (ttnn) op or a low-level tt-metal Tensix kernel (reader / compute / writer, circular buffers) — mentions AKO / AKO4ALL / AKO4X / agentic kernel optimization, Tenstorrent / Wormhole / Blackhole / Tensix / tt-metal / TT-NN, asks to "make this ttnn op faster", or has a kernel they want measured against a PyTorch reference on a Tenstorrent device. The skill handles setup, profiling (tt-metal device profiler / Tracy), PCC correctness checking, iteration logging, and git commits. Bootstraps a workspace in any directory the user points at.
---

# AKO4ALL — Agentic Kernel Optimization for Tenstorrent

Drive a profile → modify → benchmark → log → commit loop on a Tenstorrent kernel until it runs faster than the reference. The user provides at minimum a kernel; everything else (reference, inputs, bench script, hints) is optional. The reference is a plain PyTorch `class Model` that runs on the **CPU** as the numerical golden; the kernel under optimization runs on a **Tenstorrent device** (TT-NN ops and/or tt-metal Tensix kernels) and is scored by PCC (Pearson Correlation Coefficient), the standard TT correctness metric.

## When this skill applies

- "optimize this kernel" / "speed up this ttnn op / tt-metal kernel"
- "run AKO / AKO4ALL on ..."
- "benchmark this Tenstorrent kernel against PyTorch"
- "iterate on this kernel until it's faster (on Wormhole / Blackhole)"
- mentions of the tt-metal device profiler / Tracy, `TT_METAL_DEVICE_PROFILER`, perf counters, Tensix, circular buffers, a TT speedup target

Does NOT apply when:
- User wants to *write* a new kernel from scratch with no optimization target — just write code, no loop.
- User wants Codex / GPT to review or implement — use `codex:rescue` instead.
- User wants generic performance advice for code that isn't a Tenstorrent kernel.

## First action

Before doing anything else, establish the **workspace** — the directory the loop runs in. It is typically the user's CWD, or a subdirectory / path they name in the prompt.

### Inventory the workspace + prompt

Browse the workspace (don't run a fixed checklist — look around) and read the user's prompt to identify what the loop needs:

- **Kernel** (required) — the code to optimize. Either a **TT-NN** solution (a Python file whose `forward` runs `ttnn` ops on the device) or a **tt-metal** solution (a Python host wrapper that builds/launches C++ Tensix kernels — reader / compute / writer `.cpp` files, circular buffers — via `ttnn` / `tt_metal`).
- **Reference** (optional) — the correctness golden, a PyTorch `class Model` run on CPU. If absent, the original kernel is used.
- **Input data** (optional) — data files the kernel consumes (`.npz`, `.bin`, `.pt`, shape lists, custom formats, etc.)
- **Knowledge** (optional) — reference materials: algorithm notes, papers, design docs, prior PRs. Typically under `knowledge/` but anywhere the user points at. `knowledge/tenstorrent.md` (shipped with the skill) is the TT stack cheat-sheet — read it.
- **Bench mode** — user-provided bench script vs. the default `bench/kernelbench/` evaluator
- **Scaffold presence** — whether `bench-wrapper.sh`, `HINTS.md`, `ITERATIONS.md`, `bench/kernelbench/` are already at workspace root

Whether the workspace follows AKO4ALL's `source/` / `knowledge/` / `bench/` naming or some entirely different layout is **not** the signal. What matters is whether you can identify each item above with confidence.

### Ask only when genuinely uncertain

If the user's prompt + filesystem give you confidence about every required item, **don't ask** — skip straight to presenting the plan. Ask only when a piece's role is genuinely ambiguous (a kernel-shaped file with no obvious reference, two files that could both be the kernel, an input data file in an unfamiliar format you need permission to wire up a custom way, uncertainty over whether the target is TT-NN or tt-metal, etc.). When in doubt, asking is cheaper than guessing wrong.

### Always present the resolved plan before running anything

Whether you asked the user or not, list back what you decided — so the user can correct you even when you didn't think you needed to ask.

Use the format below. Bold field labels + inline-code path values + the leading emoji marker make the plan visually scannable in any terminal theme (don't flatten to a wall of prose):

**📋 Resolved Plan**

- **Workspace** — `<path>`
- **Kernel** — `<path>`
- **Backend** — TT-NN (ttnn) *(or tt-metal C++ Tensix kernels — auto-detected from source)*
- **Reference** — `<path>` *(or none — will use original kernel)*
- **Input data** — `<path>` *(or inline in ref, or none)*
- **Knowledge** — `<path>` *(or none)*
- **Bench mode** — default (TT KernelBench evaluator) *(or custom: `<path>`)*
- **PCC threshold** — `0.99` *(loose; use `0.999` / `0.9999` for fp32/bf16-strict ops)*
- **Scaffold to copy** — `<list of missing files>` *(or none — already present)*

If anything still feels uncertain at this point, **stop and ask**. Otherwise proceed to Workflow.

### Bringing in scaffold

When copying scaffold (`bench-wrapper.sh`, `bench/kernelbench/`, starter `HINTS.md` / `ITERATIONS.md`, `workspace.gitignore` → as `.gitignore` in the workspace) from this skill's own directory into the workspace, **do not overwrite** files that already exist — the user may have edited `HINTS.md`, or `ITERATIONS.md` may carry prior iteration history. Copy only what's missing.

### Persisting user-supplied hints

The user may supply behavior directives in two ways:
- **Inline in the prompt** — e.g., "prefer bfloat8_b" or "don't use sharding".
- **External file reference** — e.g., "follow rules in /tmp/x.md" or "see hints.md".

In **both cases**, merge those directives into `HINTS.md`. It's the persistence layer — directives that only live in the current session's plan are lost on resume.

### Surfacing HINTS.md changes

Whenever you merge directives into `HINTS.md`, tell the user explicitly what happened. Example phrasings:

> "I added your 'prefer bfloat8_b' directive from the prompt to HINTS.md."
> "I added the 3 rules from /tmp/user-hints.md to HINTS.md."

Without this acknowledgment the user can't tell from your reply whether you added, replaced, or silently dropped their directives. Always name the **source** ("from your prompt" / "from /tmp/x.md").

## Workflow

1. **Analyze inputs.** Building on the inventory above, confirm `class Model` and `get_inputs()` can be assembled for default bench mode; if not, **stop and ask the user**. See `bench/kernelbench/GUIDE.md` for the input assembly contract and the **device contract** — the solution's `forward` receives torch CPU tensors, does `ttnn.from_torch(...) → ttnn ops → ttnn.to_torch(...)` itself, and returns a torch tensor; the opened TT device is injected as the module global `DEVICE`.
2. **Create branch.** `git checkout -b opt/<kernel-name>`. If the workspace isn't a git repo, init one first.
3. **Initialize solution.** Create `solution/` and `scripts/`. Copy the kernel implementation files into `solution/` (the kernel itself, including any `.cpp` Tensix kernel files a tt-metal solution references — keep them next to the Python host file so `os.path.dirname(__file__)`-relative paths resolve). Reference / inputs helper files stay at their resolved locations. **Do not copy or `mkdir` canonical directories** (`source/`, `input/`, etc.) when the user's files already exist elsewhere. Point bench.sh's `--ref` and `--inputs` flags at the resolved paths in place. `solution/` is the only directory the loop owns.
4. **Generate bench.sh.** Build the bench command with adjusted paths, pipe through `2>&1 | tee _bench_output.txt`. Replace `{{BENCH_COMMAND}}` in `bench-wrapper.sh` to produce `scripts/bench.sh`. For default bench mode the command is `python bench/kernelbench/bench.py --ref <ref> --solution solution/<kernel> [--inputs <inputs-file>] --pcc 0.99 --verbose` — include `--inputs` only when inputs are defined outside the ref file. **Do not hardcode `--backend`**; bench.py auto-detects ttnn vs tt-metal from solution source (add `--backend` only to override the sniff). `scripts/bench.sh` is a starting template — when the bench env needs setup (conda activate, `TT_METAL_HOME` / `PYTHONPATH` / `ARCH_NAME` exports), edit it freely; preserve only the trajectory section (LABEL/TIMESTAMP handling and `cp -r solution/* "$TRAJ_DIR/"`).

   **Common env friction:** `ttnn` must be importable and a Tenstorrent device reachable. tt-metal usually lives outside the workspace; the wrapper needs `export TT_METAL_HOME=<tt-metal>`, `export PYTHONPATH=$TT_METAL_HOME:$PYTHONPATH`, and `export ARCH_NAME=wormhole_b0` (or `blackhole`). If python lives in a sub-env, put `PATH=<env-bin>:$PATH` (or `source <conda>/etc/profile.d/conda.sh && conda activate <env>`) at the top of `scripts/bench.sh`. Discover the tt-metal tree with `ls -d /home/*/tt-metal /opt/tt-metal ~/tt-metal 2>/dev/null` and sub-envs with `ls /home/*/anaconda3/envs/*/bin /opt/conda/envs/*/bin 2>/dev/null`. A quick sanity check: `python -c "import ttnn; d=ttnn.open_device(device_id=0); ttnn.close_device(d); print('ok')"`.
5. **Verify baseline.** Run `bash scripts/bench.sh`. Expect `CORRECT=True` (PCC ≥ threshold). If not, diagnose and fix before iterating (use DPRINT / Watcher — see Debugging below). Commit: `git add -A && git commit -m "[baseline] Initialize solution and benchmark"`. Then run the device profiler once on the baseline to inform iter-1 direction.

## Iteration protocol

Every modification to `solution/` followed by a bench run = one iteration. Number sequentially (1, 2, 3, …). Each iter is exactly three steps:

1. `bash scripts/bench.sh iter-N` — label is required, must match `iter-N` format.
2. Append a structured entry to `ITERATIONS.md` (template inside that file).
3. `git commit -m "[iter N] <short description of optimization direction>"`.

**Steps 2 and 3 MUST be the next two tool calls after step 1 — no profiling, no probes, no reads, no planning the next iter between them.** A failed or partial bench is still an iter; log + commit first, debug after. This is the most-missed step in practice: agents read the bench result and telescope into next-iter analysis (probes, profiler, hypothesis forming) without closing out the current one, leaving commit gaps with `ITERATIONS.md` entries written from memory later.

Backstop: if you catch yourself starting a new iter (Editing `solution/`, or running the profiler/probes for the next direction) and `git log -1` doesn't show `[iter N] ...`, stop and finish the prior iter's steps 2 and 3 first. Related experiments that belong together narratively get grouped in `ITERATIONS.md` analysis prose, not in batched git commits.

Profile to identify bottlenecks — see "Device profiler" below for the workflow and analytical fallback. Do not optimize blindly.

## Keeping the iteration loop fast

A bench run must be cheap enough to iterate against (seconds to low minutes). When it isn't, the cause is almost always an **expensive reference** — it's re-run for every correctness trial *and* re-timed for the speedup denominator, yet it's **invariant across solution edits**, so most of that cost is wasted. This is an eval-*time* problem, not a metric problem: never change *what* you compare against to make the bench cheaper.

Separate the per-iteration **signal** from the final **verdict**:

- **Signal** (every iter): rank candidates by the **solution's own `RUNTIME`** (its device latency, lower is better) — the reference contributes nothing to comparing two solutions.
- **Verdict** (before committing a winner / a `final`): a full run — full trial counts, reference measured, real `SPEEDUP`, PCC re-check.

**Whose eval is it determines what you may touch:**

- **Default bench** (`bench/kernelbench/bench.py` — the skill owns it): pull levers freely, cheapest-and-safest first — `--no-ref` (skip the CPU reference timing; `REF_RUNTIME`/`SPEEDUP` → -1; `COMPILED`/`CORRECT`/`PCC`/`RUNTIME` unaffected) → trim `--num-perf-trials` (e.g. 100→20; latency noise only) → trim `--num-correct-trials` (higher risk: weakens the fresh-input PCC anti-cheat, keep ≥1 in the loop, full at the gate).
- **User-provided eval** (custom `{{BENCH_COMMAND}}`): the trial counts, PCC handling, and reference handling are the **user's contract** — do **not** inject `--no-ref` or cut counts on a script you didn't author. Use only the fast-iteration switches the user exposed (flags / env vars documented in the prompt or `HINTS.md`). If iteration is too slow and none exist, **raise it with the user** — don't fabricate one.

The bench discards the first run automatically (it JIT-compiles the Tensix kernels and populates the program cache); steady-state numbers come from the cached runs, so a warm program cache is assumed. If per-iteration host dispatch overhead dominates a small kernel's time, that is itself a finding — Metal Trace and multiple command queues are the fix (see Optimization levers).

## Stall handling

When 3 consecutive iterations show no improvement (≥3% over current best), pause the loop and re-assess before iter N+1. Re-assessment combines:

- **Re-profile** with the tt-metal device profiler (per-op / per-RISC latency) and, if available, **perf counters** (`TT_METAL_PROFILE_PERF_COUNTERS=47`) for the utilization/stall breakdown; or re-read runtime stats from `ITERATIONS.md` (min vs mean, distribution shape) if the profiler is unavailable.
- **WebSearch** for op-specific best-known techniques / numbers on the same Tenstorrent hardware class (Wormhole / Blackhole).
- **Review `ITERATIONS.md`** for patterns (which axes have been tried — dtype, fidelity, layout, sharding, core grid, CB depth, trace — which haven't, where prior wins came from).

Default outcome: pick a new direction and continue. Only escalate to stop (see next section) if re-assessment produces concrete evidence the current state is at a physical floor.

## When to stop

Legitimate triggers:

1. User-specified iteration cap reached (in prompt or `HINTS.md`).
2. Stall re-assessment produced hard evidence of a physical floor — cite the evidence in `ITERATIONS.md`. On Tenstorrent the floors are:
   - **DRAM bandwidth** for memory-bound kernels: achieved GB/s (bytes moved / kernel time) near the ceiling — Wormhole ≈ 288 GB/s (12 Gbps parts) / 336 GB/s (14 Gbps); Blackhole ≈ 512 GB/s (p150). A well-tuned reader reaches >90% of ceiling.
   - **Matrix-engine FLOPS** for compute-bound kernels: achieved TFLOPS (2·M·N·K / time) near peak — Wormhole ≈ 190 TFLOPS; Blackhole ≈ 332 TFLOPS (bf16) / 664 TFLOPS (bfp8). Per-op utilization = ideal_cycles / actual_cycles.
   - **L1 residency**: working set already fits the ~1.5 MB/core L1 with no DRAM spills.
   - **Dispatch overhead**: `HOST DURATION` dominates `DEVICE KERNEL DURATION` on a small op and Metal Trace + multi-CQ have already been applied.
3. All viable directions exhausted: document at least 3 distinct directions tried (with their iteration numbers) in `ITERATIONS.md` before invoking this trigger, to prevent premature stops.

Do not stop silently because tooling is unavailable — that's a re-assessment input, not a stop reason.

### HEAD handling on stop

After deciding to stop, leave HEAD at the best-performing iter — not necessarily the latest. Procedure:

1. Identify the best iter by reading `ITERATIONS.md` Summary, the bench output for each iter under `trajectory/`, and your own reasoning notes. Useful signals from the bench output: mean `RUNTIME` (lower better), runtime std (consistency), min runtime (tail), `PCC` (accuracy margin), `CORRECT` flag. Justify your pick in the commit message (e.g., "iter 4: lowest mean AND min runtime, PCC 0.9992; iter 6 ties on mean but PCC only 0.991").

2. If best iter ≠ latest iter:
   - `git checkout <best-iter-sha> -- solution/` — verbatim copy, do NOT hand-reconstruct from memory or earlier notes.
   - `bash scripts/bench.sh final` to sanity-verify on a fresh run.
   - `git commit -m "[final] Restore iter-K (X.XXx) — <one-sentence why>"`.

The `git checkout` step is mandatory. Manual reconstruction risks introducing silent drift from the actually-benched code.

## Device profiler — best effort, not a gate

The tt-metal **device profiler** (Tracy-based) is the Tenstorrent analog of `ncu`. Probe it once after baseline. If it fails (no profiler-enabled build, no hardware, user opt-out via free-text `HINTS.md` directive), proceed analytically for the rest of the loop without re-probing within this session. Don't gate iteration progress on profiler availability; analytical reasoning (roofline vs DRAM bandwidth / matrix-engine FLOPS) plus runtime stats from the bench harness are a valid substitute for direction picking.

How to use it:

- **Build** tt-metal with the profiler — Tracy is **on by default**, so a plain `./build_metal.sh` includes it (`--disable-profiler` turns it off). At runtime, set `TT_METAL_DEVICE_PROFILER=1` (off by default — the readback adds overhead).
- **Per-op device latency**: `cd $TT_METAL_HOME && ./tools/tracy/profile_this.py -n <name> -c "pytest <test>"` (or `python -m tracy -m pytest <test>`). Read `generated/profiler/reports/ops_perf_results_<ts>.csv`. The key columns are `DEVICE KERNEL DURATION [ns]` (primary per-kernel device time), `DEVICE FW DURATION [ns]` (fixed firmware overhead), `HOST DURATION [ns]` (host dispatch), and per-RISC `DEVICE {BRISC,NCRISC,TRISC0,TRISC1,TRISC2} KERNEL DURATION [ns]`, plus `DEVICE COMPUTE CB WAIT FRONT [ns]` (compute starved) / `CB RESERVE BACK [ns]` (compute back-pressured).
- **Bottleneck metrics** (the `ncu --metrics` equivalent): `export TT_METAL_PROFILE_PERF_COUNTERS=47` (FPU|PACK|UNPACK|L1|INSTRN) then run under `python -m tracy --profiler-capture-perf-counters=all`. Triage with: MATH/FPU/SFPU Util % (compute-bound?), Thread 0/1/2 Stall Rate % (T0=unpack starvation, T1=math, T2=pack), NOC vs Compute Balance % (>60% → NOC-bound), Compute-to-Unpack Ratio (<20% → memory-bound), L1 Total Bandwidth Util %, Fidelity Stall Rate (HiFi cost).
- **Turn debug tooling OFF while timing.** DPRINT and Watcher both perturb latency and cannot run at the same time as `TT_METAL_DEVICE_PROFILER` — enable them only in the diagnose step, never the measure step.

## Optimization levers (Tenstorrent)

"Rewrite the kernel" on Tenstorrent means pulling these — not just tuning configs. Roughly ordered by typical impact:

- **Data format + math fidelity** (biggest single compute lever): move bf16 → `bfloat8_b` (HiFi2, ~1.5–1.8× faster) or `bfloat4_b` (LoFi, ~2–3.5×). Block-float also halves/quarters DRAM & L1 traffic. Re-check PCC after every downgrade; loosen the threshold only if the user allows.
- **Memory placement + sharding**: keep hot tensors in **L1** (`ttnn.L1_MEMORY_CONFIG`) not DRAM; shard to match the access pattern (HEIGHT for row-wise, WIDTH for column-wise, BLOCK for matmul) and spread across the core grid.
- **Layout**: keep compute inputs in `TILE_LAYOUT` (32×32); avoid unnecessary `tilize`/`untilize` (`ttnn.to_layout`) round-trips; keep dims multiples of 32 to avoid padding.
- **Core-grid occupancy**: use more Tensix cores (`core_grid` / `CoreRangeSet`; up to 64 on WH n150, 130 on BH) so work is balanced across the chip.
- **Circular-buffer double-buffering**: size CBs to hold >1 tile so the reader prefetches while compute consumes; split reader (RISCV_0) and writer (RISCV_1) on separate NoCs.
- **Metal Trace + multiple command queues + program cache**: `ttnn.begin_trace_capture`/`execute_trace` removes per-iteration host dispatch overhead; `num_command_queues=2` overlaps IO with compute; program cache eliminates recompiles. Biggest wins on small ops where host overhead dominates.
- **Kernel fusion / matmul tiling**: fuse elementwise chains in the DST register to avoid CB round-trips; tune matmul `per_core_M/N`, `in0_block_w`, `out_subblock_h/w`; use `fp32_dest_acc_en` only where accuracy needs it.

## Gotchas

- **Pursue genuine latency reduction, not reward hacking.** No returning a constant / precomputed / uninitialized tensor, no monkey-patching the benchmark, no skipping the device round-trip. The PCC check runs on **fresh random inputs each trial**, so a solution that doesn't actually compute the op fails correctness — investigate any suspiciously large speedup before celebrating.
- **The solution file must not contain `get_inputs` / `get_init_inputs`.** The bench strips the solution's module-level tail before eval as an anti-cheat boundary. Inputs come from the reference or `--inputs` file, never the solution.
- **`get_inputs()` must produce fresh data each call.** Bench calls it once per correctness trial and again for timing. Module-level cached tensors make PCC checks trivial and let timing measure cache-warm behavior. Use `torch.randn` or reload from disk on every call.
- **Correctness is PCC, not tight `allclose`.** TT kernels run in bf16 / bfloat8_b; individual elements diverge from an fp32 golden while staying highly correlated, so fp32-grade tolerances are unreachable. Default gate PCC ≥ 0.99; tighten to 0.999 / 0.9999 for fp32-accumulated ops. (PCC blind spot: a global scale/bias can still give PCC ≈ 1 — sanity-check magnitudes if an op is suspiciously "correct".)
- **Warm the program cache; the first run is compile, not steady state.** The bench discards it automatically; if you time manually, do the same.
- **Keep DPRINT / Watcher off while timing** — they perturb latency and conflict with the device profiler.
- **Don't be lazy.** Stay-in-one-dtype, only-tune-configs, skip-profiling — these are the default low-effort failure modes. The point of the loop is to *rewrite*: change the data format / math fidelity, restructure memory (L1, sharding), retile, add Metal Trace, or drop from a ttnn op to a hand-written tt-metal kernel when it helps.

## Debugging (when a kernel is wrong or hangs)

- **DPRINT** (`#include "api/debug/dprint.h"`, enable with `TT_METAL_DPRINT_CORES="(0,0)"`) prints device-side values; use `TSLICE()` / `print_bf16_pages` to dump circular-buffer tiles and diff against the reference. Every print must end with `\n`.
- **Watcher** (`TT_METAL_WATCHER=<poll-seconds>` → `generated/watcher/watcher.log`) is the compute-sanitizer analog: NoC-transaction, circular-buffer-bounds, and per-RISC stack checks, plus each RISC's last waypoint on a hang.
- Reduction discipline for a bad kernel: shrink to a single-op unit test, reduce the core grid, disable the program cache, and use fixed zeros/ones inputs.

## Reference files

- `bench/kernelbench/GUIDE.md` — full input assembly patterns, the device contract (`DEVICE` global, torch-in/torch-out `forward`), CLI flags, PCC thresholds, timing method. Read this before writing the bench command if anything about the solution contract, dtype, or backend is non-obvious.
- `knowledge/tenstorrent.md` — TT stack cheat-sheet: profiler, PCC, data formats, layouts, memory/sharding, optimization levers, physical ceilings, with tech-report citations.
- `HINTS.md` (workspace) — user-editable behavior directives. Read at session start; respect any constraint named there.
- `ITERATIONS.md` (workspace) — your own iteration log. Write to it every iteration.
