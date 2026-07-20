# Tenstorrent stack cheat-sheet (for AKO4ALL)

Distilled from the [tt-metal tech reports](https://github.com/tenstorrent/tt-metal/tree/main/tech_reports). This is background for the optimization loop — profiling, correctness, the hardware model, the levers, and the physical ceilings. Verify specifics against the linked reports before relying on an exact number for a stop decision.

## Hardware model (how it differs from a GPU)

- A Tenstorrent chip is a 2D grid of **Tensix cores** on a NoC (network-on-chip). There is **no warp/thread model** — parallelism is *spatial* across the core grid plus 32×32-tile SIMD.
- Each **Tensix core** = 5 small RISC-V "baby" cores + a matrix engine (**FPU**) + a vector engine (**SFPU**) + ~1.5 MB **L1 SRAM**. The RISC-Vs: **BRISC** (data movement 0 / reader), **NCRISC** (data movement 1 / writer), and **TRISC0/1/2** = compute pipeline **Unpack → Math → Pack**.
- Every op runs **three cooperating kernels** per core: a **reader** (NoC DMA in), a **compute** kernel (matrix/vector math), and a **writer** (NoC DMA out), coordinated through **circular buffers (CBs)** in L1.
- Memory tiers: per-core **L1 SRAM** (~1.5 MB, software-managed, not a cache) and off-chip **GDDR6 DRAM** striped across banks. L1 replaces the GPU's shared-memory + register hierarchy; you place data explicitly.
- All data is **tilized** to 32×32 tiles (a tile subdivides into four 16×16 faces because the matrix engine multiplies 16×16 natively). A bf16 tile = 2048 bytes.

**Generations & ceilings** (single-card):

| | Tensix (compute)† | L1/core | DRAM | DRAM BW | Peak matmul (by dtype)‡ | Clock |
|---|---|---|---|---|---|---|
| Wormhole (n150) | ~64 (8×8) | ~1.5 MB | 12 GB, 12 banks | 288 GB/s (336 @14Gbps) | ~50 (bf16) / ~190 (bfp4) TFLOPS | 1.0 GHz |
| Blackhole (p150) | ~120 (up to 13×10) | ~1.5 MB | 32 GB, 8 banks | ~512 GB/s | 332 (bf16) / 664 (bfp8) TFLOPS | 1.35 GHz |

† Core counts are nominal/marketed; the **usable** Tensix grid varies with harvesting — query `device.compute_with_storage_grid_size()` at runtime rather than hard-coding it.
‡ Matmul peak depends on data format **and** math fidelity (bf16 « bfp8 « bfp4 / LoFi). Compare achieved TFLOPS against the peak *for the dtype you actually run* — the high aggregate figures (e.g. WH ~190) are bfp4/LoFi, not bf16.

Grayskull is **discontinued** — do not target it.
Refs: [Blackhole](https://github.com/tenstorrent/tt-metal/tree/main/tech_reports/Blackhole), [matrix_engine](https://github.com/tenstorrent/tt-metal/tree/main/tech_reports/matrix_engine), [GEMM_FLOPS](https://github.com/tenstorrent/tt-metal/tree/main/tech_reports/GEMM_FLOPS), [Saturating_DRAM_bandwidth](https://github.com/tenstorrent/tt-metal/tree/main/tech_reports/Saturating_DRAM_bandwidth).

## Correctness — PCC (not tight allclose)

The standard TT correctness metric is **PCC (Pearson Correlation Coefficient)** against a torch golden, because bf16/bfloat8_b outputs diverge element-wise from fp32 while staying highly correlated — a tight `torch.allclose(rtol=1e-5, atol=1e-8)` is physically unreachable and always fails on a correct low-precision kernel.

- Helpers in tt-metal: `comp_pcc(golden, calc, pcc=0.99)` → `(passing, value)` (from `models.common.utility_functions`); `assert_with_pcc(expected, actual, pcc=0.9999)` (from `tests.ttnn.utils_for_testing`).
- Typical thresholds: **0.9999** strict (fp32/bf16, `assert_with_pcc` default), **0.999** typical, **0.99** for bfloat8_b / aggressive low-precision.
- PCC blind spot: a global scale/bias (output ×2) can still give PCC ≈ 1 — pair with a magnitude/allclose sanity check when it matters.
- Accuracy levers: `fp32_dest_acc_en` (FP32 accumulation in DST), `UnpackToDestFp32` (fp32 intermediates), higher `math_fidelity`; sum-then-divide, Welford for mean/variance, reduce over `logical_shape()` not `padded_shape()`.

Ref: [op_kernel_dev/accuracy_tips](https://github.com/tenstorrent/tt-metal/tree/main/tech_reports/op_kernel_dev/accuracy_tips), [Handling_Special_Value](https://github.com/tenstorrent/tt-metal/tree/main/tech_reports/Handling_Special_Value).

## Profiling — the tt-metal device profiler (the `ncu` replacement)

Tracy-based, built into tt-metal.

- **Build** with the profiler — Tracy is **on by default**, so a plain `./build_metal.sh` includes it (`--disable-profiler` turns it off); tools land in `build/tools/profiler/bin/`. **Runtime gate:** `export TT_METAL_DEVICE_PROFILER=1` (off by default — readback adds overhead).
- **Per-op device latency:** `cd $TT_METAL_HOME && ./tools/tracy/profile_this.py -n <name> -c "pytest <test>"` (or `python -m tracy -m pytest <test>`) → `generated/profiler/reports/ops_perf_results_<ts>.csv`. Key columns (ns): `DEVICE KERNEL DURATION` (primary), `DEVICE FW DURATION` (fixed overhead), `HOST DURATION` (dispatch), per-RISC `DEVICE {BRISC,NCRISC,TRISC0/1/2} KERNEL DURATION`, `DEVICE COMPUTE CB WAIT FRONT` (compute starved) / `CB RESERVE BACK` (compute back-pressured).
- **Bottleneck metrics** (the `ncu --metrics` equivalent): run under `python -m tracy --profiler-capture-perf-counters=all` (groups: `fpu,pack,unpack,l1_0,instrn`; `all` for everything). Triage: **MATH/FPU/SFPU Util %** (compute-bound?), **Thread 0/1/2 Stall Rate %** (T0=unpack starvation, T1=math, T2=pack), **NOC vs Compute Balance %** (>60% NOC-bound), **Compute-to-Unpack Ratio** (<20% memory-bound), **L1 Total Bandwidth Util %**, **Fidelity Stall Rate** (HiFi cost).
- **Timeline:** Tracy WASM web viewer (auto-starts on `python -m tracy`, HTTP 8080) or Tracy GUI (port 8086).
- **Do not** enable DPRINT / Watcher while profiling or timing — they perturb latency and conflict with `TT_METAL_DEVICE_PROFILER`.

Refs: [MetalProfiler](https://github.com/tenstorrent/tt-metal/tree/main/tech_reports/MetalProfiler), [PerfCounters](https://github.com/tenstorrent/tt-metal/tree/main/tech_reports/PerfCounters), [real_time_profiler](https://github.com/tenstorrent/tt-metal/tree/main/tech_reports/real_time_profiler).

## Timing an op from Python (host wall-clock)

```python
import time, ttnn
device = ttnn.open_device(device_id=0)
# warmup — first run JIT-compiles kernels + fills the program cache (discard it)
out = model(x); ttnn.synchronize_device(device)
t0 = time.perf_counter(); out = model(x); ttnn.synchronize_device(device)
latency_ms = (time.perf_counter() - t0) * 1000
ttnn.close_device(device)
```

`ttnn.synchronize_device` is **mandatory** — dispatch is asynchronous, so without it you time only host enqueue. Always discard the first (compile) run.

## Optimization levers (roughly by impact)

1. **Data format + math fidelity** — bf16 → **bfloat8_b** (HiFi2, ~1.5–1.8×) → **bfloat4_b** (LoFi, ~2–3.5×). Also halves/quarters DRAM & L1 traffic. Match fidelity to format (bfloat8_b→HiFi2, bf16→HiFi4). Re-check PCC after each downgrade. Ref: [data_formats](https://github.com/tenstorrent/tt-metal/tree/main/tech_reports/data_formats).
2. **Memory placement + sharding** — keep hot tensors in **L1** (`ttnn.L1_MEMORY_CONFIG`); shard to match access (`HEIGHT` row-wise, `WIDTH` column-wise, `BLOCK` matmul). For DRAM-bound reads, sharded + one reader per bank saturates bandwidth. Ref: [tensor_sharding](https://github.com/tenstorrent/tt-metal/tree/main/tech_reports/tensor_sharding), [memory](https://github.com/tenstorrent/tt-metal/tree/main/tech_reports/memory).
3. **Layout** — keep compute inputs in `TILE_LAYOUT` (32×32); avoid `tilize`/`untilize` (`ttnn.to_layout`) round-trips; dims multiples of 32. Ref: [tensor_layouts](https://github.com/tenstorrent/tt-metal/tree/main/tech_reports/tensor_layouts).
4. **Core-grid occupancy** — use more Tensix cores (`core_grid` / `CoreRangeSet`; ~64 on WH n150, ~120 on BH — query `device.compute_with_storage_grid_size()` at runtime, since harvesting varies it). Balance tiles across the grid.
5. **Circular-buffer double-buffering** — size CBs to hold >1 tile so the reader prefetches while compute consumes; split reader (RISCV_0) / writer (RISCV_1) on separate NoCs.
6. **Metal Trace + multi-CQ + program cache** — `ttnn.begin_trace_capture`/`execute_trace` removes per-iteration host dispatch overhead; `num_command_queues=2` overlaps IO with compute; program cache eliminates recompiles. Biggest wins on small ops. Ref: [AdvancedPerformanceOptimizationsForModels](https://github.com/tenstorrent/tt-metal/tree/main/tech_reports/AdvancedPerformanceOptimizationsForModels).
7. **Kernel fusion / matmul tiling** — fuse elementwise chains in the DST register (no CB round-trip); tune matmul `per_core_M/N`, `in0_block_w`, `out_subblock_h/w`; `fp32_dest_acc_en` only where accuracy needs it.

## Physical floors (for the "when to stop" decision)

- **Memory-bound:** achieved GB/s = bytes moved / kernel time; near the DRAM ceiling (WH 288–336, BH ~512 GB/s) → done. Well-tuned readers hit >90%.
- **Compute-bound:** achieved TFLOPS = 2·M·N·K / time; near peak **for the dtype in use** (WH ~50 bf16 / ~190 bfp4; BH 332 bf16 / 664 bfp8) → done. Per-op utilization = ideal_cycles / actual_cycles.
- **L1-resident** with no DRAM spills, or **dispatch overhead** dominating a small op after Metal Trace + multi-CQ, are also floors.

## Two kernel layers

- **TT-NN (ttnn)** — PyTorch-like Python op library. `ttnn.open_device`, `ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=DEVICE)`, `ttnn.matmul`/`ttnn.add`/…, `ttnn.to_torch`. Optimize by op choice, dtype, layout, memory config / sharding, core grid, math fidelity, trace. Ref: [ttnn](https://github.com/tenstorrent/tt-metal/tree/main/tech_reports/ttnn).
- **tt-metal** — low-level C++ Tensix kernels. Host builds a `Program`, allocates DRAM/L1 buffers + circular buffers, `CreateKernel(program, "reader.cpp"/"compute.cpp"/"writer.cpp", core, {Reader,Compute,Writer}Config{...})`, `SetRuntimeArgs`, `EnqueueProgram`/`Finish`. Kernel side: `noc_async_read/write` + barriers (dataflow); `cb_wait_front`/`cb_reserve_back`/`cb_push_back`/`cb_pop_front`, `matmul_tiles`, SFPU `*_tile` ops, `tile_regs_acquire/commit/wait/release`, `pack_tile` (compute). Ref: [prog_examples](https://github.com/tenstorrent/tt-metal/tree/main/tech_reports/prog_examples), [ttnn_operators](https://github.com/tenstorrent/tt-metal/tree/main/tech_reports/ttnn_operators).

## Debugging

- **DPRINT** — `#include "api/debug/dprint.h"`, enable `TT_METAL_DPRINT_CORES="(0,0)"`; `TSLICE()` / `print_bf16_pages` dump CB tiles. Every print ends with `\n`.
- **Watcher** — `TT_METAL_WATCHER=<poll-seconds>` → `generated/watcher/watcher.log`; NoC/CB-bounds/stack sanitizer + per-RISC last waypoint on a hang.
- Reduce a bad kernel: single-op test, smaller core grid, program cache off, fixed zeros/ones inputs.

Ref: [Debugging](https://github.com/tenstorrent/tt-metal/tree/main/tech_reports/Debugging).
