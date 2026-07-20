# Iteration Log

<!--
Per-iteration template (copy when adding a new iter entry under "## Iterations"):

### Iter N — Short title

- **Hypothesis:** Why this change is expected to help (name the lever: dtype /
  math fidelity / layout / sharding / core grid / CB depth / trace / fusion)
- **Changes:** What was modified
- **Bench:**
  - Compiled: True/False
  - Correct: True/False   (PCC ___ vs threshold ___)
  - Runtime: ___ ms (mean), ___ ~ ___ ms (min ~ max)   [solution, on device]
  - Speedup: ___x   [vs CPU reference; primary signal is Runtime, lower is better]
- **Profiler (if run):** DEVICE KERNEL DURATION, dominant bottleneck (MATH util /
  thread stall / NOC-bound / L1 BW / CB wait), or roofline note (achieved GB/s or
  TFLOPS vs ceiling)
- **Analysis:** Why it worked or failed
- **Next:** What to try next

Append one row per iter to the Summary table below.
Status values: improved / no-change / regression / failed.
-->

## Summary

| Iter | Title | Runtime(mean, ms) | PCC | Speedup | Status |
|------|-------|-------------------|-----|---------|--------|

## Iterations
