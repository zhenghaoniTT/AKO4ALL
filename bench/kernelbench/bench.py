#!/usr/bin/env python3
"""
Self-contained Tenstorrent kernel benchmark for AKO4ALL.

Evaluates an optimized Tenstorrent kernel (solution) against a reference
implementation. The reference is a plain PyTorch `class Model` that runs on the
**CPU** and serves as the numerical golden; the solution runs on a Tenstorrent
device (TT-NN ops and/or tt-metal Tensix kernels) and is compared with the
Pearson Correlation Coefficient (PCC) — the standard TT correctness metric —
rather than a tight `torch.allclose`, because bfloat16 / bfloat8_b outputs
diverge element-wise from an fp32 golden while staying highly correlated.

Timing is host wall-clock around the solution's `forward`, bracketed by
`ttnn.synchronize_device(device)` (TT dispatch is asynchronous, so without the
sync you would time only host enqueue). The first run is discarded because it
JIT-compiles the kernels and populates the program cache; steady-state numbers
come from the cached runs. For per-kernel device timing and bottleneck metrics
use the tt-metal device profiler (Tracy) separately — see GUIDE.md.

Usage:
    python3 bench/kernelbench/bench.py --ref <ref-path> --solution solution/<kernel> [options]

Output (structured, one per line):
    COMPILED: True/False
    CORRECT: True/False
    BACKEND: ttnn|tt-metal (auto|explicit)
    PCC: <value>
    RUNTIME: <ms>          # solution, on the TT device (end-to-end incl. transfer)
    REF_RUNTIME: <ms>      # reference, on CPU (torch)
    SPEEDUP: <x>           # REF_RUNTIME / RUNTIME

The KernelBench-format contract (class Model + get_inputs / get_init_inputs) and
the anti-cheat source transforms are inherited from KernelBench.

Portions of this file are derived from KernelBench
(https://github.com/ScalingIntelligence/KernelBench).

Copyright (c) 2023 Anne Ouyang, Simon Guo, Azalia Mirhoseini

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import argparse
import copy
import importlib
import importlib.util
import io
import math
import os
import re
import statistics
import sys
import tempfile
import time
import tokenize
import traceback
from dataclasses import dataclass, field
from typing import Any, Optional

# SPEEDUP compares the TT solution to a CPU reference, so large values are often
# legitimate — but an implausibly high ratio is worth a second look for reward
# hacking (a solution that doesn't really run on device). Informational only.
EXCESSIVE_SPEEDUP_THRESHOLD = 100.0

import torch
import torch.nn as nn

# --- Trust boundary note -----------------------------------------------------
# The solution is UNTRUSTED code that is imported (its module-level code runs)
# in this same interpreter. Defense-in-depth here: (1) the reference goldens are
# computed BEFORE the solution is imported; (2) the timer and the torch ops used
# to score correctness are captured below, before any solution code runs, so a
# solution that monkey-patches `time.perf_counter` / `torch.*` after import
# cannot redirect them; (3) correctness is re-verified after the timing phase.
# This does NOT make execution safe against a hostile solution (method-level
# patching, os-level actions, etc. remain possible) — for untrusted input run
# the bench inside a container/sandbox. A full guarantee needs subprocess
# isolation of the solution's forward.
_perf_counter = time.perf_counter
_t_corrcoef = torch.corrcoef
_t_nan_to_num = torch.nan_to_num
_t_isclose = torch.isclose
_t_equal = torch.equal
_t_all = torch.all
_t_isnan = torch.isnan
_t_isinf = torch.isinf
_t_stack = torch.stack
_t_vecnorm = torch.linalg.vector_norm


###############################################################################
# Device handle injected into solutions
###############################################################################
#
# A ttnn device is expensive to open and cannot be opened twice, so the bench
# opens ONE device and hands it to the solution instead of letting the solution
# open its own. It is injected two ways so a solution can use whichever it
# prefers:
#   1. as the module-level global ``DEVICE`` in the solution's namespace, and
#   2. via ``ttnn.SetDefaultDevice(device)`` (when that API is present) so ops
#      that fall back to the implicit default device also work.
# Solutions should place tensors with ``ttnn.from_torch(t, ..., device=DEVICE)``.


def _to_torch(x: Any) -> Any:
    """Convert a ttnn.Tensor back to a torch.Tensor; pass everything else through.

    Solutions normally return a torch tensor (having called ttnn.to_torch
    themselves), but tolerate a solution that returns the raw ttnn.Tensor.
    """
    if isinstance(x, torch.Tensor):
        return x
    # Duck-type a ttnn.Tensor without importing ttnn at module load time.
    if x.__class__.__module__.startswith("ttnn"):
        import ttnn

        return ttnn.to_torch(x)
    return x


###############################################################################
# Correctness — Pearson Correlation Coefficient (PCC)
###############################################################################


def comp_pcc(golden: torch.Tensor, calculated: torch.Tensor, pcc: float = 0.99):
    """Pearson Correlation Coefficient between two tensors — the standard TT
    correctness metric (see tt-metal ``models.common.utility_functions.comp_pcc``).

    Returns ``(passing: bool, pcc_value: float)``. NaN/Inf entries are masked to
    zero before correlating; constant tensors are compared on their mean value;
    identical tensors and all-NaN-vs-all-NaN return 1.0.

    PCC (not a tight allclose) is used because TT kernels run in bfloat16 /
    bfloat8_b, whose few mantissa bits make element-wise agreement to fp32-grade
    tolerance physically impossible even for a correct kernel.

    Uses the torch functions captured at module import (``_t_*``) so a solution
    that monkey-patches ``torch.*`` after being imported cannot corrupt scoring.
    """
    golden = golden.detach().flatten().to(torch.float32)
    calculated = calculated.detach().flatten().to(torch.float32)

    if golden.shape != calculated.shape:
        return False, 0.0

    g_all_nan = bool(_t_all(_t_isnan(golden)))
    c_all_nan = bool(_t_all(_t_isnan(calculated)))
    if g_all_nan and c_all_nan:
        return True, 1.0
    if g_all_nan or c_all_nan:
        return False, 0.0

    if _t_equal(golden, calculated):
        return True, 1.0

    # Zero each tensor's OWN non-finite entries independently (matches tt-metal's
    # per-tensor nan_to_num). A single union mask applied to both would also zero
    # the golden wherever the *solution* emitted NaN/Inf, hiding the divergence
    # and inflating PCC to a false pass — a NaN-producing kernel would score
    # correct.
    golden = _t_nan_to_num(golden, nan=0.0, posinf=0.0, neginf=0.0)
    calculated = _t_nan_to_num(calculated, nan=0.0, posinf=0.0, neginf=0.0)

    # Pearson correlation is undefined when a tensor has zero variance. Shapes
    # already match here, so numel<2 means both are scalars.
    g_const = golden.numel() < 2 or golden.std().item() == 0.0
    c_const = calculated.numel() < 2 or calculated.std().item() == 0.0
    if g_const or c_const:
        if g_const and c_const:
            # Both (near-)constant: compare the value with a precision-aware
            # loose tolerance — NOT torch.isclose defaults (rtol=1e-5/atol=1e-8),
            # which are unreachable in bf16 and would false-fail a correct
            # low-precision kernel.
            close = bool(
                _t_isclose(golden.mean(), calculated.mean(), rtol=2e-2, atol=1e-2)
            )
            return close, (1.0 if close else 0.0)
        # Exactly one side is constant while the other varies — a genuine
        # mismatch (e.g. a solution returning a constant vs a varying golden).
        return False, 0.0

    cc = _t_corrcoef(_t_stack([golden, calculated]))[0, 1].item()
    if math.isnan(cc):
        return False, 0.0
    return cc >= pcc, float(cc)


def relative_l2_error(golden: torch.Tensor, calculated: torch.Tensor) -> float:
    """Whole-tensor relative L2 error ``||golden - calc|| / ||golden||``.

    A loose magnitude guard meant to run ALONGSIDE PCC, not instead of it. PCC
    is invariant to a global scale/bias — a kernel returning ``2*output`` still
    scores PCC ≈ 1 — so PCC alone cannot reject magnitude errors. Relative L2
    flags them (a 2× scale → ≈ 1.0) while staying small for bf16 / bfloat8_b
    rounding noise (typically a few %), so the two together catch a broad class
    of wrong-but-correlated outputs.
    """
    g = _t_nan_to_num(golden.detach().flatten().to(torch.float32))
    c = _t_nan_to_num(calculated.detach().flatten().to(torch.float32))
    if g.shape != c.shape:
        return float("inf")
    denom = _t_vecnorm(g).item()
    num = _t_vecnorm(g - c).item()
    if denom == 0.0:
        # golden is all-zero: fall back to the absolute error magnitude.
        return num
    return num / denom


###############################################################################
# Timing
###############################################################################


def time_execution_host(
    kernel_fn: callable,
    make_args: callable,
    device: Any,
    num_warmup: int = 3,
    num_trials: int = 100,
    discard_first: int = 1,
    verbose: bool = True,
) -> list[float]:
    """Time a callable with a host wall-clock, synchronizing the TT device each
    trial. Returns a list of elapsed times in milliseconds.

    ``make_args()`` returns a fresh argument list per trial (fresh inputs are
    part of the anti-cheat and also avoid measuring cache-warm-on-identical-data
    effects). ``device`` may be None (CPU reference), in which case no device
    sync is performed. The first ``discard_first`` trials are dropped; on TT the
    very first run JIT-compiles kernels and fills the program cache.

    The output is passed through ``_to_torch`` INSIDE the timed region: if the
    solution returns a raw ttnn.Tensor, the device->host readback is part of the
    reported end-to-end RUNTIME (and a solution can't dodge the transfer cost by
    skipping to_torch). Uses the module-captured timer ``_perf_counter``.
    """
    ttnn = None
    if device is not None:
        import ttnn  # noqa: F401  (lazy — self-test / CPU path needs no ttnn)

    def _sync():
        if ttnn is not None:
            ttnn.synchronize_device(device)

    # Warm up: triggers kernel JIT build + program-cache population on TT.
    for _ in range(num_warmup):
        _to_torch(kernel_fn(*make_args()))
        _sync()

    if verbose and device is not None:
        print(f"[Profiling] Host timing on TT device, warmup {num_warmup}, trials {num_trials}")

    elapsed_times: list[float] = []
    sink = None  # keep a reference so the call isn't dead-code-eliminated
    for trial in range(num_trials + discard_first):
        args = make_args()
        _sync()
        start = _perf_counter()
        sink = _to_torch(kernel_fn(*args))
        _sync()
        end = _perf_counter()

        elapsed_ms = (end - start) * 1000.0
        if trial >= discard_first:
            if verbose:
                print(f"Trial {trial - discard_first + 1}: {elapsed_ms:.3g} ms")
            elapsed_times.append(elapsed_ms)

    del sink
    return elapsed_times


def get_timing_stats(elapsed_times: list[float]) -> dict:
    """Compute mean/std/min/max from a list of elapsed times (ms)."""
    mean_val = statistics.mean(elapsed_times)
    std_val = statistics.stdev(elapsed_times) if len(elapsed_times) > 1 else 0.0
    return {
        "mean": float(f"{mean_val:.3g}"),
        "std": float(f"{std_val:.3g}"),
        "min": float(f"{min(elapsed_times):.3g}"),
        "max": float(f"{max(elapsed_times):.3g}"),
        "num_trials": len(elapsed_times),
    }


###############################################################################
# Eval scaffolding
###############################################################################


@dataclass
class KernelExecResult:
    """Result of a single kernel evaluation."""

    compiled: bool = False
    correctness: bool = False
    metadata: dict = field(default_factory=dict)
    pcc: float = -1.0
    runtime: float = -1.0  # ms (solution, on device)
    runtime_stats: dict = field(default_factory=dict)
    ref_runtime: float = -1.0  # ms (reference, on CPU)
    ref_runtime_stats: dict = field(default_factory=dict)


def set_seed(seed: int):
    torch.manual_seed(seed)


def get_error_name(e: Exception) -> str:
    return f"{e.__class__.__module__}.{e.__class__.__name__}"


def load_original_model_and_inputs(
    model_original_src: str, context: dict, source_path: Optional[str] = None
) -> tuple:
    """exec() the reference source. Returns (Model, get_init_inputs, get_inputs).

    If source_path is given it is injected as __file__ so the reference can find
    sibling files via os.path.dirname(__file__).
    """
    if source_path is not None:
        context["__file__"] = os.path.abspath(source_path)

    try:
        compile(model_original_src, "<string>", "exec")
    except SyntaxError as e:
        print(f"Syntax Error in original code {e}")
        return None

    try:
        exec(model_original_src, context)
    except Exception as e:
        print(f"Error in executing original code {e}")
        return None

    return (
        context.get("Model"),
        context.get("get_init_inputs"),
        context.get("get_inputs"),
    )


def load_solution_model(
    model_custom_src: str,
    device: Any,
    source_path: Optional[str] = None,
    entry_point: str = "ModelNew",
):
    """Load ModelNew from solution source via importlib.

    The transformed source is written to a temp file **in the solution's own
    directory** so that a tt-metal solution referencing sibling kernel files
    (e.g. `os.path.dirname(__file__) + "/kernels/reader.cpp"`) still resolves
    them. The opened TT `device` is injected as the module global ``DEVICE``.

    Returns (ModelNew, temp_path). Caller removes temp_path.
    """
    sol_dir = (
        os.path.dirname(os.path.abspath(source_path))
        if source_path
        else tempfile.gettempdir()
    )
    fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix=".ako_eval_", dir=sol_dir)
    with os.fdopen(fd, "w") as f:
        f.write(model_custom_src)

    spec = importlib.util.spec_from_file_location("ako_solution_module", tmp_path)
    module = importlib.util.module_from_spec(spec)
    # Inject the device before executing module-level code so top-level setup can
    # use it too.
    setattr(module, "DEVICE", device)
    # Don't let the import write a __pycache__/*.pyc beside the (uniquely-named,
    # about-to-be-deleted) temp file: cleanup only removes the .py, so the .pyc
    # would orphan and accumulate in the user's solution/ dir and get copied into
    # every trajectory snapshot. Disabling covers both the success and error paths.
    _prev_dwb = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    except Exception:
        # Surface the import/compile failure to the caller; clean up the temp file.
        os.remove(tmp_path)
        raise
    finally:
        sys.dont_write_bytecode = _prev_dwb

    return getattr(module, entry_point, None), tmp_path


def graceful_eval_cleanup(temp_path: Optional[str] = None):
    if temp_path and os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except OSError:
            pass


def register_and_format_exception(
    exception_type: str,
    exception_msg,
    metadata: dict,
    verbose: bool = False,
    truncate=False,
    max_length=200,
):
    exception_str = str(exception_msg)
    if truncate and len(exception_str) > max_length:
        exception_str = exception_str[: max_length - 3] + "..."
    if verbose:
        print(f"[Exception {exception_type}] {exception_str}")
    metadata[exception_type] = exception_str
    return metadata


def _clone_inputs(inputs):
    """Deep copy an inputs list so a consumer cannot mutate the original."""
    return [
        x.clone() if isinstance(x, torch.Tensor) else copy.deepcopy(x) for x in inputs
    ]


def _compare_output(ref_cpu, sol_output, pcc_threshold, rel_tol):
    """Compare a solution output against a precomputed CPU golden.

    Returns (passing, pcc_val, rel_err, reasons). Correctness = PCC gate AND
    relative-L2 magnitude gate: PCC alone is blind to a global scale/bias
    (output×2 scores PCC≈1); relative L2 catches that while tolerating
    bf16/bfloat8_b rounding noise.
    """
    sol_cpu = _to_torch(sol_output).detach().to(torch.float32).cpu()
    if ref_cpu.shape != sol_cpu.shape:
        return False, 0.0, float("inf"), [
            f"shape mismatch: expected {ref_cpu.shape}, got {sol_cpu.shape}"
        ]
    passing_pcc, pcc_val = comp_pcc(ref_cpu, sol_cpu, pcc_threshold)
    rel_err = relative_l2_error(ref_cpu, sol_cpu)
    reasons = []
    if not passing_pcc:
        reasons.append(f"PCC {pcc_val:.6f} < {pcc_threshold}")
    if rel_err > rel_tol:
        reasons.append(f"rel_err {rel_err:.4f} > {rel_tol}")
    return (passing_pcc and rel_err <= rel_tol), pcc_val, rel_err, reasons


def run_and_check_correctness(
    new_model_instance: nn.Module,
    golden_trials: list,
    metadata: dict,
    pcc_threshold: float,
    rel_tol: float = 0.1,
    verbose: bool = False,
) -> KernelExecResult:
    """Score the solution (TT) against precomputed CPU goldens with PCC + a
    relative-L2 magnitude guard.

    ``golden_trials`` is a list of ``(sol_inputs, ref_cpu_output)`` pairs computed
    by the reference on CPU BEFORE the untrusted solution was imported (so a
    solution that monkey-patches torch cannot corrupt the golden). The checks run
    on fresh random inputs per trial — the anti-cheat: a constant / precomputed /
    scaled output fails.
    """
    pass_count = 0
    min_pcc = 1.0
    max_rel_err = 0.0
    n = len(golden_trials)

    with torch.no_grad():
        for trial, (sol_inputs, ref_cpu) in enumerate(golden_trials):
            try:
                sol_output = new_model_instance(*_clone_inputs(sol_inputs))
            except Exception as e:
                print("[Error] Exception during correctness check")
                print(f"Error running solution ModelNew: {e}")
                print("\n[Full Traceback]:")
                traceback.print_exc()
                print()
                register_and_format_exception("runtime_error", e, metadata, truncate=True)
                metadata["runtime_error_name"] = get_error_name(e)
                metadata["runtime_error_traceback"] = traceback.format_exc()
                return KernelExecResult(compiled=True, correctness=False, metadata=metadata)

            passing, pcc_val, rel_err, reasons = _compare_output(
                ref_cpu, sol_output, pcc_threshold, rel_tol
            )
            min_pcc = min(min_pcc, pcc_val)
            max_rel_err = max(max_rel_err, rel_err)
            if passing:
                pass_count += 1
                if verbose:
                    print(
                        f"[PASS] trial {trial}: PCC {pcc_val:.6f} >= {pcc_threshold}, "
                        f"rel_err {rel_err:.4f} <= {rel_tol}"
                    )
            else:
                metadata.setdefault("pcc_values", []).append(f"{pcc_val:.6f}")
                metadata["correctness_issue"] = "; ".join(reasons)
                if verbose:
                    print(f"[FAIL] trial {trial}: {'; '.join(reasons)}")

    metadata["correctness_trials"] = f"({pass_count} / {n})"
    metadata["min_pcc"] = f"{min_pcc:.6f}"
    metadata["max_rel_err"] = f"{max_rel_err:.6f}"
    return KernelExecResult(
        compiled=True, correctness=(pass_count == n), pcc=min_pcc, metadata=metadata
    )


def eval_kernel_against_ref(
    original_model_src: str,
    custom_model_src: str,
    seed_num: int = 42,
    num_correct_trials: int = 5,
    num_perf_trials: int = 100,
    measure_performance: bool = True,
    pcc_threshold: float = 0.99,
    rel_tol: float = 0.1,
    verbose: bool = False,
    device_id: int = 0,
    backend: str = "ttnn",
    measure_reference: bool = True,
    get_inputs_override: Optional[callable] = None,
    get_init_inputs_override: Optional[callable] = None,
    ref_path: Optional[str] = None,
    sol_path: Optional[str] = None,
) -> Optional[KernelExecResult]:
    """Evaluate a solution kernel on a Tenstorrent device against a CPU golden.

    Opens one TT device (via ttnn), loads the reference (CPU) and solution (TT),
    checks PCC correctness over fresh-input trials, and times the solution on the
    device (host wall-clock + ttnn.synchronize_device, first run discarded).

    ``measure_reference=False`` skips the CPU reference timing (REF_RUNTIME /
    SPEEDUP left at -1): the reference is invariant across solution edits, so for
    fast per-iteration signal rank candidates by the solution's own RUNTIME.
    """
    try:
        import ttnn
    except Exception as e:  # pragma: no cover - depends on the host
        print(
            "[Eval] Could not import ttnn — a Tenstorrent (tt-metal/TT-NN) install "
            "is required. Install tt-metal and set PYTHONPATH/TT_METAL_HOME.\n"
            f"Error: {e}"
        )
        metadata = {"error": "ttnn_import_failed", "error_message": str(e)}
        return KernelExecResult(compiled=False, metadata=metadata)

    metadata: dict = {"backend": backend}

    # --- Open the device ---
    try:
        device = ttnn.open_device(device_id=device_id)
    except Exception as e:
        print(f"[Eval] Failed to open Tenstorrent device {device_id}: {e}")
        return KernelExecResult(
            compiled=False,
            metadata={"error": "device_open_failed", "error_message": str(e)},
        )

    # Program cache + default device are best-effort conveniences; ttnn versions
    # differ on whether these exist / are needed (newer builds auto-cache).
    for fn in ("enable_program_cache",):
        try:
            getattr(device, fn)()
        except Exception:
            pass
    try:
        ttnn.SetDefaultDevice(device)
    except Exception:
        pass

    metadata["device"] = f"tt:{device_id}"
    try:
        metadata["arch"] = os.environ.get("ARCH_NAME", "unknown")
    except Exception:
        pass

    temp_path = None
    try:
        # --- Load reference (CPU golden) ---
        # The reference is a plain PyTorch model evaluated on the CPU — the fp32
        # golden. Deliberately do NOT inject DEVICE here: the reference must not
        # touch the TT device (that keeps its timing sync-free and the contract
        # unambiguous — only the solution runs on device).
        if verbose:
            print("[Eval] Loading reference model (CPU golden)")
        ref_context: dict = {}
        loaded = load_original_model_and_inputs(
            original_model_src, ref_context, source_path=ref_path
        )
        if loaded is None:
            metadata["error"] = "reference_load_failed"
            return KernelExecResult(compiled=False, metadata=metadata)
        Model, ref_get_init_inputs, ref_get_inputs = loaded

        get_inputs = get_inputs_override if get_inputs_override is not None else ref_get_inputs
        get_init_inputs = (
            get_init_inputs_override
            if get_init_inputs_override is not None
            else ref_get_init_inputs
        )

        if get_inputs is None:
            msg = (
                "get_inputs() not found. Define it in the reference file or pass "
                "--inputs <file> with a top-level get_inputs()."
            )
            print(f"[Eval] {msg}")
            metadata["error"] = "missing_get_inputs"
            metadata["error_message"] = msg
            return KernelExecResult(compiled=False, metadata=metadata)

        try:
            set_seed(seed_num)
            init_inputs = [] if get_init_inputs is None else get_init_inputs()
            with torch.no_grad():
                set_seed(seed_num)
                original_model = Model(*init_inputs)
                assert hasattr(original_model, "forward")
        except Exception as e:
            print(f"Failed to construct reference model: {e}")
            metadata["error"] = "reference_instantiation_failed"
            metadata["error_message"] = str(e)
            return KernelExecResult(compiled=False, metadata=metadata)

        # --- Precompute reference goldens on CPU, BEFORE importing the untrusted
        # solution, so a solution that monkey-patches torch cannot corrupt the
        # golden. Generates num_correct_trials + 1 trials; the extra "holdout"
        # is used to re-verify after the timing phase. A reference-side failure
        # is recorded under its own key (reference_error), never the solution's. ---
        if verbose:
            print("[Eval] Precomputing reference goldens (CPU)")
        torch.manual_seed(seed_num)
        n_gold = num_correct_trials + 1
        trial_seeds = [torch.randint(0, 2**31 - 1, (1,)).item() for _ in range(n_gold)]
        golden_trials = []
        try:
            with torch.no_grad():
                for ts in trial_seeds:
                    set_seed(ts)
                    inp = get_inputs()
                    sol_inp = _clone_inputs(inp)  # pristine copy for the solution
                    ref_out = original_model(*inp)
                    if not isinstance(ref_out, torch.Tensor):
                        metadata["error"] = "reference_not_tensor"
                        metadata["error_message"] = "Reference forward did not return a tensor"
                        return KernelExecResult(compiled=False, metadata=metadata)
                    golden_trials.append(
                        (sol_inp, ref_out.detach().to(torch.float32).cpu())
                    )
        except Exception as e:
            print(f"Reference forward failed while computing golden: {e}")
            register_and_format_exception("reference_error", e, metadata, truncate=True)
            metadata["reference_error_name"] = get_error_name(e)
            return KernelExecResult(compiled=False, metadata=metadata)

        # --- Load + build solution (TT) ---
        if verbose:
            print("[Eval] Loading solution model (TT device)")
        try:
            ModelNew, temp_path = load_solution_model(
                custom_model_src, device, source_path=sol_path
            )
        except SyntaxError as e:
            metadata["compilation_error_name"] = "SyntaxError"
            metadata["compilation_error"] = str(e)
            return KernelExecResult(compiled=False, metadata=metadata)
        except Exception as e:
            print(f"Failed to load/compile solution: {e}")
            metadata["compilation_error_name"] = get_error_name(e)
            metadata["compilation_error"] = str(e)
            return KernelExecResult(compiled=False, metadata=metadata)

        if ModelNew is None:
            metadata["compilation_error_name"] = "MissingModelNew"
            metadata["compilation_error"] = "ModelNew not found in solution"
            return KernelExecResult(compiled=False, metadata=metadata)

        try:
            with torch.no_grad():
                set_seed(seed_num)
                custom_model = ModelNew(*init_inputs)
                assert hasattr(custom_model, "forward")
        except Exception as e:
            print(f"Failed to instantiate ModelNew: {e}")
            metadata["runtime_error"] = str(e)
            metadata["runtime_error_name"] = get_error_name(e)
            return KernelExecResult(compiled=True, correctness=False, metadata=metadata)

        # --- Correctness (PCC) ---
        if verbose:
            print("[Eval] Checking correctness (PCC)")
        try:
            result = run_and_check_correctness(
                custom_model,
                golden_trials[:num_correct_trials],
                metadata=metadata,
                pcc_threshold=pcc_threshold,
                rel_tol=rel_tol,
                verbose=verbose,
            )
        except Exception as e:
            metadata["runtime_error"] = str(e)
            metadata["runtime_error_name"] = get_error_name(e)
            result = KernelExecResult(compiled=True, correctness=False, metadata=metadata)

        # --- Performance (solution on device) ---
        if measure_performance and result.correctness:
            if verbose:
                print("[Eval] Measuring solution performance on device")
            try:
                def make_sol_args():
                    return get_inputs()

                elapsed = time_execution_host(
                    custom_model,
                    make_sol_args,
                    device,
                    num_trials=num_perf_trials,
                    verbose=verbose,
                )
                stats = get_timing_stats(elapsed)
                result.runtime = stats["mean"]
                result.runtime_stats = stats
                if verbose:
                    print(f"[Eval] Solution runtime stats: {stats}")
            except Exception as e:
                if verbose:
                    print(f"[Eval] Error measuring solution performance: {e}")
                result.metadata["error_during_performance"] = str(e)

        # --- Reference timing (CPU) for a speedup denominator ---
        if measure_performance and measure_reference and result.correctness:
            if verbose:
                print("[Eval] Measuring reference performance on CPU")
            try:
                def make_ref_args():
                    return get_inputs()

                ref_elapsed = time_execution_host(
                    original_model,
                    make_ref_args,
                    None,  # CPU — no device sync
                    num_trials=num_perf_trials,
                    verbose=verbose,
                )
                ref_stats = get_timing_stats(ref_elapsed)
                result.ref_runtime = ref_stats["mean"]
                result.ref_runtime_stats = ref_stats
            except Exception as e:
                if verbose:
                    print(f"[Eval] Error measuring reference performance: {e}")
                result.metadata["error_during_ref_performance"] = str(e)

        # --- Re-verify after timing: run the solution once more on the holdout
        # golden (computed pre-import). Catches a stateful forward that computes
        # correctly during the correctness trials, then short-circuits (returns a
        # cached/trivial tensor) during the timing phase for a fraudulently low
        # RUNTIME. Done OUTSIDE any timed region. ---
        if result.correctness:
            try:
                holdout_inp, holdout_ref = golden_trials[num_correct_trials]
                with torch.no_grad():
                    holdout_out = custom_model(*_clone_inputs(holdout_inp))
                passing, pcc_val, rel_err, reasons = _compare_output(
                    holdout_ref, holdout_out, pcc_threshold, rel_tol
                )
                result.metadata["reverify_pcc"] = f"{pcc_val:.6f}"
                if not passing:
                    result.correctness = False
                    result.pcc = min(result.pcc, pcc_val)  # headline PCC reflects the failure
                    result.metadata["reverify_failed"] = "; ".join(reasons)
                    print(
                        f"[WARNING] Post-timing re-verify FAILED ({'; '.join(reasons)}) — "
                        "the solution's output changed after the correctness phase "
                        "(possible stateful short-circuit). Marking INCORRECT."
                    )
            except Exception as e:
                result.correctness = False
                result.metadata["reverify_error"] = str(e)

        return result

    finally:
        graceful_eval_cleanup(temp_path)
        try:
            ttnn.close_device(device)
        except Exception:
            pass


###############################################################################
# Source transformation (anti-cheat boundary — inherited from KernelBench)
###############################################################################


def read_file(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def rename_model_to_modelnew(src: str) -> str:
    """Rename the solution's `Model` class (and its references) to `ModelNew`.

    Rewrites only NAME tokens equal to `Model` — the class def, `super(Model,
    self)`, and in-body references like `isinstance(x, Model)` or
    `Model.some_staticmethod(...)`. Renaming only the class def would leave those
    references bound to a name that no longer exists (NameError at eval).

    Because it operates on tokens, it leaves strings, comments, and substrings
    (`MyModel`, `model_kernel.cpp`, a registered op name containing "Model")
    untouched — a blanket regex would corrupt those. Falls back to a
    word-boundary regex only if the source cannot be tokenized.
    """
    if re.search(r"\bclass\s+ModelNew\b", src):
        return src  # already has ModelNew

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return re.sub(r"\bModel\b", "ModelNew", src)

    lines = src.splitlines(keepends=True)
    edits = []  # (line_index, col_start, col_end) of each `Model` NAME token
    for tok in tokens:
        if tok.type == tokenize.NAME and tok.string == "Model":
            (srow, scol), (erow, ecol) = tok.start, tok.end
            if srow == erow:  # an identifier never spans lines
                edits.append((srow - 1, scol, ecol))
    # Apply right-to-left so earlier column offsets on a line stay valid.
    for line_idx, scol, ecol in sorted(edits, reverse=True):
        line = lines[line_idx]
        lines[line_idx] = line[:scol] + "ModelNew" + line[ecol:]
    return "".join(lines)


def _strip_input_defs(src: str) -> str:
    """Remove any TOP-LEVEL ``get_inputs`` / ``get_init_inputs`` function from the
    solution, keeping everything else (helper functions, constants, other classes).

    This is the anti-cheat boundary: those two functions define the test inputs,
    which must come from the reference / --inputs file, never the solution.

    It deliberately does NOT strip the whole post-class tail (the old KernelBench
    behavior): a ttnn / tt-metal solution commonly defines helper functions or
    weight/config constants after ``class Model``, and deleting them caused
    confusing silent NameErrors at eval. Only the two input-providing functions
    (matched at column 0) are removed; a ``def get_inputs(self)`` method inside a
    class is indented and thus preserved.
    """
    lines = src.split("\n")
    pat = re.compile(r"^(async\s+)?def\s+(get_inputs|get_init_inputs)\s*\(")
    out = []
    i, n = 0, len(lines)
    while i < n:
        if pat.match(lines[i]):
            i += 1  # skip the def line, then its indented body / blank lines
            while i < n and (lines[i].strip() == "" or lines[i][:1] in (" ", "\t")):
                i += 1
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def prepare_solution_source(sol_src: str) -> str:
    """Prepare solution source for eval:
    1. Rename class Model -> class ModelNew (and its references).
    2. Remove any top-level get_inputs / get_init_inputs definitions.

    The strip is the anti-cheat boundary: test inputs come from the reference or
    the --inputs file, never the solution. Unlike the old whole-tail strip, it
    keeps legitimate post-class helper functions and constants.
    """
    modified = rename_model_to_modelnew(sol_src)
    return _strip_input_defs(modified).rstrip() + "\n"


def _auto_detect_backend(sol_src: str) -> str:
    """Pick a backend label from solution source. Informational — both backends
    load the same way (a Python file whose forward runs on the TT device).

    tt-metal (low-level C++ Tensix kernels launched from Python) is detected by
    tt-metal host-API / kernel markers; otherwise the solution is treated as
    TT-NN (ttnn op library). Defaults to ttnn.
    """
    tt_metal_markers = (
        "CreateKernel",
        "CreateProgram",
        "EnqueueProgram",
        "CircularBufferConfig",
        "tt_metal",
        "tt::tt_metal",
        "ttnn.experimental",
        "program_factory",
        "get_program_cache",
    )
    if any(m in sol_src for m in tt_metal_markers):
        return "tt-metal"
    if "import ttnn" in sol_src or "ttnn." in sol_src:
        return "ttnn"
    return "ttnn"


def load_inputs_module(path: str):
    """Load get_inputs (required) and get_init_inputs (optional) from a file.
    Used when --inputs decouples test-input definition from the reference file.
    """
    spec = importlib.util.spec_from_file_location("ako_inputs_module", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load inputs module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    get_inputs = getattr(module, "get_inputs", None)
    get_init_inputs = getattr(module, "get_init_inputs", None)

    if get_inputs is None:
        raise ValueError(f"Inputs file {path} must define a top-level get_inputs()")

    return get_inputs, get_init_inputs


###############################################################################
# Self-test (hardware-independent — no ttnn / no device required)
###############################################################################


def _self_test():
    """Verify source transformation, backend detection, inputs loading, and PCC."""
    sol = '''import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return x * 3

N = 4096

def get_inputs():
    return [torch.randn(N)]

def get_init_inputs():
    return [42]
'''
    result = prepare_solution_source(sol)
    assert "class ModelNew(" in result, "ModelNew rename failed"
    assert "class Model(" not in result, "Original Model class still present"
    assert "super(ModelNew," in result, "super() rename failed"
    assert "def get_inputs" not in result, "Solution's get_inputs must be stripped"
    assert "def get_init_inputs" not in result, "Solution's get_init_inputs must be stripped"
    assert "return [42]" not in result, "Solution's get_init_inputs body must be stripped"
    # Targeted strip keeps module-level constants (only the two input functions go).
    assert "N = 4096" in result, "post-class constant must be kept"
    print("Self-test PASSED: source transformation is correct.")

    # --- Multi-class regression ---
    sol_multi = '''import torch
import torch.nn as nn

class Helper(nn.Module):
    def forward(self, x):
        return x

def helper_fn():
    pass

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return x * 3

N = 4096

def get_inputs():
    return [torch.randn(N)]

def get_init_inputs():
    return [42]
'''
    result_multi = prepare_solution_source(sol_multi)
    assert "class Helper(" in result_multi, "Helper class should be preserved"
    assert "class ModelNew(" in result_multi, "ModelNew rename failed (multi-class)"
    assert "def helper_fn" in result_multi, "helper_fn should be preserved"
    assert "def get_inputs" not in result_multi, "Solution's get_inputs must be stripped (multi-class)"
    assert "def get_init_inputs" not in result_multi, "get_init_inputs must be stripped (multi-class)"
    print("Self-test PASSED: multi-class transformation is correct.")

    # --- Targeted strip keeps legitimate post-class helpers/constants ---
    sol_helper = '''import torch, torch.nn as nn

class Model(nn.Module):
    def forward(self, x):
        return _tt_scale(x, W)

W = 3.0

def _tt_scale(x, w):
    return x * w

def get_inputs():
    return [torch.randn(8)]
'''
    r_help = prepare_solution_source(sol_helper)
    assert "def _tt_scale" in r_help, "post-class helper must be kept"
    assert "W = 3.0" in r_help, "post-class constant must be kept"
    assert "def get_inputs" not in r_help, "get_inputs must be stripped"
    # A get_inputs *method* (indented) must be preserved.
    r_method = _strip_input_defs("class M:\n    def get_inputs(self):\n        return 1\n")
    assert "def get_inputs(self)" in r_method, "indented get_inputs method must be kept"
    # No-class source: get_inputs stripped, other top-level code kept.
    r_noclass = prepare_solution_source("import torch\nK = 1\n\ndef get_inputs():\n    return [K]\n")
    assert "K = 1" in r_noclass and "def get_inputs" not in r_noclass, "no-class strip"
    print("Self-test PASSED: targeted input-def strip keeps helpers/constants.")

    # --- Backend auto-detection ---
    assert _auto_detect_backend("import ttnn\nx = ttnn.matmul(a, b)") == "ttnn", \
        "ttnn source should detect ttnn"
    assert _auto_detect_backend(
        "program = CreateProgram()\nCreateKernel(program, 'reader.cpp', core, cfg)"
    ) == "tt-metal", "tt-metal host API should detect tt-metal"
    assert _auto_detect_backend("import ttnn\nttnn.experimental.foo()") == "tt-metal", \
        "ttnn.experimental custom op should detect tt-metal"
    assert _auto_detect_backend("x = 1") == "ttnn", "unknown source defaults to ttnn"
    print("Self-test PASSED: backend auto-detection.")

    # --- PCC ---
    a = torch.randn(1024)
    passing, val = comp_pcc(a, a.clone(), 0.99)
    assert passing and abs(val - 1.0) < 1e-6, f"identical tensors must give PCC 1.0, got {val}"
    # A global scale is a PCC blind spot: PCC alone still ~1.0 (the magnitude
    # guard below is what actually catches it).
    passing, val = comp_pcc(a, a * 2.0, 0.99)
    assert passing and val > 0.99, f"scaled tensor should still correlate, got {val}"
    # Uncorrelated noise (constant-vs-noise style garbage) should fail.
    b = torch.randn(1024)
    passing, val = comp_pcc(a, b, 0.99)
    assert not passing, f"independent noise should fail PCC, got {val}"
    # bfloat16 round-trip stays well above a 0.99 gate.
    passing, val = comp_pcc(a, a.to(torch.bfloat16).to(torch.float32), 0.99)
    assert passing and val > 0.99, f"bf16 round-trip should pass PCC, got {val}"
    # Shape mismatch fails.
    passing, val = comp_pcc(a, torch.randn(512), 0.99)
    assert not passing, "shape mismatch must fail"
    print("Self-test PASSED: PCC metric.")

    # --- Relative-L2 magnitude guard (catches what PCC misses) ---
    assert relative_l2_error(a, a.clone()) < 1e-6, "identical tensors: zero rel error"
    # output*2 sails through PCC but the magnitude guard rejects it (~1.0 >> 0.1).
    assert relative_l2_error(a, a * 2.0) > 0.5, "a 2x scale must be a large rel error"
    # bf16 round-trip stays well under a 0.1 gate.
    assert relative_l2_error(a, a.to(torch.bfloat16).to(torch.float32)) < 0.05, \
        "bf16 round-trip rel error must stay small"
    # all-zero golden vs nonzero solution -> nonzero (absolute) error.
    assert relative_l2_error(torch.zeros(64), torch.full((64,), 0.5)) > 0.0, \
        "all-zero golden vs nonzero must be flagged"
    print("Self-test PASSED: relative-L2 magnitude guard.")

    # NaN in the solution where the golden is finite must FAIL — per-tensor
    # masking, not a union that would zero the golden there and hide it.
    g = torch.tensor([1.0, 2.0, 3.0, 100.0])
    passing, _ = comp_pcc(g, torch.tensor([1.0, 2.0, 3.0, float("nan")]), 0.99)
    assert not passing, "NaN divergence in the solution must fail PCC"
    # Constant fallback: matching constants pass; a constant vs a varying tensor
    # fails in either direction (the latter is the return-a-constant anti-cheat).
    passing, _ = comp_pcc(torch.full((64,), 2.0), torch.full((64,), 2.0), 0.99)
    assert passing, "matching constant tensors must pass"
    passing, _ = comp_pcc(torch.full((64,), 2.0), torch.randn(64) + 100.0, 0.99)
    assert not passing, "constant golden vs varying solution must fail"
    passing, _ = comp_pcc(torch.arange(64.0), torch.full((64,), 5.0), 0.99)
    assert not passing, "varying golden vs constant solution must fail (anti-cheat)"
    print("Self-test PASSED: PCC edge cases (NaN divergence, constant fallback).")

    # In-body class references must also be renamed, or they NameError at eval.
    sol_ref = '''import torch.nn as nn

class Model(nn.Module):
    def forward(self, x):
        assert isinstance(self, Model)
        return Model.scale(x)

    @staticmethod
    def scale(x):
        return x * 2
'''
    r = prepare_solution_source(sol_ref)
    assert "isinstance(self, ModelNew)" in r, "in-body isinstance ref must be renamed"
    assert "ModelNew.scale(x)" in r, "in-body staticmethod ref must be renamed"
    assert not re.search(r"\bclass Model\b", r), "class def must be renamed"
    print("Self-test PASSED: in-body Model references renamed.")

    # Token-based rename must NOT touch `Model` inside strings / comments /
    # substrings (a blanket regex would corrupt op names and kernel paths).
    sol_str = '''import torch.nn as nn

class Model(nn.Module):
    # Model note: keep this comment's word intact
    name = "Model op v1"
    path = "kernels/Model_reader.cpp"

    def forward(self, x):
        return MyModel_helper(x)
'''
    rs = rename_model_to_modelnew(sol_str)
    assert "class ModelNew(" in rs, "class def must be renamed"
    assert '"Model op v1"' in rs, "string literal 'Model' must be untouched"
    assert "kernels/Model_reader.cpp" in rs, "path substring must be untouched"
    assert "# Model note" in rs, "comment 'Model' must be untouched"
    assert "MyModel_helper" in rs, "unrelated identifier must be untouched"
    print("Self-test PASSED: token rename leaves strings/comments/substrings intact.")

    # --- load_inputs_module ---
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("def get_inputs():\n    return [1, 2, 3]\n\ndef get_init_inputs():\n    return ['a']\n")
        tmp_path = f.name
    try:
        gi, gii = load_inputs_module(tmp_path)
        assert gi() == [1, 2, 3], "get_inputs return mismatch"
        assert gii() == ["a"], "get_init_inputs return mismatch"
        print("Self-test PASSED: load_inputs_module (both functions).")
    finally:
        os.remove(tmp_path)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("def get_init_inputs():\n    return []\n")
        tmp_path = f.name
    try:
        try:
            load_inputs_module(tmp_path)
            assert False, "Should have raised for missing get_inputs"
        except ValueError as e:
            assert "get_inputs" in str(e)
        print("Self-test PASSED: load_inputs_module rejects missing get_inputs.")
    finally:
        os.remove(tmp_path)


###############################################################################
# CLI
###############################################################################


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate an optimized Tenstorrent kernel against a reference (self-contained)"
    )
    parser.add_argument(
        "--ref",
        default=None,
        help="Path to reference kernel (defines class Model, the CPU golden). "
             "Required unless --self-test is given.",
    )
    parser.add_argument(
        "--solution",
        default=None,
        help="Path to optimized solution (defines class Model — renamed to "
             "ModelNew automatically; runs on the TT device). "
             "Required unless --self-test is given.",
    )
    parser.add_argument(
        "--inputs",
        default=None,
        help="Optional file defining get_inputs() (required) and get_init_inputs() "
             "(optional); overrides definitions in --ref.",
    )
    parser.add_argument(
        "--pcc",
        type=float,
        default=0.99,
        help="PCC (Pearson Correlation Coefficient) threshold for correctness "
             "(default: 0.99). Use 0.999 / 0.9999 for stricter fp32/bf16 checks.",
    )
    parser.add_argument(
        "--rel-tol",
        type=float,
        default=0.1,
        help="Relative-L2 magnitude tolerance, checked alongside PCC (default: "
             "0.1). Catches scale/bias errors PCC is blind to (e.g. output×2) "
             "while tolerating bf16/bfloat8_b noise. Loosen for bfloat4_b.",
    )
    parser.add_argument(
        "--backend",
        default=None,
        choices=["ttnn", "tt-metal"],
        help="Backend label. Auto-detected from solution source if omitted "
             "(ttnn = TT-NN op library; tt-metal = low-level C++ Tensix kernels). "
             "Informational — both load the same way.",
    )
    parser.add_argument(
        "--device-id", type=int, default=0, help="Tenstorrent device id (default: 0)"
    )
    parser.add_argument(
        "--num-correct-trials",
        type=int,
        default=5,
        help="Number of correctness (PCC) trials (default: 5)",
    )
    parser.add_argument(
        "--num-perf-trials",
        type=int,
        default=100,
        help="Number of performance timing trials (default: 100)",
    )
    parser.add_argument(
        "--no-ref",
        action="store_true",
        help="Skip reference (CPU) timing: still emit COMPILED/CORRECT/PCC/RUNTIME "
             "but set REF_RUNTIME/SPEEDUP to -1. For fast iteration — rank "
             "candidates by the solution's own RUNTIME (the reference is "
             "invariant across solution edits).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run source-transformation / PCC self-test and exit (no device needed)",
    )

    args = parser.parse_args()

    if args.self_test:
        _self_test()
        sys.exit(0)

    if args.ref is None or args.solution is None:
        parser.error("--ref and --solution are required (unless --self-test is given)")

    if args.num_correct_trials < 1:
        parser.error(
            "--num-correct-trials must be >= 1 (a zero-trial run would vacuously "
            "pass without ever comparing to the golden)"
        )
    if args.num_perf_trials < 1:
        parser.error("--num-perf-trials must be >= 1")

    def _fail(msg):
        # Emit the structured schema (not a bare traceback) so the wrapper/agent
        # can parse a setup error the same way as a compile/eval failure.
        print("COMPILED: False")
        print("CORRECT: False")
        print(f"BACKEND: {args.backend or 'ttnn'} (n/a)")
        print("PCC: -1\nRUNTIME: -1\nREF_RUNTIME: -1\nSPEEDUP: -1")
        print(f"ERROR: {msg}")
        sys.exit(1)

    try:
        ref_src = read_file(args.ref)
    except OSError as e:
        _fail(f"cannot read --ref {args.ref}: {e}")
    try:
        sol_src = read_file(args.solution)
    except OSError as e:
        _fail(f"cannot read --solution {args.solution}: {e}")

    if args.backend is None:
        args.backend = _auto_detect_backend(sol_src)
        backend_origin = "auto"
    else:
        backend_origin = "explicit"

    modified_sol_src = prepare_solution_source(sol_src)

    get_inputs_override = None
    get_init_inputs_override = None
    if args.inputs:
        try:
            get_inputs_override, get_init_inputs_override = load_inputs_module(args.inputs)
        except (OSError, ValueError) as e:
            _fail(f"cannot load --inputs {args.inputs}: {e}")
        if args.verbose:
            print(f"[Inputs] Loaded get_inputs from {args.inputs}")
            if get_init_inputs_override is not None:
                print(f"[Inputs] Loaded get_init_inputs from {args.inputs}")

    if args.verbose:
        print("=" * 60)
        print("REFERENCE SOURCE:")
        print("=" * 60)
        print(ref_src[:500], "..." if len(ref_src) > 500 else "")
        print()
        print("=" * 60)
        print("MODIFIED SOLUTION SOURCE:")
        print("=" * 60)
        print(modified_sol_src[:500], "..." if len(modified_sol_src) > 500 else "")
        print()

    result = eval_kernel_against_ref(
        original_model_src=ref_src,
        custom_model_src=modified_sol_src,
        num_correct_trials=args.num_correct_trials,
        num_perf_trials=args.num_perf_trials,
        measure_performance=True,
        measure_reference=not args.no_ref,
        pcc_threshold=args.pcc,
        rel_tol=args.rel_tol,
        verbose=args.verbose,
        device_id=args.device_id,
        backend=args.backend,
        get_inputs_override=get_inputs_override,
        get_init_inputs_override=get_init_inputs_override,
        ref_path=args.ref,
        sol_path=args.solution,
    )

    if result is None:
        print("COMPILED: False")
        print("CORRECT: False")
        print(f"BACKEND: {args.backend} ({backend_origin})")
        print("PCC: -1")
        print("RUNTIME: -1")
        print("REF_RUNTIME: -1")
        print("SPEEDUP: -1")
        print("ERROR: eval_kernel_against_ref returned None")
        sys.exit(1)

    runtime_ms = result.runtime if result.runtime > 0 else -1
    ref_runtime_ms = result.ref_runtime if result.ref_runtime > 0 else -1
    speedup = ref_runtime_ms / runtime_ms if (runtime_ms > 0 and ref_runtime_ms > 0) else -1

    print(f"COMPILED: {result.compiled}")
    print(f"CORRECT: {result.correctness}")
    print(f"BACKEND: {args.backend} ({backend_origin})")
    print(f"PCC: {result.pcc:.6f}" if result.pcc >= 0 else "PCC: -1")
    print(f"RUNTIME: {runtime_ms:.4f}" if runtime_ms > 0 else "RUNTIME: -1")
    print(f"REF_RUNTIME: {ref_runtime_ms:.4f}" if ref_runtime_ms > 0 else "REF_RUNTIME: -1")
    print(f"SPEEDUP: {speedup:.4f}x" if speedup > 0 else "SPEEDUP: -1")

    if speedup > EXCESSIVE_SPEEDUP_THRESHOLD:
        print(
            f"[WARNING] SPEEDUP {speedup:.1f}x exceeds {EXCESSIVE_SPEEDUP_THRESHOLD:.0f}x. "
            "SPEEDUP is measured against a CPU reference, so a large value can be "
            "legitimate for a real TT kernel — but double-check the solution actually "
            "runs on the device and isn't returning trivial/precomputed output "
            "(the fresh-input PCC + magnitude checks guard against this, but verify).",
            file=sys.stderr,
        )

    if args.verbose:
        print()
        print("--- Details ---")
        if result.runtime_stats:
            print(f"Runtime stats (TT): {result.runtime_stats}")
        if result.ref_runtime_stats:
            print(f"Ref runtime stats (CPU): {result.ref_runtime_stats}")
        if result.metadata:
            for k, v in result.metadata.items():
                print(f"  {k}: {v}")

    sys.exit(0 if result.correctness else 1)


if __name__ == "__main__":
    main()
