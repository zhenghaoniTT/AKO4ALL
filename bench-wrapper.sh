#!/bin/bash
# Benchmark wrapper with trajectory tracking (Tenstorrent / tt-metal).
# Usage: bash scripts/bench.sh [label]
set -eo pipefail
cd "$(dirname "$0")/.."

# Python interpreter — override with PYTHON=... if your env's python differs.
# (Many hosts have only `python3`, not `python`.)
PYTHON="${PYTHON:-python3}"

# --- Tenstorrent environment sanity checks (warn, don't fail) ---
# The default evaluator needs an importable `ttnn` (with a usable device API)
# and a reachable TT device. tt-metal usually lives outside the workspace; if
# so, export TT_METAL_HOME, add it to PYTHONPATH, and set ARCH_NAME
# (wormhole_b0 / blackhole) HERE — this template is meant to be edited for your
# environment (see SKILL.md "env friction"). Uncomment / adjust:
#
#   export TT_METAL_HOME=/path/to/tt-metal
#   export PYTHONPATH="$TT_METAL_HOME:$PYTHONPATH"
#   export ARCH_NAME=wormhole_b0
#   # source /path/to/conda/etc/profile.d/conda.sh && conda activate <env>

if [ -z "${TT_METAL_HOME:-}" ]; then
    echo "[bench-wrapper] TT_METAL_HOME is not set. If ttnn is not already on" >&2
    echo "  PYTHONPATH, export TT_METAL_HOME/PYTHONPATH/ARCH_NAME at the top of" >&2
    echo "  this script (see SKILL.md)." >&2
fi
# `import ttnn` alone is not enough — a namespace/stub package can import while
# lacking the real API, so check for ttnn.open_device specifically.
if ! "$PYTHON" -c "import ttnn; ttnn.open_device" 2>/dev/null; then
    echo "[bench-wrapper] '$PYTHON -c \"import ttnn; ttnn.open_device\"' failed — a" >&2
    echo "  working tt-metal / TT-NN install must be importable by this python" >&2
    echo "  (check PYTHONPATH / your conda env, and set PYTHON=... if needed)." >&2
fi
if [ -z "${ARCH_NAME:-}" ]; then
    echo "[bench-wrapper] ARCH_NAME unset (expected wormhole_b0 or blackhole)." >&2
fi

LABEL="${1:-}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# --- Bench command ---
# Run the benchmark without exiting on failure — we need trajectory (and the
# real exit code) even for failed runs. Stay in `set +e` through the trajectory
# section so nothing there (e.g. an empty solution/) can clobber BENCH_EXIT.
set +e
{{BENCH_COMMAND}}
BENCH_EXIT=$?
# --- End bench command ---

# --- Trajectory ---
if [ -n "$LABEL" ]; then
    TRAJ_DIR="trajectory/${TIMESTAMP}_${LABEL}"
else
    TRAJ_DIR="trajectory/${TIMESTAMP}"
fi
mkdir -p "$TRAJ_DIR"
# `solution/.` copies the directory contents (incl. dotfiles) and does not error
# on an empty solution/, unlike `solution/*` which would fail the glob.
cp -r solution/. "$TRAJ_DIR/" 2>/dev/null
[ -f _bench_output.txt ] && mv _bench_output.txt "$TRAJ_DIR/output.txt"
echo "Trajectory saved to: $TRAJ_DIR"

exit $BENCH_EXIT
