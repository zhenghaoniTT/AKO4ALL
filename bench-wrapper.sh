#!/bin/bash
# Benchmark wrapper with trajectory tracking (Tenstorrent / tt-metal).
# Usage: bash scripts/bench.sh [label]
set -eo pipefail
cd "$(dirname "$0")/.."

# --- Tenstorrent environment sanity checks (warn, don't fail) ---
# The default evaluator needs an importable `ttnn` and a reachable TT device.
# tt-metal usually lives outside the workspace; if so, export TT_METAL_HOME,
# add it to PYTHONPATH, and set ARCH_NAME (wormhole_b0 / blackhole) HERE — this
# template is meant to be edited for your environment (see SKILL.md "env
# friction"). Uncomment / adjust:
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
if ! python -c "import ttnn" 2>/dev/null; then
    echo "[bench-wrapper] 'import ttnn' failed — a tt-metal / TT-NN install must be" >&2
    echo "  importable by this python. Check PYTHONPATH / your conda env." >&2
fi
if [ -z "${ARCH_NAME:-}" ]; then
    echo "[bench-wrapper] ARCH_NAME unset (expected wormhole_b0 or blackhole)." >&2
fi

LABEL="${1:-}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# --- Bench command ---
# Run benchmark without exiting on failure — we need trajectory even for failed runs
set +e
{{BENCH_COMMAND}}
BENCH_EXIT=$?
set -e
# --- End bench command ---

# --- Trajectory ---
if [ -n "$LABEL" ]; then
    TRAJ_DIR="trajectory/${TIMESTAMP}_${LABEL}"
else
    TRAJ_DIR="trajectory/${TIMESTAMP}"
fi
mkdir -p "$TRAJ_DIR"
cp -r solution/* "$TRAJ_DIR/"
[ -f _bench_output.txt ] && mv _bench_output.txt "$TRAJ_DIR/output.txt"
echo "Trajectory saved to: $TRAJ_DIR"

exit $BENCH_EXIT
