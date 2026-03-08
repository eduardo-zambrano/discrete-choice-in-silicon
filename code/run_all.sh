#!/usr/bin/env bash
#
# Master replication script for:
#   "Rational Inattention in Silicon"
#   Eduardo Zambrano, Cal Poly
#
# This script creates a virtual environment, installs dependencies,
# runs all diagnostics, and produces the 5 figures in output/figures/.
#
# Usage:
#   cd replication
#   bash code/run_all.sh
#
set -e

# Navigate to replication/ root regardless of where the script is called from
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPL_ROOT"

echo "============================================================"
echo "  Replication: Rational Inattention in Silicon"
echo "  Eduardo Zambrano, Cal Poly"
echo "============================================================"
echo ""

# --- Check Python version ---
PYTHON_CMD=""
for cmd in python3 python; do
    if command -v "$cmd" &> /dev/null; then
        VERSION=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
        MAJOR=$("$cmd" -c "import sys; print(sys.version_info.major)" 2>/dev/null)
        MINOR=$("$cmd" -c "import sys; print(sys.version_info.minor)" 2>/dev/null)
        if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 10 ]; then
            PYTHON_CMD="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "ERROR: Python 3.10 or higher is required."
    echo "Please install Python 3.10+ and ensure it is on your PATH."
    exit 1
fi

echo "Using Python: $PYTHON_CMD ($VERSION)"
echo ""

# --- Create virtual environment ---
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    "$PYTHON_CMD" -m venv .venv
fi

# --- Activate virtual environment ---
source .venv/bin/activate
echo "Virtual environment activated: $(which python)"
echo ""

# --- Install dependencies ---
echo "Installing dependencies..."
pip install --quiet -r requirements.txt
echo "Dependencies installed."
echo ""

# --- Create output directory ---
mkdir -p output/figures

# --- Run diagnostics ---
echo "Running all diagnostics (this may take 2-5 minutes)..."
echo "  NOTE: GPT-2 small (~500 MB) will be downloaded on first run."
echo ""
python code/attention_diagnostics.py --all

echo ""
echo "============================================================"
echo "  Replication complete."
echo ""
echo "  Output figures:"
echo "    output/figures/inclusive_value.pdf   (Figure 1)"
echo "    output/figures/iia_test.pdf          (Figure 2)"
echo "    output/figures/temperature.pdf       (Figure 3)"
echo "    output/figures/head_aggregation.pdf  (Figure 4)"
echo "    output/figures/hhi.pdf               (Figure A.1)"
echo "============================================================"

deactivate
