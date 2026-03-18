#!/usr/bin/env bash
# run_multimodal_coach.command
# ----------------------------
# One-click macOS launcher for the Multimodal Coach app.
# Double-click this file in Finder (or run it from Terminal).
#
# Prerequisites (one-time):
#   python3 -m venv venv && source venv/bin/activate
#   pip install -r requirements.txt
#
# macOS permissions required on first run:
#   System Settings → Privacy & Security → Camera  → allow Terminal / your IDE
#   System Settings → Privacy & Security → Microphone → allow Terminal / your IDE

# ── Locate this script's directory regardless of where it is double-clicked ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || { echo "ERROR: cannot cd to $SCRIPT_DIR"; exit 1; }

# ── Activate Environment (Active -> Local Venv -> Conda) ───────────────
if [ -n "$VIRTUAL_ENV" ] && [ -f "$VIRTUAL_ENV/bin/python" ]; then
  # 1. Use currently active standard venv
  echo "Using active Virtual Environment: $VIRTUAL_ENV"
  PYTHON="$VIRTUAL_ENV/bin/python"
elif [ -n "$CONDA_PREFIX" ] && [ -f "$CONDA_PREFIX/bin/python" ]; then
  # 2. Use currently active Conda environment
  echo "Using active Conda Environment: $CONDA_PREFIX"
  PYTHON="$CONDA_PREFIX/bin/python"
elif [ -d "$SCRIPT_DIR/venv" ] && [ -f "$SCRIPT_DIR/venv/bin/python" ]; then
  # 3. Use local venv if it exists in the project folder
  source "$SCRIPT_DIR/venv/bin/activate"
  PYTHON="$SCRIPT_DIR/venv/bin/python"
  echo "Using local venv environment..."
else
  # 4. Fallback to Conda 'dslcv2' environment (create if necessary)
  CONDA_BASE="$(conda info --base 2>/dev/null || echo /opt/anaconda3)"
  # shellcheck source=/dev/null
  source "$CONDA_BASE/etc/profile.d/conda.sh" 2>/dev/null || \
    source "$CONDA_BASE/bin/activate" 2>/dev/null || true
  
  conda activate dslcv2 2>/dev/null || true
  PYTHON="$CONDA_BASE/envs/dslcv2/bin/python"
  echo "Using conda dslcv2 environment..."
fi

# 5. Final Fallback to system python if still not found
if [ ! -f "$PYTHON" ]; then
  echo "Warning: Python not found in specific environments, falling back to system python3"
  PYTHON=python3
fi

"$PYTHON" - <<'EOF' 2>/dev/null
import mediapipe as mp
assert hasattr(mp, 'solutions'), "mediapipe solutions API missing"
EOF
if [ $? -ne 0 ]; then
  echo ""
  echo "⚠️  mediapipe version incompatible — installing 0.10.14 ..."
  "$PYTHON" -m pip install "mediapipe==0.10.14" --quiet
fi

# ── Launch app ───────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Multimodal Coach  —  starting up ..."
echo "  Press Q inside the camera window to quit."
echo "============================================"
echo ""

PYTHONPATH="$SCRIPT_DIR/src" "$PYTHON" "$SCRIPT_DIR/apps/run_multimodal_coach.py"
EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
  echo "App closed cleanly."
else
  echo "App exited with code $EXIT_CODE."
fi

# Keep Terminal window open so the user can read any error messages
echo ""
echo "Press any key to close this window ..."
read -r -n 1
