#!/usr/bin/env bash
# One-time setup: Python environment, API key file, and RealSense USB permissions.
#   bash scripts/setup.sh
set -euo pipefail

cd "$(dirname "$0")/.."

# uv manages Python 3.12 and every dependency (Isaac Sim, Isaac Lab, CUDA PyTorch).
if ! command -v uv >/dev/null; then
    echo "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv sync

# API keys for the vision models; fill in the ones you use.
if [ ! -f .env ]; then
    cp .env.example .env
    echo "created .env from .env.example; add your API keys"
fi

# Without udev rules the D435 is only readable as root.
RULES=/etc/udev/rules.d/99-realsense-libusb.rules
if [ ! -f "$RULES" ]; then
    echo "installing RealSense udev rules (needs sudo)"
    curl -fsSL https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules \
        | sudo tee "$RULES" >/dev/null
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    echo "replug the camera"
fi

echo
echo "done. the first 'view' run asks to accept the Isaac Sim EULA and compiles RTX shaders (~2-3 min)."
echo "  uv run main.py capture <name>"
echo "  uv run main.py generate <name>"
echo "  uv run main.py view <name>"
