#!/usr/bin/env bash
# setup.sh — create venv and install everything in one shot
# Run from inside the lab_sim/ directory:
#   bash setup.sh

set -e

echo "=== Research Lab Simulation — setup ==="

# 1. Check Python
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.10+ first."
    exit 1
fi
PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python $PYVER found."

# 2. Create venv
if [ -d "venv" ]; then
    echo "venv/ already exists — skipping creation."
else
    python3 -m venv venv
    echo "venv created."
fi

# 3. Activate and install
echo "Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo "Dependencies installed."

# 4. Check Ollama
echo ""
if command -v ollama &>/dev/null; then
    echo "Ollama found."
    echo "Pulling required models (this may take a few minutes)..."
    ollama pull llama3
    ollama pull nomic-embed-text
    echo "Models ready."
else
    echo "WARNING: Ollama not found."
    echo "Install it from https://ollama.com/download"
    echo "Then run:"
    echo "  ollama pull llama3"
    echo "  ollama pull nomic-embed-text"
fi

# 5. Done
echo ""
echo "=== Setup complete ==="
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run the simulation:"
echo "  python run_sim.py --students 3 --postdocs 2 --timesteps 5 --fairness 0.7"
echo ""
echo "To compare fair vs biased PI:"
echo "  python run_sim.py --timesteps 5 --fairness 1.0 --save-json fair.json"
echo "  python run_sim.py --timesteps 5 --fairness 0.2 --pi-favorite 0 --save-json biased.json"
