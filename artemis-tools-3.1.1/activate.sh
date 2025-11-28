#!/bin/bash
# Activate the artemis virtual environment

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/venv/bin/activate"

echo "Virtual environment activated!"
echo ""
echo "To get started:"
echo "  1. Run: artemis-runner"
echo "  2. You'll be prompted for credentials on first run"
echo ""
echo "For help: artemis-runner --help"
