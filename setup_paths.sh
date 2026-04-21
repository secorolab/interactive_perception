#!/bin/bash
# Setup script to configure Python path for interactive_perception packages
#
# Usage: source setup_paths.sh
#
# This script adds the necessary directories to PYTHONPATH so that imports
# like "from core_algorithm import ..." work correctly.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Add interactive_perception root to PYTHONPATH
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

echo "✓ PYTHONPATH configured for interactive_perception"
echo "  - core_algorithm: $SCRIPT_DIR/core_algorithm"
echo "  - simulate_algorithm: $SCRIPT_DIR/simulate_algorithm"
echo ""
echo "You can now run simulations and tests:"
echo "  python simulate_algorithm/simulation.py"
echo "  python -m unittest simulate_algorithm/test_*.py"
