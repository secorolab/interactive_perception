#!/bin/bash
# Quick runner for the interactive perception simulation
#
# Usage: ./run_simulation.sh [args]
#
# This script automatically sets up the Python path and runs the simulation.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Export PYTHONPATH with the workspace root
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Run simulation
python3 "${SCRIPT_DIR}/simulate_algorithm/simulation.py" "$@"
