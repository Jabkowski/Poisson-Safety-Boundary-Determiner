#!/usr/bin/env bash
set -euo pipefail

HERE=$(dirname "$0")
cd "$HERE/.."

echo "Compiling..."
g++ -O2 -std=c++17 main.cpp poisson_solver.cpp -I/usr/include/eigen3 -o poisson_safety

echo "Running..."
./poisson_safety

echo "Done. CSV files written to $(pwd)"
