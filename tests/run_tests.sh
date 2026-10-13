#!/bin/bash
# Stage 1: Correctness Test Runner (Bash)
# CUDA Kernel Lab — GPU Performance Benchmarking

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================================="
echo " CUDA Kernel Lab — Stage 1 Correctness Test Runner (Bash)"
echo "=========================================================="

# 1. Compile CPU Baseline if not present
if [ ! -f "src/cpu_baseline" ]; then
    echo "Compiling CPU Baseline..."
    g++ -O3 -std=c++14 src/cpu_baseline.cpp -o src/cpu_baseline
fi

# 2. Check if nvcc is available and compile CUDA kernel
if command -v nvcc &> /dev/null; then
    if [ ! -f "src/matmul_naive" ]; then
        echo "Compiling Naive CUDA Kernel with nvcc..."
        nvcc -O3 src/matmul_naive.cu -o src/matmul_naive
    fi
else
    echo "[Notice] nvcc not found in PATH. Skipping CUDA kernel compilation."
fi

# 3. Run the automated test suite
python3 tests/run_tests.py
