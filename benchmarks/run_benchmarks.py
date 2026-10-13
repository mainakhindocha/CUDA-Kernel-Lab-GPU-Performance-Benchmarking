#!/usr/bin/env python3
"""
Stage 1: Multi-Size Benchmark Runner
Project: CUDA Kernel Lab — GPU Performance Benchmarking

Executes CPU baseline and/or CUDA naive kernel across multiple matrix dimensions,
records actual measured hardware performance, and exports results directly to CSV.
(Note: Never fabricates or estimates GPU performance numbers).
"""

import sys
import subprocess
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = BASE_DIR / "benchmarks"

EXE_EXT = ".exe" if sys.platform == "win32" else ""
CPU_BIN = BASE_DIR / "src" / f"cpu_baseline{EXE_EXT}"
CUDA_BIN = BASE_DIR / "src" / f"matmul_naive{EXE_EXT}"


def run_cpu_benchmarks(sizes, iters):
    print("\n" + "=" * 70)
    print(" Running CPU Baseline Benchmark Sweep")
    print("=" * 70)
    if not CPU_BIN.exists():
        print(f"Error: CPU binary not found at {CPU_BIN}. Please compile first.")
        return False

    csv_path = BENCHMARK_DIR / "cpu_benchmark_results.csv"
    results = []

    print(f"{'Size N':<8} | {'Min Time (ms)':<14} | {'Avg Time (ms)':<14} | {'Max Time (ms)':<14} | {'GFLOPS':<10}")
    print("-" * 70)

    for N in sizes:
        # Cap CPU benchmark to avoid excessive execution time for N > 1024 in naive O(N^3)
        if N > 1024:
            print(f"{N:<8} | Skipped for CPU (O(N^3) single-threaded latency > 30s)")
            continue

        cmd = [str(CPU_BIN), str(N), str(iters), "--csv"]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if res.returncode == 0:
                line = res.stdout.strip().split("\n")[-1]
                parts = line.split(",")
                if len(parts) >= 5:
                    n_val, min_ms, avg_ms, max_ms, gflops = parts[:5]
                    print(f"{n_val:<8} | {min_ms:<14} | {avg_ms:<14} | {max_ms:<14} | {gflops:<10}")
                    results.append(line)
            else:
                print(f"Failed for N={N} (exit code {res.returncode}):")
                if res.stdout:
                    print(f"  [stdout]:\n{res.stdout.strip()}")
                if res.stderr:
                    print(f"  [stderr]:\n{res.stderr.strip()}")
        except subprocess.TimeoutExpired:
            print(f"Timeout expired for N={N}")

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("N,min_ms,avg_ms,max_ms,gflops\n")
        for line in results:
            f.write(line + "\n")

    print("-" * 70)
    print(f"Recorded {len(results)} CPU benchmark data points to: {csv_path}")
    return True


def run_cuda_benchmarks(sizes, block_dim, iters):
    print("\n" + "=" * 70)
    print(" Running CUDA Naive Kernel Benchmark Sweep")
    print("=" * 70)
    if not CUDA_BIN.exists():
        print(f"NOTICE: CUDA binary not found at {CUDA_BIN}.")
        print("GPU benchmarks require an environment with the NVIDIA CUDA Toolkit (nvcc).")
        print("To run on Google Colab:")
        print("  1. Compile: nvcc -O3 src/matmul_naive.cu -o src/matmul_naive")
        print("  2. Run:     python benchmarks/run_benchmarks.py --cuda-only")
        return False

    csv_path = BENCHMARK_DIR / "cuda_benchmark_results.csv"
    results = []

    print(f"{'Size N':<8} | {'H2D (ms)':<10} | {'Kernel (ms)':<12} | {'D2H (ms)':<10} | {'Total (ms)':<10} | {'GFLOPS':<10} | {'Verify':<8}")
    print("-" * 70)

    for N in sizes:
        cmd = [str(CUDA_BIN), str(N), str(block_dim), str(iters), "--csv"]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if res.returncode == 0:
                line = res.stdout.strip().split("\n")[-1]
                parts = line.split(",")
                if len(parts) >= 10:
                    n_val = parts[0]
                    h2d = parts[2]
                    kernel_min = parts[3]
                    d2h = parts[5]
                    total_pipe = parts[6]
                    gflops = parts[7]
                    verified = parts[8]
                    print(f"{n_val:<8} | {h2d:<10} | {kernel_min:<12} | {d2h:<10} | {total_pipe:<10} | {gflops:<10} | {verified:<8}")
                    results.append(line)
            else:
                print(f"Failed for N={N} (exit code {res.returncode}):")
                if res.stdout:
                    print(f"  [stdout]:\n{res.stdout.strip()}")
                if res.stderr:
                    print(f"  [stderr]:\n{res.stderr.strip()}")
        except subprocess.TimeoutExpired:
            print(f"Timeout expired for N={N}")

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("N,block_dim,h2d_ms,min_kernel_ms,avg_kernel_ms,d2h_ms,total_pipeline_ms,gflops,verified,max_diff\n")
        for line in results:
            f.write(line + "\n")

    print("-" * 70)
    print(f"Recorded {len(results)} CUDA benchmark data points to: {csv_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Stage 1 Matrix Multiplication Benchmark Sweeper")
    parser.add_argument("--sizes", nargs="+", type=int, default=[64, 128, 256, 512, 1024, 2048],
                        help="Matrix sizes N to benchmark (default: 64 128 256 512 1024 2048)")
    parser.add_argument("--iters", type=int, default=3, help="Iterations per size (default: 3)")
    parser.add_argument("--block-dim", type=int, default=16, help="CUDA thread block dimension (default: 16)")
    parser.add_argument("--cpu-only", action="store_true", help="Only run CPU baseline")
    parser.add_argument("--cuda-only", action="store_true", help="Only run CUDA kernel")

    args = parser.parse_args()

    print("======================================================================")
    print(" CUDA Kernel Lab — Stage 1 Performance Benchmark Sweeper")
    print("======================================================================")
    print(f"Target Matrix Sizes : {args.sizes}")
    print(f"Iterations per run  : {args.iters}")

    if not args.cuda_only:
        run_cpu_benchmarks(args.sizes, args.iters)

    if not args.cpu_only:
        run_cuda_benchmarks(args.sizes, args.block_dim, args.iters)


if __name__ == "__main__":
    main()
