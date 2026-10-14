#!/usr/bin/env python3
"""
Stage 1 & Stage 2 Benchmark Runner
Project: CUDA Kernel Lab -- GPU Performance Benchmarking

Executes CPU baseline, Naive CUDA kernel, and Shared-Memory Tiled CUDA kernel
across multiple matrix dimensions, records actual measured hardware performance,
and exports results to CSV.

Provides side-by-side comparative benchmarking between Naive and Tiled implementations
on identical matrix dimensions and iteration counts.
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
NAIVE_BIN = BASE_DIR / "src" / f"matmul_naive{EXE_EXT}"
TILED_BIN = BASE_DIR / "src" / f"matmul_tiled{EXE_EXT}"

# Fallbacks for current working directory
if not CPU_BIN.exists() and (BASE_DIR / f"cpu_baseline{EXE_EXT}").exists():
    CPU_BIN = BASE_DIR / f"cpu_baseline{EXE_EXT}"
if not NAIVE_BIN.exists() and (BASE_DIR / f"matmul_naive{EXE_EXT}").exists():
    NAIVE_BIN = BASE_DIR / f"matmul_naive{EXE_EXT}"
if not TILED_BIN.exists() and (BASE_DIR / f"matmul_tiled{EXE_EXT}").exists():
    TILED_BIN = BASE_DIR / f"matmul_tiled{EXE_EXT}"


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


def run_cuda_kernel_sweep(binary_path, kernel_name, csv_filename, sizes, dim, iters):
    print("\n" + "=" * 70)
    print(f" Running {kernel_name} Benchmark Sweep")
    print("=" * 70)
    if not binary_path.exists():
        print(f"NOTICE: {kernel_name} binary not found at {binary_path}.")
        print("GPU benchmarks require an environment with the NVIDIA CUDA Toolkit (nvcc).")
        print("To run on Google Colab:")
        src_name = binary_path.stem
        print(f"  1. Compile: nvcc -O3 src/{src_name}.cu -o src/{src_name}")
        print(f"  2. Run:     python benchmarks/run_benchmarks.py --cuda-only")
        return None

    csv_path = BENCHMARK_DIR / csv_filename
    results = []

    print(f"{'Size N':<8} | {'H2D (ms)':<10} | {'Kernel (ms)':<12} | {'D2H (ms)':<10} | {'Total (ms)':<10} | {'GFLOPS':<10} | {'Verify':<8}")
    print("-" * 70)

    for N in sizes:
        cmd = [str(binary_path), str(N), str(dim), str(iters), "--csv"]
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
                    results.append(parts)
            else:
                print(f"Failed for N={N} (exit code {res.returncode}):")
                if res.stdout:
                    print(f"  [stdout]:\n{res.stdout.strip()}")
                if res.stderr:
                    print(f"  [stderr]:\n{res.stderr.strip()}")
        except subprocess.TimeoutExpired:
            print(f"Timeout expired for N={N}")

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("N,dim,h2d_ms,min_kernel_ms,avg_kernel_ms,d2h_ms,total_pipeline_ms,gflops,verified,max_diff\n")
        for parts in results:
            f.write(",".join(parts) + "\n")

    print("-" * 70)
    print(f"Recorded {len(results)} {kernel_name} data points to: {csv_path}")
    return results


def run_comparative_benchmarks(sizes, dim, iters):
    print("\n" + "=" * 84)
    print(" Stage 1 vs. Stage 2 Comparative Benchmark (Naive vs. Shared-Memory Tiled GEMM)")
    print("=" * 84)

    if not NAIVE_BIN.exists() or not TILED_BIN.exists():
        print("NOTICE: Comparative benchmark requires both GPU binaries.")
        if not NAIVE_BIN.exists():
            print(f"  Missing Naive binary: {NAIVE_BIN}")
        if not TILED_BIN.exists():
            print(f"  Missing Tiled binary: {TILED_BIN}")
        print("\nTo compile and run on Google Colab:")
        print("  nvcc -O3 src/matmul_naive.cu -o src/matmul_naive")
        print("  nvcc -O3 src/matmul_tiled.cu -o src/matmul_tiled")
        print("  python benchmarks/run_benchmarks.py --compare\n")
        return False

    comparison_results = []
    csv_path = BENCHMARK_DIR / "kernel_comparison_results.csv"

    print(f"{'Size N':<8} | {'Naive (ms)':<12} | {'Tiled (ms)':<12} | {'Naive GFLOPS':<14} | {'Tiled GFLOPS':<14} | {'Speedup':<9} | {'Tiled Verify':<12}")
    print("-" * 84)

    for N in sizes:
        # 1. Run Naive
        cmd_naive = [str(NAIVE_BIN), str(N), str(dim), str(iters), "--csv"]
        cmd_tiled = [str(TILED_BIN), str(N), str(dim), str(iters), "--csv"]

        try:
            res_naive = subprocess.run(cmd_naive, capture_output=True, text=True, timeout=180)
            res_tiled = subprocess.run(cmd_tiled, capture_output=True, text=True, timeout=180)

            if res_naive.returncode == 0 and res_tiled.returncode == 0:
                parts_n = res_naive.stdout.strip().split("\n")[-1].split(",")
                parts_t = res_tiled.stdout.strip().split("\n")[-1].split(",")

                naive_time = float(parts_n[3])
                tiled_time = float(parts_t[3])
                naive_gflops = float(parts_n[7])
                tiled_gflops = float(parts_t[7])
                tiled_status = parts_t[8]
                speedup = naive_time / tiled_time if tiled_time > 0 else 0.0

                print(f"{N:<8} | {naive_time:<12.4f} | {tiled_time:<12.4f} | {naive_gflops:<14.2f} | {tiled_gflops:<14.2f} | {speedup:<8.2f}x | {tiled_status:<12}")

                comparison_results.append({
                    "N": N,
                    "dim": dim,
                    "naive_time_ms": naive_time,
                    "tiled_time_ms": tiled_time,
                    "naive_gflops": naive_gflops,
                    "tiled_gflops": tiled_gflops,
                    "speedup": speedup,
                    "tiled_verified": tiled_status,
                    "tiled_max_diff": parts_t[9]
                })
            else:
                print(f"{N:<8} | Failed execution during comparison")
        except subprocess.TimeoutExpired:
            print(f"{N:<8} | Timeout expired during comparison")

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("N,dim,naive_time_ms,tiled_time_ms,naive_gflops,tiled_gflops,speedup,tiled_verified,tiled_max_diff\n")
        for item in comparison_results:
            f.write(f"{item['N']},{item['dim']},{item['naive_time_ms']:.4f},{item['tiled_time_ms']:.4f},"
                    f"{item['naive_gflops']:.2f},{item['tiled_gflops']:.2f},{item['speedup']:.2f},"
                    f"{item['tiled_verified']},{item['tiled_max_diff']}\n")

    print("-" * 84)
    print(f"Recorded {len(comparison_results)} comparative benchmark rows to: {csv_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="CUDA Kernel Lab -- Stage 1 & Stage 2 Benchmark Sweeper")
    parser.add_argument("--sizes", nargs="+", type=int, default=[64, 128, 256, 512, 1024, 2048],
                        help="Matrix sizes N to benchmark (default: 64 128 256 512 1024 2048)")
    parser.add_argument("--iters", type=int, default=3, help="Iterations per size (default: 3)")
    parser.add_argument("--block-dim", "--tile-dim", dest="dim", type=int, default=16,
                        help="Block / Tile dimension (default: 16)")
    parser.add_argument("--cpu-only", action="store_true", help="Only run CPU baseline")
    parser.add_argument("--cuda-only", action="store_true", help="Run both Naive and Tiled CUDA benchmarks")
    parser.add_argument("--naive-only", action="store_true", help="Only run Naive CUDA kernel")
    parser.add_argument("--tiled-only", action="store_true", help="Only run Tiled CUDA kernel")
    parser.add_argument("--compare", action="store_true", help="Run comparative benchmark between Naive and Tiled")

    args = parser.parse_args()

    print("======================================================================")
    print(" CUDA Kernel Lab -- Performance Benchmark Sweeper (Stage 1 & Stage 2)")
    print("======================================================================")
    print(f"Target Matrix Sizes : {args.sizes}")
    print(f"Iterations per run  : {args.iters}")
    print(f"Block / Tile Dim    : {args.dim}")

    if args.cpu_only:
        run_cpu_benchmarks(args.sizes, args.iters)
        return

    if args.naive_only:
        run_cuda_kernel_sweep(NAIVE_BIN, "Naive CUDA Kernel", "cuda_benchmark_results.csv",
                              args.sizes, args.dim, args.iters)
        return

    if args.tiled_only:
        run_cuda_kernel_sweep(TILED_BIN, "Shared-Memory Tiled CUDA Kernel", "cuda_tiled_benchmark_results.csv",
                              args.sizes, args.dim, args.iters)
        return

    if args.compare:
        run_comparative_benchmarks(args.sizes, args.dim, args.iters)
        return

    # Default / --cuda-only: Run available GPU benchmarks and comparison
    naive_available = NAIVE_BIN.exists()
    tiled_available = TILED_BIN.exists()

    if not naive_available and not tiled_available:
        print("\n[NOTICE] CUDA GPU binaries not detected on this host (nvcc required).")
        print("To run GPU benchmarks on Google Colab (NVIDIA Tesla T4):")
        print("  1. Compile Naive: nvcc -O3 src/matmul_naive.cu -o src/matmul_naive")
        print("  2. Compile Tiled: nvcc -O3 src/matmul_tiled.cu -o src/matmul_tiled")
        print("  3. Run Sweep:     python benchmarks/run_benchmarks.py --cuda-only")
        print("  4. Run Compare:   python benchmarks/run_benchmarks.py --compare\n")
        return

    if naive_available:
        run_cuda_kernel_sweep(NAIVE_BIN, "Naive CUDA Kernel", "cuda_benchmark_results.csv",
                              args.sizes, args.dim, args.iters)

    if tiled_available:
        run_cuda_kernel_sweep(TILED_BIN, "Shared-Memory Tiled CUDA Kernel", "cuda_tiled_benchmark_results.csv",
                              args.sizes, args.dim, args.iters)

    if naive_available and tiled_available:
        run_comparative_benchmarks(args.sizes, args.dim, args.iters)


if __name__ == "__main__":
    main()
