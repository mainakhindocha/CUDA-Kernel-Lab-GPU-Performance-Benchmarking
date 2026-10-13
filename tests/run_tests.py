#!/usr/bin/env python3
"""
Stage 1 & Stage 2 Test Suite Runner
Project: CUDA Kernel Lab — GPU Performance Benchmarking

Validates:
1. CPU Reference Baseline (src/cpu_baseline)
2. Naive CUDA Kernel (src/matmul_naive)
3. Shared-Memory Tiled CUDA Kernel (src/matmul_tiled)
Across standard sizes, non-multiple boundary dimensions, prime sizes,
invalid parameters, and large sampled verifications.
"""

import sys
import subprocess
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Detect binary paths based on OS
EXE_EXT = ".exe" if sys.platform == "win32" else ""
CPU_BIN = BASE_DIR / "src" / f"cpu_baseline{EXE_EXT}"
NAIVE_BIN = BASE_DIR / "src" / f"matmul_naive{EXE_EXT}"
TILED_BIN = BASE_DIR / "src" / f"matmul_tiled{EXE_EXT}"

# Fallback check in current working directory or build directories
if not CPU_BIN.exists() and (BASE_DIR / f"cpu_baseline{EXE_EXT}").exists():
    CPU_BIN = BASE_DIR / f"cpu_baseline{EXE_EXT}"
if not NAIVE_BIN.exists() and (BASE_DIR / f"matmul_naive{EXE_EXT}").exists():
    NAIVE_BIN = BASE_DIR / f"matmul_naive{EXE_EXT}"
if not TILED_BIN.exists() and (BASE_DIR / f"matmul_tiled{EXE_EXT}").exists():
    TILED_BIN = BASE_DIR / f"matmul_tiled{EXE_EXT}"


def run_command(cmd, expected_returncode=0):
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        passed = (res.returncode == expected_returncode)
        return passed, res.returncode, res.stdout, res.stderr
    except Exception as e:
        return False, -1, "", str(e)


def main():
    print("=" * 72)
    print(" CUDA Kernel Lab — Stage 1 & Stage 2 Correctness Test Suite")
    print("=" * 72)

    total_tests = 0
    passed_tests = 0
    skipped_gpu_tests = 0

    # -------------------------------------------------------------
    # 1. CPU Baseline Functional & Input Validation Tests
    # -------------------------------------------------------------
    print("\n[PART 1] Testing CPU Baseline (src/cpu_baseline)")
    print("-" * 72)

    if not CPU_BIN.exists():
        print(f"Warning: CPU binary not found at {CPU_BIN}.")
        print("Please compile first: g++ -O3 -std=c++14 src/cpu_baseline.cpp -o src/cpu_baseline")
        sys.exit(1)

    cpu_test_cases = [
        # (Description, Args, Expected ReturnCode)
        ("T-CPU-01: Single element (1x1)", ["1", "1"], 0),
        ("T-CPU-02: Small power-of-2 (16x16)", ["16", "2"], 0),
        ("T-CPU-03: Small power-of-2 (32x32)", ["32", "2"], 0),
        ("T-CPU-04: Non-multiple edge (17x17)", ["17", "2"], 0),
        ("T-CPU-05: Non-multiple prime (37x37)", ["37", "2"], 0),
        ("T-CPU-06: Non-multiple decimal (100x100)", ["100", "2"], 0),
        ("T-CPU-07: Standard size (128x128)", ["128", "2"], 0),
        ("T-CPU-08: Moderate size (256x256)", ["256", "1"], 0),
        ("E-CPU-01: Negative matrix size rejection", ["-5"], 1),
        ("E-CPU-02: Zero matrix size rejection", ["0"], 1),
        ("E-CPU-03: Non-numeric argument rejection", ["abc"], 1),
    ]

    for desc, args, exp_rc in cpu_test_cases:
        total_tests += 1
        cmd = [str(CPU_BIN)] + args
        passed, rc, stdout, stderr = run_command(cmd, exp_rc)
        if passed:
            passed_tests += 1
            status = "[PASS]"
        else:
            status = f"[FAIL] (Got exit code {rc}, expected {exp_rc})"
            if stderr:
                status += f"\n    stderr: {stderr.strip()}"
        print(f"  {desc:<52} ... {status}")

    # -------------------------------------------------------------
    # 2. Naive CUDA Kernel Functional, Boundary & Sampled Tests
    # -------------------------------------------------------------
    print("\n[PART 2] Testing Naive CUDA Kernel (src/matmul_naive)")
    print("-" * 72)

    naive_test_cases = [
        # (Description, Args: [N, block_dim, iters], Expected ReturnCode)
        ("T-GPU-01: Single element (1x1)", ["1", "16", "1"], 0),
        ("T-GPU-02: Small power-of-2 (16x16, block 16)", ["16", "16", "2"], 0),
        ("T-GPU-03: Small power-of-2 (32x32, block 16)", ["32", "16", "2"], 0),
        ("T-GPU-04: Non-multiple edge (17x17, block 16)", ["17", "16", "2"], 0),
        ("T-GPU-05: Non-multiple odd (37x37, block 16)", ["37", "16", "2"], 0),
        ("T-GPU-06: Non-multiple decimal (100x100, block 16)", ["100", "16", "2"], 0),
        ("T-GPU-07: Non-multiple prime (311x311, block 16)", ["311", "16", "2"], 0),
        ("T-GPU-08: Standard size full verify (128x128)", ["128", "16", "2"], 0),
        ("T-GPU-09: Moderate size full verify (256x256)", ["256", "16", "2"], 0),
        ("T-GPU-10: Large full verified size (512x512)", ["512", "16", "2"], 0),
        ("T-GPU-11: Custom block dimension (128x128, block 32)", ["128", "32", "2"], 0),
        ("T-GPU-12: Custom block dimension (128x128, block 8)", ["128", "8", "2"], 0),
        ("T-GPU-13: Large sampled verified size (2048x2048)", ["2048", "16", "1"], 0),
        ("T-GPU-14: Explicit skip verification (--no-verify)", ["128", "16", "1", "--no-verify"], 0),
        ("E-GPU-01: Negative matrix size rejection", ["-10"], 1),
        ("E-GPU-02: Invalid block dim rejection (0)", ["128", "0"], 1),
        ("E-GPU-03: Exceeded block dim rejection (64)", ["128", "64"], 1),
    ]

    if not NAIVE_BIN.exists():
        skipped_gpu_tests += len(naive_test_cases)
        print(f"  [NOTICE] Naive CUDA binary not found at {NAIVE_BIN}.")
        print("  CUDA execution could not be verified locally because nvcc is not installed.")
        print("  To run GPU tests on Google Colab or an NVIDIA-enabled machine:")
        print("    nvcc -O3 src/matmul_naive.cu -o src/matmul_naive\n")
    else:
        for desc, args, exp_rc in naive_test_cases:
            total_tests += 1
            cmd = [str(NAIVE_BIN)] + args
            passed, rc, stdout, stderr = run_command(cmd, exp_rc)
            if passed:
                passed_tests += 1
                status = "[PASS]"
            else:
                status = f"[FAIL] (Got exit code {rc}, expected {exp_rc})"
                if stderr:
                    status += f"\n    stderr: {stderr.strip()}"
            print(f"  {desc:<52} ... {status}")

    # -------------------------------------------------------------
    # 3. Shared-Memory Tiled CUDA Kernel Functional & Boundary Tests
    # -------------------------------------------------------------
    print("\n[PART 3] Testing Shared-Memory Tiled CUDA Kernel (src/matmul_tiled)")
    print("-" * 72)

    tiled_test_cases = [
        # (Description, Args: [N, tile_dim, iters], Expected ReturnCode)
        ("T-TILED-01: Single element (1x1, tile 16)", ["1", "16", "1"], 0),
        ("T-TILED-02: Small power-of-2 (16x16, tile 16)", ["16", "16", "2"], 0),
        ("T-TILED-03: Small power-of-2 (32x32, tile 16)", ["32", "16", "2"], 0),
        ("T-TILED-04: Non-multiple edge (17x17, tile 16)", ["17", "16", "2"], 0),
        ("T-TILED-05: Non-multiple odd (37x37, tile 16)", ["37", "16", "2"], 0),
        ("T-TILED-06: Non-multiple decimal (100x100, tile 16)", ["100", "16", "2"], 0),
        ("T-TILED-07: Non-multiple prime (311x311, tile 16)", ["311", "16", "2"], 0),
        ("T-TILED-08: Standard size full verify (128x128, tile 16)", ["128", "16", "2"], 0),
        ("T-TILED-09: Moderate size full verify (256x256, tile 16)", ["256", "16", "2"], 0),
        ("T-TILED-10: Large full verified size (512x512, tile 16)", ["512", "16", "2"], 0),
        ("T-TILED-11: Boundary full verify (1024x1024, tile 16)", ["1024", "16", "1"], 0),
        ("T-TILED-12: Custom tile dimension (128x128, tile 32)", ["128", "32", "2"], 0),
        ("T-TILED-13: Custom tile dimension (128x128, tile 8)", ["128", "8", "2"], 0),
        ("T-TILED-14: Large sampled verified size (2048x2048, tile 16)", ["2048", "16", "1"], 0),
        ("T-TILED-15: Explicit skip verification (--no-verify)", ["128", "16", "1", "--no-verify"], 0),
        ("E-TILED-01: Negative matrix size rejection", ["-10"], 1),
        ("E-TILED-02: Invalid tile dim rejection (0)", ["128", "0"], 1),
        ("E-TILED-03: Exceeded tile dim rejection (64)", ["128", "64"], 1),
        ("E-TILED-04: Unsupported tile dim rejection (24)", ["128", "24"], 1),
    ]

    if not TILED_BIN.exists():
        skipped_gpu_tests += len(tiled_test_cases)
        print(f"  [NOTICE] Tiled CUDA binary not found at {TILED_BIN}.")
        print("  CUDA execution could not be verified locally because nvcc is not installed.")
        print("  To run GPU tests on Google Colab or an NVIDIA-enabled machine:")
        print("    nvcc -O3 src/matmul_tiled.cu -o src/matmul_tiled\n")
    else:
        for desc, args, exp_rc in tiled_test_cases:
            total_tests += 1
            cmd = [str(TILED_BIN)] + args
            passed, rc, stdout, stderr = run_command(cmd, exp_rc)
            if passed:
                passed_tests += 1
                status = "[PASS]"
            else:
                status = f"[FAIL] (Got exit code {rc}, expected {exp_rc})"
                if stderr:
                    status += f"\n    stderr: {stderr.strip()}"
            print(f"  {desc:<52} ... {status}")

    print("=" * 72)
    print(f" Test Results: {passed_tests} / {total_tests} executed tests passed.")
    if skipped_gpu_tests > 0:
        print(f" [Notice] {skipped_gpu_tests} GPU tests were skipped locally because nvcc is not installed.")
        print(" To run the full suite (47 tests) on Google Colab:")
        print("   1. Compile Naive: nvcc -O3 src/matmul_naive.cu -o src/matmul_naive")
        print("   2. Compile Tiled: nvcc -O3 src/matmul_tiled.cu -o src/matmul_tiled")
        print("   3. Run Suite:     python tests/run_tests.py")
    print("=" * 72)

    return 0 if (passed_tests == total_tests) else 1


if __name__ == "__main__":
    sys.exit(main())
