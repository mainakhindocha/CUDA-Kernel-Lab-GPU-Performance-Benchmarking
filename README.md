# CUDA Kernel Lab — GPU Performance Benchmarking
## Stage 1: CPU Baseline vs. Naive CUDA GEMM

A systematic high-performance computing laboratory designed for an **NVIDIA Hardware & Compute Architecture** internship portfolio. This repository studies the execution mechanics, memory access patterns, and computational scaling of General Matrix Multiplication (GEMM) starting from first principles.

---

## 📌 Stage 1 Scope & Objectives

The goal of **Stage 1** is to establish rigorous, unoptimized reference implementations and benchmarking harnesses:
1. **CPU Reference (`src/cpu_baseline.cpp`):** Standard C++ single-threaded baseline matrix multiplication with deterministic initialization and `std::chrono` timing.
2. **Naive CUDA Kernel (`src/matmul_naive.cu`):** Direct 2D grid/block thread-to-element mapping, boundary handling for non-multiple matrix dimensions, CUDA error checking, isolated PCIe vs. kernel CUDA event timing, and floating-point verification.
3. **Correctness & Edge-Case Testing (`tests/`):** Automated test plan and execution script covering small matrices, boundary non-divisible dimensions, and invalid inputs.
4. **Benchmarking Harness (`benchmarks/`):** Scripted execution across multiple dimensions ($N=64$ to $1024+$), logging measured results to CSV.

---

## 📁 Repository Structure (Stage 1)

```text
cuda-kernel-lab/
├── src/
│   ├── cpu_baseline.cpp        # Self-contained C++ CPU matrix multiplication baseline
│   └── matmul_naive.cu         # Self-contained Naive CUDA kernel with phase timing
├── tests/
│   ├── test_plan.md            # Test matrix, boundary conditions, and tolerance criteria
│   ├── run_tests.py            # Automated cross-platform test script
│   └── run_tests.sh            # Bash test runner for Linux and Google Colab
├── benchmarks/
│   ├── run_benchmarks.py       # Multi-size benchmark sweeper with CSV export
│   ├── run_benchmarks.sh       # Bash benchmark runner
│   └── cpu_benchmark_results.csv # Real measured CPU execution results
└── README.md                   # Stage 1 documentation and Colab guide
```

---

## 🧮 Mathematical Formulation & Complexity

We compute the single-precision square matrix product:
$$\mathbf{C} = \mathbf{A} \times \mathbf{B}$$

Where $\mathbf{A}, \mathbf{B}, \mathbf{C} \in \mathbb{R}^{N \times N}$ stored in contiguous row-major format:
$$\mathbf{C}[i, j] = \sum_{k=0}^{N-1} \mathbf{A}[i, k] \times \mathbf{B}[k, j]$$

- **Computational Workload:** $2 N^3$ Floating-Point Operations ($1$ multiplication $+ 1$ addition per inner iteration $k$).
- **Memory Footprint:** $3 \times N^2 \times 4\text{ bytes}$ ($3N^2$ single-precision floats).

---

## ⚙️ Algorithms & Implementation Details

### 1. CPU Baseline (`src/cpu_baseline.cpp`)
- **Algorithm:** Canonical triple-nested loop ($i$-$j$-$k$ order).
- **Initialization:** Deterministic pseudorandom generation using `std::mt19937` with fixed seeds (`42` for $\mathbf{A}$, `84` for $\mathbf{B}$) to guarantee exact numerical reproducibility across runs and environments.
- **Timing:** High-resolution wall-clock measurement via `std::chrono::high_resolution_clock`.
- **Validation:** Enforces strictly positive dimensions ($N > 0$) and iterations ($> 0$), exiting with status code `1` on invalid arguments.

### 2. Naive CUDA Kernel (`src/matmul_naive.cu`)
- **Thread Mapping:** 2D Grid of 2D Thread Blocks (default $16 \times 16 = 256$ threads/block). Each thread is mapped to a single output element $(row, col)$:
  ```cpp
  int col = blockIdx.x * blockDim.x + threadIdx.x; // Column (x -> horizontal)
  int row = blockIdx.y * blockDim.y + threadIdx.y; // Row    (y -> vertical)
  ```
- **Boundary Handling:** To support arbitrary matrix dimensions that are not multiples of the block dimension:
  ```cpp
  dim3 block(block_dim, block_dim);
  dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

  if (row < N && col < N) {
      // Execute dot product
  }
  ```
- **CUDA Error Checking:**
  - `CUDA_CHECK(call)` macro wraps every CUDA Runtime API call (`cudaMalloc`, `cudaMemcpy`, `cudaEventCreate`, etc.).
  - `CUDA_CHECK_LAST_ERROR()` checks `cudaGetLastError()` immediately after kernel launches to capture invalid grid/block dimensioning or configuration errors.
- **Phase-Isolated Timing:** Uses three pairs of `cudaEvent_t` timers to isolate memory movement from compute:
  1. **Host-to-Device (H2D) Transfer:** Measures PCIe copy of $\mathbf{A}$ and $\mathbf{B}$ to device memory.
  2. **Kernel Execution:** Measures pure GPU kernel execution time (excluding PCIe transfers).
  3. **Device-to-Host (D2H) Transfer:** Measures PCIe copy of output $\mathbf{C}$ back to host.
- **Dual-Tier Numerical Verification:**
  - **Full Verification ($N \le 1024$):** Exhaustively compares all $N^2$ elements between the GPU output and a golden CPU reference.
  - **Sampled Verification ($N > 1024$):** Evaluates at least $257$ deterministically selected output elements across matrix corners, centers, octile grid coordinates, and block boundaries using standalone CPU dot products. This eliminates the multi-second/minute CPU bottleneck of an $O(N^3)$ reference while guaranteeing rigorous numerical verification.
  - **Tolerances:** $\left| C_{\text{cpu}}[i] - C_{\text{gpu}}[i] \right| \le \text{atol} + \text{rtol} \times \left| C_{\text{cpu}}[i] \right|$ (`atol = 1e-4f`, `rtol = 1e-3f`).
  - **Non-Finite Detection:** Detects any `NaN` or `Inf` floating-point anomalies in GPU output.
  - **Timing Isolation:** Verification is strictly executed after all GPU events are synchronized and recorded, ensuring verification overhead is never added to reported GPU kernel runtime.
  - **Optional Skip:** Passing `--no-verify` allows benchmark runs without CPU verification, unambiguously logging `SKIPPED`.

---

## 🔨 Compilation Instructions (No CMake)

### Compile CPU Baseline (`g++`)
```bash
g++ -O3 -std=c++14 src/cpu_baseline.cpp -o src/cpu_baseline
```
*Run manual benchmark:*
```bash
# Usage: ./src/cpu_baseline [matrix_size N] [iterations] [--csv]
./src/cpu_baseline 512 3
```

### Compile Naive CUDA Kernel (`nvcc`)
```bash
nvcc -O3 src/matmul_naive.cu -o src/matmul_naive
```
*Run manual kernel execution:*
```bash
# Usage: ./src/matmul_naive [matrix_size N] [block_dim] [iterations] [--csv]
./src/matmul_naive 512 16 3
```

---

## 🧪 Correctness & Edge-Case Testing

The test suite in `tests/run_tests.py` validates:
1. **Small matrices:** $1 \times 1$, $16 \times 16$, $32 \times 32$.
2. **Non-multiple boundary dimensions:** $17 \times 17$, $37 \times 37$, $100 \times 100$, $311 \times 311$.
3. **Thread block dimension variations:** Block sizes $8 \times 8$, $16 \times 16$, and $32 \times 32$.
4. **Invalid inputs:** Negative dimensions (`-5`), zero dimension (`0`), non-numeric strings (`abc`), and invalid block dimensions.

*Run the test suite:*
```bash
python tests/run_tests.py
# Or on Linux:
bash tests/run_tests.sh
```

---

## ⏱️ Timing Methodology & Metrics

- **Kernel Throughput (GFLOPS):**
  $$\text{Throughput} = \frac{2 \times N^3}{\text{elapsed\_time\_seconds} \times 10^9}$$
- **Warmup:** Warmup iterations are executed before timing to prime device clocks, instruction caches, and memory controllers.
- **Iterations:** Multiple iterations (default 3) are recorded; min, average, and max times are computed to detect variance and thermal throttling.

---

## 📊 Measured CPU Baseline Benchmark Results

The following measurements were taken on the local test hardware using `g++ -O3` and recorded in `benchmarks/cpu_benchmark_results.csv`:

| Matrix Size ($N$) | Operations | Min Time (ms) | Avg Time (ms) | Max Time (ms) | Throughput (GFLOPS) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **64** | 0.52 MFLOP | $< 0.001$ ms | 0.33 ms | 1.00 ms | $> 5.0$ |
| **128** | 4.19 MFLOP | 0.94 ms | 1.29 ms | 1.96 ms | **4.48** |
| **256** | 33.55 MFLOP | 11.82 ms | 12.27 ms | 13.10 ms | **2.84** |
| **512** | 268.44 MFLOP | 94.66 ms | 95.39 ms | 95.88 ms | **2.84** |
| **1024** | 2,147.48 MFLOP | 2,527.33 ms | 2,551.43 ms | 2,568.08 ms | **0.85** |

> **CPU Architecture Observation:** As $N$ expands from $512$ to $1024$, matrix working sets exceed on-chip CPU cache (L1/L2), causing memory stalls from non-unit strides along matrix $\mathbf{B}$ and degrading CPU throughput from $2.84$ down to $0.85$ GFLOPS.

---

## ⚠️ Limitations of the Stage 1 Naive CUDA Kernel

While the naive kernel demonstrates correct parallel execution, it suffers from three critical architectural bottlenecks:
1. **Redundant High-Latency DRAM Accesses:** Each thread computes a single output by fetching $N$ elements from $\mathbf{A}$ and $N$ elements from $\mathbf{B}$ directly from off-chip DRAM (Global Memory, 400–800 cycle latency). Across all $N^2$ threads, $2 N^3$ floats are loaded from DRAM with zero data reuse between neighboring threads in the same block.
2. **Severely Low Arithmetic Intensity ($0.25\text{ FLOP / Byte}$):** For every inner multiply-accumulate (2 FLOPs), 2 floats (8 bytes) are fetched from global memory. Because modern GPUs require an arithmetic intensity of $30 - 100+\text{ FLOP / Byte}$ to saturate compute cores, the naive kernel is completely memory-bandwidth saturated.
3. **No Utilization of Shared Memory (SRAM):** On-chip shared memory is unused in Stage 1.

*These bottlenecks motivate Stage 2: Shared-Memory Tiled GEMM.*

---

## ☁️ Google Colab Execution Guide (NVIDIA GPU Runtime)

To execute and verify the CUDA kernel on an NVIDIA GPU (e.g. T4 / V100 / A100) via Google Colab:

### Step 1: Open a New Colab Notebook and Set Runtime
1. Navigate to [Google Colab](https://colab.research.google.com).
2. Go to **Runtime** $\rightarrow$ **Change runtime type**.
3. Under **Hardware accelerator**, select **T4 GPU** (or any available NVIDIA GPU) and click **Save**.

### Step 2: Verify GPU Environment in Colab
In the first Colab notebook cell:
```python
!nvidia-smi
!nvcc --version
```

### Step 3: Upload or Clone the Repository
Upload the `cuda-kernel-lab` folder or clone your repository:
```python
%cd /content
# If using git:
# !git clone https://github.com/<your-username>/cuda-kernel-lab.git
# %cd cuda-kernel-lab
```

### Step 4: Compile Both Executables
```python
!g++ -O3 -std=c++14 src/cpu_baseline.cpp -o src/cpu_baseline
!nvcc -O3 src/matmul_naive.cu -o src/matmul_naive
```

### Step 5: Run the Stage 1 Test Suite
```python
!python tests/run_tests.py
```
*Expected Output:*
```text
[PART 1] Testing CPU Baseline (src/cpu_baseline)
  T-CPU-01 ... [PASS]
  ...
[PART 2] Testing Naive CUDA Kernel (src/matmul_naive)
  T-GPU-01 to T-GPU-12 ... [PASS]
  E-GPU-01 to E-GPU-03 ... [PASS]
```

### Step 6: Run the Multi-Size Benchmark Suite
```python
!python benchmarks/run_benchmarks.py --sizes 64 128 256 512 1024 2048 --iters 3
```
This will record real, un-fabricated GPU benchmark results to `benchmarks/cuda_benchmark_results.csv` including isolated H2D, kernel, D2H times, and throughput.
