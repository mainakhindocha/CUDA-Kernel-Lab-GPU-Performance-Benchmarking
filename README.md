# CUDA Kernel Lab — GPU Performance Benchmarking
## Stage 1 & Stage 2: CPU Baseline, Naive CUDA GEMM & Shared-Memory Tiled GEMM

A systematic high-performance computing laboratory designed for an **NVIDIA Hardware & Compute Architecture** portfolio. This repository studies the execution mechanics, memory access patterns, hardware bottlenecks, and computational scaling of General Matrix Multiplication (GEMM) from first principles.

---

## 📌 Project Stages & Scope

1. **Stage 1 — CPU Baseline vs. Naive CUDA GEMM:**
   - **CPU Baseline (`src/cpu_baseline.cpp`):** Single-threaded CPU implementation with deterministic initialization and high-resolution timing.
   - **Naive CUDA Kernel (`src/matmul_naive.cu`):** Direct 2D thread-to-element mapping, boundary handling, isolated PCIe/kernel CUDA event timing, and numerical verification.
2. **Stage 2 — Shared-Memory Tiled GEMM (`src/matmul_tiled.cu`):**
   - **Shared-Memory Caching:** Cooperatively loads $16 \times 16$ (or $32 \times 32$, $8 \times 8$) sub-matrices into on-chip shared memory (SRAM).
   - **Two-Phase Synchronization Barriers:** Correct use of `__syncthreads()` before consuming tiles and before reusing tile buffers.
   - **Arbitrary Boundary Handling:** Zero-padded cooperative loads to safely execute arbitrary positive matrix sizes (e.g. 17, 37, 311, 1024, 2048).
   - **Comparative Benchmarking:** Head-to-head performance comparisons against the Stage 1 naive baseline under identical sizes and iteration counts.

---

## 📁 Repository Structure

```text
cuda-kernel-lab/
├── .gitignore                  # Ignores compiled binaries, temporary files, and local CSV data
├── README.md                   # Complete architectural guide, compilation, and Colab runbook
├── src/
│   ├── cpu_baseline.cpp        # Self-contained C++ single-threaded CPU reference
│   ├── matmul_naive.cu         # Stage 1: Naive global-memory CUDA kernel with phase timing
│   └── matmul_tiled.cu         # Stage 2: Shared-memory tiled CUDA kernel with phase timing
├── tests/
│   ├── test_plan.md            # Comprehensive test matrix (47 test configurations)
│   ├── run_tests.py            # Automated cross-platform test script (CPU, Naive, Tiled)
│   └── run_tests.sh            # Bash test runner for Linux and Google Colab
└── benchmarks/
    ├── run_benchmarks.py       # Multi-size sweeper and Naive vs. Tiled comparative harness
    ├── run_benchmarks.sh       # Bash benchmark runner
    └── cpu_benchmark_results.csv # Locally measured CPU execution baseline results
```

---

## 🧮 Mathematical Formulation & Complexity

We compute the single-precision square matrix product:
$$\mathbf{C} = \mathbf{A} \times \mathbf{B}$$

Where $\mathbf{A}, \mathbf{B}, \mathbf{C} \in \mathbb{R}^{N \times N}$ stored in contiguous row-major layout:
$$\mathbf{C}[i, j] = \sum_{k=0}^{N-1} \mathbf{A}[i, k] \times \mathbf{B}[k, j]$$

- **Computational Workload:** $2 N^3$ Floating-Point Operations ($1$ multiply $+ 1$ add per inner step $k$).
- **Memory Footprint:** $3 \times N^2 \times 4\text{ bytes}$ ($3N^2$ single-precision floats).

---

## ⚙️ Architectural Analysis: Naive vs. Tiled GEMM

### 1. Stage 1: Naive CUDA GEMM (`src/matmul_naive.cu`)
- **Execution Model:** Each thread $(row, col)$ calculates one element of $\mathbf{C}$ by reading all $N$ elements of row $row$ in $\mathbf{A}$ and column $col$ in $\mathbf{B}$ directly from off-chip DRAM (Global Memory).
- **The Bottleneck:** DRAM latency is approximately 400–800 clock cycles. For an $N \times N$ matrix, each element of $\mathbf{A}$ and $\mathbf{B}$ is fetched from DRAM $N$ times by distinct threads.
- **Arithmetic Intensity:**
  $$\text{Arithmetic Intensity}_{\text{naive}} = \frac{2 \text{ FLOPs}}{2 \times 4 \text{ Bytes}} = 0.25 \text{ FLOP / Byte}$$
  This is far below the ridge-point of modern GPUs (e.g. Tesla T4 requires $\approx 60+\text{ FLOP/Byte}$ to saturate compute cores), making the naive kernel memory-bandwidth bound.

### 2. Stage 2: Shared-Memory Tiled GEMM (`src/matmul_tiled.cu`)
- **Execution Model:** Partitions the $N \times N$ matrices into tiles of dimension $T \times T$ (default $16 \times 16$).
- **Cooperative Shared Memory Caching:**
  - In each phase $m \in [0, \lceil N / T \rceil - 1]$, threads within a block cooperatively load tile $m$ of $\mathbf{A}$ into `__shared__ float s_A[T][T]` and tile $m$ of $\mathbf{B}$ into `__shared__ float s_B[T][T]`.
  - Shared memory resides on-chip with $\approx 20\text{–}30$ cycle access latency and orders of magnitude higher aggregate bandwidth than global memory.
  - Each element loaded once into shared memory is reused $T$ times by all $T$ threads in the row/column of the thread block.
- **Arithmetic Intensity Boost:**
  $$\text{Global Loads per Element} = \frac{2 N}{T}$$
  $$\text{Arithmetic Intensity}_{\text{tiled}} \approx T \times \text{Arithmetic Intensity}_{\text{naive}} = 16 \times 0.25 = 4.0 \text{ FLOP / Byte}$$
  Global DRAM memory traffic is reduced by a factor of $T$ (16× for $16 \times 16$ tiles).

### 3. Synchronization Barrier Protocol (`__syncthreads()`)
Tiled GEMM requires exactly two synchronization barriers per phase tile:
1. **Barrier 1 (Post-Load):**
   ```cpp
   __syncthreads();
   ```
   *Purpose:* Guarantees that all $T \times T$ threads in the block have finished writing their assigned elements into `s_A` and `s_B` before any thread begins computing partial products. Prevents **Read-After-Write (RAW)** hazards.
2. **Barrier 2 (Post-Compute):**
   ```cpp
   __syncthreads();
   ```
   *Purpose:* Guarantees that all threads have completed multiplying `s_A` and `s_B` for phase $m$ before any fast thread proceeds to the next iteration $m+1$ and overwrites `s_A` and `s_B`. Prevents **Write-After-Read (WAR)** hazards.

### 4. Boundary Handling for Arbitrary Dimensions
To support arbitrary matrix dimensions $N$ (e.g. non-multiples like 17, 37, 311, 1024, 2048):
- **Cooperative Load Zero-Padding:**
  ```cpp
  int a_col = m * TILE_DIM + tx;
  s_A[ty][tx] = (row < N && a_col < N) ? A[row * N + a_col] : 0.0f;

  int b_row = m * TILE_DIM + ty;
  s_B[ty][tx] = (b_row < N && col < N) ? B[b_row * N + col] : 0.0f;
  ```
- **Deadlock-Free Execution:** Because the boundary condition is handled by zero-padding rather than divergent branching around `__syncthreads()`, **every thread unconditionally enters the barrier**, guaranteeing zero synchronization deadlocks.
- **Write-Back Guard:** Output writes to global memory are guarded:
  ```cpp
  if (row < N && col < N) {
      C[row * N + col] = p_value;
  }
  ```

---

## 🔨 Compilation Instructions (No CMake)

### Compile CPU Baseline (`g++`)
```bash
g++ -O3 -std=c++14 src/cpu_baseline.cpp -o src/cpu_baseline
```

### Compile Naive CUDA Kernel (`nvcc`)
```bash
nvcc -O3 src/matmul_naive.cu -o src/matmul_naive
```

### Compile Shared-Memory Tiled CUDA Kernel (`nvcc`)
```bash
nvcc -O3 src/matmul_tiled.cu -o src/matmul_tiled
```

---

## 🧪 Correctness & Edge-Case Testing

The automated test runner (`tests/run_tests.py`) exercises **47 distinct test cases**:
1. **Part 1 — CPU Baseline (11 tests):** Unit sizes, power-of-2 sizes, edge cases (17, 37, 100), and invalid argument rejection.
2. **Part 2 — Naive CUDA Kernel (17 tests):** Sizes from $1 \times 1$ to $2048 \times 2048$, custom block dimensions (8, 16, 32), boundary dimensions, and full/sampled numerical verification.
3. **Part 3 — Tiled CUDA Kernel (19 tests):** Exact tile fits, non-divisible partial tiles (17, 37, 311), custom tile dimensions (8, 16, 32), boundary full verification ($1024 \times 1024$), large sampled verification ($2048 \times 2048$), and invalid input checks.

### Running the Test Suite
```bash
python tests/run_tests.py
# Or via Bash:
bash tests/run_tests.sh
```
*Note: In environments without an NVIDIA GPU or `nvcc`, the 11 CPU baseline tests execute and pass locally, while GPU tests are clearly flagged with instructions for running in Google Colab.*

---

## ⏱️ Verification Methodology & Tolerances

- **Full Verification ($N \le 1024$):** Exhaustive comparison across all $N \times N$ elements between GPU output and golden single-threaded CPU reference.
- **Sampled Verification ($N > 1024$):** Evaluates at least $257$ deterministically selected output elements across corners, edges, centers, octile grid intersections, and tile boundaries using independent CPU scalar dot products to prevent prohibitive $O(N^3)$ CPU execution times.
- **Numerical Tolerances:**
  $$\left| C_{\text{cpu}}[i] - C_{\text{gpu}}[i] \right| \le \text{atol} + \text{rtol} \times \left| C_{\text{cpu}}[i] \right|$$
  - Absolute tolerance: $\text{atol} = 1.0 \times 10^{-4}$
  - Relative tolerance: $\text{rtol} = 1.0 \times 10^{-3}$
- **Non-Finite Detection:** Explicitly flags any `NaN` or `Inf` in GPU outputs.

---

## 📊 Benchmark Sweeper & Comparative Harness

The benchmark sweeper (`benchmarks/run_benchmarks.py`) executes multi-size sweeps and exports real hardware measurements:

### Benchmark Options
```bash
# 1. Run comparative benchmark between Naive and Tiled (identical sizes & iterations)
python benchmarks/run_benchmarks.py --compare --sizes 64 128 256 512 1024 2048 --iters 3

# 2. Run only Tiled CUDA sweep
python benchmarks/run_benchmarks.py --tiled-only --sizes 64 128 256 512 1024 2048

# 3. Run only Naive CUDA sweep
python benchmarks/run_benchmarks.py --naive-only --sizes 64 128 256 512 1024 2048

# 4. Run only CPU sweep
python benchmarks/run_benchmarks.py --cpu-only --sizes 64 128 256 512 1024
```

### Performance Integrity Notice
> **Empirical Rigor:** Performance speedups and GFLOPS throughput figures are never fabricated or estimated. All comparative numbers must be collected through live execution on actual GPU hardware (such as an NVIDIA Tesla T4 in Google Colab).

---

## ☁️ Google Colab Execution Guide (NVIDIA Tesla T4)

To run the complete Stage 1 & Stage 2 verification and benchmarking on Google Colab:

### 1. Set Hardware Accelerator
- In Google Colab, select **Runtime** $\rightarrow$ **Change runtime type** $\rightarrow$ **T4 GPU** $\rightarrow$ **Save**.

### 2. Clone the Repository
```bash
!git clone https://github.com/mainakhindocha/CUDA-Kernel-Lab-GPU-Performance-Benchmarking.git
%cd CUDA-Kernel-Lab-GPU-Performance-Benchmarking
```

### 3. Compile All Kernels
```bash
!g++ -O3 -std=c++14 src/cpu_baseline.cpp -o src/cpu_baseline
!nvcc -O3 src/matmul_naive.cu -o src/matmul_naive
!nvcc -O3 src/matmul_tiled.cu -o src/matmul_tiled
```

### 4. Execute the Full Correctness Test Suite (47 Tests)
```bash
!python tests/run_tests.py
```

### 5. Run Stage 1 vs. Stage 2 Comparative Benchmarking
```bash
!python benchmarks/run_benchmarks.py --compare --sizes 64 128 256 512 1024 2048 --iters 3
```
This generates `benchmarks/kernel_comparison_results.csv` and outputs a side-by-side performance table detailing Naive GFLOPS, Tiled GFLOPS, measured speedup, and numerical verification status.
