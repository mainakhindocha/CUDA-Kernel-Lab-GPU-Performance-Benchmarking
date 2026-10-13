# Test Plan — CUDA Kernel Lab: Stage 1 & Stage 2

## 1. Objective
Validate the functional correctness, numerical fidelity, boundary safety, and memory-barrier integrity of:
1. **CPU Baseline** (`src/cpu_baseline.cpp`) — Single-threaded reference
2. **Naive CUDA Kernel** (`src/matmul_naive.cu`) — Global memory 2D grid/block baseline
3. **Shared-Memory Tiled CUDA Kernel** (`src/matmul_tiled.cu`) — Cooperatively cached tiled GEMM with two-phase synchronization barriers

---

## 2. Test Matrix Configurations

### Part 1: CPU Baseline Tests (`cpu_baseline`)
| Test ID | Category | Matrix Size ($N \times N$) | Expected Result |
|:---:|:---|:---:|:---|
| **T-CPU-01** | Minimal Dimension | $1 \times 1$ | PASS |
| **T-CPU-02** | Small Power-of-2 | $16 \times 16$ | PASS |
| **T-CPU-03** | Small Power-of-2 | $32 \times 32$ | PASS |
| **T-CPU-04** | Non-Multiple Edge | $17 \times 17$ | PASS |
| **T-CPU-05** | Non-Multiple Odd Prime | $37 \times 37$ | PASS |
| **T-CPU-06** | Non-Multiple Decimal | $100 \times 100$ | PASS |
| **T-CPU-07** | Standard Power-of-2 | $128 \times 128$ | PASS |
| **T-CPU-08** | Moderate Power-of-2 | $256 \times 256$ | PASS |

### Part 2: Stage 1 Naive CUDA Kernel Tests (`matmul_naive`)
| Test ID | Category | Matrix Size ($N \times N$) | Block Dim | Verification Mode | Expected Result |
|:---:|:---|:---:|:---:|:---|:---:|
| **T-GPU-01** | Minimal Dimension | $1 \times 1$ | 16 | Full ($1$ element) | PASS (Diff $< 10^{-5}$) |
| **T-GPU-02** | Small Power-of-2 | $16 \times 16$ | 16 | Full ($256$ elements) | PASS |
| **T-GPU-03** | Small Power-of-2 | $32 \times 32$ | 16 | Full ($1024$ elements) | PASS |
| **T-GPU-04** | Non-Multiple Edge | $17 \times 17$ | 16 | Full ($289$ elements) | PASS (Boundary check guards $N$) |
| **T-GPU-05** | Non-Multiple Odd Prime | $37 \times 37$ | 16 | Full ($1369$ elements) | PASS |
| **T-GPU-06** | Non-Multiple Decimal | $100 \times 100$ | 16 | Full ($10^4$ elements) | PASS |
| **T-GPU-07** | Non-Multiple Prime | $311 \times 311$ | 16 | Full ($96,721$ elements) | PASS |
| **T-GPU-08** | Standard Power-of-2 | $128 \times 128$ | 16 | Full ($16,384$ elements) | PASS |
| **T-GPU-09** | Moderate Power-of-2 | $256 \times 256$ | 16 | Full ($65,536$ elements) | PASS |
| **T-GPU-10** | Large Full Verified | $512 \times 512$ | 16 | Full ($262,144$ elements) | PASS |
| **T-GPU-11** | Custom Block Dimension | $128 \times 128$ | 32 | Full ($16,384$ elements) | PASS (Max 1024 threads/block) |
| **T-GPU-12** | Custom Block Dimension | $128 \times 128$ | 8 | Full ($16,384$ elements) | PASS |
| **T-GPU-13** | Large Matrix Sampled | $2048 \times 2048$ | 16 | Sampled ($\ge 257$ elements) | PASS (Deterministic sampled dot products) |
| **T-GPU-14** | Explicit Skip Flag | $128 \times 128$ | 16 | Skipped (`--no-verify`) | Status reports `SKIPPED`, Exit code 0 |

### Part 3: Stage 2 Shared-Memory Tiled CUDA Kernel Tests (`matmul_tiled`)
| Test ID | Category | Matrix Size ($N \times N$) | Tile Dim | Verification Mode | Expected Result / Purpose |
|:---:|:---|:---:|:---:|:---|:---:|
| **T-TILED-01** | Minimal Dimension | $1 \times 1$ | 16 | Full ($1$ element) | PASS (Single element cooperative load) |
| **T-TILED-02** | Exact Tile Multiple | $16 \times 16$ | 16 | Full ($256$ elements) | PASS (Single tile phase) |
| **T-TILED-03** | Exact Tile Multiple | $32 \times 32$ | 16 | Full ($1024$ elements) | PASS (Multi-tile phases, exact fit) |
| **T-TILED-04** | Partial Boundary Edge | $17 \times 17$ | 16 | Full ($289$ elements) | PASS (Tile boundary $+1$ element padded) |
| **T-TILED-05** | Partial Boundary Odd | $37 \times 37$ | 16 | Full ($1369$ elements) | PASS (Arbitrary non-divisible tile boundary) |
| **T-TILED-06** | Partial Decimal Size | $100 \times 100$ | 16 | Full ($10^4$ elements) | PASS |
| **T-TILED-07** | Partial Prime Size | $311 \times 311$ | 16 | Full ($96,721$ elements) | PASS (Prime dimension partial tile safety) |
| **T-TILED-08** | Standard Power-of-2 | $128 \times 128$ | 16 | Full ($16,384$ elements) | PASS (8 tiles across K dimension) |
| **T-TILED-09** | Moderate Power-of-2 | $256 \times 256$ | 16 | Full ($65,536$ elements) | PASS (16 tiles across K dimension) |
| **T-TILED-10** | Large Full Verified | $512 \times 512$ | 16 | Full ($262,144$ elements) | PASS (32 tiles across K dimension) |
| **T-TILED-11** | Boundary Full Verified | $1024 \times 1024$ | 16 | Full ($1,048,576$ elements) | PASS (64 tiles across K dimension) |
| **T-TILED-12** | Custom Tile Dimension | $128 \times 128$ | 32 | Full ($16,384$ elements) | PASS (32x32 tile, 1024 threads, 8 KB shared mem) |
| **T-TILED-13** | Custom Tile Dimension | $128 \times 128$ | 8 | Full ($16,384$ elements) | PASS (8x8 tile, 64 threads, 512 B shared mem) |
| **T-TILED-14** | Large Matrix Sampled | $2048 \times 2048$ | 16 | Sampled ($\ge 257$ elements) | PASS (128 tiles across K dimension) |
| **T-TILED-15** | Explicit Skip Flag | $128 \times 128$ | 16 | Skipped (`--no-verify`) | Status reports `SKIPPED`, Exit code 0 |

---

## 3. Negative / Invalid Input Tests

| Test ID | Executable | Input Arguments | Expected Behavior | Expected Exit Code |
|:---:|:---|:---|:---|:---:|
| **E-CPU-01** | `cpu_baseline` | `-5` | Rejects negative matrix size | 1 |
| **E-CPU-02** | `cpu_baseline` | `0` | Rejects zero matrix size | 1 |
| **E-CPU-03** | `cpu_baseline` | `abc` | Rejects non-numeric matrix size | 1 |
| **E-GPU-01** | `matmul_naive` | `-10` | Rejects negative matrix size | 1 |
| **E-GPU-02** | `matmul_naive` | `128 0` | Rejects block dimension $\le 0$ | 1 |
| **E-GPU-03** | `matmul_naive` | `128 64` | Rejects block dimension $> 32$ | 1 |
| **E-TILED-01** | `matmul_tiled` | `-10` | Rejects negative matrix size | 1 |
| **E-TILED-02** | `matmul_tiled` | `128 0` | Rejects tile dimension $\le 0$ | 1 |
| **E-TILED-03** | `matmul_tiled` | `128 64` | Rejects tile dimension $> 32$ | 1 |
| **E-TILED-04** | `matmul_tiled` | `128 24` | Rejects unsupported tile dimension (must be 8, 16, or 32) | 1 |

---

## 4. Verification Strategies

### A. Full Verification ($N \le 1024$)
- Complete exhaustive comparison across all $N \times N$ matrix elements.
- CPU reference computed in full via standard loops.

### B. Sampled Verification ($N > 1024$)
- To avoid prohibitive $O(N^3)$ CPU compute time on large matrices (which takes minutes for $N \ge 2048$ on a single thread), both GPU kernels invoke `verify_sampled_results()`.
- Independently evaluates **at least 257 deterministically selected output elements**:
  - 4 extreme corners: $(0,0)$, $(0, N-1)$, $(N-1, 0)$, $(N-1, N-1)$
  - 5 center & edge-midpoints: $(N/2, N/2)$, $(0, N/2)$, $(N-1, N/2)$, $(N/2, 0)$, $(N/2, N-1)$
  - 49 octile grid intersections: $((i \cdot N)/8, (j \cdot N)/8)$
  - 6 block/tile boundary coordinates: $(15, 15)$, $(16, 16)$, $(31, 31)$, $(32, 32)$, $(N-17, N-17)$, $(N-33, N-33)$
  - Pseudo-random coordinates deterministically generated with fixed seed `1337`.
- Computes single-element CPU dot-product: $C[row, col] = \sum_{k=0}^{N-1} A[row, k] \times B[k, col]$.
- Detects non-finite GPU outputs (NaN/Inf) and checks against numerical tolerances.

---

## 5. Synchronization & Boundary Methodology for Tiled GEMM

### Synchronization Barrier Pairs
1. **Barrier 1 (`__syncthreads()` post-load):** Guarantees that every thread in the block has completed its cooperative global-to-shared memory load into `s_A[ty][tx]` and `s_B[ty][tx]` before any thread starts computing dot-products.
2. **Barrier 2 (`__syncthreads()` post-compute):** Guarantees that all threads have completed their arithmetic accumulation from shared memory before any faster thread loops around to overwrite `s_A` and `s_B` with the next tile.

### Deadlock Prevention
- All barrier calls `__syncthreads()` are executed unconditionally by every thread in the block.
- Boundary conditions for arbitrary $N$ are handled by assigning `0.0f` to out-of-bounds shared memory elements rather than skipping thread execution with an `if`-branch around the barrier.

---

## 6. Floating-Point Tolerance Methodology

Floating-point summation is non-associative in IEEE 754 arithmetic:
$$(a + b) + c \neq a + (b + c)$$

Outputs are evaluated using the absolute + relative tolerance formula:
$$\left| C_{\text{cpu}}[i] - C_{\text{gpu}}[i] \right| \le \text{atol} + \text{rtol} \times \left| C_{\text{cpu}}[i] \right|$$

- **Absolute tolerance (`atol`):** $1.0 \times 10^{-4}$
- **Relative tolerance (`rtol`):** $1.0 \times 10^{-3}$
