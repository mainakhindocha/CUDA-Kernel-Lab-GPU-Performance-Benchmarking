# Test Plan — CUDA Kernel Lab: Stage 1

## 1. Objective
Validate the functional correctness, numerical fidelity, and boundary safety of both the CPU Baseline (`src/cpu_baseline.cpp`) and the Naive CUDA GEMM kernel (`src/matmul_naive.cu`), including full and sampled verification strategies.

---

## 2. Test Matrix Configurations

| Test ID | Category | Matrix Size ($N \times N$) | Block Dim | Verification Mode | Expected Result |
|:---:|:---|:---:|:---:|:---|:---:|
| **T01** | Minimal Dimension | $1 \times 1$ | 16 | Full ($1$ element) | PASS (Max diff $< 10^{-5}$) |
| **T02** | Minimal Power-of-2 | $16 \times 16$ | 16 | Full ($256$ elements) | PASS (Max diff $< 10^{-5}$) |
| **T03** | Small Power-of-2 | $32 \times 32$ | 16 | Full ($1024$ elements) | PASS (Max diff $< 10^{-5}$) |
| **T04** | Boundary Edge | $17 \times 17$ | 16 | Full ($289$ elements) | PASS (Boundary check guards $N$) |
| **T05** | Boundary Odd | $37 \times 37$ | 16 | Full ($1369$ elements) | PASS (Guards against out-of-bounds) |
| **T06** | Boundary Decimal | $100 \times 100$ | 16 | Full ($10^4$ elements) | PASS |
| **T07** | Boundary Prime | $311 \times 311$ | 16 | Full ($96,721$ elements) | PASS |
| **T08** | Standard Power-of-2 | $128 \times 128$ | 16 | Full ($16,384$ elements) | PASS (Max diff $< 10^{-4}$) |
| **T09** | Medium Power-of-2 | $256 \times 256$ | 16 | Full ($65,536$ elements) | PASS (Max diff $< 10^{-4}$) |
| **T10** | Large Full Verified | $512 \times 512$ | 16 | Full ($262,144$ elements) | PASS (Max diff $< 10^{-4}$) |
| **T11** | Custom Block Size | $128 \times 128$ | 32 | Full ($16,384$ elements) | PASS |
| **T12** | Custom Block Size | $128 \times 128$ | 8 | Full ($16,384$ elements) | PASS |
| **T13** | Large Matrix Sampled | $2048 \times 2048$ | 16 | **Sampled ($\ge 257$ elements)** | **PASS** (Recomputes $\ge 257$ deterministic dot products) |
| **T14** | Explicit Skip Flag | $128 \times 128$ | 16 | Skipped (`--no-verify`) | Status reports `SKIPPED`, Exit code 0 |

---

## 3. Negative / Invalid Input Tests

| Test ID | Target Executable | Input Arguments | Expected Behavior | Expected Exit Code |
|:---:|:---|:---|:---|:---:|
| **E01** | `cpu_baseline` | `-5` | Rejects negative size with error message | 1 |
| **E02** | `cpu_baseline` | `0` | Rejects zero size with error message | 1 |
| **E03** | `cpu_baseline` | `abc` | Rejects non-numeric argument | 1 |
| **E04** | `matmul_naive` | `-10` | Rejects negative matrix size | 1 |
| **E05** | `matmul_naive` | `128 0` | Rejects invalid block dimension ($\le 0$) | 1 |
| **E06** | `matmul_naive` | `128 64` | Rejects block dimension $> 32$ ($> 1024$ threads) | 1 |

---

## 4. Verification Strategies

### A. Full Verification ($N \le 1024$)
- Complete exhaustive comparison across all $N \times N$ matrix elements.
- CPU reference computed in full via $O(N^3)$ loops.

### B. Sampled Verification ($N > 1024$)
- To avoid prohibitive $O(N^3)$ CPU compute time on large matrices (which takes minutes for $N \ge 2048$ on a single thread), `src/matmul_naive.cu` uses `verify_sampled_results()`.
- Independently evaluates **at least 257 deterministically selected output elements**:
  - 4 extreme corners: $(0,0)$, $(0, N-1)$, $(N-1, 0)$, $(N-1, N-1)$
  - 5 center & edge-midpoints: $(N/2, N/2)$, $(0, N/2)$, $(N-1, N/2)$, $(N/2, 0)$, $(N/2, N-1)$
  - 49 octile grid intersections: $((i \cdot N)/8, (j \cdot N)/8)$
  - 6 block boundary coordinates (boundary wrapping checks)
  - Pseudo-random coordinates deterministically generated with fixed seed `1337`.
- Computes single-element CPU dot-product: $C[row, col] = \sum_{k=0}^{N-1} A[row, k] \times B[k, col]$.
- Detects non-finite GPU outputs (NaN/Inf) and checks against tolerance.
- Enables unambiguous `PASSED` verification for large matrix benchmarks.

---

## 5. Floating-Point Tolerance Methodology

Floating-point summation is non-associative in IEEE 754 arithmetic:
$$(a + b) + c \neq a + (b + c)$$

Outputs are evaluated using the absolute + relative tolerance formula:

$$\left| C_{\text{cpu}}[i] - C_{\text{gpu}}[i] \right| \le \text{atol} + \text{rtol} \times \left| C_{\text{cpu}}[i] \right|$$

- **Absolute tolerance (`atol`):** $1.0 \times 10^{-4}$
- **Relative tolerance (`rtol`):** $1.0 \times 10^{-3}$
