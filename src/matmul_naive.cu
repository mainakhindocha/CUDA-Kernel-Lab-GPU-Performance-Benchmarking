#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <string>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <set>
#include <utility>

#include <cuda_runtime.h>

/**
 * Stage 1: Naive CUDA Matrix Multiplication Kernel
 * Project: CUDA Kernel Lab — GPU Performance Benchmarking
 *
 * Matrix Dimensions:
 *   Square matrices A, B, C of size N x N (row-major).
 *
 * Algorithm:
 *   C = A * B
 *   C[i * N + j] = sum_{k=0}^{N-1} (A[i * N + k] * B[k * N + j])
 *
 * Thread Mapping:
 *   - 2D Grid of 2D Thread Blocks (e.g. 16x16 threads per block).
 *   - Each thread computes one element C[row, col].
 *   - row = blockIdx.y * blockDim.y + threadIdx.y
 *   - col = blockIdx.x * blockDim.x + threadIdx.x
 *
 * Non-Divisible Dimensions:
 *   - Handled via boundary check: `if (row < N && col < N)`
 *   - Grid size: ((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y)
 *
 * Verification Strategy:
 *   - For N <= 1024: Full CPU reference verification across all N*N elements.
 *   - For N > 1024: Sampled CPU verification recomputing >= 257 deterministically
 *     selected elements across corners, boundaries, centers, and grid points
 *     to avoid prohibitive O(N^3) CPU compute time while ensuring correctness.
 */

// Macro for checking CUDA Runtime API errors
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__       \
                      << " -> " << cudaGetErrorString(err) << std::endl;       \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Macro for checking CUDA Kernel launch errors
#define CUDA_CHECK_LAST_ERROR()                                                \
    do {                                                                       \
        cudaError_t err = cudaGetLastError();                                  \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA Kernel Launch Error at " << __FILE__ << ":"     \
                      << __LINE__ << " -> " << cudaGetErrorString(err)         \
                      << std::endl;                                            \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Deterministic matrix initialization for reproducibility
void init_matrix_deterministic(std::vector<float>& mat, int N, unsigned int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (int i = 0; i < N * N; ++i) {
        mat[i] = dis(gen);
    }
}

// Full CPU Reference implementation for numerical verification
void matmul_cpu_reference(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/**
 * Naive CUDA GEMM Kernel
 * Each thread calculates a single dot-product for element C[row, col].
 */
__global__ void matmul_naive_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int N) {
    // 2D thread indexing
    int col = blockIdx.x * blockDim.x + threadIdx.x; // x dimension corresponds to columns
    int row = blockIdx.y * blockDim.y + threadIdx.y; // y dimension corresponds to rows

    // Boundary check for matrices not divisible by block dimensions
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

/**
 * Full Verification Function (Used for N <= 1024)
 * Compares all N*N elements between CPU golden reference and GPU output.
 * Floating-Point Tolerance Methodology:
 *   |C_cpu[i] - C_gpu[i]| <= atol + rtol * |C_cpu[i]|
 * Documented tolerances: atol = 1e-4f, rtol = 1e-3f.
 */
bool verify_results(const float* C_ref, const float* C_test, int N,
                    float atol, float rtol, double& out_max_diff) {
    size_t total_elements = static_cast<size_t>(N) * N;
    double max_diff = 0.0;
    size_t mismatches = 0;
    size_t non_finite_count = 0;

    for (size_t i = 0; i < total_elements; ++i) {
        float ref = C_ref[i];
        float test = C_test[i];

        // Detect non-finite GPU outputs (NaN or Inf)
        if (!std::isfinite(test)) {
            if (non_finite_count < 5) {
                std::cerr << "Non-finite GPU value (NaN/Inf) at index " << i
                          << ": GPU=" << test << ", CPU=" << ref << "\n";
            }
            non_finite_count++;
            mismatches++;
            continue;
        }

        double diff = std::fabs(static_cast<double>(ref) - static_cast<double>(test));
        if (diff > max_diff) {
            max_diff = diff;
        }

        double allowed_tol = atol + rtol * std::fabs(static_cast<double>(ref));
        if (diff > allowed_tol) {
            if (mismatches < 5) {
                std::cerr << "Mismatch at index " << i << ": CPU=" << ref
                          << ", GPU=" << test << ", diff=" << diff
                          << " (tol=" << allowed_tol << ")\n";
            }
            mismatches++;
        }
    }

    out_max_diff = max_diff;
    if (mismatches > 0) {
        std::cerr << "Full verification failed: " << mismatches << " / "
                  << total_elements << " elements differed ("
                  << non_finite_count << " non-finite). Max diff: "
                  << max_diff << "\n";
        return false;
    }
    return true;
}

/**
 * Generate at least `min_samples` deterministically selected (row, col) coordinates
 * spread across the N x N matrix:
 *   - Extreme corners (4 points)
 *   - Center and mid-edges (5 points)
 *   - Octile grid intersections (49 points)
 *   - Block boundary coordinates (6 points)
 *   - Deterministic pseudo-random points (seeded with fixed constant 1337)
 */
std::vector<std::pair<int, int>> generate_deterministic_samples(int N, size_t min_samples = 257) {
    std::vector<std::pair<int, int>> samples;
    std::set<std::pair<int, int>> seen;

    auto add_sample = [&](int r, int c) {
        if (r >= 0 && r < N && c >= 0 && c < N) {
            if (seen.insert({r, c}).second) {
                samples.push_back({r, c});
            }
        }
    };

    // 1. Extreme Corners
    add_sample(0, 0);
    add_sample(0, N - 1);
    add_sample(N - 1, 0);
    add_sample(N - 1, N - 1);

    // 2. Center and Mid-Edges
    add_sample(N / 2, N / 2);
    add_sample(0, N / 2);
    add_sample(N - 1, N / 2);
    add_sample(N / 2, 0);
    add_sample(N / 2, N - 1);

    // 3. Octile Grid Intersections
    for (int i = 1; i < 8; ++i) {
        for (int j = 1; j < 8; ++j) {
            add_sample((i * N) / 8, (j * N) / 8);
        }
    }

    // 4. Block boundaries (for 16 and 32 block dimensions)
    add_sample(15, 15);
    add_sample(16, 16);
    add_sample(31, 31);
    add_sample(32, 32);
    add_sample(N - 17, N - 17);
    add_sample(N - 33, N - 33);

    // 5. Deterministic Pseudo-Random Generation to reach >= min_samples
    std::mt19937 gen(1337); // Fixed seed for reproducibility
    std::uniform_int_distribution<int> dis(0, N - 1);

    while (samples.size() < min_samples) {
        int r = dis(gen);
        int c = dis(gen);
        add_sample(r, c);
    }

    return samples;
}

/**
 * Sampled Verification Function (Used for N > 1024)
 * Independently recomputes CPU dot-products for at least 257 deterministically
 * selected output elements and compares them against the corresponding GPU values.
 *
 * Avoids computing the full O(N^3) CPU matrix product for large matrices.
 */
bool verify_sampled_results(const float* A, const float* B, const float* C_gpu,
                            int N, size_t min_samples, float atol, float rtol,
                            double& out_max_diff, size_t& out_sampled_count) {
    auto samples = generate_deterministic_samples(N, min_samples);
    out_sampled_count = samples.size();

    double max_diff = 0.0;
    size_t mismatches = 0;
    size_t non_finite_count = 0;

    for (const auto& pt : samples) {
        int row = pt.first;
        int col = pt.second;

        // Independently recompute scalar dot-product on CPU
        float cpu_ref = 0.0f;
        size_t a_row_offset = static_cast<size_t>(row) * N;
        for (int k = 0; k < N; ++k) {
            cpu_ref += A[a_row_offset + k] * B[static_cast<size_t>(k) * N + col];
        }

        float gpu_val = C_gpu[static_cast<size_t>(row) * N + col];

        // Check for non-finite values (NaN / Inf)
        if (!std::isfinite(gpu_val)) {
            if (non_finite_count < 5) {
                std::cerr << "Non-finite GPU value (NaN/Inf) at sample (" << row << ", " << col
                          << "): GPU=" << gpu_val << ", expected CPU=" << cpu_ref << "\n";
            }
            non_finite_count++;
            mismatches++;
            continue;
        }

        double diff = std::fabs(static_cast<double>(cpu_ref) - static_cast<double>(gpu_val));
        if (diff > max_diff) {
            max_diff = diff;
        }

        double allowed_tol = atol + rtol * std::fabs(static_cast<double>(cpu_ref));
        if (diff > allowed_tol) {
            if (mismatches < 5) {
                std::cerr << "Sample mismatch at (" << row << ", " << col << "): CPU="
                          << cpu_ref << ", GPU=" << gpu_val << ", diff=" << diff
                          << " (tol=" << allowed_tol << ")\n";
            }
            mismatches++;
        }
    }

    out_max_diff = max_diff;

    if (mismatches > 0) {
        std::cerr << "Sampled verification failed with " << mismatches << " / "
                  << samples.size() << " sample mismatches ("
                  << non_finite_count << " non-finite). Max diff: "
                  << max_diff << "\n";
        return false;
    }
    return true;
}

// Print usage help
void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [matrix_size N] [block_dim] [iterations] [--csv] [--no-verify]\n"
              << "Arguments:\n"
              << "  matrix_size N : Dimension of N x N square matrices (default: 512, must be > 0)\n"
              << "  block_dim     : Thread block size (default: 16 -> 16x16, must be > 0)\n"
              << "  iterations    : Number of benchmark runs (default: 3, must be > 0)\n"
              << "  --csv         : Output result in CSV format\n"
              << "  --no-verify   : Intentionally skip CPU verification\n";
}

int main(int argc, char* argv[]) {
    // Documented defaults
    int N = 512;
    int block_dim = 16;
    int iterations = 3;
    bool csv_mode = false;
    bool skip_verify = false;

    // Parse and validate arguments
    if (argc >= 2) {
        std::string arg1 = argv[1];
        if (arg1 == "-h" || arg1 == "--help") {
            print_usage(argv[0]);
            return 0;
        }
        try {
            N = std::stoi(arg1);
        } catch (...) {
            std::cerr << "Error: Invalid matrix size '" << arg1 << "'. Must be a positive integer.\n";
            return 1;
        }
        if (N <= 0) {
            std::cerr << "Error: Matrix size N must be strictly positive (got " << N << ").\n";
            return 1;
        }
    }

    if (argc >= 3) {
        try {
            block_dim = std::stoi(argv[2]);
        } catch (...) {
            std::cerr << "Error: Invalid block dimension '" << argv[2] << "'. Must be a positive integer.\n";
            return 1;
        }
        if (block_dim <= 0 || block_dim > 32) {
            std::cerr << "Error: Block dimension must be between 1 and 32 (got " << block_dim << ").\n";
            return 1;
        }
    }

    if (argc >= 4) {
        try {
            iterations = std::stoi(argv[3]);
        } catch (...) {
            std::cerr << "Error: Invalid iterations '" << argv[3] << "'. Must be a positive integer.\n";
            return 1;
        }
        if (iterations <= 0) {
            std::cerr << "Error: Iterations must be strictly positive (got " << iterations << ").\n";
            return 1;
        }
    }

    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--csv") {
            csv_mode = true;
        } else if (std::string(argv[i]) == "--no-verify") {
            skip_verify = true;
        }
    }

    // Query active GPU properties
    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    if (!csv_mode) {
        std::cout << "==========================================================\n";
        std::cout << " CUDA Kernel Lab — Stage 1: Naive CUDA GEMM Benchmark\n";
        std::cout << "==========================================================\n";
        std::cout << "GPU Device          : " << prop.name << " (SM " << prop.major << "." << prop.minor << ")\n";
        std::cout << "Matrix Size (N x N) : " << N << " x " << N << "\n";
        std::cout << "Block Dimensions    : " << block_dim << " x " << block_dim
                  << " (" << (block_dim * block_dim) << " threads/block)\n";
        std::cout << "Grid Dimensions     : "
                  << ((N + block_dim - 1) / block_dim) << " x "
                  << ((N + block_dim - 1) / block_dim) << " blocks\n";
        std::cout << "Iterations          : " << iterations << "\n";
        std::cout << "----------------------------------------------------------\n";
    }

    // Allocate host memory
    size_t total_elements = static_cast<size_t>(N) * N;
    size_t bytes = total_elements * sizeof(float);

    std::vector<float> h_A(total_elements);
    std::vector<float> h_B(total_elements);
    std::vector<float> h_C_gpu(total_elements, 0.0f);

    // Only allocate h_C_cpu for N <= 1024 when full CPU reference is calculated
    std::vector<float> h_C_cpu;
    if (!skip_verify && N <= 1024) {
        h_C_cpu.assign(total_elements, 0.0f);
    }

    init_matrix_deterministic(h_A, N, 42);
    init_matrix_deterministic(h_B, N, 84);

    // Allocate device memory
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // CUDA Events for separate phase timing
    cudaEvent_t h2d_start, h2d_stop;
    cudaEvent_t kernel_start, kernel_stop;
    cudaEvent_t d2h_start, d2h_stop;

    CUDA_CHECK(cudaEventCreate(&h2d_start));
    CUDA_CHECK(cudaEventCreate(&h2d_stop));
    CUDA_CHECK(cudaEventCreate(&kernel_start));
    CUDA_CHECK(cudaEventCreate(&kernel_stop));
    CUDA_CHECK(cudaEventCreate(&d2h_start));
    CUDA_CHECK(cudaEventCreate(&d2h_stop));

    // Measure Host-to-Device Transfer (H2D)
    CUDA_CHECK(cudaEventRecord(h2d_start, 0));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(h2d_stop, 0));
    CUDA_CHECK(cudaEventSynchronize(h2d_stop));

    float h2d_time_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_time_ms, h2d_start, h2d_stop));

    // Setup 2D thread block and grid
    dim3 block(block_dim, block_dim);
    dim3 grid((N + block.x - 1) / block.x,
              (N + block.y - 1) / block.y);

    // Warmup run
    matmul_naive_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Kernel execution benchmark loop
    std::vector<float> kernel_times_ms;
    kernel_times_ms.reserve(iterations);

    for (int iter = 0; iter < iterations; ++iter) {
        CUDA_CHECK(cudaEventRecord(kernel_start, 0));
        matmul_naive_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
        CUDA_CHECK_LAST_ERROR();
        CUDA_CHECK(cudaEventRecord(kernel_stop, 0));
        CUDA_CHECK(cudaEventSynchronize(kernel_stop));

        float k_time = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&k_time, kernel_start, kernel_stop));
        kernel_times_ms.push_back(k_time);

        if (!csv_mode) {
            double gflops = (2.0 * N * N * N) / (k_time * 1e6);
            std::cout << "  Iter " << (iter + 1) << " Kernel Time: "
                      << std::fixed << std::setprecision(3) << k_time << " ms ("
                      << std::setprecision(2) << gflops << " GFLOPS)\n";
        }
    }

    // Measure Device-to-Host Transfer (D2H)
    CUDA_CHECK(cudaEventRecord(d2h_start, 0));
    CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(d2h_stop, 0));
    CUDA_CHECK(cudaEventSynchronize(d2h_stop));

    float d2h_time_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&d2h_time_ms, d2h_start, d2h_stop));

    // Compute GPU kernel timing statistics
    float min_kernel_ms = *std::min_element(kernel_times_ms.begin(), kernel_times_ms.end());
    float max_kernel_ms = *std::max_element(kernel_times_ms.begin(), kernel_times_ms.end());
    float avg_kernel_ms = std::accumulate(kernel_times_ms.begin(), kernel_times_ms.end(), 0.0f) / iterations;
    double best_gflops = (2.0 * N * N * N) / (min_kernel_ms * 1e6);
    float total_gpu_pipeline_ms = h2d_time_ms + min_kernel_ms + d2h_time_ms;

    // Numerical Verification (performed AFTER timing measurements to ensure complete isolation)
    bool verified = false;
    std::string verify_mode_desc = "None";
    double max_diff = 0.0;
    size_t sample_count = 0;
    double verify_time_ms = 0.0;

    if (skip_verify) {
        verify_mode_desc = "Skipped (--no-verify)";
    } else if (N <= 1024) {
        verify_mode_desc = "Full CPU Reference (" + std::to_string(total_elements) + " elements)";
        if (!csv_mode) {
            std::cout << "\nRunning full CPU reference verification (" << total_elements << " elements)...\n";
        }
        auto v_start = std::chrono::high_resolution_clock::now();
        matmul_cpu_reference(h_A.data(), h_B.data(), h_C_cpu.data(), N);
        verified = verify_results(h_C_cpu.data(), h_C_gpu.data(), N, 1e-4f, 1e-3f, max_diff);
        auto v_end = std::chrono::high_resolution_clock::now();
        verify_time_ms = std::chrono::duration<double, std::milli>(v_end - v_start).count();
    } else {
        size_t min_samples = 257;
        if (!csv_mode) {
            std::cout << "\nRunning sampled CPU verification (at least " << min_samples
                      << " deterministically sampled elements)...\n";
        }
        auto v_start = std::chrono::high_resolution_clock::now();
        verified = verify_sampled_results(h_A.data(), h_B.data(), h_C_gpu.data(), N,
                                          min_samples, 1e-4f, 1e-3f, max_diff, sample_count);
        auto v_end = std::chrono::high_resolution_clock::now();
        verify_time_ms = std::chrono::duration<double, std::milli>(v_end - v_start).count();
        verify_mode_desc = "Sampled CPU Reference (" + std::to_string(sample_count) + " elements)";
    }

    // Determine verification status string
    std::string status_str = "PASSED";
    if (skip_verify) {
        status_str = "SKIPPED";
    } else if (!verified) {
        status_str = "FAILED";
    }

    if (csv_mode) {
        // Format: N,block_dim,h2d_ms,min_kernel_ms,avg_kernel_ms,d2h_ms,total_pipeline_ms,gflops,verified,max_diff
        std::cout << N << "," << block_dim << ","
                  << std::fixed << std::setprecision(4)
                  << h2d_time_ms << ","
                  << min_kernel_ms << ","
                  << avg_kernel_ms << ","
                  << d2h_time_ms << ","
                  << total_gpu_pipeline_ms << ","
                  << best_gflops << ","
                  << status_str << ","
                  << max_diff << "\n";
    } else {
        std::cout << "\n==========================================================\n";
        std::cout << " Stage 1 Benchmark Results Summary\n";
        std::cout << "==========================================================\n";
        std::cout << "  Host-to-Device (H2D)  : " << h2d_time_ms << " ms\n";
        std::cout << "  Min Kernel Time       : " << min_kernel_ms << " ms\n";
        std::cout << "  Avg Kernel Time       : " << avg_kernel_ms << " ms\n";
        std::cout << "  Max Kernel Time       : " << max_kernel_ms << " ms\n";
        std::cout << "  Device-to-Host (D2H)  : " << d2h_time_ms << " ms\n";
        std::cout << "  Total GPU Pipeline    : " << total_gpu_pipeline_ms << " ms (H2D + Kernel + D2H)\n";
        std::cout << "  Kernel Throughput     : " << best_gflops << " GFLOPS\n";
        std::cout << "  Verification Mode     : " << verify_mode_desc << "\n";
        std::cout << "  Verification Status   : " << status_str
                  << " (Max Abs Diff: " << max_diff << ")\n";
        if (!skip_verify) {
            std::cout << "  Verification Overhead : " << verify_time_ms
                      << " ms (excluded from GPU kernel timing)\n";
        }
        if (N <= 1024 && !skip_verify && verify_time_ms > 0.0f) {
            std::cout << "  Kernel Speedup vs CPU : " << (verify_time_ms / min_kernel_ms) << "x\n";
        }
        std::cout << "==========================================================\n";
    }

    // Cleanup CUDA resources
    CUDA_CHECK(cudaEventDestroy(h2d_start));
    CUDA_CHECK(cudaEventDestroy(h2d_stop));
    CUDA_CHECK(cudaEventDestroy(kernel_start));
    CUDA_CHECK(cudaEventDestroy(kernel_stop));
    CUDA_CHECK(cudaEventDestroy(d2h_start));
    CUDA_CHECK(cudaEventDestroy(d2h_stop));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return (skip_verify || verified) ? 0 : 1;
}
