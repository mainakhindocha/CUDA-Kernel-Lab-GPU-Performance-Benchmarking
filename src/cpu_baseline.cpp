#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <string>
#include <cstdlib>
#include <algorithm>
#include <numeric>

/**
 * Stage 1: CPU Baseline Matrix Multiplication
 * Project: CUDA Kernel Lab — GPU Performance Benchmarking
 *
 * Algorithm:
 *   Square Matrix Multiplication: C = A * B
 *   A, B, C are N x N matrices in contiguous row-major format.
 *   C[i * N + j] = sum_{k=0}^{N-1} (A[i * N + k] * B[k * N + j])
 *
 * Operational Complexity:
 *   FLOPs = 2 * N^3 (1 multiplication + 1 addition per inner step k)
 */

// Deterministic matrix initialization for reproducible benchmarking
void init_matrix_deterministic(std::vector<float>& mat, int N, unsigned int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (int i = 0; i < N * N; ++i) {
        mat[i] = dis(gen);
    }
}

// Naive square matrix multiplication (i-j-k loop order)
void matmul_cpu(const float* A, const float* B, float* C, int N) {
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

// Print usage help
void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [matrix_size N] [iterations] [--csv]\n"
              << "Arguments:\n"
              << "  matrix_size N : Dimension of N x N square matrices (default: 512, must be > 0)\n"
              << "  iterations    : Number of benchmark runs (default: 3, must be > 0)\n"
              << "  --csv         : Output result in CSV format\n";
}

int main(int argc, char* argv[]) {
    // Documented defaults
    int N = 512;
    int iterations = 3;
    bool csv_mode = false;

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
            iterations = std::stoi(argv[2]);
        } catch (...) {
            std::cerr << "Error: Invalid iterations '" << argv[2] << "'. Must be a positive integer.\n";
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
        }
    }

    if (!csv_mode) {
        std::cout << "==========================================================\n";
        std::cout << " CUDA Kernel Lab — Stage 1: CPU Baseline Benchmark\n";
        std::cout << "==========================================================\n";
        std::cout << "Matrix Size (N x N) : " << N << " x " << N << "\n";
        std::cout << "Total Operations    : " << (2.0 * N * N * N / 1e6) << " MFLOP\n";
        std::cout << "Iterations          : " << iterations << "\n";
        std::cout << "----------------------------------------------------------\n";
    }

    // Allocate memory for matrices A, B, and C
    size_t total_elements = static_cast<size_t>(N) * N;
    std::vector<float> A(total_elements);
    std::vector<float> B(total_elements);
    std::vector<float> C(total_elements, 0.0f);

    // Initialize with fixed deterministic seeds for reproducibility
    init_matrix_deterministic(A, N, 42);
    init_matrix_deterministic(B, N, 84);

    // Warmup run for small dimensions to prime CPU caches
    if (N <= 256) {
        matmul_cpu(A.data(), B.data(), C.data(), N);
    }

    // Benchmark loop
    std::vector<double> run_times_ms;
    run_times_ms.reserve(iterations);

    for (int iter = 0; iter < iterations; ++iter) {
        auto start = std::chrono::high_resolution_clock::now();
        matmul_cpu(A.data(), B.data(), C.data(), N);
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        run_times_ms.push_back(elapsed_ms);

        if (!csv_mode) {
            double gflops = (2.0 * N * N * N) / (elapsed_ms * 1e6);
            std::cout << "  Iter " << (iter + 1) << ": " << std::fixed << std::setprecision(3)
                      << elapsed_ms << " ms (" << std::setprecision(2) << gflops << " GFLOPS)\n";
        }
    }

    // Compute statistics
    double min_ms = *std::min_element(run_times_ms.begin(), run_times_ms.end());
    double max_ms = *std::max_element(run_times_ms.begin(), run_times_ms.end());
    double avg_ms = std::accumulate(run_times_ms.begin(), run_times_ms.end(), 0.0) / iterations;
    double best_gflops = (2.0 * N * N * N) / (min_ms * 1e6);

    if (csv_mode) {
        // Output CSV format: N,min_ms,avg_ms,max_ms,best_gflops
        std::cout << N << "," << std::fixed << std::setprecision(4)
                  << min_ms << "," << avg_ms << "," << max_ms << ","
                  << best_gflops << "\n";
    } else {
        std::cout << "----------------------------------------------------------\n";
        std::cout << "Summary:\n";
        std::cout << "  Min Time   : " << min_ms << " ms\n";
        std::cout << "  Avg Time   : " << avg_ms << " ms\n";
        std::cout << "  Max Time   : " << max_ms << " ms\n";
        std::cout << "  Throughput : " << best_gflops << " GFLOPS\n";
        std::cout << "==========================================================\n";
    }

    return 0;
}
