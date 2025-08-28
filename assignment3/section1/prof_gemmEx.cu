#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>

int main() {
    std::ofstream file("gemm_perf.csv");
    file << "batch_size,N,K,library,gflops\n";

    int nk[][2] = {
        {512, 512},
        {4096, 4096},
        {14336, 4096},
        {4096, 1024},
        {1024, 4096}
    };

    cublasHandle_t handle;
    cublasCreate(&handle);

    for (int M = 128; M <= 2048; M += 128) {
        for (int i = 0; i < 5; ++i) {
            int N = nk[i][0];
            int K = nk[i][1];

            float *d_A, *d_B, *d_C;
            cudaMalloc(&d_A, M * K * sizeof(float));
            cudaMalloc(&d_B, K * N * sizeof(float));
            cudaMalloc(&d_C, M * N * sizeof(float));

            float alpha = 1.0f;
            float beta = 0.0f;

            // Warm-up
            cublasGemmEx(handle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        N, M, K,
                        &alpha,
                        d_B, CUDA_R_32F, N,
                        d_A, CUDA_R_32F, K,
                        &beta,
                        d_C, CUDA_R_32F, N,
                        CUDA_R_32F,
                        CUBLAS_GEMM_DEFAULT);
            cudaDeviceSynchronize();

            auto start = std::chrono::high_resolution_clock::now();

            cublasGemmEx(handle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        N, M, K,
                        &alpha,
                        d_B, CUDA_R_32F, N,
                        d_A, CUDA_R_32F, K,
                        &beta,
                        d_C, CUDA_R_32F, N,
                        CUDA_R_32F,
                        CUBLAS_GEMM_DEFAULT);
            cudaDeviceSynchronize();

            auto end = std::chrono::high_resolution_clock::now();

            double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
            double gflops = 2.0 * M * N * K / (duration_ms * 1e6);

            file << M << "," << N << "," << K << ",cublas," << gflops << "\n";
            std::cout << "M=" << M << ", N=" << N << ", K=" << K << ", GFLOPS=" << gflops << ", durationms=" << duration_ms << std::endl;

            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
        }
    }

    cublasDestroy(handle);
    file.close();
    return 0;
}
