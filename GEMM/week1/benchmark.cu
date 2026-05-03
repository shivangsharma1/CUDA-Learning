#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>

// ═══════════════════════════════════════════════════════════
//  CONFIG
// ═══════════════════════════════════════════════════════════
#define DIM_M 4096
#define DIM_N 4096
#define DIM_K 4096
#define WARMUP 3
#define RUNS   10

// ═══════════════════════════════════════════════════════════
//  Include kernels from individual files
// ═══════════════════════════════════════════════════════════
#define BLOCKSIZE 32

#include "naive_gemm.cu"
#include "coalesce_gemm.cu"
#include "shared_block_tiling.cu"
#include "1d_block_tiling.cu"

// ═══════════════════════════════════════════════════════════
//  Benchmark harness
// ═══════════════════════════════════════════════════════════

void fill_random(float* h, int size) {
    for (int i = 0; i < size; i++)
        h[i] = (float)(rand() % 100) / 100.0f;
}

float compute_gflops(float ms, int m, int n, int k) {
    double flops = 2.0 * m * n * k;
    return (float)((flops / (ms / 1000.0)) / 1e9);
}

// Verify against cuBLAS result
float max_error(float* ref, float* test, int size) {
    float maxerr = 0.0f;
    for (int i = 0; i < size; i++) {
        float err = fabsf(ref[i] - test[i]);
        if (err > maxerr) maxerr = err;
    }
    return maxerr;
}

int main() {
    printf("\n");
    printf("══════════════════════════════════════════════════════════\n");
    printf("  SGEMM Benchmark: M=%d, N=%d, K=%d\n", DIM_M, DIM_N, DIM_K);
    printf("══════════════════════════════════════════════════════════\n\n");

    // ── Allocate host ──
    int sizeA = DIM_M * DIM_K, sizeB = DIM_K * DIM_N, sizeC = DIM_M * DIM_N;
    float *hA = (float*)malloc(sizeA * sizeof(float));
    float *hB = (float*)malloc(sizeB * sizeof(float));
    float *hC = (float*)malloc(sizeC * sizeof(float));
    float *hC_ref = (float*)malloc(sizeC * sizeof(float));  // cuBLAS reference
    float *hC_test = (float*)malloc(sizeC * sizeof(float)); // kernel output

    srand(42);
    fill_random(hA, sizeA);
    fill_random(hB, sizeB);
    fill_random(hC, sizeC);

    // ── Allocate device ──
    float *dA, *dB, *dC, *dC_ref;
    cudaMalloc(&dA, sizeA * sizeof(float));
    cudaMalloc(&dB, sizeB * sizeof(float));
    cudaMalloc(&dC, sizeC * sizeof(float));
    cudaMalloc(&dC_ref, sizeC * sizeof(float));

    cudaMemcpy(dA, hA, sizeA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeB * sizeof(float), cudaMemcpyHostToDevice);

    float alpha = 1.0f, beta = 0.0f;

    // ── cuBLAS reference ──
    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaMemcpy(dC_ref, hC, sizeC * sizeof(float), cudaMemcpyHostToDevice);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                DIM_N, DIM_M, DIM_K, &alpha, dB, DIM_N, dA, DIM_K, &beta, dC_ref, DIM_N);
    cudaMemcpy(hC_ref, dC_ref, sizeC * sizeof(float), cudaMemcpyDeviceToHost);

    // Time cuBLAS
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;

    for (int i = 0; i < WARMUP; i++)
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    DIM_N, DIM_M, DIM_K, &alpha, dB, DIM_N, dA, DIM_K, &beta, dC_ref, DIM_N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < RUNS; i++)
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    DIM_N, DIM_M, DIM_K, &alpha, dB, DIM_N, dA, DIM_K, &beta, dC_ref, DIM_N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    ms /= RUNS;

    float cublas_gflops = compute_gflops(ms, DIM_M, DIM_N, DIM_K);
    printf("%-28s %10.3f ms  %10.1f GFLOPS  (baseline)\n", "cuBLAS", ms, cublas_gflops);
    printf("────────────────────────────────────────────────────────\n");

    // ═══════════════════════════════════════════════════════
    //  Macro to benchmark each kernel
    // ═══════════════════════════════════════════════════════
    #define BENCH(name, ...)                                                       \
    {                                                                              \
        /* Warmup */                                                               \
        for (int _w = 0; _w < WARMUP; _w++) {                                     \
            cudaMemcpy(dC, hC, sizeC * sizeof(float), cudaMemcpyHostToDevice);     \
            __VA_ARGS__;                                                           \
        }                                                                          \
        cudaDeviceSynchronize();                                                   \
                                                                                   \
        /* Timed runs */                                                           \
        cudaEventRecord(start);                                                    \
        for (int _r = 0; _r < RUNS; _r++) {                                       \
            cudaMemcpy(dC, hC, sizeC * sizeof(float), cudaMemcpyHostToDevice);     \
            __VA_ARGS__;                                                           \
        }                                                                          \
        cudaEventRecord(stop);                                                     \
        cudaEventSynchronize(stop);                                                \
        cudaEventElapsedTime(&ms, start, stop);                                    \
        ms /= RUNS;                                                                \
                                                                                   \
        /* Correctness check */                                                    \
        cudaMemcpy(dC, hC, sizeC * sizeof(float), cudaMemcpyHostToDevice);         \
        __VA_ARGS__;                                                               \
        cudaDeviceSynchronize();                                                   \
        cudaMemcpy(hC_test, dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost);    \
        float err = max_error(hC_ref, hC_test, sizeC);                            \
                                                                                   \
        float gf = compute_gflops(ms, DIM_M, DIM_N, DIM_K);                                   \
        float pct = (gf / cublas_gflops) * 100.0f;                                \
        printf("%-28s %10.3f ms  %10.1f GFLOPS  %5.1f%%  err=%.4e\n",            \
               name, ms, gf, pct, err);                                            \
    }

    // ── Kernel 1: Naive ──
    {
        dim3 block(BLOCKSIZE, BLOCKSIZE);
        dim3 grid((DIM_N + BLOCKSIZE - 1) / BLOCKSIZE, (DIM_M + BLOCKSIZE - 1) / BLOCKSIZE);
        BENCH("1. Naive",
              naive_sgemm<<<grid, block>>>(DIM_M, DIM_K, DIM_N, dA, dB, dC, alpha, beta));
    }

    // ── Kernel 2: Coalesced ──
    {
        dim3 block(BLOCKSIZE * BLOCKSIZE);
        dim3 grid((DIM_N + BLOCKSIZE - 1) / BLOCKSIZE, (DIM_M + BLOCKSIZE - 1) / BLOCKSIZE);
        BENCH("2. Coalesced",
              coalesce_sgemm<<<grid, block>>>(DIM_M, DIM_N, DIM_K, dA, dB, dC, alpha, beta));
    }

    // ── Kernel 3: Shared Memory Tiling ──
    {
        dim3 block(TILE_DIM, TILE_DIM);
        dim3 grid((DIM_N + TILE_DIM - 1) / TILE_DIM, (DIM_M + TILE_DIM - 1) / TILE_DIM);
        BENCH("3. Shared Memory Tiling",
              share_tiling<<<grid, block>>>(DIM_M, DIM_N, DIM_K, dA, dB, dC, alpha, beta));
    }

    // ── Kernel 4: 1D Block Tiling ──
    {
        dim3 block(BN, BM / TM);  // (64, 8) = 512 threads
        dim3 grid((DIM_N + BN - 1) / BN, (DIM_M + BM - 1) / BM);
        BENCH("4. 1D Block Tiling",
              blocktiling_1d<<<grid, block>>>(DIM_M, DIM_N, DIM_K, dA, dB, dC, alpha, beta));
    }

    printf("════════════════════════════════════════════════════════\n\n");

    // ── Cleanup ──
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dC_ref);
    free(hA); free(hB); free(hC); free(hC_ref); free(hC_test);

    return 0;
}
