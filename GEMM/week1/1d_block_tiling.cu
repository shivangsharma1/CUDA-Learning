#pragma once

#include <stdio.h>
#include <cuda_runtime.h>

#define BM 64
#define BN 64
#define BK 8
#define TM 8

__global__ void blocktiling_1d(int M, int N, int K, float* A, float* B, float* C, float alpha, float beta){
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int threadId = threadIdx.y * blockDim.x + threadIdx.x;
    int As_row = threadId / BK;
    int As_col = threadId % BK;
    int Bs_row = threadId / BN;
    int Bs_col = threadId % BN;

    int NUM_TILE = (K + BK - 1) / BK;
    float sum[TM] = {0.0f};

    for(int tile = 0; tile < NUM_TILE; tile++){
        // load A with bounds check
        int Arow_global = blockIdx.y * BM + As_row;
        int Acol_global = tile * BK + As_col;
        if (Arow_global < M && Acol_global < K)
            As[As_row][As_col] = A[Arow_global * K + Acol_global];
        else
            As[As_row][As_col] = 0.0f;

        // load B with bounds check
        int Brow_global = tile * BK + Bs_row;
        int Bcol_global = blockIdx.x * BN + Bs_col;
        if (Brow_global < K && Bcol_global < N)
            Bs[Bs_row][Bs_col] = B[Brow_global * N + Bcol_global];
        else
            Bs[Bs_row][Bs_col] = 0.0f;

        __syncthreads();
        for (int k = 0; k < BK; k++){
            float bval = Bs[k][threadIdx.x];
            for (int m = 0; m < TM; m++){
                sum[m] += As[threadIdx.y * TM + m][k] * bval;
            }
        }
        __syncthreads();
    }

    for(int m = 0; m < TM; m++){
        int C_global_row = blockIdx.y * BM + threadIdx.y * TM + m;
        int C_global_col = blockIdx.x * BN + threadIdx.x;
        if (C_global_row < M && C_global_col < N)
            C[C_global_row * N + C_global_col] = alpha * sum[m] + beta * C[C_global_row * N + C_global_col];
    }
}