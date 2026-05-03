#pragma once

#include <stdio.h>
#include <cuda_runtime.h>

#ifndef BLOCKSIZE
#define BLOCKSIZE 32
#endif

__global__ void coalesce_sgemm(int M, int N, int K, float* A, float* B, float* C, float alpha, float beta){
    int row = blockIdx.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    int col = blockIdx.x * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    if (row < M && col < N){
        float sum = 0.0f;

        for(int i = 0; i<K;i++){
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}