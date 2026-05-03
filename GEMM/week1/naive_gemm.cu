#pragma once

#include <stdio.h>
#include <cuda_runtime.h>

#ifndef BLOCKSIZE
#define BLOCKSIZE 32
#endif

__global__ void naive_sgemm(int M, int K, int N, float* A, float* B, float* C, float alpha, float beta){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N){
        float sum = 0.0f;

        for(int i = 0; i<K;i++){
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}