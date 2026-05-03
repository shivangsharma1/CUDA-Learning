#pragma once

#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_DIM 32

__global__ void share_tiling(int M, int N, int K, float* A, float* B, float* C, float alpha, float beta){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    int NUM_TILE = (K + TILE_DIM - 1)/TILE_DIM;
    float total = 0.0f;
    
    for(unsigned int tile = 0; tile < NUM_TILE; ++tile){
        if(row < M && (TILE_DIM * tile + threadIdx.x) < K){
            As[threadIdx.y][threadIdx.x] = A[row * K + TILE_DIM * tile + threadIdx.x];
        }
        else{
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if((TILE_DIM * tile + threadIdx.y)<K && (col < N)){
            Bs[threadIdx.y][threadIdx.x] = B[(TILE_DIM * tile + threadIdx.y) * N + col];
        }
        else{
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();
        for(unsigned int i=0; i<TILE_DIM;++i){
            total += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        __syncthreads();
    }
    
    if (row < M && col < N){
        C[row * N + col] = alpha * total + beta * C[row * N + col];
    }
}