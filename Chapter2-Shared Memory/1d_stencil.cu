#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>

#define N 4096
#define RADIUS 3
#define BLOCK_SIZE 16

void fill_ints(int* in, int n){
    for(int i = 0;i<n;i++){
        in[i] = 1;
    }
}

__global__ void stencil(int* in, int* out) {
    __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int lindex = threadIdx.x + RADIUS;

    // transferring from global to shared
    temp[lindex] = in[gindex]; //no for loop as copy is executed through threads
    if (threadIdx.x < RADIUS){
        temp[lindex - RADIUS] = in[gindex - RADIUS];
        temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
    }

    // synchronize the threads for data availability
    __syncthreads();

    int result = 0;
    for (int i = -RADIUS; i <= RADIUS; i++){
        result += temp[lindex + i];
    }
    out[gindex] = result;
}

int main(){
    int *in, *out;
    int *d_in, *d_out;
    int size = (N + 2* RADIUS) * sizeof(int);

    in = (int*)malloc(size); fill_ints(in, N + 2*RADIUS);
    out = (int*)malloc(size); fill_ints(out, N + 2*RADIUS);

    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);

    cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, out, size, cudaMemcpyHostToDevice);

    stencil<<<N/BLOCK_SIZE, BLOCK_SIZE>>>(d_in + RADIUS, d_out + RADIUS);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));

    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

    // Error Checking
    for (int i = 0; i < N + 2*RADIUS; i++) {
        if (i<RADIUS || i>=N+RADIUS){
        if (out[i] != 1)
            printf("Mismatch at index %d, was: %d, should be: %d\n", i, out[i], 1);
        } else {
        if (out[i] != 1 + 2*RADIUS)
            printf("Mismatch at index %d, was: %d, should be: %d\n", i, out[i], 1 + 2*RADIUS);
        }
    }

    // Cleanup
    free(in); free(out);
    cudaFree(d_in); cudaFree(d_out);
    printf("Success!\n");
    return 0;
}