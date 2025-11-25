#include <stdio.h>
#include <iostream>
#include <cassert>
#include <cuda_runtime.h>
#include <cstdlib>

using std::cout;

__global__ void vec_add(int *a, int *b, int *c, int N){
    int threadid = (blockDim.x * blockIdx.x) + threadIdx.x;

    if (threadid < N){
        c[threadid] = a[threadid] + b[threadid];
    }
}


int main(){
    const int N = 1<<16;

    int *a, *b, *c;
    size_t bytes = sizeof(int) * N;

    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    int id = cudaGetDevice(&id);

    for(int i = 0; i < N; i++){
        a[i] = rand() / 100;
        b[i] = rand() / 100;
    }

    cudaMemPrefetchAsync(a, bytes, id);
    cudaMemPrefetchAsync(b, bytes, id);
    cudaMemPrefetchAsync(c, bytes, id);

    int numthreads = 1<<10;
    int numblocks = (N + numthreads - 1) / numthreads;

    vec_add<<<numblocks, numthreads>>>(a, b, c, N);

    cudaDeviceSynchronize();

    // cudaMemPrefetchAsync(a, bytes, cudaCpuDeviceId);
    // cudaMemPrefetchAsync(b, bytes, cudaCpuDeviceId);
    cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId);

    for(int i = 0; i<N;i++){
        assert(c[i] == a[i] + b[i]);
    }

    cudaFree(a); cudaFree(b); cudaFree(c);
    cout << "COMPLETED SUCCESSFULLY";

    return 0;
}