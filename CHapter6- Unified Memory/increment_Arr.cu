#include <stdio.h>
#include <cuda_runtime.h>


const size_t ds = 32ULL * 1024ULL * 1024ULL;

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


void alloc_bytes(int* &ptr, size_t num_bytes){
    cudaMallocManaged(&ptr, num_bytes);
}

__global__ void inc(int* array, size_t n){
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    while (idx < n){
        array[idx]++;
        idx += blockDim.x * gridDim.x;
    }
}



int main(){
    int *array;
    alloc_bytes(array, ds*sizeof(int));
    cudaCheckErrors("allocation bytes error");

    memset(array, ds*sizeof(int), 0);
    inc<<<256, 256>>>(array, ds);
    cudaCheckErrors("kernel error");

    cudaMemPrefetchAsync(array, ds*sizeof(int), cudaCpuDeviceId);
    cudaDeviceSynchronize();
    cudaCheckErrors("kernel error");

    for(int i = 0; i < ds; i++){
        if (array[i] != 1) {
            printf("mismatch at %d, was: %d, expected: %d\n", i, array[i], 1); return -1;
        }
    }
}