#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cmath>

__global__ void vectorAdd(const float* A, const float* B, float* C, int num_ele){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i<num_ele){
        C[i] = A[i] + B[i];
    }
}


int main(void){
    
    cudaError_t err = cudaSuccess;
    int numele= 500000;

    size_t size = numele * sizeof(float);
    printf("Vector additon of %d elements\n", numele);
    
    float* h_A = (float*) malloc(size);
    float* h_B = (float*) malloc(size);

    float* h_C = (float*) malloc(size);
    
    for (int i = 0; i < numele; i++){
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }   

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int threadBlocks = 256;
    int blockpergrid = (numele + threadBlocks - 1) / threadBlocks;

    vectorAdd<<<blockpergrid, threadBlocks>>>(d_A, d_B, d_C, numele);
    err = cudaGetLastError();

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Catch runtime errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel runtime error: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // verification

    for(int i = 0;i<numele;i++){
        if (fabsf(h_A[i]+h_B[i]-h_C[i]) > 1e-5){
            fprintf(stderr, "Result verfication failed at element %d", i);
            exit(EXIT_FAILURE);
        }
    }
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;

}