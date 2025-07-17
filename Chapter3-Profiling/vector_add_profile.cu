#include <stdio.h>
#include <cuda_runtime.h>

const int DSIZE = 32 * 1048576;

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


__global__ void vector_add(float *d_A, float *d_B, float *d_C, int size){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = idx; i < size;i += gridDim.x * blockDim.x){
            d_C[i] = d_A[i] + d_B[i];
        }
    return ;

}


int main(){
    float *A, *B, *C, *d_A, *d_B, *d_C;
    A = new float[DSIZE]; B = new float[DSIZE]; C = new float[DSIZE];
    for (int i = 0; i < DSIZE; i++){
        A[i] = rand() / (float)RAND_MAX; 
        B[i] = rand() / (float)RAND_MAX; 
        C[i] = rand() / (float)RAND_MAX; 
    }

    cudaMalloc(&d_A, DSIZE*sizeof(float)); 
    cudaMalloc(&d_B, DSIZE*sizeof(float)); 
    cudaMalloc(&d_C, DSIZE*sizeof(float));
    cudaCheckErrors("cudaMalloc failure");

    // copy the values from hosty to device
    cudaMemcpy(d_A, A, DSIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, DSIZE, cudaMemcpyHostToDevice);

    int block  = 1;
    int thread = 1;

    vector_add<<<block, thread>>>(d_A, d_B, d_C, DSIZE);
    cudaCheckErrors("kernel launch failure");

    cudaMemcpy(C, d_C, DSIZE, cudaMemcpyDeviceToHost);

    cudaCheckErrors("kernel execution failure or cudaMemcpy");
    printf("A[0] = %f\n", A[0]);
    printf("B[0] = %f\n", B[0]);
    printf("C[0] = %f\n", C[0]);
    return 0;
}