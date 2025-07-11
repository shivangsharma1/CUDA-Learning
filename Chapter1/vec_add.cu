#include <stdio.h>
const int DSIZE = 4096;
const int block_size = 256;

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

__global__ void add(float* a, float* b, float* c, int DSIZE){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < DSIZE){
        c[idx] = a[idx] + b[idx];
    }
    return ;
}


int main(){
    float *A, *B, *C, *d_A, *d_B, *d_C;
    int size = DSIZE * sizeof(int);
    

    // allocate space on device
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    cudaCheckErrors("cudaMalloc failure"); // error checking

    // allocate space on host and initialize value
    A = new float[DSIZE];
    B = new float[DSIZE];
    C = new float[DSIZE];

    // filling the random values
    for(int i = 0; i<DSIZE;i++){
        A[i] = rand()/(float)RAND_MAX;
        B[i] = rand()/(float)RAND_MAX;
    }

    // copy to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // call kernel
    add<<<(DSIZE+block_size-1)/block_size, block_size>>>(d_A, d_B, d_C, DSIZE);

    // copy the C data back to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // printing values
    printf("A[0] = %f\n", A[0]);
    printf("B[0] = %f\n", B[0]);
    printf("C[0] = %f\n", C[0]);
    return 0;

}