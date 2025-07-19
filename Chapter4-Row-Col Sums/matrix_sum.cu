#include <stdio.h>
#include <cuda_runtime.h>

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

const size_t DSIZE = 16384;
const int block_size = 256;


__global__ void row_sum(float *A, float* sums, size_t size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size){
        float sum = 0.0f;
        for (size_t i = 0 ; i < size; i++){
            sum += A[idx * size + i];
        }
    sums[idx] = sum;
}
}

__global__ void col_sum(float *A, float* sums, size_t size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size){
        float sum = 0.0f;
        for (size_t i = 0 ; i < size; i++){
            sum += A[idx + size * i];
        }
    sums[idx] = sum;
}
}

bool validate(float *data, size_t size){
    for(size_t i = 0;i<size; i++){
        if (data[i] != (float)size) {printf("result mismatch at index %lu, was %f, but should be : %f", i, data[i], (float)size); return false;}
    }
    return true;
}


int main(){
    float *A, *sums, *d_A, *d_sums;
    A = new float[DSIZE * DSIZE];
    sums = new float[DSIZE]();

    for (int i = 0; i<DSIZE*DSIZE;i++){
        A[i] = 1.0f;
    }

    cudaMalloc(&d_A, DSIZE * DSIZE * sizeof(float));
    cudaMalloc(&d_sums, DSIZE * sizeof(float));
    cudaCheckErrors("cudaMalloc failure");

    // copy the matrix 
    cudaMemcpy(d_A, A, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("mem copy error");
    
    // row kernel launch
    row_sum<<<(DSIZE + block_size -1)/block_size, block_size>>>(d_A, d_sums, DSIZE);
    // check error
    cudaCheckErrors("kernel execution failure");

    // copy the rowsum back
    cudaMemcpy(sums, d_sums, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
    // check error
    cudaCheckErrors("memcpy failure");

    for (size_t i = 0; i < DSIZE; i++){
        printf("%f", sums[i]);
        if (i == 10){
            break;
        }
    }

    if (!validate(sums, DSIZE)) return -1; 
    printf("row sums correct!\n");

    // resetting the d_sums memory
    cudaMemset(d_sums, 0, DSIZE * sizeof(float));

    // col kernel launch
    col_sum<<<(DSIZE + block_size -1)/block_size, block_size>>>(d_A, d_sums, DSIZE);
    cudaCheckErrors("kernel launch failure");

    cudaMemcpy(sums, d_sums, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("kernel execution failure, col");
    
    if (!validate(sums, DSIZE)) return -1; 
    printf("column sums correct!\n");
    return 0;

}