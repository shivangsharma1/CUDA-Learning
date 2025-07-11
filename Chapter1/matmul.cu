#include <stdio.h>
#include <time.h>

const int DSIZE = 4096;
const int block_size = 16;
const float A_val = 1.0f;
const float B_val = 2.0f;
const int size = DSIZE * DSIZE * sizeof(float);


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


// matrix multiply
__global__ void matmul(const float *A, const float*B, float*C, int ds){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if ((idx < ds) && (idy < ds)){
        float temp = 0;
        for (int i = 0; i < ds ; i++){
            temp += A[idy * ds + i] * B[i * ds + idx];
        }
        C[idy*ds + idx] = temp;
    }
}

int main(){
    float *A, *B, *C, *d_A, *d_B, *d_C;
    clock_t t0, t1, t2;
    double t1sum = 0.0;
    double t2sum = 0.0;

    // clock start
    t0 = clock();

    A = new float[DSIZE*DSIZE];
    B = new float[DSIZE*DSIZE];
    C = new float[DSIZE*DSIZE];
    
    for (int i = 0; i<DSIZE*DSIZE; i++){
        A[i] = A_val;
        B[i] = B_val;
        C[i] = 0;
    }
    
    // Initialization timing
    t1 = clock();
    t1sum = ((double)(t1-t0))/CLOCKS_PER_SEC;
    printf("Init took %f seconds.  Begin compute\n", t1sum);
    
    // allocate device mem and copy content 
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    cudaCheckErrors("memory not allocated");
    
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    cudaCheckErrors("copy erorr");

    dim3 block(block_size, block_size);
    dim3 grid((DSIZE + block.x -1)/block.x, (DSIZE + block.y -1)/block.y);
    matmul<<<grid, block>>>(d_A, d_B, d_C, DSIZE);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    
    // GPU timing
    t2 = clock();
    t2sum = ((double)(t2-t1))/CLOCKS_PER_SEC;
    printf ("Done. Compute took %f seconds\n", t2sum);

    // Verify results
  for (int i = 0; i < DSIZE*DSIZE; i++) if (C[i] != A_val*B_val*DSIZE) {printf("mismatch at index %d, was: %f, should be: %f\n", i, C[i], A_val*B_val*DSIZE); return -1;}
  printf("Success!\n"); 
  return 0;


}
