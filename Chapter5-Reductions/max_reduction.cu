# include <stdio.h>
# include <cuda_runtime.h>

// error macro
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

const int N = 8ULL * 1024ULL * 1024ULL;
const int threads = 256;


__global__ void max_red(float* A, float* out, float N){
    __shared__ float sdata[threads];
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    sdata[tid] = 0.0f;

    while(idx < N){
        sdata[tid] = max( A[idx], sdata[tid]);
        idx += gridDim.x * blockDim.x;
    }
    for (unsigned int s = blockDim.x / 2 ; s > 0; s >>= 1){
        __syncthreads();
        if (tid < s)
            sdata[tid] = max(sdata[tid + s], sdata[tid]);
    }
    if (tid == 0){
        out[blockIdx.x] = sdata[0];
    }

}


int main(){
    float *A, *sum, *d_A, *d_sum;
    const int blocks = 640;
    A = new float[N];
    sum = new float;
    float max_val = 5.0f;

    for(size_t i=0;i<N;i++){
        A[i] = 1.0f;
    }
    A[100] = max_val;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_sum, sizeof(float));
    cudaCheckErrors("memory allocaiton error");

    cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("memory copy erorr");

    max_red<<<blocks, threads>>>(d_A, d_sum, N);
    cudaCheckErrors("reduction kernel launch failure");

    max_red<<<1, threads>>>(d_sum, d_A, blocks); // reduce stage 2
    cudaCheckErrors("reduction kernel2 launch failure");

    cudaMemcpy(sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    printf("reduction output: %f, expected sum reduction output: %f, expected max reduction output: %f\n", *sum, (float)((N-1)+max_val), max_val);
}