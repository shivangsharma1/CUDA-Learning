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


const size_t N = 8ULL * 1024ULL * 1024ULL;
const int threads = 256;
// const int num_blocks = (N + threads - 1)/threads;


__global__ void atomic_red(float* gdata, float* out){
     size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < N){
        atomicAdd(out, gdata[idx]);
    }
}

__global__ void reduce(float* gdata, float* out){ // not giving correct output, 
    __shared__ float sdata[threads]; //requires a secondary sum again , as each block sum will be saved in 640 locations
    int tid = threadIdx.x;
    sdata[tid] = 0.0f;
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    while (idx<N){ // grid stride loop to load data
        sdata[tid] = gdata[idx];
        idx += blockDim.x * gridDim.x; 
    }

    for(int s = blockDim.x/2; s > 0; s>>=1){
        __syncthreads();
        if(tid < s){
            sdata[tid] += sdata[tid + s]; // parallel sweep reduction
        }
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}


__global__ void reduce_a(float* gdata, float* out){ 
    __shared__ float sdata[threads]; 
    int tid = threadIdx.x;
    sdata[tid] = 0.0f;
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    while (idx<N){ // grid stride loop to load data
        sdata[tid] += gdata[idx];
        idx += blockDim.x * gridDim.x; 
    }

    for(int s = blockDim.x/2; s > 0; s>>=1){
        __syncthreads();
        if(tid < s){
            sdata[tid] += sdata[tid + s]; // parallel sweep reduction
        }
    }
    
    if (tid == 0) atomicAdd(out, sdata[0]);
}

int main(){
    float *A, *sum, *d_A, *d_sum;
    A = new float[N];
    sum = new float;

    for (int i = 0;i<N;i++){
        A[i] = 1.0f;
    }

    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_sum, sizeof(float));
    cudaCheckErrors("cudaMalloc failure");

    cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cuda H2D failure");

    cudaMemset(d_sum, 0, sizeof(float));
    cudaCheckErrors("memset error");

    atomic_red<<<(N + threads - 1)/threads, threads>>>(d_A, d_sum);
    cudaCheckErrors("atomic red kernel error");
    cudaMemcpy(sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    // check results:
    if (*sum != float(N)){
        printf("atomic reduction sum incorrect sum = %f and N = %lu", *sum, N);
        return -1;
    }
    printf("atomic sum reduction correct\n");

    // reduce kernel
    sum = new float;
    const int blocks = 640;
    cudaMemset(d_sum, 0, sizeof(float));
    cudaCheckErrors("cuda memeset fails");
    reduce_a<<<blocks, threads>>>(d_A, d_sum);
    cudaCheckErrors("atomic red kernel error");
    cudaMemcpy(sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    // check results:
    if (*sum != float(N)){
        printf("reduce reduction sum incorrect sum = %f and N = %lu", *sum, N);
        return -1;
    }
    printf("reduce sum reduction correct");



    return 0;
}