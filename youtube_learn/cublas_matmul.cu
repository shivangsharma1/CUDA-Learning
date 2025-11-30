#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <stdio.h>
#include <math.h>



void verify_results(float *a, float *b, float*c, int n){
    float epsilon = 0.001;
    for(int i=0; i<n;i++){
        for (int j=0;j<n;j++){
            
            float temp = 0; 
            for (int k=0;k<n;k++){
                temp += a[k*n +i] * b[j*n +k];
            }
            assert(fabs(c[j*n +i] - temp) < epsilon);
        }
    }
}

int main(){
    size_t n = 1<<10 ;
    size_t bytes = n * n *sizeof(float);

    float *a, *b, *c;
    float *da, *db, *dc;

    a = (float*)malloc(bytes);
    b = (float*)malloc(bytes);
    c = (float*)malloc(bytes);

    cudaMalloc(&da, bytes);
    cudaMalloc(&db, bytes);
    cudaMalloc(&dc, bytes);

    curandGenerator_t numgen;
    curandCreateGenerator(&numgen, CURAND_RNG_PSEUDO_DEFAULT);

    curandSetPseudoRandomGeneratorSeed(numgen, (unsigned long long) clock());
    

    // filling the matrix on GPU
    curandGenerateUniform(numgen, da, n*n);
    curandGenerateUniform(numgen, db, n*n);

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, da, n, db, n, &beta, dc, n);

    // copy back to host
    cudaMemcpy(a, da, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(b, db, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(c, dc, bytes, cudaMemcpyDeviceToHost);

    verify_results(a, b, c, n);
    return 0;

}