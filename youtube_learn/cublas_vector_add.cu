#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

// initialize vector
void vector_init(float *a, int n){
    for(int i = 0; i<n;i++){
        a[i] = (float)(rand() % 100);
    }
}

//  verify results
void verify_results(float *a, float *b, float *c, float factor,int n){
    for(int i = 0;i < n ; i++){
        assert(c[i] == factor * a[i] + b[i]);
    }
}


int main(){
    int n = 1 << 2;
    size_t bytes = n * sizeof(float);

    float *a, *b, *c;
    float *da, *db;

    a = (float*)malloc(bytes);
    b = (float*)malloc(bytes);
    c = (float*)malloc(bytes);
    cudaMalloc(&da, bytes);
    cudaMalloc(&db, bytes);

    vector_init(a, n);
    vector_init(b, n);

    cublasHandle_t handle;
    cublasCreate_v2(&handle);

    // copy the vector , cublas way
    cublasSetVector(n, sizeof(float), a, 1, da, 1);
    cublasSetVector(n, sizeof(float), b, 1, db, 1);

    // launching saxpy kernel
    const float scale = 2.0f;
    cublasSaxpy(handle, n, &scale, da, 1, db, 1);

    cublasGetVector(n, sizeof(float), db, 1, c, 1);

    verify_results(a, b, c, scale, n);

    cublasDestroy(handle);
    cudaFree(da);
    cudaFree(db);
    free(a); free(b); free(c);
}
