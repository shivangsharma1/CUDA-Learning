#include <stdio.h>
#include <cuda_runtime.h>
#include <cassert>
#include <vector>
#include <iostream>
#include <algorithm>

using std::vector;

__global__ void vectorAdd(int* da, int* db, int *dc, int N){
    int thread = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread < N){
        dc[thread] = da[thread] + db[thread];
    }
}

void verify_result(vector<int> a, vector<int> b, vector<int> c, int N){

    for(int i = 0 ; i <N ;i++){
        assert(c[i] == a[i] + b[i]);
    }

}


int main(){
    constexpr int N = 1 << 16;
    constexpr size_t bytes = sizeof(int) * N;

    vector<int> a;
    vector<int> b;
    vector<int> c;

    a.reserve(N); b.reserve(N); c.reserve(N);

    // initialize the vector
    for(int i = 0;i<N;i++){
        a.push_back(rand() % 100);
        b.push_back(rand() % 100);
    }
    
    int *da, *db, *dc;
    cudaMalloc(&da, bytes);
    cudaMalloc(&db, bytes);
    cudaMalloc(&dc, bytes);

    // data copy from host to device
    cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice);

    int NUMTHREADS = 1<<10;
    int NUMBLOCKS = (N + NUMTHREADS - 1) / NUMTHREADS;

    vectorAdd<<<NUMBLOCKS, NUMTHREADS>>>(da, db, dc, N);

    c.resize(N);

    // copy back
    cudaMemcpy(c.data(), dc, bytes, cudaMemcpyDeviceToHost);
    verify_result(a, b, c, N);
    cudaFree(da); cudaFree(db); cudaFree(dc);

    return 0;
}
