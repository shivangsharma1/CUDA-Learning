#include <cuda_runtime.h>
#include <stdio.h>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <vector>

using std::vector;
using std::cout;


__global__ void matmul(int *a, int *b, int *c, int N){
    int col = (blockIdx.x * blockDim.x) + threadIdx.x;
    int row = (blockIdx.y * blockDim.y) + threadIdx.y;
    c[row * N + col] = 0;
    if (row < N && col < N){
        for(int k = 0; k <N; k++){
            c[row * N + col] += a[row * N + k]  * b[k * N + col];
        }
    }
}


void verify_result(vector<int> &a, vector<int> &b, vector<int> &c, int N) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      int tmp = 0;
      for (int k = 0; k < N; k++) {
    
        tmp += a[i * N + k] * b[k * N + j];
      }
      assert(tmp == c[i * N + j]);
    }
  }
}


int main(){
    const size_t N = 1<<10;
    size_t bytes = N * N * sizeof(int) ;

    vector<int> a;
    vector<int> b;
    vector<int> c;

    a.reserve(N); b.reserve(N); c.reserve(N);

    for(size_t row = 0; row<N;row++){
        for(size_t col = 0; col < N;col++){
            a.push_back(rand() /100);
            b.push_back(rand() /100);
        }
    }

    c.resize(N*N);

    int *da, *db, *dc;
    cudaMalloc(&da, bytes);
    cudaMalloc(&db, bytes);
    cudaMalloc(&dc, bytes);

    cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice);

    int THREADS = 32;
    int BLOCKS = (N + THREADS - 1) / THREADS;

    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);

    matmul<<<blocks, threads>>>(da, db, dc, N);
    
    cudaMemcpy(c.data(), dc, bytes, cudaMemcpyDeviceToHost);

    verify_result(a, b, c, N);

    cout << "COMPLETED SUCCESSFULLY";
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    return 0;
}