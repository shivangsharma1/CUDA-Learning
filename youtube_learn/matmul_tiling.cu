#include <cuda_runtime.h>
#include <stdio.h>
#include <cassert>
#include <vector>
#include <functional>
#include <iostream>

using std::vector;
using std::cout;


const int N = 1 << 10;
const int sharedmemsize = 1 << 10;


__global__ void matmul(int* da, int* db, int* dc, int N){
    
    __shared__ int sa[sharedmemsize];
    __shared__ int sb[sharedmemsize];

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    int sum = 0;

    for (int tile = 0; tile <N;tile += blockDim.x){
        int sharedIdx = threadIdx.y * blockDim.x + threadIdx.x;

        int arow = row;
        int acol = tile + threadIdx.x;
        sa[sharedIdx] = da[arow * N + acol];

        int brow = tile + threadIdx.y;
        int bcol = col;
        sb[sharedIdx] = db[brow * N + bcol];

        __syncthreads();

        for (int k= 0; k<blockDim.x ;k++){
            int vala = sa[threadIdx.y * blockDim.x + k];
            int valb = sb[blockDim.y * k + threadIdx.x];

            sum += (vala * valb);
        }
        __syncthreads();
    }

    dc[row * N  + col] = sum;

}


// Check result on the CPU
void verify_result(vector<int> &a, vector<int> &b, vector<int> &c) {
  // For every row...
  for (int i = 0; i < N; i++) {
    // For every column...
    for (int j = 0; j < N; j++) {
      // For every element in the row-column pair
      int tmp = 0;
      for (int k = 0; k < N; k++) {
        // Accumulate the partial results
        tmp += a[i * N + k] * b[k * N + j];
      }

      // Check against the CPU result
      assert(tmp == c[i * N + j]);
    }
  }
}



int main(){
    size_t bytes = N * N * sizeof(int);

    vector<int> a; vector<int> b; vector<int> c;
    a.reserve(N * N); b.reserve(N * N); c.reserve(N * N);

    for(int i=0;i<N;i++){
        for (int j=0;j<N;j++){
            a.push_back(rand()/100);
            b.push_back(rand()/100);
        }
    }

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
    c.resize(N * N);
    cudaMemcpy(c.data(), dc, bytes, cudaMemcpyDeviceToHost);

    verify_result(a, b, c);

    cout << "COMPLETED SUCCESSFULLY";

    return 0;

    
}
