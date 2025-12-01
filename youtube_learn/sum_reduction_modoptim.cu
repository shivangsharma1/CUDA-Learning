#include <cstdlib>
#include <iostream>
#include <algorithm>   
#include <vector>
#include <cassert>
#include <numeric>
#include <cuda_runtime.h>

using std::accumulate;
using std::generate;
using std::cout;
using std::vector;

#define sharedmem  256

__global__ void sumreduction(int *a, int *res_a){
    __shared__ int partialsum[sharedmem];

    int thread = blockIdx.x * blockDim.x + threadIdx.x;
    partialsum[threadIdx.x] = a[thread];
    __syncthreads();

    for(int s= 1;s < blockDim.x; s *= 2){
        int index = 2 * threadIdx.x * s;
        if (index<blockDim.x){
            partialsum[index] += partialsum[index+s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0){
        res_a[blockIdx.x] = partialsum[threadIdx.x];
    }
}

int main(){
    int n = 1 << 16;
    size_t bytes = sizeof(int) * n;

    vector<int> a;
    vector<int> res_a;

    for(int i = 0;i<n;i++){
        a.push_back(rand() % 100);
    }

    int *da, *da_res;
    cudaMalloc(&da, bytes);
    cudaMalloc(&da_res, bytes);

    cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int block = (n + threads - 1)/threads;

    sumreduction<<<block, threads>>>(da, da_res);
    sumreduction<<<block, threads>>>(da_res, da_res);
    
    res_a.resize(block);
    cudaMemcpy(res_a.data(), da_res, block, cudaMemcpyDeviceToHost);

    assert(res_a[0] == std::accumulate(begin(a), end(a), 0));
    cout << "COMPLETED SUCCEEFULLY";
    return 0;

} 