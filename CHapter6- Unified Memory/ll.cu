#include <cstdio>
#include <cstdlib>
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

struct ll{
    int val;
    ll* next;
};

// template <typename T>
void alloc_bytes(ll* &list, size_t num_bytes){
    cudaMallocManaged(&list, num_bytes);
}

__host__ __device__
void print_element(ll* list, int elements){
    ll* ele = list;
    for (int i = 0;i < elements; i++){
        printf("key = %d\n", ele->val);
        ele = ele->next;
    }
}

__global__ void gpu_print_element(ll* list, int elements){
    print_element(list, elements);
}

const int num_ele = 5;
const int ele = 3;
int main(){

    ll *list_base, *list;
    alloc_bytes(list_base, sizeof(ll));
    list = list_base;
    for (int i = 0; i < num_ele ; i++){
        list->val = i;
        alloc_bytes(list->next, sizeof(ll));
        list = list->next;
    }
    print_element(list_base, ele);
    gpu_print_element<<<1,1>>>(list_base, ele);
    cudaDeviceSynchronize();
    cudaCheckErrors("cuda errors!!");

}