# Week 1 — SGEMM Kernel Benchmark Results

**GPU:** Tesla T4 (sm_75) · **CUDA:** 12.4 · **Matrix:** 4096 × 4096 × 4096 (FP32)

## Build & Run

```bash
nvcc -O3 -lcublas -arch=sm_75 benchmark.cu -o benchmark
./benchmark

# Or using Makefile:
make ARCH=sm_75 run
```

## Results

| # | Kernel | Time (ms) | GFLOPS | % of cuBLAS | Max Error |
|---|--------|----------:|-------:|------------:|----------:|
| — | **cuBLAS (baseline)** | **31.440** | **4371.5** | **100.0%** | — |
| 1 | Naive | 231.754 | 593.0 | 13.6% | 0.0000e+00 ✅ |
| 2 | Coalesced | 230.596 | 596.0 | 13.6% | 0.0000e+00 ✅ |
| 3 | Shared Memory Tiling | 162.201 | 847.3 | 19.4% | 0.0000e+00 ✅ |
| 4 | 1D Block Tiling | 82.300 | 1670.0 | 38.2% | 0.0000e+00 ✅ |

## Key Takeaways

- **All 4 kernels pass correctness** (err = 0 vs cuBLAS reference).
- **Kernel 1 → 2 (Naive → Coalesced):** ~Same performance (~596 vs 593 GFLOPS). The 1D thread-block remapping doesn't help here because the naive 2D layout already coalesces well for row-major matrices.
- **Kernel 2 → 3 (Coalesced → Shared Tiling):** **1.4× speedup**. Shared memory reuse reduces global memory traffic by ~32× (tile size).
- **Kernel 3 → 4 (Shared Tiling → 1D Block Tiling):** **2.0× speedup**. Each thread computes TM=8 output elements, increasing arithmetic intensity and hiding latency.
- **Best kernel (K4) reaches 38.2% of cuBLAS** — room to improve with 2D block tiling, vectorized loads, and double buffering.