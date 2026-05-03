# ─────────────────────────────────────────────────────
#  SGEMM Kernel Benchmark Makefile
#  Usage:
#    make             → build benchmark
#    make run         → build + run
#    make profile     → build + profile with ncu
#    make clean       → remove binaries
# ─────────────────────────────────────────────────────

NVCC     = nvcc
FLAGS    = -O3 -lcublas

# Detect GPU arch (default sm_80 for A100, change if needed)
# T4=sm_75  V100=sm_70  A100=sm_80  L4=sm_89  H100=sm_90
ARCH     ?= sm_80

all: benchmark

benchmark: benchmark.cu
	$(NVCC) $(FLAGS) -arch=$(ARCH) benchmark.cu -o benchmark

run: benchmark
	./benchmark

# Profile all 4 kernels — shows key metrics
profile: benchmark
	ncu --set full -o profile_report ./benchmark

# Profile with specific metrics only (faster)
profile-quick: benchmark
	ncu --metrics \
		sm__throughput.avg.pct_of_peak_sustained_elapsed,\
		dram__bytes.sum,\
		l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
		sm__warps_active.avg.pct_of_peak_sustained_active \
		./benchmark

# Profile one kernel at a time (cleaner reports)
profile-k1: benchmark
	ncu --kernel-name naive_sgemm --launch-count 1 -o profile_k1 ./benchmark

profile-k2: benchmark
	ncu --kernel-name coalesce_sgemm --launch-count 1 -o profile_k2 ./benchmark

profile-k3: benchmark
	ncu --kernel-name share_tiling --launch-count 1 -o profile_k3 ./benchmark

profile-k4: benchmark
	ncu --kernel-name blocktiling_1d --launch-count 1 -o profile_k4 ./benchmark

clean:
	rm -f benchmark profile_*.ncu-rep
