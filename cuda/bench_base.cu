#include "bench_base.cuh"

#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

void GPUABenchmark::InitDefaultBuffers(
	std::vector<int> host_input_buffer_sizes_bytes,
	std::vector<int> host_output_buffer_sizes_bytes,
    std::vector<int> device_input_buffer_sizes_bytes,
    std::vector<int> device_output_buffer_sizes_bytes) {

    cudaError_t err = cudaSuccess;

    // TODO(travis): Support multiple buffers, or zero.
    // this->hIn = (float*)malloc(host_input_buffer_sizes_bytes[0]);
    // this->hOut = (float*)malloc(host_output_buffer_sizes_bytes[0]);
    // Allocate memory for the input and output buffers
    cudaMallocHost((void**)hIn, sizeof(float) * host_input_buffer_sizes_bytes[0]);
    cudaMallocHost((void**)hOut, sizeof(float) * host_output_buffer_sizes_bytes[0]);

    err = cudaMalloc((void**)(this->dIn), device_input_buffer_sizes_bytes * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device input mem(error code %s)!\n",
            cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void**)(this->dOut), device_output_buffer_sizes_bytes * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device input mem(error code %s)!\n",
            cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
