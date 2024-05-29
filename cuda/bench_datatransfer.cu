#include "bench_datatransfer.cuh"

#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

void SetupBenchmarkIO(float** h_inBuf, float** h_outBuf, float** d_inBuf, float** d_outBuf,
    const int IOTEST_INBUFCOUNT, const int IOTEST_OUTBUFCOUNT) {
    cudaError_t err = cudaSuccess;

    *h_inBuf = (float*)malloc(IOTEST_INBUFCOUNT * sizeof(float));
    *h_outBuf = (float*)malloc(IOTEST_OUTBUFCOUNT * sizeof(float));

    if (*h_inBuf == NULL || *h_outBuf == NULL) {
		fprintf(stderr, "Failed to allocate host buffers\n");
		exit(EXIT_FAILURE);
	}
    
    err = cudaMalloc((void**)d_inBuf, IOTEST_INBUFCOUNT * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
            cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaMalloc((void**)d_outBuf, IOTEST_OUTBUFCOUNT * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
            cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // Init host vector with random floats
    for (int i = 0; i < IOTEST_INBUFCOUNT; i++) {
        (*h_inBuf)[i] = rand() / (float)RAND_MAX;
    }
}