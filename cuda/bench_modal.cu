#include "bench_modal.cuh"

#include "globals.cuh"

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cuComplex.h>

#include <stdio.h>
#include <iostream>
#include <chrono>
#include <algorithm>

// Test for arithmetic (modal processing)


// Implementation of cexpf for cuComplex.
// CLEANUP: This is not the version used in the paper on submission, but the version
// from a separate filterbank project (faster); unify them.
__device__ cuComplex my_cexpf(cuComplex expon) {
    cuComplex comp_1_w;
    float s, c;
    float e = exp(expon.x);
    sincosf(expon.y, &s, &c);
    comp_1_w.x = c * e;
    comp_1_w.y = s * e;
    return comp_1_w;
}

__global__ void ModalKernel_Debug(const float* bufIn, float* bufOut, int nModes) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < nModes) {
        float amp = bufIn[i * kNumModeParams + 0];
        float freq = bufIn[i * kNumModeParams + 1];
        float phase = bufIn[i * kNumModeParams + 2];
        float re = bufIn[i * kNumModeParams + 3];
        float im = bufIn[i * kNumModeParams + 4];
        float state_re = bufIn[i * kNumModeParams + 5];
        float state_im = bufIn[i * kNumModeParams + 6];
        state_re = 0.5f;
        state_im = 0.5f;
        cuComplex start = make_cuComplex(state_re, state_im);
        for (int si = 0; si < BUFSIZE; si++) {
            my_cexpf(start);
            bufOut[i * BUFSIZE + si] = amp * freq * phase;
            if (i < 32) {
                bufOut[i * BUFSIZE + si] = start.x;
            }
        }
    }
}

void SetupBenchmarkModal(float **h_inBufPtr, float**h_outBufPtr, float** d_inBufPtr, float** d_outBufPtr) {
    *h_inBufPtr = (float*)malloc(kNumModes * kNumModeParams * sizeof(float));
    *h_outBufPtr = (float*)malloc(BUFSIZE * kModalOutputTracks * sizeof(float));

    cudaError_t err = cudaSuccess;
    if ((*h_inBufPtr) == NULL || (*h_outBufPtr) == NULL) {
		fprintf(stderr, "Failed to allocate host buffers\n");
		exit(EXIT_FAILURE);
	}

    float* d_inBuf = NULL;
    err = cudaMalloc((void**)&d_inBuf, kNumModes * kNumModeParams * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
            cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float* d_outBuf = NULL;
    err = cudaMalloc((void**)&d_outBuf, BUFSIZE * kModalOutputTracks * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
            cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    float *h_inBuf = *h_inBufPtr;
    for (int i = 0; i < kNumModes; i++) {
        int pIdx = i * kNumModeParams;
        h_inBuf[pIdx + 0] = rand() / (float)RAND_MAX;
        h_inBuf[pIdx + 1] = rand() / (float)RAND_MAX;
        h_inBuf[pIdx + 2] = rand() / (float)RAND_MAX;
        h_inBuf[pIdx + 3] = rand() / (float)RAND_MAX;
        h_inBuf[pIdx + 4] = rand() / (float)RAND_MAX;
        h_inBuf[pIdx + 5] = rand() / (float)RAND_MAX;
        h_inBuf[pIdx + 6] = rand() / (float)RAND_MAX;
    }
}

// CLEANUP: Eliminate redundancy between all the .cu files for the benchmarking loop.
// It's copy/pasted at the moment which is prone to divergence.
void RunBenchmarkModal(float** d_inBuf, float** h_inBuf, float **d_outBuf,
    float **h_outBuf, int kNumModes, int kNumModeParams, std::vector<float>& latencies) {
    printf("Running modal benchmark\n");
    cudaError_t err = cudaSuccess;
    for (int i = 0; i < NRUNS; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        err = cudaMemcpy(d_inBuf, h_inBuf, kNumModes * kNumModeParams * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr,
                "Failed to copy vector input from host to device (error code %s)!\n",
                cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Launch the CUDA Kernel
        constexpr int threadsPerBlock = 256;
        int numElements_iotest = kNumModes;
        int blocksPerGrid = (kNumModes + threadsPerBlock - 1) / threadsPerBlock;
        ModalKernel_Debug << <blocksPerGrid, threadsPerBlock >> > (*d_inBuf, *d_outBuf, kNumModes);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to launch kernel(error code %s)!\n",
                cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Copy result back out
        err = cudaMemcpy(*h_outBuf, d_outBuf, kModalOutputTracks * BUFSIZE * sizeof(float), cudaMemcpyDeviceToHost);

        if (err != cudaSuccess) {
            fprintf(stderr,
                "Failed to copy vector C from device to host (error code %s)!\n",
                cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        // TODO: validate (other benchmarks -- or this one if we want to test data transfer internally).

        auto end = std::chrono::high_resolution_clock::now();
        // Compute the duration in milliseconds
        std::chrono::duration<float, std::milli> duration = end - start;
        latencies.push_back(duration.count());
        std::cout << "Duration: " << duration.count() << "ms" << std::endl;

        if (ENABLE_DAWSIM_SLEEP) {
            // Sleep for 93.75ms minus duration.
            std::this_thread::sleep_for(std::chrono::milliseconds(SLEEP_MS) - duration);
        }
        if (ENABLE_DAWSIM_SPIN) {
            while (std::chrono::high_resolution_clock::now() - start < std::chrono::milliseconds(SLEEP_MS)) {
                // Spin
            }
        }
    }
    printVectorStats(latencies);
    writeVectorToFile(latencies, OUTFILE);

    // Free host and device global memory. Not error checking since we're writing data and exit.
    cudaFree(d_inBuf); cudaFree(d_outBuf);
    free(h_inBuf); free(h_outBuf);
}
