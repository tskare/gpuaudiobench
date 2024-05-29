/**
 * CUDA GPGPU Audio Microbenchmarks and I/O benchmarks suite.
 * See README in repository root for overall project details.
 */

 // The Visual Studio project and starter code was adapted from CUDA
 // toolkit examples, with the below license.

 /* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
  *
  * Redistribution and use in source and binary forms, with or without
  * modification, are permitted provided that the following conditions
  * are met:
  *  * Redistributions of source code must retain the above copyright
  *    notice, this list of conditions and the following disclaimer.
  *  * Redistributions in binary form must reproduce the above copyright
  *    notice, this list of conditions and the following disclaimer in the
  *    documentation and/or other materials provided with the distribution.
  *  * Neither the name of NVIDIA CORPORATION nor the names of its
  *    contributors may be used to endorse or promote products derived
  *    from this software without specific prior written permission.
  *
  * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
  * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
  * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
  * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
  * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
  * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
  * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
  * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
  */

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

// CUDA complex types, used for Modal filter bank trials.
#include <cuComplex.h>

#include <iostream>
#include <vector>
#include <thread>

#include <helper_cuda.h>

#include "globals.cuh"

// Benchmark includes
#include "bench_datatransfer.cuh"
#include "bench_modal.cuh"
//#include "bench_gainstats.cuh"
#include "benchmark_rndmem.cuh"

// See globals.cuh for cross-benchmark parameters.
// See individual .cuh files for benchmark-specific parameters.

// GPGPU Audio benchmark kernel:
// Test for data transfer, weighted
__global__ void DataTransferBenchmarkKernel(const float* bufIn, float* bufOut, int numElements) {
    // No-op, but keep this piece as the Metal impl performs the same implicitly.
    // It seems it is optimized out of the intermediate code however.
    int i = blockDim.x * blockIdx.x + threadIdx.x;
}


// Kernel to compute basic gain 
__global__ void GainStatsKernel(const float* bufIn, float* bufOut, float* stats, int numElements) {
    int trackidx = blockDim.x * blockIdx.x + threadIdx.x;
    const int nstats = 2;
    float mean = 0.0f;
    for (int i=0; i<BUFSIZE; i++) {
        float samp = bufIn[trackidx * BUFSIZE + i];
        mean += samp;
        bufOut[trackidx * BUFSIZE + i] = samp * 0.5f;
        // Max
        if (samp > stats[trackidx * (NTRACKS * nstats) + 0]) {
			stats[trackidx*(NTRACKS * nstats) + 0] = samp;
		}
	}
    mean /= BUFSIZE;
}

/**
 * Host main routine
 */
int main(void) {
  // Error code to check return values for CUDA calls
  cudaError_t err = cudaSuccess;

#if DO_BENCHMARK_IO
  float* h_inBuf = NULL;
  float* h_outBuf = NULL;
  float* d_inBuf = NULL;
  float *d_outBuf = NULL;
  SetupBenchmarkIO(&h_inBuf, &h_outBuf, &d_inBuf, &d_outBuf, IOTEST_INBUFCOUNT, IOTEST_OUTBUFCOUNT);
#endif
#if DO_BENCHMARK_RNDMEM
    float* h_sampleMem = nullptr;
    float* d_sampleMem = nullptr;
    int* h_playheads = nullptr;
	int* d_playheads = nullptr;
    float* h_out = nullptr;
    float* d_out = nullptr;

    float playheadsStart[NTRACKS];
    float playheadsEnd[NTRACKS];
    int minLoopLen = 1000;
    int maxLoopLen = 48000;

    int samplebufferEnd = kSampleMemNumElems - BUFSIZE;

  SetupBenchmarkRndMem(&h_sampleMem, &d_sampleMem,
      &h_playheads, &d_playheads,
      playheadsStart,
      playheadsEnd,
      minLoopLen,
      maxLoopLen,
      samplebufferEnd,
      &h_out, &d_out);
#endif
  // TODO: Finish copying this out into its own file. Was here for debugging.
#if DO_BENCHMARK_GAINSTATS
    float* h_inBuf = (float*)malloc(NTRACKS*BUFSIZE * sizeof(float));
    float* h_outBuf = (float*)malloc(NTRACKS * BUFSIZE * sizeof(float));

  float* d_inBuf = NULL;
  err = cudaMalloc((void**)&d_inBuf, NTRACKS * BUFSIZE * sizeof(float));
  if (err != cudaSuccess) {
      fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
          cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }
  float* d_outBuf = NULL;
  err = cudaMalloc((void**)&d_outBuf, NTRACKS * BUFSIZE * sizeof(float));
  if (err != cudaSuccess) {
      fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
          cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }
  // Init host vector with random floats
  for (int i = 0; i < NTRACKS * BUFSIZE; i++) {
      h_inBuf[i] = rand() / (float)RAND_MAX;
  }
#endif
#if DO_BENCHMARK_MODAL
  float* h_inBuf = nullptr;
  float* h_outBuf = nullptr;
  float* d_inBuf = nullptr;
  float* d_outBuf = nullptr;

  SetupBenchmarkModal(&h_inBuf, &h_outBuf, &d_inBuf, &d_outBuf);
#endif

  // Main benchmark loop
  std::vector<float> latencies;
  
#if DO_BENCHMARK_IO
  printf("Running I/O benchmark\n");
  for (int i = 0; i < NRUNS; i++) {
      // Avoiding printfs
    // printf("Copy input data from the host memory to the CUDA device\n");
      auto start = std::chrono::high_resolution_clock::now();

      err = cudaMemcpy(d_inBuf, h_inBuf, IOTEST_INBUFCOUNT * sizeof(float), cudaMemcpyHostToDevice);
      if (err != cudaSuccess) {
          fprintf(stderr,
              "Failed to copy vector input from host to device (error code %s)!\n",
              cudaGetErrorString(err));
          exit(EXIT_FAILURE);
      }

     // Launch the CUDA Kernel
    constexpr int threadsPerBlock = 256;
    constexpr int numElements_iotest = threadsPerBlock*100;
    int blocksPerGrid = (numElements_iotest + threadsPerBlock - 1) / threadsPerBlock;
    DataTransferBenchmarkKernel<<<blocksPerGrid, threadsPerBlock>>>(d_inBuf, d_outBuf, numElements_iotest);
 
    err = cudaGetLastError();
      if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch kernel(error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
      }

      // Copy result back out
      err = cudaMemcpy(h_outBuf, d_outBuf, IOTEST_OUTBUFCOUNT * sizeof(float), cudaMemcpyDeviceToHost);

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
      cout << "Duration: " << duration.count() << "ms" << endl;
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
#endif

#if DO_BENCHMARK_GAINSTATS
    float* d_stats = NULL;
    cudaMalloc((void**)&d_stats, NTRACKS * (2) * sizeof(float));
  printf("Running Gain+Stats benchmark\n");
  for (int i = 0; i < NRUNS; i++) {
      auto start = std::chrono::high_resolution_clock::now();

      err = cudaMemcpy(d_inBuf, h_inBuf, NTRACKS * BUFSIZE * sizeof(float), cudaMemcpyHostToDevice);
      if (err != cudaSuccess) {
          fprintf(stderr,
              "Failed to copy vector input from host to device (error code %s)!\n",
              cudaGetErrorString(err));
          exit(EXIT_FAILURE);
      }

      // Launch the CUDA Kernel
      constexpr int threadsPerBlock = NTRACKS;
      constexpr int numElements_iotest = NTRACKS * BUFSIZE;
      int blocksPerGrid = (numElements_iotest + threadsPerBlock - 1) / threadsPerBlock;
      GainStatsKernel << <blocksPerGrid, threadsPerBlock >> > (d_inBuf, d_outBuf, d_stats, numElements_iotest);

      err = cudaGetLastError();
      if (err != cudaSuccess) {
          fprintf(stderr, "Failed to launch kernel(error code %s)!\n",
              cudaGetErrorString(err));
          exit(EXIT_FAILURE);
      }

      // Copy result back out
      /*
      err = cudaMemcpy(h_outBuf, d_outBuf, NTRACKS * BUFSIZE * sizeof(float), cudaMemcpyDeviceToHost);

      if (err != cudaSuccess) {
          fprintf(stderr,
              "Failed to copy vector C from device to host (error code %s)!\n",
              cudaGetErrorString(err));
         exit(EXIT_FAILURE); // TODO and stats
      }
      */
      // TODO: validate (other benchmarks -- or this one if we want to test data transfer internally).

      auto end = std::chrono::high_resolution_clock::now();
      // Compute the duration in milliseconds
      std::chrono::duration<float, std::milli> duration = end - start;
      latencies.push_back(duration.count());
      cout << "Duration: " << duration.count() << "ms" << endl;
  }
  printVectorStats(latencies);
  writeVectorToFile(latencies, OUTFILE);

  // Free host and device global memory. Not error checking since we're writing data and exit.
  cudaFree(d_inBuf); cudaFree(d_outBuf);
  free(h_inBuf); free(h_outBuf);
#endif

#if DO_BENCHMARK_MODAL
  RunBenchmarkModal(&d_inBuf, &h_inBuf, &d_outBuf, &h_outBuf, kNumModes, kNumModeParams, latencies);
#endif

#if DO_BENCHMARK_RNDMEM
  RunBenchmarkRndMem(&d_playheads, &h_playheads, &d_sampleMem, &d_out, &h_out,
      latencies,
      playheadsStart, playheadsEnd);
#endif

  printf("Done\n");
  return 0;
}
