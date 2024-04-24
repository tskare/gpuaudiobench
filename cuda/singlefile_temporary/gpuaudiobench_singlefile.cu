/**
 * CUDA GPGPU Audio Microbenchmarks and I/O benchmarks suite, single-file version.
 * This version is intended to be pasted over any of the single-file CUDA examples,
 * for example vectorAdd.cu
 * It is a work in progress, and will be replaced by one that is multiple files,
 * as it has lost readability.
 * However it may be used to gather I/O stats, and was designed to be able to be
 * run on any system with the CUDA toolkit installed, without any additional
 * dependencies.
 * 
 * Please note the Modal benchmark here is not the version used in the benchmarks
 * in the paper draft, but a debug/draft kernel used with the profiler for looking
 * at ALU behavior. Its performance will be close to but not exact
 * (essentially the initialization and state are a bit different).
 * This and the filtering examples (e.g. IIR_N) not benchmarked in the paper
 * were separate files.
 */

// The code was adapted to be placed on top of the aforemented example, with the
// following license preserved:

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

#include <stdio.h>
#include <chrono>
#include <algorithm>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

// CUDA complex types, used for Modal filter bank trials.
#include <cuComplex.h>

#include <iostream>
#include <vector>
#include <thread>

#include <helper_cuda.h>

// CLEANUP: These should be made commandline parameters.
#define NTRACKS 4*4096
#define FS 48000
#define BUFSIZE 512
#define NRUNS 100

// Allow sleep
#define ENABLE_DAWSIM_SLEEP false
#define SLEEP_MS 90
// Allow spinning
#define ENABLE_DAWSIM_SPIN false

// I/O mix, [1..99], as a percent of total.
#define IOTEST_INFRAC 1
#define IOTEST_INSCALE (IOTEST_INFRAC / 100.0f)
#define IOTEST_OUTSCALE (1.0f - IOTEST_INSCALE)

constexpr int ONEMEG_OF_FLOATS = (1024 * 1024 / 4);
constexpr int IOTEST_BUFSIZE = (100 * ONEMEG_OF_FLOATS);
constexpr int IOTEST_INBUFCOUNT = (((int)(IOTEST_BUFSIZE * IOTEST_INSCALE)));
constexpr int IOTEST_OUTBUFCOUNT = (((int)(IOTEST_BUFSIZE * IOTEST_OUTSCALE)));

constexpr int kModalOutputTracks = 32;
constexpr int numElements = 50000;

// Parameters for modal benchmark
constexpr int kNumModes = 1024*1024;
// We pack 8 floats per mode:
// amplitude, freq, phase, state.re, state.im, 3 reserved/padding
// [update: used for amplitude coupling in the real kernel. TODO: update]
constexpr int kNumModeParams = 8;

// Choose one of N benchmarks.
#define DO_BENCHMARK_IO false
#define DO_BENCHMARK_GAINSTATS false
#define DO_BENCHMARK_MODAL false
#define DO_BENCHMARK_RNDMEM true

using std::cout;
using std::endl;

// TODO: Write to stdout or Windows console.
// Our driver script watches for this and copies to another file to generate tables.
#define OUTFILE "c:\\tmp\\latencies.txt"
void writeVectorToFile(const std::vector<float>& vec, const std::string& filename) {
  std::ofstream file(filename);
  for (const auto& val : vec) {
	file << val << std::endl;
  }
  file.close();
}

void printVectorStats(const std::vector<float>& vec) {
    float sum = 0.0f;
    float min = std::numeric_limits<float>::max();
    float max = std::numeric_limits<float>::min();
    for (const auto& val : vec) {
        sum += val;
        if (val < min) min = val;
        if (val > max) max = val;
    }
    float avg = sum / vec.size();
    std::cout << "Min: " << min << " Max: " << max << " Avg: " << avg << std::endl;

    // Sort vector
    std::vector<float> sortedVec = vec;
    std::sort(sortedVec.begin(), sortedVec.end());
	// Print median
	float median = 0.0f;
	if (sortedVec.size() % 2 == 0) {
		median = (sortedVec[sortedVec.size() / 2 - 1] + sortedVec[sortedVec.size() / 2]) / 2;
	} else {
		median = sortedVec[sortedVec.size() / 2];
	}
	std::cout << "p50: "<< sortedVec[sortedVec.size() * 0.50] << " p95: " << sortedVec[sortedVec.size() * 0.95] << " p99: " << sortedVec[sortedVec.size() * 0.99] << std::endl;
}

/**
 * CUDA Kernel Device code, retained from VectorAdd example.
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void vectorAdd(const float *A, const float *B, float *C,
                          int numElements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < numElements) {
    C[i] = A[i] + B[i] + 0.0f;
  }
}

// GPGPU Audio benchmark kernel:
// Test for data transfer, weighted
__global__ void DataTransferBenchmarkKernel(const float* bufIn, float* bufOut, int numElements) {
    // No-op, but keep this piece as the Metal impl performs the same implicitly.
    // It seems it is optimized out of the intermediate code however.
    int i = blockDim.x * blockIdx.x + threadIdx.x;
}

// Test for arithmetic (modal processing)
// NOTE: This is not the version used in the paper at this point; it's in a separate kernel file.
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

// Simulated quasi-granular synthesis kernel.
// N virtual playheads read from areas of memory.
// This is intended to exerise some of the "unfortunate" cases:
//   - random memory access
//   - unalgned memory access between threads within a warp
//   - non-coalesced memory access between threads in a warp
// For simplicity we do not wrap around the end of the buffer inside the kernels here.
// We may wish to change this as it would disrupt caching behavior a bit further; there
// would be a point where a new cache line would be loaded.
__global__ void RndMemKernel(const float* sampleMem, const int* playheads, float* outBuf) {
    int trackidx = blockDim.x * blockIdx.x + threadIdx.x;
	int playhead = playheads[trackidx];
    for (int i = 0; i < BUFSIZE; i++) {
        // See comments in Mac impl.
        // (CLEANUP: copy the text here)
        outBuf[trackidx * BUFSIZE + i] = sampleMem[playhead] + i;
	}
}

/**
 * Host main routine
 */
int main(void) {
  // Error code to check return values for CUDA calls
  cudaError_t err = cudaSuccess;

  // Print the vector length to be used, and compute its size
  size_t size = numElements * sizeof(float);
  printf("[Vector addition of %d elements]\n", numElements);

#if DO_BENCHMARK_IO
 
  float* h_inBuf = (float *)malloc(IOTEST_INBUFCOUNT * sizeof(float));
  float* h_outBuf = (float*)malloc(IOTEST_OUTBUFCOUNT * sizeof(float));
  
  float* d_inBuf = NULL;
  err = cudaMalloc((void**)&d_inBuf, IOTEST_INBUFCOUNT*sizeof(float));
if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  float *d_outBuf = NULL;
  err = cudaMalloc((void **)&d_outBuf, IOTEST_OUTBUFCOUNT*sizeof(float));
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  // Init host vector with random floats
  for (int i = 0; i < IOTEST_INBUFCOUNT; i++) {
    h_inBuf[i] = rand() / (float)RAND_MAX;
  }
#endif
#if DO_BENCHMARK_RNDMEM

  constexpr int kSampleMemNumElems = 512 * 1024 * 1024 / sizeof(float);
  float *h_sampleMem = (float *)malloc(kSampleMemNumElems*sizeof(float));
  float *d_sampleMem = NULL;
  // Init sample buffer with random floats
  for (int i = 0; i < kSampleMemNumElems; i++) {
      h_sampleMem[i] = rand() / (float)RAND_MAX;
  }
  // Sample memory to device
  // (TODO: use constant memory here)
  err = cudaMalloc((void**)&d_sampleMem, kSampleMemNumElems * sizeof(float));
  if (err != cudaSuccess) {
      fprintf(stderr, "Failed to allocate device vector samplemem (error code %s)!\n",
          cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }
  err = cudaMemcpy(d_sampleMem, h_sampleMem, kSampleMemNumElems * sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
      fprintf(stderr,
          "Failed to copy sample memory to device!\n",
          cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }


  int* h_playheads = (int*)malloc(NTRACKS * sizeof(int));
  int* d_playheads = NULL;
  err = cudaMalloc((void**)&d_playheads, NTRACKS * sizeof(int));
  if (err != cudaSuccess) {
      fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
          cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  float playheadsStart[NTRACKS];
  float playheadsEnd[NTRACKS];
  int minLoopLen = 1000;
  int maxLoopLen = 48000;
 
  int samplebufferEnd = kSampleMemNumElems - BUFSIZE;

  for (int i = 0; i < NTRACKS; i++) {
      playheadsStart[i] = rand() % samplebufferEnd;
      int loopLen = minLoopLen + (rand() % (maxLoopLen - minLoopLen));
      playheadsEnd[i] = playheadsStart[i] + loopLen;
      h_playheads[i] = playheadsStart[i];
  }
  
  float *h_out = (float*)malloc(NTRACKS * BUFSIZE * sizeof(float));
  float* d_out = NULL;
  err = cudaMalloc((void**)&d_out, NTRACKS * BUFSIZE * sizeof(float));
  if (err != cudaSuccess) {
      fprintf(stderr, "Failed to allocate ouput buffer on device (error code %s)!\n",
          cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }
#endif
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

  float* h_inBuf = (float*)malloc(kNumModes*kNumModeParams*sizeof(float));
  float* h_outBuf = (float*)malloc(BUFSIZE*kModalOutputTracks*sizeof(float));

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
      printf("Running modal benchmark\n");
  for (int i = 0; i < NRUNS; i++) {
      auto start = std::chrono::high_resolution_clock::now();
      
      err = cudaMemcpy(d_inBuf, h_inBuf, kNumModes*kNumModeParams*sizeof(float), cudaMemcpyHostToDevice);
      if (err != cudaSuccess) {
          fprintf(stderr,
              "Failed to copy vector input from host to device (error code %s)!\n",
              cudaGetErrorString(err));
          exit(EXIT_FAILURE);
      }

      // Launch the CUDA Kernel
      constexpr int threadsPerBlock = 256;
      constexpr int numElements_iotest = kNumModes;
      int blocksPerGrid = (kNumModes + threadsPerBlock - 1) / threadsPerBlock;
      ModalKernel << <blocksPerGrid, threadsPerBlock >> > (d_inBuf, d_outBuf, kNumModes);

      err = cudaGetLastError();
      if (err != cudaSuccess) {
          fprintf(stderr, "Failed to launch kernel(error code %s)!\n",
              cudaGetErrorString(err));
          exit(EXIT_FAILURE);
      }

      // Copy result back out
      err = cudaMemcpy(h_outBuf, d_outBuf, kModalOutputTracks*BUFSIZE*sizeof(float), cudaMemcpyDeviceToHost);

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

#if DO_BENCHMARK_RNDMEM
  printf("Running RndMemN benchmark\n");
  for (int i = 0; i < NRUNS; i++) {
      // Avoiding printfs during the loop.
      // Cleanup: remove altogether.
      // printf("Copy input data from the host memory to the CUDA device\n");
      auto start = std::chrono::high_resolution_clock::now();

      // Playheads copy
      err = cudaMemcpy(d_playheads, h_playheads, NTRACKS * sizeof(int), cudaMemcpyHostToDevice);
      if (err != cudaSuccess) {
          fprintf(stderr,
              "Failed to copy playhead indices to device (error code %s)!\n",
              cudaGetErrorString(err));
          exit(EXIT_FAILURE);
      }

      // Launch the CUDA Kernel
      constexpr int threadsPerBlock = 32;
      constexpr int numElements = NTRACKS;
      int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
      RndMemKernel << <blocksPerGrid, threadsPerBlock >> > (d_sampleMem, d_playheads, d_out);
      err = cudaGetLastError();
      if (err != cudaSuccess) {
          fprintf(stderr, "Failed to launch kernel(error code %s)!\n",
              cudaGetErrorString(err));
          exit(EXIT_FAILURE);
      }

      // Copy result back out
      err = cudaMemcpy(h_out, d_out, NTRACKS * BUFSIZE * sizeof(float), cudaMemcpyDeviceToHost);

      if (err != cudaSuccess) {
          fprintf(stderr,
              "Failed to copy vector C from device to host (error code %s)!\n",
              cudaGetErrorString(err));
          exit(EXIT_FAILURE);
      }
      // TODO: validate (other benchmarks -- or this one if we want to test data transfer internally).

      // Update playneads
      for (int i = 0; i < NTRACKS; i++) {
		  h_playheads[i]+=BUFSIZE;
		  if (h_playheads[i] >= playheadsEnd[i]) {
			  h_playheads[i] = playheadsStart[i];
		  }
	  }

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

  // Free host and device global memory. Not error checking return codes since we're writing data and exit.
  cudaFree(d_playheads); cudaFree(d_out); cudaFree(d_sampleMem);
  free(h_playheads); free(h_out);
#endif

  printf("Done\n");
  return 0;
}
