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
	for (int i = 0; i < BUFSIZE; i++) {
		float samp = bufIn[trackidx * BUFSIZE + i];
		mean += samp;
		bufOut[trackidx * BUFSIZE + i] = samp * 0.5f;
		// Max
		if (samp > stats[trackidx * (NTRACKS * nstats) + 0]) {
			stats[trackidx * (NTRACKS * nstats) + 0] = samp;
		}
	}
	mean /= BUFSIZE;
}

const char* benchmarkNames[] = {
	"RndMem",
	"GainStats",
	"Modal",
	"IO",
	"Conv1D"
};

static void printHelp() {
	printf("Usage: gpu-audio-benchmark [options]\n");
	printf("Options:\n");
	printf("  --help: Print this help message\n");
	printf("  --benchmark [X]: Run benchmark |benchmarkName|\n");
	printf("  --fs: set sampling rate\n");
	printf("  --bufferSize: set buffer size\n");
	printf("  --ntracks: set number of tracks\n");
	printf("\n");
	printf("Available benchmarks:\n");
	for (int i = 0; i < sizeof(benchmarkNames) / sizeof(benchmarkNames[0]); i++) {
		printf("  %s\n", benchmarkNames[i]);
	}
}

/**
 * Host main routine
 */
 // main() with argc and argv:
int main(int argc, char** argv) {
	printf("GPGPU Audio Benchmark\n");

	std::string whichBenchmark = "RndMem";

	// Skip first argument, the executable name.
	for (int i = 1; i < argc; i++) {
		bool hasNextParameter = i + 1 < argc;
		// printf("argv[%d]: %s\n", i, argv[i]);
		if (strcmp(argv[i], "--help") == 0) {
			printHelp();
			return 0;
		}
		if (strcmp(argv[i], "--benchmark") == 0) {
			if (!hasNextParameter) {
				printf("Error: --benchmark requires an argument\n");
				return 1;
			}
			// Arg validity is checked in init and run sections.
			whichBenchmark = std::string(argv[i + 1]);
			i++;
		} else if (strcmp(argv[i], "--fs") == 0) {
			if (!hasNextParameter) {
				printf("Error: --fs requires an argument\n");
				return 1;
			}
			FS = atoi(argv[i + 1]);
			i++;
		} else if (strcmp(argv[i], "--bufferSize") == 0) {
			if (!hasNextParameter) {
				printf("Error: --bufferSize requires an argument\n");
				return 1;
			}
			printf("Warning: currently transitioning from compile-time to runtime buffer size\n");
			//BUFSIZE = atoi(argv[i + 1]);
			i++;
		} else if (strcmp(argv[i], "--nTracks") == 0) {
			if (!hasNextParameter) {
				printf("Error: --nTracks requires an argument\n");
				return 1;
			}
			// NTRACKS = atoi(argv[i + 1]);
			printf("Warning: currently transitioning from compile-time to runtime nTracks\n");
			i++;
		} else if (strcmp(argv[i], "--nRuns") == 0) {
			if (!hasNextParameter) {
				printf("Error: --nRuns requires an argument\n");
				return 1;
			}
			NRUNS = atoi(argv[i + 1]);
			i++;
		}
		else {
			printf("Warning: Unparsed argument: %s\n", argv[i]);
		}
	}

	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	// Intermediate data:
	// Moving from compile-time to runtime benchmark selection.
	// 

	// Buffer names, may be shared between benchmarks.
	// Individual benchmarks should allocate and free their own buffers.
	// Naming scheme: prefixed h_ for host, d_ for device.
	// Note CUDA provides shared memory, but we're using the more direct
	// and manual approach here, as the transfers are explicit.
	float* h_inBuf = nullptr;
	float* h_outBuf = nullptr;
	float* d_inBuf = nullptr;
	float* d_outBuf = nullptr;
	float* h_sampleMem = nullptr;
	float* d_sampleMem = nullptr;
	int* h_playheads = nullptr;
	int* d_playheads = nullptr;
	float* h_stats = nullptr;
	float* d_stats = nullptr;

	// Following is local data that should be moved to a data struct for
	// each corresponding benchmark:
	// RndMem
	float playheadsStart[NTRACKS];
	float playheadsEnd[NTRACKS];
	int minLoopLen = 1000;
	int maxLoopLen = 48000;
	int samplebufferEnd = kSampleMemNumElems - BUFSIZE;

	if (whichBenchmark == "IO") {
		SetupBenchmarkIO(&h_inBuf, &h_outBuf, &d_inBuf, &d_outBuf, IOTEST_INBUFCOUNT, IOTEST_OUTBUFCOUNT);
	}
	else if (whichBenchmark == "RndMem") {
		SetupBenchmarkRndMem(&h_sampleMem, &d_sampleMem,
			&h_playheads, &d_playheads,
			playheadsStart,
			playheadsEnd,
			minLoopLen,
			maxLoopLen,
			samplebufferEnd,
			&h_outBuf, &d_outBuf);
	}
	else if (whichBenchmark == "GainStats") {
		// TODO: Finish copying this out into its own file. Was here for debugging.
		h_inBuf = (float*)malloc(NTRACKS * BUFSIZE * sizeof(float));
		h_outBuf = (float*)malloc(NTRACKS * BUFSIZE * sizeof(float));
		constexpr int nstats = 4;
		d_stats = (float*)malloc(NTRACKS * nstats * sizeof(float));
		h_stats = (float*)malloc(NTRACKS * nstats * sizeof(float));

		err = cudaMalloc((void**)&d_inBuf, NTRACKS * BUFSIZE * sizeof(float));
		if (err != cudaSuccess) {
			fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
				cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
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
	}
	else if (whichBenchmark == "Modal") {
		SetupBenchmarkModal(&h_inBuf, &h_outBuf, &d_inBuf, &d_outBuf);
	}
	else if (whichBenchmark == "Conv1D") {
		// TODO: merge from dev
		// SetupConv1D(&h_inBuf, &h_outBuf, &d_inBuf, &d_outBuf, &h_irbuf, &d_irbuf);
	}
	else {
		printf("Unknown benchmark (init section): %s\n", whichBenchmark.c_str());
		return 1;
	}

	// Main benchmark loop
	std::vector<float> latencies;

	if (whichBenchmark == "IO") {
		printf("Running I/O benchmark\n");
		for (int i = 0; i < NRUNS; i++) {
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
			constexpr int numElements_iotest = threadsPerBlock * 100;
			int blocksPerGrid = (numElements_iotest + threadsPerBlock - 1) / threadsPerBlock;
			DataTransferBenchmarkKernel << <blocksPerGrid, threadsPerBlock >> > (d_inBuf, d_outBuf, numElements_iotest);

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

			auto end = std::chrono::high_resolution_clock::now();
			// Compute the duration in milliseconds
			std::chrono::duration<float, std::milli> duration = end - start;
			latencies.push_back(duration.count());
			cout << "Duration: " << duration.count() << "ms" << endl;
			if (ENABLE_DAWSIM_SLEEP) {
				// Sleep for buffer interarrival time minus duration.
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

		// Cleanup
		cudaFree(d_inBuf); cudaFree(d_outBuf);
		free(h_inBuf); free(h_outBuf);
	}
	else if (whichBenchmark == "Gain") {
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
			err = cudaMemcpy(h_outBuf, d_outBuf, NTRACKS * BUFSIZE * sizeof(float), cudaMemcpyDeviceToHost);

			if (err != cudaSuccess) {
				fprintf(stderr,
					"Failed to copy vector C from device to host (error code %s)!\n",
					cudaGetErrorString(err));
				exit(EXIT_FAILURE); // TODO and stats
			}
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
	}
	else if (whichBenchmark == "Modal") {
		RunBenchmarkModal(&d_inBuf, &h_inBuf, &d_outBuf, &h_outBuf, kNumModes, kNumModeParams, latencies);
	}
	else if (whichBenchmark == "RndMem") {
		RunBenchmarkRndMem(&d_playheads, &h_playheads, &d_sampleMem, &d_outBuf, &h_outBuf,
			latencies,
			playheadsStart, playheadsEnd);
	}
	else {
		printf("Unknown benchmark (run section): %s\n", whichBenchmark.c_str());
		return 1;
	}

	printf("Done\n");
	return 0;
}
