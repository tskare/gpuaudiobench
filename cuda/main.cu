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

#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include <helper_cuda.h>

#include "globals.cuh"

using std::cout;
using std::endl;

// Benchmark includes
#include "bench_noop.cuh"
#include "bench_gain.cuh"
#include "bench_gainstats.cuh"
#include "bench_datatransfer.cuh"
#include "bench_fft.cuh"
#include "bench_iir.cuh"
#include "bench_conv1d.cuh"
#include "bench_conv1d_accel.cuh"
#include "bench_modal.cuh"
#include "bench_dwg.cuh"
#include "bench_fdtd3d.cuh"
#include "bench_rndmem.cuh"

// See globals.cuh for cross-benchmark parameters.
// See individual .cuh files for benchmark-specific parameters.

using BenchmarkFactory = std::function<std::unique_ptr<GPUABenchmark>()>;

struct BenchmarkEntry {
    const char* name;
    BenchmarkFactory factory;
};

static const std::vector<BenchmarkEntry>& getBenchmarkRegistry() {
    static const std::vector<BenchmarkEntry> registry = {
        {"NoOp", [] { return std::make_unique<NoOpBenchmark>(); }},
        {"gain", [] { return std::make_unique<GainBenchmark>(); }},
        {"GainStats", [] { return std::make_unique<GainStatsBenchmark>(); }},
        {"datacopy0199", [] { return std::make_unique<DataTransferBenchmark>(0.01f, 0.99f); }},
        {"datacopy2080", [] { return std::make_unique<DataTransferBenchmark>(0.20f, 0.80f); }},
        {"datacopy5050", [] { return std::make_unique<DataTransferBenchmark>(0.50f, 0.50f); }},
        {"datacopy8020", [] { return std::make_unique<DataTransferBenchmark>(0.80f, 0.20f); }},
        {"datacopy9901", [] { return std::make_unique<DataTransferBenchmark>(0.99f, 0.01f); }},
        {"FFT1D", [] { return std::make_unique<FFTBenchmark>(); }},
        {"IIRFilter", [] { return std::make_unique<IIRBenchmark>(); }},
        {"Conv1D", [] { return std::make_unique<Conv1DBenchmark>(); }},
        {"Conv1D_accel", [] { return std::make_unique<Conv1DAccelBenchmark>(); }},
        {"ModalFilterBank", [] { return std::make_unique<ModalBenchmark>(); }},
        {"DWG1DNaive", [] { return std::make_unique<DWGBenchmark>(DWGBenchmark::Variant::NAIVE); }},
        {"DWG1DAccel", [] { return std::make_unique<DWGBenchmark>(DWGBenchmark::Variant::ACCELERATED); }},
        {"FDTD3D", [] { return std::make_unique<FDTD3DBenchmark>(); }},
        {"RndMemRead", [] { return std::make_unique<RndMemBenchmark>(); }}
    };
    return registry;
}

std::unique_ptr<GPUABenchmark> createBenchmark(const std::string& name) {
    const auto& registry = getBenchmarkRegistry();
    const auto it = std::find_if(
        registry.begin(), registry.end(),
        [&name](const BenchmarkEntry& entry) { return name == entry.name; });

    if (it == registry.end()) {
        return nullptr;
    }
    return it->factory();
}

void runSelectedBenchmark(std::unique_ptr<GPUABenchmark> benchmark,
                          const std::string& benchmarkName) {
    if (!benchmark) {
        printf("Unknown benchmark: %s\n", benchmarkName.c_str());
        return;
    }

    try {
        printf("Setting up %s benchmark...\n", benchmarkName.c_str());
        benchmark->setupBenchmark();

        printf("Running %s benchmark (%d iterations with %d warmup)...\n",
               benchmarkName.c_str(), NRUNS, 3);
        auto result = benchmark->runBenchmark(NRUNS, 3);

        printf("Validating %s benchmark results...\n", benchmarkName.c_str());
        GPUABenchmark::ValidationData validation;
        benchmark->validate(validation);

        if (validation.status != GPUABenchmark::ValidationStatus::SUCCESS) {
            printf("Validation failed for %s:\n", benchmarkName.c_str());
            for (const auto& msg : validation.messages) {
                printf("  %s\n", msg.c_str());
            }
        } else {
            printf("Validation passed for %s\n", benchmarkName.c_str());
        }

        // Print and save results using enhanced framework
        if (JSON_OUTPUT) {
            // Output JSON to stdout or file
            writeJSONResults(result.latencies, benchmarkName, OUTPUT_FILE);
        } else {
            benchmark->printResults(result);
            benchmark->writeResults(result);

            // Write CSV output if requested
            if (!OUTPUT_FILE.empty()) {
                writeCSVResults(result.latencies, benchmarkName, OUTPUT_FILE);
            }
        }

        printf("%s benchmark completed successfully!\n", benchmarkName.c_str());

    } catch (const std::exception& e) {
        printf("Benchmark %s failed: %s\n", benchmarkName.c_str(), e.what());
    }
}





static void printBenchmarkList() {
	printf("Available benchmarks:\n");
    for (const auto& entry : getBenchmarkRegistry()) {
        printf("%s\n", entry.name);
    }
}

static void printHelp() {
	printf("CUDA GPU Audio Benchmark Suite\n");
	printf("===============================\n");
	printf("Real-time GPGPU audio processing benchmarks\n\n");

	printf("Usage: gpubench [options]\n\n");

	printf("Options:\n");
	printf("  --help              Print this help message\n");
	printf("  --list              List all available benchmarks\n");
	printf("  --benchmark [name]  Run specific benchmark (see list below)\n");
	printf("  --fs [rate]         Set sampling rate (default: 48000)\n");
	printf("  --bufferSize [size] Set buffer size (default: 512)\n");
	printf("  --nTracks [count]   Set number of tracks (default: 128)\n");
	printf("  --nRuns [count]     Set number of iterations (default: 100)\n");
	printf("  --outputfile [file] Save results to CSV file\n");
	printf("  --json              Output results in JSON format\n");
	printf("\n");

	printf("Available Benchmarks:\n");
	printf("=====================\n");

	printf("\nData Transfer:\n");
	printf("  datacopy0199     - 99%% input, 1%% output transfer\n");
	printf("  datacopy2080     - 80%% input, 20%% output transfer\n");
	printf("  datacopy5050     - 50%% input, 50%% output transfer\n");
	printf("  datacopy8020     - 20%% input, 80%% output transfer\n");
	printf("  datacopy9901     - 1%% input, 99%% output transfer\n");

	printf("\nBasic Audio Processing:\n");
	printf("  NoOp             - No-operation baseline\n");
	printf("  gain             - Simple gain/volume control\n");
	printf("  GainStats        - Gain with statistical analysis\n");

	printf("\nDigital Signal Processing:\n");
	printf("  IIRFilter        - Infinite Impulse Response filter\n");
	printf("  Conv1D           - 1D convolution\n");
	printf("  Conv1D_accel     - Accelerated 1D convolution\n");
	printf("  ModalFilterBank  - Modal synthesis filter bank\n");
	printf("  FFT1D            - 1D Fast Fourier Transform\n");

	printf("\nPhysical Modeling:\n");
	printf("  DWG1DNaive       - 1D Digital Waveguide (naive)\n");
	printf("  DWG1DAccel       - 1D Digital Waveguide (accelerated)\n");
	printf("  FDTD3D           - 3D Finite Difference Time Domain\n");

	printf("\nMemory Performance:\n");
	printf("  RndMemRead       - Random memory access pattern\n");
	printf("\nExamples:\n");
	printf("  gpubench --benchmark gain\n");
	printf("  gpubench --benchmark IIRFilter --bufferSize 1024 --nTracks 128\n");
	printf("  gpubench --benchmark FFT1D --fs 44100 --nRuns 50\n");
	printf("\n");
}

/**
 * Host main routine
 */
 // main() with argc and argv:
int main(int argc, char** argv) {
	printf("GPGPU Audio Benchmark\n");

	std::string whichBenchmark = "RndMemRead";

	// Skip first argument, the executable name.
	for (int i = 1; i < argc; i++) {
		bool hasNextParameter = i + 1 < argc;
		// printf("argv[%d]: %s\n", i, argv[i]);
		if (strcmp(argv[i], "--help") == 0) {
			printHelp();
			return 0;
		} else if (strcmp(argv[i], "--list") == 0) {
			printBenchmarkList();
			return 0;
		} else if (strcmp(argv[i], "--json") == 0) {
			JSON_OUTPUT = true;
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
			BUFSIZE = atoi(argv[i + 1]);
			printf("Buffer size set to: %d\n", BUFSIZE);
			i++;
		} else if (strcmp(argv[i], "--nTracks") == 0) {
			if (!hasNextParameter) {
				printf("Error: --nTracks requires an argument\n");
				return 1;
			}
			NTRACKS = atoi(argv[i + 1]);
			printf("Number of tracks set to: %d\n", NTRACKS);
			i++;
		} else if (strcmp(argv[i], "--nRuns") == 0) {
			if (!hasNextParameter) {
				printf("Error: --nRuns requires an argument\n");
				return 1;
			}
			NRUNS = atoi(argv[i + 1]);
			i++;
		} else if (strcmp(argv[i], "--outputfile") == 0) {
			if (!hasNextParameter) {
				printf("Error: --outputfile requires an argument\n");
				return 1;
			}
			OUTPUT_FILE = std::string(argv[i + 1]);
			printf("Output file set to: %s\n", OUTPUT_FILE.c_str());
			i++;
		}
		else {
			printf("Warning: Unparsed argument: %s\n", argv[i]);
		}
	}

	// Initialize CUDA device
	cudaError_t err = cudaSuccess;
	int deviceCount;
	err = cudaGetDeviceCount(&deviceCount);
	if (err != cudaSuccess) {
		printf("Failed to get CUDA device count: %s\n", cudaGetErrorString(err));
		return 1;
	}
	printf("Found %d CUDA device(s)\n", deviceCount);

	if (auto benchmarkInstance = createBenchmark(whichBenchmark)) {
		printf("Running %s benchmark...\n", whichBenchmark.c_str());
		runSelectedBenchmark(std::move(benchmarkInstance), whichBenchmark);
		printf("Done\n");
		return 0;
	}

	// Benchmark not found - all benchmarks should be registered in createBenchmark()
	printf("Error: Unknown benchmark '%s'\n", whichBenchmark.c_str());
	printf("This benchmark is not registered. Check createBenchmark() implementation.\n");
	printf("Use --list to see available benchmarks.\n");
	return 1;
}
