#include "benchmark_rndmem.cuh"

#include "globals.cuh"

#include <cuda_runtime.h>
#include <helper_cuda.h>

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

        // Interleave samples so this access is aligned.
        outBuf[NTRACKS*i + trackidx] = sampleMem[playhead] + i;
    }
}

void SetupBenchmarkRndMem(float **h_sampleMem, float** d_sampleMem,
int **h_playheads, int **d_playheads,
    float playheadsStart[],
    float playheadsEnd[],
    int minLoopLen,
    int maxLoopLen,
    int samplebufferEnd,
    float **h_out, float **d_out
) {
    cudaError_t err = cudaSuccess;
    *h_sampleMem = (float*)malloc(kSampleMemNumElems * sizeof(float));
    // Init sample buffer with random floats
    for (int i = 0; i < kSampleMemNumElems; i++) {
        (*h_sampleMem)[i] = rand() / (float)RAND_MAX;
    }
    // Sample memory to device
    err = cudaMalloc((void**)d_sampleMem, kSampleMemNumElems * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector samplemem (error code %s)!\n",
            cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(*d_sampleMem, *h_sampleMem, kSampleMemNumElems * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr,
            "Failed to copy sample memory to device!\n",
            cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    *h_playheads = (int*)malloc(NTRACKS * sizeof(int));
    *d_playheads = NULL;
    
    err = cudaMalloc((void**)d_playheads, NTRACKS * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector playheads (error code %s)!\n",
            cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < NTRACKS; i++) {
        playheadsStart[i] = rand() % samplebufferEnd;
        int loopLen = minLoopLen + (rand() % (maxLoopLen - minLoopLen));
        (playheadsEnd)[i] = playheadsStart[i] + loopLen;
        (*h_playheads)[i] = playheadsStart[i];
    }

    *h_out = (float*)malloc(NTRACKS * BUFSIZE * sizeof(float));
    err = cudaMalloc((void**)d_out, NTRACKS * BUFSIZE * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate ouput buffer on device (error code %s)!\n",
            cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void RunBenchmarkRndMem(int** d_playheads, int** h_playheads, float** d_sampleMem, float** d_out, float** h_out, vector<float>& latencies,
    float* playheadsStart, float* playheadsEnd) {
    printf("Running RndMemN benchmark\n");
    cudaError_t err = cudaSuccess;

    for (int i = 0; i < NRUNS; i++) {
        // Avoiding printfs during the loop.
        // Cleanup: remove altogether.
        // printf("Copy input data from the host memory to the CUDA device\n");
        auto start = std::chrono::high_resolution_clock::now();

        // Playheads copy
        err = cudaMemcpy(*d_playheads, *h_playheads, NTRACKS * sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr,
                "Failed to copy playhead indices to device (Run %d) (error code %s)!\n",
                i,
                cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Launch the CUDA Kernel
        constexpr int threadsPerBlock = 32;
        constexpr int numElements = NTRACKS;
        int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
        RndMemKernel << <blocksPerGrid, threadsPerBlock >> > (*d_sampleMem, *d_playheads, *d_out);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to launch kernel(error code %s)!\n",
                cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Copy result back out.
        err = cudaMemcpy(*h_out, *d_out, NTRACKS * BUFSIZE * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr,
                "Failed to copy output vector from device to host (Run %d) (error code %s)!\n",
                i,
                cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        // TODO: validate by synthesizing on host as well.
        // (This benchmark was validated during development by looking at the memory access patterns)

        // Update playheads
        for (int i = 0; i < NTRACKS; i++) {
            (*h_playheads)[i] += BUFSIZE;
            if ((*h_playheads)[i] >= playheadsEnd[i]) {
                (*h_playheads)[i] = playheadsStart[i];
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
    cudaFree(*d_playheads); cudaFree(*d_out); cudaFree(*d_sampleMem);
    free(*h_playheads); free(*h_out);
}
