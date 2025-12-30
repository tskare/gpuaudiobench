#include "bench_rndmem.cuh"
#include "thread_config.cuh"
#include <cstring>
#include <cmath>

// Simulated quasi-granular synthesis kernel.
__global__ void RndMemKernel(const float* sampleMem, const int* playheads, float* outBuf,
                             BenchmarkUtils::BenchmarkParams params) {
    int trackidx = blockDim.x * blockIdx.x + threadIdx.x;
    const int trackCount = static_cast<int>(params.trackCount);
    const int bufferSize = static_cast<int>(params.bufferSize);

    if (trackidx >= trackCount) return;

    int playhead = playheads[trackidx];

    for (int i = 0; i < bufferSize; i++) {
        outBuf[trackCount * i + trackidx] = sampleMem[playhead + i];
    }
}

RndMemBenchmark::RndMemBenchmark(size_t buffer_size, size_t track_count,
                                 int min_loop_len, int max_loop_len)
    : GPUABenchmark("RndMem", buffer_size, track_count),
      min_loop_length_(min_loop_len), max_loop_length_(max_loop_len) {

    sample_memory_bytes = SAMPLE_MEM_NUM_ELEMS * sizeof(float);
    playheads_bytes = track_count * sizeof(int);
    output_buffer_bytes = track_count * buffer_size * sizeof(float);
    sample_buffer_end_ = SAMPLE_MEM_NUM_ELEMS - static_cast<int>(buffer_size);
}

RndMemBenchmark::~RndMemBenchmark() {
    cleanupRndMemBuffers();
    BenchmarkUtils::freeHostBuffers({cpu_reference, playheads_start, playheads_end});
    cpu_reference = nullptr;
    playheads_start = nullptr;
    playheads_end = nullptr;
}

void RndMemBenchmark::setupBenchmark() {
    allocateRndMemBuffers();

    initializeSampleMemory();

    initializePlayheads();

    cpu_reference = BenchmarkUtils::allocateHostBuffer<float>(
        getTrackCount() * getBufferSize(), "rndmem cpu reference");

    calculateCPUReference();

    printf("RndMem benchmark setup complete (512MB sample memory, %zu tracks, random access)\n",
           getTrackCount());
}

void RndMemBenchmark::runKernel() {
    CUDA_CHECK(cudaMemcpy(d_playheads, h_playheads, playheads_bytes, cudaMemcpyHostToDevice));

    auto [blocks_per_grid, threads_per_block] = calculateGridDimensions(ThreadConfig::DEFAULT_BLOCK_SIZE_1D);

    const auto params = makeBenchmarkParams();
    RndMemKernel<<<blocks_per_grid, threads_per_block>>>(
        d_sample_memory, d_playheads, d_output_buffer, params
    );

    synchronizeAndCheck();

    CUDA_CHECK(cudaMemcpy(h_output_buffer, d_output_buffer, output_buffer_bytes, cudaMemcpyDeviceToHost));

    updatePlayheads();
}

void RndMemBenchmark::performBenchmarkIteration() {
    CUDA_CHECK(cudaMemcpy(d_playheads, h_playheads, playheads_bytes, cudaMemcpyHostToDevice));

    auto [blocks_per_grid, threads_per_block] = calculateGridDimensions(ThreadConfig::DEFAULT_BLOCK_SIZE_1D);

    const auto params = makeBenchmarkParams();
    RndMemKernel<<<blocks_per_grid, threads_per_block>>>(
        d_sample_memory, d_playheads, d_output_buffer, params
    );

    synchronizeAndCheck();

    CUDA_CHECK(cudaMemcpy(h_output_buffer, d_output_buffer, output_buffer_bytes, cudaMemcpyDeviceToHost));

    updatePlayheads();
}

void RndMemBenchmark::validate(ValidationData& validation_data) {
    float max_error = 0.0f;
    float mean_error = 0.0f;
    size_t output_elements = getBufferSize() * getTrackCount();

    for (size_t i = 0; i < output_elements; ++i) {
        float error = std::abs(h_output_buffer[i] - cpu_reference[i]);
        max_error = std::max(max_error, error);
        mean_error += error;
    }
    mean_error /= output_elements;

    validation_data.max_error = max_error;
    validation_data.mean_error = mean_error;

    if (max_error > 1e-6f) {
        validation_data.status = ValidationStatus::FAILURE;
        validation_data.messages.push_back("RndMem validation failed");
    } else {
        validation_data.status = ValidationStatus::SUCCESS;
        validation_data.messages.push_back("RndMem validation passed (memory access patterns verified)");
    }
}

void RndMemBenchmark::allocateRndMemBuffers() {
    h_sample_memory = BenchmarkUtils::allocateHostBuffer<float>(
        SAMPLE_MEM_NUM_ELEMS, benchmark_name_ + " host sample memory");
    d_sample_memory = BenchmarkUtils::allocateDeviceBuffer<float>(
        SAMPLE_MEM_NUM_ELEMS, benchmark_name_ + " device sample memory");

    h_playheads = BenchmarkUtils::allocateHostBuffer<int>(
        getTrackCount(), benchmark_name_ + " host playheads");
    d_playheads = BenchmarkUtils::allocateDeviceBuffer<int>(
        getTrackCount(), benchmark_name_ + " device playheads");

    h_output_buffer = BenchmarkUtils::allocateHostBuffer<float>(
        getBufferSize() * getTrackCount(), benchmark_name_ + " host output buffer");
    d_output_buffer = BenchmarkUtils::allocateDeviceBuffer<float>(
        getBufferSize() * getTrackCount(), benchmark_name_ + " device output buffer");

    playheads_start = BenchmarkUtils::allocateHostBuffer<float>(
        getTrackCount(), "rndmem playheads start");
    playheads_end = BenchmarkUtils::allocateHostBuffer<float>(
        getTrackCount(), "rndmem playheads end");

    memset(h_output_buffer, 0, output_buffer_bytes);
    CUDA_CHECK(cudaMemset(d_output_buffer, 0, output_buffer_bytes));
}

void RndMemBenchmark::initializeSampleMemory() {
    srand(42);
    for (int i = 0; i < SAMPLE_MEM_NUM_ELEMS; ++i) {
        h_sample_memory[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    printf("Transferring 512MB sample memory to device...\n");
    CUDA_CHECK(cudaMemcpy(d_sample_memory, h_sample_memory, sample_memory_bytes, cudaMemcpyHostToDevice));
    printf("Sample memory transfer complete.\n");
}

void RndMemBenchmark::initializePlayheads() {
    srand(42);  // Fixed seed for reproducible results

    // Initialize each track with random loop parameters
    for (size_t i = 0; i < getTrackCount(); ++i) {
        // Random start position within sample buffer
        playheads_start[i] = static_cast<float>(rand() % sample_buffer_end_);

        // Random loop length within bounds
        int loop_len = min_loop_length_ + (rand() % (max_loop_length_ - min_loop_length_));
        playheads_end[i] = playheads_start[i] + loop_len;

        // Ensure we don't exceed buffer bounds
        if (playheads_end[i] >= sample_buffer_end_) {
            playheads_end[i] = sample_buffer_end_ - 1;
        }

        // Set initial playhead position
        h_playheads[i] = static_cast<int>(playheads_start[i]);
    }

    printf("Initialized %zu tracks with random loop lengths (%d-%d samples)\n",
           getTrackCount(), min_loop_length_, max_loop_length_);
}

void RndMemBenchmark::updatePlayheads() {
    // Update playheads for next iteration (simulate granular synthesis advancement)
    for (size_t i = 0; i < getTrackCount(); ++i) {
        h_playheads[i] += static_cast<int>(getBufferSize());

        // Wrap around if we've reached the end of the loop
        if (h_playheads[i] >= static_cast<int>(playheads_end[i])) {
            h_playheads[i] = static_cast<int>(playheads_start[i]);
        }
    }
}

void RndMemBenchmark::calculateCPUReference() {
    // Run CPU reference implementation
    rndMemCPUReference(h_sample_memory, h_playheads, cpu_reference,
                      static_cast<int>(getBufferSize()), static_cast<int>(getTrackCount()));
}

void RndMemBenchmark::rndMemCPUReference(const float* sample_memory, const int* playheads,
                                        float* output_buffer, int buffer_size, int track_count) {
    // CPU implementation of the same random memory access pattern
    for (int trackidx = 0; trackidx < track_count; ++trackidx) {
        int playhead = playheads[trackidx];

        for (int i = 0; i < buffer_size; ++i) {
            // Same interleaved output pattern as GPU kernel
            output_buffer[track_count * i + trackidx] = sample_memory[playhead + i];
        }
    }
}

void RndMemBenchmark::cleanupRndMemBuffers() {
    BenchmarkUtils::freeHostBuffers({h_sample_memory, reinterpret_cast<float*>(h_playheads), h_output_buffer});
    BenchmarkUtils::freeDeviceBuffers({d_sample_memory, reinterpret_cast<float*>(d_playheads), d_output_buffer});

    h_sample_memory = nullptr;
    d_sample_memory = nullptr;
    h_playheads = nullptr;
    d_playheads = nullptr;
    h_output_buffer = nullptr;
    d_output_buffer = nullptr;
}
