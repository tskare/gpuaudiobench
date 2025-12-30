#pragma once

// Random memory access benchmark emulating granular synthesis patterns.

#include "bench_base.cuh"

class RndMemBenchmark : public GPUABenchmark {
public:
    // Configuration constants
    static constexpr int SAMPLE_MEM_NUM_ELEMS = 512 * 1024 * 1024 / sizeof(float); // 512MB
    static const int DEFAULT_MIN_LOOP_LEN = 1000;
    static const int DEFAULT_MAX_LOOP_LEN = 48000;

    // ============================================================================
    // Constructor/Destructor
    // ============================================================================

    RndMemBenchmark(size_t buffer_size = BUFSIZE, size_t track_count = NTRACKS,
                    int min_loop_len = DEFAULT_MIN_LOOP_LEN,
                    int max_loop_len = DEFAULT_MAX_LOOP_LEN);
    ~RndMemBenchmark();

    // ============================================================================
    // Required Implementation from GPUABenchmark
    // ============================================================================

    void setupBenchmark() override;
    void runKernel() override;
    void performBenchmarkIteration() override;
    void validate(ValidationData& validation_data) override;

private:
    // ============================================================================
    // Private Helper Methods
    // ============================================================================

    void allocateRndMemBuffers();
    void initializeSampleMemory();
    void initializePlayheads();
    void updatePlayheads();
    void calculateCPUReference();
    void cleanupRndMemBuffers();

    // CPU reference granular synthesis implementation
    void rndMemCPUReference(const float* sample_memory, const int* playheads,
                           float* output_buffer, int buffer_size, int track_count);

    // ============================================================================
    // Private Member Variables
    // ============================================================================

    int min_loop_length_;
    int max_loop_length_;
    int sample_buffer_end_;

    // Large sample memory buffer (512MB)
    float* h_sample_memory = nullptr;
    float* d_sample_memory = nullptr;

    // Playhead management
    int* h_playheads = nullptr;
    int* d_playheads = nullptr;
    float* playheads_start = nullptr;  // Loop start positions
    float* playheads_end = nullptr;    // Loop end positions

    // Output buffers (different layout from base class)
    float* h_output_buffer = nullptr;
    float* d_output_buffer = nullptr;

    // CPU reference buffer
    float* cpu_reference = nullptr;

    // Buffer size calculations
    size_t sample_memory_bytes;
    size_t playheads_bytes;
    size_t output_buffer_bytes;
};

// ============================================================================
// CUDA Kernel Declaration
// ============================================================================

__global__ void RndMemKernel(const float* sampleMem, const int* playheads, float* outBuf,
                             BenchmarkUtils::BenchmarkParams params);
