#pragma once

// Gain benchmark with per-track statistics.

#include "bench_base.cuh"

class GainStatsBenchmark : public GPUABenchmark {
public:
    // Statistics computed per track
    static const int NSTATS = 2;  // mean, max
    // ============================================================================
    // Constructor/Destructor
    // ============================================================================

    GainStatsBenchmark(size_t buffer_size = BUFSIZE, size_t track_count = NTRACKS);
    ~GainStatsBenchmark();

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

    void allocateStatsBuffers();
    void calculateCPUReference();
    void cleanupStatsBuffers();

    // ============================================================================
    // Private Member Variables
    // ============================================================================

    // Additional buffers for statistics
    float* h_stats = nullptr;
    float* d_stats = nullptr;

    // CPU reference buffers
    float* cpu_reference = nullptr;
    float* cpu_stats_reference = nullptr;

    size_t stats_count;
    size_t stats_size_bytes;
};

// ============================================================================
// CUDA Kernel Declaration
// ============================================================================

__global__ void GainStatsKernel(const float* bufIn, float* bufOut, float* stats,
                                BenchmarkUtils::BenchmarkParams params);
