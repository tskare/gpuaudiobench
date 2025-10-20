#pragma once

// No-op benchmark measuring kernel launch overhead.

#include "bench_base.cuh"

class NoOpBenchmark : public GPUABenchmark {
public:
    // ============================================================================
    // Constructor/Destructor
    // ============================================================================

    NoOpBenchmark(size_t buffer_size = BUFSIZE, size_t track_count = NTRACKS);
    ~NoOpBenchmark() override;

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

    void calculateCPUReference();

    // ============================================================================
    // Private Member Variables
    // ============================================================================

    float* cpu_reference = nullptr;
};

// ============================================================================
// CUDA Kernel Declaration
// ============================================================================

__global__ void NoOpKernel(const float* bufIn, float* bufOut, int totalElements);
