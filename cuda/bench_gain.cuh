#pragma once

#include "bench_base.cuh"

// Forward declarations
struct IIRCoefficients;

class GainBenchmark : public GPUABenchmark {
private:
    float* cpu_reference = nullptr;  // For validation
    bool enable_validation_;

public:
    explicit GainBenchmark(size_t buffer_size = BUFSIZE, size_t track_count = NTRACKS, bool enable_validation = true);
    ~GainBenchmark() override;

    // ============================================================================
    // Required Implementation from GPUABenchmark
    // ============================================================================

    void setupBenchmark() override;
    void runKernel() override;

    void performBenchmarkIteration() override;
    void validate(ValidationData& validation_data) override;

private:
    // ============================================================================
    // Helper Methods
    // ============================================================================

    void calculateCPUReference();

    void cleanupCPUReference();
};
