#pragma once

// Biquad IIR filter benchmark using Direct Form II structure.

#include "bench_base.cuh"

// IIR biquad filter coefficients structure
struct IIRCoefficients {
    float b0, b1, b2;  // Numerator coefficients
    float a1, a2;      // Denominator coefficients (a0 = 1.0)
};

class IIRBenchmark : public GPUABenchmark {
public:
    // Filter configuration
    static const int STATES_PER_TRACK = 2;  // z1, z2 for biquad

    // ============================================================================
    // Constructor/Destructor
    // ============================================================================

    IIRBenchmark(size_t buffer_size = BUFSIZE, size_t track_count = NTRACKS);
    ~IIRBenchmark();

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

    void allocateIIRBuffers();
    void initializeCoefficients();
    void calculateCPUReference();
    void cleanupIIRBuffers();

    // CPU reference IIR implementation
    void iirFilterCPUReference(const float* input, float* output, int size,
                              const IIRCoefficients* coeffs, float* state,
                              int track_count, int buffer_size);

    // Coefficient calculation helper
    IIRCoefficients calculateButterworthCoefficients(float normalized_frequency);

    // ============================================================================
    // Private Member Variables
    // ============================================================================

    // IIR-specific buffers
    IIRCoefficients* h_coeffs = nullptr;
    IIRCoefficients* d_coeffs = nullptr;
    float* h_state = nullptr;
    float* d_state = nullptr;

    // CPU reference buffers
    float* cpu_reference = nullptr;
    float* cpu_state_reference = nullptr;

    // Buffer size calculations
    size_t state_count;
    size_t state_size_bytes;
};

// ============================================================================
// CUDA Kernel Declaration
// ============================================================================

__global__ void IIRFilterKernel(const float* bufIn, float* bufOut,
                                const IIRCoefficients* coeffs,
                                float* state,
                                int trackCount, int bufferSize);
