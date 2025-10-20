#pragma once

// 1D real-to-complex FFT benchmark using cuFFT.

#include "bench_base.cuh"
#include <cufft.h>

class FFTBenchmark : public GPUABenchmark {
public:
    // FFT configuration
    static const int FFT_SIZE = 1024;

    // ============================================================================
    // Constructor/Destructor
    // ============================================================================

    FFTBenchmark(size_t buffer_size = BUFSIZE, size_t track_count = NTRACKS);
    ~FFTBenchmark();

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

    void createFFTPlan();
    void destroyFFTPlan();
    void allocateFFTBuffers();
    void calculateCPUReference();
    void cleanupFFTBuffers();

    // CPU reference FFT implementation
    void cpuFFTReference(const float* input, float* real_output, float* imag_output, int size);

    // ============================================================================
    // Private Member Variables
    // ============================================================================

    // cuFFT plan handle
    cufftHandle fft_plan;
    bool plan_created = false;

    // FFT-specific buffers (different sizes from standard buffers)
    float* h_input_fft = nullptr;    // Real input data
    cufftComplex* h_output_fft = nullptr;  // Complex output data
    float* d_input_fft = nullptr;    // Device real input
    cufftComplex* d_output_fft = nullptr;  // Device complex output

    // CPU reference buffers
    float* cpu_reference_real = nullptr;
    float* cpu_reference_imag = nullptr;

    // Buffer size calculations
    size_t input_fft_size;           // Number of real input elements
    size_t output_fft_size;          // Number of complex output elements
    size_t input_fft_bytes;          // Input buffer size in bytes
    size_t output_fft_bytes;         // Output buffer size in bytes
};
