#pragma once

// Frequency-domain Conv1D benchmark driven by cuFFT.

#include "bench_base.cuh"
#include <cufft.h>

class Conv1DAccelBenchmark : public GPUABenchmark {
public:
    // Configuration
    static const int DEFAULT_IR_LEN = 512;
    static const int DEFAULT_FFT_SIZE = 1024;

    // ============================================================================
    // Constructor/Destructor
    // ============================================================================

    Conv1DAccelBenchmark(int ir_length = DEFAULT_IR_LEN, size_t buffer_size = BUFSIZE, size_t track_count = NTRACKS);
    ~Conv1DAccelBenchmark();

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

    void allocateAccelBuffers();
    void setupFFTPlans();
    void generateImpulseResponses();
    void precomputeImpulseResponseFFTs();
    void calculateCPUReference();
    void cleanupAccelBuffers();
    void cleanupFFTPlans();

    // CPU reference convolution implementation
    void conv1DCPUReference(const float* input, const float* impulse_response, float* output,
                           int ir_len, int buffer_size, int track_count);

    // ============================================================================
    // Private Member Variables
    // ============================================================================

    int ir_length_;
    int fft_size_;
    int overlap_size_;

    // Convolution-specific buffers
    float* h_ir_buf = nullptr;          // Host impulse response buffer
    float* d_ir_buf = nullptr;          // Device impulse response buffer

    // FFT buffers
    cufftComplex* d_fft_input = nullptr;    // Device FFT input buffer
    cufftComplex* d_fft_output = nullptr;   // Device FFT output buffer
    cufftComplex* d_ir_fft = nullptr;       // Pre-computed IR FFTs

    // cuFFT plans
    cufftHandle forward_plan = 0;
    cufftHandle inverse_plan = 0;

    // CPU reference buffers
    float* cpu_reference = nullptr;

    // Buffer size calculations
    size_t ir_buffer_size;
    size_t ir_buffer_bytes;
    size_t fft_buffer_size;
    size_t fft_buffer_bytes;
};

// ============================================================================
// CUDA Kernel Declarations
// ============================================================================

__global__ void ComplexMultiplyKernel(
    cufftComplex* input_fft,
    const cufftComplex* ir_fft,
    cufftComplex* output_fft,
    int fft_size,
    int num_tracks);

__global__ void ExtractRealPartKernel(
    const cufftComplex* fft_output,
    float* real_output,
    int buffer_size,
    int num_tracks);

// ============================================================================
// Legacy Interface Wrapper
// ============================================================================

void RunConv1DAccelBenchmark();
