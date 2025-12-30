#pragma once

// Modal filter bank benchmark for large mode counts.

#include "bench_base.cuh"
#include <cuComplex.h>

class ModalBenchmark : public GPUABenchmark {
public:
    // Modal synthesis configuration
    static const int NUM_MODES = 1024 * 1024;  // Over 1 million modes
    static const int NUM_MODE_PARAMS = 8;      // Parameters per mode
    static const int MODAL_OUTPUT_TRACKS = 32; // Output track count

    // Mode parameter indices
    enum ModeParams {
        AMPLITUDE = 0,
        FREQUENCY = 1,
        PHASE = 2,
        STATE_REAL = 3,
        STATE_IMAG = 4,
        RESERVED1 = 5,  // Used for amplitude coupling
        RESERVED2 = 6,
        RESERVED3 = 7
    };

    // ============================================================================
    // Constructor/Destructor
    // ============================================================================

    ModalBenchmark();
    ~ModalBenchmark();

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

    void allocateModalBuffers();
    void initializeModeParameters();
    void calculateCPUReference();
    void cleanupModalBuffers();

    // CPU reference modal synthesis implementation
    void modalSynthesisCPUReference(const float* mode_params, float* output,
                                   int num_modes, int buffer_size, int output_tracks);

    // ============================================================================
    // Private Member Variables
    // ============================================================================

    // Modal-specific buffers (different sizes from standard buffers)
    float* h_mode_params = nullptr;    // Host mode parameters
    float* d_mode_params = nullptr;    // Device mode parameters
    float* h_modal_output = nullptr;   // Host modal output
    float* d_modal_output = nullptr;   // Device modal output

    // CPU reference buffer
    float* cpu_reference = nullptr;

    // Buffer size calculations
    size_t mode_params_size;
    size_t mode_params_bytes;
    size_t modal_output_size;
    size_t modal_output_bytes;
};

// ============================================================================
// CUDA Kernel Declarations
// ============================================================================

// Custom complex exponential for performance
__device__ cuComplex my_cexpf(cuComplex expon);

// Modal synthesis kernel
__global__ void ModalSynthesisKernel(const float* bufIn, float* bufOut, int nModes,
                                     BenchmarkUtils::BenchmarkParams params);
