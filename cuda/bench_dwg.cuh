#pragma once

// 1D digital waveguide benchmark with naive and accelerated variants.

#include "bench_base.cuh"

// Digital Waveguide Parameters Structure
struct DWGParams {
    int numWaveguides;
    int bufferSize;
    int outputTracks;
    int minLength;
    int maxLength;
    float reflectionCoeff;
    float dampingCoeff;
};

// Digital Waveguide State Structure
struct WaveguideState {
    int length;
    int inputTapPos;
    int outputTapPos;
    int writePos;
    float gain;
    float reflection;
    float damping;
    float padding; // Align to 32 bytes
};

class DWGBenchmark : public GPUABenchmark {
public:
    // DWG variant types
    enum class Variant {
        NAIVE,
        ACCELERATED
    };

    // Configuration
    static const int DEFAULT_MIN_LENGTH = 100;
    static const int DEFAULT_MAX_LENGTH = 2000;
    static const float DEFAULT_REFLECTION_COEFF;
    static const float DEFAULT_DAMPING_COEFF;

    // ============================================================================
    // Constructor/Destructor
    // ============================================================================

    DWGBenchmark(Variant variant = Variant::NAIVE, size_t buffer_size = BUFSIZE, size_t track_count = NTRACKS);
    ~DWGBenchmark();

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

    void allocateDWGBuffers();
    void initializeDWGParameters();
    void initializeWaveguideStates();
    void calculateCPUReference();
    void cleanupDWGBuffers();

    // CPU reference DWG implementation
    void dwgCPUReference(const WaveguideState* waveguide_params,
                        float* delay_forward, float* delay_backward,
                        const float* input_signal, float* output_buffer,
                        const DWGParams* params);

    // ============================================================================
    // Private Member Variables
    // ============================================================================

    Variant variant_;

    // DWG-specific buffers
    WaveguideState* h_waveguide_params = nullptr;
    WaveguideState* d_waveguide_params = nullptr;
    DWGParams* h_dwg_params = nullptr;
    DWGParams* d_dwg_params = nullptr;

    // Delay line buffers
    float* h_delay_forward = nullptr;
    float* d_delay_forward = nullptr;
    float* h_delay_backward = nullptr;
    float* d_delay_backward = nullptr;

    // Input/output buffers
    float* h_input_signal = nullptr;
    float* d_input_signal = nullptr;
    float* h_output_buffer = nullptr;
    float* d_output_buffer = nullptr;

    // CPU reference buffers
    float* cpu_reference = nullptr;
    float* cpu_delay_forward = nullptr;
    float* cpu_delay_backward = nullptr;

    // Buffer size calculations
    size_t delay_line_size;
    size_t delay_line_bytes;
    size_t output_buffer_size;
    size_t output_buffer_bytes;
};

// ============================================================================
// CUDA Kernel Declarations
// ============================================================================

__global__ void DWG1DNaiveKernel(
    const WaveguideState* waveguideParams,
    float* delayLineForward,
    float* delayLineBackward,
    const float* inputSignal,
    float* outputBuffer,
    const DWGParams* params
);

__global__ void DWG1DAccelKernel(
    const WaveguideState* waveguideParams,
    float* delayLineForward,
    float* delayLineBackward,
    const float* inputSignal,
    float* outputBuffer,
    const DWGParams* params
);
