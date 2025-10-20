#pragma once

// Configurable data transfer benchmark covering multiple I/O ratios.

#include "bench_base.cuh"

class DataTransferBenchmark : public GPUABenchmark {
public:
    // Data transfer configuration
    struct Config {
        float inputRatio;
        float outputRatio;
        const char* name;
    };

    // Predefined configurations matching original benchmarks
    static const Config CONFIGS[];
    static const int NUM_CONFIGS;

    // Base buffer size for calculations (10MB of floats)
    static constexpr int BASE_BUFFER_SIZE = (10 * 1024 * 1024 / sizeof(float));

    // ============================================================================
    // Constructor/Destructor
    // ============================================================================

    DataTransferBenchmark(const Config& config);
    DataTransferBenchmark(float input_ratio, float output_ratio, const char* name = "DataTransfer");
    ~DataTransferBenchmark();

    // ============================================================================
    // Static Factory Methods
    // ============================================================================

    static DataTransferBenchmark* createFromName(const std::string& name);

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

    void allocateVariableSizeBuffers();
    void calculateCPUReference();
    void cleanupVariableSizeBuffers();

    // ============================================================================
    // Private Member Variables
    // ============================================================================

    Config config_;

    // Variable size buffers (different from base class standard buffers)
    float* h_input_var = nullptr;
    float* h_output_var = nullptr;
    float* d_input_var = nullptr;
    float* d_output_var = nullptr;

    // CPU reference buffer
    float* cpu_reference = nullptr;

    // Actual buffer sizes
    int input_size;
    int output_size;
    size_t input_size_bytes;
    size_t output_size_bytes;
};

// ============================================================================
// CUDA Kernel Declaration
// ============================================================================

__global__ void DataTransferKernel(const float* bufIn, float* bufOut, int inSize, int outSize);
