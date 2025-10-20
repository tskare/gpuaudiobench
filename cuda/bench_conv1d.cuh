#pragma once

// 1D convolution benchmark using texture memory for impulse responses.

#include "bench_base.cuh"
#include <cuda_texture_types.h>

class Conv1DBenchmark : public GPUABenchmark {
public:
    // Configuration
    static const int DEFAULT_IR_LEN = 1024;

    // ============================================================================
    // Constructor/Destructor
    // ============================================================================

    Conv1DBenchmark(int ir_length = DEFAULT_IR_LEN, size_t buffer_size = BUFSIZE, size_t track_count = NTRACKS);
    ~Conv1DBenchmark();

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

    void allocateConvBuffers();
    void setupTextureMemory();
    void generateImpulseResponses();
    void calculateCPUReference();
    void cleanupConvBuffers();
    void cleanupTextureMemory();

    // CPU reference convolution implementation
    void conv1DCPUReference(const float* input, const float* impulse_response, float* output,
                           int ir_len, int buffer_size, int track_count);

    // ============================================================================
    // Private Member Variables
    // ============================================================================

    int ir_length_;

    // Convolution-specific buffers
    float* h_ir_buf = nullptr;      // Host impulse response buffer
    float* d_ir_buf = nullptr;      // Device impulse response buffer

    // Texture memory objects
    cudaArray_t cu_array_irs = nullptr;
    cudaTextureObject_t tex_obj_irs = 0;
    bool texture_created = false;

    // CPU reference buffers
    float* cpu_reference = nullptr;

    // Buffer size calculations
    size_t ir_buffer_size;
    size_t ir_buffer_bytes;
};

// ============================================================================
// CUDA Kernel Declaration
// ============================================================================

__global__ void Conv1DTextureMemoryImplKernel(
    const float* bufIn, float* bufOut, const cudaTextureObject_t textureRefIRs, int irLen);

// ============================================================================
// Legacy Interface Wrapper
// ============================================================================

void RunConv1DBenchmark();
