#include "bench_conv1d.cuh"
#include "thread_config.cuh"
#include <cstring>
#include <cmath>
#include <stdexcept>

// ============================================================================
// CUDA Kernel Implementation
// ============================================================================

__global__ void Conv1DTextureMemoryImplKernel(
    const float* bufIn, float* bufOut, const cudaTextureObject_t textureRefIRs, int irLen) {
    int whichThread = blockIdx.x * blockDim.x + threadIdx.x;

    if (whichThread >= NTRACKS) return;

    for (int i = 0; i < BUFSIZE; i++) {
        // Perform time-series convolution
        float samp = 0.0f;
        for (int j = 0; j < irLen; j++) {
            // Texture memory for IR lookup (uses dedicated cache separate from global memory)
            int input_idx = whichThread * BUFSIZE + i - j;
            if (input_idx >= 0 && input_idx < NTRACKS * BUFSIZE) {
                samp += tex2D<float>(textureRefIRs, whichThread, j) * bufIn[input_idx];
            }
        }
        // Coalesced writes: consecutive threads → consecutive memory (4-16x bandwidth vs. strided)
        bufOut[whichThread * BUFSIZE + i] = samp;
    }
}

// ============================================================================
// Conv1DBenchmark Implementation
// ============================================================================

Conv1DBenchmark::Conv1DBenchmark(int ir_length, size_t buffer_size, size_t track_count)
    : GPUABenchmark("Conv1D", buffer_size, track_count), ir_length_(ir_length) {
    ir_buffer_size = track_count * ir_length;
    ir_buffer_bytes = ir_buffer_size * sizeof(float);
}

Conv1DBenchmark::~Conv1DBenchmark() {
    cleanupTextureMemory();
    cleanupConvBuffers();
    BenchmarkUtils::freeHostBuffers({cpu_reference});
    cpu_reference = nullptr;  // Reset pointer for safety
}

void Conv1DBenchmark::setupBenchmark() {
    allocateBuffers(getTotalElements());
    allocateConvBuffers();
    generateImpulseResponses();
    setupTextureMemory();
    generateTestData(42);

    cpu_reference = BenchmarkUtils::allocateHostBuffer<float>(
        getTotalElements(), "conv1d cpu reference");

    calculateCPUReference();

    printf("Conv1D benchmark setup complete (IR length = %d, using texture memory)\n", ir_length_);
}

void Conv1DBenchmark::runKernel() {
    if (!texture_created || tex_obj_irs == 0) {
        throw std::runtime_error("Conv1DBenchmark::runKernel called before texture memory initialization");
    }

    transferToDevice();

    auto [blocks_per_grid, threads_per_block] = calculateGridDimensions(ThreadConfig::DEFAULT_BLOCK_SIZE_1D);

    const double gpuMs = BenchmarkUtils::launchKernelTimed(
        Conv1DTextureMemoryImplKernel,
        dim3(blocks_per_grid),
        dim3(threads_per_block),
        getDeviceInput(),
        getDeviceOutput(),
        tex_obj_irs,
        ir_length_
    );
    recordGpuDuration(static_cast<float>(gpuMs));

    transferToHost();
}

void Conv1DBenchmark::performBenchmarkIteration() {
    if (!texture_created || tex_obj_irs == 0) {
        throw std::runtime_error("Conv1DBenchmark::performBenchmarkIteration called before texture memory initialization");
    }

    // Combined iteration pattern that handles transfers and timing
    transferToDevice();

    // Calculate grid dimensions
    auto [blocks_per_grid, threads_per_block] = calculateGridDimensions(ThreadConfig::DEFAULT_BLOCK_SIZE_1D);

    // Launch kernel with texture object and capture GPU time
    const double gpuMs = BenchmarkUtils::launchKernelTimed(
        Conv1DTextureMemoryImplKernel,
        dim3(blocks_per_grid),
        dim3(threads_per_block),
        getDeviceInput(),
        getDeviceOutput(),
        tex_obj_irs,
        ir_length_
    );
    recordGpuDuration(static_cast<float>(gpuMs));

    // Transfer output back to host
    transferToHost();
}

void Conv1DBenchmark::validate(ValidationData& validation_data) {
    // Convolution validation uses looser tolerance due to numerical accumulation
    validation_data = compareWithReference(cpu_reference, 1e-3f);

    if (validation_data.status == ValidationStatus::SUCCESS) {
        validation_data.messages.push_back("Conv1D validation passed");
    }
}

// ============================================================================
// Private Helper Methods
// ============================================================================

void Conv1DBenchmark::allocateConvBuffers() {
    // Allocate host impulse response buffer
    h_ir_buf = BenchmarkUtils::allocateHostBuffer<float>(
        ir_buffer_size, benchmark_name_ + " host IR buffer");

    // Allocate device impulse response buffer
    d_ir_buf = BenchmarkUtils::allocateDeviceBuffer<float>(
        ir_buffer_size, benchmark_name_ + " device IR buffer");
}

void Conv1DBenchmark::setupTextureMemory() {
    // Create channel descriptor for float texture
    auto channel_desc = cudaCreateChannelDesc<float>();

    // Allocate CUDA array for texture (2D: tracks x IR length)
    cudaError_t err = cudaMallocArray(&cu_array_irs, &channel_desc, getTrackCount(), ir_length_);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate CUDA array for texture");
    }

    // Copy impulse response data to CUDA array
    size_t spitch = ir_length_ * sizeof(float);
    err = cudaMemcpy2DToArray(cu_array_irs, 0, 0, h_ir_buf, spitch,
                              ir_length_ * sizeof(float), getTrackCount(), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy data to CUDA array");
    }

    // Create texture resource descriptor
    cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = cu_array_irs;

    // Create texture descriptor
    cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.addressMode[0] = cudaAddressModeClamp;
    tex_desc.addressMode[1] = cudaAddressModeClamp;
    tex_desc.filterMode = cudaFilterModePoint;
    tex_desc.readMode = cudaReadModeElementType;
    tex_desc.normalizedCoords = 0;

    // Create texture object
    err = cudaCreateTextureObject(&tex_obj_irs, &res_desc, &tex_desc, nullptr);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to create texture object");
    }

    texture_created = true;
}

void Conv1DBenchmark::generateImpulseResponses() {
    const float PI = 3.14159265358979323846f;

    // Generate windowed sinc impulse responses for each track
    for (size_t track_idx = 0; track_idx < getTrackCount(); ++track_idx) {
        for (int ir_idx = 0; ir_idx < ir_length_; ++ir_idx) {
            size_t idx = track_idx * ir_length_ + ir_idx;

            // Create a simple windowed sinc IR for each track with slight frequency variation
            float freq = 0.1f + 0.05f * static_cast<float>(track_idx) / static_cast<float>(getTrackCount());
            float t = static_cast<float>(ir_idx) - static_cast<float>(ir_length_) / 2.0f;

            // Hamming window
            float window = 0.54f - 0.46f * cosf(2.0f * PI * static_cast<float>(ir_idx) / static_cast<float>(ir_length_ - 1));

            // Sinc function
            float sinc = (t == 0.0f) ? 1.0f : sinf(2.0f * PI * freq * t) / (2.0f * PI * freq * t);

            // Windowed and normalized sinc
            float value = window * sinc / static_cast<float>(ir_length_);

            h_ir_buf[idx] = value;
        }
    }

    // Copy to device buffer (for validation purposes)
    CUDA_CHECK(cudaMemcpy(d_ir_buf, h_ir_buf, ir_buffer_bytes, cudaMemcpyHostToDevice));
}

void Conv1DBenchmark::calculateCPUReference() {
    conv1DCPUReference(getHostInput(), h_ir_buf, cpu_reference,
                      ir_length_, getBufferSize(), getTrackCount());
}

void Conv1DBenchmark::conv1DCPUReference(const float* input, const float* impulse_response, float* output,
                                        int ir_len, int buffer_size, int track_count) {
    // Initialize output to zero
    memset(output, 0, track_count * buffer_size * sizeof(float));

    // Perform convolution for each track
    for (int track = 0; track < track_count; ++track) {
        for (int i = 0; i < buffer_size; ++i) {
            float samp = 0.0f;

            for (int j = 0; j < ir_len; ++j) {
                int input_idx = track * buffer_size + i - j;
                if (input_idx >= 0 && input_idx < track_count * buffer_size) {
                    int ir_idx = track * ir_len + j;
                    samp += impulse_response[ir_idx] * input[input_idx];
                }
            }

            // Match kernel output indexing: coalesced pattern track * BUFSIZE + i
            output[track * buffer_size + i] = samp;
        }
    }
}

void Conv1DBenchmark::cleanupConvBuffers() {
    BenchmarkUtils::freeHostBuffers({h_ir_buf});
    BenchmarkUtils::freeDeviceBuffers({d_ir_buf});
    h_ir_buf = nullptr;
    d_ir_buf = nullptr;
}

void Conv1DBenchmark::cleanupTextureMemory() {
    if (texture_created) {
        cudaDestroyTextureObject(tex_obj_irs);
        texture_created = false;
        tex_obj_irs = 0;
    }
    if (cu_array_irs) {
        cudaFreeArray(cu_array_irs);
        cu_array_irs = nullptr;
    }
}
