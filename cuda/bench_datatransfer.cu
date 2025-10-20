#include "bench_datatransfer.cuh"
#include <cstring>
#include <cmath>

// Static configuration definitions
const DataTransferBenchmark::Config DataTransferBenchmark::CONFIGS[] = {
    {0.01f, 0.99f, "datacopy0199"},  // 1% input, 99% output
    {0.20f, 0.80f, "datacopy2080"},  // 20% input, 80% output
    {0.50f, 0.50f, "datacopy5050"},  // 50% input, 50% output
    {0.80f, 0.20f, "datacopy8020"},  // 80% input, 20% output
    {0.99f, 0.01f, "datacopy9901"}   // 99% input, 1% output
};

const int DataTransferBenchmark::NUM_CONFIGS = sizeof(CONFIGS) / sizeof(CONFIGS[0]);

// ============================================================================
// CUDA Kernel Implementation
// ============================================================================

__global__ void DataTransferKernel(const float* bufIn, float* bufOut, int inSize, int outSize) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // Simple copy operation with minimal processing
    // This exercises memory bandwidth rather than compute
    if (idx < outSize) {
        if (idx < inSize) {
            // Copy from input where available
            bufOut[idx] = bufIn[idx];
        } else {
            // Fill remainder with computed values
            bufOut[idx] = 0.5f + 0.5f * sinf((float)idx * 0.001f);
        }
    }
}

// ============================================================================
// DataTransferBenchmark Implementation
// ============================================================================

DataTransferBenchmark::DataTransferBenchmark(const Config& config)
    : GPUABenchmark(config.name, 1, 1), config_(config) {  // Use minimal base allocation

    // Calculate actual buffer sizes based on ratios
    input_size = static_cast<int>(BASE_BUFFER_SIZE * config_.inputRatio);
    output_size = static_cast<int>(BASE_BUFFER_SIZE * config_.outputRatio);
    input_size_bytes = input_size * sizeof(float);
    output_size_bytes = output_size * sizeof(float);
}

DataTransferBenchmark::DataTransferBenchmark(float input_ratio, float output_ratio, const char* name)
    : DataTransferBenchmark(Config{input_ratio, output_ratio, name}) {
}

DataTransferBenchmark::~DataTransferBenchmark() {
    cleanupVariableSizeBuffers();
    BenchmarkUtils::freeHostBuffers({cpu_reference});
    cpu_reference = nullptr;  // Reset pointer for safety
}

DataTransferBenchmark* DataTransferBenchmark::createFromName(const std::string& name) {
    // Find matching configuration
    for (int i = 0; i < NUM_CONFIGS; ++i) {
        if (name == CONFIGS[i].name) {
            return new DataTransferBenchmark(CONFIGS[i]);
        }
    }
    return nullptr;  // Unknown configuration
}

void DataTransferBenchmark::setupBenchmark() {
    // Don't use base class standard buffers - we need variable sizes
    // allocateBuffers(getTotalElements());  // Skip this

    // Allocate variable size buffers
    allocateVariableSizeBuffers();

    // Allocate CPU reference buffer
    cpu_reference = BenchmarkUtils::allocateHostBuffer<float>(
        output_size, std::string(config_.name) + " cpu reference");

    // Initialize input data
    for (int i = 0; i < input_size; ++i) {
        h_input_var[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    // Calculate CPU reference
    calculateCPUReference();

    printf("DataTransfer %s setup complete - Input: %d floats (%.1f%%), Output: %d floats (%.1f%%)\n",
           config_.name, input_size, config_.inputRatio * 100.0f,
           output_size, config_.outputRatio * 100.0f);
}

void DataTransferBenchmark::runKernel() {
    // Transfer input to device
    CUDA_CHECK(cudaMemcpy(d_input_var, h_input_var, input_size_bytes, cudaMemcpyHostToDevice));

    // Calculate grid dimensions for output size
    int threads_per_block = 256;
    int blocks_per_grid = (output_size + threads_per_block - 1) / threads_per_block;

    // Launch kernel
    DataTransferKernel<<<blocks_per_grid, threads_per_block>>>(
        d_input_var, d_output_var, input_size, output_size
    );

    // Synchronize and check for errors
    synchronizeAndCheck();

    // Transfer output back to host
    CUDA_CHECK(cudaMemcpy(h_output_var, d_output_var, output_size_bytes, cudaMemcpyDeviceToHost));
}

void DataTransferBenchmark::performBenchmarkIteration() {
    // Combined iteration pattern that handles transfers and timing

    // Transfer input to device
    CUDA_CHECK(cudaMemcpy(d_input_var, h_input_var, input_size_bytes, cudaMemcpyHostToDevice));

    // Calculate grid dimensions for output size
    int threads_per_block = 256;
    int blocks_per_grid = (output_size + threads_per_block - 1) / threads_per_block;

    // Launch kernel
    DataTransferKernel<<<blocks_per_grid, threads_per_block>>>(
        d_input_var, d_output_var, input_size, output_size
    );

    // Synchronize and check for errors
    synchronizeAndCheck();

    // Transfer output back to host
    CUDA_CHECK(cudaMemcpy(h_output_var, d_output_var, output_size_bytes, cudaMemcpyDeviceToHost));
}

void DataTransferBenchmark::validate(ValidationData& validation_data) {
    // Simple validation - compare with CPU reference
    float max_error = 0.0f;
    float mean_error = 0.0f;

    for (int i = 0; i < output_size; ++i) {
        float error = std::abs(h_output_var[i] - cpu_reference[i]);
        max_error = std::max(max_error, error);
        mean_error += error;
    }
    mean_error /= output_size;

    validation_data.max_error = max_error;
    validation_data.mean_error = mean_error;

    if (max_error > 1e-5f) {
        validation_data.status = ValidationStatus::FAILURE;
        validation_data.messages.push_back("DataTransfer validation failed");
    } else {
        validation_data.status = ValidationStatus::SUCCESS;
        validation_data.messages.push_back("DataTransfer validation passed");
    }
}

// ============================================================================
// Private Helper Methods
// ============================================================================

void DataTransferBenchmark::allocateVariableSizeBuffers() {
    // Allocate host buffers
    h_input_var = BenchmarkUtils::allocateHostBuffer<float>(
        input_size, std::string(config_.name) + " host input buffer");
    h_output_var = BenchmarkUtils::allocateHostBuffer<float>(
        output_size, std::string(config_.name) + " host output buffer");

    // Allocate device buffers
    d_input_var = BenchmarkUtils::allocateDeviceBuffer<float>(
        input_size, std::string(config_.name) + " device input buffer");
    d_output_var = BenchmarkUtils::allocateDeviceBuffer<float>(
        output_size, std::string(config_.name) + " device output buffer");

    // Initialize output buffer to zero
    memset(h_output_var, 0, output_size_bytes);
    CUDA_CHECK(cudaMemset(d_output_var, 0, output_size_bytes));
}

void DataTransferBenchmark::calculateCPUReference() {
    // Replicate the kernel logic on CPU
    for (int idx = 0; idx < output_size; ++idx) {
        if (idx < input_size) {
            // Copy from input where available
            cpu_reference[idx] = h_input_var[idx];
        } else {
            // Fill remainder with computed values
            cpu_reference[idx] = 0.5f + 0.5f * sinf(static_cast<float>(idx) * 0.001f);
        }
    }
}

void DataTransferBenchmark::cleanupVariableSizeBuffers() {
    BenchmarkUtils::freeHostBuffers({h_input_var, h_output_var});
    BenchmarkUtils::freeDeviceBuffers({d_input_var, d_output_var});
    h_input_var = h_output_var = nullptr;
    d_input_var = d_output_var = nullptr;
}
