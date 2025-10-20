#include "bench_noop.cuh"
#include "thread_config.cuh"
#include <cstring>

// ============================================================================
// CUDA Kernel Implementation
// ============================================================================

__global__ void NoOpKernel(const float* bufIn, float* bufOut, int totalElements) {
    // Minimal no-op: copy input to output to enable validation
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < totalElements) {
        bufOut[idx] = bufIn[idx];
    }
}

// ============================================================================
// NoOpBenchmark Implementation
// ============================================================================

NoOpBenchmark::NoOpBenchmark(size_t buffer_size, size_t track_count)
    : GPUABenchmark("NoOp", buffer_size, track_count) {
}

NoOpBenchmark::~NoOpBenchmark() {
    BenchmarkUtils::freeHostBuffers({cpu_reference});
}

void NoOpBenchmark::setupBenchmark() {
    // Allocate GPU buffers using base class utilities
    allocateBuffers(getTotalElements());

    // Generate test data (minimal since this is just measuring overhead)
    generateTestData(42);  // Fixed seed for reproducible results

    // Allocate CPU reference buffer
    cpu_reference = BenchmarkUtils::allocateHostBuffer<float>(
        getTotalElements(), "noop cpu reference");

    // Calculate CPU reference (which is just a copy since it's no-op)
    calculateCPUReference();

    printf("NoOp benchmark setup complete (measuring kernel launch overhead)\n");
}

void NoOpBenchmark::runKernel() {
    // Delegate to performBenchmarkIteration() to avoid duplication
    performBenchmarkIteration();
}

void NoOpBenchmark::performBenchmarkIteration() {
    // Combined iteration helper handles transfers and timing
    transferToDevice();

    // Calculate grid dimensions
    auto [blocks_per_grid, threads_per_block] = calculateGridDimensions(ThreadConfig::DEFAULT_BLOCK_SIZE_1D);

    // Launch kernel
    NoOpKernel<<<blocks_per_grid, threads_per_block>>>(
        getDeviceInput(),
        getDeviceOutput(),
        static_cast<int>(getTotalElements())
    );

    // Synchronize and check for errors
    synchronizeAndCheck();

    // Transfer output back to host
    transferToHost();
}

void NoOpBenchmark::validate(ValidationData& validation_data) {
    // For NoOp, validation is simple since input should equal output
    validation_data = compareWithReference(cpu_reference, 1e-5f);

    if (validation_data.status == ValidationStatus::SUCCESS) {
        validation_data.messages.push_back("NoOp validation passed");
    }
}

// ============================================================================
// Private Helper Methods
// ============================================================================

void NoOpBenchmark::calculateCPUReference() {
    const float* input = getHostInput();

    // NoOp means input equals output (no processing)
    for (size_t i = 0; i < getTotalElements(); ++i) {
        cpu_reference[i] = input[i];
    }
}

// ============================================================================
// Legacy Interface Wrapper (for easy integration)
// ============================================================================
