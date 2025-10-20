#include "bench_gain.cuh"
#include "thread_config.cuh"
#include <cstring>

// Static constant definition
const float GainBenchmark::GAIN_VALUE = 2.0f;

// ============================================================================
// CUDA Kernel (unchanged from original)
// ============================================================================

__global__ void GainKernel(const float* bufIn,
                           float* bufOut,
                           BenchmarkUtils::BenchmarkParams params) {
    const int trackIdx = blockDim.x * blockIdx.x + threadIdx.x;
    const int trackCount = static_cast<int>(params.trackCount);
    const int bufferSize = static_cast<int>(params.bufferSize);
    const int totalSamples = static_cast<int>(params.totalSamples);

    if (trackIdx >= trackCount) {
        return;
    }

    // Process entire buffer for this track
    for (int sample = 0; sample < bufferSize; ++sample) {
        const int idx = trackIdx * bufferSize + sample;
        if (idx < totalSamples) {
            bufOut[idx] = params.gainValue * bufIn[idx];
        }
    }
}

// ============================================================================
// GainBenchmark Implementation
// ============================================================================

GainBenchmark::GainBenchmark(size_t buffer_size, size_t track_count, bool enable_validation)
    : GPUABenchmark("Gain", buffer_size, track_count), enable_validation_(enable_validation) {
}

GainBenchmark::~GainBenchmark() {
    cleanupCPUReference();
}

void GainBenchmark::setupBenchmark() {
    allocateBuffers(getTotalElements());
    generateTestData(42);

    if (enable_validation_) {
        cpu_reference = BenchmarkUtils::allocateHostBuffer<float>(
            getTotalElements(), "gain cpu reference");
        calculateCPUReference();
    }

    printf("Gain benchmark setup complete (gain = %.1f)\n", GAIN_VALUE);
}

void GainBenchmark::runKernel() {
    performBenchmarkIteration();
}

void GainBenchmark::performBenchmarkIteration() {
    transferToDevice();

    auto [blocks_per_grid, threads_per_block] = calculateGridDimensions(ThreadConfig::DEFAULT_BLOCK_SIZE_1D);

    const auto params = makeBenchmarkParams(GAIN_VALUE);

    const double gpuMs = BenchmarkUtils::launchKernelTimed(
        GainKernel,
        dim3(blocks_per_grid),
        dim3(threads_per_block),
        getDeviceInput(),
        getDeviceOutput(),
        params
    );
    recordGpuDuration(static_cast<float>(gpuMs));

    transferToHost();
}

void GainBenchmark::validate(ValidationData& validation_data) {
    if (!enable_validation_) {
        validation_data.status = ValidationStatus::SUCCESS;
        validation_data.messages.push_back("Validation skipped (disabled)");
        return;
    }

    validation_data = compareWithReference(cpu_reference, 1e-5f);

    if (validation_data.status == ValidationStatus::SUCCESS) {
        validation_data.messages.push_back("Gain validation passed");
    }
}

// ============================================================================
// Private Helper Methods
// ============================================================================

void GainBenchmark::calculateCPUReference() {
    if (!enable_validation_ || !cpu_reference) {
        return;
    }

    const float* input = getHostInput();

    for (size_t i = 0; i < getTotalElements(); ++i) {
        cpu_reference[i] = GAIN_VALUE * input[i];
    }
}

void GainBenchmark::cleanupCPUReference() {
    BenchmarkUtils::freeHostBuffers({cpu_reference});
    cpu_reference = nullptr;
}
