#include "bench_gainstats.cuh"
#include "thread_config.cuh"
#include "benchmark_constants.cuh"
#include <cstring>
#include <cmath>

__global__ void GainStatsKernel(const float* bufIn, float* bufOut, float* stats,
                                BenchmarkUtils::BenchmarkParams params) {
    int trackIdx = blockDim.x * blockIdx.x + threadIdx.x;
    const int trackCount = static_cast<int>(params.trackCount);
    const int bufferSize = static_cast<int>(params.bufferSize);

    if (trackIdx >= trackCount) return;

    float mean = 0.0f;
    float maxVal = -1e9f;
    for (int i = 0; i < bufferSize; i++) {
        int idx = trackIdx * bufferSize + i;
        float samp = bufIn[idx];
        mean += samp;
        bufOut[idx] = samp * BenchmarkConstants::GAINSTATS_GAIN;

        if (samp > maxVal) {
            maxVal = samp;
        }
    }
    mean /= bufferSize;

    stats[trackIdx * GainStatsBenchmark::NSTATS + 0] = mean;
    stats[trackIdx * GainStatsBenchmark::NSTATS + 1] = maxVal;
}

GainStatsBenchmark::GainStatsBenchmark(size_t buffer_size, size_t track_count)
    : GPUABenchmark("GainStats", buffer_size, track_count) {
    stats_count = track_count * NSTATS;
    stats_size_bytes = stats_count * sizeof(float);
}

GainStatsBenchmark::~GainStatsBenchmark() {
    cleanupStatsBuffers();
    BenchmarkUtils::freeHostBuffers({cpu_reference, cpu_stats_reference});
    cpu_reference = nullptr;
    cpu_stats_reference = nullptr;
}

void GainStatsBenchmark::setupBenchmark() {
    allocateBuffers(getTotalElements());
    allocateStatsBuffers();
    generateTestData(42);

    cpu_reference = BenchmarkUtils::allocateHostBuffer<float>(
        getTotalElements(), "gainstats cpu reference");
    cpu_stats_reference = BenchmarkUtils::allocateHostBuffer<float>(
        stats_count, "gainstats cpu stats reference");

    calculateCPUReference();

    printf("GainStats benchmark setup complete (gain = %.1f, computing mean + max per track)\n",
           BenchmarkConstants::GAINSTATS_GAIN);
}

void GainStatsBenchmark::runKernel() {
    performBenchmarkIteration();
}

void GainStatsBenchmark::performBenchmarkIteration() {
    transferToDevice();

    CUDA_CHECK(cudaMemset(d_stats, 0, stats_size_bytes));

    auto [blocks_per_grid, threads_per_block] = calculateGridDimensions(ThreadConfig::DEFAULT_BLOCK_SIZE_1D);

    const auto params = makeBenchmarkParams(BenchmarkConstants::GAINSTATS_GAIN);
    GainStatsKernel<<<blocks_per_grid, threads_per_block>>>(
        getDeviceInput(),
        getDeviceOutput(),
        d_stats,
        params
    );

    synchronizeAndCheck();

    transferToHost();
    CUDA_CHECK(cudaMemcpy(h_stats, d_stats, stats_size_bytes, cudaMemcpyDeviceToHost));
}

void GainStatsBenchmark::validate(ValidationData& validation_data) {
    validation_data = compareWithReference(cpu_reference, 1e-5f);

    float max_stats_error = 0.0f;
    float mean_stats_error = 0.0f;

    for (size_t i = 0; i < stats_count; ++i) {
        float error = std::abs(h_stats[i] - cpu_stats_reference[i]);
        max_stats_error = std::max(max_stats_error, error);
        mean_stats_error += error;
    }
    mean_stats_error /= stats_count;

    if (max_stats_error > 1e-4f) {
        validation_data.status = ValidationStatus::FAILURE;
        validation_data.messages.push_back("Statistics validation failed");
        validation_data.max_error = std::max(validation_data.max_error, max_stats_error);
    } else if (validation_data.status == ValidationStatus::SUCCESS) {
        validation_data.messages.push_back("GainStats validation passed (output + statistics)");
    }
}

void GainStatsBenchmark::allocateStatsBuffers() {
    h_stats = BenchmarkUtils::allocateHostBuffer<float>(
        stats_count, benchmark_name_ + " host stats buffer");

    d_stats = BenchmarkUtils::allocateDeviceBuffer<float>(
        stats_count, benchmark_name_ + " device stats buffer");

    memset(h_stats, 0, stats_size_bytes);
    CUDA_CHECK(cudaMemset(d_stats, 0, stats_size_bytes));
}

void GainStatsBenchmark::calculateCPUReference() {
    const float* input = getHostInput();

    for (size_t i = 0; i < getTotalElements(); ++i) {
        cpu_reference[i] = BenchmarkConstants::GAINSTATS_GAIN * input[i];
    }

    for (size_t track = 0; track < getTrackCount(); ++track) {
        float mean = 0.0f;
        float maxVal = -1e9f;

        for (size_t sample = 0; sample < getBufferSize(); ++sample) {
            size_t idx = track * getBufferSize() + sample;
            float samp = input[idx];
            mean += samp;
            if (samp > maxVal) {
                maxVal = samp;
            }
        }
        mean /= getBufferSize();

        cpu_stats_reference[track * NSTATS + 0] = mean;
        cpu_stats_reference[track * NSTATS + 1] = maxVal;
    }
}

void GainStatsBenchmark::cleanupStatsBuffers() {
    if (h_stats) {
        BenchmarkUtils::freeHostBuffers({h_stats});
        h_stats = nullptr;
    }
    if (d_stats) {
        BenchmarkUtils::freeDeviceBuffers({d_stats});
        d_stats = nullptr;
    }
}
