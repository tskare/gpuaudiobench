#include "bench_base.cuh"
#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <stdexcept>
#include <utility>

void GPUABenchmark::allocateBuffers(size_t element_count) {
    if (element_count == 0) {
        throw std::invalid_argument("allocateBuffers requires element_count > 0");
    }

    buffers.element_count = element_count;
    buffers.size_bytes = element_count * sizeof(float);

    // Allocate host buffers (pinned for faster transfers)
    buffers.h_input = BenchmarkUtils::allocateHostBuffer<float>(
        element_count, benchmark_name_ + " host input buffer");
    buffers.h_output = BenchmarkUtils::allocateHostBuffer<float>(
        element_count, benchmark_name_ + " host output buffer");

    // Allocate device buffers
    buffers.d_input = BenchmarkUtils::allocateDeviceBuffer<float>(
        element_count, benchmark_name_ + " device input buffer");
    buffers.d_output = BenchmarkUtils::allocateDeviceBuffer<float>(
        element_count, benchmark_name_ + " device output buffer");
}

void GPUABenchmark::transferToDevice() {
    if (!buffers.d_input || !buffers.h_input) {
        throw std::runtime_error("transferToDevice called before input buffers were allocated");
    }
    BenchmarkUtils::copyToDevice(buffers.d_input, buffers.h_input, buffers.element_count);
}

void GPUABenchmark::transferToHost() {
    if (!buffers.d_output || !buffers.h_output) {
        throw std::runtime_error("transferToHost called before output buffers were allocated");
    }
    BenchmarkUtils::copyToHost(buffers.h_output, buffers.d_output, buffers.element_count);
}

void GPUABenchmark::generateTestData(unsigned int seed) {
    if (!buffers.h_input) {
        throw std::runtime_error("generateTestData called before host input buffer allocation");
    }
    BenchmarkUtils::generateRandomAudioData(buffers.h_input, buffers.element_count, seed);
}

GPUABenchmark::BenchmarkResult GPUABenchmark::runKernelBenchmark(int iterations, int warmupIterations) {
    return runWithIteration(iterations, warmupIterations, [this]() { this->runKernel(); });
}

GPUABenchmark::BenchmarkResult GPUABenchmark::runBenchmark(int iterations, int warmupIterations) {
    return runWithIteration(iterations, warmupIterations, [this]() { this->performBenchmarkIteration(); });
}

GPUABenchmark::BenchmarkResult GPUABenchmark::runWithIteration(
    int iterations,
    int warmupIterations,
    const std::function<void()>& iterationBody) {

    BenchmarkResult result;
    result.benchmark_name = benchmark_name_;
    result.buffer_size = buffer_size_;
    result.track_count = track_count_;
    result.iterations = iterations;

    if (warmupIterations > 0) {
        printf("Running %d warmup iterations...\n", warmupIterations);
        for (int i = 0; i < warmupIterations; ++i) {
            try {
                resetGpuIterationMetrics();
                iterationBody();
                printf("  Warmup %d/%d completed\n", i + 1, warmupIterations);
            } catch (const std::exception& e) {
                printf("  Warmup iteration %d failed: %s\n", i + 1, e.what());
            }
        }
        printf("Warmup complete, starting timed iterations...\n");
    }

    result.latencies.clear();
    result.latencies.reserve(iterations);
    std::vector<float> gpuLatencies;
    gpuLatencies.reserve(iterations);

    for (int i = 0; i < iterations; ++i) {
        resetGpuIterationMetrics();
        double latency = BenchmarkUtils::BenchmarkTimer::measureKernel(iterationBody);
        result.latencies.push_back(static_cast<float>(latency));
        gpuLatencies.push_back(current_iteration_gpu_ms_);
    }

    result.statistics = BenchmarkUtils::calculateStatistics(result.latencies);

    const bool hasGpuTiming = std::any_of(
        gpuLatencies.begin(), gpuLatencies.end(),
        [](float value) { return value > 0.0f; });

    if (hasGpuTiming) {
        result.gpu_latencies = std::move(gpuLatencies);
        result.gpu_statistics = BenchmarkUtils::calculateStatistics(result.gpu_latencies);
    } else {
        result.gpu_latencies.clear();
        result.gpu_statistics = {};
    }

    const size_t totalSamples = buffer_size_ * track_count_;
    result.bytes_processed = totalSamples * sizeof(float);
    result.mean_latency_ms = result.statistics.mean;
    const double meanLatencySec = result.mean_latency_ms / 1000.0;
    result.throughput_gbps = (result.bytes_processed / (1024.0 * 1024.0 * 1024.0)) / meanLatencySec;
    result.samples_per_sec = totalSamples / meanLatencySec;

    return result;
}

void GPUABenchmark::writeResults(const BenchmarkResult& result, const std::string& filename) {
    std::string output_file = filename;
    if (output_file.empty()) {
        output_file = "/tmp/" + result.benchmark_name + "_latencies.txt";
    }

    BenchmarkUtils::writeLatenciesToFile(result.latencies, output_file);
}

void GPUABenchmark::printResults(const BenchmarkResult& result) {
    BenchmarkUtils::printStatistics(result.latencies, result.benchmark_name);

    if (!result.gpu_latencies.empty()) {
        BenchmarkUtils::Statistics gpuStats = result.gpu_statistics;
        if (gpuStats.count == 0) {
            gpuStats = BenchmarkUtils::calculateStatistics(result.gpu_latencies);
        }

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "GPU Median:  " << gpuStats.median << " ms" << std::endl;
        std::cout << "GPU P95:     " << gpuStats.p95 << " ms" << std::endl;
        std::cout << "GPU Mean:    " << gpuStats.mean << " ms" << std::endl;
    }

    // Print performance metrics
    std::cout << "\nPerformance Metrics:" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Throughput:        " << result.throughput_gbps << " GB/s" << std::endl;
    std::cout << "Samples/sec:       " << std::fixed << std::setprecision(0) << result.samples_per_sec << std::endl;
    std::cout << "Bytes processed:   " << result.bytes_processed << std::endl;
}

BenchmarkUtils::BenchmarkParams GPUABenchmark::makeBenchmarkParams(float gainValue) const {
    return BenchmarkUtils::makeBenchmarkParams(buffer_size_, track_count_, gainValue);
}

void GPUABenchmark::resetGpuIterationMetrics() {
    current_iteration_gpu_ms_ = 0.0f;
}

void GPUABenchmark::recordGpuDuration(float milliseconds) {
    if (milliseconds > 0.0f) {
        current_iteration_gpu_ms_ += milliseconds;
    }
}

std::pair<int, int> GPUABenchmark::calculateGridDimensions(int desired_threads_per_block) const {
    // Clamp threads per block to reasonable limits
    int threads_per_block = std::min(desired_threads_per_block, 512);
    threads_per_block = std::max(threads_per_block, 32);

    // Calculate blocks needed for track count
    int blocks_per_grid = (static_cast<int>(track_count_) + threads_per_block - 1) / threads_per_block;

    return std::make_pair(blocks_per_grid, threads_per_block);
}

void GPUABenchmark::synchronizeAndCheck() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

GPUABenchmark::ValidationData GPUABenchmark::compareWithReference(
    const float* cpu_reference, float tolerance) {

    ValidationData validation;
    validation.status = ValidationStatus::SUCCESS;

    if (!buffers.h_output || !cpu_reference) {
        validation.status = ValidationStatus::FATAL;
        validation.messages.push_back("Null pointer in validation comparison");
        return validation;
    }

    float sum_error = 0.0f;
    float max_error = 0.0f;
    int error_count = 0;

    for (size_t i = 0; i < buffers.element_count; ++i) {
        float diff = std::abs(buffers.h_output[i] - cpu_reference[i]);
        sum_error += diff;
        max_error = std::max(max_error, diff);

        if (diff > tolerance) {
            error_count++;
            if (validation.messages.size() < 10) {  // Limit error messages
                validation.messages.push_back(
                    "Error at index " + std::to_string(i) +
                    ": expected " + std::to_string(cpu_reference[i]) +
                    ", got " + std::to_string(buffers.h_output[i]) +
                    ", diff " + std::to_string(diff));
            }
        }
    }

    validation.mean_error = sum_error / static_cast<float>(buffers.element_count);
    validation.max_error = max_error;

    if (error_count > 0) {
        validation.status = ValidationStatus::FAILURE;
        validation.messages.insert(validation.messages.begin(),
            "Validation failed: " + std::to_string(error_count) + " out of " +
            std::to_string(buffers.element_count) + " elements exceeded tolerance");
    }

    return validation;
}
