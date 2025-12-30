#pragma once

// Base class providing common CUDA benchmark scaffolding.

#include <functional>
#include <string>
#include <vector>
#include <iostream>
#include <memory>
#include <cstdio>

#include "globals.cuh"
#include "bench_utils.cuh"

// Kernel launch workflow: compute grid dims, launch <<<grid, block>>>, then check errors.
// For timed runs use BenchmarkUtils::launchKernelTimed() to collect GPU duration.

class GPUABenchmark {
public:
    struct BenchmarkResult {
        std::vector<float> latencies;
        std::vector<float> gpu_latencies;
        BenchmarkUtils::Statistics statistics;
        BenchmarkUtils::Statistics gpu_statistics;
        std::string benchmark_name;
        size_t buffer_size;
        size_t track_count;
        int iterations;
        // Performance metrics
        double throughput_gbps;     // Throughput in GB/s
        double samples_per_sec;     // Samples processed per second
        size_t bytes_processed;     // Total bytes processed
        float mean_latency_ms;      // Mean latency in milliseconds
    };

    enum class ValidationStatus {
        SUCCESS = 0,
        FAILURE = 1,
        FATAL = -1
    };

    struct ValidationData {
        ValidationStatus status = ValidationStatus::SUCCESS;
        std::vector<std::string> messages;
        float max_error = 0.0f;
        float mean_error = 0.0f;
    };

protected:
    struct BufferSet {
        float* h_input = nullptr;
        float* h_output = nullptr;
        float* d_input = nullptr;
        float* d_output = nullptr;
        size_t element_count = 0;
        size_t size_bytes = 0;

        ~BufferSet() {
            cleanup();
        }

        void cleanup() {
            // Synchronize to ensure all GPU work completes before freeing
            cudaError_t sync_error = cudaDeviceSynchronize();
            if (sync_error != cudaSuccess) {
                fprintf(stderr, "Warning: cudaDeviceSynchronize before cleanup failed: %s\n",
                        cudaGetErrorString(sync_error));
            }
            BenchmarkUtils::freeHostBuffers({h_input, h_output});
            BenchmarkUtils::freeDeviceBuffers({d_input, d_output});
            h_input = h_output = nullptr;
            d_input = d_output = nullptr;
        }
    };

    // Member variables
    BufferSet buffers;
    BenchmarkUtils::BenchmarkTimer timer;
    std::string benchmark_name_;
    size_t buffer_size_;
    size_t track_count_;
    float current_iteration_gpu_ms_ = 0.0f;

public:
    GPUABenchmark(const std::string& name, size_t buffer_size = BUFSIZE, size_t track_count = NTRACKS)
        : benchmark_name_(name), buffer_size_(buffer_size), track_count_(track_count) {}

    virtual ~GPUABenchmark() = default;

    // ============================================================================
    // Pure Virtual Interface - Subclasses Must Implement
    // ============================================================================

    virtual void setupBenchmark() = 0;
    virtual void runKernel() = 0;
    virtual void performBenchmarkIteration() = 0;
    virtual void validate(ValidationData& validation_data) = 0;

    // ============================================================================
    // Concrete Interface - Provided by Base Class
    // ============================================================================

    void allocateBuffers(size_t element_count);
    void transferToDevice();
    void transferToHost();
    BenchmarkResult runKernelBenchmark(int iterations = NRUNS, int warmupIterations = 3);
    BenchmarkResult runBenchmark(int iterations = NRUNS, int warmupIterations = 3);
    void generateTestData(unsigned int seed = 42);
    void writeResults(const BenchmarkResult& result, const std::string& filename = "");
    void printResults(const BenchmarkResult& result);

    // ============================================================================
    // Getters
    // ============================================================================

    const std::string& getName() const { return benchmark_name_; }
    size_t getBufferSize() const { return buffer_size_; }
    size_t getTrackCount() const { return track_count_; }
    size_t getTotalElements() const { return buffer_size_ * track_count_; }

    // ============================================================================
    // Protected Helpers for Subclasses
    // ============================================================================

protected:
    float* getHostInput() { return buffers.h_input; }
    float* getHostOutput() { return buffers.h_output; }
    float* getDeviceInput() { return buffers.d_input; }
    float* getDeviceOutput() { return buffers.d_output; }
    BenchmarkUtils::BenchmarkParams makeBenchmarkParams(float gainValue = 0.0f) const;
    std::pair<int, int> calculateGridDimensions(int desired_threads_per_block = 256) const;
    void synchronizeAndCheck();
    ValidationData compareWithReference(const float* cpu_reference, float tolerance = 1e-5f);
    void resetGpuIterationMetrics();
    void recordGpuDuration(float milliseconds);
    BenchmarkResult runWithIteration(int iterations,
                                     int warmupIterations,
                                     const std::function<void()>& iterationBody);
};
