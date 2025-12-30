#pragma once

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <chrono>
#include <cstdint>
#include <functional>
#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <cstring>

// Shared helpers for memory management, timing, and data generation.

namespace BenchmarkUtils {

    // ============================================================================
    // Shared Benchmark Parameters
    // ============================================================================

    struct BenchmarkParams {
        uint32_t bufferSize = 0;
        uint32_t trackCount = 0;
        uint32_t totalSamples = 0;
        float gainValue = 0.0f;
    };

    BenchmarkParams makeBenchmarkParams(size_t bufferSize,
                                        size_t trackCount,
                                        float gainValue = 0.0f);

    // ============================================================================
    // Benchmark Configuration
    // ============================================================================

    struct BenchmarkConfig {
        int bufferSize = 512;
        int trackCount = 128;
        int sampleRate = 48000;

        int iterations = 100;
        int warmupIterations = 3;

        bool enableValidation = false;
        bool enableProfiling = false;
        bool verboseOutput = false;

        std::string outputDirectory = "/tmp";
        std::string outputPrefix = "";
        bool writeToFile = true;
        bool printStatistics = true;

        int preferredBlockSize = 256;
        bool useOptimalOccupancy = false;

        bool enableDAWSimulation = false;
        int dawSleepMs = 90;

        static BenchmarkConfig fromCommandLine(int argc, char** argv) {
            BenchmarkConfig config;

            for (int i = 1; i < argc; i++) {
                std::string arg(argv[i]);
                bool hasNext = (i + 1) < argc;

                if (arg == "--buffersize" && hasNext) {
                    config.bufferSize = std::atoi(argv[++i]);
                } else if (arg == "--ntracks" && hasNext) {
                    config.trackCount = std::atoi(argv[++i]);
                } else if (arg == "--fs" && hasNext) {
                    config.sampleRate = std::atoi(argv[++i]);
                } else if (arg == "--nruns" && hasNext) {
                    config.iterations = std::atoi(argv[++i]);
                } else if (arg == "--warmup" && hasNext) {
                    config.warmupIterations = std::atoi(argv[++i]);
                } else if (arg == "--validate") {
                    config.enableValidation = true;
                } else if (arg == "--profile") {
                    config.enableProfiling = true;
                } else if (arg == "--verbose") {
                    config.verboseOutput = true;
                } else if (arg == "--output" && hasNext) {
                    config.outputDirectory = argv[++i];
                } else if (arg == "--prefix" && hasNext) {
                    config.outputPrefix = argv[++i];
                } else if (arg == "--no-file") {
                    config.writeToFile = false;
                } else if (arg == "--quiet") {
                    config.printStatistics = false;
                } else if (arg == "--blocksize" && hasNext) {
                    config.preferredBlockSize = std::atoi(argv[++i]);
                } else if (arg == "--optimal-occupancy") {
                    config.useOptimalOccupancy = true;
                } else if (arg == "--dawsim") {
                    config.enableDAWSimulation = true;
                }
            }

            return config;
        }

        bool validate() const {
            if (bufferSize <= 0 || bufferSize > 8192) {
                std::cerr << "Invalid buffer size: " << bufferSize << std::endl;
                return false;
            }
            if (trackCount <= 0 || trackCount > 2048) {
                std::cerr << "Invalid track count: " << trackCount << std::endl;
                return false;
            }
            if (iterations <= 0 || iterations > 10000) {
                std::cerr << "Invalid iterations: " << iterations << std::endl;
                return false;
            }
            if (preferredBlockSize < 32 || preferredBlockSize > 1024) {
                std::cerr << "Invalid block size: " << preferredBlockSize << std::endl;
                return false;
            }
            return true;
        }

        void print() const {
            std::cout << "=== Benchmark Configuration ===" << std::endl;
            std::cout << "Buffer Size: " << bufferSize << " samples" << std::endl;
            std::cout << "Track Count: " << trackCount << std::endl;
            std::cout << "Sample Rate: " << sampleRate << " Hz" << std::endl;
            std::cout << "Iterations: " << iterations << std::endl;
            std::cout << "Warmup: " << warmupIterations << std::endl;
            std::cout << "Validation: " << (enableValidation ? "ON" : "OFF") << std::endl;
            std::cout << "Profiling: " << (enableProfiling ? "ON" : "OFF") << std::endl;
            std::cout << "===============================" << std::endl;
        }
    };

    // ============================================================================
    // Memory Management Utilities
    // ============================================================================

    template<typename T>
    T* allocateDeviceBuffer(size_t count, const std::string& name = "device buffer");

    template<typename T>
    T* allocateHostBuffer(size_t count, const std::string& name = "host buffer");

    template<typename T>
    void copyToDevice(T* dst, const T* src, size_t count);

    template<typename T>
    void copyToHost(T* dst, const T* src, size_t count);

    void freeDeviceBuffers(std::initializer_list<void*> buffers);

    void freeHostBuffers(std::initializer_list<void*> buffers);

    // ============================================================================
    // Timing Utilities
    // ============================================================================

    class BenchmarkTimer {
    private:
        std::chrono::high_resolution_clock::time_point start_time;
        std::chrono::high_resolution_clock::time_point end_time;
        bool is_running = false;

    public:
        void start();

        void stop();

        double elapsed_ms() const;

        static double measureKernel(std::function<void()> kernel);

        void reset();
    };

    class CudaEventTimer {
    public:
        CudaEventTimer();
        ~CudaEventTimer();

        CudaEventTimer(const CudaEventTimer&) = delete;
        CudaEventTimer& operator=(const CudaEventTimer&) = delete;

        CudaEventTimer(CudaEventTimer&& other) noexcept;
        CudaEventTimer& operator=(CudaEventTimer&& other) noexcept;

        void start(cudaStream_t stream = 0);

        float stop(cudaStream_t stream = 0);

        void reset();

        bool isRunning() const { return running; }

    private:
        cudaEvent_t start_event = nullptr;
        cudaEvent_t stop_event = nullptr;
        bool running = false;

        void destroy();
    };

    void collectLatencies(std::vector<float>& latencies,
                          std::function<void()> benchmark,
                          int iterations);

    // ============================================================================
    // Data Generation Utilities
    // ============================================================================

    void generateRandomAudioData(float* buffer, size_t samples, unsigned int seed = 42);

    enum class WindowType {
        RECTANGULAR,
        HAMMING,
        HANN,
        BLACKMAN
    };

    void generateImpulseResponse(float* buffer, int length, float frequency,
                                 WindowType window_type = WindowType::HAMMING);

    enum class TestPattern {
        ZEROS,
        ONES,
        RAMP,
        SINE_WAVE,
        WHITE_NOISE
    };

    void initializeTestPattern(float* buffer, size_t samples, TestPattern pattern);

    struct BiquadCoefficients {
        float b0, b1, b2;  // Numerator coefficients
        float a1, a2;      // Denominator coefficients (a0 = 1)
    };

    BiquadCoefficients generateLowpassCoefficients(float cutoff_freq, float q = 0.707f);

    // ============================================================================
    // Error Handling Utilities
    // ============================================================================

    void checkCudaError(cudaError_t error, const std::string& message);

#define CUDA_CHECK(call) \
        do { \
            cudaError_t error = call; \
            if (error != cudaSuccess) { \
                BenchmarkUtils::checkCudaError(error, #call); \
            } \
        } while(0)

    // ============================================================================
    // Kernel Launch Utilities
    // ============================================================================

    template<typename KernelFunc, typename... Args>
    void launchKernel(KernelFunc kernel,
                      dim3 gridDim,
                      dim3 blockDim,
                      Args&&... args) {
        // Launch kernel
        kernel<<<gridDim, blockDim>>>(std::forward<Args>(args)...);

        // Check for kernel launch errors
        cudaError_t launchError = cudaGetLastError();
        if (launchError != cudaSuccess) {
            throw std::runtime_error("CUDA kernel launch failed: " +
                                   std::string(cudaGetErrorString(launchError)));
        }

        // Synchronize and check for execution errors
        cudaError_t syncError = cudaDeviceSynchronize();
        if (syncError != cudaSuccess) {
            throw std::runtime_error("CUDA kernel execution failed: " +
                                   std::string(cudaGetErrorString(syncError)));
        }
    }

    template<typename KernelFunc, typename... Args>
    void launchKernel1D(KernelFunc kernel,
                        size_t totalThreads,
                        int preferredBlockSize = 256,
                        Args&&... args) {
        // Calculate optimal grid dimensions
        int blockSize = std::min(preferredBlockSize, static_cast<int>(totalThreads));
        blockSize = std::max(blockSize, 32); // Minimum warp size

        int gridSize = static_cast<int>((totalThreads + blockSize - 1) / blockSize);

        // Launch kernel
        launchKernel(kernel, dim3(gridSize), dim3(blockSize), std::forward<Args>(args)...);
    }

    template<typename KernelFunc, typename... Args>
    void launchKernelOptimal(KernelFunc kernel,
                             size_t totalThreads,
                             Args&&... args) {
        int minGridSize, blockSize;

        // Use CUDA occupancy calculator
        cudaError_t err = cudaOccupancyMaxPotentialBlockSize(
            &minGridSize, &blockSize, kernel, 0, 0);

        if (err != cudaSuccess) {
            // Fallback to default if occupancy calculation fails
            launchKernel1D(kernel, totalThreads, 256, std::forward<Args>(args)...);
            return;
        }

        int gridSize = static_cast<int>((totalThreads + blockSize - 1) / blockSize);

        // Launch kernel with optimal parameters
        launchKernel(kernel, dim3(gridSize), dim3(blockSize), std::forward<Args>(args)...);
    }

    template<typename KernelFunc, typename... Args>
    double launchKernelTimed(KernelFunc kernel,
                             dim3 gridDim,
                             dim3 blockDim,
                             Args&&... args) {
        CudaEventTimer timer;
        timer.start();
        launchKernel(kernel, gridDim, blockDim, std::forward<Args>(args)...);
        return static_cast<double>(timer.stop());
    }

    // ============================================================================
    // Statistics Utilities
    // ============================================================================

    struct Statistics {
        float mean;
        float median;
        float std_dev;
        float min_val;
        float max_val;
        float p95;
        float p99;
        size_t count;
    };

    Statistics calculateStatistics(const std::vector<float>& latencies);

    void writeLatenciesToFile(const std::vector<float>& latencies, const std::string& filename);

    void printStatistics(const std::vector<float>& latencies, const std::string& benchmark_name);

} // namespace BenchmarkUtils
