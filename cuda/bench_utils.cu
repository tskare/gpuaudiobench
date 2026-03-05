#include "bench_utils.cuh"
#include <cufft.h>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <random>
#include <fstream>
#include <iomanip>
#include <utility>

namespace BenchmarkUtils {

    // ============================================================================
    // Shared Benchmark Parameters
    // ============================================================================

    BenchmarkParams makeBenchmarkParams(size_t bufferSize,
                                        size_t trackCount,
                                        float gainValue) {
        BenchmarkParams params;
        params.bufferSize = static_cast<uint32_t>(bufferSize);
        params.trackCount = static_cast<uint32_t>(trackCount);
        params.totalSamples = static_cast<uint32_t>(bufferSize * trackCount);
        params.gainValue = gainValue;
        return params;
    }

    CudaEventTimer::CudaEventTimer() {
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));
    }

    CudaEventTimer::~CudaEventTimer() {
        destroy();
    }

    CudaEventTimer::CudaEventTimer(CudaEventTimer&& other) noexcept {
        start_event = other.start_event;
        stop_event = other.stop_event;
        running = other.running;

        other.start_event = nullptr;
        other.stop_event = nullptr;
        other.running = false;
    }

    CudaEventTimer& CudaEventTimer::operator=(CudaEventTimer&& other) noexcept {
        if (this != &other) {
            destroy();
            start_event = other.start_event;
            stop_event = other.stop_event;
            running = other.running;

            other.start_event = nullptr;
            other.stop_event = nullptr;
            other.running = false;
        }
        return *this;
    }

    void CudaEventTimer::destroy() {
        if (start_event) {
            cudaEventDestroy(start_event);
            start_event = nullptr;
        }
        if (stop_event) {
            cudaEventDestroy(stop_event);
            stop_event = nullptr;
        }
        running = false;
    }

    void CudaEventTimer::start(cudaStream_t stream) {
        CUDA_CHECK(cudaEventRecord(start_event, stream));
        running = true;
    }

    float CudaEventTimer::stop(cudaStream_t stream) {
        if (!running) {
            return 0.0f;
        }

        CUDA_CHECK(cudaEventRecord(stop_event, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_event));

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));

        running = false;
        return elapsed_ms;
    }

    void CudaEventTimer::reset() {
        running = false;
    }

    // ============================================================================
    // Memory Management Implementation
    // ============================================================================

    void freeDeviceBuffers(std::initializer_list<void*> buffers) {
        for (auto* buffer : buffers) {
            if (buffer != nullptr) {
                cudaFree(buffer);
            }
        }
    }

    void freeHostBuffers(std::initializer_list<void*> buffers) {
        for (auto* buffer : buffers) {
            if (buffer != nullptr) {
                cudaFreeHost(buffer);
            }
        }
    }


    // ============================================================================
    // Timing Implementation
    // ============================================================================

    void BenchmarkTimer::start() {
        start_time = std::chrono::high_resolution_clock::now();
        is_running = true;
    }

    void BenchmarkTimer::stop() {
        end_time = std::chrono::high_resolution_clock::now();
        is_running = false;
    }

    double BenchmarkTimer::elapsed_ms() const {
        if (is_running) {
            auto current = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                current - start_time);
            return duration.count() / 1000.0;
        } else {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                end_time - start_time);
            return duration.count() / 1000.0;
        }
    }

    double BenchmarkTimer::measureKernel(std::function<void()> kernel) {
        BenchmarkTimer timer;
        timer.start();
        kernel();
        timer.stop();
        return timer.elapsed_ms();
    }

    void BenchmarkTimer::reset() {
        is_running = false;
    }

    void collectLatencies(std::vector<float>& latencies,
                         std::function<void()> benchmark,
                         int iterations) {
        latencies.clear();
        latencies.reserve(iterations);

        for (int i = 0; i < iterations; ++i) {
            double latency = BenchmarkTimer::measureKernel(benchmark);
            latencies.push_back(static_cast<float>(latency));
        }
    }

    // ============================================================================
    // Data Generation Implementation
    // ============================================================================

    void generateRandomAudioData(float* buffer, size_t samples, unsigned int seed) {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        for (size_t i = 0; i < samples; ++i) {
            buffer[i] = dist(gen);
        }
    }

    void generateImpulseResponse(float* buffer, int length, float frequency,
                                WindowType window_type) {
        // Generate windowed sinc impulse response
        for (int i = 0; i < length; ++i) {
            float t = static_cast<float>(i) - static_cast<float>(length) / 2.0f;

            // Sinc function
            float sinc_val;
            if (t == 0.0f) {
                sinc_val = 1.0f;
            } else {
                float arg = 2.0f * M_PI * frequency * t;
                sinc_val = std::sin(arg) / arg;
            }

            // Apply window function
            float window_val = 1.0f;
            float n = static_cast<float>(i) / static_cast<float>(length - 1);

            switch (window_type) {
                case WindowType::RECTANGULAR:
                    window_val = 1.0f;
                    break;
                case WindowType::HAMMING:
                    window_val = 0.54f - 0.46f * std::cos(2.0f * M_PI * n);
                    break;
                case WindowType::HANN:
                    window_val = 0.5f * (1.0f - std::cos(2.0f * M_PI * n));
                    break;
                case WindowType::BLACKMAN:
                    window_val = 0.42f - 0.5f * std::cos(2.0f * M_PI * n) +
                                0.08f * std::cos(4.0f * M_PI * n);
                    break;
            }

            buffer[i] = sinc_val * window_val;
        }

        // Normalize
        float sum = 0.0f;
        for (int i = 0; i < length; ++i) {
            sum += std::abs(buffer[i]);
        }
        if (sum > 0.0f) {
            for (int i = 0; i < length; ++i) {
                buffer[i] /= sum;
            }
        }
    }

    void initializeTestPattern(float* buffer, size_t samples, TestPattern pattern) {
        switch (pattern) {
            case TestPattern::ZEROS:
                std::fill(buffer, buffer + samples, 0.0f);
                break;

            case TestPattern::ONES:
                std::fill(buffer, buffer + samples, 1.0f);
                break;

            case TestPattern::RAMP:
                for (size_t i = 0; i < samples; ++i) {
                    buffer[i] = static_cast<float>(i) / static_cast<float>(samples - 1);
                }
                break;

            case TestPattern::SINE_WAVE:
                for (size_t i = 0; i < samples; ++i) {
                    buffer[i] = std::sin(2.0f * M_PI * static_cast<float>(i) / 64.0f);
                }
                break;

            case TestPattern::WHITE_NOISE:
                generateRandomAudioData(buffer, samples, 42);
                break;
        }
    }

    BiquadCoefficients generateLowpassCoefficients(float cutoff_freq, float q) {
        // Butterworth lowpass design using bilinear transform
        float omega = 2.0f * M_PI * cutoff_freq;
        float sin_omega = std::sin(omega);
        float cos_omega = std::cos(omega);
        float alpha = sin_omega / (2.0f * q);

        float a0 = 1.0f + alpha;

        BiquadCoefficients coeffs;
        coeffs.b0 = ((1.0f - cos_omega) / 2.0f) / a0;
        coeffs.b1 = (1.0f - cos_omega) / a0;
        coeffs.b2 = ((1.0f - cos_omega) / 2.0f) / a0;
        coeffs.a1 = (-2.0f * cos_omega) / a0;
        coeffs.a2 = (1.0f - alpha) / a0;

        return coeffs;
    }

    // ============================================================================
    // Error Handling Implementation
    // ============================================================================

    void checkCudaError(cudaError_t error, const std::string& message) {
        if (error != cudaSuccess) {
            throw std::runtime_error(message + ": " + cudaGetErrorString(error));
        }
    }

    // ============================================================================
    // Statistics Implementation
    // ============================================================================

    Statistics calculateStatistics(const std::vector<float>& latencies) {
        if (latencies.empty()) {
            return {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0};
        }

        Statistics stats;
        stats.count = latencies.size();

        // Create sorted copy for percentile calculations
        std::vector<float> sorted_latencies = latencies;
        std::sort(sorted_latencies.begin(), sorted_latencies.end());

        // Min and max
        stats.min_val = sorted_latencies.front();
        stats.max_val = sorted_latencies.back();

        // Mean
        float sum = std::accumulate(latencies.begin(), latencies.end(), 0.0f);
        stats.mean = sum / static_cast<float>(latencies.size());

        // Median
        size_t mid = latencies.size() / 2;
        if (latencies.size() % 2 == 0) {
            stats.median = (sorted_latencies[mid - 1] + sorted_latencies[mid]) / 2.0f;
        } else {
            stats.median = sorted_latencies[mid];
        }

        // Standard deviation
        float variance = 0.0f;
        for (float latency : latencies) {
            float diff = latency - stats.mean;
            variance += diff * diff;
        }
        variance /= static_cast<float>(latencies.size() - 1);
        stats.std_dev = std::sqrt(variance);

        // Percentiles
        auto percentile = [&](float p) -> float {
            float index = p / 100.0f * static_cast<float>(sorted_latencies.size() - 1);
            size_t lower = static_cast<size_t>(std::floor(index));
            size_t upper = static_cast<size_t>(std::ceil(index));

            if (lower == upper) {
                return sorted_latencies[lower];
            } else {
                float weight = index - static_cast<float>(lower);
                return sorted_latencies[lower] * (1.0f - weight) +
                       sorted_latencies[upper] * weight;
            }
        };

        stats.p95 = percentile(95.0f);
        stats.p99 = percentile(99.0f);

        return stats;
    }

    void writeLatenciesToFile(const std::vector<float>& latencies, const std::string& filename) {
        Statistics stats = calculateStatistics(latencies);

        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file for writing: " + filename);
        }

        // Write summary statistics
        file << std::fixed << std::setprecision(3);
        file << "# Latency Statistics (ms)\n";
        file << "# Count: " << stats.count << "\n";
        file << "# Mean: " << stats.mean << "\n";
        file << "# Median: " << stats.median << "\n";
        file << "# Std Dev: " << stats.std_dev << "\n";
        file << "# Min: " << stats.min_val << "\n";
        file << "# Max: " << stats.max_val << "\n";
        file << "# P95: " << stats.p95 << "\n";
        file << "# P99: " << stats.p99 << "\n";
        file << "#\n";
        file << "# Raw latencies:\n";

        // Write raw latencies
        for (float latency : latencies) {
            file << latency << "\n";
        }
    }

    void printStatistics(const std::vector<float>& latencies, const std::string& benchmark_name) {
        Statistics stats = calculateStatistics(latencies);

        std::cout << "\n=== " << benchmark_name << " Benchmark Results ===\n";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Iterations: " << stats.count << "\n";
        std::cout << "Mean:       " << stats.mean << " ms\n";
        std::cout << "Median:     " << stats.median << " ms\n";
        std::cout << "Std Dev:    " << stats.std_dev << " ms\n";
        std::cout << "Min:        " << stats.min_val << " ms\n";
        std::cout << "Max:        " << stats.max_val << " ms\n";
        std::cout << "P95:        " << stats.p95 << " ms\n";
        std::cout << "P99:        " << stats.p99 << " ms\n";
        std::cout << "==========================================\n\n";
    }

} // namespace BenchmarkUtils
