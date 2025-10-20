#include "bench_fft.cuh"
#include <cstring>
#include <cmath>

// ============================================================================
// FFTBenchmark Implementation
// ============================================================================

FFTBenchmark::FFTBenchmark(size_t buffer_size, size_t track_count)
    : GPUABenchmark("FFT1D", buffer_size, track_count) {

    // Calculate FFT buffer sizes - expand to FFT_SIZE for proper transform
    input_fft_size = track_count * FFT_SIZE;  // Total real input elements (zero-padded if needed)
    output_fft_size = track_count * (FFT_SIZE / 2 + 1);  // Complex output elements
    input_fft_bytes = input_fft_size * sizeof(float);
    output_fft_bytes = output_fft_size * sizeof(cufftComplex);
}

FFTBenchmark::~FFTBenchmark() {
    destroyFFTPlan();
    cleanupFFTBuffers();
    BenchmarkUtils::freeHostBuffers({cpu_reference_real, cpu_reference_imag});
    cpu_reference_real = nullptr;
    cpu_reference_imag = nullptr;
}

void FFTBenchmark::setupBenchmark() {
    // Note: FFT size is fixed at 1024; smaller buffers are zero-padded
    // allocateBuffers() not used - FFT requires special buffer sizing

    allocateFFTBuffers();
    createFFTPlan();

    size_t samples_per_track = std::min(getBufferSize(), static_cast<size_t>(FFT_SIZE));
    for (size_t track = 0; track < getTrackCount(); ++track) {
        for (size_t i = 0; i < samples_per_track; ++i) {
            h_input_fft[track * FFT_SIZE + i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        }
        // Zero-pad if buffer_size < FFT_SIZE
        for (size_t i = samples_per_track; i < FFT_SIZE; ++i) {
            h_input_fft[track * FFT_SIZE + i] = 0.0f;
        }
    }

    cpu_reference_real = BenchmarkUtils::allocateHostBuffer<float>(
        output_fft_size, "fft cpu reference real");
    cpu_reference_imag = BenchmarkUtils::allocateHostBuffer<float>(
        output_fft_size, "fft cpu reference imag");

    calculateCPUReference();

    printf("FFT benchmark setup complete (FFT size = %d, %zu tracks)\n",
           FFT_SIZE, getTrackCount());
}

void FFTBenchmark::runKernel() {
    performBenchmarkIteration();
}

void FFTBenchmark::performBenchmarkIteration() {
    CUDA_CHECK(cudaMemcpy(d_input_fft, h_input_fft, input_fft_bytes, cudaMemcpyHostToDevice));

    cufftResult result = cufftExecR2C(fft_plan, d_input_fft, d_output_fft);
    if (result != CUFFT_SUCCESS) {
        throw std::runtime_error("cuFFT execution failed");
    }

    synchronizeAndCheck();

    CUDA_CHECK(cudaMemcpy(h_output_fft, d_output_fft, output_fft_bytes, cudaMemcpyDeviceToHost));
}

void FFTBenchmark::validate(ValidationData& validation_data) {
    float max_error = 0.0f;
    float mean_error = 0.0f;

    for (size_t i = 0; i < output_fft_size; ++i) {
        float real_error = std::abs(h_output_fft[i].x - cpu_reference_real[i]);
        float imag_error = std::abs(h_output_fft[i].y - cpu_reference_imag[i]);

        float total_error = real_error + imag_error;
        max_error = std::max(max_error, total_error);
        mean_error += total_error;
    }
    mean_error /= (2.0f * output_fft_size);

    validation_data.max_error = max_error;
    validation_data.mean_error = mean_error;

    // FFT validation uses looser tolerance due to numerical precision
    if (max_error > 1e-3f) {
        validation_data.status = ValidationStatus::FAILURE;
        validation_data.messages.push_back("FFT validation failed");
    } else {
        validation_data.status = ValidationStatus::SUCCESS;
        validation_data.messages.push_back("FFT validation passed");
    }
}

// ============================================================================
// Private Helper Methods
// ============================================================================

void FFTBenchmark::createFFTPlan() {
    cufftResult result = cufftPlan1d(&fft_plan, FFT_SIZE, CUFFT_R2C, static_cast<int>(getTrackCount()));
    if (result != CUFFT_SUCCESS) {
        throw std::runtime_error("Failed to create cuFFT plan");
    }
    plan_created = true;
}

void FFTBenchmark::destroyFFTPlan() {
    if (plan_created) {
        cufftDestroy(fft_plan);
        plan_created = false;
    }
}

void FFTBenchmark::allocateFFTBuffers() {
    h_input_fft = BenchmarkUtils::allocateHostBuffer<float>(
        input_fft_size, benchmark_name_ + " host input FFT buffer");
    h_output_fft = BenchmarkUtils::allocateHostBuffer<cufftComplex>(
        output_fft_size, benchmark_name_ + " host output FFT buffer");

    d_input_fft = BenchmarkUtils::allocateDeviceBuffer<float>(
        input_fft_size, benchmark_name_ + " device input FFT buffer");
    d_output_fft = BenchmarkUtils::allocateDeviceBuffer<cufftComplex>(
        output_fft_size, benchmark_name_ + " device output FFT buffer");

    memset(h_output_fft, 0, output_fft_bytes);
    CUDA_CHECK(cudaMemset(d_output_fft, 0, output_fft_bytes));
}

void FFTBenchmark::calculateCPUReference() {
    size_t elements_per_track = FFT_SIZE / 2 + 1;
    for (size_t track = 0; track < getTrackCount(); ++track) {
        size_t input_offset = track * FFT_SIZE;
        size_t output_offset = track * elements_per_track;

        cpuFFTReference(
            h_input_fft + input_offset,
            cpu_reference_real + output_offset,
            cpu_reference_imag + output_offset,
            FFT_SIZE
        );
    }
}

void FFTBenchmark::cpuFFTReference(const float* input, float* real_output, float* imag_output, int size) {
    const float PI = 3.14159265358979323846f;

    for (int k = 0; k < size / 2 + 1; k++) {
        float sum_real = 0.0f;
        float sum_imag = 0.0f;

        for (int n = 0; n < size; n++) {
            float angle = -2.0f * PI * k * n / size;
            float cos_val = cosf(angle);
            float sin_val = sinf(angle);

            sum_real += input[n] * cos_val;
            sum_imag += input[n] * sin_val;
        }

        real_output[k] = sum_real;
        imag_output[k] = sum_imag;
    }
}

void FFTBenchmark::cleanupFFTBuffers() {
    BenchmarkUtils::freeHostBuffers({h_input_fft, reinterpret_cast<float*>(h_output_fft)});
    BenchmarkUtils::freeDeviceBuffers({d_input_fft, reinterpret_cast<float*>(d_output_fft)});
    h_input_fft = nullptr;
    h_output_fft = nullptr;
    d_input_fft = nullptr;
    d_output_fft = nullptr;
}
