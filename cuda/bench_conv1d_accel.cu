#include "bench_conv1d_accel.cuh"
#include <chrono>
#include <thread>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <string>

__global__ void ComplexMultiplyKernel(
    cufftComplex* input_fft,
    const cufftComplex* ir_fft,
    cufftComplex* output_fft,
    int fft_size,
    int num_tracks) {

    int track_idx = blockIdx.x;
    int freq_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (track_idx >= num_tracks || freq_idx >= fft_size) return;

    int idx = track_idx * fft_size + freq_idx;

    float a = input_fft[idx].x;
    float b = input_fft[idx].y;
    float c = ir_fft[idx].x;
    float d = ir_fft[idx].y;

    output_fft[idx].x = a * c - b * d;
    output_fft[idx].y = a * d + b * c;
}

__global__ void ExtractRealPartKernel(
    const cufftComplex* fft_output,
    float* real_output,
    int buffer_size,
    int num_tracks) {

    int track_idx = blockIdx.x;
    int sample_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (track_idx >= num_tracks || sample_idx >= buffer_size) return;

    int fft_idx = track_idx * buffer_size + sample_idx;
    int output_idx = num_tracks * sample_idx + track_idx;

    real_output[output_idx] = fft_output[fft_idx].x;
}

Conv1DAccelBenchmark::Conv1DAccelBenchmark(int ir_length, size_t buffer_size, size_t track_count)
    : GPUABenchmark("Conv1D_accel", buffer_size, track_count)
    , ir_length_(ir_length)
    , fft_size_(1 << int(ceil(log2(ir_length + buffer_size - 1))))  // Next power of 2
    , overlap_size_(ir_length - 1) {

    printf("Conv1DAccelBenchmark: IR length = %d, FFT size = %d\n", ir_length_, fft_size_);

    ir_buffer_size = track_count * ir_length;
    ir_buffer_bytes = ir_buffer_size * sizeof(float);
    fft_buffer_size = track_count * fft_size_;
    fft_buffer_bytes = fft_buffer_size * sizeof(cufftComplex);
}

Conv1DAccelBenchmark::~Conv1DAccelBenchmark() {
    cleanupAccelBuffers();
    cleanupFFTPlans();
}

void Conv1DAccelBenchmark::setupBenchmark() {
    printf("Setting up Conv1D accelerated benchmark...\n");

    allocateBuffers(getTotalElements());

    generateTestData(42);

    allocateAccelBuffers();

    setupFFTPlans();

    generateImpulseResponses();

    precomputeImpulseResponseFFTs();

    calculateCPUReference();

    printf("Conv1D accelerated benchmark setup complete.\n");
}

void Conv1DAccelBenchmark::allocateAccelBuffers() {
    cudaError_t err = cudaSuccess;

    h_ir_buf = BenchmarkUtils::allocateHostBuffer<float>(
        ir_buffer_size, "conv1d_accel host IR buffer");

    err = cudaMalloc((void**)&d_ir_buf, ir_buffer_bytes);
    if (err != cudaSuccess) {
        cleanupAccelBuffers();
        cleanupFFTPlans();
        throw std::runtime_error(std::string("Failed to allocate device IR buffer: ") + cudaGetErrorString(err));
    }

    err = cudaMalloc((void**)&d_fft_input, fft_buffer_bytes);
    if (err != cudaSuccess) {
        cleanupAccelBuffers();
        cleanupFFTPlans();
        throw std::runtime_error(std::string("Failed to allocate FFT input buffer: ") + cudaGetErrorString(err));
    }

    err = cudaMalloc((void**)&d_fft_output, fft_buffer_bytes);
    if (err != cudaSuccess) {
        cleanupAccelBuffers();
        cleanupFFTPlans();
        throw std::runtime_error(std::string("Failed to allocate FFT output buffer: ") + cudaGetErrorString(err));
    }

    err = cudaMalloc((void**)&d_ir_fft, fft_buffer_bytes);
    if (err != cudaSuccess) {
        cleanupAccelBuffers();
        cleanupFFTPlans();
        throw std::runtime_error(std::string("Failed to allocate IR FFT buffer: ") + cudaGetErrorString(err));
    }

    input_padded_bytes = getTrackCount() * fft_size_ * sizeof(float);
    err = cudaMalloc((void**)&d_input_padded, input_padded_bytes);
    if (err != cudaSuccess) {
        cleanupAccelBuffers();
        cleanupFFTPlans();
        throw std::runtime_error(std::string("Failed to allocate padded input buffer: ") + cudaGetErrorString(err));
    }

    cpu_reference = BenchmarkUtils::allocateHostBuffer<float>(
        getTotalElements(), "conv1d_accel cpu reference");
}

void Conv1DAccelBenchmark::setupFFTPlans() {
    cufftResult cufft_err;

    cufft_err = cufftPlan1d(&forward_plan, fft_size_, CUFFT_R2C, static_cast<int>(getTrackCount()));
    if (cufft_err != CUFFT_SUCCESS) {
        cleanupFFTPlans();
        cleanupAccelBuffers();
        throw std::runtime_error("Failed to create forward cuFFT plan");
    }

    cufft_err = cufftPlan1d(&inverse_plan, fft_size_, CUFFT_C2R, static_cast<int>(getTrackCount()));
    if (cufft_err != CUFFT_SUCCESS) {
        cleanupFFTPlans();
        cleanupAccelBuffers();
        throw std::runtime_error("Failed to create inverse cuFFT plan");
    }
}

void Conv1DAccelBenchmark::generateImpulseResponses() {
    for (size_t trackIdx = 0; trackIdx < getTrackCount(); trackIdx++) {
        for (int irIdx = 0; irIdx < ir_length_; irIdx++) {
            int idx = trackIdx * ir_length_ + irIdx;

            float freq = 0.1f + 0.05f * (float)trackIdx / (float)getTrackCount();
            float t = (float)irIdx - (float)ir_length_ / 2.0f;
            float window = 0.54f - 0.46f * cosf(2.0f * M_PI * (float)irIdx / (float)(ir_length_ - 1));
            float sinc = (t == 0.0f) ? 1.0f : sinf(2.0f * M_PI * freq * t) / (2.0f * M_PI * freq * t);
            float value = window * sinc / (float)ir_length_;

            h_ir_buf[idx] = value;
        }
    }

    cudaError_t err = cudaMemcpy(d_ir_buf, h_ir_buf, ir_buffer_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cleanupFFTPlans();
        cleanupAccelBuffers();
        throw std::runtime_error(std::string("Failed to copy IR buffer to device: ") + cudaGetErrorString(err));
    }
}

void Conv1DAccelBenchmark::precomputeImpulseResponseFFTs() {
    cudaError_t err = cudaSuccess;
    cufftResult cufft_err;

    float* d_ir_padded;
    size_t padded_size = getTrackCount() * fft_size_ * sizeof(float);
    err = cudaMalloc((void**)&d_ir_padded, padded_size);
    if (err != cudaSuccess) {
        cleanupFFTPlans();
        cleanupAccelBuffers();
        throw std::runtime_error(std::string("Failed to allocate padded IR buffer: ") + cudaGetErrorString(err));
    }

    err = cudaMemset(d_ir_padded, 0, padded_size);
    if (err != cudaSuccess) {
        cudaFree(d_ir_padded);
        cleanupFFTPlans();
        cleanupAccelBuffers();
        throw std::runtime_error(std::string("Failed to zero IR buffer: ") + cudaGetErrorString(err));
    }

    for (size_t trackIdx = 0; trackIdx < getTrackCount(); trackIdx++) {
        size_t src_offset = trackIdx * ir_length_ * sizeof(float);
        size_t dst_offset = trackIdx * fft_size_ * sizeof(float);
        err = cudaMemcpy((char*)d_ir_padded + dst_offset,
                        (char*)d_ir_buf + src_offset,
                        ir_length_ * sizeof(float),
                        cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
            cudaFree(d_ir_padded);
            cleanupFFTPlans();
            cleanupAccelBuffers();
            throw std::runtime_error(std::string("Failed to copy IR to padded buffer: ") + cudaGetErrorString(err));
        }
    }

    cufft_err = cufftExecR2C(forward_plan, d_ir_padded, d_ir_fft);
    if (cufft_err != CUFFT_SUCCESS) {
        cudaFree(d_ir_padded);
        cleanupFFTPlans();
        cleanupAccelBuffers();
        throw std::runtime_error("Failed to execute IR FFT");
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_ir_padded);
        cleanupFFTPlans();
        cleanupAccelBuffers();
        throw std::runtime_error(std::string("Failed to synchronize after IR FFT: ") + cudaGetErrorString(err));
    }

    cudaFree(d_ir_padded);
}

void Conv1DAccelBenchmark::calculateCPUReference() {
    conv1DCPUReference(getHostInput(), h_ir_buf, cpu_reference, ir_length_, getBufferSize(), getTrackCount());
}

void Conv1DAccelBenchmark::conv1DCPUReference(const float* input, const float* impulse_response,
                                             float* output, int ir_len, int buffer_size, int track_count) {
    for (int trackIdx = 0; trackIdx < track_count; trackIdx++) {
        for (int sampleIdx = 0; sampleIdx < buffer_size; sampleIdx++) {
            float outputSample = 0.0f;

            for (int irIdx = 0; irIdx < ir_len; irIdx++) {
                int inputIdx = sampleIdx - irIdx;
                if (inputIdx >= 0 && inputIdx < buffer_size) {
                    float inputValue = input[trackIdx * buffer_size + inputIdx];
                    float irValue = impulse_response[trackIdx * ir_len + irIdx];
                    outputSample += inputValue * irValue;
                }
            }

            output[track_count * sampleIdx + trackIdx] = outputSample;
        }
    }
}

void Conv1DAccelBenchmark::runKernel() {
    performBenchmarkIteration();
}

void Conv1DAccelBenchmark::performBenchmarkIteration() {
    if (!d_input_padded) {
        throw std::runtime_error("Padded input buffer not initialized");
    }

    transferToDevice();

    CUDA_CHECK(cudaMemset(d_input_padded, 0, input_padded_bytes));

    for (size_t trackIdx = 0; trackIdx < getTrackCount(); trackIdx++) {
        size_t src_offset = trackIdx * getBufferSize() * sizeof(float);
        size_t dst_offset = trackIdx * fft_size_ * sizeof(float);
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<char*>(d_input_padded) + dst_offset,
                              reinterpret_cast<char*>(getDeviceInput()) + src_offset,
                              getBufferSize() * sizeof(float),
                              cudaMemcpyDeviceToDevice));
    }

    cufftResult cufft_err = cufftExecR2C(forward_plan, d_input_padded, d_fft_input);
    if (cufft_err != CUFFT_SUCCESS) {
        throw std::runtime_error("Failed to execute input FFT");
    }

    dim3 blockSize(256);
    dim3 gridSize(static_cast<unsigned int>(getTrackCount()), (fft_size_ / 2 + 1 + blockSize.x - 1) / blockSize.x);

    ComplexMultiplyKernel<<<gridSize, blockSize>>>(
        d_fft_input, d_ir_fft, d_fft_output, fft_size_ / 2 + 1, static_cast<int>(getTrackCount()));

    CUDA_CHECK(cudaGetLastError());

    cufft_err = cufftExecC2R(inverse_plan, d_fft_output, d_input_padded);
    if (cufft_err != CUFFT_SUCCESS) {
        throw std::runtime_error("Failed to execute inverse FFT");
    }

    dim3 extractBlockSize(256);
    dim3 extractGridSize(static_cast<unsigned int>(getTrackCount()), (getBufferSize() + extractBlockSize.x - 1) / extractBlockSize.x);

    ExtractRealPartKernel<<<extractGridSize, extractBlockSize>>>(
        (cufftComplex*)d_input_padded, getDeviceOutput(), static_cast<int>(getBufferSize()), static_cast<int>(getTrackCount()));

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    transferToHost();
}

void Conv1DAccelBenchmark::validate(ValidationData& validation_data) {
    float maxError = 0.0f;
    float totalError = 0.0f;
    size_t totalSamples = getTotalElements();
    const float tolerance = 1e-3f; // Relaxed tolerance for FFT-based convolution

    for (size_t i = 0; i < totalSamples; i++) {
        float gpuValue = getHostOutput()[i];
        float cpuValue = cpu_reference[i];

        float error = fabsf(gpuValue - cpuValue);
        float relativeError = cpuValue != 0 ? error / fabsf(cpuValue) : error;

        maxError = fmaxf(maxError, relativeError);
        totalError += relativeError;
    }

    float meanError = totalError / totalSamples;

    validation_data.max_error = maxError;
    validation_data.mean_error = meanError;

    if (maxError < tolerance) {
        validation_data.status = ValidationStatus::SUCCESS;
        validation_data.messages.push_back("Conv1D Accel validation passed");
    } else {
        validation_data.status = ValidationStatus::FAILURE;
        validation_data.messages.push_back("Conv1D Accel validation failed (max_error=" +
                                          std::to_string(maxError) + ", tolerance=" +
                                          std::to_string(tolerance) + ")");
    }
}

void Conv1DAccelBenchmark::cleanupAccelBuffers() {
    BenchmarkUtils::freeHostBuffers({h_ir_buf, cpu_reference});
    h_ir_buf = nullptr;
    cpu_reference = nullptr;

    if (d_ir_buf) {
        cudaFree(d_ir_buf);
        d_ir_buf = nullptr;
    }
    if (d_fft_input) {
        cudaFree(d_fft_input);
        d_fft_input = nullptr;
    }
    if (d_fft_output) {
        cudaFree(d_fft_output);
        d_fft_output = nullptr;
    }
    if (d_ir_fft) {
        cudaFree(d_ir_fft);
        d_ir_fft = nullptr;
    }
    if (d_input_padded) {
        cudaFree(d_input_padded);
        d_input_padded = nullptr;
    }
}

void Conv1DAccelBenchmark::cleanupFFTPlans() {
    if (forward_plan) {
        cufftDestroy(forward_plan);
        forward_plan = 0;
    }
    if (inverse_plan) {
        cufftDestroy(inverse_plan);
        inverse_plan = 0;
    }
}

// ============================================================================
// Legacy Interface Wrapper
// ============================================================================

void RunConv1DAccelBenchmark() {
    printf("Running Conv1D Accelerated Benchmark...\n");

    Conv1DAccelBenchmark benchmark;
    benchmark.setupBenchmark();

    std::vector<float> latencies;
    benchmark.runBenchmarkIterations(latencies);

    // Validate results
    ValidationData validation;
    benchmark.validate(validation);

    // Print summary
    benchmark.printSummary(latencies, validation);
}
