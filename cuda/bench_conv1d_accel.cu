#include "bench_conv1d_accel.cuh"
#include <chrono>
#include <thread>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <string>

// ============================================================================
// CUDA Kernels
// ============================================================================

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

    // Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    float a = input_fft[idx].x;  // Real part of input FFT
    float b = input_fft[idx].y;  // Imaginary part of input FFT
    float c = ir_fft[idx].x;     // Real part of IR FFT
    float d = ir_fft[idx].y;     // Imaginary part of IR FFT

    output_fft[idx].x = a * c - b * d;  // Real part
    output_fft[idx].y = a * d + b * c;  // Imaginary part
}

__global__ void ExtractRealPartKernel(
    const cufftComplex* fft_output,
    float* real_output,
    int buffer_size,
    int num_tracks) {

    int track_idx = blockIdx.x;
    int sample_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (track_idx >= num_tracks || sample_idx >= buffer_size) return;

    // Extract real part from FFT output and store in interleaved format
    int fft_idx = track_idx * buffer_size + sample_idx; // Assuming fft_size >= buffer_size
    int output_idx = num_tracks * sample_idx + track_idx; // Interleaved output

    real_output[output_idx] = fft_output[fft_idx].x;
}

// ============================================================================
// Conv1DAccelBenchmark Implementation
// ============================================================================

Conv1DAccelBenchmark::Conv1DAccelBenchmark(int ir_length, size_t buffer_size, size_t track_count)
    : GPUABenchmark("Conv1D_accel", buffer_size, track_count)
    , ir_length_(ir_length)
    , fft_size_(1 << int(ceil(log2(ir_length + buffer_size - 1))))  // Next power of 2
    , overlap_size_(ir_length - 1) {

    printf("Conv1DAccelBenchmark: IR length = %d, FFT size = %d\n", ir_length_, fft_size_);

    // Calculate buffer sizes
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

    // Allocate base class buffers for input/output
    allocateBuffers(getTotalElements());

    // Generate test data
    generateTestData(42);

    // Allocate convolution-specific buffers
    allocateAccelBuffers();

    // Setup cuFFT plans
    setupFFTPlans();

    // Generate impulse responses
    generateImpulseResponses();

    // Pre-compute IR FFTs
    precomputeImpulseResponseFFTs();

    // Calculate CPU reference
    calculateCPUReference();

    printf("Conv1D accelerated benchmark setup complete.\n");
}

void Conv1DAccelBenchmark::allocateAccelBuffers() {
    cudaError_t err = cudaSuccess;

    // Allocate host impulse response buffer
    h_ir_buf = BenchmarkUtils::allocateHostBuffer<float>(
        ir_buffer_size, "conv1d_accel host IR buffer");

    // Allocate device impulse response buffer
    err = cudaMalloc((void**)&d_ir_buf, ir_buffer_bytes);
    if (err != cudaSuccess) {
        cleanupAccelBuffers();
        cleanupFFTPlans();
        throw std::runtime_error(std::string("Failed to allocate device IR buffer: ") + cudaGetErrorString(err));
    }

    // Allocate FFT buffers
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

    // Allocate CPU reference buffer
    cpu_reference = BenchmarkUtils::allocateHostBuffer<float>(
        getTotalElements(), "conv1d_accel cpu reference");
}

void Conv1DAccelBenchmark::setupFFTPlans() {
    cufftResult cufft_err;

    // Create forward FFT plan (R2C)
    cufft_err = cufftPlan1d(&forward_plan, fft_size_, CUFFT_R2C, static_cast<int>(getTrackCount()));
    if (cufft_err != CUFFT_SUCCESS) {
        cleanupFFTPlans();
        cleanupAccelBuffers();
        throw std::runtime_error("Failed to create forward cuFFT plan");
    }

    // Create inverse FFT plan (C2R)
    cufft_err = cufftPlan1d(&inverse_plan, fft_size_, CUFFT_C2R, static_cast<int>(getTrackCount()));
    if (cufft_err != CUFFT_SUCCESS) {
        cleanupFFTPlans();
        cleanupAccelBuffers();
        throw std::runtime_error("Failed to create inverse cuFFT plan");
    }
}

void Conv1DAccelBenchmark::generateImpulseResponses() {
    // Generate windowed sinc impulse responses for each track
    for (size_t trackIdx = 0; trackIdx < getTrackCount(); trackIdx++) {
        for (int irIdx = 0; irIdx < ir_length_; irIdx++) {
            int idx = trackIdx * ir_length_ + irIdx;

            // Create windowed sinc IR with frequency variation per track
            float freq = 0.1f + 0.05f * (float)trackIdx / (float)getTrackCount();
            float t = (float)irIdx - (float)ir_length_ / 2.0f;
            float window = 0.54f - 0.46f * cosf(2.0f * M_PI * (float)irIdx / (float)(ir_length_ - 1)); // Hamming window
            float sinc = (t == 0.0f) ? 1.0f : sinf(2.0f * M_PI * freq * t) / (2.0f * M_PI * freq * t);
            float value = window * sinc / (float)ir_length_; // Normalize

            h_ir_buf[idx] = value;
        }
    }

    // Copy to device
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

    // Allocate temporary buffer for IR preparation
    float* d_ir_padded;
    size_t padded_size = getTrackCount() * fft_size_ * sizeof(float);
    err = cudaMalloc((void**)&d_ir_padded, padded_size);
    if (err != cudaSuccess) {
        cleanupFFTPlans();
        cleanupAccelBuffers();
        throw std::runtime_error(std::string("Failed to allocate padded IR buffer: ") + cudaGetErrorString(err));
    }

    // Zero-pad impulse responses to FFT size
    err = cudaMemset(d_ir_padded, 0, padded_size);
    if (err != cudaSuccess) {
        cudaFree(d_ir_padded);
        cleanupFFTPlans();
        cleanupAccelBuffers();
        throw std::runtime_error(std::string("Failed to zero IR buffer: ") + cudaGetErrorString(err));
    }

    // Copy IRs to padded buffer (each track's IR goes to the beginning of its FFT-sized block)
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

    // Compute FFT of impulse responses
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

    // Clean up temporary buffer
    cudaFree(d_ir_padded);
}

void Conv1DAccelBenchmark::calculateCPUReference() {
    // Use time-domain convolution for golden reference
    conv1DCPUReference(getHostInput(), h_ir_buf, cpu_reference, ir_length_, getBufferSize(), getTrackCount());
}

void Conv1DAccelBenchmark::conv1DCPUReference(const float* input, const float* impulse_response,
                                             float* output, int ir_len, int buffer_size, int track_count) {
    for (int trackIdx = 0; trackIdx < track_count; trackIdx++) {
        for (int sampleIdx = 0; sampleIdx < buffer_size; sampleIdx++) {
            float outputSample = 0.0f;

            // Convolution sum
            for (int irIdx = 0; irIdx < ir_len; irIdx++) {
                int inputIdx = sampleIdx - irIdx;
                if (inputIdx >= 0 && inputIdx < buffer_size) {
                    // Input is in track-major layout
                    float inputValue = input[trackIdx * buffer_size + inputIdx];
                    float irValue = impulse_response[trackIdx * ir_len + irIdx];
                    outputSample += inputValue * irValue;
                }
            }

            // Store in interleaved format to match GPU output
            output[track_count * sampleIdx + trackIdx] = outputSample;
        }
    }
}

void Conv1DAccelBenchmark::runKernel() {
    // This will be called by performBenchmarkIteration()
    performBenchmarkIteration();
}

void Conv1DAccelBenchmark::performBenchmarkIteration() {
    cudaError_t err = cudaSuccess;
    cufftResult cufft_err;

    // Transfer input to device first
    transferToDevice();

    // Allocate temporary buffer for input preparation
    float* d_input_padded;
    size_t padded_size = getTrackCount() * fft_size_ * sizeof(float);
    err = cudaMalloc((void**)&d_input_padded, padded_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate padded input buffer (error code %s)!\n",
                cudaGetErrorString(err));
        return;
    }

    // Zero-pad input signals to FFT size
    err = cudaMemset(d_input_padded, 0, padded_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to zero input buffer (error code %s)!\n",
                cudaGetErrorString(err));
        cudaFree(d_input_padded);
        return;
    }

    // Copy input signals to padded buffer
    for (size_t trackIdx = 0; trackIdx < getTrackCount(); trackIdx++) {
        size_t src_offset = trackIdx * getBufferSize() * sizeof(float);
        size_t dst_offset = trackIdx * fft_size_ * sizeof(float);
        err = cudaMemcpy((char*)d_input_padded + dst_offset,
                        (char*)getDeviceInput() + src_offset,
                        getBufferSize() * sizeof(float),
                        cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to copy input to padded buffer (error code %s)!\n",
                    cudaGetErrorString(err));
            cudaFree(d_input_padded);
            return;
        }
    }

    // Forward FFT of input signals
    cufft_err = cufftExecR2C(forward_plan, d_input_padded, d_fft_input);
    if (cufft_err != CUFFT_SUCCESS) {
        fprintf(stderr, "Failed to execute input FFT (error code %d)!\n", cufft_err);
        cudaFree(d_input_padded);
        return;
    }

    // Complex multiplication in frequency domain
    dim3 blockSize(256);
    dim3 gridSize(static_cast<unsigned int>(getTrackCount()), (fft_size_ / 2 + 1 + blockSize.x - 1) / blockSize.x);

    ComplexMultiplyKernel<<<gridSize, blockSize>>>(
        d_fft_input, d_ir_fft, d_fft_output, fft_size_ / 2 + 1, static_cast<int>(getTrackCount()));

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch complex multiply kernel (error code %s)!\n",
                cudaGetErrorString(err));
        cudaFree(d_input_padded);
        return;
    }

    // Inverse FFT
    cufft_err = cufftExecC2R(inverse_plan, d_fft_output, d_input_padded);
    if (cufft_err != CUFFT_SUCCESS) {
        fprintf(stderr, "Failed to execute inverse FFT (error code %d)!\n", cufft_err);
        cudaFree(d_input_padded);
        return;
    }

    // Extract real part and convert to interleaved output format
    dim3 extractBlockSize(256);
    dim3 extractGridSize(static_cast<unsigned int>(getTrackCount()), (getBufferSize() + extractBlockSize.x - 1) / extractBlockSize.x);

    ExtractRealPartKernel<<<extractGridSize, extractBlockSize>>>(
        (cufftComplex*)d_input_padded, getDeviceOutput(), static_cast<int>(getBufferSize()), static_cast<int>(getTrackCount()));

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch extract kernel (error code %s)!\n",
                cudaGetErrorString(err));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to synchronize after convolution (error code %s)!\n",
                cudaGetErrorString(err));
    }

    // Clean up temporary buffer
    cudaFree(d_input_padded);

    // Transfer output back to host
    transferToHost();
}

void Conv1DAccelBenchmark::validate(ValidationData& validation_data) {
    // Output is already in host memory from transferToHost() call
    // Compare with CPU reference
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
