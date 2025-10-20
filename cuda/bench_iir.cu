#include "bench_iir.cuh"
#include "thread_config.cuh"
#include <cstring>
#include <cmath>

// ============================================================================
// CUDA Kernel Implementation
// ============================================================================

__global__ void IIRFilterKernel(const float* bufIn, float* bufOut,
                                const IIRCoefficients* coeffs,
                                float* state,
                                int trackCount, int bufferSize) {
    int trackIdx = blockDim.x * blockIdx.x + threadIdx.x;

    if (trackIdx >= trackCount) return;

    // Get state variables for this track (2 per track: z1, z2)
    float z1 = state[trackIdx * 2];
    float z2 = state[trackIdx * 2 + 1];

    // Process samples for this track
    int startIdx = trackIdx * bufferSize;

    for (int i = 0; i < bufferSize; i++) {
        float x = bufIn[startIdx + i];

        // Direct Form II biquad implementation
        // w[n] = x[n] - a1*w[n-1] - a2*w[n-2]
        // y[n] = b0*w[n] + b1*w[n-1] + b2*w[n-2]
        float w = x - coeffs->a1 * z1 - coeffs->a2 * z2;
        float y = coeffs->b0 * w + coeffs->b1 * z1 + coeffs->b2 * z2;

        // Update state variables
        z2 = z1;
        z1 = w;

        bufOut[startIdx + i] = y;
    }

    // Save state for next call
    state[trackIdx * 2] = z1;
    state[trackIdx * 2 + 1] = z2;
}

// ============================================================================
// IIRBenchmark Implementation
// ============================================================================

IIRBenchmark::IIRBenchmark(size_t buffer_size, size_t track_count)
    : GPUABenchmark("IIRFilter", buffer_size, track_count) {
    state_count = track_count * STATES_PER_TRACK;
    state_size_bytes = state_count * sizeof(float);
}

IIRBenchmark::~IIRBenchmark() {
    cleanupIIRBuffers();
    BenchmarkUtils::freeHostBuffers({cpu_reference, cpu_state_reference});
    cpu_reference = nullptr;
    cpu_state_reference = nullptr;
}

void IIRBenchmark::setupBenchmark() {
    allocateBuffers(getTotalElements());
    allocateIIRBuffers();
    initializeCoefficients();
    generateTestData(42);

    cpu_reference = BenchmarkUtils::allocateHostBuffer<float>(
        getTotalElements(), "iir cpu reference");
    cpu_state_reference = BenchmarkUtils::allocateHostBuffer<float>(
        state_count, "iir cpu state reference");

    memset(cpu_state_reference, 0, state_size_bytes);
    calculateCPUReference();

    printf("IIR filter benchmark setup complete (biquad lowpass filter)\n");
}

void IIRBenchmark::runKernel() {
    performBenchmarkIteration();
}

void IIRBenchmark::performBenchmarkIteration() {
    transferToDevice();

    auto [blocks_per_grid, threads_per_block] = calculateGridDimensions(ThreadConfig::DEFAULT_BLOCK_SIZE_1D);

    IIRFilterKernel<<<blocks_per_grid, threads_per_block>>>(
        getDeviceInput(),
        getDeviceOutput(),
        d_coeffs,
        d_state,
        static_cast<int>(getTrackCount()),
        static_cast<int>(getBufferSize())
    );

    synchronizeAndCheck();

    transferToHost();

    CUDA_CHECK(cudaMemcpy(h_state, d_state, state_size_bytes, cudaMemcpyDeviceToHost));
}

void IIRBenchmark::validate(ValidationData& validation_data) {
    validation_data = compareWithReference(cpu_reference, 1e-4f);

    float max_state_error = 0.0f;
    for (size_t i = 0; i < state_count; ++i) {
        float error = std::abs(h_state[i] - cpu_state_reference[i]);
        max_state_error = std::max(max_state_error, error);
    }

    if (max_state_error > 1e-3f) {
        validation_data.status = ValidationStatus::FAILURE;
        validation_data.messages.push_back("IIR state validation failed");
        validation_data.max_error = std::max(validation_data.max_error, max_state_error);
    } else if (validation_data.status == ValidationStatus::SUCCESS) {
        validation_data.messages.push_back("IIR validation passed (output + state)");
    }
}

// ============================================================================
// Private Helper Methods
// ============================================================================

void IIRBenchmark::allocateIIRBuffers() {
    // Allocate host coefficient buffer
    h_coeffs = BenchmarkUtils::allocateHostBuffer<IIRCoefficients>(
        1, benchmark_name_ + " host coefficients buffer");

    // Allocate device coefficient buffer
    d_coeffs = BenchmarkUtils::allocateDeviceBuffer<IIRCoefficients>(
        1, benchmark_name_ + " device coefficients buffer");

    // Allocate host state buffer
    h_state = BenchmarkUtils::allocateHostBuffer<float>(
        state_count, benchmark_name_ + " host state buffer");

    // Allocate device state buffer
    d_state = BenchmarkUtils::allocateDeviceBuffer<float>(
        state_count, benchmark_name_ + " device state buffer");

    // Initialize state to zero
    memset(h_state, 0, state_size_bytes);
    CUDA_CHECK(cudaMemset(d_state, 0, state_size_bytes));
}

void IIRBenchmark::initializeCoefficients() {
    // Create a Butterworth lowpass filter at fs/4
    *h_coeffs = calculateButterworthCoefficients(0.25f);

    // Copy coefficients to device
    CUDA_CHECK(cudaMemcpy(d_coeffs, h_coeffs, sizeof(IIRCoefficients), cudaMemcpyHostToDevice));

    printf("IIR coefficients: b0=%.6f, b1=%.6f, b2=%.6f, a1=%.6f, a2=%.6f\n",
           h_coeffs->b0, h_coeffs->b1, h_coeffs->b2, h_coeffs->a1, h_coeffs->a2);
}

void IIRBenchmark::calculateCPUReference() {
    // Reset CPU state to match initial GPU state
    memset(cpu_state_reference, 0, state_size_bytes);

    // Run CPU reference implementation
    iirFilterCPUReference(getHostInput(), cpu_reference, getTotalElements(),
                         h_coeffs, cpu_state_reference,
                         getTrackCount(), getBufferSize());
}

void IIRBenchmark::iirFilterCPUReference(const float* input, float* output, int size,
                                        const IIRCoefficients* coeffs, float* state,
                                        int track_count, int buffer_size) {
    // Process each track independently
    for (int track = 0; track < track_count; ++track) {
        float z1 = state[track * 2];
        float z2 = state[track * 2 + 1];

        int start_idx = track * buffer_size;

        for (int i = 0; i < buffer_size; ++i) {
            float x = input[start_idx + i];

            // Direct Form II biquad implementation (same as GPU)
            float w = x - coeffs->a1 * z1 - coeffs->a2 * z2;
            float y = coeffs->b0 * w + coeffs->b1 * z1 + coeffs->b2 * z2;

            z2 = z1;
            z1 = w;

            output[start_idx + i] = y;
        }

        // Save final state
        state[track * 2] = z1;
        state[track * 2 + 1] = z2;
    }
}

IIRCoefficients IIRBenchmark::calculateButterworthCoefficients(float normalized_frequency) {
    // Design a 2nd-order Butterworth lowpass filter
    // normalized_frequency is fc/fs (e.g., 0.25 for fc = fs/4)

    const float PI = 3.14159265358979323846f;
    float omega = 2.0f * PI * normalized_frequency;
    float cos_omega = cosf(omega);
    float sin_omega = sinf(omega);
    float alpha = sin_omega / (2.0f * 0.707f);  // Q = 0.707 for Butterworth

    // Calculate coefficients
    float b0 = (1.0f - cos_omega) / 2.0f;
    float b1 = 1.0f - cos_omega;
    float b2 = (1.0f - cos_omega) / 2.0f;
    float a0 = 1.0f + alpha;
    float a1 = -2.0f * cos_omega;
    float a2 = 1.0f - alpha;

    // Normalize by a0
    IIRCoefficients coeffs;
    coeffs.b0 = b0 / a0;
    coeffs.b1 = b1 / a0;
    coeffs.b2 = b2 / a0;
    coeffs.a1 = a1 / a0;
    coeffs.a2 = a2 / a0;

    return coeffs;
}

void IIRBenchmark::cleanupIIRBuffers() {
    BenchmarkUtils::freeHostBuffers({reinterpret_cast<float*>(h_coeffs), h_state});
    BenchmarkUtils::freeDeviceBuffers({reinterpret_cast<float*>(d_coeffs), d_state});
    h_coeffs = nullptr;
    d_coeffs = nullptr;
    h_state = nullptr;
    d_state = nullptr;
}
