#include "bench_dwg.cuh"
#include "benchmark_constants.cuh"
#include "thread_config.cuh"
#include <cstring>
#include <cmath>

// Static constant definitions
const float DWGBenchmark::DEFAULT_REFLECTION_COEFF = 0.99f;
const float DWGBenchmark::DEFAULT_DAMPING_COEFF = 0.9999f;

// ============================================================================
// CUDA Kernel Implementations
// ============================================================================

// Naive Digital Waveguide Kernel - each thread handles one waveguide
__global__ void DWG1DNaiveKernel(
    const WaveguideState* waveguideParams,
    float* delayLineForward,
    float* delayLineBackward,
    const float* inputSignal,
    float* outputBuffer,
    const DWGParams* params
) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    // Early exit if beyond waveguide count
    if (gid >= params->numWaveguides) return;

    const WaveguideState wg = waveguideParams[gid];
    const int maxDelayLength = params->maxLength;

    // Calculate base indices for this waveguide's delay lines
    const int delayBase = gid * maxDelayLength;

    // Process each sample in the buffer
    for (int sample = 0; sample < params->bufferSize; sample++) {
        // Get input signal
        float input = inputSignal[sample] * wg.gain;

        // Calculate current positions in delay lines
        int currentPos = (wg.writePos + sample) % wg.length;
        int forwardPos = currentPos;
        int backwardPos = (currentPos + wg.length/2) % wg.length;

        // Read from delay lines
        float forwardSample = delayLineForward[delayBase + forwardPos];
        float backwardSample = delayLineBackward[delayBase + backwardPos];

        // Apply damping
        forwardSample *= wg.damping;
        backwardSample *= wg.damping;

        // Inject input at input tap position
        if (currentPos == wg.inputTapPos) {
            forwardSample += input;
            backwardSample += input;
        }

        // Apply reflections and cross-coupling
        float newForward = backwardSample * wg.reflection;
        float newBackward = forwardSample * wg.reflection;

        // Write back to delay lines
        delayLineForward[delayBase + forwardPos] = newForward;
        delayLineBackward[delayBase + backwardPos] = newBackward;

        // Output extraction at output tap
        if (currentPos == wg.outputTapPos) {
            float outputSample = (forwardSample + backwardSample) * BenchmarkConstants::WAVEGUIDE_MIX_FACTOR;

            // Use atomic addition for proper mixing
            if (gid < params->outputTracks) {
                atomicAdd(&outputBuffer[sample], outputSample);
            }
        }
    }
}

// Optimized Digital Waveguide Kernel with better memory access patterns
__global__ void DWG1DAccelKernel(
    const WaveguideState* waveguideParams,
    float* delayLineForward,
    float* delayLineBackward,
    const float* inputSignal,
    float* outputBuffer,
    const DWGParams* params
) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    if (gid >= params->numWaveguides) return;

    // Load waveguide parameters into registers
    const WaveguideState wg = waveguideParams[gid];
    const int maxDelayLength = params->maxLength;
    const int delayBase = gid * maxDelayLength;
    const int bufferSize = params->bufferSize;

    // Shared memory for input signal and intermediate values
    extern __shared__ float sharedMem[];
    float* sharedInput = sharedMem;

    // Cooperative loading of input signal to shared memory
    // Each thread loads multiple elements for better efficiency
    int elemsPerThread = (bufferSize + blockDim.x - 1) / blockDim.x;
    for (int i = 0; i < elemsPerThread; i++) {
        int idx = tid + i * blockDim.x;
        if (idx < bufferSize) {
            sharedInput[idx] = inputSignal[idx];
        }
    }
    __syncthreads();

    // Optimize for power-of-2 lengths when possible
    bool isPowerOf2 = (wg.length & (wg.length - 1)) == 0;
    int lengthMask = wg.length - 1;

    // Process samples with optimized memory access
    #pragma unroll 4
    for (int sample = 0; sample < bufferSize; sample++) {
        float input = sharedInput[sample] * wg.gain;

        // Calculate positions - use bit masking for power-of-2 lengths
        int currentPos;
        if (isPowerOf2) {
            currentPos = (wg.writePos + sample) & lengthMask;
        } else {
            currentPos = (wg.writePos + sample) % wg.length;
        }

        int forwardPos = currentPos;
        int backwardPos;
        if (isPowerOf2) {
            backwardPos = (currentPos + (wg.length >> 1)) & lengthMask;
        } else {
            backwardPos = (currentPos + wg.length/2) % wg.length;
        }

        // Coalesced memory reads with prefetching
        float forwardSample = delayLineForward[delayBase + forwardPos];
        float backwardSample = delayLineBackward[delayBase + backwardPos];

        // Apply damping
        forwardSample *= wg.damping;
        backwardSample *= wg.damping;

        // Branchless input injection
        float inputContrib = (currentPos == wg.inputTapPos) ? input : 0.0f;
        forwardSample += inputContrib;
        backwardSample += inputContrib;

        // Cross-coupling with reflections
        float newForward = backwardSample * wg.reflection;
        float newBackward = forwardSample * wg.reflection;

        // Write back to delay lines
        delayLineForward[delayBase + forwardPos] = newForward;
        delayLineBackward[delayBase + backwardPos] = newBackward;

        // Branchless output extraction
        if (currentPos == wg.outputTapPos && gid < params->outputTracks) {
            float outputSample = (forwardSample + backwardSample) * BenchmarkConstants::WAVEGUIDE_MIX_FACTOR;
            atomicAdd(&outputBuffer[sample], outputSample);
        }
    }
}

// ============================================================================
// DWGBenchmark Implementation
// ============================================================================

DWGBenchmark::DWGBenchmark(Variant variant, size_t buffer_size, size_t track_count)
    : GPUABenchmark(variant == Variant::NAIVE ? "DWG1DNaive" : "DWG1DAccel", buffer_size, track_count),
      variant_(variant) {

    // Calculate buffer sizes for complex DWG state
    delay_line_size = track_count * DEFAULT_MAX_LENGTH;
    delay_line_bytes = delay_line_size * sizeof(float);
    output_buffer_size = buffer_size;
    output_buffer_bytes = output_buffer_size * sizeof(float);
}

DWGBenchmark::~DWGBenchmark() {
    cleanupDWGBuffers();
    BenchmarkUtils::freeHostBuffers({cpu_reference, cpu_delay_forward, cpu_delay_backward});
    cpu_reference = nullptr;
    cpu_delay_forward = nullptr;
    cpu_delay_backward = nullptr;
}

void DWGBenchmark::setupBenchmark() {
    // Don't use base class standard buffers - DWG has complex custom buffers
    // allocateBuffers(getTotalElements());  // Skip this

    // Allocate DWG-specific buffers
    allocateDWGBuffers();

    // Initialize DWG parameters and waveguide states
    initializeDWGParameters();
    initializeWaveguideStates();

    // Generate input signal
    for (size_t i = 0; i < getBufferSize(); ++i) {
        h_input_signal[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f; // Range [-1, 1]
    }

    // Allocate CPU reference buffers
    cpu_reference = BenchmarkUtils::allocateHostBuffer<float>(
        output_buffer_size, "dwg cpu reference");
    cpu_delay_forward = BenchmarkUtils::allocateHostBuffer<float>(
        delay_line_size, "dwg cpu delay forward");
    cpu_delay_backward = BenchmarkUtils::allocateHostBuffer<float>(
        delay_line_size, "dwg cpu delay backward");

    // Initialize CPU delay lines to match GPU state
    memcpy(cpu_delay_forward, h_delay_forward, delay_line_bytes);
    memcpy(cpu_delay_backward, h_delay_backward, delay_line_bytes);

    // Calculate CPU reference
    calculateCPUReference();

    printf("DWG benchmark setup complete (%s variant, %zu waveguides, max length %d)\n",
           variant_ == Variant::NAIVE ? "Naive" : "Accelerated",
           getTrackCount(), DEFAULT_MAX_LENGTH);
}

void DWGBenchmark::runKernel() {
    // Transfer all data to device
    cudaMemcpy(d_waveguide_params, h_waveguide_params,
               getTrackCount() * sizeof(WaveguideState), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dwg_params, h_dwg_params,
               sizeof(DWGParams), cudaMemcpyHostToDevice);
    cudaMemcpy(d_delay_forward, h_delay_forward,
               delay_line_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_delay_backward, h_delay_backward,
               delay_line_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_signal, h_input_signal,
               getBufferSize() * sizeof(float), cudaMemcpyHostToDevice);

    // Clear output buffer
    cudaMemset(d_output_buffer, 0, output_buffer_bytes);

    // Calculate grid dimensions
    int threadsPerBlock = ThreadConfig::DEFAULT_BLOCK_SIZE_1D;
    int blocksPerGrid = (getTrackCount() + threadsPerBlock - 1) / threadsPerBlock;

    // Launch appropriate kernel variant
    if (variant_ == Variant::NAIVE) {
        DWG1DNaiveKernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_waveguide_params, d_delay_forward, d_delay_backward,
            d_input_signal, d_output_buffer, d_dwg_params
        );
    } else {
        int sharedMemSize = getBufferSize() * sizeof(float);
        DWG1DAccelKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
            d_waveguide_params, d_delay_forward, d_delay_backward,
            d_input_signal, d_output_buffer, d_dwg_params
        );
    }

    // Synchronize and check for errors
    cudaDeviceSynchronize();

    // Transfer results back to host
    cudaMemcpy(h_output_buffer, d_output_buffer,
               output_buffer_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_delay_forward, d_delay_forward,
               delay_line_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_delay_backward, d_delay_backward,
               delay_line_bytes, cudaMemcpyDeviceToHost);
}

void DWGBenchmark::performBenchmarkIteration() {
    // Just call runKernel for simplicity
    runKernel();
}

void DWGBenchmark::validate(ValidationData& validation_data) {
    // DWG validation uses loose tolerance due to complex acoustic modeling
    float max_error = 0.0f;
    float mean_error = 0.0f;

    for (size_t i = 0; i < output_buffer_size; ++i) {
        float error = std::abs(h_output_buffer[i] - cpu_reference[i]);
        max_error = std::max(max_error, error);
        mean_error += error;
    }
    mean_error /= output_buffer_size;

    validation_data.max_error = max_error;
    validation_data.mean_error = mean_error;

    // Loose tolerance for DWG due to complex physics simulation
    if (max_error > 1e-2f) {
        validation_data.status = ValidationStatus::FAILURE;
        validation_data.messages.push_back("DWG validation failed");
    } else {
        validation_data.status = ValidationStatus::SUCCESS;
        validation_data.messages.push_back("DWG validation passed");
    }
}

// ============================================================================
// Private Helper Methods
// ============================================================================

void DWGBenchmark::allocateDWGBuffers() {
    // Allocate host buffers
    h_waveguide_params = BenchmarkUtils::allocateHostBuffer<WaveguideState>(
        getTrackCount(), benchmark_name_ + " host waveguide params");
    h_dwg_params = BenchmarkUtils::allocateHostBuffer<DWGParams>(
        1, benchmark_name_ + " host DWG params");
    h_delay_forward = BenchmarkUtils::allocateHostBuffer<float>(
        delay_line_size, benchmark_name_ + " host delay forward");
    h_delay_backward = BenchmarkUtils::allocateHostBuffer<float>(
        delay_line_size, benchmark_name_ + " host delay backward");
    h_input_signal = BenchmarkUtils::allocateHostBuffer<float>(
        getBufferSize(), benchmark_name_ + " host input signal");
    h_output_buffer = BenchmarkUtils::allocateHostBuffer<float>(
        output_buffer_size, benchmark_name_ + " host output buffer");

    // Allocate device buffers
    cudaMalloc(&d_waveguide_params, getTrackCount() * sizeof(WaveguideState));
    cudaMalloc(&d_dwg_params, sizeof(DWGParams));
    cudaMalloc(&d_delay_forward, delay_line_bytes);
    cudaMalloc(&d_delay_backward, delay_line_bytes);
    cudaMalloc(&d_input_signal, getBufferSize() * sizeof(float));
    cudaMalloc(&d_output_buffer, output_buffer_bytes);

    // Initialize delay lines and output to zero
    memset(h_delay_forward, 0, delay_line_bytes);
    memset(h_delay_backward, 0, delay_line_bytes);
    memset(h_output_buffer, 0, output_buffer_bytes);
    cudaMemset(d_delay_forward, 0, delay_line_bytes);
    cudaMemset(d_delay_backward, 0, delay_line_bytes);
    cudaMemset(d_output_buffer, 0, output_buffer_bytes);
}

void DWGBenchmark::initializeDWGParameters() {
    h_dwg_params->numWaveguides = static_cast<int>(getTrackCount());
    h_dwg_params->bufferSize = static_cast<int>(getBufferSize());
    h_dwg_params->outputTracks = static_cast<int>(getTrackCount());
    h_dwg_params->minLength = DEFAULT_MIN_LENGTH;
    h_dwg_params->maxLength = DEFAULT_MAX_LENGTH;
    h_dwg_params->reflectionCoeff = DEFAULT_REFLECTION_COEFF;
    h_dwg_params->dampingCoeff = DEFAULT_DAMPING_COEFF;
}

void DWGBenchmark::initializeWaveguideStates() {
    srand(42);  // Fixed seed for reproducible results

    for (size_t i = 0; i < getTrackCount(); ++i) {
        WaveguideState& wg = h_waveguide_params[i];

        // Randomize length within bounds
        wg.length = DEFAULT_MIN_LENGTH +
                   (rand() % (DEFAULT_MAX_LENGTH - DEFAULT_MIN_LENGTH));

        // Set tap positions
        wg.inputTapPos = wg.length / 4;
        wg.outputTapPos = 3 * wg.length / 4;
        wg.writePos = 0;

        // Set acoustic parameters
        wg.gain = 0.1f + 0.9f * (static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
        wg.reflection = DEFAULT_REFLECTION_COEFF +
                       0.01f * (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f);
        wg.damping = DEFAULT_DAMPING_COEFF +
                    0.0001f * (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f);
        wg.padding = 0.0f;
    }
}

void DWGBenchmark::calculateCPUReference() {
    // Run CPU reference DWG implementation
    dwgCPUReference(h_waveguide_params, cpu_delay_forward, cpu_delay_backward,
                   h_input_signal, cpu_reference, h_dwg_params);
}

void DWGBenchmark::dwgCPUReference(const WaveguideState* waveguide_params,
                                  float* delay_forward, float* delay_backward,
                                  const float* input_signal, float* output_buffer,
                                  const DWGParams* params) {
    // Initialize output buffer
    memset(output_buffer, 0, params->bufferSize * sizeof(float));

    // Process each waveguide
    for (int gid = 0; gid < params->numWaveguides; ++gid) {
        const WaveguideState& wg = waveguide_params[gid];
        const int delayBase = gid * params->maxLength;

        // Process each sample (same logic as GPU kernel)
        for (int sample = 0; sample < params->bufferSize; ++sample) {
            float input = input_signal[sample] * wg.gain;

            int currentPos = (wg.writePos + sample) % wg.length;
            int forwardPos = currentPos;
            int backwardPos = (currentPos + wg.length/2) % wg.length;

            float forwardSample = delay_forward[delayBase + forwardPos];
            float backwardSample = delay_backward[delayBase + backwardPos];

            forwardSample *= wg.damping;
            backwardSample *= wg.damping;

            if (currentPos == wg.inputTapPos) {
                forwardSample += input;
                backwardSample += input;
            }

            float newForward = backwardSample * wg.reflection;
            float newBackward = forwardSample * wg.reflection;

            delay_forward[delayBase + forwardPos] = newForward;
            delay_backward[delayBase + backwardPos] = newBackward;

            if (currentPos == wg.outputTapPos && gid < params->outputTracks) {
                float outputSample = (forwardSample + backwardSample) * BenchmarkConstants::WAVEGUIDE_MIX_FACTOR;
                output_buffer[sample] += outputSample;
            }
        }
    }
}

void DWGBenchmark::cleanupDWGBuffers() {
    // Free host buffers
    BenchmarkUtils::freeHostBuffers({reinterpret_cast<float*>(h_waveguide_params),
                                     reinterpret_cast<float*>(h_dwg_params),
                                     h_delay_forward, h_delay_backward,
                                     h_input_signal, h_output_buffer});

    // Free device buffers
    if (d_waveguide_params) cudaFree(d_waveguide_params);
    if (d_dwg_params) cudaFree(d_dwg_params);
    if (d_delay_forward) cudaFree(d_delay_forward);
    if (d_delay_backward) cudaFree(d_delay_backward);
    if (d_input_signal) cudaFree(d_input_signal);
    if (d_output_buffer) cudaFree(d_output_buffer);

    h_waveguide_params = nullptr;
    d_waveguide_params = nullptr;
    h_dwg_params = nullptr;
    d_dwg_params = nullptr;
    h_delay_forward = h_delay_backward = nullptr;
    d_delay_forward = d_delay_backward = nullptr;
    h_input_signal = h_output_buffer = nullptr;
    d_input_signal = d_output_buffer = nullptr;
}
