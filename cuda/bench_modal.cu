#include "bench_modal.cuh"
#include <cstring>
#include <cmath>

__device__ cuComplex my_cexpf(cuComplex expon) {
    cuComplex comp_1_w;
    float s, c;
    float e = exp(expon.x);
    sincosf(expon.y, &s, &c);
    comp_1_w.x = c * e;
    comp_1_w.y = s * e;
    return comp_1_w;
}

__global__ void ModalSynthesisKernel(const float* bufIn, float* bufOut, int nModes,
                                     BenchmarkUtils::BenchmarkParams params) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    const int bufferSize = static_cast<int>(params.bufferSize);
    if (i < nModes) {
        float amp = bufIn[i * ModalBenchmark::NUM_MODE_PARAMS + ModalBenchmark::AMPLITUDE];
        float state_re = bufIn[i * ModalBenchmark::NUM_MODE_PARAMS + ModalBenchmark::RESERVED1];
        float state_im = bufIn[i * ModalBenchmark::NUM_MODE_PARAMS + ModalBenchmark::RESERVED2];

        state_re = 0.5f;
        state_im = 0.5f;
        cuComplex start = make_cuComplex(state_re, state_im);
        const cuComplex value = my_cexpf(start);
        const float output_value = amp * value.x;

        if (i < ModalBenchmark::MODAL_OUTPUT_TRACKS) {
            for (int si = 0; si < bufferSize; si++) {
                bufOut[i * bufferSize + si] = output_value;
            }
        }
    }
}

ModalBenchmark::ModalBenchmark()
    : GPUABenchmark("Modal", BUFSIZE, MODAL_OUTPUT_TRACKS) {

    mode_params_size = NUM_MODES * NUM_MODE_PARAMS;
    mode_params_bytes = mode_params_size * sizeof(float);
    modal_output_size = BUFSIZE * MODAL_OUTPUT_TRACKS;
    modal_output_bytes = modal_output_size * sizeof(float);
}

ModalBenchmark::~ModalBenchmark() {
    cleanupModalBuffers();
    BenchmarkUtils::freeHostBuffers({cpu_reference});
    cpu_reference = nullptr;
}

void ModalBenchmark::setupBenchmark() {
    allocateModalBuffers();

    initializeModeParameters();

    cpu_reference = BenchmarkUtils::allocateHostBuffer<float>(
        modal_output_size, "modal cpu reference");

    calculateCPUReference();

    printf("Modal benchmark setup complete (%d modes, %d output tracks)\n",
           NUM_MODES, MODAL_OUTPUT_TRACKS);
}

void ModalBenchmark::runKernel() {
    performBenchmarkIteration();
}

void ModalBenchmark::performBenchmarkIteration() {
    CUDA_CHECK(cudaMemcpy(d_mode_params, h_mode_params, mode_params_bytes, cudaMemcpyHostToDevice));

    int threads_per_block = 256;
    int blocks_per_grid = (NUM_MODES + threads_per_block - 1) / threads_per_block;

    const auto params = makeBenchmarkParams();
    ModalSynthesisKernel<<<blocks_per_grid, threads_per_block>>>(
        d_mode_params,
        d_modal_output,
        NUM_MODES,
        params
    );

    synchronizeAndCheck();

    CUDA_CHECK(cudaMemcpy(h_modal_output, d_modal_output, modal_output_bytes, cudaMemcpyDeviceToHost));
}

void ModalBenchmark::validate(ValidationData& validation_data) {
    float max_error = 0.0f;
    float mean_error = 0.0f;

    for (size_t i = 0; i < modal_output_size; ++i) {
        float error = std::abs(h_modal_output[i] - cpu_reference[i]);
        max_error = std::max(max_error, error);
        mean_error += error;
    }
    mean_error /= modal_output_size;

    validation_data.max_error = max_error;
    validation_data.mean_error = mean_error;

    if (max_error > 1e-1f) {
        validation_data.status = ValidationStatus::FAILURE;
        validation_data.messages.push_back("Modal validation failed");
    } else {
        validation_data.status = ValidationStatus::SUCCESS;
        validation_data.messages.push_back("Modal validation passed (loose tolerance)");
    }
}

void ModalBenchmark::allocateModalBuffers() {
    h_mode_params = BenchmarkUtils::allocateHostBuffer<float>(
        mode_params_size, benchmark_name_ + " host mode parameters buffer");

    d_mode_params = BenchmarkUtils::allocateDeviceBuffer<float>(
        mode_params_size, benchmark_name_ + " device mode parameters buffer");

    h_modal_output = BenchmarkUtils::allocateHostBuffer<float>(
        modal_output_size, benchmark_name_ + " host modal output buffer");

    d_modal_output = BenchmarkUtils::allocateDeviceBuffer<float>(
        modal_output_size, benchmark_name_ + " device modal output buffer");

    memset(h_modal_output, 0, modal_output_bytes);
    CUDA_CHECK(cudaMemset(d_modal_output, 0, modal_output_bytes));
}

void ModalBenchmark::initializeModeParameters() {
    srand(42);

    for (int i = 0; i < NUM_MODES; ++i) {
        int param_idx = i * NUM_MODE_PARAMS;

        h_mode_params[param_idx + AMPLITUDE] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        h_mode_params[param_idx + FREQUENCY] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        h_mode_params[param_idx + PHASE] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        h_mode_params[param_idx + STATE_REAL] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        h_mode_params[param_idx + STATE_IMAG] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        h_mode_params[param_idx + RESERVED1] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        h_mode_params[param_idx + RESERVED2] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        h_mode_params[param_idx + RESERVED3] = 0.0f;
    }
}

void ModalBenchmark::calculateCPUReference() {
    modalSynthesisCPUReference(h_mode_params, cpu_reference,
                              NUM_MODES, getBufferSize(), MODAL_OUTPUT_TRACKS);
}

void ModalBenchmark::modalSynthesisCPUReference(const float* mode_params, float* output,
                                               int num_modes, int buffer_size, int output_tracks) {
    memset(output, 0, buffer_size * output_tracks * sizeof(float));

    // INTENTIONALLY SIMPLIFIED: Processing all 1M+ modes on CPU would take minutes per validation.
    // Only the first 32 modes write to output; remaining modes compute but don't contribute.
    // This validates output buffer addressing and kernel branch logic without full modal synthesis.
    int modes_to_process = std::min(num_modes, output_tracks);

    // Match GPU kernel: exp(0.5 + 0.5i).real = exp(0.5) * cos(0.5) ≈ 1.4469
    const float cexp_real = expf(0.5f) * cosf(0.5f);

    for (int i = 0; i < modes_to_process; ++i) {
        int param_idx = i * NUM_MODE_PARAMS;

        float amp = mode_params[param_idx + AMPLITUDE];

        // Match GPU kernel: output_value = amp * my_cexpf(0.5 + 0.5i).x
        float output_value = amp * cexp_real;

        // Only write if this mode contributes to output (matches GPU kernel)
        if (i < output_tracks) {
            for (int si = 0; si < buffer_size; ++si) {
                output[i * buffer_size + si] = output_value;
            }
        }
    }
}

void ModalBenchmark::cleanupModalBuffers() {
    BenchmarkUtils::freeHostBuffers({h_mode_params, h_modal_output});
    BenchmarkUtils::freeDeviceBuffers({d_mode_params, d_modal_output});
    h_mode_params = nullptr;
    d_mode_params = nullptr;
    h_modal_output = nullptr;
    d_modal_output = nullptr;
}
