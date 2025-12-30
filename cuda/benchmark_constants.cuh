#pragma once

// Shared constants used across CUDA benchmarks.
namespace BenchmarkConstants {
    // Gain processing.
    constexpr float GAIN_VALUE = 2.0f;
    constexpr float GAINSTATS_GAIN = 0.5f;

    // FDTD3D scaling.
    constexpr float FDTD3D_SOURCE_SCALE = 0.1f;
    constexpr float FDTD3D_OUTPUT_SCALE = 0.1f;
    constexpr float FDTD3D_CPU_REF_FREQ = 0.01f;

    // Digital waveguide defaults.
    constexpr float WAVEGUIDE_MIX_FACTOR = 0.5f;
    constexpr float WAVEGUIDE_GAIN_MIN = 0.1f;
    constexpr float WAVEGUIDE_GAIN_RANGE = 0.9f;
    constexpr float WAVEGUIDE_REFLECTION_PERTURBATION = 0.01f;
    constexpr float WAVEGUIDE_DAMPING_PERTURBATION = 0.0001f;

    // Convolution / IR helpers.
    constexpr float HAMMING_WINDOW_A0 = 0.54f;
    constexpr float HAMMING_WINDOW_A1 = 0.46f;
    constexpr float CONV1D_IR_BASE_FREQ = 0.1f;
    constexpr float CONV1D_IR_FREQ_RANGE = 0.05f;

    // Data transfer signal generation.
    constexpr float DATATRANSFER_SIGNAL_OFFSET = 0.5f;
    constexpr float DATATRANSFER_SIGNAL_AMPLITUDE = 0.5f;
    constexpr float DATATRANSFER_SIGNAL_FREQ = 0.001f;
    constexpr float RANDOM_SIGNAL_SCALE = 2.0f;

    // Modal synthesis initial state.
    constexpr float MODAL_STATE_INIT_REAL = 0.5f;
    constexpr float MODAL_STATE_INIT_IMAG = 0.5f;

} // namespace BenchmarkConstants
