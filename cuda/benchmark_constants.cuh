#pragma once

// Common constants shared across CUDA benchmarks.

namespace BenchmarkConstants {

    // Audio processing.
    constexpr float GAIN_VALUE = 2.0f;         // Baseline gain multiplier.
    constexpr float GAINSTATS_GAIN = 0.5f;     // Lower gain for stats validation.

    // FDTD3D solver.
    constexpr float FDTD3D_SOURCE_SCALE = 0.1f;   // Keeps the wave solver stable.
    constexpr float FDTD3D_OUTPUT_SCALE = 0.1f;   // Matches source scaling for unity gain.
    constexpr float FDTD3D_CPU_REF_FREQ = 0.01f;  // Modulation used by CPU reference path.

    // Digital waveguide.
    constexpr float WAVEGUIDE_MIX_FACTOR = 0.5f;             // Combine forward/backward rails.
    constexpr float WAVEGUIDE_GAIN_MIN = 0.1f;               // Prevent silent strings.
    constexpr float WAVEGUIDE_GAIN_RANGE = 0.9f;             // Range -> 0.1 to 1.0.
    constexpr float WAVEGUIDE_REFLECTION_PERTURBATION = 0.01f;
    constexpr float WAVEGUIDE_DAMPING_PERTURBATION = 0.0001f;

    // Convolution and IR generation.
    constexpr float HAMMING_WINDOW_A0 = 0.54f;
    constexpr float HAMMING_WINDOW_A1 = 0.46f;
    constexpr float CONV1D_IR_BASE_FREQ = 0.1f;
    constexpr float CONV1D_IR_FREQ_RANGE = 0.05f;

    // Data transfer / signal generation.
    constexpr float DATATRANSFER_SIGNAL_OFFSET = 0.5f;
    constexpr float DATATRANSFER_SIGNAL_AMPLITUDE = 0.5f;
    constexpr float DATATRANSFER_SIGNAL_FREQ = 0.001f;
    constexpr float RANDOM_SIGNAL_SCALE = 2.0f;  // Map [0,1] -> [-1,+1].

    // Modal synthesis.
    constexpr float MODAL_STATE_INIT_REAL = 0.5f;
    constexpr float MODAL_STATE_INIT_IMAG = 0.5f;

} // namespace BenchmarkConstants
