#pragma once

#include <vector>

// Benchmark declarations for modal filterbank benchmarking.

void SetupBenchmarkModal(float** h_inBufPtr, float** h_outBufPtr, float** d_inBufPtr, float** d_outBufPtr);
void RunBenchmarkModal(float** d_inBuf, float** h_inBuf, float** d_outBuf,
    float** h_outBuf, int kNumModes, int kNumModeParams, std::vector<float> &latencies);

// Configurable parameters
// CLEANUP: These should be commandline parameters
constexpr int kNumModes = 1024 * 1024;
// We pack 8 floats per mode:
// amplitude, freq, phase, state.re, state.im, 3 reserved/padding
// [update: used for amplitude coupling in the real kernel]
constexpr int kNumModeParams = 8;
// Number of output tracks to sum to.
constexpr int kModalOutputTracks = 32;