#pragma once

#include "globals.cuh"

// Virtual granular synthesizer benchmark
constexpr int kSampleMemNumElems = 512 * 1024 * 1024 / sizeof(float);

void SetupBenchmarkRndMem(float** h_sampleMem, float** d_sampleMem,
    int** h_playheads, int** d_playheads,
    float playheadsStart[NTRACKS],
    float playheadsEnd[NTRACKS],
    int minLoopLen,
    int maxLoopLen,
    int samplebufferEnd,
    float **h_out, float **d_out);

void RunBenchmarkRndMem(int** d_playheads, int** h_playheads, float** d_sampleMem, float** d_out, float** h_out,
    vector<float>& latencies,
    float* playheadsStart, float* playheadsEnd);
