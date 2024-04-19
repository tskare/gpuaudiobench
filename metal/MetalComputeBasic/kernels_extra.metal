// Additional Metal benchmark kernels

#include <metal_stdlib>

using namespace metal;

// CLEANUP: use from commandline argument. Otherwise the param requires a recompile.
#define BUFSIZE 512

// Run a sequence of |nFilters| filters in series.
// Data is thread-major:
// inSignal: [0..BUFSIZE) first sample, [BUFSIZE..2*BUFSIZE) second sample, etc.
//   Note: len(|inSignal|) will be larger than DAW's BUFSIZE [e.g. 512], as we
//   include state at the end of the buffer.
// as, bs: likewise grouped by which-parameter.
// outState:
// outAudio: processed version if |inSignal|. Full track count.
//   Next run should
// Note that for large track counts and small nFilters, this stresses I/O.
// As this benchmark overlaps narrower ones, results were not used in the paper.
kernel void BenchmarkFilters_Biquad_N(device const float* inSignal,
                                      device float *as,
                                      device float *bs,
                                      device float* outSignal,  // end of buffer contains state.
                                      constant int& bufSize,
                                      constant int& nFilters,
                                      constant int& nThreads,
                                      uint index [[thread_position_in_grid]]) {
    // We reserve space for up to 10 filters, with 6 state variables each, per thread.
    // Cleanup: avoid magic numbers. Having [10] forces a maximum of nFilters == 10,
    //     and isn't optimal for lower values.
    float x1[10] = {0}, x2[10] = {0}, y1[10] = {0}, y2[10] = {0};
    
    // Restore state.
    const device float* stateBlock = inSignal + (nThreads*bufSize);
    for (int fi = 0; fi < nFilters; fi++) {
        x1[fi] = stateBlock[60*index + 6*fi + 0];
        x2[fi] = stateBlock[60*index + 6*fi + 1];
        y1[fi] = stateBlock[60*index + 6*fi + 2];
        y2[fi] = stateBlock[60*index + 6*fi + 3];
    }
    
    const int paramStep = nThreads;
    for (int i = 0; i < BUFSIZE; i++) {
        float x = inSignal[bufSize*index + i];
        for (int fi = 0; fi < nFilters; fi++) {
            // V2: consider comparing against an approach where the parameters are grouped by thread.
            // that is, as and bs are next to each other in memory.
            // This is likely less efficient especially on discrete-memory architecture, but is
            // something that can be measured.
            float fOut = bs[0*paramStep + index] * x
                + bs[1*paramStep + index]*x1[fi]
                + bs[2*paramStep + index]*x2[fi]
                - as[1*paramStep + index]*y1[fi]
                - as[2*paramStep + index]*y2[fi];
            x2[fi] = x1[fi];
            x1[fi] = x;
            y2[fi] = y1[fi];
            y1[fi] = fOut;
            x = fOut;
        }
        outSignal[index*BUFSIZE + i] = x;
    }
    // Save state past end.
    device float* stateBlockWrite = outSignal + (nThreads*bufSize);
    for (int fi = 0; fi < nFilters; fi++) {
        stateBlockWrite[60*index + 6*fi + 0] = x1[fi];
        stateBlockWrite[60*index + 6*fi + 1] = x2[fi];
        stateBlockWrite[60*index + 6*fi + 2] = y1[fi];
        stateBlockWrite[60*index + 6*fi + 3] = y2[fi];
    }
}
