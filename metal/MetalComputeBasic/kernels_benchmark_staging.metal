// Metal benchmark kernels

#include <metal_stdlib>

using namespace metal;

// CLEANUP: This should read from the value passed in from the commandline argument.
#define BUFSIZE 512;

// A minimal no-op function to test the overhead of launching a kernel.
// |inUnused| is an unused input buffer, but it was needed to be able to launch a kernel.
// The host should send a small buffer in this param to not stress I/O.
kernel void noopFn(device const float* inUnused,
                   uint index [[thread_position_in_grid]])
{
    // Noop
}

// A minimal no-op function to test the overhead of I/O.
// |bufIn| and |bufOut| may be sized by the host with different mixes.
kernel void BenchmarkDataTransfer(device const float* bufIn,
                       device float* bufOut,
                       uint index [[thread_position_in_grid]])
{
    // Noop
}

// A minimal no-op function to test the overhead of I/O plus minimal processing.
// Each sample should be touched.
// |bufIn| and |bufOut| may be sized by the host with different mixes.
kernel void BenchmarkGain(device const float* bufIn,
                          device float* bufOut,
                          uint index [[thread_position_in_grid]])
{
    const int bufSize = BUFSIZE;
    const int startIdx = (index * bufSize);
    for (int i = startIdx; i < startIdx+bufSize; i++) {
        bufOut[i] = 2.0f * bufIn[i];
    }
}

// TODO: Port in benchmarks from kernels_filtering.metal.

// A domain-relevant function implementing a highly parallell modal filter bank.
// Phasor filters are used to simulate a bank of resonant filters.
// NOTE: A V2 version was measured in the paper and is being ported in 
//   (it lived in a separate file). Or rather this kernel file may be split
//   into multiple files.
struct Complex {
    float real;
    float imag;
};
Complex expComplex(Complex z) {
    float eReal = exp(z.real);
    float cosImag = cos(z.imag);
    float sinImag = sin(z.imag);
    Complex result;
    result.real = eReal * cosImag;
    result.imag = eReal * sinImag;
    return result;
}
// |inSignal|: Buffer of samples driving the filter.
// |input|: Buffer of parameters for each mode. "Mode-major" layout, in that
//      each mode's parameters are contiguous in memory.
//      Each mode has 5 parameters: amplitude, frequency, damping, and the
//         previous filter state, real and imaginary parts.
// |outState|: Buffer of state for each mode.
kernel void BenchmarkModalBank_ScratchV1(device const float* inSignal,
                                         device const float* input,
                                         device float* outState,
                                         device float* outAudio,
                                         uint index [[thread_position_in_grid]])
{
    // Thread |index| computes one mode.
    const int bufSize = BUFSIZE;
    // mode:<amplitude, freq, damp, state.re, state.im>
    device const float* params = &input[index*5];
    // TODO_V2: Make input coupling complex.
    const float amp = params[0];
    const float freq = params[1];
    const float damp = params[2];  // set 1-eps to allow ringing.

    Complex c;
    c.real = params[3];
    c.imag = params[4];
    
    for (int si = 0; si < bufSize; si++) {
        float samp = 0.0f;
        float inputCouple = amp * inSignal[si];
        c.real += inputCouple;
        c.imag += inputCouple;
        c.real *= damp;
        c.imag *= damp;
        c = expComplex(c);
        // Taking output as real part for this benchmark.
        samp = c.real;

        if (index < 64) {
            // TODO_V2: Tree sum
            outAudio[bufSize*index + si] = samp;
        }
    }
}

// Reads from random memory. Each thread reads a random chunk of memory.
// |virtualSamples| should be a reasonably large buffer of samples (1MB-100MB+).
// |playheads| is an array of indices into |virtualSamples|, with each index
//    representing the starting chunk of a single "grain" of audio.
// |bufOut| is the output buffer; each thread copies its own output directly.
kernel void BenchmarkRndMem(device const float* virtualSamples,
                         device const int* playheads,
                         device float* bufOut,
                         uint index [[thread_position_in_grid]])
{
    const int bufSize = BUFSIZE;
    for (int i = 0; i < bufSize; i++) {
        // This benchmark is stressing random reads and writes from global memory,
        // so it's a simple copy.
        // Note the calling (host/CPU) code should have these in diffrent
        // places across sample memory, and likely unaligned, different loop lengths, etc.
        // Future benchmarks might add processing.
        bufOut[bufSize*index + i] = virtualSamples[playheads[i]+i];
    }
}
