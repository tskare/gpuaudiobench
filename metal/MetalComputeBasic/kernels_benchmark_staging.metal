// Metal benchmark kernels

#include <metal_stdlib>

using namespace metal;

// CLEANUP: This should read from the value passed in from the commandline argument.
#define BUFSIZE 512;

// A minimal no-op function to test the overhead of launching a kernel.
// |inUnused| is an unused input buffer, but it was needed to be able to launch a kernel.
// The host should send a small buffer in this param to not stress I/O.
kernel void noopFn(device const float* inUnused,
                   uint index [[thread_position_in_grid]]) {
  // Noop
}

// A minimal no-op function to test the overhead of I/O.
// |bufIn| and |bufOut| may be sized by the host with different mixes.
kernel void BenchmarkDataTransfer(device const float* bufIn,
                                  device float* bufOut,
                                  uint index [[thread_position_in_grid]]) {
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

// A domain-relevant function to test a small amount of mathematical processing.
// Each sample should be touched.
// |bufIn| and |bufOut| may be sized by the host with different mixes, but ideally
// should not be 
kernel void BenchmarkIIR(device const float* bufIn,
                         device float* bufOut,
                         uint index [[thread_position_in_grid]])
{
    const int bufSize = BUFSIZE;
    const int startIdx = (index * bufSize);
    for (int i = startIdx; i < startIdx+bufSize; i++) {
        bufOut[i] = 3.0f * bufIn[i];
    }
}

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
    // const float freq = params[1];  // Sent to/from in state, can be adjusted on host.
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

// Conv1D runs many convolution or convolution-like operations in parallel.
// It seeks to benchmark use of constant/texture memory vs. a standard
// input buffer.
//
// This benchmark performs time-domain 1D convolution using a nested
// loop, nTracks parallel threads of O(buffer size * IR length).
// Input and output are interleaved so memory accesses are aligned.
//
// On the MacOS Metal platform, please see section 4.2 of v3.1 of Apple's
// Metal Shading Language specification for more details.
// https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
// We may enable or disable use of constant memory easily on this platform,
// by changing the bufIRs 'constant' specifier below to 'device'
kernel void BenchmarkConv1D(constant int* numTracks, // [[buffer(0)]]
                            device const float* bufIn,
                            device const float* bufIRs,  // use constant memory
                            device float* bufOut,
                            uint index [[thread_position_in_grid]]) {
    // Aliased to reduce diff for when we pass this in as an argument,
    // to support runtime parameter adjustment vs. requiring a recompile.
    // Compiler should optimize this.
    const int bufSize = BUFSIZE;
    const int IRLENGTH = 256; // Cleanup: share with globals.h; pass this into the kernel as const.
    for (int i = 0; i < bufSize; i++) {
        float outSamp = 0.0f;
        for (int j = 0; j < IRLENGTH; j++) {
            int t = i-j;
            if (t >= 0 & t < IRLENGTH) {
                outSamp += bufIn[t] * bufIRs[(*numTracks)*j + index];
            }
        }
        bufOut[(*numTracks)*i + index] = outSamp;
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
