// Metal benchmark kernels

#include <metal_stdlib>
#include "ShaderTypes.h"

using namespace metal;

// Buffer size is provided at dispatch time; avoid relying on compile-time constants.
inline uint bit_reverse(uint value, uint bits) {
    return reverse_bits(value) >> (32 - bits);
}

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
                          constant int& bufferSize,
                          uint index [[thread_position_in_grid]])
{
    const int startIdx = (index * bufferSize);
    for (int i = startIdx; i < startIdx+bufferSize; i++) {
        bufOut[i] = 2.0f * bufIn[i];
    }
}

// FFT benchmark - radix-2 Cooley–Tukey implementation of real-to-complex FFT
// |bufIn| contains real audio samples
// |bufOut| contains complex FFT results (interleaved real/imag)
kernel void BenchmarkFFT(device const float* bufIn,
                         device float* bufOut,
                         constant int& bufferSize,
                         uint index [[thread_position_in_grid]])
{
    const uint FFT_SIZE = 1024;
    const uint LOG2_FFT = 10;
    const uint outputSize = FFT_SIZE / 2 + 1;
    const uint startIdx = index * (uint)bufferSize;
    const uint outputStartIdx = index * outputSize * 2;

    thread float2 data[FFT_SIZE];

    const uint samplesToCopy = min((uint)bufferSize, FFT_SIZE);
    for (uint n = 0; n < samplesToCopy; ++n) {
        data[n] = float2(bufIn[startIdx + n], 0.0f);
    }
    for (uint n = samplesToCopy; n < FFT_SIZE; ++n) {
        data[n] = float2(0.0f, 0.0f);
    }

    // Bit-reversal permutation
    for (uint i = 0; i < FFT_SIZE; ++i) {
        uint j = bit_reverse(i, LOG2_FFT);
        if (j > i) {
            float2 temp = data[i];
            data[i] = data[j];
            data[j] = temp;
        }
    }

    // Iterative Cooley–Tukey FFT (complex form)
    for (uint stage = 1; stage <= LOG2_FFT; ++stage) {
        uint m = 1u << stage;
        uint m2 = m >> 1;
        float angleStep = -2.0f * M_PI_F / float(m);

        for (uint k = 0; k < FFT_SIZE; k += m) {
            for (uint j = 0; j < m2; ++j) {
                float angle = angleStep * float(j);
                float c = cos(angle);
                float s = sin(angle);
                float2 twiddle = float2(c, s);

                float2 t = data[k + j + m2];
                float2 u = data[k + j];

                float2 twiddled;
                twiddled.x = t.x * twiddle.x - t.y * twiddle.y;
                twiddled.y = t.x * twiddle.y + t.y * twiddle.x;

                data[k + j] = u + twiddled;
                data[k + j + m2] = u - twiddled;
            }
        }
    }

    // Store the non-redundant half of the spectrum (real input symmetry)
    for (uint k = 0; k < outputSize; ++k) {
        float2 value = data[k];
        bufOut[outputStartIdx + k * 2] = value.x;
        bufOut[outputStartIdx + k * 2 + 1] = value.y;
    }
}

// Note: The actual IIR biquad filter implementation is in kernels_iir.metal
// as BenchmarkIIRBiquad, which is used by IIRFilterBenchmark.swift

// A domain-relevant function implementing a highly parallell modal filter bank.
// Phasor filters are used to simulate a bank of resonant filters.
// NOTE: A V2 version was measured in the paper and is being ported in 
//   (it lived in a separate file). Or rather this kernel file may be split
//   into multiple files.
// Modal Filter Bank Benchmark - optimized version
// Each thread processes one mode through the entire buffer
kernel void BenchmarkModalFilterBank(device const float* input,
                                     device float* bufOut,
                                     constant uint* numModes,
                                     constant uint* outputTracks,
                                     constant BenchmarkParams& config [[buffer(4)]],
                                     uint index [[thread_position_in_grid]])
{
    const uint bufSize = config.bufferSize;
    const int kNumModeParams = 8;
    
    if (index >= *numModes) return;
    
    // Get mode parameters
    device const float* modeParams = &input[index * kNumModeParams];
    const float amp = modeParams[0];
    const float freq = modeParams[1];
    float stateRe = modeParams[3];
    float stateIm = modeParams[4];
    
    // Map modes round-robin onto tracks so wide banks stay load-balanced.
    const uint outputTrack = index % (*outputTracks);
    
    // Precalculate cos/sin for efficiency
    const float cosVal = cos(2.0f * M_PI_F * freq);
    const float sinVal = sin(2.0f * M_PI_F * freq);
    
    // Process each sample in the buffer
    for (uint i = 0; i < bufSize; i++) {
        // Complex multiplication: state *= exp(i * 2pi * freq)
        float newRe = stateRe * cosVal - stateIm * sinVal;
        float newIm = stateRe * sinVal + stateIm * cosVal;
        
        stateRe = newRe;
        stateIm = newIm;
        
        // Add contribution to output (real part only)
        // Use atomic add to safely accumulate across threads
        atomic_fetch_add_explicit((device atomic<float>*)&bufOut[outputTrack * bufSize + i], 
                                  amp * stateRe, 
                                  memory_order_relaxed);
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
kernel void BenchmarkConv1D(constant uint* numTracks,
                            device const float* bufIn,
                            constant const float* bufIRs,
                            device float* bufOut,
                            constant uint* irLength,
                            constant BenchmarkParams& config [[buffer(5)]],
                            uint index [[thread_position_in_grid]]) {
    if (index >= *numTracks) return;
    
    const uint bufSize = config.bufferSize;
    const uint IRLENGTH = *irLength;
    
    for (uint i = 0; i < bufSize; i++) {
        float outSamp = 0.0f;
        for (uint j = 0; j < IRLENGTH; j++) {
            int t = int(i) - int(j);
            if (t >= 0 && t < int(bufSize)) {
                uint ut = uint(t);
                // Input: track-major layout (each track's samples are contiguous)
                // IR: track-major layout (each track's IR is contiguous)
                outSamp += bufIn[index * bufSize + ut] * bufIRs[index * IRLENGTH + j];
            }
        }
        // Output: interleaved layout for memory coalescing
        bufOut[(*numTracks) * i + index] = outSamp;
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
                         constant BenchmarkParams& config [[buffer(3)]],
                         uint index [[thread_position_in_grid]])
{
    if (index >= config.trackCount) {
        return;
    }

    const uint bufSize = config.bufferSize;
    const uint baseIndex = index * bufSize;
    int playhead = playheads[index];

    for (uint i = 0; i < bufSize; i++) {
        // This benchmark stresses random reads from global memory.
        // Store results using track-major layout (each track contiguous)
        // Note: playhead + i simulates granular synthesis reading
        bufOut[baseIndex + i] = virtualSamples[playhead + int(i)];
    }
}

// Gain processing with statistics computation.
// Applies gain to audio samples while computing mean and max values per track.
// |bufIn| input audio buffer (track-major layout: each track's samples are contiguous)
// |bufOut| output audio buffer (same layout as input)
// |stats| statistics buffer: [mean0, max0, mean1, max1, ...] for each track
// |bufferSize| number of samples per track
// |trackCount| number of tracks
// |gainValue| gain multiplication factor
kernel void BenchmarkGainStats(device const float* bufIn,
                              device float* bufOut,
                              device float* stats,
                              constant BenchmarkParams& params [[buffer(3)]],
                              uint index [[thread_position_in_grid]])
{
    if (index >= params.trackCount) return;

    float trackMean = 0.0f;
    float trackMax = -3.402823e+38f; // -FLT_MAX approximation

    // Process all samples for this track
    const uint startIdx = index * params.bufferSize;
    for (uint i = 0; i < params.bufferSize; i++) {
        float sample = bufIn[startIdx + i];

        // Apply gain
        bufOut[startIdx + i] = sample * params.gainValue;

        // Accumulate statistics
        trackMean += sample;
        if (sample > trackMax) {
            trackMax = sample;
        }
    }

    // Finalize statistics
    trackMean /= float(params.bufferSize);

    // Store statistics: [mean, max] per track
    stats[index * 2 + 0] = trackMean;
    stats[index * 2 + 1] = trackMax;
}

inline uint bitReverse(uint value, uint bits) {
    uint reversed = 0;
    for (uint i = 0; i < bits; ++i) {
        reversed = (reversed << 1) | (value & 1u);
        value >>= 1;
    }
    return reversed;
}

inline void performFFT(device float* buffer, uint offset, uint fftSize, float direction) {
    if (fftSize <= 1u) {
        return;
    }

    uint log2N = 0;
    uint temp = fftSize;
    while (temp > 1u) {
        temp >>= 1u;
        log2N++;
    }

    for (uint i = 0; i < fftSize; ++i) {
        uint j = bitReverse(i, log2N);
        if (j > i) {
            uint ai = offset + (i << 1);
            uint aj = offset + (j << 1);
            float tmpReal = buffer[ai];
            float tmpImag = buffer[ai + 1];
            buffer[ai] = buffer[aj];
            buffer[ai + 1] = buffer[aj + 1];
            buffer[aj] = tmpReal;
            buffer[aj + 1] = tmpImag;
        }
    }

    for (uint stage = 1; stage <= log2N; ++stage) {
        uint m = 1u << stage;
        uint halfM = m >> 1;
        float angle = direction * (2.0f * M_PI_F / float(m));
        float cosAngle;
        float sinAngle = sincos(angle, cosAngle);

        for (uint k = 0; k < fftSize; k += m) {
            float wReal = 1.0f;
            float wImag = 0.0f;
            for (uint j = 0; j < halfM; ++j) {
                uint index1 = offset + ((k + j) << 1);
                uint index2 = offset + ((k + j + halfM) << 1);

                float vReal = buffer[index2];
                float vImag = buffer[index2 + 1];

                float tReal = wReal * vReal - wImag * vImag;
                float tImag = wReal * vImag + wImag * vReal;

                float uReal = buffer[index1];
                float uImag = buffer[index1 + 1];

                buffer[index1] = uReal + tReal;
                buffer[index1 + 1] = uImag + tImag;
                buffer[index2] = uReal - tReal;
                buffer[index2 + 1] = uImag - tImag;

                float nextWReal = wReal * cosAngle - wImag * sinAngle;
                float nextWImag = wReal * sinAngle + wImag * cosAngle;
                wReal = nextWReal;
                wImag = nextWImag;
            }
        }
    }
}

#define CONV_MODE_FULL 0u
#define CONV_MODE_FORWARD_FFT 1u

// Accelerated 1D convolution using GPU FFT.
// Mode 0: perform convolution using precomputed IR FFTs.
// Mode 1: compute forward FFT of bufIn and write into irFFTs (used during setup).
kernel void BenchmarkConv1DAccel(constant uint* numTracks,
                                device const float* bufIn,
                                device float* irFFTs,
                                device float* bufOut,
                                device float* fftInputBuf,
                                device float* /*fftOutputBuf*/,
                                constant uint* irLength,
                                constant uint* fftSize,
                                constant BenchmarkParams& config [[buffer(8)]],
                                constant uint* operationMode [[buffer(9)]],
                                uint index [[thread_position_in_grid]]) {
    if (index >= *numTracks) return;

    const uint FFT_SIZE = *fftSize;
    if (FFT_SIZE == 0u) return;

    const uint mode = *operationMode;
    const bool forwardOnly = (mode == CONV_MODE_FORWARD_FFT);

    const uint bufSize = config.bufferSize;
    const uint sourceLength = forwardOnly ? *irLength : bufSize;

    const uint trackInputOffset = index * sourceLength;
    const uint trackFFTOffset = index * FFT_SIZE * 2;
    const uint trackIRFFTOffset = index * FFT_SIZE * 2;

    // Zero working buffer and copy input samples (imag part initialised to zero).
    for (uint i = 0; i < FFT_SIZE; ++i) {
        uint base = trackFFTOffset + (i << 1);
        fftInputBuf[base] = 0.0f;
        fftInputBuf[base + 1] = 0.0f;
    }

    const uint copyCount = min(sourceLength, FFT_SIZE);
    for (uint i = 0; i < copyCount; ++i) {
        uint base = trackFFTOffset + (i << 1);
        fftInputBuf[base] = bufIn[trackInputOffset + i];
    }

    performFFT(fftInputBuf, trackFFTOffset, FFT_SIZE, -1.0f);

    if (forwardOnly) {
        for (uint k = 0; k < FFT_SIZE; ++k) {
            uint src = trackFFTOffset + (k << 1);
            uint dst = trackIRFFTOffset + (k << 1);
            irFFTs[dst] = fftInputBuf[src];
            irFFTs[dst + 1] = fftInputBuf[src + 1];
        }
        return;
    }

    // Multiply with precomputed IR FFT in the frequency domain.
    for (uint k = 0; k < FFT_SIZE; ++k) {
        uint base = trackFFTOffset + (k << 1);
        float xr = fftInputBuf[base];
        float xi = fftInputBuf[base + 1];
        float hr = irFFTs[trackIRFFTOffset + (k << 1)];
        float hi = irFFTs[trackIRFFTOffset + (k << 1) + 1];

        float real = fma(-xi, hi, xr * hr);
        float imag = fma(xr, hi, xi * hr);
        fftInputBuf[base] = real;
        fftInputBuf[base + 1] = imag;
    }

    performFFT(fftInputBuf, trackFFTOffset, FFT_SIZE, 1.0f);

    float scale = 1.0f / float(FFT_SIZE);
    for (uint k = 0; k < FFT_SIZE; ++k) {
        uint base = trackFFTOffset + (k << 1);
        fftInputBuf[base] *= scale;
        fftInputBuf[base + 1] *= scale;
    }

    for (uint i = 0; i < bufSize; ++i) {
        uint base = trackFFTOffset + (i << 1);
        bufOut[(*numTracks) * i + index] = fftInputBuf[base];
    }
}
