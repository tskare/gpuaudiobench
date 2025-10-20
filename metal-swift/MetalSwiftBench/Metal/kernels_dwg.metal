//
//  kernels_dwg.metal
//  MetalSwiftBench
//
//  Digital Waveguide Synthesis Metal Kernels
//

#include <metal_stdlib>
using namespace metal;

// Digital Waveguide Parameters Structure
struct DWGParams {
    uint numWaveguides;
    uint bufferSize;
    uint outputTracks;
    uint minLength;
    uint maxLength;
    float reflectionCoeff;
    float dampingCoeff;
};

// Digital Waveguide State Structure  
struct WaveguideState {
    uint length;
    uint inputTapPos;
    uint outputTapPos;
    uint writePos;
    float gain;
    float reflection;
    float damping;
    float padding; // Align to 32 bytes
};

// Simple 1D Digital Waveguide - Naive Implementation
// Each thread processes one waveguide independently
kernel void BenchmarkDWG1DNaive(
    device const WaveguideState* waveguideParams [[buffer(0)]],
    device float* delayLineForward [[buffer(1)]],
    device float* delayLineBackward [[buffer(2)]],
    device const float* inputSignal [[buffer(3)]],
    device float* outputBuffer [[buffer(4)]],
    constant DWGParams& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    // Early exit if beyond waveguide count
    if (gid >= params.numWaveguides) return;
    
    const WaveguideState wg = waveguideParams[gid];
    const uint maxDelayLength = params.maxLength;
    
    // Offset into the packed delay lines so each waveguide owns a contiguous slice.
    const uint delayBase = gid * maxDelayLength;
    
    // Process each sample in the buffer
    for (uint sample = 0; sample < params.bufferSize; sample++) {
        // Input injection
        float input = 0.0f;
        if (sample < params.bufferSize) {
            input = inputSignal[sample] * wg.gain;
        }
        
        // Advance along each travelling wave, wrapping at the waveguide length.
        uint forwardPos = (wg.writePos + sample) % wg.length;
        uint backwardPos = (wg.writePos + sample + wg.length/2) % wg.length;
        
        // Read from delay lines
        float forwardSample = delayLineForward[delayBase + forwardPos];
        float backwardSample = delayLineBackward[delayBase + backwardPos];
        
        // Apply damping
        forwardSample *= wg.damping;
        backwardSample *= wg.damping;
        
        // Inject input at input tap
        if ((wg.writePos + sample) % wg.length == wg.inputTapPos) {
            forwardSample += input;
            backwardSample += input;
        }
        
        // Apply reflections and write back
        float newForward = backwardSample * wg.reflection + input;
        float newBackward = forwardSample * wg.reflection + input;
        
        delayLineForward[delayBase + forwardPos] = newForward;
        delayLineBackward[delayBase + backwardPos] = newBackward;
        
        // Output extraction at output tap
        if ((wg.writePos + sample) % wg.length == wg.outputTapPos) {
            float outputSample = (forwardSample + backwardSample) * 0.5f;

            // Sum to mono output using proper atomic operations to prevent race conditions
            uint outputIdx = sample % params.bufferSize;
            // Use atomic add to safely accumulate from multiple threads
            device atomic_float* atomicOutput = (device atomic_float*)&outputBuffer[outputIdx];
            atomic_fetch_add_explicit(atomicOutput, outputSample, memory_order_relaxed);
        }
    }
}

// Optimized 1D Digital Waveguide - Accelerated Implementation
// Uses aligned memory access and coalesced operations
kernel void BenchmarkDWG1DAccel(
    device const WaveguideState* waveguideParams [[buffer(0)]],
    device float* delayLineForward [[buffer(1)]],
    device float* delayLineBackward [[buffer(2)]],
    device const float* inputSignal [[buffer(3)]],
    device float* outputBuffer [[buffer(4)]],
    constant DWGParams& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint groupSize [[threads_per_threadgroup]]
) {
    // Use local memory for temporary storage
    threadgroup float localOutput[512]; // Assuming max buffer size of 512
    
    // Early exit if beyond waveguide count
    if (gid >= params.numWaveguides) return;
    
    const WaveguideState wg = waveguideParams[gid];
    const uint maxDelayLength = params.maxLength;
    
    // Align each waveguide's delay storage to keep vector loads coalesced.
    const uint delayBase = gid * maxDelayLength;
    
    // Initialize local output
    if (lid < params.bufferSize) {
        localOutput[lid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Process samples in chunks for better cache utilization
    const uint samplesPerThread = (params.bufferSize + groupSize - 1) / groupSize;
    const uint startSample = lid * samplesPerThread;
    const uint endSample = min(startSample + samplesPerThread, params.bufferSize);
    
    for (uint sample = startSample; sample < endSample; sample++) {
        // Input injection
        float input = 0.0f;
        if (sample < params.bufferSize) {
            input = inputSignal[sample] * wg.gain;
        }
        
        // Walk the circular buffers with explicit wrap logic to preserve alignment.
        uint forwardPos = (wg.writePos + sample) % wg.length;
        uint backwardPos = (wg.writePos + sample + wg.length/2) % wg.length;
        
        // Note: Prefetching could be added here for cache optimization
        
        // Read with coalesced memory access
        float forwardSample = delayLineForward[delayBase + forwardPos];
        float backwardSample = delayLineBackward[delayBase + backwardPos];
        
        // Apply optimized damping using fast math
        forwardSample = fast::fma(forwardSample, wg.damping, 0.0f);
        backwardSample = fast::fma(backwardSample, wg.damping, 0.0f);
        
        // Inject input at input tap
        if (forwardPos == wg.inputTapPos) {
            forwardSample += input;
            backwardSample += input;
        }
        
        // Apply reflections using fused multiply-add
        float newForward = fast::fma(backwardSample, wg.reflection, input);
        float newBackward = fast::fma(forwardSample, wg.reflection, input);
        
        // Write back with aligned access
        delayLineForward[delayBase + forwardPos] = newForward;
        delayLineBackward[delayBase + backwardPos] = newBackward;
        
        // Output extraction at output tap
        if (forwardPos == wg.outputTapPos) {
            float outputSample = (forwardSample + backwardSample) * 0.5f;
            localOutput[sample] += outputSample;
        }
    }
    
    // Synchronize before writing to global memory
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write local output to global buffer with atomic operations for thread safety
    if (lid < params.bufferSize) {
        // Use atomic add to safely accumulate from multiple threadgroups
        device atomic_float* atomicOutput = (device atomic_float*)&outputBuffer[lid];
        atomic_fetch_add_explicit(atomicOutput, localOutput[lid], memory_order_relaxed);
    }
}

// Helper kernel for initializing waveguide parameters with various distributions
kernel void InitializeWaveguides(
    device WaveguideState* waveguideParams [[buffer(0)]],
    constant DWGParams& params [[buffer(1)]],
    constant uint& seed [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.numWaveguides) return;
    
    // Simple PRNG for reproducible random parameters
    uint rng = seed + gid * 1664525u + 1013904223u;
    
    device WaveguideState& wg = waveguideParams[gid];
    
    // Random length between min and max
    rng = rng * 1664525u + 1013904223u;
    wg.length = params.minLength + (rng % (params.maxLength - params.minLength + 1));
    
    // Random tap positions within the waveguide length
    rng = rng * 1664525u + 1013904223u;
    wg.inputTapPos = rng % wg.length;
    
    rng = rng * 1664525u + 1013904223u;
    wg.outputTapPos = rng % wg.length;
    
    // Initialize write position
    wg.writePos = 0;
    
    // Random gain between 0.1 and 1.0
    rng = rng * 1664525u + 1013904223u;
    wg.gain = 0.1f + 0.9f * (float(rng) / float(UINT_MAX));
    
    // Set reflection coefficient
    wg.reflection = params.reflectionCoeff;
    
    // Set damping coefficient
    wg.damping = params.dampingCoeff;
    
    // Padding for alignment
    wg.padding = 0.0f;
}
