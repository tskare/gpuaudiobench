//
//  kernels_iir.metal
//  MetalSwiftBench
//
//  IIR biquad filter kernel
//

#include <metal_stdlib>
#include "ShaderTypes.h"

using namespace metal;

kernel void BenchmarkIIRBiquad(
    device const float* bufIn [[buffer(0)]],
    device float* bufOut [[buffer(1)]],
    constant IIRCoefficients& coeffs [[buffer(2)]],
    device float* state [[buffer(3)]],  // 2 state variables per track
    constant BenchmarkParams& params [[buffer(4)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= params.trackCount) return;
    
    // Get state variables for this track
    device float* z = state + (index * 2);
    float z1 = z[0];
    float z2 = z[1];
    
    // Process samples for this track
    uint startIdx = index * params.bufferSize;
    
    for (uint i = 0; i < params.bufferSize; i++) {
        float x = bufIn[startIdx + i];
        
        // Direct Form II biquad implementation
        // y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
        float w = x - coeffs.a1 * z1 - coeffs.a2 * z2;
        float y = coeffs.b0 * w + coeffs.b1 * z1 + coeffs.b2 * z2;
        
        // Update state variables
        z2 = z1;
        z1 = w;
        
        bufOut[startIdx + i] = y;
    }
    
    // Save state for next call
    z[0] = z1;
    z[1] = z2;
}