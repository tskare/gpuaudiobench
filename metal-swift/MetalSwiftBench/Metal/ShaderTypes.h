#ifndef ShaderTypes_h
#define ShaderTypes_h

#ifdef __METAL_VERSION__
#define NS_ENUM(_type, _name) enum _name : _type _name; enum _name : _type
#define NSInteger metal::int32_t
#else
#import <Foundation/Foundation.h>
#endif

#include <simd/simd.h>

// Benchmark configuration passed to kernels
typedef struct {
    uint32_t bufferSize;
    uint32_t trackCount;
    uint32_t totalSamples;
    float    gainValue;
} BenchmarkParams;

// IIR filter coefficients
typedef struct {
    float b0, b1, b2;  // Feedforward coefficients
    float a1, a2;      // Feedback coefficients
} IIRCoefficients;

// Modal filter parameters
typedef struct {
    float frequency;
    float decay;
    float amplitude;
    float phase;
} ModalFilterParams;

// Convolution parameters
typedef struct {
    uint32_t kernelSize;
    uint32_t padding;
} ConvolutionParams;

// Random memory access parameters  
typedef struct {
    uint32_t tableSize;
    uint32_t seed;
} RandomAccessParams;

// FDTD3D parameters
typedef struct {
    uint32_t nx, ny, nz;
    float soundSpeed;
    float spatialStep;
    float timeStep;
    float absorptionCoeff;
    uint32_t sourceX, sourceY, sourceZ;
    uint32_t receiverX, receiverY, receiverZ;
    uint32_t bufferSize;
    uint32_t trackCount;
} FDTD3DParams;

#endif /* ShaderTypes_h */