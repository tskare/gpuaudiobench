#import <Foundation/Foundation.h>
#import <QuartzCore/QuartzCore.h>

#import "BenchmarkFilters.h"
#import "globals.h"

// BenchmarkModalBank benchmarks a modal filter bank.
@implementation BenchmarkModalBank
{
    // 512 * excitation applied to all modes.
    id<MTLBuffer> _mBufInputSignal;
    // M modes * <amplitude, freq, damp, state.re, state.im>
    id<MTLBuffer> _mBufInput;
    // M modes * output state, {re+im}
    id<MTLBuffer> _mBufOutputState;
    // 64 output channels of audio
    // (we can tree sum down into these)
    id<MTLBuffer> _mBufOutputAudio;
    
    // 1M modes * 5 params
    float outputBuffer[1000000*5*sizeof(float)];
}
- (instancetype) initWithDevice: (id<MTLDevice>) device
{
    self = [super initWithDevice:device];
    return self;
}
- (NSString*) getKernelName {
    return @"BenchmarkModalBank";
}
- (void) setup {
    _nModes = 100;
    _nOutputTracks = 64;
    _mBufInputSignal = [_mDevice newBufferWithLength:flagBufferSize*sizeof(float)
                                             options:MTLResourceStorageModeShared];
    _mBufInput = [_mDevice newBufferWithLength:_nModes*(5*sizeof(float))  // Cleanup: use sizeof(defined struct)
                                             options:MTLResourceStorageModeShared];
    _mBufOutputState = [_mDevice newBufferWithLength:_nModes*5*sizeof(float)
                                             options:MTLResourceStorageModeShared];
    _mBufOutputAudio = [_mDevice newBufferWithLength:_nOutputTracks*flagBufferSize*sizeof(float)
                                             options:MTLResourceStorageModeShared];
}
- (void)encodeModalCommand:(id<MTLComputeCommandEncoder>)computeEncoder {
    [computeEncoder setComputePipelineState:_mFunctionPSO];
    [computeEncoder setBuffer:_mBufInputSignal offset:0 atIndex:0];
    [computeEncoder setBuffer:_mBufInput offset:0 atIndex:1];
    [computeEncoder setBuffer:_mBufOutputState offset:0 atIndex:2];
    [computeEncoder setBuffer:_mBufOutputAudio offset:0 atIndex:3];
    unsigned int gridX = _nModes;
    MTLSize gridSize = MTLSizeMake(gridX, 1, 1);
    NSUInteger threadGroupSize = _mFunctionPSO.maxTotalThreadsPerThreadgroup;
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
    [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
}
- (void) runBenchmark:
        (NSMutableArray*) latencies {
    CFTimeInterval startTime = CACurrentMediaTime();

    id<MTLCommandBuffer> commandBuffer = [_mCommandQueue commandBuffer];
    assert(commandBuffer != nil);
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    assert(computeEncoder != nil);
    [self encodeModalCommand:computeEncoder];
    [computeEncoder endEncoding];
    [commandBuffer commit];

    float *buf = _mBufOutputState.contents;
    for (int i=0; i<_nModes*5; i++) {
        outputBuffer[i] = buf[i];
    }
    [commandBuffer waitUntilCompleted];

    CFTimeInterval endTime = CACurrentMediaTime();
    // NSLog(@"Total Runtime: %g ms", 1000*(endTime - startTime));
    NSNumber* latency = [NSNumber numberWithFloat:(1000*(endTime-startTime))];
    [latencies addObject:latency];
}
@end
