#import <Foundation/Foundation.h>
#import <QuartzCore/QuartzCore.h>

#import "BenchmarkRndMem.h"
#import "globals.h"

// BenchmarkModalBank benchmarks a modal filter bank.
@implementation BenchmarkRndMem
{
    // Large block of virtual sample memory
    id<MTLBuffer> _mBufSampleMemory;
    // Set of virtual playhead location
    id<MTLBuffer> _inputPlayheads;
    // N output channels
    id<MTLBuffer> _mBufOutputAudio;
    
    // Playhead configuration data.
    // Cleanup: turn into a struct.
    // Cleanup:
    int *playheadStarts;
    int *playheadEnds;
    int *playheads;
}
- (instancetype) initWithDevice: (id<MTLDevice>) device
{
    self = [super initWithDevice:device];
    // Default to >32M samples, 128MiB sample set.
    _sampleMemSize = 32 * 1024 * 1024;
    return self;
}
- (NSString*) getKernelName {
    return @"BenchmarkRndMem";
}
- (void) setup {
    _mBufSampleMemory = [_mDevice newBufferWithLength:_sampleMemSize*sizeof(float)
                                              options:MTLResourceStorageModeShared];
    _inputPlayheads = [_mDevice newBufferWithLength:nTracks*sizeof(int)
                                              options:MTLResourceStorageModeShared];
    _mBufOutputAudio = [_mDevice newBufferWithLength:nTracks*flagBufferSize*sizeof(float)
                                             options:MTLResourceStorageModeShared];
    // Cleanup: implement dealloc and free(). Or preferably use C++ and std::unique_ptr.
    // Currently this happens at process exit, but would leak if we support running
    // multiple benchmarks.
    playheadStarts = malloc(sizeof(int)*nTracks);
    playheadEnds = malloc(sizeof(int)*nTracks);
    playheads = malloc(sizeof(int)*nTracks);
    int minLoopLength = 2000;
    int maxLoopLength = 48000;
    int rndLoopRange = maxLoopLength - minLoopLength;
    int maxLoopStart = _sampleMemSize - maxLoopLength;
    for (int i = 0; i < nTracks; i++) {
        playheadStarts[i] = (int)(rand()*maxLoopStart);
        playheadEnds[i] = playheadStarts[i] + minLoopLength + (int)(rand()*rndLoopRange);
        playheads[i] = playheadStarts[i];
    }
}
- (void)encodeModalCommand:(id<MTLComputeCommandEncoder>)computeEncoder {
    [computeEncoder setComputePipelineState:_mFunctionPSO];
    [computeEncoder setBuffer:_mBufSampleMemory offset:0 atIndex:0];
    [computeEncoder setBuffer:_inputPlayheads offset:0 atIndex:1];
    [computeEncoder setBuffer:_mBufOutputAudio offset:0 atIndex:2];
    unsigned int gridX = nTracks;
    MTLSize gridSize = MTLSizeMake(gridX, 1, 1);
    NSUInteger threadGroupSize = _mFunctionPSO.maxTotalThreadsPerThreadgroup;
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
    [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
}
- (void) runBenchmark:
        (NSMutableArray*) latencies {
    CFTimeInterval startTime = CACurrentMediaTime();

    float *ph = _inputPlayheads.contents;
    for (int i=0; i<nTracks; i++) {
        ph[i] = playheads[i];
    }
    
    id<MTLCommandBuffer> commandBuffer = [_mCommandQueue commandBuffer];
    assert(commandBuffer != nil);
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    assert(computeEncoder != nil);
    [self encodeModalCommand:computeEncoder];
    [computeEncoder endEncoding];
    [commandBuffer commit];

    [commandBuffer waitUntilCompleted];

    // Update playheads; this is still in the measured time.
    for (int i=0; i<nTracks; i++) {
        ph[i] = playheads[i] + flagBufferSize;
        // Cleanup: Wrap inside of kernel as well. Currently we may read past the end,
        // but this is guaranteed to be in the buffer.
        // Loops are of different sizes so we'll get the desired unaligned access test pattern.
        if (ph[i]>playheadEnds[i]) {
            ph[i] = playheadStarts[i] + (ph[i] - playheadEnds[i]);
        }
    }
    
    CFTimeInterval endTime = CACurrentMediaTime();
    // NSLog(@"Total Runtime: %g ms", 1000*(endTime - startTime));
    NSNumber* latency = [NSNumber numberWithFloat:(1000*(endTime-startTime))];
    [latencies addObject:latency];
}
@end
