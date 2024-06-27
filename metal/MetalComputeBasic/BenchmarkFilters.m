#import <Foundation/Foundation.h>
#import <QuartzCore/QuartzCore.h>

#import "BenchmarkFilters.h"
#import "globals.h"

// BenchmarkModalBank benchmarks a modal filter bank.
@implementation BenchmarkModalBank {
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
  // CLEANUP: avoid magic constant. This is the max number of modes,
  //          here and below.
  float outputBuffer[1000000 * 5 * sizeof(float)];
}
- (instancetype)initWithDevice:(id<MTLDevice>)device {
  self = [super initWithDevice:device];
  return self;
}
- (NSString*)getKernelName {
  return @"BenchmarkModalBank";
}
- (void)setup {
  _nModes = 1000000;
  _nOutputTracks = 64;
  _mBufInputSignal =
      [_mDevice newBufferWithLength:flagBufferSize * sizeof(float)
                            options:MTLResourceStorageModeShared];
  _mBufInput = [_mDevice
      newBufferWithLength:_nModes *
                          (5 *
                           sizeof(
                               float))  // Cleanup: use sizeof(defined struct)
                  options:MTLResourceStorageModeShared];
  _mBufOutputState =
      [_mDevice newBufferWithLength:_nModes * 5 * sizeof(float)
                            options:MTLResourceStorageModeShared];
  _mBufOutputAudio = [_mDevice
      newBufferWithLength:_nOutputTracks * flagBufferSize * sizeof(float)
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
  [computeEncoder dispatchThreads:gridSize
            threadsPerThreadgroup:threadgroupSize];
}
- (void)runBenchmark:(NSMutableArray*)latencies {
  CFTimeInterval startTime = CACurrentMediaTime();

  id<MTLCommandBuffer> commandBuffer = [_mCommandQueue commandBuffer];
  assert(commandBuffer != nil);
  id<MTLComputeCommandEncoder> computeEncoder =
      [commandBuffer computeCommandEncoder];
  assert(computeEncoder != nil);
  [self encodeModalCommand:computeEncoder];
  [computeEncoder endEncoding];
  [commandBuffer commit];

  float* buf = _mBufOutputState.contents;
  for (int i = 0; i < _nModes * 5; i++) {
    outputBuffer[i] = buf[i];
  }
  [commandBuffer waitUntilCompleted];

  CFTimeInterval endTime = CACurrentMediaTime();
  // NSLog(@"Total Runtime: %g ms", 1000*(endTime - startTime));
  NSNumber* latency = [NSNumber numberWithFloat:(1000 * (endTime - startTime))];
  [latencies addObject:latency];
}
@end

// BenchmarkConvolution benchmarks parallel convolution,
// With ability to convolve with regular or texture memory.
@implementation BenchmarkConvolution {
    id<MTLBuffer> _mBufInput;
    id<MTLBuffer> _mBufIRs;
    id<MTLBuffer> _mBufOutput;
    id<MTLBuffer> _mBufConstNumTracks;
}
- (instancetype)initWithDevice:(id<MTLDevice>)device {
  self = [super initWithDevice:device];
  return self;
}
- (NSString*)getKernelName {
  return @"BenchmarkConv1D";
}
- (void)setup {
    int baseBufferSize = nTracks * 512 * sizeof(float);
    int irBufferSize = nTracks * CONV1D_IRSIZE * sizeof(float);
    _mBufInput = [_mDevice newBufferWithLength:baseBufferSize options:MTLResourceStorageModeShared];
    _mBufOutput = [_mDevice newBufferWithLength:baseBufferSize options:MTLResourceStorageModeShared];
    _mBufIRs = [_mDevice newBufferWithLength:irBufferSize options:MTLResourceStorageModeShared];
    int constantValue = nTracks;
    _mBufConstNumTracks = [_mDevice newBufferWithBytes:&constantValue
                                                    length:sizeof(int)
                                                    options:MTLResourceStorageModeShared];
}

- (void)encodeConv1DCommand:(id<MTLComputeCommandEncoder>)computeEncoder {
    [computeEncoder setComputePipelineState:_mFunctionPSO];
    [computeEncoder setBuffer:_mBufConstNumTracks offset:0 atIndex:0];
    [computeEncoder setBuffer:_mBufInput offset:0 atIndex:1];
    [computeEncoder setBuffer:_mBufIRs offset:0 atIndex:2];
    [computeEncoder setBuffer:_mBufOutput offset:0 atIndex:3];
    unsigned int gridX = nTracks;
    MTLSize gridSize = MTLSizeMake(gridX, 1, 1);
    NSUInteger threadGroupSize = _mFunctionPSO.maxTotalThreadsPerThreadgroup;
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
    [computeEncoder dispatchThreads:gridSize
                threadsPerThreadgroup:threadgroupSize];
}


- (void)runBenchmark:(NSMutableArray*)latencies {
  CFTimeInterval startTime = CACurrentMediaTime();

  id<MTLCommandBuffer> commandBuffer = [_mCommandQueue commandBuffer];
  assert(commandBuffer != nil);
  id<MTLComputeCommandEncoder> computeEncoder =
      [commandBuffer computeCommandEncoder];
  assert(computeEncoder != nil);
  [self encodeConv1DCommand:computeEncoder];
  [computeEncoder endEncoding];
  [commandBuffer commit];

  [commandBuffer waitUntilCompleted];

  CFTimeInterval endTime = CACurrentMediaTime();
  // NSLog(@"Total Runtime: %g ms", 1000*(endTime - startTime));
  NSNumber* latency = [NSNumber numberWithFloat:(1000 * (endTime - startTime))];
  [latencies addObject:latency];
}
@end
