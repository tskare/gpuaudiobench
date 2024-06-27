#import <Foundation/Foundation.h>
#import <QuartzCore/QuartzCore.h>

#import "BenchmarksMicro.h"
#import "globals.h"

@implementation BenchmarkIIRFilter {
  // Dummy buffer
  id<MTLBuffer> _mBufferA;
}
- (instancetype)initWithDevice:(id<MTLDevice>)device {
  self = [super initWithDevice:device];
  return self;
}
- (NSString*)getKernelName {
  return @"noopFn";
}
- (void)setup {
  // Syntax needs at least one input; allocate a dummy block of 1024 elements.
  _mBufferA = [_mDevice newBufferWithLength:1024
                                    options:MTLResourceStorageModeShared];
}
- (void)encodeNoopCommand:(id<MTLComputeCommandEncoder>)computeEncoder {
  // Launch a <1,1,1>-size threadgroup with maxTotalThreadsPerThreadgroup.
  // The actual kernel will immediately return.
  [computeEncoder setComputePipelineState:_mFunctionPSO];
  [computeEncoder setBuffer:_mBufferA offset:0 atIndex:0];
  unsigned int gridX = 1;
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
  [self encodeNoopCommand:computeEncoder];
  [computeEncoder endEncoding];
  [commandBuffer commit];

  [commandBuffer waitUntilCompleted];

  CFTimeInterval endTime = CACurrentMediaTime();
  NSLog(@"Total Runtime: %g ms", 1000 * (endTime - startTime));
  NSNumber* latency = [NSNumber numberWithFloat:(1000 * (endTime - startTime))];
  [latencies addObject:latency];
}
@end
