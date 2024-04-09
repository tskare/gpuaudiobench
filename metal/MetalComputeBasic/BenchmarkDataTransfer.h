#ifndef BenchmarksDataTransfer_h
#define BenchmarksDataTransfer_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import "GPUABenchmark.h"
#import "MetalAdder.h"

NS_ASSUME_NONNULL_BEGIN

@interface BenchmarkDataTransfer : GPUABenchmark
- (instancetype) initWithDevice: (id<MTLDevice>) device;
- (void) setup;
- (void) runBenchmark: (NSArray*) latencies;
- (NSString*) getKernelName;

// These scale the size of input/output buffers respectively.
@property float inputBufferSizeRatio;
@property float outputBufferSizeRatio;

@end


@interface BenchmarkGain : GPUABenchmark
- (instancetype) initWithDevice: (id<MTLDevice>) device;
- (void) setup;
- (void) runBenchmark: (NSArray*) latencies;
- (NSString*) getKernelName;

@property int baseBufferSize;

@end

NS_ASSUME_NONNULL_END

#endif /* BenchmarksDataTransfer_h */


