#ifndef BenchmarkFilters_h
#define BenchmarkFilters_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import "GPUABenchmark.h"

NS_ASSUME_NONNULL_BEGIN

@interface BenchmarkModalBank : GPUABenchmark
- (instancetype) initWithDevice: (id<MTLDevice>) device;
- (void) setup;
- (void) runBenchmark: (NSArray*) latencies;
- (NSString*) getKernelName;

// Number of modes in the filter bank
@property int nModes;
// Number of audio signals, first N modes or tree-sum.
@property int nOutputTracks;

@end

@interface BenchmarkConvolution : GPUABenchmark
- (instancetype) initWithDevice: (id<MTLDevice>) device;
- (void) setup;
- (void) runBenchmark: (NSArray*) latencies;
- (NSString*) getKernelName;

// Number of modes in the filter bank
@property int irLen;

@end


NS_ASSUME_NONNULL_END

#endif /* BenchmarkFilters_h */


