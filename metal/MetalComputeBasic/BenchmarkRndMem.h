#ifndef BenchmarkRndMem_h
#define BenchmarkRndMem_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import "GPUABenchmark.h"

NS_ASSUME_NONNULL_BEGIN

@interface BenchmarkRndMem : GPUABenchmark
- (instancetype) initWithDevice: (id<MTLDevice>) device;
- (void) setup;
- (void) runBenchmark: (NSArray*) latencies;
- (NSString*) getKernelName;

// Size of the buffer, samples
@property int sampleMemSize;

@end


NS_ASSUME_NONNULL_END

#endif /* BenchmarkRndMem_h */


