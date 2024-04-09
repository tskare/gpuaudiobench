// Interface definition
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#import "GPUABenchmark.h"

NS_ASSUME_NONNULL_BEGIN

// (TODO this was prat of the sample, move into its own file and rename this global file.)
@interface MetalAdder : GPUABenchmark
- (instancetype) initWithDevice: (id<MTLDevice>) device;
- (void) setup;
- (void) runBenchmark: (NSArray*) latencies;
@end

@interface BenchmarkNoopKernelLaunch : GPUABenchmark
- (instancetype) initWithDevice: (id<MTLDevice>) device;
- (void) setup;
- (void) runBenchmark: (NSArray*) latencies;
@end

NS_ASSUME_NONNULL_END
