#ifndef BenchmarksMicro_h
#define BenchmarksMicro_h

// Interface definition
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import "GPUABenchmark.h"
#import "MetalAdder.h"

NS_ASSUME_NONNULL_BEGIN

@interface BenchmarkIIRFilter : GPUABenchmark
- (instancetype) initWithDevice: (id<MTLDevice>) device;
- (void) setup;
- (void) runBenchmark: (NSArray*) latencies;
- (NSString*) getKernelName;
@end

NS_ASSUME_NONNULL_END

#endif /* BenchmarksMicro_h */


