#ifndef GPUABenchmark_h
#define GPUABenchmark_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@interface GPUABenchmark : NSObject {
    id<MTLDevice> _mDevice;
    
    // The compute pipeline generated from the compute kernel in the .metal shader file.
    // If you require more than the default, subclasses may override.
    id<MTLComputePipelineState> _mFunctionPSO;
    
    // The command queue used to pass commands to the device.
    id<MTLCommandQueue> _mCommandQueue;
}

- (instancetype) initWithDevice: (id<MTLDevice>) device;
- (void) setup;
- (void) runBenchmark: (NSArray*) latencies;
- (NSString*) getKernelName;

@end


#endif /* GPUABenchmark_h */
