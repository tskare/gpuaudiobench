#import <Foundation/Foundation.h>

#import "GPUABenchmark.h"

@implementation GPUABenchmark
{
}
- (instancetype) initWithDevice: (id<MTLDevice>) device {
    self = [super init];
    if (!self) return nil;
    
    _mDevice = device;
    NSError* error = nil;

    id<MTLLibrary> defaultLibrary = [_mDevice newDefaultLibrary];
    if (defaultLibrary == nil) {
        NSLog(@"[device newDefaultLibrary] nil");
        return nil;
    }

    id<MTLFunction> mtlFunction = [defaultLibrary newFunctionWithName:[self getKernelName]];
    if (mtlFunction == nil) {
        NSLog(@"Could not find Metal function: %@", [self getKernelName]);
        exit(1);
        return nil;
    }
    
    _mFunctionPSO = [_mDevice newComputePipelineStateWithFunction:mtlFunction error:&error];
    if (_mFunctionPSO == nil) {
        NSLog(@"Error creating pipeline: %@", error);
        exit(1);
        return nil;
    }
    
    _mCommandQueue = [_mDevice newCommandQueue];
    if (_mCommandQueue == nil) {
        NSLog(@"Error creating command queue.");
        exit(1);
        return nil;
    }

    return self;
}
- (void) setup {
    NSLog(@"Unimplemented setup() in subclass");
    exit(EXIT_FAILURE);
}
- (void) runBenchmark:
        (NSMutableArray*) latencies
{
    NSLog(@"Unimplemented runBenchmark() in subclass");
    exit(EXIT_FAILURE);
}
- (NSString*) getKernelName {
    NSLog(@"Unimplemented setup() in subclass");
    exit(EXIT_FAILURE);
    return @"Unimplemnted_getKernelName";
}
@end
