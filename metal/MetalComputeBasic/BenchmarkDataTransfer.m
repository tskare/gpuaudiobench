#import <Foundation/Foundation.h>
#import <QuartzCore/QuartzCore.h>

#import "BenchmarkDataTransfer.h"
#import "globals.h"

const bool includeDawTransferTime = true;

// BenchmarkDataTransfer seeks to measure trends in kernel sizes.
//
@implementation BenchmarkDataTransfer
{
    id<MTLBuffer> _mBufInput;
    id<MTLBuffer> _mBufOutput;
    
    
    float *hostBufIn;
    float *hostBufOut;
}
- (instancetype) initWithDevice: (id<MTLDevice>) device
{
    self = [super initWithDevice:device];
    _inputBufferSizeRatio = 1.0f;
    _outputBufferSizeRatio = 1.0f;
    return self;
}
- (NSString*) getKernelName {
    return @"BenchmarkDataTransfer";
}
- (void) setup {
    int baseBufferSize = nTracks * flagBufferSize * sizeof(float);
    int inputBufferSize = (int)(_inputBufferSizeRatio*baseBufferSize);
    int outputBufferSize = (int)(_outputBufferSizeRatio*baseBufferSize);
    NSLog(@"Buffer sizes: Input %d   Output %d", inputBufferSize, outputBufferSize);
    _mBufInput = [_mDevice newBufferWithLength:inputBufferSize options:MTLResourceStorageModeShared];
    _mBufOutput = [_mDevice newBufferWithLength:outputBufferSize options:MTLResourceStorageModeShared];
    
    hostBufIn = malloc(inputBufferSize);
    for(int i=0; i<inputBufferSize/sizeof(float); i++) {
        hostBufIn[i] = rand();
    }
    hostBufOut = malloc(outputBufferSize);
    for (int i=0; i<outputBufferSize/sizeof(float); i++) {
        hostBufOut[i] = 0.0f;
    }
    
}
- (void)encodeNoopCommand:(id<MTLComputeCommandEncoder>)computeEncoder {
    // Launch a <1,1,1>-size threadgroup with maxTotalThreadsPerThreadgroup.
    // The actual kernel will immediately return.
    [computeEncoder setComputePipelineState:_mFunctionPSO];
    [computeEncoder setBuffer:_mBufInput offset:0 atIndex:0];
    [computeEncoder setBuffer:_mBufOutput offset:0 atIndex:1];
    unsigned int gridX = nTracks;
    MTLSize gridSize = MTLSizeMake(gridX, 1, 1);
    NSUInteger threadGroupSize = _mFunctionPSO.maxTotalThreadsPerThreadgroup;
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
    [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
}
- (void) runBenchmark:
(NSMutableArray*) latencies {
    // Simulate DAW sendsing us this buffer

    CFTimeInterval startTime = CACurrentMediaTime();

    if (includeDawTransferTime) {
        int baseBufferSize = nTracks * flagBufferSize * sizeof(float);
        int inputBufferSize = (int)(_inputBufferSizeRatio*baseBufferSize);

        float *buf = _mBufInput.contents;
        for(int i=0; i<inputBufferSize/sizeof(float); i++) {
            buf[i] = 0.0f; // hostBufIn[i];
        }
    }
    id<MTLCommandBuffer> commandBuffer = [_mCommandQueue commandBuffer];
    assert(commandBuffer != nil);
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    assert(computeEncoder != nil);
    [self encodeNoopCommand:computeEncoder];
    [computeEncoder endEncoding];
    [commandBuffer commit];

    [commandBuffer waitUntilCompleted];

    if (includeDawTransferTime) {
        int baseBufferSize = nTracks * flagBufferSize * sizeof(float);
        int outputBufferSize = (int)(_outputBufferSizeRatio*baseBufferSize);
        float *buf = _mBufOutput.contents;
        for (int i=0; i<outputBufferSize/sizeof(float); i++) {
            hostBufOut[i] = buf[i];
        }
    }
    CFTimeInterval endTime = CACurrentMediaTime();
    // NSLog(@"Total Runtime: %g ms", 1000*(endTime - startTime));
    NSNumber* latency = [NSNumber numberWithFloat:(1000*(endTime-startTime))];
    [latencies addObject:latency];
}
@end



// Cleanup: separate BenchmarkGain out to its own file.
// This was original a subclass of the above so it resided in the same file.
@implementation BenchmarkGain
{
    id<MTLBuffer> _mBufInput;
    id<MTLBuffer> _mBufOutput;
}
- (instancetype) initWithDevice: (id<MTLDevice>) device
{
    self = [super initWithDevice:device];
    return self;
}
- (NSString*) getKernelName {
    return @"BenchmarkGain";
}
- (void) setup {
    int baseBufferSize = nTracks * flagBufferSize * sizeof(float);
    NSLog(@"Buffer size (in+out): %d", baseBufferSize);
    _mBufInput = [_mDevice newBufferWithLength:baseBufferSize options:MTLResourceStorageModeShared];
    _mBufOutput = [_mDevice newBufferWithLength:baseBufferSize options:MTLResourceStorageModeShared];
    _baseBufferSize = baseBufferSize;
}
- (void)encodeNoopCommand:(id<MTLComputeCommandEncoder>)computeEncoder {
    [computeEncoder setComputePipelineState:_mFunctionPSO];
    [computeEncoder setBuffer:_mBufInput offset:0 atIndex:0];
    [computeEncoder setBuffer:_mBufOutput offset:0 atIndex:1];
    unsigned int gridX = nTracks;
    MTLSize gridSize = MTLSizeMake(gridX, 1, 1);
    NSUInteger threadGroupSize = _mFunctionPSO.maxTotalThreadsPerThreadgroup;
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
    [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
}
- (void) runBenchmark:
        (NSMutableArray*) latencies {
    
    int nBufElements = _baseBufferSize>>2;
    float* dataPtr = _mBufInput.contents;
    for (unsigned long index = 0; index < nBufElements; index++)
    {
        dataPtr[index] = (float)rand()/(float)(RAND_MAX);
    }
    
    CFTimeInterval startTime = CACurrentMediaTime();

    id<MTLCommandBuffer> commandBuffer = [_mCommandQueue commandBuffer];
    assert(commandBuffer != nil);
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    assert(computeEncoder != nil);
    [self encodeNoopCommand:computeEncoder];
    [computeEncoder endEncoding];
    [commandBuffer commit];

    [commandBuffer waitUntilCompleted];

    CFTimeInterval endTime = CACurrentMediaTime();
    // NSLog(@"Total Runtime: %g ms", 1000*(endTime - startTime));
    NSNumber* latency = [NSNumber numberWithFloat:(1000*(endTime-startTime))];
    [latencies addObject:latency];
    
    // Validate output
    bool valid = true;
    float* dataPtrOut = _mBufOutput.contents;
    for (unsigned long index = 0; index < nBufElements; index++) {
        if (2.0*dataPtr[index] != dataPtrOut[index]) {
            NSLog(@"Gain data validation want %f got %f", 2.0*dataPtr[index], dataPtrOut[index]);
            valid = false;
        }
    }
    if (!valid) {
        // Exit at end so all validation errors print.
        exit(1);
    }
}
@end
