#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import "MetalAdder.h"
#import "BenchmarksMicro.h"
#import "BenchmarkDataTransfer.h"
#import "BenchmarkFilters.h"
#import "BenchmarkRndMem.h"

#import <QuartzCore/QuartzCore.h>

#include "globals.h"

// Storage for globals and commandline flags.
int flagBufferSize = 512;
NSString *benchmarkName = @"Conv1D";
int fs = 48000;
int nTracks = 32; // e.g. large: 4*4096;
int nRuns = 100;

// Help string
NSString *helpString = @"Usage: ./GPUAudioBench --buffersize 512 --benchmark Conv1D --fs 48000 --ntracks 128"
                       @"Supported benchmarks: \n"
                            @"  datacopy0199\n"
                            @"  datacopy2080\n"
                            @"  datacopy5050\n"
                            @"  datacopy8020\n"
                            @"  datacopy9901\n"
                            @"  gain\n"
                            @"  NoopKernelLaunch\n"
                            @"  IIRFilter\n"
                            @"  ModalFilterBank\n"
                            @"  RndMemRead\n"
                            @"  Conv1D\n"
                          @"Optional flags:\n"
                            @"  --dawsim: Simulate DAW-like behavior\n"
                            @"  --dawsim_delay: Simulated buffer interarrival time in milliseconds\n"
                            @"  --skip-human-readable-summary: Disable human-readable summary\n"
                            @"  --nruns: Number of runs\n"
                            @"  --help: Print this help message\n"
                            @"  --outputfile: Output file for stats [currently set by benchmarks dynamically]\n"
                           ;

// Feature flags storage
bool flagIncludeHumanReadableSummary = true;
// DAW Simulation configuration
bool dawsim = false;
double dawsim_delay = 0.07;  // Variable while debugging. Production cases, use (bufferSize/FS)

void fetchNextInteger(NSArray<NSString *> *args, NSString *key, int *value) {
    NSUInteger index = [args indexOfObject:key];
    if (index != NSNotFound && index + 1 < [args count]) {
        *value = [args[index + 1] intValue];
    } else {
        NSLog(@"Missing value for %@", key);
    }
}
void printLatencyStats(NSMutableArray<NSNumber*> *latencies) {
    [latencies sortUsingSelector:@selector(compare:)];
    unsigned long len = [latencies count];
    unsigned long idx50p = (unsigned long)((float)len*0.5f);
    unsigned long idx95p = (unsigned long)((float)len*0.95f);
    NSLog(@"p50: %@", [latencies objectAtIndex:idx50p]);
    NSLog(@"p95: %@", [latencies objectAtIndex:idx95p]);
    NSLog(@"max: %@", [latencies lastObject]);
    // Paper format line
    NSLog(@"%@ & %@", [latencies objectAtIndex:idx50p], [latencies objectAtIndex:idx95p]);

    // Human-readable summary:
    if (flagIncludeHumanReadableSummary) {
        float maxAllowableLatency = 1000.0f*flagBufferSize/fs;
        float latencyMax = [[latencies lastObject] floatValue];
        float latency_p95 = [[latencies objectAtIndex:idx50p] floatValue];
        float latency_median = [[latencies objectAtIndex:idx50p] floatValue];
        if (latency_median > maxAllowableLatency) {
            NSLog(@"WARNING: median latency %0.3f ms over %0.3f ms callback time limit.", latency_median, maxAllowableLatency);
        } else if (latency_p95 > maxAllowableLatency) {
            NSLog(@"WARNING: p95 latency %0.3f ms over %0.3f ms callback time limit (median ok).", latency_p95, maxAllowableLatency);
        } else if (latencyMax > maxAllowableLatency) {
            NSLog(@"WARNING: max latency %0.3f ms over %0.3f ms callback time limit (median, p95 ok).", latencyMax, maxAllowableLatency);
        } else {
            NSLog(@"OK: max latency %0.3f ms under %0.3f ms callback time limit. Please consider a margin of safety as well.", latencyMax, maxAllowableLatency);
        }
    }
}
void writeArrayToFile(NSString *filename_unused, NSArray *arr) {
    NSString *fileName = [NSString stringWithFormat:@"/tmp/gpubench_%@_%d_%d.txt", benchmarkName, flagBufferSize, nTracks];
    
    NSMutableString *fileContent = [[NSMutableString alloc] init];
    for (NSNumber *number in arr) {
        [fileContent appendFormat:@"%@\n", number];
    }

    NSError *error = nil;
    BOOL success = [fileContent writeToFile:fileName
                             atomically:YES
                               encoding:NSUTF8StringEncoding
                                  error:&error];
    if (!success) {
        // Handle error
        NSLog(@"Error writing to file %@: %@", fileName, error.localizedDescription);
    } else {
        NSLog(@"Success writing output stats to %@", fileName);
    }
}

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        // Parse commandline
        NSArray<NSString *> *args = [[NSProcessInfo processInfo] arguments];
        
        void (^fetchNextString)(NSString *, void(^)(NSString *)) = ^(NSString *key, void(^setter)(NSString *)) {
            NSUInteger index = [args indexOfObject:key];
            if (index != NSNotFound && index + 1 < [args count]) {
                setter(args[index + 1]);
            } else {
                NSLog(@"Missing value for %@", key);
            }
        };
        
        // Parse command line arguments
        [args enumerateObjectsUsingBlock:^(NSString * _Nonnull arg, NSUInteger idx, BOOL * _Nonnull stop) {
            if ([arg isEqualToString:@"--buffersize"]) {
                fetchNextInteger(args, @"--buffersize", &flagBufferSize);
            } else if ([arg isEqualToString:@"--benchmark"]) {
                fetchNextString(arg, ^(NSString *value) { benchmarkName = value; });
            } else if ([arg isEqualToString:@"--fs"]) {
                fetchNextInteger(args, @"--fs", &fs);
            } else if ([arg isEqualToString:@"--ntracks"]) {
                fetchNextInteger(args, @"--ntracks", &nTracks);
            } else if ([arg isEqualToString:@"--nruns"]) {
                fetchNextInteger(args, @"--nruns", &nRuns);
            } else if ([arg isEqualToString:@"--dawsim"]) {
                dawsim = true;
            } else if ([arg isEqualToString:@"--dawsim_delay"]) {
                fetchNextString(arg, ^(NSString *value) { dawsim_delay = [value doubleValue]; });
            } else if ([arg isEqualToString:@"--skip-human-readable-summary"]) {
                flagIncludeHumanReadableSummary = false;
            } else if ([arg isEqualToString:@"--help"]) {
                NSLog(@"%@", helpString);
                exit(0);
            }
        }];
        
        NSLog(@"Buffer Size: %d", flagBufferSize);
        NSLog(@"Benchmark: %@", benchmarkName);
        NSLog(@"FS: %d", fs);
        NSLog(@"Number of Tracks: %d", nTracks);
        
        /* ------- Below items derived from Metal ObjC example. ------- */
        
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();

        // Cleanup: Consider having benchmarks register themselves and we can simply
        // perform a dictionary lookup here.
        // Note that this would require reworking multiple benchmarks per class like
        // {datacopy1...datacopy3}, so we chain if-else's here.
        GPUABenchmark* benchmark;
        if ([benchmarkName isEqual:@"MetalAdder"]) {
            benchmark = [[MetalAdder alloc] initWithDevice:device];
        } else if ([benchmarkName isEqual:@"datacopy0199"]) {
            BenchmarkDataTransfer* b = [[BenchmarkDataTransfer alloc] initWithDevice:device];
            [b setInputBufferSizeRatio:0.01];
            [b setOutputBufferSizeRatio:0.99];
            benchmark = b;
        } else if ([benchmarkName isEqual:@"datacopy2080"]) {
            BenchmarkDataTransfer* b = [[BenchmarkDataTransfer alloc] initWithDevice:device];
            [b setInputBufferSizeRatio:0.2];
            [b setOutputBufferSizeRatio:0.8];
            benchmark = b;
        } else if ([benchmarkName isEqual:@"datacopy5050"]) {
            BenchmarkDataTransfer* b = [[BenchmarkDataTransfer alloc] initWithDevice:device];
            [b setInputBufferSizeRatio:0.5];
            [b setOutputBufferSizeRatio:0.5];
            benchmark = b;
        } else if ([benchmarkName isEqual:@"datacopy8020"]) {
            BenchmarkDataTransfer* b = [[BenchmarkDataTransfer alloc] initWithDevice:device];
            [b setInputBufferSizeRatio:0.8];
            [b setOutputBufferSizeRatio:0.2];
            benchmark = b;
        } else if ([benchmarkName isEqual:@"datacopy9901"]) {
            BenchmarkDataTransfer* b = [[BenchmarkDataTransfer alloc] initWithDevice:device];
            [b setInputBufferSizeRatio:0.99];
            [b setOutputBufferSizeRatio:0.01];
            benchmark = b;
        } else if ([benchmarkName isEqual:@"gain"]) {
            benchmark = [[BenchmarkGain alloc] initWithDevice:device];
        } else if ([benchmarkName isEqual:@"NoopKernelLaunch"]) {
            benchmark = [[BenchmarkNoopKernelLaunch alloc] initWithDevice:device];
        } else if ([benchmarkName isEqual:@"IIRFilter"]) {
            benchmark = [[BenchmarkIIRFilter alloc] initWithDevice:device];
        } else if ([benchmarkName isEqual:@"ModalFilterBank"]) {
            benchmark = [[BenchmarkModalBank alloc] initWithDevice:device];
        } else if ([benchmarkName isEqual:@"RndMemRead"]) {
            benchmark = [[BenchmarkRndMem alloc] initWithDevice:device];
        } else if ([benchmarkName isEqual:@"Conv1D"]) {
            benchmark = [[BenchmarkConvolution alloc] initWithDevice:device];
        } else {
            NSLog(@"Unknown benchmark name: %@", benchmarkName);
            exit(1);
        }
        
        // Create buffers to hold data
        [benchmark setup];
        NSMutableArray *arrLatencies = [NSMutableArray array];
        
        CFTimeInterval startTime = CACurrentMediaTime();
        CFTimeInterval nextDAWCall = startTime;
        for (unsigned int i=0; i<nRuns; i++) {
            [benchmark runBenchmark:arrLatencies];
            
            if (dawsim) {
                nextDAWCall += dawsim_delay;
                //NSLog(@"%f %f sleep", CACurrentMediaTime(), nextDAWCall);
                while (CACurrentMediaTime() < nextDAWCall) {
                    // Spin; the below can be enabled to debug.
                    // NSLog(@"%f %f sleep", CACurrentMediaTime(), nextDAWCall);
                }
                //NSLog(@"Done");
            }
        }
        CFTimeInterval endTime = CACurrentMediaTime();
        NSLog(@"Outer Runtime: %g ms", 1000*(endTime - startTime));

        float maximumValue = [[arrLatencies valueForKeyPath: @"@max.self"] floatValue];
        NSLog(@"Max latency of %lu runs: %f ms", (unsigned long)[arrLatencies count], maximumValue);
        
        writeArrayToFile(@"UnusedParam", arrLatencies);
        // Note: this sorts the array in-place so perform any time-series-aware
        // operations before.
        printLatencyStats(arrLatencies);
        
        NSLog(@"Execution finished");
    }
    return 0;
}
