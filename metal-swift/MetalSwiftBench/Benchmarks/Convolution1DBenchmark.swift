//
//  Convolution1DBenchmark.swift
//  MetalSwiftBench
//
//  1D convolution benchmark - tests constant/texture memory vs device memory
//

import Foundation
import Metal

final class Convolution1DBenchmark: Convolution1DBaseBenchmark {
    
    // Configuration parameters (tunable)
    var useConstantMemory: Bool = true  // Use constant memory for IRs vs device memory

    required convenience init(device: MTLDevice, bufferSize: Int, trackCount: Int) throws {
        try self.init(device: device, bufferSize: bufferSize, trackCount: trackCount, impulseResponseLength: 256)
    }

    override init(device: MTLDevice, bufferSize: Int, trackCount: Int, impulseResponseLength: Int) throws {
        try super.init(device: device, bufferSize: bufferSize, trackCount: trackCount, impulseResponseLength: impulseResponseLength)
        self.kernelName = "BenchmarkConv1D"
    }

    convenience init(device: MTLDevice, bufferSize: Int, trackCount: Int,
                     impulseResponseLength: Int,
                     useConstantMemory: Bool = true) throws {
        try self.init(device: device, bufferSize: bufferSize, trackCount: trackCount, impulseResponseLength: impulseResponseLength)
        self.useConstantMemory = useConstantMemory
    }
    
    override func setup() throws {
        try super.setup()
        let irOptions: MTLResourceOptions = useConstantMemory ? .cpuCacheModeWriteCombined : .storageModeShared
        try setupConvolutionBuffers(impulseResourceOptions: irOptions)
    }
    
    override func performBenchmarkIteration() throws {
        guard let pipelineState = pipelineState,
              let inputBuffer = inputBuffer,
              let impulseResponseBuffer = impulseResponseBuffer,
              let outputBuffer = outputBuffer else {
            throw BenchmarkError.pipelineCreationFailed
        }

        var numTracks = UInt32(trackCount)
        var irLength = UInt32(impulseResponseLength)
        var params = makeParams()

        let threadsPerThreadgroup = MTLSize(width: min(256, trackCount), height: 1, depth: 1)
        let threadgroupsPerGrid = MTLSize(
            width: (trackCount + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
            height: 1,
            depth: 1
        )

        let parameterBindings: [CommandEncoder.ParameterBinding] = [
            CommandEncoder.ParameterBinding(numTracks, index: 0),
            CommandEncoder.ParameterBinding(irLength, index: 4),
            CommandEncoder.ParameterBinding(params, index: 5)
        ]

        try commandEncoder.execute(
            pipelineState: pipelineState,
            buffers: [
                (inputBuffer, 1),
                (impulseResponseBuffer, 2),
                (outputBuffer, 3)
            ],
            parameterBindings: parameterBindings,
            threadsPerThreadgroup: threadsPerThreadgroup,
            threadgroupsPerGrid: threadgroupsPerGrid,
            label: "Convolution1D"
        )
    }

    override func runBenchmark(iterations: Int, warmupIterations: Int = 3) throws -> BenchmarkResult {
        guard let _ = inputBuffer,
              let _ = impulseResponseBuffer,
              let _ = outputBuffer else {
            return BenchmarkResult(latencies: [], gpuLatencies: [], metadata: ["error": "Benchmark not properly initialized"])
        }

        let result = try super.runBenchmark(iterations: iterations, warmupIterations: warmupIterations)

        var enhancedMetadata = result.metadata
        enhancedMetadata["impulseResponseLength"] = impulseResponseLength
        enhancedMetadata["useConstantMemory"] = useConstantMemory

        if shouldValidate() {
            let verificationResult = verifyAgainstGoldenReference()
            enhancedMetadata["verificationPassed"] = verificationResult.passed
            enhancedMetadata["maxError"] = verificationResult.maxError
            enhancedMetadata["meanError"] = verificationResult.meanError
            enhancedMetadata["verificationSampleCount"] = verificationResult.sampleCount
        } else {
            enhancedMetadata["verificationSkipped"] = true
        }

        return BenchmarkResult(latencies: result.latencies, gpuLatencies: result.gpuLatencies, metadata: enhancedMetadata)
    }
    
    private func verifyAgainstGoldenReference() -> (passed: Bool, maxError: Float, meanError: Float, sampleCount: Int) {
        guard let outputBuffer = self.outputBuffer,
              let cpuGoldenBuffer = cpuGoldenBuffer else {
            return (false, Float.infinity, Float.infinity, 0)
        }
        
        let outputPointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: trackCount * bufferSize)
        let totalSamples = trackCount * bufferSize
        let tolerance: Float = 1e-4  // Tolerance for floating point convolution

        return compareWithGolden(
            gpuPointer: outputPointer,
            cpuPointer: cpuGoldenBuffer,
            sampleCount: totalSamples,
            tolerance: tolerance,
            metric: { gpuPtr, cpuPtr, index in
                let gpu = gpuPtr[index]
                let cpu = cpuPtr[index]
                let error = abs(gpu - cpu)
                return cpu != 0 ? error / abs(cpu) : error
            }
        )
    }
}
