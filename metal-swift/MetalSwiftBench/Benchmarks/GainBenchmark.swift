//  GainBenchmark.swift

import Foundation
import Metal

final class GainBenchmark: BaseBenchmark {
    private var buffers: StandardBuffers?
    private let gainValue: Float = 2.0
    
    required init(device: MTLDevice, bufferSize: Int, trackCount: Int) throws {
        try super.init(device: device, bufferSize: bufferSize, trackCount: trackCount)
        self.kernelName = "BenchmarkGain"
    }
    
    override func setup() throws {
        try super.setup()

        let sampleCount = trackCount * bufferSize
        buffers = try allocateStandardIOBuffers(sampleCount: sampleCount)

        populateInputBuffers(
            gpu: buffers!.gpuInput,
            host: buffers!.hostInput,
            count: sampleCount,
            pattern: .whiteNoise,
            seed: 42
        )

        calculateCPUGoldenReference()
    }
    
    private func calculateCPUGoldenReference() {
        guard let buffers = buffers else { return }
        for i in 0..<buffers.sampleCount {
            buffers.hostGolden[i] = gainValue * buffers.hostInput[i]
        }
    }
    
    override func performBenchmarkIteration() throws {
        guard let buffers = buffers else {
            throw BenchmarkError.pipelineCreationFailed
        }

        let bufferSizeParam = Int32(bufferSize)
        try executeKernelWithParams(
            inputBuffer: buffers.gpuInput,
            outputBuffer: buffers.gpuOutput,
            parameters: bufferSizeParam,
            label: "Gain"
        )
    }

    override func runBenchmark(iterations: Int, warmupIterations: Int = 3) throws -> BenchmarkResult {
        let result = try super.runBenchmark(iterations: iterations, warmupIterations: warmupIterations)

        var enhancedMetadata = result.metadata
        enhancedMetadata["benchmark"] = "Gain"
        enhancedMetadata["gainValue"] = gainValue
        enhancedMetadata["totalSamples"] = bufferSize * trackCount

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
    
    private func verifyAgainstGoldenReference() -> VerificationResult {
        guard let buffers = buffers else {
            return (false, Float.infinity, Float.infinity, 0)
        }

        return verifyOutputBuffer(
            buffers.gpuOutput,
            against: buffers.hostGolden,
            sampleCount: buffers.sampleCount,
            tolerance: 1e-5
        )
    }
    
    override func cleanup() {
        buffers?.cleanup()
        buffers = nil
    }
}
