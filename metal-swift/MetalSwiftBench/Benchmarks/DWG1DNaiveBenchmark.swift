import Foundation
import Metal

final class DWG1DNaiveBenchmark: DWG1DBaseBenchmark {
    enum LengthDistribution {
        case uniform
        case random
    }

    var lengthDistribution: LengthDistribution = .uniform

    required init(device: MTLDevice, bufferSize: Int, trackCount: Int) throws {
        try super.init(device: device, bufferSize: bufferSize, trackCount: trackCount, numWaveguides: trackCount)
        self.kernelName = "BenchmarkDWG1DNaive"
    }

    convenience init(device: MTLDevice,
                     bufferSize: Int,
                     trackCount: Int,
                     numWaveguides: Int,
                     minLength: Int,
                     maxLength: Int,
                     lengthDistribution: LengthDistribution = .uniform) throws {
        try self.init(device: device, bufferSize: bufferSize, trackCount: trackCount)
        self.numWaveguides = numWaveguides
        self.minLength = minLength
        self.maxLength = maxLength
        self.lengthDistribution = lengthDistribution
    }

    override func generateWaveguideLengths() throws -> [Int] {
        switch lengthDistribution {
        case .uniform:
            guard numWaveguides > 1 else { return Array(repeating: maxLength, count: numWaveguides) }
            let step = max(1, (maxLength - minLength) / max(1, numWaveguides - 1))
            return (0..<numWaveguides).map { minLength + ($0 * step) }
        case .random:
            return (0..<numWaveguides).map { _ in Int.random(in: minLength...maxLength) }
        }
    }

    override func performBenchmarkIteration() throws {
        guard let pipelineState = pipelineState,
              let waveguideParamsBuffer = waveguideParamsBuffer,
              let delayLineForwardBuffer = delayLineForwardBuffer,
              let delayLineBackwardBuffer = delayLineBackwardBuffer,
              let inputSignalBuffer = inputSignalBuffer,
              let outputBuffer = outputBuffer,
              let dwgParamsBuffer = dwgParamsBuffer else {
            throw BenchmarkError.pipelineCreationFailed
        }

        let threadsPerThreadgroup = MTLSize(
            width: min(pipelineState.maxTotalThreadsPerThreadgroup, numWaveguides),
            height: 1,
            depth: 1
        )

        try commandEncoder.execute(
            pipelineState: pipelineState,
            buffers: [
                (waveguideParamsBuffer, 0),
                (delayLineForwardBuffer, 1),
                (delayLineBackwardBuffer, 2),
                (inputSignalBuffer, 3),
                (outputBuffer, 4),
                (dwgParamsBuffer, 5)
            ],
            threadsPerThreadgroup: threadsPerThreadgroup,
            threadgroupsPerGrid: MTLSize(
                width: (numWaveguides + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
                height: 1,
                depth: 1
            ),
            label: "DWG1DNaive"
        )
    }

    override func runBenchmark(iterations: Int, warmupIterations: Int = 3) throws -> BenchmarkResult {
        guard waveguideParamsBuffer != nil,
              delayLineForwardBuffer != nil,
              delayLineBackwardBuffer != nil,
              inputSignalBuffer != nil,
              outputBuffer != nil,
              dwgParamsBuffer != nil else {
            return BenchmarkResult(latencies: [], gpuLatencies: [], metadata: ["error": "Benchmark not properly initialized"])
        }

        let result = try super.runBenchmark(iterations: iterations, warmupIterations: warmupIterations)

        var enhancedMetadata = result.metadata
        enhancedMetadata["benchmark"] = "DWG1DNaive"
        enhancedMetadata["numWaveguides"] = numWaveguides
        enhancedMetadata["minLength"] = minLength
        enhancedMetadata["maxLength"] = maxLength
        enhancedMetadata["lengthDistribution"] = String(describing: lengthDistribution)
        enhancedMetadata["reflectionCoeff"] = reflectionCoeff
        enhancedMetadata["dampingCoeff"] = dampingCoeff

        if shouldValidate() {
            let verificationResult = verifyOutput(tolerance: 1e-3)
            enhancedMetadata["verificationPassed"] = verificationResult.passed
            enhancedMetadata["maxError"] = verificationResult.maxError
            enhancedMetadata["meanError"] = verificationResult.meanError
            enhancedMetadata["verificationSampleCount"] = verificationResult.sampleCount
        } else {
            enhancedMetadata["verificationSkipped"] = true
        }

        return BenchmarkResult(latencies: result.latencies, gpuLatencies: result.gpuLatencies, metadata: enhancedMetadata)
    }
}
