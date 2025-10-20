import Foundation
import Metal

final class DWG1DAccelBenchmark: DWG1DBaseBenchmark {
    enum LengthDistribution {
        case uniform
        case random
        case aligned
        case powerOfTwo
    }

    var lengthDistribution: LengthDistribution = .aligned
    private var alignedLengths: [Int] = []

    required init(device: MTLDevice, bufferSize: Int, trackCount: Int) throws {
        let maxThreadsPerGroup = 256
        let alignedWaveguides = ((trackCount + maxThreadsPerGroup - 1) / maxThreadsPerGroup) * maxThreadsPerGroup
        try super.init(device: device,
                       bufferSize: bufferSize,
                       trackCount: trackCount,
                       numWaveguides: alignedWaveguides)
        self.kernelName = "BenchmarkDWG1DAccel"
    }

    convenience init(device: MTLDevice,
                     bufferSize: Int,
                     trackCount: Int,
                     numWaveguides: Int,
                     minLength: Int,
                     maxLength: Int,
                     lengthDistribution: LengthDistribution = .aligned) throws {
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
            alignedLengths = (0..<numWaveguides).map { minLength + ($0 * step) }
        case .random:
            alignedLengths = (0..<numWaveguides).map { _ in Int.random(in: minLength...maxLength) }
        case .aligned:
            alignedLengths = (0..<numWaveguides).map { idx in
                let base = minLength + (idx % max(1, numWaveguides / 8)) * (maxLength - minLength) / max(1, numWaveguides / 8)
                return max(minLength, min(maxLength, base - base % 32 + 32))
            }
        case .powerOfTwo:
            alignedLengths = (0..<numWaveguides).map { idx in
                let base = minLength + idx
                let power = Int(pow(2.0, ceil(log2(Double(max(base, 2))))))
                return max(minLength, min(maxLength, power))
            }
        }
        return alignedLengths
    }

    override func makeWaveguideState(for length: Int, index: Int) -> WaveguideState {
        let inputTapPos: UInt32
        let outputTapPos: UInt32

        switch lengthDistribution {
        case .aligned, .powerOfTwo:
            inputTapPos = UInt32(length / 4)
            outputTapPos = UInt32((3 * length) / 4)
        default:
            inputTapPos = UInt32.random(in: 0..<UInt32(length))
            outputTapPos = UInt32.random(in: 0..<UInt32(length))
        }

        let gain = 0.5 + 0.5 * Float(index % 8) / 8.0
        return WaveguideState(
            length: UInt32(length),
            inputTapPos: inputTapPos,
            outputTapPos: outputTapPos,
            writePos: 0,
            gain: gain,
            reflection: reflectionCoeff,
            damping: dampingCoeff,
            padding: 0
        )
    }

    override func initializeInputSignal() {
        guard let inputSignalBuffer = inputSignalBuffer,
              let hostInputSignal = hostInputSignal else { return }

        let pointer = inputSignalBuffer.contents().bindMemory(to: Float.self, capacity: bufferSize)
        for i in 0..<bufferSize {
            var sample: Float = 0.0
            if i == 0 {
                sample += 1.0
            }
            let phase = 2.0 * Float.pi * Float(i) / Float(bufferSize)
            sample += 0.1 * sin(phase * 4.0)
            pointer[i] = sample
            hostInputSignal[i] = sample
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
            width: min(pipelineState.maxTotalThreadsPerThreadgroup, max(64, numWaveguides / 4)),
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
            label: "DWG1DAccel"
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
        enhancedMetadata["benchmark"] = "DWG1DAccel"
        enhancedMetadata["numWaveguides"] = numWaveguides
        enhancedMetadata["minLength"] = minLength
        enhancedMetadata["maxLength"] = maxLength
        enhancedMetadata["lengthDistribution"] = String(describing: lengthDistribution)
        enhancedMetadata["alignedLengths"] = alignedLengths.prefix(min(10, alignedLengths.count))
        enhancedMetadata["reflectionCoeff"] = reflectionCoeff
        enhancedMetadata["dampingCoeff"] = dampingCoeff
        enhancedMetadata["optimizations"] = "aligned_lengths,coalesced_memory"

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

    override func cleanup() {
        alignedLengths.removeAll()
        super.cleanup()
    }
}
