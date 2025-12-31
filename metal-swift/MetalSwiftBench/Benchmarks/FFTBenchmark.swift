import Foundation
import Metal
import Accelerate

final class FFTBenchmark: BaseBenchmark {
    private var inputBuffer: MTLBuffer?
    private var outputBuffer: MTLBuffer?
    private let fftSize: Int = 1024  // Default FFT size

    private var hostInputBuffer: UnsafeMutablePointer<Float>?
    private var cpuGoldenBuffer: UnsafeMutablePointer<Float>?

    private var fftSetup: vDSP_DFT_Setup?

    required init(device: MTLDevice, bufferSize: Int, trackCount: Int) throws {
        try super.init(device: device, bufferSize: bufferSize, trackCount: trackCount)

        fftSetup = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(fftSize), vDSP_DFT_Direction.FORWARD)
        guard fftSetup != nil else {
            throw BenchmarkError.invalidConfiguration("Failed to create vDSP FFT setup")
        }

        self.kernelName = "BenchmarkFFT"
    }

    override func setup() throws {
        try super.setup()

        let inputSizeBytes = trackCount * bufferSize * MemoryLayout<Float>.size
        let outputSizeBytes = trackCount * (fftSize / 2 + 1) * 2 * MemoryLayout<Float>.size

        guard let inBuffer = device.makeBuffer(length: inputSizeBytes, options: .storageModeShared) else {
            throw BenchmarkError.bufferCreationFailed(size: inputSizeBytes)
        }
        inputBuffer = inBuffer

        guard let outBuffer = device.makeBuffer(length: outputSizeBytes, options: .storageModeShared) else {
            throw BenchmarkError.bufferCreationFailed(size: outputSizeBytes)
        }
        outputBuffer = outBuffer

        hostInputBuffer = UnsafeMutablePointer<Float>.allocate(capacity: trackCount * bufferSize)
        cpuGoldenBuffer = UnsafeMutablePointer<Float>.allocate(capacity: trackCount * (fftSize / 2 + 1) * 2)

        let inputPointer = inBuffer.contents().bindMemory(to: Float.self, capacity: trackCount * bufferSize)
        for i in 0..<(trackCount * bufferSize) {
            let value = Float.random(in: -1...1)
            inputPointer[i] = value
            hostInputBuffer![i] = value
        }

        memset(cpuGoldenBuffer, 0, trackCount * (fftSize / 2 + 1) * 2 * MemoryLayout<Float>.size)
        calculateCPUGoldenReference()
    }

    private func calculateCPUGoldenReference() {
        guard let fftSetup = fftSetup,
              let hostInputBuffer = hostInputBuffer,
              let cpuGoldenBuffer = cpuGoldenBuffer else {
            return
        }

        var realIn = [Float](repeating: 0.0, count: fftSize)
        var imagIn = [Float](repeating: 0.0, count: fftSize)
        var realOut = [Float](repeating: 0.0, count: fftSize)
        var imagOut = [Float](repeating: 0.0, count: fftSize)

        let outputStride = (fftSize / 2 + 1) * 2

        for trackIdx in 0..<trackCount {
            let inputBase = trackIdx * bufferSize
            let copyCount = min(bufferSize, fftSize)

            for n in 0..<copyCount {
                realIn[n] = hostInputBuffer[inputBase + n]
            }
            if copyCount < fftSize {
                for n in copyCount..<fftSize {
                    realIn[n] = 0.0
                }
            }
            for n in 0..<fftSize {
                imagIn[n] = 0.0
            }

            realIn.withUnsafeBufferPointer { realInPtr in
                imagIn.withUnsafeBufferPointer { imagInPtr in
                    realOut.withUnsafeMutableBufferPointer { realOutPtr in
                        imagOut.withUnsafeMutableBufferPointer { imagOutPtr in
                            vDSP_DFT_Execute(
                                fftSetup,
                                realInPtr.baseAddress!,
                                imagInPtr.baseAddress!,
                                realOutPtr.baseAddress!,
                                imagOutPtr.baseAddress!
                            )
                        }
                    }
                }
            }

            let outputBase = trackIdx * outputStride
            for k in 0..<(fftSize / 2 + 1) {
                cpuGoldenBuffer[outputBase + k * 2] = realOut[k]
                cpuGoldenBuffer[outputBase + k * 2 + 1] = imagOut[k]
            }
        }
    }

    override func performBenchmarkIteration() throws {
        guard let inputBuffer = inputBuffer,
              let outputBuffer = outputBuffer else {
            throw BenchmarkError.pipelineCreationFailed
        }

        let bufferSizeParam = Int32(bufferSize)
        try executeKernelWithParams(
            inputBuffer: inputBuffer,
            outputBuffer: outputBuffer,
            parameters: bufferSizeParam,
            label: "FFT1D"
        )
    }

    override func runBenchmark(iterations: Int, warmupIterations: Int = 3) throws -> BenchmarkResult {
        let result = try super.runBenchmark(iterations: iterations, warmupIterations: warmupIterations)

        var enhancedMetadata = result.metadata
        enhancedMetadata["benchmark"] = "FFT1D"
        enhancedMetadata["fftSize"] = fftSize
        enhancedMetadata["totalFFTs"] = trackCount

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
        guard let outputBuffer = self.outputBuffer else {
            return (false, Float.infinity, Float.infinity, 0)
        }

        let outputCapacity = outputBuffer.length / MemoryLayout<Float>.size
        let outputPointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: outputCapacity)
        let complexCapacity = outputCapacity / 2
        let totalBins = min(trackCount * (fftSize / 2 + 1), complexCapacity)
        let tolerance: Float = 1e-4  // Tolerance for floating point FFT comparison

        return compareWithGolden(
            gpuPointer: outputPointer,
            cpuPointer: cpuGoldenBuffer!,
            sampleCount: totalBins,
            tolerance: tolerance,
            metric: { gpuPtr, cpuPtr, index in
                let base = index * 2
                let gpuReal = gpuPtr[base]
                let gpuImag = gpuPtr[base + 1]
                let cpuReal = cpuPtr[base]
                let cpuImag = cpuPtr[base + 1]
                return hypot(gpuReal - cpuReal, gpuImag - cpuImag)
            }
        )
    }

    override func cleanup() {
        inputBuffer = nil
        outputBuffer = nil
    }

    deinit {
        hostInputBuffer?.deallocate()
        cpuGoldenBuffer?.deallocate()

        if let fftSetup = fftSetup {
            vDSP_DFT_DestroySetup(fftSetup)
        }
    }
}
