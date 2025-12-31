import Foundation
import Metal

final class Convolution1DAccelBenchmark: Convolution1DBaseBenchmark {

    private var fftSize: Int = 0
    private var overlapSize: Int = 0

    private var fftInputBuffer: MTLBuffer?
    private var fftOutputBuffer: MTLBuffer?
    private var irFFTBuffer: MTLBuffer?

    override init(device: MTLDevice, bufferSize: Int, trackCount: Int, impulseResponseLength: Int = 512) throws {
        try super.init(device: device, bufferSize: bufferSize, trackCount: trackCount, impulseResponseLength: impulseResponseLength)
        self.kernelName = "BenchmarkConv1DAccel"
        updateFFTParameters()
    }

    required convenience init(device: MTLDevice, bufferSize: Int, trackCount: Int) throws {
        try self.init(device: device, bufferSize: bufferSize, trackCount: trackCount, impulseResponseLength: 512)
    }

    override func setup() throws {
        try super.setup()

        try setupConvolutionBuffers()

        // Allocate per-track complex spectra for the overlap-save path.
        let fftBufferSize = trackCount * fftSize * 2 * MemoryLayout<Float>.size // Complex numbers

        fftInputBuffer = try allocateBuffer(length: fftBufferSize)
        fftOutputBuffer = try allocateBuffer(length: fftBufferSize)
        irFFTBuffer = try allocateBuffer(length: fftBufferSize)

        // Pre-compute FFT of impulse responses on the GPU
        try precomputeImpulseResponseFFTsOnGPU()
        resetOutputBuffers()

        // Use the time-domain CPU reference to sidestep FFT rounding differences.
        calculateCPUGoldenReference()
    }

    private func precomputeImpulseResponseFFTsOnGPU() throws {
        guard let pipelineState = pipelineState,
              let impulseResponseBuffer = impulseResponseBuffer,
              let irFFTBuffer = irFFTBuffer,
              let outputBuffer = outputBuffer,
              let fftInputBuffer = fftInputBuffer,
              let fftOutputBuffer = fftOutputBuffer else {
            throw BenchmarkError.pipelineCreationFailed
        }

        var numTracks = UInt32(trackCount)
        var irLength = UInt32(impulseResponseLength)
        var fftSizeParam = UInt32(fftSize)
        var params = makeParams()
        var operationMode: UInt32 = 1 // 1 = forward FFT only

        let threadsPerThreadgroup = MTLSize(width: min(256, trackCount), height: 1, depth: 1)
        let threadgroupsPerGrid = MTLSize(
            width: (trackCount + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
            height: 1,
            depth: 1
        )

        let parameterBindings: [CommandEncoder.ParameterBinding] = [
            CommandEncoder.ParameterBinding(numTracks, index: 0),
            CommandEncoder.ParameterBinding(irLength, index: 6),
            CommandEncoder.ParameterBinding(fftSizeParam, index: 7),
            CommandEncoder.ParameterBinding(params, index: 8),
            CommandEncoder.ParameterBinding(operationMode, index: 9)
        ]

        try commandEncoder.execute(
            pipelineState: pipelineState,
            buffers: [
                (impulseResponseBuffer, 1),
                (irFFTBuffer, 2),
                (outputBuffer, 3),
                (fftInputBuffer, 4),
                (fftOutputBuffer, 5)
            ],
            parameterBindings: parameterBindings,
            threadsPerThreadgroup: threadsPerThreadgroup,
            threadgroupsPerGrid: threadgroupsPerGrid,
            label: "Conv1DAccel-Precompute"
        )
    }

    override func performBenchmarkIteration() throws {
        guard let pipelineState = pipelineState,
              let inputBuffer = inputBuffer,
              let irFFTBuffer = irFFTBuffer,
              let outputBuffer = outputBuffer,
              let fftInputBuffer = fftInputBuffer,
              let fftOutputBuffer = fftOutputBuffer else {
            throw BenchmarkError.pipelineCreationFailed
        }

        var numTracks = UInt32(trackCount)
        var irLength = UInt32(impulseResponseLength)
        var fftSizeParam = UInt32(fftSize)
        var params = makeParams()
        var operationMode: UInt32 = 0 // 0 = full convolution

        let threadsPerThreadgroup = MTLSize(width: min(256, trackCount), height: 1, depth: 1)
        let threadgroupsPerGrid = MTLSize(
            width: (trackCount + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
            height: 1,
            depth: 1
        )

        let parameterBindings: [CommandEncoder.ParameterBinding] = [
            CommandEncoder.ParameterBinding(numTracks, index: 0),
            CommandEncoder.ParameterBinding(irLength, index: 6),
            CommandEncoder.ParameterBinding(fftSizeParam, index: 7),
            CommandEncoder.ParameterBinding(params, index: 8),
            CommandEncoder.ParameterBinding(operationMode, index: 9)
        ]

        try commandEncoder.execute(
            pipelineState: pipelineState,
            buffers: [
                (inputBuffer, 1),
                (irFFTBuffer, 2),
                (outputBuffer, 3),
                (fftInputBuffer, 4),
                (fftOutputBuffer, 5)
            ],
            parameterBindings: parameterBindings,
            threadsPerThreadgroup: threadsPerThreadgroup,
            threadgroupsPerGrid: threadgroupsPerGrid,
            label: "Convolution1DAccel"
        )
    }

    override func runBenchmark(iterations: Int, warmupIterations: Int = 3) throws -> BenchmarkResult {
        guard let _ = inputBuffer,
              let _ = irFFTBuffer,
              let _ = outputBuffer else {
            return BenchmarkResult(latencies: [], gpuLatencies: [], metadata: ["error": "Benchmark not properly initialized"])
        }

        let result = try super.runBenchmark(iterations: iterations, warmupIterations: warmupIterations)

        var enhancedMetadata = result.metadata
        enhancedMetadata["benchmark"] = "Conv1D_accel"
        enhancedMetadata["impulseResponseLength"] = impulseResponseLength
        enhancedMetadata["fftSize"] = fftSize
        enhancedMetadata["overlapSize"] = overlapSize

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
        let tolerance: Float = 1e-3  // Relaxed tolerance for FFT-based convolution

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

    override func cleanup() {
        super.cleanup()
        fftInputBuffer = nil
        fftOutputBuffer = nil
        irFFTBuffer = nil
    }

    private func updateFFTParameters() {
        let minFFTSize = max(1, impulseResponseLength + bufferSize - 1)
        let fftExponent = Int(ceil(log2(Double(minFFTSize))))
        fftSize = 1 << fftExponent
        overlapSize = max(0, impulseResponseLength - 1)
    }
}
