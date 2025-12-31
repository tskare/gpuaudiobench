import Foundation
import Metal

final class IIRFilterBenchmark: BaseBenchmark {
    private var inputBuffer: MTLBuffer?
    private var outputBuffer: MTLBuffer?
    private var coefficientsBuffer: MTLBuffer?
    private var stateBuffer: MTLBuffer?
    
    private var hostInputBuffer: UnsafeMutablePointer<Float>?
    private var cpuGoldenBuffer: UnsafeMutablePointer<Float>?
    private var cpuFilterStates: UnsafeMutablePointer<Float>?  // 2 states per track
    private var filterCoefficients: IIRCoefficients = IIRCoefficients(b0: 0, b1: 0, b2: 0, a1: 0, a2: 0)
    
    required init(device: MTLDevice, bufferSize: Int, trackCount: Int) throws {
        try super.init(device: device, bufferSize: bufferSize, trackCount: trackCount)
        self.kernelName = "BenchmarkIIRBiquad"
    }
    
    override func setup() throws {
        try super.setup()
        
        // Track-major buffers keep GPU and CPU index math identical.
        let audioBufferSize = trackCount * bufferSize * MemoryLayout<Float>.size
        let coefficientsSize = MemoryLayout<IIRCoefficients>.size
        let stateSize = trackCount * 2 * MemoryLayout<Float>.size  // 2 state variables per track
        
        inputBuffer = try allocateBuffer(length: audioBufferSize)
        outputBuffer = try allocateBuffer(length: audioBufferSize)
        coefficientsBuffer = try allocateBuffer(length: coefficientsSize)
        stateBuffer = try allocateBuffer(length: stateSize)
        
        // Mirror GPU layout on the host so the CPU path exercises the same ordering.
        hostInputBuffer = UnsafeMutablePointer<Float>.allocate(capacity: trackCount * bufferSize)
        cpuGoldenBuffer = UnsafeMutablePointer<Float>.allocate(capacity: trackCount * bufferSize)
        cpuFilterStates = UnsafeMutablePointer<Float>.allocate(capacity: trackCount * 2)
        
        let coeffPtr = coefficientsBuffer!.contents().bindMemory(to: IIRCoefficients.self, capacity: 1)
        
        let omega = Float.pi / 2  // fs/4 normalized frequency
        let sin_omega = sin(omega)
        let cos_omega = cos(omega)
        let alpha = sin_omega / sqrt(2)  // Q = 0.707
        
        let a0 = 1 + alpha
        let b0 = ((1 - cos_omega) / 2) / a0
        let b1 = (1 - cos_omega) / a0
        let b2 = ((1 - cos_omega) / 2) / a0
        let a1 = (-2 * cos_omega) / a0
        let a2 = (1 - alpha) / a0
        
        coeffPtr.pointee.b0 = b0
        coeffPtr.pointee.b1 = b1
        coeffPtr.pointee.b2 = b2
        coeffPtr.pointee.a1 = a1
        coeffPtr.pointee.a2 = a2
        
        filterCoefficients.b0 = b0
        filterCoefficients.b1 = b1
        filterCoefficients.b2 = b2
        filterCoefficients.a1 = a1
        filterCoefficients.a2 = a2
        
        let inputPointer = inputBuffer!.contents().bindMemory(to: Float.self, capacity: trackCount * bufferSize)
        for i in 0..<(trackCount * bufferSize) {
            let value = Float.random(in: -1...1)
            inputPointer[i] = value
            hostInputBuffer![i] = value
        }
        
        memset(stateBuffer!.contents(), 0, stateSize)
        memset(cpuGoldenBuffer, 0, trackCount * bufferSize * MemoryLayout<Float>.size)
        memset(cpuFilterStates, 0, trackCount * 2 * MemoryLayout<Float>.size)
        
        calculateCPUGoldenReference()
    }
    
    private func calculateCPUGoldenReference() {
        guard let cpuStates = cpuFilterStates else { return }
        
        for trackIdx in 0..<trackCount {
            let state = cpuStates.advanced(by: trackIdx * 2)
            var z1 = state[0]
            var z2 = state[1]
            let trackBase = trackIdx * bufferSize
            
            for sampleIdx in 0..<bufferSize {
                let sampleIndex = trackBase + sampleIdx
                let x = hostInputBuffer![sampleIndex]
                
                let w = x - filterCoefficients.a1 * z1 - filterCoefficients.a2 * z2
                let y = filterCoefficients.b0 * w + filterCoefficients.b1 * z1 + filterCoefficients.b2 * z2
                
                z2 = z1
                z1 = w
                
                cpuGoldenBuffer![sampleIndex] = y
            }
            
            state[0] = z1
            state[1] = z2
        }
    }
    
    private func resetFilterStatesForIteration() {
        let stateByteCount = trackCount * 2 * MemoryLayout<Float>.size

        if let stateBuffer = stateBuffer {
            memset(stateBuffer.contents(), 0, stateByteCount)
        }

        resetCPUReferenceState()
    }

    private func resetCPUReferenceState() {
        let stateByteCount = trackCount * 2 * MemoryLayout<Float>.size
        if let cpuStates = cpuFilterStates {
            memset(cpuStates, 0, stateByteCount)
        }
    }
    
    override func performBenchmarkIteration() throws {
        guard let pipelineState = pipelineState,
              let inputBuffer = inputBuffer,
              let outputBuffer = outputBuffer,
              let coefficientsBuffer = coefficientsBuffer,
              let stateBuffer = stateBuffer else {
            throw BenchmarkError.pipelineCreationFailed
        }

        resetFilterStatesForIteration()

        var params = makeParams(gainValue: 1.0)

        let paramBinding = CommandEncoder.ParameterBinding(params, index: 4)

        try commandEncoder.execute(
            pipelineState: pipelineState,
            buffers: [
                (inputBuffer, 0),
                (outputBuffer, 1),
                (coefficientsBuffer, 2),
                (stateBuffer, 3)
            ],
            parameterBindings: [paramBinding],
            threadsPerThreadgroup: threadsPerThreadgroup!,
            threadgroupsPerGrid: threadgroupsPerGrid!,
            label: "IIRFilter"
        )
    }

    override func runBenchmark(iterations: Int, warmupIterations: Int = 3) throws -> BenchmarkResult {
        guard let _ = inputBuffer,
              let _ = outputBuffer,
              let _ = coefficientsBuffer,
              let _ = stateBuffer else {
            return BenchmarkResult(latencies: [], gpuLatencies: [], metadata: ["error": "Setup not completed"])
        }
        
        let result = try super.runBenchmark(iterations: iterations, warmupIterations: warmupIterations)

        var enhancedMetadata = result.metadata
        enhancedMetadata["benchmark"] = "IIRFilter"
        enhancedMetadata["filterType"] = "Butterworth Lowpass"
        enhancedMetadata["cutoffFrequency"] = "fs/4"

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

        resetCPUReferenceState()
        calculateCPUGoldenReference()
        
        let outputPointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: trackCount * bufferSize)
        let totalSamples = trackCount * bufferSize
        let tolerance: Float = 1e-4  // Tolerance for IIR filter precision

        return compareWithGolden(
            gpuPointer: outputPointer,
            cpuPointer: cpuGoldenBuffer!,
            sampleCount: totalSamples,
            tolerance: tolerance,
            metric: { gpuPtr, cpuPtr, index in
                let gpuValue = gpuPtr[index]
                let cpuValue = cpuPtr[index]
                let error = abs(gpuValue - cpuValue)
                return cpuValue != 0 ? error / abs(cpuValue) : error
            }
        )
    }
    
    override func cleanup() {
        super.cleanup()
        inputBuffer = nil
        outputBuffer = nil
        coefficientsBuffer = nil
        stateBuffer = nil
    }
    
    deinit {
        hostInputBuffer?.deallocate()
        cpuGoldenBuffer?.deallocate()
        cpuFilterStates?.deallocate()
    }
}
