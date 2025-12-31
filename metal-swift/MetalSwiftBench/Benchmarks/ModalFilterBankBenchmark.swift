import Foundation
import Metal

final class ModalFilterBankBenchmark: BaseBenchmark {
    
    var numModes: Int = 1024 * 1024
    let numModeParams: Int = 8
    var outputTracks: Int = 32
    
    private var inputBuffer: MTLBuffer?
    private var outputBuffer: MTLBuffer?
    private var hostInputBuffer: UnsafeMutablePointer<Float>?
    private var cpuGoldenBuffer: UnsafeMutablePointer<Float>?
    
    required init(device: MTLDevice, bufferSize: Int, trackCount: Int) throws {
        try super.init(device: device, bufferSize: bufferSize, trackCount: trackCount)
        self.kernelName = "BenchmarkModalFilterBank"
        
        // Scale number of modes based on track count for performance tuning
        self.numModes = min(1024 * trackCount, 1024 * 1024)
        self.outputTracks = min(trackCount, 32)
    }
    
    convenience init(device: MTLDevice, bufferSize: Int, trackCount: Int, numModes: Int, outputTracks: Int) throws {
        try self.init(device: device, bufferSize: bufferSize, trackCount: trackCount)
        self.numModes = numModes
        self.outputTracks = outputTracks
    }
    
    override func setup() throws {
        try super.setup()
        
        let inputBufferSize = numModes * numModeParams * MemoryLayout<Float>.size
        let outputBufferSize = bufferSize * outputTracks * MemoryLayout<Float>.size
        
        inputBuffer = try allocateBuffer(length: inputBufferSize)
        outputBuffer = try allocateBuffer(length: outputBufferSize)
        
        hostInputBuffer = UnsafeMutablePointer<Float>.allocate(capacity: numModes * numModeParams)
        cpuGoldenBuffer = UnsafeMutablePointer<Float>.allocate(capacity: bufferSize * outputTracks)
        
        let inputPointer = inputBuffer!.contents().bindMemory(to: Float.self, capacity: numModes * numModeParams)
        
        for i in 0..<numModes {
            let baseIdx = i * numModeParams
            inputPointer[baseIdx + 0] = Float.random(in: 0.0...1.0)
            hostInputBuffer![baseIdx + 0] = inputPointer[baseIdx + 0]
            
            inputPointer[baseIdx + 1] = Float.random(in: 0.0...0.45)  
            hostInputBuffer![baseIdx + 1] = inputPointer[baseIdx + 1]
            
            inputPointer[baseIdx + 2] = Float.random(in: 0.0...2.0 * Float.pi)
            hostInputBuffer![baseIdx + 2] = inputPointer[baseIdx + 2]
            
            inputPointer[baseIdx + 3] = Float.random(in: -1.0...1.0)
            hostInputBuffer![baseIdx + 3] = inputPointer[baseIdx + 3]
            
            inputPointer[baseIdx + 4] = Float.random(in: -1.0...1.0)
            hostInputBuffer![baseIdx + 4] = inputPointer[baseIdx + 4]
            
            for j in 5..<numModeParams {
                inputPointer[baseIdx + j] = 0.0
                hostInputBuffer![baseIdx + j] = 0.0
            }
        }
        
        memset(outputBuffer!.contents(), 0, outputBufferSize)
        memset(cpuGoldenBuffer, 0, outputBufferSize)
        
        calculateCPUGoldenReference()
    }
    
    private func calculateCPUGoldenReference() {
        for i in 0..<(bufferSize * outputTracks) {
            cpuGoldenBuffer![i] = 0.0
        }
        
        for modeIdx in 0..<numModes {
            let baseIdx = modeIdx * numModeParams
            let amp = hostInputBuffer![baseIdx + 0]
            let freq = hostInputBuffer![baseIdx + 1] 
            var stateRe = hostInputBuffer![baseIdx + 3]
            var stateIm = hostInputBuffer![baseIdx + 4]
            
            // Spread modes across tracks round-robin to mimic the GPU's accumulation pattern.
            let outputTrack = modeIdx % outputTracks
            
            for sampleIdx in 0..<bufferSize {
                let cosVal = cosf(2.0 * Float.pi * freq)
                let sinVal = sinf(2.0 * Float.pi * freq)
                
                let newRe = stateRe * cosVal - stateIm * sinVal
                let newIm = stateRe * sinVal + stateIm * cosVal
                
                stateRe = newRe
                stateIm = newIm
                
                cpuGoldenBuffer![outputTrack * bufferSize + sampleIdx] += amp * stateRe
            }
        }
    }
    
    override func performBenchmarkIteration() throws {
        guard let pipelineState = pipelineState,
              let inputBuffer = inputBuffer,
              let outputBuffer = outputBuffer else {
            throw BenchmarkError.pipelineCreationFailed
        }

        memset(outputBuffer.contents(), 0, bufferSize * outputTracks * MemoryLayout<Float>.size)

        var modeCount = UInt32(numModes)
        var outputTrackCount = UInt32(outputTracks)
        var params = makeParams()

        let threadsPerThreadgroup = MTLSize(width: min(256, numModes), height: 1, depth: 1)
        let threadgroupsPerGrid = MTLSize(
            width: (numModes + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
            height: 1,
            depth: 1
        )

        let parameterBindings: [CommandEncoder.ParameterBinding] = [
            CommandEncoder.ParameterBinding(modeCount, index: 2),
            CommandEncoder.ParameterBinding(outputTrackCount, index: 3),
            CommandEncoder.ParameterBinding(params, index: 4)
        ]

        try commandEncoder.execute(
            pipelineState: pipelineState,
            buffers: [(inputBuffer, 0), (outputBuffer, 1)],
            parameterBindings: parameterBindings,
            threadsPerThreadgroup: threadsPerThreadgroup,
            threadgroupsPerGrid: threadgroupsPerGrid,
            label: "ModalFilterBank"
        )
    }

    override func runBenchmark(iterations: Int, warmupIterations: Int = 3) throws -> BenchmarkResult {
        let result = try super.runBenchmark(iterations: iterations, warmupIterations: warmupIterations)

        var enhancedMetadata = result.metadata
        enhancedMetadata["numModes"] = numModes
        enhancedMetadata["outputTracks"] = outputTracks
        enhancedMetadata["numModeParams"] = numModeParams

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
        
        let outputPointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: bufferSize * outputTracks)
        let totalSamples = bufferSize * outputTracks
        let tolerance: Float = 1e-4  // Relative tolerance for floating point comparison

        return compareWithGolden(
            gpuPointer: outputPointer,
            cpuPointer: cpuGoldenBuffer!,
            sampleCount: totalSamples,
            tolerance: tolerance,
            metric: { gpuPtr, cpuPtr, index in
                let trackIdx = index / bufferSize
                let sampleIdx = index % bufferSize
                let bufferIndex = trackIdx * bufferSize + sampleIdx
                let gpuValue = gpuPtr[bufferIndex]
                let cpuValue = cpuPtr[bufferIndex]
                let error = abs(gpuValue - cpuValue)
                return cpuValue != 0 ? error / abs(cpuValue) : error
            }
        )
    }
    
    deinit {
        hostInputBuffer?.deallocate()
        cpuGoldenBuffer?.deallocate()
    }
}
