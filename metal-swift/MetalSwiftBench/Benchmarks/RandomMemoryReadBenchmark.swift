import Foundation
import Metal

final class RandomMemoryReadBenchmark: BaseBenchmark {
    
    var sampleMemorySize: Int = 512 * 1024 * 1024 / MemoryLayout<Float>.size  // ~128M samples
    var minLoopLength: Int = 1000
    var maxLoopLength: Int = 48000
    
    private var sampleMemoryBuffer: MTLBuffer?
    private var playheadBuffer: MTLBuffer?
    private var outputBuffer: MTLBuffer?
    private var hostSampleMemory: UnsafeMutablePointer<Float>?
    private var hostPlayheads: UnsafeMutablePointer<Int32>?
    private var cpuGoldenBuffer: UnsafeMutablePointer<Float>?
    
    required init(device: MTLDevice, bufferSize: Int, trackCount: Int) throws {
        try super.init(device: device, bufferSize: bufferSize, trackCount: trackCount)
        self.kernelName = "BenchmarkRndMem"
    }
    
    convenience init(device: MTLDevice, bufferSize: Int, trackCount: Int, 
                     sampleMemorySize: Int, minLoopLength: Int, maxLoopLength: Int) throws {
        try self.init(device: device, bufferSize: bufferSize, trackCount: trackCount)
        self.sampleMemorySize = sampleMemorySize
        self.minLoopLength = minLoopLength
        self.maxLoopLength = maxLoopLength
    }
    
    override func setup() throws {
        try super.setup()
        
        // Precompute sizes so track-major storage matches the GPU kernel math.
        let sampleMemBufferSize = sampleMemorySize * MemoryLayout<Float>.size
        let playheadBufferSize = trackCount * MemoryLayout<Int32>.size
        let outputBufferSize = trackCount * bufferSize * MemoryLayout<Float>.size
        
        sampleMemoryBuffer = try allocateBuffer(length: sampleMemBufferSize)
        playheadBuffer = try allocateBuffer(length: playheadBufferSize)
        outputBuffer = try allocateBuffer(length: outputBufferSize)
        
        hostSampleMemory = UnsafeMutablePointer<Float>.allocate(capacity: sampleMemorySize)
        hostPlayheads = UnsafeMutablePointer<Int32>.allocate(capacity: trackCount)
        cpuGoldenBuffer = UnsafeMutablePointer<Float>.allocate(capacity: trackCount * bufferSize)
        
        let samplePointer = sampleMemoryBuffer!.contents().bindMemory(to: Float.self, capacity: sampleMemorySize)
        for i in 0..<sampleMemorySize {
            let value = Float.random(in: -1.0...1.0)
            samplePointer[i] = value
            hostSampleMemory![i] = value
        }
        
        let playheadPointer = playheadBuffer!.contents().bindMemory(to: Int32.self, capacity: trackCount)
        let maxStartPosition = sampleMemorySize - maxLoopLength - bufferSize
        
        for i in 0..<trackCount {
            let loopLength = Int32.random(in: Int32(minLoopLength)...Int32(maxLoopLength))
            let maxStart = max(0, maxStartPosition - Int(loopLength))
            let startPosition = Int32.random(in: 0...Int32(maxStart))
            
            playheadPointer[i] = startPosition
            hostPlayheads![i] = startPosition
        }
        
        memset(outputBuffer!.contents(), 0, outputBufferSize)
        memset(cpuGoldenBuffer, 0, trackCount * bufferSize * MemoryLayout<Float>.size)
        
        calculateCPUGoldenReference()
    }
    
    private func calculateCPUGoldenReference() {
        for trackIdx in 0..<trackCount {
            let playhead = Int(hostPlayheads![trackIdx])
            
            let trackBase = trackIdx * bufferSize
            for sampleIdx in 0..<bufferSize {
                let sampleMemIndex = playhead + sampleIdx
                if sampleMemIndex < sampleMemorySize {
                    cpuGoldenBuffer![trackBase + sampleIdx] = hostSampleMemory![sampleMemIndex]
                } else {
                    cpuGoldenBuffer![trackBase + sampleIdx] = 0.0
                }
            }
        }
    }
    
    override func performBenchmarkIteration() throws {
        guard let sampleMemoryBuffer = self.sampleMemoryBuffer,
              let playheadBuffer = self.playheadBuffer,
              let outputBuffer = self.outputBuffer else {
            throw BenchmarkError.pipelineCreationFailed
        }

        guard let pipelineState = pipelineState,
              let threadsPerThreadgroup = threadsPerThreadgroup,
              let threadgroupsPerGrid = threadgroupsPerGrid else {
            throw BenchmarkError.pipelineCreationFailed
        }

        var params = makeParams()

        let paramBinding = CommandEncoder.ParameterBinding(params, index: 3)

        try commandEncoder.execute(
            pipelineState: pipelineState,
            buffers: [
                (sampleMemoryBuffer, 0),
                (playheadBuffer, 1),
                (outputBuffer, 2)
            ],
            parameterBindings: [paramBinding],
            threadsPerThreadgroup: threadsPerThreadgroup,
            threadgroupsPerGrid: threadgroupsPerGrid,
            label: "RandomMemoryRead"
        )
    }

    override func runBenchmark(iterations: Int, warmupIterations: Int = 3) throws -> BenchmarkResult {
        let result = try super.runBenchmark(iterations: iterations, warmupIterations: warmupIterations)

        var enhancedMetadata = result.metadata
        enhancedMetadata["sampleMemorySize"] = sampleMemorySize
        enhancedMetadata["minLoopLength"] = minLoopLength
        enhancedMetadata["maxLoopLength"] = maxLoopLength

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
        
        let outputPointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: trackCount * bufferSize)
        let totalSamples = trackCount * bufferSize
        let tolerance: Float = 1e-6  // Should be exact for this benchmark

        return compareWithGolden(
            gpuPointer: outputPointer,
            cpuPointer: cpuGoldenBuffer!,
            sampleCount: totalSamples,
            tolerance: tolerance
        )
    }
    
    deinit {
        hostSampleMemory?.deallocate()
        hostPlayheads?.deallocate()
        cpuGoldenBuffer?.deallocate()
    }
}
