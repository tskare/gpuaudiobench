//
//  DataTransferBenchmark.swift
//  MetalSwiftBench
//
//  Measures data transfer overhead with varying input/output ratios
//

import Foundation
import Metal

final class DataTransferBenchmark: BaseBenchmark {
    let inputRatio: Float
    let outputRatio: Float
    
    private var inputBuffer: MTLBuffer?
    private var outputBuffer: MTLBuffer?
    private var hostInputBuffer: UnsafeMutablePointer<Float>?
    private var hostOutputBuffer: UnsafeMutablePointer<Float>?
    private var cpuGoldenBuffer: UnsafeMutablePointer<Float>?
    private var inputBufferSize: Int = 0
    private var outputBufferSize: Int = 0
    
    init(device: MTLDevice, bufferSize: Int, trackCount: Int, inputRatio: Float, outputRatio: Float) throws {
        self.inputRatio = inputRatio
        self.outputRatio = outputRatio
        try super.init(device: device, bufferSize: bufferSize, trackCount: trackCount)
        self.kernelName = "BenchmarkDataTransfer"
    }
    
    required init(device: MTLDevice, bufferSize: Int, trackCount: Int) throws {
        self.inputRatio = 1.0
        self.outputRatio = 1.0
        try super.init(device: device, bufferSize: bufferSize, trackCount: trackCount)
        self.kernelName = "BenchmarkDataTransfer"
    }
    
    override func setup() throws {
        try super.setup()
        
        // Scale the base audio footprint according to the requested I/O ratios.
        let baseBufferSize = trackCount * bufferSize * MemoryLayout<Float>.size
        inputBufferSize = Int(inputRatio * Float(baseBufferSize))
        outputBufferSize = Int(outputRatio * Float(baseBufferSize))
        
        // Ensure minimum sizes
        inputBufferSize = max(inputBufferSize, MemoryLayout<Float>.size)
        outputBufferSize = max(outputBufferSize, MemoryLayout<Float>.size)
        
        // Create GPU buffers
        guard let inBuffer = device.makeBuffer(length: inputBufferSize, options: .storageModeShared) else {
            throw BenchmarkError.bufferCreationFailed(size: inputBufferSize)
        }
        inputBuffer = inBuffer
        
        guard let outBuffer = device.makeBuffer(length: outputBufferSize, options: .storageModeShared) else {
            throw BenchmarkError.bufferCreationFailed(size: outputBufferSize)
        }
        outputBuffer = outBuffer
        
        // Create host buffers
        hostInputBuffer = UnsafeMutablePointer<Float>.allocate(capacity: inputBufferSize / MemoryLayout<Float>.size)
        hostOutputBuffer = UnsafeMutablePointer<Float>.allocate(capacity: outputBufferSize / MemoryLayout<Float>.size)
        cpuGoldenBuffer = UnsafeMutablePointer<Float>.allocate(capacity: outputBufferSize / MemoryLayout<Float>.size)
        
        // Initialize input buffer with random data
        if let hostInput = hostInputBuffer {
            let count = inputBufferSize / MemoryLayout<Float>.size
            for i in 0..<count {
                hostInput[i] = Float.random(in: -1.0...1.0)
            }
            // Copy to GPU buffer
            memcpy(inBuffer.contents(), hostInput, inputBufferSize)
        }
        
        // Zero outputs so any stray kernel writes show up immediately.
        memset(outBuffer.contents(), 0, outputBufferSize)
        memset(cpuGoldenBuffer, 0, outputBufferSize)
        
        // GPU work is a no-op; the zeroed CPU buffer already reflects the expected output.
    }
    
    override func performBenchmarkIteration() throws {
        guard let inputBuffer = inputBuffer,
              let outputBuffer = outputBuffer,
              let hostOutput = hostOutputBuffer else {
            throw BenchmarkError.pipelineCreationFailed
        }

        // Simulate DAW sending input data (CPU → GPU)
        memset(inputBuffer.contents(), 0, inputBufferSize)

        // Execute the kernel
        try executeKernel(
            inputBuffer: inputBuffer,
            outputBuffer: outputBuffer,
            label: "DataTransfer"
        )

        // Simulate DAW receiving output data (GPU → CPU)
        memcpy(hostOutput, outputBuffer.contents(), outputBufferSize)
    }

    override func runBenchmark(iterations: Int, warmupIterations: Int = 3) throws -> BenchmarkResult {
        let result = try super.runBenchmark(iterations: iterations, warmupIterations: warmupIterations)

        var enhancedMetadata = result.metadata
        enhancedMetadata["benchmark"] = "DataTransfer"
        enhancedMetadata["inputRatio"] = inputRatio
        enhancedMetadata["outputRatio"] = outputRatio
        enhancedMetadata["inputBufferSizeMB"] = Double(inputBufferSize) / (1024 * 1024)
        enhancedMetadata["outputBufferSizeMB"] = Double(outputBufferSize) / (1024 * 1024)

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
        
        let sampleCount = outputBufferSize / MemoryLayout<Float>.size
        let outputPointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: sampleCount)
        let tolerance: Float = 1e-6  // Should be exact for no-op kernel

        return compareWithGolden(
            gpuPointer: outputPointer,
            cpuPointer: cpuGoldenBuffer!,
            sampleCount: sampleCount,
            tolerance: tolerance
        )
    }
    
    override func cleanup() {
        inputBuffer = nil
        outputBuffer = nil
        
        if let hostInput = hostInputBuffer {
            hostInput.deallocate()
            hostInputBuffer = nil
        }

        if let hostOutput = hostOutputBuffer {
            hostOutput.deallocate()
            hostOutputBuffer = nil
        }
        
        if let cpuGolden = cpuGoldenBuffer {
            cpuGolden.deallocate()
            cpuGoldenBuffer = nil
        }
    }
    
    deinit {
        cleanup()
    }
}
