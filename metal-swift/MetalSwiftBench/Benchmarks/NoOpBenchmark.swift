//
//  NoOpBenchmark.swift
//  MetalSwiftBench
//
//  Measures GPU kernel launch overhead with an empty kernel
//

import Foundation
import Metal

final class NoOpBenchmark: BaseBenchmark {
    private var dummyBuffer: MTLBuffer?
    
    required init(device: MTLDevice, bufferSize: Int, trackCount: Int) throws {
        try super.init(device: device, bufferSize: bufferSize, trackCount: trackCount)
        self.kernelName = "noopFn"
    }
    
    override func setup() throws {
        try super.setup()
        
        // Create a dummy buffer because Metal syntax requires at least one input
        guard let buffer = device.makeBuffer(length: 1024, options: .storageModeShared) else {
            throw BenchmarkError.bufferCreationFailed(size: 1024)
        }
        dummyBuffer = buffer
    }
    
    override func performBenchmarkIteration() throws {
        guard let pipelineState = pipelineState,
              let dummyBuffer = dummyBuffer else {
            throw BenchmarkError.pipelineCreationFailed
        }

        // Create command buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw BenchmarkError.failedToCreateCommandQueue
        }

        // Create compute encoder
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw BenchmarkError.failedToCreateCommandQueue
        }

        // Set pipeline state
        computeEncoder.setComputePipelineState(pipelineState)

        // Set the dummy buffer
        computeEncoder.setBuffer(dummyBuffer, offset: 0, index: 0)

        // Dispatch minimal workload (1x1x1)
        let threadsPerGrid = MTLSize(width: 1, height: 1, depth: 1)
        let threadsPerThreadgroup = MTLSize(
            width: min(pipelineState.maxTotalThreadsPerThreadgroup, 1),
            height: 1,
            depth: 1
        )

        computeEncoder.dispatchThreads(threadsPerGrid,
                                     threadsPerThreadgroup: threadsPerThreadgroup)

        // End encoding
        computeEncoder.endEncoding()

        // Commit and wait
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    override func cleanup() {
        dummyBuffer = nil
    }
}