import Foundation
import Metal
import simd

final class FDTD3DBenchmark: BaseBenchmark {
    var roomSize: (x: Int, y: Int, z: Int) = (50, 50, 50)
    var absorptionCoeff: Float = 0.2
    var soundSpeed: Float = 343.0  // m/s
    var spatialStep: Float = 0.01  // meters
    var sourcePosition: (x: Int, y: Int, z: Int) = (25, 25, 5)
    var receiverPosition: (x: Int, y: Int, z: Int) = (40, 15, 25)
    
    private var timeStep: Float = 0.0
    private var currentTime: Int = 0
    private var stepsPerSample: Int = 1
    
    private var pressureBuffer: MTLBuffer?
    private var velocityXBuffer: MTLBuffer?
    private var velocityYBuffer: MTLBuffer?
    private var velocityZBuffer: MTLBuffer?
    private var paramsBuffer: MTLBuffer?
    private var inputBuffer: MTLBuffer?
    private var outputBuffer: MTLBuffer?
    
    private var hostInputBuffer: UnsafeMutablePointer<Float>?
    private var cpuGoldenBuffer: UnsafeMutablePointer<Float>?
    private var cpuPressureField: UnsafeMutablePointer<Float>?
    private var cpuVelocityXField: UnsafeMutablePointer<Float>?
    private var cpuVelocityYField: UnsafeMutablePointer<Float>?
    private var cpuVelocityZField: UnsafeMutablePointer<Float>?
    
    private var pressureUpdatePipeline: MTLComputePipelineState?
    private var velocityUpdatePipeline: MTLComputePipelineState?
    private var sourceInjectionPipeline: MTLComputePipelineState?
    private var outputExtractionPipeline: MTLComputePipelineState?
    
    required init(device: MTLDevice, bufferSize: Int, trackCount: Int) throws {
        try super.init(device: device, bufferSize: bufferSize, trackCount: trackCount)
        
        // CFL stability limit keeps the acoustic Courant number below 1/√3 in 3D.
        let cflNumber: Float = 0.5  // Conservative, less than 1/√3 ≈ 0.577
        timeStep = cflNumber * spatialStep / (soundSpeed * sqrt(3.0))
        
        // Determine how many solver ticks land inside one 48 kHz audio sample.
        let samplePeriod: Float = 1.0 / 48000.0  // Assuming 48kHz sample rate
        stepsPerSample = max(1, Int(samplePeriod / timeStep))
        
        self.kernelName = "FDTD3D"
    }
    
    convenience init(device: MTLDevice, bufferSize: Int, trackCount: Int,
                     roomSize: (x: Int, y: Int, z: Int), absorptionCoeff: Float = 0.2) throws {
        try self.init(device: device, bufferSize: bufferSize, trackCount: trackCount)
        self.roomSize = roomSize
        self.absorptionCoeff = absorptionCoeff
        
        timeStep = 0.5 * spatialStep / (soundSpeed * sqrt(3.0))
        let samplePeriod: Float = 1.0 / 48000.0
        stepsPerSample = max(1, Int(samplePeriod / timeStep))
    }
    
    override func setup() throws {
        guard let library = device.makeDefaultLibrary() else {
            throw BenchmarkError.failedToLoadKernel("Default library not found")
        }
        
        guard let pressureFunction = library.makeFunction(name: "fdtd3d_update_pressure") else {
            throw BenchmarkError.failedToLoadKernel("fdtd3d_update_pressure")
        }
        pressureUpdatePipeline = try device.makeComputePipelineState(function: pressureFunction)
        
        guard let velocityFunction = library.makeFunction(name: "fdtd3d_update_velocity") else {
            throw BenchmarkError.failedToLoadKernel("fdtd3d_update_velocity")
        }
        velocityUpdatePipeline = try device.makeComputePipelineState(function: velocityFunction)
        
        guard let sourceFunction = library.makeFunction(name: "fdtd3d_inject_source") else {
            throw BenchmarkError.failedToLoadKernel("fdtd3d_inject_source")
        }
        sourceInjectionPipeline = try device.makeComputePipelineState(function: sourceFunction)
        
        guard let outputFunction = library.makeFunction(name: "fdtd3d_extract_output") else {
            throw BenchmarkError.failedToLoadKernel("fdtd3d_extract_output")
        }
        outputExtractionPipeline = try device.makeComputePipelineState(function: outputFunction)
        
        try setupBuffers()
        
        try initializeFDTDGrids()
        
        try setupCPUReference()
    }
    
    private func setupBuffers() throws {
        // Create ghost cells on every axis so boundary damping applies outside the simulated room.
        let nx = roomSize.x + 2  // +2 for boundary conditions
        let ny = roomSize.y + 2
        let nz = roomSize.z + 2
        
        let pressureSize = nx * ny * nz * MemoryLayout<Float>.size
        let velocityXSize = (nx + 1) * ny * nz * MemoryLayout<Float>.size
        let velocityYSize = nx * (ny + 1) * nz * MemoryLayout<Float>.size
        let velocityZSize = nx * ny * (nz + 1) * MemoryLayout<Float>.size
        
        guard let pBuffer = device.makeBuffer(length: pressureSize, options: .storageModeShared) else {
            throw BenchmarkError.bufferCreationFailed(size: pressureSize)
        }
        pressureBuffer = pBuffer
        
        guard let vxBuffer = device.makeBuffer(length: velocityXSize, options: .storageModeShared) else {
            throw BenchmarkError.bufferCreationFailed(size: velocityXSize)
        }
        velocityXBuffer = vxBuffer
        
        guard let vyBuffer = device.makeBuffer(length: velocityYSize, options: .storageModeShared) else {
            throw BenchmarkError.bufferCreationFailed(size: velocityYSize)
        }
        velocityYBuffer = vyBuffer
        
        guard let vzBuffer = device.makeBuffer(length: velocityZSize, options: .storageModeShared) else {
            throw BenchmarkError.bufferCreationFailed(size: velocityZSize)
        }
        velocityZBuffer = vzBuffer
        
        let paramsSize = MemoryLayout<FDTD3DParams>.size
        guard let paramBuffer = device.makeBuffer(length: paramsSize, options: .storageModeShared) else {
            throw BenchmarkError.bufferCreationFailed(size: paramsSize)
        }
        paramsBuffer = paramBuffer
        
        let audioBufferSize = trackCount * bufferSize * MemoryLayout<Float>.size
        
        guard let inBuffer = device.makeBuffer(length: audioBufferSize, options: .storageModeShared) else {
            throw BenchmarkError.bufferCreationFailed(size: audioBufferSize)
        }
        inputBuffer = inBuffer
        
        guard let outBuffer = device.makeBuffer(length: audioBufferSize, options: .storageModeShared) else {
            throw BenchmarkError.bufferCreationFailed(size: audioBufferSize)
        }
        outputBuffer = outBuffer
        
        // Populate kernel parameters once ghost-cell offsets are applied.
        let params = paramsBuffer!.contents().bindMemory(to: FDTD3DParams.self, capacity: 1)
        params.pointee = FDTD3DParams(
            nx: UInt32(nx),
            ny: UInt32(ny),
            nz: UInt32(nz),
            soundSpeed: soundSpeed,
            spatialStep: spatialStep,
            timeStep: timeStep,
            absorptionCoeff: absorptionCoeff,
            sourceX: UInt32(sourcePosition.x + 1),  // +1 for boundary offset
            sourceY: UInt32(sourcePosition.y + 1),
            sourceZ: UInt32(sourcePosition.z + 1),
            receiverX: UInt32(receiverPosition.x + 1),
            receiverY: UInt32(receiverPosition.y + 1),
            receiverZ: UInt32(receiverPosition.z + 1),
            bufferSize: UInt32(bufferSize),
            trackCount: UInt32(trackCount)
        )
    }
    
    private func initializeFDTDGrids() throws {
        // Start the simulation from silence so early reflections only come from the injected source.
        let nx = roomSize.x + 2
        let ny = roomSize.y + 2
        let nz = roomSize.z + 2
        
        memset(pressureBuffer!.contents(), 0, nx * ny * nz * MemoryLayout<Float>.size)
        memset(velocityXBuffer!.contents(), 0, (nx + 1) * ny * nz * MemoryLayout<Float>.size)
        memset(velocityYBuffer!.contents(), 0, nx * (ny + 1) * nz * MemoryLayout<Float>.size)
        memset(velocityZBuffer!.contents(), 0, nx * ny * (nz + 1) * MemoryLayout<Float>.size)
    }
    
    private func setupCPUReference() throws {
        let audioSampleCount = trackCount * bufferSize

        hostInputBuffer = UnsafeMutablePointer<Float>.allocate(capacity: audioSampleCount)
        cpuGoldenBuffer = UnsafeMutablePointer<Float>.allocate(capacity: audioSampleCount)

        let nx = roomSize.x + 2
        let ny = roomSize.y + 2
        let nz = roomSize.z + 2

        cpuPressureField = UnsafeMutablePointer<Float>.allocate(capacity: nx * ny * nz)
        cpuVelocityXField = UnsafeMutablePointer<Float>.allocate(capacity: (nx + 1) * ny * nz)
        cpuVelocityYField = UnsafeMutablePointer<Float>.allocate(capacity: nx * (ny + 1) * nz)
        cpuVelocityZField = UnsafeMutablePointer<Float>.allocate(capacity: nx * ny * (nz + 1))

        resetCPUFields()
        memset(cpuGoldenBuffer, 0, audioSampleCount * MemoryLayout<Float>.size)

        // Initialize with test signal (impulse in first track, silence in others)
        for trackIdx in 0..<trackCount {
            for sampleIdx in 0..<bufferSize {
                let idx = trackIdx * bufferSize + sampleIdx
                hostInputBuffer![idx] = (trackIdx == 0 && sampleIdx == 0) ? 1.0 : 0.0
            }
        }

        // Copy to GPU input buffer
        let inputPointer = inputBuffer!.contents().bindMemory(to: Float.self, capacity: audioSampleCount)
        memcpy(inputPointer, hostInputBuffer, audioSampleCount * MemoryLayout<Float>.size)
    }
    
    private func resetCPUFields() {
        guard let cpuPressureField = cpuPressureField,
              let cpuVelocityXField = cpuVelocityXField,
              let cpuVelocityYField = cpuVelocityYField,
              let cpuVelocityZField = cpuVelocityZField else {
            return
        }

        let nx = roomSize.x + 2
        let ny = roomSize.y + 2
        let nz = roomSize.z + 2

        memset(cpuPressureField, 0, nx * ny * nz * MemoryLayout<Float>.size)
        memset(cpuVelocityXField, 0, (nx + 1) * ny * nz * MemoryLayout<Float>.size)
        memset(cpuVelocityYField, 0, nx * (ny + 1) * nz * MemoryLayout<Float>.size)
        memset(cpuVelocityZField, 0, nx * ny * (nz + 1) * MemoryLayout<Float>.size)
    }

    private func runCPUReferenceSimulation() {
        guard let hostInputBuffer = hostInputBuffer,
              let cpuGoldenBuffer = cpuGoldenBuffer,
              let cpuPressureField = cpuPressureField,
              let cpuVelocityXField = cpuVelocityXField,
              let cpuVelocityYField = cpuVelocityYField,
              let cpuVelocityZField = cpuVelocityZField else {
            return
        }

        let nx = roomSize.x + 2
        let ny = roomSize.y + 2
        let nz = roomSize.z + 2

        guard nx > 2, ny > 2, nz > 2, bufferSize > 0, trackCount > 0 else {
            memset(cpuGoldenBuffer, 0, trackCount * bufferSize * MemoryLayout<Float>.size)
            return
        }

        resetCPUFields()
        memset(cpuGoldenBuffer, 0, trackCount * bufferSize * MemoryLayout<Float>.size)

        let pressure = UnsafeMutableBufferPointer(start: cpuPressureField, count: nx * ny * nz)
        let velocityX = UnsafeMutableBufferPointer(start: cpuVelocityXField, count: (nx + 1) * ny * nz)
        let velocityY = UnsafeMutableBufferPointer(start: cpuVelocityYField, count: nx * (ny + 1) * nz)
        let velocityZ = UnsafeMutableBufferPointer(start: cpuVelocityZField, count: nx * ny * (nz + 1))
        let inputSamples = UnsafeBufferPointer(start: hostInputBuffer, count: trackCount * bufferSize)
        let outputSamples = UnsafeMutableBufferPointer(start: cpuGoldenBuffer, count: trackCount * bufferSize)

        let dtOverRhoDx: Float = timeStep / (1.225 * spatialStep)
        let rhoC2DtOverDx: Float = 1.225 * soundSpeed * soundSpeed * timeStep / spatialStep
        let absorption = max(0.0, min(absorptionCoeff, 1.0))

        let sourceX = sourcePosition.x + 1
        let sourceY = sourcePosition.y + 1
        let sourceZ = sourcePosition.z + 1
        let receiverX = receiverPosition.x + 1
        let receiverY = receiverPosition.y + 1
        let receiverZ = receiverPosition.z + 1

        let sourceIndex = pressureIndex(x: sourceX, y: sourceY, z: sourceZ, nx: nx, ny: ny)
        let receiverIndex = pressureIndex(x: receiverX, y: receiverY, z: receiverZ, nx: nx, ny: ny)

        for sampleIdx in 0..<bufferSize {
            // Inject a soft source per track so CPU and GPU share the same stimulus.
            for trackIdx in 0..<trackCount {
                let audioIdx = trackIdx * bufferSize + sampleIdx
                pressure[sourceIndex] += inputSamples[audioIdx] * 0.1
            }

            for _ in 0..<stepsPerSample {
                // Propagate x-directed velocity along the staggered grid (skip ghost cells).
                if nx > 1 {
                    for z in 0..<nz {
                        for y in 0..<ny {
                            for x in 1..<nx {
                                let vxIdx = velocityXIndex(x: x, y: y, z: z, nxPlus1: nx + 1, ny: ny)
                                let pLeft = pressureIndex(x: x - 1, y: y, z: z, nx: nx, ny: ny)
                                let pRight = pressureIndex(x: x, y: y, z: z, nx: nx, ny: ny)
                                velocityX[vxIdx] -= dtOverRhoDx * (pressure[pRight] - pressure[pLeft])
                            }
                        }
                    }
                }

                // Propagate y-directed velocity, skipping the padded layer.
                if ny > 1 {
                    for z in 0..<nz {
                        for y in 1..<ny {
                            for x in 0..<nx {
                                let vyIdx = velocityYIndex(x: x, y: y, z: z, nx: nx, nyPlus1: ny + 1)
                                let pFront = pressureIndex(x: x, y: y - 1, z: z, nx: nx, ny: ny)
                                let pBack = pressureIndex(x: x, y: y, z: z, nx: nx, ny: ny)
                                velocityY[vyIdx] -= dtOverRhoDx * (pressure[pBack] - pressure[pFront])
                            }
                        }
                    }
                }

                // Propagate z-directed velocity with the same staggered-grid indexing.
                if nz > 1 {
                    for z in 1..<nz {
                        for y in 0..<ny {
                            for x in 0..<nx {
                                let vzIdx = velocityZIndex(x: x, y: y, z: z, nx: nx, ny: ny)
                                let pBottom = pressureIndex(x: x, y: y, z: z - 1, nx: nx, ny: ny)
                                let pTop = pressureIndex(x: x, y: y, z: z, nx: nx, ny: ny)
                                velocityZ[vzIdx] -= dtOverRhoDx * (pressure[pTop] - pressure[pBottom])
                            }
                        }
                    }
                }

                // Update pressure by taking the divergence of the velocity field.
                for z in 0..<nz {
                    for y in 0..<ny {
                        for x in 0..<nx {
                            let pIdx = pressureIndex(x: x, y: y, z: z, nx: nx, ny: ny)
                            if x > 0 && x < nx - 1 && y > 0 && y < ny - 1 && z > 0 && z < nz - 1 {
                                let vxLeft = velocityXIndex(x: x, y: y, z: z, nxPlus1: nx + 1, ny: ny)
                                let vxRight = velocityXIndex(x: x + 1, y: y, z: z, nxPlus1: nx + 1, ny: ny)

                                let vyFront = velocityYIndex(x: x, y: y, z: z, nx: nx, nyPlus1: ny + 1)
                                let vyBack = velocityYIndex(x: x, y: y + 1, z: z, nx: nx, nyPlus1: ny + 1)

                                let vzBottom = velocityZIndex(x: x, y: y, z: z, nx: nx, ny: ny)
                                let vzTop = velocityZIndex(x: x, y: y, z: z + 1, nx: nx, ny: ny)

                                let divergence = (velocityX[vxRight] - velocityX[vxLeft]) +
                                                 (velocityY[vyBack] - velocityY[vyFront]) +
                                                 (velocityZ[vzTop] - velocityZ[vzBottom])

                                pressure[pIdx] -= rhoC2DtOverDx * divergence
                            } else {
                                pressure[pIdx] *= (1.0 - absorption)
                            }
                        }
                    }
                }
            }

            let sampleValue = pressure[receiverIndex] * 0.1
            for trackIdx in 0..<trackCount {
                outputSamples[trackIdx * bufferSize + sampleIdx] = sampleValue
            }
        }
    }

    private func verifyAgainstGoldenReference() -> (passed: Bool, maxError: Float, meanError: Float, sampleCount: Int) {
        guard let outputBuffer = outputBuffer,
              let cpuGoldenBuffer = cpuGoldenBuffer else {
            return (false, Float.infinity, Float.infinity, 0)
        }

        runCPUReferenceSimulation()

        let totalSamples = trackCount * bufferSize
        guard totalSamples > 0 else {
            return (true, 0, 0, 0)
        }

        let gpuOutput = outputBuffer.contents().bindMemory(to: Float.self, capacity: totalSamples)
        let tolerance: Float = 5e-3

        return compareWithGolden(
            gpuPointer: gpuOutput,
            cpuPointer: cpuGoldenBuffer,
            sampleCount: totalSamples,
            tolerance: tolerance
        )
    }

    override func performBenchmarkIteration() throws {
        try runSingleFDTDIteration()
    }

    override func runBenchmark(iterations: Int, warmupIterations: Int = 3) throws -> BenchmarkResult {
        let result = try super.runBenchmark(iterations: iterations, warmupIterations: warmupIterations)

        // Merge benchmark-specific metadata with existing metadata
        var enhancedMetadata = result.metadata
        enhancedMetadata["roomSize"] = "\(roomSize.x)x\(roomSize.y)x\(roomSize.z)"
        enhancedMetadata["absorptionCoeff"] = absorptionCoeff
        enhancedMetadata["soundSpeed"] = soundSpeed
        enhancedMetadata["spatialStep"] = spatialStep
        enhancedMetadata["timeStep"] = timeStep
        enhancedMetadata["stepsPerSample"] = stepsPerSample
        enhancedMetadata["totalGridPoints"] = (roomSize.x + 2) * (roomSize.y + 2) * (roomSize.z + 2)
        enhancedMetadata["memoryUsage"] = calculateMemoryUsage()

        if shouldValidate() {
            let verification = verifyAgainstGoldenReference()
            enhancedMetadata["verificationPassed"] = verification.passed
            enhancedMetadata["maxError"] = verification.maxError
            enhancedMetadata["meanError"] = verification.meanError
            enhancedMetadata["verificationSampleCount"] = verification.sampleCount
        } else {
            enhancedMetadata["verificationSkipped"] = true
        }

        return BenchmarkResult(latencies: result.latencies, gpuLatencies: result.gpuLatencies, metadata: enhancedMetadata)
    }
    
    private func runSingleFDTDIteration() throws {
        try initializeFDTDGrids()

        if let outputBuffer = outputBuffer {
            memset(outputBuffer.contents(), 0, trackCount * bufferSize * MemoryLayout<Float>.size)
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder(),
              let velocityPipeline = velocityUpdatePipeline,
              let pressurePipeline = pressureUpdatePipeline,
              let sourcePipeline = sourceInjectionPipeline,
              let outputPipeline = outputExtractionPipeline,
              let pressureBuffer = pressureBuffer,
              let velocityXBuffer = velocityXBuffer,
              let velocityYBuffer = velocityYBuffer,
              let velocityZBuffer = velocityZBuffer,
              let paramsBuffer = paramsBuffer,
              let inputBuffer = inputBuffer,
              let outputBuffer = outputBuffer else {
            throw BenchmarkError.failedToCreateCommandQueue
        }

        let velocityThreads = makeThreadgroupSize(for: velocityPipeline)
        let velocityThreadgroups = makeThreadgroups(forGrid: velocityGridSize(), threadsPerGroup: velocityThreads)
        let pressureThreads = makeThreadgroupSize(for: pressurePipeline)
        let pressureThreadgroups = makeThreadgroups(forGrid: pressureGridSize(), threadsPerGroup: pressureThreads)
        let ioThreads = MTLSize(width: min(trackCount, 64), height: 1, depth: 1)
        let ioThreadgroups = MTLSize(width: (trackCount + 63) / 64, height: 1, depth: 1)

        for sampleIdx in 0..<bufferSize {
            var sampleIndex = UInt32(sampleIdx)
            encoder.setComputePipelineState(sourcePipeline)
            encoder.setBuffer(pressureBuffer, offset: 0, index: 0)
            encoder.setBuffer(inputBuffer, offset: 0, index: 1)
            encoder.setBuffer(paramsBuffer, offset: 0, index: 2)
            encoder.setBytes(&sampleIndex, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.dispatchThreadgroups(ioThreadgroups, threadsPerThreadgroup: ioThreads)
            
            for _ in 0..<stepsPerSample {
                encoder.setComputePipelineState(velocityPipeline)
                encoder.setBuffer(pressureBuffer, offset: 0, index: 0)
                encoder.setBuffer(velocityXBuffer, offset: 0, index: 1)
                encoder.setBuffer(velocityYBuffer, offset: 0, index: 2)
                encoder.setBuffer(velocityZBuffer, offset: 0, index: 3)
                encoder.setBuffer(paramsBuffer, offset: 0, index: 4)
                encoder.dispatchThreadgroups(velocityThreadgroups, threadsPerThreadgroup: velocityThreads)

                encoder.setComputePipelineState(pressurePipeline)
                encoder.setBuffer(pressureBuffer, offset: 0, index: 0)
                encoder.setBuffer(velocityXBuffer, offset: 0, index: 1)
                encoder.setBuffer(velocityYBuffer, offset: 0, index: 2)
                encoder.setBuffer(velocityZBuffer, offset: 0, index: 3)
                encoder.setBuffer(paramsBuffer, offset: 0, index: 4)
                encoder.dispatchThreadgroups(pressureThreadgroups, threadsPerThreadgroup: pressureThreads)
                currentTime += 1
            }
            
            encoder.setComputePipelineState(outputPipeline)
            encoder.setBuffer(pressureBuffer, offset: 0, index: 0)
            encoder.setBuffer(outputBuffer, offset: 0, index: 1)
            encoder.setBuffer(paramsBuffer, offset: 0, index: 2)
            encoder.setBytes(&sampleIndex, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.dispatchThreadgroups(ioThreadgroups, threadsPerThreadgroup: ioThreads)
        }

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Reset time counter for next iteration
        currentTime = 0
    }
    
    private func makeThreadgroupSize(for pipeline: MTLComputePipelineState) -> MTLSize {
        let maxThreads = max(1, pipeline.maxTotalThreadsPerThreadgroup)
        let baseWidth = max(1, pipeline.threadExecutionWidth)
        let width = min(baseWidth, maxThreads)
        let heightCandidate = maxThreads / width
        let height = max(1, min(4, heightCandidate))
        let depthCandidate = maxThreads / (width * height)
        let depth = max(1, min(4, depthCandidate))
        return MTLSize(width: width, height: height, depth: depth)
    }
    
    private func makeThreadgroups(forGrid grid: MTLSize, threadsPerGroup: MTLSize) -> MTLSize {
        func divRoundUp(_ value: Int, _ divisor: Int) -> Int {
            return (value + divisor - 1) / divisor
        }
        return MTLSize(
            width: divRoundUp(grid.width, max(1, threadsPerGroup.width)),
            height: divRoundUp(grid.height, max(1, threadsPerGroup.height)),
            depth: divRoundUp(grid.depth, max(1, threadsPerGroup.depth))
        )
    }
    
    private func pressureGridSize() -> MTLSize {
        let nx = roomSize.x + 2
        let ny = roomSize.y + 2
        let nz = roomSize.z + 2
        return MTLSize(width: nx, height: ny, depth: nz)
    }
    
    private func velocityGridSize() -> MTLSize {
        let nx = roomSize.x + 2
        let ny = roomSize.y + 2
        let nz = roomSize.z + 2
        return MTLSize(width: nx + 1, height: ny + 1, depth: nz + 1)
    }

    private func pressureIndex(x: Int, y: Int, z: Int, nx: Int, ny: Int) -> Int {
        return z * nx * ny + y * nx + x
    }

    private func velocityXIndex(x: Int, y: Int, z: Int, nxPlus1: Int, ny: Int) -> Int {
        return z * nxPlus1 * ny + y * nxPlus1 + x
    }

    private func velocityYIndex(x: Int, y: Int, z: Int, nx: Int, nyPlus1: Int) -> Int {
        return z * nx * nyPlus1 + y * nx + x
    }

    private func velocityZIndex(x: Int, y: Int, z: Int, nx: Int, ny: Int) -> Int {
        return z * nx * ny + y * nx + x
    }
    
    private func calculateMemoryUsage() -> Int {
        let nx = roomSize.x + 2
        let ny = roomSize.y + 2
        let nz = roomSize.z + 2
        
        let pressureSize = nx * ny * nz * MemoryLayout<Float>.size
        let velocityXSize = (nx + 1) * ny * nz * MemoryLayout<Float>.size
        let velocityYSize = nx * (ny + 1) * nz * MemoryLayout<Float>.size
        let velocityZSize = nx * ny * (nz + 1) * MemoryLayout<Float>.size
        let audioSize = 2 * trackCount * bufferSize * MemoryLayout<Float>.size
        
        return pressureSize + velocityXSize + velocityYSize + velocityZSize + audioSize
    }
    
    override func cleanup() {
        super.cleanup()

        hostInputBuffer?.deallocate()
        cpuGoldenBuffer?.deallocate()
        cpuPressureField?.deallocate()
        cpuVelocityXField?.deallocate()
        cpuVelocityYField?.deallocate()
        cpuVelocityZField?.deallocate()

        hostInputBuffer = nil
        cpuGoldenBuffer = nil
        cpuPressureField = nil
        cpuVelocityXField = nil
        cpuVelocityYField = nil
        cpuVelocityZField = nil
    }
}

private struct FDTD3DParams {
    let nx, ny, nz: UInt32
    let soundSpeed: Float
    let spatialStep: Float
    let timeStep: Float
    let absorptionCoeff: Float
    let sourceX, sourceY, sourceZ: UInt32
    let receiverX, receiverY, receiverZ: UInt32
    let bufferSize: UInt32
    let trackCount: UInt32
}
