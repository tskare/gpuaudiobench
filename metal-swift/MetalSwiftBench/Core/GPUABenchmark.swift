import Foundation
import Metal
import QuartzCore

enum BenchmarkError: LocalizedError {
    case deviceNotFound
    case failedToCreateCommandQueue
    case failedToLoadKernel(String)
    case bufferCreationFailed(size: Int)
    case pipelineCreationFailed
    case invalidConfiguration(String)
    
    var errorDescription: String? {
        switch self {
        case .deviceNotFound:
            return "No Metal device found"
        case .failedToCreateCommandQueue:
            return "Failed to create Metal command queue"
        case .failedToLoadKernel(let name):
            return "Failed to load kernel: \(name)"
        case .bufferCreationFailed(let size):
            return "Failed to create buffer of size \(size)"
        case .pipelineCreationFailed:
            return "Failed to create compute pipeline"
        case .invalidConfiguration(let message):
            return "Invalid configuration: \(message)"
        }
    }
}

struct BenchmarkResult {
    /// CPU wall-clock latencies per iteration (seconds)
    let latencies: [TimeInterval]

    /// GPU execution latencies per iteration (seconds, 0 when unavailable)
    let gpuLatencies: [TimeInterval]

    /// Benchmark-specific metadata (verification results, parameters, etc.)
    let metadata: [String: Any]

    init(latencies: [TimeInterval], gpuLatencies: [TimeInterval] = [], metadata: [String: Any]) {
        self.latencies = latencies
        self.gpuLatencies = gpuLatencies
        self.metadata = metadata
    }

    var median: TimeInterval {
        let sorted = latencies.sorted()
        let count = sorted.count
        if count == 0 { return 0 }
        if count % 2 == 0 {
            return (sorted[count/2 - 1] + sorted[count/2]) / 2
        } else {
            return sorted[count/2]
        }
    }

    var p95: TimeInterval {
        let sorted = latencies.sorted()
        let index = Int(Double(sorted.count) * 0.95)
        return sorted.indices.contains(index) ? sorted[index] : sorted.last ?? 0
    }

    var max: TimeInterval {
        return latencies.max() ?? 0
    }
}

/// Protocol for GPU audio processing benchmarks with timing and validation
protocol GPUABenchmark {
    var device: MTLDevice { get }
    var commandQueue: MTLCommandQueue { get }
    var kernelName: String { get }

    /// - Parameters:
    ///   - bufferSize: Audio samples per buffer per track
    ///   - trackCount: Number of parallel audio tracks
    init(device: MTLDevice, bufferSize: Int, trackCount: Int) throws

    /// Setup benchmark resources (buffers, pipelines, test data)
    func setup() throws

    /// Execute benchmark with timing measurements
    func runBenchmark(iterations: Int, warmupIterations: Int) throws -> BenchmarkResult

    /// Clean up allocated resources
    func cleanup()

    var validationMode: ValidationMode { get set }
    var dawSimulator: DAWSimulator? { get set }
}

extension GPUABenchmark {
    func cleanup() {}
}

enum ValidationMode: String {
    case none
    case spot
    case full
}

class BaseBenchmark: GPUABenchmark {
    static let spotSampleCount = 1024

    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    var kernelName: String
    let bufferSize: Int
    let trackCount: Int

    var pipelineState: MTLComputePipelineState?
    var threadsPerThreadgroup: MTLSize?
    var threadgroupsPerGrid: MTLSize?

    private(set) var bufferManager: BufferManager
    private(set) var timer: GPUTimer
    private(set) var profiler: PerformanceProfiler
    private(set) var commandEncoder: CommandEncoder
    private var currentIterationGPUDuration: TimeInterval = 0
    var validationMode: ValidationMode = .full
    var dawSimulator: DAWSimulator?

    required init(device: MTLDevice, bufferSize: Int, trackCount: Int) throws {
        self.device = device
        self.bufferSize = bufferSize
        self.trackCount = trackCount
        self.kernelName = "NoOp" // Subclasses should override

        guard let queue = device.makeCommandQueue() else {
            throw BenchmarkError.failedToCreateCommandQueue
        }
        self.commandQueue = queue

        self.bufferManager = BufferManager(device: device)
        self.timer = GPUTimer(mode: .cpuTime)
        self.profiler = PerformanceProfiler()
        self.commandEncoder = CommandEncoder(commandQueue: queue)
        self.commandEncoder.metricsSink = { [weak self] metrics in
            self?.recordCommandMetrics(metrics)
        }
    }

    func setup() throws {
        try profiler.measure("setup") {
            guard let library = device.makeDefaultLibrary() else {
                throw BenchmarkError.failedToLoadKernel("Default library not found")
            }

            guard let kernelFunction = library.makeFunction(name: kernelName) else {
                throw BenchmarkError.failedToLoadKernel(kernelName)
            }

            do {
                pipelineState = try device.makeComputePipelineState(function: kernelFunction)
            } catch {
                throw BenchmarkError.pipelineCreationFailed
            }

            calculateThreadConfiguration()
        }
    }

    func runBenchmark(iterations: Int, warmupIterations: Int = 3) throws -> BenchmarkResult {
        let (latencies, gpuLatencies) = try runIterations(
            iterations: iterations,
            warmupIterations: warmupIterations,
            body: { try self.performBenchmarkIteration() }
        )

        let totalSamples = bufferSize * trackCount
        let bytesProcessed = totalSamples * MemoryLayout<Float>.size
        let stats = Statistics(values: latencies.map { $0 * 1000 })
        let meanLatencySec = stats.mean / 1000.0
        let throughputGBps = Double(bytesProcessed) / (1024 * 1024 * 1024) / meanLatencySec
        let samplesPerSec = Double(totalSamples) / meanLatencySec

        var metadata: [String: Any] = [
            "bufferSize": bufferSize,
            "trackCount": trackCount,
            "kernelName": kernelName,
            "deviceName": device.name,
            "totalMemoryAllocated": bufferManager.totalMemoryAllocated,
            "warmupIterations": warmupIterations,
            "validationMode": validationMode.rawValue,
            "performance": [
                "throughputGBps": throughputGBps,
                "samplesPerSec": samplesPerSec,
                "bytesProcessed": bytesProcessed,
                "meanLatencyMs": stats.mean
            ]
        ]

        if let dawSimulator = dawSimulator {
            metadata["dawSimulation"] = [
                "mode": dawSimulator.mode.rawValue,
                "jitterUs": dawSimulator.jitterSeconds * 1_000_000,
                "bufferDurationMs": dawSimulator.bufferDuration * 1000
            ]
        }

        if validationMode == .spot {
            metadata["validationSampleTarget"] = BaseBenchmark.spotSampleCount
        }

        if gpuLatencies.contains(where: { $0 > 0 }) {
            let gpuStats = Statistics(values: gpuLatencies.map { $0 * 1000 })
            metadata["gpu_median_ms"] = gpuStats.median
            metadata["gpu_p95_ms"] = gpuStats.p95
            metadata["gpu_mean_ms"] = gpuStats.mean
            metadata["gpu_stddev_ms"] = gpuStats.standardDeviation
            metadata["gpu_latencies_ms"] = gpuLatencies.map { $0 * 1000 }
            metadata["gpu_samples"] = gpuLatencies.count
        }

        let hasGPUData = gpuLatencies.contains(where: { $0 > 0 })
        let result = BenchmarkResult(
            latencies: latencies,
            gpuLatencies: hasGPUData ? gpuLatencies : [],
            metadata: metadata
        )
        return result
    }

    func cleanup() {
        bufferManager.cleanup()
        profiler.reset()
    }

    // Helper Methods

    func allocateSharedBuffer<T>(count: Int, type: T.Type = Float.self) throws -> MTLBuffer {
        return try bufferManager.allocateShared(count: count, type: type)
    }

    func allocateManagedBuffer<T>(count: Int, type: T.Type = Float.self) throws -> MTLBuffer {
        return try bufferManager.allocateManaged(count: count, type: type)
    }

    func generateTestData(count: Int,
                         pattern: AudioDataGenerator.Pattern = .whiteNoise,
                         seed: UInt32 = 42) -> [Float] {
        return AudioDataGenerator.generateData(count: count, pattern: pattern, seed: seed)
    }

    func validateResults<T: FloatingPoint>(expected: [T],
                                         actual: [T],
                                         tolerance: T = 1e-5) -> ValidationResult<T> {
        return ValidationHelper.compare(expected: expected, actual: actual, tolerance: tolerance)
    }

    func printPerformanceProfile() {
        profiler.printSummary()
    }

    func measure<T>(_ label: String, operation: () throws -> T) rethrows -> T {
        return try profiler.measure(label, block: operation)
    }

    // Common Pattern Helpers

    /// Standard I/O buffer set for benchmarks
    struct StandardBuffers {
        let gpuInput: MTLBuffer
        let gpuOutput: MTLBuffer
        let hostInput: UnsafeMutablePointer<Float>
        let hostGolden: UnsafeMutablePointer<Float>
        let sampleCount: Int

        func cleanup() {
            hostInput.deallocate()
            hostGolden.deallocate()
        }
    }

    /// Allocate standard input/output buffers (GPU + host)
    func allocateStandardIOBuffers(sampleCount: Int) throws -> StandardBuffers {
        let bufferSizeBytes = sampleCount * MemoryLayout<Float>.size

        guard let gpuInput = device.makeBuffer(length: bufferSizeBytes, options: .storageModeShared) else {
            throw BenchmarkError.bufferCreationFailed(size: bufferSizeBytes)
        }

        guard let gpuOutput = device.makeBuffer(length: bufferSizeBytes, options: .storageModeShared) else {
            throw BenchmarkError.bufferCreationFailed(size: bufferSizeBytes)
        }

        let hostInput = UnsafeMutablePointer<Float>.allocate(capacity: sampleCount)
        let hostGolden = UnsafeMutablePointer<Float>.allocate(capacity: sampleCount)

        memset(hostGolden, 0, bufferSizeBytes)

        return StandardBuffers(
            gpuInput: gpuInput,
            gpuOutput: gpuOutput,
            hostInput: hostInput,
            hostGolden: hostGolden,
            sampleCount: sampleCount
        )
    }

    /// Populate input buffers with test data
    func populateInputBuffers(gpu: MTLBuffer, host: UnsafeMutablePointer<Float>,
                             count: Int, pattern: AudioDataGenerator.Pattern = .whiteNoise, seed: UInt32 = 42) {
        let gpuPointer = gpu.contents().bindMemory(to: Float.self, capacity: count)
        var generator = RandomNumberGenerator(seed: seed)

        for i in 0..<count {
            let value: Float
            switch pattern {
            case .whiteNoise:
                value = generator.nextFloat(in: -1...1)
            case .ones:
                value = 1.0
            case .silence:
                value = 0.0
            case .ramp:
                value = Float(i) / Float(max(count - 1, 1))
            case .sineWave(let freq):
                value = sin(2.0 * Float.pi * freq * Float(i) / 64.0)
            default:
                value = generator.nextFloat(in: -1...1)
            }
            gpuPointer[i] = value
            host[i] = value
        }
    }

    /// Verification result type
    typealias VerificationResult = (passed: Bool, maxError: Float, meanError: Float, sampleCount: Int)

    /// Standardized output verification against golden reference
    func verifyOutputBuffer(_ outputBuffer: MTLBuffer,
                          against goldenBuffer: UnsafeMutablePointer<Float>,
                          sampleCount: Int,
                          tolerance: Float = 1e-5,
                          metric: ((UnsafePointer<Float>, UnsafePointer<Float>, Int) -> Float)? = nil) -> VerificationResult {
        let outputPointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: sampleCount)
        let errorMetric = metric ?? { gpu, cpu, index in abs(gpu[index] - cpu[index]) }

        return compareWithGolden(
            gpuPointer: outputPointer,
            cpuPointer: goldenBuffer,
            sampleCount: sampleCount,
            tolerance: tolerance,
            metric: errorMetric
        )
    }

    private func runIterations(iterations: Int,
                               warmupIterations: Int,
                               body: () throws -> Void) throws -> ([TimeInterval], [TimeInterval]) {
        var latencies: [TimeInterval] = []
        latencies.reserveCapacity(iterations)
        var gpuLatencies: [TimeInterval] = []
        gpuLatencies.reserveCapacity(iterations)

        var dawState = DAWSimulationState()

        if warmupIterations > 0 {
            print("Running \(warmupIterations) warmup iterations...")
            for i in 0..<warmupIterations {
                beginIterationMetrics()
                do {
                    try body()
                    print("  Warmup \(i + 1)/\(warmupIterations) completed")
                } catch {
                    print("  Warmup iteration \(i + 1) failed: \(error)")
                }
                _ = finishIterationMetrics()
                if let dawSimulator = dawSimulator {
                    dawSimulator.wait(state: &dawState)
                }
            }
            print("Warmup complete, starting timed iterations...")
        }

        for iteration in 0..<iterations {
            beginIterationMetrics()
            do {
                let latency = try timer.measureKernel {
                    try body()
                }
                let gpuDuration = finishIterationMetrics()
                latencies.append(latency)
                gpuLatencies.append(gpuDuration)
                didCompleteIteration(iteration, latency: latency)
                if let dawSimulator = dawSimulator {
                    dawSimulator.wait(state: &dawState)
                }
            } catch {
                _ = finishIterationMetrics()
                print("Error during benchmark iteration \(iteration + 1): \(error)")
                throw error
            }
        }

        return (latencies, gpuLatencies)
    }

    /// Execute kernel with standard input/output buffer setup
    func executeKernel(inputBuffer: MTLBuffer,
                      outputBuffer: MTLBuffer,
                      additionalBuffers: [(MTLBuffer, Int)] = [],
                      label: String? = nil) throws {
        guard let pipelineState = pipelineState,
              let threadsPerThreadgroup = threadsPerThreadgroup,
              let threadgroupsPerGrid = threadgroupsPerGrid else {
            throw BenchmarkError.pipelineCreationFailed
        }

        try commandEncoder.executeSimple(
            pipelineState: pipelineState,
            inputBuffer: inputBuffer,
            outputBuffer: outputBuffer,
            threadsPerThreadgroup: threadsPerThreadgroup,
            threadgroupsPerGrid: threadgroupsPerGrid,
            additionalBuffers: additionalBuffers,
            label: label
        )
    }

    /// Execute kernel with parameter constants
    func executeKernelWithParams<T>(inputBuffer: MTLBuffer,
                                   outputBuffer: MTLBuffer,
                                   parameters: T,
                                   parameterIndex: Int = 2,
                                   additionalBuffers: [(MTLBuffer, Int)] = [],
                                   label: String? = nil) throws {
        guard let pipelineState = pipelineState,
              let threadsPerThreadgroup = threadsPerThreadgroup,
              let threadgroupsPerGrid = threadgroupsPerGrid else {
            throw BenchmarkError.pipelineCreationFailed
        }

        try commandEncoder.executeWithParams(
            pipelineState: pipelineState,
            inputBuffer: inputBuffer,
            outputBuffer: outputBuffer,
            parameters: parameters,
            parameterIndex: parameterIndex,
            threadsPerThreadgroup: threadsPerThreadgroup,
            threadgroupsPerGrid: threadgroupsPerGrid,
            additionalBuffers: additionalBuffers,
            label: label
        )
    }

    // Subclass Override Points

    /// Perform a single benchmark iteration (subclasses must override)
    func performBenchmarkIteration() throws {
        fatalError("Subclasses must override performBenchmarkIteration()")
    }

    /// Called after each iteration completes
    func didCompleteIteration(_ iteration: Int, latency: TimeInterval) {}

    // Private Helpers

    private func calculateThreadConfiguration() {
        guard let pipelineState = pipelineState else { return }

        let totalThreads = max(trackCount, 1)
        let maxThreadsPerThreadgroup = max(pipelineState.maxTotalThreadsPerThreadgroup, 1)
        let threadsPerThreadgroupValue = min(maxThreadsPerThreadgroup, totalThreads)

        threadsPerThreadgroup = MTLSize(width: threadsPerThreadgroupValue, height: 1, depth: 1)
        threadgroupsPerGrid = MTLSize(
            width: (totalThreads + threadsPerThreadgroupValue - 1) / threadsPerThreadgroupValue,
            height: 1,
            depth: 1
        )
    }

    // Legacy Compatibility

    @available(*, deprecated, message: "Use timer.measureKernel instead")
    func measureKernelLatency(executeBlock: () throws -> Void) throws -> TimeInterval {
        return try timer.measureKernel(executeBlock)
    }

    private func beginIterationMetrics() {
        currentIterationGPUDuration = 0
    }

    private func finishIterationMetrics() -> TimeInterval {
        let value = currentIterationGPUDuration
        currentIterationGPUDuration = 0
        return value
    }

    private func recordCommandMetrics(_ metrics: CommandMetrics) {
        if let gpuDuration = metrics.gpuDuration {
            currentIterationGPUDuration += gpuDuration
        }
    }

    func makeParams(gainValue: Float = 0) -> BenchmarkParams {
        return BenchmarkParams(
            bufferSize: UInt32(bufferSize),
            trackCount: UInt32(trackCount),
            totalSamples: UInt32(bufferSize * trackCount),
            gainValue: gainValue
        )
    }

    func shouldValidate() -> Bool {
        return validationMode != .none
    }

    func allocateBuffer(length: Int, options: MTLResourceOptions = .storageModeShared) throws -> MTLBuffer {
        do {
            return try bufferManager.allocateBytes(length: length, options: options)
        } catch {
            throw BenchmarkError.bufferCreationFailed(size: length)
        }
    }

    func allocateTypedBuffer<T>(count: Int,
                                type: T.Type = T.self,
                                options: MTLResourceOptions = .storageModeShared) throws -> MTLBuffer {
        let length = count * MemoryLayout<T>.size
        return try allocateBuffer(length: length, options: options)
    }

    func validationSampleLimit(totalSamples: Int) -> Int? {
        switch validationMode {
        case .none:
            return 0
        case .full:
            return nil
        case .spot:
            return min(totalSamples, BaseBenchmark.spotSampleCount)
        }
    }

    @discardableResult
    func iterateValidationSamples(totalSamples: Int, body: (Int) -> Void) -> Int {
        guard totalSamples > 0 else { return 0 }

        switch validationMode {
        case .none:
            return 0
        case .full:
            for idx in 0..<totalSamples {
                body(idx)
            }
            return totalSamples
        case .spot:
            let target = min(totalSamples, BaseBenchmark.spotSampleCount)
            let stride = max(1, totalSamples / target)
            var processed = 0
            var index = 0
            var lastIndex = -1
            while index < totalSamples && processed < target {
                body(index)
                lastIndex = index
                processed += 1
                index += stride
            }
            if processed < target && totalSamples - 1 != lastIndex {
                body(totalSamples - 1)
                processed += 1
            }
            return processed
        }
    }

    /// Helper to compare GPU output against a CPU reference buffer.
    /// - Parameters:
    ///   - gpuPointer: Pointer to GPU results (already bound to the desired type).
    ///   - cpuPointer: Pointer to CPU reference data.
    ///   - sampleCount: Number of elements to compare.
    ///   - tolerance: Maximum allowable error for validation to pass.
    ///   - metric: Optional error metric (defaults to absolute difference).
    func compareWithGolden(
        gpuPointer: UnsafePointer<Float>,
        cpuPointer: UnsafePointer<Float>,
        sampleCount: Int,
        tolerance: Float,
        metric: (UnsafePointer<Float>, UnsafePointer<Float>, Int) -> Float = { gpu, cpu, index in
            abs(gpu[index] - cpu[index])
        }
    ) -> (passed: Bool, maxError: Float, meanError: Float, sampleCount: Int) {
        var maxError: Float = 0
        var totalError: Float = 0

        let processed = iterateValidationSamples(totalSamples: sampleCount) { index in
            let error = metric(gpuPointer, cpuPointer, index)
            maxError = max(maxError, error)
            totalError += error
        }

        guard processed > 0 else {
            return (true, 0, 0, 0)
        }

        let meanError = totalError / Float(processed)
        return (maxError < tolerance, maxError, meanError, processed)
    }
}
