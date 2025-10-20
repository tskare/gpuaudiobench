//  BenchmarkUtilities.swift

import Foundation
import Metal
import QuartzCore

// Error Types

enum BenchmarkUtilityError: LocalizedError {
    case bufferAllocationFailed(size: Int, reason: String)
    case deviceNotSupported(feature: String)
    case invalidConfiguration(message: String)
    case validationFailed(details: String)

    var errorDescription: String? {
        switch self {
        case .bufferAllocationFailed(let size, let reason):
            return "Failed to allocate buffer of size \(size): \(reason)"
        case .deviceNotSupported(let feature):
            return "Device does not support feature: \(feature)"
        case .invalidConfiguration(let message):
            return "Invalid configuration: \(message)"
        case .validationFailed(let details):
            return "Validation failed: \(details)"
        }
    }
}

// Buffer Management

/// Automatic buffer manager with RAII cleanup
class BufferManager {
    private var buffers: [MTLBuffer] = []
    private let device: MTLDevice

    init(device: MTLDevice) {
        self.device = device
    }

    func allocateShared<T>(count: Int,
                          type: T.Type = Float.self,
                          options: MTLResourceOptions = .storageModeShared) throws -> MTLBuffer {
        let size = count * MemoryLayout<T>.size
        return try allocateBytes(length: size, options: options)
    }

    func allocateManaged<T>(count: Int,
                           type: T.Type = Float.self) throws -> MTLBuffer {
        return try allocateShared(count: count, type: type, options: .storageModeManaged)
    }

    func allocateBytes(length: Int,
                       options: MTLResourceOptions = .storageModeShared) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(length: length, options: options) else {
            throw BenchmarkUtilityError.bufferAllocationFailed(
                size: length,
                reason: "Device makeBuffer returned nil"
            )
        }
        buffers.append(buffer)
        return buffer
    }

    func register(buffer: MTLBuffer) {
        buffers.append(buffer)
    }

    var totalMemoryAllocated: Int {
        return buffers.reduce(0) { $0 + $1.length }
    }

    func cleanup() {
        buffers.removeAll()
    }

    deinit {
        cleanup()
    }
}

// Timing Utilities

struct GPUTimer {
    enum TimingMode {
        case cpuTime        // CACurrentMediaTime (default)
        case hostTime       // mach_absolute_time
        case gpuTimestamps  // GPU command buffer timestamps (when available)
    }

    private let mode: TimingMode

    init(mode: TimingMode = .cpuTime) {
        self.mode = mode
    }

    func measureKernel(_ block: () throws -> Void) rethrows -> TimeInterval {
        switch mode {
        case .cpuTime:
            return try measureCPUTime(block)
        case .hostTime:
            return try measureHostTime(block)
        case .gpuTimestamps:
            // Fallback to CPU time for now - GPU timestamps require more complex setup
            return try measureCPUTime(block)
        }
    }

    func measureGPUExecution(commandBuffer: MTLCommandBuffer,
                            executeBlock: (MTLCommandBuffer) -> Void) -> TimeInterval {
        let startTime = CACurrentMediaTime()

        executeBlock(commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let endTime = CACurrentMediaTime()
        return endTime - startTime
    }

    // Private timing implementations

    private func measureCPUTime(_ block: () throws -> Void) rethrows -> TimeInterval {
        let startTime = CACurrentMediaTime()
        try block()
        let endTime = CACurrentMediaTime()
        return endTime - startTime
    }

    private func measureHostTime(_ block: () throws -> Void) rethrows -> TimeInterval {
        let startTime = mach_absolute_time()
        try block()
        let endTime = mach_absolute_time()

        // Convert to TimeInterval
        var timebase = mach_timebase_info_data_t()
        mach_timebase_info(&timebase)
        let nanos = (endTime - startTime) * UInt64(timebase.numer) / UInt64(timebase.denom)
        return Double(nanos) / 1_000_000_000.0
    }
}

// Data Generation Utilities

struct AudioDataGenerator {
    enum Pattern {
        case silence
        case ones
        case ramp
        case sineWave(frequency: Float)
        case whiteNoise
        case pinkNoise
        case impulse(position: Int)
    }

    static func generateData(count: Int,
                           pattern: Pattern = .whiteNoise,
                           range: ClosedRange<Float> = -1...1,
                           seed: UInt32 = 42) -> [Float] {
        var data = [Float](repeating: 0, count: count)

        switch pattern {
        case .silence:
            break

        case .ones:
            data = [Float](repeating: 1.0, count: count)

        case .ramp:
            for i in 0..<count {
                data[i] = Float(i) / Float(count - 1)
            }

        case .sineWave(let frequency):
            for i in 0..<count {
                data[i] = sin(2.0 * Float.pi * frequency * Float(i) / 64.0)
            }

        case .whiteNoise:
            var generator = RandomNumberGenerator(seed: seed)
            for i in 0..<count {
                data[i] = generator.nextFloat(in: range)
            }

        case .pinkNoise:
            // Pink noise approximation using pole/zero filters
            var generator = RandomNumberGenerator(seed: seed)
            var b0: Float = 0, b1: Float = 0, b2: Float = 0, b3: Float = 0, b4: Float = 0, b5: Float = 0, b6: Float = 0

            for i in 0..<count {
                let white = generator.nextFloat(in: -1...1)
                b0 = 0.99886 * b0 + white * 0.0555179
                b1 = 0.99332 * b1 + white * 0.0750759
                b2 = 0.96900 * b2 + white * 0.1538520
                b3 = 0.86650 * b3 + white * 0.3104856
                b4 = 0.55000 * b4 + white * 0.5329522
                b5 = -0.7616 * b5 - white * 0.0168980
                let pink = b0 + b1 + b2 + b3 + b4 + b5 + b6 + white * 0.5362
                b6 = white * 0.115926

                data[i] = pink * 0.11 // Scale to roughly [-1, 1]
            }

        case .impulse(let position):
            if position >= 0 && position < count {
                data[position] = 1.0
            }
        }

        if range != -1...1 {
            let currentMin = data.min() ?? 0
            let currentMax = data.max() ?? 0
            let currentRange = currentMax - currentMin

            if currentRange > 0 {
                let targetRange = range.upperBound - range.lowerBound
                for i in 0..<count {
                    data[i] = range.lowerBound + (data[i] - currentMin) / currentRange * targetRange
                }
            }
        }

        return data
    }

    /// Generate windowed sinc impulse response
    static func generateImpulseResponse(length: Int,
                                      cutoffFreq: Float,
                                      windowType: WindowType = .hamming) -> [Float] {
        var ir = [Float](repeating: 0, count: length)
        let center = Float(length) / 2.0

        // Generate windowed sinc
        for i in 0..<length {
            let t = Float(i) - center

            let sinc: Float
            if abs(t) < 1e-6 {
                sinc = 1.0
            } else {
                let arg = 2.0 * Float.pi * cutoffFreq * t
                sinc = sin(arg) / arg
            }

            let window = windowType.value(at: i, length: length)
            ir[i] = sinc * window
        }

        let sum = ir.reduce(0, +)
        if abs(sum) > 1e-6 {
            for i in 0..<length {
                ir[i] /= sum
            }
        }

        return ir
    }
}

// Window Functions

enum WindowType {
    case rectangular
    case hamming
    case hann
    case blackman
    case kaiser(beta: Float)

    func value(at index: Int, length: Int) -> Float {
        let n = Float(index) / Float(length - 1)

        switch self {
        case .rectangular:
            return 1.0

        case .hamming:
            return 0.54 - 0.46 * cos(2.0 * Float.pi * n)

        case .hann:
            return 0.5 * (1.0 - cos(2.0 * Float.pi * n))

        case .blackman:
            return 0.42 - 0.5 * cos(2.0 * Float.pi * n) + 0.08 * cos(4.0 * Float.pi * n)

        case .kaiser(let beta):
            // Simplified Kaiser window (approximate)
            let arg = 2.0 * n - 1.0
            return sinh(beta * sqrt(1.0 - arg * arg)) / sinh(beta)
        }
    }
}

// Random Number Generator

struct RandomNumberGenerator {
    private var state: UInt32

    init(seed: UInt32 = 42) {
        self.state = seed
    }

    mutating func next() -> UInt32 {
        state = state &* 1664525 &+ 1013904223  // LCG
        return state
    }

    mutating func nextFloat(in range: ClosedRange<Float> = 0...1) -> Float {
        let value = Float(next()) / Float(UInt32.max)
        return range.lowerBound + value * (range.upperBound - range.lowerBound)
    }
}

// Validation Helpers

struct ValidationHelper {
    static func compare<T: FloatingPoint>(expected: [T],
                                        actual: [T],
                                        tolerance: T = 1e-5) -> ValidationResult<T> {
        guard expected.count == actual.count else {
            return ValidationResult(
                success: false,
                maxError: T.infinity,
                meanError: T.infinity,
                errorCount: expected.count,
                message: "Array size mismatch: expected \(expected.count), got \(actual.count)"
            )
        }

        var sumError: T = 0
        var maxError: T = 0
        var errorCount = 0
        var firstErrors: [(Int, T, T, T)] = [] // (index, expected, actual, error)

        for i in 0..<expected.count {
            let error = abs(actual[i] - expected[i])
            sumError += error
            maxError = max(maxError, error)

            if error > tolerance {
                errorCount += 1
                if firstErrors.count < 10 {
                    firstErrors.append((i, expected[i], actual[i], error))
                }
            }
        }

        let meanError = sumError / T(expected.count)
        let success = errorCount == 0

        var message = success ? "Validation passed" : "Validation failed"
        if !success {
            message += ": \(errorCount) of \(expected.count) elements exceeded tolerance"
            if !firstErrors.isEmpty {
                message += "\nFirst errors:"
                for (index, exp, act, err) in firstErrors.prefix(5) {
                    message += "\n  [\(index)]: expected \(exp), got \(act), error \(err)"
                }
            }
        }

        return ValidationResult(
            success: success,
            maxError: maxError,
            meanError: meanError,
            errorCount: errorCount,
            message: message
        )
    }
}

struct ValidationResult<T: FloatingPoint> {
    let success: Bool
    let maxError: T
    let meanError: T
    let errorCount: Int
    let message: String
}

// Command Encoder Wrapper

struct CommandMetrics {
    let label: String?
    let cpuDuration: TimeInterval
    let gpuDuration: TimeInterval?
}

/// Reusable command encoder wrapper that eliminates Metal boilerplate
class CommandEncoder {
    /// Packs constant data for setBytes bindings, avoiding manual withUnsafeBytes calls.
    /// For small POD structs only; use dedicated buffers for larger payloads.
    struct ParameterBinding {
        let index: Int
        private let storage: [UInt8]

        init<T>(_ value: T, index: Int) {
            self.index = index
            var mutableValue = value
            self.storage = withUnsafeBytes(of: &mutableValue) { rawBuffer in
                Array(rawBuffer)
            }
        }

        func encode(into encoder: MTLComputeCommandEncoder) {
            storage.withUnsafeBytes { rawBuffer in
                guard let baseAddress = rawBuffer.baseAddress else { return }
                encoder.setBytes(baseAddress, length: rawBuffer.count, index: index)
            }
        }
    }

    private let commandQueue: MTLCommandQueue
    var metricsSink: ((CommandMetrics) -> Void)?

    init(commandQueue: MTLCommandQueue) {
        self.commandQueue = commandQueue
    }

    func execute(pipelineState: MTLComputePipelineState,
                buffers: [(MTLBuffer, Int)] = [],
                textures: [(MTLTexture, Int)] = [],
                bytes: [(UnsafeRawPointer, Int, Int)] = [],
                parameterBindings: [ParameterBinding] = [],
                threadsPerThreadgroup: MTLSize,
                threadgroupsPerGrid: MTLSize,
                label: String? = nil) throws {

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw BenchmarkUtilityError.deviceNotSupported(feature: "Command buffer creation")
        }

        if let label = label {
            commandBuffer.label = label
        }

        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw BenchmarkUtilityError.deviceNotSupported(feature: "Compute encoder creation")
        }

        let cpuStart = CACurrentMediaTime()

        computeEncoder.setComputePipelineState(pipelineState)

        var seenBufferIndices = Set<Int>()
        for (buffer, index) in buffers {
            guard seenBufferIndices.insert(index).inserted else {
                throw BenchmarkUtilityError.invalidConfiguration(message: "Duplicate buffer binding at index \(index)")
            }
            computeEncoder.setBuffer(buffer, offset: 0, index: index)
        }

        for (texture, index) in textures {
            computeEncoder.setTexture(texture, index: index)
        }

        var seenByteIndices = Set<Int>()
        for (bytes, length, index) in bytes {
            guard seenByteIndices.insert(index).inserted else {
                throw BenchmarkUtilityError.invalidConfiguration(message: "Duplicate byte binding at index \(index)")
            }
            computeEncoder.setBytes(bytes, length: length, index: index)
        }

        for binding in parameterBindings {
            guard seenByteIndices.insert(binding.index).inserted else {
                throw BenchmarkUtilityError.invalidConfiguration(message: "Duplicate byte binding at index \(binding.index)")
            }
            binding.encode(into: computeEncoder)
        }

        computeEncoder.dispatchThreadgroups(threadgroupsPerGrid,
                                          threadsPerThreadgroup: threadsPerThreadgroup)

        computeEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let cpuDuration = CACurrentMediaTime() - cpuStart
        let gpuDuration: TimeInterval?
        if commandBuffer.gpuStartTime > 0 && commandBuffer.gpuEndTime > commandBuffer.gpuStartTime {
            gpuDuration = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
        } else {
            gpuDuration = nil
        }

        if commandBuffer.status == .error {
            if let error = commandBuffer.error {
                throw BenchmarkUtilityError.validationFailed(details: "Command buffer failed: \(error.localizedDescription)")
            } else {
                throw BenchmarkUtilityError.validationFailed(details: "Command buffer failed with unknown error")
            }
        }

        metricsSink?(CommandMetrics(label: label, cpuDuration: cpuDuration, gpuDuration: gpuDuration))
    }

    func executeSimple(pipelineState: MTLComputePipelineState,
                      inputBuffer: MTLBuffer,
                      outputBuffer: MTLBuffer,
                      threadsPerThreadgroup: MTLSize,
                      threadgroupsPerGrid: MTLSize,
                      additionalBuffers: [(MTLBuffer, Int)] = [],
                      label: String? = nil) throws {

        var allBuffers = [(inputBuffer, 0), (outputBuffer, 1)]
        allBuffers.append(contentsOf: additionalBuffers)

        try execute(pipelineState: pipelineState,
                   buffers: allBuffers,
                   threadsPerThreadgroup: threadsPerThreadgroup,
                   threadgroupsPerGrid: threadgroupsPerGrid,
                   label: label)
    }

    func executeWithParams<T>(pipelineState: MTLComputePipelineState,
                             inputBuffer: MTLBuffer,
                             outputBuffer: MTLBuffer,
                             parameters: T,
                             parameterIndex: Int = 2,
                             threadsPerThreadgroup: MTLSize,
                             threadgroupsPerGrid: MTLSize,
                             additionalBuffers: [(MTLBuffer, Int)] = [],
                             label: String? = nil) throws {

        var allBuffers = [(inputBuffer, 0), (outputBuffer, 1)]
        allBuffers.append(contentsOf: additionalBuffers)

        let binding = ParameterBinding(parameters, index: parameterIndex)

        try execute(pipelineState: pipelineState,
                   buffers: allBuffers,
                   parameterBindings: [binding],
                   threadsPerThreadgroup: threadsPerThreadgroup,
                   threadgroupsPerGrid: threadgroupsPerGrid,
                   label: label)
    }
}

// Performance Profiler

class PerformanceProfiler {
    private var measurements: [String: [TimeInterval]] = [:]

    func measure<T>(_ label: String, block: () throws -> T) rethrows -> T {
        let startTime = CACurrentMediaTime()
        let result = try block()
        let elapsed = CACurrentMediaTime() - startTime

        if measurements[label] == nil {
            measurements[label] = []
        }
        measurements[label]?.append(elapsed)

        return result
    }

    func printSummary() {
        print("\n=== Performance Profile ===")
        for (label, times) in measurements.sorted(by: { $0.key < $1.key }) {
            let stats = Statistics(values: times.map { $0 * 1000 }) // Convert to ms
            print("\(label):")
            print("  Count: \(stats.count), Mean: \(String(format: "%.3f", stats.mean))ms")
            print("  Min: \(String(format: "%.3f", stats.min))ms, Max: \(String(format: "%.3f", stats.max))ms")
        }
        print("==========================\n")
    }

    func reset() {
        measurements.removeAll()
    }
}
