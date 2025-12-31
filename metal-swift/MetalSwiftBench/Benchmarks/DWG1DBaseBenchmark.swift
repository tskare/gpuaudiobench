import Foundation
import Metal

struct WaveguideState {
    var length: UInt32
    var inputTapPos: UInt32
    var outputTapPos: UInt32
    var writePos: UInt32
    var gain: Float
    var reflection: Float
    var damping: Float
    var padding: Float
}

struct DWGParams {
    var numWaveguides: UInt32
    var bufferSize: UInt32
    var outputTracks: UInt32
    var minLength: UInt32
    var maxLength: UInt32
    var reflectionCoeff: Float
    var dampingCoeff: Float
    var padding: Float
}

class DWG1DBaseBenchmark: BaseBenchmark {
    var numWaveguides: Int
    var minLength: Int
    var maxLength: Int
    var reflectionCoeff: Float
    var dampingCoeff: Float

    var waveguideParamsBuffer: MTLBuffer?
    var delayLineForwardBuffer: MTLBuffer?
    var delayLineBackwardBuffer: MTLBuffer?
    var inputSignalBuffer: MTLBuffer?
    var outputBuffer: MTLBuffer?
    var dwgParamsBuffer: MTLBuffer?

    var hostWaveguideParams: UnsafeMutablePointer<WaveguideState>?
    var hostDelayLineForward: UnsafeMutablePointer<Float>?
    var hostDelayLineBackward: UnsafeMutablePointer<Float>?
    var hostInputSignal: UnsafeMutablePointer<Float>?
    var cpuGoldenBuffer: UnsafeMutablePointer<Float>?

    var waveguideLengths: [Int] = []

    required convenience init(device: MTLDevice, bufferSize: Int, trackCount: Int) throws {
        try self.init(device: device, bufferSize: bufferSize, trackCount: trackCount, numWaveguides: trackCount)
    }

    init(device: MTLDevice,
         bufferSize: Int,
         trackCount: Int,
         numWaveguides: Int,
         minLength: Int = 64,
         maxLength: Int = 1024,
         reflectionCoeff: Float = -0.99,
         dampingCoeff: Float = 0.9999) throws {
        self.numWaveguides = numWaveguides
        self.minLength = minLength
        self.maxLength = maxLength
        self.reflectionCoeff = reflectionCoeff
        self.dampingCoeff = dampingCoeff
        try super.init(device: device, bufferSize: bufferSize, trackCount: trackCount)
    }

    override func setup() throws {
        guard bufferSize <= 512 else {
            throw BenchmarkError.invalidConfiguration("DWG benchmarks require bufferSize <= 512 to match shared memory limits.")
        }
        try super.setup()
        let lengths = try generateWaveguideLengths()
        try prepareWaveguides(lengths: lengths)
    }

    func generateWaveguideLengths() throws -> [Int] {
        fatalError("Subclasses must implement generateWaveguideLengths()")
    }

    func makeWaveguideState(for length: Int, index: Int) -> WaveguideState {
        let inputTapPos = UInt32.random(in: 0..<UInt32(length))
        let outputTapPos = UInt32.random(in: 0..<UInt32(length))
        let gain = Float.random(in: 0.1...1.0)
        return WaveguideState(
            length: UInt32(length),
            inputTapPos: inputTapPos,
            outputTapPos: outputTapPos,
            writePos: 0,
            gain: gain,
            reflection: reflectionCoeff,
            damping: dampingCoeff,
            padding: 0
        )
    }

    func initializeInputSignal() {
        guard let inputSignalBuffer = inputSignalBuffer,
              let hostInputSignal = hostInputSignal else { return }

        let pointer = inputSignalBuffer.contents().bindMemory(to: Float.self, capacity: bufferSize)
        for i in 0..<bufferSize {
            var sample: Float = 0.0
            if i == 0 {
                sample = 1.0
            } else {
                sample = Float.random(in: -0.1...0.1)
            }
            pointer[i] = sample
            hostInputSignal[i] = sample
        }
    }

    func initializeDelayLines(maxWaveguideLength: Int) {
        let totalDelaySize = numWaveguides * maxWaveguideLength
        if let buffer = delayLineForwardBuffer {
            memset(buffer.contents(), 0, totalDelaySize * MemoryLayout<Float>.size)
        }
        if let buffer = delayLineBackwardBuffer {
            memset(buffer.contents(), 0, totalDelaySize * MemoryLayout<Float>.size)
        }
        if let hostForward = hostDelayLineForward {
            memset(hostForward, 0, totalDelaySize * MemoryLayout<Float>.size)
        }
        if let hostBackward = hostDelayLineBackward {
            memset(hostBackward, 0, totalDelaySize * MemoryLayout<Float>.size)
        }
        if let outputBuffer = outputBuffer {
            memset(outputBuffer.contents(), 0, bufferSize * MemoryLayout<Float>.size)
        }
        if let cpuGoldenBuffer = cpuGoldenBuffer {
            memset(cpuGoldenBuffer, 0, bufferSize * MemoryLayout<Float>.size)
        }
    }

    func prepareWaveguides(lengths: [Int]) throws {
        guard lengths.count == numWaveguides else {
            throw BenchmarkError.invalidConfiguration("Expected \(numWaveguides) lengths, got \(lengths.count)")
        }

        waveguideLengths = lengths
        minLength = lengths.min() ?? minLength
        maxLength = lengths.max() ?? maxLength
        let longest = maxLength

        waveguideParamsBuffer = try allocateBuffer(length: numWaveguides * MemoryLayout<WaveguideState>.size)
        delayLineForwardBuffer = try allocateBuffer(length: numWaveguides * longest * MemoryLayout<Float>.size)
        delayLineBackwardBuffer = try allocateBuffer(length: numWaveguides * longest * MemoryLayout<Float>.size)
        inputSignalBuffer = try allocateBuffer(length: bufferSize * MemoryLayout<Float>.size)
        outputBuffer = try allocateBuffer(length: bufferSize * MemoryLayout<Float>.size)
        dwgParamsBuffer = try allocateBuffer(length: MemoryLayout<DWGParams>.size)

        hostWaveguideParams = UnsafeMutablePointer<WaveguideState>.allocate(capacity: numWaveguides)
        hostDelayLineForward = UnsafeMutablePointer<Float>.allocate(capacity: numWaveguides * longest)
        hostDelayLineBackward = UnsafeMutablePointer<Float>.allocate(capacity: numWaveguides * longest)
        hostInputSignal = UnsafeMutablePointer<Float>.allocate(capacity: bufferSize)
        cpuGoldenBuffer = UnsafeMutablePointer<Float>.allocate(capacity: bufferSize)

        let gpuParamsPointer = waveguideParamsBuffer!.contents().bindMemory(to: WaveguideState.self, capacity: numWaveguides)
        for (index, length) in lengths.enumerated() {
            let state = makeWaveguideState(for: length, index: index)
            gpuParamsPointer[index] = state
            hostWaveguideParams![index] = state
        }

        let dwgParams = DWGParams(
            numWaveguides: UInt32(numWaveguides),
            bufferSize: UInt32(bufferSize),
            outputTracks: UInt32(trackCount),
            minLength: UInt32(minLength),
            maxLength: UInt32(maxLength),
            reflectionCoeff: reflectionCoeff,
            dampingCoeff: dampingCoeff,
            padding: 0
        )
        let paramsPointer = dwgParamsBuffer!.contents().bindMemory(to: DWGParams.self, capacity: 1)
        paramsPointer[0] = dwgParams

        initializeInputSignal()
        initializeDelayLines(maxWaveguideLength: longest)
        calculateCPUGoldenReference()
    }

    func calculateCPUGoldenReference() {
        guard let hostWaveguideParams = hostWaveguideParams,
              let hostDelayLineForward = hostDelayLineForward,
              let hostDelayLineBackward = hostDelayLineBackward,
              let hostInputSignal = hostInputSignal,
              let cpuGoldenBuffer = cpuGoldenBuffer else { return }

        memset(cpuGoldenBuffer, 0, bufferSize * MemoryLayout<Float>.size)

        for wgIdx in 0..<numWaveguides {
            let wg = hostWaveguideParams[wgIdx]
            let delayBase = wgIdx * maxLength

            for sample in 0..<bufferSize {
                let input = hostInputSignal[sample] * wg.gain

                let forwardPos = Int((wg.writePos + UInt32(sample)) % wg.length)
                let backwardPos = Int((wg.writePos + UInt32(sample) + wg.length / 2) % wg.length)

                var forwardSample = hostDelayLineForward[delayBase + forwardPos]
                var backwardSample = hostDelayLineBackward[delayBase + backwardPos]

                forwardSample *= wg.damping
                backwardSample *= wg.damping

                if UInt32(forwardPos) == wg.inputTapPos {
                    forwardSample += input
                    backwardSample += input
                }

                let newForward = backwardSample * wg.reflection + input
                let newBackward = forwardSample * wg.reflection + input

                hostDelayLineForward[delayBase + forwardPos] = newForward
                hostDelayLineBackward[delayBase + backwardPos] = newBackward

                if UInt32(forwardPos) == wg.outputTapPos {
                    let outputSample = (forwardSample + backwardSample) * 0.5
                    cpuGoldenBuffer[sample] += outputSample
                }
            }
        }
    }

    func verifyOutput(tolerance: Float) -> (passed: Bool, maxError: Float, meanError: Float, sampleCount: Int) {
        guard let outputBuffer = outputBuffer,
              let cpuGoldenBuffer = cpuGoldenBuffer else {
            return (false, Float.infinity, Float.infinity, 0)
        }

        let outputPointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: bufferSize)
        return compareWithGolden(
            gpuPointer: outputPointer,
            cpuPointer: cpuGoldenBuffer,
            sampleCount: bufferSize,
            tolerance: tolerance
        )
    }

    override func cleanup() {
        super.cleanup()
        waveguideParamsBuffer = nil
        delayLineForwardBuffer = nil
        delayLineBackwardBuffer = nil
        inputSignalBuffer = nil
        outputBuffer = nil
        dwgParamsBuffer = nil

        hostWaveguideParams?.deallocate()
        hostDelayLineForward?.deallocate()
        hostDelayLineBackward?.deallocate()
        hostInputSignal?.deallocate()
        cpuGoldenBuffer?.deallocate()

        hostWaveguideParams = nil
        hostDelayLineForward = nil
        hostDelayLineBackward = nil
        hostInputSignal = nil
        cpuGoldenBuffer = nil
        waveguideLengths.removeAll()
    }
}
