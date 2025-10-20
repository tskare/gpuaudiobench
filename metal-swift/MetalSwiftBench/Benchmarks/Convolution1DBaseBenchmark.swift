import Foundation
import Metal

class Convolution1DBaseBenchmark: BaseBenchmark {
    let impulseResponseLength: Int

    var inputBuffer: MTLBuffer?
    var impulseResponseBuffer: MTLBuffer?
    var outputBuffer: MTLBuffer?

    var hostInputBuffer: UnsafeMutablePointer<Float>?
    var hostIRBuffer: UnsafeMutablePointer<Float>?
    var cpuGoldenBuffer: UnsafeMutablePointer<Float>?

    required convenience init(device: MTLDevice, bufferSize: Int, trackCount: Int) throws {
        try self.init(device: device, bufferSize: bufferSize, trackCount: trackCount, impulseResponseLength: 256)
    }

    init(device: MTLDevice, bufferSize: Int, trackCount: Int, impulseResponseLength: Int) throws {
        self.impulseResponseLength = impulseResponseLength
        try super.init(device: device, bufferSize: bufferSize, trackCount: trackCount)
    }

    func setupConvolutionBuffers(impulseResourceOptions: MTLResourceOptions = .storageModeShared) throws {
        let inputSampleCount = trackCount * bufferSize
        let irSampleCount = trackCount * impulseResponseLength

        inputBuffer = try allocateBuffer(length: inputSampleCount * MemoryLayout<Float>.size)
        impulseResponseBuffer = try allocateBuffer(length: irSampleCount * MemoryLayout<Float>.size,
                                                   options: impulseResourceOptions)
        outputBuffer = try allocateBuffer(length: inputSampleCount * MemoryLayout<Float>.size)

        hostInputBuffer = UnsafeMutablePointer<Float>.allocate(capacity: inputSampleCount)
        hostIRBuffer = UnsafeMutablePointer<Float>.allocate(capacity: irSampleCount)
        cpuGoldenBuffer = UnsafeMutablePointer<Float>.allocate(capacity: inputSampleCount)

        populateInputSamples()
        populateImpulseResponses()
        resetOutputBuffers()
        calculateCPUGoldenReference()
    }

    func populateInputSamples() {
        guard let inputBuffer = inputBuffer,
              let hostInputBuffer = hostInputBuffer else { return }

        let devicePointer = inputBuffer.contents().bindMemory(to: Float.self, capacity: trackCount * bufferSize)

        for trackIdx in 0..<trackCount {
            for sampleIdx in 0..<bufferSize {
                let idx = trackIdx * bufferSize + sampleIdx
                let value = Float.random(in: -1.0...1.0)
                devicePointer[idx] = value
                hostInputBuffer[idx] = value
            }
        }
    }

    func populateImpulseResponses() {
        guard let impulseResponseBuffer = impulseResponseBuffer,
              let hostIRBuffer = hostIRBuffer else { return }

        let devicePointer = impulseResponseBuffer.contents().bindMemory(to: Float.self,
                                                                        capacity: trackCount * impulseResponseLength)

        for trackIdx in 0..<trackCount {
            for irIdx in 0..<impulseResponseLength {
                let idx = trackIdx * impulseResponseLength + irIdx
                let value = impulseResponseValue(for: trackIdx, irIndex: irIdx)
                devicePointer[idx] = value
                hostIRBuffer[idx] = value
            }
        }
    }

    /// Generate windowed sinc impulse response
    func impulseResponseValue(for trackIdx: Int, irIndex: Int) -> Float {
        let freq = 0.1 + 0.05 * Float(trackIdx) / Float(max(trackCount, 1))
        let t = Float(irIndex) - Float(impulseResponseLength) / 2.0
        let window = 0.54 - 0.46 * cosf(2.0 * Float.pi * Float(irIndex) / Float(max(impulseResponseLength - 1, 1)))
        let sincNumerator = sinf(2.0 * Float.pi * freq * t)
        let sinc = t == 0 ? 1.0 : sincNumerator / (2.0 * Float.pi * freq * t)
        return window * sinc / Float(impulseResponseLength)
    }

    func resetOutputBuffers() {
        guard let outputBuffer = outputBuffer,
              let cpuGoldenBuffer = cpuGoldenBuffer else { return }

        memset(outputBuffer.contents(), 0, outputBuffer.length)
        memset(cpuGoldenBuffer, 0, trackCount * bufferSize * MemoryLayout<Float>.size)
    }

    func calculateCPUGoldenReference() {
        guard let hostInputBuffer = hostInputBuffer,
              let hostIRBuffer = hostIRBuffer,
              let cpuGoldenBuffer = cpuGoldenBuffer else { return }

        for trackIdx in 0..<trackCount {
            for sampleIdx in 0..<bufferSize {
                var outputSample: Float = 0.0

                for irIdx in 0..<impulseResponseLength {
                    let inputIdx = sampleIdx - irIdx
                    if inputIdx >= 0 && inputIdx < bufferSize {
                        let inputValue = hostInputBuffer[trackIdx * bufferSize + inputIdx]
                        let irValue = hostIRBuffer[trackIdx * impulseResponseLength + irIdx]
                        outputSample += inputValue * irValue
                    }
                }

                cpuGoldenBuffer[trackCount * sampleIdx + trackIdx] = outputSample
            }
        }
    }

    override func cleanup() {
        super.cleanup()

        inputBuffer = nil
        impulseResponseBuffer = nil
        outputBuffer = nil

        if let hostInputBuffer = hostInputBuffer {
            hostInputBuffer.deallocate()
            self.hostInputBuffer = nil
        }

        if let hostIRBuffer = hostIRBuffer {
            hostIRBuffer.deallocate()
            self.hostIRBuffer = nil
        }

        if let cpuGoldenBuffer = cpuGoldenBuffer {
            cpuGoldenBuffer.deallocate()
            self.cpuGoldenBuffer = nil
        }
    }

    deinit {
        cleanup()
    }
}
